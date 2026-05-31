# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.action_head.action_encoder import SinusoidalPositionalEncoding, swish

from .cross_attention_dit import DiT, SelfAttentionTransformer


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


class TaskConditionedAdapter(nn.Module):
    """FiLM adapter: γ(task_emb) * x + β(task_emb).

    Applies feature-wise linear modulation conditioned on a task embedding,
    enabling task-specific scaling/shifting of encoder/decoder outputs without
    modifying the frozen embodiment-specific weight matrices.

    A hidden layer (task_emb_dim → task_emb_dim → feature_dim×2) increases
    the adapter's expressive capacity compared to a single linear projection.
    """

    def __init__(self, feature_dim: int, task_emb_dim: int):
        super().__init__()
        # Two-layer MLP: task_emb -> hidden -> (gamma, beta)
        hidden_dim = task_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(task_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim * 2),
        )
        # Init final layer to near-zero so adapter starts as identity transform
        nn.init.zeros_(self.mlp[-1].weight)
        # bias: first half (gamma) = 1 (multiplicative identity), second half (beta) = 0
        bias = torch.zeros(feature_dim * 2)
        bias[:feature_dim] = 1.0
        self.mlp[-1].bias = nn.Parameter(bias)

    def forward(self, x: torch.Tensor, task_emb: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D),  task_emb: (B, task_emb_dim)
        out = self.mlp(task_emb)                     # (B, 2D)
        feature_dim = x.shape[-1]
        gamma = out[:, :feature_dim].unsqueeze(1)    # (B, 1, D)
        beta  = out[:, feature_dim:].unsqueeze(1)    # (B, 1, D)
        return gamma * x + beta


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )
    backbone_embedding_dim: int = field(
        default=1536, metadata={"help": "Backbone embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    use_task_router: bool = field(default=False, metadata={"help": "Whether to use soft-weighted task router."})
    num_task_emb_slots: int = field(default=8, metadata={"help": "Number of learnable embedding slots in the task router bank."})
    router_hidden_dim: int = field(default=256, metadata={"help": "Hidden dim of the router MLP."})
    router_lang_tail: int = field(
        default=0,
        metadata={"help": "If > 0, pool only the last N valid backbone tokens for routing (focuses on language instruction tokens at end of VLM sequence). 0 = pool all valid tokens."},
    )
    router_diversity_coeff: float = field(
        default=0.01,
        metadata={"help": "Coefficient for the router load-balancing (diversity) loss. Encourages all K slots to be used. Set 0 to disable."},
    )
    use_task_adapter: bool = field(
        default=False,
        metadata={"help": "Add FiLM task-conditioned adapters after state_encoder, action_encoder, and action_decoder. Requires use_task_router=True."},
    )

    # Skill embedding config (gated by use_skill_emb; backward-compatible when False)
    use_skill_emb: bool = field(
        default=False,
        metadata={"help": "Enable skill embedding module: MLP classifier (Stage 1) + learnable skill token concat (Stage 2)."},
    )
    skill_vocab: list = field(
        default=None,
        metadata={"help": "Enable skill embedding module: MLP classifier (Stage 1) + learnable skill token concat (Stage 2)."},
    )
    num_skills: int = field(
        default=0,
        metadata={"help": "Number of skill classes (auto-set from discovered vocab)."},
    )
    skill_emb_dim: int = field(
        default=256,
        metadata={"help": "Dimension of learnable skill embedding vectors in skill_emb_bank."},
    )
    skill_proj_hidden_dim: int = field(
        default=256,
        metadata={"help": "Hidden dim of the MLP projector used in the skill classifier."},
    )
    skill_clf_coeff: float = field(
        default=1.0,
        metadata={"help": "Weight for cross-entropy skill classification loss."},
    )
    skill_div_coeff: float = field(
        default=0.01,
        metadata={"help": "Weight for orthogonality diversity loss on skill embedding bank."},
    )
    skill_norm_coeff: float = field(
        default=0.01,
        metadata={"help": "Weight for non-zero norm loss on skill embedding bank (prevents collapse to zero)."},
    )
    use_weighted_skill_router: bool = field(
        default=False,
        metadata={"help": "Use soft-weighted routing (softmax over all skill embeddings) instead of top-1 argmax."},
    )
    tune_skill_clf: bool = field(
        default=False,
        metadata={"help": "Stage 1: freeze all except skill classifier MLP."},
    )
    tune_skill_emb: bool = field(
        default=False,
        metadata={"help": "Stage 2: freeze skill classifier, train skill embeddings + diffusion."},
    )
    use_skill_film: bool = field(
        default=False,
        metadata={"help": "Condition action features via FiLM (action*(1+gamma)+beta) from the skill embedding instead of concatenating a separate skill token."},
    )
    use_skill_gate: bool = field(
        default=False,
        metadata={"help": "Learn a scalar gate in [0, 1] that scales skill FiLM; gate=0 makes skill FiLM an identity/no-skill path."},
    )
    skill_gate_dropout: float = field(
        default=0.0,
        metadata={"help": "During training, randomly zero the skill FiLM gate with this probability to preserve a no-skill fallback."},
    )
    skill_gate_init_bias: float = field(
        default=-2.0,
        metadata={"help": "Initial bias for the skill FiLM gate. Negative values start closer to no-skill behavior."},
    )
    skill_gate_l1_coeff: float = field(
        default=0.0,
        metadata={"help": "Coefficient for light L1 sparsity on the skill gate. Encourages the no-skill identity path when skill conditioning is not useful."},
    )
    use_skill_encoder_hidden_states: bool = field(
        default=False,
        metadata={"help": "Append the projected skill embedding to VLM encoder_hidden_states instead of DiT hidden_states."},
    )
    use_skill_dit_modulation: bool = field(
        default=False,
        metadata={"help": "Inject skill-conditioned FiLM into every DiT block and the DiT output AdaLN."},
    )
    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(
        default=32, metadata={"help": "Number of target vision tokens."}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class FlowmatchingActionHead(nn.Module):
    config_class = FlowmatchingActionHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: FlowmatchingActionHeadConfig,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        diffusion_model_cfg = dict(config.diffusion_model_cfg)
        if config.use_skill_emb and config.use_skill_dit_modulation:
            diffusion_model_cfg["skill_conditioning_dim"] = config.skill_emb_dim
        else:
            diffusion_model_cfg.pop("skill_conditioning_dim", None)
        config.diffusion_model_cfg = diffusion_model_cfg
        self.model = DiT(**diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )
        self.vl_self_attention = (
            SelfAttentionTransformer(**config.vl_self_attention_cfg)
            if config.use_vlln
            else nn.Identity()
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets

        if config.use_task_router:
            K = config.num_task_emb_slots
            H = config.router_hidden_dim
            backbone_emb_dim = config.backbone_embedding_dim
            self.task_emb_bank = nn.Embedding(K, backbone_emb_dim)
            self.router = nn.Sequential(
                nn.Linear(backbone_emb_dim, H),
                nn.ReLU(),
                nn.Linear(H, K),
            )

        if config.use_task_adapter:
            assert config.use_task_router, "use_task_adapter requires use_task_router=True"
            D = config.backbone_embedding_dim
            self.state_adapter   = TaskConditionedAdapter(config.input_embedding_dim, D)
            self.action_adapter  = TaskConditionedAdapter(config.input_embedding_dim, D)
            self.decoder_adapter = TaskConditionedAdapter(config.action_dim, D)

        if config.use_skill_emb:
            assert config.num_skills > 0, "num_skills must be > 0 when use_skill_emb=True"
            assert not (
                config.use_skill_film and config.use_skill_encoder_hidden_states
            ), "use_skill_encoder_hidden_states is mutually exclusive with use_skill_film"
            assert not config.use_skill_gate or config.use_skill_film, "use_skill_gate requires use_skill_film=True"
            backbone_emb_dim = config.backbone_embedding_dim
            # Stage 1: MLP projector + linear classifier on pooled VLM features
            self.skill_proj = nn.Sequential(
                nn.Linear(backbone_emb_dim, config.skill_proj_hidden_dim),
                nn.ReLU(),
            )
            self.skill_clf = nn.Linear(config.skill_proj_hidden_dim, config.num_skills)
            # Stage 2: learnable per-skill embedding bank + projection to DiT input dim
            self.skill_emb_bank = nn.Embedding(config.num_skills, config.skill_emb_dim)
            self.skill_emb_proj = nn.Linear(config.skill_emb_dim, config.input_embedding_dim)
            if config.use_skill_encoder_hidden_states:
                self.skill_encoder_proj = nn.Linear(config.skill_emb_dim, backbone_emb_dim)
            # Optional FiLM head: modulates action features with the skill embedding
            # (action_features * (1 + gamma) + beta) instead of concatenating a token.
            if config.use_skill_film:
                self.skill_film = nn.Sequential(
                    nn.Linear(config.skill_emb_dim, config.skill_proj_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(config.skill_proj_hidden_dim, config.input_embedding_dim * 2),
                )
                # Zero-init final layer -> gamma=0, beta=0 -> starts as identity
                # since the modulation is action_features * (1 + gamma) + beta.
                nn.init.zeros_(self.skill_film[-1].weight)
                nn.init.zeros_(self.skill_film[-1].bias)
                if config.use_skill_gate:
                    self.skill_gate = nn.Sequential(
                        nn.Linear(config.skill_emb_dim, config.skill_proj_hidden_dim),
                        nn.SiLU(),
                        nn.Linear(config.skill_proj_hidden_dim, 1),
                    )
                    nn.init.zeros_(self.skill_gate[-1].weight)
                    nn.init.constant_(self.skill_gate[-1].bias, config.skill_gate_init_bias)
            # Routed skill from the most recent get_action() call (for logging /
            # video overlay during evaluation). Populated in get_action().
            self.last_skill_idx = None
            self.last_skill_names = None
            self.last_skill_probs = None
            self.last_skill_weight_probs = None
            self.last_skill_gate_probs = None
            self.last_skill_used = None

        self.config = config
        self.set_trainable_parameters(
            config.tune_projector,
            config.tune_diffusion_model,
            config.tune_skill_clf,
            config.tune_skill_emb,
        )

    def set_trainable_parameters(
        self,
        tune_projector: bool,
        tune_diffusion_model: bool,
        tune_skill_clf: bool = False,
        tune_skill_emb: bool = False,
    ):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_skill_clf = tune_skill_clf
        self.tune_skill_emb = tune_skill_emb
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if self.config.use_skill_emb:
            if not tune_skill_clf:
                self.skill_proj.requires_grad_(False)
                self.skill_clf.requires_grad_(False)
            if self.config.use_skill_encoder_hidden_states:
                self.skill_emb_proj.requires_grad_(False)
            if not tune_skill_emb:
                self.skill_emb_bank.requires_grad_(False)
                if not self.config.use_skill_encoder_hidden_states:
                    self.skill_emb_proj.requires_grad_(False)
                if getattr(self, "skill_encoder_proj", None) is not None:
                    self.skill_encoder_proj.requires_grad_(False)
                if getattr(self, "skill_film", None) is not None:
                    self.skill_film.requires_grad_(False)
                if getattr(self, "skill_gate", None) is not None:
                    self.skill_gate.requires_grad_(False)
        # Task-conditioned adapters are always trainable when present —
        # they are the task-transfer-specific modules and must update during fine-tuning.
        for adapter_name in ("state_adapter", "action_adapter", "decoder_adapter"):
            adapter = getattr(self, adapter_name, None)
            if adapter is not None:
                adapter.requires_grad_(True)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        if self.config.use_skill_emb:
            print(f"Tune skill classifier: {self.tune_skill_clf}")
            print(f"Tune skill embedding bank: {self.tune_skill_emb}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()
            if self.config.use_skill_emb:
                if not self.tune_skill_clf:
                    self.skill_proj.eval()
                    self.skill_clf.eval()
                if self.config.use_skill_encoder_hidden_states:
                    self.skill_emb_proj.eval()
                if not self.tune_skill_emb:
                    self.skill_emb_bank.eval()
                    if not self.config.use_skill_encoder_hidden_states:
                        self.skill_emb_proj.eval()
                    if getattr(self, "skill_encoder_proj", None) is not None:
                        self.skill_encoder_proj.eval()
                    if getattr(self, "skill_film", None) is not None:
                        self.skill_film.eval()
                    if getattr(self, "skill_gate", None) is not None:
                        self.skill_gate.eval()

    def _apply_skill_film(
        self,
        action_features: torch.Tensor,
        skill_emb: torch.Tensor,
    ) -> torch.Tensor:
        gamma, beta = self.skill_film(skill_emb).chunk(2, dim=-1)
        if getattr(self.config, "use_skill_gate", False):
            gate = torch.sigmoid(self.skill_gate(skill_emb))
            gate = gate.to(device=action_features.device, dtype=action_features.dtype)
            if self.training and self.config.skill_gate_dropout > 0:
                keep = torch.rand_like(gate) >= self.config.skill_gate_dropout
                gate = gate * keep.to(dtype=gate.dtype)
        else:
            gate = torch.ones(
                gamma.shape[0],
                1,
                device=gamma.device,
                dtype=gamma.dtype,
            )

        gate = gate.unsqueeze(1)
        return action_features * (1 + gate * gamma.unsqueeze(1)) + gate * beta.unsqueeze(1)

    def _skill_gate_values(self, skill_emb: torch.Tensor) -> torch.Tensor | None:
        if not getattr(self.config, "use_skill_gate", False):
            return None
        return torch.sigmoid(self.skill_gate(skill_emb)).squeeze(-1)

    def _masked_mean_pool(
        self,
        features: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Masked mean pool (B, T, D) → (B, D), ignoring padding tokens."""
        if attn_mask is None:
            return features.mean(dim=1)
        mask = attn_mask.bool().float().unsqueeze(-1)  # (B, T, 1)
        denom = mask.sum(dim=1).clamp(min=1.0)         # (B, 1)
        return (features * mask).sum(dim=1) / denom    # (B, D)

    def _skill_embedding_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute diversity and non-zero losses directly on skill_emb_bank weights."""
        E = self.skill_emb_bank.weight                         # (K, skill_emb_dim)
        E_norm = F.normalize(E, dim=-1)                        # (K, skill_emb_dim)
        G = E_norm @ E_norm.T                                  # (K, K) cosine gram
        K = G.shape[0]
        eye = torch.eye(K, device=G.device, dtype=G.dtype)
        div_loss = ((G - eye) ** 2).sum()                      # off-diag entries penalized
        norm_loss = (1.0 / (E.norm(dim=-1) ** 2 + 1e-6)).mean()
        return div_loss, norm_loss

    def _masked_action_mse(
        self,
        pred_actions: torch.Tensor,
        velocity: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        denom = action_mask.sum().clamp(min=1.0)
        return (F.mse_loss(pred_actions, velocity, reduction="none") * action_mask).sum() / denom

    def _predict_training_velocity_with_skill_emb(
        self,
        vl_embs: torch.Tensor,
        vl_attn_mask: torch.Tensor | None,
        state_features: torch.Tensor,
        noisy_trajectory: torch.Tensor,
        t_discretized: torch.Tensor,
        embodiment_id: torch.Tensor,
        task_emb: torch.Tensor | None,
        skill_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)
        if task_emb is not None:
            action_features = self.action_adapter(action_features, task_emb)

        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=action_features.device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(
            action_features.shape[0], -1, -1
        )
        model_vl_embs = vl_embs
        model_vl_attn_mask = vl_attn_mask

        if self.config.use_skill_emb:
            if self.config.use_skill_encoder_hidden_states:
                model_vl_embs, model_vl_attn_mask = self._append_skill_to_encoder_hidden_states(
                    model_vl_embs, model_vl_attn_mask, skill_emb
                )
                sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
            elif self.config.use_skill_film:
                action_features = self._apply_skill_film(action_features, skill_emb)
                sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
            else:
                skill_token = self.skill_emb_proj(skill_emb).unsqueeze(1)
                sa_embs = torch.cat((state_features, future_tokens, skill_token, action_features), dim=1)
        else:
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=model_vl_embs,
            encoder_attention_mask=model_vl_attn_mask,
            timestep=t_discretized,
            skill_conditioning=skill_emb if self.config.use_skill_dit_modulation else None,
            return_all_hidden_states=False,
        )
        pred = self.action_decoder(model_output, embodiment_id)
        if task_emb is not None:
            pred = self.decoder_adapter(pred, task_emb)

        return pred[:, -noisy_trajectory.shape[1]:]

    @torch.no_grad()
    def _predict_actions_with_skill_emb(
        self,
        vl_embs: torch.Tensor,
        vl_attn_mask: torch.Tensor | None,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        task_emb: torch.Tensor | None,
        skill_emb: torch.Tensor | None,
        initial_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the denoising loop with a specific skill embedding."""
        if self.config.use_skill_emb and skill_emb is None:
            raise ValueError("skill_emb must be provided when use_skill_emb=True")

        device = vl_embs.device
        batch_size = vl_embs.shape[0]
        actions = (
            initial_actions.clone()
            if initial_actions is not None
            else torch.randn(
                size=(batch_size, self.config.action_horizon, self.config.action_dim),
                dtype=vl_embs.dtype,
                device=device,
            )
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)

            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            if task_emb is not None:
                action_features = self.action_adapter(action_features, task_emb)

            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            if self.config.use_skill_emb:
                if self.config.use_skill_encoder_hidden_states:
                    sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
                elif self.config.use_skill_film:
                    action_features = self._apply_skill_film(action_features, skill_emb)
                    sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
                else:
                    skill_token = self.skill_emb_proj(skill_emb).unsqueeze(1)
                    sa_embs = torch.cat((state_features, future_tokens, skill_token, action_features), dim=1)
            else:
                sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                encoder_attention_mask=vl_attn_mask,
                timestep=timesteps_tensor,
                skill_conditioning=skill_emb if self.config.use_skill_dit_modulation else None,
            )
            pred = self.action_decoder(model_output, embodiment_id)
            if task_emb is not None:
                pred = self.decoder_adapter(pred, task_emb)

            pred_velocity = pred[:, -self.action_horizon:]
            actions = actions + dt * pred_velocity

        return actions

    def _append_skill_to_encoder_hidden_states(
        self,
        vl_embs: torch.Tensor,
        vl_attn_mask: torch.Tensor | None,
        skill_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Append one projected skill token to the VLM encoder sequence."""
        skill_token = self.skill_encoder_proj(
            skill_emb.to(dtype=next(self.skill_encoder_proj.parameters()).dtype)
        ).unsqueeze(1)
        skill_token = skill_token.to(device=vl_embs.device, dtype=vl_embs.dtype)
        vl_embs = torch.cat((vl_embs, skill_token), dim=1)

        if vl_attn_mask is not None:
            skill_mask = torch.ones(
                vl_attn_mask.shape[0],
                1,
                device=vl_attn_mask.device,
                dtype=vl_attn_mask.dtype,
            )
            vl_attn_mask = torch.cat((vl_attn_mask, skill_mask), dim=1)

        return vl_embs, vl_attn_mask

    @staticmethod
    def _debug_env_enabled(name: str) -> bool:
        return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _debug_clone_batch_feature(batch: BatchFeature) -> BatchFeature:
        return BatchFeature(
            data={
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
        )

    @staticmethod
    def _debug_float_list(tensor: torch.Tensor, digits: int = 4) -> list[float]:
        return [round(float(x), digits) for x in tensor.detach().float().cpu().flatten().tolist()]

    @staticmethod
    def _summarize_action_prediction(actions: torch.Tensor, digits: int = 4) -> str:
        first_step = [
            round(float(x), digits)
            for x in actions.detach().float().cpu()[0, 0, :7].tolist()
        ]
        mean = round(float(actions.detach().float().mean().item()), digits)
        std = round(float(actions.detach().float().std(unbiased=False).item()), digits)
        return f"first7={first_step} mean={mean} std={std}"

    @staticmethod
    def _debug_action_rows(action: torch.Tensor, horizon: int, digits: int = 4) -> list[list[float]]:
        return [
            round(float(row), digits)
            for row in action.detach().float().cpu()[0, 0, :7].tolist()
        ]

    def _maybe_debug_train_get_action(
        self,
        backbone_output_for_get_action: BatchFeature,
        action_input_for_get_action: BatchFeature,
        gt_action: torch.Tensor,
        forward_pred_action: torch.Tensor,
    ) -> None:
        if not self._debug_env_enabled("GR00T_DEBUG_TRAIN_GET_ACTION"):
            return

        try:
            max_calls = int(os.environ.get("GR00T_DEBUG_TRAIN_GET_ACTION_MAX_CALLS", "5"))
        except ValueError:
            max_calls = 5

        call_idx = getattr(self, "_debug_train_get_action_calls", 0)
        if call_idx >= max_calls:
            return
        self._debug_train_get_action_calls = call_idx + 1

        was_training = self.training
        try:
            self.eval()
            with torch.no_grad():
                pred = self.get_action(
                    self._debug_clone_batch_feature(backbone_output_for_get_action),
                    self._debug_clone_batch_feature(action_input_for_get_action),
                )["action_pred"].detach()
        finally:
            self.train(was_training)

        gt_action = gt_action.detach()
        forward_pred_action = forward_pred_action.detach()
        get_action_pred_action = pred.detach()
        horizon_preview = min(
            5,
            gt_action.shape[1],
            forward_pred_action.shape[1],
            get_action_pred_action.shape[1],
        )

        # print(
        #     "[TRAIN_GET_ACTION_DEBUG] "
        #     f"call={call_idx} gt_shape={tuple(gt_action.shape)} "
        #     f"forward_pred_shape={tuple(forward_pred_action.shape)} "
        #     f"get_action_pred_shape={tuple(get_action_pred_action.shape)} "
        #     f"state_shape={tuple(action_input_for_get_action.state.shape)} "
        #     f"horizon_logged={horizon_preview}"
        # )
        print(
            "[TRAIN_GET_ACTION_DEBUG] "
            f"gt_action[:7]={self._debug_action_rows(gt_action, horizon_preview)}"
        )
        # print(
        #     "[TRAIN_GET_ACTION_DEBUG] "
        #     f"forward_pred_action[:7]={self._debug_action_rows(forward_pred_action, horizon_preview)}"
        # )
        print(
            "[TRAIN_GET_ACTION_DEBUG] "
            f"get_action_pred_action[:7]={self._debug_action_rows(get_action_pred_action, horizon_preview)}"
        )

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def _pool_for_routing(
        self,
        backbone_features: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Masked mean pool backbone features for the router.

        Uses the attention mask to exclude padding tokens.  If
        ``router_lang_tail`` is set in the config (number of trailing valid
        tokens to pool), only those tokens are used — in VLMs instruction text
        tokens appear at the end of the sequence after image tokens, so this
        focuses the routing signal on language rather than vision.

        Args:
            backbone_features: (B, T, D) — full backbone hidden states.
            attn_mask: (B, T) binary mask; 1 = valid, 0 = padding.

        Returns:
            pooled: (B, D)
        """
        if attn_mask is None:
            # Fallback: unmasked mean over all tokens
            return backbone_features.mean(dim=1)

        # Boolean mask
        mask = attn_mask.bool()  # (B, T)

        lang_tail = getattr(self.config, "router_lang_tail", 0)
        if lang_tail > 0:
            # Build a mask that keeps only the last `lang_tail` valid tokens
            # per sample. We scan from the end and accumulate counts.
            # reversed cumsum of the valid mask gives remaining valid tokens
            # from each position onward (reversed).
            rev_cumsum = mask.flip(dims=[1]).cumsum(dim=1).flip(dims=[1])  # (B, T)
            mask = mask & (rev_cumsum <= lang_tail)

        mask_f = mask.float().unsqueeze(-1)          # (B, T, 1)
        denom  = mask_f.sum(dim=1).clamp(min=1.0)   # (B, 1)
        pooled = (backbone_features * mask_f).sum(dim=1) / denom  # (B, D)
        return pooled

    def compute_task_embedding(
        self,
        backbone_features: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns (B, backbone_emb_dim) task embedding via router soft-weighted sum."""
        pooled = self._pool_for_routing(backbone_features, attn_mask)            # (B, D)
        weights = torch.softmax(self.router(pooled), dim=-1)                      # (B, K)
        task_emb = (weights.unsqueeze(-1) * self.task_emb_bank.weight).sum(dim=1) # (B, D)
        return task_emb

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)

        if hasattr(self, 'task_emb_bank') and hasattr(self, 'router'):
            attn_mask = backbone_output.get("backbone_attention_mask")
            pooled = self._pool_for_routing(backbone_features, attn_mask)      # (B, D)
            weights = torch.softmax(self.router(pooled), dim=-1)                # (B, K)
            task_tokens = weights.unsqueeze(-1) * self.task_emb_bank.weight     # (B, K, D)
            backbone_features = torch.cat([backbone_features, task_tokens], dim=1)
            # Stash router weights for the diversity loss computed in forward()
            backbone_output["_router_weights"] = weights
            if attn_mask is not None:
                task_mask = torch.ones(
                    attn_mask.shape[0], task_tokens.shape[1],
                    device=attn_mask.device, dtype=attn_mask.dtype
                )
                backbone_output["backbone_attention_mask"] = torch.cat([attn_mask, task_mask], dim=1)

        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()
        debug_backbone_output_for_get_action = None
        debug_action_input_for_get_action = None
        if self._debug_env_enabled("GR00T_DEBUG_TRAIN_GET_ACTION"):
            debug_backbone_output_for_get_action = self._debug_clone_batch_feature(backbone_output)
            debug_action_input_for_get_action = self._debug_clone_batch_feature(action_input)

        # Save raw backbone features before task tokens are appended, so
        # compute_task_embedding pools only the original VLM sequence.
        raw_bfeats = (
            backbone_output["backbone_features"].clone()
            if self.config.use_task_adapter
            else None
        )
        backbone_output = self.process_backbone_output(backbone_output)

        if self.config.expand_batch is not None:
            for k, v in backbone_output.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                backbone_output[k] = expanded

            for k, v in action_input.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                action_input[k] = expanded

            # Keep raw_bfeats aligned with expand_batch repetition.
            if raw_bfeats is not None:
                raw_bfeats = raw_bfeats.repeat(
                    self.config.expand_batch, *([1] * (raw_bfeats.dim() - 1))
                )

        # Get vision and language embeddings.
        if backbone_output.backbone_features_multi is not None:
            vl_embs = backbone_output.backbone_features_multi
            vl_attn_mask = backbone_output.backbone_attention_mask_multi
            num_action = action_input.state.shape[0]
        else:
            vl_embs = backbone_output.backbone_features
            vl_attn_mask = backbone_output.backbone_attention_mask
            # vl_attn_mask = None
            num_action = 0

        # Skill classification loss + skill embedding (computed before padding filter).
        skill_clf_loss = None
        skill_emb = None    # (B, skill_emb_dim), set when use_skill_emb
        skill_token = None  # (B, 1, input_emb_dim), set for action-side skill-token mode only
        if self.config.use_skill_emb:
            pooled = self._masked_mean_pool(vl_embs, vl_attn_mask)  # (B, D)
            pooled = pooled.to(dtype=next(self.skill_proj.parameters()).dtype)
            skill_logits = self.skill_clf(self.skill_proj(pooled))   # (B, num_skills)
            if "skill_id" in action_input:
                skill_label = action_input["skill_id"]            # (B,) long
                skill_clf_loss = F.cross_entropy(skill_logits, skill_label)

            if self.config.use_weighted_skill_router:
                # Soft-weighted: differentiable weighted sum over all skill embeddings
                skill_weights = torch.softmax(skill_logits, dim=-1)           # (B, K)
                skill_emb = skill_weights @ self.skill_emb_bank.weight        # (B, skill_emb_dim)
            elif "skill_id" in action_input:
                # Top-1 hard selection using ground-truth label
                skill_emb = self.skill_emb_bank(skill_label)                  # (B, skill_emb_dim)
            else:
                # Top-1 inference fallback: use classifier argmax
                skill_idx = skill_logits.argmax(dim=-1)
                skill_emb = self.skill_emb_bank(skill_idx)                    # (B, skill_emb_dim)

            # Project to a DiT input token only for the action-side skill-token path.
            if not self.config.use_skill_film and not self.config.use_skill_encoder_hidden_states:
                skill_token = self.skill_emb_proj(skill_emb).unsqueeze(1)     # (B, 1, input_emb_dim)

            if self.tune_skill_clf:
                # Stage 1: classifier-only loss
                assert skill_clf_loss is not None, "skill_clf_loss is None in Stage 1 — check skill_label in batch"
                loss = self.config.skill_clf_coeff * skill_clf_loss
                output_dict = {
                    "loss": loss,
                    "skill_clf_loss": skill_clf_loss.detach(),
                    "skill_pred_eval": skill_logits.argmax(dim=-1).detach(),
                    "skill_label_eval": skill_label.detach(),
                }
                return BatchFeature(data=output_dict)

        device = vl_embs.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        if num_action > 0:
            embodiment_id = embodiment_id[:1].repeat(num_action)

        # Compute task embedding (B, D) from raw VLM features for FiLM adapters.
        task_emb = self.compute_task_embedding(raw_bfeats) if raw_bfeats is not None else None

        # Embed state.
        if len(action_input.state.shape) < 3:
            state_input = action_input.state.unsqueeze(1)
        else:
            state_input = action_input.state

        # Embed noised action trajectory.
        actions = action_input.action
        action_mask = action_input.action_mask
        padding_value = 0.0
        is_padding_value = (actions == padding_value)
        padding_mask = ~is_padding_value.all(dim=2).all(dim=1)

        if not padding_mask.all():
            # Filter tensors by padding_mask
            state_input = state_input[padding_mask]
            actions = actions[padding_mask]
            action_mask = action_mask[padding_mask]
            embodiment_id = embodiment_id[padding_mask] if embodiment_id is not None else None
            if task_emb is not None:
                task_emb = task_emb[padding_mask]
            vl_embs = vl_embs[padding_mask]
            if vl_attn_mask is not None:
                vl_attn_mask = vl_attn_mask[padding_mask]
            if self.config.use_skill_emb:
                skill_emb = skill_emb[padding_mask]

        state_features = self.state_encoder(state_input, embodiment_id)
        if task_emb is not None:
            state_features = self.state_adapter(state_features, task_emb)

        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        pred_actions = self._predict_training_velocity_with_skill_emb(
            vl_embs,
            vl_attn_mask,
            state_features,
            noisy_trajectory,
            t_discretized,
            embodiment_id,
            task_emb,
            skill_emb,
        )
        forward_pred_action = noise + pred_actions

        # Stage-gated loss computation.
        # Stage 2 or normal: action denoising loss
        denoise_mse = self._masked_action_mse(pred_actions, velocity, action_mask)
        loss = denoise_mse
        output_dict = {"loss": loss}

        if self.config.use_skill_emb:
            gate_values = self._skill_gate_values(skill_emb)
            if gate_values is not None:
                gate_mean = gate_values.mean()
                output_dict["skill_gate_mean"] = gate_mean.detach()
                gate_l1_loss = gate_mean
                if getattr(self.config, "skill_gate_l1_coeff", 0.0) > 0:
                    loss = loss + self.config.skill_gate_l1_coeff * gate_l1_loss
                    output_dict["skill_gate_l1_loss"] = gate_l1_loss.detach()

            if not self.training:
                zero_skill_emb = torch.zeros_like(skill_emb)
                shuffle_idx = torch.randperm(skill_emb.shape[0], device=skill_emb.device)
                if skill_emb.shape[0] == 1:
                    shuffled_skill_emb = self.skill_emb_bank.weight[
                        torch.randint(
                            0,
                            self.config.num_skills,
                            (1,),
                            device=skill_emb.device,
                        )
                    ].to(dtype=skill_emb.dtype)
                else:
                    shuffled_skill_emb = skill_emb[shuffle_idx]
                zero_pred_actions = self._predict_training_velocity_with_skill_emb(
                    vl_embs,
                    vl_attn_mask,
                    state_features,
                    noisy_trajectory,
                    t_discretized,
                    embodiment_id,
                    task_emb,
                    zero_skill_emb,
                )
                shuffle_pred_actions = self._predict_training_velocity_with_skill_emb(
                    vl_embs,
                    vl_attn_mask,
                    state_features,
                    noisy_trajectory,
                    t_discretized,
                    embodiment_id,
                    task_emb,
                    shuffled_skill_emb,
                )
                zero_mse = self._masked_action_mse(zero_pred_actions, velocity, action_mask)
                shuffle_mse = self._masked_action_mse(
                    shuffle_pred_actions, velocity, action_mask
                )
                output_dict["skill_mse_normal_eval"] = denoise_mse.detach()
                output_dict["skill_mse_zero_eval"] = zero_mse.detach()
                output_dict["skill_mse_shuffle_eval"] = shuffle_mse.detach()
                output_dict["skill_mse_zero_delta_eval"] = (zero_mse - denoise_mse).detach()
                output_dict["skill_mse_shuffle_delta_eval"] = (
                    shuffle_mse - denoise_mse
                ).detach()

            # CE loss: skip in Stage 2 (tune_skill_emb), include otherwise
            if not self.tune_skill_emb and skill_clf_loss is not None:
                loss = loss + self.config.skill_clf_coeff * skill_clf_loss
                output_dict["skill_clf_loss"] = skill_clf_loss.detach()

            div_loss, norm_loss = self._skill_embedding_losses()
            loss = loss + self.config.skill_div_coeff * div_loss
            loss = loss + self.config.skill_norm_coeff * norm_loss
            output_dict["skill_div_loss"] = div_loss.detach()
            output_dict["skill_norm_loss"] = norm_loss.detach()

            if skill_clf_loss is not None:
                output_dict["skill_pred_eval"] = skill_logits.argmax(dim=-1).detach()
                output_dict["skill_label_eval"] = skill_label.detach()

            output_dict["loss"] = loss

        # Router diversity (load-balancing) loss — maximizes routing entropy so all K
        # embedding slots are used equally rather than collapsing to one.
        router_weights = backbone_output.get("_router_weights")
        if router_weights is not None and self.config.router_diversity_coeff > 0:
            mean_w = router_weights.mean(dim=0)                                   # (K,)
            # Negative entropy of mean routing distribution; adding this minimizes entropy
            # which means we *maximize* it by subtracting from the loss.
            neg_entropy = (mean_w * (mean_w + 1e-8).log()).sum()                  # scalar <= 0
            loss = loss + self.config.router_diversity_coeff * neg_entropy
            output_dict["loss"] = loss

        if debug_backbone_output_for_get_action is not None:
            self._maybe_debug_train_get_action(
                debug_backbone_output_for_get_action,
                debug_action_input_for_get_action,
                actions,
                forward_pred_action,
            )
        return BatchFeature(data=output_dict)

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # Save raw backbone features before task tokens are appended.
        raw_bfeats = (
            backbone_output["backbone_features"].clone()
            if self.config.use_task_adapter
            else None
        )
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        if backbone_output.get('backbone_features_multi', None) is not None:
            vl_embs = backbone_output.backbone_features_multi
            vl_attn_mask = backbone_output.backbone_attention_mask_multi
            num_action = action_input.state.shape[0]
        else:
            vl_embs = backbone_output.backbone_features
            vl_attn_mask = backbone_output.backbone_attention_mask
            num_action = 0

        device = vl_embs.device
        batch_size = vl_embs.shape[0]
        embodiment_id = action_input.embodiment_id

        # Compute task embedding (B, D) once for all denoising steps.
        task_emb = self.compute_task_embedding(raw_bfeats) if raw_bfeats is not None else None

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)
        if task_emb is not None:
            state_features = self.state_adapter(state_features, task_emb)

        # Compute skill conditioning once for all denoising steps.
        skill_emb = None
        routed_skill_emb = None
        if self.config.use_skill_emb:
            pooled = self._masked_mean_pool(vl_embs, vl_attn_mask)  # (B, D)
            pooled = pooled.to(dtype=next(self.skill_proj.parameters()).dtype)
            skill_logits = self.skill_clf(self.skill_proj(pooled))

            # Record the routed skill (argmax) for logging / video overlay.
            # Computed for both routing modes; the weighted router still uses
            # soft weights below, this is display-only.
            skill_idx = skill_logits.argmax(dim=-1)                      # (B,)
            skill_probs = torch.softmax(skill_logits.float(), dim=-1)    # (B, K)
            selected_probs = skill_probs.gather(
                -1, skill_idx.unsqueeze(-1)
            ).squeeze(-1)                                                # (B,)
            skill_vocab = self.config.skill_vocab
            if skill_vocab is None:
                skill_vocab = [str(i) for i in range(self.config.num_skills)]
            skill_names = [
                skill_vocab[i] if 0 <= i < len(skill_vocab) else f"<skill:{i}>"
                for i in skill_idx.tolist()
            ]
            self.last_skill_idx = skill_idx.tolist()
            self.last_skill_names = skill_names
            self.last_skill_probs = [round(p, 4) for p in selected_probs.tolist()]
            self.last_skill_weight_probs = [
                [round(v, 6) for v in row]
                for row in skill_probs.detach().cpu().tolist()
            ]

            if self.config.use_weighted_skill_router:
                skill_weights = torch.softmax(skill_logits, dim=-1)
                skill_emb = skill_weights @ self.skill_emb_bank.weight    # (B, skill_emb_dim)
            else:
                skill_emb = self.skill_emb_bank(skill_idx)               # (B, skill_emb_dim)

            routed_skill_emb = skill_emb
            if getattr(self.config, "use_skill_gate", False):
                gate_probs = torch.sigmoid(self.skill_gate(skill_emb)).squeeze(-1)
            else:
                gate_probs = torch.ones(
                    batch_size,
                    device=skill_emb.device,
                    dtype=skill_emb.dtype,
                )
            self.last_skill_gate_probs = [
                round(float(v), 6) for v in gate_probs.detach().float().cpu().tolist()
            ]
            self.last_skill_used = [v >= 0.1 for v in self.last_skill_gate_probs]
            print(
                "[SKILL] "
                f"skill={skill_names} idx={skill_idx.tolist()} "
                f"prob={self.last_skill_probs} gate={self.last_skill_gate_probs} "
                f"used@0.1={self.last_skill_used}"
            )

            # Inference-time skill-embedding ablations (default "normal" leaves
            # the routed embedding untouched). Set via `action_head.skill_eval_mode`.
            skill_eval_mode = getattr(self, "skill_eval_mode", "normal")
            if skill_eval_mode == "zero":
                skill_emb = torch.zeros_like(skill_emb)
                self.last_skill_idx = [-1] * batch_size
                self.last_skill_names = ["<zero>"] * batch_size
                self.last_skill_probs = None
                self.last_skill_gate_probs = [0.0] * batch_size
                self.last_skill_used = [False] * batch_size
                print(f"[SKILL] mode=zero (skill embedding zeroed)")
            elif skill_eval_mode == "shuffle":
                rand_idx = torch.randint(
                    0, self.config.num_skills, skill_idx.shape, device=skill_idx.device
                )
                skill_emb = self.skill_emb_bank(rand_idx)
                self.last_skill_idx = rand_idx.tolist()
                self.last_skill_names = [
                    skill_vocab[i] if 0 <= i < len(skill_vocab) else f"<skill:{i}>"
                    for i in rand_idx.tolist()
                ]
                rand_probs = skill_probs.gather(
                    -1, rand_idx.unsqueeze(-1)
                ).squeeze(-1)
                self.last_skill_probs = [round(p, 4) for p in rand_probs.tolist()]
                if getattr(self.config, "use_skill_gate", False):
                    gate_probs = torch.sigmoid(self.skill_gate(skill_emb)).squeeze(-1)
                else:
                    gate_probs = torch.ones(
                        batch_size,
                        device=skill_emb.device,
                        dtype=skill_emb.dtype,
                    )
                self.last_skill_gate_probs = [
                    round(float(v), 6) for v in gate_probs.detach().float().cpu().tolist()
                ]
                self.last_skill_used = [v >= 0.1 for v in self.last_skill_gate_probs]
                print(
                    "[SKILL] mode=shuffle "
                    f"skill={self.last_skill_names} idx={self.last_skill_idx} "
                    f"prob={self.last_skill_probs} gate={self.last_skill_gate_probs} "
                    f"used@0.1={self.last_skill_used}"
                )

        debug_skill_actions = self._debug_env_enabled("GR00T_DEBUG_SKILL_ACTIONS")
        initial_actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )
        action_pred = self._predict_actions_with_skill_emb(
            vl_embs,
            vl_attn_mask,
            state_features,
            embodiment_id,
            task_emb,
            skill_emb,
            initial_actions=initial_actions,
        )

        if debug_skill_actions and self.config.use_skill_emb:
            skill_vocab = self.config.skill_vocab
            if skill_vocab is None:
                skill_vocab = [str(i) for i in range(self.config.num_skills)]

            routed_preview = self._summarize_action_prediction(
                self._predict_actions_with_skill_emb(
                    vl_embs,
                    vl_attn_mask,
                    state_features,
                    embodiment_id,
                    task_emb,
                    routed_skill_emb,
                    initial_actions=initial_actions,
                )
            )
            active_preview = self._summarize_action_prediction(action_pred)
            print(
                "[SKILL-DBG] "
                f"mode={getattr(self, 'skill_eval_mode', 'normal')} "
                f"routed={self.last_skill_names} "
                f"active={active_preview} routed_preview={routed_preview}"
            )

            for skill_idx in range(self.config.num_skills):
                skill_ids = torch.full(
                    (batch_size,),
                    skill_idx,
                    device=device,
                    dtype=torch.long,
                )
                alt_skill_emb = self.skill_emb_bank(skill_ids)
                alt_pred = self._predict_actions_with_skill_emb(
                    vl_embs,
                    vl_attn_mask,
                    state_features,
                    embodiment_id,
                    task_emb,
                    alt_skill_emb,
                    initial_actions=initial_actions,
                )
                delta = (alt_pred - action_pred).abs().mean().item()
                print(
                    "[SKILL-DBG] "
                    f"skill={skill_vocab[skill_idx] if skill_idx < len(skill_vocab) else skill_idx} "
                    f"idx={skill_idx} "
                    f"action={self._summarize_action_prediction(alt_pred)} "
                    f"delta_to_active={delta:.4f}"
                )

        return BatchFeature(data={"action_pred": action_pred})

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
