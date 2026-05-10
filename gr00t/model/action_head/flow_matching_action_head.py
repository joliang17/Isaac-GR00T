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

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(
        default=32, metadata={"help": "Number of target vision tokens."}
    )

    # Skill embedding config (gated by use_skill_emb; backward-compatible when False)
    use_skill_emb: bool = field(
        default=False,
        metadata={"help": "Enable skill embedding: MLP classifier (Stage 1) + learnable skill token concat (Stage 2)."},
    )
    skill_vocab: list = field(
        default=None,
        metadata={"help": "List of skill name strings for label extraction."},
    )
    num_skills: int = field(
        default=0,
        metadata={"help": "Number of skill classes."},
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
        metadata={"help": "Weight for non-zero norm loss on skill embedding bank."},
    )
    use_weighted_skill_router: bool = field(
        default=False,
        metadata={"help": "Use soft-weighted routing (softmax) instead of top-1 argmax."},
    )
    tune_skill_clf: bool = field(
        default=False,
        metadata={"help": "Stage 1: freeze all except skill classifier MLP."},
    )
    tune_skill_emb: bool = field(
        default=False,
        metadata={"help": "Stage 2: freeze skill classifier, train skill embeddings + DiT."},
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

        self.model = DiT(**config.diffusion_model_cfg)
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

        if config.use_skill_emb:
            assert config.num_skills > 0, "num_skills must be > 0 when use_skill_emb=True"
            backbone_emb_dim = config.backbone_embedding_dim  # 2048 (clf features dim)
            self.skill_proj = nn.Sequential(
                nn.Linear(backbone_emb_dim, config.skill_proj_hidden_dim),
                nn.ReLU(),
            )
            self.skill_clf = nn.Linear(config.skill_proj_hidden_dim, config.num_skills)
            self.skill_emb_bank = nn.Embedding(config.num_skills, config.skill_emb_dim)
            self.skill_emb_proj = nn.Linear(config.skill_emb_dim, config.input_embedding_dim)

        self.config = config
        self.set_trainable_parameters(
            config.tune_projector,
            config.tune_diffusion_model,
            getattr(config, "tune_skill_clf", False),
            getattr(config, "tune_skill_emb", False),
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
            if not tune_skill_emb:
                self.skill_emb_bank.requires_grad_(False)
                self.skill_emb_proj.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        if self.config.use_skill_emb:
            print(f"Tune skill classifier: {self.tune_skill_clf}")
            print(f"Tune skill embedding bank: {self.tune_skill_emb}")
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
                if not self.tune_skill_emb:
                    self.skill_emb_bank.eval()
                    self.skill_emb_proj.eval()

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
        """Compute diversity and non-zero losses on skill_emb_bank weights."""
        E = self.skill_emb_bank.weight                         # (K, skill_emb_dim)
        E_norm = F.normalize(E, dim=-1)
        G = E_norm @ E_norm.T                                  # (K, K) cosine gram
        K = G.shape[0]
        eye = torch.eye(K, device=G.device, dtype=G.dtype)
        div_loss = ((G - eye) ** 2).sum()                      # penalize off-diagonal
        norm_loss = (1.0 / (E.norm(dim=-1) ** 2 + 1e-6)).mean()
        return div_loss, norm_loss

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

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

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        vl_attn_mask = backbone_output.backbone_attention_mask
        device = vl_embs.device

        # Skill classification (uses last-layer clf features) + skill token for DiT.
        skill_clf_loss = None
        skill_token = None
        if self.config.use_skill_emb:
            clf_embs = backbone_output.get("backbone_clf_features")
            if clf_embs is None:
                clf_embs = vl_embs  # fallback: use DiT features
            pooled = self._masked_mean_pool(clf_embs, vl_attn_mask)   # (B, D)
            pooled = pooled.to(dtype=next(self.skill_proj.parameters()).dtype)
            skill_logits = self.skill_clf(self.skill_proj(pooled))     # (B, num_skills)

            if "skill_id" in action_input:
                skill_label = action_input["skill_id"]
                skill_clf_loss = F.cross_entropy(skill_logits, skill_label)

            if self.config.use_weighted_skill_router:
                skill_weights = torch.softmax(skill_logits, dim=-1)
                skill_emb = skill_weights @ self.skill_emb_bank.weight
                skill_token = self.skill_emb_proj(skill_emb).unsqueeze(1)
            elif "skill_id" in action_input:
                skill_token = self.skill_emb_proj(
                    self.skill_emb_bank(skill_label)
                ).unsqueeze(1)
            else:
                skill_token = self.skill_emb_proj(
                    self.skill_emb_bank(skill_logits.argmax(dim=-1))
                ).unsqueeze(1)

            if self.tune_skill_clf:
                # Stage 1: classifier-only training
                assert skill_clf_loss is not None, "skill_id missing from batch in Stage 1"
                loss = self.config.skill_clf_coeff * skill_clf_loss
                return BatchFeature(data={
                    "loss": loss,
                    "skill_clf_loss": skill_clf_loss.detach(),
                    "skill_pred_eval": skill_logits.argmax(dim=-1).detach(),
                    "skill_label_eval": skill_label.detach(),
                })

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join state, future tokens, actions, and optional skill token along sequence dimension.
        # With skill_emb: [state | future_tokens | actions | skill_token]
        # Without:        [state | future_tokens | actions]
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        if self.config.use_skill_emb and skill_token is not None:
            sa_embs = torch.cat((state_features, future_tokens, action_features, skill_token), dim=1)
        else:
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )
        pred = self.action_decoder(model_output, embodiment_id)

        # Slice action tokens: layout differs by mode.
        # skill_emb: [state(1) | future(32) | actions(T) | skill(1)] → actions at [33 : 33+T]
        # original:  [state(1) | future(32) | actions(T)] → actions at [-T:]
        if self.config.use_skill_emb:
            start_index = state_features.shape[1] + future_tokens.shape[1]
            pred_actions = pred[:, start_index : start_index + actions.shape[1]]
        else:
            pred_actions = pred[:, -actions.shape[1]:]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = loss.sum() / action_mask.sum()
        output_dict = {"loss": loss}

        if self.config.use_skill_emb:
            if not self.tune_skill_emb and skill_clf_loss is not None:
                loss = loss + self.config.skill_clf_coeff * skill_clf_loss
                output_dict["skill_clf_loss"] = skill_clf_loss.detach()
            div_loss, norm_loss = self._skill_embedding_losses()
            loss = loss + self.config.skill_div_coeff * div_loss + self.config.skill_norm_coeff * norm_loss
            output_dict["skill_div_loss"] = div_loss.detach()
            output_dict["skill_norm_loss"] = norm_loss.detach()
            if skill_clf_loss is not None:
                output_dict["skill_pred_eval"] = skill_logits.argmax(dim=-1).detach()
                output_dict["skill_label_eval"] = skill_label.detach()
            output_dict["loss"] = loss

        return BatchFeature(data=output_dict)

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        vl_attn_mask = backbone_output.get("backbone_attention_mask")
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Compute skill token once for all denoising steps (uses last-layer clf features).
        skill_token = None
        if self.config.use_skill_emb:
            clf_embs = backbone_output.get("backbone_clf_features")
            if clf_embs is None:
                clf_embs = vl_embs  # fallback
            pooled = self._masked_mean_pool(clf_embs, vl_attn_mask)  # (B, D)
            pooled = pooled.to(dtype=next(self.skill_proj.parameters()).dtype)
            skill_logits = self.skill_clf(self.skill_proj(pooled))
            if self.config.use_weighted_skill_router:
                skill_weights = torch.softmax(skill_logits, dim=-1)
                skill_emb = skill_weights @ self.skill_emb_bank.weight
                skill_token = self.skill_emb_proj(skill_emb).unsqueeze(1)
            else:
                skill_idx = skill_logits.argmax(dim=-1)  # (B,)
                skill_vocab = self.config.skill_vocab
                if skill_vocab is None:
                    skill_vocab = [str(i) for i in range(self.config.num_skills)]
                skill_names = [
                    skill_vocab[i] if 0 <= i < len(skill_vocab) else f"<skill:{i}>"
                    for i in skill_idx.tolist()
                ]
                print(f"[SKILL] skill={skill_names} idx={skill_idx.tolist()}")
                skill_token = self.skill_emb_proj(
                    self.skill_emb_bank(skill_idx)
                ).unsqueeze(1)  # (B, 1, input_emb_dim)

        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join state, future tokens, actions, and optional skill token.
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            if self.config.use_skill_emb and skill_token is not None:
                sa_embs = torch.cat((state_features, future_tokens, action_features, skill_token), dim=1)
            else:
                sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output, embodiment_id)

            # Slice action tokens matching the layout used in forward().
            if self.config.use_skill_emb:
                start_index = state_features.shape[1] + future_tokens.shape[1]
                pred_velocity = pred[:, start_index : start_index + self.action_horizon]
            else:
                pred_velocity = pred[:, -self.action_horizon:]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        return BatchFeature(data={"action_pred": actions})

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
