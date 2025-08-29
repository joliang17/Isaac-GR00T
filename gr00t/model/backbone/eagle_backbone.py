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

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoProcessor
from transformers.feature_extraction_utils import BatchFeature

import gr00t

DEFAULT_EAGLE_PATH = os.path.join(
    os.path.dirname(gr00t.__file__), "model", "backbone", "eagle2_hg_model"
)


@staticmethod
def _index_batch_dict(bdict, mask_1d):
    """Index a dict of tensors by a boolean mask on the batch dim if shape[0]==B."""
    if mask_1d is None:
        return bdict
    take = mask_1d.nonzero(as_tuple=False).squeeze(-1)
    out = {}
    B = None
    # try to infer B from a common key
    for v in bdict.values():
        if isinstance(v, torch.Tensor) and v.dim() > 0:
            B = v.size(0)
            break
    for k, v in bdict.items():
        if isinstance(v, torch.Tensor) and v.dim() > 0 and v.size(0) == B:
            out[k] = v.index_select(0, take)
        else:
            out[k] = v
    return out, take


class EagleBackbone(nn.Module):

    def __init__(
        self,
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = False,
        use_flash_attention: bool = False,
        load_bf16: bool = False,
        eagle_path: str | None = None,
        project_to_dim: int = 1536,
    ):
        """
        Args:
            tune_llm: whether to tune the LLM model (default: True)
            tune_visual: whether to tune the visual model (default: False)
        """
        super().__init__()
        assert not reproject_vision, "Reproject vision is not implemented here, set to False"

        config = AutoConfig.from_pretrained(DEFAULT_EAGLE_PATH, trust_remote_code=True)
        self.eagle_model = AutoModel.from_config(config, trust_remote_code=True)

        # ADDED: add special tokens to tokenizer
        self.eagle_processor = AutoProcessor.from_pretrained(
            DEFAULT_EAGLE_PATH, trust_remote_code=True, use_fast=True
        )
        # specials = {"additional_special_tokens": ["<ACTIONS>", "<TOOLS>"]}
        specials = {"additional_special_tokens": ["[ACTIONS]", "[TOOLS]", "[EOT]", "[PAD_A]"]}
        self.eagle_processor.tokenizer.add_special_tokens(specials)
        self.eagle_tokenizer = self.eagle_processor.tokenizer

        # Cache special token ids (single-token by construction)
        self.actions_id = self.eagle_tokenizer.convert_tokens_to_ids("[ACTIONS]")
        self.tools_id = self.eagle_tokenizer.convert_tokens_to_ids("[TOOLS]")
        self.endtools_id = self.eagle_tokenizer.convert_tokens_to_ids("[EOT]")
        self.pad_action = self.eagle_tokenizer.convert_tokens_to_ids("[PAD_A]")
        self.pad_id = self.eagle_tokenizer.convert_tokens_to_ids("<|endoftext|>")

        if project_to_dim is not None:
            self.eagle_linear = torch.nn.Linear(2048, project_to_dim)
        else:
            self.eagle_linear = torch.nn.Identity()

        # needed since we don't use these layers. Also saves compute
        while len(self.eagle_model.language_model.model.layers) > select_layer:
            self.eagle_model.language_model.model.layers.pop(-1)

        self.select_layer = select_layer
        self.set_trainable_parameters(tune_llm, tune_visual)

        # Safety for generation
        if self.eagle_tokenizer.pad_token_id is None:
            self.eagle_tokenizer.pad_token = self.eagle_tokenizer.eos_token


    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.eagle_model.language_model.requires_grad_(False)
        if not tune_visual:
            self.eagle_model.vision_model.requires_grad_(False)
            self.eagle_model.mlp1.requires_grad_(False)
        print(f"Tune backbone llm: {self.tune_llm}")
        print(f"Tune backbone visual: {self.tune_visual}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_llm and not tune_visual:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Backbone trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No backbone trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if self.eagle_model.language_model and not self.tune_llm:
                self.eagle_model.language_model.eval()
            if self.eagle_model.vision_model and not self.tune_visual:
                self.eagle_model.vision_model.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward_eagle(self, vl_input: BatchFeature) -> BatchFeature:
        eagle_prefix = "eagle_"
        eagle_input = {
            k.removeprefix(eagle_prefix): v
            for k, v in vl_input.items()
            if k.startswith(eagle_prefix) and k != 'eagle_num_images'
        }
        del eagle_input["image_sizes"]

        # ADDED: add a special token here
        if "input_ids" in eagle_input and "attention_mask" in eagle_input:
            pad_token = torch.full(
                (eagle_input["input_ids"].size(0), 1),
                fill_value=self.pad_id,
                dtype=eagle_input["input_ids"].dtype,
                device=eagle_input["input_ids"].device,
            )
            pad_mask = torch.ones(
                (eagle_input["attention_mask"].size(0), 1),
                dtype=eagle_input["attention_mask"].dtype,
                device=eagle_input["attention_mask"].device,
            )
            eagle_input["input_ids"] = torch.cat([eagle_input["input_ids"], pad_token], dim=1)
            eagle_input["attention_mask"] = torch.cat([eagle_input["attention_mask"], pad_mask], dim=1)

        eagle_input.pop('llm_labels')
        eagle_output = self.eagle_model(**eagle_input, output_hidden_states=True, return_dict=True)
        eagle_features = eagle_output.hidden_states[self.select_layer]

        eagle_features = self.eagle_linear(eagle_features)
        eagle_logits = eagle_output.logits

        attn = eagle_input.get("attention_mask")   # [B, T]
        route_pos = attn.sum(dim=1) - 1        # [B]

        return eagle_logits, route_pos, eagle_features, eagle_input["attention_mask"]


    def calculate_route_loss(self, vl_input, eagle_logits, route_pos):
        """
        Two-class CE at the route position: { [ACTIONS], [TOOLS] }.
        """
        B = eagle_logits.shape[0]
        idx = torch.arange(B, device=eagle_logits.device)
        step_logits = eagle_logits[idx, route_pos, :]            # [B, vocab]
        route_logits_2 = torch.stack(
            [step_logits[:, self.actions_id], step_logits[:, self.tools_id]], dim=-1
        )  # [B, 2]

        raw_label, route_label = self._extract_route_targets(vl_input)
        if (route_label == -100).any():
            bad = torch.unique(raw_label[route_label == -100]).tolist()
            raise RuntimeError(f"Unexpected route tokens in step_input_ids: {bad}")

        loss = F.cross_entropy(route_logits_2, route_label)
        return loss, raw_label, route_label  # route_label in {0,1}


    def _extract_route_targets(self, vl_input):
        """Return (raw_token_id[B], class_index[B]) for last non-pad token in step_*."""
        step_ids  = vl_input["step_input_ids"]               # [B, T]
        step_mask = vl_input["step_attention_mask"]          # [B, T]
        (step_ids * step_mask).sum(dim=1)

        B, T = step_ids.size()
        device = step_ids.device
        idx = torch.arange(B, device=device)

        # left-padded: 0...0 1...1
        # #valid = step_mask.sum(); first_pos = T - #valid
        valid = step_mask.sum(dim=1).clamp(min=1)            # [B], avoid 0 to stay in-bounds
        first_pos = (T - valid).to(torch.long)               # [B]
        raw_label = step_ids[idx, first_pos]                 # [B] (token ids)

        # map to {0:[ACTIONS], 1:[TOOLS]}, keep -100 for unexpected
        route_label = torch.full_like(raw_label, -100, dtype=torch.long)
        route_label[raw_label == self.actions_id] = 0
        route_label[raw_label == self.tools_id]  = 1
        return raw_label, route_label


    def _tag_ce_loss(self, eagle_input, eagle_labels):
        # Only compute CE where labels != -100 (you decide what to supervise in collate)
        out = self.eagle_model(
            **eagle_input,
            labels=eagle_labels,      # same shape as input_ids, -100 where we don’t supervise
            return_dict=True
        )
        return out.loss
        
    def _tools_lm_loss(self, vl_input, is_tools: torch.Tensor):
        """
        Teacher-forcing LM loss on the target tool text in vl_input['step_input_ids'].
        Only computed for rows where the route label == [TOOLS].
        """

        # 3) Use these indices for all per-image vision tensors
        def select_per_image(key):
            if key in forward_dict and forward_dict[key] is not None:
                t = forward_dict[key]
                # Only select if first dim equals total #images across the full (pre-take) batch
                total_imgs = row_ptr[-1].item()
                if t.dim() >= 1 and t.size(0) == total_imgs:
                    forward_dict[key] = t.index_select(0, img_idx)

        if not is_tools.any():
            return torch.tensor(0.0, device=next(self.parameters()).device)

        take = is_tools.nonzero(as_tuple=False).squeeze(-1)
        bs = is_tools.shape[0]

        # Slice eagle prompt (images + task text)
        eagle_input = {k.removeprefix("eagle_"): v for k, v in vl_input.items() if k.startswith("eagle_")}
        eagle_input.pop("image_sizes", None)
        # TODO:
        eagle_llm_labels = eagle_input.pop("llm_labels", None)

        eagle_ids  = eagle_input["input_ids"].index_select(0, take)            # [Bt, Te]
        eagle_mask = eagle_input["attention_mask"].index_select(0, take)       # [Bt, Te]

        # Slice step target text
        step_ids   = vl_input["step_input_ids"].index_select(0, take)          # [Bt, Ts]
        step_mask  = vl_input["step_attention_mask"].index_select(0, take)     # [Bt, Ts]

        # Concatenate prompt ⊕ target (left-padded already; simple cat works)
        concat_ids  = torch.cat([eagle_ids,  step_ids],  dim=1)                 # [Bt, Te+Ts]
        concat_mask = torch.cat([eagle_mask, step_mask], dim=1)                 # [Bt, Te+Ts]

        # Labels: supervise ONLY on the step segment (standard HF CLM shift is internal).
        labels = concat_ids.clone()
        labels[concat_mask == 0] = -100

        # Mask out the entire eagle prompt part per row
        Te = eagle_mask.sum(dim=1)  # [Bt]
        pos = torch.arange(concat_ids.size(1), device=concat_ids.device).unsqueeze(0)  # [1, L]
        labels[pos < Te.unsqueeze(1)] = -100

        # Build a minimal forward dict for the model
        forward_dict = dict(eagle_input)
        forward_dict["input_ids"]       = concat_ids  # [B, S1 + S2]
        forward_dict["attention_mask"]  = concat_mask  # [B, S1 + S2]
        # IMPORTANT: keep the same vision inputs for conditioning
        # (pixel_values / video inputs are already batched in eagle_input)
        if "pixel_values" in forward_dict:

            # 1) Build image index spans for each original batch row
            num_images = forward_dict.pop("num_images")
            num_images = num_images.squeeze(1)
            row_ptr = torch.cat([num_images.new_zeros(1), num_images.cumsum(0)])
            # 2) Gather image indices that belong to the kept rows
            device = concat_ids.device
            starts = row_ptr.index_select(0, take)               # (Bt,)
            ends   = row_ptr.index_select(0, take + 1)           # (Bt,)

            img_idx_list = [torch.arange(s.item(), e.item(), device=device, dtype=torch.long) for s, e in zip(starts, ends)]
            img_idx = torch.cat(img_idx_list, dim=0) if img_idx_list else torch.empty(0, dtype=torch.long, device=device)

            select_per_image("pixel_values")
        if "video_pixel_values" in forward_dict:
            forward_dict["video_pixel_values"] = forward_dict["video_pixel_values"].index_select(0, take)

        out = self.eagle_model(**forward_dict, labels=labels, return_dict=True)
        return out.loss  # scalar
        

    def forward_route(self, vl_input: BatchFeature):
        # vl_input: ['state', 'state_mask', 'segmentation_target', 'segmentation_target_mask', 'has_real_action', 'action', 'action_mask', 'step_input_ids', 'step_attention_mask', 'eagle_input_ids', 'eagle_attention_mask', 'eagle_pixel_values', 'eagle_image_sizes', 'embodiment_id']
        # 1) Run backbone once
        eagle_logits, route_pos, eagle_embeds, eagle_mask = self.forward_eagle(vl_input)

        # 2) Compute route loss & labels (0=[ACTIONS], 1=[TOOLS])
        route_loss, raw_label, route_label = self.calculate_route_loss(vl_input, eagle_logits, route_pos)

        # 3) Split batch
        # if input with [TOOLS], need let model to generate text
        # if input with [ACTIONS], pass eagle embeds to DiT for action predictions
        is_tools   = route_label == 1
        is_actions = route_label == 0

        # 4) LM loss for tool text (teacher forcing on step_input_ids)
        tools_lm_loss = self._tools_lm_loss(vl_input, is_tools)

        out = {
            "route_loss": route_loss,
            "tools_lm_loss": tools_lm_loss,
            # Keep full-batch embeds for the action head (FlowMatching will mask itself)
            "eagle_embeds": eagle_embeds,
            "eagle_mask":   eagle_mask,
            "dit_indices":  is_actions.nonzero(as_tuple=False).squeeze(-1)
                            if is_actions.any() else torch.empty(
                                (0,), dtype=torch.long, device=eagle_embeds.device
                            ),
        }

        # (Optional) Inference-time generation for TOOLS rows
        if not self.training and is_tools.any():
            eagle_input = {k.removeprefix("eagle_"): v for k, v in vl_input.items() if k.startswith("eagle_")}
            eagle_input.pop("image_sizes", None)
            take = is_tools.nonzero(as_tuple=False).squeeze(-1)
            sub = {k: (v.index_select(0, take) if isinstance(v, torch.Tensor) and v.dim() > 0 else v)
                for k, v in eagle_input.items()}
            gen = self.eagle_model.generate(
                **sub, max_new_tokens=64, do_sample=False, return_dict_in_generate=True
            )
            in_len = sub["input_ids"].size(1)
            new_tokens = gen.sequences[:, in_len:]
            texts = self.eagle_tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            out["gen_text"]    = texts
            out["gen_indices"] = take
        else:
            out["gen_text"]    = []
            out["gen_indices"] = torch.empty((0,), dtype=torch.long, device=eagle_embeds.device)

        return out

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        # eagle_logits, eagle_embeds, eagle_mask = self.forward_eagle(vl_input)
        out = self.forward_route(vl_input)
        eagle_embeds = out["eagle_embeds"]
        eagle_mask   = out["eagle_mask"]
        dit_indices  = out["dit_indices"]

        # Filter to ACTION rows only (keep empty batch if none)
        if dit_indices.numel() > 0:
            eagle_embeds = eagle_embeds.index_select(0, dit_indices)
            eagle_mask   = eagle_mask.index_select(0, dit_indices)
        else:
            # Keep correct dtype/device with 0 batch size
            T = eagle_embeds.size(1)
            H = eagle_embeds.size(2)
            device = eagle_embeds.device
            dtype  = eagle_embeds.dtype
            eagle_embeds = torch.empty((0, T, H), device=device, dtype=dtype)
            eagle_mask   = torch.empty((0, T), device=device, dtype=out["eagle_mask"].dtype)


        # YL (TODO HACK): to resolve DDP issue when tune_visual=True
        # Ensure all trainable parameters in vision_model are used in the forward pass for DDP compatibility
        if self.training and self.tune_visual:
            dummy_term = torch.tensor(
                0.0, device=eagle_embeds.device, dtype=eagle_embeds.dtype, requires_grad=True
            )
            for param in self.eagle_model.vision_model.parameters():
                if param.requires_grad:
                    dummy_term = dummy_term + 0.0 * param.sum()
            eagle_embeds = eagle_embeds + dummy_term

        return BatchFeature(
            data={
                # Already filtered to ACTION rows
                "backbone_features":       eagle_embeds,      # [Bd, T, H]
                "backbone_attention_mask": eagle_mask,        # [Bd, T]

                # Mapping + diagnostics
                "dit_indices":             dit_indices,       # indices into original batch
                "route_loss":              out["route_loss"],
                "tools_lm_loss":           out["tools_lm_loss"],
                "gen_text":                out.get("gen_text", []),
                "gen_indices":             out.get("gen_indices"),
                # (Optional) expose original batch size if downstream wants it
                "orig_batch_size":         out["eagle_embeds"].size(0),
            }
        )


    @torch.no_grad()
    # TODO: 
    def get_prediction(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

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

            # Join vision, language, state and action embedding along sequence dimension.
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        return BatchFeature(data={"action_pred": actions})
