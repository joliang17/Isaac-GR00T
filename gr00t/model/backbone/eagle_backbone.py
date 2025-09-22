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
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
import re

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
        self.img_id = self.eagle_tokenizer.convert_tokens_to_ids("<img>")
        maybe_image_tokens = ["<IMG_CONTEXT>", "<img>", "</img>", ]  # include what your tokenizer actually uses
        img_ids = []
        for t in maybe_image_tokens:
            tid = self.eagle_tokenizer.convert_tokens_to_ids(t)
            if tid is not None and tid != self.eagle_tokenizer.unk_token_id:
                img_ids.append(tid)
        self.img_ids = torch.tensor(img_ids)

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

    def forward_eagle(self, vl_input: BatchFeature, past_key_values=None) -> BatchFeature:
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
        eagle_output = self.eagle_model(**eagle_input, past_key_values=past_key_values, output_hidden_states=True, return_dict=True, use_cache=True)
        past_key_values = eagle_output.past_key_values

        eagle_features = eagle_output.hidden_states[self.select_layer]
        eagle_features = self.eagle_linear(eagle_features)
        eagle_logits = eagle_output.logits

        attn = eagle_input.get("attention_mask")   # [B, T]
        route_pos = attn.sum(dim=1) - 1        # [B]

        return eagle_logits, route_pos, eagle_features, eagle_input["attention_mask"], past_key_values

    def _transcript_lm_loss(self, vl_input: BatchFeature) -> torch.Tensor:
        """
        General LM loss over the rolling transcript (teacher forcing).
        Uses:
        - eagle_input_ids / eagle_attention_mask / (pixel_values | video_pixel_values)
        - eagle_labels  (same shape; -100 where we don't supervise, e.g., [PAD_A])
        """
        if "eagle_llm_labels" not in vl_input:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        eagle_input = {
            k.removeprefix("eagle_"): v
            for k, v in vl_input.items()
            if k.startswith("eagle_") and k != "eagle_num_images"
        }
        eagle_input.pop("image_sizes", None)  # if present
        labels = eagle_input.pop("llm_labels")

        # do not calculate loss on image tokens & action pad tokens
        mask_img = torch.isin(labels, self.img_ids.to(labels.device))
        labels[mask_img] = -100

        # FOR DEBUG:
        # unique_ids = torch.unique(eagle_input['input_ids'][torch.where(labels == -100)])
        # self.eagle_tokenizer.decode(unique_ids)

        out = self.eagle_model(**eagle_input, labels=labels, return_dict=True,)
        return out.loss
        

    def split_by_img_id(self, vl_input, eagle_logits: torch.Tensor, eagle_mask: torch.Tensor):
        """
        Split sequences into per-image segments using the position-1 of the `<img>` token
        as separators. Only keep segments whose next route token is `[ACTIONS]` (skip `[TOOLS]`).

        For each batch row, this function:
        - Finds all occurrences of `<img>` in `input_ids`.
        - Defines segments starting at each `<img>` and ending just before the next `<img>`.
        - Within each candidate segment, finds the first of `[ACTIONS]` or `[TOOLS]` after the `<img>`.
          Keeps the segment only if `[ACTIONS]` occurs first, and trims the segment to end before
          the following `[PAD_A]` token (if found).

        Returns lists of segments and masks as slices of `eagle_logits`/`eagle_mask`, plus index
        tensors describing (batch, start, end) for each segment.
        """
        eagle_input = {
            k.removeprefix("eagle_"): v
            for k, v in vl_input.items()
            if k.startswith("eagle_") and k != "eagle_num_images"
        }

        input_ids = eagle_input["input_ids"]            # [B, T]
        attn_mask = eagle_input["attention_mask"]       # [B, T]

        B, T = input_ids.shape
        device = input_ids.device

        # Token ids
        img_id      = self.eagle_tokenizer.convert_tokens_to_ids("<img>")
        actions_id  = self.actions_id
        tools_id    = self.tools_id
        pad_a_id    = self.pad_action

        segments: list[torch.Tensor] = []
        segments_mask: list[torch.Tensor] = []
        seg_batch: list[int] = []
        seg_starts: list[int] = []
        seg_ends: list[int] = []

        # Compute valid lengths from attention mask (left-padded sequences)
        valid_len = attn_mask.sum(dim=1).to(torch.long)  # [B]

        for b in range(B):
            ids_b = input_ids[b]
            mask_b = eagle_mask[b]
            Lb = valid_len[b].item()
            first_valid = T - Lb  # left padding: valid tokens start here
            # Positions of <img> among valid tokens; convert to absolute positions
            img_pos = (ids_b[first_valid:] == img_id).nonzero(as_tuple=False).squeeze(-1)
            if img_pos.numel() > 0:
                img_pos = img_pos + first_valid
            if img_pos.numel() == 0:
                continue

            for i, img_p in enumerate(img_pos.tolist()):
                # Start at position-1 of <img>, clamped to first_valid
                start_pos = max(first_valid, img_p - 5)
                # Segment end boundary: just before the next <img>, or up to last valid index
                if i + 1 < img_pos.numel():
                    end_boundary = img_pos[i + 1].item() - 6
                else:
                    end_boundary = T - 1

                if end_boundary < start_pos:
                    continue  # empty

                # Within [start_pos, end_boundary], find the first route token: [ACTIONS] or [TOOLS]
                window_ids = ids_b[start_pos : end_boundary + 1]
                # Find first occurrence indices relative to the window
                act_rel = (window_ids == actions_id).nonzero(as_tuple=False)
                tol_rel = (window_ids == tools_id).nonzero(as_tuple=False)

                # Helper to get scalar position or None
                def first_pos(rel_idx):
                    return rel_idx[0, 0].item() if rel_idx.numel() > 0 else None

                act_first = first_pos(act_rel)
                tol_first = first_pos(tol_rel)

                # Keep only if [ACTIONS] appears before [TOOLS] (or [TOOLS] absent)
                if act_first is None:
                    continue
                if tol_first is not None and tol_first < act_first:
                    continue

                # Optional: trim the end to the first [PAD_A] after [ACTIONS]
                pad_rel = (window_ids[act_first:] == pad_a_id).nonzero(as_tuple=False)
                if pad_rel.numel() > 0:
                    # Absolute index of PAD_A; use half-open slice [start:end) (exclude PAD_A)
                    end_exclusive = start_pos + act_first + pad_rel[0, 0].item()
                else:
                    # If no PAD_A found before next image, keep full boundary (inclusive -> exclusive)
                    end_exclusive = end_boundary + 1

                if end_exclusive <= start_pos:
                    continue

                seg = eagle_logits[b, start_pos:end_exclusive, :]
                msk = mask_b[start_pos:end_exclusive]

                segments.append(seg)
                segments_mask.append(msk)
                seg_batch.append(b)
                seg_starts.append(start_pos)
                seg_ends.append(end_exclusive)

        if len(segments) == 0:
            return (
                [],
                [],
                torch.empty((0,), dtype=torch.long, device=device),
                torch.empty((0,), dtype=torch.long, device=device),
                torch.empty((0,), dtype=torch.long, device=device),
            )

        return (
            segments,
            segments_mask,
            torch.tensor(seg_batch, dtype=torch.long, device=device),
            torch.tensor(seg_starts, dtype=torch.long, device=device),
            torch.tensor(seg_ends, dtype=torch.long, device=device),
        )


    def flatten_actions(self, list_embeds, list_masks):
        if len(list_embeds) > 0:
            # Pad to max length across segments; zero padding for embeddings, 0/False for masks
            embeds_tensor = pad_sequence(list_embeds, batch_first=True)  # [A, Lmax, H]
            masks_tensor = pad_sequence(list_masks, batch_first=True)  # [A, Lmax]
        else:
            embeds_tensor = None
            masks_tensor = None
        return embeds_tensor, masks_tensor


    def forward_route(self, vl_input: BatchFeature, past_key_values=None):
        # vl_input: ['state', 'state_mask', 'segmentation_target', 'segmentation_target_mask', 'has_real_action', 'action', 'action_mask', 'step_input_ids', 'step_attention_mask', 'eagle_input_ids', 'eagle_attention_mask', 'eagle_pixel_values', 'eagle_image_sizes', 'embodiment_id']
        # 1) Run backbone once
        eagle_logits, route_pos, eagle_embeds, eagle_mask, past_key_values = self.forward_eagle(vl_input, past_key_values=past_key_values)
        
        # if there is no step information (single instruction input)
        input_keys = vl_input.keys()
        step_input = [item for item in input_keys if 'step' in item]
        if len(step_input) == 0:
            #########################
            # normal input
            # directly return logits and embd
            transcript_lm_loss = torch.tensor(0.0, device=eagle_logits.device)

            out = {
                "transcript_lm_loss": transcript_lm_loss, 
                "eagle_embeds": eagle_embeds,
                "eagle_mask":   eagle_mask,
                "eagle_embeds_multi": None,
                "eagle_mask_multi": None,
                "past_key_values": past_key_values
            }

        else:
            #########################
            # 2) extract action token hidden states based on action_pad_ids
            list_eagle_emb, list_eagle_mask, seg_batch, seg_start, seg_end  = self.split_by_img_id(vl_input, eagle_embeds, eagle_mask)
            embeds_tensor, masks_tensor = self.flatten_actions(list_eagle_emb, list_eagle_mask)

            # 3) Compute generated loss
            transcript_lm_loss = self._transcript_lm_loss(vl_input)

            out = {
                "transcript_lm_loss": transcript_lm_loss,
                "eagle_embeds": eagle_embeds,
                "eagle_mask":   eagle_mask,
                "eagle_embeds_multi": embeds_tensor,
                "eagle_mask_multi": masks_tensor,
                "past_key_values": past_key_values
            }

        return out

    def forward(self, vl_input: BatchFeature, past_key_values=None) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        out = self.forward_route(vl_input, past_key_values=past_key_values)
        eagle_embeds = out["eagle_embeds"]
        eagle_mask   = out["eagle_mask"]
        eagle_embeds_multi   = out["eagle_embeds_multi"]
        eagle_mask_multi     = out["eagle_mask_multi"]

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
                "backbone_features_multi":       eagle_embeds_multi,       # [A, Lmax, H]
                "backbone_attention_mask_multi": eagle_mask_multi, # [A, Lmax]

                "transcript_lm_loss":      out["transcript_lm_loss"],
                # (Optional) expose original batch size if downstream wants it
                "orig_batch_size":         out["eagle_embeds"].size(0),
                "past_key_values": past_key_values
            }
        )

    def generate(self, vl_input: BatchFeature, max_token: int = 1, past_key_values=None):
        """Greedy token generation for the language backbone.

        Args:
            vl_input: BatchFeature holding the model inputs. The tensors under
                `eagle_input_ids` and `eagle_attention_mask` are updated in-place
                with the generated tokens so the caller can reuse the batch.
            max_token: Maximum number of tokens to generate. When greater than 1,
                generation stops early once all sequences emit `[EOT]`.
            past_key_values: Optional cache passed through to the underlying
                language model. The cache is recomputed before returning so the
                caller receives a fresh set for the updated sequence.

        Returns:
            A tuple `(generated_ids, backbone_outputs)` where `generated_ids` is a
            tensor of shape `[B]` when a single token is produced or `[B, T]` for
            multiple tokens, and `backbone_outputs` matches the structure returned
            by `forward()` with features computed on the updated prompt.
        """
        if max_token is None or max_token < 1:
            max_token = 1

        if not isinstance(vl_input, BatchFeature):
            vl_input = BatchFeature(data=dict(vl_input))

        if "eagle_input_ids" not in vl_input or "eagle_attention_mask" not in vl_input:
            raise KeyError("Expected `eagle_input_ids` and `eagle_attention_mask` in vl_input")

        self.set_frozen_modules_to_eval_mode()

        input_ids = vl_input["eagle_input_ids"]
        attention_mask = vl_input["eagle_attention_mask"]

        if not isinstance(input_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
            raise TypeError("`eagle_input_ids` and `eagle_attention_mask` must be tensors")

        batch_size = input_ids.size(0)
        device = input_ids.device

        generated_tokens: list[torch.Tensor] = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        cache = past_key_values

        with torch.no_grad():
            for _ in range(max_token):
                logits, route_pos, _, _, cache = self.forward_eagle(
                    vl_input, past_key_values=cache
                )

                if logits.size(0) != batch_size:
                    raise RuntimeError("Batch size changed during generation")

                if route_pos.dtype != torch.long:
                    route_pos = route_pos.to(torch.long)
                route_pos = route_pos.clamp(min=0)

                batch_indices = torch.arange(batch_size, device=device)
                next_token_logits = logits[batch_indices, route_pos, :]
                next_token_raw = next_token_logits.argmax(dim=-1)

                prev_finished = finished.clone()
                recorded_token = torch.where(
                    prev_finished,
                    torch.full_like(next_token_raw, self.endtools_id),
                    next_token_raw,
                )
                generated_tokens.append(recorded_token)

                finished = prev_finished | (next_token_raw == self.endtools_id)

                tokens_to_append = torch.where(
                    prev_finished.unsqueeze(1),
                    torch.full_like(next_token_raw.unsqueeze(1), self.pad_id),
                    next_token_raw.unsqueeze(1),
                )
                mask_to_append = torch.where(
                    prev_finished.unsqueeze(1),
                    torch.zeros_like(tokens_to_append, dtype=attention_mask.dtype),
                    torch.ones_like(tokens_to_append, dtype=attention_mask.dtype),
                )

                input_ids = torch.cat([input_ids, tokens_to_append], dim=1)
                attention_mask = torch.cat([attention_mask, mask_to_append], dim=1)

                vl_input["eagle_input_ids"] = input_ids
                vl_input["eagle_attention_mask"] = attention_mask

                if max_token == 1 or finished.all():
                    break

        # Recompute backbone features so downstream heads consume the updated prompt.
        with torch.no_grad():
            backbone_outputs = self.forward(vl_input, past_key_values=cache)

        backbone_outputs["past_key_values"] = cache

        if not generated_tokens:
            generated_ids = torch.empty((batch_size, 0), dtype=torch.long, device=device)
        else:
            generated_ids = torch.stack(generated_tokens, dim=1)
            if generated_ids.size(1) == 1:
                generated_ids = generated_ids.squeeze(1)

        return generated_ids, backbone_outputs
