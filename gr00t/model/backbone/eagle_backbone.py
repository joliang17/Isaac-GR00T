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


class HybridEmbedding(nn.Module):
    """
    An embedding module that uses a frozen base embedding table and a new,
    trainable special embedding table.
    """
    def __init__(self, 
                base_embedding: nn.Embedding, 
                num_special_tokens_A: int, 
                special_id_lookup_A: torch.Tensor,
                num_special_tokens_B: int,
                special_id_lookup_B: torch.Tensor
            ):
        super().__init__()
        self.embedding_dim = base_embedding.embedding_dim
        
        # Keep a reference to the original, and freeze it
        self.base_embedding = base_embedding
        self.base_embedding.requires_grad_(False)
        
        # MODIFIED: Create two new, trainable embedding layers for special tokens
        self.special_embedding_A = nn.Embedding(num_special_tokens_A, self.embedding_dim)
        self.special_embedding_B = nn.Embedding(num_special_tokens_B, self.embedding_dim)

        # MODIFIED: Register buffers for both lookup tables
        self.register_buffer("special_id_lookup_A", special_id_lookup_A.clone(), persistent=False)
        self.register_buffer("special_id_lookup_B", special_id_lookup_B.clone(), persistent=False)

        self.embedding_dim = self.base_embedding.embedding_dim
        self.num_embeddings = self.special_id_lookup_A.numel() # Total vocab size
        self.padding_idx = self.base_embedding.padding_idx

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # MODIFIED: Updated forward logic for three embedding tables
        
        base_vocab_size = self.base_embedding.num_embeddings
        base_mask = input_ids < base_vocab_size

        # Use a base input, masking all special tokens to 0 (or any valid base token)
        base_input = input_ids.clone().masked_fill(~base_mask, 0)
        embeddings = self.base_embedding(base_input).clone()

        # Find tokens for group A
        special_indices_A = self.special_id_lookup_A[input_ids]
        special_mask_A = special_indices_A >= 0
        
        if special_mask_A.any():
            embeddings[special_mask_A] = self.special_embedding_A(special_indices_A[special_mask_A]).to(dtype=embeddings.dtype)

        # Find tokens for group B
        special_indices_B = self.special_id_lookup_B[input_ids]
        special_mask_B = special_indices_B >= 0

        if special_mask_B.any():
            embeddings[special_mask_B] = self.special_embedding_B(special_indices_B[special_mask_B]).to(dtype=embeddings.dtype)
            
        return embeddings

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """
        Custom loader to handle loading weights from a standard, non-hybrid GR00T checkpoint.
        """
        # Define the key for the flat weight tensor in the original GR00T checkpoint
        flat_weight_key = prefix + 'weight'
        
        # Define the key for the base embedding in our new hybrid structure
        base_embedding_key = prefix + 'base_embedding.weight'
        
        # If the old flat key exists in the checkpoint, remap it to the new key
        if flat_weight_key in state_dict and base_embedding_key not in state_dict:
            # Get the weight tensor from the checkpoint
            flat_weight = state_dict.pop(flat_weight_key)
            
            # Place it back into the state_dict with the new, correct key
            state_dict[base_embedding_key] = flat_weight
            
        # Let the default PyTorch loader handle the rest with the corrected state_dict
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


class HybridLMHead(nn.Module):
    """
    An LM head that uses a frozen base head and a new, trainable special head.
    """
    def __init__(self, 
                 base_head: nn.Linear, 
                 num_special_tokens_A: int, 
                 special_token_ids_A: torch.Tensor, 
                 num_special_tokens_B: int,
                 special_token_ids_B: torch.Tensor,
                 total_vocab_size: int,
                ):
        super().__init__()
        self.in_features = base_head.in_features
        self.out_features = total_vocab_size
        
        # Keep a reference to the original, and freeze it
        self.base_head = base_head
        self.base_head.requires_grad_(False)
        
        # MODIFIED: Create two new, trainable LM heads for special tokens
        self.special_head_A = nn.Linear(self.in_features, num_special_tokens_A, bias=False)
        self.special_head_B = nn.Linear(self.in_features, num_special_tokens_B, bias=False)

        # MODIFIED: Register buffers for both ID tensors
        self.register_buffer("special_token_ids_A", special_token_ids_A.clone(), persistent=False)
        self.register_buffer("special_token_ids_B", special_token_ids_B.clone(), persistent=False)
        self.total_vocab_size = total_vocab_size


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # MODIFIED: Updated forward logic to scatter logits from three heads
        base_logits = self.base_head(hidden_states)
        
        logits_shape = hidden_states.shape[:-1] + (self.total_vocab_size,)
        # Ensure logits tensor is on the same device and dtype as hidden_states
        logits = torch.zeros(logits_shape, dtype=hidden_states.dtype, device=hidden_states.device)

        # Place base logits
        logits[..., : base_logits.size(-1)] = base_logits

        # Place special logits for group A
        if self.special_token_ids_A.numel() > 0:
            special_logits_A = self.special_head_A(hidden_states)
            if special_logits_A.dtype != logits.dtype:
                special_logits_A = special_logits_A.to(logits.dtype)
            logits[..., self.special_token_ids_A] = special_logits_A

        # Place special logits for group B
        if self.special_token_ids_B.numel() > 0:
            special_logits_B = self.special_head_B(hidden_states)
            if special_logits_B.dtype != logits.dtype:
                special_logits_B = special_logits_B.to(logits.dtype)
            logits[..., self.special_token_ids_B] = special_logits_B
        
        return logits

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """
        Custom loader to handle loading weights from a standard, non-hybrid GR00T checkpoint.
        """
        # We need to remap both 'weight' and 'bias' if they exist.
        for suffix in ("weight", "bias"):
            # Define the key for the flat tensor in the original checkpoint (e.g., '...lm_head.weight')
            flat_key = prefix + suffix
            
            # Define the key for the base head in our new hybrid structure (e.g., '...lm_head.base_head.weight')
            nested_key = prefix + f"base_head.{suffix}"
            
            # If the old flat key exists in the checkpoint, remap it to the new key
            if flat_key in state_dict and nested_key not in state_dict:
                # Get the tensor from the checkpoint
                flat_tensor = state_dict.pop(flat_key)
                
                # Place it back into the state_dict with the new, correct key
                state_dict[nested_key] = flat_tensor
        
        # Let the default PyTorch loader handle the rest with the corrected state_dict
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


class EagleBackbone(nn.Module):

    def __init__(
        self,
        tune_llm: bool = False,
        tune_visual: bool = False,
        tune_special_A: bool = True,
        tune_special_B: bool = True,
        tune_tool_end: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = False,
        use_flash_attention: bool = False,
        load_bf16: bool = False,
        eagle_path: str | None = None,
        project_to_dim: int = 1536,
        special_token_loss_weight: float = 2.0,
        pred_nextstep: bool = False,
        tool_end_loss_weight: float = 1.0,
    ):
        """
        Args:
            tune_llm: whether to tune the LLM model (default: True)
            tune_visual: whether to tune the visual model (default: False)
        """
        super().__init__()
        assert not reproject_vision, "Reproject vision is not implemented here, set to False"

        self.pred_nextstep = pred_nextstep
        config = AutoConfig.from_pretrained(DEFAULT_EAGLE_PATH, trust_remote_code=True)
        self.eagle_model = AutoModel.from_config(config, trust_remote_code=True)
        hidden_size = self.eagle_model.language_model.lm_head.in_features

        # ADDED: add special tokens to tokenizer
        self.eagle_processor = AutoProcessor.from_pretrained(
            DEFAULT_EAGLE_PATH, trust_remote_code=True, use_fast=True
        )
        self.eagle_tokenizer = self.eagle_processor.tokenizer

        # MODIFIED: Partition special tokens into group A and B
        list_special = ["[ACTIONS]", "[TOOLS]", "[TOOLS_END]", "[SKILL_MODE]", "[TRAJ_MODE]"]
        # list_special.extend([f"[SKILL_{i}]" for i in range(1, 42)])
        specials = {"additional_special_tokens": list_special}

        list_special_A_names = set(["[ACTIONS]", "[TOOLS_END]", "[SKILL_MODE]"])
        # list_special_B_names = set(["[TOOLS]", "[TRAJ_MODE]"] + [f"[SKILL_{i}]" for i in range(1, 42)])
        list_special_B_names = set(["[TOOLS]", "[TRAJ_MODE]"])
        
        existing = set(self.eagle_tokenizer.all_special_tokens)
        to_add = [t for t in specials["additional_special_tokens"] if t not in existing] 
        # Partition the new tokens into their respective groups
        to_add_A = [t for t in to_add if t in list_special_A_names]
        to_add_B = [t for t in to_add if t in list_special_B_names]
        if to_add:
            num_added = self.eagle_tokenizer.add_special_tokens({"additional_special_tokens": to_add})
            print(f"Added {num_added} new tokens: {to_add}")

        self.eagle_tokenizer.add_special_tokens(specials)
        self.eagle_processor.tokenizer = self.eagle_tokenizer
        total_vocab_size = len(self.eagle_tokenizer)

        base_embeddings = self.eagle_model.get_input_embeddings()
        base_vocab_size = base_embeddings.num_embeddings

         # MODIFIED: Create ID/lookup tensors for both groups
        special_ids_A, special_lookup_A, num_new_A = self.get_ids_and_lookup(to_add_A, base_vocab_size, total_vocab_size)
        special_ids_B, special_lookup_B, num_new_B = self.get_ids_and_lookup(to_add_B, base_vocab_size, total_vocab_size)

        self.register_buffer("special_token_ids_A", special_ids_A, persistent=False)
        self.register_buffer("special_token_ids_B", special_ids_B, persistent=False)
        self.register_buffer("special_token_lookup_A", special_lookup_A, persistent=False)
        self.register_buffer("special_token_lookup_B", special_lookup_B, persistent=False)

        # MODIFIED: Instantiate Hybrid layers if any new tokens were added
        if num_new_A > 0 or num_new_B > 0:

            # 1. Get the original, pre-trained layers
            base_embedding = self.eagle_model.get_input_embeddings()
            base_lm_head = self.eagle_model.get_output_embeddings()

            # 2. Create your new hybrid layers
            hybrid_embedding = HybridEmbedding(base_embedding, num_new_A, special_lookup_A, num_new_B, special_lookup_B)
            hybrid_lm_head = HybridLMHead(base_lm_head, num_new_A, self.special_token_ids_A, num_new_B, self.special_token_ids_B, total_vocab_size)

            # 3. Replace the model's original layers with the new ones
            self.eagle_model.set_input_embeddings(hybrid_embedding)
            self.eagle_model.set_output_embeddings(hybrid_lm_head)

            # 4. Tie the new special head to the new special embedding
            self.tie_special_weights()

        else:
            self.register_buffer("special_token_ids", torch.empty(0, dtype=torch.long), persistent=False)
            
        # Cache special token ids (single-token by construction)
        self.actions_id = self.eagle_tokenizer.convert_tokens_to_ids("[ACTIONS]")
        self.tools_id = self.eagle_tokenizer.convert_tokens_to_ids("[TOOLS]")
        self.skills_end = self.eagle_tokenizer.convert_tokens_to_ids("[TOOLS_END]")
        
        self.pad_id = self.eagle_tokenizer.convert_tokens_to_ids(self.eagle_tokenizer.pad_token)
        self.end_id = self.eagle_tokenizer.convert_tokens_to_ids(self.eagle_tokenizer.eos_token)
        self.assistant_id = self.eagle_tokenizer.convert_tokens_to_ids("assistant")
        # user prefix pattern as persistent buffer
        user_tokens = self.eagle_tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
        self.register_buffer("user_instr_ids", torch.tensor(user_tokens, dtype=torch.long))

        self.img_id = self.eagle_tokenizer.convert_tokens_to_ids("<img>")
        # maybe_image_tokens = ["<IMG_CONTEXT>", "<img>", "</img>", '[PAD_A]', '<|im_end|>', ]
        maybe_image_tokens = ["<IMG_CONTEXT>", "<img>", "</img>", '[PAD_A]', '<|im_start|>', 'assistant', ]
        ignored_id = []
        for t in maybe_image_tokens:
            tid = self.eagle_tokenizer.convert_tokens_to_ids(t)
            if tid is not None and tid != self.eagle_tokenizer.unk_token_id:
                ignored_id.append(tid)
        self.ignored_id = torch.tensor(ignored_id)

        self.special_token_loss_weight = special_token_loss_weight
        self.tool_end_loss_weight = tool_end_loss_weight

        loss_token_ids = [self.actions_id, self.tools_id, self.skills_end]
        loss_token_ids = [tid for tid in dict.fromkeys(loss_token_ids) if tid is not None and tid >= 0]
        self.special_loss_ids = (
            torch.tensor(loss_token_ids, dtype=torch.long)
            if loss_token_ids
            else torch.empty(0, dtype=torch.long)
        )

        if project_to_dim is not None:
            self.eagle_linear = torch.nn.Linear(2048, project_to_dim)
        else:
            self.eagle_linear = torch.nn.Identity()

        # ADDED: Binary classification head for [TOOLS_END] prediction
        self.tool_end_head = nn.Linear(hidden_size, 2)

        # needed since we don't use these layers. Also saves compute
        while len(self.eagle_model.language_model.model.layers) > select_layer:
            self.eagle_model.language_model.model.layers.pop(-1)

        self.select_layer = select_layer
        self.set_trainable_parameters(tune_llm, tune_visual, tune_special_A, tune_special_B, tune_tool_end)

        # Safety for generation
        if self.eagle_tokenizer.pad_token_id is None:
            self.eagle_tokenizer.pad_token = self.eagle_tokenizer.eos_token

    def get_ids_and_lookup(self, tokens_to_add_list, base_vocab_size, total_vocab_size):
        new_special_token_ids = []
        for token in tokens_to_add_list:
            token_id = self.eagle_tokenizer.convert_tokens_to_ids(token)
            # Only consider tokens that are *actually* new (i.e., outside the base vocab)
            if token_id is not None and token_id >= base_vocab_size:
                new_special_token_ids.append(token_id)
        
        if not new_special_token_ids:
            ids_tensor = torch.empty(0, dtype=torch.long)
            # Lookup tensor still needs to cover the full vocab
            lookup_tensor = torch.full((total_vocab_size,), -1, dtype=torch.long)
            return ids_tensor, lookup_tensor, 0
        
        ids_tensor = torch.tensor(sorted(new_special_token_ids), dtype=torch.long)
        lookup_tensor = torch.full((total_vocab_size,), -1, dtype=torch.long)
        for idx, token_id in enumerate(ids_tensor.tolist()):
            lookup_tensor[token_id] = idx
        
        return ids_tensor, lookup_tensor, len(new_special_token_ids)


    def initialize_new_token_weights(self):
        """ Initializes the special embedding parts with the mean of the base weights. """
        # MODIFIED: Initialize both special embedding groups
        hybrid_embedding = self.eagle_model.get_input_embeddings()
        if not isinstance(hybrid_embedding, HybridEmbedding):
            print("Not a HybridEmbedding; skipping special token initialization.")
            return

        base_weights = hybrid_embedding.base_embedding.weight
        
        with torch.no_grad():
            mean_vec = base_weights.detach().to(torch.float32).mean(dim=0)

            if hasattr(hybrid_embedding, 'special_embedding_A') and hybrid_embedding.special_embedding_A.weight.size(0) > 0:
                target_weight_A = hybrid_embedding.special_embedding_A.weight
                mean_vec_A = mean_vec.to(dtype=target_weight_A.dtype, device=target_weight_A.device)
                target_weight_A.copy_(mean_vec_A.repeat(target_weight_A.size(0), 1))
                print("Initialized new special token embeddings (Group A) with the mean vector.")

            if hasattr(hybrid_embedding, 'special_embedding_B') and hybrid_embedding.special_embedding_B.weight.size(0) > 0:
                target_weight_B = hybrid_embedding.special_embedding_B.weight
                mean_vec_B = mean_vec.to(dtype=target_weight_B.dtype, device=target_weight_B.device)
                target_weight_B.copy_(mean_vec_B.repeat(target_weight_B.size(0), 1))
                print("Initialized new special token embeddings (Group B) with the mean vector.")


    def tie_special_weights(self):
        """ Ties the new special LM heads to the new special embeddings. """
        # MODIFIED: Tie both pairs of special weights
        hybrid_embedding = self.eagle_model.get_input_embeddings()
        hybrid_lm_head = self.eagle_model.get_output_embeddings()

        if not isinstance(hybrid_embedding, HybridEmbedding) or not isinstance(hybrid_lm_head, HybridLMHead):
            print("Not a Hybrid model; skipping special weight tying.")
            return

        if hasattr(hybrid_lm_head, 'special_head_A') and hasattr(hybrid_embedding, 'special_embedding_A'):
            hybrid_lm_head.special_head_A.weight = hybrid_embedding.special_embedding_A.weight
            print("Tied special LM head (Group A) to special embeddings (Group A).")
        
        if hasattr(hybrid_lm_head, 'special_head_B') and hasattr(hybrid_embedding, 'special_embedding_B'):
            hybrid_lm_head.special_head_B.weight = hybrid_embedding.special_embedding_B.weight
            print("Tied special LM head (Group B) to special embeddings (Group B).")


    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool, tune_special_A: bool, tune_special_B: bool, tune_tool_end: bool=False):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        self.tune_special_A = tune_special_A
        self.tune_special_B = tune_special_B
        self.tune_tool_end = tune_tool_end

        # Start with all params of this module (EagleBackbone) trainable
        # This includes eagle_linear and the entire eagle_model
        for p in self.parameters():
            p.requires_grad = True
        
        # Control the Tool End Head explicitly
        self.tool_end_head.requires_grad_(tune_tool_end)
        
        if not tune_llm:
            # This freezes the entire language_model, including all embedding layers
            self.eagle_model.language_model.requires_grad_(False)
        
        if not tune_visual:
            self.eagle_model.vision_model.requires_grad_(False)
            self.eagle_model.mlp1.requires_grad_(False)

        # Get hybrid layers
        hybrid_embedding = self.eagle_model.get_input_embeddings()
        hybrid_lm_head = self.eagle_model.get_output_embeddings()
        
        # The base layers are frozen by default in Hybrid* init, so we don't
        # need to re-freeze them if tune_llm=True.

        # If tune_llm=True, the special layers are trainable by default.
        # We add explicit controls to turn them OFF.
        if isinstance(hybrid_embedding, HybridEmbedding):
            if hasattr(hybrid_embedding, 'special_embedding_A'):
                hybrid_embedding.special_embedding_A.requires_grad_(tune_special_A)
            if hasattr(hybrid_embedding, 'special_embedding_B'):
                hybrid_embedding.special_embedding_B.requires_grad_(tune_special_B)
        
        if isinstance(hybrid_lm_head, HybridLMHead):
            # This should be redundant due to weight tying, but it's good to be explicit.
            if hasattr(hybrid_lm_head, 'special_head_A'):
                hybrid_lm_head.special_head_A.requires_grad_(tune_special_A)
            if hasattr(hybrid_lm_head, 'special_head_B'):
                hybrid_lm_head.special_head_B.requires_grad_(tune_special_B)
        
        # If tune_llm=False, all the above layers were already frozen
        # by the self.eagle_model.language_model.requires_grad_(False) call.

        print(f"Tune backbone llm: {self.tune_llm}")
        print(f"Tune backbone visual: {self.tune_visual}")
        print(f"Tune tool_end head: {self.tune_tool_end}")
        
        # MODIFIED: Print the *actual* status of the special layers
        if isinstance(hybrid_embedding, HybridEmbedding):
            status_A = hybrid_embedding.special_embedding_A.weight.requires_grad if hasattr(hybrid_embedding, 'special_embedding_A') and hybrid_embedding.special_embedding_A.weight.size(0) > 0 else 'N/A'
            status_B = hybrid_embedding.special_embedding_B.weight.requires_grad if hasattr(hybrid_embedding, 'special_embedding_B') and hybrid_embedding.special_embedding_B.weight.size(0) > 0 else 'N/A'
            print(f"Tune special tokens A: {status_A}")
            print(f"Tune special tokens B: {status_B}")
        
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
            if k.startswith(eagle_prefix) and k != 'eagle_num_images' and 'length' not in k
        }
        del eagle_input["image_sizes"]

        eagle_input.pop('llm_labels')
        eagle_output = self.eagle_model(**eagle_input, past_key_values=past_key_values, output_hidden_states=True, return_dict=True, use_cache=True)
        past_key_values = eagle_output.past_key_values
        eagle_features = eagle_output.hidden_states[self.select_layer]
        eagle_features = self.eagle_linear(eagle_features)
        eagle_logits = eagle_output.logits

        attn = eagle_input.get("attention_mask")   # [B, T]
        route_pos = attn.sum(dim=1) - 1        # [B]

        # Raw Hidden States for Tool Head (Unprojected)
        # Assuming tool_end_head was trained on the last layer
        raw_hidden_states = eagle_output.hidden_states[-1]

        return eagle_logits, route_pos, eagle_features, eagle_input["attention_mask"], past_key_values, raw_hidden_states

    def _transcript_lm_loss(self, vl_input: BatchFeature) -> torch.Tensor:
        """
        General LM loss over the rolling transcript (teacher forcing).
        Uses:
        - eagle_input_ids / eagle_attention_mask / (pixel_values | video_pixel_values)
        - eagle_labels  (same shape; -100 where we don't supervise, e.g., [PAD_A])
        """
        def find_last_step(input_ids, labels, mask_img):
            B, T = labels.shape
            final_mask = torch.zeros_like(labels, dtype=torch.bool)

            for b in range(B):
                ids = input_ids[b]

                # Positions that mark "steps":
                #  - 'assistant' role tokens
                #  - any <IMG_CONTEXT> tokens (via mask_img)
                # step_markers = ((ids == self.assistant_id) | mask_img[b])
                step_markers = ids == self.assistant_id

                if not step_markers.any():
                    # No assistant/img markers → do nothing extra
                    continue
                
                # Last index that is either 'assistant' or part of <IMG_CONTEXT>
                last_idx = torch.nonzero(step_markers, as_tuple=False)[-1, 0].item()

                # Mask everything AFTER this last step marker
                if last_idx > 0:
                    final_mask[b, :last_idx] = True

            return final_mask

        if "eagle_llm_labels" not in vl_input:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        eagle_input = {
            k.removeprefix("eagle_"): v
            for k, v in vl_input.items()
            if k.startswith("eagle_") and k != "eagle_num_images" and 'length' not in k
        }
        eagle_input.pop("image_sizes", None)
        labels = eagle_input.pop("llm_labels")

        ignored_tensor = torch.isin(labels, self.ignored_id.to(labels.device))
        labels[ignored_tensor] = -100
        
        if self.pred_nextstep:
            # only predict next step
            final_mask = find_last_step(eagle_input['input_ids'], labels, ignored_tensor)
            labels[final_mask] = -100

        # # import pdb;pdb.set_trace()
        # masked_tokens_per_sample = [ids[row == -100] for ids, row in zip(eagle_input['input_ids'], labels)]
        # unmasked_tokens_per_sample = [ids[row != -100] for ids, row in zip(eagle_input['input_ids'], labels)]
        # print(self.eagle_tokenizer.decode(eagle_input['input_ids'][0]))
        # print(self.eagle_tokenizer.decode(masked_tokens_per_sample[0]))
        # print(self.eagle_tokenizer.decode(unmasked_tokens_per_sample[0]))
        # import pdb;pdb.set_trace()

        # We need hidden states if we are tuning the tool head
        need_hidden = self.tune_tool_end
        outputs = self.eagle_model(**eagle_input, return_dict=True, output_hidden_states=need_hidden)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # --- Tool End Loss Logic (Controlled) ---
        tool_loss_avg = torch.tensor(0.0, device=shift_labels.device)
        
        if self.tune_tool_end:
            hidden_states = outputs.hidden_states[-1] 
            shift_hidden = hidden_states[..., :-1, :].contiguous()
            tool_logits = self.tool_end_head(shift_hidden)
            tool_targets = (shift_labels == self.skills_end).long()
            
            valid_mask_tool = shift_labels != -100
            
            if valid_mask_tool.any():
                tool_loss_fct = nn.CrossEntropyLoss(reduction='none')
                loss_per_token = tool_loss_fct(tool_logits.view(-1, 2), tool_targets.view(-1))
                loss_per_token = loss_per_token.view_as(shift_labels)
                tool_loss_avg = loss_per_token[valid_mask_tool].mean()

        vocab_size = shift_logits.size(-1)
        valid_mask = shift_labels != -100
        
        # MODIFIED: Get counts for both special groups
        num_special_A = self.special_token_ids_A.numel()
        num_special_B = self.special_token_ids_B.numel()
        num_special_total = num_special_A + num_special_B
        
        special_loss_A = None
        special_loss_B = None
        base_loss = None

        if num_special_total == 0:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
            per_token_loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
            per_token_loss = per_token_loss.view_as(shift_labels)
        else:
            # MODIFIED: Handle 3 disjoint groups: base, special_A, special_B
            special_ids_A = self.special_token_ids_A.to(shift_labels.device)
            special_ids_B = self.special_token_ids_B.to(shift_labels.device)
            
            special_mask_A = torch.isin(shift_labels, special_ids_A) & valid_mask
            special_mask_B = torch.isin(shift_labels, special_ids_B) & valid_mask
            base_mask = valid_mask & ~special_mask_A & ~special_mask_B
            
            per_token_loss = shift_logits.new_zeros(shift_labels.shape, dtype=shift_logits.dtype)

            # Get base vocab size from the frozen base head
            base_lm_head = self.eagle_model.get_output_embeddings().base_head
            base_vocab_size = base_lm_head.out_features

            if base_mask.any():
                base_logits = shift_logits[..., :base_vocab_size]
                # Filter logits and labels by the base mask
                base_loss = F.cross_entropy(base_logits[base_mask], shift_labels[base_mask], reduction="none")
                per_token_loss[base_mask] = base_loss.to(dtype=per_token_loss.dtype)

            if special_mask_A.any():
                # Select the logits corresponding to group A's vocab positions
                special_logits_A = shift_logits.index_select(-1, special_ids_A)
                # Get the lookup table to map full-vocab-ID -> group-A-index
                lookup_A = self.special_token_lookup_A.to(shift_labels.device)
                # Map the labels to the small index range [0, num_special_A)
                target_positions_A = lookup_A[shift_labels[special_mask_A]].to(dtype=torch.long)
                # Calculate loss
                special_loss_A = F.cross_entropy(special_logits_A[special_mask_A], target_positions_A, reduction="none")
                per_token_loss[special_mask_A] = special_loss_A.to(dtype=per_token_loss.dtype)
                
            if special_mask_B.any():
                # Repeat for group B
                special_logits_B = shift_logits.index_select(-1, special_ids_B)
                lookup_B = self.special_token_lookup_B.to(shift_labels.device)
                target_positions_B = lookup_B[shift_labels[special_mask_B]].to(dtype=torch.long)
                special_loss_B = F.cross_entropy(special_logits_B[special_mask_B], target_positions_B, reduction="none")
                per_token_loss[special_mask_B] = special_loss_B.to(dtype=per_token_loss.dtype)

            #######################
            # DEBUG
            if False:
                pred_ids = base_logits.argmax(dim=-1)
                valid_pred_ids = pred_ids[shift_labels != -100]
                valid_label_ids = shift_labels[shift_labels != -100]
                decoded_texts = self.eagle_tokenizer.batch_decode(valid_pred_ids[valid_label_ids!=self.pad_id], skip_special_tokens=False)
                print(''.join(decoded_texts))
                decoded_texts_label = self.eagle_tokenizer.batch_decode(valid_label_ids[valid_label_ids!=self.pad_id], skip_special_tokens=False)
                print(''.join(decoded_texts_label))

                if special_mask_A.any():
                    pred_sp_ids_A_small = special_logits_A[special_mask_A].argmax(dim=-1)
                    num_special_A = special_ids_A.shape[0]
                    inverse_lookup_A = torch.full((num_special_A,), -1, dtype=torch.long, device=special_ids_A.device)
                    inverse_lookup_A[torch.arange(num_special_A, device=special_ids_A.device)] = special_ids_A
                    
                    pred_sp_ids_A_full = inverse_lookup_A[pred_sp_ids_A_small]
                    label_sp_ids_A = shift_labels[special_mask_A]
                    
                    decoded_pred_A = self.eagle_tokenizer.batch_decode(pred_sp_ids_A_full, skip_special_tokens=False)
                    decoded_label_A = self.eagle_tokenizer.batch_decode(label_sp_ids_A, skip_special_tokens=False)
                    
                    print("\n--- [DEBUG] SPECIAL Tokens (Group A) ---")
                    print(f"Preds:  {''.join(decoded_pred_A)}")
                    print(f"Labels: {''.join(decoded_label_A)}")
                    import pdb;pdb.set_trace()

                if special_mask_B.any():
                    pred_sp_ids_B_small = special_logits_B[special_mask_B].argmax(dim=-1)
                    num_special_B = special_ids_B.shape[0]
                    inverse_lookup_B = torch.full((num_special_B,), -1, dtype=torch.long, device=special_ids_B.device)
                    inverse_lookup_B[torch.arange(num_special_B, device=special_ids_B.device)] = special_ids_B
                    
                    pred_sp_ids_B_full = inverse_lookup_B[pred_sp_ids_B_small]
                    label_sp_ids_B = shift_labels[special_mask_B]
                    decoded_pred_B = self.eagle_tokenizer.batch_decode(pred_sp_ids_B_full, skip_special_tokens=False)
                    decoded_label_B = self.eagle_tokenizer.batch_decode(label_sp_ids_B, skip_special_tokens=False)
                    
                    print("\n--- [DEBUG] SPECIAL Tokens (Group B) ---")
                    print(f"Preds:  {''.join(decoded_pred_B)}")
                    print(f"Labels: {''.join(decoded_label_B)}")

                    import pdb;pdb.set_trace()

        ######################################
        # loss avg per type
        if base_loss is not None:
            base_loss_avg = base_loss.mean()
        else:
            base_loss_avg = torch.tensor(0.0, device=shift_labels.device)
            
        # MODIFIED: Average special loss across both groups
        special_loss_combined = []
        if special_loss_A is not None:
            special_loss_combined.append(special_loss_A)
        if special_loss_B is not None:
            special_loss_combined.append(special_loss_B)

        if special_loss_combined:
            special_loss_avg = torch.cat(special_loss_combined).mean()
        else:
            special_loss_avg = torch.tensor(0.0, device=shift_labels.device)

        # MODIFIED: Only add tool loss if tuning is enabled
        loss = special_loss_avg + base_loss_avg
        if self.tune_tool_end:
            loss = loss + (self.tool_end_loss_weight * tool_loss_avg)
        return loss, base_loss_avg, special_loss_avg
        

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
        img_id      = self.img_id
        actions_id  = self.actions_id
        tools_id    = self.tools_id
        user_pattern = self.user_instr_ids
        K = user_pattern.numel()

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
            if Lb <= 0:
                continue
            first_valid = T - Lb  # left padding: valid tokens start here
            last_valid_idx = T - 1

            # Region we actually search in
            search_start = first_valid
            search_end_exclusive = T  # last index is T-1

            user_starts: list[int] = []
            # We can safely do a Python loop here; T is usually not huge relative to GPU
            max_start_offset = (search_end_exclusive - search_start) - K
            if max_start_offset >= 0:
                for offset in range(max_start_offset + 1):
                    window = ids_b[search_start + offset : search_start + offset + K]
                    if torch.equal(window, user_pattern):
                        user_starts.append(search_start + offset)

            if not user_starts:
                continue
                
            # Iterate over each user block
            for i, u_p in enumerate(user_starts):
                start_pos = u_p

                # End boundary: just before the next user-start, or last valid token
                if i + 1 < len(user_starts):
                    end_boundary = user_starts[i + 1] - 1
                else:
                    end_boundary = last_valid_idx

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
                # self.eagle_tokenizer.decode(window_ids)
                if act_first is None:
                    continue
                if tol_first is not None and tol_first < act_first:
                    continue
                
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
        eagle_logits, route_pos, eagle_embeds, eagle_mask, past_key_values, _ = self.forward_eagle(vl_input, past_key_values=past_key_values)

        # if there is no step information (single instruction input)
        input_keys = vl_input.keys()
        step_input = [item for item in input_keys if 'step' in item]
        #########################
        # normal input
        # directly return logits and embd
        transcript_lm_loss = torch.tensor(0.0, device=eagle_logits.device)
        base_loss_avg = torch.tensor(0.0, device=eagle_logits.device)
        special_loss_avg = torch.tensor(0.0, device=eagle_logits.device)
        embeds_tensor, masks_tensor = None, None

        if len(step_input) != 0:
            # 2) extract action token hidden states based on action_pad_ids
            has_actions = (vl_input['eagle_input_ids'] == self.actions_id).any().item()
            if has_actions:
                list_eagle_emb, list_eagle_mask, seg_batch, seg_start, seg_end  = self.split_by_img_id(vl_input, eagle_embeds, eagle_mask)
                # self.eagle_tokenizer.decode(vl_input['eagle_input_ids'][0])
                # self.eagle_tokenizer.decode(vl_input['eagle_input_ids'][0][11:606])
                embeds_tensor, masks_tensor = self.flatten_actions(list_eagle_emb, list_eagle_mask)

            if not (has_actions and 'action' not in vl_input):
                # 3) Compute generated loss
                transcript_lm_loss, base_loss_avg, special_loss_avg = self._transcript_lm_loss(vl_input)

        out = {
            "transcript_lm_loss": transcript_lm_loss,
            "text_token_loss": base_loss_avg, 
            "special_token_loss": special_loss_avg, 
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
                "text_token_loss":         out["text_token_loss"],
                "special_token_loss":      out["special_token_loss"],
                "orig_batch_size":         out["eagle_embeds"].size(0),
                "past_key_values":         past_key_values
            }
        )


    @torch.no_grad()
    def generate(self, vl_input: BatchFeature, max_token: int = 1, past_key_values=None, special_token_only=False, inside_tool=False, toolend_head=False):
        """Greedy token generation for the language backbone."""
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
        
        # MODIFIED: Removed unused num_special_all and special_ids_all variables

        for _ in range(max_token):
            logits, route_pos, _, _, cache, raw_hidden_states = self.forward_eagle(
                vl_input, past_key_values=cache
            )
            vocab_size = logits.size(-1)
            # MODIFIED: Get base_vocab_size reliably from the model
            base_vocab_size = self.eagle_model.get_output_embeddings().base_head.out_features

            if logits.size(0) != batch_size:
                raise RuntimeError("Batch size changed during generation")

            if route_pos.dtype != torch.long:
                route_pos = route_pos.to(torch.long)
            route_pos = route_pos.clamp(min=0)

            batch_indices = torch.arange(batch_size, device=device)
            next_token_logits = logits[batch_indices, route_pos, :]

            # MODIFIED: Restructured this entire logic block
            if special_token_only:
                # CASE 1: Inside Tool AND Tuning the Head -> Use Classifier
                if inside_tool and toolend_head:
                    # 1. Extract the hidden state at the exact routing position
                    # raw_hidden_states is [B, T, H]
                    current_hidden = raw_hidden_states[batch_indices, route_pos, :] # [B, H]
                    
                    # 2. Pass through the binary classifier
                    tool_logits = self.tool_end_head(current_hidden) # [B, 2]
                    
                    # 3. Predict: 0 = Keep Going ([ACTIONS]), 1 = End ([TOOLS_END])
                    tool_preds = tool_logits.argmax(dim=-1) # [B]
                    
                    # 4. Force the token ID based on prediction
                    # If pred is 1 -> SKILLS_END, else -> ACTIONS
                    next_token_raw = torch.where(
                        tool_preds == 1,
                        torch.tensor(self.skills_end, device=device),
                        torch.tensor(self.actions_id, device=device)
                    )
                # CASE 2: Text-based Fallback (Original Model behavior)
                else:
                    # 1. Define the set of allowed tokens
                    if not inside_tool:
                        allowed_ids = torch.tensor(
                            [self.tools_id, self.actions_id, self.skills_end], 
                            device=input_ids.device, dtype=torch.long
                        )
                    else:
                        allowed_ids = torch.tensor(
                            [self.actions_id, self.skills_end], 
                            device=input_ids.device, dtype=torch.long
                        )
                        # # Check the logits for the first item in the batch
                        # for token_id in allowed_ids:
                        #     # Ensure token_id is a simple integer for indexing
                        #     if isinstance(token_id, torch.Tensor):
                        #         token_id = token_id.item()
                                
                        #     print(f"Original logit for ID {token_id}: {next_token_logits[0, token_id].item()}")
                        # import pdb;pdb.set_trace()
                    
                    # 2. Create a mask that allows *only* these tokens
                    neg_inf = torch.finfo(next_token_logits.dtype).min
                    mask = torch.full_like(next_token_logits, neg_inf)
                    
                    # 3. Apply the mask
                    mask.scatter_(dim=-1, index=allowed_ids.unsqueeze(0).expand(next_token_logits.size(0), -1), value=0)
                    masked_logits = next_token_logits + mask
                    # 4. Get the next token from the *masked* full-vocabulary logits
                    temperature = getattr(self.args, "route_temperature", 0.0) if hasattr(self, "args") else 0.0
                    if temperature and temperature > 0.0:
                        probs = torch.softmax(masked_logits / temperature, dim=-1)
                        next_token_raw = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    else:
                        next_token_raw = masked_logits.argmax(dim=-1)
                    # import pdb;pdb.set_trace()

            else:
                # Normal generation: select the most likely token from the *entire* vocabulary.
                next_token_raw = next_token_logits.argmax(dim=-1)

            prev_finished = finished.clone()
            recorded_token = torch.where(
                prev_finished,
                torch.full_like(next_token_raw, self.end_id),
                next_token_raw,
            )
            generated_tokens.append(recorded_token)

            if max_token == 1:
                # only predict special tokens
                # add additional eos token (self.end_id) at the end
                if next_token_raw.item() in (self.skills_end, self.actions_id):
                    tokens_to_append = torch.tensor([next_token_raw.item(), self.end_id], device=next_token_raw.device).unsqueeze(0)
                else:
                    tokens_to_append = torch.tensor([next_token_raw.item(), ], device=next_token_raw.device).unsqueeze(0)

                mask_to_append = torch.ones_like(tokens_to_append, dtype=attention_mask.dtype)
                input_ids = torch.cat([input_ids, tokens_to_append], dim=1)
                attention_mask = torch.cat([attention_mask, mask_to_append], dim=1)
                vl_input["eagle_input_ids"] = input_ids
                vl_input["eagle_attention_mask"] = attention_mask
                # self.eagle_tokenizer.decode(input_ids[0])

                break

            finished = prev_finished | (next_token_raw == self.end_id)

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
            
            if finished.all():
                break
        
        # Cleanup return
        backbone_outputs = self.forward(vl_input, past_key_values=cache)
        backbone_outputs["past_key_values"] = cache

        if not generated_tokens:
            generated_ids = torch.empty((batch_size, 0), dtype=torch.long, device=device)
        else:
            generated_ids = torch.stack(generated_tokens, dim=1)
            if generated_ids.size(1) == 1:
                generated_ids = generated_ids.squeeze(1)

        return generated_ids, backbone_outputs