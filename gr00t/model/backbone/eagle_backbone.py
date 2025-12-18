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
import pickle
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
import re

import gr00t

DEFAULT_EAGLE_PATH = os.path.join(os.path.dirname(gr00t.__file__), "model", "backbone", "eagle2_hg_model")


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

    def __init__(self, base_embedding: nn.Embedding, num_special_tokens_A: int, special_id_lookup_A: torch.Tensor,
                 num_special_tokens_B: int, special_id_lookup_B: torch.Tensor):
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
        self.num_embeddings = self.special_id_lookup_A.numel()  # Total vocab size
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
            embeddings[special_mask_A] = self.special_embedding_A(special_indices_A[special_mask_A]).to(
                dtype=embeddings.dtype)

        # Find tokens for group B
        special_indices_B = self.special_id_lookup_B[input_ids]
        special_mask_B = special_indices_B >= 0

        if special_mask_B.any():
            embeddings[special_mask_B] = self.special_embedding_B(special_indices_B[special_mask_B]).to(
                dtype=embeddings.dtype)

        return embeddings

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs, ):
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
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs)


class HybridLMHead(nn.Module):
    """
    An LM head that uses a frozen base head and a new, trainable special head.
    """

    def __init__(self, base_head: nn.Linear, num_special_tokens_A: int, special_token_ids_A: torch.Tensor,
                 num_special_tokens_B: int, special_token_ids_B: torch.Tensor, total_vocab_size: int, ):
        super().__init__()
        self.in_features = base_head.in_features
        self.out_features = total_vocab_size

        # Keep a reference to the original, and freeze it
        self.base_head = base_head
        self.base_head.requires_grad_(False)

        # MODIFIED: Create two new, trainable LM heads for special tokens
        self.special_head_A = nn.Linear(self.in_features, num_special_tokens_A, bias=False)
        self.special_head_B = nn.Linear(self.in_features, num_special_tokens_B, bias=False)
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

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs, ):
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
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs)


def flatten_actions(list_embeds, list_masks):
    if len(list_embeds) > 0:
        # Pad to max length across segments; zero padding for embeddings, 0/False for masks
        embeds_tensor = pad_sequence(list_embeds, batch_first=True)  # [A, Lmax, H]
        masks_tensor = pad_sequence(list_masks, batch_first=True)  # [A, Lmax]
    else:
        embeds_tensor = None
        masks_tensor = None
    return embeds_tensor, masks_tensor


class EagleBackbone(nn.Module):

    def __init__(self, tune_llm: bool = False, tune_visual: bool = False, tune_special_A: bool = True,
            tune_special_B: bool = True, tune_tool_end: bool = False, select_layer: int = -1,
            reproject_vision: bool = False, use_flash_attention: bool = False, load_bf16: bool = False,
            eagle_path: str | None = None, project_to_dim: int = 1536, special_token_loss_weight: float = 2.0,
            pred_nextstep: bool = False, tool_end_loss_weight: float = 2.0, ):
        """
        Args:
            tune_llm: whether to tune the LLM model (default: True)
            tune_visual: whether to tune the visual model (default: False)
        """
        super().__init__()
        assert not reproject_vision, "Reproject vision is not implemented here, set to False"

        self.pred_nextstep = pred_nextstep
        # Load the base Eagle configuration and model
        config = AutoConfig.from_pretrained(DEFAULT_EAGLE_PATH, trust_remote_code=True)
        self.eagle_model = AutoModel.from_config(config, trust_remote_code=True)
        hidden_size = self.eagle_model.language_model.lm_head.in_features

        #########################################
        # Add special tokens to tokenizer
        self.eagle_processor = AutoProcessor.from_pretrained(DEFAULT_EAGLE_PATH, trust_remote_code=True, use_fast=True)
        self.eagle_tokenizer = self.eagle_processor.tokenizer

        # Group A/B split allows for differential training strategies (e.g., tuning 'Skill' tokens vs 'Action' tokens differently)
        list_special = ["[ACTIONS]", "[TOOLS]", "[TOOLS_END]", "[SKILL_MODE]", "[TRAJ_MODE]"]
        specials = {"additional_special_tokens": list_special}
        list_special_A_names = {"[ACTIONS]", "[TOOLS_END]", "[SKILL_MODE]"}
        list_special_B_names = {"[TOOLS]", "[TRAJ_MODE]"}

        # Check which tokens are actually new to avoid duplication
        existing = set(self.eagle_tokenizer.all_special_tokens)
        to_add = [t for t in specials["additional_special_tokens"] if t not in existing]
        to_add_A = [t for t in to_add if t in list_special_A_names]
        to_add_B = [t for t in to_add if t in list_special_B_names]

        if to_add:
            num_added = self.eagle_tokenizer.add_special_tokens({"additional_special_tokens": to_add})
            print(f"Added {num_added} new tokens: {to_add}")

        self.eagle_tokenizer.add_special_tokens(specials)
        self.eagle_processor.tokenizer = self.eagle_tokenizer
        total_vocab_size = len(self.eagle_tokenizer)

        #########################################
        # Get original embedding stats to determine boundaries for the new custom embeddings
        base_embeddings = self.eagle_model.get_input_embeddings()
        base_vocab_size = base_embeddings.num_embeddings

        # MODIFIED: Create ID/lookup tensors for both groups for embedding initialization and debugging later
        # These lookups map the new virtual token IDs back to indices in the separate trainable embedding tables
        special_ids_A, special_lookup_A, num_new_A = self.get_ids_and_lookup(to_add_A, base_vocab_size,
                                                                             total_vocab_size)
        special_ids_B, special_lookup_B, num_new_B = self.get_ids_and_lookup(to_add_B, base_vocab_size,
                                                                             total_vocab_size)

        self.register_buffer("special_token_ids_A", special_ids_A, persistent=False)
        self.register_buffer("special_token_ids_B", special_ids_B, persistent=False)
        self.register_buffer("special_token_lookup_A", special_lookup_A, persistent=False)
        self.register_buffer("special_token_lookup_B", special_lookup_B, persistent=False)

        #########################################
        # Instantiate Hybrid embeddings / heads if any new tokens were added
        if num_new_A > 0 or num_new_B > 0:

            base_embedding = self.eagle_model.get_input_embeddings()
            base_lm_head = self.eagle_model.get_output_embeddings()

            hybrid_embedding = HybridEmbedding(base_embedding, num_new_A, special_lookup_A, num_new_B, special_lookup_B)
            hybrid_lm_head = HybridLMHead(base_lm_head, num_new_A, self.special_token_ids_A, num_new_B,
                                          self.special_token_ids_B, total_vocab_size)

            self.eagle_model.set_input_embeddings(hybrid_embedding)
            self.eagle_model.set_output_embeddings(hybrid_lm_head)

            self.tie_special_weights()

        #########################################
        # Cache special token ids (single-token by construction) for quick access during forward passes
        self.actions_id = self.eagle_tokenizer.convert_tokens_to_ids("[ACTIONS]")
        self.tools_id = self.eagle_tokenizer.convert_tokens_to_ids("[TOOLS]")
        self.skills_end = self.eagle_tokenizer.convert_tokens_to_ids("[TOOLS_END]")
        self.pad_id = self.eagle_tokenizer.convert_tokens_to_ids(self.eagle_tokenizer.pad_token)
        self.end_id = self.eagle_tokenizer.convert_tokens_to_ids(self.eagle_tokenizer.eos_token)
        self.assistant_id = self.eagle_tokenizer.convert_tokens_to_ids("assistant")

        # User prefix pattern as persistent buffer (standard chat template preamble)
        user_tokens = self.eagle_tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
        self.register_buffer("user_instr_ids", torch.tensor(user_tokens, dtype=torch.long))

        # Define tokens to ignore during loss calculation (padding, structural tags)
        self.img_id = self.eagle_tokenizer.convert_tokens_to_ids("<img>")
        ignored_tokens = ["<IMG_CONTEXT>", "<img>", "</img>", '[PAD_A]', '<|im_start|>', 'assistant', ]
        ignored_id = []
        for t in ignored_tokens:
            tid = self.eagle_tokenizer.convert_tokens_to_ids(t)
            if tid is not None and tid != self.eagle_tokenizer.unk_token_id:
                ignored_id.append(tid)
        self.ignored_id = torch.tensor(ignored_id)

        #########################################
        # Loss weights for re-balancing rare events
        self.special_token_loss_weight = special_token_loss_weight
        self.tool_end_loss_weight = tool_end_loss_weight

        # Identify IDs that require special loss weighting
        loss_token_ids = [self.actions_id, self.tools_id, self.skills_end]
        loss_token_ids = [tid for tid in dict.fromkeys(loss_token_ids) if tid is not None and tid >= 0]
        self.special_loss_ids = (
            torch.tensor(loss_token_ids, dtype=torch.long) if loss_token_ids else torch.empty(0, dtype=torch.long))

        #########################################
        # Projection layer to align VLM hidden states if dimensions differ
        if project_to_dim is not None:
            self.eagle_linear = torch.nn.Linear(2048, project_to_dim)
        else:
            self.eagle_linear = torch.nn.Identity()

        # Binary classification head for [TOOLS_END] / [TOOLS] prediction
        # These auxiliary heads allow the model to predict structural boundaries without generating full tokens
        self.tool_end_head = torch.nn.Linear(hidden_size, 2)
        self.tool_head = torch.nn.Linear(hidden_size, 2)

        # Layer Pruning: Pop layers from the end to reduce model depth
        # Used if we only need intermediate features or a lighter-weight model (Early Exit)
        while len(self.eagle_model.language_model.model.layers) > select_layer:
            self.eagle_model.language_model.model.layers.pop(-1)

        self.select_layer = select_layer

        #########################################
        # Freeze/Unfreeze specific components based on configuration
        self.set_trainable_parameters(tune_llm, tune_visual, tune_special_A, tune_special_B, tune_tool_end)

        # Safety for generation
        if self.eagle_tokenizer.pad_token_id is None:
            self.eagle_tokenizer.pad_token = self.eagle_tokenizer.eos_token

    def get_ids_and_lookup(self, tokens_to_add_list, base_vocab_size, total_vocab_size):
        """
        Creates mapping tensors to route specific tokens to the 'Hybrid' embedding layer.

        In the Hybrid setup, new special tokens (like [ACTIONS]) are not added to the
        frozen base model's embedding table. Instead, they live in a separate, small
        trainable table. This function generates the lookup table needed to check
        if a token ID belongs to this separate group during the forward pass.

        Args:
            tokens_to_add_list (list): List of token strings (e.g., ["[ACTIONS]"]).
            base_vocab_size (int): The vocabulary size of the pre-trained base model.
            total_vocab_size (int): The current tokenizer size (Base + New tokens).

        Returns:
            tuple:
                - ids_tensor (torch.LongTensor): Global IDs of the new tokens.
                - lookup_tensor (torch.LongTensor): A map of shape [total_vocab_size].
                  If lookup[id] >= 0, 'id' is a new token, and the value is its index
                  in the separate embedding table.
                  If lookup[id] == -1, 'id' is a standard base token.
                - count (int): Number of new tokens found.
        """
        new_special_token_ids = []
        for token in tokens_to_add_list:
            token_id = self.eagle_tokenizer.convert_tokens_to_ids(token)

            # Logic: Only treat a token as "new/special" if its ID exceeds the base vocabulary.
            # If a token like "assistant" was in the list but already exists in Llama/Qwen,
            # we want to use the pre-trained embedding, not a new random initialization.
            if token_id is not None and token_id >= base_vocab_size:
                new_special_token_ids.append(token_id)

        # Handle case where no new tokens were actually added (prevent runtime errors)
        if not new_special_token_ids:
            ids_tensor = torch.empty(0, dtype=torch.long)
            # Lookup tensor still needs to cover the full vocab to support indexing
            lookup_tensor = torch.full((total_vocab_size,), -1, dtype=torch.long)
            return ids_tensor, lookup_tensor, 0

        # Sort IDs to ensure deterministic index assignment (0, 1, 2...) in the new embedding table
        ids_tensor = torch.tensor(sorted(new_special_token_ids), dtype=torch.long)

        # Initialize lookup with -1 (default = "Use Base Embedding")
        lookup_tensor = torch.full((total_vocab_size,), -1, dtype=torch.long)

        # Populate the lookup: Global ID -> Local Index (0 to N-1)
        # Example: If [ACTIONS] is ID 32005, and it's the first new token:
        # lookup_tensor[32005] = 0
        for idx, token_id in enumerate(ids_tensor.tolist()):
            lookup_tensor[token_id] = idx

        return ids_tensor, lookup_tensor, len(new_special_token_ids)

    def initialize_new_token_weights(self):
        """ Initializes the special embedding parts with the mean of the base weights. """
        # MODIFIED: Initialize both special embedding groups
        # Not used yet. Initialization function is in gr00t_finetune.py
        hybrid_embedding = self.eagle_model.get_input_embeddings()
        if not isinstance(hybrid_embedding, HybridEmbedding):
            print("Not a HybridEmbedding; skipping special token initialization.")
            return

        base_weights = hybrid_embedding.base_embedding.weight

        with torch.no_grad():
            mean_vec = base_weights.detach().to(torch.float32).mean(dim=0)

            if hasattr(hybrid_embedding, 'special_embedding_A') and hybrid_embedding.special_embedding_A.weight.size(
                    0) > 0:
                target_weight_A = hybrid_embedding.special_embedding_A.weight
                mean_vec_A = mean_vec.to(dtype=target_weight_A.dtype, device=target_weight_A.device)
                target_weight_A.copy_(mean_vec_A.repeat(target_weight_A.size(0), 1))
                print("Initialized new special token embeddings (Group A) with the mean vector.")

            if hasattr(hybrid_embedding, 'special_embedding_B') and hybrid_embedding.special_embedding_B.weight.size(
                    0) > 0:
                target_weight_B = hybrid_embedding.special_embedding_B.weight
                mean_vec_B = mean_vec.to(dtype=target_weight_B.dtype, device=target_weight_B.device)
                target_weight_B.copy_(mean_vec_B.repeat(target_weight_B.size(0), 1))
                print("Initialized new special token embeddings (Group B) with the mean vector.")

    def tie_special_weights(self):
        """ Ties the new special LM heads to the new special embeddings. """
        # Tie both pairs of special weights
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

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool, tune_special_A: bool, tune_special_B: bool,
                                 tune_tool_end: bool = False):
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
        self.tool_head.requires_grad_(tune_tool_end)

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

        # Print the *actual* status of the special layers
        if isinstance(hybrid_embedding, HybridEmbedding):
            emb_A = getattr(hybrid_embedding, 'special_embedding_A', None)
            emb_B = getattr(hybrid_embedding, 'special_embedding_B', None)
            status_A = emb_A.weight.requires_grad if emb_A and emb_A.weight.size(0) > 0 else 'N/A'
            status_B = emb_B.weight.requires_grad if emb_B and emb_B.weight.size(0) > 0 else 'N/A'

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
        eagle_input = {k.removeprefix(eagle_prefix): v for k, v in vl_input.items() if
            k.startswith(eagle_prefix) and k != 'eagle_num_images' and 'length' not in k}
        if 'image_sizes' in eagle_input:
            del eagle_input["image_sizes"]
        if 'llm_labels' in eagle_input:
            eagle_input.pop('llm_labels')

        eagle_output = self.eagle_model(**eagle_input, past_key_values=past_key_values, output_hidden_states=True,
                                        return_dict=True, use_cache=True)
        past_key_values = eagle_output.past_key_values
        eagle_features = eagle_output.hidden_states[self.select_layer]
        eagle_features = self.eagle_linear(eagle_features)
        eagle_logits = eagle_output.logits
        eagle_attn = eagle_input["attention_mask"]

        # Raw Hidden States for Tool Head / Tool
        raw_hidden_states = eagle_output.hidden_states[-1]

        return eagle_logits, eagle_features, eagle_attn, past_key_values, raw_hidden_states

    def _transcript_lm_loss(self, vl_input: BatchFeature) -> torch.Tensor:
        """
        Calculates the Language Modeling loss with specific handling for:
        1. Hybrid Vocabularies: Separates loss for frozen base tokens vs. trainable special tokens.
        2. Next-Step Prediction: Optionally masks out history to focus training on the immediate response.
        3. Auxiliary Tasks: Adds loss for Tool/Tool-End classification heads.

        Args:
            vl_input (BatchFeature): Batch containing 'eagle_input_ids', 'eagle_labels', etc.

        Returns:
            tuple: (logits, labels, total_loss, base_loss_avg, special_loss_avg, ...)
        """

        def find_last_step(input_ids, labels):
            """
            Identifies the start of the final 'assistant' response in the sequence.
            Used to mask out previous conversation turns so we only calculate loss on the *current* step.
            """
            B, T = labels.shape
            final_mask = torch.zeros_like(labels, dtype=torch.bool)

            for b in range(B):
                ids = input_ids[b]

                # Find all positions marked as 'assistant' (start of a response)
                step_markers = ids == self.assistant_id

                if not step_markers.any():
                    continue

                # Get the index of the *last* assistant token
                last_idx = torch.nonzero(step_markers, as_tuple=False)[-1, 0].item()

                # Mask everything BEFORE this response + \n (indices 0 to last_idx+2).
                if last_idx > 0:
                    final_mask[b, :last_idx + 2] = True

            return final_mask

        # Return 0.0 if no labels provided (inference mode check)
        if "eagle_llm_labels" not in vl_input:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        #########################################
        # 1. Prepare Inputs
        # Filter keys to match model signature (remove 'eagle_' prefix)
        eagle_input = {k.removeprefix("eagle_"): v for k, v in vl_input.items() if
                       k.startswith("eagle_") and k != "eagle_num_images" and 'length' not in k}
        eagle_input.pop("image_sizes", None)
        labels = eagle_input.pop("llm_labels")

        # Mask out ignored tokens (padding, special delimiters)
        ignored_tensor = torch.isin(labels, self.ignored_id.to(labels.device))
        labels[ignored_tensor] = -100

        #########################################
        # 2. Next-Step Masking (Optional)
        # If enabled, ignore the loss for the entire history and only train on the final response.
        if self.pred_nextstep:
            final_mask = find_last_step(eagle_input['input_ids'], labels)
            labels[final_mask] = -100

        #########################################
        # 3. Forward Pass
        need_hidden = self.tune_tool_end
        outputs = self.eagle_model(**eagle_input, return_dict=True, output_hidden_states=need_hidden)
        logits = outputs.logits

        # Standard Causal Shift: Predict t+1 given t
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        #########################################
        # --- 4. Auxiliary Loss: Tool & Tool End Prediction ---
        toolend_loss_avg = torch.tensor(0.0, device=shift_labels.device)
        tool_loss_avg = torch.tensor(0.0, device=shift_labels.device)

        if self.tune_tool_end and self.pred_nextstep:
            # Logic: Find the valid 'last token' of the sequence to predict if the tool has ended.
            seq_len = final_mask.size(1)
            range_tensor = torch.arange(seq_len, device=final_mask.device).unsqueeze(0)

            # Use masked_indices to find the true end of the sequence (ignoring padding)
            masked_indices = torch.where(final_mask, range_tensor, -1)
            last_token_indices = masked_indices.max(dim=1).values
            valid_rows_mask = last_token_indices != -1

            if valid_rows_mask.any():
                batch_indices = torch.arange(logits.size(0), device=logits.device)[valid_rows_mask]
                selected_indices = last_token_indices[valid_rows_mask]

                # Extract hidden state at the last valid step
                selected_hidden = outputs.hidden_states[-1][batch_indices, selected_indices]
                # Get the actual token ID at the next step (the ground truth for classification)
                selected_targets_ids = eagle_input['input_ids'][batch_indices, selected_indices + 1]

                # Create binary classification targets
                target_tool_end = (selected_targets_ids == self.skills_end).long()  # Is next token [TOOLS_END]?
                target_tool = (selected_targets_ids == self.tools_id).long()  # Is next token [TOOLS]?

                # Compute Auxiliary Loss
                tool_end_logits_step = self.tool_end_head(selected_hidden)
                tool_logits_step = self.tool_head(selected_hidden)

                tool_loss_fct = nn.CrossEntropyLoss(reduction='mean')
                toolend_loss_avg = tool_loss_fct(tool_end_logits_step, target_tool_end)
                tool_loss_avg = tool_loss_fct(tool_logits_step, target_tool)

        #########################################
        # --- 5. Main LM Loss (Hybrid Strategy) ---
        vocab_size = shift_logits.size(-1)
        valid_mask = shift_labels != -100

        num_special_A = self.special_token_ids_A.numel()
        num_special_B = self.special_token_ids_B.numel()
        num_special_total = num_special_A + num_special_B

        special_loss_AB = None
        base_loss = None

        if num_special_total == 0:
            # Case A: Standard Vocabulary Only (No special tokens added)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
            per_token_loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
            per_token_loss = per_token_loss.view_as(shift_labels)
        else:
            # Case B: Hybrid Vocabulary
            # We must split the loss because Base tokens and Special tokens live in different embedding spaces.
            special_ids_A = self.special_token_ids_A.to(shift_labels.device)
            special_ids_B = self.special_token_ids_B.to(shift_labels.device)

            # Create boolean masks to partition the batch into Base vs Special
            special_mask_A = torch.isin(shift_labels, special_ids_A) & valid_mask
            special_mask_B = torch.isin(shift_labels, special_ids_B) & valid_mask
            special_mask_AB = special_mask_A | special_mask_B
            base_mask = valid_mask & ~special_mask_A & ~special_mask_B

            per_token_loss = shift_logits.new_zeros(shift_labels.shape, dtype=shift_logits.dtype)

            # Get size of the original LLM head to define the boundary
            base_lm_head = self.eagle_model.get_output_embeddings().base_head
            base_vocab_size = base_lm_head.out_features

            #########################################
            # 5a. Calculate Loss for Base Tokens
            if base_mask.any():
                # Slice logits to only include base vocabulary [0 ... base_vocab_size]
                base_logits = shift_logits[..., :base_vocab_size]
                base_loss = F.cross_entropy(base_logits[base_mask], shift_labels[base_mask], reduction="none")
                per_token_loss[base_mask] = base_loss.to(dtype=per_token_loss.dtype)

            #########################################
            # 5b. Calculate Loss for Special Tokens
            if special_mask_AB.any():
                # Slice logits to only include special vocabulary [base_vocab_size ... end]
                special_logits_AB = shift_logits[..., base_vocab_size:]

                # Shift labels: Map global ID (e.g. 32005) -> Local Index (e.g. 5)
                # This aligns the label with the sliced `special_logits_AB`
                labels_AB = shift_labels[special_mask_AB]
                target_positions_AB = (labels_AB - base_vocab_size).long()

                special_loss_AB = F.cross_entropy(special_logits_AB[special_mask_AB], target_positions_AB,
                                                  reduction="none")
                per_token_loss[special_mask_AB] = special_loss_AB.to(dtype=per_token_loss.dtype)

        #########################################
        # --- 6. Aggregate Final Loss ---
        if base_loss is not None:
            base_loss_avg = base_loss.mean()
        else:
            base_loss_avg = torch.tensor(0.0, device=shift_labels.device)

        # Average special loss across both groups (A and B)
        special_loss_combined = []
        special_loss_AB_avg = torch.tensor(0.0, device=shift_labels.device)
        special_loss_B_avg = torch.tensor(0.0, device=shift_labels.device)  # Kept for logging signature

        if special_loss_AB is not None:
            special_loss_AB_avg = special_loss_AB.mean()
            special_loss_combined.append(special_loss_AB)

        # Note: Logic for B is subsumed in AB in this implementation, but variable kept for return consistency.
        if special_loss_combined:
            special_loss_avg = torch.cat(special_loss_combined).mean()
        else:
            special_loss_avg = torch.tensor(0.0, device=shift_labels.device)

        # Final Summation
        loss = special_loss_avg + base_loss_avg

        # Add weighted auxiliary loss if tuning enabled
        if self.tune_tool_end:
            loss = loss + (self.tool_end_loss_weight * tool_loss_avg) + (self.tool_end_loss_weight * toolend_loss_avg)

        return logits, labels, loss, base_loss_avg, special_loss_AB_avg, special_loss_B_avg

    def split_by_img_id(self, vl_input, eagle_logits: torch.Tensor, eagle_mask: torch.Tensor):
        """
        Segments the sequence into individual 'turns' and filters for Physical Action contexts.

        This function identifies distinct interaction turns by locating the User Instruction pattern
        (`<|im_start|>user...`). It then inspects the content of each turn to determine if it
        represents a physical action or a reasoning/tool-use step.

        Logic:
        1. Find all User Instruction starts (`<|im_start|>user`) to define segment boundaries.
        2. Within each segment, check if the model transitions to `[ACTIONS]` or `[TOOLS]`.
        3. Filter: Keep the segment ONLY if `[ACTIONS]` appears strictly before any `[TOOLS]` token.
           (This ensures we only train the Action Policy on physical tasks, not tool-use reasoning).
        4. Trim: The segment is trimmed to end just before the `[ACTIONS]` token, effectively capturing
           the context (Instruction + Images) leading up to the action.

        Args:
            vl_input (dict): Batch input containing 'input_ids'.
            eagle_logits (torch.Tensor): The model's output logits [B, T, V].
            eagle_mask (torch.Tensor): Mask tensor indicating valid regions.

        Returns:
            tuple:
                - segments (list[Tensor]): List of logit slices for valid action contexts.
                - segments_mask (list[Tensor]): Corresponding masks.
                - Indices (batch, start, end) for tracking where these segments came from.
        """
        eagle_input = {k[6:]: v for k, v in vl_input.items() if k.startswith("eagle_") and k != "eagle_num_images"}

        input_ids = eagle_input["input_ids"]  # [B, T]
        attn_mask = eagle_input["attention_mask"]  # [B, T]

        B, T = input_ids.shape
        device = input_ids.device

        # Token ids
        img_id = self.img_id
        actions_id = self.actions_id
        tools_id = self.tools_id
        user_pattern = self.user_instr_ids  # Pattern: <|im_start|>user\n
        K = user_pattern.numel()

        segments: list[torch.Tensor] = []
        segments_mask: list[torch.Tensor] = []
        seg_batch: list[int] = []
        seg_starts: list[int] = []
        seg_ends: list[int] = []

        # Compute valid lengths from attention mask (handles left-padded sequences common in HF)
        valid_len = attn_mask.sum(dim=1).to(torch.long)  # [B]

        for b in range(B):
            ids_b = input_ids[b]
            mask_b = eagle_mask[b]
            Lb = valid_len[b].item()
            if Lb <= 0:
                continue

            # Determine effective search area (ignoring padding)
            first_valid = T - Lb
            last_valid_idx = T - 1
            search_start = first_valid
            search_end_exclusive = T

            # --- 1. Find User Turn Starts ---
            user_starts: list[int] = []
            max_start_offset = (search_end_exclusive - search_start) - K
            if max_start_offset >= 0:
                for offset in range(max_start_offset + 1):
                    # Sliding window check for the User Pattern
                    window = ids_b[search_start + offset: search_start + offset + K]
                    if torch.equal(window, user_pattern):
                        user_starts.append(search_start + offset)

            if not user_starts:
                continue

            # --- 2. Process Each User Turn ---
            for i, u_p in enumerate(user_starts):
                start_pos = u_p

                # Determine the raw end boundary of this turn
                if i + 1 < len(user_starts):
                    # Turn ends just before the NEXT user instruction starts
                    next_user_start = user_starts[i + 1]
                    temp_end_boundary = next_user_start - 1
                    window_between = ids_b[start_pos:next_user_start]
                else:
                    # Last turn ends at the sequence end
                    window_between = ids_b[start_pos:last_valid_idx]
                    temp_end_boundary = last_valid_idx

                # --- 3. Refine Boundary: Trim to [ACTIONS] ---
                # We specifically want the context *leading up to* the action.
                # Find all occurrences of [ACTIONS] in this window.
                act_occurrences = (window_between == actions_id).nonzero(as_tuple=False)
                if act_occurrences.numel() > 0:
                    # If found, set boundary to index just BEFORE the last [ACTIONS] token.
                    # This captures "User Instr + Image", excluding the Action tokens themselves.
                    last_act_rel_idx = act_occurrences[-1, 0].item()
                    end_boundary = start_pos + last_act_rel_idx - 1
                else:
                    # Fallback (likely filtered out later if no ACTIONS found)
                    end_boundary = temp_end_boundary

                if end_boundary < start_pos:
                    continue  # Empty segment

                # --- 4. Filter: Action vs. Tool ---
                # We need to distinguish between physical actions and tool-use reasoning.
                # Check the window (including the boundary token) for routing tokens.
                window_ids = ids_b[start_pos: end_boundary + 1]

                # Check occurrence indices relative to the current window start
                act_rel = (ids_b[start_pos: end_boundary + 2] == actions_id).nonzero(as_tuple=False)
                tol_rel = (ids_b[start_pos: end_boundary + 2] == tools_id).nonzero(as_tuple=False)

                def first_pos(rel_idx):
                    return rel_idx[0, 0].item() if rel_idx.numel() > 0 else None

                act_first = first_pos(act_rel)
                tol_first = first_pos(tol_rel)

                # CRITICAL FILTER:
                # 1. Must contain an [ACTIONS] token.
                # 2. If a [TOOLS] token exists, [ACTIONS] must appear *before* it.
                if act_first is None:
                    continue
                if tol_first is not None and tol_first < act_first:
                    continue

                # Define final slice indices
                end_exclusive = end_boundary + 1
                if end_exclusive <= start_pos:
                    continue

                # Slice logits and mask
                seg = eagle_logits[b, start_pos:end_exclusive, :]
                msk = mask_b[start_pos:end_exclusive]

                segments.append(seg)
                segments_mask.append(msk)
                seg_batch.append(b)
                seg_starts.append(start_pos)
                seg_ends.append(end_exclusive)

        # Handle empty batch case to avoid downstream errors
        if len(segments) == 0:
            return ([], [], torch.empty((0,), dtype=torch.long, device=device),
                    torch.empty((0,), dtype=torch.long, device=device),
                    torch.empty((0,), dtype=torch.long, device=device),)

        return (segments, segments_mask, torch.tensor(seg_batch, dtype=torch.long, device=device),
                torch.tensor(seg_starts, dtype=torch.long, device=device),
                torch.tensor(seg_ends, dtype=torch.long, device=device),)

    def forward_route(self, vl_input: BatchFeature, ):
        # 1) Run backbone once
        eagle_logits, eagle_embeds, eagle_mask, past_key_values, _ = self.forward_eagle(vl_input,)

        # if there is no step information (single instruction input)
        input_keys = vl_input.keys()
        step_input = [item for item in input_keys if 'step' in item]

        #########################
        # normal input
        # directly return logits and embd
        transcript_lm_loss = torch.tensor(0.0, device=eagle_logits.device)
        base_loss_avg = torch.tensor(0.0, device=eagle_logits.device)
        special_loss_A_avg = torch.tensor(0.0, device=eagle_logits.device)
        special_loss_B_avg = torch.tensor(0.0, device=eagle_logits.device)
        embeds_tensor, masks_tensor = None, None
        logits, labels = None, None

        if len(step_input) != 0:

            # Compute generated loss
            logits, labels, transcript_lm_loss, base_loss_avg, special_loss_A_avg, special_loss_B_avg = self._transcript_lm_loss(
                vl_input)

            # extract action token hidden states based on action_pad_ids
            has_actions = (vl_input['eagle_input_ids'] == self.actions_id).any().item()

            if has_actions:
                list_eagle_emb, list_eagle_mask, seg_batch, seg_start, seg_end = self.split_by_img_id(vl_input,
                                                                                                      eagle_embeds,
                                                                                                      eagle_mask)
                # import pdb;pdb.set_trace()
                # self.eagle_tokenizer.decode(vl_input['eagle_input_ids'][0])
                # self.eagle_tokenizer.decode(vl_input['eagle_input_ids'][0][1668:2208])
                embeds_tensor, masks_tensor = flatten_actions(list_eagle_emb, list_eagle_mask)

        out = {"transcript_lm_loss": transcript_lm_loss, "text_token_loss": base_loss_avg,
            "special_token_A_loss": special_loss_A_avg, "special_token_B_loss": special_loss_B_avg,
            "eagle_embeds": eagle_embeds, "eagle_mask": eagle_mask, "eagle_embeds_multi": embeds_tensor,
            "eagle_mask_multi": masks_tensor, "past_key_values": past_key_values, "logits": logits, "labels": labels}

        return out

    def forward(self, vl_input: BatchFeature, ) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        out = self.forward_route(vl_input, )
        eagle_embeds = out["eagle_embeds"]
        eagle_mask = out["eagle_mask"]
        eagle_embeds_multi = out["eagle_embeds_multi"]
        eagle_mask_multi = out["eagle_mask_multi"]

        # YL (TODO HACK): to resolve DDP issue when tune_visual=True
        # Ensure all trainable parameters in vision_model are used in the forward pass for DDP compatibility
        if self.training and self.tune_visual:
            dummy_term = torch.tensor(0.0, device=eagle_embeds.device, dtype=eagle_embeds.dtype, requires_grad=True)
            for param in self.eagle_model.vision_model.parameters():
                if param.requires_grad:
                    dummy_term = dummy_term + 0.0 * param.sum()
            eagle_embeds = eagle_embeds + dummy_term

        return BatchFeature(data={
            "backbone_features": eagle_embeds,  # [Bd, T, H]
            "backbone_attention_mask": eagle_mask,  # [Bd, T]
            "backbone_features_multi": eagle_embeds_multi,  # [A, Lmax, H]
            "backbone_attention_mask_multi": eagle_mask_multi,  # [A, Lmax]
            "transcript_lm_loss": out["transcript_lm_loss"],
            "text_token_loss": out["text_token_loss"],
            "special_token_A_loss": out["special_token_A_loss"],
            "special_token_B_loss": out["special_token_B_loss"],
            "orig_batch_size": out["eagle_embeds"].size(0),
            "past_key_values": None,
            "logits": out["logits"],
            "labels": out["labels"], })

    @torch.no_grad()
    def generate(self, vl_input: BatchFeature, max_token: int = 1, past_key_values=None, special_token_only=False,
                 inside_tool=False, toolend_head=False):
        """
        Performs greedy autoregressive token generation for the language backbone.

        This generation process has two distinct phases:
        1. **Router Phase (First Token):** The model MUST output a control token determining the next mode:
           - `[ACTIONS]`: Output physical robot actions.
           - `[TOOLS]`: Use a tool/skill (triggers text generation for parameters).
           - `[TOOLS_END]`: End the current skill/tool execution.
           *Note:* If `toolend_head` is True, auxiliary binary classifiers override the LM's choice here.

        2. **Content Phase (Subsequent Tokens):** If the model continues, it generates natural language
           (e.g., arguments for a tool) using the standard base vocabulary.

        Args:
            vl_input (BatchFeature): Input features including input_ids and images.
            max_token (int): Maximum length of the generated sequence.
            inside_tool (bool): If True, restricts valid Router tokens to `[ACTIONS]` or `[TOOLS_END]`
                                (cannot nest `[TOOLS]` inside another tool).
            toolend_head (bool): If True, uses the lightweight auxiliary heads (`tool_end_head`)
                                 instead of the heavy LM head for the Router decision.

        Returns:
            tuple: (router_token_id, decoded_text, backbone_outputs)
        """
        if max_token is None or max_token < 1:
            max_token = 1

        if not isinstance(vl_input, BatchFeature):
            vl_input = BatchFeature(data=dict(vl_input))

        self.set_frozen_modules_to_eval_mode()

        input_ids = vl_input["eagle_input_ids"]
        batch_size = input_ids.size(0)
        device = input_ids.device
        base_vocab_size = self.eagle_model.get_output_embeddings().base_head.out_features

        # --- 1. Define Valid Router Tokens ---
        # Constrain the first token to be a structural control token.
        if inside_tool:
            allowed_ids = torch.tensor([self.actions_id, self.skills_end], device=device, dtype=torch.long)
        else:
            allowed_ids = torch.tensor([self.actions_id, self.tools_id, self.skills_end], device=device,
                                       dtype=torch.long)

        generated_tokens: list[torch.Tensor] = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # --- 2. Phase 1: Router Generation (The First Token) ---
        logits, eagle_embeds, eagle_masks, _, last_HS = self.forward_eagle(vl_input, past_key_values=past_key_values)

        # Masking: Force the model to select one of the `allowed_ids`
        next_token_logits = logits[:, -1, :]
        mask = torch.full_like(next_token_logits, float('-inf'))
        mask.scatter_(dim=-1, index=allowed_ids.unsqueeze(0).expand(batch_size, -1), value=0)
        masked_token_logits = next_token_logits + mask
        router_token_id = masked_token_logits.argmax(dim=-1)
        generated_tokens.append(router_token_id)

        # Optional: Override Router with Auxiliary Heads
        if toolend_head:
            # use hidden state from the last step
            # last_HS is [B, T, H]
            current_hidden = last_HS[:, -1, :]  # [B, H]

            # Pass through the binary classifiers
            toolend_logits = self.tool_end_head(current_hidden)  # [B, 2]
            tool_logits = self.tool_head(current_hidden)  # [B, 2]

            # Predict: 0 = Negative, 1 = Positive (Trigger)
            toolend_preds = toolend_logits.argmax(dim=-1)  # [B]
            tool_preds = tool_logits.argmax(dim=-1)  # [B]

            # Logic: Default to [ACTIONS]. Override if classifiers say otherwise.
            router_token_id_head = torch.tensor(self.actions_id, device=device)

            mask_end = (toolend_preds == 1)
            if mask_end.any():
                router_token_id_head = torch.tensor(self.skills_end, device=device)

            mask_tool = (tool_preds == 1)  # Prevent conflict
            if mask_tool.any():
                router_token_id_head = torch.tensor(self.tools_id, device=device)

            # Apply override
            if router_token_id_head != router_token_id:
                router_token_id = router_token_id_head.unsqueeze(0)

        # Prepare input for the next step (autoregressive)
        token_to_append = router_token_id.unsqueeze(0)
        vl_input["eagle_input_ids"] = torch.cat([vl_input["eagle_input_ids"][:1], token_to_append], dim=1)
        vl_input["eagle_attention_mask"] = torch.cat(
            [vl_input["eagle_attention_mask"][:1], torch.ones_like(token_to_append)], dim=1)

        # --- 3. Phase 2: Content Generation (Subsequent Tokens) ---
        for _ in range(max_token - 1):
            logits, _, _, _, _ = self.forward_eagle(vl_input, past_key_values=past_key_values)

            # Select the most likely token from the *Base Vocabulary*.
            # We exclude special tokens here to prevent the model from hallucinating new control codes mid-sentence.
            next_token_raw = logits[:, -1, :base_vocab_size].argmax(dim=-1)

            # Update inputs
            token_to_append = next_token_raw.unsqueeze(0)
            vl_input["eagle_input_ids"] = torch.cat([vl_input["eagle_input_ids"][:1], token_to_append], dim=1)
            vl_input["eagle_attention_mask"] = torch.cat(
                [vl_input["eagle_attention_mask"][:1], torch.ones_like(token_to_append)], dim=1)

            # Check for EOS (End of Sentence)
            prev_finished = finished.clone()
            finished = prev_finished | (next_token_raw == self.end_id)

            # If finished, pad with EOS; otherwise record the token
            recorded_token = torch.where(prev_finished, torch.full_like(next_token_raw, self.end_id), next_token_raw, )
            generated_tokens.append(recorded_token)

            if finished.all():
                break

        # --- 4. Final Cleanup ---
        # Run one last forward pass to get final KV cache states if needed for downstream tasks
        _, _, _, final_kv_cache, _ = self.forward_eagle(vl_input, past_key_values=past_key_values)

        backbone_outputs = BatchFeature({"backbone_features": eagle_embeds, "backbone_attention_mask": eagle_masks,
                                         "past_key_values": final_kv_cache, })

        generated_ids = torch.stack(generated_tokens, dim=1)
        if generated_ids.size(1) == 1:
            generated_ids = generated_ids.squeeze(1)

        decoded_text = self.eagle_tokenizer.batch_decode(generated_ids)[0]

        return router_token_id.unsqueeze(1), decoded_text, backbone_outputs

    @torch.no_grad()
    def generate_v2(self, vl_input: BatchFeature, max_token: int = 1, past_key_values=None, inside_tool=False,
                    toolend_head=False):
        """
        Two-stage generation:
        1. Router Step: Force pick [TOOL, ACTION, END].
        2. Branching: 
           - TOOL: Generate full text explanation.
           - ACTION: Return hidden states for policy head.
        """

        def generate_text_kvcache(input_ids, attention_mask, token_to_append):
            generation_inputs = {"input_ids": input_ids, "attention_mask": attention_mask,
                "past_key_values": past_key_values, "use_cache": True}

            if "eagle_pixel_values" in vl_input and vl_input["eagle_pixel_values"] is not None:
                generation_inputs["pixel_values"] = vl_input["eagle_pixel_values"]
            if "eagle_image_sizes" in vl_input:
                generation_inputs["image_sizes"] = vl_input["eagle_image_sizes"]

            # 2. Run Generation
            # return_dict_in_generate=True is REQUIRED to get the new past_key_values back
            gen_output = self.eagle_model.generate(**generation_inputs, max_new_tokens=max_token - 1,
                pad_token_id=self.pad_id, eos_token_id=self.end_id, do_sample=False, return_dict_in_generate=True,
                output_hidden_states=True, )

            # 3. Package Outputs
            full_ids = gen_output.sequences
            full_ids = torch.concat([token_to_append, full_ids], dim=1)
            final_kv_cache = gen_output.past_key_values  # kv cache that include text generated this step

            decoded_text = self.eagle_tokenizer.batch_decode(full_ids)[0]
            return final_kv_cache, decoded_text

        if max_token is None or max_token < 1:
            max_token = 1

        if not isinstance(vl_input, BatchFeature):
            vl_input = BatchFeature(data=dict(vl_input))

        self.set_frozen_modules_to_eval_mode()

        input_ids = vl_input["eagle_input_ids"]
        attention_mask = vl_input["eagle_attention_mask"]
        batch_size = input_ids.size(0)
        device = input_ids.device

        if inside_tool:
            allowed_ids = torch.tensor([self.actions_id, self.skills_end], device=device, dtype=torch.long)
        else:
            allowed_ids = torch.tensor([self.tools_id, self.actions_id, self.skills_end], device=device,
                dtype=torch.long)

        # 1. Forward Pass (Single Step)
        # We use your custom forward to get logits AND hidden states
        logits, eagle_embeds, eagle_masks, _, last_HS = self.forward_eagle(vl_input, past_key_values=past_key_values)

        # 2. Masking (Force selection of Action, Tool, or End)
        next_token_logits = logits[:, -1, :]
        mask = torch.full_like(next_token_logits, float('-inf'))
        mask.scatter_(dim=-1, index=allowed_ids.unsqueeze(0).expand(batch_size, -1), value=0)
        masked_token_logits = next_token_logits + mask
        router_token_id = masked_token_logits.argmax(dim=-1)

        # STEP 1: ROUTER (Generate the Special Token first)
        if toolend_head:
            # use hidden state to predict toolend or not
            # last_HS is [B, T, H]
            current_hidden = last_HS[:1, -1, :]  # [B, H]

            # 2. Pass through the binary classifier
            toolend_logits = self.tool_end_head(current_hidden)  # [B, 2]
            tool_logits = self.tool_head(current_hidden)  # [B, 2]

            # 3. Predict: 0 = Keep Going ([ACTIONS]), 1 = End ([TOOLS_END])
            toolend_preds = toolend_logits.argmax(dim=-1)  # [B]
            tool_preds = tool_logits.argmax(dim=-1)  # [B]

            # 4. OVERRIDE Logic
            router_token_id_head = torch.tensor(self.actions_id, device=device)
            mask_end = (toolend_preds == 1)
            if mask_end.any():
                router_token_id_head = torch.tensor(self.skills_end, device=device)

            mask_tool = (tool_preds == 1)  # Prevent conflict
            if mask_tool.any():
                router_token_id_head = torch.tensor(self.tools_id, device=device)

            if router_token_id_head != router_token_id:
                router_token_id = router_token_id_head.unsqueeze(0)

        if not inside_tool:
            # result different
            # append router_token_id_head to current input
            token_to_append = router_token_id.unsqueeze(0)
            input_ids_action = torch.cat([vl_input["eagle_input_ids"][:1], token_to_append], dim=1)
            attention_mask_action = torch.cat([vl_input["eagle_attention_mask"][:1], torch.ones_like(token_to_append)],
                                              dim=1)
            final_kv_cache, decoded_text = generate_text_kvcache(input_ids_action, attention_mask_action,
                                                                 token_to_append)
            # print(self.eagle_tokenizer.decode(input_ids_action[0]))

            router_token_id_tool = torch.tensor([self.tools_id], device=device)
            token_to_append = router_token_id_tool.unsqueeze(0)
            input_ids_tool = torch.cat([vl_input["eagle_input_ids"][:1], token_to_append], dim=1)
            attention_mask_tool = torch.cat([vl_input["eagle_attention_mask"][:1], torch.ones_like(token_to_append)],
                                            dim=1)
            tool_kv_cache, decoded_text_tool = generate_text_kvcache(input_ids_tool, attention_mask_tool,
                                                                     token_to_append)
            # print(self.eagle_tokenizer.decode(input_ids_tool[0]))

            if decoded_text != decoded_text_tool:
                print(f"Original: {decoded_text}")
                print(f"TOOLS: {decoded_text_tool}")
                decoded_text = decoded_text_tool
                final_kv_cache = tool_kv_cache

        else:
            # result different
            # append router_token_id to current input
            # router_token_id = torch.tensor([self.actions_id], device=device)
            token_to_append = router_token_id.unsqueeze(0)
            input_ids_action = torch.cat([vl_input["eagle_input_ids"][:1], token_to_append], dim=1)
            attention_mask_action = torch.cat([vl_input["eagle_attention_mask"][:1], torch.ones_like(token_to_append)],
                                              dim=1)
            final_kv_cache, decoded_text = generate_text_kvcache(input_ids_action, attention_mask_action,
                                                                 token_to_append)

            if '[ACTIONS]' not in decoded_text:
                print(f"in tools: {decoded_text}")

        backbone_outputs = BatchFeature({"backbone_features": eagle_embeds, "backbone_attention_mask": eagle_masks,
            "past_key_values": final_kv_cache, })

        return router_token_id.unsqueeze(1), decoded_text, backbone_outputs

    @torch.no_grad()
    def generate_entire_text(self, vl_input: BatchFeature, max_new_tokens: int = 50):
        """
        Uses the optimized Hugging Face .generate() method.
        Automatically handles KV-Caching and Vision encoding.
        """
        self.set_frozen_modules_to_eval_mode()

        generation_inputs = {"input_ids": vl_input["eagle_input_ids"],
            "attention_mask": vl_input["eagle_attention_mask"]}

        if "eagle_pixel_values" in vl_input and vl_input["eagle_pixel_values"] is not None:
            generation_inputs["pixel_values"] = vl_input["eagle_pixel_values"]
        if "eagle_image_sizes" in vl_input:
            generation_inputs["image_sizes"] = vl_input["eagle_image_sizes"]

        # 3. Configure Generation Parameters
        gen_kwargs = {"max_new_tokens": max_new_tokens, "pad_token_id": self.pad_id, "eos_token_id": self.end_id,
            "use_cache": True, "do_sample": False}

        new_tokens = self.eagle_model.generate(**generation_inputs, **gen_kwargs)
        decoded_text = self.eagle_tokenizer.batch_decode(new_tokens)

        return new_tokens, decoded_text
