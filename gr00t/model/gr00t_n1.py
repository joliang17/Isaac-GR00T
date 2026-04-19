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
from typing import Tuple
import itertools
import numpy as np
import torch
import tree
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature

from .action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)
from .backbone import EagleBackbone

BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3


# config
@dataclass
class GR00T_N1_5_Config(PretrainedConfig):
    model_type = "gr00t_n1_5"
    backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration."})

    action_head_cfg: dict = field(init=False, metadata={"help": "Action head configuration."})

    action_horizon: int = field(init=False, metadata={"help": "Action horizon."})

    action_dim: int = field(init=False, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    pred_nextstep: bool = field(default=True, metadata={"help": "Compute dtype."})

    use_skill_emb: bool = field(default=False, metadata={"help": "Compute dtype."})
    skill_vocab: list = field(default=None, metadata={"help": "List of skill name strings used for label extraction during training."})
    num_skills: int = field(default=16, metadata={"help": "Number of learnable skill embedding vectors in the bank."})
    skill_emb_dim: int = field(default=256, metadata={"help": "Dimension of each learnable skill embedding vector in the bank."})
    skill_proj_hidden_dim: int = field(default=256, metadata={"help": "Hidden dim of the MLP projector used in the skill classifier."})
    skill_clf_coeff: float = field(default=1.0, metadata={"help": "Weight for cross-entropy skill classification loss."})
    skill_div_coeff: float = field(default=0.01, metadata={"help": "Weight for orthogonality diversity loss on skill embedding bank."})
    skill_norm_coeff: float = field(default=0.01, metadata={"help": "Weight for non-zero norm loss on skill embedding bank (prevents collapse to zero)."})


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


# real model
class GR00T_N1_5(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = GR00T_N1_5_Config
    """
    we expect the backbone output to have a key 'backbone_features' with shape (batch_size, n, hidden_size)
    here n is variable and can be e.g. time, 1 or user specified
    we expect the action head output to have a key 'action_pred' with shape (batch_size, time, action_dim) during inference time
    we expect these to have type BatchFeature, and they can of course have many other user specified keys too
    """

    def __init__(
        self,
        config: GR00T_N1_5_Config,
        local_model_path: str,
    ):
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)

        super().__init__(config)
        self.local_model_path = local_model_path
        self.backbone = EagleBackbone(pred_nextstep=config.pred_nextstep, **config.backbone_cfg)

        head_kwargs = {
            **config.action_head_cfg,
            "use_skill_emb": config.use_skill_emb,
            "skill_vocab": config.skill_vocab,
            "num_skills": config.num_skills,
            "skill_emb_dim": config.skill_emb_dim,
            "skill_proj_hidden_dim": config.skill_proj_hidden_dim,
            "skill_clf_coeff": config.skill_clf_coeff,
            "skill_div_coeff": config.skill_div_coeff,
            "skill_norm_coeff": config.skill_norm_coeff,
        }
        action_head_cfg = FlowmatchingActionHeadConfig(**head_kwargs)

        self.action_head = FlowmatchingActionHead(action_head_cfg, )

        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype
        self.tie_weights()

    def tie_weights(self):
        """
        Tie the weights of the special token embeddings and the LM heads.
        """
        # Tie the base embeddings and heads
        # This is a standard practice in many transformer models
        if hasattr(self.backbone.eagle_model.language_model.model.embed_tokens, 'base_embedding'):
            base_emb_weight = self.backbone.eagle_model.language_model.model.embed_tokens.base_embedding.weight
            self.backbone.eagle_model.language_model.lm_head.base_head.weight = base_emb_weight
            self.backbone.eagle_model.language_model.lm_head.weight = base_emb_weight
            self.backbone.eagle_model.language_model.model.embed_tokens.weight = base_emb_weight

    @property
    def _tied_weights_keys(self):
        # Group 1: The new special embeddings and their corresponding heads (Group A)
        special_tied_group_A = [
            'backbone.eagle_model.language_model.lm_head.special_head_A.weight',
            'backbone.eagle_model.language_model.model.embed_tokens.special_embedding_A.weight',
        ]

        # Group 2: The new special embeddings and their corresponding heads (Group B)
        special_tied_group_B = [
            'backbone.eagle_model.language_model.lm_head.special_head_B.weight',
            'backbone.eagle_model.language_model.model.embed_tokens.special_embedding_B.weight',
        ]

        # Group 3: The base model's embeddings and heads
        base_tied_group = [
            'backbone.eagle_model.language_model.model.embed_tokens.base_embedding.weight',
            'backbone.eagle_model.language_model.lm_head.base_head.weight',
        ]
        
        # Flatten the list of lists into a single list of strings
        all_tied_keys = list(itertools.chain.from_iterable([
            special_tied_group_A, 
            special_tied_group_B, 
            base_tied_group
        ]))
        
        return all_tied_keys

    def validate_inputs(self, inputs):
        # NOTE -- this should be handled internally by the model
        # however, doing that will likely be breaking changes -- so we'll need to do it after the deadline

        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            type_ok = isinstance(action, torch.Tensor)
            shape_ok = (
                len(action.shape) == 3
                and action.shape[1] == self.action_horizon
                and action.shape[2] == self.action_dim
            )
            if not type_ok:
                error_msg += f"\n{action.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{action.shape=}"
                detected_error = True

        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\n{type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n{video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{video.shape=}"
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):
        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature)
            or BACKBONE_FEATURE_KEY not in backbone_outputs
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(backbone_outputs, BatchFeature)=}"
            error_msg += f"\n{BACKBONE_FEATURE_KEY in backbone_outputs=}"
            error_msg += f"\n{backbone_outputs[BACKBONE_FEATURE_KEY].shape=}"
            raise ValueError(error_msg)

        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training
            )  # there might not be an action prediction during training
            or (
                ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == self.action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(action_head_outputs, BatchFeature)=}"
            error_msg += f"\n{LOSS_KEY in action_head_outputs=}"
            error_msg += f"\n{action_head_outputs[ACTION_KEY].shape=}"
            error_msg += f"\n{self.action_horizon=}"
            error_msg += f"\n{self.action_dim=}"
            raise ValueError(error_msg)

    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)

        if 'action' in inputs:
            action_head_outputs = self.action_head(backbone_outputs, action_inputs)
            self.validate_data(action_head_outputs, backbone_outputs, is_training=True)
            action_head_outputs["action_head_skipped"] = False
        else:
            dit_params = next(self.action_head.parameters())
            dummy_loss = (dit_params.sum() * 0.0) 

            output_dict = {"loss": dummy_loss}
            action_head_outputs = BatchFeature(data=output_dict)
            action_head_outputs["action_head_skipped"] = True

        # Merge route/tool losses into the output and total loss.
        ah_loss = action_head_outputs["loss"]
        action_head_outputs["action_head_loss"] = ah_loss
        action_head_outputs.update({k: v for k, v in backbone_outputs.items() if "loss" in k or 'logit' in k or 'eval' in k})
        action_head_outputs["loss"] = ah_loss + action_head_outputs['transcript_lm_loss']
        
        action_head_outputs["labels"] = backbone_outputs['labels']
        return action_head_outputs

    @torch.no_grad()
    def get_action(
        self,
        inputs: dict,
        past_key_values=None,
        mode: str='baseline',
        inside_tool: bool=False,
        toolend_head: bool=False,
        if_debug: bool=False
    ) -> BatchFeature:
        def create_empty_actions(backbone_inputs, batch_size):
            zero_actions = torch.zeros(
                (batch_size, self.action_horizon, self.action_dim),
                dtype=self.action_head.dtype,
                device=self.device,
            )
            action_head_outputs = BatchFeature(data={ACTION_KEY: zero_actions})
            action_head_outputs['action_head_skipped'] = True
            return action_head_outputs

        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        tools_output = ''
        max_generation_steps = max(1, getattr(self, 'max_generation_steps', 64))
        batch_size = backbone_inputs['eagle_input_ids'].size()[0]

        if mode == 'baseline':
            backbone_outputs = self.backbone(backbone_inputs)
            action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
            action_head_outputs['action_head_skipped'] = False
            past_key_values = None
        else:
            # import pdb;pdb.set_trace()
            # DEBUG: generate text first to see what is the output
            # self.backbone.eagle_tokenizer.decode(backbone_inputs['eagle_input_ids'][0])
            # output_ids, decoded_text = self.backbone.generate_entire_text(backbone_inputs, max_new_tokens=max_generation_steps, )
            # print(decoded_text[0])

            # self.backbone.eagle_tokenizer.decode(backbone_inputs['eagle_input_ids'][0])
            token_id, tools_output, backbone_outputs = self.backbone.generate(backbone_inputs, max_token=max_generation_steps, past_key_values=past_key_values, inside_tool=inside_tool, toolend_head=toolend_head, if_debug=if_debug)

            past_key_values = backbone_outputs.get('past_key_values', None)

            if isinstance(token_id, torch.Tensor):
                token_id = token_id.item()
                        
            if token_id == self.backbone.actions_id:
                # Step 2a: use the action head when the route token is [ACTIONS]
                # tools_output = ''
                action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
                action_head_outputs['action_head_skipped'] = False
                
            elif token_id == self.backbone.tools_id:
                if getattr(self, 'skill_action_mode', False):
                    action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
                    action_head_outputs['action_head_skipped'] = False
                else:
                    # Legacy mode: [TOOLS] does not immediately produce actions.
                    action_head_outputs = create_empty_actions(backbone_inputs, batch_size)

            elif token_id == self.backbone.skills_end:
                # Step 2c: refers to the end of a skill execution
                # tools_output = ''
                action_head_outputs = create_empty_actions(backbone_inputs, batch_size)

            else:
                decode_text = self.backbone.eagle_tokenizer.decode(token_id)
                raise ValueError(f'Unexpected route token id: {token_id}, text token: {decode_text}')

        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs, backbone_outputs, tools_output, past_key_values

    def formulate_input_traj(self, inputs):
        """
        Find all pairs (x, x_length) in `inputs` and flatten them.

        For each valid pair where `inputs[x]` is a tensor shaped [B, T, ...] and
        `inputs[f"{x}_length"]` is [B], returns:
          - f"{x}_flat":       [sum(lengths), ...]
          - f"{x}_batch_idx": LongTensor [sum(lengths)] mapping time steps to batch indices
          - f"{x}_ptr":       LongTensor [B+1] prefix sums
        """

        def _pack_by_length(x: torch.Tensor, lengths: torch.Tensor):
            assert x.dim() >= 2, f"Expected [B, T, ...], got {tuple(x.shape)}"
            B, T_max = x.shape[0], x.shape[1]
            device = x.device
            # Ensure lengths is on device and clamped to [0, T_max]
            if not isinstance(lengths, torch.Tensor):
                lengths = torch.as_tensor(lengths, device=device)
            lengths = lengths.to(device=device, dtype=torch.long).clamp(min=0, max=T_max)

            segments = []
            batch_idx = []
            for b in range(B):
                Lb = int(lengths[b].item())
                if Lb > 0:
                    segments.append(x[b, :Lb])
                    batch_idx.append(torch.full((Lb,), b, dtype=torch.long, device=device))

            if len(segments) == 0:
                tail = x.shape[2:]
                flat = x.new_zeros((0,) + tail)
                batch_idx = torch.empty((0,), dtype=torch.long, device=device)
                ptr = torch.zeros((B + 1,), dtype=torch.long, device=device)
                return flat, batch_idx, ptr

            flat = torch.cat(segments, dim=0)
            batch_idx = torch.cat(batch_idx, dim=0)
            ptr = torch.empty((B + 1,), dtype=torch.long, device=device)
            ptr[0] = 0
            ptr[1:] = lengths.cumsum(0)
            return flat, batch_idx, ptr

        out = {}
        suffix = "_length"
        list_base = []
        for key, value in inputs.items():
            if not isinstance(key, str) or not key.endswith(suffix) or 'action' in key or 'state' in key:
                continue
            base = key[: -len(suffix)]
            if base not in inputs:
                continue
            seq = inputs[base]
            lengths = value
            # Only handle tensors that look like [B, T, ...]
            if not isinstance(seq, torch.Tensor) or seq.dim() < 2:
                continue
            flat, batch_idx, ptr = _pack_by_length(seq, lengths)
            if 'action' in base:
                # have action_dim
                flat = flat.reshape(-1, self.action_horizon, self.action_dim)

            list_base.append(base)
            out[f"{base}_orig"] = seq
            out[f"{base}_length"] = lengths
            out[f"{base}_flat"] = flat
            out[f"{base}_batch_idx"] = batch_idx
            out[f"{base}_ptr"] = ptr
            inputs[base] = flat

        for base in list_base:
            del inputs[base + suffix]

        if len(list_base) == 0:
            # no action tokens
            # flatten state / actions
            list_keys = ['state', 'state_mask', 'action', 'action_mask', ]
            for key in list_keys:
                if key not in inputs:
                    continue
                else:
                    if inputs[key].numel() != 0:
                        if len(inputs[key].shape) > 3:
                            inputs[key] = inputs[key].reshape(-1, inputs[key].size(2), inputs[key].size(3))
                    else:
                        del inputs[key]
        return out, inputs, list_base

    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        # if there is no step information (single instruction input)
        input_keys = inputs.keys()
        step_input = [item for item in input_keys if 'step' in item]

        if len(step_input) > 0:
            original, inputs, list_base = self.formulate_input_traj(inputs)

        self.validate_inputs(inputs)
            
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def to_device_with_maybe_dtype(x):
            # Only cast to self.compute_dtype if the tensor is floating
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.action_head.dtype)
            else:
                # Keep original dtype
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return backbone_inputs, action_inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        tune_visual = kwargs.pop("tune_visual", True)
        tune_llm = kwargs.pop("tune_llm", False)
        tune_projector = kwargs.pop("tune_projector", True)
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", True)
        tune_special_A = kwargs.pop("tune_special_A", True)
        tune_special_B = kwargs.pop("tune_special_B", False)
        tune_tool_end = kwargs.pop("tune_tool_end", False)
        tune_trace_projector = kwargs.pop("tune_trace_projector", False)
        tune_skill_clf = kwargs.pop("tune_skill_clf", False)
        tune_skill_emb = kwargs.pop("tune_skill_emb", False)

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune embedding A: {tune_special_A}")
        print(f"Tune embedding B: {tune_special_B}")
        print(f"Tune tool end head: {tune_tool_end}")
        print(f"Tune trace projector: {tune_trace_projector}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")
        print(f"Tune skill classifier: {tune_skill_clf}")
        print(f"Tune skill embedding: {tune_skill_emb}")
        # get the current model path being downloaded
        try:
            # NOTE(YL) This downloads the model to the local cache and returns the local path to the model
            # saved in ~/.cache/huggingface/hub/
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
            # HFValidationError, RepositoryNotFoundError
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {pretrained_model_name_or_path}"
            )
            local_model_path = pretrained_model_name_or_path

        pretrained_model = super().from_pretrained(local_model_path, local_model_path=local_model_path, **kwargs)
        pretrained_model.backbone.set_trainable_parameters(
            tune_visual=tune_visual, tune_llm=tune_llm, tune_special_A=tune_special_A, tune_special_B=tune_special_B, tune_tool_end=tune_tool_end, tune_trace_projector=tune_trace_projector
        )
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector,
            tune_diffusion_model=tune_diffusion_model,
            tune_skill_clf=tune_skill_clf,
            tune_skill_emb=tune_skill_emb,
        )
        return pretrained_model


# register
AutoConfig.register("gr00t_n1_5", GR00T_N1_5_Config)
AutoModel.register(GR00T_N1_5_Config, GR00T_N1_5)
