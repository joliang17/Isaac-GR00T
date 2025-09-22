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

        self.backbone = EagleBackbone(**config.backbone_cfg)
        action_head_cfg = FlowmatchingActionHeadConfig(**config.action_head_cfg)
        self.action_head = FlowmatchingActionHead(action_head_cfg)

        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

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
        transcript_lm_loss = backbone_outputs.get("transcript_lm_loss", torch.tensor(0.0, device=self.device))
        
        action_head_outputs = self.action_head(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=True)

        # Merge route/tool losses into the output and total loss.
        # Keep the pure action-head loss for logging as `action_head_loss`.
        ah_loss = action_head_outputs["loss"]
        action_head_outputs["action_head_loss"] = ah_loss
        action_head_outputs["transcript_lm_loss"] = transcript_lm_loss
        action_head_outputs["loss"] = ah_loss + transcript_lm_loss
        action_head_outputs["action_head_skipped"] = False
        return action_head_outputs

    def get_action(
        self,
        inputs: dict,
        past_key_values=None,
        mode: str='baseline'
    ) -> BatchFeature:
        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        tools_output = ''

        if mode == 'baseline':
            backbone_outputs = self.backbone(backbone_inputs)
            action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
            action_head_outputs['action_head_skipped'] = False
            past_key_values = None
        else:
            generated_ids, backbone_outputs = self.backbone.generate(
                backbone_inputs, max_token=1, past_key_values=past_key_values
            )
            past_key_values = backbone_outputs.get('past_key_values', None)

            if isinstance(generated_ids, torch.Tensor):
                if generated_ids.numel() == 0:
                    raise RuntimeError('Backbone.generate returned no tokens for routing.')
                token_id = int(generated_ids.view(-1)[-1].item())
            elif generated_ids is not None:
                token_id = int(generated_ids)
            else:
                raise RuntimeError('Backbone.generate returned an unexpected token payload.')

            if token_id == self.backbone.actions_id:
                # Step 2a: use the action head when the route token is [ACTIONS]
                action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
                action_head_outputs['action_head_skipped'] = False
            elif token_id == self.backbone.tools_id:
                # Step 2b: keep generating tool tokens until we observe [EOT]
                max_generation_steps = max(1, getattr(self, 'max_generation_steps', 64))
                tools_tokens = []
                steps = 0
                reached_end = False

                while steps < max_generation_steps:
                    generated_ids, backbone_outputs = self.backbone.generate(
                        backbone_inputs, max_token=1, past_key_values=past_key_values
                    )
                    past_key_values = backbone_outputs.get('past_key_values', None)

                    next_token = None
                    if isinstance(generated_ids, torch.Tensor) and generated_ids.numel() > 0:
                        next_token = int(generated_ids.view(-1)[-1].item())
                    elif generated_ids is not None:
                        next_token = int(generated_ids)

                    if next_token is None:
                        break
                    if next_token == self.backbone.endtools_id:
                        reached_end = True
                        break

                    tools_tokens.append(next_token)
                    steps += 1

                decode_tokens = [self.backbone.tools_id] + tools_tokens
                tools_output = self.backbone.eagle_tokenizer.decode(
                    decode_tokens, skip_special_tokens=True
                ).strip()

                if reached_end:
                    decode_tokens.append(self.backbone.endtools_id)

                token_device = backbone_inputs['eagle_input_ids'].device if 'eagle_input_ids' in backbone_inputs else self.device
                backbone_outputs['generated_tool_token_ids'] = torch.tensor(
                    decode_tokens, dtype=torch.long, device=token_device
                )

                batch_size = backbone_outputs[BACKBONE_FEATURE_KEY].shape[0]
                zero_actions = torch.zeros(
                    (batch_size, self.action_horizon, self.action_dim),
                    dtype=self.action_head.dtype,
                    device=self.device,
                )
                action_head_outputs = BatchFeature(data={ACTION_KEY: zero_actions})
                action_head_outputs['action_head_skipped'] = True
            else:
                raise ValueError(f'Unexpected route token id: {token_id}')

        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs, tools_output, past_key_values

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
            if not isinstance(key, str) or not key.endswith(suffix):
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
                inputs[key] = inputs[key].reshape(-1, inputs[key].size(2), inputs[key].size(3))

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

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")

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

        pretrained_model = super().from_pretrained(
            local_model_path, local_model_path=local_model_path, **kwargs
        )

        pretrained_model.backbone.set_trainable_parameters(
            tune_visual=tune_visual, tune_llm=tune_llm
        )
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )
        return pretrained_model


# register
AutoConfig.register("gr00t_n1_5", GR00T_N1_5_Config)
AutoModel.register(GR00T_N1_5_Config, GR00T_N1_5)
