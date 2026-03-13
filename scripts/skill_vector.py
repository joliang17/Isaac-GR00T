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
CACHE_DIR = os.getenv("CACHE_DIR", "/fs/nexus-projects/wilddiffusion/cache")

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.environ["HF_MODULES_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal
import pickle
import torch
import numpy as np
import tyro
from PIL import Image
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING
torch.autograd.set_detect_anomaly(True)


def resize_images(image_list, size=(256, 256)): 
    return [img.resize(size, Image.BICUBIC)for img in image_list]


@dataclass
class ArgsConfig:
    """Configuration for GR00T model fine-tuning."""

    # Dataset parameters
    dataset_path: List[str]
    """Path to the dataset directory or directories"""

    output_dir: str = "gr00t_model"
    """Directory to save model checkpoints."""

    run_name: str = "vla_tooluse"
    """Directory to save model checkpoints."""

    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "franka_arms_only"
    """Data configuration name from DATA_CONFIG_MAP, we assume all datasets have the same data config"""

    # Training parameters
    batch_size: int = 32
    """Batch size per GPU for training."""

    window_length: int = 10
    """Maximum number of steps per window (context length)."""

    # --- Window Generation Control ---
    windowing_mode: str = "sliding_prefix"
    """
    Determines how windows are sliced:
    - 'step': original step-wise settings
    - 'fixed': [1-10], [11-20] (Standard non-overlapping blocks)
    - 'block_prefix': [1-2]..[1-10], [11-12].. (Expand prefixes, then jump block)
    - 'sliding_prefix': [1-2]..[1-10], [2-3]..[2-11] (Expand prefixes, slide by 1)
    """
    skill_level: str = "step"
    """
    - 'step': step-wise prediction for skill-level only
    - 'window': window-wise prediction for skill-level only
    """

    frame_type: str = "normal"

    min_seq_len: int = 1
    """Minimum sequence length. Set to 2 to generate '1-2' as the smallest window."""

    stride: int = 1
    """Step size between window starts (mostly for 'sliding_prefix' mode)."""

    # --- Sampling Ratios ---
    skill_inclusion_ratio: float = 0.5
    """Ratio of 'skill' trajectories to include (0.0 to 1.0)."""

    action_ds_ratio: float = 1.0
    """Ratio of action steps to keep within a trajectory (0.0 to 1.0)."""

    toolend_upsample_ratio: float = 1.0
    """
    Upsampling ratio for windows containing [TOOLS_END]. 
    1.0 = no upsampling. 
    3.0 = repeat window 3 times.
    """

    max_steps: int = 10000
    """Maximum number of training steps."""

    num_gpus: int = 1
    """Number of GPUs to use for training."""

    save_steps: int = 1000
    """Number of steps between saving checkpoints."""

    do_eval: bool = False
    """Whether to do sanity check"""

    # Model parameters
    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    """Path or HuggingFace model ID for the base model."""

    tune_llm: bool = False
    """Whether to fine-tune the language model backbone."""

    tune_visual: bool = False
    """Whether to fine-tune the vision tower."""

    tune_projector: bool = True
    """Whether to fine-tune the projector."""

    tune_diffusion_model: bool = False
    """Whether to fine-tune the diffusion model."""

    tune_special_A: bool = False
    """Whether to fine-tune the language model backbone."""

    tune_special_B: bool = False
    """Whether to fine-tune the language model backbone."""

    tune_tool_end: bool = False
    """Whether to fine-tune the language model backbone."""

    tune_trace_projector: bool = False
    """Whether to fine-tune the language model backbone."""

    freeze_embeddings: bool = False
    """Whether to fine-tune the embedding model."""

    init_mode: bool = False
    """Whether to load model from pretrained model or self-model."""

    resume: bool = False
    """Whether to resume from a checkpoint."""

    # Advanced training parameters
    learning_rate: float = 1e-4
    """Learning rate for training."""

    grad_norm: float = 1.0
    """Learning rate for training."""

    weight_decay: float = 1e-5
    """Weight decay for AdamW optimizer."""

    warmup_ratio: float = 0.05
    """Ratio of total training steps used for warmup."""

    lora_rank: int = 0
    """Rank for the LORA model. If 0, no LORA will be used."""

    lora_alpha: int = 16
    """Alpha value for the LORA model."""

    lora_dropout: float = 0.1
    """Dropout rate for the LORA model."""

    lora_full_model: bool = False
    """Whether to use the full model for LORA. If False, only the action head will be trained."""

    lora_llm_model: bool = False
    """Whether to use the LLM model for LORA. If False, only the action head will be trained."""

    train_action_head: bool = False
    """Whether to train action head"""

    dataloader_num_workers: int = 8
    """Number of workers for data loading."""

    gradient_accumulation_steps: int = 1
    """Gradient accumulation steps for training."""

    dataloader_prefetch_factor: int = 4
    """Prefetch factor for data loading."""

    report_to: Literal["wandb", "tensorboard", "azure_ml"] = "wandb"
    """Where to report training metrics (e.g., 'wandb', 'tensorboard', 'azure_ml')."""

    # Data loading parameters
    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag to use for training. e.g. 'new_embodiment', 'gr1'"""

    video_backend: Literal["torchcodec", "decord", "torchvision_av"] = "torchcodec"
    """Video backend to use for training. [torchcodec, decord, torchvision_av]"""

    # Mixture dataset parameters
    balance_dataset_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, we will balance the dataset weights, by multiplying the total trajectory to each dataset"""

    # Mixture dataset parameters
    balance_trajectory_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, sample trajectories within a dataset weighted by their length; otherwise, equal weighting."""


#####################################################################################
# main training function
#####################################################################################


def main(config: ArgsConfig):
    """Main training function."""
    # ------------ step 1: load dataset ------------
    embodiment_tag = EmbodimentTag(config.embodiment_tag)

    # 1.1 modality configs and transforms
    data_config_cls = DATA_CONFIG_MAP[config.data_config]
    modality_configs = data_config_cls.modality_config()
    modality_transform = data_config_cls.transform()

    # 1.2 data loader: we will use either single dataset or mixture dataset
    train_dataset = LeRobotSingleDataset(
        dataset_path=config.dataset_path[0],
        modality_configs=modality_configs,
        transforms=modality_transform,
        embodiment_tag=embodiment_tag,
        video_backend=config.video_backend,
        window_length=config.window_length, 
        windowing_mode=config.windowing_mode,
        skill_level=config.skill_level,
        frame_type=config.frame_type,
        min_seq_len=config.min_seq_len,
        skill_inclusion_ratio=config.skill_inclusion_ratio,
        action_ds_ratio=config.action_ds_ratio,
        toolend_upsample_ratio=config.toolend_upsample_ratio,
    )

    # ------------ step 2: load model ------------
    # First, get the data config to determine action horizon
    data_action_horizon = len(data_config_cls.action_indices)

    # gr00t_policy = GR00TPolicy(host="localhost", port=cfg.port, headless=cfg.headless)
    gr00t_policy = Gr00tPolicy(
        model_path=config.base_model_path,
        modality_config=modality_configs,
        modality_transform=modality_transform,
        embodiment_tag=config.embodiment_tag,
        device="cuda" if torch.cuda.is_available() else "cpu",
        data_config=config.data_config, 
    )

    tools_outputs_correct = []
    tools_outputs_incorrect = []
    # select pick as example
    pick_correct = []
    pick_incorrect = []

    get_action = gr00t_policy.get_action
    midlayer_hidden_states = []
    hook_state = {"hidden": None}
    decoder_layers = gr00t_policy.model.backbone.eagle_model.language_model.model.layers
    mid_layer = decoder_layers[8]

    # def _capture_midlayer_hidden_state(_module, _inputs, output):
    #     hidden_states = output[0] if isinstance(output, tuple) else output
    #     hook_state["hidden"] = hidden_states[:, -1, :].detach().cpu()

    # midlayer_hook = mid_layer.register_forward_hook(_capture_midlayer_hidden_state)

    # get_item = train_dataset.__getitem__
    # with torch.inference_mode():
    #     try:
    #         for idx in range(len(train_dataset)):
    #             ori_item = get_item(idx)
    #             obs_dict = {
    #                 "state.x": np.zeros((1, 1)),
    #                 "state.y": np.zeros((1, 1)),
    #                 "state.z": np.zeros((1, 1)),
    #                 "state.roll": np.zeros((1, 1)),
    #                 "state.pitch": np.zeros((1, 1)),
    #                 "state.yaw": np.zeros((1, 1)),
    #                 "state.gripper": np.zeros((1, 2)),
    #             }

    #             eagle_content = ori_item.get("eagle_content", {})
    #             text_input = eagle_content['text_list']
    #             output_skill = text_input[0].split('[TOOLS]')[-1].split('<|im_end|>')[0].strip()
    #             instruct_text = text_input[0].split('[TOOLS]')[0]
    #             obs_dict['annotation.human.action.task_description'] = [instruct_text, ]

    #             agg_images = eagle_content['image_inputs']
    #             list_top = agg_images[0::2]
    #             list_wri = agg_images[1::2]
    #             list_top = resize_images(list_top)  # (256, 256)
    #             list_wri = resize_images(list_wri)
    #             obs_dict['video.image'] = np.array([np.array(img) for img in list_top])  # [N, H, W, C]
    #             obs_dict['video.wrist_image'] = np.array([np.array(img) for img in list_wri])  # [N, H, W, C]

    #             traj_img_count = len(eagle_content.get("image_inputs", []))
    #             hook_state["hidden"] = None
    #             _, tools_output, _, _ = get_action(obs_dict, img_count=traj_img_count, mode="interleaved",)
    #             if hook_state["hidden"] is not None:
    #                 cur_hidden = hook_state["hidden"]
    #             else:
    #                 continue
                
    #             pred = tools_output.replace('[TOOLS]', '').lower().strip()
    #             grth = output_skill.lower()
    #             if pred == grth:
    #                 tools_outputs_correct.append((idx, grth, pred, cur_hidden))
    #                 if 'pick' in grth:
    #                     pick_correct.append((idx, grth, pred, cur_hidden))
    #             else:
    #                 tools_outputs_incorrect.append((idx, grth, pred, cur_hidden))
    #                 if 'pick' in grth:
    #                     pick_incorrect.append((idx, grth, pred, cur_hidden))
    #     finally:
    #         midlayer_hook.remove()

    # # save
    # with open(f"skill_vector/all_correct.pkl", 'wb') as f:
    #     pickle.dump(tools_outputs_correct, f)
    # with open(f"skill_vector/all_incorrect.pkl", 'wb') as f:
    #     pickle.dump(tools_outputs_incorrect, f)
    # with open(f"skill_vector/pick_correct.pkl", 'wb') as f:
    #     pickle.dump(pick_correct, f)
    # with open(f"skill_vector/pick_incorrect.pkl", 'wb') as f:
    #     pickle.dump(pick_incorrect, f)

    # # calculate task vector for pick
    # pick_vectors = [item[-1] for item in pick_correct]
    # pick_vector = torch.stack(pick_vectors).mean(dim=0)
    # with open(f"skill_vector/pick_vector.pkl", 'wb') as f:
    #     pickle.dump(pick_vector, f)

    # save
    # TODO: change to load pickle
    with open(f"skill_vector/all_correct.pkl", 'rb') as f:
        tools_outputs_correct = pickle.load(f)
    with open(f"skill_vector/all_incorrect.pkl", 'rb') as f:
        tools_outputs_incorrect = pickle.load(f)
    with open(f"skill_vector/pick_correct.pkl", 'rb') as f:
        pick_correct = pickle.load(f)
    with open(f"skill_vector/pick_incorrect.pkl", 'rb') as f:
        pick_incorrect = pickle.load(f)

    # calculate task vector for pick
    with open(f"skill_vector/pick_vector.pkl", 'rb') as f:
        pick_vector = pickle.load(f)

    print(f"Pick correct: {len(pick_correct)}")
    print(f"Pick incorrect: {len(pick_incorrect)}")

    # reapply the skill vector into model
    def steering_hook(module, input, output):
        alpha = 0.5
        hidden_states = output[0] if isinstance(output, tuple) else output
        steer = (alpha * pick_vector).to(hidden_states.device, hidden_states.dtype)
        hidden_states[:, -1, :] += steer
        return output

    midlayer_hook = mid_layer.register_forward_hook(steering_hook)

    better_cnt = 0
    list_new_pred = []
    get_item = train_dataset.__getitem__
    with torch.inference_mode():
        try:
            for failed_case in pick_incorrect:
                idx, grth, pred, _ = failed_case
                ori_item = get_item(idx)
                obs_dict = {
                    "state.x": np.zeros((1, 1)),
                    "state.y": np.zeros((1, 1)),
                    "state.z": np.zeros((1, 1)),
                    "state.roll": np.zeros((1, 1)),
                    "state.pitch": np.zeros((1, 1)),
                    "state.yaw": np.zeros((1, 1)),
                    "state.gripper": np.zeros((1, 2)),
                }

                eagle_content = ori_item.get("eagle_content", {})
                text_input = eagle_content['text_list']
                output_skill = text_input[0].split('[TOOLS]')[-1].split('<|im_end|>')[0].strip()
                instruct_text = text_input[0].split('[TOOLS]')[0]
                obs_dict['annotation.human.action.task_description'] = [instruct_text, ]

                agg_images = eagle_content['image_inputs']
                list_top = agg_images[0::2]
                list_wri = agg_images[1::2]
                list_top = resize_images(list_top)  # (256, 256)
                list_wri = resize_images(list_wri)
                obs_dict['video.image'] = np.array([np.array(img) for img in list_top])  # [N, H, W, C]
                obs_dict['video.wrist_image'] = np.array([np.array(img) for img in list_wri])  # [N, H, W, C]

                traj_img_count = len(eagle_content.get("image_inputs", []))
                _, tools_output, _, _ = get_action(obs_dict, img_count=traj_img_count, mode="interleaved",)
                # import pdb;pdb.set_trace()
                
                pred_new = tools_output.replace('[TOOLS]', '').lower().strip()
                grth = output_skill.lower()

                if pred_new == grth:
                    better_cnt += 1
                else:
                    list_new_pred.append((idx, grth, pred_new))
        finally:
            midlayer_hook.remove()

    print(f"Pick better: {better_cnt}")
    print(f"Pick incorrect: {len(pick_incorrect)}")
    with open(f"skill_vector/failed_case.pkl", 'wb') as f:
        pickle.dump(list_new_pred, f)
    
    for item in list_new_pred[:5]:
        print(item[-1])

if __name__ == "__main__":

    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)
    main(config)
