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

import json
import os
CACHE_DIR = "/fs/nexus-projects/wilddiffusion/cache"
CACHE_DIR = os.getenv("CACHE_DIR", CACHE_DIR)

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

import torch
import tyro
from transformers import TrainingArguments
from torch.utils.data import Subset
from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING
from gr00t.utils.peft import get_lora_model, list_trainable_parameter_names, tie_all_special_weights
torch.autograd.set_detect_anomaly(True)


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

    skill_annotation_path: str | None = None
    """Path to JSON skill-annotation file. When set, overrides per-frame parquet annotation.
    Format: {episode_id: {segments: [{start_frame, end_frame, skill, ...}]}}
    If None, falls back to reading annotation.step_description from the parquet dataset."""

    skill_label_type: str = 'skill'
    """Which JSON field to use as the skill label for [TOOLS] frames.
    'skill': full phrase, e.g. "pick up the white mug".
    'primary_action_verb': atomic verb only, e.g. "pick"."""

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

    use_task_router: bool = False
    """Add soft-weighted task router + learnable embedding bank appended to VLM features; freeze all other params."""

    num_task_emb_slots: int = 8
    """Number of learnable embedding slots in the task router bank (K)."""

    router_hidden_dim: int = 256
    """Hidden dim of the router MLP (backbone_dim -> hidden -> K)."""

    router_diversity_coeff: float = 0.01
    """Coefficient for the router load-balancing loss that encourages all K slots to be used equally. Set 0 to disable."""

    router_lang_tail: int = 0
    """If > 0, pool only the last N valid backbone tokens for routing (focuses on language instruction tokens). 0 = pool all valid tokens."""

    use_task_adapter: bool = False
    """Add FiLM task-conditioned adapters after state_encoder, action_encoder, and action_decoder. Requires use_task_router=True."""

    # Skill embedding (Stage 1 + Stage 2)
    use_skill_emb: bool = False
    """Enable skill embedding module: MLP classifier (Stage 1) + learnable skill token concat (Stage 2). Gated; default=False keeps full backward compatibility."""

    skill_emb_dim: int = 256
    """Dimension of each learnable skill embedding vector in the bank."""

    skill_proj_hidden_dim: int = 256
    """Hidden dim of the MLP projector used in the skill classifier."""

    skill_clf_coeff: float = 1.0
    """Weight for cross-entropy skill classification loss."""

    skill_div_coeff: float = 0.01
    """Weight for orthogonality diversity loss on skill embedding bank."""

    skill_norm_coeff: float = 0.01
    """Weight for non-zero norm loss on skill embedding bank (prevents collapse to zero)."""

    tune_skill_clf: bool = False
    """Stage 1: train only skill MLP projector + classifier (VLM frozen). use_skill_emb must be True."""

    tune_skill_emb: bool = False
    """Stage 2: train skill embedding bank + projection + DiT (VLM + classifier frozen). use_skill_emb must be True."""

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
# skill vocab helpers
#####################################################################################


def _discover_skill_vocab(annotation_path: str, skill_label_type: str) -> list[str]:
    """Scan skill annotation JSON and return a sorted list of unique skill labels."""
    with open(annotation_path) as f:
        data = json.load(f)
    verbs: set[str] = set()
    for ep_val in data.values():
        for seg in ep_val.get("segments", []):
            if skill_label_type == "primary_action_verb":
                v = seg.get("primary_action_verb")
            else:
                v = seg.get("skill") or seg.get("primary_action_verb")
            if v:
                verbs.add(v.strip())
    return sorted(verbs)


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
    transforms = data_config_cls.transform()

    # 1.2 data loader: we will use either single dataset or mixture dataset
    if len(config.dataset_path) == 1:
        train_dataset = LeRobotSingleDataset(
            dataset_path=config.dataset_path[0],
            modality_configs=modality_configs,
            transforms=transforms,
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
            skill_annotation_path=config.skill_annotation_path,
            skill_label_type=config.skill_label_type,
            # action_only=config.tune_diffusion_model,
        )

        if config.do_eval:
            eval_sanity_set = Subset(train_dataset, indices=range(int(0.01 * len(train_dataset))))
            # eval_sanity_set = Subset(train_dataset, indices=range(20))
        else:
            eval_sanity_set = None
    else:
        single_datasets = []
        for p in config.dataset_path:
            assert os.path.exists(p), f"Dataset path {p} does not exist"
            dataset = LeRobotSingleDataset(
                dataset_path=p,
                modality_configs=modality_configs,
                transforms=transforms,
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
                skill_annotation_path=config.skill_annotation_path,
                skill_label_type=config.skill_label_type,
                # action_only=config.tune_diffusion_model,
            )
            single_datasets.append(dataset)

        train_dataset = LeRobotMixtureDataset(
            data_mixture=[
                (dataset, 1.0)  # we will use equal weights for all datasets
                for dataset in single_datasets
            ],
            mode="train",
            balance_dataset_weights=config.balance_dataset_weights,
            balance_trajectory_weights=config.balance_trajectory_weights,
            seed=42,
            metadata_config={
                "percentile_mixing_method": "weighted_average",
            },
        )
        print(f"Loaded {len(single_datasets)} datasets, with {config.dataset_path} ")

        if config.do_eval:
            eval_sanity_set = Subset(single_datasets[0], indices=range(int(0.01 * len(single_datasets[0]))))
            # eval_sanity_set = Subset(train_dataset, indices=range(20))
        else:
            eval_sanity_set = None

    # ------------ step 2: load model ------------
    # First, get the data config to determine action horizon
    data_action_horizon = len(data_config_cls.action_indices)

    # Discover skill vocab from annotation JSON (needed before model init when use_skill_emb=True)
    skill_vocab: list[str] | None = None
    num_skills = 1
    if config.use_skill_emb:
        assert config.skill_annotation_path is not None, \
            "--skill_annotation_path required when --use_skill_emb is set"
        skill_vocab = _discover_skill_vocab(config.skill_annotation_path, config.skill_label_type)
        num_skills = len(skill_vocab)
        print(f"[SkillEmb] Discovered {len(skill_vocab)} skills: {skill_vocab}")

    # Load model
    # training parameters
    pred_nextstep = False
    if 'nextstep' in config.run_name:
        pred_nextstep = True

    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
        tune_llm=config.tune_llm,  # backbone's LLM
        tune_visual=config.tune_visual,  # backbone's vision tower
        tune_projector=config.tune_projector,  # action head's projector
        tune_diffusion_model=config.tune_diffusion_model,  # action head's DiT
        tune_special_A=config.tune_special_A,  # backbone's embedding
        tune_special_B=config.tune_special_B,  # backbone's embedding
        tune_tool_end=config.tune_tool_end,
        tune_trace_projector=config.tune_trace_projector,
        use_skill_emb=config.use_skill_emb,
        num_skills=num_skills,
        tune_skill_clf=config.tune_skill_clf,
        tune_skill_emb=config.tune_skill_emb,
        skill_vocab=skill_vocab,
        skill_emb_dim=config.skill_emb_dim,
        skill_proj_hidden_dim=config.skill_proj_hidden_dim,
        skill_clf_coeff=config.skill_clf_coeff,
        skill_div_coeff=config.skill_div_coeff,
        skill_norm_coeff=config.skill_norm_coeff,
        pred_nextstep=pred_nextstep
    )

    # Update action_horizon to match data config
    # Need to recreate action head with correct config since it was initialized with old config
    if data_action_horizon != model.action_head.config.action_horizon:
        print(
            f"Recreating action head with action_horizon {data_action_horizon} (was {model.action_head.config.action_horizon})"
        )

        # Update the action head config
        new_action_head_config = model.action_head.config
        new_action_head_config.action_horizon = data_action_horizon

        # Import the FlowmatchingActionHead class
        from gr00t.model.action_head.flow_matching_action_head import (
            FlowmatchingActionHead,
        )

        # Create new action head with updated config
        new_action_head = FlowmatchingActionHead(new_action_head_config)

        # Copy the weights from the old action head to the new one
        new_action_head.load_state_dict(model.action_head.state_dict(), strict=False)

        # Replace the action head
        model.action_head = new_action_head

        # Update model config AND the action_head_cfg dictionary that gets saved
        model.config.action_horizon = data_action_horizon
        model.action_horizon = data_action_horizon
        model.config.action_head_cfg["action_horizon"] = data_action_horizon

        # Set trainable parameters for the new action head
        model.action_head.set_trainable_parameters(
            tune_projector=config.tune_projector,
            tune_diffusion_model=config.tune_diffusion_model,
            tune_skill_clf=config.tune_skill_clf,
            tune_skill_emb=config.tune_skill_emb,
        )

    # ADDED: reload special embed after pretrained
    model_emb = model.backbone.eagle_model.language_model.model.embed_tokens
    lm_head = model.backbone.eagle_model.language_model.lm_head
    if (getattr(model_emb, "special_embedding_A", None) or getattr(model_emb, "special_embedding_B", None)) and config.base_model_path == "nvidia/GR00T-N1.5-3B":
        # --- Re-init special token embeddings ---
        with torch.no_grad():
            base_weight = model_emb.base_embedding.weight

            # compute mean in fp32 for stability
            mean_vec = base_weight.detach().to(torch.float32).mean(dim=0, keepdim=True)
            
            # --- Handle Group A ---
            if hasattr(model_emb, "special_embedding_A") and model_emb.special_embedding_A.weight.size(0) > 0:
                special_layer_to_init_A = model_emb.special_embedding_A.weight
                special_head_to_init_A = lm_head.special_head_A.weight
                
                mean_vec_A = mean_vec.to(special_layer_to_init_A.device, dtype=special_layer_to_init_A.dtype)
                num_embeddings_A = special_layer_to_init_A.shape[0]
                
                special_layer_to_init_A.copy_(mean_vec_A.repeat(num_embeddings_A, 1))
                special_head_to_init_A.copy_(mean_vec_A.repeat(num_embeddings_A, 1))
                print(f"Re-initialized {num_embeddings_A} special tokens (Group A) with base embedding mean.")
                
            # --- Handle Group B ---
            if hasattr(model_emb, "special_embedding_B") and model_emb.special_embedding_B.weight.size(0) > 0:
                special_layer_to_init_B = model_emb.special_embedding_B.weight
                special_head_to_init_B = lm_head.special_head_B.weight

                mean_vec_B = mean_vec.to(special_layer_to_init_B.device, dtype=special_layer_to_init_B.dtype)
                num_embeddings_B = special_layer_to_init_B.shape[0]

                special_layer_to_init_B.copy_(mean_vec_B.repeat(num_embeddings_B, 1))
                special_head_to_init_B.copy_(mean_vec_B.repeat(num_embeddings_B, 1))
                print(f"Re-initialized {num_embeddings_B} special tokens (Group B) with base embedding mean.")
    
    # Initialize the tool head so it doesn't output garbage initially
    # if config.tune_tool_end and hasattr(model.backbone, "tool_end_head") and config.base_model_path == "nvidia/GR00T-N1.5-3B":
    if config.tune_tool_end and hasattr(model.backbone, "tool_end_head") and 'toolhead' not in config.base_model_path:
        with torch.no_grad():
            # model.backbone.tool_end_head.weight.data.normal_(mean=0.0, std=0.02)
            # model.backbone.tool_end_head.bias.data.zero_()
            # model.backbone.tool_end_head.bias.data[1] = -5.0 
            
            head = model.backbone.tool_end_head
            for layer in [head.fc1, head.fc2]:
                torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    layer.bias.data.zero_()

            head.norm.weight.data.fill_(1.0)
            head.norm.bias.data.zero_()
            head.classifier.weight.data.normal_(mean=0.0, std=0.01)
            head.classifier.bias.data.zero_()
            head.classifier.bias.data[1] = -5.0 

        print("Initialized tool_end_head with custom weights.")

    if config.tune_trace_projector and hasattr(model.backbone, "trace_projector") and 'trace' not in config.base_model_path:
        with torch.no_grad():
            model.backbone.trace_projector.weight.data.normal_(mean=0.0, std=0.02)
            model.backbone.trace_projector.bias.data.zero_()
            model.backbone.trace_projector.bias.data[1] = -5.0 

        print("Initialized trace_projector with custom weights.")

    # Set the model's compute_dtype to bfloat16
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    # --- Task Router: inject soft-weighted embedding bank into the action head ---
    if config.use_task_router:
        import torch.nn as nn
        backbone_emb_dim = model.action_head.config.backbone_embedding_dim  # 1536
        K = config.num_task_emb_slots
        H = config.router_hidden_dim

        # Update action_head config so it serializes correctly with the checkpoint
        model.action_head.config.use_task_router = True
        model.action_head.config.num_task_emb_slots = K
        model.action_head.config.router_hidden_dim = H
        model.action_head.config.router_diversity_coeff = config.router_diversity_coeff
        model.action_head.config.router_lang_tail = config.router_lang_tail
        # Sync to the top-level config dict that gets written to config.json
        model.config.action_head_cfg["use_task_router"] = True
        model.config.action_head_cfg["num_task_emb_slots"] = K
        model.config.action_head_cfg["router_hidden_dim"] = H
        model.config.action_head_cfg["router_diversity_coeff"] = config.router_diversity_coeff
        model.config.action_head_cfg["router_lang_tail"] = config.router_lang_tail

        # Inject new modules into action head
        model.action_head.task_emb_bank = nn.Embedding(K, backbone_emb_dim)
        model.action_head.router = nn.Sequential(
            nn.Linear(backbone_emb_dim, H),
            nn.ReLU(),
            nn.Linear(H, K),
        )

        # Move to same device as model
        device = next(model.parameters()).device
        model.action_head.task_emb_bank = model.action_head.task_emb_bank.to(device)
        model.action_head.router = model.action_head.router.to(device)

        # Initialize
        nn.init.normal_(model.action_head.task_emb_bank.weight, mean=0.0, std=0.02)
        for layer in [model.action_head.router[0], model.action_head.router[2]]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        # Freeze entire model, then unfreeze ONLY the router + bank
        model.requires_grad_(False)
        model.action_head.task_emb_bank.requires_grad_(True)
        model.action_head.router.requires_grad_(True)

        print(f"[TaskRouter] Injected task_emb_bank({K}, {backbone_emb_dim}) + router MLP (hidden={H})")
        print(f"[TaskRouter] All model params frozen; trainable: task_emb_bank + router only")

    # --- Task Adapter: inject FiLM adapters conditioned on the task embedding ---
    if config.use_task_adapter:
        assert config.use_task_router, "--use-task-adapter requires --use-task-router"
        from gr00t.model.action_head.flow_matching_action_head import TaskConditionedAdapter

        backbone_emb_dim = model.action_head.config.backbone_embedding_dim
        input_emb_dim = model.action_head.config.input_embedding_dim
        action_dim = model.action_head.config.action_dim
        device = next(model.parameters()).device

        # Update action_head config for correct checkpoint serialization
        model.action_head.config.use_task_adapter = True
        model.config.action_head_cfg["use_task_adapter"] = True

        # Inject and initialize adapters
        model.action_head.state_adapter   = TaskConditionedAdapter(input_emb_dim, backbone_emb_dim).to(device)
        model.action_head.action_adapter  = TaskConditionedAdapter(input_emb_dim, backbone_emb_dim).to(device)
        model.action_head.decoder_adapter = TaskConditionedAdapter(action_dim,    backbone_emb_dim).to(device)

        # Adapters are always trainable; unfreeze them on top of whatever the router block set
        model.action_head.state_adapter.requires_grad_(True)
        model.action_head.action_adapter.requires_grad_(True)
        model.action_head.decoder_adapter.requires_grad_(True)

        print(f"[TaskAdapter] Injected FiLM adapters (feature_dim={input_emb_dim}/{input_emb_dim}/{action_dim}, task_emb_dim={backbone_emb_dim})")
        print(f"[TaskAdapter] Trainable: task_emb_bank + router + state_adapter + action_adapter + decoder_adapter")

    # --- Skill Embedding: modules are initialized in __init__ via patched config in from_pretrained ---
    if config.use_skill_emb:
        assert skill_vocab is not None  # guaranteed by discovery step above
        ah = model.action_head
        if "cls_stage" not in config.base_model_path:
            # Fresh initialization for skill params not present in the base pretrained checkpoint
            for module in ah.skill_proj.modules():
                if isinstance(module, torch.nn.Linear):
                    torch.nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
            torch.nn.init.xavier_uniform_(ah.skill_clf.weight)
            torch.nn.init.zeros_(ah.skill_clf.bias)
            print(f"[SkillEmb] Skill modules (skill_proj / skill_clf) freshly initialized (num_skills={num_skills})")
        
        if "cls_stage2" not in config.base_model_path:
            torch.nn.init.normal_(ah.skill_emb_bank.weight, mean=0.0, std=0.02)
            torch.nn.init.xavier_uniform_(ah.skill_emb_proj.weight)
            torch.nn.init.zeros_(ah.skill_emb_proj.bias)
            print(f"[SkillEmb] Skill modules (skill_emb_bank / skill_emb_proj) freshly initialized (num_skills={num_skills})")

    train_action_head = False
    if 'both' in config.dataset_path[0] and 'skip_action' not in config.run_name:
        train_action_head = True

    if config.lora_rank > 0:
        # normal lora training (only for action_head / full model)
        model = get_lora_model(
            model,
            rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            freeze_embeddings=config.freeze_embeddings, 
            train_action_head=train_action_head,
            tune_special_A=config.tune_special_A,
            tune_special_B=config.tune_special_B,
            tune_tool_end=config.tune_tool_end,
            tune_trace_projector=config.tune_trace_projector,
        )
    else:
        # tie model weight
        tie_all_special_weights(model)
        # check wether head & embeddings shared the same weight
        if config.windowing_mode not in ('step', 'skill_action', 'skill_cls'):
            model.action_head.requires_grad_(train_action_head)
        elif config.windowing_mode == 'skill_action':
            model.action_head.requires_grad_(config.tune_diffusion_model)

    # skill_action_v2: enable skill_action_mode so split_by_img_id includes [TOOLS] frames
    if config.windowing_mode == 'skill_action':
        model.backbone.skill_action_mode = True
        print("[skill_action_v2] backbone.skill_action_mode = True")

    _ = list_trainable_parameter_names(model)
    import pdb;pdb.set_trace()

    # 2.1 modify training args
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=config.run_name,
        remove_unused_columns=False,
        deepspeed="",
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=False,
        # dataloader_prefetch_factor=config.dataloader_prefetch_factor,
        dataloader_persistent_workers=config.dataloader_num_workers > 0,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10.0,
        num_train_epochs=300,
        max_steps=config.max_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        # evaluation_strategy="no",
        save_total_limit=5,
        report_to=config.report_to,
        seed=42,
        # do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
        max_grad_norm=config.grad_norm,

        # --- EVALUATION SETTINGS ---
        do_eval=config.do_eval, 
        eval_strategy="steps" if config.do_eval else "no", 
        eval_steps=config.save_steps, 
        per_device_eval_batch_size=config.batch_size, 
        eval_accumulation_steps=1,
        # ---------------------------
    )

    # 2.2 run experiment
    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
        eval_dataset=eval_sanity_set,
    )

    # 2.3 run experiment
    # experiment.eval()
    experiment.train()


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)

    # Print the tyro config
    print("\n" + "=" * 50)
    print("GR00T FINE-TUNING CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Validate GPU configuration
    assert (
        config.num_gpus <= available_gpus
    ), f"Number of GPUs requested ({config.num_gpus}) is greater than the available GPUs ({available_gpus})"
    assert config.num_gpus > 0, "Number of GPUs must be greater than 0"
    print(f"Using {config.num_gpus} GPUs")

    if config.num_gpus == 1:
        # Single GPU mode - set CUDA_VISIBLE_DEVICES=0
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # Run the script normally
        main(config)
    else:
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(config)
        else:
            # Multi-GPU mode - use torchrun
            script_path = Path(__file__).absolute()
            # Remove any existing CUDA_VISIBLE_DEVICES from environment
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            # Use subprocess.run instead of os.system
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",  # default to 1 node for now
                str(script_path),
            ]

            # Convert config to command line arguments
            for key, value in vars(config).items():
                if isinstance(value, bool):
                    # For boolean values, use --flag or --no-flag format
                    if value:
                        cmd.append(f"--{key.replace('_', '-')}")
                    else:
                        cmd.append(f"--no-{key.replace('_', '-')}")
                else:
                    # For non-boolean values, use --key value format
                    cmd.append(f"--{key.replace('_', '-')}")

                    # if the value is a list (e.g. dataset_path), we need to add each element in the list
                    if isinstance(value, list):
                        for v in value:
                            cmd.append(str(v))
                    else:
                        cmd.append(str(value))
            print("Running torchrun command: ", cmd)
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)
