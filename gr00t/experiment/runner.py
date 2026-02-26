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
from pathlib import Path

import torch
import numpy as np
from transformers import TrainingArguments, set_seed
from torch.utils.data import Subset
from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
from gr00t.experiment.trainer import DualBrainTrainer
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import DefaultDataCollator
from gr00t.utils.experiment import (
    CheckpointFormatCallback,
    safe_save_model_for_hf_trainer,
)
import wandb
from functools import partial


def preprocess_logits_for_metrics(logits, labels):
    """
    Handles the dictionary output from EagleBackbone.
    Ensures all outputs are Tensors (not None) to prevent HF Trainer crashes.
    """
    # 1. Extract values from the dictionary
    if isinstance(logits, dict) or hasattr(logits, "data"):
        lm_logits = logits.get("logits")
        # These are already argmaxed/filtered in the backbone
        p_te = logits.get("predicted_tool_end_eval")
        t_te = logits.get("target_tool_end_eval")
        p_sp = logits.get("cur_pred_id_eval")
        l_sp = logits.get("cur_label_id_eval")
    else:
        lm_logits = logits
        p_te = t_te = p_sp = l_sp = None

    # 2. Handle LM Logits (standard cross-entropy tracking)
    # Shape: [Batch, Seq]
    lm_preds = lm_logits.argmax(dim=-1) if lm_logits is not None else torch.zeros_like(labels)

    # 3. Create Fallbacks for None values
    # Accelerator.gather requires a real tensor. We use empty tensors for batches
    # where no special tokens or tool-head events occurred.
    device = lm_preds.device
    
    if p_te is None: p_te = torch.empty(0, dtype=torch.long, device=device)
    if t_te is None: t_te = torch.empty(0, dtype=torch.long, device=device)
    if p_sp is None: p_sp = torch.empty(0, dtype=torch.long, device=device)
    if l_sp is None: l_sp = torch.empty(0, dtype=torch.long, device=device)

    # 4. Return as a tuple
    # Note: We include both preds and targets for the toolhead/special tokens 
    # because they are filtered/subsampled in the backbone.
    return (lm_preds, p_te, t_te, p_sp, l_sp)

def compute_metrics(
    eval_preds,
    tune_tool_end=False,
    special_token_ids_A=None,
    special_token_ids_B=None,
    skills_end_id=None,
    tools_id=None,
    actions_id=None,
):
    """
    Branched evaluation logic:
    - If tune_tool_end: Focus on Binary Classifier accuracy for the heads.
    - If not tune_tool_end: Focus on Token Generation accuracy (Special A/B).
    """
    (lm_preds, predicted_tool_end, target_tool_end_eval, cur_pred_id_eval, cur_label_id_eval), labels = eval_preds
    metrics = {}
    
    # Common mask for valid LM tokens
    
    if tune_tool_end:
        # --- BRANCH 1: AUXILIARY HEAD EVALUATION ---
        metrics["tool_end_total"] = (target_tool_end_eval == predicted_tool_end).mean()
        if (target_tool_end_eval==1).any():
            metrics["tool_end_true"] = (target_tool_end_eval[target_tool_end_eval==1] == predicted_tool_end[target_tool_end_eval==1]).mean()
        else:
            metrics["tool_end_true"] = 0.0
    else:
        valid_mask = cur_label_id_eval!=skills_end_id 
        pred_token = cur_pred_id_eval[valid_mask]
        gt_token = cur_label_id_eval[valid_mask]
        if valid_mask.any():
            metrics["special_token"] = (pred_token == gt_token).mean()
        else:
            metrics["special_token"] = 0.0
    return metrics

class TrainRunner:
    def __init__(
        self,
        model: GR00T_N1_5,
        training_args: TrainingArguments,
        train_dataset: LeRobotSingleDataset | LeRobotMixtureDataset,
        eval_dataset=None,
        resume_from_checkpoint: bool = False,
    ):
        self.training_args = training_args
        self.output_dir = Path(training_args.output_dir)
        self.exp_cfg_dir = self.output_dir / "experiment_cfg"
        self.exp_cfg_dir.mkdir(parents=True, exist_ok=True)
        self.resume_from_checkpoint = resume_from_checkpoint
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        # Set up training arguments
        training_args.run_name = (
            training_args.output_dir.split("/")[-1]
            if training_args.run_name is None
            else training_args.run_name
        )
        print(f"Run name: {training_args.run_name}")

        data_collator = DefaultDataCollator()

        # Make sure model_dtype and training_args dtype are compatible
        compute_dtype = torch.float16 if training_args.bf16 else torch.float32
        set_seed(training_args.seed)
        # Create trainer
        trainer = self.create_trainer(
            model=model,
            training_args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_dtype=compute_dtype,
        )
        self.trainer = trainer

        # write the metadata to the experiment config dir
        self.rank = int(os.environ.get("RANK", 0))
        if self.rank == 0:
            metadata_json = {}
            if os.path.exists(self.exp_cfg_dir / "metadata.json"):
                with open(self.exp_cfg_dir / "metadata.json", "r") as f:
                    metadata_json = json.load(f)
            if isinstance(train_dataset, LeRobotSingleDataset):
                metadata_json.update(
                    {train_dataset.tag: train_dataset.metadata.model_dump(mode="json")}
                )
            elif isinstance(train_dataset, LeRobotMixtureDataset):
                metadata_json.update(
                    {
                        tag: metadata.model_dump(mode="json")
                        for tag, metadata in train_dataset.merged_metadata.items()
                    }
                )
            else:
                raise ValueError(f"Invalid dataset type: {type(train_dataset)}")
            with open(self.exp_cfg_dir / "metadata.json", "w") as f:
                json.dump(metadata_json, f, indent=4)

        # Set up reporting
        report_to = training_args.report_to
        if report_to == "wandb":
            # Set the environment variables for wandb
            if "WANDB_PROJECT" not in os.environ:
                os.environ["WANDB_PROJECT"] = "gr00t-training"
            if "WANDB_RUN_ID" not in os.environ:
                runtime_id = os.environ.get("RUNTIME_ID", None)
                if runtime_id:
                    os.environ["WANDB_RUN_ID"] = runtime_id
            os.environ["WANDB_DIR"] = training_args.output_dir

            wandb_config_file = self.output_dir / "wandb_config.json"
            with open(wandb_config_file, "w") as f:
                json.dump(
                    {
                        "project": os.environ.get("WANDB_PROJECT", ""),
                        "run_id": os.environ.get("WANDB_RUN_ID", ""),
                    },
                    f,
                )
            training_args.report_to = ["wandb"]
        elif report_to == "azure_ml":
            print("azure_ml logging is enabled.")
        else:  # Default to tensorboard
            tensorboard_dir = Path(training_args.output_dir) / "runs"
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            print(f"TensorBoard logs will be saved to: {tensorboard_dir}")
            training_args.report_to = ["tensorboard"]

    def create_trainer(
        self,
        model,
        training_args,
        train_dataset,
        data_collator,
        compute_dtype,
        eval_dataset=None,
        global_batch_size=None,
    ):
        # Set the gradient accumulation steps if global_batch_size is provided
        if global_batch_size is not None:
            bs = training_args.per_device_train_batch_size
            num_gpus = torch.cuda.device_count()
            grad_acc = max(1, global_batch_size // (bs * num_gpus))
            training_args.gradient_accumulation_steps = grad_acc
            print(
                f"Set global batch size to {global_batch_size}, set gradient accumulation steps to {grad_acc}"
            )

        # ### NEW: Prepare Metrics Logic
        compute_metrics_func = None
        preprocess_logits_func = None

        # We assume 'model.eagle_tokenizer' exists.
        # If your tokenizer is stored differently, adjust 'model.eagle_tokenizer' below.
        if eval_dataset is not None:
            backbone = getattr(model, "backbone", None)
            if backbone:
                # Get the training state of the heads
                tune_tool_end = getattr(backbone, "tune_tool_end", False)
                
                compute_metrics_func = partial(
                    compute_metrics,
                    tune_tool_end=tune_tool_end,
                    special_token_ids_A=backbone.special_token_ids_A.cpu().numpy() if hasattr(backbone, "special_token_ids_A") else None,
                    special_token_ids_B=backbone.special_token_ids_B.cpu().numpy() if hasattr(backbone, "special_token_ids_B") else None,
                    skills_end_id=getattr(backbone, "skills_end", None),
                    tools_id=getattr(backbone, "tools_id", None),
                    actions_id=getattr(backbone, "actions_id", None),
                )
                preprocess_logits_func = preprocess_logits_for_metrics

        # Create the trainer
        trainer = DualBrainTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_dtype=compute_dtype,
            compute_metrics=compute_metrics_func,
            preprocess_logits_for_metrics=preprocess_logits_func,
        )
        # Add checkpoint format callback to ensure experiment_cfg is copied to each checkpoint
        run_name = training_args.run_name
        ckpt_format_callback = CheckpointFormatCallback(
            run_name=run_name, exp_cfg_dir=self.exp_cfg_dir
        )
        trainer.add_callback(ckpt_format_callback)

        # Log dataloader information
        train_dl_len = len(trainer.get_train_dataloader())
        # eval_dl_len = len(trainer.get_eval_dataloader()) # @note (k2): How to manage eval dataloader?

        print(
            f"train dataloader length: {train_dl_len}\n"
            # f"eval dataloader length: {eval_dl_len}\n"
            f"train dataset length: {len(trainer.train_dataset)}\n"
            f"GPU memory before training: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024} GB",
            flush=True,
        )
        return trainer

    def train(self):
        # Start training
        self.trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
        self.trainer.save_state()

        safe_save_model_for_hf_trainer(
            trainer=self.trainer,
            output_dir=self.training_args.output_dir,
        )

    def eval(self):
        print("***** Running Evaluation *****")
        metrics = self.trainer.evaluate()

        if self.rank == 0:
            wandb.log(metrics)
            print(metrics)            
        return metrics