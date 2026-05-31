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
import traceback

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
import pprint
from functools import partial


def preprocess_logits_for_metrics(logits, labels):
    """
    Handles the dictionary output from EagleBackbone.
    Ensures all outputs are Tensors (not None) to prevent HF Trainer crashes.
    """
    # 1. Extract values from the dictionary
    if isinstance(logits, dict) or hasattr(logits, "data"):
        lm_logits = logits.get("logits")
        p_te = logits.get("predicted_tool_end_eval")
        t_te = logits.get("target_tool_end_eval")
        p_sp = logits.get("cur_pred_id_eval")
        l_sp = logits.get("cur_label_id_eval")
        p_tt = logits.get("all_pred_id_eval")
        l_tt = logits.get("all_label_id_eval")
    else:
        lm_logits = logits
        p_te = t_te = p_sp = l_sp = p_tt = l_tt = None

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
    if p_tt is None: p_tt = torch.empty(0, dtype=torch.long, device=device)
    if l_tt is None: l_tt = torch.empty(0, dtype=torch.long, device=device)

    p_skill = logits.get("skill_pred_eval") if (isinstance(logits, dict) or hasattr(logits, "data")) else None
    l_skill = logits.get("skill_label_eval") if (isinstance(logits, dict) or hasattr(logits, "data")) else None
    if p_skill is None: p_skill = torch.empty(0, dtype=torch.long, device=device)
    if l_skill is None: l_skill = torch.empty(0, dtype=torch.long, device=device)

    skill_mse_normal = logits.get("skill_mse_normal_eval") if (isinstance(logits, dict) or hasattr(logits, "data")) else None
    skill_mse_zero = logits.get("skill_mse_zero_eval") if (isinstance(logits, dict) or hasattr(logits, "data")) else None
    skill_mse_shuffle = logits.get("skill_mse_shuffle_eval") if (isinstance(logits, dict) or hasattr(logits, "data")) else None
    skill_mse_zero_delta = logits.get("skill_mse_zero_delta_eval") if (isinstance(logits, dict) or hasattr(logits, "data")) else None
    skill_mse_shuffle_delta = logits.get("skill_mse_shuffle_delta_eval") if (isinstance(logits, dict) or hasattr(logits, "data")) else None
    empty_float = torch.empty(0, dtype=torch.float32, device=device)
    if skill_mse_normal is None: skill_mse_normal = empty_float
    if skill_mse_zero is None: skill_mse_zero = empty_float
    if skill_mse_shuffle is None: skill_mse_shuffle = empty_float
    if skill_mse_zero_delta is None: skill_mse_zero_delta = empty_float
    if skill_mse_shuffle_delta is None: skill_mse_shuffle_delta = empty_float

    # 4. Return as a tuple
    # Note: We include both preds and targets for the toolhead/special tokens
    # because they are filtered/subsampled in the backbone.
    return (
        lm_preds,
        p_te,
        t_te,
        p_sp,
        l_sp,
        p_tt,
        l_tt,
        p_skill,
        l_skill,
        skill_mse_normal,
        skill_mse_zero,
        skill_mse_shuffle,
        skill_mse_zero_delta,
        skill_mse_shuffle_delta,
    )


def compute_tool_end_counts(
    target_tool_end,
    predicted_tool_end,
    cur_label_id_eval,
    skills_end_id,
    tools_id,
    actions_id,
):
    """
    Returns correct & total counts for tool_end under
    three GT label groups.
    """

    # ---- flatten everything ----
    target = np.asarray(target_tool_end).reshape(-1)
    pred = np.asarray(predicted_tool_end).reshape(-1)
    labels = np.asarray(cur_label_id_eval).reshape(-1)

    # correctness mask
    correct_mask = (target == pred)

    # convert id lists → sets (fast lookup)
    skills_set = set(skills_end_id)
    tools_set = set(tools_id)
    actions_set = set(actions_id)

    results = {}

    def compute_group(name, id_set):
        group_mask = np.isin(labels, list(id_set))

        total = group_mask.sum()
        correct = (correct_mask & group_mask).sum()

        results[name] = {
            "correct": int(correct),
            "total": int(total),
            "acc": float(correct / total) if total > 0 else 0.0,
        }

    # ---- compute three groups ----
    compute_group("toolend_skills_end", skills_set)
    compute_group("toolend_tools", tools_set)
    compute_group("toolend_actions", actions_set)

    return results

def compute_metrics(
    eval_preds,
    tune_tool_end=False,
    use_skill_emb=False,
    special_token_ids_A=None,
    special_token_ids_B=None,
    skills_end_id=None,
    tools_id=None,
    actions_id=None,
    tokenizer=None,
):
    """
    Branched evaluation logic:
    - If use_skill_emb: Compute skill classifier accuracy only (VLM backbone frozen).
    - If tune_tool_end: Focus on Binary Classifier accuracy for the heads.
    - If not tune_tool_end: Focus on Token Generation accuracy (Special A/B).
    """
    metrics = {}
    try:
        (
            lm_preds,
            predicted_tool_end,
            target_tool_end,
            cur_pred_id_eval,
            cur_label_id_eval,
            all_pred_id_eval,
            all_label_id_eval,
            skill_pred_eval,
            skill_label_eval,
            skill_mse_normal,
            skill_mse_zero,
            skill_mse_shuffle,
            skill_mse_zero_delta,
            skill_mse_shuffle_delta,
        ), labels = eval_preds

        if len(skill_mse_normal) > 0:
            metrics["skill_mse_normal"] = float(np.asarray(skill_mse_normal).mean())
            metrics["skill_mse_zero"] = float(np.asarray(skill_mse_zero).mean())
            metrics["skill_mse_shuffle"] = float(np.asarray(skill_mse_shuffle).mean())
            metrics["skill_mse_zero_delta"] = float(np.asarray(skill_mse_zero_delta).mean())
            metrics["skill_mse_shuffle_delta"] = float(
                np.asarray(skill_mse_shuffle_delta).mean()
            )

        # Stage 1: VLM backbone frozen — skip token decode and only report skill accuracy
        if use_skill_emb:
            if len(skill_pred_eval) > 0:
                metrics["skill_clf_accuracy"] = float(
                    (skill_pred_eval == skill_label_eval).sum()
                ) / len(skill_label_eval)
            else:
                metrics["skill_clf_accuracy"] = 0.0
            return metrics
        pred_text = tokenizer.batch_decode(all_pred_id_eval, skip_special_tokens=False)
        gt_text = tokenizer.batch_decode(all_label_id_eval, skip_special_tokens=False)
        
        newline_id = tokenizer("\n", add_special_tokens=False)["input_ids"][0]
        gt_groups = []
        pred_groups = []
        start = 0

        for i, tid in enumerate(all_label_id_eval):
            if tid == newline_id:
                gt_chunk = all_label_id_eval[start:i+1]
                pred_chunk = all_pred_id_eval[start:i+1]

                gt_groups.append(tokenizer.decode(gt_chunk))
                pred_groups.append(tokenizer.decode(pred_chunk))

                start = i + 1
        for gt, pred in zip(gt_groups, pred_groups):
            print('-' * 20)
            print(f"Labels: {gt}")
            print(f"Preds : {pred}")

        valid_mask = cur_label_id_eval!=skills_end_id 
        pred_token = cur_pred_id_eval[valid_mask]
        gt_token = cur_label_id_eval[valid_mask]
        if valid_mask.any():
            metrics["special_token"] = (pred_token == gt_token).mean()
        else:
            metrics["special_token"] = 0.0

        # Common mask for valid LM tokens
        if tune_tool_end:
            result = compute_tool_end_counts(target_tool_end, predicted_tool_end, cur_label_id_eval, [skills_end_id], [tools_id], [actions_id])
            metrics['detailed'] = result

            # --- BRANCH 1: AUXILIARY HEAD EVALUATION ---
            metrics["tool_end_total"] = (target_tool_end == predicted_tool_end).mean()
            metrics["tool_end_FP"] = (target_tool_end[target_tool_end==0] == predicted_tool_end[target_tool_end==0]).mean()
            if (target_tool_end==1).any():
                metrics["tool_end_TP"] = (target_tool_end[target_tool_end==1] == predicted_tool_end[target_tool_end==1]).mean()
            else:
                metrics["tool_end_TP"] = 0.0

        # Stage 2 / any stage with skill labels: append skill classifier accuracy
        if len(skill_pred_eval) > 0:
            metrics["skill_clf_accuracy"] = float(
                (skill_pred_eval == skill_label_eval).sum()
            ) / len(skill_label_eval)
    except Exception as e:
        print(e)
        traceback.print_exc()
        # import pdb;pdb.set_trace()
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
                tokenizer = getattr(backbone, "eagle_tokenizer", None)
                action_head = getattr(model, "action_head", None)
                use_skill_emb = getattr(action_head.config, "use_skill_emb", False)
                compute_metrics_func = partial(
                    compute_metrics,
                    tune_tool_end=tune_tool_end,
                    use_skill_emb=use_skill_emb,
                    special_token_ids_A=backbone.special_token_ids_A.cpu().numpy() if hasattr(backbone, "special_token_ids_A") else None,
                    special_token_ids_B=backbone.special_token_ids_B.cpu().numpy() if hasattr(backbone, "special_token_ids_B") else None,
                    skills_end_id=getattr(backbone, "skills_end", None),
                    tools_id=getattr(backbone, "tools_id", None),
                    actions_id=getattr(backbone, "actions_id", None),
                    tokenizer=tokenizer
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
            pprint.pprint(metrics)            
        return metrics
