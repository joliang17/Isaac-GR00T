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
import glob

from typing import Optional

import numpy as np
import torch
import transformers
import wandb
from torch.utils.data import Dataset, Sampler
from transformers.trainer import (
    ALL_LAYERNORM_LAYERS,
    TRAINER_STATE_NAME,
    TrainerState,
    get_last_checkpoint,
    get_parameter_names,
    is_sagemaker_mp_enabled,
)
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoProcessor
from tempfile import TemporaryDirectory
from peft import PeftModel
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.utils.peft import tie_all_special_weights, enable_special_training


class BaseSampler(Sampler):
    """Sampler for dataset, which enables `set_epoch` for Dataset.
    `set_epoch` will be called by huggingface Trainer at the end of each epoch.
    `shuffle` is also supported for training set shuffling
    """

    def __init__(self, data_source: Dataset, shuffle: bool = False, seed: int = 0):
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # must not add rank here, or randomization will be different for each rank
            return iter(torch.randperm(len(self.data_source), generator=g).tolist())
        return iter(range(len(self.data_source)))

    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.data_source, "set_epoch"):
            # this is important for dataset
            self.data_source.set_epoch(epoch)

    def __len__(self):
        return len(self.data_source)


class DualBrainTrainer(transformers.Trainer):
    def __init__(self, **kwargs):
        self.compute_dtype = kwargs.pop("compute_dtype")
        super().__init__(**kwargs)
        # Allowlist numpy globals for safe RNG state unpickling in PyTorch 2.1+
        torch.serialization.add_safe_globals(
            [np.core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.UInt32DType]
        )

    def _get_train_sampler(self):
        return BaseSampler(self.train_dataset, shuffle=True, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset):
        return BaseSampler(eval_dataset, shuffle=False)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(inputs)
        loss = outputs["loss"]

        # Suppose model also returns other losses
        for key, value in outputs.items():
            if '_loss' in key:
                wandb.log({key: value.item()})

        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = transformers.Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def save_model(self, output_dir: Optional[str], _internal_call: bool):
        def rm_old_ckpt(out_dir, num_limit=3):
            ckpts = sorted(
                glob.glob(os.path.join(os.path.dirname(out_dir), "checkpoint-*")),
                key=os.path.getmtime
            )
            if len(ckpts) > num_limit:
                for ckpt in ckpts[:-num_limit]:  # delete all but last 3
                    print(f"Removing old checkpoint {ckpt}")
                    os.system(f"rm -rf {ckpt}")
            return 

        ## save tuned model separately
        if self.is_deepspeed_enabled:
            state_dict = self.accelerator.get_state_dict(self.deepspeed)
        else:
            state_dict = self.model.state_dict()

        if self.args.should_save:
            tokenizer = self.model.backbone.eagle_tokenizer
            tokenizer.save_pretrained(output_dir)
            # save adapter
            self.model.save_pretrained(output_dir, safe_serialization=True, )
            rm_old_ckpt(output_dir, num_limit=3)
            # p1 = self.model.backbone.eagle_model.language_model.lm_head.special_head.weight
            # p2 = self.model.backbone.eagle_model.language_model.model.embed_tokens.special_embedding.weight
            # print("special emb: Same storage?", p1.data_ptr() == p2.data_ptr())

            # save normal ckpt
            # 1. Load a fresh instance of your base model
            base_model = GR00T_N1_5.from_pretrained("nvidia/GR00T-N1.5-3B", torch_dtype=torch.bfloat16 )
            inference_model = PeftModel.from_pretrained(base_model, output_dir)
            # p3 = inference_model.backbone.eagle_model.language_model.lm_head.special_head.weight
            # p4 = inference_model.backbone.eagle_model.language_model.model.embed_tokens.special_embedding.weight
            # p5 = inference_model.backbone.eagle_model.language_model.lm_head.special_head.original_module.weight
            # print("loaded emb: Same storage?", p3.data_ptr() == p4.data_ptr() == p5.data_ptr())

            merged_model = inference_model.merge_and_unload()
            # p6 = merged_model.backbone.eagle_model.language_model.lm_head.special_head.weight
            # p7 = merged_model.backbone.eagle_model.language_model.model.embed_tokens.special_embedding.weight
            # print("loaded emb: Same storage?", p3.data_ptr() == p4.data_ptr() == p5.data_ptr())

            # 2. Save the final, merged model
            output_dir_merged = output_dir.split('/')
            output_dir_merged[-2] += '_merged'
            output_dir_merged = '/'.join(output_dir_merged)
            merged_model.save_pretrained(output_dir_merged)
            print(f"Saving merged checkpoint {output_dir_merged}")
            rm_old_ckpt(output_dir_merged, num_limit=3)
            return 

    def train(
        self,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
        **kwargs,
    ):
        """Correctly set self.state from checkpoint so get_train_dataloader can read from it."""
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(
                    f"No valid checkpoint found in output directory ({self.args.output_dir})"
                )

        if resume_from_checkpoint is not None:
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
        return super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
