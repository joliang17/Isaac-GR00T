#!/bin/bash

#SBATCH --job-name=groot_libero_action_1trace
#SBATCH --output=groot_libero_action_1trace.log
#SBATCH --error=groot_libero_action_1trace.log
#SBATCH --time=48:00:00
#SBATCH --account=cml-director
#SBATCH --partition=cml-director
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G


source /etc/profile.d/modules.sh
module add cuda/12.4.1
module add gcc/11.2.0

source /fs/nexus-scratch/yliang17/miniconda3/bin/activate gr00t

WANDB_PROJECT="vla_tooluse"
TASK_NAME="stage2_onlyembA_nextstep_skip_action"

# model_path="stage2_freezeembB_skip_action"
# model_folder="${model_path}"

# model_path='stage2_onlyembA_skip_action'
# model_folder="${model_path}"

model_path="stage2_onlyembA_nextstep_skip_action"
model_folder="${model_path}"

# python scripts/gr00t_finetune.py --dataset-path /fs/nexus-scratch/yliang17/Research/VLA/saved_folder/dataset/traj_video_both_v2_noid --num-gpus 1 --batch-size 4 --lora_llm_model --lora_rank=4 --window_length=10 --data_config libero_traj_arms --video_backend torchvision_av --save_steps 10 --output_dir "/fs/nexus-scratch/yliang17/Research/VLA/saved_folder/checkpoint/tt" --action_ds_ratio=0.5 --dataloader_num_workers=0 --windowing_mode fixed --toolend_upsample_ratio=20  --run_name ${TASK_NAME} --tune_tool_end --tune_special_A --tune_special_B  --base_model_path "/fs/nexus-scratch/yliang17/Research/VLA/saved_folder/checkpoint/skill_noid_textonly_2emb_allstep_merged/checkpoint-10000" 

python scripts/gr00t_finetune.py --dataset-path /fs/nexus-scratch/yliang17/Research/VLA/saved_folder/dataset/traj_video_both_v2_noid --num-gpus 1 --batch-size 4 --lora_llm_model --window_length=10 --data_config libero_traj_arms --video_backend torchvision_av --save_steps 20 --output_dir "/fs/nexus-scratch/yliang17/Research/VLA/saved_folder/checkpoint/tt" --base_model_path "/fs/nexus-scratch/yliang17/Research/VLA/saved_folder/checkpoint/stage2_freezeembB_nextstep_skip_action_merged/checkpoint-6000"
