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

model_path="stage2_onlyembA_nextstep_skip_action"
model_folder="${model_path}"


python scripts/gr00t_finetune.py --dataset-path /fs/nexus-projects/wilddiffusion/vla/LIBERO/traj_video_both_v5_noid --num-gpus 1 --batch-size 2 --window_length=5 --data_config libero_traj_arms_2 --video_backend torchvision_av --save_steps 200 --output_dir "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/tt" --dataloader_num_workers=0 --windowing_mode "sliding_prefix" --run_name "tt_nextstep_skip_action" --base_model_path "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/stage2_nextstep_skip_action_toolhead_only_v6_rerun/checkpoint-6000"  --tune_tool_end --do_eval

# --frame_type="key"
--do_eval
#  --tune_special_A 


