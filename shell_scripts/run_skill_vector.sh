#!/bin/bash

#SBATCH --job-name=skill_vector
#SBATCH --output=skill_vector.log
#SBATCH --error=skill_vector.log
#SBATCH --time=48:00:00
#SBATCH --account=cml-director
#SBATCH --partition=cml-director
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G


source /etc/profile.d/modules.sh
module add cuda/12.9.1
module add gcc/11.2.0

source /fs/nexus-scratch/yliang17/miniconda3/bin/activate gr00t

WANDB_PROJECT="vla_tooluse"
TASK_NAME="debug_eval"


python scripts/skill_vector.py --dataset-path /fs/nexus-projects/wilddiffusion/vla/LIBERO/traj_text_only/ --num-gpus 1 --batch-size 2 --window_length=5 --data_config libero_traj_arms_2 --video_backend torchvision_av --save_steps 200 --output_dir "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/tt" --dataloader_num_workers=0 --windowing_mode "sliding_prefix" --run_name "tt_nextstep_skip_action" --base_model_path "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/stage1_nextstep_3ds_0301_merged/checkpoint-10000"  --tune_tool_end

