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

export WANDB_PROJECT="vla_tooluse"
DATASET="action_only_video"
TASK_NAME="groot_libero_action_1trace"

python scripts/gr00t_finetune.py --dataset-path /fs/nexus-projects/wilddiffusion/vla/LIBERO/${DATASET} --num-gpus 1 --lora_rank 32  --lora_alpha 128  --batch-size 4 --lora_llm_model --window_length=10 --data_config libero_traj_arms --video_backend torchvision_av --save_steps 2000 --output_dir /fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/${TASK_NAME} --run_name ${TASK_NAME}

