#!/bin/bash

#SBATCH --job-name=libero_training
#SBATCH --output=slurm_output/libero_training.log
#SBATCH --error=slurm_output/libero_training.log
#SBATCH --time=48:00:00
#SBATCH --account=cml-director
#SBATCH --partition=cml-director
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:a100:1 
#SBATCH --nodelist=cml32
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G

source /fs/nexus-scratch/yliang17/miniconda3/bin/activate gr00t

cd /fs/nexus-scratch/yliang17/Research/VLA/GR00T

source /etc/profile.d/modules.sh
module add cuda/12.4.1
module add gcc/11.2.0

export WANDB_PROJECT="vla_tooluse"
export CACHE_DIR="/fs/nexus-projects/wilddiffusion/cache"

DATASET=libero_base
TASK_NAME=ibero_action_trace

  # --dataset-path "/fs/nexus-scratch/yliang17/Research/VLA/LIBERO_10_lerobot" \

python scripts/gr00t_finetune.py \
  --dataset-path "/fs/nexus-projects/wilddiffusion/vla/LIBERO/libero_base_action_trace" \
  --num-gpus 1 \
  --windowing_mode "step" \
  --batch-size 16 \
  --data_config "libero_trace" \
  --video_backend "torchvision_av" \
  --save_steps 30000 \
  --max_steps 60000 \
  --output_dir "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/${TASK_NAME}" \
  --run_name ${TASK_NAME} \
  --tune_diffusion_model \
  --tune_trace_projector
  # --dataloader_num_workers 0

# python scripts/gr00t_finetune.py \
#   --dataset-path "/fs/nexus-projects/wilddiffusion/vla/LIBERO/libero_base_action_unnorm" \
#   --num-gpus 1 \
#   --windowing_mode "step" \
#   --batch-size 16 \
#   --data_config "libero_original" \
#   --video_backend "torchvision_av" \
#   --save_steps 10000 \
#   --max_steps 30000 \
#   --output_dir "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/libero_base_action_unnorm" \
#   --run_name libero_base_action_unnorm \
#   --tune_diffusion_model 