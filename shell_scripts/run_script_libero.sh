#!/bin/bash

#SBATCH --job-name=libero_training_skill
#SBATCH --output=slurm_output/libero_training_skill.log
#SBATCH --error=slurm_output/libero_training_skill.log
#SBATCH --time=48:00:00
#SBATCH --account=cml-director
#SBATCH --partition=cml-director
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:a100:1 
#SBATCH --nodelist=cml32
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G

source /fs/nexus-scratch/yliang17/miniconda3/bin/activate gr00t

source /etc/profile.d/modules.sh
module add cuda/12.4.1
module add gcc/11.2.0

export WANDB_PROJECT="vla_tooluse"
export CACHE_DIR="/fs/nexus-projects/wilddiffusion/cache"

DATASET=libero_base
TASK_NAME=new_vlm

PYTHONPATH=/fs/nexus-scratch/yliang17/Research/VLA/GR00T_vlm \
python scripts/gr00t_finetune.py \
  --num-gpus 1 \
  --dataset-path "/fs/nexus-scratch/yliang17/Research/VLA/LIBERO_10_lerobot" \
  --batch-size 16 \
  --data_config "libero_original" \
  --video_backend "torchvision_av" \
  --save_steps 30000 \
  --max_steps 60000 \
  --output_dir "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/${TASK_NAME}" \
  --run_name "${TASK_NAME}" \
  --tune_diffusion_model
