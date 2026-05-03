#!/bin/bash

#SBATCH --job-name=libero_training_quater
#SBATCH --output=slurm_output/libero_training_quater.log
#SBATCH --error=slurm_output/libero_training_quater.log
#SBATCH --time=48:00:00
#SBATCH --account=scavenger 
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa5000:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G

source /fs/nexus-scratch/yliang17/miniconda3/bin/activate gr00t

cd /fs/nexus-scratch/yliang17/Research/VLA/GR00T

source /etc/profile.d/modules.sh
module add cuda/12.4.1
module add gcc/11.2.0

export WANDB_PROJECT="vla_tooluse"
export CACHE_DIR="/fs/nexus-projects/wilddiffusion/cache"
# export CUDA_VISIBLE_DEVICES=0


TASK_NAME=libero10_256_quater_30k

python scripts/gr00t_finetune.py \
  --num-gpus 1 \
  --dataset-path "/fs/nexus-projects/wilddiffusion/vla/atomic_data/libero_atomic_10_quater" \
  --windowing_mode "step" \
  --batch-size 32 \
  --data_config "libero_original" \
  --video_backend "torchvision_av" \
  --save_steps 20000 \
  --max_steps 60000 \
  --output_dir "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/${TASK_NAME}" \
  --run_name "${TASK_NAME}" \
  --tune_diffusion_model \