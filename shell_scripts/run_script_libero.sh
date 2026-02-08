#!/bin/bash

#SBATCH --job-name=libero_training_2
#SBATCH --output=/fs/nexus-scratch/yliang17/Research/VLA/GR00T/slurm_output/libero_training_2.log
#SBATCH --error=/fs/nexus-scratch/yliang17/Research/VLA/GR00T/slurm_output/libero_training_2.log
#SBATCH --time=48:00:00
#SBATCH --account=cml-director
#SBATCH --partition=cml-director
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

source /fs/nexus-scratch/yliang17/miniconda3/bin/activate gr00t

cd /fs/nexus-scratch/yliang17/Research/VLA/GR00T

source /etc/profile.d/modules.sh
module add cuda/12.4.1
module add gcc/11.2.0

export WANDB_PROJECT="vla_tooluse"
export CACHE_DIR="/fs/nexus-scratch/yliang17/Research/cache"

DATASET=libero_base
TASK_NAME=libero_training_2

python scripts/gr00t_finetune.py \
  --dataset-path "/fs/nexus-scratch/yliang17/Research/VLA/LIBERO_10_lerobot" \
  --num-gpus 1 \
  --windowing_mode "step" \
  --batch-size 16 \
  --data_config "libero_original" \
  --video_backend "torchvision_av" \
  --save_steps 30000 \
  --max_steps 60000 \
  --output_dir "/fs/nexus-scratch/yliang17/Research/VLA/GR00T/saved_checkpoint/${TASK_NAME}" \
  --run_name ${TASK_NAME} \
  --tune_diffusion_model \
  # --dataloader_num_workers 0
