#!/bin/bash

#SBATCH --job-name=libero_training_skill2
#SBATCH --output=slurm_output/libero_training_skill2.log
#SBATCH --error=slurm_output/libero_training_skill2.log
#SBATCH --time=48:00:00
#SBATCH --account=cml-director
#SBATCH --partition=cml-director
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:a100:1 
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G

source /fs/nexus-scratch/yliang17/miniconda3/bin/activate gr00t

cd /fs/nexus-scratch/yliang17/Research/VLA/GR00T

source /etc/profile.d/modules.sh
module add cuda/12.8.1
module add gcc/11.2.0

export WANDB_PROJECT="vla_tooluse"
export CACHE_DIR="/fs/nexus-projects/wilddiffusion/cache"
export CUDA_VISIBLE_DEVICES=0

SKILL_JSON="/fs/nexus-scratch/yliang17/Research/VLA/AtomicVLA/data_split_json/libero_lerobot_addskill_10_half.json"
LEROBOT_DATASET="/fs/nexus-projects/wilddiffusion/vla/atomic_data/libero_atomic_10_half"

SKILL_JSON="/fs/nexus-scratch/yliang17/Research/VLA/AtomicVLA/data_split_json/libero_lerobot_addskill_10_half_theother.json"
LEROBOT_DATASET="/fs/nexus-projects/wilddiffusion/vla/atomic_data/libero_atomic_10_half_theother"

# TASK_NAME=libero10_256_half_cls_stage1

# python scripts/gr00t_finetune.py \
#   --num-gpus 1 \
#   --dataset-path "/fs/nexus-projects/wilddiffusion/vla/atomic_data/libero_atomic_10_half" \
#   --windowing_mode "skill_cls" \
#   --batch-size 32 \
#   --data_config "libero_original" \
#   --video_backend "torchvision_av" \
#   --save_steps 1000 \
#   --max_steps 6000 \
#   --output_dir "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/${TASK_NAME}" \
#   --run_name "${TASK_NAME}" \
#   --skill_annotation_path "${SKILL_JSON}" \
#   --skill_label_type "primary_action_verb" \
#   --use_skill_emb \
#   --tune_skill_clf \
#   --do_eval \
#   # --dataloader_num_workers 0


TASK_NAME=libero10_256_half_cls_stage1_eval

python scripts/gr00t_finetune.py \
  --num-gpus 1 \
  --dataset-path "${LEROBOT_DATASET}" \
  --windowing_mode "skill_cls" \
  --batch-size 32 \
  --data_config "libero_original" \
  --video_backend "torchvision_av" \
  --save_steps 20000 \
  --max_steps 60000 \
  --base_model_path "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/libero10_256_half_cls_stage1/checkpoint-6000" \
  --output_dir "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/${TASK_NAME}" \
  --run_name "${TASK_NAME}" \
  --skill_annotation_path "${SKILL_JSON}" \
  --skill_label_type "primary_action_verb" \
  --skill_vocab "close" "pick" "place" "turn" \
  --use_skill_emb \
  --tune_skill_emb \
  --tune_skill_clf \
  --tune_diffusion_model \
  --do_eval

  # --dataloader_num_workers 0
