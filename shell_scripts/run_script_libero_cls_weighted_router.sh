#!/bin/bash

#SBATCH --job-name=libero_training_cls_weighted
#SBATCH --output=slurm_output/libero_training_cls_weighted.log
#SBATCH --error=slurm_output/libero_training_cls_weighted.log
#SBATCH --time=36:00:00
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
export CUDA_VISIBLE_DEVICES=1

SKILL_JSON="/fs/nexus-scratch/yliang17/Research/VLA/AtomicVLA/data_split_json/libero_lerobot_addskill_10_half.json"

TASK_NAME=libero10_256_half_cls_weighted_stage1_v1

python scripts/gr00t_finetune.py \
  --num-gpus 1 \
  --dataset-path "/fs/nexus-projects/wilddiffusion/vla/atomic_data/libero_atomic_10_half" \
  --windowing_mode "skill_cls" \
  --batch-size 32 \
  --data_config "libero_original" \
  --video_backend "torchvision_av" \
  --save_steps 1000 \
  --max_steps 6000 \
  --output_dir "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/${TASK_NAME}" \
  --run_name "${TASK_NAME}" \
  --skill_annotation_path "${SKILL_JSON}" \
  --skill_label_type "primary_action_verb" \
  --skill_vocab "close" "open" "pick" "place" "turn" \
  --use_skill_emb \
  --use_weighted_skill_router \
  --tune_skill_clf \
  --do_eval \
#   # --dataloader_num_workers 0


TASK_NAME=libero10_256_half_cls_weighted_stage2_v1

python scripts/gr00t_finetune.py \
  --num-gpus 1 \
  --dataset-path "/fs/nexus-projects/wilddiffusion/vla/atomic_data/libero_atomic_10_half" \
  --windowing_mode "skill_cls" \
  --batch-size 32 \
  --data_config "libero_original" \
  --video_backend "torchvision_av" \
  --save_steps 20000 \
  --max_steps 60000 \
  --base_model_path "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/libero10_256_half_cls_weighted_stage1_v1/checkpoint-6000" \
  --output_dir "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/${TASK_NAME}" \
  --run_name "${TASK_NAME}" \
  --skill_annotation_path "${SKILL_JSON}" \
  --skill_label_type "primary_action_verb" \
  --skill_vocab "close" "open" "pick" "place" "turn" \
  --use_skill_emb \
  --use_weighted_skill_router \
  --tune_skill_emb \
  --tune_diffusion_model \
  --do_eval

  # --dataloader_num_workers 0


# python3 libero_scripts/libero_eval.py \
#     --task_suite_name libero_10 \
#     --num_steps_wait 10 \
#     --num_trials_per_task 10 \
#     --port 5555 \
#     --headless True \
#     --model_path "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/libero10_256_half_cls_weighted_stage2_v1/checkpoint-60000" \
#     --skill_vocab "close" "open" "pick" "place" "turn" \
#     --embodiment_tag new_embodiment \
#     --data_config libero_original \
#     --denoising_steps 8 \
#     --action_horizon 1 \
