#!/bin/bash

# Exp A: skill token concatenated BEFORE the action features
#   layout = [state(1) | future(32) | skill(1) | actions(T)]
# This is the default skill-token path (use_skill_film unset).

#SBATCH --job-name=liberoall_skill_skillpos
#SBATCH --output=slurm_output/liberoall_skill_skillpos.log
#SBATCH --error=slurm_output/liberoall_skill_skillpos.log
#SBATCH --time=48:00:00
#SBATCH --account=cml-director
#SBATCH --partition=cml-director
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

REPO_ROOT="/fs/nexus-scratch/yliang17/Research/VLA/GR00T"
cd "${REPO_ROOT}"
mkdir -p slurm_output

source /fs/nexus-scratch/yliang17/miniconda3/bin/activate gr00t

if [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
  module add cuda/12.8.1 || true
  module add gcc/11.2.0 || true
  module add ffmpeg/7.1 || true
fi

export WANDB_PROJECT="${WANDB_PROJECT:-vla_tooluse}"
export CACHE_DIR="${CACHE_DIR:-/fs/nexus-projects/wilddiffusion/cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET="/fs/nexus-projects/wilddiffusion/vla/atomic_data/libero_atomic_all"
SKILL_JSON="/fs/nexus-scratch/yliang17/Research/VLA/AtomicVLA/data_split_json/libero_lerobot_addskill_all.json"
STAGE1_NAME="liberoall_256_full_skill_router_skillpos_stage1"
STAGE2_NAME="liberoall_256_full_skill_router_skillpos_stage2"

python scripts/gr00t_finetune.py \
  --num-gpus 1 \
  --dataset-path "${DATASET}" \
  --windowing_mode "skill_cls" \
  --batch-size 32 \
  --data_config "libero_original" \
  --video_backend "torchvision_av" \
  --save_steps 3000 \
  --max_steps 6000 \
  --output_dir "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/${STAGE1_NAME}" \
  --run_name "${STAGE1_NAME}" \
  --skill_annotation_path "${SKILL_JSON}" \
  --skill_label_type "primary_action_verb" \
  --skill_vocab "close" "open" "pick" "place" "turn" \
  --use_skill_emb \
  --use_weighted_skill_router \
  --tune_skill_clf \
  --do_eval

python scripts/gr00t_finetune.py \
  --num-gpus 1 \
  --dataset-path "${DATASET}" \
  --windowing_mode "skill_cls" \
  --batch-size 32 \
  --data_config "libero_original" \
  --video_backend "torchvision_av" \
  --save_steps 20000 \
  --max_steps 60000 \
  --base_model_path "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/${STAGE1_NAME}/checkpoint-6000" \
  --output_dir "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/${STAGE2_NAME}" \
  --run_name "${STAGE2_NAME}" \
  --skill_annotation_path "${SKILL_JSON}" \
  --skill_label_type "primary_action_verb" \
  --skill_vocab "close" "open" "pick" "place" "turn" \
  --use_skill_emb \
  --use_weighted_skill_router \
  --tune_skill_emb \
  --tune_diffusion_model \
  --do_eval
