#!/bin/bash

#SBATCH --job-name=libero10_llm_init
#SBATCH --output=slurm_output/libero10_llm_init.log
#SBATCH --error=slurm_output/libero10_llm_init.log
#SBATCH --time=48:00:00
#SBATCH --account=cml-director
#SBATCH --partition=cml-director
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -eo pipefail

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

SKILL_JSON="/fs/nexus-scratch/yliang17/Research/VLA/AtomicVLA/data_split_json/libero_lerobot_addskill_10.json"
STAGE1_CKPT="/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/libero10_256_full_cls_stage1_v2/checkpoint-6000"
TASK_NAME="libero10_256_full_cls_stage2_llm_init"

python scripts/gr00t_finetune.py \
  --num-gpus 1 \
  --dataset-path "/fs/nexus-projects/wilddiffusion/vla/atomic_data/libero_atomic_10" \
  --windowing_mode "skill_cls" \
  --batch-size 32 \
  --data_config "libero_original" \
  --video_backend "torchvision_av" \
  --save_steps 20000 \
  --max_steps 60000 \
  --base_model_path "${STAGE1_CKPT}" \
  --output_dir "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/${TASK_NAME}" \
  --run_name "${TASK_NAME}" \
  --skill_annotation_path "${SKILL_JSON}" \
  --skill_label_type "primary_action_verb" \
  --skill_vocab "close" "open" "pick" "place" "turn" \
  --use_skill_emb \
  --init-skill-emb-from-llm \
  --llm-skill-init-scale 0.02 \
  --tune_skill_emb \
  --tune_diffusion_model
