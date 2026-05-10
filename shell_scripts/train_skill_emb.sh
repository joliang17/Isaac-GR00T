#!/bin/bash

#SBATCH --job-name=gr00t_vlm_skill_emb
#SBATCH --output=slurm_output/gr00t_vlm_skill_emb.log
#SBATCH --error=slurm_output/gr00t_vlm_skill_emb.log
#SBATCH --time=48:00:00
#SBATCH --account=cml-director
#SBATCH --partition=cml-director
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=cml32
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G

source /fs/nexus-scratch/yliang17/miniconda3/bin/activate gr00t_vlm

cd /fs/nexus-scratch/yliang17/Research/VLA/GR00T_vlm

source /etc/profile.d/modules.sh
module add cuda/12.4.1
module add gcc/11.2.0

export WANDB_PROJECT="vla_tooluse"
export CACHE_DIR="/fs/nexus-projects/wilddiffusion/cache"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET_PATH="/fs/nexus-projects/wilddiffusion/vla/atomic_data/libero_atomic_10"
SKILL_JSON="/fs/nexus-scratch/yliang17/Research/VLA/AtomicVLA/data_split_json/libero_lerobot_addskill_10.json"
CKPT_ROOT="/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint"

# ---------- Stage 1: train skill classifier MLP only ----------
STAGE1_NAME=libero_qwen4b_skill_emb_stage1
STAGE1_STEPS=6000

PYTHONPATH=/fs/nexus-scratch/yliang17/Research/VLA/GR00T_vlm \
python scripts/gr00t_finetune.py \
  --num-gpus 1 \
  --dataset-path "${DATASET_PATH}" \
  --batch-size 32 \
  --data_config "libero_atomic" \
  --video_backend "torchvision_av" \
  --save_steps 3000 \
  --max_steps ${STAGE1_STEPS} \
  --output_dir "${CKPT_ROOT}/${STAGE1_NAME}" \
  --run_name "${STAGE1_NAME}" \
  --windowing_mode "skill_cls" \
  --skill_annotation_path "${SKILL_JSON}" \
  --skill_label_type "primary_action_verb" \
  --skill_vocab "close" "open" "pick" "place" "turn" \
  --use_skill_emb \
  --num_skills 5 \
  --tune_skill_clf \
  --no-tune_diffusion_model \
  --select_layer 12 \
  --select_clf_layer -1 \
  --skill_div_coeff 0.01 \
  --skill_norm_coeff 0.01

# ---------- Stage 2: train skill embedding bank + DiT from stage1 ckpt ----------
STAGE2_NAME=libero_qwen4b_skill_emb_stage2
STAGE2_STEPS=60000
STAGE1_CKPT="${CKPT_ROOT}/${STAGE1_NAME}/checkpoint-${STAGE1_STEPS}"

PYTHONPATH=/fs/nexus-scratch/yliang17/Research/VLA/GR00T_vlm \
python scripts/gr00t_finetune.py \
  --num-gpus 1 \
  --dataset-path "${DATASET_PATH}" \
  --batch-size 16 \
  --data_config "libero_atomic" \
  --video_backend "torchvision_av" \
  --save_steps 10000 \
  --max_steps ${STAGE2_STEPS} \
  --base_model_path "${STAGE1_CKPT}" \
  --output_dir "${CKPT_ROOT}/${STAGE2_NAME}" \
  --run_name "${STAGE2_NAME}" \
  --windowing_mode "skill_cls" \
  --skill_annotation_path "${SKILL_JSON}" \
  --skill_label_type "primary_action_verb" \
  --skill_vocab "close" "open" "pick" "place" "turn" \
  --use_skill_emb \
  --num_skills 5 \
  --tune_skill_emb \
  --tune_diffusion_model \
  --select_layer 12 \
  --select_clf_layer -1 \
  --skill_div_coeff 0.01 \
  --skill_norm_coeff 0.01
