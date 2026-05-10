#!/bin/bash

#SBATCH --job-name=rc_skill
#SBATCH --output=slurm_output/rc_skill.log
#SBATCH --error=slurm_output/rc_skill.log
#SBATCH --time=48:00:00
#SBATCH --account=cml-director
#SBATCH --partition=cml-director
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -eo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/nexus-scratch/yliang17/Research/VLA/GR00T}"
CONDA_SH="${CONDA_SH:-/fs/nexus-scratch/yliang17/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-gr00t}"

DATASET_PATH="${DATASET_PATH:-/fs/nexus-projects/wilddiffusion/vla/atomic_data/robocerebra_study_table_train_case13_plus}"
TRAIN_EVAL_DATASET_PATH="${TRAIN_EVAL_DATASET_PATH:-${DATASET_PATH}}"
EVAL_DATASET_PATH="${EVAL_DATASET_PATH:-/fs/nexus-projects/wilddiffusion/vla/atomic_data/robocerebra_study_table_holdout_case1_12}"

SKILL_JSON="${SKILL_JSON:-/fs/nexus-scratch/yliang17/Research/VLA/AtomicVLA/data_split_json/robocerebra_study_table_train_case13_plus_lerobot_addskill.json}"
TRAIN_EVAL_SKILL_JSON="${TRAIN_EVAL_SKILL_JSON:-${SKILL_JSON}}"
EVAL_SKILL_JSON="${EVAL_SKILL_JSON:-/fs/nexus-scratch/yliang17/Research/VLA/AtomicVLA/data_split_json/robocerebra_study_table_holdout_case1_12_lerobot_addskill.json}"

STAGE1_NAME="${STAGE1_NAME:-robocerebra_study_table_skill_stage1}"
STAGE2_NAME="${STAGE2_NAME:-robocerebra_study_table_skill_stage2_llm_init}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint}"
STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-${CHECKPOINT_ROOT}/${STAGE1_NAME}}"
STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-${CHECKPOINT_ROOT}/${STAGE2_NAME}}"
STAGE1_BASE_MODEL_PATH="${STAGE1_BASE_MODEL_PATH:-nvidia/GR00T-N1.5-3B}"
STAGE2_BASE_MODEL_PATH="${STAGE2_BASE_MODEL_PATH:-${STAGE1_OUTPUT_DIR}/checkpoint-${STAGE1_MAX_STEPS:-3000}}"

BATCH_SIZE="${BATCH_SIZE:-16}"
STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-3000}"
STAGE1_SAVE_STEPS="${STAGE1_SAVE_STEPS:-500}"
STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:-30000}"
STAGE2_SAVE_STEPS="${STAGE2_SAVE_STEPS:-5000}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
TRAIN_EVAL_MAX_SAMPLES="${TRAIN_EVAL_MAX_SAMPLES:-1024}"
TRAIN_EVAL_FRACTION="${TRAIN_EVAL_FRACTION:-0.01}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
RESUME="${RESUME:-false}"
SKIP_STAGE1_IF_COMPLETE="${SKIP_STAGE1_IF_COMPLETE:-true}"
INIT_SKILL_EMB_FROM_LLM="${INIT_SKILL_EMB_FROM_LLM:-true}"
LLM_SKILL_INIT_SCALE="${LLM_SKILL_INIT_SCALE:-0.02}"

export WANDB_PROJECT="${WANDB_PROJECT:-vla_tooluse}"
export CACHE_DIR="${CACHE_DIR:-/fs/nexus-projects/wilddiffusion/cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "${REPO_ROOT}"
mkdir -p slurm_output

if [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
  module add cuda/12.8.1 || true
  module add gcc/11.2.0 || true
  module add ffmpeg/7.1 || true
fi

if [[ -f "${CONDA_SH}" ]]; then
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
else
  source /fs/nexus-scratch/yliang17/miniconda3/bin/activate "${CONDA_ENV}"
fi

COMMON_SKILL_ARGS=(
  --skill_annotation_path "${SKILL_JSON}"
  --train_eval_skill_annotation_path "${TRAIN_EVAL_SKILL_JSON}"
  --eval_skill_annotation_path "${EVAL_SKILL_JSON}"
  --skill_label_type "primary_action_verb"
  --skill_vocab "add" "close" "heat" "mix" "move" "open" "pick" "place" "pour" "return" "store" "tilt" "turn"
  --use_skill_emb
)

COMMON_DATA_ARGS=(
  --num-gpus 1
  --dataset-path "${DATASET_PATH}"
  --train_eval_dataset_path "${TRAIN_EVAL_DATASET_PATH}"
  --train_eval_max_samples "${TRAIN_EVAL_MAX_SAMPLES}"
  --train_eval_fraction "${TRAIN_EVAL_FRACTION}"
  --eval_dataset_path "${EVAL_DATASET_PATH}"
  --windowing_mode "skill_cls"
  --batch-size "${BATCH_SIZE}"
  --data_config "libero_original"
  --video_backend "torchvision_av"
  --learning_rate "${LEARNING_RATE}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
)

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting RoboCerebra skill Stage 1 classifier"
echo "DATASET_PATH=${DATASET_PATH}"
echo "SKILL_JSON=${SKILL_JSON}"
echo "STAGE1_OUTPUT_DIR=${STAGE1_OUTPUT_DIR}"
echo "STAGE2_OUTPUT_DIR=${STAGE2_OUTPUT_DIR}"
echo "RESUME=${RESUME}"
echo "SKIP_STAGE1_IF_COMPLETE=${SKIP_STAGE1_IF_COMPLETE}"
echo "INIT_SKILL_EMB_FROM_LLM=${INIT_SKILL_EMB_FROM_LLM}"
echo "LLM_SKILL_INIT_SCALE=${LLM_SKILL_INIT_SCALE}"

STAGE1_DONE_CKPT="${STAGE1_OUTPUT_DIR}/checkpoint-${STAGE1_MAX_STEPS}"
if [[ "${SKIP_STAGE1_IF_COMPLETE}" == "true" && -d "${STAGE1_DONE_CKPT}" ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Skipping Stage 1; found ${STAGE1_DONE_CKPT}"
else
  python scripts/gr00t_finetune.py \
    "${COMMON_DATA_ARGS[@]}" \
    "${COMMON_SKILL_ARGS[@]}" \
    --base_model_path "${STAGE1_BASE_MODEL_PATH}" \
    --save_steps "${STAGE1_SAVE_STEPS}" \
    --max_steps "${STAGE1_MAX_STEPS}" \
    --output_dir "${STAGE1_OUTPUT_DIR}" \
    --run_name "${STAGE1_NAME}" \
    --tune_skill_clf \
    --do_eval
fi

if [[ ! -d "${STAGE2_BASE_MODEL_PATH}" ]]; then
  echo "Expected Stage 1 checkpoint does not exist: ${STAGE2_BASE_MODEL_PATH}" >&2
  exit 1
fi

EXTRA_STAGE2_ARGS=()
if [[ "${RESUME}" == "true" || "${RESUME}" == "1" || "${RESUME}" == "yes" ]]; then
  EXTRA_STAGE2_ARGS+=(--resume)
fi
if [[ "${INIT_SKILL_EMB_FROM_LLM}" == "true" || "${INIT_SKILL_EMB_FROM_LLM}" == "1" || "${INIT_SKILL_EMB_FROM_LLM}" == "yes" ]]; then
  EXTRA_STAGE2_ARGS+=(--init-skill-emb-from-llm --llm-skill-init-scale "${LLM_SKILL_INIT_SCALE}")
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting RoboCerebra skill Stage 2 embedding/action training"
echo "STAGE2_BASE_MODEL_PATH=${STAGE2_BASE_MODEL_PATH}"
echo "STAGE2_OUTPUT_DIR=${STAGE2_OUTPUT_DIR}"

python scripts/gr00t_finetune.py \
  "${COMMON_DATA_ARGS[@]}" \
  "${COMMON_SKILL_ARGS[@]}" \
  --base_model_path "${STAGE2_BASE_MODEL_PATH}" \
  --save_steps "${STAGE2_SAVE_STEPS}" \
  --max_steps "${STAGE2_MAX_STEPS}" \
  --output_dir "${STAGE2_OUTPUT_DIR}" \
  --run_name "${STAGE2_NAME}" \
  --tune_skill_emb \
  --tune_diffusion_model \
  --do_eval \
  "${EXTRA_STAGE2_ARGS[@]}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished RoboCerebra skill experiment"
