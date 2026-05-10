#!/bin/bash

set -eo pipefail

GR00T_DIR="${GR00T_DIR:-/fs/nexus-scratch/yliang17/Research/VLA/GR00T}"
CONDA_SH="${CONDA_SH:-/fs/nexus-scratch/yliang17/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-gr00t}"

DATASET_PATH="${DATASET_PATH:-/fs/nexus-projects/wilddiffusion/vla/atomic_data/robocerebra_study_table_train_case13_plus}"
TRAIN_EVAL_DATASET_PATH="${TRAIN_EVAL_DATASET_PATH:-${DATASET_PATH}}"
EVAL_DATASET_PATH="${EVAL_DATASET_PATH:-/fs/nexus-projects/wilddiffusion/vla/atomic_data/robocerebra_study_table_holdout_case1_12}"

SKILL_JSON="${SKILL_JSON:-/fs/nexus-scratch/yliang17/Research/VLA/AtomicVLA/data_split_json/robocerebra_study_table_train_case13_plus_lerobot_addskill.json}"
TRAIN_EVAL_SKILL_JSON="${TRAIN_EVAL_SKILL_JSON:-${SKILL_JSON}}"
EVAL_SKILL_JSON="${EVAL_SKILL_JSON:-/fs/nexus-scratch/yliang17/Research/VLA/AtomicVLA/data_split_json/robocerebra_study_table_holdout_case1_12_lerobot_addskill.json}"

TASK_NAME="${TASK_NAME:-robocerebra_study_table_case13_plus_stage1}"
OUTPUT_DIR="${OUTPUT_DIR:-/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/${TASK_NAME}}"
WANDB_PROJECT="${WANDB_PROJECT:-vla_tooluse}"
CACHE_DIR="${CACHE_DIR:-/fs/nexus-projects/wilddiffusion/cache}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TRAIN_EVAL_MAX_SAMPLES="${TRAIN_EVAL_MAX_SAMPLES:-1024}"
TRAIN_EVAL_FRACTION="${TRAIN_EVAL_FRACTION:-0.01}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-0}"
DATA_GEN_PID="${DATA_GEN_PID:-}"

required_paths=(
  "${DATASET_PATH}"
  "${EVAL_DATASET_PATH}"
  "${SKILL_JSON}"
  "${EVAL_SKILL_JSON}"
)

wait_for_data_generation() {
  if [[ -z "${DATA_GEN_PID}" ]]; then
    return
  fi

  echo "Waiting for data generation PID ${DATA_GEN_PID} to finish..."
  while kill -0 "${DATA_GEN_PID}" 2>/dev/null; do
    sleep "${WAIT_SECONDS}"
  done
  echo "Data generation PID ${DATA_GEN_PID} is no longer running."
}

wait_for_paths() {
  local waited=0
  while true; do
    local missing=()
    for path in "${required_paths[@]}"; do
      if [[ ! -e "${path}" ]]; then
        missing+=("${path}")
      fi
    done

    if [[ "${#missing[@]}" -eq 0 ]]; then
      echo "All required datasets and annotation files are present."
      return
    fi

    echo "Waiting for required paths:"
    printf '  %s\n' "${missing[@]}"

    if [[ "${MAX_WAIT_SECONDS}" -gt 0 && "${waited}" -ge "${MAX_WAIT_SECONDS}" ]]; then
      echo "Timed out after ${waited}s waiting for data." >&2
      return 1
    fi

    sleep "${WAIT_SECONDS}"
    waited=$((waited + WAIT_SECONDS))
  done
}

cd "${GR00T_DIR}"

if [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
  module add cuda/12.8.1 || true
  module add gcc/11.2.0 || true
fi

if [[ -f "${CONDA_SH}" ]]; then
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
else
  source /fs/nexus-scratch/yliang17/miniconda3/bin/activate "${CONDA_ENV}"
fi

export WANDB_PROJECT CACHE_DIR CUDA_VISIBLE_DEVICES

wait_for_data_generation
wait_for_paths

echo "Starting RoboCerebra stage-1 training at $(date)"
echo "Output dir: ${OUTPUT_DIR}"
echo "Train eval subset: max=${TRAIN_EVAL_MAX_SAMPLES}, fraction=${TRAIN_EVAL_FRACTION}"

python scripts/gr00t_finetune.py \
  --num-gpus 1 \
  --dataset-path "${DATASET_PATH}" \
  --train_eval_dataset_path "${TRAIN_EVAL_DATASET_PATH}" \
  --train_eval_skill_annotation_path "${TRAIN_EVAL_SKILL_JSON}" \
  --train_eval_max_samples "${TRAIN_EVAL_MAX_SAMPLES}" \
  --train_eval_fraction "${TRAIN_EVAL_FRACTION}" \
  --eval_dataset_path "${EVAL_DATASET_PATH}" \
  --windowing_mode "skill_cls" \
  --batch-size 16 \
  --data_config "libero_original" \
  --video_backend "torchvision_av" \
  --save_steps 500 \
  --max_steps 1000 \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${TASK_NAME}" \
  --skill_annotation_path "${SKILL_JSON}" \
  --eval_skill_annotation_path "${EVAL_SKILL_JSON}" \
  --skill_label_type "primary_action_verb" \
  --skill_vocab "add" "close" "heat" "mix" "move" "open" "pick" "place" "pour" "return" "store" "tilt" "turn" \
  --use_skill_emb \
  --tune_skill_clf \
  --do_eval

echo "Training finished at $(date)"
