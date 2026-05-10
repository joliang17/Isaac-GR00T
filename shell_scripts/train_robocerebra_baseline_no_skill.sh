#!/bin/bash

#SBATCH --job-name=rc_base_noskill
#SBATCH --output=slurm_output/rc_base_noskill.log
#SBATCH --error=slurm_output/rc_base_noskill.log
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
EVAL_SKILL_JSON="${EVAL_SKILL_JSON:-/fs/nexus-scratch/yliang17/Research/VLA/AtomicVLA/data_split_json/robocerebra_study_table_holdout_case1_12_lerobot_addskill.json}"

TASK_NAME="${TASK_NAME:-robocerebra_study_table_baseline_no_skill}"
OUTPUT_DIR="${OUTPUT_DIR:-/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/${TASK_NAME}}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-nvidia/GR00T-N1.5-3B}"

BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_STEPS="${MAX_STEPS:-30000}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
TRAIN_EVAL_MAX_SAMPLES="${TRAIN_EVAL_MAX_SAMPLES:-1024}"
TRAIN_EVAL_FRACTION="${TRAIN_EVAL_FRACTION:-0.01}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
RESUME="${RESUME:-false}"

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

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting RoboCerebra baseline without skill"
echo "DATASET_PATH=${DATASET_PATH}"
echo "EVAL_DATASET_PATH=${EVAL_DATASET_PATH}"
echo "EVAL_SKILL_JSON=${EVAL_SKILL_JSON}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "RESUME=${RESUME}"

EXTRA_ARGS=()
if [[ "${RESUME}" == "true" || "${RESUME}" == "1" || "${RESUME}" == "yes" ]]; then
  EXTRA_ARGS+=(--resume)
fi

python scripts/gr00t_finetune.py \
  --num-gpus 1 \
  --dataset-path "${DATASET_PATH}" \
  --train_eval_dataset_path "${TRAIN_EVAL_DATASET_PATH}" \
  --train_eval_max_samples "${TRAIN_EVAL_MAX_SAMPLES}" \
  --train_eval_fraction "${TRAIN_EVAL_FRACTION}" \
  --eval_dataset_path "${EVAL_DATASET_PATH}" \
  --eval_skill_annotation_path "${EVAL_SKILL_JSON}" \
  --windowing_mode "step" \
  --batch-size "${BATCH_SIZE}" \
  --data_config "libero_original" \
  --video_backend "torchvision_av" \
  --base_model_path "${BASE_MODEL_PATH}" \
  --save_steps "${SAVE_STEPS}" \
  --max_steps "${MAX_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${TASK_NAME}" \
  --tune_diffusion_model \
  --do_eval \
  "${EXTRA_ARGS[@]}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished RoboCerebra baseline without skill"
