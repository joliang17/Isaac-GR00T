#!/bin/bash

#SBATCH --job-name=robocerebra_st_cls
#SBATCH --output=slurm_output/robocerebra_st_cls.log
#SBATCH --error=slurm_output/robocerebra_st_cls.log
#SBATCH --time=12:00:00
#SBATCH --account=cml-director
#SBATCH --partition=cml-director
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G

set -eo pipefail

source /fs/nexus-scratch/yliang17/miniconda3/bin/activate gr00t

cd /fs/nexus-scratch/yliang17/Research/VLA/GR00T

source /etc/profile.d/modules.sh
module add cuda/12.8.1
module add gcc/11.2.0

export WANDB_PROJECT="${WANDB_PROJECT:-vla_tooluse}"
export CACHE_DIR="${CACHE_DIR:-/fs/nexus-projects/wilddiffusion/cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

SKILL_JSON="${SKILL_JSON:-/fs/nexus-scratch/yliang17/Research/VLA/AtomicVLA/data_split_json/robocerebra_study_table_train_case13_plus_lerobot_addskill.json}"
TRAIN_EVAL_SKILL_JSON="${TRAIN_EVAL_SKILL_JSON:-${SKILL_JSON}}"
EVAL_SKILL_JSON="${EVAL_SKILL_JSON:-/fs/nexus-scratch/yliang17/Research/VLA/AtomicVLA/data_split_json/robocerebra_study_table_holdout_case1_12_lerobot_addskill.json}"
DATASET_PATH="${DATASET_PATH:-/fs/nexus-projects/wilddiffusion/vla/atomic_data/robocerebra_study_table_train_case13_plus}"
TRAIN_EVAL_DATASET_PATH="${TRAIN_EVAL_DATASET_PATH:-${DATASET_PATH}}"
EVAL_DATASET_PATH="${EVAL_DATASET_PATH:-/fs/nexus-projects/wilddiffusion/vla/atomic_data/robocerebra_study_table_holdout_case1_12}"
TRAIN_EVAL_MAX_SAMPLES="${TRAIN_EVAL_MAX_SAMPLES:-1024}"
TRAIN_EVAL_FRACTION="${TRAIN_EVAL_FRACTION:-0.01}"
TASK_NAME="${TASK_NAME:-robocerebra_study_table_case13_plus_stage1}"

python scripts/gr00t_finetune.py \
  --num-gpus 1 \
  --dataset-path "${DATASET_PATH}" \
  --windowing_mode "skill_cls" \
  --batch-size 16 \
  --data_config "libero_original" \
  --video_backend "torchvision_av" \
  --save_steps 100 \
  --max_steps 3000 \
  --output_dir "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/${TASK_NAME}" \
  --run_name "${TASK_NAME}" \
  --skill_annotation_path "${SKILL_JSON}" \
  --train_eval_dataset_path "${TRAIN_EVAL_DATASET_PATH}" \
  --train_eval_skill_annotation_path "${TRAIN_EVAL_SKILL_JSON}" \
  --train_eval_max_samples "${TRAIN_EVAL_MAX_SAMPLES}" \
  --train_eval_fraction "${TRAIN_EVAL_FRACTION}" \
  --eval_dataset_path "${EVAL_DATASET_PATH}" \
  --eval_skill_annotation_path "${EVAL_SKILL_JSON}" \
  --skill_label_type "primary_action_verb" \
  --skill_vocab "add" "close" "heat" "mix" "move" "open" "pick" "place" "pour" "return" "store" "tilt" "turn" \
  --use_skill_emb \
  --tune_skill_clf \
  --do_eval
