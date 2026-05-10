#!/bin/bash

#SBATCH --job-name=rc_skill_router_eval
#SBATCH --output=slurm_output/rc_skill_router_eval.log
#SBATCH --error=slurm_output/rc_skill_router_eval.log
#SBATCH --time=12:00:00
#SBATCH --account=cml-director
#SBATCH --partition=cml-director
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -eo pipefail

REPO_ROOT="/fs/nexus-scratch/yliang17/Research/VLA/GR00T"
cd "${REPO_ROOT}"
mkdir -p slurm_output hidden_states

source /fs/nexus-scratch/yliang17/miniconda3/bin/activate gr00t

if [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
  module add cuda/12.8.1 || true
  module add gcc/11.2.0 || true
fi

export CACHE_DIR="${CACHE_DIR:-/fs/nexus-projects/wilddiffusion/cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SKILL_PREFIX="The robot executes atomic manipulation skills. Your job is to select the NEXT skill the robot should execute. Available skills: add, close, heat, mix, move, open, pick, place, pour, return, store, tilt, turn. Decide the NEXT skill needed. "

python scripts/evaluate_llm_skill_router.py \
  --dataset-path "/fs/nexus-projects/wilddiffusion/vla/atomic_data/robocerebra_study_table_holdout_case1_12" \
  --annotation-path "/fs/nexus-scratch/yliang17/Research/VLA/AtomicVLA/data_split_json/robocerebra_study_table_holdout_case1_12_lerobot_addskill.json" \
  --base-model-path "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/robocerebra_study_table_skill_stage2/checkpoint-30000" \
  --data-config "libero_original" \
  --video-backend "torchvision_av" \
  --skill-label-type "primary_action_verb" \
  --max-samples 256 \
  --skill-prefix "${SKILL_PREFIX}" \
  --output-path "hidden_states/robocerebra_skill_router_eval_zero_shot.json"

python scripts/evaluate_llm_skill_router.py \
  --dataset-path "/fs/nexus-projects/wilddiffusion/vla/atomic_data/robocerebra_study_table_holdout_case1_12" \
  --annotation-path "/fs/nexus-scratch/yliang17/Research/VLA/AtomicVLA/data_split_json/robocerebra_study_table_holdout_case1_12_lerobot_addskill.json" \
  --base-model-path "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/robocerebra_study_table_skill_stage2/checkpoint-30000" \
  --data-config "libero_original" \
  --video-backend "torchvision_av" \
  --skill-label-type "primary_action_verb" \
  --max-samples 256 \
  --skill-prefix "${SKILL_PREFIX}" \
  --calibration linear_probe \
  --output-path "hidden_states/robocerebra_skill_router_eval_linear_probe.json"
