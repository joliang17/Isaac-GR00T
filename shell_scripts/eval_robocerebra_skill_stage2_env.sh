#!/bin/bash

#SBATCH --job-name=rc_skill_env_eval
#SBATCH --output=slurm_output/rc_skill_env_eval.log
#SBATCH --error=slurm_output/rc_skill_env_eval.log
#SBATCH --time=48:00:00
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

set -eo pipefail

REPO_ROOT="/fs/nexus-scratch/yliang17/Research/VLA/GR00T"
cd "${REPO_ROOT}"
mkdir -p slurm_output logs results rollouts

source /fs/nexus-scratch/yliang17/miniconda3/bin/activate gr00t

if [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
  module add cuda/12.8.1 || true
  module add gcc/11.2.0 || true
  module add ffmpeg/7.1 || true
fi

export CACHE_DIR="${CACHE_DIR:-/fs/nexus-projects/wilddiffusion/cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_GL=egl

CKPT="/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/robocerebra_study_table_skill_stage2/checkpoint-30000"
CASE_IDS="${CASE_IDS:-1-12}"
NUM_TRIALS="${NUM_TRIALS:-1}"
EXEC_HORIZON="${EXEC_HORIZON:-8}"
MAX_STEPS="${MAX_STEPS:-3000}"
SEED="${SEED:-42}"

python -m robocerebra_scripts.robocerebra_eval_skill \
  --model_path "${CKPT}" \
  --case_ids "${CASE_IDS}" \
  --num_trials_per_task "${NUM_TRIALS}" \
  --num_steps_wait 10 \
  --max_steps "${MAX_STEPS}" \
  --embodiment_tag new_embodiment \
  --data_config libero_original \
  --denoising_steps 8 \
  --exec_horizon "${EXEC_HORIZON}" \
  --random_seed "${SEED}"
