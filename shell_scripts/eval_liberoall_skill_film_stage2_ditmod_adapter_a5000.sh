#!/bin/bash

# LIBERO-10 + LIBERO-PRO evaluation for
# liberoall_256_full_skill_router_film_stage2_ditmod_adapter.

#SBATCH --array=0-1
#SBATCH --job-name=eval_film_s2_a5000
#SBATCH --output=slurm_output/eval_liberoall_skill_film_stage2_ditmod_adapter_%A_%a.log
#SBATCH --error=slurm_output/eval_liberoall_skill_film_stage2_ditmod_adapter_%A_%a.log
#SBATCH --time=48:00:00
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

REPO_ROOT="/fs/nexus-scratch/yliang17/Research/VLA/GR00T"
cd "${REPO_ROOT}"
mkdir -p slurm_output results

source "${REPO_ROOT}/shell_scripts/eval_libero_wait_common.sh"

LABEL="liberoall_256_full_skill_router_film_stage2_ditmod_adapter"
EXPECTED_STEP=60000
CKPT="/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/${LABEL}/checkpoint-${EXPECTED_STEP}"
SKILL_EVAL_MODE="${SKILL_EVAL_MODE:-normal}"
expected_episodes=$((LIBERO_10_TASKS * NUM_TRIAL))
export GR00T_DEBUG_SKILL_ACTIONS="1"

task_id="${SLURM_ARRAY_TASK_ID:-0}"
if ! [[ "${task_id}" =~ ^[01]$ ]]; then
  echo "SLURM_ARRAY_TASK_ID must be 0 or 1, got: ${task_id}" >&2
  exit 1
fi

wait_for_checkpoint "${CKPT}" "${EXPECTED_STEP}" "${LABEL}"

if [[ "${task_id}" == "0" ]]; then
  echo "[$(timestamp)] LIBERO-10 eval for ${LABEL}; expected_episodes=${expected_episodes}"
  for seed in "${SEEDS[@]}"; do
    for horizon in "${HORIZONS[@]}"; do
      result_path="$(standard_result_path "${CKPT}" "${seed}" "${horizon}" "${SKILL_EVAL_MODE}")"
      if result_complete "${result_path}" "${expected_episodes}"; then
        echo "[$(timestamp)] SKIP completed LIBERO-10 eval: seed=${seed} horizon=${horizon}"
      else
        run_eval "${CKPT}" "${seed}" "${horizon}"
      fi
    done
  done
  echo "[$(timestamp)] Finished LIBERO-10 eval for ${LABEL}"
else
  echo "[$(timestamp)] LIBERO-PRO eval for ${LABEL}; expected_episodes=${expected_episodes}"
  for seed in "${SEEDS[@]}"; do
    for horizon in "${HORIZONS[@]}"; do
      for perturb in "${PERTURBATIONS[@]}"; do
        result_path="$(pro_result_path "${CKPT}" "${seed}" "${horizon}" "${perturb}" "${SKILL_EVAL_MODE}")"
        if result_complete "${result_path}" "${expected_episodes}"; then
          echo "[$(timestamp)] SKIP completed LIBERO-PRO eval: seed=${seed} horizon=${horizon} perturb=${perturb}"
        else
          run_pro_eval "${CKPT}" "${seed}" "${horizon}" "${perturb}"
        fi
      done
    done
  done
  echo "[$(timestamp)] Finished LIBERO-PRO eval for ${LABEL}"
fi
