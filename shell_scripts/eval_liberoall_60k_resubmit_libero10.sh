#!/bin/bash
#SBATCH --job-name=eval_resub_l10
#SBATCH --output=slurm_output/eval_resub_l10.log
#SBATCH --error=slurm_output/eval_resub_l10.log
#SBATCH --time=48:00:00
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

SCRIPT_DIR="/fs/nexus-scratch/yliang17/Research/VLA/GR00T/shell_scripts"
source "${SCRIPT_DIR}/eval_libero_wait_common.sh"

LABEL="liberoall_256_full_60k_resubmit"
CKPT="/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/${LABEL}/checkpoint-60000"
EXPECTED_STEP=60000

wait_for_checkpoint "${CKPT}" "${EXPECTED_STEP}" "${LABEL}"

expected_episodes=$((LIBERO_10_TASKS * NUM_TRIAL))
for seed in "${SEEDS[@]}"; do
    for horizon in "${HORIZONS[@]}"; do
        result_path="$(standard_result_path "${CKPT}" "${seed}" "${horizon}")"
        if result_complete "${result_path}" "${expected_episodes}"; then
            echo "[$(timestamp)] SKIP completed LIBERO-10 eval: seed=${seed} horizon=${horizon} result=${result_path}"
        else
            run_eval "${CKPT}" "${seed}" "${horizon}"
        fi
    done
done

echo "[$(timestamp)] Finished LIBERO-10 eval for ${LABEL}"
