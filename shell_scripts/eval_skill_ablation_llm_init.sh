#!/bin/bash

#SBATCH --job-name=eval_skill_ablation
#SBATCH --output=slurm_output/eval_skill_ablation_llm_init.log
#SBATCH --error=slurm_output/eval_skill_ablation_llm_init.log
#SBATCH --time=48:00:00
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# Skill-embedding ablation eval for libero10_256_full_cls_stage2_llm_init.
# Runs the skill-router model with the routed skill token replaced by either
# a random skill index per query (--skill_eval_mode shuffle) or a zero vector
# (--skill_eval_mode zero), on standard libero_10 and LIBERO-Pro 'object'.
# Compare overall_success_rate against the existing normal-mode results.

set -euo pipefail

REPO_ROOT="/fs/nexus-scratch/yliang17/Research/VLA/GR00T"
cd "${REPO_ROOT}"
mkdir -p slurm_output results

if [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
  module add cuda/12.8.1 gcc/11.2.0 ffmpeg/7.1 || true
fi

source /fs/nexus-scratch/yliang17/miniconda3/bin/activate gr00t

export CACHE_DIR="${CACHE_DIR:-/fs/nexus-projects/wilddiffusion/cache}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CKPT="/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/libero10_256_full_cls_stage2_llm_init"
EXPECTED_STEP=60000
SEED=42
HORIZONS=(5 10)
MODES=(shuffle zero)
PERTURBATION=object
NUM_TRIAL="${NUM_TRIAL:-10}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

checkpoint_has_expected_step() {
  python - "$1" "${EXPECTED_STEP}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    state = json.load(f)
sys.exit(0 if int(state.get("global_step", -1)) == int(sys.argv[2]) else 1)
PY
}

checkpoint_ready() {
  [[ -d "${CKPT}" ]] || return 1
  [[ -f "${CKPT}/trainer_state.json" ]] || return 1
  [[ -f "${CKPT}/config.json" ]] || return 1
  [[ -f "${CKPT}/model.safetensors.index.json" || -f "${CKPT}/model.safetensors" ]] || return 1
  checkpoint_has_expected_step "${CKPT}/trainer_state.json"
}

run_eval() {
  # $1=horizon $2=mode
  python -m libero_scripts.libero_eval \
    --model_path "${CKPT}" --task_suite_name libero_10 \
    --num_trials_per_task "${NUM_TRIAL}" --num_steps_wait 10 \
    --embodiment_tag new_embodiment --data_config libero_original \
    --denoising_steps 8 --action_horizon "$1" --random_seed "${SEED}" \
    --skill_eval_mode "$2"
}

run_pro_eval() {
  # $1=horizon $2=mode
  python -m libero_scripts.libero_pro_eval \
    --model_path "${CKPT}" --task_suite_name libero_10 \
    --perturbation_type "${PERTURBATION}" --num_trials_per_task "${NUM_TRIAL}" \
    --num_steps_wait 10 --embodiment_tag new_embodiment \
    --data_config libero_original --denoising_steps 8 \
    --action_horizon "$1" --random_seed "${SEED}" \
    --skill_eval_mode "$2"
}

echo "[$(timestamp)] Checking checkpoint: ${CKPT}"
if ! checkpoint_ready; then
  echo "[$(timestamp)] Checkpoint is not ready or is incomplete." >&2
  exit 1
fi

for mode in "${MODES[@]}"; do
  for horizon in "${HORIZONS[@]}"; do
    echo "[$(timestamp)] RUN standard mode=${mode} seed=${SEED} horizon=${horizon}"
    run_eval "${horizon}" "${mode}"

    echo "[$(timestamp)] RUN PRO mode=${mode} seed=${SEED} horizon=${horizon} perturb=${PERTURBATION}"
    run_pro_eval "${horizon}" "${mode}"
  done
done

echo "[$(timestamp)] Finished skill-ablation eval for libero10_256_full_cls_stage2_llm_init"
