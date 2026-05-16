#!/bin/bash

#SBATCH --job-name=eval_lib10_init_full
#SBATCH --output=slurm_output/eval_libero10_llm_init_full.log
#SBATCH --error=slurm_output/eval_libero10_llm_init_full.log
#SBATCH --time=48:00:00
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# Full (forced) re-evaluation of libero10_256_full_cls_stage2_llm_init on
# libero10 + libero_pro. Unlike eval_libero10_skill_llm_init.sh this does NOT
# skip already-complete result files, so every rollout video is regenerated
# with the router-selected skill name overlaid on the frames.

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
SEEDS=(42 78 98)
HORIZONS=(5 10 16)
PERTURBATIONS=(object semantic task)
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
  python -m libero_scripts.libero_eval \
    --model_path "${CKPT}" --task_suite_name libero_10 \
    --num_trials_per_task "${NUM_TRIAL}" --num_steps_wait 10 \
    --embodiment_tag new_embodiment --data_config libero_original \
    --denoising_steps 8 --action_horizon "$2" --random_seed "$1"
}

run_pro_eval() {
  python -m libero_scripts.libero_pro_eval \
    --model_path "${CKPT}" --task_suite_name libero_10 \
    --perturbation_type "$3" --num_trials_per_task "${NUM_TRIAL}" \
    --num_steps_wait 10 --embodiment_tag new_embodiment \
    --data_config libero_original --denoising_steps 8 \
    --action_horizon "$2" --random_seed "$1"
}

echo "[$(timestamp)] Checking LIBERO10 LLM-init checkpoint: ${CKPT}"
if ! checkpoint_ready; then
  echo "[$(timestamp)] Checkpoint is not ready or is incomplete." >&2
  exit 1
fi

for seed in "${SEEDS[@]}"; do
  for horizon in "${HORIZONS[@]}"; do
    echo "[$(timestamp)] RUN standard seed=${seed} horizon=${horizon}"
    run_eval "${seed}" "${horizon}"

    for perturb in "${PERTURBATIONS[@]}"; do
      echo "[$(timestamp)] RUN PRO seed=${seed} horizon=${horizon} perturb=${perturb}"
      run_pro_eval "${seed}" "${horizon}" "${perturb}"
    done
  done
done

echo "[$(timestamp)] Finished full eval for libero10_256_full_cls_stage2_llm_init"
