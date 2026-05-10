#!/bin/bash

#SBATCH --job-name=eval_lib10_init
#SBATCH --output=slurm_output/eval_libero10_llm_init.log
#SBATCH --error=slurm_output/eval_libero10_llm_init.log
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

if [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
  module add cuda/12.8.1 gcc/11.2.0 ffmpeg/7.1 || true
fi

source /fs/nexus-scratch/yliang17/miniconda3/bin/activate gr00t

export CACHE_DIR="${CACHE_DIR:-/fs/nexus-projects/wilddiffusion/cache}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CKPT="/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/libero10_256_full_cls_stage2_llm_init/checkpoint-60000"
EXPECTED_STEP=60000
SEEDS=(42 78 98)
HORIZONS=(5 10 16)
PERTURBATIONS=(object semantic task)
NUM_TRIAL="${NUM_TRIAL:-10}"
LIBERO_10_TASKS=10

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

result_complete() {
  local result_path=$1
  local expected_episodes=$2

  python - "${result_path}" "${expected_episodes}" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        result = json.load(f)
except Exception:
    sys.exit(1)

try:
    total_episodes = int(result.get("total_episodes", -1))
except (TypeError, ValueError):
    sys.exit(1)

sys.exit(0 if total_episodes == int(sys.argv[2]) else 1)
PY
}

standard_result_path() {
  printf 'results/libero_eval_modellibero10_256_full_cls_stage2_llm_init_tasklibero_10_seed%s_h%s.json\n' "$1" "$2"
}

pro_result_path() {
  printf 'results/libero_pro_modellibero10_256_full_cls_stage2_llm_init_tasklibero_10_pert%s_seed%s_h%s.json\n' "$3" "$1" "$2"
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

expected_episodes=$((LIBERO_10_TASKS * NUM_TRIAL))
for seed in "${SEEDS[@]}"; do
  for horizon in "${HORIZONS[@]}"; do
    result_path="$(standard_result_path "${seed}" "${horizon}")"
    if result_complete "${result_path}" "${expected_episodes}"; then
      echo "[$(timestamp)] SKIP standard seed=${seed} horizon=${horizon}"
    else
      echo "[$(timestamp)] RUN standard seed=${seed} horizon=${horizon}"
      run_eval "${seed}" "${horizon}"
    fi

    for perturb in "${PERTURBATIONS[@]}"; do
      result_path="$(pro_result_path "${seed}" "${horizon}" "${perturb}")"
      if result_complete "${result_path}" "${expected_episodes}"; then
        echo "[$(timestamp)] SKIP PRO seed=${seed} horizon=${horizon} perturb=${perturb}"
      else
        echo "[$(timestamp)] RUN PRO seed=${seed} horizon=${horizon} perturb=${perturb}"
        run_pro_eval "${seed}" "${horizon}" "${perturb}"
      fi
    done
  done
done
