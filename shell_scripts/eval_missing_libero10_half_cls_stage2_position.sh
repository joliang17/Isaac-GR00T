#!/bin/bash

#SBATCH --array=0-6
#SBATCH --job-name=eval_halfcls_pos
#SBATCH --output=slurm_output/eval_halfcls_pos_%A_%a.log
#SBATCH --error=slurm_output/eval_halfcls_pos_%A_%a.log
#SBATCH --time=8:00:00
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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL="libero10_256_half_cls_stage2"
CKPT="/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/${MODEL}/checkpoint-60000"
TASKS=(
  "78 5"
  "98 5"
  "78 10"
  "98 10"
  "42 16"
  "78 16"
  "98 16"
)

task="${TASKS[${SLURM_ARRAY_TASK_ID}]}"
read -r SEED HORIZON <<< "${task}"

RESULT="results/libero_pro_model${MODEL}_tasklibero_10_pertposition_seed${SEED}_h${HORIZON}.json"

if python - "${RESULT}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    sys.exit(1)
try:
    with path.open("r", encoding="utf-8") as f:
        result = json.load(f)
except Exception:
    sys.exit(1)
sys.exit(0 if int(result.get("total_episodes", -1)) == 100 else 1)
PY
then
  echo "Result already complete: ${RESULT}"
  exit 0
fi

python -m libero_scripts.libero_pro_eval \
  --model_path "${CKPT}" \
  --task_suite_name libero_10 \
  --perturbation_type position \
  --num_trials_per_task 10 \
  --num_steps_wait 10 \
  --embodiment_tag new_embodiment \
  --data_config libero_original \
  --denoising_steps 8 \
  --action_horizon "${HORIZON}" \
  --random_seed "${SEED}"
