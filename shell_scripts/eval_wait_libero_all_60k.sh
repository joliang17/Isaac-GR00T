#!/bin/bash

#SBATCH --job-name=eval_all_60k_normal
#SBATCH --output=slurm_output/eval_all_60k_normal.log
#SBATCH --error=slurm_output/eval_all_60k_normal.log
#SBATCH --time=48:00:00
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

source /etc/profile.d/modules.sh
module add cuda/12.8.1 gcc/11.2.0 ffmpeg/7.1
source /fs/nexus-scratch/yliang17/miniconda3/bin/activate gr00t

export CACHE_DIR="/fs/nexus-projects/wilddiffusion/cache"
export OPENAI_API_KEY=""
export CUDA_VISIBLE_DEVICES=0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/eval_libero_wait_common.sh"

LABEL="liberoall_256_full_60k"
CKPT="/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/${LABEL}/checkpoint-60000"
EXPECTED_STEP=60000

# wait_for_checkpoint "${CKPT}" "${EXPECTED_STEP}" "${LABEL}"
run_eval_matrix "${CKPT}" "${LABEL}"
