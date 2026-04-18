#!/bin/bash

#SBATCH --job-name=libero_eval_object_router_adapter
#SBATCH --output=slurm_output/libero_eval_object_router_adapter.log
#SBATCH --error=slurm_output/libero_eval_object_router_adapter.log
#SBATCH --time=48:00:00
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

source /etc/profile.d/modules.sh
module add cuda/12.4.1
module add gcc/11.2.0
module add ffmpeg/7.1

source /fs/nexus-scratch/yliang17/miniconda3/bin/activate gr00t

cd /fs/nexus-scratch/yliang17/Research/VLA/GR00T

export OPENAI_API_KEY=""
export CACHE_DIR="/fs/nexus-projects/wilddiffusion/cache"
export CUDA_VISIBLE_DEVICES=1

TASK_NAME=libero_object_router_adapter_k8_v2
CKPT_DIR="/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/${TASK_NAME}"

# Use MODEL_PATH if passed via environment, otherwise find the latest checkpoint
if [ -z "${MODEL_PATH}" ]; then
    MODEL_PATH=$(ls -td "${CKPT_DIR}"/checkpoint-* 2>/dev/null | head -1)
    if [ -z "${MODEL_PATH}" ]; then
        echo "ERROR: no checkpoint found in ${CKPT_DIR}"
        exit 1
    fi
fi
echo "Evaluating checkpoint: ${MODEL_PATH}"

python3 libero_scripts/libero_eval.py \
    --task_suite_name libero_object \
    --num_steps_wait 10 \
    --num_trials_per_task 10 \
    --headless True \
    --model_path "${MODEL_PATH}" \
    --embodiment_tag new_embodiment \
    --data_config libero_original \
    --denoising_steps 8 \
    --normalize_action
