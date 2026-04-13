#!/bin/bash

#SBATCH --job-name=libero_object_router
#SBATCH --output=slurm_output/libero_object_router.log
#SBATCH --error=slurm_output/libero_object_router.log
#SBATCH --time=48:00:00
#SBATCH --account=cml-director
#SBATCH --partition=cml-director
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G

source /fs/nexus-scratch/yliang17/miniconda3/bin/activate gr00t

cd /fs/nexus-scratch/yliang17/Research/VLA/GR00T

source /etc/profile.d/modules.sh
module add cuda/12.4.1
module add gcc/11.2.0

export WANDB_PROJECT="vla_tooluse"
export CACHE_DIR="/fs/nexus-projects/wilddiffusion/cache"
export CUDA_VISIBLE_DEVICES=0

TASK_NAME=libero_object_router_adapter_k8_v1

# Load libero_10 checkpoint (frozen), train router + K=8 task embeddings + FiLM adapters
# on libero_object. Adapters condition state/action encoders and action decoder on the
# task embedding, enabling task-specific modulation without touching embodiment weights.
python scripts/gr00t_finetune.py \
  --num-gpus 1 \
  --dataset-path "/fs/nexus-projects/wilddiffusion/vla/libero_lerobot/libero_object_no_noops_lerobot" \
  --windowing_mode "step" \
  --batch-size 16 \
  --data_config "libero_original" \
  --video_backend "torchvision_av" \
  --save_steps 10000 \
  --max_steps 30000 \
  --base_model_path "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/libero_10_base/checkpoint-60000" \
  --output_dir "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/${TASK_NAME}" \
  --run_name "${TASK_NAME}" \
  --use-task-router \
  --num-task-emb-slots 8 \
  --router-hidden-dim 256 \
  --use-task-adapter \
  --no-tune-projector \
  --learning_rate 1e-4 \
  --warmup_ratio 0.05
  # NOTE: --tune_diffusion_model not passed (default False -> DiT frozen)
  # NOTE: --tune_llm / --tune_visual not passed (default False -> backbone frozen)
  # NOTE: --no-tune-projector freezes state/action encoders (overrides default True)
  # NOTE: --use-task-adapter trains FiLM adapters on top of frozen encoders/decoder
