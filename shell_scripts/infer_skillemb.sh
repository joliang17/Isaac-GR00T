#!/bin/bash

#SBATCH --job-name=libero_training_full
#SBATCH --output=slurm_output/libero_training_full.log
#SBATCH --error=slurm_output/libero_training_full.log
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
export CUDA_VISIBLE_DEVICES=1


SKILL_PREFIX="The robot executes atomic manipulation skills.\nYour job is to select the NEXT skill the robot should execute.\n\nAvailable skills and definitions:\n1. close 2. open, 3. pick, 4. place, 5. turn on the button. Decide the NEXT skill needed.\n\nThink briefly about the scene\n"

python scripts/evaluate_llm_skill_router.py \
    --max-samples 128 \
    --skill-prefix "$SKILL_PREFIX" \
    --output-path hidden_states/llm_skill_router_probe_zero_shot.json

python scripts/evaluate_llm_skill_router.py \
    --max-samples 128 \
    --skill-prefix "$SKILL_PREFIX" \
    --calibration linear_probe \
    --output-path hidden_states/llm_skill_router_probe_linear_probe.json

