#!/bin/bash

#SBATCH --job-name=libero_eval_weighted
#SBATCH --output=slurm_output/libero_eval_weighted.log
#SBATCH --error=slurm_output/libero_eval_weighted.log
#SBATCH --time=48:00:00
#SBATCH --account=cml-director
#SBATCH --partition=cml-director
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G

source /etc/profile.d/modules.sh
module add cuda/12.4.1 
module add gcc/11.2.0 
module add ffmpeg/7.1
source /fs/nexus-scratch/yliang17/miniconda3/bin/activate gr00t

export CACHE_DIR="/fs/nexus-projects/wilddiffusion/cache"
export OPENAI_API_KEY=""
# export CUDA_VISIBLE_DEVICES=1

SEEDS=(42 78 98)
HORIZONS=(1 5 10 16)

run_eval() {
    local ckpt=$1 seed=$2 horizon=$3
    python -m libero_scripts.libero_eval \
        --model_path "${ckpt}" --task_suite_name libero_10 \
        --num_trials_per_task 10 --num_steps_wait 10 \
        --embodiment_tag new_embodiment --data_config libero_original \
        --denoising_steps 8 --action_horizon "${horizon}" --random_seed "${seed}"
}

run_pro_eval() {
    local ckpt=$1 seed=$2 horizon=$3
    python -m libero_scripts.libero_pro_eval \
        --model_path "${ckpt}" --task_suite_name libero_10 \
        --perturbation_type object --num_trials_per_task 10 \
        --action_horizon "${horizon}" --random_seed "${seed}"
}

CKPT="/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/libero10_256_half_cls_weighted_stage2_v1/checkpoint-60000"
for SEED in "${SEEDS[@]}"; do
    for H in "${HORIZONS[@]}"; do
        run_eval     "${CKPT}" "${SEED}" "${H}"
        run_pro_eval "${CKPT}" "${SEED}" "${H}"
    done
done
