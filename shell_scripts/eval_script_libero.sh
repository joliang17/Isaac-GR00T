#!/bin/bash

#SBATCH --job-name=libero_eval_half10
#SBATCH --output=slurm_output/libero_eval_half10.log
#SBATCH --error=slurm_output/libero_eval_half10.log
#SBATCH --time=48:00:00
#SBATCH --account=scavenger 
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

source /etc/profile.d/modules.sh
module add cuda/12.4.1
module add gcc/11.2.0
module add ffmpeg/7.1

source /fs/nexus-scratch/yliang17/miniconda3/bin/activate gr00t

export OPENAI_API_KEY=""
base_dir="/fs/nexus-scratch/yliang17/Research/VLA/saved_folder/checkpoint"
export CACHE_DIR="/fs/nexus-projects/wilddiffusion/cache"
CUDA_VISIBLE_DEVICES=0

python3 libero_scripts/libero_eval.py \
    --task_suite_name libero_10 \
    --num_steps_wait 10 \
    --num_trials_per_task 10 \
    --port 5555 \
    --headless True \
    --model_path "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/libero10_256_full_v2/checkpoint-60000" \
    --embodiment_tag new_embodiment \
    --data_config libero_original \
    --denoising_steps 8 \
    --action_horizon 5 \


python3 libero_scripts/libero_eval.py \
    --task_suite_name libero_10 \
    --num_steps_wait 10 \
    --num_trials_per_task 10 \
    --port 5555 \
    --headless True \
    --model_path "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/libero10_256_half_v2/checkpoint-60000" \
    --embodiment_tag new_embodiment \
    --data_config libero_original \
    --denoising_steps 8 \
    --action_horizon 5 \
    # --normalize_action \
    # --add_prefix \
    # --model_path youliangtan/gr00t-n1.5-libero-long-posttrain \

# python3 libero_scripts/libero_eval_interleaved.py \
#     --task_suite_name libero_10 \
#     --num_steps_wait 10 \
#     --num_trials_per_task 2 \
#     --embodiment_tag new_embodiment \
#     --data_config libero_traj_arms_2 \
#     --model_path "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/stage1_nextstep_0301_merged/checkpoint-6000" \
#     --model_name "stage2_nextstep_skip_action_toolhead_only_v4" \
#     --denoising_steps 8 \
#     --call_baseline \
#     --history_type 'reinput' \
#     --traj_history 'yes'

