#!/bin/bash

#SBATCH --job-name=vla_test
#SBATCH --output=vla_test.log
#SBATCH --error=vla_test.log
#SBATCH --time=48:00:00
#SBATCH --account=cml-director
#SBATCH --partition=cml-director
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

source /etc/profile.d/modules.sh
module add cuda/12.4.1
module add gcc/11.2.0
module add ffmpeg/7.1

source /fs/nexus-scratch/yliang17/miniconda3/bin/activate gr00t

# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_SHOW_CPP_STACKTRACES=1
# export TORCH_USE_CUDA_DSA=1

# python3 libero_scripts/libero_eval.py \
#     --task_suite_name libero_10 \
#     --num_steps_wait 10 \
#     --num_trials_per_task 5 \
#     --port 5555 \
#     --headless True \
#     --model_path youliangtan/gr00t-n1.5-libero-long-posttrain \
#     --embodiment_tag new_embodiment \
#     --data_config libero_original \
#     --denoising_steps 8

python3 libero_scripts/libero_eval_interleaved.py \
    --task_suite_name libero_10 \
    --num_steps_wait 10 \
    --num_trials_per_task 5 \
    --embodiment_tag new_embodiment \
    --data_config libero_original \
    --denoising_steps 8 \
    --model_path "/fs/nexus-scratch/yliang17/Research/VLA/GR00T/checkpoint/groot_libero_traj_1trace/checkpoint-8000"