#!/bin/bash

#SBATCH --job-name=libero_eval_gr00t
#SBATCH --output=slurm_output/libero_eval_gr00t1.log
#SBATCH --error=slurm_output/libero_eval_gr00t1.log
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

export OPENAI_API_KEY=""
base_dir="/fs/nexus-scratch/yliang17/Research/VLA/saved_folder/checkpoint"
export CACHE_DIR="/fs/nexus-projects/wilddiffusion/cache"
# export CUDA_VISIBLE_DEVICES="1"

python3 libero_scripts/libero_eval.py \
    --task_suite_name libero_10 \
    --num_steps_wait 10 \
    --num_trials_per_task 10 \
    --port 5555 \
    --headless True \
    --model_path youliangtan/gr00t-n1.5-libero-long-posttrain \
    --embodiment_tag new_embodiment \
    --data_config libero_original \
    --denoising_steps 8 \
    --normalize_action

