#!/bin/bash

#SBATCH --job-name=stage2_onlyembA_nextstep_skip_action_key
#SBATCH --output=/fs/nexus-scratch/yliang17/Research/VLA/GR00T/slurm_output/stage2_onlyembA_nextstep_skip_action_key.log
#SBATCH --error=/fs/nexus-scratch/yliang17/Research/VLA/GR00T/slurm_output/stage2_onlyembA_nextstep_skip_action_key.log
#SBATCH --time=24:00:00
#SBATCH --dependency=afterok:6316088
#SBATCH --account=cml-director
#SBATCH --partition=cml-director
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

cd /fs/nexus-scratch/yliang17/Research/VLA/GR00T

source /etc/profile.d/modules.sh
module add cuda/12.4.1
module add gcc/11.2.0

source /fs/nexus-scratch/yliang17/miniconda3/bin/activate gr00t

ROOT_FOLDER="/fs/nexus-projects/wilddiffusion/vla"

export WANDB_PROJECT="vla_tooluse"
export CUDA_VISIBLE_DEVICES=1

TASK_NAME=test_run


python scripts/gr00t_finetune.py \
  --dataset-path "/fs/nexus-projects/wilddiffusion/vla/atomic_data/libero_atomic_10" \
  --windowing_mode skill_action \
  --skill_annotation_path /fs/nexus-scratch/yliang17/Research/VLA/AtomicVLA/data_split_json/libero_lerobot_addskill_10.json \
  --num-gpus 1 \
  --batch-size 4 \
  --gradient_accumulation_steps 4 \
  --data_config "libero_traj_arms_2" \
  --video_backend "torchvision_av" \
  --save_steps 3 \
  --max_steps 6000 \
  --output_dir "${ROOT_FOLDER}/GR00T/checkpoint/${TASK_NAME}" \
  --run_name ${TASK_NAME} \
  --grad_norm 1.0 \
  --tune_special_A \
  --action_ds_ratio=0.5 \
  --dataloader_num_workers 0 \
  --do_eval


