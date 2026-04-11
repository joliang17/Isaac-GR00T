#!/bin/bash

#SBATCH --job-name=skillaction_v2
#SBATCH --output=/fs/nexus-scratch/yliang17/Research/VLA/GR00T/slurm_output/skillaction_v2.log
#SBATCH --error=/fs/nexus-scratch/yliang17/Research/VLA/GR00T/slurm_output/skillaction_v2.log
#SBATCH --time=12:00:00
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

export WANDB_PROJECT="vla_tooluse"
TASK_NAME=skillaction_v2_stage1

# skill_action_v2: task field = pure instruction only; all skill labels from JSON.
# Set skill_label_type to 'primary_action_verb' to use atomic verbs instead of full phrases.
SKILL_JSON="/fs/nexus-scratch/yliang17/Research/VLA/AtomicVLA/data_split_json/libero_lerobot_addskill_10.json"

python scripts/gr00t_finetune.py \
  --dataset-path "/fs/nexus-projects/wilddiffusion/vla/atomic_data/libero_atomic_10" \
  --num-gpus 1 \
  --lora_rank 32 \
  --lora_alpha 128 \
  --batch-size 4 \
  --lora_llm_model \
  --data_config "libero_traj_arms_2" \
  --video_backend "torchvision_av" \
  --save_steps 3000 \
  --max_steps 6000 \
  --output_dir "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/${TASK_NAME}" \
  --run_name ${TASK_NAME} \
  --grad_norm 1.0 \
  --tune_special_A \
  --tune_special_B \
  --windowing_mode "skill_action" \
  --skill_annotation_path "${SKILL_JSON}" \
  --skill_label_type "skill" \
  # --dataloader_num_workers 0


