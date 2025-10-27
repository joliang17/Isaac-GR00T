#!/bin/bash

#SBATCH --job-name=groot_libero_action_1trace
#SBATCH --output=groot_libero_action_1trace.log
#SBATCH --error=groot_libero_action_1trace.log
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

source /fs/nexus-scratch/yliang17/miniconda3/bin/activate gr00t

WANDB_PROJECT="vla_tooluse"
TASK_NAME=""
# python scripts/gr00t_finetune.py --dataset-path /fs/nexus-projects/wilddiffusion/vla/RoboMind/h5_simulation_lerobot_traj --num-gpus 1 --lora_rank 32  --lora_alpha 128  --batch-size 24 --lora_llm_model --dataloader_num_workers=0


python scripts/gr00t_finetune.py --dataset-path /fs/nexus-projects/wilddiffusion/vla/LIBERO/traj_skill_only --num-gpus 1 --lora_rank 32  --lora_alpha 128  --batch-size 4 --lora_llm_model --window_length=10 --data_config libero_traj_arms --video_backend torchvision_av --save_steps 20 --tune_special_A --tune_special_B

# python scripts/gr00t_finetune.py --dataset-path /fs/nexus-projects/wilddiffusion/vla/LIBERO/traj_skill_only --num-gpus 1 --lora_rank 32  --lora_alpha 128  --batch-size 4 --lora_llm_model --window_length=10 --data_config libero_traj_arms --video_backend torchvision_av --save_steps 20 --output_dir "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/skill_id_textonly" --resume 

# python scripts/gr00t_finetune.py --dataset-path /fs/nexus-projects/wilddiffusion/vla/LIBERO/traj_only_id --num-gpus 1 --lora_rank 32  --lora_alpha 128  --batch-size 4 --lora_llm_model --window_length=10 --data_config libero_traj_arms --video_backend torchvision_av --save_steps 20 --output_dir "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/skill_id_trajall_v2" --resume --max_steps=20000

# python scripts/gr00t_finetune.py --dataset-path /fs/nexus-projects/wilddiffusion/vla/LIBERO/traj_skill_only --num-gpus 1 --lora_rank 32  --lora_alpha 128  --batch-size 4 --lora_llm_model --window_length=10 --data_config libero_traj_arms --video_backend torchvision_av --save_steps 20 --output_dir "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/groot_libero_stage2_freezeemb" --base_model_path "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/groot_libero_skip_action_merged/checkpoint-10000" --resume --freeze_embeddings


# python scripts/gr00t_finetune.py --dataset-path /fs/nexus-projects/wilddiffusion/vla/LIBERO/traj_video_both --num-gpus 1 --batch-size 4 --lora_llm_model --window_length=10 --data_config libero_traj_arms --video_backend torchvision_av --save_steps 20 --output_dir "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/groot_libero_stage2_6ktest" --base_model_path "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/groot_libero_stage2_6k_freezeall/checkpoint-2000"

# python scripts/gr00t_finetune.py --dataset-path /fs/nexus-projects/wilddiffusion/vla/LIBERO/traj_skill_only --num-gpus 1 --batch-size 4 --lora_llm_model --window_length=10 --data_config libero_traj_arms --video_backend torchvision_av --save_steps 20 --output_dir "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/groot_libero_stage2_6k_freezeemb" --base_model_path "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/groot_libero_stage2_6k_freezeall/checkpoint-2000"
