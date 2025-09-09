#!/bin/bash

#SBATCH --job-name=vla_test
#SBATCH --output=vla_test.log
#SBATCH --error=vla_test.log
#SBATCH --time=48:00:00
#SBATCH --account=cml-zhou
#SBATCH --partition=cml-zhou
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G


source /etc/profile.d/modules.sh
module add cuda/12.4.1

SAVE_DIR="keyword_results"
source /fs/nexus-scratch/yliang17/miniconda3/bin/activate groot


# finetune
# python scripts/gr00t_finetune.py --dataset-path ./demo_data/robot_sim.PickNPlace --num-gpus 1

# python scripts/gr00t_finetune.py --dataset-path /fs/nexus-projects/wilddiffusion/vla/RoboMind/h5_simulation_lerobot --num-gpus 1


# python scripts/gr00t_finetune.py --dataset-path /fs/nexus-projects/wilddiffusion/vla/RoboMind/h5_simulation_lerobot_traj --num-gpus 1 --lora_rank 32  --lora_alpha 128  --batch-size 24 --lora_llm_model --dataloader_num_workers=0
# python scripts/gr00t_finetune.py --dataset-path /fs/nexus-projects/wilddiffusion/vla/RoboMind/h5_simulation_lerobot_traj --num-gpus 1 --lora_rank 32  --lora_alpha 128  --batch-size 2 --lora_llm_model --dataloader_num_workers=0 --window_length=10

python scripts/gr00t_finetune.py --dataset-path /fs/nexus-projects/wilddiffusion/vla/RoboMind/h5_simulation_lerobot_traj --num-gpus 1 --lora_rank 32  --lora_alpha 128  --batch-size 2 --lora_llm_model --window_length=10