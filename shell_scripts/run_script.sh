#!/bin/bash

# finetune
# python scripts/gr00t_finetune.py --dataset-path ./demo_data/robot_sim.PickNPlace --num-gpus 1

# python scripts/gr00t_finetune.py --dataset-path /fs/nexus-projects/wilddiffusion/vla/RoboMind/h5_simulation_lerobot --num-gpus 1


python scripts/gr00t_finetune.py --dataset-path /fs/nexus-projects/wilddiffusion/vla/RoboMind/h5_simulation_lerobot --num-gpus 1 --lora_rank 32  --lora_alpha 128  --batch-size 24 --lora_llm_model