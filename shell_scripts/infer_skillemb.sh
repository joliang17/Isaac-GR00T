#!/bin/bash

#SBATCH --job-name=gr00t_vlm_skillemb
#SBATCH --output=slurm_output/gr00t_vlm_skillemb.log
#SBATCH --error=slurm_output/gr00t_vlm_skillemb.log
#SBATCH --time=48:00:00
#SBATCH --account=cml-director
#SBATCH --partition=cml-director
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G

source /fs/nexus-scratch/yliang17/miniconda3/bin/activate gr00t_vlm

cd /fs/nexus-scratch/yliang17/Research/VLA/GR00T_vlm

source /etc/profile.d/modules.sh
module add cuda/12.4.1
module add gcc/11.2.0

export WANDB_PROJECT="vla_tooluse"
export CACHE_DIR="/fs/nexus-projects/wilddiffusion/cache"
export CUDA_VISIBLE_DEVICES=1

SKILL_PREFIX="Next-skill retrieval for robot manipulation. Compare the current image and task to one candidate atomic skill.\nSkill meanings:\nclose = shut a drawer, cabinet, door, lid, or gripper that is open.\nopen = open a drawer, cabinet, door, lid, or container.\npick = grasp or lift the target object before moving it.\nplace = release or put a held object at the goal location.\nturn = rotate a knob, button, burner, or control in place.\nUse object contact, gripper state, and whether the robot is holding something to separate pick from place and open from close.\nCandidate next skill: "

PYTHONPATH=/fs/nexus-scratch/yliang17/Research/VLA/GR00T_vlm \
python scripts/evaluate_llm_skill_router.py \
    --max-samples 128 \
    --skill-prefix "$SKILL_PREFIX" \
    --output-path hidden_states/qwen3vl_skill_router_probe_improved_prefix_zero_shot.json

PYTHONPATH=/fs/nexus-scratch/yliang17/Research/VLA/GR00T_vlm \
python scripts/evaluate_llm_skill_router.py \
    --max-samples 128 \
    --skill-prefix "$SKILL_PREFIX" \
    --calibration linear_probe \
    --output-path hidden_states/qwen3vl_skill_router_probe_improved_prefix_linear_probe.json
