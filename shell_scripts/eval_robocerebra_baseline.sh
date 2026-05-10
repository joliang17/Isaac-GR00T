#!/bin/bash

#SBATCH --job-name=rc_base_eval
#SBATCH --output=slurm_output/rc_base_eval.log
#SBATCH --error=slurm_output/rc_base_eval.log
#SBATCH --time=6:00:00
#SBATCH --account=cml-director
#SBATCH --partition=cml-director
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -eo pipefail

REPO_ROOT="/fs/nexus-scratch/yliang17/Research/VLA/GR00T"
cd "${REPO_ROOT}"
mkdir -p slurm_output

source /fs/nexus-scratch/yliang17/miniconda3/bin/activate gr00t

if [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
  module add cuda/12.8.1 || true
  module add gcc/11.2.0 || true
fi

export WANDB_PROJECT="${WANDB_PROJECT:-vla_tooluse}"
export CACHE_DIR="${CACHE_DIR:-/fs/nexus-projects/wilddiffusion/cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python scripts/gr00t_finetune.py \
  --num-gpus 1 \
  --dataset-path "/fs/nexus-projects/wilddiffusion/vla/atomic_data/robocerebra_study_table_train_case13_plus" \
  --train_eval_dataset_path "/fs/nexus-projects/wilddiffusion/vla/atomic_data/robocerebra_study_table_train_case13_plus" \
  --train_eval_max_samples 1024 \
  --train_eval_fraction 0.01 \
  --eval_dataset_path "/fs/nexus-projects/wilddiffusion/vla/atomic_data/robocerebra_study_table_holdout_case1_12" \
  --eval_skill_annotation_path "/fs/nexus-scratch/yliang17/Research/VLA/AtomicVLA/data_split_json/robocerebra_study_table_holdout_case1_12_lerobot_addskill.json" \
  --windowing_mode "step" \
  --batch-size 16 \
  --data_config "libero_original" \
  --video_backend "torchvision_av" \
  --base_model_path "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/robocerebra_study_table_baseline_no_skill/checkpoint-30000" \
  --save_steps 5000 \
  --max_steps 30000 \
  --output_dir "/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/robocerebra_study_table_baseline_eval_only" \
  --run_name "robocerebra_study_table_baseline_eval_only" \
  --tune_diffusion_model \
  --do_eval \
  --eval-only
