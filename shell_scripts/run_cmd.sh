#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

cd /fs/nexus-scratch/yliang17/Research/VLA/GR00T

bash shell_scripts/run_script_libero_object_router.sh
bash shell_scripts/eval_script_libero_object_router.sh
# Submit training job and capture its SLURM job ID
# TRAIN_JOB=$(sbatch --parsable shell_scripts/run_script_libero_object_router.sh)
# echo "Submitted training job: ${TRAIN_JOB}"

# # Submit eval job — runs only after training completes successfully
# EVAL_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} shell_scripts/eval_script_libero_object_router.sh)
# echo "Submitted eval job:     ${EVAL_JOB} (runs after job ${TRAIN_JOB} succeeds)"
