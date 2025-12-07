#!/bin/bash

bash slurm_scripts/nextstep_stage1_onlytext.sh
bash slurm_scripts/nextstep_frzembB_stage2.sh
sbatch slurm_scripts/nextstep_frzembB_stage3_action_llm.sh
bash slurm_scripts/nextstep_frzembB_stage3_action.sh


bash slurm_scripts/nextstep_onlyembA_stage2.sh
sbatch slurm_scripts/nextstep_onlyembA_stage3_action.sh

bash slurm_scripts/nextstep_frzembB_stage3_action.sh
bash slurm_scripts/nextstep_onlyembA_stage3_action.sh

# sbatch slurm_scripts/allstep_stage1_onlytext.sh
# sbatch slurm_scripts/allstep_frzembB_stage2.sh
# sbatch slurm_scripts/allstep_onlyembA_stage2.sh
