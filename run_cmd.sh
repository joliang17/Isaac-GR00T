#!/bin/bash

# bash shell_scripts/eval_script.sh

# sbatch nextstep_freezeembB_stage2.slurm  
# sbatch nextstep_onlyembA_stage2.slurm  
# sbatch nextstep_freezeembB_stage2_slicing.slurm 
# sbatch nextstep_onlyembA_stage2_slicing.slurm  

# sbatch nextstep_toolhead_freezeembB_stage2.slurm 
# sbatch nextstep_toolhead_onlyembA_stage2.slurm  

# sbatch nextstep_toolus_freezeembB_stage2.slurm 
# sbatch nextstep_toolus_onlyembA_stage2.slurm  


bash allstep_freezeembB_stage2.slurm  
bash allstep_onlyembA_stage2.slurm   

# bash nextstep_onlyembA_stage2.slurm   