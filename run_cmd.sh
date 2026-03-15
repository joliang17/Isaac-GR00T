#!/bin/bash

# bash /fs/nexus-scratch/yliang17/Research/VLA/VLA_proj/LIBERO/run_script_eo.sh
# bash /fs/nexus-scratch/yliang17/Research/VLA/VLA_proj/LIBERO/run_script_skill.sh
# bash /fs/nexus-scratch/yliang17/Research/VLA/VLA_proj/LIBERO/run_script_video.sh

# # output: stage1_nextstep_3ds
# # # output: stage1_nextstep_3ds_larger
# tobe tested
# bash slurm_scripts/nextstep_stage1_skillonly_2ds.sh

# bash slurm_scripts/nextstep_stage2_onlyembA_2ds.sh
# bash slurm_scripts/nextstep_stage2_onlyembA_toolhead_only.sh
# bash slurm_scripts/nextstep_stage3_onlyembA_actiononly_v6.sh


# output: stage2_onlyembA_nextstep_skip_action_v6
sbatch slurm_scripts/nextstep_stage1_skillonly_2ds.sh
sbatch slurm_scripts/nextstep_stage2_onlyembA_2ds.sh
sbatch slurm_scripts/nextstep_stage2_onlyembA_toolhead_only.sh

sbatch slurm_scripts/nextstep_stage1_skillonly_2ds_v2.sh
sbatch slurm_scripts/nextstep_stage2_onlyembA_2ds_v2.sh
sbatch slurm_scripts/nextstep_stage2_onlyembA_toolhead_only_v2.sh
# bash slurm_scripts/nextstep_stage3_onlyembA_actiononly_v7.sh

# output: stage2_nextstep_skip_action_toolhead_only_v6

# output: stage3_onlyembA_nextstep_actiononly_v6
# bash slurm_scripts/nextstep_stage3_onlyembA_actiononly_v6.sh

# output: stage3_onlyembA_nextstep_actiononly_v7
# bash slurm_scripts/nextstep_stage3_onlyembA_actiononly_v7.sh


# bash shell_scripts/eval_script.sh
