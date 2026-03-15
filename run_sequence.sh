#!/usr/bin/env bash
set -euo pipefail


cd /fs/nexus-scratch/yliang17/Research/VLA/VLA_proj/LIBERO/
gene_data=$(sbatch run_script_toolend.sh)

cd /fs/nexus-scratch/yliang17/Research/VLA/GR00T
stage2_tool_jobid=$(sbatch --parsable --dependency=afterok:6305167 slurm_scripts/nextstep_stage2_onlyembA_toolhead_only.sh)

# stage2_jobid=$(sbatch --parsable --dependency=afterok:6288618 slurm_scripts/nextstep_stage2_onlyembA_2ds.sh)

# jibv1_stage2_onlyembA=$(sbatch --parsable slurm_scripts/nextstep_stage2_onlyembA_2ds.sh)
# echo "Submitted nextstep_stage2_onlyembA_2ds: $jibv1_stage2_onlyembA"


# # stage1
# cd /fs/nexus-scratch/yliang17/Research/VLA/VLA_proj/LIBERO
# jibv1_libero=$(sbatch --parsable run_script_skill.sh)

# cd /fs/nexus-scratch/yliang17/Research/VLA/GR00T
# # jibv1_stage1=$(sbatch slurm_scripts/nextstep_stage1_skillonly_2ds_v2.sh)
# jibv1_stage1="6288687"
# echo "Submitted stage1: $jibv1_stage1 "
# jibv1_stage2_onlyembA=$(sbatch --parsable --dependency=afterok:${jibv1_stage1} slurm_scripts/nextstep_stage2_onlyembA_2ds_v2.sh)
# echo "Submitted stage2: $jibv1_stage2_onlyembA (afterok: $jibv1_stage1)"
# # jibv1_stage2_onlyembA=$(sbatch --parsable slurm_scripts/nextstep_stage2_onlyembA_2ds.sh)
# # echo "Submitted nextstep_stage2_onlyembA_2ds: $jibv1_stage2_onlyembA"

# jibv1_stage2_toolhead_only=$(sbatch --parsable --dependency=afterok:${jibv1_stage2_onlyembA} slurm_scripts/nextstep_stage2_onlyembA_toolhead_only_v2.sh)
# echo "Submitted nextstep_stage2_onlyembA_toolhead_only: $jibv1_stage2_toolhead_only (afterok: $jibv1_stage2_onlyembA)"

# # jibv1_stage2_toolhead_only=6188304
# jibv1_actiononly_onlyembA=$(sbatch --parsable --dependency=afterok:${jibv1_libero} slurm_scripts/nextstep_stage3_onlyembA_actiononly.sh)
# echo "Submitted nextstep_stage3_onlyembA_actiononly: $jibv1_actiononly_onlyembA (afterok:$jibv1_libero)"
