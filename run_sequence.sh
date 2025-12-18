#!/usr/bin/env bash
set -euo pipefail

# stage1
jibv1_stage1=$(sbatch --parsable slurm_scripts/nextstep_stage1_skillonly.sh)

jibv1_stage2_frzembB=$(sbatch --parsable --dependency=afterok:${jibv1_stage1}  slurm_scripts/nextstep_stage2_frzembB.sh)
echo "Submitted nextstep_stage2_frzembB: $jibv1_stage2_frzembB (afterok: $jibv1_stage1)"

jibv1_actiononly_frzembB=$(sbatch --parsable --dependency=afterok:${jibv1_stage2_frzembB} slurm_scripts/nextstep_stage3_frzembB_actiononly.sh)
echo "Submitted nextstep_stage3_frzembB_actiononly: $jibv1_actiononly_frzembB (afterok:$jibv1_stage2_frzembB)"

jibv1_stage2_onlyembA=$(sbatch --parsable --dependency=afterok:${jibv1_stage1}  slurm_scripts/nextstep_stage2_onlyembA.sh)
echo "Submitted nextstep_stage2_onlyembA: $jibv1_stage2_onlyembA (afterok: $jibv1_stage1)"

jibv1_actiononly_onlyembA=$(sbatch --parsable --dependency=afterok:${jibv1_stage2_onlyembA} slurm_scripts/nextstep_stage3_onlyembA_actiononly.sh)
echo "Submitted nextstep_stage3_onlyembA_actiononly: $jibv1_actiononly_onlyembA (afterok:$jibv1_stage2_onlyembA)"

