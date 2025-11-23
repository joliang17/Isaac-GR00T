#!/bin/bash

bash shell_scripts/run_slurm_umiacs.sh groot_libero_action_1trace action_only_video  

bash shell_scripts/run_slurm_umiacs.sh groot_libero_traj_1trace traj_video_both 

bash shell_scripts/run_slurm_umiacs.sh groot_libero_skill_parts traj_skill_only  

bash shell_scripts/run_slurm_umiacs_a6000.sh groot_libero_skill_parts_newemb_a6 traj_skill_only  

bash shell_scripts/run_slurm_umiacs.sh groot_libero_both_newemb  traj_video_both  

bash shell_scripts/run_slurm_umiacs.sh groot_libero_both_newemb2  traj_video_both  



bash shell_scripts/run_slurm_umiacs.sh groot_libero_skill_parts  traj_skill_only  

bash shell_scripts/run_slurm_umiacs.sh groot_libero_both2  traj_video_both  

bash shell_scripts/run_slurm_umiacs.sh groot_libero_skip_action  traj_video_both  


# annnotation with skill id on 500 trajs
# training: add skill id
bash shell_scripts/run_slurm_umiacs.sh skill_id_textonly  traj_skill_only_v2
bash shell_scripts/run_slurm_umiacs.sh skill_id_trajall  traj_only_v2

bash shell_scripts/run_slurm_umiacs.sh skill_id_textonly_v2  traj_skill_only_id
bash shell_scripts/run_slurm_umiacs.sh skill_id_trajall_v2  traj_only_id

bash shell_scripts/run_slurm_umiacs.sh skill_noid_textonly_2emb  traj_skill_only_noid
bash shell_scripts/run_slurm_umiacs.sh skill_noid_trajall  traj_only_noid


