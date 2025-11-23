#!/bin/bash

#SBATCH --job-name=vla_test
#SBATCH --output=vla_test.log
#SBATCH --error=vla_test.log
#SBATCH --time=48:00:00
#SBATCH --account=cml-director
#SBATCH --partition=cml-director
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

source /etc/profile.d/modules.sh
module add cuda/12.4.1
module add gcc/11.2.0
module add ffmpeg/7.1

source /fs/nexus-scratch/yliang17/miniconda3/bin/activate gr00t

# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_SHOW_CPP_STACKTRACES=1
# export TORCH_USE_CUDA_DSA=1
export OPENAI_API_KEY=""
run_eval () {
    model_ckpt="$1"
    base_dir="/fs/nexus-scratch/yliang17/Research/VLA/saved_folder/checkpoint"

    merged_dir="${base_dir}/${model_ckpt}_merged/checkpoint-6000"
    orig_dir="${base_dir}/${model_ckpt}/checkpoint-6000"

    echo "=============================="
    echo "[INFO] Processing model: ${model_ckpt}"
    echo "=============================="

    # ----------------------------
    # Choose checkpoint directory
    # ----------------------------
    if [ -d "${base_dir}/${model_ckpt}_merged" ]; then
        echo "[INFO] Using MERGED checkpoint: ${merged_dir}"

        # Copy experiment_cfg only if missing
        if [ ! -d "${merged_dir}/experiment_cfg" ]; then
            echo "[INFO] Copying experiment_cfg to merged checkpoint..."
            cp -r "${base_dir}/${model_ckpt}/experiment_cfg" "${merged_dir}/"
        fi

        final_model_path="${merged_dir}"
    else
        echo "[INFO] Using ORIGINAL checkpoint: ${orig_dir}"
        final_model_path="${orig_dir}"
    fi

    echo "[INFO] Final model_path = ${final_model_path}"

    # ----------------------------
    # Run evaluation
    # ----------------------------
    python3 libero_scripts/libero_eval_interleaved.py \
        --task_suite_name libero_10 \
        --num_steps_wait 10 \
        --num_trials_per_task 5 \
        --embodiment_tag new_embodiment \
        --data_config libero_traj_arms \
        --denoising_steps 8 \
        --model_path "${final_model_path}" \
        --model_name "${model_ckpt}" \
        --call_baseline

    echo "[INFO] Evaluation finished for ${model_ckpt}"
    echo
}

# run_eval "stage2_freezeembB_nextstep_skip_action"
# run_eval "stage2_onlyembA_nextstep_skip_action"

# run_eval "stage2_freezeembB_allstep_skip_action"
run_eval "stage2_onlyembA_allstep_skip_action"

# run_eval "stage2_freezeembB_nextstep_skip_action_slicing"
# run_eval "stage2_onlyembA_nextstep_skip_action_slicing"

# run_eval "stage2_freezeembB_nextstep_skip_action_toolhead"
# run_eval "stage2_onlyembA_nextstep_skip_action_toolhead"

# run_eval "stage2_freezeembB_nextstep_skip_action_toolus"
# run_eval "stage2_onlyembA_nextstep_skip_action_toolus"
