#!/bin/bash

set -euo pipefail

REPO_ROOT="/fs/nexus-scratch/yliang17/Research/VLA/GR00T"
SCRIPT_DIR="${REPO_ROOT}/shell_scripts"
LOG_DIR="${REPO_ROOT}/slurm_output"

mkdir -p "${LOG_DIR}"

launch_watcher() {
    local name=$1
    local script=$2
    local log_file="${LOG_DIR}/${name}_watch.log"

    nohup bash "${script}" > "${log_file}" 2>&1 &
    local pid=$!
    echo "${pid} ${script} -> ${log_file}"
}

echo "Launching LIBERO eval watchers with CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
launch_watcher "eval_wait_libero_half_30k" "${SCRIPT_DIR}/eval_wait_libero_half_30k.sh"
launch_watcher "eval_wait_libero_all_60k" "${SCRIPT_DIR}/eval_wait_libero_all_60k.sh"
launch_watcher "eval_wait_libero_cls_stage2_v2" "${SCRIPT_DIR}/eval_wait_libero_cls_stage2_v2.sh"
