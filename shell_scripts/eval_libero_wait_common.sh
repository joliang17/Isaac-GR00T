#!/bin/bash
#SBATCH --array=0-3
#SBATCH --job-name=eval_libero
#SBATCH --output=slurm_output/eval_libero_%A_%a.log
#SBATCH --error=slurm_output/eval_libero_%A_%a.log
#SBATCH --time=48:00:00
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

REPO_ROOT="/fs/nexus-scratch/yliang17/Research/VLA/GR00T"
CONDA_SH="/fs/nexus-scratch/yliang17/miniconda3/bin/activate"

cd "${REPO_ROOT}"

if [[ -f /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
    module add cuda/12.8.1 gcc/11.2.0 ffmpeg/7.1
fi

source "${CONDA_SH}" gr00t

export CACHE_DIR="${CACHE_DIR:-/fs/nexus-projects/wilddiffusion/cache}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_LABELS=(
    "liberoall_256_full_60k"
    "libero10_256_full_cls_stage2_v2"
    "libero10_256_half_30k"
    "libero10_256_half_cls_stage2"
)
RUN_STEPS=(
    60000
    60000
    30000
    60000
)

SEEDS_LIST="${SEEDS_LIST:-42 78 98}"
read -r -a SEEDS <<< "${SEEDS_LIST}"
HORIZONS=(5 10 16)
PERTURBATIONS=(object semantic task)
NUM_TRIAL="${NUM_TRIAL:-10}"
POLL_INTERVAL="${POLL_INTERVAL:-300}"
LIBERO_10_TASKS=10

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

checkpoint_has_expected_step() {
    local state_file=$1
    local expected_step=$2

    python - "${state_file}" "${expected_step}" <<'PY'
import json
import sys

state_file = sys.argv[1]
expected_step = int(sys.argv[2])

try:
    with open(state_file, "r", encoding="utf-8") as f:
        state = json.load(f)
except Exception:
    sys.exit(1)

sys.exit(0 if int(state.get("global_step", -1)) == expected_step else 1)
PY
}

checkpoint_ready() {
    local ckpt=$1
    local expected_step=$2

    [[ -d "${ckpt}" ]] || return 1
    [[ -f "${ckpt}/trainer_state.json" ]] || return 1
    [[ -f "${ckpt}/config.json" ]] || return 1
    [[ -f "${ckpt}/model.safetensors.index.json" || -f "${ckpt}/model.safetensors" ]] || return 1
    checkpoint_has_expected_step "${ckpt}/trainer_state.json" "${expected_step}"
}

wait_for_checkpoint() {
    local ckpt=$1
    local expected_step=$2
    local label=$3

    echo "[$(timestamp)] Waiting for ${label}: ${ckpt} (global_step=${expected_step})"
    while ! checkpoint_ready "${ckpt}" "${expected_step}"; do
        echo "[$(timestamp)] ${label} not ready; checking again in ${POLL_INTERVAL}s"
        sleep "${POLL_INTERVAL}"
    done
    echo "[$(timestamp)] ${label} is ready; starting evaluation on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
}

model_name_from_ckpt() {
    local ckpt=$1

    local base_name
    base_name="$(basename "${ckpt}")"
    if [[ "${base_name}" == checkpoint-* ]]; then
        basename "$(dirname "${ckpt}")"
    else
        printf '%s\n' "${base_name}"
    fi
}

results_dir_from_ckpt() {
    local ckpt=$1
    local skill_eval_mode="${2:-normal}"
    local eval_tag="${EVAL_TAG:-}"
    local model_name

    model_name="$(model_name_from_ckpt "${ckpt}")"
    if [[ -n "${eval_tag}" ]]; then
        model_name="${model_name}_${eval_tag}"
    fi
    if [[ "${skill_eval_mode}" != "normal" ]]; then
        printf 'results/%s_skill%s\n' "${model_name}" "${skill_eval_mode}"
    else
        printf 'results/%s\n' "${model_name}"
    fi
}

standard_result_path() {
    local ckpt=$1
    local seed=$2
    local horizon=$3
    local skill_eval_mode="${4:-normal}"
    local model_name
    local results_dir

    model_name="$(model_name_from_ckpt "${ckpt}")"
    results_dir="$(results_dir_from_ckpt "${ckpt}" "${skill_eval_mode}")"
    if [[ "${skill_eval_mode}" != "normal" ]]; then
        printf '%s/libero_eval_model%s_tasklibero_10_seed%s_h%s_skill%s.json\n' "${results_dir}" "${model_name}" "${seed}" "${horizon}" "${skill_eval_mode}"
    else
        printf '%s/libero_eval_model%s_tasklibero_10_seed%s_h%s.json\n' "${results_dir}" "${model_name}" "${seed}" "${horizon}"
    fi
}

pro_result_path() {
    local ckpt=$1
    local seed=$2
    local horizon=$3
    local perturb=$4
    local skill_eval_mode="${5:-normal}"
    local model_name
    local results_dir

    model_name="$(model_name_from_ckpt "${ckpt}")"
    results_dir="$(results_dir_from_ckpt "${ckpt}" "${skill_eval_mode}")"
    if [[ "${skill_eval_mode}" != "normal" ]]; then
        printf '%s/libero_pro_model%s_tasklibero_10_pert%s_seed%s_h%s_skill%s.json\n' "${results_dir}" "${model_name}" "${perturb}" "${seed}" "${horizon}" "${skill_eval_mode}"
    else
        printf '%s/libero_pro_model%s_tasklibero_10_pert%s_seed%s_h%s.json\n' "${results_dir}" "${model_name}" "${perturb}" "${seed}" "${horizon}"
    fi
}

result_complete() {
    local result_path=$1
    local expected_episodes=$2

    python - "${result_path}" "${expected_episodes}" <<'PY'
import json
import sys

result_path = sys.argv[1]
expected_episodes = int(sys.argv[2])

try:
    with open(result_path, "r", encoding="utf-8") as f:
        result = json.load(f)
except Exception:
    sys.exit(1)

try:
    total_episodes = int(result.get("total_episodes", -1))
except (TypeError, ValueError):
    sys.exit(1)

sys.exit(0 if total_episodes == expected_episodes else 1)
PY
}

run_eval() {
    local ckpt=$1
    local seed=$2
    local horizon=$3
    local skill_eval_mode="${SKILL_EVAL_MODE:-normal}"

    echo "[$(timestamp)] Standard eval: ckpt=${ckpt} seed=${seed} horizon=${horizon} skill_eval_mode=${skill_eval_mode}"
    python -m libero_scripts.libero_eval \
        --model_path "${ckpt}" --task_suite_name libero_10 \
        --num_trials_per_task "${NUM_TRIAL}" --num_steps_wait 10 \
        --embodiment_tag new_embodiment --data_config libero_original \
        --denoising_steps 8 --action_horizon "${horizon}" --random_seed "${seed}" \
        --skill_eval_mode "${skill_eval_mode}" \
        ${GATE_PROBE_ARGS:-}
}

run_pro_eval() {
    local ckpt=$1
    local seed=$2
    local horizon=$3
    local perturb=$4
    local skill_eval_mode="${SKILL_EVAL_MODE:-normal}"

    echo "[$(timestamp)] PRO eval: ckpt=${ckpt} seed=${seed} horizon=${horizon} perturb=${perturb} skill_eval_mode=${skill_eval_mode}"
    python -m libero_scripts.libero_pro_eval \
        --model_path "${ckpt}" --task_suite_name libero_10 \
        --perturbation_type "${perturb}" --num_trials_per_task "${NUM_TRIAL}" \
        --num_steps_wait 10 --embodiment_tag new_embodiment \
        --data_config libero_original --denoising_steps 8 \
        --action_horizon "${horizon}" --random_seed "${seed}" \
        --skill_eval_mode "${skill_eval_mode}" \
        ${GATE_PROBE_ARGS:-}
}

run_eval_matrix() {
    local ckpt=$1
    local label=$2
    local expected_episodes=$((LIBERO_10_TASKS * NUM_TRIAL))
    local result_path

    echo "[$(timestamp)] Running eval matrix for ${label}; expected_episodes=${expected_episodes}"
    for SEED in "${SEEDS[@]}"; do
        for H in "${HORIZONS[@]}"; do
            result_path="$(standard_result_path "${ckpt}" "${SEED}" "${H}")"
            if result_complete "${result_path}" "${expected_episodes}"; then
                echo "[$(timestamp)] SKIP completed standard eval: label=${label} seed=${SEED} horizon=${H} result=${result_path}"
            else
                run_eval "${ckpt}" "${SEED}" "${H}"
            fi
            for PERT in "${PERTURBATIONS[@]}"; do
                result_path="$(pro_result_path "${ckpt}" "${SEED}" "${H}" "${PERT}")"
                if result_complete "${result_path}" "${expected_episodes}"; then
                    echo "[$(timestamp)] SKIP completed PRO eval: label=${label} seed=${SEED} horizon=${H} perturb=${PERT} result=${result_path}"
                else
                    run_pro_eval "${ckpt}" "${SEED}" "${H}" "${PERT}"
                fi
            done
        done
    done
    echo "[$(timestamp)] Finished eval matrix for ${label}"
}

main() {
    local task_id="${SLURM_ARRAY_TASK_ID:-0}"

    if ! [[ "${task_id}" =~ ^[0-9]+$ ]]; then
        echo "SLURM_ARRAY_TASK_ID must be an integer, got: ${task_id}" >&2
        exit 1
    fi

    if (( task_id < 0 || task_id >= ${#RUN_LABELS[@]} )); then
        echo "SLURM_ARRAY_TASK_ID=${task_id} is out of range. Submit with --array=0-$((${#RUN_LABELS[@]} - 1))." >&2
        exit 1
    fi

    local label="${RUN_LABELS[task_id]}"
    local expected_step="${RUN_STEPS[task_id]}"
    local ckpt="/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/${label}/checkpoint-${expected_step}"

    echo "[$(timestamp)] Array task ${task_id}/${#RUN_LABELS[@]}: label=${label} expected_step=${expected_step}"
    # wait_for_checkpoint "${ckpt}" "${expected_step}" "${label}"
    run_eval_matrix "${ckpt}" "${label}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
