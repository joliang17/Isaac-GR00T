#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/eval_libero_wait_common.sh"

LABEL="libero10_256_full_cls_stage2_v2"
CKPT="/fs/nexus-projects/wilddiffusion/vla/GR00T/checkpoint/${LABEL}/checkpoint-60000"
EXPECTED_STEP=60000

# wait_for_checkpoint "${CKPT}" "${EXPECTED_STEP}" "${LABEL}"
run_eval_matrix "${CKPT}" "${LABEL}"
