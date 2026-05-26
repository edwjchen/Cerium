#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Siddharth Jayashankar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
IMAGE_NAME="${CERIUM_IMAGE:-cerium/build-cu13.3:v1}"
LOG_DIR="${SCRIPT_DIR}/logs"
PYTHONPATH_DIR="${REPO_ROOT}/build/cerium/python"
RUNTIME_LIB_DIR="${REPO_ROOT}/build/cerium/runtime"
PYTHON="${REPO_ROOT}/venv_cerium/bin/python3"
NUM_SAMPLES="${RESNET_NUM_SAMPLES:-5}"

mkdir -p "${LOG_DIR}"

run_variant() {
    local label="$1"
    local base_dir="$2"
    local use_cudagraphs="$3"
    local log_file="${LOG_DIR}/resnet_${label}.log"
    echo "Running ${label}."
    docker run --rm --gpus all --ipc=host \
        --user "$(id -u):$(id -g)" \
        -v "${REPO_ROOT}:${REPO_ROOT}" \
        -w "${SCRIPT_DIR}" \
        -e "PYTHONPATH=${PYTHONPATH_DIR}" \
        -e "LD_LIBRARY_PATH=${RUNTIME_LIB_DIR}" \
        "${IMAGE_NAME}" "${PYTHON}" resnet_runner.py --gpus 1 \
        --base-dir "${SCRIPT_DIR}/${base_dir}" --num_samples "${NUM_SAMPLES}" \
        --use_cudagraphs "${use_cudagraphs}" >"${log_file}" 2>&1
    grep -q 'Average Execution Time =' "${log_file}"
}

run_variant no_fusion no_fusion false
run_variant horizontal_fusion horizontal_fusion false
run_variant vertical_fusion vertical_fusion false
run_variant cudagraph vertical_fusion true

"${SCRIPT_DIR}/plot_fusion.sh"
