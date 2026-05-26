# SPDX-FileCopyrightText: Copyright (c) 2026 Siddharth Jayashankar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMAGE_NAME="${CERIUM_IMAGE:-cerium/build-cu13.3:v1}"
LOG_DIR="${SCRIPT_DIR}/logs"
PYTHONPATH_DIR="${REPO_ROOT}/build/cerium/python"
RUNTIME_LIB_DIR="${REPO_ROOT}/build/cerium/runtime"
PYTHON="${REPO_ROOT}/venv_cerium/bin/python3"

mkdir -p "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/bootstrap_benchmarks_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${RUN_LOG}") 2>&1

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

if ! command -v docker >/dev/null 2>&1; then
    fail "Docker is not installed or is not on PATH."
fi

if ! docker info >/dev/null 2>&1; then
    fail "Docker is not available to the current user. Check that the daemon is running and that you have permission to use it."
fi

if [[ ! -x "${PYTHON}" ]]; then
    fail "Python virtual environment not found at ${PYTHON}. Run the project setup before running benchmarks."
fi

if ! gpu_list="$(docker run --rm --gpus all "${IMAGE_NAME}" nvidia-smi -L)"; then
    fail "Could not query GPUs in ${IMAGE_NAME}. Check the image and NVIDIA Container Toolkit installation."
fi

GPU_COUNT="$(awk '/^GPU [0-9]+:/{ count++ } END { print count + 0 }' <<<"${gpu_list}")"

echo "Detected ${GPU_COUNT} GPU(s)."

if (( GPU_COUNT == 0 )); then
    echo "No GPUs are available in the container." >&2
    exit 1
fi

failed_runs=0

run_variant() {
    local variant="$1"
    local use_cudagraphs="$2"
    shift 2

    for gpus in "$@"; do
        if (( gpus > GPU_COUNT )); then
            echo "Skipping ${variant} (${gpus} GPUs): only ${GPU_COUNT} GPU(s) available."
            continue
        fi

        local log_file="${LOG_DIR}/bootstrap_${variant}_${gpus}gpus.log"
        echo "Running ${variant} (${gpus} GPUs)."
        if ! docker run --rm --gpus all \
            --user "$(id -u):$(id -g)" \
            -v "${REPO_ROOT}:${REPO_ROOT}" \
            -w "${SCRIPT_DIR}" \
            -e "PYTHONPATH=${PYTHONPATH_DIR}" \
            -e "LD_LIBRARY_PATH=${RUNTIME_LIB_DIR}" \
            "${IMAGE_NAME}" \
            "${PYTHON}" bootstrap_runner.py --gpus "${gpus}" --base-dir "${variant}" --use_cudagraphs "${use_cudagraphs}" >"${log_file}" 2>&1; then
            echo "${variant} (${gpus} GPUs) failed; see ${log_file}." >&2
            ((failed_runs += 1))
            continue
        fi

        local average
        average="$(awk -F ' = | milliseconds' '/Average Execution Time =/ { value=$2 } END { print value }' "${log_file}")"
        if [[ -z "${average}" ]]; then
            echo "${variant} (${gpus} GPUs) completed without an average execution time; see ${log_file}." >&2
            average="not found"
            ((failed_runs += 1))
        fi
    done
}

run_variant baseline true 1
run_variant no_cross_chip_opt false 2 4 8
run_variant cerium_schedule true 2 4 8
run_variant cerium_opt true 2 4 8

if (( failed_runs > 0 )); then
    fail "${failed_runs} benchmark run(s) did not complete successfully."
fi

echo "Generating multi-GPU ablation plot."
"${SCRIPT_DIR}/plot_multi_gpu.sh"
