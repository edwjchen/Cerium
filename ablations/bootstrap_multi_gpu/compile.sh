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
REPO_ROOT_NAME="$(basename "${REPO_ROOT}")"
IMAGE_NAME="${CERIUM_IMAGE:-cerium/build-cu13.3:v1}"
USER_NAME="$(whoami)"
USER_ID="$(id -u):$(id -g)"
CONTAINER_NAME="CeriumBuild_cu13.3_${USER_NAME}_${REPO_ROOT_NAME}"
PYTHON="${REPO_ROOT}/venv_cerium/bin/python3"
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/bootstrap_compile_$(date +%Y%m%d_%H%M%S).log"
exec 3>&1
printf 'Bootstrap compile started. Log: %s\n' "${RUN_LOG}" >&3
exec >"${RUN_LOG}" 2>&1

print_summary() {
    local status=$?
    if (( status == 0 )); then
        printf 'Bootstrap compile completed. Log: %s\n' "${RUN_LOG}" >&3
    else
        printf 'Bootstrap compile failed (exit %d). Log: %s\n' "${status}" "${RUN_LOG}" >&3
    fi
}
trap print_summary EXIT

ensure_container_running() {
    if docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" >/dev/null 2>&1; then
        if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" == "true" ]]; then
            return
        fi

        docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    fi

    echo "Launching ${CONTAINER_NAME}..."
    docker run --user "${USER_ID}" -v "${REPO_ROOT}:${REPO_ROOT}" \
        --name "${CONTAINER_NAME}" --rm -d "${IMAGE_NAME}" tail -f /dev/null >/dev/null
}

ensure_container_running

compile_variant() {
    local prefix="$1"
    shift

    for gpus in "$@"; do
        docker exec -w "${SCRIPT_DIR}" \
            -e CERIUM_HORIZONTAL_FUSION="${CERIUM_HORIZONTAL_FUSION}" \
            -e CERIUM_SKIP_RECEIVE_REORDER="${CERIUM_SKIP_RECEIVE_REORDER}" \
            -e CERIUM_REDUCTION_REWRITE="${CERIUM_REDUCTION_REWRITE}" \
            -e CERIUM_CROSS_CHIP_OPTIMIZE="${CERIUM_CROSS_CHIP_OPTIMIZE}" \
            "${CONTAINER_NAME}" \
            "${PYTHON}" bootstrap.py --gpus "${gpus}" --prefix "${prefix}"
    done
}

CERIUM_HORIZONTAL_FUSION=0
CERIUM_SKIP_RECEIVE_REORDER=1
CERIUM_REDUCTION_REWRITE=0
CERIUM_CROSS_CHIP_OPTIMIZE=0
compile_variant no_cross_chip_opt 2 4 8

CERIUM_HORIZONTAL_FUSION=0
CERIUM_SKIP_RECEIVE_REORDER=1
CERIUM_REDUCTION_REWRITE=0
CERIUM_CROSS_CHIP_OPTIMIZE=0
compile_variant cerium_schedule 2 4 8

CERIUM_HORIZONTAL_FUSION=1
CERIUM_SKIP_RECEIVE_REORDER=0
CERIUM_REDUCTION_REWRITE=1
CERIUM_CROSS_CHIP_OPTIMIZE=1
compile_variant cerium_opt 2 4 8

docker exec -w "${SCRIPT_DIR}" "${CONTAINER_NAME}" \
    "${PYTHON}" bootstrap.py --gpus 1 --prefix baseline

docker exec -w "${SCRIPT_DIR}" "${CONTAINER_NAME}" bash make_bootstrap.sh
