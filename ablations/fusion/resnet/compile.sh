#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Siddharth Jayashankar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPO_ROOT_NAME="$(basename "${REPO_ROOT}")"
IMAGE_NAME="${CERIUM_IMAGE:-cerium/build-cu13.3:v1}"
USER_NAME="$(id -un 2>/dev/null || id -u)"
USER_ID="$(id -u):$(id -g)"
CONTAINER_NAME="CeriumBuild_cu13.3_${USER_NAME}_${REPO_ROOT_NAME}"
PYTHON="${REPO_ROOT}/venv_cerium/bin/python3"
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/resnet_compile_$(date +%Y%m%d_%H%M%S).log"
exec 3>&1
printf 'Fusion ResNet compile started. Log: %s\n' "${RUN_LOG}" >&3
exec >"${RUN_LOG}" 2>&1
trap 'status=$?; printf "Fusion ResNet compile %s. Log: %s\\n" "$([[ $status -eq 0 ]] && echo completed || echo failed)" "${RUN_LOG}" >&3' EXIT

ensure_container_running() {
    if docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" >/dev/null 2>&1 && \
        [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" == "true" ]]; then
        return
    fi
    docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    docker run --user "${USER_ID}" -v "${REPO_ROOT}:${REPO_ROOT}" \
        --name "${CONTAINER_NAME}" --rm -d "${IMAGE_NAME}" tail -f /dev/null >/dev/null
}

compile_variant() {
    local prefix="$1"
    local horizontal_fusion="$2"
    local undo_horizontal_fusion="$3"
    local no_vertical_fusion="$4"
    docker exec -w "${SCRIPT_DIR}" \
        -e CERIUM_HORIZONTAL_FUSION="${horizontal_fusion}" \
        -e CERIUM_UNDO_HORIZONTAL_FUSION="${undo_horizontal_fusion}" \
        -e CERIUM_NO_VERTICAL_FUSION="${no_vertical_fusion}" \
        "${CONTAINER_NAME}" "${PYTHON}" resnet.py --gpus 1 --prefix "${prefix}"
}

ensure_container_running
docker exec -w "${REPO_ROOT}/benchmarks/resnet20" "${CONTAINER_NAME}" \
    "${PYTHON}" params.py

compile_variant no_fusion 0 1 1
compile_variant horizontal_fusion 1 0 1
compile_variant vertical_fusion 1 0 0

docker exec -w "${SCRIPT_DIR}" "${CONTAINER_NAME}" bash make_resnet.sh
