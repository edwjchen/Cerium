#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Siddharth Jayashankar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPO_ROOT_NAME="$(basename "${REPO_ROOT}")"
IMAGE_NAME="${CERIUM_IMAGE:-cerium/build-cu13.3:v1}"
PYTHON="${REPO_ROOT}/venv_cerium/bin/python3"
USER_NAME="$(id -un 2>/dev/null || id -u)"
USER_ID="$(id -u):$(id -g)"
CONTAINER_NAME="CeriumBuild_cu13.3_${USER_NAME}_${REPO_ROOT_NAME}"

if ! docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" >/dev/null 2>&1 || \
    [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" != "true" ]]; then
    docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    docker run --user "${USER_ID}" -v "${REPO_ROOT}:${REPO_ROOT}" \
        --name "${CONTAINER_NAME}" --rm -d "${IMAGE_NAME}" tail -f /dev/null >/dev/null
fi

if ! docker exec -w "${SCRIPT_DIR}" "${CONTAINER_NAME}" \
    "${PYTHON}" -c 'import matplotlib' >/dev/null 2>&1; then
    docker exec -w "${SCRIPT_DIR}" "${CONTAINER_NAME}" \
        "${PYTHON}" -m pip install 'matplotlib==3.10.8'
fi

docker exec -w "${SCRIPT_DIR}" -e "MPLCONFIGDIR=/tmp/cerium-matplotlib-${USER_NAME}" \
    "${CONTAINER_NAME}" "${PYTHON}" plot_fusion.py
