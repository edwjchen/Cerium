#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Siddharth Jayashankar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MAKEFILE="${REPO_ROOT}/ablations/bootstrap_multi_gpu/Makefile"
CERIUM_KERNELS_PATH="${REPO_ROOT}/cerium/compiler/backend/kernels"
FUNCTION="bootstrap_32Kslots_33levels_1gpus"

compile_variant() {
    local variant="$1"
    mkdir -p "${variant}/compiled/${FUNCTION}"
    make -f "${MAKEFILE}" all -j FNAME="${FUNCTION}" DIRNAME="${variant}" \
        CERIUM_KERNELS_PATH="${CERIUM_KERNELS_PATH}"
    cp -r "${variant}/${FUNCTION}/output/." "${variant}/compiled/${FUNCTION}/"
}

compile_variant no_fusion
compile_variant horizontal_fusion
compile_variant vertical_fusion
