#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Siddharth Jayashankar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MAKEFILE="${REPO_ROOT}/benchmarks/resnet20/Makefile"
CERIUM_KERNELS_PATH="${REPO_ROOT}/cerium/compiler/backend/kernels"

compile_variant() {
    local variant="$1"
    local function
    for function in relu bootstrap_32K_33_1gpus main_1gpus; do
        mkdir -p "${variant}/compiled/${function}"
        make -f "${MAKEFILE}" all -j FNAME="${function}" DIRNAME="${variant}" \
            CERIUM_KERNELS_PATH="${CERIUM_KERNELS_PATH}"
        cp -r "${variant}/${function}/output/." "${variant}/compiled/${function}/"
    done
}

compile_variant no_fusion
compile_variant horizontal_fusion
compile_variant vertical_fusion
