#!/bin/bash
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



set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
FUNCTION="keyswitch_1gpus"

compile_variant() {
    local variant="$1"
    mkdir -p "${variant}/compiled/${FUNCTION}"
    make -f "${REPO_ROOT}/ablations/bootstrap_multi_gpu/Makefile" all -j \
        FNAME="${FUNCTION}" \
        DIRNAME="${variant}" \
        CERIUM_KERNELS_PATH="${REPO_ROOT}/cerium/compiler/backend/kernels"
    cp -r "${variant}/${FUNCTION}/output/." "${variant}/compiled/${FUNCTION}/"
}

compile_variant no_fusion
compile_variant horizontal_fusion
compile_variant vertical_fusion
