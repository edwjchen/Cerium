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

compile_all() {
    local function_name="$1"
    local cerium_dir="$2"
    local compiled_dir="$3"
    echo "${function_name} ${cerium_dir} ${compiled_dir}"
    mkdir -p "${compiled_dir}/${function_name}"
    make all -j FNAME="${function_name}" DIRNAME="${cerium_dir}"
    cp -r "${cerium_dir}/${function_name}/output/." "${compiled_dir}/${function_name}/"
}

CERIUM_DIR="outputs"
COMPILED_DIR="${CERIUM_DIR}/compiled"

for function_name in \
    bootstrap_32K_35lvl_t1_vec \
    bootstrap_32K_33lvl_t1_vec \
    matmul_ct128x128_ct128x128 \
    matmul_ct128x128_ct128x128_transpose \
    softmax_128x128_vec \
    silu \
    ; do
    compile_all "${function_name}" "${CERIUM_DIR}" "${COMPILED_DIR}"
done

for gpus in 1 2 4 8; do
    for function_name in attention_matmul_qkv attention_matmul_o rmsnorm_att rmsnorm_ffn ffn_matmul_vw ffn_matmul_o; do
        compile_all "${function_name}_${gpus}gpu" "${CERIUM_DIR}" "${COMPILED_DIR}"
    done
done

for gpus in 2 4 8; do
    for function_name in ffn_silu attention_score_softmax; do
        compile_all "${function_name}_${gpus}gpu" "${CERIUM_DIR}" "${COMPILED_DIR}"
    done
done


# compile_all "llama_1gpu_1blocks" "${CERIUM_DIR}" "${COMPILED_DIR}"
# compile_all "llama_1gpu_2blocks" "${CERIUM_DIR}" "${COMPILED_DIR}"
# compile_all "llama_1gpu_2blocks" "${CERIUM_DIR}" "${COMPILED_DIR}"
# compile_all "llama_1gpu_8blocks" "${CERIUM_DIR}" "${COMPILED_DIR}"
# compile_all "llama_1gpu_12blocks" "${CERIUM_DIR}" "${COMPILED_DIR}"
# compile_all "llama_1gpu_16blocks" "${CERIUM_DIR}" "${COMPILED_DIR}"
compile_all "llama_1gpu_32blocks" "${CERIUM_DIR}" "${COMPILED_DIR}"
# compile_all "main_2gpu_12" "${CERIUM_DIR}" "${COMPILED_DIR}"
# compile_all "main_4gpu_12" "${CERIUM_DIR}" "${COMPILED_DIR}"

# compile_all "llama_2gpu_1blocks" "${CERIUM_DIR}" "${COMPILED_DIR}"
# compile_all "llama_2gpu_2blocks" "${CERIUM_DIR}" "${COMPILED_DIR}"
# compile_all "llama_2gpu_4blocks" "${CERIUM_DIR}" "${COMPILED_DIR}"
# compile_all "llama_2gpu_8blocks" "${CERIUM_DIR}" "${COMPILED_DIR}"
# compile_all "llama_2gpu_16blocks" "${CERIUM_DIR}" "${COMPILED_DIR}"
# compile_all "llama_2gpu_32blocks" "${CERIUM_DIR}" "${COMPILED_DIR}"

compile_all "llama_4gpu_1blocks" "${CERIUM_DIR}" "${COMPILED_DIR}"

compile_all "llama_4gpu_2blocks" "${CERIUM_DIR}" "${COMPILED_DIR}"
compile_all "llama_4gpu_4blocks" "${CERIUM_DIR}" "${COMPILED_DIR}"
# compile_all "llama_4gpu_8blocks" "${CERIUM_DIR}" "${COMPILED_DIR}"
compile_all "llama_4gpu_32blocks" "${CERIUM_DIR}" "${COMPILED_DIR}"

compile_all "llama_8gpu_1blocks" "${CERIUM_DIR}" "${COMPILED_DIR}"
compile_all "llama_8gpu_32blocks" "${CERIUM_DIR}" "${COMPILED_DIR}"
