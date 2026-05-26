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




"""Generate inputs and execute the single-GPU keyswitch micro-ablation."""

import argparse
import os
import random

import numpy as np

import cerium.runtime as cerium_runtime
from primes import Primes


SLOTS = 32 * 1024
FUNCTION_NAME = "keyswitch_1gpus"


def generate_secret_key(slots: int, hamming_weight: int) -> list[int]:
    secret_key = [0] * (2 * slots)
    count = 0
    while count < hamming_weight:
        position = random.randint(0, 2 * slots - 1)
        if secret_key[position] != 0:
            continue
        secret_key[position] = -1 if random.randint(0, 1) == 0 else 1
        count += 1
    return secret_key


def run(base_dir: str, use_cudagraphs: bool) -> None:
    random.seed(10)
    np.random.seed(10)
    os.environ["CERIUM_RUNTIME_NUM_ITERS"] = "10"

    secret_key = generate_secret_key(SLOTS, SLOTS)
    ephemeral_key = generate_secret_key(SLOTS, 32)
    seed = np.random.randint(0, 2**32 - 1, size=8).tolist()

    context = cerium_runtime.Context(SLOTS, Primes)
    encryptor_context = cerium_runtime.CKKSEncryptorContext(
        context, secret_key, ephemeral_key, seed
    )
    encryptor = cerium_runtime.CKKSEncryptor(encryptor_context, seed)

    input_scale = Primes[1] * Primes[0] / (1 << 0)
    x = np.array(
        [np.exp(1j * random.uniform(0, 2 * np.pi)) for _ in range(SLOTS)],
        dtype=np.complex128,
    )
    encrypted_inputs = {"x": (x, input_scale)}

    compiled_dir = f"{base_dir}/compiled"
    io_generator = cerium_runtime.IOGenerator(context)
    io_generator.generate_and_serialize_evalkeys(
        f"{compiled_dir}/{FUNCTION_NAME}/evalkeys",
        f"{compiled_dir}/{FUNCTION_NAME}/program_inputs",
        encryptor_context,
    )
    raw_inputs = cerium_runtime.RawInputsWrapper({})
    io_generator.generate_and_serialize_plaintexts(
        f"{compiled_dir}/{FUNCTION_NAME}/plaintexts",
        f"{compiled_dir}/{FUNCTION_NAME}/program_inputs",
        raw_inputs,
    )
    program = cerium_runtime.FusedKernelProgram(context, compiled_dir, FUNCTION_NAME)
    config = {FUNCTION_NAME: {"use_cudagraph": str(use_cudagraphs)}}
    functions = program.make_program(1, config, raw_inputs)
    function = functions[FUNCTION_NAME]

    inputs = cerium_runtime.RawInputsWrapper(encrypted_inputs)
    ciphertext_inputs = io_generator.generate_ciphertext_inputs(
        f"{compiled_dir}/{FUNCTION_NAME}/program_inputs", inputs, encryptor_context
    )
    function.copy_ciphertext_inputs_to_gpu(
        f"{compiled_dir}/{FUNCTION_NAME}/inputs", 1, ciphertext_inputs
    )
    program.run()

    output = function.get_program_outputs()["z"]
    decoded = np.array(encryptor.decrypt_and_decode(output, input_scale))
    error = np.max(np.abs(decoded - x * x))
    print(f"Max Absolute Error: {error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default="outputs")
    parser.add_argument("--use-cudagraphs", action="store_true")
    args = parser.parse_args()
    run(args.base_dir, args.use_cudagraphs)
