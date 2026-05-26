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



"""Single-GPU multiplication, keyswitch, and rescale micro-ablation."""

import argparse

from cerium.compiler import cerium_compile
from cerium.dsl import CeriumFunction, CeriumProgram, CiphertextInput, Output


def keyswitch_program() -> CeriumProgram:
    program = CeriumProgram("keyswitch", rns_bit_size=28, num_gpus=1)
    with program:
        function = CeriumFunction("keyswitch_1gpus", 1, 0)
        with function:
            x = CiphertextInput("x", 56, 48)
            z = (x * x).relinearize()
            Output("z", z)
    return program


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--vregs", type=int, default=4096)
    parser.add_argument("--prefix", type=str, default="outputs")
    args = parser.parse_args()

    if args.gpus != 1:
        raise ValueError("The keyswitch ablation supports exactly one GPU.")

    cerium_compile(keyswitch_program(), 51, args.gpus, args.vregs, 128, args.prefix)
