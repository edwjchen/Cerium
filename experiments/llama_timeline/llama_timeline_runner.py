#!/usr/bin/env python3
"""Run the Llama benchmark with an NVTX range around Program.run().

This profiling-only entry point leaves ``benchmarks/llama/llama_runner.py``
unchanged. It replaces its FusedKernelProgram factory with a proxy whose
``run`` method adds the NVTX range consumed by this experiment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "llama"
sys.path.insert(0, str(BENCHMARK_DIR))

import llama_runner  # noqa: E402


PROGRAM_RUN_RANGE = "llama.program.run"
_FusedKernelProgram = llama_runner.cerium_runtime.FusedKernelProgram


class NvtxFusedKernelProgram:
    """Delegate to Cerium's program object while marking only ``run()``."""

    def __init__(self, *args, **kwargs):
        self._program = _FusedKernelProgram(*args, **kwargs)

    def run(self, *args, **kwargs):
        torch.cuda.nvtx.range_push(PROGRAM_RUN_RANGE)
        try:
            return self._program.run(*args, **kwargs)
        finally:
            torch.cuda.nvtx.range_pop()

    def __getattr__(self, name):
        return getattr(self._program, name)


llama_runner.cerium_runtime.FusedKernelProgram = NvtxFusedKernelProgram


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Llama runner with an NVTX range around program.run()."
    )
    parser.add_argument("--block_start", type=int, required=True)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument(
        "--cudagraphs",
        type=llama_runner.str_to_bool,
        default=True,
        help="Enable CUDAGraphs",
    )
    parser.add_argument("--num_blocks", type=int, required=True)
    parser.add_argument("--data_sample", type=int, default=0)
    args = parser.parse_args()
    llama_runner.main(
        args.gpus,
        args.data_sample,
        args.block_start,
        args.cudagraphs,
        args.num_blocks,
    )


if __name__ == "__main__":
    main()
