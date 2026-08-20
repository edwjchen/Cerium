#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Siddharth Jayashankar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot multi-GPU bootstrap ablation results from the benchmark logs."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR / "logs"
OUTPUT_DIR = SCRIPT_DIR 
GPU_COUNTS = (2, 4, 8)
AVERAGE_TIME_PATTERN = re.compile(r"Average Execution Time = ([0-9.]+) milliseconds")


def read_average_time(log_file: Path) -> float:
    """Return the final average execution time reported in ``log_file``."""
    matches = AVERAGE_TIME_PATTERN.findall(log_file.read_text())
    if not matches:
        raise ValueError(f"No average execution time found in {log_file}")
    return float(matches[-1])


def collect_results() -> dict[str, float | list[float]]:
    """Collect the benchmark measurements used by the multi-GPU ablation."""
    return {
        "Baseline": read_average_time(LOG_DIR / "bootstrap_baseline_1gpus.log"),
        "No Opt": [
            read_average_time(LOG_DIR / f"bootstrap_no_cross_chip_opt_{gpus}gpus.log")
            for gpus in GPU_COUNTS
        ],
        "Cerium Schedule": [
            read_average_time(LOG_DIR / f"bootstrap_cerium_schedule_{gpus}gpus.log")
            for gpus in GPU_COUNTS
        ],
        "Cerium Opt": [
            read_average_time(LOG_DIR / f"bootstrap_cerium_opt_{gpus}gpus.log")
            for gpus in GPU_COUNTS
        ],
    }


def plot_bar_graph_multi_gpu(output_file: Path) -> None:
    vals = collect_results()
    baseline = vals["Baseline"]
    assert isinstance(baseline, float)

    fig, ax = plt.subplots()
    x_positions = np.arange(4)
    width = 0.8

    # The first bar is the one-GPU baseline.  Each subsequent bar is stacked
    # so its top denotes the speedup delivered by the final configuration.
    ax.bar(x_positions[0], 1, width=width, label="Baseline", color="blue")

    colors = {
        "No Opt": "#d2691e",
        "Cerium Schedule": "#75A702",
        "Cerium Opt": "#dcc74a",
    }
    previous_speedups = np.zeros(len(GPU_COUNTS))
    for variant in ("No Opt", "Cerium Schedule", "Cerium Opt"):
        timings = vals[variant]
        assert isinstance(timings, list)
        speedups = baseline / np.array(timings)
        ax.bar(
            x_positions[1:],
            speedups - previous_speedups,
            bottom=previous_speedups,
            width=width,
            label=variant,
            color=colors[variant],
        )
        previous_speedups = speedups

    ax.set_xticks(x_positions, labels=["1x", "2x", "4x", "8x"])
    ax.set_ylabel("Speedup over 1 GPU")
    ax.set_ylim(0, 3)
    ax.grid(axis="y", linestyle=":")
    ax.legend(loc="upper left", ncols=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_file = OUTPUT_DIR / "MultiGPUPlots.pdf"
    plot_bar_graph_multi_gpu(output_file)
    print(f"Wrote {output_file}")
