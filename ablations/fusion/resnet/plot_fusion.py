#!/usr/bin/env python3
"""Plot the single-GPU ResNet fusion ablation from benchmark logs."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR / "logs"
OUTPUT_FILE = SCRIPT_DIR / "figs" / "ResNetFusion.pdf"
PATTERN = re.compile(r"Average Execution Time = ([0-9.]+) milliseconds")
VARIANTS = ("no_fusion", "horizontal_fusion", "vertical_fusion", "cudagraph")
LABELS = ("No fusion", "+Horizontal", "+Vertical", "+CUDA graph")
COLORS = ("#d2691e", "#75A702", "#dcc74a", "#4e4c11")


def average_time(variant: str) -> float:
    matches = PATTERN.findall((LOG_DIR / f"resnet_{variant}.log").read_text())
    if not matches:
        raise ValueError(f"No average execution time found for {variant}")
    return float(matches[-1])


def main() -> None:
    timings = np.array([average_time(variant) for variant in VARIANTS])
    speedups = timings[0] / timings
    fig, ax = plt.subplots()
    previous_speedup = 0.0
    for label, color, speedup in zip(LABELS, COLORS, speedups):
        ax.bar(0, speedup - previous_speedup, bottom=previous_speedup, color=color, label=label)
        previous_speedup = speedup
    ax.set_xticks([0], labels=["ResNet"])
    ax.set_ylabel("Speedup over no fusion")
    ax.set_ylim(0, max(1.1, speedups.max() * 1.15))
    ax.grid(axis="y", linestyle=":")
    ax.legend()
    fig.tight_layout()
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_FILE)
    plt.close(fig)
    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
