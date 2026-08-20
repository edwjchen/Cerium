#!/usr/bin/env python3
"""Extract CUDA kernel timing from an Nsight Systems SQLite export and plot its CDF."""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def kernel_table(connection: sqlite3.Connection) -> str:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    for table in ("CUDA_GPU_KERNEL", "CUPTI_ACTIVITY_KIND_KERNEL"):
        if table in tables:
            return table
    if "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in tables:
        raise RuntimeError(
            "Nsight Systems recorded graph-level timing only. Re-profile with "
            "'--cuda-graph-trace=node' to collect individual CUDA Graph kernels."
        )
    raise RuntimeError(
        "The Nsight Systems export has no CUDA kernel activity table. "
        "Profile with '--trace=cuda --cuda-graph-trace=node'."
    )


def read_kernels(sqlite_path: Path) -> list[tuple[int, int, str]]:
    with sqlite3.connect(sqlite_path) as connection:
        table = kernel_table(connection)
        columns = {
            row[1] for row in connection.execute(f"PRAGMA table_info({table})")
        }
        if not {"start", "end"}.issubset(columns):
            raise RuntimeError(f"Unexpected {table} columns: {sorted(columns)}")
        if "nameId" in columns:
            query = (
                f'SELECT kernel.start, kernel.end, COALESCE(strings.value, "<unnamed>") '
                f'FROM "{table}" AS kernel '
                "LEFT JOIN StringIds AS strings ON kernel.nameId = strings.id "
                "WHERE kernel.end > kernel.start ORDER BY kernel.start"
            )
        else:
            name_column = (
                "demangledName" if "demangledName" in columns else "shortName"
            )
            query = (
                f'SELECT start, end, "{name_column}" FROM "{table}" '
                "WHERE end > start ORDER BY start"
            )
        rows = connection.execute(query).fetchall()
    return [(int(start), int(end), name or "<unnamed>") for start, end, name in rows]


def write_outputs(kernels: list[tuple[int, int, str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "kernel_times.csv"
    with csv_path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("start_ns", "end_ns", "duration_ns", "kernel_name"))
        writer.writerows((start, end, end - start, name) for start, end, name in kernels)

    count_path = output_dir / "kernel_count.txt"
    count_path.write_text(f"Kernel count: {len(kernels)}\n")

    if not kernels:
        raise RuntimeError("No CUDA kernels were recorded in the Nsight Systems report.")

    durations_us = np.sort(np.array([end - start for start, end, _ in kernels]) / 1_000)
    cdf = np.arange(1, len(durations_us) + 1) / len(durations_us)
    p99_index = int(np.ceil(0.99 * len(durations_us))) - 1
    p99_us = durations_us[p99_index]
    x_max_us = max(10, int(np.ceil(p99_us / 10)) * 10)
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    axis.step(durations_us, cdf, where="post", color="#75A702")
    axis.set_xlim(0, x_max_us)
    axis.set_xticks(np.arange(10, x_max_us + 1, 10))
    axis.set_xlabel("Kernel execution time (µs)")
    axis.set_ylabel("Cumulative fraction")
    axis.grid(True, which="both", alpha=0.3)
    figure.savefig(output_dir / "kernel_duration_cdf.png", dpi=200)
    plt.close(figure)

    print(f"Kernel count: {len(kernels)}")
    print(f"Kernel timestamps: {csv_path}")
    print(f"Kernel CDF: {output_dir / 'kernel_duration_cdf.png'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", required=True, type=Path, help="Nsight Systems SQLite export")
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    write_outputs(read_kernels(arguments.sqlite), arguments.output_dir)


if __name__ == "__main__":
    main()
