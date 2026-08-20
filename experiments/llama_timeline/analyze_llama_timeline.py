#!/usr/bin/env python3
"""Extract and plot per-GPU CUDA activity from a Llama Nsight Systems export."""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def tables(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }


def columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}


def first_present(candidates: tuple[str, ...], available: set[str]) -> str | None:
    return next((candidate for candidate in candidates if candidate in available), None)


def kernel_table(available_tables: set[str]) -> str:
    for table in ("CUDA_GPU_KERNEL", "CUPTI_ACTIVITY_KIND_KERNEL"):
        if table in available_tables:
            return table
    if "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in available_tables:
        raise RuntimeError(
            "Nsight Systems recorded graph-level timing only. Re-profile with "
            "'--cuda-graph-trace=node' to collect CUDA Graph kernel nodes."
        )
    raise RuntimeError("No CUDA kernel activity table found in the SQLite export.")


def read_kernels(connection: sqlite3.Connection) -> list[dict[str, object]]:
    table = kernel_table(tables(connection))
    available = columns(connection, table)
    device_column = first_present(("deviceId", "device", "device_id"), available)
    if not {"start", "end"}.issubset(available) or device_column is None:
        raise RuntimeError(f"Unexpected {table} columns: {sorted(available)}")

    name_column = "nameId" if "nameId" in available else first_present(
        ("demangledName", "shortName", "name"), available
    )
    if name_column is not None and "StringIds" in tables(connection):
        # Nsight versions differ: some use nameId while others store a
        # StringIds key in demangledName/shortName.
        name_expression = f"COALESCE(strings.value, CAST(kernel.\"{name_column}\" AS TEXT), '<unnamed>')"
        join = f'LEFT JOIN StringIds AS strings ON kernel."{name_column}" = strings.id'
    elif name_column is not None:
        name_expression = f"COALESCE(CAST(kernel.\"{name_column}\" AS TEXT), '<unnamed>')"
        join = ""
    else:
        name_expression = "'<unnamed>'"
        join = ""

    query = f'''
        SELECT kernel.start, kernel.end, kernel."{device_column}", {name_expression}
        FROM "{table}" AS kernel
        {join}
        WHERE kernel.end > kernel.start
        ORDER BY kernel.start
    '''
    events = []
    for start, end, device, name in connection.execute(query):
        kernel_name = str(name) if name is not None else "<unnamed>"
        events.append(
            {
                "start_ns": int(start),
                "end_ns": int(end),
                "device_id": int(device),
                "category": "nccl" if "nccl" in kernel_name.lower() else "compute",
                "name": kernel_name,
            }
        )
    return events


def is_host_to_device(src: object, dst: object) -> bool:
    """Return whether Nsight's memory-kind pair denotes a host-to-device copy."""
    if src is None or dst is None:
        return False
    if isinstance(src, int) and isinstance(dst, int):
        # Nsight's CUDA_MEMOPR enum uses 0 for pageable host memory, 1 for
        # pinned host memory, and 2 for device memory.
        return src in (0, 1) and dst == 2
    return "host" in str(src).lower() and "device" in str(dst).lower()


def read_host_to_device_memcpys(connection: sqlite3.Connection) -> list[dict[str, object]]:
    available_tables = tables(connection)
    table = next((name for name in ("CUDA_GPU_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMCPY") if name in available_tables), None)
    if table is None:
        return []

    available = columns(connection, table)
    device_column = first_present(("deviceId", "device", "device_id"), available)
    src_column = first_present(("srcKind", "src_kind", "src"), available)
    dst_column = first_present(("dstKind", "dst_kind", "dst"), available)
    if not {"start", "end"}.issubset(available) or not all((device_column, src_column, dst_column)):
        return []

    query = f'''
        SELECT start, end, "{device_column}", "{src_column}", "{dst_column}"
        FROM "{table}"
        WHERE end > start
        ORDER BY start
    '''
    events = []
    for start, end, device, src, dst in connection.execute(query):
        if is_host_to_device(src, dst):
            events.append(
                {
                    "start_ns": int(start),
                    "end_ns": int(end),
                    "device_id": int(device),
                    "category": "host_to_device_memcpy",
                    "name": "Host-to-device memcpy",
                }
            )
    return events


def read_nvtx_range(connection: sqlite3.Connection, name: str) -> tuple[int, int]:
    available_tables = tables(connection)
    table = next((table for table in available_tables if table.upper() == "NVTX_EVENTS"), None)
    if table is None:
        raise RuntimeError(
            "The SQLite export has no NVTX events. Re-profile with "
            "'--trace=cuda,nvtx' to locate llama.program.run precisely."
        )
    available = columns(connection, table)
    if not {"start", "end"}.issubset(available):
        raise RuntimeError(f"Unexpected {table} columns: {sorted(available)}")
    name_column = first_present(("text", "textId", "message", "name", "nameId"), available)
    if name_column is None:
        raise RuntimeError(f"Cannot find an NVTX message column in {table}: {sorted(available)}")

    if name_column in {"textId", "nameId"} and "StringIds" in available_tables:
        name_expression = "strings.value"
        join = f'JOIN StringIds AS strings ON nvtx."{name_column}" = strings.id'
    else:
        name_expression = f'nvtx."{name_column}"'
        join = ""
    query = f'''
        SELECT nvtx.start, nvtx.end
        FROM "{table}" AS nvtx
        {join}
        WHERE {name_expression} = ? AND nvtx.end > nvtx.start
        ORDER BY nvtx.start
    '''
    ranges = [(int(start), int(end)) for start, end in connection.execute(query, (name,))]
    if not ranges:
        raise RuntimeError(f"NVTX range {name!r} was not found in {table}.")
    if len(ranges) > 1:
        raise RuntimeError(f"Expected one {name!r} NVTX range, found {len(ranges)}.")
    return ranges[0]


def event_overlaps_range(event: dict[str, object], start_ns: int, end_ns: int) -> bool:
    return int(event["end_ns"]) > start_ns and int(event["start_ns"]) < end_ns


def read_events(sqlite_path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], tuple[int, int]]:
    with sqlite3.connect(sqlite_path) as connection:
        kernels = read_kernels(connection)
        host_to_device_memcpys = read_host_to_device_memcpys(connection)
        program_run_range = read_nvtx_range(connection, "llama.program.run")
    if not kernels:
        raise RuntimeError("No CUDA kernels were recorded in the Nsight Systems report.")
    program_kernels = [event for event in kernels if event_overlaps_range(event, *program_run_range)]
    if not program_kernels:
        raise RuntimeError("No CUDA kernels overlap the llama.program.run NVTX range.")
    program_events = [
        event for event in kernels + host_to_device_memcpys
        if event_overlaps_range(event, *program_run_range)
    ]
    return program_kernels, sorted(program_events, key=lambda event: int(event["start_ns"])), program_run_range


def write_csvs(kernels: list[dict[str, object]], events: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "kernel_times.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=("gpu", "start_ns", "end_ns", "duration_ns", "kernel_name", "category"))
        writer.writeheader()
        for event in events:
            writer.writerow({
                "gpu": event["device_id"], "start_ns": event["start_ns"], "end_ns": event["end_ns"],
                "duration_ns": int(event["end_ns"]) - int(event["start_ns"]),
                "kernel_name": event["name"], "category": event["category"],
            })
    with (output_dir / "timeline_events.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=("gpu", "start_ns", "end_ns", "duration_ns", "event_name", "category"))
        writer.writeheader()
        for event in kernels:
            writer.writerow({
                "gpu": event["device_id"], "start_ns": event["start_ns"], "end_ns": event["end_ns"],
                "duration_ns": int(event["end_ns"]) - int(event["start_ns"]),
                "event_name": event["name"], "category": event["category"],
            })


def plot_timeline(events: list[dict[str, object]], output_path: Path, unit: str) -> None:
    divisor = 1_000_000 if unit == "ms" else 1_000
    first_start = min(int(event["start_ns"]) for event in events)
    devices = sorted({int(event["device_id"]) for event in events})
    categories = ("compute", "nccl", "host_to_device_memcpy")
    colors = {"compute": "#1f77b4", "nccl": "#d62728", "host_to_device_memcpy": "#2ca02c"}
    labels = {"compute": "Compute kernels", "nccl": "NCCL kernels", "host_to_device_memcpy": "Host→device memcpy"}

    lane_height, lane_gap, gpu_gap = 0.5, 0.12, 0.55
    height_per_gpu = len(categories) * (lane_height + lane_gap) + gpu_gap
    figure, axis = plt.subplots(figsize=(12, max(3.5, 1.45 * len(devices))), constrained_layout=True)

    y_ticks, y_labels = [], []
    for gpu_index, device in enumerate(devices):
        base_y = gpu_index * height_per_gpu
        for category_index, category in enumerate(categories):
            y = base_y + category_index * (lane_height + lane_gap)
            intervals = [
                ((int(event["start_ns"]) - first_start) / divisor, (int(event["end_ns"]) - int(event["start_ns"])) / divisor)
                for event in events
                if int(event["device_id"]) == device and event["category"] == category
            ]
            if intervals:
                axis.broken_barh(intervals, (y, lane_height), facecolors=colors[category], edgecolors="none")
        group_height = len(categories) * (lane_height + lane_gap) - lane_gap
        y_ticks.append(base_y + group_height / 2)
        y_labels.append(f"GPU {device}")

    axis.set_yticks(y_ticks)
    axis.set_yticklabels(y_labels)
    axis.set_xlabel(f"Time since first CUDA event ({unit})")
    axis.set_title("Llama program.run(): 4 GPUs, 1 encrypted transformer block")
    axis.grid(axis="x", linestyle="--", alpha=0.35)
    axis.legend(
        handles=[Patch(facecolor=colors[category], label=labels[category]) for category in categories],
        loc="upper right",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True, help="Nsight Systems SQLite export")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--unit", choices=("ms", "us"), default="ms")
    args = parser.parse_args()

    kernels, events, program_run_range = read_events(args.sqlite)
    write_csvs(kernels, events, args.output_dir)
    plot_path = args.output_dir / "llama_4gpu_1block_timeline.png"
    plot_timeline(events, plot_path, args.unit)
    (args.output_dir / "program_run_window.txt").write_text(
        f"NVTX range: llama.program.run\nStart: {program_run_range[0]} ns\n"
        f"End: {program_run_range[1]} ns\n"
        "Included CUDA events overlap this host-side program.run() range.\n"
    )
    print(f"Kernel timestamps: {args.output_dir / 'kernel_times.csv'}")
    print(f"Program-run timeline events: {args.output_dir / 'timeline_events.csv'}")
    print(f"program.run NVTX range: {program_run_range[0]} ns to {program_run_range[1]} ns")
    print(f"Timeline plot: {plot_path}")


if __name__ == "__main__":
    main()
