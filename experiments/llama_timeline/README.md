# Llama 4-GPU timeline

Run one encrypted Llama transformer block on four GPUs under Nsight Systems:

```bash
bash run_llama_timeline.sh --data_sample 0
```

The experiment uses its profiling-only `llama_timeline_runner.py` entry point to add an
`llama.program.run` NVTX range around the actual `program.run()` call. The
normal `llama_runner.py` is not modified. It requires the normal Llama
benchmark setup first: the Llama
checkpoint and inputs, compiled 4-GPU/1-block programs, remappable plaintexts,
and serialized ciphertext inputs. The run enables CUDA Graphs and profiles them
at node granularity (`--cuda-graph-trace=node`).

Results are written to `artifacts/`:

- `llama_4gpu_1block.nsys-rep` — raw Nsight Systems profile.
- `llama_4gpu_1block.sqlite` — SQLite export of the CUDA activity.
- `program_run_window.txt` — start/end timestamps of the `llama.program.run`
  NVTX range.
- `kernel_times.csv` — CUDA kernels overlapping that range, with GPU,
  start time, end time, duration, name, and classification (compute or NCCL).
- `timeline_events.csv` — compute, NCCL, and explicit weight H→D-copy events
  overlapping `program.run()`.
- `llama_4gpu_1block_timeline.png` — separate compute, NCCL, and H→D-copy
  lanes for each GPU, restricted to the `program.run()` section.
- `llama_timeline.log` — profiling and analysis output.

The analyzer can also be rerun on an existing report export:

```bash
python analyze_llama_timeline.py \
  --sqlite artifacts/llama_4gpu_1block.sqlite \
  --output-dir artifacts
```
