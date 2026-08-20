# CUDA Graph kernel count

Run `bash run_kernel_count.sh`. 

The experiment writes its results to `artifacts/`:

- `bootstrap_cudagraph.nsys-rep` — the raw Nsight Systems report.
- `bootstrap_cudagraph.sqlite` — the Nsight Systems SQLite export. The profile
  uses node-level CUDA Graph tracing so this contains individual kernel events.
- `kernel_times.csv` — each CUDA kernel's start, end, and execution time in ns.
- `kernel_count.txt` — the reported number of recorded CUDA kernels.
- `kernel_duration_cdf.png` — CDF of kernel execution time in µs.
- `kernel_count.log` — complete console output from the profiling run.

The bootstrap must already be compiled, following the repository's normal
bootstrap benchmark setup.
