#!/usr/bin/env bash
# Profile one CUDA-Graph bootstrap and produce a kernel-duration CDF.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCHMARK_DIR="${REPO_ROOT}/benchmarks/bootstrap_32K"
IMAGE_NAME="${CERIUM_IMAGE:-cerium/build-cu13.3:v1}"
PYTHON="${REPO_ROOT}/venv_cerium/bin/python3"
PYTHONPATH_DIR="${REPO_ROOT}/build/cerium/python"
RUNTIME_LIB_DIR="${REPO_ROOT}/build/cerium/runtime"
ARTIFACT_DIR="${SCRIPT_DIR}/artifacts"
REPORT_BASE="${ARTIFACT_DIR}/bootstrap_cudagraph"
SQLITE_PATH="${ARTIFACT_DIR}/bootstrap_cudagraph.sqlite"
RUN_LOG="${ARTIFACT_DIR}/kernel_count.log"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

command -v docker >/dev/null 2>&1 || fail "docker is not installed or is not on PATH."
docker info >/dev/null 2>&1 || fail "Docker is unavailable to the current user."
[[ -x "${PYTHON}" ]] || fail "Python virtual environment not found at ${PYTHON}. Run the project setup first."

mkdir -p "${ARTIFACT_DIR}"
exec > >(tee "${RUN_LOG}") 2>&1

rm -f "${REPORT_BASE}" "${REPORT_BASE}.nsys-rep" "${SQLITE_PATH}" \
    "${ARTIFACT_DIR}/kernel_times.csv" "${ARTIFACT_DIR}/kernel_count.txt" \
    "${ARTIFACT_DIR}/kernel_duration_cdf.png"

echo "Profiling one bootstrap with CUDA Graphs enabled (CERIUM_RUNTIME_NUM_ITERS=1)."
docker run --rm --gpus all \
    --user "$(id -u):$(id -g)" \
    -v "${REPO_ROOT}:${REPO_ROOT}" \
    -w "${BENCHMARK_DIR}" \
    -e "PYTHONPATH=${PYTHONPATH_DIR}" \
    -e "LD_LIBRARY_PATH=${RUNTIME_LIB_DIR}" \
    -e "CERIUM_RUNTIME_NUM_ITERS=1" \
    "${IMAGE_NAME}" \
    nsys profile --force-overwrite=true --trace=cuda --sample=none \
        --cuda-graph-trace=node \
        --output "${REPORT_BASE}" \
        "${PYTHON}" bootstrap_runner.py --gpus 1 --use_cudagraphs true

echo "Exporting Nsight Systems CUDA kernel timestamps."
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "${REPO_ROOT}:${REPO_ROOT}" \
    -w "${SCRIPT_DIR}" \
    "${IMAGE_NAME}" \
    nsys export --type sqlite --force-overwrite=true \
        --output "${SQLITE_PATH}" "${REPORT_BASE}.nsys-rep"

docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "${REPO_ROOT}:${REPO_ROOT}" \
    -w "${SCRIPT_DIR}" \
    -e "MPLCONFIGDIR=${ARTIFACT_DIR}/.matplotlib" \
    "${IMAGE_NAME}" \
    "${PYTHON}" analyze_kernels.py \
        --sqlite "${SQLITE_PATH}" \
        --output-dir "${ARTIFACT_DIR}"

