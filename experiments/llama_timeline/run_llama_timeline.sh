#!/usr/bin/env bash
# Profile one 4-GPU, one-block Llama inference and produce a GPU timeline.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCHMARK_DIR="${REPO_ROOT}/benchmarks/llama"
IMAGE_NAME="${CERIUM_IMAGE:-cerium/build-cu13.3:v1}"
PYTHON="${REPO_ROOT}/venv_cerium/bin/python3"
ARTIFACT_DIR="${SCRIPT_DIR}/artifacts"
REPORT_BASE="${ARTIFACT_DIR}/llama_4gpu_1block"
SQLITE_PATH="${REPORT_BASE}.sqlite"
RUN_LOG="${ARTIFACT_DIR}/llama_timeline.log"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage: run_llama_timeline.sh [--data_sample N]

Profiles a one-block encrypted Llama inference on four GPUs with CUDA Graphs
enabled. Nsight Systems records CUDA Graph nodes and the llama.program.run
NVTX range, then exports and plots compute, NCCL, and host-to-device memcpy
lanes for each GPU.

The required Llama model inputs, compiled 4-GPU/1-block programs, remappable
plaintexts, and ciphertext inputs must have been generated beforehand.
EOF
}

DATA_SAMPLE=0
while (( $# > 0 )); do
    case "$1" in
        --data_sample)
            (( $# >= 2 )) || fail "--data_sample requires an integer value."
            DATA_SAMPLE="$2"
            shift 2
            ;;
        --data_sample=*)
            DATA_SAMPLE="${1#*=}"
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            fail "Unknown argument: $1. Run with --help for usage."
            ;;
    esac
done

[[ "${DATA_SAMPLE}" =~ ^[0-9]+$ ]] || \
    fail "--data_sample must be a non-negative integer; got: ${DATA_SAMPLE}"
command -v docker >/dev/null 2>&1 || fail "docker is not installed or is not on PATH."
docker info >/dev/null 2>&1 || fail "Docker is unavailable to the current user."
[[ -x "${PYTHON}" ]] || fail "Python virtual environment not found at ${PYTHON}. Run the project setup first."

mkdir -p "${ARTIFACT_DIR}"
exec > >(tee "${RUN_LOG}") 2>&1

echo "Profiling Llama: 4 GPUs, 1 block, data sample ${DATA_SAMPLE}, CUDA Graph nodes."
docker run --rm --gpus all \
    --user "$(id -u):$(id -g)" \
    -v "${REPO_ROOT}:${REPO_ROOT}" \
    -v /opt/dlami/nvme/cerium:/opt/dlami/nvme/cerium \
    -w "${BENCHMARK_DIR}" \
    -e "CERIUM_RUNTIME_NUM_ITERS=1" \
    -e "CERIUM_RUNTIME_REMAPABLES_USE_EXPLICIT_COPY=1" \
    "${IMAGE_NAME}" \
    nsys profile --force-overwrite=true --trace=cuda,nvtx --sample=none \
        --cuda-graph-trace=node \
        --output "${REPORT_BASE}" \
        "${PYTHON}" "${SCRIPT_DIR}/llama_timeline_runner.py" \
            --gpus 4 --data_sample "${DATA_SAMPLE}" --block_start 0 \
            --num_blocks 1 --cudagraphs true

echo "Exporting Nsight Systems CUDA activity to SQLite."
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "${REPO_ROOT}:${REPO_ROOT}" \
    -w "${SCRIPT_DIR}" \
    "${IMAGE_NAME}" \
    nsys export --type sqlite --force-overwrite=true \
        --output "${SQLITE_PATH}" "${REPORT_BASE}.nsys-rep"

echo "Writing timestamp CSVs and timeline plot."
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "${REPO_ROOT}:${REPO_ROOT}" \
    -w "${SCRIPT_DIR}" \
    -e "MPLCONFIGDIR=${ARTIFACT_DIR}/.matplotlib" \
    "${IMAGE_NAME}" \
    "${PYTHON}" analyze_llama_timeline.py \
        --sqlite "${SQLITE_PATH}" --output-dir "${ARTIFACT_DIR}"
