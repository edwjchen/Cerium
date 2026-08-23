#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/bert_rte"
NUM_SAMPLES=277
GPUS=8

fail() { echo "ERROR: $*" >&2; exit 1; }

usage() {
    echo "Usage: $(basename "$0") [--data-dir DIRECTORY] [--num-samples COUNT] [--gpus COUNT]"
}

while (( $# > 0 )); do
    case "$1" in
        --data-dir)
            (( $# >= 2 )) || fail "--data-dir requires a directory."
            DATA_DIR="$2"
            shift 2
            ;;
        --num-samples|--num_samples|---num_samples)
            (( $# >= 2 )) || fail "$1 requires a sample count."
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --gpus)
            (( $# >= 2 )) || fail "--gpus requires a GPU count."
            GPUS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "Unknown argument: $1. Run $(basename "$0") --help for usage."
            ;;
    esac
done

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/bert_accuracy_$(date +%Y%m%d_%H%M%S).log"
printf 'BERT accuracy run log: %s\n' "${RUN_LOG}"
exec > >(tee -a "${RUN_LOG}") 2>&1

REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMAGE_NAME="${CERIUM_IMAGE:-cerium/build-cu13.3:v1}"
PYTHONPATH_DIR="${REPO_ROOT}/build/cerium/python"
RUNTIME_LIB_DIR="${REPO_ROOT}/build/cerium/runtime"
PYTHON="${REPO_ROOT}/venv_cerium/bin/python3"
DATA_URL="https://drive.usercontent.google.com/download?id=1RXgP0YYFbcgf25x3uTWrO42IOwTczZR7&export=download&authuser=0&confirm=t"
DATA_SHA256="d3065e221577214a58372a98e2aecf2e1555af058c27c5903a4f7dc83c877b80"

validate_archive() {
    local archive="$1"
    tar -tzf "${archive}" >/dev/null 2>&1 &&
        printf '%s  %s\n' "${DATA_SHA256}" "${archive}" | sha256sum --check --status -
}

case "${GPUS}" in
    1|2|4|8) ;;
    *) fail "GPU count must be one of: 1, 2, 4, 8." ;;
esac

command -v docker >/dev/null 2>&1 || fail "Docker is not installed or is not on PATH."
docker info >/dev/null 2>&1 || fail "Docker is not available to the current user."
[[ -x "${PYTHON}" ]] || fail "Python virtual environment not found at ${PYTHON}. Run the project setup first."
for required_path in packed_params/params_rte_block12.pkl bert_outputs/compiled; do
    [[ -e "${SCRIPT_DIR}/${required_path}" ]] || fail "Required BERT artifact is missing: ${SCRIPT_DIR}/${required_path}. Run ${SCRIPT_DIR}/compile.sh first."
done

mkdir -p "${DATA_DIR}"
DATA_DIR="$(cd "${DATA_DIR}" && pwd)"
ARCHIVE="${DATA_DIR}/bert_all.tar.gz"

if [[ ! -f "${DATA_DIR}/inputs/labels.npy" ]]; then
    if [[ -f "${ARCHIVE}" ]] && validate_archive "${ARCHIVE}"; then
        echo "Using existing BERT dataset archive at ${ARCHIVE}"
    else
        command -v curl >/dev/null 2>&1 || fail "curl is required to download the BERT dataset."
        DOWNLOAD_ARCHIVE="${ARCHIVE}.partial"
        echo "Downloading the complete BERT dataset to ${ARCHIVE}"
        curl --fail --location --show-error --retry 3 --output "${DOWNLOAD_ARCHIVE}" "${DATA_URL}"
        validate_archive "${DOWNLOAD_ARCHIVE}" || fail "BERT dataset download failed gzip or SHA-256 validation."
        mv "${DOWNLOAD_ARCHIVE}" "${ARCHIVE}"
    fi

    tar -tzf "${ARCHIVE}" inputs/labels.npy >/dev/null 2>&1 || fail "${ARCHIVE} is not the expected BERT dataset archive."
    echo "Extracting BERT dataset into ${DATA_DIR}"
    tar -xzf "${ARCHIVE}" --no-same-owner -C "${DATA_DIR}"
fi

AVAILABLE_SAMPLE_COUNT="$(find "${DATA_DIR}/inputs" -maxdepth 1 -type f -name 'inputs_*_data.npy' | wc -l)"
(( AVAILABLE_SAMPLE_COUNT > 0 )) || fail "The BERT dataset contains no input tensors."
if [[ -z "${NUM_SAMPLES}" ]]; then
    SAMPLE_COUNT="${AVAILABLE_SAMPLE_COUNT}"
else
    [[ "${NUM_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || fail "--num-samples must be a positive integer."
    SAMPLE_COUNT="${NUM_SAMPLES}"
fi
(( SAMPLE_COUNT <= AVAILABLE_SAMPLE_COUNT )) || fail "Requested ${SAMPLE_COUNT} samples, but only ${AVAILABLE_SAMPLE_COUNT} input tensors are available."
echo "Running BERT on ${SAMPLE_COUNT} sample(s) using ${GPUS} GPU(s)."

docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" \
    -v "${REPO_ROOT}:${REPO_ROOT}" \
    -v "${DATA_DIR}:${DATA_DIR}:ro" \
    -w "${SCRIPT_DIR}" \
    -e "PYTHONPATH=${PYTHONPATH_DIR}" \
    -e "LD_LIBRARY_PATH=${RUNTIME_LIB_DIR}" \
    "${IMAGE_NAME}" "${PYTHON}" bert_runner.py \
    --gpus "${GPUS}" \
    --num-samples "${SAMPLE_COUNT}" \
    --data-dir "${DATA_DIR}"
