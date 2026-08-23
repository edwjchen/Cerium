#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/resnet_cifar"
NUM_SAMPLES=""
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
RUN_LOG="${LOG_DIR}/resnet_accuracy_$(date +%Y%m%d_%H%M%S).log"
printf 'ResNet accuracy run log: %s\n' "${RUN_LOG}"
exec > >(tee -a "${RUN_LOG}") 2>&1

REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMAGE_NAME="${CERIUM_IMAGE:-cerium/build-cu13.3:v1}"
PYTHONPATH_DIR="${REPO_ROOT}/build/cerium/python"
RUNTIME_LIB_DIR="${REPO_ROOT}/build/cerium/runtime"
PYTHON="${REPO_ROOT}/venv_cerium/bin/python3"
DATA_URL="https://drive.usercontent.google.com/download?id=1c4fbY8-D5POJHMXTMX60qX3Pageb4iaN&export=download&confirm=t"
DATA_SHA256="92aca2eab75cfdcd3a8ee7949fbfe4747cb0128e7656eb4be477eab423dc3789"

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
for required_path in resnet20_params.pkl outputs/compiled; do
    [[ -e "${SCRIPT_DIR}/${required_path}" ]] || fail "Required ResNet artifact is missing: ${SCRIPT_DIR}/${required_path}. Run ${SCRIPT_DIR}/compile.sh first."
done

mkdir -p "${DATA_DIR}"
DATA_DIR="$(cd "${DATA_DIR}" && pwd)"
ARCHIVE="${DATA_DIR}/resnet_all.tar.gz"

if [[ ! -f "${DATA_DIR}/testFile/test_label.txt" ]]; then
    if [[ -f "${ARCHIVE}" ]] && validate_archive "${ARCHIVE}"; then
        echo "Using existing ResNet dataset archive at ${ARCHIVE}"
    else
        command -v curl >/dev/null 2>&1 || fail "curl is required to download the ResNet dataset."
        DOWNLOAD_ARCHIVE="${ARCHIVE}.partial"
        echo "Downloading the complete ResNet dataset to ${ARCHIVE}"
        curl --fail --location --show-error --retry 3 --output "${DOWNLOAD_ARCHIVE}" "${DATA_URL}"
        validate_archive "${DOWNLOAD_ARCHIVE}" || fail "ResNet dataset download failed gzip or SHA-256 validation."
        mv "${DOWNLOAD_ARCHIVE}" "${ARCHIVE}"
    fi

    tar -tzf "${ARCHIVE}" testFile/test_label.txt testFile/test_values.txt >/dev/null 2>&1 || fail "${ARCHIVE} is not the expected ResNet dataset archive."
    echo "Extracting ResNet dataset into ${DATA_DIR}"
    tar -xzf "${ARCHIVE}" --no-same-owner -C "${DATA_DIR}"
fi

AVAILABLE_SAMPLE_COUNT="$(wc -l < "${DATA_DIR}/testFile/test_label.txt")"
(( AVAILABLE_SAMPLE_COUNT > 0 )) || fail "The ResNet dataset contains no labels."
if [[ -z "${NUM_SAMPLES}" ]]; then
    SAMPLE_COUNT="${AVAILABLE_SAMPLE_COUNT}"
else
    [[ "${NUM_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || fail "--num-samples must be a positive integer."
    SAMPLE_COUNT="${NUM_SAMPLES}"
fi
(( SAMPLE_COUNT <= AVAILABLE_SAMPLE_COUNT )) || fail "Requested ${SAMPLE_COUNT} samples, but only ${AVAILABLE_SAMPLE_COUNT} input tensors are available."
echo "Running ResNet on ${SAMPLE_COUNT} sample(s) using ${GPUS} GPU(s)."

docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" \
    -v "${REPO_ROOT}:${REPO_ROOT}" \
    -v "${DATA_DIR}:${DATA_DIR}:ro" \
    -w "${SCRIPT_DIR}" \
    -e "PYTHONPATH=${PYTHONPATH_DIR}" \
    -e "LD_LIBRARY_PATH=${RUNTIME_LIB_DIR}" \
    "${IMAGE_NAME}" "${PYTHON}" resnet_runner.py \
    --gpus "${GPUS}" \
    --num-samples "${SAMPLE_COUNT}" \
    --data-dir "${DATA_DIR}"
