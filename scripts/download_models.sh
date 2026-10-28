#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${1:-${ROOT_DIR}/models}"
mkdir -p "${MODELS_DIR}"

download_model() {
    local file_id="$1"
    local filename="$2"
    local expected_sha="$3"
    local destination="${MODELS_DIR}/${filename}"

    if [[ -f "${destination}" ]] && echo "${expected_sha}  ${destination}" | sha256sum -c --status; then
        echo "OK: ${filename} already exists"
        return
    fi

    echo "Downloading ${filename}..."
    curl -fL --retry 3 \
        "https://drive.usercontent.google.com/download?id=${file_id}&export=download&confirm=t" \
        -o "${destination}.tmp"
    echo "${expected_sha}  ${destination}.tmp" | sha256sum -c --status
    mv "${destination}.tmp" "${destination}"
}

download_model \
    "1D--D7glsse51oIdcyerZoYs88iokOmqU" \
    "mrz_detection_20250222_fp32.onnx" \
    "0786844728d828dc68c19b844aa87e2b29fdba3a587468f2863824d116ffc269"

download_model \
    "16t-kYHoBnI72MWDMCQ0K8GEyb90Tx2Rp" \
    "mrz_recognition_20250221_fp32.onnx" \
    "07cef987fc1852aac12f18e1269fbaf1e2644579ca5743061d6f9cec7bef77c0"

echo "Models are ready in ${MODELS_DIR}"
