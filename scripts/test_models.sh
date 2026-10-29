#!/usr/bin/env bash
# Explicit, serial model suite: no private images and no implicit model downloads.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

for model in mrz_detection_20250222_fp32.onnx mrz_recognition_20250221_fp32.onnx; do
    if [[ ! -f "models/${model}" ]]; then
        echo "Missing models/${model}; run scripts/download_models.sh first" >&2
        exit 1
    fi
done

# Run tests serially to bound memory (~1 GiB per pipeline as a rough guide).
# The batch integration test itself deliberately loads two worker pipelines.
cargo test --locked --release --all-targets -- --ignored --test-threads=1
cargo build --locked --release
TARGET_DIR="$(cargo metadata --locked --no-deps --format-version 1 |
    python3 -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])')"
# Includes slow public cases and tracks the four documented OCR XFAILs.
# The Rust model tests above enforce strict contracts for the supported fixtures.
python3 tests/regression/run.py --binary "${TARGET_DIR}/release/mrz_scanner" --include-slow
