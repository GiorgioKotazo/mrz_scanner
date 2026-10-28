# mrz_scanner

A local Rust CLI for extracting data from the machine-readable zone (MRZ) of passports and other ICAO 9303 documents.

## Features

- DOCSAID ONNX detection and recognition models via ONNX Runtime (`ort`).
- OpenCV-compatible preprocessing, MRZ detection, and perspective correction in Rust.
- Automatic rotations of 0/90/180/270 degrees and fallback with center cropping.
- Parsing of TD1/TD2/TD3/MRV-A/MRV-B and check digit validation using `mrz`.
- Output of the raw MRZ, structured fields, and the result of each check.
- Stops at the first checksum-valid TD3; on failure, the best candidate is selected.

## Building

Rust 1.92 or newer, `curl`, and `sha256sum` are required.

```bash
cd mrz_scanner
./scripts/download_models.sh
cargo build --release
```

ONNX Runtime is linked into the binary. To run it, you need the executable and two external ONNX models; Python is not required.

## Running

```bash
./target/release/mrz_scanner /path/to/passport.jpg
```

By default, models are loaded from `./models`. To specify a different directory and the number of CPU threads:

```bash
./target/release/mrz_scanner /path/to/passport.jpg \
  --models-dir /opt/mrz_scanner/models \
  --threads 2
```

JPEG and PNG are supported. The `--no-rotate` flag disables automatic rotations; center cropping may still be used as a fallback. For maximum accuracy, it is best to run the program without this flag.

Exit codes:

- `0` — the MRZ was parsed and all applicable check digits matched;
- `2` — a candidate was obtained, but parsing or at least one check digit failed;
- `1` — the model, image, or MRZ could not be processed.

With exit code `2`, the data must not be considered verified: another photo or manual verification is required.

## Debugging

```bash
./target/release/mrz_scanner /path/to/passport.jpg --debug-dir ./debug
```

The detection model input, heatmap, binarization, and aligned MRZ are saved. **These files contain personal data** and must not end up in logs or version control.

## Tests

```bash
cargo test
cargo clippy --all-targets -- -D warnings
```

## Models

The official DOCSAID MRZScanner 1.0.7 two-stage weights are used:

- `mrz_detection_20250222_fp32.onnx`;
- `mrz_recognition_20250221_fp32.onnx`.

The download script verifies the SHA-256 against the official DOCSAID catalog. The code and official weights are licensed under Apache-2.0. Sources:

- https://github.com/DocsaidLab/MRZScanner
- https://docsaid.org/en/models/mrzscanner
- https://github.com/pykeio/ort
- https://github.com/ruledicaprio/SynthPass/tree/main/crates/mrz

Check digits confirm checksum consistency, not document authenticity. In TD3, the given name and surname are not covered by a check digit, so for critical operations they must also be verified against the visual inspection zone.
