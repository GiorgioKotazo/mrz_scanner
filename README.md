# mrz_scanner

A local Rust CLI for extracting data from the machine-readable zone (MRZ) of passports and other ICAO 9303 documents.

## Features

- DOCSAID ONNX detection and recognition models via ONNX Runtime (`ort`).
- OpenCV-compatible preprocessing, MRZ detection, and perspective correction in Rust.
- Automatic rotations of 0/90/180/270 degrees and fallback with center cropping.
- Parsing of TD1/TD2/TD3/MRV-A/MRV-B and check digit validation using `mrz`.
- Output of the raw MRZ, structured fields, and the result of each check.
- Stops at the first clean checksum-valid result in any supported format; otherwise selects the best candidate.

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

### Attempt strategy and candidate selection

Attempts run on the full image at 0/90/180/270 degrees, then on its center square
crop in the same order. A square image does not need the crop pass; `--no-rotate`
limits each pass to 0 degrees. For TD1, TD2, TD3, MRV-A, and MRV-B alike, scanning
stops as soon as all applicable checks pass **without damaged recovery**
(`checksum_consistent && !damaged_recovery`).

A checksum-valid result obtained with checksum-guided OCR repair is kept, but
scanning continues to look for a clean result. Candidates are ranked in this order:

1. Clean checksum-valid result.
2. Recovered checksum-valid result.
3. Parseable result with failed checks, ranked by the fraction of successful checks
   among applicable checks (`N/A` checks are excluded).
4. Unparseable OCR from a plausible detected region.

No format has a ranking bonus. Ties retain the first candidate in attempt order;
unparseable OCR is not ranked by text length. An error in a later attempt does not
discard a candidate already found. If no attempt finds a plausible region, the
exit code is `1`.

If no clean checksum-valid result is found, a recovered checksum-valid result can
still be selected. It retains exit code `0` because all applicable check digits
match, and the output includes `Recovery: checksum-guided OCR repair was applied`.
This indicates repaired OCR, not confirmation of document authenticity.

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
