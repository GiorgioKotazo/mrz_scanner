# mrz_scanner

A local Rust CLI for extracting data from the machine-readable zone (MRZ) of passports and other ICAO 9303 documents.

## Features

- DOCSAID ONNX detection and recognition models via ONNX Runtime (`ort`).
- OpenCV-compatible preprocessing, MRZ detection, and perspective correction in Rust.
- Bounded detection buffers: direct 256×256 letterbox input and heatmap/contour
  postprocessing capped at 1536×1536, without a full-resolution padded square.
  Decoded images, crops, rotations, and rectified MRZ buffers still use memory
  proportional to their dimensions.
- Automatic rotations of 0/90/180/270 degrees and fallback with center cropping.
- Parsing of TD1/TD2/TD3/MRV-A/MRV-B and check digit validation using `mrz`.
- Output of the raw MRZ, structured fields, and the result of each check.
- Stops at the first clean checksum-valid result in any supported format; otherwise selects the best candidate.

## Building

Rust 1.92 or newer is required (also the minimum required by `ort`).
The download script needs Bash, `curl`, and `sha256sum`. Linux CPU builds and
regressions are exercised by CI on Ubuntu 24.04; other platforms and GPU execution
are not covered by this project's tests. On Ubuntu, install the build dependencies
with `sudo apt-get install build-essential pkg-config libssl-dev curl`.

```bash
cd mrz_scanner
./scripts/download_models.sh
cargo build --release
```

With the default Linux build, ONNX Runtime is linked into the binary. To run it,
you need the executable, its system runtime libraries (including the C/C++ runtime),
and two external ONNX models; Python is not required.

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

JPEG and PNG are supported. The `--no-rotate` flag disables automatic rotations;
center cropping may still be used as a fallback. Leave rotations enabled to try
all four orientations; this does not guarantee successful recognition.
`--threads` must be an integer greater than zero (default: `2`).
Use `--help` for usage or `--version` for the application version; neither requires
an image or model files.

Exit codes:

- `0` — the MRZ was parsed and all applicable check digits matched, or `--help`/`--version` was requested;
- `2` — a plausible MRZ candidate was obtained, but parsing or checksum verification failed;
- `1` — invalid command-line arguments, a model/image processing error, or no acceptable MRZ region found.

Usage errors are reported on stderr without an MRZ candidate; help and version
are written to stdout. Code `2` is reserved for unverified scan results, not CLI errors.

With exit code `2`, the data must not be considered verified: another photo or manual verification is required.

### Document fields

The output contains raw OCR lines, the MRZ quadrangle, and, when parsing succeeds,
`Document data` and `Check digits` sections. Document fields include format, type,
surname, given names, sex, birth date, nationality, full `Document number`, issuing
country, and expiration date.

Nonempty optional fields are also printed:

- `Optional data 1` (`optional_data_1`) for TD1, TD2, MRV-A, and MRV-B;
- `Optional data 2` (`optional_data_2`) for TD1;
- `Personal number` for TD3's `optional_data_1`, without duplicating the same value
  as `Optional data 1`.

Absent or empty optional fields are omitted. Non-applicable check digits remain
`N/A`. The document-number check is labelled `Document number` for every format.
If parsing fails, the raw OCR and `Checksum status: NOT VERIFIED` are still shown.

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

### Polygon coordinates

`MRZ quadrangle: [...] (original image pixels)` always refers to the original
image decoded from the input file, including when rotation or center cropping
was used. Coordinates are `(x, y)` in pixels: the top-left pixel center is `(0, 0)`,
with x increasing to the right and y downwards. The quadrangle can be drawn
directly on the original image without applying the reported rotation/crop.
Corners retain their cyclic order, but the first corner need not be the original
image's top-left corner. Fractional coordinates are preserved; detected corners
are not clipped to the image bounds.

Attempts crop before rotating clockwise. Detection letterboxing is undone inside
the scanner; then the inverse rotation and crop offset map the result back to the
original file. Full/0° borrows the original RGB buffer, crop/0° borrows one crop
buffer, and nonzero rotations allocate only their rotated buffer.

### Tested scope and known limitations

The public suite checks the official DOCSAID image, synthetic examples of all five
formats, four TD3 orientations, a center-crop case, a 6000×4000 input, and four
no-MRZ images. Additional model tests cover a 608×448 document and a mild
perspective transform. These cases are regression coverage, not a general accuracy
guarantee for arbitrary documents, perspectives, or backgrounds.

Four public fixtures still have known OCR failures: a 232×160 document, a strongly
stretched document, a 6000×800 panorama, and an odd-sized image whose given name is
misread even though its checksums pass. Odd-size geometry and bounded detection
memory are tested, but neither guarantees OCR accuracy. These cases remain explicit
`XFAIL`s, not successful recognition tests; see the
[regression coverage and limitations](tests/regression/README.md#покрытие-и-текущий-результат).

## Debugging

```bash
./target/release/mrz_scanner /path/to/passport.jpg --debug-dir ./debug
```

The detection model input, heatmap, binarization, and aligned MRZ are saved. **These files contain personal data** and must not end up in logs or version control.

## Tests

Tests are split by purpose; the default suite does not need DOCSAID model files:

| Suite | Location | Coverage |
|---|---|---|
| Fast unit tests | `src/image_processing.rs`, `src/scanner.rs`, `src/geometry_tests.rs`, `src/score_tests.rs`, `src/output_tests.rs` | BGR/NCHW, normalization, decoder, Otsu, bilinear resize/letterbox, odd dimensions, geometry, rotations/crop, ranking, formatting |
| Model-free orchestration | `src/attempt_tests.rs` | Injected scan results, attempt order/early stop, candidate selection, recovery, output and exit status |
| Model-free CLI integration | `tests/cli.rs` | Arguments, help/version, missing models, missing/undecodable images, exit codes |
| ONNX model integration (explicit) | `src/scanner_model_tests.rs`, `tests/models.rs` | DOCSAID specimen, all five formats, rotations/crop, no-MRZ, unverified candidates, small/perspective documents, fields and checksums |
| Public image regressions (explicit) | `tests/regression/` | OCR/fields/checksum/polygon baseline, large/panoramic inputs, documented OCR limitations |

Orchestration tests stay inside the binary crate to exercise its private scanner
callback without exposing a public API just for tests. Prepared results live in
`src/test_support.rs`; subprocess helpers live in `tests/support/mod.rs`.
Formatting writes to an injected buffer in tests, so output can be checked without
capturing global stdout. Assertions inspect semantic sections, not a snapshot of
all console output. There is no standalone test of the third-party MRZ parser.

### Fast checks

```bash
cargo fmt --all -- --check
cargo test --locked --all-targets
cargo clippy --locked --all-targets -- -D warnings
python3 -m unittest discover -s tests/regression -p 'test_*.py' -v

# Optional focused runs, still without model files:
cargo test --locked --bin mrz_scanner attempt_tests::
cargo test --locked --test cli
```

`--locked` keeps dependency resolution consistent with `Cargo.lock`. Compilation
may download dependencies and ONNX Runtime, but fast tests do not load DOCSAID
weights or perform inference. Python harness self-tests use only the standard library.

### Explicit model suite

On Linux with Python 3.10+ and GNU `/usr/bin/time`:

```bash
./scripts/download_models.sh
./scripts/test_models.sh

# Rust model tests only (requires both model files):
cargo test --locked --release --all-targets -- --ignored --test-threads=1
```

The script fails if models are missing; it never silently skips them or downloads
them implicitly. It runs all ignored Rust model tests serially, builds the release
CLI, then runs the full public Python regression suite, including slow cases.
Serial inference avoids loading multiple ~1 GiB pipelines concurrently. Only
public/synthetic fixtures are used; private files are never part of this script or CI.
The regression runner reports four documented OCR cases as `XFAIL` and rejects new
contract violations or baseline changes; Rust model tests have no expected failures.
See [the regression suite](tests/regression/README.md) for details.

### CI

[GitHub Actions](.github/workflows/ci.yml) runs on pushes, pull requests, and manual
requests, with two visible jobs on Ubuntu:

1. **Formatting, model-free tests and Clippy** — the fast checks above, without model files.
2. **ONNX model and public image regressions** — downloads/caches both models,
   verifies their SHA-256 hashes even on cache hits, then runs `scripts/test_models.sh`.

The model job runs after the fast job succeeds; it is not an optional or
`continue-on-error` step. Model download, build, or test failures fail the job.

## Models

The official DOCSAID MRZScanner 1.0.7 two-stage weights are used:

- `mrz_detection_20250222_fp32.onnx`;
- `mrz_recognition_20250221_fp32.onnx`.

The download script verifies the SHA-256 against pinned hashes from the official
DOCSAID catalog. Sources:

- https://github.com/DocsaidLab/MRZScanner
- https://docsaid.org/en/models/mrzscanner
- https://github.com/pykeio/ort
- https://github.com/ruledicaprio/SynthPass/tree/main/crates/mrz

Check digits confirm checksum consistency, not document authenticity. In TD3, the given name and surname are not covered by a check digit, so for critical operations they must also be verified against the visual inspection zone.

## License

This project's code is licensed under **Apache-2.0**; see the full text in
[LICENSE](LICENSE). The package metadata in `Cargo.toml` uses the same license.

DOCSAID MRZScanner and its official model weights are provided under the upstream
[Apache-2.0 license](https://github.com/DocsaidLab/MRZScanner/blob/7f98880115441c8e42424eda1dc8f1abd4e39377/LICENSE).
Dependencies and third-party fixtures retain their respective licenses; the
project's license does not replace them. Fixture provenance and license copies are
documented in the [regression README](tests/regression/README.md#происхождение-публичных-фотографий),
including [DOCSAID](tests/regression/fixtures/LICENSE-DOCSAID) and
[Pillow](tests/regression/fixtures/LICENSE-Pillow).
