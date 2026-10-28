//! Black-box CLI contract. Default tests require no model files.
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn fixture(name: &str) -> PathBuf {
    root().join("tests/regression/fixtures").join(name)
}

fn cli(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_mrz_scanner"))
        .args(args)
        .output()
        .expect("run CLI")
}

fn assert_no_candidate(output: &Output, code: i32) {
    assert_eq!(output.status.code(), Some(code), "{output:?}");
    assert!(!String::from_utf8_lossy(&output.stdout).contains("MRZ (OCR):"));
}

#[test]
fn usage_errors_exit_one_without_a_candidate() {
    // A nonexistent image ensures validation precedes image/model processing.
    for args in [
        vec![],
        vec!["missing.png", "--unknown"],
        vec!["missing.png", "--threads", "oops"],
        vec!["missing.png", "--threads", "0"],
    ] {
        let output = cli(&args);
        assert_no_candidate(&output, 1);
        assert!(output.stdout.is_empty(), "{args:?}");
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(!error.is_empty());
        assert!(!error.contains("cannot decode image"), "{error}");
        if args.last() == Some(&"0") {
            assert!(error.contains("--threads must be greater than zero"));
        }
    }
}

#[test]
fn help_and_version_succeed_without_an_image_or_models() {
    for (flag, expected) in [("--help", "Usage:"), ("--version", "mrz_scanner 0.1.0")] {
        let output = cli(&[flag]);
        assert_no_candidate(&output, 0);
        assert!(output.stderr.is_empty());
        assert!(String::from_utf8_lossy(&output.stdout).contains(expected));
    }
}

#[test]
fn missing_models_exit_one_without_a_candidate() {
    // Using a regular file as the directory guarantees model paths don't exist.
    let output = cli(&[
        fixture("white.png").to_str().unwrap(),
        "--models-dir",
        root().join("Cargo.toml").to_str().unwrap(),
    ]);
    assert_no_candidate(&output, 1);
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("model not found:"));
}

#[test]
fn undecodable_image_exits_one_without_a_candidate() {
    let output = cli(&[root().join("Cargo.toml").to_str().unwrap()]);
    assert_no_candidate(&output, 1);
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("cannot decode image"));
}

#[test]
#[ignore = "requires DOCSAID ONNX models; run with --release --ignored"]
fn scan_outcomes_have_distinct_exit_codes() {
    for (name, code, status) in [
        ("white.png", 1, None),
        ("parse-failed.png", 2, Some("NOT VERIFIED")),
        ("checksum-failed.png", 2, Some("FAILED")),
        ("td3.png", 0, Some("OK")),
        ("mrv-a.png", 0, Some("OK")),
    ] {
        let output = cli(&[
            fixture(name).to_str().unwrap(),
            "--models-dir",
            root().join("models").to_str().unwrap(),
        ]);
        assert_eq!(output.status.code(), Some(code), "{name}: {output:?}");
        let text = String::from_utf8_lossy(&output.stdout);
        if let Some(status) = status {
            assert!(text.contains("MRZ (OCR):"), "{name}");
            assert!(
                text.contains(&format!("Checksum status: {status}")),
                "{name}"
            );
            assert!(text.contains("(original image pixels)"));
            if name == "parse-failed.png" {
                assert!(text.contains("Could not parse MRZ:"));
            } else {
                assert!(text.contains("Document data:"));
                assert!(!text.contains("Passport"));
            }
            if name == "mrv-a.png" {
                assert!(text.contains("Optional data 1:  VISATEST"));
            }
        } else {
            assert_no_candidate(&output, 1);
            assert!(String::from_utf8_lossy(&output.stderr).contains("no readable MRZ found"));
        }
    }
}
