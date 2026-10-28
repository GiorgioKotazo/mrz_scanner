//! Black-box CLI contract: all tests run without model files.
mod support;
use support::{assert_no_candidate, cli, fixture, root};

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
    for (flag, expected) in [
        ("--help", "Usage:"),
        (
            "--version",
            concat!("mrz_scanner ", env!("CARGO_PKG_VERSION")),
        ),
    ] {
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
fn missing_image_exits_one_before_loading_models() {
    // A child of a regular file cannot exist, regardless of working directory.
    let output = cli(&[root().join("Cargo.toml/missing.png").to_str().unwrap()]);
    assert_no_candidate(&output, 1);
    assert!(output.stdout.is_empty());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("cannot decode image"));
    assert!(!error.contains("model not found"));
}
