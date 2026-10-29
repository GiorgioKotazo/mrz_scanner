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
fn workers_must_be_a_positive_integer() {
    for value in ["0", "oops"] {
        let output = cli(&["missing.png", "--workers", value]);
        assert_no_candidate(&output, 1);
        assert!(output.stdout.is_empty());
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(!error.contains("cannot decode image"));
        if value == "0" {
            assert!(error.contains("--workers must be greater than zero"));
        }
    }
}

#[test]
fn empty_directory_fails_without_loading_models() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("notes.txt"), b"not an image").unwrap();
    std::fs::create_dir(dir.path().join("nested")).unwrap();
    std::fs::copy(fixture("white.png"), dir.path().join("nested/white.png")).unwrap();
    let output = cli(&[
        dir.path().to_str().unwrap(),
        "--models-dir",
        "missing-models",
    ]);
    assert_no_candidate(&output, 1);
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("no JPEG or PNG images"));
}

#[test]
fn existing_output_is_not_overwritten_in_either_mode() {
    let dir = tempfile::tempdir().unwrap();
    let image = dir.path().join("input.png");
    std::fs::copy(fixture("white.png"), &image).unwrap();
    for extension in ["jsonl", "xlsx", "XLSX"] {
        let destination = dir.path().join(format!("results.{extension}"));
        std::fs::write(&destination, b"previous results").unwrap();
        for input in [dir.path(), image.as_path()] {
            let output = cli(&[
                input.to_str().unwrap(),
                "--output",
                destination.to_str().unwrap(),
            ]);
            assert_no_candidate(&output, 1);
            assert!(output.stdout.is_empty());
            assert!(String::from_utf8_lossy(&output.stderr).contains("must not already exist"));
            assert_eq!(std::fs::read(&destination).unwrap(), b"previous results");
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
