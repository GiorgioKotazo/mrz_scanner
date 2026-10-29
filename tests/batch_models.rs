//! Explicit CPU batch integration tests; run serially with the existing model suite.
mod support;
use calamine::{Data, Reader, Xlsx, open_workbook};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;
use support::{assert_no_candidate, cli, fixture, root};

fn copy(name: &str, dir: &Path, destination: &str) {
    std::fs::copy(fixture(name), dir.join(destination)).unwrap();
}

fn records(bytes: &[u8]) -> BTreeMap<String, Value> {
    let text = std::str::from_utf8(bytes).unwrap();
    let mut records = BTreeMap::new();
    for line in text.lines() {
        let json: Value = serde_json::from_str(line).unwrap();
        let path = Path::new(json["path"].as_str().unwrap());
        let name = path.file_name().unwrap().to_str().unwrap().to_owned();
        assert!(records.insert(name, json).is_none());
    }
    records
}

fn excel_records(path: &Path) -> BTreeMap<String, BTreeMap<String, Data>> {
    let mut workbook: Xlsx<_> = open_workbook(path).unwrap();
    let range = workbook.worksheet_range("Results").unwrap();
    let mut rows = range.rows();
    let headers = rows.next().unwrap();
    let mut records = BTreeMap::new();
    for row in rows {
        let fields: BTreeMap<_, _> = headers
            .iter()
            .zip(row)
            .map(|(key, value)| (key.to_string(), value.clone()))
            .collect();
        let path = fields["path"].to_string();
        let name = Path::new(&path)
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .to_owned();
        assert!(records.insert(name, fields).is_none());
    }
    records
}

#[test]
#[ignore = "requires DOCSAID ONNX models; run scripts/test_models.sh"]
fn single_image_excel_defaults_preserve_relative_and_absolute_paths() {
    let dir = tempfile::tempdir().unwrap();
    copy("td3.png", dir.path(), "input.png");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_mrz_scanner"))
        .current_dir(dir.path())
        .args(["input.png", "--output", "results.xlsx", "--models-dir"])
        .arg(root().join("models"))
        .output()
        .unwrap();
    assert_no_candidate(&output, 0);
    assert!(output.stdout.is_empty());
    let records = excel_records(&dir.path().join("results.xlsx"));
    assert_eq!(records.len(), 1);
    assert_eq!(
        records["input.png"]["path"],
        Data::String("input.png".into())
    );
    assert_eq!(
        records["input.png"]["photo_path"],
        Data::String(
            dir.path()
                .canonicalize()
                .unwrap()
                .join("input.png")
                .to_string_lossy()
                .into_owned()
        )
    );
    assert_eq!(records["input.png"]["status"], Data::String("ok".into()));
}

#[test]
#[ignore = "requires DOCSAID ONNX models; run scripts/test_models.sh"]
fn excel_directory_saves_valid_unverified_and_error_rows_before_exit_one() {
    let dir = tempfile::tempdir().unwrap();
    let destination = tempfile::tempdir().unwrap();
    copy("td3.png", dir.path(), "valid.png");
    copy("checksum-failed.png", dir.path(), "checksum.png");
    copy("parse-failed.png", dir.path(), "parse.png");
    copy("white.png", dir.path(), "white.png");
    std::fs::write(dir.path().join("broken.jpg"), b"broken").unwrap();
    let path = destination.path().join("results.xlsx");
    let output = cli(&[
        dir.path().to_str().unwrap(),
        "--workers",
        "2",
        "--threads",
        "1",
        "--output",
        path.to_str().unwrap(),
    ]);
    assert_no_candidate(&output, 1);
    assert!(output.stdout.is_empty());
    let records = excel_records(&path);
    assert_eq!(records.len(), 5);
    for (name, status) in [
        ("valid.png", "ok"),
        ("checksum.png", "unverified"),
        ("parse.png", "unverified"),
        ("white.png", "error"),
        ("broken.jpg", "error"),
    ] {
        assert_eq!(records[name]["status"], Data::String(status.into()));
        assert_eq!(
            records[name]["photo_path"],
            Data::String(dir.path().join(name).to_string_lossy().into_owned())
        );
    }
    assert_eq!(
        records["valid.png"]["surname"],
        Data::String("TESTER".into())
    );
    assert_eq!(
        records["valid.png"]["document_number"],
        Data::String("X12345678".into())
    );
    assert_eq!(
        records["valid.png"]["checksum_consistent"],
        Data::Bool(true)
    );
    assert!(!records["parse.png"]["parse_error"].to_string().is_empty());
    assert!(!records["broken.jpg"]["error"].to_string().is_empty());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("Summary: 1 ok, 2 unverified, 2 errors")
    );
}

#[test]
#[ignore = "requires DOCSAID ONNX models; run scripts/test_models.sh"]
fn directory_defaults_write_jsonl_to_stdout_and_skip_nested_files() {
    let dir = tempfile::tempdir().unwrap();
    copy("td3.png", dir.path(), "passport.PNG");
    std::fs::create_dir(dir.path().join("nested")).unwrap();
    copy("white.png", &dir.path().join("nested"), "ignored.png");
    std::fs::write(dir.path().join("notes.txt"), "ignored").unwrap();
    let output = cli(&[dir.path().to_str().unwrap()]);
    assert_no_candidate(&output, 0);
    let records = records(&output.stdout);
    assert_eq!(records.len(), 1);
    assert_eq!(records["passport.PNG"]["status"], "ok");
    assert_eq!(records["passport.PNG"]["document"]["surname"], "TESTER");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("Summary: 1 ok, 0 unverified, 0 errors")
    );
}

#[test]
#[ignore = "requires DOCSAID ONNX models; run scripts/test_models.sh"]
fn parallel_directory_preserves_single_worker_results_and_isolates_debug_files() {
    let dir = tempfile::tempdir().unwrap();
    copy("td3.png", dir.path(), "valid.png");
    copy("checksum-failed.png", dir.path(), "checksum.png");
    copy("parse-failed.png", dir.path(), "parse.png");
    copy("white.png", dir.path(), "white.png");
    std::fs::write(dir.path().join("broken.jpg"), b"not a jpeg").unwrap();
    let mut baseline = None;
    for workers in ["1", "2"] {
        let debug = tempfile::tempdir().unwrap();
        let output = cli(&[
            dir.path().to_str().unwrap(),
            "--workers",
            workers,
            "--threads",
            "1",
            "--debug-dir",
            debug.path().to_str().unwrap(),
        ]);
        assert_eq!(output.status.code(), Some(1), "{output:?}");
        let mut results = records(&output.stdout);
        assert_eq!(results.len(), 5, "workers={workers}: {output:?}");
        for (name, status) in [
            ("valid.png", "ok"),
            ("checksum.png", "unverified"),
            ("parse.png", "unverified"),
            ("white.png", "error"),
            ("broken.jpg", "error"),
        ] {
            assert_eq!(results[name]["status"], status, "{name}");
        }
        let mut debug_paths = std::collections::HashSet::new();
        for (name, result) in &mut results {
            let debug_path = result
                .as_object_mut()
                .unwrap()
                .remove("debug_directory")
                .unwrap();
            let debug_path = Path::new(debug_path.as_str().unwrap());
            assert!(debug_paths.insert(debug_path.to_owned()));
            if name != "broken.jpg" {
                assert!(debug_path.join("detection-input-full-0.png").exists());
            }
        }
        assert!(
            String::from_utf8_lossy(&output.stderr)
                .contains("Summary: 1 ok, 2 unverified, 2 errors")
        );
        if let Some(expected) = &baseline {
            assert_eq!(&results, expected);
        } else {
            baseline = Some(results);
        }
    }
}

#[test]
#[ignore = "requires DOCSAID ONNX models; run scripts/test_models.sh"]
fn output_files_work_for_single_images_and_directories() {
    let dir = tempfile::tempdir().unwrap();
    let destination = tempfile::tempdir().unwrap();
    copy("checksum-failed.png", dir.path(), "input.png");
    for (name, input) in [
        ("single", dir.path().join("input.png")),
        ("directory", dir.path().to_owned()),
    ] {
        for extension in ["jsonl", "XLSX"] {
            let output_path = destination.path().join(format!("{name}.{extension}"));
            let output = cli(&[
                input.to_str().unwrap(),
                "--output",
                output_path.to_str().unwrap(),
                // Only one input: do not allocate more than one worker's models.
                "--workers",
                "100",
                "--no-rotate",
            ]);
            assert_eq!(output.status.code(), Some(2), "{output:?}");
            assert!(output.stdout.is_empty());
            if extension == "jsonl" {
                let results = records(&std::fs::read(output_path).unwrap());
                assert_eq!(results.len(), 1);
                assert_eq!(results["input.png"]["status"], "unverified");
            } else {
                let results = excel_records(&output_path);
                assert_eq!(results.len(), 1);
                assert_eq!(
                    results["input.png"]["status"],
                    Data::String("unverified".into())
                );
            }
        }
    }
}
