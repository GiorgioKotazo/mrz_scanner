use super::*;
use crate::test_support::{FORMATS, failed, ocr, valid};

fn candidate(result: crate::scanner::ScanResult) -> Candidate {
    Candidate {
        rotation: 90,
        center_crop: true,
        result,
    }
}

#[test]
fn json_records_preserve_fields_checks_recovery_and_geometry() {
    for format in FORMATS {
        for recovered in [false, true] {
            let record = Record::from_result(
                Path::new("space \" quote\n.png"),
                Ok(candidate(valid(format, recovered))),
            );
            assert_eq!(record.status, Status::Ok);
            let mut bytes = Vec::new();
            write_record(&mut bytes, &record).unwrap();
            let text = String::from_utf8(bytes).unwrap();
            assert_eq!(text.lines().count(), 1);
            let json: Value = serde_json::from_str(text.trim()).unwrap();
            assert_eq!(json["path"], "space \" quote\n.png");
            assert_eq!(json["status"], "ok");
            assert_eq!(json["document"]["format"], format!("{format:?}"));
            assert_eq!(json["document"]["surname"], "TESTER");
            assert_eq!(json["document"]["document_number"], "X12345678");
            assert_eq!(json["checks"]["document_number"], true);
            assert_eq!(
                json["checks"]["composite"].is_null(),
                matches!(format, mrz::Format::MrvA | mrz::Format::MrvB)
            );
            assert_eq!(json["damaged_recovery"], recovered);
            assert_eq!(json["checksum_consistent"], true);
            assert_eq!(json["rotation"], 90);
            assert_eq!(json["center_crop"], true);
            assert_eq!(json["polygon"].as_array().unwrap().len(), 4);
            assert!(json["parse_error"].is_null());
            if format == mrz::Format::Td3 {
                assert_eq!(json["document"]["personal_number"], "ZE184226B");
                assert!(json["document"].get("optional_data_1").is_none());
            }
        }
    }
}

#[test]
fn json_records_distinguish_failed_checks_parse_failure_and_processing_error() {
    let checksum = Record::from_result(
        Path::new("a.png"),
        Ok(candidate(failed(mrz::Format::Td3, 1))),
    );
    assert_eq!(checksum.status, Status::Unverified);
    assert_eq!(checksum.json["status"], "unverified");
    assert_eq!(checksum.json["checksum_consistent"], false);
    assert!(!checksum.json["document"].is_null());
    let parse = Record::from_result(Path::new("b.png"), Ok(candidate(ocr("not MRZ"))));
    assert_eq!(parse.status, Status::Unverified);
    assert!(parse.json["document"].is_null());
    assert!(parse.json["parse_error"].is_string());
    assert_eq!(parse.json["ocr_lines"], json!(["not MRZ"]));
    let error = Record::from_result(Path::new("c.png"), Err(anyhow::anyhow!("broken input")));
    assert_eq!(error.status, Status::Error);
    assert_eq!(error.json["status"], "error");
    assert_eq!(error.json["error"], "broken input");
}

#[test]
fn discovery_is_case_insensitive_nonrecursive_and_sorted() {
    let dir = tempfile::tempdir().unwrap();
    for name in ["c.JPEG", "a.png", "b.JPG", "notes.txt", "extensionless"] {
        fs::write(dir.path().join(name), []).unwrap();
    }
    fs::create_dir(dir.path().join("nested.png")).unwrap();
    fs::write(dir.path().join("nested.png/hidden.jpg"), []).unwrap();
    let names: Vec<_> = images_in(dir.path())
        .unwrap()
        .into_iter()
        .map(|path| path.file_name().unwrap().to_owned())
        .collect();
    assert_eq!(names, ["a.png", "b.JPG", "c.JPEG"]);
    assert!(images_in(&dir.path().join("nested.png/hidden.jpg")).is_err());
    let empty = tempfile::tempdir().unwrap();
    assert!(
        images_in(empty.path())
            .unwrap_err()
            .to_string()
            .contains("no JPEG or PNG")
    );
}

#[cfg(unix)]
#[test]
fn discovery_skips_symlinks() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("real.png"), []).unwrap();
    std::os::unix::fs::symlink(dir.path().join("real.png"), dir.path().join("link.png")).unwrap();
    std::os::unix::fs::symlink(dir.path(), dir.path().join("loop")).unwrap();
    assert_eq!(
        images_in(dir.path()).unwrap(),
        [dir.path().join("real.png")]
    );
}

#[test]
fn output_creation_never_truncates_existing_files() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("results.jsonl");
    let mut file = output_file(&path).unwrap();
    file.write_all(b"keep me").unwrap();
    drop(file);
    assert!(output_file(&path).is_err());
    assert_eq!(fs::read(path).unwrap(), b"keep me");
}

#[test]
fn pool_reuses_states_and_processes_every_path_once_despite_image_errors() {
    for workers in [1, 2, 4] {
        let paths: Vec<_> = (0..40).map(|i| PathBuf::from(format!("{i}.png"))).collect();
        let states = vec![0usize; workers];
        let mut seen = std::collections::HashSet::new();
        let mut reused = false;
        let summary = run_pool(
            &paths,
            states,
            |count, index, path| {
                *count += 1;
                let status = match index % 3 {
                    0 => Status::Ok,
                    1 => Status::Unverified,
                    _ => Status::Error,
                };
                Record {
                    status,
                    json: json!({"path": path.to_string_lossy(), "count": count}),
                }
            },
            |record| {
                assert!(seen.insert(record.json["path"].as_str().unwrap().to_owned()));
                reused |= record.json["count"].as_u64().unwrap() > 1;
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(seen.len(), paths.len());
        assert!(reused);
        assert_eq!(
            summary,
            Summary {
                ok: 14,
                unverified: 13,
                errors: 13
            }
        );
        assert!(summary.outcome().is_err());
    }
    assert!(
        Summary {
            ok: 1,
            ..Summary::default()
        }
        .outcome()
        .unwrap()
    );
    assert!(
        !Summary {
            unverified: 1,
            ..Summary::default()
        }
        .outcome()
        .unwrap()
    );
}

#[test]
fn pool_stops_on_output_failure_without_deadlocking() {
    let paths = vec![PathBuf::from("input.png"); 1000];
    let processed = AtomicUsize::new(0);
    let error = run_pool(
        &paths,
        vec![(); 2],
        |_, _, path| {
            processed.fetch_add(1, Ordering::Relaxed);
            Record::from_result(path, Err(anyhow::anyhow!("unreadable")))
        },
        |_| bail!("output closed"),
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("output closed"));
    assert!(processed.load(Ordering::Relaxed) < paths.len());
}

#[test]
fn pool_reports_worker_panics_as_fatal_errors() {
    let error = run_pool(
        &[PathBuf::from("input.png")],
        vec![()],
        |_, _, _| panic!("simulated worker failure"),
        |_| Ok(()),
    )
    .unwrap_err();
    assert!(error.to_string().contains("worker panicked"));
}
