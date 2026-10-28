//! End-to-end ONNX tests. Run explicitly with scripts/test_models.sh.
//! Assert semantic fields/checks, never a golden copy of the entire stdout.
mod support;
use std::collections::BTreeMap;
use support::{assert_no_candidate, cli, fixture, root};

fn scan(name: &str, code: i32) -> String {
    let output = cli(&[
        fixture(name).to_str().unwrap(),
        "--models-dir",
        root().join("models").to_str().unwrap(),
        "--threads",
        "2",
    ]);
    assert_eq!(output.status.code(), Some(code), "{name}: {output:?}");
    if code == 1 {
        assert_no_candidate(&output, 1);
        assert!(String::from_utf8_lossy(&output.stderr).contains("no readable MRZ found"));
    }
    String::from_utf8(output.stdout).unwrap()
}

fn section<'a>(text: &'a str, heading: &str) -> BTreeMap<&'a str, &'a str> {
    text.lines()
        .skip_while(|line| *line != heading)
        .skip(1)
        .take_while(|line| line.starts_with("  "))
        .map(|line| {
            let (key, value) = line.trim().split_once(':').unwrap();
            (key, value.trim())
        })
        .collect()
}

fn assert_valid(text: &str, format: &str) {
    assert!(text.starts_with("MRZ (OCR):\n"));
    assert!(text.contains("(original image pixels)"));
    assert!(text.lines().any(|line| line == "Checksum status: OK"));
    assert!(!text.contains("Could not parse MRZ:"));
    assert_eq!(section(text, "Document data:")["Format"], format);
    let checks = section(text, "Check digits:");
    for label in ["Document number", "Date of birth", "Expiration date"] {
        assert_eq!(checks[label], "OK", "{format}/{label}");
    }
    assert_eq!(
        checks["Personal number"],
        if format == "Td3" { "OK" } else { "N/A" }
    );
    assert_eq!(
        checks["Composite"],
        if format.starts_with("Mrv") {
            "N/A"
        } else {
            "OK"
        }
    );
}

fn assert_synthetic_fields(text: &str, format: &str) {
    assert_valid(text, format);
    let fields = section(text, "Document data:");
    for (key, expected) in [
        ("Surname", "TESTER"),
        ("Given names", "ANNA MARIA"),
        ("Document number", "X12345678"),
        ("Date of birth", "1974-08-12"),
        ("Expiration date", "2012-04-15"),
        ("Nationality", "UTO"),
        ("Issuing country", "UTO"),
        ("Sex", "F"),
    ] {
        assert_eq!(fields[key], expected, "{format}/{key}");
    }
    if format == "Td3" {
        assert_eq!(fields["Personal number"], "ZE184226B");
    } else if format == "MrvA" {
        assert_eq!(fields["Optional data 1"], "VISATEST");
    }
}

#[test]
#[ignore = "requires DOCSAID ONNX models; run scripts/test_models.sh"]
fn official_docsaid_image_has_expected_fields_and_checksums() {
    let text = scan("docsaid.jpg", 0);
    assert_valid(&text, "Td3");
    let fields = section(&text, "Document data:");
    for (key, expected) in [
        ("Surname", "QAQARIN"),
        ("Given names", "FIDAN"),
        ("Document number", "C94630262"),
        ("Nationality", "AZE"),
        ("Date of birth", "1967-07-29"),
        ("Expiration date", "2023-03-10"),
    ] {
        assert_eq!(fields[key], expected, "{key}");
    }
}

#[test]
#[ignore = "requires DOCSAID ONNX models; run scripts/test_models.sh"]
fn all_supported_formats_have_expected_fields_and_checksums() {
    for (name, format) in [
        ("td1.png", "Td1"),
        ("td2.png", "Td2"),
        ("td3.png", "Td3"),
        ("mrv-a.png", "MrvA"),
        ("mrv-b.png", "MrvB"),
    ] {
        assert_synthetic_fields(&scan(name, 0), format);
    }
}

#[test]
#[ignore = "requires DOCSAID ONNX models; run scripts/test_models.sh"]
fn rotations_and_center_crop_preserve_document_data() {
    for (name, rotation, crop) in [
        ("td3.png", 0, false),
        ("td3-rotate-90.png", 90, false),
        ("td3-rotate-180.png", 180, false),
        ("td3-rotate-270.png", 270, false),
        ("center-crop.png", 0, true),
    ] {
        let text = scan(name, 0);
        assert_synthetic_fields(&text, "Td3");
        assert_eq!(text.contains("Center crop applied: yes"), crop, "{name}");
        if rotation == 0 {
            assert!(!text.contains("Image rotation applied:"), "{name}");
        } else {
            assert!(
                text.contains(&format!("Image rotation applied: {rotation}°")),
                "{name}"
            );
        }
    }
}

#[test]
#[ignore = "requires DOCSAID ONNX models; run scripts/test_models.sh"]
fn negative_images_have_no_candidate() {
    for name in ["white.png", "black.png", "solid.png", "photo.jpg"] {
        assert!(scan(name, 1).is_empty(), "{name}");
    }
}

#[test]
#[ignore = "requires DOCSAID ONNX models; run scripts/test_models.sh"]
fn unverified_regions_return_exit_two_with_diagnostics() {
    let text = scan("parse-failed.png", 2);
    assert!(text.starts_with("MRZ (OCR):\n"));
    assert!(text.contains("Could not parse MRZ:"));
    assert!(text.contains("Checksum status: NOT VERIFIED"));
    assert!(!text.contains("Document data:"));

    let text = scan("checksum-failed.png", 2);
    assert!(text.starts_with("MRZ (OCR):\n"));
    assert!(text.contains("Document data:"));
    assert!(text.contains("Checksum status: FAILED"));
    assert!(!text.contains("Could not parse MRZ:"));
    assert!(
        section(&text, "Check digits:")
            .values()
            .any(|value| *value == "FAILED")
    );
}
