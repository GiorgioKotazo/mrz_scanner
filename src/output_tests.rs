//! Fast formatting tests using prepared data, not model inference.
use super::*;
use crate::test_support::{FORMATS, failed, ocr, valid};
use mrz::Format;

fn formatted_data(data: &MrzData) -> String {
    let mut output = Vec::new();
    write_data(&mut output, data).unwrap();
    String::from_utf8(output).unwrap()
}

#[test]
fn document_output_is_format_neutral_and_preserves_optional_slots() {
    for format in FORMATS {
        let mut data = valid(format, false).parsed.unwrap();
        // Prepared parsed fields isolate formatting from parser/OCR behavior.
        data.optional_data_1 = Some("PRIMARY123".into());
        data.optional_data_2 = (format == Format::Td1).then(|| "SECOND456".into());
        let text = formatted_data(&data);
        let (fields, checks) = text.split_once("\nCheck digits:").unwrap();
        assert!(fields.contains("Document data:"), "{format:?}");
        assert!(fields.contains("Document number:  X12345678"));
        assert!(checks.contains("Document number:  OK"));
        assert!(!text.contains("Passport"));
        if format == Format::Td3 {
            assert!(fields.contains("Personal number:  PRIMARY123"));
            assert!(!fields.contains("Optional data 1:"));
        } else {
            assert!(fields.contains("Optional data 1:  PRIMARY123"));
            assert!(!fields.contains("Personal number:"));
            assert!(checks.contains("Personal number:  N/A"));
        }
        assert_eq!(
            fields.contains("Optional data 2:  SECOND456"),
            format == Format::Td1
        );
        assert!(!text.contains("Recovery:"));
    }
}

#[test]
fn empty_optional_fields_are_omitted_and_recovery_is_visible() {
    for format in FORMATS {
        for empty in [None, Some(String::new())] {
            let mut data = valid(format, true).parsed.unwrap();
            data.optional_data_1 = empty.clone();
            data.optional_data_2 = empty;
            let text = formatted_data(&data);
            let fields = text.split_once("\nCheck digits:").unwrap().0;
            assert!(!fields.contains("Optional data"));
            assert!(!fields.contains("Personal number:"));
            assert!(text.contains("Recovery:         checksum-guided OCR repair was applied"));
        }
    }
}

#[test]
fn result_status_and_sections_match_verification_outcome() {
    for (result, status, code) in [
        (valid(Format::Td3, false), "OK", ExitCode::SUCCESS),
        (valid(Format::Td1, true), "OK", ExitCode::SUCCESS),
        (failed(Format::Td2, 2), "FAILED", ExitCode::from(2)),
        (ocr("ABC"), "NOT VERIFIED", ExitCode::from(2)),
    ] {
        let parsed = result.parsed.is_ok();
        let recovered = result
            .parsed
            .as_ref()
            .is_ok_and(|data| data.damaged_recovery);
        let candidate = Candidate {
            rotation: 90,
            center_crop: true,
            result,
        };
        let mut output = Vec::new();
        assert_eq!(exit_code(write_result(&mut output, &candidate)), code);
        let text = String::from_utf8(output).unwrap();
        assert!(text.starts_with("MRZ (OCR):\n"));
        assert!(text.contains(&format!("Checksum status: {status}")));
        assert_eq!(text.contains("Document data:"), parsed);
        assert_eq!(text.contains("Could not parse MRZ:"), !parsed);
        assert_eq!(text.contains("Recovery:"), recovered);
        assert!(text.contains("Image rotation applied: 90°"));
        assert!(text.contains("Center crop applied: yes"));
        assert!(text.contains("(original image pixels)"));
    }
}

#[test]
fn normalized_mrz_is_shown_only_when_it_differs_from_ocr() {
    let mut candidate = Candidate {
        rotation: 0,
        center_crop: false,
        result: valid(Format::Td3, false),
    };
    for changed in [false, true] {
        if changed {
            candidate.result.ocr_lines[0] = candidate.result.ocr_lines[0].replace("UTO", "UT0");
        }
        let mut output = Vec::new();
        assert!(write_result(&mut output, &candidate).unwrap());
        let text = String::from_utf8(output).unwrap();
        assert_eq!(
            text.contains("MRZ after checksum-guided normalization/repair:"),
            changed
        );
        assert!(!text.contains("Image rotation applied:"));
        assert!(!text.contains("Center crop applied:"));
    }
}
