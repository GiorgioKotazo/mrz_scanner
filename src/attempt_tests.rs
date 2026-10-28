//! Model-free integration of attempt orchestration, selection, output and exit status.
//! Kept inside the binary crate so the scanner callback need not become public API.
use super::*;
use crate::test_support::{FORMATS, failed, ocr, valid};
use mrz::Format;
use std::collections::VecDeque;

fn attempts(results: Vec<Result<ScanResult>>) -> (Result<Candidate>, Vec<(bool, u32)>) {
    let mut results = VecDeque::from(results);
    let mut calls = Vec::new();
    let candidate = run_attempts(&RgbImage::new(6, 4), false, |_, rotation, crop| {
        calls.push((crop, rotation));
        results
            .pop_front()
            .unwrap_or_else(|| Err(anyhow::anyhow!("no plausible MRZ region was detected")))
    });
    (candidate, calls)
}

fn candidate_exit_code(candidate: &Candidate) -> ExitCode {
    exit_code(write_result(&mut Vec::new(), candidate))
}

#[test]
fn clean_valid_stops_immediately_for_every_format() {
    for format in FORMATS {
        let (best, calls) = attempts(vec![Ok(valid(format, false))]);
        assert_eq!(calls, [(false, 0)], "{format:?}");
        assert_eq!(best.unwrap().result.parsed.unwrap().format, format);
    }
}

#[test]
fn recovered_valid_never_stops_and_retains_success_contract() {
    for format in FORMATS {
        let (best, calls) = attempts(vec![Ok(valid(format, true))]);
        assert_eq!(calls.len(), 8, "{format:?}");
        let best = best.unwrap();
        assert!(best.result.parsed.as_ref().unwrap().damaged_recovery);
        // Exercises the same output/status path as main, including Recovery.
        let mut output = Vec::new();
        assert_eq!(
            exit_code(write_result(&mut output, &best)),
            ExitCode::SUCCESS
        );
        assert!(String::from_utf8(output).unwrap().contains("Recovery:"));
    }
}

#[test]
fn later_clean_replaces_recovered_across_formats() {
    for format in FORMATS {
        let (best, calls) = attempts(vec![
            Ok(valid(Format::Td3, true)),
            Err(anyhow::anyhow!("inference failed")),
            Ok(valid(format, false)),
        ]);
        assert_eq!(calls, [(false, 0), (false, 90), (false, 180)]);
        let best = best.unwrap();
        assert_eq!(best.rotation, 180);
        let data = best.result.parsed.unwrap();
        assert_eq!(data.format, format);
        assert!(!data.damaged_recovery);
    }
}

#[test]
fn failed_candidates_compare_fraction_not_count() {
    // 2/3 beats 3/5 despite fewer successful checks, and 3/4 beats 3/5.
    let td3 = failed(Format::Td3, 3);
    let visa = failed(Format::MrvB, 2);
    let td1 = failed(Format::Td1, 3);
    assert_eq!(score(&visa).cmp(&score(&td3)), Ordering::Greater);
    assert_eq!(score(&td1).cmp(&score(&td3)), Ordering::Greater);
    assert_eq!(score(&td1).cmp(&score(&visa)), Ordering::Greater);
    let (best, calls) = attempts(vec![Ok(td3), Ok(visa), Ok(td1)]);
    assert_eq!(calls.len(), 8);
    assert_eq!(best.unwrap().rotation, 180);
}

#[test]
fn equal_fractions_keep_first_even_with_different_counts() {
    let td1 = failed(Format::Td1, 2); // 2/4
    let mut td3 = failed(Format::Td3, 2);
    let checks = &mut td3.parsed.as_mut().unwrap().checks;
    checks.date_of_birth = None;
    checks.personal_number = None;
    checks.composite = None; // 1/2, with the non-applicable checks ignored
    assert_eq!(checks.verified(), 1);
    assert_eq!(checks.applicable(), 2);
    assert_eq!(score(&td1).cmp(&score(&td3)), Ordering::Equal);
    let (best, _) = attempts(vec![Ok(td1), Ok(td3)]);
    assert_eq!(best.unwrap().rotation, 0);
}

#[test]
fn zero_applicable_checks_are_not_valid_or_a_division_by_zero() {
    let mut result = failed(Format::MrvA, 0);
    let checks = &mut result.parsed.as_mut().unwrap().checks;
    checks.document_number = None;
    checks.date_of_birth = None;
    checks.date_of_expiry = None;
    assert_eq!(score(&result).quality, Quality::ChecksumFailed);
    assert_eq!(
        score(&result).cmp(&score(&failed(Format::Td3, 0))),
        Ordering::Equal
    );
    let (best, calls) = attempts(vec![Ok(result)]);
    assert_eq!(calls.len(), 8);
    assert_eq!(candidate_exit_code(&best.unwrap()), ExitCode::from(2));
}

#[test]
fn unparsed_fallback_keeps_first_regardless_of_ocr_length() {
    let first = ocr("ABC");
    let later = ocr(&"GARBAGE".repeat(100));
    assert!(first.parsed.is_err() && later.parsed.is_err());
    assert_eq!(score(&first).cmp(&score(&later)), Ordering::Equal);
    let (best, calls) = attempts(vec![Ok(first), Ok(later)]);
    assert_eq!(calls.len(), 8);
    let best = best.unwrap();
    assert_eq!(best.result.ocr_lines, ["ABC"]);
    assert_eq!(candidate_exit_code(&best), ExitCode::from(2));
}

#[test]
fn recovered_ties_keep_first_and_invalid_results_do_not_replace_it() {
    let (best, calls) = attempts(vec![
        Ok(ocr("ABC")),
        Ok(failed(Format::Td3, 4)),
        Ok(valid(Format::MrvA, true)),
        Ok(valid(Format::Td3, true)),
        Ok(failed(Format::Td1, 3)),
    ]);
    assert_eq!(calls.len(), 8);
    assert_eq!(best.unwrap().rotation, 180);
}

#[test]
fn attempt_errors_preserve_candidate_and_crop_order() {
    let (best, calls) = attempts(vec![
        Err(anyhow::anyhow!("detection failed")),
        Ok(failed(Format::Td3, 1)),
        Err(anyhow::anyhow!("recognition failed")),
        Ok(ocr("ABC")),
        Ok(failed(Format::Td2, 3)),
        Err(anyhow::anyhow!("another failed scan")),
    ]);
    assert_eq!(
        calls,
        [
            (false, 0),
            (false, 90),
            (false, 180),
            (false, 270),
            (true, 0),
            (true, 90),
            (true, 180),
            (true, 270),
        ]
    );
    let best = best.unwrap();
    assert!(best.center_crop);
    assert_eq!(best.rotation, 0);
    assert_eq!(best.result.parsed.as_ref().unwrap().format, Format::Td2);
    assert_eq!(candidate_exit_code(&best), ExitCode::from(2));
}

#[test]
fn no_regions_produce_error_and_exit_code_one() {
    let (best, calls) = attempts(vec![]);
    assert_eq!(calls.len(), 8);
    let error = best.err().expect("must not create a candidate");
    let message = error.to_string();
    for (crop, rotation) in calls {
        assert!(message.contains(&format!("center_crop={crop}, rotation {rotation}:")));
    }
    assert_eq!(exit_code(Err(error)), ExitCode::FAILURE);
}

#[test]
fn no_rotate_and_square_skip_only_redundant_attempts() {
    for (width, height, no_rotate, expected) in [
        (6, 4, true, vec![(false, 0), (true, 0)]),
        (4, 4, true, vec![(false, 0)]),
        (
            4,
            4,
            false,
            vec![(false, 0), (false, 90), (false, 180), (false, 270)],
        ),
    ] {
        let mut calls = Vec::new();
        let best = run_attempts(
            &RgbImage::new(width, height),
            no_rotate,
            |_, rotation, crop| {
                calls.push((crop, rotation));
                Ok(valid(Format::Td3, true))
            },
        );
        assert!(best.is_ok());
        assert_eq!(calls, expected);
    }
}
