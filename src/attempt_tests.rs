use super::*;
use mrz::Format;
use std::collections::VecDeque;

const FORMATS: [Format; 5] = [
    Format::Td1,
    Format::Td2,
    Format::Td3,
    Format::MrvA,
    Format::MrvB,
];

fn ocr(text: &str) -> ScanResult {
    ScanResult {
        ocr_lines: text.lines().map(str::to_owned).collect(),
        parsed: mrz::find_and_parse(text),
        polygon: [imageproc::point::Point::new(0.0, 0.0); 4],
    }
}

fn valid(format: Format, recovered: bool) -> ScanResult {
    // Public synthetic specimens from tests/regression/cases.json. Only the
    // recovery flag is injected: these tests exercise selection, not OCR repair.
    let text = match format {
        Format::Td1 => {
            "I<UTOX123456785<<<<<<<<<<<<<<<\n7408122F1204159UTO<<<<<<<<<<<0\nTESTER<<ANNA<MARIA<<<<<<<<<<<<"
        }
        Format::Td2 => "I<UTOTESTER<<ANNA<MARIA<<<<<<<<<<<<<\nX123456785UTO7408122F1204159<<<<<<<0",
        Format::Td3 => {
            "P<UTOTESTER<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<<<\nX123456785UTO7408122F1204159ZE184226B<<<<<12"
        }
        Format::MrvA => {
            "V<UTOTESTER<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<<<\nX123456785UTO7408122F1204159VISATEST<<<<<<<<"
        }
        Format::MrvB => {
            "V<UTOTESTER<<ANNA<MARIA<<<<<<<<<<<<<\nX123456785UTO7408122F1204159<<<<<<<<"
        }
        _ => panic!("unsupported test format"),
    };
    let mut result = ocr(text);
    let data = result.parsed.as_mut().unwrap();
    assert_eq!(data.format, format);
    assert!(data.checksum_consistent());
    data.damaged_recovery = recovered;
    result
}

fn failed(format: Format, verified: u8) -> ScanResult {
    let mut result = valid(format, false);
    let checks = &mut result.parsed.as_mut().unwrap().checks;
    assert!(verified < checks.applicable());
    for (index, check) in [
        &mut checks.document_number,
        &mut checks.date_of_birth,
        &mut checks.date_of_expiry,
        &mut checks.personal_number,
        &mut checks.composite,
    ]
    .into_iter()
    .flatten()
    .enumerate()
    {
        *check = index < usize::from(verified);
    }
    result
}

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
        assert_eq!(exit_code(print_result(&best)), ExitCode::SUCCESS);
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
fn quality_tiers_are_strict_and_format_neutral() {
    for format in FORMATS {
        let ordered = [
            ocr("GARBAGE"),
            failed(format, 0),
            valid(format, true),
            valid(format, false),
        ];
        for pair in ordered.windows(2) {
            assert_eq!(score(&pair[0]).cmp(&score(&pair[1])), Ordering::Less);
        }
        for other_format in FORMATS {
            for recovered in [false, true] {
                assert_eq!(
                    score(&valid(format, recovered)).cmp(&score(&failed(other_format, 2))),
                    Ordering::Greater
                );
                assert_eq!(
                    score(&valid(format, recovered)).cmp(&score(&valid(other_format, recovered))),
                    Ordering::Equal
                );
            }
        }
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
    assert_eq!(exit_code(print_result(&best.unwrap())), ExitCode::from(2));
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
    assert_eq!(exit_code(print_result(&best)), ExitCode::from(2));
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
    assert_eq!(exit_code(print_result(&best)), ExitCode::from(2));
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
