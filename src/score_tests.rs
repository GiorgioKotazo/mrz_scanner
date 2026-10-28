//! Fast, format-neutral candidate ranking tests.
use super::*;
use crate::test_support::{FORMATS, failed, ocr, valid};

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
