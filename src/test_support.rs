//! Prepared scan results shared by model-free tests; no ONNX sessions are created.
use crate::scanner::ScanResult;
use mrz::Format;

pub(super) const FORMATS: [Format; 5] = [
    Format::Td1,
    Format::Td2,
    Format::Td3,
    Format::MrvA,
    Format::MrvB,
];

pub(super) fn ocr(text: &str) -> ScanResult {
    ScanResult {
        ocr_lines: text.lines().map(str::to_owned).collect(),
        parsed: mrz::find_and_parse(text),
        polygon: [imageproc::point::Point::new(0.0, 0.0); 4],
    }
}

pub(super) fn valid(format: Format, recovered: bool) -> ScanResult {
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

pub(super) fn failed(format: Format, verified: u8) -> ScanResult {
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
