use super::*;
use calamine::{Data, Reader, Xlsx, open_workbook};
use serde_json::json;
use std::io::Read;

fn cell<'a>(row: &'a [Data], name: &str) -> &'a Data {
    &row[COLUMNS
        .iter()
        .position(|(heading, _, _)| *heading == name)
        .unwrap()]
}

fn example() -> Value {
    json!({
        "path": "photos/паспорт & sample.png", "status": "ok",
        "document": {
            "surname": "=1+1", "given_names": "+TEST", "sex": "F",
            "date_of_birth": "74<<12", "nationality": "UTO",
            "document_number": "00123456789012345678", "document_type": "P",
            "format": "Td3", "issuing_country": "UTO", "date_of_expiry": "2030-01-01",
            "personal_number": "000123", "optional_data_1": "@name",
            "optional_data_2": "-value"
        },
        "ocr_lines": ["=HYPERLINK(\"file:///tmp/a\")", "SECOND<LINE"],
        "normalized_mrz": "FIRST<LINE\nSECOND<LINE",
        "checksum_consistent": true, "damaged_recovery": true,
        "checks": {"document_number": true, "date_of_birth": false,
            "date_of_expiry": true, "personal_number": null, "composite": true},
        "parse_error": null, "rotation": 90, "center_crop": true,
        "polygon": [[0.5, 1.5], [2.5, 3.5], [4.5, 5.5], [6.5, 7.5]],
        "debug_directory": "debug/00000000"
    })
}

#[test]
fn workbook_roundtrips_fields_as_safe_text_with_filters_and_frozen_headers() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("results.xlsx");
    let mut writer = ExcelWriter::new(File::create(&path).unwrap(), dir.path()).unwrap();
    writer.write(&example()).unwrap();
    writer.finish().unwrap();
    let mut workbook: Xlsx<_> = open_workbook(&path).unwrap();
    assert_eq!(workbook.sheet_names(), ["Results"]);
    let range = workbook.worksheet_range("Results").unwrap();
    let rows: Vec<_> = range.rows().collect();
    assert_eq!(rows.len(), 2);
    for (index, (name, _, _)) in COLUMNS.iter().enumerate() {
        assert_eq!(rows[0][index], Data::String((*name).to_owned()));
    }
    for (name, expected) in [
        ("surname", "=1+1"),
        ("given_names", "+TEST"),
        ("document_number", "00123456789012345678"),
        ("personal_number", "000123"),
        ("date_of_birth", "74<<12"),
        ("date_of_expiry", "2030-01-01"),
        ("optional_data_1", "@name"),
        ("optional_data_2", "-value"),
        ("ocr_lines", "=HYPERLINK(\"file:///tmp/a\")\nSECOND<LINE"),
        ("normalized_mrz", "FIRST<LINE\nSECOND<LINE"),
        ("check_document_number", "OK"),
        ("check_date_of_birth", "FAILED"),
        ("check_personal_number", "N/A"),
        ("path", "photos/паспорт & sample.png"),
        ("debug_directory", "debug/00000000"),
    ] {
        assert_eq!(
            cell(rows[1], name),
            &Data::String(expected.to_owned()),
            "{name}"
        );
    }
    assert_eq!(
        cell(rows[1], "photo_path"),
        &Data::String(
            std::env::current_dir()
                .unwrap()
                .join("photos/паспорт & sample.png")
                .to_string_lossy()
                .into_owned()
        )
    );
    assert_eq!(cell(rows[1], "damaged_recovery"), &Data::Bool(true));
    assert_eq!(cell(rows[1], "checksum_consistent"), &Data::Bool(true));
    assert_eq!(cell(rows[1], "center_crop"), &Data::Bool(true));
    assert_eq!(cell(rows[1], "rotation"), &Data::Float(90.0));
    assert_eq!(
        cell(rows[1], "polygon"),
        &Data::String(example()["polygon"].to_string())
    );
    assert!(workbook.worksheet_formula("Results").unwrap().is_empty());

    let mut archive = zip::ZipArchive::new(File::open(path).unwrap()).unwrap();
    let mut xml = String::new();
    archive
        .by_name("xl/worksheets/sheet1.xml")
        .unwrap()
        .read_to_string(&mut xml)
        .unwrap();
    assert!(xml.contains("state=\"frozen\""));
    assert!(xml.contains("<autoFilter ref=\"A1:AE2\""));
    assert!(xml.contains("t=\"inlineStr\"")); // constant-memory worksheet
    // No formulas or external hyperlinks, even for formula-looking OCR.
    assert!(!xml.contains("<f>"));
    assert!(!xml.contains("<hyperlink"));
}

#[test]
fn error_and_unverified_rows_have_distinct_statuses_without_invented_fields() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("results.xlsx");
    let mut writer = ExcelWriter::new(File::create(&path).unwrap(), dir.path()).unwrap();
    writer
        .write(&json!({"path": "/missing.png", "status": "error", "error": "cannot decode"}))
        .unwrap();
    writer.write(&json!({"path": "parse.png", "status": "unverified", "ocr_lines": ["junk"],
        "document": null, "checks": null, "parse_error": "invalid MRZ", "checksum_consistent": false})).unwrap();
    writer.finish().unwrap();
    let mut workbook: Xlsx<_> = open_workbook(path).unwrap();
    let range = workbook.worksheet_range("Results").unwrap();
    let rows: Vec<_> = range.rows().collect();
    assert_eq!(rows.len(), 3);
    assert_eq!(
        cell(rows[1], "photo_path"),
        &Data::String("/missing.png".into())
    );
    assert_eq!(
        cell(rows[1], "error"),
        &Data::String("cannot decode".into())
    );
    assert_eq!(cell(rows[1], "status"), &Data::String("error".into()));
    assert_eq!(cell(rows[2], "status"), &Data::String("unverified".into()));
    assert_eq!(
        cell(rows[2], "parse_error"),
        &Data::String("invalid MRZ".into())
    );
    assert_eq!(cell(rows[2], "checksum_consistent"), &Data::Bool(false));
    for row in &rows[1..] {
        for field in ["surname", "document_number", "check_document_number"] {
            assert_eq!(cell(row, field), &Data::Empty);
        }
    }
}

#[test]
fn row_limit_splits_sheets_without_losing_records_or_headers() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("results.xlsx");
    let mut writer = ExcelWriter::new(File::create(&path).unwrap(), dir.path()).unwrap();
    // Exercise the same rollover as Excel's 1,048,576 row limit, cheaply.
    writer.rows_per_sheet = 3;
    for index in 0..5 {
        writer
            .write(&json!({"path": format!("{index}.png"), "status": "error"}))
            .unwrap();
    }
    writer.finish().unwrap();
    let mut workbook: Xlsx<_> = open_workbook(path).unwrap();
    assert_eq!(
        workbook.sheet_names(),
        ["Results", "Results 2", "Results 3"]
    );
    let mut paths = Vec::new();
    for name in workbook.sheet_names() {
        let range = workbook.worksheet_range(&name).unwrap();
        assert!(range.height() <= 3);
        assert_eq!(
            range.rows().next().unwrap()[0],
            Data::String("photo_path".into())
        );
        for row in range.rows().skip(1) {
            paths.push(cell(row, "path").clone());
        }
    }
    assert_eq!(
        paths,
        (0..5)
            .map(|i| Data::String(format!("{i}.png")))
            .collect::<Vec<_>>()
    );
}

#[test]
fn finalization_write_failure_is_reported() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("readonly.xlsx");
    std::fs::write(&path, []).unwrap();
    let mut writer = ExcelWriter::new(File::open(path).unwrap(), dir.path()).unwrap();
    writer.write(&example()).unwrap();
    assert!(writer.finish().is_err());
}

#[test]
fn overlong_cells_fail_explicitly_instead_of_silent_truncation() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("long.xlsx");
    let mut writer = ExcelWriter::new(File::create(path).unwrap(), dir.path()).unwrap();
    let mut record = example();
    record["error"] = json!("a".repeat(32_768));
    let error = writer.write(&record).unwrap_err();
    assert!(format!("{error:#}").contains("column error"));
}
