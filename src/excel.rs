//! Row-by-row XLSX export of the same records used for JSONL. This module does
//! not load models or accumulate scan results. Only the output thread uses it.
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rust_xlsxwriter::{Format, Workbook};
use serde_json::Value;

const EXCEL_ROWS: u32 = 1_048_576;

// Stable column order: image identity, document fields, then verification and
// diagnostics. JSON pointers keep optional/missing fields blank, not fabricated.
const COLUMNS: &[(&str, &str, f64)] = &[
    ("photo_path", "/path", 60.0),
    ("path", "/path", 45.0),
    ("status", "/status", 14.0),
    ("surname", "/document/surname", 24.0),
    ("given_names", "/document/given_names", 28.0),
    ("sex", "/document/sex", 8.0),
    ("date_of_birth", "/document/date_of_birth", 16.0),
    ("nationality", "/document/nationality", 14.0),
    ("document_number", "/document/document_number", 22.0),
    ("document_type", "/document/document_type", 18.0),
    ("format", "/document/format", 12.0),
    ("issuing_country", "/document/issuing_country", 18.0),
    ("date_of_expiry", "/document/date_of_expiry", 16.0),
    ("personal_number", "/document/personal_number", 22.0),
    ("optional_data_1", "/document/optional_data_1", 24.0),
    ("optional_data_2", "/document/optional_data_2", 24.0),
    ("ocr_lines", "/ocr_lines", 48.0),
    ("normalized_mrz", "/normalized_mrz", 48.0),
    ("checksum_consistent", "/checksum_consistent", 22.0),
    ("damaged_recovery", "/damaged_recovery", 20.0),
    ("check_document_number", "/checks/document_number", 24.0),
    ("check_date_of_birth", "/checks/date_of_birth", 22.0),
    ("check_date_of_expiry", "/checks/date_of_expiry", 22.0),
    ("check_personal_number", "/checks/personal_number", 24.0),
    ("check_composite", "/checks/composite", 18.0),
    ("parse_error", "/parse_error", 60.0),
    ("error", "/error", 60.0),
    ("rotation", "/rotation", 12.0),
    ("center_crop", "/center_crop", 14.0),
    ("polygon", "/polygon", 48.0),
    ("debug_directory", "/debug_directory", 45.0),
];

pub(super) struct ExcelWriter {
    workbook: Workbook,
    file: BufWriter<File>,
    base_dir: PathBuf,
    sheets: usize,
    next_row: u32,
    rows_per_sheet: u32,
}

impl ExcelWriter {
    /// `file` is the already exclusively created destination. Saving through the
    /// same handle avoids reopening/truncating a path and preserves no-overwrite.
    pub fn new(file: File, temp_dir: &Path) -> Result<Self> {
        let mut workbook = Workbook::new();
        // Put temporary worksheet XML beside the output, not in a potentially
        // RAM-backed system /tmp. These temporary files also contain personal data.
        workbook.set_tempdir(temp_dir)?;
        let mut writer = Self {
            workbook,
            file: BufWriter::new(file),
            base_dir: std::env::current_dir()?,
            sheets: 0,
            next_row: 1,
            rows_per_sheet: EXCEL_ROWS,
        };
        writer.start_sheet()?;
        Ok(writer)
    }

    fn start_sheet(&mut self) -> Result<()> {
        let sheet = self.workbook.add_worksheet_with_constant_memory()?;
        self.sheets += 1;
        let name = if self.sheets == 1 {
            "Results".to_owned()
        } else {
            format!("Results {}", self.sheets)
        };
        sheet.set_name(name)?;
        sheet.set_freeze_panes(1, 0)?;
        let header = Format::new().set_bold();
        for (col, &(name, _, width)) in COLUMNS.iter().enumerate() {
            sheet.set_column_width(col as u16, width)?;
            sheet.write_string_with_format(0, col as u16, name, &header)?;
        }
        self.next_row = 1;
        Ok(())
    }

    fn finish_sheet(&mut self) -> Result<()> {
        self.workbook
            .worksheet_from_index(self.sheets - 1)?
            .autofilter(0, 0, self.next_row - 1, (COLUMNS.len() - 1) as u16)?;
        Ok(())
    }

    pub fn write(&mut self, record: &Value) -> Result<()> {
        if self.next_row == self.rows_per_sheet {
            self.finish_sheet()?;
            self.start_sheet()?;
        }
        let sheet = self.workbook.worksheet_from_index(self.sheets - 1)?;
        for (col, &(name, pointer, _)) in COLUMNS.iter().enumerate() {
            let value = record.pointer(pointer).unwrap_or(&Value::Null);
            let col = col as u16;
            let result = if name == "photo_path" {
                let Some(path) = value.as_str() else { continue };
                // No canonicalize: an error row may refer to a missing file.
                // Keep the original supplied path separately in `path`.
                sheet.write_string(
                    self.next_row,
                    col,
                    self.base_dir.join(path).to_string_lossy(),
                )
            } else if pointer.starts_with("/checks/") && record["checks"].is_object() {
                let text = match value.as_bool() {
                    Some(true) => "OK",
                    Some(false) => "FAILED",
                    None => "N/A",
                };
                sheet.write_string(self.next_row, col, text)
            } else {
                match value {
                    Value::Null => continue,
                    // Explicit string cells preserve leading zeros, partial MRZ
                    // dates and formula-like OCR/path text without executing it.
                    Value::String(text) => sheet.write_string(self.next_row, col, text),
                    Value::Bool(value) => sheet.write_boolean(self.next_row, col, *value),
                    Value::Number(value) => sheet.write_number(
                        self.next_row,
                        col,
                        value.as_f64().context("invalid numeric result")?,
                    ),
                    Value::Array(lines) if name == "ocr_lines" => {
                        let text = lines
                            .iter()
                            .filter_map(Value::as_str)
                            .collect::<Vec<_>>()
                            .join("\n");
                        sheet.write_string(self.next_row, col, text)
                    }
                    // Polygon coordinates remain lossless JSON in a text cell.
                    other => sheet.write_string(self.next_row, col, other.to_string()),
                }
            };
            result.with_context(|| {
                format!("cannot write XLSX column {name}, row {}", self.next_row + 1)
            })?;
        }
        self.next_row += 1;
        Ok(())
    }

    /// Unlike JSONL, XLSX isn't readable until the ZIP container is finalized.
    /// Call this even when individual images failed or checksums did not match.
    pub fn finish(mut self) -> Result<()> {
        self.finish_sheet()?;
        self.workbook
            .save_to_writer(&mut self.file)
            .context("cannot finalize XLSX workbook")?;
        self.file.flush().context("cannot flush XLSX output")?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "excel_tests.rs"]
mod tests;
