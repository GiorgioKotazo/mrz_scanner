//! CPU directory processing. Each worker owns a reusable scanner; only paths and
//! small result records are queued, never a directory's worth of decoded images.
use std::fs::{self, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

use crate::{Args, Candidate, MrzScanner, run_attempts};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status {
    Ok,
    Unverified,
    Error,
}

struct Record {
    status: Status,
    json: Value,
}

impl Record {
    fn from_result(path: &Path, result: Result<Candidate>) -> Self {
        let candidate = match result {
            Ok(candidate) => candidate,
            Err(error) => {
                return Self {
                    status: Status::Error,
                    json: json!({
                        "path": path.to_string_lossy(),
                        "status": "error",
                        "error": format!("{error:#}"),
                    }),
                };
            }
        };
        let result = &candidate.result;
        let mut document = Value::Null;
        let mut checks = Value::Null;
        let mut normalized_mrz = Value::Null;
        let mut parse_error = Value::Null;
        let mut recovered = false;
        let mut valid = false;
        match &result.parsed {
            Ok(data) => {
                valid = data.checksum_consistent();
                recovered = data.damaged_recovery;
                normalized_mrz = json!(data.mrz_lines);
                document = json!({
                    "format": format!("{:?}", data.format),
                    "document_type": data.document_type,
                    "document_number": data.full_document_number(),
                    "surname": data.surname,
                    "given_names": data.given_names,
                    "sex": data.sex.to_string(),
                    "date_of_birth": data.date_of_birth.to_string(),
                    "nationality": data.nationality,
                    "issuing_country": data.issuing_country,
                    "date_of_expiry": data.date_of_expiry.to_string(),
                });
                let primary = if data.format == mrz::Format::Td3 {
                    "personal_number"
                } else {
                    "optional_data_1"
                };
                for (key, value) in [
                    (primary, data.optional_data_1.as_deref()),
                    ("optional_data_2", data.optional_data_2.as_deref()),
                ] {
                    if let Some(value) = value.filter(|value| !value.is_empty()) {
                        document[key] = json!(value);
                    }
                }
                checks = json!({
                    "document_number": data.checks.document_number,
                    "date_of_birth": data.checks.date_of_birth,
                    "date_of_expiry": data.checks.date_of_expiry,
                    "personal_number": data.checks.personal_number,
                    "composite": data.checks.composite,
                });
            }
            Err(error) => parse_error = json!(error.to_string()),
        }
        Self {
            status: if valid {
                Status::Ok
            } else {
                Status::Unverified
            },
            json: json!({
                "path": path.to_string_lossy(),
                "status": if valid { "ok" } else { "unverified" },
                "ocr_lines": result.ocr_lines,
                "normalized_mrz": normalized_mrz,
                "polygon": result.polygon.map(|point| [point.x, point.y]),
                "rotation": candidate.rotation,
                "center_crop": candidate.center_crop,
                "checksum_consistent": valid,
                "damaged_recovery": recovered,
                "document": document,
                "checks": checks,
                "parse_error": parse_error,
            }),
        }
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
struct Summary {
    ok: usize,
    unverified: usize,
    errors: usize,
}

impl Summary {
    fn add(&mut self, status: Status) {
        match status {
            Status::Ok => self.ok += 1,
            Status::Unverified => self.unverified += 1,
            Status::Error => self.errors += 1,
        }
    }

    fn outcome(&self) -> Result<bool> {
        if self.errors != 0 {
            bail!("{} image(s) failed; see JSONL error records", self.errors);
        }
        Ok(self.unverified == 0)
    }
}

fn images_in(directory: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(directory)
        .with_context(|| format!("cannot read directory {}", directory.display()))?
    {
        let entry = entry?;
        // Do not follow directory or file symlinks. In particular, no recursion.
        if !entry.file_type()?.is_file() {
            continue;
        }
        let path = entry.path();
        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| {
                ["jpg", "jpeg", "png"]
                    .iter()
                    .any(|supported| ext.eq_ignore_ascii_case(supported))
            })
        {
            paths.push(path);
        }
    }
    paths.sort();
    if paths.is_empty() {
        bail!("no JPEG or PNG images in {}", directory.display());
    }
    Ok(paths)
}

fn output_file(path: &Path) -> Result<fs::File> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| {
            format!(
                "cannot create output {} (must not already exist)",
                path.display()
            )
        })
}

fn write_record(out: &mut impl Write, record: &Record) -> Result<()> {
    serde_json::to_writer(&mut *out, &record.json)?;
    writeln!(out)?;
    // Persist each completed record to the OS, not just an application buffer.
    // This is not an fsync/durability guarantee in the event of power loss.
    out.flush()?;
    Ok(())
}

/// The calling thread alone consumes/writes records. Bounded results provide
/// backpressure; dropping the receiver on an output error stops all producers.
/// Scanners are created once by the caller and moved, not shared behind a lock.
fn run_pool<S: Send>(
    paths: &[PathBuf],
    states: Vec<S>,
    process: impl Fn(&mut S, usize, &Path) -> Record + Sync,
    mut consume: impl FnMut(&Record) -> Result<()>,
) -> Result<Summary> {
    assert!(!states.is_empty());
    let next = AtomicUsize::new(0);
    let (sender, receiver) = mpsc::sync_channel(states.len());
    thread::scope(|scope| {
        let mut handles = Vec::new();
        for mut state in states {
            let sender = sender.clone();
            let next = &next;
            let process = &process;
            handles.push(scope.spawn(move || {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(path) = paths.get(index) else { break };
                    if sender.send(process(&mut state, index, path)).is_err() {
                        break;
                    }
                }
            }));
        }
        drop(sender);
        let mut summary = Summary::default();
        let mut outcome = Ok(());
        for record in &receiver {
            if let Err(error) = consume(&record) {
                outcome = Err(error.context("cannot write JSONL results"));
                break;
            }
            summary.add(record.status);
        }
        drop(receiver);
        // Join explicitly so a worker panic is a fatal error, never silent loss
        // of records (or a successful exit with only part of the input done).
        for handle in handles {
            if handle.join().is_err() && outcome.is_ok() {
                outcome = Err(anyhow::anyhow!("image worker panicked"));
            }
        }
        outcome?;
        Ok(summary)
    })
}

pub(super) fn run(args: &Args) -> Result<bool> {
    let directory = args.image.is_dir();
    let paths = if directory {
        images_in(&args.image)?
    } else {
        vec![args.image.clone()]
    };
    let file = args.output.as_deref().map(output_file).transpose()?;
    let stdout = io::stdout();
    let mut out: Box<dyn Write + '_> = match file {
        Some(file) => Box::new(BufWriter::new(file)),
        None => Box::new(stdout.lock()),
    };
    let workers = args.workers.min(paths.len());
    let environment = ort::init().build()?;
    let states = (0..workers)
        .map(|_| MrzScanner::with_environment(&environment, &args.models_dir, args.threads))
        .collect::<Result<Vec<_>>>()?;
    let mut completed = 0;
    let summary = run_pool(
        &paths,
        states,
        |scanner, index, path| {
            let result = (|| {
                let image = image::open(path)
                    .with_context(|| format!("cannot decode image {}", path.display()))?
                    .to_rgb8();
                let debug_dir = args.debug_dir.as_ref().map(|dir| {
                    if directory {
                        dir.join(format!("{index:08}"))
                    } else {
                        dir.clone()
                    }
                });
                run_attempts(
                    &image,
                    args.no_rotate,
                    |oriented, rotation, crop| match &debug_dir {
                        Some(dir) => {
                            let mode = if crop { "crop" } else { "full" };
                            scanner.scan_with_debug(oriented, dir, &format!("{mode}-{rotation}"))
                        }
                        None => scanner.scan(oriented),
                    },
                )
                .with_context(|| format!("no readable MRZ found in {}", path.display()))
            })();
            let mut record = Record::from_result(path, result);
            if directory && let Some(dir) = &args.debug_dir {
                record.json["debug_directory"] =
                    json!(dir.join(format!("{index:08}")).to_string_lossy());
            }
            record
        },
        |record| {
            write_record(&mut out, record)?;
            completed += 1;
            if directory {
                // Best-effort diagnostics; a closed stderr must not stop output.
                let _ = writeln!(io::stderr().lock(), "Processed {completed}/{}", paths.len());
            }
            Ok(())
        },
    )?;
    if directory {
        let _ = writeln!(
            io::stderr().lock(),
            "Summary: {} ok, {} unverified, {} errors",
            summary.ok,
            summary.unverified,
            summary.errors
        );
    }
    summary.outcome()
}

#[cfg(test)]
#[path = "batch_tests.rs"]
mod tests;
