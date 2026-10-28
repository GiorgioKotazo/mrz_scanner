mod image_processing;
mod scanner;

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result, bail};
use clap::Parser;
use image::{RgbImage, imageops};
use mrz::{Format, MrzData};
use scanner::{MrzScanner, ScanResult};

#[derive(Parser, Debug)]
#[command(
    name = "mrz_scanner",
    about = "Read passport MRZ data locally with DOCSAID ONNX models",
    version
)]
struct Args {
    /// Path to a JPEG or PNG passport photo.
    image: PathBuf,

    /// Directory containing the detection and recognition ONNX models.
    #[arg(long, default_value = "models")]
    models_dir: PathBuf,

    /// ONNX Runtime intra-op threads per model.
    #[arg(long, default_value_t = 2)]
    threads: usize,

    /// Save intermediate heatmaps and rectified MRZ images (contains personal data).
    #[arg(long)]
    debug_dir: Option<PathBuf>,

    /// Try only the original orientation instead of all four rotations.
    #[arg(long)]
    no_rotate: bool,
}

struct Candidate {
    rotation: u32,
    center_crop: bool,
    result: ScanResult,
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(2),
        Err(error) => {
            eprintln!("Error: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<bool> {
    let args = Args::parse();
    if args.threads == 0 {
        bail!("--threads must be greater than zero");
    }

    let image = image::open(&args.image)
        .with_context(|| format!("cannot decode image {}", args.image.display()))?
        .to_rgb8();
    let mut scanner = MrzScanner::new(&args.models_dir, args.threads)?;

    let mut best: Option<Candidate> = None;
    let mut errors = Vec::new();
    let rotations: &[u32] = if args.no_rotate {
        &[0]
    } else {
        &[0, 90, 180, 270]
    };
    'attempts: for center_crop in [false, true] {
        if center_crop && image.width() == image.height() {
            continue;
        }
        let input = if center_crop {
            center_square(&image)
        } else {
            image.clone()
        };
        for &rotation in rotations {
            let oriented = rotate(&input, rotation);
            let result = match &args.debug_dir {
                Some(dir) => {
                    let mode = if center_crop { "crop" } else { "full" };
                    scanner.scan_with_debug(&oriented, dir, &format!("{mode}-{rotation}"))
                }
                None => scanner.scan(&oriented),
            };
            match result {
                Ok(result) => {
                    let is_valid_td3 = result
                        .parsed
                        .as_ref()
                        .is_ok_and(|data| data.format == Format::Td3 && data.checksum_consistent());
                    let candidate = Candidate {
                        rotation,
                        center_crop,
                        result,
                    };
                    let replace = best
                        .as_ref()
                        .is_none_or(|current| score(&candidate.result) > score(&current.result));
                    if replace {
                        best = Some(candidate);
                    }
                    if is_valid_td3 {
                        break 'attempts;
                    }
                }
                Err(error) => errors.push(format!(
                    "center_crop={center_crop}, rotation {rotation}: {error:#}"
                )),
            }
        }
    }

    let Some(candidate) = best else {
        bail!(
            "no readable MRZ found in {}. Attempts: {}",
            args.image.display(),
            errors.join("; ")
        );
    };

    print_result(&candidate)
}

fn score(result: &ScanResult) -> u32 {
    match &result.parsed {
        Ok(data) => {
            let verified = [
                data.checks.document_number,
                data.checks.date_of_birth,
                data.checks.date_of_expiry,
                data.checks.personal_number,
                data.checks.composite,
            ]
            .into_iter()
            .flatten()
            .filter(|valid| *valid)
            .count() as u32;
            100 + verified * 20
                + if data.format == Format::Td3 { 10 } else { 0 }
                + if data.checksum_consistent() { 200 } else { 0 }
                + if data.damaged_recovery { 0 } else { 5 }
        }
        Err(_) => result
            .ocr_lines
            .iter()
            .map(|line| line.len() as u32)
            .sum::<u32>()
            .min(50),
    }
}

fn print_result(candidate: &Candidate) -> Result<bool> {
    let result = &candidate.result;
    println!("MRZ (OCR):");
    for line in &result.ocr_lines {
        println!("{line}");
    }
    if candidate.rotation != 0 {
        println!("Image rotation applied: {}°", candidate.rotation);
    }
    if candidate.center_crop {
        println!("Center crop applied: yes");
    }
    println!(
        "MRZ quadrangle: [{}]",
        result
            .polygon
            .iter()
            .map(|point| format!("({:.1}, {:.1})", point.x, point.y))
            .collect::<Vec<_>>()
            .join(", ")
    );

    match &result.parsed {
        Ok(data) => {
            let raw = result.ocr_lines.join("\n");
            if data.mrz_lines != raw {
                println!("\nMRZ after checksum-guided normalization/repair:");
                println!("{}", data.mrz_lines);
            }
            print_data(data);
            let valid = data.checksum_consistent();
            if valid {
                println!("Checksum status: OK");
            } else {
                println!("Checksum status: FAILED — data may contain OCR errors");
            }
            Ok(valid)
        }
        Err(error) => {
            println!("\nCould not parse MRZ: {error}");
            println!("Checksum status: NOT VERIFIED");
            Ok(false)
        }
    }
}

fn print_data(data: &MrzData) {
    println!("\nPassport data:");
    println!("  Format:           {:?}", data.format);
    println!("  Document type:    {}", data.document_type);
    println!("  Surname:          {}", data.surname);
    println!("  Given names:      {}", data.given_names);
    println!("  Sex:              {}", data.sex);
    println!("  Date of birth:    {}", data.date_of_birth);
    println!("  Nationality:      {}", data.nationality);
    println!("  Passport number:  {}", data.full_document_number());
    println!("  Issuing country:  {}", data.issuing_country);
    println!("  Expiration date:  {}", data.date_of_expiry);
    println!(
        "  Personal number:  {}",
        data.personal_number().unwrap_or("")
    );
    println!("\nCheck digits:");
    println!("  Passport number:  {}", check(data.checks.document_number));
    println!("  Date of birth:    {}", check(data.checks.date_of_birth));
    println!("  Expiration date:  {}", check(data.checks.date_of_expiry));
    println!("  Personal number:  {}", check(data.checks.personal_number));
    println!("  Composite:        {}", check(data.checks.composite));
    if data.damaged_recovery {
        println!("  Recovery:         checksum-guided OCR repair was applied");
    }
}

fn check(value: Option<bool>) -> &'static str {
    match value {
        Some(true) => "OK",
        Some(false) => "FAILED",
        None => "N/A",
    }
}

fn center_square(image: &RgbImage) -> RgbImage {
    let side = image.width().min(image.height());
    let x = (image.width() - side) / 2;
    let y = (image.height() - side) / 2;
    imageops::crop_imm(image, x, y, side, side).to_image()
}

fn rotate(image: &RgbImage, degrees: u32) -> RgbImage {
    match degrees {
        0 => image.clone(),
        90 => imageops::rotate90(image),
        180 => imageops::rotate180(image),
        270 => imageops::rotate270(image),
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn parses_icao_specimen() {
        let data = mrz::parse_td3(
            "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
            "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
        )
        .unwrap();
        assert!(data.checksum_consistent());
        assert_eq!(data.surname, "ERIKSSON");
        assert_eq!(data.given_names, "ANNA MARIA");
        assert_eq!(data.full_document_number(), "L898902C3");
    }
}
