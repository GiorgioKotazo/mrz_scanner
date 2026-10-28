mod image_processing;
mod scanner;

use std::borrow::Cow;
use std::cmp::Ordering;
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result, bail};
use clap::Parser;
use image::{RgbImage, imageops};
use imageproc::point::Point;
use mrz::MrzData;
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
    // Unlike the scanner's result, this polygon is in original-file pixels.
    result: ScanResult,
}

/// The pipeline crops first, then rotates clockwise. The scanner undoes its
/// own detection letterbox; this transform undoes the remaining attempt steps.
#[derive(Clone, Copy, Debug)]
struct AttemptTransform {
    rotation: u32,
    crop_offset: (u32, u32),
    // Dimensions before rotation (after the optional crop).
    input_size: (u32, u32),
}

impl AttemptTransform {
    fn to_original(self, polygon: [Point<f32>; 4]) -> [Point<f32>; 4] {
        // Integer coordinates denote pixel centers, hence width/height - 1.
        let max_x = self.input_size.0 as f32 - 1.0;
        let max_y = self.input_size.1 as f32 - 1.0;
        polygon.map(|point| {
            let (x, y) = match self.rotation {
                0 => (point.x, point.y),
                90 => (point.y, max_y - point.x),
                180 => (max_x - point.x, max_y - point.y),
                270 => (max_x - point.y, point.x),
                _ => unreachable!(),
            };
            // Reverse rotation BEFORE adding the offset: crop precedes rotation.
            // Preserve corner order and fractional/out-of-frame coordinates.
            Point::new(x + self.crop_offset.0 as f32, y + self.crop_offset.1 as f32)
        })
    }
}

fn main() -> ExitCode {
    exit_code(run())
}

fn exit_code(result: Result<bool>) -> ExitCode {
    match result {
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

    let candidate =
        run_attempts(
            &image,
            args.no_rotate,
            |oriented, rotation, center_crop| match &args.debug_dir {
                Some(dir) => {
                    let mode = if center_crop { "crop" } else { "full" };
                    scanner.scan_with_debug(oriented, dir, &format!("{mode}-{rotation}"))
                }
                None => scanner.scan(oriented),
            },
        )
        .with_context(|| format!("no readable MRZ found in {}", args.image.display()))?;

    print_result(&candidate)
}

// Keep orchestration independent of ONNX so stop/selection rules can be tested
// using prepared scan results. The order is full, then crop, each at 0/90/180/270°.
fn run_attempts(
    image: &RgbImage,
    no_rotate: bool,
    mut scan: impl FnMut(&RgbImage, u32, bool) -> Result<ScanResult>,
) -> Result<Candidate> {
    let mut best: Option<Candidate> = None;
    let mut errors = Vec::new();
    let rotations: &[u32] = if no_rotate { &[0] } else { &[0, 90, 180, 270] };
    for center_crop in [false, true] {
        if center_crop && image.width() == image.height() {
            continue;
        }
        let (input, crop_offset) = if center_crop {
            (
                Cow::Owned(center_square(image)),
                center_square_offset(image),
            )
        } else {
            (Cow::Borrowed(image), (0, 0))
        };
        for &rotation in rotations {
            let transform = AttemptTransform {
                rotation,
                crop_offset,
                input_size: input.dimensions(),
            };
            let oriented = rotate(&input, rotation);
            match scan(&oriented, rotation, center_crop) {
                Ok(mut result) => {
                    result.polygon = transform.to_original(result.polygon);
                    let candidate_score = score(&result);
                    let candidate = Candidate {
                        rotation,
                        center_crop,
                        result,
                    };
                    // A clean checksum-valid result is maximal, regardless of format.
                    // Recovered results remain eligible but never stop the search.
                    if candidate_score.quality == Quality::CleanValid {
                        return Ok(candidate);
                    }
                    let replace = best.as_ref().is_none_or(|current| {
                        candidate_score.cmp(&score(&current.result)) == Ordering::Greater
                    });
                    if replace {
                        best = Some(candidate);
                    }
                }
                Err(error) => errors.push(format!(
                    "center_crop={center_crop}, rotation {rotation}: {error:#}"
                )),
            }
        }
    }

    best.with_context(|| format!("no readable MRZ found. Attempts: {}", errors.join("; ")))
}

// Declaration order is the primary, format-neutral ranking.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Quality {
    Unparseable,
    ChecksumFailed,
    RecoveredValid,
    CleanValid,
}

struct Score {
    quality: Quality,
    verified: u8,
    applicable: u8,
}

impl Score {
    fn cmp(&self, other: &Self) -> Ordering {
        self.quality.cmp(&other.quality).then_with(|| {
            if self.quality != Quality::ChecksumFailed {
                return Ordering::Equal;
            }
            // Compare exact fractions without floating-point rounding. No
            // applicable checks means no evidence: treat it as 0/1, not valid.
            let left = u16::from(self.verified) * u16::from(other.applicable.max(1));
            let right = u16::from(other.verified) * u16::from(self.applicable.max(1));
            left.cmp(&right)
        })
    }
}

fn score(result: &ScanResult) -> Score {
    match &result.parsed {
        Ok(data) => Score {
            quality: if !data.checksum_consistent() {
                Quality::ChecksumFailed
            } else if data.damaged_recovery {
                Quality::RecoveredValid
            } else {
                Quality::CleanValid
            },
            verified: data.checks.verified(),
            applicable: data.checks.applicable(),
        },
        // Without calibrated quality metrics, retain the first plausible
        // region, not the longest garbage OCR. All ties keep the first attempt.
        Err(_) => Score {
            quality: Quality::Unparseable,
            verified: 0,
            applicable: 0,
        },
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
        "MRZ quadrangle: [{}] (original image pixels)",
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

fn center_square_offset(image: &RgbImage) -> (u32, u32) {
    let side = image.width().min(image.height());
    ((image.width() - side) / 2, (image.height() - side) / 2)
}

fn center_square(image: &RgbImage) -> RgbImage {
    let side = image.width().min(image.height());
    let (x, y) = center_square_offset(image);
    imageops::crop_imm(image, x, y, side, side).to_image()
}

fn rotate(image: &RgbImage, degrees: u32) -> Cow<'_, RgbImage> {
    match degrees {
        0 => Cow::Borrowed(image),
        90 => Cow::Owned(imageops::rotate90(image)),
        180 => Cow::Owned(imageops::rotate180(image)),
        270 => Cow::Owned(imageops::rotate270(image)),
        _ => unreachable!(),
    }
}

#[cfg(test)]
#[path = "attempt_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "geometry_tests.rs"]
mod geometry_tests;
