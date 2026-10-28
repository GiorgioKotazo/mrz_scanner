use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use image::RgbImage;
use imageproc::point::Point;
use mrz::MrzData;
use ndarray::Array4;
use ort::environment::Environment;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;

use crate::image_processing::{
    DETECTION_SIZE, detection_heatmap, detection_letterbox, largest_mrz_polygon, otsu_binarize,
    to_bgr_tensor, warp_quadrangle,
};

pub const DETECTION_MODEL: &str = "mrz_detection_20250222_fp32.onnx";
pub const RECOGNITION_MODEL: &str = "mrz_recognition_20250221_fp32.onnx";

pub struct MrzScanner {
    detection: OnnxModel,
    recognition: OnnxModel,
    #[cfg(test)]
    recognition_runs: usize,
}

pub struct ScanResult {
    pub ocr_lines: Vec<String>,
    pub parsed: Result<MrzData, mrz::MrzError>,
    pub polygon: [Point<f32>; 4],
}

impl MrzScanner {
    pub fn new(models_dir: &Path, threads: usize) -> Result<Self> {
        let environment = ort::init().build()?;
        let detection_path = models_dir.join(DETECTION_MODEL);
        let recognition_path = models_dir.join(RECOGNITION_MODEL);

        Ok(Self {
            detection: OnnxModel::new(&environment, &detection_path, threads)?,
            recognition: OnnxModel::new(&environment, &recognition_path, threads)?,
            #[cfg(test)]
            recognition_runs: 0,
        })
    }

    pub fn scan(&mut self, image: &RgbImage) -> Result<ScanResult> {
        self.scan_internal(image, None)
    }

    pub fn scan_with_debug(
        &mut self,
        image: &RgbImage,
        debug_dir: &Path,
        debug_name: &str,
    ) -> Result<ScanResult> {
        self.scan_internal(image, Some((debug_dir, debug_name)))
    }

    fn scan_internal(
        &mut self,
        image: &RgbImage,
        debug: Option<(&Path, &str)>,
    ) -> Result<ScanResult> {
        let (letterboxed, transform) = detection_letterbox(image)?;
        let detection_input = to_bgr_tensor(&letterboxed, DETECTION_SIZE, DETECTION_SIZE);
        let (values, shape) = self.detection.run(detection_input)?;
        let heatmap = detection_heatmap(&values, &shape, &transform)?;
        if let Some((dir, name)) = debug {
            std::fs::create_dir_all(dir)?;
            letterboxed.save(dir.join(format!("detection-input-{name}.png")))?;
            heatmap.save(dir.join(format!("heatmap-{name}.png")))?;
            otsu_binarize(&heatmap).save(dir.join(format!("binary-{name}.png")))?;
        }
        let polygon =
            largest_mrz_polygon(&heatmap).context("no plausible MRZ region was detected")?;
        let polygon = transform.to_original(polygon, heatmap.dimensions());

        let warped = warp_quadrangle(image, &polygon)?;
        if let Some((dir, name)) = debug {
            warped.save(dir.join(format!("warped-{name}.png")))?;
        }
        let recognition_input = to_bgr_tensor(&warped, 640, 64);
        #[cfg(test)]
        {
            self.recognition_runs += 1;
        }
        let (values, shape) = self.recognition.run(recognition_input)?;
        let ocr_lines = decode_recognition(&values, &shape)?;
        let text = ocr_lines.join("\n");
        let parsed = mrz::find_and_parse(&text);

        Ok(ScanResult {
            ocr_lines,
            parsed,
            polygon,
        })
    }
}

struct OnnxModel {
    session: Session,
    output_name: String,
    _path: PathBuf,
}

impl OnnxModel {
    fn new(environment: &Environment, path: &Path, threads: usize) -> Result<Self> {
        if !path.is_file() {
            bail!(
                "model not found: {}. Run scripts/download_models.sh or pass --models-dir",
                path.display()
            );
        }

        let session = Session::builder(environment)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?
            .with_intra_threads(threads)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?
            .commit_from_file(path)
            .with_context(|| format!("failed to load ONNX model {}", path.display()))?;
        let output_name = session
            .outputs()
            .first()
            .context("ONNX model has no outputs")?
            .name()
            .to_owned();

        Ok(Self {
            session,
            output_name,
            _path: path.to_owned(),
        })
    }

    fn run(&mut self, input: Array4<f32>) -> Result<(Vec<f32>, Vec<usize>)> {
        let outputs = self.session.run(ort::inputs![Tensor::from_array(input)?])?;
        let output = outputs[self.output_name.as_str()]
            .try_extract_array::<f32>()
            .context("failed to read float tensor from ONNX output")?;
        let shape = output.shape().to_vec();
        let values = output.iter().copied().collect();
        Ok((values, shape))
    }
}

/// Decoder for DOCSAID's normal (non-CTC) recognition output.
fn decode_recognition(values: &[f32], shape: &[usize]) -> Result<Vec<String>> {
    const PAD: usize = 0;
    const EOS: usize = 1;
    const SEPARATOR: usize = 2;
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<";
    const CLASSES: usize = ALPHABET.len() + 3;

    if shape.is_empty() || shape[shape.len() - 1] != CLASSES {
        bail!("unexpected recognition output shape: {shape:?}; expected last dimension {CLASSES}");
    }
    if !values.len().is_multiple_of(CLASSES) {
        bail!("recognition output size is not divisible by the class count");
    }

    let mut lines = vec![String::new()];
    for timestep in values.as_chunks::<CLASSES>().0 {
        let class = timestep
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index)
            .unwrap_or(PAD);

        match class {
            EOS => break,
            PAD => {}
            SEPARATOR => lines.push(String::new()),
            index if index >= 3 => {
                if let Some(line) = lines.last_mut() {
                    line.push(ALPHABET[index - 3] as char);
                }
            }
            _ => unreachable!(),
        }
    }

    while lines.last().is_some_and(String::is_empty) {
        lines.pop();
    }
    if lines.is_empty() {
        bail!("recognition model returned an empty MRZ");
    }
    Ok(lines)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires DOCSAID ONNX models; run with --release --ignored"]
    fn detection_model_regression() {
        use crate::image_processing::resize_rgb_bilinear;
        use image::{Rgb, imageops};
        use imageproc::geometric_transformations::{Border, Interpolation, Projection, warp_into};

        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let fixtures = root.join("tests/regression/fixtures");
        let mut scanner = MrzScanner::new(&root.join("models"), 2).unwrap();
        for name in ["white.png", "black.png", "solid.png", "photo.jpg"] {
            let image = image::open(fixtures.join(name)).unwrap().to_rgb8();
            for input in [image.clone(), crate::center_square(&image)] {
                for rotation in [0, 90, 180, 270] {
                    assert!(
                        scanner.scan(&crate::rotate(&input, rotation)).is_err(),
                        "{name}/{rotation}"
                    );
                }
            }
        }
        assert_eq!(
            scanner.recognition_runs, 0,
            "negative fixtures must not reach recognition"
        );

        let original = image::open(fixtures.join("td3.png")).unwrap().to_rgb8();
        let small = resize_rgb_bilinear(&original, 608, 448);
        let (w, h) = (original.width() as f32, original.height() as f32);
        let projection = Projection::from_control_points(
            [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)],
            [
                (30.0, 15.0),
                (w - 50.0, 0.0),
                (w - 10.0, h - 15.0),
                (10.0, h - 40.0),
            ],
        )
        .unwrap();
        let mut perspective = RgbImage::new(original.width(), original.height());
        warp_into(
            &original,
            projection,
            Interpolation::Bilinear,
            Border::Constant(Rgb([238, 233, 221])),
            &mut perspective,
        );
        for (name, image) in [
            ("original", original),
            ("small", small),
            ("perspective", perspective),
        ] {
            let result = scanner.scan(&image).unwrap();
            let data = result.parsed.unwrap();
            assert!(data.checksum_consistent(), "{name}");
            assert_eq!(data.surname, "TESTER", "{name}");
            assert_eq!(data.given_names, "ANNA MARIA", "{name}");
            assert_eq!(data.full_document_number(), "X12345678", "{name}");
            // Also exercise a vertical oriented frame, without changing CLI strategy.
            let rotated = imageops::rotate90(&image);
            assert!(scanner.scan(&rotated).is_ok(), "{name}/90");
        }
        for name in ["parse-failed.png", "checksum-failed.png"] {
            let image = image::open(fixtures.join(name)).unwrap().to_rgb8();
            let result = scanner.scan(&image).unwrap();
            if name == "parse-failed.png" {
                assert!(result.parsed.is_err());
            } else {
                assert!(!result.parsed.unwrap().checksum_consistent());
            }
        }
    }

    #[test]
    fn decodes_lines_and_stops_at_eos() {
        const CLASSES: usize = 40;
        let mut values = vec![0.0_f32; CLASSES * 6];
        let mut set = |step: usize, class: usize| values[step * CLASSES + class] = 1.0;
        set(0, 3); // A
        set(1, 2); // separator
        set(2, 4); // B
        set(3, 39); // <
        set(4, 1); // EOS
        set(5, 5); // ignored

        assert_eq!(
            decode_recognition(&values, &[1, 6, CLASSES]).unwrap(),
            vec!["A", "B<"]
        );
    }
}
