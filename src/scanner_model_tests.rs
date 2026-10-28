//! White-box ONNX regressions, separate from the fast decoder tests.
use super::*;
use crate::image_processing::resize_rgb_bilinear;
use image::{Rgb, imageops};
use imageproc::geometric_transformations::{Border, Interpolation, Projection, warp_into};

fn scanner() -> MrzScanner {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    MrzScanner::new(&root.join("models"), 2).expect("run scripts/download_models.sh first")
}

fn fixture(name: &str) -> RgbImage {
    image::open(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/regression/fixtures")
            .join(name),
    )
    .unwrap()
    .to_rgb8()
}

#[test]
#[ignore = "requires DOCSAID ONNX models; run scripts/test_models.sh"]
fn detection_model_regression() {
    // ort permits only one Environment::build per process. Keep these scenarios
    // in one test/session rather than creating a scanner in each test function.
    let mut scanner = scanner();
    negative_regions_never_run_recognition(&mut scanner);
    small_and_perspective_documents_retain_verified_fields(&mut scanner);
    plausible_unverified_regions_still_reach_recognition(&mut scanner);
}

fn negative_regions_never_run_recognition(scanner: &mut MrzScanner) {
    for name in ["white.png", "black.png", "solid.png", "photo.jpg"] {
        let image = fixture(name);
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
}

fn small_and_perspective_documents_retain_verified_fields(scanner: &mut MrzScanner) {
    let original = fixture("td3.png");
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
}

fn plausible_unverified_regions_still_reach_recognition(scanner: &mut MrzScanner) {
    let before = scanner.recognition_runs;
    for name in ["parse-failed.png", "checksum-failed.png"] {
        let result = scanner.scan(&fixture(name)).unwrap();
        if name == "parse-failed.png" {
            assert!(result.parsed.is_err());
        } else {
            assert!(!result.parsed.unwrap().checksum_consistent());
        }
    }
    assert_eq!(scanner.recognition_runs - before, 2);
}
