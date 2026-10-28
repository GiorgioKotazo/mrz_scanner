use super::*;
use image::Rgb;

#[test]
fn zero_rotation_borrows_full_and_crop_buffers() {
    let image = RgbImage::new(9, 6);
    let crop = center_square(&image);
    for input in [&image, &crop] {
        let oriented = rotate(input, 0);
        assert!(matches!(oriented, Cow::Borrowed(_)));
        assert_eq!(oriented.as_ptr(), input.as_ptr());
        for rotation in [90, 180, 270] {
            assert!(matches!(rotate(input, rotation), Cow::Owned(_)));
        }
    }
    let mut calls = 0;
    let _ = run_attempts(&image, true, |input, rotation, cropped| {
        calls += 1;
        assert_eq!(rotation, 0);
        if !cropped {
            assert_eq!(input.as_ptr(), image.as_ptr());
        } else {
            assert_eq!(input, &crop);
        }
        bail!("continue")
    });
    assert_eq!(calls, 2);
}

#[test]
fn every_attempt_preserves_input_pixels() {
    let image = RgbImage::from_fn(9, 6, |x, y| Rgb([x as u8, y as u8, 42]));
    let mut calls = 0;
    let _ = run_attempts(&image, false, |input, rotation, crop| {
        calls += 1;
        // Reference the previous owned-buffer path, independently of helpers.
        let reference = if crop {
            imageops::crop_imm(&image, 1, 0, 6, 6).to_image()
        } else {
            image.clone()
        };
        let reference = match rotation {
            0 => reference,
            90 => imageops::rotate90(&reference),
            180 => imageops::rotate180(&reference),
            270 => imageops::rotate270(&reference),
            _ => unreachable!(),
        };
        assert_eq!(input, &reference);
        bail!("continue")
    });
    assert_eq!(calls, 8);
}

#[test]
fn selected_polygons_overlay_original_pixels_for_all_rotations_and_crops() {
    // Both axes, even/odd offsets, odd leftover padding, and square inputs.
    for (width, height) in [(10, 6), (12, 6), (9, 6), (6, 10), (6, 12), (6, 9), (7, 7)] {
        let image = RgbImage::from_fn(width, height, |x, y| Rgb([x as u8, y as u8, 42]));
        for crop in [false, true] {
            if crop && width == height {
                continue;
            }
            let side = width.min(height);
            let (x, y, w, h) = if crop {
                ((width - side) / 2, (height - side) / 2, side, side)
            } else {
                (0, 0, width, height)
            };
            let expected = [
                Point::new(x as f32, y as f32),
                Point::new((x + w - 1) as f32, y as f32),
                Point::new((x + w - 1) as f32, (y + h - 1) as f32),
                Point::new(x as f32, (y + h - 1) as f32),
            ];
            for rotation in [0, 90, 180, 270] {
                // Check both the early return and the fallback best-candidate path.
                for clean in [false, true] {
                    let candidate = run_attempts(&image, false, |input, r, c| {
                        if r != rotation || c != crop {
                            bail!("skip this attempt");
                        }
                        // Locate original corners by pixel identity, not by using
                        // a forward transform that could repeat the inverse's bug.
                        let polygon = expected.map(|p| {
                            let (ix, iy, _) = input
                                .enumerate_pixels()
                                .find(|(_, _, pixel)| pixel[0] == p.x as u8 && pixel[1] == p.y as u8)
                                .unwrap();
                            Point::new(ix as f32, iy as f32)
                        });
                        let text = if clean {
                            "P<UTOTESTER<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<<<\nX123456785UTO7408122F1204159ZE184226B<<<<<12"
                        } else {
                            "ABC"
                        };
                        Ok(ScanResult {
                            ocr_lines: text.lines().map(str::to_owned).collect(),
                            parsed: mrz::find_and_parse(text),
                            polygon,
                        })
                    })
                    .unwrap();
                    assert_eq!(candidate.rotation, rotation);
                    assert_eq!(candidate.center_crop, crop);
                    assert_eq!(candidate.result.polygon, expected);
                    assert_eq!(candidate.result.parsed.is_ok(), clean);
                }
            }
        }
    }
}

#[test]
fn inverse_preserves_subpixels_and_does_not_clip_or_reorder_corners() {
    let polygon = [
        Point::new(-0.25, 2.5),
        Point::new(4.75, 2.5),
        Point::new(4.75, 7.25),
        Point::new(-0.25, 7.25),
    ];
    let expected = [
        [(2.75, 7.5), (7.75, 7.5), (7.75, 12.25), (2.75, 12.25)],
        [(5.5, 9.25), (5.5, 4.25), (10.25, 4.25), (10.25, 9.25)],
        [(10.25, 6.5), (5.25, 6.5), (5.25, 1.75), (10.25, 1.75)],
        [(7.5, 4.75), (7.5, 9.75), (2.75, 9.75), (2.75, 4.75)],
    ];
    for (rotation, expected) in [0, 90, 180, 270].into_iter().zip(expected) {
        let transform = AttemptTransform {
            rotation,
            crop_offset: (3, 5),
            input_size: (8, 5),
        };
        assert_eq!(
            transform.to_original(polygon),
            expected.map(|(x, y)| Point::new(x, y))
        );
    }
}
