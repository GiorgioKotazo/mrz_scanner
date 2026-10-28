use anyhow::{Context, Result, bail};
use image::{GrayImage, Rgb, RgbImage};
use imageproc::contours::{BorderType, find_contours};
use imageproc::geometric_transformations::{Border, Interpolation, Projection, warp_into};
use imageproc::geometry::{contour_area, min_area_rect};
use imageproc::point::Point;
use ndarray::Array4;

/// Converts an image to the NCHW BGR float tensor expected by DOCSAID models.
pub fn to_bgr_tensor(image: &RgbImage, width: u32, height: u32) -> Array4<f32> {
    let resized = resize_rgb_bilinear(image, width, height);
    let plane = (width * height) as usize;
    let mut data = vec![0.0_f32; plane * 3];

    for (index, pixel) in resized.pixels().enumerate() {
        // The Python reference feeds OpenCV images, hence BGR rather than RGB.
        data[index] = f32::from(pixel[2]) / 255.0;
        data[plane + index] = f32::from(pixel[1]) / 255.0;
        data[plane * 2 + index] = f32::from(pixel[0]) / 255.0;
    }

    Array4::from_shape_vec((1, 3, height as usize, width as usize), data)
        .expect("tensor dimensions match the allocated buffer")
}

pub const DETECTION_SIZE: u32 = 256;
// Native 256px thresholding loses contour detail (notably on TD1). Preserve
// resize-before-Otsu at up to 1536px, even for much larger source images.
pub const MAX_POSTPROCESS_SIZE: u32 = 1536;

/// A virtual square: an odd padding pixel belongs to the right/bottom side.
/// No buffer proportional to this square is ever allocated.
#[derive(Debug, Clone, Copy)]
pub struct Letterbox {
    pub original_size: (u32, u32),
    side: u32,
    left: u32,
    top: u32,
}

impl Letterbox {
    fn new(width: u32, height: u32) -> Result<Self> {
        if width == 0 || height == 0 {
            bail!("cannot letterbox an empty image");
        }
        let side = width.max(height);
        Ok(Self {
            original_size: (width, height),
            side,
            left: (side - width) / 2,
            top: (side - height) / 2,
        })
    }

    /// Maps postprocessing pixel centers back to the oriented image passed to scan.
    pub fn to_original(self, polygon: [Point<f32>; 4], size: (u32, u32)) -> [Point<f32>; 4] {
        polygon.map(|point| {
            Point::new(
                (point.x + 0.5) * self.side as f32 / size.0 as f32 - 0.5 - self.left as f32,
                (point.y + 0.5) * self.side as f32 / size.1 as f32 - 0.5 - self.top as f32,
            )
        })
    }
}

/// Resize a virtual black-padded square directly to the model input. Sampling
/// across the padding boundary uses the same half-pixel bilinear mapping as
/// OpenCV, rather than resizing the content and rounding its target dimensions.
pub fn detection_letterbox(image: &RgbImage) -> Result<(RgbImage, Letterbox)> {
    let transform = Letterbox::new(image.width(), image.height())?;
    let output = sample_rgb_bilinear(
        transform.side,
        transform.side,
        DETECTION_SIZE,
        DETECTION_SIZE,
        |x, y| {
            if x >= transform.left
                && y >= transform.top
                && x - transform.left < transform.original_size.0
                && y - transform.top < transform.original_size.1
            {
                *image.get_pixel(x - transform.left, y - transform.top)
            } else {
                Rgb([0, 0, 0])
            }
        },
    );
    Ok((output, transform))
}

pub fn detection_heatmap(
    values: &[f32],
    shape: &[usize],
    transform: &Letterbox,
) -> Result<GrayImage> {
    if shape.len() < 2 {
        bail!("detection output has fewer than two dimensions: {shape:?}");
    }
    let source_height = shape[shape.len() - 2];
    let source_width = shape[shape.len() - 1];
    if source_width == 0
        || source_height == 0
        || source_width > DETECTION_SIZE as usize
        || source_height > DETECTION_SIZE as usize
    {
        bail!("unexpected detection heatmap dimensions: {shape:?}");
    }
    let expected = source_width
        .checked_mul(source_height)
        .context("detection output dimensions overflow")?;

    if values.len() != expected {
        bail!(
            "unexpected detection output: shape {shape:?}, {} values; expected {expected}",
            values.len()
        );
    }

    let mut heatmap = GrayImage::new(source_width as u32, source_height as u32);
    for (pixel, value) in heatmap.pixels_mut().zip(values) {
        if !value.is_finite() {
            bail!("detection heatmap contains a non-finite value");
        }
        // Model output is a sigmoid heatmap in [0, 1].
        pixel[0] = (value.clamp(0.0, 1.0) * 255.0) as u8;
    }

    let side = transform.side.min(MAX_POSTPROCESS_SIZE);
    Ok(resize_gray_bilinear(&heatmap, side, side))
}

/// Otsu threshold followed by extraction of the largest external contour,
/// matching the essential DOCSAID detection postprocessing.
pub fn largest_mrz_polygon(heatmap: &GrayImage) -> Option<[Point<f32>; 4]> {
    // Calibrated on the public regression set: negative peaks <= 24/255,
    // successful positive attempts >= 192/255 (see detection-step2.md).
    // 51/255 leaves a wide margin; confidence alone never admits a region.
    if heatmap.pixels().map(|p| p[0]).max().unwrap_or(0) < 51 {
        return None;
    }
    let binary = otsu_binarize(heatmap);
    let contours = find_contours::<u32>(&binary);

    let largest = contours
        .iter()
        .filter(|contour| contour.border_type == BorderType::Outer && contour.points.len() >= 3)
        .max_by(|a, b| contour_area(&a.points).total_cmp(&contour_area(&b.points)))?;

    let points: Vec<Point<f32>> = largest
        .points
        .iter()
        .map(|point| Point::new(point.x as f32, point.y as f32))
        .collect();
    if contour_area(&points) <= 0.0 {
        return None;
    }
    let polygon = min_area_rect(&points);
    plausible_mrz_polygon(&polygon, heatmap.dimensions()).then_some(polygon)
}

/// Conservative geometry limits in 256x256 detection coordinates, independent
/// of source resolution. Reject unresolved strips and specks, not OCR failures.
/// The broad aspect range allows TD1 and perspective foreshortening;
/// aspect ratio alone is not sufficient to admit a region.
fn plausible_mrz_polygon(polygon: &[Point<f32>; 4], size: (u32, u32)) -> bool {
    if polygon.iter().any(|p| !p.x.is_finite() || !p.y.is_finite()) {
        return false;
    }
    let normalized = polygon.map(|p| {
        Point::new(
            p.x * DETECTION_SIZE as f32 / size.0 as f32,
            p.y * DETECTION_SIZE as f32 / size.1 as f32,
        )
    });
    let a = distance(normalized[0], normalized[1]);
    let b = distance(normalized[1], normalized[2]);
    let short = a.min(b);
    let long = a.max(b);
    short >= 3.0 && long >= 8.0 && a * b >= 32.0 && (1.2..=30.0).contains(&(long / short))
}

pub fn otsu_binarize(image: &GrayImage) -> GrayImage {
    let mut histogram = [0_u64; 256];
    for pixel in image.pixels() {
        histogram[pixel[0] as usize] += 1;
    }

    let sum: f64 = histogram
        .iter()
        .enumerate()
        .map(|(level, count)| level as f64 * *count as f64)
        .sum();

    let mut background_weight = 0_u64;
    let mut background_sum = 0.0_f64;
    let mut best_variance = -1.0_f64;
    let mut threshold = 0_u8;

    for (level, count) in histogram.iter().enumerate() {
        background_weight += count;
        if background_weight == 0 {
            continue;
        }
        let foreground_weight = (image.width() * image.height()) as u64 - background_weight;
        if foreground_weight == 0 {
            break;
        }

        background_sum += level as f64 * *count as f64;
        let background_mean = background_sum / background_weight as f64;
        let foreground_mean = (sum - background_sum) / foreground_weight as f64;
        let variance = background_weight as f64
            * foreground_weight as f64
            * (background_mean - foreground_mean).powi(2);

        if variance > best_variance {
            best_variance = variance;
            threshold = level as u8;
        }
    }

    let mut binary = GrayImage::new(image.width(), image.height());
    for (output, input) in binary.pixels_mut().zip(image.pixels()) {
        output[0] = if input[0] > threshold { 255 } else { 0 };
    }
    binary
}

/// Rectifies the detected quadrangle before recognition.
pub fn warp_quadrangle(image: &RgbImage, polygon: &[Point<f32>; 4]) -> Result<RgbImage> {
    let top = distance(polygon[0], polygon[1]);
    let side = distance(polygon[1], polygon[2]);
    let diagonal = (image.width() as f32).hypot(image.height() as f32);
    if polygon.iter().any(|p| !p.x.is_finite() || !p.y.is_finite())
        || top.min(side) < 2.0
        || top.max(side) > diagonal
        || contour_area(polygon) < 4.0
    {
        bail!("degenerate or oversized MRZ quadrangle");
    }
    let center = Point::new(
        polygon.iter().map(|p| p.x).sum::<f32>() / 4.0,
        polygon.iter().map(|p| p.y).sum::<f32>() / 4.0,
    );
    if !(0.0..image.width() as f32).contains(&center.x)
        || !(0.0..image.height() as f32).contains(&center.y)
    {
        bail!("MRZ quadrangle is outside the image");
    }
    let width = top.max(side) as u32;
    let height = top.min(side) as u32;

    let source = [
        (polygon[0].x, polygon[0].y),
        (polygon[1].x, polygon[1].y),
        (polygon[2].x, polygon[2].y),
        (polygon[3].x, polygon[3].y),
    ];
    let destination = [
        (0.0, 0.0),
        (width as f32, 0.0),
        (width as f32, height as f32),
        (0.0, height as f32),
    ];
    let projection = Projection::from_control_points(source, destination)
        .context("cannot build perspective transform for detected MRZ")?;

    let mut output = RgbImage::new(width, height);
    warp_into(
        image,
        projection,
        Interpolation::Bilinear,
        Border::Constant(Rgb([0, 0, 0])),
        &mut output,
    );
    Ok(output)
}

/// OpenCV INTER_LINEAR-compatible resize. The `image` crate's triangle
/// filter uses a different kernel, which materially changes this detector's output.
pub fn resize_rgb_bilinear(image: &RgbImage, width: u32, height: u32) -> RgbImage {
    sample_rgb_bilinear(image.width(), image.height(), width, height, |x, y| {
        *image.get_pixel(x, y)
    })
}

fn sample_rgb_bilinear(
    source_width: u32,
    source_height: u32,
    width: u32,
    height: u32,
    pixel_at: impl Fn(u32, u32) -> Rgb<u8>,
) -> RgbImage {
    let x_weights = interpolation_weights(source_width, width);
    let y_weights = interpolation_weights(source_height, height);
    let mut output = RgbImage::new(width, height);

    for (dst_y, (y0, y1, fy)) in y_weights.iter().enumerate() {
        for (dst_x, (x0, x1, fx)) in x_weights.iter().enumerate() {
            let p00 = pixel_at(*x0 as u32, *y0 as u32);
            let p01 = pixel_at(*x1 as u32, *y0 as u32);
            let p10 = pixel_at(*x0 as u32, *y1 as u32);
            let p11 = pixel_at(*x1 as u32, *y1 as u32);
            let mut pixel = [0_u8; 3];
            for channel in 0..3 {
                let top = p00[channel] as f32 * (1.0 - fx) + p01[channel] as f32 * fx;
                let bottom = p10[channel] as f32 * (1.0 - fx) + p11[channel] as f32 * fx;
                pixel[channel] = (top * (1.0 - fy) + bottom * fy).round().clamp(0.0, 255.0) as u8;
            }
            output.put_pixel(dst_x as u32, dst_y as u32, Rgb(pixel));
        }
    }
    output
}

fn resize_gray_bilinear(image: &GrayImage, width: u32, height: u32) -> GrayImage {
    let x_weights = interpolation_weights(image.width(), width);
    let y_weights = interpolation_weights(image.height(), height);
    let mut output = GrayImage::new(width, height);
    for (dst_y, (y0, y1, fy)) in y_weights.iter().enumerate() {
        for (dst_x, (x0, x1, fx)) in x_weights.iter().enumerate() {
            let p00 = image.get_pixel(*x0 as u32, *y0 as u32)[0] as f32;
            let p01 = image.get_pixel(*x1 as u32, *y0 as u32)[0] as f32;
            let p10 = image.get_pixel(*x0 as u32, *y1 as u32)[0] as f32;
            let p11 = image.get_pixel(*x1 as u32, *y1 as u32)[0] as f32;
            let top = p00 * (1.0 - fx) + p01 * fx;
            let bottom = p10 * (1.0 - fx) + p11 * fx;
            output.get_pixel_mut(dst_x as u32, dst_y as u32)[0] =
                (top * (1.0 - fy) + bottom * fy).round().clamp(0.0, 255.0) as u8;
        }
    }
    output
}

/// Source indices and fraction for OpenCV's `(dst + 0.5) * scale - 0.5` mapping.
fn interpolation_weights(source: u32, destination: u32) -> Vec<(usize, usize, f32)> {
    assert!(source > 0 && destination > 0);
    if source == 1 {
        return vec![(0, 0, 0.0); destination as usize];
    }

    let scale = source as f32 / destination as f32;
    (0..destination)
        .map(|dst| {
            let position = (dst as f32 + 0.5) * scale - 0.5;
            let mut left = position.floor() as i32;
            let mut fraction = position - left as f32;
            if left < 0 {
                left = 0;
                fraction = 0.0;
            } else if left >= source as i32 - 1 {
                left = source as i32 - 2;
                fraction = 1.0;
            }
            (left as usize, left as usize + 1, fraction)
        })
        .collect()
}

fn distance(a: Point<f32>, b: Point<f32>) -> f32 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tensor_uses_bgr_channel_order() {
        let image = RgbImage::from_pixel(1, 1, Rgb([255, 128, 0]));
        let tensor = to_bgr_tensor(&image, 1, 1);
        assert_eq!(tensor.as_slice().unwrap(), &[0.0, 128.0 / 255.0, 1.0]);
    }

    #[test]
    fn otsu_separates_two_levels() {
        let image = GrayImage::from_vec(4, 1, vec![0, 0, 255, 255]).unwrap();
        assert_eq!(
            otsu_binarize(&image)
                .pixels()
                .map(|p| p[0])
                .collect::<Vec<_>>(),
            vec![0, 0, 255, 255]
        );
    }

    fn rectangle(width: f32, height: f32) -> [Point<f32>; 4] {
        [
            Point::new(0.0, 0.0),
            Point::new(width, 0.0),
            Point::new(width, height),
            Point::new(0.0, height),
        ]
    }

    #[test]
    fn tensor_preserves_nchw_spatial_order() {
        let image = RgbImage::from_raw(2, 1, vec![255, 0, 0, 0, 128, 255]).unwrap();
        let tensor = to_bgr_tensor(&image, 2, 1);
        assert_eq!(tensor.shape(), &[1, 3, 1, 2]);
        assert_eq!(
            tensor.as_slice().unwrap(),
            &[0.0, 1.0, 0.0, 128.0 / 255.0, 1.0, 0.0]
        );
    }

    #[test]
    fn letterbox_inverse_preserves_odd_padding_and_pixel_centers() {
        for (width, height) in [(2, 3), (4, 3), (3, 4), (640, 481), (6000, 4000)] {
            let transform = Letterbox::new(width, height).unwrap();
            for size in [256, 1536] {
                let original = rectangle((width - 1) as f32, (height - 1) as f32);
                let scaled = original.map(|p| {
                    Point::new(
                        (p.x + transform.left as f32 + 0.5) * size as f32 / transform.side as f32
                            - 0.5,
                        (p.y + transform.top as f32 + 0.5) * size as f32 / transform.side as f32
                            - 0.5,
                    )
                });
                let restored = transform.to_original(scaled, (size, size));
                for (a, b) in original.into_iter().zip(restored) {
                    assert!(distance(a, b) < 0.001, "{a:?} != {b:?}");
                }
            }
            assert!(transform.side - width - transform.left <= transform.left + 1);
            assert!(transform.side - height - transform.top <= transform.top + 1);
        }
    }

    #[test]
    fn postprocessing_is_bounded_for_large_and_panoramic_sources() {
        let values = vec![0.0; (DETECTION_SIZE * DETECTION_SIZE) as usize];
        for (width, height) in [(6000, 4000), (60000, 800)] {
            let transform = Letterbox::new(width, height).unwrap();
            let heatmap = detection_heatmap(&values, &[1, 1, 256, 256], &transform).unwrap();
            assert_eq!(
                heatmap.dimensions(),
                (MAX_POSTPROCESS_SIZE, MAX_POSTPROCESS_SIZE)
            );
            assert_eq!(otsu_binarize(&heatmap).dimensions(), heatmap.dimensions());
            assert!(largest_mrz_polygon(&heatmap).is_none());
        }
        let panorama = RgbImage::new(6000, 4);
        assert_eq!(
            detection_letterbox(&panorama).unwrap().0.dimensions(),
            (256, 256)
        );
    }

    #[test]
    fn malformed_detection_outputs_and_empty_images_are_rejected() {
        let transform = Letterbox::new(256, 256).unwrap();
        for (values, shape) in [
            (vec![], vec![]),
            (vec![], vec![0, 256]),
            (vec![0.0], vec![1, 257]),
            (vec![0.0], vec![2, 2]),
            (vec![f32::NAN], vec![1, 1]),
            (vec![f32::INFINITY], vec![1, 1]),
        ] {
            assert!(detection_heatmap(&values, &shape, &transform).is_err());
        }
        assert!(detection_letterbox(&RgbImage::new(0, 3)).is_err());
    }

    #[test]
    fn geometry_rejects_strips_specks_and_implausible_rectangles() {
        for (width, height) in [
            (0.0, 37.0),
            (2.0, 37.0),
            (8.0, 3.0),
            (50.0, 50.0),
            (200.0, 4.0),
        ] {
            assert!(!plausible_mrz_polygon(
                &rectangle(width, height),
                (256, 256)
            ));
        }
        for (width, height) in [(12.0, 3.0), (90.0, 10.0), (30.0, 20.0)] {
            let polygon = rectangle(width, height);
            assert!(plausible_mrz_polygon(&polygon, (256, 256)));
            let rotated = polygon.map(|p| Point::new(100.0 - p.y, p.x + 50.0));
            assert!(plausible_mrz_polygon(&rotated, (256, 256)));
            let scaled = polygon.map(|p| Point::new(p.x * 4.0, p.y * 4.0));
            assert!(plausible_mrz_polygon(&scaled, (1024, 1024)));
        }
    }

    #[test]
    fn contours_require_both_confidence_and_geometry() {
        for (width, height, confidence, accepted) in [
            (37, 2, 255, false),
            (1, 1, 255, false),
            (90, 10, 24, false),
            (90, 10, 51, true),
            (12, 3, 255, true),
        ] {
            let mut heatmap = GrayImage::new(256, 256);
            for y in 20..=20 + height {
                for x in 20..=20 + width {
                    heatmap.put_pixel(x, y, image::Luma([confidence]));
                }
            }
            assert_eq!(largest_mrz_polygon(&heatmap).is_some(), accepted);
        }
    }

    #[test]
    fn warp_rejects_invalid_geometry_without_clamping_to_one_pixel() {
        let image = RgbImage::new(100, 100);
        for polygon in [
            rectangle(50.0, 0.0),
            rectangle(50.0, 1.0),
            rectangle(500.0, 10.0),
            rectangle(f32::NAN, 10.0),
            rectangle(50.0, 10.0).map(|p| Point::new(p.x + 200.0, p.y)),
            [Point::new(1.0, 1.0); 4],
        ] {
            assert!(warp_quadrangle(&image, &polygon).is_err());
        }
        let perspective = [
            Point::new(10.0, 10.0),
            Point::new(90.0, 15.0),
            Point::new(75.0, 40.0),
            Point::new(20.0, 35.0),
        ];
        let warped = warp_quadrangle(&image, &perspective).unwrap();
        assert!(warped.width() >= 2 && warped.height() >= 2);
    }

    #[test]
    fn bilinear_resize_uses_half_pixel_centers_and_clamped_edges() {
        // Independent expected samples, not a comparison of two callers of
        // the same interpolation helper. Quarter weights give 64 and 191.
        let gray = GrayImage::from_raw(2, 1, vec![0, 255]).unwrap();
        assert_eq!(
            resize_gray_bilinear(&gray, 4, 1).into_raw(),
            [0, 64, 191, 255]
        );
        let vertical = GrayImage::from_raw(1, 2, vec![0, 255]).unwrap();
        assert_eq!(
            resize_gray_bilinear(&vertical, 1, 4).into_raw(),
            [0, 64, 191, 255]
        );
        let rgb = RgbImage::from_raw(2, 1, vec![0, 255, 100, 255, 0, 100]).unwrap();
        assert_eq!(
            resize_rgb_bilinear(&rgb, 4, 1).into_raw(),
            [0, 255, 100, 64, 191, 100, 191, 64, 100, 255, 0, 100]
        );
    }

    #[test]
    fn bilinear_resize_downsamples_both_axes_and_preserves_single_pixels() {
        let rgb = RgbImage::from_raw(
            2,
            2,
            vec![0, 40, 80, 80, 120, 160, 160, 200, 240, 240, 160, 80],
        )
        .unwrap();
        assert_eq!(resize_rgb_bilinear(&rgb, 1, 1).into_raw(), [120, 130, 140]);
        let pixel = Rgb([17, 99, 203]);
        assert_eq!(
            resize_rgb_bilinear(&RgbImage::from_pixel(1, 1, pixel), 3, 5),
            RgbImage::from_pixel(3, 5, pixel)
        );
        assert_eq!(resize_rgb_bilinear(&rgb, 2, 2), rgb);
    }

    #[test]
    fn letterbox_matches_resize_of_correctly_padded_square() {
        for (width, height) in [(2, 3), (4, 3), (3, 4), (640, 480), (333, 790)] {
            let image = RgbImage::from_fn(width, height, |x, y| {
                Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
            });
            let (actual, transform) = detection_letterbox(&image).unwrap();
            let side = width.max(height);
            let mut padded = RgbImage::new(side, side);
            image::imageops::replace(
                &mut padded,
                &image,
                i64::from((side - width) / 2),
                i64::from((side - height) / 2),
            );
            assert_eq!(
                actual,
                resize_rgb_bilinear(&padded, DETECTION_SIZE, DETECTION_SIZE)
            );
            assert_eq!(transform.original_size, (width, height));
        }
    }
}
