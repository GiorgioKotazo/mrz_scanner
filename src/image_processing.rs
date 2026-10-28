use anyhow::{Context, Result, bail};
use image::{GrayImage, Rgb, RgbImage, imageops};
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

/// Replicates DOCSAID's symmetric black padding before detection.
pub fn pad_to_square(image: &RgbImage) -> (RgbImage, (f32, f32)) {
    let (width, height) = image.dimensions();
    let (left, top) = if height < width {
        (0, (width - height) / 2)
    } else {
        ((height - width) / 2, 0)
    };

    let mut padded = RgbImage::new(width + left * 2, height + top * 2);
    imageops::replace(&mut padded, image, i64::from(left), i64::from(top));
    (padded, (left as f32, top as f32))
}

pub fn resize_heatmap(
    values: &[f32],
    shape: &[usize],
    width: u32,
    height: u32,
) -> Result<GrayImage> {
    if shape.len() < 2 {
        bail!("detection output has fewer than two dimensions: {shape:?}");
    }
    let source_height = shape[shape.len() - 2];
    let source_width = shape[shape.len() - 1];
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
        // Model output is a sigmoid heatmap in [0, 1].
        pixel[0] = (value.clamp(0.0, 1.0) * 255.0) as u8;
    }

    Ok(resize_gray_bilinear(&heatmap, width, height))
}

/// Otsu threshold followed by extraction of the largest external contour,
/// matching the essential DOCSAID detection postprocessing.
pub fn largest_mrz_polygon(heatmap: &GrayImage, shift: (f32, f32)) -> Option<[Point<f32>; 4]> {
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
    let mut polygon = min_area_rect(&points);
    for point in &mut polygon {
        point.x -= shift.0;
        point.y -= shift.1;
    }
    Some(polygon)
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
    let width = top.max(side).max(1.0) as u32;
    let height = top.min(side).max(1.0) as u32;

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
    let x_weights = interpolation_weights(image.width(), width);
    let y_weights = interpolation_weights(image.height(), height);
    let mut output = RgbImage::new(width, height);

    for (dst_y, (y0, y1, fy)) in y_weights.iter().enumerate() {
        for (dst_x, (x0, x1, fx)) in x_weights.iter().enumerate() {
            let p00 = image.get_pixel(*x0 as u32, *y0 as u32);
            let p01 = image.get_pixel(*x1 as u32, *y0 as u32);
            let p10 = image.get_pixel(*x0 as u32, *y1 as u32);
            let p11 = image.get_pixel(*x1 as u32, *y1 as u32);
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

    #[test]
    fn padding_preserves_image_and_reports_shift() {
        let image = RgbImage::from_pixel(2, 4, Rgb([1, 2, 3]));
        let (padded, shift) = pad_to_square(&image);
        assert_eq!(padded.dimensions(), (4, 4));
        assert_eq!(shift, (1.0, 0.0));
        assert_eq!(*padded.get_pixel(1, 0), Rgb([1, 2, 3]));
        assert_eq!(*padded.get_pixel(0, 0), Rgb([0, 0, 0]));
    }
}
