//! Watermark embedding.

use image::RgbaImage;
use rayon::prelude::*;

use crate::codec::PAYLOAD_BYTES;
use crate::filters::{box_blur, luma};
use crate::pattern::{WatermarkKey, CELLS};

/// Smallest and largest cell size (in pixels) the embedder will use.
pub const MIN_CELL: u32 = 3;
pub const MAX_CELL: u32 = 16;

/// Luma change, in 8-bit levels, of a cell in a moderately textured area at strength 1.
const BASE_AMPLITUDE: f32 = 2.0;

#[derive(Clone, Copy, Debug)]
pub struct EmbedOptions {
    /// Multiplier on the watermark amplitude. 1.0 is the default trade-off between
    /// invisibility and robustness.
    pub strength: f32,
    /// Cell size in pixels; `None` picks one from the image size.
    pub cell: Option<u32>,
}

impl Default for EmbedOptions {
    fn default() -> Self {
        Self {
            strength: 1.0,
            cell: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EmbedReport {
    /// Cell size used, in pixels. One tile spans `32 * cell` pixels.
    pub cell: u32,
    /// Peak signal-to-noise ratio of the marked image against the original, in dB.
    pub psnr: f32,
}

/// Cell size that keeps roughly a dozen tiles across the short side, so heavy crops
/// still contain several tiles and typical downscaling leaves cells resolvable.
pub fn auto_cell(w: u32, h: u32) -> u32 {
    ((w.min(h) as f32 / 360.0).round() as u32).clamp(MIN_CELL, MAX_CELL)
}

/// Cheap per-pixel hash in `0.0..1.0`, used to dither sub-level amplitudes.
fn dither(x: u32, y: u32) -> f32 {
    let mut v = x.wrapping_mul(0x9E37_79B1) ^ y.wrapping_mul(0x85EB_CA77);
    v ^= v >> 15;
    v = v.wrapping_mul(0x2C1B_3C6D);
    v ^= v >> 12;
    v = v.wrapping_mul(0x297A_2D39);
    v ^= v >> 15;
    (v >> 8) as f32 / (1u32 << 24) as f32
}

/// Embed `payload` into `img` in place.
pub fn embed(
    img: &mut RgbaImage,
    key: &WatermarkKey,
    payload: &[u8; PAYLOAD_BYTES],
    options: &EmbedOptions,
) -> EmbedReport {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let cell = options
        .cell
        .unwrap_or_else(|| auto_cell(w as u32, h as u32))
        .clamp(1, 64) as usize;
    if w == 0 || h == 0 {
        return EmbedReport {
            cell: cell as u32,
            psnr: f32::INFINITY,
        };
    }
    let tile = key.tile(payload);

    // Perceptual mask: busy areas hide a stronger mark than flat ones.
    let y = luma(img);
    let radius = cell.max(2);
    let mean = box_blur(&y, w, h, radius);
    let squares: Vec<f32> = y.par_iter().map(|v| v * v).collect();
    let mean_sq = box_blur(&squares, w, h, radius);
    let amplitude = BASE_AMPLITUDE * options.strength.max(0.0);

    let squared_error: f64 = img
        .par_chunks_exact_mut(w * 4)
        .enumerate()
        .map(|(py, row)| {
            let tile_row = (py / cell) % CELLS * CELLS;
            let mut err = 0f64;
            for (px, pixel) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                let i = py * w + px;
                let sigma = (mean_sq[i] - mean[i] * mean[i]).max(0.0).sqrt();
                let mask = (0.45 + sigma / 8.0).min(2.0);
                let sign = tile[tile_row + (px / cell) % CELLS] as f32;
                let delta = (amplitude * mask * sign + dither(px as u32, py as u32)).floor() as i32;
                for channel in &mut pixel[..3] {
                    let old = *channel as i32;
                    let new = (old + delta).clamp(0, 255);
                    *channel = new as u8;
                    err += ((new - old) * (new - old)) as f64;
                }
            }
            err
        })
        .sum();

    let mse = squared_error / (w * h * 3) as f64;
    let psnr = if mse > 0.0 {
        (10.0 * (255.0f64 * 255.0 / mse).log10()) as f32
    } else {
        f32::INFINITY
    };
    EmbedReport {
        cell: cell as u32,
        psnr,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_cell_scales_with_image() {
        assert_eq!(auto_cell(640, 480), MIN_CELL);
        assert_eq!(auto_cell(1920, 1080), 3);
        assert_eq!(auto_cell(4000, 3000), 8);
        assert_eq!(auto_cell(12000, 9000), MAX_CELL);
    }

    #[test]
    fn dither_is_roughly_uniform() {
        let n = 200 * 200;
        let mean: f32 = (0..200)
            .flat_map(|y| (0..200).map(move |x| dither(x, y)))
            .sum::<f32>()
            / n as f32;
        assert!((mean - 0.5).abs() < 0.01, "mean {mean}");
    }

    #[test]
    fn embedding_is_subtle_and_keeps_alpha() {
        let mut img = RgbaImage::from_fn(256, 256, |x, y| {
            image::Rgba([(x % 256) as u8, (y % 256) as u8, 128, 200])
        });
        let report = embed(
            &mut img,
            &WatermarkKey::new("k"),
            b"abcdefgh",
            &EmbedOptions::default(),
        );
        assert!(
            report.psnr > 36.0 && report.psnr < 60.0,
            "psnr {}",
            report.psnr
        );
        assert!(img.pixels().all(|p| p[3] == 200));
    }
}
