//! Small image-processing helpers shared by the embedder and the detector.

use image::RgbaImage;
use rayon::prelude::*;

/// Rec. 601 luma of every pixel, in `0.0..=255.0`.
pub fn luma(img: &RgbaImage) -> Vec<f32> {
    img.as_raw()
        .par_chunks_exact(4)
        .map(|p| 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32)
        .collect()
}

/// Mean over a `(2 * radius + 1)` window along one line, shrinking the window at the ends.
fn box_line(src: &[f32], dst: &mut [f32], radius: usize) {
    let n = src.len();
    let mut sum = 0f64;
    let mut hi = 0usize; // exclusive end of the window
    let mut lo = 0usize; // inclusive start of the window
    for (i, out) in dst.iter_mut().enumerate() {
        let want_hi = (i + radius + 1).min(n);
        while hi < want_hi {
            sum += src[hi] as f64;
            hi += 1;
        }
        let want_lo = i.saturating_sub(radius);
        while lo < want_lo {
            sum -= src[lo] as f64;
            lo += 1;
        }
        *out = (sum / (hi - lo) as f64) as f32;
    }
}

/// Box blur with a `(2 * radius + 1)` square window.
pub fn box_blur(src: &[f32], w: usize, h: usize, radius: usize) -> Vec<f32> {
    debug_assert_eq!(src.len(), w * h);
    if radius == 0 {
        return src.to_vec();
    }

    let mut rows = vec![0f32; w * h];
    rows.par_chunks_mut(w)
        .zip(src.par_chunks(w))
        .for_each(|(dst, src)| box_line(src, dst, radius));

    // Columns: work on a transposed copy so every line is contiguous.
    let mut cols = vec![0f32; w * h];
    cols.par_chunks_mut(h).enumerate().for_each(|(x, dst)| {
        let line: Vec<f32> = (0..h).map(|y| rows[y * w + x]).collect();
        box_line(&line, dst, radius);
    });

    let mut out = rows;
    out.par_chunks_mut(w).enumerate().for_each(|(y, dst)| {
        for (x, v) in dst.iter_mut().enumerate() {
            *v = cols[x * h + y];
        }
    });
    out
}

/// Shrink by an integer factor using block averages (partial edge blocks are dropped).
pub fn downscale(src: &[f32], w: usize, h: usize, factor: usize) -> (Vec<f32>, usize, usize) {
    if factor <= 1 {
        return (src.to_vec(), w, h);
    }
    let (nw, nh) = (w / factor, h / factor);
    let norm = 1.0 / (factor * factor) as f32;
    let mut out = vec![0f32; nw * nh];
    out.par_chunks_mut(nw.max(1))
        .enumerate()
        .for_each(|(y, dst)| {
            for (x, v) in dst.iter_mut().enumerate() {
                let mut sum = 0f32;
                for dy in 0..factor {
                    let row = (y * factor + dy) * w + x * factor;
                    sum += src[row..row + factor].iter().sum::<f32>();
                }
                *v = sum * norm;
            }
        });
    (out, nw, nh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_blur_preserves_constant() {
        let src = vec![7.0f32; 9 * 5];
        assert!(box_blur(&src, 9, 5, 3)
            .iter()
            .all(|v| (v - 7.0).abs() < 1e-4));
    }

    #[test]
    fn box_blur_matches_naive() {
        let (w, h, r) = (7usize, 6usize, 2i32);
        let src: Vec<f32> = (0..w * h).map(|i| ((i * 37) % 11) as f32).collect();
        let out = box_blur(&src, w, h, r as usize);
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let (mut sum, mut n) = (0f32, 0f32);
                for yy in (y - r).max(0)..=(y + r).min(h as i32 - 1) {
                    for xx in (x - r).max(0)..=(x + r).min(w as i32 - 1) {
                        sum += src[yy as usize * w + xx as usize];
                        n += 1.0;
                    }
                }
                assert!((out[y as usize * w + x as usize] - sum / n).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn downscale_averages_blocks() {
        let src = vec![1.0, 3.0, 5.0, 7.0, 1.0, 3.0, 5.0, 7.0];
        let (out, w, h) = downscale(&src, 4, 2, 2);
        assert_eq!((w, h), (2, 1));
        assert_eq!(out, vec![2.0, 6.0]);
    }
}
