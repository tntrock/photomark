//! Watermark detection.
//!
//! The detector does not know how the image was cropped or rescaled, so it has to
//! recover the tile grid first:
//!
//! 1. High-pass the luma and weight every pixel by the inverse of its local variance, so
//!    smooth regions (where the mark is cleanest) dominate.
//! 2. Estimate the tile period. The mark repeats, so the autocorrelation of the residual
//!    has a lattice of peaks one tile apart; unscaled tile sizes are also always tried.
//! 3. For each candidate period, *fold* the residual modulo the period into a single
//!    tile, which averages every tile in the image together, then cross-correlate that
//!    tile with the key's pilot pattern in all eight orientations. A sharp peak gives the
//!    grid offset (the crop position) and the orientation.
//! 4. Read the data cells from the aligned tile and decode the payload.

use std::collections::HashMap;

use image::RgbaImage;
use rayon::prelude::*;
use rustfft::num_complex::Complex32;
use rustfft::FftDirection;

use crate::codec::{self, CODED_BITS, PAYLOAD_BYTES};
use crate::embed::{MAX_CELL, MIN_CELL};
use crate::fft2::{good_size, Fft2};
use crate::filters::{box_blur, downscale, luma};
use crate::pattern::{Cell, WatermarkKey, CELLS};

/// Bins per tile side in the folded tile.
const GRID: usize = 128;
/// Bins per cell side.
const SUB: usize = GRID / CELLS;

/// Variance floor (in squared 8-bit levels) of the per-pixel weighting.
const VARIANCE_FLOOR: f32 = 2.0;
/// Shortest tile period, in working pixels, the detector looks for.
const MIN_PERIOD: f32 = 40.0;
/// Largest autocorrelation lag used, as a fraction of the image side.
const MAX_LAG: f32 = 0.7;
/// Autocorrelation period candidates to verify.
const PERIOD_CANDIDATES: usize = 10;
/// Relative half-range searched around an autocorrelation period estimate.
const REFINE_RANGE: f64 = 0.006;

/// Pilot score needed when the payload checksum also verifies.
const SCORE_WITH_PAYLOAD: f32 = 5.0;
/// Pilot score that counts as a detection on its own. A marked image checked with the
/// wrong key reaches about 8, because its folded tile is still strongly structured.
const SCORE_ALONE: f32 = 11.0;
/// Pilot score at which the search stops looking for better period candidates.
const SCORE_CERTAIN: f32 = 16.0;

#[derive(Clone, Copy, Debug)]
pub struct DetectOptions {
    /// Images larger than this are first analysed at reduced resolution.
    pub max_work_pixels: usize,
}

impl Default for DetectOptions {
    fn default() -> Self {
        Self {
            max_work_pixels: 4_000_000,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Detection {
    /// Whether a watermark made with this key is present.
    pub detected: bool,
    /// The embedded payload, if it could be read and its checksum verified.
    pub payload: Option<[u8; PAYLOAD_BYTES]>,
    /// Strength of the synchronisation peak, in standard deviations above chance.
    pub score: f32,
    /// Tile period found, in pixels of the analysed image (0 if nothing was found).
    pub tile_period: f32,
    /// Clockwise quarter turns (0..4) the image was rotated relative to the original.
    pub quarter_turns: u8,
    /// Whether the image was mirrored relative to the original.
    pub mirrored: bool,
}

impl Detection {
    fn none() -> Self {
        Self {
            detected: false,
            payload: None,
            score: 0.0,
            tile_period: 0.0,
            quarter_turns: 0,
            mirrored: false,
        }
    }
}

/// A synchronisation hypothesis: the folded tile and where the pilot matched in it.
struct Hit {
    score: f32,
    period: f64,
    orientation: usize,
    dx: usize,
    dy: usize,
    tile: Vec<f32>,
}

/// Cross-correlates folded tiles against the key's pilot pattern.
struct Matcher {
    forward: Fft2,
    inverse: Fft2,
    /// Conjugated spectrum of the pilot template.
    template: Vec<Complex32>,
}

impl Matcher {
    fn new(key: &WatermarkKey) -> Self {
        let forward = Fft2::new(GRID, GRID, FftDirection::Forward);
        let inverse = Fft2::new(GRID, GRID, FftDirection::Inverse);
        let mut template = vec![Complex32::default(); GRID * GRID];
        for (i, cell) in key.cells.iter().enumerate() {
            if let Cell::Pilot(sign) = *cell {
                let (cx, cy) = (i % CELLS, i / CELLS);
                for y in cy * SUB..(cy + 1) * SUB {
                    for x in cx * SUB..(cx + 1) * SUB {
                        template[y * GRID + x].re = sign as f32;
                    }
                }
            }
        }
        forward.process(&mut template);
        for v in &mut template {
            *v = v.conj();
        }
        Self {
            forward,
            inverse,
            template,
        }
    }

    /// Best circular alignment of `tile` with the pilot: `(score, dx, dy)` such that
    /// `tile[(x + dx, y + dy)]` lines up with template cell bin `(x, y)`.
    fn correlate(&self, tile: &[f32]) -> (f32, usize, usize) {
        let mut buf: Vec<Complex32> = tile.iter().map(|&v| Complex32::new(v, 0.0)).collect();
        self.forward.process(&mut buf);
        for (v, t) in buf.iter_mut().zip(&self.template) {
            *v *= t;
        }
        self.inverse.process(&mut buf);

        let n = buf.len() as f64;
        let (mut sum, mut sum_sq) = (0f64, 0f64);
        let (mut best, mut best_at) = (f32::MIN, 0usize);
        for (i, v) in buf.iter().enumerate() {
            sum += v.re as f64;
            sum_sq += (v.re as f64) * (v.re as f64);
            if v.re > best {
                best = v.re;
                best_at = i;
            }
        }
        let mean = sum / n;
        let std = (sum_sq / n - mean * mean).max(0.0).sqrt();
        if std <= 0.0 || !std.is_finite() {
            return (0.0, 0, 0);
        }
        (
            ((best as f64 - mean) / std) as f32,
            best_at % GRID,
            best_at / GRID,
        )
    }
}

/// Apply one of the eight square symmetries: `k & 3` quarter turns, then a mirror if
/// `k & 4`.
fn orient(tile: &[f32], k: usize) -> Vec<f32> {
    let mut out = tile.to_vec();
    for _ in 0..k & 3 {
        let prev = out.clone();
        for y in 0..GRID {
            for x in 0..GRID {
                out[y * GRID + x] = prev[x * GRID + (GRID - 1 - y)];
            }
        }
    }
    if k & 4 != 0 {
        for row in out.chunks_mut(GRID) {
            row.reverse();
        }
    }
    out
}

/// High-passed luma, weighted by inverse local variance.
fn weighted_residual(luma: &[f32], w: usize, h: usize, radius: usize) -> Vec<f32> {
    let mean = box_blur(luma, w, h, radius);
    let mut residual: Vec<f32> = luma.par_iter().zip(&mean).map(|(v, m)| v - m).collect();
    let squares: Vec<f32> = residual.par_iter().map(|v| v * v).collect();
    let variance = box_blur(&squares, w, h, radius * 2 + 1);
    residual
        .par_iter_mut()
        .zip(&variance)
        .for_each(|(r, var)| *r /= var + VARIANCE_FLOOR);
    residual
}

/// Accumulate the image modulo `period` into a `GRID` x `GRID` tile.
fn fold(data: &[f32], w: usize, period: f64) -> Vec<f32> {
    let scale = GRID as f64 / period;
    let bin = |i: usize| ((i as f64 + 0.5) * scale) as usize % GRID;
    let bin_x: Vec<usize> = (0..w).map(bin).collect();

    data.par_chunks(w)
        .enumerate()
        .fold(
            || vec![0f32; GRID * GRID],
            |mut acc, (y, row)| {
                let out = &mut acc[bin(y) * GRID..][..GRID];
                for (v, &bx) in row.iter().zip(&bin_x) {
                    out[bx] += v;
                }
                acc
            },
        )
        .reduce(
            || vec![0f32; GRID * GRID],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(&b) {
                    *x += y;
                }
                a
            },
        )
}

/// Fold at `period` and find the best pilot match over all orientations.
fn evaluate(data: &[f32], w: usize, period: f64, matcher: &Matcher) -> Hit {
    let folded = fold(data, w, period);
    let mut best = Hit {
        score: f32::MIN,
        period,
        orientation: 0,
        dx: 0,
        dy: 0,
        tile: Vec::new(),
    };
    for orientation in 0..8 {
        let tile = orient(&folded, orientation);
        let (score, dx, dy) = matcher.correlate(&tile);
        if score > best.score {
            best = Hit {
                score,
                period,
                orientation,
                dx,
                dy,
                tile,
            };
        }
    }
    best
}

/// Search periods near `estimate` for the sharpest pilot match.
fn refine(data: &[f32], w: usize, h: usize, estimate: f64, matcher: &Matcher) -> Hit {
    // A step that moves the grid by half a cell across the longer image side.
    let step = estimate / (2.0 * CELLS as f64 * w.max(h) as f64);
    let steps = (REFINE_RANGE / step).ceil().min(200.0) as i64;

    let mut best = evaluate(data, w, estimate, matcher);
    for i in (-steps..=steps).filter(|&i| i != 0) {
        let hit = evaluate(data, w, estimate * (1.0 + i as f64 * step), matcher);
        if hit.score > best.score {
            best = hit;
        }
    }

    // Hill-climb with shrinking steps.
    let mut delta = step / 2.0;
    for _ in 0..3 {
        for direction in [-1.0, 1.0] {
            let hit = evaluate(data, w, best.period * (1.0 + direction * delta), matcher);
            if hit.score > best.score {
                best = hit;
            }
        }
        delta /= 2.0;
    }
    best
}

/// Tile period candidates from the autocorrelation of `data`, strongest first.
fn period_candidates(data: &[f32], w: usize, h: usize) -> Vec<f64> {
    let lag_x = (w as f32 * MAX_LAG) as usize;
    let lag_y = (h as f32 * MAX_LAG) as usize;
    // Zero padding keeps lags up to the maximum free of wrap-around.
    let (pw, ph) = (good_size(w + lag_x), good_size(h + lag_y));

    let mut buf = vec![Complex32::default(); pw * ph];
    buf.par_chunks_mut(pw)
        .zip(data.par_chunks(w))
        .for_each(|(dst, src)| {
            for (d, &s) in dst.iter_mut().zip(src) {
                d.re = s;
            }
        });
    Fft2::new(pw, ph, FftDirection::Forward).process(&mut buf);
    buf.par_iter_mut()
        .for_each(|v| *v = Complex32::new(v.norm_sqr(), 0.0));
    Fft2::new(pw, ph, FftDirection::Inverse).process(&mut buf);

    let zero_lag = buf[0].re;
    if zero_lag <= 0.0 || !zero_lag.is_finite() {
        return Vec::new();
    }

    // Bilinear sample of the autocorrelation at lag (lx >= 0, ly of either sign).
    let sample = |lx: f32, ly: f32| -> f32 {
        let (x0, y0) = (lx.floor(), ly.floor());
        let (fx, fy) = (lx - x0, ly - y0);
        let at = |x: i64, y: i64| buf[y.rem_euclid(ph as i64) as usize * pw + x as usize].re;
        let (x0, y0) = (x0 as i64, y0 as i64);
        at(x0, y0) * (1.0 - fx) * (1.0 - fy)
            + at(x0 + 1, y0) * fx * (1.0 - fy)
            + at(x0, y0 + 1) * (1.0 - fx) * fy
            + at(x0 + 1, y0 + 1) * fx * fy
    };

    // Score a period by summing the autocorrelation over its lattice of lags.
    let score = |period: f32| -> Option<f32> {
        let (mut sum, mut overlap) = (0f32, 0f32);
        for m in 0..=4i32 {
            for n in -4..=4i32 {
                if (m == 0 && n <= 0) || m as f32 * period > lag_x as f32 {
                    continue;
                }
                if n.abs() as f32 * period > lag_y as f32 {
                    continue;
                }
                let (lx, ly) = (m as f32 * period, n as f32 * period);
                sum += sample(lx, ly);
                overlap += (1.0 - lx / w as f32) * (1.0 - ly.abs() / h as f32);
            }
        }
        (overlap > 0.0).then(|| sum / zero_lag / overlap.sqrt())
    };

    let max_period = lag_x.max(lag_y) as f32;
    let mut scored: Vec<(f32, f32)> = Vec::new();
    let mut period = MIN_PERIOD;
    while period <= max_period {
        if let Some(s) = score(period) {
            scored.push((period, s));
        }
        period *= 1.0007;
    }

    let mut peaks: Vec<(f32, f32)> = (1..scored.len().saturating_sub(1))
        .filter(|&i| scored[i].1 > scored[i - 1].1 && scored[i].1 >= scored[i + 1].1)
        .map(|i| scored[i])
        .collect();
    peaks.sort_by(|a, b| b.1.total_cmp(&a.1));

    let mut chosen: Vec<f64> = Vec::new();
    for (period, _) in peaks {
        let period = period as f64;
        if chosen.iter().all(|c| (period / c - 1.0).abs() > 0.015) {
            chosen.push(period);
            if chosen.len() == PERIOD_CANDIDATES {
                break;
            }
        }
    }
    chosen
}

/// Read the payload from an aligned hit.
fn decode_hit(hit: &Hit, key: &WatermarkKey) -> Option<[u8; PAYLOAD_BYTES]> {
    // Bins near a cell's edge are the most smeared by misalignment and resampling.
    const WEIGHTS: [f32; SUB] = [0.5, 1.0, 1.0, 0.5];

    let mut soft = [0f32; CODED_BITS];
    for (i, cell) in key.cells.iter().enumerate() {
        let Cell::Data { bit, mask } = *cell else {
            continue;
        };
        let (cx, cy) = (i % CELLS, i / CELLS);
        let mut value = 0f32;
        for (sy, wy) in WEIGHTS.iter().enumerate() {
            let y = (cy * SUB + sy + hit.dy) % GRID;
            for (sx, wx) in WEIGHTS.iter().enumerate() {
                let x = (cx * SUB + sx + hit.dx) % GRID;
                value += wy * wx * hit.tile[y * GRID + x];
            }
        }
        soft[bit as usize] += value * mask as f32;
    }
    codec::decode(&soft)
}

fn debug_enabled() -> bool {
    std::env::var_os("PHOTOMARK_DEBUG").is_some()
}

/// Run the whole search on a working-resolution luma plane. Returns the detection with
/// the tile period in working pixels.
fn search(luma: &[f32], w: usize, h: usize, factor: usize, key: &WatermarkKey) -> Detection {
    if w < 16 || h < 16 {
        return Detection::none();
    }
    let matcher = Matcher::new(key);
    let debug = debug_enabled();

    let mut residuals: HashMap<usize, Vec<f32>> = HashMap::new();
    let mut residual = |radius: usize| -> Vec<f32> {
        residuals
            .entry(radius)
            .or_insert_with(|| weighted_residual(luma, w, h, radius))
            .clone()
    };
    let radius_for = |period: f64| ((period / CELLS as f64 * 0.75).round() as usize).clamp(1, 24);

    let mut hits: Vec<Hit> = Vec::new();

    // Unscaled images are the common case and their period is known exactly.
    for cell in MIN_CELL..=MAX_CELL {
        let period = (cell as usize * CELLS) as f64 / factor as f64;
        if period < MIN_PERIOD as f64 / 2.0 {
            continue;
        }
        let data = residual(radius_for(period));
        hits.push(evaluate(&data, w, period, &matcher));
    }
    if debug {
        for hit in &hits {
            eprintln!("canonical period {:.2}: score {:.2}", hit.period, hit.score);
        }
    }

    let strong = |hits: &[Hit]| hits.iter().any(|h| h.score >= SCORE_CERTAIN);
    if !strong(&hits) {
        for radius in [2usize, 5] {
            for estimate in period_candidates(&residual(radius), w, h) {
                let data = residual(radius_for(estimate));
                let hit = refine(&data, w, h, estimate, &matcher);
                if debug {
                    eprintln!(
                        "autocorr(r={radius}) period {estimate:.2} -> {:.3}: score {:.2}",
                        hit.period, hit.score
                    );
                }
                hits.push(hit);
            }
            if strong(&hits) {
                break;
            }
        }
    }

    hits.sort_by(|a, b| b.score.total_cmp(&a.score));
    let describe = |hit: &Hit, payload: Option<[u8; PAYLOAD_BYTES]>| {
        // `orient` turns counter-clockwise, so the turns that restored the tile equal the
        // clockwise turns that were applied to the image.
        let mirrored = hit.orientation & 4 != 0;
        Detection {
            detected: (payload.is_some() && hit.score >= SCORE_WITH_PAYLOAD)
                || hit.score >= SCORE_ALONE,
            payload,
            score: hit.score,
            tile_period: (hit.period * factor as f64) as f32,
            quarter_turns: (hit.orientation & 3) as u8,
            mirrored,
        }
    };

    for hit in hits.iter().take(3) {
        if hit.score < SCORE_WITH_PAYLOAD {
            break;
        }
        if let Some(payload) = decode_hit(hit, key) {
            return describe(hit, Some(payload));
        }
    }
    match hits.first() {
        Some(hit) => describe(hit, None),
        None => Detection::none(),
    }
}

/// Look for a watermark made with `key` in `img`.
pub fn detect(img: &RgbaImage, key: &WatermarkKey, options: &DetectOptions) -> Detection {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let y = luma(img);

    let budget = options.max_work_pixels.max(1) as f64;
    let factor = ((w * h) as f64 / budget).sqrt().ceil().max(1.0) as usize;
    let (work, ww, wh) = downscale(&y, w, h, factor);
    let reduced = search(&work, ww, wh, factor, key);
    if factor == 1 || reduced.payload.is_some() {
        return reduced;
    }

    // Fine marks can vanish in the reduced image; look again at full resolution.
    let full = search(&y, w, h, 1, key);
    if full.payload.is_some() || full.score > reduced.score {
        full
    } else {
        reduced
    }
}
