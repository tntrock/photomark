//! Two-dimensional FFT built on `rustfft`.

use std::sync::Arc;

use rayon::prelude::*;
use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftDirection, FftPlanner};

/// A reusable, unnormalised 2-D FFT for row-major `w` x `h` buffers.
pub struct Fft2 {
    w: usize,
    h: usize,
    rows: Arc<dyn Fft<f32>>,
    cols: Arc<dyn Fft<f32>>,
}

impl Fft2 {
    pub fn new(w: usize, h: usize, direction: FftDirection) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        Self {
            w,
            h,
            rows: planner.plan_fft(w, direction),
            cols: planner.plan_fft(h, direction),
        }
    }

    /// Transform `data` in place.
    pub fn process(&self, data: &mut [Complex32]) {
        let (w, h) = (self.w, self.h);
        assert_eq!(data.len(), w * h);

        data.par_chunks_mut(w)
            .for_each(|row| self.rows.process(row));

        let mut cols = vec![Complex32::default(); w * h];
        let source = &*data;
        cols.par_chunks_mut(h).enumerate().for_each(|(x, col)| {
            for (y, v) in col.iter_mut().enumerate() {
                *v = source[y * w + x];
            }
            self.cols.process(col);
        });
        data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for (x, v) in row.iter_mut().enumerate() {
                *v = cols[x * h + y];
            }
        });
    }
}

/// Smallest size `>= n` whose only prime factors are 2, 3 and 5 (fast for the FFT).
pub fn good_size(n: usize) -> usize {
    let mut size = n.max(1);
    loop {
        let mut rest = size;
        for p in [2, 3, 5] {
            while rest.is_multiple_of(p) {
                rest /= p;
            }
        }
        if rest == 1 {
            return size;
        }
        size += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn good_sizes() {
        assert_eq!(good_size(1), 1);
        assert_eq!(good_size(7), 8);
        assert_eq!(good_size(121), 125);
        assert_eq!(good_size(128), 128);
    }

    #[test]
    fn forward_inverse_round_trip() {
        let (w, h) = (12, 10);
        let src: Vec<Complex32> = (0..w * h)
            .map(|i| Complex32::new(((i * 7) % 13) as f32, 0.0))
            .collect();
        let mut data = src.clone();
        Fft2::new(w, h, FftDirection::Forward).process(&mut data);
        Fft2::new(w, h, FftDirection::Inverse).process(&mut data);
        for (a, b) in data.iter().zip(&src) {
            assert!((a.re / (w * h) as f32 - b.re).abs() < 1e-3);
            assert!(a.im.abs() / ((w * h) as f32) < 1e-3);
        }
    }
}
