//! Neural watermark layer built on Adobe's TrustMark (variant Q) networks.
//!
//! An encoder network turns a 256 x 256 copy of the image and 100 bits into a faint
//! residual, which is scaled up and added to the full-resolution image; a decoder
//! network reads the bits back from a 256 x 256 copy. Because the mark is spread over
//! the whole frame as low-frequency structure it survives very heavy recompression,
//! downscaling, blur and tone changes — but not cropping, which is what the tile layer
//! in [`crate::detect`] is for.
//!
//! The bits use this crate's own coding (see [`crate::ncode`]), not TrustMark's BCH
//! schema, so they are tied to the watermark key.

use std::fmt;
use std::path::Path;
use std::sync::Mutex;

use image::imageops::{self, FilterType};
use image::{GenericImageView, RgbaImage};
use ndarray::{Array2, Array4};
use ort::session::Session;
use ort::value::Value;
use rayon::prelude::*;

use crate::codec::PAYLOAD_BYTES;
use crate::ncode::{self, MODEL_BITS};
use crate::pattern::WatermarkKey;

pub const ENCODER_FILE: &str = "encoder_Q.onnx";
pub const DECODER_FILE: &str = "decoder_Q.onnx";
/// ONNX Runtime library that must sit next to the models on builds that load it at
/// run time (every toolchain except MSVC, which links it statically).
pub const RUNTIME_FILE: &str = "onnxruntime.dll";
pub const RUNTIME_IS_BUNDLED: bool = cfg!(target_env = "msvc");

/// Side of the square the networks work on.
const SIDE: u32 = 256;
/// TrustMark's default residual scale and clamp (in `-1..1` pixel units).
const BASE_STRENGTH: f32 = 0.95;
const RESIDUAL_LIMIT: f32 = 0.2;
/// Beyond this aspect ratio only the centre square is marked, as in TrustMark.
const MAX_ASPECT: u32 = 2;
/// Images smaller than this on either side are left to the tile layer.
const MIN_SIDE: u32 = 32;

#[derive(Debug)]
pub enum NeuralError {
    /// The ONNX Runtime library or a model file could not be loaded.
    Load(String),
    /// Running a network failed.
    Inference(String),
}

impl fmt::Display for NeuralError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NeuralError::Load(message) => write!(f, "cannot load the neural models: {message}"),
            NeuralError::Inference(message) => write!(f, "neural inference failed: {message}"),
        }
    }
}

impl std::error::Error for NeuralError {}

fn inference(error: impl fmt::Display) -> NeuralError {
    NeuralError::Inference(error.to_string())
}

/// What the neural layer read from an image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeuralDetection {
    pub payload: [u8; PAYLOAD_BYTES],
    /// Raw bits (of 99) that had to be corrected.
    pub corrected: u32,
    /// Clockwise quarter turns the image had been rotated by.
    pub quarter_turns: u8,
}

/// The loaded encoder and decoder networks. Loading takes a moment; share one instance.
pub struct NeuralEngine {
    encoder: Mutex<Session>,
    decoder: Mutex<Session>,
}

impl NeuralEngine {
    /// Load the networks from `dir`, which must hold [`ENCODER_FILE`] and
    /// [`DECODER_FILE`] (and [`RUNTIME_FILE`] unless [`RUNTIME_IS_BUNDLED`]).
    pub fn load(dir: &Path) -> Result<Self, NeuralError> {
        let load = |error: ort::Error| NeuralError::Load(error.to_string());
        #[cfg(not(target_env = "msvc"))]
        {
            use std::sync::OnceLock;
            static RUNTIME: OnceLock<Result<(), String>> = OnceLock::new();
            RUNTIME
                .get_or_init(|| {
                    let library = dir.join(RUNTIME_FILE);
                    if !library.is_file() {
                        return Err(format!("{} is missing", library.display()));
                    }
                    ort::init_from(library.to_string_lossy().into_owned())
                        .commit()
                        .map(|_| ())
                        .map_err(|e| e.to_string())
                })
                .clone()
                .map_err(NeuralError::Load)?;
        }
        let session = |file: &str| -> Result<Mutex<Session>, NeuralError> {
            let path = dir.join(file);
            if !path.is_file() {
                return Err(NeuralError::Load(format!("{} is missing", path.display())));
            }
            Ok(Mutex::new(
                Session::builder()
                    .map_err(load)?
                    .commit_from_file(path)
                    .map_err(load)?,
            ))
        };
        Ok(Self {
            encoder: session(ENCODER_FILE)?,
            decoder: session(DECODER_FILE)?,
        })
    }

    /// Add the neural mark for `payload` to `img`. Returns false, leaving the image
    /// untouched, when it is too small to carry one.
    pub fn embed(
        &self,
        img: &mut RgbaImage,
        key: &WatermarkKey,
        payload: &[u8; PAYLOAD_BYTES],
        strength: f32,
    ) -> Result<bool, NeuralError> {
        let Some(region) = marked_region(img) else {
            return Ok(false);
        };
        let input = model_input(&model_view(img, region));
        let coded = ncode::encode(payload, key.neural_mask);
        let bits = Array2::from_shape_fn((1, MODEL_BITS), |(_, i)| (coded >> i & 1) as f32);

        let residual: Vec<f32> = {
            let mut encoder = self.encoder.lock().map_err(inference)?;
            let outputs = encoder
                .run(ort::inputs![
                    "onnx::Concat_0" => Value::from_array(input.clone()).map_err(inference)?,
                    "onnx::Gemm_1" => Value::from_array(bits).map_err(inference)?,
                ])
                .map_err(inference)?;
            let marked = outputs["image"]
                .try_extract_array::<f32>()
                .map_err(inference)?;
            if marked.len() != input.len() {
                return Err(inference("unexpected encoder output shape"));
            }
            marked
                .iter()
                .zip(input.iter())
                .map(|(marked, original)| {
                    (marked - original).clamp(-RESIDUAL_LIMIT, RESIDUAL_LIMIT)
                })
                .collect()
        };

        add_residual(img, region, &residual, BASE_STRENGTH * strength.max(0.0));
        Ok(true)
    }

    /// Look for a neural mark made with `key`, in all four rotations.
    pub fn detect(
        &self,
        img: &RgbaImage,
        key: &WatermarkKey,
    ) -> Result<Option<NeuralDetection>, NeuralError> {
        let Some(region) = marked_region(img) else {
            return Ok(None);
        };
        let mut view = model_view(img, region);
        let mut best: Option<NeuralDetection> = None;
        // Each pass undoes one more counter-clockwise turn, i.e. tests whether the
        // image had been turned clockwise that many times.
        for quarter_turns in 0..4u8 {
            if quarter_turns > 0 {
                view = imageops::rotate270(&view);
            }
            let logits: Vec<f32> = {
                let mut decoder = self.decoder.lock().map_err(inference)?;
                let outputs = decoder
                    .run(ort::inputs![
                        "image" => Value::from_array(model_input(&view)).map_err(inference)?,
                    ])
                    .map_err(inference)?;
                outputs["output"]
                    .try_extract_array::<f32>()
                    .map_err(inference)?
                    .iter()
                    .copied()
                    .collect()
            };
            if let Some(decoded) = ncode::decode(&logits, key.neural_mask) {
                let found = NeuralDetection {
                    payload: decoded.payload,
                    corrected: decoded.corrected,
                    quarter_turns,
                };
                if decoded.corrected == 0 {
                    return Ok(Some(found));
                }
                if best.is_none_or(|b| found.corrected < b.corrected) {
                    best = Some(found);
                }
            }
        }
        Ok(best)
    }
}

/// `(x, y, width, height)` of the part of the image the networks see: everything, or
/// the centre square of a very elongated image.
fn marked_region(img: &RgbaImage) -> Option<(u32, u32, u32, u32)> {
    let (w, h) = img.dimensions();
    if w < MIN_SIDE || h < MIN_SIDE {
        None
    } else if w > h * MAX_ASPECT {
        Some(((w - h) / 2, 0, h, h))
    } else if h > w * MAX_ASPECT {
        Some((0, (h - w) / 2, w, w))
    } else {
        Some((0, 0, w, h))
    }
}

fn model_view(img: &RgbaImage, (x, y, w, h): (u32, u32, u32, u32)) -> RgbaImage {
    imageops::resize(&*img.view(x, y, w, h), SIDE, SIDE, FilterType::Triangle)
}

/// `[1, 3, SIDE, SIDE]` tensor with channels in `-1..1`.
fn model_input(view: &RgbaImage) -> Array4<f32> {
    Array4::from_shape_fn(
        (1, 3, SIDE as usize, SIDE as usize),
        |(_, channel, y, x)| view.get_pixel(x as u32, y as u32)[channel] as f32 / 127.5 - 1.0,
    )
}

/// Scale the `3 x SIDE x SIDE` residual up to `region` with bilinear interpolation and
/// add it to the image. When the region is not the whole image its edges are feathered
/// so no seam shows.
fn add_residual(
    img: &mut RgbaImage,
    (rx, ry, rw, rh): (u32, u32, u32, u32),
    residual: &[f32],
    strength: f32,
) {
    let side = SIDE as usize;
    let plane = side * side;
    let width = img.width() as usize;
    let whole = (rw, rh) == img.dimensions();
    let feather = (rw.min(rh) as f32 * 0.03).max(1.0);
    let (rx, ry, rw, rh) = (rx as usize, ry as usize, rw as usize, rh as usize);
    // Source coordinate and interpolation weight for every destination column / row.
    let taps = |length: usize| -> Vec<(usize, usize, f32)> {
        (0..length)
            .map(|i| {
                let pos = ((i as f32 + 0.5) * side as f32 / length as f32 - 0.5)
                    .clamp(0.0, side as f32 - 1.0);
                let low = pos.floor() as usize;
                (low, (low + 1).min(side - 1), pos - low as f32)
            })
            .collect()
    };
    let (columns, rows) = (taps(rw), taps(rh));
    let edge_gain = |i: usize, length: usize| -> f32 {
        if whole {
            1.0
        } else {
            ((i.min(length - 1 - i) as f32 + 1.0) / feather).min(1.0)
        }
    };

    img.par_chunks_exact_mut(width * 4)
        .enumerate()
        .skip(ry)
        .take(rh)
        .for_each(|(y, row)| {
            let (y0, y1, fy) = rows[y - ry];
            let gain_y = strength * 127.5 * edge_gain(y - ry, rh);
            for (i, &(x0, x1, fx)) in columns.iter().enumerate() {
                let gain = gain_y * edge_gain(i, rw);
                let pixel = &mut row[(rx + i) * 4..(rx + i) * 4 + 3];
                for (channel, value) in pixel.iter_mut().enumerate() {
                    let at = |y: usize, x: usize| residual[channel * plane + y * side + x];
                    let top = at(y0, x0) + (at(y0, x1) - at(y0, x0)) * fx;
                    let bottom = at(y1, x0) + (at(y1, x1) - at(y1, x0)) * fx;
                    let delta = (top + (bottom - top) * fy) * gain;
                    *value = (*value as f32 + delta).round().clamp(0.0, 255.0) as u8;
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_is_whole_image_or_centre_square() {
        let region = |w, h| marked_region(&RgbaImage::new(w, h));
        assert_eq!(region(400, 300), Some((0, 0, 400, 300)));
        assert_eq!(region(1000, 300), Some((350, 0, 300, 300)));
        assert_eq!(region(100, 900), Some((0, 400, 100, 100)));
        assert_eq!(region(20, 300), None);
    }

    #[test]
    fn residual_is_scaled_and_leaves_alpha() {
        let mut img = RgbaImage::from_pixel(512, 384, image::Rgba([100, 100, 100, 77]));
        let residual = vec![0.1f32; 3 * 256 * 256];
        add_residual(&mut img, (0, 0, 512, 384), &residual, 1.0);
        assert!(img.pixels().all(|p| p.0 == [113, 113, 113, 77]));
    }

    #[test]
    fn partial_region_is_feathered_and_bounded() {
        let mut img = RgbaImage::from_pixel(900, 300, image::Rgba([100, 100, 100, 255]));
        let residual = vec![0.1f32; 3 * 256 * 256];
        add_residual(&mut img, (300, 0, 300, 300), &residual, 1.0);
        assert_eq!(img.get_pixel(299, 150)[0], 100);
        assert_eq!(img.get_pixel(600, 150)[0], 100);
        assert_eq!(img.get_pixel(450, 150)[0], 113);
        let edge = img.get_pixel(300, 150)[0];
        assert!(edge > 100 && edge < 113, "edge {edge}");
    }
}
