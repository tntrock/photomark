//! Embedding and detecting both watermark layers together.
//!
//! The neural layer withstands harsh processing of the whole picture; the tile layer
//! withstands cropping. They are added on top of each other and read independently.

use image::RgbaImage;

use crate::codec::PAYLOAD_BYTES;
use crate::detect::{detect, DetectOptions, Detection};
use crate::embed::{embed, EmbedOptions};
use crate::neural::{NeuralDetection, NeuralEngine, NeuralError};
use crate::pattern::WatermarkKey;

#[derive(Clone, Copy, Debug)]
pub struct LayeredReport {
    /// Cell size of the tile layer, in pixels.
    pub cell: u32,
    /// Peak signal-to-noise ratio of the result against the original, in dB.
    pub psnr: f32,
    /// Whether the neural layer was added.
    pub neural: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LayeredDetection {
    /// Whether either layer found a watermark made with this key.
    pub detected: bool,
    /// The embedded identifier, if either layer could read it.
    pub payload: Option<[u8; PAYLOAD_BYTES]>,
    /// Result of the tile layer.
    pub tile: Detection,
    /// Result of the neural layer; `None` if it found nothing or was not run.
    pub neural: Option<NeuralDetection>,
}

/// Embed `payload` with the tile layer and, when `neural` is given, the neural layer.
pub fn embed_layers(
    img: &mut RgbaImage,
    key: &WatermarkKey,
    payload: &[u8; PAYLOAD_BYTES],
    options: &EmbedOptions,
    neural: Option<&NeuralEngine>,
) -> Result<LayeredReport, NeuralError> {
    let Some(engine) = neural else {
        let report = embed(img, key, payload, options);
        return Ok(LayeredReport {
            cell: report.cell,
            psnr: report.psnr,
            neural: false,
        });
    };

    let original = img.clone();
    // The tile layer goes on last so its fine grid is not smoothed by anything.
    let neural = engine.embed(img, key, payload, options.strength)?;
    let report = embed(img, key, payload, options);

    let squared_error: f64 = img
        .pixels()
        .zip(original.pixels())
        .flat_map(|(a, b)| (0..3).map(move |c| (a[c] as f64 - b[c] as f64).powi(2)))
        .sum();
    let mse = squared_error / (img.width() as f64 * img.height() as f64 * 3.0);
    Ok(LayeredReport {
        cell: report.cell,
        psnr: if mse > 0.0 {
            (10.0 * (255.0f64 * 255.0 / mse).log10()) as f32
        } else {
            f32::INFINITY
        },
        neural,
    })
}

/// Look for both layers. The neural layer is skipped when `neural` is `None`.
pub fn detect_layers(
    img: &RgbaImage,
    key: &WatermarkKey,
    options: &DetectOptions,
    neural: Option<&NeuralEngine>,
) -> Result<LayeredDetection, NeuralError> {
    let neural = match neural {
        Some(engine) => engine.detect(img, key)?,
        None => None,
    };
    let tile = detect(img, key, options);
    Ok(LayeredDetection {
        detected: tile.detected || neural.is_some(),
        // The neural payload passed a much longer check than the tile layer's CRC.
        payload: neural.map(|n| n.payload).or(tile.payload),
        tile,
        neural,
    })
}
