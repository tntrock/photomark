//! Invisible image watermarking that survives heavy cropping, rescaling, screenshots
//! and lossy recompression.
//!
//! Two independent layers carry the same payload:
//!
//! * a *tile* layer: a key-derived tile of faint luma changes repeated across the whole
//!   image, so any crop that still contains a couple of tiles carries the full payload
//!   (see [`detect`] for how the tile grid is found again without the original image);
//! * a *neural* layer ([`NeuralEngine`]): Adobe TrustMark networks that spread the
//!   payload over the whole frame, which withstands far harsher recompression and
//!   resizing but not cropping.
//!
//! [`embed_layers`] and [`detect_layers`] apply both.

mod codec;
mod detect;
mod embed;
mod fft2;
mod filters;
pub mod io;
mod layers;
pub mod models;
mod ncode;
mod neural;
mod pattern;
mod prng;

pub use codec::PAYLOAD_BYTES;
pub use detect::{detect, DetectOptions, Detection};
pub use embed::{auto_cell, embed, EmbedOptions, EmbedReport, MAX_CELL, MIN_CELL};
pub use layers::{detect_layers, embed_layers, LayeredDetection, LayeredReport};
pub use neural::{NeuralDetection, NeuralEngine, NeuralError};
pub use pattern::{payload_from_text, payload_to_text, WatermarkKey, DEFAULT_KEY};
