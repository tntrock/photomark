//! Invisible image watermarking that survives heavy cropping, rescaling, screenshots
//! and lossy recompression.
//!
//! A key-derived tile of faint luma changes is repeated across the whole image, so any
//! crop that still contains a couple of tiles carries the full payload. See [`detect`]
//! for how the tile grid is found again without the original image.

mod codec;
mod detect;
mod embed;
mod fft2;
mod filters;
pub mod io;
mod pattern;
mod prng;

pub use codec::PAYLOAD_BYTES;
pub use detect::{detect, DetectOptions, Detection};
pub use embed::{auto_cell, embed, EmbedOptions, EmbedReport, MAX_CELL, MIN_CELL};
pub use pattern::{payload_from_text, payload_to_text, WatermarkKey, DEFAULT_KEY};
