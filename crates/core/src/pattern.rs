//! The key-derived watermark tile.
//!
//! A tile is a `CELLS` x `CELLS` grid of `+1`/`-1` cells that is repeated over the whole
//! image. Some cells form a fixed *pilot* pattern used to find the tile grid again after
//! cropping and rescaling; the rest carry the coded payload bits, each repeated in
//! several scattered cells.

use crate::codec::{self, CODED_BITS, PAYLOAD_BYTES};
use crate::prng::Rng;

/// Cells per tile side.
pub const CELLS: usize = 32;
/// Cells per tile.
pub const TILE_CELLS: usize = CELLS * CELLS;

const REPEATS: usize = 4;
const DATA_CELLS: usize = CODED_BITS * REPEATS;

/// Key used when the user does not supply one.
pub const DEFAULT_KEY: &str = "photomark";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Cell {
    /// Fixed synchronisation cell with the given sign.
    Pilot(i8),
    /// Carries coded bit `bit`, multiplied by the whitening sign `mask`.
    Data { bit: u16, mask: i8 },
}

/// The secret that determines where and how a watermark is embedded.
#[derive(Clone)]
pub struct WatermarkKey {
    pub(crate) cells: Vec<Cell>,
}

impl WatermarkKey {
    /// Derive the tile layout from a passphrase. An empty passphrase means [`DEFAULT_KEY`].
    pub fn new(passphrase: &str) -> Self {
        let passphrase = if passphrase.is_empty() {
            DEFAULT_KEY
        } else {
            passphrase
        };
        let mut rng = Rng::from_key(passphrase.as_bytes(), "tile-layout");

        let mut order: Vec<usize> = (0..TILE_CELLS).collect();
        for i in (1..TILE_CELLS).rev() {
            order.swap(i, rng.below(i as u32 + 1) as usize);
        }

        let mut cells = vec![Cell::Pilot(1); TILE_CELLS];
        for (n, &index) in order.iter().enumerate() {
            let sign = rng.sign();
            cells[index] = if n < DATA_CELLS {
                Cell::Data {
                    bit: (n % CODED_BITS) as u16,
                    mask: sign,
                }
            } else {
                Cell::Pilot(sign)
            };
        }
        Self { cells }
    }

    /// Cell signs of the tile that carries `payload`.
    pub(crate) fn tile(&self, payload: &[u8; PAYLOAD_BYTES]) -> Vec<i8> {
        let coded = codec::encode(payload);
        self.cells
            .iter()
            .map(|cell| match *cell {
                Cell::Pilot(sign) => sign,
                Cell::Data { bit, mask } => {
                    if coded[bit as usize] == 1 {
                        mask
                    } else {
                        -mask
                    }
                }
            })
            .collect()
    }
}

/// Pack a short text into a payload, padding with zero bytes.
///
/// Fails if the text is longer than [`PAYLOAD_BYTES`] bytes of UTF-8.
pub fn payload_from_text(text: &str) -> Option<[u8; PAYLOAD_BYTES]> {
    let bytes = text.as_bytes();
    if bytes.len() > PAYLOAD_BYTES {
        return None;
    }
    let mut payload = [0u8; PAYLOAD_BYTES];
    payload[..bytes.len()].copy_from_slice(bytes);
    Some(payload)
}

/// Inverse of [`payload_from_text`]; non-text payloads are shown as hex.
pub fn payload_to_text(payload: &[u8; PAYLOAD_BYTES]) -> String {
    let end = payload.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
    match std::str::from_utf8(&payload[..end]) {
        Ok(text) if !text.chars().any(char::is_control) => text.to_owned(),
        _ => payload.iter().map(|b| format!("{b:02x}")).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_has_expected_cell_counts() {
        let key = WatermarkKey::new("k");
        let pilots = key
            .cells
            .iter()
            .filter(|c| matches!(c, Cell::Pilot(_)))
            .count();
        assert_eq!(pilots, TILE_CELLS - DATA_CELLS);
        for bit in 0..CODED_BITS as u16 {
            let n = key
                .cells
                .iter()
                .filter(|c| matches!(c, Cell::Data { bit: b, .. } if *b == bit))
                .count();
            assert_eq!(n, REPEATS);
        }
    }

    #[test]
    fn keys_differ_and_empty_is_default() {
        assert!(WatermarkKey::new("a").cells != WatermarkKey::new("b").cells);
        assert!(WatermarkKey::new("").cells == WatermarkKey::new(DEFAULT_KEY).cells);
    }

    #[test]
    fn payload_text_round_trip() {
        for text in ["", "A", "ALLEN-26", "浮水"] {
            let payload = payload_from_text(text).unwrap();
            assert_eq!(payload_to_text(&payload), text);
        }
        assert!(payload_from_text("123456789").is_none());
        assert_eq!(payload_to_text(&[0xFF; 8]), "ffffffffffffffff");
    }
}
