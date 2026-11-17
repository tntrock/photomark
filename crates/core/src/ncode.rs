//! Channel code for the neural watermark.
//!
//! The network carries [`MODEL_BITS`] raw bits. The payload is protected by a shortened
//! BCH(127, 92) code — 64 payload bits followed by 35 parity bits — and the codeword is
//! whitened with a key-derived mask, so a watermark made with another key (or by another
//! TrustMark tool) does not decode. The remaining model bit carries no information.
//!
//! Decoding is soft: when the hard decisions are not a codeword, the least reliable bits
//! are flipped in small groups until one is found. Trying only a few thousand of the
//! 2^35 possible syndromes keeps the false-positive rate of an unmarked image around
//! one in ten million.

use std::sync::OnceLock;

use crate::codec::PAYLOAD_BYTES;

/// Bits the network embeds.
pub const MODEL_BITS: usize = 100;

const DATA_BITS: usize = PAYLOAD_BYTES * 8;
const PARITY_BITS: usize = 35;
const CODE_BITS: usize = DATA_BITS + PARITY_BITS;

/// Least reliable bits considered for flipping, and the most that are flipped at once.
const CHASE_BITS: usize = 14;
const CHASE_FLIPS: usize = 4;

/// Generator polynomial of the BCH code: the product of the minimal polynomials of
/// `a^1, a^3, a^5, a^7, a^9` over GF(2^7), where `a` is a root of `x^7 + x^3 + 1`.
fn generator() -> u64 {
    static GENERATOR: OnceLock<u64> = OnceLock::new();
    *GENERATOR.get_or_init(|| {
        // Powers of the primitive element.
        let mut exp = [0u8; 127];
        let mut value = 1u8;
        for slot in &mut exp {
            *slot = value;
            value <<= 1;
            if value & 0x80 != 0 {
                value ^= 0x89;
            }
        }
        let mul = |a: u8, b: u8| -> u8 {
            let mut product = 0u8;
            let mut a = a;
            for bit in 0..7 {
                if b >> bit & 1 == 1 {
                    product ^= a;
                }
                a <<= 1;
                if a & 0x80 != 0 {
                    a ^= 0x89;
                }
            }
            product
        };

        let mut generator = 1u64;
        for root in [1usize, 3, 5, 7, 9] {
            // Minimal polynomial: the product of (x + r) over the conjugates of a^root.
            // Its coefficients end up in GF(2); `poly[i]` is the coefficient of x^i.
            let mut poly = vec![1u8];
            let mut power = root;
            loop {
                let r = exp[power];
                let mut next = vec![0u8; poly.len() + 1];
                for (i, &c) in poly.iter().enumerate() {
                    next[i + 1] ^= c;
                    next[i] ^= mul(c, r);
                }
                poly = next;
                power = power * 2 % 127;
                if power == root {
                    break;
                }
            }
            let minimal = poly
                .iter()
                .enumerate()
                .fold(0u64, |acc, (i, &c)| acc | (c as u64) << i);
            // Carry-less multiply into the generator.
            let mut product = 0u64;
            for bit in 0..64 {
                if minimal >> bit & 1 == 1 {
                    product ^= generator << bit;
                }
            }
            generator = product;
        }
        debug_assert_eq!(63 - generator.leading_zeros() as usize, PARITY_BITS);
        generator
    })
}

/// Remainder of the polynomial `value` (bit `i` is the coefficient of `x^i`) modulo the
/// generator.
fn remainder(mut value: u128) -> u64 {
    let generator = generator() as u128;
    for bit in (PARITY_BITS..CODE_BITS).rev() {
        if value >> bit & 1 == 1 {
            value ^= generator << (bit - PARITY_BITS);
        }
    }
    value as u64
}

/// The bits to embed for `payload`: bit `i` of the result is model bit `i`.
pub fn encode(payload: &[u8; PAYLOAD_BYTES], mask: u128) -> u128 {
    let data = (u64::from_be_bytes(*payload) as u128) << PARITY_BITS;
    (data | remainder(data) as u128) ^ mask
}

/// A decoded neural payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Decoded {
    pub payload: [u8; PAYLOAD_BYTES],
    /// Raw bits that had to be corrected.
    pub corrected: u32,
}

/// Recover the payload from the network's `logits` (positive means 1).
pub fn decode(logits: &[f32], mask: u128) -> Option<Decoded> {
    if logits.len() < CODE_BITS {
        return None;
    }
    let hard = (0..CODE_BITS).fold(0u128, |acc, i| acc | ((logits[i] > 0.0) as u128) << i)
        ^ (mask & ((1u128 << CODE_BITS) - 1));

    let mut weakest: Vec<usize> = (0..CODE_BITS).collect();
    weakest.sort_by(|&a, &b| logits[a].abs().total_cmp(&logits[b].abs()));
    weakest.truncate(CHASE_BITS);
    let flip_syndromes: Vec<u64> = weakest.iter().map(|&i| remainder(1u128 << i)).collect();
    let syndrome = remainder(hard);

    // Subsets of the weakest bits in order of size, so the fewest flips win.
    for flips in 0..=CHASE_FLIPS {
        for subset in 0u32..1 << CHASE_BITS {
            if subset.count_ones() as usize != flips {
                continue;
            }
            let mut s = syndrome;
            let mut word = hard;
            for (n, &bit) in weakest.iter().enumerate() {
                if subset >> n & 1 == 1 {
                    s ^= flip_syndromes[n];
                    word ^= 1u128 << bit;
                }
            }
            if s == 0 {
                return Some(Decoded {
                    payload: ((word >> PARITY_BITS) as u64).to_be_bytes(),
                    corrected: flips as u32,
                });
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prng::Rng;

    fn logits_for(bits: u128) -> Vec<f32> {
        (0..MODEL_BITS)
            .map(|i| if bits >> i & 1 == 1 { 8.0 } else { -8.0 })
            .collect()
    }

    #[test]
    fn generator_has_expected_degree_and_divides_x127_plus_1() {
        let g = generator();
        assert_eq!(63 - g.leading_zeros(), 35);
        // A cyclic code's generator divides x^n + 1.
        let mut value = (1u128 << 127) | 1;
        for bit in (35..128).rev() {
            if value >> bit & 1 == 1 {
                value ^= (g as u128) << (bit - 35);
            }
        }
        assert_eq!(value, 0);
    }

    #[test]
    fn round_trip() {
        let mask = 0x1234_5678_9abc_def0_0fed_cba9_8765_4321u128;
        let payload = *b"ALLEN-26";
        let decoded = decode(&logits_for(encode(&payload, mask)), mask).unwrap();
        assert_eq!(decoded.payload, payload);
        assert_eq!(decoded.corrected, 0);
    }

    #[test]
    fn corrects_weak_wrong_bits() {
        let mask = 77u128 << 40;
        let payload = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut logits = logits_for(encode(&payload, mask));
        for i in [3, 40, 71, 98] {
            logits[i] = -logits[i] * 0.1;
        }
        let decoded = decode(&logits, mask).unwrap();
        assert_eq!(decoded.payload, payload);
        assert_eq!(decoded.corrected, 4);
    }

    #[test]
    fn wrong_mask_and_noise_do_not_decode() {
        let payload = *b"abcdefgh";
        assert_eq!(decode(&logits_for(encode(&payload, 5)), 0xffff << 50), None);

        let mut rng = Rng::from_key(b"noise", "ncode-test");
        for _ in 0..200 {
            let logits: Vec<f32> = (0..MODEL_BITS)
                .map(|_| (rng.next_u64() >> 40) as f32 / (1u64 << 23) as f32 - 1.0)
                .collect();
            assert_eq!(decode(&logits, 0), None);
        }
    }
}
