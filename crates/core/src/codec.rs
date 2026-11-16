//! Payload channel coding: CRC-16 for integrity plus a rate-1/2, constraint-length-7
//! convolutional code decoded with a soft-decision Viterbi decoder.

/// Bytes of user payload carried by the watermark.
pub const PAYLOAD_BYTES: usize = 8;

const CRC_BITS: usize = 16;
const INFO_BITS: usize = PAYLOAD_BYTES * 8 + CRC_BITS;
const CONSTRAINT: usize = 7;
const TAIL_BITS: usize = CONSTRAINT - 1;
const STEPS: usize = INFO_BITS + TAIL_BITS;
const STATES: usize = 1 << TAIL_BITS;

/// Number of coded bits embedded in each tile.
pub const CODED_BITS: usize = STEPS * 2;

/// Generator polynomials (octal 171, 133), the standard K=7 pair.
const POLYS: [u32; 2] = [0o171, 0o133];

fn crc16(data: &[u8]) -> u16 {
    let mut crc: u16 = 0xFFFF;
    for &byte in data {
        crc ^= (byte as u16) << 8;
        for _ in 0..8 {
            crc = if crc & 0x8000 != 0 {
                (crc << 1) ^ 0x1021
            } else {
                crc << 1
            };
        }
    }
    crc
}

fn parity(v: u32) -> u8 {
    (v.count_ones() & 1) as u8
}

/// Output bit pair for `input` entering the encoder in `state` (the previous six inputs,
/// most recent in the high bit).
fn branch_output(state: usize, input: usize) -> [u8; 2] {
    let reg = ((input << TAIL_BITS) | state) as u32;
    [parity(reg & POLYS[0]), parity(reg & POLYS[1])]
}

fn info_bits(payload: &[u8; PAYLOAD_BYTES]) -> [u8; INFO_BITS] {
    let mut bits = [0u8; INFO_BITS];
    let crc = crc16(payload).to_be_bytes();
    for (i, byte) in payload.iter().chain(crc.iter()).enumerate() {
        for b in 0..8 {
            bits[i * 8 + b] = (byte >> (7 - b)) & 1;
        }
    }
    bits
}

/// Encode a payload into the coded bits (each 0 or 1) of one tile.
pub fn encode(payload: &[u8; PAYLOAD_BYTES]) -> [u8; CODED_BITS] {
    let info = info_bits(payload);
    let mut out = [0u8; CODED_BITS];
    let mut state = 0usize;
    for step in 0..STEPS {
        let input = if step < INFO_BITS {
            info[step] as usize
        } else {
            0
        };
        let pair = branch_output(state, input);
        out[step * 2] = pair[0];
        out[step * 2 + 1] = pair[1];
        state = ((input << TAIL_BITS) | state) >> 1;
    }
    out
}

/// Decode soft values (positive means the coded bit is more likely 1).
///
/// Returns the payload only if its CRC verifies.
pub fn decode(soft: &[f32; CODED_BITS]) -> Option<[u8; PAYLOAD_BYTES]> {
    const UNREACHABLE: f32 = -1.0e30;

    let mut metric = [UNREACHABLE; STATES];
    metric[0] = 0.0;
    // choices[step][next_state] is the low bit of the surviving predecessor state.
    let mut choices = vec![[0u8; STATES]; STEPS];

    for step in 0..STEPS {
        let (s0, s1) = (soft[step * 2], soft[step * 2 + 1]);
        let mut next = [UNREACHABLE; STATES];
        // The tail is known to be zeros, which halves the reachable states there.
        let inputs = if step < INFO_BITS { 2 } else { 1 };
        for input in 0..inputs {
            for low in 0..STATES / 2 {
                let next_state = (input << (TAIL_BITS - 1)) | low;
                for bit in 0..2 {
                    let state = (low << 1) | bit;
                    if metric[state] <= UNREACHABLE {
                        continue;
                    }
                    let out = branch_output(state, input);
                    let gain =
                        if out[0] == 1 { s0 } else { -s0 } + if out[1] == 1 { s1 } else { -s1 };
                    let candidate = metric[state] + gain;
                    if candidate > next[next_state] {
                        next[next_state] = candidate;
                        choices[step][next_state] = bit as u8;
                    }
                }
            }
        }
        metric = next;
    }

    let mut bits = [0u8; STEPS];
    let mut state = 0usize;
    for step in (0..STEPS).rev() {
        bits[step] = (state >> (TAIL_BITS - 1)) as u8;
        let low = state & (STATES / 2 - 1);
        state = (low << 1) | choices[step][state] as usize;
    }

    let mut bytes = [0u8; PAYLOAD_BYTES + 2];
    for (i, byte) in bytes.iter_mut().enumerate() {
        for b in 0..8 {
            *byte = (*byte << 1) | bits[i * 8 + b];
        }
    }
    let mut payload = [0u8; PAYLOAD_BYTES];
    payload.copy_from_slice(&bytes[..PAYLOAD_BYTES]);
    let crc = u16::from_be_bytes([bytes[PAYLOAD_BYTES], bytes[PAYLOAD_BYTES + 1]]);
    (crc16(&payload) == crc).then_some(payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prng::Rng;

    fn to_soft(coded: &[u8; CODED_BITS]) -> [f32; CODED_BITS] {
        let mut soft = [0f32; CODED_BITS];
        for (s, &c) in soft.iter_mut().zip(coded) {
            *s = if c == 1 { 1.0 } else { -1.0 };
        }
        soft
    }

    #[test]
    fn crc_matches_reference() {
        // CRC-16/CCITT-FALSE check value.
        assert_eq!(crc16(b"123456789"), 0x29B1);
    }

    #[test]
    fn round_trip_clean() {
        for payload in [*b"ALLEN-26", [0u8; 8], [0xFF; 8]] {
            assert_eq!(decode(&to_soft(&encode(&payload))), Some(payload));
        }
    }

    #[test]
    fn corrects_bit_errors() {
        let payload = *b"photomrk";
        let mut soft = to_soft(&encode(&payload));
        // Flip roughly one coded bit in twelve, spread out.
        let mut rng = Rng::from_key(b"test", "flips");
        for chunk in soft.chunks_mut(12) {
            let i = rng.below(chunk.len() as u32) as usize;
            chunk[i] = -chunk[i];
        }
        assert_eq!(decode(&soft), Some(payload));
    }

    #[test]
    fn rejects_noise() {
        let mut rng = Rng::from_key(b"test", "noise");
        let mut accepted = 0;
        for _ in 0..200 {
            let mut soft = [0f32; CODED_BITS];
            for s in soft.iter_mut() {
                *s = rng.below(2001) as f32 / 1000.0 - 1.0;
            }
            accepted += decode(&soft).is_some() as u32;
        }
        assert_eq!(accepted, 0);
    }
}
