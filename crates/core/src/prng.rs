//! Deterministic key-derived random numbers.
//!
//! The watermark pattern must be reproducible forever from the secret key alone, so the
//! generator is implemented here instead of depending on an external crate whose output
//! could change between versions.

use sha2::{Digest, Sha256};

/// xoshiro256** seeded from SHA-256(key, domain).
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn from_key(key: &[u8], domain: &str) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"photomark/v1/");
        hasher.update(domain.as_bytes());
        hasher.update([0u8]);
        hasher.update(key);
        let digest = hasher.finalize();

        let mut s = [0u64; 4];
        for (i, word) in s.iter_mut().enumerate() {
            *word = u64::from_le_bytes(digest[i * 8..i * 8 + 8].try_into().unwrap());
        }
        if s == [0; 4] {
            s[0] = 1;
        }
        Self { s }
    }

    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform integer in `0..n`.
    pub fn below(&mut self, n: u32) -> u32 {
        (((self.next_u64() >> 32) * n as u64) >> 32) as u32
    }

    /// Uniform `+1` or `-1`.
    pub fn sign(&mut self) -> i8 {
        if self.next_u64() >> 63 == 0 {
            1
        } else {
            -1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_key_dependent() {
        let a: Vec<u64> = {
            let mut r = Rng::from_key(b"key", "d");
            (0..8).map(|_| r.next_u64()).collect()
        };
        let b: Vec<u64> = {
            let mut r = Rng::from_key(b"key", "d");
            (0..8).map(|_| r.next_u64()).collect()
        };
        let c: Vec<u64> = {
            let mut r = Rng::from_key(b"other", "d");
            (0..8).map(|_| r.next_u64()).collect()
        };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn below_stays_in_range() {
        let mut r = Rng::from_key(b"k", "range");
        assert!((0..10_000).all(|_| r.below(7) < 7));
    }
}
