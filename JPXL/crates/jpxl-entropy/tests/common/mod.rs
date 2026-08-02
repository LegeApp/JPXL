//! Test helper: a bit writer that mirrors `jpxl_bitstream::BitReader`.
//!
//! Hand-derived vectors for Annex C are far easier to check as a sequence of
//! named field writes than as a packed byte array, so the tests build streams
//! with this and state the derivation in comments beside each write.

// Test code: an out-of-range index is a failed assertion rather than an attack
// surface, and the casts are on values these tests chose themselves. The lints
// exist for the decode paths in src/.
#![allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]
// Each integration test file uses a different subset of these helpers.
#![allow(dead_code)]

/// Accumulates bits in stream order and packs them LSB-first within each byte,
/// which is exactly what `BitReader` consumes.
#[derive(Debug, Default)]
pub struct BitWriter {
    bits: Vec<bool>,
}

impl BitWriter {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Writes `u(n)`: `n` bits of `value`, least-significant bit first.
    pub fn u(&mut self, value: u32, n: u32) {
        for i in 0..n {
            self.bits.push((value >> i) & 1 == 1);
        }
    }

    /// Writes a single `Bool()`.
    pub fn bit(&mut self, value: bool) {
        self.bits.push(value);
    }

    /// Writes the bits of a prefix code in the order the decoder reads them,
    /// i.e. most-significant bit of the canonical code value first.
    pub fn code(&mut self, bits: &[u8]) {
        for &b in bits {
            self.bits.push(b != 0);
        }
    }

    /// Number of bits written so far.
    #[must_use]
    pub fn bit_len(&self) -> usize {
        self.bits.len()
    }

    /// Packs to bytes, zero-padding the final partial byte.
    #[must_use]
    pub fn finish(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.bits.len().div_ceil(8)];
        for (i, &bit) in self.bits.iter().enumerate() {
            if bit {
                out[i / 8] |= 1 << (i % 8);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jpxl_bitstream::BitReader;

    #[test]
    fn writer_round_trips_through_the_reader() {
        let mut w = BitWriter::new();
        w.u(5, 3);
        w.bit(true);
        w.u(0xABCD, 16);
        w.code(&[1, 0, 1]);
        let data = w.finish();
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bits(3), Ok(5));
        assert_eq!(r.read_bool(), Ok(true));
        assert_eq!(r.read_bits(16), Ok(0xABCD));
        assert_eq!(r.read_bits(1), Ok(1));
        assert_eq!(r.read_bits(1), Ok(0));
        assert_eq!(r.read_bits(1), Ok(1));
    }
}
