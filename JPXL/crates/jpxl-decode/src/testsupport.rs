//! Bit-writing helpers for building hand-made header fixtures in tests.
//!
//! Every test fixture in this crate is derived by hand from a spec table, so
//! the writer mirrors the B.2 field types exactly: `u(n)`, `Bool()`, `U32()`
//! and `U64()`. Encoding a fixture with the same vocabulary the table uses is
//! what makes it reviewable against the standard — a fixture written as raw
//! hex bytes proves nothing to a reader holding the spec.
//!
//! This is not an encoder. It performs no validation and makes no choices; a
//! `U32()` value is written with the selector the caller names.

/// Accumulates bits LSB-first within each byte, matching 18181-1 B.2.1.
#[derive(Debug, Clone, Default)]
pub struct BitWriter {
    bytes: Vec<u8>,
    bit_len: u64,
}

impl BitWriter {
    /// Creates an empty writer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Writes `u(n)`: the low `n` bits of `value`, least-significant first.
    pub fn u(&mut self, n: u32, value: u32) -> &mut Self {
        assert!(n <= 32, "u(n) has n <= 32");
        for i in 0..n {
            self.bit((value >> i) & 1 == 1);
        }
        self
    }

    /// Writes `Bool()`.
    pub fn bool(&mut self, value: bool) -> &mut Self {
        self.bit(value)
    }

    /// Writes a `U32()` field: the 2-bit `selector`, then `payload_bits` of
    /// `payload`. Pass `payload_bits == 0` for a constant distribution.
    pub fn u32_field(&mut self, selector: u32, payload_bits: u32, payload: u32) -> &mut Self {
        assert!(selector < 4, "the U32() selector is u(2)");
        self.u(2, selector);
        self.u(payload_bits, payload)
    }

    /// Writes a `U64()` field holding `value`, choosing the shortest of the
    /// three fixed distributions; values above `17 + 255` use the `s == 3`
    /// escape (18181-1 B.2.3).
    pub fn u64_field(&mut self, value: u64) -> &mut Self {
        if value == 0 {
            return self.u(2, 0);
        }
        if (1..=16).contains(&value) {
            self.u(2, 1);
            return self.u(4, u32::try_from(value - 1).expect("value <= 16"));
        }
        if (17..=272).contains(&value) {
            self.u(2, 2);
            return self.u(8, u32::try_from(value - 17).expect("value <= 272"));
        }

        self.u(2, 3);
        self.u(12, u32::try_from(value & 0xFFF).expect("12 bits"));
        let mut remaining = value >> 12;
        let mut shift = 12u32;
        while remaining != 0 {
            self.u(1, 1);
            if shift == 60 {
                self.u(4, u32::try_from(remaining & 0xF).expect("4 bits"));
                return self;
            }
            self.u(8, u32::try_from(remaining & 0xFF).expect("8 bits"));
            remaining >>= 8;
            shift += 8;
        }
        self.u(1, 0)
    }

    /// Writes an `Enum(EnumTable)` value using 18181-1 B.2.6's
    /// `U32(0, 1, 2 + u(4), 18 + u(6))`, picking the shortest encoding.
    pub fn enum_field(&mut self, value: u32) -> &mut Self {
        match value {
            0 => self.u32_field(0, 0, 0),
            1 => self.u32_field(1, 0, 0),
            2..=17 => self.u32_field(2, 4, value - 2),
            _ => self.u32_field(3, 6, value - 18),
        }
    }

    /// Writes a raw `F16()` given its 16 encoded bits.
    pub fn f16_bits(&mut self, bits: u16) -> &mut Self {
        self.u(16, u32::from(bits))
    }

    /// Writes one bit.
    pub fn bit(&mut self, set: bool) -> &mut Self {
        if self.bit_len.is_multiple_of(8) {
            self.bytes.push(0);
        }
        if set && let Some(byte) = self.bytes.last_mut() {
            *byte |= 1u8 << (self.bit_len % 8);
        }
        self.bit_len += 1;
        self
    }

    /// Writes `ZeroPadToByte()` (18181-1 B.2.7): zero bits up to the next
    /// byte boundary. A no-op when already aligned.
    pub fn pad_to_byte(&mut self) -> &mut Self {
        while !self.bit_len.is_multiple_of(8) {
            self.bit(false);
        }
        self
    }

    /// Number of bits written so far.
    ///
    /// Returns `u64` to match `BitReader::total_bits_read`, so tests can
    /// compare the two without a cast.
    #[must_use]
    pub const fn bit_len(&self) -> u64 {
        self.bit_len
    }

    /// The written bytes, zero-padded to a byte boundary.
    #[must_use]
    pub fn finish(&self) -> Vec<u8> {
        self.bytes.clone()
    }

    /// The written bytes plus `extra` zero bytes, so a fixture that ends
    /// mid-field does not trip an unrelated out-of-bounds error.
    #[must_use]
    pub fn finish_padded(&self, extra: usize) -> Vec<u8> {
        let mut out = self.bytes.clone();
        out.resize(out.len() + extra, 0);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jpxl_bitstream::{BitReader, read_u64};

    #[test]
    fn writes_lsb_first() {
        let mut w = BitWriter::new();
        w.u(1, 1).u(7, 0);
        assert_eq!(w.finish(), vec![0b0000_0001]);
    }

    #[test]
    fn spans_byte_boundaries() {
        let mut w = BitWriter::new();
        w.u(4, 0b1111).u(8, 0);
        assert_eq!(w.finish(), vec![0b0000_1111, 0b0000_0000]);
        assert_eq!(w.bit_len(), 12);
    }

    #[test]
    fn u64_round_trips_through_the_reader() {
        for value in [
            0u64,
            1,
            16,
            17,
            272,
            273,
            4095,
            4096,
            1 << 20,
            u64::from(u32::MAX),
        ] {
            let mut w = BitWriter::new();
            w.u64_field(value);
            let data = w.finish_padded(2);
            let mut r = BitReader::new(&data);
            assert_eq!(
                read_u64(&mut r).expect("valid U64()"),
                value,
                "value {value}"
            );
        }
    }
}
