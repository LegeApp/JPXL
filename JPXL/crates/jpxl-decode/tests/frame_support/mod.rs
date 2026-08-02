//! Bit-writing helper shared by the `frame_*` integration tests.
//!
//! The crate-internal `testsupport::BitWriter` is `#[cfg(test)]`-private, so
//! integration tests carry their own. The argument order here matches that one
//! — `u(bits, value)` — so fixtures read the same in both places.

#![allow(dead_code)]

/// Accumulates bits LSB-first within each byte, matching 18181-1 B.2.1.
#[derive(Debug, Clone, Default)]
pub struct BitWriter {
    bytes: Vec<u8>,
    bit_len: u64,
}

impl BitWriter {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
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

    /// Writes a `U32()` field: the 2-bit selector then `payload_bits` of
    /// `payload`.
    pub fn u32_field(&mut self, selector: u32, payload_bits: u32, payload: u32) -> &mut Self {
        assert!(selector < 4, "the U32() selector is u(2)");
        self.u(2, selector);
        self.u(payload_bits, payload)
    }

    /// Writes a `U64()` field holding `value` using the shortest distribution.
    pub fn u64_field(&mut self, value: u64) -> &mut Self {
        if value == 0 {
            return self.u(2, 0);
        }
        if (1..=16).contains(&value) {
            self.u(2, 1);
            return self.u(4, u32::try_from(value - 1).expect("<= 16"));
        }
        if (17..=272).contains(&value) {
            self.u(2, 2);
            return self.u(8, u32::try_from(value - 17).expect("<= 272"));
        }
        self.u(2, 3);
        self.u(12, u32::try_from(value & 0xFFF).expect("12 bits"));
        let mut remaining = value >> 12;
        let mut shift = 12u32;
        while remaining != 0 {
            self.u(1, 1);
            if shift == 60 {
                return self.u(4, u32::try_from(remaining & 0xF).expect("4 bits"));
            }
            self.u(8, u32::try_from(remaining & 0xFF).expect("8 bits"));
            remaining >>= 8;
            shift += 8;
        }
        self.u(1, 0)
    }

    /// Writes a raw `F16()` given its 16 encoded bits.
    pub fn f16_bits(&mut self, bits: u16) -> &mut Self {
        self.u(16, u32::from(bits))
    }

    /// Writes prefix-code bits in read order (C.2.4 reads them one `u(1)` at a
    /// time, so these are written most-significant first).
    pub fn code(&mut self, bits: &[u8]) -> &mut Self {
        for b in bits {
            self.bit(*b != 0);
        }
        self
    }

    /// Writes `ZeroPadToByte()` (18181-1 B.2.7).
    pub fn pad_to_byte(&mut self) -> &mut Self {
        while !self.bit_len.is_multiple_of(8) {
            self.bit(false);
        }
        self
    }

    #[must_use]
    pub const fn bit_len(&self) -> u64 {
        self.bit_len
    }

    #[must_use]
    pub fn finish(&self) -> Vec<u8> {
        self.bytes.clone()
    }

    /// The written bytes plus `extra` zero bytes.
    #[must_use]
    pub fn finish_padded(&self, extra: usize) -> Vec<u8> {
        let mut out = self.bytes.clone();
        out.resize(out.len() + extra, 0);
        out
    }
}
