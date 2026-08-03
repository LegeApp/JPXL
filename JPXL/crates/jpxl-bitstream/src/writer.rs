//! The JPEG XL bit writer.
//!
//! The exact inverse of [`BitReader`](crate::reader::BitReader): bytes are
//! produced in stream order and bits are placed **LSB-first** within each byte,
//! so `write_bits(1, 1)` on an empty writer yields the single byte
//! `0b0000_0001`.
//!
//! # Why this is not the decoder's test helper
//!
//! `jpxl-decode` carries a `testsupport::BitWriter` that makes no choices at
//! all: a `U32()` field is written with the selector the caller names, because
//! a hand-built fixture must be able to exercise every distribution including
//! the wasteful ones. This writer is for an *encoder*: it picks the first
//! distribution of a [`U32Spec`] that can represent the value, and refuses a
//! value no distribution covers. Both behaviours are wanted, and neither is a
//! substitute for the other.

use crate::error::{BitstreamError, Result};
use crate::primitives::{U32Dist, U32Spec};

/// Accumulates bits LSB-first within each byte, matching 18181-1 B.2.1.
///
/// Every method either writes exactly what it promises or returns an error
/// having written nothing; there are no partial writes and no panics.
#[derive(Debug, Clone, Default)]
pub struct BitWriter {
    bytes: Vec<u8>,
    bit_len: u64,
}

impl BitWriter {
    /// Creates an empty writer.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            bytes: Vec::new(),
            bit_len: 0,
        }
    }

    /// Writes one bit.
    pub fn write_bit(&mut self, set: bool) {
        if self.bit_len.is_multiple_of(8) {
            self.bytes.push(0);
        }
        if set && let Some(byte) = self.bytes.last_mut() {
            // `bit_len % 8` is at most 7, so the shift is always defined.
            *byte |= 1u8 << (self.bit_len % 8);
        }
        self.bit_len += 1;
    }

    /// Writes `u(n)`: the low `n` bits of `value`, least-significant first.
    ///
    /// # Errors
    ///
    /// [`BitstreamError::Overflow`] if `n > 32`, or if `value` has any bit set
    /// at or above position `n`. A silently truncated field is exactly the bug
    /// this crate exists to make impossible, so it is rejected rather than
    /// masked.
    pub fn write_bits(&mut self, n: u32, value: u32) -> Result<()> {
        if n > 32 {
            return Err(BitstreamError::Overflow);
        }
        if n < 32 && (value >> n) != 0 {
            return Err(BitstreamError::Overflow);
        }
        for i in 0..n {
            self.write_bit((value >> i) & 1 == 1);
        }
        Ok(())
    }

    /// Writes `Bool()`, i.e. `u(1)`.
    pub fn write_bool(&mut self, value: bool) {
        self.write_bit(value);
    }

    /// Writes a `U32(d0, d1, d2, d3)` field holding `value` (18181-1 B.2.2).
    ///
    /// The first distribution of `spec` that represents `value` exactly is
    /// used. Per B.2.2 the decoder computes `(offset + payload) Umod (1 << 32)`,
    /// so the payload is recovered with a wrapping subtraction and a
    /// distribution whose offset is above `value` can still be the right one.
    ///
    /// # Errors
    ///
    /// [`BitstreamError::Overflow`] if no distribution of `spec` can encode
    /// `value`.
    pub fn write_u32(&mut self, spec: &U32Spec, value: u32) -> Result<()> {
        for (selector, dist) in spec.0.iter().enumerate() {
            let payload = match *dist {
                U32Dist::Val(v) if v == value => None,
                U32Dist::Val(_) => continue,
                U32Dist::BitsOffset { bits, offset } => {
                    if bits > 32 {
                        continue;
                    }
                    let payload = value.wrapping_sub(offset);
                    if bits < 32 && (payload >> bits) != 0 {
                        continue;
                    }
                    Some((u32::from(bits), payload))
                }
            };
            // `selector` indexes a four-element array, so it fits in two bits.
            let selector = u32::try_from(selector).map_err(|_| BitstreamError::Overflow)?;
            self.write_bits(2, selector)?;
            if let Some((bits, payload)) = payload {
                self.write_bits(bits, payload)?;
            }
            return Ok(());
        }
        Err(BitstreamError::Overflow)
    }

    /// Writes a `U64()` field holding `value` (18181-1 B.2.3).
    ///
    /// Picks the shortest of the four forms: 0, `1 + u(4)`, `17 + u(8)`, then
    /// the 12-bit-plus-continuations escape.
    ///
    /// # Errors
    ///
    /// [`BitstreamError::Overflow`] only through the underlying
    /// [`write_bits`](Self::write_bits); every `u64` is representable.
    pub fn write_u64(&mut self, value: u64) -> Result<()> {
        if value == 0 {
            return self.write_bits(2, 0);
        }
        if value <= 16 {
            self.write_bits(2, 1)?;
            return self.write_bits(4, low_bits(value - 1, 4));
        }
        if value <= 272 {
            self.write_bits(2, 2)?;
            return self.write_bits(8, low_bits(value - 17, 8));
        }

        self.write_bits(2, 3)?;
        self.write_bits(12, low_bits(value, 12))?;
        let mut remaining = value >> 12;
        let mut shift = 12u32;
        while remaining != 0 {
            self.write_bits(1, 1)?;
            if shift == 60 {
                // The final continuation carries four bits and ends the field.
                return self.write_bits(4, low_bits(remaining, 4));
            }
            self.write_bits(8, low_bits(remaining, 8))?;
            remaining >>= 8;
            shift += 8;
        }
        self.write_bits(1, 0)
    }

    /// Writes `ZeroPadToByte()` (18181-1 B.2.7): zero bits up to the next byte
    /// boundary. A no-op when already aligned.
    pub fn zero_pad_to_byte(&mut self) {
        while !self.bit_len.is_multiple_of(8) {
            self.write_bit(false);
        }
    }

    /// Appends whole bytes, which requires the writer to be byte-aligned.
    ///
    /// This is how a section encoded on its own is spliced into the
    /// codestream after its length is known.
    ///
    /// # Errors
    ///
    /// [`BitstreamError::Overflow`] if the writer is not on a byte boundary.
    pub fn write_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        if !self.is_byte_aligned() {
            return Err(BitstreamError::Overflow);
        }
        self.bytes.extend_from_slice(bytes);
        self.bit_len += u64::try_from(bytes.len())
            .map_err(|_| BitstreamError::Overflow)?
            .checked_mul(8)
            .ok_or(BitstreamError::Overflow)?;
        Ok(())
    }

    /// Number of bits written so far.
    #[must_use]
    pub const fn bit_len(&self) -> u64 {
        self.bit_len
    }

    /// Whether the cursor sits on a byte boundary.
    #[must_use]
    pub const fn is_byte_aligned(&self) -> bool {
        self.bit_len.is_multiple_of(8)
    }

    /// The bytes written so far. A trailing partial byte is zero-padded.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consumes the writer and returns its bytes, zero-padded to a byte
    /// boundary.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// The low `n` bits of `value` as a `u32`; `n` is at most 12 at every call
/// site, so the narrowing is exact by construction.
fn low_bits(value: u64, n: u32) -> u32 {
    let mask = if n >= 64 { u64::MAX } else { (1u64 << n) - 1 };
    u32::try_from(value & mask).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::{read_u32, read_u64};
    use crate::reader::BitReader;

    #[test]
    fn first_bit_is_lsb_of_first_byte() {
        let mut w = BitWriter::new();
        w.write_bits(1, 1).expect("one bit");
        w.write_bits(7, 0).expect("seven bits");
        assert_eq!(w.as_bytes(), &[0b0000_0001]);
    }

    #[test]
    fn spans_byte_boundaries_like_the_reader() {
        let mut w = BitWriter::new();
        w.write_bits(32, 0x1234_5678).expect("32 bits");
        assert_eq!(w.as_bytes(), &[0x78, 0x56, 0x34, 0x12]);

        let mut r = BitReader::new(w.as_bytes());
        assert_eq!(r.read_bits(32), Ok(0x1234_5678));
    }

    #[test]
    fn oversized_values_are_rejected_not_truncated() {
        let mut w = BitWriter::new();
        assert_eq!(w.write_bits(3, 8), Err(BitstreamError::Overflow));
        assert_eq!(w.write_bits(33, 0), Err(BitstreamError::Overflow));
        assert_eq!(w.bit_len(), 0, "a rejected write leaves nothing behind");
    }

    #[test]
    fn round_trips_arbitrary_bit_runs_through_the_reader() {
        // A deterministic pseudo-random field sequence, so the writer and the
        // reader are checked against each other across byte boundaries.
        let mut seed = 0x9E37_79B9u32;
        let mut fields = Vec::new();
        for _ in 0..200 {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let n = 1 + (seed >> 27);
            let value = if n >= 32 { seed } else { seed & ((1 << n) - 1) };
            fields.push((n, value));
        }

        let mut w = BitWriter::new();
        for &(n, value) in &fields {
            w.write_bits(n, value).expect("field fits");
        }
        let bytes = w.into_bytes();

        let mut r = BitReader::new(&bytes);
        for &(n, value) in &fields {
            assert_eq!(r.read_bits(n), Ok(value), "u({n})");
        }
    }

    #[test]
    fn u32_picks_the_first_distribution_that_fits() {
        // 18181-1 D.2's SizeHeader distribution.
        let spec = U32Spec::new([
            U32Dist::BitsOffset { bits: 9, offset: 1 },
            U32Dist::BitsOffset {
                bits: 13,
                offset: 1,
            },
            U32Dist::BitsOffset {
                bits: 18,
                offset: 1,
            },
            U32Dist::BitsOffset {
                bits: 30,
                offset: 1,
            },
        ]);
        for value in [1u32, 2, 512, 513, 8192, 100_000, 1 << 20] {
            let mut w = BitWriter::new();
            w.write_u32(&spec, value).expect("representable");
            let bytes = w.into_bytes();
            let mut r = BitReader::new(&bytes);
            assert_eq!(read_u32(&mut r, &spec), Ok(value), "value {value}");
        }

        // Selector 0 is 1 + u(9), so 512 is its largest value and 513 must
        // spill into selector 1.
        let mut w = BitWriter::new();
        w.write_u32(&spec, 512).expect("fits");
        assert_eq!(w.bit_len(), 11);
        let mut w = BitWriter::new();
        w.write_u32(&spec, 513).expect("fits");
        assert_eq!(w.bit_len(), 15);
    }

    #[test]
    fn u32_constant_distributions_cost_two_bits() {
        let spec = U32Spec::new([
            U32Dist::Val(1),
            U32Dist::Val(2),
            U32Dist::Val(4),
            U32Dist::Val(8),
        ]);
        for (value, selector) in [(1u32, 0u32), (2, 1), (4, 2), (8, 3)] {
            let mut w = BitWriter::new();
            w.write_u32(&spec, value).expect("representable");
            assert_eq!(w.bit_len(), 2);
            let bytes = w.into_bytes();
            let mut r = BitReader::new(&bytes);
            assert_eq!(r.read_bits(2), Ok(selector));
        }
        let mut w = BitWriter::new();
        assert_eq!(w.write_u32(&spec, 3), Err(BitstreamError::Overflow));
    }

    #[test]
    fn u32_offsets_wrap_exactly_as_b22_says() {
        // A distribution whose offset exceeds the value: the decoder adds
        // modulo 2^32, so the payload is the wrapping difference.
        let spec = U32Spec::new(
            [U32Dist::BitsOffset {
                bits: 32,
                offset: 100,
            }; 4],
        );
        let mut w = BitWriter::new();
        w.write_u32(&spec, 7).expect("wraps");
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        assert_eq!(read_u32(&mut r, &spec), Ok(7));
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
            1 << 40,
            u64::MAX,
        ] {
            let mut w = BitWriter::new();
            w.write_u64(value).expect("representable");
            let bytes = w.into_bytes();
            let mut r = BitReader::new(&bytes);
            assert_eq!(read_u64(&mut r), Ok(value), "value {value}");
        }
    }

    #[test]
    fn zero_pad_to_byte_matches_the_reader() {
        let mut w = BitWriter::new();
        w.write_bits(3, 0b101).expect("three bits");
        w.zero_pad_to_byte();
        assert_eq!(w.bit_len(), 8);
        assert!(w.is_byte_aligned());

        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_bits(3), Ok(0b101));
        assert_eq!(r.zero_pad_to_byte(), Ok(()));
        assert_eq!(r.total_bits_read(), 8);
    }

    #[test]
    fn write_bytes_requires_alignment() {
        let mut w = BitWriter::new();
        w.write_bits(1, 1).expect("one bit");
        assert_eq!(w.write_bytes(&[0xAB]), Err(BitstreamError::Overflow));
        w.zero_pad_to_byte();
        w.write_bytes(&[0xAB, 0xCD]).expect("aligned");
        assert_eq!(w.as_bytes(), &[0x01, 0xAB, 0xCD]);
        assert_eq!(w.bit_len(), 24);
    }
}
