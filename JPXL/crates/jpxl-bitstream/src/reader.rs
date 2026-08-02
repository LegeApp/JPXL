//! The JPEG XL bit reader.
//!
//! Bit order (Part 1, "notational conventions" \[provisional: verify vs
//! 18181-1 OCR\]): bytes are consumed in stream order and bits are taken
//! **LSB-first** within each byte. `u(n)` therefore yields the next `n` bits
//! with the first-read bit as the least-significant bit of the result. The
//! single byte `0b0000_0001` gives `read_bits(1) == 1`.

use crate::error::{BitstreamError, Result};
use crate::trace::TraceLog;

/// A cursor over a byte slice that reads bits LSB-first within each byte.
///
/// The reader never panics and never wraps: every attempt to read past the end
/// of the input yields [`BitstreamError::OutOfBounds`] and leaves the position
/// untouched.
#[derive(Debug, Clone)]
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Bits consumed so far, counted from the start of `data`.
    bit_pos: u64,
    /// `data.len() * 8`, precomputed.
    total_bits: u64,
    trace: TraceLog,
}

impl<'a> BitReader<'a> {
    /// Creates a reader positioned at the first bit of `data`.
    #[must_use]
    pub fn new(data: &'a [u8]) -> Self {
        let total_bits = u64::try_from(data.len())
            .unwrap_or(u64::MAX)
            .saturating_mul(8);
        Self {
            data,
            bit_pos: 0,
            total_bits,
            trace: TraceLog::new(),
        }
    }

    /// Returns the next `n` bits **without** advancing the cursor.
    ///
    /// `n == 0` returns `Ok(0)`. `n > 32` is [`BitstreamError::Overflow`].
    ///
    /// Note that a peek near the end of the stream reports
    /// [`BitstreamError::OutOfBounds`] exactly like a read would: this method
    /// does not zero-extend past the buffer. Callers that want a tolerant
    /// lookahead should clamp `n` with [`BitReader::bits_remaining`] first.
    pub fn peek_bits(&self, n: u32) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        if n > 32 {
            return Err(BitstreamError::Overflow);
        }
        if u64::from(n) > self.bits_remaining() {
            return Err(BitstreamError::OutOfBounds {
                bit_pos: self.bit_pos,
                requested_bits: n,
            });
        }

        let mut acc: u32 = 0;
        let mut got: u32 = 0;
        let mut byte_idx = usize::try_from(self.bit_pos / 8).unwrap_or(usize::MAX);
        // Bits already consumed inside the current byte.
        let mut bit_off = (self.bit_pos % 8) as u32;

        while got < n {
            let byte = u32::from(self.data.get(byte_idx).copied().ok_or(
                BitstreamError::OutOfBounds {
                    bit_pos: self.bit_pos,
                    requested_bits: n,
                },
            )?);
            let available = 8 - bit_off;
            let take = available.min(n - got);
            // `take <= 8`, so the mask never overflows and the shifted chunk
            // always fits: `got + take <= n <= 32`.
            let mask = (1u32 << take) - 1;
            acc |= ((byte >> bit_off) & mask) << got;
            got += take;
            bit_off = 0;
            byte_idx += 1;
        }
        Ok(acc)
    }

    /// Reads `u(n)`: the next `n` bits as an unsigned integer, advancing the
    /// cursor.
    ///
    /// `n == 0` is valid and returns `0` without consuming anything.
    /// `n > 32` is [`BitstreamError::Overflow`].
    pub fn read_bits(&mut self, n: u32) -> Result<u32> {
        let value = self.peek_bits(n)?;
        self.bit_pos += u64::from(n);
        Ok(value)
    }

    /// Reads `Bool()`, i.e. `u(1) != 0`.
    pub fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read_bits(1)? != 0)
    }

    /// Advances the cursor by `n` bits without inspecting them.
    pub fn skip_bits(&mut self, n: u64) -> Result<()> {
        if n > self.bits_remaining() {
            return Err(BitstreamError::OutOfBounds {
                bit_pos: self.bit_pos,
                requested_bits: u32::try_from(n).unwrap_or(u32::MAX),
            });
        }
        self.bit_pos += n;
        Ok(())
    }

    /// `ZeroPadToByte()`: advances to the next byte boundary.
    ///
    /// Every skipped bit must be zero. A nonzero padding bit means the
    /// codestream is malformed and is reported as
    /// [`BitstreamError::Overflow`] — the fixed error contract has no
    /// dedicated variant for it \[provisional: verify vs 18181-1 OCR\].
    ///
    /// If the cursor is already byte-aligned this is a no-op and succeeds even
    /// at end of input.
    pub fn zero_pad_to_byte(&mut self) -> Result<()> {
        let pad = ((8 - (self.bit_pos % 8)) % 8) as u32;
        if pad == 0 {
            return Ok(());
        }
        if self.read_bits(pad)? != 0 {
            return Err(BitstreamError::Overflow);
        }
        Ok(())
    }

    /// Total number of bits consumed since construction.
    #[must_use]
    pub const fn total_bits_read(&self) -> u64 {
        self.bit_pos
    }

    /// Number of bits still available in the input.
    #[must_use]
    pub const fn bits_remaining(&self) -> u64 {
        self.total_bits - self.bit_pos
    }

    /// Whether the cursor sits on a byte boundary.
    #[must_use]
    pub const fn is_byte_aligned(&self) -> bool {
        self.bit_pos.is_multiple_of(8)
    }

    /// The field trace recorded so far; always empty without the `trace` feature.
    #[must_use]
    pub const fn trace(&self) -> &TraceLog {
        &self.trace
    }

    /// Mutable access to the field trace, used by
    /// [`trace_field!`](crate::trace_field).
    pub const fn trace_mut(&mut self) -> &mut TraceLog {
        &mut self.trace
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_bit_is_lsb_of_first_byte() {
        // 0b0000_0001: the first bit read is bit 0 of the byte => 1.
        let data = [0b0000_0001u8];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bits(1), Ok(1));
        assert_eq!(r.read_bits(7), Ok(0));
    }

    #[test]
    fn zero_width_read_is_free() {
        let data = [0xFFu8];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bits(0), Ok(0));
        assert_eq!(r.total_bits_read(), 0);
        // Also valid on an exhausted reader.
        r.skip_bits(8).expect("8 bits available");
        assert_eq!(r.read_bits(0), Ok(0));
    }

    #[test]
    fn peek_does_not_advance() {
        let data = [0b1010_1010u8];
        let mut r = BitReader::new(&data);
        assert_eq!(r.peek_bits(4), Ok(0b1010));
        assert_eq!(r.peek_bits(4), Ok(0b1010));
        assert_eq!(r.total_bits_read(), 0);
        assert_eq!(r.read_bits(4), Ok(0b1010));
        assert_eq!(r.total_bits_read(), 4);
    }

    #[test]
    fn full_32_bit_read_across_four_bytes() {
        // Little-endian byte order, LSB-first bits => plain LE u32.
        let data = [0x78u8, 0x56, 0x34, 0x12];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bits(32), Ok(0x1234_5678));
        assert_eq!(r.bits_remaining(), 0);
    }

    #[test]
    fn width_above_32_is_overflow() {
        let data = [0u8; 8];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bits(33), Err(BitstreamError::Overflow));
        assert_eq!(r.peek_bits(64), Err(BitstreamError::Overflow));
        assert_eq!(r.total_bits_read(), 0);
    }

    #[test]
    fn failed_read_does_not_advance() {
        let data = [0xFFu8];
        let mut r = BitReader::new(&data);
        r.read_bits(6).expect("6 bits available");
        assert_eq!(
            r.read_bits(4),
            Err(BitstreamError::OutOfBounds {
                bit_pos: 6,
                requested_bits: 4
            })
        );
        assert_eq!(r.total_bits_read(), 6);
        // The remaining 2 bits are still readable.
        assert_eq!(r.read_bits(2), Ok(0b11));
    }

    #[test]
    fn skip_bits_bounds() {
        let data = [0u8; 2];
        let mut r = BitReader::new(&data);
        assert_eq!(r.skip_bits(16), Ok(()));
        assert_eq!(r.bits_remaining(), 0);
        assert_eq!(
            r.skip_bits(1),
            Err(BitstreamError::OutOfBounds {
                bit_pos: 16,
                requested_bits: 1
            })
        );
    }

    #[test]
    fn empty_input() {
        let mut r = BitReader::new(&[]);
        assert_eq!(r.bits_remaining(), 0);
        assert_eq!(
            r.read_bool(),
            Err(BitstreamError::OutOfBounds {
                bit_pos: 0,
                requested_bits: 1
            })
        );
        assert_eq!(r.zero_pad_to_byte(), Ok(()));
    }

    #[test]
    fn trace_hook_is_callable_regardless_of_feature() {
        let data = [0b0000_0011u8];
        let mut r = BitReader::new(&data);
        let v = crate::trace_field!(r, "two_bits", r.read_bits(2)).expect("2 bits available");
        assert_eq!(v, 0b11);
        #[cfg(feature = "trace")]
        {
            let events = r.trace().events();
            assert_eq!(events.len(), 1);
            let event = events.first().expect("one recorded event");
            assert_eq!(event.name, "two_bits");
            assert_eq!(event.start_bit, 0);
            assert_eq!(event.end_bit, 2);
            assert_eq!(event.len_bits(), 2);
        }
        #[cfg(not(feature = "trace"))]
        assert!(r.trace().events().is_empty());
    }
}
