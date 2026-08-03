//! Byte-level readers over the encoded ICC stream (18181-1 E.4.2).
//!
//! After E.4.1 has entropy-decoded the encoded ICC stream into plain bytes,
//! everything else in Annex E.4 is byte addressing over two sub-streams:
//!
//! ```text
//! Table E.10 — ICC stream
//!   Varint()                  output_size
//!   Varint()                  commands_size
//!   u(8 * commands_size)      command stream
//!   u(8 * [remaining bytes])  data stream
//! ```
//!
//! The command stream and the data stream each keep their own position and are
//! read from independently by E.4.3, E.4.4 and E.4.5 in turn — a subclause
//! continues where the previous one stopped in *both*.

use super::error::{Result, malformed};

/// A cursor over one of the two sub-streams of Table E.10.
#[derive(Debug, Clone, Copy)]
pub struct ByteStream<'a> {
    bytes: &'a [u8],
    pos: usize,
    /// The subclause name used in error messages, e.g. `"command"`.
    name: &'static str,
}

impl<'a> ByteStream<'a> {
    /// Wraps `bytes` as a named sub-stream positioned at its first byte.
    #[must_use]
    pub const fn new(bytes: &'a [u8], name: &'static str) -> Self {
        Self {
            bytes,
            pos: 0,
            name,
        }
    }

    /// Whether every byte of this sub-stream has been consumed.
    ///
    /// E.4.4 and E.4.5 both terminate on this condition rather than on an
    /// explicit end marker, so it is part of the parse, not an error check.
    #[must_use]
    pub const fn at_end(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    /// Bytes consumed so far.
    #[must_use]
    pub const fn position(&self) -> usize {
        self.pos
    }

    /// Bytes remaining.
    #[must_use]
    pub const fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }

    /// Reads one byte, `u(8)`.
    ///
    /// # Errors
    ///
    /// [`IccError::Malformed`](super::IccError::Malformed) at end of stream:
    /// E.4.2 states that a stream position never moves past the last byte.
    pub fn u8(&mut self) -> Result<u8> {
        let byte = *self
            .bytes
            .get(self.pos)
            .ok_or_else(|| malformed!("E.4.2: read past the end of the {} stream", self.name))?;
        self.pos += 1;
        Ok(byte)
    }

    /// Reads `count` bytes, `u(8 * count)`.
    ///
    /// # Errors
    ///
    /// As [`u8`](Self::u8) when fewer than `count` bytes remain.
    pub fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(count).ok_or_else(|| {
            malformed!(
                "E.4.2: a {}-stream read of {count} bytes overflows",
                self.name
            )
        })?;
        let slice = self.bytes.get(self.pos..end).ok_or_else(|| {
            malformed!(
                "E.4.2: a {}-stream read of {count} bytes runs past the {} bytes available",
                self.name,
                self.remaining()
            )
        })?;
        self.pos = end;
        Ok(slice)
    }

    /// Reads a `Varint()` (18181-1 E.4.2).
    ///
    /// Seven bits per byte, little-endian groups, continuing while the high bit
    /// is set. The clause bounds the shift at 56, which caps the value at 63
    /// bits — so a ninth continuation byte is malformed rather than wrapping.
    ///
    /// # Errors
    ///
    /// [`IccError::Malformed`](super::IccError::Malformed) if the stream ends
    /// mid-varint or the encoding exceeds 63 bits.
    pub fn varint(&mut self) -> Result<u64> {
        let mut value = 0u64;
        let mut shift = 0u32;
        loop {
            let byte = self.u8()?;
            value = value
                .checked_add(u64::from(byte & 127) << shift)
                .ok_or_else(|| malformed!("E.4.2: Varint() value exceeds 63 bits"))?;
            if byte <= 127 {
                return Ok(value);
            }
            shift += 7;
            if shift > 56 {
                return Err(malformed!(
                    "E.4.2: Varint() continues past the 56-bit shift the clause permits"
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Proves the seven-bits-per-byte layout and the continuation rule against
    /// values hand-encoded from the E.4.2 pseudocode.
    #[test]
    fn varint_matches_the_clause_encoding() {
        let cases: &[(&[u8], u64)] = &[
            (&[0x00], 0),
            (&[0x7F], 127),
            (&[0x80, 0x01], 128),
            (&[0xFF, 0x7F], 16_383),
            (&[0x80, 0x80, 0x01], 1 << 14),
        ];
        for (bytes, expected) in cases {
            let mut s = ByteStream::new(bytes, "command");
            assert_eq!(s.varint().expect("valid varint"), *expected);
            assert!(s.at_end(), "the varint must consume exactly its bytes");
        }
    }

    /// Proves a varint that never terminates is rejected rather than looping or
    /// silently truncating: nine continuation bytes exceed the 56-bit shift.
    #[test]
    fn varint_rejects_an_overlong_encoding() {
        let bytes = [0xFFu8; 12];
        let mut s = ByteStream::new(&bytes, "command");
        assert!(s.varint().is_err());
    }

    /// Proves a truncated varint is an error, not a value.
    #[test]
    fn varint_rejects_a_truncated_encoding() {
        let mut s = ByteStream::new(&[0x80], "data");
        assert!(s.varint().is_err());
    }

    #[test]
    fn take_bounds_check_is_exact() {
        let bytes = [1u8, 2, 3];
        let mut s = ByteStream::new(&bytes, "data");
        assert_eq!(s.take(2).expect("fits"), &[1, 2]);
        assert!(s.take(2).is_err());
        assert_eq!(s.take(1).expect("fits"), &[3]);
        assert!(s.at_end());
        assert_eq!(s.position(), 3);
    }
}
