//! Error type for bit-level reads.
//!
//! The contract of this module is fixed: other JPXL crates compile against
//! these exact variant names and fields. Errors are values, never panics —
//! a truncated or malformed codestream must always be reportable.

use core::fmt;

/// Everything that can go wrong while reading the JPEG XL bitstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitstreamError {
    /// A read ran past the end of the input buffer.
    ///
    /// `bit_pos` is the reader position at the time of the attempt (bits from
    /// the start of the buffer); `requested_bits` is the width that was asked
    /// for. The reader position is left unchanged when this is returned.
    OutOfBounds {
        /// Bit offset from the start of the input at which the read was attempted.
        bit_pos: u64,
        /// Number of bits the caller requested.
        requested_bits: u32,
    },
    /// A `F16()` field decoded to an exponent of 31, i.e. an infinity or NaN
    /// pattern. Part 1 forbids those encodings in header fields
    /// \[provisional: verify vs 18181-1 OCR\].
    InvalidF16,
    /// An arithmetic or width constraint was violated: a `u(n)` with `n > 32`,
    /// or a `U32()` distribution whose `offset + payload` does not fit in a
    /// `u32`.
    Overflow,
}

impl fmt::Display for BitstreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::OutOfBounds {
                bit_pos,
                requested_bits,
            } => write!(
                f,
                "out of bounds: {requested_bits} bit(s) requested at bit position {bit_pos}"
            ),
            Self::InvalidF16 => {
                f.write_str("invalid F16: exponent 31 (infinity or NaN) is not allowed")
            }
            Self::Overflow => {
                f.write_str("overflow: value or field width exceeds the representable range")
            }
        }
    }
}

impl std::error::Error for BitstreamError {}

/// Convenience alias for fallible bitstream operations.
pub type Result<T> = core::result::Result<T, BitstreamError>;
