//! Error type for `jpxl-decode`.
//!
//! Follows the project style established in `jpxl_core::error`: a hand-rolled
//! `enum`, `Display` and [`std::error::Error`] written out, and one variant per
//! upstream crate so `?` composes across layers. No `thiserror`.
//!
//! Variants carry the clause that was being applied when the failure happened.
//! A decoder that says only "invalid header" forces a bisect; one that says
//! "18181-1 D.7: bits_per_sample = 0 outside [1, 31]" does not.

use core::fmt;

use jpxl_bitstream::BitstreamError;
use jpxl_core::JpxlError;

/// Anything that can go wrong while decoding a codestream.
#[derive(Debug)]
#[non_exhaustive]
pub enum DecodeError {
    /// The bit reader failed: ran off the end, bad `F16()`, or a bad width.
    Bitstream(BitstreamError),
    /// A shared primitive rejected a value (geometry, limits).
    Core(JpxlError),
    /// 18181-1 D.1: the 16-bit signature was not `0x0AFF`.
    InvalidSignature {
        /// The value actually read.
        found: u32,
    },
    /// A field decoded to a value the clause does not permit.
    FieldOutOfRange {
        /// Field name as written in the spec table.
        field: &'static str,
        /// Clause defining the permitted range, e.g. `"D.7"`.
        clause: &'static str,
        /// The offending value.
        value: u64,
    },
    /// 18181-1 B.2.6: an `Enum()` value with no row in its table.
    UnknownEnumValue {
        /// The `EnumTable` name from the spec, e.g. `"ExtraChannelType"`.
        table: &'static str,
        /// Clause defining the table.
        clause: &'static str,
        /// The offending value.
        value: u32,
    },
    /// A well-formed codestream using something JPXL does not implement yet.
    Unsupported {
        /// What was encountered.
        feature: &'static str,
        /// Clause specifying it.
        clause: &'static str,
    },
}

impl DecodeError {
    /// Builds a [`DecodeError::FieldOutOfRange`].
    #[must_use]
    pub const fn out_of_range(field: &'static str, clause: &'static str, value: u64) -> Self {
        Self::FieldOutOfRange {
            field,
            clause,
            value,
        }
    }
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bitstream(e) => write!(f, "bitstream error: {e}"),
            Self::Core(e) => write!(f, "{e}"),
            Self::InvalidSignature { found } => write!(
                f,
                "18181-1 D.1: codestream signature is {found:#06x}, expected 0x0aff (bytes ff 0a)"
            ),
            Self::FieldOutOfRange {
                field,
                clause,
                value,
            } => write!(
                f,
                "18181-1 {clause}: field {field} decoded to {value}, which the clause does not permit"
            ),
            Self::UnknownEnumValue {
                table,
                clause,
                value,
            } => write!(
                f,
                "18181-1 {clause}: value {value} has no row in enumerated type {table}"
            ),
            Self::Unsupported { feature, clause } => {
                write!(f, "18181-1 {clause}: {feature} is not implemented")
            }
        }
    }
}

impl std::error::Error for DecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Bitstream(e) => Some(e),
            Self::Core(e) => Some(e),
            Self::InvalidSignature { .. }
            | Self::FieldOutOfRange { .. }
            | Self::UnknownEnumValue { .. }
            | Self::Unsupported { .. } => None,
        }
    }
}

impl From<BitstreamError> for DecodeError {
    fn from(e: BitstreamError) -> Self {
        Self::Bitstream(e)
    }
}

impl From<JpxlError> for DecodeError {
    fn from(e: JpxlError) -> Self {
        Self::Core(e)
    }
}

/// Shorthand for a result carrying a [`DecodeError`].
pub type Result<T> = std::result::Result<T, DecodeError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_both_upstream_error_types() {
        let a: DecodeError = BitstreamError::InvalidF16.into();
        assert!(std::error::Error::source(&a).is_some());
        let b: DecodeError = JpxlError::Unsupported("x".into()).into();
        assert!(std::error::Error::source(&b).is_some());
    }

    #[test]
    fn messages_cite_the_clause() {
        let e = DecodeError::out_of_range("bits_per_sample", "D.7", 0);
        assert!(e.to_string().contains("18181-1 D.7"));
        assert!(e.to_string().contains("bits_per_sample"));

        let e = DecodeError::InvalidSignature { found: 0x1234 };
        assert!(e.to_string().contains("0x0aff"));
    }
}
