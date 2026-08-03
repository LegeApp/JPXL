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
use jpxl_entropy::EntropyError;

use crate::frame::FrameError;
use crate::icc::IccError;
use crate::modular::ModularError;

/// Anything that can go wrong while decoding a codestream.
#[derive(Debug)]
#[non_exhaustive]
pub enum DecodeError {
    /// The bit reader failed: ran off the end, bad `F16()`, or a bad width.
    Bitstream(BitstreamError),
    /// A shared primitive rejected a value (geometry, limits).
    Core(JpxlError),
    /// The entropy layer (Annex C) rejected the stream.
    Entropy(EntropyError),
    /// Frame header, TOC or geometry parsing failed (Annexes F and G).
    ///
    /// Boxed because `FrameError` nests a `DecodeError` in turn (a frame
    /// header can contain an `Extensions` bundle), and the two types would
    /// otherwise be mutually recursive without indirection.
    Frame(Box<FrameError>),
    /// A modular sub-bitstream failed (Annex H).
    Modular(ModularError),
    /// The embedded ICC profile failed to decode (E.4).
    Icc(IccError),
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
        /// Fully-qualified clause specifying it, e.g. `"18181-1 Annex I"` —
        /// including the part number, since container boxes live in Part 2.
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
            Self::Entropy(e) => write!(f, "{e}"),
            Self::Frame(e) => write!(f, "{e}"),
            Self::Modular(e) => write!(f, "{e}"),
            Self::Icc(e) => write!(f, "{e}"),
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
                // The clause carries its own part number: this is the one
                // variant that can name Part 2 (container boxes) as well as
                // Part 1.
                write!(f, "{clause}: {feature} is not implemented")
            }
        }
    }
}

impl std::error::Error for DecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Bitstream(e) => Some(e),
            Self::Core(e) => Some(e),
            Self::Entropy(e) => Some(e),
            Self::Frame(e) => Some(e),
            Self::Modular(e) => Some(e),
            Self::Icc(e) => Some(e),
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

impl From<EntropyError> for DecodeError {
    fn from(e: EntropyError) -> Self {
        Self::Entropy(e)
    }
}

/// Slice 7 wiring: see `crate::frame::error` for why `FrameError` is its own
/// type rather than a set of `DecodeError` variants.
impl From<FrameError> for DecodeError {
    fn from(e: FrameError) -> Self {
        Self::Frame(Box::new(e))
    }
}

/// Slice 7 wiring: see `crate::modular::error` for why `ModularError` is its
/// own type rather than a set of `DecodeError` variants.
impl From<ModularError> for DecodeError {
    fn from(e: ModularError) -> Self {
        Self::Modular(e)
    }
}

/// Slice 4 wiring: `IccError` (E.4) is a module-local leaf type for the same
/// reason `ModularError` is — the ICC stages have their own vocabulary of
/// failures — and composes into `DecodeError` here.
impl From<IccError> for DecodeError {
    fn from(e: IccError) -> Self {
        Self::Icc(e)
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
