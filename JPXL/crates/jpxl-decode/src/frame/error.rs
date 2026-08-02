//! Error type for frame-level parsing.
//!
//! Defined locally rather than as variants of
//! [`DecodeError`](crate::error::DecodeError) so that this slice does not
//! touch the shared error module while another slice is in flight.
//!
//! **TODO(slice 7):** add `impl From<FrameError> for DecodeError` in
//! `crate::error` and re-export, so that a whole-codestream decode can use `?`
//! across the header/frame boundary. Until then callers convert explicitly.

use core::fmt;

use jpxl_bitstream::BitstreamError;
use jpxl_core::JpxlError;
use jpxl_entropy::EntropyError;

use crate::error::DecodeError;

/// Anything that can go wrong reading a frame header, TOC, or group geometry.
#[derive(Debug)]
#[non_exhaustive]
pub enum FrameError {
    /// The bit reader failed.
    Bitstream(BitstreamError),
    /// A shared primitive rejected a value (geometry, limits).
    Core(JpxlError),
    /// The entropy decoder failed while reading the TOC permutation.
    Entropy(EntropyError),
    /// A nested image-header bundle failed (currently `Extensions`).
    Decode(DecodeError),
    /// A field decoded to a value the clause does not permit.
    FieldOutOfRange {
        /// Field name as written in the spec table.
        field: &'static str,
        /// Clause defining the permitted range, e.g. `"F.6"`.
        clause: &'static str,
        /// The offending value.
        value: u64,
    },
    /// A derived quantity exceeded a [`Limits`](jpxl_core::limits::Limits) cap.
    LimitExceeded {
        /// What was being sized, e.g. `"num_groups"`.
        what: &'static str,
        /// Clause the quantity comes from.
        clause: &'static str,
        /// The value that was too large.
        value: u64,
        /// The cap it exceeded.
        limit: u64,
    },
}

impl FrameError {
    /// Builds a [`FrameError::FieldOutOfRange`].
    #[must_use]
    pub const fn out_of_range(field: &'static str, clause: &'static str, value: u64) -> Self {
        Self::FieldOutOfRange {
            field,
            clause,
            value,
        }
    }
}

impl fmt::Display for FrameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bitstream(e) => write!(f, "bitstream error: {e}"),
            Self::Core(e) => write!(f, "{e}"),
            Self::Entropy(e) => write!(f, "entropy error: {e}"),
            Self::Decode(e) => write!(f, "{e}"),
            Self::FieldOutOfRange {
                field,
                clause,
                value,
            } => write!(
                f,
                "18181-1 {clause}: field {field} decoded to {value}, which the clause does not permit"
            ),
            Self::LimitExceeded {
                what,
                clause,
                value,
                limit,
            } => write!(
                f,
                "18181-1 {clause}: {what} is {value}, over the configured limit of {limit}"
            ),
        }
    }
}

impl std::error::Error for FrameError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Bitstream(e) => Some(e),
            Self::Core(e) => Some(e),
            Self::Entropy(e) => Some(e),
            Self::Decode(e) => Some(e),
            Self::FieldOutOfRange { .. } | Self::LimitExceeded { .. } => None,
        }
    }
}

impl From<BitstreamError> for FrameError {
    fn from(e: BitstreamError) -> Self {
        Self::Bitstream(e)
    }
}

impl From<JpxlError> for FrameError {
    fn from(e: JpxlError) -> Self {
        Self::Core(e)
    }
}

impl From<EntropyError> for FrameError {
    fn from(e: EntropyError) -> Self {
        Self::Entropy(e)
    }
}

impl From<DecodeError> for FrameError {
    fn from(e: DecodeError) -> Self {
        Self::Decode(e)
    }
}

/// Shorthand for a result carrying a [`FrameError`].
pub type Result<T> = std::result::Result<T, FrameError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_every_upstream_error_type() {
        let errors: [FrameError; 4] = [
            BitstreamError::InvalidF16.into(),
            JpxlError::Unsupported("x".into()).into(),
            EntropyError::Malformed("y".into()).into(),
            DecodeError::out_of_range("f", "D.7", 0).into(),
        ];
        for e in &errors {
            assert!(std::error::Error::source(e).is_some(), "{e}");
        }
    }

    #[test]
    fn messages_cite_the_clause() {
        let e = FrameError::out_of_range("num_ds", "F.6", 9);
        assert!(e.to_string().contains("18181-1 F.6"));

        let e = FrameError::LimitExceeded {
            what: "num_groups",
            clause: "F.1",
            value: 1_000_000,
            limit: 1024,
        };
        assert!(e.to_string().contains("num_groups is 1000000"));
    }
}
