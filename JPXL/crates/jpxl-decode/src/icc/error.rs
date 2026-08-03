//! Error type for ICC profile decoding (18181-1 E.4).
//!
//! Follows the project error style (see `jpxl_core::error`, `crate::error` and
//! `crate::modular::error`): a hand-rolled `enum`, one variant per upstream
//! crate so `?` composes, and clause citations in the messages. No `thiserror`.
//!
//! Unlike `ModularError` this type *is* wrapped into
//! [`DecodeError`](crate::error::DecodeError) from the start, by the
//! `DecodeError::Icc` variant this slice adds.

use core::fmt;

use jpxl_bitstream::BitstreamError;
use jpxl_core::JpxlError;
use jpxl_entropy::EntropyError;

/// Anything that can go wrong while decoding an embedded ICC profile.
#[derive(Debug)]
#[non_exhaustive]
pub enum IccError {
    /// The underlying bit reader failed while reading E.4.1.
    Bitstream(BitstreamError),
    /// A shared primitive rejected a value, typically an [`AllocGuard`] charge.
    ///
    /// [`AllocGuard`]: jpxl_core::limits::AllocGuard
    Core(JpxlError),
    /// The entropy layer (Annex C) rejected the ICC byte stream.
    Entropy(EntropyError),
    /// The encoded ICC stream violates a requirement of E.4.
    ///
    /// The message names the subclause and the invariant that failed. Text is
    /// for humans and is not stable.
    Malformed(String),
}

impl fmt::Display for IccError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bitstream(e) => write!(f, "bitstream error: {e}"),
            Self::Core(e) => write!(f, "{e}"),
            Self::Entropy(e) => write!(f, "{e}"),
            Self::Malformed(msg) => write!(f, "malformed ICC profile stream: {msg}"),
        }
    }
}

impl std::error::Error for IccError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Bitstream(e) => Some(e),
            Self::Core(e) => Some(e),
            Self::Entropy(e) => Some(e),
            Self::Malformed(_) => None,
        }
    }
}

impl From<BitstreamError> for IccError {
    fn from(e: BitstreamError) -> Self {
        Self::Bitstream(e)
    }
}

impl From<JpxlError> for IccError {
    fn from(e: JpxlError) -> Self {
        Self::Core(e)
    }
}

impl From<EntropyError> for IccError {
    fn from(e: EntropyError) -> Self {
        Self::Entropy(e)
    }
}

/// Shorthand for a result carrying an [`IccError`].
pub type Result<T> = std::result::Result<T, IccError>;

/// Builds an [`IccError::Malformed`] with a formatted message.
macro_rules! malformed {
    ($($arg:tt)*) => {
        $crate::icc::error::IccError::Malformed(format!($($arg)*))
    };
}

pub(crate) use malformed;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_every_upstream_error_type() {
        let a: IccError = BitstreamError::InvalidF16.into();
        assert!(std::error::Error::source(&a).is_some());
        let b: IccError = JpxlError::Unsupported("x".into()).into();
        assert!(std::error::Error::source(&b).is_some());
        let c: IccError = EntropyError::Malformed("x".into()).into();
        assert!(std::error::Error::source(&c).is_some());
    }

    #[test]
    fn malformed_messages_carry_the_subclause() {
        let e = malformed!("E.4.2: commands_size {} overruns the stream", 9);
        assert!(e.to_string().contains("E.4.2"));
        assert!(std::error::Error::source(&e).is_none());
    }
}
