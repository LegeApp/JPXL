//! Error type for entropy decoding.
//!
//! Follows the project error style (see `jpxl_core::error`): a hand-rolled
//! enum with one variant per upstream crate and `From` impls so `?` crosses
//! layer boundaries without glue.

use core::fmt;

use jpxl_bitstream::BitstreamError;
use jpxl_core::JpxlError;

/// Anything that can go wrong while decoding an entropy-coded stream.
#[derive(Debug)]
#[non_exhaustive]
pub enum EntropyError {
    /// The underlying bit reader failed.
    Bitstream(BitstreamError),
    /// A shared-primitive operation failed, typically a [`Limits`] rejection.
    ///
    /// [`Limits`]: jpxl_core::limits::Limits
    Core(JpxlError),
    /// The stream violates a requirement of ISO/IEC 18181-1 Annex C.
    ///
    /// The message names the clause and the invariant that failed. Text is for
    /// humans and is not stable.
    Malformed(String),
    /// A construct that is well-formed but not implemented by this crate.
    ///
    /// Every stub in this crate returns this rather than panicking or silently
    /// producing wrong symbols.
    Unsupported(String),
}

impl fmt::Display for EntropyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bitstream(e) => write!(f, "bitstream error: {e}"),
            Self::Core(e) => write!(f, "{e}"),
            Self::Malformed(msg) => write!(f, "malformed entropy stream: {msg}"),
            Self::Unsupported(msg) => write!(f, "unsupported entropy feature: {msg}"),
        }
    }
}

impl std::error::Error for EntropyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Bitstream(e) => Some(e),
            Self::Core(e) => Some(e),
            Self::Malformed(_) | Self::Unsupported(_) => None,
        }
    }
}

impl From<BitstreamError> for EntropyError {
    fn from(e: BitstreamError) -> Self {
        Self::Bitstream(e)
    }
}

impl From<JpxlError> for EntropyError {
    fn from(e: JpxlError) -> Self {
        Self::Core(e)
    }
}

/// Shorthand for a result carrying an [`EntropyError`].
pub type Result<T> = core::result::Result<T, EntropyError>;

/// Builds a [`EntropyError::Malformed`] with a formatted message.
macro_rules! malformed {
    ($($arg:tt)*) => {
        $crate::error::EntropyError::Malformed(format!($($arg)*))
    };
}

pub(crate) use malformed;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_upstream_errors() {
        let err: EntropyError = BitstreamError::InvalidF16.into();
        assert!(matches!(err, EntropyError::Bitstream(_)));
        assert!(std::error::Error::source(&err).is_some());

        let err: EntropyError = JpxlError::Unsupported("x".into()).into();
        assert!(matches!(err, EntropyError::Core(_)));
    }

    #[test]
    fn message_variants_have_no_source() {
        let err = malformed!("C.2.5: {} != {}", 1, 2);
        assert_eq!(err.to_string(), "malformed entropy stream: C.2.5: 1 != 2");
        assert!(std::error::Error::source(&err).is_none());
    }
}
