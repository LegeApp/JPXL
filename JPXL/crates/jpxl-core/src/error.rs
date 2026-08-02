//! Error type for `jpxl-core`, and the project-wide error style.
//!
//! # Project error style
//!
//! JPXL has zero external dependencies, so there is no `thiserror` and no
//! `anyhow`. Every crate hand-rolls its own `enum` error type and implements
//! [`Display`](core::fmt::Display) and [`std::error::Error`] by hand.
//!
//! Errors compose *upward* through `From` impls: a crate's error type wraps
//! the error types of the crates it depends on, one variant per upstream
//! crate, so `?` works across layers without any glue at the call site. Here
//! that means [`JpxlError::Bitstream`] wraps [`BitstreamError`]. A future
//! `jpxl-decode` error type would in turn wrap [`JpxlError`].
//!
//! Variants carrying a `String` are for conditions whose detail is only known
//! at runtime and is meant for humans; they are not matched on programmatically
//! and their text is not stable.

use core::fmt;

use jpxl_bitstream::BitstreamError;

/// Anything that can go wrong in the shared JPEG XL primitives.
#[derive(Debug)]
#[non_exhaustive]
pub enum JpxlError {
    /// The underlying bit reader failed (ran off the end, bad float, overflow).
    Bitstream(BitstreamError),
    /// A header field was present but its value is not valid per ISO/IEC 18181.
    InvalidHeader(String),
    /// The codestream is well-formed but uses a feature JPXL does not implement.
    Unsupported(String),
    /// A [`Limits`](crate::limits::Limits) bound would have been exceeded.
    LimitExceeded(String),
    /// An I/O error while reading or writing a file.
    Io(std::io::Error),
}

impl fmt::Display for JpxlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bitstream(e) => write!(f, "bitstream error: {e}"),
            Self::InvalidHeader(msg) => write!(f, "invalid header: {msg}"),
            Self::Unsupported(msg) => write!(f, "unsupported feature: {msg}"),
            Self::LimitExceeded(msg) => write!(f, "limit exceeded: {msg}"),
            Self::Io(e) => write!(f, "i/o error: {e}"),
        }
    }
}

impl std::error::Error for JpxlError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Bitstream(e) => Some(e),
            Self::Io(e) => Some(e),
            Self::InvalidHeader(_) | Self::Unsupported(_) | Self::LimitExceeded(_) => None,
        }
    }
}

impl From<BitstreamError> for JpxlError {
    fn from(e: BitstreamError) -> Self {
        Self::Bitstream(e)
    }
}

impl From<std::io::Error> for JpxlError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Shorthand for a result carrying a [`JpxlError`].
pub type Result<T> = std::result::Result<T, JpxlError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_bitstream_error() {
        let err: JpxlError = BitstreamError::InvalidF16.into();
        assert!(matches!(err, JpxlError::Bitstream(_)));
        assert!(err.to_string().starts_with("bitstream error:"));
        assert!(std::error::Error::source(&err).is_some());
    }

    #[test]
    fn wraps_io_error() {
        let io = std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "short read");
        let err: JpxlError = io.into();
        assert!(err.to_string().contains("short read"));
    }

    #[test]
    fn message_variants_have_no_source() {
        let err = JpxlError::Unsupported("modular tree".into());
        assert!(std::error::Error::source(&err).is_none());
        assert_eq!(err.to_string(), "unsupported feature: modular tree");
    }
}
