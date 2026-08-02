//! Error type for Modular decoding (18181-1 Annex H).
//!
//! Follows the project error style (see `jpxl_core::error` and
//! `crate::error`): a hand-rolled `enum`, one variant per upstream crate so
//! `?` composes, and clause citations in the messages. No `thiserror`.
//!
//! # Why this is not a `DecodeError` variant
//!
//! `crate::error::DecodeError` lives in `src/error.rs`, which slice 5 does not
//! own. The `From<ModularError> for DecodeError` impl is therefore **slice 7's
//! job**: add
//!
//! ```ignore
//! impl From<ModularError> for DecodeError {
//!     fn from(e: ModularError) -> Self { /* new DecodeError::Modular variant */ }
//! }
//! ```
//!
//! Until then `ModularError` is a self-contained leaf type and
//! [`modular::Result`](crate::modular::Result) shadows the crate-level alias
//! inside this module tree only.

use core::fmt;

use jpxl_bitstream::BitstreamError;
use jpxl_core::JpxlError;
use jpxl_entropy::EntropyError;

/// Anything that can go wrong while decoding a modular sub-bitstream.
#[derive(Debug)]
#[non_exhaustive]
pub enum ModularError {
    /// The underlying bit reader failed: ran off the end, or a bad width.
    Bitstream(BitstreamError),
    /// A shared primitive rejected a value, typically an [`AllocGuard`] charge.
    ///
    /// [`AllocGuard`]: jpxl_core::limits::AllocGuard
    Core(JpxlError),
    /// The entropy layer (Annex C) rejected the stream.
    Entropy(EntropyError),
    /// The sub-bitstream violates a requirement of Annex H.
    ///
    /// The message names the clause and the invariant that failed. Text is for
    /// humans and is not stable.
    Malformed(String),
    /// A construct that is well-formed per Annex H but not implemented here.
    ///
    /// Every deferral in this module returns this rather than panicking or
    /// silently producing wrong samples.
    Unsupported(String),
}

impl fmt::Display for ModularError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bitstream(e) => write!(f, "bitstream error: {e}"),
            Self::Core(e) => write!(f, "{e}"),
            Self::Entropy(e) => write!(f, "{e}"),
            Self::Malformed(msg) => write!(f, "malformed modular sub-bitstream: {msg}"),
            Self::Unsupported(msg) => write!(f, "unsupported modular feature: {msg}"),
        }
    }
}

impl std::error::Error for ModularError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Bitstream(e) => Some(e),
            Self::Core(e) => Some(e),
            Self::Entropy(e) => Some(e),
            Self::Malformed(_) | Self::Unsupported(_) => None,
        }
    }
}

impl From<BitstreamError> for ModularError {
    fn from(e: BitstreamError) -> Self {
        Self::Bitstream(e)
    }
}

impl From<JpxlError> for ModularError {
    fn from(e: JpxlError) -> Self {
        Self::Core(e)
    }
}

impl From<EntropyError> for ModularError {
    fn from(e: EntropyError) -> Self {
        Self::Entropy(e)
    }
}

/// Shorthand for a result carrying a [`ModularError`].
pub type Result<T> = std::result::Result<T, ModularError>;

/// Builds a [`ModularError::Malformed`] with a formatted message.
macro_rules! malformed {
    ($($arg:tt)*) => {
        $crate::modular::error::ModularError::Malformed(format!($($arg)*))
    };
}

pub(crate) use malformed;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_every_upstream_error_type() {
        let a: ModularError = BitstreamError::InvalidF16.into();
        assert!(std::error::Error::source(&a).is_some());
        let b: ModularError = JpxlError::Unsupported("x".into()).into();
        assert!(std::error::Error::source(&b).is_some());
        let c: ModularError = EntropyError::Malformed("x".into()).into();
        assert!(std::error::Error::source(&c).is_some());
    }

    #[test]
    fn malformed_messages_carry_the_clause() {
        let e = malformed!("H.4.2: tree has {} nodes", 5);
        assert!(e.to_string().contains("H.4.2"));
        assert!(std::error::Error::source(&e).is_none());
    }
}
