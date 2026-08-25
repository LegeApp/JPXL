//! Error type for the JPEG-1 codec.
//!
//! Follows the project error style (AGENTS.md §6, mirroring
//! `jpxl_core::error`): a hand-rolled enum, no `thiserror`/`anyhow`, with a
//! human-readable message that names the 10918-1 clause and the invariant that
//! failed. Message text is for humans and is not stable.

use core::fmt;

/// Anything that can go wrong parsing or re-emitting a JPEG-1 bitstream.
#[derive(Debug)]
#[non_exhaustive]
pub enum JpegError {
    /// The stream is truncated: more bytes were required than are present.
    UnexpectedEof {
        /// What the parser was reading when the bytes ran out.
        while_reading: &'static str,
    },
    /// The stream violates a requirement of ISO/IEC 10918-1.
    ///
    /// The message names the clause and the invariant that failed.
    Malformed(String),
    /// The stream is a well-formed JPEG the codec deliberately refuses.
    ///
    /// Arithmetic-coded, hierarchical, 12-bit-sample and lossless JPEG modes
    /// are out of Phase-A scope; each is rejected here rather than mis-parsed.
    Unsupported(String),
    /// A resource limit (allocation cap) was exceeded while parsing.
    ///
    /// Decode paths are attacker-facing (AGENTS.md §6): oversized dimensions or
    /// table counts are rejected before any large allocation.
    LimitExceeded(String),
    /// The re-emitter was asked for something it cannot serialize.
    ///
    /// A caller/logic error, not a stream error: a coefficient outside the
    /// range its magnitude category can express, a Huffman symbol absent from
    /// the table it must be coded with.
    Encode(String),
}

impl fmt::Display for JpegError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof { while_reading } => {
                write!(
                    f,
                    "unexpected end of JPEG data while reading {while_reading}"
                )
            }
            Self::Malformed(msg) => write!(f, "malformed JPEG stream: {msg}"),
            Self::Unsupported(msg) => write!(f, "unsupported JPEG feature: {msg}"),
            Self::LimitExceeded(msg) => write!(f, "JPEG resource limit exceeded: {msg}"),
            Self::Encode(msg) => write!(f, "cannot re-emit JPEG stream: {msg}"),
        }
    }
}

impl std::error::Error for JpegError {}

/// Shorthand for a result carrying a [`JpegError`].
pub type Result<T> = core::result::Result<T, JpegError>;

/// Builds a [`JpegError::Unsupported`] with a formatted message.
macro_rules! unsupported {
    ($($arg:tt)*) => {
        $crate::error::JpegError::Unsupported(format!($($arg)*))
    };
}
pub(crate) use unsupported;
