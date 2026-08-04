//! Error type for encoding.
//!
//! Hand-rolled with `From` impls chaining the layer below, per `AGENTS.md`.
//! The encoder's inputs are the caller's own image, not attacker-controlled
//! bytes, so the failure modes are narrow: an image outside the subset this
//! slice supports, or a value that no field of the syntax can carry.

use core::fmt;

use jpxl_bitstream::BitstreamError;
use jpxl_core::JpxlError;
use jpxl_entropy::EntropyError;

use crate::vardct::error::PlanError;

/// Everything that can go wrong while encoding.
///
/// Not `PartialEq`: [`JpxlError`] carries an `std::io::Error`, which has no
/// equality. Tests match on the variant instead.
#[derive(Debug)]
pub enum EncodeError {
    /// The bit writer rejected a field.
    Bitstream(BitstreamError),
    /// A shared-layer error (dimension or limit validation).
    Core(JpxlError),
    /// A plan violated a structural invariant, so it never reached the writer.
    Plan(PlanError),
    /// The entropy layer refused a symbol, a table or a configuration.
    ///
    /// Reaching this from the VarDCT writer means the census the histograms
    /// were trained on and the symbols the replay pass produced disagree —
    /// always a bug on this side, never in the caller's image.
    Entropy(EntropyError),
    /// The image is outside the subset this encoder produces.
    ///
    /// `clause` names where the restriction comes from, so the message says
    /// what would have to be implemented rather than just "no".
    Unsupported {
        /// What the caller asked for.
        what: &'static str,
        /// The clause that would govern it.
        clause: &'static str,
    },
    /// A sample or residual is outside the range the chosen coding parameters
    /// can express.
    ValueOutOfRange {
        /// Which quantity.
        what: &'static str,
        /// The offending value.
        value: i64,
    },
    /// The image dimensions and the sample buffer disagree.
    SampleCountMismatch {
        /// How many samples `width * height` calls for.
        expected: u64,
        /// How many were supplied.
        found: u64,
    },
}

impl EncodeError {
    /// Shorthand for [`EncodeError::Unsupported`].
    #[must_use]
    pub const fn unsupported(what: &'static str, clause: &'static str) -> Self {
        Self::Unsupported { what, clause }
    }
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bitstream(err) => write!(f, "bitstream: {err}"),
            Self::Core(err) => write!(f, "{err}"),
            Self::Plan(err) => write!(f, "plan rejected: {err}"),
            Self::Entropy(err) => write!(f, "entropy: {err}"),
            Self::Unsupported { what, clause } => {
                write!(f, "unsupported: {what} (18181-1 {clause})")
            }
            Self::ValueOutOfRange { what, value } => {
                write!(f, "{what} is out of range: {value}")
            }
            Self::SampleCountMismatch { expected, found } => write!(
                f,
                "sample count mismatch: {expected} expected, {found} supplied"
            ),
        }
    }
}

impl std::error::Error for EncodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Bitstream(err) => Some(err),
            Self::Core(err) => Some(err),
            Self::Plan(err) => Some(err),
            Self::Entropy(err) => Some(err),
            _ => None,
        }
    }
}

impl From<BitstreamError> for EncodeError {
    fn from(err: BitstreamError) -> Self {
        Self::Bitstream(err)
    }
}

impl From<JpxlError> for EncodeError {
    fn from(err: JpxlError) -> Self {
        Self::Core(err)
    }
}

impl From<EntropyError> for EncodeError {
    fn from(err: EntropyError) -> Self {
        Self::Entropy(err)
    }
}

impl From<PlanError> for EncodeError {
    fn from(err: PlanError) -> Self {
        Self::Plan(err)
    }
}

/// Convenience alias for fallible encoding operations.
pub type Result<T> = core::result::Result<T, EncodeError>;
