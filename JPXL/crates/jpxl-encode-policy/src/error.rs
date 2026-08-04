//! The policy crate's error type.
//!
//! Hand-rolled with `From` impls chaining the layer below, per `AGENTS.md`.

use core::fmt;

use jpxl_encode::vardct::PlanError;

/// Everything that can go wrong while planning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyError {
    /// The plan this policy built was rejected by `jpxl-encode`.
    ///
    /// Always a policy bug: the boundary's contract is that policy produces
    /// *legal* plans, and a rejection names the invariant it broke.
    Plan(PlanError),
    /// The source image is outside what this policy can plan for.
    Unsupported {
        /// What the caller asked for.
        what: &'static str,
    },
    /// The source planes and the declared dimensions disagree.
    SampleCountMismatch {
        /// How many samples the dimensions call for.
        expected: u64,
        /// How many were supplied.
        found: u64,
    },
}

impl fmt::Display for PolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plan(err) => write!(f, "{err}"),
            Self::Unsupported { what } => write!(f, "unsupported: {what}"),
            Self::SampleCountMismatch { expected, found } => write!(
                f,
                "sample count mismatch: {expected} expected, {found} supplied"
            ),
        }
    }
}

impl std::error::Error for PolicyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Plan(err) => Some(err),
            _ => None,
        }
    }
}

impl From<PlanError> for PolicyError {
    fn from(err: PlanError) -> Self {
        Self::Plan(err)
    }
}

/// Convenience alias for fallible planning.
pub type Result<T> = core::result::Result<T, PolicyError>;
