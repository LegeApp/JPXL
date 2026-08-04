//! The policy crate's error type.
//!
//! Hand-rolled with `From` impls chaining the layer below, per `AGENTS.md`.

use core::fmt;

use jpxl_encode::EncodeError;
use jpxl_encode::vardct::PlanError;

/// Everything that can go wrong while planning.
///
/// Not `PartialEq`: [`EncodeError`] wraps errors that have no equality. Tests
/// match on the variant.
#[derive(Debug)]
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
    /// The writer refused the plan, or a bit could not be written.
    ///
    /// Reaching this means the plan validated but the milestone-2 writer has
    /// no encoding for some part of it — see `jpxl_encode::vardct::write`.
    Encode(EncodeError),
    /// No representable quantizer produces a stream this small.
    ///
    /// The coarsest rung of the rate ladder still exceeds the target, so the
    /// answer is not "try harder" — it is that the frame's headers, TOC and
    /// unavoidable structure already cost more than the caller allowed. The
    /// floor is reported so the caller can raise the target to something
    /// achievable.
    TargetUnreachable {
        /// The byte budget asked for.
        target: u64,
        /// The smallest stream any representable quantizer produces.
        floor: u64,
    },
    /// The rate loop ran out of exact prices before finding any candidate that
    /// fits.
    ///
    /// A budget this small is a caller decision, not a failure of the search;
    /// the count is reported so the caller can see what it bought.
    SearchBudgetExhausted {
        /// How many candidates were priced.
        prices: usize,
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
            Self::Encode(err) => write!(f, "{err}"),
            Self::Unsupported { what } => write!(f, "unsupported: {what}"),
            Self::TargetUnreachable { target, floor } => write!(
                f,
                "no representable quantizer reaches {target} bytes: the floor is {floor}"
            ),
            Self::SearchBudgetExhausted { prices } => write!(
                f,
                "the rate search budget ran out after {prices} exact prices \
                 without a candidate under the target"
            ),
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
            Self::Encode(err) => Some(err),
            _ => None,
        }
    }
}

impl From<EncodeError> for PolicyError {
    fn from(err: EncodeError) -> Self {
        Self::Encode(err)
    }
}

impl From<PlanError> for PolicyError {
    fn from(err: PlanError) -> Self {
        Self::Plan(err)
    }
}

/// Convenience alias for fallible planning.
pub type Result<T> = core::result::Result<T, PolicyError>;
