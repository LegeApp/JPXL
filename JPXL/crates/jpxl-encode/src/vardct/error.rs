//! The plan-rejection error type.
//!
//! Every structural invariant a VarDCT plan must satisfy before it may reach
//! the writer reports through [`PlanError`]. The three variants are the three
//! *kinds* of structural failure, and each carries the name of the invariant
//! that failed plus the clause it comes from, so a rejection says which rule
//! was broken rather than "invalid plan".
//!
//! Rejection is typed, not advisory: `validate` is the only way to obtain a
//! [`ValidatedEmissionPlan`](crate::vardct::ValidatedEmissionPlan), and the
//! writer takes nothing else.

use core::fmt;

/// Why a plan cannot be emitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanError {
    /// A scalar field is outside the range its syntax element can carry, or
    /// outside the range the clause constrains it to.
    OutOfRange {
        /// The invariant's name — matched on by the per-invariant tests.
        what: &'static str,
        /// The clause the restriction comes from.
        clause: &'static str,
        /// The offending value.
        value: i64,
    },
    /// A collection has the wrong length, or a grid the wrong dimensions, for
    /// the geometry the plan declares.
    ShapeMismatch {
        /// The invariant's name.
        what: &'static str,
        /// The clause the shape comes from.
        clause: &'static str,
        /// What the geometry calls for.
        expected: u64,
        /// What the plan supplies.
        found: u64,
    },
    /// The varblock sequence is not an exact cover of an LF group's 8x8-block
    /// grid in G.2.4's greedy raster representation.
    Cover {
        /// The invariant's name.
        what: &'static str,
        /// The clause the placement rule comes from.
        clause: &'static str,
        /// The LF group whose grid failed.
        lf_group: u32,
        /// The 8x8-block position, relative to the LF group origin, where the
        /// failure was detected.
        block: (u32, u32),
    },
}

impl PlanError {
    /// The invariant name carried by any variant.
    #[must_use]
    pub const fn what(&self) -> &'static str {
        match self {
            Self::OutOfRange { what, .. }
            | Self::ShapeMismatch { what, .. }
            | Self::Cover { what, .. } => what,
        }
    }

    /// Shorthand for [`PlanError::OutOfRange`].
    #[must_use]
    pub const fn out_of_range(what: &'static str, clause: &'static str, value: i64) -> Self {
        Self::OutOfRange {
            what,
            clause,
            value,
        }
    }

    /// Shorthand for [`PlanError::ShapeMismatch`].
    #[must_use]
    pub const fn shape(
        what: &'static str,
        clause: &'static str,
        expected: u64,
        found: u64,
    ) -> Self {
        Self::ShapeMismatch {
            what,
            clause,
            expected,
            found,
        }
    }
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfRange {
                what,
                clause,
                value,
            } => write!(f, "{what} is out of range: {value} (18181-1 {clause})"),
            Self::ShapeMismatch {
                what,
                clause,
                expected,
                found,
            } => write!(
                f,
                "{what}: {expected} expected, {found} supplied (18181-1 {clause})"
            ),
            Self::Cover {
                what,
                clause,
                lf_group,
                block,
            } => write!(
                f,
                "{what} at block ({}, {}) of LF group {lf_group} (18181-1 {clause})",
                block.0, block.1
            ),
        }
    }
}

impl std::error::Error for PlanError {}

/// Convenience alias for plan validation.
pub type PlanResult<T> = core::result::Result<T, PlanError>;
