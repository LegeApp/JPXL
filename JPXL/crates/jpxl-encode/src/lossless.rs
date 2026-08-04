//! The lossless-modular track's plan type.
//!
//! Slice 11 draws the policy boundary for VarDCT in [`crate::vardct`]. The
//! lossless encoder that already exists predates that boundary, and it made
//! three choices inline in the middle of `encode_codestream`: the group size,
//! whether to apply an RCT, and how wide the flat prefix code has to be. Those
//! are search decisions — cheap, obvious ones, but decisions — and they are
//! now stated as a plan that the emission path takes as input.
//!
//! ```text
//! plan_for(...)  ──▶  LosslessPlan  ──validate──▶  ValidatedLosslessPlan
//!  (policy)                                              │
//!                                                        ▼
//!                                          crate::encode_codestream_with_plan
//! ```
//!
//! # Where the policy half goes
//!
//! [`plan_for`] is the policy half, and per `docs/PLAN.md` it moves to
//! `jpxl-encode-policy` in **slice 19** (lossless density: learned MA trees,
//! LZ77, predictor selection), which is the slice that gives it something to
//! actually decide. Moving it now would mean moving three lines and calling it
//! architecture. What matters for slice 11 — and what is done here — is that
//! the emission path no longer *makes* the decisions: it takes them.
//!
//! Behaviour is unchanged. [`plan_for`] computes exactly what the inline code
//! computed, and the encoder's end-to-end tests are the proof.

use crate::EncodeOptions;
use crate::entropy::FlatCode;
use crate::error::{EncodeError, Result};
use crate::frame::{DEFAULT_GROUP_SIZE_SHIFT, MAX_GROUP_SIZE_SHIFT};
use crate::modular::Plane;

/// Every choice the lossless modular path makes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LosslessPlan {
    /// F.2's `group_size_shift`.
    pub group_size_shift: u32,
    /// Whether the planes carry H.6.3's `kRCT` (YCoCg) transform.
    pub rct: bool,
    /// The flat prefix code the residuals are written with.
    pub code: FlatCode,
}

/// A [`LosslessPlan`] that has passed validation.
///
/// As with [`crate::vardct::ValidatedEmissionPlan`], the inner plan is private
/// and [`validate`] is the only constructor, so the emission path cannot be
/// handed an unchecked plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValidatedLosslessPlan(LosslessPlan);

impl ValidatedLosslessPlan {
    /// The plan.
    #[must_use]
    pub const fn plan(&self) -> &LosslessPlan {
        &self.0
    }
}

/// Checks a lossless plan.
///
/// # Errors
///
/// [`EncodeError::ValueOutOfRange`] if `group_size_shift` is above
/// [`MAX_GROUP_SIZE_SHIFT`] — F.2's field is two bits wide, so a larger shift
/// has no encoding.
pub fn validate(plan: LosslessPlan) -> Result<ValidatedLosslessPlan> {
    if plan.group_size_shift > MAX_GROUP_SIZE_SHIFT {
        return Err(EncodeError::ValueOutOfRange {
            what: "group_size_shift",
            value: i64::from(plan.group_size_shift),
        });
    }
    Ok(ValidatedLosslessPlan(plan))
}

/// Chooses a plan for already-transformed `planes`.
///
/// **This is the policy half** — see the module docs. It reproduces the
/// choices the encoder has made since slice 10, exactly:
///
/// * the group size is the caller's, or [`DEFAULT_GROUP_SIZE_SHIFT`];
/// * three planes get the RCT, one does not (decided by the caller, which
///   applied it);
/// * the prefix code is the narrowest one that can carry the widest residual.
///
/// # Errors
///
/// Any error from [`FlatCode::for_max_value`] or [`validate`].
pub fn plan_for(
    planes: &[Plane],
    rct: bool,
    options: &EncodeOptions,
) -> Result<ValidatedLosslessPlan> {
    let group_size_shift = options.group_size_shift.unwrap_or(DEFAULT_GROUP_SIZE_SHIFT);
    let code = FlatCode::for_max_value(max_packed_residual(planes))?;
    validate(LosslessPlan {
        group_size_shift,
        rct,
        code,
    })
}

/// An upper bound on `PackSigned(sample - prediction)` over every plane.
///
/// H.3's gradient prediction is a clamp between two neighbours, so it lies
/// inside the plane's own value range everywhere except the first sample,
/// where the substitutions make it zero. The residual is therefore bounded by
/// the wider of the plane's span and its distance from zero.
#[must_use]
pub fn max_packed_residual(planes: &[Plane]) -> u32 {
    let mut worst = 0u32;
    for plane in planes {
        let (mut lo, mut hi) = (0i64, 0i64);
        for &s in plane {
            let s = i64::from(s);
            lo = lo.min(s);
            hi = hi.max(s);
        }
        let bound = (hi - lo).max(hi.abs()).max(lo.abs());
        let packed = crate::entropy::pack_signed(i32::try_from(bound).unwrap_or(i32::MAX));
        // PackSigned is not monotone in the sign, so both directions count.
        let packed = packed.max(crate::entropy::pack_signed(
            i32::try_from(-bound).unwrap_or(i32::MIN),
        ));
        worst = worst.max(packed);
    }
    worst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_residual_bound_covers_the_widest_plane() {
        assert_eq!(max_packed_residual(&[vec![0, 0, 0]]), 0);
        // PackSigned doubles a positive residual, so a span of n bounds at 2n.
        assert_eq!(max_packed_residual(&[vec![0, 255]]), 510);
        assert_eq!(max_packed_residual(&[vec![0, 65535]]), 131_070);
        // Negative chroma from the RCT widens the span in both directions.
        assert_eq!(max_packed_residual(&[vec![-300, 300]]), 1200);
    }

    #[test]
    fn a_plan_with_an_unencodable_group_size_is_rejected() {
        let plan = LosslessPlan {
            group_size_shift: 4,
            rct: false,
            code: FlatCode::new_const(4),
        };
        assert!(matches!(
            validate(plan),
            Err(EncodeError::ValueOutOfRange {
                what: "group_size_shift",
                ..
            })
        ));
    }

    #[test]
    fn the_planner_reproduces_the_encoders_historical_choices() {
        let planes = vec![vec![0i32, 255, 128, 7]];
        let plan = plan_for(&planes, false, &EncodeOptions::default()).expect("legal plan");
        assert_eq!(plan.plan().group_size_shift, DEFAULT_GROUP_SIZE_SHIFT);
        assert!(!plan.plan().rct);
        assert_eq!(
            plan.plan().code,
            FlatCode::for_max_value(510).expect("legal code")
        );

        let forced = EncodeOptions {
            group_size_shift: Some(0),
            ..EncodeOptions::default()
        };
        let plan = plan_for(&planes, true, &forced).expect("legal plan");
        assert_eq!(plan.plan().group_size_shift, 0);
        assert!(plan.plan().rct);
    }
}
