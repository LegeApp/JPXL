//! The crossing mathematics both ladder searches navigate by.
//!
//! The byte-rate loop ([`crate::rate`]) and the perceptual loop
//! ([`crate::quality`]) each used to carry a private copy of the same
//! log-domain arithmetic: fit a local straight line through two observed
//! points, solve it for where it crosses the target, convert that back to a
//! rung, and force the step to move. This module is the single copy (plan
//! phase N1, `JPXL/docs/scheduler-unification-and-multifidelity-plan.md`).
//!
//! # The coordinate
//!
//! `x = ln(effective_scale(rung))`, never the rung index. The index is not
//! proportional to the quantizer: below `MAX_GLOBAL_SCALE` a rung is a
//! `global_scale` step, above it each rung is an `HfMul` step worth a whole
//! `MAX_GLOBAL_SCALE` of the lower segment. Interpolating on the index aims
//! badly across exactly that kink. Effective scale is smooth and strictly
//! increasing over the whole ladder, and the curves are near power-law
//! against it — `global_scale` is a reciprocal, so equal *ratios* are the
//! equal steps.
//!
//! `y` is the controller's own quantity, and each one supplies its own
//! transform so that **`y` rises with `x`** for both:
//!
//! * rate: `y = ln(bytes)` — a finer quantizer is a bigger file;
//! * quality: `y = -ln(max(100 - score, LOSS_EPSILON))` — a finer quantizer
//!   is a smaller metric loss, so the negation puts the curve the same way up.
//!
//! That is the only reason the two can share this code. What a probe *means*,
//! and what makes a candidate acceptable, stays with the controller: this
//! module never learns what a byte or a score is.
//!
//! # What is deliberately not shared
//!
//! The two loops disagree, on purpose, about how hard to push a step that
//! did not move ([`Progress`]) and about which crossing form to solve
//! ([`LocalModel::crossing`] against [`bounded_fraction_crossing`]). Those
//! are parameters here rather than a harmonised single rule, because
//! harmonising them silently would change which rung gets probed and so
//! change output bytes. Guards — degenerate brackets, non-positive values,
//! curves that do not order — stay at the call sites, which have their own
//! notion of what makes an observation untrustworthy.

use crate::quantizer_ladder::{Rung, effective_scale, rung_for_effective_scale};

/// Natural log of a rung's effective scale: the `x` every fit works in.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    reason = "effective scales stay far inside f64's exact-integer range"
)]
pub fn ln_scale(rung: Rung) -> f64 {
    (effective_scale(rung) as f64).ln()
}

/// The rung nearest an effective scale, clamped into the ladder.
///
/// A non-finite scale is the floor: the caller asked for a point the curve
/// could not place, and the coarsest rung is the safe reading of that.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a finite positive scale is clamped before the narrowing"
)]
pub fn rung_for_scale(scale: f64) -> Rung {
    if !scale.is_finite() {
        return Rung::FLOOR;
    }
    rung_for_effective_scale(scale.round().clamp(1.0, u64::MAX as f64) as u64)
}

/// The slope of the straight line through two `(x, y)` observations.
///
/// Returns whatever the arithmetic gives, including a non-finite value: the
/// call sites differ in which slopes they are willing to trust, and each
/// keeps its own guard.
#[must_use]
pub fn fit_slope(lo: (f64, f64), hi: (f64, f64)) -> f64 {
    (hi.1 - lo.1) / (hi.0 - lo.0)
}

/// A local straight-line model of one controller's curve: the slope, and the
/// observed point it is anchored at.
///
/// Local because neither curve is a straight line over the whole ladder —
/// only over the span between nearby probes, which is all a step needs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalModel {
    /// `dy/dx`, positive for both controllers under their own transform.
    pub slope: f64,
    /// The `x` the model passes through.
    pub anchor_x: f64,
    /// The `y` the model passes through.
    pub anchor_y: f64,
}

impl LocalModel {
    /// A model of known slope through an observed point.
    #[must_use]
    pub const fn new(anchor: (f64, f64), slope: f64) -> Self {
        Self {
            slope,
            anchor_x: anchor.0,
            anchor_y: anchor.1,
        }
    }

    /// The model through two observations, anchored at the first.
    #[must_use]
    pub fn through(lo: (f64, f64), hi: (f64, f64)) -> Self {
        Self::new(lo, fit_slope(lo, hi))
    }

    /// Where the model reaches `target_y`, in `x`.
    #[must_use]
    pub fn crossing(&self, target_y: f64) -> f64 {
        self.anchor_x + (target_y - self.anchor_y) / self.slope
    }
}

/// The crossing by false position, with the interpolation fraction clamped
/// into the bracket.
///
/// The rate loop's bisection aims this way rather than through
/// [`LocalModel::crossing`]: inside a bracket that is known to straddle the
/// target, a fraction that leaves `[0, 1]` means the local line disagrees
/// with the bracket it was fitted to, and the bracket is the more trustworthy
/// of the two. Clamping keeps the aim inside and lets the caller do the
/// strict-interior clamp on rungs afterwards.
#[must_use]
pub fn bounded_fraction_crossing(lo: (f64, f64), hi: (f64, f64), target_y: f64) -> Option<f64> {
    let span = hi.1 - lo.1;
    if !span.is_finite() || span <= 0.0 {
        return None;
    }
    let fraction = ((target_y - lo.1) / span).clamp(0.0, 1.0);
    Some(lo.0 + fraction * (hi.0 - lo.0))
}

/// How hard a step insists on moving when the aim lands back where it
/// started.
///
/// The two loops answer this differently and are kept that way: forcing the
/// rate loop's rule onto the perceptual one (or the reverse) changes which
/// rung is probed, and so changes output bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Progress {
    /// At least one rung of movement in the requested direction, whatever the
    /// aim said. The rate loop's rule: its bracket phases terminate on a step
    /// count, so a step that stalls — or aims backwards across a
    /// representability gap — must still make ground.
    AtLeastOneRung,
    /// Move one rung only when the aim did not move at all. The perceptual
    /// loop's rule: an aim that lands short is still an informative probe,
    /// and its expansion is already bounded by a compounding margin.
    NudgeWhenStuck,
}

/// Applies `rule` to an aimed rung, given where the step started and which
/// way it meant to go.
#[must_use]
pub fn force_progress(aim: Rung, from: Rung, finer: bool, rule: Progress) -> Rung {
    match rule {
        Progress::AtLeastOneRung => {
            if finer {
                aim.max(Rung::new(from.get().saturating_add(1)))
            } else {
                aim.min(Rung::new(from.get().saturating_sub(1)))
            }
        }
        Progress::NudgeWhenStuck => {
            if aim != from {
                return aim;
            }
            if finer {
                Rung::new(from.get().saturating_add(1))
            } else {
                Rung::new(from.get().saturating_sub(1))
            }
        }
    }
}

/// A step of `ratio` in effective scale, finer or coarser, under `rule`.
#[must_use]
pub fn geometric_step(from: Rung, ratio: f64, finer: bool, rule: Progress) -> Rung {
    #[allow(
        clippy::cast_precision_loss,
        reason = "effective scales stay far inside f64's exact-integer range"
    )]
    let scale = effective_scale(from) as f64;
    let aimed = if finer { scale * ratio } else { scale / ratio };
    force_progress(rung_for_scale(aimed), from, finer, rule)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The crossing solves the line it was fitted to: put the target at an
    /// observed point and the model returns that point's `x`.
    #[test]
    fn a_crossing_at_an_observed_point_returns_that_point() {
        let lo = (ln_scale(Rung::new(1_000)), 10.0);
        let hi = (ln_scale(Rung::new(4_000)), 12.0);
        let model = LocalModel::through(lo, hi);
        assert!(model.slope > 0.0);
        assert!((model.crossing(lo.1) - lo.0).abs() < 1e-12);
        assert!((model.crossing(hi.1) - hi.0).abs() < 1e-9);
    }

    /// Negating `y` — which is exactly what the perceptual transform does to
    /// put its falling loss curve the same way up as the rate curve — leaves
    /// the crossing bit-identical. Both the slope and the numerator flip
    /// sign, and IEEE negation and division are sign-symmetric.
    #[test]
    fn negating_the_y_axis_does_not_move_the_crossing() {
        for (lo_y, hi_y, target) in [
            (2.0_f64, 5.5, 4.25),
            (-1.5, 3.25, 0.0),
            (0.125, 0.126, 0.1255),
            (11.7, 19.3, 30.0),
        ] {
            let (lo_x, hi_x) = (ln_scale(Rung::new(700)), ln_scale(Rung::new(9_000)));
            let rising = LocalModel::through((lo_x, lo_y), (hi_x, hi_y)).crossing(target);
            let falling = LocalModel::through((lo_x, -lo_y), (hi_x, -hi_y)).crossing(-target);
            assert_eq!(rising.to_bits(), falling.to_bits());
        }
    }

    /// False position stays inside the bracket even when the target lies
    /// outside the observed span.
    #[test]
    fn the_bounded_fraction_never_leaves_the_bracket() {
        let lo = (1.0, 10.0);
        let hi = (5.0, 20.0);
        assert_eq!(bounded_fraction_crossing(lo, hi, 10.0), Some(1.0));
        assert_eq!(bounded_fraction_crossing(lo, hi, 20.0), Some(5.0));
        assert_eq!(bounded_fraction_crossing(lo, hi, 15.0), Some(3.0));
        // Outside the span in either direction clamps to an endpoint.
        assert_eq!(bounded_fraction_crossing(lo, hi, -100.0), Some(1.0));
        assert_eq!(bounded_fraction_crossing(lo, hi, 100.0), Some(5.0));
        // A bracket whose value does not rise is not a slope.
        assert_eq!(bounded_fraction_crossing(lo, (5.0, 10.0), 15.0), None);
        assert_eq!(bounded_fraction_crossing(lo, (5.0, 5.0), 15.0), None);
    }

    /// The two progress rules differ exactly where they are meant to: on an
    /// aim that went the wrong way.
    #[test]
    fn the_progress_rules_differ_only_on_an_aim_that_did_not_advance() {
        let from = Rung::new(1_000);
        let forward = Rung::new(1_400);
        for rule in [Progress::AtLeastOneRung, Progress::NudgeWhenStuck] {
            assert_eq!(force_progress(forward, from, true, rule), forward);
        }
        // Stalled: both rules step one rung.
        assert_eq!(
            force_progress(from, from, true, Progress::AtLeastOneRung),
            Rung::new(1_001)
        );
        assert_eq!(
            force_progress(from, from, true, Progress::NudgeWhenStuck),
            Rung::new(1_001)
        );
        // Backwards: the rate rule drags it forward, the perceptual rule
        // takes the probe as aimed.
        let backwards = Rung::new(900);
        assert_eq!(
            force_progress(backwards, from, true, Progress::AtLeastOneRung),
            Rung::new(1_001)
        );
        assert_eq!(
            force_progress(backwards, from, true, Progress::NudgeWhenStuck),
            backwards
        );
    }

    /// A geometric step moves by the ratio in effective scale, not in rung
    /// index, and always moves.
    #[test]
    fn a_geometric_step_moves_in_effective_scale() {
        let from = Rung::for_global_scale(65_536);
        let up = geometric_step(from, 2.0, true, Progress::AtLeastOneRung);
        assert_eq!(effective_scale(from), 65_536);
        assert_eq!(effective_scale(up), 131_072);
        let down = geometric_step(from, 2.0, false, Progress::AtLeastOneRung);
        assert_eq!(effective_scale(down), 32_768);
        // At the ends the rule still forces a move rather than stalling.
        assert!(geometric_step(Rung::FLOOR, 2.0, false, Progress::AtLeastOneRung) == Rung::FLOOR);
        assert!(geometric_step(Rung::TOP, 2.0, true, Progress::AtLeastOneRung) == Rung::TOP);
    }

    /// A non-finite aim is the coarsest rung, not a panic or a wrap.
    #[test]
    fn a_scale_that_is_not_a_number_lands_on_the_floor() {
        assert_eq!(rung_for_scale(f64::NAN), Rung::FLOOR);
        assert_eq!(rung_for_scale(f64::INFINITY), Rung::FLOOR);
        assert_eq!(rung_for_scale(-1.0), Rung::FLOOR);
        assert_eq!(rung_for_scale(0.0), Rung::FLOOR);
        assert_eq!(rung_for_scale(1.0), Rung::FLOOR);
        assert_eq!(rung_for_scale(2.4), Rung::new(1));
    }
}
