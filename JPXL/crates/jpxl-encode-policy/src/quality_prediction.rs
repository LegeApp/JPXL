//! Shadow crossing predictor for the one-shot quality program (PR 3).
//!
//! Evaluates the generated [`crate::quality_predictor_v2`] model on a
//! frame's [`SourceFeatures`] and, after the exact controller has finished,
//! reconstructs the counterfactual one-shot decision for the
//! `jpxl.quality-trace/2` record. **Nothing here influences the search or
//! the emitted bytes** — the exact controller stays authoritative; this
//! module only says what a one-shot controller *would have done*, so the
//! offline evaluation can measure it.
//!
//! Determinism: scalar `f64` arithmetic in fixed order over generated
//! constants; no worker-count or SIMD dependence.

use crate::quality::{ProbeKind, QualityProbe};
use crate::quality_features::{SourceFeatures, TransformFeatureSummary};
use crate::quality_predictor_v2::{
    QPV2_FEATURE_CENTERS, QPV2_FEATURE_DIM, QPV2_FEATURE_SCALES, QPV2_FEATURE_SCHEMA, QPV2_KNOTS,
    QPV2_MODEL_VERSION, Qpv2Knot,
};
use crate::rate::{Rung, effective_scale, rung_for_effective_scale};

/// Matches the trainer's `EPS`: logs stay finite on zero-valued features.
const FEATURE_EPS: f64 = 1e-9;

/// Loss floor, as in the navigator's crossing interpolation.
const LOSS_EPSILON: f64 = 1e-3;

/// Calibrated-interval width (in `ln scale`) beyond which the shadow routes
/// to the exact controller (the memo's `ln(1.5)` threshold).
pub const FALLBACK_LOG_WIDTH: f64 = 0.405_465;

/// Saturation risk beyond which the shadow routes to the exact controller.
pub const FALLBACK_SATURATION_RISK: f64 = 0.5;

/// Margin the training-time standardized feature range is widened by before
/// a feature counts as out of distribution.
const OOD_RANGE_MARGIN: f64 = 0.25;

/// The model's declared domain floor: below this shortest side the metric
/// sits against its own floor and the shadow routes straight to the exact
/// controller. Mirrors the trainer's `MIN_DOMAIN_SIDE`.
pub const MIN_DOMAIN_SIDE: u32 = 128;

/// Runtime clamp of the predicted local loss exponent, mirroring the
/// navigator's own slope clamp.
const BETA_RANGE: (f64, f64) = (0.2, 3.0);

/// What the shadow model predicts for one `(frame, target)` request.
#[derive(Debug, Clone, PartialEq)]
pub struct QualityPredictionV2 {
    /// Median crossing rung.
    pub median_rung: Rung,
    /// Risk-adjusted candidate rung (rounded toward the finer legal rung).
    pub candidate_rung: Rung,
    /// Lower calibrated crossing-interval rung.
    pub interval_low: Rung,
    /// Upper calibrated crossing-interval rung.
    pub interval_high: Rung,
    /// Predicted local loss exponent, clamped into [`BETA_RANGE`].
    pub local_loss_exponent: f64,
    /// Predicted probability that the target saturates the ladder.
    pub saturation_risk: f64,
    /// Out-of-distribution flags that fired, by name.
    pub ood_flags: Vec<&'static str>,
    /// Why the shadow would route to the exact controller, if it would.
    pub fallback_reason: Option<&'static str>,
}

fn loss(score: f64) -> f64 {
    (100.0 - score).max(LOSS_EPSILON)
}

fn ln_eps(value: f32) -> f64 {
    (f64::from(value).max(0.0) + FEATURE_EPS).ln()
}

/// The `qpv2-st/1` feature vector of a frame: the nine source features
/// followed by the ten transform-summary fields in sorted key order. Must
/// mirror the trainer's `feature_vector(..., "source+transform")` exactly;
/// the schema string and the generated `QPV2_FEATURE_DIM` pin it — a
/// regenerated model with a different schema fails to compile here.
#[must_use]
pub fn qpv2_features(
    features: &SourceFeatures,
    transform: &TransformFeatureSummary,
) -> [f64; QPV2_FEATURE_DIM] {
    let width = f64::from(features.width.max(1));
    let height = f64::from(features.height.max(1));
    [
        ln_eps(features.luma_variance_q10),
        ln_eps(features.luma_variance_q50),
        ln_eps(features.luma_variance_q90),
        ln_eps(features.chroma_variance_q50),
        f64::from(features.flat_fraction),
        ln_eps(features.edge_proxy),
        (width * height).log2(),
        (width / height).ln(),
        if features.grayscale { 1.0 } else { 0.0 },
        transform.chroma_ac_ratio,
        transform.dc_variance_y,
        transform.directional_asymmetry,
        transform.high_low_ratio,
        transform.ln_ac_y_mean,
        transform.ln_ac_y_q50,
        transform.ln_ac_y_q90,
        transform.ln_ac_y_q99,
        transform.near_zero_frac_1e2,
        transform.near_zero_frac_1e3,
    ]
}

fn standardized(
    features: &SourceFeatures,
    transform: &TransformFeatureSummary,
) -> [f64; QPV2_FEATURE_DIM] {
    let raw = qpv2_features(features, transform);
    let mut out = [0.0; QPV2_FEATURE_DIM];
    for (slot, ((&value, &center), &scale)) in out.iter_mut().zip(
        raw.iter()
            .zip(QPV2_FEATURE_CENTERS.iter())
            .zip(QPV2_FEATURE_SCALES.iter()),
    ) {
        *slot = (value - center) / scale;
    }
    out
}

/// `[intercept, w_0, ..]` dot a standardized feature vector.
fn affine(weights: &[f64; QPV2_FEATURE_DIM + 1], z: &[f64; QPV2_FEATURE_DIM]) -> f64 {
    let mut acc = weights.first().copied().unwrap_or(0.0);
    for (&weight, &value) in weights.iter().skip(1).zip(z.iter()) {
        acc += weight * value;
    }
    acc
}

/// Per-knot raw outputs before monotone projection.
struct KnotEval {
    ln_loss: f64,
    lower: f64,
    median: f64,
    candidate: f64,
    upper: f64,
    ln_beta: f64,
    saturation_logit: f64,
}

fn evaluate_knot(knot: &Qpv2Knot, z: &[f64; QPV2_FEATURE_DIM]) -> KnotEval {
    KnotEval {
        ln_loss: loss(knot.target).ln(),
        lower: affine(&knot.lower, z),
        median: affine(&knot.median, z),
        candidate: affine(&knot.candidate, z),
        upper: affine(&knot.upper, z),
        ln_beta: affine(&knot.beta, z),
        saturation_logit: affine(&knot.saturation, z),
    }
}

/// Linear interpolation of `field` against `ln loss(target)`, clamped to the
/// end knots. Returns 0 only on an empty table, which the caller precludes.
fn interpolate(evals: &[KnotEval], target_ln_loss: f64, field: impl Fn(&KnotEval) -> f64) -> f64 {
    // Knots ascend in target, so their ln-loss descends.
    let (Some(first), Some(last)) = (evals.first(), evals.last()) else {
        return 0.0;
    };
    if target_ln_loss >= first.ln_loss {
        return field(first);
    }
    if target_ln_loss <= last.ln_loss {
        return field(last);
    }
    for pair in evals.windows(2) {
        let [a, b] = pair else { continue };
        if target_ln_loss <= a.ln_loss && target_ln_loss >= b.ln_loss {
            let span = a.ln_loss - b.ln_loss;
            if span <= f64::EPSILON {
                return field(a);
            }
            let t = (a.ln_loss - target_ln_loss) / span;
            return field(a) + t * (field(b) - field(a));
        }
    }
    field(last)
}

/// The finer legal rung at or above an effective scale.
fn rung_ceil(scale: f64) -> Rung {
    if !scale.is_finite() {
        return Rung::TOP;
    }
    let clamped = scale.clamp(1.0, 9.0e18);
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped positive and far inside u64 range"
    )]
    let floor = rung_for_effective_scale(clamped as u64);
    #[allow(
        clippy::cast_precision_loss,
        reason = "effective scales stay far inside f64's exact-integer range"
    )]
    if (effective_scale(floor) as f64) < scale {
        Rung::new(floor.get().saturating_add(1))
    } else {
        floor
    }
}

fn rung_floor(scale: f64) -> Rung {
    if !scale.is_finite() {
        return Rung::TOP;
    }
    let clamped = scale.clamp(1.0, 9.0e18);
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped positive and far inside u64 range"
    )]
    rung_for_effective_scale(clamped as u64)
}

/// Evaluates the generated model for one request. `None` when no model has
/// been generated (empty knot table).
#[must_use]
pub fn predict_v2(
    features: &SourceFeatures,
    transform: &TransformFeatureSummary,
    target: f64,
) -> Option<QualityPredictionV2> {
    if QPV2_KNOTS.is_empty() {
        return None;
    }
    let z = standardized(features, transform);

    let mut ood_flags: Vec<&'static str> = Vec::new();
    if features.width.min(features.height) < MIN_DOMAIN_SIDE {
        ood_flags.push("tiny_frame");
    }
    for (i, (&value, &(lo, hi))) in z
        .iter()
        .zip(crate::quality_predictor_v2::QPV2_FEATURE_Z_RANGE.iter())
        .enumerate()
    {
        let span = (hi - lo).max(1e-6);
        if value < lo - OOD_RANGE_MARGIN * span || value > hi + OOD_RANGE_MARGIN * span {
            ood_flags.push(qpv2_feature_name(i));
        }
    }

    // Monotone projection: a higher target must never predict a coarser
    // scale, per output. Knots ascend in target, so run a max-accumulate.
    let mut projected: Vec<KnotEval> = QPV2_KNOTS.iter().map(|k| evaluate_knot(k, &z)).collect();
    let mut running = (
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    for eval in &mut projected {
        running.0 = running.0.max(eval.lower);
        running.1 = running.1.max(eval.median);
        running.2 = running.2.max(eval.candidate);
        running.3 = running.3.max(eval.upper);
        eval.lower = running.0;
        eval.median = running.1;
        eval.candidate = running.2;
        eval.upper = running.3;
    }

    let x = loss(target).ln();
    let ln_lower = interpolate(&projected, x, |e| e.lower);
    let ln_median = interpolate(&projected, x, |e| e.median);
    let ln_candidate = interpolate(&projected, x, |e| e.candidate);
    let ln_upper = interpolate(&projected, x, |e| e.upper);
    let ln_beta = interpolate(&projected, x, |e| e.ln_beta);
    let saturation_logit = interpolate(&projected, x, |e| e.saturation_logit);
    let saturation_risk = 1.0 / (1.0 + (-saturation_logit.clamp(-30.0, 30.0)).exp());

    let interval_width = (ln_upper - ln_lower).max(0.0);
    let fallback_reason = if !ood_flags.is_empty() {
        Some("ood_feature")
    } else if interval_width > FALLBACK_LOG_WIDTH {
        Some("wide_interval")
    } else if saturation_risk > FALLBACK_SATURATION_RISK {
        Some("saturation_risk")
    } else {
        None
    };

    Some(QualityPredictionV2 {
        median_rung: rung_floor(ln_median.exp()),
        candidate_rung: rung_ceil(ln_candidate.exp()),
        interval_low: rung_floor(ln_lower.exp()),
        interval_high: rung_ceil(ln_upper.exp()),
        local_loss_exponent: ln_beta.exp().clamp(BETA_RANGE.0, BETA_RANGE.1),
        saturation_risk,
        ood_flags,
        fallback_reason,
    })
}

/// The schema name of feature `index`, for OOD flags.
#[must_use]
pub const fn qpv2_feature_name(index: usize) -> &'static str {
    match index {
        0 => "ln_luma_q10",
        1 => "ln_luma_q50",
        2 => "ln_luma_q90",
        3 => "ln_chroma_q50",
        4 => "flat_fraction",
        5 => "ln_edge_proxy",
        6 => "log2_pixels",
        7 => "ln_aspect",
        8 => "grayscale",
        9 => "chroma_ac_ratio",
        10 => "dc_variance_y",
        11 => "directional_asymmetry",
        12 => "high_low_ratio",
        13 => "ln_ac_y_mean",
        14 => "ln_ac_y_q50",
        15 => "ln_ac_y_q90",
        16 => "ln_ac_y_q99",
        17 => "near_zero_frac_1e2",
        18 => "near_zero_frac_1e3",
        _ => "unknown",
    }
}

/// The counterfactual one-shot record of a finished exact search.
///
/// The exact controller's probes bound what the one-shot path would have
/// seen: above the coarsest canonically feasible probe the score curve is
/// assumed monotone (the same assumption the navigator's bracketing makes),
/// so a candidate at or finer than it would have met the target with its
/// first plan. `first_observed_score` is the score of the probe nearest the
/// candidate in log scale — a proxy, since the search did not necessarily
/// probe the candidate rung itself; the offline oracle evaluation measures
/// the exact counterfactual.
#[must_use]
pub fn shadow_prediction_trace(
    features: &SourceFeatures,
    transform: &TransformFeatureSummary,
    target: f64,
    probes: &[QualityProbe],
) -> Option<crate::quality::QualityPredictionTrace> {
    let prediction = predict_v2(features, transform, target)?;
    let pixel_probes: Vec<&QualityProbe> = probes
        .iter()
        .filter(|p| p.kind == ProbeKind::Pixel)
        .collect();
    let coarsest_feasible = pixel_probes
        .iter()
        .filter(|p| p.feasible == Some(true))
        .map(|p| p.effective_scale)
        .min();
    let candidate_scale = effective_scale(prediction.candidate_rung);
    #[allow(
        clippy::cast_precision_loss,
        reason = "effective scales stay far inside f64's exact-integer range"
    )]
    let nearest = pixel_probes.iter().min_by(|a, b| {
        let da = ((a.effective_scale as f64).ln() - (candidate_scale as f64).ln()).abs();
        let db = ((b.effective_scale as f64).ln() - (candidate_scale as f64).ln()).abs();
        da.total_cmp(&db)
    });
    let first_observed_score = nearest.and_then(|p| p.score);

    // One slope correction from the proxy observation, aimed at the target.
    #[allow(
        clippy::cast_precision_loss,
        reason = "effective scales stay far inside f64's exact-integer range"
    )]
    let correction_scale = first_observed_score.map(|observed| {
        let beta = prediction.local_loss_exponent;
        let shift = (loss(observed).ln() - loss(target).ln()) / beta;
        ((candidate_scale as f64).ln() + shift).exp()
    });
    let correction_rung = correction_scale.map(rung_ceil);

    let would_one_shot = coarsest_feasible.is_some_and(|feasible| candidate_scale >= feasible);
    let would_correct = correction_rung
        .map(effective_scale)
        .zip(coarsest_feasible)
        .is_some_and(|(corrected, feasible)| corrected >= feasible);
    let decision_path = if prediction.fallback_reason.is_some() {
        "fallback_exact"
    } else if would_one_shot {
        "one_shot"
    } else if would_correct {
        "corrected"
    } else {
        "fallback_exact"
    };

    Some(crate::quality::QualityPredictionTrace {
        model_version: QPV2_MODEL_VERSION,
        feature_schema: QPV2_FEATURE_SCHEMA,
        median_rung: prediction.median_rung.get(),
        candidate_rung: prediction.candidate_rung.get(),
        interval_low: prediction.interval_low.get(),
        interval_high: prediction.interval_high.get(),
        local_loss_exponent: prediction.local_loss_exponent,
        saturation_risk: prediction.saturation_risk,
        ood_flags: prediction.ood_flags,
        fallback_reason: prediction.fallback_reason,
        first_observed_score,
        correction_rung: correction_rung.map(Rung::get),
        decision_path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn features() -> SourceFeatures {
        SourceFeatures {
            width: 640,
            height: 480,
            grayscale: false,
            luma_variance_q10: 1e-6,
            luma_variance_q50: 1e-4,
            luma_variance_q90: 1e-2,
            chroma_variance_q50: 1e-5,
            flat_fraction: 0.1,
            edge_proxy: 1e-2,
        }
    }

    fn transform() -> TransformFeatureSummary {
        TransformFeatureSummary {
            blocks: 4800,
            ln_ac_y_mean: -5.0,
            ln_ac_y_q50: -6.0,
            ln_ac_y_q90: -3.0,
            ln_ac_y_q99: -2.0,
            high_low_ratio: 0.2,
            directional_asymmetry: 0.3,
            chroma_ac_ratio: 0.15,
            near_zero_frac_1e3: 0.6,
            near_zero_frac_1e2: 0.8,
            dc_variance_y: 0.02,
        }
    }

    #[test]
    fn feature_vector_matches_the_schema_order() {
        let raw = qpv2_features(&features(), &transform());
        assert!((raw[0] - (f64::from(1e-6f32) + FEATURE_EPS).ln()).abs() < 1e-12);
        assert!((raw[6] - (640.0f64 * 480.0).log2()).abs() < 1e-12);
        assert!((raw[7] - (640.0f64 / 480.0).ln()).abs() < 1e-12);
        assert!(raw[8].abs() < f64::EPSILON);
        // Transform fields follow in sorted key order.
        assert!((raw[9] - 0.15).abs() < 1e-12, "chroma_ac_ratio first");
        assert!((raw[13] - (-5.0)).abs() < 1e-12, "ln_ac_y_mean fifth");
        assert!((raw[18] - 0.6).abs() < 1e-12, "near_zero_frac_1e3 last");
    }

    #[test]
    fn predictions_are_monotone_in_the_target() {
        // With any generated model, a higher target must never predict a
        // coarser candidate scale.
        let f = features();
        let t = transform();
        let mut previous = 0u64;
        for target in [30.0, 50.0, 70.0, 80.0, 85.0, 90.0, 95.0] {
            let Some(p) = predict_v2(&f, &t, target) else {
                return; // no generated model in this build
            };
            let scale = effective_scale(p.candidate_rung);
            assert!(
                scale >= previous,
                "candidate scale fell from {previous} to {scale} at target {target}"
            );
            previous = scale;
        }
    }

    #[test]
    fn rung_ceil_rounds_toward_the_finer_rung() {
        let r = rung_ceil(1000.5);
        assert!(effective_scale(r) >= 1001);
        let exact = rung_ceil(1000.0);
        assert_eq!(effective_scale(exact), 1000);
    }
}
