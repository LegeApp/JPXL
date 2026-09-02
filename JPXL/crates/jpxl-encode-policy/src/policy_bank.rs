//! The bounded perceptual policy bank (PR 5, plan §6.2).
//!
//! A fixed-policy navigator finds the smallest stream *for one encoder
//! configuration* at a score. It cannot find the smallest legal stream,
//! because a different chroma allocation, LF/HF split, restoration, CfL or
//! truncation strength may cost fewer bytes for the same image at the same
//! perceptual score. This module names a small, ordered set of coherent
//! alternatives to the starting policy — never the Cartesian product — so the
//! quality controller can solve each to the same target and keep the cheapest.
//!
//! Each alternative moves exactly one axis away from the baseline (coordinate
//! descent), so a solved trial isolates that axis's byte effect. The axes, in
//! the order the prior evidence ranks them:
//!
//! 1. chroma QM `(x, b)` ∈ {(neutral,neutral), (3,3), (3,4), (3,5)};
//! 2. `quant_lf` ∈ {2, 3, 4, 6};
//! 3. restoration ∈ {EPF 1 Uniform7 (baseline), EPF 0, EPF 1 Zero};
//! 4. CfL on/off;
//! 5. `lambda_scale` ∈ {4.0 (baseline), 2.0, 8.0}.
//!
//! Variance AQ and the nearest/RDO quantizer are deliberately absent: both
//! measured negative in the AKR ledger and are not revived here.

use jpxl_encode::vardct::ids::{QmScale, QuantLf};

use crate::quality_features::SourceFeatures;
use crate::request::{ChromaHfPolicy, EncodeRequest, EpfSharpnessMode, RateSearchPreset};

/// One coherent encoder configuration the quality controller can solve to a
/// score and price exactly.
///
/// The seven fields are the perceptual axes PR 5 compares. `cfl` is applied at
/// the navigator (it selects fresh-CfL planning), not on the request; every
/// other field is written onto an [`EncodeRequest`] by [`Self::apply`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerceptualPolicy {
    /// F.2/I.5.3's X-channel quantization-matrix exponent.
    pub x_qm_scale: QmScale,
    /// F.2/I.5.3's B-channel quantization-matrix exponent.
    pub b_qm_scale: QmScale,
    /// I.2's `quant_lf` (the LF/HF split).
    pub quant_lf: QuantLf,
    /// J.1 EPF iteration count (`0..=3`).
    pub epf_iters: u8,
    /// Encoder policy for G.2.4's EPF sharpness plane.
    pub epf_sharpness: EpfSharpnessMode,
    /// Whether chroma-from-luma is estimated for this policy.
    pub cfl: bool,
    /// Multiplier on the cover/quantizer Lagrange weight (trailing truncation).
    pub lambda_scale: f32,
}

/// Which axis an alternative moves, used to rank it and to decide whether it
/// can reuse the baseline's structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PolicyAxis {
    /// Chroma QM `(x, b)`.
    Chroma,
    /// `quant_lf`.
    QuantLf,
    /// EPF iterations / sharpness.
    Restoration,
    /// Chroma-from-luma on/off.
    Cfl,
    /// Truncation `lambda_scale`.
    Lambda,
}

impl PolicyAxis {
    /// The prior-evidence rank of this axis (higher is tried first), used as
    /// the base priority before feature-driven boosts.
    const fn base_priority(self) -> i32 {
        match self {
            Self::Chroma => 50,
            Self::QuantLf => 40,
            Self::Restoration => 30,
            Self::Cfl => 20,
            Self::Lambda => 10,
        }
    }
}

impl PerceptualPolicy {
    /// Writes this policy's quantizer-and-restoration knobs onto a copy of
    /// `request`. `cfl` is not a request field, so it is carried on the policy
    /// and read by the navigator; everything else the planner reads from the
    /// request is set here. Chroma is pinned [`ChromaHfPolicy::Manual`] so no
    /// automatic (rate-path) branch is reachable.
    #[must_use]
    pub fn apply(&self, request: &EncodeRequest) -> EncodeRequest {
        let mut out = *request;
        out.chroma_hf_policy = ChromaHfPolicy::Manual;
        out.x_qm_scale = self.x_qm_scale;
        out.b_qm_scale = self.b_qm_scale;
        out.quant_lf = self.quant_lf;
        out.restoration.epf_iters = self.epf_iters;
        out.epf_sharpness = self.epf_sharpness;
        out.lambda_scale = self.lambda_scale;
        out
    }

    /// The starting policy of a preset: exactly the values
    /// [`EncodeRequest::for_quality`] pins, plus the navigator's per-preset CfL
    /// choice (Fast off, Balanced and Quality on). Kept in sync with
    /// `for_quality` by reading it back.
    #[must_use]
    pub fn baseline(preset: RateSearchPreset) -> Self {
        let req = EncodeRequest::for_quality(preset);
        Self {
            x_qm_scale: req.x_qm_scale,
            b_qm_scale: req.b_qm_scale,
            quant_lf: req.quant_lf,
            epf_iters: req.restoration.epf_iters,
            epf_sharpness: req.epf_sharpness,
            cfl: matches!(
                preset,
                RateSearchPreset::Balanced | RateSearchPreset::Quality
            ),
            lambda_scale: req.lambda_scale,
        }
    }

    /// Whether this policy can reuse `baseline`'s captured cover and CfL.
    ///
    /// A policy that moves only quantizer-side knobs (QM scales, `quant_lf`,
    /// `lambda_scale`) keeps the same cover and CfL as the baseline anchor, so
    /// the trial reuses [`crate::AnchorReuse::CoverAndCfl`]. A policy that
    /// moves CfL or restoration needs a fresh plan (a structural build).
    #[must_use]
    pub fn reuses_structure_of(&self, baseline: &Self) -> bool {
        self.cfl == baseline.cfl
            && self.epf_iters == baseline.epf_iters
            && self.epf_sharpness == baseline.epf_sharpness
    }
}

/// A quality-mode QM scale, falling back to neutral if the wire value is out of
/// range (it never is for the small constants used here).
fn qm(value: u8) -> QmScale {
    QmScale::new(value).unwrap_or(QmScale::NEUTRAL)
}

/// A `quant_lf`, falling back to the minimum if out of range (never here).
fn qlf(value: u32) -> QuantLf {
    QuantLf::new(value).unwrap_or(QuantLf::MIN)
}

/// The full ordered bank of single-axis alternatives to `baseline`, tagged with
/// the axis each moves. Every entry differs from `baseline` in exactly one
/// axis; entries equal to the baseline value are skipped, so the bank adapts to
/// each preset's own starting policy.
fn alternatives(baseline: &PerceptualPolicy) -> Vec<(PolicyAxis, PerceptualPolicy)> {
    let mut out: Vec<(PolicyAxis, PerceptualPolicy)> = Vec::new();

    // 1. Chroma QM (x, b): the four coherent allocations.
    for (x, b) in [
        (QmScale::NEUTRAL, QmScale::NEUTRAL),
        (qm(3), qm(3)),
        (qm(3), qm(4)),
        (qm(3), qm(5)),
    ] {
        if x != baseline.x_qm_scale || b != baseline.b_qm_scale {
            out.push((
                PolicyAxis::Chroma,
                PerceptualPolicy {
                    x_qm_scale: x,
                    b_qm_scale: b,
                    ..*baseline
                },
            ));
        }
    }

    // 2. quant_lf.
    for lf in [2u32, 3, 4, 6] {
        let q = qlf(lf);
        if q != baseline.quant_lf {
            out.push((
                PolicyAxis::QuantLf,
                PerceptualPolicy {
                    quant_lf: q,
                    ..*baseline
                },
            ));
        }
    }

    // 3. Restoration: EPF 0, then EPF 1 with the neutral (Zero) sharpness
    //    plane. The baseline (EPF 1 Uniform7) is not re-listed.
    for (iters, sharp) in [(0u8, EpfSharpnessMode::Zero), (1u8, EpfSharpnessMode::Zero)] {
        if iters != baseline.epf_iters || sharp != baseline.epf_sharpness {
            out.push((
                PolicyAxis::Restoration,
                PerceptualPolicy {
                    epf_iters: iters,
                    epf_sharpness: sharp,
                    ..*baseline
                },
            ));
        }
    }

    // 4. CfL flipped.
    out.push((
        PolicyAxis::Cfl,
        PerceptualPolicy {
            cfl: !baseline.cfl,
            ..*baseline
        },
    ));

    // 5. lambda_scale.
    for lambda in [2.0f32, 8.0] {
        if (lambda - baseline.lambda_scale).abs() > f32::EPSILON {
            out.push((
                PolicyAxis::Lambda,
                PerceptualPolicy {
                    lambda_scale: lambda,
                    ..*baseline
                },
            ));
        }
    }

    out
}

/// Orders the bank's alternatives by relevance to a source, most relevant
/// first, dropping the ones a source makes pointless.
///
/// The heuristic is deterministic:
///
/// * **Grayscale** drops every chroma alternative — a grayscale frame carries
///   no chroma to reallocate.
/// * A **high flat fraction** (`flat_fraction > 0.5`) boosts the `quant_lf`
///   and restoration axes: flat regions are where the LF/HF split and EPF pay.
/// * A **high edge proxy** (`edge_proxy > luma_variance_q50`, i.e. the busiest
///   tenth is busier than the median atom) boosts the chroma and `lambda`
///   axes: edges are where chroma bleed and terminal truncation matter.
///
/// Ties (and the no-feature case) fall back to the prior-evidence order above,
/// so the ranking is stable and reproducible for a given feature vector.
#[must_use]
pub fn rank_alternatives(
    features: &SourceFeatures,
    preset: RateSearchPreset,
) -> Vec<PerceptualPolicy> {
    let baseline = PerceptualPolicy::baseline(preset);
    let mut alts = alternatives(&baseline);
    if features.grayscale {
        alts.retain(|(axis, _)| *axis != PolicyAxis::Chroma);
    }

    let high_flat = features.flat_fraction > 0.5;
    let high_edge = features.edge_proxy > features.luma_variance_q50;

    let mut ranked: Vec<(i32, usize, PerceptualPolicy)> = alts
        .into_iter()
        .enumerate()
        .map(|(order, (axis, policy))| {
            // The boost must exceed the gap between adjacent base priorities
            // (10) so a feature actually reorders the axes rather than merely
            // tying them.
            let boost = match axis {
                PolicyAxis::QuantLf | PolicyAxis::Restoration if high_flat => 25,
                PolicyAxis::Chroma | PolicyAxis::Lambda if high_edge => 25,
                _ => 0,
            };
            (axis.base_priority() + boost, order, policy)
        })
        .collect();
    // Higher priority first; the canonical order breaks ties deterministically.
    ranked.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    ranked.into_iter().map(|(_, _, policy)| policy).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn features(grayscale: bool, flat: f32, edge: f32, median: f32) -> SourceFeatures {
        SourceFeatures {
            width: 256,
            height: 256,
            grayscale,
            luma_variance_q10: 1e-5,
            luma_variance_q50: median,
            luma_variance_q90: median + edge,
            chroma_variance_q50: 1e-4,
            flat_fraction: flat,
            edge_proxy: edge,
            preanalysis: None,
        }
    }

    #[test]
    fn baseline_matches_for_quality() {
        for preset in [
            RateSearchPreset::Fast,
            RateSearchPreset::Balanced,
            RateSearchPreset::Quality,
        ] {
            let base = PerceptualPolicy::baseline(preset);
            let req = EncodeRequest::for_quality(preset);
            assert_eq!(base.apply(&req), req, "{preset:?} baseline round-trips");
        }
    }

    #[test]
    fn every_alternative_moves_exactly_one_axis_from_the_baseline() {
        let baseline = PerceptualPolicy::baseline(RateSearchPreset::Balanced);
        for (axis, alt) in alternatives(&baseline) {
            assert_ne!(alt, baseline, "an alternative equals the baseline");
            let moved = usize::from(
                alt.x_qm_scale != baseline.x_qm_scale || alt.b_qm_scale != baseline.b_qm_scale,
            ) + usize::from(alt.quant_lf != baseline.quant_lf)
                + usize::from(
                    alt.epf_iters != baseline.epf_iters
                        || alt.epf_sharpness != baseline.epf_sharpness,
                )
                + usize::from(alt.cfl != baseline.cfl)
                + usize::from((alt.lambda_scale - baseline.lambda_scale).abs() > f32::EPSILON);
            assert_eq!(moved, 1, "axis {axis:?} moved {moved} groups");
            // Structure reuse is exactly the quantizer-side axes.
            let reuse = alt.reuses_structure_of(&baseline);
            let quantizer_side = matches!(
                axis,
                PolicyAxis::Chroma | PolicyAxis::QuantLf | PolicyAxis::Lambda
            );
            assert_eq!(reuse, quantizer_side, "axis {axis:?} reuse={reuse}");
        }
    }

    #[test]
    fn grayscale_skips_chroma_alternatives() {
        let gray = rank_alternatives(&features(true, 0.1, 1e-4, 1e-4), RateSearchPreset::Balanced);
        let baseline = PerceptualPolicy::baseline(RateSearchPreset::Balanced);
        for policy in &gray {
            assert!(
                policy.x_qm_scale == baseline.x_qm_scale
                    && policy.b_qm_scale == baseline.b_qm_scale,
                "a grayscale ranking kept a chroma alternative"
            );
        }
    }

    #[test]
    fn high_flat_prefers_quant_lf_or_restoration_first() {
        let ranked = rank_alternatives(
            &features(false, 0.9, 1e-6, 1e-3),
            RateSearchPreset::Balanced,
        );
        let baseline = PerceptualPolicy::baseline(RateSearchPreset::Balanced);
        let first = ranked.first().expect("a ranked alternative");
        let is_quant_lf = first.quant_lf != baseline.quant_lf;
        let is_restoration =
            first.epf_iters != baseline.epf_iters || first.epf_sharpness != baseline.epf_sharpness;
        assert!(
            is_quant_lf || is_restoration,
            "high-flat did not rank a quant_lf/restoration axis first: {first:?}"
        );
    }

    #[test]
    fn high_edge_prefers_chroma_first() {
        // High edge, low flat: chroma keeps its top base priority and the edge
        // boost keeps it ahead.
        let ranked = rank_alternatives(
            &features(false, 0.1, 1e-2, 1e-4),
            RateSearchPreset::Balanced,
        );
        let baseline = PerceptualPolicy::baseline(RateSearchPreset::Balanced);
        let first = ranked.first().expect("a ranked alternative");
        assert!(
            first.x_qm_scale != baseline.x_qm_scale || first.b_qm_scale != baseline.b_qm_scale,
            "high-edge did not rank a chroma axis first: {first:?}"
        );
    }

    #[test]
    fn ranking_is_deterministic() {
        let f = features(false, 0.3, 5e-4, 2e-4);
        let a = rank_alternatives(&f, RateSearchPreset::Balanced);
        let b = rank_alternatives(&f, RateSearchPreset::Balanced);
        assert_eq!(a, b);
    }
}
