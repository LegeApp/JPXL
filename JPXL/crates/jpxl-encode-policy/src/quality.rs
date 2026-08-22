//! The perceptual quality controller: the smallest exact stream whose score
//! meets a requested minimum.
//!
//! This is the score-targeted sibling of the rate controller in `rate.rs`,
//! built from the same parts — the effective-scale ladder, a log-log crossing
//! fit, anchored structure reuse and bounded corrections — with the observable
//! changed from exact bytes to the metric's loss and the constraint inverted
//! from a ceiling to a floor:
//!
//! ```text
//! predict a rung from the source features
//!   → pixel-probe it (plan pixels, render, score; no entropy)
//!   → step geometrically until one probe meets the target and one does not
//!   → aim at the crossing of log(100 − score) against log(effective scale),
//!     with a small reserve above the target, and probe it (one correction)
//!   → attach entropy and price exactly the coarsest feasible probes
//!   → emit the smallest exact stream whose score meets the target
//! ```
//!
//! Every probe is a full-frame score of reconstructed pixels; nothing here
//! infers a score from a rate. Budgets are hard: a production effort stops at
//! its probe and price caps and reports what it could verify. The target is a
//! floor — a stream is never emitted below it unless the finest quantizer
//! cannot reach it, and then it says so ([`QualityStatus::SaturatedTop`]).

use std::time::Instant;

use crate::candidate::CandidateSearchContext;
use crate::error::{PolicyError, Result};
use crate::quality_features::{SourceFeatures, source_features};
use crate::quality_predictor::{
    FALLBACK_LOG_FIT, FLAT_BUCKET_EDGES, INITIAL_RUNG_TABLE, LUMA_BUCKET_EDGES,
};
use crate::rate::{QuantizerChoice, Rung, effective_scale, rung_for_effective_scale};
use crate::request::{EncodeRequest, PerceptualTarget, RateSearchPreset};
use crate::{AnalysisAtlas, AnchorReuse, EntropySearch, PreparedFrame, StructuralAnchor};
use jpxl_encode::vardct::{
    CodestreamSizing, ValidatedEmissionPlan, ValidatedPixelPlan, VardctGeometry,
    emit_codestream_with_executor,
};

/// What one scored candidate reported.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerceptualObservation {
    /// The metric's score for the candidate against the source.
    pub score: f64,
}

/// Scores a candidate's pixels against the source.
///
/// Implemented outside this crate (by `jpxl-perceptual`, over the plan
/// renderer), so the policy layer never links a metric or a renderer. The
/// implementation must be deterministic: the same plan scores identically
/// under any worker count, or the selected stream would depend on the host.
pub trait PerceptualEvaluator {
    /// Reconstructs and scores `candidate`.
    ///
    /// # Errors
    ///
    /// Whatever the renderer or metric refuses.
    fn evaluate(&mut self, candidate: &ValidatedPixelPlan) -> Result<PerceptualObservation>;

    /// The pinned metric identity the scores come from.
    fn metric_version(&self) -> &'static str;
}

/// Score the selected stream must exceed the target by before it counts as
/// feasible. The in-tree metric is bit-reproducible across worker counts and
/// uses host-independent arithmetic, so the measured platform variation is
/// zero; the guard stays a named constant so a future SIMD navigation mode
/// has somewhere to put its measured spread.
pub const DEFAULT_SCORE_GUARD: f64 = 0.0;

/// Effective-scale ratio of one blind bracket-expansion step (used only when
/// no slope can be measured yet).
pub const BRACKET_RATIO: f64 = 1.8;

/// Prior exponent of `loss ∝ effective_scale^-α` used to aim the second probe
/// from the first one's loss alone.
pub const PRIOR_LOSS_EXPONENT: f64 = 0.9;

/// Factor by which an expansion step aims past the estimated crossing, so
/// the next probe lands on the other side of the target; it compounds with
/// every further expansion so a flat curve still brackets.
pub const EXPANSION_MARGIN: f64 = 1.25;

/// Largest effective-scale ratio one expansion step may jump.
pub const MAX_EXPANSION_JUMP: f64 = 16.0;

/// Effective-scale distance beyond which a finalist rebuilds its cover and
/// CfL instead of reusing the first probe's (the rate controller measured
/// 13–18% worse bytes on reused structure at 1.8–3.7x).
pub const STRUCTURE_REBUILD_RATIO: f64 = 1.8;

/// Floor of the metric loss used for interpolation, so a perfect score still
/// has a finite logarithm.
pub const LOSS_EPSILON: f64 = 1e-3;

/// Overshoot (achieved − requested) below which the result counts as
/// [`QualityStatus::Met`] rather than work-capped.
pub const MET_OVERSHOOT_BAND: f64 = 1.0;

/// Smallest score margin the crossing aims above the target, whatever the
/// effort's loss-relative reserve works out to.
pub const MIN_AIM_MARGIN: f64 = 0.25;

/// Hard work caps of one effort.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualityBudget {
    /// Full-frame render-and-score evaluations.
    pub pixel_probes: u32,
    /// Entropy trainings followed by an exact emission.
    pub exact_prices: u32,
    /// Fresh cover/CfL builds (the first probe is one).
    pub structural_builds: u32,
    /// Fraction of the target loss the crossing aims above the target, so a
    /// slightly optimistic interpolation still lands feasible. Small on
    /// purpose: the rate controller's equivalent is an eighth of its 2-3%
    /// tolerance band, and aiming a whole point high costs bytes on every
    /// encode.
    pub reserve: f64,
}

impl QualityBudget {
    /// The budget of a preset, as the controller plan states them.
    #[must_use]
    pub const fn for_preset(preset: RateSearchPreset) -> Self {
        match preset {
            RateSearchPreset::Fast => Self {
                pixel_probes: 3,
                exact_prices: 2,
                structural_builds: 2,
                reserve: 0.06,
            },
            RateSearchPreset::Balanced => Self {
                pixel_probes: 5,
                exact_prices: 3,
                structural_builds: 2,
                reserve: 0.03,
            },
            RateSearchPreset::Quality => Self {
                pixel_probes: 10,
                exact_prices: 4,
                structural_builds: 3,
                reserve: 0.02,
            },
        }
    }
}

/// Why a completed score-targeted search stopped where it did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityStatus {
    /// The selected stream meets the target within [`MET_OVERSHOOT_BAND`].
    Met,
    /// The selected stream meets the target and its next coarser rung was
    /// verified not to, so the overshoot is forced by the ladder's step.
    MetAdjacentRungs,
    /// The selected stream meets the target but the budget ran out before
    /// the overshoot could be tightened.
    MetWorkCap,
    /// Even the coarsest quantizer meets the target; the coarsest was chosen.
    SaturatedFloor,
    /// Even the finest quantizer misses the target; the finest verified
    /// stream was chosen and `saturated` is set.
    SaturatedTop,
    /// The probe budget (plus its one rescue probe) ran out before any
    /// candidate met the target; the finest verified stream was chosen and
    /// the reported score is below the target.
    UnderTargetWorkCap,
    /// A fresh cover/CfL build supplied the selected stream.
    RescuedFreshStructure,
}

/// What kind of work a trace entry records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeKind {
    /// Pixels planned, rendered and scored; no entropy.
    Pixel,
    /// Entropy trained and the stream emitted exactly.
    Exact,
}

/// Whether a candidate's cover and CfL were planned fresh or reused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructureSource {
    /// Cover and CfL planned for this quantizer.
    Fresh,
    /// Cover and CfL reused from the first probe, `HfMul` retargeted.
    Reused,
}

/// One unit of controller work, in order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualityProbe {
    /// What was done.
    pub kind: ProbeKind,
    /// The quantizer it was done at.
    pub quantizer: QuantizerChoice,
    /// `global_scale * HfMul` of that quantizer.
    pub effective_scale: u64,
    /// The score, for pixel probes and for exact prices of scored pixels.
    pub score: Option<f64>,
    /// The exact bytes, for exact prices.
    pub bytes: Option<u64>,
    /// Where the candidate's structure came from.
    pub structure: StructureSource,
    /// Whether the score met the target plus guard.
    pub feasible: Option<bool>,
    /// Wall time of this unit, in milliseconds.
    pub millis: u64,
}

/// Work counters and timings of one search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct QualityStats {
    /// Pixel probes spent (including finalist re-scores).
    pub pixel_probes: u32,
    /// Exact prices spent.
    pub exact_prices: u32,
    /// Fresh cover/CfL builds.
    pub structural_builds: u32,
    /// Where the predictor started the search.
    pub predicted: Option<Rung>,
    /// The final bracket `(infeasible, feasible)` if one was found.
    pub bracket: Option<(Rung, Rung)>,
    /// Milliseconds in pixel planning.
    pub plan_ms: u64,
    /// Milliseconds in rendering and scoring.
    pub render_metric_ms: u64,
    /// Milliseconds in entropy training.
    pub entropy_ms: u64,
    /// Milliseconds in exact emission.
    pub emit_ms: u64,
}

/// What a completed score-targeted search chose.
#[derive(Debug, Clone)]
pub struct QualityOutcome {
    /// The codestream at the chosen quantizer, emitted once.
    pub codestream: Vec<u8>,
    /// The validated plan it came from.
    pub plan: ValidatedEmissionPlan,
    /// Its exact accounting.
    pub sizing: CodestreamSizing,
    /// The quantizer chosen.
    pub chosen: QuantizerChoice,
    /// The score the caller asked for.
    pub requested_score: f64,
    /// The score the chosen stream's pixels achieve.
    pub achieved_score: f64,
    /// The guard the feasibility test added to the target.
    pub guard: f64,
    /// Whether the finest quantizer still missed the target.
    pub saturated: bool,
    /// Explicit terminal state.
    pub status: QualityStatus,
    /// Every unit of work, in order.
    pub trace: Vec<QualityProbe>,
    /// Counters and timings.
    pub stats: QualityStats,
    /// The source features the prediction was made from.
    pub features: SourceFeatures,
    /// The metric the scores come from.
    pub metric_version: &'static str,
}

impl QualityOutcome {
    /// Achieved minus requested score: the reserve a finalist reducer could
    /// exchange for bytes.
    #[must_use]
    pub fn reserve(&self) -> f64 {
        self.achieved_score - self.requested_score
    }

    /// The `jpxl.quality-trace/1` record of this search, one JSON line.
    #[must_use]
    pub fn trace_json(&self, effort: &str) -> String {
        let probes: Vec<String> = self
            .trace
            .iter()
            .map(|p| {
                format!(
                    "{{\"kind\":\"{}\",\"rung\":{},\"global_scale\":{},\"hf_mul\":{},\"effective_scale\":{},\
                     \"score\":{},\"bytes\":{},\"structure\":\"{}\",\"feasible\":{},\"millis\":{}}}",
                    match p.kind {
                        ProbeKind::Pixel => "pixel",
                        ProbeKind::Exact => "exact",
                    },
                    p.quantizer.rung.get(),
                    p.quantizer.global_scale.get(),
                    p.quantizer.hf_mul.get(),
                    p.effective_scale,
                    p.score.map_or_else(|| "null".to_owned(), |s| format!("{s}")),
                    p.bytes.map_or_else(|| "null".to_owned(), |b| format!("{b}")),
                    match p.structure {
                        StructureSource::Fresh => "fresh",
                        StructureSource::Reused => "reuse",
                    },
                    p.feasible.map_or_else(|| "null".to_owned(), |f| format!("{f}")),
                    p.millis,
                )
            })
            .collect();
        let bracket = self.stats.bracket.map_or_else(
            || "null".to_owned(),
            |(lo, hi)| format!("[{},{}]", lo.get(), hi.get()),
        );
        format!(
            "{{\"schema\":\"jpxl.quality-trace/1\",\"metric_version\":\"{}\",\"score_guard\":{},\
             \"effort\":\"{}\",\"source_features\":{},\"predicted_rung\":{},\"bracket\":{},\
             \"pixel_probes\":{},\"exact_prices\":{},\"structural_builds\":{},\
             \"requested_score\":{},\"achieved_score\":{},\"guard_margin\":{},\"final_exact_bytes\":{},\
             \"status\":\"{}\",\"saturated\":{},\"wall_by_phase\":{{\"plan\":{},\"render_metric\":{},\
             \"entropy\":{},\"emit\":{}}},\"probes\":[{}]}}",
            self.metric_version,
            self.guard,
            effort,
            self.features.to_json(),
            self.stats
                .predicted
                .map_or_else(|| "null".to_owned(), |r| format!("{}", r.get())),
            bracket,
            self.stats.pixel_probes,
            self.stats.exact_prices,
            self.stats.structural_builds,
            self.requested_score,
            self.achieved_score,
            self.achieved_score - self.requested_score - self.guard,
            self.sizing.total,
            status_name(self.status),
            self.saturated,
            self.stats.plan_ms,
            self.stats.render_metric_ms,
            self.stats.entropy_ms,
            self.stats.emit_ms,
            probes.join(","),
        )
    }
}

/// The snake_case name of a status, for traces and the CLI.
#[must_use]
pub const fn status_name(status: QualityStatus) -> &'static str {
    match status {
        QualityStatus::Met => "met",
        QualityStatus::MetAdjacentRungs => "met_adjacent_rungs",
        QualityStatus::MetWorkCap => "met_work_cap",
        QualityStatus::SaturatedFloor => "saturated_floor",
        QualityStatus::SaturatedTop => "saturated_top",
        QualityStatus::UnderTargetWorkCap => "under_target_work_cap",
        QualityStatus::RescuedFreshStructure => "rescued_fresh_structure",
    }
}

/// The metric loss of a score, floored so its logarithm is finite.
fn loss(score: f64) -> f64 {
    (100.0 - score).max(LOSS_EPSILON)
}

/// Natural log of a rung's effective scale.
#[allow(
    clippy::cast_precision_loss,
    reason = "effective scales stay far inside f64's exact-integer range"
)]
fn ln_scale(rung: Rung) -> f64 {
    (effective_scale(rung) as f64).ln()
}

/// The rung nearest an effective scale, clamped into the ladder.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a finite positive scale is clamped before the narrowing"
)]
fn rung_for_scale(scale: f64) -> Rung {
    if !scale.is_finite() {
        return Rung::FLOOR;
    }
    rung_for_effective_scale(scale.round().clamp(1.0, u64::MAX as f64) as u64)
}

/// Where the crossing of `log(loss)` against `log(effective scale)` with the
/// target lies, between a coarser infeasible point and a finer feasible one.
///
/// `None` when the two points do not order (the loss did not fall with the
/// finer quantizer) or the bracket has no rung strictly inside it.
pub(crate) fn log_loss_crossing(
    infeasible: (Rung, f64),
    feasible: (Rung, f64),
    aim_score: f64,
) -> Option<Rung> {
    let (lo_rung, lo_score) = infeasible;
    let (hi_rung, hi_score) = feasible;
    if hi_rung.get() <= lo_rung.get().saturating_add(1) {
        return None;
    }
    let (x_lo, x_hi) = (ln_scale(lo_rung), ln_scale(hi_rung));
    let (y_lo, y_hi) = (loss(lo_score).ln(), loss(hi_score).ln());
    if y_hi >= y_lo || x_hi <= x_lo {
        return None;
    }
    let slope = (y_hi - y_lo) / (x_hi - x_lo);
    let x = x_lo + (loss(aim_score).ln() - y_lo) / slope;
    let rung = rung_for_scale(x.exp());
    // Keep the aim strictly inside the bracket so every probe is informative.
    let inner_lo = Rung::new(lo_rung.get() + 1);
    let inner_hi = Rung::new(hi_rung.get() - 1);
    Some(rung.clamp(inner_lo, inner_hi))
}

/// One geometric step of [`BRACKET_RATIO`] in effective scale, finer or
/// coarser, clamped into the ladder. Returns the same rung only at the
/// ladder's end.
pub(crate) fn geometric_step(from: Rung, finer: bool) -> Rung {
    let scale = ln_scale(from).exp();
    let next = if finer {
        scale * BRACKET_RATIO
    } else {
        scale / BRACKET_RATIO
    };
    let rung = rung_for_scale(next);
    if rung == from {
        if finer {
            Rung::new(from.get().saturating_add(1))
        } else {
            Rung::new(from.get().saturating_sub(1))
        }
    } else {
        rung
    }
}

/// Whether a finalist at `rung` is too far from the structure anchor to
/// reuse its cover and CfL.
fn structure_is_far(anchor: Rung, rung: Rung) -> bool {
    (ln_scale(rung) - ln_scale(anchor)).abs() > STRUCTURE_REBUILD_RATIO.ln()
}

/// Bucket index of `value` against ascending upper edges.
fn bucket(value: f32, edges: &[f32]) -> u8 {
    let index = edges.iter().take_while(|&&edge| value >= edge).count();
    u8::try_from(index).unwrap_or(u8::MAX)
}

/// The effective scale the calibration predicts for `target` on a source
/// with `features`: the table entry (interpolated in target) when one
/// exists for the source's bucket, else the fallback power law.
#[must_use]
pub fn predicted_effective_scale(features: &SourceFeatures, target: f64) -> f64 {
    let luma_bucket = bucket(features.luma_variance_q50, &LUMA_BUCKET_EDGES);
    let flat_bucket = bucket(features.flat_fraction, &FLAT_BUCKET_EDGES);
    let target_loss = loss(target).ln();
    let mut entries: Vec<(f64, f64)> = INITIAL_RUNG_TABLE
        .iter()
        .filter(|e| e.luma_bucket == luma_bucket && e.flat_bucket == flat_bucket)
        .map(|e| {
            (
                loss(f64::from(e.target)).ln(),
                f64::from(e.global_scale.max(1)).ln(),
            )
        })
        .collect();
    entries.sort_by(|a, b| a.0.total_cmp(&b.0));
    if entries.len() >= 2 {
        // Log-linear interpolation in log-loss; extrapolate with the end
        // segments.
        let (below, above) = entries
            .iter()
            .zip(entries.iter().skip(1))
            .find(|(a, b)| a.0 <= target_loss && target_loss <= b.0)
            .map_or_else(
                || {
                    if target_loss < entries.first().map_or(0.0, |e| e.0) {
                        (entries.first().copied(), entries.get(1).copied())
                    } else {
                        (
                            entries.get(entries.len() - 2).copied(),
                            entries.last().copied(),
                        )
                    }
                },
                |(a, b)| (Some(*a), Some(*b)),
            );
        if let (Some(a), Some(b)) = (below, above)
            && (b.0 - a.0).abs() > f64::EPSILON
        {
            let t = (target_loss - a.0) / (b.0 - a.0);
            return (a.1 + t * (b.1 - a.1)).exp();
        }
    }
    if let Some(entry) = entries.first() {
        return entry.1.exp();
    }
    let [a, b, c, d] = FALLBACK_LOG_FIT;
    (a + b * target_loss
        + c * (f64::from(features.luma_variance_q50) + 1e-6).ln()
        + d * f64::from(features.flat_fraction))
    .exp()
}

/// One pixel probe's retained result.
struct ProbeRecord {
    rung: Rung,
    quantizer: QuantizerChoice,
    score: f64,
    feasible: bool,
    structure: StructureSource,
    pixels: Option<(ValidatedPixelPlan, VardctGeometry)>,
}

/// The search's running state.
struct Navigator<'c, 'a, 'e> {
    ctx: &'c mut CandidateSearchContext<'a>,
    evaluator: &'e mut dyn PerceptualEvaluator,
    target: f64,
    guard: f64,
    budget: QualityBudget,
    enable_cfl: bool,
    structure_tier: EntropySearch,
    finalist_entropy: EntropySearch,
    anchor: Option<StructuralAnchor>,
    anchor_rung: Option<Rung>,
    probes: Vec<ProbeRecord>,
    trace: Vec<QualityProbe>,
    stats: QualityStats,
}

impl Navigator<'_, '_, '_> {
    fn threshold(&self) -> f64 {
        self.target + self.guard
    }

    fn pixel_budget_left(&self) -> bool {
        self.stats.pixel_probes < self.budget.pixel_probes
    }

    fn already_probed(&self, rung: Rung) -> bool {
        self.probes.iter().any(|p| p.rung == rung)
    }

    /// Plans, renders and scores one rung. `fresh` builds cover and CfL for
    /// this rung (and makes it the structure anchor when none exists yet);
    /// otherwise the anchor's structure is reused.
    fn probe(&mut self, rung: Rung, fresh: bool) -> Result<usize> {
        let quantizer = QuantizerChoice::at(rung, self.ctx.request().quant_lf)?;
        let plan_start = Instant::now();
        let (pixels, geometry, structure) = if fresh || self.anchor.is_none() {
            let mut captured = None;
            let planned = self.ctx.pixel_plan(
                quantizer,
                self.enable_cfl,
                self.structure_tier,
                AnchorReuse::None,
                Some(&mut captured),
            )?;
            self.stats.structural_builds = self.stats.structural_builds.saturating_add(1);
            if self.anchor.is_none() {
                self.anchor = captured;
                self.anchor_rung = Some(rung);
            }
            (planned.0, planned.1, StructureSource::Fresh)
        } else {
            let anchor = self.anchor.as_ref().ok_or(PolicyError::Unsupported {
                what: "a reused structure before any probe captured one",
            })?;
            let planned = self.ctx.pixel_plan(
                quantizer,
                false,
                self.structure_tier,
                AnchorReuse::CoverAndCfl(anchor),
                None,
            )?;
            (planned.0, planned.1, StructureSource::Reused)
        };
        self.stats.plan_ms = self
            .stats
            .plan_ms
            .saturating_add(u64::try_from(plan_start.elapsed().as_millis()).unwrap_or(u64::MAX));

        let score_start = Instant::now();
        let score = self.evaluator.evaluate(&pixels)?.score;
        let millis = u64::try_from(score_start.elapsed().as_millis()).unwrap_or(u64::MAX);
        self.stats.render_metric_ms = self.stats.render_metric_ms.saturating_add(millis);
        self.stats.pixel_probes = self.stats.pixel_probes.saturating_add(1);
        let feasible = score >= self.threshold();
        self.trace.push(QualityProbe {
            kind: ProbeKind::Pixel,
            quantizer,
            effective_scale: effective_scale(rung),
            score: Some(score),
            bytes: None,
            structure,
            feasible: Some(feasible),
            millis,
        });
        self.probes.push(ProbeRecord {
            rung,
            quantizer,
            score,
            feasible,
            structure,
            pixels: Some((pixels, geometry)),
        });
        self.retain_finalist_pixels();
        Ok(self.probes.len() - 1)
    }

    /// Keeps the pixel plans of the two coarsest feasible probes only; every
    /// other probe's pixels are dropped so a large frame holds at most two
    /// coefficient payloads besides the one being planned.
    fn retain_finalist_pixels(&mut self) {
        let mut feasible: Vec<(u32, usize)> = self
            .probes
            .iter()
            .enumerate()
            .filter(|(_, p)| p.feasible)
            .map(|(i, p)| (p.rung.get(), i))
            .collect();
        feasible.sort_unstable();
        let keep: Vec<usize> = feasible.iter().take(2).map(|&(_, i)| i).collect();
        for (i, probe) in self.probes.iter_mut().enumerate() {
            if !keep.contains(&i) {
                probe.pixels = None;
            }
        }
    }

    /// The finest infeasible probe and the coarsest feasible probe, when both
    /// exist.
    fn bracket(&self) -> Option<((Rung, f64), (Rung, f64))> {
        let lo = self
            .probes
            .iter()
            .filter(|p| !p.feasible)
            .max_by_key(|p| p.rung)
            .map(|p| (p.rung, p.score))?;
        let hi = self
            .probes
            .iter()
            .filter(|p| p.feasible)
            .min_by_key(|p| p.rung)
            .map(|p| (p.rung, p.score))?;
        (lo.0 < hi.0).then_some((lo, hi))
    }

    fn coarsest_feasible(&self) -> Option<&ProbeRecord> {
        self.probes
            .iter()
            .filter(|p| p.feasible)
            .min_by_key(|p| p.rung)
    }

    fn finest_probe(&self) -> Option<&ProbeRecord> {
        self.probes.iter().max_by_key(|p| p.rung)
    }

    /// The next rung to probe while every probe so far lies on one side of
    /// the target: extrapolate the crossing from the measured loss slope (or
    /// the prior exponent after a single probe) and aim past it by a margin
    /// that grows with each attempt, clamped to one bounded jump.
    fn extrapolated_step(&self, finer: bool, attempt: u32) -> Option<Rung> {
        let mut points: Vec<(Rung, f64)> = self.probes.iter().map(|p| (p.rung, p.score)).collect();
        points.sort_by_key(|p| p.0);
        let (from, other) = if finer {
            (
                points.last().copied()?,
                points
                    .len()
                    .checked_sub(2)
                    .and_then(|i| points.get(i).copied()),
            )
        } else {
            (points.first().copied()?, points.get(1).copied())
        };
        if (finer && from.0 == Rung::TOP) || (!finer && from.0 == Rung::FLOOR) {
            return None;
        }
        let alpha = other
            .and_then(|o| {
                let dx = ln_scale(from.0) - ln_scale(o.0);
                let dy = loss(from.1).ln() - loss(o.1).ln();
                (dx.abs() > f64::EPSILON && dy / dx < 0.0).then(|| (-dy / dx).clamp(0.2, 3.0))
            })
            .unwrap_or(PRIOR_LOSS_EXPONENT);
        let margin = EXPANSION_MARGIN.powi(i32::try_from(attempt.saturating_add(1)).unwrap_or(1));
        let mut ratio = (loss(from.1) / loss(self.threshold())).powf(1.0 / alpha);
        ratio = if finer {
            ratio * margin
        } else {
            ratio / margin
        };
        ratio = ratio.clamp(1.0 / MAX_EXPANSION_JUMP, MAX_EXPANSION_JUMP);
        if (finer && ratio <= 1.0) || (!finer && ratio >= 1.0) {
            return Some(geometric_step(from.0, finer));
        }
        let rung = rung_for_scale(ln_scale(from.0).exp() * ratio);
        Some(if rung == from.0 {
            geometric_step(from.0, finer)
        } else {
            rung
        })
    }

    /// Expands from the current extreme until a bracket exists, the ladder
    /// saturates, or the probe budget runs out.
    fn expand_until_bracketed(&mut self) -> Result<()> {
        let mut attempt = 0u32;
        while self.bracket().is_none() && self.pixel_budget_left() {
            // All probes so far are on one side of the target.
            let Some(last) = self.probes.last() else {
                return Ok(());
            };
            let finer = !last.feasible;
            let Some(next) = self.extrapolated_step(finer, attempt) else {
                return Ok(());
            };
            if self.already_probed(next) {
                return Ok(());
            }
            self.probe(next, false)?;
            attempt = attempt.saturating_add(1);
        }
        Ok(())
    }

    /// One probe beyond the budget, only when nothing has met the target:
    /// aim well past the extrapolated crossing so the stream handed back is
    /// verified feasible whenever the ladder can reach the target at all.
    fn rescue_probe(&mut self) -> Result<()> {
        if self.coarsest_feasible().is_some() {
            return Ok(());
        }
        let Some(next) = self.extrapolated_step(true, 3) else {
            return Ok(());
        };
        if self.already_probed(next) {
            return Ok(());
        }
        self.probe(next, false)?;
        Ok(())
    }

    /// Whether the coarsest feasible probe is already tight against the
    /// target or the bracket has no room left.
    fn crossing_is_tight(&self) -> bool {
        match self.bracket() {
            Some(((lo, _), (hi, hi_score))) => {
                hi.get() <= lo.get().saturating_add(1)
                    || hi_score - self.threshold() <= MET_OVERSHOOT_BAND
            }
            None => true,
        }
    }

    /// Aims at the log-loss crossing (with the effort's reserve) and probes
    /// it, then once more from the refined bracket, while budget remains.
    fn tighten(&mut self) -> Result<()> {
        while self.pixel_budget_left() && !self.crossing_is_tight() {
            let Some((lo, hi)) = self.bracket() else {
                break;
            };
            let relative = 100.0 - loss(self.threshold()) * (1.0 - self.budget.reserve);
            let aim_score = relative.max(self.threshold() + MIN_AIM_MARGIN);
            // Aiming at (or past) the feasible end of the bracket would only
            // re-probe its neighbour: the bracket is as tight as the aim.
            if aim_score >= hi.1 {
                break;
            }
            let Some(rung) = log_loss_crossing(lo, hi, aim_score) else {
                break;
            };
            if self.already_probed(rung) {
                break;
            }
            self.probe(rung, false)?;
        }
        Ok(())
    }
}

/// One exactly priced finalist.
struct PricedFinalist {
    quantizer: QuantizerChoice,
    score: f64,
    feasible: bool,
    structure: StructureSource,
    plan: ValidatedEmissionPlan,
    bytes: Vec<u8>,
    sizing: CodestreamSizing,
}

/// Trains entropy for a probe's pixels and emits the stream exactly.
fn price_pixels(
    nav: &mut Navigator<'_, '_, '_>,
    quantizer: QuantizerChoice,
    score: f64,
    structure: StructureSource,
    pixels: &ValidatedPixelPlan,
    geometry: &VardctGeometry,
) -> Result<PricedFinalist> {
    let start = Instant::now();
    let plan = nav
        .ctx
        .attach_entropy(pixels, geometry, nav.finalist_entropy)?;
    let entropy_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
    nav.stats.entropy_ms = nav.stats.entropy_ms.saturating_add(entropy_ms);
    let emit_start = Instant::now();
    let emission = emit_codestream_with_executor(&plan, nav.ctx.executor())?;
    let emit_ms = u64::try_from(emit_start.elapsed().as_millis()).unwrap_or(u64::MAX);
    nav.stats.emit_ms = nav.stats.emit_ms.saturating_add(emit_ms);
    nav.stats.exact_prices = nav.stats.exact_prices.saturating_add(1);
    let feasible = score >= nav.threshold();
    nav.trace.push(QualityProbe {
        kind: ProbeKind::Exact,
        quantizer,
        effective_scale: effective_scale(quantizer.rung),
        score: Some(score),
        bytes: Some(emission.sizing.total),
        structure,
        feasible: Some(feasible),
        millis: entropy_ms.saturating_add(emit_ms),
    });
    Ok(PricedFinalist {
        quantizer,
        score,
        feasible,
        structure,
        plan,
        bytes: emission.bytes,
        sizing: emission.sizing,
    })
}

/// Runs the score-targeted search over a prepared frame.
///
/// `request` carries the effort (`rate_preset`) and the starting policy; its
/// rate target, if any, is ignored. `evaluator` scores every probe;
/// `executor` runs planning, entropy and emission.
///
/// # Errors
///
/// [`PolicyError::Unsupported`] for an effort this build cannot run, plus
/// anything the planner, evaluator or writer refuses.
pub fn search_frame_perceptual(
    frame: &PreparedFrame,
    atlas: &AnalysisAtlas,
    request: &EncodeRequest,
    target: PerceptualTarget,
    evaluator: &mut dyn PerceptualEvaluator,
    executor: &jpxl_encode::EncodeExecutor,
) -> Result<QualityOutcome> {
    let preset = request.rate_preset;
    let budget = QualityBudget::for_preset(preset);
    let (enable_cfl, structure_tier, finalist_entropy) = match preset {
        RateSearchPreset::Fast => (false, EntropySearch::Fast, EntropySearch::FinalFast),
        RateSearchPreset::Balanced => {
            #[cfg(feature = "g5-bounded-entropy")]
            let finalist = EntropySearch::BoundedFinal;
            #[cfg(not(feature = "g5-bounded-entropy"))]
            let finalist = EntropySearch::FinalFast;
            (true, EntropySearch::Fast, finalist)
        }
        RateSearchPreset::Quality => (true, EntropySearch::Full, EntropySearch::Full),
    };

    let transform_owned = if request.restoration.gaborish {
        Some(crate::prepare_gaborish_frame(frame)?)
    } else {
        None
    };
    let transform_frame = transform_owned.as_ref().unwrap_or(frame);
    let features = source_features(atlas, frame.width(), frame.height(), frame.is_grayscale());
    let predicted = rung_for_scale(predicted_effective_scale(&features, target.minimum_score));

    let mut ctx = CandidateSearchContext::new(frame, transform_frame, atlas, request, executor);
    let mut nav = Navigator {
        ctx: &mut ctx,
        evaluator,
        target: target.minimum_score,
        guard: DEFAULT_SCORE_GUARD,
        budget,
        enable_cfl,
        structure_tier,
        finalist_entropy,
        anchor: None,
        anchor_rung: None,
        probes: Vec::new(),
        trace: Vec::new(),
        stats: QualityStats {
            predicted: Some(predicted),
            ..QualityStats::default()
        },
    };

    // Navigation: predicted rung, bracket, crossing.
    nav.probe(predicted, true)?;
    nav.expand_until_bracketed()?;
    nav.tighten()?;
    nav.rescue_probe()?;
    nav.stats.bracket = nav.bracket().map(|((lo, _), (hi, _))| (lo, hi));

    // Finalists: the coarsest feasible probes, exactly priced.
    let mut finalists: Vec<PricedFinalist> = Vec::new();
    let mut saturated = false;
    let mut rescued = false;
    let anchor_rung = nav.anchor_rung.unwrap_or(predicted);
    let max_finalists = budget.exact_prices.clamp(1, 2);
    let mut ordered: Vec<usize> = (0..nav.probes.len())
        .filter(|&i| nav.probes.get(i).is_some_and(|p| p.feasible))
        .collect();
    ordered.sort_by_key(|&i| nav.probes.get(i).map_or(Rung::TOP, |p| p.rung));
    ordered.truncate(usize::try_from(max_finalists).unwrap_or(1));

    let mut under_target = false;
    if ordered.is_empty() {
        // Nothing met the target: emit the finest verified probe and say so —
        // saturation when that probe is the ladder's top, a work-cap miss
        // otherwise.
        under_target = true;
        saturated = nav.finest_probe().is_some_and(|p| p.rung == Rung::TOP);
        let Some(index) = nav
            .probes
            .iter()
            .enumerate()
            .max_by_key(|(_, p)| p.rung)
            .map(|(i, _)| i)
        else {
            return Err(PolicyError::Unsupported {
                what: "a perceptual search that never probed",
            });
        };
        ordered.push(index);
    }

    for index in ordered {
        if nav.stats.exact_prices >= budget.exact_prices {
            break;
        }
        let (rung, quantizer, score, structure) = {
            let Some(p) = nav.probes.get(index) else {
                continue;
            };
            (p.rung, p.quantizer, p.score, p.structure)
        };
        // Far from the anchor, a reused cover prices worse than a fresh one:
        // rebuild, re-score, and keep the rebuild only if it still qualifies.
        let rebuild = structure == StructureSource::Reused
            && structure_is_far(anchor_rung, rung)
            && nav.stats.structural_builds < budget.structural_builds
            && nav.pixel_budget_left()
            && !under_target;
        if rebuild {
            let fresh_index = nav.probe(rung, true)?;
            let fresh_ok = nav.probes.get(fresh_index).is_some_and(|p| p.feasible);
            if fresh_ok {
                let taken = nav
                    .probes
                    .get_mut(fresh_index)
                    .and_then(|p| p.pixels.take());
                if let Some((pixels, geometry)) = taken {
                    let fresh_score = nav.probes.get(fresh_index).map_or(score, |p| p.score);
                    let priced = price_pixels(
                        &mut nav,
                        quantizer,
                        fresh_score,
                        StructureSource::Fresh,
                        &pixels,
                        &geometry,
                    )?;
                    rescued = true;
                    finalists.push(priced);
                    continue;
                }
            }
        }
        // Reused (or anchor) structure: price the retained pixels, or rebuild
        // them with the same structure if they were dropped.
        let retained = nav.probes.get_mut(index).and_then(|p| p.pixels.take());
        let (pixels, geometry) = match retained {
            Some(planned) => planned,
            None => {
                let reuse = match (&nav.anchor, structure) {
                    (Some(anchor), StructureSource::Reused) => AnchorReuse::CoverAndCfl(anchor),
                    _ => AnchorReuse::None,
                };
                nav.ctx
                    .pixel_plan(quantizer, nav.enable_cfl, nav.structure_tier, reuse, None)?
            }
        };
        let priced = price_pixels(&mut nav, quantizer, score, structure, &pixels, &geometry)?;
        finalists.push(priced);
    }

    // Selection: the smallest exact feasible stream; if none is feasible (only
    // possible when saturated), the finest priced one.
    let pick = finalists
        .iter()
        .enumerate()
        .filter(|(_, f)| f.feasible)
        .min_by_key(|(_, f)| f.sizing.total)
        .or_else(|| {
            finalists
                .iter()
                .enumerate()
                .max_by_key(|(_, f)| f.quantizer.rung)
        })
        .map(|(i, _)| i);
    let Some(pick) = pick else {
        return Err(PolicyError::Unsupported {
            what: "a perceptual search with no priced finalist",
        });
    };
    let chosen = finalists.swap_remove(pick);
    let coarsest_feasible_rung = nav.coarsest_feasible().map(|p| p.rung);
    let adjacent_infeasible = coarsest_feasible_rung.is_some_and(|r| {
        nav.probes
            .iter()
            .any(|p| !p.feasible && p.rung.get().saturating_add(1) == r.get())
    });
    let status = if saturated {
        QualityStatus::SaturatedTop
    } else if under_target {
        QualityStatus::UnderTargetWorkCap
    } else if chosen.structure == StructureSource::Fresh && rescued {
        QualityStatus::RescuedFreshStructure
    } else if chosen.quantizer.rung == Rung::FLOOR {
        QualityStatus::SaturatedFloor
    } else if chosen.score - nav.threshold() <= MET_OVERSHOOT_BAND {
        QualityStatus::Met
    } else if adjacent_infeasible {
        QualityStatus::MetAdjacentRungs
    } else {
        QualityStatus::MetWorkCap
    };
    let metric_version = nav.evaluator.metric_version();
    let stats = nav.stats;
    let trace = core::mem::take(&mut nav.trace);
    Ok(QualityOutcome {
        codestream: chosen.bytes,
        plan: chosen.plan,
        sizing: chosen.sizing,
        chosen: chosen.quantizer,
        requested_score: target.minimum_score,
        achieved_score: chosen.score,
        guard: DEFAULT_SCORE_GUARD,
        saturated,
        status,
        trace,
        stats,
        features,
        metric_version,
    })
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use super::*;
    use crate::request::PerceptualMetric;

    /// An evaluator that scores a plan from its quantizer alone, with a
    /// monotone power law in the effective scale, so the search logic can be
    /// exercised without rendering.
    struct CurveEvaluator {
        calls: u32,
    }

    impl PerceptualEvaluator for CurveEvaluator {
        fn evaluate(&mut self, candidate: &ValidatedPixelPlan) -> Result<PerceptualObservation> {
            self.calls += 1;
            let q = &candidate.plan().spatial.quantizer;
            let scale = f64::from(q.global_scale.get())
                * f64::from(
                    candidate
                        .plan()
                        .spatial
                        .lf_groups
                        .first()
                        .and_then(|g| g.blocks.first())
                        .map_or(1, |b| b.hf_mul.get()),
                );
            // loss = 60 * (scale / 1000)^-0.8: score 40 at scale 1000, ~91
            // at 8000.
            let loss = 60.0 * (scale / 1000.0).powf(-0.8);
            Ok(PerceptualObservation {
                score: 100.0 - loss,
            })
        }

        fn metric_version(&self) -> &'static str {
            "curve-test"
        }
    }

    fn frame() -> PreparedFrame {
        let (w, h) = (96u32, 80u32);
        let rgb: Vec<u8> = (0..w * h)
            .flat_map(|i| {
                let x = i % w;
                let y = i / w;
                [
                    (x * 2 + (y * 7) % 23) as u8,
                    (y * 3) as u8,
                    ((x * y) % 251) as u8,
                ]
            })
            .collect();
        PreparedFrame::from_srgb8(w, h, &rgb).expect("frame")
    }

    fn run(preset: RateSearchPreset, target: f64) -> (QualityOutcome, u32) {
        let frame = frame();
        let atlas = AnalysisAtlas::analyze(&frame);
        let mut request = EncodeRequest::for_quality(preset);
        request.restoration.gaborish = false;
        request.restoration.epf_iters = 0;
        let executor = request.resources.executor();
        let mut evaluator = CurveEvaluator { calls: 0 };
        let target = PerceptualTarget::new(PerceptualMetric::Ssimulacra2, target).expect("target");
        let outcome =
            search_frame_perceptual(&frame, &atlas, &request, target, &mut evaluator, &executor)
                .expect("search");
        (outcome, evaluator.calls)
    }

    #[test]
    fn the_crossing_stays_strictly_inside_the_bracket_and_orders_the_loss() {
        let lo = (Rung::new(999), 60.0);
        let hi = (Rung::new(7999), 91.0);
        let r = log_loss_crossing(lo, hi, 85.0).expect("a crossing");
        assert!(r > lo.0 && r < hi.0, "{r:?}");
        // A finer point that scored lower cannot be interpolated.
        assert!(log_loss_crossing((Rung::new(999), 70.0), (Rung::new(7999), 65.0), 68.0).is_none());
        // Adjacent rungs leave no room.
        assert!(log_loss_crossing((Rung::new(10), 50.0), (Rung::new(11), 60.0), 55.0).is_none());
    }

    #[test]
    fn geometric_steps_move_and_stop_at_the_ladder_ends() {
        let r = Rung::new(999);
        let finer = geometric_step(r, true);
        let coarser = geometric_step(r, false);
        assert!(finer > r && coarser < r);
        assert_eq!(geometric_step(Rung::FLOOR, false), Rung::FLOOR);
        assert_eq!(geometric_step(Rung::TOP, true), Rung::TOP);
    }

    #[test]
    fn the_fallback_prediction_is_monotone_in_the_target() {
        let f = SourceFeatures {
            width: 100,
            height: 100,
            grayscale: false,
            luma_variance_q10: 1e-5,
            luma_variance_q50: 1e-4,
            luma_variance_q90: 1e-3,
            chroma_variance_q50: 1e-4,
            flat_fraction: 0.1,
            edge_proxy: 9e-4,
        };
        let a = predicted_effective_scale(&f, 50.0);
        let b = predicted_effective_scale(&f, 85.0);
        let c = predicted_effective_scale(&f, 95.0);
        // Non-decreasing; calibration cells clamped at the sweep ceiling may tie.
        assert!(a <= b && b <= c && a < c, "{a} {b} {c}");
    }

    #[test]
    fn every_effort_meets_its_target_inside_its_budget() {
        for preset in [RateSearchPreset::Fast, RateSearchPreset::Balanced] {
            for target in [50.0, 70.0, 85.0] {
                let (outcome, calls) = run(preset, target);
                let budget = QualityBudget::for_preset(preset);
                assert!(
                    outcome.achieved_score >= target,
                    "{preset:?} {target}: achieved {}",
                    outcome.achieved_score
                );
                assert!(
                    outcome.stats.pixel_probes <= budget.pixel_probes,
                    "{:?}",
                    outcome.stats
                );
                assert!(
                    outcome.stats.exact_prices <= budget.exact_prices,
                    "{:?}",
                    outcome.stats
                );
                assert_eq!(calls, outcome.stats.pixel_probes);
                assert!(!outcome.saturated);
                assert!(!outcome.codestream.is_empty());
                assert!(matches!(
                    outcome.status,
                    QualityStatus::Met
                        | QualityStatus::MetAdjacentRungs
                        | QualityStatus::MetWorkCap
                        | QualityStatus::RescuedFreshStructure
                ));
            }
        }
    }

    #[test]
    fn a_higher_target_never_costs_fewer_bytes_or_a_lower_score() {
        let (low, _) = run(RateSearchPreset::Balanced, 60.0);
        let (high, _) = run(RateSearchPreset::Balanced, 90.0);
        assert!(high.achieved_score >= low.achieved_score);
        assert!(high.sizing.total >= low.sizing.total);
    }

    #[test]
    fn an_unreachable_target_saturates_explicitly() {
        // The curve tops out near 99.96 at the ladder's finest rung.
        let (outcome, _) = run(RateSearchPreset::Fast, 99.99);
        assert!(
            outcome.saturated,
            "status {:?} achieved {} chosen {:?} trace {:?}",
            outcome.status, outcome.achieved_score, outcome.chosen, outcome.trace
        );
        assert_eq!(outcome.status, QualityStatus::SaturatedTop);
        assert!(outcome.achieved_score < 99.99);
    }

    #[test]
    fn the_trace_is_machine_readable() {
        let (outcome, _) = run(RateSearchPreset::Fast, 70.0);
        let json = outcome.trace_json("fast");
        assert!(json.starts_with("{\"schema\":\"jpxl.quality-trace/1\""));
        assert!(json.contains("\"probes\":[{\"kind\":\"pixel\""));
        assert!(json.contains("\"status\":\""));
    }
}
