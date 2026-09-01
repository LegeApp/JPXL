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
//! its navigation probe and price caps (plus at most one rescue probe beyond
//! the navigation cap when nothing has met the target yet) and reports what
//! it could verify. "Smallest" is bounded the same way: the selection is the
//! smallest among the exact-priced finalists this budget retained, not a
//! global minimum over every conceivable stream meeting the score. The
//! target is a floor — an outcome whose stream is below it says so
//! explicitly: [`QualityStatus::SaturatedTop`] when the ladder's finest rung
//! missed, [`QualityStatus::UnderTargetWorkCap`] when the probes ran out
//! first. The public facade refuses both by default rather than returning
//! them as ordinary successes.

use std::time::Instant;

use crate::candidate::CandidateSearchContext;
use crate::error::{PolicyError, Result};
use crate::navigation::{LocalModel, Progress, force_progress, ln_scale, rung_for_scale};
use crate::quality_features::{SourceFeatures, source_features};
use crate::quality_predictor::{
    FALLBACK_LOG_FIT, FLAT_BUCKET_EDGES, INITIAL_RUNG_TABLE, LUMA_BUCKET_EDGES,
};
use crate::quantizer_ladder::{QuantizerChoice, Rung, effective_scale};
use crate::reducer::ReducerLimits;
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
    /// The half-resolution surrogate's score for the same candidate, when the
    /// evaluator computed one (Phase S1 shadow instrumentation). Advisory
    /// only: feasibility and every accepted outcome read `score`.
    pub surrogate_score: Option<f64>,
    /// Wall time of the surrogate evaluation alone, in milliseconds.
    pub surrogate_millis: Option<u64>,
    /// Wall time of the candidate reconstruction alone, in milliseconds
    /// (probe attribution: the render share of this evaluation). Advisory
    /// only; `None` when the evaluator does not split its own phases.
    pub render_millis: Option<u64>,
    /// Wall time of the canonical metric pass alone, in milliseconds
    /// (probe attribution: the SSIMULACRA2 share of this evaluation).
    /// Advisory only; `None` when the evaluator does not split its own
    /// phases.
    pub metric_millis: Option<u64>,
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

    /// Reconstructs and scores an owned `candidate`, returning it when the
    /// evaluator wants the policy to retain its coefficient payload for exact
    /// pricing.
    ///
    /// The default keeps the plan. A memory-bounded evaluator may instead
    /// render it, release it before allocating metric scratch, and return
    /// `None`; the policy already knows how to rebuild a dropped finalist from
    /// its quantizer and captured structure without changing its stream.
    ///
    /// # Errors
    ///
    /// As [`Self::evaluate`].
    fn evaluate_owned(
        &mut self,
        candidate: ValidatedPixelPlan,
    ) -> Result<(PerceptualObservation, Option<ValidatedPixelPlan>)> {
        let observation = self.evaluate(&candidate)?;
        Ok((observation, Some(candidate)))
    }

    /// As [`Self::evaluate_owned`], but `with_surrogate` asks the evaluator
    /// to also report the half-resolution surrogate's score of the same
    /// rendered planes (Phase S1 calibration pairing). An evaluator without
    /// a surrogate ignores the request; the canonical score is unaffected
    /// either way.
    ///
    /// # Errors
    ///
    /// As [`Self::evaluate`].
    fn evaluate_owned_for_navigation(
        &mut self,
        candidate: ValidatedPixelPlan,
        with_surrogate: bool,
    ) -> Result<(PerceptualObservation, Option<ValidatedPixelPlan>)> {
        let _ = with_surrogate;
        self.evaluate_owned(candidate)
    }

    /// Whether [`Self::evaluate_surrogate_owned`] produces observations.
    fn supports_surrogate(&self) -> bool {
        false
    }

    /// Reconstructs `candidate` and scores it with the half-resolution
    /// surrogate only — no canonical score. `Ok(None)` when the evaluator
    /// has no surrogate. The plan is always consumed: a surrogate
    /// observation can never become a finalist, so its payload is never
    /// needed again.
    ///
    /// # Errors
    ///
    /// As [`Self::evaluate`].
    fn evaluate_surrogate_owned(
        &mut self,
        candidate: ValidatedPixelPlan,
    ) -> Result<Option<PerceptualObservation>> {
        drop(candidate);
        Ok(None)
    }

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

/// Smallest predictor interval width (`ln(interval_high / interval_low)`)
/// that routes an uncertain in-distribution cell to the surrogate bracket
/// search. Below it, the shadow corpus shows the canonical search already
/// settling in two probes, which a surrogate detour cannot beat.
pub const SURROGATE_INTERVAL_WIDTH_GATE: f64 = 0.3;

/// Smallest requested score that routes to the surrogate bracket search.
/// The calibrated estimate must land proposals inside the 1-point accept
/// band ([`MET_OVERSHOOT_BAND`]); on the Phase S2 decimated-surrogate
/// shadow corpus (development split) its mean error is 0.66 at a target of
/// 90, 1.26 at 85 and 4.6 at 30 — only the top band, where the surrogate
/// tracks the canonical curve most tightly, clears the accept band. (The
/// interim shrink-after-reconstruct surrogate cleared 85; the decimated
/// observation is coarser and buys its cheaper probes with this narrower
/// routing. Phase S3's coefficient-domain reconstruction cut a probe's
/// render+metric cost to ~0.32× canonical — ~0.33× at the routed targets,
/// where quantization leaves the high-frequency region dense and the
/// low-pass fast path rarely fires — while changing the observation itself
/// only at the f32-rounding level, so this fit carries over.)
pub const SURROGATE_TARGET_FLOOR: f64 = 90.0;

/// Most surrogate probes one baseline solve may spend (Phase S1). Surrogate
/// probes aim; they never satisfy the quality contract, so this cap trades
/// only proposal quality, never floor safety.
pub const SURROGATE_PROBE_CAP: u32 = 2;

/// Slope `k` relating canonical to surrogate log-loss movement:
/// `Δln(loss_canonical) ≈ k · Δln(loss_surrogate)`, by the surrogate's own
/// score band. Fitted on the ext calibration/development shadow-trace corpus
/// of the Phase S2 *decimated* surrogate (2026-08-29, 1127 cells,
/// `.agent/scratch/s2-banded-metric-20260829/fit_bands.py`): the decimated
/// observation under-moves everywhere — its restoration filters act at the
/// half scale, deepening the blindness to finest-scale loss — so every
/// band's slope sits higher than the interim shrink-after-reconstruct
/// surrogate's did.
fn surrogate_slope(surrogate_score: f64) -> f64 {
    if surrogate_score >= 90.0 {
        1.85
    } else if surrogate_score >= 80.0 {
        1.98
    } else if surrogate_score >= 60.0 {
        1.93
    } else {
        2.20
    }
}

/// Hard work caps of one effort.
///
/// `pixel_probes`, `exact_prices` and `structural_builds` bound the *baseline*
/// policy solve. `policy_trials` bounds the perceptual policy bank: how many
/// single-axis alternatives the coordinate descent may solve after the
/// baseline, each under its own small [`TRIAL_PIXEL_PROBES`]/
/// [`TRIAL_EXACT_PRICES`] cap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualityBudget {
    /// Full-frame render-and-score evaluations the baseline solve's
    /// *navigation* may spend. When they run out with nothing meeting the
    /// target, one rescue probe (`rescue_probe`) may still run beyond this
    /// cap, so the observable per-solve maximum is `pixel_probes + 1`; the
    /// trace and [`QualityStats::pixel_probes`] count it like any other
    /// probe.
    pub pixel_probes: u32,
    /// Entropy trainings followed by an exact emission in the baseline solve.
    pub exact_prices: u32,
    /// Fresh cover/CfL builds in the baseline solve (the first probe is one).
    pub structural_builds: u32,
    /// How many policy-bank alternatives the coordinate descent may solve
    /// (0 disables the bank — the fixed-policy controller). Set to 0 to
    /// reproduce the baseline-only result for an equal-score comparison.
    pub policy_trials: u32,
    /// Fraction of the target loss the crossing aims above the target, so a
    /// slightly optimistic interpolation still lands feasible. Small on
    /// purpose: the rate controller's equivalent is an eighth of its 2-3%
    /// tolerance band, and aiming a whole point high costs bytes on every
    /// encode.
    pub reserve: f64,
    /// The terminal coefficient reducer's work bounds, or `None` to skip it.
    /// Runs once on the winning finalist and keeps the reduced stream only if
    /// its exact size is smaller and its canonical score still meets the
    /// target.
    pub reducer: Option<ReducerLimits>,
    /// Whether the search reduces the frame's DCT8x8 candidates into a
    /// [`TransformFeatureSummary`](crate::quality_features::TransformFeatureSummary)
    /// before solving. On in Fast and Balanced with the (default)
    /// `one-shot-controller` feature: the summary feeds the crossing
    /// predictor's seed, and its parallel prefill warms the same forward
    /// cache the cover search reads, so in-search it measured net-neutral
    /// wall (promotion A/B, 2026-08-24).
    pub transform_shadow: bool,
}

/// Hard pixel-probe cap of one policy-bank trial: probe the baseline crossing
/// and at most one neighbour.
pub const TRIAL_PIXEL_PROBES: u32 = 2;

/// Hard exact-price cap of one policy-bank trial: price the one feasible
/// finalist.
pub const TRIAL_EXACT_PRICES: u32 = 1;

/// Smallest byte saving (fraction of the incumbent's bytes) a Quality
/// second-pass trial must beat to be worth another metric-and-entropy round.
pub const MIN_TRIAL_SAVING_FRACTION: f64 = 0.005;

/// Default Balanced policy-bank breadth.
///
/// **Zero — the bank is off by default on Balanced.** Two measurements agree
/// that the byte win does not pay for the wall on this corpus (five dev-split
/// images — mid, the 12 MP large, the photo scene, text-screenshot, gradient —
/// at quality 70 and 85):
///
/// * **PR 5** (2026-08-22) ran each alternative in its own candidate context,
///   so every trial rebuilt the forward DCT and cover it could not share. Bytes
///   fell only on the photo scene for a corpus-mean under 0.35%, and Balanced
///   ran +20%..+87% slower (mean ≈ +51%).
/// * **PR 5b** (2026-08-22) wired cross-policy structure reuse: quantizer-side
///   trials now reuse the baseline's cover and CfL on the shared context's warm
///   forward cache (zero structural builds; verified by
///   [`Self::reuses_structure_of`](crate::policy_bank::PerceptualPolicy::reuses_structure_of)
///   and the trial-build unit test) and structural trials rebuild cover without
///   re-transforming. That let the bank *lower* bytes at every cell where it
///   moved them (mid@70 −0.69%, scene@70 −0.59%, scene@85 −1.34%, else neutral)
///   while never regressing a byte. But the wall stayed decisively over budget —
///   +57%..+122%, mean ≈ +78% at four threads — because reuse only removes the
///   forward DCT and cover, and each trial's cost is dominated by its full-frame
///   render-and-score and its entropy-train-and-emit, which reuse cannot touch.
///
/// So the bank stays behind the [`QualityBudget`] breadth knob: the reuse
/// plumbing is retained (it is what the feature-gated Quality effort rides on
/// and what a future cheaper-metric or shared-entropy path would need), but
/// Balanced keeps `policy_trials = 0`. Set it explicitly (e.g. 2) through
/// [`search_frame_perceptual_with_budget`] to opt in.
pub const BALANCED_DEFAULT_POLICY_TRIALS: u32 = 0;

/// Whether Balanced runs the terminal reducer by default.
///
/// Off after PR 7's full locked-holdout closure: the reducer stayed bounded
/// and floor-safe, but the complete Quality candidate failed the standing
/// Contract B promotion screen, while the earlier Balanced development screen
/// exceeded its wall bound. The feature-gated Quality reference still runs it.
/// See AKR evidence `pqc-pr7-holdout-complete-2026-08-24`.
pub const BALANCED_DEFAULT_REDUCER: Option<ReducerLimits> = None;

impl QualityBudget {
    /// The budget of a preset.
    ///
    /// Fast and Balanced run the baseline only by default
    /// (`policy_trials = 0`; see [`BALANCED_DEFAULT_POLICY_TRIALS`] for why the
    /// Balanced bank is opt-in). The feature-gated Quality effort runs the whole
    /// bank as coordinate descent, where the extra wall is acceptable.
    #[must_use]
    pub const fn for_preset(preset: RateSearchPreset) -> Self {
        match preset {
            RateSearchPreset::Fast => Self {
                pixel_probes: 3,
                exact_prices: 2,
                structural_builds: 2,
                policy_trials: 0,
                reserve: 0.06,
                reducer: None,
                transform_shadow: cfg!(feature = "one-shot-controller"),
            },
            RateSearchPreset::Balanced => Self {
                pixel_probes: 5,
                exact_prices: 3,
                structural_builds: 2,
                policy_trials: BALANCED_DEFAULT_POLICY_TRIALS,
                reserve: 0.03,
                reducer: BALANCED_DEFAULT_REDUCER,
                transform_shadow: cfg!(feature = "one-shot-controller"),
            },
            RateSearchPreset::Quality => Self {
                pixel_probes: 10,
                exact_prices: 4,
                structural_builds: 3,
                policy_trials: 11,
                reserve: 0.02,
                reducer: Some(ReducerLimits::QUALITY),
                transform_shadow: false,
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
    /// Pixels planned, rendered and scored by the half-resolution surrogate
    /// only (Phase S1). Its `score` is the *estimated* canonical score from
    /// the cell's calibration; the raw surrogate score rides in
    /// `surrogate_score`. Never feasibility evidence for an outcome.
    Surrogate,
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
    /// Which policy did this work: 0 is the baseline, 1.. are the ranked
    /// policy-bank alternatives in trial order.
    pub policy_id: u32,
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
    /// The half-resolution surrogate's score, when the evaluator's Phase S1
    /// shadow instrumentation computed one. Never used by the search.
    pub surrogate_score: Option<f64>,
    /// Wall time of the shadow surrogate evaluation, in milliseconds
    /// (included in `millis`, which times the whole evaluator call).
    pub surrogate_millis: Option<u64>,
    /// B1-G shadow: the rung a paired speculative lane would have launched
    /// alongside the *previous* canonical probe — this probe's rung guessed
    /// by the ladder-next rule before the previous probe's score was known.
    /// Never used by the search; `None` where no lane would have launched
    /// (first probe, non-pixel work, or the rule aimed at an already-probed
    /// rung).
    pub spec_rung: Option<u32>,
    /// Wall time of this unit's pixel planning (cover/CfL/quantization), in
    /// milliseconds — the `encode` share of a pixel probe's cost. Exact
    /// prices carry no planning pass, so this is 0 there.
    pub plan_ms: u64,
    /// Wall time of this unit's candidate reconstruction alone, in
    /// milliseconds (probe attribution shadow). Advisory only.
    pub render_ms: Option<u64>,
    /// Wall time of this unit's canonical metric pass alone, in
    /// milliseconds (probe attribution shadow). Advisory only.
    pub metric_ms: Option<u64>,
}

/// One policy-bank trial's summary, for the trace and telemetry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolicyTrial {
    /// The policy id (1.. in trial order).
    pub id: u32,
    /// The rung its priced finalist landed on (0 if it found no feasible one).
    pub rung: u32,
    /// The finalist's score (NaN if it found no feasible one).
    pub score: f64,
    /// The finalist's exact bytes (0 if it found no feasible one).
    pub bytes: u64,
    /// Whether this trial's stream was the overall winner.
    pub kept: bool,
}

/// Work counters and timings of one search.
///
/// `pixel_probes`, `exact_prices` and `structural_builds` count the **baseline**
/// policy solve, so the facade reports the effort's stated per-solve budget;
/// the policy bank's own probes are visible in the trace (tagged with their
/// `policy_id`) and counted by `policy_trials`. The `*_ms` timings are totals
/// across the whole search, baseline and trials.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct QualityStats {
    /// Pixel probes spent in the baseline solve (including finalist re-scores).
    pub pixel_probes: u32,
    /// Surrogate probes spent in the baseline solve (Phase S1); they do not
    /// count against `pixel_probes`' canonical budget.
    pub surrogate_probes: u32,
    /// Exact prices spent in the baseline solve.
    pub exact_prices: u32,
    /// Fresh cover/CfL builds in the baseline solve.
    pub structural_builds: u32,
    /// How many policy-bank alternatives were actually solved.
    pub policy_trials: u32,
    /// Baseline bytes minus the winning stream's bytes: what the bank saved
    /// (0 when the baseline itself won).
    pub policy_winner_margin_bytes: u64,
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
    /// Canonical evaluations the terminal reducer spent.
    pub reducer_evaluations: u32,
    /// Coefficients the terminal reducer removed in the kept stream.
    pub reducer_edits: u32,
    /// Exact bytes the terminal reducer saved (0 when its stream was not kept).
    pub reducer_bytes_saved: u64,
    /// Whole-search work totals (baseline, policy trials and reducer alike),
    /// split by the units the one-shot program prices separately. A probe is
    /// one pixel plan, one full-frame reconstruction and one canonical metric
    /// evaluation; an exact price is one entropy training and one emission; a
    /// finalist rebuilt from a dropped plan is one extra pixel plan.
    pub work: QualityWork,
}

/// Whole-search work totals for the `jpxl.quality-trace/2` record.
///
/// Unlike the effort-budget counters above these are *totals across the whole
/// search* — baseline solve, policy-bank trials and the terminal reducer —
/// so the trace shows what a request actually cost, not just what counted
/// against the baseline caps. (A reducer pass that finds nothing to remove
/// reports no evaluations, so its canonical scores are not included in that
/// one case.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct QualityWork {
    /// Candidate pixel plans built (scored probes and unscored rebuilds).
    pub pixel_plans: u32,
    /// Full-frame reconstructions rendered for scoring.
    pub reconstructions: u32,
    /// Canonical metric evaluations of a reconstruction.
    pub metric_evaluations: u32,
    /// Entropy trainings.
    pub entropy_trainings: u32,
    /// Exact codestream emissions.
    pub emissions: u32,
}

/// A shadow predictor's counterfactual decision for one search, recorded in
/// the `jpxl.quality-trace/2` record without influencing the search (the
/// one-shot program's PR 3/4 shadow stage; the exact controller stays
/// authoritative). `None` until a shadow model is wired in.
#[derive(Debug, Clone, PartialEq)]
pub struct QualityPredictionTrace {
    /// The generated model's version string.
    pub model_version: &'static str,
    /// The feature-vector schema the model consumed.
    pub feature_schema: &'static str,
    /// Predicted median crossing rung for the target.
    pub median_rung: u32,
    /// Risk-adjusted candidate rung a one-shot controller would plan first.
    pub candidate_rung: u32,
    /// Calibrated lower crossing-interval rung.
    pub interval_low: u32,
    /// Calibrated upper crossing-interval rung.
    pub interval_high: u32,
    /// Predicted local loss exponent `-d ln(loss) / d ln(scale)`.
    pub local_loss_exponent: f64,
    /// Predicted risk (0..=1) that the target saturates the ladder.
    pub saturation_risk: f64,
    /// Out-of-distribution flags that fired, by name.
    pub ood_flags: Vec<&'static str>,
    /// Why the shadow would have routed to the exact controller, if it would.
    pub fallback_reason: Option<&'static str>,
    /// The canonical score the search observed at (or nearest to) the
    /// candidate rung, for counterfactual first-plan accounting.
    pub first_observed_score: Option<f64>,
    /// The slope-corrected rung the shadow would have re-planned at.
    pub correction_rung: Option<u32>,
    /// The counterfactual route: `one_shot`, `corrected`, or
    /// `fallback_exact`.
    pub decision_path: &'static str,
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
    /// Per-alternative summaries of the policy bank's trials, in trial order.
    pub policy_trials: Vec<PolicyTrial>,
    /// The source features the prediction was made from.
    pub features: SourceFeatures,
    /// The shadow predictor's counterfactual record, when one is wired in.
    pub prediction: Option<QualityPredictionTrace>,
    /// The PR 4 transform-feature summary, when the budget asked for one.
    pub transform_features: Option<crate::quality_features::TransformFeatureSummary>,
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

    /// The `jpxl.quality-trace/2` record of this search, one JSON line.
    #[must_use]
    pub fn trace_json(&self, effort: &str) -> String {
        let probes: Vec<String> = self
            .trace
            .iter()
            .map(|p| {
                format!(
                    "{{\"kind\":\"{}\",\"policy_id\":{},\"rung\":{},\"global_scale\":{},\"hf_mul\":{},\
                     \"effective_scale\":{},\"score\":{},\"bytes\":{},\"structure\":\"{}\",\
                     \"feasible\":{},\"millis\":{},\"surrogate_score\":{},\"surrogate_millis\":{},\
                     \"spec_rung\":{},\"plan_ms\":{},\"render_ms\":{},\"metric_ms\":{}}}",
                    match p.kind {
                        ProbeKind::Pixel => "pixel",
                        ProbeKind::Exact => "exact",
                        ProbeKind::Surrogate => "surrogate",
                    },
                    p.policy_id,
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
                    p.surrogate_score
                        .map_or_else(|| "null".to_owned(), |s| format!("{s}")),
                    p.surrogate_millis
                        .map_or_else(|| "null".to_owned(), |m| format!("{m}")),
                    p.spec_rung
                        .map_or_else(|| "null".to_owned(), |r| format!("{r}")),
                    p.plan_ms,
                    p.render_ms
                        .map_or_else(|| "null".to_owned(), |m| format!("{m}")),
                    p.metric_ms
                        .map_or_else(|| "null".to_owned(), |m| format!("{m}")),
                )
            })
            .collect();
        let trials: Vec<String> = self
            .policy_trials
            .iter()
            .map(|t| {
                format!(
                    "{{\"id\":{},\"rung\":{},\"score\":{},\"bytes\":{},\"kept\":{}}}",
                    t.id,
                    t.rung,
                    if t.score.is_finite() {
                        format!("{}", t.score)
                    } else {
                        "null".to_owned()
                    },
                    t.bytes,
                    t.kept,
                )
            })
            .collect();
        let bracket = self.stats.bracket.map_or_else(
            || "null".to_owned(),
            |(lo, hi)| format!("[{},{}]", lo.get(), hi.get()),
        );
        let prediction = self.prediction.as_ref().map_or_else(
            || "null".to_owned(),
            |p| {
                let flags: Vec<String> = p.ood_flags.iter().map(|f| format!("\"{f}\"")).collect();
                format!(
                    "{{\"model_version\":\"{}\",\"feature_schema\":\"{}\",\"median_rung\":{},\
                     \"candidate_rung\":{},\"interval_low\":{},\"interval_high\":{},\
                     \"local_loss_exponent\":{},\"saturation_risk\":{},\"ood_flags\":[{}],\
                     \"fallback_reason\":{},\"first_observed_score\":{},\"correction_rung\":{},\
                     \"decision_path\":\"{}\"}}",
                    p.model_version,
                    p.feature_schema,
                    p.median_rung,
                    p.candidate_rung,
                    p.interval_low,
                    p.interval_high,
                    p.local_loss_exponent,
                    p.saturation_risk,
                    flags.join(","),
                    p.fallback_reason
                        .map_or_else(|| "null".to_owned(), |r| format!("\"{r}\"")),
                    p.first_observed_score
                        .map_or_else(|| "null".to_owned(), |s| format!("{s}")),
                    p.correction_rung
                        .map_or_else(|| "null".to_owned(), |r| format!("{r}")),
                    p.decision_path,
                )
            },
        );
        format!(
            "{{\"schema\":\"jpxl.quality-trace/2\",\"metric_version\":\"{}\",\"score_guard\":{},\
             \"effort\":\"{}\",\"source_features\":{},\"predicted_rung\":{},\"bracket\":{},\
             \"pixel_probes\":{},\"surrogate_probes\":{},\"exact_prices\":{},\"structural_builds\":{},\"policy_trials\":{},\
             \"policy_winner_margin_bytes\":{},\"reducer\":{{\"evaluations\":{},\"edits\":{},\"bytes_saved\":{}}},\
             \"work\":{{\"pixel_plans\":{},\"reconstructions\":{},\"metric_evaluations\":{},\
             \"entropy_trainings\":{},\"emissions\":{}}},\"prediction\":{},\
             \"transform_features\":{},\
             \"requested_score\":{},\"achieved_score\":{},\
             \"guard_margin\":{},\"final_exact_bytes\":{},\"status\":\"{}\",\"saturated\":{},\
             \"wall_by_phase\":{{\"plan\":{},\"render_metric\":{},\"entropy\":{},\"emit\":{}}},\
             \"policy_trials_detail\":[{}],\"probes\":[{}]}}",
            self.metric_version,
            self.guard,
            effort,
            self.features.to_json(),
            self.stats
                .predicted
                .map_or_else(|| "null".to_owned(), |r| format!("{}", r.get())),
            bracket,
            self.stats.pixel_probes,
            self.stats.surrogate_probes,
            self.stats.exact_prices,
            self.stats.structural_builds,
            self.stats.policy_trials,
            self.stats.policy_winner_margin_bytes,
            self.stats.reducer_evaluations,
            self.stats.reducer_edits,
            self.stats.reducer_bytes_saved,
            self.stats.work.pixel_plans,
            self.stats.work.reconstructions,
            self.stats.work.metric_evaluations,
            self.stats.work.entropy_trainings,
            self.stats.work.emissions,
            prediction,
            self.transform_features
                .as_ref()
                .map_or_else(|| "null".to_owned(), |t| t.to_json()),
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
            trials.join(","),
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

/// This controller's `y` coordinate: `-ln(loss)`.
///
/// The metric loss falls as the quantizer gets finer, so the raw log-loss
/// curve runs the opposite way from the rate curve. Negating it puts both
/// the same way up — rising in `x` — which is the orientation
/// [`crate::navigation`] assumes, and it is an exact operation: the slope
/// and the crossing numerator both flip sign, so the crossing is unchanged
/// bit for bit (`navigation::tests::negating_the_y_axis_does_not_move_the_crossing`).
fn neg_ln_loss(score: f64) -> f64 {
    -loss(score).ln()
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
    let (y_lo, y_hi) = (neg_ln_loss(lo_score), neg_ln_loss(hi_score));
    if y_hi <= y_lo || x_hi <= x_lo {
        return None;
    }
    let x = LocalModel::through((x_lo, y_lo), (x_hi, y_hi)).crossing(neg_ln_loss(aim_score));
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
    // `ln_scale(from).exp()`, not `effective_scale(from) as f64`: the round
    // trip through the logarithm is what this step has always aimed from,
    // and it is not the identity on every rung.
    let scale = ln_scale(from).exp();
    let next = if finer {
        scale * BRACKET_RATIO
    } else {
        scale / BRACKET_RATIO
    };
    force_progress(rung_for_scale(next), from, finer, Progress::NudgeWhenStuck)
}

/// The next rung to probe while every point lies on one side of the target:
/// extrapolate the crossing from the measured loss slope (or the prior
/// exponent after a single point) and aim past it by a margin that grows
/// with each attempt, clamped to one bounded jump. The pure core of
/// [`Navigator::extrapolated_step`], shared with the B1-G speculation
/// shadow.
fn extrapolated_step_from(
    unsorted: &[(Rung, f64)],
    threshold: f64,
    prior_beta: Option<f64>,
    finer: bool,
    attempt: u32,
) -> Option<Rung> {
    let mut points: Vec<(Rung, f64)> = unsorted.to_vec();
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
    // The local slope in the shared `(x, -ln loss)` coordinate: positive
    // when the finer probe really did lose less. Rejected unless the two
    // points are far enough apart in `x` to fit through, and unless the
    // loss actually fell; otherwise the prior exponent stands in.
    let alpha = other
        .and_then(|o| {
            let model = LocalModel::through(
                (ln_scale(o.0), neg_ln_loss(o.1)),
                (ln_scale(from.0), neg_ln_loss(from.1)),
            );
            ((ln_scale(from.0) - ln_scale(o.0)).abs() > f64::EPSILON && model.slope > 0.0)
                .then(|| model.slope.clamp(0.2, 3.0))
        })
        .unwrap_or_else(|| prior_beta.unwrap_or(PRIOR_LOSS_EXPONENT));
    let margin = EXPANSION_MARGIN.powi(i32::try_from(attempt.saturating_add(1)).unwrap_or(1));
    let mut ratio = (loss(from.1) / loss(threshold)).powf(1.0 / alpha);
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

/// The rung the crossing aim (with the effort's reserve) points at from a
/// bracket. The pure core of [`Navigator::tighten_aim`], shared with the
/// B1-G speculation shadow.
fn crossing_aim_from(
    lo: (Rung, f64),
    hi: (Rung, f64),
    threshold: f64,
    reserve: f64,
) -> Option<Rung> {
    let relative = 100.0 - loss(threshold) * (1.0 - reserve);
    let aim_score = relative.max(threshold + MIN_AIM_MARGIN);
    // Aiming at (or past) the feasible end of the bracket would only
    // re-probe its neighbour: the bracket is as tight as the aim.
    if aim_score >= hi.1 {
        return None;
    }
    log_loss_crossing(lo, hi, aim_score)
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
    /// `true` for a canonically scored probe. A surrogate record carries an
    /// *estimated* canonical score, exists only during the surrogate phase
    /// to aim with, and is retired before any conclusion is drawn.
    canonical: bool,
}

/// The running state of one policy's solve.
///
/// The trace is borrowed and shared across every policy in a search, each
/// probe tagged with this navigator's [`policy_id`](Navigator::policy_id); the
/// counters and timings in `local` are this policy's own, folded into the
/// combined [`QualityStats`] by the orchestrator. `budget` here is the
/// *per-solve* cap (the baseline budget for policy 0, the small trial cap for
/// alternatives).
struct Navigator<'c, 'a, 'r, 't, 'e> {
    ctx: &'c mut CandidateSearchContext<'a>,
    /// The effective request this policy plans and prices under: the baseline's
    /// for policy 0, the alternative's for a bank trial. The context is shared
    /// across policies, so the request the planner reads is carried here rather
    /// than on the context.
    request: &'r EncodeRequest,
    evaluator: &'e mut dyn PerceptualEvaluator,
    target: f64,
    guard: f64,
    budget: QualityBudget,
    enable_cfl: bool,
    structure_tier: EntropySearch,
    finalist_entropy: EntropySearch,
    policy_id: u32,
    anchor: Option<StructuralAnchor>,
    anchor_rung: Option<Rung>,
    /// A model-predicted local loss exponent used in place of
    /// [`PRIOR_LOSS_EXPONENT`] while only one probe exists (the one-shot
    /// program's slope prior). Measured slopes always win once two probes
    /// bracket a direction.
    prior_beta: Option<f64>,
    probes: Vec<ProbeRecord>,
    trace: &'t mut Vec<QualityProbe>,
    local: QualityStats,
    /// Phase S1: pair every canonical probe with the surrogate score of the
    /// same rendered planes, keeping the calibration below current.
    pair_canonical: bool,
    /// The latest paired observation as `(ln loss_canonical, ln loss_surrogate)`:
    /// the anchor the surrogate-to-canonical estimate is drawn through.
    calibration: Option<(f64, f64)>,
}

impl Navigator<'_, '_, '_, '_, '_> {
    fn threshold(&self) -> f64 {
        self.target + self.guard
    }

    fn pixel_budget_left(&self) -> bool {
        self.local.pixel_probes < self.budget.pixel_probes
    }

    fn already_probed(&self, rung: Rung) -> bool {
        self.probes.iter().any(|p| p.rung == rung)
    }

    /// Plans, renders and scores one rung. `fresh` builds cover and CfL for
    /// this rung (and makes it the structure anchor when none exists yet);
    /// otherwise the anchor's structure is reused.
    fn probe(&mut self, rung: Rung, fresh: bool) -> Result<usize> {
        let spec_rung = self.speculate_next().map(|r| r.get());
        let quantizer = QuantizerChoice::at(rung, self.request.quant_lf)?;
        let plan_start = Instant::now();
        let (pixels, geometry, structure) = if fresh || self.anchor.is_none() {
            let mut captured = None;
            let planned = self.ctx.pixel_plan_for(
                self.request,
                quantizer,
                self.enable_cfl,
                self.structure_tier,
                AnchorReuse::None,
                Some(&mut captured),
            )?;
            self.local.structural_builds = self.local.structural_builds.saturating_add(1);
            if self.anchor.is_none() {
                self.anchor = captured;
                self.anchor_rung = Some(rung);
            }
            (planned.0, planned.1, StructureSource::Fresh)
        } else {
            let anchor = self.anchor.as_ref().ok_or(PolicyError::Unsupported {
                what: "a reused structure before any probe captured one",
            })?;
            let planned = self.ctx.pixel_plan_for(
                self.request,
                quantizer,
                false,
                self.structure_tier,
                AnchorReuse::CoverAndCfl(anchor),
                None,
            )?;
            (planned.0, planned.1, StructureSource::Reused)
        };
        let plan_millis = u64::try_from(plan_start.elapsed().as_millis()).unwrap_or(u64::MAX);
        self.local.plan_ms = self.local.plan_ms.saturating_add(plan_millis);

        let score_start = Instant::now();
        let (observation, pixels) = self
            .evaluator
            .evaluate_owned_for_navigation(pixels, self.pair_canonical)?;
        let score = observation.score;
        let surrogate_score = observation.surrogate_score;
        let surrogate_millis = observation.surrogate_millis;
        let render_ms = observation.render_millis;
        let metric_ms = observation.metric_millis;
        if let Some(surrogate) = surrogate_score {
            self.calibration = Some((loss(score).ln(), loss(surrogate).ln()));
        }
        let millis = u64::try_from(score_start.elapsed().as_millis()).unwrap_or(u64::MAX);
        self.local.render_metric_ms = self.local.render_metric_ms.saturating_add(millis);
        self.local.pixel_probes = self.local.pixel_probes.saturating_add(1);
        self.local.work.pixel_plans = self.local.work.pixel_plans.saturating_add(1);
        self.local.work.reconstructions = self.local.work.reconstructions.saturating_add(1);
        self.local.work.metric_evaluations = self.local.work.metric_evaluations.saturating_add(1);
        let feasible = score >= self.threshold();
        self.trace.push(QualityProbe {
            policy_id: self.policy_id,
            kind: ProbeKind::Pixel,
            quantizer,
            effective_scale: effective_scale(rung),
            score: Some(score),
            bytes: None,
            structure,
            feasible: Some(feasible),
            millis,
            surrogate_score,
            surrogate_millis,
            spec_rung,
            plan_ms: plan_millis,
            render_ms,
            metric_ms,
        });
        self.probes.push(ProbeRecord {
            rung,
            quantizer,
            score,
            feasible,
            structure,
            pixels: pixels.map(|pixels| (pixels, geometry)),
            canonical: true,
        });
        self.retain_finalist_pixels();
        Ok(self.probes.len() - 1)
    }

    /// Plans, renders and surrogate-scores one rung (Phase S1), recording an
    /// *estimated* canonical score drawn through the calibration pair with
    /// the fitted band slope. Returns `false` when the evaluator has no
    /// surrogate or no calibration exists yet — the caller falls back to
    /// canonical probing.
    fn probe_surrogate(&mut self, rung: Rung) -> Result<bool> {
        let Some((cal_canonical, cal_surrogate)) = self.calibration else {
            return Ok(false);
        };
        let quantizer = QuantizerChoice::at(rung, self.request.quant_lf)?;
        let anchor = self.anchor.as_ref().ok_or(PolicyError::Unsupported {
            what: "a surrogate probe before any canonical probe captured structure",
        })?;
        let plan_start = Instant::now();
        let (pixels, _geometry) = self.ctx.pixel_plan_for(
            self.request,
            quantizer,
            false,
            self.structure_tier,
            AnchorReuse::CoverAndCfl(anchor),
            None,
        )?;
        self.local.plan_ms = self
            .local
            .plan_ms
            .saturating_add(u64::try_from(plan_start.elapsed().as_millis()).unwrap_or(u64::MAX));
        let score_start = Instant::now();
        let Some(observation) = self.evaluator.evaluate_surrogate_owned(pixels)? else {
            return Ok(false);
        };
        let millis = u64::try_from(score_start.elapsed().as_millis()).unwrap_or(u64::MAX);
        self.local.render_metric_ms = self.local.render_metric_ms.saturating_add(millis);
        self.local.surrogate_probes = self.local.surrogate_probes.saturating_add(1);
        self.local.work.pixel_plans = self.local.work.pixel_plans.saturating_add(1);
        self.local.work.reconstructions = self.local.work.reconstructions.saturating_add(1);
        let surrogate = observation.score;
        let estimated = 100.0
            - (cal_canonical + surrogate_slope(surrogate) * (loss(surrogate).ln() - cal_surrogate))
                .exp();
        let feasible = estimated >= self.threshold();
        self.trace.push(QualityProbe {
            policy_id: self.policy_id,
            kind: ProbeKind::Surrogate,
            quantizer,
            effective_scale: effective_scale(rung),
            score: Some(estimated),
            bytes: None,
            structure: StructureSource::Reused,
            feasible: Some(feasible),
            millis,
            surrogate_score: Some(surrogate),
            surrogate_millis: observation.surrogate_millis,
            spec_rung: None,
            // Surrogate probes are off the canonical path; their plan share is
            // not separately timed and their half-resolution render and metric
            // are reported together as `surrogate_millis`.
            plan_ms: 0,
            render_ms: None,
            metric_ms: None,
        });
        self.probes.push(ProbeRecord {
            rung,
            quantizer,
            score: estimated,
            feasible,
            structure: StructureSource::Reused,
            pixels: None,
            canonical: false,
        });
        Ok(true)
    }

    /// Retires every surrogate record: they exist to aim, and every
    /// conclusion below — brackets kept, finalists priced, rescue and
    /// saturation semantics — must rest on canonical observations only.
    fn retire_surrogates(&mut self) {
        self.probes.retain(|p| p.canonical);
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
        let points: Vec<(Rung, f64)> = self.probes.iter().map(|p| (p.rung, p.score)).collect();
        extrapolated_step_from(&points, self.threshold(), self.prior_beta, finer, attempt)
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

    /// The rung the crossing aim (with the effort's reserve) points at from
    /// the current bracket, when one is worth probing.
    fn tighten_aim(&self) -> Option<Rung> {
        let (lo, hi) = self.bracket()?;
        let rung = crossing_aim_from(lo, hi, self.threshold(), self.budget.reserve)?;
        (!self.already_probed(rung)).then_some(rung)
    }

    /// B1-G shadow: the rung the *ladder-next* speculation rule would have
    /// launched in a paired lane alongside the previous probe — this probe's
    /// rung guessed before the previous probe's score was known.
    ///
    /// The rule replays the navigator's own decision core over the probes
    /// minus the last one, standing in for the hidden score with the
    /// hidden-state expectation: the local model through the two points
    /// nearest the previous rung, the prior exponent from a single point,
    /// or — for the very first pair — the prediction's own aim (the
    /// threshold). Purely observational: reads state, changes nothing.
    fn speculate_next(&self) -> Option<Rung> {
        let (hidden, last) = self.probes.split_at(self.probes.len().checked_sub(1)?);
        let last = last.first()?;
        let x_last = ln_scale(last.rung);
        let mut points: Vec<(f64, f64)> = hidden
            .iter()
            .map(|p| (ln_scale(p.rung), neg_ln_loss(p.score)))
            .collect();
        points.sort_by(|a, b| a.0.total_cmp(&b.0));
        let prior = self.prior_beta.unwrap_or(PRIOR_LOSS_EXPONENT);
        let y_expected = match points.as_slice() {
            [] => neg_ln_loss(self.threshold()),
            [only] => only.1 + prior * (x_last - only.0),
            _ => {
                // The segment containing the previous rung, or the nearest
                // end segment, with the same slope guard the expansion uses.
                let idx = points.partition_point(|p| p.0 < x_last);
                let (a, b) = if idx == 0 {
                    (points.first()?, points.get(1)?)
                } else if idx >= points.len() {
                    (points.get(points.len().checked_sub(2)?)?, points.last()?)
                } else {
                    (points.get(idx.checked_sub(1)?)?, points.get(idx)?)
                };
                let slope = ((b.0 - a.0).abs() > f64::EPSILON)
                    .then(|| LocalModel::through(*a, *b).slope)
                    .filter(|s| *s > 0.0)
                    .map_or(prior, |s| s.clamp(0.2, 3.0));
                a.1 + slope * (x_last - a.0)
            }
        };
        let synth_score = 100.0 - (-y_expected).exp();
        let synth_feasible = synth_score >= self.threshold();
        let synth = (last.rung, synth_score);
        // Bracket over the hidden probes plus the synthetic point.
        let lo = hidden
            .iter()
            .filter(|p| !p.feasible)
            .map(|p| (p.rung, p.score))
            .chain((!synth_feasible).then_some(synth))
            .max_by_key(|p| p.0);
        let hi = hidden
            .iter()
            .filter(|p| p.feasible)
            .map(|p| (p.rung, p.score))
            .chain(synth_feasible.then_some(synth))
            .min_by_key(|p| p.0);
        let guess = match (lo, hi) {
            (Some(lo), Some(hi)) if lo.0 < hi.0 => {
                crossing_aim_from(lo, hi, self.threshold(), self.budget.reserve)
            }
            _ => {
                let set: Vec<(Rung, f64)> = hidden
                    .iter()
                    .map(|p| (p.rung, p.score))
                    .chain(core::iter::once(synth))
                    .collect();
                let attempt = u32::try_from(hidden.len()).unwrap_or(u32::MAX);
                extrapolated_step_from(
                    &set,
                    self.threshold(),
                    self.prior_beta,
                    !synth_feasible,
                    attempt,
                )
            }
        }?;
        // A lane aimed at a rung the pair already covers would not launch.
        (guess != last.rung && !self.already_probed(guess)).then_some(guess)
    }

    /// Aims at the log-loss crossing (with the effort's reserve) and probes
    /// it, then once more from the refined bracket, while budget remains.
    fn tighten(&mut self) -> Result<()> {
        while self.pixel_budget_left() && !self.crossing_is_tight() {
            let Some(rung) = self.tighten_aim() else {
                break;
            };
            self.probe(rung, false)?;
        }
        Ok(())
    }

    /// Phase S1: acquire the bracket with surrogate probes — the same
    /// expansion aims as the canonical machinery, observed at half
    /// resolution and mapped through the cell's calibration. Exploration
    /// stops the moment a (mixed) bracket exists: a surrogate probe costs a
    /// full-resolution reconstruction, so it is spent only where a canonical
    /// expansion probe would otherwise go, never on tightening — the
    /// canonical confirmation of the crossing aim is the tightening. Every
    /// record this pushes is retired before any conclusion is drawn.
    fn surrogate_explore(&mut self) -> Result<()> {
        if !self.evaluator.supports_surrogate() || self.calibration.is_none() {
            return Ok(());
        }
        let mut attempt = 0u32;
        while self.local.surrogate_probes < SURROGATE_PROBE_CAP && self.bracket().is_none() {
            let Some(last) = self.probes.last() else {
                break;
            };
            let finer = !last.feasible;
            let Some(aim) = self.extrapolated_step(finer, attempt) else {
                break;
            };
            attempt = attempt.saturating_add(1);
            if self.already_probed(aim) {
                break;
            }
            if !self.probe_surrogate(aim)? {
                break;
            }
        }
        Ok(())
    }

    /// The S1 proposal after [`Self::surrogate_explore`]: the rung the
    /// canonical confirmation should probe, plus whether an
    /// estimated-infeasible observation bounds it from below (the
    /// surrogate's stand-in for the canonical bracket's infeasible end).
    ///
    /// With a bracket, the proposal is the crossing aim (or the bracket's
    /// feasible end when the aim has nowhere tighter to point); with every
    /// observation on the feasible side, it is the coarsest of them.
    fn surrogate_proposal(&self) -> Option<(Rung, bool, bool)> {
        let coarsest = self
            .probes
            .iter()
            .filter(|p| p.feasible)
            .min_by_key(|p| p.rung)?;
        let (rung, canonical) = if self.bracket().is_some() && !self.crossing_is_tight() {
            match self.tighten_aim() {
                Some(aim) => (aim, false),
                None => (coarsest.rung, coarsest.canonical),
            }
        } else {
            (coarsest.rung, coarsest.canonical)
        };
        let bounded_below = self.probes.iter().any(|p| !p.feasible && p.rung < rung);
        Some((rung, canonical, bounded_below))
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
    nav: &mut Navigator<'_, '_, '_, '_, '_>,
    quantizer: QuantizerChoice,
    score: f64,
    structure: StructureSource,
    pixels: &ValidatedPixelPlan,
    geometry: &VardctGeometry,
) -> Result<PricedFinalist> {
    let start = Instant::now();
    let plan = nav
        .ctx
        .attach_entropy_for(nav.request, pixels, geometry, nav.finalist_entropy)?;
    let entropy_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
    nav.local.entropy_ms = nav.local.entropy_ms.saturating_add(entropy_ms);
    let emit_start = Instant::now();
    let emission = emit_codestream_with_executor(&plan, nav.ctx.executor())?;
    let emit_ms = u64::try_from(emit_start.elapsed().as_millis()).unwrap_or(u64::MAX);
    nav.local.emit_ms = nav.local.emit_ms.saturating_add(emit_ms);
    nav.local.exact_prices = nav.local.exact_prices.saturating_add(1);
    nav.local.work.entropy_trainings = nav.local.work.entropy_trainings.saturating_add(1);
    nav.local.work.emissions = nav.local.work.emissions.saturating_add(1);
    let feasible = score >= nav.threshold();
    nav.trace.push(QualityProbe {
        policy_id: nav.policy_id,
        kind: ProbeKind::Exact,
        quantizer,
        effective_scale: effective_scale(quantizer.rung),
        score: Some(score),
        bytes: Some(emission.sizing.total),
        structure,
        feasible: Some(feasible),
        millis: entropy_ms.saturating_add(emit_ms),
        surrogate_score: None,
        surrogate_millis: None,
        spec_rung: None,
        plan_ms: 0,
        render_ms: None,
        metric_ms: None,
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

/// The per-preset planning tiers: CfL on/off, the structural entropy tier the
/// planner reads, and the finalist entropy tier the emission trains.
fn planning_tiers(preset: RateSearchPreset) -> (bool, EntropySearch, EntropySearch) {
    match preset {
        RateSearchPreset::Fast => (false, EntropySearch::Fast, EntropySearch::FinalFast),
        RateSearchPreset::Balanced => {
            #[cfg(feature = "g5-bounded-entropy")]
            let finalist = EntropySearch::BoundedFinal;
            #[cfg(not(feature = "g5-bounded-entropy"))]
            let finalist = EntropySearch::FinalFast;
            (true, EntropySearch::Fast, finalist)
        }
        RateSearchPreset::Quality => (true, EntropySearch::Full, EntropySearch::Full),
    }
}

/// What one policy's solve chose, plus the navigation facts the orchestrator
/// needs to seed the bank, reuse structure and name the terminal status.
struct PolicySolve {
    finalist: PricedFinalist,
    /// The winner's rung: where a bank alternative starts its own bracket.
    seed_rung: Rung,
    saturated: bool,
    under_target: bool,
    rescued: bool,
    adjacent_infeasible: bool,
}

/// Solves one policy fully (the baseline): predict, bracket, tighten, rescue,
/// then exact-price the coarsest feasible probes and keep the smallest stream.
///
/// This is the fixed-policy navigator PR 4 shipped, extracted so the policy
/// bank can call it once for the baseline and reuse the same machinery for
/// each alternative through [`solve_trial`].
///
/// `ctx` is the search context the whole bank shares: the baseline fills its
/// forward-DCT cache here, and the returned [`StructuralAnchor`] (the captured
/// cover and CfL) lets quantizer-side trials reuse both against that same
/// warm cache. The anchor is `None` only if the search never planned a probe.
#[allow(clippy::too_many_arguments)]
fn solve_baseline(
    ctx: &mut CandidateSearchContext<'_>,
    request: &EncodeRequest,
    evaluator: &mut dyn PerceptualEvaluator,
    target: f64,
    guard: f64,
    budget: QualityBudget,
    enable_cfl: bool,
    structure_tier: EntropySearch,
    finalist_entropy: EntropySearch,
    predicted: Rung,
    prior_beta: Option<f64>,
    confident_stop: bool,
    use_surrogates: bool,
    trace: &mut Vec<QualityProbe>,
) -> Result<(PolicySolve, QualityStats, Option<StructuralAnchor>)> {
    let use_surrogates = use_surrogates && evaluator.supports_surrogate();
    let mut nav = Navigator {
        ctx,
        request,
        evaluator,
        target,
        guard,
        budget,
        enable_cfl,
        structure_tier,
        finalist_entropy,
        policy_id: 0,
        anchor: None,
        anchor_rung: None,
        prior_beta,
        probes: Vec::new(),
        trace,
        local: QualityStats {
            predicted: Some(predicted),
            ..QualityStats::default()
        },
        pair_canonical: use_surrogates,
        calibration: None,
    };

    // Navigation: predicted rung, bracket, crossing. Phase N3: when the
    // crossing model is confident and its candidate probe lands feasible
    // within the same overshoot band a bracketed search would accept
    // (`MET_OVERSHOOT_BAND`), the coarser-verification expansion is skipped —
    // the probe already is the answer the full search would keep. Sized on
    // the ext calibration/development trace corpus (2026-08-29): every cell
    // the stop fires on, the full search chose exactly this rung, so the
    // saving is one probe and zero bytes. The stopped probe is still a
    // canonical render-and-score, and its finalist is still exactly priced:
    // nothing about what constitutes a valid answer changes.
    nav.probe(predicted, true)?;
    let one_shot_hit = confident_stop
        && nav
            .probes
            .first()
            .is_some_and(|p| p.feasible && p.score - nav.threshold() <= MET_OVERSHOOT_BAND);
    // Phase S1: on an uncertain in-distribution cell, the bracket search is
    // run at half resolution first. Surrogate probes locate the estimated
    // crossing; the proposal is then confirmed canonically, and the
    // surrogate records are retired so everything below rests on canonical
    // observations alone. The confirmed probe is accepted without a
    // canonical coarser bracket only under the same overshoot band a
    // bracketed search accepts, and only when the surrogate observed an
    // estimated-infeasible point coarser than it (its stand-in for the
    // bracket's infeasible end). A mispropose costs one canonical
    // correction through the unchanged expansion/tightening machinery —
    // the budgets and the rescue semantics are untouched.
    let mut surrogate_hit = false;
    if !one_shot_hit && use_surrogates {
        nav.surrogate_explore()?;
        let proposal = nav.surrogate_proposal();
        nav.retire_surrogates();
        // Calibration is spent; further canonical probes have no surrogate
        // consumer, so they stop paying for the paired half-resolution score.
        nav.pair_canonical = false;
        if let Some((rung, canonical, bounded_below)) = proposal {
            if !canonical && nav.pixel_budget_left() && !nav.already_probed(rung) {
                nav.probe(rung, false)?;
            }
            surrogate_hit = bounded_below
                && nav.coarsest_feasible().is_some_and(|p| {
                    p.rung <= rung && p.score - nav.threshold() <= MET_OVERSHOOT_BAND
                });
        }
    }
    if !one_shot_hit && !surrogate_hit {
        nav.expand_until_bracketed()?;
    }
    nav.tighten()?;
    nav.rescue_probe()?;
    nav.local.bracket = nav.bracket().map(|((lo, _), (hi, _))| (lo, hi));

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
        if nav.local.exact_prices >= budget.exact_prices {
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
            && nav.local.structural_builds < budget.structural_builds
            && nav.pixel_budget_left()
            && !under_target;
        if rebuild {
            let fresh_index = nav.probe(rung, true)?;
            let fresh_ok = nav.probes.get(fresh_index).is_some_and(|p| p.feasible);
            if fresh_ok {
                let fresh_score = nav.probes.get(fresh_index).map_or(score, |p| p.score);
                let retained = nav
                    .probes
                    .get_mut(fresh_index)
                    .and_then(|p| p.pixels.take());
                let (pixels, geometry) = match retained {
                    Some(planned) => planned,
                    None => {
                        nav.local.work.pixel_plans = nav.local.work.pixel_plans.saturating_add(1);
                        nav.ctx.pixel_plan_for(
                            nav.request,
                            quantizer,
                            nav.enable_cfl,
                            nav.structure_tier,
                            AnchorReuse::None,
                            None,
                        )?
                    }
                };
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
                nav.local.work.pixel_plans = nav.local.work.pixel_plans.saturating_add(1);
                nav.ctx.pixel_plan_for(
                    nav.request,
                    quantizer,
                    nav.enable_cfl,
                    nav.structure_tier,
                    reuse,
                    None,
                )?
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
    // Hand the captured cover/CfL back so quantizer-side trials reuse it on the
    // shared context's warm forward cache.
    let anchor = nav.anchor.take();
    let local = nav.local;
    let seed_rung = chosen.quantizer.rung;
    Ok((
        PolicySolve {
            finalist: chosen,
            seed_rung,
            saturated,
            under_target,
            rescued,
            adjacent_infeasible,
        },
        local,
        anchor,
    ))
}

/// Solves one policy-bank alternative to the same target under a small budget
/// ([`TRIAL_PIXEL_PROBES`] pixel probes, [`TRIAL_EXACT_PRICES`] exact price),
/// starting from the baseline's crossing rung `seed`.
///
/// `shared_anchor` is `Some` only for a quantizer-side alternative that reuses
/// the baseline's cover and CfL; a CfL or restoration alternative passes `None`
/// and builds a fresh plan. The finalist is `None` when the alternative found
/// no feasible stream inside its budget; its stats come back either way so
/// the search's whole-work totals stay honest.
///
/// `ctx` is the baseline's warm context: a quantizer-side trial reads its
/// forward coefficients from that cache and spends zero fresh structural
/// builds, while a structural trial rebuilds its cover on the same cache
/// (one structural build, no re-transform). The anchor is cloned once here so
/// several quantizer-side trials can each own a retargetable copy.
#[allow(clippy::too_many_arguments)]
fn solve_trial(
    ctx: &mut CandidateSearchContext<'_>,
    request: &EncodeRequest,
    evaluator: &mut dyn PerceptualEvaluator,
    target: f64,
    guard: f64,
    reserve: f64,
    enable_cfl: bool,
    structure_tier: EntropySearch,
    finalist_entropy: EntropySearch,
    seed: Rung,
    shared_anchor: Option<&StructuralAnchor>,
    policy_id: u32,
    trace: &mut Vec<QualityProbe>,
) -> Result<(Option<PricedFinalist>, QualityStats)> {
    let seed_fresh = shared_anchor.is_none();
    let anchor_rung = shared_anchor.as_ref().map(|_| seed);
    let mut nav = Navigator {
        ctx,
        request,
        evaluator,
        target,
        guard,
        budget: QualityBudget {
            pixel_probes: TRIAL_PIXEL_PROBES,
            exact_prices: TRIAL_EXACT_PRICES,
            structural_builds: 2,
            policy_trials: 0,
            reserve,
            reducer: None,
            transform_shadow: false,
        },
        enable_cfl,
        structure_tier,
        finalist_entropy,
        policy_id,
        anchor: shared_anchor.cloned(),
        anchor_rung,
        prior_beta: None,
        probes: Vec::new(),
        trace,
        local: QualityStats::default(),
        pair_canonical: false,
        calibration: None,
    };

    // Probe the seed, then one neighbour: coarser to shed bytes if the seed
    // already meets the target, finer to reach it if it does not.
    nav.probe(seed, seed_fresh)?;
    let seed_feasible = nav.probes.last().is_some_and(|p| p.feasible);
    if nav.pixel_budget_left() {
        let next = geometric_step(seed, !seed_feasible);
        if next != seed && !nav.already_probed(next) {
            nav.probe(next, false)?;
        }
    }

    // The coarsest feasible probe is the cheapest stream that still qualifies.
    let chosen_index = nav
        .probes
        .iter()
        .enumerate()
        .filter(|(_, p)| p.feasible)
        .min_by_key(|(_, p)| p.rung)
        .map(|(i, _)| i);
    let Some(index) = chosen_index else {
        return Ok((None, nav.local));
    };
    let (quantizer, score, structure) = {
        let Some(p) = nav.probes.get(index) else {
            return Ok((None, nav.local));
        };
        (p.quantizer, p.score, p.structure)
    };
    let retained = nav.probes.get_mut(index).and_then(|p| p.pixels.take());
    let (pixels, geometry) = match retained {
        Some(planned) => planned,
        None => {
            let reuse = match (&nav.anchor, structure) {
                (Some(anchor), StructureSource::Reused) => AnchorReuse::CoverAndCfl(anchor),
                _ => AnchorReuse::None,
            };
            nav.local.work.pixel_plans = nav.local.work.pixel_plans.saturating_add(1);
            nav.ctx.pixel_plan_for(
                nav.request,
                quantizer,
                nav.enable_cfl,
                nav.structure_tier,
                reuse,
                None,
            )?
        }
    };
    let finalist = price_pixels(&mut nav, quantizer, score, structure, &pixels, &geometry)?;
    let local = nav.local;
    Ok((Some(finalist), local))
}

/// Folds a trial's timings and whole-search work totals (not its probe/price
/// counts, which stay baseline-scoped) into the combined stats.
fn fold_timings(combined: &mut QualityStats, trial: &QualityStats) {
    combined.plan_ms = combined.plan_ms.saturating_add(trial.plan_ms);
    combined.render_metric_ms = combined
        .render_metric_ms
        .saturating_add(trial.render_metric_ms);
    combined.entropy_ms = combined.entropy_ms.saturating_add(trial.entropy_ms);
    combined.emit_ms = combined.emit_ms.saturating_add(trial.emit_ms);
    combined.work.pixel_plans = combined
        .work
        .pixel_plans
        .saturating_add(trial.work.pixel_plans);
    combined.work.reconstructions = combined
        .work
        .reconstructions
        .saturating_add(trial.work.reconstructions);
    combined.work.metric_evaluations = combined
        .work
        .metric_evaluations
        .saturating_add(trial.work.metric_evaluations);
    combined.work.entropy_trainings = combined
        .work
        .entropy_trainings
        .saturating_add(trial.work.entropy_trainings);
    combined.work.emissions = combined.work.emissions.saturating_add(trial.work.emissions);
}

/// Runs the terminal reducer on the winning finalist and replaces it when the
/// reduced stream is exactly smaller. The reducer verifies every accepted
/// batch with the canonical score, so the replacement always meets
/// `threshold`; its exact price counts as one more exact price in the trace.
#[allow(clippy::too_many_arguments)]
fn reduce_winner(
    winner: &mut PricedFinalist,
    request: &EncodeRequest,
    frame: &PreparedFrame,
    transform_frame: &PreparedFrame,
    atlas: &AnalysisAtlas,
    evaluator: &mut dyn PerceptualEvaluator,
    executor: &jpxl_encode::EncodeExecutor,
    threshold: f64,
    finalist_entropy: EntropySearch,
    limits: ReducerLimits,
    stats: &mut QualityStats,
    trace: &mut Vec<QualityProbe>,
) -> Result<()> {
    let geometry = winner
        .plan
        .geometry()
        .map_err(|_| PolicyError::Unsupported {
            what: "a finalist whose geometry cannot be derived",
        })?;
    let pixels = winner.plan.pixels();
    let start = Instant::now();
    let reduced = crate::reducer::reduce_terminal(
        &pixels,
        &geometry,
        &winner.plan.plan().entropy,
        evaluator,
        threshold,
        limits,
    )?;
    stats.render_metric_ms = stats
        .render_metric_ms
        .saturating_add(u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX));
    let Some(reduced) = reduced else {
        return Ok(());
    };
    stats.reducer_evaluations = reduced.stats.evaluations;
    stats.work.reconstructions = stats
        .work
        .reconstructions
        .saturating_add(reduced.stats.evaluations);
    stats.work.metric_evaluations = stats
        .work
        .metric_evaluations
        .saturating_add(reduced.stats.evaluations);

    let ctx = CandidateSearchContext::new(frame, transform_frame, atlas, request, executor);
    let entropy_start = Instant::now();
    let plan = ctx.attach_entropy(&reduced.pixels, &geometry, finalist_entropy)?;
    let entropy_ms = u64::try_from(entropy_start.elapsed().as_millis()).unwrap_or(u64::MAX);
    stats.entropy_ms = stats.entropy_ms.saturating_add(entropy_ms);
    let emit_start = Instant::now();
    let emission = emit_codestream_with_executor(&plan, executor)?;
    let emit_ms = u64::try_from(emit_start.elapsed().as_millis()).unwrap_or(u64::MAX);
    stats.emit_ms = stats.emit_ms.saturating_add(emit_ms);
    stats.exact_prices = stats.exact_prices.saturating_add(1);
    stats.work.entropy_trainings = stats.work.entropy_trainings.saturating_add(1);
    stats.work.emissions = stats.work.emissions.saturating_add(1);
    let kept = emission.sizing.total < winner.sizing.total;
    trace.push(QualityProbe {
        policy_id: REDUCER_POLICY_ID,
        kind: ProbeKind::Exact,
        quantizer: winner.quantizer,
        effective_scale: effective_scale(winner.quantizer.rung),
        score: Some(reduced.score),
        bytes: Some(emission.sizing.total),
        structure: winner.structure,
        feasible: Some(true),
        millis: entropy_ms.saturating_add(emit_ms),
        surrogate_score: None,
        surrogate_millis: None,
        spec_rung: None,
        plan_ms: 0,
        render_ms: None,
        metric_ms: None,
    });
    if kept {
        stats.reducer_edits = reduced.stats.edits_applied;
        stats.reducer_bytes_saved = winner.sizing.total.saturating_sub(emission.sizing.total);
        winner.score = reduced.score;
        winner.plan = plan;
        winner.bytes = emission.bytes;
        winner.sizing = emission.sizing;
    }
    Ok(())
}

/// The `policy_id` the trace gives the reducer's exact price.
pub const REDUCER_POLICY_ID: u32 = u32::MAX;

/// One fresh-structure point of a quality-oracle ladder sweep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LadderPoint {
    /// The rung that was built and scored.
    pub rung: Rung,
    /// The quantizer at that rung (with the effort's `quant_lf` coupling).
    pub quantizer: QuantizerChoice,
    /// `global_scale * HfMul` of that quantizer.
    pub effective_scale: u64,
    /// The canonical score of the fresh-structure reconstruction.
    pub score: f64,
    /// The exact codestream size, when pricing was requested.
    pub exact_bytes: Option<u64>,
    /// Milliseconds in pixel planning.
    pub plan_ms: u64,
    /// Milliseconds in rendering and scoring.
    pub render_metric_ms: u64,
    /// Milliseconds in entropy training and emission (0 when not priced).
    pub price_ms: u64,
}

/// Sweeps `rungs` with the production quality pixel policy of `request`'s
/// effort — a fresh cover/CfL build per rung over one shared forward-DCT
/// cache — scoring every rung canonically and exact-pricing it when `price`
/// is set.
///
/// This is the one-shot program's oracle-label measurement (PR 2): a
/// crossing predictor must train on the same fresh-structure operating
/// points the production controller emits, not on fixed-quantizer
/// approximations of them. No navigation happens here; every requested rung
/// is built, scored and dropped before the next, so at most one frame-sized
/// pixel plan is alive at a time.
///
/// # Errors
///
/// As [`search_frame_perceptual`].
pub fn sweep_frame_perceptual(
    frame: &PreparedFrame,
    atlas: &AnalysisAtlas,
    request: &EncodeRequest,
    rungs: &[Rung],
    price: bool,
    evaluator: &mut dyn PerceptualEvaluator,
    executor: &jpxl_encode::EncodeExecutor,
) -> Result<Vec<LadderPoint>> {
    let preset = request.rate_preset;
    let (enable_cfl, structure_tier, finalist_entropy) = planning_tiers(preset);
    let transform_owned = if request.restoration.gaborish {
        Some(crate::prepare_gaborish_frame(frame)?)
    } else {
        None
    };
    let transform_frame = transform_owned.as_ref().unwrap_or(frame);
    let baseline_policy = crate::policy_bank::PerceptualPolicy::baseline(preset);
    let base_request = baseline_policy.apply(request);
    let mut ctx =
        CandidateSearchContext::new(frame, transform_frame, atlas, &base_request, executor);

    let mut points = Vec::with_capacity(rungs.len());
    for &rung in rungs {
        let quantizer = QuantizerChoice::at(rung, base_request.quant_lf)?;
        let plan_start = Instant::now();
        let (pixels, geometry) = ctx.pixel_plan_for(
            &base_request,
            quantizer,
            enable_cfl,
            structure_tier,
            AnchorReuse::None,
            None,
        )?;
        let plan_ms = u64::try_from(plan_start.elapsed().as_millis()).unwrap_or(u64::MAX);
        let score_start = Instant::now();
        let (observation, retained) = evaluator.evaluate_owned(pixels)?;
        let render_metric_ms = u64::try_from(score_start.elapsed().as_millis()).unwrap_or(u64::MAX);
        let mut price_ms = 0u64;
        let exact_bytes = if price {
            // A memory-bounded evaluator may have dropped the plan; the fresh
            // rebuild is byte-identical by construction.
            let pixels = match retained {
                Some(pixels) => pixels,
                None => {
                    ctx.pixel_plan_for(
                        &base_request,
                        quantizer,
                        enable_cfl,
                        structure_tier,
                        AnchorReuse::None,
                        None,
                    )?
                    .0
                }
            };
            let price_start = Instant::now();
            let plan =
                ctx.attach_entropy_for(&base_request, &pixels, &geometry, finalist_entropy)?;
            let emission = emit_codestream_with_executor(&plan, ctx.executor())?;
            price_ms = u64::try_from(price_start.elapsed().as_millis()).unwrap_or(u64::MAX);
            Some(emission.sizing.total)
        } else {
            None
        };
        points.push(LadderPoint {
            rung,
            quantizer,
            effective_scale: effective_scale(rung),
            score: observation.score,
            exact_bytes,
            plan_ms,
            render_metric_ms,
            price_ms,
        });
    }
    Ok(points)
}

/// Runs the score-targeted search over a prepared frame at the preset's budget.
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
    let budget = QualityBudget::for_preset(request.rate_preset);
    search_frame_perceptual_with_budget(frame, atlas, request, target, evaluator, executor, budget)
}

/// [`search_frame_perceptual`] with an explicit budget, so a caller can
/// override the policy-bank breadth — notably `policy_trials: 0` to reproduce
/// the fixed-policy (baseline-only) result for an equal-score comparison.
///
/// `#[doc(hidden)]`: the breadth is a research/testing knob, not part of the
/// stable facade, which always uses the preset budget.
///
/// # Errors
///
/// As [`search_frame_perceptual`].
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn search_frame_perceptual_with_budget(
    frame: &PreparedFrame,
    atlas: &AnalysisAtlas,
    request: &EncodeRequest,
    target: PerceptualTarget,
    evaluator: &mut dyn PerceptualEvaluator,
    executor: &jpxl_encode::EncodeExecutor,
    budget: QualityBudget,
) -> Result<QualityOutcome> {
    let preset = request.rate_preset;
    let target_score = target.minimum_score;
    let (enable_cfl, structure_tier, finalist_entropy) = planning_tiers(preset);

    // Gaborish preconditioning depends only on the transform frame, which no
    // bank axis changes, so build it once and share it across every policy.
    let transform_owned = if request.restoration.gaborish {
        Some(crate::prepare_gaborish_frame(frame)?)
    } else {
        None
    };
    let transform_frame = transform_owned.as_ref().unwrap_or(frame);
    let features = source_features(atlas, frame.width(), frame.height(), frame.is_grayscale());
    let predicted = rung_for_scale(predicted_effective_scale(&features, target_score));

    let mut trace: Vec<QualityProbe> = Vec::new();

    // --- Baseline solve (policy 0) ---
    // One context is shared across the baseline and every trial: its forward-DCT
    // cache is filled by the baseline cover build and then read (never rebuilt)
    // by each alternative, which is what makes quantizer-side trials cheap.
    let baseline_policy = crate::policy_bank::PerceptualPolicy::baseline(preset);
    let base_request = baseline_policy.apply(request);
    let mut ctx =
        CandidateSearchContext::new(frame, transform_frame, atlas, &base_request, executor);
    // PR 4 shadow measurement: reduce the DCT8x8 candidates before the solve
    // so the summary is quantizer-independent and the fill warms the cover's
    // own cache. Off in every production preset.
    let transform_features = if budget.transform_shadow {
        Some(ctx.prepare_quality_transform_summary(&base_request)?)
    } else {
        None
    };
    // PR 5 (feature-gated): the generated crossing model's risk-adjusted
    // candidate becomes the first fresh plan and its slope steers the first
    // correction; a fallback signal (OOD, wide interval, saturation risk,
    // tiny frame, or no generated model) keeps the standard predictor start.
    // Everything downstream — canonical verification before entropy, the
    // bounded navigator, the total caps — is unchanged. The summary the
    // prediction reads warmed the cover's own cache, so this costs no
    // duplicated transform work.
    #[cfg(feature = "one-shot-controller")]
    let (predicted, prior_beta, confident_stop, use_surrogates) = match transform_features
        .as_ref()
        .and_then(|tf| crate::quality_prediction::predict_v2(&features, tf, target_score))
    {
        // Confident: the risk-adjusted candidate is the first fresh plan,
        // and (phase N3) a feasible landing inside the accept band ends
        // navigation there.
        Some(p) if p.fallback_reason.is_none() => {
            (p.candidate_rung, Some(p.local_loss_exponent), true, false)
        }
        // Uncertain but in distribution (wide interval, saturation risk):
        // the navigator still runs its full bounded search from the model's
        // median seed — and (phase S1, feature-gated) its bracket search
        // runs at half resolution first, with canonical confirmation. Only
        // a wide interval engages surrogates: below the width gate, 98% of
        // cells settle in two canonical probes on the shadow corpus
        // (2026-08-29), which no surrogate detour can beat.
        Some(p) if p.ood_flags.is_empty() => {
            let wide = p.interval_low.get() > 0
                && (f64::from(p.interval_high.get()) / f64::from(p.interval_low.get())).ln()
                    >= SURROGATE_INTERVAL_WIDTH_GATE;
            (
                p.median_rung,
                Some(p.local_loss_exponent),
                false,
                cfg!(feature = "surrogate-navigation")
                    && wide
                    && target_score >= SURROGATE_TARGET_FLOOR,
            )
        }
        // Out of distribution: keep the legacy predictor's start, all
        // canonical.
        _ => (predicted, None, false, false),
    };
    #[cfg(not(feature = "one-shot-controller"))]
    let (prior_beta, confident_stop, use_surrogates): (Option<f64>, bool, bool) =
        (None, false, false);
    let (baseline, mut stats, baseline_anchor) = solve_baseline(
        &mut ctx,
        &base_request,
        evaluator,
        target_score,
        DEFAULT_SCORE_GUARD,
        budget,
        enable_cfl,
        structure_tier,
        finalist_entropy,
        predicted,
        prior_beta,
        confident_stop,
        use_surrogates,
        &mut trace,
    )?;

    let saturated = baseline.saturated;
    let under_target = baseline.under_target;
    let baseline_rescued = baseline.rescued;
    let adjacent_infeasible = baseline.adjacent_infeasible;
    let baseline_bytes = baseline.finalist.sizing.total;
    let seed = baseline.seed_rung;

    let mut winner = baseline.finalist;
    let mut winner_policy = baseline_policy;
    let mut winner_is_trial = false;
    let mut winner_trial_id: Option<u32> = None;
    let mut incumbent_bytes = baseline_bytes;
    let mut policy_trials: Vec<PolicyTrial> = Vec::new();
    let mut next_policy_id = 1u32;

    // --- Policy bank (coordinate descent over ranked single-axis alternatives).
    // Skipped when the bank is disabled, when nothing met the target, or when
    // the ladder saturated — there is no reserve to trade in those cases.
    if budget.policy_trials > 0 && !under_target && !saturated {
        let per_pass = usize::try_from(budget.policy_trials).unwrap_or(0);
        // Fast/Balanced run one pass; the feature-gated Quality effort permits
        // a bounded second pass around the updated winner, stopping when a pass
        // yields no worthwhile saving (plan §6.3).
        let max_passes = if preset == RateSearchPreset::Quality {
            2
        } else {
            1
        };
        let mut pass_seed = seed;
        for _pass in 0..max_passes {
            let ranked = crate::policy_bank::rank_alternatives(&features, preset);
            let mut improved = false;
            for policy in ranked.into_iter().take(per_pass) {
                if policy == winner_policy {
                    continue;
                }
                let policy_id = next_policy_id;
                next_policy_id = next_policy_id.saturating_add(1);
                // Every alternative solves on the shared baseline context. A
                // quantizer-side alternative (`reuses_structure_of` true) reuses
                // the baseline's captured cover and CfL via
                // `AnchorReuse::CoverAndCfl` and spends zero structural builds;
                // a CfL/restoration alternative passes no anchor and rebuilds
                // its cover (one structural build), but still reads the warm
                // forward DCTs rather than re-transforming the frame.
                let shared_anchor = if policy.reuses_structure_of(&baseline_policy) {
                    baseline_anchor.as_ref()
                } else {
                    None
                };
                let trial_request = policy.apply(request);
                let (trial_finalist, trial_stats) = solve_trial(
                    &mut ctx,
                    &trial_request,
                    evaluator,
                    target_score,
                    DEFAULT_SCORE_GUARD,
                    budget.reserve,
                    policy.cfl,
                    structure_tier,
                    finalist_entropy,
                    pass_seed,
                    shared_anchor,
                    policy_id,
                    &mut trace,
                )?;
                stats.policy_trials = stats.policy_trials.saturating_add(1);
                fold_timings(&mut stats, &trial_stats);
                match trial_finalist {
                    Some(finalist) => {
                        let bytes = finalist.sizing.total;
                        let saving = incumbent_bytes.saturating_sub(bytes);
                        // Quality demands a minimum saving; a single Balanced
                        // pass keeps any strict improvement.
                        let enough = if preset == RateSearchPreset::Quality {
                            #[allow(clippy::cast_precision_loss)]
                            let floor = MIN_TRIAL_SAVING_FRACTION * incumbent_bytes as f64;
                            #[allow(clippy::cast_precision_loss)]
                            let saved = saving as f64;
                            saved >= floor
                        } else {
                            saving > 0
                        };
                        let keep = finalist.feasible && bytes < incumbent_bytes && enough;
                        policy_trials.push(PolicyTrial {
                            id: policy_id,
                            rung: finalist.quantizer.rung.get(),
                            score: finalist.score,
                            bytes,
                            kept: false,
                        });
                        if keep {
                            winner = finalist;
                            winner_policy = policy;
                            winner_is_trial = true;
                            winner_trial_id = Some(policy_id);
                            incumbent_bytes = bytes;
                            improved = true;
                        }
                    }
                    None => {
                        policy_trials.push(PolicyTrial {
                            id: policy_id,
                            rung: 0,
                            score: f64::NAN,
                            bytes: 0,
                            kept: false,
                        });
                    }
                }
            }
            if !improved {
                break;
            }
            pass_seed = winner.quantizer.rung;
        }
    }

    // Mark the winning trial (if any) as kept.
    if let Some(id) = winner_trial_id
        && let Some(entry) = policy_trials.iter_mut().find(|t| t.id == id)
    {
        entry.kept = true;
    }
    stats.policy_winner_margin_bytes = baseline_bytes.saturating_sub(winner.sizing.total);

    // --- Terminal reducer (PR 7): spend the winner's reserve on bytes. ---
    if let Some(limits) = budget.reducer
        && !saturated
        && !under_target
        && winner.feasible
    {
        let winner_request = winner_policy.apply(request);
        reduce_winner(
            &mut winner,
            &winner_request,
            frame,
            transform_frame,
            atlas,
            evaluator,
            executor,
            target_score + DEFAULT_SCORE_GUARD,
            finalist_entropy,
            limits,
            &mut stats,
            &mut trace,
        )?;
    }

    let threshold = target_score + DEFAULT_SCORE_GUARD;
    let status = if saturated {
        QualityStatus::SaturatedTop
    } else if under_target {
        QualityStatus::UnderTargetWorkCap
    } else if !winner_is_trial && baseline_rescued && winner.structure == StructureSource::Fresh {
        QualityStatus::RescuedFreshStructure
    } else if winner.quantizer.rung == Rung::FLOOR {
        QualityStatus::SaturatedFloor
    } else if winner.score - threshold <= MET_OVERSHOOT_BAND {
        QualityStatus::Met
    } else if adjacent_infeasible {
        QualityStatus::MetAdjacentRungs
    } else {
        QualityStatus::MetWorkCap
    };
    let metric_version = evaluator.metric_version();
    // PR 3/4 shadow: the counterfactual one-shot record, computed from the
    // finished search's own probes. Never touches the emitted bytes; absent
    // when no transform summary was computed.
    let prediction = transform_features.as_ref().and_then(|tf| {
        crate::quality_prediction::shadow_prediction_trace(&features, tf, target_score, &trace)
    });
    Ok(QualityOutcome {
        codestream: winner.bytes,
        plan: winner.plan,
        sizing: winner.sizing,
        chosen: winner.quantizer,
        requested_score: target_score,
        achieved_score: winner.score,
        guard: DEFAULT_SCORE_GUARD,
        saturated,
        status,
        trace,
        stats,
        policy_trials,
        features,
        prediction,
        transform_features,
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
        discard: bool,
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
                surrogate_score: None,
                surrogate_millis: None,
                render_millis: None,
                metric_millis: None,
            })
        }

        fn evaluate_owned(
            &mut self,
            candidate: ValidatedPixelPlan,
        ) -> Result<(PerceptualObservation, Option<ValidatedPixelPlan>)> {
            let observation = self.evaluate(&candidate)?;
            Ok((observation, (!self.discard).then_some(candidate)))
        }

        fn metric_version(&self) -> &'static str {
            "curve-test"
        }
    }

    fn sized_frame(w: u32, h: u32) -> PreparedFrame {
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

    fn frame() -> PreparedFrame {
        sized_frame(96, 80)
    }

    fn run(preset: RateSearchPreset, target: f64) -> (QualityOutcome, u32) {
        run_with_budget(preset, target, QualityBudget::for_preset(preset))
    }

    fn run_with_budget(
        preset: RateSearchPreset,
        target: f64,
        budget: QualityBudget,
    ) -> (QualityOutcome, u32) {
        run_with_budget_and_retention(preset, target, budget, true)
    }

    fn run_with_budget_and_retention(
        preset: RateSearchPreset,
        target: f64,
        budget: QualityBudget,
        retain: bool,
    ) -> (QualityOutcome, u32) {
        let frame = frame();
        let atlas = AnalysisAtlas::analyze(&frame);
        let mut request = EncodeRequest::for_quality(preset);
        request.restoration.gaborish = false;
        let executor = request.resources.executor();
        let mut evaluator = CurveEvaluator {
            calls: 0,
            discard: !retain,
        };
        let target = PerceptualTarget::new(PerceptualMetric::Ssimulacra2, target).expect("target");
        let outcome = search_frame_perceptual_with_budget(
            &frame,
            &atlas,
            &request,
            target,
            &mut evaluator,
            &executor,
            budget,
        )
        .expect("search");
        (outcome, evaluator.calls)
    }

    /// PR 5 (feature-gated): the one-shot start plans the generated model's
    /// risk-adjusted candidate as the first fresh plan when the model routes
    /// there, and keeps the standard predictor start when any fallback
    /// signal fires. The caps and floor semantics are unchanged either way.
    #[cfg(feature = "one-shot-controller")]
    #[test]
    fn the_one_shot_start_follows_the_model_routing() {
        let (w, h) = (160u32, 144u32);
        let frame = sized_frame(w, h);
        let atlas = AnalysisAtlas::analyze(&frame);
        let mut request = EncodeRequest::for_quality(RateSearchPreset::Balanced);
        request.restoration.gaborish = false;
        let executor = request.resources.executor();
        let mut evaluator = CurveEvaluator {
            calls: 0,
            discard: false,
        };
        let target = PerceptualTarget::new(PerceptualMetric::Ssimulacra2, 85.0).expect("target");
        let budget = QualityBudget::for_preset(RateSearchPreset::Balanced);
        let outcome = search_frame_perceptual_with_budget(
            &frame,
            &atlas,
            &request,
            target,
            &mut evaluator,
            &executor,
            budget,
        )
        .expect("search");

        let features = source_features(&atlas, w, h, frame.is_grayscale());
        let mut check_cache = crate::CandidateForwardCache::new();
        let summary =
            crate::quality_transform_summary(&frame, &request, &mut check_cache, Some(&executor))
                .expect("transform summary");
        match crate::quality_prediction::predict_v2(&features, &summary, 85.0) {
            Some(p) if p.fallback_reason.is_none() => {
                assert_eq!(
                    outcome.stats.predicted,
                    Some(p.candidate_rung),
                    "the model's candidate is the first fresh plan"
                );
                assert_eq!(
                    outcome.trace.first().map(|probe| probe.quantizer.rung),
                    Some(p.candidate_rung)
                );
            }
            Some(p) if p.ood_flags.is_empty() => {
                assert_eq!(
                    outcome.stats.predicted,
                    Some(p.median_rung),
                    "an uncertain in-distribution route seeds the navigator with the median"
                );
            }
            _ => {
                let standard = rung_for_scale(predicted_effective_scale(&features, 85.0));
                assert_eq!(
                    outcome.stats.predicted,
                    Some(standard),
                    "an out-of-distribution frame keeps the legacy predictor start"
                );
            }
        }
        assert!(outcome.achieved_score >= 85.0);
        // The one rescue probe beyond the navigation cap stays the maximum.
        assert!(outcome.stats.pixel_probes <= budget.pixel_probes + 1);
        assert!(outcome.prediction.is_some(), "the shadow trace still fills");
    }

    /// The B1-G speculation shadow annotates every baseline pixel probe
    /// after the first with the ladder-next rule's guess (or an explicit
    /// `None` when no lane would launch), never annotates the first probe or
    /// exact prices, and reaches the trace record — without changing what
    /// the search does.
    #[test]
    fn the_speculation_shadow_annotates_pixel_probes_only() {
        let (outcome, _) = run(RateSearchPreset::Balanced, 85.0);
        let baseline_pixels: Vec<&QualityProbe> = outcome
            .trace
            .iter()
            .filter(|p| p.policy_id == 0 && p.kind == ProbeKind::Pixel)
            .collect();
        assert!(baseline_pixels.len() >= 2, "need a multi-probe solve");
        assert_eq!(
            baseline_pixels.first().and_then(|p| p.spec_rung),
            None,
            "no previous probe exists to pair with"
        );
        for probe in &outcome.trace {
            if probe.kind != ProbeKind::Pixel {
                assert_eq!(probe.spec_rung, None, "only pixel probes speculate");
            }
            if let Some(spec) = probe.spec_rung {
                assert!(
                    (Rung::FLOOR.get()..=Rung::TOP.get()).contains(&spec),
                    "a speculated rung stays on the ladder"
                );
            }
        }
        assert!(
            outcome.trace_json("balanced").contains("\"spec_rung\":"),
            "the shadow reaches the trace record"
        );
    }

    /// A solve that runs out of probes before anything meets a reachable
    /// target reports [`QualityStatus::UnderTargetWorkCap`] — not a success
    /// status — and its counters show the one rescue probe that ran beyond
    /// the navigation cap.
    #[test]
    fn running_out_of_probes_reports_under_target_work_cap() {
        let budget = QualityBudget {
            pixel_probes: 1,
            exact_prices: 1,
            structural_builds: 1,
            policy_trials: 0,
            reserve: 0.03,
            reducer: None,
            transform_shadow: false,
        };
        // The curve tops out near 99.96, so 99.9 is reachable on the ladder;
        // one navigation probe plus one bounded rescue jump cannot get there.
        let (outcome, _) = run_with_budget(RateSearchPreset::Balanced, 99.9, budget);
        assert_eq!(outcome.status, QualityStatus::UnderTargetWorkCap);
        assert!(!outcome.saturated, "{:?}", outcome.stats);
        assert!(outcome.achieved_score < 99.9);
        // The navigation cap of one, plus the single documented rescue probe.
        assert_eq!(outcome.stats.pixel_probes, 2);
        assert_eq!(outcome.stats.exact_prices, 1);
    }

    /// Proves the failed Quality promotion screen cannot leak its policy-bank
    /// or reducer work into either production effort's default budget.
    #[test]
    fn production_budgets_keep_quality_only_work_disabled() {
        for preset in [RateSearchPreset::Fast, RateSearchPreset::Balanced] {
            let budget = QualityBudget::for_preset(preset);
            assert_eq!(budget.policy_trials, 0);
            assert_eq!(budget.reducer, None);
        }

        let quality = QualityBudget::for_preset(RateSearchPreset::Quality);
        assert!(quality.policy_trials > 0);
        assert_eq!(quality.reducer, Some(ReducerLimits::QUALITY));
    }

    #[test]
    fn dropping_probe_plans_rebuilds_the_byte_identical_finalist() {
        let budget = QualityBudget::for_preset(RateSearchPreset::Balanced);
        let (retained, retained_calls) =
            run_with_budget_and_retention(RateSearchPreset::Balanced, 85.0, budget, true);
        let (dropped, dropped_calls) =
            run_with_budget_and_retention(RateSearchPreset::Balanced, 85.0, budget, false);

        assert_eq!(dropped.codestream, retained.codestream);
        assert_eq!(dropped.plan, retained.plan);
        assert_eq!(dropped.sizing, retained.sizing);
        assert_eq!(dropped.chosen, retained.chosen);
        assert_eq!(
            dropped.achieved_score.to_bits(),
            retained.achieved_score.to_bits()
        );
        assert_eq!(dropped.status, retained.status);
        assert_eq!(dropped_calls, retained_calls);
    }

    /// Total pixel probes across every policy (the trace counts them all;
    /// `stats.pixel_probes` is only the baseline solve's share).
    fn total_pixel_probes(outcome: &QualityOutcome) -> u32 {
        u32::try_from(
            outcome
                .trace
                .iter()
                .filter(|p| p.kind == ProbeKind::Pixel)
                .count(),
        )
        .unwrap_or(u32::MAX)
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
                // The baseline solve stays inside the preset's per-solve caps.
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
                // The bank's trials are hard-capped too, and total probes never
                // exceed the baseline cap plus one trial cap per trial.
                assert!(
                    outcome.stats.policy_trials <= budget.policy_trials,
                    "{:?}",
                    outcome.stats
                );
                let ceiling = budget.pixel_probes + budget.policy_trials * TRIAL_PIXEL_PROBES;
                assert!(
                    total_pixel_probes(&outcome) <= ceiling,
                    "{preset:?} {target}: {} probes over ceiling {ceiling}",
                    total_pixel_probes(&outcome)
                );
                // Every evaluator call is one pixel probe in the trace.
                assert_eq!(calls, total_pixel_probes(&outcome));
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
    fn fast_runs_no_policy_trials() {
        let (outcome, _) = run(RateSearchPreset::Fast, 70.0);
        assert_eq!(outcome.stats.policy_trials, 0);
        assert!(outcome.policy_trials.is_empty());
        // No probe is tagged with a non-baseline policy.
        assert!(outcome.trace.iter().all(|p| p.policy_id == 0));
    }

    #[test]
    fn the_balanced_bank_never_regresses_the_baseline_at_equal_score() {
        // Under the curve evaluator a policy's score at a rung is fixed (it
        // reads only the quantizer), so every alternative is feasible wherever
        // the baseline is: the bank can only trade bytes, never score.
        for target in [50.0, 70.0, 85.0] {
            let preset = RateSearchPreset::Balanced;
            // The bank is off by default on Balanced (measured wall too high),
            // so opt it in explicitly here.
            let full = QualityBudget {
                policy_trials: 2,
                ..QualityBudget::for_preset(preset)
            };
            let baseline_only = QualityBudget {
                policy_trials: 0,
                ..full
            };
            let (with_bank, _) = run_with_budget(preset, target, full);
            let (without, _) = run_with_budget(preset, target, baseline_only);
            assert!(
                with_bank.achieved_score >= target,
                "bank missed the target: {}",
                with_bank.achieved_score
            );
            assert!(
                with_bank.sizing.total <= without.sizing.total,
                "target {target}: bank {} > baseline {}",
                with_bank.sizing.total,
                without.sizing.total
            );
            // The reported margin is exactly the saving.
            assert_eq!(
                with_bank.stats.policy_winner_margin_bytes,
                without.sizing.total - with_bank.sizing.total
            );
            // The bank actually tried alternatives and listed them.
            assert!(!with_bank.policy_trials.is_empty());
            assert_eq!(
                with_bank.stats.policy_trials as usize,
                with_bank.policy_trials.len()
            );
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
        assert!(json.starts_with("{\"schema\":\"jpxl.quality-trace/2\""));
        assert!(json.contains("\"probes\":[{\"kind\":\"pixel\""));
        assert!(json.contains("\"policy_id\":0"));
        assert!(json.contains("\"policy_trials\":0"));
        assert!(json.contains("\"policy_winner_margin_bytes\":"));
        assert!(json.contains("\"policy_trials_detail\":["));
        assert!(json.contains("\"status\":\""));
    }

    #[test]
    fn the_reducer_never_exceeds_its_evaluation_cap_and_never_grows_the_stream() {
        let limits = ReducerLimits {
            max_evaluations: 3,
            max_rounds: 2,
            initial_batch: 64,
            batch_fraction: 0.0,
            min_batch: 8,
            max_batch: 256,
            key_floor: 0.0,
        };
        let with = QualityBudget {
            reducer: Some(limits),
            transform_shadow: false,
            ..QualityBudget::for_preset(RateSearchPreset::Balanced)
        };
        let without = QualityBudget {
            reducer: None,
            transform_shadow: false,
            ..QualityBudget::for_preset(RateSearchPreset::Balanced)
        };
        let (reduced, _) = run_with_budget(RateSearchPreset::Balanced, 70.0, with);
        let (plain, _) = run_with_budget(RateSearchPreset::Balanced, 70.0, without);
        assert!(reduced.stats.reducer_evaluations <= limits.max_evaluations);
        assert!(reduced.achieved_score >= 70.0 + DEFAULT_SCORE_GUARD);
        assert!(reduced.sizing.total <= plain.sizing.total);
        assert_eq!(
            reduced.stats.reducer_bytes_saved,
            plain.sizing.total - reduced.sizing.total
        );
        assert_eq!(
            reduced.chosen, plain.chosen,
            "the reducer keeps the quantizer"
        );
        let json = reduced.trace_json("balanced");
        assert!(json.contains("\"reducer\":{\"evaluations\":"));
        assert!(
            reduced
                .trace
                .iter()
                .any(|p| p.policy_id == REDUCER_POLICY_ID),
            "the reducer's exact price is traced"
        );
        assert!(plain.trace.iter().all(|p| p.policy_id != REDUCER_POLICY_ID));
    }

    #[test]
    fn the_balanced_trace_lists_its_policy_trials() {
        // Opt the bank in (off by default on Balanced).
        let budget = QualityBudget {
            policy_trials: 2,
            ..QualityBudget::for_preset(RateSearchPreset::Balanced)
        };
        let (outcome, _) = run_with_budget(RateSearchPreset::Balanced, 70.0, budget);
        assert!(!outcome.policy_trials.is_empty());
        let json = outcome.trace_json("balanced");
        // Every trial appears with its id and a `kept` flag.
        for trial in &outcome.policy_trials {
            assert!(json.contains(&format!("\"id\":{}", trial.id)));
        }
        // At most one trial is the winner.
        assert!(outcome.policy_trials.iter().filter(|t| t.kept).count() <= 1);
        // A non-baseline probe is tagged with its policy id.
        assert!(outcome.trace.iter().any(|p| p.policy_id > 0));
    }

    #[test]
    fn a_quantizer_side_trial_reuses_structure_and_a_structural_trial_builds_one() {
        use crate::policy_bank::PerceptualPolicy;
        use jpxl_encode::vardct::ids::QmScale;

        let preset = RateSearchPreset::Balanced;
        let frame = frame();
        let atlas = AnalysisAtlas::analyze(&frame);
        let mut request = EncodeRequest::for_quality(preset);
        // Gaborish off: the transform frame is the source frame, so the shared
        // context is built against `&frame` exactly as the orchestrator would.
        request.restoration.gaborish = false;
        let executor = request.resources.executor();
        let (enable_cfl, structure_tier, finalist_entropy) = planning_tiers(preset);
        let target = 70.0;
        let features = source_features(&atlas, frame.width(), frame.height(), frame.is_grayscale());
        let predicted = rung_for_scale(predicted_effective_scale(&features, target));

        let baseline_policy = PerceptualPolicy::baseline(preset);
        let base_request = baseline_policy.apply(&request);
        let mut ctx = CandidateSearchContext::new(&frame, &frame, &atlas, &base_request, &executor);
        let mut trace = Vec::new();
        let mut evaluator = CurveEvaluator {
            calls: 0,
            discard: false,
        };
        let (baseline, _stats, anchor) = solve_baseline(
            &mut ctx,
            &base_request,
            &mut evaluator,
            target,
            DEFAULT_SCORE_GUARD,
            QualityBudget::for_preset(preset),
            enable_cfl,
            structure_tier,
            finalist_entropy,
            predicted,
            None,
            false,
            false,
            &mut trace,
        )
        .expect("baseline solve");
        let anchor = anchor.expect("the baseline captured a structural anchor");
        let seed = baseline.seed_rung;

        // Quantizer-side alternative (chroma QM only): reuses the baseline's
        // cover and CfL through the shared context, so it spends no structural
        // build.
        let mut qs_policy = baseline_policy;
        qs_policy.x_qm_scale = QmScale::new(3).expect("qm 3");
        qs_policy.b_qm_scale = QmScale::new(3).expect("qm 3");
        assert!(
            qs_policy.reuses_structure_of(&baseline_policy),
            "a chroma-only alternative must reuse structure"
        );
        let qs_request = qs_policy.apply(&request);
        let (qs_finalist, qs_stats) = solve_trial(
            &mut ctx,
            &qs_request,
            &mut evaluator,
            target,
            DEFAULT_SCORE_GUARD,
            0.03,
            qs_policy.cfl,
            structure_tier,
            finalist_entropy,
            seed,
            Some(&anchor),
            1,
            &mut trace,
        )
        .expect("quantizer-side trial");
        assert!(qs_finalist.is_some(), "a feasible quantizer-side finalist");
        assert_eq!(
            qs_stats.structural_builds, 0,
            "a quantizer-side trial rebuilt structure"
        );

        // Structural alternative (CfL flipped): builds exactly one fresh cover
        // (on the same warm forward cache).
        let mut st_policy = baseline_policy;
        st_policy.cfl = !baseline_policy.cfl;
        assert!(
            !st_policy.reuses_structure_of(&baseline_policy),
            "a CfL flip must not reuse structure"
        );
        let st_request = st_policy.apply(&request);
        let (st_finalist, st_stats) = solve_trial(
            &mut ctx,
            &st_request,
            &mut evaluator,
            target,
            DEFAULT_SCORE_GUARD,
            0.03,
            st_policy.cfl,
            structure_tier,
            finalist_entropy,
            seed,
            None,
            2,
            &mut trace,
        )
        .expect("structural trial");
        assert!(st_finalist.is_some(), "a feasible structural finalist");
        assert_eq!(
            st_stats.structural_builds, 1,
            "a structural trial did not build exactly one cover"
        );
    }
}
