//! JPEG XL encoder **policy**: everything that searches, scores or prefers.
//!
//! `docs/PLAN.md` slice 11, `docs/Encoder-plan1.md` §1. The encoder is split in
//! two, one way:
//!
//! ```text
//!   source pixels
//!        │
//!        ▼
//!   jpxl-encode-policy      analysis, block tiling, adaptive quantization,
//!        │                  CfL, entropy clustering, rate control, effort
//!        │  EmissionPlan
//!        ▼
//!   jpxl-encode             validate() -> exact bits. No heuristics.
//! ```
//!
//! `jpxl-encode` does not depend on this crate, and never may: that is what
//! keeps a heuristic from being reachable inside the writer, and it is what
//! lets policy be replaced wholesale — a different tiling search, a different
//! rate loop — without touching a line of syntax code. The manifest is the
//! enforcement; this paragraph is only the reason.
//!
//! `jpxl-decode` is not a dependency of either half. It is a **peer oracle**:
//! it validates the encoder's output in tests and must never be inside the
//! path that produces it, or an encoder bug the paired decoder happens to
//! accept would prove itself correct.
//!
//! # Stages
//!
//! | Module | `Encoder-plan1.md` | Milestone-2 state |
//! | --- | --- | --- |
//! | [`source`] | §2.1 `PreparedFrame` | resident XYB planes, from 8-bit sRGB |
//! | [`analysis`] | §2.2 `AnalysisAtlas` | per-atom mean and variance |
//! | [`block`] | §4 cover search | fixed DCT8x8 |
//! | [`quantize`] | §7.3 quantize against the decoder | LF and HF, exact |
//! | [`request`] | §12 budgets | the fields that exist |
//! | [`rate`] | §12 rate control | exact rate loop, fixed blocks (M4) |
//! | [`plan_frame`] | §16 orchestration | builds and validates a plan |
//!
//! # What milestone 2 does *not* do
//!
//! Everything in the fixed-DCT8x8 slice is a *fixed rule*, not a search:
//!
//! * **no block search** — one DCT8x8 per atom (milestone 6);
//! * **no adaptive quantization** — one `global_scale` and a constant `HfMul`
//!   for the whole frame (milestone 7);
//! * ~~no rate control~~ — **milestone 4 landed**: [`rate`] chooses
//!   `global_scale`/`HfMul` against a byte or bits-per-pixel target, pricing
//!   every candidate through the real writer. Without a
//!   [`RateTarget`](request::RateTarget) the caller still sets the scalars and
//!   nothing searches;
//! * ~~no chroma-from-luma estimation~~ — **milestone 5 landed**: frame-wide
//!   LF and per-64x64 HF factors are regressed, then refined over the exact
//!   integer representation I.6 consumes;
//! * **no entropy search** — [`cluster_of`] is a fixed six-way split and the
//!   coefficient orders are I.3.2's natural ones (milestone 8);
//! * **filter planning (partial)** — [`EncodeRequest::restoration`] defaults
//!   off; with `gaborish` set the planner inverse-Gaborish preconditions XYB
//!   before DCT (milestone 9 start). EPF iters may be signalled but have no
//!   encoder-side inverse yet.
//!
//! Each of those is where the compression is; what exists here is a correct
//! pipeline for them to improve.

pub mod analysis;
pub mod block;
pub mod candidate;
pub mod colour_reduce;
pub mod content_class;
pub mod csf;
pub mod diagnostics;
mod entropy;
mod entropy_cost;
pub mod error;
pub mod field;
pub mod navigation;
pub mod policy_bank;
pub mod quality;
pub mod quality_features;
pub mod quality_prediction;
pub mod quality_predictor;
#[cfg(feature = "case-predictor")]
pub mod quality_predictor_cases;
pub mod quality_predictor_v2;
pub mod quantize;
pub mod quantizer_ladder;
pub mod rate;
pub mod reducer;
pub mod regret;
pub mod request;
pub mod source;
pub mod stability;

use jpxl_core::dequant::{DequantMatrices, DequantMatrix};
use jpxl_core::forward::{
    CoeffView, CoeffViewMut, SampleView, SampleViewMut, TransformScratch, forward_varblock_into,
    lf_from_llf_into,
};
use jpxl_core::geometry::LfBlockPos;
use jpxl_core::varblock::TransformType;
#[cfg(test)]
use jpxl_encode::vardct::headers::NEUTRAL_QM_SCALE;
use jpxl_encode::vardct::headers::VARDCT_GROUP_SIZE_SHIFT;
use jpxl_encode::vardct::ids::{
    CflFactor, ClusterId, GlobalScale, HfMul, LfGroupId, PresetId, QuantLf,
};
use jpxl_encode::vardct::plan::PixelPlan;
use jpxl_encode::vardct::plan::{
    CflGrid, EmissionPlan, EntropyModelPlan, EntropyPlan, FrameDecision, HfBlockContextPlan,
    HfPassEntropyPlan, HistogramPlan, HybridUintPlan, LfCorrelationDecision, LfDecision,
    LfGroupPlan, LfQuantPlanes, OrderSet, QuantizedFrameIr, QuantizedLfGroup, QuantizerDecision,
    SectionLayout, SharpnessGrid, SpatialPlan, VarblockCoefficients, VarblockDecision,
};
use jpxl_encode::vardct::{
    ValidatedEmissionPlan, VardctGeometry, census_frame, census_frame_with_executor, validate,
};

use quantize::{
    CflAccumulator, DCT8X8_CELLS, DEFAULT_COLOUR_FACTOR, HfQuantizer, LfQuantizer, NUM_CHANNELS,
    cfl_multiplier,
};

pub use analysis::{
    AnalysisAtlas, AnalysisAtlasV2, AtomFeatures, AtomGrid, DiagnosticAtomFeatures,
};
pub use diagnostics::{
    ChooseStage, EncodeDiag, last_encode_diag, reset_encode_diag, take_encode_diag,
};
pub use entropy_cost::{EntropyCostSink, EntropyCostView};
pub use error::{PolicyError, Result};
pub use field::{AqMode, AqTuning};

use field::{DesiredQuantField, mul_lattice_for};
pub use policy_bank::{PerceptualPolicy, rank_alternatives};
pub use quality::{
    LadderPoint, PerceptualEvaluator, PerceptualObservation, PolicyTrial, ProbeKind, QualityBudget,
    QualityOutcome, QualityPredictionTrace, QualityProbe, QualityStats, QualityStatus, QualityWork,
    StructureSource, search_frame_perceptual, search_frame_perceptual_with_budget, status_name,
    sweep_frame_perceptual,
};
pub use quality_features::{SourceFeatures, TransformFeatureSummary, source_features};
pub use quality_prediction::{QualityPredictionV2, predict, predict_v2, shadow_prediction_trace};
pub use quantizer_ladder::{
    HF_MUL_RUNGS, LADDER_LEN, QuantizerChoice, Rung, effective_scale, rung_for_effective_scale,
};
pub use rate::{
    LadderSearch, RateOutcome, RatePhase, RateProbeStats, RateStatus, RateStep, search_frame,
};
#[cfg(feature = "anchor-sketch")]
pub use rate::{RateLadderPoint, sweep_frame_rate};
pub use request::{
    AdaptiveSharpness, ChromaHfPolicy, CoverFrequencyWeight, CoverMode, CoverRateModel,
    CoverSizePenalty, EncodeRequest, EpfSharpnessMode, QuantizerChoiceMode, RateSearchBudget,
    RateSearchPreset, RateTarget, RateTolerance, SearchBudget,
};
pub use source::PreparedFrame;
// Re-export so callers can set [`EncodeRequest::restoration`] without a
// second dependency path into the writer crate's plan module.
pub use jpxl_encode::vardct::plan::RestorationDecision;

use std::sync::Arc;
use std::{ops::Deref, slice};

/// I.4's per-block-context share of the `non_zeros` contexts.
const NON_ZEROS_CONTEXTS: u64 = 37;

/// I.4's per-block-context share of the coefficient contexts.
const COEFFICIENT_CONTEXTS: u64 = 458;

/// How many entropy clusters [`cluster_of`] produces.
const NUM_CLUSTERS: usize = 6;

/// The interoperable range of G.2.4's HF correlation samples.
///
/// Although the syntax is a Modular sample, a black-box boundary probe against
/// `djxl` established that `-128` and `127` round-trip consistently while
/// `-129` and `128` do not. Keep the policy inside the common range accepted
/// identically by all three oracle decoders.
const HF_FACTOR_MIN: i32 = -128;
const HF_FACTOR_MAX: i32 = 127;

/// Plans one VarDCT frame and hands back a plan the writer will accept.
///
/// This is `Encoder-plan1.md` §16's orchestration, at the size milestone 2
/// justifies: prepare, analyse, tile, quantize, model, lower, validate. The
/// stages are explicit function calls with typed inputs and outputs rather
/// than an `EncoderState` that mutates itself, so each can be profiled,
/// tested and replaced on its own.
///
/// # Errors
///
/// [`PolicyError::Plan`] if the plan this builds is rejected — which is always
/// a bug in this crate, never in the caller's image — and
/// [`PolicyError::Unsupported`] for a frame this policy cannot plan.
pub fn plan_frame(frame: &PreparedFrame, request: &EncodeRequest) -> Result<ValidatedEmissionPlan> {
    let atlas = AnalysisAtlas::analyze(frame);
    plan_frame_with_atlas(frame, &atlas, request)
}

/// [`plan_frame`] with an atlas the caller already computed.
///
/// If the request carries a [`RateTarget`], this runs the rate loop
/// ([`rate::search_frame`]) and returns the plan it chose; otherwise the
/// request's own quantizer scalars are used exactly as given.
///
/// # Errors
///
/// As [`plan_frame`], plus [`PolicyError::TargetUnreachable`] if no
/// representable quantizer fits the target.
pub fn plan_frame_with_atlas(
    frame: &PreparedFrame,
    atlas: &AnalysisAtlas,
    request: &EncodeRequest,
) -> Result<ValidatedEmissionPlan> {
    if let Some(target) = request.target {
        return Ok(rate::search_frame(frame, atlas, request, target)?.plan);
    }
    plan_at(
        frame,
        atlas,
        request,
        QuantizerChoice::from_request(request),
    )
}

/// How hard the planner works on entropy alternatives (Opt-V2 rate search).
///
/// **Fast** trains default block-context + natural coefficient orders only.
/// Its coded size is an **upper bound** on **Full**, because Full only adopts
/// custom orders / block contexts / multi-presets when they strictly shrink
/// the exact price. The rate loop uses that default model for navigation and
/// re-plans only the finalist with Full, so intermediate probes skip several
/// full `price_codestream`s.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EntropySearch {
    /// Default I.2.2 map, natural orders, one census + train.
    Fast,
    /// A finalist-navigation candidate: default I.2.2 map and natural orders,
    /// attributed to the finalist phase but without Full alternatives. This
    /// is also used by Quality to choose the quantizer before paying for the
    /// expensive alternative search on the finalist itself.
    FinalFast,
    /// Reuse an entropy model supplied by the anchored controller. The
    /// provisional model is retained only until the caller overlays the
    /// trained model; no census or entropy training is performed here.
    #[cfg(feature = "anchor-sketch")]
    Reuse,
    /// G5 candidate used for bounded-controller navigation: natural orders,
    /// the default context map, and at most two frame-ranked hybrid-uint
    /// configurations per context.
    #[cfg(feature = "g5-bounded-entropy")]
    BoundedAnchor,
    /// The same bounded model rebuilt at the exact finalist, separately
    /// attributed from anchor navigation.
    #[cfg(feature = "g5-bounded-entropy")]
    BoundedFinal,
    /// Slice-18 alternatives with exact-price adopt gates.
    Full,
}

impl EntropySearch {
    const fn uses_fast_entropy(self) -> bool {
        match self {
            Self::Fast => true,
            Self::FinalFast => true,
            #[cfg(feature = "anchor-sketch")]
            Self::Reuse => true,
            #[cfg(feature = "g5-bounded-entropy")]
            Self::BoundedAnchor | Self::BoundedFinal => true,
            Self::Full => false,
        }
    }
}

/// Plans one frame at an explicitly chosen, already-representable quantizer.
///
/// This is the function the rate loop calls once per exact price, and the one
/// [`plan_frame_with_atlas`] calls when there is no target. Splitting it out is
/// what keeps the loop from having to fabricate an [`EncodeRequest`] per
/// candidate — and what guarantees the plan a price was taken on and the plan
/// finally emitted are built by the same code.
///
/// # Errors
///
/// As [`plan_frame`].
pub(crate) fn plan_at(
    frame: &PreparedFrame,
    atlas: &AnalysisAtlas,
    request: &EncodeRequest,
    quantizer: QuantizerChoice,
) -> Result<ValidatedEmissionPlan> {
    let mut cache = CandidateForwardCache::new();
    let executor = request.resources.executor();
    plan_at_with_cfl(
        frame,
        atlas,
        request,
        quantizer,
        true,
        None,
        &mut cache,
        EntropySearch::Full,
        Some(&executor),
        AnchorReuse::None,
        None,
    )
}

/// Like [`plan_at`], but reuses a caller-prepared transform frame and a
/// cross-probe forward-transform cache (rate loop).
#[allow(
    clippy::too_many_arguments,
    reason = "internal plumbing: the parameter list is the stable bundle \\\n              plan_at_on_with_workspace forwards verbatim; bundling it \\\n              would add a struct used by exactly one call chain"
)]
pub(crate) fn plan_at_on(
    frame: &PreparedFrame,
    transform_frame: &PreparedFrame,
    atlas: &AnalysisAtlas,
    request: &EncodeRequest,
    quantizer: QuantizerChoice,
    cache: &mut CandidateForwardCache,
    entropy: EntropySearch,
    executor: Option<&jpxl_encode::EncodeExecutor>,
) -> Result<ValidatedEmissionPlan> {
    let mut quant_workspace = QuantizationWorkspace::new();
    plan_at_on_with_workspace(
        frame,
        transform_frame,
        atlas,
        request,
        quantizer,
        cache,
        entropy,
        executor,
        &mut quant_workspace,
    )
}

/// [`plan_at_on`] with request-scoped quantization storage supplied by the
/// rate controller.
#[allow(
    clippy::too_many_arguments,
    reason = "internal plumbing: every parameter is a distinct capability \\\n              (frame, atlas, request, quantizer, caches, executor, workspace) \\\n              with no cohesive sub-bundle to extract"
)]
fn plan_at_on_with_workspace(
    frame: &PreparedFrame,
    transform_frame: &PreparedFrame,
    atlas: &AnalysisAtlas,
    request: &EncodeRequest,
    quantizer: QuantizerChoice,
    cache: &mut CandidateForwardCache,
    entropy: EntropySearch,
    executor: Option<&jpxl_encode::EncodeExecutor>,
    quant_workspace: &mut QuantizationWorkspace,
) -> Result<ValidatedEmissionPlan> {
    plan_at_with_cfl_workspace(
        frame,
        atlas,
        request,
        quantizer,
        true,
        Some(transform_frame),
        cache,
        entropy,
        executor,
        AnchorReuse::None,
        None,
        quant_workspace,
    )
}

/// Fast-preset planning entry point with an explicit reusable spatial anchor.
///
/// Navigation captures the selected cover and CfL policy, then reuses those
/// choices while retargeting quantizer-dependent `HfMul` values. The Fast
/// preset deliberately captures a neutral-CfL/fixed-cover structure. Balanced
/// captures the configured hierarchical cover with CfL, then trains at most
/// two ranked hybrid-uint configurations at its near-target anchor and
/// finalist. A bounded correction may retain that finalist structure and
/// entropy model after its exact size is known.
/// Fast-preset planning with rate-search-owned quantization storage and an
/// explicit reusable spatial anchor.
#[cfg(feature = "anchor-sketch")]
#[allow(
    clippy::too_many_arguments,
    reason = "internal plumbing: extends plan_at_on_with_workspace with the \\\n              three anchor-reuse parameters; one call chain, no reuse elsewhere"
)]
fn plan_at_on_anchor_with_workspace(
    frame: &PreparedFrame,
    transform_frame: &PreparedFrame,
    atlas: &AnalysisAtlas,
    request: &EncodeRequest,
    quantizer: QuantizerChoice,
    enable_cfl: bool,
    cache: &mut CandidateForwardCache,
    entropy: EntropySearch,
    executor: Option<&jpxl_encode::EncodeExecutor>,
    reuse: AnchorReuse<'_>,
    capture: Option<&mut Option<StructuralAnchor>>,
    quant_workspace: &mut QuantizationWorkspace,
) -> Result<ValidatedEmissionPlan> {
    plan_at_with_cfl_workspace(
        frame,
        atlas,
        request,
        quantizer,
        enable_cfl,
        Some(transform_frame),
        cache,
        entropy,
        executor,
        reuse,
        capture,
        quant_workspace,
    )
}

/// The pre-entropy form of [`plan_at_on_anchor_with_workspace`]: the
/// candidate's pixels, validated, with its geometry — what a perceptual probe
/// renders and scores. No histogram is trained.
#[allow(
    clippy::too_many_arguments,
    reason = "internal plumbing: the same anchor-reuse bundle as \\
              plan_at_on_anchor_with_workspace; one call chain"
)]
pub(crate) fn plan_pixels_on_anchor_with_workspace(
    frame: &PreparedFrame,
    transform_frame: &PreparedFrame,
    atlas: &AnalysisAtlas,
    request: &EncodeRequest,
    quantizer: QuantizerChoice,
    enable_cfl: bool,
    cache: &mut CandidateForwardCache,
    structure_tier: EntropySearch,
    executor: Option<&jpxl_encode::EncodeExecutor>,
    reuse: AnchorReuse<'_>,
    capture: Option<&mut Option<StructuralAnchor>>,
    quant_workspace: &mut QuantizationWorkspace,
) -> Result<(
    jpxl_encode::vardct::ValidatedPixelPlan,
    jpxl_encode::vardct::VardctGeometry,
)> {
    let (pixels, geometry) = build_pixel_plan(
        frame,
        atlas,
        request,
        quantizer,
        enable_cfl,
        Some(transform_frame),
        cache,
        structure_tier,
        executor,
        reuse,
        capture,
        quant_workspace,
    )?;
    Ok((jpxl_encode::vardct::validate_pixels(pixels)?, geometry))
}

/// [`plan_at`] with the Slice-15 search switch exposed for regression tests.
///
/// Production always enables CfL. The disabled arm exists only to preserve a
/// byte-for-byte pre-slice-15 baseline for the exit evidence; it is not a
/// public encoder knobs.
///
/// `transform_override`, when `Some`, is the XYB frame DCT/quantize must see
/// (preconditioned). When `None`, inverse-Gaborish is applied here if the
/// request asks for it — once per call, which the rate loop must not do.
///
/// `cache` holds quantizer-independent forward DCTs so cover search, CfL, and
/// quantization share them, and so the rate loop reuses them across probes.
#[allow(
    clippy::too_many_arguments,
    reason = "internal plumbing: the rate-loop bundle forwarded verbatim to \\\n              plan_at_with_cfl_workspace; bundling would split parameters \\\n              across two structs for one call chain"
)]
fn plan_at_with_cfl(
    frame: &PreparedFrame,
    atlas: &AnalysisAtlas,
    request: &EncodeRequest,
    quantizer: QuantizerChoice,
    enable_cfl: bool,
    transform_override: Option<&PreparedFrame>,
    cache: &mut CandidateForwardCache,
    entropy_search: EntropySearch,
    executor: Option<&jpxl_encode::EncodeExecutor>,
    reuse: AnchorReuse<'_>,
    capture: Option<&mut Option<StructuralAnchor>>,
) -> Result<ValidatedEmissionPlan> {
    let mut quant_workspace = QuantizationWorkspace::new();
    plan_at_with_cfl_workspace(
        frame,
        atlas,
        request,
        quantizer,
        enable_cfl,
        transform_override,
        cache,
        entropy_search,
        executor,
        reuse,
        capture,
        &mut quant_workspace,
    )
}

/// The rate-loop form of [`plan_at_with_cfl`] with caller-owned coefficient
/// storage. A rate search keeps this workspace across its sequential probes;
/// the plans themselves retain only an `Arc` handle to the arena until their
/// exact price is complete.
#[allow(
    clippy::too_many_arguments,
    reason = "internal plumbing: every parameter is a distinct capability \\\n              supplied by a different owner (request, rate loop, executor); \\\n              no cohesive sub-bundle to extract"
)]
fn plan_at_with_cfl_workspace(
    frame: &PreparedFrame,
    atlas: &AnalysisAtlas,
    request: &EncodeRequest,
    quantizer: QuantizerChoice,
    enable_cfl: bool,
    transform_override: Option<&PreparedFrame>,
    cache: &mut CandidateForwardCache,
    entropy_search: EntropySearch,
    executor: Option<&jpxl_encode::EncodeExecutor>,
    reuse: AnchorReuse<'_>,
    capture: Option<&mut Option<StructuralAnchor>>,
    quant_workspace: &mut QuantizationWorkspace,
) -> Result<ValidatedEmissionPlan> {
    let (pixels, geometry) = build_pixel_plan(
        frame,
        atlas,
        request,
        quantizer,
        enable_cfl,
        transform_override,
        cache,
        entropy_search,
        executor,
        reuse,
        capture,
        quant_workspace,
    )?;
    attach_entropy(&pixels, &geometry, request, entropy_search, executor)
}

/// Everything a decoder's pixels depend on: source preparation, cover, CfL,
/// quantization and the LF planes, assembled into a [`PixelPlan`] with the
/// frame's geometry. No histogram is trained and no symbol is counted.
#[allow(
    clippy::too_many_arguments,
    reason = "internal plumbing: the same capability bundle plan_at_with_cfl_workspace \\
              forwards; no cohesive sub-bundle to extract"
)]
fn build_pixel_plan(
    frame: &PreparedFrame,
    atlas: &AnalysisAtlas,
    request: &EncodeRequest,
    quantizer: QuantizerChoice,
    enable_cfl: bool,
    transform_override: Option<&PreparedFrame>,
    cache: &mut CandidateForwardCache,
    entropy_search: EntropySearch,
    executor: Option<&jpxl_encode::EncodeExecutor>,
    reuse: AnchorReuse<'_>,
    capture: Option<&mut Option<StructuralAnchor>>,
    quant_workspace: &mut QuantizationWorkspace,
) -> Result<(PixelPlan, jpxl_encode::vardct::VardctGeometry)> {
    // Phase-0: one clean snapshot per plan_at (rate probes overwrite; last wins).
    diagnostics::reset_encode_diag();
    if request.restoration.epf_iters > 3 {
        return Err(PolicyError::Unsupported {
            what: "epf_iters outside 0..=3",
        });
    }
    debug_assert!(
        request.rate_preset != RateSearchPreset::Quality || matches!(reuse, AnchorReuse::None),
        "Quality must not receive a globally frozen structural anchor"
    );

    let decision = FrameDecision {
        width: frame.width(),
        height: frame.height(),
        // F.2 signals `group_size_shift` only for kModular, so a kVarDCT frame
        // has no say: `group_dim` is 256 and the request's field is ignored
        // rather than silently written into a field that does not exist.
        group_size_shift: VARDCT_GROUP_SIZE_SHIFT,
        num_passes: 1,
        bits_per_sample: request.bits_per_sample,
    };
    let geometry = decision.geometry()?;

    debug_assert_eq!(
        atlas.grid().area(),
        geometry.frame_blocks().area(),
        "the atom grid and the block grid are the same grid"
    );

    // When Gaborish will run on the decoder, DCT/quantize against the
    // inverse-filtered planes so J.3 restores the intended XYB. Analysis
    // (AQ masking) stays on the source planes — those describe the image the
    // user sees after restoration. The rate loop passes a precomputed
    // transform frame so this precondition runs once per request, not once
    // per quantizer probe.
    let precond_frame;
    let transform_frame: &PreparedFrame = if let Some(tf) = transform_override {
        tf
    } else if request.restoration.gaborish {
        precond_frame = prepare_gaborish_frame(frame)?;
        &precond_frame
    } else {
        frame
    };

    // §7.2's factorization first: with an AQ field the wire quantizer scalars
    // differ from the request's (exactly compensated), and everything below
    // quantizes against the wire values.
    let aq = AqSetup::build(atlas, request, quantizer);
    let lf_quant = LfQuantizer::new(
        aq.global_scale.get(),
        aq.quant_lf.get(),
        LfDecision::vardct_neutral().extra_precision,
    );
    // Trailing-truncation is the target-rate Quality policy, but it requires
    // a backwards nonzero scan for every selected block. Fast navigation uses
    // the already-vectorized nearest choice; Balanced and Quality retain the
    // promoted truncation policy exactly.
    let quantizer_choice =
        if request.rate_preset == RateSearchPreset::Fast && entropy_search.uses_fast_entropy() {
            QuantizerChoiceMode::Nearest
        } else {
            request.quantizer_choice
        };
    // Fast navigation is deliberately allowed a cheaper structural policy.
    // Fixed 8x8 blocks avoid the hierarchical cover's transform-bank scoring
    // throughout Fast navigation and finalist planning. The Quality request
    // keeps its configured mode. This is a preset-only trade: no Quality/Full
    // plan can enter this arm because those requests do not carry
    // `RateSearchPreset::Fast` here.
    let cover_mode =
        if request.rate_preset == RateSearchPreset::Fast && entropy_search.uses_fast_entropy() {
            CoverMode::FixedDct8x8
        } else {
            request.budget.cover_mode
        };
    let quantizer_transforms = if cover_mode == CoverMode::FixedDct8x8 {
        &FAST_TRANSFORMS[..]
    } else {
        &SQUARE_TRANSFORMS[..]
    };
    let hf_quants = HfQuantizers::new_with_scales_for_transforms(
        aq.global_scale.get(),
        aq.baseline,
        &aq.muls(),
        request.x_qm_scale.get(),
        request.b_qm_scale.get(),
        request.cover_size_penalty,
        request.cover_frequency_weight,
        quantizer_choice,
        request.lambda_scale,
        quantizer_transforms,
    )?
    .with_dead_zone_scale(request.dead_zone_scale)
    .with_zero_token_bits(request.zero_token_bits)
    .with_rate_model(request.cover_rate_model);
    // Reserve only the square families this search will score, on the request
    // thread, before cover fans out. Fast's fixed DCT8x8 cover never touches
    // DCT16/DCT32, and pre-creating those banks reserved two empty full-grid
    // coefficient arenas per LF group (about two thirds of the forward-cache
    // commit on the production Fast path).
    cache.prepare_families(&geometry, quantizer_transforms)?;

    // The cover is selected before chroma-from-luma is estimated: the estimate
    // regresses over the coefficients of the *selected* transforms, so the
    // block map must exist first. Selection itself scores with neutral CfL —
    // block choice is dominated by luma structure (see `block_cost`).
    // Cover selection, then one forward transform per *selected* varblock.
    // CfL estimation and HF quantization both consume those coefficients so
    // a selected DCT is not recomputed (Opt-V within-probe cache).
    let (groups, cfl) = if let AnchorReuse::CoverAndCfl(anchor) = reuse {
        if anchor.groups.len() != usize::try_from(geometry.num_lf_groups()).unwrap_or(usize::MAX) {
            return Err(PolicyError::Unsupported {
                what: "a structural anchor whose LF-group count changed",
            });
        }
        let mut groups = anchor.groups.clone();
        retarget_anchor_groups(&mut groups, &aq)?;
        (groups, Arc::clone(&anchor.cfl))
    } else if let AnchorReuse::CoverOnly(anchor) = reuse {
        if anchor.groups.len() != usize::try_from(geometry.num_lf_groups()).unwrap_or(usize::MAX) {
            return Err(PolicyError::Unsupported {
                what: "a structural anchor whose LF-group count changed",
            });
        }
        let mut groups = anchor.groups.clone();
        retarget_anchor_groups(&mut groups, &aq)?;
        let maps: Vec<PlannedVarblockRange> = groups
            .iter()
            .map(|(_, _, _, varblocks)| varblocks.range(0, varblocks.len()))
            .collect::<Result<Vec<_>>>()?;
        let cfl = diagnostics::time_stage(diagnostics::StageTimer::Cfl, || {
            estimate_cfl(
                &geometry,
                &maps,
                cache,
                &lf_quant,
                &hf_quants,
                enable_cfl && !frame.is_grayscale(),
                executor,
            )
        })?;
        (groups, Arc::new(cfl))
    } else {
        // Phase 8.2: every LF group owns independent dense transform banks.
        // Cover construction can therefore run on the request executor without
        // a global hash-table mutation point; fixed-index reduction restores
        // raster order.
        let groups: Vec<PlannedGroup> =
            diagnostics::time_stage(diagnostics::StageTimer::Cover, || {
                diagnostics::with_choose_stage(diagnostics::ChooseStage::Cover, || {
                    let n_groups = usize::try_from(geometry.num_lf_groups()).unwrap_or(0);
                    let split_cover = cfg!(feature = "parallel")
                        && executor.is_some_and(|executor| {
                            executor.resources().parallel_groups()
                                && n_groups < executor.resources().threads
                        });
                    let plan_group = |index_usize: usize| {
                        let index = u64::try_from(index_usize).unwrap_or(u64::MAX);
                        let id = LfGroupId::new(u32::try_from(index).unwrap_or(u32::MAX));
                        let blocks =
                            geometry
                                .lf_group_blocks(id)
                                .ok_or(PolicyError::Unsupported {
                                    what: "an LF group outside the frame's grid",
                                })?;
                        let rect = geometry.lf_group_rect(id).ok_or(PolicyError::Unsupported {
                            what: "an LF group outside the frame's grid",
                        })?;
                        let varblocks = match cover_mode {
                            CoverMode::FixedDct8x8 => {
                                let mut varblocks = block::fixed_dct8x8(blocks, aq.baseline)?;
                                for vb in &mut varblocks {
                                    vb.hf_mul = aq.mul_for_footprint(
                                        rect.x0 / 8 + vb.origin.bx(),
                                        rect.y0 / 8 + vb.origin.by(),
                                        1,
                                        1,
                                    );
                                }
                                let mut bank = cache.group(index_usize)?.write().map_err(|_| {
                                    PolicyError::Unsupported {
                                        what: "a poisoned LF-group forward bank",
                                    }
                                })?;
                                ensure_forwards_cached(
                                    transform_frame,
                                    &varblocks,
                                    (rect.x0, rect.y0),
                                    &mut bank,
                                    &mut ForwardScratch::new(),
                                )?;
                                varblocks
                            }
                            CoverMode::Hierarchical if split_cover => {
                                {
                                    let mut bank =
                                        cache.group(index_usize)?.write().map_err(|_| {
                                            PolicyError::Unsupported {
                                                what: "a poisoned LF-group forward bank",
                                            }
                                        })?;
                                    ensure_cover_candidates_cached(
                                        transform_frame,
                                        blocks,
                                        (rect.x0, rect.y0),
                                        &mut bank,
                                        &mut ForwardScratch::new(),
                                    )?;
                                }
                                let executor = executor.ok_or(PolicyError::Unsupported {
                                    what: "parallel cover regions without an executor",
                                })?;
                                let bank = cache.group(index_usize)?.read().map_err(|_| {
                                    PolicyError::Unsupported {
                                        what: "a poisoned LF-group forward bank",
                                    }
                                })?;
                                select_blocks_cached_parallel(
                                    transform_frame,
                                    &hf_quants,
                                    blocks,
                                    (rect.x0, rect.y0),
                                    &aq,
                                    &bank,
                                    executor,
                                )?
                            }
                            CoverMode::Hierarchical => {
                                let mut bank = cache.group(index_usize)?.write().map_err(|_| {
                                    PolicyError::Unsupported {
                                        what: "a poisoned LF-group forward bank",
                                    }
                                })?;
                                select_blocks(
                                    transform_frame,
                                    &hf_quants,
                                    blocks,
                                    (rect.x0, rect.y0),
                                    &aq,
                                    &mut bank,
                                    &mut ForwardScratch::new(),
                                )?
                            }
                        };
                        Ok::<_, PolicyError>((id, blocks, rect, PlannedVarblocks::owned(varblocks)))
                    };
                    if let Some(executor) = executor {
                        executor.map_ordered(n_groups, plan_group)
                    } else {
                        (0..n_groups).map(plan_group).collect()
                    }
                })
            })?;

        let maps: Vec<PlannedVarblockRange> = groups
            .iter()
            .map(|(_, _, _, varblocks)| varblocks.range(0, varblocks.len()))
            .collect::<Result<Vec<_>>>()?;
        let cfl = diagnostics::time_stage(diagnostics::StageTimer::Cfl, || {
            estimate_cfl(
                &geometry,
                &maps,
                cache,
                &lf_quant,
                &hf_quants,
                enable_cfl && !frame.is_grayscale(),
                executor,
            )
        })?;
        (groups, Arc::new(cfl))
    };

    if let Some(slot) = capture {
        *slot = Some(StructuralAnchor {
            groups: groups
                .iter()
                .map(|(id, blocks, rect, varblocks)| Ok((*id, *blocks, *rect, varblocks.shared()?)))
                .collect::<Result<Vec<_>>>()?,
            cfl: Arc::clone(&cfl),
        });
    }

    let quantized_groups =
        diagnostics::time_stage_units(diagnostics::StageTimer::Quantize, groups.len(), || {
            if let Some(executor) = executor
                && executor.resources().parallel_groups()
                && executor.resources().threads > 1
            {
                // LF groups can differ substantially in selected transform
                // mix. Keep sub-group work stealing available even when the
                // group count happens to equal the worker count: one heavy
                // group must not pin the request after the other workers have
                // drained their whole-group jobs. Fixed-index reduction below
                // preserves the serial coefficient and LF-plane order.
                return quantize_groups_parallel(
                    &groups,
                    &cfl,
                    cache,
                    &lf_quant,
                    &hf_quants,
                    executor,
                    quant_workspace,
                );
            }
            // As with CfL samples, reserve the frame-sized output arenas on
            // the request thread and transfer ownership to workers only after
            // allocation. This keeps repeated rate probes from accumulating
            // large worker-local allocator arenas.
            let quant_workspaces: Vec<_> = groups
                .iter()
                .enumerate()
                .map(|(index, (_, blocks, _, varblocks))| {
                    let arena =
                        quant_workspace.take_arena(index, coefficient_arena_capacity(varblocks));
                    std::sync::Mutex::new(Some(QuantWorkspace::new(varblocks, *blocks, arena)))
                })
                .collect();
            let quantize_one = |index: usize| {
                let (_, blocks, rect, varblocks) =
                    groups.get(index).ok_or(PolicyError::Unsupported {
                        what: "a missing planned LF group before quantization",
                    })?;
                let group_cfl = cfl.groups.get(index).ok_or(PolicyError::Unsupported {
                    what: "a missing CfL grid for an LF group",
                })?;
                let bank = cache
                    .group(index)?
                    .read()
                    .map_err(|_| PolicyError::Unsupported {
                        what: "a poisoned LF-group forward bank",
                    })?;
                let varblock_range = varblocks.range(0, varblocks.len())?;
                let forwards =
                    gather_forward_refs(varblock_range.decisions, (rect.x0, rect.y0), &bank)?;
                let workspace = quant_workspaces
                    .get(index)
                    .ok_or(PolicyError::Unsupported {
                        what: "a missing quantization workspace",
                    })?
                    .lock()
                    .map_err(|_| PolicyError::Unsupported {
                        what: "a poisoned quantization workspace",
                    })?
                    .take()
                    .ok_or(PolicyError::Unsupported {
                        what: "a quantization workspace used twice",
                    })?;
                diagnostics::with_choose_stage(diagnostics::ChooseStage::Final, || {
                    quantize_group(
                        &lf_quant,
                        &hf_quants,
                        &cfl.correlation,
                        group_cfl,
                        &varblock_range,
                        &forwards,
                        *blocks,
                        workspace,
                    )
                })
            };
            if let Some(executor) = executor {
                executor.map_ordered(groups.len(), quantize_one)
            } else {
                (0..groups.len()).map(quantize_one).collect()
            }
        })?;

    let mut lf_groups = Vec::new();
    let mut quantized = Vec::new();
    for (index, ((id, blocks, _rect, varblocks), quantized_group)) in
        groups.into_iter().zip(quantized_groups).enumerate()
    {
        if let Some(arena) = quantized_group.arena {
            quant_workspace.put_arena(index, arena);
        }
        let group_cfl = cfl.groups.get(index).ok_or(PolicyError::Unsupported {
            what: "a missing CfL grid for an LF group",
        })?;
        let sharpness = match request.epf_sharpness {
            EpfSharpnessMode::Zero => SharpnessGrid::zeros(blocks),
            EpfSharpnessMode::Uniform7 => {
                let len = usize::try_from(blocks.area()).unwrap_or(0);
                SharpnessGrid::new(blocks, vec![7; len])?
            }
            EpfSharpnessMode::Adaptive(model) => {
                let len = usize::try_from(blocks.area()).unwrap_or(0);
                let mut values = Vec::with_capacity(len);
                for by in 0..blocks.height {
                    for bx in 0..blocks.width {
                        let activity = atlas
                            .atom(_rect.x0 / 8 + bx, _rect.y0 / 8 + by)
                            .map_or(0.0, |f| (1.0 + f.variance_xyb[1] * 255.0 * 255.0).log2());
                        values.push(model.sharpness_for(activity));
                    }
                }
                SharpnessGrid::new(blocks, values)?
            }
        };
        lf_groups.push(LfGroupPlan {
            id,
            blocks: varblocks.into_boxed_slice(),
            cfl: group_cfl.clone(),
            sharpness,
        });
        quantized.push(QuantizedLfGroup {
            id,
            lf: quantized_group.lf,
            coefficients: quantized_group.coefficients.into_boxed_slice(),
        });
    }

    let mut lf = LfDecision::vardct_neutral();
    lf.correlation = cfl.correlation;
    let spatial = SpatialPlan {
        frame: decision,
        quantizer: QuantizerDecision {
            global_scale: aq.global_scale,
            quant_lf: aq.quant_lf,
            x_qm_scale: request.x_qm_scale,
            b_qm_scale: request.b_qm_scale,
        },
        lf,
        restoration: request.restoration,
        lf_groups: lf_groups.into_boxed_slice(),
    };

    let quantized_ir = QuantizedFrameIr {
        lf_groups: quantized.into_boxed_slice(),
    };
    Ok((PixelPlan::new(spatial, quantized_ir), geometry))
}

/// Trains the entropy models for a pixel plan and adopts the entropy
/// alternatives the search tier allows, returning the writer-ready plan.
///
/// The entropy model is chosen in two steps because the census is a
/// function of the plan: a provisional plan carries the clustering and the
/// hybrid-uint configuration, `census_frame` walks it, and the real
/// histograms replace the provisional ones. The walk lives in `jpxl-encode`
/// so that the counts trained here and the symbols emitted there cannot
/// come from two different traversals.
pub(crate) fn attach_entropy(
    pixels: &PixelPlan,
    geometry: &jpxl_encode::vardct::VardctGeometry,
    request: &EncodeRequest,
    entropy_search: EntropySearch,
    executor: Option<&jpxl_encode::EncodeExecutor>,
) -> Result<ValidatedEmissionPlan> {
    let provisional = EmissionPlan::from_pixels(
        pixels,
        entropy_plan(
            geometry,
            placeholder_histograms(),
            HfBlockContextPlan::Default,
        )?,
        SectionLayout::for_geometry(geometry),
    );
    // Slice 18 / 18b: train under the default I.2.2 map, then optionally
    // adopt custom coefficient orders on an exact price win (Full only).
    let with_default = diagnostics::time_stage(diagnostics::StageTimer::Entropy, || {
        train_entropy_with_orders(
            provisional.clone(),
            geometry,
            entropy_search,
            executor,
            matches!(
                request.rate_preset,
                RateSearchPreset::Fast | RateSearchPreset::Balanced
            ) && entropy_search.uses_fast_entropy(),
        )
    })?;

    // Fast rate probes stop here: default map + natural orders is an upper
    // bound on Full's size (Full only adopts alternatives that strictly win).
    if entropy_search.uses_fast_entropy() {
        return Ok(with_default.into_plan());
    }

    // Slice 18c: a custom I.2.2 block context changes every pre-context id,
    // so it gets its own census + train + order pass, and is adopted only
    // when the writer's exact price is strictly smaller. Flat / constant-mul
    // content proposes Default and skips the second walk.
    let mut best = with_default;
    let candidate_bc = entropy::propose_block_context(
        best.plan().spatial.as_ref(),
        best.plan().quantized.as_ref(),
    );
    if !matches!(candidate_bc, HfBlockContextPlan::Default) {
        diagnostics::note_block_context_candidate();
        let mut custom_walk = provisional.clone();
        custom_walk.entropy = entropy_plan(geometry, placeholder_histograms(), candidate_bc)?;
        let mut with_custom = diagnostics::time_stage(diagnostics::StageTimer::Entropy, || {
            train_entropy_with_orders(custom_walk, geometry, EntropySearch::Full, executor, false)
        })?;
        let best_size = best.exact_size(executor)?;
        let custom_size = with_custom.exact_size(executor)?;
        if custom_size < best_size {
            best = with_custom;
        }
    }

    // Slice 18d: multi-preset assignment. Needs ≥2 pass groups; changes the
    // walk's I.4 offset per group, so re-census + retrain + exact price.
    if let Some((num_presets, assignment)) = entropy::propose_presets(
        geometry,
        best.plan().spatial.as_ref(),
        best.plan().quantized.as_ref(),
    ) {
        diagnostics::note_preset_candidate();
        let mut multi = best.plan().clone();
        multi.entropy.num_hf_presets = num_presets;
        if let Some(pass) = multi.entropy.passes.first_mut() {
            // Stretch the provisional context map to the multi-preset
            // pre-context count so census_frame has a legal plan shape; the
            // trainer replaces it immediately after the walk.
            let nb = multi.entropy.block_context.nb_block_ctx();
            let pre = 495 * u64::from(num_presets) * nb;
            let map_len = usize::try_from(pre).unwrap_or(0);
            pass.group_presets = assignment.into_boxed_slice();
            pass.distributions.context_map = (0..map_len)
                .map(|ctx| ClusterId::new(cluster_of(u64::try_from(ctx).unwrap_or(0), nb.max(1))))
                .collect::<Vec<_>>()
                .into_boxed_slice();
            // Histograms stay the provisional six; train_entropy_with_orders
            // rebuilds them from the multi-offset census.
        }
        if let Ok(mut with_presets) =
            diagnostics::time_stage(diagnostics::StageTimer::Entropy, || {
                train_entropy_with_orders(multi, geometry, EntropySearch::Full, executor, false)
            })
        {
            let best_size = best.exact_size(executor)?;
            let multi_size = with_presets.exact_size(executor)?;
            if multi_size < best_size {
                best = with_presets;
            }
        }
    }

    Ok(best.into_plan())
}

/// A trained entropy finalist together with an exact Count result when its
/// order comparison already had to emit the plan. Carrying the price across
/// block-context and preset comparisons avoids re-emitting an unchanged
/// winner while preserving every existing strict-less-than tie rule.
struct TrainedEntropyCandidate {
    plan: ValidatedEmissionPlan,
    exact_size: Option<u64>,
}

impl TrainedEntropyCandidate {
    fn unpriced(plan: ValidatedEmissionPlan) -> Self {
        Self {
            plan,
            exact_size: None,
        }
    }

    fn priced(plan: ValidatedEmissionPlan, exact_size: u64) -> Self {
        Self {
            plan,
            exact_size: Some(exact_size),
        }
    }

    fn plan(&self) -> &EmissionPlan {
        self.plan.plan()
    }

    fn exact_size(&mut self, executor: Option<&jpxl_encode::EncodeExecutor>) -> Result<u64> {
        if let Some(size) = self.exact_size {
            return Ok(size);
        }
        let size = internal_price_total(&self.plan, executor)?;
        self.exact_size = Some(size);
        Ok(size)
    }

    fn into_plan(self) -> ValidatedEmissionPlan {
        self.plan
    }
}

/// Exact Count price used only to compare alternatives inside one Full plan.
fn internal_price_total(
    plan: &ValidatedEmissionPlan,
    executor: Option<&jpxl_encode::EncodeExecutor>,
) -> Result<u64> {
    diagnostics::with_count_kind(
        jpxl_encode::vardct::diagnostics::CountEmissionKind::Internal,
        || {
            let sizing = if let Some(executor) = executor {
                jpxl_encode::vardct::price_codestream_with(plan, executor)?
            } else {
                jpxl_encode::vardct::price_codestream(plan)?
            };
            Ok(sizing.total)
        },
    )
}

/// Test hook: sizes of the default-map plan and of a forced custom-map plan
/// over the same spatial/quantized IR, used to measure whether a proposal
/// can win before the adopt gate runs.
#[cfg(test)]
fn price_default_and_custom(
    spatial: SpatialPlan,
    quantized: QuantizedFrameIr,
    geometry: &VardctGeometry,
    candidate: HfBlockContextPlan,
) -> Result<(u64, u64)> {
    let provisional = EmissionPlan::new(
        spatial,
        quantized,
        entropy_plan(
            geometry,
            placeholder_histograms(),
            HfBlockContextPlan::Default,
        )?,
        SectionLayout::for_geometry(geometry),
    );
    let mut with_default = train_entropy_with_orders(
        provisional.clone(),
        geometry,
        EntropySearch::Full,
        None,
        false,
    )?;
    let mut custom_walk = provisional;
    custom_walk.entropy = entropy_plan(geometry, placeholder_histograms(), candidate)?;
    let mut with_custom =
        train_entropy_with_orders(custom_walk, geometry, EntropySearch::Full, None, false)?;
    Ok((
        with_default.exact_size(None)?,
        with_custom.exact_size(None)?,
    ))
}

/// Trains clusters / hybrid-uint from a census of `provisional`, then optionally
/// runs the §9.4 order candidate and keeps it only on an exact price win.
fn train_entropy_with_orders(
    provisional: EmissionPlan,
    geometry: &VardctGeometry,
    entropy_search: EntropySearch,
    executor: Option<&jpxl_encode::EncodeExecutor>,
    fast_hybrid_uint: bool,
) -> Result<TrainedEntropyCandidate> {
    #[cfg(feature = "anchor-sketch")]
    if matches!(entropy_search, EntropySearch::Reuse) {
        return Ok(TrainedEntropyCandidate::unpriced(validate(provisional)?));
    }
    if entropy_search.uses_fast_entropy() {
        diagnostics::note_census();
        diagnostics::note_entropy_training();
        let natural = train_entropy_for_orders(
            provisional,
            geometry,
            OrderSet::natural(),
            entropy_search,
            executor,
            fast_hybrid_uint,
        )?;
        return Ok(TrainedEntropyCandidate::unpriced(natural));
    }

    let orders =
        entropy::candidate_orders(provisional.spatial.as_ref(), provisional.quantized.as_ref())?;
    if orders.overrides().is_empty() {
        diagnostics::note_census();
        diagnostics::note_entropy_training();
        let natural = train_entropy_for_orders(
            provisional,
            geometry,
            OrderSet::natural(),
            entropy_search,
            executor,
            fast_hybrid_uint,
        )?;
        return Ok(TrainedEntropyCandidate::unpriced(natural));
    }
    diagnostics::note_order_candidate();
    // Natural and custom-order models consume the same immutable spatial and
    // quantized IR. Train them as two ordered executor jobs so their serial
    // cluster searches overlap, while retaining deterministic reduction and
    // the existing serial fallback at one worker / without an executor.
    diagnostics::note_census();
    diagnostics::note_entropy_training();
    diagnostics::note_census();
    diagnostics::note_entropy_training();
    let (natural, reordered) = if let Some(executor) = executor
        && executor.resources().parallel_groups()
    {
        let candidates = executor.map_ordered(2, |index| {
            let candidate_orders = if index == 0 {
                OrderSet::natural()
            } else {
                orders.clone()
            };
            train_entropy_for_orders(
                provisional.clone(),
                geometry,
                candidate_orders,
                entropy_search,
                Some(executor),
                fast_hybrid_uint,
            )
        })?;
        let mut candidates = candidates.into_iter();
        let natural = candidates.next().ok_or(PolicyError::Unsupported {
            what: "a missing natural-order entropy candidate",
        })?;
        let reordered = candidates.next().ok_or(PolicyError::Unsupported {
            what: "a missing custom-order entropy candidate",
        })?;
        (natural, reordered)
    } else {
        let natural = train_entropy_for_orders(
            provisional.clone(),
            geometry,
            OrderSet::natural(),
            entropy_search,
            executor,
            fast_hybrid_uint,
        )?;
        let reordered = train_entropy_for_orders(
            provisional,
            geometry,
            orders,
            entropy_search,
            executor,
            fast_hybrid_uint,
        )?;
        (natural, reordered)
    };

    let natural_size = internal_price_total(&natural, executor)?;
    let reordered_size = internal_price_total(&reordered, executor)?;
    Ok(if reordered_size < natural_size {
        TrainedEntropyCandidate::priced(reordered, reordered_size)
    } else {
        TrainedEntropyCandidate::priced(natural, natural_size)
    })
}

/// Census and train one coefficient-order candidate. The caller owns
/// candidate-level parallelism and records the diagnostic multiplicity on its
/// control thread, because those counters are intentionally thread-local.
fn train_entropy_for_orders(
    mut provisional: EmissionPlan,
    geometry: &VardctGeometry,
    orders: OrderSet,
    entropy_search: EntropySearch,
    executor: Option<&jpxl_encode::EncodeExecutor>,
    fast_hybrid_uint: bool,
) -> Result<ValidatedEmissionPlan> {
    if let Some(pass) = provisional.entropy.passes.first_mut() {
        pass.orders = orders.clone();
    }
    let block_context = provisional.entropy.block_context.clone();
    let num_hf_presets = provisional.entropy.num_hf_presets;
    let group_presets: Vec<PresetId> = provisional
        .entropy
        .passes
        .first()
        .map(|p| p.group_presets.to_vec())
        .unwrap_or_default();
    let census = if let Some(executor) = executor {
        census_frame_with_executor(&provisional, geometry, executor)?
    } else {
        census_frame(&provisional, geometry)?
    };
    #[cfg(feature = "g5-bounded-entropy")]
    let model = if matches!(
        entropy_search,
        EntropySearch::BoundedAnchor | EntropySearch::BoundedFinal
    ) {
        entropy::train_bounded_with_executor(&census, executor)?
    } else if fast_hybrid_uint && entropy_search.uses_fast_entropy() {
        entropy::train_fast_with_executor(&census, executor)?
    } else {
        entropy::train_with_executor(&census, executor)?
    };
    #[cfg(not(feature = "g5-bounded-entropy"))]
    let model = if fast_hybrid_uint && entropy_search.uses_fast_entropy() {
        entropy::train_fast_with_executor(&census, executor)?
    } else {
        entropy::train_with_executor(&census, executor)?
    };
    // Arc-clone spatial/quantized; only entropy is rebuilt.
    Ok(validate(EmissionPlan {
        entropy: trained_entropy_plan(
            geometry,
            model,
            orders,
            block_context,
            num_hf_presets,
            group_presets,
        )?,
        spatial: provisional.spatial,
        quantized: provisional.quantized,
        sections: provisional.sections,
    })?)
}

/// The frame-wide LF factors and one HF factor grid per LF group.
#[derive(Clone)]
struct CflEstimate {
    correlation: LfCorrelationDecision,
    groups: Vec<CflGrid>,
}

type PlannedGroup = (
    LfGroupId,
    jpxl_encode::vardct::BlockGrid,
    jpxl_encode::vardct::Rect,
    PlannedVarblocks,
);

/// Varblocks owned by a fresh plan, shared immutably by an anchored probe, or
/// shared under a compact per-probe `HfMul` overlay.
///
/// Fresh construction keeps its existing `Vec` ownership. Captured anchors
/// convert that vector to an `Arc<[VarblockDecision]>`; later probes can then
/// clone the group map without copying the decisions. Retargeting an `HfMul`
/// (Phase 24) layers a dense `Box<[HfMul]>` over the shared decisions instead
/// of copying the group's decisions: a probe re-pricing a different quantizer
/// pays four bytes per varblock, and only for groups whose multipliers moved.
///
/// **The `Deref`/`IntoIterator` views expose the base decisions.** Geometry
/// fields (`transform`, `origin`) are overlay-invariant, but `hf_mul` must be
/// read through [`PlannedVarblocks::hf_mul_at`] or a
/// [`PlannedVarblockRange`], which resolve the overlay.
#[derive(Clone)]
enum PlannedVarblocks {
    Owned(Vec<VarblockDecision>),
    Shared(Arc<[VarblockDecision]>),
    Retargeted {
        base: Arc<[VarblockDecision]>,
        muls: Box<[HfMul]>,
    },
}

impl PlannedVarblocks {
    fn owned(values: Vec<VarblockDecision>) -> Self {
        Self::Owned(values)
    }

    fn shared(&self) -> Result<Self> {
        match self {
            Self::Owned(values) => Ok(Self::Shared(Arc::from(values.clone().into_boxed_slice()))),
            Self::Shared(values) => Ok(Self::Shared(Arc::clone(values))),
            // A capture freezes this probe's effective multipliers, so the
            // overlay is applied before the decisions become the anchor's
            // shared base.
            Self::Retargeted { base, muls } => {
                let mut values = base.to_vec();
                Self::apply_overlay(&mut values, muls)?;
                Ok(Self::Shared(values.into()))
            }
        }
    }

    fn into_boxed_slice(self) -> Box<[VarblockDecision]> {
        match self {
            Self::Owned(values) => values.into_boxed_slice(),
            Self::Shared(values) => values.to_vec().into_boxed_slice(),
            Self::Retargeted { base, muls } => {
                let mut values = base.to_vec();
                // The constructor pairs an overlay with a base of the same
                // length, so this cannot fail; keeping the base values in
                // that impossible case avoids panicking on the wire path.
                if Self::apply_overlay(&mut values, &muls).is_err() {
                    values.clear();
                }
                values.into_boxed_slice()
            }
        }
    }

    /// Overwrites every decision's `hf_mul` with the overlay value. The only
    /// failure mode is an overlay whose length differs from its decisions,
    /// which the constructors pairing them already reject.
    fn apply_overlay(values: &mut [VarblockDecision], muls: &[HfMul]) -> Result<()> {
        if values.len() != muls.len() {
            return Err(PolicyError::Unsupported {
                what: "a retarget overlay whose length differs from its group",
            });
        }
        for (slot, mul) in values.iter_mut().zip(muls.iter()) {
            slot.hf_mul = *mul;
        }
        Ok(())
    }

    /// Layers `targets` (this probe's desired multiplier per varblock, in
    /// `BlockInfo` order) over the current storage, sharing the base
    /// decisions instead of copying them.
    fn retargeted(prior: Self, targets: Vec<HfMul>) -> Result<Self> {
        let base = match prior {
            Self::Owned(values) => Arc::from(values.into_boxed_slice()),
            Self::Shared(values) => values,
            // Retargeting twice keeps the one shared base; only the overlay
            // is replaced.
            Self::Retargeted { base, .. } => base,
        };
        if targets.len() != base.len() {
            return Err(PolicyError::Unsupported {
                what: "a retarget overlay whose length differs from its group",
            });
        }
        Ok(Self::Retargeted {
            base,
            muls: targets.into_boxed_slice(),
        })
    }

    /// The effective `HfMul` of the varblock at `index`, resolving any
    /// retarget overlay.
    fn hf_mul_at(&self, index: usize) -> Result<HfMul> {
        match self {
            Self::Retargeted { muls, .. } => {
                muls.get(index).copied().ok_or(PolicyError::Unsupported {
                    what: "a retarget overlay shorter than its group",
                })
            }
            _ => self
                .get(index)
                .map(|vb| vb.hf_mul)
                .ok_or(PolicyError::Unsupported {
                    what: "a planned varblock index outside its group",
                }),
        }
    }

    /// An overlay-aware borrow of the contiguous range `start..end`, for the
    /// chunked parallel quantizer.
    fn range(&self, start: usize, end: usize) -> Result<PlannedVarblockRange<'_>> {
        let decisions = self.get(start..end).ok_or(PolicyError::Unsupported {
            what: "a planned-varblock range outside its group",
        })?;
        let muls = match self {
            Self::Retargeted { muls, .. } => {
                Some(muls.get(start..end).ok_or(PolicyError::Unsupported {
                    what: "a retarget overlay range outside its group",
                })?)
            }
            _ => None,
        };
        Ok(PlannedVarblockRange { decisions, muls })
    }
}

/// A borrowed contiguous range of planned varblocks whose effective `HfMul`
/// values resolve through a retarget overlay. The decision slice itself is
/// the base storage; read geometry from it and multipliers through
/// [`PlannedVarblockRange::hf_mul`].
struct PlannedVarblockRange<'a> {
    decisions: &'a [VarblockDecision],
    muls: Option<&'a [HfMul]>,
}

impl PlannedVarblockRange<'_> {
    fn len(&self) -> usize {
        self.decisions.len()
    }

    fn hf_mul(&self, index: usize) -> Result<HfMul> {
        match self.muls {
            Some(muls) => muls.get(index).copied().ok_or(PolicyError::Unsupported {
                what: "a retarget overlay index outside its range",
            }),
            None => self
                .decisions
                .get(index)
                .map(|vb| vb.hf_mul)
                .ok_or(PolicyError::Unsupported {
                    what: "a planned varblock index outside its range",
                }),
        }
    }
}

impl Deref for PlannedVarblocks {
    type Target = [VarblockDecision];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(values) => values,
            Self::Shared(values) => values,
            // Base decisions: `hf_mul` reads must go through `hf_mul_at`.
            Self::Retargeted { base, .. } => base,
        }
    }
}

impl<'a> IntoIterator for &'a PlannedVarblocks {
    type Item = &'a VarblockDecision;
    type IntoIter = slice::Iter<'a, VarblockDecision>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Quantizer-independent structure reused by the anchored rate presets.
///
/// The cover and CfL factors follow the policy chosen for the captured plan.
/// Reused probes update every varblock's quantizer-dependent `HfMul`; the
/// explicit [`AnchorReuse`] policy decides whether the cover and/or CfL is
/// refreshed. Navigation captures neutral CfL for Fast and configured CfL for
/// Balanced. Anchored-Quality experiments can reuse only the cover and
/// re-estimate CfL at the finalist.
#[derive(Clone)]
pub(crate) struct StructuralAnchor {
    groups: Vec<PlannedGroup>,
    cfl: Arc<CflEstimate>,
}

/// Which parts of a captured structural plan a later probe may reuse.
///
/// The policy is explicit because a globally frozen cover and CfL is a
/// quality trade, not an implementation detail. `CoverOnly` is the bounded
/// middle ground for a fresh-CfL finalist; `CoverAndCfl` is reserved for the
/// current Balanced/fast anchored paths. Quality's exhaustive path always
/// uses `None`.
#[derive(Clone, Copy)]
pub(crate) enum AnchorReuse<'a> {
    /// Build both cover and CfL for this probe.
    None,
    /// Reuse the cover, but estimate fresh CfL factors for this quantizer.
    #[allow(
        dead_code,
        reason = "Anchored Quality will use cover-only reuse after its fresh-CfL gate is measured"
    )]
    CoverOnly(&'a StructuralAnchor),
    /// Reuse both cover and CfL from the captured probe.
    #[cfg_attr(
        not(feature = "anchor-sketch"),
        allow(
            dead_code,
            reason = "the two-anchor controller is disabled with anchor-sketch"
        )
    )]
    CoverAndCfl(&'a StructuralAnchor),
}

fn retarget_anchor_groups(groups: &mut [PlannedGroup], aq: &AqSetup) -> Result<()> {
    for (_, _, rect, varblocks) in groups {
        // Phase 24: compute the probe's desired multipliers first, then layer
        // them as a compact overlay over the shared decisions. Only a group
        // whose multipliers actually moved pays for the overlay; the rest
        // keep their shared storage untouched.
        let mut targets = Vec::with_capacity(varblocks.len());
        let mut moved = false;
        for index in 0..varblocks.len() {
            let varblock = varblocks.get(index).ok_or(PolicyError::Unsupported {
                what: "an anchored varblock index outside its group",
            })?;
            let (rows, cols) = varblock.transform.block_dims();
            let rows = u32::try_from(rows).map_err(|_| PolicyError::Unsupported {
                what: "an anchored transform height outside u32",
            })?;
            let cols = u32::try_from(cols).map_err(|_| PolicyError::Unsupported {
                what: "an anchored transform width outside u32",
            })?;
            let hf_mul = aq.mul_for_footprint(
                rect.x0 / 8 + varblock.origin.bx(),
                rect.y0 / 8 + varblock.origin.by(),
                rows,
                cols,
            );
            moved |= hf_mul != varblocks.hf_mul_at(index)?;
            targets.push(hf_mul);
        }
        if moved {
            let prior = std::mem::replace(varblocks, PlannedVarblocks::owned(Vec::new()));
            *varblocks = PlannedVarblocks::retargeted(prior, targets)?;
        }
    }
    Ok(())
}

/// One coefficient sample used by the integer refinement.
#[derive(Debug, Clone, Copy)]
struct CflSample {
    source: f32,
    reconstructed_y: f32,
    cell: usize,
}

#[derive(Debug, Clone, Copy)]
struct RegressionSample {
    source: f32,
    y: f32,
}

/// Regression sums plus the exact samples needed to score neighbouring wire
/// factors through the quantizer's decoder-side arithmetic.
#[derive(Debug, Default)]
struct CflSamples {
    regression: CflAccumulator,
    samples: Vec<CflSample>,
}

impl CflSamples {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            regression: CflAccumulator::default(),
            samples: Vec::with_capacity(capacity),
        }
    }

    fn push(&mut self, source: f32, regression_y: f32, reconstructed_y: f32, cell: usize) {
        self.regression.add(regression_y, source);
        self.samples.push(CflSample {
            source,
            reconstructed_y,
            cell,
        });
    }

    /// Appends a group-local sample stream in its original order.
    ///
    /// Replaying the scalar additions, rather than adding already-rounded
    /// group sums, keeps the pre-parallel floating-point decision bit-exact.
    fn append_ordered(&mut self, other: Self, regression: Vec<RegressionSample>) {
        diagnostics::note_cfl_samples(u64::try_from(other.samples.len()).unwrap_or(u64::MAX));
        self.samples.reserve(other.samples.len());
        for sample in regression {
            self.regression.add(sample.y, sample.source);
        }
        self.samples.extend(other.samples);
    }
}

/// The searchable HF factor samples of one LF group's 64x64 tiles.
struct HfCflSamples {
    tiles: jpxl_encode::vardct::BlockGrid,
    x: Vec<CflSamples>,
    b: Vec<CflSamples>,
}

struct GroupCflSamples {
    lf_x: CflSamples,
    lf_b: CflSamples,
    lf_regression_x: Vec<RegressionSample>,
    lf_regression_b: Vec<RegressionSample>,
    hf: HfCflSamples,
}

/// The sample workspace for one *band* of an LF group: the varblocks
/// (`decisions`) whose origins lie in tile row `tile_row` of a grid
/// `tiles.width` tiles wide. HF sample vectors cover that row's tiles only,
/// indexed by tile column.
fn band_cfl_workspace(
    tiles: jpxl_encode::vardct::BlockGrid,
    tile_row: u32,
    decisions: &[VarblockDecision],
) -> Result<GroupCflSamples> {
    let tile_count = usize::try_from(tiles.width).unwrap_or(0);
    let mut hf_capacities = vec![0usize; tile_count];
    let mut lf_capacity = 0usize;
    for vb in decisions {
        let n = vb.transform.block_dims().0;
        let cells = vb
            .transform
            .sample_cols()
            .checked_mul(vb.transform.sample_rows())
            .ok_or(PolicyError::Unsupported {
                what: "a CfL workspace coefficient count overflow",
            })?;
        let lf_cells = n.checked_mul(n).ok_or(PolicyError::Unsupported {
            what: "a CfL workspace LF count overflow",
        })?;
        lf_capacity = lf_capacity
            .checked_add(lf_cells)
            .ok_or(PolicyError::Unsupported {
                what: "a CfL workspace LF capacity overflow",
            })?;
        if vb.origin.by() / 8 != tile_row {
            return Err(PolicyError::Unsupported {
                what: "a varblock outside its CfL band",
            });
        }
        let tile = usize::try_from(vb.origin.bx() / 8).unwrap_or(usize::MAX);
        let capacity = hf_capacities
            .get_mut(tile)
            .ok_or(PolicyError::Unsupported {
                what: "a varblock outside its CfL tile grid",
            })?;
        *capacity = capacity.checked_add(cells.saturating_sub(lf_cells)).ok_or(
            PolicyError::Unsupported {
                what: "a CfL workspace HF capacity overflow",
            },
        )?;
    }
    Ok(GroupCflSamples {
        lf_x: CflSamples::with_capacity(lf_capacity),
        lf_b: CflSamples::with_capacity(lf_capacity),
        lf_regression_x: Vec::with_capacity(lf_capacity),
        lf_regression_b: Vec::with_capacity(lf_capacity),
        hf: HfCflSamples {
            tiles,
            x: hf_capacities
                .iter()
                .copied()
                .map(CflSamples::with_capacity)
                .collect(),
            b: hf_capacities
                .into_iter()
                .map(CflSamples::with_capacity)
                .collect(),
        },
    })
}

/// One unit of CfL sample collection: the varblocks of LF group `group`
/// whose origins fall in tile row `tile_row` (a 64-pixel band). Bands are
/// independent -- a varblock is at most 32 pixels on a side and 8-aligned, so
/// it never crosses a 64-pixel tile boundary -- and, because an LF group's
/// varblocks are in raster order, concatenating a group's bands in row order
/// reproduces the group's original varblock order exactly.
#[derive(Debug, Clone, Copy)]
struct CflBand {
    group: usize,
    tile_row: u32,
    /// The band's varblocks as a range into the group's decision list.
    first: usize,
    end: usize,
}

/// Milestone 6's square transform vocabulary: DCT8x8, DCT16x16, DCT32x32.
///
/// Rectangles and the special 8x8-footprint transforms are later milestones;
/// squares alone are unambiguous in the LLF-to-LF mapping (block_rows equals
/// block_cols, so no landscape/portrait orientation choice) and already earn
/// the exit gate's win on smooth content.
const SQUARE_TRANSFORMS: [TransformType; 3] = [
    TransformType::Dct8x8,
    TransformType::Dct16x16,
    TransformType::Dct32x32,
];

/// The transform vocabulary needed by Fast's fixed-DCT8 cover.
const FAST_TRANSFORMS: [TransformType; 1] = [TransformType::Dct8x8];

/// The square transform whose footprint is `n` atoms per side, or `None`.
const fn square_transform(n: u32) -> Option<TransformType> {
    match n {
        1 => Some(TransformType::Dct8x8),
        2 => Some(TransformType::Dct16x16),
        4 => Some(TransformType::Dct32x32),
        _ => None,
    }
}

/// §7.2's factorization of the request's quantizer into what the wire
/// carries once an adaptive-quantization field is in play.
///
/// The wire's per-varblock knob, `HfMul >= 1`, divides the step just like
/// `global_scale` does (I.2.1: `Mul = (1 << 16) / (global_scale * HfMul)`),
/// so from a baseline of 1 a varblock can only be *refined*, never coarsened.
/// Bidirectional adjustment is representable exactly anyway: **halve**
/// `global_scale`, **double** `quant_lf` — the LF step divides by
/// `global_scale * quant_lf`, so the product and every LF integer are
/// unchanged — and **double** the baseline `HfMul`, so the baseline HF
/// denominator `global_scale * HfMul` is unchanged too (both are exact
/// integer products in f64). A mul one below the doubled baseline is now an
/// octave coarser, one above an octave finer. When `global_scale` is odd or
/// the doubled `quant_lf`/baseline is not representable, the field falls
/// back to refine-only: adjustments capped at the baseline. A field that
/// turns out neutral (a flat frame) collapses to the plain wire — the
/// factorization's non-zero `mul` row costs real bytes under the gradient
/// predictor (the DctSelect row above it is zeros), and a neutral field
/// would buy nothing for them.
struct AqSetup {
    field: Option<DesiredQuantField>,
    refine_only: bool,
    /// What I.2.1 signals.
    global_scale: GlobalScale,
    /// What I.2.1 signals.
    quant_lf: QuantLf,
    /// The wire `HfMul` of a varblock the field leaves alone.
    baseline: HfMul,
}

impl AqSetup {
    fn build(atlas: &AnalysisAtlas, request: &EncodeRequest, quantizer: QuantizerChoice) -> Self {
        let off = Self {
            field: None,
            refine_only: false,
            global_scale: quantizer.global_scale,
            quant_lf: quantizer.quant_lf,
            baseline: quantizer.hf_mul,
        };
        let field = match DesiredQuantField::from_atlas_tuned(
            atlas,
            request.budget.aq_mode,
            request.budget.aq_tuning,
        ) {
            Some(field) if !field.is_neutral() => field,
            _ => return off,
        };

        // The legacy lattice factors by 2; the fine lattice (Phase Q3) by
        // `FINE_BASELINE`. A `global_scale` that is not a multiple of the
        // factor is snapped down to the nearest multiple (an off-by-a-few the
        // rate loop prices exactly) instead of falling back to refine-only: a
        // fallback keyed on divisibility would make adjacent rate rungs
        // alternate between two differently-sized encoders and put a
        // systematic sawtooth in the ladder.
        let factor = if field.is_fine() {
            crate::field::FINE_BASELINE
        } else {
            2
        };
        let snapped = quantizer.global_scale.get() - quantizer.global_scale.get() % factor;
        let factored = (
            snapped >= factor,
            GlobalScale::new((snapped / factor).max(1)),
            QuantLf::new(quantizer.quant_lf.get().saturating_mul(factor)),
            HfMul::new(quantizer.hf_mul.get().saturating_mul(factor)),
        );
        if let (true, Ok(global_scale), Ok(quant_lf), Ok(baseline)) = factored {
            Self {
                field: Some(field),
                refine_only: false,
                global_scale,
                quant_lf,
                baseline,
            }
        } else {
            Self {
                field: Some(field),
                refine_only: true,
                ..off
            }
        }
    }

    /// The `HfMul` of a varblock footprint (frame-global atom coordinates).
    fn mul_for_footprint(&self, bx: u32, by: u32, rows: u32, cols: u32) -> HfMul {
        self.field.as_ref().map_or(self.baseline, |field| {
            field.mul_for_footprint(bx, by, rows, cols, self.baseline, self.refine_only)
        })
    }

    /// Every `HfMul` this setup can assign.
    fn muls(&self) -> Vec<HfMul> {
        match &self.field {
            Some(field) => mul_lattice_for(self.baseline, self.refine_only, field.is_fine()),
            None => vec![self.baseline],
        }
    }

    /// The `metadata_bits` a varblock's `HfMul` pays in the cover objective.
    ///
    /// The legacy lattice charges [`mul_signal_bits`]. The fine field varies
    /// almost everywhere, so a per-varblock 8-bit charge would only bias the
    /// cover toward merging; its `mul` row is a smooth, small-residual plane
    /// under the LF-group entropy coder, charged at a flat estimate instead.
    fn mul_signal_bits(&self, hf_mul: HfMul) -> f64 {
        match &self.field {
            Some(field) if field.is_fine() => FINE_MUL_SIGNAL_BITS,
            _ => mul_signal_bits(hf_mul, self.baseline),
        }
    }
}

/// The per-transform HF quantizers a frame needs, built once.
///
/// I.5.3's `Mul` and the dequantization matrices depend only on the transform,
/// `global_scale` and `HfMul`; with a constant `HfMul` (adaptive quant is
/// milestone 7) there is one quantizer per transform for the whole frame.
struct HfQuantizers {
    by_key: Vec<((TransformType, u32), HfQuantizer)>,
    baseline: HfMul,
    /// §4.3's Lagrange multiplier per channel, in bits per unit of squared
    /// **sample-domain** error, from the DCT8x8 operating point: a uniform
    /// quantizer at step `s` trades one bit for `s^2/16` of squared error
    /// (halving the step costs one bit per coefficient and moves `s^2/12` to
    /// `s^2/48`), so `lambda = 16 / mean(s^2)` over the non-LLF cells.
    ///
    /// The unit matters because the forward transforms are not Parseval: one
    /// unit of squared coefficient error is `side^2` units of sample error
    /// (measured, exactly, in `probe_parseval`-style tests), so distortion
    /// must be brought into the common sample domain before transforms of
    /// different sizes can be compared. The calibration therefore uses the
    /// sample-domain step `8 * s`, and [`block_cost`] scales each candidate's
    /// coefficient error by its own `side^2`.
    lambda: [f64; NUM_CHANNELS],
    /// Phase 7.1: whether the per-varblock trailing-truncation pass runs.
    /// Carried here rather than threaded through `quantize_group` because the
    /// quantizer set is already the thing that knows the operating point.
    truncate_trailing: bool,
    /// Phase 6.2's per-transform distortion scale, applied on top of
    /// `side^2` in [`block_cost_bounded`]. `Neutral` is exactly `1.0`, so the
    /// shipped objective is bit-identical.
    size_penalty: CoverSizePenalty,
    /// Phase 6.3's per-cell frequency weight, one table per square transform,
    /// keyed by coefficient edge. Empty under `Flat`, which is what keeps the
    /// shipped objective bit-identical.
    frequency_weights: Vec<(usize, Vec<f32>)>,
    /// Phase Q4's rate model: per-size scale on the residual-bit proxy and
    /// per-varblock fixed bits. `Legacy` is exactly the shipped constants.
    rate_model: CoverRateModel,
}

impl HfQuantizers {
    /// Builds every `(transform, HfMul)` quantizer a frame can ask for: the
    /// square vocabulary crossed with `muls` (the adaptive-quantization
    /// lattice around `baseline`, or just `[baseline]` with the field off).
    #[cfg(test)]
    fn new(global_scale: u32, baseline: HfMul, muls: &[HfMul]) -> Result<Self> {
        Self::new_with_scales(
            global_scale,
            baseline,
            muls,
            NEUTRAL_QM_SCALE,
            NEUTRAL_QM_SCALE,
            CoverSizePenalty::Neutral,
            CoverFrequencyWeight::Flat,
            QuantizerChoiceMode::Nearest,
            1.0,
        )
    }

    #[cfg(test)]
    #[allow(
        clippy::too_many_arguments,
        reason = "test helper mirroring the full constructor surface under test"
    )]
    fn new_with_scales(
        global_scale: u32,
        baseline: HfMul,
        muls: &[HfMul],
        x_qm_scale: u32,
        b_qm_scale: u32,
        size_penalty: CoverSizePenalty,
        frequency_weight: CoverFrequencyWeight,
        quantizer_choice: QuantizerChoiceMode,
        lambda_scale: f32,
    ) -> Result<Self> {
        Self::new_with_scales_for_transforms(
            global_scale,
            baseline,
            muls,
            x_qm_scale,
            b_qm_scale,
            size_penalty,
            frequency_weight,
            quantizer_choice,
            lambda_scale,
            &SQUARE_TRANSFORMS,
        )
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "the transform vocabulary is an explicit bounded construction input"
    )]
    fn new_with_scales_for_transforms(
        global_scale: u32,
        baseline: HfMul,
        muls: &[HfMul],
        x_qm_scale: u32,
        b_qm_scale: u32,
        size_penalty: CoverSizePenalty,
        frequency_weight: CoverFrequencyWeight,
        quantizer_choice: QuantizerChoiceMode,
        lambda_scale: f32,
        transforms: &[TransformType],
    ) -> Result<Self> {
        let defaults = DequantMatrices::all_default().map_err(|_| PolicyError::Unsupported {
            what: "the I.2.5 default dequantization matrices",
        })?;
        let mut by_key = Vec::with_capacity(transforms.len() * muls.len());
        for &transform in transforms {
            let matrices: [DequantMatrix; NUM_CHANNELS] = [
                defaults
                    .for_transform(transform, 0)
                    .map_err(|_| PolicyError::Unsupported {
                        what: "a dequantization matrix for this transform",
                    })?,
                defaults
                    .for_transform(transform, 1)
                    .map_err(|_| PolicyError::Unsupported {
                        what: "a dequantization matrix for this transform",
                    })?,
                defaults
                    .for_transform(transform, 2)
                    .map_err(|_| PolicyError::Unsupported {
                        what: "a dequantization matrix for this transform",
                    })?,
            ];
            for &mul in muls {
                by_key.push((
                    (transform, mul.get()),
                    HfQuantizer::new_with_matrices(
                        transform,
                        global_scale,
                        mul.get(),
                        x_qm_scale,
                        b_qm_scale,
                        &matrices,
                    )?,
                ));
            }
        }
        let dct8 = &by_key
            .iter()
            .find(|((t, m), _)| *t == TransformType::Dct8x8 && *m == baseline.get())
            .ok_or(PolicyError::Unsupported {
                what: "a baseline DCT8x8 quantizer",
            })?
            .1;
        // Phase 7.2: a non-positive or non-finite research flag must not zero
        // the objective; fall back to the calibrated unit scale.
        let scale = if lambda_scale.is_finite() && lambda_scale > 0.0 {
            f64::from(lambda_scale)
        } else {
            1.0
        };
        let mut lambda = [0.0f64; NUM_CHANNELS];
        for (channel, slot) in lambda.iter_mut().enumerate() {
            let mut sum = 0.0f64;
            let mut count = 0u32;
            for cell in 0..DCT8X8_CELLS {
                if cell == 0 {
                    continue;
                }
                let step = 8.0 * f64::from(dct8.step(channel, cell));
                sum += step * step;
                count += 1;
            }
            let mean = sum / f64::from(count.max(1));
            *slot = if mean > 0.0 { scale * 16.0 / mean } else { 0.0 };
        }
        let frequency_weights = match frequency_weight {
            CoverFrequencyWeight::Flat => Vec::new(),
            CoverFrequencyWeight::Csf => SQUARE_TRANSFORMS
                .iter()
                .map(|t| {
                    let side = t.coeff_cols();
                    (side, crate::csf::square_weights(side, t.block_dims().0))
                })
                .collect(),
            CoverFrequencyWeight::QuantDonor => SQUARE_TRANSFORMS
                .iter()
                .map(|t| {
                    let side = t.coeff_cols();
                    (
                        side,
                        crate::csf::quant_donor_weights(side, t.block_dims().0),
                    )
                })
                .collect(),
        };
        let mut built = Self {
            by_key,
            baseline,
            truncate_trailing: quantizer_choice == QuantizerChoiceMode::TrailingTruncation,
            lambda,
            size_penalty,
            frequency_weights,
            rate_model: CoverRateModel::Legacy,
        };

        // Phase 7.0: hand each quantizer the *same* Lagrange weight
        // `block_cost_bounded` applies to that cell, so the quantizer's local
        // choice minimises the same objective the cover search sums. Doing it
        // here rather than in `HfQuantizer::new` is forced by ordering:
        // `lambda` is calibrated from the finished DCT8x8 baseline quantizer,
        // so it does not exist until every quantizer is built.
        // Install the Lagrange weights whenever *either* Phase 7.0's choose
        // rule or Phase 7.1's truncation pass needs them; `rd_choose` decides
        // which of the two actually reads them.
        if quantizer_choice != QuantizerChoiceMode::Nearest {
            let lambda = built.lambda;
            let penalty = built.size_penalty;
            let weights: Vec<(usize, Vec<f32>)> = built.frequency_weights.clone();
            for ((transform, _), quant) in &mut built.by_key {
                let side = transform.coeff_cols();
                #[allow(
                    clippy::cast_precision_loss,
                    reason = "side is at most 32; exact in f64"
                )]
                let to_sample_domain = (side * side) as f64 * penalty.multiplier(side);
                let freq = weights
                    .iter()
                    .find(|(s, _)| *s == side)
                    .map(|(_, w)| w.clone())
                    .unwrap_or_default();
                let rd_choose = quantizer_choice == QuantizerChoiceMode::RateDistortion;
                quant.install_rd_weights(rd_choose, |channel, cell| {
                    let w = freq.get(cell).copied().map_or(1.0, f64::from);
                    lambda.get(channel).copied().unwrap_or(0.0) * to_sample_domain * w
                });
            }
        }
        Ok(built)
    }

    /// Phase 6.3's per-cell frequency weight table for one transform, or an
    /// empty slice under the flat production policy — which the scorer reads
    /// as "charge every cell alike", keeping the shipped objective untouched.
    fn frequency_weights(&self, transform: TransformType) -> &[f32] {
        let side = transform.coeff_cols();
        self.frequency_weights
            .iter()
            .find(|(s, _)| *s == side)
            .map_or(&[][..], |(_, w)| w.as_slice())
    }

    /// Phase 6.2's distortion multiplier for one transform, keyed by its
    /// coefficient edge. Exactly `1.0` under the neutral production policy.
    fn size_penalty(&self, transform: TransformType) -> f64 {
        self.size_penalty.multiplier(transform.coeff_cols())
    }

    /// Installs Phase Q4's cover rate model (`Legacy` leaves the objective
    /// bit-identical).
    fn with_rate_model(mut self, model: CoverRateModel) -> Self {
        self.rate_model = model;
        self
    }

    /// The rate model's multiplier on a candidate's residual-bit sum.
    fn rate_scale(&self, transform: TransformType) -> f64 {
        self.rate_model.scale(transform.coeff_cols())
    }

    /// The rate model's fixed bits for one candidate varblock.
    fn rate_fixed_bits(&self, transform: TransformType) -> f64 {
        self.rate_model.fixed_bits(transform.coeff_cols())
    }

    /// Widens (or narrows) every quantizer's zero threshold by `scale`
    /// (Quality-track research control, see
    /// [`EncodeRequest::dead_zone_scale`]). `1.0`, non-finite and non-positive
    /// values leave the tables exactly as built.
    fn with_dead_zone_scale(mut self, scale: f32) -> Self {
        if scale.is_finite() && scale > 0.0 && scale != 1.0 {
            for (_, quant) in &mut self.by_key {
                quant.scale_zero_threshold(scale);
            }
        }
        self
    }

    /// Sets what the trailing-truncation pass charges per freed interior zero
    /// token (Quality-track research control, see
    /// [`EncodeRequest::zero_token_bits`]). Non-finite or negative values keep
    /// the built default.
    fn with_zero_token_bits(mut self, bits: f32) -> Self {
        if bits.is_finite() && bits >= 0.0 {
            for (_, quant) in &mut self.by_key {
                quant.set_zero_token_bits(bits);
            }
        }
        self
    }

    fn get(&self, transform: TransformType, mul: HfMul) -> Result<&HfQuantizer> {
        self.by_key
            .iter()
            .find(|((t, m), _)| *t == transform && *m == mul.get())
            .map(|(_, q)| q)
            .ok_or(PolicyError::Unsupported {
                what: "an HF quantizer outside the built (transform, HfMul) set",
            })
    }

    /// The baseline-`HfMul` quantizer for a transform: the frame's nominal
    /// operating point.
    fn baseline(&self, transform: TransformType) -> Result<&HfQuantizer> {
        self.get(transform, self.baseline)
    }
}

/// Inverse-Gaborish precondition of the source XYB planes (default J.1 weights).
///
/// Analysis stays on the caller's `frame`; this clone is only for DCT,
/// cover search, CfL, and quantization.
pub(crate) fn prepare_gaborish_frame(frame: &PreparedFrame) -> Result<PreparedFrame> {
    let width = usize::try_from(frame.width()).unwrap_or(0);
    let height = usize::try_from(frame.height()).unwrap_or(0);
    // API order is (Y, X, B); PreparedFrame stores X, Y, B.
    let (y, x, b) = jpxl_encode::vardct::gaborish::precondition_xyb_planes(
        frame.xyb().y.samples(),
        frame.xyb().x.samples(),
        frame.xyb().b.samples(),
        width,
        height,
    );
    PreparedFrame::from_xyb(frame.width(), frame.height(), x, y, b, frame.is_grayscale())
}

/// Reusable per-varblock forward-transform buffers, sized for the largest
/// square transform (DCT32x32).
struct ForwardScratch {
    transform: TransformScratch,
    samples: Vec<f32>,
    coeffs: [Vec<f32>; NUM_CHANNELS],
}

impl ForwardScratch {
    fn new() -> Self {
        let max = 32 * 32;
        Self {
            transform: TransformScratch::for_transform(TransformType::Dct32x32),
            samples: vec![0.0; max],
            coeffs: core::array::from_fn(|_| vec![0.0; max]),
        }
    }

    /// A zero-capacity marker for cache paths that cannot perform a forward
    /// transform. `CoverForwardBank::Complete` only borrows already cached
    /// coefficients, so allocating the lazy-path buffers would be wasted.
    fn empty() -> Self {
        Self {
            transform: TransformScratch::with_cells(0),
            samples: Vec::new(),
            coeffs: core::array::from_fn(|_| Vec::new()),
        }
    }
}

/// Borrowed forward coefficients for one varblock.
///
/// Channel order matches [`gather_square`]: 0 = X, 1 = Y, 2 = B. The slices
/// point into one LF-group/transform arena rather than three candidate-owned
/// vectors.
#[derive(Clone, Copy)]
struct VarblockForward<'a> {
    coeffs: [&'a [f32]; NUM_CHANNELS],
}

const EMPTY_FORWARD_SLOT: usize = usize::MAX;

/// Dense origin table plus one coefficient arena for one transform family.
struct DenseForwardBank {
    step_blocks: u32,
    slots_w: u32,
    cells: usize,
    offsets: Vec<usize>,
    coefficients: Vec<f32>,
    entries: u64,
    hits: u64,
    misses: u64,
}

impl DenseForwardBank {
    fn new(blocks: jpxl_encode::vardct::BlockGrid, transform: TransformType) -> Result<Self> {
        let step_blocks =
            u32::try_from(transform.block_dims().0).map_err(|_| PolicyError::Unsupported {
                what: "a transform block edge outside u32",
            })?;
        let slots_w = blocks.width.div_ceil(step_blocks);
        let slots_h = blocks.height.div_ceil(step_blocks);
        let slots = usize::try_from(u64::from(slots_w) * u64::from(slots_h)).map_err(|_| {
            PolicyError::Unsupported {
                what: "a forward-bank slot count outside usize",
            }
        })?;
        let cells = transform
            .sample_cols()
            .checked_mul(transform.sample_rows())
            .ok_or(PolicyError::Unsupported {
                what: "a forward-bank coefficient count overflow",
            })?;
        let capacity = slots
            .checked_mul(NUM_CHANNELS)
            .and_then(|n| n.checked_mul(cells))
            .ok_or(PolicyError::Unsupported {
                what: "a forward-bank arena capacity overflow",
            })?;
        Ok(Self {
            step_blocks,
            slots_w,
            cells,
            offsets: vec![EMPTY_FORWARD_SLOT; slots],
            coefficients: Vec::with_capacity(capacity),
            entries: 0,
            hits: 0,
            misses: 0,
        })
    }

    fn slot(&self, bx: u32, by: u32) -> Result<usize> {
        if !bx.is_multiple_of(self.step_blocks) || !by.is_multiple_of(self.step_blocks) {
            return Err(PolicyError::Unsupported {
                what: "an unaligned square-transform forward-cache origin",
            });
        }
        let sx = bx / self.step_blocks;
        let sy = by / self.step_blocks;
        let slot = usize::try_from(u64::from(sy) * u64::from(self.slots_w) + u64::from(sx))
            .unwrap_or(usize::MAX);
        if slot >= self.offsets.len() {
            return Err(PolicyError::Unsupported {
                what: "a forward-cache origin outside its LF-group bank",
            });
        }
        Ok(slot)
    }

    fn view(&self, offset: usize) -> Result<VarblockForward<'_>> {
        let x_end = offset
            .checked_add(self.cells)
            .ok_or(PolicyError::Unsupported {
                what: "a forward-bank channel range overflow",
            })?;
        let y_end = x_end
            .checked_add(self.cells)
            .ok_or(PolicyError::Unsupported {
                what: "a forward-bank channel range overflow",
            })?;
        let b_end = y_end
            .checked_add(self.cells)
            .ok_or(PolicyError::Unsupported {
                what: "a forward-bank channel range overflow",
            })?;
        Ok(VarblockForward {
            coeffs: [
                self.coefficients
                    .get(offset..x_end)
                    .ok_or(PolicyError::Unsupported {
                        what: "a missing X lane in a forward bank",
                    })?,
                self.coefficients
                    .get(x_end..y_end)
                    .ok_or(PolicyError::Unsupported {
                        what: "a missing Y lane in a forward bank",
                    })?,
                self.coefficients
                    .get(y_end..b_end)
                    .ok_or(PolicyError::Unsupported {
                        what: "a missing B lane in a forward bank",
                    })?,
            ],
        })
    }
}

/// Dense coefficient banks owned by one LF group.
struct CandidateGroupBank {
    rect: jpxl_encode::vardct::Rect,
    blocks: jpxl_encode::vardct::BlockGrid,
    banks: [Option<DenseForwardBank>; 3],
    cover_complete: bool,
    complete_hits: std::sync::atomic::AtomicU64,
}

impl CandidateGroupBank {
    fn new(
        rect: jpxl_encode::vardct::Rect,
        blocks: jpxl_encode::vardct::BlockGrid,
        families: &[TransformType],
    ) -> Result<Self> {
        // Reserve the families this search will score on the request thread
        // before planning fans out. That keeps their allocator ownership
        // stable across warm-up and timed rate probes instead of stranding an
        // arena in whichever worker first encountered a transform. Families
        // that this cover never scores stay `None`; `get_or_insert` can still
        // create one later if a rescue rebuilds structure with a wider set.
        let mut banks = [None, None, None];
        for &transform in families {
            let Some(index) = Self::family(transform) else {
                continue;
            };
            let Some(slot) = banks.get_mut(index) else {
                continue;
            };
            if slot.is_none() {
                *slot = Some(DenseForwardBank::new(blocks, transform)?);
            }
        }
        Ok(Self {
            rect,
            blocks,
            banks,
            cover_complete: false,
            complete_hits: std::sync::atomic::AtomicU64::new(0),
        })
    }

    const fn family(transform: TransformType) -> Option<usize> {
        match transform {
            TransformType::Dct8x8 => Some(0),
            TransformType::Dct16x16 => Some(1),
            TransformType::Dct32x32 => Some(2),
            _ => None,
        }
    }

    fn local_block(&self, px: u32, py: u32) -> Result<(u32, u32)> {
        let dx = px
            .checked_sub(self.rect.x0)
            .ok_or(PolicyError::Unsupported {
                what: "a forward-cache X origin before its LF group",
            })?;
        let dy = py
            .checked_sub(self.rect.y0)
            .ok_or(PolicyError::Unsupported {
                what: "a forward-cache Y origin before its LF group",
            })?;
        if !dx.is_multiple_of(8) || !dy.is_multiple_of(8) {
            return Err(PolicyError::Unsupported {
                what: "a forward-cache origin outside the 8x8 block grid",
            });
        }
        Ok((dx / 8, dy / 8))
    }

    fn get_or_insert(
        &mut self,
        frame: &PreparedFrame,
        transform: TransformType,
        px: u32,
        py: u32,
        scratch: &mut ForwardScratch,
    ) -> Result<VarblockForward<'_>> {
        let family = Self::family(transform).ok_or(PolicyError::Unsupported {
            what: "a non-square transform in the dense forward cache",
        })?;
        let (bx, by) = self.local_block(px, py)?;
        if self.banks.get(family).and_then(Option::as_ref).is_none() {
            let bank = DenseForwardBank::new(self.blocks, transform)?;
            if let Some(slot) = self.banks.get_mut(family) {
                *slot = Some(bank);
            }
        }
        let bank = self.banks.get_mut(family).and_then(Option::as_mut).ok_or(
            PolicyError::Unsupported {
                what: "a missing dense forward bank after creation",
            },
        )?;
        let slot = bank.slot(bx, by)?;
        let offset = bank
            .offsets
            .get(slot)
            .copied()
            .unwrap_or(EMPTY_FORWARD_SLOT);
        if offset != EMPTY_FORWARD_SLOT {
            bank.hits = bank.hits.saturating_add(1);
            return bank.view(offset);
        }

        let side = forward_square(frame, transform, px, py, scratch)?;
        let cells = side * side;
        if cells != bank.cells {
            return Err(PolicyError::Unsupported {
                what: "a forward transform whose size disagrees with its bank",
            });
        }
        let offset = bank.coefficients.len();
        for channel in 0..NUM_CHANNELS {
            let lane = scratch
                .coeffs
                .get(channel)
                .and_then(|coefficients| coefficients.get(..cells))
                .ok_or(PolicyError::Unsupported {
                    what: "a missing forward-transform scratch lane",
                })?;
            bank.coefficients.extend_from_slice(lane);
        }
        if let Some(stored) = bank.offsets.get_mut(slot) {
            *stored = offset;
        }
        bank.entries = bank.entries.saturating_add(1);
        bank.misses = bank.misses.saturating_add(1);
        diagnostics::note_candidate_forward(cells.saturating_mul(NUM_CHANNELS));
        bank.view(offset)
    }

    fn get(&self, transform: TransformType, px: u32, py: u32) -> Option<VarblockForward<'_>> {
        let family = Self::family(transform)?;
        let (bx, by) = self.local_block(px, py).ok()?;
        let bank = self.banks.get(family)?.as_ref()?;
        let slot = bank.slot(bx, by).ok()?;
        let offset = bank.offsets.get(slot).copied()?;
        (offset != EMPTY_FORWARD_SLOT)
            .then(|| bank.view(offset).ok())
            .flatten()
    }
}

/// Cross-probe / within-probe cache of quantizer-independent forward DCTs.
///
/// Built lazily: cover search and post-cover CfL/quantize all hit the same
/// group-local banks. The rate loop keeps one cache across quantizer probes so
/// a given (origin, transform) is transformed at most once per request.
#[derive(Default)]
pub(crate) struct CandidateForwardCache {
    groups: Vec<std::sync::RwLock<CandidateGroupBank>>,
}

impl CandidateForwardCache {
    fn new() -> Self {
        Self::default()
    }

    /// Hit count for Opt-V2 rate-loop telemetry.
    pub(crate) fn hits(&self) -> u64 {
        let mutable_hits = self.bank_stat(|bank| bank.hits);
        let immutable_hits = self
            .groups
            .iter()
            .filter_map(|group| group.read().ok())
            .map(|group| {
                group
                    .complete_hits
                    .load(std::sync::atomic::Ordering::Relaxed)
            })
            .fold(0, u64::saturating_add);
        mutable_hits.saturating_add(immutable_hits)
    }

    /// Miss count for Opt-V2 rate-loop telemetry.
    pub(crate) fn misses(&self) -> u64 {
        self.bank_stat(|bank| bank.misses)
    }

    pub(crate) fn entries(&self) -> u64 {
        self.bank_stat(|bank| bank.entries)
    }

    pub(crate) fn payload_bytes(&self) -> u64 {
        self.bank_stat(|bank| {
            u64::try_from(bank.coefficients.len())
                .unwrap_or(u64::MAX)
                .saturating_mul(4)
        })
    }

    pub(crate) fn allocations(&self) -> u64 {
        self.bank_stat(|_| 1)
    }

    fn prepare(&mut self, geometry: &VardctGeometry) -> Result<()> {
        self.prepare_families(geometry, &SQUARE_TRANSFORMS)
    }

    fn prepare_families(
        &mut self,
        geometry: &VardctGeometry,
        families: &[TransformType],
    ) -> Result<()> {
        if !self.groups.is_empty() {
            if self.groups.len() == usize::try_from(geometry.num_lf_groups()).unwrap_or(usize::MAX)
            {
                return Ok(());
            }
            return Err(PolicyError::Unsupported {
                what: "a forward cache reused with different frame geometry",
            });
        }
        let mut groups = Vec::new();
        for index in 0..geometry.num_lf_groups() {
            let id = LfGroupId::new(u32::try_from(index).unwrap_or(u32::MAX));
            let rect = geometry.lf_group_rect(id).ok_or(PolicyError::Unsupported {
                what: "an LF group outside the frame's grid",
            })?;
            let blocks = geometry
                .lf_group_blocks(id)
                .ok_or(PolicyError::Unsupported {
                    what: "an LF group outside the frame's block grid",
                })?;
            groups.push(std::sync::RwLock::new(CandidateGroupBank::new(
                rect, blocks, families,
            )?));
        }
        self.groups = groups;
        Ok(())
    }

    fn group(&self, index: usize) -> Result<&std::sync::RwLock<CandidateGroupBank>> {
        self.groups.get(index).ok_or(PolicyError::Unsupported {
            what: "a missing LF-group forward bank",
        })
    }

    fn bank_stat(&self, value: impl Fn(&DenseForwardBank) -> u64) -> u64 {
        self.groups
            .iter()
            .filter_map(|group| group.read().ok())
            .map(|group| {
                group
                    .banks
                    .iter()
                    .filter_map(Option::as_ref)
                    .map(&value)
                    .fold(0, u64::saturating_add)
            })
            .fold(0, u64::saturating_add)
    }
}

/// Ensures every selected varblock's forward coefficients are cached,
/// computing any that cover selection did not already touch (this is a
/// no-op per varblock under [`CoverMode::Hierarchical`], where scoring
/// candidates already inserted the winner; it does the real work under
/// [`CoverMode::FixedDct8x8`], which never consults the cache).
///
/// Phase-3: this replaces the old `forward_selected`, which cloned a fresh
/// `Vec<VarblockForward>` per group out of the cache (outside-advice.md §7's
/// "selected-forward clone" — 144 MB at 12 MP). Splitting "ensure computed"
/// (mutable) from "gather borrows" ([`gather_forward_refs`], immutable) lets
/// every group's forwards be borrowed directly from its dense bank while that
/// group's lock is held.
fn ensure_forwards_cached(
    frame: &PreparedFrame,
    varblocks: &[VarblockDecision],
    origin: (u32, u32),
    cache: &mut CandidateGroupBank,
    scratch: &mut ForwardScratch,
) -> Result<()> {
    let (x0, y0) = origin;
    for vb in varblocks {
        let px = x0 + vb.origin.bx() * 8;
        let py = y0 + vb.origin.by() * 8;
        cache.get_or_insert(frame, vb.transform, px, py, scratch)?;
    }
    Ok(())
}

/// Completes one LF group's square-transform candidate cache.
///
/// A hierarchical cover is a forest of independent aligned 4x4-atom trees.
/// Once every DCT8x8/DCT16x16/DCT32x32 node is present, later trees can score
/// through immutable cache reads and therefore fan out within an LF group.
fn ensure_cover_candidates_cached(
    frame: &PreparedFrame,
    grid: jpxl_encode::vardct::BlockGrid,
    origin: (u32, u32),
    cache: &mut CandidateGroupBank,
    scratch: &mut ForwardScratch,
) -> Result<()> {
    if cache.cover_complete {
        return Ok(());
    }
    let (x0, y0) = origin;
    for by in 0..grid.height {
        for bx in 0..grid.width {
            let px = x0.saturating_add(bx.saturating_mul(8));
            let py = y0.saturating_add(by.saturating_mul(8));
            cache.get_or_insert(frame, TransformType::Dct8x8, px, py, scratch)?;
            if bx.is_multiple_of(2)
                && by.is_multiple_of(2)
                && bx.saturating_add(2) <= grid.width
                && by.saturating_add(2) <= grid.height
            {
                cache.get_or_insert(frame, TransformType::Dct16x16, px, py, scratch)?;
            }
            if bx.is_multiple_of(4)
                && by.is_multiple_of(4)
                && bx.saturating_add(4) <= grid.width
                && by.saturating_add(4) <= grid.height
            {
                cache.get_or_insert(frame, TransformType::Dct32x32, px, py, scratch)?;
            }
        }
    }
    cache.cover_complete = true;
    Ok(())
}

/// Per-LF-group partial accumulators of the transform summary.
struct TransformPartial {
    histogram: Vec<u64>,
    blocks: u64,
    ac_cells: u64,
    total_y: f64,
    total_low: f64,
    total_high: f64,
    total_row: f64,
    total_col: f64,
    total_xb: f64,
    near_1e3: u64,
    near_1e2: u64,
    dc_sum: f64,
    dc_sumsq: f64,
}

/// Reduces the frame's aligned DCT8x8 candidates into a
/// [`TransformFeatureSummary`](quality_features::TransformFeatureSummary),
/// filling the shared forward cache as it goes (one-shot program PR 4).
///
/// Every coefficient computed here is one the hierarchical cover search
/// would compute anyway — the fill goes through the same
/// [`CandidateGroupBank::get_or_insert`] the cover reads — so the later
/// pixel plan reuses the warm entries rather than re-transforming. LF
/// groups fill in parallel on the request executor exactly like cover
/// construction; each group's partials accumulate in block raster order
/// and combine in LF-group index order, so the result is deterministic
/// across worker counts and SIMD modes.
pub(crate) fn quality_transform_summary(
    transform_frame: &PreparedFrame,
    request: &EncodeRequest,
    cache: &mut CandidateForwardCache,
    executor: Option<&jpxl_encode::EncodeExecutor>,
) -> Result<quality_features::TransformFeatureSummary> {
    const EPS: f64 = 1e-30;
    // Fixed-bin histogram of per-block ln(Y AC energy): [-46, 18) at 0.125.
    const HIST_LO: f64 = -46.0;
    const HIST_WIDTH: f64 = 0.125;
    const HIST_BINS: usize = 512;

    let decision = FrameDecision {
        width: transform_frame.width(),
        height: transform_frame.height(),
        group_size_shift: VARDCT_GROUP_SIZE_SHIFT,
        num_passes: 1,
        bits_per_sample: request.bits_per_sample,
    };
    let geometry = decision.geometry()?;
    cache.prepare(&geometry)?;
    let shared: &CandidateForwardCache = cache;
    let n_groups = usize::try_from(geometry.num_lf_groups()).unwrap_or(usize::MAX);

    let summarize_group = |index: usize| -> Result<TransformPartial> {
        let mut scratch = ForwardScratch::new();
        let mut partial = TransformPartial {
            histogram: vec![0u64; HIST_BINS],
            blocks: 0,
            ac_cells: 0,
            total_y: 0.0,
            total_low: 0.0,
            total_high: 0.0,
            total_row: 0.0,
            total_col: 0.0,
            total_xb: 0.0,
            near_1e3: 0,
            near_1e2: 0,
            dc_sum: 0.0,
            dc_sumsq: 0.0,
        };
        let lock = shared.group(index)?;
        let mut bank = lock.write().map_err(|_| PolicyError::Unsupported {
            what: "a poisoned forward-cache bank",
        })?;
        let (rect, grid) = (bank.rect, bank.blocks);
        for by in 0..grid.height {
            for bx in 0..grid.width {
                let px = rect.x0.saturating_add(bx.saturating_mul(8));
                let py = rect.y0.saturating_add(by.saturating_mul(8));
                let fwd = bank.get_or_insert(
                    transform_frame,
                    TransformType::Dct8x8,
                    px,
                    py,
                    &mut scratch,
                )?;
                let [cx, cy, cb] = fwd.coeffs;
                let mut block_y = 0.0f64;
                for (cell, &value) in cy.iter().enumerate() {
                    if cell == 0 {
                        let dc = f64::from(value);
                        partial.dc_sum += dc;
                        partial.dc_sumsq += dc * dc;
                        continue;
                    }
                    let (row, col) = (cell / 8, cell % 8);
                    let energy = f64::from(value) * f64::from(value);
                    block_y += energy;
                    if row + col <= 2 {
                        partial.total_low += energy;
                    }
                    if row.max(col) >= 4 {
                        partial.total_high += energy;
                    }
                    if row > col {
                        partial.total_row += energy;
                    } else if col > row {
                        partial.total_col += energy;
                    }
                    let magnitude = f64::from(value).abs();
                    if magnitude < 1e-3 {
                        partial.near_1e3 += 1;
                    }
                    if magnitude < 1e-2 {
                        partial.near_1e2 += 1;
                    }
                    partial.ac_cells += 1;
                }
                for lane in [cx, cb] {
                    for &value in lane.iter().skip(1) {
                        partial.total_xb += f64::from(value) * f64::from(value);
                    }
                }
                partial.total_y += block_y;
                partial.blocks += 1;
                let ln_energy = (block_y + EPS).ln();
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "clamped into 0..HIST_BINS before the narrowing"
                )]
                let bin = (((ln_energy - HIST_LO) / HIST_WIDTH).clamp(0.0, (HIST_BINS - 1) as f64))
                    as usize;
                if let Some(count) = partial.histogram.get_mut(bin) {
                    *count += 1;
                }
            }
        }
        Ok(partial)
    };

    let partials: Vec<TransformPartial> = if let Some(executor) = executor {
        executor.map_ordered(n_groups, summarize_group)?
    } else {
        (0..n_groups).map(summarize_group).collect::<Result<_>>()?
    };

    // Fixed-order combination of the per-group partials.
    let mut histogram = vec![0u64; HIST_BINS];
    let mut blocks = 0u64;
    let mut ac_cells = 0u64;
    let (mut total_y, mut total_low, mut total_high) = (0.0f64, 0.0f64, 0.0f64);
    let (mut total_row, mut total_col) = (0.0f64, 0.0f64);
    let mut total_xb = 0.0f64;
    let (mut near_1e3, mut near_1e2) = (0u64, 0u64);
    let (mut dc_sum, mut dc_sumsq) = (0.0f64, 0.0f64);
    for partial in &partials {
        for (total, &count) in histogram.iter_mut().zip(partial.histogram.iter()) {
            *total += count;
        }
        blocks += partial.blocks;
        ac_cells += partial.ac_cells;
        total_y += partial.total_y;
        total_low += partial.total_low;
        total_high += partial.total_high;
        total_row += partial.total_row;
        total_col += partial.total_col;
        total_xb += partial.total_xb;
        near_1e3 += partial.near_1e3;
        near_1e2 += partial.near_1e2;
        dc_sum += partial.dc_sum;
        dc_sumsq += partial.dc_sumsq;
    }

    let quantile = |q: f64| -> f64 {
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "block counts stay far inside f64's exact-integer range"
        )]
        {
            let want = (q * blocks as f64).min(blocks.saturating_sub(1) as f64) as u64;
            let mut seen = 0u64;
            for (bin, &count) in histogram.iter().enumerate() {
                seen += count;
                if seen > want {
                    return HIST_LO + (bin as f64 + 0.5) * HIST_WIDTH;
                }
            }
            HIST_LO + (HIST_BINS as f64 - 0.5) * HIST_WIDTH
        }
    };

    #[allow(
        clippy::cast_precision_loss,
        reason = "block and cell counts stay far inside f64's exact-integer range"
    )]
    Ok(quality_features::TransformFeatureSummary {
        blocks,
        ln_ac_y_mean: (total_y / (blocks as f64).max(1.0) + EPS).ln(),
        ln_ac_y_q50: quantile(0.5),
        ln_ac_y_q90: quantile(0.9),
        ln_ac_y_q99: quantile(0.99),
        high_low_ratio: total_high / (total_low + EPS),
        directional_asymmetry: (total_row - total_col).abs() / (total_row + total_col + EPS),
        chroma_ac_ratio: total_xb / (total_y + EPS),
        near_zero_frac_1e3: near_1e3 as f64 / (ac_cells as f64).max(1.0),
        near_zero_frac_1e2: near_1e2 as f64 / (ac_cells as f64).max(1.0),
        dc_variance_y: {
            let n = (blocks as f64).max(1.0);
            let mean = dc_sum / n;
            (dc_sumsq / n - mean * mean).max(0.0)
        },
    })
}

/// Standalone [`TransformFeatureSummary`](quality_features::TransformFeatureSummary)
/// of a frame, for calibration tooling (`jpxl features --transform-summary`).
///
/// Builds a throwaway forward cache; the in-search path
/// (`QualityBudget::transform_shadow`) shares the cover's cache instead.
/// Applies the request's Gaborish preconditioning so the coefficients are
/// the ones the encode itself would transform.
///
/// # Errors
///
/// Whatever the preconditioner or forward transform refuses.
pub fn transform_feature_summary(
    frame: &PreparedFrame,
    request: &EncodeRequest,
) -> Result<quality_features::TransformFeatureSummary> {
    let transform_owned = if request.restoration.gaborish {
        Some(prepare_gaborish_frame(frame)?)
    } else {
        None
    };
    let transform_frame = transform_owned.as_ref().unwrap_or(frame);
    let mut cache = CandidateForwardCache::new();
    let executor = request.resources.executor();
    quality_transform_summary(transform_frame, request, &mut cache, Some(&executor))
}

/// Borrows the (already-cached) forward coefficients for every selected
/// varblock, in `varblocks` order. See [`ensure_forwards_cached`].
fn gather_forward_refs<'cache>(
    varblocks: &[VarblockDecision],
    origin: (u32, u32),
    cache: &'cache CandidateGroupBank,
) -> Result<Vec<VarblockForward<'cache>>> {
    let (x0, y0) = origin;
    let mut out = Vec::with_capacity(varblocks.len());
    for vb in varblocks {
        let px = x0 + vb.origin.bx() * 8;
        let py = y0 + vb.origin.by() * 8;
        let fwd = cache
            .get(vb.transform, px, py)
            .ok_or(PolicyError::Unsupported {
                what: "a selected varblock's forward missing from the cache",
            })?;
        out.push(fwd);
    }
    Ok(out)
}

/// Copies a `side x side` sample window out of one XYB plane, replicating the
/// frame edge (see the DCT8x8 rationale: a hard zero edge would ring across the
/// whole clipped margin; the replicated edge puts no energy in high
/// frequencies).
fn gather_square(
    frame: &PreparedFrame,
    channel: usize,
    x0: u32,
    y0: u32,
    side: u32,
    out: &mut [f32],
) {
    let (w, h) = (frame.width(), frame.height());
    let plane = match channel {
        0 => &frame.xyb().x,
        1 => &frame.xyb().y,
        _ => &frame.xyb().b,
    };
    // Interior square (the overwhelmingly common case): straight row copies
    // from the resident plane. Same values as the clamped per-sample path.
    if x0.saturating_add(side) <= w && y0.saturating_add(side) <= h {
        let (side_u, w_u) = (side as usize, w as usize);
        let samples = plane.samples();
        let mut ok = true;
        for dy in 0..side_u {
            let row_start = (y0 as usize + dy) * w_u + x0 as usize;
            match (
                samples.get(row_start..row_start + side_u),
                out.get_mut(dy * side_u..(dy + 1) * side_u),
            ) {
                (Some(src), Some(dst)) => dst.copy_from_slice(src),
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            return;
        }
    }
    for dy in 0..side {
        for dx in 0..side {
            let x = (x0 + dx).min(w.saturating_sub(1));
            let y = (y0 + dy).min(h.saturating_sub(1));
            if let Some(slot) = out.get_mut((dy * side + dx) as usize) {
                *slot = plane.at(x, y, w).unwrap_or(0.0);
            }
        }
    }
}

/// Forward-transforms all three channels of one square varblock at pixel
/// `(px, py)` into `scratch.coeffs`, each a `side*side` row-major array.
fn forward_square(
    frame: &PreparedFrame,
    transform: TransformType,
    px: u32,
    py: u32,
    scratch: &mut ForwardScratch,
) -> Result<usize> {
    let side = transform.sample_cols();
    let cells = side * side;
    for channel in 0..NUM_CHANNELS {
        gather_square(
            frame,
            channel,
            px,
            py,
            u32::try_from(side).unwrap_or(0),
            &mut scratch.samples,
        );
        let coeff = scratch
            .coeffs
            .get_mut(channel)
            .ok_or(PolicyError::Unsupported {
                what: "a coefficient channel",
            })?;
        let (Some(view), Some(mut out)) = (
            SampleView::contiguous(scratch.samples.get(..cells).unwrap_or(&[]), side, side),
            coeff
                .get_mut(..cells)
                .and_then(|c| CoeffViewMut::contiguous(c, side, side)),
        ) else {
            return Err(PolicyError::Unsupported {
                what: "a varblock sample view",
            });
        };
        forward_varblock_into(transform, &view, &mut out, &mut scratch.transform);
    }
    Ok(side)
}

/// I.8 inverted for a square transform: the varblock's `n x n` LF samples from
/// the top-left `n x n` LLF sub-block of its coefficient array.
fn lf_samples_of(
    coeff: &[f32],
    transform: TransformType,
    n: usize,
    side: usize,
    scratch: &mut TransformScratch,
    lf_out: &mut [f32],
) -> Result<()> {
    let (Some(llf), Some(mut out)) = (
        CoeffView::new(coeff, n, n, side),
        lf_out
            .get_mut(..n * n)
            .and_then(|o| SampleViewMut::contiguous(o, n, n)),
    ) else {
        return Err(PolicyError::Unsupported {
            what: "an LLF view",
        });
    };
    lf_from_llf_into(transform, &llf, &mut out, scratch);
    Ok(())
}

/// Whether coefficient cell `(x, y)` of a square transform is an LLF cell
/// (the top-left `n x n`, which the decoder overwrites from the LF image and
/// the HF walk never codes).
const fn is_llf_cell(cell: usize, side: usize, n: usize) -> bool {
    let x = cell % side;
    let y = cell / side;
    x < n && y < n
}

/// The four chroma-from-luma multipliers a varblock applies: the frame-wide LF
/// pair (I.2.3) and its own tile's HF pair (I.6).
#[derive(Debug, Clone, Copy)]
struct VarblockCfl {
    k_x_lf: f32,
    k_b_lf: f32,
    k_x_hf: f32,
    k_b_hf: f32,
}

/// Writes one LF plane cell, ignoring out-of-grid coordinates.
fn set_lf(
    planes: &mut [Vec<i32>; NUM_CHANNELS],
    channel: usize,
    bx: u32,
    by: u32,
    width: u32,
    value: i32,
) {
    let index =
        usize::try_from(u64::from(by) * u64::from(width) + u64::from(bx)).unwrap_or(usize::MAX);
    if let Some(slot) = planes.get_mut(channel).and_then(|p| p.get_mut(index)) {
        *slot = value;
    }
}

/// Merges a chunk's group-grid LF planes into the group reduction.
///
/// Chunks of one LF group cover disjoint varblocks, and unwritten cells stay
/// zero, so adding plane-wise is the same as scattering each patch. The add
/// is the vectorisable merge; overlapping non-zero writes would wrap in
/// debug and are a cover bug.
fn add_lf_planes(
    dest: &mut [Vec<i32>; NUM_CHANNELS],
    src: &[Vec<i32>; NUM_CHANNELS],
) -> Result<()> {
    for channel in 0..NUM_CHANNELS {
        let (Some(d), Some(s)) = (dest.get_mut(channel), src.get(channel)) else {
            return Err(PolicyError::Unsupported {
                what: "a missing LF plane while merging a quantization chunk",
            });
        };
        if d.len() != s.len() {
            return Err(PolicyError::Unsupported {
                what: "a quantization chunk whose LF plane size changed",
            });
        }
        for (slot, &value) in d.iter_mut().zip(s.iter()) {
            *slot += value;
        }
    }
    Ok(())
}

/// Phase 7.1's estimate of what one interior-zero coefficient token costs.
///
/// Token *counts* are exact — the walk emits one per order position up to the
/// last nonzero — but token *bits* are not, because each symbol's real cost
/// depends on the ANS cluster and hybrid-uint configuration the census builds
/// after the walk. Phase 7.1a measured 291,954 symbols against 98,136 bytes at
/// 1 bpp, an average of 2.7 bits per symbol across the whole stream; zero
/// tokens are 62.9% of coefficient tokens there and so must sit well below that
/// average. One bit is a deliberately conservative estimate of that: it
/// under-credits truncation rather than over-credits it, so the pass errs
/// toward keeping coefficients.
///
/// This is a stated estimate, not a measurement. Sweeping it is the first thing
/// to try if the pass screens close to neutral — the Quality track does that
/// through [`EncodeRequest::zero_token_bits`], whose default this is.
pub const ZERO_TOKEN_BITS: f32 = 1.0;

/// Reused per-varblock temporaries for [`quantize_square_varblock`].
struct QuantScratch {
    d_y_lf: Vec<f32>,
    d_y_hf: Vec<f32>,
    lf_scratch: Vec<f32>,
    /// Chroma residual targets for Phase-2 lane quantize.
    chroma_targets: Vec<f32>,
}

impl QuantScratch {
    fn new() -> Self {
        Self {
            d_y_lf: Vec::new(),
            d_y_hf: Vec::new(),
            lf_scratch: Vec::new(),
            chroma_targets: Vec::new(),
        }
    }

    fn resize(&mut self, n: usize, cells: usize) {
        // Every live element is overwritten below before it is read: LF
        // transforms fill `lf_scratch`, LF quantization fills `d_y_lf`, the Y
        // reconstruction fills `d_y_hf`, and each chroma row fills
        // `chroma_targets`. `resize` still initializes newly grown tails.
        self.d_y_lf.resize(n * n, 0.0);
        self.d_y_hf.resize(cells, 0.0);
        self.lf_scratch.resize(n * n, 0.0);
        self.chroma_targets.resize(cells, 0.0);
    }
}

/// Quantizes one square varblock: LF samples into `lf_planes`, HF coefficients
/// into the three channel slices (length `side*side` each). Y first, because
/// X and B decorrelate against the reconstructed `dY` (I.6), never the source Y.
#[allow(clippy::too_many_arguments)]
fn quantize_square_varblock(
    coeffs: &[&[f32]; NUM_CHANNELS],
    transform: TransformType,
    lf_quant: &LfQuantizer,
    hf_quant: &HfQuantizer,
    cfl: VarblockCfl,
    scratch: &mut TransformScratch,
    qscratch: &mut QuantScratch,
    mut write_lf: impl FnMut(usize, usize, i32),
    quant: &mut [i32],
    truncate: bool,
) -> Result<()> {
    let n = transform.block_dims().0;
    let side = transform.sample_cols();
    let cells = side * side;
    if quant.len() < cells * NUM_CHANNELS {
        return Err(PolicyError::Unsupported {
            what: "a coefficient arena slice shorter than three full channels",
        });
    }
    qscratch.resize(n, cells);

    // Channel layout in `quant`: X | Y | B, each `cells` long.
    let (qx, rest) = quant.split_at_mut(cells);
    let (qy, qb) = rest.split_at_mut(cells);

    // --- Y (channel 1): independent, and everything else needs it ---
    let y_coeff = coeffs.get(1).ok_or(PolicyError::Unsupported {
        what: "the Y coefficient channel",
    })?;
    lf_samples_of(
        y_coeff,
        transform,
        n,
        side,
        scratch,
        &mut qscratch.lf_scratch,
    )?;
    for idx in 0..n * n {
        let q = lf_quant.quantize(qscratch.lf_scratch.get(idx).copied().unwrap_or(0.0), 1)?;
        write_lf(1, idx, q);
        if let Some(slot) = qscratch.d_y_lf.get_mut(idx) {
            *slot = lf_quant.reconstruct(q, 1);
        }
    }
    // Phase-2: final HF quant as contiguous lanes (Y, then chroma with CfL).
    // Phase 39: the lane pass also yields every cell's reconstruction, so
    // `d_y_hf` is filled here rather than by a second `reconstruct` sweep.
    hf_quant.quantize_lane_with_recon(
        1,
        y_coeff,
        qy,
        qscratch.d_y_hf.get_mut(..cells),
        side,
        n,
        true,
    )?;
    // Phase 7.1 truncation runs on Y *before* `d_y_hf` is read, so the chroma
    // CfL targets decorrelate against the Y the decoder will actually
    // reconstruct rather than against coefficients this pass then drops.
    // Phase 30: a `'static` borrow of the shared per-Order-ID cache instead
    // of a fresh clone every varblock; see `natural_coeff_order_ref`.
    let order: &[u32] = if truncate {
        transform.natural_coeff_order_ref()
    } else {
        &[]
    };
    if truncate {
        hf_quant.truncate_trailing(
            1,
            qy,
            order,
            n * n,
            hf_quant.zero_token_bits(),
            |cell| y_coeff.get(cell).copied().unwrap_or(0.0),
            |cell| {
                if let Some(slot) = qscratch.d_y_hf.get_mut(cell) {
                    *slot = hf_quant.reconstruct(0, 1, cell);
                }
            },
        );
    }

    // --- X and B: X = dX + kX*dY, B = dB + kB*dY (I.6) ---
    for (channel, out) in [(0usize, &mut *qx), (2usize, &mut *qb)] {
        let (k_lf, k_hf) = if channel == 0 {
            (cfl.k_x_lf, cfl.k_x_hf)
        } else {
            (cfl.k_b_lf, cfl.k_b_hf)
        };
        let coeff = coeffs.get(channel).ok_or(PolicyError::Unsupported {
            what: "a chroma coefficient channel",
        })?;
        lf_samples_of(coeff, transform, n, side, scratch, &mut qscratch.lf_scratch)?;
        for idx in 0..n * n {
            let target = qscratch.lf_scratch.get(idx).copied().unwrap_or(0.0)
                - k_lf * qscratch.d_y_lf.get(idx).copied().unwrap_or(0.0);
            let q = lf_quant.quantize(target, channel)?;
            write_lf(channel, idx, q);
        }
        for (row, targets) in qscratch
            .chroma_targets
            .chunks_exact_mut(side)
            .take(side)
            .enumerate()
        {
            let first_hf = if row < n { n.min(side) } else { 0 };
            let (llf, hf) = targets.split_at_mut(first_hf);
            llf.fill(0.0);
            let base = row.saturating_mul(side).saturating_add(first_hf);
            // Phase 39: written as a zip over row slices so the compiler can
            // vectorise the multiply-subtract (same per-cell arithmetic and
            // rounding; the slices exist for every row of a full arena, and
            // the per-cell fallback covers a short one).
            match (
                coeff.get(base..base + hf.len()),
                qscratch.d_y_hf.get(base..base + hf.len()),
            ) {
                (Some(src), Some(d_y)) => {
                    for ((slot, &c), &d) in hf.iter_mut().zip(src.iter()).zip(d_y.iter()) {
                        *slot = c - k_hf * d;
                    }
                }
                _ => {
                    for (offset, slot) in hf.iter_mut().enumerate() {
                        let cell = base.saturating_add(offset);
                        *slot = coeff.get(cell).copied().unwrap_or(0.0)
                            - k_hf * qscratch.d_y_hf.get(cell).copied().unwrap_or(0.0);
                    }
                }
            }
        }
        hf_quant.quantize_lane(channel, &qscratch.chroma_targets, out, side, n, true)?;
        if truncate {
            hf_quant.truncate_trailing(
                channel,
                out,
                order,
                n * n,
                hf_quant.zero_token_bits(),
                |cell| qscratch.chroma_targets.get(cell).copied().unwrap_or(0.0),
                |_| {},
            );
        }
    }

    Ok(())
}

/// The tile (64x64) chroma-from-luma multipliers for a varblock at group-local
/// atom `(bx, by)`. A milestone-6 varblock is at most 32 samples per side, so
/// it lies inside a single tile and takes that tile's factor pair.
fn varblock_cfl(
    correlation: &LfCorrelationDecision,
    cfl: &CflGrid,
    bx: u32,
    by: u32,
) -> VarblockCfl {
    let k_x_lf = cfl_multiplier(
        correlation.base_correlation_x,
        i32::from(correlation.x_factor_lf) - 128,
        correlation.colour_factor,
    );
    let k_b_lf = cfl_multiplier(
        correlation.base_correlation_b,
        i32::from(correlation.b_factor_lf) - 128,
        correlation.colour_factor,
    );
    let tile =
        usize::try_from(u64::from(by / 8) * u64::from(cfl.tiles().width) + u64::from(bx / 8))
            .unwrap_or(usize::MAX);
    let k_x_hf = cfl_multiplier(
        correlation.base_correlation_x,
        cfl.x_from_y().get(tile).copied().unwrap_or_default().get(),
        correlation.colour_factor,
    );
    let k_b_hf = cfl_multiplier(
        correlation.base_correlation_b,
        cfl.b_from_y().get(tile).copied().unwrap_or_default().get(),
        correlation.colour_factor,
    );
    VarblockCfl {
        k_x_lf,
        k_b_lf,
        k_x_hf,
        k_b_hf,
    }
}

/// Regression first, then integer refinement in the factor space I.6 consumes,
/// over the **selected** varblock map.
///
/// The least-squares seed is trained in the unquantized coefficient domain,
/// `Sum(Y*C) / Sum(Y^2)`, separately for LF and each HF tile. It is refined
/// over nearby stored factors using reconstructed `dY`, quantizing the chroma
/// residual through the exact quantizers (including I.5.3's bias). Grayscale is
/// a hard source-domain no-op, so its stream is byte-identical to neutral CfL.
///
/// Each dense group bank must contain one forward per selected varblock in
/// `maps`; cover construction establishes that invariant before this call.
fn estimate_cfl(
    geometry: &VardctGeometry,
    maps: &[PlannedVarblockRange],
    cache: &CandidateForwardCache,
    lf_quant: &LfQuantizer,
    hf_quants: &HfQuantizers,
    enabled: bool,
    executor: Option<&jpxl_encode::EncodeExecutor>,
) -> Result<CflEstimate> {
    // Grayscale / disabled: no need for the forward cache.
    if !enabled {
        let groups = (0..geometry.num_lf_groups())
            .filter_map(|index| {
                let id = LfGroupId::new(u32::try_from(index).ok()?);
                geometry.lf_group_cfl_tiles(id).map(CflGrid::zeros)
            })
            .collect();
        return Ok(CflEstimate {
            correlation: LfCorrelationDecision::default(),
            groups,
        });
    }
    // Phase 38: collection is split into 64-pixel bands (one tile row of one
    // LF group each) so a frame with only one or two LF groups still fills
    // every worker. Results are returned in (group, band) raster order, then
    // the frame-wide LF regression replays every sample in that same order so
    // worker scheduling cannot perturb floating-point sums.
    let n_groups = usize::try_from(geometry.num_lf_groups()).unwrap_or(0);
    let mut bands: Vec<CflBand> = Vec::new();
    // Allocate the large sample vectors on the request thread. Workers fill
    // them without growing, avoiding one allocator arena retaining a frame's
    // CfL samples after another thread drops the completed plan.
    let mut sample_workspaces = Vec::new();
    for group in 0..n_groups {
        let index = u64::try_from(group).unwrap_or(u64::MAX);
        let id = LfGroupId::new(u32::try_from(index).unwrap_or(u32::MAX));
        let tiles = geometry
            .lf_group_cfl_tiles(id)
            .ok_or(PolicyError::Unsupported {
                what: "an LF group outside the frame's grid",
            })?;
        let map = maps.get(group).ok_or(PolicyError::Unsupported {
            what: "a missing block map for an LF group",
        })?;
        debug_assert!(
            map.decisions
                .iter()
                .zip(map.decisions.iter().skip(1))
                .all(|(a, b)| (a.origin.by(), a.origin.bx()) <= (b.origin.by(), b.origin.bx())),
            "LF group varblocks must be in raster order"
        );
        for tile_row in 0..tiles.height {
            let first = map
                .decisions
                .partition_point(|vb| vb.origin.by() / 8 < tile_row);
            let end = map
                .decisions
                .partition_point(|vb| vb.origin.by() / 8 <= tile_row);
            let decisions = map
                .decisions
                .get(first..end)
                .ok_or(PolicyError::Unsupported {
                    what: "a CfL band range outside its block map",
                })?;
            sample_workspaces.push(std::sync::Mutex::new(Some(band_cfl_workspace(
                tiles, tile_row, decisions,
            )?)));
            bands.push(CflBand {
                group,
                tile_row,
                first,
                end,
            });
        }
    }
    let collect_band = |band_index: usize| {
        let band = bands
            .get(band_index)
            .copied()
            .ok_or(PolicyError::Unsupported {
                what: "a CfL band index outside the frame",
            })?;
        let index_usize = band.group;
        let index = u64::try_from(index_usize).unwrap_or(u64::MAX);
        let id = LfGroupId::new(u32::try_from(index).unwrap_or(u32::MAX));
        let tiles = geometry
            .lf_group_cfl_tiles(id)
            .ok_or(PolicyError::Unsupported {
                what: "an LF group outside the frame's grid",
            })?;
        let mut workspace = sample_workspaces
            .get(band_index)
            .ok_or(PolicyError::Unsupported {
                what: "a missing CfL sample workspace",
            })?
            .lock()
            .map_err(|_| PolicyError::Unsupported {
                what: "a poisoned CfL sample workspace",
            })?
            .take()
            .ok_or(PolicyError::Unsupported {
                what: "a CfL sample workspace used twice",
            })?;
        debug_assert_eq!(workspace.hf.tiles, tiles);
        let mut llf_scratch = TransformScratch::for_transform(TransformType::Dct32x32);
        // Phase 28: reused across every varblock in this group instead of
        // three fresh `Vec<f32>` allocations per varblock. `lf_samples_of`
        // always overwrites exactly the first `n * n` cells it is given, so
        // `resize` (which only reallocates when the new length exceeds a
        // capacity already grown to the largest varblock seen) is equivalent
        // to the prior always-zeroed fresh vector for every read below.
        let mut y_lf: Vec<f32> = Vec::new();
        let mut x_lf: Vec<f32> = Vec::new();
        let mut b_lf: Vec<f32> = Vec::new();
        let map = maps.get(index_usize).ok_or(PolicyError::Unsupported {
            what: "a missing block map for an LF group",
        })?;
        let rect = geometry.lf_group_rect(id).ok_or(PolicyError::Unsupported {
            what: "an LF group outside the frame's grid",
        })?;
        let bank = cache
            .group(index_usize)?
            .read()
            .map_err(|_| PolicyError::Unsupported {
                what: "a poisoned LF-group forward bank",
            })?;
        let decisions =
            map.decisions
                .get(band.first..band.end)
                .ok_or(PolicyError::Unsupported {
                    what: "a CfL band range outside its block map",
                })?;
        let group_fwd = gather_forward_refs(decisions, (rect.x0, rect.y0), &bank)?;
        if decisions.len() != group_fwd.len() {
            return Err(PolicyError::Unsupported {
                what: "a forward cache length that does not match the block map",
            });
        }

        for (offset, (vb, fwd)) in decisions.iter().zip(group_fwd.iter()).enumerate() {
            let vb_index = band.first + offset;
            let [cx, cy, cb] = fwd.coeffs;
            let transform = vb.transform;
            let n = transform.block_dims().0;
            let side = transform.sample_cols();
            let hf_quant = hf_quants.get(transform, map.hf_mul(vb_index)?)?;
            if vb.origin.by() / 8 != band.tile_row {
                return Err(PolicyError::Unsupported {
                    what: "a varblock outside its CfL band",
                });
            }
            // Tile column within the band; the band owns exactly one tile row.
            let tile = usize::try_from(vb.origin.bx() / 8).unwrap_or(usize::MAX);

            // LF: the varblock's n*n LF samples, chroma against reconstructed dY.
            y_lf.resize(n * n, 0.0);
            x_lf.resize(n * n, 0.0);
            b_lf.resize(n * n, 0.0);
            lf_samples_of(cy, transform, n, side, &mut llf_scratch, &mut y_lf)?;
            lf_samples_of(cx, transform, n, side, &mut llf_scratch, &mut x_lf)?;
            lf_samples_of(cb, transform, n, side, &mut llf_scratch, &mut b_lf)?;
            for idx in 0..n * n {
                let y = y_lf.get(idx).copied().unwrap_or(0.0);
                let q = lf_quant.quantize(y, 1)?;
                let d_y = lf_quant.reconstruct(q, 1);
                workspace
                    .lf_x
                    .push(x_lf.get(idx).copied().unwrap_or(0.0), y, d_y, 0);
                workspace
                    .lf_b
                    .push(b_lf.get(idx).copied().unwrap_or(0.0), y, d_y, 0);
                workspace.lf_regression_x.push(RegressionSample {
                    source: x_lf.get(idx).copied().unwrap_or(0.0),
                    y,
                });
                workspace.lf_regression_b.push(RegressionSample {
                    source: b_lf.get(idx).copied().unwrap_or(0.0),
                    y,
                });
            }

            // HF: every non-LLF coefficient, into the varblock's tile. The
            // sample's cell is folded onto the 8x8 frequency grid (identity
            // for DCT8x8), because `refine_hf_factor` scores every tile with
            // the DCT8x8 quantizer as the common scale and its matrix has no
            // entries beyond 8x8.
            let fold = side / 8;
            diagnostics::with_choose_stage(diagnostics::ChooseStage::CflY, || {
                // Phase 36: the non-LLF cells of each row are one contiguous
                // run, so quantize the row through the run kernel (cell-for-
                // cell identical to `choose`/`reconstruct`) and then push the
                // samples in the same raster order as before. A row the kernel
                // cannot quantize is replayed in scalar order, so the public
                // error and any earlier work match the pre-batch path exactly.
                let mut q_row = [0i32; MAX_SCORE_ROW];
                let mut recon_row = [0.0f32; MAX_SCORE_ROW];
                let mut push_row = |first_cell: usize, recons: &[f32]| {
                    for (i, &d_y) in recons.iter().enumerate() {
                        let cell = first_cell + i;
                        let cell8 = (cell / side / fold.max(1)) * 8 + (cell % side / fold.max(1));
                        let y = cy.get(cell).copied().unwrap_or(0.0);
                        if let Some(t) = workspace.hf.x.get_mut(tile) {
                            t.push(cx.get(cell).copied().unwrap_or(0.0), y, d_y, cell8);
                        }
                        if let Some(t) = workspace.hf.b.get_mut(tile) {
                            t.push(cb.get(cell).copied().unwrap_or(0.0), y, d_y, cell8);
                        }
                    }
                };
                for row in 0..side {
                    let first_col = if row < n { n } else { 0 };
                    let first_cell = row * side + first_col;
                    let run_len = side - first_col;
                    let Some(targets) = cy.get(first_cell..first_cell + run_len) else {
                        continue;
                    };
                    let batched = match (q_row.get_mut(..run_len), recon_row.get_mut(..run_len)) {
                        (Some(qs), Some(recons)) => hf_quant
                            .choose_run(1, first_cell, targets, qs, Some(recons))
                            .is_ok(),
                        _ => false,
                    };
                    if batched {
                        if let Some(recons) = recon_row.get(..run_len) {
                            push_row(first_cell, recons);
                        }
                        continue;
                    }
                    for (i, &y) in targets.iter().enumerate() {
                        let cell = first_cell + i;
                        let q_y = hf_quant.choose(y, 1, cell)?;
                        let d_y = hf_quant.reconstruct(q_y, 1, cell);
                        push_row(cell, core::slice::from_ref(&d_y));
                    }
                }
                Ok::<(), PolicyError>(())
            })?;
        }
        Ok::<_, PolicyError>(workspace)
    };
    let band_samples = if let Some(executor) = executor {
        executor.map_ordered(bands.len(), collect_band)?
    } else {
        (0..bands.len())
            .map(collect_band)
            .collect::<Result<Vec<_>>>()?
    };

    // Reassemble per LF group, bands in tile-row order: the LF streams are
    // appended in the original varblock order, and the bands' tile rows are
    // concatenated into the group's tile raster (row * width + column).
    let mut lf_x = CflSamples::default();
    let mut lf_b = CflSamples::default();
    let mut hf_groups: Vec<HfCflSamples> = Vec::with_capacity(n_groups);
    for (band, samples) in bands.iter().zip(band_samples) {
        lf_x.append_ordered(samples.lf_x, samples.lf_regression_x);
        lf_b.append_ordered(samples.lf_b, samples.lf_regression_b);
        if band.tile_row == 0 {
            hf_groups.push(HfCflSamples {
                tiles: samples.hf.tiles,
                x: Vec::with_capacity(usize::try_from(samples.hf.tiles.area()).unwrap_or(0)),
                b: Vec::with_capacity(usize::try_from(samples.hf.tiles.area()).unwrap_or(0)),
            });
        }
        if hf_groups.len() != band.group + 1 {
            return Err(PolicyError::Unsupported {
                what: "a CfL band out of group order",
            });
        }
        let group = hf_groups.last_mut().ok_or(PolicyError::Unsupported {
            what: "a CfL band before its group's first row",
        })?;
        if group.tiles != samples.hf.tiles {
            return Err(PolicyError::Unsupported {
                what: "a CfL band whose tile grid disagrees with its group",
            });
        }
        group.x.extend(samples.hf.x);
        group.b.extend(samples.hf.b);
    }

    let (x_factor, b_factor) = refine_lf_factors(&lf_x, &lf_b, lf_quant)?;
    let x_factor_lf = u8::try_from(x_factor + 128).map_err(|_| PolicyError::Unsupported {
        what: "an LF X correlation factor outside u8",
    })?;
    let b_factor_lf = u8::try_from(b_factor + 128).map_err(|_| PolicyError::Unsupported {
        what: "an LF B correlation factor outside u8",
    })?;
    let correlation = LfCorrelationDecision {
        colour_factor: DEFAULT_COLOUR_FACTOR,
        base_correlation_x: 0.0,
        base_correlation_b: 1.0,
        x_factor_lf,
        b_factor_lf,
    };

    // Phase 38: the per-tile HF factor refinement is independent per tile, so
    // it runs across the executor's workers (X tiles then B tiles of each LF
    // group, results reduced in tile order -- Contract A) instead of on the
    // calling thread alone. Each tile's arithmetic is unchanged.
    let baseline = hf_quants.baseline(TransformType::Dct8x8)?;
    let mut groups = Vec::with_capacity(hf_groups.len());
    for group in hf_groups {
        let n_x = group.x.len();
        let refine = |index: usize| -> Result<CflFactor> {
            if let Some(tile) = group.x.get(index) {
                return Ok(CflFactor::new(refine_hf_factor(tile, 0.0, 0, baseline)?));
            }
            let tile = group.b.get(index - n_x).ok_or(PolicyError::Unsupported {
                what: "a CfL tile index outside its group",
            })?;
            Ok(CflFactor::new(refine_hf_factor(tile, 1.0, 2, baseline)?))
        };
        let total = n_x + group.b.len();
        let mut factors = if let Some(executor) = executor {
            executor.map_ordered(total, refine)?
        } else {
            (0..total).map(refine).collect::<Result<Vec<_>>>()?
        };
        let b = factors.split_off(n_x);
        groups.push(CflGrid::new(group.tiles, factors, b)?);
    }
    Ok(CflEstimate {
        correlation,
        groups,
    })
}

/// The candidate integer factors to score: a small window around the
/// least-squares seed, always including the neutral factor `0`.
///
/// Phase-2: window is `seed±1` (plus 0), not `±4`. The LS seed already sits at
/// the continuous optimum; wider residual-bit search rarely moved the winner
/// and multiplied `choose` calls by ~3× on every tile.
fn factor_candidates(seed: i32, lo: i32, hi: i32) -> Vec<i32> {
    let mut out = Vec::with_capacity(4);
    for delta in -1i32..=1 {
        let candidate = seed.saturating_add(delta).clamp(lo, hi);
        if !out.contains(&candidate) {
            out.push(candidate);
        }
    }
    if !out.contains(&0) {
        out.push(0);
    }
    out
}

/// A crude but monotone bit-cost of one quantized residual.
///
/// CfL and block choice barely move the reconstructed value — the quantizer
/// re-targets either way — so the axis that actually moves is *rate*, not
/// distortion. This proxy is the magnitude class of the coefficient: zero is
/// free (the overwhelming symbol, priced near nothing by the context model),
/// and a non-zero costs its bit length plus a sign. Minimizing its sum is
/// minimizing residual entropy in the only currency a per-candidate search can
/// spend without a full re-encode.
fn residual_bits(q: i32) -> u64 {
    if q == 0 {
        0
    } else {
        u64::from(32 - q.unsigned_abs().leading_zeros()) + 1
    }
}

/// The signaling cost of one stored factor, in the same currency.
fn factor_bits(factor: i32) -> u64 {
    if factor == 0 {
        0
    } else {
        u64::from(32 - factor.unsigned_abs().leading_zeros()) + 2
    }
}

/// The residual bit-cost of one candidate LF factor over a channel's DC.
fn lf_residual_cost(
    samples: &CflSamples,
    base: f32,
    factor: i32,
    channel: usize,
    quantizer: &LfQuantizer,
) -> Result<u64> {
    let k = cfl_multiplier(base, factor, DEFAULT_COLOUR_FACTOR);
    let mut bits = 0u64;
    for sample in &samples.samples {
        let q = quantizer.quantize(sample.source - k * sample.reconstructed_y, channel)?;
        bits = bits.saturating_add(residual_bits(q));
    }
    Ok(bits)
}

/// Joint LF refinement: the two factors share one I.2.3 bundle, so charging
/// them independently would adopt a tiny win that cannot repay the bundle's
/// fixed fields.
fn refine_lf_factors(
    x: &CflSamples,
    b: &CflSamples,
    quantizer: &LfQuantizer,
) -> Result<(i32, i32)> {
    // Explicit I.2.3 costs 51 bits at the defaults used here: all_default=false
    // (1), colour_factor's selector (2), two F16s (32), the two biased factors
    // (16). The all-default form costs one bit, so a non-neutral pair must save
    // at least 50 residual bits to be worth it.
    const LF_BUNDLE_EXTRA_BITS: u64 = 50;

    let x_seed = x
        .regression
        .best_factor(0.0, DEFAULT_COLOUR_FACTOR, -128, 127);
    let b_seed = b
        .regression
        .best_factor(1.0, DEFAULT_COLOUR_FACTOR, -128, 127);
    let x_candidates = factor_candidates(x_seed, -128, 127);
    let b_candidates = factor_candidates(b_seed, -128, 127);

    // Phase-1: residual costs are independent across channels — compute once
    // per candidate, then combine (was re-pricing B inside every X pair).
    let x_costs: Result<Vec<(i32, u64)>> = x_candidates
        .iter()
        .map(|&f| Ok((f, lf_residual_cost(x, 0.0, f, 0, quantizer)?)))
        .collect();
    let x_costs = x_costs?;
    let b_costs: Result<Vec<(i32, u64)>> = b_candidates
        .iter()
        .map(|&f| Ok((f, lf_residual_cost(b, 1.0, f, 2, quantizer)?)))
        .collect();
    let b_costs = b_costs?;

    let neutral_x = x_costs
        .iter()
        .find(|(f, _)| *f == 0)
        .map(|(_, c)| *c)
        .unwrap_or(0);
    let neutral_b = b_costs
        .iter()
        .find(|(f, _)| *f == 0)
        .map(|(_, c)| *c)
        .unwrap_or(0);
    let mut best = (0i32, 0i32);
    let mut best_cost = neutral_x.saturating_add(neutral_b);
    for &(x_factor, x_cost) in &x_costs {
        for &(b_factor, b_cost) in &b_costs {
            let mut cost = x_cost.saturating_add(b_cost);
            if x_factor != 0 || b_factor != 0 {
                cost = cost
                    .saturating_add(LF_BUNDLE_EXTRA_BITS)
                    .saturating_add(factor_bits(x_factor))
                    .saturating_add(factor_bits(b_factor));
            }
            let magnitude = x_factor.unsigned_abs() + b_factor.unsigned_abs();
            let best_magnitude = best.0.unsigned_abs() + best.1.unsigned_abs();
            if cost < best_cost || (cost == best_cost && magnitude < best_magnitude) {
                best = (x_factor, b_factor);
                best_cost = cost;
            }
        }
    }
    Ok(best)
}

/// Residual bit-cost of one HF CfL factor. When `cutoff` is set, returns
/// `None` as soon as the running cost cannot beat the cutoff (ties retain the
/// previous best, so `>=` is safe).
fn hf_residual_cost_bounded(
    samples: &CflSamples,
    base: f32,
    factor: i32,
    channel: usize,
    quantizer: &HfQuantizer,
    cutoff: Option<u64>,
) -> Result<Option<u64>> {
    let k = cfl_multiplier(base, factor, DEFAULT_COLOUR_FACTOR);
    let mut bits = 0u64;
    // Phase 36: price the residuals in blocks of up to `CFL_PRICE_CHUNK`
    // samples through the indexed lane kernel, then walk the block's results
    // in sample order with the same per-sample cutoff rule as before. A block
    // the kernel cannot quantize is replayed in scalar order, which besides
    // reproducing the first concrete error preserves the old cutoff rule: an
    // earlier sample may prove the challenger loses before a later invalid
    // target would have been visited.
    let mut targets = [0.0f32; CFL_PRICE_CHUNK];
    let mut cells = [0usize; CFL_PRICE_CHUNK];
    let mut quantized = [0i32; CFL_PRICE_CHUNK];
    for block in samples.samples.chunks(CFL_PRICE_CHUNK) {
        let len = block.len();
        let (Some(targets), Some(cells), Some(quantized)) = (
            targets.get_mut(..len),
            cells.get_mut(..len),
            quantized.get_mut(..len),
        ) else {
            unreachable!("chunks() never yields more than CFL_PRICE_CHUNK samples");
        };
        for ((sample, target), cell) in block.iter().zip(targets.iter_mut()).zip(cells.iter_mut()) {
            *target = sample.source - k * sample.reconstructed_y;
            *cell = sample.cell;
        }
        if quantizer
            .choose_cells(channel, targets, cells, quantized, None)
            .is_ok()
        {
            for &q in quantized.iter() {
                bits = bits.saturating_add(residual_bits(q));
                // Strict > so equal residual+signalling costs still finish for
                // magnitude tie-breaks toward neutral/smaller factor.
                if cutoff.is_some_and(|cut| bits > cut) {
                    return Ok(None);
                }
            }
        } else {
            for (&target, &cell) in targets.iter().zip(cells.iter()) {
                let q = quantizer.choose(target, channel, cell)?;
                bits = bits.saturating_add(residual_bits(q));
                if cutoff.is_some_and(|cut| bits > cut) {
                    return Ok(None);
                }
            }
        }
    }
    Ok(Some(bits))
}

/// The best HF factor for one 64x64 tile: the candidate whose residual-plus-
/// signaling bits are lowest, ties resolved toward the neutral factor so an
/// unhelpful tile costs nothing and stays byte-neutral.
///
/// The refinement scores with a DCT8x8 quantizer even for coefficients that
/// came from larger transforms: the HF factor is a single per-tile value the
/// decoder applies uniformly, and the DCT8x8 dequant step is the common scale
/// the tile is judged on. The factor's own wire range is the interoperable
/// signed byte `[-128, 127]` established in slice 15.
fn refine_hf_factor(
    samples: &CflSamples,
    base: f32,
    channel: usize,
    quantizer: &HfQuantizer,
) -> Result<i32> {
    if samples.samples.is_empty() {
        return Ok(0);
    }
    diagnostics::with_choose_stage(diagnostics::ChooseStage::CflFactor, || {
        let seed = samples.regression.best_factor(
            base,
            DEFAULT_COLOUR_FACTOR,
            HF_FACTOR_MIN,
            HF_FACTOR_MAX,
        );
        // Phase-2: continuous LS already neutral → skip residual-bit multi-choose.
        if seed == 0 {
            return Ok(0);
        }
        let mut best_factor = 0i32;
        let mut best_cost = hf_residual_cost_bounded(samples, base, 0, channel, quantizer, None)?
            .unwrap_or(u64::MAX);
        for factor in factor_candidates(seed, HF_FACTOR_MIN, HF_FACTOR_MAX) {
            if factor == 0 {
                continue;
            }
            // Challenger needs residual + signalling < best (ties keep prior).
            let residual_cutoff = best_cost.saturating_sub(factor_bits(factor));
            let Some(residual) = hf_residual_cost_bounded(
                samples,
                base,
                factor,
                channel,
                quantizer,
                Some(residual_cutoff),
            )?
            else {
                continue;
            };
            let cost = residual.saturating_add(factor_bits(factor));
            if cost < best_cost || (cost == best_cost && factor.abs() < best_factor.abs()) {
                best_cost = cost;
                best_factor = factor;
            }
        }
        Ok(best_factor)
    })
}

/// One LF group's quantized integers.
struct QuantizedGroup {
    lf: LfQuantPlanes,
    coefficients: Vec<VarblockCoefficients>,
    /// The reusable backing arena retained by the request-scoped workspace.
    /// The coefficient handles in `coefficients` keep their own `Arc` clones
    /// while this value is lowered into an emission plan.
    arena: Option<Arc<[i32]>>,
}

fn coefficient_arena_capacity(varblocks: &[VarblockDecision]) -> usize {
    varblocks.iter().fold(0usize, |capacity, vb| {
        let side = vb.transform.sample_cols();
        capacity.saturating_add(side.saturating_mul(side).saturating_mul(NUM_CHANNELS))
    })
}

/// Reusable backing storage for sequential rate-probe quantizers.
///
/// Quantized plans retain immutable `Arc` views into these arenas. Once an
/// anchor has been priced and dropped, the workspace is the sole owner again
/// and the next probe can overwrite the same allocation. If a caller violates
/// that lifetime (for example by trying to build a correction while its
/// finalist is still live), `take_arena` allocates a second arena rather than
/// mutating coefficients that an earlier plan can still observe.
struct QuantizationWorkspace {
    arenas: Vec<Arc<[i32]>>,
    /// Per-chunk LF planes in group-grid layout, returned after each probe's
    /// reduction so the next sequential quantizer does not re-reserve them.
    chunk_lf: Vec<[Vec<i32>; NUM_CHANNELS]>,
}

impl QuantizationWorkspace {
    fn new() -> Self {
        Self {
            arenas: Vec::new(),
            chunk_lf: Vec::new(),
        }
    }

    fn take_arena(&mut self, slot: usize, capacity: usize) -> Arc<[i32]> {
        let previous = self
            .arenas
            .get_mut(slot)
            .map(|arena| std::mem::replace(arena, Arc::<[i32]>::from(Vec::new())));
        match previous {
            Some(arena) if arena.len() >= capacity && Arc::strong_count(&arena) == 1 => arena,
            _ => vec![0i32; capacity].into(),
        }
    }

    fn put_arena(&mut self, slot: usize, arena: Arc<[i32]>) {
        if self.arenas.len() <= slot {
            self.arenas
                .resize_with(slot.saturating_add(1), || Arc::<[i32]>::from(Vec::new()));
        }
        if let Some(destination) = self.arenas.get_mut(slot) {
            *destination = arena;
        }
    }

    fn take_chunk_lf(&mut self, slot: usize, capacity: usize) -> [Vec<i32>; NUM_CHANNELS] {
        if self.chunk_lf.len() <= slot {
            self.chunk_lf.resize_with(slot.saturating_add(1), || {
                core::array::from_fn(|_| Vec::new())
            });
        }
        let mut planes = core::array::from_fn(|_| Vec::new());
        if let Some(stored) = self.chunk_lf.get_mut(slot) {
            std::mem::swap(&mut planes, stored);
        }
        for plane in &mut planes {
            if plane.len() != capacity {
                plane.clear();
                plane.resize(capacity, 0);
            } else {
                plane.fill(0);
            }
        }
        planes
    }

    fn put_chunk_lf(&mut self, slot: usize, planes: [Vec<i32>; NUM_CHANNELS]) {
        if let Some(stored) = self.chunk_lf.get_mut(slot) {
            *stored = planes;
        }
    }
}

/// Large quantization buffers reserved by the request thread before fanout.
struct QuantWorkspace {
    lf_planes: [Vec<i32>; NUM_CHANNELS],
    arena: std::sync::Arc<[i32]>,
    starts: Vec<[usize; NUM_CHANNELS]>,
    coefficients: Vec<VarblockCoefficients>,
}

impl QuantWorkspace {
    fn new(
        varblocks: &[VarblockDecision],
        blocks: jpxl_encode::vardct::BlockGrid,
        arena: Arc<[i32]>,
    ) -> Self {
        let cells = usize::try_from(blocks.area()).unwrap_or(0);
        Self {
            lf_planes: core::array::from_fn(|_| vec![0i32; cells]),
            arena,
            starts: Vec::with_capacity(varblocks.len()),
            coefficients: Vec::with_capacity(varblocks.len()),
        }
    }
}

/// One deterministic sub-LF-group quantization range.
#[derive(Clone, Copy)]
struct QuantChunk {
    group: usize,
    start: usize,
    end: usize,
}

/// Quantization keeps enough work queued for load balancing without turning
/// every 8x8 varblock into a separate Rayon job.
const QUANT_CHUNKS_PER_WORKER: usize = 4;

fn quantization_chunks(groups: &[PlannedGroup], workers: usize) -> Vec<QuantChunk> {
    let total_weight = groups
        .iter()
        .flat_map(|(_, _, _, varblocks)| varblocks)
        .map(|vb| {
            let side = vb.transform.sample_cols();
            side.saturating_mul(side)
        })
        .fold(0usize, usize::saturating_add);
    let target_chunks = workers
        .max(1)
        .saturating_mul(QUANT_CHUNKS_PER_WORKER)
        .max(groups.len())
        .max(1);
    let target_weight = total_weight.div_ceil(target_chunks).max(1);
    let mut chunks = Vec::with_capacity(target_chunks);

    for (group, (_, _, _, varblocks)) in groups.iter().enumerate() {
        if varblocks.is_empty() {
            chunks.push(QuantChunk {
                group,
                start: 0,
                end: 0,
            });
            continue;
        }
        let mut start = 0usize;
        let mut weight = 0usize;
        for (index, vb) in varblocks.iter().enumerate() {
            let side = vb.transform.sample_cols();
            let vb_weight = side.saturating_mul(side);
            if index > start && weight.saturating_add(vb_weight) > target_weight {
                chunks.push(QuantChunk {
                    group,
                    start,
                    end: index,
                });
                start = index;
                weight = 0;
            }
            weight = weight.saturating_add(vb_weight);
        }
        chunks.push(QuantChunk {
            group,
            start,
            end: varblocks.len(),
        });
    }
    chunks
}

/// Large buffers for one sub-LF-group quantization job, allocated on the
/// request thread before the executor starts it.
struct QuantChunkWorkspace {
    lf_values: [Vec<i32>; NUM_CHANNELS],
    lf_width: u32,
    arena: std::sync::Arc<[i32]>,
    starts: Vec<[usize; NUM_CHANNELS]>,
    coefficients: Vec<VarblockCoefficients>,
}

impl QuantChunkWorkspace {
    fn new(
        varblocks: &[VarblockDecision],
        arena: Arc<[i32]>,
        lf_values: [Vec<i32>; NUM_CHANNELS],
        lf_width: u32,
    ) -> Self {
        Self {
            lf_values,
            lf_width,
            arena,
            starts: Vec::with_capacity(varblocks.len()),
            coefficients: Vec::with_capacity(varblocks.len()),
        }
    }
}

struct QuantizedChunk {
    lf_values: [Vec<i32>; NUM_CHANNELS],
    coefficients: Vec<VarblockCoefficients>,
    arena: Arc<[i32]>,
}

struct QuantizedGroupBuilder {
    lf_planes: [Vec<i32>; NUM_CHANNELS],
    coefficients: Vec<VarblockCoefficients>,
}

/// Quantizes cost-balanced varblock ranges across the whole frame, then
/// reduces their LF patches and coefficient handles in group/raster order.
/// No task crosses an LF-group bank, and the banks are immutable at this
/// stage, so multiple ranges from the same group can borrow forwards at once.
fn quantize_groups_parallel(
    groups: &[PlannedGroup],
    cfl: &CflEstimate,
    cache: &CandidateForwardCache,
    lf_quant: &LfQuantizer,
    hf_quants: &HfQuantizers,
    executor: &jpxl_encode::EncodeExecutor,
    quant_workspace: &mut QuantizationWorkspace,
) -> Result<Vec<QuantizedGroup>> {
    let chunks = quantization_chunks(groups, executor.resources().threads);
    let workspaces: Vec<_> = chunks
        .iter()
        .enumerate()
        .map(|(index, chunk)| {
            let (blocks, varblocks) = groups
                .get(chunk.group)
                .map(|(_, blocks, _, varblocks)| (*blocks, varblocks.get(chunk.start..chunk.end)))
                .unwrap_or((
                    jpxl_encode::vardct::BlockGrid {
                        width: 0,
                        height: 0,
                    },
                    None,
                ));
            let varblocks = varblocks.unwrap_or(&[]);
            let arena = quant_workspace.take_arena(index, coefficient_arena_capacity(varblocks));
            let lf_cells = usize::try_from(blocks.area()).unwrap_or(0);
            let lf_values = quant_workspace.take_chunk_lf(index, lf_cells);
            std::sync::Mutex::new(Some(QuantChunkWorkspace::new(
                varblocks,
                arena,
                lf_values,
                blocks.width,
            )))
        })
        .collect();
    let quantize_one = |index: usize| {
        let chunk = chunks.get(index).copied().ok_or(PolicyError::Unsupported {
            what: "a missing sub-LF-group quantization range",
        })?;
        let (_, _, rect, group_varblocks) =
            groups.get(chunk.group).ok_or(PolicyError::Unsupported {
                what: "a missing planned LF group before chunk quantization",
            })?;
        let varblocks = group_varblocks.range(chunk.start, chunk.end).map_err(|_| {
            PolicyError::Unsupported {
                what: "a sub-LF-group quantization range outside its block map",
            }
        })?;
        let group_cfl = cfl
            .groups
            .get(chunk.group)
            .ok_or(PolicyError::Unsupported {
                what: "a missing CfL grid for a quantization chunk",
            })?;
        let bank = cache
            .group(chunk.group)?
            .read()
            .map_err(|_| PolicyError::Unsupported {
                what: "a poisoned LF-group forward bank",
            })?;
        let forwards = gather_forward_refs(varblocks.decisions, (rect.x0, rect.y0), &bank)?;
        let workspace = workspaces
            .get(index)
            .ok_or(PolicyError::Unsupported {
                what: "a missing sub-LF-group quantization workspace",
            })?
            .lock()
            .map_err(|_| PolicyError::Unsupported {
                what: "a poisoned sub-LF-group quantization workspace",
            })?
            .take()
            .ok_or(PolicyError::Unsupported {
                what: "a sub-LF-group quantization workspace used twice",
            })?;
        diagnostics::with_choose_stage(diagnostics::ChooseStage::Final, || {
            quantize_chunk(
                lf_quant,
                hf_quants,
                &cfl.correlation,
                group_cfl,
                &varblocks,
                &forwards,
                workspace,
            )
        })
    };
    let quantized_chunks = executor.map_ordered(chunks.len(), quantize_one)?;

    let mut merged: Vec<_> = groups
        .iter()
        .map(|(_, blocks, _, varblocks)| {
            let cells = usize::try_from(blocks.area()).unwrap_or(0);
            QuantizedGroupBuilder {
                lf_planes: core::array::from_fn(|_| vec![0i32; cells]),
                coefficients: Vec::with_capacity(varblocks.len()),
            }
        })
        .collect();
    for (index, (chunk, quantized)) in chunks.iter().copied().zip(quantized_chunks).enumerate() {
        quant_workspace.put_arena(index, quantized.arena);
        let (_, _, _, group_varblocks) =
            groups.get(chunk.group).ok_or(PolicyError::Unsupported {
                what: "a missing planned LF group during chunk reduction",
            })?;
        let varblocks =
            group_varblocks
                .get(chunk.start..chunk.end)
                .ok_or(PolicyError::Unsupported {
                    what: "a quantized chunk outside its block map during reduction",
                })?;
        if quantized.coefficients.len() != varblocks.len() {
            return Err(PolicyError::Unsupported {
                what: "a quantized chunk whose coefficient count changed",
            });
        }
        let builder = merged
            .get_mut(chunk.group)
            .ok_or(PolicyError::Unsupported {
                what: "a missing LF-group quantization reduction",
            })?;
        add_lf_planes(&mut builder.lf_planes, &quantized.lf_values)?;
        builder.coefficients.extend(quantized.coefficients);
        quant_workspace.put_chunk_lf(index, quantized.lf_values);
    }

    groups
        .iter()
        .zip(merged)
        .map(|((_, blocks, _, varblocks), merged)| {
            if merged.coefficients.len() != varblocks.len() {
                return Err(PolicyError::Unsupported {
                    what: "an LF-group chunk reduction whose coefficient count changed",
                });
            }
            Ok(QuantizedGroup {
                lf: LfQuantPlanes::new(*blocks, merged.lf_planes)?,
                coefficients: merged.coefficients,
                // Chunk arenas are returned to the request workspace as soon
                // as their deterministic reduction is complete.
                arena: None,
            })
        })
        .collect()
}

fn quantize_chunk(
    lf_quant: &LfQuantizer,
    hf_quants: &HfQuantizers,
    correlation: &LfCorrelationDecision,
    cfl: &CflGrid,
    varblocks: &PlannedVarblockRange,
    forwards: &[VarblockForward<'_>],
    mut workspace: QuantChunkWorkspace,
) -> Result<QuantizedChunk> {
    if varblocks.len() != forwards.len() {
        return Err(PolicyError::Unsupported {
            what: "a forward cache length that does not match a quantization chunk",
        });
    }
    let mut tscratch = TransformScratch::for_transform(TransformType::Dct32x32);
    let mut qscratch = QuantScratch::new();
    let mut cursor = 0usize;

    for (vb_index, (vb, fwd)) in varblocks.decisions.iter().zip(forwards.iter()).enumerate() {
        let transform = vb.transform;
        let side = transform.sample_cols();
        let ch_cells = side.saturating_mul(side);
        let span = ch_cells.saturating_mul(NUM_CHANNELS);
        let end = cursor.saturating_add(span);
        let slot = std::sync::Arc::get_mut(&mut workspace.arena)
            .ok_or(PolicyError::Unsupported {
                what: "a shared coefficient arena before chunk quantization",
            })?
            .get_mut(cursor..end)
            .ok_or(PolicyError::Unsupported {
                what: "a chunk coefficient arena that ran short of capacity",
            })?;
        let (bx, by) = (vb.origin.bx(), vb.origin.by());
        let factors = varblock_cfl(correlation, cfl, bx, by);
        let lf_planes = &mut workspace.lf_values;
        let lf_width = workspace.lf_width;
        let n = transform.block_dims().0;
        quantize_square_varblock(
            &fwd.coeffs,
            transform,
            lf_quant,
            hf_quants.get(transform, varblocks.hf_mul(vb_index)?)?,
            factors,
            &mut tscratch,
            &mut qscratch,
            |channel, idx, value| {
                set_lf(
                    lf_planes,
                    channel,
                    bx + u32::try_from(idx % n).unwrap_or(0),
                    by + u32::try_from(idx / n).unwrap_or(0),
                    lf_width,
                    value,
                );
            },
            slot,
            hf_quants.truncate_trailing,
        )?;
        workspace
            .starts
            .push([cursor, cursor + ch_cells, cursor + ch_cells * 2]);
        cursor = end;
    }

    for (vb, channel_starts) in varblocks.decisions.iter().zip(workspace.starts) {
        workspace
            .coefficients
            .push(VarblockCoefficients::from_arena(
                vb.transform,
                std::sync::Arc::clone(&workspace.arena),
                channel_starts,
            )?);
    }

    Ok(QuantizedChunk {
        lf_values: workspace.lf_values,
        coefficients: workspace.coefficients,
        arena: workspace.arena,
    })
}

/// Quantizes one LF group's **selected** varblocks, in their `BlockInfo` order.
///
/// `forwards` are the precomputed coefficient arrays from
/// [`gather_forward_refs`] (same length and order as `varblocks`). HF
/// coefficients for every varblock share one group arena
/// ([`VarblockCoefficients::from_arena`]) so entropy alternatives do not
/// re-allocate per-varblock coefficient boxes.
#[allow(clippy::too_many_arguments)]
fn quantize_group(
    lf_quant: &LfQuantizer,
    hf_quants: &HfQuantizers,
    correlation: &LfCorrelationDecision,
    cfl: &CflGrid,
    varblocks: &PlannedVarblockRange,
    forwards: &[VarblockForward<'_>],
    blocks: jpxl_encode::vardct::BlockGrid,
    mut workspace: QuantWorkspace,
) -> Result<QuantizedGroup> {
    if varblocks.len() != forwards.len() {
        return Err(PolicyError::Unsupported {
            what: "a forward cache length that does not match the block map",
        });
    }
    let mut tscratch = TransformScratch::for_transform(TransformType::Dct32x32);
    let mut qscratch = QuantScratch::new();
    let mut cursor = 0usize;

    for (vb_index, (vb, fwd)) in varblocks.decisions.iter().zip(forwards.iter()).enumerate() {
        let transform = vb.transform;
        let side = transform.sample_cols();
        let ch_cells = side * side;
        let span = ch_cells.saturating_mul(NUM_CHANNELS);
        let end = cursor.saturating_add(span);
        let slot = std::sync::Arc::get_mut(&mut workspace.arena)
            .ok_or(PolicyError::Unsupported {
                what: "a shared coefficient arena before quantization",
            })?
            .get_mut(cursor..end)
            .ok_or(PolicyError::Unsupported {
                what: "a coefficient arena that ran short of capacity",
            })?;
        let (bx, by) = (vb.origin.bx(), vb.origin.by());
        let factors = varblock_cfl(correlation, cfl, bx, by);
        let lf_planes = &mut workspace.lf_planes;
        quantize_square_varblock(
            &fwd.coeffs,
            transform,
            lf_quant,
            hf_quants.get(transform, varblocks.hf_mul(vb_index)?)?,
            factors,
            &mut tscratch,
            &mut qscratch,
            |channel, idx, value| {
                set_lf(
                    lf_planes,
                    channel,
                    bx + u32::try_from(idx % transform.block_dims().0).unwrap_or(0),
                    by + u32::try_from(idx / transform.block_dims().0).unwrap_or(0),
                    blocks.width,
                    value,
                );
            },
            slot,
            hf_quants.truncate_trailing,
        )?;
        workspace
            .starts
            .push([cursor, cursor + ch_cells, cursor + ch_cells * 2]);
        cursor = end;
    }

    for (vb, channel_starts) in varblocks.decisions.iter().zip(workspace.starts) {
        workspace
            .coefficients
            .push(VarblockCoefficients::from_arena(
                vb.transform,
                std::sync::Arc::clone(&workspace.arena),
                channel_starts,
            )?);
    }

    Ok(QuantizedGroup {
        lf: LfQuantPlanes::new(blocks, workspace.lf_planes)?,
        coefficients: workspace.coefficients,
        arena: Some(workspace.arena),
    })
}

/// The share of §4.3's `metadata_bits` every varblock pays under the legacy
/// rate model: its two `BlockInfo` Modular samples and three `non_zeros`
/// symbols. Small, because under the original single-context entropy model a
/// run of identical samples was nearly free. The cover search now reads this
/// through `CoverRateModel::fixed_bits` (whose `Legacy` arm is exactly these
/// constants); the Phase 32 regret harness still charges it directly.
pub(crate) const PER_VARBLOCK_BITS: f64 = 2.0;

/// The extra `metadata_bits` a non-DCT8x8 varblock pays under the legacy rate
/// model: its DctSelect sample breaks the all-zeros run G.2.4's default map
/// codes for free, twice. Measured on the original coder at ~8 bits per merged
/// block net; charged higher so a merge must be paid for by real coefficient
/// savings. Phase Q4's audit put the real figure at a few bits
/// (`CoverRateModel::Calibrated`).
pub(crate) const NON_DCT8X8_SIGNAL_BITS: f64 = 32.0;

/// The `metadata_bits` a non-baseline `HfMul` pays: like DctSelect, its `mul`
/// sample breaks a constant run in `BlockInfo`'s second row. Charged at the
/// same order as a DctSelect transition but lower — the lattice keeps the
/// values small and repetitive.
fn mul_signal_bits(hf_mul: HfMul, baseline: HfMul) -> f64 {
    if hf_mul == baseline { 0.0 } else { 8.0 }
}

/// What every varblock's `mul` sample is charged under the fine field
/// (Phase Q3): a flat estimate of a smooth plane's per-sample cost.
const FINE_MUL_SIGNAL_BITS: f64 = 2.0;

/// §4.3's objective `J = R + lambda * D + metadata_bits` for one square
/// transform candidate, in bits: the residual bit proxy, plus each channel's
/// squared reconstruction error at that channel's operating-point exchange
/// rate. Scored with neutral chroma-from-luma (block choice is dominated by
/// luma structure, and CfL is estimated once the map is fixed).
///
/// Without the distortion term the comparison is dishonest: a larger
/// transform's dequant matrix has finer steps at the same `global_scale`, so
/// it spends more bits to buy quality nobody asked for and a bits-only
/// comparison mistakes that for compaction.
/// Scores one channel's non-LLF cells of a `side x side` grid (top-left
/// `n x n` corner is LLF) via 4-cell SIMD batches
/// ([`HfQuantizer::choose_lane4`]), with a scalar
/// [`HfQuantizer::choose`]/[`HfQuantizer::reconstruct`] fallback for each
/// row segment's remainder (row lengths — `side - n` on LLF rows, `side`
/// elsewhere — are not always multiples of 4).
///
/// Phase-3 (outside-advice.md §3's "vectorize adjacent coefficients, not one
/// cell's four candidates"): accumulates `bits`/`weighted_sse` and checks
/// `cutoff` after *every* cell, in the same raster order the original
/// scalar loop always used, so this is bit-identical to that loop — see
/// `choose_lane4_is_bit_identical_to_four_scalar_choose_calls` in
/// `quantize.rs`. Checking `cutoff` at cell granularity (not batch
/// granularity) keeps pruning exactly as tight as before a `choose_lane4`
/// batch was ever computed.
///
/// `target_at`/`on_recon` are the two things that differ between Y (identity
/// target, writes `d_y_hf`) and X/B (CfL-neutral target, no sink) — see the
/// two call sites in [`block_cost_bounded`].
///
/// # Errors
///
/// As [`HfQuantizer::choose`].
#[allow(clippy::too_many_arguments)]
/// Samples priced per [`HfQuantizer::choose_cells`] call in
/// [`hf_residual_cost_bounded`]: large enough to amortise the kernel's setup,
/// small enough that a speculative block replayed in scalar order stays cheap.
const CFL_PRICE_CHUNK: usize = 64;

/// Longest row [`score_channel_lanes`] scores through one
/// [`HfQuantizer::choose_run`] call; longer rows (never produced by the cover
/// candidates, whose largest side is 32) use the scalar loop.
const MAX_SCORE_ROW: usize = 64;

/// The Phase 6.3 frequency weight for one cell, or exactly `1.0` when the
/// table is empty (the flat production objective).
#[inline]
fn cell_weight(freq: &[f32], cell: usize) -> f64 {
    freq.get(cell).map_or(1.0, |w| f64::from(*w))
}

/// Where [`score_channel_lanes`] reads one channel's targets from: the
/// coefficient plane itself (Y), or the CfL-neutral residual
/// `plane - k * d_y` against the luma reconstruction (X/B).
#[derive(Clone, Copy)]
enum LaneTargets<'a> {
    Direct(&'a [f32]),
    Residual {
        plane: &'a [f32],
        k: f32,
        d_y: &'a [f32],
    },
}

impl LaneTargets<'_> {
    /// The scalar reference for one cell: what the pre-Phase-40 `target_at`
    /// closures computed.
    #[inline]
    fn at(&self, cell: usize) -> f32 {
        match *self {
            Self::Direct(plane) => plane.get(cell).copied().unwrap_or(0.0),
            Self::Residual { plane, k, d_y } => {
                plane.get(cell).copied().unwrap_or(0.0) - k * d_y.get(cell).copied().unwrap_or(0.0)
            }
        }
    }

    /// The row `first..first + len` as a slice: borrowed straight from the
    /// plane for `Direct`, or computed cell by cell into `buf` (the same
    /// `plane - k * d_y` arithmetic as [`Self::at`], written as a zip so the
    /// compiler vectorises it) for `Residual`. `None` when the row is not
    /// fully inside every source, in which case the caller falls back to
    /// [`Self::at`].
    #[inline]
    fn row<'b>(&'b self, first: usize, len: usize, buf: &'b mut [f32]) -> Option<&'b [f32]> {
        match *self {
            Self::Direct(plane) => plane.get(first..first + len),
            Self::Residual { plane, k, d_y } => {
                let (src, dy, out) = (
                    plane.get(first..first + len)?,
                    d_y.get(first..first + len)?,
                    buf.get_mut(..len)?,
                );
                for ((slot, &c), &d) in out.iter_mut().zip(src.iter()).zip(dy.iter()) {
                    *slot = c - k * d;
                }
                Some(out)
            }
        }
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "internal plumbing: a lane-scoring bundle where the target source, \
              the reconstruction sink and the two accumulators are consumed by \
              exactly two callers (block_cost_bounded and the Phase 6.3 weight tests)"
)]
fn score_channel_lanes(
    hf_quant: &HfQuantizer,
    channel: usize,
    lambda: f64,
    to_sample_domain: f64,
    side: usize,
    n: usize,
    targets: LaneTargets<'_>,
    mut recon_sink: Option<&mut [f32]>,
    bits: &mut u64,
    weighted_sse: &mut f64,
    check: &impl Fn(u64, f64) -> bool,
    // Phase 6.3's per-cell frequency weight. Empty means the flat production
    // objective: `cell_weight` then returns exactly 1.0 and the arithmetic is
    // bit-identical to the pre-Phase-6 scorer.
    freq: &[f32],
) -> Result<bool> {
    // Phase 36: one run-kernel call per row (vector chunks plus a padded
    // final chunk). `choose_run` is cell-for-cell identical to
    // `choose`/`reconstruct`; when it reports that some cell of the row would
    // fail, the row falls through to the scalar loop below, which reproduces
    // the exact per-cell cutoff-before-error order.
    //
    // Phase 40: the cutoff is tested once per row rather than after every
    // cell. Both accumulators only ever grow (`bits` by saturating adds of
    // non-negative counts, `weighted_sse` by non-negative finite terms), and a
    // pruned candidate is discarded whole -- its partial sums and its
    // reconstruction scratch are never read again -- so a candidate that the
    // per-cell test would have pruned at cell k is still pruned at the end of
    // that row, and every survivor accumulates exactly the same additions in
    // the same order. Cover decisions are therefore unchanged; only the
    // per-cell branch is gone. Under the flat policy the per-cell weight is
    // exactly 1.0, so `(lambda * to_sample_domain) * err^2` is the same
    // product as `lambda * to_sample_domain * 1.0 * err^2`.
    let flat = freq.is_empty();
    let flat_weight = lambda * to_sample_domain;
    let mut targets_buf = [0.0f32; MAX_SCORE_ROW];
    let mut q_buf = [0i32; MAX_SCORE_ROW];
    let mut recon_buf = [0.0f32; MAX_SCORE_ROW];
    for row in 0..side {
        let col_start = if row < n { n } else { 0 };
        let row_base = row * side;
        let mut col = col_start;
        let run_len = side - col_start;
        if run_len <= MAX_SCORE_ROW {
            let first_cell = row_base + col_start;
            let batched = match (
                targets.row(first_cell, run_len, &mut targets_buf),
                q_buf.get_mut(..run_len),
            ) {
                (Some(row_targets), Some(qs)) => {
                    // The luma pass writes its reconstruction straight into
                    // the caller's sink; chroma uses a row scratch.
                    let recons: &mut [f32] = match recon_sink
                        .as_deref_mut()
                        .and_then(|sink| sink.get_mut(first_cell..first_cell + run_len))
                    {
                        Some(sink_row) => sink_row,
                        None => match recon_buf.get_mut(..run_len) {
                            Some(scratch) => scratch,
                            None => unreachable!("run_len <= MAX_SCORE_ROW was checked"),
                        },
                    };
                    if hf_quant
                        .choose_run(channel, first_cell, row_targets, qs, Some(recons))
                        .is_ok()
                    {
                        let mut row_bits = *bits;
                        let mut row_sse = *weighted_sse;
                        if flat {
                            for ((&q, &recon), &target) in
                                qs.iter().zip(recons.iter()).zip(row_targets.iter())
                            {
                                row_bits = row_bits.saturating_add(residual_bits(q));
                                row_sse += flat_weight * f64::from(recon - target).powi(2);
                            }
                        } else {
                            for (i, ((&q, &recon), &target)) in qs
                                .iter()
                                .zip(recons.iter())
                                .zip(row_targets.iter())
                                .enumerate()
                            {
                                row_bits = row_bits.saturating_add(residual_bits(q));
                                row_sse += lambda
                                    * to_sample_domain
                                    * cell_weight(freq, first_cell + i)
                                    * f64::from(recon - target).powi(2);
                            }
                        }
                        *bits = row_bits;
                        *weighted_sse = row_sse;
                        if check(*bits, *weighted_sse) {
                            return Ok(true);
                        }
                        true
                    } else {
                        false
                    }
                }
                _ => false,
            };
            if batched {
                col = side;
            }
        }
        while col < side {
            let cell = row_base + col;
            let target = targets.at(cell);
            let q = hf_quant.choose(target, channel, cell)?;
            let recon = hf_quant.reconstruct(q, channel, cell);
            *bits = bits.saturating_add(residual_bits(q));
            *weighted_sse += lambda
                * to_sample_domain
                * cell_weight(freq, cell)
                * f64::from(recon - target).powi(2);
            if let Some(slot) = recon_sink
                .as_deref_mut()
                .and_then(|sink| sink.get_mut(cell))
            {
                *slot = recon;
            }
            if check(*bits, *weighted_sse) {
                return Ok(true);
            }
            col += 1;
        }
    }
    Ok(false)
}

/// S8 Phase D (AKR source `outside-advice-2026-08-06` §8, feature
/// `s8-cover-prune`):
/// the staged cheap lower bound Phase C's `regret::validate_candidate_prune`
/// proved safe (zero violations, exhaustively checked over every
/// merge-candidate node a corpus fixture produced) — checked with the
/// `bits`/`weighted_sse` already accumulated from prior channels, *before*
/// this channel's exact `choose`/`choose_lane4` loop runs, never instead of
/// it. A survivor still runs the exact loop unchanged; this can only ever
/// return early where the exact loop would eventually have pruned too, so
/// it cannot change which candidate wins — only skip paying for cells whose
/// fate is already provably decided.
///
/// # Errors
///
/// As [`HfQuantizer::cell_lower_bound`].
#[cfg(feature = "s8-cover-prune")]
#[allow(clippy::too_many_arguments)]
fn cheap_stage_would_prune(
    hf_quant: &HfQuantizer,
    channel: usize,
    lambda: f64,
    to_sample_domain: f64,
    side: usize,
    n: usize,
    mut target_at: impl FnMut(usize) -> f32,
    bits_so_far: u64,
    weighted_sse_so_far: f64,
    cutoff: f64,
    freq: &[f32],
) -> Result<bool> {
    let mut bits = bits_so_far;
    let mut weighted_sse = weighted_sse_so_far;
    for row in 0..side {
        let col_start = if row < n { n } else { 0 };
        for col in col_start..side {
            let cell = row * side + col;
            let (b, s) = hf_quant.cell_lower_bound(target_at(cell), channel, cell)?;
            bits = bits.saturating_add(b);
            weighted_sse += lambda * to_sample_domain * cell_weight(freq, cell) * s;
        }
    }
    diagnostics::note_cover_prune_check();
    #[allow(
        clippy::cast_precision_loss,
        reason = "bit counts stay far inside f64's exact integer range"
    )]
    let partial = bits as f64 + weighted_sse;
    let would_prune = partial >= cutoff;
    if would_prune {
        diagnostics::note_cover_prune_hit();
    }
    Ok(would_prune)
}

/// Mutable lazy access for the existing group path, or immutable access after
/// [`ensure_cover_candidates_cached`] completed every square candidate.
enum CoverForwardBank<'a> {
    Lazy(&'a mut CandidateGroupBank),
    Complete {
        cache: &'a CandidateGroupBank,
        hits: &'a mut u64,
    },
}

impl CoverForwardBank<'_> {
    fn get(
        &mut self,
        frame: &PreparedFrame,
        transform: TransformType,
        px: u32,
        py: u32,
        scratch: &mut ForwardScratch,
    ) -> Result<VarblockForward<'_>> {
        match self {
            Self::Lazy(cache) => cache.get_or_insert(frame, transform, px, py, scratch),
            Self::Complete { cache, hits } => {
                let result = cache
                    .get(transform, px, py)
                    .ok_or(PolicyError::Unsupported {
                        what: "a completed cover cache missing a square candidate",
                    });
                if result.is_ok() {
                    **hits = hits.saturating_add(1);
                }
                result
            }
        }
    }
}

/// §4.3 objective for one square candidate. When `cutoff` is set, returns
/// `None` as soon as the partial cost cannot beat the split (ties keep split).
#[allow(clippy::too_many_arguments)]
fn block_cost_bounded(
    frame: &PreparedFrame,
    hf_quants: &HfQuantizers,
    transform: TransformType,
    hf_mul: HfMul,
    px: u32,
    py: u32,
    cache: &mut CoverForwardBank<'_>,
    scratch: &mut ForwardScratch,
    d_y_hf: &mut [f32],
    cutoff: Option<f64>,
) -> Result<Option<f64>> {
    // S8 Phase B: measures the choose-loop's actual cost share of cover
    // scoring against the forward-DCT/cache share, before building any
    // summary-scoring machinery on the assumption that share is large.
    let fwd = diagnostics::time_stage(diagnostics::StageTimer::CoverForward, || {
        cache.get(frame, transform, px, py, scratch)
    })?;
    let side = transform.sample_cols();
    let n = transform.block_dims().0;
    let hf_quant = hf_quants.get(transform, hf_mul)?;
    // One squared coefficient unit is `side^2` squared sample units (the
    // forward transforms are not Parseval; see [`HfQuantizers::lambda`]).
    //
    // Phase 6.2 measured that `side^2` alone does *not* make candidates of
    // different sizes comparable: at equal sample-domain error a DCT32x32
    // basis costs 4-21% more butteraugli than DCT8x8 bases, because error on a
    // large support is spatially coherent where the same energy in sixteen
    // independently-signed 8x8 patches is closer to maskable noise. The
    // multiplier corrects for that; it is exactly 1.0 under the neutral
    // production policy, so the shipped objective is unchanged bit for bit.
    #[allow(
        clippy::cast_precision_loss,
        reason = "side is at most 32; exact in f64"
    )]
    let to_sample_domain = (side * side) as f64 * hf_quants.size_penalty(transform);
    // Phase 6.3: empty under the flat production policy, and `cell_weight`
    // then returns exactly 1.0, so the shipped objective is untouched.
    let freq = hf_quants.frequency_weights(transform);
    // Phase Q4: exactly 1.0 under the legacy rate model, so the shipped
    // objective -- and its pruning arithmetic -- is bit-identical.
    let rate_scale = hf_quants.rate_scale(transform);
    let [cx, cy, cb] = fwd.coeffs;
    let mut bits = 0u64;
    let mut weighted_sse = 0.0f64;
    let check = |bits: u64, weighted_sse: f64| -> bool {
        if let Some(cut) = cutoff {
            #[allow(
                clippy::cast_precision_loss,
                reason = "bit counts stay far inside f64's exact integer range"
            )]
            let partial = bits as f64 * rate_scale + weighted_sse;
            // Ties keep the split, so >= is a correct prune.
            partial >= cut
        } else {
            false
        }
    };
    #[cfg(feature = "s8-cover-prune")]
    if let Some(cut) = cutoff {
        let would_prune = diagnostics::time_stage(diagnostics::StageTimer::CoverPrune, || {
            cheap_stage_would_prune(
                hf_quant,
                1,
                hf_quants.lambda[1],
                to_sample_domain,
                side,
                n,
                |cell| cy.get(cell).copied().unwrap_or(0.0),
                bits,
                weighted_sse,
                cut,
                freq,
            )
        })?;
        if would_prune {
            return Ok(None);
        }
    }
    let pruned = diagnostics::time_stage(diagnostics::StageTimer::CoverScore, || {
        score_channel_lanes(
            hf_quant,
            1,
            hf_quants.lambda[1],
            to_sample_domain,
            side,
            n,
            LaneTargets::Direct(cy),
            Some(&mut *d_y_hf),
            &mut bits,
            &mut weighted_sse,
            &check,
            freq,
        )
    })?;
    if pruned {
        return Ok(None);
    }
    for &(channel, plane) in &[(0usize, cx), (2usize, cb)] {
        let k = if channel == 0 { 0.0 } else { 1.0 };
        #[cfg(feature = "s8-cover-prune")]
        if let Some(cut) = cutoff {
            let would_prune = diagnostics::time_stage(diagnostics::StageTimer::CoverPrune, || {
                cheap_stage_would_prune(
                    hf_quant,
                    channel,
                    hf_quants.lambda.get(channel).copied().unwrap_or(0.0),
                    to_sample_domain,
                    side,
                    n,
                    |cell| {
                        plane.get(cell).copied().unwrap_or(0.0)
                            - k * d_y_hf.get(cell).copied().unwrap_or(0.0)
                    },
                    bits,
                    weighted_sse,
                    cut,
                    freq,
                )
            })?;
            if would_prune {
                return Ok(None);
            }
        }
        let pruned = diagnostics::time_stage(diagnostics::StageTimer::CoverScore, || {
            score_channel_lanes(
                hf_quant,
                channel,
                hf_quants.lambda.get(channel).copied().unwrap_or(0.0),
                to_sample_domain,
                side,
                n,
                LaneTargets::Residual {
                    plane,
                    k,
                    d_y: d_y_hf,
                },
                None,
                &mut bits,
                &mut weighted_sse,
                &check,
                freq,
            )
        })?;
        if pruned {
            return Ok(None);
        }
    }
    #[allow(
        clippy::cast_precision_loss,
        reason = "bit counts stay far inside f64's exact integer range"
    )]
    Ok(Some(bits as f64 * rate_scale + weighted_sse))
}

#[allow(clippy::too_many_arguments)]
fn block_cost_with(
    frame: &PreparedFrame,
    hf_quants: &HfQuantizers,
    transform: TransformType,
    hf_mul: HfMul,
    px: u32,
    py: u32,
    cache: &mut CoverForwardBank<'_>,
    scratch: &mut ForwardScratch,
    d_y_hf: &mut [f32],
) -> Result<f64> {
    block_cost_bounded(
        frame, hf_quants, transform, hf_mul, px, py, cache, scratch, d_y_hf, None,
    )?
    .ok_or(PolicyError::Unsupported {
        what: "an unbounded block cost that returned None",
    })
}

#[allow(clippy::too_many_arguments)]
fn block_cost(
    frame: &PreparedFrame,
    hf_quants: &HfQuantizers,
    transform: TransformType,
    hf_mul: HfMul,
    px: u32,
    py: u32,
    cache: &mut CandidateGroupBank,
    scratch: &mut ForwardScratch,
    d_y_hf: &mut [f32],
) -> Result<f64> {
    let mut access = CoverForwardBank::Lazy(cache);
    block_cost_with(
        frame,
        hf_quants,
        transform,
        hf_mul,
        px,
        py,
        &mut access,
        scratch,
        d_y_hf,
    )
}

/// The quadtree cover of one aligned `size`-atom region, choosing at each level
/// between one square transform and four sub-quadrants by exact R-D within the
/// hierarchy. Clipped and pass-group-straddling regions are forced to split;
/// aligned placement keeps every block inside one pass group.
#[allow(clippy::too_many_arguments)]
fn tile_region_with(
    frame: &PreparedFrame,
    hf_quants: &HfQuantizers,
    grid: jpxl_encode::vardct::BlockGrid,
    bx: u32,
    by: u32,
    size: u32,
    aq: &AqSetup,
    x0: u32,
    y0: u32,
    cache: &mut CoverForwardBank<'_>,
    scratch: &mut ForwardScratch,
    d_y_hf: &mut [f32],
) -> Result<(f64, Vec<VarblockDecision>)> {
    if bx >= grid.width || by >= grid.height {
        return Ok((0.0, Vec::new()));
    }
    // A candidate's `HfMul` comes from the adaptive-quantization field over
    // its own footprint (the baseline with the field off): the solver chooses
    // the *transform* at the field's quantizer, it does not second-guess the
    // field. Atom coordinates are frame-global: the group origin is a pixel
    // rect, so `x0 / 8` is the group's first atom column.
    if size == 1 {
        let hf_mul = aq.mul_for_footprint(x0 / 8 + bx, y0 / 8 + by, 1, 1);
        let cost = block_cost_with(
            frame,
            hf_quants,
            TransformType::Dct8x8,
            hf_mul,
            x0 + bx * 8,
            y0 + by * 8,
            cache,
            scratch,
            d_y_hf,
        )? + hf_quants.rate_fixed_bits(TransformType::Dct8x8)
            + aq.mul_signal_bits(hf_mul);
        return Ok((
            cost,
            vec![VarblockDecision {
                origin: LfBlockPos::new(bx, by),
                transform: TransformType::Dct8x8,
                hf_mul,
            }],
        ));
    }

    let half = size / 2;
    let mut split_cost = 0.0f64;
    let mut split_blocks = Vec::new();
    for (qx, qy) in [
        (bx, by),
        (bx + half, by),
        (bx, by + half),
        (bx + half, by + half),
    ] {
        let (c, mut b) = tile_region_with(
            frame, hf_quants, grid, qx, qy, half, aq, x0, y0, cache, scratch, d_y_hf,
        )?;
        split_cost += c;
        split_blocks.append(&mut b);
    }

    let fits = bx + size <= grid.width && by + size <= grid.height;
    let single = match square_transform(size) {
        Some(transform) if fits => {
            let hf_mul = aq.mul_for_footprint(x0 / 8 + bx, y0 / 8 + by, size, size);
            let fixed = hf_quants.rate_fixed_bits(transform) + aq.mul_signal_bits(hf_mul);
            // Fixed metadata alone can already lose to the split.
            if fixed >= split_cost {
                return Ok((split_cost, split_blocks));
            }
            // Phase-1: prune when partial R-D already loses to the split.
            let cutoff = split_cost - fixed;
            match block_cost_bounded(
                frame,
                hf_quants,
                transform,
                hf_mul,
                x0 + bx * 8,
                y0 + by * 8,
                cache,
                scratch,
                d_y_hf,
                Some(cutoff),
            )? {
                Some(rd) => {
                    let cost = rd + fixed;
                    Some((
                        cost,
                        vec![VarblockDecision {
                            origin: LfBlockPos::new(bx, by),
                            transform,
                            hf_mul,
                        }],
                    ))
                }
                None => None,
            }
        }
        _ => None,
    };

    // Ties keep the split: four DCT8x8s are the cheaper signal and the safer
    // reconstruction, so a merge must strictly earn its place.
    Ok(match single {
        Some(single) if single.0 < split_cost => single,
        _ => (split_cost, split_blocks),
    })
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn tile_region(
    frame: &PreparedFrame,
    hf_quants: &HfQuantizers,
    grid: jpxl_encode::vardct::BlockGrid,
    bx: u32,
    by: u32,
    size: u32,
    aq: &AqSetup,
    x0: u32,
    y0: u32,
    cache: &mut CandidateGroupBank,
    scratch: &mut ForwardScratch,
    d_y_hf: &mut [f32],
) -> Result<(f64, Vec<VarblockDecision>)> {
    let mut access = CoverForwardBank::Lazy(cache);
    tile_region_with(
        frame,
        hf_quants,
        grid,
        bx,
        by,
        size,
        aq,
        x0,
        y0,
        &mut access,
        scratch,
        d_y_hf,
    )
}

/// Selects one LF group's varblock tiling by the hierarchical quadtree solver,
/// returned in `BlockInfo` (greedy earliest-uncovered raster) order.
fn select_blocks(
    frame: &PreparedFrame,
    hf_quants: &HfQuantizers,
    grid: jpxl_encode::vardct::BlockGrid,
    origin: (u32, u32),
    aq: &AqSetup,
    cache: &mut CandidateGroupBank,
    scratch: &mut ForwardScratch,
) -> Result<Vec<VarblockDecision>> {
    let (x0, y0) = origin;
    let mut cache = CoverForwardBank::Lazy(cache);
    let mut d_y_hf = vec![0.0f32; 32 * 32];
    let mut blocks = Vec::new();
    let mut sby = 0u32;
    while sby < grid.height {
        let mut sbx = 0u32;
        while sbx < grid.width {
            let (_, mut region) = tile_region_with(
                frame,
                hf_quants,
                grid,
                sbx,
                sby,
                4,
                aq,
                x0,
                y0,
                &mut cache,
                scratch,
                &mut d_y_hf,
            )?;
            blocks.append(&mut region);
            sbx += 4;
        }
        sby += 4;
    }
    // G.2.4's greedy walk places each varblock at the earliest uncovered atom
    // in raster order; a quadtree's blocks, sorted by their top-left atom's
    // raster index, are exactly that sequence (each origin is the minimum
    // raster atom of its footprint, and the cover is exact and non-overlapping).
    blocks.sort_by_key(|b| (b.origin.by(), b.origin.bx()));
    Ok(blocks)
}

/// Scores the independent aligned 4x4-atom trees of one LF group through a
/// completed immutable forward cache. Fixed-index reduction followed by the
/// same raster sort as [`select_blocks`] preserves Contract A.
fn select_blocks_cached_parallel(
    frame: &PreparedFrame,
    hf_quants: &HfQuantizers,
    grid: jpxl_encode::vardct::BlockGrid,
    origin: (u32, u32),
    aq: &AqSetup,
    cache: &CandidateGroupBank,
    executor: &jpxl_encode::EncodeExecutor,
) -> Result<Vec<VarblockDecision>> {
    let (x0, y0) = origin;
    let regions_w = usize::try_from(grid.width.div_ceil(4)).unwrap_or(0);
    let regions_h = usize::try_from(grid.height.div_ceil(4)).unwrap_or(0);
    let regions = regions_w.saturating_mul(regions_h);
    let planned = executor.map_ordered(regions, |index| {
        let rx = index % regions_w.max(1);
        let ry = index / regions_w.max(1);
        let bx = u32::try_from(rx).unwrap_or(u32::MAX).saturating_mul(4);
        let by = u32::try_from(ry).unwrap_or(u32::MAX).saturating_mul(4);
        let mut cache_hits = 0u64;
        // The completed cache path never enters `get_or_insert`, so the
        // transform/sample/coefficient buffers used by the lazy path would
        // only be per-region allocation overhead. Keep the scratch marker for
        // the shared scorer's signature, but give it no backing storage.
        let mut scratch = ForwardScratch::empty();
        let mut d_y_hf = vec![0.0f32; 32 * 32];
        let blocks = {
            let mut access = CoverForwardBank::Complete {
                cache,
                hits: &mut cache_hits,
            };
            let (_, blocks) = tile_region_with(
                frame,
                hf_quants,
                grid,
                bx,
                by,
                4,
                aq,
                x0,
                y0,
                &mut access,
                &mut scratch,
                &mut d_y_hf,
            )?;
            blocks
        };
        Ok::<_, PolicyError>((cache_hits, blocks))
    })?;
    let mut cache_hits = 0u64;
    let mut blocks = Vec::new();
    for (hits, mut region) in planned {
        cache_hits = cache_hits.saturating_add(hits);
        blocks.append(&mut region);
    }
    cache
        .complete_hits
        .fetch_add(cache_hits, std::sync::atomic::Ordering::Relaxed);
    blocks.sort_by_key(|b| (b.origin.by(), b.origin.bx()));
    Ok(blocks)
}

/// The trained model as a full [`EntropyPlan`]: the caller's block-context
/// model, preset count / assignment, the chosen orders, and the trainer's
/// map / configurations / distributions.
fn trained_entropy_plan(
    geometry: &VardctGeometry,
    model: entropy::TrainedModel,
    orders: OrderSet,
    block_context: HfBlockContextPlan,
    num_hf_presets: u32,
    group_presets: Vec<PresetId>,
) -> Result<EntropyPlan> {
    let num_groups = usize::try_from(geometry.num_groups()).unwrap_or(0);
    let mut presets = group_presets;
    if presets.len() != num_groups {
        presets = vec![PresetId::new(0); num_groups];
    }
    let distributions = EntropyModelPlan {
        context_map: model.context_map.into_boxed_slice(),
        histograms: model.histograms.into_boxed_slice(),
        hybrid_uint: model.hybrid_uint.into_boxed_slice(),
    };
    let pass = HfPassEntropyPlan {
        orders,
        distributions,
        group_presets: presets.into_boxed_slice(),
    };
    Ok(EntropyPlan {
        block_context,
        num_hf_presets,
        passes: vec![pass].into_boxed_slice(),
    })
}

/// The provisional entropy model the census pass walks with: the given block
/// context, natural orders, one preset, and [`cluster_of`]'s six clusters.
/// Nothing it carries reaches the wire — the trained model replaces it before
/// validation. The block context *does* affect the walk's pre-context ids, so
/// it must match the model the trainer will later attach.
fn entropy_plan(
    geometry: &VardctGeometry,
    histograms: Vec<HistogramPlan>,
    block_context: HfBlockContextPlan,
) -> Result<EntropyPlan> {
    let nb_block_ctx = block_context.nb_block_ctx();
    let num_hf_presets = 1u32;
    let pre_contexts = 495 * u64::from(num_hf_presets) * nb_block_ctx;
    let num_groups = usize::try_from(geometry.num_groups()).unwrap_or(0);

    let context_map: Vec<ClusterId> = (0..pre_contexts)
        .map(|ctx| ClusterId::new(cluster_of(ctx, nb_block_ctx)))
        .collect();
    // C.2.3: `split_exponent = 4` puts every value below 16 in its own token,
    // which is where the overwhelming majority of quantized coefficients live,
    // and `msb_in_token = 2` keeps the leading bits of the tail in the token
    // rather than in raw bits. One configuration for every cluster until the
    // trainer replaces these with per-cluster choices.
    let hybrid_uint = vec![
        HybridUintPlan {
            split_exponent: 4,
            msb_in_token: 2,
            lsb_in_token: 0,
        };
        histograms.len()
    ];

    let distributions = EntropyModelPlan {
        context_map: context_map.into_boxed_slice(),
        histograms: histograms.into_boxed_slice(),
        hybrid_uint: hybrid_uint.into_boxed_slice(),
    };
    let pass = HfPassEntropyPlan {
        orders: OrderSet::natural(),
        distributions,
        group_presets: vec![PresetId::new(0); num_groups].into_boxed_slice(),
    };
    Ok(EntropyPlan {
        block_context,
        num_hf_presets,
        passes: vec![pass].into_boxed_slice(),
    })
}

/// Six histograms with one symbol each, enough to build a provisional plan.
///
/// The provisional plan exists only to be walked; nothing it carries reaches
/// the wire, and the real histograms replace these before validation.
fn placeholder_histograms() -> Vec<HistogramPlan> {
    (0..NUM_CLUSTERS)
        .filter_map(|_| HistogramPlan::new(vec![1u32]).ok())
        .collect()
}

/// The clustering: which of six distributions an I.4 context uses.
///
/// A single distribution over all 7425 pre-contexts would be legal and much
/// worse; 7425 distributions is not expressible (C.2.2 caps clusters at 256)
/// and would cost more in serialized histograms than it saved. The split below
/// is the cheapest one that separates the three genuinely different symbol
/// populations:
///
/// * **`non_zeros` counts** (I.4's first 37 contexts per block context) are
///   small integers with a completely different shape from coefficients;
/// * **`prev`**, the low bit of a coefficient context, says whether the
///   previous coefficient was non-zero, and is the strongest single predictor
///   of whether this one is;
/// * **luma versus chroma**, i.e. whether the block context is the default
///   map's Y row, because the two are quantized an order of magnitude apart.
///
/// Searching a clustering — `Encoder-plan1.md` §9.3 — is milestone 8. This is
/// a fixed rule, stated so that milestone 8 has a baseline to beat.
pub fn cluster_of(ctx: u64, nb_block_ctx: u64) -> u8 {
    let split = NON_ZEROS_CONTEXTS * nb_block_ctx;
    if ctx < split {
        return u8::from(!ctx.is_multiple_of(nb_block_ctx.max(1)));
    }
    let rel = ctx - split;
    let block_ctx = rel / COEFFICIENT_CONTEXTS;
    let prev = rel % COEFFICIENT_CONTEXTS % 2;
    let chroma = u64::from(block_ctx != 0);
    u8::try_from(2 + chroma * 2 + prev).unwrap_or(2)
}

/// Encodes an 8-bit sRGB image as a naked kVarDCT codestream.
///
/// The whole slice in one call: sRGB to XYB, plan, validate, emit. With a
/// [`RateTarget`] on the request this is [`encode_srgb8_to_target`] with the
/// report thrown away — and it returns the very bytes the loop priced, not a
/// re-encode of them.
///
/// # Errors
///
/// As [`plan_frame`], plus [`PolicyError::Encode`] if the writer refuses the
/// plan.
pub fn encode_srgb8_vardct(
    width: u32,
    height: u32,
    rgb: &[u8],
    request: &EncodeRequest,
) -> Result<Vec<u8>> {
    if let Some(target) = request.target {
        return Ok(encode_srgb8_to_target(width, height, rgb, request, target)?.codestream);
    }
    let frame = PreparedFrame::from_srgb8(width, height, rgb)?;
    let plan = plan_frame(&frame, request)?;
    Ok(jpxl_encode::vardct::write_codestream_with(
        &plan,
        request.resources,
    )?)
}

/// As [`encode_srgb8_vardct`], overriding [`EncodeRequest::resources`].
///
/// # Errors
///
/// As [`encode_srgb8_vardct`].
pub fn encode_srgb8_vardct_with_resources(
    width: u32,
    height: u32,
    rgb: &[u8],
    request: &EncodeRequest,
    resources: jpxl_encode::EncodeResources,
) -> Result<Vec<u8>> {
    let mut req = *request;
    req.resources = resources;
    encode_srgb8_vardct(width, height, rgb, &req)
}

/// Encodes an 8-bit sRGB image to a byte or bits-per-pixel target.
///
/// Returns the chosen quantizer, the exact achieved size, the per-section
/// accounting and the full iteration trace — the last of which is the loop's
/// evidence, and what the tests and later milestones' telemetry read.
///
/// # Errors
///
/// As [`plan_frame`], plus [`PolicyError::TargetUnreachable`] if the target is
/// below what any representable quantizer can produce.
pub fn encode_srgb8_to_target(
    width: u32,
    height: u32,
    rgb: &[u8],
    request: &EncodeRequest,
    target: RateTarget,
) -> Result<RateOutcome> {
    let mut resolved = *request;
    resolved.b_qm_scale = request.effective_b_qm_scale();
    resolved.x_qm_scale = request.effective_x_qm_scale();
    // Phase 38: one worker pool for the whole encode; the source conversion
    // uses it too instead of running on the calling thread alone.
    let executor = resolved.resources.executor();
    let frame = PreparedFrame::from_srgb8_with(width, height, rgb, Some(&executor))?;
    let atlas = AnalysisAtlas::analyze(&frame);
    rate::search_frame_with_executor(&frame, &atlas, &resolved, target, &executor)
}

/// Encodes a high-precision sRGB image to a byte or bits-per-pixel target.
///
/// [`encode_srgb8_to_target`] for a source that carries more than eight bits
/// per sample: `bits_per_sample` is the source's own depth in `1..=16`, used
/// both to normalise the samples ([`PreparedFrame::from_srgb16`]) and as the
/// `bit_depth` written into `ImageMetadata`, so a decoder reconstructs at the
/// original precision instead of quantizing to 8-bit.
///
/// The VarDCT path itself is float XYB either way; what this changes is that
/// nothing throws the extra precision away on the way in or on the way out.
///
/// # Errors
///
/// As [`encode_srgb8_to_target`], plus [`PolicyError::Unsupported`] for a
/// `bits_per_sample` outside `1..=16`.
pub fn encode_srgb16_to_target(
    width: u32,
    height: u32,
    rgb: &[u16],
    bits_per_sample: u32,
    request: &EncodeRequest,
    target: RateTarget,
) -> Result<RateOutcome> {
    let mut resolved = *request;
    resolved.bits_per_sample = bits_per_sample;
    resolved.b_qm_scale = request.effective_b_qm_scale();
    resolved.x_qm_scale = request.effective_x_qm_scale();
    let executor = resolved.resources.executor();
    let frame =
        PreparedFrame::from_srgb16_with(width, height, rgb, bits_per_sample, Some(&executor))?;
    let atlas = AnalysisAtlas::analyze(&frame);
    rate::search_frame_with_executor(&frame, &atlas, &resolved, target, &executor)
}

/// [`encode_srgb16_to_target`] without a rate target: the request's own
/// quantizer scalars are emitted exactly as given.
///
/// # Errors
///
/// As [`encode_srgb16_to_target`].
pub fn encode_srgb16_vardct(
    width: u32,
    height: u32,
    rgb: &[u16],
    bits_per_sample: u32,
    request: &EncodeRequest,
) -> Result<Vec<u8>> {
    if let Some(target) = request.target {
        return Ok(
            encode_srgb16_to_target(width, height, rgb, bits_per_sample, request, target)?
                .codestream,
        );
    }
    let mut resolved = *request;
    resolved.bits_per_sample = bits_per_sample;
    let frame = PreparedFrame::from_srgb16(width, height, rgb, bits_per_sample)?;
    let plan = plan_frame(&frame, &resolved)?;
    Ok(jpxl_encode::vardct::write_codestream_with(
        &plan,
        resolved.resources,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jpxl_core::limits::Limits;
    use jpxl_decode::decode::decode;

    fn grey_frame(width: u32, height: u32) -> PreparedFrame {
        let n = (width * height) as usize;
        PreparedFrame::from_linear_srgb(width, height, vec![0.4; n], vec![0.4; n], vec![0.4; n])
            .expect("legal frame")
    }

    /// Opt-P Contract A: multi-group VarDCT emission is byte-identical at
    /// serial, fixed-N, and host-auto worker budgets.
    #[test]
    fn multi_group_vardct_is_byte_identical_across_thread_counts() {
        let (width, height) = (300u32, 260u32); // >256 → multi pass-group
        let rgb = synthetic_rgb(width, height, false);
        let request = EncodeRequest::defaults();
        let serial = encode_srgb8_vardct_with_resources(
            width,
            height,
            &rgb,
            &request,
            jpxl_encode::EncodeResources::serial(),
        )
        .expect("serial");
        let parallel = encode_srgb8_vardct_with_resources(
            width,
            height,
            &rgb,
            &request,
            jpxl_encode::EncodeResources::groups(4),
        )
        .expect("parallel");
        // Default request uses EncodeResources::auto() — must match serial.
        let auto = encode_srgb8_vardct(width, height, &rgb, &request).expect("auto");
        assert_eq!(
            serial, parallel,
            "Contract A: 1-thread and 4-thread VarDCT multi-group must match"
        );
        assert_eq!(
            serial, auto,
            "Contract A: EncodeResources::auto must match serial emission"
        );
        let frame = PreparedFrame::from_srgb8(width, height, &rgb).expect("frame");
        let plan = plan_frame(&frame, &request).expect("plan");
        let serial_executor = jpxl_encode::EncodeResources::serial().executor();
        let parallel_executor = jpxl_encode::EncodeResources::groups(4).executor();
        let serial_sizing =
            jpxl_encode::vardct::price_codestream_with(&plan, &serial_executor).expect("prices");
        let parallel_sizing =
            jpxl_encode::vardct::price_codestream_with(&plan, &parallel_executor).expect("prices");
        assert_eq!(
            serial_sizing, parallel_sizing,
            "Contract A: Count sizing must not depend on executor width"
        );
        let image = decode(&serial, &Limits::default()).expect("decodes");
        assert_eq!((image.width, image.height), (width, height));
    }

    /// A 16-bit version of [`synthetic_rgb`]: the same field, but resolved at
    /// the full 0..=65535 range so the extra precision is real rather than an
    /// 8-bit image in a wide container.
    fn synthetic_rgb16(width: u32, height: u32) -> Vec<u16> {
        let mut out = Vec::with_capacity(
            usize::try_from(u64::from(width) * u64::from(height) * 3).unwrap_or(0),
        );
        for y in 0..height {
            for x in 0..width {
                let ramp = u32::from(u16::MAX) * x / width.max(1) / 2
                    + u32::from(u16::MAX) * y / height.max(1) / 4;
                let luma = u16::try_from(ramp.min(u32::from(u16::MAX))).unwrap_or(u16::MAX);
                out.extend_from_slice(&[
                    luma,
                    luma.saturating_sub(4_096),
                    luma.saturating_add(8_192),
                ]);
            }
        }
        out
    }

    /// The high-precision lossy path must declare — and decode back at — the
    /// source's own bit depth. Without this the whole point of feeding 16-bit
    /// samples in is lost at the last step, when the decoder quantizes the
    /// reconstructed floats to Table D.3's default 8 bits.
    #[test]
    fn a_16_bit_source_round_trips_at_16_bit_precision() {
        let (width, height) = (96u32, 72u32);
        let rgb = synthetic_rgb16(width, height);
        let request = EncodeRequest::for_target(RateTarget::BitsPerPixel(2.0));
        let outcome = encode_srgb16_to_target(
            width,
            height,
            &rgb,
            16,
            &request,
            RateTarget::BitsPerPixel(2.0),
        )
        .expect("a 16-bit lossy encode");

        let decoded = decode(&outcome.codestream, &Limits::default()).expect("decodes");
        assert_eq!((decoded.width, decoded.height), (width, height));
        assert_eq!(decoded.num_colour_channels, 3);
        assert_eq!(
            decoded.colour_bits_per_sample(),
            16,
            "the source depth must survive into the decoded planes"
        );
        // A kVarDCT frame is float XYB; the float planes are what proves the
        // extra precision is actually carried rather than merely declared.
        assert!(
            decoded.float_planes.is_some(),
            "a VarDCT frame reconstructs to floats"
        );
    }

    /// The declared depth must not silently change what the encoder searches:
    /// an 8-bit source fed through the wide entry point is the same image, so
    /// only the declared precision may differ.
    #[test]
    fn the_wide_entry_point_agrees_with_the_8_bit_one_on_8_bit_input() {
        let (width, height) = (64u32, 48u32);
        let rgb8 = synthetic_rgb(width, height, false);
        let widened: Vec<u16> = rgb8.iter().map(|&s| u16::from(s)).collect();
        let target = RateTarget::BitsPerPixel(1.5);
        let request = EncodeRequest::for_target(target);

        let narrow = encode_srgb8_to_target(width, height, &rgb8, &request, target)
            .expect("8-bit encode")
            .codestream;
        let wide = encode_srgb16_to_target(width, height, &widened, 8, &request, target)
            .expect("8-bit-through-wide encode")
            .codestream;
        assert_eq!(
            narrow, wide,
            "the same samples at the same declared depth must emit the same bytes"
        );
    }

    fn synthetic_rgb(width: u32, height: u32, grayscale: bool) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            usize::try_from(u64::from(width) * u64::from(height) * 3).unwrap_or(0),
        );
        for y in 0..height {
            for x in 0..width {
                let ramp = (x * 170 / width.max(1)) + (y * 70 / height.max(1));
                let checker = if (x / 16 + y / 16).is_multiple_of(2) {
                    12
                } else {
                    0
                };
                let luma = u8::try_from((20 + ramp + checker).min(255)).unwrap_or(255);
                if grayscale {
                    out.extend_from_slice(&[luma, luma, luma]);
                } else {
                    // A fixed chromaticity under a natural-photo-like mixture
                    // of broad gradients and low-frequency texture: all three
                    // XYB channels vary with the same luminance field, which is
                    // exactly the content CfL should compact.
                    out.extend_from_slice(&[
                        luma,
                        u8::try_from(u16::from(luma) * 4 / 5).unwrap_or(255),
                        u8::try_from(u16::from(luma) * 3 / 5).unwrap_or(255),
                    ]);
                }
            }
        }
        out
    }

    fn encode_with_cfl(frame: &PreparedFrame, enabled: bool) -> Vec<u8> {
        let request = EncodeRequest::defaults();
        let atlas = AnalysisAtlas::analyze(frame);
        let mut cache = CandidateForwardCache::new();
        let plan = plan_at_with_cfl(
            frame,
            &atlas,
            &request,
            QuantizerChoice::from_request(&request),
            enabled,
            None,
            &mut cache,
            EntropySearch::Full,
            None,
            AnchorReuse::None,
            None,
        )
        .expect("a legal plan");
        jpxl_encode::vardct::write_codestream(&plan).expect("encodes")
    }

    fn decoded_rgb(bytes: &[u8]) -> Vec<u8> {
        let image = decode(bytes, &Limits::default()).expect("decodes");
        let count = usize::try_from(u64::from(image.width) * u64::from(image.height)).unwrap_or(0);
        let mut out = Vec::with_capacity(count * 3);
        for i in 0..count {
            for plane in image.planes.iter().take(3) {
                let sample = plane.samples.get(i).copied().unwrap_or(0);
                out.push(u8::try_from(sample.clamp(0, 255)).unwrap_or(0));
            }
        }
        out
    }

    fn rmse(a: &[u8], b: &[u8]) -> f64 {
        assert_eq!(a.len(), b.len());
        let sum = a.iter().zip(b).fold(0.0, |sum, (&x, &y)| {
            let error = f64::from(x) - f64::from(y);
            sum + error * error
        });
        sum / a.len().max(1) as f64
    }

    #[test]
    fn quantization_workspace_reuses_only_unshared_arenas() {
        let mut workspace = QuantizationWorkspace::new();
        let first = workspace.take_arena(0, 32);
        let first_ptr = Arc::as_ptr(&first);
        workspace.put_arena(0, first);

        let reused = workspace.take_arena(0, 16);
        assert_eq!(Arc::as_ptr(&reused), first_ptr);

        let held_by_plan = Arc::clone(&reused);
        workspace.put_arena(0, reused);
        let replacement = workspace.take_arena(0, 16);
        assert_ne!(Arc::as_ptr(&replacement), first_ptr);
        drop(held_by_plan);
    }

    #[test]
    fn a_planned_frame_validates() {
        let frame = grey_frame(300, 200);
        let plan = plan_frame(&frame, &EncodeRequest::defaults()).expect("a legal plan");
        let dump = jpxl_encode::vardct::dump(&plan).expect("dumps");
        assert!(dump.contains("frame 300x200"));
        assert!(dump.contains("DctSelect 0: "));
    }

    #[test]
    fn a_multi_group_frame_plans_every_lf_group_and_section() {
        // 600x520 at group_size_shift 1 is a 3x3 pass-group grid inside one
        // 2048x2048 LF group: the multi-section path, not the single-section
        // shortcut.
        let frame = grey_frame(600, 520);
        let plan = plan_frame(&frame, &EncodeRequest::defaults()).expect("a legal plan");
        let geometry = plan.geometry().expect("geometry");
        assert_eq!(geometry.num_groups(), 9);
        assert_eq!(geometry.num_lf_groups(), 1);
        assert_eq!(plan.plan().sections.kinds.len() as u64, 2 + 1 + 9);
        let group = plan.plan().spatial.lf_groups.first().expect("one group");
        // 75x65 atoms, one DCT8x8 each.
        assert_eq!(group.nb_blocks(), 75 * 65);
    }

    #[test]
    fn a_frame_spanning_several_lf_groups_plans_each_one() {
        // 5000x3000 at shift 1 spans 3x2 LF groups of 2048x2048.
        let frame = grey_frame(4200, 40);
        let plan = plan_frame(&frame, &EncodeRequest::defaults()).expect("a legal plan");
        let geometry = plan.geometry().expect("geometry");
        assert_eq!(geometry.num_lf_groups(), 3);
        assert_eq!(plan.plan().spatial.lf_groups.len(), 3);
        assert_eq!(plan.plan().quantized.lf_groups.len(), 3);
        // The last LF group is clipped: 4200 - 4096 = 104 samples wide.
        let last = plan.plan().spatial.lf_groups.get(2).expect("third group");
        assert_eq!(last.nb_blocks(), 13 * 5);
    }

    #[test]
    fn grayscale_is_byte_identical_to_the_neutral_cfl_baseline() {
        let rgb = synthetic_rgb(256, 256, true);
        let frame = PreparedFrame::from_srgb8(256, 256, &rgb).expect("frame");
        let enabled = encode_with_cfl(&frame, true);
        let neutral = encode_with_cfl(&frame, false);
        assert_eq!(
            enabled, neutral,
            "CfL estimation must be an exact no-op for grayscale"
        );
    }

    #[test]
    fn cfl_reduces_size_at_equal_quality_on_correlated_colour() {
        let rgb = synthetic_rgb(256, 256, false);
        let frame = PreparedFrame::from_srgb8(256, 256, &rgb).expect("frame");
        let request = EncodeRequest::defaults();
        let atlas = AnalysisAtlas::analyze(&frame);
        let mut cache = CandidateForwardCache::new();
        let plan = plan_at_with_cfl(
            &frame,
            &atlas,
            &request,
            QuantizerChoice::from_request(&request),
            true,
            None,
            &mut cache,
            EntropySearch::Full,
            None,
            AnchorReuse::None,
            None,
        )
        .expect("a legal plan");
        let non_neutral_lf = plan.plan().spatial.lf.correlation != LfCorrelationDecision::default();
        let non_neutral_hf = plan.plan().spatial.lf_groups.iter().any(|group| {
            group
                .cfl
                .x_from_y()
                .iter()
                .chain(group.cfl.b_from_y())
                .any(|factor| factor.get() != 0)
        });
        assert!(
            non_neutral_lf || non_neutral_hf,
            "the correlated fixture must exercise non-neutral CfL on the wire"
        );
        let enabled = jpxl_encode::vardct::write_codestream(&plan).expect("encodes");
        let neutral = encode_with_cfl(&frame, false);
        let enabled_rmse = rmse(&decoded_rgb(&enabled), &rgb).sqrt();
        let neutral_rmse = rmse(&decoded_rgb(&neutral), &rgb).sqrt();
        eprintln!(
            "correlated colour: CfL {} B / RMSE {:.4}; neutral {} B / RMSE {:.4}",
            enabled.len(),
            enabled_rmse,
            neutral.len(),
            neutral_rmse
        );
        assert!(
            enabled.len() < neutral.len(),
            "CfL {} B must beat neutral {} B",
            enabled.len(),
            neutral.len()
        );
        assert!(
            enabled_rmse <= neutral_rmse + 0.05,
            "CfL RMSE {enabled_rmse} regressed from neutral {neutral_rmse}"
        );
    }

    fn hierarchical_request() -> EncodeRequest {
        let mut request = EncodeRequest::defaults();
        request.budget.cover_mode = CoverMode::Hierarchical;
        request
    }

    /// Counts varblocks by transform edge, so a cover can be compared to
    /// another cover rather than only to its byte count.
    fn cover_mix(plan: &jpxl_encode::vardct::ValidatedEmissionPlan) -> (usize, usize, usize) {
        let mut counts = (0usize, 0usize, 0usize);
        for block in plan
            .plan()
            .spatial
            .lf_groups
            .iter()
            .flat_map(|g| g.blocks.iter())
        {
            match block.transform {
                TransformType::Dct16x16 => counts.1 += 1,
                TransformType::Dct32x32 => counts.2 += 1,
                _ => counts.0 += 1,
            }
        }
        counts
    }

    #[test]
    fn the_neutral_size_penalty_is_exactly_one_and_byte_identical() {
        // IEEE multiplication by one is exact, so this is a bit-identity claim
        // and not a tolerance: the shipped objective must be untouched by the
        // Phase 6.2 plumbing.
        for edge in [8usize, 16, 32] {
            assert_eq!(CoverSizePenalty::Neutral.multiplier(edge), 1.0);
        }
        assert_eq!(CoverSizePenalty::default(), CoverSizePenalty::Neutral);

        let rgb = mixed_detail_rgb(256, 256);
        let mut explicit = hierarchical_request();
        explicit.cover_size_penalty = CoverSizePenalty::Neutral;
        assert_eq!(
            encode_srgb8_vardct(256, 256, &rgb, &hierarchical_request()).expect("encodes"),
            encode_srgb8_vardct(256, 256, &rgb, &explicit).expect("encodes"),
            "the default request and an explicitly neutral one must agree byte for byte"
        );
    }

    #[test]
    fn the_flat_frequency_weight_is_byte_identical_and_the_csf_weight_is_not() {
        // `Flat` builds no table at all, so `cell_weight` returns exactly 1.0
        // and the arithmetic is the pre-Phase-6 scorer's, bit for bit.
        assert_eq!(
            CoverFrequencyWeight::default(),
            CoverFrequencyWeight::Flat,
            "production must stay flat"
        );
        let rgb = mixed_detail_rgb(256, 256);
        let mut explicit = hierarchical_request();
        explicit.cover_frequency_weight = CoverFrequencyWeight::Flat;
        let baseline = encode_srgb8_vardct(256, 256, &rgb, &hierarchical_request()).expect("enc");
        assert_eq!(
            baseline,
            encode_srgb8_vardct(256, 256, &rgb, &explicit).expect("enc"),
            "an explicitly flat request must be byte-identical to the default"
        );

        for mode in [CoverFrequencyWeight::Csf, CoverFrequencyWeight::QuantDonor] {
            let mut weighted_req = hierarchical_request();
            weighted_req.cover_frequency_weight = mode;
            let weighted = encode_srgb8_vardct(256, 256, &rgb, &weighted_req).expect("enc");
            assert_ne!(
                baseline, weighted,
                "{mode:?} must actually reach the objective; if this passes \
                 trivially the weight is not being applied"
            );
        }
    }

    #[test]
    fn the_weight_tables_match_a_direct_recomputation_of_the_weighted_term() {
        // Proves the scorer applies the same per-cell table `csf` publishes, on
        // each transform's own grid, rather than some other ordering, a single
        // shared table, or the wrong candidate curve.
        for (mode, build) in [
            (
                CoverFrequencyWeight::Csf,
                crate::csf::square_weights as fn(usize, usize) -> Vec<f32>,
            ),
            (
                CoverFrequencyWeight::QuantDonor,
                crate::csf::quant_donor_weights as fn(usize, usize) -> Vec<f32>,
            ),
        ] {
            for transform in SQUARE_TRANSFORMS {
                let side = transform.coeff_cols();
                let llf = transform.block_dims().0;
                let quants = HfQuantizers::new_with_scales(
                    45_000,
                    HfMul::new(1).expect("legal"),
                    &[HfMul::new(1).expect("legal")],
                    NEUTRAL_QM_SCALE,
                    NEUTRAL_QM_SCALE,
                    CoverSizePenalty::Neutral,
                    mode,
                    QuantizerChoiceMode::Nearest,
                    1.0,
                )
                .expect("quantizers");
                let table = quants.frequency_weights(transform);
                let expected = build(side, llf);
                assert_eq!(table.len(), expected.len(), "{mode:?} {transform:?} length");
                for (cell, (got, want)) in table.iter().zip(&expected).enumerate() {
                    assert!(
                        (got - want).abs() < 1e-6,
                        "{mode:?} {transform:?} cell {cell}: scorer {got} != published {want}"
                    );
                }
                // The mean-1 property is what keeps `lambda` calibrated, so
                // assert it on the table the objective actually reads.
                let mut sum = 0.0f64;
                let mut count = 0usize;
                for (cell, &weight) in table.iter().enumerate() {
                    if cell / side < llf && cell % side < llf {
                        continue;
                    }
                    sum += f64::from(weight);
                    count += 1;
                }
                #[allow(clippy::cast_precision_loss, reason = "counts are small")]
                let mean = sum / count as f64;
                assert!(
                    (mean - 1.0).abs() < 1e-5,
                    "{mode:?} {transform:?} weight mean {mean} would decalibrate lambda"
                );
            }
        }

        // And the flat policy must publish no table at all.
        let flat = HfQuantizers::new_with_scales(
            45_000,
            HfMul::new(1).expect("legal"),
            &[HfMul::new(1).expect("legal")],
            NEUTRAL_QM_SCALE,
            NEUTRAL_QM_SCALE,
            CoverSizePenalty::Neutral,
            CoverFrequencyWeight::Flat,
            QuantizerChoiceMode::Nearest,
            1.0,
        )
        .expect("quantizers");
        for transform in SQUARE_TRANSFORMS {
            assert!(
                flat.frequency_weights(transform).is_empty(),
                "the flat policy must build no table for {transform:?}"
            );
        }
    }

    #[test]
    fn the_nearest_quantizer_is_the_default_and_byte_identical() {
        assert_eq!(
            QuantizerChoiceMode::default(),
            QuantizerChoiceMode::Nearest,
            "production must stay nearest"
        );
        let rgb = mixed_detail_rgb(256, 256);
        let mut explicit = hierarchical_request();
        explicit.quantizer_choice = QuantizerChoiceMode::Nearest;
        let baseline = encode_srgb8_vardct(256, 256, &rgb, &hierarchical_request()).expect("enc");
        assert_eq!(
            baseline,
            encode_srgb8_vardct(256, 256, &rgb, &explicit).expect("enc"),
            "an explicitly nearest request must be byte-identical to the default"
        );

        let mut rd = hierarchical_request();
        rd.quantizer_choice = QuantizerChoiceMode::RateDistortion;
        assert_ne!(
            baseline,
            encode_srgb8_vardct(256, 256, &rgb, &rd).expect("enc"),
            "rate-distortion choice must reach the wire; if this passes \
             trivially the mode is not being applied"
        );
    }

    #[test]
    fn lambda_scale_defaults_to_one_and_scales_the_derived_weight() {
        // Phase 7.2: the research multiplier must be a no-op at 1.0 so the
        // shipped path stays bit-identical, and must actually move `lambda`
        // when set — otherwise a sweep that reports "no operating point"
        // would be measuring nothing.
        assert!(
            (EncodeRequest::defaults().lambda_scale - 1.0).abs() < f32::EPSILON,
            "production default must be the unit scale"
        );
        assert!(
            (EncodeRequest::for_target(RateTarget::BitsPerPixel(1.0)).lambda_scale - 4.0).abs()
                < f32::EPSILON,
            "target-rate policy carries the Phase 7.2 calibrated scale of 4.0"
        );

        let muls = [HfMul::new(1).expect("legal")];
        let base = HfQuantizers::new_with_scales(
            45_000,
            muls[0],
            &muls,
            NEUTRAL_QM_SCALE,
            NEUTRAL_QM_SCALE,
            CoverSizePenalty::Neutral,
            CoverFrequencyWeight::Flat,
            QuantizerChoiceMode::Nearest,
            1.0,
        )
        .expect("quantizers");
        let doubled = HfQuantizers::new_with_scales(
            45_000,
            muls[0],
            &muls,
            NEUTRAL_QM_SCALE,
            NEUTRAL_QM_SCALE,
            CoverSizePenalty::Neutral,
            CoverFrequencyWeight::Flat,
            QuantizerChoiceMode::Nearest,
            2.0,
        )
        .expect("quantizers");
        for channel in 0..NUM_CHANNELS {
            let a = base.lambda.get(channel).copied().unwrap_or(0.0);
            let b = doubled.lambda.get(channel).copied().unwrap_or(0.0);
            assert!(
                a > 0.0 && (b - 2.0 * a).abs() < 1e-12 * a.max(1.0),
                "channel {channel}: scale 2.0 must double lambda ({a} -> {b})"
            );
        }

        // Unit scale is the production path: an explicit 1.0 must match the
        // default request byte-for-byte. (A non-unit scale *does* move the
        // cover under Nearest, because block_cost_bounded always multiplies
        // by lambda — that is intentional for research, not a defect.)
        let rgb = mixed_detail_rgb(256, 256);
        let baseline = encode_srgb8_vardct(256, 256, &rgb, &hierarchical_request()).expect("enc");
        let mut unit = hierarchical_request();
        unit.lambda_scale = 1.0;
        assert_eq!(
            baseline,
            encode_srgb8_vardct(256, 256, &rgb, &unit).expect("enc"),
            "an explicit lambda_scale of 1.0 must be byte-identical to the default"
        );

        // Under TrailingTruncation the scale must reach the wire: a 4x weight
        // changes which trailing drops pay, so the codestream cannot match
        // the unit-scale truncation arm if the control is live.
        let mut tr_unit = hierarchical_request();
        tr_unit.quantizer_choice = QuantizerChoiceMode::TrailingTruncation;
        tr_unit.lambda_scale = 1.0;
        let mut tr_hot = tr_unit;
        tr_hot.lambda_scale = 4.0;
        assert_ne!(
            encode_srgb8_vardct(256, 256, &rgb, &tr_unit).expect("enc"),
            encode_srgb8_vardct(256, 256, &rgb, &tr_hot).expect("enc"),
            "lambda_scale must reach the trailing-truncation path"
        );
    }

    #[test]
    fn rate_distortion_choice_minimises_its_own_objective_and_widens_the_dead_zone() {
        // Two claims. First, that no candidate beats the one chosen on
        // `residual_bits + rd * error^2` — recomputed here from the public
        // surface, so a regression in the private cost function is caught.
        // Second, that the mode only ever zeroes *more* than nearest, never
        // less: zero is the one candidate that costs no bits, so a rate term
        // can only widen the dead zone.
        let muls = [HfMul::new(1).expect("legal")];
        let nearest = HfQuantizers::new_with_scales(
            45_000,
            muls[0],
            &muls,
            NEUTRAL_QM_SCALE,
            NEUTRAL_QM_SCALE,
            CoverSizePenalty::Neutral,
            CoverFrequencyWeight::Flat,
            QuantizerChoiceMode::Nearest,
            1.0,
        )
        .expect("quantizers");
        let rd = HfQuantizers::new_with_scales(
            45_000,
            muls[0],
            &muls,
            NEUTRAL_QM_SCALE,
            NEUTRAL_QM_SCALE,
            CoverSizePenalty::Neutral,
            CoverFrequencyWeight::Flat,
            QuantizerChoiceMode::RateDistortion,
            1.0,
        )
        .expect("quantizers");

        let mut widened = 0usize;
        for transform in SQUARE_TRANSFORMS {
            let side = transform.coeff_cols();
            let qn = nearest.get(transform, muls[0]).expect("nearest");
            let qr = rd.get(transform, muls[0]).expect("rd");
            let lambda = rd.lambda[1];
            #[allow(clippy::cast_precision_loss, reason = "side is at most 32")]
            let to_sample = (side * side) as f64;
            for cell in 1..side * side {
                let step = f64::from(qr.step(1, cell));
                if step <= 0.0 {
                    continue;
                }
                for k in 0..24 {
                    #[allow(
                        clippy::cast_possible_truncation,
                        reason = "the probe targets are f32 quantizer inputs by design; \\\n                                  the tested cells sit far from f32 boundaries"
                    )]
                    let target = (f64::from(k) * 0.25 * step) as f32;
                    let a = qn.choose(target, 1, cell).expect("nearest choice");
                    let b = qr.choose(target, 1, cell).expect("rd choice");
                    if b.abs() < a.abs() {
                        widened += 1;
                    }
                    assert!(
                        b.abs() <= a.abs(),
                        "{transform:?} cell {cell} target {target}: rd chose {b}, \
                         which is larger than nearest's {a} — a rate term cannot \
                         make a coefficient more expensive"
                    );
                    // No candidate may beat the chosen one on the RD objective.
                    let cost = |q: i32| -> f64 {
                        let e = f64::from(qr.reconstruct(q, 1, cell) - target);
                        let bits = if q == 0 {
                            0.0
                        } else {
                            f64::from(33 - q.unsigned_abs().leading_zeros())
                        };
                        bits + lambda * to_sample * e * e
                    };
                    let chosen = cost(b);
                    for cand in [0, b - 1, b + 1] {
                        assert!(
                            cost(cand) >= chosen - 1e-6,
                            "{transform:?} cell {cell} target {target}: candidate \
                             {cand} costs {} against the chosen {b}'s {chosen}",
                            cost(cand)
                        );
                    }
                }
            }
        }
        assert!(
            widened > 0,
            "the rate term must zero something the nearest rule kept, or it is inert"
        );
    }

    #[test]
    fn exact_sample_distortion_is_not_parseval_for_hornuss() {
        // Phase 5P `distortion-proof`: sample-domain recon SSE is not the
        // square-DCT Parseval map (coeff SSE · side²) for Hornuss. The cover
        // research mode that used this ground truth still regressed quality
        // (see JPXL/docs/experiments/2026-08-12-special-exact-distortion.md)
        // and was removed; the proof stands as a permanent constraint on any
        // future special-transform scorer.
        use jpxl_core::varblock::{CoeffMatrix, SampleBlock};

        let side = 8usize;
        let cells = side * side;
        let mut samples = vec![0.0f32; cells];
        if let Some(slot) = samples.get_mut(3 * side + 5) {
            *slot = 40.0;
        }
        let sample_block = SampleBlock::from_rows_cols(side, side, samples.clone());

        let transform = TransformType::Hornuss;
        let coeffs = transform.coefficients_from_samples(&sample_block);
        let quant = HfQuantizer::new(transform, 45_000, 1, NEUTRAL_QM_SCALE, NEUTRAL_QM_SCALE)
            .expect("quantizer");
        let mut recon_data = coeffs.as_slice().to_vec();
        let mut coeff_sse = 0.0f64;
        for cell in 0..cells {
            if is_llf_cell(cell, side, 1) {
                continue;
            }
            let t = coeffs.as_slice().get(cell).copied().unwrap_or(0.0);
            let q = quant.choose(t, 1, cell).expect("choose");
            let recon = quant.reconstruct(q, 1, cell);
            if let Some(slot) = recon_data.get_mut(cell) {
                *slot = recon;
            }
            let e = f64::from(recon - t);
            coeff_sse += e * e;
        }
        let recon_mat =
            CoeffMatrix::from_landscape(transform.coeff_rows(), transform.coeff_cols(), recon_data);
        let recon_samples = transform.samples_from_coefficients(&recon_mat);
        let mut sample_sse = 0.0f64;
        for i in 0..cells {
            let o = samples.get(i).copied().unwrap_or(0.0);
            let r = recon_samples.as_slice().get(i).copied().unwrap_or(0.0);
            let d = f64::from(r - o);
            sample_sse += d * d;
        }
        let parseval = coeff_sse * (side * side) as f64;
        let ratio = if sample_sse > 1e-9 {
            parseval / sample_sse
        } else {
            0.0
        };
        assert!(
            (ratio - 1.0).abs() > 0.05 || sample_sse > 1e-6,
            "Hornuss Parseval map accidentally matches sample SSE \
             (parseval={parseval}, sample={sample_sse}, ratio={ratio})"
        );
        assert!(sample_sse.is_finite() && sample_sse >= 0.0);
    }

    #[test]
    fn the_measured_size_penalty_charges_larger_transforms_more_and_splits_more() {
        // Phase 6.2 Result B: at equal sample-domain error a DCT32x32 basis
        // costs more butteraugli than DCT8x8 bases, so pricing it correctly
        // must move the cover *away* from merging. The constants are strictly
        // increasing in size, and the cover must respond in that direction.
        assert!(
            CoverSizePenalty::Measured.multiplier(8) < CoverSizePenalty::Measured.multiplier(16)
                && CoverSizePenalty::Measured.multiplier(16)
                    < CoverSizePenalty::Measured.multiplier(32),
            "the measured penalty must be strictly increasing in transform size"
        );

        let rgb = mixed_detail_rgb(256, 256);
        let frame = PreparedFrame::from_srgb8(256, 256, &rgb).expect("frame");
        let atlas = AnalysisAtlas::analyze(&frame);

        let neutral = hierarchical_request();
        let mut measured = neutral;
        measured.cover_size_penalty = CoverSizePenalty::Measured;

        let plan_of = |request: &EncodeRequest| {
            plan_at(
                &frame,
                &atlas,
                request,
                QuantizerChoice::from_request(request),
            )
            .expect("a legal plan")
        };
        let (n8, n16, n32) = cover_mix(&plan_of(&neutral));
        let (m8, m16, m32) = cover_mix(&plan_of(&measured));
        eprintln!(
            "neutral 8x8={n8} 16x16={n16} 32x32={n32}; measured 8x8={m8} 16x16={m16} 32x32={m32}"
        );

        assert!(
            n32 > 0 || n16 > 0,
            "the smooth fixture must merge under the neutral objective, or this proves nothing"
        );
        assert!(
            (m32, m16) != (n32, n16),
            "the measured penalty must change the cover on a fixture that merges"
        );
        assert!(
            m32 <= n32 && m8 >= n8,
            "charging large transforms more must not produce *more* merging: \
             neutral (8={n8}, 16={n16}, 32={n32}) vs measured (8={m8}, 16={m16}, 32={m32})"
        );
    }

    /// Broad smooth gradients over most of the frame — the content larger
    /// transforms exist for — with one noise-textured corner the solver must
    /// keep splitting. The corner is hash noise, not a checkerboard: a perfect
    /// pixel checker is a *single* DCT basis function at every transform size,
    /// so it merges compactly and proves nothing about detail handling.
    fn mixed_detail_rgb(width: u32, height: u32) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            usize::try_from(u64::from(width) * u64::from(height) * 3).unwrap_or(0),
        );
        for y in 0..height {
            for x in 0..width {
                let smooth = (x * 160 / width.max(1)) + (y * 60 / height.max(1));
                let busy_corner = x < width / 4 && y < height / 4;
                let texture = if busy_corner {
                    let hash = (x.wrapping_mul(0x9E37).wrapping_add(y.wrapping_mul(0x79B9)))
                        .wrapping_mul(0x85EB_CA6B);
                    (hash >> 24) & 0x5F
                } else {
                    0
                };
                let luma = u8::try_from((30 + smooth + texture).min(255)).unwrap_or(255);
                out.extend_from_slice(&[
                    luma,
                    u8::try_from(u16::from(luma) * 4 / 5).unwrap_or(255),
                    u8::try_from(u16::from(luma) * 3 / 5).unwrap_or(255),
                ]);
            }
        }
        out
    }

    /// A broad diagonal gradient: the all-smooth extreme of the corpus.
    fn gradient_rgb(width: u32, height: u32) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            usize::try_from(u64::from(width) * u64::from(height) * 3).unwrap_or(0),
        );
        for y in 0..height {
            for x in 0..width {
                let luma =
                    u8::try_from(40 + (x + y) * 160 / (width + height).max(1)).unwrap_or(255);
                out.extend_from_slice(&[
                    luma,
                    u8::try_from(u16::from(luma) * 4 / 5).unwrap_or(255),
                    u8::try_from(u16::from(luma) * 3 / 5).unwrap_or(255),
                ]);
            }
        }
        out
    }

    #[test]
    fn the_hierarchical_cover_is_exact_and_in_blockinfo_order() {
        // 300x260 is 38x33 atoms: clipped on both edges, so the solver must
        // fall back to smaller squares against the frame boundary.
        let rgb = mixed_detail_rgb(300, 260);
        let frame = PreparedFrame::from_srgb8(300, 260, &rgb).expect("frame");
        let plan = plan_frame(&frame, &hierarchical_request()).expect("a legal plan");
        let geometry = plan.geometry().expect("geometry");

        for group in &plan.plan().spatial.lf_groups {
            let grid = geometry.lf_group_blocks(group.id).expect("grid");
            let cells = usize::try_from(grid.area()).expect("small");
            let mut covered = vec![false; cells];
            let mut last_raster = None;
            for vb in &group.blocks {
                let (rows, cols) = vb.transform.block_dims();
                let (bx, by) = (vb.origin.bx(), vb.origin.by());
                let raster = u64::from(by) * u64::from(grid.width) + u64::from(bx);
                assert!(
                    last_raster < Some(raster),
                    "varblock origins must be strictly increasing in raster order"
                );
                last_raster = Some(raster);
                for dy in 0..u32::try_from(rows).expect("small") {
                    for dx in 0..u32::try_from(cols).expect("small") {
                        let (x, y) = (bx + dx, by + dy);
                        assert!(x < grid.width && y < grid.height, "footprint clipped");
                        let cell = usize::try_from(y * grid.width + x).expect("small");
                        let slot = covered.get_mut(cell).expect("inside the grid");
                        assert!(!*slot, "atom ({x},{y}) covered twice");
                        *slot = true;
                    }
                }
            }
            assert!(
                covered.iter().all(|&c| c),
                "every atom covered exactly once"
            );
        }
    }

    #[test]
    fn smooth_content_merges_and_dominates_fixed_dct8x8_at_matched_quality() {
        // The exit criterion is a *matched-quality* size win, so the fixed
        // baseline is measured at a finer quantizer chosen to close the
        // quality gap the merges open. The claim proved: the hierarchical
        // point lies strictly below the fixed R-D curve — smaller AND better
        // than a fixed encoding that spends more bytes trying to catch up.
        let rgb = gradient_rgb(256, 256);
        let frame = PreparedFrame::from_srgb8(256, 256, &rgb).expect("frame");
        let atlas = AnalysisAtlas::analyze(&frame);

        let request = hierarchical_request();
        let plan = plan_at(
            &frame,
            &atlas,
            &request,
            QuantizerChoice::from_request(&request),
        )
        .expect("a legal plan");
        let merged = plan
            .plan()
            .spatial
            .lf_groups
            .iter()
            .flat_map(|g| g.blocks.iter())
            .filter(|b| b.transform != TransformType::Dct8x8)
            .count();
        assert!(
            merged > 0,
            "the smooth fixture must put a larger transform on the wire"
        );

        let hierarchical = jpxl_encode::vardct::write_codestream(&plan).expect("encodes");
        let hierarchical_rmse = rmse(&decoded_rgb(&hierarchical), &rgb).sqrt();

        // Fixed DCT8x8 at a finer global_scale: measured 3321 B / RMSE 0.4615
        // against the hierarchical 3303 B / RMSE 0.3786 — dominated on both
        // axes, and the fixed curve flattens (60000 gives RMSE 0.4590 at
        // 3382 B), so no fixed point reaches the hierarchical quality at all.
        let mut fine = EncodeRequest::defaults();
        fine.global_scale = jpxl_encode::vardct::ids::GlobalScale::new(45_000).expect("legal");
        let fixed = encode_srgb8_vardct(256, 256, &rgb, &fine).expect("encodes");
        let fixed_rmse = rmse(&decoded_rgb(&fixed), &rgb).sqrt();
        eprintln!(
            "hierarchical {} B / RMSE {:.4} ({merged} merged); fixed(gs=45000) {} B / RMSE {:.4}",
            hierarchical.len(),
            hierarchical_rmse,
            fixed.len(),
            fixed_rmse
        );
        assert!(
            hierarchical.len() < fixed.len() && hierarchical_rmse < fixed_rmse,
            "hierarchical ({} B, RMSE {hierarchical_rmse:.4}) must dominate \
             fixed at the matched-quality quantizer ({} B, RMSE {fixed_rmse:.4})",
            hierarchical.len(),
            fixed.len()
        );
    }

    #[test]
    fn detail_keeps_small_blocks_where_the_content_needs_them() {
        let rgb = mixed_detail_rgb(256, 256);
        let frame = PreparedFrame::from_srgb8(256, 256, &rgb).expect("frame");
        let plan = plan_frame(&frame, &hierarchical_request()).expect("a legal plan");
        let blocks: Vec<_> = plan
            .plan()
            .spatial
            .lf_groups
            .iter()
            .flat_map(|g| g.blocks.iter())
            .collect();
        // The noisy corner (atoms x,y < 8) must stay predominantly at DCT8x8
        // and never reach DCT32x32; an occasional 16x16 merge over noise is a
        // legitimate marginal R-D trade, a 32x32 there would mean the
        // distortion term is broken. The smooth remainder must contain merges.
        let corner: Vec<_> = blocks
            .iter()
            .filter(|b| b.origin.bx() < 8 && b.origin.by() < 8)
            .collect();
        assert!(
            corner
                .iter()
                .all(|b| b.transform != TransformType::Dct32x32),
            "the noise corner must not merge to DCT32x32"
        );
        let small = corner
            .iter()
            .filter(|b| b.transform == TransformType::Dct8x8)
            .count();
        assert!(
            small * 2 > corner.len(),
            "the noise corner must stay predominantly DCT8x8 ({small} of {})",
            corner.len()
        );
        assert!(
            blocks
                .iter()
                .any(|b| b.transform == TransformType::Dct32x32),
            "the smooth region must merge to DCT32x32"
        );
    }

    fn aq_request(mode: AqMode) -> EncodeRequest {
        let mut request = EncodeRequest::defaults();
        request.budget.aq_mode = mode;
        request
    }

    /// Left half a gentle gradient, right half hash noise: the two-population
    /// fixture every slice-17 claim is measured on.
    fn half_flat_half_noise(width: u32, height: u32) -> Vec<u8> {
        let mut out = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let luma = if x < width / 2 {
                    u8::try_from(100 + (x + y) / 16).unwrap_or(128)
                } else {
                    let hash = (x.wrapping_mul(0x9E37).wrapping_add(y.wrapping_mul(0x79B9)))
                        .wrapping_mul(0x85EB_CA6B);
                    u8::try_from(96 + ((hash >> 24) & 0x3F)).unwrap_or(128)
                };
                out.extend_from_slice(&[
                    luma,
                    u8::try_from(u16::from(luma) * 4 / 5).unwrap_or(255),
                    u8::try_from(u16::from(luma) * 3 / 5).unwrap_or(255),
                ]);
            }
        }
        out
    }

    /// RMSE over one half of the frame, all three channels.
    fn half_rmse(decoded: &[u8], source: &[u8], width: u32, height: u32, left: bool) -> f64 {
        let mut sum = 0.0f64;
        let mut count = 0u64;
        for y in 0..height {
            for x in 0..width {
                if (x < width / 2) != left {
                    continue;
                }
                for channel in 0..3u32 {
                    let i = usize::try_from((y * width + x) * 3 + channel).unwrap_or(usize::MAX);
                    let error = f64::from(decoded.get(i).copied().unwrap_or(0))
                        - f64::from(source.get(i).copied().unwrap_or(0));
                    sum += error * error;
                    count += 1;
                }
            }
        }
        (sum / count.max(1) as f64).sqrt()
    }

    #[test]
    fn a_neutral_aq_field_collapses_to_the_plain_wire() {
        // A flat frame has no activity deviation, so the field is neutral and
        // the §7.2 factorization (which costs a non-zero `mul` row) must not
        // be paid for. Byte identity, not just size parity.
        let rgb = vec![128u8; 128 * 128 * 3];
        let off = encode_srgb8_vardct(128, 128, &rgb, &aq_request(AqMode::Off)).expect("encodes");
        let aq =
            encode_srgb8_vardct(128, 128, &rgb, &aq_request(AqMode::Masking)).expect("encodes");
        assert_eq!(off, aq, "a neutral field must cost nothing");
    }

    #[test]
    fn the_aq_factorization_leaves_every_lf_integer_unchanged() {
        // Halving global_scale and doubling quant_lf preserves their product
        // exactly, so the LF planes — quantized against that product — must
        // be identical integers, not merely close.
        let rgb = half_flat_half_noise(256, 256);
        let frame = PreparedFrame::from_srgb8(256, 256, &rgb).expect("frame");
        // The fixed cover isolates the factorization: under the hierarchical
        // default the field also steers the *cover*, and merged transforms
        // legitimately produce different LF integers via I.8.
        let mut off_request = aq_request(AqMode::Off);
        off_request.budget.cover_mode = CoverMode::FixedDct8x8;
        let mut aq_on_request = aq_request(AqMode::Masking);
        aq_on_request.budget.cover_mode = CoverMode::FixedDct8x8;
        let off = plan_frame(&frame, &off_request).expect("plan");
        let aq = plan_frame(&frame, &aq_on_request).expect("plan");
        // The factorization must actually be in play for this to prove
        // anything.
        assert_ne!(
            off.plan().spatial.quantizer.global_scale,
            aq.plan().spatial.quantizer.global_scale,
            "the fixture must trigger the factorization"
        );
        for (a, b) in off
            .plan()
            .quantized
            .lf_groups
            .iter()
            .zip(aq.plan().quantized.lf_groups.iter())
        {
            for channel in 0..3 {
                assert_eq!(
                    a.lf.plane(channel),
                    b.lf.plane(channel),
                    "LF integers must be untouched by the HF factorization"
                );
            }
        }
    }

    #[test]
    fn uniform_aq_improves_spatial_quality_uniformity() {
        // The slice-17 exit criterion, measured directly: the error gap
        // between the flat and the busy half shrinks when the field points
        // at equalization. Measured: gap 4.36 -> 2.97 at 12408 -> 17141 B
        // (the busy half is refined; the flat half's HF is already zero, so
        // coarsening it saves nothing on this fixture).
        let rgb = half_flat_half_noise(256, 256);
        let off = encode_srgb8_vardct(256, 256, &rgb, &aq_request(AqMode::Off)).expect("encodes");
        let uniform =
            encode_srgb8_vardct(256, 256, &rgb, &aq_request(AqMode::Uniform)).expect("encodes");

        let off_out = decoded_rgb(&off);
        let uniform_out = decoded_rgb(&uniform);
        let off_gap = (half_rmse(&off_out, &rgb, 256, 256, false)
            - half_rmse(&off_out, &rgb, 256, 256, true))
        .abs();
        let uniform_gap = (half_rmse(&uniform_out, &rgb, 256, 256, false)
            - half_rmse(&uniform_out, &rgb, 256, 256, true))
        .abs();
        eprintln!(
            "uniformity: off gap {off_gap:.3} ({} B), uniform gap {uniform_gap:.3} ({} B)",
            off.len(),
            uniform.len()
        );
        assert!(
            uniform_gap < off_gap * 0.8,
            "the uniform field must shrink the flat/busy error gap \
             ({off_gap:.3} -> {uniform_gap:.3})"
        );
    }

    #[test]
    fn masking_aq_saves_bytes_and_refines_the_flat_half() {
        // The perceptual direction: texture masks coarser quantization, flat
        // regions band and get refined. Both claims measured against Off on
        // the same stream: smaller output AND a better flat half. Measured:
        // 12408 -> 9799 B, flat RMSE 0.426 -> 0.333.
        let rgb = half_flat_half_noise(256, 256);
        let frame = PreparedFrame::from_srgb8(256, 256, &rgb).expect("frame");
        let plan = plan_frame(&frame, &aq_request(AqMode::Masking)).expect("plan");
        let muls: std::collections::BTreeSet<u32> = plan
            .plan()
            .spatial
            .lf_groups
            .iter()
            .flat_map(|g| g.blocks.iter())
            .map(|b| b.hf_mul.get())
            .collect();
        assert!(
            muls.len() >= 2,
            "the fixture must put a varying mul row on the wire, got {muls:?}"
        );

        let off = encode_srgb8_vardct(256, 256, &rgb, &aq_request(AqMode::Off)).expect("encodes");
        let masking = jpxl_encode::vardct::write_codestream(&plan).expect("encodes");
        let off_flat = half_rmse(&decoded_rgb(&off), &rgb, 256, 256, true);
        let masking_flat = half_rmse(&decoded_rgb(&masking), &rgb, 256, 256, true);
        eprintln!(
            "masking: {} B flat {masking_flat:.3} vs off {} B flat {off_flat:.3}",
            masking.len(),
            off.len()
        );
        assert!(
            masking.len() < off.len(),
            "masking must save bytes ({} vs {})",
            masking.len(),
            off.len()
        );
        assert!(
            masking_flat < off_flat,
            "masking must refine the flat half ({off_flat:.3} -> {masking_flat:.3})"
        );
    }

    #[test]
    fn the_trained_entropy_model_beats_the_fixed_six_cluster_baseline() {
        // Baselines measured immediately before slice 18 on this machine with
        // [`cluster_of`]'s fixed six clusters and the frame-wide (4, 2, 0)
        // hybrid-uint configuration. The trained model changes no symbol —
        // the pixels are identical — so the whole delta is entropy density,
        // and slice 18's exit criterion asks exactly for that. Measured with
        // the trained model: 1984 and 10221 bytes (-14.8% / -17.6%); the
        // assertion keeps a margin so the trainer can evolve without
        // byte-pinning.
        let flat = vec![128u8; 256 * 256 * 3];
        let flat_bytes =
            encode_srgb8_vardct(256, 256, &flat, &EncodeRequest::defaults()).expect("encodes");
        assert!(
            (flat_bytes.len() as u64) < 2329 * 95 / 100,
            "flat grey: {} B must undercut the fixed-model 2329 B by 5%",
            flat_bytes.len()
        );

        let noise = half_flat_half_noise(256, 256);
        let noise_bytes =
            encode_srgb8_vardct(256, 256, &noise, &EncodeRequest::defaults()).expect("encodes");
        assert!(
            (noise_bytes.len() as u64) < 12408 * 95 / 100,
            "half noise: {} B must undercut the fixed-model 12408 B by 5%",
            noise_bytes.len()
        );
    }

    #[test]
    fn multi_preset_can_reach_the_wire_on_a_mixed_multi_group_frame() {
        // Slice 18d: half-flat / half-noise over a 2×2 group grid so the
        // mass fingerprint splits groups. Exact-price adopt may keep one
        // preset; if two win, both external-oracle path and self-decode
        // must accept the stream.
        let w = 512u32;
        let h = 512u32;
        let mut rgb = vec![128u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in (w / 2)..w {
                let n = ((x.wrapping_mul(17) ^ y.wrapping_mul(31)) & 0xff) as u8;
                let i = ((y * w + x) * 3) as usize;
                if let Some(px) = rgb.get_mut(i..i + 3) {
                    let vals = [n, n.wrapping_add(3), n.wrapping_add(7)];
                    for (slot, &v) in px.iter_mut().zip(vals.iter()) {
                        *slot = v;
                    }
                }
            }
        }
        let plan = plan_frame(
            &PreparedFrame::from_srgb8(w, h, &rgb).expect("frame"),
            &EncodeRequest::defaults(),
        )
        .expect("plan");
        let geometry = plan.plan().spatial.frame.geometry().expect("geometry");
        assert!(
            geometry.num_groups() >= 2,
            "fixture must span multiple HF groups"
        );
        let presets = plan.plan().entropy.num_hf_presets;
        let assignment: Vec<u32> = plan
            .plan()
            .entropy
            .passes
            .first()
            .map(|p| p.group_presets.iter().map(|id| id.get()).collect())
            .unwrap_or_default();
        eprintln!(
            "18d presets: num={presets}, assignment={assignment:?}, groups={}",
            geometry.num_groups()
        );
        let bytes = jpxl_encode::vardct::write_codestream(&plan).expect("writes");
        let decoded = jpxl_decode::decode::decode(&bytes, &jpxl_core::limits::Limits::default())
            .expect("multi-preset or single-preset stream must decode");
        assert_eq!(decoded.width, w);
        assert_eq!(decoded.height, h);
        if presets > 1 {
            assert!(
                assignment.iter().any(|&p| p > 0),
                "multi-preset must assign at least one group to preset 1"
            );
        }
    }

    #[test]
    fn a_trimmed_block_context_beats_default_on_fixed_dct8x8() {
        // Slice 18c density gate: FixedDct8x8 only fires shape class 0, so a
        // trimmed I.2.2 map collapses `nb_block_ctx` well below 15. On a
        // non-flat frame that saving must show up as fewer stream bytes at
        // identical pixels (model-only change).
        let rgb = half_flat_half_noise(256, 256);
        let mut request = EncodeRequest::defaults();
        request.budget.cover_mode = CoverMode::FixedDct8x8;
        let frame = PreparedFrame::from_srgb8(256, 256, &rgb).expect("frame");
        let plan = plan_frame(&frame, &request).expect("plan");
        let shapes: std::collections::BTreeSet<usize> = plan
            .plan()
            .spatial
            .lf_groups
            .iter()
            .flat_map(|g| g.blocks.iter().map(|b| b.transform.order_id()))
            .collect();
        assert_eq!(
            shapes,
            [0].into_iter().collect(),
            "fixed cover is shape 0 only"
        );
        assert!(
            !matches!(
                plan.plan().entropy.block_context,
                HfBlockContextPlan::Default
            ),
            "trimmed map must win on a single-shape frame"
        );
        let nb = plan.plan().entropy.block_context.nb_block_ctx();
        assert!(
            nb < 15,
            "trimmed nb_block_ctx {nb} must undercut the default 15"
        );

        let geometry = plan.plan().spatial.frame.geometry().expect("geometry");
        let (default_b, custom_b) = price_default_and_custom(
            plan.plan().spatial.as_ref().clone(),
            plan.plan().quantized.as_ref().clone(),
            &geometry,
            plan.plan().entropy.block_context.clone(),
        )
        .expect("prices");
        eprintln!("trimmed I.2.2: default {default_b} B vs custom {custom_b} B, nb={nb}");
        assert!(
            custom_b < default_b,
            "trimmed map must be smaller ({custom_b} vs {default_b})"
        );

        let bytes = jpxl_encode::vardct::write_codestream(&plan).expect("writes");
        let decoded = jpxl_decode::decode::decode(&bytes, &jpxl_core::limits::Limits::default())
            .expect("custom I.2.2 must decode");
        assert_eq!(decoded.width, 256);
        assert_eq!(decoded.height, 256);
    }

    #[test]
    fn frequency_trained_orders_reach_the_wire_and_beat_the_natural_baseline() {
        // Slice 18b: the half-noise fixture at defaults measured 10221 B with
        // the trained model at natural orders; the §9.4 candidate is adopted
        // (used_orders != 0) and prices below it (measured 10038 B). Adoption
        // is exact-guarded inside plan_at, so this can never regress past the
        // natural-order size.
        let rgb = half_flat_half_noise(256, 256);
        let frame = PreparedFrame::from_srgb8(256, 256, &rgb).expect("frame");
        let plan = plan_frame(&frame, &EncodeRequest::defaults()).expect("plan");
        let used = plan
            .plan()
            .entropy
            .passes
            .first()
            .map_or(0, |pass| pass.orders.used_orders());
        assert_ne!(used, 0, "the fixture must adopt a custom order");
        let bytes = jpxl_encode::vardct::write_codestream(&plan).expect("encodes");
        assert!(
            (bytes.len() as u64) < 10221,
            "custom orders must price below the natural-order 10221 B, got {}",
            bytes.len()
        );
    }

    #[test]
    fn aq_composes_with_the_hierarchical_cover() {
        let rgb = half_flat_half_noise(300, 260);
        let frame = PreparedFrame::from_srgb8(300, 260, &rgb).expect("frame");
        let mut request = aq_request(AqMode::Masking);
        request.budget.cover_mode = CoverMode::Hierarchical;
        let plan = plan_frame(&frame, &request).expect("a legal plan");
        let blocks: Vec<_> = plan
            .plan()
            .spatial
            .lf_groups
            .iter()
            .flat_map(|g| g.blocks.iter())
            .collect();
        assert!(
            blocks.iter().any(|b| b.transform != TransformType::Dct8x8),
            "the flat half must still merge"
        );
        let muls: std::collections::BTreeSet<u32> = blocks.iter().map(|b| b.hf_mul.get()).collect();
        assert!(
            muls.len() >= 2,
            "the field must still vary the mul row, got {muls:?}"
        );
    }

    #[test]
    fn gaborish_request_signals_restoration_and_preconditions() {
        // Milestone 9 start: enable gab → plan carries RestorationDecision and
        // the bitstream differs from the unfiltered path (preconditioned coeffs).
        let rgb = synthetic_rgb(64, 64, false);
        let frame = PreparedFrame::from_srgb8(64, 64, &rgb).expect("frame");
        let off = plan_frame(&frame, &EncodeRequest::defaults()).expect("off");
        assert!(!off.plan().spatial.restoration.gaborish);

        let mut request = EncodeRequest::defaults();
        request.restoration.gaborish = true;
        let on = plan_frame(&frame, &request).expect("on");
        assert!(on.plan().spatial.restoration.gaborish);
        assert_eq!(on.plan().spatial.restoration.epf_iters, 0);

        let off_bytes = jpxl_encode::vardct::write_codestream(&off).expect("writes");
        let on_bytes = jpxl_encode::vardct::write_codestream(&on).expect("writes");
        assert_ne!(
            off_bytes, on_bytes,
            "gaborish precondition + header must change the codestream"
        );

        // Self-decode must succeed with filters applied (J.3).
        let image = decode(&on_bytes, &Limits::default()).expect("decodes with gaborish");
        assert_eq!((image.width, image.height), (64, 64));
    }

    #[test]
    fn uniform7_sharpness_activates_epf_on_textured_input() {
        let rgb = synthetic_rgb(64, 64, false);
        let frame = PreparedFrame::from_srgb8(64, 64, &rgb).expect("frame");

        let off = plan_frame(&frame, &EncodeRequest::defaults()).expect("off");
        let off_bytes = jpxl_encode::vardct::write_codestream(&off).expect("writes");

        let mut request = EncodeRequest::defaults();
        request.restoration.epf_iters = 1;
        request.epf_sharpness = EpfSharpnessMode::Uniform7;
        let on = plan_frame(&frame, &request).expect("on");
        assert_eq!(on.plan().spatial.restoration.epf_iters, 1);
        assert!(on.plan().spatial.lf_groups.iter().all(|group| {
            group
                .sharpness
                .values()
                .iter()
                .all(|&sharpness| sharpness == 7)
        }));

        let on_bytes = jpxl_encode::vardct::write_codestream(&on).expect("writes");
        assert_ne!(off_bytes, on_bytes, "EPF metadata must change the stream");
        assert_ne!(
            decoded_rgb(&off_bytes),
            decoded_rgb(&on_bytes),
            "Sharpness 7 with one EPF step must change textured reconstruction"
        );
    }

    #[test]
    fn gaborish_path_stays_within_lossy_source_tolerance() {
        // Same class as the unfiltered 64x64 rung (peak ≤70, RMSE ≤9): inverse
        // Jacobi is approximate, but at default quantizer it must not destroy
        // the R-D baseline.
        let rgb = synthetic_rgb(64, 64, false);
        let mut request = EncodeRequest::defaults();
        request.restoration.gaborish = true;
        let bytes = encode_srgb8_vardct(64, 64, &rgb, &request).expect("encodes");
        let decoded = decoded_rgb(&bytes);
        let (peak, rmse) = {
            let mut peak = 0u32;
            let mut sum = 0f64;
            for (&p, &q) in rgb.iter().zip(&decoded) {
                let d = u32::from(p.abs_diff(q));
                peak = peak.max(d);
                sum += f64::from(d) * f64::from(d);
            }
            (peak, (sum / rgb.len() as f64).sqrt())
        };
        assert!(
            peak <= 70 && rmse <= 9.0,
            "gaborish path peak {peak} RMSE {rmse:.3} outside the unfiltered 64x64 class"
        );
    }

    #[test]
    fn epf_iters_out_of_range_is_refused() {
        let frame = grey_frame(16, 16);
        let mut request = EncodeRequest::defaults();
        request.restoration.epf_iters = 4;
        assert!(matches!(
            plan_frame(&frame, &request),
            Err(PolicyError::Unsupported { .. })
        ));
    }
}
