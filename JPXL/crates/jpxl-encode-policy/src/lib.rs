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
pub mod diagnostics;
mod entropy;
pub mod error;
pub mod field;
pub mod quantize;
pub mod rate;
pub mod request;
pub mod source;

use jpxl_core::forward::{
    CoeffView, CoeffViewMut, SampleView, SampleViewMut, TransformScratch, forward_varblock_into,
    lf_from_llf_into,
};
use jpxl_core::geometry::LfBlockPos;
use jpxl_core::varblock::TransformType;
use jpxl_encode::vardct::headers::{NEUTRAL_QM_SCALE, VARDCT_GROUP_SIZE_SHIFT};
use jpxl_encode::vardct::ids::{
    CflFactor, ClusterId, GlobalScale, HfMul, LfGroupId, PresetId, QuantLf,
};
use jpxl_encode::vardct::plan::{
    CflGrid, EmissionPlan, EntropyModelPlan, EntropyPlan, FrameDecision, HfBlockContextPlan,
    HfPassEntropyPlan, HistogramPlan, HybridUintPlan, LfCorrelationDecision, LfDecision,
    LfGroupPlan, LfQuantPlanes, OrderSet, QuantizedFrameIr, QuantizedLfGroup, QuantizerDecision,
    SectionLayout, SharpnessGrid, SpatialPlan, VarblockCoefficients, VarblockDecision,
};
use jpxl_encode::vardct::{ValidatedEmissionPlan, VardctGeometry, census_frame, validate};

use quantize::{
    CflAccumulator, DCT8X8_CELLS, DEFAULT_COLOUR_FACTOR, HfQuantizer, LfQuantizer, NUM_CHANNELS,
    cfl_multiplier,
};

pub use analysis::{AnalysisAtlas, AtomGrid};
pub use diagnostics::{
    ChooseStage, EncodeDiag, last_encode_diag, reset_encode_diag, take_encode_diag,
};
pub use error::{PolicyError, Result};
pub use field::AqMode;

use field::{DesiredQuantField, mul_lattice};
pub use rate::{
    LadderSearch, QuantizerChoice, RateOutcome, RatePhase, RateProbeStats, RateStep, Rung,
    search_frame,
};
pub use request::{
    CoverMode, EncodeRequest, RateSearchBudget, RateTarget, RateTolerance, SearchBudget,
};
pub use source::PreparedFrame;
// Re-export so callers can set [`EncodeRequest::restoration`] without a
// second dependency path into the writer crate's plan module.
pub use jpxl_encode::vardct::plan::RestorationDecision;

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
/// the exact price. The rate loop prices with Fast and re-plans the winner
/// with Full so intermediate probes skip several full `price_codestream`s.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EntropySearch {
    /// Default I.2.2 map, natural orders, one census + train.
    Fast,
    /// Slice-18 alternatives with exact-price adopt gates.
    Full,
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
    plan_at_with_cfl(
        frame,
        atlas,
        request,
        quantizer,
        true,
        None,
        &mut cache,
        EntropySearch::Full,
    )
}

/// Like [`plan_at`], but reuses a caller-prepared transform frame and a
/// cross-probe forward-transform cache (rate loop).
pub(crate) fn plan_at_on(
    frame: &PreparedFrame,
    transform_frame: &PreparedFrame,
    atlas: &AnalysisAtlas,
    request: &EncodeRequest,
    quantizer: QuantizerChoice,
    cache: &mut CandidateForwardCache,
    entropy: EntropySearch,
) -> Result<ValidatedEmissionPlan> {
    plan_at_with_cfl(
        frame,
        atlas,
        request,
        quantizer,
        true,
        Some(transform_frame),
        cache,
        entropy,
    )
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
fn plan_at_with_cfl(
    frame: &PreparedFrame,
    atlas: &AnalysisAtlas,
    request: &EncodeRequest,
    quantizer: QuantizerChoice,
    enable_cfl: bool,
    transform_override: Option<&PreparedFrame>,
    cache: &mut CandidateForwardCache,
    entropy_search: EntropySearch,
) -> Result<ValidatedEmissionPlan> {
    // Phase-0: one clean snapshot per plan_at (rate probes overwrite; last wins).
    diagnostics::reset_encode_diag();
    if request.restoration.epf_iters > 3 {
        return Err(PolicyError::Unsupported {
            what: "epf_iters outside 0..=3",
        });
    }

    let decision = FrameDecision {
        width: frame.width(),
        height: frame.height(),
        // F.2 signals `group_size_shift` only for kModular, so a kVarDCT frame
        // has no say: `group_dim` is 256 and the request's field is ignored
        // rather than silently written into a field that does not exist.
        group_size_shift: VARDCT_GROUP_SIZE_SHIFT,
        num_passes: 1,
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
    let hf_quants = HfQuantizers::new(aq.global_scale.get(), aq.baseline, &aq.muls())?;

    // The cover is selected before chroma-from-luma is estimated: the estimate
    // regresses over the coefficients of the *selected* transforms, so the
    // block map must exist first. Selection itself scores with neutral CfL —
    // block choice is dominated by luma structure (see `block_cost`).
    // Cover selection, then one forward transform per *selected* varblock.
    // CfL estimation and HF quantization both consume those coefficients so
    // a selected DCT is not recomputed (Opt-V within-probe cache).
    let mut fwd_scratch = ForwardScratch::new();
    let groups = diagnostics::time_stage(diagnostics::StageTimer::Cover, || {
        diagnostics::with_choose_stage(diagnostics::ChooseStage::Cover, || {
            let mut groups = Vec::new();
            for index in 0..geometry.num_lf_groups() {
                let id = LfGroupId::new(u32::try_from(index).unwrap_or(u32::MAX));
                let blocks = geometry
                    .lf_group_blocks(id)
                    .ok_or(PolicyError::Unsupported {
                        what: "an LF group outside the frame's grid",
                    })?;
                let rect = geometry.lf_group_rect(id).ok_or(PolicyError::Unsupported {
                    what: "an LF group outside the frame's grid",
                })?;
                let varblocks = match request.budget.cover_mode {
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
                        varblocks
                    }
                    CoverMode::Hierarchical => {
                        select_blocks(
                            transform_frame,
                            &hf_quants,
                            blocks,
                            (rect.x0, rect.y0),
                            &aq,
                            cache,
                            &mut fwd_scratch,
                        )?
                    }
                };
                let forwards = forward_selected(
                    transform_frame,
                    &varblocks,
                    (rect.x0, rect.y0),
                    cache,
                    &mut fwd_scratch,
                )?;
                groups.push((id, blocks, rect, varblocks, forwards));
            }
            Ok::<_, PolicyError>(groups)
        })
    })?;

    let maps: Vec<&[VarblockDecision]> = groups
        .iter()
        .map(|(_, _, _, varblocks, _)| varblocks.as_slice())
        .collect();
    let forwards_ref: Vec<&[VarblockForward]> = groups
        .iter()
        .map(|(_, _, _, _, forwards)| forwards.as_slice())
        .collect();
    let cfl = diagnostics::time_stage(diagnostics::StageTimer::Cfl, || {
        estimate_cfl(
            &geometry,
            &maps,
            &forwards_ref,
            &lf_quant,
            &hf_quants,
            enable_cfl && !frame.is_grayscale(),
        )
    })?;

    let mut lf_groups = Vec::new();
    let mut quantized = Vec::new();
    for (index, (id, blocks, _rect, varblocks, forwards)) in groups.into_iter().enumerate() {
        let group_cfl = cfl.groups.get(index).ok_or(PolicyError::Unsupported {
            what: "a missing CfL grid for an LF group",
        })?;
        let quantized_group = diagnostics::time_stage(diagnostics::StageTimer::Quantize, || {
            diagnostics::with_choose_stage(diagnostics::ChooseStage::Final, || {
                quantize_group(
                    &lf_quant,
                    &hf_quants,
                    &cfl.correlation,
                    group_cfl,
                    &varblocks,
                    &forwards,
                    blocks,
                )
            })
        })?;

        lf_groups.push(LfGroupPlan {
            id,
            blocks: varblocks.into_boxed_slice(),
            cfl: group_cfl.clone(),
            sharpness: SharpnessGrid::zeros(blocks),
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
        },
        lf,
        restoration: request.restoration,
        lf_groups: lf_groups.into_boxed_slice(),
    };

    // The entropy model is chosen in two steps because the census is a
    // function of the plan: a provisional plan carries the clustering and the
    // hybrid-uint configuration, `census_frame` walks it, and the real
    // histograms replace the provisional ones. The walk lives in `jpxl-encode`
    // so that the counts trained here and the symbols emitted there cannot
    // come from two different traversals.
    let quantized_ir = QuantizedFrameIr {
        lf_groups: quantized.into_boxed_slice(),
    };
    let provisional = EmissionPlan::new(
        spatial,
        quantized_ir,
        entropy_plan(
            &geometry,
            placeholder_histograms(),
            HfBlockContextPlan::Default,
        )?,
        SectionLayout::for_geometry(&geometry),
    );
    // Slice 18 / 18b: train under the default I.2.2 map, then optionally
    // adopt custom coefficient orders on an exact price win (Full only).
    let with_default = diagnostics::time_stage(diagnostics::StageTimer::Entropy, || {
        train_entropy_with_orders(provisional.clone(), &geometry, entropy_search)
    })?;

    // Fast rate probes stop here: default map + natural orders is an upper
    // bound on Full's size (Full only adopts alternatives that strictly win).
    if entropy_search == EntropySearch::Fast {
        return Ok(with_default);
    }

    // Slice 18c: a custom I.2.2 block context changes every pre-context id,
    // so it gets its own census + train + order pass, and is adopted only
    // when the writer's exact price is strictly smaller. Flat / constant-mul
    // content proposes Default and skips the second walk.
    let mut best = with_default;
    let candidate_bc =
        entropy::propose_block_context(best.plan().spatial.as_ref(), best.plan().quantized.as_ref());
    if !matches!(candidate_bc, HfBlockContextPlan::Default) {
        let mut custom_walk = provisional.clone();
        custom_walk.entropy = entropy_plan(&geometry, placeholder_histograms(), candidate_bc)?;
        let with_custom = diagnostics::time_stage(diagnostics::StageTimer::Entropy, || {
            train_entropy_with_orders(custom_walk, &geometry, EntropySearch::Full)
        })?;
        let best_size = jpxl_encode::vardct::price_codestream(&best)?.total;
        let custom_size = jpxl_encode::vardct::price_codestream(&with_custom)?.total;
        if custom_size < best_size {
            best = with_custom;
        }
    }

    // Slice 18d: multi-preset assignment. Needs ≥2 pass groups; changes the
    // walk's I.4 offset per group, so re-census + retrain + exact price.
    if let Some((num_presets, assignment)) = entropy::propose_presets(
        &geometry,
        best.plan().spatial.as_ref(),
        best.plan().quantized.as_ref(),
    ) {
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
        if let Ok(with_presets) =
            diagnostics::time_stage(diagnostics::StageTimer::Entropy, || {
                train_entropy_with_orders(multi, &geometry, EntropySearch::Full)
            })
        {
            let best_size = jpxl_encode::vardct::price_codestream(&best)?.total;
            let multi_size = jpxl_encode::vardct::price_codestream(&with_presets)?.total;
            if multi_size < best_size {
                best = with_presets;
            }
        }
    }

    Ok(best)
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
    let with_default =
        train_entropy_with_orders(provisional.clone(), geometry, EntropySearch::Full)?;
    let mut custom_walk = provisional;
    custom_walk.entropy = entropy_plan(geometry, placeholder_histograms(), candidate)?;
    let with_custom = train_entropy_with_orders(custom_walk, geometry, EntropySearch::Full)?;
    Ok((
        jpxl_encode::vardct::price_codestream(&with_default)?.total,
        jpxl_encode::vardct::price_codestream(&with_custom)?.total,
    ))
}

/// Trains clusters / hybrid-uint from a census of `provisional`, then optionally
/// runs the §9.4 order candidate and keeps it only on an exact price win.
fn train_entropy_with_orders(
    provisional: EmissionPlan,
    geometry: &VardctGeometry,
    entropy_search: EntropySearch,
) -> Result<ValidatedEmissionPlan> {
    let block_context = provisional.entropy.block_context.clone();
    let num_hf_presets = provisional.entropy.num_hf_presets;
    let group_presets: Vec<PresetId> = provisional
        .entropy
        .passes
        .first()
        .map(|p| p.group_presets.to_vec())
        .unwrap_or_default();
    let census = census_frame(&provisional, geometry)?;
    let model = entropy::train(&census)?;
    // Arc-clone spatial/quantized; only entropy is rebuilt.
    let natural = validate(EmissionPlan {
        entropy: trained_entropy_plan(
            geometry,
            model,
            OrderSet::natural(),
            block_context.clone(),
            num_hf_presets,
            group_presets.clone(),
        )?,
        ..provisional.clone()
    })?;
    if entropy_search == EntropySearch::Fast {
        return Ok(natural);
    }

    let orders = entropy::candidate_orders(
        provisional.spatial.as_ref(),
        provisional.quantized.as_ref(),
    )?;
    if orders.overrides().is_empty() {
        return Ok(natural);
    }

    let mut reordered_walk = provisional;
    if let Some(pass) = reordered_walk.entropy.passes.first_mut() {
        pass.orders = orders.clone();
    }
    let census = census_frame(&reordered_walk, geometry)?;
    let model = entropy::train(&census)?;
    let reordered = validate(EmissionPlan {
        entropy: trained_entropy_plan(
            geometry,
            model,
            orders,
            block_context,
            num_hf_presets,
            group_presets,
        )?,
        spatial: reordered_walk.spatial,
        quantized: reordered_walk.quantized,
        sections: reordered_walk.sections,
    })?;

    let natural_size = jpxl_encode::vardct::price_codestream(&natural)?.total;
    let reordered_size = jpxl_encode::vardct::price_codestream(&reordered)?.total;
    Ok(if reordered_size < natural_size {
        reordered
    } else {
        natural
    })
}

/// The frame-wide LF factors and one HF factor grid per LF group.
struct CflEstimate {
    correlation: LfCorrelationDecision,
    groups: Vec<CflGrid>,
}

/// One coefficient sample used by the integer refinement.
#[derive(Debug, Clone, Copy)]
struct CflSample {
    source: f32,
    reconstructed_y: f32,
    cell: usize,
}

/// Regression sums plus the exact samples needed to score neighbouring wire
/// factors through the quantizer's decoder-side arithmetic.
#[derive(Debug, Default)]
struct CflSamples {
    regression: CflAccumulator,
    samples: Vec<CflSample>,
}

impl CflSamples {
    fn push(&mut self, source: f32, regression_y: f32, reconstructed_y: f32, cell: usize) {
        diagnostics::note_cfl_samples(1);
        self.regression.add(regression_y, source);
        self.samples.push(CflSample {
            source,
            reconstructed_y,
            cell,
        });
    }
}

/// The searchable HF factor samples of one LF group's 64x64 tiles.
struct HfCflSamples {
    tiles: jpxl_encode::vardct::BlockGrid,
    x: Vec<CflSamples>,
    b: Vec<CflSamples>,
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
        let field = match DesiredQuantField::from_atlas(atlas, request.budget.aq_mode) {
            Some(field) if !field.is_neutral() => field,
            _ => return off,
        };

        // An odd `global_scale` is snapped down to the even family (an
        // off-by-one the rate loop prices exactly) instead of falling back to
        // refine-only: a fallback keyed on parity would make adjacent rate
        // rungs alternate between two differently-sized encoders and put a
        // systematic sawtooth in the ladder.
        let even = quantizer.global_scale.get() & !1;
        let doubled = (
            even >= 2,
            GlobalScale::new((even / 2).max(1)),
            QuantLf::new(quantizer.quant_lf.get().saturating_mul(2)),
            HfMul::new(quantizer.hf_mul.get().saturating_mul(2)),
        );
        if let (true, Ok(global_scale), Ok(quant_lf), Ok(baseline)) = doubled {
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
        if self.field.is_some() {
            mul_lattice(self.baseline, self.refine_only)
        } else {
            vec![self.baseline]
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
}

impl HfQuantizers {
    /// Builds every `(transform, HfMul)` quantizer a frame can ask for: the
    /// square vocabulary crossed with `muls` (the adaptive-quantization
    /// lattice around `baseline`, or just `[baseline]` with the field off).
    fn new(global_scale: u32, baseline: HfMul, muls: &[HfMul]) -> Result<Self> {
        let mut by_key = Vec::with_capacity(SQUARE_TRANSFORMS.len() * muls.len());
        for transform in SQUARE_TRANSFORMS {
            for &mul in muls {
                by_key.push((
                    (transform, mul.get()),
                    HfQuantizer::new(
                        transform,
                        global_scale,
                        mul.get(),
                        NEUTRAL_QM_SCALE,
                        NEUTRAL_QM_SCALE,
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
            *slot = if mean > 0.0 { 16.0 / mean } else { 0.0 };
        }
        Ok(Self {
            by_key,
            baseline,
            lambda,
        })
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
    PreparedFrame::from_xyb(
        frame.width(),
        frame.height(),
        x,
        y,
        b,
        frame.is_grayscale(),
    )
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
}

/// Owned forward coefficients for one varblock (Opt-V transform cache).
///
/// Channel order matches [`gather_square`]: 0 = X, 1 = Y, 2 = B.
#[derive(Clone)]
struct VarblockForward {
    coeffs: [Vec<f32>; NUM_CHANNELS],
}

/// Key for a candidate forward: pixel origin + transform type.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct ForwardKey {
    px: u32,
    py: u32,
    transform: u8,
}

/// Cross-probe / within-probe cache of quantizer-independent forward DCTs.
///
/// Built lazily: cover search and post-cover CfL/quantize all hit the same
/// map. The rate loop keeps one cache across quantizer probes so a given
/// (origin, transform) is transformed at most once per request.
#[derive(Default)]
pub(crate) struct CandidateForwardCache {
    entries: std::collections::HashMap<ForwardKey, VarblockForward>,
    /// Times a probe reused a previously computed forward.
    hits: u64,
    /// Times a forward was computed for the first time.
    misses: u64,
}

impl CandidateForwardCache {
    fn new() -> Self {
        Self::default()
    }

    /// Hit count for Opt-V2 rate-loop telemetry.
    pub(crate) fn hits(&self) -> u64 {
        self.hits
    }

    /// Miss count for Opt-V2 rate-loop telemetry.
    pub(crate) fn misses(&self) -> u64 {
        self.misses
    }

    fn key(transform: TransformType, px: u32, py: u32) -> ForwardKey {
        ForwardKey {
            px,
            py,
            transform: transform as u8,
        }
    }

    /// Returns cached coefficients, computing them on first use.
    fn get_or_insert(
        &mut self,
        frame: &PreparedFrame,
        transform: TransformType,
        px: u32,
        py: u32,
        scratch: &mut ForwardScratch,
    ) -> Result<&VarblockForward> {
        let key = Self::key(transform, px, py);
        if self.entries.contains_key(&key) {
            self.hits = self.hits.saturating_add(1);
        } else {
            let side = forward_square(frame, transform, px, py, scratch)?;
            let cells = side * side;
            let fwd = VarblockForward {
                coeffs: core::array::from_fn(|channel| {
                    scratch
                        .coeffs
                        .get(channel)
                        .map(|c| c.get(..cells).unwrap_or(&[]).to_vec())
                        .unwrap_or_default()
                }),
            };
            let n_f32 = fwd.coeffs.iter().map(Vec::len).sum::<usize>();
            diagnostics::note_candidate_forward(n_f32);
            self.entries.insert(key, fwd);
            self.misses = self.misses.saturating_add(1);
        }
        self.entries
            .get(&key)
            .ok_or(PolicyError::Unsupported {
                what: "a missing forward-cache entry after insert",
            })
    }
}

/// Resolves forward coefficients for every selected varblock via `cache`.
fn forward_selected(
    frame: &PreparedFrame,
    varblocks: &[VarblockDecision],
    origin: (u32, u32),
    cache: &mut CandidateForwardCache,
    scratch: &mut ForwardScratch,
) -> Result<Vec<VarblockForward>> {
    let (x0, y0) = origin;
    let mut out = Vec::with_capacity(varblocks.len());
    for vb in varblocks {
        let px = x0 + vb.origin.bx() * 8;
        let py = y0 + vb.origin.by() * 8;
        let fwd = cache.get_or_insert(frame, vb.transform, px, py, scratch)?;
        let n_f32 = fwd.coeffs.iter().map(Vec::len).sum::<usize>();
        diagnostics::note_selected_forward_clone(n_f32);
        out.push(fwd.clone());
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
        self.d_y_lf.resize(n * n, 0.0);
        self.d_y_hf.resize(cells, 0.0);
        self.lf_scratch.resize(n * n, 0.0);
        self.chroma_targets.resize(cells, 0.0);
        self.d_y_lf.fill(0.0);
        self.d_y_hf.fill(0.0);
        self.lf_scratch.fill(0.0);
        self.chroma_targets.fill(0.0);
    }
}

/// Quantizes one square varblock: LF samples into `lf_planes`, HF coefficients
/// into the three channel slices (length `side*side` each). Y first, because
/// X and B decorrelate against the reconstructed `dY` (I.6), never the source Y.
#[allow(clippy::too_many_arguments)]
fn quantize_square_varblock(
    coeffs: &[Vec<f32>; NUM_CHANNELS],
    transform: TransformType,
    lf_quant: &LfQuantizer,
    hf_quant: &HfQuantizer,
    cfl: VarblockCfl,
    scratch: &mut TransformScratch,
    qscratch: &mut QuantScratch,
    bx: u32,
    by: u32,
    lf_width: u32,
    lf_planes: &mut [Vec<i32>; NUM_CHANNELS],
    quant: &mut [i32],
) -> Result<()> {
    let n = transform.block_dims().0;
    let side = transform.sample_cols();
    let cells = side * side;
    if quant.len() < cells * NUM_CHANNELS {
        return Err(PolicyError::Unsupported {
            what: "a coefficient arena slice shorter than three full channels",
        });
    }
    quant[..cells * NUM_CHANNELS].fill(0);
    qscratch.resize(n, cells);

    // Channel layout in `quant`: X | Y | B, each `cells` long.
    let (qx, rest) = quant.split_at_mut(cells);
    let (qy, qb) = rest.split_at_mut(cells);

    // --- Y (channel 1): independent, and everything else needs it ---
    let y_coeff = coeffs.get(1).ok_or(PolicyError::Unsupported {
        what: "the Y coefficient channel",
    })?;
    lf_samples_of(y_coeff, transform, n, side, scratch, &mut qscratch.lf_scratch)?;
    for idx in 0..n * n {
        let q = lf_quant.quantize(qscratch.lf_scratch.get(idx).copied().unwrap_or(0.0), 1)?;
        set_lf(
            lf_planes,
            1,
            bx + u32::try_from(idx % n).unwrap_or(0),
            by + u32::try_from(idx / n).unwrap_or(0),
            lf_width,
            q,
        );
        if let Some(slot) = qscratch.d_y_lf.get_mut(idx) {
            *slot = lf_quant.reconstruct(q, 1);
        }
    }
    // Phase-2: final HF quant as contiguous lanes (Y, then chroma with CfL).
    hf_quant.quantize_lane(1, y_coeff, qy, side, n, true)?;
    for cell in 0..cells {
        if is_llf_cell(cell, side, n) {
            continue;
        }
        let q = qy.get(cell).copied().unwrap_or(0);
        if let Some(slot) = qscratch.d_y_hf.get_mut(cell) {
            *slot = hf_quant.reconstruct(q, 1, cell);
        }
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
            set_lf(
                lf_planes,
                channel,
                bx + u32::try_from(idx % n).unwrap_or(0),
                by + u32::try_from(idx / n).unwrap_or(0),
                lf_width,
                q,
            );
        }
        qscratch.chroma_targets.resize(cells, 0.0);
        for cell in 0..cells {
            let t = if is_llf_cell(cell, side, n) {
                0.0
            } else {
                coeff.get(cell).copied().unwrap_or(0.0)
                    - k_hf * qscratch.d_y_hf.get(cell).copied().unwrap_or(0.0)
            };
            if let Some(slot) = qscratch.chroma_targets.get_mut(cell) {
                *slot = t;
            }
        }
        hf_quant.quantize_lane(channel, &qscratch.chroma_targets, out, side, n, true)?;
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
/// `forwards` must align with `maps` (one forward per selected varblock).
fn estimate_cfl(
    geometry: &VardctGeometry,
    maps: &[&[VarblockDecision]],
    forwards: &[&[VarblockForward]],
    lf_quant: &LfQuantizer,
    hf_quants: &HfQuantizers,
    enabled: bool,
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
    // Detect grayscale from empty chroma energy in the first forward sample if
    // any group is empty; the caller still passes enable_cfl=false for grey.
    let mut lf_x = CflSamples::default();
    let mut lf_b = CflSamples::default();
    let mut hf_groups = Vec::new();
    let mut llf_scratch = TransformScratch::for_transform(TransformType::Dct32x32);

    for index in 0..geometry.num_lf_groups() {
        let id = LfGroupId::new(u32::try_from(index).unwrap_or(u32::MAX));
        let tiles = geometry
            .lf_group_cfl_tiles(id)
            .ok_or(PolicyError::Unsupported {
                what: "an LF group outside the frame's grid",
            })?;
        let tile_count = usize::try_from(tiles.area()).unwrap_or(0);
        let mut group = HfCflSamples {
            tiles,
            x: (0..tile_count).map(|_| CflSamples::default()).collect(),
            b: (0..tile_count).map(|_| CflSamples::default()).collect(),
        };
        let map = maps
            .get(usize::try_from(index).unwrap_or(usize::MAX))
            .copied()
            .ok_or(PolicyError::Unsupported {
                what: "a missing block map for an LF group",
            })?;
        let group_fwd = forwards
            .get(usize::try_from(index).unwrap_or(usize::MAX))
            .copied()
            .ok_or(PolicyError::Unsupported {
                what: "a missing forward cache for an LF group",
            })?;
        if map.len() != group_fwd.len() {
            return Err(PolicyError::Unsupported {
                what: "a forward cache length that does not match the block map",
            });
        }

        for (vb, fwd) in map.iter().zip(group_fwd.iter()) {
            let transform = vb.transform;
            let n = transform.block_dims().0;
            let side = transform.sample_cols();
            let cells = side * side;
            let hf_quant = hf_quants.get(transform, vb.hf_mul)?;
            let tile = usize::try_from(
                u64::from(vb.origin.by() / 8) * u64::from(tiles.width)
                    + u64::from(vb.origin.bx() / 8),
            )
            .unwrap_or(usize::MAX);

            // LF: the varblock's n*n LF samples, chroma against reconstructed dY.
            let mut y_lf = vec![0.0f32; n * n];
            let mut x_lf = vec![0.0f32; n * n];
            let mut b_lf = vec![0.0f32; n * n];
            lf_samples_of(
                fwd.coeffs.get(1).map_or(&[][..], Vec::as_slice),
                transform,
                n,
                side,
                &mut llf_scratch,
                &mut y_lf,
            )?;
            lf_samples_of(
                fwd.coeffs.get(0).map_or(&[][..], Vec::as_slice),
                transform,
                n,
                side,
                &mut llf_scratch,
                &mut x_lf,
            )?;
            lf_samples_of(
                fwd.coeffs.get(2).map_or(&[][..], Vec::as_slice),
                transform,
                n,
                side,
                &mut llf_scratch,
                &mut b_lf,
            )?;
            for idx in 0..n * n {
                let y = y_lf.get(idx).copied().unwrap_or(0.0);
                let q = lf_quant.quantize(y, 1)?;
                let d_y = lf_quant.reconstruct(q, 1);
                lf_x.push(x_lf.get(idx).copied().unwrap_or(0.0), y, d_y, 0);
                lf_b.push(b_lf.get(idx).copied().unwrap_or(0.0), y, d_y, 0);
            }

            // HF: every non-LLF coefficient, into the varblock's tile. The
            // sample's cell is folded onto the 8x8 frequency grid (identity
            // for DCT8x8), because `refine_hf_factor` scores every tile with
            // the DCT8x8 quantizer as the common scale and its matrix has no
            // entries beyond 8x8.
            let fold = side / 8;
            let cx = fwd.coeffs.get(0).map_or(&[][..], Vec::as_slice);
            let cy = fwd.coeffs.get(1).map_or(&[][..], Vec::as_slice);
            let cb = fwd.coeffs.get(2).map_or(&[][..], Vec::as_slice);
            diagnostics::with_choose_stage(diagnostics::ChooseStage::CflY, || {
                for cell in 0..cells {
                    if is_llf_cell(cell, side, n) {
                        continue;
                    }
                    let cell8 = (cell / side / fold.max(1)) * 8 + (cell % side / fold.max(1));
                    let y = cy.get(cell).copied().unwrap_or(0.0);
                    let q_y = hf_quant.choose(y, 1, cell)?;
                    let d_y = hf_quant.reconstruct(q_y, 1, cell);
                    if let Some(t) = group.x.get_mut(tile) {
                        t.push(cx.get(cell).copied().unwrap_or(0.0), y, d_y, cell8);
                    }
                    if let Some(t) = group.b.get_mut(tile) {
                        t.push(cb.get(cell).copied().unwrap_or(0.0), y, d_y, cell8);
                    }
                }
                Ok::<(), PolicyError>(())
            })?;
        }
        hf_groups.push(group);
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

    let mut groups = Vec::with_capacity(hf_groups.len());
    for group in hf_groups {
        let mut x = Vec::with_capacity(group.x.len());
        let mut b = Vec::with_capacity(group.b.len());
        for tile in &group.x {
            x.push(CflFactor::new(refine_hf_factor(
                tile,
                0.0,
                0,
                hf_quants.baseline(TransformType::Dct8x8)?,
            )?));
        }
        for tile in &group.b {
            b.push(CflFactor::new(refine_hf_factor(
                tile,
                1.0,
                2,
                hf_quants.baseline(TransformType::Dct8x8)?,
            )?));
        }
        groups.push(CflGrid::new(group.tiles, x, b)?);
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
    for sample in &samples.samples {
        let q = quantizer.choose(
            sample.source - k * sample.reconstructed_y,
            channel,
            sample.cell,
        )?;
        bits = bits.saturating_add(residual_bits(q));
        if let Some(cut) = cutoff {
            // Strict > so equal residual+signalling costs still finish for
            // magnitude tie-breaks toward neutral/smaller factor.
            if bits > cut {
                return Ok(None);
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
}

/// Quantizes one LF group's **selected** varblocks, in their `BlockInfo` order.
///
/// `forwards` are the precomputed coefficient arrays from [`forward_selected`]
/// (same length and order as `varblocks`). HF coefficients for every varblock
/// share one group arena ([`VarblockCoefficients::from_arena`]) so entropy
/// alternatives do not re-allocate per-varblock coefficient boxes.
#[allow(clippy::too_many_arguments)]
fn quantize_group(
    lf_quant: &LfQuantizer,
    hf_quants: &HfQuantizers,
    correlation: &LfCorrelationDecision,
    cfl: &CflGrid,
    varblocks: &[VarblockDecision],
    forwards: &[VarblockForward],
    blocks: jpxl_encode::vardct::BlockGrid,
) -> Result<QuantizedGroup> {
    if varblocks.len() != forwards.len() {
        return Err(PolicyError::Unsupported {
            what: "a forward cache length that does not match the block map",
        });
    }
    let cells = usize::try_from(blocks.area()).unwrap_or(0);
    let mut lf_planes: [Vec<i32>; NUM_CHANNELS] = core::array::from_fn(|_| vec![0i32; cells]);
    let mut tscratch = TransformScratch::for_transform(TransformType::Dct32x32);
    let mut qscratch = QuantScratch::new();

    // One arena: for each varblock, three channels of `side*side` i32s.
    let mut arena_cap = 0usize;
    for vb in varblocks {
        let side = vb.transform.sample_cols();
        arena_cap = arena_cap
            .saturating_add(side.saturating_mul(side).saturating_mul(NUM_CHANNELS));
    }
    let mut arena = vec![0i32; arena_cap];
    let mut starts: Vec<[usize; NUM_CHANNELS]> = Vec::with_capacity(varblocks.len());
    let mut cursor = 0usize;

    for (vb, fwd) in varblocks.iter().zip(forwards.iter()) {
        let transform = vb.transform;
        let side = transform.sample_cols();
        let ch_cells = side * side;
        let span = ch_cells.saturating_mul(NUM_CHANNELS);
        let end = cursor.saturating_add(span);
        let slot = arena.get_mut(cursor..end).ok_or(PolicyError::Unsupported {
            what: "a coefficient arena that ran short of capacity",
        })?;
        let (bx, by) = (vb.origin.bx(), vb.origin.by());
        let factors = varblock_cfl(correlation, cfl, bx, by);
        quantize_square_varblock(
            &fwd.coeffs,
            transform,
            lf_quant,
            hf_quants.get(transform, vb.hf_mul)?,
            factors,
            &mut tscratch,
            &mut qscratch,
            bx,
            by,
            blocks.width,
            &mut lf_planes,
            slot,
        )?;
        starts.push([cursor, cursor + ch_cells, cursor + ch_cells * 2]);
        cursor = end;
    }

    let arena: std::sync::Arc<[i32]> = arena.into();
    let mut coefficients = Vec::with_capacity(varblocks.len());
    for (vb, channel_starts) in varblocks.iter().zip(starts) {
        coefficients.push(VarblockCoefficients::from_arena(
            vb.transform,
            std::sync::Arc::clone(&arena),
            channel_starts,
        )?);
    }

    Ok(QuantizedGroup {
        lf: LfQuantPlanes::new(blocks, lf_planes)?,
        coefficients,
    })
}

/// The share of §4.3's `metadata_bits` every varblock pays: its two
/// `BlockInfo` Modular samples and three `non_zeros` symbols. Small, because
/// under the current single-context entropy model a run of identical samples
/// is nearly free.
const PER_VARBLOCK_BITS: f64 = 2.0;

/// The extra `metadata_bits` a non-DCT8x8 varblock pays: its DctSelect sample
/// breaks the all-zeros run G.2.4's default map codes for free, twice (the
/// gradient residual entering and leaving the value). Measured on the current
/// coder at ~8 bits per merged block net; charged higher so a merge must be
/// paid for by real coefficient savings, not a rounding whim. Re-derive when
/// slice 18 trains the entropy model.
const NON_DCT8X8_SIGNAL_BITS: f64 = 32.0;

/// The `metadata_bits` a non-baseline `HfMul` pays: like DctSelect, its `mul`
/// sample breaks a constant run in `BlockInfo`'s second row. Charged at the
/// same order as a DctSelect transition but lower — the lattice keeps the
/// values small and repetitive.
fn mul_signal_bits(hf_mul: HfMul, baseline: HfMul) -> f64 {
    if hf_mul == baseline { 0.0 } else { 8.0 }
}

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
    cache: &mut CandidateForwardCache,
    scratch: &mut ForwardScratch,
    d_y_hf: &mut [f32],
    cutoff: Option<f64>,
) -> Result<Option<f64>> {
    let fwd = cache.get_or_insert(frame, transform, px, py, scratch)?;
    let side = transform.sample_cols();
    let n = transform.block_dims().0;
    let cells = side * side;
    let hf_quant = hf_quants.get(transform, hf_mul)?;
    // One squared coefficient unit is `side^2` squared sample units (the
    // forward transforms are not Parseval; see [`HfQuantizers::lambda`]).
    #[allow(
        clippy::cast_precision_loss,
        reason = "side is at most 32; exact in f64"
    )]
    let to_sample_domain = (side * side) as f64;
    let cy = fwd.coeffs.get(1).map_or(&[][..], Vec::as_slice);
    let cx = fwd.coeffs.get(0).map_or(&[][..], Vec::as_slice);
    let cb = fwd.coeffs.get(2).map_or(&[][..], Vec::as_slice);
    let mut bits = 0u64;
    let mut weighted_sse = 0.0f64;
    let check = |bits: u64, weighted_sse: f64| -> bool {
        if let Some(cut) = cutoff {
            #[allow(
                clippy::cast_precision_loss,
                reason = "bit counts stay far inside f64's exact integer range"
            )]
            let partial = bits as f64 + weighted_sse;
            // Ties keep the split, so >= is a correct prune.
            partial >= cut
        } else {
            false
        }
    };
    for cell in 0..cells {
        if is_llf_cell(cell, side, n) {
            continue;
        }
        let coeff = cy.get(cell).copied().unwrap_or(0.0);
        let q = hf_quant.choose(coeff, 1, cell)?;
        bits = bits.saturating_add(residual_bits(q));
        let recon = hf_quant.reconstruct(q, 1, cell);
        weighted_sse += hf_quants.lambda[1] * to_sample_domain * f64::from(recon - coeff).powi(2);
        if let Some(slot) = d_y_hf.get_mut(cell) {
            *slot = recon;
        }
        if check(bits, weighted_sse) {
            return Ok(None);
        }
    }
    for &(channel, plane) in &[(0usize, cx), (2usize, cb)] {
        let k = if channel == 0 { 0.0 } else { 1.0 };
        for cell in 0..cells {
            if is_llf_cell(cell, side, n) {
                continue;
            }
            let target =
                plane.get(cell).copied().unwrap_or(0.0) - k * d_y_hf.get(cell).copied().unwrap_or(0.0);
            let q = hf_quant.choose(target, channel, cell)?;
            bits = bits.saturating_add(residual_bits(q));
            let recon = hf_quant.reconstruct(q, channel, cell);
            weighted_sse += hf_quants.lambda.get(channel).copied().unwrap_or(0.0)
                * to_sample_domain
                * f64::from(recon - target).powi(2);
            if check(bits, weighted_sse) {
                return Ok(None);
            }
        }
    }
    #[allow(
        clippy::cast_precision_loss,
        reason = "bit counts stay far inside f64's exact integer range"
    )]
    Ok(Some(bits as f64 + weighted_sse))
}

#[allow(clippy::too_many_arguments)]
fn block_cost(
    frame: &PreparedFrame,
    hf_quants: &HfQuantizers,
    transform: TransformType,
    hf_mul: HfMul,
    px: u32,
    py: u32,
    cache: &mut CandidateForwardCache,
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

/// The quadtree cover of one aligned `size`-atom region, choosing at each level
/// between one square transform and four sub-quadrants by exact R-D within the
/// hierarchy. Clipped and pass-group-straddling regions are forced to split;
/// aligned placement keeps every block inside one pass group.
#[allow(clippy::too_many_arguments)]
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
    cache: &mut CandidateForwardCache,
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
        let cost = block_cost(
            frame,
            hf_quants,
            TransformType::Dct8x8,
            hf_mul,
            x0 + bx * 8,
            y0 + by * 8,
            cache,
            scratch,
            d_y_hf,
        )? + PER_VARBLOCK_BITS
            + mul_signal_bits(hf_mul, aq.baseline);
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
        let (c, mut b) = tile_region(
            frame, hf_quants, grid, qx, qy, half, aq, x0, y0, cache, scratch, d_y_hf,
        )?;
        split_cost += c;
        split_blocks.append(&mut b);
    }

    let fits = bx + size <= grid.width && by + size <= grid.height;
    let single = match square_transform(size) {
        Some(transform) if fits => {
            let hf_mul = aq.mul_for_footprint(x0 / 8 + bx, y0 / 8 + by, size, size);
            let fixed = PER_VARBLOCK_BITS
                + NON_DCT8X8_SIGNAL_BITS
                + mul_signal_bits(hf_mul, aq.baseline);
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

/// Selects one LF group's varblock tiling by the hierarchical quadtree solver,
/// returned in `BlockInfo` (greedy earliest-uncovered raster) order.
fn select_blocks(
    frame: &PreparedFrame,
    hf_quants: &HfQuantizers,
    grid: jpxl_encode::vardct::BlockGrid,
    origin: (u32, u32),
    aq: &AqSetup,
    cache: &mut CandidateForwardCache,
    scratch: &mut ForwardScratch,
) -> Result<Vec<VarblockDecision>> {
    let (x0, y0) = origin;
    let mut d_y_hf = vec![0.0f32; 32 * 32];
    let mut blocks = Vec::new();
    let mut sby = 0u32;
    while sby < grid.height {
        let mut sbx = 0u32;
        while sbx < grid.width {
            let (_, mut region) = tile_region(
                frame,
                hf_quants,
                grid,
                sbx,
                sby,
                4,
                aq,
                x0,
                y0,
                cache,
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
    let frame = PreparedFrame::from_srgb8(width, height, rgb)?;
    let atlas = AnalysisAtlas::analyze(&frame);
    rate::search_frame(&frame, &atlas, request, target)
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
        let image = decode(&serial, &Limits::default()).expect("decodes");
        assert_eq!((image.width, image.height), (width, height));
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
