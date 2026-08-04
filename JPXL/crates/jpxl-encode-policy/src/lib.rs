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
//! * **no filter planning** — gaborish and EPF are off (milestone 9).
//!
//! Each of those is where the compression is; what exists here is a correct
//! pipeline for them to improve.

pub mod analysis;
pub mod block;
pub mod error;
pub mod quantize;
pub mod rate;
pub mod request;
pub mod source;

use jpxl_core::forward::{
    CoeffView, CoeffViewMut, SampleView, SampleViewMut, TransformScratch, forward_varblock_into,
    lf_from_llf_into,
};
use jpxl_core::varblock::TransformType;
use jpxl_encode::vardct::headers::{NEUTRAL_QM_SCALE, VARDCT_GROUP_SIZE_SHIFT};
use jpxl_encode::vardct::ids::{CflFactor, ClusterId, LfGroupId, PreContextId, PresetId};
use jpxl_encode::vardct::plan::{
    CflGrid, EmissionPlan, EntropyModelPlan, EntropyPlan, FrameDecision, HfBlockContextPlan,
    HfPassEntropyPlan, HistogramPlan, HybridUintPlan, LfCorrelationDecision, LfDecision,
    LfGroupPlan, LfQuantPlanes, OrderSet, QuantizedFrameIr, QuantizedLfGroup, QuantizerDecision,
    RestorationDecision, SectionLayout, SharpnessGrid, SpatialPlan, VarblockCoefficients,
};
use jpxl_encode::vardct::{ValidatedEmissionPlan, VardctGeometry, census_frame, validate};

use quantize::{
    CflAccumulator, DCT8X8_CELLS, DEFAULT_COLOUR_FACTOR, HfQuantizer, LfQuantizer, NUM_CHANNELS,
    cfl_multiplier,
};

pub use analysis::{AnalysisAtlas, AtomGrid};
pub use error::{PolicyError, Result};
pub use rate::{
    LadderSearch, QuantizerChoice, RateOutcome, RatePhase, RateStep, Rung, search_frame,
};
pub use request::{
    CoverMode, EncodeRequest, RateSearchBudget, RateTarget, RateTolerance, SearchBudget,
};
pub use source::PreparedFrame;

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
    plan_at_with_cfl(frame, atlas, request, quantizer, true)
}

/// [`plan_at`] with the Slice-15 search switch exposed for regression tests.
///
/// Production always enables CfL. The disabled arm exists only to preserve a
/// byte-for-byte pre-slice-15 baseline for the exit evidence; it is not a
/// public encoder knob.
fn plan_at_with_cfl(
    frame: &PreparedFrame,
    atlas: &AnalysisAtlas,
    request: &EncodeRequest,
    quantizer: QuantizerChoice,
    enable_cfl: bool,
) -> Result<ValidatedEmissionPlan> {
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

    let lf_quant = LfQuantizer::new(
        quantizer.global_scale.get(),
        quantizer.quant_lf.get(),
        LfDecision::vardct_neutral().extra_precision,
    );
    let hf_quant = HfQuantizer::new(
        TransformType::Dct8x8,
        quantizer.global_scale.get(),
        quantizer.hf_mul.get(),
        NEUTRAL_QM_SCALE,
        NEUTRAL_QM_SCALE,
    )?;
    let cfl = estimate_cfl(frame, &geometry, &lf_quant, &hf_quant, enable_cfl)?;

    let mut lf_groups = Vec::new();
    let mut quantized = Vec::new();
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
            CoverMode::FixedDct8x8 => block::fixed_dct8x8(blocks, quantizer.hf_mul)?,
        };
        let group_cfl = cfl
            .groups
            .get(usize::try_from(index).unwrap_or(usize::MAX))
            .ok_or(PolicyError::Unsupported {
                what: "a missing CfL grid for an LF group",
            })?;
        let quantized_group = quantize_lf_group(
            frame,
            &lf_quant,
            &hf_quant,
            &cfl.correlation,
            group_cfl,
            (rect.x0, rect.y0),
            blocks,
        )?;

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
            global_scale: quantizer.global_scale,
            quant_lf: quantizer.quant_lf,
        },
        lf,
        restoration: RestorationDecision::default(),
        lf_groups: lf_groups.into_boxed_slice(),
    };

    // The entropy model is chosen in two steps because the census is a
    // function of the plan: a provisional plan carries the clustering and the
    // hybrid-uint configuration, `census_frame` walks it, and the real
    // histograms replace the provisional ones. The walk lives in `jpxl-encode`
    // so that the counts trained here and the symbols emitted there cannot
    // come from two different traversals.
    let provisional = EmissionPlan {
        spatial,
        quantized: QuantizedFrameIr {
            lf_groups: quantized.into_boxed_slice(),
        },
        entropy: entropy_plan(&geometry, placeholder_histograms())?,
        sections: SectionLayout::for_geometry(&geometry),
    };
    let census = census_frame(&provisional, &geometry)?;

    let nb_block_ctx = HfBlockContextPlan::Default.nb_block_ctx();
    let mut counts = vec![Vec::<u32>::new(); NUM_CLUSTERS];
    for ctx in 0..census.len() {
        let id = PreContextId::new(u32::try_from(ctx).unwrap_or(u32::MAX));
        let Some(histogram) = census.histogram(id) else {
            continue;
        };
        let cluster = usize::from(cluster_of(ctx as u64, nb_block_ctx));
        let Some(slot) = counts.get_mut(cluster) else {
            continue;
        };
        for (value, count) in histogram.iter() {
            let index = usize::try_from(value).unwrap_or(usize::MAX);
            if index >= slot.len() {
                slot.resize(index + 1, 0);
            }
            if let Some(cell) = slot.get_mut(index) {
                *cell = cell.saturating_add(count);
            }
        }
    }
    let histograms = counts
        .into_iter()
        .map(|mut c| {
            // A cluster the frame never used still needs a legal distribution:
            // C.2.5 has no encoding for "no mass anywhere".
            if c.iter().all(|&v| v == 0) {
                c = vec![1u32];
            }
            HistogramPlan::new(c)
        })
        .collect::<jpxl_encode::vardct::PlanResult<Vec<_>>>()?;

    let plan = EmissionPlan {
        entropy: entropy_plan(&geometry, histograms)?,
        ..provisional
    };
    Ok(validate(plan)?)
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
        self.regression.add(regression_y, source);
        self.samples.push(CflSample {
            source,
            reconstructed_y,
            cell,
        });
    }
}

/// The samples of one LF group's 64x64 CfL tiles.
struct HfCflSamples {
    tiles: jpxl_encode::vardct::BlockGrid,
    x: Vec<CflSamples>,
    b: Vec<CflSamples>,
}

/// Regression first, then integer refinement in exactly the factor space I.6
/// consumes.
///
/// The least-squares seed is trained in the unquantized coefficient domain,
/// `Σ(Y·C) / Σ(Y²)`, separately for LF and each HF tile. Its integer seed is
/// then refined over nearby stored factors using reconstructed `dY`, the value
/// the decoder actually adds back, and quantizing the chroma residual through
/// [`LfQuantizer`] or [`HfQuantizer`] (including I.5.3's quantization bias).
/// Grayscale is a hard source-domain no-op so its pre-slice-15 stream is
/// byte-identical rather than merely visually identical.
fn estimate_cfl(
    frame: &PreparedFrame,
    geometry: &VardctGeometry,
    lf_quant: &LfQuantizer,
    hf_quant: &HfQuantizer,
    enabled: bool,
) -> Result<CflEstimate> {
    if !enabled || frame.is_grayscale() {
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

    let mut lf_x = CflSamples::default();
    let mut lf_b = CflSamples::default();
    let mut hf_groups = Vec::new();
    let mut scratch = TransformScratch::for_transform(TransformType::Dct8x8);
    let mut samples = [0.0f32; DCT8X8_CELLS];
    let mut coeffs: [[f32; DCT8X8_CELLS]; NUM_CHANNELS] = [[0.0; DCT8X8_CELLS]; NUM_CHANNELS];

    for index in 0..geometry.num_lf_groups() {
        let id = LfGroupId::new(u32::try_from(index).unwrap_or(u32::MAX));
        let blocks = geometry
            .lf_group_blocks(id)
            .ok_or(PolicyError::Unsupported {
                what: "an LF group outside the frame's grid",
            })?;
        let tiles = geometry
            .lf_group_cfl_tiles(id)
            .ok_or(PolicyError::Unsupported {
                what: "an LF group outside the frame's grid",
            })?;
        let rect = geometry.lf_group_rect(id).ok_or(PolicyError::Unsupported {
            what: "an LF group outside the frame's grid",
        })?;
        let tile_count = usize::try_from(tiles.area()).unwrap_or(0);
        let mut group = HfCflSamples {
            tiles,
            x: (0..tile_count).map(|_| CflSamples::default()).collect(),
            b: (0..tile_count).map(|_| CflSamples::default()).collect(),
        };

        for by in 0..blocks.height {
            for bx in 0..blocks.width {
                transform_block(
                    frame,
                    rect.x0 + bx * 8,
                    rect.y0 + by * 8,
                    &mut samples,
                    &mut coeffs,
                    &mut scratch,
                )?;

                let y_lf = lf_target_of(&coeffs, 1, &mut scratch)?;
                let q_y_lf = lf_quant.quantize(y_lf, 1)?;
                let d_y_lf = lf_quant.reconstruct(q_y_lf, 1);
                lf_x.push(lf_target_of(&coeffs, 0, &mut scratch)?, y_lf, d_y_lf, 0);
                lf_b.push(lf_target_of(&coeffs, 2, &mut scratch)?, y_lf, d_y_lf, 0);

                let tile_index =
                    usize::try_from(u64::from(by / 8) * u64::from(tiles.width) + u64::from(bx / 8))
                        .unwrap_or(usize::MAX);
                for cell in 1..DCT8X8_CELLS {
                    let y = coeffs
                        .get(1)
                        .and_then(|plane| plane.get(cell))
                        .copied()
                        .unwrap_or(0.0);
                    let q_y = hf_quant.choose(y, 1, cell)?;
                    let d_y = hf_quant.reconstruct(q_y, 1, cell);
                    if let Some(tile) = group.x.get_mut(tile_index) {
                        tile.push(
                            coeffs
                                .first()
                                .and_then(|plane| plane.get(cell))
                                .copied()
                                .unwrap_or(0.0),
                            y,
                            d_y,
                            cell,
                        );
                    }
                    if let Some(tile) = group.b.get_mut(tile_index) {
                        tile.push(
                            coeffs
                                .get(2)
                                .and_then(|plane| plane.get(cell))
                                .copied()
                                .unwrap_or(0.0),
                            y,
                            d_y,
                            cell,
                        );
                    }
                }
            }
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
            x.push(CflFactor::new(refine_hf_factor(tile, 0.0, 0, hf_quant)?));
        }
        for tile in &group.b {
            b.push(CflFactor::new(refine_hf_factor(tile, 1.0, 2, hf_quant)?));
        }
        groups.push(CflGrid::new(group.tiles, x, b)?);
    }
    Ok(CflEstimate {
        correlation,
        groups,
    })
}

fn transform_block(
    frame: &PreparedFrame,
    x0: u32,
    y0: u32,
    samples: &mut [f32; DCT8X8_CELLS],
    coeffs: &mut [[f32; DCT8X8_CELLS]; NUM_CHANNELS],
    scratch: &mut TransformScratch,
) -> Result<()> {
    for channel in 0..NUM_CHANNELS {
        gather_block(frame, channel, x0, y0, samples);
        let (Some(view), Some(mut out)) = (
            SampleView::contiguous(samples, 8, 8),
            coeffs
                .get_mut(channel)
                .and_then(|plane| CoeffViewMut::contiguous(plane, 8, 8)),
        ) else {
            return Err(PolicyError::Unsupported {
                what: "an 8x8 block view",
            });
        };
        forward_varblock_into(TransformType::Dct8x8, &view, &mut out, scratch);
    }
    Ok(())
}

/// The candidate integer factors to score: a small window around the
/// least-squares seed, always including the neutral factor `0`.
fn factor_candidates(seed: i32, lo: i32, hi: i32) -> Vec<i32> {
    let mut out = Vec::with_capacity(10);
    for delta in -4i32..=4 {
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
/// CfL does not change the reconstructed value materially — the quantizer
/// re-targets the residual either way — so the axis that actually moves with
/// the factor is *rate*, not distortion. This proxy is the magnitude class of
/// the coefficient: zero is free (it is the overwhelming symbol and the
/// context model prices a run of them near nothing), and a non-zero costs its
/// bit length plus a sign. Minimizing its sum is minimizing residual entropy
/// in the only currency slice 15 can spend without a full re-encode per
/// candidate.
fn residual_bits(q: i32) -> u64 {
    if q == 0 {
        0
    } else {
        u64::from(32 - q.unsigned_abs().leading_zeros()) + 1
    }
}

/// The signaling cost of one stored factor, in the same currency.
///
/// A neutral factor is free: `0` is the default Modular sample, predicts
/// exactly, and (for LF) keeps the whole I.2.3 bundle at its one-bit
/// `all_default`. A non-zero factor must pay its own magnitude, so a tile whose
/// residual saving does not clear that price stays neutral — which is what
/// keeps CfL from ever enlarging weakly-correlated content.
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
    // Explicit I.2.3 costs 51 bits at the defaults used here:
    // all_default=false (1), colour_factor's selector (2), two F16s (32), and
    // the two biased factors (16). The all-default form costs one bit, hence a
    // non-neutral pair must save at least 50 residual bits before it is useful.
    const LF_BUNDLE_EXTRA_BITS: u64 = 50;

    let x_seed = x
        .regression
        .best_factor(0.0, DEFAULT_COLOUR_FACTOR, -128, 127);
    let b_seed = b
        .regression
        .best_factor(1.0, DEFAULT_COLOUR_FACTOR, -128, 127);
    let x_candidates = factor_candidates(x_seed, -128, 127);
    let b_candidates = factor_candidates(b_seed, -128, 127);

    let neutral_cost = lf_residual_cost(x, 0.0, 0, 0, quantizer)?
        .saturating_add(lf_residual_cost(b, 1.0, 0, 2, quantizer)?);
    let mut best = (0i32, 0i32);
    let mut best_cost = neutral_cost;
    for &x_factor in &x_candidates {
        let x_cost = lf_residual_cost(x, 0.0, x_factor, 0, quantizer)?;
        for &b_factor in &b_candidates {
            let mut cost = x_cost.saturating_add(lf_residual_cost(b, 1.0, b_factor, 2, quantizer)?);
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

/// The residual bit-cost of one candidate HF factor over a tile.
fn hf_residual_cost(
    samples: &CflSamples,
    base: f32,
    factor: i32,
    channel: usize,
    quantizer: &HfQuantizer,
) -> Result<u64> {
    let k = cfl_multiplier(base, factor, DEFAULT_COLOUR_FACTOR);
    let mut bits = 0u64;
    for sample in &samples.samples {
        let q = quantizer.choose(
            sample.source - k * sample.reconstructed_y,
            channel,
            sample.cell,
        )?;
        bits = bits.saturating_add(residual_bits(q));
    }
    Ok(bits)
}

/// The best HF factor for one 64x64 tile: the candidate whose residual-plus-
/// signaling bits are lowest, ties resolved toward the neutral factor so an
/// unhelpful tile costs nothing and stays byte-neutral.
fn refine_hf_factor(
    samples: &CflSamples,
    base: f32,
    channel: usize,
    quantizer: &HfQuantizer,
) -> Result<i32> {
    // G.2.4's `XFromY`/`BFromY` are Modular samples, so *our* decoder reads any
    // `i32`. The reference decoder does not: `djxl` (libjxl) stores these as
    // signed bytes, and a factor outside `[-128, 127]` decodes to a different
    // value there than in `jpxl-decode` or `jxl-oxide` — a genuine
    // interoperability constraint, not a coding-cost heuristic. It was pinned
    // with a single-tile boundary probe against `djxl`: `+127`/`-128` agree,
    // `+128`/`-129` diverge. The search space is therefore exactly the wire
    // range the reference honours.
    let seed =
        samples
            .regression
            .best_factor(base, DEFAULT_COLOUR_FACTOR, HF_FACTOR_MIN, HF_FACTOR_MAX);
    let mut best_factor = 0i32;
    let mut best_cost = hf_residual_cost(samples, base, 0, channel, quantizer)?;
    for factor in factor_candidates(seed, HF_FACTOR_MIN, HF_FACTOR_MAX) {
        if factor == 0 {
            continue;
        }
        let cost = hf_residual_cost(samples, base, factor, channel, quantizer)?
            .saturating_add(factor_bits(factor));
        if cost < best_cost || (cost == best_cost && factor.abs() < best_factor.abs()) {
            best_cost = cost;
            best_factor = factor;
        }
    }
    Ok(best_factor)
}

/// One LF group's quantized integers.
struct QuantizedGroup {
    lf: LfQuantPlanes,
    coefficients: Vec<VarblockCoefficients>,
}

/// Forward-transforms and quantizes every 8x8 block of one LF group.
///
/// The order of operations is forced by I.6: `Y` first, because `B`'s target is
/// `B - kB * dY` against the **reconstructed** `dY`, not the source one. `X`
/// has `kX == 0` and is independent.
fn quantize_lf_group(
    frame: &PreparedFrame,
    lf_quant: &LfQuantizer,
    hf_quant: &HfQuantizer,
    correlation: &LfCorrelationDecision,
    cfl: &CflGrid,
    origin: (u32, u32),
    blocks: jpxl_encode::vardct::BlockGrid,
) -> Result<QuantizedGroup> {
    let (x0, y0) = origin;
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
    let cells = usize::try_from(blocks.area()).unwrap_or(0);
    let mut lf_planes: [Vec<i32>; NUM_CHANNELS] = core::array::from_fn(|_| vec![0i32; cells]);
    let mut coefficients = Vec::with_capacity(cells);

    let mut scratch = TransformScratch::for_transform(TransformType::Dct8x8);
    let mut samples = [0.0f32; DCT8X8_CELLS];
    let mut coeffs: [[f32; DCT8X8_CELLS]; NUM_CHANNELS] = [[0.0; DCT8X8_CELLS]; NUM_CHANNELS];

    for by in 0..blocks.height {
        for bx in 0..blocks.width {
            transform_block(
                frame,
                x0 + bx * 8,
                y0 + by * 8,
                &mut samples,
                &mut coeffs,
                &mut scratch,
            )?;

            let cell_index =
                usize::try_from(u64::from(by) * u64::from(blocks.width) + u64::from(bx))
                    .unwrap_or(0);
            let mut quant: [[i32; DCT8X8_CELLS]; NUM_CHANNELS] = [[0; DCT8X8_CELLS]; NUM_CHANNELS];
            let mut reconstructed_y = [0.0f32; DCT8X8_CELLS];

            // --- Y (channel 1): independent, and everything else needs it ---
            let y_lf_target = lf_target_of(&coeffs, 1, &mut scratch)?;
            let q_y_lf = lf_quant.quantize(y_lf_target, 1)?;
            let d_y_lf = lf_quant.reconstruct(q_y_lf, 1);
            if let Some(slot) = lf_planes.get_mut(1).and_then(|p| p.get_mut(cell_index)) {
                *slot = q_y_lf;
            }
            for cell in 1..DCT8X8_CELLS {
                let target = coeffs
                    .get(1)
                    .and_then(|c| c.get(cell))
                    .copied()
                    .unwrap_or(0.0);
                let q = hf_quant.choose(target, 1, cell)?;
                if let Some(slot) = quant.get_mut(1).and_then(|c| c.get_mut(cell)) {
                    *slot = q;
                }
                if let Some(slot) = reconstructed_y.get_mut(cell) {
                    *slot = hf_quant.reconstruct(q, 1, cell);
                }
            }

            let tile_index = usize::try_from(
                u64::from(by / 8) * u64::from(cfl.tiles().width) + u64::from(bx / 8),
            )
            .unwrap_or(usize::MAX);
            let k_x_hf = cfl_multiplier(
                correlation.base_correlation_x,
                cfl.x_from_y()
                    .get(tile_index)
                    .copied()
                    .unwrap_or_default()
                    .get(),
                correlation.colour_factor,
            );
            let k_b_hf = cfl_multiplier(
                correlation.base_correlation_b,
                cfl.b_from_y()
                    .get(tile_index)
                    .copied()
                    .unwrap_or_default()
                    .get(),
                correlation.colour_factor,
            );

            // --- X and B: I.6 reconstructs X = dX + kX*dY, B = dB + kB*dY ---
            for &channel in &[0usize, 2usize] {
                let k_lf = if channel == 0 { k_x_lf } else { k_b_lf };
                let k_hf = if channel == 0 { k_x_hf } else { k_b_hf };
                let lf_target = lf_target_of(&coeffs, channel, &mut scratch)? - k_lf * d_y_lf;
                let q_lf = lf_quant.quantize(lf_target, channel)?;
                if let Some(slot) = lf_planes
                    .get_mut(channel)
                    .and_then(|p| p.get_mut(cell_index))
                {
                    *slot = q_lf;
                }
                for cell in 1..DCT8X8_CELLS {
                    let target = coeffs
                        .get(channel)
                        .and_then(|c| c.get(cell))
                        .copied()
                        .unwrap_or(0.0)
                        - k_hf * reconstructed_y.get(cell).copied().unwrap_or(0.0);
                    let q = hf_quant.choose(target, channel, cell)?;
                    if let Some(slot) = quant.get_mut(channel).and_then(|c| c.get_mut(cell)) {
                        *slot = q;
                    }
                }
            }

            coefficients.push(VarblockCoefficients::new(
                TransformType::Dct8x8,
                [
                    quant.first().copied().unwrap_or([0; DCT8X8_CELLS]).to_vec(),
                    quant.get(1).copied().unwrap_or([0; DCT8X8_CELLS]).to_vec(),
                    quant.get(2).copied().unwrap_or([0; DCT8X8_CELLS]).to_vec(),
                ],
            )?);
        }
    }

    Ok(QuantizedGroup {
        lf: LfQuantPlanes::new(blocks, lf_planes)?,
        coefficients,
    })
}

/// I.8's LF sample for one channel's DCT8x8 coefficient array.
///
/// The decoder does not code the LLF cells: it overwrites them from the LF
/// image, through `llf_from_lf`. The encoder therefore has to go the other way
/// — LLF coefficients to LF samples — and that map is `jpxl-core`'s
/// [`lf_from_llf_into`], the proven inverse of the very function the decoder
/// runs. For DCT8x8 it is the identity on a single cell, and calling it anyway
/// is what keeps I.8's `ScaleF` flip point from being re-derived here when
/// milestone 3 adds the larger transforms.
fn lf_target_of(
    coeffs: &[[f32; DCT8X8_CELLS]; NUM_CHANNELS],
    channel: usize,
    scratch: &mut TransformScratch,
) -> Result<f32> {
    let plane = coeffs.get(channel).ok_or(PolicyError::Unsupported {
        what: "a coefficient channel",
    })?;
    let mut lf = [0.0f32; 1];
    let (Some(llf), Some(mut out)) = (
        // The LLF of a DCT8x8 is the single top-left cell of an 8-wide array.
        CoeffView::new(plane, 1, 1, 8),
        SampleViewMut::contiguous(&mut lf, 1, 1),
    ) else {
        return Err(PolicyError::Unsupported {
            what: "an LLF view",
        });
    };
    lf_from_llf_into(TransformType::Dct8x8, &llf, &mut out, scratch);
    Ok(lf.first().copied().unwrap_or(0.0))
}

/// Copies an 8x8 block out of one XYB plane, replicating the frame edge.
///
/// The block grid is `ceil(width / 8) x ceil(height / 8)`, so the last column
/// and row of blocks reach past the frame. The decoder reconstructs those
/// samples and then crops them away, but what the encoder puts there still
/// decides the coefficients of the visible part: zero-filling would put a hard
/// edge into every partial block and ring across the whole right and bottom
/// margin. Replicating the edge sample puts no energy in the high frequencies
/// at all.
fn gather_block(frame: &PreparedFrame, channel: usize, x0: u32, y0: u32, out: &mut [f32; 64]) {
    let (w, h) = (frame.width(), frame.height());
    let plane = match channel {
        0 => &frame.xyb().x,
        1 => &frame.xyb().y,
        _ => &frame.xyb().b,
    };
    for dy in 0..8u32 {
        for dx in 0..8u32 {
            let x = (x0 + dx).min(w - 1);
            let y = (y0 + dy).min(h - 1);
            if let Some(slot) = out.get_mut((dy * 8 + dx) as usize) {
                *slot = plane.at(x, y, w).unwrap_or(0.0);
            }
        }
    }
}

/// The entropy model: I.2.2's default block contexts, natural orders, one
/// preset, and [`cluster_of`]'s six clusters.
fn entropy_plan(geometry: &VardctGeometry, histograms: Vec<HistogramPlan>) -> Result<EntropyPlan> {
    let block_context = HfBlockContextPlan::Default;
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
    // rather than in raw bits. One configuration for every cluster: choosing
    // per-cluster configurations is milestone 8's job.
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
    Ok(jpxl_encode::vardct::write_codestream(&plan)?)
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
        let plan = plan_at_with_cfl(
            frame,
            &atlas,
            &request,
            QuantizerChoice::from_request(&request),
            enabled,
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
        let plan = plan_at_with_cfl(
            &frame,
            &atlas,
            &request,
            QuantizerChoice::from_request(&request),
            true,
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
}
