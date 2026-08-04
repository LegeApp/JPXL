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
//! | Module | `Encoder-plan1.md` | Milestone-1 state |
//! | --- | --- | --- |
//! | [`source`] | §2.1 `PreparedFrame` | resident XYB planes |
//! | [`analysis`] | §2.2 `AnalysisAtlas` | per-atom mean and variance |
//! | [`block`] | §4 cover search | fixed DCT8x8 |
//! | [`request`] | §12 budgets | the fields that exist |
//! | [`plan_frame`] | §16 orchestration | builds and validates a plan |
//!
//! # What milestone 1 does *not* do
//!
//! It does not encode. Forward transforms, quantization against the exact
//! decoder function, real histograms and the ANS writer are milestones 2 and
//! 11.5. [`plan_frame`] produces a structurally legal plan whose coefficients
//! are zero — enough to prove the boundary works end to end, and honest about
//! what has not been built.

pub mod analysis;
pub mod block;
pub mod error;
pub mod request;
pub mod source;

use jpxl_encode::vardct::ids::{ClusterId, LfGroupId, PresetId};
use jpxl_encode::vardct::plan::{
    CflGrid, EmissionPlan, EntropyModelPlan, EntropyPlan, FrameDecision, HfBlockContextPlan,
    HfPassEntropyPlan, HistogramPlan, HybridUintPlan, LfDecision, LfGroupPlan, LfQuantPlanes,
    OrderSet, QuantizedFrameIr, QuantizedLfGroup, QuantizerDecision, RestorationDecision,
    SectionLayout, SharpnessGrid, SpatialPlan, VarblockCoefficients,
};
use jpxl_encode::vardct::{ValidatedEmissionPlan, VardctGeometry, validate};

pub use analysis::{AnalysisAtlas, AtomGrid};
pub use error::{PolicyError, Result};
pub use request::{CoverMode, EncodeRequest, SearchBudget};
pub use source::PreparedFrame;

/// Plans one VarDCT frame and hands back a plan the writer will accept.
///
/// This is `Encoder-plan1.md` §16's orchestration, at the size milestone 1
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
/// # Errors
///
/// As [`plan_frame`].
pub fn plan_frame_with_atlas(
    frame: &PreparedFrame,
    atlas: &AnalysisAtlas,
    request: &EncodeRequest,
) -> Result<ValidatedEmissionPlan> {
    let decision = FrameDecision {
        width: frame.width(),
        height: frame.height(),
        group_size_shift: request.group_size_shift,
        num_passes: 1,
    };
    let geometry = decision.geometry()?;

    debug_assert_eq!(
        atlas.grid().area(),
        geometry.frame_blocks().area(),
        "the atom grid and the block grid are the same grid"
    );

    let mut lf_groups = Vec::new();
    let mut quantized = Vec::new();
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

        let varblocks = match request.budget.cover_mode {
            CoverMode::FixedDct8x8 => block::fixed_dct8x8(blocks, request.hf_mul)?,
        };
        let coefficients: Vec<VarblockCoefficients> = varblocks
            .iter()
            .map(|b| VarblockCoefficients::zeros(b.transform))
            .collect();

        lf_groups.push(LfGroupPlan {
            id,
            blocks: varblocks.into_boxed_slice(),
            cfl: CflGrid::zeros(tiles),
            sharpness: SharpnessGrid::zeros(blocks),
        });
        quantized.push(QuantizedLfGroup {
            id,
            lf: LfQuantPlanes::zeros(blocks),
            coefficients: coefficients.into_boxed_slice(),
        });
    }

    let spatial = SpatialPlan {
        frame: decision,
        quantizer: QuantizerDecision {
            global_scale: request.global_scale,
            quant_lf: request.quant_lf,
        },
        lf: LfDecision::default(),
        restoration: RestorationDecision::default(),
        lf_groups: lf_groups.into_boxed_slice(),
    };

    let entropy = single_cluster_entropy(&geometry)?;
    let plan = EmissionPlan {
        spatial,
        quantized: QuantizedFrameIr {
            lf_groups: quantized.into_boxed_slice(),
        },
        entropy,
        sections: SectionLayout::for_geometry(&geometry),
    };
    Ok(validate(plan)?)
}

/// The simplest legal entropy model: one preset, one cluster, natural orders.
///
/// A real model comes out of §9's census-cluster-replay compiler in milestone
/// 8; this one exists so that a milestone-1 plan is *complete*, not so that it
/// compresses.
fn single_cluster_entropy(geometry: &VardctGeometry) -> Result<EntropyPlan> {
    let block_context = HfBlockContextPlan::Default;
    let num_hf_presets = 1;
    let pre_contexts = 495 * u64::from(num_hf_presets) * block_context.nb_block_ctx();
    let pre_contexts = usize::try_from(pre_contexts).unwrap_or(0);
    let num_groups = usize::try_from(geometry.num_groups()).unwrap_or(0);

    let distributions = EntropyModelPlan {
        context_map: vec![ClusterId::new(0); pre_contexts].into_boxed_slice(),
        histograms: vec![HistogramPlan::new(vec![1u32; 32])?].into_boxed_slice(),
        hybrid_uint: vec![HybridUintPlan {
            split_exponent: 4,
            msb_in_token: 2,
            lsb_in_token: 0,
        }]
        .into_boxed_slice(),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn grey_frame(width: u32, height: u32) -> PreparedFrame {
        let n = (width * height) as usize;
        PreparedFrame::from_linear_srgb(width, height, vec![0.4; n], vec![0.4; n], vec![0.4; n])
            .expect("legal frame")
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
}
