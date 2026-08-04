//! Slice 11's exit criterion, as tests.
//!
//! > A hand-built legal plan validates and dumps; malformed plans cannot reach
//! > the writer — typed rejection, with a test per invariant.
//!
//! Every test below builds the legal plan, breaks **exactly one** invariant,
//! and asserts that `validate` rejects it *by name*. Asserting the name rather
//! than "is_err" is what makes the suite proof rather than decoration: a check
//! that fired for the wrong reason would pass an `is_err` test forever.
//!
//! The one-way boundary is why these tests live here and not in
//! `jpxl-encode-policy`: they build plans by hand, without any policy, which
//! is the shape the writer has to accept.

#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    reason = "test fixtures index their own just-built vectors; a panic here \
              is a failing test, which is the intended signal"
)]

use jpxl_bitstream::BitWriter;
use jpxl_core::geometry::LfBlockPos;
use jpxl_core::varblock::TransformType;
use jpxl_encode::section::SectionStore;
use jpxl_encode::vardct::geometry::BlockGrid;
use jpxl_encode::vardct::ids::{ClusterId, GlobalScale, HfMul, LfGroupId, PresetId, QuantLf};
use jpxl_encode::vardct::plan::{
    CflGrid, EmissionPlan, EntropyModelPlan, EntropyPlan, FrameDecision, HfBlockContextPlan,
    HfPassEntropyPlan, HistogramPlan, HybridUintPlan, LfDecision, LfGroupPlan, LfQuantPlanes,
    OrderSet, QuantizedFrameIr, QuantizedLfGroup, QuantizerDecision, RestorationDecision,
    SectionLayout, SharpnessGrid, SpatialPlan, VarblockCoefficients, VarblockDecision,
};
use jpxl_encode::vardct::{PlanError, SectionKind, VardctGeometry, dump, emit, validate};

// ---------------------------------------------------------------------------
// The legal plan every test starts from
// ---------------------------------------------------------------------------

/// One DCT8x8 per atom of `grid`, in G.2.4's greedy raster order.
fn fixed_dct8x8(grid: BlockGrid) -> Vec<VarblockDecision> {
    let mut blocks = Vec::new();
    for y in 0..grid.height {
        for x in 0..grid.width {
            blocks.push(VarblockDecision {
                origin: LfBlockPos::new(x, y),
                transform: TransformType::Dct8x8,
                hf_mul: HfMul::new(1).unwrap(),
            });
        }
    }
    blocks
}

fn coefficients_for(blocks: &[VarblockDecision]) -> Box<[VarblockCoefficients]> {
    blocks
        .iter()
        .map(|b| VarblockCoefficients::zeros(b.transform))
        .collect()
}

fn entropy_for(geometry: &VardctGeometry) -> EntropyPlan {
    let block_context = HfBlockContextPlan::Default;
    let pre_contexts = usize::try_from(495 * block_context.nb_block_ctx()).unwrap();
    let num_groups = usize::try_from(geometry.num_groups()).unwrap();
    let passes = (0..geometry.num_passes())
        .map(|_| HfPassEntropyPlan {
            orders: OrderSet::natural(),
            distributions: EntropyModelPlan {
                context_map: vec![ClusterId::new(0); pre_contexts].into_boxed_slice(),
                histograms: vec![HistogramPlan::new(vec![1u32; 32]).unwrap()].into_boxed_slice(),
                hybrid_uint: vec![HybridUintPlan {
                    split_exponent: 4,
                    msb_in_token: 2,
                    lsb_in_token: 0,
                }]
                .into_boxed_slice(),
            },
            group_presets: vec![PresetId::new(0); num_groups].into_boxed_slice(),
        })
        .collect();
    EntropyPlan {
        block_context,
        num_hf_presets: 1,
        passes,
    }
}

/// The plan every test mutates: a fixed DCT8x8 tiling of a `width x height`
/// frame, one pass, everything else at its clause default.
fn legal_plan(width: u32, height: u32, shift: u32) -> EmissionPlan {
    let frame = FrameDecision {
        width,
        height,
        group_size_shift: shift,
        num_passes: 1,
    };
    let geometry = frame.geometry().expect("a legal geometry");
    let mut lf_groups = Vec::new();
    let mut quantized = Vec::new();
    for index in 0..geometry.num_lf_groups() {
        let id = LfGroupId::new(u32::try_from(index).unwrap());
        let grid = geometry.lf_group_blocks(id).unwrap();
        let tiles = geometry.lf_group_cfl_tiles(id).unwrap();
        let blocks = fixed_dct8x8(grid);
        quantized.push(QuantizedLfGroup {
            id,
            lf: LfQuantPlanes::zeros(grid),
            coefficients: coefficients_for(&blocks),
        });
        lf_groups.push(LfGroupPlan {
            id,
            blocks: blocks.into_boxed_slice(),
            cfl: CflGrid::zeros(tiles),
            sharpness: SharpnessGrid::zeros(grid),
        });
    }
    EmissionPlan {
        spatial: SpatialPlan {
            frame,
            quantizer: QuantizerDecision {
                global_scale: GlobalScale::new(4096).unwrap(),
                quant_lf: QuantLf::new(16).unwrap(),
            },
            lf: LfDecision::default(),
            restoration: RestorationDecision::default(),
            lf_groups: lf_groups.into_boxed_slice(),
        },
        quantized: QuantizedFrameIr {
            lf_groups: quantized.into_boxed_slice(),
        },
        entropy: entropy_for(&geometry),
        sections: SectionLayout::for_geometry(&geometry),
    }
}

/// Replaces LF group 0's varblocks, keeping the IR consistent with them, so
/// that a cover test breaks the cover and nothing else.
fn set_group0_blocks(plan: &mut EmissionPlan, blocks: Vec<VarblockDecision>) {
    plan.quantized.lf_groups[0].coefficients = coefficients_for(&blocks);
    plan.spatial.lf_groups[0].blocks = blocks.into_boxed_slice();
}

#[track_caller]
fn rejected_as(plan: EmissionPlan, what: &str) {
    match validate(plan) {
        Ok(_) => panic!("expected rejection by {what:?}, but the plan validated"),
        Err(err) => assert_eq!(err.what(), what, "rejected by the wrong invariant: {err}"),
    }
}

// ---------------------------------------------------------------------------
// The exit criterion itself
// ---------------------------------------------------------------------------

#[test]
fn a_hand_built_legal_plan_validates_and_dumps() {
    // 600x520 at shift 1: a 3x3 pass-group grid inside one LF group, so the
    // multi-section path — a single-group fixture hides section ordering.
    let validated = validate(legal_plan(600, 520, 1)).expect("a legal plan");
    let text = dump(&validated).expect("dumps");
    assert!(text.contains("frame 600x520"), "{text}");
    assert!(
        text.contains("lf group 0: 75x65 blocks, nb_blocks 4875"),
        "{text}"
    );
    assert!(text.contains("DctSelect 0: 4875"), "{text}");
    assert!(
        text.contains("entropy: block_ctx default nb_block_ctx 15"),
        "{text}"
    );
    assert!(text.contains("sections: 12"), "{text}");
}

#[test]
fn a_mixed_transform_tiling_validates() {
    // A 4x2 block grid covered by one DCT16x16, one DCT16x8 pair and DCT8x8s:
    // proof the cover replay accepts every footprint shape, not just 1x1.
    let mut plan = legal_plan(32, 16, 1);
    let mul = HfMul::new(7).unwrap();
    set_group0_blocks(
        &mut plan,
        vec![
            VarblockDecision {
                origin: LfBlockPos::new(0, 0),
                transform: TransformType::Dct16x16,
                hf_mul: mul,
            },
            VarblockDecision {
                origin: LfBlockPos::new(2, 0),
                transform: TransformType::Dct16x8,
                hf_mul: mul,
            },
            VarblockDecision {
                origin: LfBlockPos::new(3, 0),
                transform: TransformType::Dct16x8,
                hf_mul: mul,
            },
        ],
    );
    let validated = validate(plan).expect("an exact cover of the 4x2 grid");
    let text = dump(&validated).expect("dumps");
    assert!(text.contains("HfMul range: 7..=7"), "{text}");
}

#[test]
fn the_emitter_accepts_a_validated_plan_and_rejects_a_mismatched_store() {
    let validated = validate(legal_plan(600, 520, 1)).expect("a legal plan");
    let sections = validated.plan().sections.kinds.len();

    let mut store = SectionStore::new();
    for _ in 0..sections {
        store.push(vec![0xAA; 3]);
    }
    let mut w = BitWriter::new();
    emit::write_sections(&validated, store, &mut w).expect("the layout matches");
    assert!(!w.into_bytes().is_empty());

    let mut short = SectionStore::new();
    for _ in 0..sections - 1 {
        short.push_empty();
    }
    let mut w = BitWriter::new();
    let err = emit::write_sections(&validated, short, &mut w)
        .expect_err("a store that does not match the plan");
    assert!(err.to_string().contains("section store length"), "{err}");
}

// ---------------------------------------------------------------------------
// Frame and geometry invariants
// ---------------------------------------------------------------------------

#[test]
fn rejects_a_zero_frame_dimension() {
    let mut plan = legal_plan(64, 64, 1);
    plan.spatial.frame.width = 0;
    rejected_as(plan, "frame width");
}

#[test]
fn rejects_an_out_of_range_group_size_shift() {
    let mut plan = legal_plan(64, 64, 1);
    plan.spatial.frame.group_size_shift = 4;
    rejected_as(plan, "group_size_shift");
}

#[test]
fn rejects_an_out_of_range_pass_count() {
    let mut plan = legal_plan(64, 64, 1);
    plan.spatial.frame.num_passes = 12;
    rejected_as(plan, "num_passes");
}

#[test]
fn rejects_a_wrong_lf_group_count() {
    let mut plan = legal_plan(64, 64, 1);
    plan.spatial.lf_groups = Box::new([]);
    rejected_as(plan, "LF group count");
}

#[test]
fn rejects_out_of_order_lf_group_ids() {
    // 4200x40 at shift 1 spans three 2048-wide LF groups.
    let mut plan = legal_plan(4200, 40, 1);
    plan.spatial.lf_groups.swap(0, 1);
    plan.quantized.lf_groups.swap(0, 1);
    rejected_as(plan, "LF group raster order");
}

// ---------------------------------------------------------------------------
// Quantizer and LF invariants
// ---------------------------------------------------------------------------

#[test]
fn rejects_an_out_of_range_extra_precision() {
    let mut plan = legal_plan(64, 64, 1);
    plan.spatial.lf.extra_precision = 4;
    rejected_as(plan, "extra_precision");
}

#[test]
fn rejects_a_non_finite_lf_dequant_weight() {
    let mut plan = legal_plan(64, 64, 1);
    plan.spatial.lf.channel_dequant[1] = f32::NAN;
    rejected_as(plan, "LF channel dequantization weight");
}

#[test]
fn rejects_an_out_of_range_epf_iteration_count() {
    let mut plan = legal_plan(64, 64, 1);
    plan.spatial.restoration.epf_iters = 4;
    rejected_as(plan, "epf_iters");
}

#[test]
fn rejects_a_zero_colour_factor() {
    let mut plan = legal_plan(64, 64, 1);
    plan.spatial.lf.correlation.colour_factor = 0;
    rejected_as(plan, "colour_factor");
}

#[test]
fn an_unrepresentable_quantizer_value_cannot_be_built_at_all() {
    // `global_scale` and `quant_lf` never reach `validate`: their newtypes
    // reject at construction, which is why no plan below can carry a bad one.
    assert!(matches!(
        GlobalScale::new(0),
        Err(PlanError::OutOfRange {
            what: "global_scale",
            ..
        })
    ));
    assert!(matches!(
        QuantLf::new(QuantLf::MAX + 1),
        Err(PlanError::OutOfRange {
            what: "quant_lf",
            ..
        })
    ));
    assert!(matches!(
        HfMul::new(0),
        Err(PlanError::OutOfRange { what: "HfMul", .. })
    ));
}

// ---------------------------------------------------------------------------
// Block-cover invariants (G.2.4's greedy raster representation)
// ---------------------------------------------------------------------------

#[test]
fn rejects_a_varblock_not_at_the_earliest_uncovered_block() {
    let mut plan = legal_plan(32, 16, 1);
    let mut blocks = fixed_dct8x8(BlockGrid {
        width: 4,
        height: 2,
    });
    // Correct sequence, one wrong claimed origin: the decoder would place it
    // at (1, 0) regardless, so the plan is describing a tiling that will not
    // happen.
    blocks[1].origin = LfBlockPos::new(3, 1);
    set_group0_blocks(&mut plan, blocks);
    rejected_as(plan, "varblock origin is not the earliest uncovered block");
}

#[test]
fn rejects_a_varblock_crossing_the_lf_group_edge() {
    let mut plan = legal_plan(24, 16, 1);
    let mut blocks = fixed_dct8x8(BlockGrid {
        width: 3,
        height: 2,
    });
    // The last atom is (2, 1); a DCT16x16 there needs columns 2..4 of a
    // 3-wide grid and rows 1..3 of a 2-tall one.
    let last = blocks.len() - 1;
    blocks[last].transform = TransformType::Dct16x16;
    blocks.truncate(last + 1);
    set_group0_blocks(&mut plan, blocks);
    rejected_as(plan, "varblock crosses the LF-group edge");
}

#[test]
fn rejects_a_varblock_crossing_a_pass_group_edge() {
    // shift 0 gives 128-sample pass groups, i.e. a boundary every 16 blocks,
    // well inside the 1024-sample LF group. A DCT16x16 at block column 15
    // straddles it: its coefficients would belong to two G.4 sections.
    let mut plan = legal_plan(256, 128, 0);
    let mut blocks = fixed_dct8x8(BlockGrid {
        width: 32,
        height: 16,
    });
    blocks[15].transform = TransformType::Dct16x16;
    blocks.truncate(16);
    set_group0_blocks(&mut plan, blocks);
    rejected_as(plan, "varblock crosses a pass-group edge");
}

#[test]
fn rejects_overlapping_varblocks() {
    // A 4x2 grid. The DCT16x8 at (1, 0) covers (1, 0) and (1, 1); the
    // DCT8x16 later placed at (0, 1) — a legal *origin*, the earliest
    // uncovered atom — reaches (1, 1) and collides.
    let mut plan = legal_plan(32, 16, 1);
    let mul = HfMul::new(1).unwrap();
    set_group0_blocks(
        &mut plan,
        vec![
            VarblockDecision {
                origin: LfBlockPos::new(0, 0),
                transform: TransformType::Dct8x8,
                hf_mul: mul,
            },
            VarblockDecision {
                origin: LfBlockPos::new(1, 0),
                transform: TransformType::Dct16x8,
                hf_mul: mul,
            },
            VarblockDecision {
                origin: LfBlockPos::new(2, 0),
                transform: TransformType::Dct8x8,
                hf_mul: mul,
            },
            VarblockDecision {
                origin: LfBlockPos::new(3, 0),
                transform: TransformType::Dct8x8,
                hf_mul: mul,
            },
            VarblockDecision {
                origin: LfBlockPos::new(0, 1),
                transform: TransformType::Dct8x16,
                hf_mul: mul,
            },
        ],
    );
    rejected_as(plan, "varblock overlaps an already-placed varblock");
}

#[test]
fn rejects_an_lf_group_left_uncovered() {
    let mut plan = legal_plan(32, 16, 1);
    let mut blocks = fixed_dct8x8(BlockGrid {
        width: 4,
        height: 2,
    });
    blocks.truncate(7);
    set_group0_blocks(&mut plan, blocks);
    rejected_as(plan, "LF group left uncovered");
}

#[test]
fn rejects_more_varblocks_than_the_block_grid_holds() {
    let mut plan = legal_plan(32, 16, 1);
    let mut blocks = fixed_dct8x8(BlockGrid {
        width: 4,
        height: 2,
    });
    blocks.push(VarblockDecision {
        origin: LfBlockPos::new(0, 0),
        transform: TransformType::Dct8x8,
        hf_mul: HfMul::new(1).unwrap(),
    });
    set_group0_blocks(&mut plan, blocks);
    rejected_as(plan, "nb_blocks");
}

// ---------------------------------------------------------------------------
// Metadata grid invariants
// ---------------------------------------------------------------------------

#[test]
fn rejects_a_cfl_grid_of_the_wrong_shape() {
    let mut plan = legal_plan(600, 520, 1);
    plan.spatial.lf_groups[0].cfl = CflGrid::zeros(BlockGrid {
        width: 1,
        height: 1,
    });
    rejected_as(plan, "CfL grid shape");
}

#[test]
fn rejects_a_sharpness_grid_of_the_wrong_shape() {
    let mut plan = legal_plan(600, 520, 1);
    plan.spatial.lf_groups[0].sharpness = SharpnessGrid::zeros(BlockGrid {
        width: 2,
        height: 2,
    });
    rejected_as(plan, "Sharpness grid shape");
}

#[test]
fn rejects_an_out_of_range_sharpness_sample() {
    let mut plan = legal_plan(32, 16, 1);
    let grid = BlockGrid {
        width: 4,
        height: 2,
    };
    let mut values = vec![0u8; 8];
    values[5] = 8; // J.4.3's lookup has eight entries, 0..=7.
    plan.spatial.lf_groups[0].sharpness = SharpnessGrid::new(grid, values).unwrap();
    rejected_as(plan, "Sharpness sample");
}

// ---------------------------------------------------------------------------
// Quantized-IR invariants
// ---------------------------------------------------------------------------

#[test]
fn rejects_a_quantized_lf_group_count_mismatch() {
    let mut plan = legal_plan(64, 64, 1);
    plan.quantized.lf_groups = Box::new([]);
    rejected_as(plan, "quantized LF group count");
}

#[test]
fn rejects_lf_quant_planes_of_the_wrong_shape() {
    let mut plan = legal_plan(600, 520, 1);
    plan.quantized.lf_groups[0].lf = LfQuantPlanes::zeros(BlockGrid {
        width: 4,
        height: 4,
    });
    rejected_as(plan, "LfQuant plane shape");
}

#[test]
fn rejects_a_coefficient_set_count_mismatch() {
    let mut plan = legal_plan(32, 16, 1);
    let mut coefficients = plan.quantized.lf_groups[0].coefficients.to_vec();
    coefficients.pop();
    plan.quantized.lf_groups[0].coefficients = coefficients.into_boxed_slice();
    rejected_as(plan, "coefficient set count");
}

#[test]
fn rejects_varblock_coefficients_of_the_wrong_length() {
    let mut plan = legal_plan(32, 16, 1);
    // A DCT16x16-sized coefficient set under a DCT8x8 varblock: 256 values
    // where I.3.2 calls for 64.
    plan.quantized.lf_groups[0].coefficients[2] =
        VarblockCoefficients::zeros(TransformType::Dct16x16);
    rejected_as(plan, "varblock coefficient count");
}

// ---------------------------------------------------------------------------
// Entropy invariants
// ---------------------------------------------------------------------------

#[test]
fn rejects_a_block_context_map_of_the_wrong_size() {
    let mut plan = legal_plan(64, 64, 1);
    plan.entropy.block_context = HfBlockContextPlan::Custom {
        lf_thresholds: [Vec::new(), Vec::new(), Vec::new()],
        qf_thresholds: Vec::new(),
        // bsize is 39 with no thresholds.
        map: vec![0u8; 40],
    };
    rejected_as(plan, "block_ctx_map length");
}

#[test]
fn rejects_a_block_context_map_larger_than_i22_permits() {
    let mut plan = legal_plan(64, 64, 1);
    // 64 qf thresholds make bsize 39 * 65 = 2535, past I.2.2's 39 * 64.
    plan.entropy.block_context = HfBlockContextPlan::Custom {
        lf_thresholds: [Vec::new(), Vec::new(), Vec::new()],
        qf_thresholds: vec![1; 64],
        map: vec![0u8; 39 * 65],
    };
    rejected_as(plan, "bsize");
}

#[test]
fn rejects_a_block_context_map_with_too_many_contexts() {
    let mut plan = legal_plan(64, 64, 1);
    let mut map = vec![0u8; 39];
    for (index, entry) in map.iter_mut().enumerate() {
        *entry = u8::try_from(index.min(16)).unwrap();
    }
    plan.entropy.block_context = HfBlockContextPlan::Custom {
        lf_thresholds: [Vec::new(), Vec::new(), Vec::new()],
        qf_thresholds: Vec::new(),
        map,
    };
    rejected_as(plan, "nb_block_ctx");
}

#[test]
fn rejects_a_block_context_map_that_is_not_dense() {
    let mut plan = legal_plan(64, 64, 1);
    let mut map = vec![0u8; 39];
    map[7] = 2; // cluster 1 is never used, so C.2.2 cannot encode this map.
    plan.entropy.block_context = HfBlockContextPlan::Custom {
        lf_thresholds: [Vec::new(), Vec::new(), Vec::new()],
        qf_thresholds: Vec::new(),
        map,
    };
    rejected_as(plan, "block_ctx_map is not dense");
}

#[test]
fn rejects_an_out_of_range_hf_preset_count() {
    let mut plan = legal_plan(64, 64, 1);
    plan.entropy.num_hf_presets = 0;
    rejected_as(plan, "num_hf_presets");

    // I.2.6 reads `u(ceil(log2(num_groups)))`, so more presets than groups
    // has no encoding either.
    let mut plan = legal_plan(64, 64, 1);
    plan.entropy.num_hf_presets = 2;
    rejected_as(plan, "num_hf_presets");
}

#[test]
fn rejects_an_entropy_pass_count_mismatch() {
    let mut plan = legal_plan(64, 64, 1);
    plan.entropy.passes = Box::new([]);
    rejected_as(plan, "entropy pass count");
}

#[test]
fn rejects_a_context_map_of_the_wrong_length() {
    let mut plan = legal_plan(64, 64, 1);
    plan.entropy.passes[0].distributions.context_map =
        vec![ClusterId::new(0); 10].into_boxed_slice();
    rejected_as(plan, "context map length");
}

#[test]
fn rejects_a_context_map_entry_past_the_histogram_list() {
    let mut plan = legal_plan(64, 64, 1);
    let map = &mut plan.entropy.passes[0].distributions.context_map;
    map[3] = ClusterId::new(1); // only one histogram exists
    rejected_as(plan, "context map entry");
}

#[test]
fn rejects_a_context_map_that_leaves_a_cluster_unused() {
    let mut plan = legal_plan(64, 64, 1);
    let model = &mut plan.entropy.passes[0].distributions;
    model.histograms = vec![
        HistogramPlan::new(vec![1u32; 32]).unwrap(),
        HistogramPlan::new(vec![1u32; 32]).unwrap(),
    ]
    .into_boxed_slice();
    model.hybrid_uint = vec![HybridUintPlan::default(); 2].into_boxed_slice();
    rejected_as(plan, "context map is not dense");
}

#[test]
fn rejects_a_hybrid_uint_config_count_mismatch() {
    let mut plan = legal_plan(64, 64, 1);
    plan.entropy.passes[0].distributions.hybrid_uint = Box::new([]);
    rejected_as(plan, "HybridUintConfig count");
}

#[test]
fn rejects_an_impossible_hybrid_uint_config() {
    let mut plan = legal_plan(64, 64, 1);
    // msb + lsb may not exceed split_exponent (C.2.3).
    plan.entropy.passes[0].distributions.hybrid_uint = vec![HybridUintPlan {
        split_exponent: 2,
        msb_in_token: 2,
        lsb_in_token: 1,
    }]
    .into_boxed_slice();
    rejected_as(plan, "HybridUintConfig");
}

#[test]
fn rejects_a_group_preset_past_the_preset_count() {
    let mut plan = legal_plan(600, 520, 1);
    plan.entropy.passes[0].group_presets[4] = PresetId::new(1);
    rejected_as(plan, "hfp");
}

#[test]
fn rejects_a_group_preset_assignment_of_the_wrong_length() {
    let mut plan = legal_plan(600, 520, 1);
    plan.entropy.passes[0].group_presets = Box::new([]);
    rejected_as(plan, "HF preset assignment count");
}

// ---------------------------------------------------------------------------
// Section-layout invariants
// ---------------------------------------------------------------------------

#[test]
fn rejects_a_section_layout_that_is_not_f31() {
    let mut plan = legal_plan(600, 520, 1);
    let mut kinds = plan.sections.kinds.to_vec();
    kinds.pop();
    plan.sections.kinds = kinds.into_boxed_slice();
    rejected_as(plan, "section count");

    let mut plan = legal_plan(600, 520, 1);
    let mut kinds = plan.sections.kinds.to_vec();
    // F.3.1 puts HfGlobal after the LF groups, not before them.
    kinds.swap(1, 2);
    plan.sections.kinds = kinds.into_boxed_slice();
    rejected_as(plan, "section order");
}

#[test]
fn a_single_group_frame_has_the_one_section_f31_gives_it() {
    let plan = legal_plan(200, 200, 1);
    assert_eq!(plan.sections.kinds.as_ref(), &[SectionKind::Whole]);
    validate(plan).expect("a legal single-section plan");
}
