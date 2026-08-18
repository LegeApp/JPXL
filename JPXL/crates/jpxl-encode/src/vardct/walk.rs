//! I.4's coefficient traversal, written once and driven through a sink.
//!
//! `Encoder-plan1.md` §9.1: the HF walk is the hardest normative loop in the
//! encoder and it has to run at least twice — once to census raw values so the
//! histograms can be trained, once to emit symbols against those histograms.
//! Two copies of a loop this stateful (`non_zeros` prediction, `prev`, the
//! coefficient order, the block context) drift. So there is one copy, and both
//! passes are [`HfEventSink`] implementations over it.
//!
//! ```text
//!                       ┌──▶ CensusSink   (policy: train histograms)
//! walk_pass_group ──────┤
//!                       └──▶ symbol sink  (writer: emit ANS symbols)
//! ```
//!
//! # What the walk is, restated from I.4
//!
//! Varblocks in raster order of their top-left block. For each varblock, the
//! channels in the order **Y, X, B** — I.4 states that order explicitly, and it
//! is not the Table I.1 numbering. For each channel:
//!
//! 1. one `non_zeros` symbol, in the context
//!    `NonZerosContext(PredictedNonZeros(x, y))`, whose value is the number of
//!    non-zero coefficients at natural-order positions `[num_blocks, size)`;
//! 2. the per-block share `ceil(non_zeros / num_blocks)` written into the
//!    `NonZeros` grid at every 8x8 block the varblock covers, because the
//!    *next* varblock's prediction reads it;
//! 3. if `non_zeros > 0`, the coefficients at order positions
//!    `num_blocks, num_blocks + 1, ...` **up to and including the last
//!    non-zero one**, each in the context
//!    `CoefficientContext(k, non_zeros, num_blocks, size, prev)`.
//!
//! Step 3's stopping rule is the one an encoder gets wrong: the decoder counts
//! down and stops the instant the promised number of non-zeros has arrived, so
//! writing the trailing zeros after the last non-zero would leave symbols in
//! the stream that nothing reads. [`walk_pass_group`] stops at exactly the same
//! place, for exactly the same reason.
//!
//! Nothing in this module encodes: it produces events. Whether an event becomes
//! a histogram bucket or an ANS symbol is the sink's business.

use jpxl_core::varblock::{TransformType, natural_coeff_order_ref, order_id_dims};

use crate::entropy::pack_signed;
use crate::vardct::error::{PlanError, PlanResult};
use crate::vardct::ids::PreContextId;
use crate::vardct::plan::{
    CONTEXTS_PER_BLOCK_CTX, HfBlockContextPlan, NUM_CHANNELS, OrderSet, VarblockCoefficients,
};
use crate::vardct::sink::HfEventSink;

/// I.4's channel order: Y, then X, then B.
pub const CHANNEL_WALK_ORDER: [usize; NUM_CHANNELS] = [1, 0, 2];

/// I.4's per-block-context share of the `non_zeros` contexts.
pub const NON_ZEROS_CONTEXTS: u64 = 37;

/// I.4's per-block-context share of the coefficient contexts.
pub const COEFFICIENT_CONTEXTS: u64 = 458;

/// Number of Table I.7 shape classes a block context indexes over.
const BLOCK_CTX_SHAPE_CLASSES: usize = 13;

/// I.4's `CoeffFreqContext[64]`.
const COEFF_FREQ_CONTEXT: [u32; 64] = [
    0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, //
    15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, //
    23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, //
    27, 27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30,
];

/// I.4's `CoeffNumNonzeroContext[64]`.
const COEFF_NUM_NONZERO_CONTEXT: [u32; 64] = [
    0, 0, 31, 62, 62, 93, 93, 93, 93, 123, 123, 123, 123, //
    152, 152, 152, 152, 152, 152, 152, 152, 180, 180, 180, 180, 180, //
    180, 180, 180, 180, 180, 180, 180, 206, 206, 206, 206, 206, 206, //
    206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, //
    206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
];

/// One varblock of a pass group, with everything I.4 asks about it.
#[derive(Debug, Clone, Copy)]
pub struct WalkVarblock<'a> {
    /// Column of the varblock's top-left 8x8 block, **relative to the pass
    /// group**: I.4's `NonZeros` grid is group-scoped, not LF-group-scoped.
    pub bx: u32,
    /// Row of the varblock's top-left 8x8 block, relative to the pass group.
    pub by: u32,
    /// The `DctSelect` transform.
    pub transform: TransformType,
    /// `HfMul` — I.4's `qf`.
    pub hf_mul: u32,
    /// The quantized LF samples at the varblock's top-left block, in Table
    /// I.1's `[X, Y, B]` order — I.4's `qdc[3]`.
    pub qdc: [i32; NUM_CHANNELS],
    /// The quantized coefficients, per channel, in coefficient-array order.
    pub coefficients: &'a VarblockCoefficients,
}

/// The pass-group-invariant half of the walk.
#[derive(Debug, Clone, Copy)]
pub struct PassGroupWalk<'a> {
    /// Width of the pass group's 8x8-block grid.
    pub blocks_w: u32,
    /// Height of the pass group's 8x8-block grid.
    pub blocks_h: u32,
    /// I.2.2's block-context model.
    pub block_context: &'a HfBlockContextPlan,
    /// I.3.1's coefficient orders for this pass.
    pub orders: &'a OrderTables,
    /// I.4's `offset = 495 * nb_block_ctx * hfp`.
    pub offset: u64,
}

/// The 13 x 3 coefficient order tables of one pass (I.3.1).
///
/// Built once per pass and shared by every group, because I.3.1 signals them
/// once per pass. `used_orders == 0` — this slice's only case — makes every
/// entry the natural order of I.3.2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrderTables {
    tables: Vec<Vec<u32>>,
}

impl OrderTables {
    /// Materializes the orders an [`OrderSet`] describes.
    #[must_use]
    pub fn from_order_set(orders: &OrderSet) -> Self {
        let mut tables = Vec::with_capacity(jpxl_core::varblock::NUM_ORDER_IDS * NUM_CHANNELS);
        for order_id in 0..jpxl_core::varblock::NUM_ORDER_IDS {
            // Phase 30: one shared borrow instead of one throwaway clone
            // before the per-channel clones below.
            let natural = order_id_dims(order_id)
                .and_then(|(w, h)| natural_coeff_order_ref(w, h))
                .unwrap_or_default();
            for _ in 0..NUM_CHANNELS {
                tables.push(natural.to_vec());
            }
        }
        for over in orders.overrides() {
            let slot = usize::from(over.order_id.get()) * NUM_CHANNELS + usize::from(over.channel);
            if let Some(entry) = tables.get_mut(slot) {
                *entry = over.table.to_vec();
            }
        }
        Self { tables }
    }

    /// The order table for one Order ID and channel.
    #[must_use]
    pub fn order(&self, order_id: usize, channel: usize) -> Option<&[u32]> {
        self.tables
            .get(order_id * NUM_CHANNELS + channel)
            .map(Vec::as_slice)
    }
}

/// The `NonZeros` grid of I.4, one plane per channel, pass-group scoped.
struct NonZerosGrid {
    width: usize,
    height: usize,
    values: [Vec<u32>; NUM_CHANNELS],
}

impl NonZerosGrid {
    fn new(width: usize, height: usize) -> Self {
        let cells = width.saturating_mul(height);
        Self {
            width,
            height,
            values: core::array::from_fn(|_| vec![0u32; cells]),
        }
    }

    fn get(&self, channel: usize, x: usize, y: usize) -> u32 {
        if x >= self.width || y >= self.height {
            return 0;
        }
        self.values
            .get(channel)
            .and_then(|v| v.get(y * self.width + x))
            .copied()
            .unwrap_or(0)
    }

    fn set(&mut self, channel: usize, x: usize, y: usize, value: u32) {
        if x >= self.width || y >= self.height {
            return;
        }
        let width = self.width;
        if let Some(slot) = self
            .values
            .get_mut(channel)
            .and_then(|v| v.get_mut(y * width + x))
        {
            *slot = value;
        }
    }

    /// I.4's `PredictedNonZeros(x, y)`.
    fn predicted(&self, channel: usize, x: usize, y: usize) -> u32 {
        match (x, y) {
            (0, 0) => 32,
            (0, _) => self.get(channel, x, y - 1),
            (_, 0) => self.get(channel, x - 1, y),
            _ => (self.get(channel, x, y - 1) + self.get(channel, x - 1, y) + 1) >> 1,
        }
    }
}

/// I.4's `BlockContext()`.
///
/// `channel` is the clause's `c` in Table I.1 numbering (`0 = X`, `1 = Y`,
/// `2 = B`) — the Y/X swap lives here, not in the caller's loop order.
fn block_context(
    model: &HfBlockContextPlan,
    channel: usize,
    order_id: usize,
    qf: u32,
    qdc: [i32; NUM_CHANNELS],
) -> PlanResult<u64> {
    let channel_class = if channel < 2 { channel ^ 1 } else { 2 };
    let mut idx = channel_class * BLOCK_CTX_SHAPE_CLASSES + order_id;

    let (lf_thresholds, qf_thresholds): (&[Vec<i32>; NUM_CHANNELS], &[u32]) = match model {
        HfBlockContextPlan::Default => (&EMPTY_LF_THRESHOLDS, &[]),
        HfBlockContextPlan::Custom {
            lf_thresholds,
            qf_thresholds,
            ..
        } => (lf_thresholds, qf_thresholds),
    };

    idx *= qf_thresholds.len() + 1;
    for &t in qf_thresholds {
        if qf > t {
            idx += 1;
        }
    }
    for row in lf_thresholds {
        idx *= row.len() + 1;
    }

    // I.4 walks the LF thresholds in the order 0, 2, 1.
    let mut lf_idx = 0usize;
    let walk = [0usize, 2, 1];
    for (step, &c) in walk.iter().enumerate() {
        if step > 0 {
            lf_idx *= lf_thresholds.get(c).map_or(1, |r| r.len() + 1);
        }
        let value = qdc.get(c).copied().unwrap_or(0);
        for &t in lf_thresholds.get(c).map(Vec::as_slice).unwrap_or(&[]) {
            if value > t {
                lf_idx += 1;
            }
        }
    }

    model
        .map()
        .get(idx + lf_idx)
        .map(|&v| u64::from(v))
        .ok_or_else(|| PlanError::out_of_range("block_ctx_map index", "I.4", (idx + lf_idx) as i64))
}

/// The empty threshold array the default block-context model implies.
static EMPTY_LF_THRESHOLDS: [Vec<i32>; NUM_CHANNELS] = [Vec::new(), Vec::new(), Vec::new()];

/// I.4's `NonZerosContext(predicted)`, without the preset offset.
const fn non_zeros_context(block_ctx: u64, nb_block_ctx: u64, predicted: u32) -> u64 {
    let predicted = if predicted > 64 { 64 } else { predicted } as u64;
    let multiplier = if predicted < 8 {
        predicted
    } else {
        4 + predicted / 2
    };
    block_ctx + nb_block_ctx * multiplier
}

/// I.4's `CoefficientContext(k, non_zeros, num_blocks, size, prev)`, without
/// the preset offset.
fn coefficient_context(
    block_ctx: u64,
    nb_block_ctx: u64,
    k: usize,
    non_zeros: u32,
    num_blocks: usize,
    prev: u64,
) -> PlanResult<u64> {
    let nz = (non_zeros as usize).div_ceil(num_blocks.max(1));
    let kk = k / num_blocks.max(1);
    let (Some(&a), Some(&b)) = (
        COEFF_NUM_NONZERO_CONTEXT.get(nz),
        COEFF_FREQ_CONTEXT.get(kk),
    ) else {
        return Err(PlanError::out_of_range(
            "non_zeros",
            "I.4",
            i64::from(non_zeros),
        ));
    };
    let within = u64::from(a + b) * 2 + prev;
    Ok(within + block_ctx * COEFFICIENT_CONTEXTS + NON_ZEROS_CONTEXTS * nb_block_ctx)
}

/// Walks one pass group's HF coefficients, feeding `sink` every I.4 event.
///
/// `varblocks` must be in raster order of their top-left block, with
/// **group-relative** coordinates, and must not leave the group's block grid.
///
/// # Errors
///
/// [`PlanError::OutOfRange`] if a varblock leaves the grid, if a coefficient
/// array does not match its transform, if an order table is the wrong length,
/// or if the block-context index falls outside the map. Every one of these is
/// a bug in the planner, not in the source image; validation catches the
/// structural ones earlier, and this is the backstop for the rest.
pub fn walk_pass_group(
    walk: &PassGroupWalk<'_>,
    varblocks: &[WalkVarblock<'_>],
    sink: &mut impl HfEventSink,
) -> PlanResult<()> {
    let nb_block_ctx = walk.block_context.nb_block_ctx();
    let mut grid = NonZerosGrid::new(walk.blocks_w as usize, walk.blocks_h as usize);

    for vb in varblocks {
        let (bx, by) = (vb.bx as usize, vb.by as usize);
        let (block_rows, block_cols) = vb.transform.block_dims();
        if bx + block_cols > grid.width || by + block_rows > grid.height {
            return Err(PlanError::out_of_range(
                "varblock outside the pass-group grid",
                "I.4",
                i64::from(vb.bx),
            ));
        }
        let order_id = vb.transform.order_id();
        let num_blocks = vb.transform.num_blocks();
        let size = vb.transform.coeff_rows() * vb.transform.coeff_cols();
        sink.varblock(vb.transform, vb.hf_mul);

        for &channel in &CHANNEL_WALK_ORDER {
            let block_ctx =
                block_context(walk.block_context, channel, order_id, vb.hf_mul, vb.qdc)?;
            let coefficients = vb.coefficients.channel(channel).ok_or_else(|| {
                PlanError::out_of_range("coefficient channel", "I.4", channel as i64)
            })?;
            if coefficients.len() != size {
                return Err(PlanError::shape(
                    "varblock coefficient count",
                    "I.3.2",
                    size as u64,
                    coefficients.len() as u64,
                ));
            }
            let order = walk
                .orders
                .order(order_id, channel)
                .filter(|o| o.len() == size)
                .ok_or_else(|| {
                    PlanError::out_of_range("coefficient order length", "I.3.1", order_id as i64)
                })?;

            // I.4 counts the non-zero coefficients at the HF order positions.
            // The LLF cells (`k < num_blocks`) are never coded here: the
            // decoder overwrites them from the LF image (I.8).
            let mut non_zeros = 0u32;
            let mut last_nonzero: Option<usize> = None;
            for k in num_blocks..size {
                let cell = order.get(k).copied().unwrap_or(0) as usize;
                if coefficients.get(cell).copied().unwrap_or(0) != 0 {
                    non_zeros += 1;
                    last_nonzero = Some(k);
                }
            }

            let predicted = grid.predicted(channel, bx, by);
            let ctx = non_zeros_context(block_ctx, nb_block_ctx, predicted) + walk.offset;
            sink.nonzeros(pre_context(ctx)?, non_zeros);

            let num_blocks_u32 = u32::try_from(num_blocks).unwrap_or(1);
            let per_block = non_zeros.div_ceil(num_blocks_u32.max(1));
            for j in 0..block_rows {
                for i in 0..block_cols {
                    grid.set(channel, bx + i, by + j, per_block);
                }
            }

            let Some(last) = last_nonzero else {
                continue;
            };

            let mut remaining = non_zeros;
            let size_u32 = u32::try_from(size).unwrap_or(u32::MAX);
            let mut previous_was_nonzero = non_zeros <= size_u32 / 16;
            for k in num_blocks..=last {
                let prev = u64::from(previous_was_nonzero);
                let ctx =
                    coefficient_context(block_ctx, nb_block_ctx, k, remaining, num_blocks, prev)?
                        + walk.offset;
                let cell = order.get(k).copied().unwrap_or(0) as usize;
                let value = coefficients.get(cell).copied().unwrap_or(0);
                sink.coefficient(pre_context(ctx)?, pack_signed(value));
                if value != 0 {
                    remaining -= 1;
                }
                previous_was_nonzero = value != 0;
            }
        }
    }
    Ok(())
}

/// Narrows a computed context to the plan's context-id type.
fn pre_context(ctx: u64) -> PlanResult<PreContextId> {
    u32::try_from(ctx)
        .map(PreContextId::new)
        .map_err(|_| PlanError::out_of_range("HF context index", "I.3.3", i64::MAX))
}

/// The number of pre-clustering contexts one pass covers, `495 * presets * nb`.
#[must_use]
pub const fn pre_context_count(num_hf_presets: u32, nb_block_ctx: u64) -> u64 {
    CONTEXTS_PER_BLOCK_CTX * num_hf_presets as u64 * nb_block_ctx
}

#[cfg(test)]
#[allow(clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::vardct::sink::CensusSink;
    use jpxl_core::varblock::natural_coeff_order;

    fn dct8x8(values: [i32; 64]) -> VarblockCoefficients {
        VarblockCoefficients::new(
            TransformType::Dct8x8,
            core::array::from_fn(|_| values.to_vec()),
        )
        .expect("64 cells")
    }

    #[test]
    fn pack_signed_is_the_inverse_of_unpack_signed() {
        // B.3's UnpackSigned, restated so the pair is checked against the
        // clause rather than against one implementation.
        let unpack = |u: u32| -> i32 {
            if u.is_multiple_of(2) {
                0i32.wrapping_add_unsigned(u / 2)
            } else {
                0i32.wrapping_sub_unsigned(u / 2 + 1)
            }
        };
        for v in [0i32, 1, -1, 2, -2, 1000, -1000, i32::MAX, i32::MIN + 1] {
            assert_eq!(unpack(pack_signed(v)), v, "{v}");
        }
        assert_eq!(pack_signed(0), 0);
        assert_eq!(pack_signed(-1), 1);
        assert_eq!(pack_signed(1), 2);
    }

    #[test]
    fn the_default_model_gives_y_context_0_and_x_b_context_7_for_dct8x8() {
        // Order ID 0, so idx is channel_class * 13. The default map's rows are
        // Y at 0, X at 13 and B at 26; entries 0, 13 and 26 are 0, 7 and 7.
        let model = HfBlockContextPlan::Default;
        assert_eq!(
            block_context(&model, 1, 0, 1, [0; 3]).expect("in range"),
            0,
            "Y"
        );
        assert_eq!(
            block_context(&model, 0, 0, 1, [0; 3]).expect("in range"),
            7,
            "X"
        );
        assert_eq!(
            block_context(&model, 2, 0, 1, [0; 3]).expect("in range"),
            7,
            "B"
        );
    }

    #[test]
    fn an_all_zero_varblock_emits_one_non_zeros_symbol_per_channel_and_nothing_else() {
        let coeffs = dct8x8([0; 64]);
        let orders = OrderTables::from_order_set(&OrderSet::natural());
        let walk = PassGroupWalk {
            blocks_w: 1,
            blocks_h: 1,
            block_context: &HfBlockContextPlan::Default,
            orders: &orders,
            offset: 0,
        };
        let mut census = CensusSink::new(495 * 15);
        walk_pass_group(
            &walk,
            &[WalkVarblock {
                bx: 0,
                by: 0,
                transform: TransformType::Dct8x8,
                hf_mul: 1,
                qdc: [0; 3],
                coefficients: &coeffs,
            }],
            &mut census,
        )
        .expect("walks");

        let total: u64 = (0..495 * 15)
            .filter_map(|c| census.histogram(PreContextId::new(c)))
            .map(crate::vardct::sink::RawHistogram::total)
            .sum();
        assert_eq!(total, 3, "three non_zeros symbols, no coefficients");
    }

    #[test]
    fn the_walk_stops_after_the_last_non_zero_coefficient() {
        // Natural order position 1 is the first HF cell; put the only non-zero
        // there and exactly one coefficient symbol may follow the count.
        let natural = natural_coeff_order(8, 8);
        let mut values = [0i32; 64];
        let cell = natural.get(1).copied().expect("64 positions") as usize;
        values[cell] = 5;
        let coeffs = dct8x8(values);
        let orders = OrderTables::from_order_set(&OrderSet::natural());
        let walk = PassGroupWalk {
            blocks_w: 1,
            blocks_h: 1,
            block_context: &HfBlockContextPlan::Default,
            orders: &orders,
            offset: 0,
        };

        #[derive(Default)]
        struct Counting {
            nonzeros: Vec<u32>,
            coefficients: Vec<u32>,
        }
        impl HfEventSink for Counting {
            fn nonzeros(&mut self, _c: PreContextId, v: u32) {
                self.nonzeros.push(v);
            }
            fn coefficient(&mut self, _c: PreContextId, v: u32) {
                self.coefficients.push(v);
            }
        }

        let mut sink = Counting::default();
        walk_pass_group(
            &walk,
            &[WalkVarblock {
                bx: 0,
                by: 0,
                transform: TransformType::Dct8x8,
                hf_mul: 1,
                qdc: [0; 3],
                coefficients: &coeffs,
            }],
            &mut sink,
        )
        .expect("walks");

        assert_eq!(sink.nonzeros, vec![1, 1, 1], "one non-zero in each channel");
        assert_eq!(
            sink.coefficients,
            vec![pack_signed(5); 3],
            "exactly one coefficient symbol per channel, and no trailing zeros"
        );
    }

    #[test]
    fn a_varblock_outside_the_group_grid_is_rejected() {
        let coeffs = dct8x8([0; 64]);
        let orders = OrderTables::from_order_set(&OrderSet::natural());
        let walk = PassGroupWalk {
            blocks_w: 1,
            blocks_h: 1,
            block_context: &HfBlockContextPlan::Default,
            orders: &orders,
            offset: 0,
        };
        let mut census = CensusSink::new(495 * 15);
        assert!(
            walk_pass_group(
                &walk,
                &[WalkVarblock {
                    bx: 1,
                    by: 0,
                    transform: TransformType::Dct8x8,
                    hf_mul: 1,
                    qdc: [0; 3],
                    coefficients: &coeffs,
                }],
                &mut census,
            )
            .is_err()
        );
    }
}
