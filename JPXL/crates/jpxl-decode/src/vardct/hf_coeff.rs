//! HF coefficient histograms and the decoding of quantized HF coefficients
//! (18181-1 I.3.3 and I.4, plus the `HfPass` half of Table G.4).
//!
//! # What this module owns
//!
//! * The `HfPass` bundle of I.3: per pass, the coefficient orders (I.3.1,
//!   implemented in [`crate::vardct::order`]) followed by the histogram bundle
//!   of I.3.3.
//! * I.4's context model — `BlockContext`, `NonZerosContext`,
//!   `CoefficientContext` and `PredictedNonZeros` — and the per-group decode
//!   loop that drives it.
//!
//! It stops at *quantized* coefficients. I.5.3's bias adjustment,
//! dequantization matrices and I.6 chroma-from-luma all consume
//! [`HfCoefficients`] afterwards; nothing here is a float.
//!
//! # I.3.3 and I.4's histogram arithmetic
//!
//! I.3.3: the bundle has `495 * num_hf_presets * nb_block_ctx` pre-clustered
//! distributions. I.4: `hfp = u(ceil(log2(num_hf_presets)))` selects a preset,
//! contributing `offset = 495 * nb_block_ctx * hfp` to every context.
//!
//! The 495 is not a magic number, it is a sum, and the split is load-bearing:
//!
//! ```text
//! NonZerosContext   in [0,               37 * nb_block_ctx)
//! CoefficientContext in [37 * nb_block_ctx, 495 * nb_block_ctx)
//! ```
//!
//! because `NonZerosContext` is `BlockContext() + nb_block_ctx * m` with
//! `m <= 4 + 64/2 = 36`, and `CoefficientContext` is
//! `37 * nb_block_ctx + BlockContext() * 458 + within` with `within < 458`.
//! `37 + 458 == 495`. The `within < 458` bound is *tight* — its reachable
//! maximum is exactly 457 — which is a strong check on the two constant
//! tables; see `within_block_context_bound_is_tight`.
//!
//! # I.4's decode loop
//!
//! ```text
//! for each varblock in raster order:
//!   for c in Y, X, B:                       /* note: not X, Y, B */
//!     non_zeros = DecodeHybridVarLenUint(NonZerosContext(PredictedNonZeros(x, y)) + offset)
//!     NonZeros(x + i, y + j) = (non_zeros + num_blocks - 1) Idiv num_blocks
//!     for k in [num_blocks, size):
//!       ucoeff = DecodeHybridVarLenUint(CoefficientContext(k, non_zeros, num_blocks, size, prev) + offset)
//!       coefficient at position order[p][s][c][k] = UnpackSigned(ucoeff)
//!       if ucoeff != 0 { non_zeros -= 1; if non_zeros == 0 { break } }
//! ```
//!
//! Three things a reader should not have to rediscover:
//!
//! * `size == 64 * num_blocks` for every Table I.1 transform, so `k Idiv
//!   num_blocks` and `ceil(non_zeros / num_blocks)` are always below 64 and
//!   the two 64-entry constant tables can never be indexed out of range by a
//!   *conforming* stream. This module still rejects a non-conforming one.
//! * The channel loop is Y, X, B, while the channel *numbering* everywhere
//!   else (Table I.1, the dequantization matrices, `qdc`) is X = 0, Y = 1,
//!   B = 2. `BlockContext`'s `c ^ 1` is that swap, not a second convention.
//! * The first `num_blocks` order positions are the LLF sub-rectangle. I.4
//!   never writes them; I.8 does.

// The two 64-entry constant tables are indexed only after the index has been
// proved below 64 (see `coefficient_context`), the per-channel `NonZeros`
// grids are indexed through helpers that bound-check, and the fixed-size
// arrays below are indexed by loop variables of matching extent.
#![allow(clippy::indexing_slicing)]

use jpxl_bitstream::{BitReader, trace_field};
use jpxl_core::limits::AllocGuard;
use jpxl_core::varblock::TransformType;
use jpxl_entropy::SymbolDecoder;

use crate::error::{DecodeError, Result};
use crate::vardct::block_ctx::HfBlockContext;
use crate::vardct::order::{NaturalOrders, OrderLookup, PassOrders, read_hf_coeff_orders};

// ---------------------------------------------------------------------------
// Constants of the context model (I.4)
// ---------------------------------------------------------------------------

/// Contexts `NonZerosContext` spans, per block context.
///
/// `4 + 64 Idiv 2 = 36` is the largest multiplier, so the span is `0..=36`,
/// i.e. 37 values.
pub const NON_ZEROS_CONTEXTS: usize = 37;

/// Contexts `CoefficientContext` spans, per block context (I.4, printed
/// literally as the `* 458` factor).
pub const COEFFICIENT_CONTEXTS: usize = 458;

/// I.3.3's `495`: the total number of contexts per block context.
pub const CONTEXTS_PER_BLOCK_CTX: usize = NON_ZEROS_CONTEXTS + COEFFICIENT_CONTEXTS;

/// The 13 shape classes `BlockContext` multiplies the channel by.
const BLOCK_CTX_SHAPE_CLASSES: usize = 13;

/// I.4's channel decode order: Y, X, then B, in the clause's own `c`
/// numbering (`0 = X`, `1 = Y`, `2 = B`).
pub const CHANNEL_DECODE_ORDER: [usize; 3] = [1, 0, 2];

/// Number of coefficient channels.
pub const NUM_CHANNELS: usize = 3;

/// I.4's `CoeffFreqContext[64]`.
///
/// Verified identical in `latex/part1.tex` and
/// `markdowns/standard-markdowns/part1.md`.
const COEFF_FREQ_CONTEXT: [u32; 64] = [
    0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, //
    15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, //
    23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, //
    27, 27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30,
];

/// I.4's `CoeffNumNonzeroContext[64]`.
///
/// Verified identical in both Part 1 sources; the run structure
/// (1x0 pair, 1, 2, 4, 4, 8, 12, 31) is asserted in
/// `constant_tables_match_the_printed_runs`, which is what catches a
/// transcription that lost or gained an entry across the page break the
/// table straddles.
const COEFF_NUM_NONZERO_CONTEXT: [u32; 64] = [
    0, 0, 31, 62, 62, 93, 93, 93, 93, 123, 123, 123, 123, //
    152, 152, 152, 152, 152, 152, 152, 152, //
    180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, //
    206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
    206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
];

/// Flip-point: whether `prev` looks at the coefficient decoded by the
/// *current* pass or at the accumulated multi-pass value.
///
/// I.4 says `prev` is 0 when "the decoded coefficient at position `k - 1` is
/// 0". For a single-pass frame the two readings coincide. For a progressive
/// frame the sentence sits three lines above "If this is not the first pass,
/// the decoder adds decoded HF coefficients to previously-decoded ones", and
/// the phrase "decoded coefficient" most plausibly names the symbol this pass
/// just decoded — a decoder that consulted the accumulator would have to keep
/// the previous passes' values live inside the entropy loop, and the
/// accumulator is not mentioned in that loop at all.
///
/// `true` selects the current-pass reading. Flipping it changes the decode of
/// any frame with `num_passes > 1` and nothing else.
/// **NOT DISCRIMINATED by 8F's end-to-end probe (2026-08-03)**: every stream
/// reachable today is single-pass, where the two readings coincide.
///
/// The probe also found the `false` arm to be degenerate — it was written
/// `PREV_USES_CURRENT_PASS_COEFFICIENT && ucoeff != 0`, a constant `false`
/// rather than the accumulator reading, so flipping the constant broke the
/// decoder without testing the question. **Fixed 2026-08-03**: the decision now
/// goes through [`next_prev`], whose `false` arm consults the accumulated
/// multi-pass coefficient at the same order position, and
/// [`decode_hf_group_with_prev_reading`] lets a test drive either arm over one
/// bitstream. Settling the question still needs a progressive stream. See
/// `docs/experiments/2026-08-03-vardct-flip-point-probe.md`.
pub const PREV_USES_CURRENT_PASS_COEFFICIENT: bool = true;

// ---------------------------------------------------------------------------
// Coefficient storage
// ---------------------------------------------------------------------------

/// Quantized HF coefficients of one varblock in one channel.
///
/// Laid out exactly like [`jpxl_core::varblock::CoeffMatrix`] — row-major and
/// **landscape**, `rows = transform.coeff_rows()`, `cols =
/// transform.coeff_cols()` — so 8F's dequantization can walk the two in
/// lockstep. It is a separate type because these are integers straight off the
/// wire: nothing here has been through I.5.3's bias adjustment or any
/// dequantization matrix, and confusing the two is precisely the class of
/// mistake the unit-bearing-newtype rule exists to prevent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantCoeffBlock {
    rows: usize,
    cols: usize,
    data: Vec<i32>,
}

impl QuantCoeffBlock {
    /// An all-zero block of the shape `transform` requires.
    #[must_use]
    pub fn zeros(transform: TransformType) -> Self {
        let rows = transform.coeff_rows();
        let cols = transform.coeff_cols();
        Self {
            rows,
            cols,
            data: vec![0; rows * cols],
        }
    }

    /// Number of rows (I.3.2's `bheight`).
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns (I.3.2's `bwidth`).
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// The coefficients in row-major order.
    #[must_use]
    pub fn as_slice(&self) -> &[i32] {
        &self.data
    }

    /// The coefficient at column `x`, row `y`; 0 outside the block.
    #[must_use]
    pub fn at(&self, x: usize, y: usize) -> i32 {
        if x >= self.cols || y >= self.rows {
            return 0;
        }
        self.data.get(y * self.cols + x).copied().unwrap_or(0)
    }

    /// The coefficient at a row-major index, i.e. at an entry of the I.3.1
    /// order vector; 0 outside the block.
    ///
    /// Distinct from [`QuantCoeffBlock::at`], which takes `(x, y)`. I.4's
    /// `prev` question is asked about an *order* position, and the order
    /// vector holds row-major destinations, so this is the accessor that
    /// matches the clause.
    #[must_use]
    pub fn at_index(&self, index: usize) -> i32 {
        self.data.get(index).copied().unwrap_or(0)
    }

    /// Adds `value` at row-major index `index`, as I.4's multi-pass
    /// accumulation requires.
    fn add_at(&mut self, index: usize, value: i32) -> Result<()> {
        let slot = self.data.get_mut(index).ok_or_else(|| {
            DecodeError::out_of_range("coefficient order entry", "I.3.1", index as u64)
        })?;
        *slot = slot.checked_add(value).ok_or_else(|| {
            DecodeError::out_of_range("accumulated HF coefficient", "I.4", i64::from(*slot) as u64)
        })?;
        Ok(())
    }
}

/// One varblock as I.4 needs to see it.
///
/// Built by the caller from G.2.4's placement
/// ([`crate::vardct::hf_meta::VarblockPlacement`]) plus G.2.2's `LfQuant`,
/// because I.4 is defined per *group* while both of those are per *LF group*:
/// the caller selects the placements whose top-left block falls inside the
/// group being decoded, translates their coordinates to be group-relative,
/// and keeps them in raster order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HfVarblock {
    /// Top-left 8x8-block column, relative to the group (G.4.1).
    pub bx: u32,
    /// Top-left 8x8-block row, relative to the group.
    pub by: u32,
    /// `DctSelect`.
    pub transform: TransformType,
    /// `HfMul` — the clause's `qf`.
    pub hf_mul: u32,
    /// The quantized `LfQuant` samples of the varblock's top-left 8x8 block,
    /// in the clause's channel numbering `[X, Y, B]`.
    pub qdc: [i32; 3],
}

/// The quantized HF coefficients of one group, one entry per varblock.
///
/// A per-varblock buffer rather than a group-wide coefficient plane: 8F's
/// loop is "for each varblock, dequantize with this transform's matrix and run
/// its IDCT", and a varblock's coefficients are landscape while its footprint
/// in the group is not, so a plane would need the same reshaping at every use
/// instead of once here. Indices match the `varblocks` slice passed to
/// [`decode_hf_group`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfCoefficients {
    blocks: Vec<[QuantCoeffBlock; NUM_CHANNELS]>,
}

impl HfCoefficients {
    /// Allocates zeroed storage for `varblocks`.
    ///
    /// Multi-pass frames call [`decode_hf_group`] repeatedly against the same
    /// storage; I.4 adds each pass into it.
    ///
    /// # Errors
    ///
    /// A limit error if the storage exceeds what `guard` permits.
    pub fn new(varblocks: &[HfVarblock], guard: &mut AllocGuard) -> Result<Self> {
        let mut bytes: u64 = 0;
        for vb in varblocks {
            let size = vb.transform.coeff_rows() as u64 * vb.transform.coeff_cols() as u64;
            bytes = bytes
                .checked_add(size * 4 * NUM_CHANNELS as u64)
                .ok_or_else(|| {
                    DecodeError::out_of_range("group coefficient size", "I.4", u64::MAX)
                })?;
        }
        guard.charge(bytes)?;
        Ok(Self {
            blocks: varblocks
                .iter()
                .map(|vb| {
                    [
                        QuantCoeffBlock::zeros(vb.transform),
                        QuantCoeffBlock::zeros(vb.transform),
                        QuantCoeffBlock::zeros(vb.transform),
                    ]
                })
                .collect(),
        })
    }

    /// Number of varblocks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Is the group empty of varblocks?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// The coefficients of varblock `varblock` in channel `channel`
    /// (`0 = X`, `1 = Y`, `2 = B`).
    ///
    /// # Errors
    ///
    /// [`DecodeError::FieldOutOfRange`] if either index is out of range.
    pub fn block(&self, varblock: usize, channel: usize) -> Result<&QuantCoeffBlock> {
        self.blocks
            .get(varblock)
            .and_then(|c| c.get(channel))
            .ok_or_else(|| {
                DecodeError::out_of_range("coefficient block index", "I.4", varblock as u64)
            })
    }

    fn block_mut(&mut self, varblock: usize, channel: usize) -> Result<&mut QuantCoeffBlock> {
        self.blocks
            .get_mut(varblock)
            .and_then(|c| c.get_mut(channel))
            .ok_or_else(|| {
                DecodeError::out_of_range("coefficient block index", "I.4", varblock as u64)
            })
    }
}

// ---------------------------------------------------------------------------
// HfPass (I.3) — orders + histograms
// ---------------------------------------------------------------------------

/// The `hf_pass[num_passes]` array of Table G.4.
///
/// The histogram bundle of each pass is read here, in `HfGlobal`, but its
/// entropy-coded stream starts separately in every `PassGroup` section — the
/// same split C.1 forces on the global modular tree. [`decode_hf_group`]
/// performs the C.3.2 restart.
#[derive(Debug)]
pub struct HfPasses {
    natural: NaturalOrders,
    orders: Vec<PassOrders>,
    histograms: Vec<SymbolDecoder>,
}

impl HfPasses {
    /// Number of passes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.orders.len()
    }

    /// Are there no passes at all?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.orders.is_empty()
    }

    /// The frame's natural coefficient orders.
    #[must_use]
    pub const fn natural(&self) -> &NaturalOrders {
        &self.natural
    }

    /// Splits out pass `pass`: its order tables (borrowed immutably alongside
    /// the shared natural orders) and its histogram decoder (borrowed
    /// mutably, because decoding a group advances its ANS state).
    ///
    /// # Errors
    ///
    /// [`DecodeError::FieldOutOfRange`] if `pass` is not below [`len`](Self::len).
    pub fn split_pass(&mut self, pass: usize) -> Result<(OrderLookup<'_>, &mut SymbolDecoder)> {
        let Self {
            natural,
            orders,
            histograms,
        } = self;
        let orders = orders
            .get(pass)
            .ok_or_else(|| DecodeError::out_of_range("pass index", "I.3", pass as u64))?;
        let histograms = histograms
            .get_mut(pass)
            .ok_or_else(|| DecodeError::out_of_range("pass index", "I.3", pass as u64))?;
        Ok((OrderLookup::new(natural, orders), histograms))
    }
}

/// Reads `hf_pass[num_passes]` (18181-1 I.3), the third row of Table G.4.
///
/// Call this immediately after
/// [`read_hf_global_params`](crate::vardct::dequant_matrix::read_hf_global_params),
/// with `nb_block_ctx` from the I.2.2 model in `LfGlobal` and
/// `num_hf_presets` from that same call.
///
/// # Errors
///
/// * [`DecodeError::Bitstream`] on truncation.
/// * [`DecodeError::Entropy`] if a permutation stream or a histogram bundle is
///   malformed.
/// * [`DecodeError::FieldOutOfRange`] if the histogram count overflows.
pub fn read_hf_passes(
    reader: &mut BitReader<'_>,
    num_passes: u32,
    nb_block_ctx: usize,
    num_hf_presets: u32,
    guard: &mut AllocGuard,
) -> Result<HfPasses> {
    let num_dist = CONTEXTS_PER_BLOCK_CTX
        .checked_mul(nb_block_ctx)
        .and_then(|n| n.checked_mul(num_hf_presets as usize))
        .ok_or_else(|| {
            DecodeError::out_of_range("HF histogram count", "I.3.3", u64::from(num_hf_presets))
        })?;

    let natural = NaturalOrders::new();
    let mut orders = Vec::with_capacity(num_passes as usize);
    let mut histograms = Vec::with_capacity(num_passes as usize);
    for _ in 0..num_passes {
        orders.push(read_hf_coeff_orders(reader, &natural, guard)?);
        histograms.push(SymbolDecoder::open_deferred(reader, num_dist, guard)?);
    }

    Ok(HfPasses {
        natural,
        orders,
        histograms,
    })
}

// ---------------------------------------------------------------------------
// The context model (I.4)
// ---------------------------------------------------------------------------

/// I.4's `BlockContext()`.
///
/// `channel` is the clause's `c` (`0 = X`, `1 = Y`, `2 = B`), `order_id` its
/// `s`, `qf` the varblock's `HfMul` and `qdc` the quantized LF samples of its
/// top-left block in `[X, Y, B]` order.
///
/// `lf_idx_is_zero` implements G.2.2's `kUseLfFrame` rule: when the LF comes
/// from a reference frame, `lf_idx` is zero regardless of the thresholds.
///
/// # Errors
///
/// [`DecodeError::FieldOutOfRange`] if the composite index falls outside
/// `block_ctx_map` — which a conforming stream cannot cause, since the map's
/// length is exactly the product of the threshold counts this function walks.
fn block_context(
    model: &HfBlockContext,
    channel: usize,
    order_id: usize,
    qf: u32,
    qdc: [i32; NUM_CHANNELS],
    lf_idx_is_zero: bool,
) -> Result<usize> {
    // (c < 2 ? c ^ 1 : 2) * 13 + s — the Y/X swap of the decode order.
    let channel_class = if channel < 2 { channel ^ 1 } else { 2 };
    let mut idx = channel_class * BLOCK_CTX_SHAPE_CLASSES + order_id;

    idx *= model.qf_thresholds().len() + 1;
    for &t in model.qf_thresholds() {
        if qf > t {
            idx += 1;
        }
    }

    for i in 0..NUM_CHANNELS {
        idx *= model.lf_thresholds(i).len() + 1;
    }

    // The LF walk order is 0, 2, 1 — not 0, 1, 2.
    let mut lf_idx = 0usize;
    if !lf_idx_is_zero {
        for &t in model.lf_thresholds(0) {
            if qdc[0] > t {
                lf_idx += 1;
            }
        }
        lf_idx *= model.lf_thresholds(2).len() + 1;
        for &t in model.lf_thresholds(2) {
            if qdc[2] > t {
                lf_idx += 1;
            }
        }
        lf_idx *= model.lf_thresholds(1).len() + 1;
        for &t in model.lf_thresholds(1) {
            if qdc[1] > t {
                lf_idx += 1;
            }
        }
    }

    model.block_context(idx + lf_idx)
}

/// I.4's `NonZerosContext(predicted)`, without the histogram offset.
///
/// The result is below `37 * nb_block_ctx` for every `predicted`.
const fn non_zeros_context(block_ctx: usize, nb_block_ctx: usize, predicted: u32) -> usize {
    let predicted = if predicted > 64 { 64 } else { predicted } as usize;
    let multiplier = if predicted < 8 {
        predicted
    } else {
        4 + predicted / 2
    };
    block_ctx + nb_block_ctx * multiplier
}

/// I.4's `CoefficientContext(k, non_zeros, num_blocks, size, prev)`, without
/// the histogram offset.
///
/// # Errors
///
/// [`DecodeError::FieldOutOfRange`] if the reduced indices reach 64, which
/// means `non_zeros` exceeded what the remaining positions can hold — a
/// malformed stream, and the only way either constant table could be indexed
/// out of range.
fn coefficient_context(
    block_ctx: usize,
    nb_block_ctx: usize,
    k: usize,
    non_zeros: u32,
    num_blocks: usize,
    prev: usize,
) -> Result<usize> {
    let nz = (non_zeros as usize).div_ceil(num_blocks);
    let kk = k / num_blocks;
    if nz >= COEFF_NUM_NONZERO_CONTEXT.len() || kk >= COEFF_FREQ_CONTEXT.len() {
        return Err(DecodeError::out_of_range(
            "non_zeros",
            "I.4",
            u64::from(non_zeros),
        ));
    }
    let within = (COEFF_NUM_NONZERO_CONTEXT[nz] + COEFF_FREQ_CONTEXT[kk]) as usize * 2 + prev;
    Ok(within + block_ctx * COEFFICIENT_CONTEXTS + NON_ZEROS_CONTEXTS * nb_block_ctx)
}

/// `UnpackSigned(u)` (18181-1 B.3): `u / 2` if even, `-(u + 1) / 2` if odd.
const fn unpack_signed(u: u32) -> i32 {
    if u.is_multiple_of(2) {
        0i32.wrapping_add_unsigned(u / 2)
    } else {
        0i32.wrapping_sub_unsigned(u / 2 + 1)
    }
}

// ---------------------------------------------------------------------------
// Decoding one group (I.4)
// ---------------------------------------------------------------------------

/// Everything [`decode_hf_group`] needs that is not per-varblock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HfGroupParams {
    /// `shift[pass]` from Table F.6: the left shift applied to this pass's
    /// coefficients immediately after entropy decoding. The final pass is 0.
    pub shift: u32,
    /// Width of the group's 8x8-block grid — the extent of the `NonZeros`
    /// array `PredictedNonZeros` reads.
    pub blocks_w: u32,
    /// Height of the group's 8x8-block grid.
    pub blocks_h: u32,
    /// I.2.6 `num_hf_presets`; sets the width of the `hfp` field.
    pub num_hf_presets: u32,
    /// G.2.2's `kUseLfFrame` rule: force `lf_idx` to zero.
    pub lf_idx_is_zero: bool,
}

/// The per-channel `NonZeros` array of I.4, group-relative.
struct NonZerosGrid {
    width: usize,
    height: usize,
    values: [Vec<u32>; NUM_CHANNELS],
}

impl NonZerosGrid {
    fn new(width: usize, height: usize, guard: &mut AllocGuard) -> Result<Self> {
        let cells = width
            .checked_mul(height)
            .ok_or_else(|| DecodeError::out_of_range("group block grid", "I.4", width as u64))?;
        guard.charge(cells as u64 * 4 * NUM_CHANNELS as u64)?;
        Ok(Self {
            width,
            height,
            values: [vec![0; cells], vec![0; cells], vec![0; cells]],
        })
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

/// Decodes the HF coefficients of one `PassGroup` section (18181-1 I.4).
///
/// `reader` must be positioned at the start of the `HF coefficients` row of
/// Table G.5 — i.e. at the `hfp` field, which this function reads before
/// restarting the pass's entropy stream. On return the reader sits just past
/// the last symbol, where G.4.2's modular group data (if any) begins.
///
/// `varblocks` are the group's varblocks in raster order of their top-left
/// block, with group-relative coordinates. `coefficients` must have been built
/// from the same slice; for a multi-pass frame, pass the same storage to every
/// pass and I.4's accumulation happens here.
///
/// # Errors
///
/// * [`DecodeError::Bitstream`] on truncation.
/// * [`DecodeError::Entropy`] if the stream is malformed or does not end in
///   the C.3.2 terminal state.
/// * [`DecodeError::FieldOutOfRange`] if a varblock lies outside the group
///   grid, if `non_zeros` exceeds what the varblock can hold, if the promised
///   number of non-zero coefficients is not delivered, or if an order table
///   does not match its transform.
/// * [`DecodeError::Unsupported`] if the coefficient storage does not match
///   `varblocks`.
#[expect(
    clippy::too_many_arguments,
    reason = "I.4 genuinely takes this much context: the stream, the pass's \
              orders and histograms, the I.2.2 model, the group's varblocks, \
              the accumulator and the allocation guard. Bundling them would \
              only move the list out of view."
)]
pub fn decode_hf_group(
    reader: &mut BitReader<'_>,
    params: &HfGroupParams,
    varblocks: &[HfVarblock],
    orders: &OrderLookup<'_>,
    histograms: &mut SymbolDecoder,
    model: &HfBlockContext,
    coefficients: &mut HfCoefficients,
    guard: &mut AllocGuard,
) -> Result<()> {
    decode_hf_group_with_prev_reading(
        reader,
        params,
        varblocks,
        orders,
        histograms,
        model,
        coefficients,
        PREV_USES_CURRENT_PASS_COEFFICIENT,
        guard,
    )
}

/// [`decode_hf_group`], with [`PREV_USES_CURRENT_PASS_COEFFICIENT`] as an
/// explicit parameter rather than baked in.
///
/// Exists so the flip point is a real branch a test can drive both ways on one
/// bitstream, the same way `vardct::lf`'s channel-order flip point is driven.
/// `true` reads `prev` from the symbol this pass decoded; `false` reads it from
/// the accumulated multi-pass coefficient at the same order position.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors decode_hf_group's own list plus the one flip-point bool"
)]
fn decode_hf_group_with_prev_reading(
    reader: &mut BitReader<'_>,
    params: &HfGroupParams,
    varblocks: &[HfVarblock],
    orders: &OrderLookup<'_>,
    histograms: &mut SymbolDecoder,
    model: &HfBlockContext,
    coefficients: &mut HfCoefficients,
    prev_uses_current_pass: bool,
    guard: &mut AllocGuard,
) -> Result<()> {
    if coefficients.len() != varblocks.len() {
        return Err(DecodeError::out_of_range(
            "coefficient storage length",
            "I.4",
            coefficients.len() as u64,
        ));
    }
    let nb_block_ctx = model.nb_block_ctx();

    // I.4: hfp = u(ceil(log2(num_hf_presets))), offset = 495 * nb_block_ctx * hfp.
    let preset_bits = ceil_log2(params.num_hf_presets.max(1));
    let hfp = trace_field!(reader, "hf_coeff.hfp", reader.read_bits(preset_bits))?;
    if hfp >= params.num_hf_presets.max(1) {
        return Err(DecodeError::out_of_range("hfp", "I.4", u64::from(hfp)));
    }
    let offset = CONTEXTS_PER_BLOCK_CTX * nb_block_ctx * hfp as usize;

    // C.3.2: the histograms came from HfGlobal; the ANS state starts here.
    histograms.restart(reader)?;

    let mut grid = NonZerosGrid::new(params.blocks_w as usize, params.blocks_h as usize, guard)?;

    for (index, vb) in varblocks.iter().enumerate() {
        let (bx, by) = (vb.bx as usize, vb.by as usize);
        let (block_rows, block_cols) = vb.transform.block_dims();
        if bx + block_cols > grid.width || by + block_rows > grid.height {
            return Err(DecodeError::out_of_range(
                "varblock outside the group grid",
                "I.4",
                vb.bx as u64,
            ));
        }

        let order_id = vb.transform.order_id();
        let num_blocks = vb.transform.num_blocks();
        let size = vb.transform.coeff_rows() * vb.transform.coeff_cols();
        // Table I.1's largest varblock is 256x256 samples, so both fit u32.
        let num_blocks_u32 = u32::try_from(num_blocks)
            .map_err(|_| DecodeError::out_of_range("num_blocks", "I.4", num_blocks as u64))?;
        let size_u32 = u32::try_from(size)
            .map_err(|_| DecodeError::out_of_range("size", "I.4", size as u64))?;

        for &channel in &CHANNEL_DECODE_ORDER {
            let block_ctx = block_context(
                model,
                channel,
                order_id,
                vb.hf_mul,
                vb.qdc,
                params.lf_idx_is_zero,
            )?;

            let predicted = grid.predicted(channel, bx, by);
            let ctx = non_zeros_context(block_ctx, nb_block_ctx, predicted) + offset;
            let non_zeros = histograms.read_uint(reader, ctx)?;

            // A conforming stream cannot promise more non-zero coefficients
            // than the HF positions of the varblock can hold; the equality
            // size == 64 * num_blocks then keeps both constant tables in
            // range. Rejecting here is what makes that guarantee load-bearing.
            let capacity = size_u32 - num_blocks_u32;
            if non_zeros > capacity {
                return Err(DecodeError::out_of_range(
                    "non_zeros",
                    "I.4",
                    u64::from(non_zeros),
                ));
            }

            let per_block = non_zeros.div_ceil(num_blocks_u32);
            for j in 0..block_rows {
                for i in 0..block_cols {
                    grid.set(channel, bx + i, by + j, per_block);
                }
            }

            if non_zeros == 0 {
                continue;
            }

            let order = orders.order(order_id, channel)?;
            if order.len() != size {
                return Err(DecodeError::out_of_range(
                    "coefficient order length",
                    "I.3.1",
                    order.len() as u64,
                ));
            }

            let mut remaining = non_zeros;
            let mut previous_was_nonzero = non_zeros <= size_u32 / 16;
            let mut k = num_blocks;
            while k < size {
                let prev = usize::from(previous_was_nonzero);
                let ctx =
                    coefficient_context(block_ctx, nb_block_ctx, k, remaining, num_blocks, prev)?
                        + offset;
                let ucoeff = histograms.read_uint(reader, ctx)?;
                if ucoeff != 0 {
                    let value = unpack_signed(ucoeff);
                    let shifted = value.checked_shl(params.shift).ok_or_else(|| {
                        DecodeError::out_of_range("pass shift", "F.2", u64::from(params.shift))
                    })?;
                    let cell = order[k] as usize;
                    coefficients
                        .block_mut(index, channel)?
                        .add_at(cell, shifted)?;
                    remaining -= 1;
                    if remaining == 0 {
                        break;
                    }
                }
                // Read *after* the add above, so the accumulator arm sees this
                // pass's contribution plus every earlier pass's.
                let accumulated = coefficients
                    .block(index, channel)?
                    .at_index(order[k] as usize);
                previous_was_nonzero = next_prev(prev_uses_current_pass, ucoeff, accumulated);
                k += 1;
            }

            if remaining != 0 {
                return Err(DecodeError::out_of_range(
                    "undelivered non-zero coefficients",
                    "I.4",
                    u64::from(remaining),
                ));
            }
        }
    }

    // C.3.2: the section's HF stream must land exactly on its terminal state.
    histograms.finish()?;
    Ok(())
}

/// I.4's `prev` for the *next* order position, under either reading of
/// [`PREV_USES_CURRENT_PASS_COEFFICIENT`].
///
/// `ucoeff` is the raw symbol this pass decoded at the current position;
/// `accumulated` is the coefficient stored at the same position after this
/// pass's contribution was added, i.e. the sum over all passes so far.
///
/// The two arms coincide whenever the accumulator was zero before the pass:
/// `UnpackSigned(u) == 0` exactly when `u == 0`, and the left shift of F.2
/// cannot turn a non-zero into a zero without overflowing (which is rejected),
/// so `accumulated != 0` reduces to `ucoeff != 0`. They differ only for a
/// progressive frame in which an earlier pass already wrote this position —
/// which is the whole question the flip point asks.
const fn next_prev(prev_uses_current_pass: bool, ucoeff: u32, accumulated: i32) -> bool {
    if prev_uses_current_pass {
        ucoeff != 0
    } else {
        accumulated != 0
    }
}

/// `ceil(log2(n))` for `n >= 1`.
const fn ceil_log2(n: u32) -> u32 {
    if n <= 1 { 0 } else { (n - 1).ilog2() + 1 }
}

#[cfg(test)]
#[allow(
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    clippy::unwrap_used,
    reason = "tests index fixed-size tables they just built and truncate only \
              test-chosen small constants; the range loops index the two \
              64-entry constant tables by the very index whose structure is \
              being asserted, which enumerate() would obscure; a panic here \
              is a failing test"
)]
mod tests {
    use super::*;
    use jpxl_core::limits::Limits;

    // ------------------------------------------------------------------
    // The constant tables
    // ------------------------------------------------------------------

    #[test]
    fn constant_tables_match_the_printed_runs() {
        // Proves the transcription of two tables that straddle a page break
        // in every source: the run-length structure is what an OCR drop or
        // duplication changes first.
        assert_eq!(COEFF_FREQ_CONTEXT.len(), 64);
        assert_eq!(COEFF_NUM_NONZERO_CONTEXT.len(), 64);

        // CoeffFreqContext: 0,0 then 1..14, then pairs 15..22, then runs of 4
        // for 23..30.
        assert_eq!(&COEFF_FREQ_CONTEXT[0..2], &[0, 0]);
        for k in 2..16 {
            assert_eq!(COEFF_FREQ_CONTEXT[k], k as u32 - 1);
        }
        for k in 16..32 {
            assert_eq!(COEFF_FREQ_CONTEXT[k], 15 + (k as u32 - 16) / 2);
        }
        for k in 32..64 {
            assert_eq!(COEFF_FREQ_CONTEXT[k], 23 + (k as u32 - 32) / 4);
        }

        // CoeffNumNonzeroContext run lengths, in order.
        let mut runs: Vec<(u32, usize)> = Vec::new();
        for &v in &COEFF_NUM_NONZERO_CONTEXT {
            match runs.last_mut() {
                Some((value, count)) if *value == v => *count += 1,
                _ => runs.push((v, 1)),
            }
        }
        assert_eq!(
            runs,
            vec![
                (0, 2),
                (31, 1),
                (62, 2),
                (93, 4),
                (123, 4),
                (152, 8),
                (180, 12),
                (206, 31),
            ]
        );
    }

    // ------------------------------------------------------------------
    // Context bounds
    // ------------------------------------------------------------------

    #[test]
    fn non_zeros_context_stays_below_thirty_seven_per_block_ctx() {
        // Proves NON_ZEROS_CONTEXTS: over every block context and every
        // `predicted` a stream can produce (the clause caps it at 64, but a
        // decoder must survive any u32), the context stays inside the first
        // 37 * nb_block_ctx slots of the histogram.
        for nb in 1..=16usize {
            for block_ctx in 0..nb {
                for predicted in (0..300u32).chain([u32::MAX, u32::MAX - 1]) {
                    let ctx = non_zeros_context(block_ctx, nb, predicted);
                    assert!(
                        ctx < NON_ZEROS_CONTEXTS * nb,
                        "nb={nb} block_ctx={block_ctx} predicted={predicted} -> {ctx}"
                    );
                }
            }
        }
    }

    #[test]
    fn within_block_context_bound_is_tight() {
        // Proves COEFFICIENT_CONTEXTS: over all *reachable* (k, non_zeros,
        // prev) the within-block index stays below 458, and its maximum is
        // exactly 457. Tightness is the real evidence — it says the two
        // constant tables and the literal 458 agree to the last unit, which a
        // single mistranscribed entry would break.
        //
        // Reachability: at loop index k the running `non_zeros` counts the
        // non-zero coefficients still to come in [k, size), so
        // non_zeros <= size - k. With size == 64 * num_blocks that reduces to
        // nz' <= 64 - k' in the divided coordinates the clause uses, and
        // k >= num_blocks makes k' >= 1.
        let mut max_within = 0usize;
        for kk in 1..64usize {
            for nz in 1..=(64 - kk) {
                for prev in 0..2usize {
                    let within = (COEFF_NUM_NONZERO_CONTEXT[nz] + COEFF_FREQ_CONTEXT[kk]) as usize
                        * 2
                        + prev;
                    assert!(within < COEFFICIENT_CONTEXTS, "k'={kk} nz'={nz}");
                    max_within = max_within.max(within);
                }
            }
        }
        assert_eq!(max_within, COEFFICIENT_CONTEXTS - 1);
        // Hand-derived witness: nz' = 33 is the first index of the 206 run,
        // k' = 31 the last index of the 22 run, and 33 + 31 = 64 is exactly
        // the reachability limit. 206 + 22 = 228, 228 * 2 + 1 = 457.
        assert_eq!(COEFF_NUM_NONZERO_CONTEXT[33], 206);
        assert_eq!(COEFF_FREQ_CONTEXT[31], 22);
        assert_eq!(
            (COEFF_NUM_NONZERO_CONTEXT[33] + COEFF_FREQ_CONTEXT[31]) as usize * 2 + 1,
            COEFFICIENT_CONTEXTS - 1
        );
    }

    #[test]
    fn the_two_spans_tile_the_four_hundred_and_ninety_five() {
        // Proves the I.3.3 histogram count is exactly the two spans of I.4
        // laid end to end, with no gap and no overlap: the highest
        // NonZerosContext is one below the lowest CoefficientContext, and the
        // highest CoefficientContext is one below 495 * nb_block_ctx.
        assert_eq!(CONTEXTS_PER_BLOCK_CTX, 495);
        for nb in 1..=16usize {
            let highest_nz = non_zeros_context(nb - 1, nb, 64);
            assert_eq!(highest_nz, NON_ZEROS_CONTEXTS * nb - 1);

            let lowest_coeff = coefficient_context(0, nb, 1, 1, 1, 0).expect("in range");
            assert_eq!(lowest_coeff, highest_nz + 1);

            let highest_coeff = coefficient_context(nb - 1, nb, 31, 33, 1, 1).expect("in range");
            assert_eq!(highest_coeff, CONTEXTS_PER_BLOCK_CTX * nb - 1);
        }
    }

    #[test]
    fn coefficient_context_rejects_an_impossible_non_zeros() {
        // The only way either 64-entry table could be indexed out of range is
        // a stream claiming more non-zeros than the block can hold; that is an
        // error, not a panic.
        assert!(coefficient_context(0, 15, 1, 64 * 4, 1, 0).is_err());
    }

    // ------------------------------------------------------------------
    // BlockContext
    // ------------------------------------------------------------------

    #[test]
    fn block_context_swaps_x_and_y_only() {
        // Proves the `c < 2 ? c ^ 1 : 2` mapping: with the default map (three
        // 13-entry rows, X and B identical) the Y channel takes row 0 and the
        // X channel row 1, which is the swap relative to the X = 0 numbering.
        let model = HfBlockContext::default();
        let default_map = crate::vardct::block_ctx::DEFAULT_BLOCK_CTX_MAP;
        for order_id in 0..BLOCK_CTX_SHAPE_CLASSES {
            let y = block_context(&model, 1, order_id, 1, [0; 3], false).expect("ctx");
            let x = block_context(&model, 0, order_id, 1, [0; 3], false).expect("ctx");
            let b = block_context(&model, 2, order_id, 1, [0; 3], false).expect("ctx");
            assert_eq!(y, usize::from(default_map[order_id]));
            assert_eq!(x, usize::from(default_map[13 + order_id]));
            assert_eq!(b, usize::from(default_map[26 + order_id]));
        }
    }

    #[test]
    fn block_context_stays_below_nb_block_ctx() {
        // The map's values are cluster indices, so every BlockContext is a
        // valid multiplier for the two spans above. This is what makes the
        // 495 * nb_block_ctx histogram sufficient.
        let model = HfBlockContext::default();
        for order_id in 0..BLOCK_CTX_SHAPE_CLASSES {
            for channel in 0..NUM_CHANNELS {
                let ctx =
                    block_context(&model, channel, order_id, 7, [3, -2, 9], false).expect("ctx");
                assert!(ctx < model.nb_block_ctx());
            }
        }
    }

    #[test]
    fn use_lf_frame_forces_lf_idx_to_zero() {
        // G.2.2's rule, exercised against the default model where it cannot
        // change anything (no thresholds) plus the invariant that the flag
        // only ever removes the lf_idx term.
        let model = HfBlockContext::default();
        for order_id in 0..BLOCK_CTX_SHAPE_CLASSES {
            let with = block_context(&model, 1, order_id, 1, [9, 9, 9], false).expect("ctx");
            let without = block_context(&model, 1, order_id, 1, [9, 9, 9], true).expect("ctx");
            assert_eq!(with, without);
        }
    }

    // ------------------------------------------------------------------
    // PredictedNonZeros
    // ------------------------------------------------------------------

    #[test]
    fn predicted_non_zeros_follows_the_clause() {
        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let mut grid = NonZerosGrid::new(4, 4, &mut guard).expect("grid");
        assert_eq!(grid.predicted(1, 0, 0), 32, "the corner is seeded at 32");
        grid.set(1, 0, 0, 10);
        grid.set(1, 1, 0, 20);
        grid.set(1, 0, 1, 6);
        assert_eq!(grid.predicted(1, 1, 1), (20 + 6 + 1) >> 1);
        assert_eq!(grid.predicted(1, 0, 1), 10, "column 0 looks up");
        assert_eq!(grid.predicted(1, 1, 0), 10, "row 0 looks left");
        // Channels are independent.
        assert_eq!(grid.predicted(0, 1, 1), 0);
    }

    // ------------------------------------------------------------------
    // UnpackSigned
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // A hand-built HfPass + PassGroup
    // ------------------------------------------------------------------

    /// Bits in stream order, packed LSB-first per byte.
    ///
    /// A local copy rather than `crate::testsupport::BitWriter`, which uses
    /// the opposite `(n, value)` argument order and has no MSB-first writer;
    /// the prefix-code helper below needs both.
    #[derive(Debug, Default)]
    struct BitWriter {
        bits: Vec<bool>,
    }

    impl BitWriter {
        fn u(&mut self, value: u32, n: u32) {
            for i in 0..n {
                self.bits.push((value >> i) & 1 == 1);
            }
        }

        fn bit(&mut self, value: bool) {
            self.bits.push(value);
        }

        /// The order a canonical prefix code is consumed in.
        fn code_msb_first(&mut self, value: u32, n: u32) {
            for i in (0..n).rev() {
                self.bits.push((value >> i) & 1 == 1);
            }
        }

        fn finish(&self) -> Vec<u8> {
            let mut out = vec![0u8; self.bits.len().div_ceil(8) + 8];
            for (i, &bit) in self.bits.iter().enumerate() {
                if bit {
                    out[i / 8] |= 1 << (i % 8);
                }
            }
            out
        }
    }

    /// The smallest C.2 bundle that can carry four symbols: every one of
    /// `num_dist` contexts clustered into one distribution, a hybrid-uint
    /// config whose tokens below `1 << 2` are literal values, and RFC 7932's
    /// simple prefix code over `{0, 1, 2, 3}` with the balanced `[2, 2, 2, 2]`
    /// length pattern.
    ///
    /// Clustering every context together is deliberate: it makes the *decode*
    /// independent of the context arithmetic, so this test isolates the write
    /// positions and the `non_zeros` bookkeeping. The context values are
    /// proved by the bound tests above and, end to end, by the real-fixture
    /// ANS gate below.
    fn write_four_symbol_bundle(w: &mut BitWriter, num_dist: usize) {
        w.bit(false); // Table C.1: lz77.enabled = false
        if num_dist > 1 {
            w.bit(true); // C.2.2: simple clustering
            w.u(0, 2); // nbits = 0 -> every context is cluster 0
        }
        w.bit(true); // C.2.1: use_prefix_code

        // C.2.3: split_exponent = 2, msb_in_token = lsb_in_token = 0.
        w.u(2, 4);
        w.u(0, 2);
        w.u(0, 2);

        // C.2.1: alphabet_size = 1 + (1 << 1) + u(1) with the extra bit set.
        w.bit(true);
        w.u(1, 4);
        w.u(1, 1);

        // C.2.4 / RFC 7932 3.4: the simple code, four symbols, balanced.
        w.u(1, 2);
        w.u(3, 2);
        for symbol in 0..4u32 {
            w.u(symbol, 2);
        }
        w.bit(false);
    }

    fn write_symbol(w: &mut BitWriter, symbol: u32) {
        assert!(symbol < 4);
        w.code_msb_first(symbol, 2);
    }

    #[test]
    fn hand_built_group_writes_coefficients_at_the_order_positions() {
        // Proves the whole I.4 loop shape on a stream derived by hand:
        //
        //   * `used_orders == 0` leaves the DCT8x8 order at its natural value
        //     0, 1, 8, 16, 9, 2, ... (I.3.2);
        //   * the channel loop is Y, X, B — the Y symbols come first;
        //   * `k` starts at `num_blocks` (1 here), so the first coefficient
        //     symbol lands at order position 1, i.e. coefficient cell 1;
        //   * `UnpackSigned` maps 2 -> +1 and 1 -> -1;
        //   * decoding stops the moment `non_zeros` reaches 0, leaving the
        //     remaining 60 positions untouched;
        //   * `order[k]` is the *destination* — cell 8 and cell 16 are
        //     written, not cells 2 and 3.
        let nb_block_ctx = 15usize;
        let mut w = BitWriter::default();

        // --- HfGlobal: hf_pass[0] ---
        w.u(2, 2); // I.3.1 used_orders: U32 distribution 2 = the constant 0
        write_four_symbol_bundle(&mut w, CONTEXTS_PER_BLOCK_CTX * nb_block_ctx);

        // --- PassGroup: HF coefficients (hfp is 0 bits, num_hf_presets = 1) ---
        write_symbol(&mut w, 2); // Y non_zeros = 2
        write_symbol(&mut w, 0); // k = 1: ucoeff 0     -> cell 1  = 0
        write_symbol(&mut w, 2); // k = 2: ucoeff 2     -> cell 8  = +1
        write_symbol(&mut w, 1); // k = 3: ucoeff 1     -> cell 16 = -1, stop
        write_symbol(&mut w, 0); // X non_zeros = 0
        write_symbol(&mut w, 0); // B non_zeros = 0

        let data = w.finish();
        let mut reader = BitReader::new(&data);
        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);

        let mut passes = read_hf_passes(&mut reader, 1, nb_block_ctx, 1, &mut guard).expect("pass");
        assert_eq!(passes.len(), 1);

        let varblocks = [HfVarblock {
            bx: 0,
            by: 0,
            transform: TransformType::Dct8x8,
            hf_mul: 1,
            qdc: [0; 3],
        }];
        let mut coefficients = HfCoefficients::new(&varblocks, &mut guard).expect("storage");
        let params = HfGroupParams {
            shift: 0,
            blocks_w: 1,
            blocks_h: 1,
            num_hf_presets: 1,
            lf_idx_is_zero: false,
        };
        let model = HfBlockContext::default();
        let (orders, histograms) = passes.split_pass(0).expect("pass 0");
        decode_hf_group(
            &mut reader,
            &params,
            &varblocks,
            &orders,
            histograms,
            &model,
            &mut coefficients,
            &mut guard,
        )
        .expect("decode");

        let y = coefficients.block(0, 1).expect("Y block");
        assert_eq!(y.at(1, 0), 0, "cell 1 decoded a zero");
        assert_eq!(y.at(0, 1), 1, "cell 8 is (x=0, y=1)");
        assert_eq!(y.at(0, 2), -1, "cell 16 is (x=0, y=2)");
        assert_eq!(
            y.as_slice().iter().filter(|v| **v != 0).count(),
            2,
            "exactly the two non-zero coefficients were written"
        );
        assert_eq!(
            y.at(2, 0),
            0,
            "cell 2 was NOT written: order[k] is a target"
        );
        assert_eq!(y.at(3, 0), 0);

        for channel in [0usize, 2] {
            let block = coefficients.block(0, channel).expect("block");
            assert!(block.as_slice().iter().all(|v| *v == 0));
        }
    }

    #[test]
    fn hand_built_group_accumulates_a_second_pass_with_its_shift() {
        // Proves F.2's `shift[i]` is applied to the decoded coefficient and
        // that a later pass adds into the accumulator rather than replacing
        // it (I.4's closing sentence).
        let nb_block_ctx = 15usize;
        let mut w = BitWriter::default();
        w.u(2, 2);
        write_four_symbol_bundle(&mut w, CONTEXTS_PER_BLOCK_CTX * nb_block_ctx);
        write_symbol(&mut w, 1); // Y non_zeros = 1
        write_symbol(&mut w, 0); // k = 1: 0
        write_symbol(&mut w, 2); // k = 2: +1 at cell 8, stop
        write_symbol(&mut w, 0); // X
        write_symbol(&mut w, 0); // B

        let data = w.finish();
        let mut reader = BitReader::new(&data);
        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let mut passes = read_hf_passes(&mut reader, 1, nb_block_ctx, 1, &mut guard).expect("pass");

        let varblocks = [HfVarblock {
            bx: 0,
            by: 0,
            transform: TransformType::Dct8x8,
            hf_mul: 1,
            qdc: [0; 3],
        }];
        let mut coefficients = HfCoefficients::new(&varblocks, &mut guard).expect("storage");
        // Seed the accumulator as an earlier pass would have.
        coefficients
            .block_mut(0, 1)
            .expect("block")
            .add_at(8, 5)
            .expect("seed");

        let params = HfGroupParams {
            shift: 3,
            blocks_w: 1,
            blocks_h: 1,
            num_hf_presets: 1,
            lf_idx_is_zero: false,
        };
        let model = HfBlockContext::default();
        let (orders, histograms) = passes.split_pass(0).expect("pass 0");
        decode_hf_group(
            &mut reader,
            &params,
            &varblocks,
            &orders,
            histograms,
            &model,
            &mut coefficients,
            &mut guard,
        )
        .expect("decode");

        assert_eq!(
            coefficients.block(0, 1).expect("Y").at(0, 1),
            5 + (1 << 3),
            "the pass added UnpackSigned(2) << 3 to the seeded 5"
        );
    }

    // ------------------------------------------------------------------
    // PREV_USES_CURRENT_PASS_COEFFICIENT
    // ------------------------------------------------------------------

    #[test]
    fn the_two_prev_readings_are_genuinely_different_functions() {
        // The defect this replaces: the false arm was
        // `PREV_USES_CURRENT_PASS_COEFFICIENT && ucoeff != 0`, which is a
        // constant `false` once the constant is false. Row 3 below is the one
        // that failed under that formulation and is the whole point of the
        // flip point: this pass decoded a zero on top of an earlier pass's
        // non-zero.
        //                        (ucoeff, accumulated) -> (true arm, false arm)
        let cases: [(u32, i32, bool, bool); 5] = [
            (0, 0, false, false), // nothing decoded, nothing accumulated
            (2, 1, true, true),   // first pass writing a fresh position
            (0, 7, false, true),  // ** the discriminating case **
            (2, 0, true, false),  // only reachable if an add overflowed away
            (0, -3, false, true), // sign does not matter, only zero-ness
        ];
        for (ucoeff, accumulated, expect_true_arm, expect_false_arm) in cases {
            assert_eq!(
                next_prev(true, ucoeff, accumulated),
                expect_true_arm,
                "current-pass arm at ({ucoeff}, {accumulated})"
            );
            assert_eq!(
                next_prev(false, ucoeff, accumulated),
                expect_false_arm,
                "accumulator arm at ({ucoeff}, {accumulated})"
            );
        }
        // And the false arm is not constant: it answers both ways.
        assert!(next_prev(false, 0, 7));
        assert!(!next_prev(false, 0, 0));
    }

    #[test]
    fn prev_shifts_the_coefficient_context_by_exactly_one() {
        // Why the reading matters at all: `prev` is added straight into I.4's
        // context index, so a wrong `prev` selects the neighbouring histogram
        // for every subsequent symbol in the block and the ANS stream desyncs.
        // Together with the test above, this is what makes the flip point a
        // real behavioural fork for a progressive frame rather than a comment.
        for (block_ctx, k, non_zeros, num_blocks) in [
            (0usize, 1usize, 1u32, 1usize),
            (14, 40, 9, 1),
            (7, 12, 3, 4),
        ] {
            let c0 = coefficient_context(block_ctx, 15, k, non_zeros, num_blocks, 0).expect("ctx");
            let c1 = coefficient_context(block_ctx, 15, k, non_zeros, num_blocks, 1).expect("ctx");
            assert_eq!(c1, c0 + 1, "prev enters the context additively");
        }
    }

    #[test]
    fn the_two_prev_readings_coincide_on_a_single_pass_stream() {
        // The invariant that keeps every reachable stream — and all eight
        // passing ANS-gate fixtures — bit-identical under either arm: with a
        // zero accumulator the two readings are the same function, so the same
        // bitstream decodes to the same coefficients and both runs reach the
        // C.3.2 terminal state. This is what licenses shipping `true` while the
        // question is open.
        let nb_block_ctx = 15usize;
        let mut w = BitWriter::default();
        w.u(2, 2);
        write_four_symbol_bundle(&mut w, CONTEXTS_PER_BLOCK_CTX * nb_block_ctx);
        write_symbol(&mut w, 2); // Y non_zeros = 2
        write_symbol(&mut w, 0); // k = 1: 0    (prev becomes false either way)
        write_symbol(&mut w, 2); // k = 2: +1   (prev becomes true either way)
        write_symbol(&mut w, 1); // k = 3: -1, stop
        write_symbol(&mut w, 0); // X
        write_symbol(&mut w, 0); // B
        let data = w.finish();

        let decode_with = |prev_uses_current_pass: bool| {
            let mut reader = BitReader::new(&data);
            let limits = Limits::default();
            let mut guard = AllocGuard::new(&limits);
            let mut passes =
                read_hf_passes(&mut reader, 1, nb_block_ctx, 1, &mut guard).expect("pass");
            let varblocks = [HfVarblock {
                bx: 0,
                by: 0,
                transform: TransformType::Dct8x8,
                hf_mul: 1,
                qdc: [0; 3],
            }];
            let mut coefficients = HfCoefficients::new(&varblocks, &mut guard).expect("storage");
            let params = HfGroupParams {
                shift: 0,
                blocks_w: 1,
                blocks_h: 1,
                num_hf_presets: 1,
                lf_idx_is_zero: false,
            };
            let model = HfBlockContext::default();
            let (orders, histograms) = passes.split_pass(0).expect("pass 0");
            decode_hf_group_with_prev_reading(
                &mut reader,
                &params,
                &varblocks,
                &orders,
                histograms,
                &model,
                &mut coefficients,
                prev_uses_current_pass,
                &mut guard,
            )
            .expect("decode");
            coefficients.block(0, 1).expect("Y").as_slice().to_vec()
        };

        let current = decode_with(true);
        let accumulated = decode_with(false);
        assert_eq!(current, accumulated);
        // ...and it is not trivially equal because both are empty.
        assert_eq!(current.iter().filter(|v| **v != 0).count(), 2);
    }

    #[test]
    fn the_accumulator_arm_reads_a_seeded_earlier_pass() {
        // Drives the false arm over a stream whose accumulator is already
        // non-zero at the position `k = 1` visits, so the arm returns `true`
        // where the current-pass arm returns `false`. Proves the arm is wired
        // to real accumulator state rather than being dead code: with the old
        // `&&` formulation this decode took the identical path as `true`.
        //
        // The four-symbol test bundle maps every context to one cluster, so the
        // *decoded* symbols cannot diverge here — that is deliberate, because a
        // diverging context in a single-cluster stream would prove nothing
        // about the histograms. The behavioural consequence is established by
        // `prev_shifts_the_coefficient_context_by_exactly_one` instead.
        let nb_block_ctx = 15usize;
        let mut w = BitWriter::default();
        w.u(2, 2);
        write_four_symbol_bundle(&mut w, CONTEXTS_PER_BLOCK_CTX * nb_block_ctx);
        write_symbol(&mut w, 1); // Y non_zeros = 1
        write_symbol(&mut w, 0); // k = 1: 0 on top of the seeded cell 1
        write_symbol(&mut w, 2); // k = 2: +1 at cell 8, stop
        write_symbol(&mut w, 0); // X
        write_symbol(&mut w, 0); // B
        let data = w.finish();

        let mut reader = BitReader::new(&data);
        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let mut passes = read_hf_passes(&mut reader, 1, nb_block_ctx, 1, &mut guard).expect("pass");
        let varblocks = [HfVarblock {
            bx: 0,
            by: 0,
            transform: TransformType::Dct8x8,
            hf_mul: 1,
            qdc: [0; 3],
        }];
        let mut coefficients = HfCoefficients::new(&varblocks, &mut guard).expect("storage");
        // An earlier pass wrote cell 1 — the very cell `k = 1` revisits.
        coefficients
            .block_mut(0, 1)
            .expect("block")
            .add_at(1, 9)
            .expect("seed");

        let params = HfGroupParams {
            shift: 0,
            blocks_w: 1,
            blocks_h: 1,
            num_hf_presets: 1,
            lf_idx_is_zero: false,
        };
        let model = HfBlockContext::default();
        let (orders, histograms) = passes.split_pass(0).expect("pass 0");
        decode_hf_group_with_prev_reading(
            &mut reader,
            &params,
            &varblocks,
            &orders,
            histograms,
            &model,
            &mut coefficients,
            false,
            &mut guard,
        )
        .expect("decode");

        let y = coefficients.block(0, 1).expect("Y");
        assert_eq!(y.at_index(1), 9, "the seeded value survived a decoded zero");
        assert_eq!(y.at_index(8), 1, "this pass's coefficient landed");
        // The state the arm actually consulted after k = 1.
        assert!(next_prev(false, 0, y.at_index(1)));
        assert!(!next_prev(true, 0, y.at_index(1)));
    }

    #[test]
    fn truncated_pass_group_errors_without_panicking() {
        // Proves malformed input is an error: the stream promises two
        // non-zero coefficients and then ends.
        let nb_block_ctx = 15usize;
        let mut w = BitWriter::default();
        w.u(2, 2);
        write_four_symbol_bundle(&mut w, CONTEXTS_PER_BLOCK_CTX * nb_block_ctx);
        let prefix = w.bits.len();
        write_symbol(&mut w, 2); // Y non_zeros = 2, then nothing

        let mut data = w.finish();
        // Cut the buffer to just past the bundle plus the non_zeros symbol.
        data.truncate((prefix + 2).div_ceil(8));

        let mut reader = BitReader::new(&data);
        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let mut passes = read_hf_passes(&mut reader, 1, nb_block_ctx, 1, &mut guard).expect("pass");
        let varblocks = [HfVarblock {
            bx: 0,
            by: 0,
            transform: TransformType::Dct8x8,
            hf_mul: 1,
            qdc: [0; 3],
        }];
        let mut coefficients = HfCoefficients::new(&varblocks, &mut guard).expect("storage");
        let params = HfGroupParams {
            shift: 0,
            blocks_w: 1,
            blocks_h: 1,
            num_hf_presets: 1,
            lf_idx_is_zero: false,
        };
        let model = HfBlockContext::default();
        let (orders, histograms) = passes.split_pass(0).expect("pass 0");
        assert!(
            decode_hf_group(
                &mut reader,
                &params,
                &varblocks,
                &orders,
                histograms,
                &model,
                &mut coefficients,
                &mut guard,
            )
            .is_err()
        );
    }

    #[test]
    fn a_varblock_outside_the_group_grid_is_rejected() {
        // Proves the group-relative contract of `HfVarblock` is enforced
        // rather than silently wrapping the NonZeros array.
        let nb_block_ctx = 15usize;
        let mut w = BitWriter::default();
        w.u(2, 2);
        write_four_symbol_bundle(&mut w, CONTEXTS_PER_BLOCK_CTX * nb_block_ctx);
        let data = w.finish();
        let mut reader = BitReader::new(&data);
        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let mut passes = read_hf_passes(&mut reader, 1, nb_block_ctx, 1, &mut guard).expect("pass");

        let varblocks = [HfVarblock {
            bx: 3,
            by: 0,
            transform: TransformType::Dct16x16,
            hf_mul: 1,
            qdc: [0; 3],
        }];
        let mut coefficients = HfCoefficients::new(&varblocks, &mut guard).expect("storage");
        let params = HfGroupParams {
            shift: 0,
            blocks_w: 4,
            blocks_h: 4,
            num_hf_presets: 1,
            lf_idx_is_zero: false,
        };
        let model = HfBlockContext::default();
        let (orders, histograms) = passes.split_pass(0).expect("pass 0");
        assert!(
            decode_hf_group(
                &mut reader,
                &params,
                &varblocks,
                &orders,
                histograms,
                &model,
                &mut coefficients,
                &mut guard,
            )
            .is_err()
        );
    }

    // ------------------------------------------------------------------
    // The headline gate: real VarDCT fixtures, bitstream level only
    // ------------------------------------------------------------------

    /// Decodes the HF coefficient streams of a real `kVarDCT` codestream and
    /// nothing else — no dequantization, no IDCT, no pixels.
    ///
    /// This is the strongest oracle-free evidence available for I.4. The
    /// entropy-coded stream of every `PassGroup` section is an ANS stream, and
    /// C.3.2 requires it to end in one specific state. That state is reached
    /// only if every symbol was read with the *same context* the encoder used,
    /// in the same order — so a single wrong `BlockContext`, a swapped channel
    /// loop, an off-by-one in `k`, a mis-signed `prev` or a wrong constant in
    /// either 64-entry table desynchronizes the arithmetic decoder and the
    /// final state is wrong. It needs no reference coefficients at all.
    ///
    /// This is a test-local harness rather than a call into `decode.rs`:
    /// `decode.rs` is 8F's file and is still modular-only.
    mod fixture_gate {
        use super::super::*;
        use crate::frame::stream_index;
        use crate::frame::toc::read_toc;
        use crate::frame::{Encoding, FrameGeometry, FrameType, read_frame_header};
        use crate::modular::{ModularOptions, TreeSource};
        use crate::vardct::dequant_matrix::read_hf_global_params;
        use crate::vardct::hf_meta::{place_varblocks, read_hf_metadata};
        use crate::vardct::lf::read_lf_quant;
        use crate::vardct::quantizer::{read_lf_channel_dequantization, read_lf_global_vardct};
        use jpxl_core::limits::Limits;

        /// What one fixture's HF streams looked like, for the report the test
        /// prints and asserts on.
        #[derive(Debug, Default)]
        struct GateReport {
            varblocks: usize,
            groups: usize,
            /// Unread bits left in each `PassGroup` section after the HF
            /// stream was consumed.
            slack_bits: Vec<u64>,
            transforms: Vec<(u8, usize)>,
            /// I.3.1 `used_orders` per pass: nonzero means the shared
            /// permutation stream was exercised by real encoder output.
            used_orders: Vec<u32>,
            nb_block_ctx: usize,
            num_hf_presets: u32,
        }

        fn section_slice<'a>(
            codestream: &'a [u8],
            base: usize,
            toc: &crate::frame::Toc,
            index: usize,
        ) -> &'a [u8] {
            let offset = toc.offset_of(index).expect("section offset") as usize;
            let entry_index = match &toc.permutation {
                Some(p) => p.get(index).map_or(index, |&t| t as usize),
                None => index,
            };
            let size = toc.entries.get(entry_index).copied().unwrap_or(0) as usize;
            let start = base + offset;
            let end = (start + size).min(codestream.len());
            codestream.get(start..end).expect("section bytes")
        }

        #[allow(clippy::too_many_lines)]
        fn run(path: &std::path::Path) -> GateReport {
            let raw = std::fs::read(path).expect("fixture readable");
            let limits = Limits::default();
            let mut guard = AllocGuard::new(&limits);
            let extracted;
            let codestream: &[u8] = if crate::container::is_container(&raw) {
                extracted =
                    crate::container::extract_codestream(&raw, &mut guard).expect("container");
                &extracted
            } else {
                &raw
            };

            let mut reader = BitReader::new(codestream);
            let headers = crate::decode_image_headers(&mut reader, &limits).expect("image headers");
            if headers.metadata.colour_encoding.want_icc {
                // A.1: the profile sits between the headers and the first
                // frame in the same bit stream; skipping it would shift every
                // frame offset.
                crate::icc::read_icc_profile(&mut reader, &mut guard).expect("ICC profile");
            }
            assert_eq!(
                headers.metadata.num_extra(),
                0,
                "no extra channels expected"
            );
            reader.zero_pad_to_byte().expect("frame alignment");
            let cursor = (reader.total_bits_read() / 8) as usize;

            let mut frame_reader = BitReader::new(codestream.get(cursor..).expect("frame bytes"));
            let header = read_frame_header(
                &mut frame_reader,
                &headers.metadata,
                headers.width(),
                headers.height(),
                &limits,
                &mut guard,
            )
            .expect("frame header");
            assert_eq!(header.encoding, Encoding::VarDct, "fixture must be VarDCT");
            assert_eq!(header.frame_type, FrameType::RegularFrame);
            assert_eq!(
                header.jpeg_upsampling,
                [0, 0, 0],
                "subsampling is out of scope"
            );
            assert_eq!(header.upsampling, 1);

            let geometry = FrameGeometry::from_header(
                &header,
                headers.width(),
                headers.height(),
                &limits,
                &mut guard,
            )
            .expect("geometry");
            let toc = read_toc(
                &mut frame_reader,
                geometry.num_sections(),
                &limits,
                &mut guard,
            )
            .expect("toc");
            let base = cursor + (frame_reader.total_bits_read() / 8) as usize;

            // F.3.1: a frame with one group, one LF group and one pass is a
            // single section carrying every structure consecutively in one bit
            // stream. That makes the gate stricter, not weaker — a bit-position
            // error anywhere shifts everything after it.
            let single = geometry.is_single_section();
            let whole = section_slice(codestream, base, &toc, 0);
            let mut single_reader = BitReader::new(whole);

            // ---- LfGlobal (G.1) ----
            let modular_options = ModularOptions {
                stream_index: 0,
                bits_per_sample: headers.metadata.bit_depth.bits_per_sample(),
                ..ModularOptions::level10()
            };
            let (lf_global, global_tree) = {
                let mut owned;
                let r: &mut BitReader<'_> = if single {
                    &mut single_reader
                } else {
                    owned = BitReader::new(section_slice(codestream, base, &toc, 0));
                    &mut owned
                };
                read_lf_channel_dequantization(r).expect("G.1.2");
                let vardct = read_lf_global_vardct(r, &mut guard).expect("G.1 VarDCT");
                // G.1.3 GlobalModular: the leading Bool() is read whatever the
                // channel count; with no extra channels and no modular colour
                // channels the sub-bitstream itself decodes nothing (H.1).
                let have_global_tree = r.read_bool().expect("G.1.3 Bool()");
                let tree = have_global_tree.then(|| {
                    crate::modular::read_global_tree(r, &modular_options, &mut guard)
                        .expect("G.1.3 global tree")
                });
                (vardct, tree)
            };
            fn tree_source(t: Option<&crate::modular::GlobalTree>) -> TreeSource<'_> {
                match t {
                    Some(global) => TreeSource::Global {
                        global,
                        restart: true,
                    },
                    None => TreeSource::Local,
                }
            }
            // G.1.3 GlobalModular: with no extra channels the modular image
            // has zero channels, and H.1 says the decoder takes no action.

            let group_dim = geometry.group_dim();
            let lf_dim = group_dim * 8;
            let lf_groups_x = geometry.width().div_ceil(lf_dim);

            // ---- LfGroup sections (G.2) ----
            let num_lf_groups = geometry.num_lf_groups();
            let mut lf_quant = Vec::new();
            let mut placements = Vec::new();
            for lf_index in 0..num_lf_groups {
                let rect = geometry.lf_group_rect(lf_index).expect("lf group rect");
                let mut owned;
                let r: &mut BitReader<'_> = if single {
                    &mut single_reader
                } else {
                    owned = BitReader::new(section_slice(
                        codestream,
                        base,
                        &toc,
                        1 + lf_index as usize,
                    ));
                    &mut owned
                };

                let mut options = ModularOptions {
                    stream_index: stream_index::lf_coefficients(&geometry, lf_index)
                        .expect("stream index"),
                    ..modular_options
                };
                let quant = read_lf_quant(
                    r,
                    rect.width,
                    rect.height,
                    header.jpeg_upsampling,
                    &options,
                    tree_source(global_tree.as_ref()),
                    &mut guard,
                )
                .expect("G.2.2 LfQuant");

                // G.2.3 ModularLfGroup: zero channels, nothing read.

                options.stream_index =
                    stream_index::hf_metadata(&geometry, lf_index).expect("stream index");
                let meta = read_hf_metadata(
                    r,
                    rect.width,
                    rect.height,
                    &options,
                    tree_source(global_tree.as_ref()),
                    &mut guard,
                )
                .expect("G.2.4 HfMetadata");
                let placed = place_varblocks(
                    &meta.block_info,
                    rect.width.div_ceil(8),
                    rect.height.div_ceil(8),
                )
                .expect("G.2.4 placement");
                lf_quant.push(quant);
                placements.push(placed);
            }

            // ---- HfGlobal (G.3) ----
            let (params, mut passes) = {
                let mut owned;
                let r: &mut BitReader<'_> = if single {
                    &mut single_reader
                } else {
                    owned = BitReader::new(section_slice(
                        codestream,
                        base,
                        &toc,
                        1 + num_lf_groups as usize,
                    ));
                    &mut owned
                };
                let params = read_hf_global_params(r, geometry.num_groups(), &mut guard)
                    .expect("Table G.4 rows 1-2");
                let passes = read_hf_passes(
                    r,
                    header.passes.num_passes,
                    lf_global.hf_block_ctx.nb_block_ctx(),
                    params.num_hf_presets,
                    &mut guard,
                )
                .expect("I.3 hf_pass");
                (params, passes)
            };

            // ---- PassGroup sections (G.4) ----
            let mut report = GateReport {
                nb_block_ctx: lf_global.hf_block_ctx.nb_block_ctx(),
                num_hf_presets: params.num_hf_presets,
                ..GateReport::default()
            };
            for pass in 0..header.passes.num_passes {
                let (orders, _) = passes.split_pass(pass as usize).expect("pass");
                report.used_orders.push(orders.used_orders());
            }
            let num_groups = geometry.num_groups();
            let mut histogram: std::collections::BTreeMap<u8, usize> =
                std::collections::BTreeMap::new();
            for pass in 0..header.passes.num_passes {
                for group in 0..num_groups {
                    let rect = geometry.group_rect(group).expect("group rect");
                    let lf_index = u64::from(rect.y0 / lf_dim) * u64::from(lf_groups_x)
                        + u64::from(rect.x0 / lf_dim);
                    let lf_rect = geometry.lf_group_rect(lf_index).expect("lf rect");
                    let origin_bx = (rect.x0 - lf_rect.x0) / 8;
                    let origin_by = (rect.y0 - lf_rect.y0) / 8;
                    let blocks_w = rect.width.div_ceil(8);
                    let blocks_h = rect.height.div_ceil(8);

                    let quant = lf_quant.get(lf_index as usize).expect("lf quant");
                    let varblocks: Vec<HfVarblock> = placements
                        .get(lf_index as usize)
                        .expect("placements")
                        .iter()
                        .filter(|p| {
                            let (x, y) = (p.position.bx(), p.position.by());
                            x >= origin_bx
                                && y >= origin_by
                                && x < origin_bx + blocks_w
                                && y < origin_by + blocks_h
                        })
                        .map(|p| {
                            let (x, y) = (p.position.bx(), p.position.by());
                            HfVarblock {
                                bx: x - origin_bx,
                                by: y - origin_by,
                                transform: p.transform,
                                hf_mul: p.hf_mul,
                                qdc: [quant.x.get(x, y), quant.y.get(x, y), quant.b.get(x, y)],
                            }
                        })
                        .collect();

                    if pass == 0 {
                        for vb in &varblocks {
                            *histogram.entry(vb.transform.dct_select()).or_default() += 1;
                        }
                        report.varblocks += varblocks.len();
                        report.groups += 1;
                    }

                    let mut coefficients =
                        HfCoefficients::new(&varblocks, &mut guard).expect("storage");
                    let section = 2 + num_lf_groups + num_groups * u64::from(pass) + group;
                    let mut owned;
                    let slice: &[u8];
                    let r: &mut BitReader<'_> = if single {
                        slice = whole;
                        &mut single_reader
                    } else {
                        slice = section_slice(codestream, base, &toc, section as usize);
                        owned = BitReader::new(slice);
                        &mut owned
                    };
                    let group_params = HfGroupParams {
                        shift: header.passes.shift_for(pass),
                        blocks_w,
                        blocks_h,
                        num_hf_presets: params.num_hf_presets,
                        lf_idx_is_zero: false,
                    };
                    let (orders, histograms) = passes.split_pass(pass as usize).expect("pass");
                    decode_hf_group(
                        r,
                        &group_params,
                        &varblocks,
                        &orders,
                        histograms,
                        &lf_global.hf_block_ctx,
                        &mut coefficients,
                        &mut guard,
                    )
                    .unwrap_or_else(|e| {
                        panic!("pass {pass} group {group}: {e}");
                    });

                    let slack = slice.len() as u64 * 8 - r.total_bits_read();
                    report.slack_bits.push(slack);
                }
            }
            report.transforms = histogram.into_iter().collect();
            report
        }

        fn fixture(name: &str) -> std::path::PathBuf {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/handmade")
                .join(name)
        }

        fn gate(name: &str) {
            let path = fixture(name);
            if !path.exists() {
                panic!("missing fixture {name}");
            }
            let report = run(&path);
            assert!(report.varblocks > 0, "{name}: no varblocks decoded");
            assert!(report.groups > 0);
            // Section exhaustion: nothing but byte padding may remain after
            // the HF stream. A decoder that desynchronized would either fail
            // the C.3.2 check inside `decode_hf_group` or leave whole bytes
            // unread here.
            for (i, slack) in report.slack_bits.iter().enumerate() {
                assert!(
                    *slack < 8,
                    "{name}: pass-group section {i} left {slack} unread bits"
                );
            }
            println!(
                "{name}: {} varblocks over {} groups, DctSelect histogram {:?}, \
                 nb_block_ctx {}, num_hf_presets {}, used_orders {:?}, slack {:?}",
                report.varblocks,
                report.groups,
                report.transforms,
                report.nb_block_ctx,
                report.num_hf_presets,
                report.used_orders,
                report.slack_bits
            );
        }

        #[test]
        fn fixture_50_gray_nofilters_d1() {
            gate("50_vardct_mixed_gray_128x128_nofilters_d1.jxl");
        }

        #[test]
        fn fixture_51_gray_nofilters_d4() {
            gate("51_vardct_mixed_gray_128x128_nofilters_d4.jxl");
        }

        #[test]
        fn fixture_52_gray_filters_d1() {
            gate("52_vardct_mixed_gray_128x128_filters_d1.jxl");
        }

        #[test]
        fn fixture_53_gray_filters_d4() {
            gate("53_vardct_mixed_gray_128x128_filters_d4.jxl");
        }

        #[test]
        // BLOCKED UPSTREAM OF I.4, not by this module. The G.2.2 `LfQuant`
        // modular sub-bitstream of these two fixtures fails its own C.3.2 ANS
        // final-state check inside `read_lf_quant`, so the LfGroup section
        // never reaches G.2.4's HF metadata and no I.4 code runs at all. The
        // sibling fixtures 55 and 56 — same encoder, same geometry, the other
        // two (distance, filters) combinations — decode and pass the gate, so
        // this is an Annex H divergence on specific content, of the same
        // family as the open modular sawtooth bug in HANDOFF.md, not a VarDCT
        // issue. Not `bits_per_sample`: probed at 1, 8, 16, 24 and 32, all
        // fail identically. RESOLVED 2026-08-03: the upstream Annex H bug was
        // H.5.2's clamp guard, which is XOR and not a product (every text
        // transcription misread `^` as `*`); nothing in this file changed.
        fn fixture_54_rgb_nofilters_d1() {
            gate("54_vardct_mixed_rgb_128x128_nofilters_d1.jxl");
        }

        #[test]
        fn fixture_55_rgb_nofilters_d4() {
            gate("55_vardct_mixed_rgb_128x128_nofilters_d4.jxl");
        }

        #[test]
        fn fixture_56_rgb_filters_d1() {
            gate("56_vardct_mixed_rgb_128x128_filters_d1.jxl");
        }

        /// The conformance corpus is gitignored (fetched by
        /// `tools/fetch-conformance.sh`), so these skip rather than fail when
        /// it is absent — the house rule for anything not in the checkout.
        fn corpus_gate(case: &str) {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/conformance/testcases")
                .join(case)
                .join("input.jxl");
            if !path.exists() {
                eprintln!("skipping {case}: conformance corpus not fetched");
                return;
            }
            let report = run(&path);
            assert!(report.varblocks > 0);
            for (i, slack) in report.slack_bits.iter().enumerate() {
                assert!(*slack < 8, "{case}: section {i} left {slack} unread bits");
            }
            // The reason these two cases are here rather than only the
            // handmade fixtures: cjxl never set `used_orders` for the
            // handmade content, so I.3.1's shared permutation stream is
            // otherwise unexercised by real encoder output. This case sets
            // it, and the ANS final state then depends on the permuted order
            // tables being composed in the right direction.
            assert!(
                report.used_orders.iter().any(|m| *m != 0),
                "{case}: expected a non-zero used_orders mask"
            );
            println!(
                "{case}: {} varblocks, used_orders {:?}, slack {:?}",
                report.varblocks, report.used_orders, report.slack_bits
            );
        }

        #[test]
        fn corpus_grayscale_exercises_the_order_permutation() {
            corpus_gate("grayscale");
        }

        #[test]
        fn corpus_grayscale_5_exercises_the_order_permutation() {
            corpus_gate("grayscale_5");
        }

        #[test]
        // Was blocked upstream exactly as fixture 54 above; see that comment.
        fn fixture_57_rgb_filters_d4() {
            gate("57_vardct_mixed_rgb_128x128_filters_d4.jxl");
        }
    }

    #[test]
    fn unpack_signed_matches_b3() {
        assert_eq!(unpack_signed(0), 0);
        assert_eq!(unpack_signed(1), -1);
        assert_eq!(unpack_signed(2), 1);
        assert_eq!(unpack_signed(3), -2);
        assert_eq!(unpack_signed(4), 2);
        // Zero-ness of the coefficient and of the symbol coincide, which is
        // why `prev` can be computed from either.
        for u in 0..1000u32 {
            assert_eq!(unpack_signed(u) == 0, u == 0);
        }
    }
}
