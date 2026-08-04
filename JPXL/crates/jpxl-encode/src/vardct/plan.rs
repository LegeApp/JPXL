//! The staged VarDCT plan IR (`docs/Encoder-plan1.md` §2, milestone 1).
//!
//! `Encoder-plan1.md` stages the encoder as
//!
//! ```text
//! PreparedFrame -> AnalysisAtlas -> SpatialPlan -> QuantizedFrameIr
//!               -> EntropyPlan   -> EmissionPlan
//! ```
//!
//! The first two are *search inputs*: they hold source pixels and derived
//! statistics, they exist only to be looked at by heuristics, and they
//! therefore live in `jpxl-encode-policy`. The last four are *decisions* —
//! every one of their fields is something a decoder will read back — so they
//! live here, where the writer can be handed nothing else.
//!
//! ```text
//! jpxl-encode-policy  ──builds──▶  EmissionPlan
//!                                       │ validate()
//!                                       ▼
//! jpxl-encode                    ValidatedEmissionPlan ──▶ writer
//! ```
//!
//! Nothing in this module searches, scores or prefers. A plan says what the
//! bitstream will contain; deciding *which* plan is `jpxl-encode-policy`'s
//! entire job.
//!
//! # Milestone-1 scope
//!
//! [`SpatialPlan`], [`QuantizedFrameIr`], [`EntropyPlan`] and [`EmissionPlan`]
//! are here in the shape the fixed-DCT8x8 vertical slice (milestone 2) will
//! consume. Deliberately deferred, with the milestone that adds them:
//!
//! * spill-backed `PlaneStore`/`CoeffStore` (M10) — the stores below are
//!   resident, behind constructors that can grow a spill variant without
//!   touching their callers;
//! * `Modular` extra-channel and `GlobalModular` sub-streams in the VarDCT
//!   sections (M2+);
//! * candidate R-D curves, transform banks and search budgets — those are
//!   policy types and never appear in this crate.

use jpxl_core::geometry::LfBlockPos;
use jpxl_core::varblock::{TransformType, natural_coeff_order, order_id_dims};

use crate::vardct::error::{PlanError, PlanResult};
use crate::vardct::geometry::{BlockGrid, SectionKind, VardctGeometry};
use crate::vardct::ids::{
    CflFactor, ClusterId, GlobalScale, HfMul, LfGroupId, OrderId, PresetId, QuantLf,
};

/// Number of coefficient channels: X, Y, B.
pub const NUM_CHANNELS: usize = 3;

/// I.3.3's per-block-context distribution count: `37 + 458`.
pub const CONTEXTS_PER_BLOCK_CTX: u64 = 495;

/// I.2.2's default `block_ctx_map`, selected by its leading `u(1)`.
///
/// Three rows of 13 (channel x shape class). Transcribed from the clause, not
/// from the decoder: the write side of every table is derived independently so
/// that a shared transcription slip cannot cancel out in a round trip.
pub const DEFAULT_BLOCK_CTX_MAP: [u8; 39] = [
    0, 1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 6, 6, //
    7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14, //
    7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14,
];

/// `nb_block_ctx` of [`DEFAULT_BLOCK_CTX_MAP`]: the map's maximum plus one.
pub const DEFAULT_NB_BLOCK_CTX: u64 = 15;

/// I.2.2's ceiling on `nb_block_ctx`.
pub const MAX_NB_BLOCK_CTX: u64 = 16;

/// I.2.2's ceiling on `bsize`, the pre-clustering size of `block_ctx_map`.
pub const MAX_BLOCK_CTX_MAP_LEN: usize = 39 * 64;

/// C.2.2's ceiling on the number of entropy clusters.
pub const MAX_CLUSTERS: usize = 256;

// ---------------------------------------------------------------------------
// SpatialPlan
// ---------------------------------------------------------------------------

/// Frame-level decisions: everything that fixes the grids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameDecision {
    /// Frame width in samples.
    pub width: u32,
    /// Frame height in samples.
    pub height: u32,
    /// F.2's `group_size_shift`; `group_dim = 128 << shift`.
    pub group_size_shift: u32,
    /// F.6's `num_passes`.
    pub num_passes: u32,
}

impl FrameDecision {
    /// The grids this decision implies.
    ///
    /// # Errors
    ///
    /// [`PlanError::OutOfRange`] if the dimensions, shift or pass count are
    /// outside what the frame header can carry.
    pub fn geometry(&self) -> PlanResult<VardctGeometry> {
        VardctGeometry::new(
            self.width,
            self.height,
            self.group_size_shift,
            self.num_passes,
        )
    }
}

/// I.2's `Quantizer` bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizerDecision {
    /// The frame-wide scale.
    pub global_scale: GlobalScale,
    /// The LF-plane step.
    pub quant_lf: QuantLf,
}

/// I.2.3's `LfChannelCorrelation` bundle: the LF arm of chroma-from-luma.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LfCorrelationDecision {
    /// `colour_factor`, the divisor of both factors.
    pub colour_factor: u32,
    /// `base_correlation_x`.
    pub base_correlation_x: f32,
    /// `base_correlation_b`.
    pub base_correlation_b: f32,
    /// `x_factor_lf`, a `u(8)` biased by 128.
    pub x_factor_lf: u8,
    /// `b_factor_lf`, a `u(8)` biased by 128.
    pub b_factor_lf: u8,
}

impl Default for LfCorrelationDecision {
    /// I.2.3's `all_default` row.
    fn default() -> Self {
        Self {
            colour_factor: 84,
            base_correlation_x: 0.0,
            base_correlation_b: 1.0,
            x_factor_lf: 128,
            b_factor_lf: 128,
        }
    }
}

/// The LF-plane decisions (G.1.2, G.2.2, I.5.2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LfDecision {
    /// G.2.2's `extra_precision`, a `u(2)`.
    pub extra_precision: u8,
    /// G.1.2's `m_x_lf`, `m_y_lf`, `m_b_lf` dequantization multipliers.
    pub channel_dequant: [f32; NUM_CHANNELS],
    /// I.2.3's correlation bundle.
    pub correlation: LfCorrelationDecision,
    /// Whether I.5.2's adaptive LF smoothing runs.
    pub adaptive_smoothing: bool,
}

impl Default for LfDecision {
    /// The `all_default` rows of G.1.2 and I.2.3, smoothing on.
    fn default() -> Self {
        Self {
            extra_precision: 0,
            channel_dequant: [1.0 / 32.0, 1.0 / 4.0, 1.0 / 2.0],
            correlation: LfCorrelationDecision::default(),
            adaptive_smoothing: true,
        }
    }
}

/// J.1's `RestorationFilter` decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RestorationDecision {
    /// Whether the Gabor-like filter runs.
    pub gaborish: bool,
    /// Number of EPF iterations, `0..=3`.
    pub epf_iters: u8,
}

/// One varblock: a transform placed at an 8x8-block position, with its
/// quantization multiplier.
///
/// The `origin` is redundant with the sequence — G.2.4 derives it by replaying
/// the greedy walk — and that is exactly why it is stored: `validate` checks
/// the stored origin *against* the replay, so a planner that thinks it placed
/// a varblock somewhere the decoder will not put it is rejected rather than
/// silently re-placed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VarblockDecision {
    /// Top-left 8x8 block, relative to the LF group origin.
    pub origin: LfBlockPos,
    /// The `DctSelect` transform.
    pub transform: TransformType,
    /// `HfMul = 1 + mul`.
    pub hf_mul: HfMul,
}

/// The `XFromY`/`BFromY` planes of one LF group (G.2.4), one sample per
/// 64x64-sample tile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CflGrid {
    tiles: BlockGrid,
    x_from_y: Box<[CflFactor]>,
    b_from_y: Box<[CflFactor]>,
}

impl CflGrid {
    /// A grid of `tiles` shape with both factors zero.
    #[must_use]
    pub fn zeros(tiles: BlockGrid) -> Self {
        let n = usize::try_from(tiles.area()).unwrap_or(0);
        Self {
            tiles,
            x_from_y: vec![CflFactor::default(); n].into_boxed_slice(),
            b_from_y: vec![CflFactor::default(); n].into_boxed_slice(),
        }
    }

    /// Wraps two raster-order factor planes.
    ///
    /// # Errors
    ///
    /// [`PlanError::ShapeMismatch`] if either plane is not `tiles.area()`
    /// long.
    pub fn new(
        tiles: BlockGrid,
        x_from_y: Vec<CflFactor>,
        b_from_y: Vec<CflFactor>,
    ) -> PlanResult<Self> {
        for plane in [&x_from_y, &b_from_y] {
            if plane.len() as u64 != tiles.area() {
                return Err(PlanError::shape(
                    "CfL factor plane length",
                    "G.2.4",
                    tiles.area(),
                    plane.len() as u64,
                ));
            }
        }
        Ok(Self {
            tiles,
            x_from_y: x_from_y.into_boxed_slice(),
            b_from_y: b_from_y.into_boxed_slice(),
        })
    }

    /// The 64x64-tile grid this covers.
    #[must_use]
    pub const fn tiles(&self) -> BlockGrid {
        self.tiles
    }

    /// The `XFromY` plane in raster order.
    #[must_use]
    pub fn x_from_y(&self) -> &[CflFactor] {
        &self.x_from_y
    }

    /// The `BFromY` plane in raster order.
    #[must_use]
    pub fn b_from_y(&self) -> &[CflFactor] {
        &self.b_from_y
    }
}

/// The `Sharpness` plane of one LF group (G.2.4), one sample per 8x8 block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharpnessGrid {
    blocks: BlockGrid,
    values: Box<[u8]>,
}

impl SharpnessGrid {
    /// A grid of `blocks` shape, all zero.
    #[must_use]
    pub fn zeros(blocks: BlockGrid) -> Self {
        let n = usize::try_from(blocks.area()).unwrap_or(0);
        Self {
            blocks,
            values: vec![0u8; n].into_boxed_slice(),
        }
    }

    /// Wraps a raster-order sharpness plane.
    ///
    /// # Errors
    ///
    /// [`PlanError::ShapeMismatch`] if `values` is not `blocks.area()` long.
    pub fn new(blocks: BlockGrid, values: Vec<u8>) -> PlanResult<Self> {
        if values.len() as u64 != blocks.area() {
            return Err(PlanError::shape(
                "Sharpness plane length",
                "G.2.4",
                blocks.area(),
                values.len() as u64,
            ));
        }
        Ok(Self {
            blocks,
            values: values.into_boxed_slice(),
        })
    }

    /// The 8x8-block grid this covers.
    #[must_use]
    pub const fn blocks(&self) -> BlockGrid {
        self.blocks
    }

    /// The samples in raster order.
    #[must_use]
    pub fn values(&self) -> &[u8] {
        &self.values
    }
}

/// One LF group's spatial decisions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LfGroupPlan {
    /// Which LF group.
    pub id: LfGroupId,
    /// The varblocks, **in G.2.4 `BlockInfo` column order**, which is the
    /// order the greedy placement walk consumes them in.
    pub blocks: Box<[VarblockDecision]>,
    /// The HF chroma-from-luma factors.
    pub cfl: CflGrid,
    /// The `Sharpness` plane.
    pub sharpness: SharpnessGrid,
}

impl LfGroupPlan {
    /// Lowers the varblock list to G.2.4's two-row `BlockInfo` channel:
    /// row 0 is `DctSelect`, row 1 is `mul`.
    ///
    /// This is a pure relabelling — the sequence is already the wire
    /// sequence — which is the point of `Encoder-plan1.md` §4: the search's
    /// output *is* `BlockInfo`, with no partition-tree-to-raster conversion in
    /// between where an ordering bug could hide.
    #[must_use]
    pub fn block_info_rows(&self) -> [Vec<i32>; 2] {
        let dct_select = self
            .blocks
            .iter()
            .map(|b| i32::from(b.transform.dct_select()))
            .collect();
        let mul = self.blocks.iter().map(|b| b.hf_mul.stored_mul()).collect();
        [dct_select, mul]
    }

    /// G.2.4's `nb_blocks`: the number of `BlockInfo` columns.
    #[must_use]
    pub fn nb_blocks(&self) -> u64 {
        self.blocks.len() as u64
    }
}

/// The spatial stage: decisions, no symbols.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialPlan {
    /// Frame-level decisions.
    pub frame: FrameDecision,
    /// I.2's quantizer.
    pub quantizer: QuantizerDecision,
    /// The LF-plane decisions.
    pub lf: LfDecision,
    /// The restoration-filter decisions.
    pub restoration: RestorationDecision,
    /// One entry per LF group, in raster order.
    pub lf_groups: Box<[LfGroupPlan]>,
}

// ---------------------------------------------------------------------------
// QuantizedFrameIr
// ---------------------------------------------------------------------------

/// One LF group's quantized LF planes: one sample per 8x8 block, per channel.
///
/// `Encoder-plan1.md` §2.4 keeps the quantized and the reconstructed LF apart
/// because they feed different consumers (block context versus LLF
/// reconstruction). Milestone 1 carries only the quantized side — the integers
/// that reach the wire; the reconstruction cache is a policy-side artefact and
/// arrives with the rate loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LfQuantPlanes {
    blocks: BlockGrid,
    planes: [Box<[i32]>; NUM_CHANNELS],
}

impl LfQuantPlanes {
    /// All-zero planes of `blocks` shape.
    #[must_use]
    pub fn zeros(blocks: BlockGrid) -> Self {
        let n = usize::try_from(blocks.area()).unwrap_or(0);
        Self {
            blocks,
            planes: core::array::from_fn(|_| vec![0i32; n].into_boxed_slice()),
        }
    }

    /// Wraps three raster-order planes.
    ///
    /// # Errors
    ///
    /// [`PlanError::ShapeMismatch`] if a plane is not `blocks.area()` long.
    pub fn new(blocks: BlockGrid, planes: [Vec<i32>; NUM_CHANNELS]) -> PlanResult<Self> {
        for plane in &planes {
            if plane.len() as u64 != blocks.area() {
                return Err(PlanError::shape(
                    "LfQuant plane length",
                    "G.2.2",
                    blocks.area(),
                    plane.len() as u64,
                ));
            }
        }
        Ok(Self {
            blocks,
            planes: planes.map(Vec::into_boxed_slice),
        })
    }

    /// The 8x8-block grid the planes cover.
    #[must_use]
    pub const fn blocks(&self) -> BlockGrid {
        self.blocks
    }

    /// One channel's plane in raster order, `c` in `0..3` (X, Y, B).
    #[must_use]
    pub fn plane(&self, c: usize) -> Option<&[i32]> {
        self.planes.get(c).map(|p| &**p)
    }
}

/// One varblock's quantized HF coefficients, per channel, in the coefficient
/// array's own raster order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarblockCoefficients {
    channels: [Box<[i32]>; NUM_CHANNELS],
}

impl VarblockCoefficients {
    /// All-zero coefficients for `transform`.
    #[must_use]
    pub fn zeros(transform: TransformType) -> Self {
        let n = transform.num_blocks() * 64;
        Self {
            channels: core::array::from_fn(|_| vec![0i32; n].into_boxed_slice()),
        }
    }

    /// Wraps three coefficient arrays.
    ///
    /// # Errors
    ///
    /// [`PlanError::ShapeMismatch`] if a channel is not
    /// `64 * transform.num_blocks()` long — I.3.2's coefficient count.
    pub fn new(transform: TransformType, channels: [Vec<i32>; NUM_CHANNELS]) -> PlanResult<Self> {
        let expected = (transform.num_blocks() * 64) as u64;
        for channel in &channels {
            if channel.len() as u64 != expected {
                return Err(PlanError::shape(
                    "varblock coefficient count",
                    "I.3.2",
                    expected,
                    channel.len() as u64,
                ));
            }
        }
        Ok(Self {
            channels: channels.map(Vec::into_boxed_slice),
        })
    }

    /// One channel's coefficients, `c` in `0..3`.
    #[must_use]
    pub fn channel(&self, c: usize) -> Option<&[i32]> {
        self.channels.get(c).map(|p| &**p)
    }
}

/// One LF group's exact integers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizedLfGroup {
    /// Which LF group.
    pub id: LfGroupId,
    /// The quantized LF planes.
    pub lf: LfQuantPlanes,
    /// One entry per varblock, in the same order as
    /// [`LfGroupPlan::blocks`].
    pub coefficients: Box<[VarblockCoefficients]>,
}

/// The first IR that holds exactly what the decoder will read back.
///
/// Resident today; `Encoder-plan1.md` §2.4's spill-backed `CoeffStore` is the
/// milestone-10 substitution behind this type's accessors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizedFrameIr {
    /// One entry per LF group, in raster order.
    pub lf_groups: Box<[QuantizedLfGroup]>,
}

// ---------------------------------------------------------------------------
// EntropyPlan
// ---------------------------------------------------------------------------

/// I.2.2's HF block-context model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HfBlockContextPlan {
    /// The default map, signalled by I.2.2's leading `u(1)`.
    Default,
    /// A custom map with its thresholds.
    Custom {
        /// `lf_thresholds[0..3]`.
        lf_thresholds: [Vec<i32>; NUM_CHANNELS],
        /// `qf_thresholds`.
        qf_thresholds: Vec<u32>,
        /// `block_ctx_map`, of length
        /// `39 * (nb_qf_thr+1) * prod(nb_lf_thr[i]+1)`.
        map: Vec<u8>,
    },
}

impl HfBlockContextPlan {
    /// `nb_block_ctx`: one past the map's largest entry.
    #[must_use]
    pub fn nb_block_ctx(&self) -> u64 {
        match self {
            Self::Default => DEFAULT_NB_BLOCK_CTX,
            Self::Custom { map, .. } => map.iter().copied().max().map_or(0, |m| u64::from(m) + 1),
        }
    }

    /// The map as it reaches the wire.
    #[must_use]
    pub fn map(&self) -> &[u8] {
        match self {
            Self::Default => &DEFAULT_BLOCK_CTX_MAP,
            Self::Custom { map, .. } => map,
        }
    }
}

/// C.2.3's `HybridUintConfig`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HybridUintPlan {
    /// `split_exponent`.
    pub split_exponent: u8,
    /// `msb_in_token`.
    pub msb_in_token: u8,
    /// `lsb_in_token`.
    pub lsb_in_token: u8,
}

/// One entropy distribution, as symbol counts over its alphabet.
///
/// Milestone 1 stores the census result; turning counts into a normalized ANS
/// table and serializing it is slice 11.5's job, behind
/// [`crate::vardct::sink::SymbolSink`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistogramPlan {
    counts: Box<[u32]>,
}

impl HistogramPlan {
    /// Wraps symbol counts.
    ///
    /// # Errors
    ///
    /// [`PlanError::OutOfRange`] if the alphabet is empty or every count is
    /// zero — neither can be encoded, and both mean the census was not run.
    pub fn new(counts: Vec<u32>) -> PlanResult<Self> {
        if counts.is_empty() {
            return Err(PlanError::out_of_range("histogram alphabet size", "C.2", 0));
        }
        if counts.iter().all(|&c| c == 0) {
            return Err(PlanError::out_of_range("histogram total count", "C.2", 0));
        }
        Ok(Self {
            counts: counts.into_boxed_slice(),
        })
    }

    /// The counts.
    #[must_use]
    pub fn counts(&self) -> &[u32] {
        &self.counts
    }
}

/// A context map plus the per-cluster models it selects (C.2, C.2.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntropyModelPlan {
    /// One cluster per pre-context, in pre-context order.
    pub context_map: Box<[ClusterId]>,
    /// One distribution per cluster.
    pub histograms: Box<[HistogramPlan]>,
    /// One `HybridUintConfig` per cluster.
    pub hybrid_uint: Box<[HybridUintPlan]>,
}

/// One explicit coefficient order: `table[k]` is the coefficient cell that
/// receives order position `k` (I.3.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoefficientOrder {
    /// Which Table I.7 Order ID.
    pub order_id: OrderId,
    /// Which channel, `0..3`.
    pub channel: u8,
    /// The order table.
    pub table: Box<[u32]>,
}

/// I.3.1's coefficient orders for one pass.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OrderSet {
    permutations: Vec<CoefficientOrder>,
}

impl OrderSet {
    /// The natural order for every Order ID and channel — I.3.1's
    /// `used_orders == 0`.
    #[must_use]
    pub fn natural() -> Self {
        Self::default()
    }

    /// Overrides one `(order_id, channel)` with an explicit order table.
    ///
    /// `order[k]` is the coefficient cell that receives order position `k`
    /// (I.3.1), so `table` must be a permutation of the Order ID's natural
    /// order — the *same multiset of cells*, differently sequenced.
    ///
    /// # Errors
    ///
    /// [`PlanError::OutOfRange`] for an Order ID outside `0..13` or a channel
    /// outside `0..3`, and [`PlanError::ShapeMismatch`] if `table` is not a
    /// permutation of that Order ID's natural order.
    pub fn with_order(
        mut self,
        order_id: OrderId,
        channel: u8,
        table: Vec<u32>,
    ) -> PlanResult<Self> {
        let dims = order_id_dims(usize::from(order_id.get()))
            .ok_or_else(|| PlanError::out_of_range("Order ID", "I.7", i64::from(order_id.get())))?;
        if usize::from(channel) >= NUM_CHANNELS {
            return Err(PlanError::out_of_range(
                "coefficient order channel",
                "I.3.1",
                i64::from(channel),
            ));
        }
        let natural = natural_coeff_order(dims.0, dims.1);
        if table.len() != natural.len() {
            return Err(PlanError::shape(
                "coefficient order length",
                "I.3.1",
                natural.len() as u64,
                table.len() as u64,
            ));
        }
        let mut sorted = table.clone();
        let mut expected = natural;
        sorted.sort_unstable();
        expected.sort_unstable();
        if sorted != expected {
            return Err(PlanError::shape(
                "coefficient order is not a permutation of the natural order",
                "I.3.1",
                sorted.len() as u64,
                sorted.len() as u64,
            ));
        }
        self.permutations.push(CoefficientOrder {
            order_id,
            channel,
            table: table.into_boxed_slice(),
        });
        Ok(self)
    }

    /// I.3.1's `used_orders` bitmask over the 13 Order IDs.
    #[must_use]
    pub fn used_orders(&self) -> u32 {
        self.permutations.iter().fold(0u32, |mask, order| {
            mask | (1u32 << u32::from(order.order_id.get()))
        })
    }

    /// The explicit orders, in insertion order.
    #[must_use]
    pub fn overrides(&self) -> &[CoefficientOrder] {
        &self.permutations
    }
}

/// One pass's entropy decisions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfPassEntropyPlan {
    /// I.3.1's coefficient orders.
    pub orders: OrderSet,
    /// I.3.3's histogram bundle: one context map over
    /// `495 * num_hf_presets * nb_block_ctx` pre-contexts.
    pub distributions: EntropyModelPlan,
    /// I.4's `hfp` for each pass group, in raster order.
    pub group_presets: Box<[PresetId]>,
}

/// The entropy stage.
///
/// **Deviation from `Encoder-plan1.md` §2.5,** which puts a `context_map` on
/// each `HfPresetPlan`. I.3.3 sizes *one* pre-clustered distribution list per
/// pass at `495 * num_hf_presets * nb_block_ctx`, and I.4 selects within it by
/// adding `495 * nb_block_ctx * hfp` to every context — so the preset is an
/// offset into a single per-pass model, not a model of its own. Modelling it
/// the advisor's way would make an unrepresentable plan expressible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntropyPlan {
    /// I.2.2's block-context model.
    pub block_context: HfBlockContextPlan,
    /// I.2.6's `num_hf_presets`.
    pub num_hf_presets: u32,
    /// One entry per pass, in pass order.
    pub passes: Box<[HfPassEntropyPlan]>,
}

impl EntropyPlan {
    /// The number of pre-clustering contexts one pass's context map must
    /// cover: `495 * num_hf_presets * nb_block_ctx` (I.3.3).
    #[must_use]
    pub fn pre_contexts(&self) -> u64 {
        CONTEXTS_PER_BLOCK_CTX
            .saturating_mul(u64::from(self.num_hf_presets))
            .saturating_mul(self.block_context.nb_block_ctx())
    }
}

// ---------------------------------------------------------------------------
// EmissionPlan
// ---------------------------------------------------------------------------

/// The TOC's section list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SectionLayout {
    /// The sections, in the order F.3.1 puts them on the wire.
    pub kinds: Box<[SectionKind]>,
}

impl SectionLayout {
    /// The layout `geometry` requires.
    #[must_use]
    pub fn for_geometry(geometry: &VardctGeometry) -> Self {
        Self {
            kinds: geometry.section_layout().into_boxed_slice(),
        }
    }
}

/// The last IR before bits: everything the writer needs, nothing it does not.
///
/// **Deviation from `Encoder-plan1.md` §2.5,** which lists
/// `image_header`/`frame_header` syntax types here. Those types belong to the
/// VarDCT header writer, which is milestone 2; until it exists an
/// `EmissionPlan` carrying half-invented header structs would be validated
/// against nothing. What it carries instead is the lowered plan itself — the
/// three stages the writer reads — plus the section layout, and the header
/// fields it needs are already in [`FrameDecision`]/[`SpatialPlan`].
#[derive(Debug, Clone, PartialEq)]
pub struct EmissionPlan {
    /// The spatial decisions.
    pub spatial: SpatialPlan,
    /// The exact integers.
    pub quantized: QuantizedFrameIr,
    /// The entropy models.
    pub entropy: EntropyPlan,
    /// The TOC layout.
    pub sections: SectionLayout,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_block_context_map_has_the_shape_i22_states() {
        assert_eq!(DEFAULT_BLOCK_CTX_MAP.len(), 39);
        let max = DEFAULT_BLOCK_CTX_MAP
            .iter()
            .copied()
            .max()
            .expect("nonempty");
        assert_eq!(u64::from(max) + 1, DEFAULT_NB_BLOCK_CTX);
        const { assert!(DEFAULT_NB_BLOCK_CTX <= MAX_NB_BLOCK_CTX) };
        // Rows 2 and 3 (X and B) share their values; row 1 (Y) does not.
        assert_eq!(
            DEFAULT_BLOCK_CTX_MAP.get(13..26),
            DEFAULT_BLOCK_CTX_MAP.get(26..39)
        );
        assert_eq!(HfBlockContextPlan::Default.nb_block_ctx(), 15);
    }

    #[test]
    fn block_info_rows_are_the_varblock_sequence_relabelled() {
        let plan = LfGroupPlan {
            id: LfGroupId::new(0),
            blocks: vec![
                VarblockDecision {
                    origin: LfBlockPos::new(0, 0),
                    transform: TransformType::Dct16x16,
                    hf_mul: HfMul::new(3).expect("legal"),
                },
                VarblockDecision {
                    origin: LfBlockPos::new(2, 0),
                    transform: TransformType::Dct8x8,
                    hf_mul: HfMul::new(1).expect("legal"),
                },
            ]
            .into_boxed_slice(),
            cfl: CflGrid::zeros(BlockGrid {
                width: 1,
                height: 1,
            }),
            sharpness: SharpnessGrid::zeros(BlockGrid {
                width: 4,
                height: 2,
            }),
        };
        let [dct_select, mul] = plan.block_info_rows();
        assert_eq!(dct_select, vec![4, 0]);
        assert_eq!(mul, vec![2, 0]);
        assert_eq!(plan.nb_blocks(), 2);
    }

    #[test]
    fn an_order_override_must_be_a_permutation_of_the_natural_order() {
        // Order ID 0 is the 8x8 shape class: 64 cells.
        let natural = natural_coeff_order(8, 8);
        let mut swapped = natural.clone();
        swapped.swap(3, 40);
        let set = OrderSet::natural()
            .with_order(OrderId::new(0), 1, swapped)
            .expect("a permutation of the natural order");
        assert_eq!(set.used_orders(), 1);
        assert_eq!(set.overrides().len(), 1);

        let mut broken = natural;
        if let Some(first) = broken.first_mut() {
            *first = 4096;
        }
        assert!(matches!(
            OrderSet::natural().with_order(OrderId::new(0), 1, broken),
            Err(PlanError::ShapeMismatch { .. })
        ));
        assert!(matches!(
            OrderSet::natural().with_order(OrderId::new(13), 1, Vec::new()),
            Err(PlanError::OutOfRange {
                what: "Order ID",
                ..
            })
        ));
    }

    #[test]
    fn a_histogram_needs_a_nonempty_alphabet_and_a_positive_total() {
        assert!(HistogramPlan::new(vec![1, 0, 3]).is_ok());
        assert!(HistogramPlan::new(Vec::new()).is_err());
        assert!(HistogramPlan::new(vec![0, 0]).is_err());
    }

    #[test]
    fn the_pre_context_count_is_i33s_product() {
        let plan = EntropyPlan {
            block_context: HfBlockContextPlan::Default,
            num_hf_presets: 2,
            passes: Box::new([]),
        };
        assert_eq!(plan.pre_contexts(), 495 * 2 * 15);
    }

    #[test]
    fn coefficient_arrays_are_sized_by_the_transforms_block_count() {
        assert!(
            VarblockCoefficients::new(
                TransformType::Dct16x16,
                core::array::from_fn(|_| vec![0i32; 4 * 64])
            )
            .is_ok()
        );
        assert!(matches!(
            VarblockCoefficients::new(
                TransformType::Dct16x16,
                core::array::from_fn(|_| vec![0i32; 64])
            ),
            Err(PlanError::ShapeMismatch { .. })
        ));
        // The bound `OrderSet::with_order` checks an Order ID against.
        assert_eq!(jpxl_core::varblock::NUM_ORDER_IDS, 13);
    }
}
