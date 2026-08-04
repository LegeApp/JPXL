//! The section writers: a validated plan becomes a kVarDCT codestream
//! (18181-1 Annexes F, G, I).
//!
//! ```text
//! ImageHeaders  D.1 D.2 D.3        headers::write_image_headers
//! FrameHeader   F.2                headers::write_frame_header
//! TOC           F.3                section::SectionStore
//!   LfGlobal    G.1   quantizer, block context, LF correlation, GlobalModular
//!   LfGroup[n]  G.2   LfQuant, ModularLfGroup, HfMetadata
//!   HfGlobal    G.3   dequant matrices, num_hf_presets, HfPass[num_passes]
//!   PassGroup   G.4   hfp, the HF coefficient stream, modular group data
//! ```
//!
//! # The single-section shape is not "the same thing, padded"
//!
//! F.3.1 gives a frame **one** TOC entry when `num_groups == 1` and
//! `num_passes == 1`, and then every structure above is carried consecutively
//! in that one bit stream — with no `ZeroPadToByte()` between them. Every other
//! frame gets one section per structure, each byte-aligned because the TOC
//! measures it in bytes. So the two shapes are not the same bits with different
//! framing: a writer that padded between structures in the single-section case
//! would desynchronise `LfGroup` from the first bit. [`write_frame`] therefore
//! composes the structures into one [`BitWriter`] in that case and into
//! separate buffers otherwise.
//!
//! # What milestone 2 emits, and what it refuses
//!
//! [`check_supported`] is the list. Everything it rejects is rejected because
//! the *writer* for it does not exist yet, not because the plan IR cannot say
//! it — `validate` already accepted the plan, and refusing here rather than
//! emitting an approximation is the difference between "not implemented" and
//! "silently wrong".
//!
//! # Entropy: one bundle per pass, one stream per pass group
//!
//! I.3.3 puts the HF histograms in `HfGlobal`, once per pass, while C.3.2's ANS
//! state restarts in every `PassGroup`. That maps exactly onto `jpxl-entropy`'s
//! split: [`EntropyTables::write_bundle`] goes in `HfGlobal` and one
//! [`SymbolEncoder`] per pass group writes its own seeded, terminal-state-exact
//! stream. The tables are built from a census taken over **every** group of the
//! pass, because a per-group histogram would have nowhere to be signalled.

use jpxl_bitstream::{BitWriter, U32Dist, U32Spec};
use jpxl_core::varblock::TransformType;
use jpxl_entropy::HybridUintConfig;
use jpxl_entropy::encode::{
    CodingMode, ContextMap, EncoderPlan, EntropyTables, SymbolEncoder, TokenCensus,
};

use crate::error::{EncodeError, Result};
use crate::section::SectionStore;
use crate::vardct::geometry::VardctGeometry;
use crate::vardct::headers::{
    NEUTRAL_QM_SCALE, VARDCT_GROUP_SIZE_SHIFT, write_frame_header, write_image_headers,
};
use crate::vardct::ids::{ClusterId, LfGroupId, PreContextId};
use crate::vardct::modular_out::{OutChannel, write_modular_stream};
use crate::vardct::plan::{
    EmissionPlan, HfBlockContextPlan, LfCorrelationDecision, LfDecision, NUM_CHANNELS,
};
use crate::vardct::sink::{CensusSink, HfEventSink};
use crate::vardct::size::{CodestreamSizing, Emission, SectionSize};
use crate::vardct::validate::ValidatedEmissionPlan;
use crate::vardct::walk::{
    OrderTables, PassGroupWalk, WalkVarblock, pre_context_count, walk_pass_group,
};

/// 18181-1 I.2.1: `U32(1 + u(11), 2049 + u(11), 4097 + u(12), 8193 + u(16))`.
const GLOBAL_SCALE_SPEC: U32Spec = U32Spec::new([
    U32Dist::BitsOffset {
        bits: 11,
        offset: 1,
    },
    U32Dist::BitsOffset {
        bits: 11,
        offset: 2049,
    },
    U32Dist::BitsOffset {
        bits: 12,
        offset: 4097,
    },
    U32Dist::BitsOffset {
        bits: 16,
        offset: 8193,
    },
]);

/// 18181-1 I.2.1: `U32(16, 1 + u(5), 1 + u(8), 1 + u(16))`.
const QUANT_LF_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(16),
    U32Dist::BitsOffset { bits: 5, offset: 1 },
    U32Dist::BitsOffset { bits: 8, offset: 1 },
    U32Dist::BitsOffset {
        bits: 16,
        offset: 1,
    },
]);

/// 18181-1 I.2.3: `U32(84, 256, 2 + u(8), 258 + u(16))`.
const COLOUR_FACTOR_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(84),
    U32Dist::Val(256),
    U32Dist::BitsOffset { bits: 8, offset: 2 },
    U32Dist::BitsOffset {
        bits: 16,
        offset: 258,
    },
]);

/// 18181-1 I.3.1: `U32(0x5F, 0x13, 0x00, u(13))` for `used_orders`.
///
/// The third distribution is the constant zero, which is how the clause says
/// "natural order everywhere" in two bits.
const USED_ORDERS_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(0x5F),
    U32Dist::Val(0x13),
    U32Dist::Val(0x00),
    U32Dist::bits(13),
]);

/// `ceil(log2(n))` for `n >= 1`; zero for `n <= 1`.
const fn ceil_log2(n: u64) -> u32 {
    if n <= 1 {
        return 0;
    }
    (n - 1).ilog2() + 1
}

/// Refuses every plan shape milestone 2's writer cannot emit.
///
/// # Errors
///
/// [`EncodeError::Unsupported`], naming the clause whose writer is missing.
pub fn check_supported(plan: &EmissionPlan) -> Result<()> {
    let spatial = &plan.spatial;
    if spatial.frame.group_size_shift != VARDCT_GROUP_SIZE_SHIFT {
        return Err(EncodeError::unsupported(
            "a kVarDCT frame with a group_size_shift other than 1 (F.2 does not \
             signal the field outside kModular, so group_dim is always 256)",
            "F.2",
        ));
    }
    if spatial.frame.num_passes != 1 {
        return Err(EncodeError::unsupported("more than one pass", "F.6"));
    }
    let neutral_lf = LfDecision::vardct_neutral();
    if spatial.lf.extra_precision != neutral_lf.extra_precision
        || spatial.lf.channel_dequant != neutral_lf.channel_dequant
        || spatial.lf.adaptive_smoothing != neutral_lf.adaptive_smoothing
    {
        return Err(EncodeError::unsupported(
            "non-default LF dequantization or adaptive smoothing",
            "G.1.2",
        ));
    }
    let corr = spatial.lf.correlation;
    if corr.colour_factor != 84 || corr.base_correlation_x != 0.0 || corr.base_correlation_b != 1.0
    {
        return Err(EncodeError::unsupported(
            "a non-default CfL divisor or base correlation",
            "I.2.3",
        ));
    }
    if spatial.restoration.gaborish || spatial.restoration.epf_iters != 0 {
        return Err(EncodeError::unsupported(
            "the restoration filters (they would move every reconstructed \
             sample away from what the encoder quantized against)",
            "J.1",
        ));
    }
    if plan.entropy.block_context != HfBlockContextPlan::Default {
        return Err(EncodeError::unsupported(
            "a custom HF block context map",
            "I.2.2",
        ));
    }
    if plan.entropy.num_hf_presets != 1 {
        return Err(EncodeError::unsupported("more than one HF preset", "I.2.6"));
    }
    for pass in &plan.entropy.passes {
        if !pass.orders.overrides().is_empty() {
            return Err(EncodeError::unsupported(
                "custom coefficient orders (their F.3.2 permutation stream is \
                 not written yet)",
                "I.3.1",
            ));
        }
    }
    for group in &plan.spatial.lf_groups {
        // The square vocabulary is what has decoder-parity evidence today; the
        // walk and the writer are transform-generic, so widening this list is
        // an evidence question, not a code change elsewhere.
        if group.blocks.iter().any(|b| {
            !matches!(
                b.transform,
                TransformType::Dct8x8 | TransformType::Dct16x16 | TransformType::Dct32x32
            )
        }) {
            return Err(EncodeError::unsupported(
                "a transform outside the square DCT vocabulary (DCT8x8, DCT16x16, DCT32x32)",
                "I.1",
            ));
        }
        if group
            .cfl
            .x_from_y()
            .iter()
            .chain(group.cfl.b_from_y())
            .any(|factor| !(-128..=127).contains(&factor.get()))
        {
            return Err(EncodeError::unsupported(
                "an HF chroma-from-luma factor outside the interoperable signed-byte range",
                "G.2.4",
            ));
        }
    }
    Ok(())
}

/// Encodes a validated plan as a naked kVarDCT codestream.
///
/// # Errors
///
/// As [`emit_codestream`].
pub fn write_codestream(plan: &ValidatedEmissionPlan) -> Result<Vec<u8>> {
    Ok(emit_codestream(plan)?.bytes)
}

/// The exact size of the codestream `plan` emits, without keeping the bytes.
///
/// This is [`emit_codestream`] with the buffer dropped, and that is the whole
/// design: see [`size`](crate::vardct::size) for why a rate loop must not have
/// a second implementation of "how big would this be".
///
/// # Errors
///
/// As [`emit_codestream`].
pub fn price_codestream(plan: &ValidatedEmissionPlan) -> Result<CodestreamSizing> {
    Ok(emit_codestream(plan)?.sizing)
}

/// Encodes a validated plan, reporting where every byte went.
///
/// # Errors
///
/// [`EncodeError::Unsupported`] for a plan shape [`check_supported`] refuses,
/// [`EncodeError::Entropy`] if the entropy layer rejects a symbol, and
/// [`EncodeError::ValueOutOfRange`] for a field that cannot be expressed.
pub fn emit_codestream(plan: &ValidatedEmissionPlan) -> Result<Emission> {
    let inner = plan.plan();
    check_supported(inner)?;
    let geometry = inner.spatial.frame.geometry().map_err(EncodeError::Plan)?;

    let mut w = BitWriter::new();
    write_image_headers(&mut w, geometry.width(), geometry.height())?;
    // F.1: every frame starts on a byte boundary, so this division is exact.
    w.zero_pad_to_byte();
    let image_headers = w.bit_len() / 8;

    write_frame_header(&mut w, NEUTRAL_QM_SCALE, NEUTRAL_QM_SCALE)?;
    let frame_header_bits = w.bit_len() - image_headers * 8;

    let store = write_frame_body(inner, &geometry)?;
    let lengths = store.lengths();
    let before_toc = w.bit_len();
    store.write(&mut w)?;
    // `write_toc` ends byte-aligned and every body is appended whole, so the
    // table's own width is what is left after the bodies are subtracted. There
    // is no second measurement of the bodies: `lengths` is the very array the
    // TOC entries were written from.
    let body_bits: u64 = lengths
        .iter()
        .map(|&n| u64::try_from(n).unwrap_or(u64::MAX).saturating_mul(8))
        .sum();
    let toc_bits = w
        .bit_len()
        .saturating_sub(before_toc)
        .saturating_sub(body_bits);

    if inner.sections.kinds.len() != lengths.len() {
        return Err(EncodeError::unsupported(
            "a section layout that does not match the sections written",
            "F.3.1",
        ));
    }
    let sections: Box<[SectionSize]> = inner
        .sections
        .kinds
        .iter()
        .zip(&lengths)
        .map(|(&kind, &bytes)| SectionSize {
            kind,
            bytes: u64::try_from(bytes).unwrap_or(u64::MAX),
        })
        .collect();

    let bytes = w.into_bytes();
    let sizing = CodestreamSizing {
        total: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
        image_headers,
        frame_header_bits,
        toc_bits,
        sections,
    };
    Ok(Emission { bytes, sizing })
}

/// Builds every section of the frame, in F.3.1 order.
fn write_frame_body(plan: &EmissionPlan, geometry: &VardctGeometry) -> Result<SectionStore> {
    let orders = OrderTables::from_order_set(
        &plan
            .entropy
            .passes
            .first()
            .ok_or_else(|| EncodeError::unsupported("a frame with no pass", "F.6"))?
            .orders,
    );
    let tables = build_entropy_tables(plan, geometry, &orders)?;

    let mut store = SectionStore::new();
    if geometry.is_single_section() {
        let mut body = BitWriter::new();
        write_lf_global(plan, &mut body)?;
        write_lf_group(plan, geometry, LfGroupId::new(0), &mut body)?;
        write_hf_global(plan, geometry, &tables, &mut body)?;
        write_pass_group(plan, geometry, &orders, &tables, 0, &mut body)?;
        body.zero_pad_to_byte();
        store.push(body.into_bytes());
        return Ok(store);
    }

    store.push(section_bytes(|w| write_lf_global(plan, w))?);
    for index in 0..geometry.num_lf_groups() {
        let id = LfGroupId::new(u32::try_from(index).unwrap_or(u32::MAX));
        store.push(section_bytes(|w| write_lf_group(plan, geometry, id, w))?);
    }
    store.push(section_bytes(|w| {
        write_hf_global(plan, geometry, &tables, w)
    })?);
    for group in 0..geometry.num_groups() {
        store.push(section_bytes(|w| {
            write_pass_group(plan, geometry, &orders, &tables, group, w)
        })?);
    }
    Ok(store)
}

/// Runs `body` into a fresh byte-aligned section buffer.
fn section_bytes(body: impl FnOnce(&mut BitWriter) -> Result<()>) -> Result<Vec<u8>> {
    let mut w = BitWriter::new();
    body(&mut w)?;
    w.zero_pad_to_byte();
    Ok(w.into_bytes())
}

// ---------------------------------------------------------------------------
// G.1 — LfGlobal
// ---------------------------------------------------------------------------

fn write_lf_global(plan: &EmissionPlan, w: &mut BitWriter) -> Result<()> {
    // Table G.1's flag-guarded rows (patches, splines, noise) are all absent:
    // the frame header wrote `flags` with none of them set.

    // G.1.2 LfChannelDequantization: the Table G.2 defaults, which is what
    // `check_supported` pinned `LfDecision` to.
    w.write_bool(true);

    // I.2.1 Quantizer.
    w.write_u32(
        &GLOBAL_SCALE_SPEC,
        plan.spatial.quantizer.global_scale.get(),
    )?;
    w.write_u32(&QUANT_LF_SPEC, plan.spatial.quantizer.quant_lf.get())?;

    // I.2.2 HF block context: the leading u(1) selects the default map.
    w.write_bits(1, 1)?;

    // I.2.3 LfChannelCorrelation. Slice 15 searches the two biased u8 factors
    // while keeping the divisor and base correlations at their defaults. The
    // latter therefore have exact binary16 spellings: +0 is 0x0000 and +1 is
    // 0x3c00. `check_supported` rejects any other base before bits are written.
    let corr = plan.spatial.lf.correlation;
    if corr == LfCorrelationDecision::default() {
        w.write_bool(true);
    } else {
        w.write_bool(false);
        w.write_u32(&COLOUR_FACTOR_SPEC, corr.colour_factor)?;
        w.write_bits(16, 0x0000)?;
        w.write_bits(16, 0x3c00)?;
        w.write_bits(8, u32::from(corr.x_factor_lf))?;
        w.write_bits(8, u32::from(corr.b_factor_lf))?;
    }

    // G.1.3 GlobalModular: the leading Bool() is read whatever the channel
    // count, and then a sub-bitstream over `num_extra == 0` channels, which
    // H.1 says is not read at all.
    w.write_bool(false);
    write_modular_stream(w, &[])?;
    Ok(())
}

// ---------------------------------------------------------------------------
// G.2 — LfGroup
// ---------------------------------------------------------------------------

fn write_lf_group(
    plan: &EmissionPlan,
    geometry: &VardctGeometry,
    id: LfGroupId,
    w: &mut BitWriter,
) -> Result<()> {
    let index = usize::try_from(id.index()).unwrap_or(usize::MAX);
    let spatial = plan
        .spatial
        .lf_groups
        .get(index)
        .ok_or_else(|| EncodeError::unsupported("an LF group past the grid", "G.2"))?;
    let quantized = plan
        .quantized
        .lf_groups
        .get(index)
        .ok_or_else(|| EncodeError::unsupported("an LF group past the grid", "G.2"))?;
    let grid = geometry
        .lf_group_blocks(id)
        .ok_or_else(|| EncodeError::unsupported("an LF group past the grid", "G.2"))?;

    // --- G.2.2 LF coefficients ---
    w.write_bits(2, u32::from(plan.spatial.lf.extra_precision))?;
    // The three LfQuant channels go on the wire as Y, X, B — I.4 states that
    // order for HF coefficients and G.2.2's "three channels" was found to obey
    // it too; `jpxl-decode`'s `LF_QUANT_CHANNEL_ORDER_IS_XYB` records the
    // fixture evidence. A `[X, Y, B]` stream decodes to swapped chroma, which
    // the oracle tests below would catch immediately.
    let planes: [&[i32]; NUM_CHANNELS] = [
        quantized.lf.plane(1).unwrap_or(&[]),
        quantized.lf.plane(0).unwrap_or(&[]),
        quantized.lf.plane(2).unwrap_or(&[]),
    ];
    let lf_channels: Vec<OutChannel<'_>> = planes
        .into_iter()
        .map(|samples| OutChannel {
            width: grid.width,
            height: grid.height,
            samples,
        })
        .collect();
    write_modular_stream(w, &lf_channels)?;

    // --- G.2.3 ModularLfGroup: no extra channels, so nothing at all ---
    write_modular_stream(w, &[])?;

    // --- G.2.4 HF metadata ---
    let total_blocks = grid.area();
    let nb_blocks = spatial.nb_blocks();
    let extra =
        u32::try_from(nb_blocks.saturating_sub(1)).map_err(|_| EncodeError::ValueOutOfRange {
            what: "nb_blocks",
            value: i64::try_from(nb_blocks).unwrap_or(i64::MAX),
        })?;
    w.write_bits(ceil_log2(total_blocks.max(1)), extra)?;

    let tiles = geometry
        .lf_group_cfl_tiles(id)
        .ok_or_else(|| EncodeError::unsupported("an LF group past the grid", "G.2"))?;
    let x_from_y: Vec<i32> = spatial.cfl.x_from_y().iter().map(|f| f.get()).collect();
    let b_from_y: Vec<i32> = spatial.cfl.b_from_y().iter().map(|f| f.get()).collect();
    // BlockInfo is two rows by nb_blocks columns: DctSelect then mul.
    let [dct_select, mul] = spatial.block_info_rows();
    let block_info: Vec<i32> = dct_select.into_iter().chain(mul).collect();
    let sharpness: Vec<i32> = spatial
        .sharpness
        .values()
        .iter()
        .map(|&v| i32::from(v))
        .collect();
    let nb_blocks_u32 = u32::try_from(nb_blocks).unwrap_or(u32::MAX);

    write_modular_stream(
        w,
        &[
            OutChannel {
                width: tiles.width,
                height: tiles.height,
                samples: &x_from_y,
            },
            OutChannel {
                width: tiles.width,
                height: tiles.height,
                samples: &b_from_y,
            },
            OutChannel {
                width: nb_blocks_u32,
                height: 2,
                samples: &block_info,
            },
            OutChannel {
                width: grid.width,
                height: grid.height,
                samples: &sharpness,
            },
        ],
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// G.3 — HfGlobal
// ---------------------------------------------------------------------------

fn write_hf_global(
    plan: &EmissionPlan,
    geometry: &VardctGeometry,
    tables: &EntropyTables,
    w: &mut BitWriter,
) -> Result<()> {
    // I.2.4: one Bool() selects the whole I.2.5 default table.
    w.write_bool(true);

    // I.2.6: num_hf_presets = u(ceil(log2(num_groups))) + 1.
    let bits = ceil_log2(geometry.num_groups().max(1));
    w.write_bits(bits, plan.entropy.num_hf_presets.saturating_sub(1))?;

    // I.3 HfPass, once per pass.
    for _ in 0..plan.spatial.frame.num_passes {
        // I.3.1: used_orders == 0 means the natural order of I.3.2 everywhere,
        // and skips the permutation stream entirely.
        w.write_u32(&USED_ORDERS_SPEC, 0)?;
        // I.3.3: the pre-clustered distributions, read here and used by every
        // pass group of this pass.
        tables.write_bundle(w)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// G.4 — PassGroup
// ---------------------------------------------------------------------------

fn write_pass_group(
    plan: &EmissionPlan,
    geometry: &VardctGeometry,
    orders: &OrderTables,
    tables: &EntropyTables,
    group: u64,
    w: &mut BitWriter,
) -> Result<()> {
    // I.4: hfp is `u(ceil(log2(num_hf_presets)))`, i.e. zero bits at one preset.
    let bits = ceil_log2(u64::from(plan.entropy.num_hf_presets.max(1)));
    w.write_bits(bits, 0)?;

    let (walk, varblocks) = pass_group_walk(plan, geometry, orders, group)?;
    let mut sink = SymbolSinkAdapter {
        encoder: SymbolEncoder::new(tables),
        error: None,
    };
    walk_pass_group(&walk, &varblocks, &mut sink).map_err(EncodeError::Plan)?;
    if let Some(error) = sink.error {
        return Err(EncodeError::from(error));
    }
    sink.encoder.write_stream(w)?;

    // G.4.2 modular group data: no remaining channels, so nothing.
    write_modular_stream(w, &[])?;
    Ok(())
}

/// Feeds I.4's events into `jpxl-entropy`'s replay encoder.
///
/// [`HfEventSink`] is infallible by design — a census cannot fail — so the
/// first entropy error is parked here and raised by the caller. Dropping it
/// silently would produce a short stream that no decoder could explain.
struct SymbolSinkAdapter<'a> {
    encoder: SymbolEncoder<'a>,
    error: Option<jpxl_entropy::EntropyError>,
}

impl SymbolSinkAdapter<'_> {
    fn push(&mut self, context: PreContextId, value: u32) {
        if self.error.is_some() {
            return;
        }
        let ctx = usize::try_from(context.get()).unwrap_or(usize::MAX);
        if let Err(e) = self.encoder.push_uint(ctx, value) {
            self.error = Some(e);
        }
    }
}

impl HfEventSink for SymbolSinkAdapter<'_> {
    fn nonzeros(&mut self, context: PreContextId, value: u32) {
        self.push(context, value);
    }

    fn coefficient(&mut self, context: PreContextId, value: u32) {
        self.push(context, value);
    }
}

// ---------------------------------------------------------------------------
// The walk's inputs
// ---------------------------------------------------------------------------

/// Builds one pass group's walk inputs from the plan.
///
/// The varblocks come out of their LF group's `BlockInfo` sequence, filtered to
/// this pass group and rebased on it: I.4's `NonZeros` grid is group-scoped, so
/// a varblock's coordinates in the walk are group-relative even though the plan
/// stores them LF-group-relative. Validation has already proved no varblock
/// straddles a pass-group edge, which is what makes the filter a partition.
fn pass_group_walk<'a>(
    plan: &'a EmissionPlan,
    geometry: &VardctGeometry,
    orders: &'a OrderTables,
    group: u64,
) -> Result<(PassGroupWalk<'a>, Vec<WalkVarblock<'a>>)> {
    let rect = geometry
        .group_rect(group)
        .ok_or_else(|| EncodeError::unsupported("a group index past the grid", "G.4"))?;
    let id = geometry
        .lf_group_of(group)
        .ok_or_else(|| EncodeError::unsupported("a group outside every LF group", "G.2"))?;
    let lf_rect = geometry
        .lf_group_rect(id)
        .ok_or_else(|| EncodeError::unsupported("an LF group past the grid", "G.2"))?;
    let lf_grid = geometry
        .lf_group_blocks(id)
        .ok_or_else(|| EncodeError::unsupported("an LF group past the grid", "G.2"))?;
    let index = usize::try_from(id.index()).unwrap_or(usize::MAX);
    let spatial = plan
        .spatial
        .lf_groups
        .get(index)
        .ok_or_else(|| EncodeError::unsupported("an LF group past the grid", "G.2"))?;
    let quantized = plan
        .quantized
        .lf_groups
        .get(index)
        .ok_or_else(|| EncodeError::unsupported("an LF group past the grid", "G.2"))?;

    let origin_bx = (rect.x0 - lf_rect.x0) / 8;
    let origin_by = (rect.y0 - lf_rect.y0) / 8;
    let blocks_w = rect.width.div_ceil(8);
    let blocks_h = rect.height.div_ceil(8);

    let mut varblocks = Vec::new();
    for (i, block) in spatial.blocks.iter().enumerate() {
        let (bx, by) = (block.origin.bx(), block.origin.by());
        if bx < origin_bx || by < origin_by {
            continue;
        }
        let (lx, ly) = (bx - origin_bx, by - origin_by);
        if lx >= blocks_w || ly >= blocks_h {
            continue;
        }
        let cell = usize::try_from(u64::from(by) * u64::from(lf_grid.width) + u64::from(bx))
            .unwrap_or(usize::MAX);
        let qdc = core::array::from_fn(|c| {
            quantized
                .lf
                .plane(c)
                .and_then(|p| p.get(cell))
                .copied()
                .unwrap_or(0)
        });
        let coefficients = quantized
            .coefficients
            .get(i)
            .ok_or_else(|| EncodeError::unsupported("a varblock with no coefficients", "I.4"))?;
        varblocks.push(WalkVarblock {
            bx: lx,
            by: ly,
            transform: block.transform,
            hf_mul: block.hf_mul.get(),
            qdc,
            coefficients,
        });
    }

    Ok((
        PassGroupWalk {
            blocks_w,
            blocks_h,
            block_context: &plan.entropy.block_context,
            orders,
            // I.4's `offset = 495 * nb_block_ctx * hfp`, and `hfp` is 0 here.
            offset: 0,
        },
        varblocks,
    ))
}

// ---------------------------------------------------------------------------
// The entropy census (Encoder-plan1.md §9.2)
// ---------------------------------------------------------------------------

/// Drives every I.4 event of the frame's single pass through `sink`.
///
/// Public because the policy crate needs the same events to *choose* a
/// clustering, and it must not re-derive the walk to get them: two walks means
/// two chances to disagree about a context index, and the disagreement would
/// surface as an unexplainable entropy error at emission time.
///
/// # Errors
///
/// As [`walk_pass_group`], plus [`EncodeError::Unsupported`] for a geometry the
/// plan does not cover.
pub fn walk_frame(
    plan: &EmissionPlan,
    geometry: &VardctGeometry,
    sink: &mut impl HfEventSink,
) -> Result<()> {
    let orders = OrderTables::from_order_set(
        &plan
            .entropy
            .passes
            .first()
            .ok_or_else(|| EncodeError::unsupported("a frame with no pass", "F.6"))?
            .orders,
    );
    for group in 0..geometry.num_groups() {
        let (walk, varblocks) = pass_group_walk(plan, geometry, &orders, group)?;
        walk_pass_group(&walk, &varblocks, sink).map_err(EncodeError::Plan)?;
    }
    Ok(())
}

/// The number of pre-clustering contexts a plan's HF model covers (I.3.3).
#[must_use]
pub fn plan_pre_contexts(plan: &EmissionPlan) -> u64 {
    pre_context_count(
        plan.entropy.num_hf_presets,
        plan.entropy.block_context.nb_block_ctx(),
    )
}

/// Counts every I.4 event of the frame, for policy's clustering decision.
///
/// # Errors
///
/// As [`walk_frame`].
pub fn census_frame(plan: &EmissionPlan, geometry: &VardctGeometry) -> Result<CensusSink> {
    let contexts = usize::try_from(plan_pre_contexts(plan)).unwrap_or(0);
    let mut census = CensusSink::new(contexts);
    walk_frame(plan, geometry, &mut census)?;
    Ok(census)
}

/// A [`TokenCensus`] behind the [`HfEventSink`] interface.
struct TokenCensusSink {
    census: TokenCensus,
    error: Option<jpxl_entropy::EntropyError>,
}

impl TokenCensusSink {
    fn record(&mut self, context: PreContextId, value: u32) {
        if self.error.is_some() {
            return;
        }
        let ctx = usize::try_from(context.get()).unwrap_or(usize::MAX);
        if let Err(e) = self.census.record(ctx, value) {
            self.error = Some(e);
        }
    }
}

impl HfEventSink for TokenCensusSink {
    fn nonzeros(&mut self, context: PreContextId, value: u32) {
        self.record(context, value);
    }

    fn coefficient(&mut self, context: PreContextId, value: u32) {
        self.record(context, value);
    }
}

/// Applies the plan's clustering and hybrid-uint choices to the frame's census.
fn build_entropy_tables(
    plan: &EmissionPlan,
    geometry: &VardctGeometry,
    _orders: &OrderTables,
) -> Result<EntropyTables> {
    let pass = plan
        .entropy
        .passes
        .first()
        .ok_or_else(|| EncodeError::unsupported("a frame with no pass", "F.6"))?;

    let contexts = usize::try_from(plan_pre_contexts(plan)).unwrap_or(0);
    let mut sink = TokenCensusSink {
        census: TokenCensus::new(contexts)?,
        error: None,
    };
    walk_frame(plan, geometry, &mut sink)?;
    if let Some(error) = sink.error {
        return Err(EncodeError::from(error));
    }
    let counts = sink.census;

    let clusters: Vec<u8> = pass
        .distributions
        .context_map
        .iter()
        .map(|c| ClusterId::get(*c))
        .collect();
    let context_map = ContextMap::new(clusters)?;
    let configs: Vec<HybridUintConfig> = pass
        .distributions
        .hybrid_uint
        .iter()
        .map(|c| {
            HybridUintConfig::new(
                u32::from(c.split_exponent),
                u32::from(c.msb_in_token),
                u32::from(c.lsb_in_token),
            )
        })
        .collect::<core::result::Result<_, _>>()?;
    let encoder_plan = EncoderPlan::clustered(context_map, CodingMode::Ans, configs)?;
    Ok(EntropyTables::build(&encoder_plan, &counts)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vardct::ids::{GlobalScale, HfMul, QuantLf};
    use crate::vardct::plan::{
        CflGrid, EntropyModelPlan, EntropyPlan, FrameDecision, HfPassEntropyPlan, HistogramPlan,
        HybridUintPlan, LfGroupPlan, LfQuantPlanes, OrderSet, QuantizedFrameIr, QuantizedLfGroup,
        QuantizerDecision, RestorationDecision, SectionLayout, SharpnessGrid, SpatialPlan,
        VarblockCoefficients, VarblockDecision,
    };
    use crate::vardct::validate::validate;
    use jpxl_core::geometry::LfBlockPos;

    /// The smallest legal all-DCT8x8 plan: one 8x8 frame, everything zero.
    fn tiny_plan() -> EmissionPlan {
        let frame = FrameDecision {
            width: 8,
            height: 8,
            group_size_shift: VARDCT_GROUP_SIZE_SHIFT,
            num_passes: 1,
        };
        let geometry = frame.geometry().expect("legal geometry");
        let id = LfGroupId::new(0);
        let blocks = geometry.lf_group_blocks(id).expect("one LF group");
        let tiles = geometry.lf_group_cfl_tiles(id).expect("one LF group");

        let nb_block_ctx = HfBlockContextPlan::Default.nb_block_ctx();
        let pre_contexts = usize::try_from(495 * nb_block_ctx).unwrap_or(0);

        EmissionPlan {
            spatial: SpatialPlan {
                frame,
                quantizer: QuantizerDecision {
                    global_scale: GlobalScale::new(4096).expect("legal"),
                    quant_lf: QuantLf::new(16).expect("legal"),
                },
                lf: LfDecision::vardct_neutral(),
                restoration: RestorationDecision::default(),
                lf_groups: vec![LfGroupPlan {
                    id,
                    blocks: vec![VarblockDecision {
                        origin: LfBlockPos::new(0, 0),
                        transform: TransformType::Dct8x8,
                        hf_mul: HfMul::new(1).expect("legal"),
                    }]
                    .into_boxed_slice(),
                    cfl: CflGrid::zeros(tiles),
                    sharpness: SharpnessGrid::zeros(blocks),
                }]
                .into_boxed_slice(),
            },
            quantized: QuantizedFrameIr {
                lf_groups: vec![QuantizedLfGroup {
                    id,
                    lf: LfQuantPlanes::zeros(blocks),
                    coefficients: vec![VarblockCoefficients::zeros(TransformType::Dct8x8)]
                        .into_boxed_slice(),
                }]
                .into_boxed_slice(),
            },
            entropy: EntropyPlan {
                block_context: HfBlockContextPlan::Default,
                num_hf_presets: 1,
                passes: vec![HfPassEntropyPlan {
                    orders: OrderSet::natural(),
                    distributions: EntropyModelPlan {
                        context_map: vec![ClusterId::new(0); pre_contexts].into_boxed_slice(),
                        histograms: vec![HistogramPlan::new(vec![1u32]).expect("legal")]
                            .into_boxed_slice(),
                        hybrid_uint: vec![HybridUintPlan {
                            split_exponent: 4,
                            msb_in_token: 2,
                            lsb_in_token: 0,
                        }]
                        .into_boxed_slice(),
                    },
                    group_presets: vec![crate::vardct::ids::PresetId::new(0)].into_boxed_slice(),
                }]
                .into_boxed_slice(),
            },
            sections: SectionLayout::for_geometry(&geometry),
        }
    }

    #[test]
    fn the_smallest_legal_plan_emits_a_codestream() {
        let plan = validate(tiny_plan()).expect("legal plan");
        let bytes = write_codestream(&plan).expect("writes");
        assert_eq!(bytes.get(..2), Some(&[0xFFu8, 0x0A][..]));
        // One section: F.3.1's single-section form for one group and one pass.
        assert_eq!(plan.plan().sections.kinds.len(), 1);
    }

    #[test]
    fn a_group_size_shift_the_frame_header_cannot_signal_is_refused() {
        let mut plan = tiny_plan();
        plan.spatial.frame.group_size_shift = 2;
        // The plan is still *structurally* legal — validate accepts it — and it
        // is the writer that has nowhere to put the field.
        assert!(validate(plan.clone()).is_ok());
        assert!(matches!(
            check_supported(&plan),
            Err(EncodeError::Unsupported { clause: "F.2", .. })
        ));
    }

    #[test]
    fn adaptive_lf_smoothing_is_refused_rather_than_left_uninverted() {
        let mut plan = tiny_plan();
        plan.spatial.lf.adaptive_smoothing = true;
        assert!(matches!(
            check_supported(&plan),
            Err(EncodeError::Unsupported {
                clause: "G.1.2",
                ..
            })
        ));
    }

    #[test]
    fn the_restoration_filters_are_refused() {
        let mut plan = tiny_plan();
        plan.spatial.restoration.gaborish = true;
        assert!(matches!(
            check_supported(&plan),
            Err(EncodeError::Unsupported { clause: "J.1", .. })
        ));
        let mut plan = tiny_plan();
        plan.spatial.restoration.epf_iters = 2;
        assert!(matches!(
            check_supported(&plan),
            Err(EncodeError::Unsupported { clause: "J.1", .. })
        ));
    }

    #[test]
    fn a_non_dct8x8_transform_is_refused() {
        let mut plan = tiny_plan();
        if let Some(group) = plan.spatial.lf_groups.first_mut()
            && let Some(block) = group.blocks.first_mut()
        {
            block.transform = TransformType::Dct4x4;
        }
        assert!(matches!(
            check_supported(&plan),
            Err(EncodeError::Unsupported { clause: "I.1", .. })
        ));
    }

    #[test]
    fn a_custom_coefficient_order_is_refused_until_its_permutation_writer_exists() {
        let mut plan = tiny_plan();
        let natural = jpxl_core::varblock::natural_coeff_order(8, 8);
        let mut swapped = natural.clone();
        swapped.swap(3, 40);
        if let Some(pass) = plan.entropy.passes.first_mut() {
            pass.orders = OrderSet::natural()
                .with_order(crate::vardct::ids::OrderId::new(0), 1, swapped)
                .expect("a permutation");
        }
        assert!(matches!(
            check_supported(&plan),
            Err(EncodeError::Unsupported {
                clause: "I.3.1",
                ..
            })
        ));
    }

    /// The claim the whole rate loop rests on: the priced size **is** the
    /// emitted size, because it is the same run of the same writer.
    #[test]
    fn a_priced_size_is_the_emitted_size_byte_for_byte() {
        let plan = validate(tiny_plan()).expect("legal plan");
        let emission = emit_codestream(&plan).expect("emits");
        let sizing = price_codestream(&plan).expect("prices");
        assert_eq!(sizing, emission.sizing);
        assert_eq!(
            sizing.total,
            u64::try_from(emission.bytes.len()).expect("small"),
            "the accounted total must be the buffer length"
        );
        assert_eq!(
            write_codestream(&plan).expect("writes"),
            emission.bytes,
            "write_codestream is emit_codestream's bytes"
        );
    }

    /// Every byte is attributed to exactly one of headers, TOC or a section.
    #[test]
    fn the_accounting_partitions_the_stream() {
        let plan = validate(tiny_plan()).expect("legal plan");
        let sizing = price_codestream(&plan).expect("prices");
        // The frame header and the TOC share one alignment boundary: F.2 does
        // not pad after the header, and F.3.3 pads inside the table.
        let header_and_toc = (sizing.frame_header_bits + sizing.toc_bits).div_ceil(8);
        assert_eq!(
            sizing.image_headers + header_and_toc + sizing.section_bytes(),
            sizing.total
        );
        assert_eq!(sizing.overhead(), sizing.image_headers + header_and_toc);
        // One 8x8 frame is F.3.1's single-section form, and its one section
        // carries the coefficients.
        assert_eq!(sizing.sections.len(), 1);
        assert_eq!(
            sizing.sections.first().map(|s| s.kind),
            Some(crate::vardct::geometry::SectionKind::Whole)
        );
        assert_eq!(sizing.coefficient_bytes(), sizing.section_bytes());
    }

    #[test]
    fn ceil_log2_is_the_field_width_i26_and_g24_need() {
        assert_eq!(ceil_log2(0), 0);
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
        assert_eq!(ceil_log2(1024), 10);
    }
}
