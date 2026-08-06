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
    CodingMode, ContextMap, ContextMapForm, EncoderPlan, EntropyTables, SymbolEncoder, TokenCensus,
};

use crate::entropy::pack_signed;
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

/// 18181-1 I.2.2 `ReadThreshold()`:
/// `U32(u(4), 16 + u(8), 272 + u(16), 65808 + u(32))`.
const LF_THRESHOLD_SPEC: U32Spec = U32Spec::new([
    U32Dist::bits(4),
    U32Dist::BitsOffset {
        bits: 8,
        offset: 16,
    },
    U32Dist::BitsOffset {
        bits: 16,
        offset: 272,
    },
    U32Dist::BitsOffset {
        bits: 32,
        offset: 65808,
    },
]);

/// 18181-1 I.2.2 `qf_thresholds` loop:
/// `U32(u(2), 4 + u(3), 12 + u(5), 44 + u(8))` of the raw value; the decoder
/// then adds one, so the wire stores `threshold - 1`.
const QF_THRESHOLD_SPEC: U32Spec = U32Spec::new([
    U32Dist::bits(2),
    U32Dist::BitsOffset { bits: 3, offset: 4 },
    U32Dist::BitsOffset {
        bits: 5,
        offset: 12,
    },
    U32Dist::BitsOffset {
        bits: 8,
        offset: 44,
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
    // Restoration filters may be signalled (slice 20). Without inverse
    // Gaborish preconditioning of the XYB planes the reconstructed samples
    // will not match the encoder's quant targets; that preconditioning is a
    // planner responsibility. Emission only refuses illegal epf_iters.
    if spatial.restoration.epf_iters > 3 {
        return Err(EncodeError::ValueOutOfRange {
            what: "epf_iters",
            value: i64::from(spatial.restoration.epf_iters),
        });
    }
    // I.2.2 custom maps and I.2.6 multi-preset counts are written by the
    // section writers; validate() is the gate for density, bsize, and the
    // num_hf_presets ≤ num_groups bound.
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

/// As [`write_codestream`], with an explicit resource policy for section
/// parallelism (Opt-P).
///
/// # Errors
///
/// As [`emit_codestream`].
pub fn write_codestream_with(
    plan: &ValidatedEmissionPlan,
    resources: crate::EncodeResources,
) -> Result<Vec<u8>> {
    Ok(emit_codestream_with(plan, resources)?.bytes)
}

/// The exact size of the codestream `plan` emits, without keeping the bytes.
///
/// Uses the **same** write path as [`emit_codestream`] with count-only
/// [`BitWriter`]s and a length-only [`SectionStore`]: no second size model.
/// See [`size`](crate::vardct::size).
///
/// # Errors
///
/// As [`emit_codestream`].
pub fn price_codestream(plan: &ValidatedEmissionPlan) -> Result<CodestreamSizing> {
    Ok(emit_codestream_mode(plan, EmitMode::Count, crate::EncodeResources::serial())?.sizing)
}

/// Whether section bodies and the final codestream buffer are retained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EmitMode {
    /// Full payload for the returned codestream.
    Store,
    /// Exact bit accounting only (rate-loop intermediate prices).
    Count,
}

/// Encodes a validated plan, reporting where every byte went.
///
/// # Errors
///
/// [`EncodeError::Unsupported`] for a plan shape [`check_supported`] refuses,
/// [`EncodeError::Entropy`] if the entropy layer rejects a symbol, and
/// [`EncodeError::ValueOutOfRange`] for a field that cannot be expressed.
pub fn emit_codestream(plan: &ValidatedEmissionPlan) -> Result<Emission> {
    emit_codestream_with(plan, crate::EncodeResources::serial())
}

/// As [`emit_codestream`], with an explicit resource policy.
///
/// Independent multi-section bodies may run on multiple workers; results are
/// reduced in F.3.1 order so Contract A holds across thread counts.
///
/// # Errors
///
/// As [`emit_codestream`].
pub fn emit_codestream_with(
    plan: &ValidatedEmissionPlan,
    resources: crate::EncodeResources,
) -> Result<Emission> {
    emit_codestream_mode(plan, EmitMode::Store, resources)
}

fn emit_codestream_mode(
    plan: &ValidatedEmissionPlan,
    mode: EmitMode,
    resources: crate::EncodeResources,
) -> Result<Emission> {
    let inner = plan.plan();
    check_supported(inner)?;
    let geometry = inner.spatial.frame.geometry().map_err(EncodeError::Plan)?;

    let mut w = match mode {
        EmitMode::Store => BitWriter::new(),
        EmitMode::Count => BitWriter::counting(),
    };
    write_image_headers(&mut w, geometry.width(), geometry.height())?;
    // F.1: every frame starts on a byte boundary, so this division is exact.
    w.zero_pad_to_byte();
    let image_headers = w.bit_len() / 8;

    write_frame_header(
        &mut w,
        NEUTRAL_QM_SCALE,
        NEUTRAL_QM_SCALE,
        inner.spatial.restoration,
    )?;
    let frame_header_bits = w.bit_len() - image_headers * 8;

    let store = write_frame_body(inner, &geometry, mode, resources)?;
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

    let total = w.byte_len();
    let bytes = match mode {
        EmitMode::Store => w.into_bytes(),
        EmitMode::Count => Vec::new(),
    };
    let sizing = CodestreamSizing {
        total,
        image_headers,
        frame_header_bits,
        toc_bits,
        sections,
    };
    debug_assert!(
        mode == EmitMode::Count || sizing.total == u64::try_from(bytes.len()).unwrap_or(u64::MAX),
        "stored emit total must equal buffer length"
    );
    Ok(Emission { bytes, sizing })
}

/// Builds every section of the frame, in F.3.1 order.
fn write_frame_body(
    plan: &EmissionPlan,
    geometry: &VardctGeometry,
    mode: EmitMode,
    resources: crate::EncodeResources,
) -> Result<SectionStore> {
    let orders = OrderTables::from_order_set(
        &plan
            .entropy
            .passes
            .first()
            .ok_or_else(|| EncodeError::unsupported("a frame with no pass", "F.6"))?
            .orders,
    );
    let tables = build_entropy_tables(plan, geometry, &orders)?;

    let mut store = match mode {
        EmitMode::Store => SectionStore::new(),
        EmitMode::Count => SectionStore::counting(),
    };
    if geometry.is_single_section() {
        let mut body = match mode {
            EmitMode::Store => BitWriter::new(),
            EmitMode::Count => BitWriter::counting(),
        };
        write_lf_global(plan, &mut body)?;
        write_lf_group(plan, geometry, LfGroupId::new(0), &mut body)?;
        write_hf_global(plan, geometry, &tables, &mut body)?;
        write_pass_group(plan, geometry, &orders, &tables, 0, &mut body)?;
        body.zero_pad_to_byte();
        if mode == EmitMode::Store {
            store.push(body.into_bytes());
        } else {
            store.push_len(usize::try_from(body.byte_len()).unwrap_or(usize::MAX));
        }
        return Ok(store);
    }

    // Globals serial; LF groups and pass groups are independent given the plan
    // and entropy tables — map them, reduce in TOC index order.
    store_push(
        &mut store,
        mode,
        section_result(mode, |w| write_lf_global(plan, w))?,
    );

    let n_lf = usize::try_from(geometry.num_lf_groups()).unwrap_or(0);
    let lf_workers = resources.workers_for(n_lf);
    let lf_parts = crate::resources::ordered_map(n_lf, lf_workers, |index| {
        let id = LfGroupId::new(u32::try_from(index).unwrap_or(u32::MAX));
        section_result(mode, |w| write_lf_group(plan, geometry, id, w))
    })?;
    for part in lf_parts {
        store_push(&mut store, mode, part);
    }

    store_push(
        &mut store,
        mode,
        section_result(mode, |w| write_hf_global(plan, geometry, &tables, w))?,
    );

    let n_pg = usize::try_from(geometry.num_groups()).unwrap_or(0);
    let pg_workers = resources.workers_for(n_pg);
    let pg_parts = crate::resources::ordered_map(n_pg, pg_workers, |group| {
        let group = u64::try_from(group).unwrap_or(u64::MAX);
        section_result(mode, |w| {
            write_pass_group(plan, geometry, &orders, &tables, group, w)
        })
    })?;
    for part in pg_parts {
        store_push(&mut store, mode, part);
    }
    Ok(store)
}

/// One section as either retained bytes or a count-only length.
enum SectionPart {
    Bytes(Vec<u8>),
    Len(usize),
}

fn section_result(
    mode: EmitMode,
    body: impl FnOnce(&mut BitWriter) -> Result<()>,
) -> Result<SectionPart> {
    match mode {
        EmitMode::Store => Ok(SectionPart::Bytes(section_bytes(body)?)),
        EmitMode::Count => Ok(SectionPart::Len(section_len(body)?)),
    }
}

fn store_push(store: &mut SectionStore, mode: EmitMode, part: SectionPart) {
    match (mode, part) {
        (EmitMode::Store, SectionPart::Bytes(b)) => store.push(b),
        (EmitMode::Count, SectionPart::Len(n)) => store.push_len(n),
        (EmitMode::Store, SectionPart::Len(n)) => store.push(vec![0; n]),
        (EmitMode::Count, SectionPart::Bytes(b)) => store.push_len(b.len()),
    }
}

/// Runs `body` into a fresh byte-aligned section buffer.
fn section_bytes(body: impl FnOnce(&mut BitWriter) -> Result<()>) -> Result<Vec<u8>> {
    let mut w = BitWriter::new();
    body(&mut w)?;
    w.zero_pad_to_byte();
    Ok(w.into_bytes())
}

/// Runs `body` into a counting writer and returns the byte-aligned length.
fn section_len(body: impl FnOnce(&mut BitWriter) -> Result<()>) -> Result<usize> {
    let mut w = BitWriter::counting();
    body(&mut w)?;
    w.zero_pad_to_byte();
    Ok(usize::try_from(w.byte_len()).unwrap_or(usize::MAX))
}

// ---------------------------------------------------------------------------
// I.2.2 — HF block context
// ---------------------------------------------------------------------------

/// Writes the HF block-context model (18181-1 I.2.2).
///
/// The default map is one bit. A custom model writes the three LF threshold
/// vectors, the QF thresholds, and a C.2.2 clustering map of length `bsize`.
fn write_hf_block_context(model: &HfBlockContextPlan, w: &mut BitWriter) -> Result<()> {
    match model {
        HfBlockContextPlan::Default => {
            w.write_bits(1, 1)?;
            Ok(())
        }
        HfBlockContextPlan::Custom {
            lf_thresholds,
            qf_thresholds,
            map,
        } => {
            w.write_bits(1, 0)?;
            for row in lf_thresholds {
                let count = u32::try_from(row.len()).map_err(|_| EncodeError::ValueOutOfRange {
                    what: "nb_lf_thr",
                    value: i64::try_from(row.len()).unwrap_or(i64::MAX),
                })?;
                if count > 15 {
                    return Err(EncodeError::ValueOutOfRange {
                        what: "nb_lf_thr",
                        value: i64::from(count),
                    });
                }
                w.write_bits(4, count)?;
                for &threshold in row {
                    w.write_u32(&LF_THRESHOLD_SPEC, pack_signed(threshold))?;
                }
            }
            let qf_count =
                u32::try_from(qf_thresholds.len()).map_err(|_| EncodeError::ValueOutOfRange {
                    what: "nb_qf_thr",
                    value: i64::try_from(qf_thresholds.len()).unwrap_or(i64::MAX),
                })?;
            if qf_count > 15 {
                return Err(EncodeError::ValueOutOfRange {
                    what: "nb_qf_thr",
                    value: i64::from(qf_count),
                });
            }
            w.write_bits(4, qf_count)?;
            for &threshold in qf_thresholds {
                // Decoder stores `1 + U32(...)`; refuse a zero threshold so the
                // subtraction cannot wrap.
                let raw = threshold
                    .checked_sub(1)
                    .ok_or(EncodeError::ValueOutOfRange {
                        what: "qf_threshold",
                        value: i64::from(threshold),
                    })?;
                w.write_u32(&QF_THRESHOLD_SPEC, raw)?;
            }
            let context_map = ContextMap::new(map.clone())?;
            context_map.write(w, ContextMapForm::Auto)?;
            Ok(())
        }
    }
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

    write_hf_block_context(&plan.entropy.block_context, w)?;

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
    for pass in 0..plan.spatial.frame.num_passes {
        let entropy_pass = plan
            .entropy
            .passes
            .get(usize::try_from(pass).unwrap_or(usize::MAX))
            .or_else(|| plan.entropy.passes.first())
            .ok_or_else(|| EncodeError::unsupported("a frame with no pass", "F.6"))?;
        // I.3.1: used_orders, and the F.3.2 permutation stream when any Order
        // ID carries a custom order.
        write_hf_coeff_orders(entropy_pass, w)?;
        // I.3.3: the pre-clustered distributions, read here and used by every
        // pass group of this pass.
        tables.write_bundle(w)?;
    }
    Ok(())
}

/// F.3.2's `GetContext(x) = min(7, ceil(log2(x + 1)))`.
const fn permutation_context(x: u32) -> usize {
    let bits = if x == 0 { 0 } else { 32 - x.leading_zeros() };
    if bits > 7 { 7 } else { bits as usize }
}

/// Number of pre-clustered distributions of the F.3.2 permutation stream —
/// the count `GetContext` can address. The decoder pins the same value in
/// `jpxl-decode`'s `frame::toc`.
const PERMUTATION_NUM_DIST: usize = 8;

/// A permutation's Lehmer code: the exact inverse of F.3.2's reconstruction
/// (`temp` starts as `[0, size)`; each step emits the index of the next
/// element and removes it).
///
/// # Errors
///
/// [`EncodeError::Unsupported`] if `perm` is not a permutation of `0..len`.
fn permutation_to_lehmer(perm: &[u32]) -> Result<Vec<u32>> {
    let mut temp: Vec<u32> = (0..u32::try_from(perm.len()).unwrap_or(u32::MAX)).collect();
    let mut lehmer = Vec::with_capacity(perm.len());
    for &element in perm {
        let Some(index) = temp.iter().position(|&t| t == element) else {
            return Err(EncodeError::unsupported(
                "a coefficient order that is not a permutation",
                "F.3.2",
            ));
        };
        lehmer.push(u32::try_from(index).unwrap_or(u32::MAX));
        temp.remove(index);
    }
    Ok(lehmer)
}

/// Writes I.3.1's `used_orders` and, when non-zero, the shared F.3.2
/// permutation stream the decoder consumes for every used `(Order ID,
/// channel)` pair.
///
/// The stream mirrors `jpxl-decode`'s `read_hf_coeff_orders` exactly: one
/// entropy stream (eight pre-clustered distributions, C.1) opened once,
/// carrying per permutation an `end` symbol in context `GetContext(size)`
/// followed by `end` Lehmer values, each in the context of its predecessor;
/// permutations trim their trailing zero Lehmer entries and never permute the
/// LLF prefix (`skip = size / 64`). An override on some channels of a used
/// Order ID writes identity permutations (`end = 0`) for the others.
fn write_hf_coeff_orders(
    pass: &crate::vardct::plan::HfPassEntropyPlan,
    w: &mut BitWriter,
) -> Result<()> {
    use jpxl_core::varblock::{NUM_ORDER_IDS, natural_coeff_order, order_id_dims};

    let overrides = pass.orders.overrides();
    let mut used = 0u32;
    for over in overrides {
        used |= 1u32 << u32::from(over.order_id.get());
    }
    w.write_u32(&USED_ORDERS_SPEC, used)?;
    if used == 0 {
        return Ok(());
    }

    // The symbol sequence, in exactly the order the decoder reads it.
    let mut symbols: Vec<(usize, u32)> = Vec::new();
    for order_id in 0..NUM_ORDER_IDS {
        if (used >> order_id) & 1 == 0 {
            continue;
        }
        let natural = order_id_dims(order_id)
            .map(|(bw, bh)| natural_coeff_order(bw, bh))
            .unwrap_or_default();
        let size = u32::try_from(natural.len()).unwrap_or(u32::MAX);
        let skip = size / 64;
        // Cell -> natural-order position, for deriving `nat_ord_perm` from a
        // stored order table (which holds cells).
        let mut position_of = vec![u32::MAX; natural.len()];
        for (position, &cell) in natural.iter().enumerate() {
            if let Some(slot) = position_of.get_mut(usize::try_from(cell).unwrap_or(usize::MAX)) {
                *slot = u32::try_from(position).unwrap_or(u32::MAX);
            }
        }

        for channel in 0..3u8 {
            let perm: Vec<u32> = match overrides
                .iter()
                .find(|o| usize::from(o.order_id.get()) == order_id && o.channel == channel)
            {
                None => (0..size).collect(),
                Some(over) => over
                    .table
                    .iter()
                    .map(|&cell| {
                        position_of
                            .get(usize::try_from(cell).unwrap_or(usize::MAX))
                            .copied()
                            .unwrap_or(u32::MAX)
                    })
                    .collect(),
            };
            let lehmer = permutation_to_lehmer(&perm)?;
            if lehmer
                .iter()
                .take(usize::try_from(skip).unwrap_or(0))
                .any(|&l| l != 0)
            {
                return Err(EncodeError::unsupported(
                    "a coefficient order that permutes the LLF prefix",
                    "F.3.2",
                ));
            }
            let end = lehmer
                .iter()
                .rposition(|&l| l != 0)
                .map_or(0, |last| last + 1)
                .saturating_sub(usize::try_from(skip).unwrap_or(0));

            symbols.push((
                permutation_context(size),
                u32::try_from(end).unwrap_or(u32::MAX),
            ));
            let mut previous = 0u32;
            for index in 0..end {
                let value = lehmer
                    .get(usize::try_from(skip).unwrap_or(0) + index)
                    .copied()
                    .unwrap_or(0);
                let context = permutation_context(if index > 0 { previous } else { 0 });
                symbols.push((context, value));
                previous = value;
            }
        }
    }

    // One self-contained stream: bundle, then the ANS payload — the encoder
    // half of `SymbolDecoder::open` / `read_uint`* / `finish`.
    let mut census = TokenCensus::new(PERMUTATION_NUM_DIST)?;
    for &(context, value) in &symbols {
        census.record(context, value)?;
    }
    let context_map = ContextMap::identity(PERMUTATION_NUM_DIST)?;
    let configs = vec![HybridUintConfig::new(4, 2, 0)?; PERMUTATION_NUM_DIST];
    let encoder_plan = EncoderPlan::clustered(context_map, CodingMode::Ans, configs)?;
    let tables = EntropyTables::build(&encoder_plan, &census)?;
    tables.write_bundle(w)?;
    let mut encoder = SymbolEncoder::new(&tables);
    for &(context, value) in &symbols {
        encoder.push_uint(context, value)?;
    }
    encoder.write_stream(w)?;
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
    // I.4: hfp is `u(ceil(log2(num_hf_presets)))`, zero bits at one preset.
    let bits = ceil_log2(u64::from(plan.entropy.num_hf_presets.max(1)));
    let hfp = group_hfp(plan, group)?;
    w.write_bits(bits, hfp)?;

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

/// I.4's `hfp` for one pass group, from the plan's assignment table.
fn group_hfp(plan: &EmissionPlan, group: u64) -> Result<u32> {
    let pass = plan
        .entropy
        .passes
        .first()
        .ok_or_else(|| EncodeError::unsupported("a frame with no pass", "F.6"))?;
    let index = usize::try_from(group).unwrap_or(usize::MAX);
    let preset = pass
        .group_presets
        .get(index)
        .ok_or_else(|| EncodeError::unsupported("a group past the preset table", "I.4"))?;
    Ok(preset.get())
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

    let hfp = u64::from(group_hfp(plan, group)?);
    let nb_block_ctx = plan.entropy.block_context.nb_block_ctx();
    // I.4: `offset = 495 * nb_block_ctx * hfp`.
    let offset = 495u64.saturating_mul(nb_block_ctx).saturating_mul(hfp);

    Ok((
        PassGroupWalk {
            blocks_w,
            blocks_h,
            block_context: &plan.entropy.block_context,
            orders,
            offset,
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

        EmissionPlan::new(
            SpatialPlan {
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
            QuantizedFrameIr {
                lf_groups: vec![QuantizedLfGroup {
                    id,
                    lf: LfQuantPlanes::zeros(blocks),
                    coefficients: vec![VarblockCoefficients::zeros(TransformType::Dct8x8)]
                        .into_boxed_slice(),
                }]
                .into_boxed_slice(),
            },
            EntropyPlan {
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
            SectionLayout::for_geometry(&geometry),
        )
    }

    /// The composition-direction gate for the F.3.2 writer: two plans with
    /// identical coefficients, one at natural orders and one with a custom
    /// permutation, must produce different bytes and identical pixels — the
    /// order sequences symbols, it never moves values between cells. The
    /// decode side is `jpxl-decode`'s independent I.3.1 reader, so a writer
    /// that stored the permutation the wrong way round (or composed it on the
    /// wrong side) scrambles the coefficient array and fails the pixel
    /// comparison immediately.
    #[test]
    #[allow(
        clippy::indexing_slicing,
        reason = "test-only fixture construction over containers built above"
    )]
    fn a_custom_coefficient_order_round_trips_to_the_same_pixels() {
        use jpxl_core::varblock::{natural_coeff_order, order_id_dims};

        let mut plan = tiny_plan();
        let mut channels: [Vec<i32>; NUM_CHANNELS] = core::array::from_fn(|_| vec![0i32; 64]);
        for (cell, value) in [(1usize, 3i32), (8, -2), (9, 1), (17, 5), (26, -4), (63, 1)] {
            for (index, channel) in channels.iter_mut().enumerate() {
                channel[cell] = value + i32::try_from(index).unwrap_or(0);
            }
        }
        plan.quantized_mut().lf_groups[0].coefficients =
            vec![VarblockCoefficients::new(TransformType::Dct8x8, channels).expect("legal")]
                .into_boxed_slice();

        let natural =
            write_codestream(&validate(plan.clone()).expect("legal")).expect("writes natural");

        // Reverse the non-LLF tail of Order ID 0 (DCT8x8) for every channel:
        // a maximally un-natural but legal permutation.
        let dims = order_id_dims(0).expect("order 0");
        let mut table = natural_coeff_order(dims.0, dims.1);
        let skip = table.len() / 64;
        table[skip..].reverse();
        let mut orders = OrderSet::natural();
        for channel in 0..3u8 {
            orders = orders
                .with_order(crate::vardct::ids::OrderId::new(0), channel, table.clone())
                .expect("a legal permutation");
        }
        plan.entropy.passes[0].orders = orders;
        let custom =
            write_codestream(&validate(plan).expect("legal")).expect("writes custom order");

        assert_ne!(natural, custom, "the permutation must reach the wire");

        let limits = jpxl_core::limits::Limits::default();
        let natural_image =
            jpxl_decode::decode::decode(&natural, &limits).expect("natural decodes");
        let custom_image = jpxl_decode::decode::decode(&custom, &limits).expect("custom decodes");
        for (a, b) in natural_image.planes.iter().zip(custom_image.planes.iter()) {
            assert_eq!(
                a.samples, b.samples,
                "a coefficient order must never move values between cells"
            );
        }
    }

    /// I.2.2 custom-map gate: a hand-built non-default block context must
    /// survive the writer and round-trip through the decoder's independent
    /// reader with thresholds and map entries bit-exact. Pixels stay equal to
    /// the default-map encoding of the same coefficients — the model only
    /// renames contexts.
    #[test]
    #[allow(
        clippy::indexing_slicing,
        reason = "test-only fixture construction over containers built above"
    )]
    fn a_custom_block_context_round_trips_and_keeps_pixels() {
        use jpxl_bitstream::BitReader;
        use jpxl_core::limits::{AllocGuard, Limits};
        use jpxl_decode::vardct::block_ctx::read_hf_block_context;

        let mut plan = tiny_plan();
        // Non-trivial coefficients so entropy actually fires under both maps.
        let mut channels: [Vec<i32>; NUM_CHANNELS] = core::array::from_fn(|_| vec![0i32; 64]);
        for (cell, value) in [(1usize, 4i32), (9, -3), (18, 2), (27, -1), (45, 6)] {
            for (index, channel) in channels.iter_mut().enumerate() {
                channel[cell] = value + i32::try_from(index).unwrap_or(0);
            }
        }
        plan.quantized_mut().lf_groups[0].coefficients =
            vec![VarblockCoefficients::new(TransformType::Dct8x8, channels).expect("legal")]
                .into_boxed_slice();
        // One LF threshold per channel and one QF threshold expand bsize to
        // 39 * 2 * 2^3 = 312; keep the map dense with 8 clusters so the
        // C.2.2 writer has something non-trivial to emit.
        let bsize = 39 * 2 * 2 * 2 * 2;
        let map: Vec<u8> = (0..bsize)
            .map(|i| u8::try_from(i % 8).unwrap_or(0))
            .collect();
        plan.entropy.block_context = HfBlockContextPlan::Custom {
            lf_thresholds: [vec![0], vec![-1], vec![2]],
            qf_thresholds: vec![1],
            map: map.clone(),
        };
        // Rebuild the provisional entropy model size for the new nb_block_ctx.
        let nb = plan.entropy.block_context.nb_block_ctx();
        let pre = usize::try_from(495 * nb).unwrap_or(0);
        plan.entropy.passes[0].distributions.context_map =
            vec![ClusterId::new(0); pre].into_boxed_slice();

        let default_plan = {
            let mut p = plan.clone();
            p.entropy.block_context = HfBlockContextPlan::Default;
            let pre_d = usize::try_from(495 * 15).unwrap_or(0);
            p.entropy.passes[0].distributions.context_map =
                vec![ClusterId::new(0); pre_d].into_boxed_slice();
            p
        };
        let default_bytes =
            write_codestream(&validate(default_plan).expect("legal default")).expect("writes");
        let custom_bytes =
            write_codestream(&validate(plan).expect("legal custom")).expect("writes custom");
        assert_ne!(
            default_bytes, custom_bytes,
            "a custom I.2.2 model must reach the wire"
        );

        // Locate LfGlobal's I.2.2 field by decoding the whole frame: the
        // public decode path is the oracle; for field-level exactness, re-read
        // the block-context bits from a hand-written fragment matching what
        // `write_hf_block_context` emits.
        let mut fragment = BitWriter::new();
        write_hf_block_context(
            &HfBlockContextPlan::Custom {
                lf_thresholds: [vec![0], vec![-1], vec![2]],
                qf_thresholds: vec![1],
                map: map.clone(),
            },
            &mut fragment,
        )
        .expect("writes fragment");
        let frag_bytes = fragment.into_bytes();
        let mut reader = BitReader::new(&frag_bytes);
        let mut guard = AllocGuard::new(&Limits::default());
        let decoded = read_hf_block_context(&mut reader, &mut guard).expect("decodes I.2.2");
        assert_eq!(decoded.lf_thresholds(0), &[0]);
        assert_eq!(decoded.lf_thresholds(1), &[-1]);
        assert_eq!(decoded.lf_thresholds(2), &[2]);
        assert_eq!(decoded.qf_thresholds(), &[1]);
        assert_eq!(decoded.map(), map.as_slice());
        assert_eq!(decoded.nb_block_ctx(), 8);

        let limits = Limits::default();
        let default_image =
            jpxl_decode::decode::decode(&default_bytes, &limits).expect("default decodes");
        let custom_image =
            jpxl_decode::decode::decode(&custom_bytes, &limits).expect("custom decodes");
        for (a, b) in default_image.planes.iter().zip(custom_image.planes.iter()) {
            assert_eq!(
                a.samples, b.samples,
                "block context renames distributions; it must not move samples"
            );
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
        plan.spatial_mut().frame.group_size_shift = 2;
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
        plan.spatial_mut().lf.adaptive_smoothing = true;
        assert!(matches!(
            check_supported(&plan),
            Err(EncodeError::Unsupported {
                clause: "G.1.2",
                ..
            })
        ));
    }

    #[test]
    fn restoration_filters_may_be_signalled() {
        // Slice 20: gab/EPF are legal on the wire; inverse preconditioning of
        // the XYB planes is a planner duty, not an emission refusal.
        let mut plan = tiny_plan();
        plan.spatial_mut().restoration.gaborish = true;
        plan.spatial_mut().restoration.epf_iters = 2;
        assert!(check_supported(&plan).is_ok());
        let mut plan = tiny_plan();
        plan.spatial_mut().restoration.epf_iters = 4;
        assert!(matches!(
            check_supported(&plan),
            Err(EncodeError::ValueOutOfRange {
                what: "epf_iters",
                ..
            })
        ));
    }

    #[test]
    fn a_non_dct8x8_transform_is_refused() {
        let mut plan = tiny_plan();
        if let Some(group) = plan.spatial_mut().lf_groups.first_mut()
            && let Some(block) = group.blocks.first_mut()
        {
            block.transform = TransformType::Dct4x4;
        }
        assert!(matches!(
            check_supported(&plan),
            Err(EncodeError::Unsupported { clause: "I.1", .. })
        ));
    }

    /// F.3.2 forbids permuting the LLF prefix (`skip = size / 64`): those
    /// coefficients are never coded by I.4, so an order that moves them has
    /// no wire representation and must be refused, not silently repaired.
    #[test]
    fn an_order_permuting_the_llf_prefix_is_refused() {
        use jpxl_core::varblock::{natural_coeff_order, order_id_dims};

        let mut plan = tiny_plan();
        // Order ID 2 is large enough to have a non-empty LLF prefix.
        let dims = order_id_dims(2).expect("order 2");
        let mut table = natural_coeff_order(dims.0, dims.1);
        assert!(
            table.len() / 64 >= 1,
            "the fixture needs a non-empty prefix"
        );
        let last = table.len() - 1;
        table.swap(0, last);
        if let Some(pass) = plan.entropy.passes.first_mut() {
            pass.orders = OrderSet::natural()
                .with_order(crate::vardct::ids::OrderId::new(2), 1, table)
                .expect("a permutation");
        }
        assert!(matches!(
            write_codestream(&validate(plan).expect("legal")),
            Err(EncodeError::Unsupported {
                clause: "F.3.2",
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
