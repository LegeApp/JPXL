//! Reading the dequantization matrix parameters of 18181-1 I.2.4.
//!
//! The **tables and the weights computation moved to
//! [`jpxl_core::dequant`]** in the slice-12 wave: an encoder has to choose
//! quantized integers against exactly the same weights a decoder reconstructs
//! with, and two transcriptions of Table I.6 would be free to drift. What is
//! left here is the part that is bitstream control flow — the `u(3)` mode, the
//! `F16()` parameter matrices, and RAW's inline modular sub-bitstream — which
//! is deliberately *not* shared with the encoder (`AGENTS.md`: share math,
//! never share bitstream control flow).
//!
//! Everything the rest of the decoder used to import from this module is
//! re-exported below, so the move is invisible to its callers.

// Every index below is a loop bound over a locally allocated buffer whose
// dimensions were computed in the same function; the stream-derived indices go
// through `get`.
#![allow(clippy::indexing_slicing)]

use jpxl_bitstream::trace_field;
use jpxl_bitstream::{BitReader, read_bool, read_f16_as_f32};
use jpxl_core::limits::AllocGuard;
use jpxl_core::varblock::NUM_DEQUANT_MATRICES;

use crate::error::{DecodeError, Result};
use crate::frame::geometry::FrameGeometry;
use crate::frame::stream_index;
use crate::modular::{Channel, ChannelSpec, ModularOptions, TreeSource, decode_sub_bitstream_with};

use super::block_ctx::read_num_hf_presets;

pub use jpxl_core::dequant::{
    AFV_FREQ_POSITION_IS_TRANSPOSED, DCT2_DC_WEIGHT_IS_ONE, DCT128X256_DEFAULT_BASES_AS_PRINTED,
    DequantMatrices, DequantMatrix, DequantParams, EncodingMode, RawMatrixRequest, WeightMatrix,
    default_params, matrix_size,
};

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/// Widening for error payloads and guard charges.
fn as_u64(v: usize) -> u64 {
    u64::try_from(v).unwrap_or(u64::MAX)
}

/// I.2.4's `* 64` applied to `params` and to the first column of `dct_params`.
const PARAM_SCALE: f64 = 64.0;

/// Reads a `3 x n` matrix of `F16()` in raster order (rows are channels).
fn read_matrix_3xn(
    reader: &mut BitReader<'_>,
    n: usize,
    name: &'static str,
) -> Result<[Vec<f64>; 3]> {
    let mut rows = [Vec::new(), Vec::new(), Vec::new()];
    for row in &mut rows {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(f64::from(trace_field!(
                reader,
                name,
                read_f16_as_f32(reader)
            )?));
        }
        *row = out;
    }
    Ok(rows)
}

/// I.2.4's `ReadDctParams()`.
///
/// `num_params = u(4) + 1`, then a `3 x num_params` raster-order matrix, then
/// the first column of every row is multiplied by 64.
fn read_dct_params(reader: &mut BitReader<'_>) -> Result<[Vec<f64>; 3]> {
    let raw = trace_field!(reader, "dequant.num_params", reader.read_bits(4))?;
    let num_params = usize::try_from(raw).unwrap_or(0) + 1;
    let mut rows = read_matrix_3xn(reader, num_params, "dequant.dct_param")?;
    for row in &mut rows {
        if let Some(first) = row.first_mut() {
            *first *= PARAM_SCALE;
        }
    }
    Ok(rows)
}

fn scale_all(rows: &mut [Vec<f64>; 3]) {
    for row in rows {
        for v in row.iter_mut() {
            *v *= PARAM_SCALE;
        }
    }
}

/// **Flip point — is the RAW modular sub-bitstream byte-aligned?**
///
/// I.2.4's RAW arm reads `params.denominator` and then a modular
/// sub-bitstream, with no `ZeroPadToByte()` between them and none after. Annex
/// B always writes that primitive explicitly where it applies, and the two
/// other in-frame sub-bitstreams that have a field in front of them — G.2.2's
/// `extra_precision` and G.2.4's `nb_blocks` — are likewise unpadded and decode
/// real streams correctly.
///
/// * `true` (shipped): unaligned on both sides, so the next parameter set's
///   `u(3)` starts at the very next bit. Matches the E.4.1 ICC finding, where
///   the aligned reading failed on the first symbol.
/// * `false`: `ZeroPadToByte()` after the sub-bitstream.
///
/// RAW is the only place in the frame where a modular sub-bitstream is
/// *followed* by more fields in the same section, so it is the only construct
/// that can decide this.
///
/// **PROBED-CONFIRMED 2026-08-04.** `cjxl`'s JPEG-recompression path
/// (`cjxl in.jpg out.jxl`) emits `encoding_mode == RAW` for every `do_YCbCr`
/// stream this project has decoded — `bench_oriented_brg`/`grayscale_jpeg`
/// and their `_5` variants all use it. Under `true`, all four decode and grade
/// well inside their `test.json` budgets and match the pinned `djxl` to
/// within 1 of 255 per 8-bit sample; a wrong alignment reading would
/// desynchronise every field of `HfGlobal` after the sub-bitstream, not
/// produce a near-exact match. See
/// `docs/experiments/2026-08-03-vardct-flip-point-probe.md`'s 2026-08-04
/// addendum and `crates/jpxl-decode/tests/e2e_ycbcr.rs`.
pub const RAW_SUBBITSTREAM_IS_UNALIGNED: bool = true;

/// Decodes the 3-channel modular image of I.2.4's RAW arm, inline.
///
/// The clause asks for "a 3-channel image of the same shape as the required
/// quant matrix", so every channel is `cols` wide and `rows` tall per
/// Table I.4, at full resolution. Channels are taken in Table I.1's index
/// order — see [`RAW_MATRIX_CHANNEL_ORDER_IS_XYB`].
fn read_raw_matrix(
    reader: &mut BitReader<'_>,
    index: usize,
    ctx: &RawMatrixContext<'_>,
    guard: &mut AllocGuard,
) -> Result<[Vec<f32>; 3]> {
    let (rows, cols) = matrix_size(index)?;
    let mut options = ctx.options;
    options.stream_index = stream_index::dequant_table(ctx.geometry, as_u64(index))?;

    let spec = ChannelSpec::new(
        u32::try_from(cols)
            .map_err(|_| DecodeError::out_of_range("RAW matrix columns", "I.2.4", as_u64(cols)))?,
        u32::try_from(rows)
            .map_err(|_| DecodeError::out_of_range("RAW matrix rows", "I.2.4", as_u64(rows)))?,
    );
    let specs = [spec; 3];

    let image = decode_sub_bitstream_with(reader, &specs, &options, ctx.tree_source, guard)?;
    if !RAW_SUBBITSTREAM_IS_UNALIGNED {
        reader.zero_pad_to_byte()?;
    }

    let channels = image.into_channels();
    let planes: [Channel; 3] = channels
        .try_into()
        .map_err(|_| DecodeError::out_of_range("RAW channel count", "I.2.4", 3))?;

    let mut out = [Vec::new(), Vec::new(), Vec::new()];
    for (slot, channel) in out.iter_mut().zip(planes.iter()) {
        let samples = channel.samples();
        if samples.len() != rows * cols {
            return Err(DecodeError::out_of_range(
                "RAW channel size",
                "I.2.4",
                as_u64(samples.len()),
            ));
        }
        // The modular layer produces integers; the dequantization matrix is
        // floating point. i32 -> f32 is exact for every value a sane matrix
        // holds and monotone everywhere, and the positivity check below is what
        // actually guards the pipeline.
        #[allow(clippy::cast_precision_loss)]
        let values: Vec<f32> = samples.iter().map(|&v| v as f32).collect();
        *slot = values;
    }
    if RAW_MATRIX_CHANNEL_ORDER_IS_XYB {
        Ok(out)
    } else {
        let [first, second, b] = out;
        Ok([second, first, b])
    }
}

/// **Flip point — the channel order of the RAW modular sub-bitstream.**
///
/// I.2.4 says only "a 3-channel image". Everything else in the clause indexes
/// parameters by the channel of the weights matrix — "row `c` of `dct_params`",
/// "the weights matrix for channel `c`" — which is Table I.1's X, Y, B
/// numbering, and I.5.3 applies the matrix for channel `c` under that same
/// numbering. There is no reordering language anywhere in I.2.4.
///
/// * `true` (shipped): X, Y, B, i.e. stream position equals channel index.
/// * `false`: Y, X, B.
///
/// The `false` reading is not idle: G.2.2's `LfQuant`, which also says only
/// "three channels", turned out to be Y, X, B (see
/// [`LF_QUANT_CHANNEL_ORDER_IS_XYB`](crate::vardct::lf::LF_QUANT_CHANNEL_ORDER_IS_XYB)).
/// The difference is that I.4 states the Y, X, B order for HF *coefficients*
/// explicitly, and a RAW dequantization table is a parameter table, not
/// coefficients.
///
/// **PROBED-CONFIRMED 2026-08-04**, the same way as
/// [`RAW_SUBBITSTREAM_IS_UNALIGNED`]: the `do_YCbCr` corpus streams are the
/// first to use RAW, and a wrong channel order among matrices this
/// differently-shaped (X and B are typically near-identical chroma tables,
/// Y is the sharp luma one) would show up as a large, structured error, not
/// the sub-2e-6 peak these streams grade at.
pub const RAW_MATRIX_CHANNEL_ORDER_IS_XYB: bool = true;

/// Reads one parameter set of I.2.4.
fn read_params_set(
    reader: &mut BitReader<'_>,
    index: usize,
    raw_ctx: Option<&RawMatrixContext<'_>>,
    guard: &mut AllocGuard,
) -> Result<DequantParams> {
    let raw_mode = trace_field!(reader, "dequant.encoding_mode", reader.read_bits(3))?;
    let mode = EncodingMode::from_bits(raw_mode)?;
    if !mode.allows_index(index) {
        return Err(DecodeError::out_of_range(
            "encoding_mode",
            "I.2.4",
            u64::from(raw_mode),
        ));
    }

    // Every branch reads at most 3 * 16 F16 values plus a u(4); charge the
    // upper bound once so the metering happens before any allocation.
    guard.charge(3 * 16 * 8)?;

    match mode {
        EncodingMode::Library => Ok(default_params(index)?),
        EncodingMode::Hornuss => {
            let mut p = DequantParams::new(mode);
            let mut params = read_matrix_3xn(reader, 3, "dequant.hornuss_param")?;
            scale_all(&mut params);
            p.set_params(params);
            Ok(p)
        }
        EncodingMode::Dct2 => {
            let mut p = DequantParams::new(mode);
            let mut params = read_matrix_3xn(reader, 6, "dequant.dct2_param")?;
            scale_all(&mut params);
            p.set_params(params);
            Ok(p)
        }
        EncodingMode::Dct4 => {
            let mut p = DequantParams::new(mode);
            let mut params = read_matrix_3xn(reader, 2, "dequant.dct4_param")?;
            scale_all(&mut params);
            p.set_params(params);
            p.set_dct_params(read_dct_params(reader)?);
            Ok(p)
        }
        EncodingMode::Dct4x8 => {
            let mut p = DequantParams::new(mode);
            // Note: no `* 64` here. The clause scales `params` for Hornuss,
            // DCT2, DCT4 and AFV, and pointedly does not for DCT4x8.
            p.set_params(read_matrix_3xn(reader, 1, "dequant.dct4x8_param")?);
            p.set_dct_params(read_dct_params(reader)?);
            Ok(p)
        }
        EncodingMode::Afv => {
            let mut p = DequantParams::new(mode);
            let mut params = read_matrix_3xn(reader, 9, "dequant.afv_param")?;
            // Only the first six of the nine columns are scaled; the last three
            // are `Mult()` arguments, which are dimensionless.
            for row in &mut params {
                for v in row.iter_mut().take(6) {
                    *v *= PARAM_SCALE;
                }
            }
            p.set_params(params);
            p.set_dct_params(read_dct_params(reader)?);
            p.set_dct4x4_params(read_dct_params(reader)?);
            Ok(p)
        }
        EncodingMode::Dct => {
            let mut p = DequantParams::new(mode);
            p.set_dct_params(read_dct_params(reader)?);
            Ok(p)
        }
        EncodingMode::Raw => {
            let mut p = DequantParams::new(mode);
            p.set_denominator(trace_field!(
                reader,
                "dequant.denominator",
                read_f16_as_f32(reader)
            )?);
            // The sub-bitstream starts here. Refusing without decoding it would
            // leave the reader mid-stream, so this must be a hard stop, not a
            // "carry on and fail later".
            let Some(ctx) = raw_ctx else {
                return Err(DecodeError::Unsupported {
                    feature: "RAW dequantization matrices (no modular context was supplied, and \
                              skipping the inline sub-bitstream would desynchronise the rest of \
                              HfGlobal)",
                    clause: "18181-1 I.2.4",
                });
            };
            p.set_raw(read_raw_matrix(reader, index, ctx, guard)?);
            Ok(p)
        }
    }
}
/// What the inline RAW modular sub-bitstream of I.2.4 needs from the frame.
///
/// I.2.4's RAW arm reads `params.denominator` as an `F16()` and then, **at
/// that same bit position**, a 3-channel modular sub-bitstream. It is not a
/// separate section: G.3's section list gives `HfGlobal` exactly one section.
/// A decoder that skips the sub-bitstream therefore does not merely lack a
/// matrix, it desynchronises every field after it — which is why this context
/// has to be threaded into the parameter read rather than resolved afterwards.
#[derive(Debug, Clone, Copy)]
pub struct RawMatrixContext<'a> {
    /// Modular bounds and `bits_per_sample`. `stream_index` is overwritten per
    /// table with H.4.1's `dequant_table` value, so the caller's value for that
    /// field is ignored.
    pub options: ModularOptions,
    /// The frame's MA tree source (G.1.3), exactly as the `LfQuant` and
    /// HF-metadata sub-bitstreams receive it. `HfGlobal` is its own section, so
    /// a global tree is passed with `restart: true`.
    pub tree_source: TreeSource<'a>,
    /// The frame geometry, for H.4.1's `1 + 3 * num_lf_groups + index`.
    pub geometry: &'a FrameGeometry,
}

/// Reads the dequantization matrix parameters of I.2.4, refusing RAW.
///
/// Equivalent to [`read_dequant_matrices_with`] with no
/// [`RawMatrixContext`]: a stream that uses `encoding_mode` 7 is rejected with
/// [`DecodeError::Unsupported`] at exactly the bit where its sub-bitstream
/// would start. Prefer the `_with` form; this one exists so callers that have
/// no frame geometry to hand still compile.
///
/// # Errors
///
/// As [`read_dequant_matrices_with`].
pub fn read_dequant_matrices(
    reader: &mut BitReader<'_>,
    guard: &mut AllocGuard,
) -> Result<DequantMatrices> {
    read_dequant_matrices_with(reader, None, guard)
}

/// Reads the dequantization matrix parameters of I.2.4.
///
/// A leading `Bool()` selects the all-default case; otherwise the 17 sets are
/// read in ascending index order. `raw_ctx` supplies what
/// `encoding_mode == RAW` needs to decode its inline modular sub-bitstream;
/// pass `None` only if RAW cannot occur.
///
/// # Errors
///
/// * [`DecodeError::Bitstream`] on truncation.
/// * [`DecodeError::FieldOutOfRange`] if an `encoding_mode` is not valid for
///   its index per Table I.5.
/// * [`DecodeError::Modular`] if a RAW sub-bitstream is malformed.
/// * [`DecodeError::Unsupported`] if a set is RAW and `raw_ctx` is `None`.
pub fn read_dequant_matrices_with(
    reader: &mut BitReader<'_>,
    raw_ctx: Option<&RawMatrixContext<'_>>,
    guard: &mut AllocGuard,
) -> Result<DequantMatrices> {
    let all_default = trace_field!(reader, "dequant.all_default", read_bool(reader))?;
    if all_default {
        return Ok(DequantMatrices::all_default()?);
    }
    let mut entries = Vec::with_capacity(NUM_DEQUANT_MATRICES);
    for index in 0..NUM_DEQUANT_MATRICES {
        entries.push(read_params_set(reader, index, raw_ctx, guard)?);
    }
    Ok(DequantMatrices::from_entries(entries))
}

// ---------------------------------------------------------------------------
// Table G.4 — the first two rows of HfGlobal
// ---------------------------------------------------------------------------

/// The non-`HfPass` part of an `HfGlobal` bundle (Table G.4).
#[derive(Debug, Clone, PartialEq)]
pub struct HfGlobalParams {
    /// I.2.4 dequantization matrix parameters.
    pub matrices: DequantMatrices,
    /// I.2.6 `num_hf_presets`.
    pub num_hf_presets: u32,
}

/// Reads the dequantization matrices and `num_hf_presets` of Table G.4,
/// refusing RAW matrices.
///
/// Equivalent to [`read_hf_global_params_with`] with no
/// [`RawMatrixContext`]. Prefer the `_with` form; this signature is kept so
/// the existing `decode.rs` call site compiles unchanged.
///
/// # Errors
///
/// As [`read_hf_global_params_with`].
pub fn read_hf_global_params(
    reader: &mut BitReader<'_>,
    num_groups: u64,
    guard: &mut AllocGuard,
) -> Result<HfGlobalParams> {
    read_hf_global_params_with(reader, num_groups, None, guard)
}

/// Reads the dequantization matrices and `num_hf_presets` of Table G.4.
///
/// The caller continues with `hf_pass[num_passes]` (I.3) at the returned bit
/// position.
///
/// # Errors
///
/// As [`read_dequant_matrices_with`] and
/// [`read_num_hf_presets`](super::block_ctx::read_num_hf_presets).
pub fn read_hf_global_params_with(
    reader: &mut BitReader<'_>,
    num_groups: u64,
    raw_ctx: Option<&RawMatrixContext<'_>>,
    guard: &mut AllocGuard,
) -> Result<HfGlobalParams> {
    let matrices = read_dequant_matrices_with(reader, raw_ctx, guard)?;
    let num_hf_presets = read_num_hf_presets(reader, num_groups)?;
    Ok(HfGlobalParams {
        matrices,
        num_hf_presets,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testsupport::BitWriter;
    use jpxl_core::limits::Limits;
    use jpxl_core::varblock::TransformType;

    fn guard() -> AllocGuard {
        AllocGuard::new(&LIMITS)
    }

    static LIMITS: Limits = Limits::relaxed();

    #[test]
    fn all_default_bundle_is_one_bit() {
        let mut w = BitWriter::new();
        w.bool(true);
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let m = read_dequant_matrices(&mut r, &mut guard()).expect("valid");
        assert_eq!(r.total_bits_read(), 1);
        assert_eq!(m, DequantMatrices::all_default().expect("defaults"));
    }

    #[test]
    fn seventeen_library_modes_are_seventeen_three_bit_fields() {
        // 1 + 17 * 3 = 52 bits. Proves the set count and that encoding_mode is
        // u(3), and that Library resolves to the same thing as all_default.
        let mut w = BitWriter::new();
        w.bool(false);
        for _ in 0..NUM_DEQUANT_MATRICES {
            w.u(3, 0);
        }
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let m = read_dequant_matrices(&mut r, &mut guard()).expect("valid");
        assert_eq!(r.total_bits_read(), 1 + 17 * 3);
        assert_eq!(m, DequantMatrices::all_default().expect("defaults"));
    }

    #[test]
    fn dct_mode_reads_num_params_then_three_rows() {
        // Index 0 with mode DCT and num_params = 2: 3 + 4 + 3 * 2 * 16 bits.
        // Proves ReadDctParams's field order and the `* 64` on column 0.
        let mut w = BitWriter::new();
        w.bool(false);
        w.u(3, 6); // DCT
        w.u(4, 1); // num_params = 2
        for _ in 0..6 {
            w.f16_bits(0x3C00); // 1.0
        }
        for _ in 1..NUM_DEQUANT_MATRICES {
            w.u(3, 0); // Library for the rest
        }
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let m = read_dequant_matrices(&mut r, &mut guard()).expect("valid");
        assert_eq!(r.total_bits_read(), 1 + 3 + 4 + 6 * 16 + 16 * 3);
        let p = m.params(0).expect("index 0");
        assert_eq!(p.mode(), EncodingMode::Dct);
        assert_eq!(p.dct_params_row(0), vec![64.0, 1.0]);
        assert_eq!(p.dct_params_row(2), vec![64.0, 1.0]);
    }

    #[test]
    fn hornuss_mode_scales_every_parameter() {
        let mut w = BitWriter::new();
        w.bool(false);
        w.u(3, 0); // index 0: Library
        w.u(3, 1); // index 1: Hornuss
        for _ in 0..9 {
            w.f16_bits(0x3C00); // 1.0
        }
        for _ in 2..NUM_DEQUANT_MATRICES {
            w.u(3, 0);
        }
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let m = read_dequant_matrices(&mut r, &mut guard()).expect("valid");
        assert_eq!(r.total_bits_read(), 1 + 17 * 3 + 9 * 16);
        let p = m.params(1).expect("index 1");
        assert_eq!(p.params_row(0), vec![64.0, 64.0, 64.0]);
    }

    #[test]
    fn dct4x8_mode_does_not_scale_its_params() {
        // The one mode whose `params` the clause does not multiply by 64.
        let mut w = BitWriter::new();
        w.bool(false);
        for _ in 0..9 {
            w.u(3, 0); // indices 0..=8: Library
        }
        w.u(3, 4); // index 9: DCT4x8
        w.f16_bits(0x3C00).f16_bits(0x3C00).f16_bits(0x3C00); // params 3x1
        w.u(4, 0); // num_params = 1
        for _ in 0..3 {
            w.f16_bits(0x3C00);
        }
        for _ in 10..NUM_DEQUANT_MATRICES {
            w.u(3, 0);
        }
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let m = read_dequant_matrices(&mut r, &mut guard()).expect("valid");
        let p = m.params(9).expect("index 9");
        assert_eq!(p.mode(), EncodingMode::Dct4x8);
        assert_eq!(p.params_row(0), vec![1.0]);
        // dct_params column 0 *is* scaled.
        assert_eq!(p.dct_params_row(0), vec![64.0]);
    }

    #[test]
    fn afv_mode_scales_only_the_first_six_params() {
        let mut w = BitWriter::new();
        w.bool(false);
        for _ in 0..10 {
            w.u(3, 0);
        }
        w.u(3, 5); // index 10: AFV
        for _ in 0..27 {
            w.f16_bits(0x3C00); // params 3x9, all 1.0
        }
        w.u(4, 0);
        for _ in 0..3 {
            w.f16_bits(0x3C00); // dct_params
        }
        w.u(4, 0);
        for _ in 0..3 {
            w.f16_bits(0x3C00); // dct4x4_params
        }
        for _ in 11..NUM_DEQUANT_MATRICES {
            w.u(3, 0);
        }
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let m = read_dequant_matrices(&mut r, &mut guard()).expect("valid");
        let p = m.params(10).expect("index 10");
        assert_eq!(
            p.params_row(0),
            vec![64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 1.0, 1.0, 1.0]
        );
    }

    #[test]
    fn a_mode_invalid_for_its_index_is_rejected() {
        // Table I.5: Hornuss is not valid for index 4 (DCT16x16).
        let mut w = BitWriter::new();
        w.bool(false);
        for _ in 0..4 {
            w.u(3, 0);
        }
        w.u(3, 1); // Hornuss at index 4
        for _ in 0..9 {
            w.f16_bits(0x3C00);
        }
        let data = w.finish_padded(4);
        let mut r = BitReader::new(&data);
        let err = read_dequant_matrices(&mut r, &mut guard()).expect_err("invalid mode");
        assert!(err.to_string().contains("encoding_mode"), "{err}");
    }

    #[test]
    fn truncated_bundle_errors() {
        let mut w = BitWriter::new();
        w.bool(false).u(3, 6).u(4, 15);
        let data = w.finish();
        let mut r = BitReader::new(&data);
        assert!(read_dequant_matrices(&mut r, &mut guard()).is_err());
    }

    #[test]
    fn degenerate_parameters_are_rejected_not_returned_as_nan() {
        // A DCT row whose base is zero makes bands[0] zero, and the reciprocal
        // would be infinite. The clause says no such value occurs, so this must
        // be an error rather than an infinity propagating into the samples.
        let mut w = BitWriter::new();
        w.bool(false);
        w.u(3, 6); // DCT at index 0
        w.u(4, 0); // num_params = 1
        for _ in 0..3 {
            w.f16_bits(0x0000); // 0.0
        }
        for _ in 1..NUM_DEQUANT_MATRICES {
            w.u(3, 0);
        }
        let data = w.finish_padded(2);
        let mut r = BitReader::new(&data);
        let m = read_dequant_matrices(&mut r, &mut guard()).expect("parses");
        assert!(m.matrix(0, 0).is_err(), "a zero base must be rejected");
    }

    // ------------------------------------------------------------------
    // RAW mode (encoding_mode 7)
    // ------------------------------------------------------------------
    //
    // The modular-sub-bitstream writer below mirrors the one `vardct::lf`
    // uses for G.2.2, field for field: `ModularHeader` with no transforms
    // (Table H.1), a single-leaf MA tree (H.4.2) whose predictor is Zero, and
    // a four-symbol fixed-width prefix bundle (C.2). It is an independent
    // encoder written from the Annex H tables, which is what makes decoding
    // its output evidence rather than a tautology; the same helper shape is
    // cross-validated against real `cjxl` streams by `vardct::lf`'s and
    // `vardct::hf_meta`'s own fixtures.

    /// Writes `value` most-significant bit first, as a prefix code does.
    fn msb(w: &mut BitWriter, value: u32, n: u32) {
        for i in (0..n).rev() {
            w.bit((value >> i) & 1 == 1);
        }
    }

    /// The smallest prefix-coded distribution bundle (C.2) that carries a
    /// four-symbol fixed-width alphabet, one cluster for every context.
    fn write_prefix_bundle(w: &mut BitWriter, num_dist: usize) {
        w.bool(false); // lz77.enabled
        if num_dist > 1 {
            w.bool(true); // simple clustering
            w.u(2, 0); // nbits = 0 -> every context maps to cluster 0
        }
        w.bool(true); // use_prefix_code
        w.u(4, 15); // HybridUintConfig split_exponent = log_alphabet_size
        w.bit(true); // alphabet_size = 1 + (1 << n) + u(n)
        w.u(4, 1);
        w.u(1, 1);
        w.u(2, 1); // RFC 7932 3.4 simple code, selector 1
        w.u(2, 3); // nsym - 1 = 3
        for symbol in 0..4u32 {
            w.u(2, symbol);
        }
        w.bit(false); // balanced-pattern bit for nsym == 4
    }

    /// `ModularHeader` (Table H.1) with no transforms, then H.4.2's
    /// single-leaf tree with the Zero predictor.
    fn write_modular_prelude(w: &mut BitWriter) {
        w.bool(false); // use_global_tree
        w.bool(true); // wp_params: default_wp
        w.u(2, 0); // nb_transforms: U32 selector 0 -> constant 0
        write_prefix_bundle(w, 6);
        for token in [0u32, 0, 0, 0, 0] {
            msb(w, token, 2);
        }
        write_prefix_bundle(w, 1);
    }

    /// A whole RAW sub-bitstream for a `rows x cols` matrix whose every
    /// sample is 1.
    ///
    /// The four-symbol test alphabet carries `UnpackSigned` of 0..=3, i.e.
    /// `{0, -1, 1, -2}`, and I.2.4 forbids a non-positive dequantization
    /// value, so 1 (token 2) is the only sample a minimal alphabet can use.
    fn write_raw_ones(w: &mut BitWriter, rows: usize, cols: usize) {
        write_modular_prelude(w);
        for _ in 0..(3 * rows * cols) {
            msb(w, 2, 2);
        }
    }

    fn eight_by_eight_geometry() -> crate::frame::FrameGeometry {
        let limits = Limits::relaxed();
        let mut g = AllocGuard::new(&limits);
        crate::frame::FrameGeometry::derive(
            crate::frame::geometry::GroupLayout {
                width: 64,
                height: 64,
                upsampling: 1,
                lf_level: 0,
                group_dim: jpxl_core::geometry::GroupDim::D256,
                num_passes: 1,
            },
            &limits,
            &mut g,
        )
        .expect("64x64 is well within limits")
    }

    #[test]
    fn raw_matrices_decode_inline_and_keep_the_bundle_in_sync() {
        // What this proves, and it is the only proof available because no
        // encoder emits RAW:
        //
        // 1. The 3-channel modular sub-bitstream is consumed *at the bit
        //    position right after `params.denominator`* — I.2.4's RAW arm is
        //    sequential code, and G.3's section list gives HfGlobal one
        //    section, so there is nowhere else it could be.
        // 2. It is consumed exactly, twice over, at two different parameter
        //    indices: index 2's Hornuss parameters and the trailing
        //    `num_hf_presets` are read at the right bits afterwards. A
        //    sub-bitstream over- or under-read by a single bit corrupts
        //    index 2's mode or its nine F16 values or the preset count.
        // 3. The dequantization matrix is the decoded planes times
        //    `params.denominator`, per channel and per index — *not* the
        //    reciprocal, which is the rule for every other mode.
        // 4. The matrix shape is Table I.4's, for both a square and an oblong
        //    row.
        let mut w = BitWriter::new();
        w.bool(false); // not all_default

        // Index 0: RAW, 8x8, denominator 2.0.
        w.u(3, 7);
        w.f16_bits(0x4000);
        write_raw_ones(&mut w, 8, 8);

        // Index 1: RAW, 8x8, denominator 4.0 — a different value so a matrix
        // built from the wrong table's denominator is visible.
        w.u(3, 7);
        w.f16_bits(0x4400);
        write_raw_ones(&mut w, 8, 8);

        // Index 2: Hornuss with nine known parameters — the sync canary.
        w.u(3, 1);
        for _ in 0..9 {
            w.f16_bits(0x3C00); // 1.0, scaled to 64.0 by the * 64 rule
        }

        // Index 3..=5: Library. Index 6 is RAW again, and 8x16 rather than
        // 8x8, so the shape really comes from Table I.4.
        for _ in 3..6 {
            w.u(3, 0);
        }
        w.u(3, 7);
        w.f16_bits(0x3C00); // denominator = 1.0
        write_raw_ones(&mut w, 8, 16);

        for _ in 7..NUM_DEQUANT_MATRICES {
            w.u(3, 0);
        }
        // I.2.6 with num_groups = 8: u(3) + 1.
        w.u(3, 5);

        let data = w.finish_padded(2);
        let mut r = BitReader::new(&data);
        let geometry = eight_by_eight_geometry();
        let ctx = RawMatrixContext {
            options: ModularOptions::level10(),
            tree_source: TreeSource::Local,
            geometry: &geometry,
        };
        let hf = read_hf_global_params_with(&mut r, 8, Some(&ctx), &mut guard())
            .expect("well-formed HfGlobal with three RAW tables");

        // (2) the canary and the trailing field.
        assert_eq!(hf.num_hf_presets, 6);
        let hornuss = hf.matrices.params(2).expect("index 2");
        assert_eq!(hornuss.mode(), EncodingMode::Hornuss);
        assert_eq!(hornuss.params_row(0), vec![64.0, 64.0, 64.0]);
        assert_eq!(hornuss.params_row(2), vec![64.0, 64.0, 64.0]);

        // (3) and (4).
        for channel in 0..3 {
            let m0 = hf.matrices.matrix(0, channel).expect("index 0");
            assert_eq!((m0.rows(), m0.cols()), (8, 8));
            assert!(
                m0.as_slice().iter().all(|v| *v == 2.0),
                "index 0 is 1 * denominator 2.0 everywhere"
            );

            let m1 = hf.matrices.matrix(1, channel).expect("index 1");
            assert!(
                m1.as_slice().iter().all(|v| *v == 4.0),
                "index 1 carries its own denominator"
            );

            let m6 = hf.matrices.matrix(6, channel).expect("index 6");
            assert_eq!((m6.rows(), m6.cols()), (8, 16));
            assert_eq!(m6.as_slice().len(), 128);
            assert!(m6.as_slice().iter().all(|v| *v == 1.0));
        }

        // The Table I.4 shape is the transform's coefficient shape, so a RAW
        // matrix slots into the same pipeline as a computed one.
        assert_eq!(
            (
                TransformType::Dct16x8.coeff_rows(),
                TransformType::Dct16x8.coeff_cols()
            ),
            (8, 16)
        );

        // Nothing is left pending: the deferred-resolution path is gone.
        assert!(hf.matrices.raw_requests().is_empty());
    }

    #[test]
    fn a_raw_sub_bitstream_shifted_by_one_bit_is_caught() {
        // The sync claim above is only worth something if a desynchronised
        // stream actually fails. Truncating the last RAW symbol by one bit
        // leaves the sub-bitstream short, and the rest of the bundle is then
        // read from the wrong place.
        let mut w = BitWriter::new();
        w.bool(false);
        w.u(3, 7);
        w.f16_bits(0x4000);
        write_modular_prelude(&mut w);
        // One symbol short of the 3 * 64 an 8x8 matrix needs.
        for _ in 0..(3 * 64 - 1) {
            msb(&mut w, 2, 2);
        }
        w.u(3, 1); // what the stream *meant* to be index 1's mode
        for _ in 0..9 {
            w.f16_bits(0x3C00);
        }
        for _ in 2..NUM_DEQUANT_MATRICES {
            w.u(3, 0);
        }
        w.u(3, 5);

        let data = w.finish_padded(2);
        let mut r = BitReader::new(&data);
        let geometry = eight_by_eight_geometry();
        let ctx = RawMatrixContext {
            options: ModularOptions::level10(),
            tree_source: TreeSource::Local,
            geometry: &geometry,
        };
        let result = read_hf_global_params_with(&mut r, 8, Some(&ctx), &mut guard());
        // Either the shifted parse errors outright or it lands on different
        // values; what it must not do is silently agree with the aligned one.
        match result {
            Err(_) => {}
            Ok(hf) => {
                let hornuss = hf.matrices.params(1).expect("index 1");
                assert!(
                    hornuss.mode() != EncodingMode::Hornuss
                        || hornuss.params_row(0) != vec![64.0, 64.0, 64.0]
                        || hf.num_hf_presets != 6,
                    "a one-symbol-short RAW stream decoded as if it were correct"
                );
            }
        }
    }

    #[test]
    fn raw_without_a_modular_context_is_refused_at_the_sub_bitstream() {
        // The old behaviour returned a `RawMatrixRequest` and let the caller
        // carry on reading from a bit position that was already wrong. Now the
        // refusal happens where the sub-bitstream starts.
        let mut w = BitWriter::new();
        w.bool(false);
        w.u(3, 7); // RAW at index 0
        w.f16_bits(0x4000); // denominator = 2.0
        write_raw_ones(&mut w, 8, 8);
        for _ in 1..NUM_DEQUANT_MATRICES {
            w.u(3, 0);
        }
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let err = read_dequant_matrices(&mut r, &mut guard()).expect_err("no context supplied");
        assert!(matches!(err, DecodeError::Unsupported { .. }), "{err}");
        // Refused *after* the denominator and *before* anything else, so the
        // caller cannot mistake the reader for a usable position.
        assert_eq!(r.total_bits_read(), 1 + 3 + 16);
    }

    #[test]
    fn a_raw_matrix_with_a_non_positive_sample_is_rejected() {
        // I.2.4 states that no dequantization value is non-positive or
        // infinite. Token 0 is `UnpackSigned(0) == 0`, so a plane of zeros is a
        // well-formed modular sub-bitstream carrying an ill-formed matrix: the
        // parse must succeed and the matrix must not.
        let mut w = BitWriter::new();
        w.bool(false);
        w.u(3, 7);
        w.f16_bits(0x3C00); // denominator = 1.0
        write_modular_prelude(&mut w);
        for _ in 0..(3 * 64) {
            msb(&mut w, 0, 2); // every sample 0
        }
        for _ in 1..NUM_DEQUANT_MATRICES {
            w.u(3, 0);
        }
        let data = w.finish_padded(2);
        let mut r = BitReader::new(&data);
        let geometry = eight_by_eight_geometry();
        let ctx = RawMatrixContext {
            options: ModularOptions::level10(),
            tree_source: TreeSource::Local,
            geometry: &geometry,
        };
        let m = read_dequant_matrices_with(&mut r, Some(&ctx), &mut guard())
            .expect("the sub-bitstream itself is well formed");
        assert!(
            m.matrix(0, 0).is_err(),
            "a zero entry must be rejected, not passed to I.5.3"
        );
    }

    #[test]
    fn the_raw_stream_index_follows_h41() {
        // Property 1 feeds the MA tree, so a wrong stream index decodes a
        // different image from the same bits whenever the tree branches on it.
        // The formula lives in frame::stream_index; this pins that this module
        // uses it, per index.
        let geometry = eight_by_eight_geometry();
        assert_eq!(
            u64::from(stream_index::dequant_table(&geometry, 0).expect("index 0")),
            1 + 3 * geometry.num_lf_groups()
        );
        assert_eq!(
            stream_index::dequant_table(&geometry, 16).expect("index 16")
                - stream_index::dequant_table(&geometry, 0).expect("index 0"),
            16
        );
        assert!(stream_index::dequant_table(&geometry, 17).is_err());
    }

    #[test]
    fn hf_global_params_reads_both_rows_of_table_g4() {
        // 1 bit of all_default plus a zero-width num_hf_presets for a
        // single-group frame. Proves the Table G.4 order.
        let mut w = BitWriter::new();
        w.bool(true);
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let hf = read_hf_global_params(&mut r, 1, &mut guard()).expect("valid");
        assert_eq!(r.total_bits_read(), 1);
        assert_eq!(hf.num_hf_presets, 1);
        assert_eq!(hf.matrices, DequantMatrices::all_default().expect("d"));
    }
}
