//! LF coefficients: the `LfQuant` modular sub-bitstream (18181-1 G.2.2).
//!
//! G.2.2's own text:
//!
//! ```text
//! The decoder first reads extra_precision as a u(2). Next, the decoder reads
//! a Modular sub-bitstream as described in Annex H, to obtain the quantized
//! LF coefficients LfQuant, which consists of three channels with
//! ceil(height / 8) rows and ceil(width / 8) columns, where the number of
//! rows and columns is optionally right-shifted by one according to
//! frame_header.jpeg_upsampling. Finally the LF is dequantized as specified
//! in I.5.2.
//! ```
//!
//! `width`/`height` here are the current LF group's dimensions (G.2.1
//! General), and the NOTE at G.1 records that they are always a 1:8
//! downsampled view of the frame regardless of the varblocks actually placed
//! in the group.
//!
//! # Channel list and order
//!
//! Three channels, [`LF_QUANT_CHANNEL_ORDER_IS_XYB`] fixes their order as
//! X, Y, B. G.2.2 does not itself state an order — it just says "three
//! channels" — but I.5.1/I.5.2 always name the trio `qX, qY, qB` in that
//! order (never `qY, qX, qB`, the order I.4 uses for *HF* coefficients), and
//! Table I.1's channel numbering is `X = 0, Y = 1, B = 2`. That is the
//! reading taken here; it is a candidate flip point since G.2.2 never spells
//! the order out directly.
//!
//! # Scope: parse only
//!
//! This module produces the **quantized integer** `LfQuant` planes and
//! nothing more. I.5.2's dequantization (the `mxDC`/`myDC`/`mBDC`
//! multipliers, `extra_precision` shift, chroma-from-luma, and the adaptive
//! smoothing pass) needs 8B's parameter bundles and is a later seam — see
//! [`LfQuantPlanes`] for exactly where it picks up.

use jpxl_bitstream::{BitReader, trace_field};
use jpxl_core::limits::AllocGuard;

use crate::error::Result;
use crate::modular::{Channel, ChannelSpec, ModularOptions, TreeSource, decode_sub_bitstream_with};

/// **Flip point — `LfQuant`'s channel order.**
///
/// * `true` (shipped): the three `LfQuant` channels are read in the order
///   X, Y, B, matching Table I.1's channel numbering and I.5.1/I.5.2's
///   `qX, qY, qB` naming.
/// * `false`: some other order (e.g. the Y, X, B order I.4 uses for HF
///   coefficients).
///
/// Flipping this constant is the whole change if an oracle probe disagrees;
/// see `docs/experiments/` for the writeup once one exists.
pub const LF_QUANT_CHANNEL_ORDER_IS_XYB: bool = true;

/// The three `LfQuant` channels of G.2.2, still quantized integers.
///
/// Named `x`/`y`/`b` rather than indexed, so a caller cannot accidentally
/// treat this as Y, X, B (the *different* order I.4 uses for HF
/// coefficients) — see [`LF_QUANT_CHANNEL_ORDER_IS_XYB`].
#[derive(Debug, Clone)]
pub struct LfQuantPlanes {
    /// `extra_precision`, G.2.2's leading `u(2)`. I.5.2 divides every
    /// dequantized value by `1 << extra_precision`; this parse stage only
    /// carries the field, it does not apply it.
    pub extra_precision: u8,
    /// Quantized `qX`.
    pub x: Channel,
    /// Quantized `qY`.
    pub y: Channel,
    /// Quantized `qB`.
    pub b: Channel,
    // SEAM(8D-dequant): I.5.2 turns (extra_precision, x, y, b) into
    // (dX, dY, dB) via the per-channel multipliers `mxDC`/`myDC`/`mBDC`
    // (G.1.2, 8B), then applies I.6 chroma-from-luma and, unless
    // kSkipAdaptiveLFSmoothing, the 3x3 adaptive smoothing pass. None of
    // that runs here.
}

/// The horizontal/vertical subsampling factors G.2.2 borrows from
/// `frame_header.jpeg_upsampling` (F.2): `0` denotes `{1, 1}`, `1` denotes
/// `{2, 2}`, `2` denotes `{2, 1}`, `3` denotes `{1, 2}`.
///
/// Any other code is unreachable — the field is a `u(2)` everywhere it is
/// read — so it is treated as "no subsampling" rather than panicking.
const fn jpeg_upsampling_factors(code: u32) -> (u32, u32) {
    match code {
        1 => (2, 2),
        2 => (2, 1),
        3 => (1, 2),
        _ => (1, 1),
    }
}

/// Builds the `ChannelSpec` for `LfQuant` channel `c` (`X = 0, Y = 1, B = 2`)
/// of an LF group `group_width x group_height` samples, per G.2.2.
fn lf_quant_channel_spec(group_width: u32, group_height: u32, upsampling_code: u32) -> ChannelSpec {
    let rows = group_height.div_ceil(8);
    let cols = group_width.div_ceil(8);
    let (h_factor, v_factor) = jpeg_upsampling_factors(upsampling_code);
    ChannelSpec::new(cols.div_ceil(h_factor), rows.div_ceil(v_factor))
}

/// Reads G.2.2's `extra_precision` field and the `LfQuant` sub-bitstream.
///
/// `group_width`/`group_height` are the current LF group's pixel dimensions
/// (the `Rect` from [`crate::frame::FrameGeometry::lf_group_rect`]).
/// `jpeg_upsampling` is `frame_header.jpeg_upsampling` (F.2), read in Table
/// I.1's `X, Y, B` channel order per [`LF_QUANT_CHANNEL_ORDER_IS_XYB`].
///
/// `options.stream_index` must already be set by the caller — G.2.2's stream
/// index is `crate::frame::stream_index::lf_coefficients`.
///
/// # Errors
///
/// Any [`crate::error::DecodeError`] the sub-bitstream decode reports
/// (malformed header, entropy failure, or a limit rejection).
pub fn read_lf_quant(
    reader: &mut BitReader<'_>,
    group_width: u32,
    group_height: u32,
    jpeg_upsampling: [u32; 3],
    options: &ModularOptions,
    tree_source: TreeSource<'_>,
    guard: &mut AllocGuard,
) -> Result<LfQuantPlanes> {
    let extra_precision = trace_field!(reader, "lf_coeff.extra_precision", reader.read_bits(2))?;
    let extra_precision = u8::try_from(extra_precision).unwrap_or(0);

    let specs = [
        lf_quant_channel_spec(group_width, group_height, jpeg_upsampling[0]),
        lf_quant_channel_spec(group_width, group_height, jpeg_upsampling[1]),
        lf_quant_channel_spec(group_width, group_height, jpeg_upsampling[2]),
    ];

    let image = decode_sub_bitstream_with(reader, &specs, options, tree_source, guard)?;
    let channels = image.into_channels();
    let [x, y, b]: [Channel; 3] = channels.try_into().map_err(|_| {
        crate::error::DecodeError::out_of_range("LfQuant channel count", "G.2.2", 3)
    })?;

    Ok(LfQuantPlanes {
        extra_precision,
        x,
        y,
        b,
    })
}

#[cfg(test)]
#[allow(
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    reason = "the hand-built bit writer indexes its own byte buffer, whose \
              length it just computed, and truncates only test-chosen small \
              constants; a panic here is a failing test"
)]
mod tests {
    use super::*;
    use jpxl_core::limits::Limits;

    // -----------------------------------------------------------------
    // Minimal hand-built-bitstream helpers, mirroring
    // `tests/modular_common/mod.rs`'s style (that file is owned by
    // slice 5's own test files; this module hand-rolls its own copy
    // rather than reaching outside its file ownership).
    // -----------------------------------------------------------------

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

        fn code_msb_first(&mut self, value: u32, n: u32) {
            for i in (0..n).rev() {
                self.bits.push((value >> i) & 1 == 1);
            }
        }

        fn finish(&self) -> Vec<u8> {
            let mut out = vec![0u8; self.bits.len().div_ceil(8)];
            for (i, &bit) in self.bits.iter().enumerate() {
                if bit {
                    out[i / 8] |= 1 << (i % 8);
                }
            }
            out
        }
    }

    /// `ModularHeader` (Table H.1) with no transforms.
    fn write_header_no_transforms(w: &mut BitWriter) {
        w.bit(false); // use_global_tree
        w.bit(true); // wp_params: default_wp
        w.u(0, 2); // nb_transforms: U32 selector 0 -> constant 0
    }

    /// H.4.2's six-context tree stream for a single leaf: property test
    /// token 0 (immediate leaf), then predictor/offset/mul_log/mul_bits, all
    /// through a shared 4-symbol fixed-width code (`[2,2,2,2]`).
    fn write_single_leaf_tree(w: &mut BitWriter, predictor: u32) {
        write_prefix_bundle(w, 6);
        for token in [0u32, predictor, 0, 0, 0] {
            w.code_msb_first(token, 2);
        }
    }

    /// The smallest prefix-coded distribution bundle (C.2) that can carry a
    /// 4-symbol fixed-width alphabet, one cluster for every context.
    fn write_prefix_bundle(w: &mut BitWriter, num_dist: usize) {
        w.bit(false); // lz77.enabled
        if num_dist > 1 {
            w.bit(true); // simple clustering
            w.u(0, 2); // nbits = 0 -> every context maps to cluster 0
        }
        w.bit(true); // use_prefix_code
        // HybridUintConfig: split_exponent = 15 (log_alphabet_size), so no
        // msb/lsb-in-token fields follow.
        w.u(15, 4);
        // alphabet_size = 1 + (1 << n) + u(n), n = 1, extra = 1 -> size 4.
        w.bit(true);
        w.u(1, 4);
        w.u(1, 1);
        // RFC 7932 3.4 simple code: selector 1, nsym - 1 = 3, then symbols
        // 0..4 each in 2 bits, then the balanced-pattern bit for nsym == 4.
        w.u(1, 2);
        w.u(3, 2);
        for symbol in 0..4u32 {
            w.u(symbol, 2);
        }
        w.bit(false);
    }

    /// `UnpackSigned` inverse, restricted to the four values the 2-bit test
    /// alphabet below can carry: token 0 -> 0, 1 -> -1, 2 -> 1, 3 -> -2.
    fn pack_signed(v: i32) -> u32 {
        match v {
            0 => 0,
            -1 => 1,
            1 => 2,
            -2 => 3,
            other => panic!("{other} has no token in the 4-symbol test alphabet"),
        }
    }

    #[test]
    fn jpeg_upsampling_codes_match_f2() {
        assert_eq!(jpeg_upsampling_factors(0), (1, 1));
        assert_eq!(jpeg_upsampling_factors(1), (2, 2));
        assert_eq!(jpeg_upsampling_factors(2), (2, 1));
        assert_eq!(jpeg_upsampling_factors(3), (1, 2));
    }

    #[test]
    fn channel_spec_is_1_8_downsampled_with_no_subsampling() {
        // A 20x11 LF group: ceil(20/8) = 3 columns, ceil(11/8) = 2 rows.
        let spec = lf_quant_channel_spec(20, 11, 0);
        assert_eq!((spec.width, spec.height), (3, 2));
    }

    #[test]
    fn channel_spec_applies_the_extra_right_shift() {
        // Same 20x11 group, code 1 (2x2): both axes divide by 2, rounding up.
        let spec = lf_quant_channel_spec(20, 11, 1);
        assert_eq!((spec.width, spec.height), (2, 1));
        // Code 2 (2x1): only columns shift.
        let spec = lf_quant_channel_spec(20, 11, 2);
        assert_eq!((spec.width, spec.height), (2, 2));
        // Code 3 (1x2): only rows shift.
        let spec = lf_quant_channel_spec(20, 11, 3);
        assert_eq!((spec.width, spec.height), (3, 1));
    }

    #[test]
    fn a_hand_built_lf_quant_stream_decodes_to_the_expected_planes() {
        // A 16x8 LF group (no subsampling): each channel is
        // ceil(16/8)=2 columns x ceil(8/8)=1 row, so two samples per channel,
        // six samples total, decoded X then Y then B (Zero predictor, so each
        // sample is just UnpackSigned(token)).
        let mut w = BitWriter::default();
        w.u(0, 2); // extra_precision = 0
        write_header_no_transforms(&mut w);
        write_single_leaf_tree(&mut w, 0); // predictor 0 = Zero
        write_prefix_bundle(&mut w, 1);
        // X: [1, -1], Y: [-1, 1], B: [-2, 0] — a distinct pattern per
        // channel, drawn from the 4-symbol test alphabet's range.
        for v in [1i32, -1, -1, 1, -2, 0] {
            w.code_msb_first(pack_signed(v), 2);
        }
        let data = w.finish();

        let mut reader = BitReader::new(&data);
        let limits = Limits::relaxed();
        let mut guard = AllocGuard::new(&limits);
        let planes = read_lf_quant(
            &mut reader,
            16,
            8,
            [0, 0, 0],
            &ModularOptions::level10(),
            TreeSource::Local,
            &mut guard,
        )
        .expect("well-formed LfQuant stream");

        assert_eq!(planes.extra_precision, 0);
        assert_eq!(planes.x.samples(), &[1, -1]);
        assert_eq!(planes.y.samples(), &[-1, 1]);
        assert_eq!(planes.b.samples(), &[-2, 0]);
    }

    #[test]
    fn extra_precision_is_read_before_the_sub_bitstream() {
        // An 8x8 LF group with no subsampling: each channel is
        // ceil(8/8) = 1 column x 1 row, one sample apiece. If
        // `extra_precision` were *not* consumed before the `ModularHeader`,
        // every later field would be shifted by two bits and either the
        // header would fail to parse as a well-formed `ModularHeader` or the
        // decoded samples would not match, so this is a real ordering proof,
        // not just a field-value echo.
        let mut w = BitWriter::default();
        w.u(3, 2); // extra_precision = 3, the max u(2) value
        write_header_no_transforms(&mut w);
        write_single_leaf_tree(&mut w, 0); // predictor 0 = Zero
        write_prefix_bundle(&mut w, 1);
        for v in [1i32, -1, -2] {
            w.code_msb_first(pack_signed(v), 2);
        }
        let data = w.finish();

        let mut reader = BitReader::new(&data);
        let limits = Limits::relaxed();
        let mut guard = AllocGuard::new(&limits);
        let planes = read_lf_quant(
            &mut reader,
            8,
            8,
            [0, 0, 0],
            &ModularOptions::level10(),
            TreeSource::Local,
            &mut guard,
        )
        .expect("well-formed LfQuant stream");

        assert_eq!(planes.extra_precision, 3);
        assert_eq!(planes.x.samples(), &[1]);
        assert_eq!(planes.y.samples(), &[-1]);
        assert_eq!(planes.b.samples(), &[-2]);
    }

    #[test]
    fn truncated_input_is_rejected_not_panicking() {
        let data: [u8; 1] = [0];
        let mut reader = BitReader::new(&data);
        let limits = Limits::relaxed();
        let mut guard = AllocGuard::new(&limits);
        let result = read_lf_quant(
            &mut reader,
            2048,
            2048,
            [0, 0, 0],
            &ModularOptions::level10(),
            TreeSource::Local,
            &mut guard,
        );
        assert!(result.is_err());
    }
}
