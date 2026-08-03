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
//! Y, X, B — the same order I.4 states explicitly for *HF* coefficients.
//! G.2.2 does not itself state an order for *LF* coefficients — it just says
//! "three channels" — and I.5.1/I.5.2's prose always names the trio
//! `qX, qY, qB`, which reads as X-first. **That textual reading is not what
//! is shipped.** Decoding real fixtures (50, 51: greyscale VarDCT content)
//! under both readings and comparing the three decoded planes' variance
//! shows the first-decoded channel alone carries the source's structure
//! (variance in the thousands) while the other two are *exactly* flat
//! (variance `0.0`) — consistent with a luma-first decode order for
//! genuinely achromatic content and inconsistent with an X-first one (X
//! being one of the two chroma-like channels, so it should be the flat one,
//! not the structured one). See
//! `docs/experiments/2026-08-03-lf-quant-channel-order-fixture-evidence.md`.
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

use crate::error::{DecodeError, Result};
use crate::modular::{Channel, ChannelSpec, ModularOptions, TreeSource, decode_sub_bitstream_with};

use super::cfl::{self, CflFactors};
use super::quantizer::{LfChannelCorrelation, LfDequantMultipliers};

/// **Flip point — `LfQuant`'s channel order.**
///
/// * `true`: the three `LfQuant` channels are read in the order X, Y, B —
///   matching Table I.1's channel numbering and the textual reading of
///   I.5.1/I.5.2's `qX, qY, qB` naming, but **not shipped**: real fixtures
///   contradict it (see below).
/// * `false` (shipped): Y, X, B — the order I.4 states explicitly for *HF*
///   coefficients, here found to also govern *LF* coefficients. Settled by
///   decoding fixtures 50 and 51 (greyscale VarDCT content, two distances)
///   under both readings: the first-decoded channel alone carries the
///   source's structure (variance in the thousands) while the other two are
///   exactly flat (variance `0.0`), which only makes sense if the
///   first-decoded channel is luma (Y) — a genuinely achromatic source's two
///   chroma-like channels (X, B) have nothing to carry.
///
/// Flipping this constant is the whole change if a future finding
/// disagrees; see
/// `docs/experiments/2026-08-03-lf-quant-channel-order-fixture-evidence.md`
/// for the fixture evidence and its limits (still only greyscale content;
/// the RGB counterpart fixture was not successfully probed — see
/// `lf_quant_channel_order_probe_against_fixture_51`'s doc comment).
pub const LF_QUANT_CHANNEL_ORDER_IS_XYB: bool = false;

/// The three `LfQuant` channels of G.2.2, still quantized integers.
///
/// Named `x`/`y`/`b` by *semantics*, not by stream position — the field you
/// read is always "the X channel" regardless of where it sat in the
/// sub-bitstream, so a caller cannot be tripped up by
/// [`LF_QUANT_CHANNEL_ORDER_IS_XYB`]'s reordering; [`read_lf_quant_ordered`]
/// is where that reassignment happens.
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
    read_lf_quant_ordered(
        reader,
        group_width,
        group_height,
        jpeg_upsampling,
        LF_QUANT_CHANNEL_ORDER_IS_XYB,
        options,
        tree_source,
        guard,
    )
}

/// [`read_lf_quant`], with the channel order as an explicit parameter rather
/// than baked in from [`LF_QUANT_CHANNEL_ORDER_IS_XYB`].
///
/// Exists so the flip point is a real branch — not just a constant nobody
/// reads — and so this module's own tests can decode the same bitstream both
/// ways without recompiling. `true` reads X, Y, B; `false` reads Y, X, B
/// (I.4's order, and [`LF_QUANT_CHANNEL_ORDER_IS_XYB`]'s shipped value).
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors read_lf_quant's own signature plus one extra bool; \
              splitting the two sub-bitstream parameters (options, \
              tree_source) into a bundle would only hide the H.2 coupling \
              between them"
)]
fn read_lf_quant_ordered(
    reader: &mut BitReader<'_>,
    group_width: u32,
    group_height: u32,
    jpeg_upsampling: [u32; 3],
    xyb_order: bool,
    options: &ModularOptions,
    tree_source: TreeSource<'_>,
    guard: &mut AllocGuard,
) -> Result<LfQuantPlanes> {
    let extra_precision = trace_field!(reader, "lf_coeff.extra_precision", reader.read_bits(2))?;
    let extra_precision = u8::try_from(extra_precision).unwrap_or(0);

    // Table I.1's channel numbering is X = 0, Y = 1, B = 2; I.4's HF order
    // swaps the first two. `upsampling[c]` always indexes by Table I.1's
    // numbering (F.2 defines the field that way), so only the *order the
    // specs are listed in* changes, not which `jpeg_upsampling` entry feeds
    // which position.
    let (first, second) = if xyb_order { (0, 1) } else { (1, 0) };
    let upsampling_at = |c: usize| jpeg_upsampling.get(c).copied().unwrap_or(0);
    let specs = [
        lf_quant_channel_spec(group_width, group_height, upsampling_at(first)),
        lf_quant_channel_spec(group_width, group_height, upsampling_at(second)),
        lf_quant_channel_spec(group_width, group_height, upsampling_at(2)),
    ];

    let image = decode_sub_bitstream_with(reader, &specs, options, tree_source, guard)?;
    let channels = image.into_channels();
    let [pos0, pos1, b]: [Channel; 3] = channels
        .try_into()
        .map_err(|_| DecodeError::out_of_range("LfQuant channel count", "G.2.2", 3))?;
    let (x, y) = if xyb_order {
        (pos0, pos1)
    } else {
        (pos1, pos0)
    };

    Ok(LfQuantPlanes {
        extra_precision,
        x,
        y,
        b,
    })
}

// ---------------------------------------------------------------------------
// I.5.2 — LF dequantization and adaptive smoothing
// ---------------------------------------------------------------------------
//
// I.5.2's own text (paraphrased, no ISO text quoted):
//
// * Skipped entirely for a frame with `kUseLfFrame` set — already excluded
//   upstream, since such a frame never reaches VarDCT decoding at all yet
//   (`decode.rs::check_supported` rejects it).
// * `dX = mXDC * qX / (1 << extra_precision)`, and likewise for Y and B —
//   this is [`dequantize_channel`].
// * Then I.6 chroma-from-luma runs over the dequantized `(dX, dY, dB)`.
// * Then, unless `kSkipAdaptiveLFSmoothing` is set, the adaptive smoothing
//   pass runs over the CfL-corrected planes — this is [`adaptive_smoothing`].
//   Both I.6 and the smoothing pass require every channel to share one
//   shape (I.6: "skipped if any channel is subsampled"; smoothing: "no
//   channel is subsampled"), so [`dequantize_lf`] skips both stages together
//   whenever the caller reports a subsampled group.

/// One dequantized LF plane: `width x height` `f32` samples in raster order.
///
/// A distinct type from [`jpxl_core::varblock::SampleBlock`] on purpose —
/// this is a whole LF-group-sized plane, not one varblock's samples.
#[derive(Debug, Clone, PartialEq)]
pub struct DequantPlane {
    /// Width in samples.
    pub width: u32,
    /// Height in samples.
    pub height: u32,
    /// Samples in raster order, `width` per row.
    pub samples: Vec<f32>,
}

impl DequantPlane {
    /// `plane(x, y)`, or `0.0` out of bounds — matches
    /// [`crate::modular::Channel::get`]'s convention so a caller can treat a
    /// dequantized plane the same way.
    #[must_use]
    pub fn get(&self, x: u32, y: u32) -> f32 {
        if x >= self.width || y >= self.height {
            return 0.0;
        }
        self.samples
            .get(y as usize * self.width as usize + x as usize)
            .copied()
            .unwrap_or(0.0)
    }
}

/// I.5.2's LF dequantization formula, per channel:
/// `d = multiplier * q / (1 << extra_precision)`.
///
/// `multiplier` is one of [`super::quantizer::LfDequantMultipliers`]'s three
/// components (`mXDC`, `mYDC`, or `mBDC`).
#[must_use]
pub fn dequantize_channel(channel: &Channel, multiplier: f32, extra_precision: u8) -> DequantPlane {
    // `extra_precision` is a `u(2)`, so `1 << extra_precision` is 1, 2, 4 or
    // 8 — always exactly representable in `f32`.
    let divisor = f32::from(1u16 << extra_precision);
    let samples = channel
        .samples()
        .iter()
        .map(|&q| multiplier * (q as f32) / divisor)
        .collect();
    DequantPlane {
        width: channel.width(),
        height: channel.height(),
        samples,
    }
}

/// I.5.2's adaptive LF smoothing weights, truncated to `f32` precision (the
/// full verbatim decimal literals are in
/// [`smoothing_weights_sum_to_close_to_one`], which checks the sum in `f64`
/// against the untruncated digits) — a free check on the transcription the
/// scoping report already relied on.
const SMOOTH_CENTRE: f32 = 0.052_262_735;
/// Weight for each of the four horizontally/vertically adjacent neighbours.
const SMOOTH_EDGE: f32 = 0.203_451_4;
/// Weight for each of the four diagonal neighbours.
const SMOOTH_DIAG: f32 = 0.033_482_92;

/// The 3x3 weighted average `wa` of I.5.2's smoothing pass at `(x, y)`.
///
/// The caller must guarantee `1 <= x < width - 1` and `1 <= y < height - 1`
/// so every neighbour access is in bounds; [`adaptive_smoothing`] only calls
/// this over that interior range.
fn weighted_average(plane: &DequantPlane, x: u32, y: u32) -> f32 {
    let g = |dx: u32, dy: u32| plane.get(x + dx - 1, y + dy - 1);
    SMOOTH_CENTRE * plane.get(x, y)
        + SMOOTH_EDGE * (g(0, 1) + g(2, 1) + g(1, 0) + g(1, 2))
        + SMOOTH_DIAG * (g(0, 0) + g(2, 0) + g(0, 2) + g(2, 2))
}

/// I.5.2's adaptive LF smoothing pass over three same-shaped, CfL-corrected
/// planes.
///
/// The first/last row and column are left unchanged (I.5.2: the pass only
/// touches an LF sample "not in the first or last row or column").
///
/// # Errors
///
/// [`crate::error::DecodeError::FieldOutOfRange`] if the three planes do not
/// share one shape — the precondition I.5.2 states ("no channel is
/// subsampled") and [`dequantize_lf`] already enforces before calling this.
pub fn adaptive_smoothing(
    x: &DequantPlane,
    y: &DequantPlane,
    b: &DequantPlane,
    multipliers: &LfDequantMultipliers,
) -> Result<(DequantPlane, DequantPlane, DequantPlane)> {
    if x.width != y.width || x.width != b.width || x.height != y.height || x.height != b.height {
        return Err(DecodeError::out_of_range(
            "LF planes do not share one shape for adaptive smoothing",
            "I.5.2",
            u64::from(x.width),
        ));
    }
    let (width, height) = (x.width, x.height);
    let mut ox = x.samples.clone();
    let mut oy = y.samples.clone();
    let mut ob = b.samples.clone();

    if width >= 3 && height >= 3 {
        for py in 1..height - 1 {
            for px in 1..width - 1 {
                let (sx, sy, sb) = (x.get(px, py), y.get(px, py), b.get(px, py));
                let (wax, way, wab) = (
                    weighted_average(x, px, py),
                    weighted_average(y, px, py),
                    weighted_average(b, px, py),
                );
                let gap = 0.5f32
                    .max((wax - sx).abs() / multipliers.x())
                    .max((way - sy).abs() / multipliers.y())
                    .max((wab - sb).abs() / multipliers.b());
                let factor = (3.0 - 4.0 * gap).max(0.0);
                let idx = (py * width + px) as usize;
                if let Some(slot) = ox.get_mut(idx) {
                    *slot = (wax - sx) * factor + sx;
                }
                if let Some(slot) = oy.get_mut(idx) {
                    *slot = (way - sy) * factor + sy;
                }
                if let Some(slot) = ob.get_mut(idx) {
                    *slot = (wab - sb) * factor + sb;
                }
            }
        }
    }

    Ok((
        DequantPlane {
            width,
            height,
            samples: ox,
        },
        DequantPlane {
            width,
            height,
            samples: oy,
        },
        DequantPlane {
            width,
            height,
            samples: ob,
        },
    ))
}

/// The three dequantized LF planes, after I.6 chroma-from-luma and (unless
/// skipped) I.5.2 adaptive smoothing.
#[derive(Debug, Clone, PartialEq)]
pub struct DequantizedLf {
    /// Dequantized, CfL-corrected, (optionally) smoothed X plane.
    pub x: DequantPlane,
    /// Dequantized Y plane. CfL never modifies Y; smoothing may.
    pub y: DequantPlane,
    /// Dequantized, CfL-corrected, (optionally) smoothed B plane.
    pub b: DequantPlane,
}

/// I.5.2 end to end: dequantizes `planes`, then (unless `subsampled`) applies
/// I.6 chroma-from-luma and, if `smoothing_enabled`, the adaptive smoothing
/// pass.
///
/// `multipliers` is [`super::quantizer::Quantizer::lf_multipliers`]'s output.
/// `corr` is I.2.3's `LfChannelCorrelation`. `subsampled` is whether any of
/// `frame_header.jpeg_upsampling`'s three entries is nonzero — both I.6 and
/// the smoothing pass require it to be `false` to run at all.
/// `smoothing_enabled` is `frame_header.flags.adaptive_lf_smoothing()`
/// (`!kSkipAdaptiveLFSmoothing`).
///
/// # Errors
///
/// [`crate::error::DecodeError::FieldOutOfRange`] if `subsampled` is `false`
/// but the three planes do not in fact share one shape (a caller/geometry
/// mismatch, not a spec case).
pub fn dequantize_lf(
    planes: &LfQuantPlanes,
    multipliers: &LfDequantMultipliers,
    corr: &LfChannelCorrelation,
    subsampled: bool,
    smoothing_enabled: bool,
) -> Result<DequantizedLf> {
    let dx = dequantize_channel(&planes.x, multipliers.x(), planes.extra_precision);
    let dy = dequantize_channel(&planes.y, multipliers.y(), planes.extra_precision);
    let db = dequantize_channel(&planes.b, multipliers.b(), planes.extra_precision);

    if subsampled {
        return Ok(DequantizedLf {
            x: dx,
            y: dy,
            b: db,
        });
    }
    if dx.width != dy.width
        || dx.width != db.width
        || dx.height != dy.height
        || dx.height != db.height
    {
        return Err(DecodeError::out_of_range(
            "LfQuant channel shapes disagree although none is reported subsampled",
            "I.6",
            u64::from(dx.width),
        ));
    }

    let (k_x, k_b) = CflFactors::for_lf(corr).at(0, 0);
    let mut cx = Vec::with_capacity(dx.samples.len());
    let mut cb = Vec::with_capacity(db.samples.len());
    for i in 0..dx.samples.len() {
        // Indices past `dx.samples.len()` never happen: `dy`/`db` were just
        // proven the same length as `dx` by the shape check above.
        let dy_i = dy.samples.get(i).copied().unwrap_or(0.0);
        let db_i = db.samples.get(i).copied().unwrap_or(0.0);
        let dx_i = dx.samples.get(i).copied().unwrap_or(0.0);
        let (x_v, _y_v, b_v) = cfl::apply(dx_i, dy_i, db_i, k_x, k_b);
        cx.push(x_v);
        cb.push(b_v);
    }
    let (width, height) = (dx.width, dx.height);
    let cx = DequantPlane {
        width,
        height,
        samples: cx,
    };
    let cy = dy;
    let cb = DequantPlane {
        width,
        height,
        samples: cb,
    };

    if !smoothing_enabled {
        return Ok(DequantizedLf {
            x: cx,
            y: cy,
            b: cb,
        });
    }
    let (sx, sy, sb) = adaptive_smoothing(&cx, &cy, &cb, multipliers)?;
    Ok(DequantizedLf {
        x: sx,
        y: sy,
        b: sb,
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
        // six samples total, decoded **Y then X then B**
        // ([`LF_QUANT_CHANNEL_ORDER_IS_XYB`]'s shipped reading — the same
        // order I.4 uses for HF coefficients), Zero predictor so each sample
        // is just UnpackSigned(token).
        let mut w = BitWriter::default();
        w.u(0, 2); // extra_precision = 0
        write_header_no_transforms(&mut w);
        write_single_leaf_tree(&mut w, 0); // predictor 0 = Zero
        write_prefix_bundle(&mut w, 1);
        // Stream order Y, X, B: Y=[-1, 1], X=[1, -1], B=[-2, 0] — a distinct
        // pattern per channel, drawn from the 4-symbol test alphabet's range.
        for v in [-1i32, 1, 1, -1, -2, 0] {
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
        // Stream order Y, X, B: Y=-1, X=1, B=-2.
        for v in [-1i32, 1, -2] {
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

    // -----------------------------------------------------------------
    // I.5.2 dequantization, adaptive smoothing, and I.6 chroma-from-luma.
    // -----------------------------------------------------------------

    fn multipliers(x: f32, y: f32, b: f32) -> LfDequantMultipliers {
        // `Quantizer::lf_multipliers` is the only public constructor;
        // reverse-engineer the inputs that make it emit exactly `(x, y, b)`
        // rather than reaching into the tuple field directly.
        let weights = LfChannelCorrelation::default(); // unused by lf_multipliers
        let _ = weights; // silence "unused" if the reversal below changes
        let quantizer = super::super::quantizer::Quantizer {
            global_scale: 1,
            quant_lf: 1,
        };
        let dequant = super::super::quantizer::LfChannelDequantization {
            m_x_lf: x * super::super::quantizer::LF_WEIGHT_SCALE / 65536.0,
            m_y_lf: y * super::super::quantizer::LF_WEIGHT_SCALE / 65536.0,
            m_b_lf: b * super::super::quantizer::LF_WEIGHT_SCALE / 65536.0,
        };
        quantizer.lf_multipliers(&dequant)
    }

    #[test]
    fn smoothing_weights_sum_to_close_to_one() {
        // The scoping report verified the printed decimal literals sum to
        // exactly 1.0 in real-number arithmetic; this checks the `f32`
        // constants actually compiled into this module land within a tight
        // tolerance of that identity (bit-exact `f32` equality is not
        // guaranteed — each literal is independently rounded to 24 bits).
        let sum =
            f64::from(SMOOTH_CENTRE) + 4.0 * f64::from(SMOOTH_EDGE) + 4.0 * f64::from(SMOOTH_DIAG);
        assert!((sum - 1.0).abs() < 1e-6, "sum = {sum}");
    }

    #[test]
    fn dequantize_channel_matches_the_i52_formula() {
        // d = multiplier * q / (1 << extra_precision).
        // multiplier = 2.5, extra_precision = 1 (divisor 2):
        //   q = 4  -> d = 2.5*4/2 = 5.0
        //   q = -3 -> d = 2.5*-3/2 = -3.75
        let channel =
            Channel::from_samples(ChannelSpec::new(2, 1), vec![4, -3]).expect("2 samples");
        let plane = dequantize_channel(&channel, 2.5, 1);
        assert_eq!(plane.width, 2);
        assert_eq!(plane.height, 1);
        assert!((plane.samples[0] - 5.0).abs() < 1e-6);
        assert!((plane.samples[1] - (-3.75)).abs() < 1e-6);
    }

    #[test]
    fn dequantize_channel_extra_precision_zero_is_the_bare_multiplier() {
        let channel = Channel::from_samples(ChannelSpec::new(1, 1), vec![7]).expect("1 sample");
        let plane = dequantize_channel(&channel, 3.0, 0);
        assert!((plane.samples[0] - 21.0).abs() < 1e-6);
    }

    #[test]
    fn dequantize_lf_applies_cfl_when_not_subsampled_and_smoothing_is_skipped() {
        // A 1x1 LF "group" (no smoothing possible at 1x1 anyway, so this
        // isolates dequantization + CfL). qX=2, qY=4, qB=-1, extra_precision
        // = 0, multipliers mX=1, mY=1, mB=1 -> dX=2, dY=4, dB=-1.
        // kX = base_correlation_x=0 + x_factor/colour_factor; with
        // colour_factor=4, x_factor_lf=132 (x_factor=4) -> kX=1.0.
        // kB: base_correlation_b=1 + b_factor/colour_factor; b_factor_lf=128
        // (b_factor=0) -> kB=1.0.
        // X = dX + kX*dY = 2 + 1.0*4 = 6; Y = 4; B = dB + kB*dY = -1 + 4 = 3.
        let planes = LfQuantPlanes {
            extra_precision: 0,
            x: Channel::from_samples(ChannelSpec::new(1, 1), vec![2]).expect("1 sample"),
            y: Channel::from_samples(ChannelSpec::new(1, 1), vec![4]).expect("1 sample"),
            b: Channel::from_samples(ChannelSpec::new(1, 1), vec![-1]).expect("1 sample"),
        };
        let m = multipliers(1.0, 1.0, 1.0);
        let corr = LfChannelCorrelation {
            colour_factor: 4,
            base_correlation_x: 0.0,
            base_correlation_b: 1.0,
            x_factor_lf: 132,
            b_factor_lf: 128,
        };

        let out =
            dequantize_lf(&planes, &m, &corr, false, false).expect("well-formed, unsubsampled");
        assert!(
            (out.x.samples[0] - 6.0).abs() < 1e-6,
            "X = {}",
            out.x.samples[0]
        );
        assert!(
            (out.y.samples[0] - 4.0).abs() < 1e-6,
            "Y = {}",
            out.y.samples[0]
        );
        assert!(
            (out.b.samples[0] - 3.0).abs() < 1e-6,
            "B = {}",
            out.b.samples[0]
        );
    }

    #[test]
    fn dequantize_lf_skips_cfl_when_subsampled() {
        // Same inputs as above, but `subsampled = true`: I.6 must not run,
        // so X/Y/B are exactly dX/dY/dB with no CfL correction.
        let planes = LfQuantPlanes {
            extra_precision: 0,
            x: Channel::from_samples(ChannelSpec::new(1, 1), vec![2]).expect("1 sample"),
            y: Channel::from_samples(ChannelSpec::new(1, 1), vec![4]).expect("1 sample"),
            b: Channel::from_samples(ChannelSpec::new(1, 1), vec![-1]).expect("1 sample"),
        };
        let m = multipliers(1.0, 1.0, 1.0);
        let corr = LfChannelCorrelation {
            colour_factor: 4,
            base_correlation_x: 0.0,
            base_correlation_b: 1.0,
            x_factor_lf: 132,
            b_factor_lf: 128,
        };
        let out =
            dequantize_lf(&planes, &m, &corr, true, false).expect("subsampled path never errors");
        assert!((out.x.samples[0] - 2.0).abs() < 1e-6);
        assert!((out.y.samples[0] - 4.0).abs() < 1e-6);
        assert!((out.b.samples[0] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn smoothing_skip_path_is_bit_identical_to_its_pre_smoothing_input() {
        // "bit-identical to input" for the smoothing stage means: with
        // `smoothing_enabled = false`, the output equals exactly what CfL
        // produced, because the smoothing pass never runs at all (not
        // "runs and happens to be a no-op").
        //
        // A pure ramp is a poor choice for the second half of this test
        // (proving smoothing is not a no-op): I.5.2's weight kernel is
        // symmetric, so it reproduces any affine plane exactly (`wa == s`
        // everywhere), and a ramp's `gap` is always 0. A lone spike against
        // an otherwise-flat background is a poor choice too, for the
        // opposite reason: the centre weight (~0.052) is so small relative
        // to the rest of the kernel that *any* single-cell deviation drives
        // `gap` past the `0.75` threshold at which `3 - 4*gap` clamps to
        // zero, however the multiplier is scaled (`gap` is scale-invariant:
        // both `wa` and `s` carry the same multiplier, which cancels). What
        // *does* clear the threshold is a coordinated pattern across the
        // whole 3x3 neighbourhood — edges matching the centre, corners
        // lower — chosen below by direct computation (`gap ~= 0.536`).
        let mut x_samples = vec![10i32; 25];
        for &i in &[6usize, 8, 16, 18] {
            *x_samples.get_mut(i).expect("index in range") = 6; // the four corners of the 3x3 neighbourhood around (2,2)
        }
        let y_samples = vec![0i32; 25];
        let b_samples = vec![0i32; 25];
        let spec = ChannelSpec::new(5, 5);
        let planes = LfQuantPlanes {
            extra_precision: 0,
            x: Channel::from_samples(spec, x_samples).expect("25 samples"),
            y: Channel::from_samples(spec, y_samples).expect("25 samples"),
            b: Channel::from_samples(spec, b_samples).expect("25 samples"),
        };
        let m = multipliers(1.0, 1.0, 1.0);
        let corr = LfChannelCorrelation::default();

        let skipped = dequantize_lf(&planes, &m, &corr, false, false).expect("unsubsampled");
        // Recompute the pre-smoothing (post-dequant, post-CfL) planes by
        // hand via the same primitives, to compare against independently of
        // `dequantize_lf`'s own internals.
        let dx = dequantize_channel(&planes.x, m.x(), 0);
        let dy = dequantize_channel(&planes.y, m.y(), 0);
        let db = dequantize_channel(&planes.b, m.b(), 0);
        let (k_x, k_b) = CflFactors::for_lf(&corr).at(0, 0);
        let expected_x: Vec<f32> = dx
            .samples
            .iter()
            .zip(&dy.samples)
            .map(|(&x, &y)| x + k_x * y)
            .collect();
        let expected_b: Vec<f32> = db
            .samples
            .iter()
            .zip(&dy.samples)
            .map(|(&b, &y)| b + k_b * y)
            .collect();
        assert_eq!(skipped.x.samples, expected_x);
        assert_eq!(skipped.y.samples, dy.samples);
        assert_eq!(skipped.b.samples, expected_b);

        // And with smoothing enabled, the result must differ at the spike —
        // proving the flag actually gates something, not just that it is
        // wired through.
        let smoothed = dequantize_lf(&planes, &m, &corr, false, true).expect("unsubsampled");
        assert_ne!(
            smoothed.x.samples.get(12),
            skipped.x.samples.get(12),
            "the spike must move once smoothing runs"
        );
    }

    #[test]
    fn adaptive_smoothing_leaves_the_border_untouched() {
        // I.5.2: the smoothing pass only touches a sample "not in the first
        // or last row or column". A 3x3 plane has exactly one interior pixel
        // (1,1); every border sample must come back unchanged. The corners
        // vs. edges/centre pattern is the same well-conditioned one as
        // `smoothing_skip_path_is_bit_identical_to_its_pre_smoothing_input`
        // (a lone-spike pattern always clamps to zero change here — see that
        // test's comment for why).
        let x = DequantPlane {
            width: 3,
            height: 3,
            samples: vec![6.0, 10.0, 6.0, 10.0, 10.0, 10.0, 6.0, 10.0, 6.0],
        };
        let y = x.clone();
        let b = x.clone();
        let m = multipliers(1.0, 1.0, 1.0);
        let (sx, _sy, _sb) = adaptive_smoothing(&x, &y, &b, &m).expect("same shape");
        for (i, (&orig, &out)) in x.samples.iter().zip(&sx.samples).enumerate() {
            if i == 4 {
                continue; // the one interior pixel, expected to change
            }
            assert_eq!(orig, out, "border sample {i} was modified");
        }
        assert_ne!(sx.samples[4], 10.0, "the interior pixel must be smoothed");
    }

    #[test]
    fn adaptive_smoothing_rejects_mismatched_plane_shapes_not_panicking() {
        let x = DequantPlane {
            width: 3,
            height: 3,
            samples: vec![0.0; 9],
        };
        let y = DequantPlane {
            width: 2,
            height: 3,
            samples: vec![0.0; 6],
        };
        let b = x.clone();
        let m = multipliers(1.0, 1.0, 1.0);
        assert!(adaptive_smoothing(&x, &y, &b, &m).is_err());
    }

    #[test]
    fn dequantize_lf_rejects_mismatched_unsubsampled_shapes_not_panicking() {
        let planes = LfQuantPlanes {
            extra_precision: 0,
            x: Channel::from_samples(ChannelSpec::new(2, 1), vec![0, 0]).expect("2 samples"),
            y: Channel::from_samples(ChannelSpec::new(3, 1), vec![0, 0, 0]).expect("3 samples"),
            b: Channel::from_samples(ChannelSpec::new(2, 1), vec![0, 0]).expect("2 samples"),
        };
        let m = multipliers(1.0, 1.0, 1.0);
        let corr = LfChannelCorrelation::default();
        assert!(dequantize_lf(&planes, &m, &corr, false, false).is_err());
    }

    // -----------------------------------------------------------------
    // Flip-point probe: `LF_QUANT_CHANNEL_ORDER_IS_XYB` against a real
    // VarDCT fixture. See `docs/experiments/2026-08-03-lf-quant-channel-order.md`.
    // -----------------------------------------------------------------

    /// The byte range of TOC section `index`, relative to `base`. A local
    /// copy of `decode.rs`'s private `section_slice` — that function is not
    /// `pub`, and this probe intentionally does not touch `decode.rs`.
    fn section_slice<'a>(
        codestream: &'a [u8],
        base: usize,
        toc: &crate::frame::Toc,
        index: usize,
    ) -> &'a [u8] {
        let offset = toc.offset_of(index).expect("index in range");
        let entry_index = match &toc.permutation {
            Some(p) => p
                .get(index)
                .and_then(|&target| usize::try_from(target).ok())
                .unwrap_or(index),
            None => index,
        };
        let size = toc.entries.get(entry_index).copied().unwrap_or(0);
        let start = base + usize::try_from(offset).unwrap_or(usize::MAX);
        let end = (start + usize::try_from(size).unwrap_or(usize::MAX)).min(codestream.len());
        &codestream[start..end]
    }

    fn fixture_bytes(name: &str) -> Vec<u8> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(std::path::Path::parent)
            .expect("manifest dir has two ancestors")
            .join("tests")
            .join("fixtures")
            .join("handmade")
            .join(name);
        std::fs::read(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
    }

    /// Decodes fixture 50 up through `LfQuant` under one channel-order
    /// hypothesis, replicating the F.1/G.1 pipeline by hand (see the module
    /// doc for why `decode.rs` is not reused here) and stopping the instant
    /// `LfQuant` is decoded — nothing past G.2.2 is needed for this probe.
    fn probe_lf_quant(data: &[u8], xyb_order: bool) -> (LfQuantPlanes, LfDequantMultipliers) {
        use crate::frame::{Encoding, FrameGeometry, read_frame_header, read_toc};
        use crate::headers::decode_image_headers;
        use crate::modular::{GlobalTree, TreeSource, read_global_tree};
        use crate::vardct::quantizer::{read_lf_channel_dequantization, read_lf_global_vardct};

        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let mut reader = BitReader::new(data);
        let headers = decode_image_headers(&mut reader, &limits).expect("valid image headers");
        assert!(
            headers.metadata.ec_info.is_empty(),
            "probe assumes no extra channels (fixture 50 is plain greyscale)"
        );
        if headers.metadata.colour_encoding.want_icc {
            crate::icc::read_icc_profile(&mut reader, &mut guard).expect("valid ICC profile");
        }
        reader
            .zero_pad_to_byte()
            .expect("byte-aligned before the frame");
        let cursor = usize::try_from(reader.total_bits_read() / 8).expect("small offset");

        let rest = &data[cursor..];
        let mut frame_reader = BitReader::new(rest);
        let header = read_frame_header(
            &mut frame_reader,
            &headers.metadata,
            headers.width(),
            headers.height(),
            &limits,
            &mut guard,
        )
        .expect("valid frame header");
        assert_eq!(header.encoding, Encoding::VarDct, "fixture 50 is VarDCT");

        let geometry = FrameGeometry::from_header(
            &header,
            headers.width(),
            headers.height(),
            &limits,
            &mut guard,
        )
        .expect("valid geometry");
        let toc = read_toc(
            &mut frame_reader,
            geometry.num_sections(),
            &limits,
            &mut guard,
        )
        .expect("valid TOC");
        let section_base =
            cursor + usize::try_from(frame_reader.total_bits_read() / 8).expect("small TOC");

        let whole = section_slice(data, section_base, &toc, 0);
        let mut single = BitReader::new(whole);

        // G.1.2: LfChannelDequantization (unconditional).
        let dequant_weights = read_lf_channel_dequantization(&mut single).expect("valid G.1.2");
        // G.1's kVarDCT-only rows: Quantizer, HfBlockContext, LfChannelCorrelation.
        let lf_global_vardct =
            read_lf_global_vardct(&mut single, &mut guard).expect("valid I.2.1-I.2.3");

        // G.1.3: GlobalModular's leading Bool(), always read regardless of
        // channel count (mirrors decode.rs::decode_frame). No extra channels
        // and zero VarDCT colour channels means the sub-bitstream itself
        // decodes nothing (H.1: "N == 0" takes no action), so this probe
        // does not need to replicate `decode_sub_bitstream_partial` at all.
        let have_global_tree = single.read_bool().expect("valid Bool()");
        let global_tree: Option<GlobalTree> = if have_global_tree {
            let options = crate::modular::ModularOptions {
                bits_per_sample: headers.metadata.bit_depth.bits_per_sample(),
                ..crate::modular::ModularOptions::level10()
            };
            Some(read_global_tree(&mut single, &options, &mut guard).expect("valid global tree"))
        } else {
            None
        };

        // G.2.2: LF coefficients, the first field of the one LF group.
        let rect = geometry.lf_group_rect(0).expect("at least one LF group");
        let stream_index =
            crate::frame::stream_index::lf_coefficients(&geometry, 0).expect("valid stream index");
        let options = crate::modular::ModularOptions {
            stream_index,
            bits_per_sample: headers.metadata.bit_depth.bits_per_sample(),
            ..crate::modular::ModularOptions::level10()
        };
        let source = match &global_tree {
            Some(g) => TreeSource::Global {
                global: g,
                restart: true,
            },
            None => TreeSource::Local,
        };
        let planes = read_lf_quant_ordered(
            &mut single,
            rect.width,
            rect.height,
            header.jpeg_upsampling,
            xyb_order,
            &options,
            source,
            &mut guard,
        )
        .expect("valid LfQuant sub-bitstream");

        let multipliers = lf_global_vardct.quantizer.lf_multipliers(&dequant_weights);
        (planes, multipliers)
    }

    /// Population variance of a dequantized plane's samples — the
    /// plausibility signal: a real encode's luma-like (Y) channel carries
    /// the image's actual structure (here, a smooth ramp against a
    /// checkerboard, per the fixture's provenance sidecar) while the
    /// chroma-like (X, B) channels of a *genuinely greyscale* source should
    /// be comparatively flat.
    fn variance(plane: &DequantPlane) -> f64 {
        let n = plane.samples.len() as f64;
        if n == 0.0 {
            return 0.0;
        }
        let mean: f64 = plane.samples.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
        plane
            .samples
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / n
    }

    /// A variance far above this counts as "carries real structure"; a
    /// variance below it counts as "flat" (chroma with nothing to encode).
    /// The observed values are ~832 and ~7958 for the structured channel and
    /// exactly `0.0` for the flat ones, so this threshold has enormous
    /// headroom on both sides.
    const PLAUSIBLE_STRUCTURE_THRESHOLD: f64 = 1.0;

    #[test]
    fn lf_quant_channel_order_probe_against_fixture_50() {
        // The decisive test for `LF_QUANT_CHANNEL_ORDER_IS_XYB`: a real
        // greyscale VarDCT fixture's LF plane must decode to "Y carries the
        // structure, X and B are flat" under the shipped (Y, X, B) reading.
        // See the constant's doc and the experiment writeup for the full
        // argument; this is that argument turned into a standing regression
        // check, not just a printed observation.
        let data = fixture_bytes("50_vardct_mixed_gray_128x128_nofilters_d1.jxl");

        let (planes, _m) = probe_lf_quant(&data, LF_QUANT_CHANNEL_ORDER_IS_XYB);
        let var_x = variance(&dequantize_channel(&planes.x, 1.0, planes.extra_precision));
        let var_y = variance(&dequantize_channel(&planes.y, 1.0, planes.extra_precision));
        let var_b = variance(&dequantize_channel(&planes.b, 1.0, planes.extra_precision));
        eprintln!(
            "LF_QUANT_CHANNEL_ORDER_IS_XYB probe (fixture 50): var(x)={var_x:.6} var(y)={var_y:.6} var(b)={var_b:.6}"
        );

        assert_eq!(
            var_x, 0.0,
            "X must be flat for a genuinely greyscale source"
        );
        assert_eq!(
            var_b, 0.0,
            "B must be flat for a genuinely greyscale source"
        );
        assert!(
            var_y > PLAUSIBLE_STRUCTURE_THRESHOLD,
            "Y must carry the source's structure, got variance {var_y}"
        );
    }

    #[test]
    fn lf_quant_channel_order_probe_against_fixture_51() {
        // Corroborating data point: the same greyscale content as fixture
        // 50, at a different distance (`-d 4` vs `-d 1`). If the flat/huge
        // split seen against fixture 50 were a coincidence of one specific
        // quantization level, a different distance would be the first place
        // it broke.
        //
        // The RGB counterpart (fixture 54) was also attempted here and is
        // *not* included: this hand-rolled probe harness (which exists only
        // to avoid touching `decode.rs`, see the module doc) fails to
        // decode fixture 54's `LfQuant` sub-bitstream at all — an ANS
        // final-state mismatch, past the point where `read_frame_header`,
        // `read_lf_global_vardct` and the global MA tree have all already
        // succeeded and reported plausible values (`jpeg_upsampling`,
        // `colour_factor`, `global_scale`, `quant_lf` all sane, no
        // patches/splines/noise bundles to misalign against). That failure
        // reproduces identically under both `xyb_order` settings — it is a
        // gap in this harness's replication of the real pipeline for a
        // genuinely multi-channel source, not new evidence about channel
        // order, and it is 8F's problem once `decode.rs` grows real VarDCT
        // wiring, not this probe's.
        let data = fixture_bytes("51_vardct_mixed_gray_128x128_nofilters_d4.jxl");
        let (planes, _m) = probe_lf_quant(&data, LF_QUANT_CHANNEL_ORDER_IS_XYB);
        let var_x = variance(&dequantize_channel(&planes.x, 1.0, planes.extra_precision));
        let var_y = variance(&dequantize_channel(&planes.y, 1.0, planes.extra_precision));
        let var_b = variance(&dequantize_channel(&planes.b, 1.0, planes.extra_precision));
        eprintln!(
            "LF_QUANT_CHANNEL_ORDER_IS_XYB probe (fixture 51): var(x)={var_x:.6} var(y)={var_y:.6} var(b)={var_b:.6}"
        );

        assert_eq!(
            var_x, 0.0,
            "X must be flat for a genuinely greyscale source"
        );
        assert_eq!(
            var_b, 0.0,
            "B must be flat for a genuinely greyscale source"
        );
        assert!(
            var_y > PLAUSIBLE_STRUCTURE_THRESHOLD,
            "Y must carry the source's structure, got variance {var_y}"
        );
    }
}
