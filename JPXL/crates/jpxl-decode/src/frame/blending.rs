//! The `BlendingInfo` bundle (18181-1 Table F.7) and `BlendMode` (Table F.8).
//!
//! ```text
//! condition                                             type               default  name
//!                                                       U32(0,1,2,3+u(2))  0        mode
//! extra and (mode == kBlend or kMulAdd)                 U32(0,1,2,3+u(3))  0        alpha_channel
//! extra and (mode == kBlend or kMulAdd or kMul)         Bool()             false    clamp
//! !resets_canvas                                        u(2)               0        source
//! ```
//!
//! `extra` is "the number of extra channels is at least one".
//!
//! # `resets_canvas` is one value, not one per bundle
//!
//! F.2 defines `resets_canvas` as `full_frame and blending_info.mode ==
//! kReplace` — naming the frame's *colour* `blending_info` specifically. The
//! frame header then reads `blending_info` followed by
//! `ec_blending_info[num_extra]`, all of which contain a `source` row gated on
//! `!resets_canvas`.
//!
//! Read literally, that single expression governs every one of those bundles,
//! so the extra-channel bundles are gated by the *colour* bundle's mode rather
//! than their own. That is the reading implemented here: [`read_blending_info`]
//! takes `resets_canvas` as a parameter, and the frame header computes it once
//! from the colour bundle and passes the same value to every extra-channel
//! bundle. The alternative — each bundle evaluating the expression against its
//! own mode — differs by two bits per extra channel whenever an extra channel's
//! mode disagrees with the colour channel's about being `kReplace`.
//! **TODO(slice 7):** confirm against a real multi-extra-channel stream.

use jpxl_bitstream::{BitReader, U32Dist, U32Spec, read_bool, read_u32, trace_field};

use crate::frame::error::{FrameError, Result};

/// 18181-1 F.7: `U32(0, 1, 2, 3 + u(2))`.
const MODE_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(0),
    U32Dist::Val(1),
    U32Dist::Val(2),
    U32Dist::BitsOffset { bits: 2, offset: 3 },
]);

/// 18181-1 F.7: `U32(0, 1, 2, 3 + u(3))`.
const ALPHA_CHANNEL_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(0),
    U32Dist::Val(1),
    U32Dist::Val(2),
    U32Dist::BitsOffset { bits: 3, offset: 3 },
]);

/// How a frame is composited onto the canvas (18181-1 Table F.8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum BlendMode {
    /// `sample = new_sample`.
    #[default]
    Replace = 0,
    /// `sample = old_sample + new_sample`.
    Add = 1,
    /// Alpha-blend the new sample over the previous one.
    Blend = 2,
    /// `sample = old_sample + new_alpha * new_sample`.
    MulAdd = 3,
    /// `sample = old_sample * new_sample`.
    Mul = 4,
}

impl BlendMode {
    /// Maps a wire value to a row of Table F.8.
    #[must_use]
    pub const fn from_value(value: u32) -> Option<Self> {
        Some(match value {
            0 => Self::Replace,
            1 => Self::Add,
            2 => Self::Blend,
            3 => Self::MulAdd,
            4 => Self::Mul,
            _ => return None,
        })
    }

    /// The value as stored in the codestream.
    #[must_use]
    pub const fn value(self) -> u32 {
        self as u32
    }

    /// Whether this mode reads an `alpha_channel` index (`kBlend`/`kMulAdd`).
    #[must_use]
    pub const fn uses_alpha_channel(self) -> bool {
        matches!(self, Self::Blend | Self::MulAdd)
    }

    /// Whether this mode reads a `clamp` flag (`kBlend`/`kMulAdd`/`kMul`).
    #[must_use]
    pub const fn uses_clamp(self) -> bool {
        matches!(self, Self::Blend | Self::MulAdd | Self::Mul)
    }
}

/// A decoded `BlendingInfo` bundle (18181-1 Table F.7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlendingInfo {
    /// Compositing mode.
    pub mode: BlendMode,
    /// Index of the extra channel holding alpha, for `kBlend`/`kMulAdd`.
    pub alpha_channel: u32,
    /// Whether alpha (or the sample, for `kMul`) is clamped to `[0, 1]`.
    pub clamp: bool,
    /// Which reference buffer supplies the previous samples.
    pub source: u32,
}

/// Reads a `BlendingInfo` bundle (18181-1 Table F.7).
///
/// `extra` is true when the image has at least one extra channel.
/// `resets_canvas` gates the `source` row; see the module documentation for
/// why it is a parameter rather than derived from `mode` here.
///
/// # Errors
///
/// [`FrameError::FieldOutOfRange`] for a `mode` outside Table F.8, or a
/// bitstream error on truncation.
pub fn read_blending_info(
    reader: &mut BitReader<'_>,
    extra: bool,
    resets_canvas: bool,
    field: &'static str,
) -> Result<BlendingInfo> {
    let raw_mode = trace_field!(reader, field, read_u32(reader, &MODE_SPEC))?;
    let mode = BlendMode::from_value(raw_mode).ok_or_else(|| {
        FrameError::out_of_range("blending_info.mode", "F.8", u64::from(raw_mode))
    })?;

    let alpha_channel = if extra && mode.uses_alpha_channel() {
        trace_field!(
            reader,
            "blending_info.alpha_channel",
            read_u32(reader, &ALPHA_CHANNEL_SPEC)
        )?
    } else {
        0
    };

    let clamp = if extra && mode.uses_clamp() {
        trace_field!(reader, "blending_info.clamp", read_bool(reader))?
    } else {
        false
    };

    let source = if resets_canvas {
        0
    } else {
        trace_field!(reader, "blending_info.source", reader.read_bits(2))?
    };

    Ok(BlendingInfo {
        mode,
        alpha_channel,
        clamp,
        source,
    })
}

/// Reads just the `mode` field, used to evaluate `resets_canvas` before the
/// rest of the bundle is known.
///
/// # Errors
///
/// As [`read_blending_info`].
pub fn peek_blend_mode(reader: &BitReader<'_>) -> Result<BlendMode> {
    let mut probe = reader.clone();
    let raw = read_u32(&mut probe, &MODE_SPEC)?;
    BlendMode::from_value(raw)
        .ok_or_else(|| FrameError::out_of_range("blending_info.mode", "F.8", u64::from(raw)))
}

// ---------------------------------------------------------------------------
// F.2 — compositing a frame onto the canvas
// ---------------------------------------------------------------------------

/// One `Reference[]` buffer, or the canvas a frame is composited onto.
///
/// # Which colour space
///
/// F.2 is explicit: "the blending is done in the colour space after inverse
/// colour transforms from Annex L have been applied (except for L.4)". So a
/// canvas holds display-space samples on the nominal `[0, 1]` scale — the same
/// scale [`crate::DecodedImage::float_planes`] uses — and **not** the XYB that
/// K.3.2's patch references hold. The two uses of a `save_as_reference` slot
/// are distinguished by `save_before_ct`, and this type is the `false` one.
///
/// Samples are unclipped: `kAdd` legitimately overshoots, and clipping here
/// would change what a later frame blends against.
#[derive(Debug, Clone, PartialEq)]
pub struct Canvas {
    /// Width in samples.
    pub width: u32,
    /// Height in samples.
    pub height: u32,
    /// Colour planes, 1 for greyscale and 3 otherwise.
    pub colour: Vec<Vec<f32>>,
    /// Extra-channel planes, in `ec_info` index order.
    pub extra: Vec<Vec<f32>>,
}

impl Canvas {
    /// A canvas of zeroes — F.2's "if no frame was previously stored, the
    /// source frame is assumed to have all sample values set to zeroes".
    #[must_use]
    pub fn zeros(width: u32, height: u32, num_colour: usize, num_extra: usize) -> Self {
        let len = width as usize * height as usize;
        Self {
            width,
            height,
            colour: vec![vec![0.0; len]; num_colour],
            extra: vec![vec![0.0; len]; num_extra],
        }
    }

    /// One colour sample, or `0.0` outside the canvas.
    #[must_use]
    pub fn colour_at(&self, channel: usize, x: u32, y: u32) -> f32 {
        Self::at(self.colour.get(channel), self.width, self.height, x, y)
    }

    /// One extra-channel sample, or `0.0` outside the canvas.
    #[must_use]
    pub fn extra_at(&self, channel: usize, x: u32, y: u32) -> f32 {
        Self::at(self.extra.get(channel), self.width, self.height, x, y)
    }

    fn at(plane: Option<&Vec<f32>>, width: u32, height: u32, x: u32, y: u32) -> f32 {
        if x >= width || y >= height {
            return 0.0;
        }
        plane
            .and_then(|p| p.get(y as usize * width as usize + x as usize))
            .copied()
            .unwrap_or(0.0)
    }
}

/// **Flip point — which extra channel is "the alpha channel itself" (F.8).**
///
/// Table F.8 gives `kBlend` and `kMulAdd` a second formula "for the alpha
/// channel itself", where `kBlend` becomes the Porter-Duff `over` on opacity
/// and `kMulAdd` preserves the source frame's value. The clause never says
/// which extra channels that exception covers.
///
/// * `true` (shipped): the channel named by this rule's `alpha_channel`, i.e.
///   the exception fires exactly when the channel being blended is the same
///   one supplying `new_alpha`/`old_alpha`. Under this reading the formula is
///   self-consistent — `new_sample` and `new_alpha` are then the same number —
///   which is what makes the alternate formula a *simplification* rather than
///   a different operation.
/// * `false`: every extra channel of type `kAlpha`, whether or not it is the
///   one being blended with.
///
/// The two readings coincide for any image with a single alpha channel, which
/// is every multi-frame stream in the corpus (`blendmodes` blends its one
/// alpha channel with `alpha_channel = 0` in all four non-trivial modes).
pub const ALPHA_SELF_RULE_IS_THE_NAMED_CHANNEL: bool = true;

/// What a single [`blend_sample`] call needs from Table F.8.
#[derive(Debug, Clone, Copy)]
pub struct BlendContext {
    /// The rule in force for this channel group.
    pub info: BlendingInfo,
    /// Whether the named alpha channel has premultiplied semantics
    /// (`ec_info[alpha_channel].alpha_associated`, D.3.6).
    pub alpha_associated: bool,
    /// Whether the sample being blended is the alpha channel itself; see
    /// [`ALPHA_SELF_RULE_IS_THE_NAMED_CHANNEL`].
    pub is_alpha_itself: bool,
}

/// Table F.8, for one sample.
///
/// `new_alpha`/`old_alpha` are the named alpha channel's values at the same
/// position, from the current frame and the source canvas respectively; they
/// are ignored by the modes that do not read alpha.
#[must_use]
pub fn blend_sample(
    ctx: &BlendContext,
    old_sample: f32,
    new_sample: f32,
    old_alpha: f32,
    new_alpha: f32,
) -> f32 {
    // F.2: with `clamp`, kBlend and kMulAdd clamp new_alpha to [0, 1], and
    // kMul clamps new_sample instead.
    let new_alpha = if ctx.info.clamp {
        new_alpha.clamp(0.0, 1.0)
    } else {
        new_alpha
    };
    match ctx.info.mode {
        BlendMode::Replace => new_sample,
        BlendMode::Add => old_sample + new_sample,
        BlendMode::Blend => {
            if ctx.is_alpha_itself {
                // "The blending on the alpha channel itself always uses the
                // following formula instead."
                return new_alpha.mul_add(1.0 - old_alpha, old_alpha);
            }
            if ctx.alpha_associated {
                return new_alpha.mul_add(-old_sample, old_sample) + new_sample;
            }
            let alpha = new_alpha.mul_add(1.0 - old_alpha, old_alpha);
            if alpha == 0.0 {
                // Fully transparent on both sides: the quotient is 0/0 and
                // every term of the numerator is zero, so the only value that
                // is not an invention is zero.
                return 0.0;
            }
            (new_alpha * new_sample + old_alpha * old_sample * (1.0 - new_alpha)) / alpha
        }
        BlendMode::MulAdd => {
            if ctx.is_alpha_itself {
                // "For the alpha channel itself, the values of the source
                // frame are preserved."
                return old_alpha;
            }
            new_alpha.mul_add(new_sample, old_sample)
        }
        BlendMode::Mul => {
            let new_sample = if ctx.info.clamp {
                new_sample.clamp(0.0, 1.0)
            } else {
                new_sample
            };
            old_sample * new_sample
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testsupport::BitWriter;

    #[test]
    fn replace_with_resets_canvas_is_two_bits() {
        // mode = kReplace via selector 0; resets_canvas suppresses source.
        let mut w = BitWriter::new();
        w.u32_field(0, 0, 0);
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let bi = read_blending_info(&mut r, false, true, "blending_info").expect("valid");

        assert_eq!(r.total_bits_read(), 2);
        assert_eq!(bi, BlendingInfo::default());
        assert_eq!(bi.mode, BlendMode::Replace);
    }

    #[test]
    fn source_is_read_when_the_canvas_is_not_reset() {
        let mut w = BitWriter::new();
        w.u32_field(0, 0, 0).u(2, 3);
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let bi = read_blending_info(&mut r, false, false, "blending_info").expect("valid");

        assert_eq!(r.total_bits_read(), 4);
        assert_eq!(bi.source, 3);
    }

    #[test]
    fn blend_mode_reads_alpha_and_clamp_only_with_extra_channels() {
        // mode = kBlend (2) with extra channels: alpha_channel then clamp.
        let mut w = BitWriter::new();
        w.u32_field(2, 0, 0).u32_field(1, 0, 0).bool(true).u(2, 0);
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let bi = read_blending_info(&mut r, true, false, "blending_info").expect("valid");

        assert_eq!(bi.mode, BlendMode::Blend);
        assert_eq!(bi.alpha_channel, 1);
        assert!(bi.clamp);

        // Without extra channels neither field is present.
        let mut w = BitWriter::new();
        w.u32_field(2, 0, 0).u(2, 0);
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let bi = read_blending_info(&mut r, false, false, "blending_info").expect("valid");

        assert_eq!(r.total_bits_read(), 4);
        assert_eq!(bi.alpha_channel, 0);
        assert!(!bi.clamp);
    }

    #[test]
    fn mul_reads_clamp_but_not_alpha_channel() {
        // kMul is 4 => selector 3 with payload 1 (3 + 1).
        let mut w = BitWriter::new();
        w.u32_field(3, 2, 1).bool(true).u(2, 1);
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let bi = read_blending_info(&mut r, true, false, "blending_info").expect("valid");

        assert_eq!(bi.mode, BlendMode::Mul);
        assert_eq!(bi.alpha_channel, 0, "kMul has no alpha_channel row");
        assert!(bi.clamp);
        assert_eq!(bi.source, 1);
    }

    #[test]
    fn add_reads_neither_alpha_nor_clamp() {
        let mut w = BitWriter::new();
        w.u32_field(1, 0, 0).u(2, 2);
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let bi = read_blending_info(&mut r, true, false, "blending_info").expect("valid");

        assert_eq!(bi.mode, BlendMode::Add);
        assert_eq!(r.total_bits_read(), 4);
        assert_eq!(bi.source, 2);
    }

    #[test]
    fn wide_alpha_channel_index() {
        // alpha_channel selector 3 => 3 + u(3); payload 5 => 8.
        let mut w = BitWriter::new();
        w.u32_field(2, 0, 0).u32_field(3, 3, 5).bool(false).u(2, 0);
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let bi = read_blending_info(&mut r, true, false, "blending_info").expect("valid");
        assert_eq!(bi.alpha_channel, 8);
    }

    #[test]
    fn mode_outside_the_table_rejected() {
        // 3 + u(2) reaches 6, which Table F.8 does not define.
        let mut w = BitWriter::new();
        w.u32_field(3, 2, 3);
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let err = read_blending_info(&mut r, false, true, "blending_info")
            .expect_err("6 is not a blend mode");
        assert!(matches!(err, FrameError::FieldOutOfRange { .. }));
    }

    #[test]
    fn every_defined_mode_round_trips() {
        for m in [
            BlendMode::Replace,
            BlendMode::Add,
            BlendMode::Blend,
            BlendMode::MulAdd,
            BlendMode::Mul,
        ] {
            assert_eq!(BlendMode::from_value(m.value()), Some(m));
        }
        assert_eq!(BlendMode::from_value(5), None);
    }

    #[test]
    fn peeking_the_mode_does_not_consume() {
        let mut w = BitWriter::new();
        w.u32_field(2, 0, 0).u(2, 0);
        let data = w.finish_padded(1);
        let r = BitReader::new(&data);
        assert_eq!(peek_blend_mode(&r).expect("valid"), BlendMode::Blend);
        assert_eq!(r.total_bits_read(), 0);
    }

    // ----------------------------------------------------------------
    // F.2 — Table F.8
    // ----------------------------------------------------------------

    fn ctx(mode: BlendMode, associated: bool, is_alpha_itself: bool) -> BlendContext {
        BlendContext {
            info: BlendingInfo {
                mode,
                alpha_channel: 0,
                clamp: false,
                source: 0,
            },
            alpha_associated: associated,
            is_alpha_itself,
        }
    }

    #[test]
    fn table_f8_colour_rows_are_hand_computed() {
        // old = 0.25, new = 0.75, old_alpha = 0.5, new_alpha = 0.5.
        let (o, n, oa, na) = (0.25f32, 0.75f32, 0.5f32, 0.5f32);
        assert_eq!(
            blend_sample(&ctx(BlendMode::Replace, false, false), o, n, oa, na),
            0.75
        );
        assert_eq!(
            blend_sample(&ctx(BlendMode::Add, false, false), o, n, oa, na),
            1.0
        );
        assert_eq!(
            blend_sample(&ctx(BlendMode::Mul, false, false), o, n, oa, na),
            0.1875
        );
        // kMulAdd: old + new_alpha * new = 0.25 + 0.5 * 0.75.
        assert_eq!(
            blend_sample(&ctx(BlendMode::MulAdd, false, false), o, n, oa, na),
            0.625
        );
        // kBlend premultiplied: new + old * (1 - new_alpha) = 0.75 + 0.125.
        assert_eq!(
            blend_sample(&ctx(BlendMode::Blend, true, false), o, n, oa, na),
            0.875
        );
        // kBlend unassociated: alpha = 0.5 + 0.5 * 0.5 = 0.75;
        // (0.5 * 0.75 + 0.5 * 0.25 * 0.5) / 0.75 = 0.4375 / 0.75.
        let want = (0.5f32 * 0.75 + 0.5 * 0.25 * 0.5) / 0.75;
        assert!(
            (blend_sample(&ctx(BlendMode::Blend, false, false), o, n, oa, na) - want).abs() < 1e-7
        );
    }

    #[test]
    fn the_alpha_channel_has_its_own_two_formulas() {
        // The trap: using the generic kBlend/kMulAdd formulas on the alpha
        // channel itself. Table F.8 replaces both.
        let (o, n, oa, na) = (0.5f32, 0.25f32, 0.5f32, 0.25f32);
        // kBlend on alpha: old_alpha + new_alpha * (1 - old_alpha).
        assert_eq!(
            blend_sample(&ctx(BlendMode::Blend, false, true), o, n, oa, na),
            0.5 + 0.25 * 0.5
        );
        // ... and the premultiplied flag does not change it.
        assert_eq!(
            blend_sample(&ctx(BlendMode::Blend, true, true), o, n, oa, na),
            0.5 + 0.25 * 0.5
        );
        // kMulAdd on alpha: the source frame's value is preserved.
        assert_eq!(
            blend_sample(&ctx(BlendMode::MulAdd, false, true), o, n, oa, na),
            oa
        );
    }

    #[test]
    fn clamp_applies_to_alpha_except_under_kmul() {
        // F.2: kBlend/kMulAdd clamp new_alpha; kMul clamps new_sample instead.
        let mut c = ctx(BlendMode::MulAdd, false, false);
        c.info.clamp = true;
        // new_alpha 2.0 clamps to 1.0, so old + 1.0 * new.
        assert_eq!(blend_sample(&c, 0.25, 0.5, 0.0, 2.0), 0.75);
        c.info.clamp = false;
        assert_eq!(blend_sample(&c, 0.25, 0.5, 0.0, 2.0), 1.25);

        let mut m = ctx(BlendMode::Mul, false, false);
        m.info.clamp = true;
        // new_sample 3.0 clamps to 1.0; the (huge) alpha is not consulted.
        assert_eq!(blend_sample(&m, 0.5, 3.0, 0.0, 9.0), 0.5);
        m.info.clamp = false;
        assert_eq!(blend_sample(&m, 0.5, 3.0, 0.0, 9.0), 1.5);
    }

    #[test]
    fn a_fully_transparent_unassociated_blend_is_zero_not_nan() {
        // Both alphas zero makes the unassociated quotient 0/0. Every term of
        // the numerator is zero, so zero is the only value that is not an
        // invention -- and a NaN here would poison the whole canvas.
        let v = blend_sample(&ctx(BlendMode::Blend, false, false), 0.5, 0.75, 0.0, 0.0);
        assert_eq!(v, 0.0);
    }

    #[test]
    fn alpha_one_and_zero_are_the_endpoints_of_an_unassociated_blend() {
        // Orientation check: at new_alpha 1 the new sample wins outright, at
        // new_alpha 0 the old one does (when it is opaque).
        let over = ctx(BlendMode::Blend, false, false);
        assert_eq!(blend_sample(&over, 0.25, 0.75, 1.0, 1.0), 0.75);
        assert_eq!(blend_sample(&over, 0.25, 0.75, 1.0, 0.0), 0.25);
    }

    #[test]
    fn an_unwritten_canvas_reads_as_zeroes() {
        // F.2: "if no frame was previously stored, the source frame is assumed
        // to have all sample values set to zeroes."
        let c = Canvas::zeros(2, 3, 3, 1);
        assert_eq!(c.colour_at(2, 1, 2), 0.0);
        assert_eq!(c.extra_at(0, 1, 2), 0.0);
        // Out of bounds is zero too, not a panic.
        assert_eq!(c.colour_at(0, 9, 9), 0.0);
        assert_eq!(c.extra_at(7, 0, 0), 0.0);
    }
}
