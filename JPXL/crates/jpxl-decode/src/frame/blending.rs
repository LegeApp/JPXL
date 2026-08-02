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
}
