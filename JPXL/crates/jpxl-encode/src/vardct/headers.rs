//! `ImageMetadata` and `FrameHeader` for a kVarDCT frame (18181-1 D.3, F.2).
//!
//! # Why `all_default` *is* usable here, unlike the modular track
//!
//! [`crate::headers`] explains at length why a modular frame cannot use
//! Table D.3's one-bit `all_default`: the defaults describe an **XYB-encoded
//! sRGB 8-bit** image, and a non-XYB frame breaks the very first row. This
//! slice encodes exactly that default image, so the whole bundle collapses to
//! two bits — `all_default`, then the unconditional `default_m` — and every
//! row it stands for is one this encoder wanted anyway:
//!
//! | Row | Default | Wanted |
//! |---|---|---|
//! | `extra_fields` | false | no orientation, preview, animation, tone mapping |
//! | `bit_depth` | 8-bit integer | the source is RGB8 |
//! | `modular_16bit_buffers` | true | no modular colour data at all |
//! | `num_extra` | 0 | no extra channels |
//! | `xyb_encoded` | true | the point of the slice |
//! | `colour_encoding` | sRGB, D65, kRelative | the source is sRGB |
//! | `default_m` | set | the L.2.1 opsin matrix and the K.2 weights |
//!
//! `tone_mapping` is guarded by `extra_fields`, so `intensity_target` keeps its
//! nominal 255 and L.2.2's `itscale` is exactly 1 — which is what makes the
//! forward opsin transform in `jpxl-core` the exact inverse of what the
//! decoder will run.
//!
//! # The frame header is not defaultable
//!
//! Table F.2's `all_default` would give a kVarDCT frame already, but it also
//! gives `flags == 0` and the Table J.1 restoration defaults, and this slice
//! needs neither: **`kSkipAdaptiveLFSmoothing` is set** and both filters are
//! off. So every row is written out.
//!
//! `kSkipAdaptiveLFSmoothing` is the load-bearing one. I.5.2's smoothing pass
//! is a 3x3 weighted average over the whole frame's LF image with a
//! data-dependent gate; an encoder that left it on would have to *invert* it to
//! know what LF integers reconstruct to, and there is no reason to: the flag
//! exists precisely so an encoder can decline. Setting it makes LF
//! quantization a per-block scalar problem with an exact inverse.
//!
//! # `group_size_shift` is absent, not chosen
//!
//! Table F.2 reads `group_size_shift` only when `encoding == kModular`. A
//! kVarDCT frame therefore always has `group_dim == 256` and an LF group of
//! 2048x2048, whatever a plan would like. [`write_frame_header`] rejects a
//! request for anything else rather than writing a field that does not exist.

use jpxl_bitstream::{BitWriter, U32Dist, U32Spec};

use crate::error::{EncodeError, Result};
use crate::headers::write_size_header;
use crate::vardct::plan::RestorationDecision;

/// 18181-1 D.1: the signature as a `u(16)` value, i.e. `FF 0A` read LSB-first.
const SIGNATURE: u32 = 0x0AFF;

/// 18181-1 F.2: `U32(1, 2, 4, 8)` for `upsampling`.
const UPSAMPLING_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(1),
    U32Dist::Val(2),
    U32Dist::Val(4),
    U32Dist::Val(8),
]);

/// 18181-1 F.6: `U32(1, 2, 3, 4 + u(3))` for `num_passes`.
const NUM_PASSES_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(1),
    U32Dist::Val(2),
    U32Dist::Val(3),
    U32Dist::BitsOffset { bits: 3, offset: 4 },
]);

/// 18181-1 F.8: `U32(0, 1, 2, 3 + u(2))` for `blending_info.mode`.
const BLEND_MODE_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(0),
    U32Dist::Val(1),
    U32Dist::Val(2),
    U32Dist::BitsOffset { bits: 2, offset: 3 },
]);

/// 18181-1 F.2: `U32(0, u(4), 16 + u(5), 48 + u(10))` for `name_len`.
const NAME_LEN_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(0),
    U32Dist::bits(4),
    U32Dist::BitsOffset {
        bits: 5,
        offset: 16,
    },
    U32Dist::BitsOffset {
        bits: 10,
        offset: 48,
    },
]);

/// Table F.5's `kSkipAdaptiveLFSmoothing` bit.
pub const FLAG_SKIP_ADAPTIVE_LF_SMOOTHING: u64 = 0x80;

/// The only `group_size_shift` a kVarDCT frame can have: F.2 does not read the
/// field outside kModular, so `group_dim` is `128 << 1 == 256`.
pub const VARDCT_GROUP_SIZE_SHIFT: u32 = 1;

/// Largest value F.2's `u(3)` `x_qm_scale` / `b_qm_scale` fields can carry.
const MAX_QM_SCALE: u32 = 7;

/// I.5.3's per-channel HF scale exponents, as signalled.
///
/// I.5.3 multiplies the X and B channels by `pow(0.8, x_qm_scale - 2)` and
/// `pow(0.8, b_qm_scale - 2)`. Writing `2` for both makes each factor exactly
/// `1`, so the dequantization matrices of I.2.5 are the only frequency shaping
/// in the frame. That is a *choice*, not a default — Table F.2's default for
/// `x_qm_scale` in a kVarDCT XYB frame is 3 — and it is made because a slice
/// whose job is to prove the pipeline should have one fewer scalar that can be
/// wrong on only one side. Rate/quality tuning of these two fields belongs to
/// the adaptive-quantization slice.
pub const NEUTRAL_QM_SCALE: u32 = 2;

/// Writes the signature, `SizeHeader` and `ImageMetadata` of an XYB-encoded
/// 8-bit sRGB image (18181-1 D.1, D.2, D.3).
///
/// # Errors
///
/// [`EncodeError::ValueOutOfRange`] for a zero or oversized dimension, or a bit
/// writer error.
pub fn write_image_headers(w: &mut BitWriter, width: u32, height: u32) -> Result<()> {
    w.write_bits(16, SIGNATURE)?;
    write_size_header(w, width, height)?;
    w.write_bool(true); // ImageMetadata all_default — see the module docs
    w.write_bool(true); // default_m, read even under all_default
    Ok(())
}

/// Writes the `FrameHeader` of the single kVarDCT frame (18181-1 Table F.2).
///
/// The writer must be byte-aligned on entry: F.1 aligns every frame.
///
/// `x_qm_scale` and `b_qm_scale` are I.5.3's per-channel exponents; pass
/// [`NEUTRAL_QM_SCALE`] for both unless the caller has a reason not to.
/// `restoration` is Table J.1; the production default is filters off
/// ([`RestorationDecision::default`]).
///
/// # Errors
///
/// [`EncodeError::ValueOutOfRange`] if a `qm_scale` exceeds the `u(3)` field
/// or `epf_iters > 3`.
pub fn write_frame_header(
    w: &mut BitWriter,
    x_qm_scale: u32,
    b_qm_scale: u32,
    restoration: RestorationDecision,
) -> Result<()> {
    for (what, value) in [("x_qm_scale", x_qm_scale), ("b_qm_scale", b_qm_scale)] {
        if value > MAX_QM_SCALE {
            return Err(EncodeError::ValueOutOfRange {
                what,
                value: i64::from(value),
            });
        }
    }
    if restoration.epf_iters > 3 {
        return Err(EncodeError::ValueOutOfRange {
            what: "epf_iters",
            value: i64::from(restoration.epf_iters),
        });
    }

    w.write_bool(false); // all_default
    w.write_bits(2, 0)?; // frame_type = kRegularFrame (Table F.3)
    w.write_bits(1, 0)?; // encoding = kVarDCT (Table F.4)
    w.write_u64(FLAG_SKIP_ADAPTIVE_LF_SMOOTHING)?; // flags (Table F.5)

    // metadata.xyb_encoded is true, so do_YCbCr is absent, and so is
    // jpeg_upsampling with it.
    w.write_u32(&UPSAMPLING_SPEC, 1)?;
    // num_extra is 0, so no ec_upsampling entries.
    // encoding != kModular, so group_size_shift is absent: group_dim is 256.

    w.write_bits(3, x_qm_scale)?;
    w.write_bits(3, b_qm_scale)?;

    w.write_u32(&NUM_PASSES_SPEC, 1)?; // Passes: one pass, nothing else stored
    w.write_bool(false); // have_crop

    // BlendingInfo (Table F.7): kReplace over a full frame, so resets_canvas is
    // true and `source` is suppressed; num_extra is 0 so alpha and clamp are
    // absent too.
    w.write_u32(&BLEND_MODE_SPEC, 0)?;
    w.write_bool(true); // is_last

    // is_last, so no save_as_reference; can_reference is false, so no
    // save_before_ct.
    w.write_u32(&NAME_LEN_SPEC, 0)?;

    write_restoration_filter(w, restoration)?;

    w.write_u64(0)?; // frame extensions
    Ok(())
}

/// Writes Table J.1 with default weights (no custom gab/epf tables).
fn write_restoration_filter(w: &mut BitWriter, restoration: RestorationDecision) -> Result<()> {
    // Always non-all_default so we can force gab/epf off without taking the
    // Table J.1 defaults (gab=true, epf_iters=2).
    w.write_bool(false); // all_default
    w.write_bool(restoration.gaborish); // gab
    if restoration.gaborish {
        w.write_bool(false); // gab_custom — use Table J.1 defaults
    }
    w.write_bits(2, u32::from(restoration.epf_iters))?;
    if restoration.epf_iters != 0 {
        // kVarDCT: epf_sharp_custom, epf_weight_custom, epf_sigma_custom all false
        // when present (gated on !all_default and epf_iters).
        w.write_bool(false); // epf_sharp_custom
        w.write_bool(false); // epf_weight_custom
        w.write_bool(false); // epf_sigma_custom
        // epf_sigma_for_modular is kModular-only — absent here.
    }
    w.write_u64(0)?; // restoration-filter extensions
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use jpxl_bitstream::BitReader;
    use jpxl_core::limits::{AllocGuard, Limits};
    use jpxl_decode::frame::{Encoding, FrameType, read_frame_header};
    use jpxl_decode::headers::decode_image_headers;

    #[test]
    fn the_image_headers_are_two_bits_of_metadata_and_decode_as_xyb_srgb() {
        for (width, height) in [(1u32, 1u32), (8, 8), (13, 7), (600, 520)] {
            let mut w = BitWriter::new();
            write_image_headers(&mut w, width, height).expect("headers");
            let written = w.bit_len();
            w.zero_pad_to_byte();
            let bytes = w.into_bytes();

            let mut r = BitReader::new(&bytes);
            let parsed =
                decode_image_headers(&mut r, &Limits::default()).expect("valid image headers");
            assert_eq!(
                r.total_bits_read(),
                written,
                "{width}x{height}: field shift"
            );
            assert_eq!((parsed.width(), parsed.height()), (width, height));
            assert!(parsed.metadata.xyb_encoded);
            assert!(!parsed.metadata.colour_encoding.is_grey());
            assert_eq!(parsed.metadata.bit_depth.bits_per_sample(), 8);
            assert!(parsed.metadata.ec_info.is_empty());
            assert!(parsed.metadata.default_m);
            assert!(parsed.metadata.preview.is_none());
            assert!(parsed.metadata.animation.is_none());
        }
    }

    fn xyb_metadata() -> jpxl_decode::headers::ImageMetadata {
        let mut w = BitWriter::new();
        write_image_headers(&mut w, 64, 64).expect("headers");
        w.zero_pad_to_byte();
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        decode_image_headers(&mut r, &Limits::default())
            .expect("valid")
            .metadata
    }

    #[test]
    fn the_frame_header_round_trips_as_kvardct_with_smoothing_and_filters_off() {
        let metadata = xyb_metadata();
        let mut w = BitWriter::new();
        write_frame_header(
            &mut w,
            NEUTRAL_QM_SCALE,
            NEUTRAL_QM_SCALE,
            RestorationDecision::default(),
        )
        .expect("header");
        let written = w.bit_len();
        w.zero_pad_to_byte();
        let bytes = w.into_bytes();

        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let mut r = BitReader::new(&bytes);
        let header =
            read_frame_header(&mut r, &metadata, 600, 520, &limits, &mut guard).expect("valid");

        assert_eq!(r.total_bits_read(), written, "no field shifted");
        assert_eq!(header.frame_type, FrameType::RegularFrame);
        assert_eq!(header.encoding, Encoding::VarDct);
        assert!(header.is_last);
        assert!(!header.have_crop);
        assert_eq!(header.upsampling, 1);
        assert_eq!(header.passes.num_passes, 1);
        assert_eq!(header.x_qm_scale, NEUTRAL_QM_SCALE);
        assert_eq!(header.b_qm_scale, NEUTRAL_QM_SCALE);
        assert!(
            !header.flags.adaptive_lf_smoothing(),
            "I.5.2 smoothing must be declined, or LF quantization has no exact inverse"
        );
        assert!(!header.flags.patches());
        assert!(!header.flags.noise());
        assert!(!header.flags.use_lf_frame());
        assert!(!header.restoration_filter.gab);
        assert!(!header.restoration_filter.epf_enabled());
        // F.2 does not read group_size_shift outside kModular.
        assert_eq!(header.group_size_shift, VARDCT_GROUP_SIZE_SHIFT);
        assert_eq!(header.group_dim().expect("valid").get(), 256);
    }

    #[test]
    fn frame_header_with_default_gab_and_epf_parses() {
        let metadata = xyb_metadata();
        let mut w = BitWriter::new();
        write_frame_header(
            &mut w,
            NEUTRAL_QM_SCALE,
            NEUTRAL_QM_SCALE,
            RestorationDecision {
                gaborish: true,
                epf_iters: 2,
            },
        )
        .expect("header");
        w.zero_pad_to_byte();
        let bytes = w.into_bytes();
        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let mut r = BitReader::new(&bytes);
        let header =
            read_frame_header(&mut r, &metadata, 64, 64, &limits, &mut guard).expect("valid");
        assert!(header.restoration_filter.gab);
        assert_eq!(header.restoration_filter.epf.iters, 2);
        assert!(!header.restoration_filter.gab_custom);
    }

    #[test]
    fn an_unrepresentable_qm_scale_is_rejected() {
        let mut w = BitWriter::new();
        assert!(matches!(
            write_frame_header(&mut w, 8, 2, RestorationDecision::default()),
            Err(EncodeError::ValueOutOfRange { .. })
        ));
        assert!(matches!(
            write_frame_header(&mut w, 2, 8, RestorationDecision::default()),
            Err(EncodeError::ValueOutOfRange { .. })
        ));
    }
}
