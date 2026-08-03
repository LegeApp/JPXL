//! `FrameHeader` and TOC for a single modular frame (18181-1 F.2, F.3).
//!
//! # Why `all_default` is not usable here either
//!
//! An `all_default` frame header is one bit, but its defaults say `kVarDCT`.
//! A modular frame must set `encoding`, which un-defaults the whole bundle, so
//! every row is written explicitly. Two of them are not the Table F.2 default:
//!
//! * `encoding = kModular`;
//! * `restoration_filter` is written with `gab` off and `epf_iters` zero.
//!
//! The second is the one that would silently cost losslessness. The Table J.1
//! defaults enable the Gabor-like filter and two EPF iterations, and both are
//! *decoder-side* smoothing stages: a conforming decoder would apply them to a
//! bit-exactly reconstructed modular image and hand back different pixels. The
//! filters are therefore disabled in the codestream, not assumed away.
//!
//! # The single-section frame
//!
//! F.3.1 gives a frame one TOC entry when `num_groups == 1` and
//! `num_passes == 1`, and then every structure of Table F.1 is carried
//! consecutively in that one section. Choosing `group_size_shift` so the image
//! fits in one group is what buys that, and it is why this encoder never has to
//! implement the LF-group / pass-group split.

use jpxl_bitstream::{BitWriter, U32Dist, U32Spec};

use crate::error::{EncodeError, Result};

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

/// 18181-1 F.3.3: `U32(u(10), 1024 + u(14), 17408 + u(22), 4211712 + u(30))`.
const TOC_ENTRY_SPEC: U32Spec = U32Spec::new([
    U32Dist::bits(10),
    U32Dist::BitsOffset {
        bits: 14,
        offset: 1024,
    },
    U32Dist::BitsOffset {
        bits: 22,
        offset: 17408,
    },
    U32Dist::BitsOffset {
        bits: 30,
        offset: 4_211_712,
    },
]);

/// Largest side length a single group can cover: `128 << 3`.
pub const MAX_SINGLE_GROUP_DIM: u32 = 1024;

/// The smallest `group_size_shift` (F.2) whose `group_dim = 128 << shift`
/// covers both dimensions, so the frame is one group and therefore one section.
///
/// # Errors
///
/// [`EncodeError::Unsupported`] if either dimension exceeds
/// [`MAX_SINGLE_GROUP_DIM`]; multi-group encoding is slice 10.
pub fn single_group_size_shift(width: u32, height: u32) -> Result<u32> {
    let side = width.max(height);
    for shift in 0..=3u32 {
        if side <= 128u32 << shift {
            return Ok(shift);
        }
    }
    Err(EncodeError::unsupported(
        "an image larger than one group (multi-group encoding)",
        "F.3.1",
    ))
}

/// Writes the `FrameHeader` of the single modular frame (18181-1 Table F.2).
///
/// The reader must be byte-aligned before this is called: F.1 aligns every
/// frame with `ZeroPadToByte()`.
///
/// # Errors
///
/// [`EncodeError::ValueOutOfRange`] for a `group_size_shift` above 3, or a bit
/// writer error.
pub fn write_frame_header(w: &mut BitWriter, group_size_shift: u32) -> Result<()> {
    if group_size_shift > 3 {
        return Err(EncodeError::ValueOutOfRange {
            what: "group_size_shift",
            value: i64::from(group_size_shift),
        });
    }

    w.write_bool(false); // all_default
    w.write_bits(2, 0)?; // frame_type = kRegularFrame (Table F.3)
    w.write_bits(1, 1)?; // encoding = kModular (Table F.4)
    w.write_u64(0)?; // flags: no patches, splines, noise or LF frame

    // metadata.xyb_encoded is false, so do_YCbCr is present.
    w.write_bool(false);
    // do_YCbCr is false, so jpeg_upsampling is absent.

    w.write_u32(&UPSAMPLING_SPEC, 1)?;
    // num_extra is 0, so no ec_upsampling entries.

    w.write_bits(2, group_size_shift)?; // present because encoding == kModular
    // xyb_encoded is false, so x_qm_scale / b_qm_scale are absent.

    w.write_u32(&NUM_PASSES_SPEC, 1)?; // Passes: one pass, nothing else stored
    // frame_type is not kLFFrame, so lf_level is absent and have_crop present.
    w.write_bool(false); // have_crop

    // BlendingInfo (Table F.7): kReplace. num_extra is 0 so alpha_channel and
    // clamp are absent, and the frame is full with kReplace, which makes
    // resets_canvas true and suppresses `source`.
    w.write_u32(&BLEND_MODE_SPEC, 0)?;
    // No animation, so no duration or timecode.
    w.write_bool(true); // is_last

    // is_last, so no save_as_reference. can_reference is false (is_last), and
    // the frame is not kReferenceOnly, so save_before_ct is absent too.
    w.write_u32(&NAME_LEN_SPEC, 0)?;

    write_restoration_filter_off(w)?;
    w.write_u64(0)?; // frame extensions
    Ok(())
}

/// Writes a `RestorationFilter` bundle with every stage disabled (18181-1
/// Table J.1).
fn write_restoration_filter_off(w: &mut BitWriter) -> Result<()> {
    w.write_bool(false); // all_default — the defaults enable gab and EPF
    w.write_bool(false); // gab
    // gab is false, so gab_custom and the weights are absent.
    w.write_bits(2, 0)?; // epf_iters = 0, which suppresses every EPF field
    w.write_u64(0)?; // restoration-filter extensions
    Ok(())
}

/// Writes a one-entry TOC (18181-1 F.3).
///
/// On return the writer is byte-aligned and the next byte is where the single
/// section begins.
///
/// # Errors
///
/// [`EncodeError::ValueOutOfRange`] if the section is larger than the TOC entry
/// distribution can express.
pub fn write_single_entry_toc(w: &mut BitWriter, section_len: usize) -> Result<()> {
    let len = u32::try_from(section_len).map_err(|_| EncodeError::ValueOutOfRange {
        what: "section length",
        value: i64::try_from(section_len).unwrap_or(i64::MAX),
    })?;

    w.write_bool(false); // permuted_toc
    w.zero_pad_to_byte(); // F.3.3, before the entries
    w.write_u32(&TOC_ENTRY_SPEC, len)?;
    w.zero_pad_to_byte(); // F.3.3, after the entries
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use jpxl_bitstream::BitReader;
    use jpxl_core::limits::{AllocGuard, Limits};
    use jpxl_decode::frame::{Encoding, FrameType, read_frame_header, read_toc};
    use jpxl_decode::headers::ImageMetadata;

    fn grey8_metadata() -> ImageMetadata {
        ImageMetadata {
            xyb_encoded: false,
            ..ImageMetadata::default()
        }
    }

    #[test]
    fn frame_header_round_trips_and_disables_every_filter() {
        let mut w = BitWriter::new();
        write_frame_header(&mut w, 2).expect("header");
        let written = w.bit_len();
        w.zero_pad_to_byte();
        let bytes = w.into_bytes();

        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let mut r = BitReader::new(&bytes);
        let header = read_frame_header(&mut r, &grey8_metadata(), 300, 200, &limits, &mut guard)
            .expect("valid frame header");

        assert_eq!(r.total_bits_read(), written, "no field shifted");
        assert_eq!(header.frame_type, FrameType::RegularFrame);
        assert_eq!(header.encoding, Encoding::Modular);
        assert!(header.is_last);
        assert!(!header.have_crop);
        assert_eq!(header.upsampling, 1);
        assert_eq!(header.passes.num_passes, 1);
        assert_eq!(header.group_size_shift, 2);
        assert_eq!(header.group_dim().expect("valid").get(), 512);
        assert!(
            !header.restoration_filter.gab,
            "gab would alter losslessly decoded samples"
        );
        assert!(
            !header.restoration_filter.epf_enabled(),
            "EPF would alter losslessly decoded samples"
        );
        assert!(header.name.is_empty());
    }

    #[test]
    fn toc_round_trips_and_ends_byte_aligned() {
        for len in [0usize, 1, 1023, 1024, 17_407, 17_408, 1 << 20] {
            let mut w = BitWriter::new();
            // Start unaligned, so the pads have something to do.
            w.write_bits(3, 0b101).expect("prefix");
            write_single_entry_toc(&mut w, len).expect("toc");
            assert!(w.is_byte_aligned());
            let bytes = w.into_bytes();

            let limits = Limits::default();
            let mut guard = AllocGuard::new(&limits);
            let mut r = BitReader::new(&bytes);
            r.read_bits(3).expect("prefix");
            let toc = read_toc(&mut r, 1, &limits, &mut guard).expect("valid toc");
            assert_eq!(toc.len(), 1);
            assert_eq!(toc.total_size(), len as u64, "entry {len}");
            assert_eq!(toc.offset_of(0), Some(0));
            assert!(r.total_bits_read().is_multiple_of(8));
        }
    }

    #[test]
    fn group_size_shift_is_the_smallest_that_makes_one_group() {
        assert_eq!(single_group_size_shift(1, 1).expect("valid"), 0);
        assert_eq!(single_group_size_shift(128, 128).expect("valid"), 0);
        assert_eq!(single_group_size_shift(129, 1).expect("valid"), 1);
        assert_eq!(single_group_size_shift(13, 7).expect("valid"), 0);
        assert_eq!(single_group_size_shift(256, 300).expect("valid"), 2);
        assert_eq!(single_group_size_shift(1024, 1024).expect("valid"), 3);
        assert!(matches!(
            single_group_size_shift(1025, 8),
            Err(EncodeError::Unsupported { .. })
        ));
    }
}
