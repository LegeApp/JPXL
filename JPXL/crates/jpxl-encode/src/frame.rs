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
//! # Sections
//!
//! F.3.1 gives a frame one TOC entry when `num_groups == 1` and
//! `num_passes == 1`, and then every structure of Table F.1 is carried
//! consecutively in that one section. Otherwise the TOC has one entry for
//! `LfGlobal`, one per LF group, one for `HfGlobal`, and one per pass group.
//! [`Geometry`] computes that grid; the LF-group and `HfGlobal` sections are
//! always empty here, because nothing this encoder writes produces a channel
//! with `hshift >= 3` (that needs Squeeze) and `HfGlobal` is VarDCT-only.

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

/// Largest `group_size_shift` the two-bit F.2 field can carry.
pub const MAX_GROUP_SIZE_SHIFT: u32 = 3;

/// The `group_size_shift` used when the caller does not name one.
///
/// F.3.1 constrains nothing here: any of the four legal group sizes decodes,
/// and the choice is purely the encoder's. 512 keeps every image up to
/// 512x512 in a single section — the cheapest shape — while larger images get
/// a group grid rather than one enormous group, which is what makes the
/// multi-section path exercised by ordinary input rather than only by tests.
pub const DEFAULT_GROUP_SIZE_SHIFT: u32 = 2;

/// The group grid of a frame (18181-1 F.2, F.3.1, G.2, G.4).
///
/// Deliberately recomputed here rather than borrowed from `jpxl-decode`: an
/// encoder that shares the decoder's geometry cannot detect a geometry bug by
/// round-tripping against it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geometry {
    width: u32,
    height: u32,
    group_size_shift: u32,
    group_dim: u32,
    groups_x: u32,
    groups_y: u32,
    lf_groups_x: u32,
    lf_groups_y: u32,
}

impl Geometry {
    /// Builds the grid for a `width` x `height` frame at `group_size_shift`.
    ///
    /// # Errors
    ///
    /// [`EncodeError::ValueOutOfRange`] for a zero dimension or a shift above
    /// [`MAX_GROUP_SIZE_SHIFT`].
    pub fn new(width: u32, height: u32, group_size_shift: u32) -> Result<Self> {
        if group_size_shift > MAX_GROUP_SIZE_SHIFT {
            return Err(EncodeError::ValueOutOfRange {
                what: "group_size_shift",
                value: i64::from(group_size_shift),
            });
        }
        for (what, value) in [("width", width), ("height", height)] {
            if value == 0 {
                return Err(EncodeError::ValueOutOfRange { what, value: 0 });
            }
        }
        let group_dim = 128u32 << group_size_shift;
        // An LF group covers 8x8 groups' worth of samples (G.2.3 NOTE).
        let lf_dim = u64::from(group_dim) * 8;
        Ok(Self {
            width,
            height,
            group_size_shift,
            group_dim,
            groups_x: width.div_ceil(group_dim),
            groups_y: height.div_ceil(group_dim),
            lf_groups_x: u32::try_from(u64::from(width).div_ceil(lf_dim)).unwrap_or(1),
            lf_groups_y: u32::try_from(u64::from(height).div_ceil(lf_dim)).unwrap_or(1),
        })
    }

    /// `group_size_shift` as written in the frame header.
    #[must_use]
    pub const fn group_size_shift(&self) -> u32 {
        self.group_size_shift
    }

    /// `group_dim = 128 << group_size_shift`.
    #[must_use]
    pub const fn group_dim(&self) -> u32 {
        self.group_dim
    }

    /// Number of pass groups (`num_groups` of F.3.1).
    #[must_use]
    pub const fn num_groups(&self) -> u64 {
        self.groups_x as u64 * self.groups_y as u64
    }

    /// Number of LF groups.
    #[must_use]
    pub const fn num_lf_groups(&self) -> u64 {
        self.lf_groups_x as u64 * self.lf_groups_y as u64
    }

    /// Whether F.3.1's single-section form applies (`num_passes` is always 1).
    #[must_use]
    pub const fn is_single_section(&self) -> bool {
        self.num_groups() == 1
    }

    /// Number of TOC entries (18181-1 F.3.1).
    #[must_use]
    pub const fn num_sections(&self) -> u64 {
        if self.is_single_section() {
            1
        } else {
            2 + self.num_lf_groups() + self.num_groups()
        }
    }

    /// The rectangle `(x0, y0, width, height)` covered by group `index` in
    /// raster order, or `None` past the grid.
    #[must_use]
    pub fn group_rect(&self, index: u64) -> Option<(u32, u32, u32, u32)> {
        if index >= self.num_groups() {
            return None;
        }
        let gx = u32::try_from(index % u64::from(self.groups_x)).ok()?;
        let gy = u32::try_from(index / u64::from(self.groups_x)).ok()?;
        let x0 = gx.checked_mul(self.group_dim)?;
        let y0 = gy.checked_mul(self.group_dim)?;
        Some((
            x0,
            y0,
            self.group_dim.min(self.width.saturating_sub(x0)),
            self.group_dim.min(self.height.saturating_sub(y0)),
        ))
    }
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

/// Writes the TOC (18181-1 F.3), one entry per section length, unpermuted.
///
/// On return the writer is byte-aligned and the next byte is where the first
/// section begins.
///
/// # Errors
///
/// [`EncodeError::ValueOutOfRange`] if a section is larger than the TOC entry
/// distribution can express.
pub fn write_toc(w: &mut BitWriter, section_lens: &[usize]) -> Result<()> {
    w.write_bool(false); // permuted_toc: sections are written in order
    w.zero_pad_to_byte(); // F.3.3, before the entries
    for &section_len in section_lens {
        let len = u32::try_from(section_len).map_err(|_| EncodeError::ValueOutOfRange {
            what: "section length",
            value: i64::try_from(section_len).unwrap_or(i64::MAX),
        })?;
        w.write_u32(&TOC_ENTRY_SPEC, len)?;
    }
    w.zero_pad_to_byte(); // F.3.3, after the entries
    Ok(())
}

#[cfg(test)]
#[allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]
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
            write_toc(&mut w, &[len]).expect("toc");
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
    fn a_multi_entry_toc_round_trips_with_its_offsets() {
        // Zero-length entries are the normal case for LfGroup and HfGlobal in
        // modular mode (F.3.1 NOTE 1), so they must survive the round trip.
        let lens = [40usize, 0, 0, 1500, 1500, 20_000, 3];
        let mut w = BitWriter::new();
        write_toc(&mut w, &lens).expect("toc");
        assert!(w.is_byte_aligned());
        let bytes = w.into_bytes();

        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let mut r = BitReader::new(&bytes);
        let toc = read_toc(&mut r, lens.len() as u64, &limits, &mut guard).expect("valid toc");
        assert_eq!(toc.len(), lens.len());
        assert_eq!(toc.total_size(), lens.iter().sum::<usize>() as u64);
        let mut offset = 0u64;
        for (i, &len) in lens.iter().enumerate() {
            assert_eq!(toc.offset_of(i), Some(offset), "section {i}");
            offset += len as u64;
        }
    }

    #[test]
    fn the_group_grid_tiles_the_frame_exactly() {
        let g = Geometry::new(600, 520, 2).expect("valid");
        assert_eq!(g.group_dim(), 512);
        assert_eq!(g.num_groups(), 4);
        assert_eq!(g.num_lf_groups(), 1);
        assert!(!g.is_single_section());
        assert_eq!(g.num_sections(), 2 + 1 + 4);
        assert_eq!(g.group_rect(0), Some((0, 0, 512, 512)));
        assert_eq!(g.group_rect(1), Some((512, 0, 88, 512)));
        assert_eq!(g.group_rect(2), Some((0, 512, 512, 8)));
        assert_eq!(g.group_rect(3), Some((512, 512, 88, 8)));
        assert_eq!(g.group_rect(4), None);

        // Every group rectangle covers each pixel exactly once, at every shift.
        for shift in 0..=MAX_GROUP_SIZE_SHIFT {
            for (w, h) in [(1u32, 1u32), (13, 7), (129, 260), (600, 520)] {
                let g = Geometry::new(w, h, shift).expect("valid");
                let mut covered = vec![0u8; (w * h) as usize];
                for index in 0..g.num_groups() {
                    let (x0, y0, gw, gh) = g.group_rect(index).expect("in range");
                    assert!(gw > 0 && gh > 0, "{w}x{h} shift {shift}: empty group");
                    for y in y0..y0 + gh {
                        for x in x0..x0 + gw {
                            covered[(y * w + x) as usize] += 1;
                        }
                    }
                }
                assert!(
                    covered.iter().all(|&c| c == 1),
                    "{w}x{h} shift {shift}: groups do not tile the frame"
                );
            }
        }
    }

    #[test]
    fn a_single_group_frame_has_one_section() {
        let g = Geometry::new(512, 512, 2).expect("valid");
        assert!(g.is_single_section());
        assert_eq!(g.num_sections(), 1);
        assert_eq!(g.group_rect(0), Some((0, 0, 512, 512)));
    }

    #[test]
    fn degenerate_geometry_is_rejected() {
        assert!(matches!(
            Geometry::new(0, 4, 2),
            Err(EncodeError::ValueOutOfRange { .. })
        ));
        assert!(matches!(
            Geometry::new(4, 4, 4),
            Err(EncodeError::ValueOutOfRange { .. })
        ));
    }
}
