//! The `FrameHeader` bundle (18181-1 F.2, Table F.2).
//!
//! This is the most heavily conditioned table in the standard. Rows depend on
//! the frame's own earlier fields, on [`ImageMetadata`], and on four derived
//! predicates that F.2 names at the bottom of the table:
//!
//! ```text
//! normal_frame  = frame_type == kRegularFrame or frame_type == kSkipProgressive
//! full_frame    = !have_crop, or the crop rectangle covers the whole image
//! resets_canvas = full_frame and blending_info.mode == kReplace
//! can_reference = !is_last and (duration == 0 or save_as_reference != 0)
//!                 and frame_type != kLFFrame
//! ```
//!
//! `full_frame` is evaluated against the *image* dimensions, so those are a
//! parameter of [`read_frame_header`].
//!
//! # Fields whose guard is not `!all_default`
//!
//! Three rows are guarded by something other than `!all_default`, and each is
//! safe only because its guard is false under the defaults:
//!
//! * `jpeg_upsampling[3]` is guarded by `do_YCbCr and !flags.kUseLfFrame`, and
//!   `do_YCbCr` defaults to false.
//! * `group_size_shift` is guarded by `encoding == kModular`, and `encoding`
//!   defaults to `kVarDCT`.
//! * `lf_level` is guarded by `frame_type == kLFFrame`, and `frame_type`
//!   defaults to `kRegularFrame`.
//! * `name[name_len]` has a blank condition, but `name_len` defaults to 0.
//!
//! So an `all_default` frame header is exactly one bit, matching the worked
//! example of a minimal codestream in the JPEG XL overview paper.

use jpxl_bitstream::{BitReader, U32Dist, U32Spec, read_bool, read_u32, trace_field};
use jpxl_core::geometry::GroupDim;
use jpxl_core::limits::{AllocGuard, Limits};

use crate::frame::blending::{BlendMode, BlendingInfo, peek_blend_mode, read_blending_info};
use crate::frame::error::{FrameError, Result};
use crate::frame::passes::{Passes, read_passes};
use crate::frame::restoration::{RestorationFilter, read_restoration_filter};
use crate::headers::ImageMetadata;
use crate::headers::extensions::{Extensions, read_extensions};

/// 18181-1 F.2: `U32(1, 2, 4, 8)` for `upsampling` and `ec_upsampling`.
const UPSAMPLING_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(1),
    U32Dist::Val(2),
    U32Dist::Val(4),
    U32Dist::Val(8),
]);

/// 18181-1 F.2: `U32(u(8), 256 + u(11), 2304 + u(14), 18688 + u(30))`.
///
/// Used for `ux0`, `uy0`, `width` and `height` of a crop.
const CROP_SPEC: U32Spec = U32Spec::new([
    U32Dist::bits(8),
    U32Dist::BitsOffset {
        bits: 11,
        offset: 256,
    },
    U32Dist::BitsOffset {
        bits: 14,
        offset: 2304,
    },
    U32Dist::BitsOffset {
        bits: 30,
        offset: 18688,
    },
]);

/// 18181-1 F.2: `U32(0, 1, u(8), u(32))` for `duration`.
const DURATION_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(0),
    U32Dist::Val(1),
    U32Dist::bits(8),
    U32Dist::bits(32),
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

/// `duration` value meaning "present the next frame as the next page".
pub const DURATION_NEXT_PAGE: u32 = 0xFFFF_FFFF;

/// Frame type (18181-1 Table F.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum FrameType {
    /// Part of the decoded sequence of frames.
    #[default]
    RegularFrame = 0,
    /// The LF of a future frame; not itself part of the sequence.
    LfFrame = 1,
    /// Only a source for patches or blending; not part of the sequence.
    ReferenceOnly = 2,
    /// Like a regular frame, but decoders do not render progressive previews.
    SkipProgressive = 3,
}

impl FrameType {
    /// Maps a `u(2)` value to a row of Table F.3.
    #[must_use]
    pub const fn from_value(value: u32) -> Option<Self> {
        Some(match value {
            0 => Self::RegularFrame,
            1 => Self::LfFrame,
            2 => Self::ReferenceOnly,
            3 => Self::SkipProgressive,
            _ => return None,
        })
    }

    /// The value as stored in the codestream.
    #[must_use]
    pub const fn value(self) -> u32 {
        self as u32
    }

    /// F.2's `normal_frame`: `kRegularFrame` or `kSkipProgressive`.
    #[must_use]
    pub const fn is_normal_frame(self) -> bool {
        matches!(self, Self::RegularFrame | Self::SkipProgressive)
    }
}

/// Frame encoding mode (18181-1 Table F.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum Encoding {
    /// Var-DCT mode: the decoder performs an IDCT on varblocks.
    #[default]
    VarDct = 0,
    /// Modular mode: a signalled chain of inverse transforms.
    Modular = 1,
}

impl Encoding {
    /// Maps a `u(1)` value to a row of Table F.4.
    #[must_use]
    pub const fn from_value(value: u32) -> Option<Self> {
        Some(match value {
            0 => Self::VarDct,
            1 => Self::Modular,
            _ => return None,
        })
    }
}

/// Feature flags (18181-1 Table F.5).
///
/// Note the gap: bits 2 and 3 are unassigned, and
/// [`SKIP_ADAPTIVE_LF_SMOOTHING`](FrameFlags::SKIP_ADAPTIVE_LF_SMOOTHING) is
/// *inverted* — the bit being set means the stage is skipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FrameFlags(pub u64);

impl FrameFlags {
    /// `kNoise`: enable the noise feature stage.
    pub const NOISE: u64 = 1;
    /// `kPatches`: enable the patch feature stage.
    pub const PATCHES: u64 = 2;
    /// `kSplines`: enable the spline feature stage.
    pub const SPLINES: u64 = 16;
    /// `kUseLfFrame`: use a previously decoded LF frame.
    pub const USE_LF_FRAME: u64 = 32;
    /// `kSkipAdaptiveLFSmoothing`: set means *disable* adaptive LF smoothing.
    pub const SKIP_ADAPTIVE_LF_SMOOTHING: u64 = 128;

    /// Whether a given flag bit is set.
    #[must_use]
    pub const fn has(self, flag: u64) -> bool {
        self.0 & flag != 0
    }

    /// Whether the noise stage runs.
    #[must_use]
    pub const fn noise(self) -> bool {
        self.has(Self::NOISE)
    }

    /// Whether the patch stage runs.
    #[must_use]
    pub const fn patches(self) -> bool {
        self.has(Self::PATCHES)
    }

    /// Whether the spline stage runs.
    #[must_use]
    pub const fn splines(self) -> bool {
        self.has(Self::SPLINES)
    }

    /// Whether LF data comes from a previously decoded LF frame.
    #[must_use]
    pub const fn use_lf_frame(self) -> bool {
        self.has(Self::USE_LF_FRAME)
    }

    /// Whether adaptive LF smoothing runs (the flag is inverted).
    #[must_use]
    pub const fn adaptive_lf_smoothing(self) -> bool {
        !self.has(Self::SKIP_ADAPTIVE_LF_SMOOTHING)
    }
}

/// A decoded `FrameHeader` bundle (18181-1 Table F.2).
#[derive(Debug, Clone, PartialEq)]
pub struct FrameHeader {
    /// Whether the guarded part of the bundle took its defaults.
    pub all_default: bool,
    /// Frame type.
    pub frame_type: FrameType,
    /// VarDCT or Modular.
    pub encoding: Encoding,
    /// Feature flags.
    pub flags: FrameFlags,
    /// Whether colour samples are stored as YCbCr.
    pub do_ycbcr: bool,
    /// Per-channel JPEG chroma subsampling selectors.
    pub jpeg_upsampling: [u32; 3],
    /// Colour-channel upsampling factor: 1, 2, 4 or 8.
    pub upsampling: u32,
    /// Per-extra-channel upsampling factors.
    pub ec_upsampling: Vec<u32>,
    /// `group_size_shift`; `group_dim` is `128 << group_size_shift`.
    pub group_size_shift: u32,
    /// X-channel quantization-matrix scale.
    pub x_qm_scale: u32,
    /// B-channel quantization-matrix scale.
    pub b_qm_scale: u32,
    /// Pass structure.
    pub passes: Passes,
    /// LF level, nonzero only for `kLFFrame`.
    pub lf_level: u32,
    /// Whether the frame carries an explicit rectangle.
    pub have_crop: bool,
    /// Crop origin x, from `UnpackSigned(ux0)`; may be negative.
    pub x0: i32,
    /// Crop origin y, from `UnpackSigned(uy0)`; may be negative.
    pub y0: i32,
    /// Frame width; equals the image width when `!have_crop`.
    pub width: u32,
    /// Frame height; equals the image height when `!have_crop`.
    pub height: u32,
    /// Colour-channel blending.
    pub blending_info: BlendingInfo,
    /// Per-extra-channel blending.
    pub ec_blending_info: Vec<BlendingInfo>,
    /// Presentation duration in ticks.
    pub duration: u32,
    /// SMPTE timecode, or 0.
    pub timecode: u32,
    /// Whether this is the last frame.
    pub is_last: bool,
    /// Reference slot to save this frame into, or 0.
    pub save_as_reference: u32,
    /// Whether to save before the inverse colour transform.
    pub save_before_ct: bool,
    /// Raw frame-name bytes; the standard says to interpret them as UTF-8.
    pub name: Vec<u8>,
    /// Restoration filter parameters.
    pub restoration_filter: RestorationFilter,
    /// Frame-level extensions; payloads are skipped.
    pub extensions: Extensions,
}

impl Default for FrameHeader {
    /// The `all_default` state: a regular, VarDCT-encoded, final frame.
    fn default() -> Self {
        Self {
            all_default: true,
            frame_type: FrameType::RegularFrame,
            encoding: Encoding::VarDct,
            flags: FrameFlags(0),
            do_ycbcr: false,
            jpeg_upsampling: [0; 3],
            upsampling: 1,
            ec_upsampling: Vec::new(),
            group_size_shift: 1,
            x_qm_scale: 3,
            b_qm_scale: 2,
            passes: Passes::default(),
            lf_level: 0,
            have_crop: false,
            x0: 0,
            y0: 0,
            width: 0,
            height: 0,
            blending_info: BlendingInfo::default(),
            ec_blending_info: Vec::new(),
            duration: 0,
            timecode: 0,
            is_last: true,
            save_as_reference: 0,
            save_before_ct: false,
            name: Vec::new(),
            restoration_filter: RestorationFilter::default(),
            extensions: Extensions::default(),
        }
    }
}

impl FrameHeader {
    /// F.2's `normal_frame`.
    #[must_use]
    pub const fn is_normal_frame(&self) -> bool {
        self.frame_type.is_normal_frame()
    }

    /// F.2's `full_frame`, evaluated against the given image dimensions.
    #[must_use]
    pub fn is_full_frame(&self, image_width: u32, image_height: u32) -> bool {
        full_frame(
            self.have_crop,
            self.x0,
            self.y0,
            self.width,
            self.height,
            image_width,
            image_height,
        )
    }

    /// F.2's `can_reference`.
    #[must_use]
    pub const fn can_reference(&self) -> bool {
        !self.is_last
            && (self.duration == 0 || self.save_as_reference != 0)
            && !matches!(self.frame_type, FrameType::LfFrame)
    }

    /// `group_dim` as a checked [`GroupDim`] (`128 << group_size_shift`).
    ///
    /// # Errors
    ///
    /// [`FrameError::FieldOutOfRange`] if `group_size_shift` exceeds 3, which
    /// a `u(2)` field cannot produce but a hand-built header could.
    pub fn group_dim(&self) -> Result<GroupDim> {
        u8::try_from(self.group_size_shift)
            .ok()
            .and_then(|s| GroupDim::from_shift(s).ok())
            .ok_or_else(|| {
                FrameError::out_of_range(
                    "group_size_shift",
                    "F.2",
                    u64::from(self.group_size_shift),
                )
            })
    }

    /// The frame name as UTF-8, or `None` if the bytes are not valid UTF-8.
    ///
    /// As with extra-channel names (D.3.6), UTF-8 validity is not stated as a
    /// conformance requirement, so the raw bytes are preserved.
    #[must_use]
    pub fn name_utf8(&self) -> Option<&str> {
        core::str::from_utf8(&self.name).ok()
    }

    /// Whether this frame is presented as the next page of a multi-page image.
    #[must_use]
    pub const fn is_next_page(&self) -> bool {
        self.duration == DURATION_NEXT_PAGE
    }
}

/// `UnpackSigned(u)`: `u / 2` if even, `-(u + 1) / 2` if odd.
///
/// Duplicated from the colour-encoding module so the frame code does not
/// depend on a header-internal helper.
#[must_use]
const fn unpack_signed(u: u32) -> i32 {
    if u.is_multiple_of(2) {
        0i32.wrapping_add_unsigned(u / 2)
    } else {
        0i32.wrapping_sub_unsigned(u / 2 + 1)
    }
}

/// F.2's `full_frame` predicate.
fn full_frame(
    have_crop: bool,
    x0: i32,
    y0: i32,
    width: u32,
    height: u32,
    image_width: u32,
    image_height: u32,
) -> bool {
    if !have_crop {
        return true;
    }
    // "the frame area given by width and height and offsets x0 and y0
    // completely covers the image area". Compare in i64 so a negative origin
    // plus a large extent cannot wrap.
    let x0 = i64::from(x0);
    let y0 = i64::from(y0);
    x0 <= 0
        && y0 <= 0
        && x0 + i64::from(width) >= i64::from(image_width)
        && y0 + i64::from(height) >= i64::from(image_height)
}

/// Reads a `FrameHeader` bundle (18181-1 Table F.2).
///
/// `metadata` supplies the conditions the table takes from the image header
/// (`xyb_encoded`, the extra-channel count, animation presence). The image
/// dimensions are needed to evaluate `full_frame`.
///
/// # Errors
///
/// [`FrameError::FieldOutOfRange`] for any field outside its clause's range,
/// [`FrameError::Core`] if the frame name exceeds the allocation budget, or a
/// bitstream error on truncation.
pub fn read_frame_header(
    reader: &mut BitReader<'_>,
    metadata: &ImageMetadata,
    image_width: u32,
    image_height: u32,
    limits: &Limits,
    guard: &mut AllocGuard,
) -> Result<FrameHeader> {
    let _ = limits;
    let num_extra = metadata.num_extra();
    let extra = num_extra >= 1;

    let all_default = trace_field!(reader, "frame.all_default", read_bool(reader))?;
    let mut header = FrameHeader {
        all_default,
        width: image_width,
        height: image_height,
        ..FrameHeader::default()
    };

    if all_default {
        // Every remaining row is guarded, directly or through a default that
        // makes its guard false; see the module documentation.
        header.x_qm_scale = default_x_qm_scale(metadata, header.encoding);
        return Ok(header);
    }

    let raw_type = trace_field!(reader, "frame.frame_type", reader.read_bits(2))?;
    header.frame_type = FrameType::from_value(raw_type)
        .ok_or_else(|| FrameError::out_of_range("frame_type", "F.3", u64::from(raw_type)))?;

    let raw_encoding = trace_field!(reader, "frame.encoding", reader.read_bits(1))?;
    header.encoding = Encoding::from_value(raw_encoding)
        .ok_or_else(|| FrameError::out_of_range("encoding", "F.4", u64::from(raw_encoding)))?;

    header.flags = FrameFlags(trace_field!(
        reader,
        "frame.flags",
        jpxl_bitstream::read_u64(reader)
    )?);

    if !metadata.xyb_encoded {
        header.do_ycbcr = trace_field!(reader, "frame.do_YCbCr", read_bool(reader))?;
    }

    let use_lf_frame = header.flags.use_lf_frame();

    if header.do_ycbcr && !use_lf_frame {
        for slot in &mut header.jpeg_upsampling {
            *slot = trace_field!(reader, "frame.jpeg_upsampling", reader.read_bits(2))?;
        }
    }

    if !use_lf_frame {
        header.upsampling = trace_field!(
            reader,
            "frame.upsampling",
            read_u32(reader, &UPSAMPLING_SPEC)
        )?;
        // Bounded by num_extra, which ImageMetadata already metered.
        header.ec_upsampling = Vec::with_capacity(num_extra);
        for _ in 0..num_extra {
            header.ec_upsampling.push(trace_field!(
                reader,
                "frame.ec_upsampling",
                read_u32(reader, &UPSAMPLING_SPEC)
            )?);
        }
    } else {
        header.ec_upsampling = vec![1; num_extra];
    }

    if header.encoding == Encoding::Modular {
        header.group_size_shift =
            trace_field!(reader, "frame.group_size_shift", reader.read_bits(2))?;
    }

    header.x_qm_scale = default_x_qm_scale(metadata, header.encoding);
    if metadata.xyb_encoded && header.encoding == Encoding::VarDct {
        header.x_qm_scale = trace_field!(reader, "frame.x_qm_scale", reader.read_bits(3))?;
        header.b_qm_scale = trace_field!(reader, "frame.b_qm_scale", reader.read_bits(3))?;
    }

    if header.frame_type != FrameType::ReferenceOnly {
        header.passes = read_passes(reader)?;
    }

    if header.frame_type == FrameType::LfFrame {
        header.lf_level = trace_field!(reader, "frame.lf_level", reader.read_bits(2))? + 1;
    }

    if header.frame_type != FrameType::LfFrame {
        header.have_crop = trace_field!(reader, "frame.have_crop", read_bool(reader))?;
    }

    if header.have_crop {
        if header.frame_type != FrameType::ReferenceOnly {
            let ux0 = trace_field!(reader, "frame.ux0", read_u32(reader, &CROP_SPEC))?;
            let uy0 = trace_field!(reader, "frame.uy0", read_u32(reader, &CROP_SPEC))?;
            header.x0 = unpack_signed(ux0);
            header.y0 = unpack_signed(uy0);
        }
        header.width = trace_field!(reader, "frame.width", read_u32(reader, &CROP_SPEC))?;
        header.height = trace_field!(reader, "frame.height", read_u32(reader, &CROP_SPEC))?;
    }

    let normal_frame = header.is_normal_frame();
    let is_full_frame = full_frame(
        header.have_crop,
        header.x0,
        header.y0,
        header.width,
        header.height,
        image_width,
        image_height,
    );

    if normal_frame {
        // resets_canvas depends on this bundle's own mode, so the mode is
        // peeked before the bundle is read; see the blending module for why
        // one value then governs every ec_blending_info too.
        let mode = peek_blend_mode(reader)?;
        let resets_canvas = is_full_frame && mode == BlendMode::Replace;

        header.blending_info =
            read_blending_info(reader, extra, resets_canvas, "frame.blending_info")?;
        header.ec_blending_info = Vec::with_capacity(num_extra);
        for _ in 0..num_extra {
            header.ec_blending_info.push(read_blending_info(
                reader,
                extra,
                resets_canvas,
                "frame.ec_blending_info",
            )?);
        }

        if metadata.animation.is_some() {
            header.duration =
                trace_field!(reader, "frame.duration", read_u32(reader, &DURATION_SPEC))?;
        }
        if metadata
            .animation
            .is_some_and(|animation| animation.have_timecodes)
        {
            header.timecode = trace_field!(reader, "frame.timecode", reader.read_bits(32))?;
        }

        header.is_last = trace_field!(reader, "frame.is_last", read_bool(reader))?;
    } else {
        // Default is_last is "!frame_type", i.e. true only for kRegularFrame.
        header.is_last = header.frame_type == FrameType::RegularFrame;
    }

    if header.frame_type != FrameType::LfFrame && !header.is_last {
        header.save_as_reference =
            trace_field!(reader, "frame.save_as_reference", reader.read_bits(2))?;
    }

    // resets_canvas is re-evaluated here against the now-known blending mode.
    let resets_canvas = is_full_frame && header.blending_info.mode == BlendMode::Replace;
    header.save_before_ct = !normal_frame;
    if header.frame_type == FrameType::ReferenceOnly || (resets_canvas && header.can_reference()) {
        header.save_before_ct = trace_field!(reader, "frame.save_before_ct", read_bool(reader))?;
    }

    let name_len = trace_field!(reader, "frame.name_len", read_u32(reader, &NAME_LEN_SPEC))?;
    guard
        .charge(u64::from(name_len))
        .map_err(FrameError::Core)?;
    header.name = Vec::with_capacity(name_len as usize);
    for _ in 0..name_len {
        let byte = trace_field!(reader, "frame.name", reader.read_bits(8))?;
        header
            .name
            .push(u8::try_from(byte).map_err(|_| FrameError::out_of_range("name", "F.2", 0))?);
    }

    let (restoration_filter, restoration_extensions) =
        read_restoration_filter(reader, header.encoding)?;
    header.restoration_filter = restoration_filter;
    let _ = restoration_extensions;

    header.extensions = read_extensions(reader)?;

    Ok(header)
}

/// F.2: `d_xqms = (metadata.xyb_encoded and encoding == kVarDCT ? 3 : 2)`.
const fn default_x_qm_scale(metadata: &ImageMetadata, encoding: Encoding) -> u32 {
    if metadata.xyb_encoded && matches!(encoding, Encoding::VarDct) {
        3
    } else {
        2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::headers::animation::AnimationHeader;
    use crate::testsupport::BitWriter;

    fn default_metadata() -> ImageMetadata {
        ImageMetadata::default()
    }

    fn read_with(
        bytes: &[u8],
        metadata: &ImageMetadata,
        image_width: u32,
        image_height: u32,
    ) -> Result<(FrameHeader, u64)> {
        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let mut r = BitReader::new(bytes);
        let header = read_frame_header(
            &mut r,
            metadata,
            image_width,
            image_height,
            &limits,
            &mut guard,
        )?;
        Ok((header, r.total_bits_read()))
    }

    #[test]
    fn all_default_frame_header_is_one_bit() {
        // The overview paper's minimal codestream spends exactly one bit here.
        let mut w = BitWriter::new();
        w.bool(true);
        let data = w.finish_padded(1);
        let (header, bits) = read_with(&data, &default_metadata(), 8, 8).expect("valid");

        assert_eq!(bits, 1);
        assert!(header.all_default);
        assert_eq!(header.frame_type, FrameType::RegularFrame);
        assert_eq!(header.encoding, Encoding::VarDct);
        assert!(header.is_last, "default is_last is !frame_type");
        assert_eq!(header.upsampling, 1);
        assert_eq!(header.group_size_shift, 1, "group_dim 256");
        assert_eq!(header.group_dim().expect("valid").get(), 256);
        assert_eq!(header.x_qm_scale, 3, "xyb_encoded and VarDCT => 3");
        assert_eq!(header.passes.num_passes, 1);
        assert!(header.restoration_filter.all_default);
    }

    /// Writes the minimal explicit (non-all_default) modular frame header.
    fn minimal_modular(w: &mut BitWriter) {
        w.bool(false) // all_default
            .u(2, 0) // frame_type = kRegularFrame
            .u(1, 1) // encoding = kModular
            .u64_field(0) // flags
            // metadata.xyb_encoded is true by default, so no do_YCbCr row.
            .u32_field(0, 0, 0) // upsampling = 1
            // no extra channels => no ec_upsampling
            .u(2, 1) // group_size_shift = 1
            // xyb_encoded but Modular => no x_qm_scale / b_qm_scale
            .u32_field(0, 0, 0) // passes.num_passes = 1
            .bool(false) // have_crop
            .u32_field(0, 0, 0) // blending_info.mode = kReplace (resets canvas)
            .bool(true) // is_last
            // is_last => no save_as_reference; not referenceable => no save_before_ct
            .u32_field(0, 0, 0) // name_len = 0
            .bool(true) // restoration_filter all_default
            .u64_field(0); // extensions
    }

    #[test]
    fn minimal_modular_frame() {
        let mut w = BitWriter::new();
        minimal_modular(&mut w);
        let expected = w.bit_len();
        let data = w.finish_padded(1);
        let (header, bits) = read_with(&data, &default_metadata(), 300, 200).expect("valid");

        assert_eq!(bits, expected);
        assert!(!header.all_default);
        assert_eq!(header.encoding, Encoding::Modular);
        assert_eq!(header.group_size_shift, 1);
        assert!(header.is_last);
        assert!(!header.have_crop);
        assert_eq!(
            (header.width, header.height),
            (300, 200),
            "without a crop the frame is the image"
        );
        assert_eq!(header.x_qm_scale, 2, "Modular => d_xqms is 2");
    }

    #[test]
    fn vardct_frame_reads_qm_scales_and_restoration_filters() {
        let mut w = BitWriter::new();
        w.bool(false)
            .u(2, 0) // kRegularFrame
            .u(1, 0) // kVarDCT
            .u64_field(0)
            .u32_field(0, 0, 0) // upsampling
            // VarDCT => no group_size_shift row
            .u(3, 5) // x_qm_scale
            .u(3, 1) // b_qm_scale
            .u32_field(0, 0, 0) // num_passes = 1
            .bool(false) // have_crop
            .u32_field(0, 0, 0) // blending mode kReplace
            .bool(true) // is_last
            .u32_field(0, 0, 0); // name_len
        // RestorationFilter: gab on with custom weights, EPF with 2 iters.
        w.bool(false) // all_default
            .bool(true) // gab
            .bool(true); // gab_custom
        for bits in [0x3C00u16, 0x4000, 0x3C00, 0x4000, 0x3C00, 0x4000] {
            w.f16_bits(bits);
        }
        w.u(2, 2) // epf_iters
            .bool(false) // epf_sharp_custom
            .bool(false) // epf_weight_custom
            .bool(false) // epf_sigma_custom
            .u64_field(0); // restoration extensions
        w.u64_field(0); // frame extensions

        let data = w.finish_padded(1);
        let (header, _) = read_with(&data, &default_metadata(), 512, 512).expect("valid");

        assert_eq!(header.encoding, Encoding::VarDct);
        assert_eq!(header.x_qm_scale, 5);
        assert_eq!(header.b_qm_scale, 1);
        assert_eq!(
            header.group_size_shift, 1,
            "VarDCT keeps the default group_dim of 256"
        );
        assert!(header.restoration_filter.gab_custom);
        assert_eq!(
            header.restoration_filter.gab_weights.weight1,
            [1.0, 1.0, 1.0]
        );
        assert_eq!(
            header.restoration_filter.gab_weights.weight2,
            [2.0, 2.0, 2.0]
        );
        assert!(header.restoration_filter.epf_enabled());
    }

    #[test]
    fn cropped_frame_with_negative_origin() {
        // ux0 = 3 => UnpackSigned(3) = -2; uy0 = 1 => -1.
        let mut w = BitWriter::new();
        w.bool(false)
            .u(2, 0)
            .u(1, 1) // Modular
            .u64_field(0)
            .u32_field(0, 0, 0) // upsampling
            .u(2, 1) // group_size_shift
            .u32_field(0, 0, 0) // num_passes
            .bool(true) // have_crop
            .u32_field(0, 8, 3) // ux0 = 3 => x0 = -2
            .u32_field(0, 8, 1) // uy0 = 1 => y0 = -1
            .u32_field(0, 8, 100) // width
            .u32_field(0, 8, 80) // height
            .u32_field(0, 0, 0) // blending mode kReplace
            .u(2, 0) // source: the crop does not cover the image
            .bool(true) // is_last
            .u32_field(0, 0, 0) // name_len
            .bool(true) // restoration all_default
            .u64_field(0);
        let data = w.finish_padded(1);
        let (header, _) = read_with(&data, &default_metadata(), 300, 200).expect("valid");

        assert!(header.have_crop);
        assert_eq!((header.x0, header.y0), (-2, -1));
        assert_eq!((header.width, header.height), (100, 80));
        assert!(
            !header.is_full_frame(300, 200),
            "a 100x80 crop cannot cover a 300x200 image"
        );
    }

    #[test]
    fn crop_covering_the_whole_image_is_a_full_frame() {
        assert!(full_frame(true, 0, 0, 300, 200, 300, 200));
        assert!(full_frame(true, -10, -10, 400, 300, 300, 200));
        assert!(!full_frame(true, 1, 0, 300, 200, 300, 200), "x0 > 0");
        assert!(!full_frame(true, 0, 0, 299, 200, 300, 200), "too narrow");
        assert!(full_frame(false, 0, 0, 0, 0, 300, 200), "no crop");
    }

    #[test]
    fn unpack_signed_matches_the_convention() {
        assert_eq!(unpack_signed(0), 0);
        assert_eq!(unpack_signed(1), -1);
        assert_eq!(unpack_signed(2), 1);
        assert_eq!(unpack_signed(3), -2);
        assert_eq!(unpack_signed(u32::MAX), i32::MIN);
    }

    #[test]
    fn animation_frame_reads_duration_and_timecode() {
        let mut metadata = default_metadata();
        metadata.animation = Some(AnimationHeader {
            tps_numerator: 100,
            tps_denominator: 1,
            num_loops: 0,
            have_timecodes: true,
        });

        let mut w = BitWriter::new();
        w.bool(false)
            .u(2, 0)
            .u(1, 1)
            .u64_field(0)
            .u32_field(0, 0, 0) // upsampling
            .u(2, 1) // group_size_shift
            .u32_field(0, 0, 0) // num_passes
            .bool(false) // have_crop
            .u32_field(0, 0, 0) // blending mode kReplace
            .u32_field(2, 8, 25) // duration = u(8) = 25
            .u(32, 0x01_02_03_04) // timecode
            .bool(false) // is_last = false
            .u(2, 1) // save_as_reference
            .bool(false) // save_before_ct (resets_canvas && can_reference)
            .u32_field(0, 0, 0) // name_len
            .bool(true)
            .u64_field(0);
        let data = w.finish_padded(1);
        let (header, _) = read_with(&data, &metadata, 300, 200).expect("valid");

        assert_eq!(header.duration, 25);
        assert_eq!(header.timecode, 0x0102_0304);
        assert!(!header.is_last);
        assert_eq!(header.save_as_reference, 1);
        assert!(header.can_reference());
    }

    #[test]
    fn no_animation_means_no_duration_row() {
        // The same header without animation metadata must not read duration.
        let mut w = BitWriter::new();
        w.bool(false)
            .u(2, 0)
            .u(1, 1)
            .u64_field(0)
            .u32_field(0, 0, 0)
            .u(2, 1)
            .u32_field(0, 0, 0)
            .bool(false)
            .u32_field(0, 0, 0) // blending mode
            .bool(true) // is_last  <- straight after blending
            .u32_field(0, 0, 0)
            .bool(true)
            .u64_field(0);
        let expected = w.bit_len();
        let data = w.finish_padded(1);
        let (header, bits) = read_with(&data, &default_metadata(), 300, 200).expect("valid");

        assert_eq!(bits, expected);
        assert_eq!(header.duration, 0);
        assert!(header.is_last);
    }

    #[test]
    fn do_ycbcr_only_when_not_xyb_encoded() {
        let mut metadata = default_metadata();
        metadata.xyb_encoded = false;

        let mut w = BitWriter::new();
        w.bool(false)
            .u(2, 0)
            .u(1, 1) // Modular
            .u64_field(0)
            .bool(true) // do_YCbCr
            .u(2, 0)
            .u(2, 1)
            .u(2, 1) // jpeg_upsampling[3]
            .u32_field(0, 0, 0) // upsampling
            .u(2, 1) // group_size_shift
            .u32_field(0, 0, 0) // num_passes
            .bool(false)
            .u32_field(0, 0, 0)
            .bool(true)
            .u32_field(0, 0, 0)
            .bool(true)
            .u64_field(0);
        let data = w.finish_padded(1);
        let (header, _) = read_with(&data, &metadata, 300, 200).expect("valid");

        assert!(header.do_ycbcr);
        assert_eq!(header.jpeg_upsampling, [0, 1, 1], "4:2:0 subsampling");
    }

    #[test]
    fn use_lf_frame_suppresses_upsampling_rows() {
        // flags = kUseLfFrame (32) removes upsampling and ec_upsampling.
        let mut w = BitWriter::new();
        w.bool(false)
            .u(2, 0)
            .u(1, 1)
            .u64_field(32) // flags = kUseLfFrame
            .u(2, 1) // group_size_shift (Modular)
            .u32_field(0, 0, 0) // num_passes
            .bool(false) // have_crop
            .u32_field(0, 0, 0) // blending mode
            .bool(true) // is_last
            .u32_field(0, 0, 0)
            .bool(true)
            .u64_field(0);
        let expected = w.bit_len();
        let data = w.finish_padded(1);
        let (header, bits) = read_with(&data, &default_metadata(), 300, 200).expect("valid");

        assert_eq!(bits, expected);
        assert!(header.flags.use_lf_frame());
        assert_eq!(header.upsampling, 1);
    }

    #[test]
    fn lf_frame_reads_lf_level_and_no_crop() {
        let mut w = BitWriter::new();
        w.bool(false)
            .u(2, 1) // frame_type = kLFFrame
            .u(1, 1) // Modular
            .u64_field(0)
            .u32_field(0, 0, 0) // upsampling
            .u(2, 1) // group_size_shift
            .u32_field(0, 0, 0) // num_passes
            .u(2, 1) // lf_level = 1 + 1 = 2
            // kLFFrame => no have_crop row, not a normal frame => no blending
            .u(2, 0) // save_as_reference?  frame_type == kLFFrame => skipped
            .u32_field(0, 0, 0); // name_len
        // Rewrite: kLFFrame skips save_as_reference, so back the bits out by
        // building the exact sequence instead.
        let mut w = BitWriter::new();
        w.bool(false)
            .u(2, 1) // kLFFrame
            .u(1, 1) // Modular
            .u64_field(0)
            .u32_field(0, 0, 0) // upsampling
            .u(2, 1) // group_size_shift
            .u32_field(0, 0, 0) // num_passes
            .u(2, 1) // lf_level
            .u32_field(0, 0, 0) // name_len
            .bool(true) // restoration all_default
            .u64_field(0); // extensions
        let expected = w.bit_len();
        let data = w.finish_padded(1);
        let (header, bits) = read_with(&data, &default_metadata(), 300, 200).expect("valid");

        assert_eq!(bits, expected);
        assert_eq!(header.frame_type, FrameType::LfFrame);
        assert_eq!(header.lf_level, 2);
        assert!(!header.have_crop);
        assert!(
            !header.is_last,
            "default is_last is !frame_type, and kLFFrame is nonzero"
        );
        assert!(!header.can_reference(), "kLFFrame is never referenceable");
        assert!(header.save_before_ct, "default is !normal_frame");
    }

    #[test]
    fn reference_only_frame_skips_passes_and_crop_origin() {
        let mut w = BitWriter::new();
        w.bool(false)
            .u(2, 2) // kReferenceOnly
            .u(1, 1) // Modular
            .u64_field(0)
            .u32_field(0, 0, 0) // upsampling
            .u(2, 1) // group_size_shift
            // kReferenceOnly => no Passes bundle
            .bool(true) // have_crop
            // kReferenceOnly => no ux0/uy0
            .u32_field(0, 8, 64) // width
            .u32_field(0, 8, 64) // height
            // not a normal frame => no blending/duration/is_last
            .u(2, 2) // save_as_reference (is_last defaults false)
            .bool(true) // save_before_ct (frame_type == kReferenceOnly)
            .u32_field(0, 0, 0) // name_len
            .bool(true)
            .u64_field(0);
        let expected = w.bit_len();
        let data = w.finish_padded(1);
        let (header, bits) = read_with(&data, &default_metadata(), 300, 200).expect("valid");

        assert_eq!(bits, expected);
        assert_eq!(header.frame_type, FrameType::ReferenceOnly);
        assert_eq!(header.passes, Passes::default());
        assert_eq!((header.x0, header.y0), (0, 0));
        assert_eq!((header.width, header.height), (64, 64));
        assert_eq!(header.save_as_reference, 2);
        assert!(header.save_before_ct);
    }

    #[test]
    fn extra_channels_add_upsampling_and_blending_rows() {
        let mut metadata = default_metadata();
        metadata.ec_info = vec![
            crate::headers::extra_channels::ExtraChannelInfo::default_alpha(),
            crate::headers::extra_channels::ExtraChannelInfo::default_alpha(),
        ];

        let mut w = BitWriter::new();
        w.bool(false)
            .u(2, 0)
            .u(1, 1)
            .u64_field(0)
            .u32_field(0, 0, 0) // upsampling
            .u32_field(1, 0, 0) // ec_upsampling[0] = 2
            .u32_field(0, 0, 0) // ec_upsampling[1] = 1
            .u(2, 1) // group_size_shift
            .u32_field(0, 0, 0) // num_passes
            .bool(false) // have_crop
            .u32_field(0, 0, 0) // blending_info.mode = kReplace
            .u32_field(0, 0, 0) // ec_blending_info[0].mode
            .u32_field(0, 0, 0) // ec_blending_info[1].mode
            .bool(true) // is_last
            .u32_field(0, 0, 0)
            .bool(true)
            .u64_field(0);
        let expected = w.bit_len();
        let data = w.finish_padded(1);
        let (header, bits) = read_with(&data, &metadata, 300, 200).expect("valid");

        assert_eq!(bits, expected);
        assert_eq!(header.ec_upsampling, vec![2, 1]);
        assert_eq!(header.ec_blending_info.len(), 2);
    }

    #[test]
    fn frame_name_is_read_and_kept_raw() {
        let mut w = BitWriter::new();
        w.bool(false)
            .u(2, 0)
            .u(1, 1)
            .u64_field(0)
            .u32_field(0, 0, 0)
            .u(2, 1)
            .u32_field(0, 0, 0)
            .bool(false)
            .u32_field(0, 0, 0)
            .bool(true) // is_last
            .u32_field(1, 4, 5); // name_len = 5
        for b in b"frame" {
            w.u(8, u32::from(*b));
        }
        w.bool(true).u64_field(0);
        let data = w.finish_padded(1);
        let (header, _) = read_with(&data, &default_metadata(), 300, 200).expect("valid");

        assert_eq!(header.name_utf8(), Some("frame"));
    }

    #[test]
    fn frame_name_allocation_is_metered() {
        let mut w = BitWriter::new();
        w.bool(false)
            .u(2, 0)
            .u(1, 1)
            .u64_field(0)
            .u32_field(0, 0, 0)
            .u(2, 1)
            .u32_field(0, 0, 0)
            .bool(false)
            .u32_field(0, 0, 0)
            .bool(true)
            .u32_field(3, 10, 100); // name_len = 148
        for _ in 0..148 {
            w.u(8, u32::from(b'x'));
        }
        w.bool(true).u64_field(0);
        let data = w.finish_padded(1);

        let limits = Limits {
            max_alloc_bytes: 16,
            ..Limits::default()
        };
        let mut guard = AllocGuard::new(&limits);
        let mut r = BitReader::new(&data);
        assert!(
            read_frame_header(&mut r, &default_metadata(), 300, 200, &limits, &mut guard).is_err()
        );
    }

    #[test]
    fn flag_bit_positions_match_the_examples() {
        // F.5 EXAMPLE 1: flags == 18 enables splines and patches.
        let flags = FrameFlags(18);
        assert!(flags.patches());
        assert!(flags.splines());
        assert!(!flags.noise());
        assert!(!flags.use_lf_frame());

        // F.5 EXAMPLE 2: flags == 3 => kPatches is 1.
        let flags = FrameFlags(3);
        assert!(flags.noise());
        assert!(flags.patches());

        // The smoothing flag is inverted.
        assert!(FrameFlags(0).adaptive_lf_smoothing());
        assert!(!FrameFlags(FrameFlags::SKIP_ADAPTIVE_LF_SMOOTHING).adaptive_lf_smoothing());
    }

    #[test]
    fn frame_type_and_encoding_tables() {
        for t in [
            FrameType::RegularFrame,
            FrameType::LfFrame,
            FrameType::ReferenceOnly,
            FrameType::SkipProgressive,
        ] {
            assert_eq!(FrameType::from_value(t.value()), Some(t));
        }
        assert_eq!(FrameType::from_value(4), None);
        assert!(FrameType::RegularFrame.is_normal_frame());
        assert!(FrameType::SkipProgressive.is_normal_frame());
        assert!(!FrameType::LfFrame.is_normal_frame());
        assert!(!FrameType::ReferenceOnly.is_normal_frame());

        assert_eq!(Encoding::from_value(0), Some(Encoding::VarDct));
        assert_eq!(Encoding::from_value(1), Some(Encoding::Modular));
        assert_eq!(Encoding::from_value(2), None);
    }

    #[test]
    fn group_size_shift_maps_to_group_dim() {
        for (shift, dim) in [(0u32, 128u32), (1, 256), (2, 512), (3, 1024)] {
            let header = FrameHeader {
                group_size_shift: shift,
                ..FrameHeader::default()
            };
            assert_eq!(header.group_dim().expect("valid").get(), dim);
        }
        let header = FrameHeader {
            group_size_shift: 4,
            ..FrameHeader::default()
        };
        assert!(header.group_dim().is_err());
    }

    #[test]
    fn truncated_header_errors() {
        assert!(read_with(&[], &default_metadata(), 8, 8).is_err());
        assert!(read_with(&[0x00], &default_metadata(), 8, 8).is_err());
    }
}
