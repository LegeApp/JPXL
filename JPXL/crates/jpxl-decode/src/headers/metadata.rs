//! The `ImageMetadata` bundle (18181-1 D.3.1) and `Orientation` (18181-1 D.3.2).
//!
//! ```text
//! Table D.3 — ImageMetadata bundle
//! condition                       type                     default  name
//!                                 Bool()                   true     all_default
//! !all_default                    Bool()                   false    extra_fields
//! extra_fields                    1 + u(3)                 1        orientation
//! extra_fields                    Bool()                   false    have_intr_size
//! have_intr_size                  SizeHeader                        intrinsic_size
//! extra_fields                    Bool()                   false    have_preview
//! have_preview                    PreviewHeader                     preview
//! extra_fields                    Bool()                   false    have_animation
//! have_animation                  AnimationHeader                   animation
//! !all_default                    BitDepth                          bit_depth
//! !all_default                    Bool()                   true     modular_16bit_buffers
//! !all_default                    U32(0,1,2+u(4),1+u(12))  0        num_extra
//! !all_default                    ExtraChannelInfo                  ec_info[num_extra]
//! !all_default                    Bool()                   true     xyb_encoded
//! !all_default                    ColourEncoding                    colour_encoding
//! extra_fields                    ToneMapping                       tone_mapping
//! !all_default                    Extensions                        extensions
//!                                 Bool()                            default_m
//! !default_m and xyb_encoded      OpsinInverseMatrix                opsin_inverse_matrix
//! !default_m                      u(3)                     0        cw_mask
//! BitSet(cw_mask, 1)              F16()                    d_up2    up2_weight[15]
//! BitSet(cw_mask, 2)              F16()                    d_up4    up4_weight[55]
//! BitSet(cw_mask, 4)              F16()                    d_up8    up8_weight[210]
//! ```
//!
//! # `default_m` is not covered by `all_default`
//!
//! The `default_m` row has a **blank** condition, so it is read even when
//! `all_default` is true. This is easy to get wrong — every other row in the
//! table is guarded — and getting it wrong shifts every subsequent bit by one.
//! It is corroborated by the worked example of a minimal codestream in the
//! JPEG XL overview paper, which spends one bit on the default image header
//! and then *another* bit to say "use the default XYB and upsampling weights".
//!
//! # `BitSet`
//!
//! The conventions clause defines `BitSet(u, b)` as `u & b`, so the three
//! upsampling rows test `cw_mask` against the **masks** 1, 2 and 4 — not
//! against bit indices.

use jpxl_bitstream::{
    BitReader, U32Dist, U32Spec, read_bool, read_f16_as_f32, read_u32, trace_field,
};
use jpxl_core::limits::{AllocGuard, Limits};

use crate::error::{DecodeError, Result};
use crate::headers::animation::{AnimationHeader, read_animation_header};
use crate::headers::bit_depth::{BitDepth, read_bit_depth};
use crate::headers::colour::{
    ColourEncoding, ToneMapping, read_colour_encoding, read_tone_mapping,
};
use crate::headers::extensions::{Extensions, read_extensions};
use crate::headers::extra_channels::{ExtraChannelInfo, read_extra_channel_info};
use crate::headers::opsin::{OpsinInverseMatrix, read_opsin_inverse_matrix};
use crate::headers::size::{PreviewHeader, SizeHeader, read_preview_header, read_size_header};

/// 18181-1 D.3: `U32(0, 1, 2 + u(4), 1 + u(12))`.
const NUM_EXTRA_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(0),
    U32Dist::Val(1),
    U32Dist::BitsOffset { bits: 4, offset: 2 },
    U32Dist::BitsOffset {
        bits: 12,
        offset: 1,
    },
]);

/// Number of custom weights for 2x upsampling (18181-1 D.3).
pub const UP2_WEIGHTS: usize = 15;
/// Number of custom weights for 4x upsampling (18181-1 D.3).
pub const UP4_WEIGHTS: usize = 55;
/// Number of custom weights for 8x upsampling (18181-1 D.3).
pub const UP8_WEIGHTS: usize = 210;

/// Image orientation (18181-1 D.3.2, Table D.4).
///
/// Values match JEITA CP-3451C (Exif 2.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum Orientation {
    /// First row top, first column left; no transform.
    #[default]
    Identity = 1,
    /// First row top, first column right; flip horizontally.
    FlipHorizontal = 2,
    /// First row bottom, first column right; rotate 180.
    Rotate180 = 3,
    /// First row bottom, first column left; flip vertically.
    FlipVertical = 4,
    /// First row left, first column top; transpose.
    Transpose = 5,
    /// First row right, first column top; rotate 90 clockwise.
    Rotate90Cw = 6,
    /// First row right, first column bottom; flip then rotate 90 clockwise.
    AntiTranspose = 7,
    /// First row left, first column bottom; rotate 90 counterclockwise.
    Rotate90Ccw = 8,
}

impl Orientation {
    /// Maps the stored value `1..=8` to a row of Table D.4.
    #[must_use]
    pub const fn from_value(value: u32) -> Option<Self> {
        Some(match value {
            1 => Self::Identity,
            2 => Self::FlipHorizontal,
            3 => Self::Rotate180,
            4 => Self::FlipVertical,
            5 => Self::Transpose,
            6 => Self::Rotate90Cw,
            7 => Self::AntiTranspose,
            8 => Self::Rotate90Ccw,
            _ => return None,
        })
    }

    /// The value as stored in the codestream.
    #[must_use]
    pub const fn value(self) -> u32 {
        self as u32
    }

    /// Whether applying this orientation swaps width and height.
    #[must_use]
    pub const fn swaps_axes(self) -> bool {
        matches!(
            self,
            Self::Transpose | Self::Rotate90Cw | Self::AntiTranspose | Self::Rotate90Ccw
        )
    }
}

/// Custom upsampling weights, when signalled by `cw_mask` (18181-1 D.3).
///
/// A `None` field means the corresponding default weight table from K.2
/// applies. Those tables are not transcribed at this slice; the flag records
/// faithfully whether the stream overrode them.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct UpsamplingWeights {
    /// Raw `cw_mask` value.
    pub cw_mask: u32,
    /// 15 custom weights for 2x upsampling.
    pub up2: Option<Vec<f32>>,
    /// 55 custom weights for 4x upsampling.
    pub up4: Option<Vec<f32>>,
    /// 210 custom weights for 8x upsampling.
    pub up8: Option<Vec<f32>>,
}

/// A decoded `ImageMetadata` bundle (18181-1 D.3.1).
#[derive(Debug, Clone, PartialEq)]
pub struct ImageMetadata {
    /// Whether the guarded part of the bundle took its defaults.
    pub all_default: bool,
    /// Whether the optional presentation fields were stored.
    pub extra_fields: bool,
    /// Orientation transform to apply after decoding.
    pub orientation: Orientation,
    /// Recommended display dimensions, if signalled.
    pub intrinsic_size: Option<SizeHeader>,
    /// Preview frame dimensions, if a preview frame is present.
    pub preview: Option<PreviewHeader>,
    /// Animation timing, if the codestream is an animation.
    pub animation: Option<AnimationHeader>,
    /// Sample representation of the colour channels.
    pub bit_depth: BitDepth,
    /// Whether 16-bit buffers suffice for modular sub-bitstreams.
    pub modular_16bit_buffers: bool,
    /// Per-extra-channel information; length is `num_extra`.
    pub ec_info: Vec<ExtraChannelInfo>,
    /// Whether the stored image is in the XYB colour space.
    pub xyb_encoded: bool,
    /// Colour encoding of the original image.
    pub colour_encoding: ColourEncoding,
    /// HDR tone-mapping information.
    pub tone_mapping: ToneMapping,
    /// Extension bundle; payloads are skipped.
    pub extensions: Extensions,
    /// Whether the default opsin matrix and upsampling weights apply.
    pub default_m: bool,
    /// Custom inverse opsin matrix, if signalled.
    pub opsin_inverse_matrix: Option<OpsinInverseMatrix>,
    /// Custom upsampling weights, if signalled.
    pub upsampling: UpsamplingWeights,
}

impl Default for ImageMetadata {
    /// The `all_default` + `default_m` state: an 8-bit sRGB still image with no
    /// extra channels, XYB-encoded.
    fn default() -> Self {
        Self {
            all_default: true,
            extra_fields: false,
            orientation: Orientation::Identity,
            intrinsic_size: None,
            preview: None,
            animation: None,
            bit_depth: BitDepth::default_int8(),
            modular_16bit_buffers: true,
            ec_info: Vec::new(),
            xyb_encoded: true,
            colour_encoding: ColourEncoding::default(),
            tone_mapping: ToneMapping::default(),
            extensions: Extensions::default(),
            default_m: true,
            opsin_inverse_matrix: None,
            upsampling: UpsamplingWeights::default(),
        }
    }
}

impl ImageMetadata {
    /// Number of extra channels.
    #[must_use]
    pub fn num_extra(&self) -> usize {
        self.ec_info.len()
    }

    /// The effective inverse opsin matrix: the signalled one, or the L.1
    /// defaults.
    #[must_use]
    pub fn effective_opsin_matrix(&self) -> OpsinInverseMatrix {
        self.opsin_inverse_matrix.unwrap_or_default()
    }
}

/// Reads an `ImageMetadata` bundle (18181-1 D.3.1).
///
/// # Errors
///
/// [`DecodeError::FieldOutOfRange`] for an out-of-range orientation,
/// [`DecodeError::Core`] if the extra-channel list exceeds the allocation
/// budget, or any error from a nested bundle.
pub fn read_image_metadata(
    reader: &mut BitReader<'_>,
    limits: &Limits,
    guard: &mut AllocGuard,
) -> Result<ImageMetadata> {
    let all_default = trace_field!(reader, "metadata.all_default", read_bool(reader))?;

    let mut meta = ImageMetadata {
        all_default,
        ..ImageMetadata::default()
    };

    if !all_default {
        read_guarded_fields(reader, limits, guard, &mut meta)?;
    }

    // Blank condition in Table D.3: read even under all_default.
    meta.default_m = trace_field!(reader, "metadata.default_m", read_bool(reader))?;

    if !meta.default_m {
        if meta.xyb_encoded {
            meta.opsin_inverse_matrix = Some(read_opsin_inverse_matrix(reader)?);
        }
        meta.upsampling = read_upsampling_weights(reader)?;
    }

    Ok(meta)
}

/// Reads every row of Table D.3 guarded by `!all_default` or `extra_fields`.
fn read_guarded_fields(
    reader: &mut BitReader<'_>,
    limits: &Limits,
    guard: &mut AllocGuard,
    meta: &mut ImageMetadata,
) -> Result<()> {
    meta.extra_fields = trace_field!(reader, "metadata.extra_fields", read_bool(reader))?;

    if meta.extra_fields {
        let orientation = trace_field!(reader, "metadata.orientation", reader.read_bits(3))? + 1;
        meta.orientation = Orientation::from_value(orientation).ok_or_else(|| {
            DecodeError::out_of_range("orientation", "D.3.2", u64::from(orientation))
        })?;

        if trace_field!(reader, "metadata.have_intr_size", read_bool(reader))? {
            meta.intrinsic_size = Some(read_size_header(reader, limits)?);
        }
        if trace_field!(reader, "metadata.have_preview", read_bool(reader))? {
            meta.preview = Some(read_preview_header(reader)?);
        }
        if trace_field!(reader, "metadata.have_animation", read_bool(reader))? {
            meta.animation = Some(read_animation_header(reader)?);
        }
    }

    meta.bit_depth = read_bit_depth(reader)?;
    meta.modular_16bit_buffers =
        trace_field!(reader, "metadata.modular_16bit_buffers", read_bool(reader))?;

    let num_extra = trace_field!(
        reader,
        "metadata.num_extra",
        read_u32(reader, &NUM_EXTRA_SPEC)
    )?;
    // Meter before allocating: num_extra reaches 4096 and is attacker-chosen,
    // and each entry can pull in a channel name of its own.
    guard
        .charge(u64::from(num_extra).saturating_mul(ENTRY_BUDGET_BYTES))
        .map_err(DecodeError::Core)?;
    meta.ec_info = Vec::with_capacity(num_extra as usize);
    for _ in 0..num_extra {
        meta.ec_info.push(read_extra_channel_info(reader, guard)?);
    }

    meta.xyb_encoded = trace_field!(reader, "metadata.xyb_encoded", read_bool(reader))?;
    meta.colour_encoding = read_colour_encoding(reader)?;

    if meta.extra_fields {
        meta.tone_mapping = read_tone_mapping(reader)?;
    }

    meta.extensions = read_extensions(reader)?;
    Ok(())
}

/// Rough in-memory cost of one `ExtraChannelInfo`, charged before the list is
/// allocated. Names are charged separately as they are read.
const ENTRY_BUDGET_BYTES: u64 = 64;

/// Reads `cw_mask` and any custom upsampling weight arrays (18181-1 D.3).
fn read_upsampling_weights(reader: &mut BitReader<'_>) -> Result<UpsamplingWeights> {
    let cw_mask = trace_field!(reader, "metadata.cw_mask", reader.read_bits(3))?;

    let mut weights = UpsamplingWeights {
        cw_mask,
        ..UpsamplingWeights::default()
    };
    // BitSet(u, b) is u & b, so these are masks rather than bit indices.
    if cw_mask & 1 != 0 {
        weights.up2 = Some(read_weights(reader, UP2_WEIGHTS, "metadata.up2_weight")?);
    }
    if cw_mask & 2 != 0 {
        weights.up4 = Some(read_weights(reader, UP4_WEIGHTS, "metadata.up4_weight")?);
    }
    if cw_mask & 4 != 0 {
        weights.up8 = Some(read_weights(reader, UP8_WEIGHTS, "metadata.up8_weight")?);
    }
    Ok(weights)
}

/// Reads a fixed-length array of `F16()` weights.
fn read_weights(reader: &mut BitReader<'_>, count: usize, field: &'static str) -> Result<Vec<f32>> {
    // Lengths are fixed by the standard (15/55/210), not by the stream, so no
    // allocation metering is needed here.
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(trace_field!(reader, field, read_f16_as_f32(reader))?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::headers::enums::ExtraChannelType;
    use crate::testsupport::BitWriter;

    fn read(bytes: &[u8]) -> Result<ImageMetadata> {
        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let mut r = BitReader::new(bytes);
        read_image_metadata(&mut r, &limits, &mut guard)
    }

    #[test]
    fn all_default_costs_exactly_two_bits() {
        // all_default = 1, then default_m = 1. The second bit is read despite
        // all_default, because its row in Table D.3 has a blank condition.
        let mut w = BitWriter::new();
        w.bool(true).bool(true);

        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let meta = read_image_metadata(&mut r, &limits, &mut guard).expect("valid");

        assert_eq!(
            r.total_bits_read(),
            2,
            "all_default plus the unguarded default_m"
        );
        assert!(meta.all_default);
        assert!(meta.default_m);
        assert!(meta.xyb_encoded, "the default is XYB");
        assert_eq!(meta.bit_depth.bits_per_sample(), 8);
        assert!(meta.ec_info.is_empty());
        assert_eq!(meta.orientation, Orientation::Identity);
        assert!(meta.opsin_inverse_matrix.is_none());
        assert_eq!(meta, ImageMetadata::default());
    }

    #[test]
    fn all_default_with_custom_opsin_matrix() {
        // all_default = 1 but default_m = 0: the opsin matrix and cw_mask are
        // still readable, which is only possible because default_m is unguarded.
        let mut w = BitWriter::new();
        w.bool(true).bool(false);
        w.bool(true); // opsin all_default
        w.u(3, 0); // cw_mask = 0
        let meta = read(&w.finish_padded(1)).expect("valid");

        assert!(meta.all_default);
        assert!(!meta.default_m);
        assert!(meta.opsin_inverse_matrix.is_some());
        assert_eq!(meta.upsampling.cw_mask, 0);
    }

    /// The minimal metadata that is not `all_default`.
    fn minimal_explicit(w: &mut BitWriter) {
        w.bool(false) // all_default
            .bool(false) // extra_fields
            .bool(false) // bit_depth.float_sample
            .u32_field(0, 0, 0) // bits_per_sample = 8
            .bool(true) // modular_16bit_buffers
            .u32_field(0, 0, 0) // num_extra = 0
            .bool(true) // xyb_encoded
            .bool(true) // colour_encoding all_default
            .u32_field(0, 0, 0) // extensions = 0 (U64 selector 0)
            .bool(true); // default_m
    }

    #[test]
    fn explicit_metadata_without_extra_fields() {
        let mut w = BitWriter::new();
        minimal_explicit(&mut w);
        let expected = w.bit_len();

        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let meta = read_image_metadata(&mut r, &limits, &mut guard).expect("valid");

        assert_eq!(r.total_bits_read(), expected);
        assert!(!meta.all_default);
        assert!(!meta.extra_fields);
        assert!(meta.xyb_encoded);
        assert!(meta.modular_16bit_buffers);
        assert_eq!(
            meta.tone_mapping,
            ToneMapping::default(),
            "tone_mapping is only read when extra_fields"
        );
    }

    #[test]
    fn extra_fields_reads_orientation_and_tone_mapping() {
        // extra_fields = 1, orientation = 1 + u(3) with payload 5 => 6.
        let mut w = BitWriter::new();
        w.bool(false) // all_default
            .bool(true) // extra_fields
            .u(3, 5) // orientation = 6
            .bool(false) // have_intr_size
            .bool(false) // have_preview
            .bool(false) // have_animation
            .bool(false) // bit_depth.float_sample
            .u32_field(0, 0, 0)
            .bool(true) // modular_16bit_buffers
            .u32_field(0, 0, 0) // num_extra
            .bool(true) // xyb_encoded
            .bool(true) // colour_encoding all_default
            .bool(true) // tone_mapping all_default
            .u32_field(0, 0, 0) // extensions
            .bool(true); // default_m

        let meta = read(&w.finish_padded(1)).expect("valid");
        assert_eq!(meta.orientation, Orientation::Rotate90Cw);
        assert!(meta.orientation.swaps_axes());
        assert!(meta.extra_fields);
    }

    #[test]
    fn intrinsic_size_preview_and_animation() {
        let mut w = BitWriter::new();
        w.bool(false)
            .bool(true) // extra_fields
            .u(3, 0) // orientation = 1
            .bool(true); // have_intr_size
        // SizeHeader: div8, h_div8 payload 0 => 8; ratio 0; w_div8 => 8.
        w.bool(true).u(5, 0).u(3, 0).u(5, 0);
        w.bool(true); // have_preview
        // PreviewHeader: div8, h_div8 selector 0 => 16 => 128; ratio 0; w 16.
        w.bool(true).u32_field(0, 0, 0).u(3, 0).u32_field(0, 0, 0);
        w.bool(true); // have_animation
        // AnimationHeader: 100/1, 0 loops, no timecodes.
        w.u32_field(0, 0, 0)
            .u32_field(0, 0, 0)
            .u32_field(0, 0, 0)
            .bool(false);
        w.bool(false) // float_sample
            .u32_field(0, 0, 0)
            .bool(true)
            .u32_field(0, 0, 0) // num_extra
            .bool(true) // xyb_encoded
            .bool(true) // colour all_default
            .bool(true) // tone_mapping all_default
            .u32_field(0, 0, 0) // extensions
            .bool(true); // default_m

        let meta = read(&w.finish_padded(2)).expect("valid");

        let intrinsic = meta.intrinsic_size.expect("signalled");
        assert_eq!(intrinsic.width().get(), 8);
        assert_eq!(intrinsic.height().get(), 8);

        let preview = meta.preview.expect("signalled");
        assert_eq!(preview.height(), 128);
        assert_eq!(preview.width(), 128);

        let anim = meta.animation.expect("signalled");
        assert_eq!(anim.tps_numerator, 100);
        assert_eq!(anim.tps_denominator, 1);
    }

    #[test]
    fn extra_channels_are_parsed() {
        // num_extra = 1 (U32 selector 1), one default-alpha channel.
        let mut w = BitWriter::new();
        w.bool(false)
            .bool(false)
            .bool(false)
            .u32_field(0, 0, 0)
            .bool(true)
            .u32_field(1, 0, 0) // num_extra = 1
            .bool(true) // ec_info[0]: d_alpha = 1
            .bool(true) // xyb_encoded
            .bool(true) // colour all_default
            .u32_field(0, 0, 0) // extensions
            .bool(true); // default_m

        let meta = read(&w.finish_padded(1)).expect("valid");
        assert_eq!(meta.num_extra(), 1);
        let ec = meta.ec_info.first().expect("one channel");
        assert_eq!(ec.channel_type, ExtraChannelType::KAlpha);
        assert!(!ec.alpha_associated);
    }

    #[test]
    fn several_extra_channels_of_different_types() {
        // num_extra = 3 via selector 2 => 2 + u(4) payload 1.
        let mut w = BitWriter::new();
        w.bool(false)
            .bool(false)
            .bool(false)
            .u32_field(0, 0, 0)
            .bool(true)
            .u32_field(2, 4, 1); // num_extra = 3
        // [0] default alpha
        w.bool(true);
        // [1] kDepth, 8-bit, dim_shift 0, no name
        w.bool(false)
            .enum_field(1)
            .bool(false)
            .u32_field(0, 0, 0)
            .u32_field(0, 0, 0)
            .u32_field(0, 0, 0);
        // [2] kBlack, 8-bit, dim_shift 0, no name
        w.bool(false)
            .enum_field(4)
            .bool(false)
            .u32_field(0, 0, 0)
            .u32_field(0, 0, 0)
            .u32_field(0, 0, 0);
        w.bool(false) // xyb_encoded = false (CMYK is stored, not XYB)
            .bool(true) // colour all_default
            .u32_field(0, 0, 0)
            .bool(true); // default_m

        let meta = read(&w.finish_padded(2)).expect("valid");
        assert_eq!(meta.num_extra(), 3);
        let types: Vec<_> = meta.ec_info.iter().map(|e| e.channel_type).collect();
        assert_eq!(
            types,
            vec![
                ExtraChannelType::KAlpha,
                ExtraChannelType::KDepth,
                ExtraChannelType::KBlack
            ]
        );
        assert!(!meta.xyb_encoded);
    }

    #[test]
    fn opsin_matrix_only_read_when_xyb_encoded() {
        // xyb_encoded = false and default_m = false: the opsin matrix row is
        // skipped, but cw_mask is still read.
        let mut w = BitWriter::new();
        w.bool(false)
            .bool(false)
            .bool(false)
            .u32_field(0, 0, 0)
            .bool(true)
            .u32_field(0, 0, 0)
            .bool(false) // xyb_encoded = false
            .bool(true) // colour all_default
            .u32_field(0, 0, 0) // extensions
            .bool(false) // default_m = false
            .u(3, 0); // cw_mask

        let meta = read(&w.finish_padded(1)).expect("valid");
        assert!(!meta.xyb_encoded);
        assert!(
            meta.opsin_inverse_matrix.is_none(),
            "the row is conditioned on xyb_encoded"
        );
        // The effective matrix still falls back to the L.1 defaults.
        assert_eq!(meta.effective_opsin_matrix(), OpsinInverseMatrix::default());
    }

    #[test]
    fn custom_upsampling_weights_use_masks_not_indices() {
        // cw_mask = 5 => bits 1 and 4 set => up2 and up8 arrays present,
        // up4 absent. This is the BitSet(u, b) == u & b semantics.
        let mut w = BitWriter::new();
        w.bool(true) // all_default
            .bool(false) // default_m
            .bool(true) // opsin all_default
            .u(3, 5); // cw_mask
        for _ in 0..UP2_WEIGHTS {
            w.f16_bits(0x3C00);
        }
        for _ in 0..UP8_WEIGHTS {
            w.f16_bits(0x0000);
        }
        let expected = w.bit_len();

        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let meta = read_image_metadata(&mut r, &limits, &mut guard).expect("valid");

        assert_eq!(r.total_bits_read(), expected);
        assert_eq!(meta.upsampling.cw_mask, 5);
        assert_eq!(meta.upsampling.up2.as_deref().map(<[f32]>::len), Some(15));
        assert!(meta.upsampling.up4.is_none(), "mask bit 2 is clear");
        assert_eq!(meta.upsampling.up8.as_deref().map(<[f32]>::len), Some(210));
    }

    #[test]
    fn all_three_weight_arrays() {
        let mut w = BitWriter::new();
        w.bool(true).bool(false).bool(true).u(3, 7);
        for _ in 0..(UP2_WEIGHTS + UP4_WEIGHTS + UP8_WEIGHTS) {
            w.f16_bits(0x3C00);
        }
        let meta = read(&w.finish_padded(1)).expect("valid");
        assert_eq!(meta.upsampling.up2.as_deref().map(<[f32]>::len), Some(15));
        assert_eq!(meta.upsampling.up4.as_deref().map(<[f32]>::len), Some(55));
        assert_eq!(meta.upsampling.up8.as_deref().map(<[f32]>::len), Some(210));
    }

    #[test]
    fn orientation_out_of_table_is_impossible_but_checked() {
        // 1 + u(3) spans exactly 1..=8, so every encodable value is a table
        // row; the guard exists so a future change to the field cannot slip
        // an undefined orientation through.
        for raw in 0..8u32 {
            assert!(Orientation::from_value(raw + 1).is_some());
        }
        assert!(Orientation::from_value(0).is_none());
        assert!(Orientation::from_value(9).is_none());
    }

    #[test]
    fn extensions_payload_is_skipped_within_metadata() {
        // extensions = 1 with a 5-bit payload; the following default_m bit
        // must still land correctly, which only happens if the payload was
        // consumed.
        let mut w = BitWriter::new();
        w.bool(false)
            .bool(false)
            .bool(false)
            .u32_field(0, 0, 0)
            .bool(true)
            .u32_field(0, 0, 0)
            .bool(true)
            .bool(true); // colour all_default
        w.u64_field(1); // extensions = 1
        w.u64_field(5); // extension_bits[0] = 5
        w.u(5, 0b10101); // the payload
        w.bool(false); // default_m = false
        w.bool(true); // opsin all_default
        w.u(3, 0); // cw_mask

        let meta = read(&w.finish_padded(1)).expect("valid");
        assert_eq!(meta.extensions.extensions, 1);
        assert_eq!(meta.extensions.extension_bits, vec![5]);
        assert!(
            !meta.default_m,
            "default_m lands correctly only if the payload was skipped"
        );
        assert!(meta.opsin_inverse_matrix.is_some());
    }

    #[test]
    fn num_extra_allocation_is_metered() {
        // num_extra = 4096 (selector 3 => 1 + u(12), payload 4095) against a
        // small allocation budget.
        let mut w = BitWriter::new();
        w.bool(false)
            .bool(false)
            .bool(false)
            .u32_field(0, 0, 0)
            .bool(true)
            .u32_field(3, 12, 4095);
        let data = w.finish_padded(8);

        let limits = Limits {
            max_alloc_bytes: 1024,
            ..Limits::default()
        };
        let mut guard = AllocGuard::new(&limits);
        let mut r = BitReader::new(&data);
        assert!(read_image_metadata(&mut r, &limits, &mut guard).is_err());
    }

    #[test]
    fn truncated_metadata_errors() {
        assert!(read(&[]).is_err());
        // all_default = 0 then nothing.
        assert!(read(&[0b0000_0000]).is_err());
    }
}
