//! The `ColourEncoding` (18181-1 E.2) and `ToneMapping` (18181-1 E.3) bundles.
//!
//! ```text
//! Table E.1 — ColourEncoding bundle
//! condition                   type                     default     name
//!                             Bool()                   true        all_default
//! !all_default                Bool()                   false       want_icc
//! !all_default                Enum(ColourSpace)        kRGB        colour_space
//! use_desc and not_xyb        Enum(WhitePoint)         kD65        white_point
//! white_point == kCustom      Customxy                             white
//! has_primaries               Enum(Primaries)          kSRGB       primaries
//! primaries == kCustom        Customxy                             red
//! primaries == kCustom        Customxy                             green
//! primaries == kCustom        Customxy                             blue
//! use_desc                    CustomTransferFunction               tf
//! use_desc                    Enum(RenderingIntent)    kRelative   rendering_intent
//! ```
//!
//! E.2 defines the three compound conditions, and getting them wrong desyncs
//! everything downstream:
//!
//! * `use_desc      = !all_default && !want_icc`
//! * `not_xyb       = colour_space != kXYB`
//! * `has_primaries = use_desc && not_xyb && colour_space != kGrey`
//!
//! An ICC profile therefore suppresses *all* of the descriptive fields, and a
//! greyscale space suppresses the primaries but keeps the white point.

use jpxl_bitstream::{
    BitReader, U32Dist, U32Spec, read_bool, read_f16_as_f32, read_u32, trace_field,
};

use crate::error::Result;
use crate::headers::enums::{
    ColourSpace, Primaries, RenderingIntent, TransferFunction, WhitePoint, read_enum,
};

/// 18181-1 E.2: `U32(u(19), 524288 + u(19), 1048576 + u(20), 2097152 + u(21))`.
const CUSTOM_XY_SPEC: U32Spec = U32Spec::new([
    U32Dist::bits(19),
    U32Dist::BitsOffset {
        bits: 19,
        offset: 524_288,
    },
    U32Dist::BitsOffset {
        bits: 20,
        offset: 1_048_576,
    },
    U32Dist::BitsOffset {
        bits: 21,
        offset: 2_097_152,
    },
]);

/// `UnpackSigned(u)`: `u / 2` if `u` is even, `-(u + 1) / 2` if odd
/// (18181-1 notational conventions).
#[must_use]
pub const fn unpack_signed(u: u32) -> i32 {
    if u.is_multiple_of(2) {
        // u / 2 <= 2^31 - 1, so this is exact.
        0i32.wrapping_add_unsigned(u / 2)
    } else {
        // -(u + 1) / 2. Written as a wrapping subtraction from zero because
        // u == u32::MAX gives magnitude 2^31, which is i32::MIN and has no
        // positive counterpart to negate.
        0i32.wrapping_sub_unsigned(u / 2 + 1)
    }
}

/// A `Customxy` bundle: a CIE xy chromaticity point scaled by 10^6
/// (18181-1 Table E.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CustomXy {
    /// x coordinate, scaled by 10^6.
    pub x: i32,
    /// y coordinate, scaled by 10^6.
    pub y: i32,
}

impl CustomXy {
    /// The unscaled coordinates. May fall outside `[0, 1]` for imaginary
    /// primaries, which E.2 explicitly permits.
    #[must_use]
    pub fn unscaled(&self) -> (f64, f64) {
        (f64::from(self.x) / 1e6, f64::from(self.y) / 1e6)
    }
}

/// Reads a `Customxy` bundle (18181-1 Table E.2).
///
/// # Errors
///
/// A bitstream error on truncation.
pub fn read_custom_xy(reader: &mut BitReader<'_>, field: &'static str) -> Result<CustomXy> {
    let ux = trace_field!(reader, field, read_u32(reader, &CUSTOM_XY_SPEC))?;
    let uy = trace_field!(reader, field, read_u32(reader, &CUSTOM_XY_SPEC))?;
    Ok(CustomXy {
        x: unpack_signed(ux),
        y: unpack_signed(uy),
    })
}

/// A `CustomTransferFunction` bundle (18181-1 Table E.7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CustomTransferFunction {
    /// `have_gamma`: the OETF exponent is `gamma / 10^7`, which lies in `(0, 1]`.
    Gamma(u32),
    /// One of the enumerated transfer functions.
    Enumerated(TransferFunction),
}

impl Default for CustomTransferFunction {
    fn default() -> Self {
        Self::Enumerated(TransferFunction::KSrgb)
    }
}

/// Reads a `CustomTransferFunction` bundle (18181-1 Table E.7).
///
/// # Errors
///
/// [`DecodeError::UnknownEnumValue`](crate::DecodeError::UnknownEnumValue) for
/// an undefined transfer function, or a bitstream error on truncation.
pub fn read_custom_transfer_function(reader: &mut BitReader<'_>) -> Result<CustomTransferFunction> {
    let have_gamma = trace_field!(reader, "tf.have_gamma", read_bool(reader))?;
    if have_gamma {
        let gamma = trace_field!(reader, "tf.gamma", reader.read_bits(24))?;
        Ok(CustomTransferFunction::Gamma(gamma))
    } else {
        Ok(CustomTransferFunction::Enumerated(read_enum(
            reader,
            "tf.transfer_function",
        )?))
    }
}

/// A decoded `ColourEncoding` bundle (18181-1 E.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColourEncoding {
    /// Whether the whole bundle took its default (a single `1` bit).
    pub all_default: bool,
    /// Whether an ICC profile is stored in the codestream (see E.4).
    pub want_icc: bool,
    /// The colour space family.
    pub colour_space: ColourSpace,
    /// White point; unread (and left at `kD65`) for XYB.
    pub white_point: WhitePoint,
    /// Custom white point, present when `white_point == kCustom`.
    pub white: Option<CustomXy>,
    /// Primaries; unread for greyscale and XYB.
    pub primaries: Primaries,
    /// Custom red primary, present when `primaries == kCustom`.
    pub red: Option<CustomXy>,
    /// Custom green primary, present when `primaries == kCustom`.
    pub green: Option<CustomXy>,
    /// Custom blue primary, present when `primaries == kCustom`.
    pub blue: Option<CustomXy>,
    /// Transfer function.
    pub tf: CustomTransferFunction,
    /// Rendering intent.
    pub rendering_intent: RenderingIntent,
}

impl Default for ColourEncoding {
    /// The `all_default` state: sRGB, D65, relative colorimetric.
    fn default() -> Self {
        Self {
            all_default: true,
            want_icc: false,
            colour_space: ColourSpace::KRgb,
            white_point: WhitePoint::KD65,
            white: None,
            primaries: Primaries::KSrgb,
            red: None,
            green: None,
            blue: None,
            tf: CustomTransferFunction::default(),
            rendering_intent: RenderingIntent::KRelative,
        }
    }
}

impl ColourEncoding {
    /// Whether the descriptive fields were stored: `!all_default && !want_icc`.
    #[must_use]
    pub const fn use_desc(&self) -> bool {
        !self.all_default && !self.want_icc
    }

    /// Whether the stored image is greyscale.
    #[must_use]
    pub fn is_grey(&self) -> bool {
        self.colour_space == ColourSpace::KGrey
    }
}

/// Reads a `ColourEncoding` bundle (18181-1 E.2).
///
/// # Errors
///
/// [`DecodeError::UnknownEnumValue`](crate::DecodeError::UnknownEnumValue) for
/// any enumerated field with no table row, or a bitstream error on truncation.
pub fn read_colour_encoding(reader: &mut BitReader<'_>) -> Result<ColourEncoding> {
    let all_default = trace_field!(reader, "colour.all_default", read_bool(reader))?;
    if all_default {
        return Ok(ColourEncoding::default());
    }

    let want_icc = trace_field!(reader, "colour.want_icc", read_bool(reader))?;
    let colour_space: ColourSpace = read_enum(reader, "colour.colour_space")?;

    // E.2's compound conditions, written once.
    let use_desc = !want_icc;
    let not_xyb = colour_space != ColourSpace::KXyb;
    let has_primaries = use_desc && not_xyb && colour_space != ColourSpace::KGrey;

    let white_point = if use_desc && not_xyb {
        read_enum(reader, "colour.white_point")?
    } else {
        WhitePoint::KD65
    };
    let white = if white_point == WhitePoint::KCustom {
        Some(read_custom_xy(reader, "colour.white")?)
    } else {
        None
    };

    let primaries = if has_primaries {
        read_enum(reader, "colour.primaries")?
    } else {
        Primaries::KSrgb
    };
    let (red, green, blue) = if primaries == Primaries::KCustom {
        (
            Some(read_custom_xy(reader, "colour.red")?),
            Some(read_custom_xy(reader, "colour.green")?),
            Some(read_custom_xy(reader, "colour.blue")?),
        )
    } else {
        (None, None, None)
    };

    let tf = if use_desc {
        read_custom_transfer_function(reader)?
    } else {
        CustomTransferFunction::default()
    };
    let rendering_intent = if use_desc {
        read_enum(reader, "colour.rendering_intent")?
    } else {
        RenderingIntent::KRelative
    };

    Ok(ColourEncoding {
        all_default: false,
        want_icc,
        colour_space,
        white_point,
        white,
        primaries,
        red,
        green,
        blue,
        tf,
        rendering_intent,
    })
}

/// A decoded `ToneMapping` bundle (18181-1 E.3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToneMapping {
    /// Upper bound on image intensity in nits; the value meaning "1.0".
    pub intensity_target: f32,
    /// Lower bound on image intensity in nits.
    pub min_nits: f32,
    /// Whether `linear_below` is a ratio of max display brightness.
    pub relative_to_max_display: bool,
    /// Value below which tone mapping leaves samples unchanged.
    pub linear_below: f32,
}

impl Default for ToneMapping {
    /// Table E.9 defaults: 255 nits, 0 min, absolute, 0 linear-below.
    fn default() -> Self {
        Self {
            intensity_target: 255.0,
            min_nits: 0.0,
            relative_to_max_display: false,
            linear_below: 0.0,
        }
    }
}

/// Reads a `ToneMapping` bundle (18181-1 E.3).
///
/// # Errors
///
/// A bitstream error on truncation or an invalid `F16()`.
pub fn read_tone_mapping(reader: &mut BitReader<'_>) -> Result<ToneMapping> {
    let all_default = trace_field!(reader, "tone_mapping.all_default", read_bool(reader))?;
    if all_default {
        return Ok(ToneMapping::default());
    }

    Ok(ToneMapping {
        intensity_target: trace_field!(
            reader,
            "tone_mapping.intensity_target",
            read_f16_as_f32(reader)
        )?,
        min_nits: trace_field!(reader, "tone_mapping.min_nits", read_f16_as_f32(reader))?,
        relative_to_max_display: trace_field!(
            reader,
            "tone_mapping.relative_to_max_display",
            read_bool(reader)
        )?,
        linear_below: trace_field!(reader, "tone_mapping.linear_below", read_f16_as_f32(reader))?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testsupport::BitWriter;

    fn read_colour(bytes: &[u8]) -> Result<ColourEncoding> {
        let mut r = BitReader::new(bytes);
        read_colour_encoding(&mut r)
    }

    #[test]
    fn unpack_signed_matches_the_convention() {
        assert_eq!(unpack_signed(0), 0);
        assert_eq!(unpack_signed(1), -1);
        assert_eq!(unpack_signed(2), 1);
        assert_eq!(unpack_signed(3), -2);
        assert_eq!(unpack_signed(4), 2);
        // Must not overflow at the extremes.
        assert_eq!(unpack_signed(u32::MAX), i32::MIN);
    }

    #[test]
    fn all_default_is_one_bit_srgb() {
        let mut w = BitWriter::new();
        w.bool(true);
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let ce = read_colour_encoding(&mut r).expect("valid");

        assert_eq!(r.total_bits_read(), 1);
        assert!(ce.all_default);
        assert_eq!(ce.colour_space, ColourSpace::KRgb);
        assert_eq!(ce.white_point, WhitePoint::KD65);
        assert_eq!(ce.primaries, Primaries::KSrgb);
        assert_eq!(
            ce.tf,
            CustomTransferFunction::Enumerated(TransferFunction::KSrgb)
        );
        assert_eq!(ce.rendering_intent, RenderingIntent::KRelative);
    }

    #[test]
    fn want_icc_suppresses_all_descriptive_fields() {
        // all_default = 0, want_icc = 1, colour_space = kRGB.
        // use_desc is false, so nothing after colour_space is read.
        let mut w = BitWriter::new();
        w.bool(false).bool(true).enum_field(0);
        let expected = w.bit_len();

        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let ce = read_colour_encoding(&mut r).expect("valid");

        assert!(ce.want_icc);
        assert!(!ce.use_desc());
        assert_eq!(r.total_bits_read(), expected);
    }

    #[test]
    fn xyb_skips_white_point_and_primaries() {
        // all_default = 0, want_icc = 0, colour_space = kXYB (2).
        // not_xyb is false => no white_point, no primaries; tf and
        // rendering_intent are still read because use_desc holds.
        let mut w = BitWriter::new();
        w.bool(false)
            .bool(false)
            .enum_field(2)
            .bool(false) // tf.have_gamma
            .enum_field(13) // kSRGB
            .enum_field(1); // kRelative
        let expected = w.bit_len();

        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let ce = read_colour_encoding(&mut r).expect("valid");

        assert_eq!(ce.colour_space, ColourSpace::KXyb);
        assert_eq!(ce.white_point, WhitePoint::KD65, "defaulted, not read");
        assert!(ce.white.is_none());
        assert_eq!(r.total_bits_read(), expected);
    }

    #[test]
    fn greyscale_reads_white_point_but_not_primaries() {
        // colour_space = kGrey (1): has_primaries is false, not_xyb is true.
        let mut w = BitWriter::new();
        w.bool(false)
            .bool(false)
            .enum_field(1)
            .enum_field(1) // white_point = kD65
            .bool(false)
            .enum_field(13)
            .enum_field(1);
        let expected = w.bit_len();

        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let ce = read_colour_encoding(&mut r).expect("valid");

        assert!(ce.is_grey());
        assert_eq!(ce.white_point, WhitePoint::KD65);
        assert_eq!(ce.primaries, Primaries::KSrgb, "defaulted, not read");
        assert_eq!(r.total_bits_read(), expected);
    }

    #[test]
    fn custom_white_point_and_primaries() {
        // colour_space = kRGB, white_point = kCustom (2), primaries = kCustom.
        let mut w = BitWriter::new();
        w.bool(false).bool(false).enum_field(0).enum_field(2);
        // white: ux = 2 (=> x = 1), uy = 4 (=> y = 2), both via u(19).
        w.u32_field(0, 19, 2).u32_field(0, 19, 4);
        w.enum_field(2); // primaries = kCustom
        // red/green/blue, each two u(19) values.
        for v in [10u32, 12, 14, 16, 18, 20] {
            w.u32_field(0, 19, v);
        }
        w.bool(false).enum_field(13).enum_field(1);

        let ce = read_colour(&w.finish_padded(2)).expect("valid");
        assert_eq!(ce.white, Some(CustomXy { x: 1, y: 2 }));
        assert_eq!(ce.red, Some(CustomXy { x: 5, y: 6 }));
        assert_eq!(ce.green, Some(CustomXy { x: 7, y: 8 }));
        assert_eq!(ce.blue, Some(CustomXy { x: 9, y: 10 }));
    }

    #[test]
    fn negative_custom_coordinates() {
        // ux = 1 => UnpackSigned(1) = -1, allowed for imaginary primaries.
        let mut w = BitWriter::new();
        w.bool(false).bool(false).enum_field(0).enum_field(2);
        w.u32_field(0, 19, 1).u32_field(0, 19, 3);
        w.enum_field(1); // primaries = kSRGB, no custom triples
        w.bool(false).enum_field(13).enum_field(1);

        let ce = read_colour(&w.finish_padded(2)).expect("valid");
        assert_eq!(ce.white, Some(CustomXy { x: -1, y: -2 }));
        assert!(ce.red.is_none());
    }

    #[test]
    fn gamma_transfer_function() {
        // have_gamma = 1, gamma = u(24). 0.45 * 10^7 = 4_500_000.
        let mut w = BitWriter::new();
        w.bool(false)
            .bool(false)
            .enum_field(0) // colour_space = kRGB
            .enum_field(1) // white_point = kD65
            .enum_field(1) // primaries = kSRGB
            .bool(true) // tf.have_gamma
            .u(24, 4_500_000)
            .enum_field(1); // rendering_intent
        let ce = read_colour(&w.finish_padded(2)).expect("valid");
        assert_eq!(ce.tf, CustomTransferFunction::Gamma(4_500_000));
    }

    #[test]
    fn unknown_transfer_function_rejected() {
        // Value 3 has no row in Table E.6.
        let mut w = BitWriter::new();
        w.bool(false)
            .bool(false)
            .enum_field(0) // colour_space = kRGB
            .enum_field(1) // white_point = kD65
            .enum_field(1) // primaries = kSRGB
            .bool(false) // tf.have_gamma
            .enum_field(3); // 3 has no row in Table E.6
        assert!(read_colour(&w.finish_padded(2)).is_err());
    }

    #[test]
    fn tone_mapping_default_is_255_nits() {
        let mut w = BitWriter::new();
        w.bool(true);
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let tm = read_tone_mapping(&mut r).expect("valid");
        assert_eq!(r.total_bits_read(), 1);
        assert_eq!(tm.intensity_target, 255.0);
        assert_eq!(tm.min_nits, 0.0);
        assert!(!tm.relative_to_max_display);
    }

    #[test]
    fn tone_mapping_explicit_values() {
        // F16 1000.0 is 0x63D0; 1.0 is 0x3C00; 0.5 is 0x3800.
        let mut w = BitWriter::new();
        w.bool(false)
            .f16_bits(0x63D0)
            .f16_bits(0x3C00)
            .bool(true)
            .f16_bits(0x3800);
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let tm = read_tone_mapping(&mut r).expect("valid");
        assert_eq!(tm.intensity_target, 1000.0);
        assert_eq!(tm.min_nits, 1.0);
        assert!(tm.relative_to_max_display);
        assert_eq!(tm.linear_below, 0.5);
    }
}
