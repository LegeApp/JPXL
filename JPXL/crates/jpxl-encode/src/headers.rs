//! Signature, `SizeHeader` and `ImageMetadata` (18181-1 D.1, D.2, D.3, E.2).
//!
//! # Why `all_default` is not usable here
//!
//! The Table D.3 defaults describe an **XYB-encoded sRGB** image. Every image
//! this encoder produces is non-XYB, so `xyb_encoded` alone already breaks the
//! default, and Table D.3 has no way to override one row while defaulting the
//! rest — `all_default` is all or nothing. So the bundle is written out field
//! by field, taking the table's own default for every row that can keep it.
//!
//! The one row that is *not* guarded by `all_default` is `default_m`, which is
//! written last and set, meaning "use the L.1 opsin matrix and the K.2
//! upsampling weights". It costs one bit even though nothing in a modular
//! non-XYB frame consults either.
//!
//! # `modular_16bit_buffers`
//!
//! D.3 defines this as a claim about the *decoder's* working buffers: signed
//! 16-bit integers suffice for every decoded modular sample and every inverse
//! transform result. That is true for 8-bit samples (an RCT chroma residual
//! stays well inside `±2^15`) and false for 16-bit ones, where the samples
//! alone reach 65535. It is therefore derived from the bit depth rather than
//! pinned to the table default. Annex M ties the `false` case to level 10,
//! which is why the container writer emits a `jxll` box for 16-bit images.

use jpxl_bitstream::{BitWriter, U32Dist, U32Spec};

use crate::error::{EncodeError, Result};

/// 18181-1 D.1: the signature as a `u(16)` value, i.e. the bytes `FF 0A` read
/// LSB-first.
const SIGNATURE: u32 = 0x0AFF;

/// 18181-1 D.2: `U32(1 + u(9), 1 + u(13), 1 + u(18), 1 + u(30))`.
const SIZE_DIM_SPEC: U32Spec = U32Spec::new([
    U32Dist::BitsOffset { bits: 9, offset: 1 },
    U32Dist::BitsOffset {
        bits: 13,
        offset: 1,
    },
    U32Dist::BitsOffset {
        bits: 18,
        offset: 1,
    },
    U32Dist::BitsOffset {
        bits: 30,
        offset: 1,
    },
]);

/// 18181-1 D.7: `U32(8, 10, 12, 1 + u(6))`, the integer-sample bit depth.
const INT_BPS_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(8),
    U32Dist::Val(10),
    U32Dist::Val(12),
    U32Dist::BitsOffset { bits: 6, offset: 1 },
]);

/// 18181-1 D.3: `U32(0, 1, 2 + u(4), 1 + u(12))` for `num_extra`.
const NUM_EXTRA_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(0),
    U32Dist::Val(1),
    U32Dist::BitsOffset { bits: 4, offset: 2 },
    U32Dist::BitsOffset {
        bits: 12,
        offset: 1,
    },
]);

/// 18181-1 B.2.6: `Enum(x)` is `U32(0, 1, 2 + u(4), 18 + u(6))`.
const ENUM_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(0),
    U32Dist::Val(1),
    U32Dist::BitsOffset { bits: 4, offset: 2 },
    U32Dist::BitsOffset {
        bits: 6,
        offset: 18,
    },
]);

/// Table E.3 `ColourSpace`: `kRGB`, the bundle default.
const COLOUR_SPACE_RGB: u32 = 0;
/// Table E.3 `ColourSpace`: `kGrey`.
const COLOUR_SPACE_GREY: u32 = 1;
/// Table E.4 `WhitePoint`: `kD65`, the bundle default.
const WHITE_POINT_D65: u32 = 1;
/// Table E.5 `Primaries`: `kSRGB`, the bundle default.
const PRIMARIES_SRGB: u32 = 1;
/// Table E.5 `Primaries`: ITU-R BT.2100-2 (the Rec.2020 gamut).
const PRIMARIES_2100: u32 = 9;
/// Table E.5 `Primaries`: SMPTE ST 428-1 (P3).
const PRIMARIES_P3: u32 = 11;
/// Table E.6 `TransferFunction`: gamma exponent 1.
const TRANSFER_FUNCTION_LINEAR: u32 = 8;
/// Table E.6 `TransferFunction`: `kSrgb`, the bundle default.
const TRANSFER_FUNCTION_SRGB: u32 = 13;
/// Table E.8 `RenderingIntent`: `kRelative`, the bundle default.
const RENDERING_INTENT_RELATIVE: u32 = 1;

/// Largest dimension `SizeHeader` can carry (`1 + u(30)`).
const MAX_DIMENSION: u32 = 1 << 30;

/// Writes the `u(16)` codestream signature (18181-1 D.1).
///
/// # Errors
///
/// Only through the bit writer.
pub fn write_signature(w: &mut BitWriter) -> Result<()> {
    w.write_bits(16, SIGNATURE)?;
    Ok(())
}

/// Writes a `SizeHeader` bundle (18181-1 D.2, Table D.2).
///
/// The general (`!div8`, `ratio == 0`) form is always used: it costs a few
/// bits more than the special forms for the dimensions that have one, and it
/// works for every dimension, so there is no branch to get wrong.
///
/// # Errors
///
/// [`EncodeError::ValueOutOfRange`] for a zero or oversized dimension.
pub fn write_size_header(w: &mut BitWriter, width: u32, height: u32) -> Result<()> {
    for (name, value) in [("width", width), ("height", height)] {
        if value == 0 || value > MAX_DIMENSION {
            return Err(EncodeError::ValueOutOfRange {
                what: name,
                value: i64::from(value),
            });
        }
    }

    w.write_bool(false); // div8
    w.write_u32(&SIZE_DIM_SPEC, height)?;
    w.write_bits(3, 0)?; // ratio: the width is stored explicitly
    w.write_u32(&SIZE_DIM_SPEC, width)?;
    Ok(())
}

/// Whether an image is greyscale or RGB — the only two colour shapes this
/// encoder produces (18181-1 G.1.3 counts channels from exactly this).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColourShape {
    /// One channel, `colour_space = kGrey`.
    Grey,
    /// Three channels, `colour_space = kRGB`.
    Rgb,
}

impl ColourShape {
    /// Number of colour channels the frame carries.
    #[must_use]
    pub const fn num_channels(self) -> usize {
        match self {
            Self::Grey => 1,
            Self::Rgb => 3,
        }
    }
}

/// The colour space the caller's samples are in, signalled declaratively in
/// the `ColourEncoding` bundle (18181-1 E.2).
///
/// The Modular path stores samples untouched, so this is a pure header claim:
/// it changes how a colour-managed viewer interprets the decoded samples and
/// nothing else. Every variant is a named row combination of Tables E.3–E.8 —
/// D65 white point and relative-colorimetric intent throughout, which is what
/// every real capture pipeline this encoder feeds produces.
///
/// The XYB-encoded VarDCT path is not covered by this enum: its forward
/// transform and its perceptual metric are defined on sRGB input, so the
/// policy layer rejects a non-sRGB lossy request instead of mis-tagging it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ColourSpace {
    /// IEC 61966-2-1 sRGB — the Table E.1 `all_default` bundle.
    #[default]
    Srgb,
    /// sRGB primaries with a linear transfer function (gamma 1).
    LinearSrgb,
    /// Display P3: SMPTE ST 428-1 primaries, D65, the sRGB transfer function.
    DisplayP3,
    /// Rec.2020 gamut: ITU-R BT.2100-2 primaries, D65, the sRGB transfer
    /// function (the shape a "rec2020 working space" still image carries).
    Rec2020,
}

impl ColourSpace {
    /// `(primaries, transfer_function)` Table E.5/E.6 rows for an RGB image.
    const fn rgb_rows(self) -> (u32, u32) {
        match self {
            Self::Srgb => (PRIMARIES_SRGB, TRANSFER_FUNCTION_SRGB),
            Self::LinearSrgb => (PRIMARIES_SRGB, TRANSFER_FUNCTION_LINEAR),
            Self::DisplayP3 => (PRIMARIES_P3, TRANSFER_FUNCTION_SRGB),
            Self::Rec2020 => (PRIMARIES_2100, TRANSFER_FUNCTION_SRGB),
        }
    }
}

/// Writes an `ImageMetadata` bundle describing a non-XYB integer still image
/// (18181-1 D.3, Table D.3).
///
/// # Errors
///
/// [`EncodeError::Unsupported`] for a greyscale image in a non-sRGB colour
/// space, [`EncodeError::ValueOutOfRange`] if `bits_per_sample` is outside
/// the 1..=16 range this encoder handles, or a bit writer error.
pub fn write_metadata(
    w: &mut BitWriter,
    shape: ColourShape,
    bits_per_sample: u32,
    colour: ColourSpace,
) -> Result<()> {
    if bits_per_sample == 0 || bits_per_sample > 16 {
        return Err(EncodeError::ValueOutOfRange {
            what: "bits_per_sample",
            value: i64::from(bits_per_sample),
        });
    }

    w.write_bool(false); // all_default: see the module documentation
    w.write_bool(false); // extra_fields: no orientation, preview or animation

    // BitDepth (D.3.5, Table D.7): integer samples of the requested depth.
    w.write_bool(false); // float_sample
    w.write_u32(&INT_BPS_SPEC, bits_per_sample)?;

    // See the module documentation: this is a claim about decoder buffers.
    w.write_bool(bits_per_sample <= 8);
    w.write_u32(&NUM_EXTRA_SPEC, 0)?; // no extra channels
    w.write_bool(false); // xyb_encoded: samples are stored as-is

    write_colour_encoding(w, shape, colour)?;

    // tone_mapping is guarded by extra_fields, which is false.
    w.write_u64(0)?; // extensions (B.3)

    // Blank condition in Table D.3: read even under all_default.
    w.write_bool(true); // default_m
    Ok(())
}

/// Writes an 8-bit greyscale `ImageMetadata` bundle.
///
/// # Errors
///
/// Only through the bit writer.
pub fn write_grey8_metadata(w: &mut BitWriter) -> Result<()> {
    write_metadata(w, ColourShape::Grey, 8, ColourSpace::Srgb)
}

/// Writes a `ColourEncoding` bundle (18181-1 E.2, Table E.1).
///
/// The Table E.1 defaults are exactly sRGB: `want_icc` false, `kRGB`, `kD65`,
/// `kSRGB` primaries, the sRGB transfer function and `kRelative`. An sRGB RGB
/// image is therefore one `all_default` bit. Every other case writes the
/// bundle out: `kGrey` with `has_primaries` false (which skips the primaries
/// rows but not the white point), and a non-default RGB colour space with the
/// primaries and transfer-function rows of [`ColourSpace::rgb_rows`].
///
/// # Errors
///
/// [`EncodeError::Unsupported`] for a greyscale image in a non-sRGB colour
/// space — a combination nothing feeds this encoder — or a bit writer error.
fn write_colour_encoding(w: &mut BitWriter, shape: ColourShape, colour: ColourSpace) -> Result<()> {
    if shape == ColourShape::Grey {
        if colour != ColourSpace::Srgb {
            return Err(EncodeError::unsupported(
                "a greyscale image can only be signalled as sRGB",
                "E.2",
            ));
        }
        w.write_bool(false); // all_default (the default is kRGB)
        w.write_bool(false); // want_icc
        w.write_u32(&ENUM_SPEC, COLOUR_SPACE_GREY)?;
        w.write_u32(&ENUM_SPEC, WHITE_POINT_D65)?;
        // primaries / red / green / blue: skipped for kGrey.
        // CustomTransferFunction (Table E.7).
        w.write_bool(false); // have_gamma
        w.write_u32(&ENUM_SPEC, TRANSFER_FUNCTION_SRGB)?;
        w.write_u32(&ENUM_SPEC, RENDERING_INTENT_RELATIVE)?;
        return Ok(());
    }
    if colour == ColourSpace::Srgb {
        w.write_bool(true); // all_default
        return Ok(());
    }
    let (primaries, transfer) = colour.rgb_rows();
    w.write_bool(false); // all_default
    w.write_bool(false); // want_icc
    w.write_u32(&ENUM_SPEC, COLOUR_SPACE_RGB)?;
    w.write_u32(&ENUM_SPEC, WHITE_POINT_D65)?;
    // white: skipped, the white point is not kCustom.
    w.write_u32(&ENUM_SPEC, primaries)?;
    // red / green / blue: skipped, the primaries are not kCustom.
    // CustomTransferFunction (Table E.7).
    w.write_bool(false); // have_gamma
    w.write_u32(&ENUM_SPEC, transfer)?;
    w.write_u32(&ENUM_SPEC, RENDERING_INTENT_RELATIVE)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use jpxl_bitstream::BitReader;
    use jpxl_core::limits::Limits;
    use jpxl_decode::headers::decode_image_headers;

    fn headers(width: u32, height: u32, shape: ColourShape, bits: u32) -> Vec<u8> {
        headers_in(width, height, shape, bits, ColourSpace::Srgb)
    }

    fn headers_in(
        width: u32,
        height: u32,
        shape: ColourShape,
        bits: u32,
        colour: ColourSpace,
    ) -> Vec<u8> {
        let mut w = BitWriter::new();
        write_signature(&mut w).expect("signature");
        write_size_header(&mut w, width, height).expect("size");
        write_metadata(&mut w, shape, bits, colour).expect("metadata");
        w.zero_pad_to_byte();
        w.into_bytes()
    }

    #[test]
    fn headers_round_trip_through_the_decoder() {
        for (width, height) in [(1u32, 1u32), (8, 8), (13, 7), (300, 200), (512, 512)] {
            for shape in [ColourShape::Grey, ColourShape::Rgb] {
                for bits in [8u32, 16] {
                    let bytes = headers(width, height, shape, bits);
                    let mut r = BitReader::new(&bytes);
                    let parsed = decode_image_headers(&mut r, &Limits::default())
                        .unwrap_or_else(|e| panic!("{width}x{height} {shape:?} {bits}: {e}"));

                    assert_eq!(parsed.width(), width);
                    assert_eq!(parsed.height(), height);
                    assert!(!parsed.metadata.xyb_encoded);
                    assert_eq!(
                        parsed.metadata.colour_encoding.is_grey(),
                        shape == ColourShape::Grey
                    );
                    assert_eq!(parsed.metadata.bit_depth.bits_per_sample(), bits);
                    assert_eq!(parsed.metadata.modular_16bit_buffers, bits <= 8);
                    assert!(parsed.metadata.ec_info.is_empty());
                    assert!(parsed.metadata.default_m);
                    assert!(parsed.metadata.preview.is_none());
                    assert!(parsed.metadata.animation.is_none());
                }
            }
        }
    }

    #[test]
    fn the_header_ends_where_the_writer_says_it_does() {
        // A wrong conditional shifts every later field, so the bit count is
        // the assertion that actually catches it.
        for shape in [ColourShape::Grey, ColourShape::Rgb] {
            for bits in [8u32, 16] {
                let mut w = BitWriter::new();
                write_signature(&mut w).expect("signature");
                write_size_header(&mut w, 64, 64).expect("size");
                write_metadata(&mut w, shape, bits, ColourSpace::Srgb).expect("metadata");
                let written = w.bit_len();
                w.zero_pad_to_byte();
                let bytes = w.into_bytes();

                let mut r = BitReader::new(&bytes);
                decode_image_headers(&mut r, &Limits::default()).expect("valid");
                assert_eq!(r.total_bits_read(), written, "{shape:?} {bits}");
            }
        }
    }

    /// Each declarative colour space comes back from the decoder as the
    /// Table E.5/E.6 rows it names, with the header ending exactly where the
    /// writer says — the bit count is what catches a wrong E.1 conditional.
    #[test]
    fn non_default_colour_spaces_round_trip_through_the_decoder() {
        use jpxl_decode::headers::colour::CustomTransferFunction;
        use jpxl_decode::headers::enums::{Primaries, TransferFunction, WhitePoint};

        for (colour, primaries, tf) in [
            (
                ColourSpace::LinearSrgb,
                Primaries::KSrgb,
                TransferFunction::KLinear,
            ),
            (
                ColourSpace::DisplayP3,
                Primaries::KP3,
                TransferFunction::KSrgb,
            ),
            (
                ColourSpace::Rec2020,
                Primaries::K2100,
                TransferFunction::KSrgb,
            ),
        ] {
            let bytes = headers_in(64, 64, ColourShape::Rgb, 16, colour);
            let mut r = BitReader::new(&bytes);
            let parsed = decode_image_headers(&mut r, &Limits::default())
                .unwrap_or_else(|e| panic!("{colour:?}: {e}"));

            let ce = parsed.metadata.colour_encoding;
            assert!(!ce.all_default, "{colour:?}");
            assert!(!ce.want_icc, "{colour:?}");
            assert!(!ce.is_grey(), "{colour:?}");
            assert_eq!(ce.white_point, WhitePoint::KD65, "{colour:?}");
            assert_eq!(ce.primaries, primaries, "{colour:?}");
            assert_eq!(ce.tf, CustomTransferFunction::Enumerated(tf), "{colour:?}");
        }
    }

    #[test]
    fn a_greyscale_image_rejects_a_non_srgb_colour_space() {
        let mut w = BitWriter::new();
        assert!(matches!(
            write_metadata(&mut w, ColourShape::Grey, 8, ColourSpace::Rec2020),
            Err(EncodeError::Unsupported { .. })
        ));
    }

    #[test]
    fn an_unrepresentable_bit_depth_is_rejected() {
        let mut w = BitWriter::new();
        assert!(matches!(
            write_metadata(&mut w, ColourShape::Grey, 0, ColourSpace::Srgb),
            Err(EncodeError::ValueOutOfRange { .. })
        ));
        assert!(matches!(
            write_metadata(&mut w, ColourShape::Grey, 17, ColourSpace::Srgb),
            Err(EncodeError::ValueOutOfRange { .. })
        ));
    }

    #[test]
    fn zero_and_oversized_dimensions_are_rejected() {
        let mut w = BitWriter::new();
        assert!(matches!(
            write_size_header(&mut w, 0, 8),
            Err(EncodeError::ValueOutOfRange { .. })
        ));
        assert!(matches!(
            write_size_header(&mut w, 8, MAX_DIMENSION + 1),
            Err(EncodeError::ValueOutOfRange { .. })
        ));
    }
}
