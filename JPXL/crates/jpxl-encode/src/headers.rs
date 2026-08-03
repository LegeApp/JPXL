//! Signature, `SizeHeader` and `ImageMetadata` for a greyscale 8-bit image
//! (18181-1 D.1, D.2, D.3, E.2).
//!
//! # Why `all_default` is not usable here
//!
//! The Table D.3 defaults describe an **XYB-encoded sRGB** image. A greyscale
//! non-XYB image differs in two of the guarded rows (`xyb_encoded` and
//! `colour_encoding.colour_space`), and Table D.3 has no way to override one
//! row while defaulting the rest — `all_default` is all or nothing. So the
//! bundle is written out field by field, taking the table's own default for
//! every row that can keep it.
//!
//! The one row that is *not* guarded by `all_default` is `default_m`, which is
//! written last and set, meaning "use the L.1 opsin matrix and the K.2
//! upsampling weights". It costs one bit even though nothing in a modular
//! greyscale frame consults either.

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

/// Table E.3 `ColourSpace`: `kGrey`.
const COLOUR_SPACE_GREY: u32 = 1;
/// Table E.4 `WhitePoint`: `kD65`, the bundle default.
const WHITE_POINT_D65: u32 = 1;
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

/// Writes an `ImageMetadata` bundle describing an 8-bit greyscale still image
/// (18181-1 D.3, Table D.3).
///
/// # Errors
///
/// Only through the bit writer.
pub fn write_grey8_metadata(w: &mut BitWriter) -> Result<()> {
    w.write_bool(false); // all_default: see the module documentation
    w.write_bool(false); // extra_fields: no orientation, preview or animation

    // BitDepth (D.3.5, Table D.7): 8-bit integer samples.
    w.write_bool(false); // float_sample
    w.write_u32(&INT_BPS_SPEC, 8)?;

    w.write_bool(true); // modular_16bit_buffers, the table default
    w.write_u32(&NUM_EXTRA_SPEC, 0)?; // no extra channels
    w.write_bool(false); // xyb_encoded: samples are stored as-is

    write_grey_colour_encoding(w)?;

    // tone_mapping is guarded by extra_fields, which is false.
    w.write_u64(0)?; // extensions (B.3)

    // Blank condition in Table D.3: read even under all_default.
    w.write_bool(true); // default_m
    Ok(())
}

/// Writes a `ColourEncoding` bundle for greyscale sRGB (18181-1 E.2, Table E.1).
///
/// `use_desc` is true (no ICC profile) and `has_primaries` is false because
/// the colour space is `kGrey`, so the primaries rows are skipped but the
/// white point is still written.
fn write_grey_colour_encoding(w: &mut BitWriter) -> Result<()> {
    w.write_bool(false); // all_default (the default is kRGB)
    w.write_bool(false); // want_icc
    w.write_u32(&ENUM_SPEC, COLOUR_SPACE_GREY)?;
    w.write_u32(&ENUM_SPEC, WHITE_POINT_D65)?;
    // primaries / red / green / blue: skipped for kGrey.
    // CustomTransferFunction (Table E.7).
    w.write_bool(false); // have_gamma
    w.write_u32(&ENUM_SPEC, TRANSFER_FUNCTION_SRGB)?;
    w.write_u32(&ENUM_SPEC, RENDERING_INTENT_RELATIVE)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use jpxl_bitstream::BitReader;
    use jpxl_core::limits::Limits;
    use jpxl_decode::headers::decode_image_headers;

    fn headers(width: u32, height: u32) -> Vec<u8> {
        let mut w = BitWriter::new();
        write_signature(&mut w).expect("signature");
        write_size_header(&mut w, width, height).expect("size");
        write_grey8_metadata(&mut w).expect("metadata");
        w.zero_pad_to_byte();
        w.into_bytes()
    }

    #[test]
    fn headers_round_trip_through_the_decoder() {
        for (width, height) in [(1u32, 1u32), (8, 8), (13, 7), (300, 200), (512, 512)] {
            let bytes = headers(width, height);
            let mut r = BitReader::new(&bytes);
            let parsed = decode_image_headers(&mut r, &Limits::default())
                .unwrap_or_else(|e| panic!("{width}x{height}: {e}"));

            assert_eq!(parsed.width(), width);
            assert_eq!(parsed.height(), height);
            assert!(!parsed.metadata.xyb_encoded);
            assert!(parsed.metadata.colour_encoding.is_grey());
            assert_eq!(parsed.metadata.bit_depth.bits_per_sample(), 8);
            assert!(parsed.metadata.ec_info.is_empty());
            assert!(parsed.metadata.default_m);
            assert!(parsed.metadata.preview.is_none());
            assert!(parsed.metadata.animation.is_none());
        }
    }

    #[test]
    fn the_header_ends_where_the_writer_says_it_does() {
        // A wrong conditional shifts every later field, so the bit count is
        // the assertion that actually catches it.
        let mut w = BitWriter::new();
        write_signature(&mut w).expect("signature");
        write_size_header(&mut w, 64, 64).expect("size");
        write_grey8_metadata(&mut w).expect("metadata");
        let written = w.bit_len();
        w.zero_pad_to_byte();
        let bytes = w.into_bytes();

        let mut r = BitReader::new(&bytes);
        decode_image_headers(&mut r, &Limits::default()).expect("valid");
        assert_eq!(r.total_bits_read(), written);
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
