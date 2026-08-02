//! `Enum(EnumTable)` fields (18181-1 B.2.6) and the enumerated types used by
//! the image header.
//!
//! B.2.6 defines the wire format once, for every enumerated type:
//!
//! ```text
//! Enum(EnumTable) reads v = U32(0, 1, 2 + u(4), 18 + u(6)).
//! The value v does not exceed 63 and is defined by the table titled EnumTable.
//! ```
//!
//! Two distinct rejections follow, and both are enforced here: a value above 63
//! violates B.2.6 itself regardless of table, and a value within range but
//! absent from the table has no defined meaning.

use jpxl_bitstream::{BitReader, U32Dist, U32Spec, read_u32, trace_field};

use crate::error::{DecodeError, Result};

/// 18181-1 B.2.6: `v = U32(0, 1, 2 + u(4), 18 + u(6))`.
const ENUM_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(0),
    U32Dist::Val(1),
    U32Dist::BitsOffset { bits: 4, offset: 2 },
    U32Dist::BitsOffset {
        bits: 6,
        offset: 18,
    },
]);

/// Largest value an `Enum()` field may take (18181-1 B.2.6).
pub const MAX_ENUM_VALUE: u32 = 63;

/// An enumerated type defined by a table in the standard.
///
/// Implementors are plain C-like enums; [`read_enum`] handles the wire format
/// so no implementor repeats it.
pub trait EnumTable: Sized {
    /// Table name as titled in the standard, e.g. `"ExtraChannelType"`.
    const TABLE_NAME: &'static str;
    /// Clause containing the table, e.g. `"D.9"`.
    const CLAUSE: &'static str;

    /// Maps a wire value to a table row, or `None` if the table has no such row.
    fn from_value(value: u32) -> Option<Self>;
}

/// Reads an `Enum(EnumTable)` field (18181-1 B.2.6).
///
/// # Errors
///
/// [`DecodeError::UnknownEnumValue`] if the value exceeds
/// [`MAX_ENUM_VALUE`] or names no row of `T`'s table.
pub fn read_enum<T: EnumTable>(reader: &mut BitReader<'_>, field: &'static str) -> Result<T> {
    let value = trace_field!(reader, field, read_u32(reader, &ENUM_SPEC))?;
    if value > MAX_ENUM_VALUE {
        return Err(DecodeError::UnknownEnumValue {
            table: T::TABLE_NAME,
            clause: "B.2.6",
            value,
        });
    }
    T::from_value(value).ok_or(DecodeError::UnknownEnumValue {
        table: T::TABLE_NAME,
        clause: T::CLAUSE,
        value,
    })
}

/// Generates a C-like enum plus its [`EnumTable`] impl from a spec table.
macro_rules! spec_enum {
    (
        $(#[$meta:meta])*
        $name:ident, table = $table:expr, clause = $clause:expr, default = $default:ident {
            $($(#[$vmeta:meta])* $variant:ident = $value:expr),+ $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[non_exhaustive]
        pub enum $name {
            $($(#[$vmeta])* $variant = $value),+
        }

        impl $name {
            /// The wire value of this row.
            #[must_use]
            pub const fn value(self) -> u32 {
                match self {
                    $(Self::$variant => $value),+
                }
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::$default
            }
        }

        impl EnumTable for $name {
            const TABLE_NAME: &'static str = $table;
            const CLAUSE: &'static str = $clause;

            fn from_value(value: u32) -> Option<Self> {
                match value {
                    $($value => Some(Self::$variant),)+
                    _ => None,
                }
            }
        }
    };
}

spec_enum! {
    /// 18181-1 Table D.9 — `ExtraChannelType`.
    ExtraChannelType, table = "ExtraChannelType", clause = "D.9", default = KAlpha {
        /// Alpha transparency, where 0 means fully transparent.
        KAlpha = 0,
        /// Depth map; higher values mean farther from the camera.
        KDepth = 1,
        /// Spot colour channel; see `red`/`green`/`blue`/`solidity`.
        KSpotColour = 2,
        /// Selection mask marking a (fuzzy) region of interest.
        KSelectionMask = 3,
        /// The K channel of a CMYK image.
        KBlack = 4,
        /// Colour Filter Array (Bayer mosaic) data.
        KCfa = 5,
        /// Infrared thermography; higher values mean warmer.
        KThermal = 6,
        /// The decoder cannot safely interpret this channel's semantics.
        KNonOptional = 15,
        /// An extra channel that can be safely ignored.
        KOptional = 16,
    }
}

spec_enum! {
    /// 18181-1 Table E.3 — `ColourSpace`.
    ColourSpace, table = "ColourSpace", clause = "E.3", default = KRgb {
        /// Tristimulus RGB with the given white point and primaries.
        KRgb = 0,
        /// Luminance with a given white point; primaries are not read.
        KGrey = 1,
        /// XYB (opsin).
        KXyb = 2,
        /// None of the other entries describe the colour space.
        KUnknown = 3,
    }
}

spec_enum! {
    /// 18181-1 Table E.4 — `WhitePoint`.
    ///
    /// The meaning column holds CIE xy chromaticity coordinates.
    WhitePoint, table = "WhitePoint", clause = "E.4", default = KD65 {
        /// CIE Standard Illuminant D65: 0.3127, 0.3290.
        KD65 = 1,
        /// Custom white point stored in `colour_encoding.white`.
        KCustom = 2,
        /// CIE Standard Illuminant E (equal-energy): 1/3, 1/3.
        KE = 10,
        /// DCI-P3 from SMPTE ST 428-1: 0.314, 0.351.
        KDci = 11,
    }
}

spec_enum! {
    /// 18181-1 Table E.5 — `Primaries`.
    Primaries, table = "Primaries", clause = "E.5", default = KSrgb {
        /// sRGB primaries, quantized as an ICC profile would store them.
        KSrgb = 1,
        /// Custom primaries stored in `colour_encoding.red`/`green`/`blue`.
        KCustom = 2,
        /// ITU-R BT.2100-2 primaries.
        K2100 = 9,
        /// SMPTE ST 428-1 (DCI-P3) primaries.
        KP3 = 11,
    }
}

spec_enum! {
    /// 18181-1 Table E.6 — `TransferFunction`.
    TransferFunction, table = "TransferFunction", clause = "E.6", default = KSrgb {
        /// ITU-R BT.709-6.
        K709 = 1,
        /// None of the other entries describe the transfer function.
        KUnknown = 2,
        /// Gamma exponent 1.
        KLinear = 8,
        /// IEC 61966-2-1 (sRGB).
        KSrgb = 13,
        /// ITU-R BT.2100-2 (PQ).
        KPq = 16,
        /// SMPTE ST 428-1.
        KDci = 17,
        /// ITU-R BT.2100-2 (HLG).
        KHlg = 18,
    }
}

spec_enum! {
    /// 18181-1 Table E.8 — `RenderingIntent`, as defined by ISO 15076-1.
    RenderingIntent, table = "RenderingIntent", clause = "E.8", default = KRelative {
        /// Perceptual (vendor-specific).
        KPerceptual = 0,
        /// Media-relative colorimetric.
        KRelative = 1,
        /// Saturation (vendor-specific).
        KSaturation = 2,
        /// ICC-absolute colorimetric.
        KAbsolute = 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encodes `Enum()` selector 0 or 1 (the two constant distributions).
    #[test]
    fn reads_constant_distributions() {
        // B.2.6: U32(0, 1, 2 + u(4), 18 + u(6)).
        // selector 0 (bits "00" LSB-first) => value 0 => kRGB.
        let data = [0b0000_0000u8];
        let mut r = BitReader::new(&data);
        let cs: ColourSpace = read_enum(&mut r, "colour_space").expect("valid");
        assert_eq!(cs, ColourSpace::KRgb);
        assert_eq!(r.total_bits_read(), 2);

        // selector 1 => value 1 => kGrey.
        let data = [0b0000_0001u8];
        let mut r = BitReader::new(&data);
        let cs: ColourSpace = read_enum(&mut r, "colour_space").expect("valid");
        assert_eq!(cs, ColourSpace::KGrey);
    }

    #[test]
    fn reads_offset_distributions() {
        // selector 2 => 2 + u(4). Want value 4 (kBlack) => payload 2.
        // LSB-first bits b0..b5 = 0,1 (selector 2) then 0,1,0,0 (payload 2),
        // so the byte is 0b0000_1010.
        let data = [0b0000_1010u8];
        let mut r = BitReader::new(&data);
        let t: ExtraChannelType = read_enum(&mut r, "type").expect("valid");
        assert_eq!(t, ExtraChannelType::KBlack);
        assert_eq!(r.total_bits_read(), 6);

        // selector 2 => 2 + u(4). Want 16 (kOptional) => payload 14.
        // LSB-first bits: 0,1 then 0,1,1,1 => 0b0011_1010.
        let data = [0b0011_1010u8];
        let mut r = BitReader::new(&data);
        let t: ExtraChannelType = read_enum(&mut r, "type").expect("valid");
        assert_eq!(t, ExtraChannelType::KOptional);
        assert_eq!(r.total_bits_read(), 6);

        // selector 3 => 18 + u(6). Table E.6 defines 18 as kHLG.
        // LSB-first bits: 1,1 then six 0s => byte 0b0000_0011.
        let data = [0b0000_0011u8];
        let mut r = BitReader::new(&data);
        let tf: TransferFunction = read_enum(&mut r, "transfer_function").expect("valid");
        assert_eq!(tf, TransferFunction::KHlg);
        assert_eq!(r.total_bits_read(), 8);
    }

    #[test]
    fn rejects_value_above_63() {
        // selector 3 => 18 + u(6); payload 63 => 81 > 63, rejected by B.2.6.
        // LSB-first bits: 1,1 (selector 3) then six 1s => byte 0xFF.
        let data = [0b1111_1111u8];
        let mut r = BitReader::new(&data);
        let err = read_enum::<ExtraChannelType>(&mut r, "type").expect_err("81 > 63");
        match err {
            DecodeError::UnknownEnumValue {
                value,
                clause,
                table,
            } => {
                assert_eq!(value, 81);
                assert_eq!(clause, "B.2.6");
                assert_eq!(table, "ExtraChannelType");
            }
            other => panic!("wrong error: {other}"),
        }
    }

    #[test]
    fn rejects_value_absent_from_table() {
        // selector 2 => 2 + u(4); payload 5 => 7, which Table D.9 does not define.
        // LSB-first bits: 0,1 (selector 2) then 1,0,1,0 (payload 5) => 0b0001_0110.
        let data = [0b0001_0110u8];
        let mut r = BitReader::new(&data);
        let err = read_enum::<ExtraChannelType>(&mut r, "type").expect_err("7 is not a row");
        match err {
            DecodeError::UnknownEnumValue { value, clause, .. } => {
                assert_eq!(value, 7);
                assert_eq!(clause, "D.9", "in-range misses cite the table's clause");
            }
            other => panic!("wrong error: {other}"),
        }
    }

    #[test]
    fn table_values_match_the_standard() {
        assert_eq!(ExtraChannelType::KAlpha.value(), 0);
        assert_eq!(ExtraChannelType::KBlack.value(), 4);
        assert_eq!(ExtraChannelType::KNonOptional.value(), 15);
        assert_eq!(ExtraChannelType::KOptional.value(), 16);
        assert_eq!(WhitePoint::KDci.value(), 11);
        assert_eq!(Primaries::K2100.value(), 9);
        assert_eq!(TransferFunction::KHlg.value(), 18);
        assert_eq!(RenderingIntent::KAbsolute.value(), 3);
        assert_eq!(ColourSpace::KXyb.value(), 2);
    }

    #[test]
    fn round_trips_every_defined_value() {
        for t in [
            ExtraChannelType::KAlpha,
            ExtraChannelType::KDepth,
            ExtraChannelType::KSpotColour,
            ExtraChannelType::KSelectionMask,
            ExtraChannelType::KBlack,
            ExtraChannelType::KCfa,
            ExtraChannelType::KThermal,
            ExtraChannelType::KNonOptional,
            ExtraChannelType::KOptional,
        ] {
            assert_eq!(ExtraChannelType::from_value(t.value()), Some(t));
        }
    }
}
