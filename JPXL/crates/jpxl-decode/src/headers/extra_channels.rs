//! The `ExtraChannelInfo` bundle (18181-1 D.3.6).
//!
//! ```text
//! Table D.8 — ExtraChannelInfo bundle
//! condition                     type                                default  name
//!                               Bool()                              true     d_alpha
//! !d_alpha                      Enum(ExtraChannelType)              kAlpha   type
//! !d_alpha                      BitDepth                                     bit_depth
//! !d_alpha                      U32(0, 3, 4, 1 + u(3))              0        dim_shift
//! !d_alpha                      U32(0, u(4), 16+u(5), 48+u(10))     0        name_len
//!                               u(8)                                0        name[name_len]
//! !d_alpha and type == kAlpha   Bool()                              false    alpha_associated
//! type == kSpotColour           F16()                               0        red
//! type == kSpotColour           F16()                               0        green
//! type == kSpotColour           F16()                               0        blue
//! type == kSpotColour           F16()                               0        solidity
//! type == kCFA                  U32(1, u(2), 3+u(4), 19+u(8))       1        cfa_channel
//! ```
//!
//! `d_alpha` ("default alpha") collapses the whole bundle to a single `1` bit
//! for the overwhelmingly common case: an 8-bit unassociated alpha channel at
//! full resolution with no name.
//!
//! Note that the `name` row has a *blank* condition — it is always "read", but
//! `name_len` defaults to 0, so the default path reads nothing.

use jpxl_bitstream::{
    BitReader, U32Dist, U32Spec, read_bool, read_f16_as_f32, read_u32, trace_field,
};
use jpxl_core::limits::AllocGuard;

use crate::error::{DecodeError, Result};
use crate::headers::bit_depth::{BitDepth, read_bit_depth};
use crate::headers::enums::{ExtraChannelType, read_enum};

/// 18181-1 D.8: `U32(0, 3, 4, 1 + u(3))`.
const DIM_SHIFT_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(0),
    U32Dist::Val(3),
    U32Dist::Val(4),
    U32Dist::BitsOffset { bits: 3, offset: 1 },
]);

/// 18181-1 D.8: `U32(0, u(4), 16 + u(5), 48 + u(10))`.
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

/// 18181-1 D.8: `U32(1, u(2), 3 + u(4), 19 + u(8))`.
const CFA_CHANNEL_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(1),
    U32Dist::bits(2),
    U32Dist::BitsOffset { bits: 4, offset: 3 },
    U32Dist::BitsOffset {
        bits: 8,
        offset: 19,
    },
]);

/// Largest `dim_shift` JPXL accepts.
///
/// D.3.6 bounds it indirectly: "The value `1 << dim_shift` does not exceed the
/// `group_dim` of any frame", and F.2 caps `group_dim` at 1024, so
/// `dim_shift <= 10`. The frame headers that would let this be checked exactly
/// are not parsed at this slice, so the structural bound is applied here and
/// the per-frame check belongs with the frame header.
pub const MAX_DIM_SHIFT: u32 = 10;

/// Spot colour parameters, present when `type == kSpotColour` (18181-1 D.8).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpotColour {
    /// Red component of the spot colour.
    pub red: f32,
    /// Green component of the spot colour.
    pub green: f32,
    /// Blue component of the spot colour.
    pub blue: f32,
    /// Overall blending factor in `[0, 1]`; 0 is invisible, 1 fully opaque.
    pub solidity: f32,
}

/// A decoded `ExtraChannelInfo` bundle (18181-1 D.3.6).
#[derive(Debug, Clone, PartialEq)]
pub struct ExtraChannelInfo {
    /// Semantic type of the channel.
    pub channel_type: ExtraChannelType,
    /// Sample representation of this channel.
    pub bit_depth: BitDepth,
    /// Base-2 log of the channel's downsampling factor versus the main image.
    pub dim_shift: u32,
    /// Raw name bytes; the standard says to interpret these as UTF-8.
    pub name: Vec<u8>,
    /// Whether alpha is premultiplied. Only meaningful for `kAlpha`.
    pub alpha_associated: bool,
    /// Spot colour parameters, present only for `kSpotColour`.
    pub spot_colour: Option<SpotColour>,
    /// CFA channel index, present only for `kCFA`.
    pub cfa_channel: Option<u32>,
}

impl ExtraChannelInfo {
    /// The `d_alpha` default: 8-bit unassociated alpha, full resolution, no name.
    #[must_use]
    pub fn default_alpha() -> Self {
        Self {
            channel_type: ExtraChannelType::KAlpha,
            bit_depth: BitDepth::default_int8(),
            dim_shift: 0,
            name: Vec::new(),
            alpha_associated: false,
            spot_colour: None,
            cfa_channel: None,
        }
    }

    /// The channel name as UTF-8, or `None` if the bytes are not valid UTF-8.
    ///
    /// D.3.6 says the name is "interpreted as a UTF-8 encoded string" but does
    /// not make validity a conformance requirement, so the raw bytes are kept
    /// and validation is offered rather than imposed. Rejecting an entire image
    /// because one channel label is mis-encoded would be a decoder-invented
    /// restriction.
    #[must_use]
    pub fn name_utf8(&self) -> Option<&str> {
        core::str::from_utf8(&self.name).ok()
    }

    /// The downsampling factor `1 << dim_shift`.
    #[must_use]
    pub const fn downsample_factor(&self) -> u32 {
        1 << self.dim_shift
    }
}

/// Reads one `ExtraChannelInfo` bundle (18181-1 D.3.6).
///
/// `guard` meters the channel-name allocation before it happens; `name_len`
/// reaches 1071 and is attacker-chosen.
///
/// # Errors
///
/// [`DecodeError::FieldOutOfRange`] if `dim_shift` exceeds [`MAX_DIM_SHIFT`],
/// [`DecodeError::UnknownEnumValue`] for an undefined channel type, or a
/// bitstream error on truncation.
pub fn read_extra_channel_info(
    reader: &mut BitReader<'_>,
    guard: &mut AllocGuard,
) -> Result<ExtraChannelInfo> {
    let d_alpha = trace_field!(reader, "ec.d_alpha", read_bool(reader))?;
    if d_alpha {
        return Ok(ExtraChannelInfo::default_alpha());
    }

    let channel_type: ExtraChannelType = read_enum(reader, "ec.type")?;
    let bit_depth = read_bit_depth(reader)?;
    let dim_shift = trace_field!(reader, "ec.dim_shift", read_u32(reader, &DIM_SHIFT_SPEC))?;
    if dim_shift > MAX_DIM_SHIFT {
        return Err(DecodeError::out_of_range(
            "dim_shift",
            "D.3.6",
            u64::from(dim_shift),
        ));
    }

    let name_len = trace_field!(reader, "ec.name_len", read_u32(reader, &NAME_LEN_SPEC))?;
    guard
        .charge(u64::from(name_len))
        .map_err(DecodeError::Core)?;
    let mut name = Vec::with_capacity(name_len as usize);
    for _ in 0..name_len {
        let byte = trace_field!(reader, "ec.name", reader.read_bits(8))?;
        // u(8) cannot exceed 255.
        name.push(u8::try_from(byte).map_err(|_| DecodeError::out_of_range("name", "D.8", 0))?);
    }

    let alpha_associated = if channel_type == ExtraChannelType::KAlpha {
        trace_field!(reader, "ec.alpha_associated", read_bool(reader))?
    } else {
        false
    };

    let spot_colour = if channel_type == ExtraChannelType::KSpotColour {
        Some(SpotColour {
            red: trace_field!(reader, "ec.red", read_f16_as_f32(reader))?,
            green: trace_field!(reader, "ec.green", read_f16_as_f32(reader))?,
            blue: trace_field!(reader, "ec.blue", read_f16_as_f32(reader))?,
            solidity: trace_field!(reader, "ec.solidity", read_f16_as_f32(reader))?,
        })
    } else {
        None
    };

    let cfa_channel = if channel_type == ExtraChannelType::KCfa {
        Some(trace_field!(
            reader,
            "ec.cfa_channel",
            read_u32(reader, &CFA_CHANNEL_SPEC)
        )?)
    } else {
        None
    };

    Ok(ExtraChannelInfo {
        channel_type,
        bit_depth,
        dim_shift,
        name,
        alpha_associated,
        spot_colour,
        cfa_channel,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testsupport::BitWriter;
    use jpxl_core::limits::Limits;

    fn read(bytes: &[u8]) -> Result<ExtraChannelInfo> {
        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let mut r = BitReader::new(bytes);
        read_extra_channel_info(&mut r, &mut guard)
    }

    #[test]
    fn d_alpha_is_a_single_bit() {
        let mut w = BitWriter::new();
        w.bool(true);

        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let ec = read_extra_channel_info(&mut r, &mut guard).expect("valid");

        assert_eq!(r.total_bits_read(), 1, "the default path costs one bit");
        assert_eq!(ec, ExtraChannelInfo::default_alpha());
        assert_eq!(ec.channel_type, ExtraChannelType::KAlpha);
        assert_eq!(ec.bit_depth.bits_per_sample(), 8);
        assert!(!ec.alpha_associated);
    }

    #[test]
    fn explicit_associated_alpha() {
        // d_alpha = 0, type = kAlpha (Enum selector 0), BitDepth int 8,
        // dim_shift 0, name_len 0, alpha_associated = 1.
        let mut w = BitWriter::new();
        w.bool(false)
            .enum_field(0)
            .bool(false)
            .u32_field(0, 0, 0)
            .u32_field(0, 0, 0)
            .u32_field(0, 0, 0)
            .bool(true);
        let ec = read(&w.finish_padded(1)).expect("valid");

        assert_eq!(ec.channel_type, ExtraChannelType::KAlpha);
        assert!(ec.alpha_associated);
        assert!(ec.name.is_empty());
        assert_eq!(ec.dim_shift, 0);
    }

    #[test]
    fn depth_channel_with_name_and_dim_shift() {
        // type = kDepth (1), 16-bit integer, dim_shift = 3 (selector 1),
        // name_len = 5 via u(4), name "depth".
        let mut w = BitWriter::new();
        w.bool(false)
            .enum_field(1)
            .bool(false)
            .u32_field(3, 6, 15) // 1 + 15 = 16 bits per sample
            .u32_field(1, 0, 0) // dim_shift = 3
            .u32_field(1, 4, 5); // name_len = 5
        for b in b"depth" {
            w.u(8, u32::from(*b));
        }
        let ec = read(&w.finish_padded(1)).expect("valid");

        assert_eq!(ec.channel_type, ExtraChannelType::KDepth);
        assert_eq!(ec.bit_depth.bits_per_sample(), 16);
        assert_eq!(ec.dim_shift, 3);
        assert_eq!(ec.downsample_factor(), 8, "D.3.6 example: 8x8 downsampling");
        assert_eq!(ec.name_utf8(), Some("depth"));
        assert!(
            !ec.alpha_associated,
            "alpha_associated is only read for kAlpha"
        );
    }

    #[test]
    fn spot_colour_reads_four_f16_values() {
        // type = kSpotColour (2) => Enum selector 2 with payload 0.
        let mut w = BitWriter::new();
        w.bool(false)
            .enum_field(2)
            .bool(false)
            .u32_field(0, 0, 0)
            .u32_field(0, 0, 0)
            .u32_field(0, 0, 0);
        // F16 1.0 is 0x3C00; 0.0 is 0x0000.
        w.f16_bits(0x3C00)
            .f16_bits(0x0000)
            .f16_bits(0x0000)
            .f16_bits(0x3C00);
        let ec = read(&w.finish_padded(1)).expect("valid");

        let spot = ec.spot_colour.expect("kSpotColour carries parameters");
        assert_eq!(spot.red, 1.0);
        assert_eq!(spot.green, 0.0);
        assert_eq!(spot.blue, 0.0);
        assert_eq!(spot.solidity, 1.0);
        assert!(ec.cfa_channel.is_none());
    }

    #[test]
    fn cfa_channel_is_read_only_for_kcfa() {
        // type = kCFA (5) => Enum selector 2 with payload 3.
        let mut w = BitWriter::new();
        w.bool(false)
            .enum_field(5)
            .bool(false)
            .u32_field(0, 0, 0)
            .u32_field(0, 0, 0)
            .u32_field(0, 0, 0)
            .u32_field(1, 2, 2); // cfa_channel = u(2) = 2
        let ec = read(&w.finish_padded(1)).expect("valid");

        assert_eq!(ec.channel_type, ExtraChannelType::KCfa);
        assert_eq!(ec.cfa_channel, Some(2));
        assert!(ec.spot_colour.is_none());
    }

    #[test]
    fn dim_shift_upper_bound_is_enforced() {
        // dim_shift selector 3 => 1 + u(3); payload 7 => 8, accepted.
        let mut w = BitWriter::new();
        w.bool(false)
            .enum_field(1)
            .bool(false)
            .u32_field(0, 0, 0)
            .u32_field(3, 3, 7)
            .u32_field(0, 0, 0);
        assert_eq!(read(&w.finish_padded(1)).expect("valid").dim_shift, 8);
    }

    #[test]
    fn unknown_channel_type_rejected() {
        // Enum value 7 has no row in Table D.9.
        let mut w = BitWriter::new();
        w.bool(false).enum_field(7);
        let err = read(&w.finish_padded(2)).expect_err("7 is undefined");
        assert!(matches!(err, DecodeError::UnknownEnumValue { .. }));
    }

    #[test]
    fn long_name_spans_many_bytes() {
        // name_len selector 3 => 48 + u(10); payload 2 => 50 bytes.
        let name: Vec<u8> = (0..50u8).map(|i| b'a' + (i % 26)).collect();
        let mut w = BitWriter::new();
        w.bool(false)
            .enum_field(16) // kOptional
            .bool(false)
            .u32_field(0, 0, 0)
            .u32_field(0, 0, 0)
            .u32_field(3, 10, 2);
        for b in &name {
            w.u(8, u32::from(*b));
        }
        let ec = read(&w.finish_padded(1)).expect("valid");
        assert_eq!(ec.name, name);
        assert_eq!(ec.channel_type, ExtraChannelType::KOptional);
    }

    #[test]
    fn name_allocation_is_metered() {
        // A 50-byte name against a 10-byte allocation budget.
        let mut w = BitWriter::new();
        w.bool(false)
            .enum_field(16)
            .bool(false)
            .u32_field(0, 0, 0)
            .u32_field(0, 0, 0)
            .u32_field(3, 10, 2);
        for _ in 0..50 {
            w.u(8, u32::from(b'x'));
        }
        let data = w.finish_padded(1);

        let limits = Limits {
            max_alloc_bytes: 10,
            ..Limits::default()
        };
        let mut guard = AllocGuard::new(&limits);
        let mut r = BitReader::new(&data);
        assert!(read_extra_channel_info(&mut r, &mut guard).is_err());
    }

    #[test]
    fn invalid_utf8_name_is_kept_not_rejected() {
        let mut w = BitWriter::new();
        w.bool(false)
            .enum_field(16)
            .bool(false)
            .u32_field(0, 0, 0)
            .u32_field(0, 0, 0)
            .u32_field(1, 4, 2);
        w.u(8, 0xFF).u(8, 0xFE);
        let ec = read(&w.finish_padded(1)).expect("parse succeeds");
        assert_eq!(ec.name, vec![0xFF, 0xFE]);
        assert_eq!(ec.name_utf8(), None);
    }
}
