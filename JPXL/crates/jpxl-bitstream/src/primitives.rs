//! Normative integer and float primitives of the JPEG XL header syntax.
//!
//! Part 1 notational conventions \[provisional: verify vs 18181-1 OCR\]:
//! `u(n)` is `n` bits read LSB-first; `Bool()` is `u(1)`; `U32(d0, d1, d2, d3)`
//! reads a 2-bit selector `k = u(2)` and then decodes `d[k]`; `U64()` is
//! `U32(0, 1 + u(4), 17 + u(8), longU64())`; `F16()` is a binary16 value.

use crate::error::{BitstreamError, Result};
use crate::reader::BitReader;

/// One of the four alternatives of a `U32()` field.
///
/// A distribution is either a constant, or an offset plus a raw `u(n)` payload.
/// `Val(v)` is exactly `BitsOffset { bits: 0, offset: v }`; both are provided
/// because the spec tables are written in both forms (`U32(1, 2, 4, 8)` vs.
/// `U32(1 + u(9), …)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum U32Dist {
    /// A constant value; no further bits are read.
    Val(u32),
    /// `offset + u(bits)`. `bits` must be at most 32.
    BitsOffset {
        /// Payload width in bits.
        bits: u8,
        /// Constant added to the payload.
        offset: u32,
    },
}

impl U32Dist {
    /// Shorthand for a bare `u(bits)` alternative (offset 0).
    #[must_use]
    pub const fn bits(bits: u8) -> Self {
        Self::BitsOffset { bits, offset: 0 }
    }
}

/// The four alternatives of a `U32()` field, indexed by the 2-bit selector.
///
/// Const-constructible so header code can write
/// `const NUM_LOOPS: U32Spec = U32Spec::new([…]);`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct U32Spec(pub [U32Dist; 4]);

impl U32Spec {
    /// Builds a spec from the four alternatives in selector order.
    #[must_use]
    pub const fn new(dists: [U32Dist; 4]) -> Self {
        Self(dists)
    }
}

/// Reads `Bool()`, i.e. `u(1) != 0`.
pub fn read_bool(reader: &mut BitReader<'_>) -> Result<bool> {
    reader.read_bool()
}

/// Reads a `U32(d0, d1, d2, d3)` field.
///
/// The 2-bit selector chooses the distribution; a `BitsOffset` alternative then
/// reads its payload and adds the offset. If `offset + payload` does not fit in
/// a `u32`, [`BitstreamError::Overflow`] is returned; likewise for a payload
/// width above 32 bits.
pub fn read_u32(reader: &mut BitReader<'_>, spec: &U32Spec) -> Result<u32> {
    let selector = usize::try_from(reader.read_bits(2)?).map_err(|_| BitstreamError::Overflow)?;
    let dist = spec
        .0
        .get(selector)
        .copied()
        .ok_or(BitstreamError::Overflow)?;
    match dist {
        U32Dist::Val(v) => Ok(v),
        U32Dist::BitsOffset { bits, offset } => {
            if bits > 32 {
                return Err(BitstreamError::Overflow);
            }
            let payload = reader.read_bits(u32::from(bits))?;
            offset.checked_add(payload).ok_or(BitstreamError::Overflow)
        }
    }
}

/// Reads a `U64()` field.
///
/// `U64() = U32(0, 1 + u(4), 17 + u(8), longU64())` where
/// `longU64()` is `v = u(12); s = 12; while (u(1) == 1) { if (s == 60)
/// { v += u(4) << s; break; } v += u(8) << s; s += 8; }`.
pub fn read_u64(reader: &mut BitReader<'_>) -> Result<u64> {
    match reader.read_bits(2)? {
        0 => Ok(0),
        1 => Ok(1 + u64::from(reader.read_bits(4)?)),
        2 => Ok(17 + u64::from(reader.read_bits(8)?)),
        _ => {
            let mut value = u64::from(reader.read_bits(12)?);
            let mut shift = 12u32;
            while reader.read_bool()? {
                if shift == 60 {
                    value |= u64::from(reader.read_bits(4)?) << 60;
                    break;
                }
                value |= u64::from(reader.read_bits(8)?) << shift;
                shift += 8;
            }
            Ok(value)
        }
    }
}

/// Reads an `F16()` field and converts it to `f32`.
///
/// The 16 bits are read LSB-first, so they form a little-endian binary16:
/// mantissa (10) | exponent (5) | sign (1). Conversion is done in software
/// (bit manipulation only) so the result is bit-identical on every target.
///
/// An exponent field of 31 encodes an infinity or a NaN and is rejected with
/// [`BitstreamError::InvalidF16`]. Subnormals, zero and negative zero are
/// valid and round-trip exactly \[provisional: verify vs 18181-1 OCR\].
pub fn read_f16_as_f32(reader: &mut BitReader<'_>) -> Result<f32> {
    let raw = reader.read_bits(16)?;
    let sign = (raw & 0x8000) << 16;
    let exponent = (raw >> 10) & 0x1F;
    let mantissa = raw & 0x03FF;

    if exponent == 31 {
        return Err(BitstreamError::InvalidF16);
    }

    let bits = if exponent == 0 {
        if mantissa == 0 {
            // (Signed) zero.
            sign
        } else {
            // Subnormal: value = mantissa * 2^-24. Normalize by finding the
            // highest set bit k, so value = 2^(k-24) * (1 + r/2^k) with
            // r = mantissa - 2^k. The f32 exponent field is k - 24 + 127 and
            // the f32 mantissa field is r << (23 - k).
            let k = 31 - mantissa.leading_zeros();
            let exp_field = k + 103;
            let frac = (mantissa << (23 - k)) & 0x007F_FFFF;
            sign | (exp_field << 23) | frac
        }
    } else {
        // Normal: half bias 15, single bias 127 => exponent field + 112.
        sign | ((exponent + 112) << 23) | (mantissa << 13)
    };
    Ok(f32::from_bits(bits))
}
