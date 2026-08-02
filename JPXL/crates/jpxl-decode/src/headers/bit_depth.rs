//! The `BitDepth` bundle (18181-1 D.3.5).
//!
//! ```text
//! Table D.7 — BitDepth bundle
//! condition        type                       default   name
//!                  Bool()                     false     float_sample
//! !float_sample    U32(8, 10, 12, 1 + u(6))   8         bits_per_sample
//! float_sample     U32(32, 16, 24, 1 + u(6))  8         bits_per_sample
//! float_sample     1 + u(4)                   0         exp_bits
//! ```
//!
//! The two `bits_per_sample` rows are mutually exclusive and use *different*
//! distributions: the integer table starts at 8, the float table starts at 32.
//! Reading the wrong one silently yields a plausible-but-wrong depth, so the
//! branch is taken on `float_sample` before either is read.

use jpxl_bitstream::{BitReader, U32Dist, U32Spec, read_bool, read_u32, trace_field};

use crate::error::{DecodeError, Result};

/// 18181-1 D.7: `U32(8, 10, 12, 1 + u(6))`, the integer-sample distribution.
const INT_BPS_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(8),
    U32Dist::Val(10),
    U32Dist::Val(12),
    U32Dist::BitsOffset { bits: 6, offset: 1 },
]);

/// 18181-1 D.7: `U32(32, 16, 24, 1 + u(6))`, the float-sample distribution.
const FLOAT_BPS_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(32),
    U32Dist::Val(16),
    U32Dist::Val(24),
    U32Dist::BitsOffset { bits: 6, offset: 1 },
]);

/// A decoded `BitDepth` bundle (18181-1 D.3.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitDepth {
    float_sample: bool,
    bits_per_sample: u32,
    exp_bits: u32,
}

impl BitDepth {
    /// The default `BitDepth`: 8-bit integer samples (Table D.7 defaults).
    #[must_use]
    pub const fn default_int8() -> Self {
        Self {
            float_sample: false,
            bits_per_sample: 8,
            exp_bits: 0,
        }
    }

    /// Whether samples are floating point rather than integers.
    #[must_use]
    pub const fn is_float(&self) -> bool {
        self.float_sample
    }

    /// Bits per channel of the original image.
    #[must_use]
    pub const fn bits_per_sample(&self) -> u32 {
        self.bits_per_sample
    }

    /// Exponent bits, or 0 for integer samples.
    #[must_use]
    pub const fn exp_bits(&self) -> u32 {
        self.exp_bits
    }

    /// Mantissa bits for float samples: `bits_per_sample - exp_bits - 1`.
    ///
    /// Returns `None` for integer samples, where the concept does not apply.
    #[must_use]
    pub const fn mantissa_bits(&self) -> Option<u32> {
        if self.float_sample {
            Some(self.bits_per_sample - self.exp_bits - 1)
        } else {
            None
        }
    }

    /// Exponent bias for float samples: `(1 << (exp_bits - 1)) - 1`.
    #[must_use]
    pub const fn exponent_bias(&self) -> Option<u32> {
        if self.float_sample {
            Some((1 << (self.exp_bits - 1)) - 1)
        } else {
            None
        }
    }
}

impl Default for BitDepth {
    fn default() -> Self {
        Self::default_int8()
    }
}

/// Reads a `BitDepth` bundle (18181-1 D.3.5).
///
/// Enforces every range the clause states: integer depths in `[1, 31]`, float
/// depths in `[5, 32]`, `exp_bits` in `[2, 8]` and `mantissa_bits` in
/// `[2, 23]`.
///
/// # Errors
///
/// [`DecodeError::FieldOutOfRange`] if any of those ranges is violated.
pub fn read_bit_depth(reader: &mut BitReader<'_>) -> Result<BitDepth> {
    let float_sample = trace_field!(reader, "bit_depth.float_sample", read_bool(reader))?;

    if !float_sample {
        let bits_per_sample = trace_field!(
            reader,
            "bit_depth.bits_per_sample",
            read_u32(reader, &INT_BPS_SPEC)
        )?;
        if !(1..=31).contains(&bits_per_sample) {
            return Err(DecodeError::out_of_range(
                "bits_per_sample",
                "D.7",
                u64::from(bits_per_sample),
            ));
        }
        return Ok(BitDepth {
            float_sample: false,
            bits_per_sample,
            exp_bits: 0,
        });
    }

    let bits_per_sample = trace_field!(
        reader,
        "bit_depth.bits_per_sample",
        read_u32(reader, &FLOAT_BPS_SPEC)
    )?;
    if !(5..=32).contains(&bits_per_sample) {
        return Err(DecodeError::out_of_range(
            "bits_per_sample",
            "D.7",
            u64::from(bits_per_sample),
        ));
    }

    let exp_bits = trace_field!(reader, "bit_depth.exp_bits", reader.read_bits(4))? + 1;
    if !(2..=8).contains(&exp_bits) {
        return Err(DecodeError::out_of_range(
            "exp_bits",
            "D.7",
            u64::from(exp_bits),
        ));
    }

    // mantissa_bits = bits_per_sample - exp_bits - 1, in [2, 23]. Checked
    // before the subtraction so an oversized exp_bits cannot underflow.
    let used = exp_bits + 1;
    if bits_per_sample <= used {
        return Err(DecodeError::out_of_range(
            "mantissa_bits",
            "D.7",
            u64::from(bits_per_sample),
        ));
    }
    let mantissa_bits = bits_per_sample - used;
    if !(2..=23).contains(&mantissa_bits) {
        return Err(DecodeError::out_of_range(
            "mantissa_bits",
            "D.7",
            u64::from(mantissa_bits),
        ));
    }

    Ok(BitDepth {
        float_sample: true,
        bits_per_sample,
        exp_bits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testsupport::BitWriter;

    fn read(bytes: &[u8]) -> Result<BitDepth> {
        let mut r = BitReader::new(bytes);
        read_bit_depth(&mut r)
    }

    #[test]
    fn integer_shortcuts() {
        // float_sample = 0, then U32(8, 10, 12, 1 + u(6)).
        for (selector, expected) in [(0u32, 8u32), (1, 10), (2, 12)] {
            let mut w = BitWriter::new();
            w.bool(false).u32_field(selector, 0, 0);
            let bd = read(&w.finish_padded(1)).expect("valid");
            assert!(!bd.is_float());
            assert_eq!(bd.bits_per_sample(), expected);
            assert_eq!(bd.mantissa_bits(), None);
        }
    }

    #[test]
    fn integer_escape_distribution() {
        // selector 3 => 1 + u(6); payload 15 => 16 bits per sample.
        let mut w = BitWriter::new();
        w.bool(false).u32_field(3, 6, 15);
        let bd = read(&w.finish_padded(1)).expect("valid");
        assert_eq!(bd.bits_per_sample(), 16);
    }

    #[test]
    fn integer_depth_range_is_enforced() {
        // 1 + u(6) payload 0 => 1, the minimum, accepted.
        let mut w = BitWriter::new();
        w.bool(false).u32_field(3, 6, 0);
        assert_eq!(
            read(&w.finish_padded(1))
                .expect("1 is in range")
                .bits_per_sample(),
            1
        );

        // 1 + 31 = 32 is above the integer maximum of 31.
        let mut w = BitWriter::new();
        w.bool(false).u32_field(3, 6, 31);
        assert!(read(&w.finish_padded(1)).is_err());
    }

    #[test]
    fn float_uses_its_own_distribution() {
        // float_sample = 1, selector 0 => 32 (not 8: the tables differ).
        // exp_bits = 1 + u(4) with payload 7 => 8.
        let mut w = BitWriter::new();
        w.bool(true).u32_field(0, 0, 0).u(4, 7);
        let bd = read(&w.finish_padded(1)).expect("valid");

        assert!(bd.is_float());
        assert_eq!(bd.bits_per_sample(), 32);
        assert_eq!(bd.exp_bits(), 8);
        assert_eq!(bd.mantissa_bits(), Some(23), "binary32 layout");
        assert_eq!(bd.exponent_bias(), Some(127));
    }

    #[test]
    fn float16_layout() {
        // selector 1 => 16 bits, exp_bits = 1 + 4 = 5 => mantissa 10.
        let mut w = BitWriter::new();
        w.bool(true).u32_field(1, 0, 0).u(4, 4);
        let bd = read(&w.finish_padded(1)).expect("valid");
        assert_eq!(bd.bits_per_sample(), 16);
        assert_eq!(bd.exp_bits(), 5);
        assert_eq!(bd.mantissa_bits(), Some(10));
        assert_eq!(bd.exponent_bias(), Some(15));
    }

    #[test]
    fn float_exp_bits_range_is_enforced() {
        // exp_bits = 1 + 0 = 1, below the minimum of 2.
        let mut w = BitWriter::new();
        w.bool(true).u32_field(0, 0, 0).u(4, 0);
        assert!(read(&w.finish_padded(1)).is_err());

        // exp_bits = 1 + 15 = 16, above the maximum of 8.
        let mut w = BitWriter::new();
        w.bool(true).u32_field(0, 0, 0).u(4, 15);
        assert!(read(&w.finish_padded(1)).is_err());
    }

    #[test]
    fn float_mantissa_range_is_enforced() {
        // bits_per_sample = 1 + u(6) payload 5 => 6; exp_bits 8 leaves
        // mantissa = 6 - 8 - 1, which must not underflow.
        let mut w = BitWriter::new();
        w.bool(true).u32_field(3, 6, 5).u(4, 7);
        assert!(read(&w.finish_padded(1)).is_err());

        // bits_per_sample 8, exp_bits 5 => mantissa 2, the minimum.
        let mut w = BitWriter::new();
        w.bool(true).u32_field(3, 6, 7).u(4, 4);
        let bd = read(&w.finish_padded(1)).expect("mantissa 2 is in range");
        assert_eq!(bd.mantissa_bits(), Some(2));
    }

    #[test]
    fn float_depth_below_five_is_rejected() {
        // 1 + u(6) payload 3 => 4, below the float minimum of 5.
        let mut w = BitWriter::new();
        w.bool(true).u32_field(3, 6, 3).u(4, 2);
        assert!(read(&w.finish_padded(1)).is_err());
    }

    #[test]
    fn default_is_8_bit_integer() {
        let bd = BitDepth::default();
        assert!(!bd.is_float());
        assert_eq!(bd.bits_per_sample(), 8);
        assert_eq!(bd.exp_bits(), 0);
    }
}
