//! Coefficient magnitude coding helpers (10918-1 F.1.2.1, F.2.2.1).
//!
//! JPEG codes a signed coefficient (or DC difference) as a *magnitude
//! category* `S` — the number of significant bits — plus `S` mantissa bits.
//! These are the two directions of that mapping, shared by the baseline and
//! progressive scan coders.

use crate::error::{JpegError, Result};

/// Rebuilds a signed value from `S` received bits (`EXTEND`, F.2.2.1).
#[must_use]
pub fn extend(received: u32, size: u32) -> i32 {
    if size == 0 {
        return 0;
    }
    let v = received as i32;
    let threshold = 1i32 << (size - 1);
    if v < threshold {
        v + (-1i32 << size) + 1
    } else {
        v
    }
}

/// The magnitude category `S` (number of significant bits) of a value.
#[must_use]
pub fn magnitude_category(value: i32) -> u32 {
    let m = value.unsigned_abs();
    32 - m.leading_zeros()
}

/// The `S` mantissa bits JPEG appends for `value` (F.1.2.1): the low `size`
/// bits of `value` when non-negative, or of `value - 1` when negative.
#[must_use]
pub fn mantissa_bits(value: i32, size: u32) -> u32 {
    if size == 0 {
        return 0;
    }
    let mask = if size >= 32 {
        u32::MAX
    } else {
        (1u32 << size) - 1
    };
    let raw = if value < 0 { value - 1 } else { value } as u32;
    raw & mask
}

/// Narrows a decoded coefficient to `i16`, rejecting values that do not fit —
/// which for an 8-bit frame signals corruption or an out-of-scope precision.
pub fn to_coeff(value: i32) -> Result<i16> {
    i16::try_from(value).map_err(|_| {
        JpegError::Malformed(format!(
            "coefficient {value} exceeds the 16-bit range of an 8-bit-precision frame"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extend_is_inverse_of_magnitude_coding() {
        // For every value, its (category, mantissa) must decode back exactly.
        for v in -4096i32..=4096 {
            let s = magnitude_category(v);
            let bits = mantissa_bits(v, s);
            assert_eq!(extend(bits, s), v, "roundtrip failed for {v} (S={s})");
        }
    }

    #[test]
    fn magnitude_category_matches_spec_table() {
        assert_eq!(magnitude_category(0), 0);
        assert_eq!(magnitude_category(1), 1);
        assert_eq!(magnitude_category(-1), 1);
        assert_eq!(magnitude_category(2), 2);
        assert_eq!(magnitude_category(-3), 2);
        assert_eq!(magnitude_category(1023), 10);
        assert_eq!(magnitude_category(-2047), 11);
    }
}
