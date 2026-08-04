//! Hybrid unsigned integer configuration and reconstruction.
//!
//! ISO/IEC 18181-1 C.2.3 (reading the configuration) and the `ReadUint`
//! procedure of C.3.3 (turning an entropy-coded token plus raw extra bits back
//! into an integer).
//!
//! The scheme splits the integer range in two. Tokens below `split` are
//! literal values and cost nothing beyond the token itself. Tokens at or above
//! `split` encode a floating-point-like shape: some number of most-significant
//! bits and least-significant bits ride along inside the token, and the
//! remaining middle bits follow as raw `u(n)` in the same bit reader.
//!
//! Every path here is **bit-exact** (see `docs/PLAN.md`): the arithmetic is
//! integer and fully specified, so any divergence from the reference is a bug,
//! not a tolerance.

use jpxl_bitstream::BitReader;

use crate::error::{Result, malformed};

/// Number of bits needed to represent `n`, i.e. `ceil(log2(n + 1))`.
///
/// C.2.3 writes the field widths as `ceil(log2(x + 1))`; that is exactly the
/// bit width of `x`, which avoids a floating-point logarithm in the decoder.
pub(crate) const fn bit_width(n: u32) -> u32 {
    u32::BITS - n.leading_zeros()
}

/// Configuration of the hybrid unsigned integer decoder (18181-1 C.2.3).
///
/// Construct with [`HybridUintConfig::read`]; the invariants asserted by the
/// clause (`msb_in_token + lsb_in_token <= split_exponent`) are checked there,
/// so every value of this type is well-formed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HybridUintConfig {
    /// Log2 of the literal/extra-bits split point.
    pub split_exponent: u32,
    /// Number of most-significant value bits carried inside the token.
    pub msb_in_token: u32,
    /// Number of least-significant value bits carried inside the token.
    pub lsb_in_token: u32,
}

impl HybridUintConfig {
    /// The literal threshold `split = 1 << split_exponent` (18181-1 C.2.3).
    ///
    /// Tokens strictly below this are returned unchanged by
    /// [`read_uint`](Self::read_uint).
    #[must_use]
    pub const fn split(&self) -> u32 {
        1u32 << self.split_exponent
    }

    /// Reads a configuration, i.e. `ReadUintConfig` of 18181-1 C.2.3.
    ///
    /// `log_alphabet_size` is 15 for prefix-coded streams and `5 + u(2)` for
    /// ANS streams (C.2.1), except for the LZ77 length configuration which is
    /// read with a fixed value of 8.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Malformed`](crate::EntropyError::Malformed) if the
    /// stream violates the clause invariants, or a bitstream error if the
    /// input is exhausted.
    pub fn read(reader: &mut BitReader<'_>, log_alphabet_size: u32) -> Result<Self> {
        if log_alphabet_size > 15 {
            return Err(malformed!(
                "C.2.3: log_alphabet_size {log_alphabet_size} exceeds 15"
            ));
        }
        let split_exponent = reader.read_bits(bit_width(log_alphabet_size))?;
        if split_exponent == log_alphabet_size {
            return Ok(Self {
                split_exponent,
                msb_in_token: 0,
                lsb_in_token: 0,
            });
        }

        let msb_in_token = reader.read_bits(bit_width(split_exponent))?;
        if msb_in_token > split_exponent {
            return Err(malformed!(
                "C.2.3: msb_in_token {msb_in_token} exceeds split_exponent {split_exponent}"
            ));
        }
        let remaining = split_exponent - msb_in_token;
        let lsb_in_token = reader.read_bits(bit_width(remaining))?;
        if lsb_in_token > remaining {
            return Err(malformed!(
                "C.2.3: lsb_in_token {lsb_in_token} + msb_in_token {msb_in_token} exceeds \
                 split_exponent {split_exponent}"
            ));
        }
        Ok(Self {
            split_exponent,
            msb_in_token,
            lsb_in_token,
        })
    }

    /// Reconstructs an integer from `token`, reading extra bits as needed.
    ///
    /// This is `ReadUint(config, token)` of 18181-1 C.3.3. Extra bits come
    /// from `reader`, which must be the same reader the token was decoded
    /// from.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Malformed`](crate::EntropyError::Malformed) if the
    /// clause invariants `n < 32` or `result < (1 << 32)` are violated, which
    /// only a malformed stream can cause.
    pub fn read_uint(&self, reader: &mut BitReader<'_>, token: u32) -> Result<u32> {
        if token < self.split() {
            return Ok(token);
        }

        let in_token = self.msb_in_token + self.lsb_in_token;
        // `split_exponent >= in_token` is an invariant of the type, so this
        // subtraction cannot wrap.
        let base = self.split_exponent - in_token;
        let n = base
            .checked_add((token - self.split()) >> in_token)
            .ok_or_else(|| malformed!("C.3.3: extra-bit count overflows"))?;
        if n >= 32 {
            return Err(malformed!(
                "C.3.3: extra-bit count {n} violates the n < 32 invariant"
            ));
        }

        let low = token & ((1u32 << self.lsb_in_token) - 1);
        let mut mid = token >> self.lsb_in_token;
        mid &= (1u32 << self.msb_in_token) - 1;
        mid |= 1u32 << self.msb_in_token;

        let extra = u64::from(reader.read_bits(n)?);
        // Widened to u64 so the clause's `result < (1 << 32)` check is a real
        // test rather than a silent wrap.
        let result = (((u64::from(mid) << n) | extra) << self.lsb_in_token) | u64::from(low);
        u32::try_from(result)
            .map_err(|_| malformed!("C.3.3: reconstructed value {result} does not fit in 32 bits"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_width_matches_ceil_log2_of_n_plus_one() {
        // C.2.3 field widths: ceil(log2(n + 1)).
        assert_eq!(bit_width(0), 0);
        assert_eq!(bit_width(1), 1);
        assert_eq!(bit_width(2), 2);
        assert_eq!(bit_width(3), 2);
        assert_eq!(bit_width(4), 3);
        assert_eq!(bit_width(5), 3);
        assert_eq!(bit_width(7), 3);
        assert_eq!(bit_width(8), 4);
        assert_eq!(bit_width(15), 4);
    }

    #[test]
    fn tokens_below_split_are_literal() {
        let cfg = HybridUintConfig {
            split_exponent: 4,
            msb_in_token: 0,
            lsb_in_token: 0,
        };
        let mut r = BitReader::new(&[]);
        for token in 0..16 {
            assert_eq!(cfg.read_uint(&mut r, token).expect("literal"), token);
        }
        assert_eq!(r.total_bits_read(), 0, "literals must read no extra bits");
    }
}
