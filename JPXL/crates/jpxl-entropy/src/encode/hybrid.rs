//! Hybrid unsigned integer emission — the write side of 18181-1 C.2.3 and of
//! `ReadUint` in C.3.3.
//!
//! [`HybridUintConfig::tokenize`] is the exact inverse of
//! [`HybridUintConfig::read_uint`](crate::HybridUintConfig::read_uint): it
//! splits a value into the token an entropy coder carries and the raw extra
//! bits that follow it. [`HybridUintConfig::write`] is the inverse of
//! `ReadUintConfig`.
//!
//! # Deriving the split
//!
//! C.3.3 reconstructs a value as
//!
//! ```text
//! n      = split_exponent - msb_in_token - lsb_in_token
//!          + ((token - split) >> (msb_in_token + lsb_in_token))
//! low    = token & ((1 << lsb_in_token) - 1)
//! mid    = ((token >> lsb_in_token) & ((1 << msb_in_token) - 1)) | (1 << msb_in_token)
//! result = (((mid << n) | extra) << lsb_in_token) | low
//! ```
//!
//! so, reading it backwards for a `value >= split`: `low` is the bottom
//! `lsb_in_token` bits of the value; `high = value >> lsb_in_token` is a
//! `msb_in_token + 1 + n` bit number whose top `msb_in_token + 1` bits are
//! `mid` (its leading bit is the implicit one C.3.3 sets) and whose bottom `n`
//! bits are the extra bits. That fixes `n` from the bit width of `high`, and
//! every field of the token follows.
//!
//! Every path here is **bit-exact** (see `docs/PLAN.md`).

use jpxl_bitstream::BitWriter;

use crate::error::{Result, encode_error};
use crate::hybrid::{HybridUintConfig, bit_width};

/// Largest `split_exponent` any legal configuration can carry: C.2.1 caps
/// `log_alphabet_size` at 15 and C.2.3 caps `split_exponent` by it.
pub const MAX_SPLIT_EXPONENT: u32 = 15;

/// A value split into the entropy-coded token and the raw bits that follow it.
///
/// `extra` holds `extra_bits` significant bits and is written with `u(n)`
/// straight after the token, exactly where C.3.3 reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenSplit {
    /// The symbol handed to the prefix code or the ANS coder.
    pub token: u32,
    /// Number of raw bits that follow the token.
    pub extra_bits: u32,
    /// Payload of those raw bits.
    pub extra: u32,
}

impl HybridUintConfig {
    /// Builds a configuration, checking the invariants C.2.3 states.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Encode`](crate::EntropyError::Encode) if
    /// `split_exponent` exceeds [`MAX_SPLIT_EXPONENT`] or if
    /// `msb_in_token + lsb_in_token` exceeds `split_exponent`.
    pub fn new(split_exponent: u32, msb_in_token: u32, lsb_in_token: u32) -> Result<Self> {
        let config = Self {
            split_exponent,
            msb_in_token,
            lsb_in_token,
        };
        config.validate()?;
        Ok(config)
    }

    /// Checks the C.2.3 invariants of a configuration built by hand.
    ///
    /// The fields are public so that a decoded configuration can be inspected
    /// and copied; that also means an encoder caller can build a nonsensical
    /// one, so every write path validates first.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Encode`](crate::EntropyError::Encode) if the invariants
    /// do not hold.
    pub fn validate(&self) -> Result<()> {
        if self.split_exponent > MAX_SPLIT_EXPONENT {
            return Err(encode_error!(
                "C.2.3: split_exponent {} exceeds {MAX_SPLIT_EXPONENT}",
                self.split_exponent
            ));
        }
        let in_token = self
            .msb_in_token
            .checked_add(self.lsb_in_token)
            .ok_or_else(|| encode_error!("C.2.3: in-token bit count overflows"))?;
        if in_token > self.split_exponent {
            return Err(encode_error!(
                "C.2.3: msb_in_token {} + lsb_in_token {} exceeds split_exponent {}",
                self.msb_in_token,
                self.lsb_in_token,
                self.split_exponent
            ));
        }
        Ok(())
    }

    /// Splits `value` into a token and its raw extra bits (inverse of
    /// `ReadUint`, 18181-1 C.3.3).
    ///
    /// # Errors
    ///
    /// [`EntropyError::Encode`](crate::EntropyError::Encode) if the
    /// configuration is invalid, or if the value needs more than the 31 extra
    /// bits C.3.3's `n < 32` invariant allows.
    pub fn tokenize(&self, value: u32) -> Result<TokenSplit> {
        self.validate()?;
        if value < self.split() {
            return Ok(TokenSplit {
                token: value,
                extra_bits: 0,
                extra: 0,
            });
        }

        let msb = self.msb_in_token;
        let lsb = self.lsb_in_token;
        let in_token = msb + lsb;
        let base = self.split_exponent - in_token;

        let low = value & mask(lsb);
        let high = value >> lsb;
        // `value >= split` gives `high >= 1 << (split_exponent - lsb)`, whose
        // bit width is at least `base + msb + 1`, so this cannot wrap.
        let n = bit_width(high).checked_sub(msb + 1).ok_or_else(|| {
            encode_error!("C.3.3: value {value} is below the token's implicit bit")
        })?;
        if n >= 32 {
            return Err(encode_error!(
                "C.3.3: value {value} needs {n} extra bits, violating n < 32"
            ));
        }
        let extra = high & mask(n);
        let mid = (high >> n) & mask(msb);

        let token = u64::from(self.split())
            + (u64::from(n - base) << in_token)
            + (u64::from(mid) << lsb)
            + u64::from(low);
        let token = u32::try_from(token)
            .map_err(|_| encode_error!("C.3.3: token for value {value} does not fit in 32 bits"))?;

        Ok(TokenSplit {
            token,
            extra_bits: n,
            extra,
        })
    }

    /// The largest token [`tokenize`](Self::tokenize) can produce for any value
    /// in `[0, max_value]`.
    ///
    /// Used to size an alphabet before any value is written.
    ///
    /// # Errors
    ///
    /// As [`tokenize`](Self::tokenize).
    pub fn max_token(&self, max_value: u32) -> Result<u32> {
        Ok(self.tokenize(max_value)?.token)
    }

    /// Writes the configuration (inverse of `ReadUintConfig`, 18181-1 C.2.3).
    ///
    /// `log_alphabet_size` must be the value the decoder will be using when it
    /// reads this field — 15 for a prefix-coded bundle, `5 + u(2)` for an ANS
    /// bundle, 8 for the LZ77 length configuration.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Encode`](crate::EntropyError::Encode) if the
    /// configuration is invalid or cannot be expressed at this
    /// `log_alphabet_size`, or a bitstream error.
    pub fn write(&self, w: &mut BitWriter, log_alphabet_size: u32) -> Result<()> {
        self.validate()?;
        if log_alphabet_size > MAX_SPLIT_EXPONENT {
            return Err(encode_error!(
                "C.2.3: log_alphabet_size {log_alphabet_size} exceeds {MAX_SPLIT_EXPONENT}"
            ));
        }
        if self.split_exponent > log_alphabet_size {
            return Err(encode_error!(
                "C.2.3: split_exponent {} exceeds log_alphabet_size {log_alphabet_size}",
                self.split_exponent
            ));
        }

        w.write_bits(bit_width(log_alphabet_size), self.split_exponent)?;
        if self.split_exponent == log_alphabet_size {
            // C.2.3 reads no further fields in this case, so a configuration
            // with in-token bits would be lost.
            if self.msb_in_token != 0 || self.lsb_in_token != 0 {
                return Err(encode_error!(
                    "C.2.3: split_exponent == log_alphabet_size forces msb_in_token and \
                     lsb_in_token to 0"
                ));
            }
            return Ok(());
        }
        w.write_bits(bit_width(self.split_exponent), self.msb_in_token)?;
        w.write_bits(
            bit_width(self.split_exponent - self.msb_in_token),
            self.lsb_in_token,
        )?;
        Ok(())
    }
}

/// `(1 << n) - 1` for `n < 32`, saturating at `u32::MAX` so a wide `n` cannot
/// shift out of range.
const fn mask(n: u32) -> u32 {
    if n >= 32 { u32::MAX } else { (1u32 << n) - 1 }
}

#[cfg(test)]
// Test code: an out-of-range index is a failed assertion rather than an attack
// surface, and the casts are on values these tests chose themselves.
#[allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]
mod tests {
    use super::*;
    use jpxl_bitstream::BitReader;

    /// The defining property of this module: `tokenize` followed by the
    /// decoder's `read_uint` is the identity, for every configuration shape and
    /// a dense sweep of values.
    #[test]
    fn tokenize_inverts_read_uint() {
        for split_exponent in 0..=8u32 {
            for msb in 0..=split_exponent {
                for lsb in 0..=(split_exponent - msb) {
                    let cfg = HybridUintConfig::new(split_exponent, msb, lsb).expect("legal");
                    for value in (0..600u32).chain([1 << 16, u32::MAX / 3, u32::MAX / 2]) {
                        let split = cfg.tokenize(value).expect("value fits");
                        let mut w = BitWriter::new();
                        w.write_bits(split.extra_bits, split.extra).expect("extra");
                        let bytes = w.into_bytes();
                        let mut r = BitReader::new(&bytes);
                        assert_eq!(
                            cfg.read_uint(&mut r, split.token).expect("decodes"),
                            value,
                            "config ({split_exponent}, {msb}, {lsb}) value {value}"
                        );
                        assert_eq!(r.total_bits_read(), u64::from(split.extra_bits));
                    }
                }
            }
        }
    }

    #[test]
    fn tokens_below_split_carry_no_extra_bits() {
        let cfg = HybridUintConfig::new(4, 2, 1).expect("legal");
        for value in 0..16 {
            let split = cfg.tokenize(value).expect("literal");
            assert_eq!(
                split,
                TokenSplit {
                    token: value,
                    extra_bits: 0,
                    extra: 0
                }
            );
        }
    }

    /// The configuration field round-trips through the decoder's reader at
    /// every `log_alphabet_size` the standard uses.
    #[test]
    fn configuration_round_trips() {
        for log_alphabet_size in [5u32, 6, 7, 8, 15] {
            for split_exponent in 0..=log_alphabet_size {
                let shapes: Vec<(u32, u32)> = if split_exponent == log_alphabet_size {
                    vec![(0, 0)]
                } else {
                    (0..=split_exponent)
                        .flat_map(|m| (0..=(split_exponent - m)).map(move |l| (m, l)))
                        .collect()
                };
                for (msb, lsb) in shapes {
                    let cfg = HybridUintConfig::new(split_exponent, msb, lsb).expect("legal");
                    let mut w = BitWriter::new();
                    cfg.write(&mut w, log_alphabet_size).expect("writes");
                    let bytes = w.into_bytes();
                    let mut r = BitReader::new(&bytes);
                    let read = HybridUintConfig::read(&mut r, log_alphabet_size).expect("reads");
                    assert_eq!(read, cfg, "log_alphabet_size {log_alphabet_size}");
                }
            }
        }
    }

    #[test]
    fn invalid_configurations_are_rejected() {
        assert!(HybridUintConfig::new(16, 0, 0).is_err());
        assert!(HybridUintConfig::new(3, 2, 2).is_err());
        // A configuration that would lose its in-token fields at this width.
        let cfg = HybridUintConfig {
            split_exponent: 5,
            msb_in_token: 1,
            lsb_in_token: 1,
        };
        let mut w = BitWriter::new();
        assert!(cfg.write(&mut w, 5).is_err());
        // And one wider than the alphabet allows.
        let mut w = BitWriter::new();
        assert!(cfg.write(&mut w, 4).is_err());
    }
}
