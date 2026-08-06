//! LZ77 emission — the write side of 18181-1 Table C.1 and C.3.3 back-references.
//!
//! Decode lives in [`crate::lz77`]; this module only *writes* the bundle fields
//! and describes the policy parameters a caller supplies. Control flow is not
//! shared with the decoder (paired-bug rule).
//!
//! When enabled, C.2.1 appends one distance context after the caller's value
//! contexts. The caller's [`ContextMap`](super::ContextMap) must already
//! include that final entry; see [`EncoderPlan::identity_with_lz77`](super::EncoderPlan::identity_with_lz77).

use jpxl_bitstream::{BitWriter, U32Dist, U32Spec};

use crate::error::{Result, encode_error};
use crate::hybrid::HybridUintConfig;

/// `log_alphabet_size` fixed for the LZ77 length configuration (18181-1 C.2.1).
pub const LZ_LENGTH_LOG_ALPHABET_SIZE: u32 = 8;

/// `U32(224, 512, 4096, 8 + u(15))` — Table C.1, `min_symbol`.
const MIN_SYMBOL: U32Spec = U32Spec::new([
    U32Dist::Val(224),
    U32Dist::Val(512),
    U32Dist::Val(4096),
    U32Dist::BitsOffset {
        bits: 15,
        offset: 8,
    },
]);

/// `U32(3, 4, 5 + u(2), 9 + u(8))` — Table C.1, `min_length`.
const MIN_LENGTH: U32Spec = U32Spec::new([
    U32Dist::Val(3),
    U32Dist::Val(4),
    U32Dist::BitsOffset { bits: 2, offset: 5 },
    U32Dist::BitsOffset { bits: 8, offset: 9 },
]);

/// Policy parameters for an LZ77-enabled entropy stream (Table C.1 + C.2.1).
///
/// The length configuration is written with
/// [`LZ_LENGTH_LOG_ALPHABET_SIZE`]; the decoder always re-reads it at that width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lz77EncodeParams {
    /// First token that denotes a back-reference rather than a hybrid-uint value.
    pub min_symbol: u32,
    /// Constant added to every decoded copy length.
    pub min_length: u32,
    /// Hybrid-uint configuration for the length alphabet (`lz_len_conf`).
    pub length_config: HybridUintConfig,
}

impl Lz77EncodeParams {
    /// Builds parameters, checking Table C.1 and C.2.3 invariants.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Encode`](crate::EntropyError::Encode) if a field cannot
    /// be represented on the wire or the length configuration is illegal at
    /// log-alphabet size 8.
    pub fn new(min_symbol: u32, min_length: u32, length_config: HybridUintConfig) -> Result<Self> {
        let params = Self {
            min_symbol,
            min_length,
            length_config,
        };
        params.validate()?;
        Ok(params)
    }

    /// A common dense-friendly default: `min_symbol = 224`, `min_length = 3`,
    /// length config with `split_exponent = 8` (no in-token bits).
    ///
    /// # Errors
    ///
    /// Only if the length configuration invariants fail (they should not).
    pub fn dense_default() -> Result<Self> {
        let length_config = HybridUintConfig::new(LZ_LENGTH_LOG_ALPHABET_SIZE, 0, 0)?;
        Self::new(224, 3, length_config)
    }

    /// Checks that every field is representable as Table C.1 / C.2.3 require.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Encode`](crate::EntropyError::Encode) on an illegal field.
    pub fn validate(&self) -> Result<()> {
        if !min_symbol_representable(self.min_symbol) {
            return Err(encode_error!(
                "Table C.1: min_symbol {} is not representable by U32(224,512,4096,8+u(15))",
                self.min_symbol
            ));
        }
        if !min_length_representable(self.min_length) {
            return Err(encode_error!(
                "Table C.1: min_length {} is not representable by U32(3,4,5+u(2),9+u(8))",
                self.min_length
            ));
        }
        self.length_config.validate()?;
        if self.length_config.split_exponent > LZ_LENGTH_LOG_ALPHABET_SIZE {
            return Err(encode_error!(
                "C.2.1: lz_len_conf.split_exponent {} exceeds log_alphabet_size {LZ_LENGTH_LOG_ALPHABET_SIZE}",
                self.length_config.split_exponent
            ));
        }
        if self.length_config.split_exponent == LZ_LENGTH_LOG_ALPHABET_SIZE
            && (self.length_config.msb_in_token != 0 || self.length_config.lsb_in_token != 0)
        {
            return Err(encode_error!(
                "C.2.3: lz_len_conf with split_exponent == 8 forces msb_in_token and lsb_in_token to 0"
            ));
        }
        Ok(())
    }

    /// Writes the enabled half of Table C.1 and the length configuration.
    ///
    /// Call only when LZ77 is enabled; the leading `Bool()` is part of this
    /// write.
    ///
    /// # Errors
    ///
    /// As [`validate`](Self::validate), or a bitstream error.
    pub fn write_enabled(&self, w: &mut BitWriter) -> Result<()> {
        self.validate()?;
        w.write_bool(true);
        w.write_u32(&MIN_SYMBOL, self.min_symbol)?;
        w.write_u32(&MIN_LENGTH, self.min_length)?;
        self.length_config.write(w, LZ_LENGTH_LOG_ALPHABET_SIZE)?;
        Ok(())
    }

    /// Tokenizes a copy length into the wire token and length extra bits.
    ///
    /// Decoder reconstructs `length = ReadUint(lz_len_conf, token - min_symbol)
    /// + min_length`, so the hybrid-uint payload is `length - min_length`.
    ///
    /// # Errors
    ///
    /// If `length < min_length`, or the length configuration cannot express
    /// the base, or `min_symbol + length_token` overflows `u32`.
    pub fn tokenize_length(&self, length: u32) -> Result<LengthToken> {
        self.validate()?;
        let base = length.checked_sub(self.min_length).ok_or_else(|| {
            encode_error!(
                "C.3.3: copy length {length} is below min_length {}",
                self.min_length
            )
        })?;
        let split = self.length_config.tokenize(base)?;
        let token = self
            .min_symbol
            .checked_add(split.token)
            .ok_or_else(|| encode_error!("C.3.3: LZ77 length token overflows u32"))?;
        Ok(LengthToken {
            token,
            extra_bits: split.extra_bits,
            extra: split.extra,
        })
    }
}

/// A length back-reference token ready for the value-context alphabet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LengthToken {
    /// Entropy-coded token (`>= min_symbol`).
    pub token: u32,
    /// Raw extra bits from `lz_len_conf` (not the value-cluster hybrid-uint).
    pub extra_bits: u32,
    /// Payload of those extra bits.
    pub extra: u32,
}

fn min_symbol_representable(value: u32) -> bool {
    value == 224 || value == 512 || value == 4096 || (8..=8 + ((1u32 << 15) - 1)).contains(&value)
}

fn min_length_representable(value: u32) -> bool {
    value == 3 || value == 4 || (5..=5 + 0b11).contains(&value) || (9..=9 + 0xff).contains(&value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lz77::Lz77Params;
    use jpxl_bitstream::BitReader;

    #[test]
    fn dense_default_round_trips_through_the_decoder_reader() {
        let params = Lz77EncodeParams::dense_default().expect("default");
        let mut w = BitWriter::new();
        params.write_enabled(&mut w).expect("write");
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        let decoded = Lz77Params::read(&mut r).expect("read");
        assert!(decoded.enabled);
        assert_eq!(decoded.min_symbol, 224);
        assert_eq!(decoded.min_length, 3);
        // Table C.1 only: Bool + two U32 selectors (Val arms) = 1 + 2 + 2 = 5.
        // lz_len_conf is a separate C.2.1 field after the params.
        assert_eq!(r.total_bits_read(), 5);
        let len_conf =
            HybridUintConfig::read(&mut r, LZ_LENGTH_LOG_ALPHABET_SIZE).expect("lz_len_conf");
        assert_eq!(len_conf, params.length_config);
        assert_eq!(r.total_bits_read(), 9);
    }

    #[test]
    fn handmade_min_symbol_eight_matches_lz77_streams_fixture() {
        // Matches tests/lz77_streams.rs: min_symbol=8, min_length=3, split_exponent=8.
        let length_config = HybridUintConfig::new(8, 0, 0).expect("legal");
        let params = Lz77EncodeParams::new(8, 3, length_config).expect("params");
        let mut w = BitWriter::new();
        params.write_enabled(&mut w).expect("write");
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        let decoded = Lz77Params::read(&mut r).expect("read");
        assert_eq!(
            decoded,
            Lz77Params {
                enabled: true,
                min_symbol: 8,
                min_length: 3,
            }
        );
    }

    #[test]
    fn length_three_with_default_config_is_token_min_symbol() {
        let length_config = HybridUintConfig::new(8, 0, 0).expect("legal");
        let params = Lz77EncodeParams::new(8, 3, length_config).expect("params");
        let tok = params.tokenize_length(3).expect("len 3");
        assert_eq!(tok.token, 8);
        assert_eq!(tok.extra_bits, 0);
        assert_eq!(tok.extra, 0);
    }

    #[test]
    fn length_below_min_is_rejected() {
        let params = Lz77EncodeParams::dense_default().expect("default");
        assert!(params.tokenize_length(2).is_err());
    }

    #[test]
    fn illegal_min_symbol_is_rejected() {
        let length_config = HybridUintConfig::new(8, 0, 0).expect("legal");
        // Below the 8 + u(15) floor and not a fixed selector value.
        assert!(Lz77EncodeParams::new(7, 3, length_config).is_err());
        // Above 8 + (2^15 - 1) and not a fixed selector value.
        assert!(Lz77EncodeParams::new(8 + (1 << 15), 3, length_config).is_err());
    }
}
