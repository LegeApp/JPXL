//! The entropy-coding write side: one deliberately degenerate legal choice.
//!
//! 18181-1 Annex C offers a prefix-coded and an ANS-coded form. This crate
//! emits the prefix-coded one (C.2.4 → RFC 7932), because a prefix code has
//! **no per-stream state**: symbols are written in the order they are read, so
//! the encoder is a forward pass with no buffering, no reversal and no
//! terminal-state obligation. ANS would need all three. It is also what makes
//! multi-section output cheap — every section can restate the same code and no
//! state crosses a section boundary.
//!
//! Everything about the bundle is fixed except the alphabet width:
//!
//! | Field | Value | Why |
//! |---|---|---|
//! | `lz77.enabled` (C.1) | false | no window, no distance context |
//! | cluster map (C.2.2) | simple, `nbits = 0` | every context maps to cluster 0 |
//! | `use_prefix_code` (C.2.1) | true | see above |
//! | `HybridUintConfig` (C.2.3) | `split_exponent = 0`, no in-token bits | one token per magnitude class |
//! | alphabet size (C.2.1) | `1 << token_bits` | [`FlatCode`] |
//! | prefix code (C.2.4) | flat, `1 << token_bits` codes of `token_bits` bits | see [`FlatCode::write_bundle`] |
//!
//! # The value → token mapping
//!
//! With `split_exponent = 0` and no in-token bits, C.3.3's `ReadUint` reduces
//! to an Elias-gamma-like shape: token 0 is the literal 0, and token `t >= 1`
//! covers `[2^(t-1), 2^t)` with `t - 1` raw extra bits. An alphabet of
//! `1 << token_bits` tokens therefore spans `[0, 2^((1 << token_bits) - 1))`.
//!
//! # Why the alphabet has to be a choice
//!
//! Sixteen tokens reach `2^15 - 1`, which is enough for 8-bit residuals and
//! nothing more. A 16-bit sample has residuals up to `±65535`, so
//! `PackSigned` reaches `131071` and the stream needs token 18. C.2.3 offers
//! two ways out — a nonzero `split_exponent`, which moves value bits *into*
//! the token and so needs a wider alphabet anyway, or simply a wider alphabet.
//! The second is taken: it is one constant, it keeps `write_uint` a single
//! branch, and C.2.1's alphabet-size field encodes every power of two exactly.
//!
//! Five is the largest useful `token_bits`. C.3.3 requires the extra-bit count
//! `n` to stay below 32, and the top token of a `2^k`-symbol alphabet asks for
//! `n = 2^k - 2`, which is 30 at `k = 5` and 62 at `k = 6`.
//!
//! # Why a flat code costs nothing to build
//!
//! RFC 7932 section 3.5's code-length code becomes a **zero-bit** code when
//! exactly one of its symbols has a nonzero length: the decoder's `nonzero == 1`
//! branch turns it into a constant. Giving symbol `token_bits` the only nonzero
//! length therefore makes every alphabet code length read as `token_bits`
//! without consuming a bit, and `2^token_bits` codes of length `token_bits` is
//! a complete code whose canonical assignment maps symbol `s` to the
//! `token_bits`-bit string `s`. The whole code costs 36 bits (plus the 2-bit
//! HSKIP) and needs no histogram, no sorting and no length-limiting.

use jpxl_bitstream::BitWriter;

use crate::error::{EncodeError, Result};

/// Largest `token_bits` C.3.3's `n < 32` invariant allows (see the module
/// documentation).
pub const MAX_TOKEN_BITS: u32 = 5;

/// The code used for every MA-tree stream (18181-1 H.4.2).
///
/// Tree tokens are tiny — a predictor index, a packed zero offset, two
/// multiplier fields — so the narrow alphabet is always enough and the tree
/// stream never has to care what the data stream chose.
pub const TREE_CODE: FlatCode = FlatCode::new_const(4);

/// RFC 7932 section 3.5's code-length alphabet, in the order the lengths of
/// its own code are transmitted.
const CODE_LENGTH_ORDER: [usize; 18] =
    [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// Canonical codes of RFC 7932 section 3.5's fixed code-length code, derived
/// from its lengths `[2, 4, 3, 2, 2, 4]` and written most-significant-bit
/// first (the order a prefix code is consumed in).
///
/// Only the two entries this module emits are listed; the rest of the table is
/// not needed and transcribing it would be transcribing the RFC.
const CLC_CODE_ZERO: [u8; 2] = [0, 0];
const CLC_CODE_ONE: [u8; 4] = [1, 1, 1, 0];

/// A flat prefix code over `1 << token_bits` tokens, with the C.2.3 hybrid
/// configuration that makes token `t >= 1` mean "`t - 1` raw extra bits".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlatCode {
    token_bits: u32,
}

impl FlatCode {
    /// The code with `token_bits` bits per token, clamped to the legal range.
    ///
    /// `const` so [`TREE_CODE`] can exist; [`FlatCode::for_max_value`] is the
    /// constructor callers should reach for.
    #[must_use]
    pub const fn new_const(token_bits: u32) -> Self {
        let token_bits = if token_bits < 1 {
            1
        } else if token_bits > MAX_TOKEN_BITS {
            MAX_TOKEN_BITS
        } else {
            token_bits
        };
        Self { token_bits }
    }

    /// The narrowest code that can encode every value up to `max`.
    ///
    /// # Errors
    ///
    /// [`EncodeError::ValueOutOfRange`] if no legal alphabet reaches `max`.
    pub fn for_max_value(max: u32) -> Result<Self> {
        for token_bits in 1..=MAX_TOKEN_BITS {
            let code = Self { token_bits };
            if max <= code.max_value() {
                return Ok(code);
            }
        }
        Err(EncodeError::ValueOutOfRange {
            what: "entropy-coded value range",
            value: i64::from(max),
        })
    }

    /// Bits per token, i.e. the flat code length.
    #[must_use]
    pub const fn token_bits(&self) -> u32 {
        self.token_bits
    }

    /// Number of tokens, `1 << token_bits` (18181-1 C.2.1 alphabet size).
    #[must_use]
    pub const fn alphabet_size(&self) -> u32 {
        1u32 << self.token_bits
    }

    /// Largest value [`write_uint`](Self::write_uint) can encode: the top
    /// token spans `[2^(alphabet_size - 2), 2^(alphabet_size - 1))`.
    #[must_use]
    pub const fn max_value(&self) -> u32 {
        let top = self.alphabet_size() - 1;
        if top >= 32 {
            u32::MAX
        } else {
            (1u32 << top) - 1
        }
    }

    /// Writes the C.2.1 distribution bundle for a stream with `num_dist`
    /// pre-clustered contexts.
    ///
    /// Every context is mapped to a single cluster, so one prefix code and one
    /// hybrid-uint configuration follow.
    ///
    /// # Errors
    ///
    /// Only through the bit writer; every field here is derived from
    /// `token_bits`, which is validated on construction.
    pub fn write_bundle(&self, w: &mut BitWriter, num_dist: usize) -> Result<()> {
        // Table C.1: lz77.enabled. Disabled, so no min_symbol/min_length and
        // no extra distance context.
        w.write_bool(false);

        // C.2.2: a single context is its own cluster and nothing is written.
        if num_dist > 1 {
            w.write_bool(true); // is_simple
            w.write_bits(2, 0)?; // nbits = 0, so every context reads as cluster 0
        }

        // C.2.1: use_prefix_code, which fixes log_alphabet_size at 15.
        w.write_bool(true);

        // C.2.3 with log_alphabet_size 15: split_exponent is u(4). Zero is not
        // 15, so msb_in_token u(0) and lsb_in_token u(0) follow — both empty.
        w.write_bits(4, 0)?;

        // C.2.1 alphabet size: 1 + (1 << n) + u(n). For a power of two
        // `1 << k` that is n = k - 1 with a payload of `(1 << (k - 1)) - 1`.
        let n = self.token_bits - 1;
        w.write_bool(true);
        w.write_bits(4, n)?;
        w.write_bits(n, (1u32 << n) - 1)?;

        self.write_prefix_code(w)
    }

    /// Writes the flat prefix code of RFC 7932 section 3.5.
    fn write_prefix_code(&self, w: &mut BitWriter) -> Result<()> {
        // The 2-bit selector doubles as HSKIP; 1 would mean "simple code", so
        // 0 both selects the complex form and skips nothing.
        w.write_bits(2, 0)?;

        // The lengths of the code-length code, in CODE_LENGTH_ORDER. Exactly
        // one symbol gets a nonzero length, which makes the resulting code
        // zero-bit and every alphabet length equal to that symbol.
        let flat_length = self.token_bits as usize;
        for symbol in CODE_LENGTH_ORDER {
            if symbol == flat_length {
                write_code(w, &CLC_CODE_ONE)?;
            } else {
                write_code(w, &CLC_CODE_ZERO)?;
            }
        }
        // The alphabet code lengths follow, and cost zero bits each.
        Ok(())
    }

    /// Writes one token of the flat code: `token_bits` bits, most significant
    /// first.
    fn write_token(&self, w: &mut BitWriter, token: u32) -> Result<()> {
        if token >= self.alphabet_size() {
            return Err(EncodeError::ValueOutOfRange {
                what: "prefix-code token",
                value: i64::from(token),
            });
        }
        for shift in (0..self.token_bits).rev() {
            w.write_bits(1, (token >> shift) & 1)?;
        }
        Ok(())
    }

    /// Writes one unsigned integer, i.e. the inverse of
    /// `DecodeHybridVarLenUint` (18181-1 C.3.3) under this configuration.
    ///
    /// # Errors
    ///
    /// [`EncodeError::ValueOutOfRange`] if `value` exceeds
    /// [`max_value`](Self::max_value).
    pub fn write_uint(&self, w: &mut BitWriter, value: u32) -> Result<()> {
        if value > self.max_value() {
            return Err(EncodeError::ValueOutOfRange {
                what: "entropy-coded value",
                value: i64::from(value),
            });
        }
        if value == 0 {
            return self.write_token(w, 0);
        }
        // The token names the position of the most significant set bit; the
        // bits below it ride along as raw extra bits.
        let n = 31 - value.leading_zeros();
        self.write_token(w, n + 1)?;
        let extra = value - (1 << n);
        w.write_bits(n, extra)?;
        Ok(())
    }
}

/// Writes the bits of a prefix code word in read order, i.e. most significant
/// first (18181-1 C.2.4).
fn write_code(w: &mut BitWriter, bits: &[u8]) -> Result<()> {
    for &bit in bits {
        w.write_bits(1, u32::from(bit))?;
    }
    Ok(())
}

/// `PackSigned`, the inverse of `UnpackSigned` (18181-1 4.2).
///
/// `v >= 0` maps to `2v`; `v < 0` maps to `-2v - 1`. Computed in `i64` so
/// `i32::MIN` does not overflow before the doubling.
#[must_use]
pub fn pack_signed(value: i32) -> u32 {
    let v = i64::from(value);
    let packed = if v >= 0 { 2 * v } else { -2 * v - 1 };
    // The extremes are exactly the extremes of `u32`: `i32::MAX` maps to
    // `2^32 - 2` and `i32::MIN` to `2^32 - 1`, so the conversion never fails.
    u32::try_from(packed).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jpxl_bitstream::BitReader;
    use jpxl_core::limits::{AllocGuard, Limits};

    #[test]
    fn pack_signed_inverts_unpack_signed() {
        for v in [
            0i32,
            1,
            -1,
            2,
            -2,
            255,
            -255,
            16383,
            -16384,
            65535,
            -65535,
            i32::MAX,
            i32::MIN,
        ] {
            let packed = pack_signed(v);
            // The decoder's UnpackSigned, restated so the two are checked
            // against each other rather than against one implementation.
            let unpacked = if packed.is_multiple_of(2) {
                i64::from(packed) / 2
            } else {
                -(i64::from(packed) + 1) / 2
            };
            assert_eq!(unpacked, i64::from(v), "round trip for {v}");
        }
    }

    #[test]
    fn alphabet_widths_reach_the_documented_ranges() {
        assert_eq!(FlatCode::new_const(4).alphabet_size(), 16);
        assert_eq!(FlatCode::new_const(4).max_value(), (1 << 15) - 1);
        assert_eq!(FlatCode::new_const(5).alphabet_size(), 32);
        assert_eq!(FlatCode::new_const(5).max_value(), (1u32 << 31) - 1);

        // The 16-bit blocker of slice 7.5: PackSigned of a -65535 residual.
        let needed = pack_signed(-65535);
        assert!(needed > FlatCode::new_const(4).max_value());
        assert_eq!(
            FlatCode::for_max_value(needed).expect("legal").token_bits(),
            5
        );
        assert_eq!(FlatCode::for_max_value(511).expect("legal").token_bits(), 4);
    }

    /// Encodes `values` through one bundle and decodes them back with
    /// `jpxl-entropy`, which is the only thing that proves the bundle is legal.
    fn round_trip(code: FlatCode, num_dist: usize, contexts: &[usize], values: &[u32]) {
        let mut w = BitWriter::new();
        code.write_bundle(&mut w, num_dist).expect("bundle");
        for &v in values {
            code.write_uint(&mut w, v).expect("value fits");
        }
        w.zero_pad_to_byte();
        let bytes = w.into_bytes();

        let limits = Limits::relaxed();
        let mut guard = AllocGuard::new(&limits);
        let mut r = BitReader::new(&bytes);
        let mut dec = jpxl_entropy::SymbolDecoder::open(&mut r, num_dist, &mut guard)
            .expect("the bundle must open");
        assert!(dec.uses_prefix_code());
        assert_eq!(dec.clusters().num_clusters(), 1);
        for (i, &v) in values.iter().enumerate() {
            let ctx = contexts.get(i).copied().unwrap_or(0);
            assert_eq!(dec.read_uint(&mut r, ctx).expect("symbol"), v, "value {v}");
        }
        dec.finish().expect("prefix streams have no terminal state");
    }

    #[test]
    fn single_context_bundle_round_trips_at_every_width() {
        for token_bits in 1..=MAX_TOKEN_BITS {
            let code = FlatCode::new_const(token_bits);
            let values: Vec<u32> = (0..64)
                .chain([255, 256, 511, 512])
                .filter(|&v| v <= code.max_value())
                .chain([code.max_value()])
                .collect();
            round_trip(code, 1, &vec![0; values.len()], &values);
        }
    }

    #[test]
    fn every_context_of_a_six_context_bundle_maps_to_one_cluster() {
        // The MA-tree stream of H.4.2 uses contexts 0..6.
        let contexts: Vec<usize> = (0..6).cycle().take(30).collect();
        let values: Vec<u32> = (0..30).map(|i| i * 37).collect();
        round_trip(TREE_CODE, 6, &contexts, &values);
    }

    #[test]
    fn every_value_a_sixteen_symbol_alphabet_can_express_round_trips() {
        // Exhaustive over the full range the configuration can express: this
        // is what proves the token/extra-bit split, not a sample of it.
        let code = FlatCode::new_const(4);
        let mut w = BitWriter::new();
        code.write_bundle(&mut w, 1).expect("bundle");
        for v in 0..=code.max_value() {
            code.write_uint(&mut w, v).expect("value fits");
        }
        w.zero_pad_to_byte();
        let bytes = w.into_bytes();

        let limits = Limits::relaxed();
        let mut guard = AllocGuard::new(&limits);
        let mut r = BitReader::new(&bytes);
        let mut dec =
            jpxl_entropy::SymbolDecoder::open(&mut r, 1, &mut guard).expect("bundle opens");
        for v in 0..=code.max_value() {
            assert_eq!(dec.read_uint(&mut r, 0).expect("symbol"), v);
        }
    }

    #[test]
    fn the_wide_alphabet_covers_the_whole_sixteen_bit_residual_range() {
        // Every PackSigned value a 16-bit lossless residual can produce, at
        // the boundaries plus a dense sweep of the low end.
        let code = FlatCode::new_const(5);
        let values: Vec<u32> = (0..4096u32)
            .chain((0..=65535u32).step_by(97).map(pack_signed_of))
            .chain([pack_signed(65535), pack_signed(-65535), pack_signed(131071)])
            .collect();
        round_trip(code, 1, &vec![0; values.len()], &values);
    }

    fn pack_signed_of(v: u32) -> u32 {
        pack_signed(i32::try_from(v).unwrap_or(0) - 32768)
    }

    #[test]
    fn values_above_the_alphabet_are_rejected() {
        let code = FlatCode::new_const(4);
        let mut w = BitWriter::new();
        code.write_bundle(&mut w, 1).expect("bundle");
        assert!(matches!(
            code.write_uint(&mut w, code.max_value() + 1),
            Err(EncodeError::ValueOutOfRange { .. })
        ));
        assert!(matches!(
            FlatCode::for_max_value(u32::MAX),
            Err(EncodeError::ValueOutOfRange { .. })
        ));
    }

    #[test]
    fn the_bundle_is_the_size_the_module_documents() {
        // 1 lz77 + 1 prefix flag + 4 config + (1 + 4 + (k - 1)) alphabet size
        // + 2 hskip + 17 * 2 + 4 code-length words = 50 + k bits.
        for token_bits in 1..=MAX_TOKEN_BITS {
            let mut w = BitWriter::new();
            FlatCode::new_const(token_bits)
                .write_bundle(&mut w, 1)
                .expect("bundle");
            assert_eq!(w.bit_len(), u64::from(50 + token_bits));
        }

        // Six contexts add the three-bit simple cluster map.
        let mut w = BitWriter::new();
        TREE_CODE.write_bundle(&mut w, 6).expect("bundle");
        assert_eq!(w.bit_len(), 57);
    }
}
