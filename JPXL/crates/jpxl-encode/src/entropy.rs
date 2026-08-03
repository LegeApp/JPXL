//! The entropy-coding write side: one deliberately degenerate legal choice.
//!
//! 18181-1 Annex C offers a prefix-coded and an ANS-coded form. This slice
//! emits the prefix-coded one (C.2.4 → RFC 7932), because a prefix code has
//! **no per-stream state**: symbols are written in the order they are read, so
//! the encoder is a forward pass with no buffering, no reversal and no
//! terminal-state obligation. ANS would need all three.
//!
//! Everything about the bundle is fixed:
//!
//! | Field | Value | Why |
//! |---|---|---|
//! | `lz77.enabled` (C.1) | false | no window, no distance context |
//! | cluster map (C.2.2) | simple, `nbits = 0` | every context maps to cluster 0 |
//! | `use_prefix_code` (C.2.1) | true | see above |
//! | `HybridUintConfig` (C.2.3) | `split_exponent = 0`, no in-token bits | one token per magnitude class |
//! | alphabet size (C.2.1) | 16 | covers every value below `1 << 15` |
//! | prefix code (C.2.4) | flat, 16 codes of 4 bits | see [`write_prefix_code`] |
//!
//! # The value → token mapping
//!
//! With `split_exponent = 0` and no in-token bits, C.3.3's `ReadUint` reduces
//! to an Elias-gamma-like shape: token 0 is the literal 0, and token `t >= 1`
//! covers `[2^(t-1), 2^t)` with `t - 1` raw extra bits. Sixteen tokens
//! therefore span `[0, 2^15)`, which is [`MAX_VALUE`].
//!
//! # Why a flat code costs nothing to build
//!
//! RFC 7932 section 3.5's code-length code becomes a **zero-bit** code when
//! exactly one of its symbols has a nonzero length: the decoder's `nonzero == 1`
//! branch turns it into a constant. Giving symbol `4` the only nonzero length
//! therefore makes every one of the 16 alphabet code lengths read as 4 without
//! consuming a bit, and 16 codes of length 4 is a complete code whose canonical
//! assignment maps symbol `s` to the 4-bit string `s`. The whole code costs 40
//! bits (54 for the whole bundle) and needs no histogram, no sorting and no length-limiting.

use jpxl_bitstream::BitWriter;

use crate::error::{EncodeError, Result};

/// Number of symbols in the fixed token alphabet (18181-1 C.2.1).
pub const ALPHABET_SIZE: u32 = 16;

/// Bits per token in the flat prefix code.
pub const TOKEN_BITS: u32 = 4;

/// Largest value [`write_uint`] can encode: token 15 spans `[2^14, 2^15)`.
pub const MAX_VALUE: u32 = (1 << 15) - 1;

/// RFC 7932 section 3.5's code-length alphabet, in the order the lengths of
/// its own code are transmitted.
const CODE_LENGTH_ORDER: [usize; 18] =
    [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// The code-length symbol whose length is set nonzero, i.e. the constant the
/// degenerate code-length code emits. It is the code length every alphabet
/// symbol then receives, so it must equal [`TOKEN_BITS`].
const FLAT_CODE_LENGTH: usize = 4;

/// Canonical codes of RFC 7932 section 3.5's fixed code-length code, derived
/// from its lengths `[2, 4, 3, 2, 2, 4]` and written most-significant-bit
/// first (the order a prefix code is consumed in).
///
/// Only the two entries this module emits are listed; the rest of the table is
/// not needed and transcribing it would be transcribing the RFC.
const CLC_CODE_ZERO: [u8; 2] = [0, 0];
const CLC_CODE_ONE: [u8; 4] = [1, 1, 1, 0];

/// Writes the C.2.1 distribution bundle for a stream with `num_dist`
/// pre-clustered contexts.
///
/// Every context is mapped to a single cluster, so one prefix code and one
/// hybrid-uint configuration follow.
///
/// # Errors
///
/// Only through the bit writer; every field here is a compile-time constant.
pub fn write_bundle(w: &mut BitWriter, num_dist: usize) -> Result<()> {
    // Table C.1: lz77.enabled. Disabled, so no min_symbol/min_length and no
    // extra distance context.
    w.write_bool(false);

    // C.2.2: a single context is its own cluster and nothing is written.
    if num_dist > 1 {
        w.write_bool(true); // is_simple
        w.write_bits(2, 0)?; // nbits = 0, so every context reads as cluster 0
    }

    // C.2.1: use_prefix_code, which fixes log_alphabet_size at 15.
    w.write_bool(true);

    // C.2.3 with log_alphabet_size 15: split_exponent is u(4). Zero is not 15,
    // so msb_in_token u(0) and lsb_in_token u(0) follow — both empty.
    w.write_bits(4, 0)?;

    // C.2.1 alphabet size: 1 + (1 << n) + u(n). n = 3 and u(3) = 7 gives 16.
    w.write_bool(true);
    w.write_bits(4, 3)?;
    w.write_bits(3, 7)?;

    write_prefix_code(w)
}

/// Writes the flat 16-symbol prefix code of RFC 7932 section 3.5.
fn write_prefix_code(w: &mut BitWriter) -> Result<()> {
    // The 2-bit selector doubles as HSKIP; 1 would mean "simple code", so 0
    // both selects the complex form and skips nothing.
    w.write_bits(2, 0)?;

    // The lengths of the code-length code, in CODE_LENGTH_ORDER. Exactly one
    // symbol gets a nonzero length, which makes the resulting code zero-bit.
    for symbol in CODE_LENGTH_ORDER {
        if symbol == FLAT_CODE_LENGTH {
            write_code(w, &CLC_CODE_ONE)?;
        } else {
            write_code(w, &CLC_CODE_ZERO)?;
        }
    }
    // The 16 alphabet code lengths follow, and cost zero bits each.
    Ok(())
}

/// Writes the bits of a prefix code word in read order, i.e. most significant
/// first (18181-1 C.2.4).
fn write_code(w: &mut BitWriter, bits: &[u8]) -> Result<()> {
    for &bit in bits {
        w.write_bits(1, u32::from(bit))?;
    }
    Ok(())
}

/// Writes one token of the flat code: four bits, most significant first.
fn write_token(w: &mut BitWriter, token: u32) -> Result<()> {
    if token >= ALPHABET_SIZE {
        return Err(EncodeError::ValueOutOfRange {
            what: "prefix-code token",
            value: i64::from(token),
        });
    }
    for shift in (0..TOKEN_BITS).rev() {
        w.write_bits(1, (token >> shift) & 1)?;
    }
    Ok(())
}

/// Writes one unsigned integer, i.e. the inverse of `DecodeHybridVarLenUint`
/// (18181-1 C.3.3) under this module's fixed configuration.
///
/// # Errors
///
/// [`EncodeError::ValueOutOfRange`] if `value` exceeds [`MAX_VALUE`].
pub fn write_uint(w: &mut BitWriter, value: u32) -> Result<()> {
    if value > MAX_VALUE {
        return Err(EncodeError::ValueOutOfRange {
            what: "entropy-coded value",
            value: i64::from(value),
        });
    }
    if value == 0 {
        return write_token(w, 0);
    }
    // The token names the position of the most significant set bit; the bits
    // below it ride along as raw extra bits.
    let n = 31 - value.leading_zeros();
    write_token(w, n + 1)?;
    let extra = value - (1 << n);
    w.write_bits(n, extra)?;
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

    /// Encodes `values` through one bundle and decodes them back with
    /// `jpxl-entropy`, which is the only thing that proves the bundle is legal.
    fn round_trip(num_dist: usize, contexts: &[usize], values: &[u32]) {
        let mut w = BitWriter::new();
        write_bundle(&mut w, num_dist).expect("bundle");
        for &v in values {
            write_uint(&mut w, v).expect("value fits");
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
    fn single_context_bundle_round_trips() {
        let values: Vec<u32> = (0..64).chain([255, 256, 511, 512, MAX_VALUE]).collect();
        round_trip(1, &vec![0; values.len()], &values);
    }

    #[test]
    fn every_context_of_a_six_context_bundle_maps_to_one_cluster() {
        // The MA-tree stream of H.4.2 uses contexts 0..6.
        let contexts: Vec<usize> = (0..6).cycle().take(30).collect();
        let values: Vec<u32> = (0..30).map(|i| i * 37).collect();
        round_trip(6, &contexts, &values);
    }

    #[test]
    fn every_encodable_value_round_trips() {
        // Exhaustive over the full range the configuration can express: this
        // is what proves the token/extra-bit split, not a sample of it.
        let mut w = BitWriter::new();
        write_bundle(&mut w, 1).expect("bundle");
        for v in 0..=MAX_VALUE {
            write_uint(&mut w, v).expect("value fits");
        }
        w.zero_pad_to_byte();
        let bytes = w.into_bytes();

        let limits = Limits::relaxed();
        let mut guard = AllocGuard::new(&limits);
        let mut r = BitReader::new(&bytes);
        let mut dec =
            jpxl_entropy::SymbolDecoder::open(&mut r, 1, &mut guard).expect("bundle opens");
        for v in 0..=MAX_VALUE {
            assert_eq!(dec.read_uint(&mut r, 0).expect("symbol"), v);
        }
    }

    #[test]
    fn values_above_the_alphabet_are_rejected() {
        let mut w = BitWriter::new();
        write_bundle(&mut w, 1).expect("bundle");
        assert!(matches!(
            write_uint(&mut w, MAX_VALUE + 1),
            Err(EncodeError::ValueOutOfRange { .. })
        ));
    }

    #[test]
    fn the_bundle_is_the_size_the_module_documents() {
        // 1 lz77 + 1 prefix flag + 4 config + 8 alphabet size + 2 hskip
        // + 17 * 2 + 4 code-length words = 54 bits for one context.
        let mut w = BitWriter::new();
        write_bundle(&mut w, 1).expect("bundle");
        assert_eq!(w.bit_len(), 54);

        // Six contexts add the three-bit simple cluster map.
        let mut w = BitWriter::new();
        write_bundle(&mut w, 6).expect("bundle");
        assert_eq!(w.bit_len(), 57);
    }
}
