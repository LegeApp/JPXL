//! Canonical (Brotli-style) prefix codes.
//!
//! ISO/IEC 18181-1 C.2.4 does not restate the prefix-code format; it
//! references IETF RFC 7932:2016 sections 3.2 (canonical prefix coding), 3.4
//! (simple prefix codes) and 3.5 (complex prefix codes). This module
//! implements those three sections, plus the C.2.4 special case where an
//! alphabet size of 1 consumes no bits at all and every decoded symbol is 0.
//!
//! # Bit order
//!
//! JPEG XL reads bits least-significant-first within a byte, but a prefix code
//! is consumed most-significant-bit-first: C.2.4 says the decoder concatenates
//! single `u(1)` reads left to right until the accumulated string matches a
//! code. So the decoder accumulates `code = (code << 1) | bit`.
//!
//! This matters for reading the tables printed in the specifications. Both RFC
//! 7932 section 3.5 and 18181-1 C.2.5 print their fixed code tables in
//! *stream* order (the leftmost printed bit is read first), which is the
//! reverse of the canonical numeric value. For example RFC 7932's code-length
//! code prints symbol 1 as `0111`; its canonical value is `1110`. The two
//! agree exactly with the canonical construction of section 3.2 applied to the
//! code lengths `[2, 4, 3, 2, 2, 4]`, which is how this module builds it —
//! deriving the table rather than transcribing it removes the ambiguity.
//!
//! Every path here is **bit-exact** (see `docs/PLAN.md`).

use jpxl_bitstream::BitReader;
use jpxl_core::limits::AllocGuard;

use crate::error::{Result, malformed};

/// Longest permitted code length (RFC 7932 section 3.2).
pub const MAX_CODE_LENGTH: usize = 15;

/// Size of the code-length alphabet of RFC 7932 section 3.5: lengths 0..=15
/// plus the two repeat codes 16 and 17.
const CODE_LENGTH_ALPHABET: usize = 18;

/// Order in which code-length-alphabet lengths appear (RFC 7932 section 3.5).
const CODE_LENGTH_ORDER: [usize; CODE_LENGTH_ALPHABET] =
    [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// Code lengths of the fixed code used to read the code-length alphabet
/// (RFC 7932 section 3.5). Canonicalizing these reproduces the table printed
/// in the RFC; see the module-level note on bit order.
const CODE_LENGTH_CODE_LENGTHS: [u8; 6] = [2, 4, 3, 2, 2, 4];

/// Repeat-previous-length code of the code-length alphabet.
const REPEAT_PREVIOUS: usize = 16;
/// Repeat-zero code of the code-length alphabet.
const REPEAT_ZERO: usize = 17;

/// Number of bits needed to represent `n`, i.e. `ceil(log2(n + 1))`.
const fn bit_width(n: u32) -> u32 {
    u32::BITS - n.leading_zeros()
}

/// A canonical prefix code over an alphabet of unsigned integer symbols.
///
/// Built either from an explicit list of code lengths (RFC 7932 section 3.2)
/// or as a degenerate single-symbol code that consumes no bits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixCode {
    /// Number of codes of each length; index 0 is unused and always 0.
    length_counts: [u32; MAX_CODE_LENGTH + 1],
    /// Symbols ordered by (code length, symbol value) — canonical order.
    sorted_symbols: Vec<u32>,
    /// Set for a code whose single symbol has length 0 and costs no bits.
    single_symbol: Option<u32>,
}

impl PrefixCode {
    /// A code with one symbol and no bits, per RFC 7932 section 3.4 (`NSYM = 1`)
    /// and the 18181-1 C.2.4 `alphabet_size == 1` special case.
    #[must_use]
    pub fn single(symbol: u32) -> Self {
        Self {
            length_counts: [0; MAX_CODE_LENGTH + 1],
            sorted_symbols: Vec::new(),
            single_symbol: Some(symbol),
        }
    }

    /// Builds a canonical prefix code from per-symbol code lengths.
    ///
    /// Follows the assignment procedure of RFC 7932 section 3.2. A length of 0
    /// means the symbol is absent. The code must be complete: the Kraft sum
    /// over all present symbols must be exactly 1.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Malformed`](crate::EntropyError::Malformed) if a length
    /// exceeds [`MAX_CODE_LENGTH`], if the code is over- or under-subscribed,
    /// or if no symbol is present.
    pub fn from_lengths(lengths: &[u8]) -> Result<Self> {
        let mut length_counts = [0u32; MAX_CODE_LENGTH + 1];
        let mut present = 0u32;
        let mut last_symbol = 0u32;
        for (symbol, &len) in lengths.iter().enumerate() {
            let len = usize::from(len);
            if len == 0 {
                continue;
            }
            if len > MAX_CODE_LENGTH {
                return Err(malformed!(
                    "RFC 7932 3.2: code length {len} exceeds the maximum of {MAX_CODE_LENGTH}"
                ));
            }
            let slot = length_counts
                .get_mut(len)
                .ok_or_else(|| malformed!("RFC 7932 3.2: code length {len} out of range"))?;
            *slot += 1;
            present += 1;
            last_symbol = u32::try_from(symbol)
                .map_err(|_| malformed!("RFC 7932 3.2: symbol index out of range"))?;
        }

        match present {
            0 => Err(malformed!(
                "RFC 7932 3.2: prefix code has no symbol with a nonzero length"
            )),
            // A lone symbol cannot form a complete code of nonzero length; the
            // only consistent reading is the zero-length single-symbol code.
            1 => Ok(Self::single(last_symbol)),
            _ => {
                // Kraft sum in units of 2^-MAX_CODE_LENGTH.
                let full = 1u64 << MAX_CODE_LENGTH;
                let mut used = 0u64;
                for (len, &count) in length_counts.iter().enumerate().skip(1) {
                    used += u64::from(count) << (MAX_CODE_LENGTH - len);
                }
                if used != full {
                    return Err(malformed!(
                        "RFC 7932 3.2: prefix code is {} (Kraft sum {used}/{full})",
                        if used < full {
                            "incomplete"
                        } else {
                            "over-subscribed"
                        }
                    ));
                }

                let mut sorted_symbols = Vec::with_capacity(present as usize);
                for len in 1..=MAX_CODE_LENGTH {
                    for (symbol, &l) in lengths.iter().enumerate() {
                        if usize::from(l) == len {
                            sorted_symbols.push(u32::try_from(symbol).map_err(|_| {
                                malformed!("RFC 7932 3.2: symbol index out of range")
                            })?);
                        }
                    }
                }
                Ok(Self {
                    length_counts,
                    sorted_symbols,
                    single_symbol: None,
                })
            }
        }
    }

    /// The symbol this code always yields, if it is a zero-bit code.
    #[must_use]
    pub const fn constant_symbol(&self) -> Option<u32> {
        self.single_symbol
    }

    /// Decodes one symbol (18181-1 C.2.4).
    ///
    /// Reads single bits and accumulates them most-significant-first until the
    /// accumulated string matches a code. A zero-bit code returns its symbol
    /// without touching the reader.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Malformed`](crate::EntropyError::Malformed) if
    /// [`MAX_CODE_LENGTH`] bits are consumed without a match, or a bitstream
    /// error if the input is exhausted.
    pub fn decode(&self, reader: &mut BitReader<'_>) -> Result<u32> {
        if let Some(symbol) = self.single_symbol {
            return Ok(symbol);
        }
        let mut code = 0u32;
        let mut first = 0u32;
        let mut index = 0u32;
        for len in 1..=MAX_CODE_LENGTH {
            code |= reader.read_bits(1)?;
            let count = *self
                .length_counts
                .get(len)
                .ok_or_else(|| malformed!("RFC 7932 3.2: length {len} out of range"))?;
            if code < first + count {
                let pos = index + (code - first);
                return self
                    .sorted_symbols
                    .get(pos as usize)
                    .copied()
                    .ok_or_else(|| malformed!("RFC 7932 3.2: canonical index {pos} out of range"));
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err(malformed!(
            "RFC 7932 3.2: no code matched within {MAX_CODE_LENGTH} bits"
        ))
    }
}

/// Reads a prefix code from the bitstream (18181-1 C.2.4).
///
/// `alphabet_size` is supplied by the caller, as the clause requires. The
/// special case `alphabet_size == 1` reads no bits and yields a code that
/// always produces symbol 0.
///
/// # Errors
///
/// [`EntropyError::Malformed`](crate::EntropyError::Malformed) for a stream
/// that violates RFC 7932 section 3.4 or 3.5, or an allocation-limit error.
pub fn read_prefix_code(
    reader: &mut BitReader<'_>,
    alphabet_size: usize,
    guard: &mut AllocGuard,
) -> Result<PrefixCode> {
    if alphabet_size == 0 {
        return Err(malformed!("C.2.4: alphabet size must be at least 1"));
    }
    // C.2.4: no histogram is present and every symbol decodes to 0.
    if alphabet_size == 1 {
        return Ok(PrefixCode::single(0));
    }

    let selector = reader.read_bits(2)?;
    if selector == 1 {
        read_simple_prefix_code(reader, alphabet_size)
    } else {
        read_complex_prefix_code(reader, alphabet_size, selector, guard)
    }
}

/// RFC 7932 section 3.4.
fn read_simple_prefix_code(reader: &mut BitReader<'_>, alphabet_size: usize) -> Result<PrefixCode> {
    let alphabet_bits = bit_width(
        u32::try_from(alphabet_size - 1)
            .map_err(|_| malformed!("RFC 7932 3.4: alphabet size out of range"))?,
    );
    let nsym = reader.read_bits(2)? as usize + 1;

    let mut symbols = [0u32; 4];
    for i in 0..nsym {
        let symbol = reader.read_bits(alphabet_bits)?;
        if symbol as usize >= alphabet_size {
            return Err(malformed!(
                "RFC 7932 3.4: symbol {symbol} is outside the alphabet of size {alphabet_size}"
            ));
        }
        if symbols.get(..i).is_some_and(|prev| prev.contains(&symbol)) {
            return Err(malformed!("RFC 7932 3.4: duplicate symbol {symbol}"));
        }
        let slot = symbols
            .get_mut(i)
            .ok_or_else(|| malformed!("RFC 7932 3.4: NSYM {nsym} out of range"))?;
        *slot = symbol;
    }

    // Code lengths in the order the symbols were read.
    let code_lengths: &[u8] = match nsym {
        1 => return Ok(PrefixCode::single(symbols[0])),
        2 => &[1, 1],
        3 => &[1, 2, 2],
        4 => {
            if reader.read_bool()? {
                &[1, 2, 3, 3]
            } else {
                &[2, 2, 2, 2]
            }
        }
        _ => return Err(malformed!("RFC 7932 3.4: NSYM {nsym} out of range")),
    };

    let mut lengths = vec![0u8; alphabet_size];
    for (i, &len) in code_lengths.iter().enumerate() {
        let symbol = *symbols
            .get(i)
            .ok_or_else(|| malformed!("RFC 7932 3.4: symbol index {i} out of range"))?;
        let slot = lengths
            .get_mut(symbol as usize)
            .ok_or_else(|| malformed!("RFC 7932 3.4: symbol {symbol} out of range"))?;
        *slot = len;
    }
    PrefixCode::from_lengths(&lengths)
}

/// RFC 7932 section 3.5.
fn read_complex_prefix_code(
    reader: &mut BitReader<'_>,
    alphabet_size: usize,
    hskip: u32,
    guard: &mut AllocGuard,
) -> Result<PrefixCode> {
    // The 2-bit selector doubles as HSKIP; the value 1 was consumed by the
    // caller as the "simple code" marker, so only 0, 2 and 3 reach here.
    let hskip = hskip as usize;

    // Step 1: the lengths of the code-length code itself.
    let fixed = PrefixCode::from_lengths(&CODE_LENGTH_CODE_LENGTHS)?;
    let mut clc_lengths = [0u8; CODE_LENGTH_ALPHABET];
    let mut space = 32i32;
    let mut nonzero = 0u32;
    let order = CODE_LENGTH_ORDER
        .get(hskip..)
        .ok_or_else(|| malformed!("RFC 7932 3.5: HSKIP {hskip} out of range"))?;
    for &symbol in order {
        let len = fixed.decode(reader)?;
        let slot = clc_lengths
            .get_mut(symbol)
            .ok_or_else(|| malformed!("RFC 7932 3.5: code-length symbol {symbol} out of range"))?;
        // A length above 5 is unreachable: the fixed code has six symbols.
        *slot = u8::try_from(len)
            .map_err(|_| malformed!("RFC 7932 3.5: code length {len} out of range"))?;
        if len != 0 {
            nonzero += 1;
            space -= 32 >> len;
            if space <= 0 {
                break;
            }
        }
    }
    if nonzero != 1 && space != 0 {
        return Err(malformed!(
            "RFC 7932 3.5: code-length code is not complete (residual space {space})"
        ));
    }

    let clc = if nonzero == 1 {
        // RFC 7932 3.5: a lone nonzero length yields a one-symbol code whose
        // code has zero length.
        let symbol = clc_lengths
            .iter()
            .position(|&l| l != 0)
            .ok_or_else(|| malformed!("RFC 7932 3.5: no nonzero code length found"))?;
        PrefixCode::single(
            u32::try_from(symbol)
                .map_err(|_| malformed!("RFC 7932 3.5: symbol index out of range"))?,
        )
    } else {
        PrefixCode::from_lengths(&clc_lengths)?
    };

    // Step 2: the code lengths of the alphabet proper.
    guard.charge(alphabet_size as u64)?;
    let mut lengths = vec![0u8; alphabet_size];
    let mut space = 32_768i64;
    let mut index = 0usize;
    // RFC 7932 3.5: a leading repeat code copies a length of 8.
    let mut previous_nonzero = 8u8;
    let mut repeat = 0u32;
    let mut repeat_length = 0u8;

    while index < alphabet_size && space > 0 {
        let symbol = clc.decode(reader)? as usize;
        if symbol < REPEAT_PREVIOUS {
            repeat = 0;
            let len = u8::try_from(symbol)
                .map_err(|_| malformed!("RFC 7932 3.5: code length {symbol} out of range"))?;
            let slot = lengths
                .get_mut(index)
                .ok_or_else(|| malformed!("RFC 7932 3.5: length index {index} out of range"))?;
            *slot = len;
            index += 1;
            if len != 0 {
                previous_nonzero = len;
                space -= 32_768i64 >> len;
            }
            continue;
        }

        let (extra_bits, new_length) = if symbol == REPEAT_PREVIOUS {
            (2u32, previous_nonzero)
        } else if symbol == REPEAT_ZERO {
            (3u32, 0u8)
        } else {
            return Err(malformed!(
                "RFC 7932 3.5: code-length symbol {symbol} is outside the alphabet"
            ));
        };

        // Consecutive repeat codes of the same kind extend the previous run;
        // a different kind restarts it.
        if repeat_length != new_length {
            repeat = 0;
            repeat_length = new_length;
        }
        let previous_repeat = repeat;
        if repeat > 0 {
            repeat = repeat
                .checked_sub(2)
                .and_then(|r| r.checked_shl(extra_bits))
                .ok_or_else(|| malformed!("RFC 7932 3.5: repeat count overflow"))?;
        }
        repeat = repeat
            .checked_add(3 + reader.read_bits(extra_bits)?)
            .ok_or_else(|| malformed!("RFC 7932 3.5: repeat count overflow"))?;
        let count = repeat - previous_repeat;

        let end = index
            .checked_add(count as usize)
            .ok_or_else(|| malformed!("RFC 7932 3.5: repeat count overflow"))?;
        if end > alphabet_size {
            return Err(malformed!(
                "RFC 7932 3.5: repeat of {count} runs past the alphabet size {alphabet_size}"
            ));
        }
        let run = lengths
            .get_mut(index..end)
            .ok_or_else(|| malformed!("RFC 7932 3.5: repeat range out of bounds"))?;
        run.fill(new_length);
        index = end;
        if new_length != 0 {
            space -= i64::from(count) * (32_768i64 >> new_length);
        }
    }

    if space != 0 {
        return Err(malformed!(
            "RFC 7932 3.5: prefix code is not complete (residual space {space})"
        ));
    }
    PrefixCode::from_lengths(&lengths)
}

#[cfg(test)]
// In test code an out-of-range index is a failed assertion rather than an attack
// surface, and the casts are on values these tests chose themselves. The lints
// exist for the decode paths above.
#[allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]
mod tests {
    use super::*;

    /// RFC 7932 section 3.2 worked example: alphabet ABCD, lengths (2,1,3,3),
    /// canonical codes A=10, B=0, C=110, D=111.
    #[test]
    fn canonical_assignment_matches_the_rfc_example() {
        let code = PrefixCode::from_lengths(&[2, 1, 3, 3]).expect("complete code");
        // Codes are consumed MSB-first, so the stream bits for "10" are 1 then 0.
        // Pack LSB-first per JPEG XL: bit0 = first bit read.
        // B=0 -> bits [0]; A=10 -> [1,0]; C=110 -> [1,1,0]; D=111 -> [1,1,1]
        // sequence B,A,C,D = 0 | 1,0 | 1,1,0 | 1,1,1  (9 bits)
        // byte0 bits b0..b7 = 0,1,0,1,1,0,1,1 = 2 + 8 + 16 + 64 + 128 = 218
        // byte1 bit c0 = 1
        let data = [218u8, 1];
        let mut r = BitReader::new(&data);
        assert_eq!(code.decode(&mut r).expect("B"), 1);
        assert_eq!(code.decode(&mut r).expect("A"), 0);
        assert_eq!(code.decode(&mut r).expect("C"), 2);
        assert_eq!(code.decode(&mut r).expect("D"), 3);
        assert_eq!(r.total_bits_read(), 9);
    }

    /// The fixed code-length code of RFC 7932 section 3.5, derived by
    /// canonicalizing the lengths, must reproduce the table printed in the RFC
    /// once each printed string is reversed (see the module bit-order note).
    #[test]
    fn code_length_code_matches_the_printed_table() {
        let code = PrefixCode::from_lengths(&CODE_LENGTH_CODE_LENGTHS).expect("complete");
        // Printed in the RFC (stream order) -> canonical (MSB-first) value:
        //   0 -> "00"   -> 0b00
        //   1 -> "0111" -> 0b1110
        //   2 -> "011"  -> 0b110
        //   3 -> "10"   -> 0b01
        //   4 -> "01"   -> 0b10
        //   5 -> "1111" -> 0b1111
        let expected: [(u32, &[u8]); 6] = [
            (0, &[0, 0]),
            (1, &[1, 1, 1, 0]),
            (2, &[1, 1, 0]),
            (3, &[0, 1]),
            (4, &[1, 0]),
            (5, &[1, 1, 1, 1]),
        ];
        for (symbol, bits) in expected {
            let packed = pack_msb_first_bits(bits);
            let mut r = BitReader::new(&packed);
            assert_eq!(code.decode(&mut r).expect("decodes"), symbol);
            assert_eq!(r.total_bits_read() as usize, bits.len());
        }
    }

    /// Packs a sequence of code bits (in the order they are read) into bytes,
    /// LSB-first within each byte as JPEG XL requires.
    fn pack_msb_first_bits(bits: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; bits.len().div_ceil(8)];
        for (i, &bit) in bits.iter().enumerate() {
            if bit != 0 {
                out[i / 8] |= 1 << (i % 8);
            }
        }
        out
    }

    #[test]
    fn alphabet_size_one_reads_nothing() {
        let mut guard = AllocGuard::new(&jpxl_core::limits::Limits::relaxed());
        let data = [0xFFu8; 4];
        let mut r = BitReader::new(&data);
        let code = read_prefix_code(&mut r, 1, &mut guard).expect("degenerate code");
        assert_eq!(r.total_bits_read(), 0, "C.2.4: no histogram is read");
        assert_eq!(code.decode(&mut r).expect("symbol"), 0);
        assert_eq!(r.total_bits_read(), 0, "and no bits per symbol either");
    }

    #[test]
    fn incomplete_and_oversubscribed_codes_are_rejected() {
        // Two symbols of length 2 cover only half the code space.
        assert!(PrefixCode::from_lengths(&[2, 2]).is_err());
        // Three symbols of length 1 over-subscribe it.
        assert!(PrefixCode::from_lengths(&[1, 1, 1]).is_err());
        // No symbol at all.
        assert!(PrefixCode::from_lengths(&[0, 0]).is_err());
    }

    #[test]
    fn a_lone_length_becomes_a_zero_bit_code() {
        let code = PrefixCode::from_lengths(&[0, 3, 0]).expect("single symbol");
        assert_eq!(code.constant_symbol(), Some(1));
        let mut r = BitReader::new(&[]);
        assert_eq!(code.decode(&mut r).expect("no bits"), 1);
    }
}
