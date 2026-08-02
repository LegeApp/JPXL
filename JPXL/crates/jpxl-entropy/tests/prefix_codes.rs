//! Prefix-code vectors — ISO/IEC 18181-1 C.2.4 via IETF RFC 7932 sections
//! 3.2, 3.4 and 3.5.
//!
//! Codes are consumed most-significant-bit-first (C.2.4 concatenates single
//! `u(1)` reads left to right), so the `code()` helper writes the bits in
//! exactly the order the decoder reads them. This layer is bit-exact.

// Test code: an out-of-range index is a failed assertion rather than an attack
// surface, and the casts are on values these tests chose themselves. The lints
// exist for the decode paths in src/.
#![allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]

mod common;

use common::BitWriter;
use jpxl_bitstream::BitReader;
use jpxl_core::limits::{AllocGuard, Limits};
use jpxl_entropy::{PrefixCode, read_prefix_code};

fn guard() -> AllocGuard {
    AllocGuard::new(&Limits::relaxed())
}

/// C.2.4: an alphabet of one reads no histogram and no per-symbol bits.
#[test]
fn degenerate_alphabet_reads_nothing_at_all() {
    let mut g = guard();
    let data = [0xFFu8; 8];
    let mut r = BitReader::new(&data);
    let code = read_prefix_code(&mut r, 1, &mut g).expect("degenerate");
    assert_eq!(r.total_bits_read(), 0);
    assert_eq!(code.constant_symbol(), Some(0));
    for _ in 0..100 {
        assert_eq!(code.decode(&mut r).expect("symbol"), 0);
    }
    assert_eq!(r.total_bits_read(), 0);
}

/// RFC 7932 section 3.4, `NSYM = 1`: one symbol, code length zero.
#[test]
fn simple_code_with_one_symbol() {
    // u(2) = 1        -> simple prefix code
    // u(2) = 0        -> NSYM = 1
    // u(3) = 5        -> the single symbol (alphabet size 8 -> ALPHABET_BITS 3)
    let mut w = BitWriter::new();
    w.u(1, 2);
    w.u(0, 2);
    w.u(5, 3);
    let data = w.finish();
    let mut r = BitReader::new(&data);
    let mut g = guard();
    let code = read_prefix_code(&mut r, 8, &mut g).expect("simple code");
    assert_eq!(r.total_bits_read(), 7);
    assert_eq!(code.constant_symbol(), Some(5));

    let before = r.total_bits_read();
    assert_eq!(code.decode(&mut r).expect("symbol"), 5);
    assert_eq!(r.total_bits_read(), before, "NSYM = 1 costs no bits");
}

/// RFC 7932 section 3.4, `NSYM = 2`: both symbols get length 1.
#[test]
fn simple_code_with_two_symbols() {
    // u(2) = 1 (simple), u(2) = 1 (NSYM = 2), symbols 6 then 2 in u(3).
    // Lengths are 1 and 1, so canonical assignment orders by symbol value:
    //   symbol 2 -> code 0, symbol 6 -> code 1.
    let mut w = BitWriter::new();
    w.u(1, 2);
    w.u(1, 2);
    w.u(6, 3);
    w.u(2, 3);
    // Symbol stream: 0, 1, 1, 0 -> 2, 6, 6, 2
    w.code(&[0, 1, 1, 0]);
    let data = w.finish();
    let mut r = BitReader::new(&data);
    let mut g = guard();
    let code = read_prefix_code(&mut r, 8, &mut g).expect("simple code");
    assert_eq!(r.total_bits_read(), 10);
    assert_eq!(code.decode(&mut r).expect("s"), 2);
    assert_eq!(code.decode(&mut r).expect("s"), 6);
    assert_eq!(code.decode(&mut r).expect("s"), 6);
    assert_eq!(code.decode(&mut r).expect("s"), 2);
    assert_eq!(r.total_bits_read(), 14);
}

/// RFC 7932 section 3.4, `NSYM = 3`: lengths 1, 2, 2 in the order read.
#[test]
fn simple_code_with_three_symbols() {
    // Symbols read in order 4, 1, 7 -> lengths 1, 2, 2 respectively.
    // Canonical: symbol 4 has length 1 -> code 0.
    //            symbols 1 and 7 have length 2 -> codes 10 and 11 by value,
    //            so symbol 1 -> 10, symbol 7 -> 11.
    let mut w = BitWriter::new();
    w.u(1, 2);
    w.u(2, 2); // NSYM - 1 = 2
    w.u(4, 3);
    w.u(1, 3);
    w.u(7, 3);
    w.code(&[0]); // -> 4
    w.code(&[1, 0]); // -> 1
    w.code(&[1, 1]); // -> 7
    let data = w.finish();
    let mut r = BitReader::new(&data);
    let mut g = guard();
    let code = read_prefix_code(&mut r, 8, &mut g).expect("simple code");
    assert_eq!(code.decode(&mut r).expect("s"), 4);
    assert_eq!(code.decode(&mut r).expect("s"), 1);
    assert_eq!(code.decode(&mut r).expect("s"), 7);
}

/// RFC 7932 section 3.4, `NSYM = 4`: the tree-select bit picks between a flat
/// code and a skewed one.
#[test]
fn simple_code_with_four_symbols_and_tree_select() {
    for (tree_select, expected_lengths) in [(false, [2, 2, 2, 2]), (true, [1, 2, 3, 3])] {
        // Symbols read in order 0, 1, 2, 3.
        let mut w = BitWriter::new();
        w.u(1, 2);
        w.u(3, 2); // NSYM - 1 = 3
        for symbol in 0..4u32 {
            w.u(symbol, 2); // alphabet size 4 -> ALPHABET_BITS = 2
        }
        w.bit(tree_select);
        let data = w.finish();
        let mut r = BitReader::new(&data);
        let mut g = guard();
        let code = read_prefix_code(&mut r, 4, &mut g).expect("simple code");

        // Rebuild the same code from the documented lengths and require the
        // two to be identical, which pins the tree-select mapping.
        let reference =
            PrefixCode::from_lengths(&expected_lengths.map(|l| l as u8)).expect("reference");
        assert_eq!(code, reference, "tree_select = {tree_select}");
    }
}

/// RFC 7932 section 3.4: a repeated symbol is invalid.
#[test]
fn simple_code_rejects_duplicate_symbols() {
    let mut w = BitWriter::new();
    w.u(1, 2);
    w.u(1, 2); // NSYM = 2
    w.u(3, 3);
    w.u(3, 3); // the same symbol twice
    let data = w.finish();
    let mut r = BitReader::new(&data);
    let mut g = guard();
    assert!(read_prefix_code(&mut r, 8, &mut g).is_err());
}

/// RFC 7932 section 3.4: a symbol outside the alphabet is invalid.
#[test]
fn simple_code_rejects_out_of_range_symbols() {
    // Alphabet size 5 -> ALPHABET_BITS = 3, so 3 bits can express 5, 6 and 7,
    // none of which are in the alphabet.
    let mut w = BitWriter::new();
    w.u(1, 2);
    w.u(0, 2); // NSYM = 1
    w.u(6, 3);
    let data = w.finish();
    let mut r = BitReader::new(&data);
    let mut g = guard();
    assert!(read_prefix_code(&mut r, 5, &mut g).is_err());
}

/// RFC 7932 section 3.5, the full complex path, hand-derived end to end.
#[test]
fn complex_code_hand_derived() {
    // Target: alphabet size 3 with code lengths [1, 2, 2], i.e.
    //   symbol 0 -> 0, symbol 1 -> 10, symbol 2 -> 11.
    //
    // Step 0: u(2) = 0 -> HSKIP = 0 (the value 1 would mean a simple code).
    //
    // Step 1: code lengths for the code-length alphabet, in the fixed order
    //         1, 2, 3, 4, 0, 5, 17, 6, 16, 7, ... using the fixed code whose
    //         canonical values are 0 -> 00, 1 -> 1110, 2 -> 110, 3 -> 01,
    //         4 -> 10, 5 -> 1111.
    //         We give code-length symbols 1 and 2 a length of 1 each:
    //           write "1110" (the fixed code for the value 1) for symbol 1
    //           write "1110" again                            for symbol 2
    //         Space accounting: 32 - (32 >> 1) = 16, then 16 - 16 = 0, so
    //         reading stops after two entries.
    //         The code-length code is therefore lengths [.,1,1,...] ->
    //           code-length symbol 1 -> "0", symbol 2 -> "1".
    //
    // Step 2: the alphabet's own code lengths, using that code:
    //           "0" -> length 1  (symbol 0)   space 32768 - 16384 = 16384
    //           "1" -> length 2  (symbol 1)   space 16384 -  8192 =  8192
    //           "1" -> length 2  (symbol 2)   space  8192 -  8192 =     0
    let mut w = BitWriter::new();
    w.u(0, 2);
    w.code(&[1, 1, 1, 0]);
    w.code(&[1, 1, 1, 0]);
    w.code(&[0]);
    w.code(&[1]);
    w.code(&[1]);
    let header_bits = w.bit_len();
    assert_eq!(header_bits, 13);

    // Symbol stream: 0, 2, 1, 0
    w.code(&[0]);
    w.code(&[1, 1]);
    w.code(&[1, 0]);
    w.code(&[0]);

    let data = w.finish();
    let mut r = BitReader::new(&data);
    let mut g = guard();
    let code = read_prefix_code(&mut r, 3, &mut g).expect("complex code");
    assert_eq!(r.total_bits_read() as usize, header_bits);
    assert_eq!(
        code,
        PrefixCode::from_lengths(&[1, 2, 2]).expect("reference"),
        "the complex path must produce the lengths derived above"
    );

    assert_eq!(code.decode(&mut r).expect("s"), 0);
    assert_eq!(code.decode(&mut r).expect("s"), 2);
    assert_eq!(code.decode(&mut r).expect("s"), 1);
    assert_eq!(code.decode(&mut r).expect("s"), 0);
    assert_eq!(r.total_bits_read() as usize, header_bits + 6);
}

/// RFC 7932 section 3.5: repeat code 17 fills a run of zero lengths.
#[test]
fn complex_code_with_a_zero_run() {
    // Target: alphabet size 8 with lengths [1, 0, 0, 0, 0, 0, 2, 2].
    // Emitted as: length 1, then a run of five zeros via code 17, then 2, 2.
    //
    // Code-length alphabet: we need symbols 1, 2 and 17. Give each length 2,
    // which needs a fourth symbol to complete the code; use symbol 0.
    // Lengths of 2 for four symbols: 4 * (32 >> 2) = 32, exactly complete.
    //
    // Fixed-code order is 1, 2, 3, 4, 0, 5, 17, ... so we write, in order:
    //   symbol 1  -> length 2 -> fixed code for 2   = "110"
    //   symbol 2  -> length 2 -> "110"
    //   symbol 3  -> length 0 -> fixed code for 0   = "00"
    //   symbol 4  -> length 0 -> "00"
    //   symbol 0  -> length 2 -> "110"
    //   symbol 5  -> length 0 -> "00"
    //   symbol 17 -> length 2 -> "110"   (space now 0, reading stops)
    //
    // Canonical code-length code over lengths {0:2, 1:2, 2:2, 17:2}:
    //   symbol 0 -> 00, symbol 1 -> 01, symbol 2 -> 10, symbol 17 -> 11
    //
    // Alphabet lengths:
    //   "01"      -> 1                       space 32768 - 16384 = 16384
    //   "11" + u(3) = 2 -> code 17, repeat 3 + 2 = 5 zeros
    //   "10"      -> 2                       space 16384 -  8192 =  8192
    //   "10"      -> 2                       space  8192 -  8192 =     0
    let mut w = BitWriter::new();
    w.u(0, 2); // HSKIP = 0
    w.code(&[1, 1, 0]); // symbol 1  -> 2
    w.code(&[1, 1, 0]); // symbol 2  -> 2
    w.code(&[0, 0]); // symbol 3  -> 0
    w.code(&[0, 0]); // symbol 4  -> 0
    w.code(&[1, 1, 0]); // symbol 0  -> 2
    w.code(&[0, 0]); // symbol 5  -> 0
    w.code(&[1, 1, 0]); // symbol 17 -> 2
    w.code(&[0, 1]); // length 1 for alphabet symbol 0
    w.code(&[1, 1]); // code 17 ...
    w.u(2, 3); // ... repeat 3 + 2 = 5 zeros
    w.code(&[1, 0]); // length 2
    w.code(&[1, 0]); // length 2

    let data = w.finish();
    let mut r = BitReader::new(&data);
    let mut g = guard();
    let code = read_prefix_code(&mut r, 8, &mut g).expect("complex code");
    assert_eq!(
        code,
        PrefixCode::from_lengths(&[1, 0, 0, 0, 0, 0, 2, 2]).expect("reference")
    );
}

/// An incomplete code-length assignment must be rejected rather than decoded
/// into ambiguous symbols.
#[test]
fn complex_code_rejects_an_incomplete_alphabet() {
    // Same code-length code as above, but the alphabet lengths stop short:
    // one symbol of length 2 leaves 24576 of the 32768 space unused.
    let mut w = BitWriter::new();
    w.u(0, 2);
    w.code(&[1, 1, 0]);
    w.code(&[1, 1, 0]);
    w.code(&[0, 0]);
    w.code(&[0, 0]);
    w.code(&[1, 1, 0]);
    w.code(&[0, 0]);
    w.code(&[1, 1, 0]);
    // Alphabet size 2, both symbols given length 2 -> space 32768 - 16384.
    w.code(&[1, 0]);
    w.code(&[1, 0]);
    let data = w.finish();
    let mut r = BitReader::new(&data);
    let mut g = guard();
    assert!(read_prefix_code(&mut r, 2, &mut g).is_err());
}

/// Canonical assignment is exercised across every length in range: build a
/// code, then confirm that decoding each symbol's canonical bit string returns
/// that symbol.
#[test]
fn canonical_codes_round_trip_for_every_symbol() {
    // A deliberately uneven but complete length assignment.
    // Kraft check: 1/2 + 1/4 + 1/8 + 1/16 + 1/32 + 1/32 = 1.
    let lengths = [1u8, 2, 3, 4, 5, 5];
    let code = PrefixCode::from_lengths(&lengths).expect("complete");

    // Canonical values, computed by the RFC 7932 3.2 procedure:
    //   len 1: symbol 0 -> 0
    //   len 2: symbol 1 -> 10
    //   len 3: symbol 2 -> 110
    //   len 4: symbol 3 -> 1110
    //   len 5: symbol 4 -> 11110, symbol 5 -> 11111
    let expected: [(u32, &[u8]); 6] = [
        (0, &[0]),
        (1, &[1, 0]),
        (2, &[1, 1, 0]),
        (3, &[1, 1, 1, 0]),
        (4, &[1, 1, 1, 1, 0]),
        (5, &[1, 1, 1, 1, 1]),
    ];
    for (symbol, bits) in expected {
        let mut w = BitWriter::new();
        w.code(bits);
        let data = w.finish();
        let mut r = BitReader::new(&data);
        assert_eq!(code.decode(&mut r).expect("symbol"), symbol);
        assert_eq!(r.total_bits_read() as usize, bits.len());
    }
}
