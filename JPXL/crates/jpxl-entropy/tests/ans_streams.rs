//! ANS vectors — ISO/IEC 18181-1 C.2.5, C.2.6, C.3.2.
//!
//! Distributions are hand-derived from the C.2.5 pseudocode, with the
//! logcount codes, the `shift`/`bitcount` arithmetic, and the position of the
//! refinement bits all shown. This layer is bit-exact.

// Test code: an out-of-range index is a failed assertion rather than an attack
// surface, and the casts are on values these tests chose themselves. The lints
// exist for the decode paths in src/.
#![allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]

mod common;

use common::BitWriter;
use jpxl_bitstream::BitReader;
use jpxl_core::limits::{AllocGuard, Limits};
use jpxl_entropy::ans::PROBABILITY_TOTAL;
use jpxl_entropy::{AnsDistribution, SymbolDecoder};

fn guard() -> AllocGuard {
    AllocGuard::new(&Limits::relaxed())
}

/// Asserts the defining property of the C.2.6 alias mapping: across all 4096
/// slots, each symbol appears exactly as often as its probability, and the
/// offsets within a symbol form a permutation of `0..probability`.
fn assert_alias_mapping_is_a_bijection(dist: &AnsDistribution, expected: &[u32]) {
    let mut seen: Vec<Vec<u32>> = vec![Vec::new(); dist.table_size()];
    for x in 0..PROBABILITY_TOTAL {
        let (symbol, offset) = dist.alias_mapping(x).expect("slot in range");
        seen[symbol as usize].push(offset);
    }
    for (symbol, offsets) in seen.iter().enumerate() {
        let want = expected.get(symbol).copied().unwrap_or(0);
        assert_eq!(
            offsets.len() as u32,
            want,
            "symbol {symbol} occupies {} slots but has probability {want}",
            offsets.len()
        );
        let mut sorted = offsets.clone();
        sorted.sort_unstable();
        assert_eq!(
            sorted,
            (0..want).collect::<Vec<_>>(),
            "offsets for symbol {symbol} must be a permutation of 0..{want}"
        );
    }
}

/// C.2.5 general form with `shift = 0`, so every logcount is an exact power of
/// two and no refinement bits are read.
#[test]
fn general_form_with_no_refinement_bits() {
    // Target: D = [2048, 1024, 1024] over log_alphabet_size 5.
    //
    // Bool() = 0            not the one/two-symbol shortcut
    // Bool() = 0            not the flat form
    // Bool() = 0            len loop stops immediately -> len = 0
    // shift = u(0) + (1 << 0) - 1 = 0
    // U8()  = 0             -> alphabet_size = 0 + 3 = 3
    //
    // logcounts, using the fixed C.2.5 code (printed right to left, so the
    // value read MSB-first is the printed string reversed):
    //   12 -> printed 0000001 -> read 1000000
    //   11 -> printed 100001  -> read 100001
    //
    // With shift = 0, bitcount = min(max(0, 0 - ((12 - code + 1) >> 1)),
    // code - 1) = 0 for every code here, so D[i] = 1 << (code - 1) with no
    // extra bits:
    //   logcounts[0] = 12 -> 2048   (also the maximum, so omit_pos = 0)
    //   logcounts[1] = 11 -> 1024
    //   logcounts[2] = 11 -> 1024
    // omit_pos = 0 is skipped in the second pass, so total = 2048 and
    // D[0] = 4096 - 2048 = 2048.
    let mut w = BitWriter::new();
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(false); // U8() = 0
    w.code(&[1, 0, 0, 0, 0, 0, 0]); // 12
    w.code(&[1, 0, 0, 0, 0, 1]); // 11
    w.code(&[1, 0, 0, 0, 0, 1]); // 11

    let data = w.finish();
    let mut r = BitReader::new(&data);
    let mut g = guard();
    let dist = AnsDistribution::read(&mut r, 5, &mut g).expect("distribution");
    assert_eq!(r.total_bits_read() as usize, w.bit_len());

    assert_eq!(dist.probability(0), 2048);
    assert_eq!(dist.probability(1), 1024);
    assert_eq!(dist.probability(2), 1024);
    assert_eq!(dist.probability(3), 0);
    assert_alias_mapping_is_a_bijection(&dist, &[2048, 1024, 1024]);
}

/// C.2.5 general form with `shift = 2`, which turns on the refinement bits.
/// Those bits are read in the *second* pass, so the entry at `omit_pos` reads
/// none at all — the ordering is what this vector pins down.
#[test]
fn general_form_with_refinement_bits() {
    // Bool() = 0, Bool() = 0        general form
    // len loop: Bool() = 1, Bool() = 0 -> len = 1
    // shift = u(1) + (1 << 1) - 1 = 1 + 1 = 2
    // U8() = 0 -> alphabet_size = 3
    //
    // logcounts: 12, 11, 10.  omit_pos = 0 (the first maximum).
    //
    // bitcount = min(max(0, shift - ((12 - code + 1) >> 1)), code - 1):
    //   code 12 -> min(max(0, 2 - 0), 11) = 2   but i == omit_pos, so skipped
    //   code 11 -> min(max(0, 2 - 1), 10) = 1
    //   code 10 -> min(max(0, 2 - 1),  9) = 1
    //
    // D[i] = (1 << (code - 1)) + (u(bitcount) << (code - 1 - bitcount)):
    //   D[1] = 1024 + (u(1) = 0) << 9 = 1024
    //   D[2] =  512 + (u(1) = 1) << 8 =  768
    // total = 1792, so D[0] = 4096 - 1792 = 2304.
    let mut w = BitWriter::new();
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.bit(false);
    w.u(1, 1); // shift = 2
    w.bit(false); // U8() = 0 -> alphabet_size 3
    w.code(&[1, 0, 0, 0, 0, 0, 0]); // logcounts[0] = 12
    w.code(&[1, 0, 0, 0, 0, 1]); // logcounts[1] = 11
    w.code(&[0, 0, 0]); // logcounts[2] = 10
    w.u(0, 1); // refinement for i = 1
    w.u(1, 1); // refinement for i = 2

    let data = w.finish();
    let mut r = BitReader::new(&data);
    let mut g = guard();
    let dist = AnsDistribution::read(&mut r, 5, &mut g).expect("distribution");
    assert_eq!(
        r.total_bits_read() as usize,
        w.bit_len(),
        "omit_pos must not consume refinement bits"
    );

    assert_eq!(dist.probability(0), 2304);
    assert_eq!(dist.probability(1), 1024);
    assert_eq!(dist.probability(2), 768);
    assert_alias_mapping_is_a_bijection(&dist, &[2304, 1024, 768]);
}

/// C.2.5 run-length form: logcount 13 repeats the preceding probability.
#[test]
fn general_form_with_a_run_length_repeat() {
    // Bool() = 0, Bool() = 0, len = 0 -> shift = 0
    // U8() = 5 -> alphabet_size = 8
    //   U8() for 5: bit 1, n = u(3) = 2, u(2) = 1 -> 1 + (1 << 2) = 5
    //
    // logcounts:
    //   i = 0: 10 -> D = 512
    //   i = 1: 13 -> run length. U8() = 0 -> same[1] = 5, and the clause
    //          advances by rle + 3 then the loop increments, so the next
    //          logcount is read at i = 5.
    //   i = 5: 11 -> the new maximum, so omit_pos = 5
    //   i = 6: 10
    //   i = 7: 10
    //
    // Second pass: same[1] = 5 sets numsame = 4 and prev = D[0] = 512, so
    // positions 1..4 copy 512. Then D[5] is skipped (omit_pos), D[6] = 512
    // and D[7] = 512.
    //   total = 512 * 5 + 512 * 2 = 3584, so D[5] = 4096 - 3584 = 512.
    // Every entry ends up at 512, which sums to 4096 across 8 symbols.
    let mut w = BitWriter::new();
    w.bit(false);
    w.bit(false);
    w.bit(false); // len = 0 -> shift = 0
    w.bit(true); // U8() nonzero
    w.u(2, 3); // n = 2
    w.u(1, 2); // u(2) = 1 -> U8() = 5 -> alphabet_size = 8
    w.code(&[0, 0, 0]); // logcounts[0] = 10
    w.code(&[1, 0, 0, 0, 0, 0, 1]); // logcounts[1] = 13 (run length)
    w.bit(false); // U8() = 0 -> rle = 0
    w.code(&[1, 0, 0, 0, 0, 1]); // logcounts[5] = 11
    w.code(&[0, 0, 0]); // logcounts[6] = 10
    w.code(&[0, 0, 0]); // logcounts[7] = 10

    let data = w.finish();
    let mut r = BitReader::new(&data);
    let mut g = guard();
    let dist = AnsDistribution::read(&mut r, 5, &mut g).expect("distribution");
    assert_eq!(r.total_bits_read() as usize, w.bit_len());

    for symbol in 0..8u32 {
        assert_eq!(dist.probability(symbol), 512, "symbol {symbol}");
    }
    assert_alias_mapping_is_a_bijection(&dist, &[512; 8]);
}

/// C.2.5 two-symbol shortcut.
#[test]
fn two_symbol_shortcut_form() {
    // Bool() = 1, Bool() = 1 -> the two-symbol form
    // U8() = 2, U8() = 5, then u(12) = 1000
    //   U8() for 2: bit 1, n = u(3) = 1, u(1) = 0 -> 0 + 2 = 2
    //   U8() for 5: bit 1, n = u(3) = 2, u(2) = 1 -> 1 + 4 = 5
    // D[2] = 1000, D[5] = 4096 - 1000 = 3096, alphabet_size = 1 + 5 = 6
    let mut w = BitWriter::new();
    w.bit(true);
    w.bit(true);
    w.bit(true);
    w.u(1, 3);
    w.u(0, 1); // U8() = 2
    w.bit(true);
    w.u(2, 3);
    w.u(1, 2); // U8() = 5
    w.u(1000, 12);

    let data = w.finish();
    let mut r = BitReader::new(&data);
    let mut g = guard();
    let dist = AnsDistribution::read(&mut r, 5, &mut g).expect("distribution");
    assert_eq!(r.total_bits_read() as usize, w.bit_len());
    assert_eq!(dist.probability(2), 1000);
    assert_eq!(dist.probability(5), 3096);

    let mut expected = vec![0u32; 32];
    expected[2] = 1000;
    expected[5] = 3096;
    assert_alias_mapping_is_a_bijection(&dist, &expected);
}

/// C.2.5 flat form: the mass is split as evenly as the alphabet allows, with
/// the remainder handed to the lowest symbols.
#[test]
fn flat_form_distributes_the_remainder() {
    // Bool() = 0, Bool() = 1 -> flat form
    // U8() = 2 -> alphabet_size = 3
    //   4096 / 3 = 1365 remainder 1, so D[0] = 1366 and D[1] = D[2] = 1365.
    let mut w = BitWriter::new();
    w.bit(false);
    w.bit(true);
    w.bit(true);
    w.u(1, 3);
    w.u(0, 1); // U8() = 2

    let data = w.finish();
    let mut r = BitReader::new(&data);
    let mut g = guard();
    let dist = AnsDistribution::read(&mut r, 5, &mut g).expect("distribution");
    assert_eq!(dist.probability(0), 1366);
    assert_eq!(dist.probability(1), 1365);
    assert_eq!(dist.probability(2), 1365);
    assert_eq!(
        dist.probability(0) + dist.probability(1) + dist.probability(2),
        PROBABILITY_TOTAL
    );
    assert_alias_mapping_is_a_bijection(&dist, &[1366, 1365, 1365]);
}

/// A full ANS bundle, opened and read through the public facade, including the
/// C.3.2 terminal-state check.
#[test]
fn ans_bundle_end_to_end() {
    // C.2.1 bundle for a single context:
    //   Bool() = 0        lz77.enabled = false        (Table C.1)
    //   -                 num_dist == 1, so C.2.2 reads nothing
    //   Bool() = 0        use_prefix_code = false
    //   u(2)   = 0        log_alphabet_size = 5 + 0 = 5
    //   u(3)   = 5        configs[0].split_exponent; 5 == log_alphabet_size,
    //                     so msb_in_token and lsb_in_token are absent and
    //                     split = 32
    //   distribution (C.2.5 one-symbol shortcut):
    //     Bool() = 1, Bool() = 0, U8() = 1  ->  D[1] = 4096
    //     U8() for 1: bit 1, n = u(3) = 0, u(0) -> 0 + (1 << 0) = 1
    //   u(32)             the initial ANS state (C.3.2)
    //
    // With a single symbol holding the whole mass, AliasMapping(x) = (1, x)
    // and the state update is 4096 * (state >> 12) + (state & 0xFFF), which is
    // the identity. So the state never changes and never renormalizes: seeding
    // it with the terminal value 0x130000 means every symbol is free and the
    // stream is already in its end state.
    //
    // Token 1 is below split = 32, so each read_uint yields the value 1.
    let mut w = BitWriter::new();
    w.bit(false);
    w.bit(false);
    w.u(0, 2);
    w.u(5, 3);
    w.bit(true);
    w.bit(false);
    w.bit(true);
    w.u(0, 3);
    w.u(0x0013_0000, 32);

    let data = w.finish();
    let mut r = BitReader::new(&data);
    let mut g = guard();
    let mut dec = SymbolDecoder::open(&mut r, 1, &mut g).expect("bundle");
    assert!(!dec.uses_prefix_code());
    assert_eq!(r.total_bits_read() as usize, w.bit_len());

    let after_header = r.total_bits_read();
    for _ in 0..10 {
        assert_eq!(dec.read_uint(&mut r, 0).expect("symbol"), 1);
    }
    assert_eq!(
        r.total_bits_read(),
        after_header,
        "a sole symbol of full mass consumes no bits"
    );
    dec.finish().expect("C.3.2 terminal state");
}

/// The C.3.2 terminal-state check must actually reject a stream that ends in
/// the wrong state, otherwise it proves nothing.
#[test]
fn wrong_terminal_state_is_rejected() {
    let mut w = BitWriter::new();
    w.bit(false);
    w.bit(false);
    w.u(0, 2);
    w.u(5, 3);
    w.bit(true);
    w.bit(false);
    w.bit(true);
    w.u(0, 3);
    w.u(0x0013_0001, 32); // one off the terminal value

    let data = w.finish();
    let mut r = BitReader::new(&data);
    let mut g = guard();
    let dec = SymbolDecoder::open(&mut r, 1, &mut g).expect("bundle");
    assert!(dec.finish().is_err());
}

/// A distribution whose counts do not reach 4096 must be rejected at
/// construction, not silently normalized.
#[test]
fn under_full_distribution_is_rejected() {
    // Same shape as the shift = 0 vector but with logcounts 10, 10, 10:
    // omit_pos = 0, total = 512 + 512 = 1024, so D[0] = 3072. That is still a
    // valid distribution, so instead drive the total past the budget: with
    // logcounts 12, 12, 12 the second pass totals 2048 + 2048 = 4096 and
    // D[omit_pos] would have to be 0 -- still valid. Use four entries of 12:
    // total = 3 * 2048 = 6144 > 4096, which the clause forbids.
    let mut w = BitWriter::new();
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.u(0, 3);
    w.u(0, 0); // U8(): n = 0 -> value 1 -> alphabet_size = 4
    for _ in 0..4 {
        w.code(&[1, 0, 0, 0, 0, 0, 0]); // logcount 12 each
    }
    let data = w.finish();
    let mut r = BitReader::new(&data);
    let mut g = guard();
    assert!(AnsDistribution::read(&mut r, 5, &mut g).is_err());
}
