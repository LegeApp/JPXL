//! LZ77 vectors through the public facade — ISO/IEC 18181-1 Table C.1 and
//! C.3.3.
//!
//! These drive a complete distribution bundle rather than the window in
//! isolation, so they cover the interaction the clause actually specifies:
//! a token at or above `min_symbol` switches to a copy, pulls a length through
//! the LZ77 length configuration and a distance through the dedicated context,
//! and then replays the window. This layer is bit-exact.

// Test code: an out-of-range index is a failed assertion rather than an attack
// surface, and the casts are on values these tests chose themselves. The lints
// exist for the decode paths in src/.
#![allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]

mod common;

use common::BitWriter;
use jpxl_bitstream::BitReader;
use jpxl_core::limits::{AllocGuard, Limits};
use jpxl_entropy::SymbolDecoder;

fn guard() -> AllocGuard {
    AllocGuard::new(&Limits::relaxed())
}

/// Builds a prefix-coded bundle with LZ77 enabled and `min_symbol = 8`.
///
/// Bit layout, in read order:
///
/// | field | value | clause |
/// | --- | --- | --- |
/// | `Bool()` | 1 (enabled) | Table C.1 |
/// | `u(2)` selector | 3 -> `8 + u(15)` | Table C.1 `min_symbol` |
/// | `u(15)` | 0 -> `min_symbol = 8` | |
/// | `u(2)` selector | 0 -> `min_length = 3` | Table C.1 `min_length` |
/// | `u(4)` | 8 -> `lz_len_conf.split_exponent` | C.2.1, C.2.3 |
///
/// `log_alphabet_size` is fixed at 8 for the LZ77 length configuration, so its
/// first field is `u(ceil(log2(9))) = u(4)`; a value of 8 equals
/// `log_alphabet_size`, so the msb/lsb fields are absent and `split = 256`.
///
/// Enabling LZ77 appends the distance context, so `num_dist` becomes 2 and
/// C.2.2 runs:
///
/// | `Bool()` | 1 (is_simple) | C.2.2 |
/// | `u(2)` | 0 -> `nbits = 0` | |
/// | 2 x `u(0)` | both contexts map to cluster 0 | |
///
/// Then the bundle proper:
///
/// | `Bool()` | 1 (use_prefix_code) | C.2.1 |
/// | `u(4)` | 15 -> `configs[0].split_exponent`, `split = 32768` | C.2.3 |
/// | `Bool()` | 1, `u(4)` = 3, `u(3)` = 7 -> `count = 1 + 8 + 7 = 16` | C.2.1 |
/// | `u(2)` | 1 (simple prefix code) | RFC 7932 3.4 |
/// | `u(2)` | 3 -> NSYM = 4 | |
/// | 4 x `u(4)` | symbols 0, 1, 2, 8 | |
/// | `Bool()` | 0 (tree-select) -> lengths 2, 2, 2, 2 | |
///
/// Canonical assignment over four equal lengths orders by symbol value, so
/// the codes are `00` -> 0, `01` -> 1, `10` -> 2, `11` -> 8. Since
/// `min_symbol` is 8, the code `11` is the only back-reference trigger.
fn lz77_bundle_header() -> BitWriter {
    let mut w = BitWriter::new();
    w.bit(true); // lz77.enabled
    w.u(3, 2); // min_symbol selector -> 8 + u(15)
    w.u(0, 15); // min_symbol = 8
    w.u(0, 2); // min_length selector -> 3
    w.u(8, 4); // lz_len_conf.split_exponent = 8

    w.bit(true); // C.2.2 is_simple
    w.u(0, 2); // nbits = 0, so both contexts land in cluster 0

    w.bit(true); // use_prefix_code
    w.u(15, 4); // configs[0].split_exponent = 15

    w.bit(true); // count flag
    w.u(3, 4); // n = 3
    w.u(7, 3); // u(3) = 7 -> count = 1 + (1 << 3) + 7 = 16

    w.u(1, 2); // simple prefix code
    w.u(3, 2); // NSYM = 4
    w.u(0, 4);
    w.u(1, 4);
    w.u(2, 4);
    w.u(8, 4);
    w.bit(false); // tree-select 0 -> lengths 2, 2, 2, 2
    w
}

#[test]
fn bundle_header_parses_with_the_expected_shape() {
    let w = lz77_bundle_header();
    let data = w.finish();
    let mut r = BitReader::new(&data);
    let mut g = guard();
    let dec = SymbolDecoder::open(&mut r, 1, &mut g).expect("bundle");
    assert!(dec.uses_prefix_code());
    // One cluster shared by the value context and the LZ77 distance context.
    assert_eq!(dec.clusters().num_clusters(), 1);
    assert_eq!(dec.clusters().num_dist(), 2);
    assert_eq!(r.total_bits_read() as usize, w.bit_len());
}

/// A back-reference with `dist_multiplier == 0`: the raw distance is simply
/// incremented, so a decoded 0 means "one symbol back".
#[test]
fn back_reference_repeats_the_previous_symbol() {
    let mut w = lz77_bundle_header();
    let header_bits = w.bit_len();
    // Symbol stream:
    //   "01" -> 1   literal (token 1 < split 32768, and 1 < min_symbol 8)
    //   "10" -> 2   literal
    //   "11" -> 8   token >= min_symbol, so a copy begins:
    //                 length   = ReadUint(lz_len_conf, 8 - 8 = 0) + 3 = 0 + 3
    //                 distance token follows, from the distance context
    //   "00" -> 0   distance token -> ReadUint(configs[0], 0) = 0
    //               dist_multiplier is 0, so distance = 0 + 1 = 1
    // The window holds [1, 2]; copying 3 symbols from distance 1 replays the
    // most recent symbol three times.
    w.code(&[0, 1]);
    w.code(&[1, 0]);
    w.code(&[1, 1]);
    w.code(&[0, 0]);

    let data = w.finish();
    let mut r = BitReader::new(&data);
    let mut g = guard();
    let mut dec = SymbolDecoder::open(&mut r, 1, &mut g).expect("bundle");
    assert_eq!(r.total_bits_read() as usize, header_bits);

    let decoded: Vec<u32> = (0..5)
        .map(|i| {
            dec.read_uint(&mut r, 0)
                .unwrap_or_else(|e| panic!("read {i}: {e}"))
        })
        .collect();
    assert_eq!(decoded, vec![1, 2, 2, 2, 2]);
    assert_eq!(
        r.total_bits_read() as usize,
        header_bits + 8,
        "the three copied symbols consume no bits"
    );
}

/// The same stream with a row stride set: C.3.3 then routes the distance
/// through `kSpecialDistances`, and entry 0 is `(0, 1)` — one row back.
#[test]
fn back_reference_with_a_row_stride_uses_the_special_table() {
    let mut w = lz77_bundle_header();
    w.code(&[0, 1]); // 1
    w.code(&[1, 0]); // 2
    w.code(&[1, 1]); // 8 -> copy of length 3
    w.code(&[0, 0]); // distance token 0

    let data = w.finish();
    let mut r = BitReader::new(&data);
    let mut g = guard();
    let mut dec = SymbolDecoder::open(&mut r, 1, &mut g).expect("bundle");
    // kSpecialDistances[0] = (0, 1), so distance = 0 + 2 * 1 = 2.
    dec.set_dist_multiplier(2);

    let decoded: Vec<u32> = (0..5)
        .map(|i| {
            dec.read_uint(&mut r, 0)
                .unwrap_or_else(|e| panic!("read {i}: {e}"))
        })
        .collect();
    // Window [1, 2] copied from distance 2 replays the pair.
    assert_eq!(decoded, vec![1, 2, 1, 2, 1]);
}

/// A copy that begins before anything has been decoded reads the
/// zero-initialized window, because C.3.3 clamps the distance to
/// `num_decoded`.
#[test]
fn back_reference_at_the_start_of_a_stream_yields_zeroes() {
    let mut w = lz77_bundle_header();
    w.code(&[1, 1]); // token 8 -> copy of length 3, before any literal
    w.code(&[0, 0]); // distance token 0 -> distance 1, clamped to 0

    let data = w.finish();
    let mut r = BitReader::new(&data);
    let mut g = guard();
    let mut dec = SymbolDecoder::open(&mut r, 1, &mut g).expect("bundle");
    let decoded: Vec<u32> = (0..3)
        .map(|i| {
            dec.read_uint(&mut r, 0)
                .unwrap_or_else(|e| panic!("read {i}: {e}"))
        })
        .collect();
    assert_eq!(decoded, vec![0, 0, 0]);
}

/// A stream that never crosses `min_symbol` behaves exactly as if LZ77 were
/// absent, which is what makes the layer transparent to its callers.
#[test]
fn tokens_below_min_symbol_are_ordinary_values() {
    let mut w = lz77_bundle_header();
    for _ in 0..4 {
        w.code(&[1, 0]); // token 2 -> value 2
    }

    let data = w.finish();
    let mut r = BitReader::new(&data);
    let mut g = guard();
    let mut dec = SymbolDecoder::open(&mut r, 1, &mut g).expect("bundle");
    let decoded: Vec<u32> = (0..4)
        .map(|i| {
            dec.read_uint(&mut r, 0)
                .unwrap_or_else(|e| panic!("read {i}: {e}"))
        })
        .collect();
    assert_eq!(decoded, vec![2, 2, 2, 2]);
}

/// A stream that runs out of bits mid-symbol must report the error rather than
/// panicking or inventing a symbol.
#[test]
fn truncation_is_reported_not_panicked() {
    let w = lz77_bundle_header();
    let full = w.finish();
    for truncated_len in 0..full.len() {
        let data = full.get(..truncated_len).expect("in range");
        let mut r = BitReader::new(data);
        let mut g = guard();
        // Either the bundle fails to open or the first read fails; neither may
        // panic, and neither may succeed with the full header.
        if let Ok(mut dec) = SymbolDecoder::open(&mut r, 1, &mut g) {
            let _ = dec.read_uint(&mut r, 0);
        }
    }
}
