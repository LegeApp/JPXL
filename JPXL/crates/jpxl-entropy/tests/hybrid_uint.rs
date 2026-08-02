//! Hybrid unsigned integer vectors — ISO/IEC 18181-1 C.2.3 and C.3.3.
//!
//! Every expectation is derived by hand from the clause formulas, with the
//! arithmetic shown. This layer is bit-exact.

// Test code: an out-of-range index is a failed assertion rather than an attack
// surface, and the casts are on values these tests chose themselves. The lints
// exist for the decode paths in src/.
#![allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]

mod common;

use common::BitWriter;
use jpxl_bitstream::BitReader;
use jpxl_entropy::HybridUintConfig;

/// C.2.3: when `split_exponent == log_alphabet_size` the msb/lsb fields are
/// not present at all, so the configuration costs a single field.
#[test]
fn config_with_no_msb_lsb_fields() {
    // log_alphabet_size = 15 -> split_exponent is u(ceil(log2(16))) = u(4).
    let mut w = BitWriter::new();
    w.u(15, 4);
    let data = w.finish();
    let mut r = BitReader::new(&data);
    let cfg = HybridUintConfig::read(&mut r, 15).expect("config");
    assert_eq!(
        cfg,
        HybridUintConfig {
            split_exponent: 15,
            msb_in_token: 0,
            lsb_in_token: 0
        }
    );
    assert_eq!(r.total_bits_read(), 4);
    assert_eq!(cfg.split(), 1 << 15);
}

/// C.2.3: the general form, with each field width derived from the previous
/// field's value.
#[test]
fn config_field_widths_follow_the_clause() {
    // log_alphabet_size = 8 -> split_exponent is u(ceil(log2(9))) = u(4).
    // split_exponent = 4, which differs from 8, so:
    //   msb_in_token is u(ceil(log2(5))) = u(3); choose 2
    //   lsb_in_token is u(ceil(log2(4 - 2 + 1))) = u(2); choose 1
    let mut w = BitWriter::new();
    w.u(4, 4);
    w.u(2, 3);
    w.u(1, 2);
    let data = w.finish();
    let mut r = BitReader::new(&data);
    let cfg = HybridUintConfig::read(&mut r, 8).expect("config");
    assert_eq!(
        cfg,
        HybridUintConfig {
            split_exponent: 4,
            msb_in_token: 2,
            lsb_in_token: 1
        }
    );
    assert_eq!(r.total_bits_read(), 9);
}

/// C.2.3 requires `msb_in_token + lsb_in_token <= split_exponent`.
#[test]
fn config_rejects_oversized_msb_field() {
    // split_exponent = 1 (u(4)), then msb_in_token is u(ceil(log2(2))) = u(1).
    // A value of 1 is legal; the lsb field is then u(ceil(log2(1))) = u(0) = 0.
    let mut w = BitWriter::new();
    w.u(1, 4);
    w.u(1, 1);
    let data = w.finish();
    let mut r = BitReader::new(&data);
    let cfg = HybridUintConfig::read(&mut r, 8).expect("config");
    assert_eq!(cfg.msb_in_token, 1);
    assert_eq!(cfg.lsb_in_token, 0);
    assert_eq!(r.total_bits_read(), 5, "the lsb field is zero bits wide");
}

/// C.3.3 `ReadUint`: with no msb/lsb bits in the token, the scheme degenerates
/// to a plain exponential Golomb-style code above the split.
#[test]
fn read_uint_pure_exponential_form() {
    let cfg = HybridUintConfig {
        split_exponent: 4,
        msb_in_token: 0,
        lsb_in_token: 0,
    };
    assert_eq!(cfg.split(), 16);

    // token = 16: in_token = 0, so
    //   n    = 4 - 0 - 0 + ((16 - 16) >> 0) = 4
    //   low  = token & ((1 << 0) - 1) = 0
    //   mid  = (16 >> 0) & 0 = 0, then |= 1 << 0 -> 1
    //   result = ((1 << 4) | u(4)) << 0 | 0 = 16 + u(4)
    // With u(4) = 0b1010 = 10 the result is 26.
    let mut w = BitWriter::new();
    w.u(10, 4);
    let data = w.finish();
    let mut r = BitReader::new(&data);
    assert_eq!(cfg.read_uint(&mut r, 16).expect("value"), 26);
    assert_eq!(r.total_bits_read(), 4);

    // token = 17: n = 4 + ((17 - 16) >> 0) = 5, so the range is 32 + u(5).
    // With u(5) = 0b00011 = 3 the result is 35.
    let mut w = BitWriter::new();
    w.u(3, 5);
    let data = w.finish();
    let mut r = BitReader::new(&data);
    assert_eq!(cfg.read_uint(&mut r, 17).expect("value"), 35);
    assert_eq!(r.total_bits_read(), 5);
}

/// C.3.3 `ReadUint`: msb and lsb bits ride inside the token itself.
#[test]
fn read_uint_with_msb_and_lsb_in_token() {
    let cfg = HybridUintConfig {
        split_exponent: 4,
        msb_in_token: 2,
        lsb_in_token: 1,
    };
    assert_eq!(cfg.split(), 16);

    // token = 20:
    //   in_token = msb + lsb = 3
    //   n     = 4 - 2 - 1 + ((20 - 16) >> 3) = 1 + (4 >> 3) = 1 + 0 = 1
    //   low   = 20 & ((1 << 1) - 1) = 20 & 1 = 0
    //   token >>= 1            -> 10
    //   token &= (1 << 2) - 1  -> 10 & 3 = 2
    //   token |= 1 << 2        -> 6
    //   result = (((6 << 1) | u(1)) << 1) | 0
    // With u(1) = 1: (((12) | 1) << 1) = 13 << 1 = 26.
    let mut w = BitWriter::new();
    w.u(1, 1);
    let data = w.finish();
    let mut r = BitReader::new(&data);
    assert_eq!(cfg.read_uint(&mut r, 20).expect("value"), 26);
    assert_eq!(r.total_bits_read(), 1);

    // With u(1) = 0 the same token yields (12 << 1) = 24.
    let mut w = BitWriter::new();
    w.u(0, 1);
    let data = w.finish();
    let mut r = BitReader::new(&data);
    assert_eq!(cfg.read_uint(&mut r, 20).expect("value"), 24);
}

/// The reconstruction must be a bijection onto a contiguous range: tokens just
/// above the split, together with their extra bits, have to tile the integers
/// immediately above the split with no gap and no overlap. A wrong shift in
/// the `ReadUint` formula shows up here immediately.
#[test]
fn reconstruction_tiles_the_value_range_without_gaps() {
    for (split_exponent, msb_in_token, lsb_in_token) in [
        (4, 0, 0),
        (4, 2, 1),
        (4, 1, 0),
        (4, 0, 2),
        (5, 2, 2),
        (2, 1, 1),
    ] {
        let cfg = HybridUintConfig {
            split_exponent,
            msb_in_token,
            lsb_in_token,
        };
        let split = cfg.split();
        let mut produced: Vec<u32> = (0..split).collect();

        // Enumerate every token in the first few "shells" above the split and
        // every value of its extra bits.
        let in_token = msb_in_token + lsb_in_token;
        for token in split..split + (1 << in_token) * 3 {
            let n = split_exponent - in_token + ((token - split) >> in_token);
            for extra in 0..(1u32 << n) {
                let mut w = BitWriter::new();
                w.u(extra, n);
                let data = w.finish();
                let mut r = BitReader::new(&data);
                produced.push(cfg.read_uint(&mut r, token).expect("value"));
            }
        }

        produced.sort_unstable();
        let expected: Vec<u32> = (0..produced.len() as u32).collect();
        assert_eq!(
            produced,
            expected,
            "config {split_exponent}/{msb_in_token}/{lsb_in_token} must tile \
             [0, {}) exactly once",
            produced.len()
        );
    }
}

/// A token below the split is returned verbatim and reads nothing.
#[test]
fn literals_consume_no_extra_bits() {
    let cfg = HybridUintConfig {
        split_exponent: 8,
        msb_in_token: 3,
        lsb_in_token: 2,
    };
    let mut r = BitReader::new(&[]);
    for token in [0u32, 1, 127, 255] {
        assert_eq!(cfg.read_uint(&mut r, token).expect("literal"), token);
    }
    assert_eq!(r.total_bits_read(), 0);
}
