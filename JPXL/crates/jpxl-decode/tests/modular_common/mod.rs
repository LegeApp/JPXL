//! Test helpers for hand-building modular sub-bitstreams (18181-1 Annex H).
//!
//! Slice 5 owns `src/modular/**` and its own test files only, so this is a
//! local bit writer rather than a change to `src/testsupport.rs`.
//!
//! The interesting part is [`write_prefix_bundle`]: it emits the smallest
//! Annex C distribution bundle that can still carry more than one symbol, so a
//! test can spell out an entropy-coded stream field by field and state what
//! each field is for. Everything it writes is derived from clauses C.2.1
//! through C.2.4 and RFC 7932 section 3.4; the derivations are in the comments
//! beside each write.

// Test code: an out-of-range index is a failed assertion rather than an attack
// surface, and the casts are on values these tests chose themselves.
#![allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]
// Each integration test file uses a different subset of these helpers.
#![allow(dead_code)]

/// Accumulates bits in stream order and packs them LSB-first within each byte,
/// which is what `jpxl_bitstream::BitReader` consumes.
#[derive(Debug, Default)]
pub struct BitWriter {
    bits: Vec<bool>,
}

impl BitWriter {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Writes `u(n)`: `n` bits of `value`, least-significant bit first.
    pub fn u(&mut self, value: u32, n: u32) {
        for i in 0..n {
            self.bits.push((value >> i) & 1 == 1);
        }
    }

    /// Writes a single `Bool()`.
    pub fn bit(&mut self, value: bool) {
        self.bits.push(value);
    }

    /// Writes `n` bits of `value` most-significant bit first, which is the
    /// order `PrefixCode::decode` consumes a canonical code in.
    pub fn code_msb_first(&mut self, value: u32, n: u32) {
        for i in (0..n).rev() {
            self.bits.push((value >> i) & 1 == 1);
        }
    }

    /// Writes a `U32()` selector plus payload for the distribution at `index`.
    pub fn u32_dist(&mut self, index: u32, payload: u32, payload_bits: u32) {
        self.u(index, 2);
        self.u(payload, payload_bits);
    }

    #[must_use]
    pub fn bit_len(&self) -> usize {
        self.bits.len()
    }

    /// Packs to bytes, zero-padding the final partial byte.
    #[must_use]
    pub fn finish(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.bits.len().div_ceil(8)];
        for (i, &bit) in self.bits.iter().enumerate() {
            if bit {
                out[i / 8] |= 1 << (i % 8);
            }
        }
        out
    }
}

/// `ceil(log2(v + 1))`: the width of the field that carries `v`.
#[must_use]
pub fn bit_width(v: u32) -> u32 {
    32 - v.leading_zeros()
}

/// C.2.1: prefix-coded bundles fix `log_alphabet_size` at 15.
pub const PREFIX_LOG_ALPHABET_SIZE: u32 = 15;

/// A prefix-coded distribution bundle small enough to hand-derive.
///
/// * every one of `num_dist` contexts is clustered into a single distribution,
/// * that distribution's hybrid-uint config has `msb_in_token == lsb_in_token
///   == 0`, so tokens below `1 << split_exponent` are literal values,
/// * the code is RFC 7932's *simple* form over symbols `0..alphabet_size`.
///
/// `alphabet_size` must be 1, 2 or 4: those are the sizes whose simple-code
/// length patterns are `[]`, `[1, 1]` and `[2, 2, 2, 2]`, i.e. fixed-width, so
/// [`write_token`] can emit a symbol without building a code table.
pub fn write_prefix_bundle(
    w: &mut BitWriter,
    num_dist: usize,
    alphabet_size: usize,
    split_exponent: u32,
) {
    assert!(
        matches!(alphabet_size, 1 | 2 | 4),
        "only fixed-width simple codes are supported here"
    );

    // Table C.1: lz77.enabled.
    w.bit(false);

    // C.2.2: a single context is its own cluster and nothing is coded. For more
    // than one, the "simple" form spends `nbits` per context; `nbits == 0` maps
    // every context to cluster 0, which is the whole point of this helper.
    if num_dist > 1 {
        w.bit(true); // simple clustering
        w.u(0, 2); // nbits = 0
    }

    // C.2.1: use_prefix_code.
    w.bit(true);

    // C.2.3: one HybridUintConfig for the single cluster.
    w.u(split_exponent, bit_width(PREFIX_LOG_ALPHABET_SIZE));
    if split_exponent != PREFIX_LOG_ALPHABET_SIZE {
        w.u(0, bit_width(split_exponent)); // msb_in_token = 0
        w.u(0, bit_width(split_exponent)); // lsb_in_token = 0, remaining == split_exponent
    }

    // C.2.1: the alphabet size, as `1 + (1 << n) + u(n)`.
    if alphabet_size == 1 {
        w.bit(false);
    } else {
        w.bit(true);
        let mut n = 0u32;
        while 1 + (1usize << n) > alphabet_size {
            n += 1;
        }
        while 1 + (1usize << n) + ((1usize << n) - 1) < alphabet_size {
            n += 1;
        }
        let extra = alphabet_size - 1 - (1usize << n);
        assert!(extra < (1usize << n) || n == 0);
        w.u(n, 4);
        w.u(extra as u32, n);
    }

    // C.2.4 / RFC 7932 3.4: the simple prefix code.
    if alphabet_size > 1 {
        w.u(1, 2); // selector 1 = simple
        let nsym = alphabet_size as u32;
        w.u(nsym - 1, 2);
        let alphabet_bits = bit_width(alphabet_size as u32 - 1);
        for symbol in 0..nsym {
            w.u(symbol, alphabet_bits);
        }
        if nsym == 4 {
            // false selects the balanced pattern [2, 2, 2, 2].
            w.bit(false);
        }
    }
}

/// Emits one symbol of a bundle written by [`write_prefix_bundle`].
///
/// The canonical assignment for `[1, 1]` is `symbol == bit`, and for
/// `[2, 2, 2, 2]` it is `symbol == the two bits read MSB first`. A one-symbol
/// alphabet costs no bits at all.
pub fn write_token(w: &mut BitWriter, alphabet_size: usize, symbol: u32) {
    match alphabet_size {
        1 => assert_eq!(symbol, 0, "a one-symbol alphabet can only carry 0"),
        2 => {
            assert!(symbol < 2);
            w.code_msb_first(symbol, 1);
        }
        4 => {
            assert!(symbol < 4);
            w.code_msb_first(symbol, 2);
        }
        other => panic!("unsupported alphabet size {other}"),
    }
}

/// Writes the `ModularHeader` of Table H.1 with no transforms.
pub fn write_modular_header_no_transforms(w: &mut BitWriter, use_global_tree: bool) {
    w.bit(use_global_tree); // use_global_tree
    w.bit(true); // wp_params: default_wp
    w.u32_dist(0, 0, 0); // nb_transforms: U32 distribution 0 = the constant 0
}

/// `UnpackSigned` inverse: the token that decodes to `v`.
#[must_use]
pub fn pack_signed(v: i32) -> u32 {
    if v >= 0 {
        (i64::from(v) * 2) as u32
    } else {
        (-i64::from(v) * 2 - 1) as u32
    }
}
