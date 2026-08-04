//! Round-trip property tests for the entropy encoder — ISO/IEC 18181-1
//! Annex C, write side.
//!
//! The contract of the encoder slice is exactly one sentence: **every stream
//! the encoder emits, this crate's decoder decodes bit-exactly, with the C.3.2
//! terminal state holding at the end and no bit left over.** That is what
//! [`check`] asserts, and every test here is a way of feeding it a stream
//! shape that could break it.
//!
//! The generators are seeded and deterministic: a failure reproduces from its
//! printed case description alone.
//!
//! The two sides share tables but no control flow (see `encode`'s module
//! documentation), so agreement here is evidence, not tautology.

// Test code: an out-of-range index is a failed assertion rather than an attack
// surface, and the casts are on values these tests chose themselves. The lints
// exist for the decode paths in src/.
#![allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]

use jpxl_bitstream::{BitReader, BitWriter};
use jpxl_core::limits::{AllocGuard, Limits};
use jpxl_entropy::encode::{
    CodingMode, ContextMap, ContextMapForm, EncoderPlan, EntropyTables, SymbolEncoder, TokenCensus,
};
use jpxl_entropy::{HybridUintConfig, SymbolDecoder};

/// A deterministic 32-bit generator, so every case reproduces from its seed.
struct Rng(u32);

impl Rng {
    fn new(seed: u32) -> Self {
        Self(seed | 1)
    }

    fn next(&mut self) -> u32 {
        // xorshift32: full period over the nonzero states, no dependencies.
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    fn below(&mut self, bound: u32) -> u32 {
        if bound == 0 { 0 } else { self.next() % bound }
    }
}

/// Encodes `values` under `plan`, decodes the result, and asserts equality,
/// the C.3.2 terminal state, and that the stream is consumed to the bit.
///
/// Returns the encoded size in bits, so density-sensitive tests can compare.
fn check(plan: &EncoderPlan, values: &[(usize, u32)], case: &str) -> u64 {
    let num_contexts = plan.context_map.num_dist();

    let mut census = TokenCensus::new(num_contexts).unwrap_or_else(|e| panic!("{case}: {e}"));
    for &(ctx, value) in values {
        census
            .record(ctx, value)
            .unwrap_or_else(|e| panic!("{case}: census {e}"));
    }
    let tables =
        EntropyTables::build(plan, &census).unwrap_or_else(|e| panic!("{case}: tables {e}"));

    let mut w = BitWriter::new();
    tables
        .write_bundle(&mut w)
        .unwrap_or_else(|e| panic!("{case}: bundle {e}"));
    let mut encoder = SymbolEncoder::new(&tables);
    for &(ctx, value) in values {
        encoder
            .push_uint(ctx, value)
            .unwrap_or_else(|e| panic!("{case}: push {value} in {ctx}: {e}"));
    }
    encoder
        .write_stream(&mut w)
        .unwrap_or_else(|e| panic!("{case}: stream {e}"));
    let bits = w.bit_len();
    let bytes = w.into_bytes();

    let mut guard = AllocGuard::new(&Limits::relaxed());
    let mut r = BitReader::new(&bytes);
    let mut decoder = SymbolDecoder::open(&mut r, num_contexts, &mut guard)
        .unwrap_or_else(|e| panic!("{case}: the bundle must open: {e}"));
    assert_eq!(
        decoder.uses_prefix_code(),
        matches!(plan.mode, CodingMode::Prefix),
        "{case}: backend flag"
    );
    for (i, &(ctx, value)) in values.iter().enumerate() {
        let got = decoder
            .read_uint(&mut r, ctx)
            .unwrap_or_else(|e| panic!("{case}: value {i}: {e}"));
        assert_eq!(got, value, "{case}: value {i} in context {ctx}");
    }
    decoder
        .finish()
        .unwrap_or_else(|e| panic!("{case}: C.3.2 terminal state: {e}"));
    assert_eq!(
        r.total_bits_read(),
        bits,
        "{case}: the decoder must consume exactly the emitted bits"
    );
    bits
}

/// Every hybrid-uint shape worth sweeping, in `(split_exponent, msb, lsb)`.
fn configurations() -> Vec<HybridUintConfig> {
    let mut out = Vec::new();
    for split_exponent in [0u32, 1, 3, 4, 6, 8] {
        for msb in 0..=split_exponent.min(3) {
            for lsb in 0..=(split_exponent - msb).min(3) {
                if let Ok(config) = HybridUintConfig::new(split_exponent, msb, lsb) {
                    out.push(config);
                }
            }
        }
    }
    out
}

/// Value generators covering the distributions that break different things.
fn values_for(shape: usize, rng: &mut Rng, count: usize, num_contexts: usize) -> Vec<(usize, u32)> {
    (0..count)
        .map(|i| {
            let ctx = if num_contexts == 1 {
                0
            } else {
                (rng.below(num_contexts as u32)) as usize
            };
            let value = match shape {
                // Tiny alphabet: mostly literal tokens.
                0 => rng.below(4),
                // Small values: the common JPEG XL case.
                1 => rng.below(64),
                // Heavily skewed: one value dominates, so one symbol takes
                // nearly the whole probability mass.
                2 => {
                    if rng.below(100) < 97 {
                        0
                    } else {
                        rng.below(1000)
                    }
                }
                // Wide: forces long extra-bit runs.
                3 => rng.next() >> (rng.below(24) + 1),
                // Constant.
                4 => 42,
                // Alternating extremes.
                _ => {
                    if i % 2 == 0 {
                        0
                    } else {
                        u32::MAX / 4
                    }
                }
            };
            (ctx, value)
        })
        .collect()
}

/// The main matrix: backends × context counts × configurations × value shapes.
#[test]
fn the_encoder_round_trips_across_the_whole_matrix() {
    for (mode_index, mode) in [CodingMode::Ans, CodingMode::Prefix]
        .into_iter()
        .enumerate()
    {
        for num_contexts in [1usize, 2, 3, 8] {
            for (config_index, config) in configurations().into_iter().enumerate() {
                for shape in 0..6usize {
                    let seed = (mode_index * 1000 + num_contexts * 100 + config_index * 10 + shape)
                        as u32
                        + 1;
                    let mut rng = Rng::new(seed);
                    let values = values_for(shape, &mut rng, 400, num_contexts);
                    let plan = match EncoderPlan::identity(num_contexts, mode, config) {
                        Ok(plan) => plan,
                        Err(_) => continue,
                    };
                    let case = format!(
                        "mode {mode:?}, {num_contexts} contexts, config {config:?}, shape \
                         {shape}, seed {seed}"
                    );
                    // Two legal refusals rather than failures, both from
                    // C.2.1's `5 + u(2)` alphabet width, and neither reachable
                    // by prefix codes (which are fixed at 15):
                    //   * a token at or above 256 has no slot in the table;
                    //   * a configuration needs `split_exponent` bits, plus one
                    //     more when it has in-token bits, because C.2.3 stops
                    //     reading at `split_exponent == log_alphabet_size`.
                    let in_token = config.msb_in_token + config.lsb_in_token;
                    let needed = config.split_exponent + u32::from(in_token != 0);
                    if matches!(mode, CodingMode::Ans)
                        && (needed > 8
                            || values.iter().any(|&(_, v)| {
                                config.tokenize(v).map(|s| s.token).unwrap_or(u32::MAX) >= 256
                            }))
                    {
                        continue;
                    }
                    check(&plan, &values, &case);
                }
            }
        }
    }
}

/// Clusterings other than the identity, in every context-map form.
#[test]
fn clustered_streams_round_trip_in_every_context_map_form() {
    let config = HybridUintConfig::new(4, 1, 0).expect("legal");
    let clusterings: Vec<Vec<u8>> = vec![
        vec![0, 0],                          // two contexts, one cluster
        vec![0, 1],                          // identity
        vec![0, 1, 0, 1, 2],                 // interleaved
        (0..40u8).map(|i| i % 3).collect(),  // many contexts, few clusters
        (0..40u8).map(|i| i / 4).collect(),  // ten clusters: nested form only
        (0..64u8).map(|i| i % 13).collect(), // wider than the simple form
    ];
    for (index, clustering) in clusterings.into_iter().enumerate() {
        let num_contexts = clustering.len();
        let map = ContextMap::new(clustering.clone()).expect("dense");
        let num_clusters = map.num_clusters();
        for form in [
            ContextMapForm::Auto,
            ContextMapForm::Simple,
            ContextMapForm::Nested { use_mtf: false },
            ContextMapForm::Nested { use_mtf: true },
        ] {
            for mode in [CodingMode::Ans, CodingMode::Prefix] {
                let mut plan =
                    EncoderPlan::clustered(map.clone(), mode, vec![config; num_clusters])
                        .expect("plan");
                plan.context_map_form = form;
                // The simple form cannot express a cluster index above 7.
                if matches!(form, ContextMapForm::Simple) && clustering.iter().any(|&c| c > 7) {
                    continue;
                }
                let mut rng = Rng::new(index as u32 * 7 + 13);
                let values = values_for(1, &mut rng, 300, num_contexts);
                let case = format!("clustering {index}, form {form:?}, mode {mode:?}");
                check(&plan, &values, &case);
            }
        }
    }
}

/// Degenerate streams: no symbols at all, one symbol repeated, one context.
#[test]
fn degenerate_streams_round_trip() {
    let config = HybridUintConfig::new(4, 0, 0).expect("legal");
    for mode in [CodingMode::Ans, CodingMode::Prefix] {
        let plan = EncoderPlan::identity(1, mode, config).expect("plan");
        check(&plan, &[], &format!("empty, {mode:?}"));
        check(&plan, &[(0, 0)], &format!("one zero, {mode:?}"));
        let repeated: Vec<(usize, u32)> = vec![(0, 7); 5000];
        check(&plan, &repeated, &format!("repeated, {mode:?}"));

        // Two symbols only: the two-symbol histogram form and a one-bit prefix
        // code.
        let two: Vec<(usize, u32)> = (0..1000).map(|i| (0, u32::from(i % 2 == 0))).collect();
        check(&plan, &two, &format!("two symbols, {mode:?}"));

        // Every context empty except one.
        let plan = EncoderPlan::identity(4, mode, config).expect("plan");
        let sparse: Vec<(usize, u32)> = (0..100).map(|i| (2, i % 5)).collect();
        check(&plan, &sparse, &format!("one live context, {mode:?}"));
    }
}

/// A single-symbol ANS distribution never renormalizes, so the whole stream is
/// the 32-bit seed and nothing else. That is the sharpest possible check on the
/// terminal-state contract.
#[test]
fn a_constant_ans_stream_is_exactly_the_terminal_state() {
    let config = HybridUintConfig::new(4, 0, 0).expect("legal");
    let plan = EncoderPlan::identity(1, CodingMode::Ans, config).expect("plan");
    let values: Vec<(usize, u32)> = vec![(0, 3); 1000];

    let mut census = TokenCensus::new(1).expect("census");
    for &(ctx, value) in &values {
        census.record(ctx, value).expect("records");
    }
    let tables = EntropyTables::build(&plan, &census).expect("tables");
    let mut bundle = BitWriter::new();
    tables.write_bundle(&mut bundle).expect("bundle");
    let bundle_bits = bundle.bit_len();

    let mut w = BitWriter::new();
    tables.write_bundle(&mut w).expect("bundle");
    let mut encoder = SymbolEncoder::new(&tables);
    for &(ctx, value) in &values {
        encoder.push_uint(ctx, value).expect("records");
    }
    encoder.write_stream(&mut w).expect("stream");
    assert_eq!(
        w.bit_len(),
        bundle_bits + 32,
        "1000 copies of one symbol must cost exactly the ANS seed"
    );
    check(&plan, &values, "constant ANS stream");
}

/// Corrupting the stream must be caught, either as a wrong value or by the
/// C.3.2 terminal-state check. This is what proves the terminal state is a real
/// assertion and not an accident of construction.
#[test]
fn a_corrupted_ans_stream_fails_the_terminal_state_check() {
    let config = HybridUintConfig::new(4, 1, 1).expect("legal");
    let plan = EncoderPlan::identity(2, CodingMode::Ans, config).expect("plan");
    let mut rng = Rng::new(4242);
    let values = values_for(1, &mut rng, 200, 2);

    let mut census = TokenCensus::new(2).expect("census");
    for &(ctx, value) in &values {
        census.record(ctx, value).expect("records");
    }
    let tables = EntropyTables::build(&plan, &census).expect("tables");
    let mut w = BitWriter::new();
    tables.write_bundle(&mut w).expect("bundle");
    let mut encoder = SymbolEncoder::new(&tables);
    for &(ctx, value) in &values {
        encoder.push_uint(ctx, value).expect("records");
    }
    encoder.write_stream(&mut w).expect("stream");
    let bytes = w.into_bytes();

    let mut caught = 0usize;
    let mut examined = 0usize;
    for byte in (bytes.len().saturating_sub(24))..bytes.len() {
        let mut corrupted = bytes.clone();
        corrupted[byte] ^= 0x55;
        examined += 1;
        let mut guard = AllocGuard::new(&Limits::relaxed());
        let mut r = BitReader::new(&corrupted);
        let Ok(mut decoder) = SymbolDecoder::open(&mut r, 2, &mut guard) else {
            caught += 1;
            continue;
        };
        let mut diverged = false;
        for &(ctx, value) in &values {
            match decoder.read_uint(&mut r, ctx) {
                Ok(got) if got == value => {}
                _ => {
                    diverged = true;
                    break;
                }
            }
        }
        if diverged || decoder.finish().is_err() {
            caught += 1;
        }
    }
    assert_eq!(
        caught, examined,
        "every corruption of the coded payload must be detected"
    );
}

/// One bundle, several streams: the split that
/// [`SymbolDecoder::open_deferred`] and [`SymbolDecoder::restart`] exist for.
#[test]
fn one_bundle_can_carry_several_streams() {
    let config = HybridUintConfig::new(4, 0, 0).expect("legal");
    for mode in [CodingMode::Ans, CodingMode::Prefix] {
        let plan = EncoderPlan::identity(2, mode, config).expect("plan");
        let mut rng = Rng::new(99);
        let groups: Vec<Vec<(usize, u32)>> =
            (0..3).map(|_| values_for(1, &mut rng, 120, 2)).collect();

        let mut census = TokenCensus::new(2).expect("census");
        for group in &groups {
            for &(ctx, value) in group {
                census.record(ctx, value).expect("records");
            }
        }
        let tables = EntropyTables::build(&plan, &census).expect("tables");

        let mut w = BitWriter::new();
        tables.write_bundle(&mut w).expect("bundle");
        for group in &groups {
            let mut encoder = SymbolEncoder::new(&tables);
            for &(ctx, value) in group {
                encoder.push_uint(ctx, value).expect("records");
            }
            encoder.write_stream(&mut w).expect("stream");
        }
        let bits = w.bit_len();
        let bytes = w.into_bytes();

        let mut guard = AllocGuard::new(&Limits::relaxed());
        let mut r = BitReader::new(&bytes);
        let mut decoder =
            SymbolDecoder::open_deferred(&mut r, 2, &mut guard).expect("bundle opens");
        for group in &groups {
            decoder.restart(&mut r).expect("new stream");
            for &(ctx, value) in group {
                assert_eq!(decoder.read_uint(&mut r, ctx).expect("value"), value);
            }
            decoder.finish().expect("terminal state per stream");
        }
        assert_eq!(r.total_bits_read(), bits, "{mode:?}: bits consumed");
    }
}

/// A stream long enough that the ANS state renormalizes constantly, and one
/// whose values ride entirely in raw extra bits: the two ends of the
/// interleaving between the 16-bit renormalization words and `u(n)` payloads.
#[test]
fn renormalization_and_extra_bits_interleave_correctly() {
    // Sixteen equally likely symbols: the state loses four bits per symbol, so
    // a renormalization lands every four symbols or so.
    let config = HybridUintConfig::new(4, 0, 0).expect("legal");
    let plan = EncoderPlan::identity(1, CodingMode::Ans, config).expect("plan");
    let mut rng = Rng::new(7);
    let values: Vec<(usize, u32)> = (0..20_000).map(|_| (0, rng.below(16))).collect();
    check(&plan, &values, "dense renormalization");

    // split_exponent 0: every value above 0 is a token plus a run of raw bits,
    // so nearly every symbol is followed by an extra-bit field.
    let config = HybridUintConfig::new(0, 0, 0).expect("legal");
    let plan = EncoderPlan::identity(1, CodingMode::Ans, config).expect("plan");
    let mut rng = Rng::new(11);
    let values: Vec<(usize, u32)> = (0..5_000)
        .map(|_| (0, rng.next() >> (rng.below(20) + 2)))
        .collect();
    check(&plan, &values, "extra-bit heavy");
}

/// A sweep over `log_alphabet_size`: C.2.1 signals `5 + u(2)`, and the
/// histogram, alias table and token width all depend on it.
#[test]
fn every_ans_alphabet_width_round_trips() {
    for log_alphabet_size in 5..=8u32 {
        let config = HybridUintConfig::new(log_alphabet_size, 0, 0).expect("legal");
        let mut plan = EncoderPlan::identity(2, CodingMode::Ans, config).expect("plan");
        plan.log_alphabet_size = Some(log_alphabet_size);
        let mut rng = Rng::new(log_alphabet_size);
        let top = 1u32 << log_alphabet_size;
        let values: Vec<(usize, u32)> = (0..2000)
            .map(|_| ((rng.below(2)) as usize, rng.below(top)))
            .collect();
        check(
            &plan,
            &values,
            &format!("log_alphabet_size {log_alphabet_size}"),
        );
    }

    // A token past the widest ANS alphabet must be refused, not truncated.
    let config = HybridUintConfig::new(8, 0, 0).expect("legal");
    let plan = EncoderPlan::identity(1, CodingMode::Ans, config).expect("plan");
    let mut census = TokenCensus::new(1).expect("census");
    // With split_exponent 8 and no in-token bits, value 2^40 needs token
    // 8 + (40 - 8) = well past 256.
    for value in [0u32, 1 << 30] {
        census.record(0, value).expect("records");
    }
    assert!(
        EntropyTables::build(&plan, &census).is_err(),
        "an ANS alphabet cannot exceed 256 tokens"
    );
}

/// Prefix codes have no 256-token ceiling, so the same census that ANS refuses
/// must encode cleanly.
#[test]
fn prefix_codes_cover_the_wide_token_range() {
    let config = HybridUintConfig::new(8, 0, 0).expect("legal");
    let plan = EncoderPlan::identity(1, CodingMode::Prefix, config).expect("plan");
    let mut rng = Rng::new(3);
    let mut values: Vec<(usize, u32)> = (0..2000)
        .map(|_| (0, rng.next() >> rng.below(31)))
        .collect();
    values.push((0, u32::MAX));
    check(&plan, &values, "wide prefix alphabet");
}
