//! Encode-side LZ77 round-trips through the public entropy facade.
//!
//! Policy chooses literals vs copies; this crate emits Table C.1 / C.3.3 and
//! the paired decoder must recover the expanded raw sequence bit-exactly.

#![allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]

use jpxl_bitstream::{BitReader, BitWriter};
use jpxl_core::limits::{AllocGuard, Limits};
use jpxl_entropy::encode::{
    CodingMode, EncoderPlan, EntropyTables, Lz77EncodeParams, SymbolEncoder, TokenCensus,
};
use jpxl_entropy::{HybridUintConfig, SymbolDecoder};

fn guard() -> AllocGuard {
    AllocGuard::new(&Limits::relaxed())
}

/// Events the policy layer would produce for a residual stream.
#[derive(Clone, Copy)]
enum Ev {
    Lit(u32),
    /// `(length, raw_distance)` — raw_distance 0 means one symbol back when
    /// `dist_multiplier == 0`.
    Copy(u32, u32),
}

fn encode_and_decode(
    mode: CodingMode,
    config: HybridUintConfig,
    lz77: Lz77EncodeParams,
    events: &[Ev],
    case: &str,
) -> Vec<u32> {
    let plan = EncoderPlan::identity_with_lz77(1, mode, config, lz77)
        .unwrap_or_else(|e| panic!("{case}: plan {e}"));
    let dist_ctx = plan.lz_dist_ctx().expect("lz77 on");
    let num_dist = plan.context_map.num_dist();

    let mut census = TokenCensus::new(num_dist).unwrap_or_else(|e| panic!("{case}: census {e}"));
    for &ev in events {
        match ev {
            Ev::Lit(v) => census
                .record(0, v)
                .unwrap_or_else(|e| panic!("{case}: record lit {e}")),
            Ev::Copy(len, dist) => census
                .record_copy(0, len, dist, dist_ctx, &lz77)
                .unwrap_or_else(|e| panic!("{case}: record copy {e}")),
        }
    }

    let tables =
        EntropyTables::build(&plan, &census).unwrap_or_else(|e| panic!("{case}: tables {e}"));
    assert_eq!(tables.num_value_contexts(), 1);
    assert!(tables.lz77().is_some());

    let mut w = BitWriter::new();
    tables
        .write_bundle(&mut w)
        .unwrap_or_else(|e| panic!("{case}: bundle {e}"));
    let mut encoder = SymbolEncoder::new(&tables);
    for &ev in events {
        match ev {
            Ev::Lit(v) => encoder
                .push_uint(0, v)
                .unwrap_or_else(|e| panic!("{case}: push lit {v}: {e}")),
            Ev::Copy(len, dist) => encoder
                .push_copy(0, len, dist)
                .unwrap_or_else(|e| panic!("{case}: push copy: {e}")),
        }
    }
    encoder
        .write_stream(&mut w)
        .unwrap_or_else(|e| panic!("{case}: stream {e}"));
    let bits = w.bit_len();
    let bytes = w.into_bytes();

    // Expand expected logical sequence.
    let mut expected = Vec::new();
    for &ev in events {
        match ev {
            Ev::Lit(v) => expected.push(v),
            Ev::Copy(len, raw_dist) => {
                let distance = u64::from(raw_dist) + 1;
                for _ in 0..len {
                    let idx = expected
                        .len()
                        .checked_sub(distance as usize)
                        .expect("distance in range for test");
                    expected.push(expected[idx]);
                }
            }
        }
    }

    let mut g = guard();
    let mut r = BitReader::new(&bytes);
    let mut dec = SymbolDecoder::open(&mut r, tables.num_value_contexts(), &mut g)
        .unwrap_or_else(|e| panic!("{case}: open {e}"));
    assert!(dec.lz77().enabled, "{case}: lz77 flag");
    assert_eq!(dec.lz77().min_symbol, lz77.min_symbol);
    assert_eq!(dec.lz77().min_length, lz77.min_length);
    assert_eq!(dec.clusters().num_dist(), 2);

    let mut got = Vec::with_capacity(expected.len());
    for i in 0..expected.len() {
        let v = dec
            .read_uint(&mut r, 0)
            .unwrap_or_else(|e| panic!("{case}: read {i}: {e}"));
        got.push(v);
    }
    dec.finish()
        .unwrap_or_else(|e| panic!("{case}: finish {e}"));
    assert_eq!(
        r.total_bits_read(),
        bits,
        "{case}: no trailing bits (read {} of {bits})",
        r.total_bits_read()
    );
    assert_eq!(got, expected, "{case}: sample sequence");
    got
}

fn lz77_params_like_fixture() -> Lz77EncodeParams {
    // Matches tests/lz77_streams.rs: min_symbol=8, min_length=3, split_exponent=8.
    let length_config = HybridUintConfig::new(8, 0, 0).expect("legal");
    Lz77EncodeParams::new(8, 3, length_config).expect("params")
}

/// Hybrid-uint for value + distance clusters.
///
/// `split_exponent = 7` keeps ANS `log_alphabet_size` ≤ 8 while still letting
/// small literals stay strictly below `min_symbol = 8`.
fn value_config() -> HybridUintConfig {
    HybridUintConfig::new(7, 0, 0).expect("legal")
}

#[test]
fn literals_only_with_lz77_enabled_round_trip() {
    // Values 0..7 tokenize to themselves under (7,0,0) and stay below min_symbol=8.
    let events = [Ev::Lit(1), Ev::Lit(2), Ev::Lit(3), Ev::Lit(2)];
    for mode in [CodingMode::Prefix, CodingMode::Ans] {
        encode_and_decode(
            mode,
            value_config(),
            lz77_params_like_fixture(),
            &events,
            &format!("literals-only {mode:?}"),
        );
    }
}

#[test]
fn back_reference_repeats_the_previous_symbol() {
    // Same logical sequence as lz77_streams::back_reference_repeats_the_previous_symbol:
    // literals 1, 2 then copy length 3 distance raw 0 → [1, 2, 2, 2, 2].
    let events = [Ev::Lit(1), Ev::Lit(2), Ev::Copy(3, 0)];
    for mode in [CodingMode::Prefix, CodingMode::Ans] {
        let got = encode_and_decode(
            mode,
            value_config(),
            lz77_params_like_fixture(),
            &events,
            &format!("repeat-prev {mode:?}"),
        );
        assert_eq!(got, vec![1, 2, 2, 2, 2]);
    }
}

#[test]
fn overlapping_copy_replays_the_recent_pair() {
    // Window [1, 2], copy length 4 at distance 2 (raw_distance 1) → 1,2,1,2,1,2.
    let events = [Ev::Lit(1), Ev::Lit(2), Ev::Copy(4, 1)];
    let got = encode_and_decode(
        CodingMode::Prefix,
        value_config(),
        lz77_params_like_fixture(),
        &events,
        "overlap",
    );
    assert_eq!(got, vec![1, 2, 1, 2, 1, 2]);
}

#[test]
fn long_run_via_distance_one() {
    // One literal then copy 20 of the same symbol (raw_distance 0 → distance 1).
    let events = [Ev::Lit(5), Ev::Copy(20, 0)];
    let got = encode_and_decode(
        CodingMode::Ans,
        value_config(),
        lz77_params_like_fixture(),
        &events,
        "long-run",
    );
    assert_eq!(got, vec![5; 21]);
}

#[test]
fn lz77_disabled_plan_rejects_push_copy() {
    let config = HybridUintConfig::new(4, 0, 0).expect("legal");
    let mut census = TokenCensus::new(1).expect("census");
    census.record(0, 1).expect("rec");
    let plan = EncoderPlan::identity(1, CodingMode::Ans, config).expect("plan");
    let tables = EntropyTables::build(&plan, &census).expect("tables");
    let mut enc = SymbolEncoder::new(&tables);
    assert!(enc.push_copy(0, 3, 0).is_err());
}

#[test]
fn literal_token_colliding_with_min_symbol_is_rejected() {
    // min_symbol=8; hybrid-uint (7,0,0) emits token==value for values < 128.
    let lz77 = lz77_params_like_fixture();
    let plan =
        EncoderPlan::identity_with_lz77(1, CodingMode::Prefix, value_config(), lz77).expect("plan");
    let mut census = TokenCensus::new(plan.context_map.num_dist()).expect("census");
    // Only token 8 in the alphabet (as a length trigger), no safe literals recorded.
    census.record_token(0, 8).expect("length token");
    census.record(1, 0).expect("dist");
    let tables = EntropyTables::build(&plan, &census).expect("tables");
    let mut enc = SymbolEncoder::new(&tables);
    // Value 8 tokenizes to token 8 == min_symbol → must reject as literal.
    assert!(enc.push_uint(0, 8).is_err());
}

#[test]
fn empty_lz77_stream_opens_and_finishes() {
    let lz77 = lz77_params_like_fixture();
    encode_and_decode(CodingMode::Ans, value_config(), lz77, &[], "empty");
    encode_and_decode(
        CodingMode::Prefix,
        value_config(),
        lz77_params_like_fixture(),
        &[],
        "empty-prefix",
    );
}
