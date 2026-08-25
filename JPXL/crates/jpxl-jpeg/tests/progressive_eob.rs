//! Regression: progressive AC EOB-run lengths must be recorded at decode and
//! replayed at encode, not re-derived.
//!
//! An encoder may split one end-of-band run into several `EOBn` codes (libjpeg
//! flushes when its correction-bit buffer fills). That split is invisible in
//! the coefficients, so a re-encoder that greedily re-derives run lengths emits
//! a different bit count — the "N bits pending after padding" and
//! "symbol 0xN0 absent from Huffman table" failures seen on the archive sweep.
//! `prog_large.jpg` is large and detailed enough to force such splits.

#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    reason = "test indexes its own just-loaded data; a panic is a failing test"
)]

use std::path::PathBuf;

use jpxl_jpeg::{Segment, parse, serialize};

fn fixture(name: &str) -> Vec<u8> {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(name);
    std::fs::read(&p).unwrap_or_else(|e| panic!("reading {name}: {e}"))
}

#[test]
fn roundtrip_progressive_large_is_bit_exact() {
    let original = fixture("prog_large.jpg");
    let jpeg = parse(&original).expect("parse");
    let round = serialize(&jpeg).expect("serialize");
    assert_eq!(
        original, round,
        "prog_large.jpg must round-trip byte-exactly"
    );
}

#[test]
fn recorded_eob_runs_are_load_bearing() {
    let original = fixture("prog_large.jpg");
    let jpeg = parse(&original).expect("parse");

    // The fixture must actually contain a *split* EOB run for this test to
    // prove anything: two consecutive recorded runs whose total the greedy
    // re-derivation would have merged into one. Any scan with >1 recorded run
    // has been split by the original encoder.
    let split_seen = jpeg.segments.iter().any(|s| match s {
        Segment::Sos(scan) => scan.eob_runs.len() > 1,
        _ => false,
    });
    assert!(
        split_seen,
        "prog_large.jpg should exercise a multi-EOBn (split) run"
    );

    // With the recorded runs cleared, the encoder must fall back to greedy
    // re-derivation, which cannot reproduce the split — so the output must
    // differ from the original (or fail outright). Either proves the recorded
    // runs are necessary for bit-exactness.
    let mut stripped = jpeg.clone();
    for seg in &mut stripped.segments {
        if let Segment::Sos(scan) = seg {
            scan.eob_runs.clear();
        }
    }
    match serialize(&stripped) {
        Ok(bytes) => assert_ne!(
            bytes, original,
            "clearing EOB runs must change the output (they are load-bearing)"
        ),
        Err(_) => { /* greedy re-derivation produced an illegal symbol: also proves the point */ }
    }
}
