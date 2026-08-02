//! Harness for entropy-coded sections lifted from oracle-produced files.
//!
//! The hand-derived vectors in the sibling test files prove that each clause
//! of Annex C is implemented as written. They cannot prove that the *reading*
//! of the clause matches what real encoders emit — in particular the two
//! ambiguities recorded in the crate documentation (the C.2.2 nested-LZ77 flag
//! and, less riskily, the C.3.3 recursive call) are decided here by argument
//! rather than by evidence.
//!
//! TODO(slice 7): feed this harness with entropy-coded sections extracted from
//! `djxl`-decodable streams, per `docs/PLAN.md` slice 3's second acceptance
//! criterion ("decode entropy-coded sections lifted from oracle files
//! bit-exactly"). Each fixture is a `.bin` payload plus a `.expected` list of
//! decoded integers and the context sequence used to read them, with a
//! provenance sidecar as `AGENTS.md` section 9 requires. Producing them needs
//! the frame header and TOC work of slice 6 to locate the sections, which is
//! why they cannot exist yet.
//!
//! Until then [`decodes_oracle_fixtures`] is `#[ignore]`d so that it reports
//! as ignored rather than as a vacuous pass.

// Test code: an out-of-range index is a failed assertion rather than an attack
// surface, and the casts are on values these tests chose themselves. The lints
// exist for the decode paths in src/.
#![allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]

mod common;

use std::path::{Path, PathBuf};

use jpxl_bitstream::BitReader;
use jpxl_core::limits::{AllocGuard, Limits};
use jpxl_entropy::SymbolDecoder;

/// Directory the slice 7 fixtures will live in.
fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/entropy")
}

/// One extracted entropy-coded section and its expected decoding.
#[derive(Debug)]
struct OracleVector {
    name: String,
    /// Raw bits of the entropy-coded section.
    payload: Vec<u8>,
    /// Number of pre-clustered contexts the referencing clause supplies.
    num_dist: usize,
    /// Context to use for each successive `read_uint` call.
    contexts: Vec<usize>,
    /// The integers the section must decode to.
    expected: Vec<u32>,
}

/// Decodes one vector and compares it against the recorded expectation.
///
/// Kept separate from fixture discovery so that the decode path is exercised
/// by [`harness_decodes_a_synthetic_vector`] even while no fixtures exist.
fn check_vector(vector: &OracleVector) -> Result<(), String> {
    let mut reader = BitReader::new(&vector.payload);
    let limits = Limits::default();
    let mut guard = AllocGuard::new(&limits);
    let mut decoder = SymbolDecoder::open(&mut reader, vector.num_dist, &mut guard)
        .map_err(|e| format!("{}: opening the bundle failed: {e}", vector.name))?;

    let mut decoded = Vec::with_capacity(vector.contexts.len());
    for (i, &ctx) in vector.contexts.iter().enumerate() {
        let value = decoder
            .read_uint(&mut reader, ctx)
            .map_err(|e| format!("{}: symbol {i} (context {ctx}) failed: {e}", vector.name))?;
        decoded.push(value);
    }
    decoder
        .finish()
        .map_err(|e| format!("{}: terminal state: {e}", vector.name))?;

    if decoded == vector.expected {
        Ok(())
    } else {
        Err(format!(
            "{}: decoded {decoded:?}, expected {:?}",
            vector.name, vector.expected
        ))
    }
}

/// Proves the harness itself decodes and compares correctly, using the same
/// hand-derived ANS bundle as `ans_streams.rs`. This is a test of the harness,
/// not of conformance against the oracle.
#[test]
fn harness_decodes_a_synthetic_vector() {
    use common::BitWriter;

    // See `ans_streams::ans_bundle_end_to_end` for the field-by-field
    // derivation of this bundle: one context, ANS, a single symbol holding the
    // whole probability mass, seeded at the C.3.2 terminal state.
    let mut w = BitWriter::new();
    w.bit(false); // lz77.enabled
    w.bit(false); // use_prefix_code
    w.u(0, 2); // log_alphabet_size = 5
    w.u(5, 3); // split_exponent = 5
    w.bit(true);
    w.bit(false);
    w.bit(true);
    w.u(0, 3); // D[1] = 4096
    w.u(0x0013_0000, 32); // initial ANS state

    let vector = OracleVector {
        name: "synthetic/single-symbol".into(),
        payload: w.finish(),
        num_dist: 1,
        contexts: vec![0; 6],
        expected: vec![1; 6],
    };
    check_vector(&vector).expect("the harness must accept a known-good vector");

    // And it must reject a wrong expectation, otherwise it proves nothing.
    let wrong = OracleVector {
        expected: vec![0; 6],
        ..vector
    };
    assert!(check_vector(&wrong).is_err());
}

/// Decodes every fixture in [`fixture_dir`].
///
/// Ignored until slice 7 produces the fixtures; see the module documentation.
#[test]
#[ignore = "TODO(slice 7): needs entropy sections extracted from oracle files"]
fn decodes_oracle_fixtures() {
    let dir = fixture_dir();
    let entries = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("fixture directory {} is missing: {e}", dir.display()));

    let mut checked = 0usize;
    let mut failures = Vec::new();
    for entry in entries {
        let path = entry.expect("readable directory entry").path();
        if path.extension().is_none_or(|ext| ext != "bin") {
            continue;
        }
        let vector = load_vector(&path);
        if let Err(message) = check_vector(&vector) {
            failures.push(message);
        }
        checked += 1;
    }

    assert!(
        checked > 0,
        "no fixtures found in {}; this test must not pass vacuously",
        dir.display()
    );
    assert!(
        failures.is_empty(),
        "{} failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Parses a fixture pair. The `.expected` sidecar format is three
/// whitespace-separated lines: `num_dist`, the context sequence, and the
/// expected values.
fn load_vector(payload_path: &Path) -> OracleVector {
    let expected_path = payload_path.with_extension("expected");
    let payload = std::fs::read(payload_path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", payload_path.display()));
    let sidecar = std::fs::read_to_string(&expected_path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", expected_path.display()));

    let mut lines = sidecar.lines();
    let parse_list = |line: Option<&str>, what: &str| -> Vec<u32> {
        line.unwrap_or_else(|| panic!("{} missing {what}", expected_path.display()))
            .split_whitespace()
            .map(|t| {
                t.parse().unwrap_or_else(|e| {
                    panic!("{}: bad {what} token {t:?}: {e}", expected_path.display())
                })
            })
            .collect()
    };

    let num_dist = parse_list(lines.next(), "num_dist")
        .first()
        .copied()
        .unwrap_or_else(|| panic!("{}: empty num_dist", expected_path.display()))
        as usize;
    let contexts = parse_list(lines.next(), "contexts")
        .into_iter()
        .map(|c| c as usize)
        .collect();
    let expected = parse_list(lines.next(), "expected values");

    OracleVector {
        name: payload_path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
        payload,
        num_dist,
        contexts,
        expected,
    }
}
