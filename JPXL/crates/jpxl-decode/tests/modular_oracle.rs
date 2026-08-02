//! Harness for modular sub-bitstreams lifted from oracle-produced files.
//!
//! The hand-derived streams in `modular_stream.rs` and `modular_transforms.rs`
//! prove that Annex H is implemented as written. They cannot prove that the
//! *reading* of Annex H matches what real encoders emit. Four decisions in
//! `src/modular/mod.rs` — recorded there under "Specification ambiguities" —
//! are settled by argument rather than by evidence, and one more, the `Idiv`
//! versus `>>` question in Table H.3 row 13 (`AvgAll`), differs for every
//! negative sample:
//!
//! 1. H.2's global-tree case: whether a group re-reads its own C.1 distribution
//!    bundle or shares the GlobalModular one. This is the highest-risk of the
//!    five — a wrong answer desynchronises the whole group.
//! 2. H.6.2's inverse squeeze and the `hshift`/`vshift` restoration.
//! 3. H.6.4's `/ 4` in the implicit-palette formulas.
//! 4. H.6.4's `index & 1 == 0` operator precedence.
//! 5. Table H.3's `Idiv 16` in `AvgAll`.
//!
//! Fixtures are carved from `cjxl`-produced streams by
//! `tools/make-modular-fixtures.sh` plus the byte ranges recorded in each
//! `.txt` sidecar. A fixture is one whole **`LfGlobal` section** (18181-1 G.1)
//! rather than a bare modular sub-bitstream, because a real sub-bitstream
//! generally sets `use_global_tree` and so is not decodable without the tree
//! that precedes it in the same section. The harness therefore drives the
//! short G.1 preamble — `LfChannelDequantization` (G.1.2) and the
//! `GlobalModular` tree flag (G.1.3) — before handing over to Annex H.
//!
//! Fixture format, one pair per case in [`fixture_dir`]:
//!
//! * `<name>.bin` — the raw bytes of the `LfGlobal` section, from its first
//!   byte (the section boundary is byte-aligned by F.3.3).
//! * `<name>.expected` — a text sidecar:
//!   - line 1: the initial channel list, as `width,height,hshift,vshift`
//!     groups separated by whitespace;
//!   - line 2: `stream_index bits_per_sample`;
//!   - lines 3+: one line of whitespace-separated samples per decoded channel,
//!     in raster order, after the inverse transforms.
//!
//! Every fixture needs the provenance sidecar `AGENTS.md` section 9 requires.
//!
//! Until then [`decodes_oracle_fixtures`] is `#[ignore]`d so it reports as
//! ignored rather than as a vacuous pass.

#![allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]

mod modular_common;

use std::path::{Path, PathBuf};

use jpxl_bitstream::BitReader;
use jpxl_core::limits::{AllocGuard, Limits};
use jpxl_decode::modular::{
    ChannelSpec, ChannelStop, ModularOptions, TreeSource, decode_sub_bitstream_partial,
    read_global_tree,
};

/// Directory the slice 7 fixtures will live in.
fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/modular")
}

/// One extracted modular sub-bitstream and the samples it must decode to.
#[derive(Debug)]
struct OracleCase {
    name: String,
    payload: Vec<u8>,
    initial: Vec<ChannelSpec>,
    stream_index: u32,
    bits_per_sample: u32,
    /// One vector of samples per decoded channel, in raster order.
    expected: Vec<Vec<i32>>,
}

/// Decodes one case and compares it against the recorded expectation.
///
/// Kept separate from fixture discovery so the comparison path is exercised by
/// [`harness_accepts_a_known_good_case_and_rejects_a_wrong_one`] even while no
/// fixtures exist.
fn check_case(case: &OracleCase) -> Result<(), String> {
    let limits = Limits::default();
    let mut guard = AllocGuard::new(&limits);
    let mut reader = BitReader::new(&case.payload);
    let options = ModularOptions {
        stream_index: case.stream_index,
        bits_per_sample: case.bits_per_sample,
        ..ModularOptions::default()
    };

    // G.1.2: LfChannelDequantization is present for every encoding. Modular
    // mode never uses the weights, but the bits are still there.
    let all_default = reader
        .read_bool()
        .map_err(|e| format!("{}: LfChannelDequantization: {e}", case.name))?;
    if !all_default {
        for _ in 0..3 {
            jpxl_bitstream::read_f16_as_f32(&mut reader)
                .map_err(|e| format!("{}: LfChannelDequantization: {e}", case.name))?;
        }
    }

    // G.1.3: the optional global MA tree, then the modular sub-bitstream.
    let have_global_tree = reader
        .read_bool()
        .map_err(|e| format!("{}: global tree flag: {e}", case.name))?;
    let global = if have_global_tree {
        Some(
            read_global_tree(&mut reader, &options, &mut guard)
                .map_err(|e| format!("{}: global tree: {e}", case.name))?,
        )
    } else {
        None
    };
    let source = match global.as_ref() {
        Some(global) => TreeSource::Global {
            global,
            restart: true,
        },
        None => TreeSource::Local,
    };
    let image = decode_sub_bitstream_partial(
        &mut reader,
        &case.initial,
        &options,
        source,
        ChannelStop::All,
        &mut guard,
    )
    .and_then(|partial| partial.into_image(&options, &mut guard))
    .map_err(|e| format!("{}: decode failed: {e}", case.name))?;

    let decoded: Vec<Vec<i32>> = image
        .channels()
        .iter()
        .map(|c| c.samples().to_vec())
        .collect();
    if decoded == case.expected {
        Ok(())
    } else {
        Err(format!(
            "{}: decoded {} channels {decoded:?}, expected {} channels {:?}",
            case.name,
            decoded.len(),
            case.expected.len(),
            case.expected
        ))
    }
}

/// Proves the harness itself decodes and compares correctly, using the same
/// hand-built stream as `modular_stream::one_leaf_zero_predictor_decodes_bit_
/// exactly`. This tests the harness, not conformance against the oracle.
#[test]
fn harness_accepts_a_known_good_case_and_rejects_a_wrong_one() {
    use modular_common::{
        BitWriter, write_modular_header_no_transforms, write_prefix_bundle, write_token,
    };

    const ALPHABET: usize = 4;
    let mut w = BitWriter::new();
    // The G.1 preamble the harness now expects: all-default LF dequantization
    // and no global tree.
    w.bit(true);
    w.bit(false);
    write_modular_header_no_transforms(&mut w, false);
    write_prefix_bundle(&mut w, 6, ALPHABET, 4);
    for token in [0u32, 0, 0, 0, 0] {
        write_token(&mut w, ALPHABET, token); // one leaf, predictor 0
    }
    write_prefix_bundle(&mut w, 1, ALPHABET, 4);
    for token in [2u32, 1, 0, 3] {
        write_token(&mut w, ALPHABET, token);
    }

    let case = OracleCase {
        name: "synthetic/one-leaf-zero".into(),
        payload: w.finish(),
        initial: vec![ChannelSpec::new(2, 2)],
        stream_index: 0,
        bits_per_sample: 8,
        expected: vec![vec![1, -1, 0, -2]],
    };
    check_case(&case).expect("the harness must accept a known-good case");

    let wrong = OracleCase {
        expected: vec![vec![0, 0, 0, 0]],
        ..case
    };
    assert!(
        check_case(&wrong).is_err(),
        "a harness that accepts wrong samples proves nothing"
    );
}

/// Decodes every fixture in [`fixture_dir`].
#[test]
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
        if let Err(message) = check_case(&load_case(&path)) {
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

/// Parses a fixture pair in the format documented at the top of this file.
fn load_case(payload_path: &Path) -> OracleCase {
    let expected_path = payload_path.with_extension("expected");
    let payload = std::fs::read(payload_path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", payload_path.display()));
    let sidecar = std::fs::read_to_string(&expected_path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", expected_path.display()));
    let mut lines = sidecar.lines();

    let initial = lines
        .next()
        .unwrap_or_else(|| panic!("{}: missing channel list", expected_path.display()))
        .split_whitespace()
        .map(|group| {
            let mut parts = group.split(',').map(|f| {
                f.parse::<i64>()
                    .unwrap_or_else(|e| panic!("{}: bad field {f:?}: {e}", expected_path.display()))
            });
            let mut next = || parts.next().unwrap_or(0);
            ChannelSpec::with_shifts(next() as u32, next() as u32, next() as i32, next() as i32)
        })
        .collect();

    let mut meta = lines
        .next()
        .unwrap_or_else(|| panic!("{}: missing metadata line", expected_path.display()))
        .split_whitespace()
        .map(|t| t.parse::<u32>().unwrap_or(0));
    let stream_index = meta.next().unwrap_or(0);
    let bits_per_sample = meta.next().unwrap_or(8);

    let expected = lines
        .map(|line| {
            line.split_whitespace()
                .map(|t| {
                    t.parse().unwrap_or_else(|e| {
                        panic!("{}: bad sample {t:?}: {e}", expected_path.display())
                    })
                })
                .collect()
        })
        .collect();

    OracleCase {
        name: payload_path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
        payload,
        initial,
        stream_index,
        bits_per_sample,
        expected,
    }
}
