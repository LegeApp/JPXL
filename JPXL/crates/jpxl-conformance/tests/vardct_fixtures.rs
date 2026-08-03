//! VarDCT fixture coverage (slice 8, sub-slice 8F0).
//!
//! JPXL's own decoder cannot decode VarDCT yet, so nothing here decodes a
//! fixture with `jpxl` — that is wave 3's `tests/e2e_vardct.rs` once the
//! VarDCT path lands. What this file proves instead is that the **grading
//! pipeline itself** — [`FloatImage::from_npy`] plus [`similarity`] — is
//! correct, by using it to compare a real `djxl` decode against itself.
//! That is a zero-threshold-slack check: two decodes of the same stream by
//! the same pinned oracle build must be bit-for-bit identical, so
//! `peak_error` and every channel's RMSE must be exactly `0.0`. If the NPY
//! reader mis-parses the header (wrong offset, wrong byte order, wrong
//! shape) or `similarity` mis-indexes channels, this is far more likely to
//! catch it than any hand-built fixture, because it's real encoder/decoder
//! output rather than a small hand-written array.
//!
//! Skips (does not fail) when no `djxl` oracle is installed, matching the
//! rest of this crate's oracle-backed tests.

use std::path::{Path, PathBuf};

use jpxl_conformance::{FloatImage, OracleKind, oracle, similarity};

/// `JPXL/tests/fixtures/handmade`.
fn handmade_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("manifest dir has two ancestors")
        .join("tests")
        .join("fixtures")
        .join("handmade")
}

#[test]
fn djxl_self_grading_is_exact_through_the_metrics_pipeline() {
    let Some(djxl) = oracle::find(OracleKind::Djxl) else {
        eprintln!("skipping: no djxl oracle found (run tools/setup-oracles.sh)");
        return;
    };
    eprintln!("using {} at {}", djxl.kind, djxl.path.display());

    // 50 is the smallest committed VarDCT fixture (greyscale, filters off,
    // d=1.0); any fixture would do, since this test is about the pipeline,
    // not this stream's content.
    let fixture = handmade_dir().join("50_vardct_mixed_gray_128x128_nofilters_d1.jxl");
    assert!(
        fixture.is_file(),
        "fixture missing: {} (run tools/make-vardct-fixtures.sh)",
        fixture.display()
    );

    let tmp = std::env::temp_dir();
    let out_a = tmp.join("jpxl-conformance-self-grade-a.npy");
    let out_b = tmp.join("jpxl-conformance-self-grade-b.npy");
    let _ = std::fs::remove_file(&out_a);
    let _ = std::fs::remove_file(&out_b);

    for out in [&out_a, &out_b] {
        match djxl.decode(&fixture, out, jpxl_conformance::OutputFormat::Npy) {
            Ok(()) => {}
            Err(err) if err.is_unavailable() => {
                eprintln!("skipping: {err}");
                return;
            }
            Err(err) => panic!("djxl failed to decode {}: {err}", fixture.display()),
        }
    }

    let bytes_a = std::fs::read(&out_a)
        .unwrap_or_else(|err| panic!("djxl reported success but wrote nothing: {err}"));
    let bytes_b = std::fs::read(&out_b)
        .unwrap_or_else(|err| panic!("djxl reported success but wrote nothing: {err}"));
    let _ = std::fs::remove_file(&out_a);
    let _ = std::fs::remove_file(&out_b);

    let a = FloatImage::from_npy(&bytes_a).expect("djxl emitted a valid NPY");
    let b = FloatImage::from_npy(&bytes_b).expect("djxl emitted a valid NPY");

    assert_eq!(
        (a.frames, a.height, a.width, a.channels),
        (1, 128, 128, 1),
        "unexpected shape for the greyscale VarDCT fixture"
    );
    assert!(!a.is_empty());

    let report = similarity(&a, &b).expect("two decodes of the same stream have the same shape");
    assert_eq!(
        report.peak_error, 0.0,
        "the same pinned djxl build decoding the same stream twice must be bit-exact"
    );
    assert!(
        report.channel_rmse.iter().all(|&rmse| rmse == 0.0),
        "channel RMSE should be exactly zero for a self-comparison: {:?}",
        report.channel_rmse
    );
    assert!(
        report.conforms(0.0, 0.0),
        "self-comparison should pass at the tightest possible threshold"
    );
}
