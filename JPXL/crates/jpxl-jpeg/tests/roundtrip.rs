//! Byte-exact `serialize(parse(x)) == x` round-trip over real JPEG fixtures.
//!
//! Every fixture is produced by libjpeg-turbo (`cjpeg`) or Pillow from
//! synthetic original content; see `tests/fixtures/generate.py` and the
//! per-file `*.jpg.prov` provenance sidecars. The images are 385×259 — many
//! MCUs, non-multiple-of-16 edges, and (for subsampled cases) differing
//! interleaved / non-interleaved block counts — so the round-trip exercises
//! edge padding blocks, restart cadence, and padding bits, not just a happy
//! single-block path.

#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    reason = "tests index their own just-loaded data; a panic here is a failing \
              test, which is the intended signal"
)]

use std::path::PathBuf;

use jpxl_jpeg::{parse, serialize};

fn fixture(name: &str) -> Vec<u8> {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(name);
    std::fs::read(&p).unwrap_or_else(|e| panic!("reading fixture {name}: {e}"))
}

/// Parses then re-serializes, asserting byte-for-byte identity, and returns the
/// parsed model so callers can additionally assert structural facts.
fn assert_roundtrip(name: &str) -> jpxl_jpeg::Jpeg {
    let original = fixture(name);
    let jpeg = parse(&original).unwrap_or_else(|e| panic!("parse {name}: {e}"));
    let reemitted = serialize(&jpeg).unwrap_or_else(|e| panic!("serialize {name}: {e}"));
    assert_eq!(
        original.len(),
        reemitted.len(),
        "{name}: length differs ({} vs {})",
        original.len(),
        reemitted.len()
    );
    if original != reemitted {
        // Report the first divergence to make failures debuggable.
        let at = original
            .iter()
            .zip(&reemitted)
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        panic!(
            "{name}: byte mismatch at offset {at}: original 0x{:02X} != re-emitted 0x{:02X}",
            original[at], reemitted[at]
        );
    }
    jpeg
}

// ---- Baseline sequential ---------------------------------------------------

#[test]
fn roundtrip_baseline_444_is_bit_exact() {
    let j = assert_roundtrip("base_444.jpg");
    let f = j.frame.as_ref().expect("frame");
    assert_eq!((f.width, f.height), (385, 259));
    assert_eq!(f.components.len(), 3, "YCbCr");
    // Prove parsing produced real coefficients, not an empty pass-through.
    let any_ac = j
        .planes
        .iter()
        .any(|p| p.blocks.iter().any(|b| b.iter().skip(1).any(|&c| c != 0)));
    assert!(any_ac, "expected some nonzero AC coefficients");
}

#[test]
fn roundtrip_baseline_422_is_bit_exact() {
    let j = assert_roundtrip("base_422.jpg");
    let f = j.frame.as_ref().unwrap();
    assert_eq!((f.components[0].h, f.components[0].v), (2, 1));
}

#[test]
fn roundtrip_baseline_420_is_bit_exact() {
    let j = assert_roundtrip("base_420.jpg");
    let f = j.frame.as_ref().unwrap();
    assert_eq!((f.components[0].h, f.components[0].v), (2, 2));
}

#[test]
fn roundtrip_baseline_440_is_bit_exact() {
    assert_roundtrip("base_440.jpg");
}

#[test]
fn roundtrip_baseline_grayscale_is_bit_exact() {
    let j = assert_roundtrip("base_gray.jpg");
    assert_eq!(j.frame.as_ref().unwrap().components.len(), 1);
}

#[test]
fn roundtrip_baseline_high_quality_dense_coeffs_is_bit_exact() {
    assert_roundtrip("base_q98.jpg");
}

#[test]
fn roundtrip_baseline_low_quality_sparse_coeffs_is_bit_exact() {
    assert_roundtrip("base_q20.jpg");
}

// ---- Restart intervals -----------------------------------------------------

#[test]
fn roundtrip_baseline_restart_interval_is_bit_exact() {
    let j = assert_roundtrip("base_restart.jpg");
    // The restart interval must have been captured as a DRI segment.
    let has_dri = j
        .segments
        .iter()
        .any(|s| matches!(s, jpxl_jpeg::Segment::Dri(_)));
    assert!(has_dri, "expected a DRI segment");
}

#[test]
fn roundtrip_baseline_restart_rows_is_bit_exact() {
    assert_roundtrip("base_restart_rows.jpg");
}

// ---- Progressive -----------------------------------------------------------

#[test]
fn roundtrip_progressive_444_is_bit_exact() {
    let j = assert_roundtrip("prog_444.jpg");
    // A progressive frame has multiple SOS segments (spectral selection).
    let scans = j
        .segments
        .iter()
        .filter(|s| matches!(s, jpxl_jpeg::Segment::Sos(_)))
        .count();
    assert!(
        scans > 1,
        "progressive stream should have several scans, got {scans}"
    );
}

#[test]
fn roundtrip_progressive_420_is_bit_exact() {
    assert_roundtrip("prog_420.jpg");
}

#[test]
fn roundtrip_progressive_grayscale_is_bit_exact() {
    assert_roundtrip("prog_gray.jpg");
}

#[test]
fn roundtrip_progressive_restart_is_bit_exact() {
    assert_roundtrip("prog_restart.jpg");
}

#[test]
fn roundtrip_progressive_low_quality_is_bit_exact() {
    assert_roundtrip("prog_q30.jpg");
}

// ---- Metadata and trailing data -------------------------------------------

#[test]
fn roundtrip_exif_icc_metadata_is_bit_exact() {
    let j = assert_roundtrip("meta_exif_icc.jpg");
    let has_app = j
        .segments
        .iter()
        .any(|s| matches!(s, jpxl_jpeg::Segment::App(_)));
    assert!(has_app, "expected APPn metadata segments");
}

#[test]
fn roundtrip_xmp_metadata_is_bit_exact() {
    let j = assert_roundtrip("meta_xmp.jpg");
    let has_app = j
        .segments
        .iter()
        .any(|s| matches!(s, jpxl_jpeg::Segment::App(_)));
    assert!(has_app, "expected an APP1 XMP segment");
}

#[test]
fn roundtrip_trailing_garbage_after_eoi_is_bit_exact() {
    let j = assert_roundtrip("trailing_garbage.jpg");
    assert!(!j.tail.is_empty(), "trailing bytes should be captured");
}
