//! Regression coverage for the four named flip-point experiments resolved in
//! `docs/experiments/2026-08-03-flip-point-fixtures.md`:
//! `NESTED_LZ77_REJECTS_ENABLED` (jpxl-entropy), `GAB_CUSTOM_REQUIRES_NOT_ALL_DEFAULT`
//! and `RESETS_CANVAS_SHARED_ACROSS_BUNDLES` (both jpxl-decode), and the
//! Table H.3 `AvgAll` `Idiv` question (found not to be ambiguous at all —
//! nothing to test here, see the report).
//!
//! None of fixtures 40-42 exercises the ambiguity it was aimed at (every
//! probe tried left the two readings agreeing) — see the fixtures' `.txt`
//! sidecars and the experiment report for why each is still worth keeping.
//! The genuine bitstream-level regression for `RESETS_CANVAS_SHARED_ACROSS_BUNDLES`
//! is a hand-built unit test in `frame::header::tests`, not here; the
//! `NESTED_LZ77_REJECTS_ENABLED` regression is a hand-built unit test in
//! `jpxl-entropy`'s `decoder::tests`, not here.

#![allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]

use std::path::{Path, PathBuf};

use jpxl_core::limits::Limits;
use jpxl_decode::decode;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("manifest dir has two ancestors")
        .join("tests")
        .join("fixtures")
        .join("handmade")
}

fn read_fixture(name: &str) -> Vec<u8> {
    let path = fixture_dir().join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
}

/// Fixture 40 (nested-LZ77 probe): a plain 64x64 checkerboard, modular,
/// lossless. The flip point was never exercised (see the sidecar), but the
/// fixture still proves the decoder handles an effort-1, highly repetitive
/// modular stream bit-exactly against the deterministic source formula.
#[test]
fn fixture_40_checker_decodes_bit_exactly() {
    let bytes = read_fixture("40_checker_64x64_lossless.jxl");
    let image = decode(&bytes, &Limits::default()).expect("fixture 40 decodes");

    assert_eq!((image.width, image.height), (64, 64));
    assert_eq!(image.num_colour_channels, 3);

    for y in 0..64u32 {
        for x in 0..64u32 {
            let white = (x / 8 + y / 8) % 2 == 0;
            let want = if white { 255 } else { 0 };
            for c in 0..3 {
                let got = image.planes[c].get(x, y);
                assert_eq!(got, want, "channel {c} sample ({x}, {y})");
            }
        }
    }
}

/// Fixture 41 (resets_canvas probe): a 16x16 RGBA gradient, modular,
/// lossless. The flip point was never exercised (colour and extra-channel
/// blend modes always agree in a single-frame stream — see the sidecar), but
/// the fixture proves this decoder handles a modular extra channel (alpha)
/// end to end for the first time in this project's test corpus.
#[test]
fn fixture_41_rgba_decodes_bit_exactly() {
    let bytes = read_fixture("41_rgba_gradient_16x16_lossless.jxl");
    let image = decode(&bytes, &Limits::default()).expect("fixture 41 decodes");

    assert_eq!((image.width, image.height), (16, 16));
    assert_eq!(image.num_colour_channels, 3);
    assert_eq!(image.planes.len(), 4, "3 colour + 1 alpha");

    for y in 0..16u32 {
        for x in 0..16u32 {
            let want = [
                (x * 15) % 256,
                (y * 15) % 256,
                (x + y) % 256,
                if (x + y) % 2 == 0 { 128 } else { 255 },
            ];
            for (c, &w) in want.iter().enumerate() {
                let got = image.planes[c].get(x, y);
                assert_eq!(got, w as i32, "channel {c} sample ({x}, {y})");
            }
        }
    }
}

/// Fixture 42 (gab_custom probe): a 16x16 RGB gradient, modular, with
/// `--gaborish=1` forced. Not lossless (see the sidecar) and this decoder
/// does not apply the Gaborish filter to pixels (18181-1 J.3 is out of
/// scope, per `restoration.rs`'s module doc), so this only proves the header
/// and every section parse without a bitstream error — the signal the
/// experiment used in place of pixel comparison.
#[test]
fn fixture_42_gab_forced_header_and_sections_parse() {
    let bytes = read_fixture("42_gab_forced_16x16_lossless.jxl");
    let image = decode(&bytes, &Limits::default())
        .expect("fixture 42's header/TOC/sections parse without a bitstream error");

    assert_eq!((image.width, image.height), (16, 16));
    assert_eq!(image.num_colour_channels, 3);
}
