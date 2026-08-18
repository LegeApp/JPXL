// SPDX-License-Identifier: MIT
//! End-to-end K.5 noise synthesis, graded against the normative corpus.
//!
//! The corpus `noise`/`noise_5` case (500x606, `kVarDCT`, single frame,
//! `upsampling == 1`) is the only exercise this decoder has for
//! [`jpxl_decode::frame::noise`]: no handmade fixture ladder exists because
//! no available encoder exposes a documented flag that reliably turns on
//! `kNoise` alone (`cjxl --photon_noise_iso` was tried and does not set the
//! frame flag on the build available to this repo — see the module's own
//! doc comment for what *is* pinned without a discriminating stream). The
//! corpus case is graded at its own `test.json` thresholds (18181-3 §4.2).
//!
//! Same slow-test convention as the other corpus rungs (`e2e_upsampling.rs`
//! etc.): unconditional in a release build, opt-in via `JPXL_SLOW_TESTS` in
//! debug.

use std::path::{Path, PathBuf};

use jpxl_conformance::{FloatImage, Similarity, similarity};
use jpxl_core::limits::Limits;
use jpxl_decode::{DecodedImage, decode};

fn corpus(case: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/conformance/testcases")
        .join(case)
}

fn handmade(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/handmade")
        .join(name)
}

fn as_float_image(image: &DecodedImage) -> FloatImage {
    let pixels = image.width as usize * image.height as usize;
    let planes = image
        .float_planes
        .as_ref()
        .expect("a kVarDCT decode has float planes");
    let mut samples = Vec::with_capacity(pixels * planes.len());
    for y in 0..image.height {
        for x in 0..image.width {
            for plane in planes {
                samples.push(plane.get(x, y));
            }
        }
    }
    FloatImage {
        frames: 1,
        height: image.height,
        width: image.width,
        channels: u32::try_from(planes.len()).expect("a sane channel count"),
        samples,
    }
}

fn decode_fixture(path: &Path) -> DecodedImage {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    decode(&bytes, &Limits::default()).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn read_npy(path: &Path) -> FloatImage {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    FloatImage::from_npy(&bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn grade(label: &str, ours: &FloatImage, reference: &FloatImage) -> Similarity {
    assert_eq!(
        (ours.width, ours.height, ours.channels),
        (reference.width, reference.height, reference.channels),
        "{label}: 18181-3 4.2 condition 1 (shape) failed"
    );
    let report = similarity(ours, reference).expect("same shape");
    println!(
        "{label}: peak {:.8}, channel peak {:?}, channel RMSE {:?}",
        report.peak_error, report.channel_peak, report.channel_rmse
    );
    report
}

fn assert_conforms(label: &str, report: &Similarity, peak: f32, rmse: f32) {
    assert!(
        report.conforms(peak, rmse),
        "{label}: peak {:.8} (limit {peak}), channel RMSE {:?} (limit {rmse})",
        report.peak_error,
        report.channel_rmse
    );
}

fn corpus_thresholds(case: &Path) -> Option<(f32, f32)> {
    let text = std::fs::read_to_string(case.join("test.json")).ok()?;
    let find = |key: &str| -> Option<f32> {
        text.lines()
            .find(|l| l.contains(key))
            .and_then(|l| l.split(':').nth(1))
            .map(|v| v.trim().trim_end_matches(',').to_owned())
            .and_then(|v| v.parse().ok())
    };
    Some((find("peak_error")?, find("rms_error")?))
}

fn slow_tests_enabled() -> bool {
    !cfg!(debug_assertions) || std::env::var_os("JPXL_SLOW_TESTS").is_some()
}

fn corpus_rung(case: &str) {
    let dir = corpus(case);
    let input = dir.join("input.jxl");
    let reference = dir.join("reference_image.npy");
    if !input.exists() || !reference.exists() {
        eprintln!("skipping corpus {case}: not fetched (see tools/fetch-conformance.sh)");
        return;
    }
    if !slow_tests_enabled() {
        eprintln!(
            "skipping corpus {case}: a 0.3-Mpixel decode is slow unoptimized. \
             Re-run with --release, or set JPXL_SLOW_TESTS=1."
        );
        return;
    }

    let image = decode_fixture(&input);
    assert_eq!(
        (image.width, image.height),
        (500, 606),
        "{case}: output size"
    );
    let report = grade(case, &as_float_image(&image), &read_npy(&reference));
    let (peak, rmse) = corpus_thresholds(&dir).unwrap_or_else(|| panic!("{case}/test.json"));
    assert_conforms(case, &report, peak, rmse);
}

#[test]
fn corpus_noise() {
    corpus_rung("noise");
}

#[test]
fn corpus_noise_5() {
    corpus_rung("noise_5");
}

/// Fixture 110: the smallest expressible `kNoise` stream (32x32, one group).
/// See `110_noise_rgb_32x32.jxl.txt` for why this rung exists alongside the
/// corpus case.
#[test]
fn fixture_110_smallest_noise() {
    let fixture = handmade("110_noise_rgb_32x32.jxl");
    let npy = handmade("110_noise_rgb_32x32.npy");
    if !fixture.exists() || !npy.exists() {
        eprintln!("skipping: fixture 110 not present (run tools/make-noise-fixtures.sh)");
        return;
    }
    let image = decode_fixture(&fixture);
    assert_eq!((image.width, image.height), (32, 32), "fixture 110: size");
    let report = grade("fixture 110", &as_float_image(&image), &read_npy(&npy));
    // Annex A's "with filters" class -- cjxl's own default filter choice is
    // left in place, same as the K.2 upsampling fixtures.
    assert_conforms("fixture 110", &report, 0.02, 1e-3);
}
