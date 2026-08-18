// SPDX-License-Identifier: MIT
//! Corpus cases that no prior wave attempted grading for, discovered while
//! auditing the remaining 21 gates for the 2026-08-04 K.5/kModular wave.
//! None of them needed new code: `delta_palette` and `lz77_flower` are
//! lossless kModular streams (Annex H's delta-palette transform and Annex
//! C's LZ77 layer, both already exercised elsewhere but never against these
//! particular corpus streams), and `opsin_inverse`/`opsin_inverse_5` are a
//! plain kVarDCT stream that happens to be named for the L.2 matrix rather
//! than for a specific construct. All four decode and grade well inside
//! their own `test.json` thresholds with the code already on `main`.
//!
//! `lossless_pfm` (32-bit float samples, a "possibly lossless" `rms_error:
//! 0.0` case) and `grayscale_public_university` were probed at the same
//! time and do **not** pass (peak 1.75 and 0.27 respectively against
//! thresholds of 0.0 and 9.8e-4) — left as future work, not chased here
//! since neither is on this wave's brief and neither regressed by anything
//! in it.
//!
//! Same slow-test convention as the other corpus rungs.

use std::path::{Path, PathBuf};

use jpxl_conformance::{FloatImage, Similarity, similarity};
use jpxl_core::limits::Limits;
use jpxl_decode::{DecodedImage, decode};

fn corpus(case: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/conformance/testcases")
        .join(case)
}

/// Colour planes as `[0, 1]` floats, from whichever of `float_planes`
/// (kVarDCT) or the integer `planes` (kModular) the decode populated.
fn as_float_image(image: &DecodedImage) -> FloatImage {
    let pixels = image.width as usize * image.height as usize;
    if let Some(planes) = &image.float_planes {
        let mut samples = Vec::with_capacity(pixels * planes.len());
        for y in 0..image.height {
            for x in 0..image.width {
                for plane in planes {
                    samples.push(plane.get(x, y));
                }
            }
        }
        return FloatImage {
            frames: 1,
            height: image.height,
            width: image.width,
            channels: u32::try_from(planes.len()).expect("a sane channel count"),
            samples,
        };
    }
    let mut samples = Vec::with_capacity(pixels * image.planes.len());
    for y in 0..image.height {
        for x in 0..image.width {
            for plane in &image.planes {
                let max = plane.max_value() as f32;
                samples.push(plane.get(x, y) as f32 / max);
            }
        }
    }
    FloatImage {
        frames: 1,
        height: image.height,
        width: image.width,
        channels: u32::try_from(image.planes.len()).expect("a sane channel count"),
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

fn corpus_rung(case: &str, expected_size: (u32, u32)) {
    let dir = corpus(case);
    let input = dir.join("input.jxl");
    let reference = dir.join("reference_image.npy");
    if !input.exists() || !reference.exists() {
        eprintln!("skipping corpus {case}: not fetched (see tools/fetch-conformance.sh)");
        return;
    }
    if !slow_tests_enabled() {
        eprintln!(
            "skipping corpus {case}: slow unoptimized. Re-run with --release, \
             or set JPXL_SLOW_TESTS=1."
        );
        return;
    }

    let image = decode_fixture(&input);
    assert_eq!(
        (image.width, image.height),
        expected_size,
        "{case}: output size"
    );
    let report = grade(case, &as_float_image(&image), &read_npy(&reference));
    let (peak, rmse) = corpus_thresholds(&dir).unwrap_or_else(|| panic!("{case}/test.json"));
    assert_conforms(case, &report, peak, rmse);
}

#[test]
fn corpus_delta_palette() {
    corpus_rung("delta_palette", (555, 751));
}

#[test]
fn corpus_lz77_flower() {
    corpus_rung("lz77_flower", (834, 244));
}

#[test]
fn corpus_opsin_inverse() {
    corpus_rung("opsin_inverse", (500, 606));
}

#[test]
fn corpus_opsin_inverse_5() {
    corpus_rung("opsin_inverse_5", (500, 606));
}
