// SPDX-License-Identifier: MIT
//! End-to-end kModular displayed-frame gaps: L.2.2's `kModular` pre-step for
//! an `xyb_encoded` displayed frame (`bicycles`), and K.3.2 patch blending on
//! a non-`xyb_encoded` kModular frame reading back a non-`xyb_encoded`
//! `kReferenceOnly` frame (`patches_lossless`).
//!
//! Both gaps are closed by `jpxl_decode::decode`'s
//! `modular_displayed_pipeline` (private) plus the non-XYB arm of
//! `reference_from_modular` — see `docs/HANDOFF.md` for the wave that added
//! them and `docs/CONFORMANCE.md`'s K.3/L.2 rows.
//!
//! No handmade fixture ladder exists here: the constructs are exercised
//! precisely by the two corpus streams (`bicycles`: kModular, xyb_encoded,
//! no patches, no extra channels, 12 groups; `patches_lossless`: kModular,
//! not xyb_encoded, patches, one alpha extra channel, a cropped
//! `kReferenceOnly` atlas frame), and no available encoder flag reliably
//! reproduces the same combination smaller. Both are graded at their own
//! `test.json` thresholds (18181-3 §4.2). See the corresponding entry in
//! `docs/HANDOFF.md` for the pixel-exact djxl comparison this decoder's
//! `jpxl decode` output was checked against before this test was written.
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

fn as_float_image(image: &DecodedImage) -> FloatImage {
    let pixels = image.width as usize * image.height as usize;
    let planes = image
        .float_planes
        .as_ref()
        .expect("a displayed kModular decode routed through the float pipeline");
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
            "skipping corpus {case}: a multi-group decode is slow unoptimized. \
             Re-run with --release, or set JPXL_SLOW_TESTS=1."
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

/// L.2.2's `kModular` pre-step for a *displayed* `xyb_encoded` frame — the
/// same X = x'*m_x / Y = y'*m_y / B = (B'+Y')*m_b scaling `xyb_from_modular`
/// already applies to a `kLFFrame` and a `kReferenceOnly` frame, now also
/// reachable when the frame IS the displayed output. No patches, no extra
/// channels, no restoration filters, 12 groups (multi-group per AGENTS.md
/// §6).
#[test]
fn corpus_bicycles() {
    corpus_rung("bicycles", (1024, 631));
}

/// K.3.2 patch blending on a `kModular` frame, reading back a
/// `kReferenceOnly` frame that is *also* not `xyb_encoded` (a small,
/// cropped 198x198 atlas inside a 1600x1096 canvas). Exercises the
/// non-XYB arm of both `modular_displayed_pipeline` and
/// `reference_from_modular` at once — patches without any colour
/// transform at all, matching a `[0, 1]`-scale blend on both sides.
#[test]
fn corpus_patches_lossless() {
    corpus_rung("patches_lossless", (1600, 1096));
}
