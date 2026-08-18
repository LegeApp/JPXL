// SPDX-License-Identifier: MIT
//! End-to-end `do_YCbCr` colour reconstruction (18181-1 L.3), for the two
//! corpus streams with `jpeg_upsampling == [0, 0, 0]` (no J.2 chroma
//! upsampling needed): `bench_oriented_brg`/`_5` and
//! `grayscale_jpeg`/`_5`. Both also incidentally exercise I.2.4 `RAW`
//! dequantization matrices, which `cjxl`'s JPEG-recompression path emits for
//! every `do_YCbCr` stream this decoder has seen (real JPEG quantization
//! tables, carried through unchanged) — see `vardct::render::ycbcr_to_rgb`
//! and `vardct::dequant_matrix::RawMatrixContext`.
//!
//! `cafe`/`_5` is NOT covered here: its `jpeg_upsampling` is `[0, 1, 0]`
//! (chroma-**and**-luma-subsampled — the corpus deliberately does not put
//! the subsampled channel at index 0 or 2, which would be indistinguishable
//! from a channel-order bug), which needs J.2's triangle filter. That stays
//! refused (`18181-1 J.2`).
//!
//! Both corpus streams carry a `reconstructed_jpeg` box (jbrd); this decoder
//! never reconstructs the original JPEG — it parses/skips the box and grades
//! pixels only, per this wave's brief.
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
        .expect("a do_YCbCr decode routes through the float pipeline");
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

/// `jpeg_upsampling == [0, 0, 0]`, RGB, orientation 90-degree rotate
/// (500x606 stored -> 606x500 displayed) plus a RAW dequantization matrix.
#[test]
fn corpus_bench_oriented_brg() {
    corpus_rung("bench_oriented_brg", (606, 500));
}

#[test]
fn corpus_bench_oriented_brg_5() {
    corpus_rung("bench_oriented_brg_5", (606, 500));
}

/// `jpeg_upsampling == [0, 0, 0]`, greyscale `colour_encoding` — the R'G'B'
/// planes L.3 produces are identical (a genuinely grey source), and
/// `assemble_float` keeps only the first when `colour_encoding.is_grey()`.
#[test]
fn corpus_grayscale_jpeg() {
    corpus_rung("grayscale_jpeg", (200, 200));
}

#[test]
fn corpus_grayscale_jpeg_5() {
    corpus_rung("grayscale_jpeg_5", (200, 200));
}
