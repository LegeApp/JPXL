//! Parity against the rust-av `ssimulacra2` crate (the permitted cross-check
//! implementation), on synthetic images always and on repository photographs
//! when they are present on disk.
//!
//! Compiled only with `--features parity-oracle`; the oracle never enters a
//! production build.
//!
//! # Known, measured deviation
//!
//! The oracle runs its recursive Gaussian in `f32`. That filter's cosine
//! generators are poles on the unit circle, so rounding drifts along every
//! row and column and leaves a ripple of a few 1e-6 in the blurred planes.
//! The edge maps rectify that ripple (`max(0, ·)`), which inflates the
//! oracle's error wherever the image is nearly flat — and the effect grows
//! with image size. This crate runs the recursion in `f64` (≈ exact) and
//! therefore scores *above* the oracle at high quality on flat-heavy content:
//! measured +0.14 at 256×192, +0.73 at 640×480 and +1.07 at 1024×768 on the
//! synthetic fixture (score ≈ 93), but only +0.03 → +0.18 from 0.27 MP to
//! 4.3 MP on a photograph (score ≈ 82), where real texture dominates. The
//! tolerances below encode that envelope so a regression in this crate is
//! still caught.

#![cfg(feature = "parity-oracle")]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

mod common;

use common::{Planes, add_noise, box_blur, quantize, read_ppm, repo_root, synthetic};
use jpxl_perceptual::score_pair;
use ssimulacra2::{LinearRgb, compute_frame_ssimulacra2};

/// Megapixels of an image.
fn megapixels(p: &Planes) -> f64 {
    (f64::from(p.width) * f64::from(p.height)) / 1e6
}

/// Accepted difference on synthetic content: tight below score 90, and the
/// measured size-scaled drift envelope above it.
fn synthetic_tolerance(p: &Planes, oracle_score: f64) -> f64 {
    let root_mp = megapixels(p).sqrt();
    if oracle_score > 90.0 {
        0.1 + 1.6 * root_mp
    } else {
        0.05 + 0.5 * root_mp
    }
}

/// Accepted difference on photographs.
fn photo_tolerance(p: &Planes) -> f64 {
    0.1 + 0.1 * megapixels(p).sqrt()
}

fn oracle(reference: &Planes, candidate: &Planes) -> f64 {
    let a = LinearRgb::new(
        reference.interleaved(),
        reference.width as usize,
        reference.height as usize,
    )
    .unwrap();
    let b = LinearRgb::new(
        candidate.interleaved(),
        candidate.width as usize,
        candidate.height as usize,
    )
    .unwrap();
    compute_frame_ssimulacra2(a, b).unwrap()
}

/// Scores one pair both ways, prints the comparison, and returns the absolute
/// difference together with a failure line when it exceeds `tolerance`.
fn compare(
    label: &str,
    reference: &Planes,
    candidate: &Planes,
    tolerance: impl Fn(f64) -> f64,
) -> (f64, Option<String>) {
    let ours = score_pair(reference.view(), candidate.view())
        .unwrap()
        .score;
    let theirs = oracle(reference, candidate);
    let delta = (ours - theirs).abs();
    let tolerance = tolerance(theirs);
    eprintln!(
        "{label}: ours {ours:.4} oracle {theirs:.4} delta {delta:.4} (tolerance {tolerance:.3})"
    );
    let failure = (delta > tolerance).then(|| {
        format!("{label}: ours {ours:.4} vs oracle {theirs:.4} (delta {delta:.4} > {tolerance:.3})")
    });
    (delta, failure)
}

#[test]
fn synthetic_pairs_agree_with_the_oracle() {
    let mut worst: f64 = 0.0;
    let mut failures = Vec::new();
    for (seed, (w, h)) in [
        (1u32, (256u32, 192u32)),
        (2, (333, 257)),
        (3, (640, 480)),
        (4, (97, 131)),
        (5, (1024, 768)),
    ] {
        let img = synthetic(w, h, seed);
        for (name, cand) in [
            ("noise-0.003", add_noise(&img, 0.003, seed + 10)),
            ("noise-0.02", add_noise(&img, 0.02, seed + 20)),
            ("noise-0.08", add_noise(&img, 0.08, seed + 30)),
            ("blur-1", box_blur(&img, 1)),
            ("blur-4", box_blur(&img, 4)),
            ("quant-16", quantize(&img, 16.0)),
            ("blur-noise", add_noise(&box_blur(&img, 2), 0.01, seed + 40)),
        ] {
            let (delta, failure) = compare(&format!("{w}x{h}/{name}"), &img, &cand, |score| {
                synthetic_tolerance(&img, score)
            });
            worst = worst.max(delta);
            failures.extend(failure);
        }
    }
    eprintln!("worst synthetic delta {worst:.5}");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn repository_photographs_agree_with_the_oracle_when_present() {
    let root = repo_root();
    let candidates = [
        root.join(".agent/scratch/gap-g0-smoke-20260821/mid.ppm"),
        root.join(".agent/scratch/gap-g0-smoke-20260821/source.ppm"),
    ];
    let mut compared = 0;
    let mut worst: f64 = 0.0;
    let mut failures = Vec::new();
    for path in candidates {
        let Some(photo) = read_ppm(&path) else {
            eprintln!("skipping absent fixture {}", path.display());
            continue;
        };
        // Keep the oracle's runtime bounded: score a 1024x768 crop.
        let photo = crop(&photo, 1024, 768);
        let label = path.file_name().unwrap().to_string_lossy().into_owned();
        for (name, cand) in [
            ("noise-0.005", add_noise(&photo, 0.005, 5)),
            ("blur-1", box_blur(&photo, 1)),
            ("quant-32", quantize(&photo, 32.0)),
        ] {
            let (delta, failure) = compare(&format!("{label}/{name}"), &photo, &cand, |_| {
                photo_tolerance(&photo)
            });
            worst = worst.max(delta);
            failures.extend(failure);
            compared += 1;
        }
    }
    eprintln!("compared {compared} photograph pairs, worst delta {worst:.5}");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn crop(src: &Planes, width: u32, height: u32) -> Planes {
    let w = src.width.min(width) as usize;
    let h = src.height.min(height) as usize;
    let stride = src.width as usize;
    let take = |plane: &[f32]| -> Vec<f32> {
        let mut out = Vec::with_capacity(w * h);
        for y in 0..h {
            out.extend_from_slice(&plane[y * stride..y * stride + w]);
        }
        out
    };
    Planes {
        width: w as u32,
        height: h as u32,
        r: take(&src.r),
        g: take(&src.g),
        b: take(&src.b),
    }
}

/// Stage-level diagnostic: compares the colour conversion and the blur
/// against the oracle crate's own, so a score mismatch can be localised.
#[test]
fn stage_outputs_match_the_oracle() {
    use jpxl_perceptual::SerialExecutor;
    use jpxl_perceptual::blur::Blur;
    use jpxl_perceptual::color::planes_to_positive_xyb;
    use ssimulacra2::Xyb;

    let img = synthetic(1024, 768, 5);
    let (w, h) = (img.width as usize, img.height as usize);

    // Colour: ours vs yuvxyb + the positive shift.
    let mut x = vec![0.0f32; w * h];
    let mut y = vec![0.0f32; w * h];
    let mut b = vec![0.0f32; w * h];
    planes_to_positive_xyb(&img.r, &img.g, &img.b, &mut x, &mut y, &mut b);
    let theirs = Xyb::from(LinearRgb::new(img.interleaved(), w, h).unwrap());
    let mut worst = [0.0f32; 3];
    for (i, px) in theirs.data().iter().enumerate() {
        let tb = (px[2] - px[1]) + 0.55;
        let tx = px[0].mul_add(14.0, 0.42);
        let ty = px[1] + 0.01;
        worst[0] = worst[0].max((x[i] - tx).abs());
        worst[1] = worst[1].max((y[i] - ty).abs());
        worst[2] = worst[2].max((b[i] - tb).abs());
    }
    eprintln!("positive-XYB max abs diff per channel: {worst:?}");

    // Blur: ours vs the oracle's Blur on the oracle's own planes.
    let planes: [Vec<f32>; 3] = [x.clone(), y.clone(), b.clone()];
    let mut oracle_blur = ssimulacra2::Blur::new(w, h);
    let theirs = oracle_blur.blur(&planes);
    let mut ours = vec![0.0f32; w * h];
    let mut blur = Blur::new();
    for c in 0..3 {
        blur.blur_plane(&planes[c], &mut ours, w, h, &SerialExecutor);
        let (mut max_diff, mut at) = (0.0f32, 0usize);
        for (i, (&a, &t)) in ours.iter().zip(&theirs[c]).enumerate() {
            let d = (a - t).abs();
            if d > max_diff {
                max_diff = d;
                at = i;
            }
        }
        eprintln!(
            "blur channel {c}: max abs diff {max_diff:.3e} at ({}, {}) ours {} theirs {}",
            at % w,
            at / w,
            ours[at],
            theirs[c][at]
        );
    }
    assert!(worst.iter().all(|&d| d < 1e-5), "{worst:?}");
}

/// High-quality drift versus image size on a photograph (see the module
/// docs): reported per sqrt(megapixel) and bounded by the photo envelope.
#[test]
fn high_quality_drift_versus_size_is_reported() {
    let root = repo_root();
    let Some(photo) = read_ppm(&root.join(".agent/scratch/gap-g0-smoke-20260821/mid.ppm")) else {
        eprintln!("skipping: mid.ppm absent");
        return;
    };
    for (w, h) in [(600u32, 450u32), (1200, 900), (2400, 1800)] {
        let img = crop(&photo, w, h);
        let cand = add_noise(&img, 0.002, 17);
        let (delta, failure) = compare(&format!("mid-{w}x{h}/noise-0.002"), &img, &cand, |_| {
            photo_tolerance(&img)
        });
        eprintln!(
            "  {w}x{h}: delta {delta:.4} per sqrt(MP) {:.4}",
            delta / megapixels(&img).sqrt()
        );
        assert!(failure.is_none(), "{}", failure.unwrap_or_default());
    }
}
