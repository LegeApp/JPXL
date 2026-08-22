//! What the metric promises regardless of any oracle: a perfect score for an
//! identical image, a score that falls as distortion rises, and one score no
//! matter how the work is partitioned or what the reference retained.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

mod common;

use common::{add_noise, box_blur, quantize, synthetic};
use jpxl_perceptual::{
    LinearRgbView, MetricError, PrecomputedReference, ReferenceRetention, ScopedThreadExecutor,
    SerialExecutor, Ssimulacra2, score_pair,
};

#[test]
fn an_identical_image_scores_exactly_one_hundred() {
    let img = synthetic(256, 192, 1);
    let result = score_pair(img.view(), img.view()).unwrap();
    assert_eq!(result.score, 100.0);
    assert_eq!(result.raw_error, 0.0);
    assert_eq!(
        result.scales.len(),
        6,
        "256x192 is halved down to 8x6: six scales"
    );
}

#[test]
fn the_score_falls_as_noise_rises_and_as_blur_deepens() {
    let img = synthetic(320, 240, 2);
    let mut last = 100.0;
    for amp in [0.002, 0.01, 0.03, 0.08] {
        let score = score_pair(img.view(), add_noise(&img, amp, 7).view())
            .unwrap()
            .score;
        assert!(score < last, "noise {amp}: {score} should be below {last}");
        last = score;
    }
    let one = score_pair(img.view(), box_blur(&img, 1).view())
        .unwrap()
        .score;
    let three = score_pair(img.view(), box_blur(&img, 3).view())
        .unwrap()
        .score;
    assert!(three < one && one < 100.0, "blur: {one} then {three}");
}

#[test]
fn banding_is_penalised() {
    let img = synthetic(256, 256, 3);
    let coarse = score_pair(img.view(), quantize(&img, 12.0).view())
        .unwrap()
        .score;
    let fine = score_pair(img.view(), quantize(&img, 120.0).view())
        .unwrap()
        .score;
    assert!(coarse < fine && fine < 100.0, "{coarse} vs {fine}");
}

#[test]
fn scores_are_in_the_published_range_for_typical_distortions() {
    let img = synthetic(400, 300, 4);
    let mild = score_pair(img.view(), add_noise(&img, 0.004, 9).view())
        .unwrap()
        .score;
    assert!(mild > 60.0 && mild < 100.0, "mild noise scored {mild}");
    let harsh = score_pair(img.view(), box_blur(&img, 6).view())
        .unwrap()
        .score;
    assert!(harsh < 60.0, "heavy blur scored {harsh}");
}

#[test]
fn the_executor_and_the_retention_mode_do_not_change_the_score() {
    let img = synthetic(333, 257, 5);
    let cand = add_noise(&box_blur(&img, 1), 0.01, 11);
    let serial = score_pair(img.view(), cand.view()).unwrap();

    let threaded_ref = PrecomputedReference::new(
        img.view(),
        ReferenceRetention::Moments,
        &ScopedThreadExecutor { workers: 4 },
    )
    .unwrap();
    let threaded = Ssimulacra2::new()
        .score(
            &threaded_ref,
            cand.view(),
            &ScopedThreadExecutor { workers: 4 },
        )
        .unwrap();
    assert_eq!(
        serial.score.to_bits(),
        threaded.score.to_bits(),
        "{} vs {}",
        serial.score,
        threaded.score
    );
    assert_eq!(serial.scales, threaded.scales);

    let planes_only =
        PrecomputedReference::new(img.view(), ReferenceRetention::PlanesOnly, &SerialExecutor)
            .unwrap();
    assert!(planes_only.retained_bytes() < threaded_ref.retained_bytes());
    let recomputed = Ssimulacra2::new()
        .score(
            &planes_only,
            cand.view(),
            &ScopedThreadExecutor { workers: 3 },
        )
        .unwrap();
    assert_eq!(serial.score.to_bits(), recomputed.score.to_bits());
    assert_eq!(serial.scales, recomputed.scales);
}

#[test]
fn a_reused_scorer_gives_the_same_answer_as_a_fresh_one() {
    let img = synthetic(200, 150, 6);
    let a = add_noise(&img, 0.02, 3);
    let b = box_blur(&img, 2);
    let reference =
        PrecomputedReference::new(img.view(), ReferenceRetention::Moments, &SerialExecutor)
            .unwrap();
    let mut scorer = Ssimulacra2::new();
    let first_a = scorer.score(&reference, a.view(), &SerialExecutor).unwrap();
    let first_b = scorer.score(&reference, b.view(), &SerialExecutor).unwrap();
    let again_a = scorer.score(&reference, a.view(), &SerialExecutor).unwrap();
    assert_eq!(first_a, again_a);
    assert_eq!(
        first_b,
        Ssimulacra2::new()
            .score(&reference, b.view(), &SerialExecutor)
            .unwrap()
    );
}

#[test]
fn small_and_mismatched_inputs_are_refused_explicitly() {
    let tiny = synthetic(7, 12, 1);
    assert_eq!(
        PrecomputedReference::new(tiny.view(), ReferenceRetention::Moments, &SerialExecutor).err(),
        Some(MetricError::TooSmall {
            width: 7,
            height: 12
        })
    );
    let eight = synthetic(8, 8, 1);
    let reference =
        PrecomputedReference::new(eight.view(), ReferenceRetention::Moments, &SerialExecutor)
            .unwrap();
    assert_eq!(
        reference.scale_count(),
        2,
        "8x8 is still halved once, to 4x4"
    );
    assert_eq!(
        Ssimulacra2::new()
            .score(&reference, eight.view(), &SerialExecutor)
            .unwrap()
            .score,
        100.0
    );

    let other = synthetic(16, 8, 1);
    assert_eq!(
        Ssimulacra2::new()
            .score(&reference, other.view(), &SerialExecutor)
            .err(),
        Some(MetricError::DimensionMismatch {
            reference: (8, 8),
            candidate: (16, 8)
        })
    );
    assert_eq!(
        LinearRgbView::new(4, 4, &[0.0; 15], &[0.0; 16], &[0.0; 16]).err(),
        Some(MetricError::PlaneLength {
            expected: 16,
            found: 15
        })
    );
    assert_eq!(
        LinearRgbView::new(0, 4, &[], &[], &[]).err(),
        Some(MetricError::ZeroDimension)
    );
}

#[test]
fn grayscale_input_is_scored_like_any_other() {
    let mut img = synthetic(128, 96, 8);
    img.g = img.r.clone();
    img.b = img.r.clone();
    let cand = add_noise(&img, 0.01, 2);
    let score = score_pair(img.view(), cand.view()).unwrap().score;
    assert!(score < 100.0 && score > 30.0, "{score}");
}

/// Peak resident set of the metric alone at 12 MP, for each retention mode.
/// Run one mode per process (peak RSS is a high-water mark):
/// `JPXL_RETENTION=moments cargo test --release -p jpxl-perceptual --test metric -- --ignored --nocapture twelve_megapixel_memory`
/// then again with `JPXL_RETENTION=planes`.
#[test]
#[ignore = "memory aid; prints, does not assert"]
fn twelve_megapixel_memory() {
    let retention = match std::env::var("JPXL_RETENTION").ok().as_deref() {
        Some("planes") => ReferenceRetention::PlanesOnly,
        _ => ReferenceRetention::Moments,
    };
    let img = synthetic(4000, 3000, 9);
    let cand = add_noise(&img, 0.01, 3);
    let exec = ScopedThreadExecutor { workers: 4 };
    let reference = PrecomputedReference::new(img.view(), retention, &exec).unwrap();
    let mut scorer = Ssimulacra2::new();
    let _ = scorer.score(&reference, cand.view(), &exec).unwrap();
    let r = scorer.score(&reference, cand.view(), &exec).unwrap();
    let hwm = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM"))
                .map(std::string::ToString::to_string)
        })
        .unwrap_or_else(|| "VmHWM: n/a".to_string());
    eprintln!(
        "retention {retention:?}: reference retained {} MB, peak {} (score {:.3})",
        reference.retained_bytes() / (1024 * 1024),
        hwm.trim(),
        r.score
    );
}

/// Wall-time split of one 4 MP comparison, serial and on four threads.
/// Run with `cargo test --release -p jpxl-perceptual --test metric -- --ignored --nocapture timing`.
#[test]
#[ignore = "timing aid; prints, does not assert"]
fn timing_four_megapixels() {
    use std::time::Instant;
    let img = synthetic(2400, 1800, 9);
    let cand = add_noise(&img, 0.01, 3);
    for workers in [1usize, 4] {
        let exec = ScopedThreadExecutor { workers };
        let t = Instant::now();
        let reference =
            PrecomputedReference::new(img.view(), ReferenceRetention::Moments, &exec).unwrap();
        let prep = t.elapsed();
        let mut scorer = Ssimulacra2::new();
        let t = Instant::now();
        let r = scorer.score(&reference, cand.view(), &exec).unwrap();
        let first = t.elapsed();
        let t = Instant::now();
        let _ = scorer.score(&reference, cand.view(), &exec).unwrap();
        let second = t.elapsed();
        eprintln!(
            "workers {workers}: reference {prep:?}, score {first:?} then {second:?} (score {:.3})",
            r.score
        );
    }
}
