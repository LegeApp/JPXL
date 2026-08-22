//! PR 1 of the perceptual quality controller: the public lossy-target surface.
//!
//! No perceptual encode behaviour lands here yet — a score below 100 is an
//! explicit `Unsupported`, a score of 100 routes to the lossless encoder, and
//! the rate and fixed-quantizer expert modes behave exactly as before. These
//! tests pin that contract so PR 4 changes behaviour, not shape.

use jpxl::{Decoder, Effort, EncodeReport, Encoder, Error, PerceptualStatus};

/// A deterministic ≥256×256 RGB gradient with enough structure that the
/// lossless and lossy paths both have real work to do.
fn synthetic_rgb8(width: u32, height: u32) -> Vec<u8> {
    let mut rgb = Vec::with_capacity((width as usize) * (height as usize) * 3);
    for y in 0..height {
        for x in 0..width {
            let r = u8::try_from((x * 3) % 256).unwrap_or(0);
            let g = u8::try_from((y * 5) % 256).unwrap_or(0);
            let b = u8::try_from((x + y) % 256).unwrap_or(0);
            rgb.extend_from_slice(&[r, g, b]);
        }
    }
    rgb
}

#[test]
fn score_100_routes_to_lossless() {
    let (w, h) = (256u32, 256u32);
    let rgb = synthetic_rgb8(w, h);

    let lossless = Encoder::new()
        .with_threads(1)
        .expect("threads")
        .lossless()
        .encode_rgb8(w, h, &rgb)
        .expect("lossless encode");

    let (routed, report) = Encoder::new()
        .with_threads(1)
        .expect("threads")
        .with_ssimulacra2_score(100.0)
        .expect("score")
        .encode_rgb8_reported(w, h, &rgb)
        .expect("perceptual encode");

    assert_eq!(routed, lossless, "score 100 must be the lossless bytes");
    match report {
        EncodeReport::Perceptual(outcome) => {
            assert_eq!(outcome.status, PerceptualStatus::RoutedToLossless);
            assert_eq!(outcome.achieved_score, Some(100.0));
            assert!((outcome.requested_score - 100.0).abs() < f64::EPSILON);
            assert_eq!(outcome.metric_version.as_str(), "ssimulacra2-jpxl-1");
            assert_eq!(
                outcome.exact_bytes,
                u64::try_from(routed.len()).unwrap_or(u64::MAX)
            );
        }
        other => panic!("expected a perceptual report, got {other:?}"),
    }
}

#[test]
fn scores_outside_range_rejected() {
    for score in [-0.1, 100.1, f64::NAN, f64::INFINITY] {
        assert!(
            matches!(
                Encoder::new().with_ssimulacra2_score(score),
                Err(Error::InvalidOption(_))
            ),
            "score {score} must be rejected"
        );
    }
    // The band's endpoints are valid.
    assert!(Encoder::new().with_ssimulacra2_score(0.0).is_ok());
    assert!(Encoder::new().with_ssimulacra2_score(100.0).is_ok());
}

#[test]
fn targets_are_mutually_exclusive() {
    // Quality, then a size.
    let quality = Encoder::new().with_ssimulacra2_score(100.0).expect("score");
    assert!(matches!(
        quality.with_target_bpp(1.0),
        Err(Error::InvalidOption(_))
    ));

    // A size, then a fixed quantizer.
    let bpp = Encoder::new().with_target_bpp(1.0).expect("bpp");
    assert!(matches!(
        bpp.with_global_scale(32_768),
        Err(Error::InvalidOption(_))
    ));

    // `.lossless()` resets, so a fresh target is accepted afterward.
    let reset = Encoder::new()
        .with_target_bpp(1.0)
        .expect("bpp")
        .lossless()
        .with_ssimulacra2_score(100.0);
    assert!(reset.is_ok());
}

#[test]
fn a_perceptual_target_below_100_runs_the_quality_controller() {
    let (w, h) = (256u32, 256u32);
    let rgb = synthetic_rgb8(w, h);
    let (bytes, report) = Encoder::new()
        .with_effort(Effort::Balanced)
        .with_ssimulacra2_score(70.0)
        .expect("score")
        .encode_rgb8_reported(w, h, &rgb)
        .expect("encode");
    let EncodeReport::Perceptual(outcome) = report else {
        panic!("expected a perceptual report");
    };
    assert!(
        outcome.achieved_score.is_some_and(|s| s >= 70.0),
        "{outcome:?}"
    );
    assert!(!bytes.is_empty());
}

#[test]
fn global_scale_encodes_and_decodes() {
    let (w, h) = (256u32, 256u32);
    let rgb = synthetic_rgb8(w, h);
    let (bytes, report) = Encoder::new()
        .with_threads(1)
        .expect("threads")
        .with_global_scale(32_768)
        .expect("global scale")
        .encode_rgb8_reported(w, h, &rgb)
        .expect("fixed-quantizer encode");

    match report {
        EncodeReport::FixedQuantizer { bytes: reported } => {
            assert_eq!(reported, u64::try_from(bytes.len()).unwrap_or(u64::MAX));
        }
        other => panic!("expected a fixed-quantizer report, got {other:?}"),
    }

    let decoded = Decoder::new().decode(&bytes).expect("decode");
    assert_eq!((decoded.width, decoded.height), (w, h));
}

#[test]
fn rate_target_report_matches_bytes() {
    let (w, h) = (256u32, 256u32);
    let rgb = synthetic_rgb8(w, h);
    let target = 20_000u64;
    let (bytes, report) = Encoder::new()
        .with_threads(1)
        .expect("threads")
        .with_target_bytes(target)
        .expect("target")
        .with_effort(Effort::Fast)
        .encode_rgb8_reported(w, h, &rgb)
        .expect("rate encode");

    match report {
        EncodeReport::Rate(summary) => {
            assert_eq!(summary.target_bytes, target);
            assert_eq!(
                summary.achieved_bytes,
                u64::try_from(bytes.len()).unwrap_or(u64::MAX)
            );
            assert!(
                summary.achieved_bytes <= target,
                "the loop never overshoots"
            );
        }
        other => panic!("expected a rate report, got {other:?}"),
    }
}
