//! The perceptual quality contract through the public facade: every
//! production effort meets the requested SSIMULACRA2 score inside its
//! budget, the achieved score is what an independent re-score of the decoded
//! bytes measures, higher targets never cost fewer bytes, and the stream is
//! byte-identical across worker counts.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use jpxl::{Decoder, Effort, EncodeReport, Encoder, PerceptualStatus};
use jpxl_perceptual::{LinearRgbView, score_pair};

struct Rng(u32);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        (x >> 8) as f32 / (1u32 << 24) as f32
    }
}

fn synthetic(width: u32, height: u32, seed: u32) -> Vec<u8> {
    let mut rng = Rng(seed.max(1));
    let mut out = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let fx = x as f32 / width as f32;
            let fy = y as f32 / height as f32;
            let mut r = 0.2 + 0.6 * fx;
            let mut g = 0.25 + 0.5 * fy;
            let mut b = 0.3 + 0.4 * (1.0 - fx) * fy;
            if (0.3..0.6).contains(&fy) {
                let t = (((x * 7 + y * 3) % 11) as f32 / 11.0 - 0.5) * 0.25;
                r += t;
                g += 0.6 * t;
                b -= 0.4 * t;
            }
            if fx > 0.7 {
                r *= 0.5;
                g *= 0.5;
                b *= 0.5;
            }
            if (0.1..0.2).contains(&fx) && (0.7..0.8).contains(&fy) {
                r = 0.95;
                g = 0.95;
                b = 0.9;
            }
            let n = (rng.next_f32() - 0.5) * 0.02;
            for v in [r + n, g + n, b + n] {
                out.push((v.clamp(0.0, 1.0) * 255.0).round() as u8);
            }
        }
    }
    out
}

fn srgb_to_linear(v: f32) -> f32 {
    if v <= 12.92 * 0.003_130_8 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Scores the decoded bytes against the 8-bit source, the way `jpxl compare`
/// would on the written file.
fn rescore(width: u32, height: u32, rgb: &[u8], bytes: &[u8]) -> f64 {
    let image = Decoder::new().decode(bytes).expect("the stream decodes");
    let planes = image.float_planes.expect("VarDCT decodes to float planes");
    let decoded: [Vec<f32>; 3] = core::array::from_fn(|c| {
        planes[c]
            .samples
            .iter()
            .map(|&v| srgb_to_linear((v * 255.0).round().clamp(0.0, 255.0) / 255.0))
            .collect()
    });
    let source: [Vec<f32>; 3] = core::array::from_fn(|c| {
        rgb.chunks_exact(3)
            .map(|px| srgb_to_linear(f32::from(px[c]) / 255.0))
            .collect()
    });
    score_pair(
        LinearRgbView::new(width, height, &source[0], &source[1], &source[2]).unwrap(),
        LinearRgbView::new(width, height, &decoded[0], &decoded[1], &decoded[2]).unwrap(),
    )
    .unwrap()
    .score
}

fn perceptual(report: &EncodeReport) -> &jpxl::PerceptualOutcome {
    match report {
        EncodeReport::Perceptual(outcome) => outcome,
        other => panic!("expected a perceptual report, got {other:?}"),
    }
}

#[test]
fn every_effort_meets_every_target_and_reports_the_score_the_file_has() {
    let (w, h) = (320u32, 240u32);
    let rgb = synthetic(w, h, 1);
    for effort in [Effort::Fast, Effort::Balanced] {
        let (max_probes, max_prices) = match effort {
            Effort::Fast => (3, 2),
            _ => (5, 3),
        };
        let mut previous_bytes = 0usize;
        let mut previous_score = 0.0f64;
        for target in [50.0, 70.0, 85.0] {
            let (bytes, report) = Encoder::new()
                .with_ssimulacra2_score(target)
                .unwrap()
                .with_effort(effort)
                .with_threads(4)
                .unwrap()
                .encode_rgb8_reported(w, h, &rgb)
                .unwrap();
            let outcome = perceptual(&report);
            let achieved = outcome.achieved_score.expect("a measured score");
            eprintln!(
                "{effort:?} target {target}: achieved {achieved:.3}, {} bytes, {} probes, {} prices, {:?}",
                bytes.len(),
                outcome.probes,
                outcome.prices,
                outcome.status
            );
            assert!(
                achieved >= target,
                "{effort:?} {target}: achieved {achieved}"
            );
            assert!(
                outcome.probes <= max_probes && outcome.prices <= max_prices,
                "{outcome:?}"
            );
            assert!(!outcome.saturated);
            assert_eq!(outcome.exact_bytes, bytes.len() as u64);
            assert!(
                outcome
                    .trace_json
                    .as_deref()
                    .is_some_and(|t| t.starts_with("{\"schema\":\"jpxl.quality-trace/1\""))
            );
            let independent = rescore(w, h, &rgb, &bytes);
            assert!(
                (independent - achieved).abs() < 1e-6,
                "{effort:?} {target}: reported {achieved} but the file scores {independent}"
            );
            assert!(
                bytes.len() >= previous_bytes,
                "{effort:?}: bytes fell from {previous_bytes} to {}",
                bytes.len()
            );
            assert!(achieved >= previous_score);
            previous_bytes = bytes.len();
            previous_score = achieved;
        }
    }
}

#[test]
fn the_stream_is_byte_identical_across_worker_counts() {
    let (w, h) = (288u32, 264u32);
    let rgb = synthetic(w, h, 2);
    let encode = |threads: usize| {
        Encoder::new()
            .with_ssimulacra2_score(80.0)
            .unwrap()
            .with_effort(Effort::Balanced)
            .with_threads(threads)
            .unwrap()
            .encode_rgb8(w, h, &rgb)
            .unwrap()
    };
    assert_eq!(encode(1), encode(4));
}

#[test]
fn a_twelve_bit_source_is_scored_at_its_own_depth() {
    let (w, h) = (256u32, 200u32);
    let rgb8 = synthetic(w, h, 3);
    let rgb16: Vec<u16> = rgb8
        .iter()
        .map(|&v| u16::from(v) * 16 + u16::from(v % 16))
        .collect();
    let (bytes, report) = Encoder::new()
        .with_ssimulacra2_score(80.0)
        .unwrap()
        .with_effort(Effort::Balanced)
        .encode_rgb16_reported(w, h, 12, &rgb16)
        .unwrap();
    let outcome = perceptual(&report);
    assert!(outcome.achieved_score.unwrap() >= 80.0, "{outcome:?}");
    assert!(Decoder::new().decode(&bytes).is_ok());
}

#[test]
fn tiny_frames_and_a_perfect_score_route_to_lossless() {
    let tiny = synthetic(6, 6, 4);
    let (_, report) = Encoder::new()
        .with_ssimulacra2_score(85.0)
        .unwrap()
        .encode_rgb8_reported(6, 6, &tiny)
        .unwrap();
    assert_eq!(
        perceptual(&report).status,
        PerceptualStatus::UnsupportedTooSmall
    );

    let rgb = synthetic(64, 64, 5);
    let (bytes, report) = Encoder::new()
        .with_ssimulacra2_score(100.0)
        .unwrap()
        .encode_rgb8_reported(64, 64, &rgb)
        .unwrap();
    assert_eq!(
        perceptual(&report).status,
        PerceptualStatus::RoutedToLossless
    );
    assert_eq!(
        bytes,
        Encoder::new().lossless().encode_rgb8(64, 64, &rgb).unwrap()
    );
}

/// PR 5: the Balanced perceptual policy bank meets the target and is never
/// larger than the baseline-only (fixed-policy) result at the same score.
///
/// The public facade always runs the preset budget, so this drives the policy
/// layer directly to toggle the bank via `QualityBudget { policy_trials: 0 }`
/// — the `#[doc(hidden)]` breadth knob — while leaving the facade unchanged.
#[test]
fn the_balanced_bank_is_never_larger_than_the_baseline_only_result() {
    use jpxl_encode_policy::request::{PerceptualMetric, PerceptualTarget};
    use jpxl_encode_policy::{
        AnalysisAtlas, EncodeRequest, PreparedFrame, QualityBudget, RateSearchPreset,
        search_frame_perceptual_with_budget,
    };
    use jpxl_perceptual::PlanRenderEvaluator;

    let (w, h) = (320u32, 240u32);
    let rgb = synthetic(w, h, 7);
    let request = EncodeRequest::for_quality(RateSearchPreset::Balanced);
    let executor = request.resources.executor();
    let target = PerceptualTarget::new(PerceptualMetric::Ssimulacra2, 80.0).unwrap();
    // The bank is off by default on Balanced (its measured wall exceeds the
    // +25% budget), so opt it in explicitly to compare it against baseline-only.
    let full = QualityBudget {
        policy_trials: 2,
        ..QualityBudget::for_preset(RateSearchPreset::Balanced)
    };
    let baseline_only = QualityBudget {
        policy_trials: 0,
        ..full
    };

    let solve = |budget| {
        let frame = PreparedFrame::from_srgb8_with(w, h, &rgb, Some(&executor)).unwrap();
        let atlas = AnalysisAtlas::analyze(&frame);
        let mut ev = PlanRenderEvaluator::from_srgb8(w, h, &rgb, &executor).unwrap();
        search_frame_perceptual_with_budget(
            &frame, &atlas, &request, target, &mut ev, &executor, budget,
        )
        .unwrap()
    };

    let bank = solve(full);
    let base = solve(baseline_only);

    assert!(
        bank.achieved_score >= 80.0,
        "the bank missed the target: {}",
        bank.achieved_score
    );
    assert!(
        bank.sizing.total <= base.sizing.total,
        "bank {} bytes > baseline-only {} bytes",
        bank.sizing.total,
        base.sizing.total
    );
    // When the bank helps, the reported winner margin is the saving.
    assert_eq!(
        bank.stats.policy_winner_margin_bytes,
        base.sizing.total - bank.sizing.total
    );
}
