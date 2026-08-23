//! Plan-render parity: the renderer's pixels must match what `jpxl-decode`
//! reconstructs from the emitted codestream, for every reconstruction feature
//! the encoder can signal, and the perceptual score of the rendered frame
//! must match the score of the emitted-and-decoded one.
//!
//! `jpxl-decode` is a dev-dependency here and nowhere else in this crate:
//! parity is proved by running it, never by sharing its code.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use jpxl_core::limits::Limits;
use jpxl_encode::vardct::ids::GlobalScale;
use jpxl_encode::vardct::{
    ValidatedEmissionPlan, attach_and_validate_entropy, emit_codestream, validate_pixels,
};
use jpxl_encode_policy::{
    EncodeRequest, EpfSharpnessMode, PreparedFrame, RateSearchPreset, RateTarget,
    RestorationDecision, encode_srgb8_to_target, plan_frame,
};
use jpxl_perceptual::{LinearRgbView, score_pair};
use jpxl_plan_render::{PlanRenderer, RenderedFrame};

/// xorshift32, so fixtures never depend on a random crate.
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

/// A multi-group fixture with smooth regions, texture, sharp edges and a
/// saturated patch, so the hierarchical cover picks several transform sizes
/// and chroma-from-luma has something to regress on.
fn synthetic_rgb8(width: u32, height: u32, seed: u32) -> Vec<u8> {
    let mut rng = Rng(seed.max(1));
    let mut out = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let fx = x as f32 / width as f32;
            let fy = y as f32 / height as f32;
            let mut r = 0.25 + 0.5 * fx;
            let mut g = 0.3 + 0.4 * fy;
            let mut b = 0.35 + 0.3 * (1.0 - fx) * fy;
            if (0.25..0.55).contains(&fy) {
                let t = (((x * 5 + y * 3) % 13) as f32 / 13.0 - 0.5) * 0.3;
                r += t;
                g += 0.5 * t;
            }
            if fx > 0.72 {
                r *= 0.45;
                g *= 0.45;
                b *= 0.45;
            }
            if (0.08..0.2).contains(&fx) && (0.65..0.85).contains(&fy) {
                r = 0.95;
                g = 0.2;
                b = 0.15;
            }
            let n = (rng.next_f32() - 0.5) * 0.03;
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

/// Decodes `bytes` with the in-tree decoder and returns its three float
/// colour planes (sRGB-encoded, unclipped).
fn decode_planes(bytes: &[u8]) -> (u32, u32, [Vec<f32>; 3]) {
    let image =
        jpxl_decode::decode(bytes, &Limits::default()).expect("jpxl-decode accepts the stream");
    let planes = image
        .float_planes
        .expect("a VarDCT frame decodes to float planes");
    assert!(planes.len() >= 3, "three colour planes");
    let take = |i: usize| planes[i].samples.clone();
    (image.width, image.height, [take(0), take(1), take(2)])
}

struct Parity {
    max_abs: f32,
    mismatched_samples: usize,
    total_samples: usize,
    max_lsb: i32,
}

/// Compares the renderer's output with the decoder's, as floats and as the
/// quantized integers a file would hold.
fn compare(label: &str, rendered: &RenderedFrame, decoded: &[Vec<f32>; 3], bits: u32) -> Parity {
    let max = ((1u32 << bits) - 1) as f32;
    let mut max_abs = 0.0f32;
    let mut mismatched = 0usize;
    let mut max_lsb = 0i32;
    let mut total = 0usize;
    let quantized = rendered.quantized(bits);
    for ((ours, theirs), q) in rendered
        .encoded_planes()
        .iter()
        .zip(decoded)
        .zip(&quantized)
    {
        assert_eq!(ours.len(), theirs.len(), "{label}: plane length");
        for ((&a, &b), &qa) in ours.iter().zip(theirs).zip(q) {
            max_abs = max_abs.max((a - b).abs());
            let qb = (b * max).round().clamp(0.0, max) as i32;
            if qa != qb {
                mismatched += 1;
                max_lsb = max_lsb.max((qa - qb).abs());
            }
            total += 1;
        }
    }
    eprintln!(
        "{label}: max |Δ| {max_abs:.3e}, {mismatched}/{total} quantized samples differ (max {max_lsb} LSB)"
    );
    Parity {
        max_abs,
        mismatched_samples: mismatched,
        total_samples: total,
        max_lsb,
    }
}

/// Renders a validated emission plan's pixels.
fn render(plan: &ValidatedEmissionPlan) -> RenderedFrame {
    let mut renderer = PlanRenderer::new().expect("default renderer");
    renderer.render(&plan.pixels()).expect("the plan renders")
}

/// Asserts the float planes agree to rounding and the integers to at most
/// one LSB on a vanishing fraction of samples.
fn assert_parity(label: &str, parity: &Parity) {
    assert!(
        parity.max_abs < 2e-4,
        "{label}: float planes differ by {:.3e}",
        parity.max_abs
    );
    assert!(
        parity.max_lsb <= 1,
        "{label}: quantized samples differ by {} LSB",
        parity.max_lsb
    );
    let fraction = parity.mismatched_samples as f64 / parity.total_samples.max(1) as f64;
    assert!(
        fraction < 1e-3,
        "{label}: {fraction:.2e} of samples differ after quantization"
    );
}

fn fixed_quantizer_request(
    restoration: RestorationDecision,
    sharpness: EpfSharpnessMode,
) -> EncodeRequest {
    let mut request = EncodeRequest::defaults();
    request.restoration = restoration;
    request.epf_sharpness = sharpness;
    request.global_scale = GlobalScale::new(6_000).expect("a legal global scale");
    request
}

fn transform_histogram(plan: &ValidatedEmissionPlan) -> Vec<(u8, usize)> {
    let mut counts = std::collections::BTreeMap::new();
    for group in plan.plan().spatial.lf_groups.iter() {
        for vb in group.blocks.iter() {
            *counts.entry(vb.transform.dct_select()).or_insert(0usize) += 1;
        }
    }
    counts.into_iter().collect()
}

#[test]
fn fixed_quantizer_without_filters_matches_the_decoder() {
    let (w, h) = (320u32, 272u32);
    let rgb = synthetic_rgb8(w, h, 1);
    let frame = PreparedFrame::from_srgb8(w, h, &rgb).unwrap();
    let request = fixed_quantizer_request(
        RestorationDecision {
            gaborish: false,
            epf_iters: 0,
        },
        EpfSharpnessMode::Zero,
    );
    let plan = plan_frame(&frame, &request).unwrap();
    eprintln!("transforms used: {:?}", transform_histogram(&plan));
    let bytes = emit_codestream(&plan).unwrap().bytes;
    let (dw, dh, decoded) = decode_planes(&bytes);
    assert_eq!((dw, dh), (w, h));
    let rendered = render(&plan);
    assert_parity("no-filters", &compare("no-filters", &rendered, &decoded, 8));
}

#[test]
fn every_restoration_setting_matches_the_decoder() {
    let (w, h) = (288u32, 264u32);
    let rgb = synthetic_rgb8(w, h, 2);
    let frame = PreparedFrame::from_srgb8(w, h, &rgb).unwrap();
    let cases = [
        ("gaborish", true, 0u8, EpfSharpnessMode::Zero),
        ("epf1-uniform7", false, 1, EpfSharpnessMode::Uniform7),
        ("gab+epf1", true, 1, EpfSharpnessMode::Uniform7),
        ("gab+epf2", true, 2, EpfSharpnessMode::Uniform7),
        ("gab+epf3", true, 3, EpfSharpnessMode::Uniform7),
        (
            "epf2-adaptive",
            false,
            2,
            EpfSharpnessMode::Adaptive(jpxl_encode_policy::AdaptiveSharpness::default()),
        ),
    ];
    for (label, gaborish, epf_iters, sharpness) in cases {
        let request = fixed_quantizer_request(
            RestorationDecision {
                gaborish,
                epf_iters,
            },
            sharpness,
        );
        let plan = plan_frame(&frame, &request).unwrap();
        let bytes = emit_codestream(&plan).unwrap().bytes;
        let (_, _, decoded) = decode_planes(&bytes);
        let rendered = render(&plan);
        assert_parity(label, &compare(label, &rendered, &decoded, 8));
    }
}

#[test]
fn the_production_rate_presets_match_the_decoder() {
    let (w, h) = (384u32, 320u32);
    let rgb = synthetic_rgb8(w, h, 3);
    for (label, preset) in [
        ("balanced", RateSearchPreset::Balanced),
        ("fast", RateSearchPreset::Fast),
    ] {
        let target = RateTarget::BitsPerPixel(1.2);
        let mut request = EncodeRequest::for_target(target);
        request.rate_preset = preset;
        let outcome = encode_srgb8_to_target(w, h, &rgb, &request, target).unwrap();
        eprintln!(
            "{label}: transforms used {:?}",
            transform_histogram(&outcome.plan)
        );
        let (_, _, decoded) = decode_planes(&outcome.codestream);
        let rendered = render(&outcome.plan);
        assert_parity(label, &compare(label, &rendered, &decoded, 8));
    }
}

#[test]
fn chroma_qm_scales_and_grayscale_match_the_decoder() {
    let (w, h) = (272u32, 256u32);
    // Non-neutral chroma matrices.
    let rgb = synthetic_rgb8(w, h, 4);
    let frame = PreparedFrame::from_srgb8(w, h, &rgb).unwrap();
    let mut request = fixed_quantizer_request(
        RestorationDecision {
            gaborish: true,
            epf_iters: 1,
        },
        EpfSharpnessMode::Uniform7,
    );
    request.x_qm_scale = jpxl_encode::vardct::ids::QmScale::new(3).unwrap();
    request.b_qm_scale = jpxl_encode::vardct::ids::QmScale::new(4).unwrap();
    let plan = plan_frame(&frame, &request).unwrap();
    let bytes = emit_codestream(&plan).unwrap().bytes;
    let (_, _, decoded) = decode_planes(&bytes);
    assert_parity("qm-3-4", &compare("qm-3-4", &render(&plan), &decoded, 8));

    // Grayscale source: the encoder pins neutral CfL and the renderer must
    // still agree.
    let gray: Vec<u8> = rgb
        .chunks_exact(3)
        .flat_map(|px| {
            let v = ((u32::from(px[0]) * 54 + u32::from(px[1]) * 183 + u32::from(px[2]) * 19) / 256)
                as u8;
            [v, v, v]
        })
        .collect();
    let frame = PreparedFrame::from_srgb8(w, h, &gray).unwrap();
    let plan = plan_frame(&frame, &request).unwrap();
    let bytes = emit_codestream(&plan).unwrap().bytes;
    let (_, _, decoded) = decode_planes(&bytes);
    assert_parity(
        "grayscale",
        &compare("grayscale", &render(&plan), &decoded, 8),
    );
}

#[test]
fn a_twelve_bit_source_matches_the_decoder_at_its_own_depth() {
    let (w, h) = (256u32, 256u32);
    let rgb8 = synthetic_rgb8(w, h, 5);
    let rgb16: Vec<u16> = rgb8
        .iter()
        .map(|&v| u16::from(v) * 16 + (v % 16) as u16)
        .collect();
    let frame = PreparedFrame::from_srgb16(w, h, &rgb16, 12).unwrap();
    let mut request = fixed_quantizer_request(
        RestorationDecision {
            gaborish: true,
            epf_iters: 1,
        },
        EpfSharpnessMode::Uniform7,
    );
    request.bits_per_sample = 12;
    let plan = plan_frame(&frame, &request).unwrap();
    let bytes = emit_codestream(&plan).unwrap().bytes;
    let (_, _, decoded) = decode_planes(&bytes);
    let rendered = render(&plan);
    assert_eq!(rendered.bits_per_sample(), 12);
    assert_parity("12-bit", &compare("12-bit", &rendered, &decoded, 12));
}

#[test]
fn the_rendered_score_equals_the_emitted_and_decoded_score() {
    let (w, h) = (320u32, 256u32);
    let rgb = synthetic_rgb8(w, h, 6);
    let target = RateTarget::BitsPerPixel(1.0);
    let request = EncodeRequest::for_target(target);
    let outcome = encode_srgb8_to_target(w, h, &rgb, &request, target).unwrap();

    let source: [Vec<f32>; 3] = core::array::from_fn(|c| {
        rgb.chunks_exact(3)
            .map(|px| srgb_to_linear(f32::from(px[c]) / 255.0))
            .collect()
    });
    let reference = LinearRgbView::new(w, h, &source[0], &source[1], &source[2]).unwrap();

    let rendered = render(&outcome.plan).linear_rgb_at_depth(8);
    let ours = score_pair(
        reference,
        LinearRgbView::new(w, h, &rendered[0], &rendered[1], &rendered[2]).unwrap(),
    )
    .unwrap()
    .score;

    let (_, _, decoded) = decode_planes(&outcome.codestream);
    let decoded_linear: [Vec<f32>; 3] = decoded.map(|plane| {
        plane
            .iter()
            .map(|&v| srgb_to_linear(((v * 255.0).round().clamp(0.0, 255.0)) / 255.0))
            .collect()
    });
    let theirs = score_pair(
        reference,
        LinearRgbView::new(
            w,
            h,
            &decoded_linear[0],
            &decoded_linear[1],
            &decoded_linear[2],
        )
        .unwrap(),
    )
    .unwrap()
    .score;
    eprintln!("rendered score {ours:.6} vs emitted-and-decoded score {theirs:.6}");
    assert!((ours - theirs).abs() < 0.01, "{ours} vs {theirs}");
}

#[test]
fn consuming_linear_conversion_matches_the_reusable_buffer_path() {
    let (w, h) = (264u32, 256u32);
    let rgb = synthetic_rgb8(w, h, 13);
    let target = RateTarget::BitsPerPixel(1.0);
    let request = EncodeRequest::for_target(target);
    let outcome = encode_srgb8_to_target(w, h, &rgb, &request, target).unwrap();
    let rendered = render(&outcome.plan);

    let reusable = rendered.linear_rgb_at_depth(8);
    let consumed = rendered.into_linear_rgb_at_depth(8);
    assert_eq!(consumed, reusable);
}

#[test]
fn the_pixel_plan_split_reassembles_the_same_emission_plan() {
    let (w, h) = (264u32, 256u32);
    let rgb = synthetic_rgb8(w, h, 7);
    let frame = PreparedFrame::from_srgb8(w, h, &rgb).unwrap();
    let plan = plan_frame(&frame, &EncodeRequest::defaults()).unwrap();
    let pixels = validate_pixels(plan.plan().pixels()).expect("a validated plan's pixels validate");
    let reassembled = attach_and_validate_entropy(
        pixels,
        plan.plan().entropy.clone(),
        plan.plan().sections.clone(),
    )
    .expect("the original entropy reattaches");
    assert_eq!(reassembled, plan);
    assert_eq!(
        emit_codestream(&reassembled).unwrap().bytes,
        emit_codestream(&plan).unwrap().bytes
    );
}

/// Wall time of one 4 MP render under the production Balanced policy.
/// Run with `cargo test --release -p jpxl-plan-render --test parity -- --ignored --nocapture timing`.
#[test]
#[ignore = "timing aid; prints, does not assert"]
fn timing_four_megapixel_render() {
    use std::time::Instant;
    let (w, h) = (2400u32, 1800u32);
    let rgb = synthetic_rgb8(w, h, 11);
    let target = RateTarget::BitsPerPixel(1.0);
    let request = EncodeRequest::for_target(target);
    let outcome = encode_srgb8_to_target(w, h, &rgb, &request, target).unwrap();
    let pixels = outcome.plan.pixels();
    let mut renderer = PlanRenderer::new().expect("renderer");
    let serial = renderer.render(&pixels).expect("render");
    for threads in [1usize, 4] {
        let executor = jpxl_encode::EncodeResources::groups(threads).executor();
        let t = Instant::now();
        let (frame, timings) = renderer
            .render_timed(&pixels, Some(&executor))
            .expect("render");
        let rendered = t.elapsed();
        assert_eq!(frame, serial, "banded render must equal the serial render");
        let t = Instant::now();
        let _ = frame.linear_rgb_at_depth(8);
        eprintln!(
            "threads {threads}: render {rendered:?} {timings:?}, quantize+linearise {:?}",
            t.elapsed()
        );
    }
}
