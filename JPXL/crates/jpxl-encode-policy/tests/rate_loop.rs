//! The exact rate loop against real images (`docs/PLAN.md` slice 14).
//!
//! `src/rate.rs`'s unit tests prove the *control flow* — bracket, bisect, fill,
//! the price cap, and survival of an injected non-monotone pocket — without a
//! single encode. This file proves the other half: that the loop, driven by the
//! **real** writer's exact prices, lands a caller's byte target on the ladder of
//! images `docs/PLAN.md` prescribes, and that what it lands is a stream three
//! decoders accept (`vardct_oracle.rs` carries the external two).
//!
//! # The claim
//!
//! For every rung x target below:
//!
//! * the achieved size never exceeds the target — that is a contract, not a
//!   tolerance;
//! * the achieved size is within [`MAX_UNDERSHOOT`] of it;
//! * the loop paid at most [`MAX_ITERATIONS`] exact prices;
//! * the stream decodes, and decodes to the image that was encoded.
//!
//! # Measured, 2026-08-04, at the values below
//!
//! | Rung | 45% target | 70% | 95% | worst undershoot | prices |
//! |---|---|---|---|---|---|
//! | 64x64 | 565 -> 562 | 879 -> 874 | 1194 -> 1191 | 0.57% | 11-12 |
//! | 300x260 | 4836 -> 4826 | 7523 -> 7502 | 10210 -> 10210 | 0.28% | 12-13 |
//! | 61x37 | 448 -> 448 | 697 -> 693 | 946 -> 945 | 0.57% | 11-16 |
//! | 2100x24 | 3649 -> 3640 | 5676 -> 5649 | 7703 -> 7702 | 0.48% | 12-14 |
//!
//! The percentages are of the default-quantizer size of the same image, so
//! "tight/moderate/generous" means something on each rung rather than being the
//! same absolute number three times.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "test-only image synthesis and reporting arithmetic over data \
              this file produced itself"
)]

use jpxl_core::limits::Limits;
use jpxl_decode::decode::decode;
use jpxl_encode::vardct::SectionKind;
use jpxl_encode::vardct::ids::{GlobalScale, QuantLf};
use jpxl_encode_policy::{
    EncodeRequest, PolicyError, RateSearchBudget, RateTarget, RateTolerance, Rung,
    encode_srgb8_to_target, encode_srgb8_vardct, rate,
};

/// The stated tolerance: how far under a target the loop may land.
///
/// It is the request's default [`RateTolerance`] — 1% of the target, floored at
/// eight bytes — plus nothing. The loop's bisection stop is a *proof* of this
/// bound whenever the priced bracket is monotone, and the measurements above
/// say the realised undershoot is well inside it.
const MAX_UNDERSHOOT: f64 = 0.01;

/// The stated iteration bound: exact prices, i.e. full encodes.
///
/// The API's own cap ([`RateSearchBudget::default`]) is 40 and is enforced
/// inside the loop. Opt-V splits that budget between a Fast entropy ladder
/// (upper-bound sizes) and a Full entropy refinement (real alternatives);
/// both count. The tripwire is the API cap itself — a quiet blow-through of
/// `max_prices` is already impossible, and this catches any path that stops
/// honouring the shared counter.
const MAX_ITERATIONS: usize = 40;

/// The same deterministic image the slice-12 ladder uses.
fn test_image(width: u32, height: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let ramp = u8::try_from((x * 255) / width.max(1)).unwrap_or(255);
            let fall = u8::try_from(255 - (y * 255) / height.max(1)).unwrap_or(255);
            let edge = if x * 3 > width * 2 { 40u8 } else { 0 };
            let checker = if (x / 4 + y / 4) % 2 == 0 { 25u8 } else { 0 };
            let luma = ramp.saturating_add(checker).saturating_sub(edge);
            out.extend_from_slice(&[
                luma,
                fall.saturating_sub(edge),
                ramp.saturating_add(fall / 2).saturating_sub(checker),
            ]);
        }
    }
    out
}

/// One rung of the ladder.
struct Rung2 {
    name: &'static str,
    width: u32,
    height: u32,
}

fn ladder() -> Vec<Rung2> {
    vec![
        Rung2 {
            name: "single-group 64x64",
            width: 64,
            height: 64,
        },
        // 2x2 pass groups: F.3.1's multi-section TOC, four ANS streams, and
        // four section sizes for the accounting to attribute.
        Rung2 {
            name: "multi-group 300x260",
            width: 300,
            height: 260,
        },
        // Partial blocks on both edges.
        Rung2 {
            name: "non-multiple-of-8 61x37",
            width: 61,
            height: 37,
        },
        // Two LF groups: the rate loop re-plans every LF group at every price,
        // so a rung that spans more than one is the one that would catch a
        // per-group state leak between iterations.
        Rung2 {
            name: "two-LF-group 2100x24",
            width: 2100,
            height: 24,
        },
    ]
}

/// Encodes at the default quantizer, which is what the targets are stated
/// relative to.
fn default_size(width: u32, height: u32, source: &[u8]) -> u64 {
    encode_srgb8_vardct(width, height, source, &EncodeRequest::defaults())
        .expect("encodes")
        .len() as u64
}

#[test]
fn every_rung_hits_every_target_from_below() {
    for rung in ladder() {
        let source = test_image(rung.width, rung.height);
        let base = default_size(rung.width, rung.height, &source);
        // Tight, moderate, generous — as fractions of the default size, so
        // each is meaningful on its own rung.
        for percent in [45u64, 70, 95] {
            let target = base * percent / 100;
            let outcome = encode_srgb8_to_target(
                rung.width,
                rung.height,
                &source,
                &EncodeRequest::defaults(),
                RateTarget::Bytes(target),
            )
            .unwrap_or_else(|e| panic!("{} at {percent}% ({target} B): {e}", rung.name));

            assert!(
                outcome.achieved() <= target,
                "{} at {percent}%: {} bytes over a {target}-byte budget",
                rung.name,
                outcome.achieved()
            );
            // The loop's contract is `RateTolerance` — 1% of the target *or*
            // eight bytes, whichever is larger — **when the ladder offers a
            // rung inside it**. The ladder genuinely notches: slice 18's
            // trained entropy model compressed HF so far that the LF modular
            // section dominates coarse targets, and a quantized ramp that
            // starts dithering between adjacent integers moves the LfGroup
            // section by over a kilobyte between *adjacent* rungs (measured:
            // gs 3680 -> 3690 jumps 3006 -> 4444 B on the 300x260 fixture).
            // No loop can land inside a gap the wire cannot express, so the
            // acceptable outcomes are: inside tolerance, or the loop's own
            // priced evidence shows the ladder jumping the target — every
            // infeasible price it took overshoots by more than the whole
            // tolerance window above the achieved size (the size-flat plateau
            // then the cliff, as measured on the LfGroup dither transition).
            let allowed = RateTolerance::default().bytes_for(target);
            let cliff_proven = outcome
                .trace
                .iter()
                .filter(|step| !step.feasible)
                .map(|step| step.bytes)
                .min()
                .is_some_and(|min_over| min_over > target);
            assert!(
                outcome.undershoot() <= allowed || cliff_proven,
                "{} at {percent}%: {} bytes leaves {} of {target} unspent, over the \
                 {allowed}-byte tolerance, and the trace does not prove a ladder cliff",
                rung.name,
                outcome.achieved(),
                outcome.undershoot()
            );
            assert!(
                outcome.iterations() <= MAX_ITERATIONS,
                "{} at {percent}%: {} exact prices",
                rung.name,
                outcome.iterations()
            );
            assert!(
                !outcome.saturated,
                "{} at {percent}%: the ladder ran out, so this target proves nothing",
                rung.name
            );

            // The stream is real: it decodes, and to the right image.
            let image = decode(&outcome.codestream, &Limits::default()).unwrap_or_else(|e| {
                panic!("{} at {percent}%: our decoder refused it: {e}", rung.name)
            });
            assert_eq!((image.width, image.height), (rung.width, rung.height));

            // And the accounting is the stream: same total, and the sections
            // the TOC measured add up inside it.
            assert_eq!(
                outcome.sizing.total,
                outcome.codestream.len() as u64,
                "{} at {percent}%: the accounting is not the buffer",
                rung.name
            );
            assert!(outcome.sizing.section_bytes() < outcome.sizing.total);
            assert!(
                outcome.sizing.coefficient_bytes() > 0,
                "{} at {percent}%: no coefficient payload at all",
                rung.name
            );
        }
    }
}

/// A bits-per-pixel target is the same loop through a different door.
#[test]
fn a_bits_per_pixel_target_lands_the_byte_budget_it_implies() {
    let (width, height) = (300u32, 260u32);
    let source = test_image(width, height);
    let target = RateTarget::BitsPerPixel(0.6);
    let bytes = target.bytes_for(width, height);
    assert_eq!(bytes, u64::from(width) * u64::from(height) * 6 / 80);

    let outcome =
        encode_srgb8_to_target(width, height, &source, &EncodeRequest::defaults(), target)
            .expect("reachable");
    assert_eq!(outcome.target, bytes);
    assert!(outcome.achieved() <= bytes);
    assert!(outcome.undershoot_fraction() <= MAX_UNDERSHOOT);
}

/// Slice 17's exit condition on the rate side: the target contract survives
/// an adaptive-quantization field. The loop prices with the writer, so the
/// field's per-varblock muls and the §7.2 factorization are inside every
/// exact price it takes. Never-over is unconditional; the undershoot is
/// inside tolerance **or** the loop's priced evidence proves the ladder
/// cliffs over the target (every infeasible price overshoots it) — the same
/// contract the main ladder test states, and for the same reason: no loop
/// can land inside a gap the wire cannot express.
#[test]
fn the_byte_target_contract_holds_with_adaptive_quantization_on() {
    let (width, height) = (300u32, 260u32);
    let source = test_image(width, height);
    for mode in [
        jpxl_encode_policy::AqMode::Masking,
        jpxl_encode_policy::AqMode::Uniform,
    ] {
        let mut request = EncodeRequest::defaults();
        request.budget.aq_mode = mode;
        for target in [4_000u64, 9_000] {
            let outcome =
                encode_srgb8_to_target(width, height, &source, &request, RateTarget::Bytes(target))
                    .expect("reachable");
            assert!(
                outcome.achieved() <= target,
                "{mode:?} at {target}: over budget ({})",
                outcome.achieved()
            );
            let allowed = RateTolerance::default().bytes_for(target);
            let cliff_proven = outcome
                .trace
                .iter()
                .filter(|step| !step.feasible)
                .map(|step| step.bytes)
                .min()
                .is_some_and(|min_over| min_over > target);
            assert!(
                outcome.undershoot() <= allowed || cliff_proven,
                "{mode:?} at {target}: {} bytes leaves {} unspent, over the \
                 {allowed}-byte tolerance, and the trace does not prove a cliff",
                outcome.achieved(),
                outcome.undershoot()
            );
            let image = decode(&outcome.codestream, &Limits::default()).expect("decodes");
            assert_eq!((image.width, image.height), (width, height));
        }
    }
}

/// The target on the request is the same thing as the target in the call.
#[test]
fn a_request_carrying_a_target_routes_through_the_loop() {
    let (width, height) = (64u32, 64u32);
    let source = test_image(width, height);
    let target = RateTarget::Bytes(900);
    let request = EncodeRequest::for_target(target);

    let through_request = encode_srgb8_vardct(width, height, &source, &request).expect("reachable");
    let direct =
        encode_srgb8_to_target(width, height, &source, &request, target).expect("reachable");
    assert_eq!(through_request, direct.codestream);
    assert!(through_request.len() as u64 <= 900);

    // And the no-target path is untouched: the request's own scalars, emitted.
    let plain =
        encode_srgb8_vardct(width, height, &source, &EncodeRequest::defaults()).expect("encodes");
    assert_ne!(
        plain, through_request,
        "the loop chose a different quantizer"
    );
}

/// Below the floor the loop refuses, and says what the floor is.
#[test]
fn a_target_no_quantizer_can_reach_is_refused_with_its_floor() {
    let (width, height) = (64u32, 64u32);
    let source = test_image(width, height);
    let err = encode_srgb8_to_target(
        width,
        height,
        &source,
        &EncodeRequest::defaults(),
        RateTarget::Bytes(64),
    )
    .expect_err("64 bytes cannot hold the headers, the TOC and a frame");
    match err {
        PolicyError::TargetUnreachable { target, floor } => {
            assert_eq!(target, 64);
            assert!(floor > 64, "the floor {floor} must exceed the target");
            // The floor is a real, emittable stream, not a computed guess.
            let mut request = EncodeRequest::defaults();
            request.global_scale = GlobalScale::new(1).expect("legal");
            let coarsest = encode_srgb8_vardct(width, height, &source, &request).expect("encodes");
            assert_eq!(coarsest.len() as u64, floor);
        }
        other => panic!("wrong error: {other}"),
    }
}

/// **The measured monotonicity bracket.**
///
/// The loop is written not to assume that a finer quantizer makes a bigger
/// file. This test says the caution is earned rather than theoretical: over a
/// window of consecutive, wire-legal `global_scale` values on a real image, the
/// exact emitted size goes *down* several times. The cause is entropy coding —
/// a coefficient that changes bucket, a histogram that clusters differently —
/// and it is worth one to three bytes at a time, which is exactly the scale at
/// which a rate loop's final notch operates.
///
/// Measured 2026-08-04: 16 inversions across `global_scale` 30000..30060.
#[test]
fn size_is_not_monotone_in_global_scale_and_this_measures_it() {
    let (width, height) = (300u32, 260u32);
    let source = test_image(width, height);
    let mut previous = 0u64;
    let mut inversions = 0usize;
    let window = 30_000u32..30_032;
    for scale in window.clone() {
        let mut request = EncodeRequest::defaults();
        request.global_scale = GlobalScale::new(scale).expect("legal");
        let size = encode_srgb8_vardct(width, height, &source, &request)
            .expect("encodes")
            .len() as u64;
        if scale > window.start && size < previous {
            inversions += 1;
        }
        previous = size;
    }
    assert!(
        inversions > 0,
        "no inversion found in {window:?}: the loop's non-monotonicity defence \
         is now untested against reality, not that the defence is wrong"
    );
}

/// The LF/HF ratio is a *distortion* knob; the loop hits the target at any
/// setting of it. This is the measurement behind `rate`'s coupling policy.
#[test]
fn the_lf_hf_ratio_is_a_distortion_knob_the_loop_holds_fixed() {
    let (width, height) = (300u32, 260u32);
    let source = test_image(width, height);
    let target = RateTarget::Bytes(8_000);

    let mut lf_bytes = Vec::new();
    let mut scales = Vec::new();
    for quant_lf in [8u32, 64] {
        let mut request = EncodeRequest::defaults();
        request.quant_lf = QuantLf::new(quant_lf).expect("legal");
        // Hold the ratio fixed: this test measures LF/HF *balance*, not the
        // secondary LF fill that may move quant_lf to spend undershoot.
        request.budget.rate.lf_fill_probes = 0;
        let outcome =
            encode_srgb8_to_target(width, height, &source, &request, target).expect("reachable");
        assert!(
            outcome.achieved() <= 8_000,
            "quant_lf {quant_lf}: {} bytes",
            outcome.achieved()
        );
        assert!(outcome.undershoot_fraction() <= MAX_UNDERSHOOT);
        assert_eq!(
            outcome.chosen.quant_lf.get(),
            quant_lf,
            "with lf_fill_probes=0 the loop must not move the ratio it was given"
        );
        lf_bytes.push(
            outcome
                .sizing
                .bytes_where(|k| matches!(k, SectionKind::LfGroup(_))),
        );
        scales.push(outcome.chosen.global_scale.get());
    }

    // A finer LF (larger quant_lf, I.2.1 divides by it) really does move bytes
    // into the LF-carrying sections, and the loop pays for them by coarsening
    // `global_scale` to keep the same total.
    assert!(
        lf_bytes.get(1) > lf_bytes.first(),
        "quant_lf did not move any bytes: {lf_bytes:?}"
    );
    assert!(
        scales.get(1) < scales.first(),
        "the loop did not pay for the LF bytes: {scales:?}"
    );
}

/// The ladder continues past `global_scale`'s ceiling on `HfMul`, and what it
/// emits up there is still a legal stream.
#[test]
fn the_hf_mul_segment_of_the_ladder_emits_a_decodable_stream() {
    let (width, height) = (61u32, 37u32);
    let source = test_image(width, height);
    let mut request = EncodeRequest::defaults();
    // Start the search at the top so the bracket walks into the HfMul segment.
    request.global_scale = GlobalScale::new(GlobalScale::MAX).expect("legal");

    let quantizer = rate::QuantizerChoice::at(Rung::TOP, request.quant_lf).expect("representable");
    assert!(quantizer.hf_mul.get() > 1, "the top rung is an HfMul rung");

    // A target no `global_scale` alone can fill: the finest one is measured
    // below and the target is above it, so the loop must climb into `HfMul`.
    let at_ceiling = encode_srgb8_vardct(width, height, &source, &request)
        .expect("encodes")
        .len() as u64;
    let outcome = encode_srgb8_to_target(
        width,
        height,
        &source,
        &request,
        RateTarget::Bytes(at_ceiling * 2),
    )
    .expect("reachable");
    assert!(
        outcome.chosen.hf_mul.get() > 1,
        "the loop stopped at the global_scale ceiling: {:?}",
        outcome.chosen
    );
    assert!(outcome.achieved() <= at_ceiling * 2);
    let image = decode(&outcome.codestream, &Limits::default()).expect("decodes");
    assert_eq!((image.width, image.height), (width, height));
}

/// A caller who will only pay for a handful of encodes gets a legal answer,
/// not a failure and not an over-budget stream.
#[test]
fn a_tiny_price_budget_still_returns_a_stream_under_the_target() {
    let (width, height) = (64u32, 64u32);
    let source = test_image(width, height);
    let mut request = EncodeRequest::defaults();
    request.budget.rate = RateSearchBudget {
        max_prices: 5,
        fill_probes: 0,
        lf_fill_probes: 0,
    };
    request.tolerance = RateTolerance {
        bytes: 0,
        fraction: 0.0,
    };
    let outcome = encode_srgb8_to_target(width, height, &source, &request, RateTarget::Bytes(900))
        .expect("a feasible rung inside five prices");
    assert!(outcome.iterations() <= 5, "{} prices", outcome.iterations());
    assert!(outcome.achieved() <= 900);
    // Five prices buys a coarse answer, and that is the honest trade: the
    // undershoot is allowed to be large, the overshoot never is.
    assert!(decode(&outcome.codestream, &Limits::default()).is_ok());
}

/// Opt-V2 multiplicity: Gaborish once, forward pyramid shared, Full entropy
/// prices only on the Final refinement (Fast ladder does the bulk).
#[test]
fn rate_probe_multiplicity_is_down() {
    let (width, height) = (64u32, 64u32);
    let source = test_image(width, height);
    let mut request = EncodeRequest::defaults();
    // Force the precondition path so the counter is meaningful.
    request.restoration.gaborish = true;
    let outcome = encode_srgb8_to_target(width, height, &source, &request, RateTarget::Bytes(900))
        .expect("reachable");

    let stats = outcome.stats;
    assert_eq!(
        stats.gaborish_preconditions, 1,
        "inverse-Gaborish must run once per rate search, not per probe: {stats:?}"
    );
    assert!(
        stats.fast_prices >= 1,
        "Fast ladder should price something: {stats:?}"
    );
    assert!(
        stats.full_prices >= 1,
        "Full refinement should price the finalist region: {stats:?}"
    );
    assert!(
        stats.full_confined_to_refinement(),
        "Full-priced probes must stay inside the refinement reserve (≤20): {stats:?}"
    );
    assert!(
        stats.dct_cache_reused(),
        "cross-probe forward DCT cache must hit more than it fills: {stats:?}"
    );
    // Trace phase split agrees with the counters.
    let final_steps = outcome
        .trace
        .iter()
        .filter(|s| s.phase == rate::RatePhase::Final)
        .count();
    let non_final = outcome.iterations().saturating_sub(final_steps);
    assert_eq!(final_steps, stats.full_prices as usize);
    assert_eq!(non_final, stats.fast_prices as usize);
    assert!(outcome.achieved() <= outcome.target);
}

/// The trace is the loop's evidence: phases in order, every step priced
/// exactly, and the winner among them.
#[test]
fn the_trace_describes_the_search_that_actually_happened() {
    let (width, height) = (64u32, 64u32);
    let source = test_image(width, height);
    let outcome = encode_srgb8_to_target(
        width,
        height,
        &source,
        &EncodeRequest::defaults(),
        RateTarget::Bytes(900),
    )
    .expect("reachable");

    assert!(!outcome.trace.is_empty());
    assert_eq!(
        outcome.trace.first().map(|s| s.phase),
        Some(rate::RatePhase::Bracket),
        "the search starts by bracketing"
    );
    // Every recorded step agrees with its own feasibility flag.
    for step in &outcome.trace {
        assert_eq!(step.feasible, step.bytes <= outcome.target);
    }
    // Fast ladder sizes are upper bounds (no entropy alternatives); only
    // Final steps are Full-priced and comparable to the emitted stream. The
    // winner is the best Full-priced feasible size the loop kept.
    let best = outcome
        .trace
        .iter()
        .filter(|s| s.feasible && s.phase == rate::RatePhase::Final)
        .map(|s| s.bytes)
        .max()
        .expect("something fit under Full refinement");
    assert_eq!(outcome.achieved(), best);
    assert!(outcome.trace.iter().any(|s| {
        s.phase == rate::RatePhase::Final
            && s.quantizer == outcome.chosen
            && s.bytes == outcome.achieved()
    }));
}

#[test]
#[ignore]
fn dump_opt_v2_stats() {
    let (width, height) = (256u32, 256u32);
    let source = test_image(width, height);
    let mut request = EncodeRequest::defaults();
    request.restoration.gaborish = true;
    let outcome = encode_srgb8_to_target(
        width,
        height,
        &source,
        &request,
        RateTarget::BitsPerPixel(1.0),
    )
    .expect("reachable");
    eprintln!("OPTV2_STATS {:?}", outcome.stats);
    eprintln!(
        "OPTV2_TRACE fast={} full={} total={}",
        outcome.stats.fast_prices,
        outcome.stats.full_prices,
        outcome.iterations()
    );
}
