//! Phase 7.1: does the truncation pass's claimed saving match the real walk?
//!
//! `HfQuantizer::truncate_trailing` prices dropping a block's last nonzero at
//! its own token plus the interior zeros it exposes. That arithmetic is only
//! worth anything if it agrees with what 18181-1 I.4 actually stops emitting,
//! so this checks it against `jpxl_encode::vardct::walk_frame` — the writer's
//! own walk — rather than against a restatement of the same assumption.
//!
//! The check is a difference: encode the same image with the pass off and on,
//! count the symbols each stream's walk emits, and confirm the reduction is
//! consistent with the pass having removed trailing nonzeros and nothing else.

use jpxl_encode::vardct::{HfEventSink, walk_frame};
use jpxl_encode_policy::{EncodeRequest, QuantizerChoiceMode, RateTarget};

#[derive(Default)]
struct Counts {
    nonzeros_symbols: u64,
    zero_tokens: u64,
    nonzero_tokens: u64,
}

impl HfEventSink for Counts {
    fn nonzeros(&mut self, _c: jpxl_encode::vardct::ids::PreContextId, _v: u32) {
        self.nonzeros_symbols += 1;
    }
    fn coefficient(&mut self, _c: jpxl_encode::vardct::ids::PreContextId, v: u32) {
        if v == 0 {
            self.zero_tokens += 1;
        } else {
            self.nonzero_tokens += 1;
        }
    }
}

fn walk_counts(rgb: &[u8], w: u32, h: u32, mode: QuantizerChoiceMode) -> (Counts, usize) {
    let target = RateTarget::BitsPerPixel(1.0);
    let mut request = EncodeRequest::for_target(target);
    request.quantizer_choice = mode;
    // Pin unit lambda so this measures the truncation pass itself, not Phase
    // 7.2's calibrated scale (which also moves cover selection).
    request.lambda_scale = 1.0;
    let report = jpxl_encode_policy::encode_srgb8_to_target(w, h, rgb, &request, target)
        .expect("a targeted encode");
    let geometry = report.plan.geometry().expect("geometry");
    let mut sink = Counts::default();
    walk_frame(report.plan.plan(), &geometry, &mut sink).expect("walk");
    (sink, report.codestream.len())
}

/// Deterministic mixed content: smooth gradients with a textured corner, so
/// blocks span the sparse-to-dense range the pass has to discriminate over.
fn mixed_rgb(width: u32, height: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let smooth = u8::try_from((x * 255) / width.max(1)).unwrap_or(u8::MAX);
            let textured = if x > width / 2 && y > height / 2 {
                let mut h =
                    (x as u64).wrapping_mul(0x9E37_79B9) ^ (y as u64).wrapping_mul(0x85EB_CA6B);
                h ^= h >> 29;
                (h & 0x3F) as u8
            } else {
                0
            };
            out.push(smooth.saturating_add(textured));
            out.push(u8::try_from((y * 255) / height.max(1)).unwrap_or(u8::MAX));
            out.push(smooth.wrapping_sub(textured));
        }
    }
    out
}

#[test]
fn truncation_removes_only_trailing_nonzeros_and_shortens_the_walk() {
    let (w, h) = (256u32, 256u32);
    let rgb = mixed_rgb(w, h);

    let (off, off_bytes) = walk_counts(&rgb, w, h, QuantizerChoiceMode::Nearest);
    let (on, on_bytes) = walk_counts(&rgb, w, h, QuantizerChoiceMode::TrailingTruncation);

    eprintln!(
        "off: {} nonzeros-symbols, {} zero tokens, {} nonzero tokens, {off_bytes} B",
        off.nonzeros_symbols, off.zero_tokens, off.nonzero_tokens
    );
    eprintln!(
        "on : {} nonzeros-symbols, {} zero tokens, {} nonzero tokens, {on_bytes} B",
        on.nonzeros_symbols, on.zero_tokens, on.nonzero_tokens
    );

    // The pass must actually fire, or every other assertion here is vacuous.
    assert!(
        on.nonzero_tokens < off.nonzero_tokens,
        "the pass removed no nonzero coefficient at all"
    );

    // Same varblocks, so the same number of `non_zeros` symbols: the pass
    // changes coefficients, never the cover.
    assert_eq!(
        on.nonzeros_symbols, off.nonzeros_symbols,
        "the truncation pass must not change the block partition"
    );

    // The claim being tested. Dropping a trailing nonzero removes its own
    // token AND every interior zero it was keeping inside the walk, so zero
    // tokens must fall too — a pass that only removed the nonzero itself would
    // leave this count unchanged, and one that zeroed mid-run would *raise* it.
    assert!(
        on.zero_tokens < off.zero_tokens,
        "zero tokens did not fall ({} -> {}), so the pass is not truncating the \
         walk — it is zeroing coefficients without shortening it",
        off.zero_tokens,
        on.zero_tokens
    );

    // And the whole point: fewer symbols on the wire.
    let off_total = off.nonzeros_symbols + off.zero_tokens + off.nonzero_tokens;
    let on_total = on.nonzeros_symbols + on.zero_tokens + on.nonzero_tokens;
    assert!(
        on_total < off_total,
        "total emitted symbols did not fall: {off_total} -> {on_total}"
    );
    eprintln!(
        "symbols {off_total} -> {on_total} ({:.1}% fewer); zero tokens {:.1}% fewer",
        100.0 * (off_total - on_total) as f64 / off_total as f64,
        100.0 * (off.zero_tokens - on.zero_tokens) as f64 / off.zero_tokens as f64
    );
}

#[test]
fn fixed_quantizer_defaults_stay_nearest_after_target_rate_promotion() {
    // Phase 7.2 promoted trailing truncation on `for_target` only. The
    // fixed-quantizer defaults path must stay nearest so Contract A
    // fingerprints and non-rate encodes do not move.
    let (w, h) = (128u32, 128u32);
    let rgb = mixed_rgb(w, h);
    let target = RateTarget::BitsPerPixel(1.0);
    let promoted = EncodeRequest::for_target(target);
    assert_eq!(
        promoted.quantizer_choice,
        QuantizerChoiceMode::TrailingTruncation,
        "target-rate policy carries Phase 7.2's promoted rule"
    );
    let mut nearest = promoted;
    nearest.quantizer_choice = QuantizerChoiceMode::Nearest;
    nearest.lambda_scale = 1.0;
    let a = jpxl_encode_policy::encode_srgb8_to_target(w, h, &rgb, &promoted, target)
        .expect("encode")
        .codestream;
    let b = jpxl_encode_policy::encode_srgb8_to_target(w, h, &rgb, &nearest, target)
        .expect("encode")
        .codestream;
    assert_ne!(
        a, b,
        "the promoted target-rate path must differ from nearest; if equal the promotion is not live"
    );
    assert_eq!(
        EncodeRequest::defaults().quantizer_choice,
        QuantizerChoiceMode::Nearest,
        "fixed-quantizer defaults must stay nearest"
    );
}
