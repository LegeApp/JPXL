//! Phase 7.1: how wrong is `residual_bits` about what a coefficient costs?
//!
//! Phase 7.0 made the HF quantizer rate-aware and produced the largest quality
//! movement on the track — butteraugli better in 20 of 28 cells — but regressed
//! SSIMULACRA2 in 24 of 28 by over-zeroing. The diagnosis was that the rate
//! proxy is blind to structure, so it zeroes uniformly instead of selectively.
//!
//! This measures exactly how blind. It runs the **real** coefficient walk
//! (`jpxl_encode::vardct::walk_frame`, the same one the writer and the census
//! use) over a planned frame and classifies every symbol the encoder actually
//! emits, against what `residual_bits` charges for the same coefficients.
//!
//! # What the walk really does (18181-1 I.4)
//!
//! Per varblock and channel:
//!
//! 1. Count `non_zeros` over the HF order positions and emit it as **one
//!    symbol**, in a context predicted from neighbouring blocks.
//! 2. Emit a coefficient token for every order position `num_blocks..=last`,
//!    where `last` is the position of the final nonzero. Each token's context
//!    depends on position, how many nonzeros remain, and whether the previous
//!    coefficient was nonzero.
//! 3. **Stop.** Positions after `last` cost nothing at all.
//!
//! `residual_bits(q)` charges `0` for a zero and `bit_length(|q|) + 1`
//! otherwise. So it is wrong in three distinct ways, and this harness sizes
//! each:
//!
//! * **Interior zeros are free to it but cost a real token** — every zero at a
//!   position before `last`.
//! * **The `non_zeros` symbol is invisible to it** — one per varblock-channel.
//! * **Truncation is invisible to it.** Zeroing the *last* nonzero does not
//!   just save that coefficient's token; it shortens the walk, so every
//!   interior zero back to the previous nonzero becomes free too. This is the
//!   selectivity Phase 7.0 lacked, and the reason it took texture everywhere
//!   instead of where a run would actually pay.
//!
//! Ignored by default and skipped without a reference image. Run it as:
//!
//! ```text
//! JPXL_RATE_REF=/path/to/ref.ppm JPXL_RATE_BPP=1 \
//!   cargo test -p jpxl-encode-policy --test rate_proxy_gap \
//!   -- --ignored --nocapture
//! ```

use jpxl_encode::vardct::{HfEventSink, walk_frame};
use jpxl_encode_policy::{EncodeRequest, PreparedFrame, RateTarget};

/// Classifies every symbol the walk emits, and records the trailing-run
/// structure that decides how much a truncation would save.
#[derive(Default)]
struct Classify {
    nonzeros_symbols: u64,
    /// Coefficient tokens whose value is zero: emitted, and charged nothing by
    /// `residual_bits`.
    interior_zero_tokens: u64,
    /// Coefficient tokens whose value is nonzero.
    nonzero_tokens: u64,
    /// `residual_bits` over the nonzero tokens — what the objective charges for
    /// this frame's HF coefficients today.
    proxy_bits: u64,
    /// Interior zeros sitting between the last nonzero and the one before it.
    /// Zeroing that last nonzero would make all of them free as well, so this
    /// is the truncation bonus the proxy cannot see.
    run_before_last: u64,
    /// Per-varblock-channel state.
    since_last_nonzero: u64,
    pending_run: u64,
}

impl Classify {
    fn end_block(&mut self) {
        // Whatever trailing zeros were pending were never emitted (the walk
        // stops at the last nonzero), so they are already free and are not
        // counted as interior. `pending_run` holds the zeros immediately before
        // the final nonzero, which is the truncation bonus.
        self.run_before_last += self.pending_run;
        self.pending_run = 0;
        self.since_last_nonzero = 0;
    }
}

impl HfEventSink for Classify {
    fn nonzeros(&mut self, _context: jpxl_encode::vardct::ids::PreContextId, _value: u32) {
        self.end_block();
        self.nonzeros_symbols += 1;
    }

    fn coefficient(&mut self, _context: jpxl_encode::vardct::ids::PreContextId, value: u32) {
        if value == 0 {
            self.interior_zero_tokens += 1;
            self.since_last_nonzero += 1;
        } else {
            self.nonzero_tokens += 1;
            // `value` is PackSigned; recover the magnitude for the proxy.
            let magnitude = value.div_ceil(2);
            self.proxy_bits += u64::from(32 - magnitude.max(1).leading_zeros()) + 1;
            self.pending_run = self.since_last_nonzero;
            self.since_last_nonzero = 0;
        }
    }
}

#[test]
#[ignore = "measurement harness: needs JPXL_RATE_REF"]
fn residual_bits_against_the_real_walk() {
    let Ok(reference) = std::env::var("JPXL_RATE_REF") else {
        eprintln!("skipped: set JPXL_RATE_REF to an 8-bit binary PPM");
        return;
    };
    let bpp: f64 = std::env::var("JPXL_RATE_BPP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);

    let bytes = std::fs::read(&reference).expect("read reference PPM");
    let img = jpxl_conformance::metrics::Image::from_ppm(&bytes).expect("parse PPM");
    assert_eq!(img.channels, 3, "the harness plans RGB");
    let rgb: Vec<u8> = img.samples.iter().map(|&s| s as u8).collect();
    let frame = PreparedFrame::from_srgb8(img.w, img.h, &rgb).expect("frame");

    let request = EncodeRequest::for_target(RateTarget::BitsPerPixel(bpp));
    let report = jpxl_encode_policy::encode_srgb8_to_target(
        img.w,
        img.h,
        &rgb,
        &request,
        RateTarget::BitsPerPixel(bpp),
    )
    .expect("a targeted encode");
    let _ = &frame;

    let plan = &report.plan;
    let geometry = plan.geometry().expect("the plan's geometry");
    let mut sink = Classify::default();
    walk_frame(plan.plan(), &geometry, &mut sink).expect("walk");
    sink.end_block();

    let coeff_tokens = sink.interior_zero_tokens + sink.nonzero_tokens;
    let emitted = coeff_tokens + sink.nonzeros_symbols;
    #[allow(clippy::cast_precision_loss, reason = "counts are far inside f64")]
    let pct = |n: u64, d: u64| -> f64 {
        if d == 0 {
            0.0
        } else {
            100.0 * n as f64 / d as f64
        }
    };

    println!("# Phase 7.1: residual_bits against the real coefficient walk");
    println!(
        "# reference {reference} at {bpp} bpp, {} bytes",
        report.codestream.len()
    );
    println!();
    println!("symbols the encoder actually emits");
    println!("  non_zeros symbols        {:>10}", sink.nonzeros_symbols);
    println!("  coefficient tokens       {:>10}", coeff_tokens);
    println!(
        "    of which zero-valued   {:>10}  ({:.1}% of coefficient tokens)",
        sink.interior_zero_tokens,
        pct(sink.interior_zero_tokens, coeff_tokens)
    );
    println!("    of which nonzero       {:>10}", sink.nonzero_tokens);
    println!("  TOTAL symbols            {:>10}", emitted);
    println!();
    println!("what residual_bits charges");
    println!(
        "  symbols it prices        {:>10}  ({:.1}% of emitted)",
        sink.nonzero_tokens,
        pct(sink.nonzero_tokens, emitted)
    );
    println!(
        "  symbols it prices at 0   {:>10}  ({:.1}% of emitted)",
        sink.interior_zero_tokens + sink.nonzeros_symbols,
        pct(sink.interior_zero_tokens + sink.nonzeros_symbols, emitted)
    );
    println!("  its total charge, bits   {:>10}", sink.proxy_bits);
    println!();
    println!("the truncation lever it cannot see");
    println!(
        "  zeros immediately before each block's last nonzero: {} ({:.2} per varblock-channel)",
        sink.run_before_last,
        if sink.nonzeros_symbols == 0 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss, reason = "counts are small")]
            let v = sink.run_before_last as f64 / sink.nonzeros_symbols as f64;
            v
        }
    );
    println!("  zeroing one block's last nonzero frees its own token PLUS those zeros,");
    println!("  which residual_bits prices as a flat saving of ~2 bits regardless.");
}
