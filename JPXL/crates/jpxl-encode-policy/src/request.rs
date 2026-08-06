//! What the caller asks for, and how hard the encoder is allowed to work.
//!
//! `Encoder-plan1.md` §12: a quality target and an effort target are different
//! things, and effort is a *budget* rather than an integer examined all over
//! the encoder. Both are declared here at the stage boundary; no field of
//! [`SearchBudget`] is ever read inside a kernel.

use jpxl_encode::vardct::ids::{GlobalScale, HfMul, QuantLf};
use jpxl_encode::vardct::plan::RestorationDecision;

/// How the cover search explores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoverMode {
    /// One DCT8x8 per atom; no search at all (milestone 1 and 2).
    FixedDct8x8,
    /// The quadtree solver over the square transforms (milestone 6): each
    /// aligned 32x32 region chooses between one transform and four
    /// sub-quadrants by §4.3's rate-distortion objective within the
    /// hierarchy.
    ///
    /// The default since slice 18b: on smooth content it strictly dominates
    /// the fixed cover at matched quality (the slice-16 exit evidence), on
    /// busy content it declines to merge and ties it, and every stream shape
    /// it emits has external-decoder parity evidence.
    #[default]
    Hierarchical,
}

/// How much of the file the caller is willing to spend.
///
/// Two spellings of one thing: [`RateTarget::bytes_for`] turns either into the
/// byte budget the rate loop actually searches against, because the loop's
/// exact prices are byte counts and converting once at the edge is what keeps
/// a rounding rule from appearing in the middle of a search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RateTarget {
    /// An exact byte budget for the whole codestream, headers and TOC
    /// included.
    Bytes(u64),
    /// Bits per pixel of the frame's own `width * height`.
    BitsPerPixel(f64),
}

impl RateTarget {
    /// The byte budget for a `width x height` frame.
    ///
    /// Bits per pixel rounds **down**: the target is a ceiling the loop must
    /// not cross, so a fractional byte is not available to spend.
    #[must_use]
    pub fn bytes_for(self, width: u32, height: u32) -> u64 {
        match self {
            Self::Bytes(bytes) => bytes,
            Self::BitsPerPixel(bpp) => {
                let pixels = u64::from(width) * u64::from(height);
                #[allow(
                    clippy::cast_precision_loss,
                    reason = "pixel counts stay far inside f64's exact integer range"
                )]
                let bits = bpp.max(0.0) * pixels as f64;
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "non-negative and floored immediately below; \
                              non-finite is mapped to zero first"
                )]
                let bytes = if bits.is_finite() {
                    (bits / 8.0).floor()
                } else {
                    0.0
                } as u64;
                bytes
            }
        }
    }
}

/// How far under the target the loop may stop.
///
/// The loop never exceeds the target — that is not a tolerance, it is the
/// contract. This is the *undershoot* it is allowed to leave unspent, and it
/// is what buys back encodes: bisection stops as soon as the priced bracket
/// proves that no candidate inside it is worth more than this many bytes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RateTolerance {
    /// A floor in bytes, for targets small enough that a fraction is nothing.
    pub bytes: u64,
    /// A fraction of the target.
    pub fraction: f64,
}

impl RateTolerance {
    /// The slack for one target: the larger of the two spellings.
    #[must_use]
    pub fn bytes_for(self, target: u64) -> u64 {
        #[allow(
            clippy::cast_precision_loss,
            reason = "byte counts stay far inside f64's exact integer range"
        )]
        let scaled = self.fraction.max(0.0) * target as f64;
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "non-negative and floored; non-finite is mapped to zero"
        )]
        let scaled = if scaled.is_finite() {
            scaled.floor()
        } else {
            0.0
        } as u64;
        self.bytes.max(scaled)
    }
}

impl Default for RateTolerance {
    /// One percent of the target, never less than eight bytes.
    ///
    /// Eight bytes is roughly one TOC entry: below that the loop would be
    /// paying a whole encode to chase a difference smaller than the framing.
    fn default() -> Self {
        Self {
            bytes: 8,
            fraction: 0.01,
        }
    }
}

/// How many exact prices the rate loop may pay.
///
/// Each price is a full encode, so this is the effort knob that matters most
/// in milestone 4. The defaults are sized from the ladder: bracketing is
/// geometric and bisection is binary, so both are logarithmic in a ladder of
/// about 2^17 rungs, and a real search settles in well under half the cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RateSearchBudget {
    /// Hard cap on exact prices — bracket, bisect, fill and LF fill together.
    pub max_prices: u32,
    /// How many one-notch refinements the discrete budget fill may try.
    pub fill_probes: u32,
    /// Secondary `quant_lf` probes after the ladder settles (0 disables).
    ///
    /// Used at LF-dominated coarse targets where adjacent `global_scale`
    /// rungs cliff; leave 0 when the caller needs the request's `quant_lf`
    /// held fixed as a distortion knob.
    pub lf_fill_probes: u32,
}

impl Default for RateSearchBudget {
    fn default() -> Self {
        Self {
            max_prices: 40,
            fill_probes: 4,
            lf_fill_probes: 8,
        }
    }
}

/// The effort budget.
///
/// Milestone 1 exposes only what exists. Every field `Encoder-plan1.md` §12
/// lists — beam width, retained plans, quantization points per candidate,
/// iteration counts — joins this struct with the milestone that first reads
/// it, so that a budget field never means "ignored".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SearchBudget {
    /// How the cover search explores.
    pub cover_mode: CoverMode,
    /// How adaptive quantization points its field (default Masking).
    pub aq_mode: crate::field::AqMode,
    /// What the rate loop may spend (milestone 4).
    pub rate: RateSearchBudget,
}

/// One encode request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EncodeRequest {
    /// I.2's `global_scale`.
    ///
    /// With a [`target`](Self::target) set this is only the rate loop's
    /// *starting point*; the loop chooses the value that is emitted.
    pub global_scale: GlobalScale,
    /// I.2's `quant_lf`.
    ///
    /// The LF/HF ratio, held fixed by the rate loop — see
    /// [`crate::rate`] for the coupling policy.
    pub quant_lf: QuantLf,
    /// The constant per-varblock multiplier, until adaptive quantization
    /// (milestone 7) computes one per varblock.
    ///
    /// With a [`target`](Self::target) set the rate loop owns this too: it is
    /// how the ladder continues past `global_scale`'s ceiling.
    pub hf_mul: HfMul,
    /// F.2's `group_size_shift`.
    pub group_size_shift: u32,
    /// The effort budget.
    pub budget: SearchBudget,
    /// A size target, if the caller wants one chosen for them.
    ///
    /// `None` is the milestone-2 path, unchanged: the three scalars above are
    /// emitted exactly as given.
    pub target: Option<RateTarget>,
    /// How much of the target the loop may leave unspent.
    pub tolerance: RateTolerance,
    /// J.1 restoration-filter decisions written into the frame header.
    ///
    /// Default is all off (the unfiltered R-D baseline). When
    /// [`RestorationDecision::gaborish`] is set, the planner inverse-Gaborish
    /// preconditions the XYB planes before DCT so decoder J.3 restores the
    /// intended samples. `epf_iters > 0` is signalled on the wire but has no
    /// encoder-side inverse yet (deeper EPF is a later filter-planning item).
    pub restoration: RestorationDecision,
    /// Coarse section-parallelism policy for emission (Opt-P).
    ///
    /// Default is [`jpxl_encode::EncodeResources::auto`]. Rate-loop intermediate
    /// prices stay serial for predictability; Final Full emits use this budget.
    pub resources: jpxl_encode::EncodeResources,
}

impl EncodeRequest {
    /// A request at a mid-quality quantizer.
    ///
    /// **`global_scale` runs the opposite way from intuition.** I.2.1 divides
    /// by it — `mDC = (1 << 16) * w / (global_scale * quant_lf)` and
    /// `Mul = (1 << 16) / (global_scale * HfMul)` — so a *larger*
    /// `global_scale` is a *finer* quantizer and a bigger file. The value below
    /// is roughly the middle of the `U32(1 + u(11), ...)` range's useful part.
    ///
    /// There is no distance or bits-per-pixel target here, because there is no
    /// rate loop yet: choosing `global_scale` from a byte budget is milestone 4
    /// (`docs/PLAN.md` slice 14). Until then the caller sets the scalar.
    ///
    /// `group_size_shift` is carried for the modular track's benefit and is
    /// **ignored** by the VarDCT planner: F.2 does not signal the field outside
    /// kModular, so a kVarDCT frame's `group_dim` is always 256.
    ///
    /// # Panics
    ///
    /// Never: the three constants are inside their fields' ranges, which the
    /// `default_request_is_representable` test pins.
    #[must_use]
    pub fn defaults() -> Self {
        Self {
            global_scale: GlobalScale::new(32_768).unwrap_or(GlobalScale::MIN),
            quant_lf: QuantLf::new(16).unwrap_or(QuantLf::MIN),
            hf_mul: HfMul::new(1).unwrap_or(HfMul::MIN),
            group_size_shift: 1,
            budget: SearchBudget::default(),
            target: None,
            tolerance: RateTolerance::default(),
            restoration: RestorationDecision::default(),
            resources: jpxl_encode::EncodeResources::auto(),
        }
    }

    /// [`Self::defaults`] with a size target, so the rate loop chooses the
    /// quantizer.
    ///
    /// The scalars keep their default values and become the search's starting
    /// point rather than its answer.
    #[must_use]
    pub fn for_target(target: RateTarget) -> Self {
        Self {
            target: Some(target),
            ..Self::defaults()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_request_is_representable() {
        let request = EncodeRequest::defaults();
        assert_eq!(request.global_scale.get(), 32_768);
        assert_eq!(request.quant_lf.get(), 16);
        assert_eq!(request.hf_mul.get(), 1);
        assert_eq!(request.group_size_shift, 1);
        assert_eq!(request.budget.cover_mode, CoverMode::Hierarchical);
        assert_eq!(
            request.budget.aq_mode,
            crate::field::AqMode::Masking,
            "production AQ default is Masking"
        );
        assert_eq!(request.target, None, "the default path has no rate loop");
        assert!(
            !request.restoration.gaborish && request.restoration.epf_iters == 0,
            "default path keeps filters off"
        );
    }

    #[test]
    fn bits_per_pixel_is_the_byte_budget_of_the_whole_stream() {
        // 1 bpp over 64x64 pixels is 4096 bits, i.e. 512 bytes.
        assert_eq!(RateTarget::BitsPerPixel(1.0).bytes_for(64, 64), 512);
        // Fractional bytes are not spendable: the target is a ceiling.
        assert_eq!(RateTarget::BitsPerPixel(0.5).bytes_for(3, 3), 0);
        assert_eq!(RateTarget::Bytes(1234).bytes_for(64, 64), 1234);
        assert_eq!(RateTarget::BitsPerPixel(f64::NAN).bytes_for(64, 64), 0);
        assert_eq!(RateTarget::BitsPerPixel(-2.0).bytes_for(64, 64), 0);
    }

    #[test]
    fn the_tolerance_is_the_larger_of_its_two_spellings() {
        let tolerance = RateTolerance::default();
        // Small targets take the byte floor, large ones the fraction.
        assert_eq!(tolerance.bytes_for(100), 8);
        assert_eq!(tolerance.bytes_for(10_000), 100);
        assert_eq!(
            RateTolerance {
                bytes: 0,
                fraction: 0.0
            }
            .bytes_for(10_000),
            0
        );
    }
}
