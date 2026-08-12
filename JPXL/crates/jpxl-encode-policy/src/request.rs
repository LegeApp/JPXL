//! What the caller asks for, and how hard the encoder is allowed to work.
//!
//! `Encoder-plan1.md` §12: a quality target and an effort target are different
//! things, and effort is a *budget* rather than an integer examined all over
//! the encoder. Both are declared here at the stage boundary; no field of
//! [`SearchBudget`] is ever read inside a kernel.

use jpxl_encode::vardct::ids::{GlobalScale, HfMul, QmScale, QuantLf};
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
// `Eq` is deliberately absent: `aq_tuning` holds floats, and a budget that
// carries a perceptual tuning is not a thing with exact equality.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SearchBudget {
    /// How the cover search explores.
    pub cover_mode: CoverMode,
    /// How adaptive quantization points its field (default Masking).
    pub aq_mode: crate::field::AqMode,
    /// The perceptual constants that shape that field.
    ///
    /// Defaults to the historical hardcoded values for explicit Masking and
    /// Uniform research requests. Off does not read them.
    pub aq_tuning: crate::field::AqTuning,
    /// What the rate loop may spend (milestone 4).
    pub rate: RateSearchBudget,
}

/// Research policy for G.2.4's per-block EPF sharpness plane.
///
/// This is separate from [`RestorationDecision`] because the latter is the
/// decoder-visible J.1 header decision, while choosing the sharpness samples
/// is encoder analysis. Production stays at [`Self::Zero`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EpfSharpnessMode {
    /// Emit the neutral all-zero sharpness plane; EPF sigma is zero.
    #[default]
    Zero,
    /// Emit sharpness 7 for every 8x8 block, selecting the default LUT's
    /// nonzero multiplier. Phase 5I promoted this for target-rate requests
    /// after it improved Butteraugli in all twelve corpus/rate cells.
    Uniform7,
}

/// Research policy for the cover objective's per-transform distortion scale.
///
/// `block_cost_bounded` brings each candidate's coefficient error into the
/// sample domain by multiplying it by `side^2`, and then compares candidates of
/// different sizes as though a unit of sample-domain squared error were equally
/// visible whichever transform carried it.
///
/// Phase 6.2 measured that it is not
/// (`jpegxl-rs.observation.one-frequency-curve-fits-all-squares-2026-08-12`).
/// At *equal total injected sample-domain error*, a DCT32x32 basis costs 4% to
/// 21% more butteraugli than DCT8x8 bases — monotonically in size, in every one
/// of nine radial-frequency bins, on both corpus photographs. Error on a 32x32
/// support is spatially coherent over a large region, where the same energy
/// spread across sixteen independently-signed 8x8 patches is closer to noise,
/// and noise is easier to mask. The objective therefore under-penalises large
/// transforms, which biases the hierarchical cover toward merging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoverSizePenalty {
    /// Price every transform's sample-domain error identically: the shipped
    /// objective, and byte-identical to the pre-Phase-6 encoder.
    #[default]
    Neutral,
    /// Scale each candidate's distortion by [`CoverSizePenalty::measured`].
    Measured,
}

impl CoverSizePenalty {
    /// The per-transform distortion multiplier, keyed by the transform's
    /// coefficient edge (8, 16 or 32).
    ///
    /// [`Self::Neutral`] returns exactly `1.0`, so the shipped objective is
    /// bit-identical rather than merely close: IEEE multiplication by one is
    /// exact.
    ///
    /// [`Self::Measured`]'s constants come from Phase 6.2's equalised-energy
    /// sweep, converted from a butteraugli ratio into an energy-equivalent
    /// distortion ratio. Butteraugli is not linear in injected energy: over the
    /// measured 4x energy step it follows `ba ~ E^p` with `p = 0.4477`,
    /// `0.4477` and `0.4433` for the three sizes (mean `0.4462`), so a measured
    /// butteraugli ratio `r` corresponds to `r^(1/p)` of distortion. That maps
    /// the 3.8%/5.7% mean butteraugli excess onto the multipliers below, which
    /// were stable across both probe amplitudes (DCT16x16 identical to four
    /// decimals; DCT32x32 within 1.4%).
    #[must_use]
    pub fn multiplier(self, coeff_edge: usize) -> f64 {
        match self {
            Self::Neutral => 1.0,
            Self::Measured => match coeff_edge {
                16 => 1.0881,
                32 => 1.1331,
                _ => 1.0,
            },
        }
    }
}

/// Research policy for the cover objective's per-cell frequency weight.
///
/// [`Self::Flat`] is the shipped objective: every coefficient cell of every
/// transform is charged identically for the same error. Phase 6.0 measured that
/// costs butteraugli by up to 2.65x across the DCT8x8 frequency plane, and
/// Phase 6.2 showed the same curve applies to DCT16x16 and DCT32x32 once
/// expressed against normalised spatial frequency.
///
/// [`Self::Csf`] weights each cell by [`crate::csf`]'s Mannos-Sakrison
/// contrast-sensitivity model, normalised to mean 1 so `lambda` stays
/// calibrated. **It is a research arm, not a candidate default:** Phase 6.3's
/// pre-registered check found the model does not agree with the measured
/// response (worst bin ratio 3.14x against a 1.35x limit, Pearson r 0.41
/// against a 0.80 limit), so it is retained to test whether frequency
/// reweighting moves the objective at all, not because its curve is right.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoverFrequencyWeight {
    /// Charge every cell identically: the shipped objective, bit-identical.
    #[default]
    Flat,
    /// Weight by the Mannos-Sakrison CSF at 60 pixels per degree.
    Csf,
    /// Weight by the standard's own DCT8x8 dequantization matrix, read as a
    /// curve in normalised radial frequency and resampled onto each transform's
    /// grid ([`crate::csf::quant_donor_weights`]).
    ///
    /// Phase 6.5's candidate. Scored against the Phase 6.2 measurement it sits
    /// at median 1.41x, against the CSF's 2.06x and a 1.18x ceiling for a curve
    /// fitted directly to butteraugli -- so it captures most of the available
    /// improvement while being derived from the standard rather than from the
    /// metric, which is what keeps this project's perceptual model its own.
    QuantDonor,
}

/// Research policy for how the HF quantizer picks an integer.
///
/// [`Self::Nearest`] is the shipped rule: among `[0, est-1, est, est+1]`, take
/// the smallest `|recon - target|`. It has no rate term at all, so it can spend
/// bits on coefficients whose distortion saving does not pay for them --
/// exactly what `sources/outside-advice.md` names.
///
/// [`Self::RateDistortion`] minimises `residual_bits(q) + rd * (recon-target)^2`
/// instead, with `rd` the same Lagrange weight `block_cost_bounded` applies to
/// that cell, so the quantizer and the cover search minimise one currency.
///
/// **Bounded by its rate proxy.** `residual_bits` charges magnitude bit length
/// plus sign and knows nothing about zero runs, entropy context or coefficient
/// order, so this captures the first-order "is this coefficient worth any bits"
/// decision -- a widened dead zone -- and not the run-extension win that makes
/// rate-distortion quantization pay in a mature encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuantizerChoiceMode {
    /// Nearest reconstruction: the shipped rule, bit-identical.
    #[default]
    Nearest,
    /// Rate-aware choice against the cover objective's own Lagrange weight.
    RateDistortion,
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
    /// F.2/I.5.3's X-channel quantization-matrix exponent.
    pub x_qm_scale: QmScale,
    /// F.2/I.5.3's B-channel quantization-matrix exponent.
    pub b_qm_scale: QmScale,
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
    /// Encoder policy for G.2.4's EPF sharpness plane.
    ///
    /// The fixed-quantizer default is all zero. Target-rate requests use
    /// [`EpfSharpnessMode::Uniform7`] after the Phase 5I corpus gate. The plane
    /// has an effect only when [`RestorationDecision::epf_iters`] is nonzero.
    pub epf_sharpness: EpfSharpnessMode,
    /// Research policy for the cover objective's per-transform distortion
    /// scale. Production stays at [`CoverSizePenalty::Neutral`], which is
    /// bit-identical to the pre-Phase-6 objective.
    pub cover_size_penalty: CoverSizePenalty,
    /// Research policy for the cover objective's per-cell frequency weight.
    /// Production stays at [`CoverFrequencyWeight::Flat`], which is
    /// bit-identical to the pre-Phase-6 objective.
    pub cover_frequency_weight: CoverFrequencyWeight,
    /// Research policy for how the HF quantizer picks an integer. Production
    /// stays at [`QuantizerChoiceMode::Nearest`], which is bit-identical.
    pub quantizer_choice: QuantizerChoiceMode,
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
            x_qm_scale: QmScale::NEUTRAL,
            b_qm_scale: QmScale::NEUTRAL,
            group_size_shift: 1,
            budget: SearchBudget::default(),
            target: None,
            tolerance: RateTolerance::default(),
            restoration: RestorationDecision::default(),
            epf_sharpness: EpfSharpnessMode::default(),
            cover_size_penalty: CoverSizePenalty::default(),
            cover_frequency_weight: CoverFrequencyWeight::default(),
            quantizer_choice: QuantizerChoiceMode::default(),
            resources: jpxl_encode::EncodeResources::auto(),
        }
    }

    /// The production target-rate policy, so the rate loop chooses the
    /// quantizer.
    ///
    /// Phase 5G's six-scene, two-rate gate found AQ Off with `quant_lf = 8`
    /// improved Butteraugli distance in all twelve cells at matched achieved
    /// rate. LF fill stays disabled because it changes that distortion knob
    /// after the primary rate ladder settles. Phase 5I then found one active
    /// EPF step with uniform Sharpness 7 improved Butteraugli in all twelve
    /// cells as well. [`Self::defaults`] remains the stable fixed-quantizer,
    /// restoration-off request.
    #[must_use]
    pub fn for_target(target: RateTarget) -> Self {
        let mut request = Self::defaults();
        request.target = Some(target);
        request.quant_lf = QuantLf::new(8).unwrap_or(QuantLf::MIN);
        request.budget.aq_mode = crate::field::AqMode::Off;
        request.budget.rate.lf_fill_probes = 0;
        request.restoration.epf_iters = 1;
        request.epf_sharpness = EpfSharpnessMode::Uniform7;
        request.cover_frequency_weight = CoverFrequencyWeight::QuantDonor;
        request
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_donor_weight_is_promoted_for_target_rate_only() {
        // Phase 6.5b promoted under an SSIMULACRA2-primary rule. The scope
        // matters as much as the choice: the screen measured target-rate
        // encodes, so `defaults` must stay flat or the Contract A
        // fixed-quantizer fingerprints move on evidence that never covered
        // them.
        assert_eq!(
            EncodeRequest::defaults().cover_frequency_weight,
            CoverFrequencyWeight::Flat,
            "the fixed-quantizer request must stay flat"
        );
        assert_eq!(
            EncodeRequest::for_target(RateTarget::BitsPerPixel(1.0)).cover_frequency_weight,
            CoverFrequencyWeight::QuantDonor,
            "the target-rate policy carries the promoted weight"
        );
        // The other two research controls stay neutral: 6.2b's size penalty was
        // an honest negative and 7.0's quantizer rule regressed SSIMULACRA2.
        assert_eq!(
            EncodeRequest::for_target(RateTarget::BitsPerPixel(1.0)).cover_size_penalty,
            CoverSizePenalty::Neutral
        );
        assert_eq!(
            EncodeRequest::for_target(RateTarget::BitsPerPixel(1.0)).quantizer_choice,
            QuantizerChoiceMode::Nearest
        );
    }

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
            "production AQ default remains the historical Masking policy"
        );
        assert_eq!(request.target, None, "the default path has no rate loop");
        assert_eq!(request.x_qm_scale, QmScale::NEUTRAL);
        assert_eq!(request.b_qm_scale, QmScale::NEUTRAL);
        assert!(
            !request.restoration.gaborish && request.restoration.epf_iters == 0,
            "default path keeps filters off"
        );
        assert_eq!(request.epf_sharpness, EpfSharpnessMode::Zero);
    }

    #[test]
    fn target_request_uses_the_phase5_quality_policy() {
        let target = RateTarget::Bytes(12_345);
        let request = EncodeRequest::for_target(target);
        assert_eq!(request.target, Some(target));
        assert_eq!(request.quant_lf.get(), 8);
        assert_eq!(request.budget.aq_mode, crate::field::AqMode::Off);
        assert_eq!(request.budget.rate.lf_fill_probes, 0);
        assert_eq!(request.restoration.epf_iters, 1);
        assert_eq!(request.epf_sharpness, EpfSharpnessMode::Uniform7);
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
