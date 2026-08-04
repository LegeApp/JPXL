//! What the caller asks for, and how hard the encoder is allowed to work.
//!
//! `Encoder-plan1.md` §12: a quality target and an effort target are different
//! things, and effort is a *budget* rather than an integer examined all over
//! the encoder. Both are declared here at the stage boundary; no field of
//! [`SearchBudget`] is ever read inside a kernel.

use jpxl_encode::vardct::ids::{GlobalScale, HfMul, QuantLf};

/// How the cover search explores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoverMode {
    /// One DCT8x8 per atom; no search at all (milestone 1 and 2).
    #[default]
    FixedDct8x8,
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
}

/// One encode request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodeRequest {
    /// I.2's `global_scale`.
    pub global_scale: GlobalScale,
    /// I.2's `quant_lf`.
    pub quant_lf: QuantLf,
    /// The constant per-varblock multiplier, until adaptive quantization
    /// (milestone 7) computes one per varblock.
    pub hf_mul: HfMul,
    /// F.2's `group_size_shift`.
    pub group_size_shift: u32,
    /// The effort budget.
    pub budget: SearchBudget,
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
        assert_eq!(request.budget.cover_mode, CoverMode::FixedDct8x8);
    }
}
