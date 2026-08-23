//! The scalar rules of VarDCT reconstruction that both trees must agree on:
//! I.5.2's LF multipliers, I.5.3's HF multiplier and bias adjustment, and
//! I.6's chroma-from-luma factors and reconstruction.
//!
//! They are pure functions of header values, with no opinion about where the
//! values came from, which is what makes them neutral: the decoder reads
//! them from a bitstream, the encoder's plan renderer takes them from a plan,
//! and both must produce the same sample from the same integer.

/// I.2.1 / I.5.3's `(1 << 16)` numerator shared by the LF and HF multipliers.
pub const QUANT_NUMERATOR: f64 = 65536.0;

/// I.5.3's per-channel quantization-matrix scale base, `pow(0.8, qm_scale - 2)`.
pub const QM_SCALE_BASE: f32 = 0.8;

/// G.1.2's divisor: the signalled LF weights are stored times 128.
pub const LF_WEIGHT_SCALE: f32 = 128.0;

/// I.5.3's `Mul = (1 << 16) / (global_scale * HfMul)`, formed in `f64` and
/// narrowed once. Zero when either factor is zero (unreachable for a legal
/// header).
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    reason = "the single deliberate f64 -> f32 narrowing of the multiplier"
)]
pub fn hf_multiplier(global_scale: u32, hf_mul: u32) -> f32 {
    let denom = f64::from(global_scale) * f64::from(hf_mul);
    if denom <= 0.0 {
        return 0.0;
    }
    (QUANT_NUMERATOR / denom) as f32
}

/// I.5.3's `pow(0.8, qm_scale - 2)`; `qm_scale` is a `u(3)`.
#[must_use]
pub fn qm_multiplier(qm_scale: u32) -> f32 {
    let exponent = i32::try_from(qm_scale).unwrap_or(2) - 2;
    QM_SCALE_BASE.powi(exponent)
}

/// I.5.3's bias adjustment of one quantized coefficient: values of magnitude
/// at most one are scaled by `quant_bias`, larger ones pulled toward zero by
/// `quant_bias_numerator / quant`.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    reason = "quantized coefficients are bounded far inside f32's exact range"
)]
pub fn bias_adjust(quant: i32, quant_bias: f32, quant_bias_numerator: f32) -> f32 {
    let q = quant as f32;
    if quant.abs() <= 1 {
        q * quant_bias
    } else {
        q - quant_bias_numerator / q
    }
}

/// I.2.1's LF multipliers `mDC[c] = (1 << 16) * w[c] / (global_scale *
/// quant_lf)` from the three **unscaled** G.1.2 weights (already divided by
/// [`LF_WEIGHT_SCALE`]), formed in `f64` and narrowed once per channel.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    reason = "the single deliberate f64 -> f32 narrowing per multiplier"
)]
pub fn lf_multipliers(global_scale: u32, quant_lf: u32, unscaled_weights: [f32; 3]) -> [f32; 3] {
    let denom = f64::from(global_scale) * f64::from(quant_lf);
    unscaled_weights.map(|w| {
        if denom > 0.0 {
            (QUANT_NUMERATOR * f64::from(w) / denom) as f32
        } else {
            0.0
        }
    })
}

/// I.5.2's `d = mDC * q / (1 << extra_precision)`.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    reason = "quantized LF samples are bounded far inside f32's exact range"
)]
pub fn lf_dequantize(quant: i32, multiplier: f32, extra_precision: u8) -> f32 {
    let divisor = f32::from(1u16 << extra_precision.min(3));
    multiplier * (quant as f32) / divisor
}

/// I.6's `(kX, kB)` from I.2.3's bundle and a stored factor pair:
/// `k = base_correlation + factor / colour_factor`.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    reason = "colour_factor and the stored factors are bounded far inside f32's exact range"
)]
pub fn cfl_factors(
    base_correlation_x: f32,
    base_correlation_b: f32,
    colour_factor: u32,
    x_factor: i32,
    b_factor: i32,
) -> (f32, f32) {
    let colour_factor = colour_factor.max(1) as f32;
    (
        base_correlation_x + x_factor as f32 / colour_factor,
        base_correlation_b + b_factor as f32 / colour_factor,
    )
}

/// I.6's linear chroma-from-luma reconstruction: `Y = dY; X = dX + kX*Y;
/// B = dB + kB*Y`, returned as `(X, Y, B)`.
#[must_use]
pub fn cfl_apply(dx: f32, dy: f32, db: f32, k_x: f32, k_b: f32) -> (f32, f32, f32) {
    let y = dy;
    (dx + k_x * y, y, db + k_b * y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_hf_multiplier_is_the_clause_formula() {
        assert_eq!(hf_multiplier(32_768, 2), 1.0);
        assert_eq!(hf_multiplier(1, 1), 65_536.0);
        assert_eq!(hf_multiplier(0, 5), 0.0);
    }

    #[test]
    fn qm_scale_two_is_neutral() {
        assert!((qm_multiplier(2) - 1.0).abs() < 1e-7);
        assert!((qm_multiplier(3) - 0.8).abs() < 1e-6);
        assert!((qm_multiplier(1) - 1.25).abs() < 1e-6);
    }

    #[test]
    fn bias_adjustment_leaves_zero_at_zero_and_pulls_large_values_in() {
        assert_eq!(bias_adjust(0, 0.9, 0.145), 0.0);
        assert!((bias_adjust(1, 0.9, 0.145) - 0.9).abs() < 1e-7);
        assert!((bias_adjust(-1, 0.9, 0.145) + 0.9).abs() < 1e-7);
        assert!((bias_adjust(10, 0.9, 0.145) - (10.0 - 0.0145)).abs() < 1e-6);
    }

    #[test]
    fn lf_multipliers_follow_the_numerator_over_the_product() {
        let m = lf_multipliers(
            32_768,
            2,
            [1.0 / 32.0 / 128.0, 1.0 / 4.0 / 128.0, 0.5 / 128.0],
        );
        assert!((m[0] - 1.0 / 32.0 / 128.0).abs() < 1e-9);
        assert!((m[1] - 1.0 / 4.0 / 128.0).abs() < 1e-9);
        assert_eq!(lf_dequantize(4, 0.5, 1), 1.0);
    }

    #[test]
    fn neutral_cfl_is_the_identity_on_chroma() {
        let (kx, kb) = cfl_factors(0.0, 1.0, 84, 0, 0);
        assert_eq!((kx, kb), (0.0, 1.0));
        assert_eq!(cfl_apply(0.5, 2.0, -1.0, kx, kb), (0.5, 2.0, 1.0));
        let (kx, _) = cfl_factors(0.0, 1.0, 84, 42, 0);
        assert!((kx - 0.5).abs() < 1e-6);
    }
}
