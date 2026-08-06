//! Quantization against the exact decoder function (`Encoder-plan1.md` §7.3).
//!
//! An encoder that quantizes by dividing and rounding is guessing. The
//! decoder's reconstruction is a specific, non-linear, per-channel function of
//! the stored integer — I.5.2 for LF, I.5.3 for HF, with L.2.1's `quant_bias`
//! bending the small values — and the only way to know which integer is best is
//! to evaluate that function. So [`HfQuantizer::choose`] builds the candidate
//! integers arithmetically and then **scores them through the reconstruction**,
//! keeping the one whose reconstructed value is closest to the target.
//!
//! # The reconstruction, restated
//!
//! For a coefficient at cell `(x, y)` of channel `c` in a varblock with
//! `HfMul`:
//!
//! ```text
//! adj   = |q| <= 1 ? q * quant_bias[c] : q - quant_bias_numerator / q   (I.5.3)
//! Mul   = (1 << 16) / (global_scale * HfMul)                            (I.5.3)
//! qm    = pow(0.8, x_qm_scale - 2)   for X, ...b_qm_scale... for B      (I.5.3)
//! d     = adj * Mul * qm * dequant_matrix[c](x, y)                      (I.2.4)
//! ```
//!
//! and for an LF sample:
//!
//! ```text
//! mDC   = (1 << 16) * m_lf_unscaled[c] / (global_scale * quant_lf)      (I.2.1)
//! d     = mDC * q / (1 << extra_precision)                              (I.5.2)
//! ```
//!
//! The LF branch is linear and its inverse is exact; the HF branch is not, and
//! that is the whole reason this module exists rather than a division.
//!
//! # Chroma from luma is not optional
//!
//! I.6 runs on **every** dequantized coefficient, and its default parameters
//! are not neutral: `base_correlation_b` is `1.0`, so a decoder reconstructs
//! `B = dB + 1.0 * dY` whatever this encoder intends. The signalled factors
//! `XFromY`, `BFromY`, `x_factor_lf` and `b_factor_lf` are the *searchable*
//! part. Slice 15 estimates them, but the rule is unchanged: the resulting
//! whole `kB` still has to be subtracted on the way in, from the
//! **reconstructed** `dY`, not from source Y. Getting this wrong does not
//! produce a subtle error: it doubles the blue-yellow axis.
//!
//! `base_correlation_x` is `0.0`, so `kX` really is neutral and X passes
//! through.

use jpxl_core::dequant::{DequantMatrices, DequantMatrix};
use jpxl_core::varblock::TransformType;

use crate::error::{PolicyError, Result};

/// Number of coefficient channels.
pub const NUM_CHANNELS: usize = 3;

/// Cells in a DCT8x8 coefficient array.
pub const DCT8X8_CELLS: usize = 64;

/// I.5.3's `(1 << 16)` numerator, shared by the LF and HF multipliers.
const QUANT_NUMERATOR: f64 = 65536.0;

/// G.1.2's `/ 128`, applied to `m_x_lf`, `m_y_lf` and `m_b_lf`.
const LF_WEIGHT_SCALE: f64 = 128.0;

/// The G.1.2 default LF dequantization weights, before the `/ 128`.
const DEFAULT_LF_WEIGHTS: [f64; NUM_CHANNELS] = [1.0 / 32.0, 1.0 / 4.0, 1.0 / 2.0];

/// I.2.3's `base_correlation_x` default: chroma-from-luma leaves X alone.
const BASE_CORRELATION_X: f32 = 0.0;

/// I.2.3's `base_correlation_b` default: the decoder adds a whole `dY` to `dB`.
const BASE_CORRELATION_B: f32 = 1.0;

/// The largest magnitude a quantized coefficient may reach.
///
/// Not a clause limit — G.2.4 stores the coefficient as a modular sample, so
/// any `i32` is representable — but a working bound. A target that needs more
/// than this means the quantizer was handed a scale it cannot serve, and a
/// silent clamp there would show up as a saturated block nobody can explain.
const MAX_QUANT: i32 = 1 << 20;

/// I.5.2's LF quantizer: linear, and exactly invertible.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LfQuantizer {
    m_dc: [f32; NUM_CHANNELS],
    extra_precision: u8,
}

impl LfQuantizer {
    /// Builds the quantizer for a frame's `global_scale` and `quant_lf`, with
    /// the G.1.2 default channel weights.
    #[must_use]
    pub fn new(global_scale: u32, quant_lf: u32, extra_precision: u8) -> Self {
        let denom = f64::from(global_scale) * f64::from(quant_lf);
        let m_dc = core::array::from_fn(|c| {
            let w = DEFAULT_LF_WEIGHTS.get(c).copied().unwrap_or(0.0) / LF_WEIGHT_SCALE;
            #[allow(
                clippy::cast_possible_truncation,
                reason = "the single deliberate f64 -> f32 narrowing, mirroring \
                          the decoder's own lf_multipliers"
            )]
            if denom > 0.0 {
                (QUANT_NUMERATOR * w / denom) as f32
            } else {
                0.0
            }
        });
        Self {
            m_dc,
            extra_precision,
        }
    }

    /// I.5.2's `d = mDC * q / (1 << extra_precision)`.
    #[must_use]
    pub fn reconstruct(&self, q: i32, channel: usize) -> f32 {
        let m = self.m_dc.get(channel).copied().unwrap_or(0.0);
        #[allow(
            clippy::cast_precision_loss,
            reason = "|q| stays far inside f32's exact integer range; MAX_QUANT \
                      is 2^20"
        )]
        let q = q as f32;
        m * q / (1u32 << self.extra_precision) as f32
    }

    /// The integer whose reconstruction is nearest `target`.
    ///
    /// # Errors
    ///
    /// [`PolicyError::Unsupported`] if the quantizer is degenerate or the
    /// target needs an integer outside [`MAX_QUANT`].
    pub fn quantize(&self, target: f32, channel: usize) -> Result<i32> {
        let m = self.m_dc.get(channel).copied().unwrap_or(0.0);
        if !(m.is_finite() && m > 0.0) {
            return Err(PolicyError::Unsupported {
                what: "a degenerate LF dequantization multiplier",
            });
        }
        #[allow(
            clippy::cast_precision_loss,
            reason = "extra_precision is at most 3, so the shift is exact"
        )]
        let scaled = target * (1u32 << self.extra_precision) as f32 / m;
        clamp_round(scaled)
    }
}

/// I.5.3's HF quantizer for one transform type, over all three channels.
#[derive(Debug, Clone)]
pub struct HfQuantizer {
    quant_bias: [f32; NUM_CHANNELS],
    quant_bias_numerator: f32,
    /// Flattened `Mul * qm * matrix[c](x,y)` in coefficient order, per channel.
    /// Phase-1: avoid matrix lookups and multiplies on every [`Self::choose`].
    steps: [Box<[f32]>; NUM_CHANNELS],
    /// `0.5 * quant_bias[c] * steps[c][cell]` — zero wins under the nearest-
    /// reconstruction + lower-magnitude tie rule when `|target| <=` this.
    zero_threshold: [Box<[f32]>; NUM_CHANNELS],
}

impl HfQuantizer {
    /// Builds the quantizer for one transform, `global_scale` and `HfMul`.
    ///
    /// `qm_scale` is `[x_qm_scale, y_qm_scale (unused), b_qm_scale]` as the
    /// frame header signals them; the Y channel has no such factor.
    ///
    /// # Errors
    ///
    /// [`PolicyError::Unsupported`] if the I.2.5 default matrices cannot be
    /// built, which would mean the shared table is broken.
    pub fn new(
        transform: TransformType,
        global_scale: u32,
        hf_mul: u32,
        x_qm_scale: u32,
        b_qm_scale: u32,
    ) -> Result<Self> {
        let defaults = DequantMatrices::all_default().map_err(|_| PolicyError::Unsupported {
            what: "the I.2.5 default dequantization matrices",
        })?;
        let matrices: [DequantMatrix; NUM_CHANNELS] = [
            defaults
                .for_transform(transform, 0)
                .map_err(|_| PolicyError::Unsupported {
                    what: "a dequantization matrix for this transform",
                })?,
            defaults
                .for_transform(transform, 1)
                .map_err(|_| PolicyError::Unsupported {
                    what: "a dequantization matrix for this transform",
                })?,
            defaults
                .for_transform(transform, 2)
                .map_err(|_| PolicyError::Unsupported {
                    what: "a dequantization matrix for this transform",
                })?,
        ];

        let denom = f64::from(global_scale) * f64::from(hf_mul);
        #[allow(
            clippy::cast_possible_truncation,
            reason = "the deliberate f64 -> f32 narrowing of I.5.3's Mul"
        )]
        let mul = if denom > 0.0 {
            (QUANT_NUMERATOR / denom) as f32
        } else {
            0.0
        };
        let scale = [
            mul * qm_multiplier(x_qm_scale),
            mul,
            mul * qm_multiplier(b_qm_scale),
        ];
        let cols = transform.coeff_cols();
        let rows = transform.coeff_rows();
        let cells = cols.saturating_mul(rows);
        let quant_bias = jpxl_core::color::DEFAULT_QUANT_BIAS;
        let mut steps: [Box<[f32]>; NUM_CHANNELS] =
            core::array::from_fn(|_| vec![0.0f32; cells].into_boxed_slice());
        let mut zero_threshold: [Box<[f32]>; NUM_CHANNELS] =
            core::array::from_fn(|_| vec![0.0f32; cells].into_boxed_slice());
        for channel in 0..NUM_CHANNELS {
            let sc = scale.get(channel).copied().unwrap_or(0.0);
            let bias = quant_bias.get(channel).copied().unwrap_or(1.0);
            let matrix = &matrices[channel];
            for cell in 0..cells {
                let (x, y) = (cell % cols.max(1), cell / cols.max(1));
                let step = sc * matrix.at(x, y);
                if let Some(slot) = steps[channel].get_mut(cell) {
                    *slot = step;
                }
                if let Some(slot) = zero_threshold[channel].get_mut(cell) {
                    // |recon(±1)| = bias * step; ties prefer smaller |q|, so
                    // zero wins when |target| <= 0.5 * |recon(±1)|.
                    *slot = 0.5 * bias * step;
                }
            }
        }

        Ok(Self {
            quant_bias,
            quant_bias_numerator: jpxl_core::color::DEFAULT_QUANT_BIAS_NUMERATOR,
            steps,
            zero_threshold,
        })
    }

    /// I.5.3's bias adjustment.
    fn bias_adjust(&self, q: i32, channel: usize) -> f32 {
        #[allow(
            clippy::cast_precision_loss,
            reason = "|q| <= MAX_QUANT == 2^20, exact in f32"
        )]
        let f = q as f32;
        if q.abs() <= 1 {
            f * self.quant_bias.get(channel).copied().unwrap_or(1.0)
        } else {
            f - self.quant_bias_numerator / f
        }
    }

    /// Precomputed step for `(channel, cell)`, or 0 if out of range.
    #[inline]
    fn step_at(&self, channel: usize, cell: usize) -> f32 {
        self.steps
            .get(channel)
            .and_then(|s| s.get(cell).copied())
            .unwrap_or(0.0)
    }

    /// I.5.3's full reconstruction of one coefficient, before I.6.
    #[must_use]
    pub fn reconstruct(&self, q: i32, channel: usize, cell: usize) -> f32 {
        self.bias_adjust(q, channel) * self.step_at(channel, cell)
    }

    /// I.5.3's quantization step for one cell: `Mul * qm * dequant_matrix`.
    ///
    /// Zero for a cell outside the matrix; the block solver reads it to price
    /// distortion in the same units [`Self::choose`] quantizes in.
    #[must_use]
    pub fn step(&self, channel: usize, cell: usize) -> f32 {
        self.step_at(channel, cell)
    }

    /// The integer whose reconstruction is nearest `target`.
    ///
    /// Candidates come from the linear estimate — the bias adjustment is a
    /// small perturbation, never a sign change — plus zero, which the linear
    /// estimate can miss when `quant_bias` shrinks `±1` below half a step.
    /// Ties go to the smaller magnitude, so a coefficient that reconstructs
    /// equally well as `0` or `±1` costs the fewest bits.
    ///
    /// Phase-1: precomputed steps, exact zero early-out (output-identical),
    /// and optional `wide` comparison of the four candidates.
    ///
    /// # Errors
    ///
    /// [`PolicyError::Unsupported`] if the step is degenerate or the target
    /// needs an integer outside [`MAX_QUANT`].
    pub fn choose(&self, target: f32, channel: usize, cell: usize) -> Result<i32> {
        crate::diagnostics::note_choose();
        let step = self.step_at(channel, cell);
        if !(step.is_finite() && step > 0.0) {
            return Err(PolicyError::Unsupported {
                what: "a degenerate HF quantization step",
            });
        }
        // Exact zero shortcut: |target| <= 0.5 * quant_bias * step ⇒ zero
        // reconstructs at least as close as ±1 and wins ties by magnitude.
        let thr = self
            .zero_threshold
            .get(channel)
            .and_then(|t| t.get(cell).copied())
            .unwrap_or(0.0);
        if target.abs() <= thr {
            return Ok(0);
        }

        let estimate = clamp_round(target / step)?;
        let candidates = [0, estimate - 1, estimate, estimate + 1];

        #[cfg(feature = "simd")]
        {
            use wide::f32x4;
            let mut recon = [0.0f32; 4];
            let mut legal = [false; 4];
            for (i, &q) in candidates.iter().enumerate() {
                if q.abs() > MAX_QUANT {
                    recon[i] = f32::INFINITY;
                    legal[i] = false;
                } else {
                    recon[i] = self.bias_adjust(q, channel) * step;
                    legal[i] = true;
                }
            }
            let r = f32x4::new(recon);
            let t = f32x4::splat(target);
            let err = (r - t).abs().to_array();
            let mut best = 0i32;
            let mut best_error = f32::INFINITY;
            for (i, &q) in candidates.iter().enumerate() {
                if !legal[i] {
                    continue;
                }
                let error = err[i];
                if error < best_error || (error == best_error && q.abs() < best.abs()) {
                    best = q;
                    best_error = error;
                }
            }
            return Ok(best);
        }

        #[cfg(not(feature = "simd"))]
        {
            let mut best = 0i32;
            let mut best_error = f32::INFINITY;
            for q in candidates {
                if q.abs() > MAX_QUANT {
                    continue;
                }
                let error = (self.bias_adjust(q, channel) * step - target).abs();
                if error < best_error || (error == best_error && q.abs() < best.abs()) {
                    best = q;
                    best_error = error;
                }
            }
            Ok(best)
        }
    }
}

/// I.5.3's `pow(0.8, qm_scale - 2)`, exactly as the decoder computes it.
fn qm_multiplier(qm_scale: u32) -> f32 {
    let exponent = i32::try_from(qm_scale).unwrap_or(2) - 2;
    0.8f32.powi(exponent)
}

/// I.6's `kX` and `kB` with the I.2.3 defaults and neutral signalled factors.
///
/// Returned as a pair so a caller cannot apply one and forget the other.
#[must_use]
pub const fn neutral_cfl_factors() -> (f32, f32) {
    (BASE_CORRELATION_X, BASE_CORRELATION_B)
}

/// I.2.3's default `colour_factor`: the divisor I.6 turns a stored factor into
/// a correlation coefficient with. Slice 15 keeps it at the default so the
/// searched quantity is the biased factor alone, in a space every decoder
/// reads the same way.
pub const DEFAULT_COLOUR_FACTOR: u32 = 84;

/// I.6's `k = base_correlation + factor / colour_factor`, computed the way the
/// decoder computes it: the division is done in `f32`, from the same integer
/// the wire carries, so the encoder's `k` is bit-identical to the one the
/// decoder will apply.
#[must_use]
pub fn cfl_multiplier(base: f32, factor: i32, colour_factor: u32) -> f32 {
    #[allow(
        clippy::cast_precision_loss,
        reason = "colour_factor and the searched factor stay far inside f32's \
                  exact-integer range, mirroring the decoder's own widening"
    )]
    let ratio = factor as f32 / colour_factor as f32;
    base + ratio
}

/// The least-squares seed of one channel against luma, as sufficient
/// statistics of the unquantized residual `C - k*Y`
/// (`Encoder-plan1.md` §8).
///
/// The residual energy of a candidate correlation coefficient `k` is
/// `Σ(C - k·Y)² = ΣC² - 2k·ΣYC + k²·ΣY²`, so the three running sums are all a
/// seed calculation needs: every candidate is then an `O(1)` parabola
/// evaluation, not a re-scan of the coefficients. The policy layer follows
/// this regression with a separate integer refinement through reconstructed
/// `dY` and the exact quantizer arithmetic.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CflAccumulator {
    s_yy: f64,
    s_yc: f64,
    s_cc: f64,
}

impl CflAccumulator {
    /// Adds one `(unquantized Y, unquantized channel coefficient)` pair.
    pub fn add(&mut self, y: f32, coeff: f32) {
        let y = f64::from(y);
        let c = f64::from(coeff);
        self.s_yy += y * y;
        self.s_yc += y * c;
        self.s_cc += c * c;
    }

    /// Residual energy `Σ(C - k·Y)²` at a given `k`.
    fn residual_energy(&self, k: f64) -> f64 {
        self.s_cc - 2.0 * k * self.s_yc + k * k * self.s_yy
    }

    /// The integer factor in `[lo, hi]` whose correlation coefficient
    /// minimizes the residual energy, refining in the wire's representable
    /// space rather than rounding a float.
    ///
    /// The energy is convex in `k`, and `k` is affine in the factor, so the
    /// least-squares optimum's integer neighbours bracket the best factor;
    /// the neutral factor `0` is always among the candidates and wins every
    /// tie. That tie rule is what makes an already-decorrelated channel —
    /// grayscale's `X ≡ 0` and `B ≡ Y` — degenerate to exactly the neutral
    /// factor, byte for byte.
    #[must_use]
    pub fn best_factor(&self, base: f32, colour_factor: u32, lo: i32, hi: i32) -> i32 {
        if self.s_yy <= 0.0 {
            // No luma energy to correlate against: nothing to subtract.
            return 0;
        }
        let k_ls = self.s_yc / self.s_yy;
        let star = (k_ls - f64::from(base)) * f64::from(colour_factor);
        #[allow(
            clippy::cast_possible_truncation,
            reason = "star is finite here (s_yy > 0), and the result is clamped \
                      into [lo, hi] before use"
        )]
        let centre = star.round() as i64;
        let mut best: Option<(i32, f64)> = None;
        // The floor/ceil of the continuous optimum bracket the best integer;
        // widen by one on each side against float slack, and always weigh the
        // neutral factor so the degenerate case cannot drift off zero.
        for delta in [-2i64, -1, 0, 1, 2] {
            let cand = centre.saturating_add(delta);
            let cand = i32::try_from(cand.clamp(i64::from(lo), i64::from(hi))).unwrap_or(0);
            consider(&mut best, cand, self, base, colour_factor);
        }
        consider(&mut best, 0, self, base, colour_factor);
        best.map_or(0, |(factor, _)| factor)
    }
}

/// Keeps the lower-energy candidate, breaking ties toward the smaller
/// magnitude and then toward zero (the neutral factor).
fn consider(
    best: &mut Option<(i32, f64)>,
    cand: i32,
    acc: &CflAccumulator,
    base: f32,
    colour_factor: u32,
) {
    let k = f64::from(cfl_multiplier(base, cand, colour_factor));
    let energy = acc.residual_energy(k);
    let better = match *best {
        None => true,
        Some((factor, best_energy)) => {
            energy < best_energy || (energy == best_energy && cand.abs() < factor.abs())
        }
    };
    if better {
        *best = Some((cand, energy));
    }
}

/// Rounds to the nearest integer and rejects anything past [`MAX_QUANT`].
fn clamp_round(v: f32) -> Result<i32> {
    if !v.is_finite() {
        return Err(PolicyError::Unsupported {
            what: "a non-finite quantization target",
        });
    }
    let rounded = v.round();
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the range is checked against MAX_QUANT immediately below"
    )]
    let value = if rounded.abs() > MAX_QUANT as f32 {
        return Err(PolicyError::Unsupported {
            what: "a coefficient outside the quantizer's working range",
        });
    } else {
        rounded as i32
    };
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_lf_quantizer_inverts_its_own_reconstruction() {
        let q = LfQuantizer::new(4096, 16, 0);
        for channel in 0..NUM_CHANNELS {
            for step in -40i32..=40 {
                let target = q.reconstruct(step, channel);
                assert_eq!(
                    q.quantize(target, channel).expect("in range"),
                    step,
                    "channel {channel} step {step}"
                );
            }
        }
    }

    #[test]
    fn the_lf_multipliers_order_the_channels_x_y_b() {
        // G.1.2's defaults are 1/32, 1/4, 1/2, so Y's step is eight times X's
        // and B's is sixteen times X's: chroma is quantized far more coarsely.
        let q = LfQuantizer::new(4096, 16, 0);
        let x = q.reconstruct(1, 0);
        let y = q.reconstruct(1, 1);
        let b = q.reconstruct(1, 2);
        assert!((y / x - 8.0).abs() < 1e-3, "{x} {y}");
        assert!((b / x - 16.0).abs() < 1e-3, "{x} {b}");
    }

    #[test]
    fn extra_precision_divides_the_lf_step() {
        let coarse = LfQuantizer::new(4096, 16, 0);
        let fine = LfQuantizer::new(4096, 16, 2);
        assert!((coarse.reconstruct(1, 1) / fine.reconstruct(1, 1) - 4.0).abs() < 1e-4);
    }

    #[test]
    fn the_hf_quantizer_never_reconstructs_further_than_a_naive_rounding_would() {
        let q = HfQuantizer::new(TransformType::Dct8x8, 4096, 1, 2, 2).expect("defaults");
        for channel in 0..NUM_CHANNELS {
            for cell in 1..DCT8X8_CELLS {
                let step = q.reconstruct(1, channel, cell) / 0.945;
                for numerator in -12i32..=12 {
                    let target = step * (numerator as f32) / 4.0;
                    let chosen = q.choose(target, channel, cell).expect("in range");
                    let chosen_error = (q.reconstruct(chosen, channel, cell) - target).abs();
                    for other in [chosen - 1, chosen + 1, 0] {
                        let other_error = (q.reconstruct(other, channel, cell) - target).abs();
                        assert!(
                            chosen_error <= other_error + 1e-6,
                            "c{channel} cell{cell} target {target}: chose {chosen} \
                             (err {chosen_error}) over {other} (err {other_error})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_zero_target_quantizes_to_zero_in_every_channel_and_cell() {
        let q = HfQuantizer::new(TransformType::Dct8x8, 4096, 1, 2, 2).expect("defaults");
        for channel in 0..NUM_CHANNELS {
            for cell in 0..DCT8X8_CELLS {
                assert_eq!(q.choose(0.0, channel, cell).expect("in range"), 0);
                assert_eq!(q.reconstruct(0, channel, cell), 0.0);
            }
        }
    }

    #[test]
    fn the_dc_weight_of_dct8x8_is_the_coarsest_and_high_frequencies_are_finer() {
        // The I.2.5 DCT8x8 weights fall with distance from (0, 0), so the
        // dequantization matrix — their reciprocal — rises: one quantization
        // step reconstructs to a *larger* value at high frequency.
        let q = HfQuantizer::new(TransformType::Dct8x8, 4096, 1, 2, 2).expect("defaults");
        for channel in 0..NUM_CHANNELS {
            let low = q.reconstruct(1, channel, 1);
            let high = q.reconstruct(1, channel, 63);
            assert!(high > low, "channel {channel}: {low} then {high}");
        }
    }

    #[test]
    fn hf_mul_scales_every_step_together() {
        let one = HfQuantizer::new(TransformType::Dct8x8, 4096, 1, 2, 2).expect("defaults");
        let four = HfQuantizer::new(TransformType::Dct8x8, 4096, 4, 2, 2).expect("defaults");
        for cell in [1usize, 9, 63] {
            let ratio = one.reconstruct(2, 1, cell) / four.reconstruct(2, 1, cell);
            assert!((ratio - 4.0).abs() < 1e-3, "cell {cell}: {ratio}");
        }
    }

    #[test]
    fn the_qm_scale_is_neutral_at_two_and_shrinks_above_it() {
        assert!((qm_multiplier(2) - 1.0).abs() < 1e-7);
        assert!((qm_multiplier(3) - 0.8).abs() < 1e-6);
        assert!((qm_multiplier(1) - 1.25).abs() < 1e-6);
    }

    #[test]
    fn an_unreachable_target_is_refused_rather_than_clamped() {
        let q = HfQuantizer::new(TransformType::Dct8x8, 4096, 1, 2, 2).expect("defaults");
        assert!(q.choose(f32::INFINITY, 1, 1).is_err());
        assert!(q.choose(1e30, 1, 1).is_err());
        let lf = LfQuantizer::new(4096, 16, 0);
        assert!(lf.quantize(f32::NAN, 1).is_err());
    }

    #[test]
    fn the_neutral_cfl_factors_are_the_i23_defaults() {
        // kX is genuinely zero; kB is *one*, which is why B has to be
        // decorrelated on the way in.
        assert_eq!(neutral_cfl_factors(), (0.0, 1.0));
    }

    #[test]
    fn cfl_regression_refines_in_the_integer_wire_space() {
        let mut acc = CflAccumulator::default();
        // C = 0.5 * Y, and colour_factor = 84, so factor 42 is exact.
        for y in [-2.0f32, -1.0, 0.5, 3.0] {
            acc.add(y, 0.5 * y);
        }
        assert_eq!(acc.best_factor(0.0, DEFAULT_COLOUR_FACTOR, -128, 127), 42);
        assert_eq!(cfl_multiplier(0.0, 42, DEFAULT_COLOUR_FACTOR), 0.5);
    }

    #[test]
    fn a_degenerate_cfl_regression_stays_neutral() {
        let mut acc = CflAccumulator::default();
        for value in [-3.0f32, 0.0, 8.0] {
            acc.add(0.0, value);
        }
        assert_eq!(acc.best_factor(1.0, DEFAULT_COLOUR_FACTOR, -128, 127), 0);
    }
}
