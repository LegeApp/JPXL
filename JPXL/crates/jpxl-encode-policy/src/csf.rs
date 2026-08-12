//! A first-principles contrast-sensitivity weight for the cover/CfL objective.
//!
//! # Why this exists
//!
//! `block_cost_bounded` prices a candidate's distortion as a **flat** sum of
//! squared dequantized-coefficient errors: cell `(0,1)` and cell `(7,7)` cost
//! the same for the same error magnitude. Phase 6.0 measured that they do not —
//! for identical charged cost, butteraugli varies by 2.65x across the DCT8x8
//! frequency plane, along a shape stable across content and amplitude
//! (every cross-run Pearson r >= 0.961). Phase 6.2 then showed that shape is
//! the *same curve* for DCT16x16 and DCT32x32 once expressed against normalised
//! spatial frequency (worst shared-bin disagreement 1.15x).
//!
//! # Derived, not fitted
//!
//! The weight here is **not** taken from that measurement. It is the
//! Mannos–Sakrison contrast-sensitivity function, a published psychovisual
//! model that predates and is independent of any JPEG XL implementation, at a
//! conventional viewing assumption. The measurement's role is to *check* it
//! (`csf_weight_agrees_with_the_phase62_measurement`), not to supply it.
//!
//! That distinction is the whole clean-room point (AGENTS.md §2, and the
//! `butteraugli` note in the workspace manifest): fitting an encoder weight
//! table to butteraugli's measured response would make this project's
//! perceptual model a derivative of libjxl's. Deriving one from open literature
//! and reporting how well it agrees does not.
//!
//! Nothing in this module may be tuned to improve agreement. If the check
//! fails, that is a result to record, not a parameter to adjust.

/// Pixels per degree of visual angle.
///
/// 60 ppd is the standard "one pixel per arcminute" viewing assumption — the
/// conventional definition of a display viewed at the distance where pixels
/// reach the eye's nominal resolution limit. It is fixed by that convention,
/// **not** chosen to fit the measurement.
const PIXELS_PER_DEGREE: f64 = 60.0;

/// Mannos–Sakrison contrast sensitivity, `A(f)`, for `f` in cycles per degree.
///
/// `A(f) = 2.6 * (0.0192 + 0.114 f) * exp(-(0.114 f)^1.1)`.
///
/// Band-pass: near-zero response at DC, a peak around 8 cycles/degree, and a
/// steep high-frequency rolloff.
#[must_use]
pub fn mannos_sakrison(cycles_per_degree: f64) -> f64 {
    let x = 0.114 * cycles_per_degree;
    2.6 * (0.0192 + x) * (-x.powf(1.1)).exp()
}

/// The cycles-per-degree of one DCT cell of a square transform.
///
/// A size-`n` DCT-II's basis `k` carries `k / (2n)` cycles per sample, so `k/n`
/// is its frequency as a fraction of Nyquist. The radial frequency of cell
/// `(u, v)` combines the two axes, and [`PIXELS_PER_DEGREE`] converts sample
/// frequency to visual angle: Nyquist is `ppd / 2` cycles per degree.
///
/// Expressing it this way is what lets one curve serve all three squares — the
/// property Phase 6.2 Result A established.
#[must_use]
pub fn cell_cycles_per_degree(u: usize, v: usize, side: usize) -> f64 {
    if side == 0 {
        return 0.0;
    }
    #[allow(
        clippy::cast_precision_loss,
        reason = "u, v and side are at most 32; exact in f64"
    )]
    let (fu, fv, n) = (u as f64, v as f64, side as f64);
    let radial = ((fu / n).powi(2) + (fv / n).powi(2)).sqrt();
    radial * PIXELS_PER_DEGREE / 2.0
}

/// The per-cell distortion weight for one square transform, in coefficient
/// order (`cell = u * side + v`), normalised to **mean 1** over the non-LLF
/// cells.
///
/// The weight is `A(f)^2`, not `A(f)`: `A` scales perceived error *amplitude*,
/// and the objective's term is squared error.
///
/// Mean-1 normalisation is what keeps [`crate::HfQuantizers`]'s `lambda`
/// calibrated. `lambda` is derived as `16 / mean(s^2)` in units of bits per
/// unit of squared sample error, so a weight whose mean is 1 leaves the
/// expected distortion of a white error field — and therefore the rate/
/// distortion balance — unchanged. Only the *distribution* across frequency
/// moves, which is the entire intent.
///
/// LLF cells (the top-left `llf x llf` sub-block) are excluded from the
/// normalisation and given weight 1, because `score_channel_lanes` skips them:
/// LF is quantized by a separate path.
#[must_use]
pub fn square_weights(side: usize, llf: usize) -> Vec<f32> {
    let cells = side.saturating_mul(side);
    let mut weights = vec![1.0f32; cells];
    if side == 0 {
        return weights;
    }
    let mut raw = vec![0.0f64; cells];
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for cell in 0..cells {
        let (u, v) = (cell / side, cell % side);
        if u < llf && v < llf {
            continue;
        }
        let a = mannos_sakrison(cell_cycles_per_degree(u, v, side));
        let w = a * a;
        raw[cell] = w;
        sum += w;
        count += 1;
    }
    if count == 0 || sum <= 0.0 {
        return weights;
    }
    #[allow(
        clippy::cast_precision_loss,
        reason = "cell counts are at most 1024; exact in f64"
    )]
    let mean = sum / count as f64;
    for cell in 0..cells {
        let (u, v) = (cell / side, cell % side);
        if u < llf && v < llf {
            continue;
        }
        #[allow(
            clippy::cast_possible_truncation,
            reason = "a normalised sensitivity ratio is well inside f32"
        )]
        let w = (raw[cell] / mean) as f32;
        weights[cell] = w;
    }
    weights
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mannos_sakrison_is_band_pass_with_a_peak_near_eight_cycles() {
        assert!(
            mannos_sakrison(0.0) < mannos_sakrison(8.0),
            "the model attenuates DC relative to its peak"
        );
        assert!(
            mannos_sakrison(30.0) < mannos_sakrison(8.0),
            "the model rolls off above its peak"
        );
        // Locate the peak on a fine grid rather than asserting a hard-coded
        // argmax, so this pins the shape and not an arithmetic accident.
        let mut best = (0.0f64, 0.0f64);
        for i in 0..=4000 {
            let f = f64::from(i) * 0.02;
            let a = mannos_sakrison(f);
            if a > best.1 {
                best = (f, a);
            }
        }
        assert!(
            (5.0..12.0).contains(&best.0),
            "the peak should sit near 8 cycles/degree, found {}",
            best.0
        );
    }

    #[test]
    fn square_weights_have_mean_one_over_the_non_llf_cells() {
        // This is the property that keeps `lambda` calibrated; if it drifts,
        // the weight silently reweights rate against distortion.
        for (side, llf) in [(8usize, 1usize), (16, 2), (32, 4)] {
            let w = square_weights(side, llf);
            let mut sum = 0.0f64;
            let mut count = 0usize;
            for cell in 0..side * side {
                let (u, v) = (cell / side, cell % side);
                if u < llf && v < llf {
                    assert_eq!(w[cell], 1.0, "LLF cells keep weight 1");
                    continue;
                }
                sum += f64::from(w[cell]);
                count += 1;
            }
            #[allow(clippy::cast_precision_loss, reason = "counts are small")]
            let mean = sum / count as f64;
            assert!(
                (mean - 1.0).abs() < 1e-5,
                "DCT{side}x{side} weight mean is {mean}, not 1"
            );
        }
    }

    #[test]
    fn square_weights_fall_with_frequency_above_the_peak() {
        // At 60 ppd, Nyquist is 30 cycles/degree and the model's peak (~8) sits
        // at about 0.27 of Nyquist, so the top of the band must be attenuated
        // relative to the low band on every size.
        for (side, llf) in [(8usize, 1usize), (16, 2), (32, 4)] {
            let w = square_weights(side, llf);
            let corner = w[side * side - 1];
            let low = w[llf * side + llf];
            assert!(
                corner < low,
                "DCT{side}x{side}: corner weight {corner} should be below the \
                 low-frequency weight {low}"
            );
        }
    }
}
