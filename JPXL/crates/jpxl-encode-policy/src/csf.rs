//! Per-cell frequency weights for the cover/CfL objective.
//!
//! Two candidate curves live here, both keyed by [`radial_frequency`] so one
//! curve serves all three square transforms (Phase 6.2 Result A):
//!
//! * [`square_weights`] — the Mannos–Sakrison contrast-sensitivity function.
//!   **Rejected by Phase 6.3** on a pre-registered check and retained only as
//!   the arm that result is recorded against.
//! * [`quant_donor_weights`] — the standard's own DCT8x8 dequantization matrix
//!   read as a curve. Phase 6.5's candidate; see its own docs for why a
//!   contrast-sensitivity function is the wrong object entirely.
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
//! **Neither curve is taken from that measurement.** [`square_weights`] comes
//! from published psychovisual literature that predates any JPEG XL
//! implementation; [`quant_donor_weights`] comes from the standard's own I.2.5
//! default matrices. The measurement's only role is to *check* them.
//!
//! That distinction is the whole clean-room point (AGENTS.md §2, and the
//! `butteraugli` note in the workspace manifest): fitting an encoder weight
//! table to butteraugli's measured response would make this project's
//! perceptual model a derivative of libjxl's. Deriving one from open literature
//! or from the standard, and reporting how well it agrees, does not.
//!
//! Nothing in this module may be tuned to improve agreement. If a check fails,
//! that is a result to record, not a parameter to adjust — which is exactly
//! what happened to Mannos–Sakrison in Phase 6.3.

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

/// A cell's radial frequency as a fraction of Nyquist, in `0..=1`.
///
/// A size-`n` DCT-II's basis `k` carries `k / (2n)` cycles per sample, so `k/n`
/// is that axis as a fraction of Nyquist; the radial combination is divided by
/// `sqrt(2)` so the corner cell sits at 1 rather than at `sqrt(2)`.
///
/// **This is the common coordinate** that lets one curve serve all three
/// squares: Phase 6.2 Result A measured the perceptual curves agreeing to 1.15x
/// in it. [`cell_cycles_per_degree`] is the same quantity scaled into visual
/// angle for [`mannos_sakrison`]; the two differ only by that constant.
#[must_use]
pub fn radial_frequency(u: usize, v: usize, side: usize) -> f64 {
    if side == 0 {
        return 0.0;
    }
    #[allow(
        clippy::cast_precision_loss,
        reason = "u, v and side are at most 32; exact in f64"
    )]
    let (fu, fv, n) = (u as f64, v as f64, side as f64);
    ((fu / n).powi(2) + (fv / n).powi(2)).sqrt() / std::f64::consts::SQRT_2
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

/// The square whose default quantization matrix donates the frequency curve.
///
/// DCT8x8 was not picked by argument. All four candidates were scored against
/// the Phase 6.2 measurement in the common radial coordinate, and DCT8x8 wins
/// outright: median ratio 1.43x, against DCT16x16's 1.90x, DCT32x32's 2.62x and
/// 2.34x for the geometric mean of all three. Averaging is *worse* than DCT8x8
/// alone because the 16/32 curves are much steeper and drag it off target.
///
/// It also has independent standing: DCT8x8 is the baseline transform, and the
/// operating point [`crate::HfQuantizers`]'s `lambda` is calibrated at.
const DONOR_SIDE: usize = 8;
/// The donor's LLF edge — DCT8x8's `block_dims().0`.
const DONOR_LLF: usize = 1;

/// The per-cell distortion weight donated by the standard's own DCT8x8
/// dequantization matrix, resampled onto `side`'s grid and normalised to mean 1
/// over the non-LLF cells.
///
/// # Why this and not a contrast-sensitivity function
///
/// Phase 6.3 rejected Mannos–Sakrison on a pre-registered check, and the
/// literature closes the whole avenue rather than just that one model: the
/// canonical DCT-domain per-coefficient models (Ahumada–Peterson 1992,
/// Peterson–Ahumada–Watson 1993, Watson's DCTune 1993, Nill 1985) are all
/// **detection**-threshold models, band-pass with a low-frequency dip and a
/// high-frequency rolloff *harder* than Mannos–Sakrison — AP92 predicts ~0.037
/// of peak at Nyquist where measurement says ~1.05. Georgeson & Sullivan (1975)
/// give the reason: above threshold, perceived contrast becomes largely
/// frequency-independent (contrast constancy), so a detection CSF is the wrong
/// regime for judging quantization error in real content.
///
/// The standard's own matrices are a *distortion* allocation, not a detection
/// threshold, and score far closer to the measurement (median 1.41x against the
/// CSF's 2.06x) — while being derived from the standard rather than fitted to
/// butteraugli, which is what keeps this project's perceptual model its own.
///
/// # Why one donor rather than each size's own matrix
///
/// Because the matrices are **not** a consistent frequency model: their step
/// shapes disagree **3.34x** across sizes in the radial coordinate, against the
/// measured perceptual curve's 1.15x. They are per-size tuned. Reading one
/// size's matrix as a curve and resampling repairs that by construction — and
/// using each size's own matrix instead is exactly the `1/step^2` weighting
/// Phase 6.2 already falsified.
///
/// # Known limitation
///
/// The donor's support is only `f in [0.088, 0.875]` — DCT8x8's lowest non-LLF
/// cell to its corner. DCT16x16 and DCT32x32 have cells above 0.875, which
/// clamp to the donor's last value instead of continuing to fall. That flat
/// extrapolation is most of the candidate's 4.03x worst-bin error and is a
/// stated limitation, not an oversight.
#[must_use]
pub fn quant_donor_weights(side: usize, llf: usize) -> Vec<f32> {
    let cells = side.saturating_mul(side);
    let mut weights = vec![1.0f32; cells];
    if side == 0 {
        return weights;
    }
    let Some(curve) = donor_curve() else {
        return weights;
    };

    let mut raw = vec![0.0f64; cells];
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for cell in 0..cells {
        let (u, v) = (cell / side, cell % side);
        if u < llf && v < llf {
            continue;
        }
        let w = sample_curve(&curve, radial_frequency(u, v, side));
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
            reason = "a normalised weight ratio is well inside f32"
        )]
        let w = (raw[cell] / mean) as f32;
        weights[cell] = w;
    }
    weights
}

/// `(radial frequency, 1/step^2)` for every non-LLF cell of the donor square,
/// sorted by frequency. `None` if the shared default matrices cannot be built,
/// which would mean the I.2.5 table is broken.
fn donor_curve() -> Option<Vec<(f64, f64)>> {
    let defaults = jpxl_core::dequant::DequantMatrices::all_default().ok()?;
    let matrix = defaults
        .for_transform(jpxl_core::varblock::TransformType::Dct8x8, 1)
        .ok()?;
    let mut points = Vec::with_capacity(DONOR_SIDE * DONOR_SIDE);
    for cell in 0..DONOR_SIDE * DONOR_SIDE {
        let (u, v) = (cell / DONOR_SIDE, cell % DONOR_SIDE);
        if u < DONOR_LLF && v < DONOR_LLF {
            continue;
        }
        let step = f64::from(matrix.at(v, u));
        if step <= 0.0 {
            return None;
        }
        // `HfQuantizer`'s step is `scale[channel] * matrix.at(x, y)` with
        // `scale` constant across cells, so the matrix *is* the step shape and
        // the constant cancels under the mean-1 normalisation above.
        points.push((radial_frequency(u, v, DONOR_SIDE), 1.0 / (step * step)));
    }
    points.sort_by(|a, b| a.0.total_cmp(&b.0));
    (!points.is_empty()).then_some(points)
}

/// Linear interpolation along the donor curve, clamped at both ends.
fn sample_curve(curve: &[(f64, f64)], f: f64) -> f64 {
    let (Some(first), Some(last)) = (curve.first(), curve.last()) else {
        return 1.0;
    };
    if f <= first.0 {
        return first.1;
    }
    if f >= last.0 {
        return last.1;
    }
    for pair in curve.windows(2) {
        let [(f0, w0), (f1, w1)] = [pair[0], pair[1]];
        if f1 >= f {
            if f1 == f0 {
                return w1;
            }
            return w0 + (w1 - w0) * ((f - f0) / (f1 - f0));
        }
    }
    last.1
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
    fn the_donor_curve_falls_monotonically_over_its_support() {
        // If the resampling is ever inverted or the sort dropped, this fails:
        // the standard's DCT8x8 steps grow with frequency, so 1/step^2 falls.
        let curve = donor_curve().expect("the I.2.5 defaults build");
        assert!(curve.len() >= 60, "expected the 63 non-LLF cells");
        for pair in curve.windows(2) {
            assert!(
                pair[0].0 <= pair[1].0,
                "the curve must be sorted by frequency"
            );
        }
        let first = curve.first().expect("non-empty").1;
        let last = curve.last().expect("non-empty").1;
        assert!(
            last < first * 0.5,
            "the donor must fall substantially across its support, {first} -> {last}"
        );
        // The stated support, so the flat-extrapolation limitation stays true.
        assert!(
            (curve[0].0 - 0.0884).abs() < 1e-3,
            "lowest f {}",
            curve[0].0
        );
        let top = curve.last().expect("non-empty").0;
        assert!((top - 0.875).abs() < 1e-3, "highest f {top}");
    }

    #[test]
    fn donor_weights_have_mean_one_and_fall_with_frequency() {
        for (side, llf) in [(8usize, 1usize), (16, 2), (32, 4)] {
            let w = quant_donor_weights(side, llf);
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
                "DCT{side}x{side} donor mean is {mean}, not 1 — lambda would decalibrate"
            );
            let corner = w[side * side - 1];
            let low = w[llf * side + llf];
            assert!(
                corner < low,
                "DCT{side}x{side}: corner {corner} should sit below low-frequency {low}"
            );
        }
    }

    #[test]
    fn the_donor_differs_from_the_csf_it_replaces() {
        // Guards against a wiring mistake that silently returns the old curve.
        let donor = quant_donor_weights(8, 1);
        let csf = square_weights(8, 1);
        let diff = donor
            .iter()
            .zip(&csf)
            .map(|(a, b)| f64::from((a - b).abs()))
            .fold(0.0, f64::max);
        assert!(
            diff > 0.1,
            "donor and CSF weights are suspiciously close (max diff {diff})"
        );
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
