//! The metric's opponent colour space.
//!
//! SSIMULACRA2 evaluates its maps on XYB — the same opsin-absorbance space
//! JPEG XL codes in — shifted and scaled so every component lies in roughly
//! `0..1` and a pixel-wise difference is at most `1`. This module is the
//! metric's *own* copy of that transform: it must not drift with the
//! encoder's colour code, and it uses a software-defined cube root so the
//! same input scores identically on every host.
//!
//! The constants are the public definition of the metric (the opsin matrix
//! and bias are those of 18181-1 L.2.1 Table L.1; the shifts are SSIMULACRA2's
//! "positive XYB" adjustment).

/// Row-major linear-sRGB → opsin absorbance matrix (L, M, S rows).
const OPSIN_ABSORBANCE_MATRIX: [[f32; 3]; 3] = [
    [0.3, 0.622, 0.078],
    [0.23, 0.692, 0.078],
    [0.243_422_7, 0.204_767_4, 0.551_809_9],
];

/// Added to each absorbance before the cube root so black maps to the origin.
const OPSIN_BIAS: f32 = 0.003_793_073_4;

/// `cbrt(OPSIN_BIAS)`, subtracted after the cube root.
const OPSIN_BIAS_CBRT: f32 = 0.155_954_2;

/// Cube root with a host-independent result.
///
/// `f32::cbrt` delegates to the platform C library, whose rounding differs by
/// an ULP between hosts. This spells out the arithmetic instead: a bit-pattern
/// estimate followed by four Newton steps in `f64`, then one rounding to
/// `f32`. Every step is an IEEE basic operation, so the result is the same on
/// every IEEE-754 host. Zero, infinities and NaN return unchanged.
#[must_use]
// The `f64` Newton result is rounded to `f32` exactly once; that is the
// function's contract, not an accidental truncation.
#[allow(clippy::cast_possible_truncation)]
pub fn reproducible_cbrt(value: f32) -> f32 {
    if value == 0.0 || !value.is_finite() {
        return value;
    }
    let x = f64::from(value);
    let a = x.abs();
    // Exponent-thirding estimate (the classic cube-root bit hack on a double).
    let mut y = f64::from_bits(a.to_bits() / 3 + (0x2A9F_7893_u64 << 32));
    for _ in 0..4 {
        y -= (y * y * y - a) / (3.0 * y * y);
    }
    let root = y as f32;
    if x < 0.0 { -root } else { root }
}

/// Converts one linear-sRGB triple to the metric's positive XYB.
#[must_use]
pub fn linear_srgb_to_positive_xyb(rgb: [f32; 3]) -> [f32; 3] {
    let [r, g, b] = rgb;
    let [ml, mm, ms] = OPSIN_ABSORBANCE_MATRIX;
    let lm = ml[0] * r + ml[1] * g + ml[2] * b + OPSIN_BIAS;
    let mm_ = mm[0] * r + mm[1] * g + mm[2] * b + OPSIN_BIAS;
    let sm = ms[0] * r + ms[1] * g + ms[2] * b + OPSIN_BIAS;
    let lg = reproducible_cbrt(lm) - OPSIN_BIAS_CBRT;
    let mg = reproducible_cbrt(mm_) - OPSIN_BIAS_CBRT;
    let sg = reproducible_cbrt(sm) - OPSIN_BIAS_CBRT;
    let x = 0.5 * (lg - mg);
    let y = 0.5 * (lg + mg);
    let bb = sg;
    // "Positive XYB": B becomes B-Y, and each component is shifted (and X
    // scaled) into roughly 0..1 so the SSIM constants keep their meaning.
    [x * 14.0 + 0.42, y + 0.01, (bb - y) + 0.55]
}

/// Converts three linear-sRGB planes into three positive-XYB planes of the
/// same length. Purely per pixel, so any row partition gives identical output.
pub fn planes_to_positive_xyb(
    r: &[f32],
    g: &[f32],
    b: &[f32],
    x: &mut [f32],
    y: &mut [f32],
    bb: &mut [f32],
) {
    for (((((&r, &g), &b), x), y), bb) in r
        .iter()
        .zip(g)
        .zip(b)
        .zip(x.iter_mut())
        .zip(y.iter_mut())
        .zip(bb.iter_mut())
    {
        let [px, py, pb] = linear_srgb_to_positive_xyb([r, g, b]);
        *x = px;
        *y = py;
        *bb = pb;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cube_root_is_accurate_and_odd() {
        for v in [1e-6_f32, 0.003_793_073_4, 0.1, 0.5, 1.0, 8.0, 123.456, 1e6] {
            let got = f64::from(reproducible_cbrt(v));
            let want = f64::from(v).cbrt();
            assert!(
                (got - want).abs() <= want * 1e-6,
                "cbrt({v}) = {got}, want {want}"
            );
            assert_eq!(reproducible_cbrt(-v), -reproducible_cbrt(v));
        }
        assert_eq!(reproducible_cbrt(0.0), 0.0);
        assert_eq!(reproducible_cbrt(27.0), 3.0);
    }

    #[test]
    fn black_lands_at_the_documented_origin() {
        let [x, y, b] = linear_srgb_to_positive_xyb([0.0, 0.0, 0.0]);
        assert!((x - 0.42).abs() < 1e-5, "{x}");
        assert!((y - 0.01).abs() < 1e-5, "{y}");
        assert!((b - 0.55).abs() < 1e-5, "{b}");
    }

    #[test]
    fn srgb_white_stays_inside_the_unit_range() {
        let [x, y, b] = linear_srgb_to_positive_xyb([1.0, 1.0, 1.0]);
        assert!(
            (0.0..=1.0).contains(&x) && (0.0..=1.0).contains(&y) && (0.0..=1.0).contains(&b),
            "{x} {y} {b}"
        );
    }
}
