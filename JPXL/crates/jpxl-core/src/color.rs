//! The XYB color model.
//!
//! XYB is JPEG XL's internal color space for lossy coding. It is an absolute
//! space derived from a model of the long/medium/short-wavelength cone
//! responses of the human retina, and it is opponent in the same sense as
//! CIELAB: a luminance axis `Y`, a red-green axis `X`, and a blue-yellow axis
//! `B`.
//!
//! The forward transform, starting from linear sRGB with a nominal `[0, 1]`
//! range where `(1, 1, 1)` is white at `intensity_target` nits:
//!
//! 1. mix R, G, B into cone-like `(Lm, Mm, Sm)` with a 3x3 matrix, adding a
//!    small bias `b` that models spontaneous opsin activation;
//! 2. gamma-compress each channel with a cube root, subtracting `cbrt(b)` so
//!    that black maps to zero;
//! 3. rotate into opponent axes: `X = (Lg - Mg) / 2`, `Y = (Lg + Mg) / 2`,
//!    `B = Sg`.
//!
//! The inverse undoes each step; the cube root becomes a cube, which is why
//! the decoder side is the cheap one.
//!
//! # Constant provenance
//!
//! The bias and the forward mixing matrix are taken verbatim from the JPEG XL
//! paper (arXiv:2506.05987, "RGB to XYB conversion"). The inverse matrix is
//! not published there; it was computed here by exact rational inversion of
//! the forward matrix and rounded to `f32`. All of these are **`[provisional]`
//! until re-verified against ISO/IEC 18181-1** — bit-exact conformance depends
//! on the standard's own constants, not on a paper's typesetting of them.
//!
//! Everything here is software-defined `f32` arithmetic: no FMA contraction is
//! relied on, no platform intrinsics, no fast-math. Results are reproducible
//! across targets.

/// `[provisional]` Opsin bias `b`, modelling spontaneous opsin activation.
///
/// The paper gives `0.00379307325527544933`; this is the nearest `f32`.
pub const OPSIN_BIAS: f32 = 0.003_793_073_4;

/// `[provisional]` Linear sRGB -> `(Lm, Mm, Sm)` mixing matrix, row-major.
///
/// Each row sums to 1, so the neutral axis maps to `Lm == Mm == Sm`.
pub const OPSIN_ABSORBANCE_MATRIX: [[f32; 3]; 3] = [
    [0.3, 0.622, 0.078],
    [0.23, 0.692, 0.078],
    [0.243_422_7, 0.204_767_4, 0.551_809_9],
];

/// `[provisional]` Exact inverse of [`OPSIN_ABSORBANCE_MATRIX`], row-major.
pub const OPSIN_ABSORBANCE_INVERSE_MATRIX: [[f32; 3]; 3] = [
    [11.031_567, -9.866_944, -0.164_622_98],
    [-3.254_147_4, 4.418_770_3, -0.164_622_98],
    [-3.658_851_6, 2.712_923_5, 1.945_928_1],
];

/// Cube root of [`OPSIN_BIAS`], subtracted after gamma compression so that
/// linear black maps to the XYB origin.
const OPSIN_BIAS_CBRT: f32 = 0.155_954_2;

/// Converts one linear-sRGB triple to XYB.
///
/// Input is `[r, g, b]` in linear light relative to the sRGB primaries and the
/// D65 white point, nominally in `[0, 1]`. Out-of-range values are allowed
/// (wide gamut, HDR) and are transformed by the same formula.
#[must_use]
pub fn linear_srgb_to_xyb(rgb: [f32; 3]) -> [f32; 3] {
    let [r, g, b] = rgb;
    let [ml, mm, ms] = OPSIN_ABSORBANCE_MATRIX;

    let lm = ml[0] * r + ml[1] * g + ml[2] * b + OPSIN_BIAS;
    let mm_ = mm[0] * r + mm[1] * g + mm[2] * b + OPSIN_BIAS;
    let sm = ms[0] * r + ms[1] * g + ms[2] * b + OPSIN_BIAS;

    let lg = lm.cbrt() - OPSIN_BIAS_CBRT;
    let mg = mm_.cbrt() - OPSIN_BIAS_CBRT;
    let sg = sm.cbrt() - OPSIN_BIAS_CBRT;

    [0.5 * (lg - mg), 0.5 * (lg + mg), sg]
}

/// Converts one XYB triple back to linear sRGB.
///
/// Exact inverse of [`linear_srgb_to_xyb`] up to `f32` rounding.
#[must_use]
pub fn xyb_to_linear_srgb(xyb: [f32; 3]) -> [f32; 3] {
    let [x, y, b] = xyb;

    let lg = y + x;
    let mg = y - x;
    let sg = b;

    let lm = cube(lg + OPSIN_BIAS_CBRT) - OPSIN_BIAS;
    let mm = cube(mg + OPSIN_BIAS_CBRT) - OPSIN_BIAS;
    let sm = cube(sg + OPSIN_BIAS_CBRT) - OPSIN_BIAS;

    let [ir, ig, ib] = OPSIN_ABSORBANCE_INVERSE_MATRIX;
    [
        ir[0] * lm + ir[1] * mm + ir[2] * sm,
        ig[0] * lm + ig[1] * mm + ig[2] * sm,
        ib[0] * lm + ib[1] * mm + ib[2] * sm,
    ]
}

/// Converts three planar linear-sRGB channels to XYB in place.
///
/// # Panics
///
/// Panics if the three planes do not have equal length.
pub fn linear_srgb_to_xyb_planes(r: &mut [f32], g: &mut [f32], b: &mut [f32]) {
    assert!(
        r.len() == g.len() && g.len() == b.len(),
        "planar XYB conversion needs three equally sized planes"
    );
    for ((rp, gp), bp) in r.iter_mut().zip(g.iter_mut()).zip(b.iter_mut()) {
        let [x, y, bb] = linear_srgb_to_xyb([*rp, *gp, *bp]);
        *rp = x;
        *gp = y;
        *bp = bb;
    }
}

/// Converts three planar XYB channels back to linear sRGB in place.
///
/// # Panics
///
/// Panics if the three planes do not have equal length.
pub fn xyb_to_linear_srgb_planes(x: &mut [f32], y: &mut [f32], b: &mut [f32]) {
    assert!(
        x.len() == y.len() && y.len() == b.len(),
        "planar XYB conversion needs three equally sized planes"
    );
    for ((xp, yp), bp) in x.iter_mut().zip(y.iter_mut()).zip(b.iter_mut()) {
        let [r, g, bb] = xyb_to_linear_srgb([*xp, *yp, *bp]);
        *xp = r;
        *yp = g;
        *bp = bb;
    }
}

/// `v^3`, written out so the cube is three multiplies rather than a `powf`.
#[inline]
fn cube(v: f32) -> f32 {
    v * v * v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity-check that the tabulated cube root of the bias is right.
    #[test]
    fn bias_cbrt_matches_bias() {
        assert!((OPSIN_BIAS_CBRT - OPSIN_BIAS.cbrt()).abs() < 1e-7);
    }

    #[test]
    fn matrix_rows_sum_to_one() {
        for row in OPSIN_ABSORBANCE_MATRIX {
            let sum = row[0] + row[1] + row[2];
            assert!((sum - 1.0).abs() < 1e-6, "row sums to {sum}");
        }
    }

    #[test]
    fn inverse_matrix_is_an_inverse() {
        for (i, inv_row) in OPSIN_ABSORBANCE_INVERSE_MATRIX.into_iter().enumerate() {
            // Row i of M^-1 times M is a linear combination of M's rows.
            let mut product_row = [0.0f32; 3];
            for (coeff, fwd_row) in inv_row.into_iter().zip(OPSIN_ABSORBANCE_MATRIX) {
                for (acc, v) in product_row.iter_mut().zip(fwd_row) {
                    *acc += coeff * v;
                }
            }
            for (j, acc) in product_row.into_iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((acc - expected).abs() < 1e-5, "M^-1 M [{i}][{j}] = {acc}");
            }
        }
    }

    #[test]
    fn black_maps_near_the_origin() {
        let xyb = linear_srgb_to_xyb([0.0, 0.0, 0.0]);
        for c in xyb {
            assert!(
                c.abs() < 1e-6,
                "black should map to the XYB origin, got {xyb:?}"
            );
        }
    }

    #[test]
    fn neutral_gray_has_no_chroma_on_x() {
        // Every matrix row sums to 1, so R == G == B gives Lg == Mg and X == 0.
        for v in [0.1f32, 0.25, 0.5, 1.0] {
            let [x, y, _] = linear_srgb_to_xyb([v, v, v]);
            assert!(x.abs() < 1e-6, "gray {v} produced X = {x}");
            assert!(y > 0.0, "gray {v} should have positive luminance");
        }
    }

    #[test]
    fn luminance_is_monotonic_in_intensity() {
        let mut prev = f32::NEG_INFINITY;
        for step in 0..=16u8 {
            let v = f32::from(step) / 16.0;
            let [_, y, _] = linear_srgb_to_xyb([v, v, v]);
            assert!(y > prev, "Y must increase with intensity at v = {v}");
            prev = y;
        }
    }

    #[test]
    fn round_trips_over_a_color_grid() {
        const STEPS: u8 = 8;
        let mut worst = 0.0f32;
        for ri in 0..=STEPS {
            for gi in 0..=STEPS {
                for bi in 0..=STEPS {
                    let rgb = [
                        f32::from(ri) / f32::from(STEPS),
                        f32::from(gi) / f32::from(STEPS),
                        f32::from(bi) / f32::from(STEPS),
                    ];
                    let back = xyb_to_linear_srgb(linear_srgb_to_xyb(rgb));
                    for (a, b) in rgb.into_iter().zip(back) {
                        let err = (a - b).abs();
                        worst = worst.max(err);
                        assert!(err < 1e-4, "round trip {rgb:?} -> {back:?} erred by {err}");
                    }
                }
            }
        }
        assert!(worst < 1e-4, "worst round-trip error {worst}");
    }

    #[test]
    fn round_trips_outside_the_srgb_gamut() {
        // Wide-gamut and HDR values are legal inputs.
        for rgb in [
            [1.5f32, 0.2, -0.1],
            [-0.05, 1.0, 2.0],
            [4.0, 4.0, 4.0],
            [0.001, 0.002, 0.0005],
        ] {
            let back = xyb_to_linear_srgb(linear_srgb_to_xyb(rgb));
            for (a, b) in rgb.into_iter().zip(back) {
                assert!(
                    (a - b).abs() < 1e-3 * a.abs().max(1.0),
                    "round trip {rgb:?} -> {back:?}"
                );
            }
        }
    }

    #[test]
    fn planar_matches_scalar() {
        let source = [[0.0f32, 0.25, 1.0], [0.5, 0.5, 0.5], [1.0, 0.75, 0.0]];
        let expected: Vec<[f32; 3]> = source.into_iter().map(linear_srgb_to_xyb).collect();

        let mut r: Vec<f32> = source.iter().map(|px| px[0]).collect();
        let mut g: Vec<f32> = source.iter().map(|px| px[1]).collect();
        let mut b: Vec<f32> = source.iter().map(|px| px[2]).collect();

        linear_srgb_to_xyb_planes(&mut r, &mut g, &mut b);
        let planar = r.iter().zip(&g).zip(&b).map(|((x, y), z)| [*x, *y, *z]);
        for (got, want) in planar.zip(&expected) {
            assert_eq!(got, *want, "planar must match the scalar path exactly");
        }

        xyb_to_linear_srgb_planes(&mut r, &mut g, &mut b);
        let back = r.iter().zip(&g).zip(&b).map(|((x, y), z)| [*x, *y, *z]);
        for (got, want) in back.zip(source) {
            for (a, e) in got.into_iter().zip(want) {
                assert!((a - e).abs() < 1e-4, "{got:?} vs {want:?}");
            }
        }
    }
}
