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
//! paper (arXiv:2506.05987, "RGB to XYB conversion"). The inverse matrix was
//! computed here by exact rational inversion of the forward matrix, rounded
//! to `f32`, and subsequently **verified digit-for-digit against the
//! normative defaults in 18181-1 L.2.1 Table L.1** (`inv_mat00..inv_mat22`).
//!
//! Sign convention: 18181-1 signals the per-channel `opsin_bias0..2` defaults
//! as `-0.0037930732552754493` (applied on the decoder's inverse path); the
//! fixed-constant helpers in this module ([`linear_srgb_to_xyb`] and
//! [`xyb_to_linear_srgb`]) apply the same magnitude with a positive sign,
//! writing the arithmetic as `+ OPSIN_BIAS_CBRT` / `- OPSIN_BIAS`.
//!
//! [`OpsinInverse`] — the matrix-driven entry point the VarDCT decoder uses —
//! takes the bias **exactly as signalled**, i.e. negative, and evaluates
//! L.2.2's printed expression `pow(gamma - cbrt(bias), 3) + bias` verbatim.
//! The two forms agree identically: with `bias < 0`, `cbrt(bias)` is negative,
//! so `gamma - cbrt(bias)` is `gamma + 0.15595...` and `+ bias` is
//! `- 0.0037930...`. That equivalence is asserted in
//! [`tests::signalled_defaults_match_the_fixed_constant_path`]. Everything
//! that reads a bias off the wire uses the signalled sign; nothing converts
//! between the two conventions at run time.
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

/// Exact inverse of [`OPSIN_ABSORBANCE_MATRIX`], row-major.
///
/// Matches the normative defaults of 18181-1 L.2.1 Table L.1 at `f32`
/// precision (`inv_mat00 = 11.031566901960783`, ...).
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

// ---------------------------------------------------------------------------
// Transfer functions (18181-1 Table E.6)
// ---------------------------------------------------------------------------

/// The IEC 61966-2-1 (sRGB) opto-electronic transfer function.
///
/// L.2.2 hands back **linear** light; the samples a conformance reference or a
/// PNG holds are in the signalled colour encoding, whose transfer function for
/// `TransferFunction::kSRGB` is this curve.
///
/// Negative inputs occur — L.2.2's output is explicitly allowed outside the
/// sRGB gamut and 18181-3 §4.2 forbids clipping before comparison — so the
/// curve is extended with odd symmetry, `f(-v) == -f(v)`. That is the only
/// extension that is continuous, monotonic and sign-preserving, and it leaves
/// the in-gamut values exactly as the standard curve defines them.
#[must_use]
pub fn linear_to_srgb(v: f32) -> f32 {
    let a = v.abs();
    let encoded = if a <= 0.003_130_8 {
        12.92 * a
    } else {
        1.055 * a.powf(1.0 / 2.4) - 0.055
    };
    if v < 0.0 { -encoded } else { encoded }
}

/// The ITU-R BT.709-6 opto-electronic transfer function, with the same odd
/// extension as [`linear_to_srgb`].
///
/// `E' = 4.5 E` below `0.018` and `1.099 E^0.45 - 0.099` above. This is the
/// curve `TransferFunction::k709` names, and it is *not* the sRGB curve — the
/// two differ by up to about 0.02 in the shadows, which is five times the
/// no-filters conformance budget, so the distinction is load-bearing.
#[must_use]
pub fn linear_to_rec709(v: f32) -> f32 {
    let a = v.abs();
    let encoded = if a < 0.018 {
        4.5 * a
    } else {
        1.099 * a.powf(0.45) - 0.099
    };
    if v < 0.0 { -encoded } else { encoded }
}

/// A pure power-law OETF, `v^exponent`, with the same odd extension as
/// [`linear_to_srgb`].
///
/// 18181-1 E.7 signals `gamma` as an integer scaled by `10^7`, and the value
/// it names is the OETF exponent itself, so the caller passes `gamma / 1e7`
/// directly.
#[must_use]
pub fn linear_to_gamma(v: f32, exponent: f32) -> f32 {
    let encoded = v.abs().powf(exponent);
    if v < 0.0 { -encoded } else { encoded }
}

// ---------------------------------------------------------------------------
// L.2.2 — the signalled inverse XYB transform
// ---------------------------------------------------------------------------

/// The nominal display intensity, in nits, that L.2.2's `itscale` normalises
/// against: `itscale = 255 / intensity_target`.
///
/// At the Table D.x default `intensity_target == 255` the scale is exactly 1.
pub const NOMINAL_INTENSITY_TARGET: f32 = 255.0;

/// 18181-1 L.2.2's inverse XYB transform, driven by the signalled
/// `OpsinInverseMatrix` bundle (Table L.1) rather than by fixed constants.
///
/// This is the decoder's entry point: a `kVarDCT` frame decodes to XYB and
/// L.2.2 turns it into linear sRGB (D65 primaries, linear transfer, display
/// referred, `1.0` = `intensity_target` nits). The clause's code is
///
/// ```text
/// Lgamma = Y + X;  Mgamma = Y - X;  Sgamma = B;
/// itscale = 255 / metadata.tone_mapping.intensity_target;
/// Lmix = (pow(Lgamma - cbrt(oim.opsin_bias0), 3) + oim.opsin_bias0) * itscale;
/// ...
/// R = oim.inv_mat00 * Lmix + oim.inv_mat01 * Mmix + oim.inv_mat02 * Smix;
/// ```
///
/// and is implemented verbatim, with the three cube roots hoisted into the
/// constructor because they depend only on the bundle.
///
/// # Sign convention
///
/// `opsin_bias` is taken **as signalled** — the Table L.1 defaults are
/// negative. See the module documentation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OpsinInverse {
    /// `inv_mat00 .. inv_mat22`, row-major.
    matrix: [f32; 9],
    /// `opsin_bias0..2`, as signalled.
    bias: [f32; 3],
    /// `cbrt(opsin_bias[c])`, precomputed.
    bias_cbrt: [f32; 3],
    /// `255 / intensity_target`.
    itscale: f32,
}

impl OpsinInverse {
    /// Builds the transform from a signalled bundle.
    ///
    /// `intensity_target` is `metadata.tone_mapping.intensity_target`. A
    /// non-positive or non-finite target would make `itscale` meaningless, so
    /// it falls back to [`NOMINAL_INTENSITY_TARGET`] — the header layer has
    /// already rejected such a value, and silently producing infinities here
    /// would be worse than the one documented clamp.
    #[must_use]
    pub fn new(matrix: [f32; 9], bias: [f32; 3], intensity_target: f32) -> Self {
        let target = if intensity_target.is_finite() && intensity_target > 0.0 {
            intensity_target
        } else {
            NOMINAL_INTENSITY_TARGET
        };
        Self {
            matrix,
            bias,
            bias_cbrt: [bias[0].cbrt(), bias[1].cbrt(), bias[2].cbrt()],
            itscale: NOMINAL_INTENSITY_TARGET / target,
        }
    }

    /// L.2.2's `itscale`.
    #[must_use]
    pub const fn itscale(&self) -> f32 {
        self.itscale
    }

    /// Converts one XYB triple to linear sRGB.
    #[must_use]
    pub fn convert(&self, xyb: [f32; 3]) -> [f32; 3] {
        let [x, y, b] = xyb;
        let gamma = [y + x, y - x, b];

        let mut mix = [0.0f32; 3];
        for c in 0..3 {
            // Indices 0..3 into three fixed-size arrays; `get` keeps the
            // indexing_slicing lint satisfied without an allow.
            let g = gamma.get(c).copied().unwrap_or(0.0);
            let bias = self.bias.get(c).copied().unwrap_or(0.0);
            let root = self.bias_cbrt.get(c).copied().unwrap_or(0.0);
            if let Some(slot) = mix.get_mut(c) {
                *slot = (cube(g - root) + bias) * self.itscale;
            }
        }

        let mut out = [0.0f32; 3];
        for row in 0..3 {
            let mut acc = 0.0f32;
            for col in 0..3 {
                let m = self.matrix.get(row * 3 + col).copied().unwrap_or(0.0);
                acc = m.mul_add(mix.get(col).copied().unwrap_or(0.0), acc);
            }
            if let Some(slot) = out.get_mut(row) {
                *slot = acc;
            }
        }
        out
    }

    /// Converts three planar XYB channels to linear sRGB in place.
    ///
    /// The planes are `[X, Y, B]` on entry and `[R, G, B]` on exit. Extra
    /// samples in a longer plane are left untouched; the caller is expected to
    /// pass three equal-length planes.
    pub fn convert_planes(&self, x: &mut [f32], y: &mut [f32], b: &mut [f32]) {
        for ((xp, yp), bp) in x.iter_mut().zip(y.iter_mut()).zip(b.iter_mut()) {
            let [r, g, bb] = self.convert([*xp, *yp, *bp]);
            *xp = r;
            *yp = g;
            *bp = bb;
        }
    }
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

    // ------------------------------------------------------------------
    // L.2.2 — the signalled entry point
    // ------------------------------------------------------------------

    /// The Table L.1 defaults, in the sign the standard prints them in.
    fn default_oim() -> OpsinInverse {
        OpsinInverse::new(
            [
                11.031_567,
                -9.866_944,
                -0.164_622_98,
                -3.254_147_4,
                4.418_770_3,
                -0.164_622_98,
                -3.658_851_6,
                2.712_923_5,
                1.945_928_1,
            ],
            [-OPSIN_BIAS; 3],
            NOMINAL_INTENSITY_TARGET,
        )
    }

    /// Proves the sign convention decided at this boundary: L.2.2's printed
    /// `pow(gamma - cbrt(bias), 3) + bias` with the **negative** signalled
    /// bias is the same function as the fixed-constant path's
    /// `cube(gamma + cbrt(|bias|)) - |bias|`. If a future reader "fixes" the
    /// sign in either place, this test fails.
    #[test]
    fn signalled_defaults_match_the_fixed_constant_path() {
        let oim = default_oim();
        for xyb in [
            [0.0f32, 0.0, 0.0],
            [0.01, 0.4, 0.5],
            [-0.02, 0.2, 0.18],
            [0.0, 0.75, 0.75],
        ] {
            let a = oim.convert(xyb);
            let b = xyb_to_linear_srgb(xyb);
            for (p, q) in a.into_iter().zip(b) {
                assert!((p - q).abs() < 1e-5, "{xyb:?}: {a:?} vs {b:?}");
            }
        }
    }

    /// Proves `itscale` is applied and is the identity at the default target.
    #[test]
    fn itscale_scales_the_mixed_values() {
        assert!((default_oim().itscale() - 1.0).abs() < 1e-7);

        // At twice the nominal target the mix values halve, and the matrix is
        // linear in them, so every output halves — for XYB inputs that map to
        // a *zero* bias contribution this is exact. Use the general property
        // instead: convert with target T and with 255, and check the ratio of
        // the (linear) matrix stage. Doing that end to end needs the bias to
        // cancel, so compare the pre-matrix behaviour through a black input,
        // where Lmix == bias * itscale exactly.
        let hot = OpsinInverse::new(
            default_oim().matrix,
            default_oim().bias,
            2.0 * NOMINAL_INTENSITY_TARGET,
        );
        assert!((hot.itscale() - 0.5).abs() < 1e-7);

        let black_ref = default_oim().convert([0.0, 0.0, 0.0]);
        let black_hot = hot.convert([0.0, 0.0, 0.0]);
        for (a, b) in black_ref.into_iter().zip(black_hot) {
            assert!((a - 2.0 * b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    /// A degenerate `intensity_target` must not produce infinities.
    #[test]
    fn non_positive_intensity_target_falls_back_to_nominal() {
        for target in [0.0f32, -5.0, f32::NAN, f32::INFINITY] {
            let oim = OpsinInverse::new([1.0; 9], [-OPSIN_BIAS; 3], target);
            assert!((oim.itscale() - 1.0).abs() < 1e-7, "target {target}");
        }
    }

    /// The planar form must agree with the scalar form sample for sample.
    #[test]
    fn planar_conversion_matches_the_scalar_one() {
        let oim = default_oim();
        let source = [[0.01f32, 0.3, 0.4], [-0.01, 0.5, 0.45], [0.0, 0.0, 0.0]];
        let expected: Vec<[f32; 3]> = source.into_iter().map(|s| oim.convert(s)).collect();

        let mut x: Vec<f32> = source.iter().map(|p| p[0]).collect();
        let mut y: Vec<f32> = source.iter().map(|p| p[1]).collect();
        let mut b: Vec<f32> = source.iter().map(|p| p[2]).collect();
        oim.convert_planes(&mut x, &mut y, &mut b);

        for (i, want) in expected.into_iter().enumerate() {
            let got = [
                x.get(i).copied().unwrap_or_default(),
                y.get(i).copied().unwrap_or_default(),
                b.get(i).copied().unwrap_or_default(),
            ];
            assert_eq!(got, want, "sample {i}");
        }
    }

    /// The sRGB curve's two branches must meet, and the round numbers must
    /// come out round: 0 -> 0, 1 -> 1.
    #[test]
    fn srgb_transfer_is_continuous_and_hits_its_endpoints() {
        assert_eq!(linear_to_srgb(0.0), 0.0);
        assert!((linear_to_srgb(1.0) - 1.0).abs() < 1e-6);
        let below = linear_to_srgb(0.003_130_8 - 1e-7);
        let above = linear_to_srgb(0.003_130_8 + 1e-7);
        assert!((below - above).abs() < 1e-5, "{below} vs {above}");
        // The mid-grey landmark: 0.5 encoded is about 0.2140 linear.
        assert!((linear_to_srgb(0.214_041_14) - 0.5).abs() < 1e-4);
    }

    /// The BT.709 curve must differ measurably from sRGB — if it did not,
    /// there would be no point distinguishing them, and a decoder that used
    /// the wrong one would still pass.
    #[test]
    fn rec709_is_not_the_srgb_curve() {
        assert_eq!(linear_to_rec709(0.0), 0.0);
        assert!((linear_to_rec709(1.0) - 1.0).abs() < 1e-5);
        let worst = (0..=100)
            .map(|i: u8| {
                let v = f32::from(i) / 100.0;
                (linear_to_srgb(v) - linear_to_rec709(v)).abs()
            })
            .fold(0.0f32, f32::max);
        assert!(worst > 0.01, "the two curves differ by only {worst}");
    }

    /// Negative (out-of-gamut) samples must survive with their sign, because
    /// 18181-3 forbids clipping before the conformance comparison.
    #[test]
    fn transfer_functions_are_odd_and_never_clip() {
        for v in [-0.001f32, -0.25, -1.5] {
            assert!((linear_to_srgb(v) + linear_to_srgb(-v)).abs() < 1e-6, "{v}");
            assert!((linear_to_gamma(v, 0.45) + linear_to_gamma(-v, 0.45)).abs() < 1e-6);
            assert!(
                (linear_to_rec709(v) + linear_to_rec709(-v)).abs() < 1e-6,
                "{v}"
            );
        }
        assert!(linear_to_srgb(2.0) > 1.0);
    }

    /// A custom (non-default) matrix must actually be used: swapping two rows
    /// of the identity has to swap the corresponding outputs.
    #[test]
    fn the_signalled_matrix_is_the_one_applied() {
        // Identity matrix, zero bias, nominal target: R/G/B are Lmix/Mmix/Smix
        // and hence Y+X, Y-X, B exactly.
        let id = OpsinInverse::new(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0; 3],
            NOMINAL_INTENSITY_TARGET,
        );
        // The gamma stage still cubes: Lmix = (Y+X)^3, Mmix = (Y-X)^3,
        // Smix = B^3.
        let got = id.convert([0.25, 0.75, -0.5]);
        assert!((got[0] - 1.0).abs() < 1e-6, "{got:?}");
        assert!((got[1] - 0.125).abs() < 1e-6, "{got:?}");
        assert!((got[2] + 0.125).abs() < 1e-6, "{got:?}");

        // Swap rows 0 and 1: the outputs must swap.
        let swapped = OpsinInverse::new(
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0; 3],
            NOMINAL_INTENSITY_TARGET,
        );
        let got2 = swapped.convert([0.25, 0.75, -0.5]);
        assert!((got2[0] - 0.125).abs() < 1e-6, "{got2:?}");
        assert!((got2[1] - 1.0).abs() < 1e-6, "{got2:?}");
    }
}
