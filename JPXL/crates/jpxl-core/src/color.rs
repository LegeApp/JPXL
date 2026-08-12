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
/// With `simd`, processes four pixels at a time: the opsin matrix is applied
/// with `wide::f32x4`, then per-lane scalar `cbrt` matches the single-pixel
/// path (Contract B vs a pure-`powf` SIMD cube root).
///
/// # Panics
///
/// Panics if the three planes do not have equal length.
pub fn linear_srgb_to_xyb_planes(r: &mut [f32], g: &mut [f32], b: &mut [f32]) {
    assert!(
        r.len() == g.len() && g.len() == b.len(),
        "planar XYB conversion needs three equally sized planes"
    );
    #[cfg(feature = "simd")]
    {
        linear_srgb_to_xyb_planes_simd(r, g, b);
    }
    #[cfg(not(feature = "simd"))]
    {
        for ((rp, gp), bp) in r.iter_mut().zip(g.iter_mut()).zip(b.iter_mut()) {
            let [x, y, bb] = linear_srgb_to_xyb([*rp, *gp, *bp]);
            *rp = x;
            *gp = y;
            *bp = bb;
        }
    }
}

#[cfg(feature = "simd")]
fn linear_srgb_to_xyb_planes_simd(r: &mut [f32], g: &mut [f32], b: &mut [f32]) {
    use wide::f32x4;
    let n = r.len();
    let [ml, mm, ms] = OPSIN_ABSORBANCE_MATRIX;
    let bias = f32x4::splat(OPSIN_BIAS);
    let bias_c = f32x4::splat(OPSIN_BIAS_CBRT);
    let half = f32x4::splat(0.5);
    let mut i = 0usize;
    while i + 4 <= n {
        let rv = f32x4::new([
            r.get(i).copied().unwrap_or(0.0),
            r.get(i + 1).copied().unwrap_or(0.0),
            r.get(i + 2).copied().unwrap_or(0.0),
            r.get(i + 3).copied().unwrap_or(0.0),
        ]);
        let gv = f32x4::new([
            g.get(i).copied().unwrap_or(0.0),
            g.get(i + 1).copied().unwrap_or(0.0),
            g.get(i + 2).copied().unwrap_or(0.0),
            g.get(i + 3).copied().unwrap_or(0.0),
        ]);
        let bv = f32x4::new([
            b.get(i).copied().unwrap_or(0.0),
            b.get(i + 1).copied().unwrap_or(0.0),
            b.get(i + 2).copied().unwrap_or(0.0),
            b.get(i + 3).copied().unwrap_or(0.0),
        ]);
        let lm =
            f32x4::splat(ml[0]) * rv + f32x4::splat(ml[1]) * gv + f32x4::splat(ml[2]) * bv + bias;
        let mm_ =
            f32x4::splat(mm[0]) * rv + f32x4::splat(mm[1]) * gv + f32x4::splat(mm[2]) * bv + bias;
        let sm =
            f32x4::splat(ms[0]) * rv + f32x4::splat(ms[1]) * gv + f32x4::splat(ms[2]) * bv + bias;
        // Scalar cbrt per lane so nonlinearities match linear_srgb_to_xyb.
        let lma = lm.to_array();
        let mma = mm_.to_array();
        let sma = sm.to_array();
        let lg = f32x4::new([lma[0].cbrt(), lma[1].cbrt(), lma[2].cbrt(), lma[3].cbrt()]) - bias_c;
        let mg = f32x4::new([mma[0].cbrt(), mma[1].cbrt(), mma[2].cbrt(), mma[3].cbrt()]) - bias_c;
        let sg = f32x4::new([sma[0].cbrt(), sma[1].cbrt(), sma[2].cbrt(), sma[3].cbrt()]) - bias_c;
        let x = half * (lg - mg);
        let y = half * (lg + mg);
        let xa = x.to_array();
        let ya = y.to_array();
        let sa = sg.to_array();
        for j in 0..4 {
            if let Some(slot) = r.get_mut(i + j) {
                *slot = xa[j];
            }
            if let Some(slot) = g.get_mut(i + j) {
                *slot = ya[j];
            }
            if let Some(slot) = b.get_mut(i + j) {
                *slot = sa[j];
            }
        }
        i += 4;
    }
    while i < n {
        let [x, y, bb] = linear_srgb_to_xyb([
            r.get(i).copied().unwrap_or(0.0),
            g.get(i).copied().unwrap_or(0.0),
            b.get(i).copied().unwrap_or(0.0),
        ]);
        if let Some(slot) = r.get_mut(i) {
            *slot = x;
        }
        if let Some(slot) = g.get_mut(i) {
            *slot = y;
        }
        if let Some(slot) = b.get_mut(i) {
            *slot = bb;
        }
        i += 1;
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
    #[cfg(feature = "simd")]
    {
        xyb_to_linear_srgb_planes_simd(x, y, b);
    }
    #[cfg(not(feature = "simd"))]
    {
        for ((xp, yp), bp) in x.iter_mut().zip(y.iter_mut()).zip(b.iter_mut()) {
            let [r, g, bb] = xyb_to_linear_srgb([*xp, *yp, *bp]);
            *xp = r;
            *yp = g;
            *bp = bb;
        }
    }
}

#[cfg(feature = "simd")]
fn xyb_to_linear_srgb_planes_simd(x: &mut [f32], y: &mut [f32], b: &mut [f32]) {
    use wide::f32x4;
    let n = x.len();
    let [ir, ig, ib] = OPSIN_ABSORBANCE_INVERSE_MATRIX;
    let bias = f32x4::splat(OPSIN_BIAS);
    let bias_c = f32x4::splat(OPSIN_BIAS_CBRT);
    let mut i = 0usize;
    while i + 4 <= n {
        let xv = f32x4::new([
            x.get(i).copied().unwrap_or(0.0),
            x.get(i + 1).copied().unwrap_or(0.0),
            x.get(i + 2).copied().unwrap_or(0.0),
            x.get(i + 3).copied().unwrap_or(0.0),
        ]);
        let yv = f32x4::new([
            y.get(i).copied().unwrap_or(0.0),
            y.get(i + 1).copied().unwrap_or(0.0),
            y.get(i + 2).copied().unwrap_or(0.0),
            y.get(i + 3).copied().unwrap_or(0.0),
        ]);
        let bv = f32x4::new([
            b.get(i).copied().unwrap_or(0.0),
            b.get(i + 1).copied().unwrap_or(0.0),
            b.get(i + 2).copied().unwrap_or(0.0),
            b.get(i + 3).copied().unwrap_or(0.0),
        ]);
        let lg = yv + xv;
        let mg = yv - xv;
        let sg = bv;
        // cube(v + bias_cbrt) - bias  (scalar cube per lane for exactness)
        let cube_lane = |t: f32x4| -> f32x4 {
            let a = (t + bias_c).to_array();
            f32x4::new([cube(a[0]), cube(a[1]), cube(a[2]), cube(a[3])]) - bias
        };
        let lm = cube_lane(lg);
        let mm = cube_lane(mg);
        let sm = cube_lane(sg);
        let r = f32x4::splat(ir[0]) * lm + f32x4::splat(ir[1]) * mm + f32x4::splat(ir[2]) * sm;
        let g = f32x4::splat(ig[0]) * lm + f32x4::splat(ig[1]) * mm + f32x4::splat(ig[2]) * sm;
        let bb = f32x4::splat(ib[0]) * lm + f32x4::splat(ib[1]) * mm + f32x4::splat(ib[2]) * sm;
        let ra = r.to_array();
        let ga = g.to_array();
        let ba = bb.to_array();
        for j in 0..4 {
            if let Some(slot) = x.get_mut(i + j) {
                *slot = ra[j];
            }
            if let Some(slot) = y.get_mut(i + j) {
                *slot = ga[j];
            }
            if let Some(slot) = b.get_mut(i + j) {
                *slot = ba[j];
            }
        }
        i += 4;
    }
    while i < n {
        let [r, g, bb] = xyb_to_linear_srgb([
            x.get(i).copied().unwrap_or(0.0),
            y.get(i).copied().unwrap_or(0.0),
            b.get(i).copied().unwrap_or(0.0),
        ]);
        if let Some(slot) = x.get_mut(i) {
            *slot = r;
        }
        if let Some(slot) = y.get_mut(i) {
            *slot = g;
        }
        if let Some(slot) = b.get_mut(i) {
            *slot = bb;
        }
        i += 1;
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

/// Flip point: how a piecewise OETF with a linear toe is extended below zero.
///
/// L.2.2's output is explicitly allowed outside the coded gamut and 18181-3
/// §4.2 forbids clipping before the conformance comparison, so a decoder that
/// signals `TransferFunction::kSRGB` or `k709` has to evaluate its OETF at
/// negative arguments. Neither 18181-1 Table E.6 nor the referenced transfer
/// standards (IEC 61966-2-1, ITU-R BT.709-6) says what happens there: both
/// print a two-branch curve whose stated domain starts at zero.
///
/// Two readings are defensible:
///
/// * *literal* — evaluate the printed condition on the **signed** value.
///   Every negative argument is below the toe threshold, so it takes the
///   linear branch: `12.92 * v` (sRGB), `4.5 * v` (BT.709).
/// * *odd* — extend the whole curve with odd symmetry, `f(-v) == -f(v)`, so a
///   negative argument takes the *power* branch on its magnitude.
///
/// The two disagree by a lot — at `v == -0.13` BT.709 gives `-0.587` literally
/// and `-0.341` under odd symmetry — and, measured against published
/// references, **the two curves do not answer the same way**: this array is
/// `[sRGB, BT.709] = [odd, literal]`. Both entries are measurements, not
/// deductions; see
/// `docs/experiments/2026-08-04-negative-transfer-function-branch.md` for the
/// evidence and the residuals.
///
/// [`linear_to_gamma`] is not covered: a pure power law has no linear branch,
/// so odd symmetry is the only continuous extension available there.
pub const NEGATIVES_TAKE_THE_LINEAR_SEGMENT: [bool; 2] = [false, true];

/// The IEC 61966-2-1 (sRGB) opto-electronic transfer function.
///
/// L.2.2 hands back **linear** light; the samples a conformance reference or a
/// PNG holds are in the signalled colour encoding, whose transfer function for
/// `TransferFunction::kSRGB` is this curve.
///
/// Negative (out-of-gamut) samples are extended with odd symmetry,
/// `f(-v) == -f(v)` — see [`NEGATIVES_TAKE_THE_LINEAR_SEGMENT`], entry 0.
/// Nothing is clipped.
#[must_use]
pub fn linear_to_srgb(v: f32) -> f32 {
    if NEGATIVES_TAKE_THE_LINEAR_SEGMENT[0] && v <= 0.003_130_8 {
        return 12.92 * v;
    }
    let a = v.abs();
    let encoded = if a <= 0.003_130_8 {
        12.92 * a
    } else {
        1.055 * a.powf(1.0 / 2.4) - 0.055
    };
    if v < 0.0 { -encoded } else { encoded }
}

/// The inverse of [`linear_to_srgb`]: the IEC 61966-2-1 electro-optical
/// transfer function.
///
/// An encoder starts where a decoder stops. A PNG or Netpbm sample is in the
/// signalled colour encoding, and L.2's forward direction wants linear light,
/// so this is the first stage of any encode from 8-bit sRGB.
///
/// It is written as the algebraic inverse of [`linear_to_srgb`], branch for
/// branch and constant for constant, including the negative extension: with
/// [`NEGATIVES_TAKE_THE_LINEAR_SEGMENT`] entry 0 false the sRGB curve is odd,
/// so this one is too. `srgb_to_linear(linear_to_srgb(v)) == v` to `f32`
/// rounding for every finite `v`, which is what
/// [`tests::the_srgb_transfer_pair_round_trips`] asserts.
#[must_use]
pub fn srgb_to_linear(v: f32) -> f32 {
    if NEGATIVES_TAKE_THE_LINEAR_SEGMENT[0] && v <= 12.92 * 0.003_130_8 {
        return v / 12.92;
    }
    let a = v.abs();
    let decoded = if a <= 12.92 * 0.003_130_8 {
        a / 12.92
    } else {
        ((a + 0.055) / 1.055).powf(2.4)
    };
    if v < 0.0 { -decoded } else { decoded }
}

/// The ITU-R BT.709-6 opto-electronic transfer function.
///
/// `E' = 4.5 E` below `0.018` and `1.099 E^0.45 - 0.099` above. This is the
/// curve `TransferFunction::k709` names, and it is *not* the sRGB curve — the
/// two differ by up to about 0.02 in the shadows, which is five times the
/// no-filters conformance budget, so the distinction is load-bearing.
///
/// The difference extends below zero, where the two curves are extended
/// *differently*: this one takes its printed condition on the signed value, so
/// a negative sample lands on the linear branch and encodes as `4.5 * v` —
/// see [`NEGATIVES_TAKE_THE_LINEAR_SEGMENT`], entry 1.
#[must_use]
pub fn linear_to_rec709(v: f32) -> f32 {
    if NEGATIVES_TAKE_THE_LINEAR_SEGMENT[1] && v < 0.018 {
        return 4.5 * v;
    }
    let a = v.abs();
    let encoded = if a < 0.018 {
        4.5 * a
    } else {
        1.099 * a.powf(0.45) - 0.099
    };
    if v < 0.0 { -encoded } else { encoded }
}

/// A pure power-law OETF, `v^exponent`, extended below zero with odd
/// symmetry.
///
/// Unlike [`linear_to_rec709`] this curve has no linear toe to fall back on,
/// so odd symmetry is the only continuous, monotonic, sign-preserving
/// extension available. See [`NEGATIVES_TAKE_THE_LINEAR_SEGMENT`].
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

/// Table L.1's default `quant_bias`, the small-coefficient shrink of I.5.3.
///
/// I.5.3 adjusts every quantized HF coefficient before dequantizing it:
/// `|q| <= 1` is multiplied by `quant_bias[channel]`, everything else has
/// `quant_bias_numerator / q` subtracted. Both directions need the same
/// numbers — a decoder to reconstruct, an encoder to *choose* the integer whose
/// reconstruction is nearest its target — so they live here rather than only on
/// the read side.
///
/// The printed defaults are the expressions `1 - 0.05465007330715401`,
/// `1 - 0.07005449891748593` and `1 - 0.049935103337343655`; both OCR
/// conversions collapsed the leading `1 -` into an ambiguous glyph, and the
/// values below are the evaluated results of the scan-verified expressions.
/// `jpxl-decode`'s `headers::opsin` carries the same three constants for the
/// signalled-bundle path, and the encoder's test suite asserts the two agree —
/// see `crates/jpxl-encode/tests/vardct_roundtrip.rs`.
pub const DEFAULT_QUANT_BIAS: [f32; 3] =
    [1.0 - 0.054_650_073, 1.0 - 0.070_054_5, 1.0 - 0.049_935_103];

/// Table L.1's default `quant_bias_numerator`. See [`DEFAULT_QUANT_BIAS`].
pub const DEFAULT_QUANT_BIAS_NUMERATOR: f32 = 0.145;

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

    /// The encode-side EOTF must undo the decode-side OETF exactly, including
    /// below zero and outside `[0, 1]`, or an 8-bit sRGB encode starts from
    /// the wrong linear values and every later measurement is off by a
    /// constant nobody can localise.
    #[test]
    fn the_srgb_transfer_pair_round_trips() {
        let mut worst = 0.0f32;
        for i in -200i32..=400 {
            let v = i as f32 / 200.0;
            let back = srgb_to_linear(linear_to_srgb(v));
            worst = worst.max((back - v).abs());
            assert!(
                (back - v).abs() < 1e-5 * v.abs().max(1.0),
                "linear {v} -> {} -> {back}",
                linear_to_srgb(v)
            );
        }
        assert!(worst < 1e-4, "worst sRGB round-trip error {worst}");

        // Landmarks: 0, 1 and mid-grey, from the encoded side.
        assert_eq!(srgb_to_linear(0.0), 0.0);
        assert!((srgb_to_linear(1.0) - 1.0).abs() < 1e-6);
        assert!((srgb_to_linear(0.5) - 0.214_041_14).abs() < 1e-5);
        // Every 8-bit code point survives the trip back and forth.
        for code in 0..=255u16 {
            let encoded = f32::from(code) / 255.0;
            let back = linear_to_srgb(srgb_to_linear(encoded));
            assert!((back - encoded).abs() < 1e-6, "code {code}");
        }
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
    ///
    /// Which *branch* they take is [`NEGATIVES_TAKE_THE_LINEAR_SEGMENT`];
    /// this test only pins sign preservation, strict monotonicity and the
    /// absence of clipping, which hold under either reading.
    #[test]
    fn transfer_functions_preserve_sign_and_never_clip() {
        for v in [-0.001f32, -0.25, -1.5] {
            assert!(linear_to_srgb(v) < 0.0, "{v}");
            assert!(linear_to_rec709(v) < 0.0, "{v}");
            assert!(linear_to_gamma(v, 0.45) < 0.0, "{v}");
        }
        for pair in [(-1.5f32, -0.25), (-0.25, -0.001), (-0.001, 0.0)] {
            assert!(linear_to_srgb(pair.0) < linear_to_srgb(pair.1), "{pair:?}");
            assert!(
                linear_to_rec709(pair.0) < linear_to_rec709(pair.1),
                "{pair:?}"
            );
        }
        assert!(linear_to_srgb(2.0) > 1.0);
        // A pure power law has no toe, so it stays odd whatever the flip
        // point says.
        assert!((linear_to_gamma(-0.25, 0.45) + linear_to_gamma(0.25, 0.45)).abs() < 1e-6);
    }

    /// The flip point itself, in both directions. Below zero BT.709 takes its
    /// *linear* branch while sRGB stays odd — measured, not deduced (see the
    /// experiment note). Flipping BT.709 back to odd moves `bike`'s blue
    /// channel by up to 0.25, 35x its conformance budget; flipping sRGB to the
    /// linear branch moves an out-of-gamut synthetic by over 1.0.
    #[test]
    fn negatives_take_the_branch_each_curve_was_measured_to_take() {
        assert_eq!(NEGATIVES_TAKE_THE_LINEAR_SEGMENT, [false, true]);
        for v in [-0.001f32, -0.13, -0.25, -1.5] {
            assert!((linear_to_rec709(v) - 4.5 * v).abs() < 1e-6, "709 {v}");
            assert!(
                (linear_to_srgb(v) + linear_to_srgb(-v)).abs() < 1e-6,
                "sRGB {v}"
            );
        }
        // -0.13 is where the two readings are furthest apart in practice.
        assert!((linear_to_rec709(-0.13) + 0.585).abs() < 1e-3);
        assert!((linear_to_srgb(-0.13) + 0.396).abs() < 1e-3);
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
