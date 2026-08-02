//! Discrete cosine transforms for the VarDCT path (slice 8 of `docs/PLAN.md`).
//!
//! This module is *spec-independent* transform math: it implements the textbook
//! DCT-II (analysis / "forward") and DCT-III (synthesis / "inverse") pair for
//! the block sizes VarDCT needs, and nothing else. No bitstream concepts, no
//! quantization, no JPEG XL constants live here. Everything was derived from
//! the standard definitions below; no reference implementation was consulted.
//!
//! # Definitions
//!
//! For a length-`N` signal `x`, the forward transform computed here is
//!
//! ```text
//! X[u] = s(u) * sum_{n=0}^{N-1} x[n] * cos(pi * (2n + 1) * u / (2N))
//! s(0) = sqrt(1/N),   s(u) = sqrt(2/N) for u > 0
//! ```
//!
//! and the inverse is its transpose,
//!
//! ```text
//! x[n] = sum_{u=0}^{N-1} s(u) * X[u] * cos(pi * (2n + 1) * u / (2N)).
//! ```
//!
//! # Normalization convention: ORTHONORMAL
//!
//! The `s(u)` factors above make the transform matrix orthogonal, so that:
//!
//! * `idct(dct(x)) == x` exactly in exact arithmetic (round trip is identity,
//!   with no external `1/N`, `2/N` or `4/N` fudge factor anywhere),
//! * Parseval holds: `sum(x[n]^2) == sum(X[u]^2)`,
//! * a DC-only coefficient block decodes to a constant-valued sample block,
//! * the 2-D transform is just the 1-D transform applied to rows then columns,
//!   with no extra scaling in between.
//!
//! This convention is chosen because it is self-checking: every test in this
//! file is an identity that fails loudly if a scale factor drifts. It is
//! deliberately *not* asserted to be JPEG XL's convention.
//!
//! **\[provisional: verify scaling vs 18181-1 OCR\]** JPEG XL's VarDCT is
//! specified with its own scaling (and its own DC/AC ordering, and per-size
//! normalization for the rectangular and large transforms). When the spec text
//! is available, the *only* thing expected to change here is a scalar prefactor
//! applied at the call boundary — the butterfly structure and the coefficient
//! layout documented below are convention-independent. Do not bake a
//! JPEG-XL-specific factor into these kernels; wrap them instead.
//!
//! # Coefficient layout (read this before debugging a VarDCT bug)
//!
//! All 2-D buffers are **row-major**. For an `H`-row by `W`-column block,
//! sample `(row, col)` lives at index `row * W + col`, and after a forward
//! transform coefficient `(u, v)` lives at index **`u * W + v`**, where
//!
//! * `u` is the **vertical** frequency (varies down the block, index of the
//!   column-direction transform),
//! * `v` is the **horizontal** frequency (varies across a row).
//!
//! So for 8x8, coefficient `(u, v)` is at `u * 8 + v`; DC is index 0; the
//! lowest *horizontal* frequency AC coefficient is index 1; the lowest
//! *vertical* frequency AC coefficient is index 8. Transposing these two is
//! the single most common VarDCT failure mode, so the tests below pin the
//! layout explicitly rather than only checking round trips.
//!
//! # Structure
//!
//! Two implementations of the same math are kept side by side:
//!
//! * [`dct2_naive`] / [`dct3_naive`] — direct `O(N^2)` evaluation of the
//!   definitions in `f64`. This is the oracle, not the production path.
//! * `dct_ii_*` / `dct_iii_*` — a factored `O(N log N)`-ish version built from
//!   an even/odd decomposition. The fast and naive paths are cross-checked in
//!   the tests.
//!
//! The fast path is written as straight-line operations on fixed-size arrays so
//! a later SIMD pass can lift the butterflies to vectors and the DCT-IV leaves
//! to small matrix products without restructuring anything.
//!
//! # Fast-path derivation
//!
//! Split the input into sums and differences of mirrored pairs,
//! `a[n] = x[n] + x[N-1-n]`, `b[n] = x[n] - x[N-1-n]`, for `n < M`, `M = N/2`.
//! Using `cos(pi*(2(N-1-n)+1)*k/(2N)) = (-1)^k * cos(pi*(2n+1)*k/(2N))`, the
//! even-indexed outputs depend only on `a` and the odd-indexed only on `b`:
//!
//! ```text
//! X[2u]     = DCT-II_M (a)[u] / sqrt(2)
//! X[2u + 1] = DCT-IV_M (b)[u] / sqrt(2)
//! ```
//!
//! where DCT-IV is the orthonormal transform with kernel
//! `sqrt(2/M) * cos(pi * (2n+1) * (2u+1) / (4M))`. Both scale factors come out
//! to the same `1/sqrt(2)`, which is what keeps the recursion clean. The
//! inverse is the exact transpose of that network: scale, run DCT-III on the
//! even coefficients and DCT-IV (which is symmetric, hence self-transposed) on
//! the odd ones, then undo the mirrored butterfly.

#![allow(clippy::indexing_slicing)]

use core::f32::consts::FRAC_1_SQRT_2;

// ---------------------------------------------------------------------------
// DCT-IV leaf kernels
// ---------------------------------------------------------------------------
//
// These are the leaves of the even/odd recursion. They are kept as explicit
// matrix products: at these sizes that is a handful of fused multiply-adds, it
// is trivially verifiable against the definition, and it is the shape a SIMD
// pass wants anyway. Entries are `sqrt(2/M) * cos(pi*(2n+1)*(2u+1)/(4M))`.

/// `cos(pi/8)`, the DCT-IV_2 rotation cosine.
const COS_PI_8: f32 = 0.923_879_5;
/// `sin(pi/8)`, the DCT-IV_2 rotation sine.
const SIN_PI_8: f32 = 0.382_683_43;

#[rustfmt::skip]
const DCT4_4: [[f32; 4]; 4] = [
    [0.693_519_95, 0.587_937_8, 0.392_847_48, 0.137_949_69],
    [0.587_937_8, -0.137_949_69, -0.693_519_95, -0.392_847_48],
    [0.392_847_48, -0.693_519_95, 0.137_949_69, 0.587_937_8],
    [0.137_949_69, -0.392_847_48, 0.587_937_8, -0.693_519_95],
];

#[rustfmt::skip]
const DCT4_8: [[f32; 8]; 8] = [
    [0.497_592_36, 0.478_470_18, 0.440_960_65, 0.386_505_22,
     0.317_196_64, 0.235_698_37, 0.145_142_33, 0.049_008_57],
    [0.478_470_18, 0.317_196_64, 0.049_008_57, -0.235_698_37,
     -0.440_960_65, -0.497_592_36, -0.386_505_22, -0.145_142_33],
    [0.440_960_65, 0.049_008_57, -0.386_505_22, -0.478_470_18,
     -0.145_142_33, 0.317_196_64, 0.497_592_36, 0.235_698_37],
    [0.386_505_22, -0.235_698_37, -0.478_470_18, 0.049_008_57,
     0.497_592_36, 0.145_142_33, -0.440_960_65, -0.317_196_64],
    [0.317_196_64, -0.440_960_65, -0.145_142_33, 0.497_592_36,
     -0.049_008_57, -0.478_470_18, 0.235_698_37, 0.386_505_22],
    [0.235_698_37, -0.497_592_36, 0.317_196_64, 0.145_142_33,
     -0.478_470_18, 0.386_505_22, 0.049_008_57, -0.440_960_65],
    [0.145_142_33, -0.386_505_22, 0.497_592_36, -0.440_960_65,
     0.235_698_37, 0.049_008_57, -0.317_196_64, 0.478_470_18],
    [0.049_008_57, -0.145_142_33, 0.235_698_37, -0.317_196_64,
     0.386_505_22, -0.440_960_65, 0.478_470_18, -0.497_592_36],
];

/// Orthonormal 2-point DCT-IV, in place. A single Givens rotation.
fn dct_iv_2(v: &mut [f32; 2]) {
    let (x0, x1) = (v[0], v[1]);
    v[0] = COS_PI_8 * x0 + SIN_PI_8 * x1;
    v[1] = SIN_PI_8 * x0 - COS_PI_8 * x1;
}

/// Orthonormal 4-point DCT-IV, in place.
fn dct_iv_4(v: &mut [f32; 4]) {
    let x = *v;
    for (out, row) in v.iter_mut().zip(DCT4_4.iter()) {
        let mut acc = 0.0;
        for (c, xn) in row.iter().zip(x.iter()) {
            acc += c * xn;
        }
        *out = acc;
    }
}

/// Orthonormal 8-point DCT-IV, in place.
fn dct_iv_8(v: &mut [f32; 8]) {
    let x = *v;
    for (out, row) in v.iter_mut().zip(DCT4_8.iter()) {
        let mut acc = 0.0;
        for (c, xn) in row.iter().zip(x.iter()) {
            acc += c * xn;
        }
        *out = acc;
    }
}

// ---------------------------------------------------------------------------
// 1-D fast kernels
// ---------------------------------------------------------------------------

/// Orthonormal 2-point DCT-II (also its own DCT-III), in place.
fn dct_ii_2(v: &mut [f32; 2]) {
    let (x0, x1) = (v[0], v[1]);
    v[0] = (x0 + x1) * FRAC_1_SQRT_2;
    v[1] = (x0 - x1) * FRAC_1_SQRT_2;
}

/// Orthonormal 4-point DCT-II, in place.
fn dct_ii_4(v: &mut [f32; 4]) {
    let mut even = [v[0] + v[3], v[1] + v[2]];
    let mut odd = [v[0] - v[3], v[1] - v[2]];
    dct_ii_2(&mut even);
    dct_iv_2(&mut odd);
    v[0] = even[0] * FRAC_1_SQRT_2;
    v[1] = odd[0] * FRAC_1_SQRT_2;
    v[2] = even[1] * FRAC_1_SQRT_2;
    v[3] = odd[1] * FRAC_1_SQRT_2;
}

/// Orthonormal 4-point DCT-III, in place. Transpose of [`dct_ii_4`].
fn dct_iii_4(v: &mut [f32; 4]) {
    let mut even = [v[0] * FRAC_1_SQRT_2, v[2] * FRAC_1_SQRT_2];
    let mut odd = [v[1] * FRAC_1_SQRT_2, v[3] * FRAC_1_SQRT_2];
    dct_ii_2(&mut even);
    dct_iv_2(&mut odd);
    v[0] = even[0] + odd[0];
    v[3] = even[0] - odd[0];
    v[1] = even[1] + odd[1];
    v[2] = even[1] - odd[1];
}

/// Orthonormal 8-point DCT-II (forward transform), in place.
///
/// Output `v[u]` is the coefficient for frequency `u`; `v[0]` is DC.
pub fn dct_ii_8(v: &mut [f32; 8]) {
    let mut even = [v[0] + v[7], v[1] + v[6], v[2] + v[5], v[3] + v[4]];
    let mut odd = [v[0] - v[7], v[1] - v[6], v[2] - v[5], v[3] - v[4]];
    dct_ii_4(&mut even);
    dct_iv_4(&mut odd);
    for k in 0..4 {
        v[2 * k] = even[k] * FRAC_1_SQRT_2;
        v[2 * k + 1] = odd[k] * FRAC_1_SQRT_2;
    }
}

/// Orthonormal 8-point DCT-III (inverse transform), in place.
///
/// Exact inverse of [`dct_ii_8`]: input `v[u]` is the coefficient for
/// frequency `u`, output `v[n]` is sample `n`.
pub fn dct_iii_8(v: &mut [f32; 8]) {
    let mut even = [0.0f32; 4];
    let mut odd = [0.0f32; 4];
    for k in 0..4 {
        even[k] = v[2 * k] * FRAC_1_SQRT_2;
        odd[k] = v[2 * k + 1] * FRAC_1_SQRT_2;
    }
    dct_iii_4(&mut even);
    dct_iv_4(&mut odd);
    for k in 0..4 {
        v[k] = even[k] + odd[k];
        v[7 - k] = even[k] - odd[k];
    }
}

/// Orthonormal 16-point DCT-II (forward transform), in place.
pub fn dct_ii_16(v: &mut [f32; 16]) {
    let mut even = [0.0f32; 8];
    let mut odd = [0.0f32; 8];
    for k in 0..8 {
        even[k] = v[k] + v[15 - k];
        odd[k] = v[k] - v[15 - k];
    }
    dct_ii_8(&mut even);
    dct_iv_8(&mut odd);
    for k in 0..8 {
        v[2 * k] = even[k] * FRAC_1_SQRT_2;
        v[2 * k + 1] = odd[k] * FRAC_1_SQRT_2;
    }
}

/// Orthonormal 16-point DCT-III (inverse transform), in place.
///
/// Exact inverse of [`dct_ii_16`].
pub fn dct_iii_16(v: &mut [f32; 16]) {
    let mut even = [0.0f32; 8];
    let mut odd = [0.0f32; 8];
    for k in 0..8 {
        even[k] = v[2 * k] * FRAC_1_SQRT_2;
        odd[k] = v[2 * k + 1] * FRAC_1_SQRT_2;
    }
    dct_iii_8(&mut even);
    dct_iv_8(&mut odd);
    for k in 0..8 {
        v[k] = even[k] + odd[k];
        v[15 - k] = even[k] - odd[k];
    }
}

// ---------------------------------------------------------------------------
// Separable 2-D transforms
// ---------------------------------------------------------------------------

/// Applies a 1-D kernel of length `W` to every row of a row-major block.
fn transform_rows<const W: usize>(block: &mut [f32], kernel: fn(&mut [f32; W])) {
    for row in block.chunks_exact_mut(W) {
        let mut buf = [0.0f32; W];
        buf.copy_from_slice(row);
        kernel(&mut buf);
        row.copy_from_slice(&buf);
    }
}

/// Applies a 1-D kernel of length `H` to every column of a row-major block
/// that is `width` samples wide.
fn transform_cols<const H: usize>(block: &mut [f32], width: usize, kernel: fn(&mut [f32; H])) {
    for col in 0..width {
        let mut buf = [0.0f32; H];
        for (r, slot) in buf.iter_mut().enumerate() {
            *slot = block[r * width + col];
        }
        kernel(&mut buf);
        for (r, slot) in buf.iter().enumerate() {
            block[r * width + col] = *slot;
        }
    }
}

/// Forward 8x8 DCT-II, in place on a row-major block.
///
/// Input is samples in raster order (`row * 8 + col`); output is coefficients
/// with `(u, v)` at index `u * 8 + v`, `u` vertical frequency, `v` horizontal.
pub fn dct2d_8x8(block: &mut [f32; 64]) {
    transform_rows::<8>(block, dct_ii_8);
    transform_cols::<8>(block, 8, dct_ii_8);
}

/// Inverse 8x8 DCT-III, in place. Exact inverse of [`dct2d_8x8`].
pub fn idct2d_8x8(block: &mut [f32; 64]) {
    transform_cols::<8>(block, 8, dct_iii_8);
    transform_rows::<8>(block, dct_iii_8);
}

/// Forward 16x16 DCT-II, in place on a row-major block.
///
/// Coefficient `(u, v)` lands at index `u * 16 + v`.
pub fn dct2d_16x16(block: &mut [f32; 256]) {
    transform_rows::<16>(block, dct_ii_16);
    transform_cols::<16>(block, 16, dct_ii_16);
}

/// Inverse 16x16 DCT-III, in place. Exact inverse of [`dct2d_16x16`].
pub fn idct2d_16x16(block: &mut [f32; 256]) {
    transform_cols::<16>(block, 16, dct_iii_16);
    transform_rows::<16>(block, dct_iii_16);
}

/// Forward DCT-II on a block of **16 rows by 8 columns**, in place.
///
/// The buffer is row-major with a stride of 8, so sample `(row, col)` is at
/// `row * 8 + col` and coefficient `(u, v)` is at `u * 8 + v` with `u` in
/// `0..16` (vertical frequency) and `v` in `0..8` (horizontal frequency).
pub fn dct2d_16x8(block: &mut [f32; 128]) {
    transform_rows::<8>(block, dct_ii_8);
    transform_cols::<16>(block, 8, dct_ii_16);
}

/// Inverse DCT-III for a 16-row by 8-column block. Inverse of [`dct2d_16x8`].
pub fn idct2d_16x8(block: &mut [f32; 128]) {
    transform_cols::<16>(block, 8, dct_iii_16);
    transform_rows::<8>(block, dct_iii_8);
}

/// Forward DCT-II on a block of **8 rows by 16 columns**, in place.
///
/// The buffer is row-major with a stride of 16, so coefficient `(u, v)` is at
/// `u * 16 + v` with `u` in `0..8` and `v` in `0..16`.
pub fn dct2d_8x16(block: &mut [f32; 128]) {
    transform_rows::<16>(block, dct_ii_16);
    transform_cols::<8>(block, 16, dct_ii_8);
}

/// Inverse DCT-III for an 8-row by 16-column block. Inverse of [`dct2d_8x16`].
pub fn idct2d_8x16(block: &mut [f32; 128]) {
    transform_cols::<8>(block, 16, dct_iii_8);
    transform_rows::<16>(block, dct_iii_16);
}

// ---------------------------------------------------------------------------
// Naive reference implementations
// ---------------------------------------------------------------------------

/// Orthonormal DCT-II evaluated directly from the definition, in `f64`.
///
/// Reference oracle for the factored kernels; `O(n^2)` and not for production
/// use. `input` and `output` must both hold at least `n` values.
// Kept crate-visible (not `cfg(test)`) so other slices can use it as an
// oracle; unused in the production path by design.
#[allow(dead_code)]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub(crate) fn dct2_naive(input: &[f32], output: &mut [f32], n: usize) {
    let nf = n as f64;
    for (u, out) in output.iter_mut().take(n).enumerate() {
        let scale = if u == 0 { 1.0 / nf } else { 2.0 / nf }.sqrt();
        let mut acc = 0.0f64;
        for (x, sample) in input.iter().take(n).enumerate() {
            let angle = core::f64::consts::PI * ((2 * x + 1) as f64) * (u as f64) / (2.0 * nf);
            acc += f64::from(*sample) * angle.cos();
        }
        *out = (scale * acc) as f32;
    }
}

/// Orthonormal DCT-III evaluated directly from the definition, in `f64`.
///
/// Reference oracle and exact inverse of [`dct2_naive`].
#[allow(dead_code)]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub(crate) fn dct3_naive(input: &[f32], output: &mut [f32], n: usize) {
    let nf = n as f64;
    for (x, out) in output.iter_mut().take(n).enumerate() {
        let mut acc = 0.0f64;
        for (u, coeff) in input.iter().take(n).enumerate() {
            let scale = if u == 0 { 1.0 / nf } else { 2.0 / nf }.sqrt();
            let angle = core::f64::consts::PI * ((2 * x + 1) as f64) * (u as f64) / (2.0 * nf);
            acc += scale * f64::from(*coeff) * angle.cos();
        }
        *out = acc as f32;
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
mod tests {
    use super::*;

    /// Deterministic linear congruential generator (Numerical Recipes
    /// constants) so the tests need no `rand` dependency.
    struct Lcg(u32);

    impl Lcg {
        fn new(seed: u32) -> Self {
            Self(seed)
        }

        /// Uniform value in `[-scale, scale)`.
        fn next(&mut self, scale: f32) -> f32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = f32::from(u16::try_from(self.0 >> 16).unwrap_or(0)) / 65536.0;
            (unit * 2.0 - 1.0) * scale
        }
    }

    fn assert_close(a: f32, b: f32, tol: f32, what: &str) {
        assert!(
            (a - b).abs() <= tol,
            "{what}: {a} vs {b} (delta {})",
            (a - b).abs()
        );
    }

    fn assert_slice_close(a: &[f32], b: &[f32], tol: f32, what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_close(*x, *y, tol, &format!("{what}[{i}]"));
        }
    }

    // -- 1-D: fast kernels agree with the definition ------------------------

    #[test]
    fn dct_ii_8_matches_naive() {
        let mut rng = Lcg::new(0x1234_5678);
        for _ in 0..64 {
            let mut fast = [0.0f32; 8];
            for slot in &mut fast {
                *slot = rng.next(1.0);
            }
            let input = fast;
            let mut reference = [0.0f32; 8];
            dct2_naive(&input, &mut reference, 8);
            dct_ii_8(&mut fast);
            assert_slice_close(&fast, &reference, 1e-5, "dct_ii_8");
        }
    }

    #[test]
    fn dct_iii_8_matches_naive() {
        let mut rng = Lcg::new(0x0bad_c0de);
        for _ in 0..64 {
            let mut fast = [0.0f32; 8];
            for slot in &mut fast {
                *slot = rng.next(1.0);
            }
            let input = fast;
            let mut reference = [0.0f32; 8];
            dct3_naive(&input, &mut reference, 8);
            dct_iii_8(&mut fast);
            assert_slice_close(&fast, &reference, 1e-5, "dct_iii_8");
        }
    }

    #[test]
    fn dct_ii_16_matches_naive() {
        let mut rng = Lcg::new(0xfeed_face);
        for _ in 0..64 {
            let mut fast = [0.0f32; 16];
            for slot in &mut fast {
                *slot = rng.next(1.0);
            }
            let input = fast;
            let mut reference = [0.0f32; 16];
            dct2_naive(&input, &mut reference, 16);
            dct_ii_16(&mut fast);
            assert_slice_close(&fast, &reference, 1e-5, "dct_ii_16");
        }
    }

    #[test]
    fn dct_iii_16_matches_naive() {
        let mut rng = Lcg::new(0x5eed_1e55);
        for _ in 0..64 {
            let mut fast = [0.0f32; 16];
            for slot in &mut fast {
                *slot = rng.next(1.0);
            }
            let input = fast;
            let mut reference = [0.0f32; 16];
            dct3_naive(&input, &mut reference, 16);
            dct_iii_16(&mut fast);
            assert_slice_close(&fast, &reference, 1e-5, "dct_iii_16");
        }
    }

    /// Hand-derived value. For a constant input `x[n] = 1` (n = 0..7) only the
    /// DC term survives, because for `u > 0` the cosines over a full set of
    /// half-integer sample points sum to zero. For `u = 0`:
    ///
    /// ```text
    /// X[0] = sqrt(1/8) * sum_{n=0}^{7} 1 * cos(0) = 8 / sqrt(8) = sqrt(8)
    ///      = 2.8284271...
    /// ```
    ///
    /// Second hand-derived value, unit impulse `x = [1, 0, 0, 0, 0, 0, 0, 0]`:
    ///
    /// ```text
    /// X[0] = sqrt(1/8) * cos(0)      = 0.35355339
    /// X[1] = sqrt(2/8) * cos(pi/16)  = 0.5 * 0.98078528 = 0.49039264
    /// ```
    #[test]
    fn hand_derived_values() {
        let mut constant = [1.0f32; 8];
        dct_ii_8(&mut constant);
        assert_close(constant[0], 2.828_427, 1e-5, "DC of constant block");
        for (u, c) in constant.iter().enumerate().skip(1) {
            assert_close(*c, 0.0, 1e-6, &format!("AC[{u}] of constant block"));
        }

        let mut impulse = [0.0f32; 8];
        impulse[0] = 1.0;
        dct_ii_8(&mut impulse);
        assert_close(impulse[0], 0.353_553_39, 1e-6, "impulse X[0]");
        assert_close(impulse[1], 0.490_392_64, 1e-6, "impulse X[1]");
    }

    // -- 2-D round trips ----------------------------------------------------

    #[test]
    fn round_trip_8x8_random() {
        let mut rng = Lcg::new(0x00c0_ffee);
        for _ in 0..32 {
            let mut block = [0.0f32; 64];
            for slot in &mut block {
                *slot = rng.next(1.0);
            }
            let original = block;
            dct2d_8x8(&mut block);
            idct2d_8x8(&mut block);
            assert_slice_close(&block, &original, 1e-5, "8x8 round trip");
        }
    }

    #[test]
    fn round_trip_16x16_random() {
        let mut rng = Lcg::new(0x1337_beef);
        for _ in 0..32 {
            let mut block = [0.0f32; 256];
            for slot in &mut block {
                *slot = rng.next(1.0);
            }
            let original = block;
            dct2d_16x16(&mut block);
            idct2d_16x16(&mut block);
            assert_slice_close(&block, &original, 1e-5, "16x16 round trip");
        }
    }

    #[test]
    fn round_trip_rectangular_random() {
        let mut rng = Lcg::new(0x2bad_f00d);
        for _ in 0..32 {
            let mut tall = [0.0f32; 128];
            for slot in &mut tall {
                *slot = rng.next(1.0);
            }
            let original = tall;
            dct2d_16x8(&mut tall);
            idct2d_16x8(&mut tall);
            assert_slice_close(&tall, &original, 1e-5, "16x8 round trip");

            let mut wide = original;
            dct2d_8x16(&mut wide);
            idct2d_8x16(&mut wide);
            assert_slice_close(&wide, &original, 1e-5, "8x16 round trip");
        }
    }

    /// Extreme inputs: saturated white, and an alternating +/-255 checkerboard
    /// (the worst case for high-frequency coefficient growth). Tolerance is
    /// scaled with the input magnitude (255 * f32 epsilon * a few operations).
    #[test]
    fn round_trip_extreme_blocks() {
        let tol = 255.0 * 1e-5;

        let mut white = [255.0f32; 64];
        dct2d_8x8(&mut white);
        idct2d_8x8(&mut white);
        assert_slice_close(&white, &[255.0f32; 64], tol, "all-255 8x8 round trip");

        let mut checker = [0.0f32; 64];
        for (i, slot) in checker.iter_mut().enumerate() {
            *slot = if (i / 8 + i % 8) % 2 == 0 {
                255.0
            } else {
                -255.0
            };
        }
        let original = checker;
        dct2d_8x8(&mut checker);
        idct2d_8x8(&mut checker);
        assert_slice_close(&checker, &original, tol, "checker round trip");

        let mut big16 = [255.0f32; 256];
        dct2d_16x16(&mut big16);
        idct2d_16x16(&mut big16);
        assert_slice_close(&big16, &[255.0f32; 256], tol, "all-255 16x16 round trip");
    }

    #[test]
    fn dc_only_decodes_to_constant() {
        // DC coefficient c over an NxN block decodes to the constant c/N.
        let mut block = [0.0f32; 64];
        block[0] = 8.0;
        idct2d_8x8(&mut block);
        for (i, s) in block.iter().enumerate() {
            assert_close(*s, 1.0, 1e-6, &format!("8x8 dc-only sample {i}"));
        }

        let mut block16 = [0.0f32; 256];
        block16[0] = 16.0;
        idct2d_16x16(&mut block16);
        for (i, s) in block16.iter().enumerate() {
            assert_close(*s, 1.0, 1e-6, &format!("16x16 dc-only sample {i}"));
        }
    }

    #[test]
    fn parseval_energy_is_preserved() {
        let mut rng = Lcg::new(0x9e37_79b9);
        for _ in 0..16 {
            let mut block = [0.0f32; 64];
            for slot in &mut block {
                *slot = rng.next(4.0);
            }
            let before: f32 = block.iter().map(|s| s * s).sum();
            dct2d_8x8(&mut block);
            let after: f32 = block.iter().map(|s| s * s).sum();
            assert_close(after, before, before * 1e-4, "8x8 Parseval");

            let mut wide = [0.0f32; 128];
            for slot in &mut wide {
                *slot = rng.next(4.0);
            }
            let before: f32 = wide.iter().map(|s| s * s).sum();
            dct2d_8x16(&mut wide);
            let after: f32 = wide.iter().map(|s| s * s).sum();
            assert_close(after, before, before * 1e-4, "8x16 Parseval");
        }
    }

    // -- Coefficient layout: the important tests ---------------------------
    //
    // These pin down *where* a coefficient lives, not just that the transform
    // inverts. A transposed 2-D implementation passes every round-trip test
    // above and fails everything below.

    /// A pattern that varies only horizontally must produce coefficients only
    /// in row 0 (`u = 0`), i.e. at indices `0..8`. A vertical-only pattern must
    /// produce coefficients only in column 0, i.e. at indices `0, 8, 16, ...`.
    #[test]
    fn layout_horizontal_ramp_lands_in_row_zero() {
        let angle = |c: usize, v: usize| {
            (core::f64::consts::PI * ((2 * c + 1) as f64) * (v as f64) / 16.0).cos() as f32
        };

        // Horizontal cosine at v = 1, constant down the columns.
        let mut block = [0.0f32; 64];
        for r in 0..8 {
            for c in 0..8 {
                block[r * 8 + c] = angle(c, 1);
            }
        }
        dct2d_8x8(&mut block);
        // (u, v) = (0, 1) -> index 0 * 8 + 1 = 1.
        assert!(
            block[1].abs() > 1.0,
            "horizontal cosine should excite index 1, got {}",
            block[1]
        );
        // Its transpose, (1, 0) -> index 8, must be silent.
        assert_close(block[8], 0.0, 1e-4, "index 8 (u=1,v=0) must be zero");
        for (i, c) in block.iter().enumerate() {
            if i != 1 {
                assert_close(*c, 0.0, 1e-4, &format!("horizontal ramp coeff {i}"));
            }
        }

        // Vertical cosine at u = 1, constant across the rows.
        let mut block = [0.0f32; 64];
        for r in 0..8 {
            for c in 0..8 {
                block[r * 8 + c] = angle(r, 1);
            }
        }
        dct2d_8x8(&mut block);
        assert!(
            block[8].abs() > 1.0,
            "vertical cosine should excite index 8, got {}",
            block[8]
        );
        assert_close(block[1], 0.0, 1e-4, "index 1 (u=0,v=1) must be zero");
    }

    /// An impulse at sample `(r, c)` must produce the separable basis pattern
    /// `X[u * 8 + v] = C[u][r] * C[v][c]`, where `C` is the orthonormal DCT-II
    /// matrix. This checks both the layout and the exact basis values.
    #[test]
    fn layout_impulse_produces_analytic_basis() {
        let basis = |u: usize, n: usize| -> f32 {
            let scale = if u == 0 { 1.0f64 / 8.0 } else { 2.0f64 / 8.0 }.sqrt();
            let angle = core::f64::consts::PI * ((2 * n + 1) as f64) * (u as f64) / 16.0;
            (scale * angle.cos()) as f32
        };

        for &(r, c) in &[(0usize, 0usize), (0, 3), (3, 0), (2, 5), (7, 1)] {
            let mut block = [0.0f32; 64];
            block[r * 8 + c] = 1.0;
            dct2d_8x8(&mut block);
            for u in 0..8 {
                for v in 0..8 {
                    let expected = basis(u, r) * basis(v, c);
                    assert_close(
                        block[u * 8 + v],
                        expected,
                        1e-5,
                        &format!("impulse ({r},{c}) coeff ({u},{v})"),
                    );
                }
            }
        }
    }

    /// The same check for a rectangular block, where a transposed
    /// implementation would not even have the right buffer shape but a
    /// row/column swap inside the kernel would still "work".
    #[test]
    fn layout_rectangular_impulse() {
        let basis = |u: usize, n: usize, len: usize| -> f32 {
            let lf = len as f64;
            let scale = if u == 0 { 1.0 / lf } else { 2.0 / lf }.sqrt();
            let angle = core::f64::consts::PI * ((2 * n + 1) as f64) * (u as f64) / (2.0 * lf);
            (scale * angle.cos()) as f32
        };

        // 16 rows x 8 columns, impulse at row 5, column 2.
        let (r, c) = (5usize, 2usize);
        let mut tall = [0.0f32; 128];
        tall[r * 8 + c] = 1.0;
        dct2d_16x8(&mut tall);
        for u in 0..16 {
            for v in 0..8 {
                let expected = basis(u, r, 16) * basis(v, c, 8);
                assert_close(
                    tall[u * 8 + v],
                    expected,
                    1e-5,
                    &format!("16x8 impulse coeff ({u},{v})"),
                );
            }
        }

        // 8 rows x 16 columns, impulse at row 2, column 5.
        let (r, c) = (2usize, 5usize);
        let mut wide = [0.0f32; 128];
        wide[r * 16 + c] = 1.0;
        dct2d_8x16(&mut wide);
        for u in 0..8 {
            for v in 0..16 {
                let expected = basis(u, r, 8) * basis(v, c, 16);
                assert_close(
                    wide[u * 16 + v],
                    expected,
                    1e-5,
                    &format!("8x16 impulse coeff ({u},{v})"),
                );
            }
        }
    }

    /// An asymmetric block transformed directly must equal the same block
    /// transformed with the naive 1-D reference applied rows-then-columns.
    /// This is the belt-and-braces layout check against the definition.
    #[test]
    fn layout_matches_naive_separable_reference() {
        let mut rng = Lcg::new(0xa5a5_5a5a);
        let mut block = [0.0f32; 64];
        for slot in &mut block {
            *slot = rng.next(1.0);
        }

        let mut reference = block;
        let mut buf = [0.0f32; 8];
        for r in 0..8 {
            let row: [f32; 8] = core::array::from_fn(|c| reference[r * 8 + c]);
            dct2_naive(&row, &mut buf, 8);
            for (c, value) in buf.iter().enumerate() {
                reference[r * 8 + c] = *value;
            }
        }
        for c in 0..8 {
            let col: [f32; 8] = core::array::from_fn(|r| reference[r * 8 + c]);
            dct2_naive(&col, &mut buf, 8);
            for (r, value) in buf.iter().enumerate() {
                reference[r * 8 + c] = *value;
            }
        }

        dct2d_8x8(&mut block);
        assert_slice_close(&block, &reference, 1e-5, "2-D vs naive separable");
    }
}
