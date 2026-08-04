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
//! # Relation to the normative transform (18181-1 I.7.2) — RESOLVED
//!
//! The `[provisional]` note that used to sit here predicted that the standard's
//! transform would differ from the orthonormal one by a scalar prefactor
//! applied at the call boundary. It was read out of `latex/part1.tex` I.7.2 and
//! that prediction is correct. The normative 1-D pair for a length-`s` vector is
//!
//! ```text
//! forward (I.1):  out_k = (1/s) * (k == 0 ? 1 : sqrt(2))
//!                       * sum_{n=0}^{s-1} in_n * cos((pi*k/s) * (n + 1/2))
//! inverse (I.2):  in_k  = out_0 + sum_{n=1}^{s-1} sqrt(2) * out_n
//!                       * cos((pi*n/s) * (k + 1/2))
//! ```
//!
//! Both use the same cosine kernel as the orthonormal pair above; only the
//! per-coefficient constants differ, and they differ *uniformly*:
//!
//! * forward: `(1/s) / sqrt(1/s) == 1/sqrt(s)` at `k == 0` and
//!   `(sqrt(2)/s) / sqrt(2/s) == 1/sqrt(s)` at `k > 0`,
//! * inverse: `1 / sqrt(1/s) == sqrt(s)` at `k == 0` and
//!   `sqrt(2) / sqrt(2/s) == sqrt(s)` at `k > 0`.
//!
//! So the normative transform is `orthonormal * (1/sqrt(s))` forward and
//! `orthonormal * sqrt(s)` inverse, **per 1-D pass**. A 2-D `IDCT_2D` over an
//! `R x C` block runs two 1-D passes and therefore differs from an orthonormal
//! 2-D IDCT by `sqrt(R * C)`.
//!
//! The kernels below stay orthonormal. [`dct_1d`] and [`idct_1d`] are the
//! normative wrappers; nothing JPEG-XL-specific is baked into a kernel.
//!
//! The single load-bearing consequence, and the one the tests pin: under I.7.2
//! a DC-only coefficient block inverts to a **constant equal to the DC value**,
//! for every block size. Under the orthonormal convention it inverts to
//! `dc / sqrt(R*C)`.
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

// Every index in this module is derived from a locally computed loop bound over
// a buffer whose length was checked at the top of the function, so bounds checks
// here would be noise rather than a defence.
#![allow(clippy::indexing_slicing)]

use core::f32::consts::FRAC_1_SQRT_2;
use std::sync::OnceLock;

use crate::varblock::{CoeffMatrix, SampleBlock, TransformType};

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
// Generic power-of-two kernels (orthonormal), sizes 1..=256
// ---------------------------------------------------------------------------
//
// Table I.1 needs 1-D lengths 1, 2, 4, 8, 16, 32, 64, 128 and 256. Length 1 is
// reachable: I.8 runs `DCT_2D` over the `bwidth/8 x bheight/8` LF rectangle,
// which is 1 x 1 for DCT8x8 and 1 x N for the flat rectangular transforms.
//
// The hand-factored butterflies above cover 2, 4, 8 and 16 and are kept as the
// production path at those sizes. 32 and up use a cached orthonormal DCT-II
// matrix and an f64 accumulator: a reference decoder wants "obviously correct"
// far more than it wants a radix-2 recursion at DCT256x256, and every size is
// cross-checked against `dct2_naive` in the tests.

/// Largest 1-D transform length VarDCT can ask for (DCT256x256, 18181-1 I.1).
pub const MAX_TRANSFORM_SIZE: usize = 256;

/// Cached orthonormal DCT-II matrices for lengths 32, 64, 128 and 256.
///
/// Entry `[u * n + x]` is `s(u) * cos(pi * (2x + 1) * u / (2n))`. The DCT-III
/// matrix is the transpose, so one table serves both directions.
static DCT_MATRICES: [OnceLock<Vec<f32>>; 4] = [const { OnceLock::new() }; 4];

/// Is `n` a 1-D length this module can transform?
#[must_use]
pub const fn is_supported_length(n: usize) -> bool {
    n.is_power_of_two() && n <= MAX_TRANSFORM_SIZE
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn build_dct_matrix(n: usize) -> Vec<f32> {
    let nf = n as f64;
    let mut m = vec![0.0f32; n * n];
    for u in 0..n {
        let scale = if u == 0 { 1.0 / nf } else { 2.0 / nf }.sqrt();
        for x in 0..n {
            let angle = core::f64::consts::PI * ((2 * x + 1) as f64) * (u as f64) / (2.0 * nf);
            m[u * n + x] = (scale * angle.cos()) as f32;
        }
    }
    m
}

/// The cached DCT-II matrix for `n`, or `None` if `n` is not a tabled size.
fn dct_matrix(n: usize) -> Option<&'static [f32]> {
    let slot = match n {
        32 => 0usize,
        64 => 1,
        128 => 2,
        256 => 3,
        _ => return None,
    };
    let cell = DCT_MATRICES.get(slot)?;
    Some(cell.get_or_init(|| build_dct_matrix(n)).as_slice())
}

/// Runs a fixed-size kernel over a slice already known to have length `N`.
fn apply_fixed<const N: usize>(v: &mut [f32], kernel: fn(&mut [f32; N])) {
    let mut buf = [0.0f32; N];
    buf.copy_from_slice(v);
    kernel(&mut buf);
    v.copy_from_slice(&buf);
}

#[allow(clippy::cast_possible_truncation)]
fn matrix_pass(v: &mut [f32], m: &[f32], n: usize, forward: bool) {
    let mut out = [0.0f32; MAX_TRANSFORM_SIZE];
    for (k, slot) in out.iter_mut().take(n).enumerate() {
        let mut acc = 0.0f64;
        for j in 0..n {
            // Forward reads row `k`; inverse reads column `k` (the transpose).
            let coeff = if forward { m[k * n + j] } else { m[j * n + k] };
            acc += f64::from(coeff) * f64::from(v[j]);
        }
        *slot = acc as f32;
    }
    v.copy_from_slice(&out[..n]);
}

/// Orthonormal DCT-II over any supported length. Lengths 0 and 1 are the
/// identity; unsupported lengths leave `v` untouched.
fn dct_ii_any(v: &mut [f32]) {
    match v.len() {
        0 | 1 => {}
        2 => apply_fixed::<2>(v, dct_ii_2),
        4 => apply_fixed::<4>(v, dct_ii_4),
        8 => apply_fixed::<8>(v, dct_ii_8),
        16 => apply_fixed::<16>(v, dct_ii_16),
        n => {
            if let Some(m) = dct_matrix(n) {
                matrix_pass(v, m, n, true);
            } else {
                debug_assert!(false, "unsupported DCT length {n}");
            }
        }
    }
}

/// Orthonormal DCT-III over any supported length; exact inverse of
/// [`dct_ii_any`].
fn dct_iii_any(v: &mut [f32]) {
    match v.len() {
        0 | 1 => {}
        // The orthonormal 2-point DCT-II is its own inverse.
        2 => apply_fixed::<2>(v, dct_ii_2),
        4 => apply_fixed::<4>(v, dct_iii_4),
        8 => apply_fixed::<8>(v, dct_iii_8),
        16 => apply_fixed::<16>(v, dct_iii_16),
        n => {
            if let Some(m) = dct_matrix(n) {
                matrix_pass(v, m, n, false);
            } else {
                debug_assert!(false, "unsupported IDCT length {n}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 18181-1 I.7.2 — the normative 1-D pair
// ---------------------------------------------------------------------------

/// The normative forward 1-D DCT of 18181-1 I.7.2, equation (I.1), in place.
///
/// This is the orthonormal DCT-II scaled by `1 / sqrt(s)`; see the module
/// documentation for the derivation. Lengths must satisfy
/// [`is_supported_length`]; anything else leaves `v` unchanged.
#[allow(clippy::cast_precision_loss)]
pub fn dct_1d(v: &mut [f32]) {
    let s = v.len();
    if !is_supported_length(s) {
        debug_assert!(false, "unsupported DCT length {s}");
        return;
    }
    dct_ii_any(v);
    let factor = 1.0 / (s as f32).sqrt();
    for slot in v.iter_mut() {
        *slot *= factor;
    }
}

/// The normative inverse 1-D DCT of 18181-1 I.7.2, equation (I.2), in place.
///
/// This is the orthonormal DCT-III scaled by `sqrt(s)`. Exact inverse of
/// [`dct_1d`].
#[allow(clippy::cast_precision_loss)]
pub fn idct_1d(v: &mut [f32]) {
    let s = v.len();
    if !is_supported_length(s) {
        debug_assert!(false, "unsupported IDCT length {s}");
        return;
    }
    dct_iii_any(v);
    let factor = (s as f32).sqrt();
    for slot in v.iter_mut() {
        *slot *= factor;
    }
}

// ---------------------------------------------------------------------------
// 18181-1 I.7.3 — the normative 2-D pair
// ---------------------------------------------------------------------------

/// Transposes a row-major `rows x cols` matrix into a `cols x rows` one.
///
/// I.7.3 calls this three times inside `IDCT_2D`; it is exported because
/// coefficient orientation is the single highest-risk convention in Annex I and
/// a shared, tested helper is cheaper than three open-coded index swaps.
#[must_use]
pub fn transpose(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; rows * cols];
    transpose_into(src, &mut dst, rows, cols);
    dst
}

/// [`transpose`] into a caller-provided buffer; the allocation-free form.
///
/// `dst[..rows * cols]` receives the `cols x rows` transpose. Buffers that are
/// too short leave `dst` untouched (in a debug build, they assert).
pub fn transpose_into(src: &[f32], dst: &mut [f32], rows: usize, cols: usize) {
    let n = rows * cols;
    if src.len() < n || dst.len() < n {
        debug_assert!(false, "transpose buffers are too short");
        return;
    }
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}

/// `ColumnDCT` of I.7.3: the 1-D forward DCT down each column of a row-major
/// `rows x cols` matrix.
fn column_dct(m: &mut [f32], rows: usize, cols: usize) {
    let mut buf = [0.0f32; MAX_TRANSFORM_SIZE];
    for c in 0..cols {
        let col = &mut buf[..rows];
        for (r, slot) in col.iter_mut().enumerate() {
            *slot = m[r * cols + c];
        }
        dct_1d(col);
        for r in 0..rows {
            m[r * cols + c] = buf[r];
        }
    }
}

/// `ColumnIDCT` of I.7.3.
fn column_idct(m: &mut [f32], rows: usize, cols: usize) {
    let mut buf = [0.0f32; MAX_TRANSFORM_SIZE];
    for c in 0..cols {
        let col = &mut buf[..rows];
        for (r, slot) in col.iter_mut().enumerate() {
            *slot = m[r * cols + c];
        }
        idct_1d(col);
        for r in 0..rows {
            m[r * cols + c] = buf[r];
        }
    }
}

/// The shape of the coefficient matrix produced by `DCT_2D` on `rows x cols`
/// samples, as `(coefficient rows, coefficient columns)`.
///
/// I.7.3's final conditional transpose makes this **always landscape**:
/// `(min(rows, cols), max(rows, cols))`. That is the same rule as I.3.2's
/// `bheight = max(8, min(N, M))`, `bwidth = max(8, max(N, M))` for the
/// transforms whose varblock is at least 8x8, and it is why a DCT16x8 varblock
/// has 16x8 samples but 8x16 coefficients.
#[must_use]
pub const fn coeff_dims(rows: usize, cols: usize) -> (usize, usize) {
    if cols > rows {
        (rows, cols)
    } else {
        (cols, rows)
    }
}

/// Is `(rows, cols)` a shape `DCT_2D` / `IDCT_2D` can handle, and is `buf` long
/// enough to hold a matrix of that many cells?
fn shape_is_ok(buf_len: usize, rows: usize, cols: usize) -> bool {
    is_supported_length(rows) && is_supported_length(cols) && buf_len >= rows * cols
}

/// `DCT_2D` of 18181-1 I.7.3, in place on a caller-provided buffer.
///
/// On entry `work[..rows * cols]` holds the row-major `rows x cols` samples; on
/// exit it holds the landscape coefficient matrix described by [`coeff_dims`]
/// (the same cell count, a possibly different shape). `scratch` is clobbered
/// and must be at least as long.
///
/// This is the allocation-free entry point and the single implementation of
/// I.7.3's forward direction; [`dct_2d_into`] and [`dct_2d_raw`] wrap it.
pub fn dct_2d_in_place(work: &mut [f32], scratch: &mut [f32], rows: usize, cols: usize) {
    let n = rows * cols;
    if !shape_is_ok(work.len(), rows, cols) || scratch.len() < n {
        debug_assert!(false, "bad DCT_2D shape {rows}x{cols}");
        return;
    }
    column_dct(work, rows, cols);
    transpose_into(work, scratch, rows, cols);
    // `scratch` is `cols x rows` from here on.
    column_dct(scratch, cols, rows);
    if cols > rows {
        transpose_into(scratch, work, cols, rows);
    } else {
        work[..n].copy_from_slice(&scratch[..n]);
    }
}

/// `IDCT_2D` of 18181-1 I.7.3, in place on a caller-provided buffer.
///
/// On entry `work[..rows * cols]` holds the landscape coefficient matrix of
/// [`coeff_dims`]; on exit it holds the row-major `rows x cols` samples.
/// `scratch` is clobbered.
pub fn idct_2d_in_place(work: &mut [f32], scratch: &mut [f32], rows: usize, cols: usize) {
    let n = rows * cols;
    if !shape_is_ok(work.len(), rows, cols) || scratch.len() < n {
        debug_assert!(false, "bad IDCT_2D shape {rows}x{cols}");
        return;
    }
    // `scratch` is `cols x rows` here, whichever branch produced it.
    if cols > rows {
        transpose_into(work, scratch, rows, cols);
    } else {
        scratch[..n].copy_from_slice(&work[..n]);
    }
    column_idct(scratch, cols, rows);
    transpose_into(scratch, work, cols, rows);
    column_idct(work, rows, cols);
}

/// `DCT_2D` of 18181-1 I.7.3, into caller-provided buffers.
///
/// `out[..rows * cols]` receives the landscape coefficient matrix described by
/// [`coeff_dims`]; `scratch` is clobbered. Both buffers must hold at least
/// `rows * cols` values.
pub fn dct_2d_into(
    samples: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
    scratch: &mut [f32],
) {
    let n = rows * cols;
    if samples.len() != n || !shape_is_ok(out.len(), rows, cols) || scratch.len() < n {
        debug_assert!(false, "bad DCT_2D shape {rows}x{cols}");
        out.iter_mut().take(n).for_each(|s| *s = 0.0);
        return;
    }
    out[..n].copy_from_slice(samples);
    dct_2d_in_place(out, scratch, rows, cols);
}

/// `IDCT_2D` of 18181-1 I.7.3, into caller-provided buffers.
///
/// `out[..rows * cols]` receives the `rows x cols` samples; `scratch` is
/// clobbered. `coeffs` is the landscape matrix of [`coeff_dims`].
pub fn idct_2d_into(
    coeffs: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
    scratch: &mut [f32],
) {
    let n = rows * cols;
    if coeffs.len() != n || !shape_is_ok(out.len(), rows, cols) || scratch.len() < n {
        debug_assert!(false, "bad IDCT_2D shape {rows}x{cols}");
        out.iter_mut().take(n).for_each(|s| *s = 0.0);
        return;
    }
    out[..n].copy_from_slice(coeffs);
    idct_2d_in_place(out, scratch, rows, cols);
}

/// `DCT_2D` of 18181-1 I.7.3 on a row-major `rows x cols` sample matrix.
///
/// Returns the landscape coefficient matrix described by [`coeff_dims`].
/// Owning wrapper around [`dct_2d_into`].
#[must_use]
pub fn dct_2d_raw(samples: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let (cr, cc) = coeff_dims(rows, cols);
    if !is_supported_length(rows) || !is_supported_length(cols) || samples.len() != rows * cols {
        debug_assert!(false, "bad DCT_2D shape {rows}x{cols}");
        return vec![0.0f32; cr * cc];
    }
    let mut out = vec![0.0f32; rows * cols];
    let mut scratch = vec![0.0f32; rows * cols];
    dct_2d_into(samples, rows, cols, &mut out, &mut scratch);
    out
}

/// `IDCT_2D` of 18181-1 I.7.3: landscape coefficients to `rows x cols` samples.
///
/// `coeffs` must be laid out as [`coeff_dims`] says, i.e. row-major
/// `min(rows, cols) x max(rows, cols)`. Owning wrapper around
/// [`idct_2d_into`].
#[must_use]
pub fn idct_2d_raw(coeffs: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    if !is_supported_length(rows) || !is_supported_length(cols) || coeffs.len() != rows * cols {
        debug_assert!(false, "bad IDCT_2D shape {rows}x{cols}");
        return vec![0.0f32; rows * cols];
    }
    let mut out = vec![0.0f32; rows * cols];
    let mut scratch = vec![0.0f32; rows * cols];
    idct_2d_into(coeffs, rows, cols, &mut out, &mut scratch);
    out
}

// ---------------------------------------------------------------------------
// Typed entry points
// ---------------------------------------------------------------------------

/// `DCT_2D` over the typed sample block, yielding a landscape [`CoeffMatrix`].
#[must_use]
pub fn dct_2d(samples: &SampleBlock) -> CoeffMatrix {
    let (rows, cols) = (samples.rows(), samples.cols());
    let (cr, cc) = coeff_dims(rows, cols);
    CoeffMatrix::from_landscape(cr, cc, dct_2d_raw(samples.as_slice(), rows, cols))
}

/// `IDCT_2D` over the typed coefficient matrix, yielding `rows x cols` samples.
///
/// The `rows`/`cols` arguments are the **sample** dimensions `R`/`C` of I.7.3;
/// `coeffs` is the landscape matrix, so for `R > C` it is the transpose of the
/// output shape. That asymmetry is the whole point of having two types.
#[must_use]
pub fn idct_2d(coeffs: &CoeffMatrix, rows: usize, cols: usize) -> SampleBlock {
    SampleBlock::from_rows_cols(rows, cols, idct_2d_raw(coeffs.as_slice(), rows, cols))
}

/// Runtime dispatch of `IDCT_2D` on a [`TransformType`] (18181-1 I.9.2).
///
/// Returns `None` for the transform types that are *not* a plain `DCTRxC`
/// (Hornuss, DCT2x2, DCT4x4, DCT4x8, DCT8x4, AFV0-3); those have their own
/// reconstructions in I.9.3-I.9.8, dispatched by
/// [`TransformType::samples_from_coefficients`].
#[must_use]
pub fn idct_2d_for_transform(
    transform: TransformType,
    coeffs: &CoeffMatrix,
) -> Option<SampleBlock> {
    let (rows, cols) = transform.dct_shape()?;
    Some(idct_2d(coeffs, rows, cols))
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

    // -- 18181-1 I.7: the normative transform ------------------------------
    //
    // Everything above pins the orthonormal kernels. Everything below pins the
    // *standard's* transform, which differs by 1/sqrt(s) per forward pass and
    // sqrt(s) per inverse pass (see the module documentation). These are the
    // tests that would fail if the orthonormal convention leaked out of the
    // kernels and into the VarDCT pipeline.

    /// Every 1-D length Table I.1 can ask for, largest first so a missing size
    /// fails loudly rather than silently no-op'ing.
    const LENGTHS: [usize; 9] = [1, 2, 4, 8, 16, 32, 64, 128, 256];

    /// Every distinct `(R, C)` sample shape in Table I.1.
    const TABLE_I1_SHAPES: [(usize, usize); 17] = [
        (8, 8),
        (16, 16),
        (32, 32),
        (16, 8),
        (8, 16),
        (32, 8),
        (8, 32),
        (32, 16),
        (16, 32),
        (64, 64),
        (64, 32),
        (32, 64),
        (128, 128),
        (128, 64),
        (64, 128),
        (256, 256),
        (256, 128),
    ];

    /// The generic kernels must agree with the `f64` evaluation of the
    /// orthonormal definition at every length, including the ones with no
    /// hand-factored butterfly (32 and up). Proves the cached matrices are
    /// built correctly and that the DCT-III path really is the transpose.
    #[test]
    fn generic_kernels_match_the_definition() {
        let mut rng = Lcg::new(0x0c0f_fee1);
        for &n in &LENGTHS {
            let mut fast = vec![0.0f32; n];
            for slot in &mut fast {
                *slot = rng.next(1.0);
            }
            let input = fast.clone();

            let mut reference = vec![0.0f32; n];
            dct2_naive(&input, &mut reference, n);
            dct_ii_any(&mut fast);
            assert_slice_close(&fast, &reference, 1e-4, &format!("dct_ii_any({n})"));

            let mut fast3 = input.clone();
            let mut reference3 = vec![0.0f32; n];
            dct3_naive(&input, &mut reference3, n);
            dct_iii_any(&mut fast3);
            assert_slice_close(&fast3, &reference3, 1e-4, &format!("dct_iii_any({n})"));
        }
    }

    /// **The single most load-bearing test in the DCT engine.**
    ///
    /// I.7.2's inverse is `in_k = out_0 + sum_{n>0} sqrt(2)*out_n*cos(...)`, so
    /// a DC-only vector inverts to the constant `out_0` *at every length*. The
    /// orthonormal DCT-III would give `out_0 / sqrt(s)` instead. This is what
    /// discriminates the two conventions; no round-trip test can.
    #[test]
    fn i72_dc_only_inverts_to_the_dc_value() {
        for &n in &LENGTHS {
            let mut v = vec![0.0f32; n];
            v[0] = 3.5;
            idct_1d(&mut v);
            for (k, s) in v.iter().enumerate() {
                assert_close(*s, 3.5, 1e-4, &format!("idct_1d({n}) sample {k}"));
            }
        }
    }

    /// The same statement in two dimensions, for every Table I.1 shape: a
    /// DC-only coefficient matrix inverts to a constant block equal to the DC.
    /// Under the orthonormal convention this would be `dc / sqrt(R*C)`, which
    /// is wrong by up to 256x at DCT256x256.
    #[test]
    fn i73_dc_only_inverts_to_a_constant_block() {
        for &(r, c) in &TABLE_I1_SHAPES {
            let (cr, cc) = coeff_dims(r, c);
            let mut coeffs = vec![0.0f32; cr * cc];
            coeffs[0] = -2.25;
            let samples = idct_2d_raw(&coeffs, r, c);
            assert_eq!(samples.len(), r * c, "{r}x{c} sample count");
            for (i, s) in samples.iter().enumerate() {
                assert_close(*s, -2.25, 1e-3, &format!("{r}x{c} dc-only sample {i}"));
            }
        }
    }

    /// I.7.2 equation (I.2) states the inverse basis explicitly: a unit
    /// coefficient at index `n > 0` must produce `sqrt(2) * cos(pi*n/s *
    /// (k + 1/2))`, and at `n == 0` the constant 1. Checked against the closed
    /// form rather than against the forward transform, so a matched pair of
    /// wrong scale factors cannot hide.
    #[test]
    fn i72_impulse_response_matches_the_closed_form() {
        for &s in &LENGTHS {
            for n in 0..s {
                let mut v = vec![0.0f32; s];
                v[n] = 1.0;
                idct_1d(&mut v);
                for (k, got) in v.iter().enumerate() {
                    let expected = if n == 0 {
                        1.0f64
                    } else {
                        let angle =
                            core::f64::consts::PI * (n as f64) / (s as f64) * ((k as f64) + 0.5);
                        core::f64::consts::SQRT_2 * angle.cos()
                    };
                    assert_close(
                        *got,
                        expected as f32,
                        1e-4,
                        &format!("idct_1d({s}) impulse {n} sample {k}"),
                    );
                }
            }
        }
    }

    /// I.7.2 equation (I.1) evaluated directly. Proves the forward wrapper's
    /// `1/sqrt(s)` and, together with the test above, that the two wrappers are
    /// not merely each other's inverse.
    #[test]
    fn i72_forward_matches_the_closed_form() {
        let mut rng = Lcg::new(0x51de_8a1a);
        for &s in &LENGTHS {
            let input: Vec<f32> = (0..s).map(|_| rng.next(1.0)).collect();
            let mut got = input.clone();
            dct_1d(&mut got);
            for (k, out) in got.iter().enumerate() {
                let scale = if k == 0 {
                    1.0
                } else {
                    core::f64::consts::SQRT_2
                } / (s as f64);
                let mut acc = 0.0f64;
                for (n, x) in input.iter().enumerate() {
                    let angle =
                        core::f64::consts::PI * (k as f64) / (s as f64) * ((n as f64) + 0.5);
                    acc += f64::from(*x) * angle.cos();
                }
                assert_close(
                    *out,
                    (scale * acc) as f32,
                    1e-4,
                    &format!("dct_1d({s}) coefficient {k}"),
                );
            }
        }
    }

    /// `IDCT_2D(DCT_2D(x)) == x` for every `(R, C)` in Table I.1.
    ///
    /// Proves that the leading conditional transpose of `IDCT_2D`, its two
    /// `ColumnIDCT` passes and the transpose between them are mutually
    /// consistent with the forward pipeline — three orientation decisions in
    /// five lines of I.7.3, and a rectangular shape catches a swap that a
    /// square one cannot.
    #[test]
    fn i73_round_trips_every_table_i1_shape() {
        let mut rng = Lcg::new(0x2bad_1dea);
        for &(r, c) in &TABLE_I1_SHAPES {
            let samples: Vec<f32> = (0..r * c).map(|_| rng.next(1.0)).collect();
            let coeffs = dct_2d_raw(&samples, r, c);
            let (cr, cc) = coeff_dims(r, c);
            assert_eq!(coeffs.len(), cr * cc, "{r}x{c} coefficient count");
            let back = idct_2d_raw(&coeffs, r, c);
            assert_slice_close(&back, &samples, 2e-3, &format!("{r}x{c} round trip"));
        }
    }

    /// Coefficient matrices are landscape for every Table I.1 shape: the
    /// coefficient array of a portrait varblock is the transpose of its sample
    /// array's shape. Storage-order statement, not a value statement.
    #[test]
    fn coefficient_shape_is_always_landscape() {
        for &(r, c) in &TABLE_I1_SHAPES {
            let (cr, cc) = coeff_dims(r, c);
            assert!(cr <= cc, "{r}x{c} coefficients {cr}x{cc} are not landscape");
            assert_eq!((cr, cc), (r.min(c), r.max(c)), "{r}x{c} coefficient dims");
        }
    }

    /// Storage-order snapshot for the two rectangular 8x16 families.
    ///
    /// In the landscape coefficient matrix the *column* index `x` selects the
    /// frequency of whichever axis has length `max(R, C)`. For DCT16x8 that is
    /// the vertical axis, so coefficient `(x = 1, y = 0)` must vary down the
    /// block and be constant across it; for DCT8x16 it is the horizontal axis
    /// and the pattern is transposed. A row/column swap anywhere in `IDCT_2D`
    /// flips exactly this and nothing that a round trip would notice.
    #[test]
    fn layout_landscape_column_index_selects_the_long_axis() {
        // DCT16x8: 16 rows x 8 columns of samples, 8 x 16 coefficients.
        let mut coeffs = vec![0.0f32; 8 * 16];
        coeffs[1] = 1.0; // (x, y) = (1, 0)
        let tall = idct_2d_raw(&coeffs, 16, 8);
        for row in 0..16 {
            for col in 1..8 {
                assert_close(
                    tall[row * 8 + col],
                    tall[row * 8],
                    1e-5,
                    "DCT16x8 must be constant along a row",
                );
            }
        }
        assert!(
            (tall[0] - tall[15 * 8]).abs() > 1.0,
            "DCT16x8 must vary down the block"
        );

        // DCT8x16: 8 rows x 16 columns of samples, also 8 x 16 coefficients.
        let wide = idct_2d_raw(&coeffs, 8, 16);
        for row in 1..8 {
            for col in 0..16 {
                assert_close(
                    wide[row * 16 + col],
                    wide[col],
                    1e-5,
                    "DCT8x16 must be constant down a column",
                );
            }
        }
        assert!(
            (wide[0] - wide[15]).abs() > 1.0,
            "DCT8x16 must vary across the block"
        );
    }

    /// `transpose` is an involution and moves `(r, c)` to `(c, r)`.
    #[test]
    fn transpose_round_trips() {
        let src: Vec<f32> = (0..4 * 6).map(|i| i as f32).collect();
        let t = transpose(&src, 4, 6);
        assert_eq!(t.len(), 24);
        for r in 0..4 {
            for c in 0..6 {
                assert_close(t[c * 4 + r], src[r * 6 + c], 0.0, "transpose entry");
            }
        }
        assert_slice_close(&transpose(&t, 6, 4), &src, 0.0, "double transpose");
    }
}
