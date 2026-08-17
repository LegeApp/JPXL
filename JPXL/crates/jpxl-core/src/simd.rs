//! Portable lane vectors for the codec's kernels, in one trait with three
//! kinds of implementation.
//!
//! A kernel written once against [`F32Vec`] can run as plain scalar code
//! (`f32`), as `wide` vectors (`f32x4` / `f32x8`, the baseline SIMD this
//! workspace compiles for), or on 256-bit AVX2 registers
//! ([`avx2::F32x8`]) selected at run time through [`crate::cpu::has_avx2`].
//! Every implementation performs the same IEEE-754 operation per lane — no
//! fused multiply-add, no reassociation, no approximate reciprocals — so a
//! kernel produces bit-identical results whichever implementation it is
//! instantiated with. That is what lets the encoder and decoder emit the same
//! bytes on every host and build, and it is pinned by the kernels' own tests
//! (`dct::tests::lane_widths_are_bit_identical_to_scalar` and the quantizer's
//! lane tests).
//!
//! Comparison results are masks in the same type: a lane is all-one bits when
//! the comparison holds and all-zero bits otherwise, exactly `wide`'s idiom,
//! so [`F32Vec::blend`] and the bit operations compose without a second mask
//! type.

use core::ops::{Add, Div, Mul, Sub};

/// One value per lane, with the operations the codec's kernels need.
///
/// See the module documentation for the bit-identity contract every
/// implementation upholds.
pub trait F32Vec:
    Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self>
{
    /// Values per vector.
    const LANES: usize;

    /// Every lane set to `value`.
    fn splat(value: f32) -> Self;

    /// The first `Self::LANES` values of `src`, which must be at least that
    /// long (an implementation asserts in debug builds and returns zeros
    /// otherwise).
    fn load(src: &[f32]) -> Self;

    /// Writes the lanes to the first `Self::LANES` slots of `dst`, which must
    /// be at least that long (a short `dst` asserts in debug builds and is
    /// left untouched otherwise).
    fn store(self, dst: &mut [f32]);

    /// Writes each lane converted to `i32` by truncation toward zero, for
    /// lanes holding integers of magnitude below 2^31; other lanes produce an
    /// unspecified value. Same length rule as [`Self::store`].
    fn store_trunc_i32(self, dst: &mut [i32]);

    /// Per-lane absolute value (clears the sign bit).
    fn abs(self) -> Self;

    /// Per-lane truncation toward zero, exact for magnitudes below 2^31.
    /// Lanes outside that range, infinities and NaNs produce a value of
    /// magnitude at least 2^31 (or NaN), never a small integer.
    fn trunc(self) -> Self;

    /// Mask of lanes where `self < rhs` (false for unordered).
    fn cmp_lt(self, rhs: Self) -> Self;
    /// Mask of lanes where `self <= rhs` (false for unordered).
    fn cmp_le(self, rhs: Self) -> Self;
    /// Mask of lanes where `self == rhs` (false for unordered).
    fn cmp_eq(self, rhs: Self) -> Self;
    /// Mask of lanes where `self > rhs` (false for unordered).
    fn cmp_gt(self, rhs: Self) -> Self;

    /// Bitwise and.
    fn and(self, rhs: Self) -> Self;
    /// Bitwise or.
    fn or(self, rhs: Self) -> Self;
    /// Bitwise and-not: `!self & rhs`.
    fn andnot(self, rhs: Self) -> Self;

    /// Per lane, `t` where the mask `self` is set and `f` where it is not.
    fn blend(self, t: Self, f: Self) -> Self;

    /// Is every lane of the mask set?
    fn all(self) -> bool;
    /// Is any lane of the mask set?
    fn any(self) -> bool;
}

/// A mask lane that holds: all-one bits, viewed as `f32`.
const MASK_SET: f32 = f32::from_bits(u32::MAX);

impl F32Vec for f32 {
    const LANES: usize = 1;

    #[inline(always)]
    fn splat(value: f32) -> Self {
        value
    }

    #[inline(always)]
    fn load(src: &[f32]) -> Self {
        src.first().copied().unwrap_or_else(|| {
            debug_assert!(false, "scalar lane load from an empty slice");
            0.0
        })
    }

    #[inline(always)]
    fn store(self, dst: &mut [f32]) {
        match dst.first_mut() {
            Some(slot) => *slot = self,
            None => debug_assert!(false, "scalar lane store to an empty slice"),
        }
    }

    #[inline(always)]
    fn store_trunc_i32(self, dst: &mut [i32]) {
        match dst.first_mut() {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "documented: only integer lanes below 2^31 are meaningful"
            )]
            Some(slot) => *slot = self as i32,
            None => debug_assert!(false, "scalar lane store to an empty slice"),
        }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        f32::abs(self)
    }

    #[inline(always)]
    fn trunc(self) -> Self {
        // Mirror the vector implementations exactly: convert through i32 with
        // truncation (saturating; NaN -> 0 in Rust, INT_MIN on x86 vectors) and
        // back. Both agree on every lane the contract calls exact, and both
        // land outside +-2^31 or on NaN/0 for the rest -- and 0 only for a NaN
        // input, which every caller has already rejected through `cmp_lt`.
        #[allow(
            clippy::cast_possible_truncation,
            reason = "saturating cast is the documented behaviour outside 2^31"
        )]
        let i = self as i32;
        #[allow(
            clippy::cast_precision_loss,
            reason = "i32 -> f32 is the documented conversion"
        )]
        {
            i as f32
        }
    }

    #[inline(always)]
    fn cmp_lt(self, rhs: Self) -> Self {
        if self < rhs { MASK_SET } else { 0.0 }
    }

    #[inline(always)]
    fn cmp_le(self, rhs: Self) -> Self {
        if self <= rhs { MASK_SET } else { 0.0 }
    }

    #[inline(always)]
    fn cmp_eq(self, rhs: Self) -> Self {
        if self == rhs { MASK_SET } else { 0.0 }
    }

    #[inline(always)]
    fn cmp_gt(self, rhs: Self) -> Self {
        if self > rhs { MASK_SET } else { 0.0 }
    }

    #[inline(always)]
    fn and(self, rhs: Self) -> Self {
        f32::from_bits(self.to_bits() & rhs.to_bits())
    }

    #[inline(always)]
    fn or(self, rhs: Self) -> Self {
        f32::from_bits(self.to_bits() | rhs.to_bits())
    }

    #[inline(always)]
    fn andnot(self, rhs: Self) -> Self {
        f32::from_bits(!self.to_bits() & rhs.to_bits())
    }

    #[inline(always)]
    fn blend(self, t: Self, f: Self) -> Self {
        let m = self.to_bits();
        f32::from_bits((t.to_bits() & m) | (f.to_bits() & !m))
    }

    #[inline(always)]
    fn all(self) -> bool {
        self.to_bits() == u32::MAX
    }

    #[inline(always)]
    fn any(self) -> bool {
        self.to_bits() != 0
    }
}

/// Implements [`F32Vec`] for a `wide` float vector type.
#[cfg(feature = "simd")]
macro_rules! wide_f32vec {
    ($ty:ty, $ity:ty, $lanes:literal) => {
        impl F32Vec for $ty {
            const LANES: usize = $lanes;

            #[inline(always)]
            fn splat(value: f32) -> Self {
                <$ty>::splat(value)
            }

            #[inline(always)]
            fn load(src: &[f32]) -> Self {
                match src.first_chunk::<$lanes>() {
                    Some(chunk) => <$ty>::from(*chunk),
                    None => {
                        debug_assert!(false, "lane load from a short slice");
                        <$ty>::splat(0.0)
                    }
                }
            }

            #[inline(always)]
            fn store(self, dst: &mut [f32]) {
                match dst.first_chunk_mut::<$lanes>() {
                    Some(chunk) => *chunk = self.to_array(),
                    None => debug_assert!(false, "lane store to a short slice"),
                }
            }

            #[inline(always)]
            fn store_trunc_i32(self, dst: &mut [i32]) {
                match dst.first_chunk_mut::<$lanes>() {
                    Some(chunk) => *chunk = self.fast_trunc_int().to_array(),
                    None => debug_assert!(false, "lane store to a short slice"),
                }
            }

            #[inline(always)]
            fn abs(self) -> Self {
                <$ty>::abs(self)
            }

            #[inline(always)]
            fn trunc(self) -> Self {
                <$ity>::round_float(self.fast_trunc_int())
            }

            #[inline(always)]
            fn cmp_lt(self, rhs: Self) -> Self {
                wide::CmpLt::cmp_lt(self, rhs)
            }

            #[inline(always)]
            fn cmp_le(self, rhs: Self) -> Self {
                wide::CmpLe::cmp_le(self, rhs)
            }

            #[inline(always)]
            fn cmp_eq(self, rhs: Self) -> Self {
                wide::CmpEq::cmp_eq(self, rhs)
            }

            #[inline(always)]
            fn cmp_gt(self, rhs: Self) -> Self {
                wide::CmpGt::cmp_gt(self, rhs)
            }

            #[inline(always)]
            fn and(self, rhs: Self) -> Self {
                self & rhs
            }

            #[inline(always)]
            fn or(self, rhs: Self) -> Self {
                self | rhs
            }

            #[inline(always)]
            fn andnot(self, rhs: Self) -> Self {
                !self & rhs
            }

            #[inline(always)]
            fn blend(self, t: Self, f: Self) -> Self {
                <$ty>::blend(self, t, f)
            }

            #[inline(always)]
            fn all(self) -> bool {
                <$ty>::all(self)
            }

            #[inline(always)]
            fn any(self) -> bool {
                <$ty>::any(self)
            }
        }
    };
}

#[cfg(feature = "simd")]
wide_f32vec!(wide::f32x4, wide::i32x4, 4);
#[cfg(feature = "simd")]
wide_f32vec!(wide::f32x8, wide::i32x8, 8);

/// [`F32Vec`] on 256-bit AVX2 registers.
///
/// This workspace is compiled for baseline x86-64, so `wide::f32x8` is two
/// SSE registers and LLVM does not re-widen it when a caller enables AVX2.
/// Kernels that want real 256-bit lanes are instantiated with [`F32x8`]
/// inside a `#[target_feature(enable = "avx2")]` function that the caller
/// reaches only after [`crate::cpu::has_avx2`] returned true.
///
/// # Contract
///
/// Every method of [`F32x8`] executes AVX/AVX2 instructions. Using the type on
/// a host without AVX2 is undefined behaviour (in practice an illegal
/// instruction fault). The type therefore has exactly one legitimate use:
/// as the lane parameter of a kernel whose entry point is a
/// `#[target_feature(enable = "avx2")]` function guarded by
/// [`crate::cpu::has_avx2`]. Every `unsafe` block below relies on that
/// contract and nothing else; the two pointer-based intrinsics additionally
/// receive references to exactly-eight-element arrays. This module is the
/// only place in `jpxl-core` that uses `unsafe`, which is why the allowance
/// is scoped to it.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[allow(unsafe_code)]
pub mod avx2 {
    use core::arch::x86_64::{
        __m256, _CMP_EQ_OQ, _CMP_GT_OQ, _CMP_LE_OQ, _CMP_LT_OQ, _mm256_add_ps, _mm256_and_ps,
        _mm256_andnot_ps, _mm256_blendv_ps, _mm256_cmp_ps, _mm256_cvtepi32_ps, _mm256_cvttps_epi32,
        _mm256_div_ps, _mm256_loadu_ps, _mm256_movemask_ps, _mm256_mul_ps, _mm256_or_ps,
        _mm256_set1_ps, _mm256_storeu_ps, _mm256_storeu_si256, _mm256_sub_ps,
    };

    use super::F32Vec;

    /// Eight `f32` lanes in one AVX register. See the module contract.
    #[derive(Clone, Copy)]
    pub struct F32x8(__m256);

    impl F32x8 {
        /// Per-lane cube root, bit-identical to Rust's `f32::cbrt` (the
        /// FreeBSD/musl algorithm in `compiler_builtins`: a 5-bit integer
        /// estimate, then two Newton steps in `f64`, then one rounding to
        /// `f32`), for lanes that are finite, normal and non-zero. Returns
        /// `None` when any lane is zero, subnormal, infinite or NaN — the
        /// scalar function's special cases — so the caller can fall back to
        /// `f32::cbrt` for that chunk. `tests::avx2_cbrt_matches_std` pins
        /// the identity.
        #[inline(always)]
        #[must_use]
        pub fn cbrt(self) -> Option<Self> {
            use core::arch::x86_64::{
                __m128i, __m256d, __m256i, _mm256_add_epi32, _mm256_add_pd, _mm256_and_si256,
                _mm256_castps_si256, _mm256_castps128_ps256, _mm256_castps256_ps128,
                _mm256_castsi128_si256, _mm256_castsi256_ps, _mm256_castsi256_si128,
                _mm256_cmpgt_epi32, _mm256_cvtepi32_pd, _mm256_cvtpd_ps, _mm256_cvtps_pd,
                _mm256_cvttpd_epi32, _mm256_div_pd, _mm256_extractf128_ps,
                _mm256_extracti128_si256, _mm256_floor_pd, _mm256_insertf128_ps,
                _mm256_inserti128_si256, _mm256_movemask_epi8, _mm256_mul_pd, _mm256_or_si256,
                _mm256_set1_epi32, _mm256_set1_pd,
            };
            /// `(127 - 127/3 - 0.03306235651) * 2^23`: the scalar algorithm's B1.
            const B1: i32 = 709_958_130;

            // SAFETY: module contract (AVX2 host, AVX2-enabled caller); the
            // whole body is register arithmetic on `self`.
            unsafe {
                let ui: __m256i = _mm256_castps_si256(self.0);
                let hx = _mm256_and_si256(ui, _mm256_set1_epi32(0x7fff_ffff));
                // Normal, finite, non-zero: 0x00800000 <= hx < 0x7f800000.
                let ge_min = _mm256_cmpgt_epi32(hx, _mm256_set1_epi32(0x007f_ffff));
                let lt_inf = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x7f80_0000), hx);
                if _mm256_movemask_epi8(_mm256_and_si256(ge_min, lt_inf)) != -1 {
                    return None;
                }
                // hx / 3 exactly: the quotient of two exact integers below 2^31
                // is correctly rounded in f64, and its floor is the integer
                // quotient (the true value is k, k + 1/3 or k + 2/3).
                let three = _mm256_set1_pd(3.0);
                let lo: __m128i = _mm256_castsi256_si128(hx);
                let hi: __m128i = _mm256_extracti128_si256::<1>(hx);
                let q_lo = _mm256_cvttpd_epi32(_mm256_floor_pd(_mm256_div_pd(
                    _mm256_cvtepi32_pd(lo),
                    three,
                )));
                let q_hi = _mm256_cvttpd_epi32(_mm256_floor_pd(_mm256_div_pd(
                    _mm256_cvtepi32_pd(hi),
                    three,
                )));
                let q = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(q_lo), q_hi);
                let est_bits = _mm256_or_si256(
                    _mm256_and_si256(ui, _mm256_set1_epi32(u32::MAX.cast_signed() ^ 0x7fff_ffff)),
                    _mm256_add_epi32(q, _mm256_set1_epi32(B1)),
                );
                let est = _mm256_castsi256_ps(est_bits);

                // Two Newton steps in f64, exactly the scalar operation order:
                // r = (t*t)*t; t = (t * ((x+x)+r)) / ((x+r)+r). Written out
                // per half rather than through a closure: a closure would not
                // carry the caller's target feature, so the intrinsics inside
                // it would become out-of-line calls.
                let x_lo = _mm256_castps256_ps128(self.0);
                let x_hi = _mm256_extractf128_ps::<1>(self.0);
                let e_lo = _mm256_castps256_ps128(est);
                let e_hi = _mm256_extractf128_ps::<1>(est);
                let xd_lo: __m256d = _mm256_cvtps_pd(x_lo);
                let xd_hi: __m256d = _mm256_cvtps_pd(x_hi);
                let mut t_lo: __m256d = _mm256_cvtps_pd(e_lo);
                let mut t_hi: __m256d = _mm256_cvtps_pd(e_hi);
                for _ in 0..2 {
                    let r_lo = _mm256_mul_pd(_mm256_mul_pd(t_lo, t_lo), t_lo);
                    let r_hi = _mm256_mul_pd(_mm256_mul_pd(t_hi, t_hi), t_hi);
                    let num_lo = _mm256_add_pd(_mm256_add_pd(xd_lo, xd_lo), r_lo);
                    let num_hi = _mm256_add_pd(_mm256_add_pd(xd_hi, xd_hi), r_hi);
                    let den_lo = _mm256_add_pd(_mm256_add_pd(xd_lo, r_lo), r_lo);
                    let den_hi = _mm256_add_pd(_mm256_add_pd(xd_hi, r_hi), r_hi);
                    t_lo = _mm256_div_pd(_mm256_mul_pd(t_lo, num_lo), den_lo);
                    t_hi = _mm256_div_pd(_mm256_mul_pd(t_hi, num_hi), den_hi);
                }
                let out_lo = _mm256_cvtpd_ps(t_lo);
                let out_hi = _mm256_cvtpd_ps(t_hi);
                Some(Self(_mm256_insertf128_ps::<1>(
                    _mm256_castps128_ps256(out_lo),
                    out_hi,
                )))
            }
        }
    }

    impl core::ops::Add for F32x8 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            // SAFETY: module contract (AVX2 host, AVX2-enabled caller).
            Self(unsafe { _mm256_add_ps(self.0, rhs.0) })
        }
    }

    impl core::ops::Sub for F32x8 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            // SAFETY: module contract (AVX2 host, AVX2-enabled caller).
            Self(unsafe { _mm256_sub_ps(self.0, rhs.0) })
        }
    }

    impl core::ops::Mul for F32x8 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            // SAFETY: module contract (AVX2 host, AVX2-enabled caller).
            Self(unsafe { _mm256_mul_ps(self.0, rhs.0) })
        }
    }

    impl core::ops::Div for F32x8 {
        type Output = Self;
        #[inline(always)]
        fn div(self, rhs: Self) -> Self {
            // SAFETY: module contract (AVX2 host, AVX2-enabled caller).
            Self(unsafe { _mm256_div_ps(self.0, rhs.0) })
        }
    }

    impl F32Vec for F32x8 {
        const LANES: usize = 8;

        #[inline(always)]
        fn splat(value: f32) -> Self {
            // SAFETY: module contract (AVX2 host, AVX2-enabled caller).
            Self(unsafe { _mm256_set1_ps(value) })
        }

        #[inline(always)]
        fn load(src: &[f32]) -> Self {
            let Some(src) = src.first_chunk::<8>() else {
                debug_assert!(false, "AVX2 lane load from a short slice");
                return Self::splat(0.0);
            };
            // SAFETY: module contract for the instruction; `src` is a valid,
            // readable array of exactly eight `f32`s and the load is the
            // unaligned form.
            Self(unsafe { _mm256_loadu_ps(src.as_ptr()) })
        }

        #[inline(always)]
        fn store(self, dst: &mut [f32]) {
            let Some(dst) = dst.first_chunk_mut::<8>() else {
                debug_assert!(false, "AVX2 lane store to a short slice");
                return;
            };
            // SAFETY: module contract for the instruction; `dst` is a valid,
            // writable array of exactly eight `f32`s and the store is the
            // unaligned form.
            unsafe { _mm256_storeu_ps(dst.as_mut_ptr(), self.0) }
        }

        #[inline(always)]
        fn store_trunc_i32(self, dst: &mut [i32]) {
            let Some(dst) = dst.first_chunk_mut::<8>() else {
                debug_assert!(false, "AVX2 lane store to a short slice");
                return;
            };
            // SAFETY: module contract for the instructions; `dst` is a valid,
            // writable array of exactly eight `i32`s (32 bytes, the size of
            // one `__m256i`) and the store is the unaligned form.
            unsafe { _mm256_storeu_si256(dst.as_mut_ptr().cast(), _mm256_cvttps_epi32(self.0)) }
        }

        #[inline(always)]
        fn abs(self) -> Self {
            Self::splat(-0.0).andnot(self)
        }

        #[inline(always)]
        fn trunc(self) -> Self {
            // SAFETY: module contract (AVX2 host, AVX2-enabled caller).
            Self(unsafe { _mm256_cvtepi32_ps(_mm256_cvttps_epi32(self.0)) })
        }

        #[inline(always)]
        fn cmp_lt(self, rhs: Self) -> Self {
            // SAFETY: module contract (AVX2 host, AVX2-enabled caller).
            Self(unsafe { _mm256_cmp_ps::<_CMP_LT_OQ>(self.0, rhs.0) })
        }

        #[inline(always)]
        fn cmp_le(self, rhs: Self) -> Self {
            // SAFETY: module contract (AVX2 host, AVX2-enabled caller).
            Self(unsafe { _mm256_cmp_ps::<_CMP_LE_OQ>(self.0, rhs.0) })
        }

        #[inline(always)]
        fn cmp_eq(self, rhs: Self) -> Self {
            // SAFETY: module contract (AVX2 host, AVX2-enabled caller).
            Self(unsafe { _mm256_cmp_ps::<_CMP_EQ_OQ>(self.0, rhs.0) })
        }

        #[inline(always)]
        fn cmp_gt(self, rhs: Self) -> Self {
            // SAFETY: module contract (AVX2 host, AVX2-enabled caller).
            Self(unsafe { _mm256_cmp_ps::<_CMP_GT_OQ>(self.0, rhs.0) })
        }

        #[inline(always)]
        fn and(self, rhs: Self) -> Self {
            // SAFETY: module contract (AVX2 host, AVX2-enabled caller).
            Self(unsafe { _mm256_and_ps(self.0, rhs.0) })
        }

        #[inline(always)]
        fn or(self, rhs: Self) -> Self {
            // SAFETY: module contract (AVX2 host, AVX2-enabled caller).
            Self(unsafe { _mm256_or_ps(self.0, rhs.0) })
        }

        #[inline(always)]
        fn andnot(self, rhs: Self) -> Self {
            // SAFETY: module contract (AVX2 host, AVX2-enabled caller).
            Self(unsafe { _mm256_andnot_ps(self.0, rhs.0) })
        }

        #[inline(always)]
        fn blend(self, t: Self, f: Self) -> Self {
            // `blendv` picks the *second* operand where the mask is set.
            // SAFETY: module contract (AVX2 host, AVX2-enabled caller).
            Self(unsafe { _mm256_blendv_ps(f.0, t.0, self.0) })
        }

        #[inline(always)]
        fn all(self) -> bool {
            // SAFETY: module contract (AVX2 host, AVX2-enabled caller).
            unsafe { _mm256_movemask_ps(self.0) == 0xff }
        }

        #[inline(always)]
        fn any(self) -> bool {
            // SAFETY: module contract (AVX2 host, AVX2-enabled caller).
            unsafe { _mm256_movemask_ps(self.0) != 0 }
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::indexing_slicing,
    reason = "test fixtures index buffers they sized themselves"
)]
mod tests {
    use super::*;

    /// A tiny kernel exercising every trait operation, so each implementation
    /// can be compared lane-for-lane against the scalar one.
    fn exercise<V: F32Vec>(input: &[f32], out: &mut [f32], out_i: &mut [i32]) {
        for c in (0..input.len()).step_by(V::LANES) {
            let x = V::load(&input[c..]);
            let half = V::splat(0.5).or(x.and(V::splat(-0.0)));
            let t = (x + half).trunc();
            let big = t.abs().cmp_gt(V::splat(4.0));
            let small = t.abs().cmp_le(V::splat(1.0));
            let eq = t.cmp_eq(V::splat(2.0));
            let lt = x.cmp_lt(V::splat(0.0));
            let mask = big.or(small).or(eq.and(lt));
            let y = mask.blend(x * V::splat(3.0) - V::splat(1.0), (x + V::splat(2.0)) / x);
            let z = mask.andnot(y.abs()).or(mask.and(y));
            z.store(&mut out[c..]);
            t.store_trunc_i32(&mut out_i[c..]);
            // `all`/`any` summarise the whole vector, so check them against
            // the lanes of this very mask rather than against another width.
            let mut lanes = vec![0.0f32; V::LANES];
            mask.store(&mut lanes);
            let set = lanes.iter().filter(|m| m.to_bits() == u32::MAX).count();
            assert!(
                lanes
                    .iter()
                    .all(|m| m.to_bits() == u32::MAX || m.to_bits() == 0)
            );
            assert_eq!(mask.all(), set == V::LANES);
            assert_eq!(mask.any(), set > 0);
        }
    }

    fn inputs() -> Vec<f32> {
        let mut v = Vec::new();
        let mut state = 0x1234_5678u32;
        for i in 0..256 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            #[allow(clippy::cast_precision_loss)]
            let r = (state >> 8) as f32 / 16_777_216.0;
            let magnitude = if i % 3 == 0 { 8.0 } else { 1.5 };
            v.push((r - 0.5) * magnitude);
        }
        // Signed zeros and exact half-integers matter for `trunc`/sign logic.
        v[0] = 0.0;
        v[1] = -0.0;
        v[2] = 2.5;
        v[3] = -2.5;
        v
    }

    fn assert_same<V: F32Vec>(name: &str) {
        let input = inputs();
        let mut want = vec![0.0f32; input.len()];
        let mut want_i = vec![0i32; input.len()];
        exercise::<f32>(&input, &mut want, &mut want_i);
        let mut got = vec![0.0f32; input.len()];
        let mut got_i = vec![0i32; input.len()];
        exercise::<V>(&input, &mut got, &mut got_i);
        for (i, (w, g)) in want.iter().zip(got.iter()).enumerate() {
            assert_eq!(w.to_bits(), g.to_bits(), "{name} lane {i}: {w} vs {g}");
        }
        assert_eq!(want_i, got_i, "{name} integer lanes");
    }

    #[cfg(feature = "simd")]
    #[test]
    fn wide_vectors_match_scalar_bitwise() {
        assert_same::<wide::f32x4>("f32x4");
        assert_same::<wide::f32x8>("f32x8");
    }

    /// The AVX2 cube root is `f32::cbrt` bit for bit on every normal input
    /// tried (positive and negative, across the exponent range and around
    /// the opsin bias the XYB conversion feeds it), and declines exactly the
    /// scalar special cases.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[test]
    fn avx2_cbrt_matches_std() {
        if !crate::cpu::has_avx2() {
            return;
        }
        #[target_feature(enable = "avx2")]
        fn run() {
            let mut state = 0x9e37_79b9u32;
            let mut next = || {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                state
            };
            let mut inputs = Vec::new();
            for _ in 0..20_000 {
                // Random bit patterns restricted to finite normals.
                let bits = next();
                let exp = 1 + (bits >> 24) % 253;
                let mant = next() & 0x007f_ffff;
                let sign = bits & 0x8000_0000;
                inputs.push(f32::from_bits(sign | (exp << 23) | mant));
            }
            for i in 0..4096 {
                #[allow(clippy::cast_precision_loss)]
                let v = i as f32 / 4096.0;
                inputs.push(v + 0.003_793_073_4);
                inputs.push(v * 3.0 + 0.003_793_073_4);
            }
            inputs.extend_from_slice(&[1.0, -1.0, 8.0, 27.0, 0.001, 1e30, -1e-30, 3.5, 100.0]);
            for chunk in inputs.chunks_exact(8) {
                let v = avx2::F32x8::load(chunk);
                let got = v.cbrt().expect("normal inputs are handled");
                let mut out = [0.0f32; 8];
                got.store(&mut out);
                for (x, g) in chunk.iter().zip(out.iter()) {
                    assert_eq!(x.cbrt().to_bits(), g.to_bits(), "cbrt({x})");
                }
            }
            for special in [
                [0.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, -0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, f32::INFINITY, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, f32::NAN, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1e-40, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, f32::NEG_INFINITY, 1.0, 1.0],
            ] {
                assert!(avx2::F32x8::load(&special).cbrt().is_none(), "{special:?}");
            }
        }
        // SAFETY: AVX2 support was checked just above.
        #[allow(unsafe_code)]
        unsafe {
            run();
        }
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[test]
    fn avx2_matches_scalar_bitwise() {
        if !crate::cpu::has_avx2() {
            return;
        }
        #[target_feature(enable = "avx2")]
        fn run() {
            assert_same::<avx2::F32x8>("avx2");
        }
        // SAFETY: AVX2 support was checked just above.
        #[allow(unsafe_code)]
        unsafe {
            run();
        }
    }
}
