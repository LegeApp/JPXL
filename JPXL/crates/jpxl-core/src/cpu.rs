//! Runtime CPU-feature detection for the SIMD kernels.
//!
//! The workspace is compiled for baseline x86-64, so `wide`'s 8-lane vectors
//! are two SSE registers unless a kernel is compiled for AVX2. Kernels that
//! matter are therefore built twice — once for the baseline, once inside a
//! `#[target_feature(enable = "avx2")]` function — and the caller picks the
//! AVX2 build at run time when [`has_avx2`] reports the host supports it.
//!
//! Selecting a wider register never changes a result: every kernel performs
//! the same IEEE-754 operations in the same order on every lane, and no path
//! contracts multiply-add into a fused operation, so the AVX2 build and the
//! baseline build are bit-identical (the DCT tests pin this for the lane
//! kernels). The dispatch is a pure speed choice, which is also why it can be
//! disabled with the `JPXL_DISABLE_AVX2` environment variable for A/B timing
//! or to exercise the fallback path on an AVX2 host.

use std::sync::OnceLock;

/// Cached verdict of the one-time detection.
static AVX2: OnceLock<bool> = OnceLock::new();

/// Cached verdict of the one-time FMA detection.
static FMA: OnceLock<bool> = OnceLock::new();

/// Does the running host support AVX2, and has it not been disabled through
/// `JPXL_DISABLE_AVX2`?
///
/// The answer is computed once and cached; calling this on a hot path costs
/// one relaxed atomic load.
#[must_use]
pub fn has_avx2() -> bool {
    *AVX2.get_or_init(detect_avx2)
}

/// Does the running host support AVX2 *and* FMA, and has dispatch not been
/// disabled through `JPXL_DISABLE_AVX2`?
///
/// Kernels that spell out `mul_add` are bit-identical with or without a
/// fused instruction — `f32::mul_add` is a single rounding either way — so
/// this, too, is a pure speed choice: without it every `mul_add` is a
/// library call on baseline x86-64.
#[must_use]
pub fn has_fma() -> bool {
    *FMA.get_or_init(detect_fma)
}

fn detect_fma() -> bool {
    if std::env::var_os("JPXL_DISABLE_AVX2").is_some() {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn detect_avx2() -> bool {
    if std::env::var_os("JPXL_DISABLE_AVX2").is_some() {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection_is_stable() {
        assert_eq!(has_avx2(), has_avx2());
        assert_eq!(has_fma(), has_fma());
    }
}
