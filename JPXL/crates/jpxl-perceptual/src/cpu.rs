//! Runtime AVX2 detection for the metric's bit-identical SIMD dispatch.
//!
//! Kept inside this crate so the default metric path does not depend on
//! `jpxl-core`. The verdict is the same as `jpxl_core::cpu::has_avx2`: host
//! AVX2, unless `JPXL_DISABLE_AVX2` is set (the JPXL A/B kill switch, still
//! honoured when the metric is used from another codec).

use std::sync::OnceLock;

/// Cached verdict of the one-time detection.
static AVX2: OnceLock<bool> = OnceLock::new();

/// Does the running host support AVX2, and has it not been disabled through
/// `JPXL_DISABLE_AVX2`?
///
/// Computed once and cached; a hot-path call is one relaxed atomic load.
#[must_use]
pub(crate) fn has_avx2() -> bool {
    *AVX2.get_or_init(detect_avx2)
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
        let first = has_avx2();
        assert_eq!(has_avx2(), first);
        assert_eq!(has_avx2(), first);
    }
}
