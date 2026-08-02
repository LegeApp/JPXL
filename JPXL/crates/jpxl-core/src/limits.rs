//! Resource limits for attacker-facing decoding.
//!
//! A decoder reads bytes chosen by someone else. A codestream is a few hundred
//! bytes of header that can claim a 2^32 x 2^32 image, thousands of frames, or
//! a Huffman table with an absurd alphabet size — each of which turns into an
//! allocation if the parser believes it. The mitigation is structural rather
//! than ad-hoc:
//!
//! * every parser takes a `&Limits`, and
//! * every allocation whose size is derived from the codestream is metered
//!   through an [`AllocGuard`] *before* the memory is requested.
//!
//! "Before" is the whole point: `guard.charge(n)?` must run while `n` is still
//! just a number. Charging after `Vec::with_capacity(n)` is not a limit, it is
//! a post-mortem. The guard is cumulative across a decode, so a stream cannot
//! evade it by requesting many individually-plausible buffers.

use crate::error::{JpxlError, Result};

/// Upper bounds a decode is allowed to consume.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Maximum total pixels (width x height, summed over frames where relevant).
    pub max_pixels: u64,
    /// Maximum number of frames in an image.
    pub max_frames: u32,
    /// Maximum cumulative bytes that may be charged to an [`AllocGuard`].
    pub max_alloc_bytes: u64,
}

impl Limits {
    /// Defaults sized for untrusted input.
    ///
    /// * `max_pixels` = 2^30 (~1.07 gigapixels): far above any real photograph,
    ///   far below what a 32-bit-by-32-bit dimension pair can claim.
    /// * `max_frames` = 65536: generous for animation, bounded for per-frame
    ///   bookkeeping.
    /// * `max_alloc_bytes` = 4 GiB.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            max_pixels: 1 << 30,
            max_frames: 1 << 16,
            max_alloc_bytes: 4 << 30,
        }
    }

    /// Effectively unbounded limits, for trusted local input and for tests.
    ///
    /// Do not use this on data that arrived over a network.
    #[must_use]
    pub const fn relaxed() -> Self {
        Self {
            max_pixels: u64::MAX,
            max_frames: u32::MAX,
            max_alloc_bytes: u64::MAX,
        }
    }
}

impl Default for Limits {
    fn default() -> Self {
        Self::new()
    }
}

/// Cumulative allocation meter checked against [`Limits::max_alloc_bytes`].
///
/// Charge before you allocate:
///
/// ```
/// use jpxl_core::limits::{AllocGuard, Limits};
///
/// let limits = Limits::default();
/// let mut guard = AllocGuard::new(&limits);
/// let count: usize = 1024; // from the codestream
/// guard.charge(count as u64 * 4)?;
/// let buf = vec![0u32; count];
/// # assert_eq!(buf.len(), 1024);
/// # Ok::<(), jpxl_core::JpxlError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct AllocGuard {
    charged: u64,
    max_bytes: u64,
}

impl AllocGuard {
    /// Creates a guard enforcing `limits.max_alloc_bytes`.
    #[must_use]
    pub const fn new(limits: &Limits) -> Self {
        Self {
            charged: 0,
            max_bytes: limits.max_alloc_bytes,
        }
    }

    /// Bytes charged so far.
    #[must_use]
    pub const fn charged(&self) -> u64 {
        self.charged
    }

    /// Bytes still available before the limit is hit.
    #[must_use]
    pub const fn remaining(&self) -> u64 {
        self.max_bytes.saturating_sub(self.charged)
    }

    /// Accounts for `bytes` of upcoming allocation.
    ///
    /// # Errors
    ///
    /// Returns [`JpxlError::LimitExceeded`] if the running total would exceed
    /// the configured maximum. The guard is left unchanged on failure, so a
    /// caller may recover and try a smaller request.
    pub fn charge(&mut self, bytes: u64) -> Result<()> {
        let Some(total) = self.charged.checked_add(bytes) else {
            return Err(JpxlError::LimitExceeded(format!(
                "allocation of {bytes} bytes overflows the {} bytes already charged",
                self.charged
            )));
        };
        if total > self.max_bytes {
            return Err(JpxlError::LimitExceeded(format!(
                "allocation of {bytes} bytes would bring the total to {total}, over the \
                 max_alloc_bytes limit of {}",
                self.max_bytes
            )));
        }
        self.charged = total;
        Ok(())
    }

    /// Releases `bytes` previously charged, for buffers with a scoped lifetime.
    ///
    /// Saturates at zero rather than panicking on an over-release.
    pub const fn release(&mut self, bytes: u64) {
        self.charged = self.charged.saturating_sub(bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_bounded_but_generous() {
        let l = Limits::default();
        assert_eq!(l.max_pixels, 1 << 30);
        assert_eq!(l.max_frames, 1 << 16);
        assert_eq!(l.max_alloc_bytes, 4 << 30);
        assert_eq!(Limits::relaxed().max_pixels, u64::MAX);
    }

    #[test]
    fn charges_accumulate() {
        let limits = Limits {
            max_alloc_bytes: 100,
            ..Limits::default()
        };
        let mut guard = AllocGuard::new(&limits);
        guard.charge(60).expect("first charge fits");
        assert_eq!(guard.charged(), 60);
        assert_eq!(guard.remaining(), 40);
        guard.charge(40).expect("second charge exactly fills");
        assert_eq!(guard.remaining(), 0);
    }

    #[test]
    fn rejects_over_limit_and_preserves_state() {
        let limits = Limits {
            max_alloc_bytes: 100,
            ..Limits::default()
        };
        let mut guard = AllocGuard::new(&limits);
        guard.charge(90).expect("fits");
        let err = guard.charge(11).expect_err("11 more must not fit");
        assert!(matches!(err, JpxlError::LimitExceeded(_)));
        assert_eq!(guard.charged(), 90, "failed charge must not be recorded");
        guard.charge(10).expect("recovery with a smaller request");
    }

    #[test]
    fn rejects_overflowing_charge() {
        let mut guard = AllocGuard::new(&Limits::relaxed());
        guard
            .charge(u64::MAX - 1)
            .expect("fits under relaxed limits");
        let err = guard.charge(u64::MAX).expect_err("must detect overflow");
        assert!(matches!(err, JpxlError::LimitExceeded(_)));
    }

    #[test]
    fn release_saturates() {
        let mut guard = AllocGuard::new(&Limits::default());
        guard.charge(10).expect("fits");
        guard.release(25);
        assert_eq!(guard.charged(), 0);
    }
}
