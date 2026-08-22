//! In-tree perceptual metric for the JPXL quality controller.
//!
//! The encoder's normal lossy contract is a minimum perceptual score, so the
//! metric that defines that score has to live with the encoder: pinned,
//! deterministic, precomputing everything source-only once, and free of
//! third-party code on the production path. This crate is that metric.
//!
//! # Structure
//!
//! The implementation is a pipeline whose stages are separable on purpose —
//! a later backend may swap the decomposition, the masking or the pooling
//! while keeping the rest:
//!
//! * [`color`] — the opponent colour space the maps are evaluated in;
//! * [`pyramid`] — the 2:1 scale pyramid;
//! * [`blur`] — the Gaussian local-moment filter;
//! * [`pool`] — the per-pixel error maps and their norms;
//! * [`reference`] — the source-only work, done once;
//! * [`ssimulacra2`] — the candidate-side walk, weights and remap that make
//!   the first backend reproduce SSIMULACRA2.
//!
//! # Determinism
//!
//! Every reduction is in fixed order over fixed-size row bands, every kernel
//! uses only IEEE basic operations (including the cube root), and no stage
//! depends on the worker count. The same inputs score bit-identically under
//! any [`BandExecutor`].
//!
//! # Derivation boundary
//!
//! The algorithm and constants are the public definition of SSIMULACRA2
//! (cloudinary/ssimulacra2), cross-checked against the BSD-2 rust-av crate.
//! Nothing under `libjxl/` was consulted (AKR
//! `jpegxl-rs.policy.perceptual-metric-clean-room`).

pub mod bands;
pub mod blur;
pub mod color;
#[cfg(feature = "evaluator")]
pub mod evaluator;
pub mod executor;
pub mod pool;
pub mod pyramid;
pub mod reference;
pub mod ssimulacra2;
pub mod version;

#[cfg(feature = "evaluator")]
pub use evaluator::{EvaluatorError, PlanRenderEvaluator};
pub use executor::{BandExecutor, ScopedThreadExecutor, SerialExecutor};
pub use pool::ChannelTerms;
pub use reference::{PrecomputedReference, ReferenceRetention};
pub use ssimulacra2::{ScaleTerms, Ssimulacra2, Ssimulacra2Result};
pub use version::{METRIC_VERSION, SCALES};

/// Smallest width and height the metric is defined for.
pub const MIN_DIMENSION: u32 = 8;

/// Why a comparison could not be scored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricError {
    /// A dimension is zero.
    ZeroDimension,
    /// The image is smaller than [`MIN_DIMENSION`] in a dimension.
    TooSmall {
        /// Offending width.
        width: u32,
        /// Offending height.
        height: u32,
    },
    /// A plane does not hold `width * height` samples.
    PlaneLength {
        /// Samples the dimensions imply.
        expected: u64,
        /// Samples found.
        found: u64,
    },
    /// Reference and candidate dimensions differ.
    DimensionMismatch {
        /// Reference `(width, height)`.
        reference: (u32, u32),
        /// Candidate `(width, height)`.
        candidate: (u32, u32),
    },
}

impl core::fmt::Display for MetricError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "image has a zero dimension"),
            Self::TooSmall { width, height } => write!(
                f,
                "image {width}x{height} is smaller than the metric's {MIN_DIMENSION}x{MIN_DIMENSION} floor"
            ),
            Self::PlaneLength { expected, found } => {
                write!(
                    f,
                    "plane holds {found} samples, dimensions imply {expected}"
                )
            }
            Self::DimensionMismatch {
                reference,
                candidate,
            } => write!(
                f,
                "candidate {}x{} does not match reference {}x{}",
                candidate.0, candidate.1, reference.0, reference.1
            ),
        }
    }
}

impl std::error::Error for MetricError {}

/// Three planar linear-sRGB channels, nominally in `0..=1`, row-major.
#[derive(Debug, Clone, Copy)]
pub struct LinearRgbView<'a> {
    width: u32,
    height: u32,
    r: &'a [f32],
    g: &'a [f32],
    b: &'a [f32],
}

impl<'a> LinearRgbView<'a> {
    /// Wraps three planes after checking their lengths.
    ///
    /// # Errors
    ///
    /// [`MetricError::ZeroDimension`] or [`MetricError::PlaneLength`].
    pub fn new(
        width: u32,
        height: u32,
        r: &'a [f32],
        g: &'a [f32],
        b: &'a [f32],
    ) -> Result<Self, MetricError> {
        if width == 0 || height == 0 {
            return Err(MetricError::ZeroDimension);
        }
        let expected = u64::from(width) * u64::from(height);
        for plane in [r, g, b] {
            let found = plane.len() as u64;
            if found != expected {
                return Err(MetricError::PlaneLength { expected, found });
            }
        }
        Ok(Self {
            width,
            height,
            r,
            g,
            b,
        })
    }

    /// Width in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// The red plane.
    #[must_use]
    pub const fn r(&self) -> &'a [f32] {
        self.r
    }

    /// The green plane.
    #[must_use]
    pub const fn g(&self) -> &'a [f32] {
        self.g
    }

    /// The blue plane.
    #[must_use]
    pub const fn b(&self) -> &'a [f32] {
        self.b
    }
}

/// Scores one pair from scratch on the calling thread, retaining nothing.
///
/// # Errors
///
/// Any [`MetricError`] the reference or the comparison raises.
pub fn score_pair(
    reference: LinearRgbView<'_>,
    candidate: LinearRgbView<'_>,
) -> Result<Ssimulacra2Result, MetricError> {
    let prepared =
        PrecomputedReference::new(reference, ReferenceRetention::Moments, &SerialExecutor)?;
    Ssimulacra2::new().score(&prepared, candidate, &SerialExecutor)
}
