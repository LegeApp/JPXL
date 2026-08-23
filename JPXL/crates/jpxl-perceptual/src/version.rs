//! Metric identity.
//!
//! A perceptual encode is only reproducible if the metric that selected its
//! codestream is named. Changing anything that moves a score — the colour
//! transform, the blur, the pooling, a weight — bumps this string, and a bump
//! is an encoder-behaviour change that needs its own Contract B screen.

/// The production metric implemented by this crate.
pub const METRIC_VERSION: &str = "ssimulacra2-jpxl-1";

/// The Gaussian standard deviation the blur is derived for.
pub const BLUR_SIGMA: f64 = 1.5;

/// Number of pyramid scales evaluated (1:1 through 1:32) when the image is
/// large enough for all of them.
pub const SCALES: usize = 6;
