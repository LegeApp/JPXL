//! High-level Rust API for JPXL.
//!
//! This facade is the normal integration point for applications. It accepts
//! ordinary interleaved pixel buffers, supplies safe decoder limits, and hides
//! the policy/writer split used inside the encoder. Lower-level crates remain
//! public for callers that need exact JPEG XL syntax or research controls.
//!
//! # Lossy contract
//!
//! [`Encoder::with_ssimulacra2_score`] is the normal way to ask for lossy
//! output: name the minimum perceptual quality and let the encoder find the
//! bytes. [`Encoder::with_target_bpp`] / [`Encoder::with_target_bytes`] (an
//! exact size) and [`Encoder::with_global_scale`] (a pinned quantizer) are
//! expert modes. Exactly one lossy target may be set; call
//! [`Encoder::lossless`] to reset before choosing another.
//!
//! # Lossless RGB
//!
//! ```
//! let rgb = vec![128u8; 16 * 16 * 3];
//! let encoded = jpxl::Encoder::new().encode_rgb8(16, 16, &rgb)?;
//! let decoded = jpxl::decode(&encoded)?;
//! assert_eq!((decoded.width, decoded.height), (16, 16));
//! # Ok::<(), jpxl::Error>(())
//! ```
//!
//! # Target-rate RGB
//!
//! ```no_run
//! let rgb = vec![128u8; 256 * 256 * 3];
//! let encoded = jpxl::Encoder::new()
//!     .with_target_bpp(1.0)?
//!     .with_effort(jpxl::Effort::Balanced)
//!     .encode_rgb8(256, 256, &rgb)?;
//! # Ok::<(), jpxl::Error>(())
//! ```

use std::fmt;

pub use jpxl_core::limits::Limits;
pub use jpxl_decode::decode::FloatPlane;
pub use jpxl_decode::{DecodedImage, Plane};
pub use jpxl_encode_policy::RateStatus;
pub use jpxl_encode_policy::request::MetricVersion;

use jpxl_encode_policy::request::{
    FixedQuantizerTarget, LossyTarget, PerceptualMetric, PerceptualTarget,
};

/// A convenient result type for the high-level API.
pub type Result<T> = std::result::Result<T, Error>;

/// A failure from decoding, lossless encoding, lossy policy, or option validation.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// The JPEG XL decoder rejected the input.
    Decode(jpxl_decode::DecodeError),
    /// The lossless writer or container wrapper rejected the request.
    Encode(jpxl_encode::EncodeError),
    /// The lossy policy or VarDCT writer rejected the request.
    Policy(jpxl_encode_policy::PolicyError),
    /// A high-level option was nonsensical.
    InvalidOption(&'static str),
    /// A well-formed request that this build does not yet implement.
    ///
    /// Distinct from [`Self::InvalidOption`]: the request is valid and will be
    /// honoured in a later release, so callers can special-case it rather than
    /// treat it as their own bug.
    Unsupported(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decode(error) => write!(f, "decode failed: {error}"),
            Self::Encode(error) => write!(f, "encode failed: {error}"),
            Self::Policy(error) => write!(f, "encode policy failed: {error}"),
            Self::InvalidOption(message) | Self::Unsupported(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Decode(error) => Some(error),
            Self::Encode(error) => Some(error),
            Self::Policy(error) => Some(error),
            Self::InvalidOption(_) | Self::Unsupported(_) => None,
        }
    }
}

impl From<jpxl_decode::DecodeError> for Error {
    fn from(value: jpxl_decode::DecodeError) -> Self {
        Self::Decode(value)
    }
}

impl From<jpxl_encode::EncodeError> for Error {
    fn from(value: jpxl_encode::EncodeError) -> Self {
        Self::Encode(value)
    }
}

impl From<jpxl_encode_policy::PolicyError> for Error {
    fn from(value: jpxl_encode_policy::PolicyError) -> Self {
        Self::Policy(value)
    }
}

/// Decode a JPEG XL codestream or container with conservative default limits.
///
/// Use [`Decoder`] when an application needs a custom allocation or dimension
/// budget for untrusted inputs.
pub fn decode(data: &[u8]) -> Result<DecodedImage> {
    Decoder::new().decode(data)
}

/// A decoder configured with explicit resource limits.
#[derive(Debug, Clone, Default)]
pub struct Decoder {
    limits: Limits,
}

impl Decoder {
    /// Construct a decoder with [`Limits::default`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace the resource limits used for subsequent decodes.
    #[must_use]
    pub const fn with_limits(mut self, limits: Limits) -> Self {
        self.limits = limits;
        self
    }

    /// Decode one still image from a naked codestream or Part 2 container.
    pub fn decode(&self, data: &[u8]) -> Result<DecodedImage> {
        Ok(jpxl_decode::decode(data, &self.limits)?)
    }
}

/// How hard the lossy encoder is allowed to work.
///
/// This is the perceptual-quality *effort* — the search-latency budget — and
/// is distinct from the lossless Modular effort set by
/// [`Encoder::with_lossless_effort`]. Each variant also carries a default
/// quality score ([`Self::default_score`]) used when a caller asks for lossy
/// output without naming one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Effort {
    /// Lower latency, with a bounded search.
    Fast,
    /// The normal production balance of density, quality, and latency.
    #[default]
    Balanced,
    /// Exhaustive reference search; substantially slower than production modes.
    #[cfg(feature = "quality-effort")]
    Quality,
}

impl Effort {
    /// The default minimum SSIMULACRA2 score for this effort, used when lossy
    /// output is requested without an explicit quality.
    #[must_use]
    pub const fn default_score(self) -> f64 {
        match self {
            Self::Fast => 70.0,
            Self::Balanced => 85.0,
            #[cfg(feature = "quality-effort")]
            Self::Quality => 90.0,
        }
    }
}

impl From<Effort> for jpxl_encode_policy::RateSearchPreset {
    fn from(value: Effort) -> Self {
        match value {
            Effort::Fast => Self::Fast,
            Effort::Balanced => Self::Balanced,
            #[cfg(feature = "quality-effort")]
            Effort::Quality => Self::Quality,
        }
    }
}

/// What a completed target-rate encode achieved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RateSummary {
    /// The byte budget the caller set.
    pub target_bytes: u64,
    /// The exact achieved size of the emitted codestream.
    pub achieved_bytes: u64,
    /// The controller's terminal state.
    pub status: RateStatus,
    /// Writer prices paid under the fast entropy model.
    pub fast_prices: u32,
    /// Exact writer prices attributed to finalist refinement.
    pub full_prices: u32,
}

/// Why a perceptual encode stopped where it did: the quality controller's
/// terminal state, or one of the two routes that bypass it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerceptualStatus {
    /// The selected stream met the requested score.
    Met,
    /// Adjacent quantizer rungs straddled the score; the meeting one was used.
    MetAdjacentRungs,
    /// The bounded controller met the score at its work cap.
    MetWorkCap,
    /// The coarsest quantizer already exceeds the score (a floor).
    SaturatedFloor,
    /// The finest quantizer still misses the score (a ceiling).
    SaturatedTop,
    /// The bounded controller ran out of probes before any candidate met the
    /// score; the emitted stream's `achieved_score` is below the request.
    UnderTargetWorkCap,
    /// A bounded fresh-structure rescue supplied the selected stream.
    RescuedFreshStructure,
    /// A score of 100 was satisfied by the mathematically lossless path.
    RoutedToLossless,
    /// The frame is too small for the perceptual path to apply.
    UnsupportedTooSmall,
}

/// What a completed perceptual encode achieved.
#[derive(Debug, Clone, PartialEq)]
pub struct PerceptualOutcome {
    /// The minimum score the caller asked for.
    pub requested_score: f64,
    /// The score the emitted stream achieved, when it was measured.
    pub achieved_score: Option<f64>,
    /// The exact achieved size of the emitted stream.
    pub exact_bytes: u64,
    /// The metric definition the scores are on.
    pub metric_version: MetricVersion,
    /// The controller's terminal state.
    pub status: PerceptualStatus,
    /// How many candidate streams were scored.
    pub probes: u32,
    /// How many exact writer prices were paid.
    pub prices: u32,
    /// Whether the quantizer ladder ran out of rungs.
    pub saturated: bool,
    /// The controller's `jpxl.quality-trace/1` record, when a search ran.
    pub trace_json: Option<String>,
}

/// What an encode produced, alongside its bytes.
///
/// Returned by [`Encoder::encode_rgb8_reported`] /
/// [`Encoder::encode_rgb16_reported`] so a caller learns how the encode was
/// resolved without re-deriving it from the bytes.
#[derive(Debug, Clone, PartialEq)]
pub enum EncodeReport {
    /// Lossless Modular output.
    Lossless,
    /// Target-rate VarDCT output.
    Rate(RateSummary),
    /// Perceptual-quality VarDCT output.
    Perceptual(PerceptualOutcome),
    /// Fixed-quantizer VarDCT output.
    FixedQuantizer {
        /// The exact achieved size of the emitted stream.
        bytes: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Mode {
    Lossless,
    Lossy(LossyTarget),
}

/// Builder for lossless or lossy JPEG XL encoding.
///
/// The default is lossless Modular encoding at effort 1, automatic worker
/// count, and a naked codestream. Select lossy VarDCT with exactly one of
/// [`with_ssimulacra2_score`](Self::with_ssimulacra2_score) (the normal
/// contract), [`with_target_bpp`](Self::with_target_bpp) /
/// [`with_target_bytes`](Self::with_target_bytes), or
/// [`with_global_scale`](Self::with_global_scale).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Encoder {
    mode: Mode,
    lossless_effort: jpxl_encode::Effort,
    effort: Effort,
    resources: jpxl_encode::EncodeResources,
    container: bool,
}

impl Default for Encoder {
    fn default() -> Self {
        Self {
            mode: Mode::Lossless,
            lossless_effort: jpxl_encode::Effort::DEFAULT,
            effort: Effort::default(),
            resources: jpxl_encode::EncodeResources::default(),
            container: false,
        }
    }
}

/// The pixels a perceptual encode scores against, borrowed from the caller.
#[derive(Debug, Clone, Copy)]
enum PerceptualSource<'a> {
    /// Interleaved 8-bit sRGB.
    Rgb8 {
        /// Width in samples.
        width: u32,
        /// Height in samples.
        height: u32,
        /// The samples.
        rgb: &'a [u8],
    },
    /// Interleaved high-precision sRGB.
    Rgb16 {
        /// Width in samples.
        width: u32,
        /// Height in samples.
        height: u32,
        /// The samples.
        rgb: &'a [u16],
        /// Bits per sample, `1..=16`.
        bits_per_sample: u32,
    },
}

impl PerceptualSource<'_> {
    /// `(width, height, bits_per_sample)`.
    const fn dimensions(&self) -> (u32, u32, u32) {
        match *self {
            Self::Rgb8 { width, height, .. } => (width, height, 8),
            Self::Rgb16 {
                width,
                height,
                bits_per_sample,
                ..
            } => (width, height, bits_per_sample),
        }
    }
}

impl Encoder {
    /// Construct a lossless encoder with production defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Select lossless Modular encoding, clearing any lossy target.
    #[must_use]
    pub const fn lossless(mut self) -> Self {
        self.mode = Mode::Lossless;
        self
    }

    /// Rejects a second lossy target: exactly one may be set.
    fn set_lossy(mut self, target: LossyTarget) -> Result<Self> {
        if matches!(self.mode, Mode::Lossy(_)) {
            return Err(Error::InvalidOption(
                "one lossy target only: call .lossless() to reset before choosing another",
            ));
        }
        self.mode = Mode::Lossy(target);
        Ok(self)
    }

    /// Select lossy VarDCT to a minimum SSIMULACRA2 score (the normal
    /// contract).
    ///
    /// A score of 100 means mathematically lossless. The score must be finite
    /// and in `0.0..=100.0`.
    pub fn with_ssimulacra2_score(self, score: f64) -> Result<Self> {
        let target = PerceptualTarget::new(PerceptualMetric::Ssimulacra2, score)
            .map_err(|_| Error::InvalidOption("ssimulacra2 score must be finite and in 0..=100"))?;
        self.set_lossy(LossyTarget::Perceptual(target))
    }

    /// Select lossy VarDCT to a bits-per-pixel ceiling (expert mode).
    pub fn with_target_bpp(self, bits_per_pixel: f64) -> Result<Self> {
        if !bits_per_pixel.is_finite() || bits_per_pixel <= 0.0 {
            return Err(Error::InvalidOption(
                "target bits per pixel must be finite and greater than zero",
            ));
        }
        self.set_lossy(LossyTarget::Rate(
            jpxl_encode_policy::RateTarget::BitsPerPixel(bits_per_pixel),
        ))
    }

    /// Select lossy VarDCT to an exact byte ceiling (expert mode).
    pub fn with_target_bytes(self, bytes: u64) -> Result<Self> {
        if bytes == 0 {
            return Err(Error::InvalidOption(
                "target byte count must be greater than zero",
            ));
        }
        self.set_lossy(LossyTarget::Rate(jpxl_encode_policy::RateTarget::Bytes(
            bytes,
        )))
    }

    /// Select lossy VarDCT at a pinned `global_scale` (expert mode).
    ///
    /// The other two quantizer scalars take the request's fixed-quantizer
    /// defaults.
    pub fn with_global_scale(self, global_scale: u32) -> Result<Self> {
        let defaults = jpxl_encode_policy::EncodeRequest::defaults();
        let target = FixedQuantizerTarget::new(
            global_scale,
            defaults.quant_lf.get(),
            defaults.hf_mul.get(),
        )?;
        self.set_lossy(LossyTarget::FixedQuantizer(target))
    }

    /// Choose the lossy effort (search-latency budget).
    #[must_use]
    pub const fn with_effort(mut self, effort: Effort) -> Self {
        self.effort = effort;
        self
    }

    /// Choose the lossless Modular search effort, from 1 (fastest) to 9
    /// (densest).
    pub fn with_lossless_effort(mut self, effort: u8) -> Result<Self> {
        self.lossless_effort = jpxl_encode::Effort::new(effort)?;
        Ok(self)
    }

    /// Limit section-parallel work to `threads` workers. One is serial.
    pub fn with_threads(mut self, threads: usize) -> Result<Self> {
        if threads == 0 {
            return Err(Error::InvalidOption("thread count must be at least one"));
        }
        self.resources = jpxl_encode::EncodeResources::groups(threads);
        Ok(self)
    }

    /// Replace the section-parallel work budget wholesale.
    #[must_use]
    pub const fn with_resources(mut self, resources: jpxl_encode::EncodeResources) -> Self {
        self.resources = resources;
        self
    }

    /// Wrap output in a standard Part 2 JPEG XL container.
    #[must_use]
    pub const fn with_container(mut self, container: bool) -> Self {
        self.container = container;
        self
    }

    /// Encode interleaved 8-bit sRGB samples.
    pub fn encode_rgb8(&self, width: u32, height: u32, rgb: &[u8]) -> Result<Vec<u8>> {
        Ok(self.encode_rgb8_reported(width, height, rgb)?.0)
    }

    /// Encode interleaved 8-bit sRGB samples, reporting how the encode resolved.
    pub fn encode_rgb8_reported(
        &self,
        width: u32,
        height: u32,
        rgb: &[u8],
    ) -> Result<(Vec<u8>, EncodeReport)> {
        match self.mode {
            Mode::Lossless => {
                let samples: Vec<u16> = rgb.iter().map(|&sample| u16::from(sample)).collect();
                let bytes = self.encode_lossless(width, height, 3, 8, &samples)?;
                Ok((bytes, EncodeReport::Lossless))
            }
            Mode::Lossy(LossyTarget::Rate(target)) => {
                let mut request = self.lossy_request(target);
                request.bits_per_sample = 8;
                let outcome = jpxl_encode_policy::encode_srgb8_to_target(
                    width, height, rgb, &request, target,
                )?;
                let report = EncodeReport::Rate(rate_summary(&outcome));
                let bytes = self.wrap_lossy(outcome.codestream, 8);
                Ok((bytes, report))
            }
            Mode::Lossy(LossyTarget::Perceptual(target)) => self.perceptual_encode(
                target,
                PerceptualSource::Rgb8 { width, height, rgb },
                || {
                    let samples: Vec<u16> = rgb.iter().map(|&sample| u16::from(sample)).collect();
                    self.encode_lossless(width, height, 3, 8, &samples)
                },
            ),
            Mode::Lossy(LossyTarget::FixedQuantizer(fixed)) => {
                let mut request = jpxl_encode_policy::EncodeRequest::for_fixed_quantizer(fixed);
                request.resources = self.resources;
                request.bits_per_sample = 8;
                let bytes = jpxl_encode_policy::encode_srgb8_vardct(width, height, rgb, &request)?;
                let wrapped = self.wrap_lossy(bytes, 8);
                let count = u64::try_from(wrapped.len()).unwrap_or(u64::MAX);
                Ok((wrapped, EncodeReport::FixedQuantizer { bytes: count }))
            }
        }
    }

    /// Encode interleaved high-precision sRGB samples (`1..=16` bits each).
    pub fn encode_rgb16(
        &self,
        width: u32,
        height: u32,
        bits_per_sample: u32,
        rgb: &[u16],
    ) -> Result<Vec<u8>> {
        Ok(self
            .encode_rgb16_reported(width, height, bits_per_sample, rgb)?
            .0)
    }

    /// Encode high-precision sRGB samples, reporting how the encode resolved.
    pub fn encode_rgb16_reported(
        &self,
        width: u32,
        height: u32,
        bits_per_sample: u32,
        rgb: &[u16],
    ) -> Result<(Vec<u8>, EncodeReport)> {
        match self.mode {
            Mode::Lossless => {
                let bytes = self.encode_lossless(width, height, 3, bits_per_sample, rgb)?;
                Ok((bytes, EncodeReport::Lossless))
            }
            Mode::Lossy(LossyTarget::Rate(target)) => {
                let request = self.lossy_request(target);
                let outcome = jpxl_encode_policy::encode_srgb16_to_target(
                    width,
                    height,
                    rgb,
                    bits_per_sample,
                    &request,
                    target,
                )?;
                let report = EncodeReport::Rate(rate_summary(&outcome));
                let bytes = self.wrap_lossy(outcome.codestream, bits_per_sample);
                Ok((bytes, report))
            }
            Mode::Lossy(LossyTarget::Perceptual(target)) => self.perceptual_encode(
                target,
                PerceptualSource::Rgb16 {
                    width,
                    height,
                    rgb,
                    bits_per_sample,
                },
                || self.encode_lossless(width, height, 3, bits_per_sample, rgb),
            ),
            Mode::Lossy(LossyTarget::FixedQuantizer(fixed)) => {
                let mut request = jpxl_encode_policy::EncodeRequest::for_fixed_quantizer(fixed);
                request.resources = self.resources;
                let bytes = jpxl_encode_policy::encode_srgb16_vardct(
                    width,
                    height,
                    rgb,
                    bits_per_sample,
                    &request,
                )?;
                let wrapped = self.wrap_lossy(bytes, bits_per_sample);
                let count = u64::try_from(wrapped.len()).unwrap_or(u64::MAX);
                Ok((wrapped, EncodeReport::FixedQuantizer { bytes: count }))
            }
        }
    }

    /// Encode 8-bit greyscale samples losslessly.
    ///
    /// The current VarDCT policy is RGB-only, so a lossy encoder returns a
    /// clear error instead of silently expanding greyscale to RGB.
    pub fn encode_gray8(&self, width: u32, height: u32, gray: &[u8]) -> Result<Vec<u8>> {
        let samples: Vec<u16> = gray.iter().map(|&sample| u16::from(sample)).collect();
        self.encode_gray16(width, height, 8, &samples)
    }

    /// Encode high-precision greyscale samples losslessly.
    pub fn encode_gray16(
        &self,
        width: u32,
        height: u32,
        bits_per_sample: u32,
        gray: &[u16],
    ) -> Result<Vec<u8>> {
        if matches!(self.mode, Mode::Lossy(_)) {
            return Err(Error::InvalidOption(
                "lossy VarDCT currently requires RGB input; use lossless mode for greyscale",
            ));
        }
        self.encode_lossless(width, height, 1, bits_per_sample, gray)
    }

    /// The perceptual path: score 100 routes to the lossless encoder, a frame
    /// below the metric's floor likewise, and everything else runs the
    /// quality controller with a plan-rendering SSIMULACRA2 evaluator.
    fn perceptual_encode<F>(
        &self,
        target: PerceptualTarget,
        source: PerceptualSource<'_>,
        lossless: F,
    ) -> Result<(Vec<u8>, EncodeReport)>
    where
        F: FnOnce() -> Result<Vec<u8>>,
    {
        let routed = |status: PerceptualStatus| -> Result<(Vec<u8>, EncodeReport)> {
            let bytes = lossless()?;
            let exact = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
            let outcome = PerceptualOutcome {
                requested_score: target.minimum_score,
                achieved_score: Some(100.0),
                exact_bytes: exact,
                metric_version: target.metric.version(),
                status,
                probes: 0,
                prices: 0,
                saturated: false,
                trace_json: None,
            };
            Ok((bytes, EncodeReport::Perceptual(outcome)))
        };
        // `minimum_score` is validated into `0.0..=100.0`, so `>= 100.0` is the
        // exact-lossless request.
        if target.minimum_score >= 100.0 {
            return routed(PerceptualStatus::RoutedToLossless);
        }
        let (width, height, bits) = source.dimensions();
        if width < jpxl_perceptual::MIN_DIMENSION || height < jpxl_perceptual::MIN_DIMENSION {
            return routed(PerceptualStatus::UnsupportedTooSmall);
        }

        let mut request = jpxl_encode_policy::EncodeRequest::for_quality(self.effort.into());
        request.resources = self.resources;
        request.bits_per_sample = bits;
        let executor = request.resources.executor();
        let (frame, mut evaluator) = match source {
            PerceptualSource::Rgb8 { width, height, rgb } => (
                jpxl_encode_policy::PreparedFrame::from_srgb8_with(
                    width,
                    height,
                    rgb,
                    Some(&executor),
                )?,
                jpxl_perceptual::PlanRenderEvaluator::from_srgb8(width, height, rgb, &executor)
                    .map_err(|_| {
                        Error::Unsupported("the frame cannot be scored by the perceptual metric")
                    })?,
            ),
            PerceptualSource::Rgb16 {
                width,
                height,
                rgb,
                bits_per_sample,
            } => (
                jpxl_encode_policy::PreparedFrame::from_srgb16_with(
                    width,
                    height,
                    rgb,
                    bits_per_sample,
                    Some(&executor),
                )?,
                jpxl_perceptual::PlanRenderEvaluator::from_srgb16(
                    width,
                    height,
                    rgb,
                    bits_per_sample,
                    &executor,
                )
                .map_err(|_| {
                    Error::Unsupported("the frame cannot be scored by the perceptual metric")
                })?,
            ),
        };
        let atlas = jpxl_encode_policy::AnalysisAtlas::analyze(&frame);
        let outcome = jpxl_encode_policy::search_frame_perceptual(
            &frame,
            &atlas,
            &request,
            target,
            &mut evaluator,
            &executor,
        )?;
        let effort_name = match self.effort {
            Effort::Fast => "fast",
            Effort::Balanced => "balanced",
            #[cfg(feature = "quality-effort")]
            Effort::Quality => "quality",
        };
        let trace_json = Some(outcome.trace_json(effort_name));
        let bytes = self.wrap_lossy(outcome.codestream, bits);
        let report = PerceptualOutcome {
            requested_score: target.minimum_score,
            achieved_score: Some(outcome.achieved_score),
            exact_bytes: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            metric_version: target.metric.version(),
            status: match outcome.status {
                jpxl_encode_policy::QualityStatus::Met => PerceptualStatus::Met,
                jpxl_encode_policy::QualityStatus::MetAdjacentRungs => {
                    PerceptualStatus::MetAdjacentRungs
                }
                jpxl_encode_policy::QualityStatus::MetWorkCap => PerceptualStatus::MetWorkCap,
                jpxl_encode_policy::QualityStatus::SaturatedFloor => {
                    PerceptualStatus::SaturatedFloor
                }
                jpxl_encode_policy::QualityStatus::SaturatedTop => PerceptualStatus::SaturatedTop,
                jpxl_encode_policy::QualityStatus::UnderTargetWorkCap => {
                    PerceptualStatus::UnderTargetWorkCap
                }
                jpxl_encode_policy::QualityStatus::RescuedFreshStructure => {
                    PerceptualStatus::RescuedFreshStructure
                }
            },
            probes: outcome.stats.pixel_probes,
            prices: outcome.stats.exact_prices,
            saturated: outcome.saturated,
            trace_json,
        };
        Ok((bytes, EncodeReport::Perceptual(report)))
    }

    fn encode_lossless(
        &self,
        width: u32,
        height: u32,
        channels: usize,
        bits_per_sample: u32,
        samples: &[u16],
    ) -> Result<Vec<u8>> {
        let image = jpxl_encode::Image::from_interleaved(
            width,
            height,
            channels,
            bits_per_sample,
            samples,
        )?;
        let options = jpxl_encode::EncodeOptions {
            container: self.container,
            resources: self.resources,
            effort: self.lossless_effort,
            ..jpxl_encode::EncodeOptions::default()
        };
        Ok(jpxl_encode::encode(&image, &options)?)
    }

    fn lossy_request(
        &self,
        target: jpxl_encode_policy::RateTarget,
    ) -> jpxl_encode_policy::EncodeRequest {
        let mut request = jpxl_encode_policy::EncodeRequest::for_target(target);
        request.rate_preset = self.effort.into();
        request.resources = self.resources;
        request
    }

    fn wrap_lossy(&self, codestream: Vec<u8>, bits_per_sample: u32) -> Vec<u8> {
        if !self.container {
            return codestream;
        }
        let level = if bits_per_sample > 8 {
            jpxl_encode::container::EXTENDED_LEVEL
        } else {
            jpxl_encode::container::DEFAULT_LEVEL
        };
        jpxl_encode::container::wrap(&codestream, level)
    }
}

/// Distils a policy [`RateOutcome`](jpxl_encode_policy::RateOutcome) into the
/// facade's [`RateSummary`].
fn rate_summary(outcome: &jpxl_encode_policy::RateOutcome) -> RateSummary {
    RateSummary {
        target_bytes: outcome.target,
        achieved_bytes: outcome.achieved(),
        status: outcome.status,
        fast_prices: outcome.stats.fast_prices,
        full_prices: outcome.stats.full_prices,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lossless_rgb8_roundtrips_through_facade() {
        let rgb = [0, 16, 32, 64, 128, 255];
        let encoded = Encoder::new()
            .with_container(true)
            .encode_rgb8(2, 1, &rgb)
            .expect("encode");
        let decoded = decode(&encoded).expect("decode");
        assert_eq!(decoded.interleaved_colour(), rgb.map(u16::from));
    }

    #[test]
    fn lossless_gray16_roundtrips_through_facade() {
        let gray = [0, 1024, 65_535, 7];
        let encoded = Encoder::new()
            .with_lossless_effort(2)
            .expect("effort")
            .encode_gray16(2, 2, 16, &gray)
            .expect("encode");
        let decoded = Decoder::new().decode(&encoded).expect("decode");
        assert_eq!(decoded.interleaved_colour(), gray);
    }

    #[test]
    fn builder_rejects_nonsensical_targets() {
        assert!(Encoder::new().with_target_bpp(0.0).is_err());
        assert!(Encoder::new().with_target_bpp(f64::NAN).is_err());
        assert!(Encoder::new().with_target_bytes(0).is_err());
        assert!(Encoder::new().with_threads(0).is_err());
    }

    #[test]
    fn target_rate_rgb8_uses_the_production_pipeline() {
        let (width, height) = (64u32, 64u32);
        let mut rgb = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height {
            for x in 0..width {
                rgb.extend_from_slice(&[
                    u8::try_from(x * 4).unwrap_or(u8::MAX),
                    u8::try_from(y * 4).unwrap_or(u8::MAX),
                    u8::try_from((x + y) * 2).unwrap_or(u8::MAX),
                ]);
            }
        }
        let target = 2_048u64;
        let encoded = Encoder::new()
            .with_target_bytes(target)
            .expect("target")
            .with_effort(Effort::Fast)
            .with_threads(1)
            .expect("threads")
            .encode_rgb8(width, height, &rgb)
            .expect("target-rate encode");
        assert!(u64::try_from(encoded.len()).unwrap_or(u64::MAX) <= target);
        let decoded = decode(&encoded).expect("decode");
        assert_eq!((decoded.width, decoded.height), (width, height));
    }
}
