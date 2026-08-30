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
//! bytes. The score is a hard floor: a successful perceptual encode has had
//! its actual reconstruction canonically scored at or above the request.
//! When the bounded controller cannot verify such a stream, the encode fails
//! with [`Error::TargetNotMet`] unless [`Encoder::with_quality_fallback`]
//! opted into an explicit fallback. [`Encoder::with_target_bpp`] /
//! [`Encoder::with_target_bytes`] (an exact size) and
//! [`Encoder::with_global_scale`] (a pinned quantizer) are expert modes.
//! Exactly one lossy target may be set; call [`Encoder::lossless`] to reset
//! before choosing another.
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
pub use jpxl_encode::ColourSpace;
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
    /// A perceptual encode whose bounded controller stopped without any
    /// stream canonically verified at the requested minimum score, while the
    /// encoder was left at [`QualityFallback::Refuse`] (the default).
    ///
    /// Nothing was emitted. The carried [`QualityMiss`] says how close the
    /// search got and why it stopped; [`Encoder::with_quality_fallback`]
    /// selects what to emit instead of failing.
    TargetNotMet(QualityMiss),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decode(error) => write!(f, "decode failed: {error}"),
            Self::Encode(error) => write!(f, "encode failed: {error}"),
            Self::Policy(error) => write!(f, "encode policy failed: {error}"),
            Self::InvalidOption(message) | Self::Unsupported(message) => f.write_str(message),
            Self::TargetNotMet(miss) => {
                let why = match miss.kind {
                    QualityMissKind::LadderSaturated => "even the finest quantizer",
                    QualityMissKind::WorkBudgetExhausted => "the probe budget's best candidate",
                };
                write!(
                    f,
                    "quality target not met: {why} verified {:.4}, below the requested \
                     minimum {:.4} ({}); nothing was written — choose a quality fallback \
                     to emit anyway",
                    miss.best_score, miss.requested_score, miss.metric_version,
                )
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Decode(error) => Some(error),
            Self::Encode(error) => Some(error),
            Self::Policy(error) => Some(error),
            Self::InvalidOption(_) | Self::Unsupported(_) | Self::TargetNotMet(_) => None,
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

/// What a perceptual encode emits when the bounded controller stops without
/// any stream canonically verified at the requested minimum score.
///
/// The score is a hard floor: an under-target stream is never an ordinary
/// success. [`Self::Refuse`] (the default) turns such an encode into
/// [`Error::TargetNotMet`]; the two alternatives are explicit contracts a
/// caller opts into with [`Encoder::with_quality_fallback`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QualityFallback {
    /// Fail with [`Error::TargetNotMet`] and emit nothing. The default.
    #[default]
    Refuse,
    /// Emit a mathematically lossless stream instead, at the encoder's
    /// lossless effort, reported as [`PerceptualStatus::FallbackLossless`].
    /// The floor holds (lossless trivially meets any score) at whatever byte
    /// cost lossless carries — often far more than the lossy request implied.
    Lossless,
    /// Emit the finest canonically verified under-target stream, reported by
    /// [`PerceptualStatus::SaturatedTop`] or
    /// [`PerceptualStatus::UnderTargetWorkCap`] with its true
    /// `achieved_score`. This knowingly weakens the floor for this encode;
    /// the report says so explicitly.
    BestEffort,
}

/// Why a refused perceptual encode could not verify the requested score.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityMissKind {
    /// Even the quantizer ladder's finest rung scored below the request: the
    /// lossy path cannot reach this score on this image.
    LadderSaturated,
    /// The effort's bounded probe budget ran out before any candidate met
    /// the request; a finer candidate may exist but was never verified.
    WorkBudgetExhausted,
}

/// What a refused perceptual encode verified before it stopped, carried by
/// [`Error::TargetNotMet`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct QualityMiss {
    /// Why the search stopped short.
    pub kind: QualityMissKind,
    /// The minimum score the caller asked for.
    pub requested_score: f64,
    /// The canonical score of the finest verified candidate — the closest
    /// the bounded search got to the request.
    pub best_score: f64,
    /// The metric definition the scores are on.
    pub metric_version: MetricVersion,
    /// How many candidate streams were scored.
    pub probes: u32,
    /// How many exact writer prices were paid.
    pub prices: u32,
    /// The controller's `jpxl.quality-trace/2` record.
    pub trace_json: Option<String>,
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
    /// The finest quantizer still misses the score (a ceiling). Reported only
    /// under [`QualityFallback::BestEffort`]; the default refuses instead
    /// with [`Error::TargetNotMet`].
    SaturatedTop,
    /// The bounded controller ran out of probes before any candidate met the
    /// score; the emitted stream's `achieved_score` is below the request.
    /// Reported only under [`QualityFallback::BestEffort`]; the default
    /// refuses instead with [`Error::TargetNotMet`].
    UnderTargetWorkCap,
    /// A bounded fresh-structure rescue supplied the selected stream.
    RescuedFreshStructure,
    /// A score of 100 was satisfied by the mathematically lossless path.
    RoutedToLossless,
    /// The bounded lossy controller could not verify the requested score and
    /// [`QualityFallback::Lossless`] emitted a mathematically lossless stream
    /// in its place. `probes`/`prices` count the failed lossy search.
    FallbackLossless,
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
    /// The controller's `jpxl.quality-trace/2` record, when a search ran.
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
/// The lossless Modular effort every routed text/UI candidate is encoded at:
/// dense enough that the palette/squeeze search engages (effort 1 leaves
/// ~3x bytes on paletteable screens), cheap enough to price two or three
/// candidates per routed frame.
const TEXT_CANDIDATE_EFFORT: u8 = 5;

/// One priced candidate of the routed text/UI ladder.
struct TextCandidate {
    codestream: Vec<u8>,
    score: f64,
}

/// The routed ladder's outcome: the winning candidate (if any survived
/// encoding) and the attempts as a JSON array for the quality trace.
struct TextCompetition {
    winner: Option<TextCandidate>,
    attempts_json: String,
}

/// The default is lossless Modular encoding at effort 1, automatic worker
/// count, and a naked codestream. Select lossy VarDCT with exactly one of
/// [`with_ssimulacra2_score`](Self::with_ssimulacra2_score) (the normal
/// contract), [`with_target_bpp`](Self::with_target_bpp) /
/// [`with_target_bytes`](Self::with_target_bytes), or
/// [`with_global_scale`](Self::with_global_scale).
#[derive(Debug, Clone, PartialEq)]
pub struct Encoder {
    mode: Mode,
    lossless_effort: jpxl_encode::Effort,
    effort: Effort,
    resources: jpxl_encode::EncodeResources,
    container: bool,
    jxlp_fragment_size: Option<usize>,
    quality_fallback: QualityFallback,
    colour_space: ColourSpace,
    exif: Option<Vec<u8>>,
    text_routing: bool,
}

impl Default for Encoder {
    fn default() -> Self {
        Self {
            mode: Mode::Lossless,
            lossless_effort: jpxl_encode::Effort::DEFAULT,
            effort: Effort::default(),
            resources: jpxl_encode::EncodeResources::default(),
            container: false,
            jxlp_fragment_size: None,
            quality_fallback: QualityFallback::Refuse,
            colour_space: ColourSpace::Srgb,
            exif: None,
            text_routing: false,
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
    /// and in `0.0..=100.0`. The score is a hard floor: when the bounded
    /// controller cannot canonically verify a stream at or above it, the
    /// encode fails with [`Error::TargetNotMet`] unless
    /// [`with_quality_fallback`](Self::with_quality_fallback) chose an
    /// explicit fallback.
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

    /// Choose what a perceptual encode emits when its bounded controller
    /// cannot verify a stream at the requested minimum score.
    ///
    /// The default is [`QualityFallback::Refuse`]: such an encode fails with
    /// [`Error::TargetNotMet`] rather than returning under-target bytes as a
    /// success.
    #[must_use]
    pub const fn with_quality_fallback(mut self, fallback: QualityFallback) -> Self {
        self.quality_fallback = fallback;
        self
    }

    /// Enable the text/UI routed candidate competition (experimental).
    ///
    /// On a lossy encode whose source the content classifier labels
    /// `TextUiLineArt` (census-sparse screenshots, diagrams, line art), the
    /// encoder also prices a small ladder of colour-reduced lossless Modular
    /// candidates, scores each canonically against the original, and emits
    /// the smallest stream that holds the requested SSIMULACRA2 floor. The
    /// competition can only shrink the output or leave it unchanged;
    /// classifier-negative sources are untouched.
    #[must_use]
    pub const fn with_text_routing(mut self, enabled: bool) -> Self {
        self.text_routing = enabled;
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

    /// Split the codestream across `jxlp` boxes of at most `size` bytes.
    ///
    /// Implies [`with_container`](Self::with_container): a fragmented
    /// codestream has nowhere to live outside a container.
    #[must_use]
    pub const fn with_jxlp_fragment_size(mut self, size: Option<usize>) -> Self {
        self.jxlp_fragment_size = size;
        self
    }

    /// Declare the colour space of the samples handed to the encoder
    /// (default: sRGB), signalled declaratively in the image header.
    ///
    /// The lossless Modular path stores samples untouched, so this changes
    /// only how a colour-managed viewer interprets them. The lossy VarDCT
    /// path is defined on sRGB input — its XYB transform and its perceptual
    /// metric assume it — so a lossy encode of a non-sRGB colour space fails
    /// with [`Error::Unsupported`] rather than mis-tagging the pixels.
    #[must_use]
    pub fn with_colour_space(mut self, colour_space: ColourSpace) -> Self {
        self.colour_space = colour_space;
        self
    }

    /// Attach an Exif metadata block, carried in a Part 2 `Exif` box.
    ///
    /// `exif` must be the raw Exif/TIFF payload as JEITA CP-3451E defines it,
    /// beginning with the TIFF byte-order header (`II*\0` or `MM\0*`) — the
    /// same bytes a camera stores after the `Exif\0\0` marker of a JPEG
    /// `APP1` segment, without that marker. Implies
    /// [`with_container`](Self::with_container): metadata boxes have nowhere
    /// to live outside a container.
    ///
    /// Per 18181-2 9.5 the codestream's own fields (dimensions, orientation)
    /// take precedence over Exif equivalents at decode time.
    pub fn with_exif(mut self, exif: Vec<u8>) -> Result<Self> {
        if !matches!(
            exif.first_chunk(),
            Some([0x49, 0x49, 0x2A, 0x00] | [0x4D, 0x4D, 0x00, 0x2A])
        ) {
            return Err(Error::InvalidOption(
                "the Exif payload must begin with a TIFF byte-order header (II*\\0 or MM\\0*)",
            ));
        }
        self.exif = Some(exif);
        Ok(self)
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
        self.require_srgb_for_lossy()?;
        match self.mode {
            Mode::Lossless => {
                let bytes = self.encode_lossless(width, height, 3, 8, rgb)?;
                Ok((bytes, EncodeReport::Lossless))
            }
            Mode::Lossy(LossyTarget::Rate(target)) => {
                let mut request = self.lossy_request(target);
                request.bits_per_sample = 8;
                let outcome = jpxl_encode_policy::encode_srgb8_to_target(
                    width, height, rgb, &request, target,
                )?;
                let report = EncodeReport::Rate(rate_summary(&outcome));
                let bytes = self.finish(outcome.codestream, 8);
                Ok((bytes, report))
            }
            Mode::Lossy(LossyTarget::Perceptual(target)) => self.perceptual_encode(
                target,
                PerceptualSource::Rgb8 { width, height, rgb },
                || self.encode_lossless(width, height, 3, 8, rgb),
            ),
            Mode::Lossy(LossyTarget::FixedQuantizer(fixed)) => {
                let mut request = jpxl_encode_policy::EncodeRequest::for_fixed_quantizer(fixed);
                request.resources = self.resources;
                request.bits_per_sample = 8;
                let bytes = jpxl_encode_policy::encode_srgb8_vardct(width, height, rgb, &request)?;
                let wrapped = self.finish(bytes, 8);
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
        self.require_srgb_for_lossy()?;
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
                let bytes = self.finish(outcome.codestream, bits_per_sample);
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
                let wrapped = self.finish(bytes, bits_per_sample);
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
        if matches!(self.mode, Mode::Lossy(_)) {
            return Err(Error::InvalidOption(
                "lossy VarDCT currently requires RGB input; use lossless mode for greyscale",
            ));
        }
        self.encode_lossless(width, height, 1, 8, gray)
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
        // A lossless stream trivially meets any score, so every lossless
        // route reports `achieved_score` 100; `probes`/`prices` are those of
        // whatever lossy search ran first (zero on the two early routes).
        let routed = |bytes: Vec<u8>,
                      status: PerceptualStatus,
                      probes: u32,
                      prices: u32,
                      trace_json: Option<String>|
         -> (Vec<u8>, EncodeReport) {
            let exact = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
            let outcome = PerceptualOutcome {
                requested_score: target.minimum_score,
                achieved_score: Some(100.0),
                exact_bytes: exact,
                metric_version: target.metric.version(),
                status,
                probes,
                prices,
                saturated: false,
                trace_json,
            };
            (bytes, EncodeReport::Perceptual(outcome))
        };
        // `minimum_score` is validated into `0.0..=100.0`, so `>= 100.0` is the
        // exact-lossless request.
        if target.minimum_score >= 100.0 {
            return Ok(routed(
                lossless()?,
                PerceptualStatus::RoutedToLossless,
                0,
                0,
                None,
            ));
        }
        let (width, height, bits) = source.dimensions();
        if width < jpxl_perceptual::MIN_DIMENSION || height < jpxl_perceptual::MIN_DIMENSION {
            return Ok(routed(
                lossless()?,
                PerceptualStatus::UnsupportedTooSmall,
                0,
                0,
                None,
            ));
        }

        let mut request = jpxl_encode_policy::EncodeRequest::for_quality(self.effort.into());
        request.resources = self.resources;
        request.bits_per_sample = bits;
        // Shadow content classification (text/UI-aware routing, stage B1):
        // computed from the source alone and recorded in the trace. It does
        // not touch the search and cannot change a single emitted byte.
        let content_hint = match source {
            PerceptualSource::Rgb8 { width, height, rgb } => {
                jpxl_encode_policy::content_class::classify_srgb8(width, height, rgb)
            }
            PerceptualSource::Rgb16 {
                width,
                height,
                rgb,
                bits_per_sample,
            } => jpxl_encode_policy::content_class::classify_srgb16(
                width,
                height,
                rgb,
                bits_per_sample,
            ),
        };
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
        // The routed text/UI candidate competition (opt-in): on a
        // classifier-positive frame, price a small ladder of colour-reduced
        // lossless Modular candidates against the search's result. The
        // canonical metric arbitrates; a candidate can only replace the
        // VarDCT stream when it holds the requested floor AND is smaller (or
        // the search itself came in under target).
        let routed_competition = if self.text_routing
            && content_hint.class == jpxl_encode_policy::content_class::ContentClass::TextUiLineArt
        {
            self.text_candidate_competition(source, target.minimum_score, &mut evaluator)
        } else {
            None
        };

        // The content hint (and the competition's attempts, when it ran) ride
        // the quality trace as additive fields (`jpxl.quality-trace/2`
        // consumers match known fields and tolerate extras, the same contract
        // the speculation shadow used).
        let trace_json = Some({
            let trace = outcome.trace_json(effort_name);
            match trace.strip_suffix('}') {
                Some(rest) => {
                    let mut extended =
                        format!("{rest},\"content_hint\":{}", content_hint.to_json());
                    if let Some(ref competition) = routed_competition {
                        extended.push_str(",\"routed_candidates\":");
                        extended.push_str(&competition.attempts_json);
                    }
                    extended.push('}');
                    extended
                }
                None => trace,
            }
        });

        // A routed candidate that holds the floor replaces the search result
        // when it is smaller, or when the search itself missed the target.
        let search_missed = matches!(
            outcome.status,
            jpxl_encode_policy::QualityStatus::SaturatedTop
                | jpxl_encode_policy::QualityStatus::UnderTargetWorkCap
        );
        if let Some(TextCompetition {
            winner: Some(winner),
            ..
        }) = routed_competition
            && (search_missed || winner.codestream.len() < outcome.codestream.len())
        {
            let bytes = self.finish(winner.codestream, bits);
            let report = PerceptualOutcome {
                requested_score: target.minimum_score,
                achieved_score: Some(winner.score),
                exact_bytes: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
                metric_version: target.metric.version(),
                status: PerceptualStatus::Met,
                probes: outcome.stats.pixel_probes,
                prices: outcome.stats.exact_prices,
                saturated: false,
                trace_json,
            };
            return Ok((bytes, EncodeReport::Perceptual(report)));
        }

        // The hard floor: a stream the controller verified below the request
        // is never an ordinary success. What happens instead is the encoder's
        // [`QualityFallback`]; only `BestEffort` proceeds to emit it.
        if matches!(
            outcome.status,
            jpxl_encode_policy::QualityStatus::SaturatedTop
                | jpxl_encode_policy::QualityStatus::UnderTargetWorkCap
        ) {
            match self.quality_fallback {
                QualityFallback::Refuse => {
                    return Err(Error::TargetNotMet(QualityMiss {
                        kind: quality_miss_kind(outcome.status),
                        requested_score: target.minimum_score,
                        best_score: outcome.achieved_score,
                        metric_version: target.metric.version(),
                        probes: outcome.stats.pixel_probes,
                        prices: outcome.stats.exact_prices,
                        trace_json,
                    }));
                }
                QualityFallback::Lossless => {
                    return Ok(routed(
                        lossless()?,
                        PerceptualStatus::FallbackLossless,
                        outcome.stats.pixel_probes,
                        outcome.stats.exact_prices,
                        trace_json,
                    ));
                }
                QualityFallback::BestEffort => {}
            }
        }

        let bytes = self.finish(outcome.codestream, bits);
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

    /// Runs the routed text/UI candidate ladder for
    /// [`Self::perceptual_encode`]: colour-reduce, score canonically, encode
    /// losslessly, first candidate that holds `floor` wins the ladder.
    ///
    /// Byte monotonicity makes the early exit sound: a 64-colour raster's
    /// lossless stream is never materially larger than a 256-colour or
    /// full-colour one of the same frame, so the first floor-holding rung is
    /// also the smallest. The final arm — the source itself, exactly
    /// lossless — always holds the floor, so a `Some` return always carries
    /// a winner; whether it beats the VarDCT stream is the caller's check.
    /// `None` means the ladder does not apply (deep samples, or a census the
    /// classifier should not have passed).
    fn text_candidate_competition(
        &self,
        source: PerceptualSource<'_>,
        floor: f64,
        evaluator: &mut jpxl_perceptual::PlanRenderEvaluator<'_>,
    ) -> Option<TextCompetition> {
        use jpxl_encode_policy::colour_reduce::reduce_to_k_colours;
        use jpxl_encode_policy::content_class::ROUTE_MAX_COLOURS;

        let (width, height, bits) = source.dimensions();
        if bits != 8 {
            return None;
        }
        let rgb8: std::borrow::Cow<'_, [u8]> = match source {
            PerceptualSource::Rgb8 { rgb, .. } => std::borrow::Cow::Borrowed(rgb),
            PerceptualSource::Rgb16 { rgb, .. } =>
            {
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "bits_per_sample == 8 bounds every sample to 0..=255"
                )]
                std::borrow::Cow::Owned(rgb.iter().map(|&s| s.min(255) as u8).collect())
            }
        };

        let mut attempts = Vec::new();
        let mut winner = None;
        for k in [64u32, 256] {
            let reduced = reduce_to_k_colours(&rgb8, k, ROUTE_MAX_COLOURS)?;
            let score = if reduced.exact {
                100.0
            } else {
                match evaluator.score_srgb8_candidate(width, height, &reduced.rgb) {
                    Ok(score) => score,
                    Err(_) => continue,
                }
            };
            let holds = score >= floor;
            let bytes = if holds {
                self.encode_text_candidate(width, height, &reduced.rgb).ok()
            } else {
                None
            };
            attempts.push(format!(
                "{{\"k\":{k},\"palette_len\":{},\"exact\":{},\"score\":{score},\"bytes\":{}}}",
                reduced.palette_len,
                reduced.exact,
                bytes
                    .as_ref()
                    .map_or_else(|| "null".into(), |b: &Vec<u8>| b.len().to_string()),
            ));
            if holds {
                if let Some(codestream) = bytes {
                    winner = Some(TextCandidate { codestream, score });
                }
                // `exact` already IS the lossless source; a coarser pass
                // met the floor, so finer (larger) rungs cannot win.
                break;
            }
            if reduced.exact {
                break;
            }
        }
        if winner.is_none() {
            // The exactly-lossless source always holds the floor.
            if let Ok(codestream) = self.encode_text_candidate(width, height, &rgb8) {
                attempts.push(format!(
                    "{{\"k\":null,\"palette_len\":null,\"exact\":true,\"score\":100.0,\"bytes\":{}}}",
                    codestream.len()
                ));
                winner = Some(TextCandidate {
                    codestream,
                    score: 100.0,
                });
            }
        }
        Some(TextCompetition {
            winner,
            attempts_json: format!("[{}]", attempts.join(",")),
        })
    }

    /// Encodes one 8-bit candidate raster as a naked lossless codestream at
    /// the ladder's fixed Modular effort.
    fn encode_text_candidate(&self, width: u32, height: u32, rgb: &[u8]) -> Result<Vec<u8>> {
        let image = jpxl_encode::Image::from_interleaved(width, height, 3, 8, rgb)?;
        let options = jpxl_encode::EncodeOptions {
            container: false,
            jxlp_fragment_size: None,
            resources: self.resources,
            effort: jpxl_encode::Effort::new(TEXT_CANDIDATE_EFFORT)?,
            colour_space: self.colour_space,
            ..jpxl_encode::EncodeOptions::default()
        };
        Ok(jpxl_encode::encode(&image, &options)?)
    }

    fn encode_lossless<S: Copy + Into<i32>>(
        &self,
        width: u32,
        height: u32,
        channels: usize,
        bits_per_sample: u32,
        samples: &[S],
    ) -> Result<Vec<u8>> {
        let image = jpxl_encode::Image::from_interleaved(
            width,
            height,
            channels,
            bits_per_sample,
            samples,
        )?;
        // Ask for the naked codestream and wrap in `finish`, so the
        // container logic (including the Exif box) lives in one place for
        // the lossless and lossy paths alike.
        let options = jpxl_encode::EncodeOptions {
            container: false,
            jxlp_fragment_size: None,
            resources: self.resources,
            effort: self.lossless_effort,
            colour_space: self.colour_space,
            ..jpxl_encode::EncodeOptions::default()
        };
        Ok(self.finish(jpxl_encode::encode(&image, &options)?, bits_per_sample))
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

    /// Wraps a finished codestream per the encoder's container options: a
    /// container when asked for (or implied by a `jxlp` fragment size or an
    /// Exif payload), with the Exif box appended after the codestream boxes.
    fn finish(&self, codestream: Vec<u8>, bits_per_sample: u32) -> Vec<u8> {
        if !self.container && self.jxlp_fragment_size.is_none() && self.exif.is_none() {
            return codestream;
        }
        let level = if bits_per_sample > 8 {
            jpxl_encode::container::EXTENDED_LEVEL
        } else {
            jpxl_encode::container::DEFAULT_LEVEL
        };
        let mut file = match self.jxlp_fragment_size {
            Some(size) => jpxl_encode::container::wrap_fragmented(&codestream, level, size),
            None => jpxl_encode::container::wrap(&codestream, level),
        };
        if let Some(ref exif) = self.exif {
            jpxl_encode::container::append_exif(&mut file, exif);
        }
        file
    }

    /// The lossy VarDCT pipeline is defined on sRGB input: its XYB transform
    /// and its perceptual metric assume it, so anything else must be encoded
    /// losslessly rather than mis-tagged.
    const fn require_srgb_for_lossy(&self) -> Result<()> {
        if matches!(self.mode, Mode::Lossy(_)) && !matches!(self.colour_space, ColourSpace::Srgb) {
            return Err(Error::Unsupported(
                "lossy VarDCT is defined on sRGB input; encode other colour spaces losslessly",
            ));
        }
        Ok(())
    }
}

/// Which [`QualityMissKind`] an under-target terminal status names.
///
/// Only [`QualityStatus::SaturatedTop`](jpxl_encode_policy::QualityStatus) and
/// [`QualityStatus::UnderTargetWorkCap`](jpxl_encode_policy::QualityStatus)
/// are misses; every other status carries a verified at-or-above-target
/// stream and never reaches this mapping.
const fn quality_miss_kind(status: jpxl_encode_policy::QualityStatus) -> QualityMissKind {
    match status {
        jpxl_encode_policy::QualityStatus::SaturatedTop => QualityMissKind::LadderSaturated,
        _ => QualityMissKind::WorkBudgetExhausted,
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

    /// The archive-pipeline claim: an Exif payload attached at the facade
    /// comes back byte-identical from the decoder's box walk, on both the
    /// lossless and the lossy path, and implies the container.
    #[test]
    fn an_exif_payload_survives_encode_to_the_decoders_box_walk() {
        let mut exif = vec![0x4D, 0x4D, 0x00, 0x2A];
        exif.extend((0..32u32).map(|i| (i % 5) as u8));

        let (width, height) = (64u32, 64u32);
        let rgb = gradient_rgb8(width, height);
        for encoded in [
            Encoder::new()
                .with_exif(exif.clone())
                .expect("valid payload")
                .encode_rgb8(width, height, &rgb)
                .expect("lossless encode"),
            Encoder::new()
                .with_target_bytes(2_048)
                .expect("target")
                .with_effort(Effort::Fast)
                .with_threads(1)
                .expect("threads")
                .with_exif(exif.clone())
                .expect("valid payload")
                .encode_rgb8(width, height, &rgb)
                .expect("lossy encode"),
        ] {
            assert!(jpxl_decode::container::is_container(&encoded));
            let mut guard =
                jpxl_core::limits::AllocGuard::new(&jpxl_core::limits::Limits::relaxed());
            let tree = jpxl_decode::container::BoxTree::parse(&encoded, &mut guard).expect("boxes");
            tree.validate().expect("conforming");
            let boxes = tree.exif().expect("well-formed Exif");
            assert_eq!(boxes.len(), 1);
            let first = boxes.first().expect("length asserted above");
            assert_eq!(first.payload, &exif[..]);
            decode(&encoded).expect("still decodes");
        }
    }

    #[test]
    fn a_payload_without_a_tiff_header_is_rejected() {
        assert!(Encoder::new().with_exif(vec![]).is_err());
        assert!(
            Encoder::new()
                .with_exif(vec![0xFF, 0xD8, 0xFF, 0xE1])
                .is_err()
        );
    }

    /// A non-sRGB colour space is signalled on the lossless path and refused
    /// (not mis-tagged) on the lossy path.
    #[test]
    fn colour_space_signalling_and_the_lossy_guard() {
        let rgb: Vec<u16> = (0..4 * 4 * 3u16).map(|i| i * 512).collect();
        let encoded = Encoder::new()
            .with_colour_space(ColourSpace::Rec2020)
            .encode_rgb16(4, 4, 16, &rgb)
            .expect("lossless rec2020");
        let decoded = decode(&encoded).expect("decode");
        assert_eq!(decoded.interleaved_colour(), rgb);

        let lossy = Encoder::new()
            .with_target_bytes(2_048)
            .expect("target")
            .with_colour_space(ColourSpace::Rec2020);
        assert!(matches!(
            lossy.encode_rgb16(4, 4, 16, &rgb),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn builder_rejects_nonsensical_targets() {
        assert!(Encoder::new().with_target_bpp(0.0).is_err());
        assert!(Encoder::new().with_target_bpp(f64::NAN).is_err());
        assert!(Encoder::new().with_target_bytes(0).is_err());
        assert!(Encoder::new().with_threads(0).is_err());
    }

    /// A gradient the VarDCT path can actually spend bits on.
    fn gradient_rgb8(width: u32, height: u32) -> Vec<u8> {
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
        rgb
    }

    #[test]
    fn target_rate_rgb8_uses_the_production_pipeline() {
        let (width, height) = (64u32, 64u32);
        let rgb = gradient_rgb8(width, height);
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

    /// A lossy encode wraps like a lossless one: `--container` and a `jxlp`
    /// fragment size both reach the VarDCT paths, and a fragment size alone
    /// implies the container.
    #[test]
    fn a_lossy_encode_wraps_in_a_container_when_asked() {
        let (width, height) = (64u32, 64u32);
        let rgb = gradient_rgb8(width, height);
        let encoder = Encoder::new()
            .with_target_bytes(2_048)
            .expect("target")
            .with_effort(Effort::Fast)
            .with_threads(1)
            .expect("threads");

        let naked = encoder.encode_rgb8(width, height, &rgb).expect("encode");
        assert!(!jpxl_decode::container::is_container(&naked));

        for wrapped in [
            encoder
                .clone()
                .with_container(true)
                .encode_rgb8(width, height, &rgb)
                .expect("container encode"),
            // Small enough to need several `jxlp` boxes, so a single-`jxlc`
            // fallback would not round-trip the same bytes.
            encoder
                .with_jxlp_fragment_size(Some(128))
                .encode_rgb8(width, height, &rgb)
                .expect("fragmented encode"),
        ] {
            assert!(jpxl_decode::container::is_container(&wrapped));
            let decoded = decode(&wrapped).expect("decode");
            assert_eq!((decoded.width, decoded.height), (width, height));
        }
    }
}
