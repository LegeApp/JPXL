//! High-level Rust API for JPXL.
//!
//! This facade is the normal integration point for applications. It accepts
//! ordinary interleaved pixel buffers, supplies safe decoder limits, and hides
//! the policy/writer split used inside the encoder. Lower-level crates remain
//! public for callers that need exact JPEG XL syntax or research controls.
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
//!     .with_preset(jpxl::Preset::Balanced)
//!     .encode_rgb8(256, 256, &rgb)?;
//! # Ok::<(), jpxl::Error>(())
//! ```

use std::fmt;

pub use jpxl_core::limits::Limits;
pub use jpxl_decode::decode::FloatPlane;
pub use jpxl_decode::{DecodedImage, Plane};

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
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decode(error) => write!(f, "decode failed: {error}"),
            Self::Encode(error) => write!(f, "encode failed: {error}"),
            Self::Policy(error) => write!(f, "encode policy failed: {error}"),
            Self::InvalidOption(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Decode(error) => Some(error),
            Self::Encode(error) => Some(error),
            Self::Policy(error) => Some(error),
            Self::InvalidOption(_) => None,
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

/// Production presets for target-rate lossy encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Preset {
    /// Lower latency, with a bounded exact-size search.
    Fast,
    /// The normal production balance of density, quality, and latency.
    #[default]
    Balanced,
    /// Exhaustive reference search; substantially slower than production modes.
    Quality,
}

impl From<Preset> for jpxl_encode_policy::RateSearchPreset {
    fn from(value: Preset) -> Self {
        match value {
            Preset::Fast => Self::Fast,
            Preset::Balanced => Self::Balanced,
            Preset::Quality => Self::Quality,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Mode {
    Lossless,
    Lossy(jpxl_encode_policy::RateTarget),
}

/// Builder for lossless or target-rate JPEG XL encoding.
///
/// The default is lossless Modular encoding at effort 1, automatic worker
/// count, and a naked codestream. Call [`with_target_bpp`](Self::with_target_bpp)
/// or [`with_target_bytes`](Self::with_target_bytes) to select lossy VarDCT.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Encoder {
    mode: Mode,
    effort: jpxl_encode::Effort,
    preset: Preset,
    resources: jpxl_encode::EncodeResources,
    container: bool,
}

impl Default for Encoder {
    fn default() -> Self {
        Self {
            mode: Mode::Lossless,
            effort: jpxl_encode::Effort::DEFAULT,
            preset: Preset::default(),
            resources: jpxl_encode::EncodeResources::default(),
            container: false,
        }
    }
}

impl Encoder {
    /// Construct a lossless encoder with production defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Select lossless Modular encoding.
    #[must_use]
    pub const fn lossless(mut self) -> Self {
        self.mode = Mode::Lossless;
        self
    }

    /// Select lossy VarDCT encoding to a bits-per-pixel ceiling.
    pub fn with_target_bpp(mut self, bits_per_pixel: f64) -> Result<Self> {
        if !bits_per_pixel.is_finite() || bits_per_pixel <= 0.0 {
            return Err(Error::InvalidOption(
                "target bits per pixel must be finite and greater than zero",
            ));
        }
        self.mode = Mode::Lossy(jpxl_encode_policy::RateTarget::BitsPerPixel(bits_per_pixel));
        Ok(self)
    }

    /// Select lossy VarDCT encoding to an exact byte ceiling.
    pub fn with_target_bytes(mut self, bytes: u64) -> Result<Self> {
        if bytes == 0 {
            return Err(Error::InvalidOption(
                "target byte count must be greater than zero",
            ));
        }
        self.mode = Mode::Lossy(jpxl_encode_policy::RateTarget::Bytes(bytes));
        Ok(self)
    }

    /// Choose the target-rate search preset.
    #[must_use]
    pub const fn with_preset(mut self, preset: Preset) -> Self {
        self.preset = preset;
        self
    }

    /// Choose the lossless search effort, from 1 (fastest) to 9 (densest).
    pub fn with_effort(mut self, effort: u8) -> Result<Self> {
        self.effort = jpxl_encode::Effort::new(effort)?;
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

    /// Wrap output in a standard Part 2 JPEG XL container.
    #[must_use]
    pub const fn with_container(mut self, container: bool) -> Self {
        self.container = container;
        self
    }

    /// Encode interleaved 8-bit sRGB samples.
    pub fn encode_rgb8(&self, width: u32, height: u32, rgb: &[u8]) -> Result<Vec<u8>> {
        match self.mode {
            Mode::Lossless => {
                let samples: Vec<u16> = rgb.iter().map(|&sample| u16::from(sample)).collect();
                self.encode_lossless(width, height, 3, 8, &samples)
            }
            Mode::Lossy(target) => {
                let mut request = self.lossy_request(target);
                request.bits_per_sample = 8;
                let bytes = jpxl_encode_policy::encode_srgb8_vardct(width, height, rgb, &request)?;
                Ok(self.wrap_lossy(bytes, 8))
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
        match self.mode {
            Mode::Lossless => self.encode_lossless(width, height, 3, bits_per_sample, rgb),
            Mode::Lossy(target) => {
                let request = self.lossy_request(target);
                let bytes = jpxl_encode_policy::encode_srgb16_vardct(
                    width,
                    height,
                    rgb,
                    bits_per_sample,
                    &request,
                )?;
                Ok(self.wrap_lossy(bytes, bits_per_sample))
            }
        }
    }

    /// Encode 8-bit greyscale samples losslessly.
    ///
    /// The current VarDCT policy is RGB-only, so a target-rate encoder returns
    /// a clear error instead of silently expanding greyscale to RGB.
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
            effort: self.effort,
            ..jpxl_encode::EncodeOptions::default()
        };
        Ok(jpxl_encode::encode(&image, &options)?)
    }

    fn lossy_request(
        &self,
        target: jpxl_encode_policy::RateTarget,
    ) -> jpxl_encode_policy::EncodeRequest {
        let mut request = jpxl_encode_policy::EncodeRequest::for_target(target);
        request.rate_preset = self.preset.into();
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
            .with_effort(2)
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
            .with_preset(Preset::Fast)
            .with_threads(1)
            .expect("threads")
            .encode_rgb8(width, height, &rgb)
            .expect("target-rate encode");
        assert!(u64::try_from(encoded.len()).unwrap_or(u64::MAX) <= target);
        let decoded = decode(&encoded).expect("decode");
        assert_eq!((decoded.width, decoded.height), (width, height));
    }
}
