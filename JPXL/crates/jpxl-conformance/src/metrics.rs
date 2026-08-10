//! A minimal PPM/NPY reader and pixel-difference metrics.
//!
//! Oracles emit binary PPM (`P6`) or NumPy `.npy`; this module reads just
//! enough of each format to compare two decodes. It is deliberately
//! **harness-local**: [`Image`] and [`FloatImage`] are comparison buffers, not
//! codec types, and nothing in `jpxl-decode` should grow a dependency on them.
//!
//! [`Image`] widens PPM samples to `u16`. An 8-bit file keeps its 0..=255
//! range (samples are *not* rescaled), so comparing an 8-bit PPM against a
//! 16-bit one is meaningless — [`max_abs_error`] refuses mismatched
//! [`Image::max_value`]s for exactly that reason. Modular-mode tests use
//! [`Image`] and [`max_abs_error`]/[`peak_error_per_channel`] for bit-exact
//! comparisons.
//!
//! [`FloatImage`] and [`similarity`] implement 18181-3 §4.2's comparison
//! surface instead: f32 samples on the nominal `[0, 1]` scale, no clipping,
//! peak error and per-channel RMSE. This is what VarDCT (and any tolerance-
//! graded) fixtures use, and it is what the official conformance corpus's
//! `test.json` thresholds are stated against.

use std::fmt;

/// A decoded PPM image held as interleaved samples.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    /// Width in pixels.
    pub w: u32,
    /// Height in pixels.
    pub h: u32,
    /// Samples per pixel. Always 3 for `P6`.
    pub channels: u32,
    /// The declared `maxval`: 255 for 8-bit files, 65535 for 16-bit ones.
    pub max_value: u16,
    /// Interleaved samples, row-major, `w * h * channels` long.
    pub samples: Vec<u16>,
}

impl Image {
    /// Total sample count, `w * h * channels`.
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether the image has no samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Whether two images have the same dimensions, channel count and depth.
    #[must_use]
    pub fn same_shape(&self, other: &Self) -> bool {
        self.w == other.w
            && self.h == other.h
            && self.channels == other.channels
            && self.max_value == other.max_value
            && self.samples.len() == other.samples.len()
    }

    /// Parse a binary PPM (`P6`), 8- or 16-bit.
    ///
    /// # Errors
    ///
    /// Returns [`PpmError`] for a wrong magic number, a malformed header, an
    /// unsupported `maxval`, or a truncated pixel payload.
    pub fn from_ppm(bytes: &[u8]) -> Result<Self, PpmError> {
        let mut cursor = Cursor::new(bytes);

        let magic = cursor.token()?;
        if magic != b"P6" {
            return Err(PpmError::BadMagic);
        }

        let w = cursor.number()?;
        let h = cursor.number()?;
        let declared_max = cursor.number()?;
        let max_value = match u16::try_from(declared_max) {
            Ok(0) | Err(_) => return Err(PpmError::UnsupportedMaxValue(declared_max)),
            Ok(v) => v,
        };
        // Exactly one whitespace byte separates the header from the payload.
        cursor.consume_single_whitespace()?;

        let channels: u32 = 3;
        let to_usize = |v: u32| usize::try_from(v).map_err(|_| PpmError::DimensionsOverflow);
        let sample_count = to_usize(w)?
            .checked_mul(to_usize(h)?)
            .and_then(|px| px.checked_mul(to_usize(channels).ok()?))
            .ok_or(PpmError::DimensionsOverflow)?;

        let wide = max_value > 255;
        let bytes_per_sample = if wide { 2 } else { 1 };
        let needed = sample_count
            .checked_mul(bytes_per_sample)
            .ok_or(PpmError::DimensionsOverflow)?;
        let payload = cursor.take(needed)?;

        let samples = if wide {
            payload
                .chunks_exact(2)
                .map(|pair| match pair {
                    [hi, lo] => u16::from(*hi) << 8 | u16::from(*lo),
                    _ => unreachable!("chunks_exact(2) yields pairs"),
                })
                .collect()
        } else {
            payload.iter().copied().map(u16::from).collect()
        };

        Ok(Self {
            w,
            h,
            channels,
            max_value,
            samples,
        })
    }
}

/// Largest absolute per-sample difference between two images.
///
/// Returns `None` when the images differ in shape (dimensions, channel count
/// or bit depth), which is a comparison failure rather than a large error.
///
/// ```
/// use jpxl_conformance::{Image, max_abs_error};
///
/// let ppm = |v: u8| {
///     let mut b = b"P6\n1 1\n255\n".to_vec();
///     b.extend_from_slice(&[v, v, v]);
///     Image::from_ppm(&b).expect("valid PPM")
/// };
/// assert_eq!(max_abs_error(&ppm(10), &ppm(13)), Some(3));
/// ```
#[must_use]
pub fn max_abs_error(a: &Image, b: &Image) -> Option<u32> {
    if !a.same_shape(b) {
        return None;
    }
    let worst = a
        .samples
        .iter()
        .zip(&b.samples)
        .map(|(&x, &y)| u32::from(x.abs_diff(y)))
        .max()
        .unwrap_or(0);
    Some(worst)
}

/// Largest absolute difference within each channel, in channel order.
///
/// Returns `None` on a shape mismatch, mirroring [`max_abs_error`]. The vector
/// is `channels` long; a channel with no samples contributes `0`.
#[must_use]
pub fn peak_error_per_channel(a: &Image, b: &Image) -> Option<Vec<u32>> {
    if !a.same_shape(b) {
        return None;
    }
    let Ok(channels) = usize::try_from(a.channels) else {
        return None;
    };
    if channels == 0 {
        return Some(Vec::new());
    }
    let mut peaks = vec![0_u32; channels];
    for (index, (&x, &y)) in a.samples.iter().zip(&b.samples).enumerate() {
        let diff = u32::from(x.abs_diff(y));
        if let Some(slot) = peaks.get_mut(index % channels)
            && diff > *slot
        {
            *slot = diff;
        }
    }
    Some(peaks)
}

/// Root-mean-square difference over all samples, in sample units.
///
/// Returns `None` on a shape mismatch, mirroring [`max_abs_error`]. Unlike
/// [`Similarity::channel_rmse`], which grades a conformance decode per channel
/// against a float reference, this is a single whole-image figure over the
/// integer PPM domain — what a rate/distortion comparison of two *encoders*
/// needs.
#[must_use]
pub fn rmse(a: &Image, b: &Image) -> Option<f64> {
    if !a.same_shape(b) {
        return None;
    }
    if a.samples.is_empty() {
        return Some(0.0);
    }
    let sq: f64 = a
        .samples
        .iter()
        .zip(&b.samples)
        .map(|(&x, &y)| {
            let d = f64::from(x.abs_diff(y));
            d * d
        })
        .sum();
    #[allow(
        clippy::cast_precision_loss,
        reason = "sample counts stay far inside f64's exact-integer range"
    )]
    let count = a.samples.len() as f64;
    Some((sq / count).sqrt())
}

/// Peak signal-to-noise ratio in decibels, against the image's own `max_value`.
///
/// Returns `None` on a shape mismatch, and [`f64::INFINITY`] for identical
/// images (zero error) — the mathematically correct value, which callers
/// formatting a table should special-case rather than print.
///
/// **This is not a perceptual metric.** `cjxl -d` targets butteraugli, which
/// this repository does not implement; PSNR and butteraugli disagree, sometimes
/// sharply, about which of two images looks better. Use this to compare
/// encoders along a rate/distortion *curve*, never to claim one encoder's
/// output is perceptually better at a single operating point.
#[must_use]
pub fn psnr(a: &Image, b: &Image) -> Option<f64> {
    let err = rmse(a, b)?;
    if err == 0.0 {
        return Some(f64::INFINITY);
    }
    let peak = f64::from(a.max_value);
    Some(20.0 * (peak / err).log10())
}

/// Why a PPM could not be parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PpmError {
    /// The file does not start with the `P6` magic number.
    BadMagic,
    /// The header ended before all required fields were read.
    TruncatedHeader,
    /// A header field was not a decimal number.
    BadHeaderField,
    /// `maxval` was zero or larger than 65535.
    UnsupportedMaxValue(u32),
    /// `width * height * channels` does not fit in a `usize`.
    DimensionsOverflow,
    /// The pixel payload is shorter than the header promises.
    TruncatedPixels {
        /// Bytes the header calls for.
        expected: usize,
        /// Bytes actually present.
        found: usize,
    },
}

impl fmt::Display for PpmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic => f.write_str("not a binary PPM (expected the `P6` magic number)"),
            Self::TruncatedHeader => f.write_str("PPM header ended early"),
            Self::BadHeaderField => f.write_str("PPM header field is not a decimal number"),
            Self::UnsupportedMaxValue(v) => write!(f, "unsupported PPM maxval {v}"),
            Self::DimensionsOverflow => f.write_str("PPM dimensions overflow a usize"),
            Self::TruncatedPixels { expected, found } => {
                write!(
                    f,
                    "PPM pixel data truncated: expected {expected} bytes, found {found}"
                )
            }
        }
    }
}

impl std::error::Error for PpmError {}

/// A forward-only reader over PPM header bytes.
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    /// Skip whitespace and `#`-to-end-of-line comments.
    fn skip_blanks(&mut self) {
        while let Some(byte) = self.peek() {
            if byte.is_ascii_whitespace() {
                self.pos += 1;
            } else if byte == b'#' {
                while let Some(b) = self.peek() {
                    self.pos += 1;
                    if b == b'\n' {
                        break;
                    }
                }
            } else {
                break;
            }
        }
    }

    /// Read the next whitespace-delimited token.
    fn token(&mut self) -> Result<&'a [u8], PpmError> {
        self.skip_blanks();
        let start = self.pos;
        while self.peek().is_some_and(|b| !b.is_ascii_whitespace()) {
            self.pos += 1;
        }
        self.bytes
            .get(start..self.pos)
            .filter(|token| !token.is_empty())
            .ok_or(PpmError::TruncatedHeader)
    }

    /// Read the next token as a decimal `u32`.
    fn number(&mut self) -> Result<u32, PpmError> {
        let token = self.token()?;
        let text = std::str::from_utf8(token).map_err(|_| PpmError::BadHeaderField)?;
        text.parse::<u32>().map_err(|_| PpmError::BadHeaderField)
    }

    /// Consume the single whitespace byte that terminates the header.
    fn consume_single_whitespace(&mut self) -> Result<(), PpmError> {
        match self.peek() {
            Some(b) if b.is_ascii_whitespace() => {
                self.pos += 1;
                Ok(())
            }
            _ => Err(PpmError::TruncatedHeader),
        }
    }

    /// Take exactly `n` payload bytes.
    fn take(&mut self, n: usize) -> Result<&'a [u8], PpmError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or(PpmError::DimensionsOverflow)?;
        let found = self.bytes.len().saturating_sub(self.pos);
        let slice = self
            .bytes
            .get(self.pos..end)
            .ok_or(PpmError::TruncatedPixels { expected: n, found })?;
        self.pos = end;
        Ok(slice)
    }
}

/// A decoded image held as `f32` samples on the nominal `[0, 1]` scale.
///
/// This is the comparison surface 18181-3 §4.2 specifies: samples are never
/// clipped or rescaled, so a value outside `[0, 1]` (out-of-gamut, or an
/// intermediate representation before display mapping) is compared exactly
/// as read.
///
/// Layout mirrors the NPY array shape `(frames, height, width, channels)`
/// (18181-3 §4.1.2): [`Self::samples`] is `frames * height * width *
/// channels` long, C-order (the `channels` axis fastest, then `width`, then
/// `height`, then `frames`).
#[derive(Debug, Clone, PartialEq)]
pub struct FloatImage {
    /// Number of frames (the NPY array's first axis).
    pub frames: u32,
    /// Height in pixels.
    pub height: u32,
    /// Width in pixels.
    pub width: u32,
    /// Number of channels (samples per pixel).
    pub channels: u32,
    /// Samples, C-order, `frames * height * width * channels` long.
    pub samples: Vec<f32>,
}

impl FloatImage {
    /// Total sample count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether the image has no samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Whether two images have identical dimensions, frame count and channel
    /// count — condition 1 of 18181-3 §4.2.
    #[must_use]
    pub fn same_shape(&self, other: &Self) -> bool {
        self.frames == other.frames
            && self.height == other.height
            && self.width == other.width
            && self.channels == other.channels
    }

    /// Parse the NPY subset `djxl --output_format npy` (and the conformance
    /// corpus's `reference_image.npy`) emits: version 1.0, little-endian
    /// `f32` (`<f4`), C-order (`fortran_order: False`), a 4-D shape
    /// `(frames, height, width, channels)`.
    ///
    /// This is a hand-rolled parser for exactly that subset, not a general
    /// NPY reader: 18181-3 §4.1.2 itself only specifies this subset (its
    /// `NOTE` says the full NPY format is a superset).
    ///
    /// # Errors
    ///
    /// Returns [`NpyError`] for a bad magic number, an unsupported version,
    /// a malformed or unsupported header (wrong dtype, Fortran order, or a
    /// shape that is not rank-4), or a truncated data payload.
    pub fn from_npy(bytes: &[u8]) -> Result<Self, NpyError> {
        const MAGIC: &[u8; 6] = b"\x93NUMPY";

        let magic = bytes.get(0..6).ok_or(NpyError::TruncatedHeader)?;
        if magic != MAGIC {
            return Err(NpyError::BadMagic);
        }
        let version = bytes.get(6..8).ok_or(NpyError::TruncatedHeader)?;
        // Only version 1.0 is handled: its 2-byte header-length field is the
        // layout 18181-3 §4.1.2 documents. Versions >= 2.0 use a 4-byte
        // length instead and are out of scope for the subset we read.
        if version != [1, 0] {
            return Err(NpyError::UnsupportedVersion {
                major: *version.first().unwrap_or(&0),
                minor: *version.get(1).unwrap_or(&0),
            });
        }
        let len_bytes: [u8; 2] = bytes
            .get(8..10)
            .and_then(|s| s.try_into().ok())
            .ok_or(NpyError::TruncatedHeader)?;
        let header_len = usize::from(u16::from_le_bytes(len_bytes));
        let header_end = 10_usize
            .checked_add(header_len)
            .ok_or(NpyError::DimensionsOverflow)?;
        let header_bytes = bytes.get(10..header_end).ok_or(NpyError::TruncatedHeader)?;
        let header = std::str::from_utf8(header_bytes).map_err(|_| NpyError::HeaderNotUtf8)?;

        let descr = header_field(header, "'descr'").ok_or(NpyError::MissingField("descr"))?;
        let descr = quoted_string(descr).ok_or(NpyError::MissingField("descr"))?;
        if descr != "<f4" {
            return Err(NpyError::UnsupportedDtype(descr.to_owned()));
        }

        let fortran_order = header_field(header, "'fortran_order'")
            .ok_or(NpyError::MissingField("fortran_order"))?;
        let fortran_order =
            bare_token(fortran_order).ok_or(NpyError::MissingField("fortran_order"))?;
        if fortran_order != "False" {
            return Err(NpyError::NotCOrder);
        }

        let shape = header_field(header, "'shape'").ok_or(NpyError::MissingField("shape"))?;
        let dims = shape_tuple(shape).ok_or(NpyError::BadShape)?;
        let [frames, height, width, channels] =
            <[usize; 4]>::try_from(dims.as_slice()).map_err(|_| NpyError::BadShape)?;
        let to_u32 = |v: usize| u32::try_from(v).map_err(|_| NpyError::DimensionsOverflow);
        let (frames, height, width, channels) = (
            to_u32(frames)?,
            to_u32(height)?,
            to_u32(width)?,
            to_u32(channels)?,
        );

        let element_count = dims
            .iter()
            .copied()
            .try_fold(1_usize, |acc, d| acc.checked_mul(d))
            .ok_or(NpyError::DimensionsOverflow)?;
        let needed = element_count
            .checked_mul(4)
            .ok_or(NpyError::DimensionsOverflow)?;
        let data = bytes.get(header_end..).unwrap_or(&[]);
        let payload = data.get(..needed).ok_or(NpyError::TruncatedData {
            expected: needed,
            found: data.len(),
        })?;

        let samples = payload
            .chunks_exact(4)
            .map(|w| match w {
                [a, b, c, d] => f32::from_le_bytes([*a, *b, *c, *d]),
                _ => unreachable!("chunks_exact(4) yields quads"),
            })
            .collect();

        Ok(Self {
            frames,
            height,
            width,
            channels,
            samples,
        })
    }
}

/// Locate a `'key': value` header field and return the raw text after the
/// colon, trimmed of leading whitespace, up to (but not including) the next
/// top-level comma or the end of the header.
///
/// "Top-level" skips commas nested inside `(...)` or `'...'`, which matters
/// for `shape`'s tuple. This is a scanner over the one dict-literal shape
/// NPY headers use, not a general Python-literal parser.
fn header_field<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let after_key = header.split_once(key)?.1;
    let after_colon = after_key.trim_start().strip_prefix(':')?.trim_start();
    let mut depth = 0_i32;
    let mut quoted = false;
    for (i, ch) in after_colon.char_indices() {
        match ch {
            '\'' => quoted = !quoted,
            '(' if !quoted => depth += 1,
            ')' if !quoted => depth -= 1,
            ',' if !quoted && depth == 0 => return Some(after_colon.get(..i).unwrap_or("")),
            _ => {}
        }
    }
    Some(after_colon)
}

/// Parse `'text'` (single-quoted) into `text`.
fn quoted_string(field: &str) -> Option<&str> {
    let field = field.trim();
    field.strip_prefix('\'')?.strip_suffix('\'')
}

/// Parse a bare token (`False`, `True`), trimmed of surrounding whitespace.
fn bare_token(field: &str) -> Option<&str> {
    let token = field.trim();
    (!token.is_empty()).then_some(token)
}

/// Parse `(a, b, c, d)` into `[a, b, c, d, ...]`. Handles the trailing comma
/// a single-element Python tuple would have (`(5,)`); the shapes this reader
/// accepts are always rank 4, so that case never actually matches, but the
/// parser stays correct for it rather than silently mis-splitting.
fn shape_tuple(field: &str) -> Option<Vec<usize>> {
    let field = field.trim();
    let inner = field.strip_prefix('(')?.strip_suffix(')')?;
    inner
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().ok())
        .collect()
}

/// Why an NPY file could not be parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NpyError {
    /// The file does not start with the `\x93NUMPY` magic number.
    BadMagic,
    /// The header ended, or the file itself ended, before all fixed fields
    /// were read.
    TruncatedHeader,
    /// The header is not valid UTF-8 (it should be plain ASCII).
    HeaderNotUtf8,
    /// The header's version is not `1.0`, the only version this subset
    /// reader understands.
    UnsupportedVersion {
        /// The major version byte read.
        major: u8,
        /// The minor version byte read.
        minor: u8,
    },
    /// A required header key (`descr`, `fortran_order`, `shape`) was not
    /// found in the header string.
    MissingField(&'static str),
    /// `descr` was present but was not little-endian `f32` (`<f4`).
    UnsupportedDtype(String),
    /// `fortran_order` was `True`; this reader only understands C order.
    NotCOrder,
    /// `shape` was present but did not parse as a 4-element integer tuple.
    BadShape,
    /// A dimension, or a product of dimensions, does not fit the type used
    /// to hold it.
    DimensionsOverflow,
    /// The data payload is shorter than the header's shape promises.
    TruncatedData {
        /// Bytes the shape calls for.
        expected: usize,
        /// Bytes actually present after the header.
        found: usize,
    },
}

impl fmt::Display for NpyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic => f.write_str("not an NPY file (bad magic number)"),
            Self::TruncatedHeader => f.write_str("NPY header ended early"),
            Self::HeaderNotUtf8 => f.write_str("NPY header is not valid UTF-8"),
            Self::UnsupportedVersion { major, minor } => {
                write!(
                    f,
                    "unsupported NPY version {major}.{minor} (only 1.0 is read)"
                )
            }
            Self::MissingField(name) => write!(f, "NPY header is missing '{name}'"),
            Self::UnsupportedDtype(descr) => {
                write!(f, "unsupported NPY dtype {descr} (only '<f4' is read)")
            }
            Self::NotCOrder => f.write_str("NPY array is Fortran-order, not C-order"),
            Self::BadShape => f.write_str("NPY 'shape' is not a 4-element integer tuple"),
            Self::DimensionsOverflow => f.write_str("NPY dimensions overflow"),
            Self::TruncatedData { expected, found } => {
                write!(
                    f,
                    "NPY data truncated: expected {expected} bytes, found {found}"
                )
            }
        }
    }
}

impl std::error::Error for NpyError {}

/// The result of comparing two [`FloatImage`]s under 18181-3 §4.2.
#[derive(Debug, Clone, PartialEq)]
pub struct Similarity {
    /// The largest `|D - R|` over every sample of every channel — the
    /// "peak error" of condition 2.
    pub peak_error: f32,
    /// The largest `|D - R|` within each channel, in channel order.
    pub channel_peak: Vec<f32>,
    /// The root-mean-square error within each channel, in channel order —
    /// condition 3's per-channel RMSE.
    pub channel_rmse: Vec<f32>,
}

impl Similarity {
    /// Whether this comparison meets a Part 3 error class: the global peak
    /// error is at most `peak_threshold`, and every channel's RMSE is at
    /// most `rmse_threshold`.
    #[must_use]
    pub fn conforms(&self, peak_threshold: f32, rmse_threshold: f32) -> bool {
        self.peak_error <= peak_threshold
            && self.channel_rmse.iter().all(|&rmse| rmse <= rmse_threshold)
    }
}

/// Compare two [`FloatImage`]s under 18181-3 §4.2's three conditions.
///
/// Returns `None` on condition 1's failure (dimensions, frame count or
/// channel count differ) — that is not a similarity score, it is a
/// comparison that cannot be made. Samples are compared exactly as stored,
/// with **no clipping**, matching §4.2's "nominal values in the interval
/// `[0, 1]`, no clipping is to be applied to values outside this range".
///
/// RMSE is computed as `sqrt(mean((D - R)^2))` over each channel's samples
/// (the standard definition the clause's name states; its body's "root of
/// the sum" is read as elliptical for "root of the mean", since an
/// unnormalised sum could never stay under corpus thresholds like `1e-5` on
/// any real image).
///
/// ```
/// use jpxl_conformance::{FloatImage, similarity};
///
/// let img = |v: f32| FloatImage {
///     frames: 1,
///     height: 1,
///     width: 1,
///     channels: 1,
///     samples: vec![v],
/// };
/// let report = similarity(&img(0.5), &img(0.5)).expect("same shape");
/// assert_eq!(report.peak_error, 0.0);
/// assert!(report.conforms(0.0, 0.0));
/// ```
#[must_use]
pub fn similarity(a: &FloatImage, b: &FloatImage) -> Option<Similarity> {
    if !a.same_shape(b) || a.samples.len() != b.samples.len() {
        return None;
    }
    let channels = usize::try_from(a.channels).ok()?;
    if channels == 0 || a.samples.is_empty() {
        return Some(Similarity {
            peak_error: 0.0,
            channel_peak: vec![0.0; channels],
            channel_rmse: vec![0.0; channels],
        });
    }

    let mut channel_peak = vec![0.0_f32; channels];
    let mut channel_sq_sum = vec![0.0_f64; channels];
    let mut channel_count = vec![0_u64; channels];
    let mut peak_error = 0.0_f32;

    for (index, (&d, &r)) in a.samples.iter().zip(&b.samples).enumerate() {
        let diff = (d - r).abs();
        if diff > peak_error {
            peak_error = diff;
        }
        let Some(slot) = channel_peak.get_mut(index % channels) else {
            continue;
        };
        if diff > *slot {
            *slot = diff;
        }
        if let Some(sq) = channel_sq_sum.get_mut(index % channels) {
            *sq += f64::from(diff) * f64::from(diff);
        }
        if let Some(count) = channel_count.get_mut(index % channels) {
            *count += 1;
        }
    }

    let channel_rmse = channel_sq_sum
        .iter()
        .zip(&channel_count)
        .map(|(&sq, &count)| {
            if count == 0 {
                0.0
            } else {
                // The accumulator is f64 so the running sum of squares does
                // not itself lose precision across a large image; the final
                // narrowing back to f32 is intentional; f32 is the sample
                // precision `similarity`'s inputs (and Part 3 thresholds)
                // are already stated in.
                #[allow(clippy::cast_possible_truncation)]
                let rmse = (sq / count as f64).sqrt() as f32;
                rmse
            }
        })
        .collect();

    Some(Similarity {
        peak_error,
        channel_peak,
        channel_rmse,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an 8-bit P6 with the given interleaved RGB bytes.
    fn ppm8(w: u32, h: u32, rgb: &[u8]) -> Vec<u8> {
        let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
        out.extend_from_slice(rgb);
        out
    }

    /// Build a 16-bit P6 with the given interleaved RGB samples.
    fn ppm16(w: u32, h: u32, rgb: &[u16]) -> Vec<u8> {
        let mut out = format!("P6\n{w} {h}\n65535\n").into_bytes();
        for sample in rgb {
            out.extend_from_slice(&sample.to_be_bytes());
        }
        out
    }

    #[test]
    fn parses_a_one_pixel_8bit_image() {
        let img = Image::from_ppm(&ppm8(1, 1, &[1, 2, 3])).expect("valid PPM");
        assert_eq!((img.w, img.h, img.channels), (1, 1, 3));
        assert_eq!(img.max_value, 255);
        assert_eq!(img.samples, vec![1, 2, 3]);
        assert_eq!(img.len(), 3);
        assert!(!img.is_empty());
    }

    #[test]
    fn parses_a_16bit_image_big_endian() {
        let img = Image::from_ppm(&ppm16(2, 1, &[0, 1, 256, 65535, 4096, 7])).expect("valid PPM");
        assert_eq!(img.max_value, 65535);
        assert_eq!(img.samples, vec![0, 1, 256, 65535, 4096, 7]);
    }

    #[test]
    fn tolerates_comments_and_extra_whitespace() {
        let mut bytes = b"P6\n# written by a test\n  2\t1\n# another comment\n255\n".to_vec();
        bytes.extend_from_slice(&[9, 8, 7, 6, 5, 4]);
        let img = Image::from_ppm(&bytes).expect("valid PPM");
        assert_eq!((img.w, img.h), (2, 1));
        assert_eq!(img.samples, vec![9, 8, 7, 6, 5, 4]);
    }

    #[test]
    fn rejects_wrong_magic() {
        assert_eq!(
            Image::from_ppm(b"P5\n1 1\n255\n\0"),
            Err(PpmError::BadMagic)
        );
        assert_eq!(Image::from_ppm(b""), Err(PpmError::TruncatedHeader));
    }

    #[test]
    fn rejects_truncated_pixels() {
        let mut bytes = ppm8(2, 2, &[0; 12]);
        bytes.truncate(bytes.len() - 1);
        let err = Image::from_ppm(&bytes).expect_err("payload is one byte short");
        assert!(matches!(
            err,
            PpmError::TruncatedPixels { expected: 12, .. }
        ));
    }

    #[test]
    fn rejects_bad_maxval() {
        assert_eq!(
            Image::from_ppm(b"P6\n1 1\n0\n\0\0\0"),
            Err(PpmError::UnsupportedMaxValue(0))
        );
        assert_eq!(
            Image::from_ppm(b"P6\n1 1\n70000\n"),
            Err(PpmError::UnsupportedMaxValue(70000))
        );
        assert_eq!(
            Image::from_ppm(b"P6\n1 x\n255\n"),
            Err(PpmError::BadHeaderField)
        );
    }

    #[test]
    fn identical_images_have_zero_error() {
        let a = Image::from_ppm(&ppm8(2, 1, &[1, 2, 3, 4, 5, 6])).expect("valid PPM");
        let b = a.clone();
        assert_eq!(max_abs_error(&a, &b), Some(0));
        assert_eq!(peak_error_per_channel(&a, &b), Some(vec![0, 0, 0]));
    }

    #[test]
    fn max_abs_error_finds_the_worst_sample() {
        let a = Image::from_ppm(&ppm8(2, 1, &[0, 0, 0, 0, 0, 0])).expect("valid PPM");
        let b = Image::from_ppm(&ppm8(2, 1, &[1, 0, 0, 0, 40, 0])).expect("valid PPM");
        assert_eq!(max_abs_error(&a, &b), Some(40));
        // Channel 1 (green) holds the 40; channel 0 holds the 1.
        assert_eq!(peak_error_per_channel(&a, &b), Some(vec![1, 40, 0]));
    }

    #[test]
    fn error_is_symmetric_and_unsigned() {
        let a = Image::from_ppm(&ppm8(1, 1, &[200, 0, 0])).expect("valid PPM");
        let b = Image::from_ppm(&ppm8(1, 1, &[5, 0, 0])).expect("valid PPM");
        assert_eq!(max_abs_error(&a, &b), Some(195));
        assert_eq!(max_abs_error(&b, &a), Some(195));
    }

    #[test]
    fn shape_mismatch_yields_none() {
        let a = Image::from_ppm(&ppm8(2, 1, &[0; 6])).expect("valid PPM");
        let b = Image::from_ppm(&ppm8(1, 2, &[0; 6])).expect("valid PPM");
        let deep = Image::from_ppm(&ppm16(2, 1, &[0; 6])).expect("valid PPM");
        assert_eq!(max_abs_error(&a, &b), None, "dimensions differ");
        assert_eq!(max_abs_error(&a, &deep), None, "bit depth differs");
        assert_eq!(peak_error_per_channel(&a, &deep), None);
    }

    #[test]
    fn sixteen_bit_differences_exceed_eight_bit_range() {
        let a = Image::from_ppm(&ppm16(1, 1, &[0, 0, 0])).expect("valid PPM");
        let b = Image::from_ppm(&ppm16(1, 1, &[65535, 0, 0])).expect("valid PPM");
        assert_eq!(max_abs_error(&a, &b), Some(65535));
    }

    /// Build an NPY file in exactly the layout `djxl --output_format npy`
    /// was verified to emit (see [`FloatImage::from_npy`]'s doc comment):
    /// version 1.0, a 2-byte little-endian header length, the header dict
    /// as ASCII ending in `\n`, then raw little-endian `f32` data. No extra
    /// alignment padding is added — the parser must not require any, since
    /// real NPY writers pad to different boundaries.
    fn npy_f32(shape: [usize; 4], samples: &[f32]) -> Vec<u8> {
        let [frames, height, width, channels] = shape;
        let header = format!(
            "{{'descr': '<f4', 'fortran_order': False, 'shape': ({frames}, {height}, {width}, {channels}), }}\n"
        );
        let len = u16::try_from(header.len()).expect("test header fits in u16");
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend_from_slice(&len.to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn parses_the_verified_djxl_npy_layout() {
        // Mirrors the header djxl v0.13.0 was observed to write for a
        // (1, 64, 64, 3) decode: `{'descr': '<f4', 'fortran_order': False,
        // 'shape': (1, 64, 64, 3), }\n`, magic `\x93NUMPY`, version `01 00`.
        // Shrunk to (1, 1, 2, 3) here so the test data is legible.
        let bytes = npy_f32([1, 1, 2, 3], &[0.0, 0.25, 0.5, 0.75, 1.0, -1.0]);
        let img = FloatImage::from_npy(&bytes).expect("valid NPY");
        assert_eq!(
            (img.frames, img.height, img.width, img.channels),
            (1, 1, 2, 3)
        );
        assert_eq!(img.samples, vec![0.0, 0.25, 0.5, 0.75, 1.0, -1.0]);
        assert_eq!(img.len(), 6);
        assert!(!img.is_empty());
    }

    #[test]
    fn npy_rejects_bad_magic() {
        let mut bytes = npy_f32([1, 1, 1, 1], &[0.0]);
        if let Some(first) = bytes.get_mut(0) {
            *first = 0;
        }
        assert_eq!(FloatImage::from_npy(&bytes), Err(NpyError::BadMagic));
    }

    #[test]
    fn npy_rejects_fortran_order() {
        let header = "{'descr': '<f4', 'fortran_order': True, 'shape': (1, 1, 1, 1), }\n";
        let len = u16::try_from(header.len()).expect("fits");
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend_from_slice(&len.to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&0.0_f32.to_le_bytes());
        assert_eq!(FloatImage::from_npy(&bytes), Err(NpyError::NotCOrder));
    }

    #[test]
    fn npy_rejects_non_f32_dtype() {
        let header = "{'descr': '<u2', 'fortran_order': False, 'shape': (1, 1, 1, 1), }\n";
        let len = u16::try_from(header.len()).expect("fits");
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend_from_slice(&len.to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0, 0]);
        assert_eq!(
            FloatImage::from_npy(&bytes),
            Err(NpyError::UnsupportedDtype("<u2".to_owned()))
        );
    }

    #[test]
    fn npy_rejects_truncated_data() {
        let mut bytes = npy_f32([1, 1, 1, 2], &[0.0, 0.0]);
        bytes.truncate(bytes.len() - 1);
        let err = FloatImage::from_npy(&bytes).expect_err("payload is one byte short");
        assert!(matches!(err, NpyError::TruncatedData { expected: 8, .. }));
    }

    #[test]
    fn npy_rejects_wrong_rank_shape() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 1, 1), }\n";
        let len = u16::try_from(header.len()).expect("fits");
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend_from_slice(&len.to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        assert_eq!(FloatImage::from_npy(&bytes), Err(NpyError::BadShape));
    }

    /// A 2x1 grey PPM whose two pixels differ from the reference by 0 and 4 in
    /// every channel: RMSE is `sqrt((0*3 + 16*3)/6) = sqrt(8)`, and PSNR is
    /// `20*log10(255/sqrt(8))`. Both hand-computed, so this pins the formula
    /// rather than whatever the code happens to do.
    #[test]
    fn psnr_and_rmse_match_hand_computed_values() {
        let ppm = |a: u8, b: u8| {
            let mut bytes = b"P6\n2 1\n255\n".to_vec();
            bytes.extend_from_slice(&[a, a, a, b, b, b]);
            Image::from_ppm(&bytes).expect("valid PPM")
        };
        let reference = ppm(100, 100);
        let decoded = ppm(100, 104);

        let err = rmse(&reference, &decoded).expect("same shape");
        assert!(
            (err - 8.0_f64.sqrt()).abs() < 1e-12,
            "rmse was {err}, expected sqrt(8)"
        );

        let db = psnr(&reference, &decoded).expect("same shape");
        let expected = 20.0 * (255.0 / 8.0_f64.sqrt()).log10();
        assert!(
            (db - expected).abs() < 1e-12,
            "psnr was {db}, expected {expected}"
        );
    }

    /// Identical images have zero error, so PSNR is infinite rather than a
    /// large finite number or a division-by-zero NaN.
    #[test]
    fn identical_images_have_zero_rmse_and_infinite_psnr() {
        let bytes = b"P6\n1 1\n255\n\x10\x20\x30".to_vec();
        let image = Image::from_ppm(&bytes).expect("valid PPM");
        assert_eq!(rmse(&image, &image), Some(0.0));
        assert_eq!(psnr(&image, &image), Some(f64::INFINITY));
    }

    /// Shape mismatch is `None`, not a panic and not a meaningless number —
    /// the same contract `max_abs_error` and `peak_error_per_channel` keep.
    #[test]
    fn mismatched_shapes_have_no_rmse_or_psnr() {
        let one = Image::from_ppm(b"P6\n1 1\n255\n\x00\x00\x00").expect("valid");
        let two = Image::from_ppm(b"P6\n2 1\n255\n\x00\x00\x00\x00\x00\x00").expect("valid");
        assert_eq!(rmse(&one, &two), None);
        assert_eq!(psnr(&one, &two), None);
    }

    /// Hand-computed: reference is all zero, decoded is `[0, 0, 0, 1]` in a
    /// single channel. The diffs are `[0, 0, 0, 1]`, so peak error is `1`
    /// and RMSE is `sqrt((0+0+0+1)/4) = sqrt(0.25) = 0.5` exactly (both
    /// values are exact in binary floating point, so this needs no
    /// tolerance).
    #[test]
    fn similarity_matches_a_hand_computed_rmse() {
        let reference = FloatImage {
            frames: 1,
            height: 1,
            width: 4,
            channels: 1,
            samples: vec![0.0, 0.0, 0.0, 0.0],
        };
        let decoded = FloatImage {
            samples: vec![0.0, 0.0, 0.0, 1.0],
            ..reference.clone()
        };
        let report = similarity(&decoded, &reference).expect("same shape");
        assert_eq!(report.peak_error, 1.0);
        assert_eq!(report.channel_peak, vec![1.0]);
        assert_eq!(report.channel_rmse, vec![0.5]);
        assert!(report.conforms(1.0, 0.5));
        assert!(!report.conforms(0.999, 0.5), "peak exceeds 0.999");
        assert!(!report.conforms(1.0, 0.499), "rmse exceeds 0.499");
    }

    /// Per-channel independence: channel 0 differs by exactly 1 in every
    /// pixel, channel 1 never differs. Condition 3 (§4.2) grades RMSE **per
    /// channel**, so a noisy channel must not be washed out by a clean one.
    #[test]
    fn similarity_is_computed_independently_per_channel() {
        let reference = FloatImage {
            frames: 1,
            height: 1,
            width: 2,
            channels: 2,
            samples: vec![0.0, 0.0, 0.0, 0.0],
        };
        let decoded = FloatImage {
            samples: vec![1.0, 0.0, 1.0, 0.0],
            ..reference.clone()
        };
        let report = similarity(&decoded, &reference).expect("same shape");
        assert_eq!(report.channel_peak, vec![1.0, 0.0]);
        assert_eq!(report.channel_rmse, vec![1.0, 0.0]);
        assert_eq!(report.peak_error, 1.0);
    }

    #[test]
    fn similarity_no_clipping_of_out_of_range_samples() {
        // 18181-3 4.2: nominal [0, 1], no clipping applied outside that
        // range. A sample of -1.0 compared against 1.0 must report a peak
        // of 2.0, not something clamped into [0, 1] first.
        let a = FloatImage {
            frames: 1,
            height: 1,
            width: 1,
            channels: 1,
            samples: vec![-1.0],
        };
        let b = FloatImage {
            samples: vec![1.0],
            ..a.clone()
        };
        let report = similarity(&a, &b).expect("same shape");
        assert_eq!(report.peak_error, 2.0);
    }

    #[test]
    fn similarity_shape_mismatch_yields_none() {
        let a = FloatImage {
            frames: 1,
            height: 1,
            width: 2,
            channels: 1,
            samples: vec![0.0, 0.0],
        };
        let b = FloatImage {
            width: 1,
            samples: vec![0.0],
            ..a.clone()
        };
        assert_eq!(similarity(&a, &b), None);
    }

    #[test]
    fn identical_float_images_have_zero_error() {
        let img = FloatImage {
            frames: 1,
            height: 2,
            width: 2,
            channels: 3,
            samples: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.0, 0.1, 0.2],
        };
        let report = similarity(&img, &img).expect("same shape");
        assert_eq!(report.peak_error, 0.0);
        assert_eq!(report.channel_rmse, vec![0.0, 0.0, 0.0]);
        assert!(report.conforms(0.0, 0.0));
    }
}
