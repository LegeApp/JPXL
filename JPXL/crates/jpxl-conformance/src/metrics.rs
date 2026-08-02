//! A minimal PPM reader and pixel-difference metrics.
//!
//! Oracles emit binary PPM (`P6`); this module reads just enough of that format
//! to compare two decodes. It is deliberately **harness-local**: [`Image`] is a
//! comparison buffer, not a codec type, and nothing in `jpxl-decode` should
//! grow a dependency on it.
//!
//! Samples are always widened to `u16`. An 8-bit file keeps its 0..=255 range
//! (samples are *not* rescaled), so comparing an 8-bit PPM against a 16-bit one
//! is meaningless — [`max_abs_error`] refuses mismatched [`Image::max_value`]s
//! for exactly that reason.

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
}
