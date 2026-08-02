//! Image headers: the `Headers` bundle of 18181-1 Annex D.
//!
//! ```text
//! Table D.1 — Headers bundle
//! condition   type           name        subclause
//!             u(16)          signature
//!             SizeHeader     size        D.2
//!             ImageMetadata  metadata    D.3
//! ```
//!
//! Bundles are read exactly as B.1.1 describes: a nested bundle behaves as if
//! its rows were spliced into the parent in place of the row naming it, so the
//! whole of Annex D is one depth-first traversal with no alignment or padding
//! between bundles. Nothing here calls `ZeroPadToByte()`; the header ends
//! mid-byte and the next stage continues from that bit.

pub mod animation;
pub mod bit_depth;
pub mod colour;
pub mod enums;
pub mod extensions;
pub mod extra_channels;
pub mod metadata;
pub mod opsin;
pub mod size;

use jpxl_bitstream::BitReader;
use jpxl_core::limits::{AllocGuard, Limits};

use crate::error::Result;
use crate::signature::read_signature;

pub use metadata::{ImageMetadata, Orientation};
pub use size::{PreviewHeader, SizeHeader};

/// The complete `Headers` bundle (18181-1 D.1).
#[derive(Debug, Clone, PartialEq)]
pub struct ImageHeaders {
    /// Image dimensions.
    pub size: SizeHeader,
    /// Everything else that applies to all frames.
    pub metadata: ImageMetadata,
}

impl ImageHeaders {
    /// Image width in pixels, before any orientation transform.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.size.width().get()
    }

    /// Image height in pixels, before any orientation transform.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.size.height().get()
    }

    /// Displayed dimensions after applying `metadata.orientation`.
    #[must_use]
    pub fn oriented_dimensions(&self) -> (u32, u32) {
        if self.metadata.orientation.swaps_axes() {
            (self.height(), self.width())
        } else {
            (self.width(), self.height())
        }
    }
}

/// Reads the signature, `SizeHeader` and `ImageMetadata` from the start of a
/// codestream (18181-1 D.1).
///
/// The reader is left positioned immediately after `ImageMetadata`, which is
/// generally not a byte boundary.
///
/// # Errors
///
/// [`DecodeError::InvalidSignature`](crate::DecodeError::InvalidSignature) if
/// the codestream does not begin with `FF 0A`, or any error from a nested
/// bundle.
///
/// # Examples
///
/// ```
/// use jpxl_bitstream::BitReader;
/// use jpxl_core::limits::Limits;
/// use jpxl_decode::headers::decode_image_headers;
///
/// // FF 0A, then a div8 8x8 SizeHeader and an all-default ImageMetadata.
/// let data = [0xFF, 0x0A, 0b0000_0001, 0b1100_0000];
/// let mut reader = BitReader::new(&data);
/// let headers = decode_image_headers(&mut reader, &Limits::default())?;
///
/// assert_eq!(headers.width(), 8);
/// assert_eq!(headers.height(), 8);
/// assert!(headers.metadata.all_default);
/// # Ok::<(), jpxl_decode::DecodeError>(())
/// ```
pub fn decode_image_headers(reader: &mut BitReader<'_>, limits: &Limits) -> Result<ImageHeaders> {
    let mut guard = AllocGuard::new(limits);
    decode_image_headers_metered(reader, limits, &mut guard)
}

/// Like [`decode_image_headers`], but charges allocations to a caller-supplied
/// guard so a whole decode shares one budget.
///
/// # Errors
///
/// As [`decode_image_headers`].
pub fn decode_image_headers_metered(
    reader: &mut BitReader<'_>,
    limits: &Limits,
    guard: &mut AllocGuard,
) -> Result<ImageHeaders> {
    read_signature(reader)?;
    let size = size::read_size_header(reader, limits)?;
    let metadata = metadata::read_image_metadata(reader, limits, guard)?;
    Ok(ImageHeaders { size, metadata })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::DecodeError;
    use crate::testsupport::BitWriter;

    /// Builds the smallest legal header: signature, 8x8 div8 size, defaults.
    fn minimal_headers() -> Vec<u8> {
        let mut w = BitWriter::new();
        w.u(16, 0x0AFF);
        // SizeHeader: div8 = 1, h_div8 = 0 => 8; ratio = 0; w_div8 = 0 => 8.
        w.bool(true).u(5, 0).u(3, 0).u(5, 0);
        // ImageMetadata: all_default = 1, default_m = 1.
        w.bool(true).bool(true);
        w.finish_padded(1)
    }

    #[test]
    fn decodes_the_minimal_header() {
        let data = minimal_headers();
        let mut r = BitReader::new(&data);
        let headers = decode_image_headers(&mut r, &Limits::default()).expect("valid");

        assert_eq!(headers.width(), 8);
        assert_eq!(headers.height(), 8);
        assert!(headers.metadata.all_default);
        assert!(headers.metadata.default_m);
        assert_eq!(headers.oriented_dimensions(), (8, 8));

        // 16 signature + 14 size + 2 metadata = 32 bits.
        assert_eq!(r.total_bits_read(), 32);
    }

    #[test]
    fn rejects_a_bad_signature() {
        let mut data = minimal_headers();
        if let Some(byte) = data.first_mut() {
            *byte = 0x00;
        }
        let mut r = BitReader::new(&data);
        let err = decode_image_headers(&mut r, &Limits::default()).expect_err("bad signature");
        assert!(matches!(err, DecodeError::InvalidSignature { .. }));
    }

    #[test]
    fn orientation_swaps_reported_dimensions() {
        let mut w = BitWriter::new();
        w.u(16, 0x0AFF);
        // 1920x1080 via !div8 with a 16:9 ratio.
        w.bool(false).u32_field(1, 13, 1079).u(3, 5);
        // Metadata with extra_fields and orientation 6 (rotate 90 cw).
        w.bool(false)
            .bool(true)
            .u(3, 5) // orientation = 6
            .bool(false)
            .bool(false)
            .bool(false)
            .bool(false)
            .u32_field(0, 0, 0)
            .bool(true)
            .u32_field(0, 0, 0)
            .bool(true)
            .bool(true) // colour all_default
            .bool(true) // tone_mapping all_default
            .u32_field(0, 0, 0)
            .bool(true);
        let data = w.finish_padded(1);

        let mut r = BitReader::new(&data);
        let headers = decode_image_headers(&mut r, &Limits::default()).expect("valid");

        assert_eq!(headers.width(), 1920);
        assert_eq!(headers.height(), 1080);
        assert_eq!(headers.metadata.orientation, Orientation::Rotate90Cw);
        assert_eq!(headers.oriented_dimensions(), (1080, 1920));
    }

    #[test]
    fn limits_apply_through_the_top_level_entry_point() {
        let data = minimal_headers();
        let tight = Limits {
            max_pixels: 10,
            ..Limits::default()
        };
        let mut r = BitReader::new(&data);
        assert!(decode_image_headers(&mut r, &tight).is_err());
    }

    #[test]
    fn a_shared_guard_accumulates_across_calls() {
        let data = minimal_headers();
        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);

        let mut r = BitReader::new(&data);
        decode_image_headers_metered(&mut r, &limits, &mut guard).expect("valid");
        let after_first = guard.charged();

        let mut r = BitReader::new(&data);
        decode_image_headers_metered(&mut r, &limits, &mut guard).expect("valid");
        assert!(guard.charged() >= after_first);
    }

    #[test]
    fn truncated_stream_errors_without_panicking() {
        let full = minimal_headers();
        for cut in 0..full.len() {
            let prefix = full.get(..cut).expect("cut is within the buffer");
            let mut r = BitReader::new(prefix);
            let result = decode_image_headers(&mut r, &Limits::default());
            // The header occupies exactly 32 bits, so any shorter prefix must
            // report an error rather than panic or invent a value.
            assert_eq!(
                result.is_err(),
                cut < 4,
                "prefix of {cut} bytes decoded unexpectedly"
            );
        }
    }

    #[cfg(feature = "trace")]
    #[test]
    fn every_field_is_traced_in_order() {
        let data = minimal_headers();
        let mut r = BitReader::new(&data);
        decode_image_headers(&mut r, &Limits::default()).expect("valid");

        let events = r.trace().events();
        assert!(!events.is_empty());

        let names: Vec<&str> = events.iter().map(|e| e.name).collect();
        assert_eq!(names.first().copied(), Some("signature"));
        assert!(names.contains(&"size.div8"));
        assert!(names.contains(&"metadata.default_m"));

        // Intervals must tile the consumed prefix with no gaps or overlaps.
        let mut cursor = 0u64;
        for event in events {
            assert_eq!(event.start_bit, cursor, "gap or overlap at {}", event.name);
            cursor = event.end_bit;
        }
        assert_eq!(cursor, r.total_bits_read());
    }
}
