//! Stream classification: is this a naked codestream, a container, or neither?
//!
//! Both signatures are normative (ISO/IEC 18181-1 and -2):
//!
//! * A **naked codestream** begins with the two-byte marker `FF 0A`.
//! * A **container** file begins with the 12-byte JXL signature box
//!   `00 00 00 0C 4A 58 4C 20 0D 0A 87 0A`, i.e. a box of length 12 with type
//!   `JXL ` whose payload is `\r\n\x87\n` — the usual "detect accidental
//!   CRLF/8-bit mangling" trick borrowed from PNG.
//!
//! Sniffing is deliberately shallow: it looks at the first few bytes and
//! nothing else. A [`StreamKind::NakedCodestream`] verdict says only that the
//! signature matched, never that the stream decodes.

/// The signature box that opens every JPEG XL container file.
pub const CONTAINER_SIGNATURE: [u8; 12] = [
    0x00, 0x00, 0x00, 0x0C, 0x4A, 0x58, 0x4C, 0x20, 0x0D, 0x0A, 0x87, 0x0A,
];

/// The marker that opens every naked JPEG XL codestream.
pub const CODESTREAM_SIGNATURE: [u8; 2] = [0xFF, 0x0A];

/// What a byte stream looks like from the outside.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamKind {
    /// Starts with `FF 0A`: a bare JPEG XL codestream.
    NakedCodestream,
    /// Starts with the 12-byte JXL signature box: a container (`.jxl` file
    /// with boxes, possibly carrying Exif/XMP/JBRD alongside the codestream).
    Container,
    /// Neither signature matched, or the input is too short to tell.
    Unknown,
}

impl StreamKind {
    /// A short human-readable label, as printed by the `jpxl info` subcommand.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::NakedCodestream => "naked codestream",
            Self::Container => "container",
            Self::Unknown => "unknown",
        }
    }

    /// Whether this is a shape JPXL recognises as JPEG XL at all.
    #[must_use]
    pub const fn is_recognized(self) -> bool {
        matches!(self, Self::NakedCodestream | Self::Container)
    }
}

/// Classify `bytes` by its leading signature.
///
/// Truncated input is never an error: anything shorter than a matching
/// signature is simply [`StreamKind::Unknown`].
///
/// ```
/// use jpxl_conformance::{StreamKind, sniff};
///
/// assert_eq!(sniff(&[0xFF, 0x0A, 0x00]), StreamKind::NakedCodestream);
/// assert_eq!(sniff(&[0xFF]), StreamKind::Unknown);
/// assert_eq!(sniff(&[]), StreamKind::Unknown);
/// ```
#[must_use]
pub fn sniff(bytes: &[u8]) -> StreamKind {
    if bytes.starts_with(&CONTAINER_SIGNATURE) {
        StreamKind::Container
    } else if bytes.starts_with(&CODESTREAM_SIGNATURE) {
        StreamKind::NakedCodestream
    } else {
        StreamKind::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_is_unknown() {
        assert_eq!(sniff(&[]), StreamKind::Unknown);
    }

    #[test]
    fn one_byte_of_codestream_signature_is_unknown() {
        assert_eq!(sniff(&[0xFF]), StreamKind::Unknown);
    }

    #[test]
    fn bare_codestream_signature_is_enough() {
        assert_eq!(sniff(&[0xFF, 0x0A]), StreamKind::NakedCodestream);
    }

    #[test]
    fn codestream_signature_with_payload() {
        let bytes = [0xFF, 0x0A, 0x00, 0x11, 0x22, 0x33];
        assert_eq!(sniff(&bytes), StreamKind::NakedCodestream);
    }

    #[test]
    fn container_signature_box() {
        assert_eq!(sniff(&CONTAINER_SIGNATURE), StreamKind::Container);
    }

    #[test]
    fn container_signature_with_trailing_boxes() {
        let mut bytes = CONTAINER_SIGNATURE.to_vec();
        bytes.extend_from_slice(&[0x00, 0x00, 0x00, 0x14, 0x66, 0x74, 0x79, 0x70]);
        assert_eq!(sniff(&bytes), StreamKind::Container);
    }

    #[test]
    fn every_truncation_of_the_container_signature_is_unknown() {
        for len in 0..CONTAINER_SIGNATURE.len() {
            let truncated = CONTAINER_SIGNATURE
                .get(..len)
                .expect("len is within the signature");
            assert_eq!(
                sniff(truncated),
                StreamKind::Unknown,
                "{len}-byte prefix should not classify"
            );
        }
    }

    #[test]
    fn container_signature_with_wrong_final_byte_is_unknown() {
        let mut bytes = CONTAINER_SIGNATURE;
        *bytes.last_mut().expect("signature is non-empty") = 0x0B;
        assert_eq!(sniff(&bytes), StreamKind::Unknown);
    }

    #[test]
    fn png_magic_is_unknown() {
        let png = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        assert_eq!(sniff(&png), StreamKind::Unknown);
    }

    #[test]
    fn jpeg_soi_is_unknown() {
        // FF D8 shares its first byte with the codestream marker but not the
        // second; make sure we require both.
        assert_eq!(sniff(&[0xFF, 0xD8, 0xFF, 0xE0]), StreamKind::Unknown);
    }

    #[test]
    fn labels_and_recognition() {
        assert_eq!(StreamKind::NakedCodestream.label(), "naked codestream");
        assert_eq!(StreamKind::Container.label(), "container");
        assert_eq!(StreamKind::Unknown.label(), "unknown");
        assert!(StreamKind::NakedCodestream.is_recognized());
        assert!(StreamKind::Container.is_recognized());
        assert!(!StreamKind::Unknown.is_recognized());
    }
}
