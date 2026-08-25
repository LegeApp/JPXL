//! JPEG marker bytes and start-of-frame classification (10918-1 Table B.1).
//!
//! A marker is two bytes: `0xFF` then a code byte. This module names the codes
//! this codec cares about and, crucially, decides which start-of-frame codes
//! are *in scope* for Phase A and which are refused.

/// The marker-prefix byte. Every marker is `MARKER_PREFIX` then a code byte.
pub const MARKER_PREFIX: u8 = 0xFF;

// Standalone markers (no length, no payload).
/// Start of image.
pub const SOI: u8 = 0xD8;
/// End of image.
pub const EOI: u8 = 0xD9;
/// Temporary (arithmetic), standalone.
pub const TEM: u8 = 0x01;

/// First restart marker `RST0`; restart markers are `RST0..=RST7`.
pub const RST0: u8 = 0xD0;
/// Last restart marker `RST7`.
pub const RST7: u8 = 0xD7;

// Segment markers (a 2-byte big-endian length follows the code).
/// Define quantization table(s).
pub const DQT: u8 = 0xDB;
/// Define Huffman table(s).
pub const DHT: u8 = 0xC4;
/// Define restart interval.
pub const DRI: u8 = 0xDD;
/// Start of scan.
pub const SOS: u8 = 0xDA;
/// Comment.
pub const COM: u8 = 0xFE;
/// First application segment `APP0`.
pub const APP0: u8 = 0xE0;
/// Last application segment `APP15`.
pub const APP15: u8 = 0xEF;
/// Define arithmetic conditioning (arithmetic coding — refused).
pub const DAC: u8 = 0xCC;
/// Define hierarchical progression (hierarchical mode — refused).
pub const DHP: u8 = 0xDE;
/// Expand reference component (hierarchical mode — refused).
pub const EXP: u8 = 0xDF;
/// Define number of lines.
pub const DNL: u8 = 0xDC;

/// Whether `code` is a restart marker `RST0..=RST7`.
#[must_use]
pub fn is_restart(code: u8) -> bool {
    (RST0..=RST7).contains(&code)
}

/// Whether `code` names a start-of-frame segment (`SOF0..=SOF15`, excluding the
/// codes `0xC4` DHT, `0xC8` JPG and `0xCC` DAC that fall in the same range).
#[must_use]
pub fn is_sof(code: u8) -> bool {
    matches!(code, 0xC0..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF)
}

/// The entropy-coding and process class a start-of-frame code selects, used to
/// decide Phase-A support (10918-1 Table B.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SofKind {
    /// Baseline sequential DCT, Huffman (SOF0). Supported.
    BaselineSequential,
    /// Extended sequential DCT, Huffman (SOF1). Supported (8-bit only).
    ExtendedSequential,
    /// Progressive DCT, Huffman (SOF2). Supported.
    Progressive,
    /// Lossless (sequential), Huffman (SOF3). Refused.
    Lossless,
    /// Differential/hierarchical Huffman (SOF5/6/7). Refused.
    Hierarchical,
    /// Arithmetic-coded (SOF9/10/11/13/14/15). Refused.
    Arithmetic,
}

impl SofKind {
    /// Classifies a start-of-frame code byte.
    ///
    /// Returns `None` for codes that are not start-of-frame markers.
    #[must_use]
    pub fn from_code(code: u8) -> Option<Self> {
        Some(match code {
            0xC0 => Self::BaselineSequential,
            0xC1 => Self::ExtendedSequential,
            0xC2 => Self::Progressive,
            0xC3 => Self::Lossless,
            0xC5..=0xC7 => Self::Hierarchical,
            0xC9..=0xCB | 0xCD..=0xCF => Self::Arithmetic,
            _ => return None,
        })
    }

    /// Whether this frame kind is decoded by Phase A (Huffman DCT modes only).
    #[must_use]
    pub fn is_supported(self) -> bool {
        matches!(
            self,
            Self::BaselineSequential | Self::ExtendedSequential | Self::Progressive
        )
    }

    /// Whether this scan is coded progressively (spectral selection / successive
    /// approximation apply).
    #[must_use]
    pub fn is_progressive(self) -> bool {
        matches!(self, Self::Progressive)
    }
}
