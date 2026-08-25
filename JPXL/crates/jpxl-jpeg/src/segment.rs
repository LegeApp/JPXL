//! The parsed JPEG document model.
//!
//! A JPEG is `SOI`, an ordered list of marker segments (some carrying entropy
//! data), `EOI`, then optional trailing bytes. Byte-exact re-emission depends
//! on preserving that order exactly, including how multiple tables were grouped
//! into a single `DQT`/`DHT` segment, so segments are kept as an ordered list
//! rather than folded into a normalized structure.

use crate::bitio::Padding;
use crate::frame::FrameHeader;
use crate::huffman::HuffmanTable;
use crate::quant::QuantTable;
use crate::scan::ScanHeader;
use crate::units::ComponentId;

/// An application segment `APPn` (`0xE0..=0xEF`).
#[derive(Clone, Debug)]
pub struct AppSegment {
    /// The marker code byte (`0xE0` = APP0 … `0xEF` = APP15).
    pub code: u8,
    /// The raw payload (everything after the 2-byte length field).
    pub payload: Vec<u8>,
}

/// A start-of-scan segment: the header plus the artifacts needed to reproduce
/// its entropy-coded data bit-for-bit.
#[derive(Clone, Debug)]
pub struct ScanSegment {
    /// The `SOS` header.
    pub header: ScanHeader,
    /// The end-of-segment padding bits for each entropy segment, in order:
    /// one entry per restart interval plus one for the final segment
    /// (10918-1 F.1.2.3).
    pub padding: Vec<Padding>,
    /// For a progressive AC scan, the end-of-band run lengths exactly as the
    /// original encoder emitted them, in decode order (10918-1 G.1.2.2).
    ///
    /// An encoder is free to split a long EOB run into several `EOBn` codes
    /// (libjpeg flushes when its correction-bit buffer fills), and that split
    /// is *not* recoverable from the coefficients alone. Recording the observed
    /// run lengths and replaying them is what keeps the re-encode bit-exact.
    /// Empty for DC and baseline scans.
    pub eob_runs: Vec<u32>,
}

/// A marker segment the codec does not model in detail but must preserve
/// verbatim (e.g. `DNL`).
#[derive(Clone, Debug)]
pub struct OtherSegment {
    /// The marker code byte.
    pub code: u8,
    /// The raw payload (everything after the 2-byte length field).
    pub payload: Vec<u8>,
}

/// One element of the ordered segment list between `SOI` and `EOI`.
#[derive(Clone, Debug)]
pub enum Segment {
    /// `APPn` application data.
    App(AppSegment),
    /// `COM` comment payload.
    Com(Vec<u8>),
    /// `DQT` — one or more quantization tables, in segment order.
    Dqt(Vec<QuantTable>),
    /// `DHT` — one or more Huffman tables, in segment order.
    Dht(Vec<HuffmanTable>),
    /// `DRI` — restart interval in MCUs.
    Dri(u16),
    /// `SOF` — the frame header.
    Sof(FrameHeader),
    /// `SOS` — a scan header and its entropy segment padding.
    Sos(ScanSegment),
    /// Any other length-bearing marker segment, preserved verbatim.
    Other(OtherSegment),
}

/// Decoded quantized coefficients for one frame component.
///
/// Blocks are stored row-major at the *interleaved* (MCU-padded) dimensions,
/// so a single plane serves both interleaved and non-interleaved scans of the
/// same component. Each block holds 64 coefficients in **natural** (raster)
/// order; element 0 is DC.
#[derive(Clone, Debug)]
pub struct ComponentPlane {
    /// Component identifier `Ci`.
    pub id: ComponentId,
    /// Horizontal sampling factor `Hi`.
    pub h: u8,
    /// Vertical sampling factor `Vi`.
    pub v: u8,
    /// Blocks per line (interleaved / MCU-padded).
    pub blocks_per_line: usize,
    /// Block rows (interleaved / MCU-padded).
    pub block_rows: usize,
    /// The blocks, row-major: block `(bx, by)` is at `by * blocks_per_line + bx`.
    pub blocks: Vec<[i16; 64]>,
}

impl ComponentPlane {
    /// Immutable access to block `(bx, by)`.
    #[must_use]
    pub fn block(&self, bx: usize, by: usize) -> Option<&[i16; 64]> {
        if bx >= self.blocks_per_line || by >= self.block_rows {
            return None;
        }
        self.blocks.get(by * self.blocks_per_line + bx)
    }
}

/// A fully parsed JPEG: its segment structure, its decoded coefficients, and
/// its trailing bytes.
#[derive(Clone, Debug)]
pub struct Jpeg {
    /// The ordered segments between `SOI` and `EOI`.
    pub segments: Vec<Segment>,
    /// The frame header (`None` only for a malformed frame-less stream, which
    /// parsing rejects before returning).
    pub frame: Option<FrameHeader>,
    /// Decoded coefficient planes, indexed like `frame.components`.
    pub planes: Vec<ComponentPlane>,
    /// Bytes after `EOI` (the "tail data" Annex A preserves).
    pub tail: Vec<u8>,
}
