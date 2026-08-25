//! Clean-room ISO/IEC 10918-1 (JPEG-1 / ITU-T T.81) coefficient-level codec.
//!
//! This crate parses a JPEG-1 bitstream into typed structures — markers,
//! quantization tables, Huffman tables, frame and scan headers, and the
//! entropy-decoded DCT coefficient planes — and re-emits a **byte-for-byte
//! identical** JPEG from those structures. It has no pixel path and no IDCT: it
//! stops at quantized coefficients, which is exactly the surface JPEG XL's
//! lossless recompression needs (see `JPXL/docs/jpeg-recompression-plan.md`,
//! Phase A).
//!
//! ```no_run
//! # fn demo(bytes: &[u8]) -> Result<(), jpxl_jpeg::JpegError> {
//! let jpeg = jpxl_jpeg::parse(bytes)?;      // -> typed model + coefficients
//! let round = jpxl_jpeg::serialize(&jpeg)?; // exact inverse
//! assert_eq!(bytes, round.as_slice());      // bit-exact round-trip
//! # Ok(())
//! # }
//! ```
//!
//! # Scope
//!
//! Supported: baseline sequential (SOF0), extended sequential Huffman (SOF1)
//! and progressive Huffman (SOF2) JPEGs at 8-bit precision, with 1–4
//! components, any 4:4:4 / 4:2:2 / 4:2:0 / 4:4:0 subsampling, restart
//! intervals, and arbitrary `APPn` / `COM` / trailing-garbage payloads.
//!
//! Refused with a typed [`JpegError::Unsupported`], never mis-decoded:
//! arithmetic coding (SOF9–15, DAC), hierarchical mode (SOF5–7, DHP/EXP),
//! lossless mode (SOF3), and 12-bit samples.
//!
//! # Clean-room
//!
//! Every field order, table layout and entropy-coding rule here derives from
//! ITU-T T.81 / ISO-IEC 10918-1 and this workspace's own reasoning. libjxl is
//! not consulted as a source.

pub mod bitio;
pub mod codec;
pub mod coeff;
pub mod error;
pub mod frame;
pub mod huffman;
pub mod limits;
pub mod marker;
pub mod parse;
pub mod progressive;
pub mod quant;
pub mod scan;
pub mod segment;
pub mod serialize;
pub mod units;

pub use error::{JpegError, Result};
pub use limits::Limits;
pub use parse::{parse, parse_with_limits};
pub use segment::{ComponentPlane, Jpeg, Segment};
pub use serialize::serialize;
