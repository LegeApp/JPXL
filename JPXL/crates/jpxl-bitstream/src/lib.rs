//! Bit-level I/O and normative integer primitives for JPEG XL.
//!
//! This crate is shared by the decoder and the encoder. It provides the
//! lowest layer of the codestream: a [`BitReader`] that consumes bits
//! LSB-first within each byte, the mirror-image [`BitWriter`], and the
//! header-syntax primitives built on them — `u(n)`, `Bool()`,
//! [`U32()`](read_u32), [`U64()`](read_u64) and [`F16()`](read_f16_as_f32).
//!
//! # Bit order
//!
//! Bytes are consumed in stream order; within a byte, bits are taken starting
//! at the least significant one. Reading `n` bits places the first-read bit in
//! the least-significant position of the result, so a 32-bit read over four
//! bytes is exactly a little-endian `u32`.
//!
//! ```
//! use jpxl_bitstream::BitReader;
//! let mut r = BitReader::new(&[0b0000_0001]);
//! assert_eq!(r.read_bits(1)?, 1);
//! # Ok::<(), jpxl_bitstream::BitstreamError>(())
//! ```
//!
//! # Errors
//!
//! Nothing here panics on malformed input. Running off the end of the buffer,
//! an out-of-range field width and an invalid `F16()` are all reported as
//! [`BitstreamError`] values.
//!
//! # Tracing
//!
//! With the `trace` Cargo feature enabled, [`trace_field!`] records the bit
//! interval consumed by each named field on the reader's
//! [`TraceLog`](trace::TraceLog); with the feature off it compiles to nothing.
//!
//! # Specification references
//!
//! Clause references point at ISO/IEC 18181-1. Items marked
//! `[provisional: verify vs 18181-1 OCR]` follow the consensus definition and
//! the published overview paper, pending the normative text.

pub mod error;
pub mod primitives;
pub mod reader;
pub mod trace;
pub mod writer;

pub use error::{BitstreamError, Result};
pub use primitives::{U32Dist, U32Spec, read_bool, read_f16_as_f32, read_u32, read_u64};
pub use reader::BitReader;
pub use trace::{TraceEvent, TraceLog};
pub use writer::BitWriter;
