//! JPEG XL decoder.
//!
//! Decodes an ISO/IEC 18181 codestream into pixel data. This crate is
//! deliberately a peer of the (future) encoder rather than its owner: all
//! primitives shared between the two directions live in [`jpxl_core`], and all
//! bit-level codestream access goes through [`jpxl_bitstream`].
//!
//! # What is implemented
//!
//! The image headers of Annex D: the codestream signature (D.1), `SizeHeader`
//! (D.2) and `ImageMetadata` (D.3) with everything it nests — `BitDepth`,
//! `ExtraChannelInfo`, `PreviewHeader`, `AnimationHeader`, `ColourEncoding`
//! (Annex E), `ToneMapping`, `OpsinInverseMatrix` (L.2.1) and the `Extensions`
//! bundle (B.3). Frames, entropy coding and pixel reconstruction are not.
//!
//! ```
//! use jpxl_bitstream::BitReader;
//! use jpxl_core::limits::Limits;
//!
//! // The signature FF 0A, an 8x8 image, and default metadata.
//! let data = [0xFF, 0x0A, 0b0000_0001, 0b1100_0000];
//! let mut reader = BitReader::new(&data);
//! let headers = jpxl_decode::decode_image_headers(&mut reader, &Limits::default())?;
//! assert_eq!((headers.width(), headers.height()), (8, 8));
//! # Ok::<(), jpxl_decode::DecodeError>(())
//! ```
//!
//! # Reading the header code against the standard
//!
//! Each bundle gets a module, each module's doc-comment reproduces the spec
//! table it implements, and each field read is wrapped in
//! [`trace_field!`](jpxl_bitstream::trace_field) under its spec name. With the
//! `trace` feature on, a mis-parsed header can be localised to the exact bit at
//! which the field intervals stop matching the reference, which is the only
//! debugging question that matters for a bitstream format.
//!
//! # Untrusted input
//!
//! Every parser takes a [`Limits`](jpxl_core::limits::Limits) and meters
//! stream-controlled allocations through an
//! [`AllocGuard`](jpxl_core::limits::AllocGuard) before allocating. Malformed
//! input is always an error, never a panic.

pub mod error;
pub mod frame;
pub mod headers;
pub mod modular;
pub mod signature;

#[cfg(test)]
mod testsupport;

pub use error::{DecodeError, Result};
pub use headers::{ImageHeaders, ImageMetadata, Orientation, SizeHeader, decode_image_headers};
pub use signature::{CODESTREAM_SIGNATURE, SIGNATURE_BYTES, starts_with_signature};
