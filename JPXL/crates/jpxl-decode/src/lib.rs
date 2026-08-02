//! JPEG XL decoder.
//!
//! Decodes an ISO/IEC 18181 codestream into pixel data. This crate is
//! deliberately a peer of the (future) encoder rather than its owner: all
//! primitives shared between the two directions live in [`jpxl_core`], and all
//! bit-level codestream access goes through [`jpxl_bitstream`].
//!
//! # What is implemented
//!
//! End-to-end decoding of **lossless modular** codestreams: [`decode`] takes a
//! naked codestream or a Part 2 container and returns pixels.
//!
//! * Annex D image headers — signature (D.1), `SizeHeader` (D.2),
//!   `ImageMetadata` (D.3) and everything it nests, including `ColourEncoding`
//!   (Annex E) and `OpsinInverseMatrix` (L.2.1).
//! * Annex F frames — `FrameHeader` with its conditional forest, `TOC` with
//!   the entropy-coded permutation, group and section geometry.
//! * Annex G sections — `LfGlobal`/`GlobalModular` (G.1.3), `ModularLfGroup`
//!   (G.2.3) and `Modular group data` (G.4.2), sharing one channel list across
//!   the frame as the annex requires.
//! * Annex H modular — MA trees, all fourteen predictors including the
//!   self-correcting one, and the RCT, palette and squeeze inverses.
//! * Annex C entropy coding, via [`jpxl_entropy`].
//!
//! Not implemented: VarDCT (Annex I), XYB and YCbCr reconstruction, upsampling,
//! patches, splines, noise, and multi-frame blending. Each is a typed
//! [`DecodeError::Unsupported`] naming its clause — never wrong pixels.
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
//! # Bit-exactness
//!
//! `docs/PLAN.md` puts modular lossless in the bit-exact regime. The
//! end-to-end proof is `tests/e2e_lossless.rs`, which checks decoded samples
//! against the deterministic formulas the fixtures were generated from and,
//! when an oracle is installed, against `djxl` byte for byte.
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

pub mod container;
pub mod decode;
pub mod error;
pub mod frame;
pub mod headers;
pub mod modular;
pub mod signature;

#[cfg(test)]
mod testsupport;

pub use decode::{DecodedImage, Plane, decode};
pub use error::{DecodeError, Result};
pub use headers::{ImageHeaders, ImageMetadata, Orientation, SizeHeader, decode_image_headers};
pub use signature::{CODESTREAM_SIGNATURE, SIGNATURE_BYTES, starts_with_signature};
