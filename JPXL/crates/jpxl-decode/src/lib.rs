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
//! * Annex E.4 embedded ICC profiles — the compressed representation carried
//!   in the codestream when `want_icc` is set, decoded to the exact profile
//!   bytes ([`extract_icc_profile`]).
//! * Annex C entropy coding, via [`jpxl_entropy`].
//!
//! * Annex J restoration filters, K.2 non-separable upsampling (for both
//!   `upsampling` and `ec_upsampling`), K.3 patches, and Annex L colour
//!   transforms.
//!
//! Not implemented: J.2 simple upsampling (`do_YCbCr` chroma subsampling),
//! YCbCr reconstruction, splines, noise, animation, and upsampling in a
//! `kModular` frame. Each is a typed [`DecodeError::Unsupported`] naming its
//! clause — never wrong pixels.
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
pub mod icc;
pub mod modular;
pub mod signature;
pub mod vardct;

#[cfg(test)]
mod testsupport;

pub use decode::{DecodedImage, Plane, decode};
pub use error::{DecodeError, Result};
pub use headers::{ImageHeaders, ImageMetadata, Orientation, SizeHeader, decode_image_headers};
pub use icc::{IccError, extract_icc_profile};
pub use signature::{CODESTREAM_SIGNATURE, SIGNATURE_BYTES, starts_with_signature};
