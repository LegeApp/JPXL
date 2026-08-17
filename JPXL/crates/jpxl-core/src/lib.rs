//! Shared primitives for the JPXL JPEG XL codec.
//!
//! This crate holds the pieces that the decoder and the encoder both need, and
//! it treats them as peers: nothing here is phrased in terms of "decode side"
//! versus "encode side". A transform lives here when both directions use it
//! (XYB color, the DCT), and a type lives here when both directions must agree
//! on it (errors, resource limits, geometry).
//!
//! The only dependency is [`jpxl_bitstream`], and through it nothing at all —
//! JPXL is a zero-external-dependency workspace.
//!
//! # Modules
//!
//! * [`error`] — the crate error type and the project's hand-rolled error style.
//! * [`limits`] — bounds and allocation metering for attacker-facing decodes.
//! * [`geometry`] — checked dimension, bit-depth, and group-size newtypes.
//! * [`color`] — the XYB color model and its inverse.
//! * [`dct`] — the integer/float DCT shared by both directions.
//! * [`dequant`] — the I.2.4/I.2.5 quantization weights and their defaults.
//! * [`varblock`] — the VarDCT transform-type vocabulary (18181-1 Annex I) and
//!   the coefficients-to-samples reconstruction.
//! * [`forward`] — the exact inverse of that reconstruction: the allocation-free
//!   analysis transforms an encoder needs, and `lf_from_llf`.

pub mod color;
pub mod cpu;
pub mod dct;
pub mod dequant;
pub mod error;
pub mod forward;
pub mod geometry;
pub mod limits;
pub mod modular_weighted;
pub mod simd;
pub mod varblock;

pub use error::{JpxlError, Result};
