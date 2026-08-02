//! JPEG XL decoder.
//!
//! Decodes an ISO/IEC 18181 codestream into pixel data. This crate is
//! deliberately a peer of the (future) encoder rather than its owner: all
//! primitives shared between the two directions live in [`jpxl_core`], and all
//! bit-level codestream access goes through [`jpxl_bitstream`].
//!
//! Nothing is implemented yet; this is a placeholder for the decode pipeline.
