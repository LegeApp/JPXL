//! Test-harness support library for the JPXL codec.
//!
//! This crate is **not published** and is not part of the codec proper: nothing
//! in `jpxl-decode` or `jpxl-cli`'s decode path may depend on it for correctness.
//! It exists so that integration tests, fuzz drivers and the conformance sweep
//! share one implementation of the boring parts of a codec test rig:
//!
//! * [`sniff`] — cheap classification of a byte stream as a naked JPEG XL
//!   codestream, an ISOBMFF-style container, or something else entirely.
//! * [`oracle`] — discovery of and shell-outs to reference decoders (`djxl`
//!   from libjxl, `jxl-oxide`). Oracles are treated strictly as **black
//!   boxes**: we feed them bytes and compare their pixels. Their source is
//!   never consulted, which is what keeps this a clean-room implementation.
//! * [`metrics`] — a minimal PPM reader and pixel-difference metrics used to
//!   grade our output against an oracle's.
//!
//! # Oracles are optional
//!
//! Every oracle entry point degrades gracefully when the binary is absent
//! ([`oracle::OracleError::Unavailable`]). Tests that need an oracle are
//! expected to *skip*, never to fail, on a machine where none is installed —
//! CI must stay green without a libjxl build.
//!
//! # Zero dependencies
//!
//! Like the rest of the workspace, this crate uses only `std`.

#![forbid(unsafe_code)]

pub mod metrics;
pub mod oracle;
pub mod sniff;

pub use metrics::{Image, max_abs_error, peak_error_per_channel};
pub use oracle::{Oracle, OracleError, OracleKind};
pub use sniff::{StreamKind, sniff};
