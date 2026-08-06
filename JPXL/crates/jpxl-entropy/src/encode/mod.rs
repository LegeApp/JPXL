//! Entropy encoding for JPEG XL — the write side of ISO/IEC 18181-1 Annex C.
//!
//! Everything in [`crate`] outside this module reads; everything inside it
//! writes. The two sides share *types* and *tables* — the alias mapping of
//! C.2.6, the fixed logarithmic-count code of C.2.5, the code-length order of
//! RFC 7932 — but never control flow. That is deliberate: a shared write/read
//! implementation makes an encoder bug automatically acceptable to the paired
//! decoder, and the round-trip tests would prove nothing.
//!
//! # The shape of an encode
//!
//! ```no_run
//! use jpxl_bitstream::BitWriter;
//! use jpxl_entropy::encode::{CodingMode, EncoderPlan, EntropyTables, SymbolEncoder, TokenCensus};
//! use jpxl_entropy::HybridUintConfig;
//!
//! # fn demo(values: &[(usize, u32)]) -> Result<Vec<u8>, jpxl_entropy::EntropyError> {
//! let num_contexts = 4;
//!
//! // 1. Census: count raw values per pre-clustered context.
//! let mut census = TokenCensus::new(num_contexts)?;
//! for &(ctx, value) in values {
//!     census.record(ctx, value)?;
//! }
//!
//! // 2. Tables: apply the caller's policy to the census.
//! let config = HybridUintConfig::new(4, 2, 1)?;
//! let plan = EncoderPlan::identity(num_contexts, CodingMode::Ans, config)?;
//! let tables = EntropyTables::build(&plan, &census)?;
//!
//! // 3. Replay: record the same values again and emit.
//! let mut w = BitWriter::new();
//! tables.write_bundle(&mut w)?;
//! let mut encoder = SymbolEncoder::new(&tables);
//! for &(ctx, value) in values {
//!     encoder.push_uint(ctx, value)?;
//! }
//! encoder.write_stream(&mut w)?;
//! Ok(w.into_bytes())
//! # }
//! ```
//!
//! # LZ77
//!
//! [`EncoderPlan::lz77`] is `None` by default (flag false). Enable with
//! [`EncoderPlan::identity_with_lz77`] or by setting the field and sizing the
//! context map with a trailing distance context. Census copies with
//! [`TokenCensus::record_copy`]; emit with [`SymbolEncoder::push_copy`].
//!
//! # What is not implemented
//!
//! * **Run-length compression of logarithmic counts** (C.2.5's `logcounts ==
//!   13` escape) and **repeat codes in prefix code lengths** (RFC 7932 section
//!   3.5's symbols 16 and 17). Both are density optimizations of the *table*
//!   encodings; the decoder reads them either way.
//! * **Clustering search.** [`ContextMap`] executes a clustering; choosing one
//!   is policy and belongs to the policy crate.
//! * **Match finding.** This crate emits copies the caller chooses; greedy
//!   search lives in `jpxl-encode`.
//!
//! Every path here is **bit-exact** (see `docs/PLAN.md`): the contract is that
//! this crate's decoder reproduces exactly the values that went in, with the
//! ANS terminal state of C.3.2 holding at the end.

pub mod ans;
pub mod cluster;
pub mod histogram;
pub mod hybrid;
pub mod lz77;
pub mod prefix;
pub mod stream;

pub use ans::{AnsEncodeTable, AnsPayload, AnsSymbol, encode_symbols};
pub use cluster::{ContextMap, ContextMapForm, move_to_front};
pub use histogram::Histogram;
pub use hybrid::TokenSplit;
pub use lz77::{LZ_LENGTH_LOG_ALPHABET_SIZE, LengthToken, Lz77EncodeParams};
pub use prefix::{PrefixEncoder, canonical_codes, huffman_lengths};
pub use stream::{
    CodingMode, EncoderPlan, EntropyTables, RawHistogram, SymbolEncoder, TokenCensus,
};
