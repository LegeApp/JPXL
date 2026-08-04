//! Entropy coding for JPEG XL — ISO/IEC 18181-1 Annex C.
//!
//! A JPEG XL codestream carries many independently entropy-coded streams. Each
//! opens with a *distribution bundle* describing how its symbols are coded,
//! followed by the symbols themselves. [`SymbolDecoder`] is the facade for
//! both halves: [`SymbolDecoder::open`] reads a bundle from a [`BitReader`],
//! and [`SymbolDecoder::read_uint`] pulls integers out of it.
//!
//! ```no_run
//! use jpxl_bitstream::BitReader;
//! use jpxl_core::limits::{AllocGuard, Limits};
//! use jpxl_entropy::SymbolDecoder;
//!
//! # fn demo(codestream: &[u8]) -> Result<(), jpxl_entropy::EntropyError> {
//! let limits = Limits::default();
//! let mut guard = AllocGuard::new(&limits);
//! let mut reader = BitReader::new(codestream);
//!
//! let num_contexts = 6; // supplied by the referencing clause
//! let mut decoder = SymbolDecoder::open(&mut reader, num_contexts, &mut guard)?;
//! let first = decoder.read_uint(&mut reader, 0)?;
//! decoder.finish()?; // C.3.2 terminal-state check
//! # let _ = first;
//! # Ok(())
//! # }
//! ```
//!
//! # Layers
//!
//! Annex C stacks four mechanisms, and this crate keeps them in separate
//! modules so each can be tested on its own:
//!
//! * [`prefix`] — canonical prefix codes (C.2.4, delegating to RFC 7932).
//! * [`ans`] — 12-bit probability distributions, the alias mapping, and the
//!   rANS state machine (C.2.5, C.2.6, C.3.2).
//! * [`hybrid`] — the hybrid unsigned integer scheme that turns a token plus
//!   raw extra bits into a value (C.2.3, C.3.3).
//! * [`lz77`] — the back-reference layer over the decoded symbol history
//!   (Table C.1, C.3.3).
//!
//! [`dist`] reads the context-to-cluster mapping (C.2.2) and [`decoder`]
//! assembles everything into the bundle of C.2.1.
//!
//! # Encoding
//!
//! [`encode`] is the write side: a two-pass entropy compiler (census the raw
//! values, build tables, replay and emit) covering rANS emission, histogram
//! serialization, prefix-code construction, hybrid-uint token building and
//! context maps. It shares this crate's *tables* and *types* with the decoder
//! but none of its control flow, so the round-trip tests are a real check
//! rather than one implementation agreeing with itself.
//!
//! # Bit-exactness
//!
//! Every path in this crate is **bit-exact** per `docs/PLAN.md`: the
//! arithmetic is integer and fully specified by the standard, so there is no
//! tolerance regime. Any divergence from a reference decoder is a bug.
//!
//! # Resource safety
//!
//! Alphabet sizes, cluster counts and the LZ77 window are all attacker
//! controlled. Every allocation derived from the stream is charged to an
//! [`AllocGuard`](jpxl_core::limits::AllocGuard) *before* it is made,
//! arithmetic on stream-derived values is checked, and no index is taken
//! without a bounds check. Nothing in this crate panics on malformed input.
//!
//! # Specification ambiguities
//!
//! Two places in Annex C admit more than one reading. Both are recorded here
//! because they are decided by this implementation and must be confirmed
//! against real streams in slice 7.
//!
//! 1. **C.2.2, nested LZ77.** The clause says that when `num_dist == 2` the
//!    recursive distribution decoding has `lz77.enabled` false. That is either
//!    a constraint on well-formed streams (the flag is read and must be zero)
//!    or a parameter override (the flag is not read at all), and the two
//!    differ by one bit. This crate takes the constraint reading, because
//!    Table C.1 lists the flag unconditionally, and rejects a stream that sets
//!    it. The rule exists to bound the recursion — a nested decoder that
//!    enabled LZ77 would itself reach `num_dist == 2` and recurse again — so
//!    [`dist::MAX_NESTING_DEPTH`] enforces that bound independently.
//!
//! 2. **C.3.3, the recursive call.** After starting an LZ77 copy the clause
//!    calls `DecodeHybridVarLenUint(clusters[ctx])`, passing an already
//!    clustered index to a parameter documented as pre-clustered. This is
//!    moot: `min_length` is at least 3, so the copy counter is positive and
//!    the recursive call can only take the copy branch, which never looks at
//!    its argument. Both transcriptions agree on the text, so this is a quirk
//!    of the specification rather than an OCR artefact.
//!
//! A third point is an OCR artefact rather than an ambiguity: the LaTeX
//! transcription of C.2.6 renders the alias-construction line as `symbols[u] =
//! 0`, while the markdown conversion has `symbols[u] = o`. Only the latter is
//! consistent with the surrounding algorithm, and
//! [`ans`]'s alias-table tests pin the resulting behaviour.

pub mod ans;
pub mod decoder;
pub mod dist;
pub mod encode;
pub mod error;
pub mod hybrid;
pub mod lz77;
pub mod prefix;

pub use ans::{AnsDistribution, AnsState};
pub use decoder::SymbolDecoder;
pub use dist::{ClusterMap, inverse_move_to_front, read_cluster_map};
pub use encode::{
    CodingMode, ContextMap, ContextMapForm, EncoderPlan, EntropyTables, Histogram, PrefixEncoder,
    SymbolEncoder, TokenCensus,
};
pub use error::{EntropyError, Result};
pub use hybrid::HybridUintConfig;
pub use lz77::{Lz77Params, Lz77Window};
pub use prefix::{PrefixCode, read_prefix_code};
