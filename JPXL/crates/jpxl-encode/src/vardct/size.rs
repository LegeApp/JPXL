//! Exact size accounting for an emitted codestream (`docs/PLAN.md` slice 14).
//!
//! # There is exactly one sizing implementation, and it is the writer
//!
//! A rate loop needs the size of a candidate plan. The tempting design is a
//! *size model*: a second traversal that adds up field widths and estimated
//! entropy costs without producing bits. That design is a paired-bug shape —
//! the model and the writer are two implementations of one question, they drift
//! the moment either changes, and the loop then converges confidently on a
//! number nobody emits.
//!
//! So there is no model. [`price_codestream`](super::price_codestream) runs the
//! **same write path** as [`emit_codestream`](super::emit_codestream) with
//! count-only [`BitWriter`](jpxl_bitstream::BitWriter)s (bit length only, no
//! payload buffers). The cost is still one encode per candidate; the benefit
//! is that a priced size and an emitted size cannot disagree, intermediate
//! rate probes do not retain full section bodies, and the tests that assert
//! equality are trivial rather than aspirational.
//!
//! # What "per-section" means here
//!
//! F.3.1's TOC already measures every section in bytes — the encoder must know
//! those lengths before it may write the table (see
//! [`SectionStore`](crate::section::SectionStore)). This module does not
//! recompute them; it *surfaces* them, paired with the [`SectionKind`] each one
//! carries, plus the three things the TOC does not measure: the image headers,
//! the frame header, and the table itself.

use crate::vardct::geometry::SectionKind;

/// One TOC section's exact emitted size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SectionSize {
    /// What the section carries.
    pub kind: SectionKind,
    /// Its byte length — the value F.3.3 writes into the TOC entry.
    pub bytes: u64,
}

/// Every byte of one emitted codestream, attributed.
///
/// The invariant, asserted in this module's tests and by
/// `price_matches_write`: `total` equals the emitted `Vec<u8>`'s length, and
/// equals `image_headers + ceil((frame_header_bits + toc_bits) / 8) + sum of
/// section bytes` — the frame header and the TOC share their byte-alignment
/// boundary with nothing else, so the two are rounded up together.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodestreamSizing {
    /// Total emitted bytes.
    pub total: u64,
    /// The signature and image headers, through F.1's byte alignment.
    pub image_headers: u64,
    /// The frame header, in bits: F.2 does not byte-align after it.
    pub frame_header_bits: u64,
    /// The TOC, in bits, including F.3.3's two `ZeroPadToByte()`s.
    pub toc_bits: u64,
    /// Each section, in the order F.3.1 puts it on the wire.
    pub sections: Box<[SectionSize]>,
}

impl CodestreamSizing {
    /// The bytes carried by sections, i.e. everything the TOC measures.
    #[must_use]
    pub fn section_bytes(&self) -> u64 {
        self.sections.iter().map(|s| s.bytes).sum()
    }

    /// The bytes that are not section payload: headers plus the table.
    #[must_use]
    pub fn overhead(&self) -> u64 {
        self.total.saturating_sub(self.section_bytes())
    }

    /// The bytes carried by every section for which `pick` holds.
    ///
    /// The rate loop uses this to say which part of the stream a quantizer
    /// change actually moved — `PassGroup` payload is where the coefficients
    /// are, and a change that only moved the headers is a change that did
    /// nothing.
    #[must_use]
    pub fn bytes_where(&self, pick: impl Fn(SectionKind) -> bool) -> u64 {
        self.sections
            .iter()
            .filter(|s| pick(s.kind))
            .map(|s| s.bytes)
            .sum()
    }

    /// The bytes carried by `PassGroup` sections — or, in F.3.1's
    /// single-section form, by the one section that contains them.
    #[must_use]
    pub fn coefficient_bytes(&self) -> u64 {
        self.bytes_where(|k| matches!(k, SectionKind::PassGroup { .. } | SectionKind::Whole))
    }
}

/// One emitted codestream and its exact accounting.
#[derive(Debug, Clone)]
pub struct Emission {
    /// The codestream.
    pub bytes: Vec<u8>,
    /// Where its bytes went.
    pub sizing: CodestreamSizing,
}
