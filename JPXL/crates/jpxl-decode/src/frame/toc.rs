//! The TOC (18181-1 F.3) and its optional permutation.
//!
//! ```text
//! permuted_toc = Bool()
//! if permuted_toc: permutation   (F.3.2, entropy coded, 8 pre-clustered dists)
//! ZeroPadToByte()
//! entry[i] = U32(u(10), 1024 + u(14), 17408 + u(22), 4211712 + u(30))
//! ZeroPadToByte()
//! ```
//!
//! Each entry is a section's size in bytes. `group_offsets[0]` is 0 and
//! `group_offsets[i]` is the sum of entries `[0, i)`; after the byte position
//! `P` reached by the final `ZeroPadToByte()`, section `i` starts at
//! `P + group_offsets[i]`.
//!
//! # The permutation applies to the offsets, not the sizes
//!
//! F.3.3 says: "If `permuted_toc`, the decoder permutes `group_offsets`
//! according to `permutation`, such that `group_offsets[i]` is what was
//! previously `group_offsets[permutation[i]]`." The prefix sum is therefore
//! computed over the entries **as read**, and only then reindexed. Permuting
//! the sizes first and summing afterwards gives different offsets, so the
//! order of the two steps is load-bearing.
//!
//! # Permutation decoding (F.3.2)
//!
//! `GetContext(x) = min(7, ceil(log2(x + 1)))`. The eight contexts are why the
//! stream is opened with `num_dist == 8`.
//!
//! The `end` count and each Lehmer code are read with
//! `DecodeHybridVarLenUint`. The Lehmer sequence is then converted to a
//! permutation by repeatedly removing `temp[lehmer[i]]` from a working list.

use jpxl_bitstream::{BitReader, U32Dist, U32Spec, read_bool, read_u32, trace_field};
use jpxl_core::limits::{AllocGuard, Limits};
use jpxl_entropy::SymbolDecoder;

use crate::frame::error::{FrameError, Result};

/// 18181-1 F.3.3: `U32(u(10), 1024 + u(14), 17408 + u(22), 4211712 + u(30))`.
const TOC_ENTRY_SPEC: U32Spec = U32Spec::new([
    U32Dist::bits(10),
    U32Dist::BitsOffset {
        bits: 14,
        offset: 1024,
    },
    U32Dist::BitsOffset {
        bits: 22,
        offset: 17408,
    },
    U32Dist::BitsOffset {
        bits: 30,
        offset: 4_211_712,
    },
]);

/// Number of pre-clustered distributions for the permutation stream (F.3.1).
pub const PERMUTATION_NUM_DIST: usize = 8;

/// Bytes charged per TOC entry before the tables are allocated.
const ENTRY_BUDGET_BYTES: u64 = 24;

/// `GetContext(x) = min(7, ceil(log2(x + 1)))` (18181-1 F.3.2).
///
/// The LaTeX transcription renders the constant as `?`; `part1.md` has `7`,
/// which is also the only value consistent with the stream being opened with
/// eight distributions.
#[must_use]
pub const fn get_context(x: u32) -> usize {
    // ceil(log2(x + 1)) is the number of bits needed to represent x.
    let bits = if x == 0 { 0 } else { 32 - x.leading_zeros() };
    if bits > 7 { 7 } else { bits as usize }
}

/// A decoded table of contents (18181-1 F.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Toc {
    /// Whether a permutation was signalled.
    pub permuted: bool,
    /// Section sizes in bytes, in bitstream order.
    pub entries: Vec<u64>,
    /// Byte offsets of each section from the post-TOC byte position,
    /// after the permutation has been applied.
    pub offsets: Vec<u64>,
    /// The decoded permutation, if one was present.
    pub permutation: Option<Vec<u32>>,
}

impl Toc {
    /// Number of sections.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the TOC is empty (never true for a well-formed frame).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total size of all sections in bytes.
    #[must_use]
    pub fn total_size(&self) -> u64 {
        self.entries.iter().sum()
    }

    /// Byte offset of section `index` relative to the end of the TOC.
    #[must_use]
    pub fn offset_of(&self, index: usize) -> Option<u64> {
        self.offsets.get(index).copied()
    }
}

/// Reads the TOC of a frame (18181-1 F.3).
///
/// `num_sections` comes from the frame's geometry
/// ([`FrameGeometry::num_sections`](crate::frame::geometry::FrameGeometry::num_sections)).
/// On return the reader sits at the byte position `P` from which section
/// offsets are measured.
///
/// # Errors
///
/// [`FrameError::LimitExceeded`] if the section count exceeds the allocation
/// budget, [`FrameError::Entropy`] if the permutation stream is malformed, or
/// a bitstream error on truncation or a nonzero padding bit.
pub fn read_toc(
    reader: &mut BitReader<'_>,
    num_sections: u64,
    limits: &Limits,
    guard: &mut AllocGuard,
) -> Result<Toc> {
    if num_sections == 0 {
        return Err(FrameError::out_of_range("num_sections", "F.3.1", 0));
    }
    // The caller derives num_sections from attacker-controlled geometry, so
    // meter before allocating three vectors of that length.
    guard
        .charge(num_sections.saturating_mul(ENTRY_BUDGET_BYTES))
        .map_err(FrameError::Core)?;

    let size = u32::try_from(num_sections).map_err(|_| FrameError::LimitExceeded {
        what: "num_sections",
        clause: "F.3.1",
        value: num_sections,
        limit: u64::from(u32::MAX),
    })?;

    let permuted = trace_field!(reader, "toc.permuted_toc", read_bool(reader))?;
    let permutation = if permuted {
        Some(read_permutation(reader, size, 0, limits, guard)?)
    } else {
        None
    };

    // F.3.3: before the entries, after the permutation.
    trace_field!(reader, "toc.pad_before_entries", reader.zero_pad_to_byte())?;

    let mut entries = Vec::with_capacity(size as usize);
    for _ in 0..size {
        let entry = trace_field!(reader, "toc.entry", read_u32(reader, &TOC_ENTRY_SPEC))?;
        entries.push(u64::from(entry));
    }

    // Prefix sum over the entries as read; offsets[i] = sum(entries[0..i]).
    let mut offsets = Vec::with_capacity(size as usize);
    let mut running: u64 = 0;
    for entry in &entries {
        offsets.push(running);
        running = running
            .checked_add(*entry)
            .ok_or(FrameError::LimitExceeded {
                what: "TOC total size",
                clause: "F.3.3",
                value: u64::MAX,
                limit: u64::MAX,
            })?;
    }

    // Only now is the permutation applied, and it reindexes offsets.
    if let Some(permutation) = &permutation {
        let mut permuted_offsets = Vec::with_capacity(offsets.len());
        for target in permutation {
            let source = usize::try_from(*target).map_err(|_| {
                FrameError::out_of_range("permutation", "F.3.2", u64::from(*target))
            })?;
            let value = offsets.get(source).copied().ok_or_else(|| {
                FrameError::out_of_range("permutation", "F.3.2", u64::from(*target))
            })?;
            permuted_offsets.push(value);
        }
        offsets = permuted_offsets;
    }

    trace_field!(reader, "toc.pad_after_entries", reader.zero_pad_to_byte())?;

    Ok(Toc {
        permuted,
        entries,
        offsets,
        permutation,
    })
}

/// Decodes a permutation of `size` elements whose first `skip` are fixed
/// (18181-1 F.3.2).
///
/// # Errors
///
/// [`FrameError::Entropy`] for a malformed stream, or
/// [`FrameError::FieldOutOfRange`] if a decoded value violates the bounds the
/// clause states (`end <= size - skip`, `lehmer[i] < size - i`).
pub fn read_permutation(
    reader: &mut BitReader<'_>,
    size: u32,
    skip: u32,
    limits: &Limits,
    guard: &mut AllocGuard,
) -> Result<Vec<u32>> {
    let _ = limits;
    if skip > size {
        return Err(FrameError::out_of_range("skip", "F.3.2", u64::from(skip)));
    }

    let mut decoder = SymbolDecoder::open(reader, PERMUTATION_NUM_DIST, guard)?;

    let end = decoder.read_uint(reader, get_context(size))?;
    if end > size - skip {
        return Err(FrameError::out_of_range("end", "F.3.2", u64::from(end)));
    }

    let mut lehmer = vec![0u32; size as usize];
    let mut previous = 0u32;
    for i in skip..(skip + end) {
        let ctx = get_context(if i > skip { previous } else { 0 });
        let value = decoder.read_uint(reader, ctx)?;
        // "this value is strictly less than size - i"
        if value >= size - i {
            return Err(FrameError::out_of_range(
                "lehmer",
                "F.3.2",
                u64::from(value),
            ));
        }
        if let Some(slot) = lehmer.get_mut(i as usize) {
            *slot = value;
        }
        previous = value;
    }

    decoder.finish()?;

    Ok(lehmer_to_permutation(&lehmer))
}

/// Converts a Lehmer code to a permutation (18181-1 F.3.2).
///
/// `temp` starts as `[0, size)`; for each `i`, element `temp[lehmer[i]]` is
/// appended to the result and removed from `temp`.
#[must_use]
pub fn lehmer_to_permutation(lehmer: &[u32]) -> Vec<u32> {
    let size = lehmer.len();
    let mut temp: Vec<u32> = (0..size)
        .map(|v| u32::try_from(v).unwrap_or(u32::MAX))
        .collect();
    let mut permutation = Vec::with_capacity(size);

    for code in lehmer {
        let index = usize::try_from(*code).unwrap_or(usize::MAX);
        if index >= temp.len() {
            // Cannot happen for a validated Lehmer code; take the last
            // element rather than panicking.
            if let Some(last) = temp.pop() {
                permutation.push(last);
            }
            continue;
        }
        permutation.push(temp.remove(index));
    }
    permutation
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testsupport::BitWriter;

    fn read(bytes: &[u8], num_sections: u64) -> Result<Toc> {
        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let mut r = BitReader::new(bytes);
        read_toc(&mut r, num_sections, &limits, &mut guard)
    }

    #[test]
    fn get_context_is_capped_at_seven() {
        // ceil(log2(x + 1)) is the bit length of x.
        assert_eq!(get_context(0), 0);
        assert_eq!(get_context(1), 1);
        assert_eq!(get_context(2), 2);
        assert_eq!(get_context(3), 2);
        assert_eq!(get_context(4), 3);
        assert_eq!(get_context(127), 7);
        assert_eq!(get_context(128), 7, "capped, not 8");
        assert_eq!(get_context(u32::MAX), 7);
        // Every context is a valid index into the eight distributions.
        for x in [0u32, 1, 5, 99, u32::MAX] {
            assert!(get_context(x) < PERMUTATION_NUM_DIST);
        }
    }

    #[test]
    fn single_section_toc_without_permutation() {
        // permuted_toc = 0, pad to byte, one entry of 42 bytes, pad.
        let mut w = BitWriter::new();
        w.bool(false);
        w.pad_to_byte();
        w.u32_field(0, 10, 42);
        w.pad_to_byte();
        let data = w.finish_padded(1);
        let toc = read(&data, 1).expect("valid");

        assert!(!toc.permuted);
        assert_eq!(toc.entries, vec![42]);
        assert_eq!(toc.offsets, vec![0]);
        assert_eq!(toc.total_size(), 42);
        assert_eq!(toc.len(), 1);
        assert!(toc.permutation.is_none());
    }

    #[test]
    fn offsets_are_the_prefix_sum_of_entries() {
        let sizes = [10u32, 20, 5, 100];
        let mut w = BitWriter::new();
        w.bool(false);
        w.pad_to_byte();
        for s in sizes {
            w.u32_field(0, 10, s);
        }
        w.pad_to_byte();
        let data = w.finish_padded(1);
        let toc = read(&data, 4).expect("valid");

        assert_eq!(toc.entries, vec![10, 20, 5, 100]);
        assert_eq!(
            toc.offsets,
            vec![0, 10, 30, 35],
            "offset[i] is the sum of entries before i, not including its own"
        );
        assert_eq!(toc.total_size(), 135);
        assert_eq!(toc.offset_of(3), Some(35));
        assert_eq!(toc.offset_of(4), None);
    }

    #[test]
    fn wide_entry_distributions() {
        // Selector 1 => 1024 + u(14), selector 3 => 4211712 + u(30).
        let mut w = BitWriter::new();
        w.bool(false);
        w.pad_to_byte();
        w.u32_field(1, 14, 100); // 1124
        w.u32_field(3, 30, 5); // 4211717
        w.pad_to_byte();
        let data = w.finish_padded(1);
        let toc = read(&data, 2).expect("valid");

        assert_eq!(toc.entries, vec![1124, 4_211_717]);
        assert_eq!(toc.offsets, vec![0, 1124]);
    }

    #[test]
    fn zero_pad_to_byte_is_applied_before_and_after_entries() {
        // permuted_toc is one bit, so seven padding bits precede the entries.
        let mut w = BitWriter::new();
        w.bool(false);
        w.pad_to_byte();
        w.u32_field(0, 10, 1); // 12 bits, so four padding bits follow
        w.pad_to_byte();
        let data = w.finish_padded(1);

        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let mut r = BitReader::new(&data);
        read_toc(&mut r, 1, &limits, &mut guard).expect("valid");

        assert!(r.is_byte_aligned(), "the TOC ends byte-aligned");
        assert_eq!(r.total_bits_read(), 24, "8 + 12 rounded up to 24");
    }

    #[test]
    fn nonzero_padding_is_rejected() {
        // A 1 bit in the padding after permuted_toc violates B.2.7.
        let data = [0b0000_0010u8, 0x01, 0x00, 0x00];
        assert!(read(&data, 1).is_err());
    }

    #[test]
    fn lehmer_to_permutation_matches_the_worked_procedure() {
        // temp = [0,1,2,3]; lehmer [0,0,0,0] removes the head each time.
        assert_eq!(lehmer_to_permutation(&[0, 0, 0, 0]), vec![0, 1, 2, 3]);

        // lehmer [3,2,1,0]: take temp[3]=3, then temp[2] of [0,1,2] = 2,
        // then temp[1] of [0,1] = 1, then temp[0] of [0] = 0.
        assert_eq!(lehmer_to_permutation(&[3, 2, 1, 0]), vec![3, 2, 1, 0]);

        // lehmer [1,0,0]: take temp[1]=1 from [0,1,2] leaving [0,2];
        // temp[0]=0 leaving [2]; temp[0]=2.
        assert_eq!(lehmer_to_permutation(&[1, 0, 0]), vec![1, 0, 2]);

        assert_eq!(lehmer_to_permutation(&[]), Vec::<u32>::new());
    }

    #[test]
    fn identity_permutation_leaves_offsets_untouched() {
        let offsets = [0u64, 10, 30];
        let permutation = lehmer_to_permutation(&[0, 0, 0]);
        assert_eq!(permutation, vec![0, 1, 2]);

        let permuted: Vec<u64> = permutation
            .iter()
            .map(|i| offsets.get(*i as usize).copied().unwrap_or_default())
            .collect();
        assert_eq!(permuted.as_slice(), offsets.as_slice());
    }

    #[test]
    fn permutation_reindexes_offsets_not_sizes() {
        // Sizes [10, 20, 5] give offsets [0, 10, 30]. Under permutation
        // [2, 0, 1] the result is [30, 0, 10] — which is NOT the prefix sum
        // of the permuted sizes [5, 10, 20] (that would be [0, 5, 15]).
        let offsets = [0u64, 10, 30];
        let permutation = [2u32, 0, 1];
        let permuted: Vec<u64> = permutation
            .iter()
            .map(|i| offsets.get(*i as usize).copied().unwrap_or_default())
            .collect();
        assert_eq!(permuted, vec![30, 0, 10]);

        let permuted_sizes = [5u64, 10, 20];
        let mut wrong = Vec::new();
        let mut running = 0;
        for s in permuted_sizes {
            wrong.push(running);
            running += s;
        }
        assert_ne!(
            permuted, wrong,
            "the two orderings must not be confused: F.3.3 sums first"
        );
    }

    #[test]
    fn zero_sections_rejected() {
        assert!(read(&[0u8; 4], 0).is_err());
    }

    #[test]
    fn section_table_allocation_is_metered() {
        let limits = Limits {
            max_alloc_bytes: 64,
            ..Limits::default()
        };
        let mut guard = AllocGuard::new(&limits);
        let mut r = BitReader::new(&[0u8; 16]);
        let err = read_toc(&mut r, 100_000, &limits, &mut guard).expect_err("must be metered");
        assert!(matches!(err, FrameError::Core(_)));
    }

    #[test]
    fn truncated_toc_errors() {
        assert!(read(&[], 1).is_err());
        assert!(read(&[0x00], 4).is_err());
    }
}
