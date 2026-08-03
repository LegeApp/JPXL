//! HF block context decoding and the HF preset count (18181-1 I.2.2, I.2.6).
//!
//! I.2.2 describes the context model that I.4 uses to pick a distribution for
//! every HF symbol. The model is a lookup table `block_ctx_map` indexed by a
//! composite of the channel, the quantization-factor band and three LF bands:
//!
//! ```text
//! /* qf_thresholds starts empty; lf_thresholds is 3 empty vectors */
//! if (u(1))
//!     block_ctx_map = { 39 default entries };
//! else {
//!     for (i = 0; i < 3; i++) {
//!         nb_lf_thr[i] = u(4);
//!         for (j = 0; j < nb_lf_thr[i]; j++)
//!             lf_thresholds[i].push_back(UnpackSigned(ReadThreshold()));
//!     }
//!     nb_qf_thr = u(4);
//!     for (i = 0; i < nb_qf_thr; i++)
//!         qf_thresholds.push_back(1 + U32(u(2), 4 + u(3), 12 + u(5), 44 + u(8)));
//!     bsize = 39 * (nb_qf_thr + 1)
//!               * (nb_lf_thr[0] + 1) * (nb_lf_thr[1] + 1) * (nb_lf_thr[2] + 1);
//!     block_ctx_map = ReadBlockCtxMap();   /* C.2.2, num_dist = bsize */
//! }
//! ```
//!
//! with `ReadThreshold()` denoting `U32(u(4), 16 + u(8), 272 + u(16),
//! 65808 + u(32))`. The clause states two constraints on a conforming stream:
//! `bsize <= 39 * 64` and the resulting cluster count is at most 16. Both are
//! enforced here — they are what bounds an attacker-controlled allocation and
//! what bounds I.4's context arithmetic.
//!
//! # The 39
//!
//! 39 = 3 channels x 13 shape classes. I.4 builds its index as
//! `(c < 2 ? c ^ 1 : 2) * 13 + s`, so the map's row stride is 13 and the
//! default map below is three 13-entry rows: one for Y, one for X and one for
//! B, with X and B sharing the same row values.
//!
//! # I.2.6
//!
//! `num_hf_presets = u(ceil(log2(num_groups))) + 1`. The field width depends on
//! the frame geometry, which is why it cannot be a constant.

// The indices below are all loop bounds over locally sized arrays; the two
// stream-derived indices (`lf_thresholds` and the map) are `get`-checked.
#![allow(clippy::indexing_slicing)]

use jpxl_bitstream::trace_field;
use jpxl_bitstream::{BitReader, U32Dist, U32Spec, read_u32};
use jpxl_core::limits::AllocGuard;
use jpxl_entropy::read_cluster_map;

use crate::error::{DecodeError, Result};

/// Number of block context rows before any threshold refinement: 3 channels
/// times the 13 shape classes of I.4.
pub const BLOCK_CTX_ROWS: usize = 39;

/// I.2.2's upper bound on `bsize`, i.e. on the number of pre-clustered
/// contexts a conforming stream may declare.
pub const MAX_BSIZE: usize = BLOCK_CTX_ROWS * 64;

/// I.2.2's upper bound on the number of clusters, i.e. on `nb_block_ctx`.
pub const MAX_NB_BLOCK_CTX: usize = 16;

/// The default `block_ctx_map` of I.2.2, used when the leading `u(1)` is set.
///
/// Verified digit by digit against the printed Table (page 60 of the scan);
/// three rows of 13, maximum value 14, so `nb_block_ctx` is 15.
pub const DEFAULT_BLOCK_CTX_MAP: [u8; BLOCK_CTX_ROWS] = [
    0, 1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 6, 6, //
    7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14, //
    7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14,
];

/// `ReadThreshold()`: `U32(u(4), 16 + u(8), 272 + u(16), 65808 + u(32))`.
const THRESHOLD_SPEC: U32Spec = U32Spec::new([
    U32Dist::bits(4),
    U32Dist::BitsOffset {
        bits: 8,
        offset: 16,
    },
    U32Dist::BitsOffset {
        bits: 16,
        offset: 272,
    },
    U32Dist::BitsOffset {
        bits: 32,
        offset: 65808,
    },
]);

/// The `U32(u(2), 4 + u(3), 12 + u(5), 44 + u(8))` of the `qf_thresholds` loop.
const QF_THRESHOLD_SPEC: U32Spec = U32Spec::new([
    U32Dist::bits(2),
    U32Dist::BitsOffset { bits: 3, offset: 4 },
    U32Dist::BitsOffset {
        bits: 5,
        offset: 12,
    },
    U32Dist::BitsOffset {
        bits: 8,
        offset: 44,
    },
]);

/// `UnpackSigned(u)`: `u / 2` if even, `-(u + 1) / 2` if odd.
///
/// Exact for every `u32`: the even branch is at most `2^31 - 1` and the odd
/// branch at least `-2^31`, so no threshold can overflow `i32`.
const fn unpack_signed(u: u32) -> i32 {
    if u.is_multiple_of(2) {
        0i32.wrapping_add_unsigned(u / 2)
    } else {
        0i32.wrapping_sub_unsigned(u / 2 + 1)
    }
}

/// The HF block context model of 18181-1 I.2.2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfBlockContext {
    /// Per-channel LF thresholds, in the clause's `i = 0, 1, 2` order.
    ///
    /// Note that I.4 walks these in the order 0, 2, 1 when composing its index;
    /// the storage here is the read order, not the walk order.
    lf_thresholds: [Vec<i32>; 3],
    /// Quantization-factor thresholds.
    qf_thresholds: Vec<u32>,
    /// The clustering map, of length `bsize`.
    block_ctx_map: Vec<u8>,
    /// `nb_block_ctx`: one more than the largest entry of the map.
    nb_block_ctx: usize,
}

impl Default for HfBlockContext {
    /// The model the leading `u(1)` selects: the default map and no thresholds.
    fn default() -> Self {
        Self {
            lf_thresholds: [Vec::new(), Vec::new(), Vec::new()],
            qf_thresholds: Vec::new(),
            block_ctx_map: DEFAULT_BLOCK_CTX_MAP.to_vec(),
            nb_block_ctx: 15,
        }
    }
}

impl HfBlockContext {
    /// The LF thresholds of channel `i` in the clause's numbering.
    ///
    /// Returns an empty slice for `i > 2` rather than panicking.
    #[must_use]
    pub fn lf_thresholds(&self, i: usize) -> &[i32] {
        self.lf_thresholds.get(i).map_or(&[], Vec::as_slice)
    }

    /// The quantization-factor thresholds.
    #[must_use]
    pub fn qf_thresholds(&self) -> &[u32] {
        &self.qf_thresholds
    }

    /// The clustering map itself, of length [`HfBlockContext::bsize`].
    #[must_use]
    pub fn map(&self) -> &[u8] {
        &self.block_ctx_map
    }

    /// `bsize`: the number of pre-clustered contexts.
    #[must_use]
    pub fn bsize(&self) -> usize {
        self.block_ctx_map.len()
    }

    /// `nb_block_ctx`: the number of distinct block contexts, at most
    /// [`MAX_NB_BLOCK_CTX`].
    #[must_use]
    pub const fn nb_block_ctx(&self) -> usize {
        self.nb_block_ctx
    }

    /// Maps a composite index (I.4's `idx + lf_idx`) through the map.
    ///
    /// # Errors
    ///
    /// [`DecodeError::FieldOutOfRange`] if `index` is not below `bsize`. I.4
    /// can only produce indices below `bsize` from a well-formed stream, so
    /// this is an assertion of that arithmetic rather than a stream check.
    pub fn block_context(&self, index: usize) -> Result<usize> {
        self.block_ctx_map
            .get(index)
            .map(|&c| usize::from(c))
            .ok_or_else(|| {
                DecodeError::out_of_range(
                    "block_ctx_map index",
                    "I.2.2",
                    u64::try_from(index).unwrap_or(u64::MAX),
                )
            })
    }
}

/// Reads the HF block context model (18181-1 I.2.2).
///
/// # Errors
///
/// * [`DecodeError::Bitstream`] on truncation.
/// * [`DecodeError::Entropy`] if the C.2.2 clustering map is malformed.
/// * [`DecodeError::FieldOutOfRange`] if `bsize` exceeds [`MAX_BSIZE`] or the
///   cluster count exceeds [`MAX_NB_BLOCK_CTX`]. Both are stated by the clause;
///   rejecting is what keeps `bsize` from inflating the C.2.2 allocation and
///   keeps I.4's context arithmetic inside its proved bounds.
pub fn read_hf_block_context(
    reader: &mut BitReader<'_>,
    guard: &mut AllocGuard,
) -> Result<HfBlockContext> {
    let use_default = trace_field!(reader, "hf_block_ctx.use_default", reader.read_bool())?;
    if use_default {
        return Ok(HfBlockContext::default());
    }

    let mut lf_thresholds: [Vec<i32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut nb_lf_thr = [0usize; 3];
    for i in 0..3 {
        let raw_count = trace_field!(reader, "hf_block_ctx.nb_lf_thr", reader.read_bits(4))?;
        let count = usize::try_from(raw_count).unwrap_or(usize::MAX);
        nb_lf_thr[i] = count;
        // At most 15 thresholds per channel; the guard still meters the bytes
        // so that a stream cannot repeat this in many bundles for free.
        guard.charge(as_u64(count) * 4)?;
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let raw = trace_field!(
                reader,
                "hf_block_ctx.lf_threshold",
                read_u32(reader, &THRESHOLD_SPEC)
            )?;
            out.push(unpack_signed(raw));
        }
        lf_thresholds[i] = out;
    }

    let raw_qf_count = trace_field!(reader, "hf_block_ctx.nb_qf_thr", reader.read_bits(4))?;
    let nb_qf_thr = usize::try_from(raw_qf_count).unwrap_or(usize::MAX);
    guard.charge(as_u64(nb_qf_thr) * 4)?;
    let mut qf_thresholds = Vec::with_capacity(nb_qf_thr);
    for _ in 0..nb_qf_thr {
        let raw = trace_field!(
            reader,
            "hf_block_ctx.qf_threshold",
            read_u32(reader, &QF_THRESHOLD_SPEC)
        )?;
        qf_thresholds.push(raw.saturating_add(1));
    }

    // bsize = 39 * (nb_qf_thr + 1) * prod(nb_lf_thr[i] + 1). Each factor is at
    // most 16, so the product cannot overflow usize, but the clause caps it at
    // 39 * 64 and a larger value is a malformed stream.
    let factors = (nb_qf_thr + 1) * (nb_lf_thr[0] + 1) * (nb_lf_thr[1] + 1) * (nb_lf_thr[2] + 1);
    let bsize = BLOCK_CTX_ROWS * factors;
    if bsize > MAX_BSIZE {
        return Err(DecodeError::out_of_range("bsize", "I.2.2", as_u64(bsize)));
    }

    let clusters = read_cluster_map(reader, bsize, 0, guard)?;
    if clusters.num_clusters() > MAX_NB_BLOCK_CTX {
        return Err(DecodeError::out_of_range(
            "num_clusters",
            "I.2.2",
            as_u64(clusters.num_clusters()),
        ));
    }

    guard.charge(as_u64(bsize))?;
    let mut block_ctx_map = Vec::with_capacity(bsize);
    for ctx in 0..bsize {
        let cluster = clusters.cluster_of(ctx)?;
        block_ctx_map.push(u8::try_from(cluster).map_err(|_| {
            DecodeError::out_of_range("block_ctx_map entry", "I.2.2", as_u64(cluster))
        })?);
    }

    Ok(HfBlockContext {
        lf_thresholds,
        qf_thresholds,
        block_ctx_map,
        nb_block_ctx: clusters.num_clusters(),
    })
}

// ---------------------------------------------------------------------------
// I.2.6 — number of HF decoding presets
// ---------------------------------------------------------------------------

/// Reads `num_hf_presets` (18181-1 I.2.6).
///
/// The field is `u(ceil(log2(num_groups))) + 1`, so it is zero bits wide for a
/// single-group frame and `num_hf_presets` is then necessarily 1.
///
/// # Errors
///
/// [`DecodeError::Bitstream`] on truncation, or
/// [`DecodeError::FieldOutOfRange`] if `num_groups` is zero — a frame always
/// has at least one group, and `log2(0)` has no meaning.
pub fn read_num_hf_presets(reader: &mut BitReader<'_>, num_groups: u64) -> Result<u32> {
    if num_groups == 0 {
        return Err(DecodeError::out_of_range("num_groups", "I.2.6", 0));
    }
    let bits = ceil_log2(num_groups);
    let raw = trace_field!(reader, "hf_global.num_hf_presets", reader.read_bits(bits))?;
    Ok(raw.saturating_add(1))
}

/// Widening for error payloads and guard charges; saturates rather than
/// wrapping so a 32-bit host cannot report a misleading value.
fn as_u64(v: usize) -> u64 {
    u64::try_from(v).unwrap_or(u64::MAX)
}

/// `ceil(log2(n))` for `n >= 1`: the number of bits needed to index `n` items.
///
/// `ceil_log2(1) == 0`, which is why a single-group frame reads no bits at all.
const fn ceil_log2(n: u64) -> u32 {
    if n <= 1 {
        return 0;
    }
    u64::BITS - (n - 1).leading_zeros()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testsupport::BitWriter;
    use jpxl_core::limits::Limits;

    fn guard() -> AllocGuard {
        AllocGuard::new(&LIMITS)
    }

    static LIMITS: Limits = Limits::new();

    #[test]
    fn default_map_has_39_entries_and_15_contexts() {
        // The self-check the clause hands us: bsize with all threshold vectors
        // empty is 39 * 1 * 1 * 1 * 1 = 39, and the printed map's maximum is
        // 14. A transcription that dropped or duplicated an entry fails here.
        let ctx = HfBlockContext::default();
        assert_eq!(ctx.bsize(), 39);
        assert_eq!(ctx.bsize(), BLOCK_CTX_ROWS);
        assert_eq!(ctx.map().len(), 39);
        let max = ctx.map().iter().copied().max().expect("non-empty");
        assert_eq!(max, 14);
        assert_eq!(ctx.nb_block_ctx(), 15);
        assert_eq!(ctx.nb_block_ctx(), usize::from(max) + 1);
    }

    #[test]
    fn default_map_is_dense_and_three_rows_of_thirteen() {
        // Density: C.2.2 requires every cluster below the maximum to occur, so
        // an entry mistyped upwards would leave a hole. Row structure: I.4's
        // index is (channel) * 13 + shape, and the clause prints the X and B
        // rows identically.
        let ctx = HfBlockContext::default();
        let mut seen = [false; 15];
        for &c in ctx.map() {
            seen[usize::from(c)] = true;
        }
        assert!(seen.iter().all(|&s| s), "default map is not dense");

        let map = ctx.map();
        assert_eq!(&map[0..13], &[0, 1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 6, 6]);
        assert_eq!(&map[13..26], &map[26..39]);
    }

    #[test]
    fn default_path_is_one_bit() {
        let mut w = BitWriter::new();
        w.bool(true);
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let ctx = read_hf_block_context(&mut r, &mut guard()).expect("valid");
        assert_eq!(r.total_bits_read(), 1);
        assert_eq!(ctx, HfBlockContext::default());
    }

    #[test]
    fn explicit_empty_thresholds_reads_the_five_counts() {
        // 1 (not default) + 3 * u(4) + u(4) = 17 bits of counts, then C.2.2
        // with num_dist = 39. The simple clustering path is 1 + 2 + 39 * nbits.
        // Proves the count fields precede the map and that bsize is 39 when
        // every count is zero.
        let mut w = BitWriter::new();
        w.bool(false);
        w.u(4, 0).u(4, 0).u(4, 0); // nb_lf_thr
        w.u(4, 0); // nb_qf_thr
        w.bool(true).u(2, 0); // C.2.2 simple clustering, nbits = 0
        let expected = w.bit_len();
        let data = w.finish_padded(2);
        let mut r = BitReader::new(&data);
        let ctx = read_hf_block_context(&mut r, &mut guard()).expect("valid");
        assert_eq!(r.total_bits_read(), expected);
        assert_eq!(r.total_bits_read(), 1 + 4 * 4 + 1 + 2);
        assert_eq!(ctx.bsize(), 39);
        // Every context clusters to 0, so there is exactly one block context.
        assert_eq!(ctx.nb_block_ctx(), 1);
        assert!(ctx.qf_thresholds().is_empty());
        assert!(ctx.lf_thresholds(0).is_empty());
    }

    #[test]
    fn thresholds_are_read_in_clause_order_and_unpacked() {
        // nb_lf_thr = {1, 0, 2}, nb_qf_thr = 1. Proves the nesting of the j
        // loop inside the i loop (the LaTeX drops the braces there) and the
        // UnpackSigned applied to the LF thresholds but not the QF ones.
        let mut w = BitWriter::new();
        w.bool(false);
        w.u(4, 1);
        w.u32_field(0, 4, 3); // ReadThreshold -> 3 -> UnpackSigned -> -2
        w.u(4, 0);
        w.u(4, 2);
        w.u32_field(0, 4, 4); // -> 4 -> 2
        w.u32_field(1, 8, 0); // -> 16 -> 8
        w.u(4, 1);
        w.u32_field(0, 2, 3); // -> 3, plus 1 -> 4
        // C.2.2 with num_dist = 39 * 2 * 2 * 1 * 3 = 468: simple clustering,
        // nbits = 0 so every context maps to cluster 0.
        w.bool(true).u(2, 0);
        let expected = w.bit_len();
        let data = w.finish_padded(2);
        let mut r = BitReader::new(&data);
        let ctx = read_hf_block_context(&mut r, &mut guard()).expect("valid");

        assert_eq!(r.total_bits_read(), expected);
        assert_eq!(ctx.lf_thresholds(0), &[-2]);
        assert_eq!(ctx.lf_thresholds(1), &[] as &[i32]);
        assert_eq!(ctx.lf_thresholds(2), &[2, 8]);
        assert_eq!(ctx.qf_thresholds(), &[4]);
        // 39 * (nb_qf_thr + 1) * (1 + 1) * (0 + 1) * (2 + 1) = 39 * 12.
        assert_eq!(ctx.bsize(), 468);
    }

    #[test]
    fn unpack_signed_matches_the_convention() {
        assert_eq!(unpack_signed(0), 0);
        assert_eq!(unpack_signed(1), -1);
        assert_eq!(unpack_signed(2), 1);
        assert_eq!(unpack_signed(3), -2);
        // Exactness at the extremes: no i32 overflow for any u32 input.
        assert_eq!(unpack_signed(u32::MAX), -2_147_483_648);
        assert_eq!(unpack_signed(u32::MAX - 1), 2_147_483_647);
    }

    #[test]
    fn oversized_bsize_is_rejected() {
        // nb_lf_thr = {15, 15, 15}, nb_qf_thr = 15 gives 39 * 16^4, far above
        // the clause's 39 * 64. Rejected before the C.2.2 allocation, so the
        // thresholds themselves are the only memory a hostile stream buys.
        let mut w = BitWriter::new();
        w.bool(false);
        for _ in 0..3 {
            w.u(4, 15);
            for _ in 0..15 {
                w.u32_field(0, 4, 0);
            }
        }
        w.u(4, 15);
        for _ in 0..15 {
            w.u32_field(0, 2, 0);
        }
        let data = w.finish_padded(4);
        let mut r = BitReader::new(&data);
        let err = read_hf_block_context(&mut r, &mut guard()).expect_err("bsize is too large");
        assert!(err.to_string().contains("bsize"), "{err}");
    }

    #[test]
    fn simple_clustering_cannot_reach_the_cluster_cap() {
        // C.2.2's simple path is `nbits = u(2)` then `u(nbits)` per context, so
        // its largest cluster index is 7 and it can never trip I.2.2's cap of
        // 16. Documenting that here records *why* there is no hand-built
        // bitstream test for the cap: only C.2.2's nested-decoder path can
        // produce more than eight clusters, and that path needs an entropy
        // encoder to construct. The cap itself is still enforced above, for the
        // streams that can reach it.
        let mut w = BitWriter::new();
        w.bool(false);
        w.u(4, 0).u(4, 0).u(4, 0).u(4, 0);
        w.bool(true).u(2, 3); // simple clustering, nbits = 3
        for i in 0..39u32 {
            w.u(3, i % 8);
        }
        let data = w.finish_padded(4);
        let mut r = BitReader::new(&data);
        let ctx = read_hf_block_context(&mut r, &mut guard()).expect("valid");
        assert_eq!(ctx.nb_block_ctx(), 8);
        assert!(ctx.nb_block_ctx() <= MAX_NB_BLOCK_CTX);
    }

    #[test]
    fn truncated_after_the_counts_errors() {
        let mut w = BitWriter::new();
        w.bool(false).u(4, 3);
        let data = w.finish();
        let mut r = BitReader::new(&data);
        assert!(read_hf_block_context(&mut r, &mut guard()).is_err());
    }

    #[test]
    fn block_context_lookup_is_bounds_checked() {
        let ctx = HfBlockContext::default();
        assert_eq!(ctx.block_context(0).expect("in range"), 0);
        assert_eq!(ctx.block_context(38).expect("in range"), 14);
        assert!(ctx.block_context(39).is_err());
    }

    #[test]
    fn num_hf_presets_field_width_follows_num_groups() {
        // Proves ceil(log2(num_groups)) and the +1: a single-group frame reads
        // nothing, and 5 groups read 3 bits.
        let mut w = BitWriter::new();
        let data = w.finish_padded(2);
        let mut r = BitReader::new(&data);
        assert_eq!(read_num_hf_presets(&mut r, 1).expect("valid"), 1);
        assert_eq!(r.total_bits_read(), 0);

        w = BitWriter::new();
        w.u(3, 5);
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        assert_eq!(read_num_hf_presets(&mut r, 5).expect("valid"), 6);
        assert_eq!(r.total_bits_read(), 3);
    }

    #[test]
    fn ceil_log2_is_exact_at_the_powers_of_two() {
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
        assert_eq!(ceil_log2(256), 8);
        assert_eq!(ceil_log2(257), 9);
    }

    #[test]
    fn num_hf_presets_rejects_zero_groups() {
        let data = [0u8; 2];
        let mut r = BitReader::new(&data);
        assert!(read_num_hf_presets(&mut r, 0).is_err());
    }
}
