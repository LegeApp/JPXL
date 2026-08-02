//! Distribution clustering.
//!
//! ISO/IEC 18181-1 C.2.2: the mapping from the caller's pre-clustered context
//! identifiers onto the smaller set of distributions actually carried in the
//! stream.
//!
//! Two encodings exist. The simple one spends a fixed number of bits per
//! context. The general one bootstraps an entire nested symbol decoder over a
//! single distribution and reads the mapping through it, optionally followed
//! by an inverse move-to-front transform.
//!
//! Every path here is **bit-exact** (see `docs/PLAN.md`).

use jpxl_bitstream::BitReader;
use jpxl_core::limits::AllocGuard;

use crate::decoder::SymbolDecoder;
use crate::error::{Result, malformed};

/// Upper bound on cluster indices (18181-1 C.2.2).
pub const MAX_CLUSTERS: usize = 256;

/// Deepest nesting of C.2.2's recursive distribution decoding that this
/// implementation will follow.
///
/// C.2.2 forbids LZ77 in the nested decoder when `num_dist == 2`, which is the
/// case a nested decoder's own LZ77 context would produce, so a conforming
/// stream nests at most one level. This cap is defence in depth against a
/// crafted stream: the recursion is driven by stream contents and would
/// otherwise be unbounded.
pub(crate) const MAX_NESTING_DEPTH: u32 = 4;

/// The context-to-cluster mapping of 18181-1 C.2.2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClusterMap {
    /// One cluster index per pre-clustered context.
    clusters: Vec<u8>,
    /// Number of distinct clusters; every value in `[0, num_clusters)` occurs.
    num_clusters: usize,
}

impl ClusterMap {
    /// Cluster index of a pre-clustered context.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Malformed`](crate::EntropyError::Malformed) if `ctx` is
    /// not a context of this stream.
    pub fn cluster_of(&self, ctx: usize) -> Result<usize> {
        self.clusters
            .get(ctx)
            .map(|&c| usize::from(c))
            .ok_or_else(|| malformed!("C.2.2: context {ctx} is outside this stream"))
    }

    /// Number of post-clustering distributions.
    #[must_use]
    pub const fn num_clusters(&self) -> usize {
        self.num_clusters
    }

    /// Number of pre-clustering contexts.
    #[must_use]
    pub fn num_dist(&self) -> usize {
        self.clusters.len()
    }
}

/// Reads the clustering of `num_dist` contexts (18181-1 C.2.2).
///
/// `depth` guards the recursion described by the clause; callers start at 0.
///
/// # Errors
///
/// [`EntropyError::Malformed`](crate::EntropyError::Malformed) if the mapping
/// is not dense, exceeds [`MAX_CLUSTERS`], or nests too deeply.
pub fn read_cluster_map(
    reader: &mut BitReader<'_>,
    num_dist: usize,
    depth: u32,
    guard: &mut AllocGuard,
) -> Result<ClusterMap> {
    if num_dist == 0 {
        return Err(malformed!("C.2.2: num_dist must be at least 1"));
    }
    // C.2.2: a single context is its own cluster and nothing is read.
    if num_dist == 1 {
        return Ok(ClusterMap {
            clusters: vec![0],
            num_clusters: 1,
        });
    }

    guard.charge(num_dist as u64)?;
    let mut clusters = vec![0u8; num_dist];

    if reader.read_bool()? {
        // Simple clustering: a fixed field width per context.
        let nbits = reader.read_bits(2)?;
        for slot in &mut clusters {
            let value = reader.read_bits(nbits)?;
            *slot = u8::try_from(value)
                .map_err(|_| malformed!("C.2.2: cluster index {value} exceeds 255"))?;
        }
    } else {
        let use_mtf = reader.read_bool()?;
        if depth >= MAX_NESTING_DEPTH {
            return Err(malformed!(
                "C.2.2: distribution decoding nested deeper than {MAX_NESTING_DEPTH} levels"
            ));
        }
        // The nested decoder carries a single distribution. C.2.2 additionally
        // requires that LZ77 be absent from it when num_dist == 2.
        let mut nested =
            SymbolDecoder::open_nested(reader, 1, depth + 1, num_dist == 2, true, guard)?;
        for slot in &mut clusters {
            let value = nested.read_uint(reader, 0)?;
            *slot = u8::try_from(value)
                .map_err(|_| malformed!("C.2.2: cluster index {value} exceeds 255"))?;
        }
        // The nested stream ends here, so C.3.2's terminal-state requirement
        // applies to it.
        nested.finish()?;
        if use_mtf {
            inverse_move_to_front(&mut clusters);
        }
    }

    let num_clusters = clusters
        .iter()
        .map(|&c| usize::from(c) + 1)
        .max()
        .unwrap_or(0);
    if num_clusters > MAX_CLUSTERS {
        return Err(malformed!(
            "C.2.2: {num_clusters} clusters exceeds the maximum of {MAX_CLUSTERS}"
        ));
    }
    // C.2.2 requires the mapping to be dense: every index below num_clusters
    // must be used, so an allocation sized by num_clusters cannot be inflated
    // by a stream that only ever references one of them.
    let mut seen = vec![false; num_clusters];
    for &c in &clusters {
        if let Some(slot) = seen.get_mut(usize::from(c)) {
            *slot = true;
        }
    }
    if let Some(missing) = seen.iter().position(|&s| !s) {
        return Err(malformed!(
            "C.2.2: cluster {missing} of {num_clusters} is never used, so the mapping is not dense"
        ));
    }

    Ok(ClusterMap {
        clusters,
        num_clusters,
    })
}

/// `InverseMoveToFrontTransform` of 18181-1 C.2.2.
///
/// Rewrites each entry through a table that promotes the most recently used
/// value to the front, which is how the encoder turns runs of a repeated
/// cluster index into runs of zero.
pub fn inverse_move_to_front(clusters: &mut [u8]) {
    let mut mtf = [0u8; 256];
    for (i, slot) in mtf.iter_mut().enumerate() {
        // The loop bound matches the array length, so the cast is exact.
        *slot = u8::try_from(i).unwrap_or(u8::MAX);
    }
    for entry in clusters.iter_mut() {
        let index = usize::from(*entry);
        let Some(&value) = mtf.get(index) else {
            continue;
        };
        *entry = value;
        if index != 0 {
            // Move `value` to the front, shifting the prefix up one slot.
            if let Some(prefix) = mtf.get_mut(..=index) {
                prefix.rotate_right(1);
            }
        }
    }
}

#[cfg(test)]
// In test code an out-of-range index is a failed assertion rather than an attack
// surface, and the casts are on values these tests chose themselves. The lints
// exist for the decode paths above.
#[allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]
mod tests {
    use super::*;
    use jpxl_core::limits::Limits;

    #[test]
    fn single_context_reads_nothing() {
        let mut g = AllocGuard::new(&Limits::relaxed());
        let data = [0xFFu8; 4];
        let mut r = BitReader::new(&data);
        let map = read_cluster_map(&mut r, 1, 0, &mut g).expect("trivial map");
        assert_eq!(r.total_bits_read(), 0);
        assert_eq!(map.num_clusters(), 1);
        assert_eq!(map.cluster_of(0).expect("ctx 0"), 0);
        assert!(map.cluster_of(1).is_err());
    }

    #[test]
    fn simple_clustering_reads_fixed_width_fields() {
        // b0 = 1 (is_simple)
        // b1, b2 = u(2) = 2 -> nbits = 2      (b1 = 0, b2 = 1)
        // three contexts, 2 bits each: 0, 1, 1
        //   ctx0: b3, b4 = 0, 0 -> 0
        //   ctx1: b5, b6 = 1, 0 -> 1
        //   ctx2: b7, c0 = 1, 0 -> 1
        // byte0 = b0 | b2 | b5 | b7 = 1 + 4 + 32 + 128 = 165
        let mut g = AllocGuard::new(&Limits::relaxed());
        let data = [165u8, 0x00];
        let mut r = BitReader::new(&data);
        let map = read_cluster_map(&mut r, 3, 0, &mut g).expect("simple map");
        assert_eq!(r.total_bits_read(), 9);
        assert_eq!(map.num_clusters(), 2);
        assert_eq!(map.cluster_of(0).expect("c0"), 0);
        assert_eq!(map.cluster_of(1).expect("c1"), 1);
        assert_eq!(map.cluster_of(2).expect("c2"), 1);
    }

    #[test]
    fn non_dense_mappings_are_rejected() {
        // Same shape as above but the contexts are 0, 2, 2: cluster 1 is never
        // used, so num_clusters would be 3 with a hole.
        //   b0 = 1, nbits = 2, ctx values 0, 2, 2
        //   ctx0: b3, b4 = 0, 0
        //   ctx1: b5, b6 = 0, 1
        //   ctx2: b7, c0 = 0, 1
        // byte0 = b0 | b2 | b6 = 1 + 4 + 64 = 69 ; byte1 = c1 = 1 -> 2
        let mut g = AllocGuard::new(&Limits::relaxed());
        let data = [69u8, 0b0000_0010];
        let mut r = BitReader::new(&data);
        assert!(read_cluster_map(&mut r, 3, 0, &mut g).is_err());
    }

    #[test]
    fn inverse_mtf_of_all_zeroes_is_all_zeroes() {
        // Every entry selects the front of the table, which never changes.
        let mut v = [0u8; 5];
        inverse_move_to_front(&mut v);
        assert_eq!(v, [0; 5]);
    }

    #[test]
    fn inverse_mtf_matches_a_hand_traced_example() {
        // Table starts as identity 0,1,2,...
        // input 2 -> output mtf[2] = 2, then 2 moves to front: 2,0,1,3,...
        // input 0 -> output mtf[0] = 2, index 0 so no move:    2,0,1,3,...
        // input 2 -> output mtf[2] = 1, then move to front:    1,2,0,3,...
        // input 1 -> output mtf[1] = 2, then move to front:    2,1,0,3,...
        // input 3 -> output mtf[3] = 3, then move to front:    3,2,1,0,...
        let mut v = [2u8, 0, 2, 1, 3];
        inverse_move_to_front(&mut v);
        assert_eq!(v, [2, 2, 1, 2, 3]);
    }

    #[test]
    fn inverse_mtf_is_the_inverse_of_the_forward_transform() {
        // Round-trip against a straightforward forward MTF encoder.
        let original: Vec<u8> = vec![5, 5, 3, 0, 3, 7, 7, 7, 1, 0, 5];
        let mut table: Vec<u8> = (0..=255u8).collect();
        let mut encoded = Vec::new();
        for &value in &original {
            let index = table.iter().position(|&t| t == value).expect("present");
            encoded.push(u8::try_from(index).expect("fits"));
            if index != 0 {
                table[..=index].rotate_right(1);
            }
        }
        let mut decoded = encoded.clone();
        inverse_move_to_front(&mut decoded);
        assert_eq!(decoded, original);
    }
}
