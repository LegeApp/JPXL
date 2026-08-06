//! Context map construction and emission — the write side of 18181-1 C.2.2.
//!
//! A [`ContextMap`] sends each pre-clustered context to the distribution that
//! actually codes it. C.2.2 offers two encodings:
//!
//! * the **simple** one spends a fixed `u(nbits)` per context, with `nbits`
//!   itself a `u(2)`, so it can only express cluster indices below 8;
//! * the **general** one bootstraps a whole nested symbol decoder over a single
//!   distribution and reads the map through it, optionally undoing a
//!   move-to-front transform afterwards.
//!
//! [`ContextMapForm::Auto`] measures both (and, inside the general form, both
//! move-to-front settings and both entropy backends) and keeps the shortest.
//! Every form yields the same map, so the choice is purely a density decision.
//!
//! # Move-to-front
//!
//! C.2.2's decoder applies an *inverse* move-to-front transform, so the encoder
//! applies the forward one: each entry is replaced by its position in a table
//! that keeps the most recently used value at the front. Runs of one cluster
//! collapse to runs of zero, which is what makes the nested distribution
//! cheap. The decoder's transform is already pinned by its own round-trip test.
//!
//! Every path here is **bit-exact** (see `docs/PLAN.md`).

use jpxl_bitstream::BitWriter;

use crate::dist::MAX_CLUSTERS;
use crate::error::{Result, encode_error};
use crate::hybrid::{HybridUintConfig, bit_width};

use super::stream::{CodingMode, EncoderPlan, EntropyTables, SymbolEncoder, TokenCensus};

/// Which C.2.2 encoding to use for a context map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ContextMapForm {
    /// Measure every applicable form and keep the shortest.
    #[default]
    Auto,
    /// Force the fixed-width form; fails if a cluster index exceeds 7.
    Simple,
    /// Force the nested form, with the move-to-front transform as given.
    Nested {
        /// Whether to apply the forward move-to-front transform.
        use_mtf: bool,
    },
}

/// The context-to-cluster mapping of 18181-1 C.2.2, on the write side.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextMap {
    clusters: Vec<u8>,
    num_clusters: usize,
}

impl ContextMap {
    /// The identity map: every context is its own cluster.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Encode`](crate::EntropyError::Encode) if `num_contexts`
    /// is zero or exceeds [`MAX_CLUSTERS`].
    pub fn identity(num_contexts: usize) -> Result<Self> {
        let clusters: Vec<u8> = (0..num_contexts)
            .map(|i| {
                u8::try_from(i).map_err(|_| {
                    encode_error!("C.2.2: {num_contexts} contexts exceed the cluster range")
                })
            })
            .collect::<Result<_>>()?;
        Self::new(clusters)
    }

    /// A caller-supplied clustering, one cluster index per context.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Encode`](crate::EntropyError::Encode) if the map is
    /// empty, exceeds [`MAX_CLUSTERS`], or is not dense — C.2.2 requires every
    /// index below `num_clusters` to occur.
    pub fn new(clusters: Vec<u8>) -> Result<Self> {
        if clusters.is_empty() {
            return Err(encode_error!("C.2.2: num_dist must be at least 1"));
        }
        let num_clusters = clusters
            .iter()
            .map(|&c| usize::from(c) + 1)
            .max()
            .unwrap_or(0);
        if num_clusters > MAX_CLUSTERS {
            return Err(encode_error!(
                "C.2.2: {num_clusters} clusters exceed the maximum of {MAX_CLUSTERS}"
            ));
        }
        let mut seen = vec![false; num_clusters];
        for &c in &clusters {
            if let Some(slot) = seen.get_mut(usize::from(c)) {
                *slot = true;
            }
        }
        if let Some(missing) = seen.iter().position(|&s| !s) {
            return Err(encode_error!(
                "C.2.2: cluster {missing} is never used, so the mapping is not dense"
            ));
        }
        Ok(Self {
            clusters,
            num_clusters,
        })
    }

    /// Cluster of a pre-clustered context.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Encode`](crate::EntropyError::Encode) if `ctx` is not a
    /// context of this map.
    pub fn cluster_of(&self, ctx: usize) -> Result<usize> {
        self.clusters
            .get(ctx)
            .map(|&c| usize::from(c))
            .ok_or_else(|| encode_error!("C.2.2: context {ctx} is outside this stream"))
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

    /// The raw mapping.
    #[must_use]
    pub fn entries(&self) -> &[u8] {
        &self.clusters
    }

    /// Writes the mapping (18181-1 C.2.2).
    ///
    /// # Errors
    ///
    /// [`EntropyError::Encode`](crate::EntropyError::Encode) if the requested
    /// form cannot express this map, or a bitstream error.
    pub fn write(&self, w: &mut BitWriter, form: ContextMapForm) -> Result<()> {
        // C.2.2: a single context is its own cluster and nothing is written.
        if self.num_dist() == 1 {
            return Ok(());
        }
        match form {
            ContextMapForm::Simple => self.write_simple(w),
            ContextMapForm::Nested { use_mtf } => self.write_nested(w, use_mtf),
            ContextMapForm::Auto => {
                let mut best: Option<(ContextMapForm, u64)> = None;
                for candidate in [
                    ContextMapForm::Simple,
                    ContextMapForm::Nested { use_mtf: false },
                    ContextMapForm::Nested { use_mtf: true },
                ] {
                    let mut probe = BitWriter::new();
                    if self.write(&mut probe, candidate).is_ok() {
                        let len = probe.bit_len();
                        if best.is_none_or(|(_, best_len)| len < best_len) {
                            best = Some((candidate, len));
                        }
                    }
                }
                let (candidate, _) = best.ok_or_else(|| {
                    encode_error!("C.2.2: no encoding can express this context map")
                })?;
                self.write(w, candidate)
            }
        }
    }

    fn write_simple(&self, w: &mut BitWriter) -> Result<()> {
        let max = self.clusters.iter().copied().max().unwrap_or(0);
        let nbits = bit_width(u32::from(max));
        if nbits > 3 {
            return Err(encode_error!(
                "C.2.2: the simple form cannot express cluster index {max}"
            ));
        }
        w.write_bool(true);
        w.write_bits(2, nbits)?;
        for &cluster in &self.clusters {
            w.write_bits(nbits, u32::from(cluster))?;
        }
        Ok(())
    }

    fn write_nested(&self, w: &mut BitWriter, use_mtf: bool) -> Result<()> {
        let mut values = self.clusters.clone();
        if use_mtf {
            move_to_front(&mut values);
        }

        w.write_bool(false);
        w.write_bool(use_mtf);

        // The nested bundle carries exactly one distribution over the cluster
        // indices, so its own context map is trivial and it never recurses.
        let max = values.iter().copied().max().unwrap_or(0);
        let log_alphabet_size = bit_width(u32::from(max)).max(5);
        if log_alphabet_size > 8 {
            return Err(encode_error!(
                "C.2.2: cluster index {max} does not fit an ANS alphabet"
            ));
        }
        // `split_exponent == log_alphabet_size` makes every value its own
        // token, so the nested stream carries no raw extra bits at all.
        let config = HybridUintConfig::new(log_alphabet_size, 0, 0)?;

        let mut census = TokenCensus::new(1)?;
        for &value in &values {
            census.record(0, u32::from(value))?;
        }

        // Both backends are legal here; take whichever is shorter.
        let mut best: Option<(CodingMode, u64)> = None;
        for mode in [CodingMode::Ans, CodingMode::Prefix] {
            let mut probe = BitWriter::new();
            if write_nested_payload(
                &mut probe,
                &census,
                &values,
                mode,
                config,
                log_alphabet_size,
            )
            .is_ok()
            {
                let len = probe.bit_len();
                if best.is_none_or(|(_, best_len)| len < best_len) {
                    best = Some((mode, len));
                }
            }
        }
        let (mode, _) =
            best.ok_or_else(|| encode_error!("C.2.2: the nested distribution cannot be encoded"))?;
        write_nested_payload(w, &census, &values, mode, config, log_alphabet_size)
    }
}

/// Writes the nested bundle and its symbols (the recursive half of C.2.2).
fn write_nested_payload(
    w: &mut BitWriter,
    census: &TokenCensus,
    values: &[u8],
    mode: CodingMode,
    config: HybridUintConfig,
    log_alphabet_size: u32,
) -> Result<()> {
    let plan = EncoderPlan {
        mode,
        context_map: ContextMap::identity(1)?,
        context_map_form: ContextMapForm::Auto,
        configs: vec![config],
        log_alphabet_size: match mode {
            CodingMode::Ans => Some(log_alphabet_size),
            CodingMode::Prefix => None,
        },
        lz77: None,
    };
    let tables = EntropyTables::build(&plan, census)?;
    tables.write_bundle(w)?;
    let mut encoder = SymbolEncoder::new(&tables);
    for &value in values {
        encoder.push_uint(0, u32::from(value))?;
    }
    encoder.write_stream(w)
}

/// The forward move-to-front transform, i.e. the inverse of
/// `InverseMoveToFrontTransform` in 18181-1 C.2.2.
pub fn move_to_front(values: &mut [u8]) {
    let mut table = [0u8; 256];
    for (i, slot) in table.iter_mut().enumerate() {
        // The loop bound matches the array length, so the conversion is exact.
        *slot = u8::try_from(i).unwrap_or(u8::MAX);
    }
    for entry in values.iter_mut() {
        let Some(index) = table.iter().position(|&t| t == *entry) else {
            continue;
        };
        // `index` is a position in a 256-entry table.
        *entry = u8::try_from(index).unwrap_or(u8::MAX);
        if index != 0
            && let Some(prefix) = table.get_mut(..=index)
        {
            prefix.rotate_right(1);
        }
    }
}

#[cfg(test)]
// Test code: an out-of-range index is a failed assertion rather than an attack
// surface, and the casts are on values these tests chose themselves.
#[allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]
mod tests {
    use super::*;
    use crate::dist::{inverse_move_to_front, read_cluster_map};
    use jpxl_bitstream::BitReader;
    use jpxl_core::limits::{AllocGuard, Limits};

    /// Writes a map in the given form and reads it back with the decoder.
    fn round_trip(entries: &[u8], form: ContextMapForm) -> u64 {
        let map = ContextMap::new(entries.to_vec()).expect("dense");
        let mut w = BitWriter::new();
        map.write(&mut w, form).expect("writes");
        let bits = w.bit_len();
        let bytes = w.into_bytes();

        let mut guard = AllocGuard::new(&Limits::relaxed());
        let mut r = BitReader::new(&bytes);
        let read = read_cluster_map(&mut r, entries.len(), 0, &mut guard).expect("decoder reads");
        for (ctx, &cluster) in entries.iter().enumerate() {
            assert_eq!(read.cluster_of(ctx).expect("ctx"), usize::from(cluster));
        }
        assert_eq!(read.num_clusters(), map.num_clusters());
        assert_eq!(r.total_bits_read(), bits, "no bits left unread");
        bits
    }

    #[test]
    fn a_single_context_writes_nothing() {
        let map = ContextMap::identity(1).expect("valid");
        let mut w = BitWriter::new();
        map.write(&mut w, ContextMapForm::Auto).expect("writes");
        assert_eq!(w.bit_len(), 0);
    }

    #[test]
    fn identity_maps_round_trip_in_every_form() {
        for n in [2usize, 3, 6, 8] {
            let entries: Vec<u8> = (0..n as u8).collect();
            round_trip(&entries, ContextMapForm::Auto);
            round_trip(&entries, ContextMapForm::Simple);
            round_trip(&entries, ContextMapForm::Nested { use_mtf: false });
            round_trip(&entries, ContextMapForm::Nested { use_mtf: true });
        }
    }

    #[test]
    fn wide_maps_need_the_nested_form() {
        // Cluster 8 does not fit the simple form's u(2) width field.
        let entries: Vec<u8> = (0..40u8).map(|i| i % 12).collect();
        let map = ContextMap::new(entries.clone()).expect("dense");
        assert!(
            map.write(&mut BitWriter::new(), ContextMapForm::Simple)
                .is_err()
        );
        round_trip(&entries, ContextMapForm::Auto);
        round_trip(&entries, ContextMapForm::Nested { use_mtf: true });
        round_trip(&entries, ContextMapForm::Nested { use_mtf: false });
    }

    #[test]
    fn a_two_context_map_uses_the_lz77_free_nested_decoder() {
        // C.2.2's num_dist == 2 rule: the nested decoder must not enable LZ77.
        // This encoder never does, so the constrained path must still decode.
        round_trip(&[0, 1], ContextMapForm::Nested { use_mtf: false });
        round_trip(&[0, 1], ContextMapForm::Nested { use_mtf: true });
    }

    #[test]
    fn repetitive_maps_are_smaller_with_move_to_front() {
        // 64 contexts in long runs: MTF turns the runs into zeroes.
        let entries: Vec<u8> = (0..64u8).map(|i| i / 8).collect();
        let with = round_trip(&entries, ContextMapForm::Nested { use_mtf: true });
        let without = round_trip(&entries, ContextMapForm::Nested { use_mtf: false });
        assert!(
            with <= without,
            "move-to-front should not hurt a run-structured map: {with} vs {without}"
        );
    }

    #[test]
    fn forward_move_to_front_inverts_the_decoders_transform() {
        let cases: Vec<Vec<u8>> = vec![
            vec![0, 0, 0, 0],
            vec![5, 5, 3, 0, 3, 7, 7, 7, 1, 0, 5],
            (0..64u8).map(|i| i % 9).collect(),
        ];
        for original in cases {
            let mut encoded = original.clone();
            move_to_front(&mut encoded);
            let mut decoded = encoded.clone();
            inverse_move_to_front(&mut decoded);
            assert_eq!(decoded, original);
        }
    }

    #[test]
    fn non_dense_maps_are_rejected() {
        assert!(ContextMap::new(vec![0, 2]).is_err());
        assert!(ContextMap::new(Vec::new()).is_err());
        assert!(ContextMap::identity(300).is_err());
    }
}
