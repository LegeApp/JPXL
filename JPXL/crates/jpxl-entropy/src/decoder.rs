//! The unified symbol decoder.
//!
//! ISO/IEC 18181-1 C.2.1 (opening a distribution bundle) and C.3.1/C.3.3
//! (reading integers from it). This is the facade the rest of the decoder
//! uses: open a [`SymbolDecoder`] over a [`BitReader`], then pull integers
//! with [`SymbolDecoder::read_uint`].
//!
//! The two coding modes — canonical prefix codes and ANS — differ only in how
//! a token is produced. Everything above that (hybrid-uint reconstruction,
//! LZ77, context clustering) is shared, so [`SymbolDecoder::read_uint`] is
//! written once.
//!
//! Every path here is **bit-exact** (see `docs/PLAN.md`).

use jpxl_bitstream::BitReader;
use jpxl_core::limits::AllocGuard;

use crate::ans::{AnsDistribution, AnsState};
use crate::dist::{ClusterMap, MAX_NESTING_DEPTH, read_cluster_map};
use crate::error::{Result, malformed};
use crate::hybrid::HybridUintConfig;
use crate::lz77::{Lz77Params, Lz77Window, resolve_distance};
use crate::prefix::{PrefixCode, read_prefix_code};

/// Largest alphabet a prefix-coded distribution may declare (18181-1 C.2.1).
const MAX_PREFIX_ALPHABET: u32 = 1 << 15;

/// `log_alphabet_size` for prefix-coded streams (18181-1 C.2.1).
const PREFIX_LOG_ALPHABET_SIZE: u32 = 15;

/// `log_alphabet_size` used for the LZ77 length configuration (18181-1 C.2.1).
const LZ_LENGTH_LOG_ALPHABET_SIZE: u32 = 8;

/// The per-cluster entropy codes of a stream.
#[derive(Debug)]
enum ClusterCodes {
    /// Canonical prefix codes, one per cluster (18181-1 C.2.4).
    Prefix(Vec<PrefixCode>),
    /// ANS distributions plus the single shared state (18181-1 C.2.5, C.3.2).
    Ans {
        distributions: Vec<AnsDistribution>,
        state: AnsState,
    },
}

/// A decoder for one entropy-coded stream (18181-1 Annex C).
///
/// Holds everything C.1 lists as entropy decoder state except the bit reader
/// itself, which the caller owns and passes to each read so that raw bits and
/// entropy-coded symbols interleave in the one stream, as C.3.1 requires.
#[derive(Debug)]
pub struct SymbolDecoder {
    lz77: Lz77Params,
    /// Context reserved for LZ77 distances; valid only when `lz77.enabled`.
    lz_dist_ctx: usize,
    lz_len_conf: HybridUintConfig,
    clusters: ClusterMap,
    /// Hybrid-uint configuration per cluster.
    configs: Vec<HybridUintConfig>,
    codes: ClusterCodes,
    /// Present only when `lz77.enabled`, as the C.1 note permits.
    window: Option<Lz77Window>,
    dist_multiplier: u32,
}

impl SymbolDecoder {
    /// Opens a distribution bundle (18181-1 C.2.1).
    ///
    /// `num_dist` is the number of pre-clustered contexts, supplied by the
    /// referencing clause. All allocations derived from the stream are metered
    /// through `guard` before they are made.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Malformed`](crate::EntropyError::Malformed) for a
    /// stream that violates Annex C, or a limit error if the bundle would
    /// allocate more than the guard permits.
    pub fn open(
        reader: &mut BitReader<'_>,
        num_dist: usize,
        guard: &mut AllocGuard,
    ) -> Result<Self> {
        Self::open_nested(reader, num_dist, 0, false, guard)
    }

    /// Opens a bundle at a given recursion depth.
    ///
    /// `forbid_lz77` carries the C.2.2 rule for a nested decode whose parent
    /// had `num_dist == 2`; see the ambiguity note on that clause in the crate
    /// documentation.
    pub(crate) fn open_nested(
        reader: &mut BitReader<'_>,
        num_dist: usize,
        depth: u32,
        forbid_lz77: bool,
        guard: &mut AllocGuard,
    ) -> Result<Self> {
        if depth > MAX_NESTING_DEPTH {
            return Err(malformed!(
                "C.2.1: distribution decoding nested deeper than {MAX_NESTING_DEPTH} levels"
            ));
        }

        let lz77 = Lz77Params::read(reader)?;
        if forbid_lz77 && lz77.enabled {
            return Err(malformed!(
                "C.2.2: the nested distribution decoder of a two-context stream must not enable \
                 LZ77"
            ));
        }

        // C.2.1: enabling LZ77 appends one context for the distance symbols.
        let mut num_dist = num_dist;
        let mut lz_dist_ctx = 0;
        let mut lz_len_conf = HybridUintConfig::default();
        if lz77.enabled {
            lz_dist_ctx = num_dist;
            num_dist = num_dist
                .checked_add(1)
                .ok_or_else(|| malformed!("C.2.1: context count overflows"))?;
            lz_len_conf = HybridUintConfig::read(reader, LZ_LENGTH_LOG_ALPHABET_SIZE)?;
        }

        let clusters = read_cluster_map(reader, num_dist, depth, guard)?;
        let num_clusters = clusters.num_clusters();

        let use_prefix_code = reader.read_bool()?;
        let log_alphabet_size = if use_prefix_code {
            PREFIX_LOG_ALPHABET_SIZE
        } else {
            5 + reader.read_bits(2)?
        };

        guard.charge(num_clusters as u64 * 12)?;
        let mut configs = Vec::with_capacity(num_clusters);
        for _ in 0..num_clusters {
            configs.push(HybridUintConfig::read(reader, log_alphabet_size)?);
        }

        let codes = if use_prefix_code {
            // C.2.1: every alphabet size is read first, then every code.
            let mut counts = Vec::with_capacity(num_clusters);
            for _ in 0..num_clusters {
                let count = if reader.read_bool()? {
                    let n = reader.read_bits(4)?;
                    1 + (1u32 << n) + reader.read_bits(n)?
                } else {
                    1
                };
                if count > MAX_PREFIX_ALPHABET {
                    return Err(malformed!(
                        "C.2.1: alphabet size {count} exceeds the maximum of \
                         {MAX_PREFIX_ALPHABET}"
                    ));
                }
                counts.push(count as usize);
            }
            let mut codes = Vec::with_capacity(num_clusters);
            for count in counts {
                codes.push(read_prefix_code(reader, count, guard)?);
            }
            ClusterCodes::Prefix(codes)
        } else {
            let mut distributions = Vec::with_capacity(num_clusters);
            for _ in 0..num_clusters {
                distributions.push(AnsDistribution::read(reader, log_alphabet_size, guard)?);
            }
            // C.3.2: the shared state is seeded once the distributions are in
            // place, at the start of the entropy-coded stream proper.
            let state = AnsState::init(reader)?;
            ClusterCodes::Ans {
                distributions,
                state,
            }
        };

        let window = if lz77.enabled {
            Some(Lz77Window::new(guard)?)
        } else {
            None
        };

        Ok(Self {
            lz77,
            lz_dist_ctx,
            lz_len_conf,
            clusters,
            configs,
            codes,
            window,
            dist_multiplier: 0,
        })
    }

    /// Sets the row stride used by the two-dimensional distance shorthand of
    /// C.3.3.
    ///
    /// C.3.3 states that `dist_multiplier` is 0 unless the referencing clause
    /// says otherwise, so this starts at 0 and only clauses that define a
    /// stride call this.
    pub const fn set_dist_multiplier(&mut self, dist_multiplier: u32) {
        self.dist_multiplier = dist_multiplier;
    }

    /// The context-to-cluster mapping of this stream.
    #[must_use]
    pub const fn clusters(&self) -> &ClusterMap {
        &self.clusters
    }

    /// Whether this stream uses canonical prefix codes rather than ANS.
    #[must_use]
    pub const fn uses_prefix_code(&self) -> bool {
        matches!(self.codes, ClusterCodes::Prefix(_))
    }

    /// Reads one raw entropy-coded token for `ctx` (18181-1 C.3.1).
    ///
    /// This is the token *before* hybrid-uint reconstruction and before the
    /// LZ77 layer. Most callers want [`read_uint`](Self::read_uint) instead.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Malformed`](crate::EntropyError::Malformed) if `ctx` is
    /// unknown or the stream is inconsistent.
    pub fn read_token(&mut self, reader: &mut BitReader<'_>, ctx: usize) -> Result<u32> {
        let cluster = self.clusters.cluster_of(ctx)?;
        self.read_token_for_cluster(reader, cluster)
    }

    fn read_token_for_cluster(
        &mut self,
        reader: &mut BitReader<'_>,
        cluster: usize,
    ) -> Result<u32> {
        match &mut self.codes {
            ClusterCodes::Prefix(codes) => codes
                .get(cluster)
                .ok_or_else(|| malformed!("C.2.1: cluster {cluster} has no prefix code"))?
                .decode(reader),
            ClusterCodes::Ans {
                distributions,
                state,
            } => {
                let dist = distributions
                    .get(cluster)
                    .ok_or_else(|| malformed!("C.2.1: cluster {cluster} has no distribution"))?;
                state.decode(reader, dist)
            }
        }
    }

    /// Reads one unsigned integer (`DecodeHybridVarLenUint`, 18181-1 C.3.3).
    ///
    /// Resolves an in-progress LZ77 copy if there is one; otherwise decodes a
    /// token, and either starts a copy or reconstructs a value from the
    /// token's hybrid-uint configuration plus raw extra bits.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Malformed`](crate::EntropyError::Malformed) for a
    /// stream that violates C.3.3, or a bitstream error at end of input.
    pub fn read_uint(&mut self, reader: &mut BitReader<'_>, ctx: usize) -> Result<u32> {
        // An outstanding copy takes precedence over reading a new token, and
        // consumes no bits.
        if let Some(window) = &mut self.window
            && window.copying()
        {
            return window.next_copied();
        }

        let cluster = self.clusters.cluster_of(ctx)?;
        let token = self.read_token_for_cluster(reader, cluster)?;

        if self.lz77.enabled && token >= self.lz77.min_symbol {
            let length = self
                .lz_len_conf
                .read_uint(reader, token - self.lz77.min_symbol)?
                .checked_add(self.lz77.min_length)
                .ok_or_else(|| malformed!("C.3.3: LZ77 copy length overflows"))?;

            let dist_cluster = self.clusters.cluster_of(self.lz_dist_ctx)?;
            let dist_token = self.read_token_for_cluster(reader, dist_cluster)?;
            let raw_distance = self
                .configs
                .get(dist_cluster)
                .ok_or_else(|| malformed!("C.2.1: cluster {dist_cluster} has no configuration"))?
                .read_uint(reader, dist_token)?;
            let distance = resolve_distance(raw_distance, self.dist_multiplier)?;

            let window = self
                .window
                .as_mut()
                .ok_or_else(|| malformed!("C.3.3: LZ77 is enabled but no window was allocated"))?;
            window.start_copy(length, distance);
            // C.3.3 expresses this as a recursive call. `min_length` is at
            // least 3 (Table C.1), so the copy counter is always positive here
            // and the recursion can only take the copy branch above — which is
            // why the clause's choice of argument for that call (`clusters
            // [ctx]` rather than `ctx`) cannot affect the result.
            return window.next_copied();
        }

        let value = self
            .configs
            .get(cluster)
            .ok_or_else(|| malformed!("C.2.1: cluster {cluster} has no configuration"))?
            .read_uint(reader, token)?;
        if let Some(window) = &mut self.window {
            window.push(value);
        }
        Ok(value)
    }

    /// Checks the terminal condition of the stream (18181-1 C.3.2).
    ///
    /// An ANS stream must end with the state at
    /// [`FINAL_ANS_STATE`](crate::ans::FINAL_ANS_STATE). Prefix-coded streams
    /// carry no such marker and always succeed.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Malformed`](crate::EntropyError::Malformed) if the ANS
    /// state does not match, which means the stream was truncated, corrupted,
    /// or read with the wrong context sequence.
    pub fn finish(&self) -> Result<()> {
        match &self.codes {
            ClusterCodes::Prefix(_) => Ok(()),
            ClusterCodes::Ans { state, .. } => {
                if state.is_final() {
                    Ok(())
                } else {
                    Err(malformed!(
                        "C.3.2: ANS stream ended with state {:#x}, expected {:#x}",
                        state.raw(),
                        crate::ans::FINAL_ANS_STATE
                    ))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jpxl_core::limits::Limits;

    /// A minimal prefix-coded bundle: one context, no LZ77, one cluster whose
    /// alphabet size is 1, so every symbol is 0 and no code bits exist.
    ///
    /// Bit layout, in read order:
    ///   b0      = 0      lz77.enabled = false                 (Table C.1)
    ///   -                num_dist == 1, so C.2.2 reads nothing
    ///   b1      = 1      use_prefix_code                       (C.2.1)
    ///   b2..b5  = 0      HybridUintConfig split_exponent u(4) = 0
    ///                    (log_alphabet_size 15 -> ceil(log2(16)) = 4 bits;
    ///                     0 != 15, so msb/lsb fields follow)
    ///   b6      = 0      msb_in_token u(ceil(log2(1))) = u(0)  -> no bits
    ///                    lsb_in_token u(0)                     -> no bits
    ///                    (so b6 is already the count flag)
    ///   b6      = 0      count flag false -> alphabet size 1   (C.2.1)
    /// Total: 7 bits. byte = b1 = 2.
    #[test]
    fn opens_a_degenerate_prefix_bundle() {
        let mut g = AllocGuard::new(&Limits::relaxed());
        let data = [0b0000_0010u8];
        let mut r = BitReader::new(&data);
        let mut dec = SymbolDecoder::open(&mut r, 1, &mut g).expect("bundle opens");
        assert!(dec.uses_prefix_code());
        assert_eq!(dec.clusters().num_clusters(), 1);
        assert_eq!(r.total_bits_read(), 7);

        // split_exponent 0 means split = 1, so token 0 is literal and every
        // read yields 0 while consuming nothing.
        for _ in 0..4 {
            assert_eq!(dec.read_uint(&mut r, 0).expect("symbol"), 0);
        }
        assert_eq!(r.total_bits_read(), 7, "a 1-symbol alphabet costs no bits");
        dec.finish().expect("prefix streams have no terminal state");
    }

    #[test]
    fn unknown_contexts_are_rejected() {
        let mut g = AllocGuard::new(&Limits::relaxed());
        let data = [0b0000_0010u8];
        let mut r = BitReader::new(&data);
        let mut dec = SymbolDecoder::open(&mut r, 1, &mut g).expect("bundle opens");
        assert!(dec.read_uint(&mut r, 1).is_err());
    }

    #[test]
    fn a_tiny_allocation_budget_stops_the_bundle() {
        // The window alone is 4 MiB, so an LZ77 stream cannot open under a
        // small budget. This proves the metering runs before allocation.
        let limits = Limits {
            max_alloc_bytes: 64,
            ..Limits::default()
        };
        let mut g = AllocGuard::new(&limits);
        // b0 = 1 (lz77 enabled), then min_symbol/min_length selectors 0, 0.
        let data = [0b0000_0001u8, 0, 0, 0, 0, 0, 0, 0];
        let mut r = BitReader::new(&data);
        assert!(SymbolDecoder::open(&mut r, 1, &mut g).is_err());
    }
}
