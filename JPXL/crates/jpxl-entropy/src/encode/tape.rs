//! A stream's tokens recorded once, replayed for both table construction and
//! emission (Phase 41).
//!
//! The replay encoder ([`super::stream::SymbolEncoder`]) needs entropy tables
//! before it can record a stream, and the tables need the whole frame's token
//! counts. A writer that derives its events from a costly traversal (the
//! VarDCT coefficient walk) therefore used to run that traversal twice per
//! plan: once into a census, once into the encoder. [`TokenTapeRecorder`]
//! removes the second traversal: it tokenizes each value under its cluster's
//! configuration as it arrives, counts the token per cluster (exactly the
//! per-cluster token counts [`super::stream::EntropyTables::build`] derives
//! from a raw census), and appends the token to a compact [`TokenTape`]. Once
//! the tables exist, [`TokenTape::write_stream`] emits the recorded tokens
//! bit for bit as the replay encoder would have.
//!
//! Everything here is a re-arrangement of the same integers: the same
//! `(cluster, token, extra)` triples in the same order, the same per-cluster
//! counts, so the tables and the bytes are identical to the two-walk path.
//! `stream::tests::token_tape_matches_symbol_encoder` pins that.

use jpxl_bitstream::BitWriter;

use super::ans::{AnsSymbol, encode_symbols_with};
use super::hybrid::TokenSplit;
use super::stream::{ClusterCodes, EncoderPlan, EntropyTables};
use crate::error::{Result, encode_error};
use crate::hybrid::HybridUintConfig;

/// A recorded token stream: structure-of-arrays, one entry per symbol in
/// emission order.
///
/// Clusters fit a byte (18181-1 C.2.2 clusters are at most 256) and ANS tokens
/// fit a byte too, but prefix-coded tokens may need up to fifteen bits, so the
/// token column is `u16`. Extra-bit counts are below 32.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TokenTape {
    cluster: Vec<u8>,
    token: Vec<u16>,
    extra_bits: Vec<u8>,
    extra: Vec<u32>,
}

impl TokenTape {
    /// An empty tape.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of recorded symbols.
    #[must_use]
    pub fn len(&self) -> usize {
        self.token.len()
    }

    /// Whether nothing was recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.token.is_empty()
    }

    /// Bytes this tape occupies (for memory accounting).
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.cluster.len() + self.token.len() * 2 + self.extra_bits.len() + self.extra.len() * 4
    }

    /// Appends one token.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Encode`](crate::EntropyError::Encode) if the cluster,
    /// token or extra-bit count does not fit the tape's columns.
    pub fn push(&mut self, cluster: usize, split: TokenSplit) -> Result<()> {
        let cluster = u8::try_from(cluster)
            .map_err(|_| encode_error!("C.2.2: cluster {cluster} does not fit the token tape"))?;
        let token = u16::try_from(split.token).map_err(|_| {
            encode_error!("C.2.1: token {} does not fit the token tape", split.token)
        })?;
        let extra_bits = u8::try_from(split.extra_bits).map_err(|_| {
            encode_error!(
                "C.3.3: {} extra bits do not fit the token tape",
                split.extra_bits
            )
        })?;
        self.cluster.push(cluster);
        self.token.push(token);
        self.extra_bits.push(extra_bits);
        self.extra.push(split.extra);
        Ok(())
    }

    /// [`Self::push`] with the cluster already narrowed.
    #[inline]
    fn push_cluster_u8(&mut self, cluster: u8, split: TokenSplit) -> Result<()> {
        let token = u16::try_from(split.token).map_err(|_| {
            encode_error!("C.2.1: token {} does not fit the token tape", split.token)
        })?;
        let extra_bits = u8::try_from(split.extra_bits).map_err(|_| {
            encode_error!(
                "C.3.3: {} extra bits do not fit the token tape",
                split.extra_bits
            )
        })?;
        self.cluster.push(cluster);
        self.token.push(token);
        self.extra_bits.push(extra_bits);
        self.extra.push(split.extra);
        Ok(())
    }

    /// Emits the recorded stream under `tables`, exactly as
    /// [`super::stream::SymbolEncoder::write_stream`] would emit the same
    /// tokens: prefix codes symbol by symbol, or one ANS backward pass whose
    /// seed and renormalisation words interleave with each symbol's extra bits.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Encode`](crate::EntropyError::Encode) if a cluster has
    /// no code or a token has no mass in its cluster.
    pub fn write_stream(&self, tables: &EntropyTables, w: &mut BitWriter) -> Result<()> {
        match tables.codes() {
            ClusterCodes::Prefix(codes) => {
                for (((&cluster, &token), &extra_bits), &extra) in self
                    .cluster
                    .iter()
                    .zip(self.token.iter())
                    .zip(self.extra_bits.iter())
                    .zip(self.extra.iter())
                {
                    let code = codes.get(usize::from(cluster)).ok_or_else(|| {
                        encode_error!("C.2.1: cluster {cluster} has no prefix code")
                    })?;
                    code.write_symbol(w, u32::from(token))?;
                    if extra_bits != 0 {
                        w.write_bits(u32::from(extra_bits), extra)?;
                    }
                }
                Ok(())
            }
            ClusterCodes::Ans { tables, .. } => {
                let payload = encode_symbols_with(tables, self.len(), |index| AnsSymbol {
                    cluster: self
                        .cluster
                        .get(index)
                        .copied()
                        .map_or(usize::MAX, usize::from),
                    token: self.token.get(index).copied().map_or(u32::MAX, u32::from),
                })?;
                // C.3.2 seeds the state from a u(32) at the start of the stream.
                w.write_bits(32, payload.initial_state())?;
                for (index, (&extra_bits, &extra)) in
                    self.extra_bits.iter().zip(self.extra.iter()).enumerate()
                {
                    // The decoder renormalizes inside the symbol's decode step,
                    // before it reads that symbol's raw extra bits.
                    if let Some(word) = payload.renormalization(index) {
                        w.write_bits(16, u32::from(word))?;
                    }
                    // `write_bits(0, 0)` is a no-op; most tokens carry no extra
                    // bits, so skip the call rather than pay for it.
                    if extra_bits != 0 {
                        w.write_bits(u32::from(extra_bits), extra)?;
                    }
                }
                Ok(())
            }
        }
    }
}

/// Records value streams into [`TokenTape`]s while accumulating the
/// per-cluster token counts the tables need.
///
/// One recorder serves many tapes (one per section): call
/// [`take_tape`](Self::take_tape) at each section boundary; the counts keep
/// accumulating across sections. When every section is recorded,
/// [`into_counts`](Self::into_counts) yields the counts for
/// [`EntropyTables::build_from_token_counts`]. Counts from several recorders
/// (parallel workers) are merged with [`merge_token_counts`]; integer sums do
/// not depend on the merge order.
///
/// LZ77 is not supported: a back-reference's length token is not a
/// hybrid-uint token of the value cluster, and the tape stores only those.
#[derive(Debug, Clone)]
pub struct TokenTapeRecorder<'a> {
    /// Kept for the lifetime tie: the recorder's counts describe this plan.
    _plan: core::marker::PhantomData<&'a EncoderPlan>,
    /// `cluster_of[ctx]`, flattened once so the hot path is one lookup.
    cluster_of: Vec<u8>,
    /// Per cluster: its configuration and the `split` below which a value is
    /// its own token with no extra bits (the common case, taken inline).
    configs: Vec<(HybridUintConfig, u32)>,
    counts: Vec<Vec<u64>>,
    tape: TokenTape,
}

impl<'a> TokenTapeRecorder<'a> {
    /// A recorder for `plan`, whose configurations are validated once here.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Encode`](crate::EntropyError::Encode) if the plan
    /// enables LZ77, has the wrong number of configurations, or a
    /// configuration is invalid.
    pub fn new(plan: &'a EncoderPlan) -> Result<Self> {
        if plan.lz77.is_some() {
            return Err(encode_error!(
                "C.3.3: the token tape records hybrid-uint values only; LZ77 needs the replay encoder"
            ));
        }
        let num_clusters = plan.context_map.num_clusters();
        if plan.configs.len() != num_clusters {
            return Err(encode_error!(
                "C.2.1: {} configurations for {num_clusters} clusters",
                plan.configs.len()
            ));
        }
        for config in &plan.configs {
            config.validate()?;
        }
        let num_dist = plan.context_map.num_dist();
        let mut cluster_of = Vec::with_capacity(num_dist);
        for ctx in 0..num_dist {
            let cluster = plan.context_map.cluster_of(ctx)?;
            cluster_of.push(u8::try_from(cluster).map_err(|_| {
                encode_error!("C.2.2: cluster {cluster} does not fit the token tape")
            })?);
        }
        Ok(Self {
            _plan: core::marker::PhantomData,
            cluster_of,
            configs: plan
                .configs
                .iter()
                .map(|config| (*config, config.split()))
                .collect(),
            counts: vec![Vec::new(); num_clusters],
            tape: TokenTape::new(),
        })
    }

    /// Records one hybrid-uint value in context `ctx`.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Encode`](crate::EntropyError::Encode) if `ctx` is
    /// outside the context map or the value cannot be tokenized.
    #[inline]
    pub fn record(&mut self, ctx: usize, value: u32) -> Result<()> {
        let cluster = self
            .cluster_of
            .get(ctx)
            .copied()
            .ok_or_else(|| encode_error!("C.2.2: context {ctx} is outside this stream"))?;
        let (config, split_at) = self
            .configs
            .get(usize::from(cluster))
            .ok_or_else(|| encode_error!("C.2.1: cluster {cluster} has no configuration"))?;
        // Below the split a value is its own token with no extra bits (C.3.3);
        // that is what `tokenize` returns there, taken inline because it is
        // the overwhelming case.
        let split = if value < *split_at {
            TokenSplit {
                token: value,
                extra_bits: 0,
                extra: 0,
            }
        } else {
            config.tokenize(value)?
        };
        let slot = self
            .counts
            .get_mut(usize::from(cluster))
            .ok_or_else(|| encode_error!("C.2.1: cluster {cluster} is out of range"))?;
        let token = split.token as usize;
        if slot.len() <= token {
            slot.resize(token + 1, 0);
        }
        if let Some(entry) = slot.get_mut(token) {
            *entry = entry.saturating_add(1);
        }
        self.tape.push_cluster_u8(cluster, split)
    }

    /// Hands out the tape recorded since the last call and starts a new one.
    #[must_use]
    pub fn take_tape(&mut self) -> TokenTape {
        core::mem::take(&mut self.tape)
    }

    /// The accumulated per-cluster token counts.
    #[must_use]
    pub fn into_counts(self) -> Vec<Vec<u64>> {
        self.counts
    }
}

/// Adds `from`'s per-cluster token counts into `into`.
///
/// # Errors
///
/// [`EntropyError::Encode`](crate::EntropyError::Encode) if the two count
/// sets cover different cluster counts.
pub fn merge_token_counts(into: &mut [Vec<u64>], from: Vec<Vec<u64>>) -> Result<()> {
    if into.len() != from.len() {
        return Err(encode_error!(
            "C.2.1: cannot merge token counts over different cluster counts"
        ));
    }
    for (target, source) in into.iter_mut().zip(from) {
        if target.len() < source.len() {
            target.resize(source.len(), 0);
        }
        for (slot, count) in target.iter_mut().zip(source) {
            *slot = slot.saturating_add(count);
        }
    }
    Ok(())
}
