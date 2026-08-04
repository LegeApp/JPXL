//! The two-pass entropy compiler: census, tables, replay.
//!
//! This is the write-side counterpart of [`SymbolDecoder`](crate::SymbolDecoder)
//! and the shape `docs/Encoder-plan1.md` asks for. Coefficient loops must never
//! call an ANS coder directly — rANS emission runs backwards, so the tokens
//! cannot be written as they are produced, and the histograms cannot be chosen
//! until every token is known. The three phases are therefore separate types:
//!
//! 1. **Census.** [`TokenCensus`] counts raw *values* per pre-clustered
//!    context. Values, not tokens: which token a value becomes depends on the
//!    hybrid-uint configuration, which belongs to the cluster, which is not
//!    decided yet.
//! 2. **Tables.** [`EntropyTables::build`] applies a caller-supplied
//!    [`EncoderPlan`] — clustering, configurations, backend — to the census and
//!    produces the histograms or prefix codes, ready to serialize. All policy
//!    lives in the plan; this crate makes no density decisions beyond picking
//!    the shortest legal encoding of a thing it has already been told to write.
//! 3. **Replay.** [`SymbolEncoder`] takes the same `(context, value)` sequence
//!    again, records tokens and their raw extra bits, and emits the stream.
//!
//! # Where the bits go
//!
//! [`EntropyTables::write_bundle`] writes everything C.2.1 reads before the
//! symbols: the LZ77 flag, the context map, the backend flag, the hybrid-uint
//! configurations and the histograms or prefix codes.
//! [`SymbolEncoder::write_stream`] writes what C.3 reads: for ANS the `u(32)`
//! seed of C.3.2 followed by the renormalization words and raw extra bits
//! interleaved exactly as the decoder consumes them; for prefix codes just the
//! code words and extra bits.
//!
//! Calling the two back to back matches [`SymbolDecoder::open`]. Splitting them
//! matches [`SymbolDecoder::open_deferred`] followed by
//! [`SymbolDecoder::restart`], which is how a modular sub-bitstream signals one
//! bundle in `LfGlobal` and then several streams.
//!
//! # LZ77
//!
//! Back-reference emission is **not implemented** (slice 19). The `lz77.enabled`
//! flag is written as false, which is a complete and legal framing: no distance
//! context is appended and no `lz_len_conf` is written, exactly as C.2.1
//! requires when the flag is clear.
//!
//! Every path here is **bit-exact** (see `docs/PLAN.md`).

use jpxl_bitstream::BitWriter;

use crate::error::{Result, encode_error};
use crate::hybrid::{HybridUintConfig, bit_width};

use super::ans::{AnsEncodeTable, AnsSymbol, encode_symbols};
use super::cluster::{ContextMap, ContextMapForm};
use super::histogram::Histogram;
use super::prefix::{MAX_PREFIX_ALPHABET, PrefixEncoder};

/// `log_alphabet_size` for prefix-coded streams (18181-1 C.2.1).
const PREFIX_LOG_ALPHABET_SIZE: u32 = 15;

/// Smallest `log_alphabet_size` an ANS stream can signal: C.2.1 reads it as
/// `5 + u(2)`.
pub const MIN_ANS_LOG_ALPHABET_SIZE: u32 = 5;
/// Largest `log_alphabet_size` an ANS stream can signal.
pub const MAX_ANS_LOG_ALPHABET_SIZE: u32 = 8;

/// Values below this are counted in the census's inline array.
const SMALL_VALUES: usize = 32;

/// Which entropy backend a stream uses (18181-1 C.2.1 `use_prefix_code`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodingMode {
    /// Asymmetric numeral systems (C.2.5, C.3.2).
    Ans,
    /// Canonical prefix codes (C.2.4).
    Prefix,
}

/// Counts of the raw values seen in one pre-clustered context.
///
/// Small values — which dominate every JPEG XL context — land in an inline
/// array; the rest go into a sparse tail kept sorted by value, as
/// `docs/Encoder-plan1.md` describes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RawHistogram {
    small: [u64; SMALL_VALUES],
    tail: Vec<(u32, u64)>,
}

impl RawHistogram {
    /// An empty histogram.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Counts one occurrence of `value`.
    pub fn record(&mut self, value: u32) {
        self.add(value, 1);
    }

    /// Counts `count` occurrences of `value`.
    pub fn add(&mut self, value: u32, count: u64) {
        if let Some(slot) = self.small.get_mut(value as usize) {
            *slot = slot.saturating_add(count);
            return;
        }
        match self.tail.binary_search_by_key(&value, |&(v, _)| v) {
            Ok(index) => {
                if let Some((_, slot)) = self.tail.get_mut(index) {
                    *slot = slot.saturating_add(count);
                }
            }
            Err(index) => self.tail.insert(index, (value, count)),
        }
    }

    /// Occurrences of `value`.
    #[must_use]
    pub fn count(&self, value: u32) -> u64 {
        if let Some(&count) = self.small.get(value as usize) {
            return count;
        }
        self.tail
            .binary_search_by_key(&value, |&(v, _)| v)
            .ok()
            .and_then(|index| self.tail.get(index))
            .map_or(0, |&(_, count)| count)
    }

    /// Every value that occurs, with its count, in increasing value order.
    pub fn iter(&self) -> impl Iterator<Item = (u32, u64)> + '_ {
        self.small
            .iter()
            .enumerate()
            .filter(|&(_, &count)| count != 0)
            .map(|(value, &count)| (u32::try_from(value).unwrap_or(u32::MAX), count))
            .chain(self.tail.iter().copied())
    }

    /// Total number of values recorded.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.iter().map(|(_, count)| count).sum()
    }

    /// Whether nothing was recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.total() == 0
    }
}

/// The census pass: raw value counts for every pre-clustered context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenCensus {
    contexts: Vec<RawHistogram>,
}

impl TokenCensus {
    /// A census over `num_contexts` pre-clustered contexts.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Encode`](crate::EntropyError::Encode) if
    /// `num_contexts` is zero.
    pub fn new(num_contexts: usize) -> Result<Self> {
        if num_contexts == 0 {
            return Err(encode_error!("C.2.1: num_dist must be at least 1"));
        }
        Ok(Self {
            contexts: vec![RawHistogram::new(); num_contexts],
        })
    }

    /// Records one value in `ctx`.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Encode`](crate::EntropyError::Encode) if `ctx` is not a
    /// context of this census.
    pub fn record(&mut self, ctx: usize, value: u32) -> Result<()> {
        self.contexts
            .get_mut(ctx)
            .ok_or_else(|| encode_error!("C.2.1: context {ctx} is outside this census"))?
            .record(value);
        Ok(())
    }

    /// The raw histogram of one context.
    #[must_use]
    pub fn context(&self, ctx: usize) -> Option<&RawHistogram> {
        self.contexts.get(ctx)
    }

    /// Number of pre-clustered contexts.
    #[must_use]
    pub fn num_contexts(&self) -> usize {
        self.contexts.len()
    }
}

/// The policy decisions a caller makes before tables can be built.
///
/// Everything here is a choice: which backend, how contexts are clustered, how
/// values split into tokens and extra bits. This crate validates and executes
/// them; it does not invent them. A later policy crate produces plans, and the
/// tests here produce them by hand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncoderPlan {
    /// Prefix codes or ANS.
    pub mode: CodingMode,
    /// Context-to-cluster mapping.
    pub context_map: ContextMap,
    /// Which C.2.2 encoding to use for it.
    pub context_map_form: ContextMapForm,
    /// One hybrid-uint configuration per cluster (18181-1 C.2.3).
    pub configs: Vec<HybridUintConfig>,
    /// ANS `log_alphabet_size`; `None` derives the narrowest legal value.
    /// Ignored for prefix codes, which C.2.1 fixes at 15.
    pub log_alphabet_size: Option<u32>,
}

impl EncoderPlan {
    /// A plan with one cluster per context and the same configuration for all
    /// of them.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Encode`](crate::EntropyError::Encode) if the context
    /// count or the configuration is out of range.
    pub fn identity(
        num_contexts: usize,
        mode: CodingMode,
        config: HybridUintConfig,
    ) -> Result<Self> {
        config.validate()?;
        let context_map = ContextMap::identity(num_contexts)?;
        let configs = vec![config; context_map.num_clusters()];
        Ok(Self {
            mode,
            context_map,
            context_map_form: ContextMapForm::Auto,
            configs,
            log_alphabet_size: None,
        })
    }

    /// A plan with a caller-supplied clustering and one configuration per
    /// cluster.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Encode`](crate::EntropyError::Encode) if the clustering
    /// is not dense or the configuration count does not match it.
    pub fn clustered(
        context_map: ContextMap,
        mode: CodingMode,
        configs: Vec<HybridUintConfig>,
    ) -> Result<Self> {
        if configs.len() != context_map.num_clusters() {
            return Err(encode_error!(
                "C.2.1: {} configurations for {} clusters",
                configs.len(),
                context_map.num_clusters()
            ));
        }
        for config in &configs {
            config.validate()?;
        }
        Ok(Self {
            mode,
            context_map,
            context_map_form: ContextMapForm::Auto,
            configs,
            log_alphabet_size: None,
        })
    }
}

/// The per-cluster codes of a stream, on the write side.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ClusterCodes {
    Prefix(Vec<PrefixEncoder>),
    Ans {
        histograms: Vec<Histogram>,
        tables: Vec<AnsEncodeTable>,
        log_alphabet_size: u32,
    },
}

/// Everything C.2.1 signals, built and validated, ready to serialize.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntropyTables {
    context_map: ContextMap,
    context_map_form: ContextMapForm,
    configs: Vec<HybridUintConfig>,
    codes: ClusterCodes,
}

impl EntropyTables {
    /// Applies `plan` to `census`.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Encode`](crate::EntropyError::Encode) if the plan is
    /// inconsistent with the census, or if the tokens the plan produces do not
    /// fit the alphabet its backend allows.
    pub fn build(plan: &EncoderPlan, census: &TokenCensus) -> Result<Self> {
        let num_dist = plan.context_map.num_dist();
        if census.num_contexts() != num_dist {
            return Err(encode_error!(
                "C.2.1: the census has {} contexts but the map has {num_dist}",
                census.num_contexts()
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

        // Tokenize the census: raw value counts become token counts, per
        // cluster, under that cluster's configuration.
        let mut counts: Vec<Vec<u64>> = vec![Vec::new(); num_clusters];
        for ctx in 0..num_dist {
            let cluster = plan.context_map.cluster_of(ctx)?;
            let config = plan
                .configs
                .get(cluster)
                .ok_or_else(|| encode_error!("C.2.1: cluster {cluster} has no configuration"))?;
            let histogram = census
                .context(ctx)
                .ok_or_else(|| encode_error!("C.2.1: context {ctx} is outside the census"))?;
            let slot = counts
                .get_mut(cluster)
                .ok_or_else(|| encode_error!("C.2.1: cluster {cluster} is out of range"))?;
            for (value, count) in histogram.iter() {
                let token = config.tokenize(value)?.token as usize;
                if slot.len() <= token {
                    slot.resize(token + 1, 0);
                }
                let entry = slot
                    .get_mut(token)
                    .ok_or_else(|| encode_error!("C.2.1: token {token} is out of range"))?;
                *entry = entry.saturating_add(count);
            }
        }

        let codes = match plan.mode {
            CodingMode::Prefix => {
                let mut codes = Vec::with_capacity(num_clusters);
                for cluster_counts in &counts {
                    let size = cluster_counts.len().max(1);
                    if size > MAX_PREFIX_ALPHABET {
                        return Err(encode_error!(
                            "C.2.1: alphabet size {size} exceeds {MAX_PREFIX_ALPHABET}"
                        ));
                    }
                    let mut padded = cluster_counts.clone();
                    padded.resize(size, 0);
                    codes.push(PrefixEncoder::from_counts(&padded)?);
                }
                ClusterCodes::Prefix(codes)
            }
            CodingMode::Ans => {
                let max_token = counts
                    .iter()
                    .map(|c| c.len())
                    .max()
                    .unwrap_or(1)
                    .saturating_sub(1);
                // C.2.3 also caps every configuration's split_exponent by
                // log_alphabet_size, so the alphabet must be wide enough for
                // the configurations as well as for the tokens. A configuration
                // with in-token bits needs one more: at
                // `split_exponent == log_alphabet_size` the clause stops
                // reading and forces both in-token fields to zero.
                let widest_split = plan
                    .configs
                    .iter()
                    .map(|c| c.split_exponent + u32::from(c.msb_in_token + c.lsb_in_token != 0))
                    .max()
                    .unwrap_or(0);
                let needed = bit_width(
                    u32::try_from(max_token)
                        .map_err(|_| encode_error!("C.2.1: token {max_token} is out of range"))?,
                )
                .max(MIN_ANS_LOG_ALPHABET_SIZE)
                .max(widest_split);
                let log_alphabet_size = plan.log_alphabet_size.unwrap_or(needed);
                if !(MIN_ANS_LOG_ALPHABET_SIZE..=MAX_ANS_LOG_ALPHABET_SIZE)
                    .contains(&log_alphabet_size)
                {
                    return Err(encode_error!(
                        "C.2.1: an ANS stream needs log_alphabet_size in \
                         [{MIN_ANS_LOG_ALPHABET_SIZE}, {MAX_ANS_LOG_ALPHABET_SIZE}], not \
                         {log_alphabet_size}"
                    ));
                }
                if log_alphabet_size < needed {
                    return Err(encode_error!(
                        "C.2.1: token {max_token} does not fit log_alphabet_size \
                         {log_alphabet_size}"
                    ));
                }
                let mut histograms = Vec::with_capacity(num_clusters);
                let mut tables = Vec::with_capacity(num_clusters);
                for cluster_counts in &counts {
                    let histogram = Histogram::from_counts(cluster_counts, log_alphabet_size)?;
                    tables.push(AnsEncodeTable::new(&histogram.distribution()?)?);
                    histograms.push(histogram);
                }
                ClusterCodes::Ans {
                    histograms,
                    tables,
                    log_alphabet_size,
                }
            }
        };

        Ok(Self {
            context_map: plan.context_map.clone(),
            context_map_form: plan.context_map_form,
            configs: plan.configs.clone(),
            codes,
        })
    }

    /// The context map these tables were built for.
    #[must_use]
    pub const fn context_map(&self) -> &ContextMap {
        &self.context_map
    }

    /// Whether the stream uses prefix codes rather than ANS.
    #[must_use]
    pub const fn uses_prefix_code(&self) -> bool {
        matches!(self.codes, ClusterCodes::Prefix(_))
    }

    /// The `log_alphabet_size` the bundle signals (18181-1 C.2.1).
    #[must_use]
    pub const fn log_alphabet_size(&self) -> u32 {
        match self.codes {
            ClusterCodes::Prefix(_) => PREFIX_LOG_ALPHABET_SIZE,
            ClusterCodes::Ans {
                log_alphabet_size, ..
            } => log_alphabet_size,
        }
    }

    /// The hybrid-uint configuration of a cluster.
    #[must_use]
    pub fn config(&self, cluster: usize) -> Option<HybridUintConfig> {
        self.configs.get(cluster).copied()
    }

    /// Writes the distribution bundle (18181-1 C.2.1).
    ///
    /// This stops short of the ANS seed of C.3.2, which belongs to the stream;
    /// see the module documentation.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Encode`](crate::EntropyError::Encode) if a table cannot
    /// be expressed, or a bitstream error.
    pub fn write_bundle(&self, w: &mut BitWriter) -> Result<()> {
        // Table C.1: LZ77 is not emitted (see the module documentation), so the
        // flag is false and neither min_symbol/min_length nor the distance
        // context exists.
        w.write_bool(false);

        self.context_map.write(w, self.context_map_form)?;

        match &self.codes {
            ClusterCodes::Prefix(codes) => {
                w.write_bool(true);
                for config in &self.configs {
                    config.write(w, PREFIX_LOG_ALPHABET_SIZE)?;
                }
                // C.2.1: every alphabet size first, then every code.
                for code in codes {
                    write_alphabet_size(w, code.alphabet_size())?;
                }
                for code in codes {
                    code.write_code(w)?;
                }
            }
            ClusterCodes::Ans {
                histograms,
                log_alphabet_size,
                ..
            } => {
                w.write_bool(false);
                w.write_bits(2, log_alphabet_size - MIN_ANS_LOG_ALPHABET_SIZE)?;
                for config in &self.configs {
                    config.write(w, *log_alphabet_size)?;
                }
                for histogram in histograms {
                    histogram.write(w)?;
                }
            }
        }
        Ok(())
    }
}

/// One recorded token and the raw bits that follow it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Event {
    cluster: usize,
    token: u32,
    extra_bits: u32,
    extra: u32,
}

/// The replay pass: records tokens, then emits the entropy-coded stream.
///
/// Recording and emission are separate because rANS is written backwards: no
/// bit of an ANS stream can be produced until its last symbol is known.
#[derive(Debug, Clone)]
pub struct SymbolEncoder<'a> {
    tables: &'a EntropyTables,
    events: Vec<Event>,
}

impl<'a> SymbolEncoder<'a> {
    /// Starts an entropy-coded stream over `tables`.
    #[must_use]
    pub fn new(tables: &'a EntropyTables) -> Self {
        Self {
            tables,
            events: Vec::new(),
        }
    }

    /// Records one unsigned integer in `ctx` (the inverse of
    /// `DecodeHybridVarLenUint`, 18181-1 C.3.3).
    ///
    /// # Errors
    ///
    /// [`EntropyError::Encode`](crate::EntropyError::Encode) if `ctx` is
    /// unknown, or if the value's token is outside the alphabet its cluster's
    /// code covers — which means the census this stream's tables were built
    /// from did not include this value.
    pub fn push_uint(&mut self, ctx: usize, value: u32) -> Result<()> {
        let cluster = self.tables.context_map.cluster_of(ctx)?;
        let config = self
            .tables
            .configs
            .get(cluster)
            .ok_or_else(|| encode_error!("C.2.1: cluster {cluster} has no configuration"))?;
        let split = config.tokenize(value)?;
        match &self.tables.codes {
            ClusterCodes::Prefix(codes) => {
                let code = codes
                    .get(cluster)
                    .ok_or_else(|| encode_error!("C.2.1: cluster {cluster} has no prefix code"))?;
                if code.constant_symbol() != Some(split.token) && code.code_length(split.token) == 0
                {
                    return Err(encode_error!(
                        "C.2.4: token {} of value {value} is not in the code of cluster {cluster}",
                        split.token
                    ));
                }
            }
            ClusterCodes::Ans { tables, .. } => {
                let table = tables
                    .get(cluster)
                    .ok_or_else(|| encode_error!("C.2.1: cluster {cluster} has no distribution"))?;
                if table.probability(split.token) == 0 {
                    return Err(encode_error!(
                        "C.2.5: token {} of value {value} has no probability mass in cluster \
                         {cluster}",
                        split.token
                    ));
                }
            }
        }
        self.events.push(Event {
            cluster,
            token: split.token,
            extra_bits: split.extra_bits,
            extra: split.extra,
        });
        Ok(())
    }

    /// Number of integers recorded so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether nothing has been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Writes the entropy-coded stream (18181-1 C.3).
    ///
    /// # Errors
    ///
    /// [`EntropyError::Encode`](crate::EntropyError::Encode) if a token cannot
    /// be coded, or a bitstream error.
    pub fn write_stream(&self, w: &mut BitWriter) -> Result<()> {
        match &self.tables.codes {
            ClusterCodes::Prefix(codes) => {
                for event in &self.events {
                    let code = codes.get(event.cluster).ok_or_else(|| {
                        encode_error!("C.2.1: cluster {} has no prefix code", event.cluster)
                    })?;
                    code.write_symbol(w, event.token)?;
                    w.write_bits(event.extra_bits, event.extra)?;
                }
                Ok(())
            }
            ClusterCodes::Ans { tables, .. } => {
                let symbols: Vec<AnsSymbol> = self
                    .events
                    .iter()
                    .map(|event| AnsSymbol {
                        cluster: event.cluster,
                        token: event.token,
                    })
                    .collect();
                let payload = encode_symbols(tables, &symbols)?;
                // C.3.2 seeds the state from a u(32) at the start of the stream.
                w.write_bits(32, payload.initial_state())?;
                for (index, event) in self.events.iter().enumerate() {
                    // The decoder renormalizes inside the symbol's decode step,
                    // before it reads that symbol's raw extra bits.
                    if let Some(word) = payload.renormalization(index) {
                        w.write_bits(16, u32::from(word))?;
                    }
                    w.write_bits(event.extra_bits, event.extra)?;
                }
                Ok(())
            }
        }
    }
}

/// Writes C.2.1's prefix-code alphabet size: `1`, or `1 + (1 << n) + u(n)`.
fn write_alphabet_size(w: &mut BitWriter, size: usize) -> Result<()> {
    if size == 0 || size > MAX_PREFIX_ALPHABET {
        return Err(encode_error!(
            "C.2.1: alphabet size {size} is outside [1, {MAX_PREFIX_ALPHABET}]"
        ));
    }
    if size == 1 {
        w.write_bool(false);
        return Ok(());
    }
    let size = u32::try_from(size).map_err(|_| encode_error!("C.2.1: alphabet size overflows"))?;
    let n = bit_width(size - 1) - 1;
    w.write_bool(true);
    w.write_bits(4, n)?;
    w.write_bits(n, size - 1 - (1 << n))?;
    Ok(())
}

#[cfg(test)]
// Test code: an out-of-range index is a failed assertion rather than an attack
// surface, and the casts are on values these tests chose themselves.
#[allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]
mod tests {
    use super::*;
    use crate::SymbolDecoder;
    use jpxl_bitstream::BitReader;
    use jpxl_core::limits::{AllocGuard, Limits};

    #[test]
    fn alphabet_size_field_round_trips() {
        for size in [1usize, 2, 3, 4, 5, 16, 17, 255, 256, 1024, 1 << 15] {
            let mut w = BitWriter::new();
            write_alphabet_size(&mut w, size).expect("in range");
            let bytes = w.into_bytes();
            let mut r = BitReader::new(&bytes);
            let read = if r.read_bool().expect("flag") {
                let n = r.read_bits(4).expect("width");
                1 + (1u32 << n) + r.read_bits(n).expect("payload")
            } else {
                1
            };
            assert_eq!(read as usize, size);
        }
        assert!(write_alphabet_size(&mut BitWriter::new(), 0).is_err());
        assert!(write_alphabet_size(&mut BitWriter::new(), (1 << 15) + 1).is_err());
    }

    /// The whole compiler in one call: census, tables, replay, decode.
    fn round_trip(plan_mode: CodingMode, values: &[(usize, u32)], num_contexts: usize) {
        let config = HybridUintConfig::new(4, 2, 1).expect("legal");
        let mut census = TokenCensus::new(num_contexts).expect("census");
        for &(ctx, value) in values {
            census.record(ctx, value).expect("records");
        }
        let plan = EncoderPlan::identity(num_contexts, plan_mode, config).expect("plan");
        let tables = EntropyTables::build(&plan, &census).expect("tables");

        let mut w = BitWriter::new();
        tables.write_bundle(&mut w).expect("bundle");
        let mut encoder = SymbolEncoder::new(&tables);
        for &(ctx, value) in values {
            encoder.push_uint(ctx, value).expect("records");
        }
        encoder.write_stream(&mut w).expect("stream");
        let bits = w.bit_len();
        let bytes = w.into_bytes();

        let mut guard = AllocGuard::new(&Limits::relaxed());
        let mut r = BitReader::new(&bytes);
        let mut decoder =
            SymbolDecoder::open(&mut r, num_contexts, &mut guard).expect("bundle opens");
        for &(ctx, value) in values {
            assert_eq!(decoder.read_uint(&mut r, ctx).expect("value"), value);
        }
        decoder.finish().expect("terminal state");
        assert_eq!(r.total_bits_read(), bits, "no bits left unread");
    }

    #[test]
    fn a_multi_context_stream_round_trips_in_both_modes() {
        let values: Vec<(usize, u32)> = (0..500u32)
            .map(|i| ((i % 4) as usize, (i * 7) % 130))
            .collect();
        round_trip(CodingMode::Ans, &values, 4);
        round_trip(CodingMode::Prefix, &values, 4);
    }

    #[test]
    fn an_empty_stream_round_trips() {
        round_trip(CodingMode::Ans, &[], 1);
        round_trip(CodingMode::Prefix, &[], 1);
    }

    #[test]
    fn a_single_symbol_stream_round_trips() {
        let values: Vec<(usize, u32)> = vec![(0, 3); 100];
        round_trip(CodingMode::Ans, &values, 1);
        round_trip(CodingMode::Prefix, &values, 1);
    }

    #[test]
    fn values_outside_the_census_are_rejected() {
        let config = HybridUintConfig::new(4, 0, 0).expect("legal");
        let mut census = TokenCensus::new(1).expect("census");
        census.record(0, 1).expect("records");
        census.record(0, 2).expect("records");
        let plan = EncoderPlan::identity(1, CodingMode::Ans, config).expect("plan");
        let tables = EntropyTables::build(&plan, &census).expect("tables");
        let mut encoder = SymbolEncoder::new(&tables);
        assert!(encoder.push_uint(0, 1).is_ok());
        assert!(
            encoder.push_uint(0, 4000).is_err(),
            "a token the census never saw has no mass"
        );
        assert!(encoder.push_uint(1, 1).is_err(), "unknown context");
    }

    #[test]
    fn the_raw_histogram_counts_small_and_large_values() {
        let mut histogram = RawHistogram::new();
        histogram.record(0);
        histogram.record(31);
        histogram.record(32);
        histogram.add(1_000_000, 5);
        histogram.record(32);
        assert_eq!(histogram.count(0), 1);
        assert_eq!(histogram.count(31), 1);
        assert_eq!(histogram.count(32), 2);
        assert_eq!(histogram.count(1_000_000), 5);
        assert_eq!(histogram.count(7), 0);
        assert_eq!(histogram.total(), 9);
        let values: Vec<(u32, u64)> = histogram.iter().collect();
        assert_eq!(values, vec![(0, 1), (31, 1), (32, 2), (1_000_000, 5)]);
    }
}
