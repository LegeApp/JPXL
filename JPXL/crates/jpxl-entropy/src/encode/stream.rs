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
//! When [`EncoderPlan::lz77`] is [`None`], the `lz77.enabled` flag is written
//! as false — complete and legal framing with no distance context and no
//! `lz_len_conf`. When `Some`, Table C.1 fields and the length configuration
//! are written, the context map must already include the distance context as
//! its **last** entry (C.2.1 appends it on the read side from the caller's
//! value-context count), and [`SymbolEncoder::push_copy`] emits back-references.
//! Length trigger tokens are counted with [`TokenCensus::record_token`] because
//! they bypass the value-cluster hybrid-uint configuration.
//!
//! Every path here is **bit-exact** (see `docs/PLAN.md`).

use jpxl_bitstream::BitWriter;

use crate::error::{Result, encode_error};
use crate::hybrid::{HybridUintConfig, bit_width};

use super::ans::{AnsEncodeTable, AnsSymbol, encode_symbols};
use super::cluster::{ContextMap, ContextMapForm};
use super::histogram::Histogram;
use super::lz77::Lz77EncodeParams;
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

/// The census pass: raw value counts (and optional direct tokens) per context.
///
/// Ordinary integers go through [`record`](Self::record) and are later
/// hybrid-uint tokenized under the cluster configuration. LZ77 length-trigger
/// tokens go through [`record_token`](Self::record_token): they are already
/// alphabet symbols and must not be re-tokenized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenCensus {
    contexts: Vec<RawHistogram>,
    /// Pre-tokenized symbols (LZ77 length triggers), same indexing as `contexts`.
    tokens: Vec<RawHistogram>,
}

impl TokenCensus {
    /// A census over `num_contexts` pre-clustered contexts.
    ///
    /// When LZ77 is enabled this must equal the context map's `num_dist`
    /// (value contexts **plus** the distance context).
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
            tokens: vec![RawHistogram::new(); num_contexts],
        })
    }

    /// Records one hybrid-uint *value* in `ctx` (literals and distance values).
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

    /// Records one already-tokenized alphabet symbol in `ctx`.
    ///
    /// Used for LZ77 length-trigger tokens (`>= min_symbol`), which the decoder
    /// does not pass through the value-cluster hybrid-uint configuration.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Encode`](crate::EntropyError::Encode) if `ctx` is outside
    /// this census.
    pub fn record_token(&mut self, ctx: usize, token: u32) -> Result<()> {
        self.tokens
            .get_mut(ctx)
            .ok_or_else(|| encode_error!("C.2.1: context {ctx} is outside this census"))?
            .record(token);
        Ok(())
    }

    /// Records a back-reference for census: length token on `ctx`, distance on
    /// the stream's distance context.
    ///
    /// # Errors
    ///
    /// If tokenization fails or either context is out of range.
    pub fn record_copy(
        &mut self,
        ctx: usize,
        length: u32,
        raw_distance: u32,
        dist_ctx: usize,
        lz77: &Lz77EncodeParams,
    ) -> Result<()> {
        let length_tok = lz77.tokenize_length(length)?;
        self.record_token(ctx, length_tok.token)?;
        self.record(dist_ctx, raw_distance)
    }

    /// The raw-value histogram of one context.
    #[must_use]
    pub fn context(&self, ctx: usize) -> Option<&RawHistogram> {
        self.contexts.get(ctx)
    }

    /// Number of pre-clustered contexts.
    #[must_use]
    pub fn num_contexts(&self) -> usize {
        self.contexts.len()
    }

    /// Merges an independently collected census into this one.
    ///
    /// Counts are integers, so callers may collect independent groups in
    /// parallel and merge them in their canonical order without changing the
    /// resulting entropy tables.
    ///
    /// # Errors
    ///
    /// If the two censuses cover different context counts.
    pub fn merge_from(&mut self, other: Self) -> Result<()> {
        if self.contexts.len() != other.contexts.len() || self.tokens.len() != other.tokens.len() {
            return Err(encode_error!(
                "C.2.1: cannot merge censuses with different context counts"
            ));
        }
        for (target, source) in self.contexts.iter_mut().zip(other.contexts) {
            for (value, count) in source.iter() {
                target.add(value, count);
            }
        }
        for (target, source) in self.tokens.iter_mut().zip(other.tokens) {
            for (value, count) in source.iter() {
                target.add(value, count);
            }
        }
        Ok(())
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
    ///
    /// When [`lz77`](Self::lz77) is `Some`, this map must include the distance
    /// context as its **last** entry (one more than the caller's value-context
    /// count). [`identity_with_lz77`](Self::identity_with_lz77) builds that shape.
    pub context_map: ContextMap,
    /// Which C.2.2 encoding to use for it.
    pub context_map_form: ContextMapForm,
    /// One hybrid-uint configuration per cluster (18181-1 C.2.3).
    pub configs: Vec<HybridUintConfig>,
    /// ANS `log_alphabet_size`; `None` derives the narrowest legal value.
    /// Ignored for prefix codes, which C.2.1 fixes at 15.
    pub log_alphabet_size: Option<u32>,
    /// LZ77 back-references; `None` writes `lz77.enabled = false`.
    pub lz77: Option<Lz77EncodeParams>,
}

impl EncoderPlan {
    /// A plan with one cluster per context and the same configuration for all
    /// of them. LZ77 is disabled.
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
            lz77: None,
        })
    }

    /// Identity value contexts plus a distance context sharing cluster 0.
    ///
    /// The map has `num_value_contexts + 1` entries: contexts `0..n` are the
    /// identity clustering, and the final entry (the distance context) maps to
    /// cluster `0`. That matches the handmade LZ77 fixture layout and is a
    /// good default when a single cluster codes both values and distances.
    ///
    /// # Errors
    ///
    /// As [`identity`](Self::identity), or if the LZ77 parameters are illegal.
    pub fn identity_with_lz77(
        num_value_contexts: usize,
        mode: CodingMode,
        config: HybridUintConfig,
        lz77: Lz77EncodeParams,
    ) -> Result<Self> {
        config.validate()?;
        lz77.validate()?;
        if num_value_contexts == 0 {
            return Err(encode_error!("C.2.1: num_dist must be at least 1"));
        }
        let mut clusters = Vec::with_capacity(num_value_contexts + 1);
        for i in 0..num_value_contexts {
            let cluster = u8::try_from(i).map_err(|_| {
                encode_error!("C.2.2: {num_value_contexts} contexts exceed the cluster range")
            })?;
            clusters.push(cluster);
        }
        // Distance context shares cluster 0 (C.2.1 appends it after value contexts).
        clusters.push(0);
        let context_map = ContextMap::new(clusters)?;
        let configs = vec![config; context_map.num_clusters()];
        Ok(Self {
            mode,
            context_map,
            context_map_form: ContextMapForm::Auto,
            configs,
            log_alphabet_size: None,
            lz77: Some(lz77),
        })
    }

    /// A plan with a caller-supplied clustering and one configuration per
    /// cluster. LZ77 is disabled; set [`lz77`](Self::lz77) afterwards if needed
    /// (and size the map with a trailing distance context).
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
            lz77: None,
        })
    }

    /// Pre-clustered context index reserved for LZ77 distances, if enabled.
    ///
    /// Always the last entry of the context map when LZ77 is on.
    #[must_use]
    pub fn lz_dist_ctx(&self) -> Option<usize> {
        self.lz77
            .as_ref()
            .map(|_| self.context_map.num_dist().saturating_sub(1))
    }

    /// Number of *value* contexts (excludes the distance context when LZ77 is on).
    #[must_use]
    pub fn num_value_contexts(&self) -> usize {
        match self.lz77 {
            Some(_) => self.context_map.num_dist().saturating_sub(1),
            None => self.context_map.num_dist(),
        }
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
    lz77: Option<Lz77EncodeParams>,
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
        if let Some(lz77) = &plan.lz77 {
            lz77.validate()?;
            if plan.context_map.num_dist() < 2 {
                return Err(encode_error!(
                    "C.2.1: LZ77 requires a distance context, so num_dist must be at least 2"
                ));
            }
        }
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
        // cluster, under that cluster's configuration. Direct tokens (LZ77
        // length triggers) are added without re-tokenization.
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
            let direct = census
                .tokens
                .get(ctx)
                .ok_or_else(|| encode_error!("C.2.1: context {ctx} is outside the census"))?;
            for (token, count) in direct.iter() {
                let token = token as usize;
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
            lz77: plan.lz77,
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

    /// LZ77 parameters if the bundle enables back-references.
    #[must_use]
    pub const fn lz77(&self) -> Option<Lz77EncodeParams> {
        self.lz77
    }

    /// Pre-clustered index of the distance context when LZ77 is enabled.
    #[must_use]
    pub fn lz_dist_ctx(&self) -> Option<usize> {
        self.lz77
            .as_ref()
            .map(|_| self.context_map.num_dist().saturating_sub(1))
    }

    /// Number of value contexts (excludes the distance context when LZ77 is on).
    ///
    /// Pass this to [`SymbolDecoder::open`](crate::SymbolDecoder::open): the
    /// decoder appends the distance context itself when it sees the flag.
    #[must_use]
    pub fn num_value_contexts(&self) -> usize {
        match self.lz77 {
            Some(_) => self.context_map.num_dist().saturating_sub(1),
            None => self.context_map.num_dist(),
        }
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
        // Table C.1 + optional lz_len_conf (C.2.1).
        match &self.lz77 {
            None => w.write_bool(false),
            Some(params) => params.write_enabled(w)?,
        }

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
    /// When LZ77 is enabled the hybrid-uint **token** must be strictly below
    /// `min_symbol`; otherwise the decoder would treat it as a copy.
    ///
    /// # Errors
    ///
    /// [`EntropyError::Encode`](crate::EntropyError::Encode) if `ctx` is
    /// unknown, if the token collides with the LZ77 range, or if the value's
    /// token is outside the alphabet its cluster's code covers — which means
    /// the census this stream's tables were built from did not include this
    /// value.
    pub fn push_uint(&mut self, ctx: usize, value: u32) -> Result<()> {
        if let Some(dist) = self.tables.lz_dist_ctx()
            && ctx == dist
        {
            return Err(encode_error!(
                "C.3.3: context {ctx} is the LZ77 distance context; use push_copy for distances"
            ));
        }
        let cluster = self.tables.context_map.cluster_of(ctx)?;
        let config = self
            .tables
            .configs
            .get(cluster)
            .ok_or_else(|| encode_error!("C.2.1: cluster {cluster} has no configuration"))?;
        let split = config.tokenize(value)?;
        if let Some(lz77) = self.tables.lz77
            && split.token >= lz77.min_symbol
        {
            return Err(encode_error!(
                "C.3.3: literal value {value} tokenizes to {} which is >= min_symbol {}; \
                 raise min_symbol or change the hybrid-uint configuration",
                split.token,
                lz77.min_symbol
            ));
        }
        self.push_event(cluster, split.token, split.extra_bits, split.extra, value)?;
        Ok(())
    }

    /// Records an LZ77 back-reference (18181-1 C.3.3).
    ///
    /// Emits a length-trigger token on `ctx` (with `lz_len_conf` extra bits)
    /// followed by a hybrid-uint distance on the distance context. The decoder
    /// reconstructs `length` symbols from the window at the resolved distance.
    ///
    /// `raw_distance` is the value *before* the C.3.3 distance transform: with
    /// `dist_multiplier == 0` the decoder uses `raw_distance + 1` as the window
    /// offset, so `0` means "one symbol back".
    ///
    /// # Errors
    ///
    /// If LZ77 is disabled, `length` is below `min_length`, `ctx` is invalid,
    /// or either token is missing from the alphabet.
    pub fn push_copy(&mut self, ctx: usize, length: u32, raw_distance: u32) -> Result<()> {
        let lz77 = self.tables.lz77.ok_or_else(|| {
            encode_error!("C.3.3: push_copy requires EncoderPlan::lz77 to be set")
        })?;
        let dist_ctx = self.tables.lz_dist_ctx().ok_or_else(|| {
            encode_error!("C.3.3: LZ77 is enabled but no distance context is present")
        })?;
        if let Some(d) = self.tables.lz_dist_ctx()
            && ctx == d
        {
            return Err(encode_error!(
                "C.3.3: copy trigger must use a value context, not the distance context"
            ));
        }

        let length_tok = lz77.tokenize_length(length)?;
        let value_cluster = self.tables.context_map.cluster_of(ctx)?;
        self.push_event(
            value_cluster,
            length_tok.token,
            length_tok.extra_bits,
            length_tok.extra,
            length_tok.token,
        )?;

        let dist_cluster = self.tables.context_map.cluster_of(dist_ctx)?;
        let dist_config =
            self.tables.configs.get(dist_cluster).ok_or_else(|| {
                encode_error!("C.2.1: cluster {dist_cluster} has no configuration")
            })?;
        let dist_split = dist_config.tokenize(raw_distance)?;
        self.push_event(
            dist_cluster,
            dist_split.token,
            dist_split.extra_bits,
            dist_split.extra,
            raw_distance,
        )?;
        Ok(())
    }

    fn push_event(
        &mut self,
        cluster: usize,
        token: u32,
        extra_bits: u32,
        extra: u32,
        label: u32,
    ) -> Result<()> {
        match &self.tables.codes {
            ClusterCodes::Prefix(codes) => {
                let code = codes
                    .get(cluster)
                    .ok_or_else(|| encode_error!("C.2.1: cluster {cluster} has no prefix code"))?;
                if code.constant_symbol() != Some(token) && code.code_length(token) == 0 {
                    return Err(encode_error!(
                        "C.2.4: token {token} of value {label} is not in the code of cluster \
                         {cluster}"
                    ));
                }
            }
            ClusterCodes::Ans { tables, .. } => {
                let table = tables
                    .get(cluster)
                    .ok_or_else(|| encode_error!("C.2.1: cluster {cluster} has no distribution"))?;
                if table.probability(token) == 0 {
                    return Err(encode_error!(
                        "C.2.5: token {token} of value {label} has no probability mass in cluster \
                         {cluster}"
                    ));
                }
            }
        }
        self.events.push(Event {
            cluster,
            token,
            extra_bits,
            extra,
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

    #[test]
    fn merged_group_censuses_equal_one_serial_census() {
        let values = [(0usize, 3u32), (1, 40), (0, 3), (1, 1_000_000)];
        let mut serial = TokenCensus::new(2).expect("serial census");
        for &(context, value) in &values {
            serial.record(context, value).expect("serial record");
        }

        let mut left = TokenCensus::new(2).expect("left census");
        let mut right = TokenCensus::new(2).expect("right census");
        for &(context, value) in &values[..2] {
            left.record(context, value).expect("left record");
        }
        for &(context, value) in &values[2..] {
            right.record(context, value).expect("right record");
        }
        left.merge_from(right).expect("compatible censuses");
        assert_eq!(left, serial);
    }
}
