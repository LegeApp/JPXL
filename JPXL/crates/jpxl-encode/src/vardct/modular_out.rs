//! A modular sub-bitstream writer for VarDCT's *control* images (Annex H).
//!
//! VarDCT is not only DCT coefficients. G.2.2's quantized LF planes and
//! G.2.4's `XFromY`/`BFromY`/`BlockInfo`/`Sharpness` planes are all carried as
//! ordinary modular sub-bitstreams — but with **per-channel dimensions**, which
//! is what [`crate::modular`] cannot express: that module encodes one frame's
//! worth of equally sized planes and knows about group rectangles, RCT and the
//! `LfGlobal`/pass-group split. None of that applies here. A control image is a
//! standalone list of small channels of unrelated shapes, written whole.
//!
//! # Coding choices (Phase Q0b)
//!
//! Until Phase Q0b every control image went out under a one-leaf tree and one
//! flat prefix code sized by the largest residual in the stream. Q0's
//! attribution (`--sections`) put the LF-group sections at ~38% of a 1 bpp
//! photograph — a 57,600-block Sharpness plane of constant zeros cost 3 bits a
//! sample, and the LF planes paid ~30% over their order-0 entropy. This writer
//! now spends a little encoder time on the model the decoder already supports:
//!
//! | Field | Value | Clause |
//! |---|---|---|
//! | `use_global_tree` | false | H.2 |
//! | `wp_params` | all-default | H.5.1 |
//! | `nb_transforms` | 0 | H.2 |
//! | MA tree | channel-index chain, then per-channel splits on static Table H.4 neighbourhood properties (never property 15), Gradient at every leaf, offset 0, multiplier 1 | H.4.1, H.4.2, Table H.3 |
//! | entropy | one context per leaf, ANS or prefix codes (whichever is shorter), one searched hybrid-uint configuration, no LZ77 | C.2 |
//!
//! The tree is learnt per stream by a bounded greedy search
//! ([`learn_channel_tree`]): every leaf tries every property of a fixed set
//! against a fixed threshold grid, priced by the token entropy of the two sides
//! minus a per-context table penalty, on a row-subsample when the channel is
//! large. Property 15 (`max_error`) is never used, so the decoder's
//! self-correcting predictor stays idle, and the previous-channel properties
//! are used only where H.4.1 defines them (an earlier channel of identical
//! shape). Every property is evaluated here exactly as `jpxl-decode`'s
//! `PropertyBuilder` evaluates it — the oracle tests round-trip through it —
//! and this file keeps its own copy of those rules on purpose: sharing them
//! with the lossless track would mean sharing a coordinate convention that is
//! not the same on both sides (a channel here is standalone, there it is a
//! rectangle of a wider plane).
//!
//! # The empty case is not a special case, it is the clause
//!
//! H.1 says a sub-bitstream over zero channels is not read at all — not an
//! empty header, *nothing*. Both places this module is called from can hit it:
//! a kVarDCT frame's `GlobalModular` (G.1.3) has `num_channels == num_extra`,
//! which is zero without extra channels, and so do `ModularLfGroup` (G.2.3)
//! and the modular half of a `PassGroup` (G.4.2). Writing a header there would
//! desynchronise every field after it, so [`write_modular_stream`] returns
//! without emitting a bit. Channels of zero width or height inside a
//! non-empty list are skipped exactly as H.2 says, but they still count in the
//! channel-index property.

use jpxl_bitstream::{BitWriter, U32Dist, U32Spec};
use jpxl_core::modular_weighted::narrow_to_i32;
use jpxl_entropy::encode::{
    CodingMode, EncoderPlan, EntropyTables, TokenCensus, TokenTapeRecorder,
};

use crate::entropy::{TREE_CODE, pack_signed};
use crate::error::{EncodeError, Result};
use crate::modular::{best_hybrid_config, seed_empty_contexts};

/// 18181-1 H.2: `U32(0, 1, 2 + u(4), 18 + u(8))` for `nb_transforms`.
const NB_TRANSFORMS_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(0),
    U32Dist::Val(1),
    U32Dist::BitsOffset { bits: 4, offset: 2 },
    U32Dist::BitsOffset {
        bits: 8,
        offset: 18,
    },
]);

/// Number of pre-clustered contexts the MA-tree stream uses (18181-1 H.4.2).
const TREE_NUM_CONTEXTS: usize = 6;

/// Table H.3 row 5: `clamp(W + N - NW, min(W, N), max(W, N))`.
const PREDICTOR_GRADIENT: u32 = 5;

/// Table H.4 property 0: the channel index.
const PROPERTY_CHANNEL: u32 = 0;

/// Table H.4 has sixteen static properties before the per-channel ones.
const NUM_STATIC_PROPERTIES: u32 = 16;

/// The static Table H.4 properties the learner may split on: `|N|`, `|W|`,
/// `W - property9(x-1, y)`, `W - NW`, `NW - N`, `N - NE`, `N - NN`, `W - WW`.
///
/// Properties 2/3 (`y`/`x`) would only learn the picture, 6/7 (`N`/`W`) add
/// nothing over their magnitudes for residual statistics, 9 is the predictor
/// itself, and 15 would oblige the decoder to run the self-correcting
/// predictor for every sample.
const STATIC_SPLIT_PROPERTIES: [u32; 8] = [4, 5, 8, 10, 11, 12, 13, 14];

/// Whether the learner may split on H.4.1's previous-channel properties
/// (`abs(rC - rG)` of every earlier channel of identical shape, Table H.4
/// index `16 + 4j + 2`).
///
/// **Off in production.** The evaluation here follows H.4.1 to the letter
/// (`rW` is zero at the left edge, `rN`/`rNW` fall back to `rW`, channels
/// counted from `i - 1` downwards, identical shape and shifts only), and both
/// `jpxl-decode` and djxl decode such streams sample-exactly — but jxl-oxide
/// 0.12.6 reads them differently and fails its ANS final-state check
/// ("ANS stream verification failed") on every stream whose tree uses one.
/// They were worth about 0.3% of the file on the mid photo (671,281 vs
/// 669,109 bytes at fixed decisions), which is not worth losing a decoder
/// over. The code stays, exercised by the round-trip tests through
/// `jpxl-decode`, for the day the third-party decoder catches up.
const USE_PREVIOUS_CHANNEL_PROPERTIES: bool = false;

/// Threshold grid for a property (`property > value` sends a sample left).
///
/// One symmetric ladder serves both the magnitude properties (whose negative
/// half is simply never chosen) and the signed differences.
const THRESHOLDS: [i32; 21] = [
    -256, -128, -64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
];

/// A leaf smaller than this is not split further.
const MIN_LEAF_SAMPLES: usize = 512;

/// Splits per channel stop at this depth (so at most `2^MAX_DEPTH` leaves).
/// Measured on the mid photo at fixed decisions: depth 0 (per-channel
/// histograms only) 678,115 B, depth 1 672,092, depth 2 669,775, depth 3
/// 669,168, depth 4 669,109 — against 738,930 B under the flat code.
const MAX_DEPTH: u32 = 3;

/// Bits a split must save beyond its own cost: one more ANS histogram (or
/// prefix code) in the bundle plus the tree node. A generous, deliberately
/// pessimistic constant so the learner never chases noise.
const SPLIT_PENALTY_BITS: f64 = 30.0 * 8.0;

/// A channel with more samples than this is learnt on a row subsample.
const LEARN_SAMPLE_TARGET: usize = 8 * 1024;

/// One channel of a control image: `width x height` samples in raster order.
#[derive(Debug, Clone, Copy)]
pub struct OutChannel<'a> {
    /// Columns.
    pub width: u32,
    /// Rows.
    pub height: u32,
    /// `width * height` samples, row-major.
    pub samples: &'a [i32],
}

impl OutChannel<'_> {
    /// H.2's "skipping any channels having width or height zero".
    fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    #[cfg(test)]
    fn at(&self, x: u32, y: u32) -> i64 {
        let index = usize::try_from(u64::from(y) * u64::from(self.width) + u64::from(x));
        index
            .ok()
            .and_then(|i| self.samples.get(i))
            .map_or(0, |&v| i64::from(v))
    }
}

/// Writes a whole modular sub-bitstream carrying `channels` (18181-1 H.2).
///
/// Emits nothing at all when `channels` is empty or every channel is empty —
/// see the module documentation.
///
/// # Errors
///
/// [`EncodeError::SampleCountMismatch`] if a channel's sample slice is not
/// `width * height` long, [`EncodeError::ValueOutOfRange`] if a residual is
/// outside what the entropy code can carry, or a bit writer error.
pub fn write_modular_stream(w: &mut BitWriter, channels: &[OutChannel<'_>]) -> Result<()> {
    write_modular_stream_with(w, channels, USE_PREVIOUS_CHANNEL_PROPERTIES)
}

/// [`write_modular_stream`] with the previous-channel properties switch
/// exposed, so the round-trip tests can keep [`USE_PREVIOUS_CHANNEL_PROPERTIES`]'s
/// off-by-default path honest against `jpxl-decode`.
///
/// # Errors
///
/// As [`write_modular_stream`].
pub fn write_modular_stream_with(
    w: &mut BitWriter,
    channels: &[OutChannel<'_>],
    previous_channel_properties: bool,
) -> Result<()> {
    if channels.is_empty() {
        return Ok(());
    }
    for channel in channels {
        let expected = u64::from(channel.width) * u64::from(channel.height);
        let found = u64::try_from(channel.samples.len()).unwrap_or(u64::MAX);
        if found != expected {
            return Err(EncodeError::SampleCountMismatch { expected, found });
        }
    }

    // Residuals and the property columns of every non-empty channel, then a
    // tree per channel, then the whole stream under one set of tables.
    let mut coded: Vec<CodedChannel> = Vec::with_capacity(channels.len());
    for (index, channel) in channels.iter().enumerate() {
        if channel.is_empty() {
            continue;
        }
        coded.push(CodedChannel::analyse(
            channels,
            index,
            previous_channel_properties,
        )?);
    }
    let trees: Vec<ChannelTree> = coded.iter().map(learn_channel_tree).collect();
    let tree = StreamTree::assemble(&coded, &trees);

    let mut contexts: Vec<Vec<u32>> = Vec::with_capacity(coded.len());
    for (channel, (local, base)) in coded.iter().zip(&tree.leaf_maps) {
        contexts.push(local.contexts(channel, base));
    }

    write_modular_header(w)?;
    tree.write(w)?;
    write_residual_payload(w, tree.num_contexts, &coded, &contexts)
}

/// Writes Table H.1 with no transforms.
fn write_modular_header(w: &mut BitWriter) -> Result<()> {
    w.write_bool(false); // use_global_tree
    w.write_bool(true); // WPHeader: default_wp
    w.write_u32(&NB_TRANSFORMS_SPEC, 0)?;
    Ok(())
}

/// Residual values below this are counted in a dense array before the census.
const DENSE_VALUES: usize = 4096;

/// A stream with fewer symbols than this also tries prefix codes; larger
/// streams are ANS-coded outright (fractional bits per symbol win as soon as
/// the table cost is amortised, and one full trial encode is not free).
const PREFIX_TRIAL_MAX_SYMBOLS: usize = 4096;

/// The residuals of every coded channel under one hybrid-uint configuration:
/// a census chooses the configuration, a token tape records the stream once
/// and its per-cluster counts build the tables, ANS by default and prefix
/// codes when a small stream is shorter that way.
fn write_residual_payload(
    w: &mut BitWriter,
    num_contexts: usize,
    coded: &[CodedChannel],
    contexts: &[Vec<u32>],
) -> Result<()> {
    let num_contexts = num_contexts.max(1);
    // Dense per-context value counts first (one increment per sample), then
    // one census entry per distinct value: the census's sparse tail is sorted
    // on insert, which is the wrong shape for 200,000 samples.
    let mut dense: Vec<u32> = vec![0; num_contexts * DENSE_VALUES];
    let mut sparse: Vec<Vec<(u32, u64)>> = vec![Vec::new(); num_contexts];
    let mut total = 0usize;
    for (channel, ctxs) in coded.iter().zip(contexts) {
        for (&value, &ctx) in channel.packed.iter().zip(ctxs) {
            let ctx = ctx as usize;
            if let Some(slot) = dense.get_mut(ctx * DENSE_VALUES + value as usize)
                && (value as usize) < DENSE_VALUES
            {
                *slot += 1;
            } else if let Some(tail) = sparse.get_mut(ctx) {
                tail.push((value, 1));
            }
        }
        total += channel.packed.len();
    }
    let mut census = TokenCensus::new(num_contexts)?;
    for ctx in 0..num_contexts {
        for (value, &count) in dense
            .get(ctx * DENSE_VALUES..(ctx + 1) * DENSE_VALUES)
            .unwrap_or(&[])
            .iter()
            .enumerate()
        {
            if count != 0 {
                census.record_many(
                    ctx,
                    u32::try_from(value).unwrap_or(u32::MAX),
                    u64::from(count),
                )?;
            }
        }
        for &(value, count) in sparse.get(ctx).map_or(&[][..], Vec::as_slice) {
            census.record_many(ctx, value, count)?;
        }
    }
    seed_empty_contexts(&mut census, num_contexts, total == 0);
    let config = best_hybrid_config(&census, num_contexts, None)?;

    let ans_plan = EncoderPlan::identity(num_contexts, CodingMode::Ans, config)?;
    let mut recorder = TokenTapeRecorder::new(&ans_plan)?;
    for (channel, ctxs) in coded.iter().zip(contexts) {
        for (&value, &ctx) in channel.packed.iter().zip(ctxs) {
            recorder.record(ctx as usize, value)?;
        }
    }
    let tape = recorder.take_tape();
    let mut counts = recorder.into_counts();
    // A context the tree defines but no sample reached still needs a legal
    // (one-symbol) distribution; the census was seeded the same way.
    for slot in counts.iter_mut() {
        if slot.iter().all(|&c| c == 0) {
            if slot.is_empty() {
                slot.push(0);
            }
            if let Some(first) = slot.first_mut() {
                *first = 1;
            }
        }
    }

    let encode = |plan: &EncoderPlan, counts: Vec<Vec<u64>>| -> Result<BitWriter> {
        let tables = EntropyTables::build_from_token_counts(plan, counts)?;
        let mut scratch = BitWriter::new();
        tables.write_bundle(&mut scratch)?;
        tape.write_stream(&tables, &mut scratch)?;
        Ok(scratch)
    };
    let mut best = encode(&ans_plan, counts.clone())?;
    if total < PREFIX_TRIAL_MAX_SYMBOLS {
        let prefix_plan = EncoderPlan::identity(num_contexts, CodingMode::Prefix, config)?;
        let prefix = encode(&prefix_plan, counts)?;
        if prefix.bit_len() < best.bit_len() {
            best = prefix;
        }
    }
    w.append_writer(&best)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Residuals and properties
// ---------------------------------------------------------------------------

/// The residual stream of one non-empty channel plus the property columns the
/// learner may split on.
struct CodedChannel {
    /// Index in the caller's channel list (Table H.4 property 0).
    channel_index: u32,
    width: u32,
    height: u32,
    /// `pack_signed(sample - gradient prediction)`, raster order.
    packed: Vec<u32>,
    /// The Table H.4 property index of every column of `props`.
    properties: Vec<u32>,
    /// `packed.len() * properties.len()` values, sample-major.
    props: Vec<i32>,
}

impl CodedChannel {
    /// Residuals and properties of `channels[index]`.
    ///
    /// `previous_channel_properties` offers H.4.1's per-earlier-channel
    /// `|rC - rG|` to the learner; production passes
    /// [`USE_PREVIOUS_CHANNEL_PROPERTIES`].
    fn analyse(
        channels: &[OutChannel<'_>],
        index: usize,
        previous_channel_properties: bool,
    ) -> Result<Self> {
        let channel = channels
            .get(index)
            .copied()
            .ok_or_else(|| EncodeError::unsupported("a control channel past its list", "H.2"))?;
        // H.4.1: earlier channels of identical shape, most recent first, each
        // contributing four properties; only `|rC - rG|` (the third) is offered
        // to the learner.
        let mut properties: Vec<u32> = STATIC_SPLIT_PROPERTIES.to_vec();
        let mut previous: Vec<OutChannel<'_>> = Vec::new();
        for earlier in channels.iter().take(index).rev() {
            if previous_channel_properties
                && earlier.width == channel.width
                && earlier.height == channel.height
            {
                // Slots count matching channels only, most recent first.
                let slot = u32::try_from(previous.len()).unwrap_or(u32::MAX);
                properties.push(NUM_STATIC_PROPERTIES + 4 * slot + 2);
                previous.push(*earlier);
            }
        }
        // Zero-size channels are skipped by H.2 and take no part in the
        // "identical shape" comparison because the current one is non-empty.

        let width = usize::try_from(channel.width).unwrap_or(usize::MAX).max(1);
        let n = channel.samples.len();
        let num_props = properties.len();
        let mut packed = Vec::with_capacity(n);
        let mut props: Vec<i32> = Vec::with_capacity(n * num_props);
        let mut prev_row: Option<&[i32]> = None;
        let mut prev2_row: Option<&[i32]> = None;
        // The same rows of every earlier identical-shape channel.
        let mut other_rows: Vec<(&[i32], Option<&[i32]>)> = Vec::with_capacity(previous.len());
        for (y, row) in channel.samples.chunks_exact(width).enumerate() {
            other_rows.clear();
            for other in &previous {
                let cur = other.samples.get(y * width..(y + 1) * width).unwrap_or(&[]);
                let up = if y > 0 {
                    other.samples.get((y - 1) * width..y * width)
                } else {
                    None
                };
                other_rows.push((cur, up));
            }
            // Property 8 wants property 9 of the west neighbour, which is the
            // previous iteration's unclamped `W + N - NW`.
            let mut west_nb: Option<Neighbours> = None;
            for (x, &sample) in row.iter().enumerate() {
                let sample = i64::from(sample);
                let nb = Neighbours::gather(row, prev_row, prev2_row, x);
                let prediction = gradient(nb.w, nb.n, nb.nw);
                let residual = sample - prediction;
                let residual =
                    i32::try_from(residual).map_err(|_| EncodeError::ValueOutOfRange {
                        what: "modular residual",
                        value: residual,
                    })?;
                packed.push(pack_signed(residual));

                let west_gradient_error = match &west_nb {
                    Some(left) => nb.w - (left.w + left.n - left.nw),
                    None => nb.w,
                };
                props.extend_from_slice(&[
                    narrow_to_i32(nb.n.abs()),
                    narrow_to_i32(nb.w.abs()),
                    narrow_to_i32(west_gradient_error),
                    narrow_to_i32(nb.w - nb.nw),
                    narrow_to_i32(nb.nw - nb.n),
                    narrow_to_i32(nb.n - nb.ne),
                    narrow_to_i32(nb.n - nb.nn),
                    narrow_to_i32(nb.w - nb.ww),
                ]);
                for (cur, up) in &other_rows {
                    // H.4.1's previous-channel edge rules are NOT the H.3 ones:
                    // rW is zero at the left edge, and rN/rNW fall back to rW.
                    let at = |r: &[i32], i: usize| r.get(i).copied().map_or(0, i64::from);
                    let rc = at(cur, x);
                    let rw = if x > 0 { at(cur, x - 1) } else { 0 };
                    let rn = up.map_or(rw, |u| at(u, x));
                    let rnw = if x > 0 {
                        up.map_or(rw, |u| at(u, x - 1))
                    } else {
                        rw
                    };
                    let rg = gradient(rw, rn, rnw);
                    props.push(narrow_to_i32((rc - rg).abs()));
                }
                west_nb = Some(nb);
            }
            prev2_row = prev_row;
            prev_row = Some(row);
        }
        Ok(Self {
            channel_index: u32::try_from(index).unwrap_or(u32::MAX),
            width: channel.width,
            height: channel.height,
            packed,
            properties,
            props,
        })
    }

    fn num_props(&self) -> usize {
        self.properties.len()
    }

    /// Property column `column` of sample `i`.
    #[inline]
    fn prop(&self, i: usize, column: usize) -> i32 {
        self.props
            .get(i * self.num_props() + column)
            .copied()
            .unwrap_or(0)
    }
}

/// The H.3 neighbourhood of one sample, with the clause's edge substitutions
/// (which cascade: `NE` falls back to the already substituted `N`, which on the
/// first row is `W`, which at the origin is zero).
struct Neighbours {
    w: i64,
    n: i64,
    nw: i64,
    ne: i64,
    nn: i64,
    ww: i64,
}

impl Neighbours {
    #[inline]
    fn gather(row: &[i32], prev: Option<&[i32]>, prev2: Option<&[i32]>, x: usize) -> Self {
        let width = row.len();
        let cur = |cx: usize| row.get(cx).copied().map_or(0, i64::from);
        let up = |cx: usize| prev.and_then(|p| p.get(cx)).copied().map_or(0, i64::from);
        let up2 = |cx: usize| prev2.and_then(|p| p.get(cx)).copied().map_or(0, i64::from);
        let has_prev = prev.is_some();
        let w = if x > 0 {
            cur(x - 1)
        } else if has_prev {
            up(x)
        } else {
            0
        };
        let n = if has_prev { up(x) } else { w };
        let nw = if x > 0 && has_prev { up(x - 1) } else { w };
        let ne = if x + 1 < width && has_prev {
            up(x + 1)
        } else {
            n
        };
        let nn = if prev2.is_some() { up2(x) } else { n };
        let ww = if x > 1 { cur(x - 2) } else { w };
        Self {
            w,
            n,
            nw,
            ne,
            nn,
            ww,
        }
    }
}

/// Table H.3 row 5.
#[inline]
const fn gradient(w: i64, n: i64, nw: i64) -> i64 {
    let lo = if w < n { w } else { n };
    let hi = if w < n { n } else { w };
    let g = w + n - nw;
    if g < lo {
        lo
    } else if g > hi {
        hi
    } else {
        g
    }
}

/// The Table H.3 row 5 prediction at `(x, y)` of a standalone channel; the
/// readable statement of the rule [`CodedChannel::analyse`] evaluates row-wise
/// and the oracle its tests compare against.
#[cfg(test)]
fn gradient_prediction(channel: &OutChannel<'_>, x: u32, y: u32) -> i64 {
    let w = if x > 0 {
        channel.at(x - 1, y)
    } else if y > 0 {
        channel.at(x, y - 1)
    } else {
        0
    };
    let n = if y > 0 { channel.at(x, y - 1) } else { w };
    let nw = if x > 0 && y > 0 {
        channel.at(x - 1, y - 1)
    } else {
        w
    };
    gradient(w, n, nw)
}

// ---------------------------------------------------------------------------
// The per-channel tree and its learner
// ---------------------------------------------------------------------------

/// A node of one channel's subtree; leaves carry a channel-local leaf number.
#[derive(Debug, Clone, Copy)]
enum ChannelNode {
    Decision {
        /// Column of [`CodedChannel::props`].
        column: usize,
        value: i32,
        left: usize,
        right: usize,
    },
    Leaf {
        local: usize,
    },
}

/// One channel's context tree over its property columns.
#[derive(Debug, Clone)]
struct ChannelTree {
    nodes: Vec<ChannelNode>,
    num_leaves: usize,
}

impl ChannelTree {
    fn single_leaf() -> Self {
        Self {
            nodes: vec![ChannelNode::Leaf { local: 0 }],
            num_leaves: 1,
        }
    }

    /// The channel-local leaf of sample `i`.
    #[inline]
    fn leaf_of(&self, channel: &CodedChannel, i: usize) -> usize {
        let mut index = 0usize;
        loop {
            match self.nodes.get(index) {
                Some(ChannelNode::Leaf { local }) => return *local,
                Some(ChannelNode::Decision {
                    column,
                    value,
                    left,
                    right,
                }) => {
                    index = if channel.prop(i, *column) > *value {
                        *left
                    } else {
                        *right
                    };
                }
                None => return 0,
            }
        }
    }

    /// Stream contexts of every sample of `channel`, given the local-to-stream
    /// leaf map.
    fn contexts(&self, channel: &CodedChannel, base: &[u32]) -> Vec<u32> {
        (0..channel.packed.len())
            .map(|i| base.get(self.leaf_of(channel, i)).copied().unwrap_or(0))
            .collect()
    }
}

/// Cheap tokenizer for pricing splits: values below 16 are their own token,
/// larger values are split into a `(bit length, next two bits)` token plus
/// `bit length - 3` raw bits — the C.2.3 configuration `(4, 2, 0)`, which is
/// the first candidate the real configuration search tries.
#[inline]
fn cheap_token(value: u32) -> (usize, u32) {
    if value < 16 {
        (value as usize, 0)
    } else {
        let n = 32 - value.leading_zeros(); // >= 5
        let msb = (value >> (n - 3)) & 3;
        ((16 + (n - 5) * 4 + msb) as usize, n - 3)
    }
}

/// Shannon cost in bits of a token histogram, plus its raw extra bits.
fn histogram_bits(hist: &[u32], extra_bits: f64) -> f64 {
    let total: u64 = hist.iter().map(|&c| u64::from(c)).sum();
    if total == 0 {
        return 0.0;
    }
    #[allow(
        clippy::cast_precision_loss,
        reason = "counts are far below 2^53; this is a price, not a bit-exact quantity"
    )]
    let total_f = total as f64;
    let mut bits = 0.0f64;
    for &c in hist {
        if c != 0 {
            #[allow(clippy::cast_precision_loss, reason = "as above")]
            let c = f64::from(c);
            bits += c * (total_f / c).log2();
        }
    }
    bits + extra_bits
}

/// Learns a bounded context tree for one channel by greedy splitting.
///
/// Every leaf considers every property column against [`THRESHOLDS`]; the
/// best split of every leaf whose gain exceeds [`SPLIT_PENALTY_BITS`] is
/// applied in the same round (so a round costs one pass over the learning
/// sample), up to [`MAX_DEPTH`] rounds. Large channels are learnt on a row
/// subsample of about [`LEARN_SAMPLE_TARGET`] samples. Threshold bins and
/// cheap tokens are computed once per sample; a round only counts them.
fn learn_channel_tree(channel: &CodedChannel) -> ChannelTree {
    let n = channel.packed.len();
    let mut tree = ChannelTree::single_leaf();
    if n < 2 * MIN_LEAF_SAMPLES || channel.num_props() == 0 {
        return tree;
    }

    // Learning sample: whole rows, every `stride`-th, so the property columns
    // keep their real neighbourhoods.
    let width = usize::try_from(channel.width).unwrap_or(1).max(1);
    let height = usize::try_from(channel.height).unwrap_or(0);
    let stride = n.div_ceil(LEARN_SAMPLE_TARGET).max(1);
    let num_props = channel.num_props();
    let num_thresholds = THRESHOLDS.len();
    let bins = num_thresholds + 1;

    // Per learning sample: cheap token, its extra bits, and the threshold bin
    // of every property column (`u8`, there are 22 bins).
    let mut tokens: Vec<u16> = Vec::new();
    let mut extras: Vec<u8> = Vec::new();
    let mut sample_bins: Vec<u8> = Vec::new();
    let mut max_token = 0usize;
    for y in (0..height).step_by(stride) {
        let start = y * width;
        for i in start..(start + width).min(n) {
            let (t, e) = cheap_token(channel.packed.get(i).copied().unwrap_or(0));
            max_token = max_token.max(t);
            tokens.push(u16::try_from(t).unwrap_or(u16::MAX));
            extras.push(u8::try_from(e).unwrap_or(u8::MAX));
            for column in 0..num_props {
                let value = channel.prop(i, column);
                // Bin = number of thresholds strictly below the value, so a
                // split at threshold k sends bins `k+1..` left.
                let bin = THRESHOLDS.partition_point(|&t| t < value);
                sample_bins.push(u8::try_from(bin).unwrap_or(u8::MAX));
            }
        }
    }
    let alphabet = max_token + 1;
    let num_samples = tokens.len();

    let mut leaf_of: Vec<u8> = vec![0; num_samples];
    let mut depths: Vec<u32> = vec![0];

    for _round in 0..MAX_DEPTH {
        let num_leaves = tree.num_leaves;
        // Per leaf: token histogram and extra bits; per leaf x property:
        // (bin x token) counts and per-bin extra bits.
        let mut base: Vec<u32> = vec![0; num_leaves * alphabet];
        let mut base_extra: Vec<f64> = vec![0.0; num_leaves];
        let mut leaf_size: Vec<usize> = vec![0; num_leaves];
        let mut splittable: Vec<bool> = depths.iter().map(|&d| d < MAX_DEPTH).collect();
        let cell_len = bins * alphabet;
        let mut binned: Vec<u32> = vec![0; num_leaves * num_props * cell_len];
        let mut binned_extra: Vec<f64> = vec![0.0; num_leaves * num_props * bins];

        for slot in 0..num_samples {
            let leaf = usize::from(leaf_of.get(slot).copied().unwrap_or(0));
            let token = usize::from(tokens.get(slot).copied().unwrap_or(0));
            let extra = f64::from(extras.get(slot).copied().unwrap_or(0));
            if let Some(h) = base.get_mut(leaf * alphabet + token) {
                *h += 1;
            }
            if let Some(e) = base_extra.get_mut(leaf) {
                *e += extra;
            }
            if let Some(s) = leaf_size.get_mut(leaf) {
                *s += 1;
            }
            if !splittable.get(leaf).copied().unwrap_or(false) {
                continue;
            }
            let Some(row) = sample_bins.get(slot * num_props..(slot + 1) * num_props) else {
                continue;
            };
            let leaf_base = leaf * num_props;
            for (column, &bin) in row.iter().enumerate() {
                let bin = usize::from(bin);
                let cell = leaf_base + column;
                if let Some(h) = binned.get_mut(cell * cell_len + bin * alphabet + token) {
                    *h += 1;
                }
                if let Some(e) = binned_extra.get_mut(cell * bins + bin) {
                    *e += extra;
                }
            }
        }

        // Best split per leaf.
        let mut splits: Vec<Option<(usize, i32)>> = vec![None; num_leaves];
        let mut left = vec![0u32; alphabet];
        let mut right = vec![0u32; alphabet];
        for leaf in 0..num_leaves {
            let total_n = leaf_size.get(leaf).copied().unwrap_or(0);
            if total_n < 2 * MIN_LEAF_SAMPLES || !splittable.get(leaf).copied().unwrap_or(false) {
                if let Some(s) = splittable.get_mut(leaf) {
                    *s = false;
                }
                continue;
            }
            let Some(base_hist) = base.get(leaf * alphabet..(leaf + 1) * alphabet) else {
                continue;
            };
            let total_extra = base_extra.get(leaf).copied().unwrap_or(0.0);
            let base_cost = histogram_bits(base_hist, total_extra);
            let mut best_gain = SPLIT_PENALTY_BITS;
            let mut best: Option<(usize, i32)> = None;
            for column in 0..num_props {
                let cell = leaf * num_props + column;
                let Some(counts) = binned.get(cell * cell_len..(cell + 1) * cell_len) else {
                    continue;
                };
                let Some(cell_extras) = binned_extra.get(cell * bins..(cell + 1) * bins) else {
                    continue;
                };
                // Suffix accumulation: the left side at threshold k is bins
                // k+1 and up. A bin that holds no sample leaves the split
                // unchanged from the previous threshold, so it is skipped.
                left.iter_mut().for_each(|c| *c = 0);
                let mut left_extra = 0.0f64;
                let mut left_n = 0usize;
                for k in (0..num_thresholds).rev() {
                    let bin = k + 1;
                    let Some(row) = counts.get(bin * alphabet..(bin + 1) * alphabet) else {
                        continue;
                    };
                    let mut moved = 0usize;
                    for (l, &c) in left.iter_mut().zip(row) {
                        *l += c;
                        moved += c as usize;
                    }
                    if moved == 0 {
                        continue;
                    }
                    left_n += moved;
                    left_extra += cell_extras.get(bin).copied().unwrap_or(0.0);
                    let right_n = total_n.saturating_sub(left_n);
                    if left_n < MIN_LEAF_SAMPLES {
                        continue;
                    }
                    if right_n < MIN_LEAF_SAMPLES {
                        break;
                    }
                    for ((r, &b), &l) in right.iter_mut().zip(base_hist).zip(&left) {
                        *r = b.saturating_sub(l);
                    }
                    let cost = histogram_bits(&left, left_extra)
                        + histogram_bits(&right, total_extra - left_extra);
                    let gain = base_cost - cost;
                    if gain > best_gain {
                        best_gain = gain;
                        best = Some((column, THRESHOLDS.get(k).copied().unwrap_or(0)));
                    }
                }
            }
            if let Some(slot) = splits.get_mut(leaf) {
                *slot = best;
            }
        }
        if splits.iter().all(Option::is_none) {
            break;
        }

        // Apply: a split leaf keeps its number for the right side and the new
        // leaf takes the next number; the tree node is rewritten in place.
        let mut leaf_node_index: Vec<usize> = vec![usize::MAX; num_leaves];
        for (index, node) in tree.nodes.iter().enumerate() {
            if let ChannelNode::Leaf { local } = node
                && let Some(slot) = leaf_node_index.get_mut(*local)
            {
                *slot = index;
            }
        }
        // Per old leaf: (column, threshold bin boundary, new leaf number).
        let mut new_leaf_map: Vec<Option<(usize, usize, usize)>> = vec![None; num_leaves];
        for (leaf, split) in splits.iter().enumerate() {
            let Some((column, value)) = *split else {
                continue;
            };
            let Some(&node_index) = leaf_node_index.get(leaf) else {
                continue;
            };
            let new_local = tree.num_leaves;
            if new_local > usize::from(u8::MAX) {
                break;
            }
            let left_node = tree.nodes.len();
            tree.nodes.push(ChannelNode::Leaf { local: new_local });
            let right_node = tree.nodes.len();
            tree.nodes.push(ChannelNode::Leaf { local: leaf });
            if let Some(node) = tree.nodes.get_mut(node_index) {
                *node = ChannelNode::Decision {
                    column,
                    value,
                    left: left_node,
                    right: right_node,
                };
            }
            tree.num_leaves += 1;
            let depth = depths.get(leaf).copied().unwrap_or(0) + 1;
            depths.push(depth);
            if let Some(d) = depths.get_mut(leaf) {
                *d = depth;
            }
            // `value` is THRESHOLDS[k]; samples with bin > k go left.
            let k = THRESHOLDS.iter().position(|&t| t == value).unwrap_or(0);
            if let Some(slot) = new_leaf_map.get_mut(leaf) {
                *slot = Some((column, k, new_local));
            }
        }
        for slot in 0..num_samples {
            let Some(leaf_slot) = leaf_of.get_mut(slot) else {
                continue;
            };
            let leaf = usize::from(*leaf_slot);
            if let Some(Some((column, k, new_local))) = new_leaf_map.get(leaf)
                && usize::from(
                    sample_bins
                        .get(slot * num_props + column)
                        .copied()
                        .unwrap_or(0),
                ) > *k
            {
                *leaf_slot = u8::try_from(*new_local).unwrap_or(0);
            }
        }
    }
    tree
}

// ---------------------------------------------------------------------------
// The stream tree: channel chain + per-channel subtrees, BFS-serialised
// ---------------------------------------------------------------------------

/// A node of the whole stream's tree in wire order. H.4.2's breadth-first
/// decode implies every child index and leaf context from the order alone,
/// so only the decision itself goes on the wire.
#[derive(Debug, Clone, Copy)]
enum StreamNode {
    Decision { property: u32, value: i32 },
    Leaf,
}

/// The MA tree of one control-image stream and the leaf numbering H.4.2's
/// breadth-first decode assigns to it.
struct StreamTree {
    /// Nodes in breadth-first (wire) order.
    nodes: Vec<StreamNode>,
    num_contexts: usize,
    /// Per coded channel: `(its subtree, local leaf -> stream context)`.
    leaf_maps: Vec<(ChannelTree, Vec<u32>)>,
}

impl StreamTree {
    /// Chains the coded channels on property 0 (`channel > c` sends the later
    /// channels left, the channel itself right) and hangs each channel's
    /// subtree under its slot, then renumbers everything breadth-first.
    fn assemble(coded: &[CodedChannel], trees: &[ChannelTree]) -> Self {
        // Build in an arbitrary arena first (children need not follow parents).
        #[derive(Clone, Copy)]
        enum Raw {
            Decision {
                property: u32,
                value: i32,
                left: usize,
                right: usize,
            },
            Leaf {
                channel: usize,
                local: usize,
            },
        }
        let mut arena: Vec<Raw> = Vec::new();
        // Each channel's subtree, translated into the arena.
        let mut subtree_root: Vec<usize> = Vec::with_capacity(coded.len());
        for (c, (channel, tree)) in coded.iter().zip(trees).enumerate() {
            let offset = arena.len();
            for node in &tree.nodes {
                arena.push(match *node {
                    ChannelNode::Decision {
                        column,
                        value,
                        left,
                        right,
                    } => Raw::Decision {
                        property: channel.properties.get(column).copied().unwrap_or(0),
                        value,
                        left: offset + left,
                        right: offset + right,
                    },
                    ChannelNode::Leaf { local } => Raw::Leaf { channel: c, local },
                });
            }
            subtree_root.push(offset);
        }
        // The chain, from the last channel backwards: the last channel's
        // subtree is reached when every earlier test fails to send left.
        let mut root = subtree_root.last().copied().unwrap_or(0);
        for c in (0..coded.len().saturating_sub(1)).rev() {
            let index = arena.len();
            arena.push(Raw::Decision {
                property: PROPERTY_CHANNEL,
                value: narrow_to_i32(i64::from(coded.get(c).map_or(0, |ch| ch.channel_index))),
                left: root,
                right: subtree_root.get(c).copied().unwrap_or(0),
            });
            root = index;
        }
        if arena.is_empty() {
            arena.push(Raw::Leaf {
                channel: 0,
                local: 0,
            });
        }

        // Breadth-first renumbering, exactly the decoder's queue.
        let mut order: Vec<usize> = vec![root];
        let mut head = 0usize;
        while head < order.len() {
            let raw = order.get(head).copied().unwrap_or(0);
            if let Some(Raw::Decision { left, right, .. }) = arena.get(raw) {
                order.push(*left);
                order.push(*right);
            }
            head += 1;
        }
        let mut nodes: Vec<StreamNode> = Vec::with_capacity(order.len());
        let mut leaf_maps: Vec<(ChannelTree, Vec<u32>)> = trees
            .iter()
            .map(|t| (t.clone(), vec![0u32; t.num_leaves]))
            .collect();
        let mut ctx = 0usize;
        for &raw in &order {
            match arena.get(raw).copied() {
                Some(Raw::Decision {
                    property, value, ..
                }) => nodes.push(StreamNode::Decision { property, value }),
                Some(Raw::Leaf { channel, local }) => {
                    if let Some(slot) = leaf_maps
                        .get_mut(channel)
                        .and_then(|(_, map)| map.get_mut(local))
                    {
                        *slot = u32::try_from(ctx).unwrap_or(u32::MAX);
                    }
                    nodes.push(StreamNode::Leaf);
                    ctx += 1;
                }
                None => {}
            }
        }
        Self {
            nodes,
            num_contexts: ctx.max(1),
            leaf_maps,
        }
    }

    /// Writes the tree as H.4.2's breadth-first stream under [`TREE_CODE`].
    fn write(&self, w: &mut BitWriter) -> Result<()> {
        TREE_CODE.write_bundle(w, TREE_NUM_CONTEXTS)?;
        for node in &self.nodes {
            match *node {
                StreamNode::Decision { property, value } => {
                    TREE_CODE.write_uint(w, property.saturating_add(1))?; // ctx 1
                    TREE_CODE.write_uint(w, pack_signed(value))?; // ctx 0
                }
                StreamNode::Leaf => {
                    TREE_CODE.write_uint(w, 0)?; // ctx 1: property + 1 == 0 marks a leaf
                    TREE_CODE.write_uint(w, PREDICTOR_GRADIENT)?; // ctx 2: predictor
                    TREE_CODE.write_uint(w, pack_signed(0))?; // ctx 3: offset
                    TREE_CODE.write_uint(w, 0)?; // ctx 4: mul_log
                    TREE_CODE.write_uint(w, 0)?; // ctx 5: mul_bits, so multiplier = 1
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::identity_op,
    clippy::useless_conversion,
    reason = "test fixtures index known-length vectors and panic on error by design"
)]
mod tests {
    use super::*;

    #[test]
    fn zero_channels_emit_no_bits_at_all() {
        let mut w = BitWriter::new();
        write_modular_stream(&mut w, &[]).expect("writes");
        assert_eq!(
            w.bit_len(),
            0,
            "H.1: N == 0 means the decoder reads nothing"
        );
    }

    #[test]
    fn a_mismatched_channel_is_rejected() {
        let samples = [0i32; 5];
        let mut w = BitWriter::new();
        assert!(matches!(
            write_modular_stream(
                &mut w,
                &[OutChannel {
                    width: 3,
                    height: 2,
                    samples: &samples,
                }],
            ),
            Err(EncodeError::SampleCountMismatch {
                expected: 6,
                found: 5
            })
        ));
    }

    #[test]
    fn the_gradient_matches_the_clause_at_the_edges() {
        let samples = [10i32, 20, 30, 40, 50, 60];
        let channel = OutChannel {
            width: 3,
            height: 2,
            samples: &samples,
        };
        assert_eq!(gradient_prediction(&channel, 0, 0), 0);
        assert_eq!(gradient_prediction(&channel, 1, 0), 10);
        assert_eq!(gradient_prediction(&channel, 0, 1), 10);
        assert_eq!(gradient_prediction(&channel, 1, 1), 40);
    }

    #[test]
    fn row_wise_residuals_match_the_per_cell_gradient_oracle() {
        let width = 7u32;
        let height = 5u32;
        let samples: Vec<i32> = (0..(width * height))
            .map(|i| ((i * 37) % 23) as i32 - 11)
            .collect();
        let channels = [OutChannel {
            width,
            height,
            samples: &samples,
        }];
        let coded = CodedChannel::analyse(&channels, 0, true).expect("analyses");
        for y in 0..height {
            for x in 0..width {
                let expected =
                    i64::from(channels[0].at(x, y)) - gradient_prediction(&channels[0], x, y);
                let packed = coded.packed[(y * width + x) as usize];
                assert_eq!(packed, pack_signed(expected as i32), "({x}, {y})");
            }
        }
    }

    #[test]
    fn static_properties_match_table_h4_on_the_decoder_ramp() {
        // The 4x3 ramp jpxl-decode's tree tests use: at (2, 2) W = 9, N = 6,
        // NW = 5, NE = 7, NN = 2, WW = 8, and property 8 = W - 9 = 0.
        let samples: Vec<i32> = (0..12).collect();
        let channels = [OutChannel {
            width: 4,
            height: 3,
            samples: &samples,
        }];
        let coded = CodedChannel::analyse(&channels, 0, true).expect("analyses");
        let i = 2 * 4 + 2;
        let by_property = |p: u32| {
            let column = coded
                .properties
                .iter()
                .position(|&q| q == p)
                .expect("column");
            coded.prop(i, column)
        };
        assert_eq!(by_property(4), 6, "abs(N)");
        assert_eq!(by_property(5), 9, "abs(W)");
        assert_eq!(by_property(8), 0, "W - property9(x-1, y)");
        assert_eq!(by_property(10), 9 - 5, "W - NW");
        assert_eq!(by_property(11), 5 - 6, "NW - N");
        assert_eq!(by_property(12), 6 - 7, "N - NE");
        assert_eq!(by_property(13), 6 - 2, "N - NN");
        assert_eq!(by_property(14), 9 - 8, "W - WW");
        // First column: property 8 is just W.
        let j = 1 * 4;
        let column8 = coded
            .properties
            .iter()
            .position(|&q| q == 8)
            .expect("column");
        let column5 = coded
            .properties
            .iter()
            .position(|&q| q == 5)
            .expect("column");
        assert_eq!(coded.prop(j, column8), coded.prop(j, column5));
    }

    #[test]
    fn previous_channel_property_uses_h41_edge_rules() {
        // Channel 0 holds 1..=4; channel 1 is the coded one. At (0, 1):
        // rC = 3, rW = 0, rN = 1, rNW = 0 -> rG = 1, |rC - rG| = 2.
        let prev = [1i32, 2, 3, 4];
        let cur = [0i32; 4];
        let channels = [
            OutChannel {
                width: 2,
                height: 2,
                samples: &prev,
            },
            OutChannel {
                width: 2,
                height: 2,
                samples: &cur,
            },
        ];
        let coded = CodedChannel::analyse(&channels, 1, true).expect("analyses");
        assert_eq!(
            coded.properties.last().copied(),
            Some(NUM_STATIC_PROPERTIES + 2)
        );
        let column = coded.properties.len() - 1;
        assert_eq!(coded.prop(2, column), 2);
    }

    #[test]
    fn a_stream_of_two_channels_chains_on_the_channel_index() {
        let a = [5i32; 12];
        let b = [0i32; 6];
        let channels = [
            OutChannel {
                width: 4,
                height: 3,
                samples: &a,
            },
            OutChannel {
                width: 3,
                height: 2,
                samples: &b,
            },
        ];
        let coded: Vec<CodedChannel> = (0..2)
            .map(|i| CodedChannel::analyse(&channels, i, true).expect("analyses"))
            .collect();
        let trees = vec![ChannelTree::single_leaf(), ChannelTree::single_leaf()];
        let tree = StreamTree::assemble(&coded, &trees);
        assert_eq!(tree.num_contexts, 2);
        assert!(matches!(
            tree.nodes[0],
            StreamNode::Decision {
                property: PROPERTY_CHANNEL,
                value: 0,
            }
        ));
        assert_eq!(tree.nodes.len(), 3);
        // Left (channel > 0) is channel 1's leaf, decoded first -> ctx 0.
        assert_eq!(tree.leaf_maps[1].1, vec![0]);
        assert_eq!(tree.leaf_maps[0].1, vec![1]);
    }

    #[test]
    fn a_split_channel_learns_when_the_residual_scale_follows_a_property() {
        // Left half of every row is flat, right half is noisy: |W - WW| (and
        // friends) separate the two regimes.
        let width = 64u32;
        let height = 64u32;
        let mut state = 0x9E37_79B9u32;
        let samples: Vec<i32> = (0..(width * height))
            .map(|i| {
                let x = i % width;
                if x < width / 2 {
                    100
                } else {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    100 + (state % 61) as i32 - 30
                }
            })
            .collect();
        let channels = [OutChannel {
            width,
            height,
            samples: &samples,
        }];
        let coded = CodedChannel::analyse(&channels, 0, true).expect("analyses");
        let tree = learn_channel_tree(&coded);
        assert!(
            tree.num_leaves >= 2,
            "expected at least one split, got {tree:?}"
        );
        // And the stream still writes.
        let mut w = BitWriter::new();
        write_modular_stream(&mut w, &channels).expect("writes");
        assert!(w.bit_len() > 0);
    }

    #[test]
    fn a_constant_channel_costs_almost_nothing() {
        let samples = vec![7i32; 256 * 64];
        let channels = [OutChannel {
            width: 256,
            height: 64,
            samples: &samples,
        }];
        let mut w = BitWriter::new();
        write_modular_stream(&mut w, &channels).expect("writes");
        assert!(
            w.bit_len() < 64 * 8,
            "16384 identical residuals should cost a handful of bytes, not {} bits",
            w.bit_len()
        );
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "test fixtures panic on error by design")]
mod count_tests {
    use super::*;

    #[test]
    fn counting_and_storing_writers_agree_on_the_stream_length() {
        let width = 37u32;
        let height = 29u32;
        let mut state = 0x1234_5678u32;
        let samples: Vec<i32> = (0..(width * height))
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state % 200) as i32 - 100
            })
            .collect();
        let sharp = vec![0i32; (width * height) as usize];
        let channels = [
            OutChannel {
                width,
                height,
                samples: &samples,
            },
            OutChannel {
                width,
                height,
                samples: &sharp,
            },
        ];
        for prefix_bits in 0..8u32 {
            let mut store = BitWriter::new();
            let mut count = BitWriter::counting();
            if prefix_bits > 0 {
                store.write_bits(prefix_bits, 0).unwrap();
                count.write_bits(prefix_bits, 0).unwrap();
            }
            write_modular_stream(&mut store, &channels).expect("store");
            write_modular_stream(&mut count, &channels).expect("count");
            assert_eq!(store.bit_len(), count.bit_len(), "prefix {prefix_bits}");
        }
    }
}
