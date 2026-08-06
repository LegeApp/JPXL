//! The trained entropy model: `Encoder-plan1.md` §9.2 steps 2–5 and §9.3's
//! clusterer, over the raw census the walk already produces.
//!
//! # What is being chosen
//!
//! The wire's entropy model is a context map (pre-context → cluster), one
//! hybrid-uint configuration per cluster, and one distribution per cluster.
//! The writer rebuilds the ANS tables from its own exact token census, so the
//! *levers* are the map and the configurations — this module chooses both
//! from the raw value census and nothing else. Slice 18c also proposes an
//! I.2.2 HF block-context model (thresholds + clustering map); that choice
//! *does* change the event stream, so the caller re-censuses under it and
//! adopts only on an exact `price_codestream` win — the same discipline as
//! §9.4's coefficient-order pass.
//!
//! # Why the refinement bound is one re-census
//!
//! The event stream depends on the block context model and the coefficient
//! orders. Clustering and hybrid-uint alone leave it unchanged, so training
//! them needs no second walk. A proposed custom block context (or custom
//! orders) changes which pre-context each symbol lands in, so the caller
//! walks once under the candidate and trains a fresh model. The loop is
//! closed: one proposal, one re-census, exact-price adopt. The clusterer's
//! own termination is separately bounded: every merge reduces the cluster
//! count by one, so at most `n - 1` merges happen.
//!
//! # Cost model
//!
//! A merge is accepted only when it saves bits under
//! `data + histogram signaling + per-cluster overhead` (§9.3's criterion).
//! Data cost is exact Shannon over the tokens the cluster's best hybrid-uint
//! configuration produces, plus that configuration's raw bits — computed with
//! the same `HybridUintConfig::tokenize` arithmetic the writer uses. The
//! signaling terms are stated estimates (the C.2.5 serialization varies with
//! ANS normalization, which depends on counts the final table build owns);
//! they steer merges, they are not prices, and the oracle suite verifies the
//! streams the trained model produces.

use jpxl_core::varblock::{NUM_ORDER_IDS, natural_coeff_order, order_id_dims};
use jpxl_encode::vardct::ids::{ClusterId, OrderId, PresetId};
use jpxl_encode::vardct::plan::{
    DEFAULT_BLOCK_CTX_MAP, HfBlockContextPlan, HistogramPlan, HybridUintPlan, OrderSet,
    QuantizedFrameIr, SpatialPlan,
};
use jpxl_encode::vardct::{CensusSink, PlanResult, VardctGeometry};
use jpxl_entropy::HybridUintConfig;

/// At most this many QF thresholds in a proposal — each doubles (roughly)
/// `bsize`, and I.2.2 caps `bsize` at `39 * 64`.
const MAX_QF_THRESHOLDS: usize = 3;

/// The candidate hybrid-uint configurations a cluster may use, as
/// `(split_exponent, msb_in_token, lsb_in_token)`. All satisfy C.2.3's
/// `msb + lsb <= split`. The legacy frame-wide `(4, 2, 0)` is included so the
/// trainer can never do worse than slice 12's fixed choice on any cluster.
const CANDIDATE_CONFIGS: &[(u32, u32, u32)] = &[
    (4, 2, 0),
    (0, 0, 0),
    (1, 0, 0),
    (2, 1, 0),
    (3, 1, 0),
    (3, 2, 0),
    (4, 1, 0),
    (5, 2, 0),
    (6, 2, 0),
];

/// Estimated fixed bits to serialize one histogram (C.2.5 preamble, counts
/// header, ANS bookkeeping).
const HISTOGRAM_FIXED_BITS: f64 = 40.0;

/// Estimated bits per alphabet slot of a serialized histogram.
const HISTOGRAM_PER_SYMBOL_BITS: f64 = 5.5;

/// Estimated context-map and bookkeeping bits each additional cluster costs
/// beyond its histogram (C.2.2 map entropy, hybrid-uint configuration).
const CLUSTER_OVERHEAD_BITS: f64 = 24.0;

/// C.2.2's hard ceiling on distributions per pass.
const MAX_CLUSTERS: usize = 255;

/// How many sorted neighbours each cluster proposes merges with. The sort
/// key is the fingerprint, so nearby entries are the plausible merges —
/// §9.3's bucket idea as a sliding window over one ordering.
const CANDIDATE_WINDOW: usize = 16;

/// The chosen entropy model.
pub(crate) struct TrainedModel {
    /// One cluster per pre-context.
    pub context_map: Vec<ClusterId>,
    /// One distribution per cluster, as token counts under that cluster's
    /// configuration.
    pub histograms: Vec<HistogramPlan>,
    /// One configuration per cluster.
    pub hybrid_uint: Vec<HybridUintPlan>,
}

/// One cluster's state during the merge loop.
#[derive(Clone)]
struct Cluster {
    /// Which pre-contexts it covers (ascending).
    contexts: Vec<usize>,
    /// Raw value counts, ascending by value.
    values: Vec<(u32, u64)>,
    /// Cost of coding this cluster alone, in bits.
    cost: f64,
    /// The configuration that cost was achieved with.
    config: (u32, u32, u32),
    /// Bumped on every merge; stale queue entries check it.
    generation: u32,
}

/// Merges two ascending `(value, count)` lists.
fn merge_values(a: &[(u32, u64)], b: &[(u32, u64)]) -> Vec<(u32, u64)> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() || j < b.len() {
        match (a.get(i), b.get(j)) {
            (Some(&(va, ca)), Some(&(vb, cb))) if va == vb => {
                out.push((va, ca + cb));
                i += 1;
                j += 1;
            }
            (Some(&(va, ca)), Some(&(vb, _))) if va < vb => {
                out.push((va, ca));
                i += 1;
            }
            (Some(_), Some(&(vb, cb))) => {
                out.push((vb, cb));
                j += 1;
            }
            (Some(&(va, ca)), None) => {
                out.push((va, ca));
                i += 1;
            }
            (None, Some(&(vb, cb))) => {
                out.push((vb, cb));
                j += 1;
            }
            (None, None) => break,
        }
    }
    out
}

/// Exact data cost of coding `values` under `config`: Shannon over the token
/// distribution plus the raw bits every value carries.
fn data_cost(values: &[(u32, u64)], config: (u32, u32, u32)) -> Option<f64> {
    let config = HybridUintConfig::new(config.0, config.1, config.2).ok()?;
    let mut token_counts: Vec<(u32, u64)> = Vec::new();
    let mut raw_bits = 0.0f64;
    for &(value, count) in values {
        let split = config.tokenize(value).ok()?;
        #[allow(
            clippy::cast_precision_loss,
            reason = "counts and bit widths stay far inside f64's exact range"
        )]
        {
            raw_bits += count as f64 * f64::from(split.extra_bits);
        }
        match token_counts.binary_search_by_key(&split.token, |&(t, _)| t) {
            Ok(index) => {
                if let Some(entry) = token_counts.get_mut(index) {
                    entry.1 += count;
                }
            }
            Err(index) => token_counts.insert(index, (split.token, count)),
        }
    }
    let total: u64 = token_counts.iter().map(|&(_, c)| c).sum();
    if total == 0 {
        return Some(0.0);
    }
    #[allow(
        clippy::cast_precision_loss,
        reason = "counts stay far inside f64's exact range"
    )]
    let shannon: f64 = token_counts
        .iter()
        .map(|&(_, c)| {
            let p = c as f64 / total as f64;
            -(c as f64) * p.log2()
        })
        .sum();
    // Alphabet size for the signaling estimate: tokens up to the largest.
    let alphabet = token_counts.last().map_or(1, |&(t, _)| u64::from(t) + 1);
    #[allow(
        clippy::cast_precision_loss,
        reason = "alphabet sizes stay small; tokens are logarithmic in value"
    )]
    let signal = HISTOGRAM_FIXED_BITS + HISTOGRAM_PER_SYMBOL_BITS * alphabet as f64;
    Some(shannon + raw_bits + signal)
}

/// The cheapest configuration for `values` and its total cost.
fn best_config(values: &[(u32, u64)]) -> (f64, (u32, u32, u32)) {
    let mut best = (f64::INFINITY, (4u32, 2u32, 0u32));
    for &candidate in CANDIDATE_CONFIGS {
        if let Some(cost) = data_cost(values, candidate)
            && cost < best.0
        {
            best = (cost, candidate);
        }
    }
    best
}

/// A fingerprint that sorts similar raw distributions near each other: the
/// zero fraction first (the dominant axis for quantized coefficients), then
/// the mean magnitude class.
fn fingerprint(values: &[(u32, u64)]) -> (u64, u64) {
    let total: u64 = values.iter().map(|&(_, c)| c).sum();
    if total == 0 {
        return (0, 0);
    }
    let zeros = values.iter().find(|&&(v, _)| v == 0).map_or(0, |&(_, c)| c);
    let mean_class: f64 = values
        .iter()
        .map(|&(v, c)| {
            #[allow(
                clippy::cast_precision_loss,
                reason = "counts stay far inside f64's exact range"
            )]
            {
                f64::from(32 - v.leading_zeros()) * c as f64
            }
        })
        .sum::<f64>()
        / {
            #[allow(
                clippy::cast_precision_loss,
                reason = "counts stay far inside f64's exact range"
            )]
            {
                total as f64
            }
        };
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "both terms are small non-negative fractions scaled to integers"
    )]
    ((zeros * 1000 / total), (mean_class * 100.0).max(0.0) as u64)
}

/// Proposes an I.2.2 HF block-context model from the quantized IR.
///
/// Returns [`HfBlockContextPlan::Default`] when no candidate is expected to
/// beat the default map — the caller then keeps the existing path and stays
/// byte-identical to pre-18c. A non-default proposal is always a *candidate*:
/// the caller re-censuses, retrains, and keeps it only on a strict exact-price
/// win.
///
/// # What is being searched
///
/// Two levers, tried in order:
///
/// 1. **Shape-class trim (no thresholds).** The default map has 15 contexts
///    covering every Order ID; a frame that only uses a few transform shapes
///    never fires the rest. Collapsing unused shape classes shrinks
///    `nb_block_ctx`, which shrinks I.4's pre-context count (`495 × nb`) and
///    the HF context map the entropy trainer has to signal. This is the
///    cheapest density win and needs no QF/LF thresholds.
/// 2. **QF split.** When `HfMul` varies, one QF threshold expands `bsize` and
///    the Y row of the finer band gets its own half of a 16-way map.
///
/// LF thresholds are deferred: alone they rarely amortise the map, and with
/// QF they multiply `bsize` past the break-even point of a single-pass encoder.
pub(crate) fn propose_block_context(
    spatial: &SpatialPlan,
    _quantized: &QuantizedFrameIr,
) -> HfBlockContextPlan {
    let used_shapes = used_shape_classes(spatial);
    let qf_thresholds = propose_qf_thresholds(spatial);

    if qf_thresholds.is_empty() {
        return propose_trimmed_map(&used_shapes);
    }

    // QF path: one threshold, Y fine-band split under the 16-context ceiling.
    // Unused shape classes alias to 0 so density holds after densify.
    let n_qf = qf_thresholds.len() + 1;
    let bsize = 39 * n_qf;
    if bsize > 39 * 64 {
        return HfBlockContextPlan::Default;
    }
    let mut map: Vec<u8> = Vec::with_capacity(bsize);
    for base in 0..39usize {
        let shape = base % 13;
        let base_label = DEFAULT_BLOCK_CTX_MAP.get(base).copied().unwrap_or(0);
        let coarse = base_label / 2; // 0..14 → 0..7
        for qf_band in 0..n_qf {
            let label = if !used_shapes.contains(&shape) {
                0
            } else if base < 13 && qf_band > 0 {
                // Y on the fine QF band.
                8 + coarse
            } else {
                coarse
            };
            map.push(label);
        }
    }
    densify_inplace(&mut map);
    HfBlockContextPlan::Custom {
        lf_thresholds: [Vec::new(), Vec::new(), Vec::new()],
        qf_thresholds,
        map,
    }
}

/// Shape classes (`order_id` in `0..13`) the frame's varblocks actually use.
fn used_shape_classes(spatial: &SpatialPlan) -> std::collections::BTreeSet<usize> {
    spatial
        .lf_groups
        .iter()
        .flat_map(|g| g.blocks.iter().map(|b| b.transform.order_id()))
        .collect()
}

/// Custom map with empty thresholds: keep the default labels of every used
/// shape class and collapse the rest onto context 0, then densify. When the
/// frame only exercises a few Order IDs this drops `nb_block_ctx` well below
/// 15; when every class is used the densified map matches the default and we
/// return [`HfBlockContextPlan::Default`] so the wire stays one bit.
fn propose_trimmed_map(used_shapes: &std::collections::BTreeSet<usize>) -> HfBlockContextPlan {
    if used_shapes.is_empty() || used_shapes.len() >= 13 {
        return HfBlockContextPlan::Default;
    }
    let mut map = DEFAULT_BLOCK_CTX_MAP.to_vec();
    for (base, slot) in map.iter_mut().enumerate() {
        let shape = base % 13;
        if !used_shapes.contains(&shape) {
            *slot = 0;
        }
    }
    densify_inplace(&mut map);
    let nb = map.iter().copied().max().map_or(0, |m| u64::from(m) + 1);
    // No win if we did not actually shrink the context count.
    if nb >= 15 || map.as_slice() == DEFAULT_BLOCK_CTX_MAP.as_slice() {
        return HfBlockContextPlan::Default;
    }
    HfBlockContextPlan::Custom {
        lf_thresholds: [Vec::new(), Vec::new(), Vec::new()],
        qf_thresholds: Vec::new(),
        map,
    }
}

/// Remap labels onto a dense `0..k` in place.
fn densify_inplace(map: &mut [u8]) {
    let mut used: Vec<u8> = map.to_vec();
    used.sort_unstable();
    used.dedup();
    for slot in map.iter_mut() {
        let dense = used
            .iter()
            .position(|&x| x == *slot)
            .and_then(|i| u8::try_from(i).ok())
            .unwrap_or(0);
        *slot = dense;
    }
}

/// Caps multi-preset proposals: each preset multiplies the pre-context
/// count and the HF histogram bundle. Two is enough to prove the wire
/// path; more is a later density search.
const MAX_HF_PRESETS: u32 = 2;

/// Proposes an I.2.6 multi-preset assignment from per-group coefficient
/// mass fingerprints.
///
/// Returns `None` when a split cannot help (single group, uniform mass).
/// Otherwise `(num_hf_presets, group_presets)` with every group assigned
/// a dense preset id in `0..num_hf_presets`. The caller re-censuses under
/// the assignment (I.4's `offset = 495·nb·hfp` depends on it) and adopts
/// only on an exact price win.
pub(crate) fn propose_presets(
    geometry: &VardctGeometry,
    spatial: &SpatialPlan,
    quantized: &QuantizedFrameIr,
) -> Option<(u32, Vec<PresetId>)> {
    let num_groups = usize::try_from(geometry.num_groups()).unwrap_or(0);
    if num_groups < 2 {
        return None;
    }
    let scores: Vec<u64> = (0..num_groups)
        .map(|g| group_mass_score(geometry, spatial, quantized, g as u64))
        .collect();
    let min = scores.iter().copied().min().unwrap_or(0);
    let max = scores.iter().copied().max().unwrap_or(0);
    // A flat score field means every group looks the same to the fingerprint;
    // splitting only pays the second histogram bank.
    if max == min {
        return None;
    }
    let mid = min.saturating_add(max.saturating_sub(min) / 2);
    let mut assignment = Vec::with_capacity(num_groups);
    let mut saw_lo = false;
    let mut saw_hi = false;
    for &score in &scores {
        if score <= mid {
            assignment.push(PresetId::new(0));
            saw_lo = true;
        } else {
            assignment.push(PresetId::new(1));
            saw_hi = true;
        }
    }
    if !(saw_lo && saw_hi) {
        return None;
    }
    // I.2.6: num_hf_presets ≤ num_groups (field width is ceil(log2(num_groups))).
    let num = MAX_HF_PRESETS.min(u32::try_from(num_groups).unwrap_or(1));
    if num < 2 {
        return None;
    }
    Some((num, assignment))
}

/// A cheap per-group fingerprint: total absolute HF coefficient mass.
///
/// Groups with heavy residual structure want a different histogram bank from
/// near-flat groups; the absolute mass is enough to seed a two-way split
/// without a second full entropy census.
fn group_mass_score(
    geometry: &VardctGeometry,
    spatial: &SpatialPlan,
    quantized: &QuantizedFrameIr,
    group: u64,
) -> u64 {
    let Some(rect) = geometry.group_rect(group) else {
        return 0;
    };
    let Some(lf_id) = geometry.lf_group_of(group) else {
        return 0;
    };
    let Some(lf_rect) = geometry.lf_group_rect(lf_id) else {
        return 0;
    };
    let index = usize::try_from(lf_id.index()).unwrap_or(usize::MAX);
    let Some(spatial_g) = spatial.lf_groups.get(index) else {
        return 0;
    };
    let Some(quant_g) = quantized.lf_groups.get(index) else {
        return 0;
    };
    let origin_bx = (rect.x0 - lf_rect.x0) / 8;
    let origin_by = (rect.y0 - lf_rect.y0) / 8;
    let blocks_w = rect.width.div_ceil(8);
    let blocks_h = rect.height.div_ceil(8);
    let mut mass = 0u64;
    for (i, block) in spatial_g.blocks.iter().enumerate() {
        let (bx, by) = (block.origin.bx(), block.origin.by());
        if bx < origin_bx || by < origin_by {
            continue;
        }
        let (lx, ly) = (bx - origin_bx, by - origin_by);
        if lx >= blocks_w || ly >= blocks_h {
            continue;
        }
        let Some(coeff) = quant_g.coefficients.get(i) else {
            continue;
        };
        for channel in 0..3usize {
            if let Some(values) = coeff.channel(channel) {
                for &v in values {
                    mass = mass.saturating_add(u64::from(v.unsigned_abs()));
                }
            }
        }
    }
    mass
}

/// QF thresholds from the distinct `HfMul` values on the plan. I.4 compares
/// `qf > threshold`, so placing a threshold at each mul (except the largest)
/// puts every distinct mul into its own band.
fn propose_qf_thresholds(spatial: &SpatialPlan) -> Vec<u32> {
    let mut muls: Vec<u32> = spatial
        .lf_groups
        .iter()
        .flat_map(|g| g.blocks.iter().map(|b| b.hf_mul.get()))
        .collect();
    muls.sort_unstable();
    muls.dedup();
    if muls.len() < 2 {
        return Vec::new();
    }
    // Drop the largest: a threshold equal to the max mul leaves an empty top
    // band because nothing is strictly greater.
    muls.pop();
    muls.truncate(MAX_QF_THRESHOLDS);
    muls
}

/// Trains the model from a raw census (§9.2 steps 2–5).
///
/// # Errors
///
/// Only [`HistogramPlan::new`]'s own rejection of an empty alphabet, which a
/// census with at least one counted event cannot produce.
pub(crate) fn train(census: &CensusSink) -> PlanResult<TrainedModel> {
    // Step 1: live pre-contexts, one cluster each.
    let mut clusters: Vec<Cluster> = Vec::new();
    let mut context_cluster: Vec<Option<usize>> = vec![None; census.len()];
    for index in 0..census.len() {
        let Some(histogram) = census.histogram(jpxl_encode::vardct::ids::PreContextId::new(
            u32::try_from(index).unwrap_or(u32::MAX),
        )) else {
            continue;
        };
        let values: Vec<(u32, u64)> = histogram.iter().map(|(v, c)| (v, u64::from(c))).collect();
        if values.is_empty() {
            continue;
        }
        let (cost, config) = best_config(&values);
        if let Some(slot) = context_cluster.get_mut(index) {
            *slot = Some(clusters.len());
        }
        clusters.push(Cluster {
            contexts: vec![index],
            values,
            cost,
            config,
            generation: 0,
        });
    }
    if clusters.is_empty() {
        // A frame with no coded events still needs a legal model.
        return Ok(TrainedModel {
            context_map: vec![ClusterId::new(0); census.len()],
            histograms: vec![HistogramPlan::new(vec![1u32])?],
            hybrid_uint: vec![HybridUintPlan {
                split_exponent: 4,
                msb_in_token: 2,
                lsb_in_token: 0,
            }],
        });
    }

    // Step 2: candidate edges between fingerprint neighbours (§9.3's buckets
    // as a window over one sorted order), on a max-saving queue with
    // generation counters.
    let mut order: Vec<usize> = (0..clusters.len()).collect();
    order.sort_by_key(|&i| clusters.get(i).map_or((0, 0), |c| fingerprint(&c.values)));

    // (saving, a, b, gen_a, gen_b) — BinaryHeap on f64 via sortable bits.
    let mut queue: std::collections::BinaryHeap<(i64, usize, usize, u32, u32)> =
        std::collections::BinaryHeap::new();
    #[allow(
        clippy::cast_possible_truncation,
        reason = "savings are bounded by total census bits, far inside i64 scaled by 256"
    )]
    let push_edge = |queue: &mut std::collections::BinaryHeap<(i64, usize, usize, u32, u32)>,
                     clusters: &[Cluster],
                     a: usize,
                     b: usize| {
        let (Some(ca), Some(cb)) = (clusters.get(a), clusters.get(b)) else {
            return;
        };
        let merged = merge_values(&ca.values, &cb.values);
        let (merged_cost, _) = best_config(&merged);
        let saving = ca.cost + cb.cost + CLUSTER_OVERHEAD_BITS - merged_cost;
        if saving > 0.0 {
            queue.push(((saving * 256.0) as i64, a, b, ca.generation, cb.generation));
        }
    };
    for (position, &first) in order.iter().enumerate() {
        for &other in order.iter().skip(position + 1).take(CANDIDATE_WINDOW) {
            push_edge(&mut queue, &clusters, first, other);
        }
    }

    // Step 3: merge while a merge saves bits. Each accepted merge empties one
    // cluster, so the loop runs at most `n - 1` times; stale entries are
    // dropped by their generation stamps.
    let mut live = clusters.len();
    while let Some((_, a, b, gen_a, gen_b)) = queue.pop() {
        let (Some(ca), Some(cb)) = (clusters.get(a), clusters.get(b)) else {
            continue;
        };
        if ca.generation != gen_a || cb.generation != gen_b || ca.contexts.is_empty() {
            continue;
        }
        let merged = merge_values(&ca.values, &cb.values);
        let (merged_cost, merged_config) = best_config(&merged);
        if ca.cost + cb.cost + CLUSTER_OVERHEAD_BITS - merged_cost <= 0.0 {
            continue;
        }
        // Merge b into a.
        let mut moved = clusters
            .get(b)
            .map(|c| c.contexts.clone())
            .unwrap_or_default();
        for &ctx in &moved {
            if let Some(slot) = context_cluster.get_mut(ctx) {
                *slot = Some(a);
            }
        }
        if let Some(cb) = clusters.get_mut(b) {
            cb.contexts.clear();
            cb.values.clear();
            cb.generation += 1;
        }
        if let Some(ca) = clusters.get_mut(a) {
            ca.contexts.append(&mut moved);
            ca.contexts.sort_unstable();
            ca.values = merged;
            ca.cost = merged_cost;
            ca.config = merged_config;
            ca.generation += 1;
        }
        live -= 1;
        // New candidates for the merged cluster against its window in the
        // original ordering (a linear scan: live cluster counts are small,
        // and there are at most n - 1 merges).
        if let Some(position) = order.iter().position(|&i| i == a) {
            let lo = position.saturating_sub(CANDIDATE_WINDOW / 2);
            for &other in order.iter().skip(lo).take(CANDIDATE_WINDOW) {
                if other != a && clusters.get(other).is_some_and(|c| !c.contexts.is_empty()) {
                    push_edge(&mut queue, &clusters, a, other);
                }
            }
        }
    }

    // Step 4: C.2.2's ceiling. Positive-saving merges are exhausted, so any
    // remaining excess is folded by the least-costly merges regardless of
    // sign (rare: it needs > 255 genuinely different live distributions).
    while live > MAX_CLUSTERS {
        let live_indices: Vec<usize> = (0..clusters.len())
            .filter(|&i| clusters.get(i).is_some_and(|c| !c.contexts.is_empty()))
            .collect();
        let mut best: Option<(f64, usize, usize)> = None;
        for pair in live_indices.windows(2) {
            let (&a, &b) = (pair.first().unwrap_or(&0), pair.get(1).unwrap_or(&0));
            let (Some(ca), Some(cb)) = (clusters.get(a), clusters.get(b)) else {
                continue;
            };
            let merged = merge_values(&ca.values, &cb.values);
            let (merged_cost, _) = best_config(&merged);
            let loss = merged_cost - ca.cost - cb.cost;
            if best.is_none_or(|(l, _, _)| loss < l) {
                best = Some((loss, a, b));
            }
        }
        let Some((_, a, b)) = best else { break };
        let merged = clusters
            .get(a)
            .zip(clusters.get(b))
            .map(|(ca, cb)| merge_values(&ca.values, &cb.values))
            .unwrap_or_default();
        let (merged_cost, merged_config) = best_config(&merged);
        let mut moved = clusters
            .get(b)
            .map(|c| c.contexts.clone())
            .unwrap_or_default();
        for &ctx in &moved {
            if let Some(slot) = context_cluster.get_mut(ctx) {
                *slot = Some(a);
            }
        }
        if let Some(cb) = clusters.get_mut(b) {
            cb.contexts.clear();
            cb.values.clear();
        }
        if let Some(ca) = clusters.get_mut(a) {
            ca.contexts.append(&mut moved);
            ca.contexts.sort_unstable();
            ca.values = merged;
            ca.cost = merged_cost;
            ca.config = merged_config;
        }
        live -= 1;
    }

    // Step 5: the model. Clusters are numbered in first-context order so the
    // map starts at 0 and stays runs-friendly; dead pre-contexts inherit the
    // previous entry's cluster, which costs nothing beyond the run.
    let mut number_of: Vec<Option<u8>> = vec![None; clusters.len()];
    let mut histograms = Vec::new();
    let mut hybrid_uint = Vec::new();
    let mut context_map = Vec::with_capacity(census.len());
    let mut previous = ClusterId::new(0);
    for slot in &context_cluster {
        let id = match slot {
            None => previous,
            Some(cluster_index) => {
                let number = match number_of.get(*cluster_index).copied().flatten() {
                    Some(number) => number,
                    None => {
                        let number = u8::try_from(histograms.len()).unwrap_or(u8::MAX);
                        if let Some(slot) = number_of.get_mut(*cluster_index) {
                            *slot = Some(number);
                        }
                        let cluster = clusters.get(*cluster_index);
                        let (values, config) = cluster
                            .map(|c| (c.values.clone(), c.config))
                            .unwrap_or((vec![(0, 1)], (4, 2, 0)));
                        histograms.push(token_histogram(&values, config)?);
                        #[allow(
                            clippy::cast_possible_truncation,
                            reason = "C.2.3 bounds every field below 16"
                        )]
                        hybrid_uint.push(HybridUintPlan {
                            split_exponent: config.0 as u8,
                            msb_in_token: config.1 as u8,
                            lsb_in_token: config.2 as u8,
                        });
                        number
                    }
                };
                ClusterId::new(number)
            }
        };
        context_map.push(id);
        previous = id;
    }
    if histograms.is_empty() {
        histograms.push(HistogramPlan::new(vec![1u32])?);
        hybrid_uint.push(HybridUintPlan {
            split_exponent: 4,
            msb_in_token: 2,
            lsb_in_token: 0,
        });
    }

    Ok(TrainedModel {
        context_map,
        histograms,
        hybrid_uint,
    })
}

/// Token counts of `values` under `config`, as a dense histogram.
fn token_histogram(values: &[(u32, u64)], config: (u32, u32, u32)) -> PlanResult<HistogramPlan> {
    let mut counts: Vec<u32> = vec![0];
    if let Ok(config) = HybridUintConfig::new(config.0, config.1, config.2) {
        for &(value, count) in values {
            let Ok(split) = config.tokenize(value) else {
                continue;
            };
            let index = usize::try_from(split.token).unwrap_or(usize::MAX);
            if index >= counts.len() {
                counts.resize(index + 1, 0);
            }
            if let Some(slot) = counts.get_mut(index) {
                *slot = slot.saturating_add(u32::try_from(count).unwrap_or(u32::MAX));
            }
        }
    }
    if counts.iter().all(|&c| c == 0) {
        counts = vec![1];
    }
    HistogramPlan::new(counts)
}

/// The fewest coded (non-LLF) coefficient positions an `(Order ID, channel)`
/// pair must have seen before a custom order is proposed for it: below this,
/// the F.3.2 stream costs more than any resequencing can save.
const MIN_ORDER_SAMPLES: u64 = 512;

/// §9.4's candidate coefficient orders, from the quantized IR's own
/// per-position nonzero frequencies.
///
/// For every `(Order ID, channel)` the frame uses, the non-LLF tail of the
/// natural order is stably re-sorted by descending nonzero frequency: runs of
/// trailing zeros are what I.4's `non_zeros` countdown ends early on, so
/// front-loading the positions that actually carry mass shortens every
/// varblock's coded suffix. The LLF prefix (`skip = size / 64`) is never
/// touched — F.3.2 cannot express it and I.4 never codes it. §9.4 warns that
/// frequency sorting is a *candidate generator*, not the answer: the caller
/// re-censuses under the candidate and adopts it only on an exact price win.
pub(crate) fn candidate_orders(
    spatial: &SpatialPlan,
    quantized: &QuantizedFrameIr,
) -> PlanResult<OrderSet> {
    // counts[order_id][channel][position] over non-LLF positions.
    let mut counts: Vec<[Vec<u64>; 3]> = (0..NUM_ORDER_IDS)
        .map(|order_id| {
            let cells = order_id_dims(order_id).map_or(0, |(w, h)| w * h);
            core::array::from_fn(|_| vec![0u64; cells])
        })
        .collect();
    let mut totals: Vec<u64> = vec![0; NUM_ORDER_IDS];

    for (group, coefficients) in spatial.lf_groups.iter().zip(quantized.lf_groups.iter()) {
        for (vb, coeff) in group.blocks.iter().zip(coefficients.coefficients.iter()) {
            let order_id = vb.transform.order_id();
            let Some(natural) = order_id_dims(order_id).map(|(w, h)| natural_coeff_order(w, h))
            else {
                continue;
            };
            let skip = natural.len() / 64;
            if let Some(total) = totals.get_mut(order_id) {
                *total += u64::try_from(natural.len() - skip).unwrap_or(0);
            }
            for channel in 0..3usize {
                let Some(values) = coeff.channel(channel) else {
                    continue;
                };
                let Some(slots) = counts.get_mut(order_id).and_then(|c| c.get_mut(channel)) else {
                    continue;
                };
                for (position, &cell) in natural.iter().enumerate().skip(skip) {
                    let nonzero = values
                        .get(usize::try_from(cell).unwrap_or(usize::MAX))
                        .copied()
                        .unwrap_or(0)
                        != 0;
                    if nonzero && let Some(slot) = slots.get_mut(position) {
                        *slot += 1;
                    }
                }
            }
        }
    }

    let mut orders = OrderSet::natural();
    for order_id in 0..NUM_ORDER_IDS {
        if totals.get(order_id).copied().unwrap_or(0) < MIN_ORDER_SAMPLES {
            continue;
        }
        let Some(natural) = order_id_dims(order_id).map(|(w, h)| natural_coeff_order(w, h)) else {
            continue;
        };
        let skip = natural.len() / 64;
        for channel in 0..3u8 {
            let Some(slots) = counts
                .get(order_id)
                .and_then(|c| c.get(usize::from(channel)))
            else {
                continue;
            };
            // Stable sort keeps the natural sequence among equal frequencies,
            // so an all-equal channel produces the identity and is skipped.
            let mut positions: Vec<usize> = (skip..natural.len()).collect();
            positions.sort_by_key(|&p| core::cmp::Reverse(slots.get(p).copied().unwrap_or(0)));
            if positions.iter().enumerate().all(|(i, &p)| p == skip + i) {
                continue;
            }
            let mut table: Vec<u32> = natural.get(..skip).unwrap_or_default().to_vec();
            table.extend(positions.iter().filter_map(|&p| natural.get(p).copied()));
            orders = orders.with_order(
                OrderId::new(u8::try_from(order_id).unwrap_or(0)),
                channel,
                table,
            )?;
        }
    }
    Ok(orders)
}

#[cfg(test)]
#[allow(
    clippy::indexing_slicing,
    reason = "test-only assertions over models this module just built"
)]
mod tests {
    use super::*;
    use jpxl_encode::vardct::ids::PreContextId;

    fn census_with(populations: &[&[(u32, u32)]]) -> CensusSink {
        let mut census = CensusSink::new(populations.len());
        for (context, values) in populations.iter().enumerate() {
            for &(value, count) in *values {
                for _ in 0..count {
                    census_add(&mut census, context, value);
                }
            }
        }
        census
    }

    fn census_add(census: &mut CensusSink, context: usize, value: u32) {
        use jpxl_encode::vardct::HfEventSink;
        census.coefficient(
            PreContextId::new(u32::try_from(context).unwrap_or(u32::MAX)),
            value,
        );
    }

    #[test]
    fn identical_contexts_merge_and_different_ones_stay_apart() {
        let zeros: &[(u32, u32)] = &[(0, 900), (1, 40), (2, 10)];
        let heavy: &[(u32, u32)] = &[(30, 200), (60, 200), (120, 200)];
        let model = train(&census_with(&[zeros, zeros, heavy, zeros, heavy])).expect("trains");
        // The three zero-heavy contexts share one cluster, the two heavy ones
        // another.
        assert_eq!(model.histograms.len(), 2);
        assert_eq!(model.context_map[0], model.context_map[1]);
        assert_eq!(model.context_map[0], model.context_map[3]);
        assert_eq!(model.context_map[2], model.context_map[4]);
        assert_ne!(model.context_map[0], model.context_map[2]);
    }

    #[test]
    fn the_chosen_config_never_models_worse_than_the_legacy_one() {
        // The invariant the candidate list guarantees by construction, stated
        // as a test so a candidate-list edit cannot silently drop the legacy
        // configuration: whatever the population, the trainer's choice prices
        // at or below slice 12's frame-wide (4, 2, 0).
        for values in [
            vec![(0u32, 1000u64), (1, 20)],
            vec![(200, 300), (900, 300), (3000, 300)],
            vec![(0, 10), (1, 10), (2, 10), (3, 10), (31, 10)],
            vec![(65_000, 5)],
        ] {
            let (best_cost, config) = best_config(&values);
            let legacy = data_cost(&values, (4, 2, 0)).expect("legacy prices");
            assert!(
                best_cost <= legacy,
                "config {config:?} at {best_cost:.1} bits must not lose to \
                 the legacy (4, 2, 0) at {legacy:.1} bits for {values:?}"
            );
        }
    }

    #[test]
    fn dead_contexts_inherit_their_neighbour_and_the_model_stays_legal() {
        let mut census = CensusSink::new(6);
        census_add(&mut census, 1, 0);
        census_add(&mut census, 1, 0);
        census_add(&mut census, 4, 7);
        let model = train(&census).expect("trains");
        assert_eq!(model.context_map.len(), 6);
        assert_eq!(model.histograms.len(), model.hybrid_uint.len());
        for id in &model.context_map {
            assert!(usize::from(id.get()) < model.histograms.len());
        }
    }

    #[test]
    fn densify_inplace_is_dense() {
        let mut map = vec![4u8, 0, 4, 9, 0, 9];
        densify_inplace(&mut map);
        assert_eq!(map, vec![1, 0, 1, 2, 0, 2]);
        let max = map.iter().copied().max().unwrap_or(0);
        assert!(max < 16, "densified labels must stay under I.2.2's ceiling");
    }

    #[test]
    fn the_merge_loop_is_bounded_and_respects_the_cluster_ceiling() {
        // 300 deliberately incompatible distributions: values far apart, so
        // positive-saving merges are rare and the ceiling logic runs.
        let populations: Vec<Vec<(u32, u32)>> = (0..300u32)
            .map(|i| vec![(i * 97, 50), (i * 97 + 13, 25)])
            .collect();
        let refs: Vec<&[(u32, u32)]> = populations.iter().map(Vec::as_slice).collect();
        let model = train(&census_with(&refs)).expect("trains");
        assert!(
            model.histograms.len() <= MAX_CLUSTERS,
            "C.2.2 ceiling: {} clusters",
            model.histograms.len()
        );
    }
}
