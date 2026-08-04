//! The trained entropy model: `Encoder-plan1.md` §9.2 steps 2–5 and §9.3's
//! clusterer, over the raw census the walk already produces.
//!
//! # What is being chosen
//!
//! The wire's entropy model is a context map (pre-context → cluster), one
//! hybrid-uint configuration per cluster, and one distribution per cluster.
//! The writer rebuilds the ANS tables from its own exact token census, so the
//! *levers* are the map and the configurations — this module chooses both
//! from the raw value census and nothing else.
//!
//! # Why one pass is exact
//!
//! The event stream — which contexts fire, with which raw values — depends on
//! the block context model and the coefficient orders, and on nothing this
//! module chooses. With the default block context and natural orders (both
//! fixed until their writers exist), the census taken before training is
//! byte-for-byte the census the writer will take after it. §9's "bounded
//! refinement pass" therefore has its bound at **zero iterations** at this
//! scope: re-running the census after training would count the same events.
//! The clusterer's own termination is separately bounded: every merge reduces
//! the cluster count by one, so at most `n - 1` merges happen, and the
//! stale-edge recomputations are bounded by the candidate edge count.
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

use jpxl_encode::vardct::ids::ClusterId;
use jpxl_encode::vardct::plan::{HistogramPlan, HybridUintPlan};
use jpxl_encode::vardct::{CensusSink, PlanResult};
use jpxl_entropy::HybridUintConfig;

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
