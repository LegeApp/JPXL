//! The lossless-modular track's plan type.
//!
//! Slice 19: the plan names **group size**, **predictor**, and **MA tree
//! shape** (learned under depth/leaf caps). Residual ANS tables (and optional
//! LZ77) are built at emission from a census.
//!
//! ```text
//! plan_for(...)  ──▶  LosslessPlan  ──validate──▶  ValidatedLosslessPlan
//!  (policy: predictor + greedy tree search)             │
//!                                                       ▼
//!                                         encode_codestream_with_plan
//! ```

use crate::EncodeOptions;
use crate::error::{EncodeError, Result};
use crate::frame::{DEFAULT_GROUP_SIZE_SHIFT, MAX_GROUP_SIZE_SHIFT};
use crate::modular::{
    self, MaTree, PaletteForward, Plane, Predictor, Rect, squeeze, try_exact_palette,
};

/// Every choice the lossless modular path makes before emission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LosslessPlan {
    /// F.2's `group_size_shift`.
    pub group_size_shift: u32,
    /// Whether the planes carry H.6.3's `kRCT` (YCoCg) transform.
    ///
    /// Mutually exclusive with [`palette`](Self::palette) in wave 1.
    pub rct: bool,
    /// Exact-colour palette when cheaper than RCT/direct (wave 1).
    pub palette: Option<PaletteForward>,
    /// Default squeeze (`num_sq = 0` on wire) after optional RCT.
    ///
    /// Mutually exclusive with palette in wave 1.
    pub squeeze: bool,
    /// MA tree (predictors + optional property splits).
    pub tree: MaTree,
}

/// A [`LosslessPlan`] that has passed validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedLosslessPlan(LosslessPlan);

impl ValidatedLosslessPlan {
    /// The plan.
    #[must_use]
    pub const fn plan(&self) -> &LosslessPlan {
        &self.0
    }
}

/// Checks a lossless plan.
///
/// # Errors
///
/// [`EncodeError::ValueOutOfRange`] if `group_size_shift` is above
/// [`MAX_GROUP_SIZE_SHIFT`].
pub fn validate(plan: LosslessPlan) -> Result<ValidatedLosslessPlan> {
    if plan.group_size_shift > MAX_GROUP_SIZE_SHIFT {
        return Err(EncodeError::ValueOutOfRange {
            what: "group_size_shift",
            value: i64::from(plan.group_size_shift),
        });
    }
    Ok(ValidatedLosslessPlan(plan))
}

/// Table H.3 predictors this track will consider (slice 19; Phase 4A added
/// Weighted).
///
/// Weighted (6) is tried in the single-leaf sweep and per-leaf refinement
/// (both loop over this list already) but deliberately NOT reachable from
/// inside the greedy split search: `split_leaf` always carries the parent
/// leaf's already-chosen predictor forward unchanged ("same predictor both
/// sides first" below), so a Weighted-winning leaf from the sweep propagates
/// into splits for free without the split loop needing its own Weighted
/// branch. Every Weighted-containing candidate is priced by a full
/// sequential scan (`modular::collect_plane_residuals_weighted`), not
/// Phase 4B's sampled scorer -- see `estimate_residual_bits_sampled`'s
/// Weighted fallback.
const PREDICTOR_CANDIDATES: &[Predictor] = &[
    Predictor::Zero,
    Predictor::West,
    Predictor::North,
    Predictor::AverageWestNorth,
    Predictor::Select,
    Predictor::Gradient,
    Predictor::Weighted,
];

/// Property indices tried for splits (Table H.4 static rows).
///
/// Deliberately excludes 15 (`max_error`, H.5's output): the encoder does
/// not run the Weighted state machine for trees that never select predictor
/// 6, so property 15 would read as a constant 0 for them, and computing it
/// unconditionally would mean paying Weighted's full-sequential-scan cost on
/// every split trial regardless of whether any leaf uses Weighted. Splitting
/// on `max_error` is a possible later increment, not required for Weighted
/// itself: predictor selection and property splitting are independent axes
/// (H.4.1), and the decoder's behaviour is unaffected by which trees the
/// encoder's search considers -- see jpegxl-rs.work.arch-phase4a-weighted-
/// predictor-scoped.
const SPLIT_PROPERTIES: &[u32] = &[4, 5, 6, 7, 9, 10, 11];

/// Thresholds tried for `property > value` decisions.
const SPLIT_THRESHOLDS: &[i32] = &[0, 1, 2, 4, 8, 16, 32, 64];

/// Cap on MA tree depth (root = 0) for small frames.
const MAX_TREE_DEPTH: u32 = 4;
/// Cap on residual contexts / leaves for small frames.
const MAX_TREE_LEAVES: usize = 8;
/// Above this sample count, tree search is capped to one split (two leaves).
/// Full multi-split search is O(candidates × sections × residual cost) and
/// dominates multi-group planning time.
const DEEP_SEARCH_SAMPLE_CAP: u64 = 64 * 64;

/// Chooses a plan for already-transformed `planes`.
///
/// Policy (slice 19d + Opt-M tiered scoring):
/// 1. Score each predictor as a single-leaf tree with a **cheap** residual
///    estimate (collect + Shannon hybrid cost, no ANS emit).
/// 2. Greedy recursive splitting with the same cheap ranker.
/// 3. Per-leaf predictor refinement (cheap).
/// 4. **Exact** residual ANS price of the MA winner, then exact palette and
///    squeeze trials against that baseline.
///
/// Frames larger than [`DEEP_SEARCH_SAMPLE_CAP`] samples use at most one
/// binary split (depth 1, two leaves) so multi-group planning stays usable.
///
/// # Errors
///
/// Any error from residual collection, ANS table build, or [`validate`].
pub fn plan_for(
    width: u32,
    height: u32,
    planes: &[Plane],
    rct: bool,
    options: &EncodeOptions,
) -> Result<ValidatedLosslessPlan> {
    reset_multiplicity();
    let group_size_shift = options.group_size_shift.unwrap_or(DEFAULT_GROUP_SIZE_SHIFT);
    let samples = u64::from(width).saturating_mul(u64::from(height));
    let max_leaves = if samples > DEEP_SEARCH_SAMPLE_CAP {
        2
    } else {
        MAX_TREE_LEAVES
    };
    let max_depth = if samples > DEEP_SEARCH_SAMPLE_CAP {
        1
    } else {
        MAX_TREE_DEPTH
    };

    let mut best: Option<(u64, MaTree)> = None;

    // When RCT is desired, score MA trees on RCT-transformed samples.
    let rct_planes_storage: Option<Vec<Plane>> = if rct && planes.len() == 3 {
        let mut p = planes.to_vec();
        modular::apply_rct(&mut p)?;
        Some(p)
    } else {
        None
    };
    let score_planes: &[Plane] = rct_planes_storage.as_deref().unwrap_or(planes);

    // Phase-1: wrap score planes once; every trial Arc-clones only.
    let shared_score: Vec<modular::SharedPlane> = score_planes
        .iter()
        .map(|p| {
            let bytes = (p.len() as u64).saturating_mul(4);
            note_plane_clone_bytes(bytes);
            std::sync::Arc::<[i32]>::from(p.as_slice())
        })
        .collect();

    for &predictor in PREDICTOR_CANDIDATES {
        let tree = MaTree::single_leaf(predictor);
        let cost = total_cost_shared(
            width,
            height,
            &shared_score,
            &tree,
            group_size_shift,
            PriceTier::Cheap,
        )?;
        if best.as_ref().is_none_or(|(c, _)| cost < *c) {
            best = Some((cost, tree));
        }
    }

    let (mut best_cost, mut best_tree) =
        best.unwrap_or_else(|| (u64::MAX, MaTree::single_leaf(Predictor::Gradient)));

    // Greedy splits: repeatedly try to split every current leaf (cheap rank).
    loop {
        if best_tree.num_contexts() >= max_leaves || best_tree.depth() >= max_depth {
            break;
        }
        let mut improved = false;
        let leaf_count = best_tree.num_contexts();
        let leaf_preds = best_tree.leaf_predictors();
        let mut round_best: Option<(u64, MaTree)> = None;

        for ctx in 0..leaf_count {
            let parent_pred = leaf_preds.get(ctx).copied().unwrap_or(Predictor::Gradient);
            for &property in SPLIT_PROPERTIES {
                for &value in SPLIT_THRESHOLDS {
                    // Same predictor both sides first (cheap topology search).
                    let Ok(candidate) =
                        best_tree.split_leaf(ctx, property, value, parent_pred, parent_pred)
                    else {
                        continue;
                    };
                    if candidate.depth() > max_depth || candidate.num_contexts() > max_leaves {
                        continue;
                    }
                    let cost = total_cost_shared(
                        width,
                        height,
                        &shared_score,
                        &candidate,
                        group_size_shift,
                        PriceTier::Cheap,
                    )?;
                    if cost < best_cost && round_best.as_ref().is_none_or(|(c, _)| cost < *c) {
                        round_best = Some((cost, candidate));
                    }
                }
            }
        }

        if let Some((cost, tree)) = round_best {
            best_cost = cost;
            best_tree = tree;
            improved = true;
        }
        if !improved {
            break;
        }
    }

    // Per-leaf predictor refinement on the settled topology (cheap).
    let leaf_count = best_tree.num_contexts();
    for ctx in 0..leaf_count {
        let current = best_tree
            .leaf_predictors()
            .get(ctx)
            .copied()
            .unwrap_or(Predictor::Gradient);
        for &predictor in PREDICTOR_CANDIDATES {
            if predictor == current {
                continue;
            }
            let Ok(candidate) = best_tree.with_leaf_predictor(ctx, predictor) else {
                continue;
            };
            let cost = total_cost_shared(
                width,
                height,
                &shared_score,
                &candidate,
                group_size_shift,
                PriceTier::Cheap,
            )?;
            if cost < best_cost {
                best_cost = cost;
                best_tree = candidate;
            }
        }
    }

    // Stage C: exact residual price of the MA finalist (baseline for transforms).
    let ma_source = modular::ModularSource::direct_shared(
        width,
        height,
        &shared_score,
        false,
        best_tree.clone(),
        false,
    );
    best_cost = total_cost_source(&ma_source, group_size_shift, PriceTier::Exact)?;

    // Nested palette trial on *source* samples (not RCT). Exact-price gate.
    let mut palette: Option<PaletteForward> = None;
    let mut use_rct = rct;
    let mut use_squeeze = false;
    let num_c = planes.len();
    if (num_c == 1 || num_c == 3)
        && let Some(fwd) = try_exact_palette(width, height, planes, 0, num_c)?
    {
        let tree = MaTree::single_leaf(Predictor::Zero);
        let cost = total_cost_source(
            &modular::ModularSource::from_palette(fwd.clone(), tree.clone(), false),
            group_size_shift,
            PriceTier::Exact,
        )?;
        if cost < best_cost {
            best_cost = cost;
            best_tree = tree;
            palette = Some(fwd);
            use_rct = false;
        }
    }

    // Default squeeze on MA-scored planes. Skip when palette won. Exact gate.
    if palette.is_none() && squeeze::default_would_run(width, height, score_planes.len()) {
        let tree = MaTree::single_leaf(Predictor::Gradient);
        let source = modular::ModularSource::with_default_squeeze(
            width,
            height,
            score_planes,
            false, // RCT already in score_planes samples
            tree.clone(),
            false,
        )?;
        let source = if use_rct {
            let mut s = source;
            s.transforms.insert(
                0,
                modular::ModularTransform::Rct {
                    rct_type: modular::RCT_TYPE_YCOCG,
                },
            );
            s
        } else {
            source
        };
        let cost = total_cost_source(&source, group_size_shift, PriceTier::Exact)?;
        if cost < best_cost {
            best_cost = cost;
            best_tree = tree;
            use_squeeze = true;
        }
    }
    let _ = best_cost;

    let no_palette = palette.is_none();
    validate(LosslessPlan {
        group_size_shift,
        rct: use_rct && no_palette,
        palette,
        squeeze: use_squeeze && no_palette,
        tree: best_tree,
    })
}

/// How residual candidates are priced (Opt-M tiered planner).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PriceTier {
    /// Collect residuals + Shannon hybrid estimate (no ANS emit).
    Cheap,
    /// Full residual ANS write path (count-only BitWriter).
    Exact,
}

/// Multiplicity counters for one [`plan_for`] call (Opt-M + Phase-0 diagnostics).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PlanMultiplicity {
    /// Exact residual stream prices (finalists + palette/squeeze trials).
    pub exact_residual_prices: u32,
    /// Cheap ranking scores during tree search.
    pub cheap_scores: u32,
    /// Full residual-field scans (cheap + exact combined).
    pub residual_scans: u32,
    /// Approximate bytes deep-cloned when building modular sources (planes × 4).
    pub plane_clone_bytes: u64,
}

impl PlanMultiplicity {
    /// One-line summary for CLI diagnostics.
    #[must_use]
    pub fn summary_line(self) -> String {
        format!(
            "cheap_scores={} exact_prices={} residual_scans={} plane_clone_bytes={}",
            self.cheap_scores,
            self.exact_residual_prices,
            self.residual_scans,
            self.plane_clone_bytes
        )
    }
}

std::thread_local! {
    static LAST_PLAN_MULTIPLICITY: std::cell::Cell<PlanMultiplicity> =
        const { std::cell::Cell::new(PlanMultiplicity {
            exact_residual_prices: 0,
            cheap_scores: 0,
            residual_scans: 0,
            plane_clone_bytes: 0,
        }) };
}

/// Multiplicity counters from the most recent [`plan_for`] on this thread.
#[must_use]
pub fn last_plan_multiplicity() -> PlanMultiplicity {
    LAST_PLAN_MULTIPLICITY.with(std::cell::Cell::get)
}

fn reset_multiplicity() {
    LAST_PLAN_MULTIPLICITY.with(|c| {
        c.set(PlanMultiplicity::default());
    });
}

fn bump_exact() {
    LAST_PLAN_MULTIPLICITY.with(|c| {
        let mut m = c.get();
        m.exact_residual_prices = m.exact_residual_prices.saturating_add(1);
        m.residual_scans = m.residual_scans.saturating_add(1);
        c.set(m);
    });
}

/// Records plane deep-clone traffic from modular source construction (Phase-0).
pub(crate) fn note_plane_clone_bytes(bytes: u64) {
    LAST_PLAN_MULTIPLICITY.with(|c| {
        let mut m = c.get();
        m.plane_clone_bytes = m.plane_clone_bytes.saturating_add(bytes);
        c.set(m);
    });
}

fn bump_cheap() {
    LAST_PLAN_MULTIPLICITY.with(|c| {
        let mut m = c.get();
        m.cheap_scores = m.cheap_scores.saturating_add(1);
        m.residual_scans = m.residual_scans.saturating_add(1);
        c.set(m);
    });
}

/// Residual-section bits for shared plane Arcs (Phase-1 plan trials).
fn total_cost_shared(
    width: u32,
    height: u32,
    planes: &[modular::SharedPlane],
    tree: &MaTree,
    group_size_shift: u32,
    tier: PriceTier,
) -> Result<u64> {
    let source =
        modular::ModularSource::direct_shared(width, height, planes, false, tree.clone(), false);
    total_cost_source(&source, group_size_shift, tier)
}

fn total_cost_source(
    source: &modular::ModularSource,
    group_size_shift: u32,
    tier: PriceTier,
) -> Result<u64> {
    let residual = residual_stream_cost_source(source, group_size_shift, tier)?;
    let tree_bits = modular::ma_tree_bit_cost(&source.tree)?;
    let geometry = crate::frame::Geometry::new(source.width, source.height, group_size_shift)?;
    let sections = if geometry.is_single_section() {
        1u64
    } else {
        geometry.num_groups()
    };
    Ok(residual.saturating_add(tree_bits.saturating_mul(sections)))
}

/// Residual-section bit cost under the chosen tier.
fn residual_stream_cost_source(
    source: &modular::ModularSource,
    group_size_shift: u32,
    tier: PriceTier,
) -> Result<u64> {
    match tier {
        PriceTier::Cheap => {
            bump_cheap();
            #[cfg(feature = "phase4b-sampled-gather")]
            {
                sampled_cheap_residual_bits(source)
            }
            #[cfg(not(feature = "phase4b-sampled-gather"))]
            {
                cheap_residual_bits(source, group_size_shift)
            }
        }
        PriceTier::Exact => {
            bump_exact();
            exact_residual_bits(source, group_size_shift)
        }
    }
}

/// Row stride for [`sampled_cheap_residual_bits`] (Phase 4B). Every 4th row
/// (plus the last), so the topology search's cost scales roughly a quarter
/// as fast with pixel count as the full-plane scan it stands in for.
#[cfg(any(feature = "phase4b-sampled-gather", test))]
const SAMPLED_GATHER_ROW_STRIDE: u32 = 4;

/// Phase 4B additive scorer: rows a strided subset instead of a full plane
/// scan. Always compiled (regardless of the feature) so decision-preservation
/// tests can compare it against [`cheap_residual_bits`] directly; only
/// [`residual_stream_cost_source`]'s dispatch is feature-gated, so a default
/// build's search path never calls this function.
///
/// See jpegxl-rs.work.arch-phase4b-sampled-gather-scoped.
#[cfg(any(feature = "phase4b-sampled-gather", test))]
fn sampled_cheap_residual_bits(source: &modular::ModularSource) -> Result<u64> {
    modular::estimate_residual_bits_sampled(source, SAMPLED_GATHER_ROW_STRIDE)
}

/// Stage-B estimate: one residual collect + hybrid Shannon cost (no ANS emit).
///
/// Always compiled when the Phase 4B feature is off (the default search
/// path); also compiled under `test` (feature on or off) so decision-
/// preservation tests can compare it against
/// [`sampled_cheap_residual_bits`] directly.
#[cfg(any(not(feature = "phase4b-sampled-gather"), test))]
fn cheap_residual_bits(source: &modular::ModularSource, group_size_shift: u32) -> Result<u64> {
    let geometry = crate::frame::Geometry::new(source.width, source.height, group_size_shift)?;
    if geometry.is_single_section() {
        return modular::estimate_residual_bits_full(source);
    }
    // Multi-section: estimate full-frame residuals once (ranking quality), not
    // per-group exact streams. Finalist Exact re-prices with the real layout.
    modular::estimate_residual_bits_full(source)
}

/// Stage-C exact residual ANS cost (count-only writer; no payload retain).
fn exact_residual_bits(source: &modular::ModularSource, group_size_shift: u32) -> Result<u64> {
    use jpxl_bitstream::BitWriter;

    let geometry = crate::frame::Geometry::new(source.width, source.height, group_size_shift)?;
    let mut bits = 0u64;
    if geometry.is_single_section() {
        let mut w = BitWriter::counting();
        modular::write_residual_payload_full(&mut w, source)?;
        bits = bits.saturating_add(w.bit_len());
    } else {
        let part = modular::partition_channels(source, geometry.group_dim());
        if !part.lf_global.is_empty() {
            let mut w = BitWriter::counting();
            modular::write_residual_payload_indices(&mut w, source, &part.lf_global, None)?;
            bits = bits.saturating_add(w.bit_len());
        }
        for index in 0..geometry.num_lf_groups() {
            let (x0, y0, gw, gh) = geometry.lf_group_rect(index).ok_or_else(|| {
                EncodeError::unsupported("an LF group index past the grid", "G.2")
            })?;
            if part.lf_group.is_empty() {
                continue;
            }
            let mut w = BitWriter::counting();
            modular::write_residual_payload_indices(
                &mut w,
                source,
                &part.lf_group,
                Some(Rect {
                    x0,
                    y0,
                    width: gw,
                    height: gh,
                }),
            )?;
            bits = bits.saturating_add(w.bit_len());
        }
        for index in 0..geometry.num_groups() {
            let (x0, y0, gw, gh) = geometry
                .group_rect(index)
                .ok_or_else(|| EncodeError::unsupported("a group index past the grid", "G.4"))?;
            if part.pass_group.is_empty() {
                continue;
            }
            let mut w = BitWriter::counting();
            modular::write_residual_payload_indices(
                &mut w,
                source,
                &part.pass_group,
                Some(Rect {
                    x0,
                    y0,
                    width: gw,
                    height: gh,
                }),
            )?;
            bits = bits.saturating_add(w.bit_len());
        }
    }
    Ok(bits)
}

#[cfg(test)]
#[allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]
mod tests {
    use super::*;
    use crate::EncodeOptions;

    /// Phase 4B decision-preservation regret harness (echoes S8 Phase C's
    /// independent-second-walk pattern from `regret.rs`): for a handful of
    /// representative planes, compares the predictor the sampled scorer
    /// ([`sampled_cheap_residual_bits`]) would pick against the one the
    /// full-scan baseline ([`cheap_residual_bits`]) picks, at Phase 4B's
    /// production row stride. When they disagree, the sampled winner's
    /// EXACT price (the real safety net -- see [`PriceTier::Exact`]) must
    /// not exceed the baseline winner's exact price by more than a small,
    /// explicit tolerance. Runs regardless of the `phase4b-sampled-gather`
    /// feature (both scorers are always compiled under `test`, see their
    /// `#[cfg]`s) so this regresses even when the search path isn't wired.
    #[test]
    fn sampled_gather_regret_stays_within_tolerance_of_full_scan() {
        const REGRET_TOLERANCE: f64 = 0.02; // 2% of the baseline's exact price.

        let width = 96u32;
        let height = 96u32;
        let planes: Vec<Plane> = vec![
            // Smooth ramp: predictors should agree trivially.
            (0..height)
                .flat_map(|y| (0..width).map(move |x| (x + y) as i32))
                .collect(),
            // Higher-frequency texture: closer to where sampling could miss
            // structure a full scan would catch.
            (0..height)
                .flat_map(|y| (0..width).map(move |x| ((x * 7 + y * 13) % 251) as i32))
                .collect(),
            // Sparse edges on a flat field: a case where most sampled rows
            // are uninformative and the edge rows matter disproportionately.
            (0..height)
                .flat_map(|y| (0..width).map(move |x| if (x + y) % 17 == 0 { 200 } else { 20 }))
                .collect(),
        ];

        for plane in planes {
            let shared: modular::SharedPlane = std::sync::Arc::from(plane.as_slice());

            let mut baseline_best: Option<(u64, Predictor)> = None;
            let mut sampled_best: Option<(u64, Predictor)> = None;
            for &predictor in PREDICTOR_CANDIDATES {
                let tree = MaTree::single_leaf(predictor);
                let source = modular::ModularSource::direct_shared(
                    width,
                    height,
                    std::slice::from_ref(&shared),
                    false,
                    tree,
                    false,
                );
                let full_cost =
                    cheap_residual_bits(&source, DEFAULT_GROUP_SIZE_SHIFT).expect("full cost");
                let sampled_cost = sampled_cheap_residual_bits(&source).expect("sampled cost");
                if baseline_best.as_ref().is_none_or(|(c, _)| full_cost < *c) {
                    baseline_best = Some((full_cost, predictor));
                }
                if sampled_best.as_ref().is_none_or(|(c, _)| sampled_cost < *c) {
                    sampled_best = Some((sampled_cost, predictor));
                }
            }
            let (_, baseline_predictor) = baseline_best.expect("candidates non-empty");
            let (_, sampled_predictor) = sampled_best.expect("candidates non-empty");

            let exact_price_of = |predictor: Predictor| -> u64 {
                let tree = MaTree::single_leaf(predictor);
                let source = modular::ModularSource::direct_shared(
                    width,
                    height,
                    std::slice::from_ref(&shared),
                    false,
                    tree,
                    false,
                );
                total_cost_source(&source, DEFAULT_GROUP_SIZE_SHIFT, PriceTier::Exact)
                    .expect("exact price")
            };

            let baseline_exact = exact_price_of(baseline_predictor);
            let sampled_exact = exact_price_of(sampled_predictor);
            if sampled_predictor != baseline_predictor {
                let regret = sampled_exact.saturating_sub(baseline_exact) as f64;
                let bound = baseline_exact as f64 * REGRET_TOLERANCE;
                assert!(
                    regret <= bound,
                    "sampled scorer picked {sampled_predictor:?} (exact {sampled_exact}) over \
                     baseline's {baseline_predictor:?} (exact {baseline_exact}); regret {regret} \
                     exceeds {REGRET_TOLERANCE:.0}% tolerance ({bound})"
                );
            }
        }
    }

    /// Edge case the "always include the last row" rule in
    /// [`modular::estimate_residual_bits_sampled`]'s row selection exists
    /// for: a frame shorter than the sample stride must not panic or starve
    /// the estimate down to zero rows.
    #[test]
    fn sampled_gather_handles_frames_shorter_than_the_row_stride() {
        let width = 5u32;
        let height = 3u32; // < SAMPLED_GATHER_ROW_STRIDE (4).
        let plane: Plane = (0..height)
            .flat_map(|y| (0..width).map(move |x| (x + y * 2) as i32))
            .collect();
        let shared: modular::SharedPlane = std::sync::Arc::from(plane.as_slice());
        let tree = MaTree::single_leaf(Predictor::Gradient);
        let source = modular::ModularSource::direct_shared(
            width,
            height,
            std::slice::from_ref(&shared),
            false,
            tree,
            false,
        );
        let cost = sampled_cheap_residual_bits(&source).expect("sampled cost on a tiny frame");
        assert!(cost > 0, "a non-empty frame must not cost zero bits");
    }

    #[test]
    fn plan_for_returns_a_validated_tree() {
        // Smooth ramp: predictor search + optional splits must not panic.
        let width = 32u32;
        let height = 32u32;
        let plane: Plane = (0..height)
            .flat_map(|y| (0..width).map(move |x| (x + y) as i32))
            .collect();
        let plan =
            plan_for(width, height, &[plane], false, &EncodeOptions::default()).expect("plan");
        assert!(plan.plan().tree.num_contexts() >= 1);
        assert!(plan.plan().tree.num_contexts() <= MAX_TREE_LEAVES);
        assert!(plan.plan().tree.depth() <= MAX_TREE_DEPTH);
    }

    /// Opt-M: exact residual prices only for finalists (MA winner + optional
    /// palette/squeeze), not for every property×threshold trial.
    #[test]
    fn plan_for_prices_exact_only_on_finalists() {
        let width = 48u32;
        let height = 48u32;
        let plane: Plane = (0..height)
            .flat_map(|y| (0..width).map(move |x| ((x * 3 + y * 5) % 200) as i32))
            .collect();
        let _ = plan_for(width, height, &[plane], false, &EncodeOptions::default()).expect("plan");
        let m = last_plan_multiplicity();
        assert!(
            m.cheap_scores >= 6,
            "predictor search alone should cheap-score every candidate: {m:?}"
        );
        // At most: 1 MA exact + 1 palette + 1 squeeze.
        assert!(
            m.exact_residual_prices <= 3,
            "exact residual prices must stay on finalists only: {m:?}"
        );
        assert!(
            m.exact_residual_prices >= 1,
            "the MA winner must be exact-priced: {m:?}"
        );
        assert!(
            m.cheap_scores > m.exact_residual_prices * 5,
            "cheap ranking should dominate exact finalist prices: {m:?}"
        );
    }

    #[test]
    fn plan_for_adopts_palette_on_scattered_few_colours() {
        // Few unique levels with NO exploitable structure between
        // neighbours -> every predictor loses, including Phase 4A's
        // Weighted, AND the index sequence has no runs an LZ77 pass could
        // exploit either (the search's cost model deliberately excludes
        // LZ77 -- see `ModularSource::from_palette`'s `allow_lz77: false`
        // in `plan_for` vs `true` in the real final emission, a known,
        // pre-existing, Phase-4D-scoped gap; a weaker scatter than a full
        // avalanche hash left enough residual periodicity for LZ77 to make
        // the real emission cheaper than the search predicted, which made
        // an earlier version of this fixture flaky once Weighted became a
        // candidate strong enough to expose the gap). A murmur3-style
        // finalizer avalanches (x, y) well enough that neither prediction
        // nor LZ77 can find structure, so exact-price palette should still
        // win cleanly.
        let width = 48u32;
        let height = 48u32;
        let levels = [10i32, 80, 160, 240];
        let plane: Plane = (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    let mut h = u64::from(x) ^ (u64::from(y) << 32);
                    h ^= h >> 33;
                    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
                    h ^= h >> 33;
                    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
                    h ^= h >> 33;
                    levels[(h % 4) as usize]
                })
            })
            .collect();
        let plan =
            plan_for(width, height, &[plane], false, &EncodeOptions::default()).expect("plan");
        assert!(
            plan.plan().palette.is_some(),
            "scattered few-colour gray should adopt an exact palette"
        );
        assert_eq!(
            plan.plan().palette.as_ref().map(|p| p.params.nb_colours),
            Some(4)
        );
    }

    #[test]
    fn multi_section_palette_roundtrips_through_jpxl_decode() {
        use jpxl_core::limits::Limits;

        // group_dim = 128 with group_size_shift 0 → 200×200 is multi-section.
        let width = 200u32;
        let height = 200u32;
        let levels = [0i32, 80, 160];
        let plane: Plane = (0..height)
            .flat_map(|y| {
                (0..width)
                    .map(move |x| levels[((x.wrapping_mul(5) + y.wrapping_mul(3)) % 3) as usize])
            })
            .collect();
        let fwd = try_exact_palette(width, height, std::slice::from_ref(&plane), 0, 1)
            .expect("ok")
            .expect("palette");
        let image = crate::Image::new(width, height, 8, vec![plane.clone()]).expect("image");
        let forced = validate(LosslessPlan {
            group_size_shift: 0,
            rct: false,
            palette: Some(fwd),
            squeeze: false,
            tree: MaTree::single_leaf(Predictor::Zero),
        })
        .expect("validate");
        let bytes =
            crate::encode_codestream_with_plan(&image, image.planes(), &forced).expect("encode");
        let decoded = jpxl_decode::decode(&bytes, &Limits::default()).expect("decode");
        let got = &decoded.planes.first().expect("plane").samples;
        assert_eq!(got.len(), plane.len());
        for (i, (&a, &b)) in got.iter().zip(plane.iter()).enumerate() {
            assert_eq!(a, b, "sample {i}");
        }
    }

    #[test]
    fn multi_section_squeeze_roundtrips_through_jpxl_decode() {
        use jpxl_core::limits::Limits;

        let width = 300u32;
        let height = 200u32;
        let plane: Plane = (0..height)
            .flat_map(|y| (0..width).map(move |x| ((x + 2 * y) % 180) as i32))
            .collect();
        let image = crate::Image::new(width, height, 8, vec![plane.clone()]).expect("image");
        let forced = validate(LosslessPlan {
            group_size_shift: 0, // group_dim 128
            rct: false,
            palette: None,
            squeeze: true,
            tree: MaTree::single_leaf(Predictor::Gradient),
        })
        .expect("validate");
        let bytes =
            crate::encode_codestream_with_plan(&image, image.planes(), &forced).expect("encode");
        let decoded = jpxl_decode::decode(&bytes, &Limits::default()).expect("decode");
        let got = &decoded.planes.first().expect("plane").samples;
        assert_eq!(got.len(), plane.len());
        for (i, (&a, &b)) in got.iter().zip(plane.iter()).enumerate() {
            assert_eq!(a, b, "sample {i}");
        }
    }

    #[test]
    fn forced_squeeze_roundtrips_through_jpxl_decode() {
        use jpxl_core::limits::Limits;

        // Large enough that default squeeze runs (w,h > 8).
        let width = 32u32;
        let height = 24u32;
        let plane: Plane = (0..height)
            .flat_map(|y| (0..width).map(move |x| ((x + 3 * y) % 200) as i32))
            .collect();
        assert!(squeeze::default_would_run(width, height, 1));
        let image = crate::Image::new(width, height, 8, vec![plane.clone()]).expect("image");
        let forced = validate(LosslessPlan {
            group_size_shift: DEFAULT_GROUP_SIZE_SHIFT,
            rct: false,
            palette: None,
            squeeze: true,
            tree: MaTree::single_leaf(Predictor::Gradient),
        })
        .expect("validate");
        let bytes =
            crate::encode_codestream_with_plan(&image, image.planes(), &forced).expect("encode");
        let decoded = jpxl_decode::decode(&bytes, &Limits::default()).expect("decode");
        let got = &decoded.planes.first().expect("plane").samples;
        assert_eq!(got.len(), plane.len());
        for (i, (&a, &b)) in got.iter().zip(plane.iter()).enumerate() {
            assert_eq!(a, b, "sample {i}");
        }
    }

    #[test]
    fn forced_palette_roundtrips_through_jpxl_decode() {
        use jpxl_core::limits::Limits;

        let width = 32u32;
        let height = 24u32;
        let levels = [0i32, 100, 200];
        let plane: Plane = (0..height)
            .flat_map(|y| (0..width).map(move |x| levels[((x + y) % 3) as usize]))
            .collect();
        let fwd = try_exact_palette(width, height, std::slice::from_ref(&plane), 0, 1)
            .expect("ok")
            .expect("palette");
        let image = crate::Image::new(width, height, 8, vec![plane.clone()]).expect("image");
        let forced = validate(LosslessPlan {
            group_size_shift: DEFAULT_GROUP_SIZE_SHIFT,
            rct: false,
            palette: Some(fwd),
            squeeze: false,
            tree: MaTree::single_leaf(Predictor::Zero),
        })
        .expect("validate");
        let bytes =
            crate::encode_codestream_with_plan(&image, image.planes(), &forced).expect("encode");
        let decoded = jpxl_decode::decode(&bytes, &Limits::default()).expect("decode");
        assert_eq!(decoded.width, width);
        assert_eq!(decoded.height, height);
        let got = &decoded.planes.first().expect("plane").samples;
        assert_eq!(got.len(), plane.len());
        for (i, (&a, &b)) in got.iter().zip(plane.iter()).enumerate() {
            assert_eq!(a, b, "sample {i}");
        }
    }

    /// Phase 4A correctness gate, part 1 (jpegxl-rs.work.arch-phase4a-
    /// weighted-predictor-scoped): a forced single-leaf Weighted tree
    /// round-trips through jpxl-decode. See
    /// `tests/oracle.rs::djxl_decodes_a_forced_weighted_stream_to_the_
    /// source_samples` for the independent-decoder half of this gate (this
    /// crate and jpxl-decode share `jpxl_core::modular_weighted`, so a
    /// round trip against jpxl-decode alone cannot distinguish "correct"
    /// from "the same shared bug on both sides" -- djxl can).
    #[test]
    fn forced_weighted_predictor_roundtrips_through_jpxl_decode() {
        use jpxl_core::limits::Limits;

        let width = 40u32;
        let height = 37u32; // odd height: exercises advance_row on a ragged end.
        let plane: Plane = (0..height)
            .flat_map(|y| (0..width).map(move |x| ((x * 5 + y * 11) % 200) as i32))
            .collect();
        let image = crate::Image::new(width, height, 8, vec![plane.clone()]).expect("image");
        let forced = validate(LosslessPlan {
            group_size_shift: DEFAULT_GROUP_SIZE_SHIFT,
            rct: false,
            palette: None,
            squeeze: false,
            tree: MaTree::single_leaf(Predictor::Weighted),
        })
        .expect("validate");
        let bytes =
            crate::encode_codestream_with_plan(&image, image.planes(), &forced).expect("encode");
        let decoded = jpxl_decode::decode(&bytes, &Limits::default()).expect("decode");
        let got = &decoded.planes.first().expect("plane").samples;
        assert_eq!(got.len(), plane.len());
        for (i, (&a, &b)) in got.iter().zip(plane.iter()).enumerate() {
            assert_eq!(a, b, "sample {i}");
        }
    }

    /// Phase 4A correctness gate, part 2: a forced MIXED tree (Weighted on
    /// one leaf, Gradient on the other) round-trips. This is the case
    /// H.5.1's "invoked for every sample regardless of which leaf it
    /// selects" rule is actually about -- a single-leaf-only test can't
    /// exercise the error state advancing on samples a non-Weighted leaf
    /// produced.
    #[test]
    fn forced_mixed_weighted_tree_roundtrips_through_jpxl_decode() {
        use jpxl_core::limits::Limits;

        let width = 40u32;
        let height = 37u32;
        let plane: Plane = (0..height)
            .flat_map(|y| (0..width).map(move |x| ((x * 5 + y * 11) % 200) as i32))
            .collect();
        let image = crate::Image::new(width, height, 8, vec![plane.clone()]).expect("image");
        let tree = MaTree::binary_split_preds(6, 100, Predictor::Weighted, Predictor::Gradient);
        let forced = validate(LosslessPlan {
            group_size_shift: DEFAULT_GROUP_SIZE_SHIFT,
            rct: false,
            palette: None,
            squeeze: false,
            tree,
        })
        .expect("validate");
        let bytes =
            crate::encode_codestream_with_plan(&image, image.planes(), &forced).expect("encode");
        let decoded = jpxl_decode::decode(&bytes, &Limits::default()).expect("decode");
        let got = &decoded.planes.first().expect("plane").samples;
        assert_eq!(got.len(), plane.len());
        for (i, (&a, &b)) in got.iter().zip(plane.iter()).enumerate() {
            assert_eq!(a, b, "sample {i}");
        }
    }

    /// Phase 4A correctness gate, part 3 -- the one novel architectural
    /// claim this sub-phase rests on: `WeightedState` resets at group
    /// boundaries, not the full channel, matching how every other
    /// predictor's H.3 edge substitution already treats a group rect as a
    /// self-contained scan region (`neighbours`'s `x > 0` / `y > 0` tests
    /// are rect-relative). A 300x200 image at `group_size_shift = 0`
    /// (group_dim 128) is multi-section -- more than one group per channel
    /// -- so this actually exercises `collect_plane_residuals_weighted`
    /// being invoked once per group with a fresh `WeightedState`, not just
    /// once for a whole single-section channel.
    #[test]
    fn multi_section_weighted_roundtrips_through_jpxl_decode() {
        use jpxl_core::limits::Limits;

        let width = 300u32;
        let height = 200u32;
        let plane: Plane = (0..height)
            .flat_map(|y| (0..width).map(move |x| ((x * 5 + y * 11) % 200) as i32))
            .collect();
        let image = crate::Image::new(width, height, 8, vec![plane.clone()]).expect("image");
        let forced = validate(LosslessPlan {
            group_size_shift: 0, // group_dim 128 -> multi-section at 300x200
            rct: false,
            palette: None,
            squeeze: false,
            tree: MaTree::single_leaf(Predictor::Weighted),
        })
        .expect("validate");
        let bytes =
            crate::encode_codestream_with_plan(&image, image.planes(), &forced).expect("encode");
        let decoded = jpxl_decode::decode(&bytes, &Limits::default()).expect("decode");
        let got = &decoded.planes.first().expect("plane").samples;
        assert_eq!(got.len(), plane.len());
        for (i, (&a, &b)) in got.iter().zip(plane.iter()).enumerate() {
            assert_eq!(a, b, "sample {i}");
        }
    }

    /// Phase 4A's decision-preservation guarantee is structural, not a
    /// tolerance check like Phase 4B's: Weighted is a strictly ADDED
    /// candidate scored by the same exact final-price gate every other
    /// predictor goes through (`total_cost_source(..., PriceTier::Exact)`,
    /// unconditional regardless of which tier's search proposed the
    /// finalist), so adding it can only ever match or beat what the
    /// pre-4A six-predictor search would have chosen -- never lose. This
    /// pins that down directly: plan_for's real total size for a
    /// Weighted-friendly pattern must be <= a forced six-predictor-only
    /// baseline's, not merely "close".
    #[test]
    fn plan_for_with_weighted_never_loses_to_the_pre_4a_six_predictor_search() {
        let width = 40u32;
        let height = 37u32;
        let plane: Plane = (0..height)
            .flat_map(|y| (0..width).map(move |x| ((x * 5 + y * 11) % 200) as i32))
            .collect();
        let image = crate::Image::new(width, height, 8, vec![plane.clone()]).expect("image");

        let plan = plan_for(
            width,
            height,
            &[plane.clone()],
            false,
            &EncodeOptions::default(),
        )
        .expect("plan");
        let with_weighted =
            crate::encode_codestream_with_plan(&image, image.planes(), &plan).expect("encode");

        // The pre-4A baseline: best of the original six predictors only,
        // exact-priced the same way plan_for's own Stage C does.
        let shared: modular::SharedPlane = std::sync::Arc::from(plane.as_slice());
        let six_predictors = [
            Predictor::Zero,
            Predictor::West,
            Predictor::North,
            Predictor::AverageWestNorth,
            Predictor::Select,
            Predictor::Gradient,
        ];
        let mut best: Option<(u64, Predictor)> = None;
        for &predictor in &six_predictors {
            let tree = MaTree::single_leaf(predictor);
            let cost = total_cost_shared(
                width,
                height,
                std::slice::from_ref(&shared),
                &tree,
                DEFAULT_GROUP_SIZE_SHIFT,
                PriceTier::Exact,
            )
            .expect("exact cost");
            if best.as_ref().is_none_or(|(c, _)| cost < *c) {
                best = Some((cost, predictor));
            }
        }
        let (_, baseline_predictor) = best.expect("six candidates");
        let baseline_plan = validate(LosslessPlan {
            group_size_shift: DEFAULT_GROUP_SIZE_SHIFT,
            rct: false,
            palette: None,
            squeeze: false,
            tree: MaTree::single_leaf(baseline_predictor),
        })
        .expect("validate");
        let baseline_bytes =
            crate::encode_codestream_with_plan(&image, image.planes(), &baseline_plan)
                .expect("encode");

        assert!(
            with_weighted.len() <= baseline_bytes.len(),
            "Weighted-enabled plan ({} bytes) must never exceed the six-predictor \
             baseline ({} bytes)",
            with_weighted.len(),
            baseline_bytes.len()
        );
    }
}
