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

/// The lossless-modular encoder's **effort** budget: how hard the planner
/// searches for a small file, from `1` (fastest, largest) to `9` (slowest,
/// densest). Effort is an encoder-side *speed/size* dial the standard leaves
/// entirely free; every level is exact-lossless, so it never changes the
/// pixels a conforming decoder reconstructs — only the encoded byte count and
/// the time the search spends.
///
/// Effort and a (future) *quality* target are separate axes
/// (`docs/Encoder-plan1.md` §12): quality picks the fidelity, effort picks how
/// much compute is spent reaching it. This type is the effort axis for the
/// lossless track. It is expanded **once**, at the [`plan_for`] stage
/// boundary, into a `ModularSearchBudget`; no kernel re-reads the level.
///
/// [`Effort::DEFAULT`] is the **lean** level 1. On photographic and smooth
/// content it produces byte-identical output to the full search (level 7) in a
/// fraction of the time: that search's extra predictors, split grid, per-leaf
/// refinement and finer stride are chaff on such content — measured at 71
/// residual scans versus 1 for the *same* bytes
/// (jpegxl-rs.evidence.modular-search-chaff-2026-08-10). Levels above the
/// default only help, and only by a few percent, on specific content
/// (flat/paletteable, or squeezing the last few percent out of a photo); that
/// cost is opt-in, not a tax every encode pays. Level 7 is retained as the
/// full pre-ramp search — a byte-identical density anchor for content the lean
/// default leaves on the table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Effort(u8);

impl Effort {
    /// Fastest, leanest search — and the default: one strong predictor, no
    /// transform trials, collapsed tree search.
    pub const MIN: Effort = Effort(1);
    /// Slowest, densest output.
    pub const MAX: Effort = Effort(9);
    /// The default effort: the lean level 1. More effort rarely reduces size on
    /// typical content, so the sensible default is the fast one and density
    /// chasing is opt-in via a higher level.
    pub const DEFAULT: Effort = Effort(1);

    /// Wraps `level`, which must be in `1..=9`.
    ///
    /// # Errors
    ///
    /// [`EncodeError::ValueOutOfRange`] if `level` is `0` or above `9`.
    pub fn new(level: u8) -> Result<Self> {
        if !(1..=9).contains(&level) {
            return Err(EncodeError::ValueOutOfRange {
                what: "effort",
                value: i64::from(level),
            });
        }
        Ok(Self(level))
    }

    /// The effort level, in `1..=9`.
    #[must_use]
    pub const fn level(self) -> u8 {
        self.0
    }
}

impl Default for Effort {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// The concrete search levers an [`Effort`] expands into, chosen **once** at
/// the [`plan_for`] stage boundary (`docs/Encoder-plan1.md` §12: select the
/// budget at a stage boundary, never scatter effort checks through kernels).
/// Every field controls only the *search*: none of them affects decoded pixels
/// (lossless is exact at every level), and the Stage-C exact-price gate
/// re-prices the settled winner regardless of how cheaply the search ranked it.
#[derive(Debug, Clone, Copy)]
struct ModularSearchBudget {
    /// Predictors tried in the single-leaf sweep and per-leaf refinement.
    predictors: &'static [Predictor],
    /// Table H.4 property indices tried for splits.
    split_properties: &'static [u32],
    /// Thresholds tried for `property > value` decisions.
    split_thresholds: &'static [i32],
    /// Cap on MA-tree depth (root = 0).
    max_tree_depth: u32,
    /// Cap on MA-tree leaves / residual contexts.
    max_tree_leaves: usize,
    /// Above this sample count the tree search collapses to one split.
    deep_search_sample_cap: u64,
    /// Whether to run per-leaf predictor refinement.
    refine_leaves: bool,
    /// Whether to trial an exact-colour palette transform.
    try_palette: bool,
    /// Whether to trial the default squeeze transform.
    try_squeeze: bool,
    /// **Minimum** row stride for the sampled cheap-tier scorer (`1` = every
    /// row). A floor, not the stride itself: the stride actually used is
    /// [`effective_cheap_stride`], which raises this when the frame would
    /// otherwise blow [`Self::cheap_sample_budget`].
    cheap_row_stride: u32,
    /// Target number of samples the cheap tier may score per candidate,
    /// summed across score planes.
    ///
    /// This is what keeps ranking cost *bounded*. A stride is a ratio, so a
    /// fixed stride still costs `frame_area / stride` — linear in frame size,
    /// which is why the tree had to be collapsed on large frames
    /// ([`Self::deep_search_sample_cap`]). A sample budget is an absolute
    /// bound: the stride is derived from it, so a 12 MP frame costs the same
    /// per candidate as a 1 MP one and depth becomes affordable at any size.
    ///
    /// [`u64::MAX`] means "unbounded" — the floor stride wins and behaviour is
    /// exactly the pre-budget fixed-stride search.
    cheap_sample_budget: u64,
}

/// The row stride the cheap tier actually uses: [`ModularSearchBudget::cheap_row_stride`]
/// raised until scoring one candidate costs at most `sample_budget` samples.
///
/// Pure in its arguments — constants and frame dimensions only, no RNG and no
/// ambient state — so the search stays a deterministic function of its budget
/// (`same_effort_is_deterministic`).
///
/// Never returns less than `floor_stride`, so a frame already inside the budget
/// is scored exactly as before and this can only ever *remove* work, never add
/// it. A zero budget (or arithmetic that saturates) also falls back to the
/// floor rather than inventing a stride.
fn effective_cheap_stride(
    sample_budget: u64,
    floor_stride: u32,
    width: u32,
    height: u32,
    num_planes: usize,
) -> u32 {
    if sample_budget == 0 {
        return floor_stride;
    }
    let samples = u64::from(width)
        .saturating_mul(u64::from(height))
        .saturating_mul(u64::try_from(num_planes).unwrap_or(u64::MAX));
    // Ceiling division: the smallest stride whose scored-sample count fits.
    let needed = samples.div_ceil(sample_budget).max(1);
    let needed = u32::try_from(needed).unwrap_or(u32::MAX);
    floor_stride.max(needed)
}

// Predictor ladders. Every subset keeps `PREDICTOR_CANDIDATES`' order so that
// level 7's tie-breaks (the sweep keeps the earliest predictor at the minimum
// cost) match the pre-ramp search exactly. Efforts 1-6 exclude the stateful
// Weighted predictor (6), whose every trial forces a full sequential scan
// (`estimate_residual_bits_sampled` falls back to the full scan for any
// Weighted-containing tree) — dropping it is the single largest low-effort
// speed win.
const PREDS_E1: &[Predictor] = &[Predictor::Gradient];
const PREDS_E2: &[Predictor] = &[Predictor::West, Predictor::North, Predictor::Gradient];
const PREDS_E3: &[Predictor] = &[
    Predictor::Zero,
    Predictor::West,
    Predictor::North,
    Predictor::Gradient,
];
/// The six order-independent predictors (the pre-4A set): `PREDICTOR_CANDIDATES`
/// without Weighted. Used by efforts 4-6.
const PREDS_STATELESS: &[Predictor] = &[
    Predictor::Zero,
    Predictor::West,
    Predictor::North,
    Predictor::AverageWestNorth,
    Predictor::Select,
    Predictor::Gradient,
];

const PROPS_E2: &[u32] = &[6, 7];
const PROPS_E3: &[u32] = &[4, 6, 7];
const PROPS_E4: &[u32] = &[4, 5, 6, 7];
const PROPS_E5: &[u32] = &[4, 5, 6, 7, 9];

const THRESH_E2: &[i32] = &[0, 4];
const THRESH_E3: &[i32] = &[0, 2, 8];
const THRESH_E4: &[i32] = &[0, 1, 4, 16];
const THRESH_E5: &[i32] = &[0, 1, 2, 8, 32];
/// Finer threshold grid for efforts 8-9. The *property* axis is already maxed
/// at level 7 — the encoder computes only properties {4,5,6,7,9,10,11}
/// (`modular::property_value`; every other index reads as a constant 0), so a
/// higher rung earns density from finer thresholds, deeper trees, a larger
/// deep-search cap, and a finer sampling stride, not from new properties.
const THRESH_FINE: &[i32] = &[0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64];

impl ModularSearchBudget {
    /// Expands an [`Effort`] level into concrete search levers.
    ///
    /// Level 7's *levers* are the pre-ramp search, field for field, and
    /// `effort_7_budget_is_the_pre_ramp_search` pins them. Its *output* is
    /// deliberately no longer byte-identical to the pre-ramp encoder: the
    /// cheap tier used to compare a `1/row_stride` residual estimate against
    /// an unscaled whole-frame tree cost, which over-priced every extra
    /// context by the sampling ratio and hid the winning split at any stride
    /// above 1. `modular::estimate_residual_bits_sampled` now extrapolates to
    /// whole-frame bits, so level 7 finds the tree only level 9 could reach
    /// before (measured -3.5% at 0.8 MP, -4.2% at 4 MP, unchanged at 12 MP).
    /// Levels 1-6 trade density for speed; 8-9 currently find nothing level 7
    /// does not.
    fn for_effort(effort: Effort) -> Self {
        // The default/level-7 search, reused for level 7 and as a defensive
        // fallback. Every other arm is a deliberate deviation from it.
        let default = Self {
            predictors: PREDICTOR_CANDIDATES,
            split_properties: SPLIT_PROPERTIES,
            split_thresholds: SPLIT_THRESHOLDS,
            max_tree_depth: MAX_TREE_DEPTH,
            max_tree_leaves: MAX_TREE_LEAVES,
            deep_search_sample_cap: DEEP_SEARCH_SAMPLE_CAP,
            refine_leaves: true,
            try_palette: true,
            try_squeeze: true,
            cheap_row_stride: SAMPLED_GATHER_ROW_STRIDE,
            // Unbounded for now at every level: the sample-budget plumbing
            // lands as a provable no-op, and the finite per-level budgets are
            // set only once the depth measurement says depth is worth paying
            // for. Level 7 must stay `MAX` permanently — it is the
            // byte-identical density anchor.
            cheap_sample_budget: u64::MAX,
        };
        match effort.level() {
            1 => Self {
                predictors: PREDS_E1,
                split_properties: &[],
                split_thresholds: &[],
                max_tree_depth: 0,
                max_tree_leaves: 1,
                refine_leaves: false,
                try_palette: false,
                try_squeeze: false,
                cheap_row_stride: 8,
                ..default
            },
            2 => Self {
                predictors: PREDS_E2,
                split_properties: PROPS_E2,
                split_thresholds: THRESH_E2,
                max_tree_depth: 1,
                max_tree_leaves: 2,
                refine_leaves: false,
                try_palette: true,
                try_squeeze: false,
                cheap_row_stride: 8,
                ..default
            },
            3 => Self {
                predictors: PREDS_E3,
                split_properties: PROPS_E3,
                split_thresholds: THRESH_E3,
                max_tree_depth: 2,
                max_tree_leaves: 4,
                refine_leaves: false,
                cheap_row_stride: 6,
                ..default
            },
            4 => Self {
                predictors: PREDS_STATELESS,
                split_properties: PROPS_E4,
                split_thresholds: THRESH_E4,
                max_tree_depth: 3,
                max_tree_leaves: 6,
                ..default
            },
            5 => Self {
                predictors: PREDS_STATELESS,
                split_properties: PROPS_E5,
                split_thresholds: THRESH_E5,
                ..default
            },
            6 => Self {
                predictors: PREDS_STATELESS,
                ..default
            },
            8 => Self {
                split_thresholds: THRESH_FINE,
                max_tree_depth: 5,
                max_tree_leaves: 10,
                deep_search_sample_cap: 1 << 16,
                cheap_row_stride: 2,
                ..default
            },
            9 => Self {
                split_thresholds: THRESH_FINE,
                max_tree_depth: 6,
                max_tree_leaves: 12,
                deep_search_sample_cap: 1 << 18,
                cheap_row_stride: 1,
                ..default
            },
            // 7 (and any out-of-range level a caller forced past `Effort::new`).
            _ => default,
        }
    }

    /// Applies the caller's measurement overrides on top of the effort's
    /// levers, at the same stage boundary the effort itself is expanded.
    ///
    /// Each `None` keeps the effort's choice, so an empty override set is the
    /// identity and the shipped ladder is untouched.
    fn with_overrides(self, overrides: &crate::ModularSearchOverrides) -> Self {
        Self {
            max_tree_depth: overrides.max_tree_depth.unwrap_or(self.max_tree_depth),
            max_tree_leaves: overrides.max_tree_leaves.unwrap_or(self.max_tree_leaves),
            cheap_sample_budget: overrides
                .cheap_sample_budget
                .unwrap_or(self.cheap_sample_budget),
            deep_search_sample_cap: overrides
                .deep_search_sample_cap
                .unwrap_or(self.deep_search_sample_cap),
            ..self
        }
    }

    /// Whether this budget makes [`plan_for`]'s search a foregone conclusion.
    ///
    /// With a single predictor there is nothing for the sweep to rank; with an
    /// empty property *or* threshold grid the greedy split loop's inner loops
    /// never execute, so it breaks on its first round having improved nothing;
    /// without leaf refinement and without the palette and squeeze trials there
    /// is no later stage that can move the answer. The plan is then exactly
    /// `single_leaf(predictors[0])` with the caller's `rct` and neither
    /// transform — reachable without wrapping a single plane or scanning a
    /// single residual.
    ///
    /// This is a pure work-elision predicate: when it holds, the short-circuit
    /// must emit the same plan the full path would have, and
    /// `degenerate_budget_plans_what_the_full_search_would_have` proves it.
    fn search_is_a_foregone_conclusion(&self) -> bool {
        self.predictors.len() == 1
            && (self.split_properties.is_empty() || self.split_thresholds.is_empty())
            && !self.refine_leaves
            && !self.try_palette
            && !self.try_squeeze
    }
}

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
    let budget = ModularSearchBudget::for_effort(options.effort)
        .with_overrides(&options.modular_search_overrides);
    let group_size_shift = options.group_size_shift.unwrap_or(DEFAULT_GROUP_SIZE_SHIFT);

    // Nothing in the search can move the answer at this budget, so skip
    // straight to it: the RCT scoring copy, the Arc wrap of every plane, and
    // the cheap residual scan are all pure overhead on the way to a plan the
    // budget has already determined. This is the *whole* cost of planning at
    // the leanest effort — at 12 MP two full 144 MB plane copies plus a
    // strided residual scan of the frame — and none of it changes a byte of
    // output.
    if budget.search_is_a_foregone_conclusion() {
        let predictor = budget
            .predictors
            .first()
            .copied()
            .unwrap_or(Predictor::Gradient);
        return validate(LosslessPlan {
            group_size_shift,
            rct,
            palette: None,
            squeeze: false,
            tree: MaTree::single_leaf(predictor),
        });
    }

    plan_by_full_search(
        width,
        height,
        planes,
        rct,
        budget,
        group_size_shift,
        #[cfg(test)]
        FullSearchProbe::Normal,
    )
}

/// In tests, forces [`plan_by_full_search`] to run even for a budget the
/// short-circuit would have answered — the only way to compare the two paths.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FullSearchProbe {
    Normal,
    /// Run the full search on a budget [`plan_for`] would have short-circuited.
    ForegoneBudget,
}

/// The full tiered search: predictor sweep, greedy splits, leaf refinement,
/// then the exact-priced palette and squeeze trials.
///
/// Split out from [`plan_for`] so that the short-circuit above can be *proved*
/// equivalent on the budgets it claims rather than merely argued: the test
/// `degenerate_budget_plans_what_the_full_search_would_have` calls both and
/// compares the resulting plans.
///
/// # Errors
///
/// Any error from residual collection, ANS table build, or [`validate`].
fn plan_by_full_search(
    width: u32,
    height: u32,
    planes: &[Plane],
    rct: bool,
    budget: ModularSearchBudget,
    group_size_shift: u32,
    #[cfg(test)] probe: FullSearchProbe,
) -> Result<ValidatedLosslessPlan> {
    #[cfg(test)]
    debug_assert!(
        probe == FullSearchProbe::ForegoneBudget || !budget.search_is_a_foregone_conclusion(),
        "plan_for should have short-circuited this budget"
    );
    let samples = u64::from(width).saturating_mul(u64::from(height));
    let max_leaves = if samples > budget.deep_search_sample_cap {
        2
    } else {
        budget.max_tree_leaves
    };
    let max_depth = if samples > budget.deep_search_sample_cap {
        1
    } else {
        budget.max_tree_depth
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

    // The one place the sample budget becomes a concrete stride: dimensions
    // and plane count are both known here for the first time, and every cheap
    // price below reads this single value. Per `docs/Encoder-plan1.md` §12 the
    // budget is expanded at a stage boundary, never re-derived in a kernel.
    let cheap_row_stride = effective_cheap_stride(
        budget.cheap_sample_budget,
        budget.cheap_row_stride,
        width,
        height,
        shared_score.len(),
    );

    for &predictor in budget.predictors {
        let tree = MaTree::single_leaf(predictor);
        let cost = total_cost_shared(
            width,
            height,
            &shared_score,
            &tree,
            group_size_shift,
            PriceTier::Cheap {
                row_stride: cheap_row_stride,
            },
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
            for &property in budget.split_properties {
                for &value in budget.split_thresholds {
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
                        PriceTier::Cheap {
                            row_stride: cheap_row_stride,
                        },
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
    if budget.refine_leaves {
        let leaf_count = best_tree.num_contexts();
        for ctx in 0..leaf_count {
            let current = best_tree
                .leaf_predictors()
                .get(ctx)
                .copied()
                .unwrap_or(Predictor::Gradient);
            for &predictor in budget.predictors {
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
                    PriceTier::Cheap {
                        row_stride: cheap_row_stride,
                    },
                )?;
                if cost < best_cost {
                    best_cost = cost;
                    best_tree = candidate;
                }
            }
        }
    }

    // Stage C: exact residual price of the MA finalist, the baseline the
    // palette/squeeze trials compare against. Skipped when neither trial will
    // run (effort 1 drops both; effort 2 keeps only palette), because the
    // exact re-price is then pure overhead — the emitted tree is `best_tree`
    // regardless of this number.
    //
    // Phase 4D: Exact finalists use allow_lz77=true so ranking sees the same
    // residual LZ77 gate as final emission (`encode_codestream_with_plan`).
    // Cheap-tier trials stay LZ77-free via `total_cost_shared` (Shannon residual
    // estimate only).
    if budget.try_palette || budget.try_squeeze {
        let ma_source = modular::ModularSource::direct_shared(
            width,
            height,
            &shared_score,
            false,
            best_tree.clone(),
            true,
        );
        best_cost = total_cost_source(&ma_source, group_size_shift, PriceTier::Exact)?;
    }

    // Nested palette trial on *source* samples (not RCT). Exact-price gate.
    let mut palette: Option<PaletteForward> = None;
    let mut use_rct = rct;
    let mut use_squeeze = false;
    let num_c = planes.len();
    if budget.try_palette
        && (num_c == 1 || num_c == 3)
        && let Some(fwd) = try_exact_palette(width, height, planes, 0, num_c)?
    {
        let tree = MaTree::single_leaf(Predictor::Zero);
        let cost = total_cost_source(
            &modular::ModularSource::from_palette(fwd.clone(), tree.clone(), true),
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
    if budget.try_squeeze
        && palette.is_none()
        && squeeze::default_would_run(width, height, score_planes.len())
    {
        let tree = MaTree::single_leaf(Predictor::Gradient);
        let source = modular::ModularSource::with_default_squeeze(
            width,
            height,
            score_planes,
            false, // RCT already in score_planes samples
            tree.clone(),
            true,
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
    /// Collect residuals + Shannon hybrid estimate (no ANS emit). `row_stride`
    /// is the sampled-gather stride the effort budget chose (`1` = every row).
    Cheap { row_stride: u32 },
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
    static PLAN_DIAGNOSTICS_ENABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static LAST_PLAN_MULTIPLICITY: std::cell::Cell<PlanMultiplicity> =
        const { std::cell::Cell::new(PlanMultiplicity {
            exact_residual_prices: 0,
            cheap_scores: 0,
            residual_scans: 0,
            plane_clone_bytes: 0,
        }) };
}

/// Enables or disables modular planning diagnostics on the current thread.
///
/// The normal encoder leaves this off. `jpxl bench --diag` enables it around
/// its warm-up and timed iterations so production planning does not pay for
/// counter updates nobody reads.
pub fn set_plan_diagnostics_enabled(enabled: bool) {
    PLAN_DIAGNOSTICS_ENABLED.with(|cell| cell.set(enabled));
}

#[inline]
fn plan_diagnostics_enabled() -> bool {
    PLAN_DIAGNOSTICS_ENABLED.with(std::cell::Cell::get)
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
    if !plan_diagnostics_enabled() {
        return;
    }
    LAST_PLAN_MULTIPLICITY.with(|c| {
        let mut m = c.get();
        m.exact_residual_prices = m.exact_residual_prices.saturating_add(1);
        m.residual_scans = m.residual_scans.saturating_add(1);
        c.set(m);
    });
}

/// Records plane deep-clone traffic from modular source construction (Phase-0).
pub(crate) fn note_plane_clone_bytes(bytes: u64) {
    if !plan_diagnostics_enabled() {
        return;
    }
    LAST_PLAN_MULTIPLICITY.with(|c| {
        let mut m = c.get();
        m.plane_clone_bytes = m.plane_clone_bytes.saturating_add(bytes);
        c.set(m);
    });
}

fn bump_cheap() {
    if !plan_diagnostics_enabled() {
        return;
    }
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
    // Phase 4C: multi-section emission pays the MA tree once at G.1.3; the
    // single-section path also pays it once. Search must not charge tree ×
    // group count or it over-prices every multi-group frame.
    let _ = crate::frame::Geometry::new(source.width, source.height, group_size_shift)?;
    Ok(residual.saturating_add(tree_bits))
}

/// Residual-section bit cost under the chosen tier.
fn residual_stream_cost_source(
    source: &modular::ModularSource,
    group_size_shift: u32,
    tier: PriceTier,
) -> Result<u64> {
    match tier {
        PriceTier::Cheap { row_stride } => {
            bump_cheap();
            #[cfg(feature = "phase4b-sampled-gather")]
            {
                sampled_cheap_residual_bits(source, row_stride)
            }
            #[cfg(not(feature = "phase4b-sampled-gather"))]
            {
                let _ = row_stride;
                cheap_residual_bits(source, group_size_shift)
            }
        }
        PriceTier::Exact => {
            bump_exact();
            exact_residual_bits(source, group_size_shift)
        }
    }
}

/// The level-7 (default) cheap-tier row stride: every 4th row (plus the last),
/// so the topology search costs roughly a quarter of a full-plane scan. The
/// effort budget varies this (efforts 1-2 use `8`, effort 9 uses `1` = every
/// row); `ModularSearchBudget::for_effort` reads this for the default level, so
/// it is compiled in every configuration.
const SAMPLED_GATHER_ROW_STRIDE: u32 = 4;

/// Phase 4B additive scorer: rows a strided subset instead of a full plane
/// scan. Always compiled (regardless of the feature) so decision-preservation
/// tests can compare it against [`cheap_residual_bits`] directly; only
/// [`residual_stream_cost_source`]'s dispatch is feature-gated, so a default
/// build's search path never calls this function.
///
/// See jpegxl-rs.work.arch-phase4b-sampled-gather-scoped.
#[cfg(any(feature = "phase4b-sampled-gather", test))]
fn sampled_cheap_residual_bits(source: &modular::ModularSource, row_stride: u32) -> Result<u64> {
    modular::estimate_residual_bits_sampled(source, row_stride)
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

    #[test]
    fn disabled_plan_diagnostics_do_not_accumulate() {
        set_plan_diagnostics_enabled(false);
        reset_multiplicity();
        bump_exact();
        bump_cheap();
        note_plane_clone_bytes(4096);
        assert_eq!(last_plan_multiplicity(), PlanMultiplicity::default());
    }
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
                let sampled_cost = sampled_cheap_residual_bits(&source, SAMPLED_GATHER_ROW_STRIDE)
                    .expect("sampled cost");
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
        let cost = sampled_cheap_residual_bits(&source, SAMPLED_GATHER_ROW_STRIDE)
            .expect("sampled cost on a tiny frame");
        assert!(cost > 0, "a non-empty frame must not cost zero bits");
    }

    /// The short-circuit in [`plan_for`] claims that for a budget where
    /// [`ModularSearchBudget::search_is_a_foregone_conclusion`] holds, running
    /// the search would land on the same plan. Prove it rather than argue it:
    /// run both paths on the same input and compare the plans field for field.
    ///
    /// Covers both `rct` polarities and both channel counts that matter (one
    /// plane, and the three-plane case where the skipped work includes the RCT
    /// scoring copy), on content with enough structure that a search which
    /// *could* move the answer would.
    #[test]
    fn degenerate_budget_plans_what_the_full_search_would_have() {
        let width = 37u32; // Deliberately not a multiple of any stride.
        let height = 29u32;
        let planes: Vec<Plane> = (0..3)
            .map(|c: i32| {
                (0..height)
                    .flat_map(|y| {
                        (0..width).map(move |x| {
                            // Structured but not flat: gradient, a diagonal
                            // edge, and a per-channel offset, so the split
                            // properties see real variation.
                            let base = (x as i32) * 3 + (y as i32) * 5;
                            let edge = if x as i32 > y as i32 { 90 } else { 0 };
                            (base + edge + c * 17) & 0xff
                        })
                    })
                    .collect()
            })
            .collect();

        for level in 1..=9u8 {
            let effort = Effort::new(level).expect("level in range");
            let budget = ModularSearchBudget::for_effort(effort);
            if !budget.search_is_a_foregone_conclusion() {
                continue;
            }
            for rct in [false, true] {
                for channels in [1usize, 3] {
                    let subset = &planes[..channels];
                    let options = EncodeOptions {
                        effort,
                        ..EncodeOptions::default()
                    };
                    let group_size_shift =
                        options.group_size_shift.unwrap_or(DEFAULT_GROUP_SIZE_SHIFT);

                    let short = plan_for(width, height, subset, rct, &options)
                        .expect("short-circuited plan");
                    let full = plan_by_full_search(
                        width,
                        height,
                        subset,
                        rct,
                        budget,
                        group_size_shift,
                        FullSearchProbe::ForegoneBudget,
                    )
                    .expect("full-search plan");

                    let (s, f) = (short.plan(), full.plan());
                    let where_ = format!("effort {level}, rct {rct}, {channels} channel(s)");
                    assert_eq!(s.group_size_shift, f.group_size_shift, "{where_}");
                    assert_eq!(s.rct, f.rct, "{where_}");
                    assert_eq!(s.squeeze, f.squeeze, "{where_}");
                    assert_eq!(s.palette.is_some(), f.palette.is_some(), "{where_}");
                    assert_eq!(
                        s.tree.leaf_predictors(),
                        f.tree.leaf_predictors(),
                        "{where_}"
                    );
                    assert_eq!(s.tree.num_contexts(), f.tree.num_contexts(), "{where_}");
                    assert_eq!(s.tree.depth(), f.tree.depth(), "{where_}");
                }
            }
        }
    }

    /// The short-circuit is only worth its risk if it is actually reached by
    /// the default. If a future budget edit makes level 1 non-degenerate, this
    /// fails loudly rather than silently restoring the two 144 MB plane copies.
    #[test]
    fn the_default_effort_reaches_the_short_circuit() {
        assert!(
            ModularSearchBudget::for_effort(Effort::DEFAULT).search_is_a_foregone_conclusion(),
            "the default effort must skip the search it cannot use"
        );
        assert!(
            !ModularSearchBudget::for_effort(Effort::new(7).expect("7"))
                .search_is_a_foregone_conclusion(),
            "the density anchor must still run the full search"
        );
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
        set_plan_diagnostics_enabled(true);
        let width = 48u32;
        let height = 48u32;
        let plane: Plane = (0..height)
            .flat_map(|y| (0..width).map(move |x| ((x * 3 + y * 5) % 200) as i32))
            .collect();
        let _ = plan_for(width, height, &[plane], false, &full_search_options()).expect("plan");
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
        set_plan_diagnostics_enabled(false);
        assert!(
            m.cheap_scores > m.exact_residual_prices * 5,
            "cheap ranking should dominate exact finalist prices: {m:?}"
        );
    }

    #[test]
    fn exact_finalist_pricing_sees_residual_lz77() {
        // Phase 4D: Exact-tier sources must use allow_lz77=true so ranking
        // matches emission. On a multi-value repeating residual pattern the
        // LZ77 stream is strictly shorter than plain ANS (constant zeros are
        // near-free under plain ANS and do not adopt LZ77). Pricing with
        // allow_lz77=false would over-estimate that finalist.
        use jpxl_bitstream::BitWriter;
        let width = 32u32;
        let height = 40u32;
        // Zero predictor ⇒ residual == sample. Period-32 values repeated for
        // 40 rows match residual_lz77_beats_plain_on_a_repeating_multi_value_pattern.
        let plane: Plane = (0..height)
            .flat_map(|_| (0..width).map(|x| x as i32))
            .collect();
        let tree = MaTree::single_leaf(Predictor::Zero);
        let with = modular::ModularSource::direct(
            width,
            height,
            std::slice::from_ref(&plane),
            false,
            tree.clone(),
            true,
        );
        let without = modular::ModularSource::direct(width, height, &[plane], false, tree, false);
        let mut w_lz = BitWriter::counting();
        modular::write_residual_payload_full(&mut w_lz, &with).expect("lz");
        let mut w_plain = BitWriter::counting();
        modular::write_residual_payload_full(&mut w_plain, &without).expect("plain");
        assert!(
            w_lz.bit_len() < w_plain.bit_len(),
            "allow_lz77=true must price a repeating residual cheaper than plain: {} vs {}",
            w_lz.bit_len(),
            w_plain.bit_len()
        );
    }

    #[test]
    fn plan_for_adopts_palette_on_scattered_few_colours() {
        // Few unique levels with NO exploitable structure between
        // neighbours -> every predictor loses, including Phase 4A's
        // Weighted, AND the index sequence has no runs an LZ77 pass could
        // exploit either. Phase 4D prices Exact finalists with allow_lz77
        // (matching emission); a weaker scatter than a full avalanche hash
        // left enough residual periodicity for LZ77 to distort ranking in
        // older fixtures. A murmur3-style finalizer avalanches (x, y) well
        // enough that neither prediction nor LZ77 can find structure, so
        // exact-price palette should still win cleanly.
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
        let plan = plan_for(width, height, &[plane], false, &full_search_options()).expect("plan");
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
            std::slice::from_ref(&plane),
            false,
            &full_search_options(),
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

    // --- Effort ramp (speed/size dial) ---

    /// Level 7's levers are the pre-ramp search, field for field.
    ///
    /// This pins the *budget*, not the output. Level 7's bytes intentionally
    /// changed when the cheap tier stopped comparing a sampled residual
    /// estimate against an unscaled whole-frame tree cost; see
    /// [`ModularSearchBudget::for_effort`]. The default effort (1) is the one
    /// whose byte-identity the corpus fingerprint gate still proves, and it
    /// short-circuits this search entirely.
    #[test]
    fn effort_7_budget_is_the_pre_ramp_search() {
        let b = ModularSearchBudget::for_effort(Effort::new(7).expect("valid"));
        assert_eq!(b.predictors, PREDICTOR_CANDIDATES);
        assert_eq!(b.split_properties, SPLIT_PROPERTIES);
        assert_eq!(b.split_thresholds, SPLIT_THRESHOLDS);
        assert_eq!(b.max_tree_depth, MAX_TREE_DEPTH);
        assert_eq!(b.max_tree_leaves, MAX_TREE_LEAVES);
        assert_eq!(b.deep_search_sample_cap, DEEP_SEARCH_SAMPLE_CAP);
        assert!(b.refine_leaves && b.try_palette && b.try_squeeze);
        assert_eq!(b.cheap_row_stride, SAMPLED_GATHER_ROW_STRIDE);
        // The anchor must never acquire a finite sample budget: a bounded
        // cheap tier would re-rank candidates on a subset and could move the
        // emitted tree, which is exactly what this level exists not to do.
        assert_eq!(
            b.cheap_sample_budget,
            u64::MAX,
            "level 7 is the byte-identical density anchor; it must stay unbounded"
        );
    }

    /// The sample budget is a *bound*, not a replacement policy: whenever the
    /// frame already fits inside it, the derived stride must be exactly the
    /// effort's floor, so scoring is identical to the pre-budget search. This
    /// is what makes the plumbing step provably output-preserving.
    #[test]
    fn bounded_budget_is_a_noop_below_the_budget() {
        // Every shipped effort level, against a frame comfortably inside any
        // plausible budget.
        for level in 1..=9u8 {
            let b = ModularSearchBudget::for_effort(Effort::new(level).expect("level in range"));
            let stride =
                effective_cheap_stride(b.cheap_sample_budget, b.cheap_row_stride, 64, 64, 3);
            assert_eq!(
                stride, b.cheap_row_stride,
                "effort {level}: a sub-budget frame must score at the floor stride"
            );
        }

        // The boundary itself: exactly at the budget is still "fits".
        assert_eq!(effective_cheap_stride(3 * 100 * 100, 4, 100, 100, 3), 4);
        // One sample over, but the implied stride (2) is below the floor (4),
        // so the floor still wins — the budget can only ever *raise* it.
        assert_eq!(effective_cheap_stride(3 * 100 * 100 - 1, 4, 100, 100, 3), 4);
    }

    /// The other half of the contract: above the budget the stride rises so
    /// that scored samples stay bounded, and it rises with frame size rather
    /// than staying a fixed ratio. This is the property that lets the tree
    /// depth cap be retired.
    #[test]
    fn bounded_budget_raises_the_stride_with_frame_size() {
        let budget = 1u64 << 16; // 65,536 scored samples.
        let floor = 1u32;

        let small = effective_cheap_stride(budget, floor, 256, 256, 3);
        let large = effective_cheap_stride(budget, floor, 4000, 3000, 3);
        assert!(
            large > small,
            "a bigger frame must be scored more sparsely, not proportionally: {small} vs {large}"
        );

        // The bound actually holds, and — the point of the whole change — it
        // does NOT grow with frame size. Sampling is row-granular, so the
        // scored count can exceed the budget by at most the two partial rows
        // the ceiling and the always-included final row contribute; that slack
        // is a function of row cost, not of frame area.
        for (w, h) in [(256u32, 256u32), (2400, 1800), (4000, 3000)] {
            let stride = effective_cheap_stride(budget, floor, w, h, 3);
            let row_cost = u64::from(w) * 3;
            let rows = u64::from(h).div_ceil(u64::from(stride)) + 1;
            let scored = rows * row_cost;
            assert!(
                scored <= budget + 2 * row_cost,
                "{w}x{h}: scored {scored} samples against a {budget} budget \
                 (stride {stride}, slack {} for two partial rows)",
                2 * row_cost
            );
        }
    }

    /// The measurement escape hatch must be inert unless asked: an empty
    /// override set is the identity on every effort's budget, so the shipped
    /// ladder cannot drift just because the hatch exists.
    #[test]
    fn empty_overrides_leave_every_effort_budget_untouched() {
        let none = crate::ModularSearchOverrides::default();
        assert!(none.is_empty());
        for level in 1..=9u8 {
            let base = ModularSearchBudget::for_effort(Effort::new(level).expect("level in range"));
            let after = base.with_overrides(&none);
            let where_ = format!("effort {level}");
            assert_eq!(after.max_tree_depth, base.max_tree_depth, "{where_}");
            assert_eq!(after.max_tree_leaves, base.max_tree_leaves, "{where_}");
            assert_eq!(
                after.cheap_sample_budget, base.cheap_sample_budget,
                "{where_}"
            );
            assert_eq!(
                after.deep_search_sample_cap, base.deep_search_sample_cap,
                "{where_}"
            );
            assert_eq!(after.predictors, base.predictors, "{where_}");
            assert_eq!(after.cheap_row_stride, base.cheap_row_stride, "{where_}");
            assert_eq!(after.refine_leaves, base.refine_leaves, "{where_}");
        }
    }

    /// And when asked, each override replaces exactly its own lever.
    #[test]
    fn each_override_replaces_only_its_own_lever() {
        let base = ModularSearchBudget::for_effort(Effort::new(7).expect("7"));
        let after = base.with_overrides(&crate::ModularSearchOverrides {
            max_tree_depth: Some(6),
            cheap_sample_budget: Some(1 << 20),
            ..crate::ModularSearchOverrides::default()
        });
        assert_eq!(after.max_tree_depth, 6);
        assert_eq!(after.cheap_sample_budget, 1 << 20);
        // Untouched levers keep the effort's values.
        assert_eq!(after.max_tree_leaves, base.max_tree_leaves);
        assert_eq!(after.deep_search_sample_cap, base.deep_search_sample_cap);
    }

    /// Degenerate inputs must fall back to the floor rather than inventing a
    /// stride: a zero budget is "unset", not "score nothing".
    #[test]
    fn a_zero_budget_falls_back_to_the_floor_stride() {
        assert_eq!(effective_cheap_stride(0, 4, 4000, 3000, 3), 4);
        assert_eq!(effective_cheap_stride(u64::MAX, 4, 4000, 3000, 3), 4);
        // Never below the floor, whatever the arithmetic.
        assert!(effective_cheap_stride(u64::MAX, 8, 1, 1, 1) >= 8);
    }

    #[test]
    fn default_effort_is_the_lean_level_1() {
        assert_eq!(Effort::default().level(), 1);
        assert_eq!(EncodeOptions::default().effort, Effort::DEFAULT);
        assert_eq!(Effort::DEFAULT, Effort::MIN);
    }

    #[test]
    fn effort_new_rejects_out_of_range() {
        assert!(Effort::new(0).is_err());
        assert!(Effort::new(10).is_err());
        assert!(Effort::new(255).is_err());
        for level in 1..=9u8 {
            assert_eq!(Effort::new(level).expect("valid").level(), level);
        }
    }

    /// A multi-group RGB fixture: gradient plus a hashed speckle so the tree
    /// search has real decisions but no single predictor trivially wins.
    fn effort_fixture() -> crate::Image {
        let (width, height) = (160u32, 160u32);
        let make = |seed: u32| -> Plane {
            (0..height)
                .flat_map(|y| {
                    (0..width).map(move |x| {
                        let h = x
                            .wrapping_mul(2_654_435_761)
                            .wrapping_add(y.wrapping_mul(40_503))
                            .wrapping_add(seed.wrapping_mul(2_246_822_519));
                        (((x + y).wrapping_add(h >> 27)) % 256) as i32
                    })
                })
                .collect()
        };
        crate::Image::new(width, height, 8, vec![make(0), make(1), make(2)]).expect("image")
    }

    /// Options that force the full pre-ramp search (effort 7), for tests that
    /// exercise methods the lean default (effort 1) deliberately skips: the
    /// multi-predictor sweep, greedy splits, per-leaf refinement, Weighted, and
    /// the palette/squeeze transform trials.
    fn full_search_options() -> EncodeOptions {
        EncodeOptions {
            effort: Effort::new(7).expect("valid effort"),
            ..EncodeOptions::default()
        }
    }

    /// Every effort level must plan without panicking on the full (non-collapsed)
    /// deep-search path, and honour its own leaf cap. 48×48 is below every
    /// level's `deep_search_sample_cap`, so this exercises efforts 1-9 through
    /// the multi-split search cheaply.
    #[test]
    fn every_effort_plans_a_valid_tree_on_the_deep_path() {
        let (w, h) = (48u32, 48u32);
        let plane: Plane = (0..h)
            .flat_map(|y| (0..w).map(move |x| ((x * 3 + y * 5) % 200) as i32))
            .collect();
        for level in 1..=9u8 {
            let effort = Effort::new(level).expect("valid");
            let options = EncodeOptions {
                effort,
                ..EncodeOptions::default()
            };
            let plan = plan_for(w, h, std::slice::from_ref(&plane), false, &options)
                .unwrap_or_else(|e| panic!("effort {level} failed to plan: {e}"));
            let contexts = plan.plan().tree.num_contexts();
            let budget = ModularSearchBudget::for_effort(effort);
            assert!(contexts >= 1, "effort {level}: at least one context");
            assert!(
                contexts <= budget.max_tree_leaves.max(1),
                "effort {level}: {contexts} contexts exceeds cap {}",
                budget.max_tree_leaves
            );
        }
    }

    /// The ramp's correctness gate: every effort level reconstructs the source
    /// pixels exactly (lossless is exact at every level) and round-trips
    /// through the in-tree decoder on a multi-group frame; level 7 matches the
    /// default-options encode byte for byte. Speed and size may vary across
    /// levels; pixels may not.
    #[test]
    fn every_effort_round_trips_losslessly_on_a_multi_group_frame() {
        use jpxl_core::limits::Limits;
        let image = effort_fixture();
        let opts = |effort: Effort| EncodeOptions {
            group_size_shift: Some(0), // group_dim 128 -> 160x160 is a 2x2 grid
            resources: crate::EncodeResources::serial(),
            effort,
            ..EncodeOptions::default()
        };
        let mut level_default: Option<Vec<u8>> = None;
        for level in 1..=9u8 {
            let effort = Effort::new(level).expect("valid");
            let bytes = crate::encode(&image, &opts(effort)).expect("encode");
            let decoded = jpxl_decode::decode(&bytes, &Limits::default()).expect("decode");
            assert_eq!(
                decoded.planes.len(),
                image.planes().len(),
                "effort {level}: channel count"
            );
            for (ch, src) in image.planes().iter().enumerate() {
                let got = &decoded.planes.get(ch).expect("plane").samples;
                assert!(
                    got == src,
                    "effort {level} channel {ch}: pixels must be exact"
                );
            }
            if level == 1 {
                level_default = Some(bytes);
            }
        }
        let default = crate::encode(
            &image,
            &EncodeOptions {
                group_size_shift: Some(0),
                resources: crate::EncodeResources::serial(),
                ..EncodeOptions::default()
            },
        )
        .expect("encode");
        assert_eq!(
            level_default.expect("level 1 ran"),
            default,
            "default options must encode identically to explicit effort 1"
        );
    }

    /// Same source + same effort + same options is byte-identical: the search
    /// is a pure function of its budget.
    #[test]
    fn same_effort_is_deterministic() {
        let image = effort_fixture();
        for level in [1u8, 4, 7, 9] {
            let options = EncodeOptions {
                group_size_shift: Some(0),
                resources: crate::EncodeResources::serial(),
                effort: Effort::new(level).expect("valid"),
                ..EncodeOptions::default()
            };
            let a = crate::encode(&image, &options).expect("encode a");
            let b = crate::encode(&image, &options).expect("encode b");
            assert_eq!(
                a, b,
                "effort {level}: repeated encode must be byte-identical"
            );
        }
    }
}
