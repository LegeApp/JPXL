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

/// Table H.3 predictors this track will consider (slice 19).
///
/// Weighted (6) needs the H.5 self-correcting state machine and is deferred.
const PREDICTOR_CANDIDATES: &[Predictor] = &[
    Predictor::Zero,
    Predictor::West,
    Predictor::North,
    Predictor::AverageWestNorth,
    Predictor::Select,
    Predictor::Gradient,
];

/// Property indices tried for splits (Table H.4 static rows).
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
/// Policy (slice 19d):
/// 1. Score each predictor as a single-leaf tree (residual + tree bits).
/// 2. Greedy recursive splitting: at each leaf, try property×threshold with
///    the parent predictor on both children; adopt strict total-cost wins.
/// 3. Per-leaf predictor refinement over the settled topology.
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

    for &predictor in PREDICTOR_CANDIDATES {
        let tree = MaTree::single_leaf(predictor);
        let cost = total_cost(width, height, score_planes, &tree, group_size_shift)?;
        if best.as_ref().is_none_or(|(c, _)| cost < *c) {
            best = Some((cost, tree));
        }
    }

    let (mut best_cost, mut best_tree) =
        best.unwrap_or_else(|| (u64::MAX, MaTree::single_leaf(Predictor::Gradient)));

    // Greedy splits: repeatedly try to split every current leaf.
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
                    let cost =
                        total_cost(width, height, score_planes, &candidate, group_size_shift)?;
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

    // Per-leaf predictor refinement on the settled topology.
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
            let cost = total_cost(width, height, score_planes, &candidate, group_size_shift)?;
            if cost < best_cost {
                best_cost = cost;
                best_tree = candidate;
            }
        }
    }

    // Nested palette trial on *source* samples (not RCT). Single-section only
    // in wave 1; gray or RGB exact-colour. Adopts if strictly cheaper.
    let mut palette: Option<PaletteForward> = None;
    let mut use_rct = rct;
    let mut use_squeeze = false;
    // Palette / squeeze trials (single- and multi-section). Exact-price.
    let num_c = planes.len();
    if (num_c == 1 || num_c == 3)
        && let Some(fwd) = try_exact_palette(width, height, planes, 0, num_c)?
    {
        let tree = MaTree::single_leaf(Predictor::Zero);
        let cost = total_cost_source(
            &modular::ModularSource::from_palette(fwd.clone(), tree.clone(), false),
            group_size_shift,
        )?;
        if cost < best_cost {
            best_cost = cost;
            best_tree = tree;
            palette = Some(fwd);
            use_rct = false;
        }
    }

    // Default squeeze on MA-scored planes. Skip when palette won.
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
        let cost = total_cost_source(&source, group_size_shift)?;
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

/// Residual-section bits plus measured MA tree bits for direct planes.
fn total_cost(
    width: u32,
    height: u32,
    planes: &[Plane],
    tree: &MaTree,
    group_size_shift: u32,
) -> Result<u64> {
    let source = modular::ModularSource::direct(width, height, planes, false, tree.clone(), false);
    total_cost_source(&source, group_size_shift)
}

fn total_cost_source(source: &modular::ModularSource, group_size_shift: u32) -> Result<u64> {
    let residual = residual_stream_cost_source(source, group_size_shift)?;
    let tree_bits = modular::ma_tree_bit_cost(&source.tree)?;
    let geometry = crate::frame::Geometry::new(source.width, source.height, group_size_shift)?;
    let sections = if geometry.is_single_section() {
        1u64
    } else {
        geometry.num_groups()
    };
    Ok(residual.saturating_add(tree_bits.saturating_mul(sections)))
}

/// Exact residual-section bit cost under ANS for a prepared source.
fn residual_stream_cost_source(
    source: &modular::ModularSource,
    group_size_shift: u32,
) -> Result<u64> {
    use jpxl_bitstream::BitWriter;

    let geometry = crate::frame::Geometry::new(source.width, source.height, group_size_shift)?;
    let mut bits = 0u64;
    if geometry.is_single_section() {
        let mut w = BitWriter::new();
        modular::write_residual_payload_full(&mut w, source)?;
        bits = bits.saturating_add(w.bit_len());
    } else {
        let part = modular::partition_channels(source, geometry.group_dim());
        // Approximate multi-section residual cost: LfGlobal bands + one full
        // pass over all pass-group tiles (same as emission).
        if !part.lf_global.is_empty() {
            let mut w = BitWriter::new();
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
            let mut w = BitWriter::new();
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
            let mut w = BitWriter::new();
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

    #[test]
    fn plan_for_adopts_palette_on_scattered_few_colours() {
        // Few unique levels but high spatial frequency → predictors lose;
        // exact-price palette should win.
        let width = 48u32;
        let height = 48u32;
        let levels = [10i32, 80, 160, 240];
        let plane: Plane = (0..height)
            .flat_map(|y| {
                (0..width)
                    .map(move |x| levels[((x.wrapping_mul(3) + y.wrapping_mul(7)) % 4) as usize])
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
}
