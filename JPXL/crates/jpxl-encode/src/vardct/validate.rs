//! The validation layer: every structural invariant a plan must satisfy
//! before it may reach the writer.
//!
//! # Why this exists as a type, not a function call
//!
//! [`ValidatedEmissionPlan`] wraps a private field. `validate` is the only
//! constructor, and the writer accepts nothing else, so "we forgot to
//! validate" is not a reachable state — it is a compile error. That is the
//! milestone-1 exit gate: *malformed plans cannot reach the writer*.
//!
//! # What is checked
//!
//! The invariants are the structural facts the decoder's own parse would
//! enforce, restated on the write side so a bad plan dies at its author rather
//! than in someone else's decoder. They are derived from the clauses, not from
//! `jpxl-decode`: a check copied from the reader would agree with a reader bug.
//!
//! | Group | Invariants |
//! | --- | --- |
//! | Frame (F.2, F.6) | dimensions nonzero, `group_size_shift <= 3`, `num_passes` in `1..=11` |
//! | LF groups (G.2) | count and raster order match the geometry |
//! | Quantizer (I.2, G.2.2) | `global_scale`/`quant_lf` representable (enforced by their newtypes), `extra_precision <= 3`, finite LF dequant weights |
//! | Cover (G.2.4, I.1) | greedy raster origin, LF-group containment, pass-group containment, no overlap, exact cover, `nb_blocks` in range |
//! | Grids (G.2.4) | `XFromY`/`BFromY` on the 64x64 tile grid, `Sharpness` on the block grid with samples `<= 7` |
//! | IR (G.2.2, I.3.2) | one quantized LF group per planned one, LF planes on the block grid, one coefficient set per varblock at `64 * num_blocks` per channel |
//! | Entropy (I.2.2, I.2.6, I.3.1, I.3.3, C.2) | block-context map size and range, preset count, context-map length and cluster bounds, cluster count, hybrid-uint configs, per-pass structure |
//! | Sections (F.3.1) | the layout is exactly the one the geometry implies |
//!
//! # What is *not* checked
//!
//! Nothing here says a plan is a *good* plan. Rate, distortion, and whether
//! the coefficients reconstruct anything resembling the source are policy's
//! business. Validation answers one question: will a conforming decoder be
//! able to parse what this plan describes?

use crate::vardct::error::{PlanError, PlanResult};
use crate::vardct::geometry::VardctGeometry;
use crate::vardct::ids::{LfGroupId, MAX_EXTRA_PRECISION, MAX_SHARPNESS};
use crate::vardct::plan::{
    EmissionPlan, EntropyModelPlan, EntropyPlan, HfBlockContextPlan, LfGroupPlan,
    MAX_BLOCK_CTX_MAP_LEN, MAX_CLUSTERS, MAX_NB_BLOCK_CTX, NUM_CHANNELS, PixelPlan,
    QuantizedFrameIr, QuantizedLfGroup, SectionLayout, SpatialPlan,
};

/// A plan that has passed every structural invariant.
///
/// The only way to build one is [`validate`]. The inner plan is readable but
/// not constructible from outside, so a plan cannot be mutated after
/// validation and re-submitted.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedEmissionPlan(EmissionPlan);

impl ValidatedEmissionPlan {
    /// The plan.
    #[must_use]
    pub const fn plan(&self) -> &EmissionPlan {
        &self.0
    }

    /// The grids the plan was validated against.
    ///
    /// # Errors
    ///
    /// Cannot fail in practice — the geometry was derived during validation —
    /// but the derivation is fallible, so the result is propagated rather than
    /// unwrapped.
    pub fn geometry(&self) -> PlanResult<VardctGeometry> {
        self.0.spatial.frame.geometry()
    }

    /// Unwraps the plan.
    #[must_use]
    pub fn into_inner(self) -> EmissionPlan {
        self.0
    }

    /// The pre-entropy part, already validated, sharing the payloads.
    #[must_use]
    pub fn pixels(&self) -> ValidatedPixelPlan {
        ValidatedPixelPlan(self.0.pixels())
    }
}

/// Checks every structural invariant and admits the plan to the writer.
///
/// # Errors
///
/// The first [`PlanError`] found. Each carries the invariant's name, so a
/// caller — and this module's tests — can tell which rule was broken.
pub fn validate(plan: EmissionPlan) -> PlanResult<ValidatedEmissionPlan> {
    let geometry = plan.spatial.frame.geometry()?;
    validate_spatial(&plan.spatial, &geometry)?;
    validate_quantized(&plan.spatial, &plan.quantized, &geometry)?;
    validate_entropy(&plan.entropy, &geometry)?;
    validate_sections(&plan, &geometry)?;
    Ok(ValidatedEmissionPlan(plan))
}

/// A pre-entropy plan that has passed every non-entropy invariant: frame,
/// LF groups, quantizer, cover, grids and the coefficient IR.
///
/// Built only by [`validate_pixels`] (or taken from an already validated
/// emission plan by [`ValidatedEmissionPlan::pixels`]). It is what an
/// encoder-side renderer accepts, so a probe that needs pixels cannot skip
/// the checks a decoder's parse would enforce.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedPixelPlan(PixelPlan);

impl ValidatedPixelPlan {
    /// The plan.
    #[must_use]
    pub const fn plan(&self) -> &PixelPlan {
        &self.0
    }

    /// The grids the plan was validated against.
    ///
    /// # Errors
    ///
    /// Cannot fail in practice — see [`ValidatedEmissionPlan::geometry`].
    pub fn geometry(&self) -> PlanResult<VardctGeometry> {
        self.0.spatial.frame.geometry()
    }

    /// Unwraps the plan.
    #[must_use]
    pub fn into_inner(self) -> PixelPlan {
        self.0
    }
}

/// Checks every invariant that does not involve entropy or sections.
///
/// # Errors
///
/// The first [`PlanError`] found, as [`validate`].
pub fn validate_pixels(plan: PixelPlan) -> PlanResult<ValidatedPixelPlan> {
    let geometry = plan.spatial.frame.geometry()?;
    validate_spatial(&plan.spatial, &geometry)?;
    validate_quantized(&plan.spatial, &plan.quantized, &geometry)?;
    Ok(ValidatedPixelPlan(plan))
}

/// Attaches entropy models and a section layout to validated pixels and
/// checks the two remaining invariant groups.
///
/// # Errors
///
/// The first entropy or section [`PlanError`] found.
pub fn attach_and_validate_entropy(
    pixels: ValidatedPixelPlan,
    entropy: EntropyPlan,
    sections: SectionLayout,
) -> PlanResult<ValidatedEmissionPlan> {
    let geometry = pixels.geometry()?;
    validate_entropy(&entropy, &geometry)?;
    let plan = EmissionPlan::from_pixels(&pixels.0, entropy, sections);
    validate_sections(&plan, &geometry)?;
    Ok(ValidatedEmissionPlan(plan))
}

// ---------------------------------------------------------------------------
// Spatial
// ---------------------------------------------------------------------------

fn validate_spatial(spatial: &SpatialPlan, geometry: &VardctGeometry) -> PlanResult<()> {
    if spatial.lf.extra_precision > MAX_EXTRA_PRECISION {
        return Err(PlanError::out_of_range(
            "extra_precision",
            "G.2.2",
            i64::from(spatial.lf.extra_precision),
        ));
    }
    for weight in spatial.lf.channel_dequant {
        if !weight.is_finite() || weight == 0.0 {
            return Err(PlanError::out_of_range(
                "LF channel dequantization weight",
                "G.1.2",
                0,
            ));
        }
    }
    if spatial.restoration.epf_iters > 3 {
        return Err(PlanError::out_of_range(
            "epf_iters",
            "J.1",
            i64::from(spatial.restoration.epf_iters),
        ));
    }
    if spatial.lf.correlation.colour_factor == 0 {
        return Err(PlanError::out_of_range("colour_factor", "I.2.3", 0));
    }

    let expected = geometry.num_lf_groups();
    if spatial.lf_groups.len() as u64 != expected {
        return Err(PlanError::shape(
            "LF group count",
            "G.2",
            expected,
            spatial.lf_groups.len() as u64,
        ));
    }
    for (index, group) in spatial.lf_groups.iter().enumerate() {
        let expected_id = u32::try_from(index).unwrap_or(u32::MAX);
        if group.id.get() != expected_id {
            return Err(PlanError::shape(
                "LF group raster order",
                "G.2",
                u64::from(expected_id),
                group.id.index(),
            ));
        }
        validate_lf_group(group, geometry)?;
    }
    Ok(())
}

/// The exact-cover replay of G.2.4's greedy placement walk.
///
/// This is the invariant the whole IR is shaped around: a plan's varblock list
/// is the `BlockInfo` column sequence, so replaying the decoder's placement
/// rule over it either reproduces the planner's own origins or the plan is
/// describing a tiling the decoder will not build.
fn validate_lf_group(group: &LfGroupPlan, geometry: &VardctGeometry) -> PlanResult<()> {
    let id = group.id;
    let grid = geometry.lf_group_blocks(id).ok_or_else(|| {
        PlanError::shape(
            "LF group index",
            "G.2",
            geometry.num_lf_groups(),
            id.index(),
        )
    })?;
    let total = grid.area();
    let total_usize = usize::try_from(total)
        .map_err(|_| PlanError::out_of_range("LF group block count", "G.2.4", 0))?;

    if group.blocks.is_empty() && total > 0 {
        return Err(PlanError::shape("nb_blocks", "G.2.4", 1, 0));
    }
    if group.nb_blocks() > total {
        return Err(PlanError::shape(
            "nb_blocks",
            "G.2.4",
            total,
            group.nb_blocks(),
        ));
    }

    // The pass-group ("HF group") side in blocks. A varblock's coefficients
    // are carried by exactly one G.4 `PassGroup` section, so a varblock that
    // straddled two of them would have no section to live in.
    let pass_blocks = geometry.group_blocks();

    let mut covered = vec![false; total_usize];
    let mut cursor: u64 = 0;
    for block in &group.blocks {
        while cursor < total
            && usize::try_from(cursor)
                .ok()
                .and_then(|i| covered.get(i))
                .copied()
                .unwrap_or(false)
        {
            cursor += 1;
        }
        if cursor >= total {
            return Err(PlanError::Cover {
                what: "varblock past a fully covered LF group",
                clause: "G.2.4",
                lf_group: id.get(),
                block: (0, 0),
            });
        }
        let bx = u32::try_from(cursor % u64::from(grid.width)).unwrap_or(u32::MAX);
        let by = u32::try_from(cursor / u64::from(grid.width)).unwrap_or(u32::MAX);

        if block.origin.bx() != bx || block.origin.by() != by {
            return Err(PlanError::Cover {
                what: "varblock origin is not the earliest uncovered block",
                clause: "G.2.4",
                lf_group: id.get(),
                block: (block.origin.bx(), block.origin.by()),
            });
        }

        let (rows, cols) = block.transform.block_dims();
        let rows = u32::try_from(rows).unwrap_or(u32::MAX);
        let cols = u32::try_from(cols).unwrap_or(u32::MAX);
        let (right, bottom) = match (bx.checked_add(cols), by.checked_add(rows)) {
            (Some(r), Some(b)) => (r, b),
            _ => (u32::MAX, u32::MAX),
        };
        if right > grid.width || bottom > grid.height {
            return Err(PlanError::Cover {
                what: "varblock crosses the LF-group edge",
                clause: "G.2.4",
                lf_group: id.get(),
                block: (bx, by),
            });
        }
        if pass_blocks != 0
            && (bx / pass_blocks != (right - 1) / pass_blocks
                || by / pass_blocks != (bottom - 1) / pass_blocks)
        {
            return Err(PlanError::Cover {
                what: "varblock crosses a pass-group edge",
                clause: "G.4",
                lf_group: id.get(),
                block: (bx, by),
            });
        }

        for dy in 0..rows {
            for dx in 0..cols {
                let idx = u64::from(by + dy) * u64::from(grid.width) + u64::from(bx + dx);
                let slot = usize::try_from(idx)
                    .ok()
                    .and_then(|i| covered.get_mut(i))
                    .ok_or(PlanError::Cover {
                        what: "varblock crosses the LF-group edge",
                        clause: "G.2.4",
                        lf_group: id.get(),
                        block: (bx, by),
                    })?;
                if *slot {
                    return Err(PlanError::Cover {
                        what: "varblock overlaps an already-placed varblock",
                        clause: "G.2.4",
                        lf_group: id.get(),
                        block: (bx + dx, by + dy),
                    });
                }
                *slot = true;
            }
        }
    }

    if let Some(index) = covered.iter().position(|&c| !c) {
        let index = u64::try_from(index).unwrap_or(0);
        let bx = u32::try_from(index % u64::from(grid.width)).unwrap_or(0);
        let by = u32::try_from(index / u64::from(grid.width)).unwrap_or(0);
        return Err(PlanError::Cover {
            what: "LF group left uncovered",
            clause: "G.2.4",
            lf_group: id.get(),
            block: (bx, by),
        });
    }

    // The metadata grids of the same LF group.
    let tiles = geometry
        .lf_group_cfl_tiles(id)
        .ok_or_else(|| PlanError::shape("LF group index", "G.2", 0, id.index()))?;
    if group.cfl.tiles() != tiles {
        return Err(PlanError::shape(
            "CfL grid shape",
            "G.2.4",
            tiles.area(),
            group.cfl.tiles().area(),
        ));
    }
    if group.sharpness.blocks() != grid {
        return Err(PlanError::shape(
            "Sharpness grid shape",
            "G.2.4",
            grid.area(),
            group.sharpness.blocks().area(),
        ));
    }
    if let Some(&bad) = group
        .sharpness
        .values()
        .iter()
        .find(|&&v| v > MAX_SHARPNESS)
    {
        return Err(PlanError::out_of_range(
            "Sharpness sample",
            "J.4.3",
            i64::from(bad),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Quantized IR
// ---------------------------------------------------------------------------

fn validate_quantized(
    spatial: &SpatialPlan,
    quantized: &QuantizedFrameIr,
    geometry: &VardctGeometry,
) -> PlanResult<()> {
    let planned = spatial.lf_groups.len() as u64;
    let ir = quantized.lf_groups.len() as u64;
    if planned != ir {
        return Err(PlanError::shape(
            "quantized LF group count",
            "G.2",
            planned,
            ir,
        ));
    }
    for (spatial, quantized) in spatial.lf_groups.iter().zip(&quantized.lf_groups) {
        validate_quantized_group(spatial, quantized, geometry)?;
    }
    Ok(())
}

fn validate_quantized_group(
    spatial: &LfGroupPlan,
    quantized: &QuantizedLfGroup,
    geometry: &VardctGeometry,
) -> PlanResult<()> {
    if spatial.id != quantized.id {
        return Err(PlanError::shape(
            "quantized LF group identity",
            "G.2",
            spatial.id.index(),
            quantized.id.index(),
        ));
    }
    let grid = geometry
        .lf_group_blocks(spatial.id)
        .ok_or_else(|| PlanError::shape("LF group index", "G.2", 0, spatial.id.index()))?;
    if quantized.lf.blocks() != grid {
        return Err(PlanError::shape(
            "LfQuant plane shape",
            "G.2.2",
            grid.area(),
            quantized.lf.blocks().area(),
        ));
    }
    if quantized.coefficients.len() != spatial.blocks.len() {
        return Err(PlanError::shape(
            "coefficient set count",
            "I.4",
            spatial.blocks.len() as u64,
            quantized.coefficients.len() as u64,
        ));
    }
    for (block, coeffs) in spatial.blocks.iter().zip(&quantized.coefficients) {
        let expected = (block.transform.num_blocks() * 64) as u64;
        for c in 0..NUM_CHANNELS {
            let found = coeffs.channel(c).map_or(0, <[i32]>::len) as u64;
            if found != expected {
                return Err(PlanError::shape(
                    "varblock coefficient count",
                    "I.3.2",
                    expected,
                    found,
                ));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Entropy
// ---------------------------------------------------------------------------

fn validate_entropy(entropy: &EntropyPlan, geometry: &VardctGeometry) -> PlanResult<()> {
    match &entropy.block_context {
        HfBlockContextPlan::Default => {}
        HfBlockContextPlan::Custom {
            lf_thresholds,
            qf_thresholds,
            map,
        } => {
            let bsize = 39u64
                .saturating_mul(qf_thresholds.len() as u64 + 1)
                .saturating_mul(
                    lf_thresholds
                        .iter()
                        .fold(1u64, |acc, t| acc.saturating_mul(t.len() as u64 + 1)),
                );
            if map.len() as u64 != bsize {
                return Err(PlanError::shape(
                    "block_ctx_map length",
                    "I.2.2",
                    bsize,
                    map.len() as u64,
                ));
            }
            if map.len() > MAX_BLOCK_CTX_MAP_LEN {
                return Err(PlanError::out_of_range(
                    "bsize",
                    "I.2.2",
                    i64::try_from(map.len()).unwrap_or(i64::MAX),
                ));
            }
        }
    }
    let nb_block_ctx = entropy.block_context.nb_block_ctx();
    if nb_block_ctx == 0 || nb_block_ctx > MAX_NB_BLOCK_CTX {
        return Err(PlanError::out_of_range(
            "nb_block_ctx",
            "I.2.2",
            i64::try_from(nb_block_ctx).unwrap_or(i64::MAX),
        ));
    }
    // The map must be dense in `0..nb_block_ctx`: C.2.2's clustering has no
    // representation for an index nothing uses. (No entry can be *above* the
    // range — `nb_block_ctx` is defined as one past the map's maximum — so
    // density is the only reachable failure.)
    let mut seen = vec![false; usize::try_from(nb_block_ctx).unwrap_or(0)];
    for &entry in entropy.block_context.map() {
        if let Some(slot) = seen.get_mut(usize::from(entry)) {
            *slot = true;
        }
    }
    if let Some(missing) = seen.iter().position(|&s| !s) {
        return Err(PlanError::out_of_range(
            "block_ctx_map is not dense",
            "C.2.2",
            i64::try_from(missing).unwrap_or(i64::MAX),
        ));
    }

    if entropy.num_hf_presets == 0 || u64::from(entropy.num_hf_presets) > geometry.num_groups() {
        return Err(PlanError::out_of_range(
            "num_hf_presets",
            "I.2.6",
            i64::from(entropy.num_hf_presets),
        ));
    }

    let expected_passes = u64::from(geometry.num_passes());
    if entropy.passes.len() as u64 != expected_passes {
        return Err(PlanError::shape(
            "entropy pass count",
            "F.6",
            expected_passes,
            entropy.passes.len() as u64,
        ));
    }

    let pre_contexts = entropy.pre_contexts();
    for pass in &entropy.passes {
        validate_model(&pass.distributions, pre_contexts)?;
        if pass.group_presets.len() as u64 != geometry.num_groups() {
            return Err(PlanError::shape(
                "HF preset assignment count",
                "I.4",
                geometry.num_groups(),
                pass.group_presets.len() as u64,
            ));
        }
        if let Some(bad) = pass
            .group_presets
            .iter()
            .find(|p| p.get() >= entropy.num_hf_presets)
        {
            return Err(PlanError::out_of_range("hfp", "I.4", i64::from(bad.get())));
        }
        for order in pass.orders.overrides() {
            if usize::from(order.order_id.get()) >= jpxl_core::varblock::NUM_ORDER_IDS {
                return Err(PlanError::out_of_range(
                    "used_orders bit",
                    "I.3.1",
                    i64::from(order.order_id.get()),
                ));
            }
        }
    }
    Ok(())
}

fn validate_model(model: &EntropyModelPlan, pre_contexts: u64) -> PlanResult<()> {
    if model.context_map.len() as u64 != pre_contexts {
        return Err(PlanError::shape(
            "context map length",
            "I.3.3",
            pre_contexts,
            model.context_map.len() as u64,
        ));
    }
    if model.histograms.is_empty() || model.histograms.len() > MAX_CLUSTERS {
        return Err(PlanError::out_of_range(
            "cluster count",
            "C.2.2",
            i64::try_from(model.histograms.len()).unwrap_or(i64::MAX),
        ));
    }
    if model.hybrid_uint.len() != model.histograms.len() {
        return Err(PlanError::shape(
            "HybridUintConfig count",
            "C.2.3",
            model.histograms.len() as u64,
            model.hybrid_uint.len() as u64,
        ));
    }
    let clusters = model.histograms.len();
    let mut used = vec![false; clusters];
    for cluster in &model.context_map {
        match used.get_mut(usize::from(cluster.get())) {
            Some(slot) => *slot = true,
            None => {
                return Err(PlanError::out_of_range(
                    "context map entry",
                    "C.2.2",
                    i64::from(cluster.get()),
                ));
            }
        }
    }
    if let Some(missing) = used.iter().position(|&u| !u) {
        return Err(PlanError::out_of_range(
            "context map is not dense",
            "C.2.2",
            i64::try_from(missing).unwrap_or(i64::MAX),
        ));
    }
    for config in &model.hybrid_uint {
        let split = u32::from(config.split_exponent);
        if split > 15 || u32::from(config.msb_in_token) + u32::from(config.lsb_in_token) > split {
            return Err(PlanError::out_of_range(
                "HybridUintConfig",
                "C.2.3",
                i64::from(split),
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Sections
// ---------------------------------------------------------------------------

fn validate_sections(plan: &EmissionPlan, geometry: &VardctGeometry) -> PlanResult<()> {
    let expected = geometry.section_layout();
    if plan.sections.kinds.len() != expected.len() {
        return Err(PlanError::shape(
            "section count",
            "F.3.1",
            expected.len() as u64,
            plan.sections.kinds.len() as u64,
        ));
    }
    for (index, (found, want)) in plan.sections.kinds.iter().zip(&expected).enumerate() {
        if found != want {
            return Err(PlanError::shape(
                "section order",
                "F.3.1",
                index as u64,
                index as u64,
            ));
        }
    }
    Ok(())
}

/// Convenience: an LF group id from a raster index.
#[must_use]
pub fn lf_group_id(index: usize) -> LfGroupId {
    LfGroupId::new(u32::try_from(index).unwrap_or(u32::MAX))
}
