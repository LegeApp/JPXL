//! A human-readable dump of a validated plan.
//!
//! The milestone-1 exit gate is "a hand-built legal plan validates **and
//! dumps**". The dump is not decoration: a plan is a few thousand small
//! integers, and the first thing anyone debugging a wrong bitstream needs is
//! to see what the encoder *intended* before looking at what it wrote. It is
//! deliberately line-oriented and stable, so two plans can be diffed.
//!
//! Only a [`ValidatedEmissionPlan`] can be dumped — the same gate the writer
//! uses — so a dump never shows a shape that could not be emitted.

use core::fmt::Write as _;

use crate::vardct::error::PlanResult;
use crate::vardct::plan::HfBlockContextPlan;
use crate::vardct::validate::ValidatedEmissionPlan;

/// Renders `plan` as a stable, diffable text dump.
///
/// # Errors
///
/// [`PlanError`](crate::vardct::PlanError) if the plan's geometry cannot be
/// re-derived, which validation already proved it can.
pub fn dump(plan: &ValidatedEmissionPlan) -> PlanResult<String> {
    let geometry = plan.geometry()?;
    let plan = plan.plan();
    let spatial = &plan.spatial;
    let mut out = String::new();

    let _ = writeln!(
        out,
        "frame {}x{} group_dim {} lf_group_dim {} passes {}",
        spatial.frame.width,
        spatial.frame.height,
        geometry.group_dim(),
        geometry.lf_group_dim(),
        spatial.frame.num_passes
    );
    let _ = writeln!(
        out,
        "grids: {} blocks ({}x{}), {} lf groups, {} pass groups",
        geometry.frame_blocks().area(),
        geometry.frame_blocks().width,
        geometry.frame_blocks().height,
        geometry.num_lf_groups(),
        geometry.num_groups()
    );
    let _ = writeln!(
        out,
        "quantizer: global_scale {} quant_lf {}",
        spatial.quantizer.global_scale.get(),
        spatial.quantizer.quant_lf.get()
    );
    let _ = writeln!(
        out,
        "lf: extra_precision {} smoothing {} dequant {:?}",
        spatial.lf.extra_precision, spatial.lf.adaptive_smoothing, spatial.lf.channel_dequant
    );
    let _ = writeln!(
        out,
        "lf correlation: colour_factor {} base [{}, {}] lf factors [{}, {}]",
        spatial.lf.correlation.colour_factor,
        spatial.lf.correlation.base_correlation_x,
        spatial.lf.correlation.base_correlation_b,
        spatial.lf.correlation.x_factor_lf,
        spatial.lf.correlation.b_factor_lf
    );
    let _ = writeln!(
        out,
        "restoration: gaborish {} epf_iters {}",
        spatial.restoration.gaborish, spatial.restoration.epf_iters
    );

    for group in &spatial.lf_groups {
        let blocks = geometry.lf_group_blocks(group.id);
        let (bw, bh) = blocks.map_or((0, 0), |g| (g.width, g.height));
        let _ = writeln!(
            out,
            "lf group {}: {}x{} blocks, nb_blocks {}, cfl tiles {}x{}",
            group.id.get(),
            bw,
            bh,
            group.nb_blocks(),
            group.cfl.tiles().width,
            group.cfl.tiles().height
        );
        let mut histogram = [0u32; 27];
        for block in &group.blocks {
            if let Some(slot) = histogram.get_mut(usize::from(block.transform.dct_select())) {
                *slot += 1;
            }
        }
        for (select, count) in histogram.iter().enumerate().filter(|&(_, &c)| c > 0) {
            let _ = writeln!(out, "  DctSelect {select}: {count}");
        }
        let mut mul_lo = u32::MAX;
        let mut mul_hi = 0u32;
        for block in &group.blocks {
            mul_lo = mul_lo.min(block.hf_mul.get());
            mul_hi = mul_hi.max(block.hf_mul.get());
        }
        if !group.blocks.is_empty() {
            let _ = writeln!(out, "  HfMul range: {mul_lo}..={mul_hi}");
        }
    }

    let nb_block_ctx = plan.entropy.block_context.nb_block_ctx();
    let context_kind = match plan.entropy.block_context {
        HfBlockContextPlan::Default => "default",
        HfBlockContextPlan::Custom { .. } => "custom",
    };
    let _ = writeln!(
        out,
        "entropy: block_ctx {context_kind} nb_block_ctx {nb_block_ctx} presets {} pre_contexts {}",
        plan.entropy.num_hf_presets,
        plan.entropy.pre_contexts()
    );
    for (index, pass) in plan.entropy.passes.iter().enumerate() {
        let _ = writeln!(
            out,
            "  pass {index}: used_orders {:#x} clusters {} context map {}",
            pass.orders.used_orders(),
            pass.distributions.histograms.len(),
            pass.distributions.context_map.len()
        );
    }

    let _ = writeln!(out, "sections: {}", plan.sections.kinds.len());
    for (index, kind) in plan.sections.kinds.iter().enumerate() {
        let _ = writeln!(out, "  {index}: {kind:?}");
    }
    Ok(out)
}
