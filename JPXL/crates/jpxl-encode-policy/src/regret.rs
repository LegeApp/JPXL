//! Phase A of `sources/outside-advice.md` §8's scoping
//! (`jpegxl-rs.work.arch-s8-full-redesign-scoped`): a regret/agreement
//! harness for candidate cover-selection policies ("surrogates"), proven as
//! a no-op against the exact scorer before it is trusted for anything else.
//!
//! This module is read-only measurement infrastructure. It does not touch
//! [`crate::tile_region`] or the production encode path at all — it walks
//! the same quadtree independently, always computing the *unbounded* exact
//! cost of every candidate (never applying `tile_region`'s own Phase-1
//! partial-sum cutoff), because a regret measurement needs true ground-truth
//! costs, not a bound. That duplication is deliberate: it keeps this module
//! from ever being able to perturb what the encoder actually emits, at the
//! cost of one extra exact scoring pass when the harness runs. The harness
//! is not part of any hot path; it is a corpus-driven verification tool.
//!
//! No approximate scorer exists yet (that is Phase C). What lands here is
//! the harness itself, [`ExactPolicy`] as the trivial surrogate that must
//! reproduce zero regret and full agreement, and a cross-check that this
//! module's notion of "exact" matches `tile_region`'s own decisions exactly
//! (so the two implementations cannot silently drift apart).

use crate::error::Result;
use crate::{
    AqSetup, CandidateForwardCache, ForwardScratch, HfQuantizers, NON_DCT8X8_SIGNAL_BITS,
    PER_VARBLOCK_BITS, PreparedFrame, block_cost, mul_signal_bits, square_transform,
};
use jpxl_core::varblock::TransformType;

/// One quadtree node's outcome: keep the four-way split, or merge into the
/// single larger transform available at this node's size (if legal).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoverDecision {
    Split,
    Merge,
}

/// A pluggable cover-selection policy under study. Given the same two
/// numbers [`crate::tile_region`] itself compares (`split_cost`, and
/// `single_cost` when a merge is legal at this node — `None` otherwise),
/// decides split vs merge.
///
/// The harness never trusts a policy's own notion of cost: every decision is
/// re-priced through [`crate::block_cost`] (the exact scorer) before being
/// compared, so regret is always measured in real `bits + weighted_sse`
/// units, never in whatever internal units a future approximate policy might
/// use to reach its decision.
///
/// Returning [`CoverDecision::Merge`] when `single_cost` is `None` is a
/// policy bug (there is no legal merge to choose): the harness treats it as
/// picking [`CoverDecision::Split`] instead, defensively, rather than
/// panicking on a malformed policy under study.
pub trait CoverSurrogate {
    fn decide(&self, split_cost: f64, single_cost: Option<f64>) -> CoverDecision;
}

/// The production policy — [`crate::tile_region`]'s own tie rule — wrapped
/// as a [`CoverSurrogate`] so the harness can be proven against itself
/// first. Ties keep the split (a merge must strictly earn its place), same
/// as the real encoder.
pub struct ExactPolicy;

impl CoverSurrogate for ExactPolicy {
    fn decide(&self, split_cost: f64, single_cost: Option<f64>) -> CoverDecision {
        match single_cost {
            Some(c) if c < split_cost => CoverDecision::Merge,
            _ => CoverDecision::Split,
        }
    }
}

/// One measured node: did the surrogate agree with the true optimum, and if
/// not, its non-negative exact-cost regret (the surrogate's exact cost minus
/// the true optimum's exact cost — always `>= 0`, since the true optimum is
/// the minimum of the two available costs by construction).
#[derive(Debug, Clone, Copy)]
pub struct RegretSample {
    pub agree: bool,
    pub regret: f64,
}

/// An aggregate over many [`RegretSample`]s, reported with an explicit tail
/// (not just a mean): both recorded Phase-2 chroma-from-luma regressions
/// (`jpegxl-rs.work.arch-phase2-cfl-quant`) were rare, systematic wrong
/// picks, not average-case drift, and a mean alone would have hidden them.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RegretSummary {
    pub samples: u64,
    pub agreements: u64,
    pub mean_regret: f64,
    pub max_regret: f64,
    pub p99_regret: f64,
}

impl RegretSummary {
    /// Fraction of nodes where the surrogate picked the true optimum, in
    /// `[0.0, 1.0]`; `1.0` (vacuously) when no samples were recorded.
    #[must_use]
    pub fn agreement_rate(self) -> f64 {
        if self.samples == 0 {
            1.0
        } else {
            #[allow(
                clippy::cast_precision_loss,
                reason = "sample counts stay far inside f64's exact-integer range \
                          for any corpus this harness runs over"
            )]
            let rate = self.agreements as f64 / self.samples as f64;
            rate
        }
    }

    /// Summarizes `samples`, sorting a private copy to find the tail —
    /// `samples` itself is left untouched.
    #[must_use]
    pub fn from_samples(samples: &[RegretSample]) -> Self {
        if samples.is_empty() {
            return Self::default();
        }
        let mut regrets: Vec<f64> = samples.iter().map(|s| s.regret).collect();
        regrets.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        #[allow(
            clippy::cast_precision_loss,
            reason = "sample counts stay far inside f64's exact-integer range"
        )]
        let count = regrets.len() as f64;
        let mean_regret = regrets.iter().sum::<f64>() / count;
        let max_regret = regrets.last().copied().unwrap_or(0.0);
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "p99 index is bounded by regrets.len() - 1, checked via .min() below"
        )]
        let p99_index = ((0.99 * (regrets.len() - 1) as f64).round() as usize)
            .min(regrets.len().saturating_sub(1));
        let p99_regret = regrets.get(p99_index).copied().unwrap_or(max_regret);
        let agreements = samples.iter().filter(|s| s.agree).count();
        Self {
            samples: u64::try_from(samples.len()).unwrap_or(u64::MAX),
            agreements: u64::try_from(agreements).unwrap_or(u64::MAX),
            mean_regret,
            max_regret,
            p99_regret,
        }
    }
}

/// Walks the same quadtree [`crate::tile_region`] walks, computing both the
/// exact split cost (recursively) and the *unbounded* exact merge cost at
/// every internal node — deliberately ignoring `tile_region`'s own Phase-1
/// cutoff, since ground truth for regret must be a real cost, not a bound.
/// Scores `surrogate`'s decision against the true optimum at every node,
/// pushing one [`RegretSample`] per node into `samples`, and returns the
/// true optimal cost of this region (so the parent's `split_cost` is built
/// from true optimal sub-costs, matching what the real encoder would
/// actually produce — this measures each decision's *local* regret against
/// an otherwise-optimal tree, not a compounding whole-cover regret from
/// chaining the surrogate's own picks through the recursion).
///
/// # Errors
///
/// As [`crate::block_cost`].
#[allow(clippy::too_many_arguments)]
#[allow(
    dead_code,
    reason = "Phase A infrastructure: only called from this module's own \
              tests today (proving the harness itself). Phase B/C/D call \
              sites (measuring real corpora, validating a real surrogate) \
              land in later, separately-scoped work; its parameter types \
              (`AqSetup`, `HfQuantizers`, ...) are crate-private, so this \
              cannot be made a public API of the crate without a broader \
              visibility change out of scope for Phase A."
)]
pub(crate) fn measure_region<S: CoverSurrogate>(
    frame: &PreparedFrame,
    hf_quants: &HfQuantizers,
    grid: jpxl_encode::vardct::BlockGrid,
    bx: u32,
    by: u32,
    size: u32,
    aq: &AqSetup,
    x0: u32,
    y0: u32,
    cache: &mut CandidateForwardCache,
    scratch: &mut ForwardScratch,
    d_y_hf: &mut [f32],
    surrogate: &S,
    samples: &mut Vec<RegretSample>,
) -> Result<f64> {
    if bx >= grid.width || by >= grid.height {
        return Ok(0.0);
    }
    if size == 1 {
        let hf_mul = aq.mul_for_footprint(x0 / 8 + bx, y0 / 8 + by, 1, 1);
        let cost = block_cost(
            frame,
            hf_quants,
            TransformType::Dct8x8,
            hf_mul,
            x0 + bx * 8,
            y0 + by * 8,
            cache,
            scratch,
            d_y_hf,
        )? + PER_VARBLOCK_BITS
            + mul_signal_bits(hf_mul, aq.baseline);
        return Ok(cost);
    }

    let half = size / 2;
    let mut split_cost = 0.0f64;
    for (qx, qy) in [
        (bx, by),
        (bx + half, by),
        (bx, by + half),
        (bx + half, by + half),
    ] {
        split_cost += measure_region(
            frame, hf_quants, grid, qx, qy, half, aq, x0, y0, cache, scratch, d_y_hf, surrogate,
            samples,
        )?;
    }

    let fits = bx + size <= grid.width && by + size <= grid.height;
    let single_cost = match square_transform(size) {
        Some(transform) if fits => {
            let hf_mul = aq.mul_for_footprint(x0 / 8 + bx, y0 / 8 + by, size, size);
            let fixed =
                PER_VARBLOCK_BITS + NON_DCT8X8_SIGNAL_BITS + mul_signal_bits(hf_mul, aq.baseline);
            let rd = block_cost(
                frame,
                hf_quants,
                transform,
                hf_mul,
                x0 + bx * 8,
                y0 + by * 8,
                cache,
                scratch,
                d_y_hf,
            )?;
            Some(rd + fixed)
        }
        _ => None,
    };

    let true_decision = match single_cost {
        Some(c) if c < split_cost => CoverDecision::Merge,
        _ => CoverDecision::Split,
    };
    let true_cost = match true_decision {
        CoverDecision::Merge => single_cost.unwrap_or(split_cost),
        CoverDecision::Split => split_cost,
    };

    let surrogate_decision = surrogate.decide(split_cost, single_cost);
    let surrogate_cost = match surrogate_decision {
        CoverDecision::Merge => single_cost.unwrap_or(split_cost),
        CoverDecision::Split => split_cost,
    };

    samples.push(RegretSample {
        agree: surrogate_decision == true_decision,
        regret: (surrogate_cost - true_cost).max(0.0),
    });

    Ok(true_cost)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AnalysisAtlas, EncodeRequest, QuantizerChoice};

    fn ramp_frame(width: u32, height: u32) -> PreparedFrame {
        let n = usize::try_from(width).unwrap_or(0) * usize::try_from(height).unwrap_or(0);
        let mut x_plane = vec![0.0f32; n];
        let mut y_plane = vec![0.0f32; n];
        let mut b_plane = vec![0.0f32; n];
        for row in 0..height {
            for col in 0..width {
                let idx = usize::try_from(row * width + col).unwrap_or(0);
                // A smooth ramp plus a small textured block, so the quadtree
                // actually has real split-vs-merge decisions to make (a flat
                // field degenerates to "always merge," a poor test of the
                // harness's node-by-node comparison).
                let ramp = f32::from(u16::try_from(col % width.max(1)).unwrap_or(0)) / 512.0;
                let texture = if (row / 4 + col / 4) % 2 == 0 {
                    0.05
                } else {
                    -0.05
                };
                let v = (0.3 + ramp + texture).clamp(0.0, 1.0);
                if let (Some(x), Some(y), Some(b)) = (
                    x_plane.get_mut(idx),
                    y_plane.get_mut(idx),
                    b_plane.get_mut(idx),
                ) {
                    *x = 0.0;
                    *y = v;
                    *b = v;
                }
            }
        }
        PreparedFrame::from_linear_srgb(width, height, x_plane, y_plane, b_plane)
            .expect("legal frame")
    }

    /// Runs `measure_region` over every LF group of `frame` with `surrogate`,
    /// returning the accumulated samples.
    fn measure_frame<S: CoverSurrogate>(
        frame: &PreparedFrame,
        surrogate: &S,
    ) -> (Vec<RegretSample>, f64) {
        let request = EncodeRequest::defaults();
        let atlas = AnalysisAtlas::analyze(frame);
        let quantizer = QuantizerChoice::from_request(&request);
        let aq = AqSetup::build(&atlas, &request, quantizer);
        let hf_quants =
            HfQuantizers::new(aq.global_scale.get(), aq.baseline, &aq.muls()).expect("quantizers");
        let decision = jpxl_encode::vardct::FrameDecision {
            width: frame.width(),
            height: frame.height(),
            group_size_shift: crate::VARDCT_GROUP_SIZE_SHIFT,
            num_passes: 1,
        };
        let geometry = decision.geometry().expect("geometry");
        let mut cache = CandidateForwardCache::new();
        let mut scratch = ForwardScratch::new();
        let mut samples = Vec::new();
        let mut total = 0.0f64;
        for index in 0..geometry.num_lf_groups() {
            let id = crate::LfGroupId::new(u32::try_from(index).unwrap_or(u32::MAX));
            let blocks = geometry.lf_group_blocks(id).expect("blocks");
            let rect = geometry.lf_group_rect(id).expect("rect");
            let mut d_y_hf = vec![0.0f32; 32 * 32];
            let mut sby = 0u32;
            while sby < blocks.height {
                let mut sbx = 0u32;
                while sbx < blocks.width {
                    total += measure_region(
                        frame,
                        &hf_quants,
                        blocks,
                        sbx,
                        sby,
                        4,
                        &aq,
                        rect.x0,
                        rect.y0,
                        &mut cache,
                        &mut scratch,
                        &mut d_y_hf,
                        surrogate,
                        &mut samples,
                    )
                    .expect("measures");
                    sbx += 4;
                }
                sby += 4;
            }
        }
        (samples, total)
    }

    /// Phase A's exit gate: the harness proven as a no-op. Wiring the exact
    /// production policy in as its own surrogate must report zero regret and
    /// full agreement at every node — anything else is a bug in the harness
    /// itself, not in a surrogate (there is no approximate surrogate yet).
    #[test]
    fn exact_policy_is_a_true_no_op() {
        let frame = ramp_frame(128, 128);
        let (samples, _total) = measure_frame(&frame, &ExactPolicy);
        assert!(
            !samples.is_empty(),
            "the fixture must exercise real decisions"
        );
        for sample in &samples {
            assert!(sample.agree, "ExactPolicy must always agree with itself");
            assert_eq!(
                sample.regret, 0.0,
                "ExactPolicy must always incur exactly zero regret against itself"
            );
        }
        let summary = RegretSummary::from_samples(&samples);
        assert_eq!(summary.agreement_rate(), 1.0);
        assert_eq!(summary.mean_regret, 0.0);
        assert_eq!(summary.max_regret, 0.0);
        assert_eq!(summary.p99_regret, 0.0);
    }

    /// Cross-validates this module's independent quadtree walk against
    /// `tile_region`'s own production decisions, so the two cannot silently
    /// drift apart: the harness's notion of "exact" must match the real
    /// encoder's, not just agree with itself in isolation.
    #[test]
    fn measure_region_matches_tile_regions_own_total_cost() {
        let frame = ramp_frame(128, 128);
        let request = EncodeRequest::defaults();
        let atlas = AnalysisAtlas::analyze(&frame);
        let quantizer = QuantizerChoice::from_request(&request);
        let aq = AqSetup::build(&atlas, &request, quantizer);
        let hf_quants =
            HfQuantizers::new(aq.global_scale.get(), aq.baseline, &aq.muls()).expect("quantizers");
        let decision = jpxl_encode::vardct::FrameDecision {
            width: frame.width(),
            height: frame.height(),
            group_size_shift: crate::VARDCT_GROUP_SIZE_SHIFT,
            num_passes: 1,
        };
        let geometry = decision.geometry().expect("geometry");
        let mut cache_tr = CandidateForwardCache::new();
        let mut scratch_tr = ForwardScratch::new();
        let mut tile_region_total = 0.0f64;
        for index in 0..geometry.num_lf_groups() {
            let id = crate::LfGroupId::new(u32::try_from(index).unwrap_or(u32::MAX));
            let blocks = geometry.lf_group_blocks(id).expect("blocks");
            let rect = geometry.lf_group_rect(id).expect("rect");
            let mut d_y_hf = vec![0.0f32; 32 * 32];
            let mut sby = 0u32;
            while sby < blocks.height {
                let mut sbx = 0u32;
                while sbx < blocks.width {
                    let (cost, _blocks) = crate::tile_region(
                        &frame,
                        &hf_quants,
                        blocks,
                        sbx,
                        sby,
                        4,
                        &aq,
                        rect.x0,
                        rect.y0,
                        &mut cache_tr,
                        &mut scratch_tr,
                        &mut d_y_hf,
                    )
                    .expect("tile_region");
                    tile_region_total += cost;
                    sbx += 4;
                }
                sby += 4;
            }
        }

        let (_samples, harness_total) = measure_frame(&frame, &ExactPolicy);

        // Not bit-identical: `tile_region`'s cutoff prunes some candidates
        // before their cost fully accumulates, while this harness always
        // computes the unbounded cost — but both must reach the SAME final
        // decision at every node (a cutoff correctly identifies a losing
        // candidate; it never changes which side wins), so the two total
        // costs must match exactly.
        assert!(
            (tile_region_total - harness_total).abs() < 1e-6,
            "tile_region total {tile_region_total} vs harness total {harness_total}: \
             the harness's notion of exact has drifted from production"
        );
    }
}
