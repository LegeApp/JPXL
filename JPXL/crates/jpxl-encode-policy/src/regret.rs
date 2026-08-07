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
//! Phase A landed the harness itself, [`ExactPolicy`] as the trivial
//! surrogate that must reproduce zero regret and full agreement, and a
//! cross-check that this module's notion of "exact" matches `tile_region`'s
//! own decisions exactly (so the two implementations cannot silently drift
//! apart).
//!
//! Phase C adds the first real approximate primitive under study: a
//! provable, staged lower bound
//! ([`crate::quantize::HfQuantizer::cell_lower_bound`]) checked at Y, then
//! X, then B — the same order and running-total units
//! `block_cost_bounded` uses in production, so what is validated here is
//! the integration Phase D would actually wire in, not an idealization of
//! it. [`validate_candidate_prune`] and [`measure_prune_safety`] are this
//! module's second independent quadtree walk (alongside [`measure_region`]),
//! read-only in the same sense: it never touches `tile_region` or
//! `block_cost_bounded`, only re-derives the same decisions to check the
//! bound's safety property (`PruneSummary::safety_violations` must be `0`)
//! and record its usefulness (`PruneSummary::prune_rate`).

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

/// S8 Phase C (`jpegxl-rs.work.arch-s8-full-redesign-scoped`): whether the
/// cheap, staged summary-based prune (`HfQuantizer::cell_lower_bound`,
/// checked at Y, then X, then B — exactly where a real integration would
/// check it, never before) would prune this one candidate, and its true
/// exact cost, so a caller can assert the safety property: `would_prune`
/// must never fire when `exact_cost < cutoff` (the candidate actually
/// wins).
#[derive(Debug, Clone, Copy)]
pub struct PruneSample {
    pub would_prune: bool,
    pub exact_cost: f64,
    pub cutoff: f64,
}

/// Aggregate over many [`PruneSample`]s.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PruneSummary {
    pub candidates: u64,
    /// Must be `0` for the prune to be trustworthy — any nonzero value means
    /// the bound pruned a candidate that would actually have won, a bug in
    /// [`crate::quantize::HfQuantizer::cell_lower_bound`] or in how it is
    /// staged here, not a tolerance to relax.
    pub safety_violations: u64,
    /// How often the bound actually prunes — the "usefulness" half of
    /// Phase C's exit gate, independent of safety.
    pub pruned: u64,
}

impl PruneSummary {
    /// Fraction of candidates the bound would prune, in `[0.0, 1.0]`.
    #[must_use]
    pub fn prune_rate(self) -> f64 {
        if self.candidates == 0 {
            0.0
        } else {
            #[allow(
                clippy::cast_precision_loss,
                reason = "candidate counts stay far inside f64's exact-integer range"
            )]
            let rate = self.pruned as f64 / self.candidates as f64;
            rate
        }
    }

    #[must_use]
    pub fn from_samples(samples: &[PruneSample]) -> Self {
        let mut violations = 0u64;
        let mut pruned = 0u64;
        for s in samples {
            if s.would_prune {
                pruned += 1;
                // Ties are fine (the real algorithm's `check` also treats
                // `>=` as a correct prune); only a candidate that would have
                // *strictly* won is a violation.
                if s.exact_cost < s.cutoff - 1e-6 {
                    violations += 1;
                }
            }
        }
        Self {
            candidates: u64::try_from(samples.len()).unwrap_or(u64::MAX),
            safety_violations: violations,
            pruned,
        }
    }
}

/// Computes the staged cheap-bound prune check for **one** candidate's
/// already-cached forward coefficients, mirroring `block_cost_bounded`'s
/// exact Y-then-X-then-B running-total order and units (`bits: u64` summed
/// across channels, `weighted_sse: f64` summed as `lambda[channel] *
/// to_sample_domain * (recon - target)^2`) — so this validates the *same*
/// integration Phase D would wire in, without touching `block_cost_bounded`
/// itself. X's target needs no correction (`kX == 0.0`, I.6); B's does
/// (`kB == 1.0`), and only after Y's *exact* reconstruction exists — this
/// function computes Y's exact quantization first (as the real algorithm
/// always does regardless of X/B's outcome) specifically to make B's bound
/// check use the real coupling, not an approximation of it.
///
/// # Errors
///
/// As [`crate::quantize::HfQuantizer::choose`] /
/// [`crate::quantize::HfQuantizer::cell_lower_bound`].
#[allow(
    dead_code,
    reason = "Phase C infrastructure: only called from this module's own \
              tests today (validating the bound's safety exhaustively). \
              Phase D's wiring is separately-scoped work."
)]
fn validate_candidate_prune(
    hf_quant: &crate::HfQuantizer,
    lambda: &[f64; crate::NUM_CHANNELS],
    to_sample_domain: f64,
    side: usize,
    n: usize,
    fwd: &crate::VarblockForward,
    cutoff: f64,
) -> Result<PruneSample> {
    let cells = side * side;
    let cy = fwd.coeffs.get(1).map_or(&[][..], Vec::as_slice);
    let cx = fwd.coeffs.get(0).map_or(&[][..], Vec::as_slice);
    let cb = fwd.coeffs.get(2).map_or(&[][..], Vec::as_slice);

    let mut bits = 0u64;
    let mut weighted_sse = 0.0f64;
    let mut would_prune = false;
    let check = |bits: u64, weighted_sse: f64| -> bool {
        #[allow(
            clippy::cast_precision_loss,
            reason = "bit counts stay far inside f64's exact integer range"
        )]
        let partial = bits as f64 + weighted_sse;
        partial >= cutoff
    };

    // --- Y: cheap bound first, then the exact scoring the real algorithm
    // always runs (regardless of X/B's fate) to get d_y_hf for B.
    let mut lb_bits = 0u64;
    let mut lb_sse = 0.0f64;
    let mut d_y_hf = vec![0.0f32; cells];
    for cell in 0..cells {
        if crate::is_llf_cell(cell, side, n) {
            continue;
        }
        let target = cy.get(cell).copied().unwrap_or(0.0);
        let (b, s) = hf_quant.cell_lower_bound(target, 1, cell)?;
        lb_bits = lb_bits.saturating_add(b);
        lb_sse += lambda[1] * to_sample_domain * s;
    }
    if check(bits.saturating_add(lb_bits), weighted_sse + lb_sse) {
        would_prune = true;
    }
    for cell in 0..cells {
        if crate::is_llf_cell(cell, side, n) {
            continue;
        }
        let target = cy.get(cell).copied().unwrap_or(0.0);
        let q = hf_quant.choose(target, 1, cell)?;
        let recon = hf_quant.reconstruct(q, 1, cell);
        bits = bits.saturating_add(crate::residual_bits(q));
        weighted_sse += lambda[1] * to_sample_domain * f64::from(recon - target).powi(2);
        if let Some(slot) = d_y_hf.get_mut(cell) {
            *slot = recon;
        }
    }

    // --- X and B ---
    for &(channel, plane) in &[(0usize, cx), (2usize, cb)] {
        let k = if channel == 0 { 0.0 } else { 1.0 };
        let mut lb_bits = 0u64;
        let mut lb_sse = 0.0f64;
        for cell in 0..cells {
            if crate::is_llf_cell(cell, side, n) {
                continue;
            }
            let target = plane.get(cell).copied().unwrap_or(0.0)
                - k * d_y_hf.get(cell).copied().unwrap_or(0.0);
            let (b, s) = hf_quant.cell_lower_bound(target, channel, cell)?;
            lb_bits = lb_bits.saturating_add(b);
            lb_sse += lambda.get(channel).copied().unwrap_or(0.0) * to_sample_domain * s;
        }
        if check(bits.saturating_add(lb_bits), weighted_sse + lb_sse) {
            would_prune = true;
        }
        for cell in 0..cells {
            if crate::is_llf_cell(cell, side, n) {
                continue;
            }
            let target = plane.get(cell).copied().unwrap_or(0.0)
                - k * d_y_hf.get(cell).copied().unwrap_or(0.0);
            let q = hf_quant.choose(target, channel, cell)?;
            let recon = hf_quant.reconstruct(q, channel, cell);
            bits = bits.saturating_add(crate::residual_bits(q));
            weighted_sse += lambda.get(channel).copied().unwrap_or(0.0)
                * to_sample_domain
                * f64::from(recon - target).powi(2);
        }
    }

    #[allow(
        clippy::cast_precision_loss,
        reason = "bit counts stay far inside f64's exact integer range"
    )]
    let exact_cost = bits as f64 + weighted_sse;
    Ok(PruneSample {
        would_prune,
        exact_cost,
        cutoff,
    })
}

/// Walks the same quadtree [`measure_region`] walks, calling
/// [`validate_candidate_prune`] at every merge candidate with the *same*
/// `cutoff` `tile_region` would actually use (`split_cost - fixed`) —
/// read-only, never touches `tile_region`/`block_cost_bounded`.
///
/// # Errors
///
/// As [`crate::block_cost`] / [`validate_candidate_prune`].
#[allow(clippy::too_many_arguments)]
#[allow(
    dead_code,
    reason = "Phase C infrastructure: only called from this module's own \
              tests today. Phase D's wiring is separately-scoped work."
)]
pub(crate) fn measure_prune_safety(
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
    samples: &mut Vec<PruneSample>,
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
        split_cost += measure_prune_safety(
            frame, hf_quants, grid, qx, qy, half, aq, x0, y0, cache, scratch, d_y_hf, samples,
        )?;
    }

    let fits = bx + size <= grid.width && by + size <= grid.height;
    let single_cost = match square_transform(size) {
        Some(transform) if fits => {
            let hf_mul = aq.mul_for_footprint(x0 / 8 + bx, y0 / 8 + by, size, size);
            let fixed =
                PER_VARBLOCK_BITS + NON_DCT8X8_SIGNAL_BITS + mul_signal_bits(hf_mul, aq.baseline);
            let px = x0 + bx * 8;
            let py = y0 + by * 8;
            // Same fetch `block_cost` would do internally — idempotent on a
            // cache hit, and this function needs the coefficients directly
            // (block_cost only returns their exact *score*).
            let hf_quant = hf_quants.get(transform, hf_mul)?;
            let fwd = cache.get_or_insert(frame, transform, px, py, scratch)?;
            let cutoff = split_cost - fixed;
            let side = transform.sample_cols();
            // Same unit conversion `block_cost_bounded` uses (`lib.rs`): one
            // squared coefficient unit is `side^2` squared sample units.
            #[allow(
                clippy::cast_precision_loss,
                reason = "side is at most 32; exact in f64"
            )]
            let to_sample_domain = (side * side) as f64;
            let sample = validate_candidate_prune(
                hf_quant,
                &hf_quants.lambda,
                to_sample_domain,
                side,
                transform.block_dims().0,
                fwd,
                cutoff,
            )?;
            samples.push(sample);
            Some(sample.exact_cost + fixed)
        }
        _ => None,
    };

    Ok(match single_cost {
        Some(c) if c < split_cost => c,
        _ => split_cost,
    })
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

    /// Runs `measure_prune_safety` over every LF group of `frame`, returning
    /// the accumulated samples (mirrors `measure_frame`'s shape).
    fn measure_prune_frame(frame: &PreparedFrame) -> Vec<PruneSample> {
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
        for index in 0..geometry.num_lf_groups() {
            let id = crate::LfGroupId::new(u32::try_from(index).unwrap_or(u32::MAX));
            let blocks = geometry.lf_group_blocks(id).expect("blocks");
            let rect = geometry.lf_group_rect(id).expect("rect");
            let mut d_y_hf = vec![0.0f32; 32 * 32];
            let mut sby = 0u32;
            while sby < blocks.height {
                let mut sbx = 0u32;
                while sbx < blocks.width {
                    measure_prune_safety(
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
                        &mut samples,
                    )
                    .expect("measures");
                    sbx += 4;
                }
                sby += 4;
            }
        }
        samples
    }

    /// S8 Phase C's exit gate: the staged cheap bound must never prune a
    /// candidate that would actually win. This is a provable-bound property,
    /// so it is checked exhaustively over every merge-candidate node the
    /// fixture produces, not sampled statistically — a single violation is a
    /// bug in `HfQuantizer::cell_lower_bound` or its staging here, not a
    /// tolerance to relax. Also reports the prune rate (the bound's
    /// usefulness), which is not asserted against a threshold here since
    /// Phase C's job is to prove safety and measure usefulness, not commit
    /// to a wiring decision — that is Phase D's.
    #[test]
    fn cell_lower_bound_prune_never_discards_a_true_winner() {
        let frame = ramp_frame(128, 128);
        let samples = measure_prune_frame(&frame);
        assert!(
            !samples.is_empty(),
            "the fixture must exercise real merge-candidate decisions"
        );
        let summary = PruneSummary::from_samples(&samples);
        eprintln!(
            "S8_PHASE_C_PRUNE_SAFETY candidates={} safety_violations={} pruned={} \
             prune_rate={:.4}",
            summary.candidates,
            summary.safety_violations,
            summary.pruned,
            summary.prune_rate()
        );
        assert_eq!(
            summary.safety_violations, 0,
            "the staged cheap bound pruned at least one candidate that would \
             actually have won: unsafe, not merely imprecise"
        );
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

    /// Gradient left half, hash-noise right half (same construction as
    /// `vardct_oracle.rs`'s `both_oracles_decode_an_adaptive_quantization_stream`):
    /// a field with both smooth and high-frequency content, complementing
    /// `ramp_frame`'s milder texture with a fixture more likely to produce
    /// guaranteed-nonzero cells (the cheap bound's loose, `2`-bit-only case).
    #[cfg(feature = "s8-cover-prune")]
    fn noisy_frame(width: u32, height: u32) -> PreparedFrame {
        let n = usize::try_from(width).unwrap_or(0) * usize::try_from(height).unwrap_or(0);
        let mut x_plane = vec![0.0f32; n];
        let mut y_plane = vec![0.0f32; n];
        let mut b_plane = vec![0.0f32; n];
        for row in 0..height {
            for col in 0..width {
                let idx = usize::try_from(row * width + col).unwrap_or(0);
                let v = if col < width / 2 {
                    (0.3 + f32::from(u16::try_from((col + row) % 64).unwrap_or(0)) / 128.0)
                        .clamp(0.0, 1.0)
                } else {
                    let hash = col
                        .wrapping_mul(0x9E37)
                        .wrapping_add(row.wrapping_mul(0x79B9))
                        .wrapping_mul(0x85EB_CA6B);
                    f32::from(u16::try_from((hash >> 24) & 0x3F).unwrap_or(0)) / 63.0
                };
                if let (Some(x), Some(y), Some(b)) = (
                    x_plane.get_mut(idx),
                    y_plane.get_mut(idx),
                    b_plane.get_mut(idx),
                ) {
                    *x = 0.0;
                    *y = v;
                    *b = v * 0.6;
                }
            }
        }
        PreparedFrame::from_linear_srgb(width, height, x_plane, y_plane, b_plane)
            .expect("legal frame")
    }

    /// `tile_region`'s total cost over every LF group of `frame`, exactly as
    /// `measure_region_matches_tile_regions_own_total_cost` computes it.
    #[cfg(feature = "s8-cover-prune")]
    fn tile_region_total(frame: &PreparedFrame) -> f64 {
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
                    let (cost, _blocks) = crate::tile_region(
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
                    )
                    .expect("tile_region");
                    total += cost;
                    sbx += 4;
                }
                sby += 4;
            }
        }
        total
    }

    /// S8 Phase D's decision-quality exit gate (`s8-cover-prune` feature
    /// only, so it only compiles/runs when the wiring under test is
    /// actually active): with the cheap staged bound live inside
    /// `block_cost_bounded`'s real cutoff-bounded calls (exactly the calls
    /// `tile_region` makes), `tile_region`'s own total cost must still
    /// exactly match this module's independent ground-truth total — which
    /// never applies the prune, or any cutoff at all
    /// ([`measure_frame`]/[`ExactPolicy`]). Phase C already proved the bound
    /// safe exhaustively in isolation
    /// (`cell_lower_bound_prune_never_discards_a_true_winner`); this proves
    /// the *wiring* preserves that safety once the prune is actually
    /// exercised by production code, over two fixtures with different
    /// spectral character. A tail-regret budget would be the wrong bar here
    /// — Phase C's bound is provable, not merely usually-right, so the
    /// correct budget is exactly zero, and that is what this asserts.
    #[cfg(feature = "s8-cover-prune")]
    #[test]
    fn wired_prune_does_not_change_tile_regions_decisions() {
        for (name, frame) in [
            ("ramp", ramp_frame(128, 128)),
            ("noisy", noisy_frame(128, 128)),
        ] {
            crate::diagnostics::reset_encode_diag();
            let tile_region_total = tile_region_total(&frame);
            let diag = crate::diagnostics::take_encode_diag();
            let (_samples, harness_total) = measure_frame(&frame, &ExactPolicy);
            eprintln!(
                "S8_PHASE_D_WIRED_PRUNE fixture={name} tile_region_total={tile_region_total:.6} \
                 harness_total={harness_total:.6} cover_prune_checks={} cover_prune_hits={}",
                diag.cover_prune_checks, diag.cover_prune_hits
            );
            assert!(
                diag.cover_prune_checks > 0,
                "fixture {name} exercised zero staged-bound checks: not a useful test of the \
                 wiring"
            );
            assert!(
                (tile_region_total - harness_total).abs() < 1e-6,
                "fixture {name}: tile_region total {tile_region_total} vs harness total \
                 {harness_total} with the prune wired live — the wiring changed a decision, \
                 which Phase C's exhaustive safety proof says should be impossible"
            );
        }
    }
}
