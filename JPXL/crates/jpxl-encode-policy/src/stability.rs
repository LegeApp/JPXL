//! S8 Phase B measurement 2 (`jpegxl-rs.work.arch-s8-full-redesign-scoped`):
//! is the *winning* cover transform per region stable across the rate
//! loop's quantizer probes?
//!
//! `jpegxl-rs.work.arch-phase3-forward-cache` measured a 96-97% cross-probe
//! hit rate on `CandidateForwardCache` — but that only shows the *candidate
//! set* scored at each probe is stable (position + transform is
//! quantizer-independent, only `block_cost`'s score of each candidate
//! depends on the probe's quantizer). Whether the *winner* per region is
//! also probe-stable was explicitly left unverified, and is load-bearing
//! for whether a "cache summaries, not coefficients" redesign can preserve
//! that cross-probe reuse at all: if winners churn heavily across probes,
//! discarding raw coefficients for the losers of one probe could mean
//! re-computing them as winners of the next.
//!
//! This module is read-only measurement: it drives the real rate loop
//! ([`crate::rate::search_frame`]) to get the exact sequence of quantizers
//! a real search prices, then replays each one through the real planner
//! ([`crate::plan_at_on`], sharing one cache the way the rate loop
//! actually does) to recover that probe's cover, and compares consecutive
//! covers *at atom granularity* (not varblock origins, which are not
//! comparable across probes when the quadtree decomposition itself
//! changes — an atom is always covered by exactly one varblock in every
//! probe, so "which transform's footprint covers this atom" is the stable
//! unit of comparison).

use crate::error::Result;
use crate::rate::{QuantizerChoice, RateOutcome};
use crate::{
    AnalysisAtlas, CandidateForwardCache, EncodeRequest, EntropySearch, PreparedFrame, RateTarget,
};
use jpxl_core::varblock::TransformType;
use jpxl_encode::vardct::EmissionPlan;
use std::collections::HashMap;

/// `(LF group index, atom x, atom y)` — stable across probes because frame
/// geometry does not change when only the quantizer does.
type AtomKey = (u32, u32, u32);

/// Every atom's covering transform, for one probe's cover.
fn atom_transform_map(plan: &EmissionPlan) -> HashMap<AtomKey, TransformType> {
    let mut map = HashMap::new();
    for (group_index, group) in plan.spatial.lf_groups.iter().enumerate() {
        let group_index = u32::try_from(group_index).unwrap_or(u32::MAX);
        for vb in &group.blocks {
            let (rows, cols) = vb.transform.block_dims();
            let (bx, by) = (vb.origin.bx(), vb.origin.by());
            for dy in 0..u32::try_from(rows).unwrap_or(0) {
                for dx in 0..u32::try_from(cols).unwrap_or(0) {
                    map.insert((group_index, bx + dx, by + dy), vb.transform);
                }
            }
        }
    }
    map
}

/// Fraction of atoms whose covering transform differs between `a` and `b`,
/// over atoms present in both (they always are, for two covers of the same
/// frame geometry — this falls back to `0.0` defensively if that invariant
/// is ever violated, rather than dividing by zero).
fn churn(a: &HashMap<AtomKey, TransformType>, b: &HashMap<AtomKey, TransformType>) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    let mut differ = 0usize;
    let mut total = 0usize;
    for (key, &transform) in a {
        if let Some(&other) = b.get(key) {
            total += 1;
            if other != transform {
                differ += 1;
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        #[allow(
            clippy::cast_precision_loss,
            reason = "atom counts stay far inside f64's exact-integer range"
        )]
        let rate = differ as f64 / total as f64;
        rate
    }
}

/// One probe's cover, keyed by the quantizer that priced it.
struct ProbeCover {
    quantizer: QuantizerChoice,
    atoms: HashMap<AtomKey, TransformType>,
}

/// Summary of cross-probe winner churn over one rate search.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StabilitySummary {
    /// Number of probes actually replayed and compared.
    pub probes: u64,
    /// Mean churn between *consecutive* probes in the search's own
    /// (bracket/bisect/fill/refine) order.
    pub mean_adjacent_churn: f64,
    /// Maximum churn between any two consecutive probes.
    pub max_adjacent_churn: f64,
    /// Churn between the very first and very last probe replayed — the
    /// worst case a rate search's whole quantizer range can produce.
    pub first_vs_last_churn: f64,
    /// Each consecutive pair's churn, in probe order — lets a caller see
    /// *where* churn concentrates (e.g. the initial geometric bracket phase
    /// jumping across a wide quantizer range, versus the later bisect/fill/
    /// refine phases narrowing toward the answer) rather than only the
    /// aggregate. Same length as `probes - 1` (or `0` if `probes < 2`).
    pub adjacent_churn: Vec<f64>,
    /// Each probe's rung index, in the same order as `adjacent_churn`'s
    /// pairs (`rungs[i]`/`rungs[i+1]` are the pair `adjacent_churn[i]`
    /// measures), so a caller can correlate a churn spike with how far
    /// apart the two probes' quantizers actually were.
    pub rungs: Vec<u32>,
}

/// Runs a real rate-targeted search over `frame`, then replays every probe's
/// quantizer through the real planner (sharing one cache, as the rate loop
/// itself does) to recover and compare each probe's cover.
///
/// # Errors
///
/// As [`crate::rate::search_frame`] / [`crate::plan_at_on`].
pub fn measure_winner_stability(
    frame: &PreparedFrame,
    atlas: &AnalysisAtlas,
    request: &EncodeRequest,
    target: RateTarget,
) -> Result<StabilitySummary> {
    let outcome: RateOutcome = crate::rate::search_frame(frame, atlas, request, target)?;

    // Consecutive duplicate quantizers (the search can re-price the same
    // rung, e.g. Fast then Full) would contribute manufactured zero-churn
    // "agreement" that says nothing about probe-to-probe stability.
    let mut quantizers: Vec<QuantizerChoice> = Vec::new();
    for step in &outcome.trace {
        if quantizers.last() != Some(&step.quantizer) {
            quantizers.push(step.quantizer);
        }
    }

    let mut cache = CandidateForwardCache::new();
    let mut covers = Vec::with_capacity(quantizers.len());
    for quantizer in quantizers {
        let plan = crate::plan_at_on(
            frame,
            frame,
            atlas,
            request,
            quantizer,
            &mut cache,
            EntropySearch::Fast,
        )?;
        covers.push(ProbeCover {
            quantizer,
            atoms: atom_transform_map(plan.plan()),
        });
    }

    if covers.len() < 2 {
        return Ok(StabilitySummary {
            probes: u64::try_from(covers.len()).unwrap_or(0),
            ..StabilitySummary::default()
        });
    }

    let mut adjacent = Vec::with_capacity(covers.len() - 1);
    let mut rungs = Vec::with_capacity(covers.len());
    for pair in covers.windows(2) {
        let (Some(a), Some(b)) = (pair.first(), pair.get(1)) else {
            continue;
        };
        adjacent.push(churn(&a.atoms, &b.atoms));
        rungs.push(a.quantizer.rung.get());
    }
    if let Some(last) = covers.last() {
        rungs.push(last.quantizer.rung.get());
    }
    #[allow(
        clippy::cast_precision_loss,
        reason = "probe counts stay far inside f64's exact-integer range"
    )]
    let mean_adjacent_churn = adjacent.iter().sum::<f64>() / adjacent.len().max(1) as f64;
    let max_adjacent_churn = adjacent.iter().copied().fold(0.0, f64::max);
    let first_vs_last_churn = match (covers.first(), covers.last()) {
        (Some(first), Some(last)) => churn(&first.atoms, &last.atoms),
        _ => 0.0,
    };

    Ok(StabilitySummary {
        probes: u64::try_from(covers.len()).unwrap_or(0),
        mean_adjacent_churn,
        max_adjacent_churn,
        first_vs_last_churn,
        adjacent_churn: adjacent,
        rungs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp_rgb(width: u32, height: u32) -> Vec<u8> {
        let mut out = Vec::with_capacity(usize::try_from(width * height * 3).unwrap_or(0));
        for y in 0..height {
            for x in 0..width {
                let ramp = (x * 170 / width.max(1)) + (y * 70 / height.max(1));
                let checker = if (x / 16 + y / 16).is_multiple_of(2) {
                    12
                } else {
                    0
                };
                let luma = u8::try_from((20 + ramp + checker).min(255)).unwrap_or(255);
                out.extend_from_slice(&[
                    luma,
                    u8::try_from(u16::from(luma) * 4 / 5).unwrap_or(255),
                    u8::try_from(u16::from(luma) * 3 / 5).unwrap_or(255),
                ]);
            }
        }
        out
    }

    /// S8 Phase B, measurement 2: prints the churn numbers (this is a
    /// measurement, not a threshold assertion — nothing about the *answer*
    /// is known ahead of running it) and asserts only that the harness
    /// itself produced a real measurement over more than one probe.
    #[test]
    fn winner_stability_over_a_real_rate_search() {
        let (width, height) = (256u32, 256u32);
        let rgb = ramp_rgb(width, height);
        let frame = PreparedFrame::from_srgb8(width, height, &rgb).expect("frame");
        let atlas = AnalysisAtlas::analyze(&frame);
        let request = EncodeRequest::defaults();
        let summary =
            measure_winner_stability(&frame, &atlas, &request, RateTarget::BitsPerPixel(1.0))
                .expect("search succeeds");
        eprintln!(
            "S8_PHASE_B_WINNER_STABILITY probes={} mean_adjacent_churn={:.4} \
             max_adjacent_churn={:.4} first_vs_last_churn={:.4}",
            summary.probes,
            summary.mean_adjacent_churn,
            summary.max_adjacent_churn,
            summary.first_vs_last_churn
        );
        for (i, &c) in summary.adjacent_churn.iter().enumerate() {
            let (r0, r1) = (
                summary.rungs.get(i).copied().unwrap_or(0),
                summary.rungs.get(i + 1).copied().unwrap_or(0),
            );
            eprintln!("S8_PHASE_B_PAIR i={i} rung {r0}->{r1} churn={c:.4}");
        }
        assert!(
            summary.probes > 1,
            "the search must actually price more than one distinct quantizer"
        );
        assert!(
            (0.0..=1.0).contains(&summary.mean_adjacent_churn),
            "churn is a fraction of atoms"
        );
    }
}
