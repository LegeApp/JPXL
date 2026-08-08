//! Phase-0 encode diagnostics (outside-advice instrumentation).
//!
//! Thread-local counters and stage wall times for one VarDCT `plan_at` (or
//! rate search). Always recorded; zero cost when unread. Call
//! [`reset_encode_diag`] at the start of a measured encode and
//! [`take_encode_diag`] / [`last_encode_diag`] afterward.
//!
//! These numbers prove architectural multiplicity (how many times
//! [`crate::quantize::HfQuantizer::choose`] runs per stage, how much
//! candidate coefficient storage is retained). They are not a promoted
//! performance baseline by themselves.

use std::cell::Cell;
use std::time::Instant;

/// Where a scalar HF quantize call originates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum ChooseStage {
    /// Cover / `block_cost` scoring of candidate transforms.
    Cover = 0,
    /// Y quantize+reconstruct while building CfL sample lists.
    CflY = 1,
    /// X/B factor trials via `hf_residual_cost`.
    CflFactor = 2,
    /// Final selected-block HF quantization.
    Final = 3,
    /// Unclassified (should stay near zero).
    #[default]
    Other = 4,
}

/// Snapshot of one plan/encode attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EncodeDiag {
    /// S8 (outside-advice.md §8, narrowed): most of cover scoring now runs
    /// through `HfQuantizer::choose_lane4` (4 adjacent cells via SIMD, only
    /// falling back to scalar `choose` for a row segment's non-multiple-of-4
    /// remainder), and `choose_lane4` does not call `note_choose` — so this
    /// undercounts total quantization decisions in cover scoring since S8
    /// landed. It still tracks the scalar-remainder fraction accurately; it
    /// is no longer "total decisions," only "decisions that went through
    /// `choose` specifically." `choose_total()` is affected the same way.
    pub choose_cover: u64,
    pub choose_cfl_y: u64,
    pub choose_cfl_factor: u64,
    pub choose_final: u64,
    pub choose_other: u64,
    /// Wall nanoseconds spent in hierarchical (or fixed) cover selection + forwards.
    pub stage_cover_ns: u64,
    /// S8 Phase B: wall nanoseconds spent inside `block_cost_bounded` calling
    /// `CandidateForwardCache::get_or_insert` — the forward-DCT/cache-lookup
    /// share of cover scoring. Overlaps `stage_cover_ns` (a subset of it, not
    /// additional time); the two together answer outside-advice.md §8's
    /// "how much of cover scoring is the choose-loop vs. the forward" question.
    pub stage_cover_forward_ns: u64,
    /// S8 Phase B: wall nanoseconds spent inside `block_cost_bounded`'s
    /// `score_channel_lanes` calls (the `choose`/`choose_lane4` scoring loop)
    /// — the complementary share of `stage_cover_ns` to
    /// [`Self::stage_cover_forward_ns`].
    pub stage_cover_score_ns: u64,
    /// S8 Phase D (`s8-cover-prune` feature only): wall nanoseconds spent
    /// computing the cheap staged lower-bound checks
    /// (`HfQuantizer::cell_lower_bound`) that gate `score_channel_lanes`.
    /// Overlaps `stage_cover_ns`, not `stage_cover_score_ns` — this is the
    /// prune's own cost, to be weighed against the `stage_cover_score_ns`
    /// time it manages to skip. Compiled only under the feature so a
    /// default build carries no permanently-zero surface for a mechanism
    /// that measured slower and stays off (see HANDOFF S8 Phase D).
    #[cfg(feature = "s8-cover-prune")]
    pub stage_cover_prune_ns: u64,
    /// S8 Phase D: staged bound checks attempted (one per Y/X/B checkpoint
    /// per candidate, only when a `cutoff` exists).
    #[cfg(feature = "s8-cover-prune")]
    pub cover_prune_checks: u64,
    /// S8 Phase D: of `cover_prune_checks`, how many actually pruned
    /// (skipped that channel's exact `choose`/`choose_lane4` loop). The
    /// live analogue of Phase C's `PruneSummary::prune_rate`, measured on
    /// real corpora rather than the Phase C fixture.
    #[cfg(feature = "s8-cover-prune")]
    pub cover_prune_hits: u64,
    /// Wall nanoseconds spent in CfL sample construction and factor search.
    pub stage_cfl_ns: u64,
    /// Wall nanoseconds spent in final `quantize_group` over selected blocks.
    pub stage_quantize_ns: u64,
    /// Wall nanoseconds spent in entropy census / train / plan materialize.
    pub stage_entropy_ns: u64,
    /// Candidate forward coefficient vectors inserted into the DCT cache.
    pub candidate_forwards: u64,
    /// Approximate bytes of f32 coeffs retained in the candidate cache (sum of vector lens × 4).
    pub candidate_forward_bytes: u64,
    /// Selected-forward clones (one per selected varblock × 3 channels
    /// payload). Phase-3: always `0` now — `gather_forward_refs` borrows each
    /// selected varblock's forward from [`crate::CandidateForwardCache`]
    /// instead of cloning it into a second owned copy. Kept as a regression
    /// guard: if a future change reintroduces cloning, this stops being zero.
    pub selected_forward_clones: u64,
    /// Approximate bytes this would have cloned pre-Phase-3. Always `0` now;
    /// see [`Self::selected_forward_clones`].
    pub selected_forward_bytes: u64,
    /// CfL samples pushed (X and B together).
    pub cfl_samples: u64,
    /// Approximate bytes for CfL samples (16 B/sample estimate).
    pub cfl_sample_bytes: u64,
}

impl EncodeDiag {
    /// Total `HfQuantizer::choose` invocations across all stages.
    #[must_use]
    pub fn choose_total(self) -> u64 {
        self.choose_cover
            .saturating_add(self.choose_cfl_y)
            .saturating_add(self.choose_cfl_factor)
            .saturating_add(self.choose_final)
            .saturating_add(self.choose_other)
    }

    /// Human-readable one-line summary for CLI / scratch logs.
    #[must_use]
    pub fn summary_line(self) -> String {
        let base = format!(
            "choose_total={} cover={} cfl_y={} cfl_factor={} final={} other={} \
             cover_ms={:.1} cfl_ms={:.1} quant_ms={:.1} entropy_ms={:.1} \
             cover_forward_ms={:.1} cover_score_ms={:.1}",
            self.choose_total(),
            self.choose_cover,
            self.choose_cfl_y,
            self.choose_cfl_factor,
            self.choose_final,
            self.choose_other,
            self.stage_cover_ns as f64 / 1e6,
            self.stage_cfl_ns as f64 / 1e6,
            self.stage_quantize_ns as f64 / 1e6,
            self.stage_entropy_ns as f64 / 1e6,
            self.stage_cover_forward_ns as f64 / 1e6,
            self.stage_cover_score_ns as f64 / 1e6,
        );
        let tail = format!(
            "cand_fwd={} cand_bytes={} sel_clones={} sel_bytes={} \
             cfl_samples={} cfl_sample_bytes={}",
            self.candidate_forwards,
            self.candidate_forward_bytes,
            self.selected_forward_clones,
            self.selected_forward_bytes,
            self.cfl_samples,
            self.cfl_sample_bytes,
        );
        #[cfg(feature = "s8-cover-prune")]
        {
            let prune = format!(
                "cover_prune_ms={:.1} cover_prune_checks={} cover_prune_hits={}",
                self.stage_cover_prune_ns as f64 / 1e6,
                self.cover_prune_checks,
                self.cover_prune_hits,
            );
            format!("{base} {prune} {tail}")
        }
        #[cfg(not(feature = "s8-cover-prune"))]
        {
            format!("{base} {tail}")
        }
    }
}

std::thread_local! {
    static DIAG: Cell<EncodeDiag> = const { Cell::new(EncodeDiag {
        choose_cover: 0,
        choose_cfl_y: 0,
        choose_cfl_factor: 0,
        choose_final: 0,
        choose_other: 0,
        stage_cover_ns: 0,
        stage_cover_forward_ns: 0,
        stage_cover_score_ns: 0,
        #[cfg(feature = "s8-cover-prune")]
        stage_cover_prune_ns: 0,
        #[cfg(feature = "s8-cover-prune")]
        cover_prune_checks: 0,
        #[cfg(feature = "s8-cover-prune")]
        cover_prune_hits: 0,
        stage_cfl_ns: 0,
        stage_quantize_ns: 0,
        stage_entropy_ns: 0,
        candidate_forwards: 0,
        candidate_forward_bytes: 0,
        selected_forward_clones: 0,
        selected_forward_bytes: 0,
        cfl_samples: 0,
        cfl_sample_bytes: 0,
    }) };
    static STAGE: Cell<ChooseStage> = const { Cell::new(ChooseStage::Other) };
}

/// Clears counters for a new measured encode on this thread.
pub fn reset_encode_diag() {
    DIAG.with(|c| c.set(EncodeDiag::default()));
    STAGE.with(|c| c.set(ChooseStage::Other));
}

/// Snapshot of the current counters without clearing.
#[must_use]
pub fn last_encode_diag() -> EncodeDiag {
    DIAG.with(Cell::get)
}

/// Snapshot and clear.
#[must_use]
pub fn take_encode_diag() -> EncodeDiag {
    DIAG.with(|c| {
        let v = c.get();
        c.set(EncodeDiag::default());
        v
    })
}

/// Sets the stage that subsequent [`note_choose`] calls attribute to.
pub fn set_choose_stage(stage: ChooseStage) {
    STAGE.with(|c| c.set(stage));
}

/// Runs `f` with [`ChooseStage`] set, restoring the previous stage afterward.
pub fn with_choose_stage<R>(stage: ChooseStage, f: impl FnOnce() -> R) -> R {
    let prev = STAGE.with(Cell::get);
    STAGE.with(|c| c.set(stage));
    let out = f();
    STAGE.with(|c| c.set(prev));
    out
}

/// Records one `HfQuantizer::choose` call against the current stage.
#[inline]
pub fn note_choose() {
    let stage = STAGE.with(Cell::get);
    DIAG.with(|c| {
        let mut d = c.get();
        match stage {
            ChooseStage::Cover => d.choose_cover = d.choose_cover.saturating_add(1),
            ChooseStage::CflY => d.choose_cfl_y = d.choose_cfl_y.saturating_add(1),
            ChooseStage::CflFactor => d.choose_cfl_factor = d.choose_cfl_factor.saturating_add(1),
            ChooseStage::Final => d.choose_final = d.choose_final.saturating_add(1),
            ChooseStage::Other => d.choose_other = d.choose_other.saturating_add(1),
        }
        c.set(d);
    });
}

/// Accumulates wall time for a named plan stage.
pub fn note_stage_ns(which: StageTimer, ns: u64) {
    DIAG.with(|c| {
        let mut d = c.get();
        match which {
            StageTimer::Cover => d.stage_cover_ns = d.stage_cover_ns.saturating_add(ns),
            StageTimer::CoverForward => {
                d.stage_cover_forward_ns = d.stage_cover_forward_ns.saturating_add(ns);
            }
            StageTimer::CoverScore => {
                d.stage_cover_score_ns = d.stage_cover_score_ns.saturating_add(ns);
            }
            #[cfg(feature = "s8-cover-prune")]
            StageTimer::CoverPrune => {
                d.stage_cover_prune_ns = d.stage_cover_prune_ns.saturating_add(ns);
            }
            StageTimer::Cfl => d.stage_cfl_ns = d.stage_cfl_ns.saturating_add(ns),
            StageTimer::Quantize => d.stage_quantize_ns = d.stage_quantize_ns.saturating_add(ns),
            StageTimer::Entropy => d.stage_entropy_ns = d.stage_entropy_ns.saturating_add(ns),
        }
        c.set(d);
    });
}

/// Plan-side stage keys (emit is timed outside policy).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageTimer {
    Cover,
    /// S8 Phase B sub-stage of [`Self::Cover`]: forward-DCT/cache access.
    CoverForward,
    /// S8 Phase B sub-stage of [`Self::Cover`]: the choose-loop scoring pass.
    CoverScore,
    /// S8 Phase D sub-stage of [`Self::Cover`] (`s8-cover-prune` feature
    /// only): the cheap staged lower-bound checks, timed separately from
    /// [`Self::CoverScore`] so the prune's own cost can be weighed against
    /// the scoring time it manages to skip.
    #[cfg(feature = "s8-cover-prune")]
    CoverPrune,
    Cfl,
    Quantize,
    Entropy,
}

/// Times `f` and adds the elapsed wall nanoseconds to `which`.
pub fn time_stage<R>(which: StageTimer, f: impl FnOnce() -> R) -> R {
    let t0 = Instant::now();
    let out = f();
    let ns = u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX);
    note_stage_ns(which, ns);
    out
}

/// Records one candidate-forward cache insert (`n_f32` coefficients total across channels).
pub fn note_candidate_forward(n_f32: usize) {
    DIAG.with(|c| {
        let mut d = c.get();
        d.candidate_forwards = d.candidate_forwards.saturating_add(1);
        let bytes = (n_f32 as u64).saturating_mul(4);
        d.candidate_forward_bytes = d.candidate_forward_bytes.saturating_add(bytes);
        c.set(d);
    });
}

/// Records one selected-forward clone (`n_f32` coefficients).
pub fn note_selected_forward_clone(n_f32: usize) {
    DIAG.with(|c| {
        let mut d = c.get();
        d.selected_forward_clones = d.selected_forward_clones.saturating_add(1);
        let bytes = (n_f32 as u64).saturating_mul(4);
        d.selected_forward_bytes = d.selected_forward_bytes.saturating_add(bytes);
        c.set(d);
    });
}

/// Records one S8 Phase D staged cheap-bound check attempt
/// (`s8-cover-prune` feature only).
#[cfg(feature = "s8-cover-prune")]
pub fn note_cover_prune_check() {
    DIAG.with(|c| {
        let mut d = c.get();
        d.cover_prune_checks = d.cover_prune_checks.saturating_add(1);
        c.set(d);
    });
}

/// Records one S8 Phase D staged cheap-bound check that actually pruned
/// (`s8-cover-prune` feature only).
#[cfg(feature = "s8-cover-prune")]
pub fn note_cover_prune_hit() {
    DIAG.with(|c| {
        let mut d = c.get();
        d.cover_prune_hits = d.cover_prune_hits.saturating_add(1);
        c.set(d);
    });
}

/// Records CfL sample pushes (`count` samples at ~16 B each).
pub fn note_cfl_samples(count: u64) {
    DIAG.with(|c| {
        let mut d = c.get();
        d.cfl_samples = d.cfl_samples.saturating_add(count);
        d.cfl_sample_bytes = d.cfl_sample_bytes.saturating_add(count.saturating_mul(16));
        c.set(d);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choose_stages_accumulate() {
        reset_encode_diag();
        with_choose_stage(ChooseStage::Cover, || {
            note_choose();
            note_choose();
        });
        with_choose_stage(ChooseStage::Final, || note_choose());
        let d = take_encode_diag();
        assert_eq!(d.choose_cover, 2);
        assert_eq!(d.choose_final, 1);
        assert_eq!(d.choose_total(), 3);
    }
}
