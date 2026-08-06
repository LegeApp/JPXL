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
    pub choose_cover: u64,
    pub choose_cfl_y: u64,
    pub choose_cfl_factor: u64,
    pub choose_final: u64,
    pub choose_other: u64,
    /// Wall nanoseconds spent in hierarchical (or fixed) cover selection + forwards.
    pub stage_cover_ns: u64,
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
    /// Selected-forward clones (one per selected varblock × 3 channels payload).
    pub selected_forward_clones: u64,
    /// Approximate bytes cloned by `forward_selected`.
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
        format!(
            "choose_total={} cover={} cfl_y={} cfl_factor={} final={} other={} \
             cover_ms={:.1} cfl_ms={:.1} quant_ms={:.1} entropy_ms={:.1} \
             cand_fwd={} cand_bytes={} sel_clones={} sel_bytes={} \
             cfl_samples={} cfl_sample_bytes={}",
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
            self.candidate_forwards,
            self.candidate_forward_bytes,
            self.selected_forward_clones,
            self.selected_forward_bytes,
            self.cfl_samples,
            self.cfl_sample_bytes,
        )
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
