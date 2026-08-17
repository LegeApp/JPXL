//! Opt-in aggregate diagnostics for VarDCT Count/Store emission.
//!
//! The policy crate selects the current Fast/Full search phase, while this
//! normative writer records only execution multiplicity. Diagnostics are
//! disabled by default and do not affect emitted bytes.

use std::cell::Cell;
use std::time::Instant;

/// Rate-search phase that owns a writer operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DiagnosticPhase {
    Fast,
    Full,
    #[default]
    Other,
}

/// Why a count-only writer traversal was requested.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CountEmissionKind {
    /// Entropy alternative comparison inside one plan.
    Internal,
    /// The outer Fast rate-probe price.
    Outer,
    #[default]
    Other,
}

/// Writer work attributed to one search phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WriterPhaseDiagnostics {
    pub internal_count_emissions: u64,
    pub outer_count_emissions: u64,
    pub other_count_emissions: u64,
    pub stored_emissions: u64,
    pub section_body_traversals: u64,
    pub lf_section_encodes: u64,
    pub pass_group_section_encodes: u64,
    pub executor_pool_builds: u64,
    pub count_emission_ns: u64,
    pub stored_emission_ns: u64,
    pub executor_pool_build_ns: u64,
    /// Phase 41: HF tokens recorded on pass-group tapes (0 when the
    /// `hf-token-tape` feature is off).
    pub tape_symbols: u64,
}

impl WriterPhaseDiagnostics {
    #[must_use]
    pub fn count_emissions(self) -> u64 {
        self.internal_count_emissions
            .saturating_add(self.outer_count_emissions)
            .saturating_add(self.other_count_emissions)
    }
}

/// Aggregate writer work for one target-rate search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WriterDiagnostics {
    pub fast: WriterPhaseDiagnostics,
    pub full: WriterPhaseDiagnostics,
    pub other: WriterPhaseDiagnostics,
}

impl WriterDiagnostics {
    #[must_use]
    pub fn total_section_traversals(self) -> u64 {
        self.fast
            .section_body_traversals
            .saturating_add(self.full.section_body_traversals)
            .saturating_add(self.other.section_body_traversals)
    }
}

std::thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static PHASE: Cell<DiagnosticPhase> = const { Cell::new(DiagnosticPhase::Other) };
    static COUNT_KIND: Cell<CountEmissionKind> = const { Cell::new(CountEmissionKind::Other) };
    static DIAGNOSTICS: Cell<WriterDiagnostics> = const { Cell::new(WriterDiagnostics {
        fast: WriterPhaseDiagnostics {
            internal_count_emissions: 0, outer_count_emissions: 0,
            other_count_emissions: 0, stored_emissions: 0,
            section_body_traversals: 0, lf_section_encodes: 0,
            pass_group_section_encodes: 0, executor_pool_builds: 0,
            count_emission_ns: 0, stored_emission_ns: 0,
            executor_pool_build_ns: 0, tape_symbols: 0,
        },
        full: WriterPhaseDiagnostics {
            internal_count_emissions: 0, outer_count_emissions: 0,
            other_count_emissions: 0, stored_emissions: 0,
            section_body_traversals: 0, lf_section_encodes: 0,
            pass_group_section_encodes: 0, executor_pool_builds: 0,
            count_emission_ns: 0, stored_emission_ns: 0,
            executor_pool_build_ns: 0, tape_symbols: 0,
        },
        other: WriterPhaseDiagnostics {
            internal_count_emissions: 0, outer_count_emissions: 0,
            other_count_emissions: 0, stored_emissions: 0,
            section_body_traversals: 0, lf_section_encodes: 0,
            pass_group_section_encodes: 0, executor_pool_builds: 0,
            count_emission_ns: 0, stored_emission_ns: 0,
            executor_pool_build_ns: 0, tape_symbols: 0,
        },
    }) };
}

pub fn set_enabled(enabled: bool) {
    ENABLED.with(|cell| cell.set(enabled));
}

#[inline]
pub(crate) fn enabled() -> bool {
    ENABLED.with(Cell::get)
}

pub fn reset() {
    DIAGNOSTICS.with(|cell| cell.set(WriterDiagnostics::default()));
    PHASE.with(|cell| cell.set(DiagnosticPhase::Other));
    COUNT_KIND.with(|cell| cell.set(CountEmissionKind::Other));
}

#[must_use]
pub fn snapshot() -> WriterDiagnostics {
    DIAGNOSTICS.with(Cell::get)
}

pub fn with_phase<R>(phase: DiagnosticPhase, f: impl FnOnce() -> R) -> R {
    if !enabled() {
        return f();
    }
    let previous = PHASE.with(Cell::get);
    PHASE.with(|cell| cell.set(phase));
    let result = f();
    PHASE.with(|cell| cell.set(previous));
    result
}

pub fn with_count_kind<R>(kind: CountEmissionKind, f: impl FnOnce() -> R) -> R {
    if !enabled() {
        return f();
    }
    let previous = COUNT_KIND.with(Cell::get);
    COUNT_KIND.with(|cell| cell.set(kind));
    let result = f();
    COUNT_KIND.with(|cell| cell.set(previous));
    result
}

fn update(f: impl FnOnce(&mut WriterPhaseDiagnostics)) {
    if !enabled() {
        return;
    }
    let phase = PHASE.with(Cell::get);
    DIAGNOSTICS.with(|cell| {
        let mut diagnostics = cell.get();
        let target = match phase {
            DiagnosticPhase::Fast => &mut diagnostics.fast,
            DiagnosticPhase::Full => &mut diagnostics.full,
            DiagnosticPhase::Other => &mut diagnostics.other,
        };
        f(target);
        cell.set(diagnostics);
    });
}

pub(crate) fn time_count_emission<R>(f: impl FnOnce() -> R) -> R {
    if !enabled() {
        return f();
    }
    let started = Instant::now();
    let result = f();
    let elapsed = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let kind = COUNT_KIND.with(Cell::get);
    update(|diagnostics| {
        match kind {
            CountEmissionKind::Internal => {
                diagnostics.internal_count_emissions =
                    diagnostics.internal_count_emissions.saturating_add(1);
            }
            CountEmissionKind::Outer => {
                diagnostics.outer_count_emissions =
                    diagnostics.outer_count_emissions.saturating_add(1);
            }
            CountEmissionKind::Other => {
                diagnostics.other_count_emissions =
                    diagnostics.other_count_emissions.saturating_add(1);
            }
        }
        diagnostics.count_emission_ns = diagnostics.count_emission_ns.saturating_add(elapsed);
    });
    result
}

pub(crate) fn time_stored_emission<R>(f: impl FnOnce() -> R) -> R {
    if !enabled() {
        return f();
    }
    let started = Instant::now();
    let result = f();
    let elapsed = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    update(|diagnostics| {
        diagnostics.stored_emissions = diagnostics.stored_emissions.saturating_add(1);
        diagnostics.stored_emission_ns = diagnostics.stored_emission_ns.saturating_add(elapsed);
    });
    result
}

/// Phase 41: records the number of HF tokens a frame's pass-group tapes hold.
#[cfg_attr(not(feature = "hf-token-tape"), allow(dead_code))]
pub(crate) fn note_tape_symbols(symbols: usize) {
    update(|diagnostics| {
        diagnostics.tape_symbols = diagnostics
            .tape_symbols
            .saturating_add(u64::try_from(symbols).unwrap_or(u64::MAX));
    });
}

pub(crate) fn note_sections(total: usize, lf: usize, pass_groups: usize) {
    update(|diagnostics| {
        diagnostics.section_body_traversals = diagnostics
            .section_body_traversals
            .saturating_add(u64::try_from(total).unwrap_or(u64::MAX));
        diagnostics.lf_section_encodes = diagnostics
            .lf_section_encodes
            .saturating_add(u64::try_from(lf).unwrap_or(u64::MAX));
        diagnostics.pass_group_section_encodes = diagnostics
            .pass_group_section_encodes
            .saturating_add(u64::try_from(pass_groups).unwrap_or(u64::MAX));
    });
}

#[cfg(any(feature = "parallel", test))]
pub(crate) fn note_pool_build(elapsed_ns: u64) {
    update(|diagnostics| {
        diagnostics.executor_pool_builds = diagnostics.executor_pool_builds.saturating_add(1);
        diagnostics.executor_pool_build_ns = diagnostics
            .executor_pool_build_ns
            .saturating_add(elapsed_ns);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_writer_diagnostics_remain_empty() {
        set_enabled(false);
        reset();
        with_phase(DiagnosticPhase::Fast, || {
            with_count_kind(CountEmissionKind::Outer, || time_count_emission(|| {}));
            note_sections(4, 1, 1);
            note_pool_build(9);
        });
        assert_eq!(snapshot(), WriterDiagnostics::default());
    }

    #[test]
    fn phase_and_count_kind_are_attributed() {
        set_enabled(true);
        reset();
        with_phase(DiagnosticPhase::Fast, || {
            with_count_kind(CountEmissionKind::Outer, || time_count_emission(|| {}));
            note_sections(7, 2, 3);
        });
        with_phase(DiagnosticPhase::Full, || time_stored_emission(|| {}));
        let diagnostics = snapshot();
        assert_eq!(diagnostics.fast.outer_count_emissions, 1);
        assert_eq!(diagnostics.fast.section_body_traversals, 7);
        assert_eq!(diagnostics.full.stored_emissions, 1);
        set_enabled(false);
    }
}
