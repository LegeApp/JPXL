//! The exact rate loop (`docs/PLAN.md` slice 14, `Encoder-plan1.md` milestone
//! 4): pick a quantizer that lands a byte target.
//!
//! # Three properties this loop is built around
//!
//! **1. Production candidates are priced exactly.** In default builds a
//! candidate's size is [`jpxl_encode::vardct::price_codestream`], which runs
//! the real writer into a scratch buffer. The feature-gated Fast preset
//! controller fits a local rate curve from two exact anchors, then prices an
//! anchored finalist exactly and falls back to the exhaustive controller
//! unless it satisfies the preset's byte contract.
//!
//! **2. The search moves over wire-legal values only.** I.2.1's `global_scale`
//! is a `U32()` field with a largest expressible value, `HfMul` is a Modular
//! sample, and `quant_lf` has its own distribution; a search that moved a float
//! and rounded at the end would evaluate points it cannot emit and report a
//! size for a plan that does not exist. The search space here is
//! [`Rung`] — an index into an enumerated ladder of **representable**
//! quantizers, and the only thing the loop ever moves.
//!
//! **3. Monotonicity is bracketed, never assumed.** Size is *intended* to rise
//! with the ladder, and mostly does; but the entropy coder can make a finer
//! quantizer produce a smaller file — a histogram that clusters better, a token
//! that falls into a cheaper bucket. The loop therefore
//!
//! * keeps `best` = the largest **feasible** size ever priced, not "wherever
//!   the bisection stopped", so a non-monotone pocket cannot lose a candidate
//!   that was already proved to fit;
//! * terminates on a step count, never on a convergence predicate that a
//!   non-monotone function could keep false forever;
//! * finishes with a bounded linear probe above the bisection boundary, which
//!   is exactly where a pocket hides.
//!
//! # The ladder, and the LF/HF coupling policy
//!
//! I.2.1 factors the quantizer: the HF step is `(1 << 16) / (global_scale *
//! HfMul)` and the LF step is `(1 << 16) * w / (global_scale * quant_lf)`. So
//! `global_scale` moves **both** planes together and the other two are ratios
//! against it.
//!
//! The primary search holds `quant_lf` at the request's value and moves
//! `global_scale` (then `HfMul` past the scale ceiling). That keeps the LF/HF
//! *balance* fixed across the ladder so the loop changes rate without silently
//! changing reconstruction character.
//!
//! **Secondary LF fill (M8 leftover):** when HF is small (trained entropy),
//! coarse targets become LF-dominated and adjacent `global_scale` rungs can
//! cliff by ~kB across a modular dither threshold (wave 13). After the ladder
//! settles, if undershoot remains and budget allows, a short discrete probe
//! over legal `quant_lf` values at the winning rung can spend that slack
//! without reopening the whole R-D search. The request's `quant_lf` remains
//! the default balance; the fill only moves it when it strictly improves
//! achieved size under the never-over contract.
//!
//! `HfMul` is **not** an independent rate axis at fixed blocks: `HfMul = 2` at
//! `global_scale = g` is the same HF quantizer as `HfMul = 1` at
//! `global_scale = 2g`. It earns its place at the top of the ladder, where
//! `global_scale` has hit the largest value I.2.1 can express and `HfMul` is
//! the only way to go finer — and there the ladder interleaves it with
//! `global_scale` (see [`HF_MUL_RUNGS`]) and couples `quant_lf` to it, so
//! neither the HF step nor the LF/HF balance jumps at the ceiling.
//! Per-varblock `HfMul` is milestone 7.

use jpxl_encode::vardct::ids::{GlobalScale, HfMul, MAX_GLOBAL_SCALE, QuantLf};
#[cfg(feature = "anchor-sketch")]
use jpxl_encode::vardct::plan::EntropyPlan;
use jpxl_encode::vardct::size::CodestreamSizing;
use jpxl_encode::vardct::{
    Emission, ValidatedEmissionPlan, emit_codestream_with_executor, price_codestream_with,
};

use crate::error::{PolicyError, Result};
#[cfg(feature = "anchor-sketch")]
use crate::request::RateSearchPreset;
use crate::request::{EncodeRequest, RateSearchBudget, RateTarget, RateTolerance};
#[cfg(feature = "anchor-sketch")]
use crate::{AnchorReuse, StructuralAnchor};
use crate::{CandidateForwardCache, EntropySearch, diagnostics};

/// How many `HfMul` *octaves* the ladder extends above `global_scale`'s
/// ceiling: the top of the ladder is `HfMul = HF_MUL_RUNGS + 1` at the largest
/// `global_scale`, an HF step 65 times finer than the finest `global_scale`
/// alone can reach. Past that the coefficients are integers in the millions
/// and the LF plane is the accuracy floor anyway.
///
/// Above the ceiling the ladder does **not** step by whole `HfMul` multiples
/// (Phase Q3). `HfMul = 2` at the ceiling is a full octave finer than
/// `HfMul = 1` there, and on a smooth 4 MP photograph that one rung was a
/// 65% jump in bytes: every target between the two landed 35% under, "budget
/// spent, not at the ladder's limit". Instead each `HfMul = k` segment walks
/// `global_scale` from just above `(k - 1) / k` of the ceiling up to the
/// ceiling, so consecutive rungs differ by `k` units of effective scale and
/// the segments join without a gap: the last rung of segment `k` is
/// `k * MAX` and the first of `k + 1` is at most `k + 1` above it.
pub const HF_MUL_RUNGS: u32 = 64;

/// The largest frame-constant `HfMul` the ladder reaches.
const MAX_LADDER_MUL: u32 = HF_MUL_RUNGS + 1;

/// The `global_scale` a rung of segment `k` starts at: the smallest scale
/// whose product with `k` exceeds the previous segment's top `(k - 1) * MAX`.
const fn segment_first_scale(k: u32) -> u32 {
    // (k - 1) * MAX / k, floored, plus one — never below 1. The quotient is
    // below MAX_GLOBAL_SCALE by construction, so the narrowing is exact.
    let prev_top = (k as u64 - 1) * MAX_GLOBAL_SCALE as u64;
    #[allow(
        clippy::cast_possible_truncation,
        reason = "(k - 1) * MAX / k < MAX fits u32"
    )]
    let floor = (prev_top / k as u64) as u32;
    if floor >= MAX_GLOBAL_SCALE {
        MAX_GLOBAL_SCALE
    } else {
        floor + 1
    }
}

/// How many rungs segment `k` holds.
const fn segment_len(k: u32) -> u32 {
    MAX_GLOBAL_SCALE - segment_first_scale(k) + 1
}

/// The rung count of every segment above the ceiling.
const fn upper_rungs() -> u32 {
    let mut total = 0u32;
    let mut k = 2u32;
    while k <= MAX_LADDER_MUL {
        total += segment_len(k);
        k += 1;
    }
    total
}

/// Above the ceiling, the LF quantiser is coupled to the segment: `quant_lf`
/// is multiplied by `k` while that keeps it at or below this bound, so as
/// `global_scale` halves across the first segments the LF step is unchanged
/// and the LF/HF balance the ladder holds below the ceiling carries through
/// it. The bound exists because jxl-oxide narrows `LfQuant` samples to signed
/// 16 bits (Phase 5F): the LF sample magnitude grows with
/// `global_scale * quant_lf`, and `MAX_GLOBAL_SCALE * 16` is the product the
/// fixed-quantizer defaults (`quant_lf` 16 at the ceiling) have always been
/// allowed to reach; a synthetic high-contrast oracle fixture wraps at twice
/// that. Beyond the bound `quant_lf` stays put and the LF step sawtooths by
/// at most `1 / (k + 1)` across a segment join, which for the target policy's
/// `quant_lf` 4 starts only above four octaves past the ceiling.
const LF_COUPLING_MAX_QUANT_LF: u32 = 16;

/// The coupled `quant_lf` for segment `k`: the largest `base * j`, `j <= k`,
/// that stays within [`LF_COUPLING_MAX_QUANT_LF`] — or `base` itself when
/// even `k = 1` would not (a request already finer than the bound is left
/// exactly as asked).
fn coupled_quant_lf(base: QuantLf, k: u32) -> QuantLf {
    let mut best = base;
    for j in 2..=k {
        match QuantLf::new(base.get().saturating_mul(j)) {
            Ok(candidate) if candidate.get() <= LF_COUPLING_MAX_QUANT_LF => best = candidate,
            _ => break,
        }
    }
    best
}

/// One past the last ladder index.
pub const LADDER_LEN: u32 = MAX_GLOBAL_SCALE + upper_rungs();

/// The `(global_scale, HfMul)` pair of a rung, on either side of the ceiling.
fn rung_fields(rung: Rung) -> (u32, u32) {
    if rung.get() < MAX_GLOBAL_SCALE {
        return (rung.get() + 1, 1);
    }
    let mut offset = rung.get() - MAX_GLOBAL_SCALE;
    let mut k = 2u32;
    while k < MAX_LADDER_MUL && offset >= segment_len(k) {
        offset -= segment_len(k);
        k += 1;
    }
    ((segment_first_scale(k) + offset).min(MAX_GLOBAL_SCALE), k)
}

/// A position on the quantizer ladder: **larger is finer, and bigger**.
///
/// Not a bare `u32` on purpose (`AGENTS.md`): a rung, a `global_scale` and a
/// byte count are three different things that are all small integers, and the
/// loop below mixes all three.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Rung(u32);

impl Rung {
    /// The coarsest quantizer the ladder holds.
    pub const FLOOR: Self = Self(0);
    /// The finest quantizer the ladder holds.
    pub const TOP: Self = Self(LADDER_LEN - 1);

    /// Wraps an index, clamped into the ladder.
    #[must_use]
    pub const fn new(index: u32) -> Self {
        if index >= LADDER_LEN {
            return Self::TOP;
        }
        Self(index)
    }

    /// The index.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }

    /// The rung whose `global_scale` is `scale` (and whose `HfMul` is one).
    #[must_use]
    pub const fn for_global_scale(scale: u32) -> Self {
        if scale == 0 {
            return Self::FLOOR;
        }
        Self::new(scale - 1)
    }
}

/// A representable quantizer: the three wire fields, already range-checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizerChoice {
    /// Where on the ladder this sits.
    pub rung: Rung,
    /// I.2.1's `global_scale`.
    pub global_scale: GlobalScale,
    /// The frame-constant `HfMul` (G.2.4).
    pub hf_mul: HfMul,
    /// I.2.1's `quant_lf`.
    pub quant_lf: QuantLf,
}

impl QuantizerChoice {
    /// The quantizer at `rung`, at the LF/HF ratio `quant_lf` names.
    ///
    /// # Errors
    ///
    /// [`PolicyError::Plan`] can only fire if [`LADDER_LEN`] and the fields'
    /// own ranges disagree, which the `every_rung_is_representable` test pins.
    pub fn at(rung: Rung, quant_lf: QuantLf) -> Result<Self> {
        let (scale, mul) = rung_fields(rung);
        Ok(Self {
            rung,
            global_scale: GlobalScale::new(scale)?,
            hf_mul: HfMul::new(mul)?,
            quant_lf: coupled_quant_lf(quant_lf, mul),
        })
    }

    /// The quantizer a request names when it sets no target.
    ///
    /// The rung is the one whose `global_scale` matches; a request with a
    /// non-unit `HfMul` has no exact rung, and gets the `global_scale` one,
    /// which is only ever used as the search's starting point.
    #[must_use]
    pub fn from_request(request: &EncodeRequest) -> Self {
        Self {
            rung: Rung::for_global_scale(request.global_scale.get()),
            global_scale: request.global_scale,
            hf_mul: request.hf_mul,
            quant_lf: request.quant_lf,
        }
    }
}

/// Which phase of the loop priced a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RatePhase {
    /// Geometric search for a feasible/infeasible pair around the target.
    Bracket,
    /// Bisection inside that pair.
    Bisect,
    /// The discrete budget fill: notches above the bisection boundary.
    Fill,
    /// Secondary fill over legal `quant_lf` at the winning rung (LF cliffs).
    LfFill,
    /// Finalist refinement after the Fast ladder: navigate with the trained
    /// default model, then pay for real entropy alternatives at the finalist
    /// and in a bounded exact correction window.
    Final,
    /// Bounded fresh-structure rescue after the anchored finalist missed.
    Rescue,
}

/// Why a completed target-rate search stopped where it did.
///
/// Production presets report bounded misses instead of silently escalating to
/// the exhaustive reference controller. Callers can therefore distinguish a
/// target-band hit, an expressiveness limit, and a deliberate work cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateStatus {
    /// The selected stream is within the preset's requested target band.
    InsideBand,
    /// Adjacent priced rungs straddle the target but the feasible one is
    /// outside the requested undershoot band.
    UnderTargetAdjacentRungs,
    /// The bounded production controller stopped outside the target band.
    UnderTargetWorkCap,
    /// The finest representable rung is still below the target.
    SaturatedTop,
    /// A bounded fresh cover/CfL rescue supplied the selected stream.
    RescuedFreshStructure,
    /// The explicitly requested Quality reference controller was used.
    ExhaustiveReference,
}

#[cfg(feature = "anchor-sketch")]
const MAX_BOUNDED_EXACT_PRICES: usize = 6;

#[cfg(feature = "anchor-sketch")]
const MAX_FRESH_RESCUE_PRICES: u32 = 2;

/// How many prices to hold back from the Fast ladder for finalist refinement.
///
/// Fast entropy overestimates size (it skips alternatives that shrink the
/// stream). The Fast ladder therefore lands coarser than Full would. The
/// refinement reserve must cover cheap default-model navigation plus an exact
/// Full finalist and bounded corrections — not a one-notch walk — or the
/// residual undershoot blows the 1% contract.
fn full_refinement_reserve(max_prices: u32) -> u32 {
    if max_prices <= 6 {
        // Tiny budgets: one Full re-plan of the Fast winner; undershoot is
        // the honest trade for a five-price cap (see the rate_loop test).
        1
    } else {
        // Geometric bracket + bisect + fill on a ~2^17-rung ladder needs on
        // the order of a dozen prices. Take half the budget, clamp to 8..=20,
        // and never leave Fast with zero.
        16u32
            .min(max_prices / 2)
            .max(8)
            .min(max_prices.saturating_sub(1))
    }
}

/// Largest frame on which post-ceiling Quality may reinvest one saved bracket
/// price in an exact density top-off.
///
/// One Full alternative price is cheap enough to be a net win on the standing
/// 4.3 MP frames, but costs seconds on the 12 MP frame. The guard is therefore
/// a work bound, not a quality heuristic: larger frames keep the caller's
/// tolerance as their stop, while moderate frames may spend exactly one
/// already-budgeted correction after first entering that band.
const QUALITY_TOPOFF_MAX_PIXELS: u64 = 5_000_000;

fn quality_topoff_prices(
    preset: crate::request::RateSearchPreset,
    finalist: Rung,
    pixels: u64,
) -> u32 {
    if preset == crate::request::RateSearchPreset::Quality
        && finalist.get() >= MAX_GLOBAL_SCALE
        && pixels <= QUALITY_TOPOFF_MAX_PIXELS
    {
        1
    } else {
        0
    }
}

/// Discrete `quant_lf` values tried during [`RatePhase::LfFill`].
///
/// All are representable under I.2.1's `U32(16, 1+u(5), 1+u(8), 1+u(16))`.
/// The set is intentionally sparse: each probe is a full encode.
const QUANT_LF_FILL: &[u32] = &[8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 1024];

/// LF-fill candidates in distortion-preserving direction.
///
/// The fill runs only when the selected stream is under target. I.2.1 divides
/// the LF step by `quant_lf`, so values below the caller's balance make LF
/// coarser: even if entropy non-monotonicity made such a stream a few bytes
/// larger, spending bytes by throwing away LF precision is the wrong trade.
/// Probe only finer LF values, nearest first; exact writer pricing still makes
/// the never-over decision.
fn quant_lf_fill_candidates(base: QuantLf) -> Vec<u32> {
    let mut candidates: Vec<u32> = QUANT_LF_FILL
        .iter()
        .copied()
        .filter(|&value| value > base.get())
        .collect();
    candidates.sort_by_key(|&value| value.abs_diff(base.get()));
    candidates
}

/// One priced candidate. The trace is the loop's evidence, and the tests read
/// it: an iteration count, a bracket, and where non-monotonicity showed up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RateStep {
    /// Which phase priced it.
    pub phase: RatePhase,
    /// The quantizer priced.
    pub quantizer: QuantizerChoice,
    /// Its exact emitted size.
    pub bytes: u64,
    /// Whether it fits the target.
    pub feasible: bool,
}

/// Multiplicity counters for one rate search (Opt-V2 acceptance telemetry).
///
/// These are the measured claims behind
/// `rate-probe-multiplicity-down`: Gaborish and the forward DCT pyramid are
/// request-scoped, anchored ladder prices skip entropy alternatives, and
/// finalist-only Full entropy is limited to the Quality/refinement phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RateProbeStats {
    /// Inverse-Gaborish precondition runs (0 or 1 for a single search).
    pub gaborish_preconditions: u32,
    /// Writer prices under [`crate::EntropySearch::Fast`] (ladder + LF fill).
    pub fast_prices: u32,
    /// Exact writer prices attributed to finalist refinement, including its
    /// cheap default-model navigation and Full finalist/correction stores.
    pub full_prices: u32,
    /// Forward-DCT cache hits across every probe.
    pub dct_cache_hits: u64,
    /// Forward-DCT first-time fills across every probe.
    pub dct_cache_misses: u64,
    /// Complete retained candidate set at the end of the search.
    pub candidate_cache_entries: u64,
    /// Retained f32 coefficient payload, excluding collection metadata.
    pub candidate_payload_bytes: u64,
    /// Dense coefficient-arena allocations (one per reserved transform bank).
    pub candidate_allocations: u64,
    /// Cover/CfL builds on the bounded anchored path.
    pub structural_builds: u32,
    /// Reserved legacy counter for approximate probes (zero in the two-anchor path).
    pub sketch_probes: u32,
    /// Exact writer prices on the bounded anchor path, including retained
    /// Store emissions. Hard-capped by the controller.
    pub exact_candidates: u32,
    /// Legacy telemetry: always zero now that production presets cannot enter
    /// the exhaustive controller.
    pub anchor_fallbacks: u32,
    /// Number of fresh cover/CfL rescue sequences (zero or one).
    pub fresh_structure_rescues: u32,
    /// Exact rescue prices (at most two within the one rescue sequence).
    pub rescue_prices: u32,
    /// Exact byte size of the first anchored finalist.
    pub anchor_first_finalist_bytes: u64,
    /// Exact byte size of the one anchored correction, or zero when the
    /// first finalist already satisfied tolerance.
    pub anchor_correction_bytes: u64,
    /// Aggregate Fast planning work, including nested entropy passes.
    pub fast: diagnostics::SearchPhaseDiagnostics,
    /// Aggregate Full planning work, including nested entropy passes.
    pub full: diagnostics::SearchPhaseDiagnostics,
    /// Aggregate Count/Store writer and executor work.
    pub writer: jpxl_encode::vardct::diagnostics::WriterDiagnostics,
}

impl RateProbeStats {
    /// Whether Full-entropy prices stayed inside the refinement budget.
    ///
    /// The Fast ladder owns most of a default 40-price budget; Full refinement
    /// is reserved at most [`full_refinement_reserve`] (≤20). Full may briefly
    /// outnumber Fast on a short ladder that then climbs with Full — that is
    /// still "finalist-only Full", not a full-alternative price loop.
    #[must_use]
    pub fn full_confined_to_refinement(self) -> bool {
        self.full_prices > 0 && self.fast_prices > 0 && self.full_prices <= 20
    }

    /// Whether the forward pyramid was reused across probes.
    #[must_use]
    pub fn dct_cache_reused(self) -> bool {
        self.dct_cache_hits > 0 && self.dct_cache_hits >= self.dct_cache_misses
    }
}

/// What a completed search chose.
#[derive(Debug, Clone)]
pub struct RateOutcome {
    /// The codestream at the chosen quantizer — emitted once, not re-emitted.
    pub codestream: Vec<u8>,
    /// The validated plan it came from.
    pub plan: ValidatedEmissionPlan,
    /// Its exact accounting.
    pub sizing: CodestreamSizing,
    /// The quantizer chosen.
    pub chosen: QuantizerChoice,
    /// The byte target the caller set.
    pub target: u64,
    /// Every candidate priced, in order.
    pub trace: Vec<RateStep>,
    /// Whether the finest ladder rung was still under target, i.e. the target
    /// was unreachably generous and the loop returned the best it can express.
    pub saturated: bool,
    /// Explicit terminal state of the controller that produced this stream.
    pub status: RateStatus,
    /// Multiplicity counters for this search (Opt-V2).
    pub stats: RateProbeStats,
}

impl RateOutcome {
    /// The achieved size.
    #[must_use]
    pub fn achieved(&self) -> u64 {
        self.sizing.total
    }

    /// How many exact writer prices the search paid for.
    #[must_use]
    pub fn iterations(&self) -> usize {
        self.trace.len()
    }

    /// Bytes left unspent under the target.
    #[must_use]
    pub fn undershoot(&self) -> u64 {
        self.target.saturating_sub(self.achieved())
    }

    /// Unspent bytes as a fraction of the target.
    #[must_use]
    pub fn undershoot_fraction(&self) -> f64 {
        if self.target == 0 {
            return 0.0;
        }
        #[allow(
            clippy::cast_precision_loss,
            reason = "byte counts far below f64's exact integer range"
        )]
        let (under, target) = (self.undershoot() as f64, self.target as f64);
        under / target
    }

    /// Priced pairs where the finer quantizer produced the smaller file — the
    /// direct, measured evidence of a non-monotone pocket.
    #[must_use]
    pub fn non_monotone_pairs(&self) -> usize {
        non_monotone_pairs(&self.trace)
    }
}

/// Counts priced pairs that contradict "finer is bigger".
///
/// The loop does not need this — it is written not to care — but the slice's
/// exit criterion says monotonicity is *bracketed, not assumed*, and this is
/// the bracket: a number, from real prices, that says how often the assumption
/// would have been wrong.
#[must_use]
pub fn non_monotone_pairs(trace: &[RateStep]) -> usize {
    let mut count = 0usize;
    for (index, a) in trace.iter().enumerate() {
        for b in trace.iter().skip(index + 1) {
            let (lower, higher) = if a.quantizer.rung <= b.quantizer.rung {
                (a, b)
            } else {
                (b, a)
            };
            if lower.quantizer.rung < higher.quantizer.rung && higher.bytes < lower.bytes {
                count += 1;
            }
        }
    }
    count
}

/// The result of a ladder search over an abstract size function.
#[derive(Debug, Clone)]
pub struct LadderSearch {
    /// The chosen rung: the largest **feasible** size the search ever priced.
    pub rung: Rung,
    /// That candidate's size.
    pub bytes: u64,
    /// Every price, in order.
    pub trace: Vec<RateStep>,
    /// Whether [`Rung::TOP`] itself was feasible.
    pub saturated: bool,
}

/// The generic search: bracket, bisect, fill — over any exact size function.
///
/// Separated from the encoder so that the loop's *control flow* can be tested
/// against injected size functions, including deliberately non-monotone ones,
/// without a single encode. The encoder path passes the real pricer.
///
/// # Errors
///
/// [`PolicyError::TargetUnreachable`] if even [`Rung::FLOOR`] is over budget,
/// [`PolicyError::SearchBudgetExhausted`] if the price budget runs out before
/// any feasible candidate is found, and whatever `price` returns.
pub fn search_ladder(
    start: Rung,
    quant_lf: QuantLf,
    target: u64,
    tolerance: RateTolerance,
    budget: RateSearchBudget,
    price: impl FnMut(QuantizerChoice) -> Result<u64>,
) -> Result<LadderSearch> {
    search_ladder_with(
        start,
        BracketMode::Cold,
        quant_lf,
        target,
        tolerance,
        budget,
        price,
    )
}

/// How the first phase of [`search_ladder`] brackets the crossing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BracketMode {
    /// Double or halve the effective scale from `start` (the request's
    /// default rung is typically far from the crossing).
    Cold,
    /// `start` is believed close to the crossing (a Fast winner being
    /// re-priced under a finer entropy model, or a seed from an anchored
    /// attempt): step the effective scale by an expanding ratio
    /// ([`LOCAL_BRACKET_RATIO`], squared each step) so a crossing within a
    /// few percent is bracketed in one or two prices instead of a doubling
    /// that then needs a dozen bisections back (Phase Q6).
    Local,
}

/// The first step of a [`BracketMode::Local`] bracket, as a ratio of
/// effective scale; each further step squares it (1.1, 1.21, 1.46, 2.14…)
/// so a badly seeded start still reaches a doubling walk in four prices.
pub const LOCAL_BRACKET_RATIO: f64 = 1.1;

/// [`search_ladder`] with an explicit [`BracketMode`].
///
/// # Errors
///
/// As [`search_ladder`].
pub fn search_ladder_with(
    start: Rung,
    bracket: BracketMode,
    quant_lf: QuantLf,
    target: u64,
    tolerance: RateTolerance,
    budget: RateSearchBudget,
    price: impl FnMut(QuantizerChoice) -> Result<u64>,
) -> Result<LadderSearch> {
    let mut search = Search {
        price,
        quant_lf,
        target,
        max_prices: budget.max_prices.max(1),
        trace: Vec::new(),
        best: None,
    };
    search.run(start, bracket, tolerance, budget)
}

/// The next rung up or down from `current` for a [`BracketMode::Local`]
/// step of `ratio` in effective scale (never `current` itself).
fn local_step(current: Rung, ratio: f64, up: bool) -> Rung {
    #[allow(
        clippy::cast_precision_loss,
        reason = "effective scales are far below f64's exact-integer range"
    )]
    let scale = effective_scale(current) as f64;
    let aimed = if up { scale * ratio } else { scale / ratio };
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "finite positive scale, clamped through the ladder constructor"
    )]
    let rung = rung_for_effective_scale(aimed.round().clamp(1.0, u64::MAX as f64) as u64);
    if up {
        rung.max(Rung::new(current.get().saturating_add(1)))
    } else {
        rung.min(Rung::new(current.get().saturating_sub(1)))
    }
}

/// The next rung up or down for a [`BracketMode::Cold`] bracket.
///
/// Cold search promises geometric movement in *effective scale*. That is the
/// same as doubling/halving the rung index below `MAX_GLOBAL_SCALE`, but not
/// above it: Phase Q3 made the post-ceiling ladder dense by interleaving
/// `(global_scale, HfMul)` pairs, so a doubled index can jump more than three
/// times in effective scale and spend several exact prices bisecting back.
///
/// Clamp through [`rung_for_effective_scale`] and force one-rung progress at a
/// representability gap. The latter also keeps the floor/top boundary cases
/// terminating without relying on floating-point rounding.
fn cold_step(current: Rung, up: bool) -> Rung {
    let scale = effective_scale(current);
    let aimed = if up {
        scale.saturating_mul(2)
    } else {
        scale / 2
    }
    .max(1);
    let rung = rung_for_effective_scale(aimed);
    if up {
        rung.max(Rung::new(current.get().saturating_add(1)))
    } else {
        rung.min(Rung::new(current.get().saturating_sub(1)))
    }
}

/// The effective quantizer fineness at a rung: `global_scale * HfMul`.
///
/// The rung *index* is not proportional to the quantizer. Below
/// `MAX_GLOBAL_SCALE` a rung is a `global_scale` step; above it `global_scale`
/// is pinned and each rung is an `HfMul` step worth `MAX_GLOBAL_SCALE` of the
/// lower segment. Interpolating on the index therefore aims badly across that
/// kink — which is exactly where high-rate targets live. This quantity is
/// smooth and strictly increasing across the whole ladder.
pub fn effective_scale(rung: Rung) -> u64 {
    let (scale, mul) = rung_fields(rung);
    u64::from(scale) * u64::from(mul)
}

/// The inverse of [`effective_scale`], rounded down to the finest
/// representable rung whose effective scale does not exceed `scale`.
pub fn rung_for_effective_scale(scale: u64) -> Rung {
    let max = u64::from(MAX_GLOBAL_SCALE);
    if scale <= max {
        return Rung::new(u32::try_from(scale.saturating_sub(1)).unwrap_or(u32::MAX));
    }
    if scale >= max * u64::from(MAX_LADDER_MUL) {
        return Rung::TOP;
    }
    // Segment k covers ((k - 1) * MAX, k * MAX].
    let k = u32::try_from(scale.div_ceil(max)).unwrap_or(MAX_LADDER_MUL);
    let k = k.clamp(2, MAX_LADDER_MUL);
    let gs = u32::try_from(scale / u64::from(k)).unwrap_or(MAX_GLOBAL_SCALE);
    let mut index = MAX_GLOBAL_SCALE;
    let mut j = 2u32;
    while j < k {
        index += segment_len(j);
        j += 1;
    }
    if gs < segment_first_scale(k) {
        // Below this segment's first rung: the previous segment's top is the
        // finest rung not above `scale`.
        return Rung::new(index.saturating_sub(1));
    }
    Rung::new(index + (gs - segment_first_scale(k)))
}

/// Where to aim next inside a bracket: false position on log(size) against
/// log(effective quantizer scale).
///
/// `lo` is feasible, `hi` is not. Returns a rung strictly between them, or
/// `None` when the interpolation cannot be trusted — a degenerate bracket, a
/// non-positive size, or a bracket whose size does not increase — in which
/// case the caller bisects.
///
/// Log-log because size against quantizer scale is near power-law over the
/// range the ladder spans: `global_scale` is a reciprocal, so equal *ratios*
/// are the equal steps.
///
/// Pure, so the stepping rule is testable without encoding anything.
fn interpolated_rung(lo: (Rung, u64), hi: (Rung, u64), target: u64) -> Option<Rung> {
    let (lo_rung, lo_bytes) = lo;
    let (hi_rung, hi_bytes) = hi;
    if hi_rung.get().saturating_sub(lo_rung.get()) <= 1 || lo_bytes == 0 || target == 0 {
        return None;
    }
    if hi_bytes <= lo_bytes {
        // Not increasing across the bracket: a pocket, not a slope.
        return None;
    }
    #[allow(
        clippy::cast_precision_loss,
        reason = "scales and byte counts stay inside f64's exact-integer range"
    )]
    let (lo_x, hi_x, lo_y, hi_y, t_y) = (
        (effective_scale(lo_rung) as f64).ln(),
        (effective_scale(hi_rung) as f64).ln(),
        (lo_bytes as f64).ln(),
        (hi_bytes as f64).ln(),
        (target as f64).ln(),
    );
    let span = hi_y - lo_y;
    if !span.is_finite() || span <= 0.0 {
        return None;
    }
    let fraction = ((t_y - lo_y) / span).clamp(0.0, 1.0);
    let guess_x = lo_x + fraction * (hi_x - lo_x);
    if !guess_x.is_finite() {
        return None;
    }
    let guess_scale = guess_x.exp();
    if !guess_scale.is_finite() || guess_scale < 0.0 {
        return None;
    }
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped into the bracket immediately below"
    )]
    let guess_scale = guess_scale.round() as u64;
    let guess = rung_for_effective_scale(guess_scale).get();
    let (low, high) = (lo_rung.get() + 1, hi_rung.get().checked_sub(1)?);
    if low > high {
        return None;
    }
    Some(Rung::new(guess.clamp(low, high)))
}

/// Aim a bounded exact correction from the default-model navigation bracket.
///
/// The Full finalist can be materially smaller than the default plan even at
/// the same rung. A one-notch correction therefore wastes the exact slot on
/// the flat part of the ladder. Treat the Full/default ratio at the finalist
/// as locally stable, inflate the target in the cheap model's byte space, and
/// interpolate through the already-priced navigation bracket. The candidate
/// is still gated by the exact writer; this only chooses where to spend the
/// bounded correction prices.
fn correction_rung_from_navigation(
    navigation: &LadderSearch,
    finalist: Rung,
    finalist_bytes: u64,
    target: u64,
) -> Rung {
    let minimum = finalist.get().saturating_add(1).min(Rung::TOP.get());
    if finalist == Rung::TOP || finalist_bytes == 0 || navigation.bytes == 0 {
        return Rung::new(minimum);
    }

    let inflate_once = |value: u64| {
        u64::try_from(
            u128::from(value)
                .saturating_mul(u128::from(navigation.bytes))
                .saturating_add(u128::from(finalist_bytes.saturating_sub(1)))
                / u128::from(finalist_bytes),
        )
        .unwrap_or(u64::MAX)
    };
    // The alternative search tends to save a little more at finer rungs than
    // it saves at the finalist. Apply one and a fifth times the observed
    // shrink allowance: one factor accounts for the measured saving, and a
    // small fifth-factor allowance covers local drift without routinely
    // jumping past the target. The exact gate below still rejects any
    // over-target correction.
    let once = inflate_once(target);
    let estimated_default_target = once
        .saturating_add(once.saturating_sub(target) / 5)
        .max(navigation.bytes);

    let upper = navigation
        .trace
        .iter()
        .filter(|step| step.quantizer.rung > finalist && step.bytes >= estimated_default_target)
        .min_by_key(|step| step.quantizer.rung)
        .map(|step| (step.quantizer.rung, step.bytes));
    let Some(upper) = upper else {
        // Phase Q6: a locally bracketed navigation may hold no priced point
        // as far up as the shrink-inflated target (the Full alternatives can
        // save a large share on small frames). Extrapolate along the
        // navigation's own log-log slope instead of notching one rung at a
        // time from the finalist.
        let above = navigation
            .trace
            .iter()
            .filter(|step| step.quantizer.rung > finalist && step.bytes > navigation.bytes)
            .max_by_key(|step| step.quantizer.rung)
            .map(|step| (step.quantizer.rung, step.bytes));
        let extrapolated = above.and_then(|(hi_rung, hi_bytes)| {
            #[allow(
                clippy::cast_precision_loss,
                reason = "scales and byte counts stay inside f64's exact-integer range"
            )]
            let slope = ((hi_bytes as f64).ln() - (navigation.bytes as f64).ln())
                / ((effective_scale(hi_rung) as f64).ln()
                    - (effective_scale(finalist) as f64).ln());
            target_rung_from_slope(
                (finalist, navigation.bytes),
                estimated_default_target,
                slope,
            )
        });
        return extrapolated.map_or(Rung::new(minimum), |rung| {
            Rung::new(rung.get().max(minimum))
        });
    };
    interpolated_rung(
        (finalist, navigation.bytes),
        upper,
        estimated_default_target,
    )
    .map_or(Rung::new(minimum), |rung| {
        Rung::new(rung.get().max(minimum))
    })
}

/// The search's mutable state; a struct because the price counter, the trace
/// and the incumbent are all updated by the same one place.
struct Search<F> {
    price: F,
    quant_lf: QuantLf,
    target: u64,
    max_prices: u32,
    trace: Vec<RateStep>,
    /// The largest feasible size ever priced, with its rung.
    best: Option<(Rung, u64)>,
}

impl<F: FnMut(QuantizerChoice) -> Result<u64>> Search<F> {
    /// Whether another price is affordable.
    fn affordable(&self) -> bool {
        u32::try_from(self.trace.len()).unwrap_or(u32::MAX) < self.max_prices
    }

    /// Prices one rung, records it, and updates the incumbent.
    ///
    /// A rung already present in the trace is returned from the trace rather
    /// than priced again. This matters when an infeasible aimed probe cannot
    /// tighten `hi` and the following mandatory midpoint lands on the same
    /// rung: the cached exact result can tighten the bracket without paying a
    /// second full encode. The trace therefore remains a record of actual
    /// price calls, and its length remains the hard-budget counter.
    ///
    /// The incumbent is the largest feasible size **ever seen**, not the last
    /// one accepted: that single choice is what makes the loop indifferent to
    /// the order the phases visit candidates in, and is what a non-monotone
    /// pocket would otherwise cost. Equal sizes break towards the **finer**
    /// quantizer, because two candidates that cost the same are not equally
    /// good: the finer one spends the same bytes on a better reconstruction.
    fn eval(&mut self, rung: Rung, phase: RatePhase) -> Result<u64> {
        if let Some(step) = self.trace.iter().find(|step| step.quantizer.rung == rung) {
            return Ok(step.bytes);
        }
        let quantizer = QuantizerChoice::at(rung, self.quant_lf)?;
        let bytes = (self.price)(quantizer)?;
        let feasible = bytes <= self.target;
        self.trace.push(RateStep {
            phase,
            quantizer,
            bytes,
            feasible,
        });
        let better = self
            .best
            .is_none_or(|(best_rung, best)| bytes > best || (bytes == best && rung > best_rung));
        if feasible && better {
            self.best = Some((rung, bytes));
        }
        Ok(bytes)
    }

    fn run(
        &mut self,
        start: Rung,
        bracket: BracketMode,
        tolerance: RateTolerance,
        budget: RateSearchBudget,
    ) -> Result<LadderSearch> {
        let slack = tolerance.bytes_for(self.target);
        let mut ratio = LOCAL_BRACKET_RATIO;

        // --- Phase 1: bracket, geometrically ---
        //
        // Doubling and halving the *scale* rather than stepping the index: the
        // quantizer is a reciprocal of `global_scale`, so equal ratios are the
        // equal steps, and a linear walk from the default would take thousands
        // of encodes to reach a low-rate target.
        // `lo` carries its size too: the aimed step below interpolates through
        // both bracket ends, so it needs the feasible end's price, not just
        // its index.
        let mut lo: Option<(Rung, u64)> = None;
        let mut hi: Option<(Rung, u64)> = None;
        let mut floor_bytes = None;

        let first = self.eval(start, RatePhase::Bracket)?;
        if first <= self.target {
            lo = Some((start, first));
            let mut current = start;
            while current < Rung::TOP && self.affordable() {
                let next = match bracket {
                    BracketMode::Cold => cold_step(current, true),
                    BracketMode::Local => {
                        let next = local_step(current, ratio, true);
                        ratio *= ratio;
                        next
                    }
                };
                if next <= current {
                    break;
                }
                let bytes = self.eval(next, RatePhase::Bracket)?;
                if bytes > self.target {
                    hi = Some((next, bytes));
                    break;
                }
                lo = Some((next, bytes));
                current = next;
            }
        } else {
            hi = Some((start, first));
            let mut current = start;
            loop {
                if current == Rung::FLOOR {
                    floor_bytes = Some(hi.map_or(first, |(_, b)| b));
                    break;
                }
                if !self.affordable() {
                    break;
                }
                // `current` is not the floor here, so `div_ceil` halves the
                // *scale* — rung `r` is scale `r + 1`.
                let next = match bracket {
                    BracketMode::Cold => cold_step(current, false),
                    BracketMode::Local => {
                        let next = local_step(current, ratio, false);
                        ratio *= ratio;
                        next
                    }
                };
                let bytes = self.eval(next, RatePhase::Bracket)?;
                if bytes <= self.target {
                    lo = Some((next, bytes));
                    break;
                }
                hi = Some((next, bytes));
                current = next;
            }
        }

        let Some((mut lo, mut lo_bytes)) = lo else {
            if let Some(floor) = floor_bytes {
                return Err(PolicyError::TargetUnreachable {
                    target: self.target,
                    floor,
                });
            }
            return Err(PolicyError::SearchBudgetExhausted {
                prices: self.trace.len(),
            });
        };

        // --- Phase 2: bisect, on wire-legal indices ---
        //
        // The stop is a *bound on what is left*, not a convergence test: with
        // `hi` priced, no candidate between `lo` and `hi` can be worth more
        // than `hi - best` bytes, so once that gap is inside the tolerance the
        // remaining encodes cannot buy the tolerance back.
        let mut take_aimed_step = true;
        let mut aiming_pays = true;
        while let Some((high, high_bytes)) = hi {
            if high.get().saturating_sub(lo.get()) <= 1 || !self.affordable() {
                break;
            }
            let best_bytes = self.best.map_or(0, |(_, b)| b);
            if high_bytes.saturating_sub(best_bytes) <= slack {
                break;
            }
            // The caller asked to land within `slack` of the target, and we
            // already have. Every further price is a full encode spent
            // resolving a rung the tolerance does not care about.
            //
            // Safe by construction: `best` only ever improves, so stopping
            // early cannot change which candidate is returned among those
            // priced — it only declines to price more. Distinct from the bound
            // above, which asks "can anything left still help"; this asks "do
            // we still need help". On a smooth curve the bound needs the
            // bracket narrowed to ~1% in *size*, which costs most of the
            // bisection; this fires as soon as the target is met.
            if self.target.saturating_sub(best_bytes) <= slack {
                break;
            }
            // Alternate an *aimed* step with a plain midpoint.
            //
            // Bisection needs ~log2(width) prices, and on a ~2^17-rung ladder
            // that is more than the budget has left after bracketing — which
            // is how a target gets missed with rungs still available. A
            // false-position step on log(size) usually lands within a rung or
            // two instead.
            //
            // THE SAFETY RULE: an aimed probe may only move `lo` up; it never
            // becomes the new `hi`. That is what makes this provably no worse
            // than bisection on a non-monotone function. `hi` infeasible does
            // NOT imply everything above `hi` is infeasible when the size
            // function has pockets, so a badly-aimed probe that tightened `hi`
            // could fence the incumbent off from a better feasible rung above
            // it — which is exactly how a naive version of this loses
            // `the_loop_survives_a_non_monotone_pocket`'s optimum. Moving `lo`
            // up is always safe: `lo` only ever advances to rungs proved
            // feasible, and `best` is the largest feasible ever priced, so a
            // longer jump can only find the answer sooner, never skip past it.
            //
            // Termination still comes from the interleaved midpoint steps,
            // which do tighten `hi`: the bracket strictly halves every other
            // iteration, so the worst case is twice bisection's step count and
            // the price cap bounds it regardless.
            let midpoint = Rung::new(lo.get() + (high.get() - lo.get()) / 2);
            let (probe, aimed) = if take_aimed_step && aiming_pays {
                match interpolated_rung((lo, lo_bytes), (high, high_bytes), self.target) {
                    Some(guess) if guess != midpoint => (guess, true),
                    _ => (midpoint, false),
                }
            } else {
                (midpoint, false)
            };
            take_aimed_step = !take_aimed_step;

            let bytes = self.eval(probe, RatePhase::Bisect)?;
            if bytes <= self.target {
                lo = probe;
                lo_bytes = bytes;
            } else if aimed {
                // The safety rule means this probe could not tighten `hi`, so
                // it bought nothing: a wasted full encode. Aiming is a bet on
                // the curve being well-behaved here; one lost bet is enough to
                // stop making it, because the same misfit will keep costing a
                // probe per round. Falling back to pure bisection from here
                // makes the worst case "bisection plus one wasted probe"
                // rather than "half the budget spent on probes that do not
                // narrow the bracket" — which is what a high-rate target on a
                // stiff curve was measured doing.
                aiming_pays = false;
            } else {
                hi = Some((probe, bytes));
            }
        }

        // --- Phase 3: the discrete budget fill ---
        //
        // Notch order: `+1, +2, ... +fill_probes` rungs above the incumbent,
        // the finest representable increments there are, and every one of them
        // is priced even after an infeasible one. Stepping past an infeasible
        // notch is the entire point: that is what a non-monotone pocket looks
        // like from inside the loop, and stopping at the first refusal would
        // hand the pocket back.
        // Phase Q6: the fill closes what the tolerance still cares about.
        // Bisection stops as soon as the incumbent is within `slack` of the
        // target, and on a dense ladder — a 4 MP photo at 1 bpp sits near
        // effective scale 20,000-140,000 — a notch is worth tens of bytes, so
        // notching from inside the band paid four full encodes for nothing
        // (about a third of every exhaustive search). The fill still runs
        // when the incumbent is outside the band: the bracket closed to
        // adjacent rungs around a pocket, or the budget ran out early.
        let fill_worthwhile = self
            .best
            .is_some_and(|(_, bytes)| self.target.saturating_sub(bytes) > slack);
        if let Some((base, _)) = self.best.filter(|_| fill_worthwhile) {
            for offset in 1..=budget.fill_probes {
                if !self.affordable() {
                    break;
                }
                let candidate = Rung::new(base.get().saturating_add(offset));
                if candidate <= base {
                    break;
                }
                // A notch bisection already priced costs nothing to skip and a
                // whole encode to repeat.
                if self.trace.iter().any(|s| s.quantizer.rung == candidate) {
                    continue;
                }
                self.eval(candidate, RatePhase::Fill)?;
            }
        }

        let (rung, bytes) = self.best.ok_or(PolicyError::SearchBudgetExhausted {
            prices: self.trace.len(),
        })?;
        let saturated = self
            .trace
            .iter()
            .any(|step| step.quantizer.rung == Rung::TOP && step.feasible);
        Ok(LadderSearch {
            rung,
            bytes,
            trace: core::mem::take(&mut self.trace),
            saturated,
        })
    }
}

/// Persistent state shared across every quantizer probe of one rate search.
///
/// This is the Opt-V2 `PreparedSearch` split: geometry, preconditioned frame,
/// analysis, candidate-forward cache, and request-scoped executor live here.
/// Per-probe work (quantizers, cover rescoring, CfL, entropy, emission) does
/// not rebuild them.
struct PreparedSearch<'a> {
    frame: &'a crate::PreparedFrame,
    transform_frame: &'a crate::PreparedFrame,
    atlas: &'a crate::AnalysisAtlas,
    request: &'a EncodeRequest,
    executor: &'a jpxl_encode::EncodeExecutor,
    fwd_cache: CandidateForwardCache,
    quant_workspace: crate::QuantizationWorkspace,
    stats: RateProbeStats,
}

impl<'a> PreparedSearch<'a> {
    fn plan(
        &mut self,
        quantizer: QuantizerChoice,
        entropy: EntropySearch,
    ) -> Result<ValidatedEmissionPlan> {
        let phase = if entropy == EntropySearch::Fast {
            diagnostics::SearchDiagnosticPhase::Fast
        } else {
            diagnostics::SearchDiagnosticPhase::Full
        };
        diagnostics::with_search_phase(phase, || {
            diagnostics::time_search_plan(|| {
                crate::plan_at_on_with_workspace(
                    self.frame,
                    self.transform_frame,
                    self.atlas,
                    self.request,
                    quantizer,
                    &mut self.fwd_cache,
                    entropy,
                    Some(self.executor),
                    &mut self.quant_workspace,
                )
            })
        })
    }

    #[cfg(feature = "anchor-sketch")]
    fn plan_anchor(
        &mut self,
        quantizer: QuantizerChoice,
        enable_cfl: bool,
        entropy: EntropySearch,
        reuse: AnchorReuse<'_>,
        capture: Option<&mut Option<StructuralAnchor>>,
    ) -> Result<ValidatedEmissionPlan> {
        let phase = match entropy {
            EntropySearch::Fast => diagnostics::SearchDiagnosticPhase::Fast,
            EntropySearch::FinalFast | EntropySearch::Reuse | EntropySearch::Full => {
                diagnostics::SearchDiagnosticPhase::Full
            }
            #[cfg(feature = "g5-bounded-entropy")]
            EntropySearch::BoundedAnchor => diagnostics::SearchDiagnosticPhase::Fast,
            #[cfg(feature = "g5-bounded-entropy")]
            EntropySearch::BoundedFinal => diagnostics::SearchDiagnosticPhase::Full,
        };
        diagnostics::with_search_phase(phase, || {
            diagnostics::time_search_plan(|| {
                crate::plan_at_on_anchor_with_workspace(
                    self.frame,
                    self.transform_frame,
                    self.atlas,
                    self.request,
                    quantizer,
                    enable_cfl,
                    &mut self.fwd_cache,
                    entropy,
                    Some(self.executor),
                    reuse,
                    capture,
                    &mut self.quant_workspace,
                )
            })
        })
    }
}

/// Builds and stores one exact Full-entropy finalist.
///
/// The Quality refinement navigator uses the same trained default model as a
/// Full plan, but does not search the alternative orders, block contexts, or
/// preset assignments. Only this bounded finalist/correction path pays for
/// those alternatives and retains a Store emission.
fn emit_exact_full_candidate(
    prepared: &mut PreparedSearch<'_>,
    quantizer: QuantizerChoice,
) -> Result<(ValidatedEmissionPlan, Emission)> {
    let plan = prepared.plan(quantizer, EntropySearch::Full)?;
    let emission =
        diagnostics::with_search_phase(diagnostics::SearchDiagnosticPhase::Full, || {
            emit_codestream_with_executor(&plan, prepared.executor)
        })?;
    prepared.stats.full_prices = prepared.stats.full_prices.saturating_add(1);
    Ok((plan, emission))
}

/// Whether every G.2.2 `LfQuant` sample fits the legacy signed-16-bit range.
///
/// D.3 permits signed 32-bit Modular samples when
/// `modular_16bit_buffers == false`, and the fixed-quantizer API deliberately
/// retains that full range. The target-rate policy is more conservative:
/// jxl-oxide 0.12.6 wraps `LfQuant` as soon as a sample crosses 32767 even
/// though the header requests 32-bit buffers. LF fill is optional refinement,
/// so it does not adopt a candidate that would lose independent-decoder
/// compatibility merely to spend a small target undershoot.
fn lf_quant_fits_legacy_16bit(plan: &ValidatedEmissionPlan) -> bool {
    plan.plan()
        .quantized
        .lf_groups
        .iter()
        .flat_map(|group| (0..3).flat_map(|channel| group.lf.plane(channel).unwrap_or(&[])))
        .all(|&sample| lf_sample_fits_legacy_16bit(sample))
}

const fn lf_sample_fits_legacy_16bit(sample: i32) -> bool {
    sample >= i16::MIN as i32 && sample <= i16::MAX as i32
}

/// Runs the rate loop over a real frame and returns the chosen codestream.
///
/// Quality requests retain the exact final-price controller and use the
/// default entropy model for refinement navigation. Fast and Balanced requests
/// in a build with the `anchor-sketch` compatibility feature use the bounded
/// two-anchor controller and never enter the exhaustive Quality path.
pub fn search_frame(
    frame: &crate::PreparedFrame,
    atlas: &crate::AnalysisAtlas,
    request: &EncodeRequest,
    target: RateTarget,
) -> Result<RateOutcome> {
    let executor = diagnostics::with_search_phase(diagnostics::SearchDiagnosticPhase::Fast, || {
        request.resources.executor()
    });
    search_frame_with_executor(frame, atlas, request, target, &executor)
}

/// [`search_frame`] on a caller-provided executor, so the worker pool built
/// for the search can also serve the source preparation before it (Phase 38)
/// and is built exactly once per encode.
///
/// # Errors
///
/// As [`search_frame`].
pub fn search_frame_with_executor(
    frame: &crate::PreparedFrame,
    atlas: &crate::AnalysisAtlas,
    request: &EncodeRequest,
    target: RateTarget,
    executor: &jpxl_encode::EncodeExecutor,
) -> Result<RateOutcome> {
    #[cfg(not(feature = "anchor-sketch"))]
    if matches!(
        request.rate_preset,
        crate::request::RateSearchPreset::Fast | crate::request::RateSearchPreset::Balanced
    ) {
        return Err(PolicyError::Unsupported {
            what: "an anchored rate preset without the anchor-sketch crate feature",
        });
    }

    // Inverse-Gaborish is quantizer- and controller-independent. Own it at the
    // request boundary so every anchored probe and its bounded rescue share
    // the same transform frame instead of paying twice.
    let transform_owned = if request.restoration.gaborish {
        Some(crate::prepare_gaborish_frame(frame)?)
    } else {
        None
    };
    let transform_frame = transform_owned.as_ref().unwrap_or(frame);
    let gaborish_preconditions = u32::from(transform_owned.is_some());

    #[cfg(feature = "anchor-sketch")]
    {
        if request.rate_preset == RateSearchPreset::Quality {
            return search_frame_exhaustive(
                frame,
                transform_frame,
                atlas,
                request,
                target,
                executor,
                None,
                gaborish_preconditions,
            );
        }
        search_frame_two_anchor(
            frame,
            transform_frame,
            atlas,
            request,
            target,
            executor,
            gaborish_preconditions,
        )
    }

    #[cfg(not(feature = "anchor-sketch"))]
    {
        search_frame_exhaustive(
            frame,
            transform_frame,
            atlas,
            request,
            target,
            executor,
            None,
            gaborish_preconditions,
        )
    }
}

#[cfg(feature = "anchor-sketch")]
fn two_anchor_target_rung(first: (Rung, u64), second: (Rung, u64), target: u64) -> Option<Rung> {
    let ((lo_rung, lo_bytes), (hi_rung, hi_bytes)) = if first.0 <= second.0 {
        (first, second)
    } else {
        (second, first)
    };
    if lo_rung == hi_rung || lo_bytes == 0 || hi_bytes <= lo_bytes || target == 0 {
        return None;
    }
    #[allow(
        clippy::cast_precision_loss,
        reason = "scales and byte counts stay inside f64's exact-integer range"
    )]
    let (lo_x, hi_x, lo_y, hi_y) = (
        (effective_scale(lo_rung) as f64).ln(),
        (effective_scale(hi_rung) as f64).ln(),
        (lo_bytes as f64).ln(),
        (hi_bytes as f64).ln(),
    );
    let slope = (hi_y - lo_y) / (hi_x - lo_x);
    if !slope.is_finite() || slope <= 0.0 {
        return None;
    }
    target_rung_from_slope((lo_rung, lo_bytes), target, slope)
}

fn target_rung_from_slope(anchor: (Rung, u64), target: u64, slope: f64) -> Option<Rung> {
    if anchor.1 == 0 || target == 0 || !slope.is_finite() || slope <= 0.0 {
        return None;
    }
    #[allow(
        clippy::cast_precision_loss,
        reason = "scales and byte counts stay inside f64's exact-integer range"
    )]
    let guess_x = (effective_scale(anchor.0) as f64).ln()
        + ((target as f64).ln() - (anchor.1 as f64).ln()) / slope;
    let guess_scale = guess_x.exp();
    if !guess_scale.is_finite() || guess_scale <= 0.0 {
        return None;
    }
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "finite positive estimate is clamped by the ladder constructor"
    )]
    let guess = rung_for_effective_scale(guess_scale.round().clamp(1.0, u64::MAX as f64) as u64);
    Some(guess)
}

#[cfg(feature = "anchor-sketch")]
fn two_anchor_correction_rung(
    first: (Rung, u64),
    second: (Rung, u64),
    finalist: (Rung, u64),
    target: u64,
) -> Option<Rung> {
    let ((lo_rung, lo_bytes), (hi_rung, hi_bytes)) = if first.0 <= second.0 {
        (first, second)
    } else {
        (second, first)
    };
    if lo_rung == hi_rung || lo_bytes == 0 || hi_bytes <= lo_bytes {
        return None;
    }
    #[allow(
        clippy::cast_precision_loss,
        reason = "scales and byte counts stay inside f64's exact-integer range"
    )]
    let slope = ((hi_bytes as f64).ln() - (lo_bytes as f64).ln())
        / ((effective_scale(hi_rung) as f64).ln() - (effective_scale(lo_rung) as f64).ln());
    target_rung_from_slope(finalist, target, slope)
}

/// Aim the fresh rescue's one correction from its exact measured size.
///
/// A fresh structural plan may cross the hard ceiling by a small amount even
/// when the reused-structure finalist was well predicted. Jumping directly
/// back to the old feasible anchor in that case can discard most of the byte
/// budget. Re-aim through the already measured anchor slope first; the old
/// feasible rung remains the conservative fallback when the slope is unusable.
#[cfg(feature = "anchor-sketch")]
fn fresh_rescue_correction_rung(
    first: (Rung, u64),
    second: (Rung, u64),
    rescue: (Rung, u64),
    target: u64,
    prior_feasible: Option<Rung>,
) -> Rung {
    two_anchor_correction_rung(first, second, rescue, target)
        .filter(|&rung| rung != rescue.0)
        .unwrap_or_else(|| {
            if rescue.1 > target {
                prior_feasible.unwrap_or(Rung::FLOOR)
            } else {
                Rung::new(rescue.0.get().saturating_add(1))
            }
        })
}

/// Phase Q5 screened three changes to the anchored controller together and
/// kept the mechanism but not the settings:
///
/// * a second-anchor exponent of `1.5` instead of `2.0` (the measured
///   `bytes ~ E^alpha` exponents `1/alpha` are 1.57-1.80 on the busy photos
///   and 0.95-1.05 on the smooth one, so `2.0` overshoots on smooth content);
/// * rebuilding the structure (cover, CfL, entropy model) at the second
///   anchor when the first anchor priced more than a factor of
///   [`STRUCTURE_REBUILD_RATIO`] from the target;
/// * a second exact correction aimed with the local slope between the two
///   exact points already priced.
///
/// Together they removed every exhaustive fallback on the standing corpus —
/// mid2 at 2 bpp Balanced went from 5.9 s to 0.8 s — but the fallback had been
/// a hidden quality tier: its exhaustive path re-plans cover/CfL/entropy at
/// every probe, so the cells that used to fall back (mid2 at 2 bpp, the
/// 20240503_105759 scene at 1 bpp) had been getting Quality-tier output under
/// a Fast/Balanced label. Removing the fallback showed the true tier there:
/// −0.25 SSIMULACRA2 on mid2 2 bpp Balanced and −4.0 on the scene at Fast,
/// which fails the Contract B bound, while every other cell moved within
/// noise (photos +0.02 mean). G3 therefore keeps the proven predictor settings
/// (`2.0`, no second-anchor rebuild, one correction) and replaces the hidden
/// escalation with a cheaper, capped fresh-structure rescue seeded from those
/// anchors.
#[cfg(feature = "anchor-sketch")]
const SECOND_ANCHOR_EXPONENT: f64 = 2.0;

/// How many exact corrections the anchored controller may pay after a
/// finalist that missed the band before entering the bounded fresh-structure
/// rescue (see [`SECOND_ANCHOR_EXPONENT`] for the Phase Q5 screen of 2).
#[cfg(feature = "anchor-sketch")]
const MAX_ANCHOR_CORRECTIONS: u32 = 1;

/// Below this target, entropy-table signaling dominates and changing the
/// anchor model can make the bounded size curve discontinuous. Keep the
/// legacy model rather than spending extra training on an unamortized stream.
#[cfg(all(feature = "anchor-sketch", feature = "g5-bounded-entropy"))]
const MIN_BOUNDED_ENTROPY_TARGET_BYTES: u64 = 4 * 1024;

/// The byte ratio between the first anchor and the target beyond which the
/// anchor's structure is rebuilt at the second anchor. `INFINITY` disables
/// the rebuild (see [`SECOND_ANCHOR_EXPONENT`]); the Phase Q5 screen used 2.
#[cfg(feature = "anchor-sketch")]
const STRUCTURE_REBUILD_RATIO: f64 = f64::INFINITY;

/// Whether the first anchor's structure is too far from the target to carry.
#[cfg(feature = "anchor-sketch")]
fn structure_is_stale(anchor_bytes: u64, target: u64) -> bool {
    #[allow(
        clippy::cast_precision_loss,
        reason = "byte counts are far below f64's exact-integer range"
    )]
    let ratio = target.max(1) as f64 / anchor_bytes.max(1) as f64;
    !(1.0 / STRUCTURE_REBUILD_RATIO..=STRUCTURE_REBUILD_RATIO).contains(&ratio)
}

#[cfg(feature = "anchor-sketch")]
fn second_anchor_rung(start: Rung, anchor_bytes: u64, target: u64) -> Rung {
    #[allow(
        clippy::cast_precision_loss,
        reason = "wire-scale and byte counts are far below f64's exact-integer range"
    )]
    let ratio = target.max(1) as f64 / anchor_bytes.max(1) as f64;
    #[allow(
        clippy::cast_precision_loss,
        reason = "wire-scale is far below f64's exact-integer range"
    )]
    let proposed = effective_scale(start) as f64 * ratio.powf(SECOND_ANCHOR_EXPONENT);
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "finite positive scale is clamped through the ladder constructor"
    )]
    let mut rung = rung_for_effective_scale(proposed.max(1.0).round() as u64);
    if rung == start {
        rung = if anchor_bytes > target {
            Rung::new(start.get() / 2)
        } else {
            Rung::new(start.get().saturating_mul(2).saturating_add(1))
        };
    }
    rung
}

#[cfg(feature = "anchor-sketch")]
fn reuse_entropy(
    plan: ValidatedEmissionPlan,
    entropy: &EntropyPlan,
) -> Result<ValidatedEmissionPlan> {
    let mut inner = plan.into_inner();
    inner.entropy = entropy.clone();
    Ok(jpxl_encode::vardct::validate(inner)?)
}

#[cfg(feature = "anchor-sketch")]
fn bounded_status(
    trace: &[RateStep],
    target: u64,
    slack: u64,
    achieved: u64,
    saturated: bool,
) -> RateStatus {
    if saturated {
        return RateStatus::SaturatedTop;
    }
    if target.saturating_sub(achieved) <= slack {
        return RateStatus::InsideBand;
    }
    let adjacent_crossing = trace.iter().any(|feasible| {
        feasible.feasible
            && trace.iter().any(|infeasible| {
                !infeasible.feasible
                    && feasible
                        .quantizer
                        .rung
                        .get()
                        .abs_diff(infeasible.quantizer.rung.get())
                        == 1
            })
    });
    if adjacent_crossing {
        RateStatus::UnderTargetAdjacentRungs
    } else {
        RateStatus::UnderTargetWorkCap
    }
}

#[cfg(feature = "anchor-sketch")]
fn finish_bounded_outcome(
    prepared: &mut PreparedSearch<'_>,
    trace: Vec<RateStep>,
    candidate: (QuantizerChoice, ValidatedEmissionPlan, Emission),
    target: u64,
    slack: u64,
    rescued: bool,
) -> Result<RateOutcome> {
    let (chosen, plan, emission) = candidate;
    if trace.len() > MAX_BOUNDED_EXACT_PRICES {
        return Err(PolicyError::Unsupported {
            what: "a bounded rate search that exceeded its exact-price cap",
        });
    }
    prepared.stats.exact_candidates = u32::try_from(trace.len()).unwrap_or(u32::MAX);
    prepared.stats.dct_cache_hits = prepared.fwd_cache.hits();
    prepared.stats.dct_cache_misses = prepared.fwd_cache.misses();
    prepared.stats.candidate_cache_entries = prepared.fwd_cache.entries();
    prepared.stats.candidate_payload_bytes = prepared.fwd_cache.payload_bytes();
    prepared.stats.candidate_allocations = prepared.fwd_cache.allocations();
    let aggregate = diagnostics::search_diag();
    prepared.stats.fast = aggregate.fast;
    prepared.stats.full = aggregate.full;
    prepared.stats.writer = jpxl_encode::vardct::diagnostics::snapshot();

    let saturated = trace
        .iter()
        .any(|step| step.quantizer.rung == Rung::TOP && step.feasible);
    let status = if rescued {
        RateStatus::RescuedFreshStructure
    } else {
        bounded_status(&trace, target, slack, emission.sizing.total, saturated)
    };
    Ok(RateOutcome {
        codestream: emission.bytes,
        plan,
        sizing: emission.sizing,
        chosen,
        target,
        trace,
        saturated,
        status,
        stats: prepared.stats,
    })
}

/// One fresh cover/CfL build plus at most one exact correction on that fresh
/// structure. This is the only miss path for production presets; it shares the
/// frame, atlas, executor, DCT cache, and quantization workspace owned by the
/// request and cannot enter the exhaustive controller.
#[cfg(feature = "anchor-sketch")]
#[allow(
    clippy::too_many_arguments,
    reason = "the rescue consumes the bounded controller's already-priced anchors and request state"
)]
fn search_frame_fresh_rescue(
    prepared: &mut PreparedSearch<'_>,
    request: &EncodeRequest,
    target: u64,
    slack: u64,
    mut trace: Vec<RateStep>,
    first: (Rung, u64),
    second: (Rung, u64),
    seed: Rung,
    enable_cfl: bool,
    entropy_search: EntropySearch,
) -> Result<RateOutcome> {
    prepared.stats.fresh_structure_rescues = 1;
    let mut fresh_anchor = None;
    let rescue_quantizer = QuantizerChoice::at(seed, request.quant_lf)?;
    let rescue_plan = prepared.plan_anchor(
        rescue_quantizer,
        enable_cfl,
        entropy_search,
        AnchorReuse::None,
        Some(&mut fresh_anchor),
    )?;
    prepared.stats.structural_builds = prepared.stats.structural_builds.saturating_add(1);
    let rescue_emission =
        diagnostics::with_search_phase(diagnostics::SearchDiagnosticPhase::Full, || {
            emit_codestream_with_executor(&rescue_plan, prepared.executor)
        })?;
    prepared.stats.full_prices = prepared.stats.full_prices.saturating_add(1);
    prepared.stats.rescue_prices = 1;
    let rescue_bytes = rescue_emission.sizing.total;
    trace.push(RateStep {
        phase: RatePhase::Rescue,
        quantizer: rescue_quantizer,
        bytes: rescue_bytes,
        feasible: rescue_bytes <= target,
    });

    let within_target = |bytes: u64| bytes <= target && target.saturating_sub(bytes) <= slack;
    let mut selected = if rescue_bytes <= target {
        Some((rescue_quantizer, rescue_plan, rescue_emission))
    } else {
        drop(rescue_plan);
        drop(rescue_emission);
        None
    };
    if !within_target(rescue_bytes)
        && prepared.stats.rescue_prices < MAX_FRESH_RESCUE_PRICES
        && trace.len() < MAX_BOUNDED_EXACT_PRICES
    {
        let correction_target = target.saturating_sub(slack / 2);
        let prior_feasible_rung = trace
            .iter()
            .rev()
            .find(|step| step.feasible && step.quantizer.rung != rescue_quantizer.rung)
            .map(|step| step.quantizer.rung);
        let correction_rung = fresh_rescue_correction_rung(
            first,
            second,
            (rescue_quantizer.rung, rescue_bytes),
            correction_target,
            prior_feasible_rung,
        );
        let fresh_anchor = fresh_anchor.as_ref().ok_or(PolicyError::Unsupported {
            what: "a fresh rescue that failed to capture its structure",
        })?;
        let correction_quantizer = QuantizerChoice::at(correction_rung, request.quant_lf)?;
        let correction_plan = prepared.plan_anchor(
            correction_quantizer,
            false,
            entropy_search,
            AnchorReuse::CoverAndCfl(fresh_anchor),
            None,
        )?;
        let correction_emission =
            diagnostics::with_search_phase(diagnostics::SearchDiagnosticPhase::Full, || {
                emit_codestream_with_executor(&correction_plan, prepared.executor)
            })?;
        prepared.stats.full_prices = prepared.stats.full_prices.saturating_add(1);
        prepared.stats.rescue_prices = 2;
        let correction_bytes = correction_emission.sizing.total;
        trace.push(RateStep {
            phase: RatePhase::Rescue,
            quantizer: correction_quantizer,
            bytes: correction_bytes,
            feasible: correction_bytes <= target,
        });
        if correction_bytes <= target
            && selected
                .as_ref()
                .is_none_or(|(_, _, emission)| correction_bytes > emission.sizing.total)
        {
            selected = Some((correction_quantizer, correction_plan, correction_emission));
        }
    }

    let Some((chosen, plan, emission)) = selected else {
        let floor = trace.iter().map(|step| step.bytes).min().unwrap_or(0);
        return Err(PolicyError::TargetUnreachable { target, floor });
    };
    finish_bounded_outcome(
        prepared,
        trace,
        (chosen, plan, emission),
        target,
        slack,
        true,
    )
}

#[cfg(feature = "anchor-sketch")]
#[allow(
    clippy::too_many_arguments,
    reason = "controller plumbing keeps request-scoped frame/executor state and \
              bounded-controller state explicit at the one anchored entry point"
)]
fn search_frame_two_anchor(
    frame: &crate::PreparedFrame,
    transform_frame: &crate::PreparedFrame,
    atlas: &crate::AnalysisAtlas,
    request: &EncodeRequest,
    target: RateTarget,
    executor: &jpxl_encode::EncodeExecutor,
    gaborish_preconditions: u32,
) -> Result<RateOutcome> {
    diagnostics::reset_search_diag();
    let target_bytes = target.bytes_for(frame.width(), frame.height());
    let start = QuantizerChoice::from_request(request).rung;

    let stats = RateProbeStats {
        gaborish_preconditions,
        ..RateProbeStats::default()
    };
    let mut prepared = PreparedSearch {
        frame,
        transform_frame,
        atlas,
        request,
        executor,
        fwd_cache: CandidateForwardCache::new(),
        quant_workspace: crate::QuantizationWorkspace::new(),
        stats,
    };

    let first_quantizer = QuantizerChoice::at(start, request.quant_lf)?;
    let (enable_cfl, reuse_entropy_model, final_entropy) = match request.rate_preset {
        RateSearchPreset::Fast => (false, false, EntropySearch::FinalFast),
        RateSearchPreset::Balanced => (true, true, EntropySearch::Reuse),
        RateSearchPreset::Quality => {
            return Err(PolicyError::Unsupported {
                what: "a Quality request routed into the bounded controller",
            });
        }
    };
    let mut captured = None;
    let first_plan = prepared.plan_anchor(
        first_quantizer,
        enable_cfl,
        EntropySearch::Fast,
        AnchorReuse::None,
        Some(&mut captured),
    )?;
    let first_entropy_model = first_plan.plan().entropy.clone();
    prepared.stats.structural_builds = 1;
    let anchor = captured.ok_or(PolicyError::Unsupported {
        what: "an anchor search that failed to capture its structure",
    })?;
    let first_size =
        diagnostics::with_search_phase(diagnostics::SearchDiagnosticPhase::Fast, || {
            diagnostics::with_count_kind(
                jpxl_encode::vardct::diagnostics::CountEmissionKind::Outer,
                || price_codestream_with(&first_plan, prepared.executor),
            )
        })?;
    prepared.stats.fast_prices = 1;
    prepared.stats.exact_candidates = 1;
    // The first anchor has done its only job. Release its quantized views so
    // the request-scoped workspace can reuse the same HF arena for the next
    // probe instead of retaining another frame-sized coefficient payload.
    drop(first_plan);

    let second_rung = second_anchor_rung(start, first_size.total, target_bytes);
    let second_quantizer = QuantizerChoice::at(second_rung, request.quant_lf)?;
    // Phase Q5: when the first anchor priced far from the target, its cover,
    // CfL and entropy model were chosen for a very different operating point,
    // and a finalist that reuses them prices structurally worse than a fresh
    // plan at the same rung (mid2 at 2 bpp: 1.18 MB reused against 1.05 MB
    // fresh, so the anchored path missed its band and fell back to the
    // exhaustive controller). Rebuild the structure at the second anchor in
    // that case and let the finalist reuse *that*; near the target the first
    // anchor is kept, so the standing cells are untouched.
    let rebuild_structure = structure_is_stale(first_size.total, target_bytes);
    #[cfg(feature = "g5-bounded-entropy")]
    let refresh_bounded_entropy = request.rate_preset == RateSearchPreset::Balanced
        && target_bytes >= MIN_BOUNDED_ENTROPY_TARGET_BYTES;
    #[cfg(not(feature = "g5-bounded-entropy"))]
    let refresh_bounded_entropy = false;
    let mut second_captured = None;
    let second_plan = if rebuild_structure {
        prepared.plan_anchor(
            second_quantizer,
            enable_cfl,
            EntropySearch::Fast,
            AnchorReuse::None,
            Some(&mut second_captured),
        )?
    } else {
        #[cfg(feature = "g5-bounded-entropy")]
        let second_entropy = if refresh_bounded_entropy {
            EntropySearch::BoundedAnchor
        } else if reuse_entropy_model {
            EntropySearch::Reuse
        } else {
            EntropySearch::Fast
        };
        #[cfg(not(feature = "g5-bounded-entropy"))]
        let second_entropy = if reuse_entropy_model {
            EntropySearch::Reuse
        } else {
            EntropySearch::Fast
        };
        let plan = prepared.plan_anchor(
            second_quantizer,
            false,
            second_entropy,
            AnchorReuse::CoverAndCfl(&anchor),
            None,
        )?;
        if reuse_entropy_model && !refresh_bounded_entropy {
            reuse_entropy(plan, &first_entropy_model)?
        } else {
            plan
        }
    };
    let second_entropy_model = second_plan.plan().entropy.clone();
    let (anchor, anchor_entropy_model) = match second_captured {
        Some(fresh) => {
            prepared.stats.structural_builds = 2;
            (fresh, second_entropy_model)
        }
        None if refresh_bounded_entropy => (anchor, second_entropy_model),
        None => (anchor, first_entropy_model),
    };
    let second_size =
        diagnostics::with_search_phase(diagnostics::SearchDiagnosticPhase::Fast, || {
            diagnostics::with_count_kind(
                jpxl_encode::vardct::diagnostics::CountEmissionKind::Outer,
                || price_codestream_with(&second_plan, prepared.executor),
            )
        })?;
    prepared.stats.fast_prices = 2;
    prepared.stats.exact_candidates = 2;
    drop(second_plan);
    let mut trace = vec![
        RateStep {
            phase: RatePhase::Bracket,
            quantizer: first_quantizer,
            bytes: first_size.total,
            feasible: first_size.total <= target_bytes,
        },
        RateStep {
            phase: RatePhase::Bracket,
            quantizer: second_quantizer,
            bytes: second_size.total,
            feasible: second_size.total <= target_bytes,
        },
    ];

    // Bias the one-shot prediction a little below the ceiling. Fast retains
    // one eighth of its band. Balanced's bounded entropy model uses one
    // quarter: the release corpus screen found that its smaller exact model
    // otherwise crossed the ceiling by a few hundred bytes and paid a second
    // finalist. The exact over-target check remains the safety net for
    // steep/non-monotone curves.
    #[cfg(feature = "g5-bounded-entropy")]
    let prediction_slack_divisor = if refresh_bounded_entropy { 4 } else { 8 };
    #[cfg(not(feature = "g5-bounded-entropy"))]
    let prediction_slack_divisor = 8;
    let prediction_slack = request
        .rate_preset
        .tolerance(request.tolerance)
        .bytes_for(target_bytes)
        / prediction_slack_divisor;
    let prediction_target = target_bytes.saturating_sub(prediction_slack);
    let slack = request
        .rate_preset
        .tolerance(request.tolerance)
        .bytes_for(target_bytes);
    let Some(finalist_rung) = two_anchor_target_rung(
        (first_quantizer.rung, first_size.total),
        (second_quantizer.rung, second_size.total),
        prediction_target,
    ) else {
        let rescue_entropy = if reuse_entropy_model {
            EntropySearch::Full
        } else {
            final_entropy
        };
        return search_frame_fresh_rescue(
            &mut prepared,
            request,
            target_bytes,
            slack,
            trace,
            (first_quantizer.rung, first_size.total),
            (second_quantizer.rung, second_size.total),
            second_quantizer.rung,
            enable_cfl,
            rescue_entropy,
        );
    };
    let finalist_quantizer = QuantizerChoice::at(finalist_rung, request.quant_lf)?;
    let mut finalist_anchor = None;
    // Reuse the captured structure for the finalist instead of rebuilding
    // cover, forward coefficients, and (for the current anchored presets)
    // CfL. Fast intentionally uses the cheaper fixed-cover/nearest/fast-
    // entropy policy. Balanced keeps hierarchical/trailing structure and
    // retrains only its two frame-ranked hybrid-uint configurations. Quality
    // remains the full-alternative oracle.
    let finalist_entropy = if reuse_entropy_model {
        EntropySearch::Reuse
    } else {
        final_entropy
    };
    #[cfg(feature = "g5-bounded-entropy")]
    let finalist_entropy = if refresh_bounded_entropy {
        EntropySearch::BoundedFinal
    } else {
        finalist_entropy
    };
    let reuse_finalist_entropy = matches!(finalist_entropy, EntropySearch::Reuse);
    // A fresh structural finalist was screened in Phase Q5 (cover, CfL and
    // entropy model re-planned at the predicted rung): +0.10 / -0.03
    // SSIMULACRA2 on the standing 1 bpp cells for +6-22% time, and it made
    // the anchors' curve a worse predictor of the finalist's bytes (mid2 at
    // 1 bpp fell back). The reused structure stays.
    let finalist_plan = prepared.plan_anchor(
        finalist_quantizer,
        false,
        finalist_entropy,
        AnchorReuse::CoverAndCfl(&anchor),
        Some(&mut finalist_anchor),
    )?;
    let finalist = if reuse_finalist_entropy {
        reuse_entropy(finalist_plan, &anchor_entropy_model)?
    } else {
        // Fast trains its own final model. Balanced reaches this arm only for
        // the bounded two-configuration finalist.
        finalist_plan
    };
    let correction_entropy_model = finalist.plan().entropy.clone();
    prepared.stats.structural_builds = if rebuild_structure { 2 } else { 1 };
    let finalist_emission =
        diagnostics::with_search_phase(diagnostics::SearchDiagnosticPhase::Full, || {
            emit_codestream_with_executor(&finalist, prepared.executor)
        })?;
    let finalist_bytes = finalist_emission.sizing.total;
    prepared.stats.full_prices = 1;
    prepared.stats.anchor_first_finalist_bytes = finalist_bytes;
    let within_target =
        |bytes: u64| bytes <= target_bytes && target_bytes.saturating_sub(bytes) <= slack;
    trace.push(RateStep {
        phase: RatePhase::Final,
        quantizer: finalist_quantizer,
        bytes: finalist_bytes,
        feasible: finalist_bytes <= target_bytes,
    });
    let (chosen_quantizer, chosen_plan, emission) = if within_target(finalist_bytes) {
        (finalist_quantizer, finalist, finalist_emission)
    } else {
        // Fast entropy is an upper-bound navigation mode, so its calibrated
        // crossing can miss after the anchored structural finalist. Spend
        // bounded exact corrections on quantizer moves that preserve the
        // finalist's cover and CfL decisions. The first correction aims with
        // the anchors' log-log slope from the finalist; a second (Phase Q5)
        // aims with the *local* slope between the two exact points already
        // priced, which is what a curve that is not one power law over the
        // anchor span needs — before that, mid2 at 2 bpp missed by 1.8% after
        // one correction and paid the ~5 s exhaustive fallback.
        let correction_target = target_bytes.saturating_sub(slack / 2);
        let anchor = finalist_anchor.as_ref().ok_or(PolicyError::Unsupported {
            what: "an anchored finalist that failed to capture its structure",
        })?;
        // The finalist is not the selected result on this branch. Drop its
        // coefficient views before building the correction so that the same
        // reusable arena can serve the final probe.
        drop(finalist);
        let mut previous = (finalist_quantizer.rung, finalist_bytes);
        let mut last: Option<(Rung, u64)> = None;
        let mut selected = None;
        for attempt in 0..MAX_ANCHOR_CORRECTIONS {
            let correction_rung = match last {
                None => two_anchor_correction_rung(
                    (first_quantizer.rung, first_size.total),
                    (second_quantizer.rung, second_size.total),
                    previous,
                    correction_target,
                ),
                Some(exact) => {
                    two_anchor_correction_rung(exact, previous, exact, correction_target)
                }
            };
            let Some(correction_rung) = correction_rung else {
                break;
            };
            if trace
                .iter()
                .any(|step| step.quantizer.rung == correction_rung)
            {
                break;
            }
            let correction_quantizer = QuantizerChoice::at(correction_rung, request.quant_lf)?;
            let correction_entropy = if reuse_entropy_model || refresh_bounded_entropy {
                EntropySearch::Reuse
            } else {
                finalist_entropy
            };
            let correction_plan = prepared.plan_anchor(
                correction_quantizer,
                true,
                correction_entropy,
                AnchorReuse::CoverAndCfl(anchor),
                None,
            )?;
            let correction = if matches!(correction_entropy, EntropySearch::Reuse) {
                reuse_entropy(correction_plan, &correction_entropy_model)?
            } else {
                correction_plan
            };
            let correction_emission =
                diagnostics::with_search_phase(diagnostics::SearchDiagnosticPhase::Full, || {
                    emit_codestream_with_executor(&correction, prepared.executor)
                })?;
            let correction_bytes = correction_emission.sizing.total;
            prepared.stats.full_prices = 2 + attempt;
            prepared.stats.anchor_correction_bytes = correction_bytes;
            trace.push(RateStep {
                phase: RatePhase::Final,
                quantizer: correction_quantizer,
                bytes: correction_bytes,
                feasible: correction_bytes <= target_bytes,
            });
            if within_target(correction_bytes) {
                selected = Some((correction_quantizer, correction, correction_emission));
                break;
            }
            last = Some(previous);
            previous = (correction_quantizer.rung, correction_bytes);
        }
        let Some(selected) = selected else {
            // Seed the sole fresh-structure rescue from the crossing estimate
            // between the exact points nearest the target. Production
            // presets never call the exhaustive reference controller.
            let rescue_seed = match last {
                Some(exact) => two_anchor_correction_rung(exact, previous, previous, target_bytes)
                    .unwrap_or(previous.0),
                None => two_anchor_correction_rung(
                    (first_quantizer.rung, first_size.total),
                    (second_quantizer.rung, second_size.total),
                    previous,
                    target_bytes,
                )
                .unwrap_or(previous.0),
            };
            let rescue_entropy = if reuse_entropy_model {
                EntropySearch::Full
            } else {
                final_entropy
            };
            return search_frame_fresh_rescue(
                &mut prepared,
                request,
                target_bytes,
                slack,
                trace,
                (first_quantizer.rung, first_size.total),
                (second_quantizer.rung, second_size.total),
                rescue_seed,
                enable_cfl,
                rescue_entropy,
            );
        };
        selected
    };

    if emission.sizing.total != trace.last().map_or(0, |step| step.bytes) {
        return Err(PolicyError::Unsupported {
            what: "a selected anchored emission whose recorded size changed",
        });
    }
    finish_bounded_outcome(
        &mut prepared,
        trace,
        (chosen_quantizer, chosen_plan, emission),
        target_bytes,
        slack,
        false,
    )
}

/// Exact target controller retained as the explicit Quality reference path.
///
/// Two entropy pricing modes share the price budget:
///
/// 1. **Fast ladder** — default I.2.2 entropy, no alternatives. Exact writer
///    sizes, but an *upper bound* on the Full plan at the same quantizer.
///    Geometric bracket / bisect / fill find an approximate incumbent cheaply.
/// 2. **Finalist refinement** — re-runs the same ladder control flow from the
///    Fast incumbent with the trained default model, then pays for slice-18
///    Full alternatives only at the finalist and a bounded exact correction
///    window.
///    Because Full only shrinks, the default-model feasible set is a safe
///    navigation lower bound and the final exact gate never crosses the target.
///
/// The returned codestream is a Full emission from the finalist path. Fast
/// probe sizes stay in the trace as ladder guidance; refinement navigation and
/// exact finalist/correction steps are [`RatePhase::Final`].
///
/// # Errors
///
/// As [`search_ladder`], plus anything the planner or writer refuses.
#[allow(
    clippy::too_many_arguments,
    reason = "the explicit Quality controller receives request-scoped \
              transform state rather than rebuilding it"
)]
fn search_frame_exhaustive(
    frame: &crate::PreparedFrame,
    transform_frame: &crate::PreparedFrame,
    atlas: &crate::AnalysisAtlas,
    request: &EncodeRequest,
    target: RateTarget,
    executor: &jpxl_encode::EncodeExecutor,
    seed: Option<Rung>,
    gaborish_preconditions: u32,
) -> Result<RateOutcome> {
    diagnostics::reset_search_diag();
    let target_bytes = target.bytes_for(frame.width(), frame.height());
    // Phase Q6: an anchored attempt that missed its band still knows where
    // the crossing is to within a few percent; start there with a local
    // bracket instead of the request's default rung with a doubling one.
    let (start, bracket) = match seed {
        Some(seed) => (seed, BracketMode::Local),
        None => (
            QuantizerChoice::from_request(request).rung,
            BracketMode::Cold,
        ),
    };

    let stats = RateProbeStats {
        gaborish_preconditions,
        ..RateProbeStats::default()
    };
    let mut prepared = PreparedSearch {
        frame,
        transform_frame,
        atlas,
        request,
        executor,
        fwd_cache: CandidateForwardCache::new(),
        quant_workspace: crate::QuantizationWorkspace::new(),
        stats,
    };

    let max_prices = request.budget.rate.max_prices.max(1);
    let full_reserve = full_refinement_reserve(max_prices);
    let fast_cap = max_prices.saturating_sub(full_reserve).max(1);
    let mut fast_budget = request.budget.rate;
    fast_budget.max_prices = fast_cap;

    // Incumbent under Fast pricing. Sizes are upper bounds; only the quantizer
    // identity carries into Full refinement.
    let mut kept: Option<(Rung, QuantLf, CodestreamSizing)> = None;

    let mut search = search_ladder_with(
        start,
        bracket,
        request.quant_lf,
        target_bytes,
        request.tolerance,
        fast_budget,
        |quantizer| {
            let plan = prepared.plan(quantizer, EntropySearch::Fast)?;
            let sizing =
                diagnostics::with_search_phase(diagnostics::SearchDiagnosticPhase::Fast, || {
                    diagnostics::with_count_kind(
                        jpxl_encode::vardct::diagnostics::CountEmissionKind::Outer,
                        || price_codestream_with(&plan, prepared.executor),
                    )
                })?;
            prepared.stats.fast_prices = prepared.stats.fast_prices.saturating_add(1);
            let bytes = sizing.total;
            let better = kept.as_ref().is_none_or(|&(rung, _, ref prev)| {
                bytes > prev.total || (bytes == prev.total && quantizer.rung > rung)
            });
            if bytes <= target_bytes && better {
                // The tuple carries the *request's* LF balance, not the wire
                // `quant_lf` (which `QuantizerChoice::at` couples above the
                // ceiling and would couple twice if fed back).
                kept = Some((quantizer.rung, request.quant_lf, sizing));
            }
            Ok(bytes)
        },
    )?;

    // --- quant_lf secondary fill at the Fast winning rung ---
    //
    // Only when the ladder undershoots past tolerance and Fast budget remains.
    // Probes stay near the caller's intended LF/HF balance.
    let slack = request.tolerance.bytes_for(target_bytes);
    let fast_cap_usize = usize::try_from(fast_cap).unwrap_or(1);
    let lf_probes = usize::try_from(request.budget.rate.lf_fill_probes).unwrap_or(0);
    let lf_fill = kept
        .as_ref()
        .map(|(rung, base_lf, sizing)| (*rung, *base_lf, sizing.total));
    if let Some((rung, base_lf, base_bytes)) = lf_fill {
        let undershoot = target_bytes.saturating_sub(base_bytes);
        if lf_probes > 0 && undershoot > slack {
            let mut tried = 0usize;
            for lf_val in quant_lf_fill_candidates(base_lf) {
                if search.trace.len() >= fast_cap_usize || tried >= lf_probes {
                    break;
                }
                let Ok(lf) = QuantLf::new(lf_val) else {
                    continue;
                };
                let Ok(quantizer) = QuantizerChoice::at(rung, lf) else {
                    continue;
                };
                let Ok(plan) = prepared.plan(quantizer, EntropySearch::Fast) else {
                    continue;
                };
                tried += 1;
                if !lf_quant_fits_legacy_16bit(&plan) {
                    continue;
                }
                let Ok(sizing) = diagnostics::with_search_phase(
                    diagnostics::SearchDiagnosticPhase::Fast,
                    || {
                        diagnostics::with_count_kind(
                            jpxl_encode::vardct::diagnostics::CountEmissionKind::Outer,
                            || price_codestream_with(&plan, prepared.executor),
                        )
                    },
                ) else {
                    continue;
                };
                prepared.stats.fast_prices = prepared.stats.fast_prices.saturating_add(1);
                let bytes = sizing.total;
                let feasible = bytes <= target_bytes;
                search.trace.push(RateStep {
                    phase: RatePhase::LfFill,
                    quantizer,
                    bytes,
                    feasible,
                });
                let better = kept.as_ref().is_none_or(|(_, _, prev)| bytes > prev.total);
                if feasible && better {
                    kept = Some((rung, lf, sizing));
                }
            }
        }
    }

    let (rung, quant_lf, _fast_sizing) = kept.ok_or(PolicyError::TargetUnreachable {
        target: target_bytes,
        floor: search.bytes,
    })?;
    if rung != search.rung {
        // Ladder incumbent rung must match; LF fill keeps the same rung.
        return Err(PolicyError::Unsupported {
            what: "a rate search whose incumbent and cached emission disagree",
        });
    }

    // Full refinement: use the default trained model to navigate the same
    // ladder, then pay for the expensive Full alternatives only at the
    // finalist. Full only adopts alternatives that strictly shrink the
    // default plan, so a feasible navigator finalist remains feasible after
    // the exact Full gate. A bounded correction window is reserved when the
    // first exact finalist leaves too much of the target unspent.
    let used = u32::try_from(search.trace.len()).unwrap_or(u32::MAX);
    // The reserve is a CAP on refinement, not a floor. Keep a small exact
    // correction window when the budget permits: one finalist gate and up to
    // five follow-up candidates (Phase Q0b widened this from three: once the
    // LF-group sections stopped padding every total, the Full alternatives'
    // shrink relative to the FinalFast navigator grew on small frames and the
    // bracket needed one more interpolation to land inside the tolerance).
    // The remaining slots navigate with FinalFast, which still runs an exact
    // writer Count but avoids every nested entropy alternative.
    let remaining = max_prices.saturating_sub(used).min(full_reserve).max(1);
    let exact_slots = remaining.min(6);
    let navigation_budget = remaining.saturating_sub(exact_slots);
    let navigation = if navigation_budget > 0 {
        let mut navigation_budget_request = request.budget.rate;
        navigation_budget_request.max_prices = navigation_budget;
        navigation_budget_request.lf_fill_probes = 0;
        // Phase Q6: the Fast winner is next to the crossing, so bracket
        // locally instead of doubling away from it and bisecting back.
        Some(search_ladder_with(
            rung,
            BracketMode::Local,
            quant_lf,
            target_bytes,
            request.tolerance,
            navigation_budget_request,
            |quantizer| {
                let plan = prepared.plan(quantizer, EntropySearch::FinalFast)?;
                let sizing = diagnostics::with_search_phase(
                    diagnostics::SearchDiagnosticPhase::Full,
                    || {
                        diagnostics::with_count_kind(
                            jpxl_encode::vardct::diagnostics::CountEmissionKind::Outer,
                            || price_codestream_with(&plan, prepared.executor),
                        )
                    },
                )?;
                prepared.stats.full_prices = prepared.stats.full_prices.saturating_add(1);
                Ok(sizing.total)
            },
        )?)
    } else {
        None
    };

    if let Some(navigation) = navigation.as_ref() {
        for step in &navigation.trace {
            search.trace.push(RateStep {
                phase: RatePhase::Final,
                quantizer: step.quantizer,
                bytes: step.bytes,
                feasible: step.feasible,
            });
        }
    }

    let finalist_rung = navigation.as_ref().map_or(rung, |result| result.rung);
    let finalist = QuantizerChoice::at(finalist_rung, quant_lf)?;
    let (finalist_plan, finalist_emission) = emit_exact_full_candidate(&mut prepared, finalist)?;
    let finalist_bytes = finalist_emission.sizing.total;
    search.trace.push(RateStep {
        phase: RatePhase::Final,
        quantizer: finalist,
        bytes: finalist_bytes,
        feasible: finalist_bytes <= target_bytes,
    });
    if finalist_bytes > target_bytes {
        return Err(PolicyError::TargetUnreachable {
            target: target_bytes,
            floor: finalist_bytes,
        });
    }

    let slack = request.tolerance.bytes_for(target_bytes);
    let pixels = u64::from(frame.width()) * u64::from(frame.height());
    let mut topoff_prices = quality_topoff_prices(request.rate_preset, finalist.rung, pixels);
    let mut chosen = finalist;
    let mut plan = finalist_plan;
    let mut emission = finalist_emission;
    if exact_slots >= 2
        && finalist.rung < Rung::TOP
        && (target_bytes.saturating_sub(finalist_bytes) > slack || topoff_prices > 0)
    {
        // Full alternatives can only shrink the default navigation price, so
        // aim through the already-priced navigation bracket after accounting
        // for the finalist's observed shrink ratio. Corrections then narrow
        // an exact feasible/infeasible pair when the first aim crosses the
        // target, or re-aim from the new exact point when it does not. The
        // window is bounded by the shared price budget.
        let mut lower_rung = finalist.rung;
        let mut lower_bytes = finalist_bytes;
        let mut upper = None;
        let mut correction_rung = navigation.as_ref().map_or(
            Rung::new(finalist.rung.get().saturating_add(1)),
            |navigation| {
                correction_rung_from_navigation(
                    navigation,
                    finalist.rung,
                    finalist_bytes,
                    target_bytes,
                )
            },
        );
        for _ in 0..exact_slots.saturating_sub(1) {
            // Once the exact incumbent is already inside the requested band,
            // Quality may spend at most one saved post-ceiling price to try a
            // denser legal rung. An over-target top-off is simply discarded;
            // it never opens a second correction window.
            if target_bytes.saturating_sub(emission.sizing.total) <= slack {
                if topoff_prices == 0 {
                    break;
                }
                topoff_prices -= 1;
            }
            if correction_rung <= lower_rung
                || search
                    .trace
                    .iter()
                    .any(|step| step.quantizer.rung == correction_rung)
            {
                break;
            }
            let correction = QuantizerChoice::at(correction_rung, quant_lf)?;
            let (correction_plan, correction_emission) =
                emit_exact_full_candidate(&mut prepared, correction)?;
            let correction_bytes = correction_emission.sizing.total;
            search.trace.push(RateStep {
                phase: RatePhase::Final,
                quantizer: correction,
                bytes: correction_bytes,
                feasible: correction_bytes <= target_bytes,
            });
            if correction_bytes <= target_bytes {
                if correction_bytes > emission.sizing.total
                    || (correction_bytes == emission.sizing.total && correction.rung > chosen.rung)
                {
                    chosen = correction;
                    plan = correction_plan;
                    emission = correction_emission;
                }
                lower_rung = correction.rung;
                lower_bytes = correction_bytes;
            } else {
                drop(correction_plan);
                drop(correction_emission);
                upper = Some((correction.rung, correction_bytes));
            }

            if target_bytes.saturating_sub(emission.sizing.total) <= slack && topoff_prices == 0 {
                break;
            }
            correction_rung = if let Some(upper) = upper {
                let gap = upper.0.get().saturating_sub(lower_rung.get());
                if gap <= 1 {
                    break;
                }
                interpolated_rung((lower_rung, lower_bytes), upper, target_bytes)
                    .unwrap_or_else(|| Rung::new(lower_rung.get() + gap / 2))
            } else {
                navigation.as_ref().map_or(
                    Rung::new(lower_rung.get().saturating_add(1)),
                    |navigation| {
                        correction_rung_from_navigation(
                            navigation,
                            lower_rung,
                            lower_bytes,
                            target_bytes,
                        )
                    },
                )
            };
        }
    }

    prepared.stats.dct_cache_hits = prepared.fwd_cache.hits();
    prepared.stats.dct_cache_misses = prepared.fwd_cache.misses();
    prepared.stats.candidate_cache_entries = prepared.fwd_cache.entries();
    prepared.stats.candidate_payload_bytes = prepared.fwd_cache.payload_bytes();
    prepared.stats.candidate_allocations = prepared.fwd_cache.allocations();
    let aggregate = diagnostics::search_diag();
    prepared.stats.fast = aggregate.fast;
    prepared.stats.full = aggregate.full;
    prepared.stats.writer = jpxl_encode::vardct::diagnostics::snapshot();

    let saturated = search
        .trace
        .iter()
        .any(|step| step.quantizer.rung == Rung::TOP && step.feasible);
    Ok(RateOutcome {
        codestream: emission.bytes,
        chosen,
        sizing: emission.sizing,
        plan,
        target: target_bytes,
        trace: search.trace,
        saturated,
        status: RateStatus::ExhaustiveReference,
        stats: prepared.stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn budget() -> RateSearchBudget {
        RateSearchBudget::default()
    }

    fn quant_lf() -> QuantLf {
        QuantLf::new(16).expect("legal")
    }

    #[cfg(feature = "anchor-sketch")]
    #[test]
    fn bounded_terminal_statuses_are_explicit() {
        let feasible = RateStep {
            phase: RatePhase::Final,
            quantizer: QuantizerChoice::at(Rung::new(10), quant_lf()).expect("legal"),
            bytes: 80,
            feasible: true,
        };
        let adjacent_over = RateStep {
            phase: RatePhase::Final,
            quantizer: QuantizerChoice::at(Rung::new(11), quant_lf()).expect("legal"),
            bytes: 105,
            feasible: false,
        };
        let distant_over = RateStep {
            phase: RatePhase::Final,
            quantizer: QuantizerChoice::at(Rung::new(20), quant_lf()).expect("legal"),
            bytes: 105,
            feasible: false,
        };

        assert_eq!(
            bounded_status(&[feasible], 100, 25, 80, false),
            RateStatus::InsideBand
        );
        assert_eq!(
            bounded_status(&[feasible, adjacent_over], 100, 5, 80, false),
            RateStatus::UnderTargetAdjacentRungs
        );
        assert_eq!(
            bounded_status(&[feasible, distant_over], 100, 5, 80, false),
            RateStatus::UnderTargetWorkCap
        );
        assert_eq!(
            bounded_status(&[feasible], 100, 5, 80, true),
            RateStatus::SaturatedTop
        );
    }

    #[cfg(feature = "anchor-sketch")]
    #[test]
    fn a_slightly_over_target_fresh_rescue_does_not_collapse_to_the_old_anchor() {
        // Reproduces the geometry of a 6000x4000 Balanced 3 bpp miss: the
        // fresh rescue crossed the 9 MB ceiling by only 0.51%, but the old
        // fallback repriced the 2.85 MB starting anchor and returned 2.76 MB.
        let first = (Rung::new(32_767), 2_851_569);
        let second = (Rung::new(159_899), 13_494_991);
        let rescue = (Rung::new(112_887), 9_045_597);
        let correction =
            fresh_rescue_correction_rung(first, second, rescue, 8_910_000, Some(first.0));

        assert!(
            correction > first.0,
            "must not discard most of the rate budget"
        );
        assert!(
            correction < rescue.0,
            "an over-target rescue must move coarser"
        );
    }

    #[test]
    fn lf_fill_only_spends_probes_on_finer_lf_values() {
        let base = quant_lf();
        let candidates = quant_lf_fill_candidates(base);
        assert_eq!(
            candidates,
            vec![24, 32, 48, 64, 96, 128, 192, 256, 512, 1024]
        );
        assert!(candidates.iter().all(|&value| value > base.get()));
    }

    #[test]
    fn legacy_lf_compatibility_range_is_exactly_signed_16_bit() {
        assert!(lf_sample_fits_legacy_16bit(-32_768));
        assert!(lf_sample_fits_legacy_16bit(32_767));
        assert!(!lf_sample_fits_legacy_16bit(-32_769));
        assert!(!lf_sample_fits_legacy_16bit(32_768));
    }

    #[test]
    fn every_rung_is_representable_and_the_ladder_is_ordered() {
        let lf = quant_lf();
        for index in [
            0u32,
            1,
            1000,
            MAX_GLOBAL_SCALE - 1,
            MAX_GLOBAL_SCALE,
            LADDER_LEN - 1,
        ] {
            let q = QuantizerChoice::at(Rung::new(index), lf).expect("representable");
            assert!(q.global_scale.get() >= 1 && q.global_scale.get() <= MAX_GLOBAL_SCALE);
            assert!(q.hf_mul.get() >= 1);
        }
        // The HF step is `1 / (global_scale * HfMul)`: strictly finer with the
        // rung, across the join between the two segments.
        let mut previous = 0u64;
        for index in [
            0u32,
            1,
            MAX_GLOBAL_SCALE - 2,
            MAX_GLOBAL_SCALE - 1,
            MAX_GLOBAL_SCALE,
            MAX_GLOBAL_SCALE + 1,
            LADDER_LEN - 1,
        ] {
            let q = QuantizerChoice::at(Rung::new(index), lf).expect("representable");
            let product = u64::from(q.global_scale.get()) * u64::from(q.hf_mul.get());
            assert!(product > previous, "rung {index} did not go finer");
            previous = product;
        }
    }

    #[test]
    fn lf_coupling_follows_the_segment_up_to_the_bound_and_never_below_the_request() {
        let four = QuantLf::new(4).expect("legal");
        assert_eq!(coupled_quant_lf(four, 1).get(), 4);
        assert_eq!(coupled_quant_lf(four, 2).get(), 8);
        assert_eq!(coupled_quant_lf(four, 4).get(), 16);
        assert_eq!(coupled_quant_lf(four, 5).get(), 16, "the bound holds");
        assert_eq!(coupled_quant_lf(four, 65).get(), 16);
        let sixteen = QuantLf::new(16).expect("legal");
        assert_eq!(
            coupled_quant_lf(sixteen, 3).get(),
            16,
            "already at the bound"
        );
        let coarse_lf = QuantLf::new(3).expect("legal");
        assert_eq!(
            coupled_quant_lf(coarse_lf, 6).get(),
            15,
            "largest multiple within the bound"
        );
        let finer_than_bound = QuantLf::new(64).expect("legal");
        assert_eq!(
            coupled_quant_lf(finer_than_bound, 4).get(),
            64,
            "left as asked"
        );
    }

    /// Phase Q3: above the ceiling every rung is a distinct, strictly finer
    /// effective scale, the inverse map is exact, the segments join without a
    /// gap, and `quant_lf` follows the segment up to the coupling cap.
    #[test]
    fn the_upper_ladder_is_dense_invertible_and_lf_coupled() {
        let lf = quant_lf();
        let mut previous = effective_scale(Rung::new(MAX_GLOBAL_SCALE - 1));
        assert_eq!(previous, u64::from(MAX_GLOBAL_SCALE));
        for index in MAX_GLOBAL_SCALE..LADDER_LEN {
            let rung = Rung::new(index);
            let scale = effective_scale(rung);
            assert!(scale > previous, "rung {index} did not go finer");
            // Adjacent rungs of segment k differ by exactly k; the first rung
            // of a segment sits at most k above the previous segment's top.
            let (gs, mul) = rung_fields(rung);
            assert!((1..=MAX_GLOBAL_SCALE).contains(&gs));
            if gs == segment_first_scale(mul) {
                assert!(scale - previous <= u64::from(mul), "rung {index}");
            } else {
                assert_eq!(scale - previous, u64::from(mul), "rung {index}");
            }
            assert_eq!(rung_for_effective_scale(scale), rung, "rung {index}");
            // Any scale strictly between two rungs rounds down to the lower.
            if scale - previous > 1 {
                assert_eq!(rung_for_effective_scale(scale - 1), Rung::new(index - 1));
            }
            let q = QuantizerChoice::at(rung, lf).expect("representable");
            assert_eq!(q.hf_mul.get(), mul);
            assert_eq!(q.quant_lf, coupled_quant_lf(lf, mul));
            assert!(q.quant_lf.get() >= lf.get());
            assert!(q.quant_lf.get() <= LF_COUPLING_MAX_QUANT_LF.max(lf.get()));
            previous = scale;
        }
        // The top is unchanged from the whole-multiple ladder: 65 x MAX.
        assert_eq!(
            effective_scale(Rung::TOP),
            u64::from(MAX_GLOBAL_SCALE) * u64::from(MAX_LADDER_MUL)
        );
        assert_eq!(rung_fields(Rung::TOP), (MAX_GLOBAL_SCALE, MAX_LADDER_MUL));
        // Below the ceiling nothing moved: rung == global_scale - 1, HfMul 1,
        // the request's own quant_lf.
        for index in [0u32, 4_096, MAX_GLOBAL_SCALE - 1] {
            let q = QuantizerChoice::at(Rung::new(index), lf).expect("representable");
            assert_eq!(q.global_scale.get(), index + 1);
            assert_eq!(q.hf_mul.get(), 1);
            assert_eq!(q.quant_lf, lf);
        }
        // For the target policy's `quant_lf` 4 the LF step
        // (1 / (global_scale * quant_lf)) is continuous across the ceiling
        // and the first coupled segments: the product at the first rung above
        // the ceiling is at least the ceiling's own, and it keeps rising
        // through the segments the bound admits.
        let four = QuantLf::new(4).expect("legal");
        let lf_product =
            |q: QuantizerChoice| u64::from(q.global_scale.get()) * u64::from(q.quant_lf.get());
        let ceiling = QuantizerChoice::at(Rung::new(MAX_GLOBAL_SCALE - 1), four).expect("ok");
        let above = QuantizerChoice::at(Rung::new(MAX_GLOBAL_SCALE), four).expect("ok");
        assert!(lf_product(above) >= lf_product(ceiling));
        assert_eq!(above.quant_lf.get(), 8);
        // A request already at the bound is left alone above the ceiling.
        let sixteen = QuantizerChoice::at(Rung::new(MAX_GLOBAL_SCALE), lf).expect("ok");
        assert_eq!(sixteen.quant_lf, lf);
    }

    #[test]
    fn a_rung_out_of_range_saturates_rather_than_wrapping() {
        assert_eq!(Rung::new(u32::MAX), Rung::TOP);
        assert_eq!(Rung::for_global_scale(0), Rung::FLOOR);
        assert_eq!(Rung::for_global_scale(1), Rung::FLOOR);
        assert_eq!(Rung::for_global_scale(32_768).get(), 32_767);
    }

    #[test]
    fn cold_bracketing_is_geometric_in_effective_scale_across_the_ceiling() {
        let below = Rung::for_global_scale(65_536);
        let doubled = cold_step(below, true);
        assert_eq!(effective_scale(below), 65_536);
        assert_eq!(effective_scale(doubled), 131_072);
        assert_eq!(cold_step(doubled, false), below);

        // The old raw-index doubling landed much farther into the third
        // HfMul segment. Pin that the cold step follows the documented scale
        // geometry rather than accidentally returning to index geometry.
        let old_index_step = Rung::new(below.get().saturating_mul(2).saturating_add(1));
        assert!(effective_scale(old_index_step) > effective_scale(doubled));

        assert_eq!(cold_step(Rung::FLOOR, false), Rung::FLOOR);
        assert_eq!(cold_step(Rung::TOP, true), Rung::TOP);
    }

    #[test]
    fn an_already_priced_rung_is_reused_without_spending_the_budget_twice() {
        let calls = core::cell::Cell::new(0u32);
        let mut search = Search {
            price: |quantizer: QuantizerChoice| {
                calls.set(calls.get().saturating_add(1));
                Ok(u64::from(quantizer.global_scale.get()))
            },
            quant_lf: quant_lf(),
            target: u64::MAX,
            max_prices: 2,
            trace: Vec::new(),
            best: None,
        };
        let rung = Rung::for_global_scale(4_096);
        let first = search.eval(rung, RatePhase::Bracket).expect("first price");
        let reused = search.eval(rung, RatePhase::Bisect).expect("cached price");
        assert_eq!(reused, first);
        assert_eq!(calls.get(), 1);
        assert_eq!(search.trace.len(), 1);
        assert_eq!(
            search.trace.first().map(|step| step.phase),
            Some(RatePhase::Bracket)
        );
    }

    #[test]
    fn quality_topoff_is_one_price_and_bounded_to_moderate_frames() {
        assert_eq!(
            quality_topoff_prices(
                crate::request::RateSearchPreset::Quality,
                Rung::new(MAX_GLOBAL_SCALE),
                QUALITY_TOPOFF_MAX_PIXELS,
            ),
            1
        );
        assert_eq!(
            quality_topoff_prices(
                crate::request::RateSearchPreset::Quality,
                Rung::new(MAX_GLOBAL_SCALE - 1),
                QUALITY_TOPOFF_MAX_PIXELS,
            ),
            0
        );
        assert_eq!(
            quality_topoff_prices(
                crate::request::RateSearchPreset::Balanced,
                Rung::new(MAX_GLOBAL_SCALE),
                QUALITY_TOPOFF_MAX_PIXELS,
            ),
            0
        );
        assert_eq!(
            quality_topoff_prices(
                crate::request::RateSearchPreset::Quality,
                Rung::new(MAX_GLOBAL_SCALE),
                QUALITY_TOPOFF_MAX_PIXELS + 1,
            ),
            0,
            "large frames keep the requested tolerance as their stop"
        );
    }

    /// A smooth, strictly increasing size function: the easy case, and the one
    /// that pins the tolerance claim.
    fn smooth(q: QuantizerChoice) -> u64 {
        let scale = u64::from(q.global_scale.get()) * u64::from(q.hf_mul.get());
        100 + scale * 3
    }

    #[test]
    fn the_loop_lands_under_a_target_and_close_to_it() {
        for target in [200u64, 1_000, 50_000, 98_404] {
            let result = search_ladder(
                Rung::for_global_scale(32_768),
                quant_lf(),
                target,
                RateTolerance::default(),
                budget(),
                |q| Ok(smooth(q)),
            )
            .expect("a feasible rung exists");
            assert!(result.bytes <= target, "target {target}: {}", result.bytes);
            let slack = RateTolerance::default().bytes_for(target);
            assert!(
                target - result.bytes <= slack.max(3),
                "target {target}: landed {} ({} short, slack {slack})",
                result.bytes,
                target - result.bytes
            );
            assert!(
                result.trace.len() <= budget().max_prices as usize,
                "target {target}: {} prices",
                result.trace.len()
            );
        }
    }

    /// The aimed step, in isolation: false position must land inside the
    /// bracket and near the target end, not at the middle, on a power-law
    /// curve like the real one.
    #[test]
    fn an_aimed_step_targets_the_crossing_not_the_middle() {
        let lo = (Rung::new(100), 1_000u64);
        let hi = (Rung::new(100_000), 31_600_000u64);
        let probe = interpolated_rung(lo, hi, 2_000).expect("a usable interpolation");
        let midpoint = 100 + (100_000 - 100) / 2;
        assert!(
            probe.get() > 100 && probe.get() < 100_000,
            "must stay strictly inside the bracket, got {}",
            probe.get()
        );
        assert!(
            probe.get() < midpoint / 10,
            "should aim near the target end, got {} against midpoint {midpoint}",
            probe.get()
        );
    }

    /// Degenerate brackets decline rather than returning something that would
    /// re-price a known point or stall the loop.
    #[test]
    fn an_aimed_step_declines_what_it_cannot_aim_at() {
        // Adjacent rungs: nothing strictly between them.
        assert_eq!(
            interpolated_rung((Rung::new(10), 100), (Rung::new(11), 200), 150),
            None
        );
        // Size does not increase across the bracket: a pocket, not a slope.
        assert_eq!(
            interpolated_rung((Rung::new(10), 500), (Rung::new(1_000), 400), 450),
            None
        );
        // Zero size is not logarithm-able.
        assert_eq!(
            interpolated_rung((Rung::new(10), 0), (Rung::new(1_000), 400), 200),
            None
        );
    }

    /// **The safety rule.** An aimed probe may move `lo` up but must never
    /// become the new `hi`, because `hi` infeasible does not imply everything
    /// above `hi` is infeasible on a non-monotone curve.
    ///
    /// The fixture makes that concrete: a narrow infeasible spike sits exactly
    /// where a false-position step aims, and the real optimum lies *above* the
    /// spike. A loop that let the aimed probe tighten `hi` would fence itself
    /// below the spike and return the smaller answer; this one must still
    /// reach past it.
    #[test]
    fn an_aimed_probe_that_lands_in_a_spike_does_not_fence_off_the_rungs_above_it() {
        let target = 10_000u64;
        // Smooth and crossing the target near scale 3300, except for a narrow
        // band of scales that price far above target — a spike the aimed step
        // is likely to land in, since it aims at the crossing.
        let price = |q: QuantizerChoice| -> u64 {
            let scale = u64::from(q.global_scale.get()) * u64::from(q.hf_mul.get());
            let base = 100 + scale * 3;
            if (3_000..3_200).contains(&scale) {
                base + 50_000
            } else {
                base
            }
        };
        let result = search_ladder(
            Rung::for_global_scale(32_768),
            quant_lf(),
            target,
            RateTolerance {
                bytes: 0,
                fraction: 0.0,
            },
            budget(),
            |q| Ok(price(q)),
        )
        .expect("feasible rungs exist");

        assert!(result.bytes <= target, "never over: {}", result.bytes);
        let optimum = (0..LADDER_LEN)
            .filter_map(|i| QuantizerChoice::at(Rung::new(i), quant_lf()).ok())
            .map(price)
            .filter(|&b| b <= target)
            .max()
            .expect("a feasible rung");
        assert_eq!(
            result.bytes, optimum,
            "an aimed probe landing in the spike fenced off the better rungs above it"
        );
        // And the answer really is above the spike, or the fixture proves
        // nothing.
        assert!(
            optimum > 3_200 * 3,
            "fixture is wrong: the optimum should sit above the spike"
        );
    }

    /// The point of the change: on a smooth curve the search now reaches the
    /// target in materially fewer full encodes than blind bisection needed.
    /// Targets are drawn from the `global_scale` segment, as
    /// `the_loop_lands_under_a_target_and_close_to_it` does; the separate cold
    /// bracket test pins the dense post-ceiling segment's scale geometry.
    #[test]
    fn aimed_stepping_reaches_the_target_in_fewer_prices() {
        for target in [1_000u64, 50_000, 98_404] {
            let mut prices = 0usize;
            let result = search_ladder(
                Rung::for_global_scale(32_768),
                quant_lf(),
                target,
                RateTolerance::default(),
                budget(),
                |q| {
                    prices += 1;
                    Ok(smooth(q))
                },
            )
            .expect("feasible");
            assert!(result.bytes <= target, "never over: {}", result.bytes);
            let slack = RateTolerance::default().bytes_for(target);
            assert!(
                target - result.bytes <= slack.max(3),
                "target {target}: landed {} short",
                target - result.bytes
            );
            assert!(
                prices <= 16,
                "target {target}: took {prices} prices; blind bisection of this \
                 ladder needs far more"
            );
        }
    }

    #[test]
    fn a_target_below_the_floor_is_refused_with_the_floor_in_the_error() {
        let err = search_ladder(
            Rung::for_global_scale(32_768),
            quant_lf(),
            50,
            RateTolerance::default(),
            budget(),
            |q| Ok(smooth(q)),
        )
        .expect_err("103 bytes is the floor");
        assert!(matches!(
            err,
            PolicyError::TargetUnreachable { floor: 103, .. }
        ));
    }

    #[test]
    fn a_target_above_the_ceiling_saturates_at_the_finest_rung() {
        let result = search_ladder(
            Rung::for_global_scale(32_768),
            quant_lf(),
            u64::MAX / 4,
            RateTolerance::default(),
            budget(),
            |q| Ok(smooth(q)),
        )
        .expect("the ceiling is feasible");
        assert!(result.saturated);
        assert_eq!(result.rung, Rung::TOP);
    }

    #[test]
    fn saturation_reports_a_feasible_top_even_when_a_coarser_pocket_is_closer() {
        let result = search_ladder(
            Rung::FLOOR,
            quant_lf(),
            1_000,
            RateTolerance {
                bytes: 0,
                fraction: 0.0,
            },
            budget(),
            |q| Ok(if q.rung == Rung::TOP { 1 } else { 900 }),
        )
        .expect("every rung is feasible");
        assert!(result.rung < Rung::TOP, "the coarser 900-byte pocket wins");
        assert_eq!(result.bytes, 900);
        assert!(
            result
                .trace
                .iter()
                .any(|step| step.quantizer.rung == Rung::TOP && step.feasible)
        );
        assert!(
            result.saturated,
            "saturation describes the priced top rung, not the incumbent identity"
        );
    }

    /// **The bracketing test.** A size function with a deep, wide pocket: a
    /// band of rungs below the target's crossing point prices *above* target,
    /// and a band above it prices *below*. A loop that assumed monotonicity
    /// would either return the pocket's coarse edge or fail to terminate.
    ///
    /// What is asserted is what the loop promises: it terminates inside the
    /// price budget, it never returns an infeasible candidate, and it does at
    /// least as well as the best candidate it actually priced.
    #[test]
    fn the_loop_survives_a_non_monotone_pocket() {
        let target = 10_000u64;
        // Base is smooth and crosses the target at global_scale == 3300.
        // The pocket: rungs whose scale is in [2000, 4000) get 4000 bytes
        // *added* if the scale is odd and 4000 *removed* if it is even, so the
        // size sawtooths across the target line for two thousand consecutive
        // representable values.
        let price = |q: QuantizerChoice| -> u64 {
            let scale = u64::from(q.global_scale.get()) * u64::from(q.hf_mul.get());
            let base = 100 + scale * 3;
            if (2_000..4_000).contains(&scale) {
                if scale % 2 == 0 {
                    base.saturating_sub(4_000)
                } else {
                    base + 4_000
                }
            } else {
                base
            }
        };

        let result = search_ladder(
            Rung::for_global_scale(32_768),
            quant_lf(),
            target,
            RateTolerance::default(),
            budget(),
            |q| Ok(price(q)),
        )
        .expect("feasible rungs exist");

        assert!(result.bytes <= target, "returned {} bytes", result.bytes);
        assert!(
            result.trace.len() <= budget().max_prices as usize,
            "{} prices",
            result.trace.len()
        );
        // Never worse than the best candidate the loop actually looked at.
        let best_seen = result
            .trace
            .iter()
            .filter(|s| s.feasible)
            .map(|s| s.bytes)
            .max()
            .expect("a feasible price");
        assert_eq!(result.bytes, best_seen);

        // It really did walk into the pocket: the prices it took contradict
        // "finer is bigger" — which is the assumption a naive bisection would
        // have made.
        assert!(
            non_monotone_pairs(&result.trace) > 0,
            "the pocket was never entered: {:?}",
            result.trace
        );

        // And, on this function, it found the *global* optimum: the largest
        // feasible size anywhere on the ladder. Brute force is affordable here
        // precisely because the size function is injected — which is the
        // reason the search is generic over one.
        let lf = quant_lf();
        let optimum = (0..LADDER_LEN)
            .filter_map(|index| QuantizerChoice::at(Rung::new(index), lf).ok())
            .map(price)
            .filter(|&bytes| bytes <= target)
            .max()
            .expect("a feasible rung");
        assert_eq!(
            result.bytes, optimum,
            "the pocket cost the loop the optimum"
        );
    }

    /// The fill phase must step *past* an infeasible notch, not stop at it.
    #[test]
    fn the_fill_recovers_a_notch_hidden_behind_an_infeasible_one() {
        let target = 10_000u64;
        // Flat below 5000 so bisection lands exactly there, then: +1 is over
        // target, +2 is under it and larger. Only a fill that keeps probing
        // after a refusal can find it.
        let price = |q: QuantizerChoice| -> u64 {
            match q.global_scale.get() {
                s if s <= 5_000 => 9_000,
                5_001 => 10_001,
                5_002 => 9_500,
                _ => 20_000,
            }
        };
        let result = search_ladder(
            Rung::for_global_scale(32_768),
            quant_lf(),
            target,
            RateTolerance {
                bytes: 0,
                fraction: 0.0,
            },
            budget(),
            |q| Ok(price(q)),
        )
        .expect("feasible");
        assert_eq!(result.bytes, 9_500, "the fill missed the hidden notch");
        assert_eq!(result.rung.get(), 5_001);
    }

    #[test]
    fn the_price_budget_is_a_hard_cap() {
        let mut prices = 0usize;
        let budget = RateSearchBudget {
            max_prices: 6,
            fill_probes: 4,
            lf_fill_probes: 0,
        };
        let result = search_ladder(
            Rung::for_global_scale(32_768),
            quant_lf(),
            10_000,
            RateTolerance {
                bytes: 0,
                fraction: 0.0,
            },
            budget,
            |q| {
                prices += 1;
                Ok(smooth(q))
            },
        )
        .expect("a feasible rung was found inside the cap");
        assert_eq!(prices, result.trace.len());
        assert!(prices <= 6, "{prices} prices past a cap of 6");
        assert!(result.bytes <= 10_000);
    }

    #[cfg(feature = "anchor-sketch")]
    #[test]
    fn two_anchor_rate_curve_interpolates_and_extrapolates() {
        let lo = Rung::for_global_scale(100);
        let hi = Rung::for_global_scale(400);
        assert_eq!(
            two_anchor_target_rung((lo, 1_000), (hi, 4_000), 2_000),
            Some(Rung::for_global_scale(200))
        );
        assert_eq!(
            two_anchor_target_rung((lo, 1_000), (hi, 4_000), 8_000),
            Some(Rung::for_global_scale(800))
        );
        assert_eq!(
            two_anchor_target_rung((hi, 4_000), (lo, 1_000), 500),
            Some(Rung::for_global_scale(50))
        );
        assert_eq!(
            two_anchor_correction_rung(
                (lo, 1_000),
                (hi, 4_000),
                (Rung::for_global_scale(200), 1_600),
                2_000,
            ),
            Some(Rung::for_global_scale(250))
        );
    }

    #[cfg(feature = "anchor-sketch")]
    #[test]
    fn second_anchor_moves_toward_the_target() {
        let start = Rung::for_global_scale(32_768);
        let coarser = second_anchor_rung(start, 20_000, 10_000);
        let finer = second_anchor_rung(start, 5_000, 10_000);
        assert!(coarser < start);
        assert!(finer > start);
    }
}
