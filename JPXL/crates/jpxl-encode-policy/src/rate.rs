//! The exact rate loop (`docs/PLAN.md` slice 14, `Encoder-plan1.md` milestone
//! 4): pick a quantizer that lands a byte target.
//!
//! # Three properties this loop is built around
//!
//! **1. Every candidate is priced exactly.** There is no size model. A
//! candidate's size is [`jpxl_encode::vardct::price_codestream`], which is the
//! real writer run into a scratch buffer — see that module for why a second
//! implementation of "how big would this be" is a paired-bug shape. One
//! iteration therefore costs one full encode, and the loop is written to spend
//! as few of them as it can.
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
//! the only way to go finer. Per-varblock `HfMul` is milestone 7.

use jpxl_encode::vardct::ids::{GlobalScale, HfMul, MAX_GLOBAL_SCALE, QuantLf};
use jpxl_encode::vardct::size::CodestreamSizing;
use jpxl_encode::vardct::{
    Emission, ValidatedEmissionPlan, emit_codestream_with, price_codestream,
};

use crate::error::{PolicyError, Result};
use crate::request::{EncodeRequest, RateSearchBudget, RateTarget, RateTolerance};
use crate::{CandidateForwardCache, EntropySearch};

/// How many `HfMul` rungs extend the ladder above `global_scale`'s ceiling.
///
/// Each one halves nothing and multiplies everything: rung `MAX_GLOBAL_SCALE +
/// k` uses `HfMul = k + 2`, so the top of the ladder is an HF step 65 times
/// finer than the finest `global_scale` alone can reach. Past that the
/// coefficients are integers in the millions and the LF plane — which `HfMul`
/// does not touch — is the accuracy floor anyway.
pub const HF_MUL_RUNGS: u32 = 64;

/// One past the last ladder index.
pub const LADDER_LEN: u32 = MAX_GLOBAL_SCALE + HF_MUL_RUNGS;

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
        let (scale, mul) = if rung.get() < MAX_GLOBAL_SCALE {
            (rung.get() + 1, 1)
        } else {
            (MAX_GLOBAL_SCALE, rung.get() - MAX_GLOBAL_SCALE + 2)
        };
        Ok(Self {
            rung,
            global_scale: GlobalScale::new(scale)?,
            hf_mul: HfMul::new(mul)?,
            quant_lf,
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
    /// Full-entropy refinement after the Fast ladder: re-price the Fast
    /// incumbent and climb/bisect/fill with the real entropy alternatives so
    /// the achieved size is not an overestimate-guided undershoot.
    Final,
}

/// How many prices to hold back from the Fast ladder for Full refinement.
///
/// Fast entropy overestimates size (it skips alternatives that shrink the
/// stream). The Fast ladder therefore lands coarser than Full would, and the
/// Full pass must be allowed a geometric climb + bisect — not a one-notch
/// walk — or the residual undershoot blows the 1% contract.
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
/// request-scoped, Fast ladder prices skip entropy alternatives, and Full
/// entropy + kept emissions are limited to the refinement phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RateProbeStats {
    /// Inverse-Gaborish precondition runs (0 or 1 for a single search).
    pub gaborish_preconditions: u32,
    /// Writer prices under [`crate::EntropySearch::Fast`] (ladder + LF fill).
    pub fast_prices: u32,
    /// Writer prices under [`crate::EntropySearch::Full`] (Final refinement).
    pub full_prices: u32,
    /// Forward-DCT cache hits across every probe.
    pub dct_cache_hits: u64,
    /// Forward-DCT first-time fills across every probe.
    pub dct_cache_misses: u64,
}

impl RateProbeStats {
    /// Whether Full-entropy prices stayed inside the refinement budget.
    ///
    /// The Fast ladder owns most of a default 40-price budget; Full refinement
    /// is reserved at most [`full_refinement_reserve`] (≤20). Full may briefly
    /// outnumber Fast on a short ladder that then climbs with Full — that is
    /// still "finalist-only Full", not a full-price loop.
    #[must_use]
    pub fn full_confined_to_refinement(self) -> bool {
        self.full_prices > 0
            && self.fast_prices > 0
            && self.full_prices <= 20
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
    /// Multiplicity counters for this search (Opt-V2).
    pub stats: RateProbeStats,
}

impl RateOutcome {
    /// The achieved size.
    #[must_use]
    pub fn achieved(&self) -> u64 {
        self.sizing.total
    }

    /// How many exact prices — i.e. full encodes — the search paid for.
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
    let mut search = Search {
        price,
        quant_lf,
        target,
        max_prices: budget.max_prices.max(1),
        trace: Vec::new(),
        best: None,
    };
    search.run(start, tolerance, budget)
}

/// The effective quantizer fineness at a rung: `global_scale * HfMul`.
///
/// The rung *index* is not proportional to the quantizer. Below
/// `MAX_GLOBAL_SCALE` a rung is a `global_scale` step; above it `global_scale`
/// is pinned and each rung is an `HfMul` step worth `MAX_GLOBAL_SCALE` of the
/// lower segment. Interpolating on the index therefore aims badly across that
/// kink — which is exactly where high-rate targets live. This quantity is
/// smooth and strictly increasing across the whole ladder.
fn effective_scale(rung: Rung) -> u64 {
    if rung.get() < MAX_GLOBAL_SCALE {
        u64::from(rung.get()) + 1
    } else {
        u64::from(MAX_GLOBAL_SCALE) * u64::from(rung.get() - MAX_GLOBAL_SCALE + 2)
    }
}

/// The inverse of [`effective_scale`], rounded down to a representable rung.
fn rung_for_effective_scale(scale: u64) -> Rung {
    let max = u64::from(MAX_GLOBAL_SCALE);
    if scale <= max {
        return Rung::new(u32::try_from(scale.saturating_sub(1)).unwrap_or(u32::MAX));
    }
    let mul = (scale / max).max(2);
    Rung::new(u32::try_from(max + mul - 2).unwrap_or(u32::MAX))
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
    /// The incumbent is the largest feasible size **ever seen**, not the last
    /// one accepted: that single choice is what makes the loop indifferent to
    /// the order the phases visit candidates in, and is what a non-monotone
    /// pocket would otherwise cost. Equal sizes break towards the **finer**
    /// quantizer, because two candidates that cost the same are not equally
    /// good: the finer one spends the same bytes on a better reconstruction.
    fn eval(&mut self, rung: Rung, phase: RatePhase) -> Result<u64> {
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
        tolerance: RateTolerance,
        budget: RateSearchBudget,
    ) -> Result<LadderSearch> {
        let slack = tolerance.bytes_for(self.target);

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
                let next = Rung::new(current.get().saturating_mul(2).saturating_add(1));
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
                let next = Rung::new(current.get().div_ceil(2).saturating_sub(1));
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
        if let Some((base, _)) = self.best {
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
        Ok(LadderSearch {
            rung,
            bytes,
            trace: core::mem::take(&mut self.trace),
            saturated: rung == Rung::TOP,
        })
    }
}

/// Persistent state shared across every quantizer probe of one rate search.
///
/// This is the Opt-V2 `PreparedSearch` split: geometry, preconditioned frame,
/// analysis, and the candidate-forward cache live here. Per-probe work
/// (quantizers, cover rescoring, CfL, entropy, emission) does not rebuild them.
struct PreparedSearch<'a> {
    frame: &'a crate::PreparedFrame,
    transform_frame: &'a crate::PreparedFrame,
    atlas: &'a crate::AnalysisAtlas,
    request: &'a EncodeRequest,
    fwd_cache: CandidateForwardCache,
    stats: RateProbeStats,
}

impl<'a> PreparedSearch<'a> {
    fn plan(
        &mut self,
        quantizer: QuantizerChoice,
        entropy: EntropySearch,
    ) -> Result<ValidatedEmissionPlan> {
        crate::plan_at_on(
            self.frame,
            self.transform_frame,
            self.atlas,
            self.request,
            quantizer,
            &mut self.fwd_cache,
            entropy,
        )
    }
}

/// Runs the rate loop over a real frame and returns the chosen codestream.
///
/// Two entropy pricing modes share the price budget:
///
/// 1. **Fast ladder** — default I.2.2 entropy, no alternatives. Exact writer
///    sizes, but an *upper bound* on the Full plan at the same quantizer.
///    Geometric bracket / bisect / fill find an approximate incumbent cheaply.
/// 2. **Full refinement** — re-runs the same ladder control flow from the Fast
///    incumbent with Full entropy (slice-18 alternatives). Because Full only
///    shrinks, the Fast feasible set is a lower bound on the Full feasible
///    set; the refinement climbs/bisects to spend residual undershoot.
///
/// The returned codestream is a Full emission kept during refinement, so the
/// winner is not encoded twice. Fast probe sizes stay in the trace as ladder
/// guidance; only [`RatePhase::Final`] steps are Full-priced.
///
/// # Errors
///
/// As [`search_ladder`], plus anything the planner or writer refuses.
pub fn search_frame(
    frame: &crate::PreparedFrame,
    atlas: &crate::AnalysisAtlas,
    request: &EncodeRequest,
    target: RateTarget,
) -> Result<RateOutcome> {
    let target_bytes = target.bytes_for(frame.width(), frame.height());
    let start = QuantizerChoice::from_request(request).rung;

    // Inverse-Gaborish is quantizer-independent: once per request, not per probe.
    let precond_owned;
    let mut stats = RateProbeStats::default();
    let transform_frame: &crate::PreparedFrame = if request.restoration.gaborish {
        precond_owned = crate::prepare_gaborish_frame(frame)?;
        stats.gaborish_preconditions = 1;
        &precond_owned
    } else {
        frame
    };
    let mut prepared = PreparedSearch {
        frame,
        transform_frame,
        atlas,
        request,
        fwd_cache: CandidateForwardCache::new(),
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

    let mut search = search_ladder(
        start,
        request.quant_lf,
        target_bytes,
        request.tolerance,
        fast_budget,
        |quantizer| {
            let plan = prepared.plan(quantizer, EntropySearch::Fast)?;
            let sizing = price_codestream(&plan)?;
            prepared.stats.fast_prices = prepared.stats.fast_prices.saturating_add(1);
            let bytes = sizing.total;
            let better = kept.as_ref().is_none_or(|&(rung, _, ref prev)| {
                bytes > prev.total || (bytes == prev.total && quantizer.rung > rung)
            });
            if bytes <= target_bytes && better {
                kept = Some((quantizer.rung, quantizer.quant_lf, sizing));
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
                let Ok(sizing) = price_codestream(&plan) else {
                    continue;
                };
                prepared.stats.fast_prices = prepared.stats.fast_prices.saturating_add(1);
                tried += 1;
                let bytes = sizing.total;
                let feasible = bytes <= target_bytes;
                search.trace.push(RateStep {
                    phase: RatePhase::LfFill,
                    quantizer,
                    bytes,
                    feasible,
                });
                let better = kept
                    .as_ref()
                    .is_none_or(|(_, _, prev)| bytes > prev.total);
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

    // Full refinement: same ladder control flow from the Fast incumbent, with
    // remaining budget. Fast overestimates, so Full at that rung is still
    // under target and the geometric climb reclaims undershoot.
    let used = u32::try_from(search.trace.len()).unwrap_or(u32::MAX);
    // The reserve is a CAP on Full refinement, not a floor. Before aimed
    // stepping the Fast ladder nearly always spent its whole allowance, so
    // `max_prices - used` was already about the reserve and the distinction
    // never showed. Now that Fast can finish in a handful of prices, handing
    // Full everything Fast did not use would spend the saving on more
    // expensive-entropy probes instead of banking it — and would break
    // `full_confined_to_refinement`'s "finalist-only Full" invariant.
    let remaining = max_prices.saturating_sub(used).min(full_reserve).max(1);
    let mut full_budget = request.budget.rate;
    full_budget.max_prices = remaining;
    full_budget.lf_fill_probes = 0;

    let mut full_best: Option<(QuantizerChoice, ValidatedEmissionPlan, Emission)> = None;
    let full = search_ladder(
        rung,
        quant_lf,
        target_bytes,
        request.tolerance,
        full_budget,
        |quantizer| {
            let plan = prepared.plan(quantizer, EntropySearch::Full)?;
            // Keep the emission of the incumbent so the winner is not re-encoded.
            // Finalist Full emits use the request's EncodeResources (parallel groups).
            let emission = emit_codestream_with(&plan, prepared.request.resources)?;
            prepared.stats.full_prices = prepared.stats.full_prices.saturating_add(1);
            let bytes = emission.sizing.total;
            let better = full_best.as_ref().is_none_or(|(prev_q, _, prev)| {
                bytes > prev.sizing.total
                    || (bytes == prev.sizing.total && quantizer.rung > prev_q.rung)
            });
            if bytes <= target_bytes && better {
                full_best = Some((quantizer, plan, emission));
            }
            Ok(bytes)
        },
    )?;

    for step in full.trace {
        search.trace.push(RateStep {
            phase: RatePhase::Final,
            quantizer: step.quantizer,
            bytes: step.bytes,
            feasible: step.feasible,
        });
    }

    let (chosen, plan, emission) = full_best.ok_or(PolicyError::TargetUnreachable {
        target: target_bytes,
        floor: full.bytes,
    })?;

    prepared.stats.dct_cache_hits = prepared.fwd_cache.hits();
    prepared.stats.dct_cache_misses = prepared.fwd_cache.misses();

    Ok(RateOutcome {
        codestream: emission.bytes,
        chosen,
        sizing: emission.sizing,
        plan,
        target: target_bytes,
        trace: search.trace,
        saturated: full.saturated || chosen.rung == Rung::TOP,
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
    fn a_rung_out_of_range_saturates_rather_than_wrapping() {
        assert_eq!(Rung::new(u32::MAX), Rung::TOP);
        assert_eq!(Rung::for_global_scale(0), Rung::FLOOR);
        assert_eq!(Rung::for_global_scale(1), Rung::FLOOR);
        assert_eq!(Rung::for_global_scale(32_768).get(), 32_767);
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
    /// `the_loop_lands_under_a_target_and_close_to_it` does. Above
    /// `MAX_GLOBAL_SCALE` the ladder steps by whole `HfMul` multiples, so
    /// adjacent rungs differ by a large factor and *no* search can land within
    /// a 1% tolerance there — that is the ladder's own resolution, not a
    /// property of the stepping rule, and
    /// `a_target_above_the_ceiling_saturates_at_the_finest_rung` covers it.
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
}
