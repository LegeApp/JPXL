# Scheduler unification and multi-fidelity search plan

Status: proposed 2026-08-28. Authoritative planning state lives in AKR
(`@jpegxl-rs.work.ladder-navigator-unification`,
`@jpegxl-rs.work.multifidelity-quality-search`); this document is the
narrative reference for those records.

Source: two advisor memos (2026-08-28) reviewed against the current tree at
`e895c32`. Their factual claims were verified against the code before this
plan was written:

- `rate.rs` and `quality.rs` each carry their own copy of the same log-domain
  crossing mathematics (`interpolated_rung` / `two_anchor_target_rung` /
  `target_rung_from_slope` in `rate.rs:749,1386,863`; `log_loss_crossing` /
  `extrapolated_step` / `geometric_step` in `quality.rs:683,982,710`).
- `quality.rs:41` imports `Rung`, `QuantizerChoice`, `effective_scale`,
  `rung_for_effective_scale` from `rate.rs` — the quality controller depends
  on the byte-rate controller just to name a quantizer rung.
- `predict_v2` (`quality_prediction.rs:50`) already returns a calibrated
  interval (`interval_low`/`interval_high`), `local_loss_exponent`,
  `saturation_risk`, and OOD flags, but the navigation budget is still
  preset-driven; the prediction is only used as a seed rung and prior slope.
- The varblock reconstruction path (`jpxl-plan-render/src/lib.rs:673-1103`)
  builds per-block owned `CoeffMatrix`/`SampleBlock` objects, a
  `sigma_writes: Vec`, a `RenderedVarblock`, and chunk staging buffers for
  every perceptual probe.
- Current wall attribution (AKR, 12 MP quality path ≈ 2.4–2.5 s): metric blur
  ~24 %, EPF ~10 %, executor ~7 %, XYB ~7 %, plan-side quantization ~6 %;
  entropy pricing and scheduler bookkeeping are small. Three canonical
  render+score probes dominate the wall; the search algorithm itself is
  effectively free.

## The governing principle

**Surrogates may propose or reject candidates; only canonical evaluation may
accept one.**

Every externally observable result must continue to satisfy the exact
contract: full-resolution reconstruction, canonical SSIMULACRA2, exact
entropy pricing, and the hard quality floor (`Error::TargetNotMet`
semantics). Inside that barrier the scheduler is free to use learned
prediction, reduced-resolution scoring, approximate pricing, and cached
state. Uncertainty increases effort; it never weakens the guarantee.

Corollary for the refactor: **unify where to probe; do not unify what a
probe means or what constitutes a valid final answer.** The shared engine is
pure ladder-navigation policy. Byte-rate specifics (never-over ceiling,
non-monotone byte pockets, LF fill, entropy-tier refinement, maximize-bytes
ranking) stay in `rate.rs`; perceptual specifics (canonical render+metric,
score guard, price-after-feasibility, policy bank, terminal reducer) stay in
`quality.rs`.

## Why refactor before optimizing

Reduced-resolution navigation, adaptive probe budgets, learned rate priors,
and approximate entropy navigation would otherwise each be implemented twice
— once per controller. With one navigator they become properties of a single
engine while the two exact contracts remain isolated. The unification is
therefore sequenced first, in behavior-preserving steps that are cheap to
verify (byte-identical outputs), and the speed work lands on top of it.

Explicitly rejected shapes, so nobody re-attempts them:

- Do **not** make `rate::Search<F>` the shared abstraction. It is generic
  only in how bytes are obtained; its internals assume `u64` byte ceilings,
  rate tolerance, largest-feasible tracking, pocket/fill phases.
- Do **not** make `quality::Navigator` the shared abstraction. It owns
  `CandidateSearchContext`, the evaluator, structural anchors, entropy tiers
  — too much execution policy.
- Do **not** build `LadderNavigator<Evaluator, Objective, …>` generics that
  own the probe work. The navigator asks "give me the value at rung X"; the
  adapter decides what obtaining it means.

## Phase N0 — extract the quantizer ladder

New module `jpxl-encode-policy/src/quantizer_ladder.rs` (name may be
shortened to `ladder.rs`): move `Rung`, `QuantizerChoice`, `HF_MUL_RUNGS`,
`LADDER_LEN`, rung field helpers, `effective_scale`,
`rung_for_effective_scale`, `coupled_quant_lf` out of `rate.rs`. `rate.rs`
and `quality.rs` both import from it; `rate.rs` re-exports for downstream
compatibility (`lib.rs:139` currently re-exports from `rate`).

Purely mechanical. **Gate:** byte-identical Fast/Balanced fingerprints on the
standard corpus; all tests pass; no wall claim.

## Phase N1 — one crossing mathematics

New module `jpxl-encode-policy/src/navigation.rs` with pure types/functions:

```rust
// x = ln(effective_scale(rung))
struct LocalModel { slope: f64, anchor_x: f64, anchor_y: f64 }
impl LocalModel { fn crossing(&self, target_y: f64) -> f64 } // x*
enum Constraint { Ceiling, Floor }   // bytes <= target  vs  score >= target
fn fit_slope(observations: &[(f64, f64)]) -> Option<f64>;
fn geometric_step(x: f64, direction: Direction, ratio: f64) -> f64;
fn bounded_crossing(...) -> f64;     // clamp + minimum-progress rules
```

Coordinate transforms live in the adapters and give both curves the same
orientation (rising in x):

- rate: `y = ln(bytes)`, `Constraint::Ceiling`;
- quality: `y = -ln(max(100 − score, LOSS_EPSILON))`, `Constraint::Floor`.

Replace the duplicated implementations in both controllers with calls into
`navigation.rs`, preserving current rounding/clamping semantics exactly
(including `rate.rs`'s force-one-rung-progress and `quality.rs`'s
`MIN_AIM_MARGIN` behavior — these become explicit parameters, not silently
harmonized). The quality `prior_beta` and the rate two-anchor slope both
become a `LocalModel` slope.

**Gate:** identical probe sequences (trace comparison) and byte-identical
outputs across the corpus for both `--target-bytes` and quality paths. This
is the milestone that proves the math is truly shared.

## Phase N2 — the stateful `LadderNavigator`

Move observation tracking, no-repeat caching, bracket discovery, predictor
fallback ordering (observed slope → prior slope → geometric step), crossing
tightening, adjacency detection, and work-cap termination into:

```rust
struct NavigationObservation { rung: Rung, value: f64, authority: Authority }
enum Authority { Surrogate, Canonical }
struct NavigationPrior {
    predicted_rung: Rung,
    slope: Option<f64>,
    interval: Option<(Rung, Rung)>,
    confidence: f32,
}
struct NavigationBudget { probes: u32, rescue_probes: u32 }
enum NavigationAction { Probe(Rung), Done(NavigationResult) }
```

The navigator knows nothing about plans, pixels, SSIMULACRA2, or entropy.
The caller loop is `while let Probe(rung) = nav.next_action() {
nav.observe(rung, adapter_value(rung)?) }`. `Authority` is designed in now
(Phase S1 consumes it): the navigator uses any observation as evidence, but
`NavigationResult` records whether the resolved candidate is backed by a
canonical observation, and the quality adapter refuses to finalize a
candidate that is not.

Convert `quality::Navigator` into a perceptual adapter around
`LadderNavigator` first; convert the rate controller's navigation phase
second (its pocket/fill/LF/entropy-tier phases remain untouched around it).
Two exact rate anchors produce a `NavigationPrior` exactly as `predict_v2`
does, so the navigator does not distinguish a learned model from an
empirical two-point fit.

Budgets stay split: `NavigationBudget` is all the navigator sees;
`RateSearchBudget` and `QualityBudget` (pixel probes, exact prices,
structural builds, policy trials, reducer limits) stay controller-owned.

**Gate:** behavior parity — identical probe sequences and outputs on the
corpus; wall neutral (±noise). Where exact parity would require a
pathological parameter, the deviation is enumerated, justified, and
A/B-swept before merge.

## Phase N3 — confidence-adaptive probe budgets

Turn `predict_v2`'s interval into the work scheduler. Routing table
(hard preset caps of 3/5 canonical probes remain the emergency ceiling):

| Predictor state | Surrogate probes (once S1 lands) | Canonical budget |
|---|---|---|
| High confidence (tight interval, low saturation, no OOD) | 0–1 | normally 1, max 2 |
| Normal in-distribution | 1–2 | normally 1–2 |
| High uncertainty (wide interval) | optional | current 3/5 logic |
| OOD (`ood_flags` nonempty / `fallback_reason`) | none | existing conservative path |
| Near saturation (`saturation_risk` high) | conservative | existing rescue semantics |

First canonical probe on a confident image: a deliberately conservative
point inside the interval (loss-space reserve side), not a bracket endpoint.
After that one exact score the navigator has both the prior slope and this
image's residual, so the second candidate (if needed) is aimed, not generic.
Thresholds for "tight"/"wide" are chosen from the existing shadow-trace
corpus, not invented.

**Gate:** A/B corpus sweep (the 441-cell harness): 0 floor violations,
bytes ratio within the established parity band (≥ ~0.997 of current),
mean canonical probe count strictly down, wall down on in-distribution
images, no regression on the OOD/saturated tail.

## Phase R1 — `RenderScratch` and `*_into()` reconstruction (exact; independent)

In `jpxl-plan-render`: per-worker

```rust
struct RenderScratch {
    coeff: [AlignedBuffer<f32>; 3],
    samples: [AlignedBuffer<f32>; 3],
    // transform-specific temporaries
}
```

with `dequantize_into` / `idct_into` / `render_varblock_into` APIs replacing
the per-varblock owned `CoeffMatrix`/`SampleBlock`/`RenderedVarblock`
production, the `sigma_writes` `Vec`, and as much chunk staging/scatter as
ownership allows (workers writing non-overlapping frame regions directly).
Distinct from the Phase-25 coefficient-workspace work: that reused arenas
between quantizer evaluations; this removes allocation and staging inside
the reconstruction performed for every perceptual probe.

No numerical decision changes. Can proceed in parallel with N0–N3 (different
crate, different owner if multi-agent).

**Gate:** byte-identical fingerprints; wall and RSS measured at 4.3 MP and
12 MP under the quiet-host `wall_current.py` recipe.

## Phase S1 — surrogate navigation evaluator

The largest remaining scheduler win. Add a `NavigationEvaluator` producing
`Authority::Surrogate` observations: half-linear-resolution reconstruction
scored with the SSIMULACRA2 scales that survive meaningfully at that
resolution (start conservative at 1/2; quarter-resolution only if evidence
later supports it). Pipeline per image:

```text
predict_v2 → (0–2 surrogate probes) → canonical render+score at the
proposed rung → accept, or one canonical correction → exact entropy price
```

Hard rule, enforced in the adapter: a surrogate observation can never
satisfy the quality contract; `QualityOutcome` is only ever backed by a
canonical score. On a canonical miss after surrogate proposal, one canonical
correction; on a second miss, fall back to the current full navigator.
Routing follows the Phase N3 table. Probe traces record authority so sweeps
can attribute cost.

This is the "Contract-B-screened reduced-resolution navigation" the active
cost record (`@jpegxl-rs.work.pqc-usable-efforts-cost`) already names as the
remaining credible large wall reduction.

**Gate:** floor violations remain 0 by construction (verified over the
corpus anyway); bytes parity band held; 12 MP quality wall materially down
(target: canonical probes/image → ~1.2–1.5 mean on in-distribution images);
surrogate mispropose rate (canonical correction needed) reported per sweep.

## Phase S2 — banded renderer→metric pipeline

First build the **exact** streaming scorer: renderer band → EPF band with
halo → linear/XYB conversion → horizontal blur (row-wise) → vertical
recursive-blur per-column state → pooling accumulation, without
materializing full-frame intermediates. Constraint carried over from
`pool.rs`: the f64 pooling sums are accumulated in original pixel order —
the banded design must preserve that order (band-sequential accumulation
does) or the metric version must be renegotiated; this plan preserves it.

Once exactness is proven (bit-identical scores across the corpus), derive
the reduced-resolution surrogate variant from the same pipeline by
decimating as pixels leave the renderer — replacing S1's interim
shrink-after-reconstruct implementation if S1 landed first.

**Gate (exact variant):** bit-identical metric scores; RSS reduction
measured at 12 MP and 50 MP; wall neutral-or-better. **Gate (surrogate
variant):** same as S1's sweep gates.

## Later / conditional phases

- **B1 — paired-candidate evaluation.** Only if profiling after S1/S2 shows
  two nearby probes per image remain common: make candidate count a
  SIMD/data-layout lane inside the already-parallel traversal (shared
  structure, matrices, CfL, reference metric data). Explicitly not
  "run two probes under Rayon simultaneously" (nested parallelism, cache
  and RSS damage).
- **E1 — exact metric micro-work.** Worker-owned scratch in `blur.rs`
  (padding, strip state), inspect the f64 vertical-recurrence codegen,
  4-vs-8 row lanes by width; EPF persistent ping-pong storage and
  band-with-halo execution; XYB explicit SIMD if codegen is poor. All
  byte-identical. `pool.rs` reduction SIMD stays closed (exactness wall).
- **RT1 — rate-path fused multi-rung quantization.** Walk source
  coefficients once, evaluate 2–3 quantizer scales in candidate lanes;
  establishes how much of the remaining `--target-bytes` cost is removable
  before attempting the more ambitious quantizer-event (chunk-census
  invalidation) scheduler. Final candidate always goes through the ordinary
  exact Store path.
- **RT2 — learned rate prior.** A source+transform-feature predictor of the
  rate crossing emitting a `NavigationPrior` (the old fixed-slope experiment
  failed for lack of content features; the qpv2 feature set changes that).
  High confidence can skip the second anchor; the bounded controller remains
  the fallback.

## Closed directions (do not revisit)

Global fixed-DCT8 anchor tricks; universal fixed rate slope; broad
fresh-finalist reconstruction; selective cover refresh; Fast/Balanced
policy-bank searches; additional entropy alternatives; scheduler
data-structure micro-optimization (3–5 element vectors are free);
`pool.rs` accumulation-order changes under the current metric version.

## Sequencing and ownership

```text
N0 → N1 → N2 → N3 → S1 → S2 (surrogate variant)
R1 (independent, any time)     S2 exact variant may start after N2
E1 (independent, any time)     B1, RT1, RT2 conditional, after S1 evidence
```

Each phase is a separate commit series with its gate evidence recorded in
AKR before the next begins. Behavior-preserving phases (N0–N2, R1, S2-exact,
E1) gate on byte-identical fingerprints; policy phases (N3, S1,
S2-surrogate) gate on the A/B sweep harness with the hard floor at zero
violations. Wall numbers only from the quiet-host interleaved recipe;
day-to-day testing on debug-release, shipped numbers on release-final.

## Risks

- **Parity drift in N1/N2.** The two controllers' rounding and
  minimum-progress rules differ subtly; harmonizing them silently would
  change outputs. Mitigation: parameterize, trace-diff every corpus image.
- **Surrogate systematically biased on some content class.** Mitigation:
  canonical acceptance makes this a cost problem, not a quality problem;
  the mispropose-rate metric in every sweep catches it, and OOD routing
  bypasses surrogates entirely.
- **Banded blur state complexity (S2).** The recursive vertical filter's
  per-column state and delayed output are the hard part; building the exact
  version first isolates the numerics from the decimation.
- **Refactor opportunity cost.** N0–N2 buy no wall time by themselves.
  Accepted deliberately: they make every subsequent speed feature
  single-implementation, and their gates are cheap (byte-identity).
