# First-guess accuracy across the quality range: from the linear knot model to a case table (2026-09-02)

## Question

The operator asked for better first-guess (predicted rung) accuracy throughout
the quality range. The deployed `qpv2-st-1` crossing model is a pooled linear
fit — one affine model per output over 19 standardized source and transform
features, with a global slope in ln loss(target) and three feature
interactions — folded into seven per-target knots. The 2026-09-02 retrain of
the same structure on the current corpus failed its gates
(`2026-09-02-qpv2-current-corpus-retrain.md`), and the promoted controller's
low-target traces showed a consistent +0.36 ln bias on photos
(`@jpegxl-rs.observation.qpv2-photo-low-target-bias-2026-09-02`). What model
structure, on the labels we have, predicts the crossing best — and does it
remove probes when encoded for real on never-seen images?

## Data and scoring

The 2026-09-02 all-knot oracle labels: 91 corpus images × 7 targets
(30/50/70/80/85/90/95), each row an exact crossing scale on the image's
measured ladder curve and the local loss exponent β there. Two `tiny` images
sit below the 128-pixel model domain and route to the exact controller;
the study uses the remaining 623 rows (89 images, 89 families after the
photo variants are grouped). Error is `ln(predicted scale / oracle crossing
scale)`. Cross-validation is leave-one-family-out (LOFO): every prediction
is made with the image's whole family absent from the fit. Classes are
grouped as natural (photo, photo-scene, grayscale, noise-lowlight: 21
families), graphic (text-screenshot, line-art: 13) and synthetic (gradient,
saturated: 51).

A Python port of the deployed model reproduces the runtime's median rung on
393 of 393 comparable trace cells (the 90/95 cells differ only by rung-index
quantisation), so the deployed row below is the shipped model's own error.

## Offline study

| model (LOFO unless noted) | median | p90 | ≤ 0.05 | ≤ 0.10 | bias | natural median | graphic | synthetic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deployed qpv2-st-1 (no CV; trained 08-24 on older labels) | 0.320 | 1.454 | 0.13 | 0.22 | +0.293 | 0.141 | 0.147 | 0.617 |
| pooled linear, current structure, ridge 1 | 0.460 | 1.172 | 0.05 | 0.11 | −0.021 | 0.259 | 0.391 | 0.564 |
| pooled linear, slope from every feature, ridge 1 | 0.443 | 1.150 | 0.07 | 0.12 | −0.021 | 0.292 | 0.427 | 0.488 |
| pooled quadratic in ln loss, ridge 3 | 0.459 | 1.226 | 0.04 | 0.11 | −0.015 | 0.305 | 0.506 | 0.506 |
| class-mean curve (oracle class, upper bound on routing) | 0.386 | 1.913 | 0.09 | 0.17 | −0.002 | 0.136 | 0.303 | 1.016 |
| nearest case, k = 1 | 0.111 | 0.655 | 0.31 | 0.48 | −0.059 | 0.114 | 0.075 | 0.132 |
| **nearest cases, k = 2, weight 1/(d + 0.05)** | **0.104** | 0.681 | 0.33 | 0.49 | −0.040 | 0.110 | 0.069 | 0.113 |
| nearest cases, k = 3 | 0.125 | 0.766 | 0.29 | 0.43 | −0.027 | 0.112 | 0.100 | 0.152 |
| k = 2 on top of the linear fit's residuals | 0.180 | 0.937 | 0.14 | 0.30 | −0.024 | 0.230 | 0.118 | 0.192 |

The deployed model's bias is target-dependent: +0.37/+0.39/+0.43/+0.30 at
30/50/70/80 (natural +0.31 at 30), +0.21 at 85, +0.19 at 95. The case table
is within ±0.09 at every target. Its neighbour β is also the better slope
prior: median |ln(β̂/β)| 0.24 against 0.36 for the knot model and 0.45 for
the constant 0.9.

What did not help: per-feature weight search on the case distance (no
feature worth dropping, ±2× reweighting moves the median by 0.007);
routing by a nearest-neighbour class vote and then fitting natural-only
linear models (natural stays at 0.135–0.19); restricting the neighbours to
the natural class (0.141). Natural content sits at 0.11–0.14 under every
method: with 21 families it is data-limited, and more labelled photographs
are the lever there.

### Simulated probe sequences

Replaying the promoted controller's rules on the oracle curves — first probe,
prior-slope correction gated at 0.5 ln, measured-slope correction, else the
bracket search charged at its trial mean of 4.6 probes — with the
reserve-coupled band:

| first guess | mean probes | natural | graphic | synthetic | t30 | t50 | t85 | t95 | 1-probe stops / 623 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deployed candidate-or-median, prior β 0.9 | 3.01 | 2.18 | 2.28 | 3.60 | 3.61 | 3.27 | 2.84 | 2.68 | 84 |
| case table k = 2 at the aim score, neighbour β | 2.41 | 1.93 | 1.95 | 2.77 | 2.61 | 2.67 | 2.45 | 1.91 | 187 |
| same, far cells (d > 5 or spread > 0.6) routed to the exact controller | 2.58 | 2.18 | 2.38 | 2.83 | 2.65 | 2.76 | 2.60 | 2.28 | 160 |

On the holdout split alone (133 rows, table without that split): deployed
2.95 → case table 2.09 (natural 2.40 → 1.71, target 30 3.33 → 2.21). Routing
far cells away costs probes, so the case predictor has no distance fallback:
only a tiny frame is out of distribution.

## Candidate

Feature `case-predictor`. `tools/quality_predictor_v2.py train-cases` emits
`quality_predictor_cases.rs`: per labelled image its standardized feature
vector, ln crossing scale at the seven knots and ln β at the seven knots,
plus the standardizer (about 90 × 33 numbers). At runtime
`predict_cases(features, transform, target, reserve)` takes the two nearest
cases in standardized feature space, weights them 1/(d + 0.05), and returns:
median = the weighted crossing of `target`; candidate = the weighted crossing
of the effort's aim score `target + reserve × loss(target)` (the score the
navigator's own corrections aim at; never coarser than the median); interval
= the neighbours' spread at the target; β = the neighbours' weighted
geometric mean. The routing, corrected stop, band and everything downstream
are unchanged; `predict` dispatches to the case table under the feature and
to the knot model otherwise, and the trace records the active model version.

## Blind trial

Reference arm: the promoted default (band + corrected stop), `bin/jpxl-ref`
sha256 `269ae5b6…`. Candidate: the same tree with `--features case-predictor`
and the table trained on calibration+development only (70 cases,
`cases-caldev.rs`), so the holdout split's 19 in-domain families are never
in the table. Harness `one_shot_promotion_ab.py`, 4 threads, P-cores
0,2,4,6, both arms interleaved. Scratch `.agent/scratch/qpv2-firstguess-20260902/`
(kept). The gate was preregistered in its README; wall is reported, not
claimed.

### Three rounds, two disclosed changes

The trial took three rounds; both changes between rounds are disclosed
because the holdout was seen before they were made.

* **Round 1** (`round1/`, `jpxl-cases-blind` 10ae582f…, no range rule):
  Balanced holdout 0 floor violations, byte geomean 0.9925, probes
  3.13 → 2.59, cells at ≤ 2 probes 47 → 79. The Fast run then hit a failure
  the offline simulation could not see: on `saturated-block-holdout-256x256`
  at target 70 the table's first guess (rung 63, score −73, from two
  solid-colour neighbours whose curves cross at scale ~60) was so coarse that
  Fast's three probes plus rescue ended at 69.86 and the encoder refused to
  write the stream. The knot model avoids that image only because it is
  outside the older corpus's feature range and takes the legacy start.
* **Round 2** (`round2/`, 7fba6fe3…) added the knot model's feature-range
  rule (`OOD_RANGE_MARGIN` over the cases' standardized range → legacy start;
  only-larger frames keep the case median). Balanced holdout 0 floor,
  geomean 0.9932, probes 3.13 → 2.67, 21 cells identical. The 256×256 block is
  *inside* the current corpus's range (which has 256-pixel images), so Fast
  target 70 failed again. Its two neighbours' crossings disagree by 0.42 ln.
* **Round 3** (`jpxl-cases-blind3` and the results below) routes a
  prediction whose neighbours disagree by more than `FALLBACK_LOG_WIDTH`
  (0.405, the knot model's own width) to the legacy start (`case_spread`
  flag, `wide_interval` reason). The failing cell then matches the reference
  at every Fast target. The Balanced holdout was re-run once more with
  per-process harness temp files after a stale round-2 process had briefly
  overlapped it; the re-run reproduced the round-3 numbers exactly.

### Balanced, holdout, seven targets (140 cells)

| target | floor ref/cases | byte GM | worst | identical | probes ref → cases | ≤ 2 probes | overshoot ref → cases | wall GM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | 0/0 | 0.990 | 1.102 | 4 | 3.65 → 2.15 | 3 → 15 | +7.10 → +5.09 | 0.74 |
| 50 | 0/0 | 0.972 | 1.091 | 3 | 3.40 → 2.60 | 4 → 13 | +4.14 → +4.88 | 0.87 |
| 70 | 0/0 | 0.986 | 1.056 | 4 | 3.45 → 2.75 | 3 → 9 | +1.51 → +1.41 | 0.89 |
| 80 | 0/0 | 0.999 | 1.532 | 4 | 3.30 → 3.10 | 4 → 8 | +1.11 → +1.55 | 0.99 |
| 85 | 0/0 | 1.002 | 1.447 | 4 | 3.20 → 3.05 | 8 → 8 | +1.26 → +1.04 | 0.91 |
| 90 | 0/0 | 0.980 | 1.023 | 4 | 2.75 → 2.70 | 11 → 11 | +0.62 → +0.54 | 0.91 |
| 95 | 0/0 | 0.974 | 1.056 | 4 | 2.15 → 2.25 | 14 → 13 | +0.62 → +0.51 | 0.99 |
| all | 0/0 | 0.986 | 1.532 | 27 | 3.13 → 2.66 | 47 → 77 | +2.34 → +2.15 | 0.90 |

By class: gradient 3.64 → 2.46 probes (bytes 0.981), line-art 2.86 → 1.57
(0.989), text-screenshot 2.43 → 1.71 (0.997), saturated 3.69 → 3.37 (0.977),
photo 2.41 → 2.27 (0.991, wall 0.96), tiny unchanged (legacy route). The
eight holdout photos gain at targets 30–50 (3 → 1–2 probes) and lose at 95,
where the knot model's fine-biased candidate already landed feasible in one
probe and the case table's tighter landing costs a second probe on two
frames for 3.75% fewer bytes; the 12–50 MP camera photos have no close case
(nearest distance 4.6–5.5, error 0.3–0.7 ln) and gain little. Two synthetic
cells exceed the 1.15 byte tail, both `saturated-hue-wheel-holdout-320x320`
at 80 and 85 (1.53, 1.45): the new wide-spread fallback sends them to the
legacy start, which lands 20 points low and needs the whole budget, where
the knot model's median start had landed near the target.

### Fast, holdout, seven targets (140 cells)

0 floor violations, 0 errors; byte geomean 0.988 (30: 0.933, 50: 0.980,
70: 1.009, 80: 1.016, 85: 1.030, 90: 0.958, 95: 0.992); probes 2.16 → 2.11;
cells at ≤ 2 probes 84 → 87; wall geomean 0.97. Fast's reference already
ends in one or two probes on 84 of 140 cells after the band change, so the
first guess has less to remove there; eight synthetic cells sit above 1.10
(worst `gradient-bilinear-holdout-640x640` at 95, 1.907).

### Calibration+development at 30/50/85 (213 cells, not blind)

These images are in the table, so this is the in-table behaviour, not an
accuracy measurement: 0 floor violations, byte geomean 0.962, probes
3.43 → 2.12, cells at one probe 19 → 127 of 213, wall geomean 0.78.

### Verdict against the preregistered gate

| criterion | result |
| --- | --- |
| zero floor violations in every run | **met** (0 of 493 cells, 0 errors after round 3) |
| Balanced holdout byte geomean ≤ 1.01 | **met** (0.986) |
| Balanced holdout probes down ≥ 0.4 | **met** (−0.47) |
| no Balanced holdout cell > 1.15 | **missed** (2 of 140, both one synthetic hue wheel via the new fallback) |
| Fast: zero floor, geomean ≤ 1.015 | **met** (0, 0.988) |
| t1/t4 identity on spot cells | **met** (12/12, `identity-cases-blind3.txt`) |

Strictly **Partial** on the byte tail; and the holdout was seen twice before
the final rules, so the trial no longer counts as fully blind for the two
fallback rules (both are the knot model's existing criteria applied to the
case table, not fits to the holdout). The accuracy claims rest on the
leave-one-family-out study.

## Conclusion

On the labels we have, a case table is the better first guess by a wide
margin: a third of the knot model's error overall, no target-dependent bias,
a better slope prior, and on never-seen images half a probe fewer per encode
at Balanced (3.13 → 2.66, 1.4% fewer bytes, wall −10%), most of it at
targets 30–70 and on graphic content. Natural content is where the data
runs out: with 21 natural families in the corpus, no method — linear, class
routed or case based — gets photos below about 0.11 ln median, and the
large camera photos have no close case at all.

**Landed off by default.** `case-predictor` is in the tree with the
production table trained on every split (89 cases,
`quality_predictor_cases.rs`, `train-cases --blind-split none`); the blind
trial's 70-case table, binaries and traces are kept in scratch. Promotion is
a Contract B change on almost every stream (112 of 140 holdout cells) with a
Partial verdict on the byte tail, and is the operator's call
(`@jpegxl-rs.question.promote-case-predictor-2026-09-02`). If promoted, the
shipped model will not be the measured one: it also contains the holdout
images, exactly as any retrain ships more than it was validated on.

**Next lever, in order:** (1) more labelled natural images — every method
is data-limited there, and each new photo is one oracle sweep (about a
minute at 12 MP); (2) a feature that separates block-edged synthetic
content from solid colour, which is the one failure the trial found; (3) the
Fast preset's three-probe budget has no recovery from a catastrophic first
probe on a non-monotone curve — a navigator question, not a predictor one.
