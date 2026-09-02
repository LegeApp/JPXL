# Case predictor v2: probe-objective neighbour geometry, structure features and a paintings corpus (2026-09-02)

## Question

The 2026-09-02 case-table predictor (`2026-09-02-case-predictor-first-guess.md`)
cut the first-guess error to a third of the knot model's, but three things
bounded it: the neighbour geometry was plain Euclidean distance over 19
standardized features chosen for a linear model, the objective it was tuned
against was ln error rather than what the controller pays for (probes to a
stop inside the accept band), and natural content had only 21 labelled
families. The operator added 258 paintings and pointed at the bpg-rs
`still265` variance preanalysis as the missing block-versus-solid signal, and
outside advice (`sources/external/advisor-probe-campaign-2026-09-01-a53fa9eb`)
proposed, in order: optimise for the first probe landing in the band, learn a
codec-specific feature distance, make k adaptive, transfer whole curves,
add structure features, and grow the corpus where feature space is empty.

Does a case predictor searched against the simulated probe count, over the
19 features plus a frame-level structure vector, on a corpus with 258 more
natural images, remove probes on never-seen paintings — and does it still
hold the floor on the synthetic and graphic holdout?

## Data

* **Paintings corpus.** `test-set/new-test-set-paintings` (258 JPEG/PNG
  reproductions of paintings, 4.4 GB) → `test-set/paintings/*.ppm` +
  `test-set/paintings-manifest.json` via `tools/quality_corpus_extend.py`
  (one family per source file, splits by a deterministic family hash:
  calibration 120, development 86, ext-holdout 52). 90 sources over the 13 MP
  cap (median 19 MP, max 149 MP) are Lanczos-resampled to fit it by the new
  `--downscale-large` option and carry the factor in their provenance; the
  tool's PPM writes are now idempotent so a rerun that only adds images
  never rewrites a file a sweep is reading. Oracle sweeps
  (`tools/quality_oracle_labels.py sweep`, Balanced, seven knots, binary at
  1556bfb) ran in three pinned lanes; labels in
  `.agent/scratch/paintings-labels-20260902/labels.jsonl`.
* **Existing corpus.** The 2026-09-02 all-knot labels (91 images, 89
  families; 623 in-domain rows).
* **Structure features.** `crates/jpxl-encode-policy/src/preanalysis.rs`
  ports the still265 32×32-cell pass (the same Sobel thresholds and seven
  cell classes the text/UI classifier already ports) and reduces it to a
  frame vector: seven class shares, log-variance mean/std/q10/q90, edge
  mean/q90, flat share, noise mean/q90, chroma-activity log mean/q90,
  orientation entropy, direction dominance, axis alignment, a heterogeneity
  share, plus three derived logarithms (25 values, `qpv2-stp/1` = the 19
  `qpv2-st/1` features + these). It is computed from the 8-bit samples at
  frame preparation (`PreparedFrame::with_preanalysis`) on the quality path
  and the calibration tools, emitted in `source_features.preanalysis`, and
  read only by the case predictor; a frame without it keeps the legacy start.
  Cost: one integer pass over the frame (about 0.1 s at 12 MP, measured
  unpinned on a busy host, not claimed).
* **Features for training.** `jpxl features --json --transform-summary`
  with the preanalysis-carrying binary over every corpus and paintings image
  (`.agent/scratch/firstguess2-20260902/features-rust.jsonl`); the labels'
  own `source_features` predate the field, so `train-cases` merges the
  per-image map (`--preanalysis-features`).

## Method

`.agent/scratch/firstguess2-20260902/study.py` and `search.py`.

* **Predictor.** Nearest cases in *weighted* standardized feature space
  (`distance² = Σ wⱼ² Δzⱼ²`), at most k, and with an optional adaptive cut
  (a further neighbour only while within `ratio ×` the nearest distance);
  each neighbour's whole seven-knot curve is interpolated in ln loss at the
  target (median) and at the effort's aim score (candidate); the neighbours'
  spread at the target is the interval; a spread above a threshold routes to
  the legacy start; an optional offset shifts the first guess by a multiple
  of the spread.
* **Objective.** Not ln error: the corrected-stop navigator replayed on the
  image's measured oracle curve — Balanced (reserve 0.03, reserve-coupled
  band, five probes, a bracket tail charged 4.6, a legacy-start fallback
  charged 4.0) plus Fast (reserve 0.06, three probes, an unmet target charged
  6). The search minimises the sum of the two means.
* **Search.** Coordinate descent over per-feature weights in {0, ½, 1, 2, 4}
  and over {k, ratio, spread threshold, offsets, distance eps}, accepting a
  change only when it gains at least 0.02 probes, alternated twice. Every
  objective evaluation is leave-one-family-out over the **calibration +
  development families only**. The holdout families (the old corpus
  `holdout` split and the paintings `ext-holdout`) are never in a fold or a
  table during the search; they are scored once, blind, with the searched
  configuration against a table of the search families.

### Preregistered runtime trial (written before the paintings labels were analysed)

Reference arm R: the working tree's default build (promoted controller,
case predictor off). Candidate C2: the same tree with `case-predictor`, the
`qpv2-stp/1` table trained on the old corpus (all splits) plus the paintings
calibration + development images, with the searched weights and
configuration — the paintings ext-holdout is not in the table. Comparison
arm C1 (Balanced only): the landed v1 case table at 2b1a055 (19 features,
all old-corpus splits), so the paintings gain can be split between the new
data and the new geometry. Harness `tools/one_shot_promotion_ab.py`, four
threads, P-cores 0,2,4,6.

1. Paintings ext-holdout, seven targets, Balanced (364 cells) — the gate.
2. Paintings ext-holdout, targets 30/50/70/85, Fast (208 cells).
3. Old-corpus holdout, seven targets, Balanced (140 cells) — the regression
   check on synthetic and graphic content; that split has been looked at
   twice before and is reported, not gated.

* **Pass**: zero floor violations on C2 in runs 1–3; run 1 byte geomean
  ≤ 1.01 and mean pixel probes down ≥ 0.4 per encode against R; no run-1
  cell with byte ratio > 1.15; run 2 zero floor violations, zero encode
  errors, geomean ≤ 1.015; run 3 zero floor violations and geomean ≤ 1.01;
  thread-count identity (1 vs 4) on spot cells.
* **Partial**: floor held but a probe or byte criterion misses — report,
  keep off by default.
* **Fail**: any floor violation or encode error, or run-1 geomean > 1.01.

Wall is reported, not claimed (the host is at full load with the sweeps).

## Results

### Offline study

Paintings labels: 258 images × 7 knots = 1806 rows, none censored; median
crossing scale 4.5k / 7.1k / 13.8k / 29k / 128k at 30/50/70/85/95, a tight
band (q10–q90 within ±30% at every knot). In the weighted feature space of
the old-corpus search a painting's nearest existing-corpus image sits at median distance
5.6 (p90 7.5) while its nearest other painting is at 1.6 (p90 2.5): the
paintings fill a region the old corpus did not cover, and cover it densely.

Simulated probes to stop (Balanced / Fast) under the corrected-stop
navigator on the oracle curves; "hit" is the first probe landing in the
accept band; "fail" the Fast budget ending under target.

| predictor | rows | Balanced probes | Fast probes | Fast fail | hit | fallback | ln err median / p90 | natural | painting | graphic | synthetic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deployed knot model, all rows | 2429 | 2.33 | 2.05 | 2.7% | 0.19 | 0 | 0.164 / 0.531 | 2.07 | 2.08 | 2.38 | 3.71 |
| landed v1 case table (19 f, k=2), LOFO all | 2429 | 2.14 | 1.70 | 0.9% | 0.28 | 11% | 0.083 / 0.250 | 2.07 | 1.97 | 2.60 | 2.88 |
| searched v2 (44 f, weights, k=2 ratio 3, offset ¼ spread, spread 0.6), LOFO all | 2429 | 1.88 | 1.57 | 1.9% | 0.37 | 2% | 0.071 / 0.228 | 2.00 | 1.70 | 2.01 | 2.70 |
| landed v1, blind (old holdout + paintings ext-holdout) | 497 | 2.09 | 1.67 | 0.6% | 0.30 | 11% | 0.075 / 0.254 | 2.68 | 1.93 | 2.19 | 2.46 |
| searched v2, blind | 497 | 1.97 | 1.73 | 3.4% | 0.34 | 1% | 0.075 / 0.308 | 2.82 | 1.77 | 1.43 | 2.52 |

Blind per target, landed → searched (Balanced probes / hit): 30: 1.92/0.32 →
1.95/0.35; 50: 1.97/0.28 → 1.88/0.32; 70: 1.98/0.30 → 1.94/0.32; 80: 2.21/0.30
→ 2.16/0.25; 85: 2.42/0.20 → 2.10/0.25; 90: 2.14/0.37 → 1.86/0.45; 95:
1.99/0.37 → 1.92/0.44.

The search kept 17 of 44 features at non-unit weight or dropped them:
×4 on `ln_chroma_q50`, `log2_pixels`, `high_low_ratio`,
`near_zero_frac_1e3` and `pa_noise_q90`; ×½ on `directional_asymmetry`;
`ln_aspect` and `chroma_ac_ratio` dropped; `pa_share_chroma_critical` on
at 1; every other preanalysis feature stayed at 0. It chose k = 2 with an
adaptive cut at 3× the nearest distance, a first guess shifted finer by a
quarter of the neighbours' spread, and a wider spread threshold (0.6).

The wider threshold is where the searched configuration's simulated Fast
failures come from (17 of 497 blind rows, at 90/95 on natural and synthetic
frames, against 3 for the landed table). On the **search set alone**
(leave-one-family-out, no holdout row consulted), restoring the existing
0.405 threshold costs 0.05 Balanced probes (1.885 → 1.934) and returns the
failure rate to the landed level (1.3% → 0.9%); the trial table therefore
uses the searched weights, k, ratio and offset with the existing spread
threshold. That choice was made before any runtime cell was encoded.

Three earlier searches on the old corpus alone (before the paintings
labels existed, `search-base*.out`) are recorded for honesty: with the
Python prototype features, blind 2.22 → 2.06; with the Rust features and
the ln-error-free Balanced objective, 2.22 → 2.19 while dropping every
fallback; with the Balanced + Fast objective, 2.22 → 2.21. The 17-family
old holdout (133 rows) cannot resolve differences of that size; the
paintings ext-holdout (364 rows) can, and the base-only searches are not
used further.

The trial table (`train-cases --blind-split ext-holdout`): 295 cases
(old corpus all splits + paintings calibration/development), LOFO median
ln error 0.071 (p90 0.250, 62% within 0.10), paintings ext-holdout blind
0.067 (p90 0.183, 67% within 0.10, bias −0.004).

### Runtime trial, round 1 (C2: searched geometry, spread threshold 0.405, legacy start on fallback)

Binaries: reference `jpxl-ref2` (49cb230a…, the working tree, default
features), C2 `jpxl-cases-v2-blind` (1b1f1435…, `case-predictor`, 295-case
table without the paintings ext-holdout), C1 `jpxl-cases-v1` (4c49b228…,
HEAD 1556bfb with the landed 19-feature table, built in a worktree). Four
threads, P-cores 0,2,4,6; `analyze.py` in the scratch directory renders the
full per-target tables.

| run | cells | floor | errors | byte GM | worst | probes ref → C2 | ≤ 2 probes | wall GM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1. paintings ext-holdout, Balanced, 7 targets (blind) | 364 | 0 | 0 | 0.990 | 1.414 | 2.21 → 1.80 | 242 → 317 | 0.89 |
| 2. paintings ext-holdout, Fast, 30/50/70/85 (blind) | 208 | 0 | 0 | 0.977 | 1.075 | 1.84 → 1.74 | 185 → 196 | 0.96 |
| 3. old-corpus holdout, Balanced, 7 targets (**in the table**) | 140 | 0 | 0 | 0.981 | 2.582 | 3.13 → 1.64 | 47 → 119 | 0.73 |

Run 1 per target (probes ref → C2 / byte GM): 30: 2.35 → 1.73 / 0.995;
50: 2.21 → 1.81 / 0.997; 70: 2.27 → 1.83 / 0.997; 80: 2.42 → 1.92 / 0.995;
85: 2.44 → 1.81 / 0.996; 90: 2.27 → 1.67 / 0.991; 95: 1.50 → 1.87 / 0.962.
Every cell changed bytes (no identical cells: the first plan differs on
every painting). Thread-count identity held on 9/9 spot cells (two
paintings and a screenshot at 50/85/95, one vs four threads).

The 357 run-1 cells the table was confident on: byte geomean 0.9895,
worst 1.092, probes 2.21 → 1.77. The seven `wide_interval` cells took the
*legacy* start (the pre-knot-model table) and cost 2.00 → 3.71 probes with
byte geomean 1.040, including the run's only cell over the 1.15 tail:
`paintings-0199` at target 95 (1.414×), where the legacy start at rung
73727 scored 92.1, the corrections overshot to 148802 (95.96) and the
rescue kept it, while the reference's knot-model start (117127, 94.86)
finished one rung later at 122260. The reference controller on such cells
uses the knot model; the case predictor's fallback used the older table —
a worse fallback than the model it replaced, not a property of the table.

Run 3 is no longer blind (the old holdout is in the 295-case table) and
is reported for the floor and the fallback behaviour: the seven
`ood_feature` cells (tiny and 256-pixel frames) are byte-identical to the
reference, which takes the legacy start there too; the 2.58× cell is
`gradient-chroma-shallow-green-holdout-640x640` at 70, a 416-byte encode
where a *confident* case prediction (rung 67, nearest neighbours are the
corpus's other shallow chroma gradients) met a non-monotone curve
(scores 51.9 → 44.0 → 69.3 → −7.2 → 17.5 → 94.0) and ended at 1074 bytes;
the reference's knot model flagged the frame out of range and its legacy
start happened to land well. Sub-kilobyte synthetic gradients remain the
case table's weak spot, as in the first trial.

**Verdict, round 1 (preregistered): Partial.** Floor held on all 712
cells, no encode errors, run-1 byte geomean 0.990 ≤ 1.01, probes down
0.41 ≥ 0.4 per encode, Fast 0.977 ≤ 1.015, run 3 0.981 ≤ 1.01; the one
run-1 cell over the 1.15 tail (a fallback cell) misses the tail criterion.

**C1, the landed v1 table, on the same 364 blind paintings cells:** floor
held, byte geomean 0.999, worst 1.164, but probes 2.21 → **2.49** and ≤ 2-probe
cells 242 → 195 — a table with no painting in it is worse than the knot model
on paintings (its nearest cases are the corpus's few photographs, at median
distance 5.6). The paintings gain of C2 is coverage and geometry together,
not geometry alone; and it is the direct measurement of what promoting v1
would have done to this content class.

### Round 2 (C3: fallback to the knot model instead of the legacy start) — disclosed post-hoc change

Round 1 was looked at before this change: on any case-table fallback
(`tiny_frame`, an out-of-range weighted feature, `case_spread`, or a frame
with no structure vector) `predict` now returns the knot model's own
prediction, so a fallback cell behaves exactly as the reference controller
does — a large-frame-only range flag keeps the case median as before. The
rule is a strict improvement in construction (the fallback becomes the
current default, not the oldest predictor) and touches only cells the
table had already declared uncertain; it is not a fit to the holdout.

Binary C3 `jpxl-cases-v2b` (7d7ef9d3…), same table as C2.

| run | cells | floor | errors | byte GM | worst | identical | probes ref → C3 | ≤ 2 probes | wall GM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1. paintings ext-holdout, Balanced, 7 targets (blind) | 364 | 0 | 0 | 0.990 | 1.092 | 7 | 2.21 → 1.77 | 242 → 322 | 0.89 |
| 2. paintings ext-holdout, Fast, 30/50/70/85 (blind) | 208 | 0 | 0 | 0.977 | 1.075 | 1 | 1.84 → 1.73 | 185 → 197 | 0.96 |
| 3. old-corpus holdout, Balanced, 7 targets (in the table) | 140 | 0 | 0 | 0.981 | 2.582 | 7 | 3.13 → 1.64 | 47 → 119 | 0.73 |

Run 1 per target (probes ref → C3 / byte GM): 30: 2.35 → 1.73 / 0.995;
50: 2.21 → 1.75 / 0.998; 70: 2.27 → 1.83 / 0.997; 80: 2.42 → 1.92 / 0.995;
85: 2.44 → 1.81 / 0.996; 90: 2.27 → 1.69 / 0.990; 95: 1.50 → 1.67 / 0.957.
The seven former fallback cells are byte-identical to the reference; the
357 confident cells are unchanged from round 1. Run 2 Fast per target:
30: 1.98 → 1.94 / 0.956; 50: 1.77 → 1.81 / 0.980; 70: 1.83 → 1.65 / 0.987;
85: 1.79 → 1.50 / 0.986, overshoot +2.56 → +1.73. Run 3 is identical to
round 1 except the wall column (its fallback cells were already the
reference's).

**Verdict, round 2 (preregistered criteria): Pass.** Zero floor
violations on 712 cells and zero encode errors; run 1 byte geomean 0.990 ≤
1.01, probes −0.44 ≥ 0.4 per encode, no cell over 1.15 (worst 1.092); run 2
geomean 0.977 ≤ 1.015; run 3 geomean 0.981 ≤ 1.01; thread identity 9/9.
The verdict carries the disclosure that the fallback rule was changed
after round 1 was seen, and that run 3 is in-table.

## Conclusion

* **What was landed.** `preanalysis.rs` (the still265 cell pass as a
  25-value frame vector, attached at frame preparation on the quality path
  and the calibration tools, emitted in `source_features.preanalysis`);
  the case predictor reads the 44-feature `qpv2-stp/1` vector with
  per-feature distance weights, k = 2 with a 3× adaptive cut, a
  quarter-spread finer offset and the 0.405 spread threshold, all as table
  constants; any fallback hands the frame to the knot model; `train-cases`
  takes several label files, per-image transform and preanalysis maps,
  weights, ratio and threshold; `quality_corpus_extend.py` downscales
  oversized sources and writes idempotently. The production table
  (`quality_predictor_cases.rs`, `qpv2-cases-2`) has 347 cases over every
  split of both corpora; LOFO median ln error 0.073 (paintings 0.066,
  photo 0.19, saturated 0.24). The feature stays **off by default**: it is
  a Contract B change and the promotion decision is the operator's
  (`@jpegxl-rs.question.promote-case-predictor-2026-09-02`, revised with
  this evidence).
* **What the trial says.** On never-seen paintings the v2 predictor holds
  the floor, saves 0.44 probes per Balanced encode (a fifth), lifts the
  share of encodes done in two probes from 66% to 88%, and trims bytes 1%;
  Fast saves 0.11 probes for 2.3% fewer bytes. The v1 table, with no
  painting in it, *costs* 0.28 probes on the same cells: a case table is
  exactly as good as its coverage of the content class, which is the
  strongest argument both for the paintings and against promoting a table
  that has never seen the user's content.
* **What did the work.** In the offline study the gain came, in order,
  from the paintings themselves (coverage), from the probe-count objective
  choosing the weights and the adaptive cut, and from four structure
  features (`pa_noise_q90` at ×4, `pa_share_chroma_critical`; the class
  shares and heterogeneity measure were not selected once the paintings
  were in). The block-versus-solid failure of the first trial no longer
  reproduces because the 256-pixel frames are out of the weighted range
  and take the knot model's path, identical to the reference.
* **What remains.** Camera photographs at 12–50 MP have no close case
  (photo LOFO error 0.19) and are the next coverage lever; screenshots and
  line art next. Sub-kilobyte synthetic gradients with non-monotone curves
  are a table weakness the navigator, not the predictor, would have to
  absorb. The simulated Fast refusal rate is a useful search penalty but
  is not calibrated against the real rescue probe; the real Fast run had
  no refusal on 208 cells.
