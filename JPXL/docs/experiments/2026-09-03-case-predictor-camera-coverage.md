# Case predictor: camera-photo coverage (2026-09-03)

## Question

The v2 case predictor (`2026-09-02-case-predictor-v2-paintings.md`) removed
probes on never-seen paintings once paintings were in the table, and the
v1 table without paintings made paintings *worse* — the lever was
coverage, not geometry. Camera photographs were the class with the least
coverage left: 25 natural images in the table, photo LOFO first-probe hit
0.19, nearest-case distances far above the paintings' 1.6. The operator's
direction: do not tune for particular picture content; add content where
it helps.

Does adding one phone camera roll (220 labelled frames) to the table,
*without changing the searched geometry*, remove probes on never-seen
camera photographs, and does it leave paintings and the synthetic/graphic
holdout where they were? Secondary: does re-running the weight/config
search on the enlarged set add anything over coverage alone?

## Data

* **Camera corpus.** `../../DCIM/Pictures 2024/Summer 2024` (one Galaxy
  S24+ roll: 105 JPEG + 510 HEIC, 4000×3000 for the bulk, 8160×6120 for the
  50 MP mode, a few 3392×2544 and one panorama) →
  `test-set/camera/*.ppm` + `test-set/camera-manifest.json` via
  `tools/quality_corpus_extend.py 1.1.0`, which gained: HEIC/HEIF decode
  through ImageMagick (`magick <file> ppm:-`), camera-burst folding (files
  whose `YYYYMMDD_HHMMSS` stamps are within 120 s share one family, so
  near-identical frames never straddle splits and only one is picked),
  round-robin sampling over capture days so one day cannot dominate a
  single-directory roll, `--exclude-stems` for the seven files already in
  the old corpus as `scene-*`, and `--native-large 6` so six of the 23
  over-cap frames stay at native 50 MP (the rest are Lanczos-resampled to
  13 MP as before). 615 files → 437 burst families → 220 picked (189 HEIC,
  31 JPEG), splits by family hash: calibration 120, development 61,
  ext-holdout 39. All six native 50 MP frames fell into calibration or
  development, so the blind set has no native 50 MP frame — reported as a
  gap, not fixed by hand. Oracle sweeps (`tools/quality_oracle_labels.py
  sweep`, Balanced, seven knots, binary `jpxl-ref2` = f141360 default build)
  in three pinned lanes; labels in
  `.agent/scratch/camera-labels-20260903/labels.jsonl`.
* **Existing corpora.** The old corpus all-knot labels (91 images) and the
  paintings labels (258 images).
* **Features.** Unchanged: `qpv2-stp/1` (19 standardized source+transform
  features + 25 preanalysis values), extracted with `jpxl features --json
  --transform-summary` for the camera frames and appended to the study-2
  feature file (`.agent/scratch/firstguess3-20260903/features-rust.jsonl`).

## Method

Offline (`.agent/scratch/firstguess3-20260903/`, `study.py` = study 2 with
the camera labels as a third source):

1. **Coverage only** (`coverage.py`): the landed v2 weights and
   configuration (`searched-paint.json`: k=2, ratio 3.0, spread fallback
   0.405, offset 0.25×spread in the simulation) with the camera
   calibration+development cases added to the table. Blind = camera
   ext-holdout, paintings ext-holdout, old holdout. Also: the landed table
   *without* camera cases queried by the camera frames, and the deployed
   knot model on the camera rows — the two baselines the gain is measured
   against.
2. **Re-search** (`search.py --paintings --camera`): the same coordinate
   descent (objective: simulated Balanced + Fast probes, LOFO over
   calibration+development families of all three corpora, MIN_GAIN 0.02)
   from the landed configuration.

The arm that goes to the runtime trial is the coverage-only table unless
the re-searched configuration beats it by ≥ 0.10 simulated probes on
*both* blind sets (camera and paintings) — decided before either number
is known.

### Preregistered runtime trial (written before any camera label was analysed)

Reference arm R: HEAD's default build (case predictor off). Candidate C3:
`case-predictor` with the `qpv2-stp/1` table trained on the old corpus
(all splits) + paintings calibration+development + camera
calibration+development — both ext-holdouts are not in the table.
Comparison arm C2: the landed f141360 table (no camera cases) on the same
camera frames, so the camera gain is attributed to coverage alone (C3
differs from C2 only by the added cases when the coverage-only arm is
chosen). Harness `tools/one_shot_promotion_ab.py`, four threads, P-cores
0,2,4,6, host under sweep load (wall reported, not claimed).

1. Camera ext-holdout, seven targets, Balanced (39 × 7 = 273 cells): R, C2, C3 — the gate.
2. Camera ext-holdout, targets 30/50/70/85, Fast (156 cells): R vs C3.
3. Paintings ext-holdout, seven targets, Balanced (364 cells): R vs C3 — the
   regression check that photos did not displace paintings' neighbours
   (blind for C3 too).
4. Old-corpus holdout, seven targets, Balanced (140 cells): R vs C3 — the
   in-table regression check on synthetic and graphic content.

* **Pass**: zero floor violations on C3 in runs 1–4; run 1 byte geomean
  ≤ 1.01, mean pixel probes down ≥ 0.3 per encode against R, no cell with
  byte ratio > 1.15; run 2 zero encode errors, geomean ≤ 1.015; run 3
  geomean ≤ 1.01 and mean probes within +0.10 of the v2 trial's 1.77;
  run 4 geomean ≤ 1.01; thread-count identity (1 vs 4) on spot cells.
* **Partial**: floor held but a probe or byte criterion misses — report,
  keep off by default.
* **Fail**: any floor violation or encode error, or run-1 geomean > 1.01.

If Pass or Partial, the landed table is rebuilt from all splits of all
three corpora (the same rule as v2); the feature stays off by default and
`@jpegxl-rs.question.promote-case-predictor-2026-09-02` stays the
operator's.

### Amendment, 2026-09-03 10:50 (before any camera or CamSDD label was analysed)

Written after an advisor review of the scripts and after the operator
pointed at a second source; the camera sweeps were 209/220 done and no
label file existed yet.

1. **Second corpus: CamSDD test split.** One phone roll is one device's
   processing signature; the operator has no multi-camera large-frame set
   but does have CamSDD (`../raw-autotune/CamSDD`, CC BY-NC-SA 4.0): 11,100
   JPEGs at 576×384 in 30 scene classes (portrait, night, fireworks, text
   documents, computer screens, QR, neon, underwater, …), no EXIF. Its 600
   image `test` split (20 per class) → `test-set/camsdd/*.ppm` +
   `test-set/camsdd-manifest.json`, class `photo-small`, splits by family
   hash: calibration 316, development 176, ext-holdout 108. It covers the
   small-frame regime with the widest scene diversity available, the
   opposite corner from the camera roll. The tool gained `--license`.
2. **"Coverage only" means a frozen standardizer.** The v2 table
   standardizes features by per-feature median/MAD over the emitted cases.
   Adding 220 frames at 12–13 MP collapses the MAD of `log2_pixels` (0.747
   → 0.171, weight 4) and shifts `high_low_ratio`, `flat_fraction`,
   `pa_noise_q90`; with a refit standardizer 35 of 89 old-corpus images
   change their nearest neighbour without any label changing. So the
   primary coverage arm freezes centers and scales at the landed table's
   values (`train-cases --standardizer-from`, new option; the z-range still
   comes from the emitted cases) and only adds cases. The refit arm is
   reported as a diagnostic. Consequence to state: the z-range on
   `log2_pixels` now reaches 50 MP, so 13–50 MP frames are no longer
   `large_frame_only_ood` (which kept the case median and skipped the
   spread check) but in-range queries.
3. **Simulation mirrors the runtime.** The study-2 search configuration
   carried `spread_fb 0.6` and `offset_spread 0.25`; the runtime at f141360
   uses spread fallback 0.405 and has no spread offset. `coverage.py` and
   the search start (`--start-landed`) use the runtime values with the
   searched weights (`landed-config.json`); the offset is dropped from the
   simulation rather than added to the runtime.
4. **Decision rule, extended.** The re-searched configuration replaces the
   coverage-only table only if it wins by ≥ 0.10 simulated probes on *each*
   blind set (camera, CamSDD, paintings ext-holdouts), regresses no group's
   LOFO probes (old natural, graphic, synthetic, painting, camera, CamSDD)
   by more than 0.05, does not raise the fallback or Fast-fail rate on any
   blind set, and changes at most four weights and no configuration key.
   Otherwise coverage-only lands. The old holdout (17 families) cannot
   resolve 0.1 probes and is reported, not gated.
5. **Trial, extended.** Run 5: CamSDD ext-holdout, seven targets, Balanced
   (108 × 7 = 756 cells), R vs C2 vs C3, same Pass criteria as run 1 (0
   floor, geomean ≤ 1.01, probes down ≥ 0.3, no cell > 1.15). Both gates
   must pass for Pass.
6. **Additional reads before the trial:** case count in the report equals
   the images with all seven knots crossed in each corpus; per-feature
   scale ratios (frozen arm: identically 1); how many old-corpus and
   paintings queries pick a new-corpus neighbour; LOFO on the six native
   50 MP frames, labelled non-blind; `compared == cells` in every trial
   summary; run 1 sliced by target, HEIC/JPEG source and
   downscaled/native; Fast traces checked for rescue probes.
7. **Post-hoc temptations, declared:** no moving the 50 MP frames into the
   holdout; no change to MIN_GAIN, grid, search start or the 0.10 rule
   after seeing the search output; no spread threshold choice after blind
   numbers; no excluding downscaled or JPEG frames from a gate.

## Results

### Labels

Camera: 220 sweeps, 1540 rows, 0 censored (every frame has all seven knots).
CamSDD: 600 sweeps, 4200 rows, 0 censored. The 35 CamSDD sweeps that a
session restart killed were re-run by `chain2.sh`; the sweep tool is
idempotent per image.

### Offline study (`coverage.out`, runtime configuration, searched v2 weights)

Simulated Balanced probes per encode (mean over rows; `hit` = first probe
inside the band; `fb` = table fallback rate). Deployed = the knot model.
"Landed table" = the f141360 cases (old corpus + paintings) queried by the
new frames; "+cases" = the same geometry with the camera and CamSDD
calibration+development cases added.

| blind set (rows) | deployed | landed table | +cases, frozen std | +cases, refit std |
| --- | --- | --- | --- | --- |
| camera ext-holdout (273) | 2.415 / hit 0.13 | 2.527 / 0.13 / fb 0.09 | **1.725 / 0.36 / fb 0.02** | 1.641 / 0.39 |
| CamSDD ext-holdout (756) | 2.068 / 0.22 | 2.923 / 0.09 / fb 0.17 | **1.912 / 0.30 / fb 0.04** | 1.875 / 0.33 |
| paintings ext-holdout (364) | 2.08 | 1.853 / 0.34 | 1.863 / 0.31 | 1.765 / 0.34 |
| old holdout (133) | — | 2.505 (natural 2.84) | 2.071 (natural 1.80) | 2.060 |
| LOFO, all families (8169) | 2.176 | — | 1.881 | 1.829 |

The landed table without the new cases is *worse than the knot model* on
both new sets (camera +0.11, CamSDD +0.86 probes, with 9–17 % fallbacks):
the same pattern the v1 table showed on paintings. With the cases added
and the metric untouched, camera probes fall 2.42 → 1.73 and CamSDD 2.07
→ 1.91; paintings are unchanged (+0.01); the old natural holdout gains
its first close neighbours (2.84 → 1.80). Per-group LOFO under the frozen
arm: camera 1.74, CamSDD 1.88, graphic 1.99, natural 1.99, painting 1.80,
synthetic 2.78 — against the same-config table without the new cases
(natural 2.17, painting 1.77, graphic 1.99, synthetic 2.77): no group
regresses by more than 0.05. The refit-standardizer arm is 0.03–0.10
better on the blind sets but moves synthetic LOFO 2.78 → 2.84; it stays
diagnostic as preregistered. Non-blind: the six native 50 MP frames under
LOFO take 1.71 probes with one fallback in 42 rows, and their neighbours
are each other.

### Re-search from the landed configuration (`search-all-frozen.out`, `compare_search.py`)

Coordinate descent from the landed weights and runtime configuration,
frozen standardizer, objective = LOFO Balanced + Fast probes over the 939
calibration+development families of all four corpora (MIN_GAIN 0.02). Six
moves were accepted: `ln_chroma_q50` 4 → 2, `flat_fraction` 1 → 0,
`pa_share_texture` 0 → 2, `pa_ln_noise` 0 → 2, spread fallback 0.405 →
0.9, k 2 → 3. Search-set LOFO 2.067 → 1.781 (from the v1 unit-weight
start; the landed start is not printed separately by the script).

| blind set | coverage-only | re-searched | Δ |
| --- | --- | --- | --- |
| camera ext-holdout | 1.725 | 1.685 | −0.04 |
| CamSDD ext-holdout | 1.912 | 1.700 | −0.21 |
| paintings ext-holdout | 1.863 | 1.746 | −0.12 |
| old holdout | 2.071 | 2.098 | +0.03 |

Per-group LOFO deltas: camera −0.15, CamSDD −0.13, natural −0.13, painting
−0.08, graphic −0.02, synthetic +0.01; fallback rate 0.

**Decision (preregistered rule): coverage-only lands.** The re-searched
configuration misses the ≥ 0.10 margin on the camera set (−0.04) and
changes two configuration keys (k, spread fallback), either of which
disqualifies it under the amended rule. It is a legitimate future
candidate — most of its gain comes from the wider spread threshold and
k = 3 on the small CamSDD frames — but it is exactly the kind of re-tuning
the operator asked to avoid in this round, and a spread threshold of 0.9
was the value the v2 study rejected for raising simulated Fast refusals.
No further search was run.

### Table diagnostics before the trial (`cases-cov-report.json`, `neighbour_sources.py`)

* 968 cases = 89 old (all splits) + 206 paintings + 181 camera + 492 CamSDD
  calibration+development images with all seven knots; the report's
  centers and scales are byte-identical to the landed table's (frozen), so
  every per-feature scale ratio is 1. The `log2_pixels` z-range upper
  bound rises to 3.66 (50 MP).
* Nearest neighbour by source (table = calibration+development, query's
  own family excluded): camera queries 206/220 from camera, 13 paintings,
  1 old; CamSDD 597/600 from CamSDD; graphic 13/13 old; synthetic 43/51
  old, 6 CamSDD, 2 paintings; old natural 14/25 old, 9 camera, 2 paintings;
  paintings 232/258 paintings, 24 camera. 15 of 89 old-corpus and 24 of 258
  paintings queries now have a new-corpus nearest neighbour (they had a
  farther one before), which is the coverage effect on the old holdout
  natural rows (2.84 → 1.80 in the simulation).
* Binary: `jpxl` grows 65,338,520 → 65,663,768 bytes (+325 KB for 621
  cases, table only). The lookup is one 968 × 44 weighted distance scan per
  encode, microseconds against a 12 MP encode; instruction counts were not
  measured (`perf_event_paranoid` reset by the reboot) and wall is not
  claimed.

### Runtime trial

Binaries from the working tree at f141360 plus this study's tool changes:
R = `jpxl-ref3` (default build), C3 = `jpxl-cases-cov` (`case-predictor`,
968-case frozen-standardizer table, both ext-holdouts excluded), C2 =
`jpxl-cases-v2-landed` (the committed 347-case table). Two lanes: the
camera and paintings runs on P-cores 0,2,4,6, the CamSDD and old-holdout
runs on E-cores 12–19, four threads each; wall is reported per run but
not claimed. Results in `.agent/scratch/firstguess3-20260903/analysis.md`.

**Run 5 — CamSDD ext-holdout, Balanced, seven targets (756 cells, 0
errors, `compared == cells`).** Floor violations 0/0 (reference/band).
Byte geomean 0.9959, probes 2.15 → 1.94, encodes finishing in ≤ 2 probes
527 → 603 of 756, overshoot +1.01 → +0.91. Per target the gain is largest
at 90 (2.41 → 1.77) and absent at 95 (1.67 → 1.76). Worst cell 1.208
(`camsdd-0221` at 95: the case first guess landed at 94.49, the
neighbours' loss exponent 0.60 against the knot model's 0.74 made the
corrected step too short, the second probe stayed just under the band,
and the bracket step then overshot to 95.84, 60.6 kB against 50.1 kB);
one further cell at 1.122 (also t95); eight cells under 0.90. Against the
preregistered run-5 criteria this is **Partial**: the floor and geomean
criteria pass, the probe criterion (≥ 0.3) misses at 0.21 — the offline
simulation had predicted 0.16 for this set, but the threshold was fixed
before that number existed — and one cell exceeds 1.15.

**Run 5, comparison arm C2 (landed table, no CamSDD or camera cases) on
the same 756 cells.** Floor 0/0, byte geomean 1.0054, probes 2.15 → 2.66
(≤ 2 probes 527 → 364), worst 1.192, wall geomean 1.13. The landed table
is worse than the knot model on the small-frame photographs at every
target from 50 upward (t90: 2.41 → 3.74 probes) — the second runtime result in which a case
table without the query's content class has lost to the parametric
model (v1 on paintings 2.21 → 2.49); see run 1's C2 arm for the camera
set, where the runtime did not reproduce the simulation's loss.
The C3 − C2 difference, 2.66 → 1.94 on identical cells, is attributable to
the added cases alone: C2 and C3 share weights, configuration and
standardizer.

**Run 1 — camera ext-holdout, Balanced, seven targets (273 cells, 0
errors, `compared == cells`) — the gate.** Floor violations 0/0. Byte
geomean 0.9853, probes 2.65 → 1.85, encodes in ≤ 2 probes 91 → 220 of
273, overshoot +1.01 → +0.89, worst cell 1.138 (t30), best 0.800. Per
target the probe gain is 1.0–1.2 from 30 to 85 and 0.7 at 90; at 95 the
default already hit its first probe 38/39 times but overshot by +0.54,
and the case predictor lands closer (+0.18, 7 % fewer bytes) at the price
of 0.5 more probes. Slices: HEIC source (33 images) 0.983 / 2.65 → 1.86,
JPEG (6) 0.996 / 2.62 → 1.83; native 12 MP (33) 0.985 / 2.65 → 1.76,
Lanczos-downscaled 50 MP (6) 0.988 / 2.64 → 2.36 — the downscaled frames
gain least, which is consistent with their sharper per-pixel statistics
having few neighbours (six downscaled frames in calibration+development).
Six of 273 cells fell back to the knot model (`wide_interval`); decision
paths: one-shot 185, corrected 4, exact fallback 84. **All run-1
criteria pass** (0 floor, geomean ≤ 1.01, probes down ≥ 0.3, no cell
> 1.15).

**Run 1, comparison arm C2 (landed table, no camera or CamSDD cases) on
the same 273 cells.** Floor 0/0, byte geomean 0.9951, probes 2.65 → 1.99
(≤ 2 probes 91 → 204), worst 1.176 (t95), overshoot unchanged. So on this
phone roll the landed table already beats the default by 0.66 probes, and
the added cases contribute a further 1.99 → 1.85 with 1 % fewer bytes and
16 more two-probe encodes. **This contradicts the offline simulation**,
which had the landed table at 2.53 against the knot model's 2.42 on these
rows: the simulator charges every table fallback four probes and models
the corrected stop only approximately, and the runtime falls back to the
knot model at its real cost; the earlier "worse than the knot model" claim
for camera therefore stands only for the simulation, while for CamSDD the
runtime confirmed it (2.15 → 2.66). The coverage effect on 12 MP camera
frames is real but modest (−0.14 probes); on the small-frame CamSDD set
it is large (−0.72).

**Run 2 — camera ext-holdout, Fast, targets 30/50/70/85 (156 cells, 0
errors, all cells `met`).** Floor 0/0, byte geomean 0.9554, probes 1.73 →
1.79, overshoot +3.19 → +1.61, worst 1.078, best 0.837; candidate probe
histogram 1: 38, 2: 112, 3: 6, no work-cap or refusal. Fast's default
first probe lands well above the target (+3.2 points on average) and the
case first guess lands 1.6 above it, so bytes fall 4.5 % at the cost of
0.06 probes. Criteria pass (0 floor, 0 errors, geomean ≤ 1.015).

**Run 4 — old-corpus holdout, Balanced, seven targets (140 cells, 0
errors, in-table).** Floor 0/0, byte geomean 0.9814, probes 3.13 → 1.64,
worst 2.582 — the same sub-kilobyte gradient cell at t70 the v2 trial
reported (416 → 1074 bytes), best 0.583. Criteria pass (0 floor,
geomean ≤ 1.01); the tail cell is unchanged from v2 and remains a
navigator issue on a non-monotone sub-kB curve.

**Run 3 — paintings ext-holdout, Balanced, seven targets (364 cells, 0
errors; the added cases are blind, the geometry has seen this split
twice).** Floor 0/0, byte geomean 0.9889, probes 2.21 → 1.83 (v2 trial:
1.77; the criterion allowed 1.87), encodes in ≤ 2 probes 242 → 322 (v2:
322), worst 1.061, best 0.719. Adding 673 camera and CamSDD cases did
not displace the paintings' neighbours in any way that shows in the
encoder. Criteria pass.

**Thread identity.** Eight cells (two camera and two CamSDD ext-holdout
images at targets 50 and 90) encoded with C3 at 1 and 4 threads: 8/8
byte-identical streams.

### Verdict (preregistered criteria, amended 10:50)

| run | cells | floor | byte GM | probes | worst | criteria |
| --- | --- | --- | --- | --- | --- | --- |
| 1 camera Balanced (gate) | 273 | 0 | 0.985 | 2.65 → 1.85 | 1.138 | pass |
| 2 camera Fast | 156 | 0 | 0.955 | 1.73 → 1.79 | 1.078 | pass |
| 3 paintings Balanced | 364 | 0 | 0.989 | 2.21 → 1.83 | 1.061 | pass |
| 4 old holdout Balanced | 140 | 0 | 0.981 | 3.13 → 1.64 | 2.582 | pass (tail cell known) |
| 5 CamSDD Balanced (gate) | 756 | 0 | 0.996 | 2.15 → 1.94 | 1.208 | **partial** (Δ 0.21 < 0.3; one cell > 1.15) |

**Partial.** Every floor held on 1689 cells and every byte geomean is
below 1; the camera gate passes outright; the CamSDD gate misses its
probe margin and has one 1.21× cell. Under the preregistered outcome
list the table is rebuilt from all splits (1167 cases, frozen
standardizer, `cases-final-report.json`) and landed with the feature
still off by default; promotion remains
`@jpegxl-rs.question.promote-case-predictor-2026-09-02`.

### Disclosures

* The amendment (CamSDD, frozen standardizer, runtime-mirroring
  simulation, extended decision rule, run 5) was written at 10:50, after
  the camera sweeps were 209/220 done but before any label file existed
  and before the CamSDD sweeps ran. The run-5 probe threshold (0.3) was
  set before the simulation (which predicted 0.16) was seen.
* The offline simulation mispredicted the landed table on the camera
  set (worse than the knot model offline, 0.66 probes better in the
  encoder); an earlier sentence in this document that generalised the
  "worse without coverage" finding to three classes was corrected in
  place, and the CamSDD C2 arm is the runtime confirmation that
  survives.
* The trial ran while the search and other lanes shared the host; wall
  geomeans are in `analysis.md` but are not claimed.
* The coverage-only trial ran before the re-search's per-set comparison
  was read; the choice between the two arms was fixed by the amended
  rule (the re-search changed two configuration keys and missed the
  camera margin), not by the trial result.
* The preregistration's 50 MP intent was only partly met: six native 50
  MP frames are in the table, none in any blind set.
* 35 CamSDD sweeps were re-run after a session restart killed them
  (idempotent per image); the first build of the C3 table failed on a
  relative path and was re-run unchanged.

## Conclusion

Adding 220 camera frames and 600 small diverse photographs to the case
table, with the distance metric, weights and configuration frozen at the
landed values, removes 0.8 probes per encode on never-seen frames from
the same camera roll (2.65 → 1.85, 1.5 % fewer bytes) and 0.2 on
never-seen small photographs (2.15 → 1.94), holds every floor, and
leaves paintings and the synthetic holdout where they were. The landed
table without the small photographs is worse than the parametric model
on them; with the phone roll it was already better than the default and
the new cases add a modest further gain. Coverage, not per-content
tuning, is what the table needs; a class the table has not seen must
fall back to the knot model, which it does. The remaining coverage gap
is large frames from cameras other than one phone. The re-searched
configuration (k = 3, spread fallback 0.9) is a separate future
candidate that would need its own trial.
