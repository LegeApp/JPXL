# Case predictor promotion trial: an unseen camera, the re-searched configuration, and the default decision

Date: 2026-09-03. Follows [case-predictor-camera-coverage](2026-09-03-case-predictor-camera-coverage.md)
(commit 6e06b05: 1167-case table, frozen standardizer, feature `case-predictor` off by default).

## Question

Is the case-table crossing predictor, in totality, better than the default knot model
(`qpv2-st-1`, `predict_v2`) in quality, speed and size — including on a camera the table has
never seen — and if so, with which configuration: the landed geometry (k = 2, spread fallback
0.405, `weights-paint`) or the configuration the frozen-standardizer re-search found in the
previous round (k = 3, spread fallback 0.9, four weight changes) but which that round's
preregistered rule disqualified for changing configuration keys?

The operator delegated the promotion decision with the rule "promote if it is unequivocally
better in quality, speed and size", and asked for a limited set of the Sony frames and the
raw-autotune renders under `../raw-autotune` so that the labelling does not take four hours.

## Data

`test-set/rawcam` (manifest `test-set/rawcam-manifest.json`, gitignored like the other
corpora), built by `quality_corpus_extend.py` 1.2.0 from a staging directory of symlinks
(`.agent/scratch/rawcam-labels-20260903/src`, mapping in `SOURCES.txt`):

| source | files offered | picked |
| --- | --- | --- |
| `raw/jpeg` (Sony ILCE-7C in-camera JPEG, 6000×4000) | 42 | 7 |
| `raw/arw_better`, `indoor_tungsten`, `raw_backlit`, `raw_backlit2`, `raw_extremely_bright` (ILCE-7C JPEG) | 48 | 33 |
| `raw-autotune-output/raw_3rd_batch` (raw-autotune renders: ARW → 6000×4000, Samsung DNG → 4064×3044) | 58 | 8 (6 Sony, 2 Samsung) |

48 images, all at native resolution (`--max-pixels 25000000`, no downscaling), class `photo`,
splits calibration 27 / development 9 / ext-holdout 12 (`--holdout-fraction 0.25
--dev-fraction 0.25`), `--per-dir-cap 8`, `--burst-window-seconds 120`, and the new
`--fold-similar 0.9`: sequence-numbered Sony frames carry no time stamp, so a frame whose
32×32 luma thumbnail correlates ≥ 0.9 with the previous frame in name order joins its family.
A survey of consecutive-frame correlations put genuine near-duplicates at 0.94–0.96 and
distinct frames of one session mostly below 0.7. No picked scene appears both as an in-camera JPEG and as a raw-autotune render. The provenance strings name the staging
symlinks; `SOURCES.txt` maps them to the originals.

The 46 Sony frames are the first frames in any corpus from a camera other than the operator's
phone, and the first 24 MP frames. No rawcam image is in any case table used in the trial
below, so the whole set (336 cells) is blind for both candidate arms.

Labels: `quality_oracle_labels.py sweep`, 7 knots, Balanced, two lanes (8 E-core threads;
2 P-cores), `.agent/scratch/rawcam-labels-20260903/sweeps`.

## Arms (all built from the 6e06b05 tree, `.agent/scratch/firstguess4-20260903/bin`)

| binary | features | table | role |
| --- | --- | --- | --- |
| `jpxl-ref4` | default | none (knot model) | reference: what is on by default today |
| `jpxl-landed-full` | `case-predictor` | HEAD table: 1167 cases, landed cfg | arm B on rawcam |
| `jpxl-rs-cov` | `case-predictor` | cal+dev of the four corpora (968 cases), searched cfg | arm C on the existing blind sets |
| `jpxl-rs-full` | `case-predictor` | all 1167 cases, searched cfg | arm C on rawcam |

Searched cfg: `weights-searched.json` (20 non-zero weights from
`firstguess3/searched-all-frozen.json`), k = 3, neighbour ratio 3.0, spread fallback 0.9,
frozen standardizer. Arm B's results on the existing blind sets are the previous round's
runs 1–5 (`cases-cov`, same 968-case membership and landed cfg); they are not re-run.

Preanalysis is computed by the CLI for every build, so the default already pays for it; the
predictor's own extra cost is a 44-feature distance over 1167 cases, and the speed difference
between arms is the probe count.

## Preregistered decision rule (written 2026-09-03 15:36 local, before any rawcam label or trial row was read)

Sets (harness `one_shot_promotion_ab.py`, threads 4; bytes and probes are thread-count
independent, wall is not claimed):

| set | cells | blind for B | blind for C |
| --- | --- | --- | --- |
| S1 camera ext-holdout, Balanced, 7 targets | 273 | yes (run 1) | yes |
| S2 camera ext-holdout, Fast, 4 targets | 156 | yes (run 2) | yes |
| S3 paintings ext-holdout, Balanced, 7 targets | 364 | yes (run 3) | yes |
| S4 old-corpus holdout, Balanced, 7 targets | 140 | no (in table) | no |
| S5 CamSDD ext-holdout, Balanced, 7 targets | 756 | yes (run 5) | yes |
| S6 rawcam, all splits, Balanced, 7 targets | 336 | yes | yes |

Per-set criteria, Balanced (S1, S3, S5, S6):

1. quality: no cell whose achieved SSIMULACRA2 score is under the target (the reference is
   expected to have none either; if it has some, the candidate may have no more);
2. size: byte geomean candidate/reference ≤ 1.005, no cell over 1.25, at most 2 % of cells
   over 1.10;
3. speed: mean probes lower than the reference by at least 0.10.

Fast (S2): quality as above, byte geomean ≤ 1.015, probes not higher by more than 0.10.
S4 (in table for both arms, sanity only): quality as above, byte geomean ≤ 1.01.

An arm is promotable when it meets every criterion on S1, S2, S3, S5 and S6 and the S4 sanity.
If both arms are promotable, the one with fewer pooled probes over S1+S3+S5+S6 is chosen unless
the other's pooled byte geomean is better by more than 0.5 %. If exactly one is promotable it is
chosen. If neither is, the feature stays off and the round records why.

After the choice: the table is rebuilt with the rawcam calibration and development cases added
(frozen standardizer, chosen configuration), and checked on the rawcam ext-holdout (12 images,
84 cells) against the reference: quality as above, byte geomean ≤ 1.01, mean probes ≤ the
reference; thread identity on 8 spot cells. Only if that check passes is `case-predictor` added
to the default features of `jpxl-encode-policy` and `jpxl-cli`, followed by the usual gates
(clippy default / feature / no-default, fmt, workspace tests, tool tests). A quiet pinned wall
spot-check on a few cells is reported as a sanity figure, not a criterion.

Nothing in the rawcam labels is used to tune anything in this round; they extend the table's
coverage after the decision, and they feed the offline diagnostics.

## Results

### Labels

48 sweeps, 336 label rows, none censored (every knot crossed), two lanes, 2 h 10 min wall
from launch to the last sweep on a host that was also running the trial lanes.

### Trial (analysis `.agent/scratch/firstguess4-20260903/analyze4.py`, output `analysis4.txt`)

Floor = cells whose achieved score is under the target (candidate / reference). Bytes = geomean
of candidate/reference. Probes = mean encoder probes, reference → candidate. Wall = geomean of
the per-cell candidate/reference wall ratio measured by the harness on a fully loaded host,
shown as a sanity figure only.

| set | arm | cells | floor | bytes | worst | >1.10 | probes | wall | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 camera Balanced | B | 273 | 0/0 | 0.9853 | 1.138 | 1 | 2.65 → 1.85 | 0.81 | pass |
| S1 camera Balanced | C | 273 | 0/0 | 0.9881 | 1.063 | 0 | 2.65 → 1.74 | 0.79 | pass |
| S2 camera Fast | B | 156 | 0/0 | 0.9554 | 1.078 | 0 | 1.73 → 1.79 | 0.89 | pass |
| S2 camera Fast | C | 156 | 0/0 | 0.9513 | 1.068 | 0 | 1.73 → 1.74 | 0.97 | pass |
| S3 paintings Balanced | B | 364 | 0/0 | 0.9889 | 1.061 | 0 | 2.21 → 1.83 | 0.88 | pass |
| S3 paintings Balanced | C | 364 | 0/0 | 0.9875 | 1.091 | 0 | 2.21 → 1.80 | 0.90 | pass |
| S4 old holdout (in table) | B | 140 | 0/0 | 0.9814 | 2.582 | 5 | 3.13 → 1.64 | 0.73 | pass |
| S4 old holdout (in table) | C | 140 | 0/0 | 0.9814 | 2.582 | 5 | 3.13 → 1.64 | 0.75 | pass |
| S5 CamSDD Balanced | B | 756 | 0/0 | 0.9959 | 1.208 | 2 | 2.15 → 1.94 | 0.94 | pass |
| S5 CamSDD Balanced | C | 756 | 0/0 | 0.9959 | 1.203 | 1 | 2.15 → 1.80 | 0.93 | pass |
| S6 rawcam Balanced (unseen camera) | B | 336 | 0/0 | 0.9961 | 1.172 | 2 | 2.46 → 2.11 | 0.91 | pass |
| S6 rawcam Balanced (unseen camera) | C | 336 | 0/0 | 0.9971 | 1.172 | 3 | 2.46 → 2.15 | 0.91 | pass |

Pooled over S1 + S3 + S5 + S6 (1729 blind Balanced cells): B bytes 0.9928, probes 2.302 → 1.936;
C bytes 0.9931, probes 2.302 → 1.860. The two reference binaries (ref3 from the previous round,
ref4 from this one) are byte-identical on all 1689 shared cells. S4 is identical for both arms:
every holdout image is its own nearest case, and the ratio cut leaves no second neighbour to
differ over.

Both arms are promotable. The tie-break picks C: 0.076 fewer pooled probes, pooled bytes within
0.03 %. On the unseen camera alone B is marginally ahead (2.11 vs 2.15 probes, 0.9961 vs
0.9971), which the pooled rule does not weigh. On the unseen camera the worst cell for both arms
is one Sony frame at target 30 (1.172), and the t30/t50 byte geomeans are slightly above 1
(B 1.005/1.006, C 1.008/1.010) while t90/t95 are 2–2.5 % below; the per-set geomean carries the
criterion and no per-target figure exceeds the tail bounds.

### Offline diagnostics on the rawcam labels (`rawcam_study.py`, `rawcam-study.out`)

Simulated Balanced probes on the 48 unseen frames: knot model 2.34; landed cfg with the 1167-case
table 1.88; searched cfg 1.88. Nearest-case distance of a rawcam frame to the shipped table:
median 6.0 (q90 7.8), against 4.5 (q90 11.4) for the phone frames to the table without them in
the previous round — farther from the table on the median, and the predictor still helps. LOFO
over everything with rawcam added: landed 1.88, searched 1.76 probes; per group rawcam 1.80 /
1.71.

### Final table and check

`train-cases` over all splits of the four existing corpora plus rawcam calibration and
development (1203 cases; rawcam ext-holdout excluded), searched weights, k = 3, ratio 3.0,
spread fallback 0.9, frozen standardizer (`cases-final-rs-report.json`; LOFO median
|ln error| 0.071, p90 0.209). Blind check on the 12 rawcam ext-holdout images, 84 cells:
0 floor, bytes 0.9905, worst 1.073, probes 2.38 → 2.11; thread identity 8/8 at 1 and 8 threads.
The landed configuration's final table was checked the same way (0 floor, 0.9868, worst 1.075,
2.38 → 2.13) and set aside.

### Promotion and gates

`case-predictor` is added to the default features of `jpxl-encode-policy` (the CLI inherits
it). The default-feature build is byte-identical to the trial binary `jpxl-final-rs` on 6/6 spot
cells. Gates (`gates4.sh`, `gates4.log`): clippy `-D warnings` on the workspace (default) and on
the policy crate with `--no-default-features`, `cargo fmt --check`, the release workspace test
suite (76 result lines, 1638 passed, 0 failed), and the 88 Python tool tests all pass. The policy
crate's `--no-default-features` test run has three failures (`winner_stability_over_a_real_rate_search`,
`a_16_bit_source_round_trips_at_16_bit_precision`, `the_wide_entry_point_agrees_with_the_8_bit_one_on_8_bit_input`);
the same three fail at 6e06b05 with this round's changes stashed, so they predate it and are not
a gate of this round.

### Disclosures

- The rawcam set was labelled with this round's reference binary; the labels were not used for
  any tuning, only for the final table's coverage and the offline diagnostics.
- Six of the eight raw-autotune renders are of Sony ARW files whose in-camera JPEGs are also in
  the source pool; the sampler picked no scene in both forms.
- Wall figures come from the harness on a host running three lanes; they agree with the probe
  savings and are not a claim.
- The shipped table embeds statistics derived from the CamSDD test split (CC BY-NC-SA 4.0), as it
  has since 6e06b05; the rawcam frames are private.
- The preregistered rule chose on pooled probes; a rule weighting the unseen camera alone would
  have picked B. The two differ by 0.04 probes there.

## Conclusion

On 1729 blind Balanced cells over five corpora, including 336 cells from a camera the table had
never seen, the case predictor holds every quality floor, is 0.7 % smaller and needs 0.44 fewer
probes per encode than the knot model; on 156 Fast cells it is 4.9 % smaller at neutral probes.
The re-searched configuration (k = 3, spread 0.9) is chosen and the feature is on by default.
