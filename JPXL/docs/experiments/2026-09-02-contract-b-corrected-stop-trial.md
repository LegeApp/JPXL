# Contract B trial: trusted first probe, corrected stop, and the large-frame median start (2026-09-02)

## Question

The 2026-09-02 offline replay
(`2026-09-02-two-probe-corrected-stop-replay.md`) named two bounded
Contract B candidates on the navigator side and left the decision to the
operator. The operator authorised a feature-gated trial of both, and of the
one-shot calibration lever on its own, measured alone and combined. Does each
candidate, encoded for real by the production binary, hold the quality floor,
keep bytes neutral, and remove the probes the replay predicted? Are the two
candidates independent?

The three levers under test:

* **trust** (`trusted-first-probe`): a first canonical probe that lands
  feasible inside the one-point accept band ends navigation whatever the
  crossing model's confidence said. This is the N3 stop with its confidence
  gate removed — the "one-shot calibration" lever, without retraining.
* **corrected** (`corrected-stop`, implies `trust`): after a first probe outside
  the band, one direct correction aimed at the crossing from the measured loss
  with the prior exponent 0.9, only when the implied move is at most 0.5 in ln
  effective scale; if that misses, one more correction through the slope the
  two probes measured, anchored at the probe nearer the aim. A probe landing in
  band ends navigation; anything else hands over to the unchanged bracket
  expansion with every probe kept. The replay's `prior|g0.5|trust=1|c2=1`.
* **median** (`flagged-median-start`): a frame whose only out-of-distribution
  signal is being larger than the training domain (`log2_pixels` above its
  range and no other flag) seeds the navigator with the model's median rung
  instead of the legacy table start, all canonical, no slope prior.

## Preregistered gate

Written in `.agent/scratch/contract-b-trial-20260902/README.md` before any arm
ran:

* **Pass** (promotable as a default): zero floor violations on the candidate
  arm; byte geometric mean candidate/default at most 1.005; no in-domain cell
  (prediction fallback none or `wide_interval`) with byte ratio above 1.10;
  mean baseline pixel probes reduced by at least 0.3 per encode; t1/t4 byte
  identity on the spot-check cells; the 7-target holdout run shows zero floor
  violations and byte geomean at most 1.01.
* **Partial**: zero floor violations and byte geomean at most 1.01, but the
  tail, the probe reduction, or the off-target run misses.
* **Fail**: any floor violation, or byte geomean above 1.01.

Additivity: the combined arm's probe saving is compared with the sum of the
single-arm savings; independent if within 0.05 probes.

Wall is reported but not claimed: the arms differ in probe count, which is the
deterministic quantity, and the host is not a quiet benchmarking host.

## Method

Everything ran on the working tree at `22be3b9` plus the feature-gated
implementation this report is committed with. Nothing is in `default`; the
`default` arm is the deployed controller built from the same tree.

| arm | cargo features | binary sha256 |
| --- | --- | --- |
| default | (defaults) | `778a08ca…47a2` |
| trust | `trusted-first-probe` | `2ba2fe1b…2e2b` |
| corrected | `corrected-stop` | `83ae1ca4…bd1a` |
| median | `flagged-median-start` | `c7436f3d…b124` |
| both | `corrected-stop,flagged-median-start` | `8da72c9c…259f` |

Primary run, per arm against `default`, interleaved cell by cell:

```sh
taskset -c 0,2,4,6 python3 JPXL/tools/one_shot_promotion_ab.py \
  --manifest test-set/quality-corpus.json \
  --splits calibration development holdout --targets 85 --threads 4 \
  --default-binary .agent/scratch/contract-b-trial-20260902/bin/jpxl-default \
  --one-shot-binary .agent/scratch/contract-b-trial-20260902/bin/jpxl-<arm> \
  --output .agent/scratch/contract-b-trial-20260902/ab-q85-<arm>.json \
  --keep-traces .agent/scratch/contract-b-trial-20260902/traces-<arm>
```

`one_shot_promotion_ab.py` gained `--keep-traces` (append every
`jpxl.quality-trace/2` record) and per-cell `pixel_rungs` / `trace_status`
fields, plus the mean baseline pixel probes in its summary; nothing it already
reported changed. 91 of the 95 manifest images are scorable by the tool (it
skips frames under 64 px on a side), the same cell set as the deployed q85
baseline once the tiny fixtures are excluded. `analyze.py` produces
`analysis.md` / `analysis.json` from the four outputs and the kept traces;
`routes.py` classifies each cell by the route the corrected stop took.

| input | sha256 |
| --- | --- |
| `test-set/quality-corpus.json` | `ec986536c986faa16a0df7d2bf8d4aef4e6bb61a8ec3e017a6a52938f7490681` |
| `ab-q85-trust.json` | `b34b46c04737d31c3fdbb36161b0ea16aa59dc6a84d212882d8a45e0dc6f5713` |
| `ab-q85-corrected.json` | `5b2071590762274b8039eb61c8864e721d3f2d61426e77f4cbcba9ba46a47c3b` |
| `ab-q85-median.json` | `1621a7f24f3fc05b211bcffd999f4585ff83729238614bab897ee7ec7567f01b` |
| `ab-q85-both.json` | `85c8abc94c065b852d4acba40581a13415d76f0adb0347a2624de855e2e9b04e` |
| `analyze.py` | `acadedd63d39b5a4bbc1d79dea7e75c0714f691645a36a520d28cc6e9bf1b6c2` |
| `identity-both.txt` | `650711f966207d4e87c4d786ab12bc71fbcc1a2a7e56ea839dbfe54ce1520536` |

Workspace gates before the runs: `cargo clippy --all-targets -- -D warnings`
clean for the policy and CLI crates under no trial feature, each trial feature
alone, and `corrected-stop,flagged-median-start`; the policy crate's tests pass
under default features and under both trial features (169 + 14 + 12 + 12 + 2 +
2); `cargo fmt --all --check` clean.

## Raw results: q85, all splits, 91 cells

| arm | floor def/cand | pixel probes def→cand | ≤2 probes | in-domain probes (n=80) | bytes GM | worst cell | cells >1.02 / <0.98 | cells with changed bytes | reconstructions | wall GM (≥4 MP) |
| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- | --- |
| trust | 0/0 | 3.462→3.374 | 17.6%→17.6% | 3.26→3.16 | 1.0000 | 1.000 | 0 / 0 | 0 | 315→307 | 0.968 (0.916) |
| corrected | 0/0 | 3.462→3.066 | 17.6%→42.9% | 3.26→2.81 | 0.9950 | 1.060 | 6 / 9 | 55 | 315→279 | 0.924 (0.916) |
| median | 0/0 | 3.462→3.407 | 17.6%→17.6% | 3.26→3.26 | 1.0010 | 1.040 | 3 / 0 | 3 | 315→310 | 0.993 (0.929) |
| both | 0/0 | 3.462→3.022 | 17.6%→42.9% | 3.26→2.81 | 0.9954 | 1.060 | 6 / 9 | 58 | 315→275 | 0.907 (0.832) |

Probe-count histograms (cells with 1/2/3/4/5 probes): default 4/12/40/8/27;
trust 12/4/40/8/27; corrected 12/27/20/7/25; median 4/12/42/9/24; both
12/27/21/9/22.

Smallest margin above the requested score on any candidate cell: 0.041 points
(default arm: 0.044). Mean overshoot above target: default 1.136, corrected
1.130, median 1.148, both 1.128.

In-domain byte geomean (80 cells): trust 1.0000, corrected 0.9998, both 0.9998;
worst in-domain cell 1.0596 (`gradient-dither-lowamp-1024x512`). The 0.9950
overall geomean is carried by one out-of-domain cell,
`gradient-chroma-shallow-green-holdout-640x640` at 0.681, where the corrected
route found a coarser feasible rung the deployed five-probe search missed.

**Additivity.** Saved probes: corrected 0.396 + median 0.055 = 0.451; both
0.440 (difference −0.011, inside the 0.05 bound). The two levers touch
disjoint cells except the three large photos, where the median start puts the
first probe close enough for the corrected stop to fire as well.

### Routes taken (both arm, 91 cells)

| route | cells | mean probes |
| --- | ---: | ---: |
| first probe in band (stop1) | 12 | 1.00 |
| first correction in band (stop2) | 27 | 2.00 |
| second correction in band (stop3) | 23 | 3.09 |
| move gate > 0.5, straight to expansion | 20 | 4.80 |
| both corrections missed, expansion | 9 | 4.67 |

The trust lever alone converts 8 two-probe cells to one probe (12 one-probe
cells in total against the deployed 4), all byte-identical to the default: the
N3 finding that the stop only fires where the search would pick the same rung
holds without the confidence gate.

### Per class (both arm)

| class | n | probes def→cand | bytes GM | worst |
| --- | ---: | --- | ---: | ---: |
| gradient | 25 | 3.56→3.44 | 0.985 | 1.060 |
| grayscale | 3 | 2.33→1.67 | 1.010 | 1.040 |
| line-art | 6 | 3.00→2.33 | 1.003 | 1.015 |
| noise-lowlight | 4 | 3.00→2.00 | 0.985 | 0.993 |
| photo | 11 | 3.36→2.55 | 1.005 | 1.032 |
| photo-scene | 7 | 2.57→2.00 | 1.002 | 1.019 |
| saturated | 26 | 4.12→3.77 | 0.998 | 1.022 |
| text-screenshot | 7 | 2.43→1.71 | 1.001 | 1.015 |
| tiny | 2 | 5.00→5.00 | 0.973 | 1.000 |

### The three large out-of-domain photos

These are the only cells the median start touches (`log2_pixels` is their
sole OOD flag). Deployed: legacy start 73727, five probes, a fresh-structure
rescue on every one.

| cell | deployed rungs | median arm | bytes | both arm | bytes | wall def→median→both (s) |
| --- | --- | --- | ---: | --- | ---: | --- |
| photo-large-20260606_150624 | 73727, 23243, 10178, 18580, 18580 | 34512, 16486, 19040, 19040 | 1.017 | 34512, 21317, 18111, 18111 | 1.005 | 17.3→15.1→— |
| photo-large-20260607_155040 | 73727, 27390, 14316, 24399, 24399 | 39351, 22042, 24590 | 1.034 | 39351, 28502, 23829 | 1.015 | 19.5→12.6→— |
| photo-large-20260607_155124 | 73727, 24116, 11423, 20545, 20545 | 36333, 18180, 21092 | 1.040 | 36333, 23507, 19193, 20422 | 1.018 | 18.1→12.0→— |

The median start alone pays 2-4% bytes on these cells because three probes
settle on a slightly finer rung than five did; with the corrected stop as
well the byte cost falls to 0.5-1.8%, and the status on all three goes from
`rescued_fresh_structure` to `met`.

### Byte-ratio tail (both arm)

Above 1.02: `gradient-dither-lowamp-1024x512` 1.060, `gradient-diamond-768x768`
1.044, `grayscale-photo-203230-crop-1024x1024` 1.040,
`photo-201839-mid2-2832x2124` 1.032, `gradient-angled-150-holdout-640x640`
1.028, `saturated-rings-400x400` 1.022. Below 0.98: nine cells, from 0.681 to
0.976. Status changes: three gradient cells go from `met` to `met_work_cap`
(the corrected route spent its budget tightening an already-feasible answer),
three saturated cells and the three large photos go the other way.

### Thread identity (both arm)

`identity_check.sh both` on eight cells spanning the classes and the 12 MP
anchor: t1 and t4 streams identical on 8/8 (`identity-both.txt`).

## Raw results: holdout split, seven targets

The same tool on the 20 holdout images at targets 30, 50, 70, 80, 85, 90 and
95 (140 cells per arm), the off-target generalisation check the gate asks for.

### both arm (`ab-holdout-7t-both.json`, sha256 `c4fc5144…85d7`)

| target | cells | floor def/cand | probes def→cand | bytes GM | worst cell | in-domain worst | min margin def/cand |
| ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| 30 | 20 | 0/0 | 4.35→4.60 | 1.0058 | 1.111 | 1.111 | 0.301 / 0.619 |
| 50 | 20 | 0/0 | 4.40→4.45 | 0.9910 | 1.020 | 1.020 | 0.894 / 0.166 |
| 70 | 20 | 0/0 | 4.05→3.75 | 0.9875 | 1.019 | 1.019 | 0.213 / 0.213 |
| 80 | 20 | 0/0 | 4.00→3.45 | 1.0095 | 1.293 | 1.009 | 0.253 / 0.342 |
| 85 | 20 | 0/0 | 3.85→3.20 | 0.9852 | 1.032 | 1.032 | 0.097 / 0.041 |
| 90 | 20 | 0/0 | 3.15→2.75 | 1.0098 | 1.092 | 1.058 | 0.010 / 0.010 |
| 95 | 20 | 0/0 | 2.60→2.15 | 0.9600 | 1.041 | 1.041 | 0.302 / 0.067 |
| all | 140 | 0/0 | 3.77→3.48 | 0.9925 | 1.293 | 1.111 | — |

In-domain (91 cells) byte geomean 1.0005. Cells above 1.02: 15; above 1.10:
2; below 0.98: 17. Reconstructions 528→487; wall geomean 0.929 (0.882 on the
≥ 4 MP cells).

The two cells above 1.10 are both `gradient-chroma-shallow-green-holdout-640x640`
at target 80 (1.293) and `saturated-holdout-640x480` at target 30 (1.111). The
first is the out-of-domain gradient whose loss curve is not monotone enough
for any local model: the same cell gives 0.800 at 70, 0.681 at 85 and 1.092 at
90 under the corrected route, with both arms ending on `met_work_cap` every
time — the search there is a coin toss in either arm, not a regression of the
candidate. The second is a low-target saturated cell where the corrected route
tightened to a finer rung and hit the work cap.

The low targets are where the corrected stop does not pay: at 30 and 50 the
mean probe count rises slightly (4.35→4.60, 4.40→4.45), because the first
probe is rarely within the 0.5 ln-scale gate there and the corrected probes
that do fire land wide, leaving the expansion to finish with less budget. The
saving concentrates at 70 and above, the production quality range.

The three large out-of-domain photos at target 95 are the largest single
win of the run: the median start lands in band on the first probe on all
three (one probe against the deployed four with a fresh-structure rescue),
at 0.81-0.85 of the deployed bytes.

### corrected arm (`ab-holdout-7t-corrected.json`)

sha256 `83950b1e…2235`.

| target | cells | floor def/cand | probes def→cand | bytes GM | worst cell | in-domain worst | min margin def/cand |
| ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| 30 | 20 | 0/0 | 4.35→4.55 | 1.0075 | 1.111 | 1.111 | 0.301 / 0.619 |
| 50 | 20 | 0/0 | 4.40→4.45 | 0.9918 | 1.020 | 1.020 | 0.894 / 0.166 |
| 70 | 20 | 0/0 | 4.05→3.90 | 0.9880 | 1.019 | 1.019 | 0.213 / 0.213 |
| 80 | 20 | 0/0 | 4.00→3.75 | 1.0094 | 1.293 | 1.009 | 0.253 / 0.342 |
| 85 | 20 | 0/0 | 3.85→3.40 | 0.9833 | 1.032 | 1.032 | 0.097 / 0.041 |
| 90 | 20 | 0/0 | 3.15→2.85 | 1.0097 | 1.092 | 1.058 | 0.010 / 0.010 |
| 95 | 20 | 0/0 | 2.60→2.60 | 0.9630 | 1.041 | 1.041 | 0.302 / 0.067 |
| all | 140 | 0/0 | 3.77→3.64 | 0.9931 | 1.293 | 1.111 | — |

In-domain byte geomean 1.0005; the same two cells above 1.10 as the combined
arm; reconstructions 528→510; wall geomean 0.968 (0.994 on the ≥ 4 MP cells).
The difference between this arm and the combined arm is exactly the median
start's contribution on the five large holdout photos: at 95 the corrected
stop alone cannot help them (the legacy start at 73727 is coarser than the
crossing by more than the gate) and they keep their four probes.



## Conclusion

**Corrected stop with the large-frame median start: pass** on every
preregistered criterion. Zero floor violations on 91 q85 cells and on 140
holdout cells at seven targets; byte geomean 0.9954 at q85 (in-domain 0.9998)
and 0.9925 on the holdout run (in-domain 1.0005), both inside the 1.005 /
1.01 bounds; worst in-domain q85 cell 1.060, under the 1.10 tail bound; mean
pixel probes 3.462→3.022, a saving of 0.44 per encode against the 0.3 bar and
in line with the replay's forecast of about 0.45; t1/t4 streams identical on
8/8 spot-check cells. The two levers are additive (0.396 + 0.055 against
0.440 combined).

**Corrected stop alone: pass** on the same terms (3.462→3.066, GM 0.9950,
holdout GM 0.9931, zero floor violations).

**Trusted first probe alone: partial.** It is the safest of the three —
every changed cell is byte-identical to the default, because the N3 stop only
fires where the search would pick the same rung — but at 0.088 probes per
encode it is under the 0.3 bar on its own. It is contained in the corrected
stop.

**Large-frame median start alone: partial.** It touches only the three
out-of-domain 12 MP photos on this corpus, so its corpus-mean saving (0.055
probes) cannot meet a per-encode bar; on the cells it touches it removes
one to two probes and a fresh-structure rescue each, for 12-35% less wall at
q85 and 2-4% more bytes, which the corrected stop then reduces to 0.5-1.8%.
At target 95 it takes those cells to one probe at 0.81-0.85 of the bytes.

What the trial did not deliver: the advisor's 1.5-probe / 90%-within-two
target. The combined arm reaches 43% of cells within two probes (49%
in-domain), as the replay predicted; the 20 gated cells and 9 double-miss
cells still average 4.7-4.8 probes, and the saturated class stays at 3.8.
That ceiling is the first-probe predictor's, as recorded in
`@jpegxl-rs.assessment.two-probe-path-is-predictor-bound-2026-09-02`.

Two honest caveats. The corrected stop is slightly negative at targets 30
and 50 (+0.2 and +0.05 probes, GM 1.006-1.008) and only pays from 70
upward, which covers Fast (70) and Balanced (85). And every candidate cell
changes the achieved-score distribution a little: the minimum margin above
the requested score falls from 0.097 to 0.041 at q85 and from 0.302 to 0.067
at 95, still above the floor on every cell but closer to it.


## Consequences

* The three features are committed **off by default**. Turning
  `corrected-stop` and `flagged-median-start` on in the policy crate's
  `default` set is the operator's Contract B promotion decision; this report
  is the corpus gate it asked for. If promoted, the deployed q85 baseline
  trace (`corpus-shadow.jsonl`, 3.51 mean probes) and the byte-identity
  matrices that pin the current controller must be re-baselined, since 58 of
  91 q85 streams change.
* The trusted-first-probe lever can be promoted on its own at any time with
  no byte change; it is the one-shot calibration question's zero-cost answer.
* Not pursued further here: retuning the move gate or the second correction
  for low targets, and the model-slope variants the replay already ranked
  below the prior.
* Artefacts are kept under `.agent/scratch/contract-b-trial-20260902/`
  (`akr scratch keep`).

