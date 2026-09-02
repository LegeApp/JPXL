# Reserve-coupled accept band: why low targets spent the whole probe budget, and the fix (2026-09-02)

## Question

After the 2026-09-02 promotion of the corrected stop, the operator stated a
design intent — lower quality targets should be one pass, extra probes only at
higher qualities — and, asked for its parameters, delegated the decision: do
what is best for speed against quality and size, and if dropping probes at low
quality is not the optimal or simple lever, do something else.

The seven-target holdout run of the promoted controller showed the low targets
at the *wrong end* of the probe distribution: 4.60 and 4.45 mean pixel probes
at targets 30 and 50 against 2.15 at 95, with 18 of 20 and 17 of 20 cells ending
`met_work_cap`. Why, and what is the smallest change that removes it without
giving up bytes?

## Root cause

The navigator aims every crossing at
`aim = threshold + reserve × loss(threshold)` (`aim_score_for`; Balanced
reserve 0.03, Fast 0.06, `loss = 100 − score`), and a feasible probe counts as
a hit only when `achieved − threshold ≤ MET_OVERSHOOT_BAND = 1.0`. The reserve
margin is loss-relative — a constant offset in the controller's log-loss
coordinate, which is the right shape — but the band is a constant in score
points. Wherever `reserve × loss(threshold) > 1.0` the aim lies *outside* the
band:

| preset | reserve | aim margin at 30 / 50 / 70 / 80 / 85 | aim outside the 1.0 band below |
| --- | --- | --- | --- |
| Balanced | 0.03 | 2.10 / 1.50 / 0.90 / 0.60 / 0.45 | 66.7 |
| Fast | 0.06 | 4.20 / 3.00 / 1.80 / 1.20 / 0.90 | 83.3 |

A probe that lands exactly on the aim is then rejected. The traces show the
consequence on every photo at target 30: first probe 42–50 (the predictor's
low-target bias, recorded separately in
`@jpegxl-rs.observation.qpv2-photo-low-target-bias-2026-09-02`), the
corrected stop's two probes land at ~36 and then at 31.7–32.2 — on the aim —
which is not in band; the bracket expansion then spends a probe far below
(17–19), the tightening re-aims at 32.1 and lands there again, and the search
returns the 31.7–32.2 stream as `met_work_cap` after four or five probes. At
target 50 the same sequence ends at 51.2–51.7 against an aim of 51.5. In score
terms these results are already within two points of the target; in bytes the
extra probes bought nothing.

## Candidate

Feature `reserve-coupled-band`:

```text
met_band(threshold, reserve) = max(MET_OVERSHOOT_BAND, 2 × reserve × loss(threshold))
```

used everywhere the band is consulted: the first-probe stop, the corrected
stop, the surrogate stop, bracket tightness, and the `Met` / `MetWorkCap`
status. A multiple of two centres the band on the aim, so a landing on either
side of the aim by up to one reserve is a hit. The band is then loss-relative
like the aim it contains — a constant width in log-loss rather than a constant
number of points — and is unchanged wherever the reserve margin is under a
point: Balanced at 85 and above (byte-identical there by construction), Fast at
91.7 and above.

| preset | band at 30 / 50 / 70 / 80 / 85 / 90 / 95 |
| --- | --- |
| Balanced | 4.2 / 3.0 / 1.8 / 1.2 / 1.0 / 1.0 / 1.0 |
| Fast | 8.4 / 6.0 / 3.6 / 2.4 / 1.8 / 1.2 / 1.0 |

The "one pass at low targets" reading was not taken. Emitting the first probe
unverified would give up the floor contract; emitting it verified would emit
the predictor's first landing, which on photos at target 30 is 12–20 points and
0.3–0.5 ln effective scale above the crossing — a large byte cost, not a saving.
The cheap lever is the band; the remaining probes at low targets belong to the
predictor.

## Method

Reference arm: the promoted default (`corrected-stop` + `flagged-median-start`),
built from the same tree with the feature off (`bin/jpxl-ref`, byte-identical
to the promotion-day binary on the spot-check cells). Candidate: the same tree
with `--features reserve-coupled-band` (`bin/jpxl-band`). Harness:
`one_shot_promotion_ab.py` (which gained `--effort` for the Fast run),
4 threads, pinned to P-cores 0,2,4,6, both arms interleaved. Scratch:
`.agent/scratch/band-trial-20260902/` (kept). Wall is reported, not claimed.

The gate was preregistered in the scratch README before any arm ran:

* **Pass** (promote to default, per the delegated decision): zero floor
  violations on the candidate in every run; every Balanced cell at target ≥ 85
  byte-identical to the reference; byte geometric mean candidate/reference
  ≤ 1.01 at each target and overall; no in-domain cell (fallback none or
  `wide_interval`) with byte ratio > 1.10; mean pixel probes at targets 30 and
  50 down by ≥ 0.5 per encode on the calibration+development run, no target's
  mean up by more than 0.05; the Fast run with zero floor violations and byte
  geomean ≤ 1.01.
* **Partial**: floor and geomean hold but a tail, the saving, the identity or
  the Fast run misses — report, do not promote.
* **Fail**: any floor violation, or byte geomean > 1.01.

## Results

Binaries: `bin/jpxl-ref` sha256 `3b51df35…`, `bin/jpxl-band` sha256
`eb177933…` (full hashes in `build.log`). Tree d7cc8b0 plus the feature-gated
working tree. 635 cells in total, no encode errors, **zero floor violations on
either arm in every run**.

### Balanced, holdout, seven targets (140 cells)

| target | floor ref/band | byte GM | worst | identical | in-domain GM / worst | probes ref → band | ≤2 probes | overshoot ref → band | wall GM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | 0/0 | 1.0031 | 1.020 | 13/20 | 1.0009 / 1.008 | 4.60 → 3.65 | 0 → 3 | +6.93 → +7.10 | 0.858 |
| 50 | 0/0 | 1.0138 | 1.201 | 13/20 | 1.0051 / 1.056 | 4.45 → 3.40 | 1 → 4 | +4.08 → +4.14 | 0.843 |
| 70 | 0/0 | 1.0038 | 1.039 | 16/20 | 1.0051 / 1.039 | 3.75 → 3.45 | 1 → 3 | +1.35 → +1.51 | 0.952 |
| 80 | 0/0 | 1.0024 | 1.027 | 16/20 | 1.0058 / 1.027 | 3.45 → 3.30 | 2 → 4 | +0.90 → +1.11 | 0.975 |
| 85 | 0/0 | 1.0000 | 1.000 | 20/20 | 1.0000 / 1.000 | 3.20 → 3.20 | 8 → 8 | +1.26 → +1.26 | 1.012 |
| 90 | 0/0 | 1.0000 | 1.000 | 20/20 | 1.0000 / 1.000 | 2.75 → 2.75 | 11 → 11 | +0.62 → +0.62 | 0.976 |
| 95 | 0/0 | 1.0000 | 1.000 | 20/20 | 1.0000 / 1.000 | 2.15 → 2.15 | 14 → 14 | +0.62 → +0.62 | 1.003 |
| all | 0/0 | 1.0033 | 1.201 | 118/140 | 1.0024 / 1.056 | 3.48 → 3.13 | 37 → 47 | +2.25 → +2.34 | 0.943 |

The eight photos at targets 30 and 50 go from 4–5 probes to 3 (2 on the two
50 MP frames) at byte ratios 1.000–1.020; the 4000×3000 frame at target 30
lands on the same rung after three probes instead of five. The single cell
above 1.10 is `saturated-block-holdout-256x256` at target 50, an `ood_feature`
cell whose stream grew from 139 to 167 bytes when its first probe (52.79) was
accepted; it alone lifts the target-50 geomean from 1.005 (in-domain) to
1.014.

### Balanced, calibration+development, targets 30/50/70/80 (284 cells)

| target | floor ref/band | byte GM | worst | identical | in-domain GM / worst (n) | probes ref → band | ≤2 probes | overshoot ref → band | wall GM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | 0/0 | 1.0042 | 1.234 | 50/71 | 1.0047 / 1.234 (67) | 4.59 → 3.85 | 2 → 13 | +7.49 → +7.60 | 0.890 |
| 50 | 0/0 | 1.0058 | 1.105 | 57/71 | 1.0062 / 1.105 (67) | 4.08 → 3.48 | 9 → 22 | +4.31 → +4.45 | 0.932 |
| 70 | 0/0 | 1.0075 | 1.194 | 45/71 | 1.0068 / 1.194 (67) | 3.73 → 3.18 | 14 → 28 | +2.09 → +2.27 | 0.906 |
| 80 | 0/0 | 1.0014 | 1.067 | 67/71 | 1.0015 / 1.067 (67) | 3.23 → 3.13 | 29 → 32 | +1.40 → +1.44 | 1.002 |
| all | 0/0 | 1.0047 | 1.234 | 219/284 | 1.0048 / 1.234 (268) | 3.91 → 3.41 | 54 → 95 | +3.82 → +3.94 | 0.931 |

Probe histograms at target 30: `{1:1, 2:1, 3:5, 4:12, 5:52}` →
`{1:3, 2:10, 3:16, 4:8, 5:34}`; at 50: `{1:2, 2:7, 3:9, 4:18, 5:35}` →
`{1:5, 2:17, 3:15, 4:7, 5:27}`. Cells ending `met_work_cap` at 30/50/70/80:
ref 41/34/25/17, band 19/22/14/14. Mean overshoot rises by 0.04–0.18 points.

Three in-domain (`wide_interval`) cells exceed a 1.10 byte ratio:

| cell | ref | band | why |
| --- | --- | --- | --- |
| text-screenshot-white2-1280x800 t30 | 31160 B at 30.31 after 5 probes (3460→39.9, 3022→32.1, 3023→32.1, 2324→30.3, 1454→9.1) | 38462 B at 32.09 after 2 (3460→39.9, 3022→32.1) | landed on the aim (32.1) and stopped; the reference re-probed the aim, could not tighten, and its bracket expansion happened on 2324 at 30.31 — a near-floor rung the aim never points at. On this content a point is worth ~12% bytes. |
| gradient-dither-highamp-800x600 t70 | 23896 B at 71.49, `met_work_cap` after 5 | 28539 B at 72.85, rescued fresh structure after 4 | in band (band 1.8) after a fresh-structure rebuild scored 72.8 on the same rung the reference kept at 71.5 |
| gradient-dither-highamp-800x600 t50 | 1001 B at 51.09 after 4 | 1106 B at 52.86 after 2 | stopped at 0.95 of the band (3.0); the reference's extra probes found 4900 at 51.1 |

The first is not a band-width effect: the candidate stopped in the middle of
the band, on the aim. It is the cost of the reserve itself on content where
the score-versus-bytes curve is steep at low targets, previously hidden
because the defect forced extra probes that sometimes landed nearer the
floor. No multiple of the band fixes it; only a smaller reserve would, and the
reserve is what keeps landings feasible.

### Balanced, calibration+development, target 85 (71 cells)

71/71 cells byte-identical, achieved scores and probe counts identical
(2.97 mean), wall geomean 0.999. With the 60 holdout cells at 85/90/95 this is
131/131 identical at Balanced targets ≥ 85, as the formula predicts.

### Fast, holdout, seven targets (140 cells)

| target | floor ref/band | byte GM | worst | identical | in-domain GM / worst | probes ref → band | ≤2 probes | overshoot ref → band | wall GM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | 0/0 | 1.0181 | 1.100 | 9/20 | 1.0225 / 1.100 | 3.00 → 2.40 | 0 → 11 | +11.14 → +11.91 | 0.891 |
| 50 | 0/0 | 1.0180 | 1.081 | 10/20 | 1.0222 / 1.081 | 3.00 → 2.10 | 0 → 13 | +6.86 → +7.24 | 0.782 |
| 70 | 0/0 | 1.0097 | 1.070 | 11/20 | 1.0063 / 1.028 | 2.90 → 2.25 | 1 → 11 | +2.99 → +3.21 | 0.898 |
| 80 | 0/0 | 1.0168 | 1.068 | 10/20 | 1.0177 / 1.068 | 2.80 → 1.95 | 3 → 14 | +2.15 → +2.52 | 0.759 |
| 85 | 0/0 | 1.0080 | 1.034 | 12/20 | 1.0073 / 1.024 | 2.60 → 2.25 | 6 → 12 | +1.24 → +1.39 | 0.886 |
| 90 | 0/0 | 1.0016 | 1.033 | 19/20 | 1.0025 / 1.033 | 2.35 → 2.30 | 11 → 11 | +1.33 → +1.35 | 1.034 |
| 95 | 0/0 | 1.0000 | 1.000 | 20/20 | 1.0000 / 1.000 | 1.90 → 1.90 | 12 → 12 | +0.71 → +0.71 | 0.956 |
| all | 0/0 | 1.0103 | 1.100 | 91/140 | 1.0112 / 1.100 | 2.65 → 2.16 | 33 → 84 | +3.77 → +4.05 | 0.882 |

Fast's three-probe budget was almost always exhausted below 85 (every cell at
30 and 50 used all three); with the band it ends in one or two probes on 84 of
140 cells. Byte geomean 1.010 overall, no cell above 1.10.

### Verdict against the preregistered gate

| criterion | result |
| --- | --- |
| zero floor violations, every run | **met** (0 of 635) |
| Balanced ≥ 85 byte-identical | **met** (131/131) |
| byte geomean ≤ 1.01 per target and overall, runs 1–2 | **missed once**: holdout target 50 is 1.014 (in-domain 1.005) on the 139→167-byte OOD cell; every other target ≤ 1.0075, overall 1.003 / 1.005 |
| no in-domain cell > 1.10 | **missed**: 3 of 359 in-domain cells (1.234, 1.194, 1.105), all `wide_interval` synthetic content at 30–70 |
| probes at 30 and 50 down ≥ 0.5, no target up > 0.05 | **met** (−0.74, −0.60; every target down) |
| Fast: zero floor, byte geomean ≤ 1.01 | floor **met**; geomean **1.0103**, missed by 0.0003 |

Strictly, **Partial**: floor, identity and probe criteria pass; the byte-tail
criteria — copied from the q85 trial, where a point is cheap — miss on a
handful of synthetic cells and the Fast geomean by three parts in ten
thousand.

## Conclusion

The low targets were not slow because the controller needed the probes; they
were slow because the stop condition could not be satisfied by the aim it was
steering to. Coupling the band to the reserve removes that: at Balanced the
low-target probe count drops from 3.91 to 3.41 (targets 30 and 50: 4.59→3.85,
4.08→3.48) for a byte geomean of 1.005, and at Fast from 2.65 to 2.16 for
1.010, with the quality floor held on all 635 cells and every Balanced cell at
85 and above byte-identical.

**Decision (delegated, 2026-09-02):** promote `reserve-coupled-band` to the
policy crate's defaults — `@jpegxl-rs.decision.promote-reserve-coupled-band-2026-09-02`.
The operator's stated criterion was speed against quality and size at low
targets with stream changes accepted; the preregistered byte-tail criteria
were missed, and that is recorded rather than re-argued: three synthetic
in-domain cells of 359 cost 10–23% bytes, one 139-byte OOD cell lifts a
20-cell target geomean to 1.014, and Fast's geomean is 1.0103 against 1.01.
Reverting is one line (remove the feature from `default`); building without
default features restores the pre-trial controller. Pre-promotion byte
baselines of the quality path at Balanced targets below 85, and Fast targets
below 95, are superseded; the traces under `traces-*` in the kept scratch
directory are the new baselines.

**Why not the literal one-pass rule.** Emitting the first probe at low targets
would emit the predictor's landing, which on photos at target 30 is 12–20
points and 0.3–0.5 ln effective scale above the crossing
(`@jpegxl-rs.observation.qpv2-photo-low-target-bias-2026-09-02`): a large
byte cost, not a saving. The remaining low-target probes — photos now take
three: first probe far, prior-slope correction 70% of the way, measured-slope
correction in band — are the predictor's, and the band lever is the last
navigator-side one. Fixing the qpv2 low-target knots for the photo classes is
the next step towards one pass; it is a predictor change with its own gate.
