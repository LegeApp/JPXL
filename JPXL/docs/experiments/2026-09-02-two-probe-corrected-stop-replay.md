# Two-probe corrected-stop replay on the current corpus

## Question

Can a deterministic two-probe common case — the deployed first probe, then
one direct correction aimed at the crossing from the measured loss and a
slope estimate, accepted when it lands feasible inside the existing
one-point overshoot band, with the unchanged navigator as the fallback —
reach the advisor's probe target (mean canonical pixel probes at most 1.5,
at most 2 probes for 90% of in-domain requests, zero floor violations, no
matched-quality byte regression) at quality 85 Balanced?

## Preregistered gate

Set by the 2026-09-01 advisor pack and restated in the session handoff before
this replay was written:

* **Pass:** some variant reaches mean probes at most 1.5 and at most 2 probes
  on at least 90% of in-domain cells, with no simulated cell left without a
  feasible final rung and a byte geometric mean against the deployed path not
  above 1.0.
* **Partial:** a variant clears the floor and byte gates with a probe
  reduction worth a Contract-B trial (the standing bar from the frontier
  assessment is one avoided probe per encode, about 21-25% of the quality
  phase wall) but misses the 1.5 / 90% target.
* **Fail:** nothing clears the floor and byte gates, or the reduction is too
  small to justify a stream-changing trial.

The replay itself has a fidelity precondition: the Python port of the
navigator must reproduce the deployed probe sequences from the trace corpus
on the large majority of cells before its counterfactuals are read.

## Method

This is an offline replay of the JPXL controller against measured curves. No
image was encoded for this experiment and no timing was measured; the only
wall figures below are sums of the per-probe milliseconds already recorded in
the deployed trace, on an unpinned busy host, and are proxies only.

Inputs:

* the deployed q85 Balanced t4 trace corpus of 2026-09-01
  (`.agent/scratch/exact-base-encoder-20260901/corpus-shadow.jsonl`, schema
  `jpxl.quality-trace/2`, 94 scored fixtures) — the real first-probe rung and
  score, the real probe sequence, and the exact finalist bytes of every cell;
* the all-knot oracle sweeps and labels of 2026-09-02
  (`.agent/scratch/qpv2-retrain-20260902/all-knot-sweeps/*.jsonl` and
  `all-knot-labels.jsonl`) — the measured (effective scale, SSIMULACRA2)
  curve of each of 91 fixtures and the priced neighbourhood of the q85
  crossing.

The script `.agent/scratch/two-probe-replay-20260902/replay_two_probe.py`
ports the quantizer ladder (`quantizer_ladder.rs`) and the canonical
navigator core of `quality.rs` (`extrapolated_step_from`,
`crossing_aim_from`, `log_loss_crossing`, `geometric_step`,
`expand_until_bracketed`, `tighten`, `rescue_probe`, the N3 one-shot stop,
and the finalist structure-rebuild re-probe that fires when a reused-structure
finalist sits more than 1.8x from the anchor) with the Balanced budget
(5 pixel probes, reserve 0.03, band 1.0, 2 structural builds). Trace rows are
matched to fixtures by the full source-feature vector (three fixtures with
identical features are separated by which curve reproduces the deployed
scores). Three trace rows (16x16, 33x20, 8x8) have no oracle curve and are
excluded, leaving 91 cells.

Scores at rungs the deployed run actually probed use the real observation;
scores anywhere else are read off the oracle curve, piecewise-linear in
(ln scale, ln loss). Bytes at a counterfactual final rung are interpolated in
(ln scale, ln bytes) over the cell's priced points (the label neighbourhood
plus the deployed exact prices).

The corrected-stop policy, per cell:

1. probe 1 exactly as deployed; if it is feasible and within the band
   (score in [85, 86]) and the variant trusts in-band first probes, stop;
2. otherwise aim one correction at the crossing with the effort's reserve
   (aim score 85.45, as `tighten` does) from the measured loss and a slope
   `beta`: `scale2 = scale1 * (loss1 / loss_aim)^(1/beta)`, clamped to one
   bounded jump; if the landing is feasible and in band, stop;
3. optionally one second correction from the slope through the two measured
   points (clamped 0.2-3.0), same acceptance;
4. otherwise hand the probes to the unchanged navigator
   (expand, tighten, rescue), then apply the finalist rebuild rule.

Variants: slope from the fixed prior 0.9 (`prior`), the deployed model's
`local_loss_exponent` (`model`), or the label's local slope at the crossing
(`oracle`, an upper bound no runtime can reach); an optional move gate that
sends the cell straight to the navigator when the predicted correction
exceeds 0.25 / 0.35 / 0.5 in ln scale; trusting in-band first probes without
the confidence gate; the second correction on or off.

Commands:

```sh
cd .agent/scratch/two-probe-replay-20260902
python3 replay_two_probe.py          # writes replay-report.json, replay-cells.jsonl
```

| Item | SHA-256 |
| --- | --- |
| deployed trace corpus `corpus-shadow.jsonl` | `f8bc20bad8659be1c6361441491323fbba0edd2d32d0b634b94a9bf66077b119` |
| 637-row label set `all-knot-labels.jsonl` | `8092d6ecd196c377482f5d52b15cbde6107404c93b86d9e9838729dfd05c950a` |
| corpus manifest `test-set/quality-corpus.json` | `ec986536c986faa16a0df7d2bf8d4aef4e6bb61a8ec3e017a6a52938f7490681` |
| `replay_two_probe.py` | `9dd3eec9e2754201b427f3e3ff5e8dce7f47cebe953ff3333f74cfacb8721d81` |
| `replay-report.json` | `f4aeb0895b4eb374c0743541f9f5415824c530a4af0fdf8a3c44ce5be3f8f8bc` |
| `replay-cells.jsonl` | `bb44425fa8d954d409b23c064626452b6c70397e6bd8596e274ed9ee637ce47b` |

Git revision `2b87b7bd07aca849a489a6bf0707362a9971532f` (worktree dirty from
the registered advisor source and unrelated user artifacts). Host Linux
6.17.0-41-generic, x86_64.

## Raw results

### Port fidelity

Replaying the deployed policy (real first probe, then real scores where the
deployed run probed the rung, the curve elsewhere) reproduces the deployed
probe sequence rung-for-rung on 85/91 cells and the probe count on 89/91;
replay mean 3.484 probes against the deployed 3.462 (histogram 4/12/40/6/29
against 4/12/40/8/27). Every mismatch is on a saturated-class cell or an
oracle-curve interpolation miss.

Oracle-curve fidelity on the 315 deployed probes (|real - curve| in score
points): median 0.20, p90 2.24, p99 10.1, max 25.4. The tail is the
saturated class (p90 6.92 points; every one of the twelve worst probes); the
per-class p90 is at most 1.28 elsewhere. Counterfactual scores inside the
saturated class are therefore not trustworthy at band resolution.

Deployed statuses on the 94 traced cells: `met` 68, `met_work_cap` 15,
`rescued_fresh_structure` 11; 15 cells carry a repeated final rung, which is
the finalist rebuild re-probe. The three photo-large fixtures that need five
probes are OOD (`log2_pixels`) and start at the legacy predictor's rung 73727
(ln ratio +1.14..+1.40 above the oracle crossing; the flagged model median
would have been +0.51..+0.64), then pay two expansion probes, one tightening
probe and the rebuild.

### Policies, all 91 cells

| Policy (slope, move gate, trust in-band first probe, second correction) | mean probes | <=2 probes | histogram 1/2/3/4/5 | stops 1/2/3 | no feasible final | byte geo-mean vs deployed | max | cells > +2% | cells < -2% |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| deployed (trace) | 3.462 | 17.6% | 4/12/40/8/27 | 4/-/- | 0 | 1.0000 | - | - | - |
| prior, no gate, trust=0, c2=0 | 3.187 | 48.4% | 0/44/10/13/24 | 0/48/0 | 0 | 1.0039 | 1.299 | 24 | 11 |
| prior, no gate, trust=1, c2=0 | 3.055 | 48.4% | 12/32/10/13/24 | 12/36/0 | 0 | 1.0048 | 1.299 | 24 | 10 |
| prior, no gate, trust=1, c2=1 | 2.846 | 48.4% | 12/32/21/10/16 | 12/36/25 | 0 | 0.9951 | 1.299 | 23 | 11 |
| prior, gate 0.25, trust=1, c2=1 | 3.132 | 40.7% | 12/25/21/5/28 | 12/24/10 | 0 | 1.0049 | 1.204 | 12 | 4 |
| prior, gate 0.35, trust=1, c2=1 | 3.110 | 42.9% | 12/27/19/5/28 | 12/27/17 | 0 | 1.0058 | 1.204 | 15 | 5 |
| prior, gate 0.5, trust=1, c2=1 | 3.011 | 47.3% | 12/31/17/6/25 | 12/31/18 | 0 | 1.0007 | 1.204 | 16 | 8 |
| model, no gate, trust=1, c2=1 | 3.022 | 42.9% | 12/27/20/11/21 | 12/29/24 | 0 | 1.0125 | 4.037 | 19 | 10 |
| oracle, no gate, trust=1, c2=1 | 3.066 | 47.3% | 12/31/14/7/27 | 12/33/17 | 0 | 1.0129 | 4.111 | 18 | 7 |

The 4.0x byte cells under `model` and `oracle` slopes are plateau accepts: a
second correction landed in band on a gradient whose score is flat across a
wide scale range, so "in band" no longer implies "near the coarsest feasible
rung". The +30% cell under the prior slope (gradient-luma-shallow-v) is the
opposite failure: the correction from a 96.75-point first probe overshot far
too coarse, the navigator then spent its remaining budget expanding back and
ended at the first probe under `met_work_cap`.

### Fair overshoot against the oracle crossing

Final effective scale divided by the label's crossing scale at 85 (geometric
mean / p90 / cells whose final score exceeds 86):

| Path | geo-mean | p90 | out of band |
| --- | ---: | ---: | ---: |
| deployed | 1.150 | 1.159 | 12/91 |
| prior, no gate, trust=1, c2=1 | 1.168 | 1.208 | 8/91 |
| prior, no gate, trust=1, c2=0 | 1.167 | 1.289 | 7/91 |
| prior, gate 0.25, trust=1, c2=1 | 1.143 | 1.159 | 13/91 |
| prior, gate 0.5, trust=1, c2=1 | 1.145 | 1.176 | 10/91 |

### Per class, policy prior / no gate / trust=1 / c2=1

| Class | cells | deployed mean | replay mean | stop after 1 / 2 / 3 | fallback | byte geo-mean | max | oracle-curve p90 abs error |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| gradient | 25 | 3.56 | 2.76 | 2 / 12 / 8 | 3 | 1.006 | 1.299 | 1.28 |
| grayscale | 3 | 2.33 | 1.67 | 1 / 2 / 0 | 0 | 1.010 | 1.040 | 0.10 |
| line-art | 6 | 3.00 | 2.50 | 0 / 3 / 3 | 0 | 1.017 | 1.065 | 0.54 |
| noise-lowlight | 4 | 3.00 | 2.00 | 0 / 4 / 0 | 0 | 0.988 | 0.994 | 0.13 |
| photo | 11 | 3.36 | 2.73 | 2 / 4 / 3 | 2 | 1.002 | 1.031 | 0.24 |
| photo-scene | 7 | 2.57 | 2.00 | 2 / 3 / 2 | 0 | 1.002 | 1.018 | 0.21 |
| saturated | 26 | 4.12 | 3.73 | 2 / 5 / 7 | 12 | 0.972 | 1.219 | 6.92 |
| text-screenshot | 7 | 2.43 | 1.71 | 3 / 3 / 1 | 0 | 1.003 | 1.015 | 0.12 |
| tiny | 2 | 5.00 | 4.50 | 0 / 0 / 1 | 1 | 1.001 | 1.011 | 1.27 |

On the 63 cells outside the saturated and tiny classes (deployed mean 3.143):
prior / no gate / trust=1 / c2=1 gives mean 2.429 (60.3% at most two probes;
byte geo-mean 1.005, max 1.299, 14 cells above +2%); prior / gate 0.5 /
trust=1 / c2=1 gives 2.587 (58.7%; 0.998, max 1.093, 10 cells above +2%);
prior / gate 0.25 gives 2.651 (54.0%; 1.004, max 1.093, 8 cells above +2%).

### What decides a two-probe finish

For the 79 cells whose first probe is not already in band, the distance of
the first probe from the crossing at the aim score (|ln scale1 / ln
scale_aim|) predicts the corrected stop almost completely (prior slope, no
gate):

| first-probe distance (ln scale) | corrected stop lands in band |
| --- | ---: |
| [0, 0.25) | 23 / 29 |
| [0.25, 0.5) | 7 / 23 |
| [0.5, 0.75) | 3 / 7 |
| [0.75, 1.0) | 3 / 6 |
| [1.0, 1.25) | 0 / 4 |
| [1.25, 1.5+) | 0 / 10 |

Distance distribution on those 79 cells: median 0.32, p75 0.75, p90 1.38;
36.7% are within 0.25 and 65.8% within 0.5. The oracle slope does not change
this (33 stops against 36 with the prior): the curve between a distant first
probe and the crossing is not a single power law, so the slope estimate is
not the limiting error once the move exceeds about 0.3.

By class (median first-probe distance, then ideal / model / oracle slope
medians): gradient +0.11 (0.85 / 1.17 / 1.22), photo +0.30 (0.65 / 0.73 /
0.71), photo-scene -0.10 (0.71 / 0.77 / 0.77), text-screenshot -0.08 (0.62 /
0.73 / 0.68), noise-lowlight -0.18 (0.88 / 0.68 / 0.88), line-art +0.10
(0.83 / 1.44 / 1.61), saturated +0.49 (0.66 / 1.14 / 2.21), tiny -1.19.

### Probe-wall proxy

Sum of the trace's per-probe plan+render+metric milliseconds with each
counterfactual probe priced at the cell's own mean reused-probe cost (first
probe and rebuild at the fresh cost); all 91 cells, and the 11 cells at or
above 4 MP which carry 87% of the total:

| Policy | all cells | >= 4 MP cells |
| --- | ---: | ---: |
| prior, no gate, trust=1, c2=1 | 0.884 | 0.895 |
| prior, no gate, trust=1, c2=0 | 0.959 | 0.974 |
| prior, gate 0.25 / 0.35 / 0.5, trust=1, c2=1 | 0.937-0.940 | 0.952 |

## Conclusion

**Fail against the advisor's target; partial as a Contract-B candidate.** No
variant approaches mean 1.5 or two probes on 90% of cells: the best
ungated variant reaches 2.85 on the corpus and 2.43 on the 63 in-domain
cells, with 48% (60% in-domain) at two probes or fewer. The floor gate holds
in every variant (no cell ends without a feasible final rung; minimum final
margin 0.01 points, within the oracle-curve error, so the floor claim rests
on the deployed first probe and the real scores, not on the interpolation).
The byte gate holds in geometric mean only (0.995-1.006); the ungated
variants carry a fallback tail up to +30% and a slightly worse fair overshoot
(1.168 against the deployed 1.150), while the 0.25-0.5 move gates hold the
tail to +9% in-domain and give a fair overshoot equal to or better than the
deployed path, at 2.6-2.7 in-domain probes.

The limiting quantity is the first probe's distance from the crossing, not
the slope: a corrected stop lands in band 79% of the time inside 0.25 ln
scale and essentially never beyond 1.0, and only 37% of non-in-band first
probes are inside 0.25. Reaching the advisor's target would need a first-probe
predictor with p90 log-scale error near 0.25 on this corpus; the 2026-09-02
retrain measured a blind p90 of 1.854 and median 0.467, and the deployed
starts have median 0.32 / p90 1.38 here. That target is out of reach of any
navigator-side change on this corpus, whose 28 saturated and tiny cells are
both the worst-predicted and the least simulable.

What this does not establish: real-encoder behaviour of the corrected path
(the counterfactual scores are interpolated, unreliable inside the saturated
class), wall time (the proxy sums unpinned per-probe millis), and the
finalist byte ordering at the counterfactual rungs. The replay's ceiling for
the corrected stop is roughly 12% of probe wall, about 7-8% of end-to-end
Balanced wall at 12 MP given the 68% render+metric share, with the ungated
variant; the gated variants give about 5-6% with the byte tail controlled.

## Consequences

Nothing changed in production code. Two candidate Contract-B decisions are
handed to the operator with these numbers: (1) the corrected stop with a
0.5 move gate, trusted in-band first probes and one second correction, as a
feature-gated trial run through `one_shot_promotion_ab.py` on the corpus
(expected about -0.45 probes per encode, byte-neutral in geometric mean, tail
to be bounded by the standing per-cell gate); (2) replacing the legacy
ladder-ceiling start for `log2_pixels`-flagged large photos with the flagged
model median, which by this trace would remove two expansion probes on each
of the three 12 MP photo cells. The advisor's 1.5-probe target is recorded as
predictor-bound and not pursued through the navigator.
