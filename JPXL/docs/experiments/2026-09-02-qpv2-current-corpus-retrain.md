# QPv2 current-corpus all-knot retrain

## Question

Does retraining the QPv2 one-shot crossing predictor on the current
quality-guard corpus make it suitable for a Contract-B production trial?

## Preregistered gate

The predictor tool's existing gates apply before this run: quick-screen needs
p90 absolute log-scale error at most 0.45, first-plan success at least 0.80,
and simulated byte geometric mean at most 1.015. Production additionally
requires median/p90/p99 absolute log-scale error at most 0.10/0.30/0.70.

The active probe-campaign plan also requires a meaningful common-case work
reduction; a candidate that routes most requests to the exact controller is
not promotable even if it preserves the quality floor.

## Method

This is an offline JPXL-controller simulation, not a libjxl experiment and
not a timing measurement. The q85 pilot first established that the current
corpus needed fresh labels. The recorded all-knot run used the same release
binary, balanced effort, four encoder threads, and every manifest split:

```sh
taskset -c 0,2,4,6 python3 JPXL/tools/quality_oracle_labels.py sweep \
  --manifest test-set/quality-corpus.json --jpxl JPXL/target/release/jpxl \
  --out-dir .agent/scratch/qpv2-retrain-20260902/all-knot-sweeps \
  --splits calibration development holdout --targets 30 50 70 80 85 90 95 \
  --threads 4 --time-budget-minutes 60
python3 JPXL/tools/quality_oracle_labels.py labels \
  --manifest test-set/quality-corpus.json \
  --sweep-dir .agent/scratch/qpv2-retrain-20260902/all-knot-sweeps \
  --output .agent/scratch/qpv2-retrain-20260902/all-knot-labels.jsonl
python3 JPXL/tools/quality_predictor_v2.py train \
  --labels .agent/scratch/qpv2-retrain-20260902/all-knot-labels.jsonl \
  --repo-root . --feature-sets source source+transform \
  --transform-features .agent/scratch/qpv2-retrain-20260902/transform-features.json \
  --sweep-dir .agent/scratch/qpv2-retrain-20260902/all-knot-sweeps \
  --blind-split holdout --emit-feature-set source+transform \
  --rust-out .agent/scratch/qpv2-retrain-20260902/qpv2-all-knots-generated.rs \
  --report-out .agent/scratch/qpv2-retrain-20260902/qpv2-all-knots-report.json
```

The transform map was independently validated against lexical manifest-path
order, PPM dimensions, and each trace record's complete source-feature vector.
The model trained only calibration and development rows; the holdout was never
used for ridge selection or emitted-model fitting. Frames below the preexisting
128-pixel model-domain floor route to the exact controller and were excluded:
two of 91 labeled fixtures.

| Item | SHA-256 |
| --- | --- |
| JPXL `jpxl` release binary | `636f75beac6c42d413a406bd0e6df4b3604f475baee31d3a5ec24b8e52d61340` |
| corpus manifest | `ec986536c986faa16a0df7d2bf8d4aef4e6bb61a8ec3e017a6a52938f7490681` |
| 637-row label set | `8092d6ecd196c377482f5d52b15cbde6107404c93b86d9e9838729dfd05c950a` |
| transform map | `2fca59248c52c90013b29ba559a5677e60abe3270fda7705803d60274b9332ab` |
| report | `4d77e6472fae10aee957e7bc9772f54e6f834ea5e3115a5c713d8a17dc98d569` |

Host: Linux 6.17.0-41-generic, x86_64. Git revision:
`d85bad6d5d66c7587593fd3ace910d312aee09fe` (worktree already dirty from
unrelated user artifacts). No elapsed time is reported or interpreted.

## Raw results

The sweep yielded 91 complete, uncensored labels at each of seven requested
scores (637 rows). Training used 70 in-domain fixtures (490 rows); the blind
holdout had 19 in-domain fixtures (133 rows).

| Feature set | Evaluation | median / p90 / p99 abs log-scale error | first plan | fallback | byte geo-mean | expected reconstructions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| source | grouped CV | 0.340 / 1.489 / 4.679 | 88.4% | 96.7% | 1.0145 | 3.931 |
| source | blind holdout | 0.978 / 2.386 / 3.124 | 88.7% | 98.5% | 1.0098 | 3.970 |
| source + transform | grouped CV | 0.271 / 1.085 / 4.503 | 85.1% | 93.1% | 1.0054 | 3.859 |
| source + transform | blind holdout | 0.467 / 1.854 / 2.966 | 90.2% | 85.0% | 1.0102 | 3.669 |

The source-plus-transform fit selected ridge 3.0. On the blind holdout, its
routes were 85.0% preflight exact fallback, 9.8% tightened, 3.0% one-shot,
and 2.3% tightening that retained the verified first plan. Its first-or-one-
correction feasibility was 100.0%; this says the fallback guard protects the
quality floor in the simulator, not that the model is accurate enough to ship.

## Conclusion

**Fail: do not promote QPv2 from this retrain.** Transform summaries are
useful: compared with source-only, blind median error drops from 0.978 to
0.467 and fallback drops from 98.5% to 85.0%. But the best candidate misses
the quick-screen p90 gate by over four times (1.854 versus 0.45) and every
production error gate. It also does not approach the intended common-case
work reduction: 3.669 expected reconstructions is near the exact-controller
charge of four.

This does not show that a different model family, content router, confidence
calibration, or probe policy cannot work. It establishes only that the current
transparent pooled linear QPv2 fit, trained on these labels, is not a
Contract-B candidate.

## Consequences

Nothing changed in production code or generated predictor tables. Keep the
current exact fallback and use the failed result to direct the next experiment
toward class-aware routing/confidence calibration before any production policy
change.
