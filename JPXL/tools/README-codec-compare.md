# codec_compare.py — perceptual quality measurement

`codec_compare.py` builds reproducible JPXL / libjxl rate–distortion curves and
times frozen settings. It is Python 3 standard-library only. Raw JSONL is
authoritative; summaries, TSV and Markdown are all derived from it.

This document covers the perceptual **quality axis** (PR 4). The distance/bpp
curve, `time`, `timing-report` and `risk-report` flows are unchanged.

## Record schema

Curve and timing rows now carry `jpxl.codec-comparison/3`. Distance/bpp curve
rows and timing rows keep the exact shape they had under `/2`; the bump only
signals that a jpxl curve row *may* carry quality fields. `load_jsonl` accepts
both `/2` and `/3`, so old captures keep loading — the change is backward
compatible.

A jpxl **quality** curve row adds:

- `setting.kind = "quality"`, `setting.value` = the requested SSIMULACRA2 target,
  `setting.effort` ∈ {`fast`,`balanced`}
- `requested_score`, `achieved_score`, `quality_status`, `quality_metric`,
  `reported_bytes`, `probes`, `prices`
- `trace_path` and `wall_by_phase{analysis,plan,render,metric,entropy,emit}` when
  a trace was captured (see below)

## The encoder contract

`jpxl encode --quality Q [--effort fast|balanced] --threads N in.ppm out.jxl`
selects the perceptual VarDCT path with a minimum SSIMULACRA2 score `Q`
(0..100; 100 = lossless) and, on success, prints exactly one line:

```
quality_target=85.0000 achieved=85.1372 bytes=412883 metric=ssimulacra2-jpxl-1 effort=balanced probes=3 prices=2 status=met
```

`status` is one of: `met`, `met_adjacent_rungs`, `met_work_cap`,
`saturated_floor`, `saturated_top`, `under_target_work_cap`,
`rescued_fresh_structure`, `routed_to_lossless`, `fallback_lossless`,
`unsupported_too_small`.

The score is a hard floor by default: an encode whose bounded search cannot
verify `Q` exits 1 with `quality target not met` on stderr and writes no
output file. The harness therefore passes `--quality-fallback best-effort`,
which emits the finest verified under-target stream with its true
`saturated_top` / `under_target_work_cap` status so a curve point is always
measurable (`--quality-fallback lossless` instead emits a lossless stream as
`fallback_lossless`).

When `JPXL_QUALITY_TRACE=<path>` is set, a `jpxl.quality-trace/2` JSONL file is
written (also for a refused encode — the failed search is still calibration
input). The harness sets this per curve point (under the work dir) unless
`--no-quality-trace` is given, and merges `wall_by_phase` into the record.

## Subcommands

### `curve` — quality axis

`--quality` is mutually exclusive with `--bpp`. In quality mode the jpxl curve
follows the score axis while cjxl points stay on the distance axis.

```
codec_compare.py curve --manifest corpus.json --output curve.jsonl \
  --work-dir work --quality 30 50 70 80 85 90 95 --quality-effort balanced \
  --distance 0.5,1.0,2.0,4.0 \
  --jpxl ./jpxl --cjxl ./cjxl --djxl ./djxl
```

- `--quality S ...` — one or more SSIMULACRA2 targets in [0, 100].
- `--quality-effort {fast,balanced}` (default `balanced`) — the jpxl perceptual
  effort. (`--effort` remains the integer libjxl `-e` effort for cjxl.)
- `--quality-trace` / `--no-quality-trace` (default on) — capture and merge the
  per-point trace.

### `summarize` — quality analysis

```
codec_compare.py summarize --input curve.jsonl --output summary.json \
  --timing-plan plan.json --score-guard 0.30 --tsv quality.tsv
```

Adds a `quality` block to the summary JSON when quality rows are present. Per
image × target: `floor_violation` (achieved < requested − guard), `overshoot`
(achieved − requested), matched-**achieved**-score byte ratio JPXL/cjxl
(cjxl curve interpolated at JPXL's achieved score), status, probes, prices.
Per image: BD-rate over SSIMULACRA2 (standard Bjøntegaard — cubic fit of
log-bytes vs score, integrated over the overlapping score range), geomean of
matched-score byte ratios, and a monotonicity check (achieved non-decreasing in
requested). Aggregate: floor-violation count, median |achieved − requested|,
probe/price distributions, and geomean byte ratio vs cjxl. `--tsv` exports the
per-target rows. The bpp `rows`/timing-plan output is unchanged.

### `quality-report` — Markdown

```
codec_compare.py quality-report --input summary.json --output report.md
```

Renders a table per image × target (requested, achieved, bytes, bpp, status,
probes, prices, matched cjxl bytes, ratio) plus per-image BD-rate / geomean /
monotonicity and an aggregate section.

### `metric-variation` — score guard derivation

```
codec_compare.py metric-variation --pairs pairs.json \
  --binaries ./jpxl-scalar ./jpxl-avx2 --repeats 3 --output metric-variation.json
```

`pairs.json` is `{"schema":"jpxl.metric-variation-input/1","pairs":[{"id":..,
"reference":ref.ppm,"candidate":cand.ppm}, ...]}`. Runs `jpxl compare` for every
pair under each build, repeated `--repeats` times (default 3), and reports the
max |Δscore| per pair and overall. The derived guard is

```
guard = ceil_1e-2(2 × max|Δ|)
```

Feed that value to `summarize --score-guard`. Pass absolute binary paths — a
relative `./jpxl` normalises to `jpxl` and will not be found.
