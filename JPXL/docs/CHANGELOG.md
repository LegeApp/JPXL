# CHANGELOG

Released, user-visible changes only. One line per change, newest release first.

**Not** for: work in progress (AKR and `docs/generated/ACTIVE-WORK.md`),
experiments and hypotheses (`docs/experiments/`), benchmark reports (AKR
observations and `.agent/scratch/`), clause coverage (`CONFORMANCE.md`), or
internal refactors with no observable effect. The previous project's changelog
reached 6,965 lines by ignoring this rule and became unreadable. Git preserves
history; this file preserves signal.

Format: [Keep a Changelog](https://keepachangelog.com/), semantic versioning.

## Unreleased

- 2026-08-25 — Large-frame `--quality` probes are faster: the varblock
  reconstruction and the transfer-curve linearization — previously serial
  on every probe — now run banded across the worker pool with byte-identical
  output at any thread count. On the locked 12 MP anchor at 8 threads the
  quality-mode wall drops 21 % (release), with peak memory and every emitted
  stream unchanged.
- 2026-08-25 — The one-shot crossing predictor is now the default `--quality`
  controller seed: a generated transparent model (`qpv2-st-1`, source +
  DCT8-summary features, trained on 130 image families) picks the first
  fresh plan — its risk-adjusted candidate when confident, its median when
  uncertain — and the bounded navigator continues under the same probe/price
  caps with canonical verification before every emission, so the hard score
  floor is unchanged. Promotion A/B on 441 never-tuned holdout cells: zero
  floor violations on either arm, byte geomean 0.997 (locked 13-image
  holdout) and 0.990 (50 painting families), reconstructions −18% and wall
  −16% on the paintings, 12 MP anchor wall −21%. Known bounded regressions:
  sub-kilobyte saturated fixtures (worst +131 bytes) and target-95 cells
  (≤1.12× with higher achieved scores). Build `jpxl-encode-policy` without
  default features for the legacy table-seeded controller.
- 2026-08-24 — Quality trace schema is now `jpxl.quality-trace/2`: adds a
  whole-search `work` block (pixel plans, reconstructions, metric
  evaluations, entropy trainings, emissions — policy trials and the reducer
  included), a shadow `prediction` block (null until a generated crossing
  model is present), and a `transform_features` block (null unless the
  research budget asked for it). `/1` traces stay readable by the harness.
  New calibration tooling: `jpxl quality-ladder` (fresh production-policy
  pixel plans at pinned effective scales, canonically scored, optionally
  exact-priced, as JSONL), `jpxl features --transform-summary` (DCT8-derived
  transform features), and `tools/quality_oracle_labels.py` /
  `tools/quality_predictor_v2.py` (oracle-label sweeps over the full
  effective ladder and the trained crossing predictor). The quality-corpus
  manifest gains `family_id`/`variant_id`/`generator_family`/
  `source_capture_id`, and the fixture generator fails if any image family
  crosses a split.
- 2026-08-24 — The `--quality` score is now a hard floor at the public
  surface: an encode whose bounded search cannot verify the requested score
  fails (`quality target not met` on stderr, exit 1, no output file) instead
  of writing an under-target stream with exit 0. `--quality-fallback
  lossless` emits a mathematically lossless stream instead
  (`status=fallback_lossless`); `--quality-fallback best-effort` emits the
  finest verified under-target stream with its true `saturated_top` /
  `under_target_work_cap` status (the previous behavior, now explicit). The
  `jpxl` facade gained `Error::TargetNotMet(QualityMiss)`, `QualityFallback`,
  and `Encoder::with_quality_fallback`; a refused encode still appends its
  `JPXL_QUALITY_TRACE` record, and every met-path stream is byte-identical.
- 2026-08-22 — Lossy encoding now leads with quality: `jpxl encode --quality
  [N]` (alias `--ssimulacra2`, `--lossy`) sets a minimum SSIMULACRA2 score
  (0..100, 100 = lossless); `--bpp`, `--target-bytes`, and the new
  `--global-scale` are expert modes, and the four are mutually exclusive.
  `--effort` now also takes `fast`/`balanced` for the lossy effort. A score
  below 100 runs the perceptual quality controller (see the next entry); 100
  routes to the lossless path. Rate-mode output is unchanged (byte-identical).
- 2026-08-22 — Perceptual quality controller: `--quality N` emits the smallest
  stream whose reconstructed pixels score at least N on the in-tree
  SSIMULACRA2 (`jpxl-perceptual`, clean-room, deterministic across worker
  counts), scored from the plan without decoding (`jpxl-plan-render`). Fast
  spends at most 3 scored probes and 2 exact prices, Balanced 5 and 3; the
  printed `quality_target=… achieved=… status=…` line and the optional
  `JPXL_QUALITY_TRACE=<path>` JSONL trace report what was verified. Below-
  target output is never silent: `saturated_top` / `under_target_work_cap`
  name it.
- 2026-08-22 — Behind the `quality-effort` build feature, the Quality effort's
  perceptual search runs a bounded policy bank (chroma QM, `quant_lf`, EPF,
  CfL, truncation lambda; coordinate descent sharing one candidate context and
  the baseline cover/CfL) and a terminal coefficient reducer that spends the
  score reserve on bytes (finalist-priced last-nonzero removals, each batch
  re-verified by the canonical score). Both are measured and kept off for
  Fast and Balanced: bytes fall 0.4–1.8 % but wall rises past the +25 %
  budget (AKR evidence `pqc-pr5b-structure-reuse`, `pqc-pr7-dev-split`).
  Quality-mode chroma QM is now a per-preset constant, never a per-bitrate
  branch; Fast and Balanced rate-mode streams are byte-identical.
- 2026-08-22 — The `jpxl` facade gained `with_ssimulacra2_score`,
  `with_global_scale`, `with_effort(Effort)`, `with_lossless_effort`, and
  `encode_rgb8_reported` / `encode_rgb16_reported` returning an `EncodeReport`;
  `Preset`/`with_preset` are replaced by `Effort`/`with_effort`.
- 2026-08-20 — Added the dependency-light `jpxl` facade with safe-default
  decoding and builder-based lossless or target-rate encoding from interleaved
  8/16-bit RGB and greyscale buffers.
- 2026-08-20 — The `jpxl` CLI now reads and writes PNG, JPEG, WebP, TIFF, BMP,
  GIF, ICO, TGA, QOI, PGM, and PPM; supports stdin/stdout, 16-bit-preserving
  outputs, alpha-preserving decode, and explicit transparency flattening.
- 2026-08-02 — Repository scaffolded. Nothing released yet.
