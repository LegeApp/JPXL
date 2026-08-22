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
