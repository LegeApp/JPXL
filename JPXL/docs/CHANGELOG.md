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

- 2026-08-20 — Added the dependency-light `jpxl` facade with safe-default
  decoding and builder-based lossless or target-rate encoding from interleaved
  8/16-bit RGB and greyscale buffers.
- 2026-08-20 — The `jpxl` CLI now reads and writes PNG, JPEG, WebP, TIFF, BMP,
  GIF, ICO, TGA, QOI, PGM, and PPM; supports stdin/stdout, 16-bit-preserving
  outputs, alpha-preserving decode, and explicit transparency flattening.
- 2026-08-02 — Repository scaffolded. Nothing released yet.
