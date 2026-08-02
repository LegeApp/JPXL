# PERFORMANCE

**Current reproducible baselines only.** Not a history of attempts, not a
record of ideas tried. Superseded numbers are deleted, not archived — git has
them, and a completed measurement whose conclusion is interesting belongs in
`docs/experiments/` as a frozen report.

Optimization is out of scope for the current phase (`PLAN.md`: scalar,
single-threaded reference pair first). This file exists so that when the time
comes, the first number recorded is already trustworthy.

## Rules for any number that appears here

A row without all of these is not a baseline and does not go in this file:

1. **Immutable binaries.** Hash every binary involved; recheck the hash before
   every timed run and reject the run if it changed.
2. **Pinned inputs.** Exact image, its hash, dimensions, bit depth, and color
   encoding.
3. **Interleaved A/B**, never separate batches. Report minimum, median, and
   dispersion — not a single mean.
4. **Correctness gate alongside timing.** Record output bytes and hash, and the
   decoded-pixel result. A fast wrong answer is not a datum.
5. **Host state.** Thread count, CPU affinity and policy, load, free memory,
   swap and tmpfs use, competing jobs. Measurements taken under resource
   pressure are diagnostics, never baselines.
6. **Cache regime stated.** Cold-cache, warm-cache, and warm-process results
   are three different claims; never merge them.
7. **Size classes.** At minimum a small (~4 MP) and a large (≥12 MP) image.
   Allocation pathologies that scale with image size are invisible at 4 MP.
8. **Unknown stays unknown.** If profiling explains 48% of samples, say 48% and
   label the rest `UNKNOWN`. Do not invent a mapping.

Profiling notes: on hybrid Intel hosts, generic `cycles` splits into
`cpu_core` and `cpu_atom` and will silently produce a near-empty flamegraph.
Use an explicit userspace event (`cpu_core/cycles/u`) and verify that the
folded period total equals the `perf` event total before trusting the SVG.

Last reviewed: 2026-08-02.

## Baselines

| Workload | Configuration | Median | Dispersion | Peak RSS | Provenance |
| --- | --- | --- | --- | --- | --- |
| — | — | — | — | — | No baselines. No measurable code yet. |
