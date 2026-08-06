# PERFORMANCE

**Not the plan-of-record.** Durable performance claims live in the AKR ledger;
full raw run logs and flamegraph artifacts live under `.agent/scratch/` (not
pruned by `akr build`). Do not re-accumulate baseline tables or rules here.

## Authority

| Kind | Where |
| --- | --- |
| Baseline promotion rules | `@jpegxl-rs.policy.performance-baseline-rules` |
| `jpxl bench` entry points | `@jpegxl-rs.policy.jpxl-bench-entry-points` |
| Docs retirement decision | `@jpegxl-rs.decision.retire-performance-md` |
| Pre-opt encode ladder (summary) | `@jpegxl-rs.observation.preopt-encode-baseline-2026-08-06` |
| Pre-opt flamegraphs (summary) | `@jpegxl-rs.observation.preopt-flamegraphs-2026-08-06` |

## Scratch (full logs / artifacts)

- Ladder raw log: `.agent/scratch/preopt-baseline-2026-08-06.md` (and `.log`)
- Flamegraphs: `.agent/scratch/flamegraphs/preopt-2026-08-06/`  
  (`*.v2.folded`, `*.v2.svg`, `HOTSPOTS.md`, `PROVENANCE.md`)

Frozen experiment writeups, when needed: `JPXL/docs/experiments/`.
