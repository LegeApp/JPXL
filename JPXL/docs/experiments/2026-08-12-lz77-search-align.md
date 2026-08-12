# Phase 4D: Exact-tier LZ77 alignment + residual lookback measurement

Date: 2026-08-12
Status: complete. **Promoted:** Exact-tier `allow_lz77=true` and 3-symbol
hash-chain match finding at lookback 256. **Honest negative:** expanding
lookback to `max(256, dist_multiplier)` (one residual row) for vertical matches.

## 1. Problem

1. **Search/emission mismatch.** `plan_for` Exact finalists built
   `ModularSource` with `allow_lz77=false` while emission always used `true`.
   Ranking could over-price LZ77-friendly modes (documented in Phase 4A palette
   fixture notes). Cheap-tier stays Shannon-only (no LZ77).
2. **Lookback.** `greedy_lz77_events` used a hard 256-symbol reverse linear
   scan. Vertical residual matches at distance = channel width need a wider
   window on channels >256 wide.

## 2. What was built

- Exact finalists (MA / palette / squeeze) use `allow_lz77=true`.
- Match finder: 3-symbol hash chain with the same greedy longest-match and
  nearest-distance tie rule as the pre-4D reverse linear scan (equivalence
  tested).
- Lookback remains 256 after measurement (below). Explicit lookback parameter
  kept for tests and a future revisit.

## 3. Results

| Fixture | Result |
| --- | --- |
| Photo 0.8 / 4 / 12 MP effort 1 | **Byte-identical** to pre-4D (1,075,465 / 5,157,974 / 9,795,599 B; same sha256) |
| Photo 0.8 MP effort 7 | Byte-identical (1,037,868 B) |
| `exact_finalist_pricing_sees_residual_lz77` | LZ77-aware Exact cost strictly < plain on repeating residual pattern |
| Hash-chain vs linear (same lookback) | Identical event streams on synthetic patterns |
| Lookback `max(256, row)` on 12 MP | Size-identical, ~10× slower linear (135 s vs 14 s); hash-chain ~17 s still worse than 256 |
| jpxl-decode + djxl | Exact pixels (rmse=0) |

## 4. Decision

Promote Exact LZ77 alignment and hash-chain finder at lookback 256. Do **not**
expand production lookback to one residual row until a corpus stratum shows a
size win that pays for the wall cost. Phase 4 modular sub-phases 4.0–4D are
closed on the density-first track; remaining encoder work is outside this
parent (e.g. Phase 6.4).

Raw notes: `.agent/scratch/phase4d-lz77-2026-08-12/`.
