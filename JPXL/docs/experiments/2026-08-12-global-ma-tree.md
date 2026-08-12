# Phase 4C: global MA tree for multi-section modular

Date: 2026-08-12
Status: complete. **Promoted** for multi-section emission when it does not
grow the section payload versus per-section local trees.

## 1. Problem

`plan_for` already builds one frame MA tree, but emission re-wrote the tree
(and residual D bundle) in every modular section (`use_global_tree = false`,
G.1.3 global tree absent). On multi-group frames that multiplies tree/table
bytes by the section count.

## 2. What was built

- G.1.3 `have_global_tree = true` + MA tree + shared residual D bundle.
- Group modular headers: `use_global_tree = true`, residual tokens only
  (decoder `GLOBAL_TREE_SHARES_DISTRIBUTIONS`).
- Shared residual model: joint plain ANS, or joint residual LZ77 (per-section
  matches, shared Table C.1) when shorter.
- **Size gate:** multi-section encodes both global and local payloads and keeps
  the smaller (`global.total_len() <= local.total_len()`). Single-section stays
  local-tree (no dual encode).
- Search cost `total_cost_source` charges the MA tree once (not × groups).

## 3. Results

| Fixture | Result |
| --- | --- |
| 300×200 grey, group_dim 128 (6 groups) | Global 491 B vs local modular re-pay 3318 B; jpxl-decode exact |
| 1024×768 RGB effort 1 | Size gate keeps local; 1,075,465 B (= phase4-0b baseline) |
| Oracle suite with djxl | Green |
| Roundtrip suite | Green |

## 4. Decision

Default multi-section path **tries** global tree and keeps it only when it does
not grow the payload. Production never ships a larger multi-section stream
than the pre-4C local re-pay path. Remaining Phase 4 work is **4D** (LZ77
search/emission alignment and lookback).

Raw notes: `.agent/scratch/phase4c-global-tree-2026-08-12/`.
