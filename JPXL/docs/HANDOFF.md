> **Plan of record: AKR.** Milestones, decisions, policies, constraints and the experiment
> findings now live in the AKR ledger (`.akr/`) and its generated views under
> `docs/generated/`. This file is retained as a working log / legacy reference — not the
> authoritative plan. See `AGENTS.md` and `docs/generated/ROADMAP.md`.
# HANDOFF

Dated working ledger. **Prepend** new entries — newest first. Each entry: what
changed, what is now proved, what is next, what is blocked.

## 2026-08-10 (6) — Phase 4J: the adaptive-quantization field is a net perceptual loss; and the undershoot cause, measured at last

**What changed (files: `crates/jpxl-encode-policy/src/{field.rs,request.rs,lib.rs}`, `crates/jpxl-cli/src/main.rs`):**
- `AqTuning { strength, clamp, chroma_weight }` on `SearchBudget`, `DesiredQuantField::from_atlas_tuned`, and `jpxl encode --aq-strength/--aq-clamp/--aq-chroma`. `AqTuning::default()` is exactly the historical constants — pinned by `the_default_tuning_reproduces_the_hardcoded_field`, which compares atom-for-atom in both AQ directions. 68/68 policy lib tests.
- `SearchBudget` loses its `Eq` derive (it now carries floats). `PartialEq` remains.

**The undershoot cause, finally measured** (entry 4 finding 3, entry 5's correction). `jpxl encode --bpp 4` on the 4 MP image:

> `undershot target by 191763 bytes (8.9%) — search ended short of target with budget spent, not at the ladder's limit [prices: 24 fast, 9 full]`

24 is *exactly* the Fast cap (`max_prices` 40 − `full_refinement_reserve` 16). The Fast ladder spent its entire allowance and stopped short. **The ladder was never the constraint** — my original diagnosis was wrong, and so was the trap I wrote against raising `max_prices`. Findings (2) speed and (3) undershoot are therefore the *same* problem: probe count.

**The finding: AQ is making quality worse.** Sweeping the field's constants at *fixed bitrate* (AQ moves where bits go, not how many, so matched size is the only honest comparison):

| 0.8 MP | 1.0 bpp ssim2 / btrgli | 2.0 bpp ssim2 / btrgli |
| --- | --- | --- |
| **AQ off** (strength 0) | 59.38 / **6.075** | **81.97** / **3.185** |
| shipped (0.25/1.0/0.35) | 59.62 / 6.636 | 80.01 / 3.562 |
| every other tuning tried | — / 6.58–6.78 | 79.6–80.0 / 3.45–3.83 |

Two things stand out. **AQ-off is an outlier, not the end of a trend** — every non-zero strength, clamp and chroma weight clusters tightly, while off sits well outside. So this is not a mistuned constant; having a non-neutral field *at all* costs perceptual quality here. And **the penalty does not shrink with rate** — at 2 bpp off wins on *both* metrics — which rules out the obvious explanation that the per-varblock `HfMul` signalling row is eating the coefficient budget.

**Why this was never caught.** M7's acceptance evidence reads *"spatial quality uniformity improved with stable target size"*. Uniformity is what `AqMode::Uniform` optimises. **Production ships `AqMode::Masking` — the opposite direction** — whose case is perceptual masking, which a uniformity criterion does not test. The shipped direction was accepted on a criterion the *other* direction satisfies, and no perceptual metric existed here to check it against. This is precisely the AGENTS.md §6 failure mode: green criteria that do not prove the thing they are taken to prove.

**Explicitly NOT done: the default is unchanged.** One image, two rates, one host is not enough to flip production behaviour. Before that: the other two corpus images, more rates, `AqMode::Uniform` (to separate "wrong direction" from "the machinery costs quality either way"), and non-photographic content, where masking is likeliest to earn its keep.

**Traps — do not "fix" these:**
- Do **not** flip `AqMode` to `Off` on this evidence. It is one image. The infrastructure to settle it now exists; use it.
- Do **not** re-tune `AQ_STRENGTH` and call it fixed. Every strength tried loses to off; the constant is not the problem.
- Do **not** cite M7's uniformity evidence as validation of the Masking default again. It validates the direction that is not shipped.

## 2026-08-10 (5) — Phase 4I: perceptual metrics (SSIMULACRA2 + butteraugli); the RD gap depends on which metric you ask; and a corrected undershoot diagnosis

**What changed (files: `Cargo.toml`, `crates/jpxl-conformance/{Cargo.toml,src/metrics.rs}`, `crates/jpxl-cli/{Cargo.toml,src/main.rs}`):**
- `metrics::ssimulacra2_score` (higher better, 100 = identical) and `metrics::butteraugli_distance` → `Butteraugli { distance, pnorm3 }` (**lower** better, 0 = identical; `distance` is the max-norm libjxl calls the butteraugli distance, i.e. what `cjxl -d` targets).
- Both behind **off-by-default** features (`ssimulacra2`, `butteraugli`, or `perceptual` for both), in `jpxl-conformance` only. `jpxl compare` reports them when built `--features perceptual`. No normative crate depends on either; a default `cargo build --workspace` fetches neither.
- `jpxl encode --bpp` now says when it undershoots its target, and distinguishes *ladder saturated* (no extra budget can help) from *budget spent short of target* (more search could).

**Dependency decision.** `ssimulacra2` 0.5 (BSD-2-Clause, → `yuvxyb`/`thiserror`/`num-traits`) and `butteraugli` 0.9 (BSD-3-Clause, pure Rust, → `imgref`/`rgb`/`archmage`). Both permissive. They stay opt-in because AGENTS.md §6 bans `thiserror` from this project's own error types, and that ban should not be dodged by pulling it in transitively on the default path.

**Clean-room note.** The `butteraugli` crate is a *port of libjxl's* butteraugli. Using it to **grade** output is the same category as running `cjxl`/`djxl` as oracles (§2), and its licence is permissive. **Wiring it into this encoder's own rate control would be a different decision** — it would make our perceptual model a derivative of libjxl's rather than one derived from the standard — and must be recorded as such before anyone does it.

**The finding: which metric you ask changes the answer.** Equal-quality bitrate against `cjxl`, 4 MP:

| metric | jpxl bpp | cjxl bpp | cjxl saving |
| --- | --- | --- | --- |
| SSIMULACRA2 85.3 | 1.996 | ~1.76 | **−12%** |
| PSNR 37.4 dB | 1.996 | ~1.49 | −25% |
| butteraugli 2.01 | 1.996 | ~1.21 | **−39%** |

So the "~25–40% behind" from entry 4 was a PSNR artefact at both ends. On SSIMULACRA2 we are **much closer than PSNR suggested** (−12%); on butteraugli — the axis `cjxl` actually optimises — we are **further behind than PSNR suggested** (−39%). Both readings are useful: the first says the codec's core is sound, the second says the gap is concentrated in perceptual bit-allocation (butteraugli-driven adaptive quantization), not in the transform or entropy layers.

**Corrected diagnosis of the high-rate undershoot** (entry 4, finding 3, now amended there). I attributed it to the quantizer ladder running out of rungs. **That was inferred from the existence of `RateOutcome::saturated`, never measured, and the data contradicts it:** at 12 MP `--bpp 4` succeeds (3.942) while `--bpp 2` misses (1.673). A ladder ceiling cannot do that. The trap I wrote against raising `max_prices` rested on the same unmeasured inference and is withdrawn.

**A change I wrote and then reverted, deliberately.** I implemented false-position interpolation for the bisection phase (aiming at the target instead of stepping blindly) on the theory that the undershoot is budget exhaustion. It broke `the_loop_survives_a_non_monotone_pocket`: a different probe sequence loses the sawtooth's optimum. Gating it on bracket width cannot separate the cases — the pocket fixture's bracket is ~2048 rungs wide and a small target's is ~256, so any threshold that helps one disables it for the other. **Reverted rather than shipped**, because (a) it changes search outcomes in non-monotone cases, which is a real contract, and (b) I still had not measured that budget exhaustion is the cause. Building a fix for an unverified cause is the same mistake as the ladder-ceiling claim above.

**Next — measure before building, this time:**
1. Run `jpxl encode --bpp` on the undershooting cases with the new note, and read `saturated` + the fast/full price split. That single number decides between "ran out of ladder" and "ran out of prices", and nothing should be built until it does.
2. If it is prices: the interpolation idea is still right, but needs a stepping rule proved safe under non-monotonicity — probably interpolate to *choose the bracket*, then keep the existing dense endgame intact.
3. **Cover-search bounding is NOT the lossy speed lever** — see the 2026-08-07 (9) entry: cover scoring is a small slice next to CfL/entropy/quantize, and the prior prune attempt measured a net *loss*. The lever is the probe count (15–40 full encodes), not per-probe cover cost.

**Traps — do not "fix" these:**
- Do **not** re-land bisection interpolation without a test proving it preserves `the_loop_survives_a_non_monotone_pocket`'s global-optimum property. It is easy to make the loop faster and quietly worse.
- Do **not** rank two encoders on a single (bytes, metric) point. Use curves, and say which metric — the 4 MP table above shows the same pair of encoders looking 12% or 39% apart depending on the choice.
- Do **not** enable the perceptual features by default to make a harness simpler. They exist opt-in on purpose.

## 2026-08-10 (4) — Phase 4H: the lossy path is finally reachable and measurable

**What changed (files: `crates/jpxl-cli/src/main.rs`, `crates/jpxl-conformance/src/metrics.rs` — only these two; no encoder change):**
- `jpxl encode --bpp <f>` / `--target-bytes <n>` route to `jpxl_encode_policy::encode_srgb8_to_target`. No rate flag → the lossless modular encoder, unchanged. 8-bit RGB only (VarDCT converts sRGB8→XYB; greyscale and deeper samples are refused, not mangled).
- **No `--distance` flag, on purpose.** `cjxl -d` targets butteraugli; this encoder has no perceptual model and its rate loop hits a *size*. A `--distance` flag would promise something the encoder cannot deliver.
- `jpxl_conformance::metrics::{rmse, psnr}` over the integer PPM `Image`, plus `jpxl compare <ref.ppm> <b.ppm>`. RMSE previously existed only on the NPY `FloatImage` conformance path (per-channel, decode-vs-float-reference) — not what comparing two *encoders* needs.
- `.agent/scratch/effort-ramp-2026-08-10/headsup-lossy.ps1`: RD sweep, jpxl `--bpp` vs `cjxl -d`, decoding cjxl's output with the oracle's own `djxl` so a JPXL decode bug can't masquerade as a cjxl quality difference.

**Proved:** `jpxl encode --bpp 1.0` on the 0.8 MP image emits 97,769 B against a 98,304 B target, and **both** `jpxl decode` and the independent `djxl` oracle accept it. 44/44 `jpxl-conformance` lib tests, with PSNR/RMSE pinned to hand-computed values.

**Why this mattered enough to do now:** VarDCT has been the more complete half of this encoder since M1–M8, but nothing outside the library and `jpxl bench` could invoke it, and it had **never been compared against the oracle**. That is the same blind spot that let the modular effort ladder ship inert for a whole phase.

**The numbers** (`jpegxl-rs.observation.lossy-vardct-headsup-2026-08-10`; raw rows in `headsup-lossy-2026-08-10.txt`). Read at **equal PSNR**, which is the only fair comparison:

| | jpxl bpp | cjxl bpp | cjxl saving |
| --- | --- | --- | --- |
| 0.8 MP @ ~34 dB | 1.988 | 1.466 | −26% |
| 4 MP @ ~37.5 dB | 1.996 | 1.490 | −25%, *and* 0.4 dB better |
| 12 MP @ ~39.4 dB | 1.673 | 1.074 | −36% |

Three findings:
1. **Density: cjxl reaches equal PSNR at ~25–40% fewer bits, and the gap widens with image size.** This comparison is if anything *unfavourable* to cjxl — it optimises butteraugli, not the PSNR being measured — and it still wins on every rung.
2. **Speed: 58–530× slower**, at every size and rate (8.7–55.6 s vs 0.09–0.15 s at 0.8 MP; 63–472 s vs 0.75–1.53 s at 12 MP). Cause is structural and familiar: `rate::search_frame` runs 15–40 full frame encodes, and `tile_region`/`block_cost_bounded` re-runs the exact `HfQuantizer::choose` loop over every coefficient cell *per candidate transform* inside each one. Same defect as the modular ranker — per-candidate work proportional to frame size with no sample bound — one layer up.
3. **The rate loop undershoots at high rates, and nobody was looking for this.** Asked for 4.0 bpp at 4 MP it delivered 3.645; asked for 2.0 bpp at 12 MP it delivered 1.673 (−16%).

   **Correction (2026-08-10, entry 5):** this entry originally attributed the undershoot to "the finest ladder rung still under target". That was inferred from the existence of `RateOutcome::saturated`, not measured, and the data contradicts it: at 12 MP `--bpp 4` *succeeds* (3.942) while `--bpp 2` misses. A ladder ceiling cannot behave that way — if the ladder could not reach 2 bpp it could not reach 4 bpp either. See entry 5 for the measured cause.

**Traps — do not "fix" these:**
- Do **not** add `--distance` as an alias for `--bpp`. It would read as a butteraugli target that does not exist.
- Do **not** quote a single (bytes, PSNR) pair as beating or losing to cjxl. Only the curve is meaningful; PSNR is used because it is exactly reproducible in-repo, not because it is the right quality model.
- `--effort` is lossless-only and is ignored on the lossy path; the lossy path has no effort ramp at all yet (`RateSearchBudget{max_prices: 40}` is its only dial).
- ~~Do **not** "fix" the high-rate undershoot by raising `max_prices`. The loop is not running out of *probes*, it is running out of *ladder*.~~ **Withdrawn — this trap was wrong**, and rested on the same unmeasured inference corrected in finding 3 above. See entry 5.

**Next for the lossy path, in priority order:**
1. **The high-rate ceiling** (finding 3) — a correctness-shaped limit, not a tuning one. Nothing else matters if 12 MP tops out at 1.67 bpp.
2. **Bound the cover search's per-candidate cost** (finding 2) — the same fix shape as 4G, one layer up: `block_cost_bounded` should score on a bounded sample rather than every coefficient cell.
3. **A lossy effort ramp** — there is currently no dial between "40 full encodes" and nothing.

## 2026-08-10 (3) — Phase 4G: the effort ladder was inert because the cheap ranker compared two different units, not because trees were too shallow

**What changed (files: `crates/jpxl-encode/src/lossless.rs`, `crates/jpxl-encode/src/modular/mod.rs`, `crates/jpxl-cli/src/main.rs` — only these three):**
- `ModularSearchBudget::cheap_sample_budget` + `effective_cheap_stride(budget, floor, w, h, planes)`, expanded **once** at the `plan_by_full_search` stage boundary. `cheap_row_stride` becomes a *floor*. A stride is a ratio, so a fixed stride still costs `frame_area / stride`; a sample budget is an absolute bound. Every level ships at `u64::MAX` (unbounded) for now — the plumbing is a provable no-op.
- **The real fix.** `total_cost_source` added an *unscaled* whole-frame `ma_tree_bit_cost × sections` to a residual estimate covering only `1/row_stride` of the frame. **The two terms were in different units.** A split's saving was divided by the stride while its cost was not, so every extra context was over-priced by exactly the sampling ratio — and the ratio grows with frame size. `estimate_residual_bits_sampled` now extrapolates its data cost to whole-frame bits (the fixed ANS-table overhead is deliberately *not* extrapolated: one set of tables is emitted however densely you sampled).
- `EncodeOptions::modular_search_overrides` + `jpxl bench modular --modular-{max-depth,max-leaves,sample-budget,deep-cap}`, so ladder constants get chosen from evidence instead of guessed.

**The gate fired with the opposite answer to the one planned for.** The plan assumed `deep_search_sample_cap` was the blocker and depth was the prize. Retiring the cap changes **nothing**: with `--modular-deep-cap full` at 4 MP, efforts 7, 8 and 9 all converge on the same 4,942,547 B / `771811b28ce7ab6e` they reach *with* the cap. Depth beyond one split, and the finer `THRESH_FINE` grid, buy exactly zero bytes.

**Mechanism, measured before the fix** (4 MP, same content, same candidate split): stride 1 (e9) **finds** it → 4,942,547 B. Stride 4 (e7) **rejects** it → 5,157,974 B. The verdict flipped purely on the sampling ratio.

**Proved after the fix:**

| | before | after | Δ |
| --- | --- | --- | --- |
| e7 0.8 MP | 1,075,465 | 1,037,868 | −3.5% |
| e7 4 MP | 5,157,974 | 4,942,547 | −4.2% |
| e7 12 MP | 9,795,599 | 9,795,599 | — |

Default (effort 1) is **byte-identical on all three sizes** — it short-circuits the search entirely. 122/122 `jpxl-encode` lib tests. vs `cjxl -e7` at 4 MP the gap narrows from +25.9% to +20.6% — real, and nowhere near closed.

**Two things I changed that were previously load-bearing, deliberately:**
1. **Effort 7 is no longer byte-identical to the pre-ramp encoder.** That anchor property is retired; `verify_default_flip.ps1` now pins e7 to its own post-fix baseline and the *default* remains the identity gate. `effort_7_budget_is_the_pre_ramp_search` still pins the budget **fields** — it was always a budget test, not an output test.
2. **Effort 7 at 4 MP went from ~5.7 s to ~47–50 s** for the same ~71 residual scans. Cause: the tree that now wins contains the Weighted (H.5) predictor, and `estimate_residual_bits_sampled` must full-scan any Weighted-containing tree; once Weighted wins the sweep, all 56 split candidates carrying it forward become full-frame scans. This is a real regression in an opt-in level, traded for the density above.

**Next, in priority order:**
1. **Bound the Weighted cost** — score split *topology* with a stateless proxy, restore Weighted only in per-leaf refinement. This is what makes the −4.2% affordable rather than an 8× tax.
2. Only then revisit finite `cheap_sample_budget` values; with depth proven inert there is no reason to spend the budget on deeper search.
3. The remaining +20.6% vs `cjxl -e7` is **not** depth. Suspects, unmeasured: context modelling / clustering quality, and the known Phase 4D gap (`allow_lz77 = false` during search, `true` at emission).

**Traps — do not "fix" these:**
- Do **not** re-pin effort 7 to the old fingerprints. They encode the units bug.
- `effective_cheap_stride` must never return below its floor — it is a work-*reducer*; letting it lower the stride would silently make small frames slower.
- The table-overhead term in `hybrid_estimate_from_events` must stay **outside** the extrapolation. Scaling it would re-introduce a units error in the other direction (~51 kbit of phantom cost at a 200× stride).
- `both_oracles_decode_a_stream_from_the_hf_mul_segment` (`jpxl-encode-policy/tests/vardct_oracle.rs`) fails on this host — **verified pre-existing** by stashing and re-running at clean HEAD. Lossy path; unrelated to this work.

## 2026-08-10 (2) — Phase 4F: elide the search the budget already answered; and a libjxl head-to-head that reframes the density story

**What changed (files: `crates/jpxl-encode/src/lossless.rs` — only this one):**
- `ModularSearchBudget::search_is_a_foregone_conclusion()`: true when the budget has one predictor, an empty split property *or* threshold grid, no leaf refinement, and neither transform trial. Under it the plan is provably `single_leaf(predictors[0])` with the caller's `rct` and no palette/squeeze, so `plan_for` now returns it directly.
- The search body moved to `plan_by_full_search`, so the elision is *proved* rather than argued: `degenerate_budget_plans_what_the_full_search_would_have` runs both paths on the same input across every degenerate level × both `rct` polarities × 1 and 3 channels and compares plans field-for-field. `the_default_effort_reaches_the_short_circuit` fails loudly if a future budget edit stops level 1 being degenerate.

**Proved:** byte-identical everywhere (corpus sha256 unchanged at default and at `--effort 7`); 12 MP default 1025.6 → 873.5 ms median (**−14.8 %**), `residual_scans` 1 → 0, `plane_clone_bytes` 288 MB → 144 MB. 117/117 `jpxl-encode` lib tests.

**The head-to-head that matters more than the change** (`jpegxl-rs.observation.libjxl-headsup-modular-2026-08-10`; driver + raw output in `.agent/scratch/effort-ramp-2026-08-10/headsup.ps1` / `headsup-2026-08-10.txt`). vs `cjxl` v0.13.0 on the three-image corpus, lossless, default threads:

| | jpxl e1 | jpxl e9 | cjxl -e1 | cjxl -e7 |
| --- | --- | --- | --- | --- |
| 0.8 MP | 1,075,465 / 295 ms | 1,037,868 / 10,291 ms | 1,082,673 / 61 ms | 873,503 / 144 ms |
| 4 MP | 5,157,974 / 521 ms | 4,942,547 / 55,156 ms | 5,414,251 / 33 ms | 4,097,956 / 520 ms |
| 12 MP | 9,795,599 / 1,272 ms | 9,795,599 / 65,437 ms | 9,870,951 / 54 ms | 6,921,886 / 1,147 ms |

1. **Density: we are a `cjxl -e1` encoder.** We beat `-e1` by 0.8–4.7 % and lose to `-e7` by 20–31 %. At 12 MP `cjxl -e7` is 29 % smaller in the *same* wall time our default takes.
2. **Speed: 5× / 16× / 24× slower than `cjxl -e1`.** The gap grows with image size, so it is throughput, not fixed overhead.
3. **The ladder above level 1 is inert, not merely chaffy.** At 12 MP levels 1–9 emit *byte-identical* output for 51× the time. 4E called this "chaff on photos"; the head-to-head shows it is worse than that — the ladder cannot reach density at any real image size.

**Root cause of (3), found by reading the code, not guessing:** `plan_for` collapses to `max_leaves = 2, max_depth = 1` whenever the frame exceeds `deep_search_sample_cap`, whose *largest* value is `1 << 18` = 262,144 samples at effort 9. The smallest corpus image is 786,432 samples. **Every real photograph is over the cap at every level**, so no effort ever searches past one binary split. The extra predictors, finer thresholds and finer stride that levels 2–9 buy are all applied to a tree that can never grow.

**Next — the higher band, in priority order (scoped, not built):**
1. **Make tree-search cost independent of frame size**, then retire the sample cap. The cap exists because the cheap ranker scans proportionally to the frame; bound the *ranker's* sample budget instead (score on a fixed number of sampled rows) and depth becomes affordable at 12 MP. Until this lands, nothing else in the higher band can be measured, because the tree is pinned at 2 leaves.
2. **RCT type selection.** `LosslessPlan.rct` is a `bool` — only `RCT_TYPE_YCOCG` is ever emitted, where H.6.3 defines 42 types. Cheap to search, unknown payoff, currently untested.
3. **LZ77 inside the cost model** (the pre-existing Phase 4D gap: `allow_lz77 = false` during search, `true` at emission).
4. **Per-sample throughput.** 873 ms for 36 M samples is ~24 ns/sample against `cjxl -e1`'s ~1.5 ns. Needs a profile before any guess.

**Traps — do not "fix" these:**
- Do **not** raise `deep_search_sample_cap` on its own to unblock the higher band. The cap is load-bearing against the *current* ranker's cost: at 12 MP a depth-4 search over 56 split candidates is ~71 full-frame residual scans. Fix item 1 first, then the cap can go.
- `search_is_a_foregone_conclusion` is a **work-elision predicate, not a heuristic**. If a future budget makes any skipped stage able to move the answer, the predicate must stop returning true for it — do not "extend" it to near-degenerate budgets.
- Environment correction to the 4E entry: **libjxl *is* built for Windows here.** `JPXL/tools/oracle-bin/` holds `cjxl.exe` / `djxl.exe` / `jxlinfo.exe` (v0.13.0, 196a43d9, MinGW, AVX2) with their MinGW DLLs beside them, per `PINNED_REVISIONS.txt` dated 2026-08-05. The WSL + `LD_LIBRARY_PATH=.agent/scratch/oraclelibs` route 4E used was unnecessary; use the Windows binaries directly.
- `.mcp.json` pointed `akr-mcp` at `/home/dk/.local/bin/akr-mcp`, a WSL path, which is why the `knowledge.*` MCP server "went unresponsive" in 4E. Now resolved from `PATH` (`C:\Users\dk\.local\bin\akr-mcp.exe`). The `akr` CLI was and remains a fine substitute.

## 2026-08-10 — Phase 4E: lossless modular effort ramp, then reoriented to a lean default (the search above e1 is chaff on real content)

**What changed (files: `crates/jpxl-encode/src/lossless.rs`, `crates/jpxl-encode/src/lib.rs`, `crates/jpxl-cli/src/main.rs` — only these three):**
- Added `Effort(u8)` (1..=9) and an internal `ModularSearchBudget` expanded **once** at the `plan_for` stage boundary (per `Encoder-plan1.md` §12: a budget selected at a stage boundary, never effort checks scattered through kernels). The budget parameterises every lossless search lever: predictor set, split property/threshold grid, tree depth/leaf caps, deep-search sample cap, per-leaf refinement, palette/squeeze trials, and the sampled-gather row stride. `EncodeOptions.effort` + `jpxl encode --effort N` + `jpxl bench modular --effort N`.
- Stage-C exact re-price is now skipped when neither palette nor squeeze will run (pure overhead otherwise; the emitted tree is unaffected).

**The finding that reoriented the work (user's steer + bpg-rs `4e68042` pattern):** on real content the full search is **chaff**. Measured at 12MP (`jpxl bench modular --diag`): effort 1 does **1** residual scan, effort 7 does **71**, and both emit the **identical fingerprint `dc380915d983b360`** / **9,795,599 bytes**. Content sweep (`.agent/scratch/effort-ramp-2026-08-10/`): effort 1 is within **0%** of the best on photos and gradients, ~3.7% on a flat-block screenshot, at **5–17× the speed**. The extra predictors, Weighted full-scans, 56-candidate split grid, per-leaf refinement, and finer stride buy essentially nothing on the common case.

**Decision (default output changes on non-photo content only):** `Effort::DEFAULT` is now the lean **level 1**, not the full search. The full pre-ramp search is retained as **level 7**, and higher levels (8–9) add finer stride/deeper search for density chasing. This prunes the chaff out of the *default path* while keeping every method reachable for the content that needs it (real screenshots → palette; textured → Weighted; etc.), following bpg-rs's "canonical presets, keep the expensive tier opt-in" discipline rather than deleting load-bearing methods.

**Proved:**
1. **Default flip is byte-identical on the photo corpus.** `jpxl encode` (default) vs the pre-change default: sha256 IDENTICAL on small/mid/large (`C3C60CEE…`, `5C49AB90…`, `CA27799D…`), at 297ms/561ms/1338ms vs the old 1359ms/6752ms/22629ms — a 4.6–16.9× speedup for the *same bytes*. The common case does not regress.
2. **`--effort 7` reproduces the pre-change fingerprints exactly** (the retained density anchor); a unit test (`effort_7_budget_is_the_pre_ramp_search`) pins level 7's budget field-for-field to the old constants.
3. **Every effort round-trips losslessly** through `jpxl-decode` on a multi-group frame, and is deterministic per level (`every_effort_round_trips_losslessly_on_a_multi_group_frame`, `same_effort_is_deterministic`). `jpxl-encode` lib: **115/115** green.
4. Oracle usable again: WSL `cjxl`/`djxl` run with `LD_LIBRARY_PATH=.agent/scratch/oraclelibs` (locally-extracted `libgif.so.7`, no system install). cjxl e1/e7 anchors reproduced from the cached corpus.

**Next (the actual pruning follow-up, not done here):**
- Deeper prune needs a **broader real-world corpus** (UI/screenshots/line-art with anti-aliasing) to measure per-method load-bearing rates before deleting any method's *code* — my synthetic fixtures are too degenerate to prove any method universally dead (e.g. the flat-block screenshot didn't even need palette). Until then, prune from the *default path* only (done), keep methods opt-in.
- A content-adaptive default (cheap classifier → enable palette/splits only when they'll pay) would recover the ≤3.7% the lean default leaves on flat/paletteable content without the 5–17× tax. Scoped, not built.

**Traps — do not "fix" these:**
- `jpxl-conformance` `bike_5_reference_npy_and_thresholds_are_readable` fails with `BadMagic`: the gitignored corpus `reference_image.npy` is absent/placeholder on this host. Pre-existing (fails identically with my changes stashed); fix is `tools/fetch-conformance.sh`, not code.
- `cargo fmt --all --check` and `cargo clippy --workspace -- -D warnings` are **red on clean HEAD** (28 fmt-drift files incl. `jpxl-core`/`jpxl-bitstream`/`jpxl-encode-policy`; `jpxl-core` `excessive_precision` ×70) — toolchain-1.97.1 drift that predates this change. Do **not** reformat/rewrite those 28 files to make the gate green (AGENTS §8: no edits outside the task's file set). My three files are clippy- and fmt-clean; `lib.rs:598` fmt drift is pre-existing code I did not touch.
- Do **not** re-inflate the default effort back to the full search "for density": it buys ~0% on photos/gradients and costs 5–17×. Density chasing is `--effort 7-9`, opt-in by design.

**Ledger (created this session, via the `akr` CLI — the `knowledge.*` MCP server went unresponsive mid-session with "unsupported call"):** `jpegxl-rs.work.arch-phase4e-modular-effort-ramp` (**proposed**, part_of `jpegxl-rs.track.encoder-optimization`, depends_on 4B), `jpegxl-rs.assessment.modular-effort-lean-default` (verified), `jpegxl-rs.decision.modular-lean-default` (active). **Deferred to the commit (forward-only):** the two evidence records (chaff measurement; ramp-landed verification) and `knowledge.complete` on the work record — recording evidence against pre-change HEAD then completing after a commit is the documented "evidence predates completion" landmine (AGENTS §10 AKR gotcha), so the code must be committed via `akr change`/`akr git commit` first, then evidence pinned to that commit and the work completed with `verify_default_flip.ps1` + the lib tests as check evidence. Scratch: `.agent/scratch/effort-ramp-2026-08-10/` (ramp.ps1, content_experiment.ps1, verify_default_flip.ps1, make_fixtures.ps1, rec-*.akr body files) and `.agent/scratch/oraclelibs/` (extracted libgif for the WSL oracle).

## 2026-08-07 (9) — S8 Phase D: prune wired behind a flag; safe and decision-preserving, but slower — flag stays off

**What changed:**
- `Cargo.toml`: new `s8-cover-prune` feature, **not** in `default` — with it
  off, `block_cost_bounded` compiles byte-for-byte as before this phase (the
  new code is entirely behind `#[cfg(feature = "s8-cover-prune")]`).
- `lib.rs`: `cheap_stage_would_prune` — the Phase C-proved staged bound,
  checked with the running `bits`/`weighted_sse` already accumulated from
  prior channels, at the same Y-then-X-then-B checkpoints
  `block_cost_bounded` visits, *before* that channel's exact
  `choose`/`choose_lane4` loop runs. Only fires when `cutoff.is_some()`
  (merge-candidate scoring, same as the existing exact `check` cutoff);
  `block_cost`'s unbounded calls (`cutoff: None`) are untouched.
- `diagnostics.rs`: `stage_cover_prune_ns`, `cover_prune_checks`,
  `cover_prune_hits` — the live analogue of Phase C's
  `PruneSummary`/`prune_rate`, timed separately from `stage_cover_score_ns`
  so the prune's own cost can be weighed against what it skips.
- `regret.rs`: `wired_prune_does_not_change_tile_regions_decisions`
  (`s8-cover-prune`-only test) — with the prune *actually* live inside
  `tile_region`'s real cutoff-bounded calls, `tile_region`'s total cost must
  still exactly match the harness's independent, never-pruned ground truth,
  over two fixtures (`ramp_frame`'s mild texture, a new `noisy_frame` —
  gradient-plus-hash-noise, same construction as `vardct_oracle.rs`'s AQ
  fixture — for more guaranteed-nonzero cells). `(tile_region_total -
  harness_total).abs() < 1e-6` on both. This is the *wiring's* proof, on top
  of Phase C's proof that the *primitive* is safe in isolation.

**Proved, both statically and by direct measurement:**
1. Lib tests: 65/65 with the feature off (unchanged), 66/66 with it on.
   `vardct_roundtrip` 12/12 both configs. `vardct_oracle` 9/10 both configs,
   identically — `both_oracles_decode_a_stream_from_the_hf_mul_segment`
   fails with the exact same numbers (peak 255, RMSE 119.627) regardless of
   the flag: still the same pre-existing, unrelated failure documented in
   Phase C.
2. **Real byte-identity**, not just the harness's cost-total proxy: `jpxl
   bench vardct-rate --diag`, flag off vs flag on, same input —
   - Synthetic 512×512, bpp=1.0: `fingerprint=cf22c12684ce773a` both runs,
     `output_bytes=31495` both. `cover_prune_checks=3072
     cover_prune_hits=277` (9.0% prune rate) with the flag on.
   - Real photo, 1024×768 (`.agent/scratch/realworld-bench-20260806T072202Z/small_0p8MP.ppm`),
     bpp=1.0: `fingerprint=576ebe32464ae23a` both runs, `output_bytes=97769`
     both. `cover_prune_checks=7943 cover_prune_hits=765` (9.6% prune rate).
   Fingerprint and byte count identical in every case: the wired prune
   changes zero bits of encoder output, on real content, not only on the
   two unit-test fixtures. Consistent with Phase C's 10% prune rate on its
   own fixture.

**Measured, honestly negative — the exit gate's "confirm the win
materializes" clause did not pass:**
- Synthetic 512×512: `cover_ms` 13.1 → 15.7ms (+20%). `cover_score_ms` did
  drop slightly (11.9 → 10.8ms — the skipped exact loops are real), but
  `cover_prune_ms` (3.4ms) costs more than that saving.
- Real photo 1024×768: `cover_ms` 36.9 → 46.3ms (+25%). Same shape:
  `cover_score_ms` 31.9 → 30.8ms, `cover_prune_ms` 10.5ms — net loss.
- Total wall time barely moves either way (cover scoring is a small slice
  of total encode time next to CfL/entropy/quantize), but the mechanism
  itself, measured in isolation, is a net loss, not a win.

**Why, most likely:** the cheap bound's per-cell loop is scalar
(`HfQuantizer::cell_lower_bound`, one call per cell) while the exact loop it
sometimes replaces is SIMD-batched (`choose_lane4`, 4 cells/call) — and
`choose` already has its own zero-threshold fast path for the
guaranteed-zero case, the same one `cell_lower_bound` reuses, so a
guaranteed-zero cell was already cheap before this phase. Checking *every*
candidate at *all three* stages to catch the ~9-10% that turn out prunable
costs more than it saves. This is the honest negative result the plan
explicitly allowed for ("if not, an honest negative result... is an
acceptable outcome, not a failure to fix by loosening a check") — not a bug
to chase.

**Decision:** the infrastructure lands, proven safe and decision-preserving
by both the exhaustive Phase C proof and this phase's real-content
byte-identity measurement — but `s8-cover-prune` is **not** promoted to
`default` and should not be turned on. If a future session wants to revisit
this, the fix is architectural (batch `cell_lower_bound` the way
`choose_lane4` batches `choose`, or check only at Y — the stage where the
running total is smallest and the check is cheapest relative to what
remains) — not a reason to touch the safety proof.

Evidence: `jpegxl-rs.evidence.s8-phase-d-prune-wired-verified`. Work item
`jpegxl-rs.work.arch-s8-phase-d-prune-wired` completed; parent scoping
record `jpegxl-rs.work.arch-s8-full-redesign-scoped` updated.

**Next: Phase E** (corpus provenance — required before any *new* fixture
from this plan can merge; Phase D introduced none, reusing existing
fixtures, so E's trigger condition may not even fire) or **Phase F**
(explicitly staged out, gated on a Phase D win that did not materialize —
if pursued at all, it would need its own justification, not inherit Phase
D's).

Two sections at the bottom are permanent and must be kept current:
"Already fixed — do not redo" and "Traps — do not fix these by loosening a
check". When a diagnosis turns out to be wrong, correct it **in place** and
mark it corrected; do not leave a wrong explanation standing.

Keep this file small. Entries whose content has landed in `PLAN.md`,
`CONFORMANCE.md`, or `docs/experiments/` get deleted from here.

---

## 2026-08-07 (8) — S8 Phase C: transient provable lower bound, zero safety violations

**What changed, purely additive** (`lib.rs` — the production cover-scoring
path — is untouched by this phase):
- `quantize.rs`: `HfQuantizer::cell_lower_bound(target, channel, cell)` —
  exact distortion floor for cells guaranteed to zero-quantize
  (`lambda*side^2*target^2`, via the same `zero_threshold` shortcut
  `choose` already uses), loose-but-*provable* `>=2`-bit rate floor for
  cells guaranteed nonzero (the minimum any nonzero `residual_bits` value
  can be). Same error semantics as `choose`.
- `regret.rs`: `validate_candidate_prune`/`measure_prune_safety` — a
  *second* independent quadtree walk (alongside Phase A's `measure_region`),
  staging the bound exactly as `block_cost_bounded`'s real Y-then-X-then-B
  running-total order and units would, so what's validated here is the
  actual integration Phase D would wire in. B's cheap check uses Y's *real*
  reconstruction (`d_y_hf`), never raw `cb` — `kB=1.0` makes that coupling
  safety-load-bearing (an unsafe bound could otherwise exceed the true
  cost); `kX=0.0` needs no such care.

**Proved, exhaustively (not statistically — a provable bound means one
counterexample is a bug):**
1. `quantize::tests::cell_lower_bound_never_exceeds_the_exact_cost` — 6
   quantizer configs × 12 boundary-stressing targets, `bits_lb <=
   bits_exact` and `sse_lb <= sse_exact + 1e-9` in every case.
2. `regret::tests::cell_lower_bound_prune_never_discards_a_true_winner` —
   over a 128×128 ramp+texture fixture: `S8_PHASE_C_PRUNE_SAFETY
   candidates=80 safety_violations=0 pruned=8 prune_rate=0.1000`.
   `safety_violations == 0` is the exit gate; `prune_rate` is recorded as
   the separate usefulness measurement, not asserted against a threshold —
   that's Phase D's decision to make with real corpus data.

All 65 `jpxl-encode-policy` lib tests and all 12 `vardct_roundtrip` tests
pass. `vardct_oracle`'s `both_oracles_decode_a_stream_from_the_hf_mul_segment`
fails identically on `main` before this change (confirmed via `git stash`)
— pre-existing, unrelated. No `--diag` byte-identity check applies: nothing
in the decision path changed.

Evidence: `jpegxl-rs.evidence.s8-phase-c-prune-bound-verified`. Work item
`jpegxl-rs.work.arch-s8-phase-c-prune-bound` completed; parent scoping
record `jpegxl-rs.work.arch-s8-full-redesign-scoped` updated.

**Next: Phase D** — wire the safe prune behind a feature flag; survivors
still scored exactly (`block_cost_bounded` unchanged); new decision-quality
fixtures with an explicit tail-regret budget (placement-conformance tests
alone don't catch a legal-but-worse pick); confirm the realistic (low-teens
percent, not the 31-35% upper bound) win actually materializes.

---

## 2026-08-07 (7) — S8 Phase B: measured, advisor consulted, memory-discard
## design dropped — Phase C simplifies

**What changed.** Two measurements, both landed as new diagnostics/modules,
zero decision-path impact:
- `diagnostics.rs`: `StageTimer::CoverForward`/`CoverScore` wrap
  `block_cost_bounded`'s `cache.get_or_insert` call and its
  `score_channel_lanes` calls respectively.
- New `stability.rs`: `measure_winner_stability` drives a real
  `rate::search_frame`, replays every distinct probed quantizer through
  `plan_at_on` sharing one cache (mirroring the real rate loop exactly), and
  diffs consecutive probes' covers at **atom granularity** — not varblock
  origins, which aren't comparable across probes once the quadtree
  decomposition itself differs; an atom always belongs to exactly one
  varblock in every probe, so atom→transform is the invariant comparison
  unit. (Confirmed correct by the advisor, no redo needed.)

**Measured.** Choose-loop score share: 31–35% of cover time, stable across
a 12MP and a 1024×768 image (well above the ~15% kill threshold). Winner
churn over a real 33-probe search: mean 9.4%, but the aggregate hides a
clean pattern — churn is *exactly* 0.0000 for 14 of 33 adjacent pairs
(monotonic bisection convergence) and concentrates in 4 pairs at the
initial geometric bracket phase and the Fast→Full refinement handoff. One
unexplained spike (a tiny 32-rung step producing 97.27% churn inside the
converged region) is flagged, not asserted-explained — folded into Phase E
as a natural-photo re-run, not blocking.

**Advisor consulted with the real data (same advisor session reused, via
`SendMessage`, as the full-plan synthesis).** The verdict is decisive
independent of the churn pattern: `CandidateForwardCache` keys on
`(transform, px, py)` only — forwards are **probe-invariant by
construction**, already get the measured 96–97% cross-probe reuse for
free, and are the *dominant* 65–69% share of cover cost. This session had
conflated two separate things: "the prune" (skip the choose-loop for
losers — a within-probe compute optimization that never touches the cache)
versus "the memory-discard design" (§8's original cache-summaries-not-
coefficients idea — the *only* thing winner churn actually bears on).
Given probe-invariant, dominant, already-free-to-reuse forwards, discarding
them to save memory is a strictly bad trade **regardless of churn** — the
advisor's explicit correction to this session's own Q1 (which proposed
phase-gating the prune to the stable-convergence region) called that a
*category error*: the prune can't hurt cross-probe reuse in any phase since
it never touches the cache, so it should just run always.

**Decision: drop the memory-discard design outright.** Keep the existing
full-coefficient cross-probe cache exactly as-is. Phase C **simplifies**:
the compact per-candidate summary becomes a *transient* structure built
from the already-cached forward, used only to compute the provable lower
bound for the within-probe prune, then dropped — it does not replace what
the cache stores. This deletes the entire "summarize-then-discard-
coefficients" hazard surface and the cross-probe-winner-stability question
from Phase C/D's critical path. Phase D is unchanged in substance but
easier to reason about with no cross-probe interaction left. Realistic
sizing correction: 31–35% is an *upper bound* on the prune's win (only
losers get skipped; the dominant forward cost is paid by everyone
regardless) — expect a modest, low-teens-percent-of-cover-time win, not a
transformative one.

Evidence: `jpegxl-rs.evidence.s8-phase-b-measurements-verified`. Work item
`jpegxl-rs.work.arch-s8-phase-b-measurements` completed; parent scoping
record `jpegxl-rs.work.arch-s8-full-redesign-scoped` (rev 2) updated with
the revised Phase C/D/E scope.

**Next: Phase C** — the transient provable-lower-bound summary, validated
exhaustively against the Phase-A regret harness's exact-scorer ground truth
over the corpus.

---

## 2026-08-07 (6) — S8 Phase A landed: regret harness, proven as a no-op

**What changed.** New `crates/jpxl-encode-policy/src/regret.rs`: a
`CoverSurrogate` trait (given the same two numbers `tile_region` compares —
`split_cost`, `single_cost: Option<f64>` — decides split vs merge),
`ExactPolicy` (wraps `tile_region`'s own tie rule so the harness can be
proven against itself), `RegretSample`/`RegretSummary` (agreement rate,
mean/max/p99 regret — tail reported explicitly, not just mean, since both
CfL regressions were rare/systematic wrong picks), and `measure_region` — an
**independent** recursive quadtree walk that never touches `tile_region` or
the production encode path at all, always computing the *unbounded* exact
merge cost at every node (deliberately ignoring `tile_region`'s own Phase-1
cutoff, since regret needs a true cost, not a bound).

**Proved, exceeding what the plan asked for.** Two tests:
- `exact_policy_is_a_true_no_op` — the Phase-A exit gate: wiring the exact
  scorer in as both surrogate and oracle yields exactly 0 regret and 100%
  agreement at every node.
- `measure_region_matches_tile_regions_own_total_cost` — not required by the
  plan, added anyway: cross-validates this module's independent walk
  against `tile_region`'s **actual production decisions** (not just
  self-consistency), so the two implementations can't silently drift apart
  over time. Both pass.

`--diag` fingerprint/byte identity on the 12 MP image confirms zero
production-path impact (`1,530,188` / `30e55ba3faef4d64`, unchanged) — this
module is purely additive by construction, not just by testing. Full
`jpxl-encode-policy` lib suite 62/62 (was 60), `vardct_roundtrip` 12/12.
Evidence: `jpegxl-rs.evidence.s8-phase-a-regret-harness-verified`. Work item
`jpegxl-rs.work.arch-s8-phase-a-regret-harness` completed; parent scoping
record `jpegxl-rs.work.arch-s8-full-redesign-scoped` updated to `active`.

**Next: Phase B.** Two measurements that can kill or reshape the rest of
the plan, before any summary machinery is built: the `choose`-loop's actual
cost share of cover scoring (vs. the forward DCT), and whether the
*winning* transform per region is stable across rate-loop probes (currently
unverified, load-bearing for whether a memory-saving "cache summaries, not
coefficients" design can preserve the measured cross-probe cache reuse).

---

## 2026-08-07 (5) — Full S8 redesign scoped (not implemented): six gated
## phases, honest deliverable is infrastructure + one conservative prune

**What happened.** Per explicit user request, scoped (did not implement)
the full outside-advice.md §8 surrogate-first cover-selection redesign as
an executable plan. Process: two parallel Explore agents (architecture/
machinery; safety/verification/prerequisites), each producing a
file:line-cited research report, then an Opus advisor synthesis of both,
independently checked against the code rather than accepted at face value.
Recorded durably as `jpegxl-rs.work.arch-s8-full-redesign-scoped` (state
`proposed` — this is a plan awaiting execution, not completed work).

**Two findings govern the whole plan:**
1. §8's core premise — "avoid forward-transforming losing candidates via
   cheap summaries" — is **mostly foreclosed**. Ranking a DCT16/DCT32
   candidate needs coefficient-domain data (a DCT32 spans 16 atoms; per-atom
   source mean/variance in `AnalysisAtlas` cannot recover its spectrum), so
   a useful summary requires the forward DCT to already exist — the forward
   is unavoidable. The achievable win shrinks to "skip the exact per-cell
   `choose`/`choose_lane4` scoring loop for losers after the transform
   already ran" — smaller than §8 as written implies, and **currently
   unmeasured**.
2. Rate (`residual_bits(q)`, discrete/non-smooth) can't be *tightly*
   bounded from a summary without approaching per-cell data, but a *loose,
   provable* bound exists (every guaranteed-nonzero cell costs ≥2 bits) and
   distortion *can* be tightly, exactly bounded for cells guaranteed to
   zero (a smooth quadratic, no quantizer call needed). So a provable lower
   bound is achievable — just looser than the existing exact partial-sum
   cutoff in `block_cost_bounded`'s `check`.

**The honest deliverable is infrastructure + measurement + one
conservative, provably-safe pruning application — not "the full §8
redesign, landed."** Six phases, each independently gated (full detail in
the AKR record's `note`, condensed here):
- **A** — a regret harness (agreement rate + exact-cost regret, with tail
  reporting) using the exact scorer as its own ground truth, proven as a
  no-op first (0 regret / 100% agreement wired to itself) before being
  trusted — no Butteraugli needed for this question, zero new deps.
- **B** — two measurements that can *kill or reshape* the plan before any
  machinery is built: the `choose`-loop's actual cost share of cover
  scoring (vs. the forward DCT), and whether the *winning* transform per
  region is stable across rate-loop probes (the measured 96-97% cross-probe
  cache hit rate only covers the *candidate set* being probe-invariant —
  winner-stability is unverified and load-bearing for whether a
  memory-saving "cache summaries, not coefficients" design survives).
- **C** — the compact per-candidate summary + provable bound, validated
  *exhaustively* (not statistically) against the exact scorer.
- **D** — the safe prune, flagged, survivors still scored exactly; new
  decision-quality fixtures required (the existing oracle suite proves
  *placement* conformance only, not decision *quality* — exactly the gap
  that let both CfL regressions through until a bespoke fixture caught
  them).
- **E** — `test-set/` provenance sidecars (currently missing, violates
  `AGENTS.md` §9) — required before Phase D's fixtures can merge.
- **F** — explicitly *not* attempted: the tight, continuous, closed-form
  rate estimate (§8's ambitious half), named so it isn't silently smuggled
  into an earlier phase under time pressure later.

**Out of scope for the whole plan:** `AnalysisAtlas` expansion (can't rank
DCT16/32 from source-pixel stats regardless; its raw-`frame.xyb()` vs.
Gaborish-preconditioned-planes mismatch is a separate real bug, noted but
not fixed here); Butteraugli/SSIMULACRA2 build-out (Phase 5 territory).

**Next.** A future session executes Phase A first.

---

## 2026-08-07 (4) — S8 (outside-advice.md §8): Opus advisor review, then a
## bit-identical coefficient-lane SIMD slice, not the full redesign

**Advisor review first.** Given Phase 2's two near-miss regressions from an
uncalibrated closed-form estimate at a much *smaller* decision surface
(chroma-from-luma factor), an Opus-model architecture review was run against
the actual §8 machinery (`AnalysisAtlas`, `CandidateForwardCache`,
`block_cost_bounded`, `tile_region`, `HfQuantizer::choose`) before writing
any code. Verdict: do not attempt the full §8 redesign (analysis pass +
compact per-candidate summaries + lower-bound pruning) this session — cover
selection changes *which transform exists*, an unbounded-tail failure mode
at a much larger blast radius than CfL's near-miss. Recommended instead: a
**bit-identical** coefficient-lane SIMD restructuring (outside-advice.md
§3's "vectorize adjacent coefficients, not one cell's four candidates" — the
existing `choose` SIMD path vectorizes the wrong axis), paired with an
agreement/regret test harness, as safe infrastructure that doesn't gamble on
cover-decision correctness. Leave `AnalysisAtlas` alone — expanding it now
tunes features against a consumer (the future summary scorer) that's
deliberately not being built yet, repeating the atlas's own documented
Milestone-1 mistake.

**What changed (landed).** `HfQuantizer::choose_lane4` (`quantize.rs`): four
adjacent coefficients at once via `wide::f32x4`, replicating `choose`'s exact
zero-threshold shortcut, `[0, estimate-1, estimate, estimate+1]` candidate
order/tie rule, and error semantics exactly — plus a whole-lane zero fast
path (added after measurement — see below). Non-SIMD fallback with the same
signature (four scalar `choose` calls), so callers don't feature-gate.
`score_channel_lanes` (`lib.rs`) drives it from `block_cost_bounded`:
row-segment iteration (skip each row's LLF prefix, batch 4 at a time, scalar
remainder), same raster accumulation order, same Phase-1 cutoff pruning at
cell granularity.

**A real regression caught before landing, not after.** The first version
(no whole-lane zero fast path) was bit-identical but made `cover_ms` ~35%
*slower*: `choose_lane4` unconditionally ran the full candidate search for
every lane, but the scalar `choose`'s zero-threshold shortcut is a
~2-instruction early return that most HF coefficients on real photos take —
a SIMD lane can't skip per-element work the way a scalar early return can,
so vectorizing without also fast-pathing the common all-zero case is *more*
total arithmetic, not less. Added `if zero_mask.all() { return zero }` before
the candidate search; re-measured. This is exactly the kind of thing the
exhaustive bit-identity test doesn't catch (it proves correctness, not
speed) — caught by actually benchmarking before declaring done, not by
trusting the "SIMD" label.

**Proved.** `choose_lane4_is_bit_identical_to_four_scalar_choose_calls`
(new, `quantize.rs`): 6 quantizer configs (transform, `HfMul`, `global_scale`,
`qm_scale` varied) × 10 target patterns per lane chosen to stress every
boundary (zero threshold from both sides, half-integer ties, the `|q|<=1`
bias-adjust branch edge, both sides of `MAX_QUANT`) — all bit-identical.
`--diag` fingerprint/byte identity on 12 MP, 3 repeated runs:
`output_bytes=1,530,188 fingerprint=30e55ba3faef4d64`, unchanged.
`cover_ms`: ~1784ms → stable ~1524–1532ms (14–15%). Full suite green modulo
the pre-existing, unrelated `both_oracles_decode_a_stream_from_the_hf_mul_segment`
failure: lib 60/60 (default) + 59/59 (`--no-default-features`, confirming the
non-SIMD fallback), `vardct_roundtrip` 12/12, `vardct_oracle` 9/10,
`rate_loop` 11/11 (+1 pre-existing ignore) — `rate_loop`'s own wall time
dropped from ~280s to ~28s, consistent with the change directly speeding up
the repeated-cover-search workload that test exercises. Evidence:
`jpegxl-rs.evidence.s8-cover-lane-simd-verified`. Work item
`jpegxl-rs.work.arch-s8-cover-lane-simd` completed.

**Diagnostics note.** `choose_cover`/`choose_total()` now undercount: most
cover-scoring decisions go through `choose_lane4`, which does not call
`note_choose`. Doc comment updated on `EncodeDiag::choose_cover`
(`diagnostics.rs`) so this isn't mistaken for the earlier phases' "fewer
calls" story — this is "fewer calls *to `choose` specifically*," not fewer
quantization decisions.

**Deliberately not attempted**, per the advisor's plan: FastDeadZone-in-
scoring (a cheaper approximate quantizer for candidate scoring only, safe
*if* gated by a bounded-regret generalization of this session's exact-
agreement harness) and the full §8 closed-form summary redesign (needs
either richer per-tile statistics than 2nd moments, or the matched-quality
harness outside-advice.md §19 describes — neither exists yet).

---

## 2026-08-07 (3) — Rate-loop `CandidateForwardCache` measured: size cap rejected

**What changed.** `jpxl bench vardct-rate --diag` now also prints
`rate_diag=fast_prices=… full_prices=… dct_cache_hits=… dct_cache_misses=…
dct_cache_hit_rate=…`, sourced from `RateOutcome.stats` (`RateProbeStats`,
which already existed but was never surfaced — `--diag` previously only
printed the *last individual probe's* `plan_at` breakdown, hiding cross-probe
cache behavior entirely). `BenchReport` gained an `Option<RateProbeStats>`
field; every other bench mode leaves it `None`.

**Measured, answering the open question from the previous entry's note:**
two images (4000×3000 and 1024×768, `--bpp 1.0`, default `max_prices=40`
budget) both land a **~96–97% cross-probe cache hit rate** over 23–25 total
probes (6,349,453/6,595,028 hits and 497,050/513,177 hits respectively).
Miss counts in both runs closely match a *single* `vardct-fixed` probe's
unique candidate count — confirming every probe after the first scores
exactly the same candidate set (position + transform is quantizer-
independent; only `block_cost`'s *score* of each depends on the probe's
quantizer) and gets an almost-total cache hit.

**Conclusion: a size cap on `CandidateForwardCache` is not viable.** It would
need to retain essentially 100% of one frame's unique candidates to preserve
this reuse; capping it below that trades wall-time (up to ~23× more
forward-DCT recomputation for evicted-then-needed entries) for a memory
saving that shrinks toward nothing as the cap approaches the size actually
needed. This closes option (a) from the prior entry's note. The only
remaining path to outside-advice.md §7's ~428 MB reduction is §8's
compact-per-candidate-summary redesign — score cover candidates from
summaries, materialize full coefficients only for the winner. That's a
bigger, separately-scoped change (touches `block_cost`/`tile_region`'s
candidate representation, carries real correctness/quality risk the way
Phase 2's CfL rewrite did) and was *not* attempted this session. Evidence:
`jpegxl-rs.evidence.phase3-rate-loop-cache-reuse-measured`. Note on
`jpegxl-rs.work.arch-phase3-forward-cache` updated with the full writeup.

**Also this session:** wrote a findings document (delivered to the user, not
committed to this repo) on a separate AKR-workflow observation — papercuts
about AKR's own behavior, logged while working in a consuming project, are
invisible to AKR's own maintainers unless someone specifically reads that
project's ledger. Confirmed via a real precedent (`bpg-rs` →
`AKR/docs/DECISIONS.md` D-028). No changes made to AKR or its docs; purely
investigative, for the user's own follow-up.

---

## 2026-08-07 (2) — Phase 3 (partial): selected-forward clone eliminated

**What changed.** `plan_at_with_cfl`'s per-LF-group cover-selection loop
(`lib.rs`) splits into two passes. Pass A does everything that mutates
`CandidateForwardCache` — cover selection, then the new
`ensure_forwards_cached` (a no-op per varblock under `CoverMode::Hierarchical`,
which already inserted the winner while scoring; real work under
`CoverMode::FixedDct8x8`, which never touches the cache) — across *every*
group. Only once no group needs `&mut cache` again does pass B run: the new
`CandidateForwardCache::get` (read-only) and `gather_forward_refs` borrow each
selected varblock's forward straight out of the cache, replacing the old
`forward_selected`'s `fwd.clone()`. `estimate_cfl`/`quantize_group` signatures
changed from `&[VarblockForward]`/`&[&[VarblockForward]]` to
`&[&VarblockForward]`/`&[&[&VarblockForward]]` — bodies untouched, Rust
auto-derefs through the extra reference.

This is a pure ownership/borrow restructuring — no decision logic touched, so
it's output-preserving by construction, not just by testing.

**Proved:** `--diag` on 12 MP `vardct-fixed`, `threads=1`: `output_bytes` and
`fingerprint` identical before/after (`1,530,188` /
`30e55ba3faef4d64`); `sel_clones`/`sel_bytes` `41,949`/`144,000,000` → `0`/`0`.
`jpxl-encode` + `jpxl-encode-policy` lib tests (59+103), `vardct_roundtrip`
(12), `rate_loop` (11+1 ignored) all pass; `vardct_oracle` 9/10 (the one
failure is the same pre-existing, unrelated
`both_oracles_decode_a_stream_from_the_hf_mul_segment` from the Phase 2
entry). Evidence: `jpegxl-rs.evidence.phase3-selected-forward-clone-eliminated`.
Work item `jpegxl-rs.work.arch-phase3-forward-cache` completed.

**Deliberately not touched: `CandidateForwardCache` itself** (the larger of
outside-advice.md §7's two numbers, ~428 MB at 12 MP — every *scored*
candidate, winners and losers, not just selected ones). `rate.rs:589` keeps
one `CandidateForwardCache` across all quantizer probes of one encode
specifically so a probe's cover search can skip re-running the forward DCT
for positions a prior probe already scored — up to ~40 probes per
`RateSearchBudget`. A naive LRU/FIFO cap risks evicting exactly the entries a
later probe needs, turning a cache hit into 40x the forward-DCT work: a real
wall-clock regression, not just a missed win. See the `note` on
`jpegxl-rs.work.arch-phase3-forward-cache` for the two ways to do this safely
(measure-then-cap, or the fuller outside-advice §8 compact-summary redesign)
— next session's actual next step, not this partial win.

---

## 2026-08-07 — Phase 2 closed: safe HF CfL window narrowing; full analytic
## elimination tried and reverted (regressed correctness twice)

**What changed (landed).** `factor_candidates` (`lib.rs`) narrows its HF CfL
search window from `seed±4` to `seed±1`; `refine_hf_factor` short-circuits to
the neutral factor when the closed-form least-squares seed is already `0`.
Both still score every remaining candidate through the exact per-sample
`HfQuantizer::choose` oracle (`hf_residual_cost_bounded`) — no scoring
semantics changed, just fewer candidates. `quantize_square_varblock` batches
final HF quantization through the new `HfQuantizer::quantize_lane` (a
cell-by-cell wrapper over `choose`, same integers, fewer call sites to read).
This was already staged, uncommitted, at the start of this session; this
entry validates and lands it as-is.

**What was tried and reverted.** The Phase-2 work item's original acceptance
target — `choose_cfl_y`/`choose_cfl_factor` *near zero* via a fully
closed-form HF factor decision from `CflAccumulator`'s 3 sufficient statistics
(`s_yy`, `s_yc`, `s_cc`), no per-sample scoring at all — was attempted twice
and regressed correctness both times:
1. An invented Shannon-style rate proxy (residual variance vs. a
   representative quantization step, priced against `factor_bits` signaling
   cost) made `refine_hf_factor` pick the neutral factor where the exact
   search picks non-neutral, failing the `vardct_oracle` "rgb-64x64" fixture's
   own precondition (`jxl_oxide_decodes_our_vardct_output`).
2. Trusting `CflAccumulator::best_factor`'s closed-form seed directly (no rate
   gate) fixed that, but then picked a non-neutral factor that cost *more*
   bytes than neutral for equal-or-better quality, failing
   `cfl_reduces_size_at_equal_quality_on_correlated_colour`.

Conclusion: 3 second-moment sufficient statistics aren't enough information to
reproduce the exact search's rate/distortion tradeoff at tile granularity. A
safe elimination needs either richer per-tile statistics (a histogram-ish
summary, not just 2nd moments) or a calibrated rate model built against the
matched-quality harness outside-advice.md §19 describes — guessing thresholds
against two ad hoc test fixtures is exactly what §4 of that document warns
against. Both attempts were fully reverted to the pre-session diff (verified
byte-for-byte against the original patch); no trace of either remains in the
landed code.

**Proved (`--diag` on the 12 MP test-set image, `vardct-fixed`, `threads=1`,
vs the Phase 0-1 baseline at `f73590d`):**
- `choose_total`: 285,647,118 → 175,406,647 (**-39%**).
- `choose_cfl_factor`: 154,335,540 → 44,095,069 (**-71%**), from the narrower
  window alone. `choose_cfl_y` unchanged at 11,812,500 — still exact, still
  needed by the retained per-sample scoring.
- Output size moved -0.09% (1,531,492 → 1,530,188 bytes), consistent with a
  narrower-but-still-exact search.
- `cargo test -p jpxl-encode -p jpxl-encode-policy --lib --release`: 59 + 103
  passed. `cargo test -p jpxl-encode-policy --test vardct_oracle`: 9/10 passed
  — the one failure (`both_oracles_decode_a_stream_from_the_hf_mul_segment`,
  jxl-oxide vs. jpxl-decode disagree by peak 255 at an extreme `global_scale`)
  reproduces identically on unmodified `f73590d`; **pre-existing, unrelated to
  this work, not yet triaged.**
- Evidence: `jpegxl-rs.evidence.phase2-safe-window-narrowing-landed` (landed
  state), `jpegxl-rs.evidence.phase2-analytic-cfl-regressed-correctness` (what
  was tried and why it was reverted). Work item
  `jpegxl-rs.work.arch-phase2-cfl-quant` completed against a revised,
  narrower acceptance target; its `note` slot carries this finding for the
  next attempt.

**Next.** Two independent items, neither started:
- **Re-open full analytic HF CfL** only alongside a matched-quality harness or
  richer per-tile summary — not as another isolated heuristic guess.
- **Phase 3** (`CandidateForwardCache` / `forward_selected`, `lib.rs`) still
  retain full per-candidate coefficient vectors frame-wide (outside-advice.md
  §7) — cover search still calls `HfQuantizer::choose` per-coefficient
  per-candidate inside `block_cost_bounded` (the ~84M `choose_cover` calls,
  unchanged by this session), and the candidate/selected-forward caches still
  hold ~570 MB combined at 12 MP per the unchanged `cand_bytes`/`sel_bytes`
  diagnostics. That's the next architectural unit of work per
  `jpegxl-rs.decision.encoder-architecture-phases`.
- **Triage** `both_oracles_decode_a_stream_from_the_hf_mul_segment`
  separately — it's a real decoder disagreement at `GlobalScale::MAX`,
  unrelated to CfL, and was failing before this session too.

---

## 2026-08-05 (wave 20c) — VarDCT policy wires inverse Gaborish

**Request knob.** `EncodeRequest::restoration` (`RestorationDecision`,
default all off). `epf_iters > 3` refused at plan time.

**Precondition in `plan_at`.** When `restoration.gaborish`,
`prepare_gaborish_frame` runs Jacobi inverse-Gaborish on XYB and all
DCT / cover / CfL / quantize paths use that frame. AQ analysis stays on
the source planes (intended post-J.3 appearance). Plan header carries the
request's restoration bits.

**Proved:** policy unit + roundtrip + oracle (`djxl` / `jxl-oxide`) agree
on a 64×64 gaborish stream within 1 code point; source RMSE stays in the
unfiltered 64×64 class. Filters still default off.

**Next:** cjxl density pin (optimization session first item); optional
deeper EPF inverse / sharpness / filter search; multi-section polish.

---

## 2026-08-05 (wave 20b) — multi-section modular + filter precondition API

**G.1.3 partition.** `ModularSource.nb_meta_channels` + `partition_channels`
(meta always; then channels ≤ `group_dim`; remainder → LF group if
`hshift≥3 && vshift≥3`, else pass group). Tracks shifts on squeeze.

**P2b palette multi-section.** LfGlobal residual-codes meta; pass groups
code index rects. 200×200 / `group_size_shift=0` sample-exact via
jpxl-decode.

**S3 squeeze multi-section.** Same partition; LF-group sections emit
(empty stream when no LF-shifted channels). 300×200 multi-group squeeze
sample-exact.

**F2 precondition API.** `gaborish::precondition_xyb_planes` for planners
before DCT when gab on. Header/write already signal filters; default off.

**Proved:** multi-section palette + squeeze unit/e2e; lib suite. Policy
wiring landed as wave 20c.

---

## 2026-08-05 (wave 20a) — palette, squeeze (single-section), filter write

See git history for 20a detail (palette/squeeze single-section + filter
header).

---

## 2026-08-05 (wave 19e) — deeper MA learning (slice 19 large, workstream B)

**General tree IR.** `MaTree` is now an owned full binary tree (`MaNode::{Decision, Leaf}`) with per-leaf Table H.3 predictors and BFS `ctx_id` assignment (H.4.2 leaf encounter order). Writers emit breadth-first; residual collection uses each leaf’s own predictor.

**Policy.** `plan_for`: (1) best single-leaf predictor by residual+tree bit cost; (2) greedy splits (property×threshold, parent predictor on both children) until no win; (3) per-leaf predictor refinement. Tree bits measured via `ma_tree_bit_cost` (no 64-bit fudge). Depth ≤ 4 / leaves ≤ 8 on frames ≤ 64×64; larger frames cap at one binary split so multi-group planning stays usable. Policy scoring sets `allow_lz77 = false`; final emission still adopts residual LZ77 when cheaper (nested compose).

**Proved:** full `jpxl-encode` suite green (roundtrip, oracle djxl/jxl-oxide, containers); unit tests for BFS ctx order, per-leaf predictors, `plan_for` bounds.

**Next:** palette / Squeeze; density pin vs `cjxl` (optimization session first item); slice 20 filters.

---

## 2026-08-05 (wave 19d) — encoder LZ77 emission (slice 19 large, workstream A)

**Phase 1 — `jpxl-entropy` LZ77 encode.** `Lz77EncodeParams` on
`EncoderPlan`; `write_bundle` emits Table C.1 + `lz_len_conf` (log alphabet
8); context map must already include the trailing distance context
(`identity_with_lz77`). `TokenCensus::record_token` / `record_copy` for
length triggers (bypass value hybrid-uint). `SymbolEncoder::push_copy`
emits length token + distance. Encode↔decode suite in
`tests/encode_lz77.rs`. Default remains LZ77 off.

**Phase 2 — modular residual LZ77.** Greedy match finder (lookback 256,
adaptive `min_symbol` just above max literal token so ANS alphabets stay
small). Exact-price adopt-if-cheaper vs plain ANS. **Trap:** H.3 sets
`dist_multiplier` to channel width — wire distances must use C.3.3
(`raw = distance + 119` when M > 0), not `distance - 1`. Verified with
`resolve_distance` invert test + three-decoder gates.

**Proved:** `cargo test -p jpxl-entropy -p jpxl-encode` green including
oracle `djxl` / `jxl-oxide`; repeating multi-value residual patterns
strictly smaller with LZ77; constant single-symbol residuals keep plain
ANS (adopt-if-cheaper).

**Next:** landed as wave 19e (deeper MA). Density pin vs `cjxl` stays first
item of optimization session.

---

## 2026-08-05 (wave 19) — encoder slice 19c: multi-leaf MA tree (small)

**Binary property split.** `MaTree::{SingleLeaf, BinarySplit}`: one
decision `property[k] > value` then two leaves (same Table H.3 predictor,
contexts 0/1). Residual ANS uses identity map over `num_contexts` leaves.
Tree emission follows H.4.2 breadth-first (decision, left leaf, right leaf)
with the existing six-context flat tree code.

**Policy.** After choosing the best single-leaf predictor, try a small grid
of splits (properties 4/5/6/7/9/10/11 × thresholds 0…64). Adopt only if
residual cost improves by ≥64 bits (covers unmeasured tree overhead).

**Proved:** full `jpxl-encode` suite green (roundtrip, oracle, containers);
unit test for left/right context assignment.

**Still large:** deeper / learned MA trees (wave 19d landed LZ77); palette /
Squeeze; density pin vs `cjxl -e N`. Slice 20 filters after unfiltered
baseline.

---

## 2026-08-05 (wave 18) — encoder slice 19a/b: lossless density (ANS + predictors)

**Decoder status (for orientation):** largely complete for the planned
reference path — corpus **30/39** green; remaining 9 are animation ×5
(deferred forever), J.2 chroma-subsampled `cafe`/`_5`, and two probed
failing cases (`lossless_pfm`, `grayscale_public_university`). Rare
VarDCT transforms lack pixel-comparison coverage but parse. Encoder
slices 11–18 + M8 leftovers done; this wave starts **slice 19**.

**Slice 19a — ANS residual coding.** Modular sample streams no longer use
flat prefix codes. Each residual section: census → best hybrid-uint among
a fixed candidate set → `EncoderPlan::identity` ANS → bundle + payload.
Tree stream stays the tiny flat `TREE_CODE` (one leaf). Multi-section
`LfGlobal` still header-only (empty residual ANS stream).

**Slice 19b — predictor selection.** `LosslessPlan` carries a Table H.3
predictor. `plan_for` scores Zero / West / North / Avg(W,N) / Select /
Gradient by exact residual-section bit cost and adopts the cheapest
(Gradient on ties). Select formula fixed to match decoder Table H.3
(`abs(N−NW) < abs(W−NW) ? W : N`). Weighted (6) deferred (H.5 state).

**Proved:** full `jpxl-encode` roundtrip + djxl + jxl-oxide sample-exact
on oracle ladder; constant 64×64 grey ≪ 256 B; smooth 128×128 ramp
≪ 800 B; noise 64×64 stays under 1 bpp (table overhead dominates).

**Still in slice 19:** multi-leaf MA trees, LZ77 emission (entropy encode
still writes `lz77.enabled = false`), optional global tree, palette /
Squeeze transforms, density-vs-`cjxl -e N` corpus pin. **Slice 20**
filters only after unfiltered R-D baseline is trusted.

---

## 2026-08-05 (wave 17) — M8 leftovers: census, AqMode default, quant_lf fill

**Census “unification” (settled, not merged).** `CensusSink` (raw
PackSigned per pre-context in `jpxl-encode`) and `TokenCensus`
(tokenized, in `jpxl-entropy`) are **two stages on purpose**: policy
trains before hybrid-uint configs exist; ANS tables need tokens after
configs exist. Documented in `vardct/sink.rs`. Deleted the unused
placeholder `SymbolSink` trait (emission uses `SymbolEncoder` via an
`HfEventSink` adapter).

**`AqMode::Masking` is the production default.** Flat frames still
collapse to a neutral field (byte-identical to Off). Explicit `Off`
remains for baselines. Hierarchical cover was already the default
cover.

**`quant_lf` secondary fill.** After the global_scale/HfMul ladder
settles, if undershoot exceeds tolerance and budget remains, a discrete
probe over legal `quant_lf` values at the winning rung spends LF-dominated
slack without reopening R-D (wave 13 LF cliffs). Controlled by
`RateSearchBudget::lf_fill_probes` (default 8; set 0 to hold the
request's ratio fixed). `RatePhase::LfFill` appears in the trace.
Rate-loop iteration tripwire raised 20→32 to cover fill probes
(measured ≤27).

**M8 complete** for the encoder entropy/R-D track except rectangles
(separate evidence track). Next: slice 19 lossless density, or slice 20
filters only after an unfiltered R-D baseline is trusted.

---

## 2026-08-05 (wave 16) — Windows oracles + encoder slice 18d: HF presets

**Oracle rebuild (Windows).** The vendored `tools/oracle-bin/{djxl,cjxl,jxlinfo}`
were Linux ELFs (os error 193 on spawn). Rebuilt libjxl with MinGW
(`libjxl/build-win`, Ninja, static libjxl) and installed PE
`djxl.exe`/`cjxl.exe`/`jxlinfo.exe` plus MinGW runtime DLLs beside them.
`jxl-oxide-cli 0.12.6` installed to `~/.cargo/bin`. Discovery now prefers
`.exe`, rejects ELF magic on Windows, and resolves `USERPROFILE/.cargo/bin`.
`tools/setup-oracles.ps1` is the Windows counterpart of `setup-oracles.sh`.
Provenance: `PINNED_REVISIONS.txt` (libjxl `196a43d9`, GNU 16.1.0).
Full `vardct_oracle` suite: **9/9 green** including both external oracles.

**Slice 18d: multi HF preset.** Writer: removed `num_hf_presets != 1`
refusal; `write_pass_group` emits I.4 `hfp`; walk offset =
`495 · nb_block_ctx · hfp`. Policy: per-group absolute HF mass fingerprint,
median split into two presets when mass differs and `num_groups ≥ 2`;
re-census + retrain + exact-price adopt. Measured on 512×512 half-flat /
half-noise: **`num_hf_presets=2`, assignment `[0,1,0,1]`** (checkerboard
of groups), self-decode + both oracles accept.

**Still queued from M8:** census-type unification, `AqMode` default,
`quant_lf` fill. Next: those polish items or slice 19 lossless density.

---

## 2026-08-05 (wave 15) — encoder slice 18c: I.2.2 custom block context

**Slice 18c: trained HF block context.** Writer emits the full I.2.2
non-default path (LF/QF thresholds + C.2.2 `block_ctx_map` via the
existing `ContextMap` encoder); `check_supported` no longer refuses
`HfBlockContextPlan::Custom`. Policy proposes a candidate and adopts
only on a strict exact `price_codestream` win (one re-census + retrain,
same discipline as 18b orders).

**Winning lever today: shape-class trim (no thresholds).** A frame that
only uses a few Order IDs (FixedDct8x8 = shape 0) collapses unused
default-map rows onto context 0 and densifies; `nb_block_ctx` drops
(measured 15 → 2) and I.4's pre-context count shrinks with it. Density
on half-noise 256×256 FixedDct8x8: **10038 → 9995 B (−0.4%)**, pixels
unchanged, our decoder accepts the custom map. QF-threshold candidates
are still proposed under varying `HfMul` but currently lose the price
gate on the pinned fixtures (map cost > HF savings) — kept as a
candidate path, not deleted.

**Proved:** hand-built custom I.2.2 fragment round-trips through
`jpxl-decode::read_hf_block_context` (thresholds + map bit-exact);
full-frame custom map decodes under jpxl-decode; adopt-if-cheaper never
regresses size.

**Still queued from M8:** HF presets, census-type unification
(`CensusSink` vs `TokenCensus`), `AqMode` default (still Off),
`quant_lf` fill knob for LF-dominated rate targets. LF-threshold
proposals deferred (did not amortise). Next encoder: 18d presets or
slice 19 lossless density.

**Env note:** `djxl` oracle on this host is a non-Win32 binary (os error
193); external-oracle parity for the new map was not re-run here —
self-decoder + prior ladder remain the gate.

---

## 2026-08-04 (wave 14) — encoder slice 18b: custom orders, Hierarchical default

**Slice 18b (d0b6581): F.3.2 writer + §9.4 optimizer + the real
refinement pass.** `write_hf_coeff_orders` mirrors the decoder's
shared-stream structure exactly (one C.1 stream, eight distributions,
`end` + Lehmer per `(Order ID, channel)`, LLF prefix never permuted —
refused, not repaired). Policy proposes orders from per-position
nonzero frequencies in the quantized IR, re-censuses under them
(contexts depend on order position), retrains clusters, and adopts
only on a strict exact-price win. Density: −1.8% to −9.1% on top of
wave 13. The permutation composition direction — the documented
two-readings trap in `jpxl-decode/vardct/order.rs` — now has
external-decoder parity evidence, not just internal consistency.

**`CoverMode::Hierarchical` is the production default.** Consequence
worth knowing: the 2100×24 roundtrip rung's peak error moved 42→89
(bound restated 70→96, RMSE unchanged inside bound) because a merge
straddles that fixture's hard vertical edge and rings — §4.3's J has
no edge term. Slice 20's perceptual work owns edge-aware splitting;
do not "fix" this by hand-tuning NON_DCT8X8_SIGNAL_BITS.

**Still queued from M8 (updated wave 15):** HF presets, census-type
unification, `AqMode` default, `quant_lf` fill. Block context: done.

---

## 2026-08-04 (wave 13) — encoder slice 18: trained entropy model

**Slice 18 (0b493cd): §9.3 clustering + per-cluster hybrid-uint.**
The fixed six-cluster `cluster_of` map and frame-wide (4,2,0) config
are replaced by a census-trained model: agglomerative merge with
exact token-Shannon data cost (jpxl-entropy is now a policy
dependency for the C.2.3 tokenize arithmetic) + signaling estimates,
fingerprint-windowed candidates, generation-stamped queue, 255-cap.
Pure-entropy density win, identical pixels: −9% to −20% across the
fixture set (flat 2329→1984, half-noise 12408→10221, masking/hier
6161→5605). §9's refinement bound is **zero** at this scope — the
event stream doesn't depend on anything the trainer chooses — and
the clusterer terminates in ≤ n−1 merges.

**New trap: LF dither cliffs in the rate ladder.** With HF this
compressed, the LF modular section dominates coarse targets; a
quantized ramp crossing a dither threshold moves LfGroup ~1.4 kB
between *adjacent* gs rungs. Do not "fix" a missed undershoot bound
by loosening the loop: the tests now accept either tolerance or the
trace-proven cliff (every infeasible price overshoots the target).
Real fix queued: `quant_lf` as a secondary fill knob at LF-dominated
rates (it IS a rate knob there — the wave-10 "distortion knob"
finding predates HF being this small).

**Still queued from M8:** custom coefficient orders (needs the F.3.2
permutation writer), trained block context (needs its writer), HF
presets, census-type unification (`CensusSink` raw vs `TokenCensus`),
and the production default flip (Hierarchical + Masking) — the
BlockInfo DctSelect/mul rows are modular-coded, outside the HF model,
so their pricing is still the untrained single-context tree.

---

## 2026-08-04 (wave 12) — encoder slice 17: adaptive quantization

**Slice 17 (5f2c8be): perceptual field + `HfMul` factorization.**
x265-family variance AQ (broad strokes from the bpg-rs/still265 AQ
research digest, reimplemented for this wire): per-atom octave
adjustments from log2-variance deviation, half-octave lattice,
`Masking` (perceptual) and `Uniform` (error-equalizing) directions.
**Trap-grade wire fact: `HfMul` divides the step like `global_scale`
does (I.2.1), so larger `HfMul` = finer.** The first implementation
assumed the opposite and doubled gs+mul together (denominator ×4 =
two octaves finer everywhere); the correct exact factorization is
halve `global_scale` / double `quant_lf` / double baseline `HfMul` —
both products preserved, LF integers proven unchanged. Odd gs snaps
to the even family (no parity sawtooth in the rate ladder). Neutral
fields collapse to the plain wire (flat content byte-identical);
a non-zero constant mul row costs ~3 bits/block because the gradient
predictor sees the zero DctSelect row above it — slice 18's trained
tree should fix that pricing. Uniformity exit: flat/busy RMSE gap
4.36 → 2.97 (`Uniform`); `Masking` saves 21–32% while refining flat
regions. Rate contract with AQ: never-over unchanged; undershoot
stated at 2% — at Uniform/4000 the loop's 3955 IS the brute-force
optimum over every integer gs in the bracket. Varying mul row
decoded by both external oracles within 1 code point (new wire
content). **Open decision for slice 18:** defaults are still
`FixedDct8x8` + `AqMode::Off`; flip production to Hierarchical +
Masking after the entropy model prices DctSelect/mul rows properly
and a corpus-level sweep exists.

---

## 2026-08-04 (wave 11) — encoder slices 15–16: CfL and the block selector

**Slice 15 (d7b9b59): CfL estimation.** Real I.6 factors: frame-wide
LF pair (I.2.3) + per-64×64-tile HF pair (G.2.4), least-squares seed
refined over nearby wire integers by scoring residuals through the
exact quantizers (incl. I.5.3 quant_bias). Adopt-only-if-it-nets-a-
saving objective; grayscale is a hard no-op (byte-identity tested).
Boundary probe: HF factors interoperate at `[-128, 127]` — `-129`/
`128` decode inconsistently across oracles; writer rejects outside
the range (`docs/experiments/2026-08-04-i6-cfl-sign-and-wire-range`).

**Slice 16 (b82e7bc): hierarchical block selector, squares only.**
Estimation/quantization core is transform-generic and varblock-list-
driven; `CoverMode::Hierarchical` runs a quadtree per aligned 32×32-
atom region over DCT8x8/16x16/32x32 with §4.3's `J = R + λ·D +
metadata_bits`. Non-obvious facts, each paid for in debugging: the
forward transforms are **not Parseval** — one squared coefficient
unit is `side²` squared sample units, so cross-transform distortion
must be sample-domain or merges buy invisible quality; λ per channel
from the DCT8x8 operating point (1 bit ⇌ s²/16); a non-DCT8x8
varblock costs ~8 real bits of DctSelect signal under the current
single-context modular coder (charged 32, re-derive in slice 18).
CfL HF refinement folds larger-transform cells onto the 8×8 grid
(identity for 8×8; no non-LLF cell folds to DC). Writer admits the
square vocabulary only; rectangles/special transforms still refused
(no parity evidence). **Default cover mode stays FixedDct8x8** —
flipping production to Hierarchical is a slice-17 decision, with
adaptive quant in hand. Exit evidence in the lib tests: exact cover
in strict BlockInfo raster order on a clipped multi-group frame;
gradient at defaults (3303 B / RMSE 0.379) strictly dominates fixed
at matched-quality gs=45000 (3321 B / RMSE 0.462, curve flattens at
~0.459); both external oracles decode a merged-transform stream
within 1 code point. A perfect pixel checkerboard is a *single* DCT
basis function at every size — useless as a "detail" fixture; the
detail test uses hash noise.

**Scope note:** PLAN's slice-16 row says "DCT8/16/32 + common
rectangles"; rectangles were deliberately deferred — squares are
unambiguous in the LLF mapping (no orientation choice) and already
earn the exit gate. Rectangles join a later milestone with their own
oracle evidence.

---

## 2026-08-04 (wave 10) — encoder slices 12–14: lossy VarDCT encoding is real

**Slice 13 (768f1fb):** `jpxl_core::forward` — allocation-free forward
forms for all 27 transform types + `lf_from_llf`, exact inverses of
the in-tree I.7.2 kernels (worst pair error 5.4e-7, flat across
sizes). Proofs: pair tests both directions, full impulse sweep
against the PROVEN inverse (kills shared 1/√s errors the roundtrip
alone hides), AFV corner discriminators separating all four variants
(the composition test alone was measured blind to flip_x — the
wave-8 involution lesson recurring). No new flip points: forward
algebra inherits every decision by reference (shared `scale_f`,
shared half-index constant).

**Slice 12 (d5937d4): all three decoders accept our lossy output.**
Full vertical slice: policy plan → validated plan → header/LfGlobal/
LfGroup/HfGlobal/PassGroup writers → ANS. Ladder 8×8 grey →
multi-section 300×260 → non-multiple-of-8 → two LF groups; djxl and
jxl-oxide agree with jpxl-decode within 1 8-bit code point on every
rung. Non-obvious clause facts: kVarDCT frames signal NO
group_size_shift (group_dim fixed 256); I.2.3 `base_correlation_b`
defaults to 1.0 so "neutral" CfL still adds Y into B — encoder
quantizes Y first, targets `B − 1.0·dY_recon` (flat grey → peak 0).
Gated mechanical move: I.2.4/I.2.5 dequant tables →
`jpxl-core/src/dequant.rs` verbatim (flip points intact, decoder
suite green = proof). Policy boundary test rescoped to the real
`[dependencies]` table; peer oracle allowed as dev-dependency.

**Slice 14 (ff6c84d): exact rate control.** Pricing IS the writer
(`emit_codestream` → bytes + sizing; a parallel size model is
rejected in-doc as a paired-bug shape); sizing partitions the stream
byte-exactly. Rate loop over wire-representable rungs (global_scale
ladder extended by HfMul>1 past the I.2.1 ceiling): bracket →
bisect → discrete fill probing past infeasible notches. Contract:
never over target, ≤1% undershoot; measured worst 0.57% at ≤16
prices. Size is genuinely non-monotone in global_scale (16
decreases across 60 consecutive scales — regression-tested);
injected-sawtooth test proves the loop finds the brute-force
optimum. `quant_lf` measured to be a distortion knob, not a rate
knob — held fixed, slice 17 owns it. First HfMul>1 stream on the
wire; oracles accept.

**R-D baseline for slices 15–18 to beat:** 300×260 mixed content:
ours 10748 B / RMSE 6.46 (default quantizer) vs `cjxl -d 1` 4165 B /
2.73. Gap drivers, in expected order: no CfL (15), fixed 8×8 (16),
constant HfMul (17), untrained entropy (18).

**Known debt, scheduled:** two census types
(`vardct::sink::CensusSink` vs `jpxl_entropy::encode::TokenCensus`)
and the placeholder `SymbolSink` trait — unify/delete in slice 18;
`x_qm_scale`/`b_qm_scale` written neutral 2 — slice 17; tolerance
stop can settle one rung coarse at equal bytes in size-flat regions
— milestone-7 R-D question. No jpxl-cli VarDCT wiring yet.

**Next:** slice 15 (CfL estimation), 16 (hierarchical block
selector), 17 (adaptive quant) per PLAN. Decoder side unchanged this
wave (corpus stays 30/39; `cafe` via J.2 and the two probed-failing
cases remain the decoder candidates).

---

## 2026-08-04 (wave 9) — noise + kModular gaps + YCbCr; corpus 30/39; encoder slices 11 + 11.5 landed

**Decoder: corpus 18 → 30 of 39.** Three tracks, in order:
noise (K.5 — LUT parse in LfGlobal, XorShift128Plus/SplitMix64,
zero-sum Laplacian, applied between patches and Annex L;
`noise`/`noise_5` peak 4.9e-5); kModular displayed-frame gaps
(`bicycles` xyb_encoded kModular and `patches_lossless` via
`modular_displayed_pipeline`, reusing the existing
ColourPlanes/patches/Annex-L machinery); do_YCbCr with
`jpeg_upsampling == [0,0,0]` (L.3 linear formula, (Cb,Y,Cr) order;
`bench_oriented_brg`/`_5` + `grayscale_jpeg`/`_5`, peaks 1.9e-6).
YCbCr wiring finally exercised the I.2.4 RAW dequant-matrix path —
`RAW_MATRIX_CHANNEL_ORDER_IS_XYB` and `RAW_SUBBITSTREAM_IS_UNALIGNED`
are now PROBED-CONFIRMED true (addendum in the vardct flip-point
probe entry). **Bonus: four cases were passing all along, ungraded**
— `delta_palette`, `lz77_flower`, `opsin_inverse`/`_5`
(`e2e_previously_unattempted_corpus.rs`).

**Noise PRNG pseudocode `*` is a `^` OCR misread** (same systemic
class as the documented H.5.2 bug), triangulated by exact structural
match to Vigna's public-domain XorShift128+/SplitMix64 plus
pixel-exact corpus behaviour — NOT claimed as a standard defect, so
no scan read was required; entry:
`docs/experiments/2026-08-04-noise-xorshift-ocr-reading.md`. If
anyone later suspects the printed text itself, read the scan first.

**Remaining 9:** animation ×5 (user decision 2026-08-04: **deferred
forever**); `cafe`/`_5` (needs J.2 chroma-subsampled group grid);
`lossless_pfm` (peak 1.75 vs 0.0 — probed, failing, unexplored);
`grayscale_public_university` (peak 0.27 vs 9.8e-4 — probed, failing,
unexplored). jbrd/JPEG reconstruction: user decision — not wanted.

**Encoder slices 11 + 11.5 landed** (commits 1244a4d, 4b7628b):
`jpxl-encode-policy` split with manifest-tested one-way boundary;
VarDCT plan IR + validate() (41 typed-rejection tests, exact-cover /
LF- and pass-group containment / context-map density / F.3.1 layout);
ANS encoder in jpxl-entropy (backward rANS with C.3.2 terminal state,
all four C.2.5 histogram forms, hybrid-uint, context maps ±MTF, full
prefix path; LZ77 emission deferred to slice 19). Exit gate: seeded
roundtrip matrix through our proven decoder, exact bit consumption,
24/24 payload corruptions caught. Two derived constraints, enforced
and tested: an ANS `log_alphabet_size` must exceed a cluster's
`split_exponent` by one when the config has in-token bits (C.2.3
stops reading at equality); probability exactly 4096 collides with
C.2.5's `logcounts == 13` run-length escape — full mass must use the
one-symbol form. Deliberate deviations from Encoder-plan1.md recorded
in-source: PreContextId is u32 (I.3.3 exceeds u16 at nine presets);
context map lives per-pass not per-preset (I.4 indexing); added
pass-group containment (encoder-only strictness).

**Next candidates (decoder):** `cafe` via J.2; the two probed-failing
cases (`lossless_pfm` first — a lossless miss of peak 1.75 smells
like a wholesale misinterpretation, likely cheap to localise);
grey-Y 2.3e-4 residual; rare-transform pixel coverage. **Next
(encoder):** slice 12, the fixed-DCT8×8 vertical slice integrating
11 + 11.5 (exit: JPXL + djxl + jxl-oxide all decode).

---

## 2026-08-04 (wave 8) — cropped frames + orientation + kBlack; corpus 18/39; encoder phase planned

**`spot`/`cmyk_layers`/`sunset_logo` pass** (peaks 6e-8/1.2e-7/4.8e-7,
orders inside thresholds). All three were gated on ONE refusal: F.2
cropped frames. Built: `composite_frame` placing frames at (x0,y0) with
i64 `CropRect::intersect` (negative UnpackSigned origins, any-edge
overhang), lifted for ALL encodings (compositing is display-space,
encoding-agnostic); Table D.4 orientation — all 8 rows, derived by
inverting the first-row/first-column pair, cross-checked against the
prose, applied to integer AND float planes at the very end (D.3.2;
every in-codestream dimension is pre-orientation). **kBlack needed NO
code**: 18181-3 §4.1.2 grades every extra channel as itself in ec_info
order — CMYK conversion would FAIL shape condition 1. `bench_oriented_brg`
is NOT unlocked by orientation despite the name: its gate is do_YCbCr.
Orientation was previously silently IGNORED (not refused) — sunset_logo's
shape assertion is what caught it. Fixtures 100–109 (hand-built eXIf
orientation tags, jxlinfo-verified by the script; cjxl cannot emit
cropped non-animation frames, so crop evidence is the corpus streams +
unit tests). Mutation-tested: anti-transpose↔transpose killed by the
corner test (both are involutions — the inverse-composition test alone
cannot kill it).

**Flip point, recorded unexercised:** `CROP_LEAVES_RUNNING_IMAGE_OUTSIDE
= true` — what "the image" holds outside a cropped frame's rectangle
(F.2 names two buffers and never says). Proven undiscriminated: all
three corpus cases store to slot 1 and read source==1; both arms
implemented.

**Corpus 18/39.** Remaining gates, enumerated across all 39: animation
×5, do_YCbCr ×6 (incl. bench_oriented_brg ×2), noise ×2, `bicycles`
(xyb modular displayed frame), `patches_lossless` (patches in kModular
+ stored frame in non-XYB). Gate: 1002 tests, 0 failed, zero ignores.

**Encoder phase adopted into PLAN.md (slices 11–20)** from external
advisor doc `docs/Encoder-plan1.md`, with recorded adjustments: ANS
encoder is its own slice 11.5; lossless density (MA trees/LZ77)
interleaves as slice 19; the inverse-primitives-into-core refactor is a
gated mechanical slice; SIMD/threading stay out until a scalar R-D
baseline. DO NOT START until the user triggers it (their session-limit
budgeting).

**Next candidates (decoder):** noise (Annex K.4 — synthesis, likely
self-contained); `bicycles`/`patches_lossless` (kModular displayed-frame
gaps); YCbCr + jbrd (big, unlocks 6+); animation (scope decision);
rare-transform coverage (Hornuss/DCT4x4/≥DCT128 still no pixel proof);
the grey-Y 2.3e-4 residual family.

---

## 2026-08-04 (wave 7) — K.2 upsampling; THIRD scan-verified Part 1 defect (I.2.4 AFV transpose); corpus 15/39

**`upsampling`/`upsampling_5` pass** (peak 4.3e-5 / 1.6e-2 vs 0.004 /
0.06). Frame upsampling is **K.2**, not J.2 — J.2's triangle filter is
only for `jpeg_upsampling` (still refused). Built `frame/upsampling.rs`:
K.2 index formula, default tables (validated WITHOUT a scan read: all 84
positions' 25 weights sum to 1 within 3e-8 — a digit slip cannot
survive), 5×5 mirrored window with per-output [min,max] clamp, top-left
crop, L.4's 8×-then-f/8 split, D.3 custom weights (unit-tested; no
stream exercises `cw_mask != 0`). Pipeline order per K.1: Annex J at the
STORED frame size → K.2 → patches → Annex L. Groups/modular channels
stay on the downsampled frame grid. Flip settled by fixture:
`EC_DIMS_INCLUDE_EC_UPSAMPLING=true` (F.2 cumulative; false desyncs
fixture 94's modular stream). Fixtures 90–95 + `e2e_upsampling.rs`.

**I.2.4 AFV weight placement is a PUBLISHED DEFECT (scan-verified,
printed p.59): the text writes `weights(2*y, 2*x)` for `freqs[y*4+x]`,
the transpose of the coefficient's actual position (I.9.8 puts basis
`y*4+x` at column 2x, row 2y; the freqs table's four zero entries match
the four skipped positions).** Shipped transposed as
`AFV_FREQ_POSITION_IS_TRANSPOSED=true` with a directional unit test.
Localisation was the proof: under the literal reading the four worst
tiles on `upsampling` were AFV0–3 varblocks holding ~100% of the squared
error while 94 DCT4x8/DCT8x4 blocks sharing the same IDCT were clean;
transposing drops the corpus case 7.8e-2 → 4.3e-5 and a no-resampling
control 8.7e-3 → 3.8e-5. Scan-verified Part 1 defect tally: I.8 ScaleF,
Table I.6 index 16 (candidate), I.2.4 AFV — first AFV pixel coverage,
closing part of the rare-transform gap.

**Still open:** `bike` 2.48e-4 / `grayscale` 2.28e-4 residual family is
NOT AFV (unchanged by the fix). Unexercised: custom upsampling weights,
factors >8 beyond unit tests, `dim_shift > 0`, upsampling in kModular
(typed refusal), J.2 chroma upsampling.

**Next candidates:** cropped/oriented displayed frames + kBlack
(unlocks spot/cmyk_layers/sunset_logo); animation (scope decision);
Hornuss/DCT4x4/≥DCT128 pixel coverage; the 2.3e-4 grey-Y residual;
jxli/jbrd (scan first); Brotli decision.

---

## 2026-08-04 (wave 6) — extra channels + alpha + frame blending; corpus 6 → 13 cases green

**Seven more corpus cases pass their test.json thresholds:**
`alpha_nonpremultiplied`/`alpha_triangles` (needed only 4-channel
grading), `alpha_premultiplied` (extra channels in kVarDCT),
`patches`/`patches_5` (K.3.2 per-channel-group alpha patch blending),
`blendmodes`/`blendmodes_5` (multi-frame F.2 compositing, all five
Table F.8 modes). Corpus total: 13 of 39.

**Built:** G.1.3/G.2.3/G.4.2 extra-channel modular streams in kVarDCT
frames (selection/copy-back FACTORED into shared
`lf_group_selection`/`pass_group_selection`/`decode_group_channels` —
do not re-duplicate); `render::ExtraPlanes` appended to DecodedImage at
each channel's own ec bit depth; K.3.2 alpha in `apply_patches` (K.3.2's
`c` ranges over [0, num_extra], honours clamp); `frame/blending.rs`
`Canvas` + `blend_sample` + `composite_frame` with per-channel-group
source slots. Pre-CT reference slots (XYB, K.3) and post-CT canvases
(display space, F.2) are SEPARATE — F.2 blends after Annex L. An identity
first frame returns its own integer planes, so lossless stays bit-exact.
Animation (a presented frame with duration) is now an EXPLICIT refusal —
the old "more than one regular frame" guard no longer covers it.

**Flip points:** none settled — corpus can't discriminate
(all reference frames at origin; every patch case has exactly one extra
channel). Two NEW recorded as unexercised:
`PATCH_ALPHA_IS_THE_PATCHS_OWN`, `ALPHA_SELF_RULE_IS_THE_NAMED_CHANNEL`
(readings coincide for single-alpha images — every corpus stream).
Addendum in 2026-08-03-patches-k3.md.

**Known residual (pre-existing, not this wave):** greyscale-VarDCT Y
carries ~1.9e-4 RMSE vs oracle (fixture 84 discriminator: identical grey
source with NO alpha reproduces it to 4 s.f.) — same family as corpus
`grayscale`'s 2.3e-4 peak. A future hunt should start from grey-only
VarDCT, not alpha.

**Out of scope, enumerated:** upsampling ×2 (J.2 + K.2 ec_upsampling=4),
`spot`/`cmyk_layers`/`sunset_logo` (cropped displayed frames +
orientation + kBlack), `patches_lossless` (patches in kModular +
stored frame in non-XYB), animation ×5, `noise`, `cafe`,
`bench_oriented_brg`, `grayscale_jpeg`, `bicycles`.

**Next candidates:** J.2/K.2 upsampling (unlocks 2 cases); cropped
displayed frames + orientation (unlocks 3, incl. spot/cmyk kBlack);
animation compositing (5 cases — needs a decision on presentation
semantics, PLAN lists it out of scope); rare-transform pixel coverage;
the grey-Y 1.9e-4 residual; jxli/jbrd (scan first); Brotli decision.

---

## 2026-08-04 (wave 5) — bike + progressive corpus PASS; six corpus cases green, zero ignores

**bike divergence killed, two real bugs.** (1) Transfer functions below
zero: BT.709 evaluates its piecewise condition on the SIGNED value (the
linear toe `4.5·v`), sRGB extends with odd symmetry — measured both ways
(bike spikes fit slope 1.000/const 4.4993; an out-of-gamut sRGB fixture
matches odd at 8.1e-6 and misses literal by 0.11). Neither the standard
nor IEC/ITU define the negative domain; pinned as flip point
`NEGATIVES_TAKE_THE_LINEAR_SEGMENT = [false, true]` ([sRGB, 709]) with
directional tests + fixture 63. (2) I.5.2 adaptive LF smoothing is
FRAME-WIDE ("each LF sample of the image"), not per LF group — per-group
loops skipped every group's edge rows, leaving a 16-row band at bike's
only internal LF-group seam (y=2048). Fixed by assembling the frame-wide
LF image (dequant and LF CfL commute with assembly — LF CfL uses the
frame-wide I.2.3 factors, not per-tile). Fixture 64 (128×2176, `-d 6
-e 3` load-bearing: `-d 1` sets kSkipAdaptiveLFSmoothing) is the only
multi-LF-group fixture in the tree — the standing seam regression. bike
0.2466 → 2.5e-4. Eliminated for bike: per-tile B CfL, I.5.3 B terms.
Rare transforms (Hornuss/AFV/DCT4x4/≥DCT128) still have zero pixel
coverage.

**Progressive corpus done.** `progressive`(_5 is a symlink to it) = patch
atlas + Squeezed kModular kLFFrame (lf_level 1) + 2-pass kVarDCT with
kUseLfFrame. Built: LfFrame slots + L.2.2 kModular pre-step shared via
`xyb_from_modular` (flip `LF_FRAME_IS_XYB_PRESTEP=true`: no Quantizer in
an LF frame's LfGlobal, so raw integers have no shared scale; false gives
peak 9e10); kUseLfFrame skips ALL of G.2.2/I.5.2 incl. smoothing (the
corpus frame has 4 LF groups + smoothing bit clear and still grades
2.0e-5 — independent confirmation). Multi-pass needed NO new code, only
reachability. Flips settled: `PREV_USES_CURRENT_PASS_COEFFICIENT=true`
(five multi-pass streams: true → exact TOC exhaustion + C.3.2 terminal
states; false → entropy desync inside I.4; single-pass control byte-
identical); `G42_SIZE_TEST_IS_SHIFTED=true` (unshifted leaves channels
37/38/41/42 of a Squeeze pyramid decoded by NO rule; false runs off the
section end at exactly its TOC length). Ladder fixtures 70–74 +
`e2e_progressive.rs`.

**State:** six corpus cases pass their test.json thresholds (grayscale,
grayscale_5, bike, bike_5, progressive, progressive_5); ZERO `#[ignore]`
in jpxl-decode tests; 945 tests green. jxlinfo now built/installed by
setup-oracles.sh. Note: bike rungs cost ~78 s debug (6 s release); the
progressive corpus rung is release-always but debug-opt-in via
`JPXL_SLOW_TESTS=1`.

**Traps:** the seam bug is invisible on every ≤1-LF-group image — do not
"optimize" smoothing back into the per-group loop; fixture 64's rung is
the only thing that would catch it. `smooth_lf_image` must stay gated on
`!use_lf_frame`.

**Next candidates:** extra channels in kVarDCT + alpha blend rows
(Table K.1) — the alpha corpus cases; rare-transform pixel coverage;
`lf_level > 1` / chained LF frames (nothing exercises them); jxli/jbrd
parsing (scan first); Brotli decision; open flips still without streams:
PATCH_REFERENCE_IS_CANVAS_COORDINATES, ALPHA_GUARD_COUNTS_EXTRA_CHANNELS,
`num_hf_presets > 1`, nonzero `lf_idx`.

---

## 2026-08-03 (wave 4) — THE MODULAR BUG IS DEAD: `^` misread as `*`; slice 9; patches; RAW fixed

**Root cause of the project's oldest open bug — one character.** The original
image scan (printed p.50) shows H.5.2's clamp guard as
`((true_err_N ^ true_err_W) | (true_err_N ^ true_err_NW)) <= 0` — XOR,
sign-bit arithmetic matching its own comment. ALL THREE transcriptions
misread `^` as `*`; they descend from one scan and share one glyph
confusion, so their agreement corroborated nothing. **Both previously
claimed H.5.2 "standard defects" are WITHDRAWN** — the published standard
is correct; `weighted.rs` now has one symmetric clamp behind the literal
guard (`EXPERIMENT_CLAMP_SYMMETRIC` and the asymmetric complex deleted).
Fixed at once: sawtooth (60), LfQuant repros (61/62), fixtures 54/57 (8C
gates un-ignored, green), the silent-corruption case, and the corpus
blockage. All 11 lossless fixtures stay bit-exact; e2e_lossless 19/19,
zero ignores. Scan-verified defect tally now: I.8 ScaleF divide-by-zero
(confirmed at scan) and Table I.6 index-16 bases (candidate, scan-read).

**Slice 9 (container) done.** `BoxTree::parse` (clause-8 framing) separate
from `validate` (clause-9 shalls); order-VALIDATING jxlp reassembly
(sorting hid corruption), proven vs independent jxlc reference and both
external decoders; `jpxl boxes` subcommand; jxlinfo as box oracle (build
via `cmake --build libjxl/build --target jxlinfo`, not in setup script
yet). brob stays compressed (typed Unsupported; Brotli dependency is a
PLAN decision); jbrd unparsed pending scan cross-check. Trap: a final box
whose declared length overruns the file is REJECTED; jxlinfo lists
nonexistent bytes — do not loosen to match. Container errors ride
`JpxlError::InvalidHeader` because `FieldOutOfRange` hard-codes 18181-1.

**Patches (K.3) done.** Dictionary is the FIRST row of Table G.1 (proven
by exact LfGlobal exhaustion on bike_5, 12293/12296 bits); rendering on
XYB planes between Annex J and Annex L; kReferenceOnly kModular reference
frames in four slots via L.2.2's kModular pre-step. Traps: L.2.2 kModular
channel order is y',x',B' (luma first — XYB reading swaps patch chroma);
a missing LfGlobal bundle reports itself hundreds of bytes downstream
under an unrelated error — measure section exhaustion to localise.

**RAW dequant matrices decode (Trap removed)**: I.2.4 reads the 3-channel
modular sub-bitstream INLINE; HfGlobal is one section, H.4.1's formula is
a stream index. `read_*_with(…, Option<&RawMatrixContext>, …)`;
context-less callers refuse at the sub-bitstream's first bit. `PREV` flip
point's false arm now really consults the accumulator (do not "simplify"
`next_prev` back — the point is the `ucoeff=0, accumulated≠0` case).

**Open — one B-channel divergence on corpus bike/bike_5** (now decoding
END TO END): peak 0.2466 on B only, X/Y ≈0.017, RMSEs near-passing.
`#[ignore]`d with forensics in e2e_vardct.rs. Probe already done:
`DCT8X4_HALF_INDEX_IS_LOW_COORDINATE` is CONFIRMED and now DISCRIMINATED
(flipping fails all channels at 1.1 — bike contains DCT8x4). Suspects:
B-channel dequant of a rare transform (Hornuss/AFV/DCT4x4 have zero pixel
coverage), per-tile B CfL, B-specific I.5.3 terms.

**METHODOLOGICAL TRAP (permanent):** when every available transcription
agrees on something that reads as nonsense, that is evidence about the
transcription pipeline, not the standard. The AGENTS.md §2 chain works
only if followed TO THE IMAGE SCAN; three sessions modelled the nonsense
instead. Any "defect in the standard" claim requires a scan read first.

**Next candidates:** the bike B-channel divergence; kLFFrame + multi-pass
(progressive corpus cases); extra channels in kVarDCT + alpha patch blends
(alpha corpus cases); jxli/jbrd parsing (scan first); Brotli decision.

---

## 2026-08-03 (VarDCT wave 3) — 8F assembly: VarDCT DECODES END TO END; slice 8 core complete

**Every acceptance rung passes, 2–3 orders of magnitude inside its class:**
fixture 04 first-light peak 7e-6; filters-off 50/51/55 ≤5e-5 (class 0.004 /
1e-5); filters-on 52/53/56 ≤5.4e-5 (class 0.06 / 0.02); **conformance corpus
`grayscale`/`grayscale_5` at 2.3e-4 against PUBLISHED references** — true
standard conformance, not libjxl agreement. `bike_5` skips (needs patches,
K.3, out of slice scope). `DecodedImage` gains `float_planes` (unclipped
f32, the §4.2 surface); integer planes quantize at the edge for VarDCT.
Output colour space is the SIGNALLED encoding, except under `want_icc`
where output stays linear (clause 4 — worth 0.287 → 0.001 peak on corpus
`grayscale`).

**Flip-point probe pass: 9 tested, 1 REVERSED** —
`EPF_SKIP_IS_PER_VARBLOCK = false` (per 8×8 block; the literal per-varblock
reading cost three orders of magnitude and masked the other EPF probes).
Confirmed load-bearing by flip: LLF ScaleF argument, I.3.1 order-table
direction, LfQuant Y,X,B, three EPF readings. Still open for want of
streams: `DCT8X4_HALF_INDEX_IS_LOW_COORDINATE`,
`PREV_USES_CURRENT_PASS_COEFFICIENT` (whose false-arm is also degenerate —
8C defect, needs a real accumulator consult before it can be tested).
See `docs/experiments/2026-08-03-vardct-flip-point-probe.md`.

**Traps:**
- RAW dequant matrices are REFUSED (`Unsupported`), not missing: I.2.4
  reads the 3-channel matrix inline mid-bitstream; 8B's
  `raw_requests()`/`set_raw_matrix()` split cannot express that. Fixing it
  means restructuring `read_dequant_matrices` to take the modular decoder.
- Multi-section VarDCT is PROVEN working (264×100 / 100×264 at 2e-6);
  anything failing inside G.2.2 LfQuant is the modular bug below, upstream
  of VarDCT.

**Modular H.5 bug — sharpened, not fixed (deliberately).** A clamp
lower-gate refinement fits all 18 834 harvested clamp decisions and cuts
the sawtooth to ONE wrong sample — but that sample ((31,19): all four
true_err equal, no `max_error` reading satisfies the encoder's branch)
REFUTES the model, so it was not shipped. Minimal repros checked in:
fixture 60 (277 B sawtooth, one bad sample), 61 (644 B VarDCT LfQuant;
trigger is channel size ≥15×15 both dimensions, content-irrelevant). 8F
adds: reproduces losslessly at 300×100; can corrupt samples SILENTLY
(268×100 decodes at 500× normal error). Eliminated with evidence: err_sum
last-column, wider clamp bounds, corrected-weight symmetric clamp, bit
depth. Next surfaces: `err[i]`/`err_sum` weight computation, last-column NE
substitution. Method note: branching probes must prefer the decoder's own
context or they desync (the old (2,1) first-divergence was that artifact;
the true one is (29,17)). See
`docs/experiments/2026-08-03-h52-clamp-lower-gate-and-sawtooth.md`. 8C's
54/57 gate tests remain `#[ignore]`d.

**Slice 8 residuals (beyond the modular bug):** patches K.3 (unlocks
bike_5/bike/progressive), extra channels in kVarDCT (alpha corpus cases),
Hornuss/DCT4x4/AFV/≥DCT128 untested by any pixel comparison (cjxl never
emitted them), RAW matrices, progressive/multi-pass streams.

---

## 2026-08-03 (VarDCT wave 2) — 8C HF decode proven on real streams; 8D-dequant + CfL

**8C** — `vardct/{order,hf_coeff}.rs`: I.3.1 orders, I.3.3 histograms, I.4
full context model to quantized integers. **The ANS final-state +
section-exhaustion gate passes on fixtures 50/51/52/53/55/56 and corpus
`grayscale`/`grayscale_5`** — six Order IDs, four non-square transforms,
permutation branch exercised by the corpus (`used_orders = 20`).
Mutation-verified (channel order, `c ^ 1`, `prev` seed all caught). New
flip-point `PREV_USES_CURRENT_PASS_COEFFICIENT`. **Trap:** the passing ANS
gate is structurally blind to the order-table direction (contexts depend on
`k`, never `order[k]`); `order[k]` = destination cell per I.3.1's assignment,
and 8F's pixel comparison is the decisive evidence. Unexercised by any
stream found: LF/QF thresholds (`lf_idx ≡ 0` everywhere), `num_hf_presets
> 1`, `num_passes > 1`.

**8D-dequant** — `vardct/cfl.rs` + `lf.rs`'s dequant half: I.5.2
(dequant → LF CfL → smoothing, in that clause-stated order), I.6 (LF: one
frame-wide `(kX,kB)`; HF: per-64×64-tile via `CflFactors::for_hf`, applied
in I.5.3 by 8F). **Wave-1 flip-point REVERSED by fixture evidence:**
`LF_QUANT_CHANNEL_ORDER_IS_XYB = false` — LfQuant is Y,X,B (under the XYB
reading two channels decode exactly flat while channel 0 carries all
structure). See `docs/experiments/2026-08-03-lf-quant-channel-order-fixture-evidence.md`.

**ESCALATION — the open modular bug now blocks VarDCT:** fixtures 54 and 57
fail inside G.2.2 `LfQuant`'s own modular decode (C.3.2 terminal check),
same content-dependent family as the sawtooth trap. Siblings 55/56 pass.
Root-cause hunt dispatched alongside wave 3; 8C's two gate tests un-ignore
when it's fixed.

**Next:** wave 3 = 8F assembly + acceptance (fix HfGlobal skip, wire
everything, e2e_vardct.rs ladder) in parallel with the modular bug hunt.

---

## 2026-08-03 (VarDCT wave 1) — 8B parameter bundles, 8D-parse sub-bitstreams

**8B** — `vardct/{quantizer,block_ctx,dequant_matrix}.rs`: G.1.2, I.2.1–I.2.6
complete; reuses `DecodeError` (no error.rs wiring needed). **FOURTH DEFECT
CANDIDATE:** Table I.6's DCT128x256 Y/B bases break the per-family doubling
regularities while preserving exactly the doubled fractional parts; verified
at the image scan (not OCR). Printed values ship behind
`DCT128X256_DEFAULT_BASES_AS_PRINTED`, sentinel test
`large_dct_bases_double_per_size_step`; see
`docs/experiments/2026-08-03-i25-default-dequant-constants.md`. Eight I.2.5
OCR garbles settled at page-ranged scan reads. RAW dequant matrices expose
`raw_requests()`/`set_raw_matrix()`; `matrix()` is typed `Unsupported` until
8F wires section `3*num_lf_groups + index`. 8F integration snippet is in the
8B report (read_lf_channel_dequantization → read_lf_global_vardct →
read_hf_global_params).

**8D-parse** — `vardct/{lf,hf_meta}.rs` + `frame/stream_index.rs`: G.2.2
LfQuant to quantized planes (I.5.2 seam marked), G.2.4 four channels +
greedy varblock placement (covered-exactly-once, no LF-group crossing,
reject-not-clamp), all five H.4.1 stream-index formulas typed (decode.rs's
two inline sites migrate in 8F). Open flip-point:
`LF_QUANT_CHANNEL_ORDER_IS_XYB` (needs nonzero-LF-chroma fixture, 8F).

**Traps:** `latex/part1.tex` and the transcription PDF are BYTE-IDENTICAL
for Part 1 numeric tables — they are one source, not two; the only
independent pair is `part1.md` vs that pair. Do not "fix" Table I.6 index 16
without flipping the constant. Do not apply ×64 to I.2.5 defaults — they are
already post-scale (Hornuss 280 vs DCT8x8 3150 is the cross-check).

**Next:** wave 2 = 8C (HfPass/coefficient decode, opus) + 8D-dequant
(I.5.2/I.6, callback to the 8D agent); then wave 3 = 8F assembly.

---

## 2026-08-03 (VarDCT wave 0) — 8A math, 8E filters, 8F0 conformance metrics

Slice 8 (VarDCT) is underway per the approved plan (sub-slices 8A–8F + 8F0,
four waves). Wave 0 landed:

**8A** — `jpxl-core` gains `varblock.rs` (Table I.1/I.4/I.7 vocabulary,
`CoeffMatrix`/`SampleBlock` distinct types — coefficients always landscape,
I.3.2 natural order, I.8 LLF, I.9.2–I.9.8 reconstructions), block-coordinate
newtypes in `geometry.rs`, I.7.2/I.7.3 wrappers + power-of-two kernels to 256
in `dct.rs` (its `[provisional]` scaling note is RESOLVED: I.7.2 = orthonormal
× uniform `1/√s` forward / `√s` inverse per 1-D pass — do not fold the factor
into dequant matrices). 8B consumes `TransformType::{dequant_matrix_index,
coeff_rows, coeff_cols, order_id}`; 8C consumes `natural_coeff_order`.
**THIRD DEFECT IN THE PUBLISHED STANDARD:** I.8's `ScaleF` divides by zero
from DCT16x16 up, identically in all three Part 1 sources; the shipped
reading passes the varblock dimension (Dirichlet-identity derivation, exact
<1e-9), flip-point `LLF_SCALEF_ARG_IS_VARBLOCK_DIMENSION`, see
`docs/experiments/2026-08-03-i8-scalef-argument.md`. DCT8x4 half placement
settled from I.9.8's stated layout (`DCT8X4_HALF_INDEX_IS_LOW_COORDINATE`,
probe-worthy in 8F but not blocking).

**8E** — `frame/{gaborish,epf}.rs`: J.3 with sum-to-1 rescale, J.4.1–J.4.4
with all three steps; pure f32-plane functions, 8F wires them. OCR: step-0
EPF kernel coord is `{0,-2}` (part1.md right, LaTeX `{9,-2}` wrong — third
markdown-beats-LaTeX case); `epf_quant_mul=0.46` / `epf_sigma_for_modular=1.0`
LaTeX-only. FOUR OPEN FLIP-POINTS in `epf.rs` awaiting 8F's filters-on
probe: `EPF_STEPS_FROM_EXPLICIT_CONDITIONS`,
`EPF_BORDER_SAD_AT_REFERENCE_PIXEL`, `EPF_SKIP_IS_PER_VARBLOCK`,
`EPF_DISTANCE_USES_STEP_INPUT` (`docs/experiments/2026-08-03-epf-flip-points.md`).

**8F0** — `jpxl-conformance` gains Part 3 §4.2 grading: `FloatImage`,
hand-rolled NPY reader (djxl grayscale is channels=1, NOT replicated RGB —
trap), normalized f32 peak + per-channel RMSE ("root of the sum" read as
root-mean, documented). Conformance corpus references downloaded (39/39,
`bike_5` verified). Fixtures 50–57 (filters on/off × d1/d4 × gray/RGB);
zero-slack djxl-vs-djxl self-grading test proves the pipeline.

**Traps (permanent copies below):** do not "fix" `scale_f` back to the
printed I.8 call; `AFV_BASIS` is f64 on purpose (verbatim spec digits,
orthonormality to 1.5e-14 proves the two OCR repairs).

**Next:** wave 1 = 8B (I.2 parameter bundles, opus) + 8D-parse (G.2.2/G.2.4
modular sub-bitstreams, sonnet); then wave 2 = 8C + 8D-dequant; wave 3 = 8F
assembly/acceptance.

---

## 2026-08-03 (wave 2) — slices 4 and 10 complete; flip-points pinned; gab_custom dead-code bug fixed

**1. Slice 4 (ICC, E.4) done.** `jpxl-decode/src/icc/` decodes the
compressed ICC representation; fixtures 30–36 (script-built profiles, v2 and
v4, 336–6676 bytes) byte-exact vs `djxl --orig_icc_out`. Key readings, all in
`docs/experiments/2026-08-03-icc-stream-placement.md`: the E.4.1 payload is
UNALIGNED after the headers (aligned reading fails on the first symbol — no
flip-point needed); E.4.4's dictionary has 17 entries (`part1.md` truncates
to 15 — trap below); `output_size` is a constraint (growth refused), metered
by AllocGuard, capped per Table M.1 level 10 as a module-local constant
(promote to a `Limits` field if configurability is ever wanted). New API:
`extract_icc_profile()`, `DecodedImage::icc_profile`.

**2. Slice 10 (encoder breadth) done.** 16-bit gray, RGB via YCoCg-R
(`rct_type = 6` declared once in LfGlobal), multi-group via SectionStore
(each section encoded once into its own buffer → TOC from measured lengths →
bodies appended), `jxlc` container behind a CLI flag with a `jxll` level-10
box for >8-bit. Self-roundtrip + djxl + jxl-oxide sample-exact across the
full matrix (both depths, both channel counts, all four `group_size_shift`
values, naked and boxed, up to 600×520). 16-bit blocker settled by widening
the token alphabet (power-of-two sizes keep the flat prefix code free;
`token_bits` capped at 5 by C.3.3's `n < 32`); `split_exponent` bought
nothing. Externally confirmed readings: multi-section `LfGlobal` carries
ModularHeader + tree + C.1 bundle and ZERO samples; group sub-bitstreams
predict rectangle-relative (H.3 edges are the group's own).

**3. Flip-point sweep.** Real bug found and fixed: `read_restoration_filter`
returned on `all_default` before computing the gaborish fields, so
`GAB_CUSTOM_REQUIRES_NOT_ALL_DEFAULT` was dead code either way. AvgAll was
withdrawn as a flip point — both primary sources agree on `Idiv 16`, never
ambiguous. New named constants `NESTED_LZ77_REJECTS_ENABLED` and
`RESETS_CANVAS_SHARED_ACROSS_BUNDLES` (both keep the shipped reading). All
four remain UNEXERCISED by real cjxl output (~24 probes; cjxl never emits
those configurations) — documented as negative results in
`docs/experiments/2026-08-03-flip-point-fixtures.md`; each reading is pinned
by hand-built-bitstream unit tests instead. Fixture 41 (RGBA) is the first
end-to-end extra-channel decode, and it works.

**Known open bug (queued, do not lose):** a 32×32 grey source of
`x*7 + y*3` (wrapping sawtooth) encoded `cjxl -d 0 -e 3` fails to decode:
`out of bounds: 16 bit(s) requested at bit position 2112`. Reproduces
without ICC and at commit 26d8df3, so it is in the modular/frame layer, not
ICC. Needs a minimised probe fixture and a root-cause hunt.

**Next:** slice 8 (VarDCT) or slice 9 (container breadth: `jxlp`, Exif,
brob); encoder future work list is at the end of the slice-10 report themes
(ANS backend, real MA trees, palette/squeeze write side, alpha, TOC
permutation, `jpxl-encode-policy`).

---

## 2026-08-03 — H.5.2 clamp fixed, multi-section proven, slice 7.5 encoder complete

Three concurrent tasks, all landed:

**1. Fixtures 05/09/10 resolved — the H.5.2 CLAMP was the cause, not
`max_error`.** The slice-7 diagnosis below was wrong and is corrected there in
place. `max_error` is exactly the clause as written; the corrupt input was
`true_err`, because the *prediction* was being clamped by the printed
symmetric clamp. Four oracle-pinned samples (tabulated in
`docs/experiments/2026-08-03-h52-clamp-asymmetry.md`) prove no single guard
over the two printed products can be right — the two halves of the clamp are
gated differently: cap above by `max(W3,N3,NE3)` when `p1<=0 || p2<=0`; floor
below by `min(W3,N3,NE3)` only on strict disagreement (`p1<0 || p2<0`) or
when all three neighbour errors are zero. Flip-point
`EXPERIMENT_CLAMP_SYMMETRIC = false` (replaces `EXPERIMENT_CLAMP_BITWISE_OR`),
tagged `[provisional]` — this describes libjxl 0.13.0, which the printed
clause does not. All 11 lossless fixtures now bit-exact vs djxl (new debug
fixtures 20 palette-bands 24×24 and 21 gradient 260×10, generated by
`tools/make-debug-fixtures.sh`). Newly resolved flip-points:
`EXPERIMENT_MAX_ERROR_RULE = 0` (settled), `EXPERIMENT_ERR_SUM_LAST_COLUMN =
1` (now exercised — 0 and 2 break 05/09 under the corrected clamp).

**2. Multi-section decode PROVEN** (was implemented-but-unproven). Fixtures
14 (600×520 gray8, 12 sections), 15 (600×520 RGB, 12 sections), 16 (511×8,
5 sections) decode bit-exactly vs djxl; `tests/e2e_multisection.rs` asserts
`num_sections > 1` against the sidecar-recorded value so a regenerated
single-section fixture fails loudly. cjxl v0.13 has no group-size flag; its
heuristic drops to group_dim 256 when a dimension exceeds 512 or the image is
very thin — the only lever is image shape.

**3. Slice 7.5 complete — first interoperable pair.** New `jpxl-encode`
crate: gray8 lossless modular naked codestreams (no transforms, one-leaf MA
tree, gradient predictor, prefix codes with a flat 16×4-bit code via RFC 7932
§3.5's single-nonzero-length degeneracy — the alphabet lengths cost zero
bits; LZ77 off; single group/section, ≤1024×1024). All three acceptance
criteria pass sample-exact: self-roundtrip, djxl, jxl-oxide. `BitWriter`
added to `jpxl-bitstream` (chooses the first fitting U32 distribution,
rejects unrepresentable values); `jpxl encode` CLI subcommand (P5 PGM in).
`jpxl-encode` uses `jpxl-decode`/`jpxl-entropy` as dev-dependencies only.

**Still open (unexercised flip-points):** AvgAll Idiv-vs-shift, nested-LZ77,
`gab_custom`, `resets_canvas`.

**Next:** slice 10 encoder breadth (16-bit needs a wider alphabet or nonzero
`split_exponent` — `jpxl-encode::entropy` caps at 2^15−1; RGB+RCT;
multi-group via SectionStore; `jxlc` container) or slice 4 (ICC) / 8 (VarDCT).

---

## 2026-08-02 — slice 7 (end-to-end lossless decode) — 6 of 9 fixtures bit-exact vs djxl

**State:** `jpxl_decode::decode()` works end to end for lossless modular:
signature → headers → frame/TOC → sections (G.1.3/G.2.3/G.4.2) → modular →
inverse transforms → pixels, plus ~90-line jxlc/jxlp container extraction and
a `jpxl decode` CLI subcommand (hand-rolled P5/P6). Bit-exact against djxl:
fixtures 03, 07, 08 (container + 16-bit), 11 (256×256 -e7), 12 (300×200), 13.
Caveat: all current fixtures are single-section (cjxl chose group_dim 512);
multi-section decode is implemented per spec but unproven.

**Experiments resolved by oracle evidence:**
- H.2 global tree: distributions are SHARED from LfGlobal; each sub-bitstream
  re-initialises only per-stream state (ANS seed, LZ77 window) after its
  ModularHeader (`SymbolDecoder::open_deferred`/`restart`;
  `GLOBAL_TREE_SHARES_DISTRIBUTIONS`). Evidence: fixture 12, 32-bit desync.
- **H.5.2 clamp guard is a defect in the standard itself**: both sources print
  `(p1 | p2) <= 0`; the operationally-correct reading (matching the prose) is
  `(p1 <= 0) or (p2 <= 0)`, differing exactly when one neighbour error is
  zero. `EXPERIMENT_CLAMP_BITWISE_OR = false`. Fixed fixtures 11 and 12.
- H.5.2 true_err is used CLAMPED; max_error tie-break is strict `>`.

**Not exercised by these fixtures** (flip-points unchanged, still open):
err_sum last column, AvgAll Idiv-vs-shift, nested-LZ77, gab_custom,
resets_canvas.

**~~Unresolved~~ CORRECTED 2026-08-03 (diagnosis was wrong):** this entry
originally blamed H.5.2 `max_error` selection for the 05/09/10 divergence.
The real cause was the H.5.2 *clamp* corrupting the prediction (and hence
`true_err`) upstream; `max_error` is normative as written. See the
2026-08-03 entry above and
`docs/experiments/2026-08-03-h52-clamp-asymmetry.md`. The suspects listed
here (Table H.4 numbering, shift −1 state) were investigated and eliminated.

---

## 2026-08-02 — slice 5 (Modular mode, Annex H) complete

**State:** `jpxl-decode::modular` decodes modular sub-bitstreams end to end:
ModularHeader, MA trees (decode/validate/traverse, Limits-capped), all 14
predictors incl. the weighted predictor (H.5.1/H.5.2), UnpackSigned, and
inverse RCT (42 variants, round-tripped) / palette + delta-palette / squeeze
(round-tripped against an independent forward implementation over odd sizes).
113 tests incl. a 2000-case no-panic fuzz sweep. `decode_channels` is public
so slice 7 can supply its own SymbolDecoder.

**Open for slice 7 (oracle experiments, in priority order):**
1. H.2 global-tree distributions: reuse the global clustered bundle vs read a
   fresh C.1 bundle per group (spec text contradicts itself; literal second
   reading implemented). Wrong answer desynchronises a whole group.
2. Table H.3 row 13 `AvgAll` uses `Idiv 16` (differs from `>> 4` for every
   negative sample) — implemented as `Idiv`, verify.
3. H.5.2 `err_sum` last-column `+= err[i]_W` (LaTeX-only text) — one addition,
   verify.

**Resolved-by-reasoning (documented in modular/mod.rs):** H.6.2 shift restore
omission; H.6.4 `/4` as integer division; H.6.4 `(index & 1) == 0` despite
Table 1 precedence making the literal text constant-false.

---

## 2026-08-02 — slice 6 (FrameHeader/TOC/groups, Annexes F/G/J.1) complete

**State:** `jpxl-decode::frame` parses FrameHeader with its full conditional
forest, passes, blending, RestorationFilter (J.1), TOC with entropy-coded
Lehmer permutation, and group/section geometry. 102 new tests.

**Spec gotchas encoded as tests:** `HfGlobal` section exists (zero-length) in
Modular mode — `num_sections` is always `2 + num_lf_groups + num_groups ×
num_passes` regardless of encoding (F.3.1 NOTE 1); F.3.3 permutes *offsets*
computed from as-read order, not sizes; F.3.2 `GetContext` uses `min(7, …)` —
the LaTeX corrupted the 7 (second confirmed markdown-beats-LaTeX case; the
LaTeX fails specifically on numeric constants inside prose).

**Open for slice 7 (oracle experiments, one-bit differences):**
- J.1 `gab_custom` guard: implemented as `!all_default && gab` (the literal
  bare `gab` guard would cost a bit even under `all_default`, violating the
  invariant every other bundle obeys). Constant
  `GAB_CUSTOM_REQUIRES_NOT_ALL_DEFAULT` flips it in one place. Highest-value
  oracle check — differs on nearly every real frame.
- F.2 `resets_canvas`: computed once from colour blending_info and shared
  with every ec_blending_info (vs per-bundle evaluation; 2 bits per extra
  channel).

---

## 2026-08-02 — Part 2 clause map re-audited, no longer provisional

**State:** `STANDARDS_INDEX.md`'s Part 2 section replaced the arXiv-derived
provisional topic map with a real clause map verified against `part2.md`
(735 lines, full 22-page OCR, read in full). Real structure: clauses 1–9 are
the main body (1 scope, 2 normative references, 3 terms, 4 general, 5 file
organization, 6 data types, 7 graphical descriptions, 8 binary box format,
9 box types 9.1–9.11), and there are exactly two annexes, **both normative**
— A (JPEG Bitstream Reconstruction procedure, A.1–A.11) and B (JPEG XL Media
Type registration, B.1–B.2). No informative annex, unlike Part 1.

Confirmed the box set from clause 9: signature box (9.1, the 12 fixed bytes),
`ftyp` (9.2), `jxll` level box (9.3, at most one, third box if present,
default level 5), `jumb` (9.4, delegates to 19566-5), `Exif` (9.5, codestream
wins on overlap), `xml ` (9.6), `brob` Brotli-wrapper (9.7), `jxli` frame
index (9.8), `jxlc` full codestream (9.9), `jxlp` partial codestream (9.10,
index-ordered concatenation semantics), `jbrd` JPEG reconstruction data (9.11,
Tables 11–18). **`jhgm` (HDR gain map) is not in this 2nd-edition text at
all** — the old provisional entry listing it was wrong for this edition;
dropped rather than carried forward unverified.

Crosswalk gained two Part 2 rows: clause 9.1 signature box → `jpxl-conformance::sniff`
(exists) and clauses 8–9 box parsing → `jpxl-decode` (slice 9, not started).

**OCR quality note:** Table 11 (the `jbrd` `JPEGBitstream` bundle, pages
12–14) is badly garbled — subscripted field names collapse into glyph noise
(`Tyyw`, `Tpey`, `OFse`, etc.) and the marker-array loop condition reads as
nonsense. Flagged in the clause map; do not implement slice 9's `jbrd`
parsing from this table without a scan cross-check. Everything else in
`part2.md` reads cleanly, including Annex A's segment-reconstruction rules.

**Next:** unchanged — slices 2 and 3 remain ahead of slice 9 in the plan.

---

## 2026-08-02 — slice 3 (entropy, Annex C) complete; oracles live

**State:** `jpxl-entropy` covers all of Annex C with nothing stubbed: C.2.1
bundle, C.2.2 clustering + inverse MTF, C.2.3 hybrid-uint, C.2.4 prefix codes
(RFC 7932 derivation, not transcription), C.2.5/C.2.6 ANS histograms + alias
mapping, C.3.2 state machine, LZ77 with the reconstructed 120-entry
`kSpecialDistances` (validated by monotonic `dx²+dy²` ordering). 75 tests,
layer-by-layer. Oracle infra is live: djxl/cjxl v0.13.0 pinned + built,
jxl-oxide 0.12.6 (ignores output extensions — always pass `--output-format`),
conformance corpus at 4bf05352, four reproducible cjxl fixtures incl. a
300×200 multi-group case.

**Open for slice 7 (oracle experiments queued):**
- C.2.2 nested-LZ77 reading: implemented as a *constraint* (nested
  `lz77.enabled` flag is read and must be 0, stream rejected otherwise), not
  an override. One-bit difference; verify against djxl-produced streams.
- `tests/oracle_vectors.rs` harness is ready; its fixture-driven test is
  `#[ignore]`d with TODO(slice 7).

---

## 2026-08-02 — slice 2 (image headers) complete

**State:** `jpxl-decode` parses signature + the full `ImageMetadata` bundle
tree (D.2/D.3, E.2/E.3 colour encoding, L.2.1 opsin, B.3 extensions, B.2.6
enums) — 97 tests, every field traced, trace intervals proven gap/overlap-free.
Public API: `jpxl_decode::headers::decode_image_headers(&mut BitReader,
&Limits)`.

**Source-fidelity corrections (both directions now proven):**
- `latex/part1.tex` is NOT uniformly better than `part1.md`: AspectRatio
  ratio 5 reads `16 Idiv 39` in the LaTeX (wrong); the markdown's `16 Idiv 9`
  is right (16:9). Cross-check numeric constants in BOTH sources.
- `quant_bias0..2` (Table L.1): ~~sign ambiguous, taken positive~~ —
  **resolved 2026-08-02 by a clean-scan screenshot**: the printed defaults are
  the expressions `1 − 0.05465…`, `1 − 0.07005…`, `1 − 0.049935…`, i.e.
  ≈ 0.9453 / 0.9299 / 0.9501. Both OCRs had collapsed the leading `1 −`. The
  original "positive 0.05465" reading was **wrong** and is fixed in
  `headers/opsin.rs` (see Already fixed).

**Spec gotchas encoded as tests (do not relearn):** `default_m` is NOT under
`all_default` (minimal metadata is two bits, not one); `BitSet(cw_mask, b)`
takes masks 1/2/4, not bit indices; extra-channel names kept as raw bytes
(UTF-8 validity is not a conformance requirement).

---

## 2026-08-02 — Part 1 LaTeX landed; STANDARDS_INDEX re-audited

**State:** `latex/part1.tex` (6236 lines, one TeX page per source page, all 96
pages) is present, alongside a text-only transcription PDF at
`original-pdfs-do-not-read-first-if-markdown-exists/ISO_IEC_18181-1_2024_transcription.pdf`.
The LaTeX is now the highest-fidelity Part 1 source: it restores pseudocode
bodies that `part1.md` truncated (B.2.3 `U64()` continuation loop, B.2.4
`F16()`) and corrects OCR digit noise in tables and examples.

**Worked example of that noise:** B.2.2's example reads `U32(8, 16, 32, u(7))`,
bits `10` → 32, and `U32(u(2), u(4), u(6), u(8))`, bits `010111` → 7. The
markdown misreads the constants. Treat every numeric constant taken from
`part1.md` as unverified until checked against `part1.tex` — a wrong
distribution constant produces a plausible-looking parse that desynchronises
every later field.

**Done:** `STANDARDS_INDEX.md` re-audited. The `latex/` row moved from pending
to present; the transcription PDF added to the locator note; the provisional
arXiv-derived Part 1 topic map **replaced** by a real clause map (Annexes A–O
with titles, ToC page numbers, and key subclauses, each letter verified against
the text — note J is restoration filters, K image features, L colour
transforms, and simple upsampling is J.2 while non-separable upsampling is
K.2). The crosswalk now carries real clause numbers (B.2.x → `jpxl-bitstream`,
I.7/I.9 → `jpxl-core::dct`, L.2/L.3 → `jpxl-core::color`, 5.1/5.3 →
`jpxl-core::geometry`, M → `jpxl-core::limits`). `AGENTS.md` §2 and §3 updated:
the resolution chain is now part1.tex → markdowns → transcription PDF → arXiv
paper → image scans → oracle experiment; `latex/` is confirmed gitignored.

**Still provisional:** ~~the Part 2 topic map in `STANDARDS_INDEX.md` is still
arXiv-derived; `part2.md` is complete and it should be re-audited the same
way.~~ Done, see the 2026-08-02 "Part 2 clause map re-audited" entry above.

**Next:** unchanged — slices 2 and 3, slice 3 the critical path.

---

## 2026-08-02 — standard OCR landed and audited

**State:** ISO/IEC 18181 Parts 1–4 are now complete OCR markdowns at
`markdowns/standard-markdowns/part1.md` … `part4.md` (4230 / 735 / 325 / 163
lines). A first OCR pass was **rejected**: it dropped comparison and shift
operator glyphs (`<`, `<<`, `<=`, `>>`) — fatal for bitstream pseudocode — and
lost whole pages. The accepted pass has 30–48 % more words, intact operators,
text reflowed into paragraphs and code blocks, and the lost pages recovered
(Part 1 Annex N, Part 2 A.11). Caveat: dense syntax-table and formula pages can
still scramble; spot-check them against the original scan (now in
`original-pdfs-do-not-read-first-if-markdown-exists/original/`) before treating
the markdown as sole normative source.

`part1.md` is the primary normative source from now on; the arXiv paper drops
to design rationale and cross-checking. (Superseded by the entry above: the
LaTeX conversion has since landed and outranks `part1.md`.)

**Queued:** re-audit every `[provisional]` tag in `jpxl-bitstream` and
`jpxl-core`. `STANDARDS_INDEX.md` is done — see the entry above.

**Next:** slices 2 and 3 are unblocked; slice 3 remains the critical path.

---

## 2026-08-02 — scaffold wave complete

**State:** the five-task scaffold wave has landed and the full gate is green:
`cargo build/test/clippy -D warnings/fmt --check` across the workspace, 111
tests passing. What exists and is proved:

- `jpxl-bitstream`: `BitReader` (LSB-first), `Bool`/`U32`/`U64`/`F16`/
  `ZeroPadToByte`, feature-gated bit-position tracing (`trace`), 37 tests with
  hand-derived vectors. `longU64()` definition confirmed verbatim from the
  arXiv paper. F16 is pure bit-manipulation (deterministic on all targets).
- `jpxl-core`: error style established (`JpxlError`, hand-rolled, `From`
  chains); `Limits`/`AllocGuard` (charge-before-allocate); checked geometry
  newtypes; XYB forward constants taken verbatim from the paper (p. 24),
  inverse derived by exact rational inversion (verified to 1e-5) — all
  `[provisional]`; `dct.rs` with orthonormal DCT-II/III 8/16 (1-D, 2-D square
  and rectangular), naive-reference and coefficient-layout tests. JPEG XL's
  own scaling is a wrapper prefactor at the call boundary, never baked into
  the kernels.
- `jpxl-conformance`: `sniff` (FF0A / container box), oracle discovery+runner
  (djxl, jxl-oxide; `JPXL_ORACLE_BIN` override), PPM parser + peak-error
  metrics. `jxl-oxide` CLI invocation is `[verify at first use]`.
- `jpxl-cli`: `jpxl info <file>` works on all three handmade fixtures with the
  specified exit codes (0 recognized / 2 unknown / 1 I/O error).
- `tools/setup-oracles.sh` and `tools/fetch-conformance.sh` written, NOT yet
  run (network/cmake). `fetch-conformance.sh` refuses to run until
  `PINNED_COMMIT` is set — deliberate, keep it that way.
- Slice 1 of `PLAN.md` is complete; slice 8's standalone math groundwork
  (DCT, XYB) is in place.

**Design notes for later:**
- Nonzero `ZeroPadToByte` padding maps to `BitstreamError::Overflow` (no
  dedicated variant yet); add `MalformedPadding` when header work starts if
  wanted.
- `jpxl-core::color` implements plain `B = S_gamma`; the paper's XYB′
  (`B′ = B − Y`) decorrelation step is NOT implemented — decide when the
  bitstream work reaches it.

**Blocked on the user:** ~~OCR of ISO/IEC 18181 Parts 1, 2, and 3~~ —
**resolved same day**, see the entry above. Everything derived here came from
the arXiv paper and is tagged `[provisional]`; none of it has been checked
against the real text yet.

**Next:** `PLAN.md` slice 2 (signature + `SizeHeader`/`ImageMetadata`, with
oracle header-dump cross-check) and slice 3 (entropy coding core: prefix
codes, rANS, hybrid-uint, LZ77, clustering). Slice 3 unblocks slices 4, 5, and
7, so it is the critical path.

---

## Already fixed — do not redo

- **`read_u32` wraps, it does not error** (2026-08-02). 18181-1 B.2.2:
  `(offset + v) Umod (1 << 32)`. The scaffold version returned `Overflow` on
  `offset + payload` overflow; fixed to `wrapping_add` with a clause citation
  and the test `u32_offset_plus_payload_wraps_mod_2_pow_32`. Do not "harden"
  this back into an error.
- **XYB inverse matrix is verified normative** (2026-08-02). The rationally
  derived `OPSIN_ABSORBANCE_INVERSE_MATRIX` matches 18181-1 L.2.1 Table L.1
  defaults digit-for-digit at `f32`; no longer `[provisional]`. The spec
  signals `opsin_bias0..2` as negative (decoder-side); our forward-side
  positive bias is the same convention mirrored — documented in
  `jpxl-core/src/color.rs`.

- **`quant_bias` defaults are `1 − x`, not `x`** (2026-08-02). Table L.1
  prints the defaults as literal expressions (`1 - 0.05465007330715401`, …);
  verified against a clean scan after both OCRs collapsed the `1 −` prefix.
  `DEFAULT_QUANT_BIAS` ≈ [0.9453, 0.9299, 0.9501] in `headers/opsin.rs`. Do
  not "simplify" these back to the small constants.

## Traps — do not fix these by loosening a check

- **Modular residual LZ77 must use H.3 `dist_multiplier`** (2026-08-05).
  The residual `SymbolDecoder` gets `set_dist_multiplier(max_channel_width)`.
  Encoding distances as `raw = distance - 1` (as if M=0) produces streams that
  round-trip in isolation but fail under full modular decode. Use
  `raw = distance + 119` when M > 0 (C.3.3 past the special table). Do not
  "fix" by forcing M=0 on the decoder.
- **`EXPERIMENT_CLAMP_SYMMETRIC = false` is not a loosened check**
  (2026-08-03). Restoring the printed symmetric H.5.2 clamp "to match the
  spec" re-breaks fixtures 05/09/10/20; the contradicting oracle samples are
  tabulated in `docs/experiments/2026-08-03-h52-clamp-asymmetry.md`. The
  printed clause is wrong for libjxl 0.13.0 streams.
- **The encoder writes `RestorationFilter` explicitly OFF** (`gab = false`,
  `epf_iters = 0`, 2026-08-03). The Table J.1 *defaults* are `gab = true`,
  `epf_iters = 2` — decoder-side smoothing our decoder does not implement
  yet, so an `all_default` J.1 bundle self-roundtrips green while djxl and
  jxl-oxide return different pixels. If external decodes ever drift while
  self-roundtrip stays green, look here first.
- **`part1.md` truncates E.4.4's tag dictionary to 15 entries** (2026-08-03).
  The real list has 17 (`bTRC`, `dmda` dropped by the OCR), fixed by the
  tagcode range 4..=20 and confirmed by byte-exact fixtures. Use the LaTeX.
- **`modular_16bit_buffers = false` for >8-bit encodes is deliberate**
  (2026-08-03). It is a truthful claim about decoder working buffers (D.3);
  the paired consequence is the `jxll` level-10 box in container output
  (Annex M). Do not "restore the Table D.3 default".
- ~~Sawtooth 32×32 decode bug~~ **RESOLVED 2026-08-03** (corrected in
  place): root cause was H.5.2's clamp guard OCR'd as `*` instead of `^` in
  every transcription. Fixed in `weighted.rs`; fixtures 60/61/62 are the
  regression tests. Replacement trap: **when every transcription agrees on
  nonsense, suspect the transcription pipeline, not the standard — escalate
  to the image scan before claiming a standard defect.** Three sessions
  modelled the OCR artifact instead of reading one scan page.
- **Annex H OCR corruptions, resolved 2026-08-02 — do not re-transcribe from
  the corrupted source:** Table H.4 rows 4/5 are `abs(N)`/`abs(W)` (both
  sources garble one each); `kDeltaPalette[4]` is `{0,-12,0}` (LaTeX's
  `{0,-12,9}` is wrong); Table H.3 row 13 is `WW` not `WH` (pinned by the
  coefficients-sum-to-16 test); H.5.2 weight normalisation and `error2weight`
  exist ONLY in the LaTeX (part1.md drops the whole block); H.6.3 in part1.md
  is scrambled — use the LaTeX, where `B = B + A&A` means `B = B + A`.
- **C.2.6 alias mapping is `symbols[u] = o` (the overfull index), NOT
  `symbols[u] = 0`** (2026-08-02). The LaTeX renders it as `0` — an OCR
  corruption; only `o` is consistent with the algorithm. The invariant test
  (each symbol s appears exactly D[s] times across all slots, offsets a
  permutation of 0..D[s]) fails under `= 0`. If that test ever fires, the bug
  is in new code, not the test.
- **jxl-oxide ignores the output-file extension** and writes PNG bytes into
  any filename. The harness rejects PPM-from-jxl-oxide before spawning
  (`OracleError::UnsupportedFormat`). Do not "fix" a BadMagic PPM parse error
  by loosening the PPM parser — pass `--output-format` explicitly.
