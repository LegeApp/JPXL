# Advisor pack: post-frontier probe campaign priorities — 2026-09-01

Non-authoritative outside advice, dispositioned by assessment in the AKR
ledger. The advisor inspected the September 1 agent pack (AKR graph and
generated views) and proposed the next optimization campaign.

## Bottom line (advisor)

There is no credible remaining path to another 2x speedup through ordinary
Contract-A codec-core optimization; the remaining route is to stop doing so
much perceptual search work — better prediction of which candidates deserve
the full encode → render → SSIMULACRA2 treatment, with canonical verification
and fallback retained.

The 12 MP traced arithmetic: plan 433 ms (21.8%), render+metric 1350 ms
(67.9%), entropy 90 ms (4.5%), emission 115 ms (5.8%). One avoided probe is
~450 ms (~22-23% of the trace); 3→2 probes ≈ 1.29x; 3→1 ≈ 1.83x. Probe count
is the only demonstrated double-digit lever.

## Advisor's prioritized campaign

1. QPv2 retrain on the current corpus + cost-calibrated candidate/confidence
   model (shadow first, Contract B). Train an explicit second head for
   P(first candidate is an acceptable final rung | features) calibrated
   against expected wall time; make candidate selection optimize expected
   remaining encode cost. The tau=0.90 candidate is systematically
   conservative (pred/winner median 1.34). First experiment is zero risk:
   generate missing oracle labels for all 95 families, retrain offline,
   replay traces, search thresholds without touching production. Target mean
   canonical probes <= 2.0, preferably <= 1.5, not merely "raise one-shot
   above 4.3%".
2. Design a deterministic two-probe common case: predicted crossing → one
   direct correction from the measured error + learned local slope → finish
   if certified → old navigator only on exceptions. QPv2 evidence already
   recorded first-or-one-correction success = 1.00 on the blind holdout.
   `expand_until_bracketed()` + `tighten()` is where 81/94 cases disappear
   into generic search. Contract B: may change which equally valid rung
   wins; insist the promotion gate shows zero quality-efficiency regression.
3. Certified early rejection: SSIMULACRA2's final error is a sum of
   non-negative weights times absolute terms, so partial sums are an exact
   lower bound; once remap(partial) crosses the threshold the probe is
   provably infeasible and need not finish all scales. Instrument existing
   traces first: at what scale would each infeasible probe have become
   provably infeasible? Discard if they cross only at the last scale. A
   Contract-B navigator enhancement, not a Contract-A leaf.
4. Per-probe parallel efficiency: re-measure with a worker-occupancy trace
   (barrier/idle time per stage) at t1/t2/t4/t6; expect 5-15% if a
   scheduling defect remains; obey the September 1 closure (no reopening of
   coarse-scale scheduling, cross-stage colour fusion, XYB regroup, blur
   transpose without new contrary evidence).
5. Text/UI router: 96% of routing overhead is the competing Modular encode;
   train a high-precision P(Modular candidate wins) decision and directly
   route a high-confidence subset, keeping the competition for the uncertain
   remainder. Large content-specific win; not the photographic 2x.
6. Decline single-finalist pricing as formulated (75/76 coarsest-smallest;
   1/76 reversal by 18 bytes; ~100 ms; unnecessary under a no-giveback
   rule). A certified lower bound on the finer finalist's size could recover
   some of it without the rare loss, but ranked low (<= 5% prize).
   Count-both/Store-winner is already contradicted by ledger evidence
   (Phase 8.4: Store-every-finalist slower; naive count-both turns two
   traversals into three).
7. Exact codec-core is housekeeping: no broad hotspot rounds without a
   fresh > 5% defect; AVX-512 blur is a platform-specific frontier (extra
   registers vs AVX2 register pressure) irrelevant to the 13700H host; use
   release-final (fat LTO/PGO) for competitive measurements — an easy
   remaining 2-8%.

## Advisor's target

Balanced q85: <= 1.5 mean full canonical probes on the 95-fixture corpus,
zero floor violations, no matched-quality byte regression, and <= 2 probes
for at least 90% of in-distribution inputs; current search kept as
fallback. "Stop optimizing the three-probe implementation; optimize how
rarely you need three probes."
