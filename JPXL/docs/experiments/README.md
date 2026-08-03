# experiments/

Frozen reports on questions the standard, the paper, and the code could not
answer on their own — chiefly oracle behavioral experiments (resolution step 4
in `AGENTS.md` §2) and measurement work too specific for `PERFORMANCE.md`.

## Rules

**Immutable.** Once a report is committed it is not edited. If its conclusion
turns out to be wrong, write a new report that supersedes it and add a
`Superseded by:` line to the old one — that single line is the only permitted
edit. The record of having believed something is part of the evidence.

**Preregistered gates.** State the question, the method, and the pass/fail
threshold **before** running. A threshold chosen after seeing the data is not a
result. This applies to correctness experiments as much as to timing ones.

**Negative results are retained.** A failed idea that is deleted gets
rediscovered. Reports that found nothing, or found the opposite of the
hypothesis, are as valuable as the ones that worked and are kept in full.

**Oracle experiments describe libjxl, not the standard.** An experiment tells
you what one implementation does. Say so in the conclusion, and mark any
resulting decision `[provisional]` until the normative text confirms it. Never
read libjxl source to explain a result — that crosses the clean-room boundary
(`AGENTS.md` §2).

**Provenance is part of the report.** Binary hashes, input hashes, exact
commands, host state. A report that cannot be re-run is an anecdote.

## Filing

One file per experiment: `YYYY-MM-DD-short-slug.md`. Sections:

1. **Question** — one sentence.
2. **Preregistered gate** — what result would count as pass, fail, or
   inconclusive.
3. **Method** — exact commands, inputs and their hashes, binaries and their
   hashes, host state.
4. **Raw results** — the data, not a summary of it.
5. **Conclusion** — including what this does *not* establish.
6. **Consequences** — what changed in the code or docs as a result, or
   "nothing".

## Index

| Date | Report | Question | Outcome |
| --- | --- | --- | --- |
| 2026-08-03 | [h52-clamp-xor-scan-resolution](2026-08-03-h52-clamp-xor-scan-resolution.md) | What does H.5.2's clamp guard actually say? | **XOR, not multiplication** — every text transcription misread `^` as `*`. One symmetric clamp; the standard is correct. Resolves the sawtooth, 61/62 and VarDCT 54/57. |
| 2026-08-03 | [h52-subpredictor-localisation](2026-08-03-h52-subpredictor-localisation.md) | Which layer of the H.5 weighted predictor is actually wrong? | The sub-predictors. An impossibility proof at fixture 62 (11,5) eliminates max_error, clamp gating, clamp bounds, err_sum/weights, the NE substitution and the transcription. |
| 2026-08-03 | [h52-clamp-lower-gate-and-sawtooth](2026-08-03-h52-clamp-lower-gate-and-sawtooth.md) | Are the sawtooth trap and VarDCT 54/57's LfQuant failure one bug, and what is it? | Same family (H.5 weighted predictor, property 15); lower clamp gate proven too narrow but not shipped; one sample refutes the clamp model. Two minimal reproducers landed. |
| 2026-08-03 | [h52-clamp-asymmetry](2026-08-03-h52-clamp-asymmetry.md) | What single reading of Annex H decodes all nine handmade lossless-modular fixtures bit-exactly? | H.5.2's clamp is asymmetric, not the symmetric clamp the clause prints; `max_error` is the clause as written and the slice-7 diagnosis is withdrawn. |
| 2026-08-03 | [icc-stream-placement](2026-08-03-icc-stream-placement.md) | Does the E.4 ICC payload start at the bit where the headers ended or after a `ZeroPadToByte()`, and does E.4.4's tag dictionary have 15 or 17 entries? | Unaligned (no padding), and 17 entries; all seven ICC fixtures decode byte-identically to `djxl --orig_icc_out`. |
| 2026-08-03 | [epf-flip-points](2026-08-03-epf-flip-points.md) | Where do the two Part 1 transcriptions disagree on J.3/J.4, and which reading of the four underdetermined EPF behaviours does the clause support? | `{0,-2}` (markdown) is the step-0 kernel's 13th coordinate and `epf_quant_mul`/`epf_sigma_for_modular` come from the LaTeX; all four flip points are argued from the clause and pinned by constants, none decided by evidence until wave 3. |
| 2026-08-03 | [flip-point-fixtures](2026-08-03-flip-point-fixtures.md) | Do targeted `cjxl` fixtures exercise the four previously-unexercised flip points (`AvgAll` `Idiv`, nested-LZ77, `gab_custom`, `resets_canvas`)? | No stream tried exercises any of the four; one (`AvgAll`) was never actually ambiguous, and fixing `gab_custom`'s dead-code bug shows the other three are exercisable in principle but not by any `cjxl` v0.13.0 output found. |
| 2026-08-03 | [i8-scalef-argument](2026-08-03-i8-scalef-argument.md) | I.8's printed `ScaleF` is infinite at `c == b/2`, which every transform from DCT16x16 up reaches — what is the intended formula? | `ScaleF`'s second argument is the varblock dimension in samples, not the LF-sample count; derived from the exactness I.8 must have and verified numerically over every Table I.1 shape. |
| 2026-08-03 | [i9-dct8x4-half-placement](2026-08-03-i9-dct8x4-half-placement.md) | I.9.6/I.9.7 never say where their two half-blocks land in the 8x8 output — which half goes where? | Half index 0 takes the low coordinates: I.9.8's 4x8 sub-block is I.9.7's half 1 verbatim and I.9.8 does state its placement. Settled from the text; an 8F oracle probe would still be decisive for the DCT8x4 side. |
| 2026-08-03 | [i25-default-dequant-constants](2026-08-03-i25-default-dequant-constants.md) | Which of Table I.6's printed digits survive the two Part 1 transcriptions, and does anything stay inconsistent once the OCR damage is removed? | Eight garbles settled (all re-confirmed against the scan); a ninth is not OCR — the DCT128x256 Y and B bases contradict the doubling and family-ratio regularities every other row obeys, so the printed values ship behind `DCT128X256_DEFAULT_BASES_AS_PRINTED` with a sentinel test. **Addendum 2026-08-03:** I.2.4's RAW arm derived — the 3-channel matrix is read inline at that bit position, is the planes times `params.denominator` (not reciprocated), and is now decoded rather than refused; two new untestable flip points. |
| 2026-08-03 | [i4-context-model-ans-gate](2026-08-03-i4-context-model-ans-gate.md) | Does JPXL's I.4 context model reproduce a real encoder's context sequence with no reference data, and what does that proof *not* reach? | Eight real `kVarDCT` streams (six handmade fixtures plus corpus `grayscale`/`grayscale_5`) reach C.3.2's terminal ANS state with only byte padding left; mutation testing shows the gate catches the channel order, the `c ^ 1` swap and the `prev` seed, but is structurally blind to the I.3.1 order-table direction, and no available stream exercises the LF/QF thresholds or `num_hf_presets > 1`. |
| 2026-08-03 | [lf-quant-channel-order](2026-08-03-lf-quant-channel-order.md) | G.2.2 never states an order for `LfQuant`'s three channels — which order are they read in? | Provisionally X, Y, B, from I.5.1/I.5.2's repeated `qX, qY, qB` naming; explicitly flagged as weaker than the DCT8x4 precedent and resolvable by one real fixture. **Superseded** by the next entry. |
| 2026-08-03 | [lf-quant-channel-order-fixture-evidence](2026-08-03-lf-quant-channel-order-fixture-evidence.md) | Does decoding a real fixture's `LfQuant` under both channel-order hypotheses resolve the question the previous entry left open? | Yes: two greyscale fixtures at different distances both show the first-decoded channel alone carrying real structure while the other two are exactly flat, which only fits a luma-first (Y, X, B) order — the opposite of the earlier entry's textual reading. `LF_QUANT_CHANNEL_ORDER_IS_XYB` flipped to `false`. |
