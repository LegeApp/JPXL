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
| 2026-08-03 | [h52-clamp-asymmetry](2026-08-03-h52-clamp-asymmetry.md) | What single reading of Annex H decodes all nine handmade lossless-modular fixtures bit-exactly? | H.5.2's clamp is asymmetric, not the symmetric clamp the clause prints; `max_error` is the clause as written and the slice-7 diagnosis is withdrawn. |
