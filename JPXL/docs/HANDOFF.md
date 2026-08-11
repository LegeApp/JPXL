# HANDOFF (retired)

This file is no longer a working ledger. Do not prepend entries here.

Live project state is stored in AKR (`.akr/`) and rendered under
`docs/generated/`:

- `ACTIVE-WORK.md` — current proposed/active work and acceptance checks
- `ROADMAP.md` — tracks, milestones, and ordered work
- `CURRENT-STATE.md` — durable policies, decisions, observations, and evidence
- `DECISION-HISTORY.md` — decision revisions and supersessions

Start work with `knowledge.context --goal <record>` and the expected paths.
Record durable changes through `knowledge.propose`, `knowledge.revise`,
`knowledge.evidence_add`, and `knowledge.complete`; run `knowledge.validate`
before handoff. Raw working logs belong in `.agent/scratch/`.

The former dated ledger and its “Already fixed” / “Traps” sections were folded
into AKR decisions, policies, work, observations, and evidence under the
`jpegxl-rs` namespace. The migration rule is
`@jpegxl-rs.decision.ledger-conventions/1`.

Frozen experiment reports may still cite headings from the retired ledger.
Resolve those historical citations with:

```text
git show 315e763:JPXL/docs/HANDOFF.md
```

Git history preserves the narrative; AKR records are the current authority.
