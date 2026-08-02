# HANDOFF

Dated working ledger. **Prepend** new entries — newest first. Each entry: what
changed, what is now proved, what is next, what is blocked.

Two sections at the bottom are permanent and must be kept current:
"Already fixed — do not redo" and "Traps — do not fix these by loosening a
check". When a diagnosis turns out to be wrong, correct it **in place** and
mark it corrected; do not leave a wrong explanation standing.

Keep this file small. Entries whose content has landed in `PLAN.md`,
`CONFORMANCE.md`, or `docs/experiments/` get deleted from here.

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
- `quant_bias0..2` (Table L.1) sign is ambiguous in both sources (`|1-0.05…`);
  taken **positive** because L.2.3 multiplies small coefficients by it and a
  negative would invert them. **Needs a check against a clean scan** — flagged
  to the user.

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

**Still provisional:** the Part 2 topic map in `STANDARDS_INDEX.md` is still
arXiv-derived; `part2.md` is complete and it should be re-audited the same way.

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

## Traps — do not fix these by loosening a check

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
