# ICC payload placement and the E.4.4 tag dictionary

Date: 2026-08-03
Slice: 4 (ICC decode, 18181-1 E.4)

## Question

Two questions, settled by the same run:

1. **Placement.** Table A.1 lists the ICC profile between `Headers` and the
   first `Frame`, but no clause states whether the E.4.1 payload begins at the
   bit where `ImageMetadata` ended or after a `ZeroPadToByte()`. The two
   readings differ by up to seven bits and desync everything downstream.
2. **Dictionary length.** The two Part 1 sources disagree on E.4.4's tag
   signature dictionary. `latex/part1.tex` lists seventeen entries; the
   `part1.md` OCR lists fifteen (it drops `bTRC` and `dmda` from the run of
   strings). `tagcode` selects from it over the range 4..20 inclusive, which is
   seventeen values.

## Preregistered gate

Implement the *unaligned* reading (continue at the current bit) and the
*seventeen-entry* dictionary, then decode the seven `3N_icc_*.jxl` fixtures.

* **Pass** — all seven decoded profiles are byte-identical to
  `djxl --orig_icc_out`, *and* the Annex C terminal-state check of C.3.2
  succeeds on each ICC stream, *and* the frame following the profile decodes
  to the source formula.
* **Fail** — any byte differs, or any stream fails the terminal-state check.
* **Inconclusive** — the fixtures all use prefix codes rather than ANS (no
  terminal-state evidence) *and* the profiles are short enough that a
  misalignment could go unnoticed.

The three conditions are independent evidence for the placement question. A
wrong start bit corrupts the very first symbol, so a byte-exact profile alone
would already be decisive; the ANS terminal state and the following frame make
it triply so.

## Method

Binaries, both black boxes (AGENTS.md §2):

```
cjxl/djxl v0.13.0 196a43d9  (libjxl 196a43d996aa6ed33ebf98812a7c6d43b2b6d01b)
recorded in JPXL/tools/oracle-bin/PINNED_REVISIONS.txt
```

Fixtures and reference profiles produced by `JPXL/tools/make-icc-fixtures.sh`,
which builds seven ICC profiles byte by byte in Python, encodes each with
`cjxl -d 0 -e 3 -x icc_pathname=<profile>`, and verifies with
`djxl --orig_icc_out` that the stored profile survives the round trip. Digests
are in each fixture's `.txt` sidecar.

Decoder: `jpxl_decode::icc`, entry `read_icc_profile`, called from `decode()`
immediately after `decode_image_headers_metered` and **before**
`reader.zero_pad_to_byte()`.

Test: `JPXL/crates/jpxl-decode/tests/e2e_icc.rs`.

## Raw results

Bit position at which the ICC payload starts and ends, measured with
`BitReader::total_bits_read()`:

| fixture | after headers | after ICC stream | profile bytes |
| --- | --- | --- | --- |
| 30 gray | 41 | 709 (byte 88, bit 5) | 336 |
| 31 rgb matrix | 41 | 971 (byte 121, bit 3) | 456 |
| 34 gray table | 41 | 1717 (byte 214, bit 5) | 1356 |

The header ends at bit 41 in every case — not a byte boundary — and the ICC
stream is read from bit 41 onwards with no padding.

`cargo test -p jpxl-decode --test e2e_icc`:

```
test every_decoded_profile_is_self_consistent ... ok
test decodes_every_profile_byte_for_byte ... ok
test a_codestream_without_a_profile_reports_none ... ok
compared 7 ICC fixture(s) against djxl byte for byte
test matches_djxl_orig_icc_out ... ok
test truncation_inside_the_profile_never_panics ... ok
test consuming_the_profile_leaves_the_frame_readable ... ok
```

All seven profiles byte-identical, sizes 336, 452, 456, 492, 504, 1356 and
6676 bytes. `SymbolDecoder::finish()` (C.3.2) succeeded on every stream. The
frames following the profiles decode to their source formulas, including
fixture 35's multi-section 300x200 frame whose TOC offsets depend on the
profile having been consumed to exactly the right bit.

## Conclusion

1. **Placement: unaligned.** The E.4.1 payload begins at the bit where
   `ImageMetadata` ended; there is no `ZeroPadToByte()` between them. The
   `ZeroPadToByte()` of F.1 belongs to the frame and happens after the profile.
   Gate passed on all three independent conditions.
2. **Dictionary: seventeen entries.** `cprt, wtpt, bkpt, rXYZ, gXYZ, bXYZ,
   kXYZ, rTRC, gTRC, bTRC, kTRC, chad, desc, chrm, dmnd, dmda, lumi`. The
   `part1.md` rendering is an OCR truncation, not a variant reading: fifteen
   entries cannot cover `tagcode` 4..20, and fixtures 30, 31 and 34 decode
   correctly only with the seventeen-entry list (they use `desc`, `wtpt`,
   `kTRC` and the grouping codes, whose indices shift if entries are missing).
   `latex/part1.tex` is confirmed as canonical for Part 1 here, as AGENTS.md §2
   already ranks it.

**What this does not establish.** It is evidence from seven streams produced by
one encoder at one revision. It does not show that the alignment question is
answered *normatively* — only that the unaligned reading is the one that
decodes conformant streams, which is the same standard of proof the rest of the
slice rests on. It says nothing about profiles larger than 6676 bytes, about
`output_size > 2^22` (the level-5 cap), or about the E.4.4 dictionary entries
no fixture reaches (`bkpt`, `chrm`, `dmnd`, `dmda`, `lumi`, `kXYZ`, `kTRC` are
only covered by unit tests with hand-written command streams, not by a real
encoder's output).

## Consequences

* `jpxl_decode::icc` implements the unaligned placement and the seventeen-entry
  dictionary; no flip-point constant was added, because the evidence is not
  one-bit marginal — the alternative reading fails on the first symbol of every
  fixture.
* `decode()` reads the profile between the headers and the frame loop and
  exposes it as `DecodedImage::icc_profile`.
* An unrelated observation, recorded here because it was found by this run and
  is *not* an ICC bug: a 32x32 greyscale source of `x * 7 + y * 3` (a wrapping
  sawtooth) encoded with `cjxl -d 0 -e 3` fails to decode with
  `out of bounds: 16 bit(s) requested at bit position 2112`, with or without an
  embedded ICC profile, and reproduces at commit 26d8df3 (before this slice).
  The ICC fixtures use a smooth ramp instead. This belongs to the modular
  layer; see the slice-4 handoff entry.
