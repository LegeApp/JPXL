# Four flip points, targeted fixtures, and one dead-code bug

Date: 2026-08-03
Status: complete.

## 1. Question

Four named flip-point constants existed with no fixture that made their two
readings diverge:

1. `AvgAll` (Table H.3 row 13): `Idiv 16` versus `>> 4`, differing on
   negative intermediate sums.
2. C.2.2's nested-LZ77 rule: reject a stream that sets `lz77.enabled` on a
   nested distribution decoder whose parent had `num_dist == 2` (constraint
   reading) versus silently overriding the flag to disabled.
3. `GAB_CUSTOM_REQUIRES_NOT_ALL_DEFAULT` (Table J.1): whether `gab_custom` is
   gated on `!all_default && gab` or on bare `gab`.
4. `resets_canvas` (F.2/F.7): computed once from the colour `blending_info`
   and shared with every `ec_blending_info` bundle, versus re-evaluated per
   bundle against that bundle's own mode.

Question: do targeted `cjxl` fixtures make each pair of readings actually
diverge, and if so, which reading matches real streams?

## 2. Preregistered gate

* **Pass** (per flip point): a fixture (or, failing that, a hand-built
  bitstream) exists where the two readings produce different parses, and one
  reading is shown correct — either because pixel decode only succeeds under
  it, or because the wrong reading corrupts the TOC/section structure.
* **Fail**: a reading is shown wrong on some fixture.
* **Inconclusive / negative**: no stream tried (probed via `cjxl`, or
  constructible via a hand-built bitstream check) makes the two readings
  diverge at all. Per `docs/experiments/README.md`, a negative result is
  retained, not discarded.

## 3. Method

Host: Linux 6.17.0-40-generic, x86-64. Oracle: `JPXL/tools/oracle-bin/cjxl`
and `djxl`, JPEG XL v0.13.0 196a43d9 (libjxl rev
`196a43d996aa6ed33ebf98812a7c6d43b2b6d01b`), as pinned in
`tools/oracle-bin/PINNED_REVISIONS.txt`. libjxl source was not read; both
binaries were used only to encode and to decode.

1. **`cjxl --help -v -v -v -v` was read in full.** No flag forces a modular
   predictor, forces LZ77, or forces a per-extra-channel blend mode. `--gaborish=0|1`
   and `--epf=-1..3` exist and do apply to modular frames (Table J.1's rows are
   not `kVarDCT`-only).
2. **Twelve `cjxl`-produced probes**, generated with a throwaway Python/PPM/PAM
   script (not committed — the three that turned out useful are the recipe in
   `tools/make-experiment-fixtures.sh`, fixtures 40-42): a 64x64 checkerboard,
   128x16 four-value stripes, and a 32x32 two-value noise pattern, each at
   `cjxl -e 1/3/9`; plus the existing gradient fixtures re-encoded at
   `-m 0`/`-m 1`, `-d 0`/`-d 1`, `--gaborish=0/1`, effort 1/3/5/7/9; plus a
   16x16 RGBA PAM.
3. **A temporary instrumented trace** (`eprintln!` behind
   `std::env::var_os("JPXL_DBG_NESTED_LZ77")`, in
   `crates/jpxl-entropy/src/decoder.rs`, fully reverted — confirmed by
   `git diff` showing no trace of it — before this commit) recorded every call
   to `SymbolDecoder::open_nested` where `forbid_lz77` was true, over the
   eleven existing lossless fixtures plus the twelve new probes.
4. **Direct TOC/header-consistency parsing**, using the public
   `jpxl_decode::{decode_image_headers, frame::header::read_frame_header,
   frame::geometry::FrameGeometry, frame::toc::read_toc}` API from a scratch
   binary (not committed; equivalent logic now lives in the unit tests listed
   in §6), comparing `section_base + toc.total_size()` against the file
   length, and printing `all_default`/`blending_info.mode`/
   `ec_blending_info[..].mode` directly.
5. **Full pixel decode** via `jpxl_decode::decode()` where the encoding was
   modular (VarDCT pixel decode is out of scope for this decoder regardless of
   this experiment; see `decode.rs::check_supported`).

## 4. Raw results

### 4.1 `AvgAll`: not actually ambiguous — withdrawn as a flip point

Both `latex/part1.tex:3269` and `markdowns/standard-markdowns/part1.md:2175`
render Table H.3 row 13 as `... Idiv 16`, not `>> 4`. There is no
disagreement between the two Part 1 sources, so per `AGENTS.md` §2's
ambiguity-resolution order this never reaches step 6 (oracle experiment) —
step 1/2 already answers it. `crates/jpxl-decode/src/modular/predictor.rs`
already implements `Idiv` (Rust's truncating `/`) for row 13, already
documents the `Idiv`-vs-`>>` distinction in its module doc, and already has
unit tests (`idiv_truncates_towards_zero_but_the_shift_floors`,
`avgall_coefficients_sum_to_sixteen`) pinning the negative-value behaviour.
**No `EXPERIMENT_AVG_ALL_*` constant
exists in the codebase and none was added**: introducing one to represent a
choice that both primary sources already agree on would misstate an
unambiguous clause as an open question. No fixture was built for this one.

### 4.2 Nested-LZ77: unreached in every probe

The instrumented trace (method 3) never fired: across the eleven existing
fixtures and twelve new probes, `SymbolDecoder::open_nested` is never called
with `forbid_lz77 == true`. That requires a real entropy-coded bundle with
exactly two pre-clustered contexts (`num_dist == 2`, e.g. a two-leaf MA tree's
data stream) that *also* chooses the general/nested clustering encoding over
the simple fixed-width one — an encoder heuristic decision with no CLI lever,
and evidently one `cjxl` v0.13.0 does not make for any of these probes.

Since no reachable case was found, the constraint-vs-override behaviour was
still made testable at the unit level directly on `SymbolDecoder::open_nested`
(`crates/jpxl-entropy/src/decoder.rs`,
`nested_lz77_rejects_the_flag_when_forbidden`): a 5-bit hand-built stream with
`lz77.enabled = 1` and `forbid_lz77 = true` is rejected with the expected
message under the current (`NESTED_LZ77_REJECTS_ENABLED = true`) reading. This
confirms the code does what it claims; it is not evidence about which reading
libjxl uses, since libjxl was never observed choosing the ambiguous path at
all.

### 4.3 `gab_custom`: found unreachable, fixed, still unexercised by `cjxl`

Before this experiment, `read_restoration_filter`
(`crates/jpxl-decode/src/frame/restoration.rs`) returned immediately after
reading `all_default`:

```
let all_default = read_bool(reader)?;
if all_default {
    return Ok((RestorationFilter::default(), None));
}
let gab = read_bool(reader)?;
let read_gab_custom = gab && (!GAB_CUSTOM_REQUIRES_NOT_ALL_DEFAULT || !all_default);
```

`all_default` is provably `false` at the point `read_gab_custom` is computed
(the early return already handled `true`), so `!all_default` is always `true`
there, making `read_gab_custom = gab` **regardless of
`GAB_CUSTOM_REQUIRES_NOT_ALL_DEFAULT`'s value**. The constant was dead code:
flipping it could never change parsed bits. This was caught by running the
same fixture through both settings of the constant (via the header/TOC probe,
method 4) and observing byte-identical cursor positions in every case — which
is also consistent with "no fixture exercises it", so the dead code was only
confirmed by reading the function, not inferred from the null result alone.

**Fixed**: `gab` (and, if `gab_custom` is read true under the literal
reading, its six weights) is now computed *before* the `all_default` early
return, so the constant is live. The existing unit test
`all_default_costs_one_bit` already asserted the bit count for the current
reading and already carried the comment "this is the assertion that changes
if the literal reading ... wins" — it was previously true independent of the
constant; it is now a real assertion of the constant's effect.

With the fix in place, the same twelve probes (plus `--gaborish=0/1` on the
16x16 gradient, `-m 0`/`-m 1`, `-d 0`/`-d 1`, and the existing
04/06/11/12/13/20/21 fixtures) were parsed with
`GAB_CUSTOM_REQUIRES_NOT_ALL_DEFAULT` at both `true` and `false`: every one
gave the identical cursor position and TOC total under both settings, because
`RestorationFilter.all_default` is `false` in every one of them — `cjxl`
v0.13.0 always writes at least one explicit non-default field in this
bundle (a Gaborish or EPF choice), even when asked for defaults. The flip
point is real (now that the bug is fixed) but still unexercised by any stream
found.

Fixture 42 (`--gaborish=1` forced) is *not* mathematically lossless despite
`-d 0`: `djxl` decodes it to pixels that differ from the source, while the
otherwise-identical `--gaborish=0` encode round-trips byte-identical. J.3
(the Gaborish transform) evidently smooths reconstructed pixels regardless of
the requested distortion target. This decoder does not implement J.3/J.4
pixel-stage filtering (out of scope per `restoration.rs`'s existing module
doc), so fixture 42 is used for header/TOC parsing only.

### 4.4 `resets_canvas`: unexercised — no CLI lever, and single-frame streams always agree

The header/TOC probe (method 4) confirmed `blending_info.mode` and every
`ec_blending_info[i].mode` are `kReplace` in every single-frame stream tried,
including the RGBA fixture (41). `cjxl`'s CLI has no flag that sets an extra
channel's blend mode independently of the colour channel's; that only varies
across frames of an animation or under patches, neither of which this
decoder's scope currently reaches (`decode.rs::check_supported` rejects
`flags.patches()`, and multi-frame blending is rejected as "more than one
regular frame"). No stream was found — or is currently reachable by this
decoder — that could tell the two readings apart.

The ambiguity was still made testable at the bitstream level directly: a
hand-built `FrameHeader` stream
(`frame::header::tests::resets_canvas_is_shared_from_the_colour_bundle`) with
colour `mode = kReplace` and `ec_blending_info[0].mode = kAdd` parses to
completion, with `ec_blending_info[0].source == 0`, under the current
(`RESETS_CANVAS_SHARED_ACROSS_BUNDLES = true`) reading — and would be two bits
short of valid under the unshared reading, since that reading would expect an
explicit `source` row for the `kAdd` bundle that this stream does not carry.

## 5. Conclusion

No flip point was resolved by oracle evidence — none of the four differences
were ever observed to matter for a real `cjxl` v0.13.0 stream. This is a
negative result for all four, with one bug found and fixed along the way:

* **`AvgAll`**: withdrawn as a flip point. Both primary sources of Table H.3
  agree on `Idiv`; there is nothing to test.
* **Nested-LZ77**: `NESTED_LZ77_REJECTS_ENABLED = true` (constraint reading,
  unchanged) — unexercised by any stream tried; libjxl 0.13.0 appears never
  to reach the ambiguous configuration (`num_dist == 2` with general
  clustering) at all in these probes.
* **`gab_custom`**: `GAB_CUSTOM_REQUIRES_NOT_ALL_DEFAULT = true` (unchanged) —
  was dead code (now fixed, so the constant is live); still unexercised, since
  `cjxl` never emits an `all_default == true` `RestorationFilter`.
* **`resets_canvas`**: `RESETS_CANVAS_SHARED_ACROSS_BUNDLES = true`
  (unchanged) — unexercised; no single-frame stream can differ per-bundle,
  and multi-frame streams are outside this decoder's current scope.

What this does **not** establish: that any of the three still-open readings
is what the standard intends, or even what libjxl does in general — only that
these specific probes never reached the divergent case. A future `cjxl`
version, a hand-crafted (non-`cjxl`) conformance stream, or support for
multi-frame blending could still exercise `resets_canvas`; a stream with a
genuinely two-leaf general-clustered distribution could still exercise the
nested-LZ77 rule; and any encoder that ever emits a truly-default restoration
filter could exercise `gab_custom`.

## 6. Consequences

* `crates/jpxl-decode/src/frame/restoration.rs`: fixed
  `read_restoration_filter` so `gab`/`gab_custom` are computed before the
  `all_default` early return, making `GAB_CUSTOM_REQUIRES_NOT_ALL_DEFAULT`
  live rather than dead code. Constant unchanged (`true`). Module doc and the
  constant's doc comment updated with the "unexercised (negative result)"
  finding.
* `crates/jpxl-decode/src/frame/header.rs`: added
  `RESETS_CANVAS_SHARED_ACROSS_BUNDLES` (new flip point, `true`, matching the
  existing behaviour) and wired the alternative (per-bundle peek) reading
  behind it. New unit test
  `resets_canvas_is_shared_from_the_colour_bundle`.
* `crates/jpxl-entropy/src/dist.rs`: added `NESTED_LZ77_REJECTS_ENABLED` (new
  flip point, `true`, matching the existing behaviour) and its doc comment.
* `crates/jpxl-entropy/src/decoder.rs`: wired the override reading behind
  `NESTED_LZ77_REJECTS_ENABLED`. New unit test
  `nested_lz77_rejects_the_flag_when_forbidden`.
* `tools/make-experiment-fixtures.sh` (new) and fixtures
  `40_checker_64x64_lossless.jxl`, `41_rgba_gradient_16x16_lossless.jxl`,
  `42_gab_forced_16x16_lossless.jxl` with provenance sidecars.
  `crates/jpxl-decode/tests/e2e_experiments.rs` (new) exercises all three:
  40 and 41 bit-exactly against their source formulas (41 is this project's
  first end-to-end RGBA/extra-channel decode), 42 for header/TOC/section
  parsing only (it is not pixel-lossless; see its sidecar).
* No `AvgAll` constant was added; `predictor.rs` is unchanged.
