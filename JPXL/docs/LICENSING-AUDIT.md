# JPXL licensing audit (MIT-only)

Scope: the JPXL Rust workspace under `JPXL/` and the repository's licensing
surface. This audit answers one question: **is JPXL cleanly MIT-licensed, with
no copyleft and nothing that blocks redistribution under MIT?**

Answer: **yes.** Every JPXL crate declares MIT; every third-party dependency is
permissive and MIT-compatible; no dependency is copyleft (no GPL / LGPL / AGPL /
MPL / CDDL / EUPL / SSPL / OSL / CeCILL anywhere in the graph).

- Method: `cargo tree --workspace [--all-features] -e normal,build,dev
  --format "{p}|{l}"`, plus manual inspection of `LICENSE`, `JPXL/LICENSE-MIT`,
  and every crate's `Cargo.toml`.
- Host / date: recorded per run; the dependency set below reflects
  `JPXL/Cargo.lock` as committed.

---

## 1. JPXL's own packages — MIT

- Workspace default: `JPXL/Cargo.toml` sets `[workspace.package] license = "MIT"`.
- Every member crate (`jpxl`, `jpxl-bitstream`, `jpxl-core`, `jpxl-entropy`,
  `jpxl-decode`, `jpxl-encode`, `jpxl-encode-policy`, `jpxl-cli`,
  `jpxl-conformance`, `jpxl-perceptual`, `jpxl-plan-render`, and the WIP
  `jpxl-jpeg`) declares `license.workspace = true`, i.e. inherits MIT. No crate
  overrides it. No crate uses a dual or `OR`-ed expression of its own.
- Repository license files:
  - `LICENSE` (repo root) — MIT, `SPDX-License-Identifier: MIT`,
    "Copyright (c) 2026 JPXL contributors", plus a trailing note that the
    *gitignored* working material (a BSD-3-Clause libjxl checkout used only as a
    black-box oracle, copyrighted ISO/IEC standards documents, and local test
    images) is **not** covered and **not** distributed.
  - `JPXL/LICENSE-MIT` — the identical MIT grant, same copyright line.

**Consistency check:** the two files carry the same MIT terms and the same
copyright holder and year. They are consistent; the only difference is the root
`LICENSE`'s explanatory note about gitignored non-distributed material, which is
appropriate to keep at the repo root. **No edit required.**

---

## 2. Copyleft scan — none

```
cargo tree --workspace --all-features -e normal,build,dev --format "{p}|{l}" \
  | grep -iE 'GPL|LGPL|AGPL|MPL|CDDL|EUPL|CPAL|SSPL|OSL|CeCILL'
```

Result: **no matches.** The old AGPL `jxl-encoder` project (AGPL-3.0-only OR a
commercial license) is **not present** in this repository and is not a
dependency; per AGENTS.md §2 its only permitted channel of inheritance is the
`JPEG_XL_CLEAN_IMPLEMENTATION_LESSONS.md` lessons file, which carries no code.

---

## 3. Third-party dependencies

Every dependency below is permissive and compatible with distributing the
aggregate under MIT. Licenses shown are the crates' own SPDX expressions; where
an expression is an `OR`, the project may elect the MIT arm when one is offered.

### 3.1 Default build (`cargo build --workspace`, no optional features)

These are the crates a normal build and the shipped CLI actually pull. All but
two offer MIT directly; the two that do not (`moxcms`, `pxfm`) offer
BSD-3-Clause, which is permissive and MIT-compatible.

| Crate | License (SPDX) | MIT arm? |
|---|---|---|
| adler2 | 0BSD OR MIT OR Apache-2.0 | yes |
| autocfg | Apache-2.0 OR MIT | yes |
| bitflags | MIT OR Apache-2.0 | yes |
| bytemuck | Zlib OR Apache-2.0 OR MIT | yes |
| byteorder-lite | Unlicense OR MIT | yes |
| cfg-if | MIT OR Apache-2.0 | yes |
| color_quant | MIT | yes |
| crc32fast | MIT OR Apache-2.0 | yes |
| crossbeam-deque | MIT OR Apache-2.0 | yes |
| crossbeam-epoch | MIT OR Apache-2.0 | yes |
| crossbeam-utils | MIT OR Apache-2.0 | yes |
| either | MIT OR Apache-2.0 | yes |
| fax | MIT | yes |
| fdeflate | MIT OR Apache-2.0 | yes |
| flate2 | MIT OR Apache-2.0 | yes |
| gif | MIT OR Apache-2.0 | yes |
| half | MIT OR Apache-2.0 | yes |
| image | MIT OR Apache-2.0 | yes |
| image-webp | MIT OR Apache-2.0 | yes |
| miniz_oxide | MIT OR Zlib OR Apache-2.0 | yes |
| **moxcms** | **BSD-3-Clause OR Apache-2.0** | **no (BSD-3-Clause)** |
| num-traits | MIT OR Apache-2.0 | yes |
| png | MIT OR Apache-2.0 | yes |
| proc-macro2 | MIT OR Apache-2.0 | yes |
| **pxfm** | **BSD-3-Clause OR Apache-2.0** | **no (BSD-3-Clause)** |
| qoi | MIT OR Apache-2.0 | yes |
| quick-error | MIT OR Apache-2.0 | yes |
| quote | MIT OR Apache-2.0 | yes |
| rayon | MIT OR Apache-2.0 | yes |
| rayon-core | MIT OR Apache-2.0 | yes |
| safe_arch | Zlib OR Apache-2.0 OR MIT | yes |
| simd-adler32 | MIT | yes |
| syn | MIT OR Apache-2.0 | yes |
| tiff | MIT | yes |
| unicode-ident | (MIT OR Apache-2.0) AND Unicode-3.0 | yes (code); Unicode-3.0 covers data tables |
| weezl | MIT OR Apache-2.0 | yes |
| wide | Zlib OR Apache-2.0 OR MIT | yes |
| zerocopy | BSD-2-Clause OR Apache-2.0 OR MIT | yes |
| zerocopy-derive | BSD-2-Clause OR Apache-2.0 OR MIT | yes |
| zune-core | MIT OR Apache-2.0 OR Zlib | yes |
| zune-jpeg | MIT OR Apache-2.0 OR Zlib | yes |

`moxcms`, `pxfm`, `image`, `image-webp`, `png`, `gif`, `tiff`, `qoi`,
`zune-*`, `weezl`, `color_quant`, `fax`, `byteorder-lite` reach the graph only
through the `image` raster adapter used by `jpxl-cli`; the public `jpxl` facade
and all normative codec crates are pixel-buffer based and do not pull the file
adapters (see `JPXL/Cargo.toml` notes).

### 3.2 Optional, measurement-only (feature-gated, off by default)

These enter **only** with the `jpxl-conformance` metric features
(`ssimulacra2` / `butteraugli`), used to *grade* lossy encodes against the
oracle. A default `cargo build --workspace` does not fetch them, and no
normative crate depends on them (see the extended note in `JPXL/Cargo.toml`).

| Crate | License (SPDX) | MIT arm? |
|---|---|---|
| aligned-vec | MIT | yes |
| archmage | MIT OR Apache-2.0 | yes |
| archmage-macros | MIT OR Apache-2.0 | yes |
| av-data | MIT | yes |
| byte-slice-cast | MIT | yes |
| **butteraugli** | **BSD-3-Clause** | **no (BSD-3-Clause)** |
| bytes | MIT | yes |
| equator | MIT | yes |
| equator-macro | MIT | yes |
| **imgref** | **CC0-1.0 OR Apache-2.0** | **no (CC0 / Apache-2.0)** |
| log | MIT OR Apache-2.0 | yes |
| magetypes | MIT OR Apache-2.0 | yes |
| num-bigint | MIT OR Apache-2.0 | yes |
| num-derive | MIT OR Apache-2.0 | yes |
| num-integer | MIT OR Apache-2.0 | yes |
| num-rational | MIT OR Apache-2.0 | yes |
| rgb | MIT | yes |
| rustversion | MIT OR Apache-2.0 | yes |
| safe_unaligned_simd | MIT OR Apache-2.0 | yes |
| **ssimulacra2** | **BSD-2-Clause** | **no (BSD-2-Clause)** |
| syn (3.x) | MIT OR Apache-2.0 | yes |
| thiserror | MIT OR Apache-2.0 | yes |
| thiserror-impl | MIT OR Apache-2.0 | yes |
| **v_frame** | **BSD-2-Clause** | **no (BSD-2-Clause)** |
| yuvxyb | MIT | yes |
| yuvxyb-math | MIT | yes |

---

## 4. Items worth a human's eye (none blocking)

1. **`moxcms` and `pxfm` on the default path** are `BSD-3-Clause OR Apache-2.0`
   — MIT is not one of their options. This is fully compatible with shipping the
   aggregate under MIT (permissive dependencies inside an MIT project are
   normal), but the BSD-3-Clause copyright/attribution notices for these two
   crates should be retained in any distributed `NOTICE`/third-party-licenses
   bundle. They enter only through the `image` raster adapter in `jpxl-cli`.

2. **`butteraugli` (measurement-only) is a Rust port of libjxl's butteraugli.**
   BSD-3-Clause, permissive. It is off by default and never linked into a normal
   build. AGENTS.md §2 and the `JPXL/Cargo.toml` clean-room note already record
   that using it as a *metric* is the same category as running `cjxl`/`djxl` as
   oracles, whereas wiring it into the encoder's own rate control would make the
   perceptual model a derivative of libjxl and needs a recorded decision first.
   No such wiring exists today.

3. **`unicode-ident`** carries `Unicode-3.0` for its data tables (in addition to
   MIT/Apache for code). Unicode-3.0 is permissive and only asks that its notice
   be retained. `imgref` (measurement path) offers CC0-1.0 (public-domain
   equivalent) or Apache-2.0. Neither is copyleft.

4. **Attribution hygiene for release:** because the graph mixes MIT with
   Apache-2.0, BSD-2/3-Clause, Zlib, 0BSD, Unlicense, CC0 and Unicode-3.0
   (all permissive), a shipped binary should carry a generated third-party
   license notice (e.g. `cargo about` / `cargo-bundle-licenses`). This is an
   attribution convenience, not a licensing conflict.

## 5. ISO / standards text

No ISO/IEC 18181 text is referenced or embedded in any shipped crate. The
standards markdown/PDF sources live in gitignored directories
(`markdowns/`, `latex/`, `original-pdfs-.../`) and never enter the build or git
history, per AGENTS.md §3 and §9. Clause-number citations in comments are fine;
quoted normative passages are not, and none were found in the workspace crates.

## Reproduce

```sh
cd JPXL
# Full dependency + license graph (default features):
cargo tree --workspace -e normal,build,dev --format "{p} {l}"
# Superset including the optional measurement metrics:
cargo tree --workspace --all-features -e normal,build,dev --format "{p} {l}"
# Copyleft check (expect no output):
cargo tree --workspace --all-features -e normal,build,dev --format "{l}" \
  | grep -iE 'GPL|LGPL|AGPL|MPL|CDDL|EUPL|SSPL'
```
