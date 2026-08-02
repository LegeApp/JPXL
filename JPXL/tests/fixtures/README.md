# JPXL test fixtures

Everything a JPXL test reads from disk lives here. The directory is split by
**provenance**, because provenance decides what may be committed and what a
failure means.

```
tests/fixtures/
├── handmade/      committed        tiny, hand-authored, byte-level
├── generated/     gitignored       produced by our own tools
└── conformance/   gitignored       fetched from libjxl/conformance
```

## `handmade/` — committed

Small fixtures with fully documented provenance. Two kinds live here:

| Fixture | Bytes | What it is |
|---------|-------|------------|
| `00_signature_naked.bin` | 6 | `FF 0A` + filler; sniffing only, not decodable |
| `01_signature_container.bin` | 20 | 12-byte JXL signature box + filler; not decodable |
| `02_not_jxl.bin` | 16 | PNG magic; the negative case |
| `03_gradient_8x8_lossless.jxl` | 242 | real stream, lossless, single group |
| `04_gradient_8x8_lossy.jxl` | 86 | real stream, VarDCT `-d 1`, single group |
| `05_gradient_300x200_lossless.jxl` | 414 | real stream, lossless, **multi-group** |
| `06_gradient_300x200_lossy.jxl` | 2512 | real stream, VarDCT `-d 1`, **multi-group** |

The `.bin` files are hand-authored byte by byte with `printf`. The `.jxl` files
are produced by `tools/make-handmade-fixtures.sh`, which synthesises a gradient
with a few lines of arithmetic and encodes it with `tools/oracle-bin/cjxl` —
that script *is* the recipe the sidecars refer to, and it re-verifies every
round-trip when run. The synthesised sources land in `generated/` and are not
committed; each sidecar records their sha256 so a regenerated source can be
checked against the one the fixture was made from.

Rules:

* **Every fixture has a `<name>.txt` sidecar** naming its exact bytes, what each
  byte means, and where it came from. A fixture whose provenance is not written
  down is a clean-room liability; delete it rather than guess.
* **Keep them small.** If a fixture is over a few hundred bytes it probably
  belongs in `generated/`.
* **Hand-authored means hand-authored.** Do not paste bytes out of a file
  produced by libjxl, cjxl, or any other implementation. Published constants
  from the specification (signatures, magic numbers) are fine, and the sidecar
  says so.
* Signature-only fixtures are **not decodable streams**. They exercise sniffing
  and error paths, nothing more; say so in the sidecar.
* Encoder-produced fixtures record the **exact oracle revision and flags** they
  came from, plus the sha256 of both the source and the fixture. `cjxl` is a
  black box that emits conformant streams; its source is never read.

## `generated/` — gitignored

Fixtures our own tools produce: synthetic images, encoder round-trip outputs,
fuzz-corpus minimisations, oracle decodes. Regenerable by definition, so they
are not committed. Anything that generates into this directory must be
re-runnable from a clean checkout with no network access.

## `conformance/` — gitignored

The official [libjxl/conformance](https://github.com/libjxl/conformance) test
suite, fetched by `tools/fetch-conformance.sh` at a pinned commit. Requires
network access; not committed, because it is a third-party tree with its own
licence and history. The pinned commit is recorded in the script so a given
JPXL revision always grades against the same suite.

## Multi-group coverage is mandatory

A JPEG XL frame is tiled into groups (256×256 by default), and a very large
share of real decoder bugs — group ordering, TOC handling, per-group entropy
state, edge groups that are not full size — are invisible on a single-group
image. So:

> **Once a decode path is real, it must be covered by at least one fixture of
> at least 256×256, and preferably one whose dimensions are not a multiple of
> the group size**, so partial edge groups are exercised too.

"Passes on 32×32" is not evidence that a stage works. Treat a stage without a
multi-group fixture as untested.

`05_gradient_300x200_lossless.jxl` and `06_gradient_300x200_lossy.jxl` are that
fixture for the lossless and lossy paths respectively. 300 exceeds one group in
x, and neither 300 nor 200 is a multiple of 256, so the right-hand *and* bottom
group edges are both partial.

## Expected-output caching

Stages that are meant to be **bit-exact** (bitstream parsing, entropy decode,
integer transforms, lossless modulars) cache their expected output as a
`.sha256` sidecar next to the fixture:

```
handmade/07_modular_lossless.jxl
handmade/07_modular_lossless.jxl.txt      provenance
generated/07_modular_lossless.rgba.sha256 expected decode digest
```

The digest is over the raw decoded sample buffer in a documented layout (the
sidecar states width, height, channel count, sample type and byte order — a
digest without that description is unfalsifiable). A test recomputes the digest
and compares; a mismatch is a hard failure, never a tolerance question.

Lossy stages do **not** get digests. They are graded with
`jpxl_conformance::metrics` against an oracle decode, with an explicit error
budget stated in the test.

Regenerating a `.sha256` because a test failed is how a regression gets
enshrined. Update a digest only alongside an explanation of why the previous
value was wrong.
