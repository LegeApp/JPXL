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

Tiny fixtures written by hand, byte by byte, usually with `printf`. They exist
to pin down *format* facts (signatures, header bit patterns, boundary
conditions) rather than image content, so they stay in the repository and in
review.

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
> the group size** (e.g. 300×260), so partial edge groups are exercised too.

"Passes on 32×32" is not evidence that a stage works. Treat a stage without a
multi-group fixture as untested.

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
