# JPXL

JPXL is a clean-room, native Rust implementation of JPEG XL (ISO/IEC 18181).
It is a correctness-first decoder and encoder workspace: the decoder follows
the standard, and the encoder is tested against independent decoders rather
than accepted merely because it round-trips through itself.

**Status:** ready for application integration within the supported feature
set, but not yet a drop-in replacement for every libjxl codec feature. The
Rust facade accepts ordinary pixel buffers and the CLI converts common image
formats. The decoder still deliberately rejects several valid JPEG XL feature
combinations instead of guessing. Its optimized lossy `Balanced` and `Fast`
paths have reached a matched-SSIMULACRA2 speed-parity window against the pinned
libjxl build on the project’s two photo anchors; quality and density still
trail libjxl on important perceptual axes.

## What works today

- Decoding: entropy coding, Modular, VarDCT, ICC, containers (`jxlc`/`jxlp`),
  extra channels, blending/compositing, progressive LF frames, filters,
  orientation, patches, noise, XYB and supported YCbCr reconstruction.
- Lossless encoding: grayscale/RGB, reversible colour transform, palette and
  Squeeze transforms, MA trees, ANS/LZ77, multi-section streams, and
  `jxlc` containers.
- Lossy encoding: RGB8/RGB16 VarDCT with square DCT 8/16/32 transforms, CfL,
  entropy clustering, and a perceptual quality controller (`--quality N`, a
  minimum SSIMULACRA2 score verified on the reconstructed pixels) with a
  byte-targeted rate loop and a fixed quantizer as expert modes.
- Interoperability: supported encoder streams are validated by JPXL, `djxl`,
  and `jxl-oxide`; lossless paths require exact samples.

The detailed, test-backed feature matrix is in
[JPXL/docs/CONFORMANCE.md](JPXL/docs/CONFORMANCE.md). The project ledger and
generated views under [docs/generated](docs/generated) are the plan of record.

## Deliberate limits

The decoder returns a typed `Unsupported` error for unimplemented syntax,
including animation, previews, splines, `brob` decompression, modular-frame
upsampling, YCbCr in modular frames, chroma-subsampled YCbCr reconstruction,
and several less-exercised frame combinations. It does not silently substitute
a result. See the conformance document for the complete list and coverage
boundaries.

The encoder does not yet write alpha channels. Opaque alpha in raster inputs is
accepted; non-opaque alpha is rejected unless the user explicitly flattens it
with `--background`, so the CLI never discards transparency silently. The
lossy encoder currently accepts RGB input; greyscale remains available on the
lossless path. Quality/density parity with libjxl is a separate ongoing target,
not a claim of this integration pass.

## Build and test

The Rust workspace is in `JPXL/` and uses Rust 1.97.1 (edition 2024).

```sh
cd JPXL
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

For fast local test cycles, use `cargo test --workspace --profile fast-debug`.
Release builds are required for benchmarks. Optional oracle setup is available
through `tools/setup-oracles.sh`; it provides `cjxl`, `djxl`, and `jxlinfo` as
black-box validation tools.

## CLI

```sh
# Lossless Modular encode (default)
jpxl encode input.png output.jxl

# Lossy VarDCT to a minimum SSIMULACRA2 quality (the normal contract):
# the smallest stream that scores at least N. Balanced (archival) defaults
# to 85, `--effort fast` (web) to 70; 100 is exact-lossless.
jpxl encode --quality 85 input.jpg output.jxl
jpxl encode --quality --effort fast input.jpg output.jxl

# Expert modes: an exact byte rate, or a pinned VarDCT quantizer
jpxl encode --bpp 1.0 input.jpg output.jxl
jpxl encode --global-scale 40000 input.jpg output.jxl

# Decode and inspect
jpxl decode output.jxl decoded.png
jpxl info output.jxl

# Transparency must be handled explicitly until alpha encoding lands
jpxl encode --background '#ffffff' transparent.png flattened.jxl

# Pipelines are supported; output format is explicit when there is no suffix
cat output.jxl | jpxl decode --format png - - > decoded.png
```

Raster input and output support PNG, JPEG, WebP, TIFF, BMP, GIF, ICO, TGA,
QOI, PGM, and PPM. Input format is detected from its contents. Output format
is inferred from the filename or selected with `--format`; `-` means stdin or
stdout. PNG, TIFF, and PNM retain 16-bit samples. Run `jpxl --help` and
`jpxl bench --help` for the full option set and isolated encoder timing modes.

`--quality` is a minimum SSIMULACRA2 score (0..100, 100 = lossless), not a
distance: cjxl's `-d` targets Butteraugli, a different and inverted scale. The
lossy effort defaults to `balanced`; `--effort fast` (or `--lossy-preset fast`)
trades quality for lower latency and also lowers the `--quality` default score
(fast 70, balanced 85). The exhaustive-reference `quality` effort is gated
behind the `quality-effort` build feature; it also runs the perceptual policy
bank and the terminal coefficient reducer, both measured and kept off for
`fast`/`balanced` because their byte savings (0.4–1.8 %) cost more wall than
those efforts' budgets allow.

## Perceptual quality controller

A `--quality N` encode minimises exact codestream bytes subject to
`SSIMULACRA2(source, decoded) >= N`. Nothing in the loop infers quality from a
byte count: every candidate quantizer is planned, its pixels are reconstructed
from the plan by `jpxl-plan-render` (no codestream is serialised or decoded for
a probe), and the reconstruction is scored by `jpxl-perceptual`, an in-tree,
clean-room SSIMULACRA2 with a precomputed source reference. The score that is
reported is the score the written file has: the reconstruction is bit-exact
with `jpxl-decode`, and the scorer quantizes to the frame's bit depth the way
a viewer would see it.

- **Search.** A trained one-shot model (pooled quantile regression over
  `jpxl features` source statistics plus a DCT8 transform summary; the
  `one-shot-controller` feature, on by default) picks the first probe when the
  input is in its confidence envelope, and falls back to the calibrated
  starting table when it is not (tiny frames, out-of-envelope features). From
  that seed the controller brackets the target by extrapolating the measured
  loss slope, aims at the log-loss crossing, and then attaches entropy coding
  to only the coarsest qualifying candidates and keeps the smallest exact
  stream. Probes reuse the first candidate's cover and chroma-from-luma; a
  finalist far from that anchor is re-planned fresh and re-scored. Every
  emission is verified at full resolution against the canonical scorer before
  it is reported.
- **Budgets.** Fast: at most 3 scored probes and 2 exact prices. Balanced: 5
  and 3. These are hard caps; there is no hidden exhaustive fallback.
- **Reporting.** Every perceptual encode prints one line,
  `quality_target=85.0000 achieved=85.1372 bytes=… metric=ssimulacra2-jpxl-1
  effort=balanced probes=3 prices=2 status=met`. Statuses are explicit:
  `met`, `met_work_cap` (target met, budget stopped the tightening),
  `met_adjacent_rungs`, `rescued_fresh_structure`, `saturated_top` (even the
  finest quantizer misses the target; reported, never silent),
  `under_target_work_cap`, `saturated_floor`, `routed_to_lossless` (score 100)
  and `unsupported_too_small` (below the metric's 8×8 floor). Setting
  `JPXL_QUALITY_TRACE=<path>` appends a machine-readable
  `jpxl.quality-trace/2` record per encode: source features, the predicted
  rung, every probe's quantizer/score/bytes, and wall time by phase.
- **Determinism.** The same input produces the same codestream across worker
  counts: the metric reduces fixed-size row bands in fixed order, the renderer
  bands rows without cross-band reductions, and the cube root and blur use
  host-independent arithmetic.

On the development split of the quality-guard corpus (15 images × 5 targets,
2026-08-22) the controller met the requested score in 150 of 150 encodes, with
a median overshoot of 0.9 points (Balanced) and 2.2 (Fast); achieved score is
monotone in the request on every image larger than 8×8. At matched achieved
score JPXL is smaller than `cjxl -e 7` on photographs, gradients, grayscale
and noisy content (BD-rate −4% to −20%) and larger on synthetic text and line
art (up to +135%), which is a VarDCT-on-synthetic-content gap rather than a
controller one. The probing costs about 3.5× the wall time of a fixed-rate
encode at the same score (4 MP, four threads: `--quality 85` in 4.2 s against
0.46 s); reducing that overhead is the next performance target.
`JPXL/tools/codec_compare.py curve --quality …` reproduces the measurement.

## Rust API

Applications should depend on the `jpxl` facade crate. It keeps file-format
dependencies out of the library path and presents the encoder as ordinary
interleaved pixel buffers:

```rust
use jpxl::{Effort, Encoder};

fn encode_generated(width: u32, height: u32, rgb: &[u8]) -> jpxl::Result<Vec<u8>> {
    // Exact-lossless by default.
    let lossless = Encoder::new().encode_rgb8(width, height, rgb)?;
    let decoded = jpxl::decode(&lossless)?;
    assert_eq!((decoded.width, decoded.height), (width, height));

    // The normal lossy contract: a minimum SSIMULACRA2 quality, verified on
    // the reconstructed pixels; 100 routes to the lossless path. Expert
    // modes `with_target_bpp` / `with_global_scale` pin a size or quantizer.
    Encoder::new()
        .with_ssimulacra2_score(85.0)?
        .with_effort(Effort::Balanced)
        .encode_rgb8(width, height, rgb)
}
```

The same builder accepts RGB16, greyscale 8/16-bit, explicit thread limits,
Part 2 containers, target byte counts, and custom decoder resource limits.
Low-level `jpxl-decode`, `jpxl-encode`, and `jpxl-encode-policy` crates remain
available for callers that need syntax-level or research controls.

## Quality, density, and speed versus libjxl

The project has two distinct comparison modes. They must not be conflated.

- **Speed-parity window:** `Balanced`/`Fast` JPXL and `cjxl -e 7`, with the
  same four pinned P-cores and SSIMULACRA2-matched outputs. The Phase 42
  result recorded 0.42–0.43 s JPXL Balanced versus 0.49–0.51 s `cjxl` on
  2400×1800, and 1.05–1.43 s versus 1.51–1.66 s on 4000×3000. `Fast` was
  quicker still. These runs were about 13% larger and Butteraugli still
  favoured `cjxl`, so “speed parity” is not “codec parity.”
- **Quality/density curves:** JPXL targets a byte rate, while `cjxl -d`
  targets Butteraugli. Sweep each curve and compare rate, PSNR,
  SSIMULACRA2, Butteraugli, and output size; a same-setting comparison is not
  a quality claim. The exhaustive JPXL `Quality` preset is deliberately much
  slower and is not the speed-parity path.

Two reproduction entry points exist. The portable, dependency-light one is
`JPXL/tools/bench_vs_libjxl.sh`: given a corpus and the oracle binaries from
`tools/setup-oracles.sh`, it emits a `bytes / bpp / SSIMULACRA2 / wall` row per
image and setting for both encoders — every stream decoded and scored with the
*same* in-tree production SSIMULACRA2 — behind a provenance header (UTC date,
host, each binary's version and sha256, the exact flags, per-input hashes and
dimensions), with optional `--jsonl` output. It grades JPXL alone, clearly
marked, when no runnable oracle is present on the host.

```sh
cd JPXL
# JPXL quality targets vs cjxl distances, one identical PPM corpus, 4 threads:
tools/bench_vs_libjxl.sh --quality "70 85 90" --distance "3.0 1.5 1.0" \
  --effort balanced --threads 4 --runs 3 --jsonl bench.jsonl <corpus-dir>
```

The fuller, tracked harness (`tools/codec_compare.py`, and the Windows-native
`tools/compare-libjxl.ps1`) alternates the encoders on identical P6 PPM inputs,
uses an explicit equal thread count, decodes both streams with `djxl`, and
records provenance, hashes, bitrate, quality metrics, and timing dispersion.

```powershell
cd JPXL
pwsh ./tools/compare-libjxl.ps1 `
  -Input ..\.agent\scratch\quality-track\q2-inputs\mid-photo.ppm `
  -JpxlBpp 1.0 -CjxlDistance 2.25 `
  -JpxlPreset balanced -Threads 4 -CjxlThreads 4 -CjxlEffort 7 -Runs 3
```

For the 4000×3000 anchor, rerun with `large-photo.ppm` and
`-CjxlDistance 1.25`.

The measured parity window and the correction to the earlier mismatched
comparison are recorded in
[JPXL/docs/experiments/2026-08-18-speed-parity-reconciliation.md](JPXL/docs/experiments/2026-08-18-speed-parity-reconciliation.md).

At matched bytes on the three standing photographs at 0.5, 1 and 2 bpp
(`Balanced` against `cjxl -e 7`, 2026-08-18), JPXL is ahead on SSIMULACRA2 in
all nine cells (+0.4 to +2.6), behind on PSNR in eight (by 0.03–0.54 dB), and
behind on Butteraugli in seven of nine on both the max-norm (up to 43%) and
the 3-norm (up to 14%). The per-cell table, its localisation, why the metrics
diverge, and the levers that did and did not close the gap are in the Phase
Q3 and Q4 sections of [JPXL/docs/optimize.md](JPXL/docs/optimize.md).
Raw artifacts stay in `.agent/scratch/` so reports remain reviewable without
committing test images or generated streams.

## Design commitments

- Clean-room and standard-first: ISO/IEC 18181 is authoritative. `libjxl` is
  used only as a black-box oracle; its source is not used as architecture or
  implementation guidance.
- Decoder first: every encoder layer is validated against independent decoding.
- Safe parser boundaries: checked arithmetic, allocation limits, typed errors,
  and no `unsafe` code beyond the documented runtime CPU-feature dispatch
  call sites in `jpxl-core`.
- Traceable bitstreams: bit-position tracing exists before field parsing and is
  feature-gated away when disabled.
- Honest coverage: unsupported features are named and rejected, and benchmark
  claims retain the commands, inputs, hashes, and host context that produced
  them.

## License

JPXL is licensed under the [MIT License](LICENSE) (mirrored at
[JPXL/LICENSE-MIT](JPXL/LICENSE-MIT)). Every workspace crate declares MIT.

Every third-party dependency is permissive and MIT-compatible, and the
dependency graph contains no copyleft (no GPL/LGPL/AGPL/MPL/CDDL). The
full crate-by-crate audit — including the handful of dependencies that offer
BSD-2/3-Clause rather than MIT, and the measurement-only metrics that stay off
by default — is in
[JPXL/docs/LICENSING-AUDIT.md](JPXL/docs/LICENSING-AUDIT.md). The gitignored
libjxl oracle checkout (BSD-3-Clause), the ISO/IEC standards documents, and the
local test images are working material only: they are not part of the build and
are not distributed.

JPEG XL may be subject to patent claims; this repository’s software license
does not provide patent advice or a patent grant.
