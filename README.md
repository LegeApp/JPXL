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
- Lossy encoding: RGB8 VarDCT with square DCT 8/16/32 transforms, CfL,
  adaptive quantization, entropy clustering, and a byte-targeted rate loop.
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

# Lossy VarDCT encode to a byte rate
jpxl encode --bpp 1.0 input.jpg output.jxl

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

Target-rate encoding defaults to the production `Balanced` preset. Use
`--lossy-preset fast` when lower latency matters more than the extra quality,
or `--lossy-preset quality` for the deliberately exhaustive reference path;
`Quality` is not the production speed preset.

## Rust API

Applications should depend on the `jpxl` facade crate. It keeps file-format
dependencies out of the library path and presents the encoder as ordinary
interleaved pixel buffers:

```rust
use jpxl::{Encoder, Preset};

fn encode_generated(width: u32, height: u32, rgb: &[u8]) -> jpxl::Result<Vec<u8>> {
    // Exact-lossless by default.
    let lossless = Encoder::new().encode_rgb8(width, height, rgb)?;
    let decoded = jpxl::decode(&lossless)?;
    assert_eq!((decoded.width, decoded.height), (width, height));

    // Production target-rate encoding. This is a byte ceiling, not a
    // libjxl-style perceptual-distance promise.
    Encoder::new()
        .with_target_bpp(1.0)?
        .with_preset(Preset::Balanced)
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

The tracked harness alternates the encoders on identical P6 PPM inputs,
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
  and no `unsafe` code in the workspace.
- Traceable bitstreams: bit-position tracing exists before field parsing and is
  feature-gated away when disabled.
- Honest coverage: unsupported features are named and rejected, and benchmark
  claims retain the commands, inputs, hashes, and host context that produced
  them.

## License

JPXL is licensed under the [MIT License](LICENSE).

JPEG XL may be subject to patent claims; this repository’s software license
does not provide patent advice or a patent grant.
