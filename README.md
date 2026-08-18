# JPXL

JPXL is a clean-room, native Rust implementation of JPEG XL (ISO/IEC 18181).
It is a correctness-first decoder and encoder workspace: the decoder follows
the standard, and the encoder is tested against independent decoders rather
than accepted merely because it round-trips through itself.

**Status:** actively developed and not yet a drop-in replacement for libjxl.
The implemented subset is substantial, but it deliberately rejects several
valid JPEG XL feature combinations instead of guessing. Its optimized lossy
`Balanced` and `Fast` paths have reached a matched-SSIMULACRA2 speed-parity
window against the pinned libjxl build on the project’s two photo anchors;
quality and density still trail libjxl on important perceptual axes.

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

The public CLI currently accepts binary PGM/PPM input and writes binary
PGM/PPM output. It is a reference-oriented tool, not a polished end-user image
converter.

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
jpxl encode input.ppm output.jxl

# Lossy VarDCT encode to a byte rate
jpxl encode --bpp 1.0 input.ppm output.jxl

# Decode and inspect
jpxl decode output.jxl decoded.ppm
jpxl info output.jxl
```

Run `jpxl --help` and `jpxl bench --help` for the supported options and
isolated encoder timing modes.

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
