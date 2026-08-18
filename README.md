# JPXL

JPXL is a clean-room, native Rust implementation of JPEG XL (ISO/IEC 18181).
It is a correctness-first decoder and encoder workspace: the decoder follows
the standard, and the encoder is tested against independent decoders rather
than accepted merely because it round-trips through itself.

**Status:** actively developed and not yet a drop-in replacement for libjxl.
The implemented subset is substantial, but it deliberately rejects several
valid JPEG XL feature combinations instead of guessing. The lossy encoder is
also currently much slower and less rate-distortion efficient than libjxl;
see the measured comparison below.

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

## Quality and speed versus libjxl

JPXL and `cjxl` expose different encoder controls: JPXL targets a byte rate,
whereas `cjxl -d` targets Butteraugli distance. The meaningful comparison is
therefore a rate/distortion curve, not a same-setting shootout. The tracked
harness alternates timed JPXL and `cjxl` runs on identical P6 PPM inputs,
decodes both streams with `djxl`, and reports file size, PSNR, SSIMULACRA2,
Butteraugli, hashes, and timing dispersion.

```powershell
cd JPXL
pwsh ./tools/compare-libjxl.ps1 `
  -Input ..\.agent\scratch\quality-track\q2-inputs\mid-photo.ppm,` 
         ..\.agent\scratch\quality-track\q2-inputs\large-photo.ppm `
  -JpxlBpp 1.0 -CjxlDistance 1.0 -Runs 3
```

The frozen current run, its exact host/oracle/build provenance, and its
interpretation are recorded in
[JPXL/docs/experiments/2026-08-18-libjxl-comparison.md](JPXL/docs/experiments/2026-08-18-libjxl-comparison.md).
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
