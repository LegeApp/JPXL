# jpxl

High-level Rust API for [JPXL](https://github.com/LegeApp/JPXL), a clean-room
implementation of JPEG XL (ISO/IEC 18181) with no C dependencies.

The facade takes ordinary interleaved pixel buffers, supplies safe decoder
limits, and hides the policy/writer split inside the encoder.

```rust
use jpxl::{Effort, Encoder};

// Exact-lossless by default.
let rgb = vec![128u8; 64 * 64 * 3];
let lossless = Encoder::new().encode_rgb8(64, 64, &rgb)?;
let decoded = jpxl::decode(&lossless)?;

// Lossy: name a minimum quality and let the encoder find the bytes. The
// score is SSIMULACRA2, verified on the reconstructed pixels, and it is a
// hard floor — a successful encode has actually been scored at or above it.
let lossy = Encoder::new()
    .with_quality(85.0)?
    .with_effort(Effort::Balanced)
    .encode_rgb8(64, 64, &rgb)?;
```

`--quality` is a minimum score in `0..=100` (100 = lossless), not a distance:
`cjxl -d` targets Butteraugli, a different and inverted scale.

The builder also accepts RGB16, greyscale 8/16-bit, explicit thread limits,
Part 2 containers, Exif, target byte counts, and custom decoder limits. The
lower-level `jpxl-decode`, `jpxl-encode` and `jpxl-encode-policy` crates stay
public for callers that need syntax-level or research controls.

## Status

Ready for application integration within the supported feature set, not a
drop-in replacement for every libjxl feature. The decoder returns a typed
`Unsupported` error for syntax it does not implement rather than guessing, and
the encoder does not yet write alpha. See the
[repository README](https://github.com/LegeApp/JPXL#readme) and
`JPXL/docs/CONFORMANCE.md` for the test-backed feature matrix.

## License

MIT.
