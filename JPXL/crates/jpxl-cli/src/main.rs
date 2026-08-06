//! `jpxl` — command-line front end for the JPXL JPEG XL codec.
//!
//! Argument parsing is hand-rolled: the workspace takes no external
//! dependencies, so there is no `clap` here and there will not be one.
//!
//! # Exit codes
//!
//! | Code | Meaning |
//! |------|---------|
//! | 0 | Success; for `info`, the file was recognised as JPEG XL |
//! | 1 | I/O, usage, or decode error |
//! | 2 | `info`: the file is not JPEG XL |

use std::io::Write as _;
use std::path::Path;
use std::process::ExitCode;

use jpxl_conformance::sniff;
use jpxl_core::limits::Limits;

/// Everything went as asked.
const EXIT_OK: u8 = 0;
/// The file could not be read, or the command line made no sense.
const EXIT_ERROR: u8 = 1;
/// The file was read but is not a JPEG XL stream.
const EXIT_UNRECOGNIZED: u8 = 2;

const USAGE: &str = "\
jpxl — JPEG XL codec (JPXL)

Usage:
    jpxl info <file>              Identify a file and print its stream kind
    jpxl boxes <file.jxl>         List the Part 2 box structure of a container
    jpxl decode <in.jxl> <out>    Decode to a binary PGM (P5) or PPM (P6)
    jpxl encode [opts] <in> <out> Encode a binary PGM (P5) or PPM (P6) losslessly
    jpxl bench <mode> [opts]      Time one encode path (see `jpxl bench --help`)
    jpxl --help                   Show this message
    jpxl --version                Show the version

Encode options:
    --container                   Wrap the codestream in a Part 2 container
    --group-size-shift <0..3>     Force group_dim = 128 << shift (default 2)
    --jxlp <bytes>                Split the codestream across jxlp boxes
                                  (18181-2 9.10); implies --container
    --threads <n>                 Section-parallel workers (default: host
                                  available_parallelism; 1 = serial)

Exit codes:
    0  success (info: recognised as JPEG XL)
    1  I/O, usage, or codec error
    2  info: not a JPEG XL stream

`decode` picks P5 for a one-channel image and P6 for three, and writes
16-bit big-endian samples when the bit depth exceeds 8, as Netpbm requires.

`encode` accepts P5 and P6 with maxval 255 or 65535 and writes a lossless
modular codestream: greyscale as-is, RGB through the reversible colour
transform, split into groups when the image exceeds one group.

`bench` isolates Modular lossless, VarDCT fixed-quantizer, VarDCT target-rate,
and a single VarDCT probe so flamegraphs are not mixed across paths.
";

const BENCH_USAGE: &str = "\
jpxl bench <mode> [options]

Modes (Opt-F measurement entry points):
    modular         Lossless Modular encode (same path as `jpxl encode`)
    vardct-fixed    VarDCT at request quantizer scalars (no rate loop)
    vardct-rate     VarDCT rate loop to a bits-per-pixel target
    vardct-probe    Exactly one VarDCT plan+emit at the default quantizer
                    (separates one-encode cost from multi-probe rate search)

Options:
    --width <n>           Synthetic frame width (default 256)
    --height <n>          Synthetic frame height (default 256)
    --iters <n>           Timed iterations after one warm-up (default 3)
    --bpp <f>             Target bits/pixel for vardct-rate (default 1.0)
    --threads <n>         Section-parallel workers (default: auto; 1 = serial)
    --input <path.ppm>    Use a real P6 image instead of the synthetic RGB

Prints one line per run: mode, size, iters, wall_ms_total, wall_ms_median,
output_bytes, output_sha256_prefix. Timing excludes decode validation.

This is tooling, not a baseline: see JPXL/docs/PERFORMANCE.md for the rules
that turn a run into a published number.
";

fn main() -> ExitCode {
    // Skip argv[0]; it is the program path, not an argument.
    let args: Vec<String> = std::env::args().skip(1).collect();
    ExitCode::from(run(&args))
}

fn run(args: &[String]) -> u8 {
    let Some((command, rest)) = args.split_first() else {
        print!("{USAGE}");
        return EXIT_ERROR;
    };

    match command.as_str() {
        "-h" | "--help" | "help" => {
            print!("{USAGE}");
            EXIT_OK
        }
        "-V" | "--version" | "version" => {
            println!("jpxl {}", env!("CARGO_PKG_VERSION"));
            EXIT_OK
        }
        "info" => cmd_info(rest),
        "boxes" => cmd_boxes(rest),
        "decode" => cmd_decode(rest),
        "encode" => cmd_encode(rest),
        "bench" => cmd_bench(rest),
        other => {
            fail(&format!("unknown command `{other}`"));
            EXIT_ERROR
        }
    }
}

/// `jpxl info <file>`: classify a file by its signature.
fn cmd_info(args: &[String]) -> u8 {
    let [path] = args else {
        fail(if args.is_empty() {
            "`info` needs a file argument"
        } else {
            "`info` takes exactly one file argument"
        });
        return EXIT_ERROR;
    };

    let path = Path::new(path);
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(err) => {
            fail(&format!("{}: {err}", path.display()));
            return EXIT_ERROR;
        }
    };

    let kind = sniff(&bytes);
    println!("file:  {}", path.display());
    println!("kind:  {}", kind.label());
    println!("size:  {} bytes", bytes.len());

    if kind.is_recognized() {
        EXIT_OK
    } else {
        EXIT_UNRECOGNIZED
    }
}

/// `jpxl decode <in.jxl> <out.pgm|out.ppm>`: decode to a binary Netpbm file.
fn cmd_decode(args: &[String]) -> u8 {
    let [input, output] = args else {
        fail("`decode` takes an input and an output path");
        return EXIT_ERROR;
    };

    let bytes = match std::fs::read(Path::new(input)) {
        Ok(bytes) => bytes,
        Err(err) => {
            fail(&format!("{input}: {err}"));
            return EXIT_ERROR;
        }
    };

    let image = match jpxl_decode::decode(&bytes, &Limits::default()) {
        Ok(image) => image,
        Err(err) => {
            fail(&format!("{input}: {err}"));
            return EXIT_ERROR;
        }
    };

    match std::fs::write(Path::new(output), encode_netpbm(&image)) {
        Ok(()) => {
            println!(
                "{output}: {}x{}, {} channel(s), {} bits per sample",
                image.width,
                image.height,
                image.num_colour_channels,
                image.colour_bits_per_sample()
            );
            EXIT_OK
        }
        Err(err) => {
            fail(&format!("{output}: {err}"));
            EXIT_ERROR
        }
    }
}

/// `jpxl boxes <file.jxl>`: list the Part 2 box structure.
///
/// One line per box — offset, type, total size, payload size — then the level
/// and whether the file satisfies the clause-9 "shall" requirements. The
/// listing is what a box-lister oracle can be diffed against; the validation
/// verdict is printed rather than turned into an exit code, because a
/// decodable file that breaks a "shall" is still worth listing.
fn cmd_boxes(args: &[String]) -> u8 {
    let [path] = args else {
        fail("`boxes` takes exactly one file argument");
        return EXIT_ERROR;
    };

    let bytes = match std::fs::read(Path::new(path)) {
        Ok(bytes) => bytes,
        Err(err) => {
            fail(&format!("{path}: {err}"));
            return EXIT_ERROR;
        }
    };
    if !jpxl_decode::container::is_container(&bytes) {
        println!("{path}: not a container (naked codestream or unknown)");
        return EXIT_UNRECOGNIZED;
    }

    let limits = Limits::default();
    let mut guard = jpxl_core::limits::AllocGuard::new(&limits);
    let tree = match jpxl_decode::container::BoxTree::parse(&bytes, &mut guard) {
        Ok(tree) => tree,
        Err(err) => {
            fail(&format!("{path}: {err}"));
            return EXIT_ERROR;
        }
    };

    for b in tree.boxes() {
        let name = core::str::from_utf8(&b.type_code).unwrap_or("????");
        println!(
            "{:>10}  {name}  size {:>10}  payload {:>10}  {}",
            b.offset,
            b.total_len(),
            b.payload.len(),
            b.kind.clause()
        );
    }
    match tree.level() {
        Ok(level) => println!("level: {level}"),
        Err(err) => println!("level: {err}"),
    }
    match tree.validate() {
        Ok(()) => println!("clause 9: conforming"),
        Err(err) => println!("clause 9: {err}"),
    }
    EXIT_OK
}

/// `jpxl encode [opts] <in.pgm|in.ppm> <out.jxl>`: encode losslessly.
fn cmd_encode(args: &[String]) -> u8 {
    let mut options = jpxl_encode::EncodeOptions::default();
    let mut positional: Vec<&String> = Vec::new();
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--container" => options.container = true,
            "--group-size-shift" => {
                let Some(value) = rest.next().and_then(|v| v.parse::<u32>().ok()) else {
                    fail("`--group-size-shift` needs a number in 0..=3");
                    return EXIT_ERROR;
                };
                options.group_size_shift = Some(value);
            }
            "--jxlp" => {
                let Some(value) = rest.next().and_then(|v| v.parse::<usize>().ok()) else {
                    fail("`--jxlp` needs a fragment size in bytes");
                    return EXIT_ERROR;
                };
                options.jxlp_fragment_size = Some(value);
            }
            "--threads" => {
                let Some(value) = rest.next().and_then(|v| v.parse::<usize>().ok()) else {
                    fail("`--threads` needs a positive integer (1 = serial)");
                    return EXIT_ERROR;
                };
                options.resources = if value <= 1 {
                    jpxl_encode::EncodeResources::serial()
                } else {
                    jpxl_encode::EncodeResources::groups(value)
                };
            }
            other if other.starts_with("--") => {
                fail(&format!("unknown `encode` option `{other}`"));
                return EXIT_ERROR;
            }
            _ => positional.push(arg),
        }
    }
    let [input, output] = positional.as_slice() else {
        fail("`encode` takes an input and an output path");
        return EXIT_ERROR;
    };

    let bytes = match std::fs::read(Path::new(input.as_str())) {
        Ok(bytes) => bytes,
        Err(err) => {
            fail(&format!("{input}: {err}"));
            return EXIT_ERROR;
        }
    };

    let image = match decode_netpbm(&bytes) {
        Ok(image) => image,
        Err(err) => {
            fail(&format!("{input}: {err}"));
            return EXIT_ERROR;
        }
    };

    let encoded = match jpxl_encode::encode(&image, &options) {
        Ok(encoded) => encoded,
        Err(err) => {
            fail(&format!("{input}: {err}"));
            return EXIT_ERROR;
        }
    };

    match std::fs::write(Path::new(output.as_str()), &encoded) {
        Ok(()) => {
            println!(
                "{output}: {}x{}, {} channel(s), {} bits per sample, {} bytes",
                image.width(),
                image.height(),
                image.num_channels(),
                image.bits_per_sample(),
                encoded.len()
            );
            EXIT_OK
        }
        Err(err) => {
            fail(&format!("{output}: {err}"));
            EXIT_ERROR
        }
    }
}

/// `jpxl bench <mode> [opts]`: time one isolated encode path.
fn cmd_bench(args: &[String]) -> u8 {
    let Some(first) = args.first() else {
        print!("{BENCH_USAGE}");
        return EXIT_ERROR;
    };
    if matches!(first.as_str(), "-h" | "--help" | "help") {
        print!("{BENCH_USAGE}");
        return EXIT_OK;
    }

    let mode = first.as_str();
    let mut width = 256u32;
    let mut height = 256u32;
    let mut iters = 3usize;
    let mut bpp = 1.0f64;
    let mut input: Option<&str> = None;
    let mut resources = jpxl_encode::EncodeResources::auto();

    let mut rest = args.get(1..).unwrap_or(&[]).iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--width" => {
                let Some(v) = rest.next().and_then(|s| s.parse().ok()) else {
                    fail("`--width` needs a positive integer");
                    return EXIT_ERROR;
                };
                width = v;
            }
            "--height" => {
                let Some(v) = rest.next().and_then(|s| s.parse().ok()) else {
                    fail("`--height` needs a positive integer");
                    return EXIT_ERROR;
                };
                height = v;
            }
            "--iters" => {
                let Some(v) = rest.next().and_then(|s| s.parse().ok()).filter(|&n| n > 0) else {
                    fail("`--iters` needs a positive integer");
                    return EXIT_ERROR;
                };
                iters = v;
            }
            "--bpp" => {
                let Some(v) = rest.next().and_then(|s| s.parse().ok()) else {
                    fail("`--bpp` needs a floating-point number");
                    return EXIT_ERROR;
                };
                bpp = v;
            }
            "--threads" => {
                let Some(v) = rest.next().and_then(|s| s.parse::<usize>().ok()) else {
                    fail("`--threads` needs a positive integer (1 = serial)");
                    return EXIT_ERROR;
                };
                resources = if v <= 1 {
                    jpxl_encode::EncodeResources::serial()
                } else {
                    jpxl_encode::EncodeResources::groups(v)
                };
            }
            "--input" => {
                let Some(path) = rest.next() else {
                    fail("`--input` needs a path");
                    return EXIT_ERROR;
                };
                input = Some(path.as_str());
            }
            other => {
                fail(&format!("unknown `bench` option `{other}`"));
                return EXIT_ERROR;
            }
        }
    }

    if width == 0 || height == 0 {
        fail("width and height must be non-zero");
        return EXIT_ERROR;
    }

    let rgb = match input {
        Some(path) => match load_rgb8_ppm(path) {
            Ok((w, h, bytes)) => {
                width = w;
                height = h;
                bytes
            }
            Err(err) => {
                fail(&err);
                return EXIT_ERROR;
            }
        },
        None => synthetic_rgb8(width, height),
    };

    let timed = match mode {
        "modular" => bench_modular(&rgb, width, height, iters, resources),
        "vardct-fixed" => bench_vardct_fixed(&rgb, width, height, iters, resources),
        "vardct-rate" => bench_vardct_rate(&rgb, width, height, bpp, iters, resources),
        "vardct-probe" => bench_vardct_probe(&rgb, width, height, iters, resources),
        other => {
            fail(&format!(
                "unknown bench mode `{other}` (modular|vardct-fixed|vardct-rate|vardct-probe)"
            ));
            return EXIT_ERROR;
        }
    };

    match timed {
        Ok(report) => {
            println!(
                "mode={mode} size={width}x{height} iters={iters} \
                 wall_ms_total={:.3} wall_ms_median={:.3} \
                 output_bytes={} fingerprint={:016x}",
                report.wall_ms_total,
                report.wall_ms_median,
                report.output_bytes,
                report.fingerprint
            );
            EXIT_OK
        }
        Err(err) => {
            fail(&err);
            EXIT_ERROR
        }
    }
}

struct BenchReport {
    wall_ms_total: f64,
    wall_ms_median: f64,
    output_bytes: usize,
    fingerprint: u64,
}

fn bench_modular(
    rgb: &[u8],
    width: u32,
    height: u32,
    iters: usize,
    resources: jpxl_encode::EncodeResources,
) -> Result<BenchReport, String> {
    let samples: Vec<u16> = rgb.iter().map(|&b| u16::from(b)).collect();
    let image = jpxl_encode::Image::from_interleaved(width, height, 3, 8, &samples)
        .map_err(|e| e.to_string())?;
    let mut options = jpxl_encode::EncodeOptions::default();
    options.resources = resources;
    // Warm-up (not timed).
    let warm = jpxl_encode::encode(&image, &options).map_err(|e| e.to_string())?;
    time_iters(iters, warm.len(), fnv1a64(&warm), || {
        jpxl_encode::encode(&image, &options).map_err(|e| e.to_string())
    })
}

fn bench_vardct_fixed(
    rgb: &[u8],
    width: u32,
    height: u32,
    iters: usize,
    resources: jpxl_encode::EncodeResources,
) -> Result<BenchReport, String> {
    let mut request = jpxl_encode_policy::EncodeRequest::defaults();
    request.resources = resources;
    let warm = jpxl_encode_policy::encode_srgb8_vardct(width, height, rgb, &request)
        .map_err(|e| e.to_string())?;
    time_iters(iters, warm.len(), fnv1a64(&warm), || {
        jpxl_encode_policy::encode_srgb8_vardct(width, height, rgb, &request)
            .map_err(|e| e.to_string())
    })
}

fn bench_vardct_rate(
    rgb: &[u8],
    width: u32,
    height: u32,
    bpp: f64,
    iters: usize,
    resources: jpxl_encode::EncodeResources,
) -> Result<BenchReport, String> {
    let mut request = jpxl_encode_policy::EncodeRequest::defaults();
    request.resources = resources;
    let target = jpxl_encode_policy::RateTarget::BitsPerPixel(bpp);
    request.target = Some(target);
    let warm = jpxl_encode_policy::encode_srgb8_to_target(width, height, rgb, &request, target)
        .map_err(|e| e.to_string())?;
    let warm_len = warm.codestream.len();
    let warm_fp = fnv1a64(&warm.codestream);
    time_iters(iters, warm_len, warm_fp, || {
        jpxl_encode_policy::encode_srgb8_to_target(width, height, rgb, &request, target)
            .map(|o| o.codestream)
            .map_err(|e| e.to_string())
    })
}

fn bench_vardct_probe(
    rgb: &[u8],
    width: u32,
    height: u32,
    iters: usize,
    resources: jpxl_encode::EncodeResources,
) -> Result<BenchReport, String> {
    // One predetermined quantizer: plan_frame without a rate target is a single
    // plan_at + emit — the fourth measurement the advisor asked for.
    let mut request = jpxl_encode_policy::EncodeRequest::defaults();
    request.resources = resources;
    let frame = jpxl_encode_policy::PreparedFrame::from_srgb8(width, height, rgb)
        .map_err(|e| e.to_string())?;
    let warm_plan = jpxl_encode_policy::plan_frame(&frame, &request).map_err(|e| e.to_string())?;
    let warm = jpxl_encode::vardct::write_codestream_with(&warm_plan, resources)
        .map_err(|e| e.to_string())?;
    time_iters(iters, warm.len(), fnv1a64(&warm), || {
        let plan = jpxl_encode_policy::plan_frame(&frame, &request).map_err(|e| e.to_string())?;
        jpxl_encode::vardct::write_codestream_with(&plan, resources).map_err(|e| e.to_string())
    })
}

fn time_iters<F>(
    iters: usize,
    output_bytes: usize,
    fingerprint: u64,
    mut run: F,
) -> Result<BenchReport, String>
where
    F: FnMut() -> Result<Vec<u8>, String>,
{
    let mut samples_ms = Vec::with_capacity(iters);
    let start_all = std::time::Instant::now();
    for _ in 0..iters {
        let t0 = std::time::Instant::now();
        let out = run()?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        if out.len() != output_bytes {
            return Err(format!(
                "output size changed across iterations ({} vs {output_bytes})",
                out.len()
            ));
        }
        if fnv1a64(&out) != fingerprint {
            return Err("output bytes changed across iterations".to_owned());
        }
        samples_ms.push(ms);
    }
    let wall_ms_total = start_all.elapsed().as_secs_f64() * 1000.0;
    samples_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = samples_ms.len() / 2;
    let wall_ms_median = if samples_ms.len().is_multiple_of(2) && samples_ms.len() >= 2 {
        let lo = samples_ms.get(mid - 1).copied().unwrap_or(0.0);
        let hi = samples_ms.get(mid).copied().unwrap_or(0.0);
        (lo + hi) / 2.0
    } else {
        samples_ms.get(mid).copied().unwrap_or(0.0)
    };
    Ok(BenchReport {
        wall_ms_total,
        wall_ms_median,
        output_bytes,
        fingerprint,
    })
}

/// Deterministic synthetic RGB for reproducible benches (not a corpus fixture).
fn synthetic_rgb8(width: u32, height: u32) -> Vec<u8> {
    let n = usize::try_from(u64::from(width) * u64::from(height) * 3).unwrap_or(0);
    let mut out = Vec::with_capacity(n);
    for y in 0..height {
        for x in 0..width {
            let ramp = (x.wrapping_mul(170) / width.max(1)) + (y.wrapping_mul(70) / height.max(1));
            let checker = if (x / 16 + y / 16).is_multiple_of(2) {
                12
            } else {
                0
            };
            let luma = u8::try_from((20 + ramp + checker).min(255)).unwrap_or(255);
            out.extend_from_slice(&[
                luma,
                u8::try_from(u16::from(luma) * 4 / 5).unwrap_or(255),
                u8::try_from(u16::from(luma) * 3 / 5).unwrap_or(255),
            ]);
        }
    }
    out
}

fn load_rgb8_ppm(path: &str) -> Result<(u32, u32, Vec<u8>), String> {
    let bytes = std::fs::read(Path::new(path)).map_err(|e| format!("{path}: {e}"))?;
    let image = decode_netpbm(&bytes)?;
    if image.num_channels() != 3 || image.bits_per_sample() != 8 {
        return Err("bench --input requires 8-bit RGB (P6 maxval 255)".to_owned());
    }
    let mut rgb = Vec::with_capacity(
        usize::try_from(u64::from(image.width()) * u64::from(image.height()) * 3).unwrap_or(0),
    );
    let planes = image.planes();
    let (Some(r), Some(g), Some(b)) = (planes.first(), planes.get(1), planes.get(2)) else {
        return Err("expected three RGB planes".to_owned());
    };
    if r.len() != g.len() || g.len() != b.len() {
        return Err("RGB plane lengths disagree".to_owned());
    }
    for i in 0..r.len() {
        let rv = r.get(i).copied().unwrap_or(0);
        let gv = g.get(i).copied().unwrap_or(0);
        let bv = b.get(i).copied().unwrap_or(0);
        rgb.push(u8::try_from(rv).unwrap_or(0));
        rgb.push(u8::try_from(gv).unwrap_or(0));
        rgb.push(u8::try_from(bv).unwrap_or(0));
    }
    Ok((image.width(), image.height(), rgb))
}

/// FNV-1a 64-bit fingerprint for cross-iteration identity (not a crypto hash).
fn fnv1a64(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// Parses a binary PGM (P5) or PPM (P6) with `maxval` 255 or 65535.
///
/// Hand-rolled to match `encode_netpbm`: the header is three whitespace-
/// separated ASCII tokens after the magic, with `#` comments allowed anywhere
/// in the header, and exactly one whitespace byte between `maxval` and the
/// samples. Netpbm stores samples above `maxval` 255 as **big-endian** 16-bit
/// pairs, which is the opposite of every byte order in the codestream.
fn decode_netpbm(bytes: &[u8]) -> Result<jpxl_encode::Image, String> {
    let (channels, rest) = match bytes.get(..2) {
        Some(b"P5") => (1usize, bytes.get(2..).unwrap_or_default()),
        Some(b"P6") => (3usize, bytes.get(2..).unwrap_or_default()),
        _ => return Err("not a binary PGM (P5) or PPM (P6)".to_owned()),
    };

    let mut cursor = 0usize;
    let mut fields = [0u32; 3];
    for field in &mut fields {
        *field = next_header_token(rest, &mut cursor)?;
    }
    let [width, height, maxval] = fields;
    let bits_per_sample = match maxval {
        255 => 8u32,
        65535 => 16,
        other => {
            return Err(format!(
                "only maxval 255 and 65535 are supported, this file has {other}"
            ));
        }
    };
    let bytes_per_sample = if bits_per_sample > 8 { 2usize } else { 1 };

    let body = rest
        .get(cursor..)
        .ok_or_else(|| "truncated Netpbm body".to_owned())?;
    let count = u64::from(width) * u64::from(height) * channels as u64;
    let needed = count
        .checked_mul(bytes_per_sample as u64)
        .ok_or_else(|| "Netpbm body size overflows".to_owned())?;
    if u64::try_from(body.len()).unwrap_or(u64::MAX) < needed {
        return Err(format!(
            "truncated Netpbm body: {needed} bytes expected, {} present",
            body.len()
        ));
    }

    let count = usize::try_from(count).map_err(|_| "image too large".to_owned())?;
    let mut samples = Vec::with_capacity(count);
    for i in 0..count {
        let value = if bytes_per_sample == 2 {
            let hi = body.get(i * 2).copied().unwrap_or(0);
            let lo = body.get(i * 2 + 1).copied().unwrap_or(0);
            u16::from_be_bytes([hi, lo])
        } else {
            u16::from(body.get(i).copied().unwrap_or(0))
        };
        samples.push(value);
    }

    jpxl_encode::Image::from_interleaved(width, height, channels, bits_per_sample, &samples)
        .map_err(|e| e.to_string())
}

/// Reads one decimal header token, skipping whitespace and `#` comments, and
/// consumes exactly one whitespace byte after it.
fn next_header_token(bytes: &[u8], cursor: &mut usize) -> Result<u32, String> {
    loop {
        match bytes.get(*cursor) {
            Some(b'#') => {
                while !matches!(bytes.get(*cursor), None | Some(b'\n')) {
                    *cursor += 1;
                }
            }
            Some(b) if b.is_ascii_whitespace() => *cursor += 1,
            Some(_) => break,
            None => return Err("truncated PGM header".to_owned()),
        }
    }

    let mut value: u32 = 0;
    let mut digits = 0u32;
    while let Some(&b) = bytes.get(*cursor) {
        if !b.is_ascii_digit() {
            break;
        }
        value = value
            .checked_mul(10)
            .and_then(|v| v.checked_add(u32::from(b - b'0')))
            .ok_or_else(|| "PGM header field overflows".to_owned())?;
        digits += 1;
        *cursor += 1;
    }
    if digits == 0 {
        return Err("malformed PGM header".to_owned());
    }
    // A single whitespace byte separates the header from the samples.
    if bytes.get(*cursor).is_some_and(u8::is_ascii_whitespace) {
        *cursor += 1;
    }
    Ok(value)
}

/// Serialises the colour channels as a binary Netpbm file.
///
/// P5 (greyscale) for one channel, P6 (RGB) for three. Netpbm stores samples
/// above `maxval` 255 as **big-endian** 16-bit pairs, which is the opposite of
/// every other byte order in this codebase, so it is written out explicitly.
#[must_use]
pub fn encode_netpbm(image: &jpxl_decode::DecodedImage) -> Vec<u8> {
    let channels = image.num_colour_channels.max(1);
    let magic = if channels == 1 { "P5" } else { "P6" };
    let bits = image.colour_bits_per_sample().clamp(1, 16);
    let maxval = (1u32 << bits) - 1;

    let samples = image.interleaved_colour();
    let mut out = Vec::with_capacity(samples.len() * 2 + 32);
    out.extend_from_slice(
        format!("{magic}\n{} {}\n{maxval}\n", image.width, image.height).as_bytes(),
    );
    if maxval > 255 {
        for s in samples {
            out.extend_from_slice(&s.to_be_bytes());
        }
    } else {
        for s in samples {
            out.push(u8::try_from(s).unwrap_or(u8::MAX));
        }
    }
    out
}

/// Print an error to stderr, with the usage hint.
fn fail(message: &str) {
    let mut stderr = std::io::stderr().lock();
    let _ = writeln!(stderr, "jpxl: error: {message}");
    let _ = writeln!(stderr, "try `jpxl --help`");
}
