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

use std::io::{Read as _, Write as _};
use std::path::Path;
use std::process::ExitCode;

use jpxl_conformance::sniff;
use jpxl_core::limits::Limits;

mod image_io;

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
    jpxl decode [opts] <in.jxl> <out>
                                  Decode to PNG, JPEG, WebP, TIFF, BMP, GIF,
                                  ICO, TGA, QOI, PGM, or PPM
    jpxl encode [opts] <in> <out> Encode PNG, JPEG, WebP, TIFF, BMP, GIF, ICO,
                                  TGA, QOI, PGM, or PPM; lossless modular by
                                  default, lossy VarDCT with --bpp
    jpxl compare <ref.ppm> <b.ppm>
                                  Print RMSE, PSNR, and the in-tree production
                                  SSIMULACRA2 between two decoded PPMs (plus
                                  independent-reference SSIMULACRA2 and
                                  butteraugli with --features perceptual)
    jpxl analyze-atlas <in> <out.jsonl>
                                  Export the diagnostic AnalysisAtlasV2; this
                                  research command does not affect encoding
    jpxl features <in> [--json] [--transform-summary]
                                  Print the quality controller's frame source
                                  features as one JSON line (calibration tool);
                                  --transform-summary adds the DCT8-derived
                                  transform features
    jpxl quality-ladder --scales s1,s2,... [--effort fast|balanced] [--price]
                        [--threads N] <in>
                                  Build the production quality pixel plan
                                  fresh at each effective scale, score each
                                  canonically, optionally exact-price it, and
                                  print JSONL (oracle-label calibration tool)
    jpxl rate-ladder --scales s1,s2,... [--preset fast|balanced]
                     [--threads N] <in>
                                  Trial-price a fresh anchored rate plan at
                                  each effective scale, exactly as the bounded
                                  controller's first anchor, and print JSONL
                                  (rate-prior calibration tool; no output
                                  file, no Store)
    jpxl bench <mode> [opts]      Time one encode path (see `jpxl bench --help`)
    jpxl --help                   Show this message
    jpxl --version                Show the version

Encode options:
    --background <#RRGGBB>       Explicitly flatten a transparent input;
                                  transparent JPEG XL output is not yet encoded
    --container                   Wrap the codestream in a Part 2 container
    --effort <1..9|mode>          A digit 1..9 is the lossless Modular search
                                  effort (default 1 = fastest); every level is
                                  exact-lossless (pixels identical). A name
                                  (fast|balanced) sets the lossy effort instead
                                  — see the lossy options below.
    --group-size-shift <0..3>     Force group_dim = 128 << shift (default 2)
    --jxlp <bytes>                Split the codestream across jxlp boxes
                                  (18181-2 9.10); implies --container
    --threads <n>                 Section-parallel workers (default: host
                                  available_parallelism; 1 = serial)

Decode options:
    -f, --format <name>           Output format; required for stdout (`-`),
                                  otherwise inferred from the output extension
    --background <#RRGGBB>       Flatten alpha when writing JPEG or PNM

Lossy options (8- or 16-bit RGB; any one selects the VarDCT path):
    --quality [N]                 Minimum SSIMULACRA2 score to hold (0..100,
                                  100 = lossless). Alias: --ssimulacra2. The
                                  number is optional; omitted, it is the
                                  effort's default (fast 70, balanced 85). This
                                  is the normal way to ask for lossy output.
    --lossy                       --quality with the effort's default score.
    --quality-fallback <mode>     What to emit when the bounded search cannot
                                  verify the requested score: lossless (a
                                  mathematically lossless stream) or
                                  best-effort (the finest verified under-target
                                  stream, reported by its true score). Without
                                  this flag such an encode fails, exits 1, and
                                  writes nothing.
    --text-routing                Experimental: on census-sparse text/UI/line-
                                  art sources, also price colour-reduced
                                  lossless Modular candidates and emit the
                                  smallest stream holding the requested score.
    --effort <mode>               Lossy effort fast|balanced (search-latency
                                  budget, also picks the --quality default), or
                                  a digit 1..9 for lossless Modular effort.
    --bpp <f>                     Expert mode: target bits per pixel
    --target-bytes <n>            Expert mode: target output size in bytes
    --global-scale <n>            Expert mode: pinned VarDCT global_scale (works
                                  with --quant-lf); emits an exact quantizer
    --lossy-preset <mode>         Alias for --effort: balanced (default) or fast
    --aq-mode <mode>              Per-block HF allocation: off (target-rate
                                  default), masking, uniform, fine-masking,
                                  fine-uniform, or edge-refine (Phase Q3 fields
                                  on a 1/16-octave HfMul lattice; all screened
                                  negative); research control
    --aq-strength <f>             Activity-field strength; research control
    --aq-clamp <f>                Activity-field clamp; research control
    --aq-chroma <f>               Activity-field chroma weight; research control
    --aq-edge-contrast <f>        edge-refine dead zone in activity octaves
                                  (default 6); research control
    --quant-lf <n>                Hold the LF quantizer at n and disable the
                                  secondary LF fill (target-rate default 4
                                  after Phase Q1; 8 was Phase 5G's); research
                                  control
    --x-qm-scale <0..7>           X-channel QM exponent (Fast neutral 2, Balanced
                                  3, Quality 3; per preset, never per bitrate);
                                  setting it pins the manual chroma policy
    --b-qm-scale <0..7>           B-channel QM exponent (Fast neutral 2, Balanced
                                  3, Quality 5; per preset, never per bitrate);
                                  research chroma-allocation control
    --epf-iters <0..3>            Decoder EPF iteration count (target-rate
                                  default 1); research control
    --epf-sharpness <mode>        EPF sharpness plane: zero (fixed default),
                                  uniform7 (target-rate default), or adaptive
                                  (Phase Q3 activity ramp); research control
    --epf-adaptive <f,k,s>        Adaptive sharpness constants floor,knee,span
                                  (implies --epf-sharpness adaptive); research
                                  control
    --cover-size-penalty <mode>   Cover objective's per-transform distortion
                                  scale: neutral (default) or measured (Phase
                                  6.2's large-transform correction); research
                                  control
    --quantizer-choice <mode>     HF quantizer rule: nearest (fixed-quantizer
                                  default), rate-distortion (Phase 7.0,
                                  rejected), or trailing-truncation (Phase
                                  7.1; target-rate default after Phase 7.2)
    --lambda-scale <f>            Multiplier on the cover/quantizer Lagrange
                                  weight lambda. Fixed-quantizer default 1.0;
                                  target-rate default 4.0 after Phase 7.2.
    --dead-zone-scale <f>         Multiplier on every HF cell's zero threshold
                                  (1.0 = the exact nearest rule; >1 widens the
                                  dead zone); quality-track research control
    --zero-token-bits <f>         Bits the trailing-truncation pass charges per
                                  interior zero token it frees (default 1.0);
                                  quality-track research control
    --tolerance <f>               Undershoot the rate loop may leave, as a
                                  fraction of the target (presets keep their
                                  own floor: Balanced 0.02, Fast 0.03)
    --sections                    After a lossy encode, print where the bytes
                                  went by section kind (headers/TOC, LfGlobal,
                                  LF groups, HfGlobal, pass groups)
    --cover-rate-model <mode>     Cover objective's rate proxy: calibrated
                                  (target-rate default after Phase Q4: measured
                                  per-size scales and fixed bits), legacy (the
                                  Phase Q3 proxy; fixed-quantizer default), or
                                  custom:s8,s16,s32,f8,f16,f32; research control
    --cover-freq-weight <mode>    Cover objective's per-cell frequency weight:
                                  flat (default), csf (Mannos-Sakrison at 60
                                  ppd, rejected by Phase 6.3), or quant-donor
                                  (the standard's own DCT8x8 matrix as a curve);
                                  research control

    --quality is a minimum SSIMULACRA2 score (0..100, 100 = lossless), not a
    distance: cjxl's -d targets butteraugli, a different (and inverted) scale.

Exit codes:
    0  success (info: recognised as JPEG XL)
    1  I/O, usage, or codec error
    2  info: not a JPEG XL stream

Use `-` as an input or output path for pipelines. Binary output goes to stdout
and the human-readable summary moves to stderr.

`encode` preserves 8- or 16-bit greyscale/RGB precision where the input format
provides it. Opaque alpha is discarded; non-opaque alpha must be flattened
explicitly with `--background` so transparency is never lost silently.

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
    --lossy-preset <mode> Rate controller: balanced (default), fast, or quality
    --threads <n>         Section-parallel workers (default: auto; 1 = serial)
    --input <path.ppm>    Use a real P6 image instead of the synthetic RGB
    --effort <1..9>       Modular search effort (default 1; modular mode only)
    --diag                After the timed run, print Phase-0 architecture
                          counters (choose stages, residual scans, plane clones)

Modular search overrides (modular mode only). These sweep the effort ladder's
constants so they can be chosen from evidence instead of guessed. They are not
quality dials: every setting is exact-lossless, so they move the byte count and
the encode time, never the decoded pixels.

    --modular-max-depth <n>       Override the MA-tree depth cap (root = 0)
    --modular-max-leaves <n>      Override the MA-tree leaf/context cap
    --modular-sample-budget <n|full>
                          Samples the cheap ranker may score per candidate,
                          summed across planes. This is what bounds ranking
                          cost independently of frame size; `full` = unbounded.
    --modular-deep-cap <n|full>
                          Frame sample count above which the tree search
                          collapses to one split. `full` retires the collapse
                          (needed to reach depth at all on a real photo).

Prints one line per run: mode, size, iters, wall_ms_total, wall_ms_min,
wall_ms_median, wall_ms_mad, output_bytes, fingerprint. With --diag,
additional stage and amplification lines follow.

Note: vardct-fixed still uses EncodeRequest::defaults (hierarchical cover,
CfL, AQ) — only the rate loop is off. See AKR source
outside-advice-2026-08-06 §2.

This is tooling, not a baseline: AKR performance-baseline-rules govern
promoted numbers; raw logs live under .agent/scratch/.
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

    // Every ordinary subcommand accepts the conventional help spelling.
    // `bench` has a dedicated, longer help page and handles its own flag.
    if command != "bench" && matches!(rest, [flag] if flag == "-h" || flag == "--help") {
        print!("{USAGE}");
        return EXIT_OK;
    }

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
        "compare" => cmd_compare(rest),
        "analyze-atlas" => cmd_analyze_atlas(rest),
        "features" => cmd_features(rest),
        "quality-ladder" => cmd_quality_ladder(rest),
        "rate-ladder" => cmd_rate_ladder(rest),
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

/// `jpxl decode [opts] <in.jxl> <out>`: decode to a common raster format.
fn cmd_decode(args: &[String]) -> u8 {
    let mut format = None;
    let mut background = None;
    let mut positional = Vec::new();
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "-f" | "--format" => {
                let Some(value) = rest.next() else {
                    fail("`--format` needs an output format name");
                    return EXIT_ERROR;
                };
                match image_io::RasterFormat::parse(value) {
                    Ok(value) => format = Some(value),
                    Err(error) => {
                        fail(&error);
                        return EXIT_ERROR;
                    }
                }
            }
            "--background" => {
                let Some(value) = rest.next() else {
                    fail("`--background` needs #RGB or #RRGGBB");
                    return EXIT_ERROR;
                };
                match image_io::parse_background(value) {
                    Ok(value) => background = Some(value),
                    Err(error) => {
                        fail(&error);
                        return EXIT_ERROR;
                    }
                }
            }
            other if other.starts_with('-') && other != "-" => {
                fail(&format!("unknown `decode` option `{other}`"));
                return EXIT_ERROR;
            }
            _ => positional.push(arg),
        }
    }
    let [input, output] = positional.as_slice() else {
        fail("`decode` takes an input and an output path");
        return EXIT_ERROR;
    };

    let bytes = match read_path(input) {
        Ok(bytes) => bytes,
        Err(err) => {
            fail(&format!("{input}: {err}"));
            return EXIT_ERROR;
        }
    };

    let image = match jpxl::decode(&bytes) {
        Ok(image) => image,
        Err(err) => {
            fail(&format!("{input}: {err}"));
            return EXIT_ERROR;
        }
    };

    let format = match format.or_else(|| image_io::RasterFormat::from_path(output).ok()) {
        Some(format) => format,
        None => {
            fail("cannot infer the output format; pass `--format <name>`");
            return EXIT_ERROR;
        }
    };
    let raster = match image_io::encode_output(&bytes, &image, format, background) {
        Ok(raster) => raster,
        Err(error) => {
            fail(&error);
            return EXIT_ERROR;
        }
    };

    match write_path(output, &raster) {
        Ok(()) => {
            status_line(
                output,
                &format!(
                    "{output}: {}x{}, {} colour channel(s), {} bits per sample",
                    image.width,
                    image.height,
                    image.num_colour_channels,
                    image.colour_bits_per_sample()
                ),
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

/// `jpxl encode [opts] <raster> <out.jxl>`: encode a common raster image.
fn cmd_encode(args: &[String]) -> u8 {
    let mut options = jpxl_encode::EncodeOptions::default();
    let mut rate_target: Option<jpxl_encode_policy::RateTarget> = None;
    let mut rate_preset: Option<jpxl_encode_policy::RateSearchPreset> = None;
    let mut aq_mode: Option<jpxl_encode_policy::AqMode> = None;
    let mut aq_tuning = jpxl_encode_policy::AqTuning::default();
    let mut fixed_quant_lf: Option<u32> = None;
    let mut x_qm_scale: Option<u8> = None;
    let mut b_qm_scale: Option<u8> = None;
    let mut epf_iters: Option<u8> = None;
    let mut epf_sharpness: Option<jpxl_encode_policy::EpfSharpnessMode> = None;
    let mut cover_size_penalty: Option<jpxl_encode_policy::CoverSizePenalty> = None;
    let mut cover_frequency_weight: Option<jpxl_encode_policy::CoverFrequencyWeight> = None;
    let mut cover_rate_model: Option<jpxl_encode_policy::CoverRateModel> = None;
    let mut quantizer_choice: Option<jpxl_encode_policy::QuantizerChoiceMode> = None;
    let mut lambda_scale: Option<f32> = None;
    let mut dead_zone_scale: Option<f32> = None;
    let mut zero_token_bits: Option<f32> = None;
    let mut tolerance: Option<f64> = None;
    let mut background = None;
    let mut sections = false;
    // Perceptual-quality target: outer `Some` selects the quality path, inner
    // `Some(score)` is an explicit score, inner `None` means "use the effort's
    // default score".
    let mut quality: Option<Option<f64>> = None;
    // What `--quality` emits when the bounded search cannot verify the score;
    // `Refuse` (fail, write nothing) unless `--quality-fallback` says otherwise.
    let mut quality_fallback = jpxl::QualityFallback::Refuse;
    // Experimental text/UI routed candidate competition (`--text-routing`).
    let mut text_routing = false;
    // Fixed-quantizer expert mode.
    let mut global_scale: Option<u32> = None;
    // Lossy effort (search-latency budget); also picks the `--quality` default.
    let mut lossy_effort = jpxl::Effort::default();
    let mut positional: Vec<&String> = Vec::new();
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--container" => options.container = true,
            "--background" => {
                let Some(value) = rest.next() else {
                    fail("`--background` needs #RGB or #RRGGBB");
                    return EXIT_ERROR;
                };
                match image_io::parse_background(value) {
                    Ok(value) => background = Some(value),
                    Err(error) => {
                        fail(&error);
                        return EXIT_ERROR;
                    }
                }
            }
            "--bpp" => {
                let Some(v) = rest.next().and_then(|v| v.parse::<f64>().ok()) else {
                    fail("`--bpp` needs a positive bits-per-pixel target");
                    return EXIT_ERROR;
                };
                if !(v.is_finite() && v > 0.0) {
                    fail("`--bpp` needs a positive, finite bits-per-pixel target");
                    return EXIT_ERROR;
                }
                rate_target = Some(jpxl_encode_policy::RateTarget::BitsPerPixel(v));
            }
            "--quality" | "--ssimulacra2" => {
                // The numeric argument is optional: consume the next token only
                // if it parses as a score, otherwise leave it for the parser.
                let mut peek = rest.clone();
                match peek.next().and_then(|v| v.parse::<f64>().ok()) {
                    Some(score) => {
                        if !(score.is_finite() && (0.0..=100.0).contains(&score)) {
                            fail("`--quality` needs a score in 0..=100 (100 = lossless)");
                            return EXIT_ERROR;
                        }
                        quality = Some(Some(score));
                        rest = peek;
                    }
                    None => quality = Some(None),
                }
            }
            "--lossy" => quality = Some(None),
            "--text-routing" => text_routing = true,
            "--quality-fallback" => {
                let Some(mode) = rest.next() else {
                    fail("`--quality-fallback` needs `lossless` or `best-effort`");
                    return EXIT_ERROR;
                };
                quality_fallback = match mode.as_str() {
                    "lossless" => jpxl::QualityFallback::Lossless,
                    "best-effort" => jpxl::QualityFallback::BestEffort,
                    _ => {
                        fail("`--quality-fallback` needs `lossless` or `best-effort`");
                        return EXIT_ERROR;
                    }
                };
            }
            "--global-scale" => {
                let Some(value) = rest.next().and_then(|v| v.parse::<u32>().ok()) else {
                    fail("`--global-scale` needs a positive representable integer");
                    return EXIT_ERROR;
                };
                global_scale = Some(value);
            }
            // Perceptual bit-allocation sweep knobs (lossy only). These shape
            // the adaptive-quantization field: how hard it reacts to activity,
            // how far it may swing, and how much chroma counts. They exist to
            // be measured against `jpxl compare`'s perceptual metrics. Phase 4J
            // found the single-pass masking field harmful. Phase 5G promoted
            // Off for target-rate requests after a twelve-cell corpus gate;
            // the explicit modes remain useful research controls.
            "--aq-mode" => {
                let Some(value) = rest.next() else {
                    fail("`--aq-mode` needs one of: off, masking, uniform, fine-masking");
                    return EXIT_ERROR;
                };
                aq_mode = Some(match value.as_str() {
                    "off" => jpxl_encode_policy::AqMode::Off,
                    "masking" => jpxl_encode_policy::AqMode::Masking,
                    "uniform" => jpxl_encode_policy::AqMode::Uniform,
                    "fine-masking" => jpxl_encode_policy::AqMode::FineMasking,
                    "fine-uniform" => jpxl_encode_policy::AqMode::FineUniform,
                    "edge-refine" => jpxl_encode_policy::AqMode::EdgeRefine,
                    _ => {
                        fail("`--aq-mode` needs one of: off, masking, uniform, fine-masking");
                        return EXIT_ERROR;
                    }
                });
            }
            "--aq-strength" => {
                let Some(v) = rest.next().and_then(|v| v.parse::<f32>().ok()) else {
                    fail("`--aq-strength` needs a number (octaves per octave of activity)");
                    return EXIT_ERROR;
                };
                aq_tuning.strength = v;
            }
            "--aq-clamp" => {
                let Some(v) = rest.next().and_then(|v| v.parse::<f32>().ok()) else {
                    fail("`--aq-clamp` needs a number (octaves)");
                    return EXIT_ERROR;
                };
                aq_tuning.clamp = v;
            }
            "--aq-chroma" => {
                let Some(v) = rest.next().and_then(|v| v.parse::<f32>().ok()) else {
                    fail("`--aq-chroma` needs a number (chroma weight relative to luma)");
                    return EXIT_ERROR;
                };
                aq_tuning.chroma_weight = v;
            }
            "--aq-edge-contrast" => {
                let Some(v) = rest.next().and_then(|v| v.parse::<f32>().ok()) else {
                    fail("`--aq-edge-contrast` needs a number (octaves of activity)");
                    return EXIT_ERROR;
                };
                aq_tuning.edge_contrast = v;
            }
            "--quant-lf" => {
                let Some(value) = rest.next().and_then(|v| v.parse::<u32>().ok()) else {
                    fail("`--quant-lf` needs a positive representable integer");
                    return EXIT_ERROR;
                };
                fixed_quant_lf = Some(value);
            }
            "--x-qm-scale" | "--b-qm-scale" => {
                let Some(value) = rest.next().and_then(|v| v.parse::<u8>().ok()) else {
                    fail(&format!("`{arg}` needs an integer in 0..=7"));
                    return EXIT_ERROR;
                };
                if value > 7 {
                    fail(&format!("`{arg}` needs an integer in 0..=7"));
                    return EXIT_ERROR;
                }
                if arg == "--x-qm-scale" {
                    x_qm_scale = Some(value);
                } else {
                    b_qm_scale = Some(value);
                }
            }
            "--epf-iters" => {
                let Some(value) = rest.next().and_then(|v| v.parse::<u8>().ok()) else {
                    fail("`--epf-iters` needs an integer in 0..=3");
                    return EXIT_ERROR;
                };
                if value > 3 {
                    fail("`--epf-iters` needs an integer in 0..=3");
                    return EXIT_ERROR;
                }
                epf_iters = Some(value);
            }
            "--epf-sharpness" => {
                let Some(value) = rest.next() else {
                    fail("`--epf-sharpness` needs one of: zero, uniform7");
                    return EXIT_ERROR;
                };
                epf_sharpness = Some(match value.as_str() {
                    "zero" => jpxl_encode_policy::EpfSharpnessMode::Zero,
                    "uniform7" => jpxl_encode_policy::EpfSharpnessMode::Uniform7,
                    "adaptive" => jpxl_encode_policy::EpfSharpnessMode::Adaptive(
                        jpxl_encode_policy::AdaptiveSharpness::default(),
                    ),
                    _ => {
                        fail("`--epf-sharpness` needs one of: zero, uniform7, adaptive");
                        return EXIT_ERROR;
                    }
                });
            }
            "--epf-adaptive" => {
                // Research spelling: `--epf-adaptive floor,knee,span`.
                let Some(value) = rest.next() else {
                    fail("`--epf-adaptive` needs `floor,knee,span`");
                    return EXIT_ERROR;
                };
                let parts: Vec<&str> = value.split(',').collect();
                let parsed = match parts.as_slice() {
                    [f, k, s] => match (f.parse::<u8>(), k.parse::<f32>(), s.parse::<f32>()) {
                        (Ok(floor), Ok(knee), Ok(span)) if floor <= 7 => {
                            Some(jpxl_encode_policy::AdaptiveSharpness { floor, knee, span })
                        }
                        _ => None,
                    },
                    _ => None,
                };
                let Some(model) = parsed else {
                    fail("`--epf-adaptive` needs `floor(0..7),knee,span`");
                    return EXIT_ERROR;
                };
                epf_sharpness = Some(jpxl_encode_policy::EpfSharpnessMode::Adaptive(model));
            }
            "--cover-size-penalty" => {
                let Some(value) = rest.next() else {
                    fail("`--cover-size-penalty` needs one of: neutral, measured");
                    return EXIT_ERROR;
                };
                cover_size_penalty = Some(match value.as_str() {
                    "neutral" => jpxl_encode_policy::CoverSizePenalty::Neutral,
                    "measured" => jpxl_encode_policy::CoverSizePenalty::Measured,
                    custom if custom.starts_with("custom:") => {
                        let nums: Vec<f64> = custom["custom:".len()..]
                            .split(',')
                            .filter_map(|n| n.parse().ok())
                            .collect();
                        let [dct16, dct32] = nums.as_slice() else {
                            fail("`--cover-size-penalty custom:<dct16>,<dct32>` needs two numbers");
                            return EXIT_ERROR;
                        };
                        jpxl_encode_policy::CoverSizePenalty::Custom {
                            dct16: *dct16,
                            dct32: *dct32,
                        }
                    }
                    _ => {
                        fail("`--cover-size-penalty` needs one of: neutral, measured");
                        return EXIT_ERROR;
                    }
                });
            }
            "--cover-freq-weight" => {
                let Some(value) = rest.next() else {
                    fail("`--cover-freq-weight` needs one of: flat, csf, quant-donor");
                    return EXIT_ERROR;
                };
                cover_frequency_weight = Some(match value.as_str() {
                    "flat" => jpxl_encode_policy::CoverFrequencyWeight::Flat,
                    "csf" => jpxl_encode_policy::CoverFrequencyWeight::Csf,
                    "quant-donor" => jpxl_encode_policy::CoverFrequencyWeight::QuantDonor,
                    _ => {
                        fail("`--cover-freq-weight` needs one of: flat, csf, quant-donor");
                        return EXIT_ERROR;
                    }
                });
            }
            "--cover-rate-model" => {
                let Some(value) = rest.next() else {
                    fail(
                        "`--cover-rate-model` needs one of: legacy, calibrated, custom:s8,s16,s32,f8,f16,f32",
                    );
                    return EXIT_ERROR;
                };
                cover_rate_model = Some(match value.as_str() {
                    "legacy" => jpxl_encode_policy::CoverRateModel::Legacy,
                    "calibrated" => jpxl_encode_policy::CoverRateModel::Calibrated,
                    custom if custom.starts_with("custom:") => {
                        let nums: Vec<f64> = custom["custom:".len()..]
                            .split(',')
                            .filter_map(|n| n.parse().ok())
                            .collect();
                        let [scale8, scale16, scale32, fixed8, fixed16, fixed32] = nums.as_slice()
                        else {
                            fail(
                                "`--cover-rate-model custom:` needs six numbers s8,s16,s32,f8,f16,f32",
                            );
                            return EXIT_ERROR;
                        };
                        jpxl_encode_policy::CoverRateModel::Custom {
                            scale8: *scale8,
                            scale16: *scale16,
                            scale32: *scale32,
                            fixed8: *fixed8,
                            fixed16: *fixed16,
                            fixed32: *fixed32,
                        }
                    }
                    _ => {
                        fail("`--cover-rate-model` needs one of: legacy, calibrated, custom:...");
                        return EXIT_ERROR;
                    }
                });
            }
            "--quantizer-choice" => {
                let Some(value) = rest.next() else {
                    fail(
                        "`--quantizer-choice` needs one of: nearest, rate-distortion, \
                         trailing-truncation",
                    );
                    return EXIT_ERROR;
                };
                quantizer_choice = Some(match value.as_str() {
                    "nearest" => jpxl_encode_policy::QuantizerChoiceMode::Nearest,
                    "rate-distortion" => jpxl_encode_policy::QuantizerChoiceMode::RateDistortion,
                    "trailing-truncation" => {
                        jpxl_encode_policy::QuantizerChoiceMode::TrailingTruncation
                    }
                    _ => {
                        fail(
                            "`--quantizer-choice` needs one of: nearest, rate-distortion, \
                             trailing-truncation",
                        );
                        return EXIT_ERROR;
                    }
                });
            }
            "--lambda-scale" => {
                let Some(v) = rest.next().and_then(|v| v.parse::<f32>().ok()) else {
                    fail("`--lambda-scale` needs a positive finite multiplier");
                    return EXIT_ERROR;
                };
                if !(v.is_finite() && v > 0.0) {
                    fail("`--lambda-scale` needs a positive finite multiplier");
                    return EXIT_ERROR;
                }
                lambda_scale = Some(v);
            }
            "--dead-zone-scale" => {
                let Some(v) = rest.next().and_then(|v| v.parse::<f32>().ok()) else {
                    fail("`--dead-zone-scale` needs a positive finite multiplier");
                    return EXIT_ERROR;
                };
                if !(v.is_finite() && v > 0.0) {
                    fail("`--dead-zone-scale` needs a positive finite multiplier");
                    return EXIT_ERROR;
                }
                dead_zone_scale = Some(v);
            }
            "--zero-token-bits" => {
                let Some(v) = rest.next().and_then(|v| v.parse::<f32>().ok()) else {
                    fail("`--zero-token-bits` needs a non-negative finite bit count");
                    return EXIT_ERROR;
                };
                if !(v.is_finite() && v >= 0.0) {
                    fail("`--zero-token-bits` needs a non-negative finite bit count");
                    return EXIT_ERROR;
                }
                zero_token_bits = Some(v);
            }
            "--tolerance" => {
                let Some(v) = rest.next().and_then(|v| v.parse::<f64>().ok()) else {
                    fail("`--tolerance` needs a fraction of the target in [0, 1)");
                    return EXIT_ERROR;
                };
                if !(v.is_finite() && (0.0..1.0).contains(&v)) {
                    fail("`--tolerance` needs a fraction of the target in [0, 1)");
                    return EXIT_ERROR;
                }
                tolerance = Some(v);
            }
            "--sections" => sections = true,
            "--target-bytes" => {
                let Some(v) = rest.next().and_then(|v| v.parse::<u64>().ok()) else {
                    fail("`--target-bytes` needs a positive byte count");
                    return EXIT_ERROR;
                };
                rate_target = Some(jpxl_encode_policy::RateTarget::Bytes(v));
            }
            "--lossy-preset" => {
                let Some(value) = rest.next() else {
                    fail("`--lossy-preset` needs one of: balanced, fast");
                    return EXIT_ERROR;
                };
                match parse_lossy_effort(value.as_str()) {
                    Some((effort, preset)) => {
                        lossy_effort = effort;
                        rate_preset = Some(preset);
                    }
                    None => {
                        fail("`--lossy-preset` needs one of: balanced, fast");
                        return EXIT_ERROR;
                    }
                }
            }
            "--effort" => {
                let Some(value) = rest.next() else {
                    fail("`--effort` needs fast, balanced, or an integer in 1..=9");
                    return EXIT_ERROR;
                };
                match parse_lossy_effort(value.as_str()) {
                    Some((effort, preset)) => {
                        // A name sets the lossy effort (and its rate preset).
                        lossy_effort = effort;
                        rate_preset = Some(preset);
                    }
                    None => {
                        // Otherwise a digit is the lossless Modular effort.
                        match value
                            .parse::<u8>()
                            .ok()
                            .and_then(|level| jpxl_encode::Effort::new(level).ok())
                        {
                            Some(effort) => options.effort = effort,
                            None => {
                                fail("`--effort` needs fast, balanced, or an integer in 1..=9");
                                return EXIT_ERROR;
                            }
                        }
                    }
                }
            }
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

    let bytes = match read_path(input) {
        Ok(bytes) => bytes,
        Err(err) => {
            fail(&format!("{input}: {err}"));
            return EXIT_ERROR;
        }
    };

    let image = match image_io::decode_input(&bytes, background) {
        Ok(image) => image,
        Err(err) => {
            fail(&format!("{input}: {err}"));
            return EXIT_ERROR;
        }
    };

    // Exactly one lossy target: --quality (the normal contract), --bpp /
    // --target-bytes (a size), or --global-scale (a pinned quantizer).
    let selectors = u8::from(quality.is_some())
        + u8::from(rate_target.is_some())
        + u8::from(global_scale.is_some());
    if selectors > 1 {
        fail("choose one lossy target: --quality, --bpp/--target-bytes, or --global-scale");
        return EXIT_ERROR;
    }

    // A lossy target selects the VarDCT path; without one this stays the
    // lossless modular encoder it has always been.
    let mut lossy: Option<LossyReport> = None;
    let mut perceptual_line: Option<String> = None;
    let mut mode_label: Option<String> = None;
    let encoded = if let Some(explicit) = quality {
        let score = explicit.unwrap_or_else(|| lossy_effort.default_score());
        match encode_quality(
            &image,
            score,
            &options,
            lossy_effort,
            quality_fallback,
            text_routing,
        ) {
            Ok((bytes, line, mode)) => {
                perceptual_line = Some(line);
                mode_label = Some(mode);
                bytes
            }
            Err(err) => {
                fail(&format!("{input}: {err}"));
                return EXIT_ERROR;
            }
        }
    } else if let Some(gs) = global_scale {
        match encode_global_scale(&image, gs, fixed_quant_lf, &options) {
            Ok((bytes, mode)) => {
                mode_label = Some(mode);
                bytes
            }
            Err(err) => {
                fail(&format!("{input}: {err}"));
                return EXIT_ERROR;
            }
        }
    } else if let Some(target) = rate_target {
        match encode_lossy_to_target(
            &image,
            target,
            LossyOverrides {
                aq_mode,
                aq_tuning,
                fixed_quant_lf,
                x_qm_scale,
                b_qm_scale,
                epf_iters,
                epf_sharpness,
                cover_size_penalty,
                cover_frequency_weight,
                cover_rate_model,
                quantizer_choice,
                lambda_scale,
                dead_zone_scale,
                zero_token_bits,
                tolerance,
                rate_preset,
            },
        ) {
            Ok(report) => {
                let bytes =
                    wrap_for_output(report.codestream.clone(), image.bits_per_sample(), &options);
                lossy = Some(report);
                bytes
            }
            Err(err) => {
                fail(&format!("{input}: {err}"));
                return EXIT_ERROR;
            }
        }
    } else {
        match jpxl_encode::encode(&image, &options) {
            Ok(encoded) => encoded,
            Err(err) => {
                fail(&format!("{input}: {err}"));
                return EXIT_ERROR;
            }
        }
    };

    match write_path(output, &encoded) {
        Ok(()) => {
            let mode = mode_label.clone().unwrap_or_else(|| match rate_target {
                Some(jpxl_encode_policy::RateTarget::BitsPerPixel(b)) => {
                    format!("lossy VarDCT, target {b} bpp")
                }
                Some(jpxl_encode_policy::RateTarget::Bytes(n)) => {
                    format!("lossy VarDCT, target {n} bytes")
                }
                None => format!("lossless modular, effort {}", options.effort.level()),
            });
            status_line(
                output,
                &format!(
                    "{output}: {}x{}, {} channel(s), {} bits per sample, {mode}, {} bytes",
                    image.width(),
                    image.height(),
                    image.num_channels(),
                    image.bits_per_sample(),
                    encoded.len()
                ),
            );
            // After a perceptual encode, one machine-readable line of the
            // controller's decision.
            if let Some(line) = &perceptual_line {
                status_line(output, line);
            }
            // A missed rate target is not a failure — the loop's contract is
            // "never over" — but it is silent unless said out loud, and the two
            // reasons for it want opposite responses. Saturated means the
            // ladder ran out of rungs and more budget cannot help; unsaturated
            // means the search ran out of prices short of the target.
            if let Some(report) = lossy {
                if sections {
                    print_section_breakdown(&report.sizing, output);
                }
                // Research aid: `JPXL_RATE_TRACE=1` dumps every priced rung so a
                // rate-search miss can be attributed to the ladder shape.
                if let Some(path) = std::env::var_os("JPXL_COVER_DUMP") {
                    dump_cover_map(&report.plan, &path.to_string_lossy());
                }
                if std::env::var_os("JPXL_RATE_TRACE").is_some() {
                    status_line(
                        output,
                        &format!(
                            "  controller: status={:?} fallbacks={} fresh_rescues={} \
                             rescue_prices={} first_finalist_bytes={} correction_bytes={} \
                             fast_prices={} full_prices={}",
                            report.status,
                            report.stats.anchor_fallbacks,
                            report.stats.fresh_structure_rescues,
                            report.stats.rescue_prices,
                            report.stats.anchor_first_finalist_bytes,
                            report.stats.anchor_correction_bytes,
                            report.stats.fast_prices,
                            report.stats.full_prices
                        ),
                    );
                    for step in &report.trace {
                        status_line(
                            output,
                            &format!(
                                "  trace: {:?} rung={} scale={} hf_mul={} bytes={} feasible={}",
                                step.phase,
                                step.quantizer.rung.get(),
                                step.quantizer.global_scale.get(),
                                step.quantizer.hf_mul.get(),
                                step.bytes,
                                step.feasible
                            ),
                        );
                    }
                }
                let miss = report.target_bytes.saturating_sub(report.achieved);
                let slack = report.allowed_undershoot;
                if miss > slack {
                    let pct = if report.target_bytes == 0 {
                        0.0
                    } else {
                        #[allow(
                            clippy::cast_precision_loss,
                            reason = "byte counts stay inside f64's exact-integer range"
                        )]
                        let f = (miss as f64) * 100.0 / (report.target_bytes as f64);
                        f
                    };
                    let why = match report.status {
                        jpxl_encode_policy::RateStatus::SaturatedTop => {
                            "ladder saturated: the finest quantizer is still under target, \
                             so no extra search budget can close this"
                        }
                        jpxl_encode_policy::RateStatus::UnderTargetAdjacentRungs => {
                            "adjacent priced rungs straddle the target"
                        }
                        jpxl_encode_policy::RateStatus::UnderTargetWorkCap
                        | jpxl_encode_policy::RateStatus::RescuedFreshStructure => {
                            "the bounded production controller reached its work cap"
                        }
                        jpxl_encode_policy::RateStatus::ExhaustiveReference => {
                            "the exhaustive Quality reference ended outside its band"
                        }
                        jpxl_encode_policy::RateStatus::InsideBand => {
                            "the selected stream is inside the requested band"
                        }
                    };
                    status_line(
                        output,
                        &format!(
                            "  note: undershot target by {miss} bytes ({pct:.1}%) — {why} \
                             [prices: {} fast, {} full]",
                            report.fast_prices, report.full_prices
                        ),
                    );
                }
            }
            EXIT_OK
        }
        Err(err) => {
            fail(&format!("{output}: {err}"));
            EXIT_ERROR
        }
    }
}

/// `jpxl compare <a.ppm> <b.ppm>`: distortion between two decoded images.
///
/// Exists so a rate/distortion sweep can be driven from a shell script without
/// a Python or numpy dependency: encode at a rate, decode, compare. Prints
/// `rmse=<f> psnr_db=<f>`, or `psnr_db=inf` when the images are identical.
///
/// Always prints `ssimulacra2_jpxl=<f>`, the production metric used by the
/// score-targeted encoder. Built with `--features perceptual`, also prints the
/// independent rust-av reference as `ssimulacra2=<f>` (higher is better, 100 =
/// identical) and `butteraugli=<f>` with its `butteraugli_pnorm3` (lower is
/// better, 0 = identical; `butteraugli` is the distance `cjxl -d` targets).
/// PSNR is printed because it is always available, not because it is the
/// quality axis.
fn in_tree_ssimulacra2_score(
    reference: &jpxl_conformance::metrics::Image,
    candidate: &jpxl_conformance::metrics::Image,
) -> Option<f64> {
    if !reference.same_shape(candidate) {
        return None;
    }

    fn linear_planes(image: &jpxl_conformance::metrics::Image) -> [Vec<f32>; 3] {
        let pixel_count = image.samples.len() / 3;
        let mut red = Vec::with_capacity(pixel_count);
        let mut green = Vec::with_capacity(pixel_count);
        let mut blue = Vec::with_capacity(pixel_count);
        let scale = f32::from(image.max_value);
        let linear = |sample: u16| {
            let value = f32::from(sample) / scale;
            if value <= 0.040_45 {
                value / 12.92
            } else {
                ((value + 0.055) / 1.055).powf(2.4)
            }
        };
        for pixel in image.samples.chunks_exact(3) {
            if let [r, g, b] = pixel {
                red.push(linear(*r));
                green.push(linear(*g));
                blue.push(linear(*b));
            }
        }
        [red, green, blue]
    }

    let [reference_r, reference_g, reference_b] = linear_planes(reference);
    let [candidate_r, candidate_g, candidate_b] = linear_planes(candidate);
    let reference_view = jpxl_perceptual::LinearRgbView::new(
        reference.w,
        reference.h,
        &reference_r,
        &reference_g,
        &reference_b,
    )
    .ok()?;
    let candidate_view = jpxl_perceptual::LinearRgbView::new(
        candidate.w,
        candidate.h,
        &candidate_r,
        &candidate_g,
        &candidate_b,
    )
    .ok()?;
    jpxl_perceptual::score_pair(reference_view, candidate_view)
        .ok()
        .map(|result| result.score)
}

fn cmd_compare(args: &[String]) -> u8 {
    let [a_path, b_path] = args else {
        fail("`compare` takes two PPM paths (reference, then decoded)");
        return EXIT_ERROR;
    };

    let mut images = Vec::with_capacity(2);
    for path in [a_path, b_path] {
        let bytes = match std::fs::read(Path::new(path.as_str())) {
            Ok(bytes) => bytes,
            Err(err) => {
                fail(&format!("{path}: {err}"));
                return EXIT_ERROR;
            }
        };
        match jpxl_conformance::metrics::Image::from_ppm(&bytes) {
            Ok(image) => images.push(image),
            Err(err) => {
                fail(&format!("{path}: {err}"));
                return EXIT_ERROR;
            }
        }
    }
    let (Some(a), Some(b)) = (images.first(), images.get(1)) else {
        fail("`compare` needs two readable PPMs");
        return EXIT_ERROR;
    };

    let (Some(err), Some(db)) = (
        jpxl_conformance::metrics::rmse(a, b),
        jpxl_conformance::metrics::psnr(a, b),
    ) else {
        fail(&format!(
            "shape mismatch: {}x{}x{} vs {}x{}x{}",
            a.w, a.h, a.channels, b.w, b.h, b.channels
        ));
        return EXIT_ERROR;
    };

    let mut line = if db.is_infinite() {
        "rmse=0 psnr_db=inf".to_owned()
    } else {
        format!("rmse={err:.6} psnr_db={db:.4}")
    };

    match in_tree_ssimulacra2_score(a, b) {
        Some(score) => line.push_str(&format!(
            " ssimulacra2_jpxl={score:.6} ssimulacra2_jpxl_version={}",
            jpxl_perceptual::METRIC_VERSION
        )),
        None => line.push_str(&format!(
            " ssimulacra2_jpxl=n/a ssimulacra2_jpxl_version={}",
            jpxl_perceptual::METRIC_VERSION
        )),
    }

    // The perceptual metrics are the ones to rank lossy encoders on; PSNR is
    // here because it is always available. When they disagree with PSNR,
    // believe them. Note the scales run opposite ways: ssimulacra2 is
    // higher-is-better (100 = identical), butteraugli lower-is-better (0 =
    // identical, and `distance` is what `cjxl -d` targets).
    #[cfg(feature = "ssimulacra2")]
    {
        match jpxl_conformance::metrics::ssimulacra2_score(a, b) {
            Some(score) => line.push_str(&format!(" ssimulacra2={score:.4}")),
            // Not fatal: the exactness metrics above are still valid. Says why
            // rather than silently omitting the column.
            None => line.push_str(" ssimulacra2=n/a"),
        }
    }
    #[cfg(feature = "butteraugli")]
    {
        match jpxl_conformance::metrics::butteraugli_distance(a, b) {
            Some(s) => line.push_str(&format!(
                " butteraugli={:.4} butteraugli_pnorm3={:.4}",
                s.distance, s.pnorm3
            )),
            None => line.push_str(" butteraugli=n/a"),
        }
    }

    println!("{line}");
    EXIT_OK
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
    let mut rate_preset = jpxl_encode_policy::RateSearchPreset::default();
    let mut input: Option<&str> = None;
    let mut resources = jpxl_encode::EncodeResources::auto();
    let mut diag = false;
    let mut effort = jpxl_encode::Effort::DEFAULT;
    let mut overrides = jpxl_encode::ModularSearchOverrides::default();

    let mut rest = args.get(1..).unwrap_or(&[]).iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--diag" => {
                diag = true;
            }
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
            "--lossy-preset" => {
                let Some(value) = rest.next() else {
                    fail("`--lossy-preset` needs one of: quality, balanced, fast");
                    return EXIT_ERROR;
                };
                rate_preset = match value.as_str() {
                    "quality" => jpxl_encode_policy::RateSearchPreset::Quality,
                    "balanced" => jpxl_encode_policy::RateSearchPreset::Balanced,
                    "fast" => jpxl_encode_policy::RateSearchPreset::Fast,
                    _ => {
                        fail("`--lossy-preset` needs one of: quality, balanced, fast");
                        return EXIT_ERROR;
                    }
                };
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
            "--effort" => {
                let Some(level) = rest.next().and_then(|s| s.parse::<u8>().ok()) else {
                    fail("`--effort` needs an integer in 1..=9");
                    return EXIT_ERROR;
                };
                match jpxl_encode::Effort::new(level) {
                    Ok(e) => effort = e,
                    Err(_) => {
                        fail("`--effort` needs an integer in 1..=9");
                        return EXIT_ERROR;
                    }
                }
            }
            // Measurement overrides for the modular search budget. These sweep
            // the effort ladder's constants so they can be chosen from
            // evidence rather than guessed; they are not quality dials, and
            // every setting stays exact-lossless.
            "--modular-max-depth" => {
                let Some(v) = rest.next().and_then(|s| s.parse::<u32>().ok()) else {
                    fail("`--modular-max-depth` needs a non-negative integer");
                    return EXIT_ERROR;
                };
                overrides.max_tree_depth = Some(v);
            }
            "--modular-max-leaves" => {
                let Some(v) = rest.next().and_then(|s| s.parse::<usize>().ok()) else {
                    fail("`--modular-max-leaves` needs a positive integer");
                    return EXIT_ERROR;
                };
                overrides.max_tree_leaves = Some(v.max(1));
            }
            "--modular-sample-budget" => {
                let Some(v) = rest.next().and_then(|s| parse_sample_budget(s)) else {
                    fail("`--modular-sample-budget` needs an integer or `full`");
                    return EXIT_ERROR;
                };
                overrides.cheap_sample_budget = Some(v);
            }
            "--modular-deep-cap" => {
                let Some(v) = rest.next().and_then(|s| parse_sample_budget(s)) else {
                    fail("`--modular-deep-cap` needs an integer or `full`");
                    return EXIT_ERROR;
                };
                overrides.deep_search_sample_cap = Some(v);
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

    jpxl_encode::lossless::set_plan_diagnostics_enabled(diag);
    jpxl_encode_policy::diagnostics::set_encode_diag_enabled(diag);
    let timed = match mode {
        "modular" => bench_modular(&rgb, width, height, iters, resources, effort, overrides),
        "vardct-fixed" => bench_vardct_fixed(&rgb, width, height, iters, resources),
        "vardct-rate" => bench_vardct_rate(&rgb, width, height, bpp, iters, resources, rate_preset),
        "vardct-probe" => bench_vardct_probe(&rgb, width, height, iters, resources),
        other => {
            fail(&format!(
                "unknown bench mode `{other}` (modular|vardct-fixed|vardct-rate|vardct-probe)"
            ));
            return EXIT_ERROR;
        }
    };

    // Do not leak measurement state if the CLI grows another command in this
    // process later (or when `run` is called repeatedly by tests).
    jpxl_encode::lossless::set_plan_diagnostics_enabled(false);
    jpxl_encode_policy::diagnostics::set_encode_diag_enabled(false);

    match timed {
        Ok(report) => {
            println!(
                "mode={mode} size={width}x{height} iters={iters} \
                 wall_ms_total={:.3} wall_ms_min={:.3} wall_ms_median={:.3} \
                 wall_ms_mad={:.3} \
                 output_bytes={} fingerprint={:016x}",
                report.wall_ms_total,
                report.wall_ms_min,
                report.wall_ms_median,
                report.wall_ms_mad,
                report.output_bytes,
                report.fingerprint
            );
            if diag {
                match mode {
                    "modular" => {
                        let m = jpxl_encode::lossless::last_plan_multiplicity();
                        println!("diag={}", m.summary_line());
                    }
                    "vardct-fixed" | "vardct-rate" | "vardct-probe" => {
                        let d = jpxl_encode_policy::last_encode_diag();
                        println!("diag={}", d.summary_line());
                        if let Some(s) = report.rate_stats {
                            // Aggregate over every probe of the rate search
                            // (the diag= line above is only the *last* probe's
                            // plan_at breakdown). Phase-3: measures whether
                            // CandidateForwardCache's cross-probe reuse is
                            // real before any change to its size/eviction.
                            let total = s.dct_cache_hits + s.dct_cache_misses;
                            #[allow(
                                clippy::cast_precision_loss,
                                reason = "probe/cache counts stay far inside f64's exact-integer \
                                          range; this is a printed ratio, not a stored value"
                            )]
                            let hit_rate = if total > 0 {
                                s.dct_cache_hits as f64 / total as f64
                            } else {
                                0.0
                            };
                            println!(
                                "rate_diag=fast_prices={} full_prices={} \
                                 structural_builds={} sketch_probes={} \
                                 exact_candidates={} anchor_fallbacks={} fresh_rescues={} \
                                 rescue_prices={} \
                                 anchor_first_finalist_bytes={} \
                                 anchor_correction_bytes={} \
                                 dct_cache_hits={} dct_cache_misses={} \
                                 dct_cache_hit_rate={hit_rate:.4} candidate_entries={} \
                                 candidate_payload_bytes={} candidate_allocations={}",
                                s.fast_prices,
                                s.full_prices,
                                s.structural_builds,
                                s.sketch_probes,
                                s.exact_candidates,
                                s.anchor_fallbacks,
                                s.fresh_structure_rescues,
                                s.rescue_prices,
                                s.anchor_first_finalist_bytes,
                                s.anchor_correction_bytes,
                                s.dct_cache_hits,
                                s.dct_cache_misses,
                                s.candidate_cache_entries,
                                s.candidate_payload_bytes,
                                s.candidate_allocations,
                            );
                            print_rate_phase("rate_plan_fast", s.fast);
                            print_rate_phase("rate_plan_full", s.full);
                            print_writer_phase("rate_writer_fast", s.writer.fast);
                            print_writer_phase("rate_writer_full", s.writer.full);
                        }
                        if let Some(t) = report.rate_trace {
                            println!(
                                "rate_trace=bracket={} bisect={} fill={} lf_fill={} final={} \
                                 fast_best_rung={:?} fast_best_quant_lf={:?} \
                                 fast_upper_rung={:?} full_start_rung={:?} \
                                 full_start_quant_lf={:?} chosen_rung={} chosen_quant_lf={}",
                                t.bracket,
                                t.bisect,
                                t.fill,
                                t.lf_fill,
                                t.final_prices,
                                t.fast_best_rung,
                                t.fast_best_quant_lf,
                                t.fast_upper_rung,
                                t.full_start_rung,
                                t.full_start_quant_lf,
                                t.chosen_rung,
                                t.chosen_quant_lf,
                            );
                        }
                    }
                    _ => {}
                }
            }
            if mode == "vardct-rate"
                && let Some(a) = report.rate_amplification
            {
                if diag {
                    println!(
                        "rate_amp=selected_emit_ms_median={:.3} \
                         search_amplification={:.3} writer_amplification={:.3}",
                        a.selected_emit_ms_median, a.search_amplification, a.writer_amplification,
                    );
                } else {
                    println!(
                        "rate_amp=selected_emit_ms_median={:.3} \
                         search_amplification={:.3} writer_amplification=disabled",
                        a.selected_emit_ms_median, a.search_amplification,
                    );
                }
            }
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
    wall_ms_min: f64,
    wall_ms_median: f64,
    wall_ms_mad: f64,
    output_bytes: usize,
    fingerprint: u64,
    /// Rate-search multiplicity counters from the last timed iteration.
    /// `vardct-rate` only; every other mode leaves this `None`.
    rate_stats: Option<jpxl_encode_policy::RateProbeStats>,
    /// Control-flow and rung summary from the last timed rate search.
    rate_trace: Option<RateTraceStats>,
    /// Target-rate search and exact-writer amplification (`vardct-rate`).
    rate_amplification: Option<RateAmplification>,
}

#[derive(Debug, Clone, Copy)]
struct RateAmplification {
    selected_emit_ms_median: f64,
    search_amplification: f64,
    writer_amplification: f64,
}

fn ns_ms(ns: u64) -> f64 {
    #[allow(
        clippy::cast_precision_loss,
        reason = "diagnostic nanoseconds are printed approximately as milliseconds"
    )]
    let ms = ns as f64 / 1e6;
    ms
}

fn print_rate_phase(label: &str, phase: jpxl_encode_policy::diagnostics::SearchPhaseDiagnostics) {
    println!(
        "{label}=plans={} cover_passes={} cfl_searches={} quantize_group_passes={} \
         census_passes={} entropy_trainings={} order_candidates={} \
         block_context_candidates={} preset_candidates={} plan_ms={:.1} cover_ms={:.1} \
         cfl_ms={:.1} quantize_ms={:.1} entropy_ms={:.1}",
        phase.plans,
        phase.cover_passes,
        phase.cfl_searches,
        phase.quantize_group_passes,
        phase.census_passes,
        phase.entropy_trainings,
        phase.order_candidates,
        phase.block_context_candidates,
        phase.preset_candidates,
        ns_ms(phase.plan_ns),
        ns_ms(phase.cover_ns),
        ns_ms(phase.cfl_ns),
        ns_ms(phase.quantize_ns),
        ns_ms(phase.entropy_ns),
    );
}

fn print_writer_phase(
    label: &str,
    phase: jpxl_encode::vardct::diagnostics::WriterPhaseDiagnostics,
) {
    println!(
        "{label}=internal_counts={} outer_counts={} other_counts={} stores={} \
         section_traversals={} lf_sections={} pass_group_sections={} pool_builds={} \
         count_ms={:.1} store_ms={:.1} pool_build_ms={:.1} tape_symbols={} \
         tape_extra_symbols={} tape_payload_bytes={} tape_legacy_payload_bytes={} \
         tape_record_ms={:.1}",
        phase.internal_count_emissions,
        phase.outer_count_emissions,
        phase.other_count_emissions,
        phase.stored_emissions,
        phase.section_body_traversals,
        phase.lf_section_encodes,
        phase.pass_group_section_encodes,
        phase.executor_pool_builds,
        ns_ms(phase.count_emission_ns),
        ns_ms(phase.stored_emission_ns),
        ns_ms(phase.executor_pool_build_ns),
        phase.tape_symbols,
        phase.tape_extra_symbols,
        phase.tape_payload_bytes,
        phase.tape_legacy_payload_bytes,
        ns_ms(phase.tape_record_ns),
    );
}

/// Compact handoff telemetry for `vardct-rate --diag`.
///
/// The full trace can contain forty prices and is intentionally not printed.
/// These counts and boundary rungs distinguish time spent locating a Fast
/// incumbent from time spent turning it into the final Full-priced winner.
#[derive(Debug, Clone, Copy, Default)]
struct RateTraceStats {
    bracket: usize,
    bisect: usize,
    fill: usize,
    lf_fill: usize,
    final_prices: usize,
    fast_best_rung: Option<u32>,
    fast_best_quant_lf: Option<u32>,
    fast_upper_rung: Option<u32>,
    full_start_rung: Option<u32>,
    full_start_quant_lf: Option<u32>,
    chosen_rung: u32,
    chosen_quant_lf: u32,
}

impl RateTraceStats {
    fn from_outcome(outcome: &jpxl_encode_policy::RateOutcome) -> Self {
        use jpxl_encode_policy::RatePhase;

        let mut out = Self {
            chosen_rung: outcome.chosen.rung.get(),
            chosen_quant_lf: outcome.chosen.quant_lf.get(),
            ..Self::default()
        };
        for step in &outcome.trace {
            match step.phase {
                RatePhase::Bracket => out.bracket += 1,
                RatePhase::Bisect => out.bisect += 1,
                RatePhase::Fill => out.fill += 1,
                RatePhase::LfFill => out.lf_fill += 1,
                RatePhase::Final | RatePhase::Rescue => out.final_prices += 1,
            }
        }

        let fast_best = outcome
            .trace
            .iter()
            .filter(|step| {
                !matches!(step.phase, RatePhase::Final | RatePhase::Rescue) && step.feasible
            })
            .max_by_key(|step| (step.bytes, step.quantizer.rung));
        out.fast_best_rung = fast_best.map(|step| step.quantizer.rung.get());
        out.fast_best_quant_lf = fast_best.map(|step| step.quantizer.quant_lf.get());
        out.fast_upper_rung = fast_best.and_then(|best| {
            outcome
                .trace
                .iter()
                .filter(|step| {
                    !matches!(step.phase, RatePhase::Final | RatePhase::Rescue)
                        && !step.feasible
                        && step.quantizer.rung > best.quantizer.rung
                })
                .min_by_key(|step| step.quantizer.rung)
                .map(|step| step.quantizer.rung.get())
        });
        let full_start = outcome
            .trace
            .iter()
            .find(|step| matches!(step.phase, RatePhase::Final | RatePhase::Rescue));
        out.full_start_rung = full_start.map(|step| step.quantizer.rung.get());
        out.full_start_quant_lf = full_start.map(|step| step.quantizer.quant_lf.get());
        out
    }
}

fn bench_modular(
    rgb: &[u8],
    width: u32,
    height: u32,
    iters: usize,
    resources: jpxl_encode::EncodeResources,
    effort: jpxl_encode::Effort,
    modular_search_overrides: jpxl_encode::ModularSearchOverrides,
) -> Result<BenchReport, String> {
    let samples: Vec<u16> = rgb.iter().map(|&b| u16::from(b)).collect();
    let image = jpxl_encode::Image::from_interleaved(width, height, 3, 8, &samples)
        .map_err(|e| e.to_string())?;
    let options = jpxl_encode::EncodeOptions {
        resources,
        effort,
        modular_search_overrides,
        ..jpxl_encode::EncodeOptions::default()
    };
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
    rate_preset: jpxl_encode_policy::RateSearchPreset,
) -> Result<BenchReport, String> {
    let target = jpxl_encode_policy::RateTarget::BitsPerPixel(bpp);
    let mut request = jpxl_encode_policy::EncodeRequest::for_target(target);
    request.resources = resources;
    request.rate_preset = rate_preset;
    let warm = jpxl_encode_policy::encode_srgb8_to_target(width, height, rgb, &request, target)
        .map_err(|e| e.to_string())?;
    let output_bytes = warm.codestream.len();
    let fingerprint = fnv1a64(&warm.codestream);
    let mut search_samples_ms = Vec::with_capacity(iters);
    let mut selected_emit_samples_ms = Vec::with_capacity(iters);
    let mut last_stats = warm.stats;
    let mut last_trace = RateTraceStats::from_outcome(&warm);
    let mut last_writer_amplification = 0.0;
    for _ in 0..iters {
        let search_started = std::time::Instant::now();
        let outcome =
            jpxl_encode_policy::encode_srgb8_to_target(width, height, rgb, &request, target)
                .map_err(|e| e.to_string())?;
        let search_ms = search_started.elapsed().as_secs_f64() * 1000.0;
        if outcome.codestream.len() != output_bytes || fnv1a64(&outcome.codestream) != fingerprint {
            return Err("rate-search output changed across iterations".to_owned());
        }

        // Measure one exact emission of the selected plan outside the search
        // timer. This is both the denominator for search amplification and a
        // byte-for-byte guard that the retained winning emission matches the
        // plan handed back by the search.
        let emit_started = std::time::Instant::now();
        let selected = jpxl_encode::vardct::emit_codestream_with(&outcome.plan, resources)
            .map_err(|e| e.to_string())?;
        let selected_emit_ms = emit_started.elapsed().as_secs_f64() * 1000.0;
        if selected.bytes != outcome.codestream || selected.sizing != outcome.sizing {
            return Err("selected-plan re-emission differs from retained winner".to_owned());
        }

        let one_emission_sections = u64::try_from(outcome.sizing.sections.len()).unwrap_or(0);
        #[allow(
            clippy::cast_precision_loss,
            reason = "diagnostic traversal counts stay inside f64's exact integer range"
        )]
        let writer_amplification = if one_emission_sections == 0 {
            0.0
        } else {
            outcome.stats.writer.total_section_traversals() as f64 / one_emission_sections as f64
        };
        search_samples_ms.push(search_ms);
        selected_emit_samples_ms.push(selected_emit_ms);
        last_writer_amplification = writer_amplification;
        last_stats = outcome.stats;
        last_trace = RateTraceStats::from_outcome(&outcome);
    }
    let (wall_ms_min, wall_ms_median, wall_ms_mad) = summarize_ms(&mut search_samples_ms);
    let selected_emit_ms_median = median_ms(&mut selected_emit_samples_ms);
    let search_amplification = if selected_emit_ms_median > 0.0 {
        wall_ms_median / selected_emit_ms_median
    } else {
        0.0
    };
    Ok(BenchReport {
        wall_ms_total: search_samples_ms.iter().sum(),
        wall_ms_min,
        wall_ms_median,
        wall_ms_mad,
        output_bytes,
        fingerprint,
        rate_stats: Some(last_stats),
        rate_trace: Some(last_trace),
        rate_amplification: Some(RateAmplification {
            selected_emit_ms_median,
            search_amplification,
            writer_amplification: last_writer_amplification,
        }),
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
    let (wall_ms_min, wall_ms_median, wall_ms_mad) = summarize_ms(&mut samples_ms);
    Ok(BenchReport {
        wall_ms_total,
        wall_ms_min,
        wall_ms_median,
        wall_ms_mad,
        output_bytes,
        fingerprint,
        rate_stats: None,
        rate_trace: None,
        rate_amplification: None,
    })
}

fn median_ms(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = samples.len() / 2;
    if samples.len().is_multiple_of(2) && samples.len() >= 2 {
        let lo = samples.get(mid - 1).copied().unwrap_or(0.0);
        let hi = samples.get(mid).copied().unwrap_or(0.0);
        (lo + hi) / 2.0
    } else {
        samples.get(mid).copied().unwrap_or(0.0)
    }
}

fn summarize_ms(samples: &mut [f64]) -> (f64, f64, f64) {
    let median = median_ms(samples);
    let minimum = samples.first().copied().unwrap_or(0.0);
    let mut deviations: Vec<f64> = samples
        .iter()
        .map(|sample| (sample - median).abs())
        .collect();
    let mad = median_ms(&mut deviations);
    (minimum, median, mad)
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

/// Encodes `image` through the lossy VarDCT rate loop to `target`.
///
/// The policy crate's high-precision façade takes interleaved sRGB samples, so
/// this re-interleaves the validated planes rather than re-reading the file.
/// Greyscale is refused rather than silently expanded to RGB.
/// What a lossy encode did, beyond the bytes: enough to tell a *hit* target
/// from a *missed* one and to say why it was missed.
struct LossyReport {
    codestream: Vec<u8>,
    /// Exact per-section byte accounting of the emitted codestream.
    sizing: jpxl_encode::vardct::CodestreamSizing,
    target_bytes: u64,
    achieved: u64,
    allowed_undershoot: u64,
    /// Explicit terminal state of the target-rate controller.
    status: jpxl_encode_policy::RateStatus,
    fast_prices: u32,
    full_prices: u32,
    /// Every priced candidate, for the `JPXL_RATE_TRACE` research dump.
    trace: Vec<jpxl_encode_policy::RateStep>,
    /// The chosen plan, for the `JPXL_COVER_DUMP` research dump.
    plan: jpxl_encode::vardct::ValidatedEmissionPlan,
    /// The search's multiplicity counters, for the `JPXL_RATE_TRACE` dump.
    stats: jpxl_encode_policy::RateProbeStats,
}

/// Research aid: writes the chosen cover as a P5 map with one sample per 8x8
/// block — 64 for DCT8x8, 128 for DCT16x16, 255 for DCT32x32 (other shapes
/// 32) — so a perceptual hot spot can be matched against the transform under
/// it.
fn dump_cover_map(plan: &jpxl_encode::vardct::ValidatedEmissionPlan, path: &str) {
    let Ok(geometry) = plan.geometry() else {
        return;
    };
    let grid = geometry.frame_blocks();
    let (w, h) = (grid.width as usize, grid.height as usize);
    let mut map = vec![0u8; w * h];
    for group in plan.plan().spatial.lf_groups.iter() {
        let Some(rect) = geometry.lf_group_rect(group.id) else {
            continue;
        };
        for vb in group.blocks.iter() {
            let (rows, cols) = vb.transform.block_dims();
            let value = match vb.transform.sample_cols() {
                8 => 64u8,
                16 => 128,
                32 => 255,
                _ => 32,
            };
            let x0 = (rect.x0 / 8 + vb.origin.bx()) as usize;
            let y0 = (rect.y0 / 8 + vb.origin.by()) as usize;
            for dy in 0..rows {
                for dx in 0..cols {
                    if let Some(slot) = map.get_mut((y0 + dy) * w + x0 + dx) {
                        *slot = value;
                    }
                }
            }
        }
    }
    let mut pgm = format!(
        "P5
{w} {h}
255
"
    )
    .into_bytes();
    pgm.extend_from_slice(&map);
    if let Err(err) = std::fs::write(path, pgm) {
        eprintln!("cover dump failed: {err}");
    }
}

/// `jpxl analyze-atlas <input> <output.jsonl>`: export diagnostic features.
fn cmd_analyze_atlas(args: &[String]) -> u8 {
    let [input, output] = args else {
        fail("`analyze-atlas` takes an input raster and an output JSONL path");
        return EXIT_ERROR;
    };
    let bytes = match read_path(input) {
        Ok(bytes) => bytes,
        Err(error) => {
            fail(&format!("{input}: {error}"));
            return EXIT_ERROR;
        }
    };
    let image = match image_io::decode_input(&bytes, None) {
        Ok(image) => image,
        Err(error) => {
            fail(&format!("{input}: {error}"));
            return EXIT_ERROR;
        }
    };
    let frame = match analysis_frame(&image) {
        Ok(frame) => frame,
        Err(error) => {
            fail(&error);
            return EXIT_ERROR;
        }
    };
    let atlas = jpxl_encode_policy::AnalysisAtlasV2::analyze(&frame);
    let file = match std::fs::File::create(output) {
        Ok(file) => file,
        Err(error) => {
            fail(&format!("{output}: {error}"));
            return EXIT_ERROR;
        }
    };
    let mut writer = std::io::BufWriter::new(file);
    let grid = atlas.grid();
    if writeln!(
        writer,
        "{{\"schema\":\"jpxl.analysis-atlas/1\",\"kind\":\"header\",\"width\":{},\"height\":{},\"grid_width\":{},\"grid_height\":{},\"atom_bytes\":{},\"total_bytes\":{}}}",
        image.width(),
        image.height(),
        grid.width,
        grid.height,
        core::mem::size_of::<jpxl_encode_policy::DiagnosticAtomFeatures>()
            + core::mem::size_of::<jpxl_encode_policy::AtomFeatures>(),
        atlas.byte_size(),
    )
    .is_err()
    {
        fail(&format!("{output}: cannot write atlas header"));
        return EXIT_ERROR;
    }
    for (index, (base, diagnostic)) in atlas.base().atoms().iter().zip(atlas.atoms()).enumerate() {
        let Ok(index) = u32::try_from(index) else {
            fail("analysis atlas has too many atoms to address");
            return EXIT_ERROR;
        };
        let atom_x = index % grid.width;
        let atom_y = index / grid.width;
        if writeln!(
            writer,
            "{{\"schema\":\"jpxl.analysis-atlas/1\",\"kind\":\"atom\",\"x\":{atom_x},\"y\":{atom_y},\"mean_xyb\":{:?},\"variance_xyb\":{:?},\"gradient_energy_xyb\":{:?},\"gradient_cross_xyb\":{:?},\"laplacian_energy_xyb\":{:?},\"plane_residual_xyb\":{:?},\"noise_mad_xyb\":{:?},\"dynamic_range_xyb\":{:?},\"covariance_xyb\":{:?},\"orientation_coherence_y\":{},\"flat_side_asymmetry_y\":{}}}",
            base.mean_xyb,
            base.variance_xyb,
            diagnostic.gradient_energy_xyb,
            diagnostic.gradient_cross_xyb,
            diagnostic.laplacian_energy_xyb,
            diagnostic.plane_residual_xyb,
            diagnostic.noise_mad_xyb,
            diagnostic.dynamic_range_xyb,
            diagnostic.covariance_xyb,
            diagnostic.orientation_coherence_y,
            diagnostic.flat_side_asymmetry_y,
        )
        .is_err()
        {
            fail(&format!("{output}: cannot write atlas atom"));
            return EXIT_ERROR;
        }
    }
    if writer.flush().is_err() {
        fail(&format!("{output}: cannot finish atlas export"));
        return EXIT_ERROR;
    }
    status_line(
        output,
        &format!(
            "{output}: {}x{} atoms, {} feature bytes",
            grid.width,
            grid.height,
            atlas.byte_size()
        ),
    );
    EXIT_OK
}

/// `jpxl features <input> [--json]`: print the quality controller's frame
/// source features as one JSON line, for the initial-rung calibration tooling.
fn cmd_features(args: &[String]) -> u8 {
    let mut input: Option<&String> = None;
    let mut transform_summary = false;
    for arg in args {
        match arg.as_str() {
            "--json" => {}
            "--transform-summary" => transform_summary = true,
            other if other.starts_with("--") => {
                fail(
                    "`features` takes an input raster and optional `--json` / `--transform-summary` flags",
                );
                return EXIT_ERROR;
            }
            _ if input.is_none() => input = Some(arg),
            _ => {
                fail("`features` takes exactly one input raster");
                return EXIT_ERROR;
            }
        }
    }
    let Some(input) = input else {
        fail(
            "`features` takes an input raster and optional `--json` / `--transform-summary` flags",
        );
        return EXIT_ERROR;
    };
    let bytes = match read_path(input) {
        Ok(bytes) => bytes,
        Err(error) => {
            fail(&format!("{input}: {error}"));
            return EXIT_ERROR;
        }
    };
    let image = match image_io::decode_input(&bytes, None) {
        Ok(image) => image,
        Err(error) => {
            fail(&format!("{input}: {error}"));
            return EXIT_ERROR;
        }
    };
    let frame = match analysis_frame(&image) {
        Ok(frame) => frame,
        Err(error) => {
            fail(&error);
            return EXIT_ERROR;
        }
    };
    let atlas = jpxl_encode_policy::AnalysisAtlas::analyze(&frame);
    let features = jpxl_encode_policy::quality_features::source_features(
        &atlas,
        image.width(),
        image.height(),
        frame.is_grayscale(),
        frame.preanalysis(),
    );
    if transform_summary {
        let mut request = jpxl_encode_policy::EncodeRequest::for_quality(
            jpxl_encode_policy::RateSearchPreset::Balanced,
        );
        request.bits_per_sample = image.bits_per_sample();
        match jpxl_encode_policy::transform_feature_summary(&frame, &request) {
            Ok(summary) => println!(
                "{{\"source_features\":{},\"transform_features\":{}}}",
                features.to_json(),
                summary.to_json()
            ),
            Err(error) => {
                fail(&format!("{input}: {error}"));
                return EXIT_ERROR;
            }
        }
    } else {
        println!("{}", features.to_json());
    }
    EXIT_OK
}

/// `jpxl quality-ladder --scales s1,s2,... [--effort fast|balanced] [--price]
/// [--threads N] <raster>`: the one-shot program's oracle-label sweep.
///
/// Builds the production quality pixel plan fresh at every requested
/// effective scale, scores each reconstruction canonically, optionally
/// exact-prices it, and prints one `jpxl.quality-ladder/1` JSONL record per
/// point after a header record carrying the source features. No navigation
/// and no output file: this is measurement for the offline crossing trainer
/// (`tools/quality_oracle_labels.py`), not an encoder mode.
fn cmd_quality_ladder(args: &[String]) -> u8 {
    let mut scales: Vec<u64> = Vec::new();
    let mut effort = jpxl::Effort::Balanced;
    let mut price = false;
    let mut threads: Option<usize> = None;
    let mut positional: Vec<&String> = Vec::new();
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--scales" => {
                let Some(list) = rest.next() else {
                    fail("`--scales` needs a comma-separated list of effective scales");
                    return EXIT_ERROR;
                };
                for part in list.split(',') {
                    match part.trim().parse::<u64>() {
                        Ok(scale) if scale >= 1 => scales.push(scale),
                        _ => {
                            fail("`--scales` entries must be positive integers");
                            return EXIT_ERROR;
                        }
                    }
                }
            }
            "--effort" => {
                let Some(mode) = rest.next() else {
                    fail("`--effort` needs fast or balanced");
                    return EXIT_ERROR;
                };
                effort = match mode.as_str() {
                    "fast" => jpxl::Effort::Fast,
                    "balanced" => jpxl::Effort::Balanced,
                    _ => {
                        fail("`--effort` needs fast or balanced");
                        return EXIT_ERROR;
                    }
                };
            }
            "--price" => price = true,
            "--threads" => {
                let Some(value) = rest.next().and_then(|v| v.parse::<usize>().ok()) else {
                    fail("`--threads` needs a positive worker count");
                    return EXIT_ERROR;
                };
                if value == 0 {
                    fail("`--threads` needs a positive worker count");
                    return EXIT_ERROR;
                }
                threads = Some(value);
            }
            other if other.starts_with("--") => {
                fail(&format!("unknown quality-ladder option `{other}`"));
                return EXIT_ERROR;
            }
            _ => positional.push(arg),
        }
    }
    let [input] = positional.as_slice() else {
        fail("`quality-ladder` takes exactly one input raster");
        return EXIT_ERROR;
    };
    if scales.is_empty() {
        fail("`quality-ladder` needs `--scales s1,s2,...`");
        return EXIT_ERROR;
    }
    if scales.len() > 4096 {
        fail("`quality-ladder` caps a sweep at 4096 points");
        return EXIT_ERROR;
    }

    let bytes = match read_path(input) {
        Ok(bytes) => bytes,
        Err(error) => {
            fail(&format!("{input}: {error}"));
            return EXIT_ERROR;
        }
    };
    let image = match image_io::decode_input(&bytes, None) {
        Ok(image) => image,
        Err(error) => {
            fail(&format!("{input}: {error}"));
            return EXIT_ERROR;
        }
    };
    let (width, height, bits_per_sample, rgb) = match image_to_rgb16(&image) {
        Ok(parts) => parts,
        Err(error) => {
            fail(&format!("{input}: {error}"));
            return EXIT_ERROR;
        }
    };
    if width < jpxl_perceptual::MIN_DIMENSION || height < jpxl_perceptual::MIN_DIMENSION {
        fail(&format!(
            "{input}: below the perceptual metric's {0}x{0} floor",
            jpxl_perceptual::MIN_DIMENSION
        ));
        return EXIT_ERROR;
    }

    // Map requested effective scales onto ladder rungs, ascending, deduped.
    let mut rungs: Vec<jpxl_encode_policy::Rung> = scales
        .iter()
        .map(|&scale| jpxl_encode_policy::rung_for_effective_scale(scale))
        .collect();
    rungs.sort_unstable();
    rungs.dedup();

    let mut request = jpxl_encode_policy::EncodeRequest::for_quality(effort.into());
    if let Some(threads) = threads {
        request.resources = jpxl_encode::EncodeResources::groups(threads);
    }
    request.bits_per_sample = bits_per_sample;
    let executor = request.resources.executor();
    let frame = match jpxl_encode_policy::PreparedFrame::from_srgb16_with(
        width,
        height,
        &rgb,
        bits_per_sample,
        Some(&executor),
    ) {
        Ok(frame) => frame.with_preanalysis(jpxl_encode_policy::preanalysis_srgb16(
            width,
            height,
            &rgb,
            bits_per_sample,
        )),
        Err(error) => {
            fail(&format!("{input}: {error}"));
            return EXIT_ERROR;
        }
    };
    let mut evaluator = match jpxl_perceptual::PlanRenderEvaluator::from_srgb16(
        width,
        height,
        &rgb,
        bits_per_sample,
        &executor,
    ) {
        Ok(evaluator) => evaluator,
        Err(_) => {
            fail(&format!(
                "{input}: the frame cannot be scored by the perceptual metric"
            ));
            return EXIT_ERROR;
        }
    };
    let atlas = jpxl_encode_policy::AnalysisAtlas::analyze(&frame);
    let features = jpxl_encode_policy::source_features(
        &atlas,
        width,
        height,
        frame.is_grayscale(),
        frame.preanalysis(),
    );
    use jpxl_encode_policy::PerceptualEvaluator as _;
    println!(
        "{{\"schema\":\"jpxl.quality-ladder/1\",\"input\":\"{}\",\"width\":{width},\
         \"height\":{height},\"bit_depth\":{bits_per_sample},\"effort\":\"{}\",\
         \"metric_version\":\"{}\",\"price\":{price},\"points\":{},\"source_features\":{}}}",
        input.replace('\\', "/"),
        effort_name(effort),
        evaluator.metric_version(),
        rungs.len(),
        features.to_json(),
    );
    let points = match jpxl_encode_policy::sweep_frame_perceptual(
        &frame,
        &atlas,
        &request,
        &rungs,
        price,
        &mut evaluator,
        &executor,
    ) {
        Ok(points) => points,
        Err(error) => {
            fail(&format!("{input}: {error}"));
            return EXIT_ERROR;
        }
    };
    for p in points {
        println!(
            "{{\"rung\":{},\"global_scale\":{},\"hf_mul\":{},\"quant_lf\":{},\
             \"effective_scale\":{},\"score\":{},\"bytes\":{},\"plan_ms\":{},\
             \"render_metric_ms\":{},\"price_ms\":{}}}",
            p.rung.get(),
            p.quantizer.global_scale.get(),
            p.quantizer.hf_mul.get(),
            p.quantizer.quant_lf.get(),
            p.effective_scale,
            p.score,
            p.exact_bytes
                .map_or_else(|| "null".to_owned(), |b| format!("{b}")),
            p.plan_ms,
            p.render_metric_ms,
            p.price_ms,
        );
    }
    EXIT_OK
}

/// `jpxl rate-ladder --scales s1,s2,... [--preset fast|balanced]
/// [--threads N] <raster>`: the rate-prior trainer's oracle-label sweep
/// (Phase RT2).
///
/// Trial-prices a fresh anchored rate plan at every requested effective
/// scale — the same plan and Fast-entropy Count price the bounded
/// controller's first anchor pays — and prints one `jpxl.rate-ladder/1`
/// JSONL record per point after a header carrying the source features.
/// No navigation, no Store, no output file: measurement only.
fn cmd_rate_ladder(args: &[String]) -> u8 {
    let mut scales: Vec<u64> = Vec::new();
    let mut preset = jpxl_encode_policy::RateSearchPreset::default();
    let mut threads: Option<usize> = None;
    let mut positional: Vec<&String> = Vec::new();
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--scales" => {
                let Some(list) = rest.next() else {
                    fail("`--scales` needs a comma-separated list of effective scales");
                    return EXIT_ERROR;
                };
                for part in list.split(',') {
                    match part.trim().parse::<u64>() {
                        Ok(scale) if scale >= 1 => scales.push(scale),
                        _ => {
                            fail("`--scales` entries must be positive integers");
                            return EXIT_ERROR;
                        }
                    }
                }
            }
            "--preset" => {
                let Some(mode) = rest.next() else {
                    fail("`--preset` needs fast or balanced");
                    return EXIT_ERROR;
                };
                preset = match mode.as_str() {
                    "fast" => jpxl_encode_policy::RateSearchPreset::Fast,
                    "balanced" => jpxl_encode_policy::RateSearchPreset::Balanced,
                    _ => {
                        fail("`--preset` needs fast or balanced");
                        return EXIT_ERROR;
                    }
                };
            }
            "--threads" => {
                let Some(value) = rest.next().and_then(|v| v.parse::<usize>().ok()) else {
                    fail("`--threads` needs a positive worker count");
                    return EXIT_ERROR;
                };
                if value == 0 {
                    fail("`--threads` needs a positive worker count");
                    return EXIT_ERROR;
                }
                threads = Some(value);
            }
            other if other.starts_with("--") => {
                fail(&format!("unknown rate-ladder option `{other}`"));
                return EXIT_ERROR;
            }
            _ => positional.push(arg),
        }
    }
    let [input] = positional.as_slice() else {
        fail("`rate-ladder` takes exactly one input raster");
        return EXIT_ERROR;
    };
    if scales.is_empty() {
        fail("`rate-ladder` needs `--scales s1,s2,...`");
        return EXIT_ERROR;
    }
    if scales.len() > 4096 {
        fail("`rate-ladder` caps a sweep at 4096 points");
        return EXIT_ERROR;
    }

    let bytes = match read_path(input) {
        Ok(bytes) => bytes,
        Err(error) => {
            fail(&format!("{input}: {error}"));
            return EXIT_ERROR;
        }
    };
    let image = match image_io::decode_input(&bytes, None) {
        Ok(image) => image,
        Err(error) => {
            fail(&format!("{input}: {error}"));
            return EXIT_ERROR;
        }
    };
    let (width, height, bits_per_sample, rgb) = match image_to_rgb16(&image) {
        Ok(parts) => parts,
        Err(error) => {
            fail(&format!("{input}: {error}"));
            return EXIT_ERROR;
        }
    };

    let mut rungs: Vec<jpxl_encode_policy::Rung> = scales
        .iter()
        .map(|&scale| jpxl_encode_policy::rung_for_effective_scale(scale))
        .collect();
    rungs.sort_unstable();
    rungs.dedup();

    // The target is never navigated to — only per-rung anchor prices are
    // taken — but the request shape (restoration, quant_lf coupling, preset
    // policies) must be the bounded controller's own.
    let target = jpxl_encode_policy::RateTarget::BitsPerPixel(1.0);
    let mut request = jpxl_encode_policy::EncodeRequest::for_target(target);
    request.rate_preset = preset;
    request.bits_per_sample = bits_per_sample;
    if let Some(threads) = threads {
        request.resources = jpxl_encode::EncodeResources::groups(threads);
    }
    let executor = request.resources.executor();
    let frame = match jpxl_encode_policy::PreparedFrame::from_srgb16_with(
        width,
        height,
        &rgb,
        bits_per_sample,
        Some(&executor),
    ) {
        Ok(frame) => frame,
        Err(error) => {
            fail(&format!("{input}: {error}"));
            return EXIT_ERROR;
        }
    };
    let atlas = jpxl_encode_policy::AnalysisAtlas::analyze(&frame);
    let features = jpxl_encode_policy::source_features(
        &atlas,
        width,
        height,
        frame.is_grayscale(),
        frame.preanalysis(),
    );
    println!(
        "{{\"schema\":\"jpxl.rate-ladder/1\",\"input\":\"{}\",\"width\":{width},\
         \"height\":{height},\"bit_depth\":{bits_per_sample},\"preset\":\"{}\",\
         \"points\":{},\"source_features\":{}}}",
        input.replace('\\', "/"),
        match preset {
            jpxl_encode_policy::RateSearchPreset::Fast => "fast",
            jpxl_encode_policy::RateSearchPreset::Balanced => "balanced",
            jpxl_encode_policy::RateSearchPreset::Quality => "quality",
        },
        rungs.len(),
        features.to_json(),
    );
    let points =
        match jpxl_encode_policy::sweep_frame_rate(&frame, &atlas, &request, &rungs, &executor) {
            Ok(points) => points,
            Err(error) => {
                fail(&format!("{input}: {error}"));
                return EXIT_ERROR;
            }
        };
    for p in points {
        println!(
            "{{\"rung\":{},\"global_scale\":{},\"hf_mul\":{},\"quant_lf\":{},\
             \"effective_scale\":{},\"bytes\":{},\"plan_ms\":{},\"price_ms\":{}}}",
            p.rung.get(),
            p.quantizer.global_scale.get(),
            p.quantizer.hf_mul.get(),
            p.quantizer.quant_lf.get(),
            p.effective_scale,
            p.bytes,
            p.plan_ms,
            p.price_ms,
        );
    }
    EXIT_OK
}

fn analysis_frame(image: &jpxl_encode::Image) -> Result<jpxl_encode_policy::PreparedFrame, String> {
    let planes = image.planes();
    let mut rgb = Vec::with_capacity(planes.first().map(Vec::len).unwrap_or(0).saturating_mul(3));
    match planes {
        [gray] => {
            for &sample in gray {
                let value = u16::try_from(sample)
                    .map_err(|_| "analysis input contains a negative sample".to_owned())?;
                rgb.extend_from_slice(&[value; 3]);
            }
        }
        [red, green, blue] => {
            for index in 0..red.len() {
                for plane in [red, green, blue] {
                    let sample = plane.get(index).copied().unwrap_or(0);
                    rgb.push(
                        u16::try_from(sample)
                            .map_err(|_| "analysis input contains a negative sample".to_owned())?,
                    );
                }
            }
        }
        _ => return Err("analysis input must have one or three colour channels".to_owned()),
    }
    jpxl_encode_policy::PreparedFrame::from_srgb16(
        image.width(),
        image.height(),
        &rgb,
        image.bits_per_sample(),
    )
    .map(|frame| {
        frame.with_preanalysis(jpxl_encode_policy::preanalysis_srgb16(
            image.width(),
            image.height(),
            &rgb,
            image.bits_per_sample(),
        ))
    })
    .map_err(|error| error.to_string())
}

/// Research controls that override the production target-rate policy.
struct LossyOverrides {
    aq_mode: Option<jpxl_encode_policy::AqMode>,
    aq_tuning: jpxl_encode_policy::AqTuning,
    fixed_quant_lf: Option<u32>,
    x_qm_scale: Option<u8>,
    b_qm_scale: Option<u8>,
    epf_iters: Option<u8>,
    epf_sharpness: Option<jpxl_encode_policy::EpfSharpnessMode>,
    cover_size_penalty: Option<jpxl_encode_policy::CoverSizePenalty>,
    cover_frequency_weight: Option<jpxl_encode_policy::CoverFrequencyWeight>,
    cover_rate_model: Option<jpxl_encode_policy::CoverRateModel>,
    quantizer_choice: Option<jpxl_encode_policy::QuantizerChoiceMode>,
    lambda_scale: Option<f32>,
    dead_zone_scale: Option<f32>,
    zero_token_bits: Option<f32>,
    /// Requested undershoot tolerance as a fraction of the target (the preset
    /// still applies its own floor).
    tolerance: Option<f64>,
    rate_preset: Option<jpxl_encode_policy::RateSearchPreset>,
}

fn encode_lossy_to_target(
    image: &jpxl_encode::Image,
    target: jpxl_encode_policy::RateTarget,
    overrides: LossyOverrides,
) -> Result<LossyReport, String> {
    if image.num_channels() != 3 {
        return Err(format!(
            "lossy encoding needs RGB input; this is {} channel(s) at {} bits",
            image.num_channels(),
            image.bits_per_sample()
        ));
    }
    let planes = image.planes();
    let (Some(r), Some(g), Some(b)) = (planes.first(), planes.get(1), planes.get(2)) else {
        return Err("lossy encoding needs three colour planes".to_owned());
    };
    let bits_per_sample = image.bits_per_sample();
    let max = if bits_per_sample >= 16 {
        i32::from(u16::MAX)
    } else {
        i32::try_from((1u32 << bits_per_sample) - 1).unwrap_or(i32::MAX)
    };
    let mut rgb = Vec::with_capacity(r.len().saturating_mul(3));
    for i in 0..r.len() {
        for plane in [r, g, b] {
            // Planes are validated to the declared bit depth, so the
            // clamp is belt-and-braces rather than load-bearing.
            let v = plane.get(i).copied().unwrap_or(0).clamp(0, max);
            rgb.push(u16::try_from(v).unwrap_or(0));
        }
    }
    let mut request = jpxl_encode_policy::EncodeRequest::for_target(target);
    if let Some(preset) = overrides.rate_preset {
        request.rate_preset = preset;
    }
    if let Some(mode) = overrides.aq_mode {
        request.budget.aq_mode = mode;
    }
    request.budget.aq_tuning = overrides.aq_tuning;
    if let Some(value) = overrides.fixed_quant_lf {
        request.quant_lf = jpxl_encode::vardct::ids::QuantLf::new(value)
            .map_err(|_| format!("quant_lf {value} is outside the wire range"))?;
        request.budget.rate.lf_fill_probes = 0;
    }
    if let Some(value) = overrides.x_qm_scale {
        // An explicit chroma scale is a research override: pin Manual so the
        // automatic preset policy does not resolve X back to its own value.
        request.chroma_hf_policy = jpxl_encode_policy::ChromaHfPolicy::Manual;
        request.x_qm_scale = jpxl_encode::vardct::ids::QmScale::new(value)
            .map_err(|_| format!("x_qm_scale {value} is outside the wire range"))?;
    }
    if let Some(value) = overrides.b_qm_scale {
        request.chroma_hf_policy = jpxl_encode_policy::ChromaHfPolicy::Manual;
        request.b_qm_scale = jpxl_encode::vardct::ids::QmScale::new(value)
            .map_err(|_| format!("b_qm_scale {value} is outside the wire range"))?;
    }
    if let Some(iters) = overrides.epf_iters {
        request.restoration.epf_iters = iters;
        if iters == 0 && overrides.epf_sharpness.is_none() {
            // `--epf-iters 0` is the ergonomic spelling of restoration off.
            // Do not retain the target policy's otherwise inert sharpness
            // plane and its wire cost unless the caller explicitly asks for it.
            request.epf_sharpness = jpxl_encode_policy::EpfSharpnessMode::Zero;
        }
    }
    if let Some(sharpness) = overrides.epf_sharpness {
        request.epf_sharpness = sharpness;
    }
    if let Some(penalty) = overrides.cover_size_penalty {
        request.cover_size_penalty = penalty;
    }
    if let Some(weight) = overrides.cover_frequency_weight {
        request.cover_frequency_weight = weight;
    }
    if let Some(model) = overrides.cover_rate_model {
        request.cover_rate_model = model;
    }
    if let Some(mode) = overrides.quantizer_choice {
        request.quantizer_choice = mode;
    }
    if let Some(scale) = overrides.lambda_scale {
        request.lambda_scale = scale;
    }
    if let Some(scale) = overrides.dead_zone_scale {
        request.dead_zone_scale = scale;
    }
    if let Some(bits) = overrides.zero_token_bits {
        request.zero_token_bits = bits;
    }
    if let Some(fraction) = overrides.tolerance {
        request.tolerance = jpxl_encode_policy::RateTolerance {
            bytes: request.tolerance.bytes,
            fraction,
        };
    }
    jpxl_encode_policy::encode_srgb16_to_target(
        image.width(),
        image.height(),
        &rgb,
        bits_per_sample,
        &request,
        target,
    )
    .map(|outcome| LossyReport {
        target_bytes: outcome.target,
        achieved: outcome.achieved(),
        allowed_undershoot: request
            .rate_preset
            .tolerance(request.tolerance)
            .bytes_for(outcome.target),
        status: outcome.status,
        fast_prices: outcome.stats.fast_prices,
        full_prices: outcome.stats.full_prices,
        stats: outcome.stats,
        sizing: outcome.sizing,
        trace: outcome.trace,
        plan: outcome.plan,
        codestream: outcome.codestream,
    })
    .map_err(|e| e.to_string())
}

/// Maps a lossy effort name to its facade effort and rate-search preset.
///
/// `quality` is only a name when the `quality-effort` feature forwards the
/// exhaustive-reference effort; otherwise it is rejected like any other
/// non-name.
fn parse_lossy_effort(name: &str) -> Option<(jpxl::Effort, jpxl_encode_policy::RateSearchPreset)> {
    match name {
        "fast" => Some((
            jpxl::Effort::Fast,
            jpxl_encode_policy::RateSearchPreset::Fast,
        )),
        "balanced" => Some((
            jpxl::Effort::Balanced,
            jpxl_encode_policy::RateSearchPreset::Balanced,
        )),
        #[cfg(feature = "quality-effort")]
        "quality" => Some((
            jpxl::Effort::Quality,
            jpxl_encode_policy::RateSearchPreset::Quality,
        )),
        _ => None,
    }
}

/// The lower-case name of a lossy effort, for the perceptual report line.
fn effort_name(effort: jpxl::Effort) -> &'static str {
    match effort {
        jpxl::Effort::Fast => "fast",
        jpxl::Effort::Balanced => "balanced",
        #[cfg(feature = "quality-effort")]
        jpxl::Effort::Quality => "quality",
    }
}

/// The snake-case wire name of a perceptual controller status.
fn perceptual_status_str(status: jpxl::PerceptualStatus) -> &'static str {
    match status {
        jpxl::PerceptualStatus::Met => "met",
        jpxl::PerceptualStatus::MetAdjacentRungs => "met_adjacent_rungs",
        jpxl::PerceptualStatus::MetWorkCap => "met_work_cap",
        jpxl::PerceptualStatus::SaturatedFloor => "saturated_floor",
        jpxl::PerceptualStatus::SaturatedTop => "saturated_top",
        jpxl::PerceptualStatus::UnderTargetWorkCap => "under_target_work_cap",
        jpxl::PerceptualStatus::RescuedFreshStructure => "rescued_fresh_structure",
        jpxl::PerceptualStatus::RoutedToLossless => "routed_to_lossless",
        jpxl::PerceptualStatus::FallbackLossless => "fallback_lossless",
        jpxl::PerceptualStatus::UnsupportedTooSmall => "unsupported_too_small",
    }
}

/// The single machine-readable line printed after a perceptual encode.
fn format_perceptual_line(outcome: &jpxl::PerceptualOutcome, effort: &str) -> String {
    let achieved = outcome
        .achieved_score
        .map_or_else(|| "n/a".to_owned(), |score| format!("{score:.4}"));
    format!(
        "quality_target={:.4} achieved={achieved} bytes={} metric={} effort={effort} \
         probes={} prices={} status={}",
        outcome.requested_score,
        outcome.exact_bytes,
        outcome.metric_version,
        outcome.probes,
        outcome.prices,
        perceptual_status_str(outcome.status),
    )
}

/// Builds an interleaved RGB `u16` buffer from a three-channel image.
///
/// Returns `(width, height, bits_per_sample, interleaved_rgb)`.
fn image_to_rgb16(image: &jpxl_encode::Image) -> Result<(u32, u32, u32, Vec<u16>), String> {
    if image.num_channels() != 3 {
        return Err(format!(
            "lossy encoding needs RGB input; this is {} channel(s) at {} bits",
            image.num_channels(),
            image.bits_per_sample()
        ));
    }
    let planes = image.planes();
    let (Some(r), Some(g), Some(b)) = (planes.first(), planes.get(1), planes.get(2)) else {
        return Err("lossy encoding needs three colour planes".to_owned());
    };
    let bits_per_sample = image.bits_per_sample();
    let max = if bits_per_sample >= 16 {
        i32::from(u16::MAX)
    } else {
        i32::try_from((1u32 << bits_per_sample) - 1).unwrap_or(i32::MAX)
    };
    let mut rgb = Vec::with_capacity(r.len().saturating_mul(3));
    for i in 0..r.len() {
        for plane in [r, g, b] {
            let v = plane.get(i).copied().unwrap_or(0).clamp(0, max);
            rgb.push(u16::try_from(v).unwrap_or(0));
        }
    }
    Ok((image.width(), image.height(), bits_per_sample, rgb))
}

/// The `--quality` path: a minimum-SSIMULACRA2 encode through the facade.
///
/// Returns the codestream, the perceptual report line, and the summary mode
/// string. A score of 100 routes to the lossless encoder; anything lower runs
/// the quality controller. When the controller cannot verify the score and
/// `fallback` is [`jpxl::QualityFallback::Refuse`], this returns `Err` and
/// the caller writes nothing.
fn encode_quality(
    image: &jpxl_encode::Image,
    score: f64,
    options: &jpxl_encode::EncodeOptions,
    effort: jpxl::Effort,
    fallback: jpxl::QualityFallback,
    text_routing: bool,
) -> Result<(Vec<u8>, String, String), String> {
    let (width, height, bits_per_sample, rgb) = image_to_rgb16(image)?;
    let encoder = jpxl::Encoder::new()
        .with_resources(options.resources)
        .with_container(options.container)
        .with_jxlp_fragment_size(options.jxlp_fragment_size)
        .with_effort(effort)
        .with_quality_fallback(fallback)
        .with_text_routing(text_routing)
        .with_ssimulacra2_score(score)
        .map_err(|error| error.to_string())?;
    match encoder.encode_rgb16_reported(width, height, bits_per_sample, &rgb) {
        Ok((bytes, jpxl::EncodeReport::Perceptual(outcome))) => {
            let line = format_perceptual_line(&outcome, effort_name(effort));
            append_quality_trace(outcome.trace_json.as_deref());
            let mode = if outcome.status == jpxl::PerceptualStatus::FallbackLossless {
                format!("lossless Modular (fallback: ssimulacra2>={score:.4} unmet)")
            } else {
                format!("lossy VarDCT (perceptual), ssimulacra2>={score:.4}")
            };
            Ok((bytes, line, mode))
        }
        Ok(_) => Err("perceptual encode produced an unexpected report".to_owned()),
        Err(jpxl::Error::Unsupported(what)) => Err(format!("ssimulacra2>={score:.4}: {what}")),
        // A refused under-target encode still ran a full search; its trace is
        // calibration input, so the harness sees it even though nothing is
        // written.
        Err(jpxl::Error::TargetNotMet(miss)) => {
            append_quality_trace(miss.trace_json.as_deref());
            Err(jpxl::Error::TargetNotMet(miss).to_string())
        }
        Err(error) => Err(error.to_string()),
    }
}

/// Appends one `jpxl.quality-trace/2` record to the `JPXL_QUALITY_TRACE`
/// path, when both exist. The harness's and the predictor calibration's
/// input; a write failure warns and never fails the encode.
fn append_quality_trace(trace: Option<&str>) {
    if let (Some(path), Some(trace)) = (std::env::var_os("JPXL_QUALITY_TRACE"), trace) {
        use std::io::Write as _;
        let appended = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .and_then(|mut file| writeln!(file, "{trace}"));
        if let Err(error) = appended {
            eprintln!("warning: could not write JPXL_QUALITY_TRACE: {error}");
        }
    }
}

/// Wraps a lossy codestream the way [`jpxl_encode::encode`] wraps a lossless
/// one, so `--container` and `--jxlp` reach the VarDCT paths too.
///
/// 18181-2 9.3 with Annex M of Part 1: a >8-bit image needs the extended
/// level, which is why this takes the depth rather than assuming one.
fn wrap_for_output(
    codestream: Vec<u8>,
    bits_per_sample: u32,
    options: &jpxl_encode::EncodeOptions,
) -> Vec<u8> {
    if !options.container && options.jxlp_fragment_size.is_none() {
        return codestream;
    }
    let level = if bits_per_sample > 8 {
        jpxl_encode::container::EXTENDED_LEVEL
    } else {
        jpxl_encode::container::DEFAULT_LEVEL
    };
    match options.jxlp_fragment_size {
        Some(size) => jpxl_encode::container::wrap_fragmented(&codestream, level, size),
        None => jpxl_encode::container::wrap(&codestream, level),
    }
}

/// The `--global-scale` path: a fixed-quantizer VarDCT encode.
///
/// Honours `--quant-lf`; the HF multiplier stays at the request default.
fn encode_global_scale(
    image: &jpxl_encode::Image,
    global_scale: u32,
    quant_lf: Option<u32>,
    options: &jpxl_encode::EncodeOptions,
) -> Result<(Vec<u8>, String), String> {
    let (width, height, bits_per_sample, rgb) = image_to_rgb16(image)?;
    let defaults = jpxl_encode_policy::EncodeRequest::defaults();
    let quant_lf = quant_lf.unwrap_or_else(|| defaults.quant_lf.get());
    let target = jpxl_encode_policy::request::FixedQuantizerTarget::new(
        global_scale,
        quant_lf,
        defaults.hf_mul.get(),
    )
    .map_err(|error| error.to_string())?;
    let mut request = jpxl_encode_policy::EncodeRequest::for_fixed_quantizer(target);
    request.resources = options.resources;
    let bytes =
        jpxl_encode_policy::encode_srgb16_vardct(width, height, &rgb, bits_per_sample, &request)
            .map_err(|error| error.to_string())?;
    let bytes = wrap_for_output(bytes, bits_per_sample, options);
    let mode = format!("lossy VarDCT (fixed quantizer), global_scale {global_scale}");
    Ok((bytes, mode))
}

/// `encode --sections`: where the bytes of a lossy codestream went, by
/// section kind (F.3.1), so a density change can be attributed to HF
/// coefficients, LF/DC, entropy tables or headers.
fn print_section_breakdown(sizing: &jpxl_encode::vardct::CodestreamSizing, output: &str) {
    use jpxl_encode::vardct::SectionKind;
    let pick = |f: &dyn Fn(SectionKind) -> bool| sizing.bytes_where(f);
    let lf_global = pick(&|k| matches!(k, SectionKind::LfGlobal));
    let lf_groups = pick(&|k| matches!(k, SectionKind::LfGroup(_)));
    let hf_global = pick(&|k| matches!(k, SectionKind::HfGlobal));
    let pass_groups = pick(&|k| matches!(k, SectionKind::PassGroup { .. }));
    let whole = pick(&|k| matches!(k, SectionKind::Whole));
    let headers_toc = sizing.overhead();
    let total = sizing.total.max(1);
    #[allow(
        clippy::cast_precision_loss,
        reason = "byte counts stay far inside f64's exact integer range"
    )]
    let pct = |b: u64| b as f64 * 100.0 / total as f64;
    status_line(
        output,
        &format!(
            "  sections: total={} headers_toc={} ({:.1}%) lf_global={} ({:.1}%) lf_groups={} \
             ({:.1}%) hf_global={} ({:.1}%) pass_groups={} ({:.1}%) whole={} sections={}",
            sizing.total,
            headers_toc,
            pct(headers_toc),
            lf_global,
            pct(lf_global),
            lf_groups,
            pct(lf_groups),
            hf_global,
            pct(hf_global),
            pass_groups,
            pct(pass_groups),
            whole,
            sizing.sections.len(),
        ),
    );
}

/// Parses a sample-count argument: a plain integer, or `full` for "unbounded".
///
/// `full` is spelled out because `u64::MAX` is the value that means "do not
/// bound this", and writing that on a command line is unreadable.
fn parse_sample_budget(s: &str) -> Option<u64> {
    if s.eq_ignore_ascii_case("full") {
        return Some(u64::MAX);
    }
    s.parse::<u64>().ok()
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

/// Read a file, or stdin when `path` is `-`.
fn read_path(path: &str) -> std::io::Result<Vec<u8>> {
    if path == "-" {
        let mut bytes = Vec::new();
        std::io::stdin().lock().read_to_end(&mut bytes)?;
        Ok(bytes)
    } else {
        std::fs::read(Path::new(path))
    }
}

/// Write a file, or stdout when `path` is `-`.
fn write_path(path: &str, bytes: &[u8]) -> std::io::Result<()> {
    if path == "-" {
        let mut stdout = std::io::stdout().lock();
        stdout.write_all(bytes)?;
        stdout.flush()
    } else {
        std::fs::write(Path::new(path), bytes)
    }
}

/// Keep binary stdout clean in pipeline mode.
fn status_line(output: &str, message: &str) {
    if output == "-" {
        let _ = writeln!(std::io::stderr().lock(), "{message}");
    } else {
        println!("{message}");
    }
}

/// Print an error to stderr, with the usage hint.
fn fail(message: &str) {
    let mut stderr = std::io::stderr().lock();
    let _ = writeln!(stderr, "jpxl: error: {message}");
    let _ = writeln!(stderr, "try `jpxl --help`");
}

#[cfg(test)]
mod cli_tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use image::{DynamicImage, ImageBuffer, ImageFormat, Rgb};

    use super::*;

    static NEXT_TEMP: AtomicUsize = AtomicUsize::new(0);

    fn temp_dir() -> std::path::PathBuf {
        let id = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("jpxl-cli-test-{}-{id}", std::process::id()))
    }

    #[test]
    fn decoded_pair_metric_is_the_in_tree_production_score() {
        let (width, height) = (32u32, 32u32);
        let mut samples = Vec::with_capacity(32 * 32 * 3);
        for y in 0..32u16 {
            for x in 0..32u16 {
                samples.extend_from_slice(&[
                    x.saturating_mul(8),
                    y.saturating_mul(8),
                    x.saturating_add(y).saturating_mul(4),
                ]);
            }
        }
        let reference = jpxl_conformance::metrics::Image {
            w: width,
            h: height,
            channels: 3,
            max_value: 255,
            samples,
        };
        assert_eq!(
            in_tree_ssimulacra2_score(&reference, &reference),
            Some(100.0)
        );

        let mut distorted = reference.clone();
        for pixel in distorted.samples.chunks_exact_mut(3) {
            if let [_, _, blue] = pixel {
                *blue = blue.saturating_add(8).min(255);
            }
        }
        let distorted_score = in_tree_ssimulacra2_score(&reference, &distorted).expect("score");
        assert!(distorted_score < 100.0);
    }

    #[test]
    fn png_to_jxl_to_png_is_lossless() {
        let temp = temp_dir();
        std::fs::create_dir_all(&temp).expect("temp directory");
        let input = temp.join("input.png");
        let encoded = temp.join("encoded.jxl");
        let output = temp.join("output.png");
        let original = vec![1, 2, 3, 10, 20, 30, 100, 150, 200, 255, 254, 253];
        let dynamic = DynamicImage::ImageRgb8(
            ImageBuffer::<Rgb<u8>, _>::from_raw(2, 2, original.clone()).expect("shape"),
        );
        dynamic
            .save_with_format(&input, ImageFormat::Png)
            .expect("write input PNG");

        let encode_args = vec![
            "encode".to_owned(),
            input.to_string_lossy().into_owned(),
            encoded.to_string_lossy().into_owned(),
        ];
        assert_eq!(run(&encode_args), EXIT_OK);
        let decode_args = vec![
            "decode".to_owned(),
            encoded.to_string_lossy().into_owned(),
            output.to_string_lossy().into_owned(),
        ];
        assert_eq!(run(&decode_args), EXIT_OK);

        let roundtrip = image::open(&output).expect("open output PNG").into_rgb8();
        assert_eq!(roundtrip.into_raw(), original);

        if temp.starts_with(std::env::temp_dir()) {
            std::fs::remove_dir_all(&temp).expect("remove isolated test directory");
        }
    }

    #[test]
    fn analyze_atlas_exports_header_and_raster_order_atoms() {
        let temp = temp_dir();
        std::fs::create_dir_all(&temp).expect("temp directory");
        let input = temp.join("input.png");
        let output = temp.join("atlas.jsonl");
        let original = vec![128u8; 9 * 10 * 3];
        let dynamic = DynamicImage::ImageRgb8(
            ImageBuffer::<Rgb<u8>, _>::from_raw(9, 10, original).expect("shape"),
        );
        dynamic
            .save_with_format(&input, ImageFormat::Png)
            .expect("write input PNG");

        let args = vec![
            "analyze-atlas".to_owned(),
            input.to_string_lossy().into_owned(),
            output.to_string_lossy().into_owned(),
        ];
        assert_eq!(run(&args), EXIT_OK);
        let text = std::fs::read_to_string(&output).expect("read atlas");
        let rows: Vec<&str> = text.lines().collect();
        assert_eq!(rows.len(), 5);
        let header = rows.first().expect("header");
        assert!(header.contains("\"schema\":\"jpxl.analysis-atlas/1\""));
        assert!(header.contains("\"grid_width\":2"));
        assert!(header.contains("\"grid_height\":2"));
        assert!(header.contains("\"atom_bytes\":128"));
        assert!(rows.get(1).expect("first atom").contains("\"x\":0,\"y\":0"));
        assert!(rows.get(4).expect("last atom").contains("\"x\":1,\"y\":1"));

        if temp.starts_with(std::env::temp_dir()) {
            std::fs::remove_dir_all(&temp).expect("remove isolated test directory");
        }
    }
}
