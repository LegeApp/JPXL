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
                                  Print RMSE and PSNR between two decoded PPMs
                                  (plus SSIMULACRA2 and butteraugli, if built
                                  with --features perceptual)
    jpxl bench <mode> [opts]      Time one encode path (see `jpxl bench --help`)
    jpxl --help                   Show this message
    jpxl --version                Show the version

Encode options:
    --background <#RRGGBB>       Explicitly flatten a transparent input;
                                  transparent JPEG XL output is not yet encoded
    --container                   Wrap the codestream in a Part 2 container
    --effort <1..9>               Lossless search effort (default 1 = fastest).
                                  Higher is slower and only occasionally smaller;
                                  every level is exact-lossless (pixels identical).
    --group-size-shift <0..3>     Force group_dim = 128 << shift (default 2)
    --jxlp <bytes>                Split the codestream across jxlp boxes
                                  (18181-2 9.10); implies --container
    --threads <n>                 Section-parallel workers (default: host
                                  available_parallelism; 1 = serial)

Decode options:
    -f, --format <name>           Output format; required for stdout (`-`),
                                  otherwise inferred from the output extension
    --background <#RRGGBB>       Flatten alpha when writing JPEG or PNM

Lossy options (8- or 16-bit RGB; either one selects the VarDCT path):
    --bpp <f>                     Target bits per pixel
    --target-bytes <n>            Target output size in bytes
    --lossy-preset <mode>         Rate controller: balanced (default production
                                  path), fast (lower-latency production path),
                                  or quality (exhaustive reference)
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
    --x-qm-scale <0..7>           X-channel QM exponent (Balanced defaults to 3;
                                  Quality defaults to 3 at <=1 bpp and 2 above
                                  it); setting it pins the manual chroma policy
    --b-qm-scale <0..7>           B-channel QM exponent (Balanced defaults to 3;
                                  Quality defaults to 5 at <=1 bpp and 4 above
                                  it; Fast defaults to 2); research
                                  chroma-allocation control
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

    There is no `--distance`. cjxl's -d targets butteraugli; JPXL has no
    perceptual model, so its rate loop hits a *size*, not a visual quality.
    Naming a flag --distance would promise something this encoder cannot
    deliver. --effort is lossless-only and is ignored on the lossy path.

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
CfL, AQ) — only the rate loop is off. See sources/outside-advice.md §2.

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
                    fail("`--lossy-preset` needs one of: quality, balanced, fast");
                    return EXIT_ERROR;
                };
                rate_preset = Some(match value.as_str() {
                    "quality" => jpxl_encode_policy::RateSearchPreset::Quality,
                    "balanced" => jpxl_encode_policy::RateSearchPreset::Balanced,
                    "fast" => jpxl_encode_policy::RateSearchPreset::Fast,
                    _ => {
                        fail("`--lossy-preset` needs one of: quality, balanced, fast");
                        return EXIT_ERROR;
                    }
                });
            }
            "--effort" => {
                let Some(level) = rest.next().and_then(|v| v.parse::<u8>().ok()) else {
                    fail("`--effort` needs an integer in 1..=9");
                    return EXIT_ERROR;
                };
                match jpxl_encode::Effort::new(level) {
                    Ok(effort) => options.effort = effort,
                    Err(_) => {
                        fail("`--effort` needs an integer in 1..=9");
                        return EXIT_ERROR;
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

    // A rate target selects the lossy VarDCT path; without one this stays the
    // lossless modular encoder it has always been.
    let mut lossy: Option<LossyReport> = None;
    let encoded = match rate_target {
        Some(target) => {
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
                    let bytes = report.codestream.clone();
                    lossy = Some(report);
                    bytes
                }
                Err(err) => {
                    fail(&format!("{input}: {err}"));
                    return EXIT_ERROR;
                }
            }
        }
        None => match jpxl_encode::encode(&image, &options) {
            Ok(encoded) => encoded,
            Err(err) => {
                fail(&format!("{input}: {err}"));
                return EXIT_ERROR;
            }
        },
    };

    match write_path(output, &encoded) {
        Ok(()) => {
            let mode = match rate_target {
                Some(jpxl_encode_policy::RateTarget::BitsPerPixel(b)) => {
                    format!("lossy VarDCT, target {b} bpp")
                }
                Some(jpxl_encode_policy::RateTarget::Bytes(n)) => {
                    format!("lossy VarDCT, target {n} bytes")
                }
                None => format!("lossless modular, effort {}", options.effort.level()),
            };
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
                            "  anchor: fallbacks={} first_finalist_bytes={} correction_bytes={}                          fast_prices={} full_prices={}",
                            report.stats.anchor_fallbacks,
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
                    let why = if report.saturated {
                        "ladder saturated: the finest quantizer is still under target, \
                         so no extra search budget can close this"
                    } else {
                        "search ended short of target with budget spent, not at the \
                         ladder's limit"
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
/// Built with `--features perceptual`, also prints `ssimulacra2=<f>`
/// (higher is better, 100 = identical) and `butteraugli=<f>` with its
/// `butteraugli_pnorm3` (lower is better, 0 = identical; `butteraugli` is the
/// distance `cjxl -d` targets). Those are the numbers to rank lossy encoders
/// on — PSNR is printed because it is always available, not because it is
/// right.
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

    // Only the perceptual blocks below append to this, so with both features
    // off it is never mutated.
    #[cfg_attr(
        not(any(feature = "ssimulacra2", feature = "butteraugli")),
        allow(unused_mut, reason = "appended to only under the perceptual features")
    )]
    let mut line = if db.is_infinite() {
        "rmse=0 psnr_db=inf".to_owned()
    } else {
        format!("rmse={err:.6} psnr_db={db:.4}")
    };

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
                                 exact_candidates={} anchor_fallbacks={} \
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
         count_ms={:.1} store_ms={:.1} pool_build_ms={:.1} tape_symbols={}",
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
                RatePhase::Final => out.final_prices += 1,
            }
        }

        let fast_best = outcome
            .trace
            .iter()
            .filter(|step| step.phase != RatePhase::Final && step.feasible)
            .max_by_key(|step| (step.bytes, step.quantizer.rung));
        out.fast_best_rung = fast_best.map(|step| step.quantizer.rung.get());
        out.fast_best_quant_lf = fast_best.map(|step| step.quantizer.quant_lf.get());
        out.fast_upper_rung = fast_best.and_then(|best| {
            outcome
                .trace
                .iter()
                .filter(|step| {
                    step.phase != RatePhase::Final
                        && !step.feasible
                        && step.quantizer.rung > best.quantizer.rung
                })
                .min_by_key(|step| step.quantizer.rung)
                .map(|step| step.quantizer.rung.get())
        });
        let full_start = outcome
            .trace
            .iter()
            .find(|step| step.phase == RatePhase::Final);
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
    /// The ladder ran out of rungs — the target is finer than the quantizer can
    /// express. More search budget cannot help.
    saturated: bool,
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
        saturated: outcome.saturated,
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
}
