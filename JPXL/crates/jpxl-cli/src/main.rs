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
    jpxl encode [opts] <in> <out> Encode a binary PGM (P5) or PPM (P6); lossless
                                  modular by default, lossy VarDCT with --bpp
    jpxl compare <ref.ppm> <b.ppm>
                                  Print RMSE and PSNR between two decoded PPMs
                                  (plus SSIMULACRA2 and butteraugli, if built
                                  with --features perceptual)
    jpxl bench <mode> [opts]      Time one encode path (see `jpxl bench --help`)
    jpxl --help                   Show this message
    jpxl --version                Show the version

Encode options:
    --container                   Wrap the codestream in a Part 2 container
    --effort <1..9>               Lossless search effort (default 1 = fastest).
                                  Higher is slower and only occasionally smaller;
                                  every level is exact-lossless (pixels identical).
    --group-size-shift <0..3>     Force group_dim = 128 << shift (default 2)
    --jxlp <bytes>                Split the codestream across jxlp boxes
                                  (18181-2 9.10); implies --container
    --threads <n>                 Section-parallel workers (default: host
                                  available_parallelism; 1 = serial)

Lossy options (8-bit RGB only; either one selects the VarDCT path):
    --bpp <f>                     Target bits per pixel
    --target-bytes <n>            Target output size in bytes
    --aq-mode <mode>              Per-block HF allocation: off (target-rate
                                  default), masking, or uniform; research control
    --aq-strength <f>             Activity-field strength; research control
    --aq-clamp <f>                Activity-field clamp; research control
    --aq-chroma <f>               Activity-field chroma weight; research control
    --quant-lf <n>                Hold the LF quantizer at n and disable the
                                  secondary LF fill; research control
    --x-qm-scale <0..7>           X-channel QM exponent (default 2); research
                                  chroma-allocation control
    --b-qm-scale <0..7>           B-channel QM exponent (default 2); research
                                  chroma-allocation control
    --epf-iters <0..3>            Decoder EPF iteration count (target-rate
                                  default 1); research control
    --epf-sharpness <mode>        EPF sharpness plane: zero (fixed default) or
                                  uniform7 (target-rate default); research control

    There is no `--distance`. cjxl's -d targets butteraugli; JPXL has no
    perceptual model, so its rate loop hits a *size*, not a visual quality.
    Naming a flag --distance would promise something this encoder cannot
    deliver. --effort is lossless-only and is ignored on the lossy path.

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

Prints one line per run: mode, size, iters, wall_ms_total, wall_ms_median,
output_bytes, fingerprint. With --diag, a second `diag=...` line follows.

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
    let mut rate_target: Option<jpxl_encode_policy::RateTarget> = None;
    let mut aq_mode: Option<jpxl_encode_policy::AqMode> = None;
    let mut aq_tuning = jpxl_encode_policy::AqTuning::default();
    let mut fixed_quant_lf: Option<u32> = None;
    let mut x_qm_scale: Option<u8> = None;
    let mut b_qm_scale: Option<u8> = None;
    let mut epf_iters: Option<u8> = None;
    let mut epf_sharpness: Option<jpxl_encode_policy::EpfSharpnessMode> = None;
    let mut positional: Vec<&String> = Vec::new();
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--container" => options.container = true,
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
                    fail("`--aq-mode` needs one of: off, masking, uniform");
                    return EXIT_ERROR;
                };
                aq_mode = Some(match value.as_str() {
                    "off" => jpxl_encode_policy::AqMode::Off,
                    "masking" => jpxl_encode_policy::AqMode::Masking,
                    "uniform" => jpxl_encode_policy::AqMode::Uniform,
                    _ => {
                        fail("`--aq-mode` needs one of: off, masking, uniform");
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
                    _ => {
                        fail("`--epf-sharpness` needs one of: zero, uniform7");
                        return EXIT_ERROR;
                    }
                });
            }
            "--target-bytes" => {
                let Some(v) = rest.next().and_then(|v| v.parse::<u64>().ok()) else {
                    fail("`--target-bytes` needs a positive byte count");
                    return EXIT_ERROR;
                };
                rate_target = Some(jpxl_encode_policy::RateTarget::Bytes(v));
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

    match std::fs::write(Path::new(output.as_str()), &encoded) {
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
            println!(
                "{output}: {}x{}, {} channel(s), {} bits per sample, {mode}, {} bytes",
                image.width(),
                image.height(),
                image.num_channels(),
                image.bits_per_sample(),
                encoded.len()
            );
            // A missed rate target is not a failure — the loop's contract is
            // "never over" — but it is silent unless said out loud, and the two
            // reasons for it want opposite responses. Saturated means the
            // ladder ran out of rungs and more budget cannot help; unsaturated
            // means the search ran out of prices short of the target.
            if let Some(report) = lossy {
                let miss = report.target_bytes.saturating_sub(report.achieved);
                let slack = report.target_bytes / 100; // the loop's own 1% tolerance
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
                    println!(
                        "  note: undershot target by {miss} bytes ({pct:.1}%) — {why} \
                         [prices: {} fast, {} full]",
                        report.fast_prices, report.full_prices
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

    let timed = match mode {
        "modular" => bench_modular(&rgb, width, height, iters, resources, effort, overrides),
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
                                 dct_cache_hits={} dct_cache_misses={} \
                                 dct_cache_hit_rate={hit_rate:.4}",
                                s.fast_prices, s.full_prices, s.dct_cache_hits, s.dct_cache_misses,
                            );
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
    /// Rate-search multiplicity counters from the last timed iteration.
    /// `vardct-rate` only; every other mode leaves this `None`.
    rate_stats: Option<jpxl_encode_policy::RateProbeStats>,
    /// Control-flow and rung summary from the last timed rate search.
    rate_trace: Option<RateTraceStats>,
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
) -> Result<BenchReport, String> {
    let target = jpxl_encode_policy::RateTarget::BitsPerPixel(bpp);
    let mut request = jpxl_encode_policy::EncodeRequest::for_target(target);
    request.resources = resources;
    let warm = jpxl_encode_policy::encode_srgb8_to_target(width, height, rgb, &request, target)
        .map_err(|e| e.to_string())?;
    let warm_len = warm.codestream.len();
    let warm_fp = fnv1a64(&warm.codestream);
    // Rate-loop-specific telemetry from the last timed iteration; attached to
    // `time_iters`'s shared report below.
    let last_stats = std::cell::Cell::new(warm.stats);
    let last_trace = std::cell::Cell::new(RateTraceStats::from_outcome(&warm));
    let mut report = time_iters(iters, warm_len, warm_fp, || {
        let outcome =
            jpxl_encode_policy::encode_srgb8_to_target(width, height, rgb, &request, target)
                .map_err(|e| e.to_string())?;
        last_stats.set(outcome.stats);
        last_trace.set(RateTraceStats::from_outcome(&outcome));
        Ok(outcome.codestream)
    })?;
    report.rate_stats = Some(last_stats.get());
    report.rate_trace = Some(last_trace.get());
    Ok(report)
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
        rate_stats: None,
        rate_trace: None,
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

/// Encodes `image` through the lossy VarDCT rate loop to `target`.
///
/// The policy crate's façade takes interleaved sRGB8, so this re-interleaves
/// the validated planes rather than re-reading the file. 8-bit RGB only: the
/// VarDCT path converts sRGB8 to XYB and has no route for greyscale or deeper
/// samples yet, so anything else is refused rather than silently mangled.
/// What a lossy encode did, beyond the bytes: enough to tell a *hit* target
/// from a *missed* one and to say why it was missed.
struct LossyReport {
    codestream: Vec<u8>,
    target_bytes: u64,
    achieved: u64,
    /// The ladder ran out of rungs — the target is finer than the quantizer can
    /// express. More search budget cannot help.
    saturated: bool,
    fast_prices: u32,
    full_prices: u32,
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
}

fn encode_lossy_to_target(
    image: &jpxl_encode::Image,
    target: jpxl_encode_policy::RateTarget,
    overrides: LossyOverrides,
) -> Result<LossyReport, String> {
    if image.num_channels() != 3 || image.bits_per_sample() != 8 {
        return Err(format!(
            "lossy encoding needs 8-bit RGB (P6 with maxval 255); this is {} channel(s) at {} bits",
            image.num_channels(),
            image.bits_per_sample()
        ));
    }
    let planes = image.planes();
    let (Some(r), Some(g), Some(b)) = (planes.first(), planes.get(1), planes.get(2)) else {
        return Err("lossy encoding needs three colour planes".to_owned());
    };
    let mut rgb = Vec::with_capacity(r.len().saturating_mul(3));
    for i in 0..r.len() {
        for plane in [r, g, b] {
            // Planes are validated to [0, 255] for an 8-bit image, so the
            // clamp is belt-and-braces rather than load-bearing.
            let v = plane.get(i).copied().unwrap_or(0).clamp(0, 255);
            rgb.push(u8::try_from(v).unwrap_or(0));
        }
    }
    let mut request = jpxl_encode_policy::EncodeRequest::for_target(target);
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
        request.x_qm_scale = jpxl_encode::vardct::ids::QmScale::new(value)
            .map_err(|_| format!("x_qm_scale {value} is outside the wire range"))?;
    }
    if let Some(value) = overrides.b_qm_scale {
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
    jpxl_encode_policy::encode_srgb8_to_target(
        image.width(),
        image.height(),
        &rgb,
        &request,
        target,
    )
    .map(|outcome| LossyReport {
        target_bytes: outcome.target,
        achieved: outcome.achieved(),
        saturated: outcome.saturated,
        fast_prices: outcome.stats.fast_prices,
        full_prices: outcome.stats.full_prices,
        codestream: outcome.codestream,
    })
    .map_err(|e| e.to_string())
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

/// Print an error to stderr, with the usage hint.
fn fail(message: &str) {
    let mut stderr = std::io::stderr().lock();
    let _ = writeln!(stderr, "jpxl: error: {message}");
    let _ = writeln!(stderr, "try `jpxl --help`");
}
