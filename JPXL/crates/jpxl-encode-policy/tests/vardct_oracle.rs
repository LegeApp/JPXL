//! The headline claim of slice 12: **other people's decoders** read the
//! kVarDCT streams this encoder writes, and agree with ours about the pixels.
//!
//! `vardct_roundtrip.rs` proves the encoder and `jpxl-decode` agree, which is
//! necessary and insufficient: two implementations written from one reading of
//! one clause fail together. These tests replace one side with a decoder JPXL
//! had no hand in:
//!
//! * `djxl` (libjxl, the reference implementation), through its PPM output;
//! * `jxl-oxide` (an independent Rust decoder), through its `.npy` output.
//!
//! # What each comparison means
//!
//! Two different things are measured, and conflating them is how a lossy
//! encoder ends up with a meaningless "tolerance":
//!
//! * **decoder against decoder, on one stream.** This is the conformance-shaped
//!   comparison, and it is the one a Part 3 peak-error class is defined for:
//!   the same coefficients, the same dequantization, the same inverse
//!   transform, so any difference is a decoder bug or a float-rounding
//!   difference. The bound asserted here is one 8-bit code point, which is the
//!   quantization step of the output itself.
//! * **decoder against the encoder's source.** This is rate-distortion, not
//!   conformance — the encoder threw information away on purpose. The bound is
//!   a stated RMSE, and it is a claim about *this encoder at this quantizer*,
//!   not about conformance.
//!
//! # Skipping, not failing
//!
//! A checkout with no oracle installed stays green: each test returns early
//! with a printed note. A *present* oracle that disagrees is a hard failure.

#![allow(
    clippy::cast_possible_truncation,
    clippy::indexing_slicing,
    reason = "test-only image synthesis and .npy parsing, over data this file \
              produced itself"
)]

use std::path::{Path, PathBuf};

use jpxl_conformance::{Image as PnmImage, OracleKind, OutputFormat, oracle};
use jpxl_core::limits::Limits;
use jpxl_decode::decode::decode;
use jpxl_encode_policy::{
    EncodeRequest, PreparedFrame, RateTarget, encode_srgb8_to_target, encode_srgb8_vardct,
    plan_frame,
};

/// A distinct directory per test, so parallel runs cannot collide.
fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("jpxl-vardct-oracle-{tag}"));
    std::fs::create_dir_all(&dir).expect("scratch directory");
    dir
}

/// One rung of the ladder `docs/PLAN.md` slice 12 prescribes.
struct Case {
    name: &'static str,
    width: u32,
    height: u32,
    grey: bool,
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "grey-8x8",
            width: 8,
            height: 8,
            grey: true,
        },
        Case {
            name: "rgb-8x8",
            width: 8,
            height: 8,
            grey: false,
        },
        Case {
            name: "rgb-64x64",
            width: 64,
            height: 64,
            grey: false,
        },
        // 300x260 at kVarDCT's fixed group_dim of 256 is a 2x2 pass-group
        // grid: F.3.1's multi-section TOC and four independent ANS streams.
        Case {
            name: "rgb-multigroup-300x260",
            width: 300,
            height: 260,
            grey: false,
        },
        // Partial blocks on both the right and the bottom edge.
        Case {
            name: "rgb-61x37",
            width: 61,
            height: 37,
            grey: false,
        },
        // An LF group is 2048 samples per side, so 2100 wide is the first
        // width that needs two of them.
        Case {
            name: "rgb-two-lf-groups-2100x24",
            width: 2100,
            height: 24,
            grey: false,
        },
    ]
}

/// The same deterministic image `vardct_roundtrip.rs` uses.
fn test_image(width: u32, height: u32, grey: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let ramp = u8::try_from((x * 255) / width.max(1)).unwrap_or(255);
            let fall = u8::try_from(255 - (y * 255) / height.max(1)).unwrap_or(255);
            let edge = if x * 3 > width * 2 { 40u8 } else { 0 };
            let checker = if (x / 4 + y / 4) % 2 == 0 { 25u8 } else { 0 };
            let luma = ramp.saturating_add(checker).saturating_sub(edge);
            if grey {
                out.extend_from_slice(&[luma, luma, luma]);
            } else {
                out.extend_from_slice(&[
                    luma,
                    fall.saturating_sub(edge),
                    ramp.saturating_add(fall / 2).saturating_sub(checker),
                ]);
            }
        }
    }
    out
}

/// `jpxl-decode`'s 8-bit sRGB samples, interleaved as a PPM stores them.
fn ours(codestream: &[u8]) -> Vec<u8> {
    let image = decode(codestream, &Limits::default()).expect("jpxl-decode accepts it");
    let count = (image.width * image.height) as usize;
    let mut out = Vec::with_capacity(count * 3);
    for i in 0..count {
        for plane in image.planes.iter().take(3) {
            let sample = plane.samples.get(i).copied().unwrap_or(0);
            out.push(u8::try_from(sample.clamp(0, 255)).unwrap_or(0));
        }
    }
    out
}

/// Peak absolute error and RMSE, in 8-bit code points.
fn error(a: &[u8], b: &[u8]) -> (u32, f64) {
    assert_eq!(a.len(), b.len(), "same sample count");
    let mut peak = 0u32;
    let mut sum = 0f64;
    for (&p, &q) in a.iter().zip(b) {
        let d = u32::from(p.abs_diff(q));
        peak = peak.max(d);
        sum += f64::from(d) * f64::from(d);
    }
    (peak, (sum / a.len() as f64).sqrt())
}

/// Encodes one case and writes the codestream, returning it and the source.
fn encode_case(case: &Case, dir: &Path) -> (Vec<u8>, Vec<u8>, PathBuf) {
    let source = test_image(case.width, case.height, case.grey);
    let frame = PreparedFrame::from_srgb8(case.width, case.height, &source)
        .unwrap_or_else(|e| panic!("{}: prepare failed: {e}", case.name));
    let plan = plan_frame(&frame, &EncodeRequest::defaults())
        .unwrap_or_else(|e| panic!("{}: plan failed: {e}", case.name));
    if !case.grey {
        let non_neutral_lf = plan.plan().spatial.lf.correlation.x_factor_lf != 128
            || plan.plan().spatial.lf.correlation.b_factor_lf != 128;
        let non_neutral_hf = plan.plan().spatial.lf_groups.iter().any(|group| {
            group
                .cfl
                .x_from_y()
                .iter()
                .chain(group.cfl.b_from_y())
                .any(|factor| factor.get() != 0)
        });
        assert!(
            non_neutral_lf || non_neutral_hf,
            "{}: the oracle fixture must put non-neutral CfL on the wire",
            case.name
        );
    }
    let codestream = jpxl_encode::vardct::write_codestream(&plan)
        .unwrap_or_else(|e| panic!("{}: encode failed: {e}", case.name));
    let jxl = dir.join(format!("{}.jxl", case.name));
    std::fs::write(&jxl, &codestream).expect("write");
    (codestream, source, jxl)
}

/// The tolerance class this slice claims against the source image.
///
/// Not a Part 3 class: see the module documentation. Measured worst values at
/// the default quantizer are peak 56 and RMSE 11.1 (the 8x8 RGB rung, where
/// one block carries a saturated synthetic pattern); every other rung is
/// inside RMSE 7.3.
const MAX_RMSE_VS_SOURCE: f64 = 14.0;

/// The tolerance between two decoders reading the *same* stream.
///
/// One 8-bit code point: the output quantization step. Anything larger is a
/// decoder disagreement, not a rounding difference.
const MAX_PEAK_BETWEEN_DECODERS: u32 = 1;

#[test]
fn djxl_decodes_our_vardct_output() {
    let Some(oracle) = oracle::find(OracleKind::Djxl) else {
        println!("skipping: djxl is not installed");
        return;
    };
    let dir = scratch("djxl");

    for case in cases() {
        let (codestream, source, jxl) = encode_case(&case, &dir);
        let ppm = dir.join(format!("{}.ppm", case.name));
        match oracle.decode(&jxl, &ppm, OutputFormat::Ppm) {
            Ok(()) => {}
            Err(err) if err.is_unavailable() => {
                println!("skipping: {err}");
                return;
            }
            Err(err) => panic!(
                "{}: djxl refused our codestream ({} bytes): {err}",
                case.name,
                codestream.len()
            ),
        }

        let bytes = std::fs::read(&ppm).unwrap_or_else(|e| panic!("{}: {e}", case.name));
        let decoded = PnmImage::from_ppm(&bytes).unwrap_or_else(|e| panic!("{}: {e}", case.name));
        assert_eq!(
            (decoded.w, decoded.h),
            (case.width, case.height),
            "{}: dimensions",
            case.name
        );
        assert_eq!(decoded.channels, 3, "{}: djxl writes P6", case.name);
        assert_eq!(
            u32::from(decoded.max_value),
            255,
            "{}: bit depth",
            case.name
        );

        let samples: Vec<u8> = decoded
            .samples
            .iter()
            .map(|&v| u8::try_from(v.min(255)).unwrap_or(0))
            .collect();

        // Conformance-shaped: djxl and jpxl-decode on the same coefficients.
        let (peak, rmse) = error(&samples, &ours(&codestream));
        assert!(
            peak <= MAX_PEAK_BETWEEN_DECODERS,
            "{}: djxl and jpxl-decode disagree by {peak} (RMSE {rmse:.3})",
            case.name
        );

        // Rate-distortion: djxl's pixels against what was encoded.
        let (peak, rmse) = error(&samples, &source);
        assert!(
            rmse <= MAX_RMSE_VS_SOURCE,
            "{}: djxl vs source RMSE {rmse:.3} (peak {peak}, {} bytes)",
            case.name,
            codestream.len()
        );
    }
}

#[test]
fn jxl_oxide_decodes_our_vardct_output() {
    let Some(oracle) = oracle::find(OracleKind::JxlOxide) else {
        println!("skipping: jxl-oxide is not installed");
        return;
    };
    let dir = scratch("jxl-oxide");

    for case in cases() {
        let (codestream, source, jxl) = encode_case(&case, &dir);
        let npy = dir.join(format!("{}.npy", case.name));
        match oracle.decode(&jxl, &npy, OutputFormat::Npy) {
            Ok(()) => {}
            Err(err) if err.is_unavailable() => {
                println!("skipping: {err}");
                return;
            }
            Err(err) => panic!(
                "{}: jxl-oxide refused our codestream ({} bytes): {err}",
                case.name,
                codestream.len()
            ),
        }

        let bytes = std::fs::read(&npy).unwrap_or_else(|e| panic!("{}: {e}", case.name));
        let values = read_npy_f32(&bytes).unwrap_or_else(|e| panic!("{}: {e}", case.name));
        assert_eq!(
            values.len(),
            source.len(),
            "{}: sample count (one frame expected)",
            case.name
        );
        // The `.npy` convention is normalised f32 in the signalled colour
        // encoding; quantizing to 8 bits is what puts it on the same scale as
        // a PPM and as `jpxl-decode`'s integer planes.
        let samples: Vec<u8> = values
            .iter()
            .map(|&v| {
                let scaled = (v.clamp(0.0, 1.0) * 255.0).round();
                u8::try_from(scaled as i32).unwrap_or(255)
            })
            .collect();

        let (peak, rmse) = error(&samples, &ours(&codestream));
        assert!(
            peak <= MAX_PEAK_BETWEEN_DECODERS,
            "{}: jxl-oxide and jpxl-decode disagree by {peak} (RMSE {rmse:.3})",
            case.name
        );

        let (peak, rmse) = error(&samples, &source);
        assert!(
            rmse <= MAX_RMSE_VS_SOURCE,
            "{}: jxl-oxide vs source RMSE {rmse:.3} (peak {peak}, {} bytes)",
            case.name,
            codestream.len()
        );
    }
}

/// Slice 14's streams are slice 12's streams: a codestream whose quantizer was
/// chosen by the rate loop is decoded by both external decoders, and to the
/// same pixels ours produces.
///
/// This is the gate that keeps rate control honest. The loop searches a ladder
/// of `global_scale` and `HfMul` values that nothing else in the test suite
/// visits — including quantizers far coarser and far finer than the default —
/// and a value that only *our* decoder tolerates would otherwise sail through.
#[test]
fn both_oracles_decode_a_rate_targeted_stream() {
    let case = Case {
        name: "rate-targeted-300x260",
        width: 300,
        height: 260,
        grey: false,
    };
    let source = test_image(case.width, case.height, case.grey);

    for (tag, target) in [
        ("tight", RateTarget::Bytes(4_000)),
        ("moderate", RateTarget::Bytes(8_000)),
        ("generous", RateTarget::BitsPerPixel(1.5)),
    ] {
        let outcome = encode_srgb8_to_target(
            case.width,
            case.height,
            &source,
            &EncodeRequest::defaults(),
            target,
        )
        .unwrap_or_else(|e| panic!("{tag}: rate loop failed: {e}"));
        assert!(outcome.achieved() <= outcome.target, "{tag}: over budget");

        let dir = scratch("rate");
        let jxl = dir.join(format!("{}-{tag}.jxl", case.name));
        std::fs::write(&jxl, &outcome.codestream).expect("write");
        let ours = ours(&outcome.codestream);

        for kind in [OracleKind::Djxl, OracleKind::JxlOxide] {
            let Some(oracle) = oracle::find(kind) else {
                println!("skipping {kind:?}: not installed");
                continue;
            };
            let (out, format) = match kind {
                OracleKind::Djxl => (
                    dir.join(format!("{}-{tag}.ppm", case.name)),
                    OutputFormat::Ppm,
                ),
                _ => (
                    dir.join(format!("{}-{tag}.npy", case.name)),
                    OutputFormat::Npy,
                ),
            };
            match oracle.decode(&jxl, &out, format) {
                Ok(()) => {}
                Err(err) if err.is_unavailable() => {
                    println!("skipping {kind:?}: {err}");
                    continue;
                }
                Err(err) => panic!(
                    "{tag}: {kind:?} refused a rate-targeted stream ({} bytes, \
                     global_scale {}, HfMul {}): {err}",
                    outcome.achieved(),
                    outcome.chosen.global_scale.get(),
                    outcome.chosen.hf_mul.get()
                ),
            }
            let bytes = std::fs::read(&out).unwrap_or_else(|e| panic!("{tag}: {e}"));
            let samples: Vec<u8> = match kind {
                OracleKind::Djxl => PnmImage::from_ppm(&bytes)
                    .unwrap_or_else(|e| panic!("{tag}: {e}"))
                    .samples
                    .iter()
                    .map(|&v| u8::try_from(v.min(255)).unwrap_or(0))
                    .collect(),
                _ => read_npy_f32(&bytes)
                    .unwrap_or_else(|e| panic!("{tag}: {e}"))
                    .iter()
                    .map(|&v| {
                        u8::try_from((v.clamp(0.0, 1.0) * 255.0).round() as i32).unwrap_or(255)
                    })
                    .collect(),
            };
            let (peak, rmse) = error(&samples, &ours);
            assert!(
                peak <= MAX_PEAK_BETWEEN_DECODERS,
                "{tag}: {kind:?} and jpxl-decode disagree by {peak} (RMSE {rmse:.3})"
            );
        }
    }
}

/// `HfMul > 1` is new wire content — G.2.4's `BlockInfo` second row stops being
/// all zeros — and the rate ladder is the first thing that emits it. An
/// external decoder is the only witness that says so correctly.
#[test]
fn both_oracles_decode_a_stream_from_the_hf_mul_segment() {
    let (width, height) = (61u32, 37u32);
    let source = test_image(width, height, false);
    let mut request = EncodeRequest::defaults();
    request.global_scale =
        jpxl_encode::vardct::ids::GlobalScale::new(jpxl_encode::vardct::ids::GlobalScale::MAX)
            .expect("legal");
    let ceiling = encode_srgb8_vardct(width, height, &source, &request)
        .expect("encodes")
        .len() as u64;
    let outcome = encode_srgb8_to_target(
        width,
        height,
        &source,
        &request,
        RateTarget::Bytes(ceiling * 2),
    )
    .expect("reachable");
    assert!(outcome.chosen.hf_mul.get() > 1, "not an HfMul stream");

    let dir = scratch("hfmul");
    let jxl = dir.join("hfmul.jxl");
    std::fs::write(&jxl, &outcome.codestream).expect("write");
    let ours = ours(&outcome.codestream);

    if let Some(oracle) = oracle::find(OracleKind::Djxl) {
        let ppm = dir.join("hfmul.ppm");
        match oracle.decode(&jxl, &ppm, OutputFormat::Ppm) {
            Ok(()) => {
                let bytes = std::fs::read(&ppm).expect("read");
                let decoded = PnmImage::from_ppm(&bytes).expect("ppm");
                let samples: Vec<u8> = decoded
                    .samples
                    .iter()
                    .map(|&v| u8::try_from(v.min(255)).unwrap_or(0))
                    .collect();
                let (peak, rmse) = error(&samples, &ours);
                assert!(
                    peak <= MAX_PEAK_BETWEEN_DECODERS,
                    "djxl and jpxl-decode disagree by {peak} (RMSE {rmse:.3}) on HfMul {}",
                    outcome.chosen.hf_mul.get()
                );
            }
            Err(err) if err.is_unavailable() => println!("skipping djxl: {err}"),
            Err(err) => panic!(
                "djxl refused an HfMul {} stream: {err}",
                outcome.chosen.hf_mul.get()
            ),
        }
    }
    if let Some(oracle) = oracle::find(OracleKind::JxlOxide) {
        let npy = dir.join("hfmul.npy");
        match oracle.decode(&jxl, &npy, OutputFormat::Npy) {
            Ok(()) => {
                let bytes = std::fs::read(&npy).expect("read");
                let samples: Vec<u8> = read_npy_f32(&bytes)
                    .expect("npy")
                    .iter()
                    .map(|&v| {
                        u8::try_from((v.clamp(0.0, 1.0) * 255.0).round() as i32).unwrap_or(255)
                    })
                    .collect();
                let (peak, rmse) = error(&samples, &ours);
                assert!(
                    peak <= MAX_PEAK_BETWEEN_DECODERS,
                    "jxl-oxide and jpxl-decode disagree by {peak} (RMSE {rmse:.3})"
                );
            }
            Err(err) if err.is_unavailable() => println!("skipping jxl-oxide: {err}"),
            Err(err) => panic!(
                "jxl-oxide refused an HfMul {} stream: {err}",
                outcome.chosen.hf_mul.get()
            ),
        }
    }
}

/// Slice 16's new wire content: DctSelect values other than zero, and the HF
/// coefficient walk of DCT16x16/DCT32x32 varblocks. Only a decoder JPXL had
/// no hand in can say the placement, the LLF handling and the coefficient
/// order of the merged transforms are the standard's and not merely our own.
#[test]
fn both_oracles_decode_a_hierarchical_stream() {
    let mut request = EncodeRequest::defaults();
    request.budget.cover_mode = jpxl_encode_policy::CoverMode::Hierarchical;

    // A gradient over a multi-group frame with clipped edges: merges
    // everywhere the grid allows them, boundary fallbacks where it does not.
    let case = Case {
        name: "hierarchical-300x260",
        width: 300,
        height: 260,
        grey: false,
    };
    let mut source = Vec::new();
    for y in 0..case.height {
        for x in 0..case.width {
            let luma = u8::try_from(40 + (x + y) * 160 / (case.width + case.height)).unwrap_or(255);
            source.extend_from_slice(&[
                luma,
                u8::try_from(u16::from(luma) * 4 / 5).unwrap_or(255),
                u8::try_from(u16::from(luma) * 3 / 5).unwrap_or(255),
            ]);
        }
    }

    let frame = PreparedFrame::from_srgb8(case.width, case.height, &source).expect("frame");
    let plan = plan_frame(&frame, &request).expect("a legal plan");
    let merged = plan
        .plan()
        .spatial
        .lf_groups
        .iter()
        .flat_map(|g| g.blocks.iter())
        .filter(|b| b.transform != jpxl_core::varblock::TransformType::Dct8x8)
        .count();
    assert!(
        merged > 0,
        "the oracle fixture must put a larger transform on the wire"
    );
    let codestream = jpxl_encode::vardct::write_codestream(&plan).expect("encodes");

    let dir = scratch("hierarchical");
    let jxl = dir.join(format!("{}.jxl", case.name));
    std::fs::write(&jxl, &codestream).expect("write");
    let ours = ours(&codestream);

    for kind in [OracleKind::Djxl, OracleKind::JxlOxide] {
        let Some(oracle) = oracle::find(kind) else {
            println!("skipping {kind:?}: not installed");
            continue;
        };
        let (out, format) = match kind {
            OracleKind::Djxl => (dir.join(format!("{}.ppm", case.name)), OutputFormat::Ppm),
            _ => (dir.join(format!("{}.npy", case.name)), OutputFormat::Npy),
        };
        match oracle.decode(&jxl, &out, format) {
            Ok(()) => {}
            Err(err) if err.is_unavailable() => {
                println!("skipping {kind:?}: {err}");
                continue;
            }
            Err(err) => panic!(
                "{kind:?} refused a hierarchical stream ({} bytes, {merged} merged blocks): {err}",
                codestream.len()
            ),
        }
        let bytes = std::fs::read(&out).expect("read oracle output");
        let samples: Vec<u8> = match kind {
            OracleKind::Djxl => {
                let decoded = PnmImage::from_ppm(&bytes).expect("ppm");
                assert_eq!(
                    (decoded.w, decoded.h),
                    (case.width, case.height),
                    "dimensions"
                );
                decoded
                    .samples
                    .iter()
                    .map(|&v| u8::try_from(v.min(255)).unwrap_or(0))
                    .collect()
            }
            _ => read_npy_f32(&bytes)
                .expect("npy")
                .iter()
                .map(|&v| u8::try_from((v.clamp(0.0, 1.0) * 255.0).round() as i32).unwrap_or(255))
                .collect(),
        };

        // Conformance-shaped: the same stream through two decoders.
        let (peak, rmse) = error(&samples, &ours);
        assert!(
            peak <= MAX_PEAK_BETWEEN_DECODERS,
            "{kind:?} and jpxl-decode disagree by {peak} on merged transforms (RMSE {rmse:.3})"
        );

        // Rate-distortion: the merged transforms must not wreck the pixels.
        let (_, rmse) = error(&samples, &source);
        assert!(
            rmse <= MAX_RMSE_VS_SOURCE,
            "{kind:?} vs source RMSE {rmse:.3} ({} bytes)",
            codestream.len()
        );
    }
}

/// A corrupted stream must be refused, never silently mis-decoded.
///
/// The ANS terminal state of C.3.2 is the check that makes this cheap: a
/// pass-group payload that has been altered lands the decoder on a state that
/// is not the one the encoder finished on. This is the encoder-side half of
/// slice 11.5's corruption gate, on a real VarDCT stream.
#[test]
fn a_corrupted_pass_group_is_rejected_by_our_decoder() {
    let case = Case {
        name: "rgb-64x64",
        width: 64,
        height: 64,
        grey: false,
    };
    let source = test_image(case.width, case.height, case.grey);
    let codestream =
        encode_srgb8_vardct(case.width, case.height, &source, &EncodeRequest::defaults())
            .expect("encodes");

    let mut caught = 0usize;
    let mut attempted = 0usize;
    // Only the tail is perturbed: the headers are covered by their own tests,
    // and the point here is the entropy-coded payload.
    let start = codestream.len() / 2;
    for offset in (start..codestream.len()).step_by(7) {
        let mut broken = codestream.clone();
        if let Some(byte) = broken.get_mut(offset) {
            *byte ^= 0x5A;
        }
        attempted += 1;
        if decode(&broken, &Limits::default()).is_err() {
            caught += 1;
        }
    }
    assert!(attempted > 10, "enough corruptions to be meaningful");
    assert!(
        caught * 2 > attempted,
        "only {caught} of {attempted} payload corruptions were caught"
    );
}

/// Reads the little-endian `f32` payload of a NumPy `.npy` file.
fn read_npy_f32(bytes: &[u8]) -> Result<Vec<f32>, String> {
    const MAGIC: &[u8] = b"\x93NUMPY";
    if bytes.get(..MAGIC.len()) != Some(MAGIC) {
        return Err("not a .npy file".into());
    }
    let header_len = bytes
        .get(8..10)
        .ok_or("truncated .npy header length")
        .map(|b| usize::from(u16::from_le_bytes([b[0], b[1]])))?;
    let header = bytes
        .get(10..10 + header_len)
        .ok_or("truncated .npy header")?;
    let header = core::str::from_utf8(header).map_err(|_| "non-UTF-8 .npy header")?;
    if !header.contains("'<f4'") {
        return Err(format!("unexpected .npy dtype in {header}"));
    }
    let body = bytes.get(10 + header_len..).ok_or("truncated .npy body")?;
    Ok(body
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
