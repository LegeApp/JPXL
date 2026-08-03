// SPDX-License-Identifier: MIT OR Apache-2.0
//! End-to-end extra-channel decoding and frame blending: extra channels in a
//! `kVarDCT` frame (18181-1 G.1.3, G.2.3, G.4.2), Table F.8's blend modes over
//! several regular frames (F.2), and K.3.2's per-channel patch blending — all
//! graded under 18181-3 §4.2.
//!
//! # The ladder, and what each rung proves
//!
//! Every rung asserts the construct it is named for is really present in the
//! stream *before* it grades pixels: a comparison against a decode that quietly
//! lost its alpha channel would otherwise pass for the wrong reason, and an
//! all-zero alpha plane over an opaque image looks exactly like a correct one
//! in every colour channel.
//!
//! 1. **An extra channel at all** — fixture 80, 8x8 RGBA lossless modular. The
//!    smallest expressible RGBA image: the alpha channel fits inside
//!    `group_dim`, so G.1.3's `GlobalModular` decodes the whole of it and
//!    neither group rule is reached.
//! 2. **The extra channel across pass groups** — fixture 81, 600x520 RGBA
//!    lossless modular, a 3x3 group grid. Larger than `group_dim` in both
//!    directions, so the alpha channel is split over G.4.2's per-group
//!    sub-bitstreams.
//! 3. **An extra channel in a `kVarDCT` frame** — fixture 82, 64x64 RGBA at
//!    `d = 1.0`. This is the wave's target construct: a `kVarDCT` frame has no
//!    modular colour channels, so before this its `GlobalModular` row was an
//!    empty sub-bitstream. With alpha present the row is live, and the samples
//!    ride sections that are otherwise full of VarDCT structures.
//! 4. **The same, multi-group** — fixture 83, 384x320 RGBA, four groups. Each
//!    pass group's modular data sits immediately *after* that group's HF
//!    coefficients in the same section (Table G.5), so a decoder that read it
//!    from the wrong place, or skipped it, desynchronises.
//! 5. **Greyscale plus alpha** — fixture 84, 128x128: one colour channel out,
//!    three XYB planes in.
//! 6. **The normative corpus.** Six published cases, each against its own
//!    `reference_image.npy` at its own `test.json` thresholds:
//!    * `alpha_nonpremultiplied` and `alpha_triangles` — modular alpha at 12
//!      and 9 bits.
//!    * `alpha_premultiplied` — a `kVarDCT` frame with 12-bit colour and a
//!      16-bit premultiplied alpha channel.
//!    * `patches` (and `patches_5`) — a `kVarDCT` frame with alpha *and* a K.3
//!      patch dictionary, whose 139 patches carry a blend rule per channel
//!      group.
//!    * `blendmodes` (and `blendmodes_5`) — five full-size regular frames
//!      composited with all five of Table F.8's modes, including the alpha
//!      channel's own formulas.
//!
//! # Skipping
//!
//! A fixture's reference decode is committed in-tree when it is under 100 KB
//! and regenerated from the pinned `djxl` otherwise; a rung whose reference is
//! neither available **skips**. The corpus NPYs are gitignored, so the corpus
//! rungs skip when the corpus has not been fetched. No test here fails for a
//! missing tool.
//!
//! # The corpus rungs are slow in a debug build
//!
//! The corpus cases are 1 to 1.75 Mpixel each and one of them composites five
//! frames. In a release build they take a few seconds together; an unoptimized
//! build is roughly twenty times slower. So they run unconditionally in a
//! release build and, in a debug build, only when `JPXL_SLOW_TESTS` is set —
//! the same rule `e2e_progressive.rs` uses, and for the same reason. They are
//! not `#[ignore]`d: in the configuration where they are affordable they must
//! run by default.

use std::path::{Path, PathBuf};

use jpxl_bitstream::BitReader;
use jpxl_conformance::{FloatImage, OracleKind, OutputFormat, Similarity, oracle, similarity};
use jpxl_core::limits::{AllocGuard, Limits};
use jpxl_decode::frame::{
    BlendMode, Encoding, FrameGeometry, FrameHeader, FrameType, read_frame_header, read_toc,
};
use jpxl_decode::headers::decode_image_headers_metered;
use jpxl_decode::{DecodedImage, decode};

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

fn handmade(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/handmade")
        .join(name)
}

fn corpus(case: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/conformance/testcases")
        .join(case)
}

/// Our decode, as the `FloatImage` 18181-3 §4.2 compares.
///
/// The float planes are authoritative wherever they exist — a `kVarDCT` decode
/// and a composited multi-frame one both produce them, colour channels first
/// and then the extra channels in `ec_info` order, which is the interleaving
/// `djxl --output_format npy` writes. A lossless modular decode has no float
/// planes and is exact, so its integer samples are put on the same nominal
/// `[0, 1]` scale here by dividing by each plane's own
/// `(1 << bits_per_sample) - 1` — G.4.2's last paragraph, which interprets the
/// colour channels by `metadata.bit_depth` and each extra channel by its own
/// `ec_info[i].bit_depth`. Those two depths differ on one of the corpus cases,
/// so a single shared divisor would fail it by a factor of 16.
fn as_float_image(image: &DecodedImage) -> FloatImage {
    let pixels = image.width as usize * image.height as usize;
    let (channels, samples) = match image.float_planes.as_ref() {
        Some(planes) => {
            let mut samples = Vec::with_capacity(pixels * planes.len());
            for y in 0..image.height {
                for x in 0..image.width {
                    for plane in planes {
                        samples.push(plane.get(x, y));
                    }
                }
            }
            (planes.len(), samples)
        }
        None => {
            let mut samples = Vec::with_capacity(pixels * image.planes.len());
            for y in 0..image.height {
                for x in 0..image.width {
                    for plane in &image.planes {
                        let max = plane.max_value() as f32;
                        samples.push(plane.get(x, y) as f32 / max);
                    }
                }
            }
            (image.planes.len(), samples)
        }
    };
    FloatImage {
        frames: 1,
        height: image.height,
        width: image.width,
        channels: u32::try_from(channels).expect("a sane channel count"),
        samples,
    }
}

fn decode_fixture(path: &Path) -> DecodedImage {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    decode(&bytes, &Limits::default()).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn read_npy(path: &Path) -> FloatImage {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    FloatImage::from_npy(&bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The committed reference decode, or one produced by the pinned oracle.
///
/// `None` is a **skip**, not a failure: CI must stay green on a machine with
/// no libjxl build.
fn reference_for(fixture: &Path) -> Option<FloatImage> {
    let committed = fixture.with_extension("npy");
    if committed.exists() {
        return Some(read_npy(&committed));
    }
    let oracle = oracle::find(OracleKind::Djxl)?;
    if !oracle.supports(OutputFormat::Npy) {
        return None;
    }
    let dir = std::env::temp_dir().join("jpxl-e2e-alpha");
    std::fs::create_dir_all(&dir).ok()?;
    let out = dir.join(format!("{}.npy", fixture.file_stem()?.to_string_lossy()));
    oracle.decode(fixture, &out, OutputFormat::Npy).ok()?;
    Some(read_npy(&out))
}

/// Grades one decode and prints the numbers, so a passing run still reports
/// how much margin it had.
fn grade(label: &str, ours: &FloatImage, reference: &FloatImage) -> Similarity {
    assert_eq!(
        (ours.width, ours.height, ours.channels),
        (reference.width, reference.height, reference.channels),
        "{label}: 18181-3 4.2 condition 1 (shape) failed"
    );
    let report = similarity(ours, reference).expect("same shape");
    println!(
        "{label}: peak {:.8}, channel peak {:?}, channel RMSE {:?}",
        report.peak_error, report.channel_peak, report.channel_rmse
    );
    report
}

fn assert_conforms(label: &str, report: &Similarity, peak: f32, rmse: f32) {
    assert!(
        report.conforms(peak, rmse),
        "{label}: peak {:.8} (limit {peak}), channel RMSE {:?} (limit {rmse})",
        report.peak_error,
        report.channel_rmse
    );
}

// ---------------------------------------------------------------------------
// Structural assertions: what the fixture actually signals
// ---------------------------------------------------------------------------

/// Every frame header of a codestream, parsed by JPXL's own reader.
///
/// The same walk `decode` performs — a frame's sections are addressed from
/// just past its TOC, and the TOC's total size advances the cursor to the next
/// frame — reproduced here so a structural claim about a fixture is checked
/// against the bitstream rather than against a comment.
fn frame_headers(codestream: &[u8]) -> Vec<FrameHeader> {
    let limits = Limits::default();
    let mut guard = AllocGuard::new(&limits);
    let mut reader = BitReader::new(codestream);
    let headers = decode_image_headers_metered(&mut reader, &limits, &mut guard).expect("headers");
    if headers.metadata.colour_encoding.want_icc {
        jpxl_decode::icc::read_icc_profile(&mut reader, &mut guard).expect("ICC profile");
    }
    reader.zero_pad_to_byte().expect("F.1 alignment");
    let mut cursor = usize::try_from(reader.total_bits_read() / 8).expect("in range");

    let mut out = Vec::new();
    loop {
        let rest = codestream.get(cursor..).expect("frame start in range");
        let mut frame_reader = BitReader::new(rest);
        let header = read_frame_header(
            &mut frame_reader,
            &headers.metadata,
            headers.width(),
            headers.height(),
            &limits,
            &mut guard,
        )
        .expect("frame header");
        let geometry = FrameGeometry::from_header(
            &header,
            headers.width(),
            headers.height(),
            &limits,
            &mut guard,
        )
        .expect("frame geometry");
        let toc = read_toc(
            &mut frame_reader,
            geometry.num_sections(),
            &limits,
            &mut guard,
        )
        .expect("TOC");
        cursor += usize::try_from(frame_reader.total_bits_read() / 8).expect("in range")
            + usize::try_from(toc.total_size()).expect("in range");
        let is_last = header.is_last;
        out.push(header);
        if is_last {
            return out;
        }
    }
}

/// The codestream of a fixture, unwrapped from its container if it has one.
fn codestream_of(path: &Path) -> Vec<u8> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    if bytes.starts_with(&[0xFF, 0x0A]) {
        return bytes;
    }
    let limits = Limits::default();
    let mut guard = AllocGuard::new(&limits);
    jpxl_decode::container::extract_codestream(&bytes, &mut guard).expect("a Part 2 container")
}

/// The image header's extra-channel count, read by JPXL's own reader.
fn num_extra_channels(codestream: &[u8]) -> usize {
    let limits = Limits::default();
    let mut guard = AllocGuard::new(&limits);
    let mut reader = BitReader::new(codestream);
    decode_image_headers_metered(&mut reader, &limits, &mut guard)
        .expect("headers")
        .metadata
        .num_extra()
}

/// Asserts a fixture really carries `extra` extra channels in a frame of the
/// given encoding and group count, that our decode returns them as planes, and
/// that the alpha plane is not constant — then grades it.
///
/// The non-constant check is what makes this rung mean anything: an image
/// whose alpha was dropped decodes to an all-zero (or all-max) plane, which is
/// a perfectly plausible alpha channel and is caught by nothing else.
fn extra_channel_rung(
    name: &str,
    encoding: Encoding,
    extra: usize,
    groups: u64,
    peak: f32,
    rmse: f32,
) {
    let path = handmade(name);
    assert!(path.exists(), "missing fixture {name}");
    let codestream = codestream_of(&path);
    assert_eq!(
        num_extra_channels(&codestream),
        extra,
        "{name}: this rung only proves anything if the image really has \
         {extra} extra channel(s)"
    );

    let headers = frame_headers(&codestream);
    let frame = headers
        .iter()
        .find(|h| h.frame_type == FrameType::RegularFrame)
        .unwrap_or_else(|| panic!("{name}: no regular frame"));
    assert_eq!(frame.encoding, encoding, "{name}: frame encoding");
    assert_eq!(
        frame.ec_upsampling,
        vec![1; extra],
        "{name}: the rung assumes no K.2 extra-channel upsampling"
    );

    let limits = Limits::default();
    let mut guard = AllocGuard::new(&limits);
    let geometry =
        FrameGeometry::from_header(frame, frame.width, frame.height, &limits, &mut guard)
            .expect("geometry");
    assert_eq!(
        geometry.num_groups(),
        groups,
        "{name}: group count — the multi-group rungs are only multi-group if \
         this is above one"
    );

    let image = decode_fixture(&path);
    assert_eq!(
        image.planes.len(),
        image.num_colour_channels + extra,
        "{name}: the decode must return the extra channels as planes"
    );
    let alpha = image
        .planes
        .get(image.num_colour_channels)
        .unwrap_or_else(|| panic!("{name}: no alpha plane"));
    let first = alpha.samples.first().copied().unwrap_or(0);
    assert!(
        alpha.samples.iter().any(|&v| v != first),
        "{name}: the alpha plane is constant, which is what a dropped extra \
         channel looks like"
    );

    let Some(reference) = reference_for(&path) else {
        eprintln!("skipping {name}: no committed .npy and no djxl oracle");
        return;
    };
    let report = grade(name, &as_float_image(&image), &reference);
    assert_conforms(name, &report, peak, rmse);

    // The extra channels themselves are coded losslessly in every fixture of
    // this ladder — they are modular integers, and cjxl does not quantize them
    // at these settings — so their error is the f32 round-trip and nothing
    // else. Asserting that separately keeps the rung's own subject held to a
    // tolerance the colour channels' lossy budget cannot hide.
    for channel in image.num_colour_channels..image.num_colour_channels + extra {
        let observed = report.channel_peak.get(channel).copied().unwrap_or(0.0);
        assert!(
            observed <= 1e-6,
            "{name}: extra channel {channel} peak error {observed:e}, which is \
             larger than a lossless round-trip"
        );
    }
}

// ---------------------------------------------------------------------------
// Rungs 1-2 — an extra channel in a kModular frame
// ---------------------------------------------------------------------------

/// The smallest expressible RGBA image: G.1.3 decodes the alpha channel whole.
///
/// Lossless, so the only tolerance needed is the `f32` round-trip of the
/// integers — an exact decode scores zero here, not merely a small number.
#[test]
fn fixture_80_modular_rgba_single_group() {
    extra_channel_rung(
        "80_alpha_rgba_8x8_lossless.jxl",
        Encoding::Modular,
        1,
        1,
        1e-6,
        1e-6,
    );
}

/// 600x520: bigger than `group_dim` in both directions, so the alpha channel
/// is split over a 3x3 grid of G.4.2 pass groups.
#[test]
fn fixture_81_modular_rgba_multi_group() {
    extra_channel_rung(
        "81_alpha_rgba_600x520_lossless.jxl",
        Encoding::Modular,
        1,
        9,
        1e-6,
        1e-6,
    );
}

// ---------------------------------------------------------------------------
// Rungs 3-5 — an extra channel in a kVarDCT frame
// ---------------------------------------------------------------------------

/// The wave's target construct at its smallest: one group, one section.
#[test]
fn fixture_82_vardct_rgba_single_group() {
    extra_channel_rung(
        "82_alpha_vardct_rgba_64x64_nofilters.jxl",
        Encoding::VarDct,
        1,
        1,
        0.004,
        1e-5,
    );
}

/// Four groups: each pass group's modular data follows that group's HF
/// coefficients inside the same section.
#[test]
fn fixture_83_vardct_rgba_multi_group() {
    extra_channel_rung(
        "83_alpha_vardct_rgba_384x320_nofilters.jxl",
        Encoding::VarDct,
        1,
        4,
        0.004,
        1e-5,
    );
}

/// Greyscale plus alpha: one colour channel out, three XYB planes in.
///
/// # Why the colour RMSE bound is not the no-filters class
///
/// The grey channel of this fixture carries a systematic ~1.9e-4 RMSE against
/// the oracle that has **nothing to do with the extra channel**: encoding the
/// identical grey source with no alpha at all reproduces it to within 2e-8,
/// and the same content in colour (fixture 83) grades at 4.8e-6. It is a
/// pre-existing greyscale-VarDCT residual of the same family as the corpus
/// `grayscale` case's 2.3e-4 peak, and it is recorded in
/// `docs/experiments/2026-08-04-vardct-extra-channels-and-frame-blending.md`
/// §5 rather than papered over. The peak bound stays at the no-filters class,
/// and the alpha channel — this rung's actual subject — is held to a lossless
/// round-trip by `extra_channel_rung`.
#[test]
fn fixture_84_vardct_greyscale_alpha() {
    extra_channel_rung(
        "84_alpha_vardct_ga_128x128_nofilters.jxl",
        Encoding::VarDct,
        1,
        1,
        0.004,
        3e-4,
    );
}

// ---------------------------------------------------------------------------
// Rung 6 — the normative corpus
// ---------------------------------------------------------------------------

/// The Part 3 error class of a corpus case, read from its own `test.json`.
///
/// A three-line scan rather than a JSON parser: the workspace takes no
/// third-party dependencies, and the two numbers wanted are on their own lines
/// in every corpus `test.json`.
fn corpus_thresholds(case: &Path) -> Option<(f32, f32)> {
    let text = std::fs::read_to_string(case.join("test.json")).ok()?;
    let find = |key: &str| -> Option<f32> {
        text.lines()
            .find(|l| l.contains(key))
            .and_then(|l| l.split(':').nth(1))
            .map(|v| v.trim().trim_end_matches(',').to_owned())
            .and_then(|v| v.parse().ok())
    };
    Some((find("peak_error")?, find("rms_error")?))
}

/// Whether the corpus decodes are affordable in this build. See the module
/// documentation.
fn slow_tests_enabled() -> bool {
    !cfg!(debug_assertions) || std::env::var_os("JPXL_SLOW_TESTS").is_some()
}

/// Decodes one corpus case and grades it against its own `test.json`.
///
/// Returns the decoded image so a caller can assert more about it, and `None`
/// when the rung skipped.
fn corpus_rung(case: &str) -> Option<DecodedImage> {
    let dir = corpus(case);
    let input = dir.join("input.jxl");
    let reference = dir.join("reference_image.npy");
    if !input.exists() || !reference.exists() {
        eprintln!("skipping corpus {case}: not fetched (see tools/fetch-conformance.sh)");
        return None;
    }
    if !slow_tests_enabled() {
        eprintln!(
            "skipping corpus {case}: a 1-Mpixel decode is slow unoptimized. \
             Re-run with --release, or set JPXL_SLOW_TESTS=1."
        );
        return None;
    }
    let image = decode_fixture(&input);
    let report = grade(case, &as_float_image(&image), &read_npy(&reference));
    let (peak, rmse) = corpus_thresholds(&dir).unwrap_or_else(|| panic!("{case}/test.json"));
    assert_conforms(case, &report, peak, rmse);
    Some(image)
}

/// 12-bit modular RGB with a 12-bit unassociated alpha channel.
#[test]
fn corpus_alpha_nonpremultiplied() {
    let Some(image) = corpus_rung("alpha_nonpremultiplied") else {
        return;
    };
    assert_eq!(image.planes.len(), 4, "RGB + alpha");
    assert_eq!(
        image.planes.get(3).map(|p| p.bits_per_sample),
        Some(12),
        "the alpha channel keeps its own ec_info bit depth"
    );
}

/// 9-bit modular RGBA — colour and alpha at the same depth, but neither a
/// whole number of bytes.
#[test]
fn corpus_alpha_triangles() {
    let _ = corpus_rung("alpha_triangles");
}

/// A `kVarDCT` frame with 12-bit colour and a 16-bit premultiplied alpha
/// channel: the target construct of this wave, on the normative corpus.
///
/// The two depths differ, which is the case a decoder that scaled every plane
/// by `metadata.bit_depth` gets wrong by a factor of 16.
#[test]
fn corpus_alpha_premultiplied() {
    let Some(image) = corpus_rung("alpha_premultiplied") else {
        return;
    };
    assert_eq!(
        image.planes.get(3).map(|p| p.bits_per_sample),
        Some(16),
        "ec_info[0].bit_depth is 16 while metadata.bit_depth is 12"
    );
}

/// A `kVarDCT` frame with alpha *and* a K.3 patch dictionary, graded at both
/// published error classes.
///
/// Both features at once is the point: K.3.1's per-position blend rules are
/// `num_extra + 1` long, so the dictionary's *bit count* depends on the extra
/// channel count, and K.3.2 then blends each channel group under its own rule.
#[test]
fn corpus_patches() {
    if corpus_rung("patches").is_none() {
        return;
    }
    // `patches_5` is the same content at the looser Part 3 error class; it has
    // its own codestream, so it gets its own decode.
    let _ = corpus_rung("patches_5");
}

/// Five full-size regular frames composited with every mode in Table F.8.
///
/// The feature assertion is what makes this a blend-mode test rather than a
/// five-frame test: the codestream is required to contain `kReplace`,
/// `kBlend`, `kAdd`, `kMul` and `kMulAdd`, on the colour channels and on the
/// alpha channel alike, so every row of the table — including the two special
/// formulas Table F.8 gives for "the alpha channel itself" — is exercised.
#[test]
fn corpus_blendmodes() {
    let dir = corpus("blendmodes");
    if !dir.join("input.jxl").exists() {
        eprintln!("skipping corpus blendmodes: not fetched");
        return;
    }
    let codestream = codestream_of(&dir.join("input.jxl"));
    let headers = frame_headers(&codestream);
    let regular: Vec<&FrameHeader> = headers
        .iter()
        .filter(|h| h.frame_type == FrameType::RegularFrame)
        .collect();
    assert_eq!(
        regular.len(),
        5,
        "blendmodes composites five regular frames"
    );
    for mode in [
        BlendMode::Replace,
        BlendMode::Add,
        BlendMode::Blend,
        BlendMode::Mul,
        BlendMode::MulAdd,
    ] {
        assert!(
            regular.iter().any(|h| h.blending_info.mode == mode),
            "blendmodes: no frame uses {mode:?} on its colour channels"
        );
        assert!(
            regular
                .iter()
                .any(|h| h.ec_blending_info.first().map(|i| i.mode) == Some(mode)),
            "blendmodes: no frame uses {mode:?} on its alpha channel"
        );
    }
    // Every frame but the last is a source for the next one.
    assert!(
        regular.iter().take(4).all(|h| h.can_reference()),
        "blendmodes: the composition chain needs each frame stored as a reference"
    );

    if corpus_rung("blendmodes").is_none() {
        return;
    }
    // The same five-frame composition at the looser Part 3 error class, from
    // its own codestream.
    let _ = corpus_rung("blendmodes_5");
}
