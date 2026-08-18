// SPDX-License-Identifier: MIT
//! End-to-end progressive decoding: `kLFFrame` + `kUseLfFrame` (18181-1 F.2,
//! G.2.2) and multi-pass HF coefficients (F.2 Table F.6, I.4), graded under
//! 18181-3 §4.2.
//!
//! # The ladder, and what each rung proves
//!
//! Each handmade rung isolates one construct, and every rung is asserted to
//! *contain* that construct before it is graded — a pixel comparison against
//! a stream that quietly lost its second pass would pass for the wrong
//! reason.
//!
//! 1. **Multi-pass accumulation** — fixture 70, three passes with every
//!    `shift` zero. Proves the per-pass coefficient orders (I.3.1), the
//!    per-pass histogram bundles (I.3.3), the per-pass ANS restart, and I.4's
//!    accumulation across passes. No LF frame is involved.
//! 2. **The per-pass left shift** — fixture 71, two passes with
//!    `shift = [1]`. Fixture 70 is structurally blind to F.2's shift (all its
//!    shifts are zero); this rung is not.
//! 3. **`kUseLfFrame`** — fixture 72, one pass, LF supplied by a `kLFFrame`.
//!    Proves the whole substitution: G.2.2 and I.5.2 skipped, LF read from
//!    `LFFrame[0]` through L.2.2's `kModular` pre-step, and I.4's `lf_idx`
//!    pinned to zero.
//! 4. **The combination** — fixture 73, the smallest stream with the same
//!    feature set as the conformance case.
//! 5. **Multi-group and colour** — fixture 74, 384x320 RGB: a 2x2 group grid,
//!    so eight pass-group sections whose (pass, group) interleaving has to be
//!    right, and three colour planes that cannot be swapped without showing.
//! 6. **The normative corpus** — `progressive` and `progressive_5` against
//!    their published `reference_image.npy` at their own `test.json`
//!    thresholds. `progressive_5/input.jxl` is a symlink to
//!    `progressive/input.jxl`: one stream, two error classes, so a single
//!    decode is graded twice.
//!
//! # Skipping
//!
//! The greyscale fixtures ship their `.npy` in-tree (65 KB each); fixture 74's
//! would be 1.4 MB, so it is regenerated from the pinned `djxl` when that is
//! available and the rung **skips** otherwise. The corpus NPYs are gitignored,
//! so the corpus rung skips when the corpus has not been fetched. No test in
//! this file fails for a missing tool.
//!
//! # The corpus rung is slow in a debug build
//!
//! `progressive` is 4064x2704 — 11 Mpixel, 275x the area of the `grayscale`
//! corpus case. A release build decodes it in about 20 s; an unoptimized one
//! takes about 4.5 minutes, which would dominate `cargo test --workspace`
//! several times over. So the corpus rung runs unconditionally in a release
//! build and, in a debug build, only when `JPXL_SLOW_TESTS` is set. Run
//! `cargo test --release -p jpxl-decode --test e2e_progressive` (or set the
//! variable) to exercise it; it is not `#[ignore]`d, because in the build
//! configuration where it is affordable it must run by default.

use std::path::{Path, PathBuf};

use jpxl_bitstream::BitReader;
use jpxl_conformance::{FloatImage, OracleKind, OutputFormat, Similarity, oracle, similarity};
use jpxl_core::limits::{AllocGuard, Limits};
use jpxl_decode::frame::{Encoding, FrameHeader, FrameType, read_frame_header};
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
/// Uses the unclipped `f32` planes, never the quantized integer ones:
/// quantizing to 8 bits would put a floor of `1/510` under the peak error,
/// and the no-filters class is `0.004`.
fn as_float_image(image: &DecodedImage) -> FloatImage {
    let planes = image
        .float_planes
        .as_ref()
        .expect("a kVarDCT decode must produce float planes");
    let channels = planes.len();
    let mut samples = Vec::with_capacity(image.width as usize * image.height as usize * channels);
    for y in 0..image.height {
        for x in 0..image.width {
            for plane in planes {
                samples.push(plane.get(x, y));
            }
        }
    }
    FloatImage {
        frames: 1,
        height: image.height,
        width: image.width,
        channels: u32::try_from(channels).expect("at most 3 colour channels"),
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
    let dir = std::env::temp_dir().join("jpxl-e2e-progressive");
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
        "{label}: peak {:.6}, channel peak {:?}, channel RMSE {:?}",
        report.peak_error, report.channel_peak, report.channel_rmse
    );
    report
}

fn assert_conforms(label: &str, report: &Similarity, peak: f32, rmse: f32) {
    assert!(
        report.conforms(peak, rmse),
        "{label}: peak {:.6} (limit {peak}), channel RMSE {:?} (limit {rmse})",
        report.peak_error,
        report.channel_rmse
    );
}

// ---------------------------------------------------------------------------
// Structural assertions: what the fixture actually signals
// ---------------------------------------------------------------------------

/// Every frame header of a codestream, parsed by JPXL's own reader.
///
/// A frame's sections are addressed from just past its TOC, and the TOC's
/// total size is what advances the cursor to the next frame — the same walk
/// `decode` performs, reproduced here so a structural claim about a fixture is
/// checked against the bitstream rather than against a comment.
fn frame_headers(codestream: &[u8]) -> Vec<FrameHeader> {
    use jpxl_decode::frame::{FrameGeometry, read_toc};

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

/// The regular frame of a codestream — the one `decode` returns pixels for.
fn regular_frame(headers: &[FrameHeader]) -> &FrameHeader {
    headers
        .iter()
        .find(|h| h.frame_type == FrameType::RegularFrame)
        .expect("a codestream has a regular frame")
}

/// Asserts that `name` signals `num_passes` passes with the given shifts, and
/// grades it.
fn multipass_rung(name: &str, num_passes: u32, shift: &[u32], peak: f32, rmse: f32) {
    let path = handmade(name);
    assert!(path.exists(), "missing fixture {name}");
    let bytes = std::fs::read(&path).expect("readable");
    let headers = frame_headers(&bytes);
    let frame = regular_frame(&headers);
    assert_eq!(
        frame.passes.num_passes, num_passes,
        "{name}: this rung only proves anything if the frame really has \
         {num_passes} passes"
    );
    assert_eq!(frame.passes.shift, shift, "{name}: per-pass shifts");
    graded(name, &path, peak, rmse);
}

/// Asserts that `name` carries a `kLFFrame` whose LF the regular frame
/// consumes, and grades it.
fn lf_frame_rung(name: &str, groups: u64, peak: f32, rmse: f32) {
    use jpxl_decode::frame::FrameGeometry;

    let path = handmade(name);
    assert!(path.exists(), "missing fixture {name}");
    let bytes = std::fs::read(&path).expect("readable");
    let headers = frame_headers(&bytes);

    let lf = headers
        .iter()
        .find(|h| h.frame_type == FrameType::LfFrame)
        .unwrap_or_else(|| panic!("{name}: no kLFFrame in the codestream"));
    assert_eq!(lf.lf_level, 1, "{name}: LF frame level");
    assert_eq!(
        lf.encoding,
        Encoding::Modular,
        "{name}: the LF frame is kModular, so L.2.2's pre-step applies"
    );

    let frame = regular_frame(&headers);
    assert!(
        frame.flags.use_lf_frame(),
        "{name}: the regular frame must signal kUseLfFrame, or G.2.2's \
         substitution is never reached"
    );

    let limits = Limits::default();
    let mut guard = AllocGuard::new(&limits);
    let geometry =
        FrameGeometry::from_header(frame, frame.width, frame.height, &limits, &mut guard)
            .expect("geometry");
    assert_eq!(
        geometry.num_groups(),
        groups,
        "{name}: group count — the multi-group rung is only multi-group if \
         this is above one"
    );

    graded(name, &path, peak, rmse);
}

/// Decodes and grades one handmade fixture, skipping if no reference exists.
fn graded(name: &str, path: &Path, peak: f32, rmse: f32) {
    let Some(reference) = reference_for(path) else {
        eprintln!("skipping {name}: no committed .npy and no djxl oracle");
        return;
    };
    let image = decode_fixture(path);
    let report = grade(name, &as_float_image(&image), &reference);
    assert_conforms(name, &report, peak, rmse);
}

// ---------------------------------------------------------------------------
// Rungs 1-2 — multi-pass HF, no LF frame
// ---------------------------------------------------------------------------

/// Three passes, every `shift` zero: I.4's accumulation on its own.
#[test]
fn fixture_70_three_passes_no_shift() {
    multipass_rung(
        "70_progressive_ac_gray_128x128_nofilters.jxl",
        3,
        &[0, 0],
        0.004,
        1e-5,
    );
}

/// Two passes with `shift[0] = 1`: adds F.2's per-pass left shift, which
/// fixture 70 cannot see.
#[test]
fn fixture_71_two_passes_with_shift() {
    multipass_rung(
        "71_qprogressive_ac_gray_128x128_nofilters.jxl",
        2,
        &[1],
        0.004,
        1e-5,
    );
}

// ---------------------------------------------------------------------------
// Rungs 3-5 — kLFFrame
// ---------------------------------------------------------------------------

/// A single-pass frame whose LF comes from a `kLFFrame`: G.2.2's substitution
/// with nothing else moving.
#[test]
fn fixture_72_lf_frame_single_pass() {
    lf_frame_rung("72_lfframe_gray_128x128_nofilters.jxl", 1, 0.004, 1e-5);
}

/// LF frame plus two shifted passes, one group — the conformance case's
/// feature set at the smallest size that expresses it.
#[test]
fn fixture_73_lf_frame_and_passes() {
    lf_frame_rung(
        "73_progressive_all_gray_128x128_nofilters.jxl",
        1,
        0.004,
        1e-5,
    );
}

/// The same on 384x320 RGB: four groups, so eight pass-group sections, and
/// three colour planes through the LF frame.
///
/// Skips without a `djxl` oracle — its reference decode is 1.4 MB, over this
/// tree's commit cutoff.
#[test]
fn fixture_74_lf_frame_and_passes_multigroup_rgb() {
    lf_frame_rung(
        "74_progressive_all_rgb_384x320_nofilters.jxl",
        4,
        0.004,
        1e-5,
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

/// Whether the 11-Mpixel corpus decode is affordable in this build.
///
/// See the module documentation: unconditional in a release build, opt-in via
/// `JPXL_SLOW_TESTS` in a debug one.
fn slow_tests_enabled() -> bool {
    !cfg!(debug_assertions) || std::env::var_os("JPXL_SLOW_TESTS").is_some()
}

/// `progressive` and `progressive_5`: the two published conformance cases for
/// this feature set, decoded once and graded twice.
///
/// The two cases share one codestream — `progressive_5/input.jxl` is a symlink
/// to `progressive/input.jxl` — and differ only in their error class
/// (`progressive` is peak 0.02 / RMSE 1e-4, `progressive_5` peak 0.06 / RMSE
/// 0.02). Decoding once and grading against both `test.json` files is the same
/// assertion as two tests would make, at half the runtime, and it is the only
/// rung here that measures conformance to the *standard* rather than agreement
/// with libjxl: the reference image ships with the corpus.
///
/// The stream needs, in one frame sequence: a `kReferenceOnly` 29x28 modular
/// frame for K.3's patch dictionary, a `kLFFrame` at `lf_level = 1` whose
/// `GlobalModular` image is Squeezed across four groups, and a `kUseLfFrame`
/// `kVarDCT` regular frame of 176 groups over two passes with `shift = [1]`,
/// under `want_icc` (so clause 4's linear output).
#[test]
fn corpus_progressive() {
    let dir = corpus("progressive");
    let dir_5 = corpus("progressive_5");
    let input = dir.join("input.jxl");
    let reference = dir.join("reference_image.npy");
    if !input.exists() || !reference.exists() {
        eprintln!("skipping corpus progressive: not fetched (see tools/fetch-conformance.sh)");
        return;
    }
    if !slow_tests_enabled() {
        eprintln!(
            "skipping corpus progressive: 4064x2704 takes ~4.5 min unoptimized. \
             Re-run with --release, or set JPXL_SLOW_TESTS=1."
        );
        return;
    }

    let bytes = std::fs::read(&input).expect("readable");
    let image = decode(&bytes, &Limits::default()).expect("progressive/input.jxl decodes");
    let ours = as_float_image(&image);
    let report = grade("progressive", &ours, &read_npy(&reference));

    let (peak, rmse) = corpus_thresholds(&dir).expect("progressive/test.json");
    assert_conforms("progressive", &report, peak, rmse);

    if dir_5.join("test.json").exists() {
        let (peak_5, rmse_5) = corpus_thresholds(&dir_5).expect("progressive_5/test.json");
        // The two cases are the same bitstream; if the corpus ever stops
        // symlinking them this assertion is what says so.
        assert_eq!(
            std::fs::read(dir_5.join("input.jxl")).expect("readable"),
            bytes,
            "progressive_5 is no longer the same codestream as progressive; \
             it needs its own decode"
        );
        let reference_5 = dir_5.join("reference_image.npy");
        let report_5 = grade("progressive_5", &ours, &read_npy(&reference_5));
        assert_conforms("progressive_5", &report_5, peak_5, rmse_5);
    }
}
