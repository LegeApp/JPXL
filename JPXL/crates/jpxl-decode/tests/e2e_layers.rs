// SPDX-License-Identifier: MIT OR Apache-2.0
//! End-to-end **cropped frames** (18181-1 F.2), **image orientation**
//! (D.3.2, Table D.4) and **`kBlack` extra channels** (D.3.6) — the three
//! constructs a layered still image needs, all graded under 18181-3 §4.2.
//!
//! # What is actually new here, and what is not
//!
//! Wave 6 built F.2's compositing for frames that cover the canvas; this file
//! covers the two ways a frame stops covering it. A cropped frame occupies the
//! rectangle `(x0, y0) .. (x0 + width, y0 + height)`, which F.2 allows to hang
//! off any edge — `x0` and `y0` come from `UnpackSigned` and are legitimately
//! negative. And orientation moves the whole finished image, including its
//! extra channels, after everything else has run.
//!
//! `kBlack` needs no code of its own: 18181-3 §4.1.2's reference arrays store
//! every extra channel as itself, in `ec_info` order, so a CMYK image's black
//! separation is graded as the fourth plane it is. The rung below asserts that
//! is what the corpus reference actually holds rather than assuming it.
//!
//! # The ladder, and what each rung proves
//!
//! **Orientation (handmade fixtures 100–109).** Table D.4 has eight rows and
//! the four transposing ones swap the reported dimensions, so the ladder is
//! the whole table rather than a sample of it:
//!
//! 1. **Fixtures 100–107** — one 24×16 RGBA payload encoded eight times with
//!    nothing different but `metadata.orientation`. Each rung asserts the
//!    field really carries the row it is named for, that the displayed
//!    dimensions are the ones D.3.2 implies, and then grades against `djxl`'s
//!    own decode of the same stream. Rung 100 is the identity control.
//! 2. **The rows are pairwise distinct** — proved from the eight reference
//!    decodes, so the ladder cannot pass by eight decoders all agreeing to do
//!    nothing.
//! 3. **Fixture 108** — 600×520 anti-transposed: a 3×3 group grid, so the
//!    image is assembled from nine sub-bitstreams before the turn.
//! 4. **Fixture 109** — 64×64 `kVarDCT` transposed: the float-plane path. A
//!    decoder that turned only the integer planes passes 100–108, whose
//!    lossless modular decodes have no float planes, and fails here.
//!
//! **Cropped frames and `kBlack` (the normative corpus).** `cjxl` emits a
//! cropped *displayed* frame only for animation input, whose frames carry a
//! duration and are a separate presented image — so there is no handmade rung
//! for cropping, and the corpus streams are the evidence. Each rung asserts
//! the crop rectangles it is named for are really in the bitstream before it
//! grades a pixel:
//!
//! 5. **`spot`** — 600×400, two frames: a full-frame background and a
//!    381×145 `kBlend` layer at `(89, 114)`. Six output channels: RGB, alpha
//!    and two `kSpotColour` extra channels, the latter blended with `kAdd`.
//! 6. **`cmyk_layers`** — 512×512, four frames: a full-frame background and
//!    three cropped `kBlend` layers. Five output channels: CMY, `kBlack` and
//!    alpha. The alpha channel is extra channel **1**, not 0, so every
//!    blending rule names it explicitly.
//! 7. **`sunset_logo`** — the hard one. Two 2048×1024 frames at
//!    `(-662, -100)`: **negative** crop origins, and a frame larger than the
//!    image in both directions, so only F.2's "intersection of the frame with
//!    the image" contributes. On top of that the image is anti-transposed, so
//!    it is the only case where cropping and orientation have to be right at
//!    the same time — and where getting the order wrong (turning before
//!    compositing) produces a plausible-looking image of the correct shape.
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
//! Same rule as `e2e_alpha.rs` and `e2e_progressive.rs`: the corpus cases run
//! unconditionally in a release build and, in a debug build, only when
//! `JPXL_SLOW_TESTS` is set. They are not `#[ignore]`d — in the configuration
//! where they are affordable they must run by default.

use std::path::{Path, PathBuf};

use jpxl_bitstream::BitReader;
use jpxl_conformance::{FloatImage, OracleKind, OutputFormat, Similarity, oracle, similarity};
use jpxl_core::limits::{AllocGuard, Limits};
use jpxl_decode::frame::{
    BlendMode, FrameGeometry, FrameHeader, FrameType, read_frame_header, read_toc,
};
use jpxl_decode::headers::enums::ExtraChannelType;
use jpxl_decode::headers::{Orientation, decode_image_headers_metered};
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
/// Identical in intent to `e2e_alpha.rs`'s function of the same name: the
/// float planes are authoritative where they exist, and a lossless modular
/// decode's integers are put on the nominal `[0, 1]` scale by each plane's own
/// `(1 << bits_per_sample) - 1`, which G.4.2's last paragraph makes per-channel
/// rather than image-wide.
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
    let dir = std::env::temp_dir().join("jpxl-e2e-layers");
    std::fs::create_dir_all(&dir).ok()?;
    let out = dir.join(format!("{}.npy", fixture.file_stem()?.to_string_lossy()));
    oracle.decode(fixture, &out, OutputFormat::Npy).ok()?;
    Some(read_npy(&out))
}

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

/// Every frame header of a codestream, parsed by JPXL's own reader.
///
/// The same walk `decode` performs, reproduced here so a structural claim
/// about a fixture is checked against the bitstream rather than a comment.
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

/// The stored (pre-orientation) size and the orientation of a codestream.
fn size_and_orientation(codestream: &[u8]) -> (u32, u32, Orientation) {
    let limits = Limits::default();
    let mut guard = AllocGuard::new(&limits);
    let mut reader = BitReader::new(codestream);
    let headers = decode_image_headers_metered(&mut reader, &limits, &mut guard).expect("headers");
    (
        headers.width(),
        headers.height(),
        headers.metadata.orientation,
    )
}

/// The image header's extra-channel types, in `ec_info` order.
fn extra_channel_types(codestream: &[u8]) -> Vec<ExtraChannelType> {
    let limits = Limits::default();
    let mut guard = AllocGuard::new(&limits);
    let mut reader = BitReader::new(codestream);
    decode_image_headers_metered(&mut reader, &limits, &mut guard)
        .expect("headers")
        .metadata
        .ec_info
        .iter()
        .map(|info| info.channel_type)
        .collect()
}

// ---------------------------------------------------------------------------
// Rungs 1-2 — Table D.4, one row at a time
// ---------------------------------------------------------------------------

/// The eight fixtures of the orientation ladder: file name and the row of
/// Table D.4 it is required to signal.
const ORIENTATION_LADDER: [(&str, Orientation); 8] = [
    (
        "100_orientation_identity_rgba_24x16.jxl",
        Orientation::Identity,
    ),
    (
        "101_orientation_flip_h_rgba_24x16.jxl",
        Orientation::FlipHorizontal,
    ),
    (
        "102_orientation_rot180_rgba_24x16.jxl",
        Orientation::Rotate180,
    ),
    (
        "103_orientation_flip_v_rgba_24x16.jxl",
        Orientation::FlipVertical,
    ),
    (
        "104_orientation_transpose_rgba_24x16.jxl",
        Orientation::Transpose,
    ),
    (
        "105_orientation_rot90cw_rgba_24x16.jxl",
        Orientation::Rotate90Cw,
    ),
    (
        "106_orientation_antitranspose_rgba_24x16.jxl",
        Orientation::AntiTranspose,
    ),
    (
        "107_orientation_rot90ccw_rgba_24x16.jxl",
        Orientation::Rotate90Ccw,
    ),
];

/// Asserts one fixture signals the orientation it is named for, that our
/// decode has the dimensions D.3.2 implies, and grades it.
///
/// Returns the graded decode, or `None` when the rung skipped.
fn orientation_rung(name: &str, want: Orientation, peak: f32, rmse: f32) -> Option<FloatImage> {
    let path = handmade(name);
    assert!(path.exists(), "missing fixture {name}");
    let codestream = codestream_of(&path);

    let (stored_width, stored_height, orientation) = size_and_orientation(&codestream);
    assert_eq!(
        orientation, want,
        "{name}: this rung only proves anything if the stream really signals \
         {want:?}"
    );

    // F.2: the frame's own width and height are on the sample grid "before
    // taking metadata.orientation into account", so no frame in a correctly
    // written fixture is pre-swapped.
    for header in frame_headers(&codestream) {
        assert!(
            !header.have_crop,
            "{name}: the orientation ladder is deliberately crop-free"
        );
        assert_eq!(
            (header.width, header.height),
            (stored_width, stored_height),
            "{name}: a frame's dimensions are pre-orientation (F.2)"
        );
    }

    let image = decode_fixture(&path);
    let (want_width, want_height) = orientation.displayed_size(stored_width, stored_height);
    assert_eq!(
        (image.width, image.height),
        (want_width, want_height),
        "{name}: {orientation:?} of a {stored_width}x{stored_height} sample \
         grid displays as {want_width}x{want_height}"
    );
    for plane in &image.planes {
        assert_eq!(
            (plane.width, plane.height),
            (want_width, want_height),
            "{name}: every plane, colour and extra alike, is turned (D.3.2)"
        );
    }

    let ours = as_float_image(&image);
    let reference = reference_for(&path)?;
    let report = grade(name, &ours, &reference);
    assert_conforms(name, &report, peak, rmse);
    Some(ours)
}

/// Every row of Table D.4, against `djxl`'s decode of the same stream.
///
/// Lossless, so the tolerance is the `f32` round-trip of the integers and
/// nothing else: an exact decode scores zero here, not merely a small number.
/// A wrong row of the table is not a small error — the ramps in the fixture
/// put a wrong flip at O(1) — so this bound has enormous margin for the thing
/// it is measuring and none at all for the thing it is testing.
#[test]
fn fixtures_100_to_107_cover_table_d4() {
    for (name, want) in ORIENTATION_LADDER {
        orientation_rung(name, want, 1e-6, 1e-6);
    }
}

/// The eight rows produce eight different images.
///
/// Without this the ladder above could pass for the worst possible reason: if
/// `djxl` and JPXL both ignored `orientation`, every rung would agree with
/// every other and all eight would be green. So the *references* are compared
/// against each other — they come from the oracle, not from us — and every
/// pair is required to differ, either in shape or in samples.
#[test]
fn table_d4_rows_are_pairwise_distinct() {
    let mut decodes: Vec<(&str, FloatImage)> = Vec::new();
    for (name, _) in ORIENTATION_LADDER {
        let Some(reference) = reference_for(&handmade(name)) else {
            eprintln!("skipping: no reference for {name}");
            return;
        };
        decodes.push((name, reference));
    }
    for (i, (name_a, a)) in decodes.iter().enumerate() {
        for (name_b, b) in decodes.iter().skip(i + 1) {
            let differs = (a.width, a.height) != (b.width, b.height) || a.samples != b.samples;
            assert!(
                differs,
                "{name_a} and {name_b} decode identically, so the ladder \
                 cannot tell the two orientations apart"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Rungs 3-4 — orientation over several groups, and over the float planes
// ---------------------------------------------------------------------------

/// 600x520 anti-transposed: nine groups assembled, then turned once.
///
/// The multi-group rung AGENTS.md section 6 requires. Anti-transpose moves
/// every sample the furthest of the eight rows, so a turn applied per group,
/// or before assembly, scrambles the image rather than missing narrowly.
#[test]
fn fixture_108_orientation_over_nine_groups() {
    let name = "108_orientation_antitranspose_rgba_600x520.jxl";
    let path = handmade(name);
    assert!(path.exists(), "missing fixture {name}");

    let codestream = codestream_of(&path);
    let limits = Limits::default();
    let mut guard = AllocGuard::new(&limits);
    let header = frame_headers(&codestream)
        .into_iter()
        .find(|h| h.frame_type == FrameType::RegularFrame)
        .expect("a regular frame");
    let geometry =
        FrameGeometry::from_header(&header, header.width, header.height, &limits, &mut guard)
            .expect("geometry");
    assert_eq!(
        geometry.num_groups(),
        9,
        "{name}: this rung is only multi-group if the grid is 3x3"
    );

    orientation_rung(name, Orientation::AntiTranspose, 1e-6, 1e-6);
}

/// A `kVarDCT` frame transposed: the float planes are turned too.
///
/// The image is square, so the swapped dimensions cannot reveal the bug on
/// their own — only the sample positions can. That is what makes this rung
/// different from 104, which is the same row of Table D.4 on a modular decode
/// that has no float planes at all.
#[test]
fn fixture_109_orientation_of_the_float_planes() {
    let name = "109_orientation_transpose_vardct_rgba_64x64.jxl";
    let path = handmade(name);
    assert!(path.exists(), "missing fixture {name}");

    let image = decode_fixture(&path);
    let planes = image
        .float_planes
        .as_ref()
        .unwrap_or_else(|| panic!("{name}: a kVarDCT decode must carry float planes"));
    assert_eq!(planes.len(), 4, "{name}: RGB + alpha as floats");
    for plane in planes {
        assert_eq!(
            (plane.width, plane.height),
            (image.width, image.height),
            "{name}: the float planes are turned with the integer ones"
        );
    }

    // Filters are left at cjxl's own choice, so 18181-3 Annex A's ordinary
    // error class applies; the rung asserts a much tighter bound than that
    // because a wrong orientation is an O(1) error, not a rounding one.
    orientation_rung(name, Orientation::Transpose, 0.004, 1e-4);
}

// ---------------------------------------------------------------------------
// Rungs 5-7 — the normative corpus: cropped frames and kBlack
// ---------------------------------------------------------------------------

/// The Part 3 error class of a corpus case, read from its own `test.json`.
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

/// Whether the corpus decodes are affordable in this build.
fn slow_tests_enabled() -> bool {
    !cfg!(debug_assertions) || std::env::var_os("JPXL_SLOW_TESTS").is_some()
}

/// Decodes one corpus case and grades it against its own `test.json`.
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
            "skipping corpus {case}: a 1-Mpixel multi-frame decode is slow \
             unoptimized. Re-run with --release, or set JPXL_SLOW_TESTS=1."
        );
        return None;
    }
    let image = decode_fixture(&input);
    let report = grade(case, &as_float_image(&image), &read_npy(&reference));
    let (peak, rmse) = corpus_thresholds(&dir).unwrap_or_else(|| panic!("{case}/test.json"));
    assert_conforms(case, &report, peak, rmse);
    Some(image)
}

/// The cropped frames of a corpus case, as `(x0, y0, width, height)`.
///
/// Read with JPXL's own frame-header reader, so the rectangles asserted below
/// are the bitstream's and not an oracle's opinion of it.
fn crop_rectangles(case: &str) -> Vec<(i32, i32, u32, u32)> {
    let codestream = codestream_of(&corpus(case).join("input.jxl"));
    frame_headers(&codestream)
        .into_iter()
        .filter(|h| h.frame_type == FrameType::RegularFrame && h.have_crop)
        .map(|h| (h.x0, h.y0, h.width, h.height))
        .collect()
}

/// `spot` — a cropped `kBlend` layer over a full-frame background, with two
/// `kSpotColour` extra channels blended by `kAdd`.
///
/// The feature assertions are what make this a cropped-frame test rather than
/// a two-frame test: the second frame is required to be a strict sub-rectangle
/// of the image, so a decoder that ignored `have_crop` and blended the 381x145
/// layer at the origin would score an O(1) error over the whole overlap.
///
/// The two spot-colour channels earn their own assertion because 18181-3
/// §4.1.2 stores them as themselves — the reference array has six channels,
/// not three — so "render the spot colours into RGB" would fail the shape
/// check, and "drop them" would fail the channel count.
#[test]
fn corpus_spot_cropped_layer_and_spot_colours() {
    let dir = corpus("spot");
    if !dir.join("input.jxl").exists() {
        eprintln!("skipping corpus spot: not fetched");
        return;
    }
    let codestream = codestream_of(&dir.join("input.jxl"));
    assert_eq!(
        extra_channel_types(&codestream),
        vec![
            ExtraChannelType::KAlpha,
            ExtraChannelType::KSpotColour,
            ExtraChannelType::KSpotColour,
        ],
        "spot: alpha plus two spot-colour channels"
    );
    assert_eq!(
        crop_rectangles("spot"),
        vec![(89, 114, 381, 145)],
        "spot: exactly one cropped frame, at the offset jxlinfo calls layer \
         \"TEST\""
    );
    let headers = frame_headers(&codestream);
    let cropped = headers
        .iter()
        .find(|h| h.have_crop)
        .expect("the cropped frame");
    assert_eq!(
        cropped.blending_info.mode,
        BlendMode::Blend,
        "spot: the layer is alpha-blended, so the crop rectangle also bounds \
         where alpha is read"
    );
    assert_eq!(
        cropped.ec_blending_info.get(1).map(|i| i.mode),
        Some(BlendMode::Add),
        "spot: the spot-colour channels use kAdd, a different rule from the \
         colour channels'"
    );

    let Some(image) = corpus_rung("spot") else {
        return;
    };
    assert_eq!(image.planes.len(), 6, "RGB + alpha + two spot channels");
    assert_eq!((image.width, image.height), (600, 400));
}

/// `cmyk_layers` — three cropped `kBlend` layers over a background, in a CMYK
/// image whose black separation is a `kBlack` extra channel.
///
/// Two things at once, and both are asserted before grading. The three crops
/// are at different offsets and sizes, so a decoder with a sign error or a
/// transposed offset cannot land all three. And the alpha channel is extra
/// channel **1** — `kBlack` is 0 — so every blending rule names its alpha
/// explicitly rather than defaulting to it; a decoder that assumed
/// `alpha_channel == 0` would alpha-blend against the black separation.
#[test]
fn corpus_cmyk_layers_cropped_layers_and_kblack() {
    let dir = corpus("cmyk_layers");
    if !dir.join("input.jxl").exists() {
        eprintln!("skipping corpus cmyk_layers: not fetched");
        return;
    }
    let codestream = codestream_of(&dir.join("input.jxl"));
    assert_eq!(
        extra_channel_types(&codestream),
        vec![ExtraChannelType::KBlack, ExtraChannelType::KAlpha],
        "cmyk_layers: the black separation comes first, alpha second"
    );
    assert_eq!(
        crop_rectangles("cmyk_layers"),
        vec![(143, 166, 200, 107), (98, 311, 300, 88), (134, 13, 110, 68)],
        "cmyk_layers: three cropped layers, all different"
    );
    for header in frame_headers(&codestream).iter().filter(|h| h.have_crop) {
        assert_eq!(
            header.blending_info.alpha_channel, 1,
            "cmyk_layers: alpha is extra channel 1, and the rules say so"
        );
        assert_eq!(header.blending_info.mode, BlendMode::Blend);
    }

    let Some(image) = corpus_rung("cmyk_layers") else {
        return;
    };
    assert_eq!(image.planes.len(), 5, "CMY + black + alpha");
    assert_eq!((image.width, image.height), (512, 512));
    // 18181-3 4.1.2 stores every extra channel as itself, so the black
    // separation is graded as the plane it is — no CMYK conversion is part of
    // a codestream decoder's job, and the reference proves it is not expected.
    let black = image.planes.get(3).expect("the kBlack plane");
    let first = black.samples.first().copied().unwrap_or(0);
    assert!(
        black.samples.iter().any(|&v| v != first),
        "cmyk_layers: the kBlack plane is constant, which is what a dropped \
         extra channel looks like"
    );
}

/// `sunset_logo` — negative crop origins, frames larger than the image, and an
/// anti-transposed output.
///
/// This is the rung that pins the sign of `UnpackSigned` in F.2's `ux0`/`uy0`:
/// both frames sit at `(-662, -100)` and are 2048x1024 against a 1386x924
/// sample grid, so the frame hangs 662 columns off the left edge and 100 rows
/// off the top, and its far edges land exactly on the image's. A decoder that
/// read the offsets unsigned would place the frame at `(1324, 200)` and
/// composite almost nothing; one that clamped the rectangle to the canvas but
/// forgot to shift the *source* coordinate with it would read the top-left
/// 1386x924 window of the frame instead of the bottom-right one — an image
/// with the right shape, the right histogram, and everything in the wrong
/// place.
///
/// It also shows why `have_crop` and `full_frame` are different questions:
/// this rectangle covers the image, so F.2's `full_frame` is true and the
/// frame's `blending_info` still gates `resets_canvas` — but the frame's own
/// samples are 2048x1024 and cannot be used as the canvas directly.
///
/// It is also the only case where cropping and orientation must both be right:
/// the anti-transpose swaps the dimensions, so turning *before* compositing
/// would still produce a 924x1386 image — the correct shape, the wrong pixels.
#[test]
fn corpus_sunset_logo_negative_crop_and_orientation() {
    let dir = corpus("sunset_logo");
    if !dir.join("input.jxl").exists() {
        eprintln!("skipping corpus sunset_logo: not fetched");
        return;
    }
    let codestream = codestream_of(&dir.join("input.jxl"));
    let (stored_width, stored_height, orientation) = size_and_orientation(&codestream);
    assert_eq!((stored_width, stored_height), (1386, 924));
    assert_eq!(
        orientation,
        Orientation::AntiTranspose,
        "sunset_logo: the case is named for its orientation as much as its crop"
    );

    let rectangles = crop_rectangles("sunset_logo");
    assert_eq!(
        rectangles,
        vec![(-662, -100, 2048, 1024), (-662, -100, 2048, 1024)],
        "sunset_logo: two frames, both at a negative origin and both larger \
         than the image"
    );
    for &(x0, y0, width, height) in &rectangles {
        assert!(x0 < 0 && y0 < 0, "the offsets are UnpackSigned negatives");
        // The rectangle starts off the top-left corner and reaches exactly the
        // far edges, so `full_frame` is true even though `have_crop` is: the
        // frame covers the image, it just does not start at the origin.
        assert_eq!(
            (
                i64::from(x0) + i64::from(width),
                i64::from(y0) + i64::from(height)
            ),
            (i64::from(stored_width), i64::from(stored_height)),
            "the frame's far edges land exactly on the image's"
        );
    }

    let Some(image) = corpus_rung("sunset_logo") else {
        return;
    };
    assert_eq!(image.planes.len(), 4, "RGB + alpha");
    assert_eq!(
        (image.width, image.height),
        orientation.displayed_size(stored_width, stored_height),
        "sunset_logo: the displayed image is the sample grid transposed"
    );
    assert_eq!((image.width, image.height), (924, 1386));
}
