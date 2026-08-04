// SPDX-License-Identifier: MIT OR Apache-2.0
//! End-to-end K.2 upsampling: `frame_header.upsampling` for the colour
//! channels and `frame_header.ec_upsampling` for the extra channels, graded
//! under 18181-3 §4.2.
//!
//! # The two sizes, and why they are the whole subject
//!
//! F.1 divides a frame's dimensions by `upsampling`, so an upsampled frame
//! stores fewer samples than the image has. Everything up to and including
//! Annex J lives on that smaller grid — groups, LF groups, varblocks, the EPF
//! sigma field, and every modular channel. K.1 then upsamples the colour
//! channels by `upsampling` and the extra channels by
//! `ec_upsampling[n] << dim_shift[n]` (L.4's factor), and Annex K's features
//! and Annex L's colour transforms run at the larger size.
//!
//! Two failures are therefore invisible to a fixture that only checks total
//! error: gridding groups on the output size instead of the frame size, and
//! running K.2 per group instead of once over the frame. Rung 4 exists for
//! both.
//!
//! # The ladder, and what each rung proves
//!
//! Every rung asserts the construct it is named for is really present in the
//! stream *before* it grades pixels. A decode that quietly ignored
//! `upsampling` would return an image of the wrong size and fail on shape, but
//! a decode that upsampled the alpha channel by the *colour* factor would
//! return the right shape and plausible pixels, so the ec factors are asserted
//! from the parsed header too.
//!
//! 1. **Factor 2, smallest expressible** — fixture 90, a 16x16 RGBA image
//!    stored as an 8x8 frame. One varblock, one group, one section: nothing
//!    but the filter itself is left to blame.
//! 2. **Factor 4** — fixture 91, 64x64 from a 16x16 frame. The conformance
//!    corpus's own shape at one hundredth the pixel count.
//! 3. **Factor 8** — fixture 92, 64x64 from an 8x8 frame. The largest single
//!    K.2 step and its 210-value weight table; the 5x5 window is wider than
//!    the frame in both directions, so every output sample reads mirrored
//!    (5.2) neighbours.
//! 4. **Multi-group** — fixture 93, 1024x768 from a 512x384 frame, a 2x2 group
//!    grid at `group_dim` 256. F.2: group splitting proceeds "after
//!    subsampling by a factor of upsampling", so the groups tile the *frame*.
//!    Running K.2 per group instead of once over the assembled frame would
//!    mirror at the internal edges and leave a seam at x = 512 and y = 384 —
//!    the same failure family as the I.5.2 LF-smoothing seam.
//! 5. **Extra-channel upsampling alone** — fixture 94, `upsampling = 1` with
//!    `ec_upsampling = 4`. The discriminator for
//!    `decode::EC_DIMS_INCLUDE_EC_UPSAMPLING`: the alpha channel is stored 16x16
//!    inside a 64x64 frame, and G.1.3's literal sizing would read it at 64x64
//!    and desynchronise the modular stream.
//! 6. **Both, unequal** — fixture 95, `upsampling = 2` with
//!    `ec_upsampling = 8`. The colour channels take factor 2 and the alpha
//!    channel factor 8, so the alpha channel is subsampled by 4 *relative to
//!    the frame grid*. The only configuration where the double shift can go
//!    wrong in either direction.
//! 7. **The normative corpus** — `upsampling` and `upsampling_5`, an 800x600
//!    image stored as a 200x150 kVarDCT frame with `upsampling = 4` and
//!    `ec_upsampling = [4]`, graded at its own `test.json` thresholds.
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
//! Same rule as `e2e_alpha.rs` and `e2e_progressive.rs`: the corpus rungs run
//! unconditionally in a release build and, in a debug build, only when
//! `JPXL_SLOW_TESTS` is set. They are not `#[ignore]`d.

use std::path::{Path, PathBuf};

use jpxl_bitstream::BitReader;
use jpxl_conformance::{FloatImage, OracleKind, OutputFormat, Similarity, oracle, similarity};
use jpxl_core::limits::{AllocGuard, Limits};
use jpxl_decode::frame::{Encoding, FrameGeometry, FrameHeader, read_frame_header};
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
/// Colour channels first, then the extra channels in `ec_info` order — the
/// interleaving `djxl --output_format npy` writes. Every fixture here is
/// `kVarDCT`, so the float planes are always present and always authoritative.
fn as_float_image(image: &DecodedImage) -> FloatImage {
    let pixels = image.width as usize * image.height as usize;
    let planes = image
        .float_planes
        .as_ref()
        .expect("a kVarDCT decode has float planes");
    let mut samples = Vec::with_capacity(pixels * planes.len());
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
        channels: u32::try_from(planes.len()).expect("a sane channel count"),
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
    let dir = std::env::temp_dir().join("jpxl-e2e-upsampling");
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

/// What a stream signals, read by JPXL's own parsers rather than by an oracle:
/// the image size, the first frame's header, and the geometry it implies.
struct Signals {
    image: (u32, u32),
    header: FrameHeader,
    geometry: FrameGeometry,
    dim_shift: Vec<u32>,
}

fn signals(codestream: &[u8]) -> Signals {
    let limits = Limits::default();
    let mut guard = AllocGuard::new(&limits);
    let mut reader = BitReader::new(codestream);
    let headers = decode_image_headers_metered(&mut reader, &limits, &mut guard).expect("headers");
    if headers.metadata.colour_encoding.want_icc {
        jpxl_decode::icc::read_icc_profile(&mut reader, &mut guard).expect("ICC profile");
    }
    reader.zero_pad_to_byte().expect("F.1 alignment");
    let cursor = usize::try_from(reader.total_bits_read() / 8).expect("in range");
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
    Signals {
        image: (headers.width(), headers.height()),
        header,
        geometry,
        dim_shift: headers
            .metadata
            .ec_info
            .iter()
            .map(|info| info.dim_shift)
            .collect(),
    }
}

/// What a rung claims about the stream it grades.
struct Rung {
    name: &'static str,
    /// `frame_header.upsampling`.
    upsampling: u32,
    /// `frame_header.ec_upsampling`, one per extra channel.
    ec_upsampling: Vec<u32>,
    /// The frame's stored size, before K.2.
    frame: (u32, u32),
    /// The image size, which is also the frame's size after K.2.
    image: (u32, u32),
    /// `num_groups` over the *frame* grid.
    groups: u64,
}

/// Asserts a fixture really signals the upsampling the rung is named for, that
/// the decode came out at the image size rather than the frame size, that the
/// alpha plane survived, and only then grades it.
///
/// The alpha assertions are what make the extra-channel rungs mean anything.
/// An alpha channel upsampled by the wrong factor, or dropped and left at
/// zero, is still a perfectly plausible-looking alpha channel; a decode that
/// silently returned the *frame*-sized alpha would be caught only by the plane
/// dimensions, which is why those are checked separately from the image's.
fn rung(spec: &Rung) {
    let path = handmade(spec.name);
    assert!(path.exists(), "missing fixture {}", spec.name);
    let codestream = codestream_of(&path);
    let s = signals(&codestream);

    assert_eq!(
        s.header.encoding,
        Encoding::VarDct,
        "{}: the ladder is kVarDCT throughout",
        spec.name
    );
    assert_eq!(
        s.header.upsampling, spec.upsampling,
        "{}: this rung only proves anything if the stream really signals \
         upsampling {}",
        spec.name, spec.upsampling
    );
    assert_eq!(
        s.header.ec_upsampling, spec.ec_upsampling,
        "{}: ec_upsampling",
        spec.name
    );
    assert!(
        s.dim_shift.iter().all(|&d| d == 0),
        "{}: the ladder keeps dim_shift at 0 so ec_upsampling is the only \
         extra-channel factor in play",
        spec.name
    );
    assert_eq!(s.image, spec.image, "{}: image size", spec.name);
    assert_eq!(
        (s.geometry.width(), s.geometry.height()),
        spec.frame,
        "{}: F.1's frame size is the image size divided by upsampling",
        spec.name
    );
    assert_eq!(
        (s.geometry.upsampled_width(), s.geometry.upsampled_height()),
        spec.image,
        "{}: K.2's output size",
        spec.name
    );
    assert_eq!(
        s.geometry.num_groups(),
        spec.groups,
        "{}: groups tile the FRAME (F.2), not the upsampled output",
        spec.name
    );

    let image = decode_fixture(&path);
    assert_eq!(
        (image.width, image.height),
        spec.image,
        "{}: the decode is delivered at the image size, not the frame size",
        spec.name
    );
    let extra = spec.ec_upsampling.len();
    assert_eq!(
        image.planes.len(),
        image.num_colour_channels + extra,
        "{}: the decode must return the extra channels as planes",
        spec.name
    );
    for plane in &image.planes {
        assert_eq!(
            (plane.width, plane.height),
            spec.image,
            "{}: every plane, extra channels included, is delivered upsampled",
            spec.name
        );
    }
    let alpha = image
        .planes
        .get(image.num_colour_channels)
        .expect("an alpha plane");
    let first = alpha.get(0, 0);
    assert!(
        (0..spec.image.0).any(|x| (0..spec.image.1).any(|y| alpha.get(x, y) != first)),
        "{}: the alpha plane is constant — it was dropped, not decoded",
        spec.name
    );

    let Some(reference) = reference_for(&path) else {
        eprintln!(
            "skipping {}: no committed .npy and no djxl (see tools/setup-oracles.sh)",
            spec.name
        );
        return;
    };
    let report = grade(spec.name, &as_float_image(&image), &reference);
    assert_conforms(spec.name, &report, HANDMADE_PEAK, HANDMADE_RMSE);
}

/// The bounds every handmade rung is held to.
///
/// 18181-3 Annex A's class for a filtered stream is peak 0.02 / RMSE 1e-3.
/// These are two to three orders of magnitude tighter, because by the time K.2
/// runs the samples are already decoded: the only error it can legitimately
/// add is `f32` rounding. Anything looser would let a wrong weight table or a
/// half-sample phase shift through. Measured margin at the time of writing is
/// peak 5.4e-5 on the worst rung.
const HANDMADE_PEAK: f32 = 1e-4;
/// See [`HANDMADE_PEAK`].
const HANDMADE_RMSE: f32 = 1e-5;

// ---------------------------------------------------------------------------
// Rungs 1-3: the three K.2 factors
// ---------------------------------------------------------------------------

#[test]
fn factor_two_upsampling() {
    rung(&Rung {
        name: "90_upsampling_rgba_16x16_up2.jxl",
        upsampling: 2,
        ec_upsampling: vec![2],
        frame: (8, 8),
        image: (16, 16),
        groups: 1,
    });
}

#[test]
fn factor_four_upsampling() {
    rung(&Rung {
        name: "91_upsampling_rgba_64x64_up4.jxl",
        upsampling: 4,
        ec_upsampling: vec![4],
        frame: (16, 16),
        image: (64, 64),
        groups: 1,
    });
}

#[test]
fn factor_eight_upsampling() {
    rung(&Rung {
        name: "92_upsampling_rgba_64x64_up8.jxl",
        upsampling: 8,
        ec_upsampling: vec![8],
        frame: (8, 8),
        image: (64, 64),
        groups: 1,
    });
}

// ---------------------------------------------------------------------------
// Rung 4: multi-group
// ---------------------------------------------------------------------------

/// A 512x384 frame is a 2x2 group grid; the 1024x768 output is not.
///
/// This is the rung that would catch K.2 being run per group (a seam at
/// x = 512 and y = 384, where the filter would mirror instead of reading
/// across) and the group grid being derived from the output size (8 groups
/// instead of 4, and a TOC read at the wrong length).
#[test]
fn multi_group_upsampling() {
    rung(&Rung {
        name: "93_upsampling_rgba_1024x768_up2.jxl",
        upsampling: 2,
        ec_upsampling: vec![2],
        frame: (512, 384),
        image: (1024, 768),
        groups: 4,
    });
}

// ---------------------------------------------------------------------------
// Rungs 5-6: the extra-channel factor, alone and composed
// ---------------------------------------------------------------------------

/// `upsampling = 1`, `ec_upsampling = 4`: the flip-point discriminator.
///
/// See `decode::EC_DIMS_INCLUDE_EC_UPSAMPLING`. Under G.1.3's literal sizing the
/// alpha channel would be read at the frame's 64x64 instead of its own 16x16,
/// and the modular sub-bitstream runs off its end — so flipping that constant
/// fails this rung with a decode error, not with a pixel difference. Verified
/// by mutation: `OutOfBounds { bit_pos: 13579, requested_bits: 16 }`.
#[test]
fn extra_channel_upsampling_alone() {
    rung(&Rung {
        name: "94_ecupsampling_rgba_64x64_up1ec4.jxl",
        upsampling: 1,
        ec_upsampling: vec![4],
        frame: (64, 64),
        image: (64, 64),
        groups: 1,
    });
}

/// `upsampling = 2`, `ec_upsampling = 8`: the colour and extra factors
/// composed and unequal.
#[test]
fn colour_and_extra_channel_factors_compose() {
    rung(&Rung {
        name: "95_upsampling_rgba_64x64_up2ec8.jxl",
        upsampling: 2,
        ec_upsampling: vec![8],
        frame: (32, 32),
        image: (64, 64),
        groups: 1,
    });
}

// ---------------------------------------------------------------------------
// Rung 7: the normative corpus
// ---------------------------------------------------------------------------

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
fn corpus_rung(case: &str) {
    let dir = corpus(case);
    let input = dir.join("input.jxl");
    let reference = dir.join("reference_image.npy");
    if !input.exists() || !reference.exists() {
        eprintln!("skipping corpus {case}: not fetched (see tools/fetch-conformance.sh)");
        return;
    }
    if !slow_tests_enabled() {
        eprintln!(
            "skipping corpus {case}: a 0.5-Mpixel decode is slow unoptimized. \
             Re-run with --release, or set JPXL_SLOW_TESTS=1."
        );
        return;
    }

    // The construct, before the pixels: an 800x600 image carried by a 200x150
    // frame, with the alpha channel on the same factor.
    let s = signals(&codestream_of(&input));
    assert_eq!(s.image, (800, 600), "{case}: image size");
    assert_eq!(s.header.upsampling, 4, "{case}: upsampling");
    assert_eq!(s.header.ec_upsampling, vec![4], "{case}: ec_upsampling");
    assert_eq!(
        (s.geometry.width(), s.geometry.height()),
        (200, 150),
        "{case}: F.1 frame size"
    );

    let image = decode_fixture(&input);
    assert_eq!(
        (image.width, image.height),
        (800, 600),
        "{case}: output size"
    );
    let report = grade(case, &as_float_image(&image), &read_npy(&reference));
    let (peak, rmse) = corpus_thresholds(&dir).unwrap_or_else(|| panic!("{case}/test.json"));
    assert_conforms(case, &report, peak, rmse);
}

#[test]
fn corpus_upsampling() {
    corpus_rung("upsampling");
}

#[test]
fn corpus_upsampling_5() {
    corpus_rung("upsampling_5");
}
