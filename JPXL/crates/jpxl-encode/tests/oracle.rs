//! The headline claim of slices 7.5 and 10: **other people's decoders** read
//! what this encoder writes, sample for sample.
//!
//! `tests/roundtrip.rs` proves the encoder and `jpxl-decode` agree. That is a
//! necessary check and an insufficient one — two implementations written from
//! the same reading of the same clause fail together. These tests replace one
//! side with a decoder JPXL had no hand in:
//!
//! * `djxl` (libjxl, the reference implementation), compared through its PPM
//!   output;
//! * `jxl-oxide` (an independent Rust decoder), compared through its `.npy`
//!   output — it has no PNM writer, and it **ignores the output extension**, so
//!   `--output-format` is always passed explicitly.
//!
//! # Skipping, not failing
//!
//! A checkout with no oracle installed is expected to be green: every test
//! here returns early with a printed note when its binary is missing. A
//! *present* oracle that disagrees is a hard failure.
//!
//! # Bit-exactness
//!
//! Lossless modular is in the bit-exact regime (`docs/PLAN.md`). Both
//! comparisons are sample equality. `jxl-oxide`'s `.npy` is `f32` in `[0, 1]`,
//! which for a `b`-bit image is `k / (2^b - 1)` for integer `k`; the check
//! asserts that each value is *exactly* that ratio rather than close to it.

#![allow(clippy::cast_possible_truncation, clippy::type_complexity)]

use std::path::{Path, PathBuf};

use jpxl_conformance::{Image as PnmImage, OracleKind, OutputFormat, oracle};
use jpxl_encode::{EncodeOptions, Image, encode};

/// A distinct directory per test, so parallel runs cannot collide.
fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("jpxl-encode-oracle-{tag}"));
    std::fs::create_dir_all(&dir).expect("scratch directory");
    dir
}

/// One oracle case: a name, the image, and how it is to be encoded.
struct Case {
    name: &'static str,
    image: Image,
    options: EncodeOptions,
}

fn build(
    width: u32,
    height: u32,
    channels: usize,
    bits: u32,
    f: fn(u32, u32, usize, u32) -> u32,
) -> Image {
    let max = (1u32 << bits) - 1;
    let planes: Vec<Vec<i32>> = (0..channels)
        .map(|c| {
            (0..height)
                .flat_map(|y| (0..width).map(move |x| f(x, y, c, max) as i32))
                .collect()
        })
        .collect();
    Image::new(width, height, bits, planes).expect("valid image")
}

/// The cases the oracle tests run over.
///
/// Between them they cover: a flat field, a predictor-friendly ramp, a
/// predictor-hostile pattern, the full range of both bit depths, greyscale and
/// RGB (hence the RCT), sizes that are neither square nor multiples of eight, a
/// forced small group grid, a >512-per-side image at the default group size,
/// and the container form.
fn cases() -> Vec<Case> {
    let default = EncodeOptions::default();
    let tiny_groups = EncodeOptions {
        container: false,
        group_size_shift: Some(0),
        ..EncodeOptions::default()
    };
    let boxed = EncodeOptions {
        container: true,
        group_size_shift: None,
        ..EncodeOptions::default()
    };

    fn flat(_: u32, _: u32, c: usize, max: u32) -> u32 {
        max / (2 + c as u32)
    }
    fn ramp(x: u32, y: u32, c: usize, max: u32) -> u32 {
        (x * 7 + y * 13 + c as u32 * 29) % (max + 1)
    }
    fn full_range(x: u32, y: u32, c: usize, max: u32) -> u32 {
        // Walks the whole value range so a 16-bit case really reaches 65535.
        (x.wrapping_add(y.wrapping_mul(101))
            .wrapping_add(c as u32 * 7)
            .wrapping_mul(2_654_435_761))
            % (max + 1)
    }
    fn noise(x: u32, y: u32, c: usize, max: u32) -> u32 {
        let v = x
            .wrapping_mul(2_654_435_761)
            .wrapping_add(y.wrapping_mul(40_503))
            .wrapping_add((c as u32).wrapping_mul(2_246_822_519));
        (v >> 13) % (max + 1)
    }
    fn black(_: u32, _: u32, _: usize, _: u32) -> u32 {
        0
    }
    fn white(_: u32, _: u32, _: usize, max: u32) -> u32 {
        max
    }

    vec![
        Case {
            name: "grey8-flat",
            image: build(32, 32, 1, 8, flat),
            options: default,
        },
        Case {
            name: "grey8-ramp-13x7",
            image: build(13, 7, 1, 8, ramp),
            options: default,
        },
        Case {
            name: "grey8-noise-101x97",
            image: build(101, 97, 1, 8, noise),
            options: default,
        },
        Case {
            name: "grey16-full-range-64x48",
            image: build(64, 48, 1, 16, full_range),
            options: default,
        },
        Case {
            name: "grey16-black",
            image: build(13, 7, 1, 16, black),
            options: default,
        },
        Case {
            name: "grey16-white",
            image: build(13, 7, 1, 16, white),
            options: default,
        },
        Case {
            name: "rgb8-ramp-64x64",
            image: build(64, 64, 3, 8, ramp),
            options: default,
        },
        Case {
            name: "rgb8-noise-101x97",
            image: build(101, 97, 3, 8, noise),
            options: default,
        },
        Case {
            name: "rgb16-full-range-101x97",
            image: build(101, 97, 3, 16, full_range),
            options: default,
        },
        Case {
            name: "grey8-multigroup-forced-300x200",
            image: build(300, 200, 1, 8, ramp),
            options: tiny_groups,
        },
        Case {
            name: "rgb16-multigroup-forced-101x97",
            image: build(101, 97, 3, 16, noise),
            options: tiny_groups,
        },
        Case {
            name: "grey8-multigroup-600x520",
            image: build(600, 520, 1, 8, ramp),
            options: default,
        },
        Case {
            name: "rgb8-multigroup-600x520",
            image: build(600, 520, 3, 8, noise),
            options: default,
        },
        Case {
            name: "grey16-container",
            image: build(37, 41, 1, 16, full_range),
            options: boxed,
        },
        Case {
            name: "rgb8-container",
            image: build(37, 41, 3, 8, ramp),
            options: boxed,
        },
    ]
}

/// The image's samples, interleaved the way a Netpbm file stores them.
fn interleaved(image: &Image) -> Vec<u16> {
    let pixels = image.width() as usize * image.height() as usize;
    let mut out = Vec::with_capacity(pixels * image.num_channels());
    for i in 0..pixels {
        for plane in image.planes() {
            out.push(plane.get(i).copied().unwrap_or(0) as u16);
        }
    }
    out
}

#[test]
fn djxl_decodes_our_output_to_the_source_samples() {
    let Some(oracle) = oracle::find(OracleKind::Djxl) else {
        println!("skipping: djxl is not installed");
        return;
    };
    let dir = scratch("djxl");

    for case in cases() {
        let name = case.name;
        let jxl = dir.join(format!("{name}.jxl"));
        let ppm = dir.join(format!("{name}.ppm"));
        std::fs::write(&jxl, encode(&case.image, &case.options).expect("encodes")).expect("write");

        match oracle.decode(&jxl, &ppm, OutputFormat::Ppm) {
            Ok(()) => {}
            Err(err) if err.is_unavailable() => {
                println!("skipping: {err}");
                return;
            }
            Err(err) => panic!("{name}: djxl refused our codestream: {err}"),
        }

        let bytes = std::fs::read(&ppm).unwrap_or_else(|e| panic!("{name}: {e}"));
        let decoded = PnmImage::from_ppm(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            (decoded.w, decoded.h),
            (case.image.width(), case.image.height()),
            "{name}: dimensions"
        );
        let max = (1u32 << case.image.bits_per_sample()) - 1;
        assert_eq!(u32::from(decoded.max_value), max, "{name}: bit depth");
        // djxl always writes P6, so a greyscale image comes back with its
        // single value replicated across the three channels.
        assert_eq!(decoded.channels, 3, "{name}: djxl always writes P6");

        let want = interleaved(&case.image);
        let repeat = 3 / case.image.num_channels();
        assert_eq!(decoded.len(), want.len() * repeat, "{name}: sample count");
        for (i, &value) in want.iter().enumerate() {
            for r in 0..repeat {
                let got = decoded.samples.get(i * repeat + r).copied();
                assert_eq!(got, Some(value), "{name}: sample {i} copy {r}");
            }
        }
    }
}

/// Phase 4A's correctness gate (jpegxl-rs.work.arch-phase4a-weighted-
/// predictor-scoped): a stream with a forced Weighted leaf, and a forced
/// MIXED tree (Weighted + Gradient), decoded by djxl -- an independent
/// decoder that catches a bug shared between the encoder and jpxl-decode's
/// H.5 arithmetic (they now share `jpxl_core::modular_weighted`, so a
/// `jpxl-decode` round trip alone couldn't distinguish "correct" from
/// "consistently wrong the same way"). Forced rather than search-selected so
/// the gate has teeth regardless of whether these particular patterns would
/// win the real predictor sweep.
#[test]
fn djxl_decodes_a_forced_weighted_stream_to_the_source_samples() {
    let Some(oracle) = oracle::find(OracleKind::Djxl) else {
        println!("skipping: djxl is not installed");
        return;
    };
    let dir = scratch("djxl-weighted");

    let width = 64u32;
    let height = 64u32;
    // A pattern with local structure (not flat, not pure noise) -- the kind
    // Weighted's error-correction is meant for -- but built the image with
    // one channel per case so this is a real (bits-per-sample = 8, RGB)
    // plane, matching how the encoder actually carries samples.
    let plane = |seed: u32| -> Vec<i32> {
        (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    let base = (x.wrapping_add(y.wrapping_mul(3)).wrapping_add(seed)) % 200;
                    base as i32
                })
            })
            .collect()
    };
    let planes = vec![plane(0), plane(50), plane(100)];
    let image = Image::new(width, height, 8, planes.clone()).expect("image");

    let cases: Vec<(&str, jpxl_encode::modular::MaTree)> = vec![
        (
            "single_leaf_weighted",
            jpxl_encode::modular::MaTree::single_leaf(jpxl_encode::modular::Predictor::Weighted),
        ),
        (
            "mixed_weighted_gradient",
            jpxl_encode::modular::MaTree::binary_split_preds(
                6, // property 6 (Table H.4): a static row available at every depth.
                50,
                jpxl_encode::modular::Predictor::Weighted,
                jpxl_encode::modular::Predictor::Gradient,
            ),
        ),
    ];

    for (name, tree) in cases {
        let plan = jpxl_encode::lossless::validate(jpxl_encode::lossless::LosslessPlan {
            group_size_shift: 2,
            rct: false,
            palette: None,
            squeeze: false,
            tree,
        })
        .expect("validate");
        let bytes = jpxl_encode::encode_codestream_with_plan(&image, image.planes(), &plan)
            .expect("encode");

        let jxl = dir.join(format!("{name}.jxl"));
        let ppm = dir.join(format!("{name}.ppm"));
        std::fs::write(&jxl, &bytes).expect("write");

        match oracle.decode(&jxl, &ppm, OutputFormat::Ppm) {
            Ok(()) => {}
            Err(err) if err.is_unavailable() => {
                println!("skipping: {err}");
                return;
            }
            Err(err) => panic!("{name}: djxl refused our Weighted codestream: {err}"),
        }

        let ppm_bytes = std::fs::read(&ppm).unwrap_or_else(|e| panic!("{name}: {e}"));
        let decoded = PnmImage::from_ppm(&ppm_bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            (decoded.w, decoded.h),
            (width, height),
            "{name}: dimensions"
        );

        let want = interleaved(&image);
        assert_eq!(decoded.len(), want.len(), "{name}: sample count");
        for (i, &value) in want.iter().enumerate() {
            assert_eq!(
                decoded.samples.get(i).copied(),
                Some(value),
                "{name}: sample {i}"
            );
        }
    }
}

#[test]
fn jxl_oxide_decodes_our_output_to_the_source_samples() {
    let Some(oracle) = oracle::find(OracleKind::JxlOxide) else {
        println!("skipping: jxl-oxide is not installed");
        return;
    };
    let dir = scratch("jxl-oxide");

    for case in cases() {
        let name = case.name;
        let jxl = dir.join(format!("{name}.jxl"));
        let npy = dir.join(format!("{name}.npy"));
        std::fs::write(&jxl, encode(&case.image, &case.options).expect("encodes")).expect("write");

        match oracle.decode(&jxl, &npy, OutputFormat::Npy) {
            Ok(()) => {}
            Err(err) if err.is_unavailable() => {
                println!("skipping: {err}");
                return;
            }
            Err(err) => panic!("{name}: jxl-oxide refused our codestream: {err}"),
        }

        let bytes = std::fs::read(&npy).unwrap_or_else(|e| panic!("{name}: {e}"));
        let values = read_npy_f32(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        let want = interleaved(&case.image);
        assert_eq!(
            values.len(),
            want.len(),
            "{name}: sample count (one frame expected)"
        );
        let max = ((1u32 << case.image.bits_per_sample()) - 1) as f32;
        for (i, (&got, &value)) in values.iter().zip(want.iter()).enumerate() {
            // The conformance .npy convention is normalised f32. For integer
            // samples the only exact representation is k/max, so equality is
            // the right test and a tolerance would hide a real error.
            assert_eq!(got, f32::from(value) / max, "{name}: sample {i} ({value})");
        }
    }
}

/// Reads the little-endian `f32` payload of a NumPy `.npy` file.
///
/// Only what the conformance convention emits is supported: version 1.0, a
/// `<f4` C-order array. The shape is not interpreted — the caller knows how
/// many samples it expects — but the dtype is checked, because reading `f8`
/// bytes as `f4` would silently produce garbage that still compares.
fn read_npy_f32(bytes: &[u8]) -> Result<Vec<f32>, String> {
    let rest = bytes
        .strip_prefix(b"\x93NUMPY")
        .ok_or_else(|| "not a .npy file".to_owned())?;
    let major = rest.first().copied().ok_or("truncated .npy header")?;
    if major != 1 {
        return Err(format!("unsupported .npy version {major}"));
    }
    let len_bytes = rest.get(2..4).ok_or("truncated .npy header")?;
    let header_len = usize::from(u16::from_le_bytes([
        len_bytes.first().copied().unwrap_or(0),
        len_bytes.get(1).copied().unwrap_or(0),
    ]));
    let header = rest.get(4..4 + header_len).ok_or("truncated .npy header")?;
    let header = core::str::from_utf8(header).map_err(|e| e.to_string())?;
    if !header.contains("'<f4'") {
        return Err(format!("expected a '<f4' array, header is {header}"));
    }
    if header.contains("'fortran_order': True") {
        return Err("expected a C-order array".to_owned());
    }

    let payload = rest.get(4 + header_len..).ok_or("truncated .npy payload")?;
    let mut values = Vec::with_capacity(payload.len() / 4);
    for chunk in payload.chunks_exact(4) {
        let word = u32::from_le_bytes([
            chunk.first().copied().unwrap_or(0),
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
            chunk.get(3).copied().unwrap_or(0),
        ]);
        values.push(f32::from_bits(word));
    }
    Ok(values)
}

/// Every artefact these tests write lands under one predictable root, so a
/// failing run can be inspected and a passing one leaves nothing surprising.
#[test]
fn scratch_directories_are_under_the_temp_dir() {
    let dir = scratch("probe");
    assert!(dir.starts_with(std::env::temp_dir()));
    assert!(Path::new(&dir).is_dir());
}
