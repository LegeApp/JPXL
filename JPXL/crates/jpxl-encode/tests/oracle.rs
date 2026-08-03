//! The headline claim of slice 7.5: **other people's decoders** read what this
//! encoder writes, sample for sample.
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
//! which for an 8-bit image is `k / 255` for integer `k`; the check asserts
//! that each value is *exactly* that ratio rather than close to it.

#![allow(clippy::cast_possible_truncation, clippy::type_complexity)]

use std::path::{Path, PathBuf};

use jpxl_conformance::{Image, OracleKind, OutputFormat, oracle};
use jpxl_encode::{GreyImage, encode_grey8};

/// A distinct directory per test, so parallel runs cannot collide.
fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("jpxl-encode-oracle-{tag}"));
    std::fs::create_dir_all(&dir).expect("scratch directory");
    dir
}

/// The images the oracle tests run over. Small, deterministic, and between
/// them covering a flat field, a predictor-friendly ramp, a predictor-hostile
/// pattern, and both a non-multiple-of-8 and a multi-hundred-pixel size.
fn cases() -> Vec<(&'static str, GreyImage)> {
    let mut out = Vec::new();
    let specs: [(&'static str, u32, u32, fn(u32, u32) -> u8); 5] = [
        ("flat", 32, 32, |_, _| 200),
        ("ramp", 13, 7, |x, y| ((x * 7 + y * 13) % 256) as u8),
        (
            "checker",
            64,
            64,
            |x, y| {
                if (x + y) % 2 == 0 { 0 } else { 255 }
            },
        ),
        ("noise", 48, 33, |x, y| {
            let v = x
                .wrapping_mul(2_654_435_761)
                .wrapping_add(y.wrapping_mul(40_503));
            ((v >> 13) & 0xFF) as u8
        }),
        ("large", 300, 200, |x, y| ((x + y) / 2 % 256) as u8),
    ];
    for (name, w, h, f) in specs {
        let samples: Vec<u8> = (0..h).flat_map(|y| (0..w).map(move |x| f(x, y))).collect();
        out.push((name, GreyImage::new(w, h, samples).expect("valid image")));
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

    for (name, image) in cases() {
        let jxl = dir.join(format!("{name}.jxl"));
        let ppm = dir.join(format!("{name}.ppm"));
        std::fs::write(&jxl, encode_grey8(&image).expect("encodes")).expect("write");

        match oracle.decode(&jxl, &ppm, OutputFormat::Ppm) {
            Ok(()) => {}
            Err(err) if err.is_unavailable() => {
                println!("skipping: {err}");
                return;
            }
            Err(err) => panic!("{name}: djxl refused our codestream: {err}"),
        }

        let bytes = std::fs::read(&ppm).unwrap_or_else(|e| panic!("{name}: {e}"));
        let decoded = Image::from_ppm(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            (decoded.w, decoded.h),
            (image.width(), image.height()),
            "{name}: dimensions"
        );
        assert_eq!(decoded.max_value, 255, "{name}: bit depth");
        // djxl writes P6, so a greyscale image comes back with its single
        // value replicated across the three channels.
        assert_eq!(decoded.channels, 3, "{name}: djxl always writes P6");
        assert_eq!(
            decoded.len(),
            image.samples().len() * 3,
            "{name}: sample count"
        );
        for (i, &want) in image.samples().iter().enumerate() {
            for c in 0..3 {
                let got = decoded.samples.get(i * 3 + c).copied();
                assert_eq!(got, Some(u16::from(want)), "{name}: sample {i} channel {c}");
            }
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

    for (name, image) in cases() {
        let jxl = dir.join(format!("{name}.jxl"));
        let npy = dir.join(format!("{name}.npy"));
        std::fs::write(&jxl, encode_grey8(&image).expect("encodes")).expect("write");

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
        assert_eq!(
            values.len(),
            image.samples().len(),
            "{name}: sample count (one channel, one frame expected)"
        );
        for (i, (&got, &want)) in values.iter().zip(image.samples().iter()).enumerate() {
            // The conformance .npy convention is normalised f32. For 8-bit
            // samples the only exact representation is k/255, so equality is
            // the right test and a tolerance would hide a real error.
            let expected = f32::from(want) / 255.0;
            assert_eq!(got, expected, "{name}: sample {i} ({want})");
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
