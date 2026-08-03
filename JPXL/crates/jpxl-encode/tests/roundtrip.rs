//! End-to-end: encode, then decode with `jpxl-decode` and assert the samples
//! come back exactly (`docs/PLAN.md` slice 7.5, stage 1).
//!
//! Stage 1 alone proves nothing about interoperability — a paired encoder and
//! decoder bug cancels — which is what `oracle.rs` is for. What stage 1 *does*
//! prove is that the bit layout is internally consistent, and it localises a
//! failure to one side long before an oracle is involved.
//!
//! # Bit-exactness
//!
//! `docs/PLAN.md` puts modular lossless in the bit-exact regime. Every
//! assertion here is sample equality; there is no tolerance anywhere.

#![allow(clippy::cast_possible_truncation)]

use jpxl_core::limits::Limits;
use jpxl_encode::{GreyImage, encode_grey8};

/// The images every round-trip test runs over.
///
/// The sizes are chosen for the edge cases they expose: 1x1 (every neighbour
/// substitution at once), 13x7 (neither dimension a multiple of 8), 129x1 and
/// 1x129 (the `group_size_shift` step), 256x256 and 300x200 (the sizes the
/// decoder's own fixtures use), 1024x1 (the largest single group).
fn cases() -> Vec<(&'static str, GreyImage)> {
    let mut out = Vec::new();
    for &(w, h) in &[
        (1u32, 1u32),
        (2, 3),
        (8, 8),
        (13, 7),
        (129, 1),
        (1, 129),
        (64, 64),
        (256, 256),
        (300, 200),
        (1024, 1),
    ] {
        for (name, f) in patterns() {
            let samples: Vec<u8> = (0..h).flat_map(|y| (0..w).map(move |x| f(x, y))).collect();
            let image = GreyImage::new(w, h, samples).expect("valid image");
            out.push((name, image));
        }
    }
    out
}

/// A deterministic `(x, y) -> sample` formula.
type Pattern = fn(u32, u32) -> u8;

/// Sample formulas, chosen so the residual distribution differs sharply
/// between them: a constant image codes every residual as 0, a gradient codes
/// almost all of them as 0 with a nonzero first column, noise defeats the
/// predictor entirely, and the extremes exercise the widest tokens.
fn patterns() -> Vec<(&'static str, Pattern)> {
    vec![
        ("constant", |_, _| 127),
        ("black", |_, _| 0),
        ("white", |_, _| 255),
        ("ramp", |x, y| ((x + y) % 256) as u8),
        ("checker", |x, y| if (x + y) % 2 == 0 { 0 } else { 255 }),
        ("noise", |x, y| {
            // A cheap deterministic hash; nothing about it is spec-related,
            // it just has to be reproducible and predictor-hostile.
            let v = x
                .wrapping_mul(2_654_435_761)
                .wrapping_add(y.wrapping_mul(40_503));
            ((v >> 13) & 0xFF) as u8
        }),
    ]
}

/// Encodes, decodes and asserts sample equality; returns the encoded bytes.
fn round_trip(name: &str, image: &GreyImage) -> Vec<u8> {
    let bytes = encode_grey8(image)
        .unwrap_or_else(|e| panic!("{name} {}x{}: encode: {e}", image.width(), image.height()));

    let decoded = jpxl_decode::decode(&bytes, &Limits::default())
        .unwrap_or_else(|e| panic!("{name} {}x{}: decode: {e}", image.width(), image.height()));

    assert_eq!(
        (decoded.width, decoded.height),
        (image.width(), image.height()),
        "{name}: dimensions"
    );
    assert_eq!(decoded.num_colour_channels, 1, "{name}: greyscale");
    assert_eq!(decoded.colour_bits_per_sample(), 8, "{name}: bit depth");

    let got = decoded.interleaved_colour();
    let want: Vec<u16> = image.samples().iter().map(|&s| u16::from(s)).collect();
    assert_eq!(
        got.len(),
        want.len(),
        "{name} {}x{}: sample count",
        image.width(),
        image.height()
    );
    if got != want {
        let first = got
            .iter()
            .zip(want.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        let width = image.width() as usize;
        panic!(
            "{name} {}x{}: first mismatch at index {first} = ({}, {}): decoded {:?}, expected {:?}",
            image.width(),
            image.height(),
            first % width,
            first / width,
            got.get(first),
            want.get(first)
        );
    }
    bytes
}

#[test]
fn every_case_round_trips_bit_exactly() {
    for (name, image) in cases() {
        round_trip(name, &image);
    }
}

#[test]
fn a_full_range_image_round_trips() {
    // Every 8-bit value appears, so the residual reaches its extremes in both
    // directions and the widest token this configuration can emit is used.
    let samples: Vec<u8> = (0..=255u8).collect();
    let image = GreyImage::new(16, 16, samples).expect("valid image");
    round_trip("full-range", &image);
}

#[test]
fn a_constant_image_is_small() {
    // Not a compression claim, a sanity check: with the gradient predictor
    // every residual of a constant image is zero, so each sample costs the
    // four bits of token 0 and nothing else.
    let image = GreyImage::new(64, 64, vec![42; 64 * 64]).expect("valid image");
    let bytes = round_trip("constant", &image);
    assert!(
        bytes.len() < 64 * 64 / 2 + 64,
        "a constant 64x64 image should not exceed half a byte per sample, got {}",
        bytes.len()
    );
}

#[test]
fn truncating_the_output_errors_and_never_panics() {
    let image = GreyImage::new(32, 32, (0..32u32 * 32).map(|i| (i % 251) as u8).collect())
        .expect("valid image");
    let bytes = encode_grey8(&image).expect("encodes");
    for cut in 0..bytes.len() {
        let prefix = bytes.get(..cut).expect("cut is within the buffer");
        let _ = jpxl_decode::decode(prefix, &Limits::default());
    }
}

#[test]
fn corrupting_the_output_errors_and_never_panics() {
    let image = GreyImage::new(16, 16, (0..256u32).map(|i| (i % 251) as u8).collect())
        .expect("valid image");
    let bytes = encode_grey8(&image).expect("encodes");
    for index in 0..bytes.len() {
        for flip in [0x01u8, 0x80, 0xFF] {
            let mut corrupted = bytes.clone();
            if let Some(byte) = corrupted.get_mut(index) {
                *byte ^= flip;
            }
            let _ = jpxl_decode::decode(&corrupted, &Limits::default());
        }
    }
}
