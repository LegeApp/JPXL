//! End-to-end: encode, then decode with `jpxl-decode` and assert the samples
//! come back exactly (`docs/PLAN.md` slice 10, stage 1).
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
use jpxl_encode::{EncodeOptions, GreyImage, Image, encode, encode_grey8};

/// A deterministic `(x, y, channel, max) -> sample` formula.
type Pattern = fn(u32, u32, usize, u32) -> u32;

/// Sample formulas, chosen so the residual distribution differs sharply
/// between them: a constant image codes every residual as 0, a gradient codes
/// almost all of them as 0 with a nonzero first column, noise defeats the
/// predictor entirely, and the extremes exercise the widest tokens.
fn patterns() -> Vec<(&'static str, Pattern)> {
    vec![
        ("black", |_, _, _, _| 0),
        ("white", |_, _, _, max| max),
        ("constant", |_, _, c, max| max / (2 + c as u32)),
        ("ramp", |x, y, c, max| (x + y + c as u32 * 37) % (max + 1)),
        ("full-range-ramp", |x, y, c, max| {
            // Sweeps the whole value range regardless of image size, so a
            // 16-bit image really reaches 65535 and not just 255.
            ((x.wrapping_mul(2_654_435_761)
                .wrapping_add(y)
                .wrapping_add(c as u32))
                % 7)
                * max
                / 7
        }),
        ("checker", |x, y, c, max| {
            if (x + y + c as u32).is_multiple_of(2) {
                0
            } else {
                max
            }
        }),
        ("noise", |x, y, c, max| {
            // A cheap deterministic hash; nothing about it is spec-related,
            // it just has to be reproducible and predictor-hostile.
            let v = x
                .wrapping_mul(2_654_435_761)
                .wrapping_add(y.wrapping_mul(40_503))
                .wrapping_add((c as u32).wrapping_mul(2_246_822_519));
            (v >> 11) % (max + 1)
        }),
    ]
}

/// Builds an image from a pattern.
fn build(width: u32, height: u32, channels: usize, bits: u32, f: Pattern) -> Image {
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

/// Encodes, decodes and asserts sample equality; returns the encoded bytes.
fn round_trip(name: &str, image: &Image, options: &EncodeOptions) -> Vec<u8> {
    let what = format!(
        "{name} {}x{} ch{} {}bit shift{:?}{}",
        image.width(),
        image.height(),
        image.num_channels(),
        image.bits_per_sample(),
        options.group_size_shift,
        if options.container { " boxed" } else { "" }
    );
    let bytes = encode(image, options).unwrap_or_else(|e| panic!("{what}: encode: {e}"));

    let decoded = jpxl_decode::decode(&bytes, &Limits::default())
        .unwrap_or_else(|e| panic!("{what}: decode: {e}"));

    assert_eq!(
        (decoded.width, decoded.height),
        (image.width(), image.height()),
        "{what}: dimensions"
    );
    assert_eq!(
        decoded.num_colour_channels,
        image.num_channels(),
        "{what}: channel count"
    );
    assert_eq!(
        decoded.colour_bits_per_sample(),
        image.bits_per_sample(),
        "{what}: bit depth"
    );

    let got = decoded.interleaved_colour();
    let pixels = image.width() as usize * image.height() as usize;
    let mut want = Vec::with_capacity(pixels * image.num_channels());
    for i in 0..pixels {
        for plane in image.planes() {
            want.push(plane.get(i).copied().unwrap_or(0) as u16);
        }
    }
    assert_eq!(got.len(), want.len(), "{what}: sample count");
    if got != want {
        let first = got
            .iter()
            .zip(want.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        let stride = image.width() as usize * image.num_channels();
        panic!(
            "{what}: first mismatch at index {first} = (x {}, y {}): decoded {:?}, expected {:?}",
            (first % stride) / image.num_channels(),
            first / stride,
            got.get(first),
            want.get(first)
        );
    }
    bytes
}

/// The sizes every configuration is run over.
///
/// Chosen for the edge cases they expose: 1x1 (every neighbour substitution at
/// once), 13x7 and 101x97 (neither dimension a multiple of 8, non-square),
/// 129x1 and 1x129 (a degenerate single row and column), 260x9 (a strip that
/// crosses a group boundary in one axis only).
const SIZES: [(u32, u32); 7] = [
    (1, 1),
    (2, 3),
    (13, 7),
    (101, 97),
    (129, 1),
    (1, 129),
    (260, 9),
];

#[test]
fn grey_round_trips_at_both_bit_depths() {
    for &(w, h) in &SIZES {
        for bits in [8u32, 16] {
            for (name, f) in patterns() {
                let image = build(w, h, 1, bits, f);
                round_trip(name, &image, &EncodeOptions::default());
            }
        }
    }
}

#[test]
fn rgb_round_trips_at_both_bit_depths() {
    for &(w, h) in &SIZES {
        for bits in [8u32, 16] {
            for (name, f) in patterns() {
                let image = build(w, h, 3, bits, f);
                round_trip(name, &image, &EncodeOptions::default());
            }
        }
    }
}

#[test]
fn every_group_size_shift_round_trips() {
    // Forcing a small group turns a modest image into a multi-section frame,
    // so the LfGlobal/LfGroup/HfGlobal/PassGroup layout is exercised without
    // needing a large fixture in every test.
    for shift in 0..=3u32 {
        for &(w, h) in &[(1u32, 1u32), (13, 7), (101, 97), (260, 9), (300, 200)] {
            for channels in [1usize, 3] {
                for bits in [8u32, 16] {
                    let image = build(w, h, channels, bits, |x, y, c, max| {
                        (x * 31 + y * 17 + c as u32 * 5) % (max + 1)
                    });
                    let options = EncodeOptions {
                        container: false,
                        group_size_shift: Some(shift),
                        ..EncodeOptions::default()
                    };
                    round_trip("grid", &image, &options);
                }
            }
        }
    }
}

#[test]
fn a_multi_group_image_round_trips_at_the_default_group_size() {
    // 600x520 with the default group_dim of 512 is a 2x2 group grid whose
    // right and bottom groups are partial — the shape that catches an
    // off-by-one in the group rectangle.
    for channels in [1usize, 3] {
        for bits in [8u32, 16] {
            let image = build(600, 520, channels, bits, |x, y, c, max| {
                let v = x
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(y.wrapping_mul(40_503))
                    .wrapping_add(c as u32);
                (v >> 13) % (max + 1)
            });
            let bytes = round_trip("600x520", &image, &EncodeOptions::default());
            assert!(!bytes.is_empty());
        }
    }
}

#[test]
fn the_container_form_round_trips() {
    for channels in [1usize, 3] {
        for bits in [8u32, 16] {
            let image = build(37, 41, channels, bits, |x, y, c, max| {
                (x * 7 + y * 13 + c as u32) % (max + 1)
            });
            let options = EncodeOptions {
                container: true,
                group_size_shift: None,
                ..EncodeOptions::default()
            };
            let bytes = round_trip("boxed", &image, &options);
            assert_eq!(bytes.get(4..8), Some(&b"JXL "[..]));
        }
    }
}

#[test]
fn the_full_sixteen_bit_range_round_trips() {
    // Every one of the 65536 values appears exactly once, so the residual
    // reaches its extremes in both directions and the widest token the
    // configuration can emit is used.
    let plane: Vec<i32> = (0..65536i32).collect();
    let image = Image::new(256, 256, 16, vec![plane]).expect("valid image");
    round_trip("full-16-bit", &image, &EncodeOptions::default());

    // And again split across groups, so a group boundary lands in the middle
    // of the range rather than at a value that predicts well.
    let options = EncodeOptions {
        container: false,
        group_size_shift: Some(0),
        ..EncodeOptions::default()
    };
    round_trip("full-16-bit-grouped", &image, &options);
}

#[test]
fn the_full_eight_bit_range_round_trips_in_every_channel() {
    let planes: Vec<Vec<i32>> = (0..3)
        .map(|c| (0..256i32).map(|v| (v + c * 85) % 256).collect())
        .collect();
    let image = Image::new(16, 16, 8, planes).expect("valid image");
    round_trip("full-8-bit-rgb", &image, &EncodeOptions::default());
}

#[test]
fn the_legacy_grey8_entry_point_still_works() {
    let image = GreyImage::new(16, 16, (0..256u32).map(|i| (i % 251) as u8).collect())
        .expect("valid image");
    let bytes = encode_grey8(&image).expect("encodes");
    let decoded = jpxl_decode::decode(&bytes, &Limits::default()).expect("decodes");
    assert_eq!(decoded.num_colour_channels, 1);
    assert_eq!(
        decoded.interleaved_colour(),
        image
            .samples()
            .iter()
            .map(|&s| u16::from(s))
            .collect::<Vec<_>>()
    );
}

#[test]
fn a_constant_image_is_small() {
    // Not a compression claim, a sanity check: with the gradient predictor
    // every residual of a constant image is zero, so each sample costs the
    // token bits and nothing else.
    let image = Image::new(64, 64, 8, vec![vec![42; 64 * 64]]).expect("valid image");
    let bytes = round_trip("constant", &image, &EncodeOptions::default());
    assert!(
        bytes.len() < 64 * 64 / 2 + 64,
        "a constant 64x64 image should not exceed half a byte per sample, got {}",
        bytes.len()
    );
}

#[test]
fn truncating_the_output_errors_and_never_panics() {
    // Both shapes: a single-section frame and a multi-section one, because
    // truncation inside a TOC is a different code path from truncation inside
    // a section.
    for options in [
        EncodeOptions::default(),
        EncodeOptions {
            container: false,
            group_size_shift: Some(0),
            ..EncodeOptions::default()
        },
        EncodeOptions {
            container: true,
            group_size_shift: None,
            ..EncodeOptions::default()
        },
    ] {
        let image = build(40, 40, 3, 16, |x, y, c, max| {
            (x * 601 + y * 307 + c as u32 * 11) % (max + 1)
        });
        let bytes = encode(&image, &options).expect("encodes");
        for cut in 0..bytes.len() {
            let prefix = bytes.get(..cut).expect("cut is within the buffer");
            let _ = jpxl_decode::decode(prefix, &Limits::default());
        }
    }
}

#[test]
fn corrupting_the_output_errors_and_never_panics() {
    let image = build(16, 16, 3, 8, |x, y, c, max| {
        (x * 31 + y * 17 + c as u32) % (max + 1)
    });
    for options in [
        EncodeOptions::default(),
        EncodeOptions {
            container: false,
            group_size_shift: Some(0),
            ..EncodeOptions::default()
        },
    ] {
        let bytes = encode(&image, &options).expect("encodes");
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
}
