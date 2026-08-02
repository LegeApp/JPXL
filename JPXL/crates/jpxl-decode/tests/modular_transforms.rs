//! Inverse-transform properties, proved against independently written forward
//! transforms (18181-1 H.6).
//!
//! The in-module unit tests pin each inverse against hand-computed vectors.
//! These tests prove the stronger statement the transforms exist for: they are
//! *reversible*. The forward direction is written here, from the algebra of the
//! inverse, so a sign error shared between the two would have to be made twice
//! in two different shapes to survive.

#![allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]

mod modular_common;

use jpxl_bitstream::BitReader;
use jpxl_core::limits::{AllocGuard, Limits};
use jpxl_decode::modular::{
    Channel, ChannelSpec, ModularOptions, TreeSource, decode_sub_bitstream,
    squeeze::{horiz_isqueeze, tendency, vert_isqueeze},
};
use modular_common::{BitWriter, write_prefix_bundle, write_token};

const ALPHABET: usize = 4;
const SPLIT_EXPONENT: u32 = 4;

fn guard() -> AllocGuard {
    AllocGuard::new(&Limits::relaxed())
}

/// The forward horizontal squeeze, derived by solving H.6.2.2 for its inputs.
///
/// The inverse computes, for each output pair `(a, b) = (out(2x), out(2x+1))`:
///
/// ```text
/// diff  = residu + tendency(left, avg, next_avg)
/// first = avg + diff Idiv 2      -> a
/// a - diff                       -> b
/// ```
///
/// so `diff = a - b` and `avg = a - (diff Idiv 2)`. `residu` then follows once
/// `next_avg` is known, which needs the *whole* low-pass channel, hence two
/// passes.
fn horiz_squeeze(input: &[i32], width: usize) -> (Vec<i32>, Vec<i32>) {
    let w2 = width / 2;
    let w1 = width.div_ceil(2);

    let mut low = vec![0i32; w1];
    for x in 0..w2 {
        let a = i64::from(input[2 * x]);
        let b = i64::from(input[2 * x + 1]);
        let diff = a - b;
        low[x] = (a - diff / 2) as i32;
    }
    if w1 > w2 {
        low[w2] = input[2 * w2];
    }

    let mut residual = vec![0i32; w2];
    for x in 0..w2 {
        let a = i64::from(input[2 * x]);
        let b = i64::from(input[2 * x + 1]);
        let diff = a - b;
        let avg = i64::from(low[x]);
        let next_avg = if x + 1 < w1 {
            i64::from(low[x + 1])
        } else {
            avg
        };
        let left = if x > 0 {
            i64::from(input[2 * x - 1])
        } else {
            avg
        };
        residual[x] = (diff - tendency(left, avg, next_avg)) as i32;
    }
    (low, residual)
}

/// The vertical twin of [`horiz_squeeze`], over one column at a time.
fn vert_squeeze(input: &[i32], width: usize, height: usize) -> (Vec<i32>, Vec<i32>) {
    let h2 = height / 2;
    let h1 = height.div_ceil(2);
    let mut low = vec![0i32; width * h1];
    let mut residual = vec![0i32; width * h2];

    for x in 0..width {
        for y in 0..h2 {
            let a = i64::from(input[2 * y * width + x]);
            let b = i64::from(input[(2 * y + 1) * width + x]);
            let diff = a - b;
            low[y * width + x] = (a - diff / 2) as i32;
        }
        if h1 > h2 {
            low[h2 * width + x] = input[2 * h2 * width + x];
        }
    }
    for x in 0..width {
        for y in 0..h2 {
            let a = i64::from(input[2 * y * width + x]);
            let b = i64::from(input[(2 * y + 1) * width + x]);
            let diff = a - b;
            let avg = i64::from(low[y * width + x]);
            let next_avg = if y + 1 < h1 {
                i64::from(low[(y + 1) * width + x])
            } else {
                avg
            };
            let top = if y > 0 {
                i64::from(input[(2 * y - 1) * width + x])
            } else {
                avg
            };
            residual[y * width + x] = (diff - tendency(top, avg, next_avg)) as i32;
        }
    }
    (low, residual)
}

/// A deterministic pseudo-random sample generator, so the test data is fixed.
fn samples(count: usize, seed: u32, spread: i32) -> Vec<i32> {
    let mut s = seed | 1;
    (0..count)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 13) as i32).rem_euclid(2 * spread + 1) - spread
        })
        .collect()
}

#[test]
fn horizontal_squeeze_round_trips_for_even_and_odd_widths() {
    for width in 1..=17usize {
        for height in [1usize, 3] {
            for (seed, spread) in [(1u32, 3i32), (7, 200), (99, 100_000)] {
                let original = samples(width * height, seed, spread);
                // Squeeze each row, then rebuild.
                let mut low = Vec::new();
                let mut res = Vec::new();
                for y in 0..height {
                    let row = &original[y * width..(y + 1) * width];
                    let (l, r) = horiz_squeeze(row, width);
                    low.extend_from_slice(&l);
                    res.extend_from_slice(&r);
                }
                let w1 = width.div_ceil(2) as u32;
                let w2 = (width / 2) as u32;
                let lo = Channel::from_samples(ChannelSpec::new(w1, height as u32), low)
                    .expect("low-pass");
                let hi = Channel::from_samples(ChannelSpec::new(w2, height as u32), res)
                    .expect("residual");
                let out = horiz_isqueeze(&lo, &hi, &mut guard()).expect("inverse");
                assert_eq!(
                    out.samples(),
                    original.as_slice(),
                    "width {width} height {height} seed {seed} spread {spread}"
                );
            }
        }
    }
}

#[test]
fn vertical_squeeze_round_trips_for_even_and_odd_heights() {
    for height in 1..=17usize {
        for width in [1usize, 4] {
            for (seed, spread) in [(3u32, 5i32), (11, 1000)] {
                let original = samples(width * height, seed, spread);
                let (low, res) = vert_squeeze(&original, width, height);
                let h1 = height.div_ceil(2) as u32;
                let h2 = (height / 2) as u32;
                let lo = Channel::from_samples(ChannelSpec::new(width as u32, h1), low)
                    .expect("low-pass");
                let hi = Channel::from_samples(ChannelSpec::new(width as u32, h2), res)
                    .expect("residual");
                let out = vert_isqueeze(&lo, &hi, &mut guard()).expect("inverse");
                assert_eq!(
                    out.samples(),
                    original.as_slice(),
                    "width {width} height {height} seed {seed}"
                );
            }
        }
    }
}

#[test]
fn a_horizontal_then_vertical_squeeze_pair_round_trips() {
    // The two directions compose, which is what a real squeeze chain does.
    let (width, height) = (9usize, 6usize);
    let original = samples(width * height, 42, 500);

    let (low_v, res_v) = vert_squeeze(&original, width, height);
    let h1 = height.div_ceil(2);
    let h2 = height / 2;

    let mut low_h = Vec::new();
    let mut res_h = Vec::new();
    for y in 0..h1 {
        let (l, r) = horiz_squeeze(&low_v[y * width..(y + 1) * width], width);
        low_h.extend_from_slice(&l);
        res_h.extend_from_slice(&r);
    }
    let w1 = width.div_ceil(2) as u32;
    let w2 = (width / 2) as u32;
    let lo = Channel::from_samples(ChannelSpec::new(w1, h1 as u32), low_h).expect("ll");
    let hi = Channel::from_samples(ChannelSpec::new(w2, h1 as u32), res_h).expect("lh");
    let rebuilt_low_v = horiz_isqueeze(&lo, &hi, &mut guard()).expect("h inverse");
    assert_eq!(rebuilt_low_v.samples(), low_v.as_slice());

    let lo = Channel::from_samples(ChannelSpec::new(width as u32, h1 as u32), low_v).expect("low");
    let hi =
        Channel::from_samples(ChannelSpec::new(width as u32, h2 as u32), res_v).expect("residual");
    let out = vert_isqueeze(&lo, &hi, &mut guard()).expect("v inverse");
    assert_eq!(out.samples(), original.as_slice());
}

#[test]
fn a_palette_transform_decodes_end_to_end() {
    // nb_transforms = 1, kPalette over one channel with two colours and no
    // deltas. The channel list the decoder must derive is:
    //   channel 0: the palette meta-channel, nb_colours x num_c = 2 x 1
    //   channel 1: the index channel, 3 x 1
    let mut w = BitWriter::new();
    w.bit(false); // use_global_tree
    w.bit(true); // default_wp
    w.u32_dist(1, 0, 0); // nb_transforms = 1
    w.u(1, 2); // tr = kPalette
    w.u32_dist(0, 0, 3); // begin_c = 0
    w.u32_dist(0, 0, 0); // num_c: distribution 0 = the constant 1
    w.u32_dist(0, 2, 8); // nb_colours: u(8) = 2
    w.u32_dist(0, 0, 0); // nb_deltas: distribution 0 = the constant 0
    w.u(0, 4); // d_pred = 0 (unused: nb_deltas is 0)

    // One MA leaf, predictor 0 (Zero), offset 0, multiplier 1.
    write_prefix_bundle(&mut w, 6, ALPHABET, SPLIT_EXPONENT);
    for token in [0u32, 0, 0, 0, 0] {
        write_token(&mut w, ALPHABET, token);
    }

    write_prefix_bundle(&mut w, 1, ALPHABET, SPLIT_EXPONENT);
    // Palette meta-channel, 2 entries:  UnpackSigned(2) = 1, UnpackSigned(1) = -1.
    for token in [2u32, 1] {
        write_token(&mut w, ALPHABET, token);
    }
    // Index channel, 3 samples: 0, 1, 0 via tokens 0, 2, 0.
    for token in [0u32, 2, 0] {
        write_token(&mut w, ALPHABET, token);
    }
    let data = w.finish();

    let mut reader = BitReader::new(&data);
    let image = decode_sub_bitstream(
        &mut reader,
        &[ChannelSpec::new(3, 1)],
        &ModularOptions::default(),
        TreeSource::Local,
        &Limits::relaxed(),
    )
    .expect("palette sub-bitstream");

    // The meta-channel is consumed by the inverse, leaving the original shape.
    assert_eq!(image.channels().len(), 1);
    assert_eq!(image.channels()[0].samples(), &[1, -1, 1]);
    assert_eq!(
        image.nb_meta_channels(),
        0,
        "the counter is restored to its pre-transform value"
    );
}

#[test]
fn a_delta_palette_adds_the_signalled_predictor() {
    // Same shape, but nb_deltas = 2 so both indices are deltas, and d_pred = 1
    // (West). Colours are 1 and -1 as before.
    //   x = 0: index 0 -> 1, W = 0 (origin)      -> 1
    //   x = 1: index 1 -> -1, W = 1              -> 0
    //   x = 2: index 0 -> 1, W = 0               -> 1
    let mut w = BitWriter::new();
    w.bit(false);
    w.bit(true);
    w.u32_dist(1, 0, 0); // nb_transforms = 1
    w.u(1, 2); // kPalette
    w.u32_dist(0, 0, 3); // begin_c = 0
    w.u32_dist(0, 0, 0); // num_c = 1
    w.u32_dist(0, 2, 8); // nb_colours = 2
    w.u32_dist(1, 1, 8); // nb_deltas: 1 + u(8) = 2
    w.u(1, 4); // d_pred = 1 (West)

    write_prefix_bundle(&mut w, 6, ALPHABET, SPLIT_EXPONENT);
    for token in [0u32, 0, 0, 0, 0] {
        write_token(&mut w, ALPHABET, token);
    }
    write_prefix_bundle(&mut w, 1, ALPHABET, SPLIT_EXPONENT);
    for token in [2u32, 1] {
        write_token(&mut w, ALPHABET, token); // palette entries 1, -1
    }
    for token in [0u32, 2, 0] {
        write_token(&mut w, ALPHABET, token); // indices 0, 1, 0
    }
    let data = w.finish();

    let mut reader = BitReader::new(&data);
    let image = decode_sub_bitstream(
        &mut reader,
        &[ChannelSpec::new(3, 1)],
        &ModularOptions::default(),
        TreeSource::Local,
        &Limits::relaxed(),
    )
    .expect("delta palette sub-bitstream");
    assert_eq!(image.channels()[0].samples(), &[1, 0, 1]);
}
