//! End-to-end decodes of hand-built modular sub-bitstreams (18181-1 Annex H).
//!
//! Every stream here is spelled out field by field with its clause citation,
//! and every expected sample is hand-derived in a comment above the assertion.
//! What these tests prove that the unit tests cannot: that the *composition*
//! is right — header, MA tree stream, data stream and inverse transforms in the
//! order H.2 states, with the bit positions lining up.

#![allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]

mod modular_common;

use jpxl_bitstream::BitReader;
use jpxl_core::limits::Limits;
use jpxl_decode::modular::{
    ChannelSpec, ModularOptions, TreeSource, decode_sub_bitstream, transform::Transform,
};
use modular_common::{
    BitWriter, write_modular_header_no_transforms, write_prefix_bundle, write_token,
};

/// The alphabet used by every data stream below: symbols 0..=3, each a 2-bit
/// code, and `split_exponent = 4` so every token is a literal value.
const ALPHABET: usize = 4;
const SPLIT_EXPONENT: u32 = 4;

/// Writes the six-context MA tree stream of H.4.2 for a single leaf.
///
/// The leaf's five fields are read from contexts 2, 3, 4 and 5 after context 1
/// reports "leaf". All six contexts share one cluster, so all six use the same
/// four-symbol code.
fn write_single_leaf_tree(w: &mut BitWriter, predictor: u32) {
    write_prefix_bundle(w, 6, ALPHABET, SPLIT_EXPONENT);
    // node 0: property = DecodeHybridVarLenUint(1) - 1. Token 0 -> -1 -> leaf.
    write_token(w, ALPHABET, 0);
    write_token(w, ALPHABET, predictor); // leaf_node.predictor
    write_token(w, ALPHABET, 0); // leaf_node.offset = UnpackSigned(0) = 0
    write_token(w, ALPHABET, 0); // mul_log  = 0
    write_token(w, ALPHABET, 0); // mul_bits = 0 -> multiplier = 1
}

#[test]
fn one_leaf_zero_predictor_decodes_bit_exactly() {
    // A 2x2 single-channel image, one MA leaf, predictor 0 (Zero), multiplier
    // 1, offset 0. Every sample is therefore just UnpackSigned(token):
    //   token 2 -> 1, token 1 -> -1, token 0 -> 0, token 3 -> -2.
    let mut w = BitWriter::new();
    write_modular_header_no_transforms(&mut w, false);
    let header_bits = w.bit_len();
    assert_eq!(header_bits, 4, "Bool + default_wp Bool + U32 selector");

    write_single_leaf_tree(&mut w, 0);
    write_prefix_bundle(&mut w, 1, ALPHABET, SPLIT_EXPONENT);
    for token in [2u32, 1, 0, 3] {
        write_token(&mut w, ALPHABET, token);
    }
    let data = w.finish();

    let mut reader = BitReader::new(&data);
    let image = decode_sub_bitstream(
        &mut reader,
        &[ChannelSpec::new(2, 2)],
        &ModularOptions::default(),
        TreeSource::Local,
        &Limits::relaxed(),
    )
    .expect("a well-formed one-leaf sub-bitstream");

    assert_eq!(image.channels().len(), 1);
    assert_eq!(image.channels()[0].samples(), &[1, -1, 0, -2]);
    assert_eq!(image.nb_meta_channels(), 0);
}

#[test]
fn the_west_predictor_accumulates_along_a_row() {
    // Predictor 1 (West). W is 0 at the origin, and on the first row of a
    // 4x1 channel it is the previous sample.
    //   x=0: diff = UnpackSigned(2) =  1, W = 0 -> 1
    //   x=1: diff = UnpackSigned(2) =  1, W = 1 -> 2
    //   x=2: diff = UnpackSigned(1) = -1, W = 2 -> 1
    //   x=3: diff = UnpackSigned(0) =  0, W = 1 -> 1
    let mut w = BitWriter::new();
    write_modular_header_no_transforms(&mut w, false);
    write_single_leaf_tree(&mut w, 1);
    write_prefix_bundle(&mut w, 1, ALPHABET, SPLIT_EXPONENT);
    for token in [2u32, 2, 1, 0] {
        write_token(&mut w, ALPHABET, token);
    }
    let data = w.finish();

    let mut reader = BitReader::new(&data);
    let image = decode_sub_bitstream(
        &mut reader,
        &[ChannelSpec::new(4, 1)],
        &ModularOptions::default(),
        TreeSource::Local,
        &Limits::relaxed(),
    )
    .expect("West-predicted stream");
    assert_eq!(image.channels()[0].samples(), &[1, 2, 1, 1]);
}

#[test]
fn the_north_predictor_reads_the_row_above() {
    // Predictor 2 (North). This is the smallest end-to-end case that proves the
    // raster loop advances rows correctly and that the H.5 state, which is
    // rotated once per row whatever the leaf predictor, does not disturb it.
    // (Predictor 6 itself needs a tree symbol of 6, which the four-symbol
    // helper alphabet cannot carry; it is covered by the H.5 unit tests and by
    // the palette `d_pred == 6` path.)
    let mut w = BitWriter::new();
    write_modular_header_no_transforms(&mut w, false);
    write_single_leaf_tree(&mut w, 2);
    write_prefix_bundle(&mut w, 1, ALPHABET, SPLIT_EXPONENT);
    // 3x2 channel: row 0 has no north neighbour (N falls back to W, which at
    // the origin is 0).
    //   (0,0): W = 0, N = W = 0; diff = 1 -> 1
    //   (1,0): W = 1, N = W = 1; diff = 0 -> 1
    //   (2,0): W = 1, N = W = 1; diff = 1 -> 2
    //   (0,1): N = c(0,0) = 1; diff = 0 -> 1
    //   (1,1): N = c(1,0) = 1; diff = 1 -> 2
    //   (2,1): N = c(2,0) = 2; diff = -1 -> 1
    for token in [2u32, 0, 2, 0, 2, 1] {
        write_token(&mut w, ALPHABET, token);
    }
    let data = w.finish();

    let mut reader = BitReader::new(&data);
    let image = decode_sub_bitstream(
        &mut reader,
        &[ChannelSpec::new(3, 2)],
        &ModularOptions::default(),
        TreeSource::Local,
        &Limits::relaxed(),
    )
    .expect("North-predicted stream");
    assert_eq!(image.channels()[0].samples(), &[1, 1, 2, 1, 2, 1]);
}

#[test]
fn a_two_leaf_tree_selects_by_property_and_by_context() {
    // MA tree: root tests `property[2] > 0`, i.e. "is this not the first row".
    //   left  (ctx 0): predictor 2 (North)
    //   right (ctx 1): predictor 0 (Zero)
    //
    // Tree symbols, all through the shared four-symbol code:
    //   ctx 1 -> 3  => property = 3 - 1 = 2 (y)
    //   ctx 0 -> 0  => value = UnpackSigned(0) = 0
    //   ctx 1 -> 0  => leaf; ctx 2 -> 2 (North); ctx 3,4,5 -> 0
    //   ctx 1 -> 0  => leaf; ctx 2 -> 0 (Zero);  ctx 3,4,5 -> 0
    let mut w = BitWriter::new();
    write_modular_header_no_transforms(&mut w, false);
    write_prefix_bundle(&mut w, 6, ALPHABET, SPLIT_EXPONENT);
    write_token(&mut w, ALPHABET, 3); // decision on property 2
    write_token(&mut w, ALPHABET, 0); // threshold 0
    for predictor in [2u32, 0] {
        write_token(&mut w, ALPHABET, 0); // leaf marker
        write_token(&mut w, ALPHABET, predictor);
        write_token(&mut w, ALPHABET, 0); // offset
        write_token(&mut w, ALPHABET, 0); // mul_log
        write_token(&mut w, ALPHABET, 0); // mul_bits
    }

    // Two leaves -> the data stream has two pre-clustered contexts.
    write_prefix_bundle(&mut w, 2, ALPHABET, SPLIT_EXPONENT);
    //   (0,0): y = 0 -> right leaf, Zero. diff = UnpackSigned(2) = 1  -> 1
    //   (1,0): y = 0 -> right leaf, Zero. diff = UnpackSigned(1) = -1 -> -1
    //   (0,1): y = 1 -> left leaf, North = c(0,0) = 1. diff = 0       -> 1
    //   (1,1): y = 1 -> left leaf, North = c(1,0) = -1. diff = 1      -> 0
    for token in [2u32, 1, 0, 2] {
        write_token(&mut w, ALPHABET, token);
    }
    let data = w.finish();

    let mut reader = BitReader::new(&data);
    let image = decode_sub_bitstream(
        &mut reader,
        &[ChannelSpec::new(2, 2)],
        &ModularOptions::default(),
        TreeSource::Local,
        &Limits::relaxed(),
    )
    .expect("two-leaf sub-bitstream");
    assert_eq!(image.channels()[0].samples(), &[1, -1, 1, 0]);
}

#[test]
fn three_channels_decode_independently_then_the_rct_recombines_them() {
    // nb_transforms = 1, kRCT with begin_c = 0 and rct_type = 10, which H.6.3's
    // own example says turns (G, B', R') into (R' + G, G, B' + G).
    let mut w = BitWriter::new();
    w.bit(false); // use_global_tree
    w.bit(true); // default_wp
    w.u32_dist(1, 0, 0); // nb_transforms: distribution 1 = the constant 1
    w.u(0, 2); // tr = kRCT
    w.u32_dist(0, 0, 3); // begin_c: u(3) = 0
    w.u32_dist(2, 10 - 2, 4); // rct_type: 2 + u(4) = 10

    write_single_leaf_tree(&mut w, 0); // predictor Zero everywhere
    write_prefix_bundle(&mut w, 1, ALPHABET, SPLIT_EXPONENT);
    // Three 1x1 channels: A = 2 (token 4 is out of range, so use the
    // UnpackSigned values reachable from tokens 0..3: 0, -1, 1, -2).
    //   channel 0 (A) = UnpackSigned(2) =  1
    //   channel 1 (B) = UnpackSigned(1) = -1
    //   channel 2 (C) = UnpackSigned(0) =  0
    for token in [2u32, 1, 0] {
        write_token(&mut w, ALPHABET, token);
    }
    let data = w.finish();

    let mut reader = BitReader::new(&data);
    let image = decode_sub_bitstream(
        &mut reader,
        &[ChannelSpec::new(1, 1); 3],
        &ModularOptions::default(),
        TreeSource::Local,
        &Limits::relaxed(),
    )
    .expect("RCT sub-bitstream");

    // rct_type 10: permutation 1, type 3. type & 1 -> C += A = 0 + 1 = 1;
    // (type >> 1) == 1 -> B += A = -1 + 1 = 0. D = A = 1, E = B = 0, F = C = 1.
    // V[1] = D = 1, V[2] = E = 0, V[0] = F = 1.
    assert_eq!(image.channels().len(), 3);
    assert_eq!(image.channels()[0].samples(), &[1]);
    assert_eq!(image.channels()[1].samples(), &[1]);
    assert_eq!(image.channels()[2].samples(), &[0]);
}

#[test]
fn a_squeeze_transform_reshapes_the_channel_list_and_is_inverted() {
    // nb_transforms = 1, kSqueeze with one explicit horizontal in-place step
    // over the single 4x1 channel. Forward: 4x1 becomes a 2x1 low-pass plus a
    // 2x1 residual. The decoder must decode two channels and hand back one.
    let mut w = BitWriter::new();
    w.bit(false); // use_global_tree
    w.bit(true); // default_wp
    w.u32_dist(1, 0, 0); // nb_transforms = 1
    w.u(2, 2); // tr = kSqueeze
    w.u32_dist(1, 0, 4); // num_sq: 1 + u(4) = 1
    w.bit(true); // horizontal
    w.bit(true); // in_place
    w.u32_dist(0, 0, 3); // begin_c = 0
    w.u32_dist(0, 0, 0); // num_c: distribution 0 = the constant 1

    write_single_leaf_tree(&mut w, 0); // predictor Zero
    write_prefix_bundle(&mut w, 1, ALPHABET, SPLIT_EXPONENT);
    // Channel 0 (low-pass, 2x1) then channel 1 (residual, 2x1), each in raster
    // order:  low = [1, -1] from tokens 2, 1;  residual = [0, 0] from tokens 0.
    for token in [2u32, 1, 0, 0] {
        write_token(&mut w, ALPHABET, token);
    }
    let data = w.finish();

    let mut reader = BitReader::new(&data);
    let image = decode_sub_bitstream(
        &mut reader,
        &[ChannelSpec::new(4, 1)],
        &ModularOptions::default(),
        TreeSource::Local,
        &Limits::relaxed(),
    )
    .expect("squeeze sub-bitstream");

    assert_eq!(image.channels().len(), 1, "the residual is consumed");
    let ch = &image.channels()[0];
    assert_eq!((ch.width(), ch.height()), (4, 1));

    // Hand-derived inverse (H.6.2.2), low = [1, -1], residual = [0, 0]:
    //   x = 0: avg = 1, residu = 0, next_avg = -1, left = avg = 1.
    //     tendency(1, 1, -1): A >= B >= C, so the descending branch.
    //       X = (4*1 - 3*(-1) - 1 + 6) Idiv 12 = (4 + 3 - 1 + 6)/12 = 12/12 = 1
    //       X - (X & 1) = 0 > 2*(A - B) = 0 ? no
    //       X + (X & 1) = 2 > 2*(B - C) = 4 ? no
    //       -> 1
    //     diff  = 0 + 1 = 1
    //     first = 1 + (1 Idiv 2) = 1
    //     out(0) = 1, out(1) = 1 - 1 = 0
    //   x = 1: avg = -1, residu = 0, next_avg = avg = -1, left = out(1) = 0.
    //     tendency(0, -1, -1): A >= B >= C, descending.
    //       X = (0 - 3*(-1) - (-1) + 6) Idiv 12 = (0 + 3 + 1 + 6)/12 = 10/12 = 0
    //       X - (X & 1) = 0 > 2*(A - B) = 2 ? no
    //       X + (X & 1) = 0 > 2*(B - C) = 0 ? no
    //       -> 0
    //     diff  = 0
    //     first = -1 + 0 = -1
    //     out(2) = -1, out(3) = -1
    assert_eq!(ch.samples(), &[1, 0, -1, -1]);
    assert_eq!(ch.hshift(), 0, "the forward shift increment is undone");
}

#[test]
fn a_multi_group_sized_channel_decodes_row_by_row() {
    // AGENTS.md requires a >= 256x256 fixture on any path that can see more
    // than one group. The modular sub-bitstream itself never sees group
    // geometry, but a channel this size is what a group-spanning decode hands
    // it, and it exercises the row rotation of the H.5 error state and the
    // per-row `advance_row` far past the hand-checkable cases.
    const W: u32 = 260;
    const H: u32 = 260;

    let mut w = BitWriter::new();
    write_modular_header_no_transforms(&mut w, false);
    write_single_leaf_tree(&mut w, 1); // predictor 1 = West
    write_prefix_bundle(&mut w, 1, ALPHABET, SPLIT_EXPONENT);

    // A deterministic token pattern, and the same recurrence computed here so
    // the expectation is independent of the decoder.
    let mut expected = Vec::with_capacity((W * H) as usize);
    for y in 0..H {
        for x in 0..W {
            let token = (x.wrapping_mul(7).wrapping_add(y.wrapping_mul(3))) % 4;
            write_token(&mut w, ALPHABET, token);
            let diff = match token {
                0 => 0i32,
                1 => -1,
                2 => 1,
                _ => -2,
            };
            // H.3: W is the sample to the left, or the sample above on x == 0,
            // or 0 at the origin.
            let west = if x > 0 {
                expected[(y * W + x - 1) as usize]
            } else if y > 0 {
                expected[((y - 1) * W) as usize]
            } else {
                0
            };
            expected.push(diff + west);
        }
    }
    let data = w.finish();

    let mut reader = BitReader::new(&data);
    let image = decode_sub_bitstream(
        &mut reader,
        &[ChannelSpec::new(W, H)],
        &ModularOptions::default(),
        TreeSource::Local,
        &Limits::relaxed(),
    )
    .expect("260x260 stream");
    assert_eq!(image.channels()[0].samples(), expected.as_slice());
}

#[test]
fn zero_sized_channels_are_skipped_not_decoded() {
    // H.2: "skipping any channels having width or height zero". If the zero
    // channel consumed symbols the following channel would decode garbage, so
    // the assertion on the 1x1 channel is the real check.
    let mut w = BitWriter::new();
    write_modular_header_no_transforms(&mut w, false);
    write_single_leaf_tree(&mut w, 0);
    write_prefix_bundle(&mut w, 1, ALPHABET, SPLIT_EXPONENT);
    write_token(&mut w, ALPHABET, 3); // UnpackSigned(3) = -2
    let data = w.finish();

    let mut reader = BitReader::new(&data);
    let image = decode_sub_bitstream(
        &mut reader,
        &[
            ChannelSpec::new(0, 5),
            ChannelSpec::new(1, 1),
            ChannelSpec::new(7, 0),
        ],
        &ModularOptions::default(),
        TreeSource::Local,
        &Limits::relaxed(),
    )
    .expect("stream with empty channels");
    assert_eq!(image.channels().len(), 3);
    assert!(image.channels()[0].samples().is_empty());
    assert_eq!(image.channels()[1].samples(), &[-2]);
    assert!(image.channels()[2].samples().is_empty());
}

#[test]
fn the_transform_list_is_exposed_after_parsing() {
    // Slice 6/7 needs to see what was signalled; prove the accessor reports the
    // resolved chain rather than the raw bits.
    let mut w = BitWriter::new();
    w.bit(false);
    w.bit(true);
    w.u32_dist(1, 0, 0); // nb_transforms = 1
    w.u(0, 2); // kRCT
    w.u32_dist(0, 0, 3); // begin_c = 0
    w.u32_dist(0, 0, 0); // rct_type: distribution 0 = the constant 6
    write_single_leaf_tree(&mut w, 0);
    write_prefix_bundle(&mut w, 1, ALPHABET, SPLIT_EXPONENT);
    for _ in 0..3 {
        write_token(&mut w, ALPHABET, 0);
    }
    let data = w.finish();

    let mut reader = BitReader::new(&data);
    let mut guard = jpxl_core::limits::AllocGuard::new(&Limits::relaxed());
    let header = jpxl_decode::modular::ModularHeader::read(
        &mut reader,
        &[ChannelSpec::new(1, 1); 3],
        &ModularOptions::default(),
        &mut guard,
    )
    .expect("header");
    assert!(!header.use_global_tree());
    assert_eq!(
        header.transforms(),
        &[Transform::Rct {
            begin_c: 0,
            rct_type: 6
        }]
    );
    assert_eq!(header.layout().specs.len(), 3, "the RCT changes no shapes");
}

// ---------------------------------------------------------------------------
// Rejection of malformed and oversized input
// ---------------------------------------------------------------------------

#[test]
fn use_global_tree_without_a_global_tree_is_an_error() {
    let mut w = BitWriter::new();
    write_modular_header_no_transforms(&mut w, true);
    write_prefix_bundle(&mut w, 1, ALPHABET, SPLIT_EXPONENT);
    let data = w.finish();

    let mut reader = BitReader::new(&data);
    let err = decode_sub_bitstream(
        &mut reader,
        &[ChannelSpec::new(1, 1)],
        &ModularOptions::default(),
        TreeSource::Local,
        &Limits::relaxed(),
    )
    .expect_err("no global tree was supplied");
    assert!(err.to_string().contains("H.2"));
}

#[test]
fn nb_transforms_over_the_level_limit_is_rejected() {
    // Level 5 caps nb_transforms at 8; signal 18 through the fourth
    // distribution and require rejection before any transform is parsed.
    let mut w = BitWriter::new();
    w.bit(false);
    w.bit(true);
    w.u32_dist(3, 0, 8); // 18 + u(8) = 18
    let data = w.finish();

    let mut reader = BitReader::new(&data);
    let err = decode_sub_bitstream(
        &mut reader,
        &[ChannelSpec::new(1, 1)],
        &ModularOptions::level5(),
        TreeSource::Local,
        &Limits::relaxed(),
    )
    .expect_err("18 transforms exceeds the level-5 limit of 8");
    assert!(err.to_string().contains("nb_transforms"));
}

#[test]
fn an_rct_over_channels_that_do_not_exist_is_rejected_before_decoding() {
    let mut w = BitWriter::new();
    w.bit(false);
    w.bit(true);
    w.u32_dist(1, 0, 0); // nb_transforms = 1
    w.u(0, 2); // kRCT
    w.u32_dist(0, 2, 3); // begin_c = 2, so it needs channels 2, 3, 4
    w.u32_dist(0, 0, 0); // rct_type = 6
    let data = w.finish();

    let mut reader = BitReader::new(&data);
    let err = decode_sub_bitstream(
        &mut reader,
        &[ChannelSpec::new(1, 1); 3],
        &ModularOptions::default(),
        TreeSource::Local,
        &Limits::relaxed(),
    )
    .expect_err("only three channels exist");
    assert!(err.to_string().contains("H.6.3"));
}

#[test]
fn an_allocation_budget_stops_an_oversized_channel_list() {
    // A 1000x1000 channel is 4 MB of samples; a 1 KB budget must refuse it, and
    // must refuse it before the allocation rather than after.
    let mut w = BitWriter::new();
    write_modular_header_no_transforms(&mut w, false);
    write_single_leaf_tree(&mut w, 0);
    write_prefix_bundle(&mut w, 1, ALPHABET, SPLIT_EXPONENT);
    let data = w.finish();

    let limits = Limits {
        max_alloc_bytes: 1024,
        ..Limits::default()
    };
    let mut reader = BitReader::new(&data);
    let err = decode_sub_bitstream(
        &mut reader,
        &[ChannelSpec::new(1000, 1000)],
        &ModularOptions::default(),
        TreeSource::Local,
        &limits,
    )
    .expect_err("4 MB does not fit in 1 KB");
    assert!(err.to_string().contains("limit exceeded"));
}

#[test]
fn a_truncated_stream_errors_rather_than_panicking() {
    // Take a known-good stream and cut it progressively shorter. Every prefix
    // must produce an error or a decode, never a panic and never a hang.
    let mut w = BitWriter::new();
    write_modular_header_no_transforms(&mut w, false);
    write_single_leaf_tree(&mut w, 3); // Avg(W, N)
    write_prefix_bundle(&mut w, 1, ALPHABET, SPLIT_EXPONENT);
    for token in [2u32, 1, 0, 3] {
        write_token(&mut w, ALPHABET, token);
    }
    let data = w.finish();

    for cut in 0..data.len() {
        let mut reader = BitReader::new(&data[..cut]);
        let _ = decode_sub_bitstream(
            &mut reader,
            &[ChannelSpec::new(2, 2)],
            &ModularOptions::default(),
            TreeSource::Local,
            &Limits::relaxed(),
        );
    }
}

#[test]
fn arbitrary_bit_patterns_never_panic() {
    // A cheap structured fuzz over the header and tree region. The point is not
    // coverage, it is that a decode path that is reached with garbage input
    // returns `Err` instead of unwinding.
    let mut seed = 0x9E37_79B9u32;
    for _ in 0..2000 {
        let mut bytes = Vec::with_capacity(24);
        for _ in 0..24 {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            bytes.push((seed >> 16) as u8);
        }
        let mut reader = BitReader::new(&bytes);
        let _ = decode_sub_bitstream(
            &mut reader,
            &[ChannelSpec::new(3, 3), ChannelSpec::new(3, 3)],
            &ModularOptions::level5(),
            TreeSource::Local,
            &Limits {
                max_alloc_bytes: 1 << 20,
                ..Limits::default()
            },
        );
    }
}
