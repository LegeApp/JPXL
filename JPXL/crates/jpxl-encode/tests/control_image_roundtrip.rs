//! Phase Q0b: VarDCT's control-image writer (`vardct::modular_out`) against
//! `jpxl-decode`'s modular sub-bitstream reader.
//!
//! The writer learns an MA tree over Table H.4 properties and evaluates those
//! properties itself; the only proof that its evaluation matches the decoder's
//! `PropertyBuilder` — edge rules, property 8's west-gradient error, the
//! previous-channel properties, the breadth-first leaf numbering — is a
//! sample-exact round trip. Every shape the LF group writes is exercised:
//! three equal LF planes (previous-channel properties live), CfL tiles, a
//! two-row `BlockInfo`, and a constant `Sharpness` plane, plus a stream with a
//! zero-size channel in the middle (H.2 skips it, the channel index does not).

use jpxl_bitstream::{BitReader, BitWriter};
use jpxl_core::limits::Limits;
use jpxl_decode::modular::{ChannelSpec, ModularOptions, TreeSource, decode_sub_bitstream};
use jpxl_encode::vardct::modular_out::{
    OutChannel, write_modular_stream, write_modular_stream_with,
};

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// A plane with smooth structure on the left and noise on the right, so the
/// learner has something to split on.
fn textured(width: u32, height: u32, seed: u32, amplitude: i32) -> Vec<i32> {
    let mut state = seed | 1;
    (0..width * height)
        .map(|i| {
            let x = i % width;
            let y = i / width;
            let base = (x as i32 * 3 + y as i32 * 2) / 4;
            if x < width / 2 {
                base
            } else {
                base + (xorshift(&mut state) % (2 * amplitude as u32 + 1)) as i32 - amplitude
            }
        })
        .collect()
}

fn roundtrip(channels: &[OutChannel<'_>]) {
    roundtrip_with(channels, None);
}

/// `previous_channel_properties`: `None` = the production switch, `Some(b)` =
/// forced (the on-path is what jxl-oxide 0.12.6 misreads; `jpxl-decode` must
/// keep reading it per H.4.1 regardless).
fn roundtrip_with(channels: &[OutChannel<'_>], previous_channel_properties: Option<bool>) {
    let mut w = BitWriter::new();
    match previous_channel_properties {
        None => write_modular_stream(&mut w, channels).expect("writes"),
        Some(on) => write_modular_stream_with(&mut w, channels, on).expect("writes"),
    }
    // A trailing sentinel proves the reader stops exactly at the stream's end.
    w.write_bits(11, 0x5A5).expect("sentinel");
    let bytes = w.into_bytes();

    let specs: Vec<ChannelSpec> = channels
        .iter()
        .map(|c| ChannelSpec::new(c.width, c.height))
        .collect();
    let mut reader = BitReader::new(&bytes);
    let image = decode_sub_bitstream(
        &mut reader,
        &specs,
        &ModularOptions::level10(),
        TreeSource::Local,
        &Limits::default(),
    )
    .expect("decodes");
    assert_eq!(reader.read_bits(11).expect("sentinel"), 0x5A5);

    let decoded = image.channels();
    assert_eq!(decoded.len(), channels.len());
    for (index, (want, got)) in channels.iter().zip(decoded).enumerate() {
        assert_eq!(got.width(), want.width, "channel {index} width");
        assert_eq!(got.height(), want.height, "channel {index} height");
        if want.width == 0 || want.height == 0 {
            continue;
        }
        assert_eq!(got.samples(), want.samples, "channel {index} samples");
    }
}

#[test]
fn three_lf_planes_round_trip_sample_exact_with_previous_channel_properties() {
    lf_planes_roundtrip(Some(true));
}

#[test]
fn three_lf_planes_round_trip_sample_exact() {
    lf_planes_roundtrip(None);
}

fn lf_planes_roundtrip(previous_channel_properties: Option<bool>) {
    let (width, height) = (97u32, 61u32);
    let y = textured(width, height, 0x1111, 300);
    // X and B follow Y loosely so the previous-channel property earns a split.
    let mut state = 0x2222u32;
    let x: Vec<i32> = y
        .iter()
        .map(|&v| v / 3 + (xorshift(&mut state) % 21) as i32 - 10)
        .collect();
    let b: Vec<i32> = y
        .iter()
        .map(|&v| -v / 5 + (xorshift(&mut state) % 9) as i32 - 4)
        .collect();
    roundtrip_with(
        &[
            OutChannel {
                width,
                height,
                samples: &y,
            },
            OutChannel {
                width,
                height,
                samples: &x,
            },
            OutChannel {
                width,
                height,
                samples: &b,
            },
        ],
        previous_channel_properties,
    );
}

#[test]
fn hf_metadata_channels_round_trip_sample_exact() {
    let (tiles_w, tiles_h) = (13u32, 9u32);
    let nb_blocks = 5000u32;
    let (grid_w, grid_h) = (100u32, 75u32);
    let mut state = 0x3333u32;
    let x_from_y: Vec<i32> = (0..tiles_w * tiles_h)
        .map(|_| (xorshift(&mut state) % 40) as i32 - 20)
        .collect();
    let b_from_y: Vec<i32> = (0..tiles_w * tiles_h)
        .map(|_| (xorshift(&mut state) % 60) as i32 - 30)
        .collect();
    let mut block_info: Vec<i32> = (0..nb_blocks)
        .map(|_| {
            let r = xorshift(&mut state) % 100;
            if r < 80 {
                0
            } else if r < 95 {
                1
            } else {
                2
            }
        })
        .collect();
    block_info.extend(std::iter::repeat_n(0i32, nb_blocks as usize));
    let sharpness = vec![0i32; (grid_w * grid_h) as usize];
    roundtrip(&[
        OutChannel {
            width: tiles_w,
            height: tiles_h,
            samples: &x_from_y,
        },
        OutChannel {
            width: tiles_w,
            height: tiles_h,
            samples: &b_from_y,
        },
        OutChannel {
            width: nb_blocks,
            height: 2,
            samples: &block_info,
        },
        OutChannel {
            width: grid_w,
            height: grid_h,
            samples: &sharpness,
        },
    ]);
}

#[test]
fn a_zero_size_channel_is_skipped_but_still_counted() {
    let (width, height) = (40u32, 30u32);
    let a = textured(width, height, 0x4444, 50);
    let c = textured(width, height, 0x5555, 5);
    roundtrip(&[
        OutChannel {
            width,
            height,
            samples: &a,
        },
        OutChannel {
            width: 0,
            height: 7,
            samples: &[],
        },
        OutChannel {
            width,
            height,
            samples: &c,
        },
    ]);
}

#[test]
fn a_large_noisy_plane_with_wide_residuals_round_trips() {
    // Residuals into the thousands exercise the hybrid-uint extra bits and the
    // subsampled learner.
    let (width, height) = (256u32, 200u32);
    let plane = textured(width, height, 0x6666, 3000);
    roundtrip(&[OutChannel {
        width,
        height,
        samples: &plane,
    }]);
}
