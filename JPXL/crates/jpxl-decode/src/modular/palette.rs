//! The (delta-)palette transform (18181-1 H.6.4).
//!
//! Forward, the transform replaces `num_c` equally-sized channels with a
//! single index channel plus a meta-channel of `nb_colours` palette entries
//! (`width = nb_colours`, `height = num_c`, both shifts `-1`) inserted at the
//! head of the channel list. Inverse, the index channel is copied `num_c`
//! times and each copy is mapped through the palette.
//!
//! An index is looked up in one of four ways, tested in this order:
//!
//! 1. `0 <= index < nb_colours` — a real palette entry, `channel[0](index, c)`.
//! 2. `index >= nb_colours` — an *implicit* palette entry, either a 2-bit
//!    per-component cube (`index < 64`) or a base-5 cube.
//! 3. `index < 0` and `c < 3` — an entry of the fixed 72-row delta table,
//!    signed by the parity of the folded index.
//! 4. otherwise — zero.
//!
//! Indices below `nb_deltas` are *deltas*: after the lookup, the value of
//! `prediction(x, y, d_pred)` for the channel being rebuilt is added. Note that
//! this test uses the raw index, so negative indices are always deltas.
//!
//! # Transcription notes
//!
//! * `kDeltaPalette[4]` reads `{0, -12, 9}` in `latex/part1.tex` and
//!   `{0, -12, 0}` in `part1.md`. The markdown is taken: entries 2, 3 and 4 of
//!   the table are the single-component deltas `{11,0,0}`, `{0,0,-13}`,
//!   `{0,-12,0}`, and a stray third component would break that pattern.
//!   Recorded as an OCR corruption of the LaTeX.
//! * `if (index & 1 == 0)` is written without parentheses, and 18181-1 Table 1
//!   puts `==` *above* `&` in precedence, which would make the condition
//!   `index & (1 == 0)` and therefore always false. That reading makes the
//!   whole ± structure of the table dead code, so `(index & 1) == 0` is taken.
//! * The implicit-palette formulas use `/ 4`, which 4.3 defines as exact real
//!   division, while every sample in Annex H is an integer. Integer division
//!   is taken; both operands are non-negative so truncation and flooring agree.

use jpxl_core::limits::AllocGuard;

use super::channel::{BYTES_PER_SAMPLE, Channel};
use super::error::{Result, malformed};
use super::predictor::{Neighbours, Predictor};
use super::weighted::{WeightedState, WpHeader, narrow_to_i32};

/// The number of rows in [`DELTA_PALETTE`] (H.6.4: `kDeltaPalette[72][3]`).
pub const DELTA_PALETTE_ROWS: usize = 72;

/// The modulus applied to a folded negative index before the table lookup.
///
/// 143 = 2 * 72 - 1: row 0 is used by one index and every later row by two,
/// one for each sign.
pub const DELTA_PALETTE_MODULUS: i64 = 143;

/// `kDeltaPalette[72][3]` of H.6.4.
pub const DELTA_PALETTE: [[i32; 3]; DELTA_PALETTE_ROWS] = [
    [0, 0, 0],
    [4, 4, 4],
    [11, 0, 0],
    [0, 0, -13],
    [0, -12, 0],
    [-10, -10, -10],
    [-18, -18, -18],
    [-27, -27, -27],
    [-18, -18, 0],
    [0, 0, -32],
    [-32, 0, 0],
    [-37, -37, -37],
    [0, -32, -32],
    [24, 24, 45],
    [50, 50, 50],
    [-45, -24, -24],
    [-24, -45, -45],
    [0, -24, -24],
    [-34, -34, 0],
    [-24, 0, -24],
    [-45, -45, -24],
    [64, 64, 64],
    [-32, 0, -32],
    [0, -32, 0],
    [-32, 0, 32],
    [-24, -45, -24],
    [45, 24, 45],
    [24, -24, -45],
    [-45, -24, 24],
    [80, 80, 80],
    [64, 0, 0],
    [0, 0, -64],
    [0, -64, -64],
    [-24, -24, 45],
    [96, 96, 96],
    [64, 64, 0],
    [45, -24, -24],
    [34, -34, 0],
    [112, 112, 112],
    [24, -45, -45],
    [45, 45, -24],
    [0, -32, 32],
    [24, -24, 45],
    [0, 96, 96],
    [45, -24, 24],
    [24, -45, -24],
    [-24, -45, 24],
    [0, -64, 0],
    [96, 0, 0],
    [128, 128, 128],
    [64, 0, 64],
    [144, 144, 144],
    [96, 96, 0],
    [-36, -36, 36],
    [45, -24, -45],
    [45, -45, -24],
    [0, 0, -96],
    [0, 128, 128],
    [0, 96, 0],
    [45, 24, -45],
    [-128, 0, 0],
    [24, -45, 24],
    [-45, 24, -45],
    [64, 0, -64],
    [64, -64, -64],
    [96, 0, 96],
    [45, -45, 24],
    [24, 45, -45],
    [64, 64, -64],
    [128, 128, 0],
    [0, 0, -128],
    [-24, 45, -45],
];

/// Parameters of one palette transform (Table H.7 rows for `tr == kPalette`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PaletteParams {
    /// First affected channel in the *pre-transform* channel list.
    pub begin_c: u32,
    /// Number of channels the palette covers.
    pub num_c: u32,
    /// Number of explicit palette entries.
    pub nb_colours: u32,
    /// Indices below this are deltas rather than absolute colours.
    pub nb_deltas: u32,
    /// Predictor number used for delta reconstruction.
    pub d_pred: u32,
}

/// Everything the palette inverse needs beyond the channel list.
#[derive(Debug, Clone, Copy)]
pub struct PaletteContext {
    /// `metadata.bit_depth.bits_per_sample` (D.3.5), supplied by the caller.
    pub bits_per_sample: u32,
    /// The sub-bitstream's `wp_params`, needed only when `d_pred == 6`.
    pub wp_header: WpHeader,
}

/// Resolves one palette index for colour component `c` (H.6.4).
///
/// Returns the value *before* the delta prediction is added.
///
/// # Errors
///
/// [`ModularError::Malformed`](super::ModularError::Malformed) if
/// `bits_per_sample` is too large for the shifts the clause performs.
pub fn lookup(
    palette: &Channel,
    params: &PaletteParams,
    bits_per_sample: u32,
    c: u32,
    index: i32,
) -> Result<i64> {
    if bits_per_sample == 0 || bits_per_sample > 32 {
        return Err(malformed!(
            "H.6.4: bits_per_sample = {bits_per_sample} is outside 1..=32"
        ));
    }
    let bitdepth = i64::from(bits_per_sample);
    let full_scale = (1i64 << bitdepth) - 1;
    let index = i64::from(index);
    let nb_colours = i64::from(params.nb_colours);

    if index >= 0 && index < nb_colours {
        // A real entry: the meta-channel is nb_colours wide and num_c tall.
        let x = u32::try_from(index).map_err(|_| malformed!("H.6.4: palette index overflows"))?;
        return Ok(i64::from(palette.get(x, c)));
    }

    if index >= nb_colours {
        let index = index - nb_colours;
        if index < 64 {
            // 2 bits per component, plus half a step so the cube is centred.
            let shift = 2u32.saturating_mul(c);
            let bucket = if shift >= 63 { 0 } else { (index >> shift) & 3 };
            let bias = 1i64 << (bitdepth - 3).max(0);
            return Ok(bucket * full_scale / 4 + bias);
        }
        // Base-5 cube: divide out the lower components' digits.
        let mut index = index - 64;
        for _ in 0..c {
            if index == 0 {
                break;
            }
            index /= 5;
        }
        return Ok((index % 5) * full_scale / 4);
    }

    // index < 0
    if c >= 3 {
        return Ok(0);
    }
    // (-index - 1) Umod 143, taken in i64 so index == i32::MIN is safe.
    let folded = (-index - 1).rem_euclid(DELTA_PALETTE_MODULUS);
    let row = ((folded + 1) >> 1) as usize;
    let mut value = i64::from(
        DELTA_PALETTE
            .get(row)
            .and_then(|entry| entry.get(c as usize))
            .copied()
            .unwrap_or(0),
    );
    if (folded & 1) == 0 {
        value = -value;
    }
    if bitdepth > 8 {
        value <<= bitdepth.min(24) - 8;
    }
    Ok(value)
}

/// Applies the inverse palette transform to `channels` (H.6.4).
///
/// `channels` must be the post-transform list, i.e. with the palette
/// meta-channel at index 0 and the index channel at `begin_c + 1`.
///
/// # Errors
///
/// [`ModularError::Malformed`](super::ModularError::Malformed) for an
/// inconsistent channel list or an out-of-range `d_pred`, and
/// [`ModularError::Core`](super::ModularError::Core) if expanding the index
/// channel exceeds the guard.
pub fn apply_inverse(
    channels: &mut Vec<Channel>,
    params: &PaletteParams,
    ctx: &PaletteContext,
    guard: &mut AllocGuard,
) -> Result<()> {
    let num_c = params.num_c as usize;
    if num_c == 0 {
        return Err(malformed!("H.6.4: palette with num_c = 0"));
    }
    if channels.is_empty() {
        return Err(malformed!("H.6.4: no palette meta-channel to invert"));
    }
    let d_pred = Predictor::from_value(params.d_pred)?;

    // Remove the meta-channel first. Every index below is then relative to the
    // pre-transform list, which is what `begin_c` is expressed in.
    let palette = channels.remove(0);
    if palette.height() < params.num_c {
        return Err(malformed!(
            "H.6.4: palette meta-channel is {} tall but num_c is {}",
            palette.height(),
            params.num_c
        ));
    }
    let first = params.begin_c as usize;
    if first >= channels.len() {
        return Err(malformed!(
            "H.6.4: begin_c = {first} but only {} channels remain",
            channels.len()
        ));
    }

    // "for (i = first + 1; i <= last; i++) insert a copy of channel[first]".
    let index_channel = channels
        .get(first)
        .ok_or_else(|| malformed!("H.6.4: missing index channel"))?
        .clone();
    let spec = index_channel.spec();
    for offset in 1..num_c {
        guard.charge(spec.sample_count().saturating_mul(BYTES_PER_SAMPLE))?;
        let at = first
            .checked_add(offset)
            .ok_or_else(|| malformed!("H.6.4: channel index overflows"))?;
        if at > channels.len() {
            return Err(malformed!("H.6.4: cannot insert palette channel at {at}"));
        }
        channels.insert(at, index_channel.clone());
    }

    let (width, height) = (spec.width, spec.height);
    for c in 0..num_c {
        let Some(target) = channels.get_mut(first + c) else {
            return Err(malformed!("H.6.4: palette channel {} vanished", first + c));
        };
        let mut wp = if d_pred.uses_weighted() {
            Some(WeightedState::new(width, guard)?)
        } else {
            None
        };
        let c_u32 = u32::try_from(c).map_err(|_| malformed!("H.6.4: num_c overflows"))?;

        for y in 0..height {
            for x in 0..width {
                let index = target.get(x, y);
                let is_delta = i64::from(index) < i64::from(params.nb_deltas);
                let mut value = lookup(&palette, params, ctx.bits_per_sample, c_u32, index)?;

                // The prediction is taken over the channel as reconstructed so
                // far, in raster order, so W/N/NW are already final.
                let nb = Neighbours::gather(target, x, y);
                let weighted = wp
                    .as_ref()
                    .map(|state| state.predict(&ctx.wp_header, &nb.into(), x));
                if is_delta {
                    value += nb.predict(d_pred, weighted.map_or(0, |w| w.prediction));
                }
                let value = narrow_to_i32(value);
                target.set(x, y, value);
                if let (Some(state), Some(w)) = (wp.as_mut(), weighted.as_ref()) {
                    state.update(x, w, value);
                }
            }
            if let Some(state) = wp.as_mut() {
                state.advance_row();
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    reason = "hand-written spec vectors read better with direct indexing; a panic \
              in a test is a failing test"
)]
mod tests {
    use jpxl_core::limits::Limits;

    use super::super::channel::ChannelSpec;
    use super::*;

    fn guard() -> AllocGuard {
        AllocGuard::new(&Limits::relaxed())
    }

    fn params() -> PaletteParams {
        PaletteParams {
            begin_c: 0,
            num_c: 3,
            nb_colours: 2,
            nb_deltas: 0,
            d_pred: 0,
        }
    }

    /// A 2-colour, 3-component palette: colour 0 is (10, 20, 30), colour 1 is
    /// (40, 50, 60). The meta-channel is `nb_colours` wide and `num_c` tall.
    fn palette_channel() -> Channel {
        Channel::from_samples(
            ChannelSpec::with_shifts(2, 3, -1, -1),
            vec![10, 40, 20, 50, 30, 60],
        )
        .expect("2x3 palette")
    }

    #[test]
    fn delta_table_has_seventy_two_rows_of_three() {
        assert_eq!(DELTA_PALETTE.len(), DELTA_PALETTE_ROWS);
        assert_eq!(DELTA_PALETTE[0], [0, 0, 0], "row 0 is the zero delta");
        assert_eq!(DELTA_PALETTE[71], [-24, 45, -45], "last row");
        // The three single-component rows that decide the disputed row 4.
        assert_eq!(DELTA_PALETTE[2], [11, 0, 0]);
        assert_eq!(DELTA_PALETTE[3], [0, 0, -13]);
        assert_eq!(
            DELTA_PALETTE[4],
            [0, -12, 0],
            "part1.md reading, not the LaTeX"
        );
    }

    #[test]
    fn in_range_indices_read_the_meta_channel() {
        let p = palette_channel();
        let params = params();
        for (c, expected) in [(0u32, 10i64), (1, 20), (2, 30)] {
            assert_eq!(lookup(&p, &params, 8, c, 0).expect("colour 0"), expected);
        }
        for (c, expected) in [(0u32, 40i64), (1, 50), (2, 60)] {
            assert_eq!(lookup(&p, &params, 8, c, 1).expect("colour 1"), expected);
        }
    }

    #[test]
    fn implicit_cube_entries_are_hand_computed() {
        let p = palette_channel();
        let params = params();
        // index = nb_colours + 0 = 2 -> index' = 0, below 64.
        //   bucket = (0 >> 0) Umod 4 = 0
        //   value  = 0 * 255 / 4 + (1 << (8 - 3)) = 0 + 32 = 32
        assert_eq!(lookup(&p, &params, 8, 0, 2).expect("implicit"), 32);
        // index' = 1, c = 0: bucket = 1 -> 1 * 255 Idiv 4 + 32 = 63 + 32 = 95
        assert_eq!(lookup(&p, &params, 8, 0, 3).expect("implicit"), 95);
        // index' = 1, c = 1: bucket = (1 >> 2) Umod 4 = 0 -> 32
        assert_eq!(lookup(&p, &params, 8, 1, 3).expect("implicit"), 32);
        // index' = 4, c = 1: bucket = (4 >> 2) Umod 4 = 1 -> 95
        assert_eq!(lookup(&p, &params, 8, 1, 6).expect("implicit"), 95);
    }

    #[test]
    fn base_five_entries_divide_out_the_lower_components() {
        let p = palette_channel();
        let params = params();
        // index' = 64 + 7 -> after the -64 it is 7.
        //   c = 0: 7 Umod 5 = 2 -> 2 * 255 Idiv 4 = 127
        //   c = 1: 7 Idiv 5 = 1, 1 Umod 5 = 1 -> 255 Idiv 4 = 63
        //   c = 2: 7 Idiv 5 Idiv 5 = 0 -> 0
        let index = 2 + 64 + 7;
        assert_eq!(lookup(&p, &params, 8, 0, index).expect("base 5"), 127);
        assert_eq!(lookup(&p, &params, 8, 1, index).expect("base 5"), 63);
        assert_eq!(lookup(&p, &params, 8, 2, index).expect("base 5"), 0);
    }

    #[test]
    fn negative_indices_use_the_delta_table_with_alternating_signs() {
        let p = palette_channel();
        let params = params();
        // index = -1 -> folded = 0, row = (0 + 1) >> 1 = 0 -> {0,0,0};
        // folded is even so the value is negated, which is still 0.
        assert_eq!(lookup(&p, &params, 8, 0, -1).expect("delta"), 0);
        // index = -2 -> folded = 1, row = 1 -> {4,4,4}; folded is odd, keep +.
        assert_eq!(lookup(&p, &params, 8, 0, -2).expect("delta"), 4);
        // index = -3 -> folded = 2, row = (2 + 1) >> 1 = 1 -> {4,4,4};
        // folded is even, so negate.
        assert_eq!(lookup(&p, &params, 8, 0, -3).expect("delta"), -4);
        // index = -4 -> folded = 3, row = 2 -> {11, 0, 0}, odd -> +11.
        assert_eq!(lookup(&p, &params, 8, 0, -4).expect("delta"), 11);
        assert_eq!(lookup(&p, &params, 8, 1, -4).expect("delta"), 0);
    }

    #[test]
    fn delta_table_is_scaled_up_above_eight_bit() {
        let p = palette_channel();
        let params = params();
        // bitdepth 12 -> value <<= min(12, 24) - 8 = 4.
        assert_eq!(lookup(&p, &params, 12, 0, -2).expect("delta"), 4 << 4);
        // bitdepth 32 -> the shift saturates at 24 - 8 = 16.
        assert_eq!(lookup(&p, &params, 32, 0, -2).expect("delta"), 4 << 16);
        // bitdepth 8 -> no shift at all.
        assert_eq!(lookup(&p, &params, 8, 0, -2).expect("delta"), 4);
    }

    #[test]
    fn the_folding_modulus_wraps_at_143() {
        let p = palette_channel();
        let params = params();
        // index = -144 -> -index - 1 = 143 -> folded = 0, same as index = -1.
        assert_eq!(
            lookup(&p, &params, 8, 0, -144).expect("wrapped"),
            lookup(&p, &params, 8, 0, -1).expect("unwrapped")
        );
        // i32::MIN must not overflow.
        assert!(lookup(&p, &params, 8, 0, i32::MIN).is_ok());
    }

    #[test]
    fn components_past_the_third_are_zero_for_negative_indices() {
        let p = palette_channel();
        let params = params();
        assert_eq!(lookup(&p, &params, 8, 3, -4).expect("c >= 3"), 0);
        assert_eq!(lookup(&p, &params, 8, 99, -4).expect("c >= 3"), 0);
    }

    #[test]
    fn absurd_bit_depths_are_rejected() {
        let p = palette_channel();
        let params = params();
        assert!(lookup(&p, &params, 0, 0, 0).is_err());
        assert!(lookup(&p, &params, 33, 0, 0).is_err());
    }

    #[test]
    fn inverse_expands_one_index_channel_into_num_c_channels() {
        // Post-transform list: [palette, indices]. begin_c = 0, num_c = 3.
        let indices = Channel::from_samples(ChannelSpec::new(2, 2), vec![0, 1, 1, 0]).expect("2x2");
        let mut channels = vec![palette_channel(), indices];
        let ctx = PaletteContext {
            bits_per_sample: 8,
            wp_header: WpHeader::default_wp(),
        };
        apply_inverse(&mut channels, &params(), &ctx, &mut guard()).expect("inverse");

        assert_eq!(
            channels.len(),
            3,
            "the meta-channel is gone, 3 colours added"
        );
        // Colour 0 is (10, 20, 30) and colour 1 is (40, 50, 60).
        assert_eq!(channels[0].samples(), &[10, 40, 40, 10]);
        assert_eq!(channels[1].samples(), &[20, 50, 50, 20]);
        assert_eq!(channels[2].samples(), &[30, 60, 60, 30]);
    }

    #[test]
    fn delta_indices_add_the_predictor_output() {
        // nb_deltas = 2, so indices 0 and 1 are both deltas. d_pred = 1 (West).
        let p = PaletteParams {
            begin_c: 0,
            num_c: 1,
            nb_colours: 2,
            nb_deltas: 2,
            d_pred: 1,
        };
        // A 1x3 palette (one colour, one component) holding the value 5.
        let palette =
            Channel::from_samples(ChannelSpec::with_shifts(2, 1, -1, -1), vec![5, 7]).expect("2x1");
        let indices = Channel::from_samples(ChannelSpec::new(3, 1), vec![0, 1, 0]).expect("3x1");
        let mut channels = vec![palette, indices];
        let ctx = PaletteContext {
            bits_per_sample: 8,
            wp_header: WpHeader::default_wp(),
        };
        apply_inverse(&mut channels, &p, &ctx, &mut guard()).expect("inverse");
        // x = 0: value = 5, W = 0 (origin) -> 5.
        // x = 1: value = 7, W = 5 -> 12.
        // x = 2: value = 5, W = 12 -> 17.
        assert_eq!(channels[0].samples(), &[5, 12, 17]);
    }

    #[test]
    fn a_self_correcting_delta_predictor_is_accepted() {
        // d_pred == 6 needs the H.5 state; this proves the path runs rather
        // than rejecting, and that the first sample uses a zero prediction.
        let p = PaletteParams {
            begin_c: 0,
            num_c: 1,
            nb_colours: 1,
            nb_deltas: 1,
            d_pred: 6,
        };
        let palette =
            Channel::from_samples(ChannelSpec::with_shifts(1, 1, -1, -1), vec![3]).expect("1x1");
        let indices = Channel::from_samples(ChannelSpec::new(2, 2), vec![0; 4]).expect("2x2");
        let mut channels = vec![palette, indices];
        let ctx = PaletteContext {
            bits_per_sample: 8,
            wp_header: WpHeader::default_wp(),
        };
        apply_inverse(&mut channels, &p, &ctx, &mut guard()).expect("inverse with d_pred 6");
        assert_eq!(channels[0].get(0, 0), 3, "no prediction at the origin");
    }

    #[test]
    fn a_bad_d_pred_is_rejected() {
        let p = PaletteParams {
            begin_c: 0,
            num_c: 1,
            nb_colours: 1,
            nb_deltas: 0,
            d_pred: 14,
        };
        let palette =
            Channel::from_samples(ChannelSpec::with_shifts(1, 1, -1, -1), vec![3]).expect("1x1");
        let indices = Channel::from_samples(ChannelSpec::new(1, 1), vec![0]).expect("1x1");
        let mut channels = vec![palette, indices];
        let ctx = PaletteContext {
            bits_per_sample: 8,
            wp_header: WpHeader::default_wp(),
        };
        assert!(apply_inverse(&mut channels, &p, &ctx, &mut guard()).is_err());
    }
}
