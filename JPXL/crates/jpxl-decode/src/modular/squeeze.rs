//! The squeeze transform: a modified Haar wavelet (18181-1 H.6.2).
//!
//! ```text
//! Table H.9 — SqueezeParams bundle
//! condition  type                                          name
//!            Bool()                                        horizontal
//!            Bool()                                        in_place
//!            U32(u(3), 8 + u(6), 72 + u(10), 1096 + u(13)) begin_c
//!            U32(1, 2, 3, 4 + u(4))                        num_c
//! ```
//!
//! A squeeze step replaces `num_c` channels starting at `begin_c` with their
//! low-pass halves and inserts the corresponding residual channels either
//! immediately after (`in_place`) or at the end of the channel list. Forward
//! steps are applied in signalled order to derive the channel *shapes*;
//! inverse steps run in reverse order once the sample data is decoded.
//!
//! # The tendency function
//!
//! The inverse is not a plain Haar synthesis: the residual is offset by
//! `tendency(left, avg, next_avg)`, a monotonicity-preserving correction that
//! makes a smooth low-pass signal reconstruct to a smooth full-resolution one.
//!
//! ```text
//! tendency(A, B, C) {
//!   if (A >= B and B >= C) {
//!     X = (4*A - 3*C - B + 6) Idiv 12;
//!     if (X - (X & 1) > 2 * (A - B)) X = 2 * (A - B) + 1;
//!     if (X + (X & 1) > 2 * (B - C)) X = 2 * (B - C);
//!     return X;
//!   } else if (A <= B and B <= C) {
//!     X = (4*A - 3*C - B - 6) Idiv 12;
//!     if (X + (X & 1) < 2 * (A - B)) X = 2 * (A - B) - 1;
//!     if (X - (X & 1) < 2 * (B - C)) X = 2 * (B - C);
//!     return X;
//!   } else return 0;
//! }
//! ```
//!
//! Three details decide whether a transcription of this is right:
//!
//! * `Idiv` truncates towards zero (4.3), so `-50 Idiv 12` is `-4`, not `-5`.
//! * `X & 1` is a two's-complement bitwise AND (4.4), so it is `1` for every
//!   odd `X` including negative ones — `-3 & 1 == 1`.
//! * the two branches are not mirror images: the `+ 6`/`- 6` and the `+ 1`/
//!   `- 1` both flip, and the second guard drops the `± 1` in both branches.
//!
//! `tendency` returns 0 whenever the three samples are not monotonic, which is
//! what keeps the transform reversible on noisy data.

use jpxl_bitstream::{BitReader, U32Dist, U32Spec, read_bool, read_u32, trace_field};
use jpxl_core::limits::AllocGuard;

use super::channel::{Channel, ChannelSpec};
use super::error::{Result, malformed};

/// H.6.2.1: `U32(u(3), 8 + u(6), 72 + u(10), 1096 + u(13))`, also used for
/// `TransformInfo.begin_c`.
pub(super) const BEGIN_C_SPEC: U32Spec = U32Spec::new([
    U32Dist::BitsOffset { bits: 3, offset: 0 },
    U32Dist::BitsOffset { bits: 6, offset: 8 },
    U32Dist::BitsOffset {
        bits: 10,
        offset: 72,
    },
    U32Dist::BitsOffset {
        bits: 13,
        offset: 1096,
    },
]);

/// H.6.2.1: `U32(1, 2, 3, 4 + u(4))`.
const NUM_C_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(1),
    U32Dist::Val(2),
    U32Dist::Val(3),
    U32Dist::BitsOffset { bits: 4, offset: 4 },
]);

/// One squeeze step (18181-1 Table H.9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SqueezeParams {
    /// Squeeze horizontally rather than vertically.
    pub horizontal: bool,
    /// Insert the residual channels right after the squeezed ones rather than
    /// at the end of the channel list.
    pub in_place: bool,
    /// Index of the first affected channel.
    pub begin_c: u32,
    /// Number of affected channels.
    pub num_c: u32,
}

impl SqueezeParams {
    /// Reads one `SqueezeParams` bundle (Table H.9).
    ///
    /// # Errors
    ///
    /// [`ModularError::Bitstream`](super::ModularError::Bitstream) at end of
    /// input.
    pub fn read(reader: &mut BitReader<'_>) -> Result<Self> {
        let horizontal = trace_field!(reader, "squeeze.horizontal", read_bool(reader))?;
        let in_place = trace_field!(reader, "squeeze.in_place", read_bool(reader))?;
        let begin_c = trace_field!(reader, "squeeze.begin_c", read_u32(reader, &BEGIN_C_SPEC))?;
        let num_c = trace_field!(reader, "squeeze.num_c", read_u32(reader, &NUM_C_SPEC))?;
        Ok(Self {
            horizontal,
            in_place,
            begin_c,
            num_c,
        })
    }

    /// Last affected channel index, `begin_c + num_c - 1`.
    ///
    /// # Errors
    ///
    /// [`ModularError::Malformed`](super::ModularError::Malformed) if `num_c`
    /// is zero or the sum overflows.
    pub fn end_c(&self) -> Result<u32> {
        if self.num_c == 0 {
            return Err(malformed!("H.6.2.1: squeeze step with num_c = 0"));
        }
        self.begin_c
            .checked_add(self.num_c - 1)
            .ok_or_else(|| malformed!("H.6.2.1: begin_c + num_c overflows"))
    }
}

/// `tendency(A, B, C)` of H.6.2.2.
///
/// See the module documentation for the three transcription hazards.
#[must_use]
pub fn tendency(a: i64, b: i64, c: i64) -> i64 {
    if a >= b && b >= c {
        // Descending run.
        let mut x = (4 * a - 3 * c - b + 6) / 12;
        if x - (x & 1) > 2 * (a - b) {
            x = 2 * (a - b) + 1;
        }
        if x + (x & 1) > 2 * (b - c) {
            x = 2 * (b - c);
        }
        x
    } else if a <= b && b <= c {
        // Ascending run.
        let mut x = (4 * a - 3 * c - b - 6) / 12;
        if x + (x & 1) < 2 * (a - b) {
            x = 2 * (a - b) - 1;
        }
        if x - (x & 1) < 2 * (b - c) {
            x = 2 * (b - c);
        }
        x
    } else {
        0
    }
}

/// `horiz_isqueeze(input_1, input_2, output)` of H.6.2.2.
///
/// `input_1` is the low-pass channel of width `w1`, `input_2` the residual
/// channel of width `w2`, and `w1` is either `w2` or `w2 + 1`. The result is
/// `w1 + w2` wide.
///
/// # Errors
///
/// [`ModularError::Malformed`](super::ModularError::Malformed) if the two
/// inputs do not have equal heights or compatible widths, and
/// [`ModularError::Core`](super::ModularError::Core) if the output allocation
/// exceeds the guard.
pub fn horiz_isqueeze(
    input_1: &Channel,
    input_2: &Channel,
    guard: &mut AllocGuard,
) -> Result<Channel> {
    let (w1, w2) = (input_1.width(), input_2.width());
    let h = input_1.height();
    if input_2.height() != h {
        return Err(malformed!(
            "H.6.2.2: horizontal inverse squeeze of {w1}x{h} and {w2}x{} — heights differ",
            input_2.height()
        ));
    }
    if w1 != w2 && w1 != w2 + 1 {
        return Err(malformed!(
            "H.6.2.2: horizontal inverse squeeze needs w1 == w2 or w1 == w2 + 1, got {w1} and {w2}"
        ));
    }
    let out_width = w1
        .checked_add(w2)
        .ok_or_else(|| malformed!("H.6.2.2: output width overflows"))?;
    let mut spec = input_1.spec();
    spec.width = out_width;
    // The forward step incremented hshift; undo it so the reconstructed channel
    // describes its own geometry. H.6.2's inverse pseudocode omits this, which
    // is an omission rather than a rule: a full-resolution channel carrying a
    // subsampled shift would mislead every later stage.
    if spec.hshift > 0 {
        spec.hshift -= 1;
    }
    let mut output = Channel::new(spec, guard)?;

    for y in 0..h {
        for x in 0..w2 {
            let avg = i64::from(input_1.get(x, y));
            let residu = i64::from(input_2.get(x, y));
            let next_avg = if x + 1 < w1 {
                i64::from(input_1.get(x + 1, y))
            } else {
                avg
            };
            let left = if x > 0 {
                i64::from(output.get((x << 1) - 1, y))
            } else {
                avg
            };
            let diff = residu + tendency(left, avg, next_avg);
            let first = avg + diff / 2;
            output.set(2 * x, y, narrow(first));
            output.set(2 * x + 1, y, narrow(first - diff));
        }
        if w1 > w2 {
            output.set(2 * w2, y, input_1.get(w2, y));
        }
    }
    Ok(output)
}

/// `vert_isqueeze(input_1, input_2, output)` of H.6.2.3.
///
/// The vertical twin of [`horiz_isqueeze`]: `input_1` is `w x h1`, `input_2` is
/// `w x h2` with `h1 == h2` or `h1 == h2 + 1`, and the output is `w x (h1+h2)`.
///
/// # Errors
///
/// As [`horiz_isqueeze`], with widths and heights exchanged.
pub fn vert_isqueeze(
    input_1: &Channel,
    input_2: &Channel,
    guard: &mut AllocGuard,
) -> Result<Channel> {
    let (h1, h2) = (input_1.height(), input_2.height());
    let w = input_1.width();
    if input_2.width() != w {
        return Err(malformed!(
            "H.6.2.3: vertical inverse squeeze of {w}x{h1} and {}x{h2} — widths differ",
            input_2.width()
        ));
    }
    if h1 != h2 && h1 != h2 + 1 {
        return Err(malformed!(
            "H.6.2.3: vertical inverse squeeze needs h1 == h2 or h1 == h2 + 1, got {h1} and {h2}"
        ));
    }
    let out_height = h1
        .checked_add(h2)
        .ok_or_else(|| malformed!("H.6.2.3: output height overflows"))?;
    let mut spec = input_1.spec();
    spec.height = out_height;
    if spec.vshift > 0 {
        spec.vshift -= 1;
    }
    let mut output = Channel::new(spec, guard)?;

    for y in 0..h2 {
        for x in 0..w {
            let avg = i64::from(input_1.get(x, y));
            let residu = i64::from(input_2.get(x, y));
            let next_avg = if y + 1 < h1 {
                i64::from(input_1.get(x, y + 1))
            } else {
                avg
            };
            let top = if y > 0 {
                i64::from(output.get(x, (y << 1) - 1))
            } else {
                avg
            };
            let diff = residu + tendency(top, avg, next_avg);
            let first = avg + diff / 2;
            output.set(x, 2 * y, narrow(first));
            output.set(x, 2 * y + 1, narrow(first - diff));
        }
    }
    if h1 > h2 {
        for x in 0..w {
            output.set(x, 2 * h2, input_1.get(x, h2));
        }
    }
    Ok(output)
}

/// H.1: modular samples are 32-bit; the transform results are too.
fn narrow(v: i64) -> i32 {
    super::weighted::narrow_to_i32(v)
}

/// Derives the default squeeze step list when `sp` is empty (H.6.2.1).
///
/// ```text
/// first = nb_meta_channels; count = channel.size() - first;
/// w = channel[first].width; h = channel[first].height;
/// if (count > 2 and channel[first+1].width == w and channel[first+1].height == h) {
///   param.begin_c = first + 1; param.num_c = 2; param.in_place = false;
///   param.horizontal = true;  sp.push_back(param);
///   param.horizontal = false; sp.push_back(param);
/// }
/// param.begin_c = first; param.num_c = count; param.in_place = true;
/// if (h >= w and h > 8) { param.horizontal = false; sp.push_back(param); h = (h+1) Idiv 2; }
/// while (w > 8 or h > 8) {
///   if (w > 8) { param.horizontal = true;  sp.push_back(param); w = (w+1) Idiv 2; }
///   if (h > 8) { param.horizontal = false; sp.push_back(param); h = (h+1) Idiv 2; }
/// }
/// ```
///
/// Note that `param` is a single reused record: the first two steps carry
/// `begin_c = first + 1, num_c = 2, in_place = false` and every later step
/// carries `begin_c = first, num_c = count, in_place = true`.
///
/// # Errors
///
/// [`ModularError::Malformed`](super::ModularError::Malformed) if there are no
/// non-meta channels to squeeze.
pub fn default_params(channels: &[ChannelSpec], nb_meta_channels: usize) -> Result<Vec<SqueezeParams>> {
    let first = nb_meta_channels;
    let Some(base) = channels.get(first) else {
        return Err(malformed!(
            "H.6.2.1: default squeeze parameters need at least one non-meta channel"
        ));
    };
    let count = channels.len() - first;
    let mut sp = Vec::new();
    let first_u32 = u32::try_from(first)
        .map_err(|_| malformed!("H.6.2.1: nb_meta_channels does not fit in a u32"))?;
    let count_u32 = u32::try_from(count)
        .map_err(|_| malformed!("H.6.2.1: channel count does not fit in a u32"))?;

    let (mut w, mut h) = (base.width, base.height);
    if count > 2
        && channels
            .get(first + 1)
            .is_some_and(|c| c.width == w && c.height == h)
    {
        for horizontal in [true, false] {
            sp.push(SqueezeParams {
                horizontal,
                in_place: false,
                begin_c: first_u32 + 1,
                num_c: 2,
            });
        }
    }

    let mut push = |sp: &mut Vec<SqueezeParams>, horizontal: bool| {
        sp.push(SqueezeParams {
            horizontal,
            in_place: true,
            begin_c: first_u32,
            num_c: count_u32,
        });
    };

    if h >= w && h > 8 {
        push(&mut sp, false);
        h = h.div_ceil(2);
    }
    while w > 8 || h > 8 {
        if w > 8 {
            push(&mut sp, true);
            w = w.div_ceil(2);
        }
        if h > 8 {
            push(&mut sp, false);
            h = h.div_ceil(2);
        }
    }
    Ok(sp)
}

#[cfg(test)]
mod tests {
    use jpxl_core::limits::Limits;

    use super::*;

    fn guard() -> AllocGuard {
        AllocGuard::new(&Limits::relaxed())
    }

    #[test]
    fn tendency_is_zero_off_a_monotonic_run() {
        // A < B but B > C: neither branch applies.
        assert_eq!(tendency(1, 5, 2), 0);
        // A > B but B < C.
        assert_eq!(tendency(5, 1, 3), 0);
    }

    #[test]
    fn tendency_ascending_hand_derived() {
        // tendency(10, 10, 20): A <= B <= C, so the ascending branch.
        //   X = (4*10 - 3*20 - 10 - 6) Idiv 12 = -36 Idiv 12 = -3
        //   X & 1 = 1 (two's complement), X + 1 = -2, 2*(A-B) = 0
        //     -2 < 0 -> X = 2*(A-B) - 1 = -1
        //   X & 1 = 1, X - 1 = -2, 2*(B-C) = -20; -2 < -20 is false
        //   -> -1
        assert_eq!(tendency(10, 10, 20), -1);

        // tendency(9, 20, 20):
        //   X = (36 - 60 - 20 - 6) Idiv 12 = -50 Idiv 12 = -4 (towards zero!)
        //   X & 1 = 0, X + 0 = -4, 2*(A-B) = -22; -4 < -22 false
        //   X - 0 = -4, 2*(B-C) = 0; -4 < 0 -> X = 0
        //   -> 0
        assert_eq!(tendency(9, 20, 20), 0);
    }

    #[test]
    fn tendency_descending_hand_derived() {
        // tendency(20, 10, 10): A >= B >= C, descending branch.
        //   X = (4*20 - 3*10 - 10 + 6) Idiv 12 = (80 - 30 - 10 + 6)/12
        //     = 46 Idiv 12 = 3
        //   X & 1 = 1, X - 1 = 2, 2*(A-B) = 20; 2 > 20 false
        //   X + 1 = 4, 2*(B-C) = 0; 4 > 0 -> X = 0
        //   -> 0
        assert_eq!(tendency(20, 10, 10), 0);

        // tendency(20, 20, 10): descending.
        //   X = (80 - 30 - 20 + 6) Idiv 12 = 36 Idiv 12 = 3
        //   X & 1 = 1, X - 1 = 2, 2*(A-B) = 0; 2 > 0 -> X = 2*(A-B) + 1 = 1
        //   X = 1, X & 1 = 1, X + 1 = 2, 2*(B-C) = 20; 2 > 20 false
        //   -> 1
        assert_eq!(tendency(20, 20, 10), 1);
    }

    #[test]
    fn tendency_negates_under_reflection_of_a_flat_signal() {
        // For a constant run every branch collapses: A == B == C hits the
        // descending branch (both >= hold), X = (4v - 3v - v + 6)/12 = 0, and
        // both guards compare against 0.
        for v in [-100i64, 0, 7, 1000] {
            assert_eq!(tendency(v, v, v), 0, "flat at {v}");
        }
    }

    /// The 4x1 worked example promised by the brief, derived by hand above the
    /// assertions and re-derived in the test body.
    #[test]
    fn horizontal_inverse_squeeze_4x1_worked_example() {
        // input_1 (low-pass) = [10, 20], input_2 (residual) = [2, -4].
        //
        // x = 0: avg = 10, residu = 2, next_avg = input_1(1) = 20,
        //        left = avg = 10 (x == 0).
        //        tendency(10, 10, 20) = -1  (see the ascending test)
        //        diff  = 2 + (-1) = 1
        //        first = 10 + (1 Idiv 2) = 10 + 0 = 10
        //        output(0) = 10, output(1) = 10 - 1 = 9
        //
        // x = 1: avg = 20, residu = -4, next_avg = avg = 20 (x+1 == w1),
        //        left = output(1) = 9.
        //        tendency(9, 20, 20) = 0    (see the ascending test)
        //        diff  = -4 + 0 = -4
        //        first = 20 + (-4 Idiv 2) = 20 - 2 = 18
        //        output(2) = 18, output(3) = 18 - (-4) = 22
        let lo = Channel::from_samples(ChannelSpec::new(2, 1), vec![10, 20]).expect("2x1");
        let hi = Channel::from_samples(ChannelSpec::new(2, 1), vec![2, -4]).expect("2x1");
        let out = horiz_isqueeze(&lo, &hi, &mut guard()).expect("inverse");
        assert_eq!(out.width(), 4);
        assert_eq!(out.samples(), &[10, 9, 18, 22]);

        // Cross-check against the forward relation the inverse must satisfy:
        // for each pair, diff = a - b and avg = a - (diff Idiv 2).
        for (i, (a, b)) in [(10i64, 9i64), (18, 22)].iter().enumerate() {
            let diff = a - b;
            let avg = a - diff / 2;
            assert_eq!(avg, i64::from(lo.samples()[i]), "pair {i} recovers its avg");
        }
    }

    #[test]
    fn horizontal_inverse_squeeze_carries_the_odd_column_through() {
        // w1 = 2, w2 = 1: the extra low-pass column is copied verbatim.
        let lo = Channel::from_samples(ChannelSpec::new(2, 1), vec![10, 77]).expect("2x1");
        let hi = Channel::from_samples(ChannelSpec::new(1, 1), vec![0]).expect("1x1");
        let out = horiz_isqueeze(&lo, &hi, &mut guard()).expect("inverse");
        assert_eq!(out.width(), 3);
        assert_eq!(out.samples()[2], 77, "output(2 * w2) = input_1(w2)");
    }

    #[test]
    fn vertical_inverse_squeeze_mirrors_the_horizontal_one() {
        // Same numbers as the 4x1 example, transposed to 1x4.
        let lo = Channel::from_samples(ChannelSpec::new(1, 2), vec![10, 20]).expect("1x2");
        let hi = Channel::from_samples(ChannelSpec::new(1, 2), vec![2, -4]).expect("1x2");
        let out = vert_isqueeze(&lo, &hi, &mut guard()).expect("inverse");
        assert_eq!((out.width(), out.height()), (1, 4));
        assert_eq!(out.samples(), &[10, 9, 18, 22]);
    }

    #[test]
    fn vertical_inverse_squeeze_carries_the_odd_row_through() {
        let lo = Channel::from_samples(ChannelSpec::new(2, 2), vec![1, 2, 30, 40]).expect("2x2");
        let hi = Channel::from_samples(ChannelSpec::new(2, 1), vec![0, 0]).expect("2x1");
        let out = vert_isqueeze(&lo, &hi, &mut guard()).expect("inverse");
        assert_eq!(out.height(), 3);
        assert_eq!(&out.samples()[4..6], &[30, 40], "the last row is copied");
    }

    #[test]
    fn a_flat_signal_with_zero_residuals_reconstructs_flat() {
        // tendency is 0 on a constant run and diff = 0, so every output pair is
        // (avg, avg). This is the sanity property that catches a sign error in
        // `first - diff`.
        let lo = Channel::from_samples(ChannelSpec::new(3, 2), vec![5; 6]).expect("3x2");
        let hi = Channel::from_samples(ChannelSpec::new(3, 2), vec![0; 6]).expect("3x2");
        let out = horiz_isqueeze(&lo, &hi, &mut guard()).expect("inverse");
        assert_eq!(out.samples(), &[5; 12]);
    }

    #[test]
    fn inverse_squeeze_undoes_the_shift_increment() {
        let lo = Channel::from_samples(ChannelSpec::with_shifts(2, 1, 1, 0), vec![10, 20])
            .expect("2x1");
        let hi = Channel::from_samples(ChannelSpec::with_shifts(2, 1, 1, 0), vec![0, 0])
            .expect("2x1");
        let out = horiz_isqueeze(&lo, &hi, &mut guard()).expect("inverse");
        assert_eq!(out.hshift(), 0);

        // A channel whose shifts are "unrelated" (-1) keeps them.
        let lo = Channel::from_samples(ChannelSpec::with_shifts(2, 1, -1, -1), vec![10, 20])
            .expect("2x1");
        let hi = Channel::from_samples(ChannelSpec::with_shifts(2, 1, -1, -1), vec![0, 0])
            .expect("2x1");
        let out = horiz_isqueeze(&lo, &hi, &mut guard()).expect("inverse");
        assert_eq!(out.hshift(), -1);
    }

    #[test]
    fn mismatched_inputs_are_rejected() {
        let a = Channel::from_samples(ChannelSpec::new(2, 1), vec![0, 0]).expect("2x1");
        let b = Channel::from_samples(ChannelSpec::new(2, 2), vec![0; 4]).expect("2x2");
        assert!(horiz_isqueeze(&a, &b, &mut guard()).is_err(), "height mismatch");

        let c = Channel::from_samples(ChannelSpec::new(9, 1), vec![0; 9]).expect("9x1");
        let d = Channel::from_samples(ChannelSpec::new(2, 1), vec![0, 0]).expect("2x1");
        assert!(horiz_isqueeze(&c, &d, &mut guard()).is_err(), "width mismatch");
    }

    #[test]
    fn default_params_for_a_small_single_channel_image_are_empty() {
        // 8x8 needs no squeezing at all: neither dimension exceeds 8.
        let sp = default_params(&[ChannelSpec::new(8, 8)], 0).expect("defaults");
        assert!(sp.is_empty());
    }

    #[test]
    fn default_params_for_a_wide_image() {
        // 32x8, one channel. count == 1, so the leading two-channel pair is
        // skipped. h >= w is false, so the leading vertical step is skipped.
        // Then: w = 32 > 8 -> horizontal, w = 16; h = 8 not > 8.
        //       w = 16 > 8 -> horizontal, w = 8.
        //       loop ends.
        let sp = default_params(&[ChannelSpec::new(32, 8)], 0).expect("defaults");
        assert_eq!(sp.len(), 2);
        assert!(sp.iter().all(|s| s.horizontal && s.in_place));
        assert!(sp.iter().all(|s| s.begin_c == 0 && s.num_c == 1));
    }

    #[test]
    fn default_params_start_with_the_chroma_pair_for_three_equal_channels() {
        // count == 3 and channel[1] matches channel[0], so two non-in-place
        // steps over channels 1..=2 come first.
        let ch = vec![ChannelSpec::new(16, 16); 3];
        let sp = default_params(&ch, 0).expect("defaults");
        assert_eq!(sp[0].begin_c, 1);
        assert_eq!(sp[0].num_c, 2);
        assert!(!sp[0].in_place);
        assert!(sp[0].horizontal);
        assert!(!sp[1].horizontal);
        // Then the h >= w && h > 8 vertical step over all three channels.
        assert_eq!(sp[2].begin_c, 0);
        assert_eq!(sp[2].num_c, 3);
        assert!(sp[2].in_place);
        assert!(!sp[2].horizontal);
    }

    #[test]
    fn default_params_need_a_non_meta_channel() {
        assert!(default_params(&[ChannelSpec::new(8, 8)], 1).is_err());
    }
}
