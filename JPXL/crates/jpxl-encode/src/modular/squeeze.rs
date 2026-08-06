//! Forward squeeze transform (18181-1 H.6.2) — encode peer of the decoder inverse.
//!
//! The inverse is `horiz_isqueeze` / `vert_isqueeze` in `jpxl-decode`. Forward
//! is derived so that inverse(forward(x)) recovers x exactly. Control flow is
//! not shared with the decoder (paired-bug rule).

use super::{CodedChannel, Plane};
use crate::error::{EncodeError, Result};

/// One squeeze step (Table H.9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SqueezeParams {
    /// Squeeze horizontally rather than vertically.
    pub horizontal: bool,
    /// Insert residuals after the squeezed band (`true`) or at list end.
    pub in_place: bool,
    /// First affected channel.
    pub begin_c: u32,
    /// Number of affected channels.
    pub num_c: u32,
}

/// Default step list when the wire carries `num_sq = 0` (H.6.2.1).
///
/// Mirrors the decoder algorithm so empty-on-wire and explicit defaults match.
#[must_use]
pub fn default_params(channels: &[CodedChannel], nb_meta_channels: usize) -> Vec<SqueezeParams> {
    let first = nb_meta_channels;
    let Some(base) = channels.get(first) else {
        return Vec::new();
    };
    let count = channels.len() - first;
    let mut sp = Vec::new();
    let first_u32 = u32::try_from(first).unwrap_or(0);
    let count_u32 = u32::try_from(count).unwrap_or(0);
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

    let push = |sp: &mut Vec<SqueezeParams>, horizontal: bool| {
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
    sp
}

/// Whether default squeeze would emit any steps for these frame-sized channels.
#[must_use]
pub fn default_would_run(width: u32, height: u32, num_channels: usize) -> bool {
    let channels: Vec<CodedChannel> = (0..num_channels)
        .map(|_| CodedChannel::full(width, height, Vec::new()))
        .collect();
    !default_params(&channels, 0).is_empty()
}

/// Applies a sequence of forward squeeze steps to channel samples.
///
/// Channel-list algebra matches Table H.8 / the decoder's `forward_squeeze`.
///
/// # Errors
///
/// Out-of-range steps or sample geometry mismatches.
pub fn apply_steps(channels: &mut Vec<CodedChannel>, steps: &[SqueezeParams]) -> Result<()> {
    for step in steps {
        apply_one_step(channels, *step)?;
    }
    Ok(())
}

fn apply_one_step(channels: &mut Vec<CodedChannel>, step: SqueezeParams) -> Result<()> {
    if step.num_c == 0 {
        return Err(EncodeError::unsupported("squeeze with num_c = 0", "H.6.2"));
    }
    let begin = step.begin_c as usize;
    let end = begin
        .checked_add(step.num_c as usize - 1)
        .ok_or_else(|| EncodeError::unsupported("squeeze range overflows", "H.6.2"))?;
    if end >= channels.len() {
        return Err(EncodeError::unsupported(
            "squeeze over missing channels",
            "H.6.2",
        ));
    }
    let r = if step.in_place {
        end + 1
    } else {
        channels.len()
    };

    // Process from begin..=end; each iteration inserts a residual, shifting
    // later indices — residuals land at r + offset with offset 0..num_c-1.
    for offset in 0..(step.num_c as usize) {
        let c = begin + offset;
        let ch = channels
            .get(c)
            .ok_or_else(|| EncodeError::unsupported("squeeze channel vanished", "H.6.2"))?
            .clone();
        if ch.width == 0 || ch.height == 0 {
            return Err(EncodeError::unsupported("squeeze empty channel", "H.6.2"));
        }
        let mut lo_hshift = ch.hshift;
        let mut lo_vshift = ch.vshift;
        let (lo, lo_w, lo_h, hi, hi_w, hi_h) = if step.horizontal {
            if lo_hshift >= 0 {
                lo_hshift += 1;
            }
            let (lo, w1, hi, w2) = horiz_fsqueeze(&ch.data, ch.width, ch.height)?;
            (lo, w1, ch.height, hi, w2, ch.height)
        } else {
            if lo_vshift >= 0 {
                lo_vshift += 1;
            }
            let (lo, h1, hi, h2) = vert_fsqueeze(&ch.data, ch.width, ch.height)?;
            (lo, ch.width, h1, hi, ch.width, h2)
        };
        // Residuals share the low-pass shifts after the increment (decoder).
        let hi_hshift = lo_hshift;
        let hi_vshift = lo_vshift;
        if let Some(slot) = channels.get_mut(c) {
            *slot = CodedChannel {
                width: lo_w,
                height: lo_h,
                hshift: lo_hshift,
                vshift: lo_vshift,
                data: lo,
            };
        }
        let at = r
            .checked_add(offset)
            .ok_or_else(|| EncodeError::unsupported("residual index overflows", "H.6.2"))?;
        if at > channels.len() {
            return Err(EncodeError::unsupported(
                "cannot insert residual channel",
                "H.6.2",
            ));
        }
        channels.insert(
            at,
            CodedChannel {
                width: hi_w,
                height: hi_h,
                hshift: hi_hshift,
                vshift: hi_vshift,
                data: hi,
            },
        );
    }
    Ok(())
}

/// Builds frame-sized channels, applies default squeeze, returns post-transform list.
///
/// # Errors
///
/// As [`apply_steps`].
pub fn apply_default_to_planes(
    width: u32,
    height: u32,
    planes: &[Plane],
) -> Result<Vec<CodedChannel>> {
    let mut channels: Vec<CodedChannel> = planes
        .iter()
        .map(|p| CodedChannel::full(width, height, p.clone()))
        .collect();
    let steps = default_params(&channels, 0);
    apply_steps(&mut channels, &steps)?;
    Ok(channels)
}

/// `tendency(A, B, C)` of H.6.2.2 — same integer arithmetic as the decoder.
#[must_use]
pub fn tendency(a: i64, b: i64, c: i64) -> i64 {
    if a >= b && b >= c {
        let mut x = (4 * a - 3 * c - b + 6) / 12;
        if x - (x & 1) > 2 * (a - b) {
            x = 2 * (a - b) + 1;
        }
        if x + (x & 1) > 2 * (b - c) {
            x = 2 * (b - c);
        }
        x
    } else if a <= b && b <= c {
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

/// Horizontal forward squeeze: one channel → (low-pass, residual).
///
/// Output low-pass width is `ceil(w/2)`, residual width is `floor(w/2)`.
///
/// # Errors
///
/// [`EncodeError::SampleCountMismatch`] if `data` length is not `width * height`.
pub fn horiz_fsqueeze(data: &[i32], width: u32, height: u32) -> Result<(Plane, u32, Plane, u32)> {
    let expected = (width as usize).saturating_mul(height as usize);
    if data.len() != expected {
        return Err(EncodeError::SampleCountMismatch {
            expected: expected as u64,
            found: data.len() as u64,
        });
    }
    let w1 = width.div_ceil(2);
    let w2 = width / 2;
    let mut lo = vec![0i32; (w1 as usize).saturating_mul(height as usize)];
    let mut hi = vec![0i32; (w2 as usize).saturating_mul(height as usize)];

    for y in 0..height {
        // First pass: averages for all pairs.
        for x in 0..w2 {
            let a = i64::from(sample(data, width, x * 2, y));
            let b = i64::from(sample(data, width, x * 2 + 1, y));
            let diff = a - b;
            let avg = a - diff / 2;
            set(&mut lo, w1, x, y, narrow(avg));
        }
        if w1 > w2 {
            set(&mut lo, w1, w2, y, sample(data, width, width - 1, y));
        }
        // Second pass: residuals need neighbouring averages and previous odd.
        for x in 0..w2 {
            let a = i64::from(sample(data, width, x * 2, y));
            let b = i64::from(sample(data, width, x * 2 + 1, y));
            let diff = a - b;
            let avg = i64::from(sample(&lo, w1, x, y));
            let next_avg = if x + 1 < w1 {
                i64::from(sample(&lo, w1, x + 1, y))
            } else {
                avg
            };
            let left = if x > 0 {
                i64::from(sample(data, width, x * 2 - 1, y))
            } else {
                avg
            };
            let residu = diff - tendency(left, avg, next_avg);
            set(&mut hi, w2, x, y, narrow(residu));
        }
    }
    Ok((lo, w1, hi, w2))
}

/// Vertical forward squeeze: one channel → (low-pass, residual).
///
/// # Errors
///
/// As [`horiz_fsqueeze`].
pub fn vert_fsqueeze(data: &[i32], width: u32, height: u32) -> Result<(Plane, u32, Plane, u32)> {
    let expected = (width as usize).saturating_mul(height as usize);
    if data.len() != expected {
        return Err(EncodeError::SampleCountMismatch {
            expected: expected as u64,
            found: data.len() as u64,
        });
    }
    let h1 = height.div_ceil(2);
    let h2 = height / 2;
    let mut lo = vec![0i32; (width as usize).saturating_mul(h1 as usize)];
    let mut hi = vec![0i32; (width as usize).saturating_mul(h2 as usize)];

    for x in 0..width {
        for y in 0..h2 {
            let a = i64::from(sample(data, width, x, y * 2));
            let b = i64::from(sample(data, width, x, y * 2 + 1));
            let diff = a - b;
            let avg = a - diff / 2;
            set(&mut lo, width, x, y, narrow(avg));
        }
        if h1 > h2 {
            set(&mut lo, width, x, h2, sample(data, width, x, height - 1));
        }
        for y in 0..h2 {
            let a = i64::from(sample(data, width, x, y * 2));
            let b = i64::from(sample(data, width, x, y * 2 + 1));
            let diff = a - b;
            let avg = i64::from(sample(&lo, width, x, y));
            let next_avg = if y + 1 < h1 {
                i64::from(sample(&lo, width, x, y + 1))
            } else {
                avg
            };
            let left = if y > 0 {
                i64::from(sample(data, width, x, y * 2 - 1))
            } else {
                avg
            };
            let residu = diff - tendency(left, avg, next_avg);
            set(&mut hi, width, x, y, narrow(residu));
        }
    }
    Ok((lo, h1, hi, h2))
}

fn sample(data: &[i32], stride: u32, x: u32, y: u32) -> i32 {
    let i = (y as usize)
        .saturating_mul(stride as usize)
        .saturating_add(x as usize);
    data.get(i).copied().unwrap_or(0)
}

fn set(data: &mut [i32], stride: u32, x: u32, y: u32, value: i32) {
    let i = (y as usize)
        .saturating_mul(stride as usize)
        .saturating_add(x as usize);
    if let Some(slot) = data.get_mut(i) {
        *slot = value;
    }
}

fn narrow(value: i64) -> i32 {
    i32::try_from(value).unwrap_or(if value < 0 { i32::MIN } else { i32::MAX })
}

#[cfg(test)]
mod tests {
    use super::*;
    use jpxl_core::limits::{AllocGuard, Limits};
    use jpxl_decode::modular::channel::{Channel, ChannelSpec};
    use jpxl_decode::modular::squeeze::{horiz_isqueeze, vert_isqueeze};

    fn guard() -> AllocGuard {
        AllocGuard::new(&Limits::relaxed())
    }

    #[test]
    fn horiz_forward_inverts_through_decoder() {
        let width = 4u32;
        let height = 1u32;
        let data = vec![10i32, 9, 18, 22];
        let (lo, w1, hi, w2) = horiz_fsqueeze(&data, width, height).expect("fwd");
        assert_eq!((w1, w2), (2, 2));
        // Matches the decoder worked example (lo=[10,20], hi=[2,-4]).
        assert_eq!(lo, vec![10, 20]);
        assert_eq!(hi, vec![2, -4]);

        let lo_ch = Channel::from_samples(ChannelSpec::new(w1, height), lo).expect("lo");
        let hi_ch = Channel::from_samples(ChannelSpec::new(w2, height), hi).expect("hi");
        let out = horiz_isqueeze(&lo_ch, &hi_ch, &mut guard()).expect("inv");
        assert_eq!(out.samples(), &data);
    }

    #[test]
    fn horiz_odd_width_carries_last_column() {
        let data = vec![1i32, 2, 3, 4, 5];
        let (lo, w1, hi, w2) = horiz_fsqueeze(&data, 5, 1).expect("fwd");
        assert_eq!((w1, w2), (3, 2));
        let lo_ch = Channel::from_samples(ChannelSpec::new(w1, 1), lo).expect("lo");
        let hi_ch = Channel::from_samples(ChannelSpec::new(w2, 1), hi).expect("hi");
        let out = horiz_isqueeze(&lo_ch, &hi_ch, &mut guard()).expect("inv");
        assert_eq!(out.samples(), &data);
    }

    #[test]
    fn vert_forward_inverts_through_decoder() {
        let data = vec![10i32, 9, 18, 22];
        let (lo, h1, hi, h2) = vert_fsqueeze(&data, 1, 4).expect("fwd");
        assert_eq!((h1, h2), (2, 2));
        let lo_ch = Channel::from_samples(ChannelSpec::new(1, h1), lo).expect("lo");
        let hi_ch = Channel::from_samples(ChannelSpec::new(1, h2), hi).expect("hi");
        let out = vert_isqueeze(&lo_ch, &hi_ch, &mut guard()).expect("inv");
        assert_eq!(out.samples(), &data);
    }

    #[test]
    fn random_grids_roundtrip_horiz_and_vert() {
        // Deterministic pseudo-random fills.
        for &(w, h) in &[(1u32, 1), (2, 2), (3, 5), (8, 7), (16, 16)] {
            let data: Plane = (0..(w * h))
                .map(|i| {
                    let i = i as i32;
                    ((i * 17 + 3) % 511) - 255
                })
                .collect();
            let (lo, w1, hi, w2) = horiz_fsqueeze(&data, w, h).expect("h");
            let lo_ch = Channel::from_samples(ChannelSpec::new(w1, h), lo).expect("lo");
            let hi_ch = Channel::from_samples(ChannelSpec::new(w2, h), hi).expect("hi");
            let out = horiz_isqueeze(&lo_ch, &hi_ch, &mut guard()).expect("inv h");
            assert_eq!(out.samples(), &data, "horiz {w}x{h}");

            let (lo, h1, hi, h2) = vert_fsqueeze(&data, w, h).expect("v");
            let lo_ch = Channel::from_samples(ChannelSpec::new(w, h1), lo).expect("lo");
            let hi_ch = Channel::from_samples(ChannelSpec::new(w, h2), hi).expect("hi");
            let out = vert_isqueeze(&lo_ch, &hi_ch, &mut guard()).expect("inv v");
            assert_eq!(out.samples(), &data, "vert {w}x{h}");
        }
    }
}
