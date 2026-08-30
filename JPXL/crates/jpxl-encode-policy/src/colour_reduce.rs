//! Deterministic colour reduction for the text/UI candidate ladder.
//!
//! The routed competition ([`crate::content_class`]) encodes colour-reduced
//! copies of a census-sparse frame with the lossless Modular path and lets
//! the canonical metric arbitrate. This module produces those copies: a
//! weighted median-cut palette over the frame's exact colour histogram,
//! nearest-palette remapping, and no dithering (dither entropy is exactly
//! what the ladder exists to avoid). Everything is integer arithmetic with
//! total orderings, so the same frame reduces identically on every host,
//! worker count, and allocator.
//!
//! The input contract matches the classifier's routing rule: the frame has
//! at most [`crate::content_class::ROUTE_MAX_COLOURS`] unique colours, so
//! the histogram, the cut, and the remap all run over the unique colours
//! (tens of thousands), not the pixels.

#![allow(
    clippy::indexing_slicing,
    reason = "the median cut walks box ranges it constructed inside the colour array"
)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "colour components are u8 by construction and counts fit their targets"
)]

use std::collections::HashMap;

/// A frame reduced to at most `k` colours, in interleaved 8-bit sRGB.
#[derive(Debug, Clone)]
pub struct ReducedFrame {
    /// Interleaved RGB, same dimensions as the input.
    pub rgb: Vec<u8>,
    /// Palette size actually used (min of `k` and the unique-colour count).
    pub palette_len: u32,
    /// The reduction changed no pixel: the source already fit `k` colours.
    pub exact: bool,
}

/// One box of the median cut: a range of the colour array plus its bounds.
struct CutBox {
    start: usize,
    end: usize,
    weight: u64,
    min: [u8; 3],
    max: [u8; 3],
}

impl CutBox {
    fn of(colours: &[([u8; 3], u64)]) -> (u64, [u8; 3], [u8; 3]) {
        let mut weight = 0u64;
        let mut min = [255u8; 3];
        let mut max = [0u8; 3];
        for (c, w) in colours {
            weight += w;
            for axis in 0..3 {
                min[axis] = min[axis].min(c[axis]);
                max[axis] = max[axis].max(c[axis]);
            }
        }
        (weight, min, max)
    }

    fn longest_axis(&self) -> usize {
        let spans = [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ];
        // Ties break toward the lowest axis index: deterministic.
        let mut best = 0;
        for axis in 1..3 {
            if spans[axis] > spans[best] {
                best = axis;
            }
        }
        best
    }

    fn span(&self) -> u32 {
        u32::from(self.max[0] - self.min[0])
            + u32::from(self.max[1] - self.min[1])
            + u32::from(self.max[2] - self.min[2])
    }
}

/// Reduces an interleaved 8-bit sRGB frame to at most `k` colours.
///
/// Returns `None` when the frame has more than `colour_cap` unique colours —
/// the caller routed something the classifier should not have passed — so a
/// misrouted dense frame degrades to "no candidate", never to a slow one.
#[must_use]
pub fn reduce_to_k_colours(rgb: &[u8], k: u32, colour_cap: u32) -> Option<ReducedFrame> {
    let k = k.max(2);
    // Exact histogram over unique colours, keyed and sorted by packed RGB.
    let mut histogram: HashMap<u32, u64> = HashMap::new();
    for px in rgb.chunks_exact(3) {
        let key = (u32::from(px[0]) << 16) | (u32::from(px[1]) << 8) | u32::from(px[2]);
        *histogram.entry(key).or_insert(0) += 1;
        if histogram.len() > colour_cap as usize {
            return None;
        }
    }
    let mut colours: Vec<([u8; 3], u64)> = histogram
        .iter()
        .map(|(&key, &w)| ([(key >> 16) as u8, (key >> 8) as u8, key as u8], w))
        .collect();
    colours.sort_unstable_by_key(|(c, _)| (c[0], c[1], c[2]));

    if colours.len() <= k as usize {
        return Some(ReducedFrame {
            rgb: rgb.to_vec(),
            palette_len: colours.len() as u32,
            exact: true,
        });
    }

    // Weighted median cut: repeatedly split the box with the widest colour
    // span at the weighted median of its longest axis.
    let (weight, min, max) = CutBox::of(&colours);
    let mut boxes = vec![CutBox {
        start: 0,
        end: colours.len(),
        weight,
        min,
        max,
    }];
    while boxes.len() < k as usize {
        // The splittable box with the largest span; ties break to the lowest
        // index. A box of one colour (span 0) cannot split.
        let Some(pick) = (0..boxes.len())
            .filter(|&i| boxes[i].end - boxes[i].start > 1)
            .max_by_key(|&i| (boxes[i].span(), usize::MAX - i))
        else {
            break;
        };
        let cut = &boxes[pick];
        let axis = cut.longest_axis();
        let slice = &mut colours[cut.start..cut.end];
        slice.sort_unstable_by_key(|(c, _)| (c[axis], c[0], c[1], c[2]));
        // Weighted median index, kept strictly inside the slice.
        let half = cut.weight / 2;
        let mut acc = 0u64;
        let mut split = 1usize;
        for (i, (_, w)) in slice.iter().enumerate() {
            acc += w;
            if acc >= half {
                split = i + 1;
                break;
            }
        }
        let split = split.clamp(1, slice.len() - 1);
        let (start, end) = (cut.start, cut.end);
        let mid = start + split;
        let (w_lo, min_lo, max_lo) = CutBox::of(&colours[start..mid]);
        let (w_hi, min_hi, max_hi) = CutBox::of(&colours[mid..end]);
        boxes[pick] = CutBox {
            start,
            end: mid,
            weight: w_lo,
            min: min_lo,
            max: max_lo,
        };
        boxes.push(CutBox {
            start: mid,
            end,
            weight: w_hi,
            min: min_hi,
            max: max_hi,
        });
    }

    // Each box becomes its weighted mean colour (rounded half up).
    let palette: Vec<[u8; 3]> = boxes
        .iter()
        .map(|b| {
            let mut sums = [0u64; 3];
            let mut weight = 0u64;
            for (c, w) in &colours[b.start..b.end] {
                for axis in 0..3 {
                    sums[axis] += u64::from(c[axis]) * w;
                }
                weight += w;
            }
            let weight = weight.max(1);
            #[allow(
                clippy::cast_possible_truncation,
                reason = "a weighted mean of u8 samples is a u8"
            )]
            [
                ((sums[0] + weight / 2) / weight) as u8,
                ((sums[1] + weight / 2) / weight) as u8,
                ((sums[2] + weight / 2) / weight) as u8,
            ]
        })
        .collect();

    // Nearest palette entry per unique colour (squared distance; ties break
    // to the lowest palette index), then one remap pass over the pixels.
    let mut mapping: HashMap<u32, [u8; 3]> = HashMap::with_capacity(colours.len());
    for (c, _) in &colours {
        let mut best = 0usize;
        let mut best_d = u32::MAX;
        for (i, p) in palette.iter().enumerate() {
            let d: u32 = (0..3)
                .map(|axis| {
                    let delta = i32::from(c[axis]) - i32::from(p[axis]);
                    (delta * delta) as u32
                })
                .sum();
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        let key = (u32::from(c[0]) << 16) | (u32::from(c[1]) << 8) | u32::from(c[2]);
        mapping.insert(key, palette.get(best).copied().unwrap_or([0; 3]));
    }

    let mut out = Vec::with_capacity(rgb.len());
    for px in rgb.chunks_exact(3) {
        let key = (u32::from(px[0]) << 16) | (u32::from(px[1]) << 8) | u32::from(px[2]);
        let mapped = mapping.get(&key).copied().unwrap_or([px[0], px[1], px[2]]);
        out.extend_from_slice(&mapped);
    }
    Some(ReducedFrame {
        rgb: out,
        palette_len: palette.len() as u32,
        exact: false,
    })
}

#[cfg(test)]
#[allow(
    clippy::indexing_slicing,
    reason = "test fixtures index buffers they sized"
)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_within_the_budget_passes_through_exactly() {
        let rgb: Vec<u8> = (0..64u8).flat_map(|i| [i % 4 * 60, 10, 200]).collect();
        let reduced = reduce_to_k_colours(&rgb, 8, 1 << 16).expect("sparse");
        assert!(reduced.exact);
        assert_eq!(reduced.rgb, rgb);
        assert_eq!(reduced.palette_len, 4);
    }

    #[test]
    fn reduction_is_deterministic_and_respects_k() {
        // 256 distinct greys -> 16 palette entries, same result twice.
        let rgb: Vec<u8> = (0..=255u8).flat_map(|v| [v, v, v]).collect();
        let a = reduce_to_k_colours(&rgb, 16, 1 << 16).expect("sparse");
        let b = reduce_to_k_colours(&rgb, 16, 1 << 16).expect("sparse");
        assert_eq!(a.rgb, b.rgb);
        assert!(!a.exact);
        assert!(a.palette_len <= 16);
        let unique: std::collections::HashSet<[u8; 3]> =
            a.rgb.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect();
        assert!(unique.len() <= 16, "{} colours", unique.len());
        // Weighted means of grey inputs stay grey and close to their box.
        for (src, out) in rgb.chunks_exact(3).zip(a.rgb.chunks_exact(3)) {
            assert_eq!(out[0], out[1]);
            assert_eq!(out[1], out[2]);
            assert!((i32::from(src[0]) - i32::from(out[0])).abs() <= 16);
        }
    }

    #[test]
    fn a_dense_frame_is_refused() {
        let rgb: Vec<u8> = (0..3000u32)
            .flat_map(|i| (i * 3..i * 3 + 3).map(|v| (v % 251) as u8))
            .collect();
        assert!(reduce_to_k_colours(&rgb, 64, 16).is_none());
    }
}
