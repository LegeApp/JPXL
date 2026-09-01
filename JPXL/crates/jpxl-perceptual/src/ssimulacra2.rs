//! The SSIMULACRA2 backend: candidate-side work, weighting, and the final
//! score.
//!
//! For every scale the reference supports, the candidate is downscaled (in
//! linear RGB), converted to positive XYB, blurred with its square and with
//! its product against the reference, pooled per component into the three
//! error maps' 1- and 4-norms, and the resulting `3 components × scales × 2
//! norms × 3 maps` terms (108 for a full pyramid) are combined with the
//! metric's published weights and remapped onto its 0..100 scale.

use crate::executor::BandExecutor;
use crate::pool::ChannelTerms;
#[cfg(feature = "evaluator")]
use crate::reference::convert_planes_in_place;
use crate::reference::{
    PrecomputedReference, ReferenceRetention, ReferenceScale, convert_planes, downscale_planes,
};
use crate::{LinearRgbView, MetricError, pyramid, streamed};

/// Pooled terms of one scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaleTerms {
    /// Width at this scale.
    pub width: u32,
    /// Height at this scale.
    pub height: u32,
    /// Terms per component (X', Y', B').
    pub channels: [ChannelTerms; 3],
}

/// A scored comparison.
#[derive(Debug, Clone, PartialEq)]
pub struct Ssimulacra2Result {
    /// The SSIMULACRA2 score: 100 is identical, ~90 visually lossless, ~70
    /// high quality, and it can go negative for badly damaged images.
    pub score: f64,
    /// The weighted error before remapping (the metric's internal distance).
    pub raw_error: f64,
    /// The pooled terms that produced it, one entry per evaluated scale.
    pub scales: Vec<ScaleTerms>,
}

/// Reusable candidate-side scratch for scoring against one or many references.
#[derive(Debug, Default)]
pub struct Ssimulacra2 {
    prev_rgb: [Vec<f32>; 3],
    cur_rgb: [Vec<f32>; 3],
    xyb: [Vec<f32>; 3],
    /// The streamed moment pipeline's rings and per-strip recursion state —
    /// `O(width)`, never a full frame.
    stream: streamed::Scratch,
}

impl Ssimulacra2 {
    /// A scorer with no scratch allocated yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Releases candidate-side full-frame scratch while retaining no semantic
    /// metric state.
    ///
    /// Large-image controller probes call this between evaluations so the
    /// scorer's planes do not overlap the next coefficient plan and rendered
    /// frame. The next score recreates the same zeroed buffers and therefore
    /// performs identical arithmetic.
    #[cfg(feature = "evaluator")]
    pub(crate) fn release_scratch(&mut self) {
        *self = Self::new();
    }

    /// Scores `candidate` against `reference`.
    ///
    /// # Errors
    ///
    /// [`MetricError::DimensionMismatch`] when the candidate's dimensions
    /// differ from the reference's.
    pub fn score(
        &mut self,
        reference: &PrecomputedReference,
        candidate: LinearRgbView<'_>,
        executor: &dyn BandExecutor,
    ) -> Result<Ssimulacra2Result, MetricError> {
        if candidate.width() != reference.width() || candidate.height() != reference.height() {
            return Err(MetricError::DimensionMismatch {
                reference: (reference.width(), reference.height()),
                candidate: (candidate.width(), candidate.height()),
            });
        }
        let mut scales = Vec::with_capacity(reference.scale_count());
        let mut w = usize::try_from(candidate.width()).unwrap_or(usize::MAX);
        let mut h = usize::try_from(candidate.height()).unwrap_or(usize::MAX);
        for (scale, rs) in reference.scales().iter().enumerate() {
            if scale > 0 {
                core::mem::swap(&mut self.prev_rgb, &mut self.cur_rgb);
                let [pr, pg, pb] = &self.prev_rgb;
                let src: [&[f32]; 3] = if scale == 1 {
                    [candidate.r(), candidate.g(), candidate.b()]
                } else {
                    [pr, pg, pb]
                };
                downscale_planes(src, w, h, &mut self.cur_rgb, executor);
                w = pyramid::half(w);
                h = pyramid::half(h);
            }
            debug_assert_eq!((w, h), (rs.width, rs.height));
            let pixels = w * h;
            for plane in self.xyb.iter_mut() {
                plane.clear();
                plane.resize(pixels, 0.0);
            }
            {
                let src: [&[f32]; 3] = if scale == 0 {
                    [candidate.r(), candidate.g(), candidate.b()]
                } else {
                    let [cr, cg, cb] = &self.cur_rgb;
                    [cr, cg, cb]
                };
                convert_planes(src, &mut self.xyb, w, executor);
            }
            let channels = self.score_converted_scale(rs, reference.retention(), w, h, executor);
            scales.push(ScaleTerms {
                width: u32::try_from(w).unwrap_or(u32::MAX),
                height: u32::try_from(h).unwrap_or(u32::MAX),
                channels,
            });
        }
        let raw_error = weighted_error(&scales);
        Ok(Ssimulacra2Result {
            score: remap(raw_error),
            raw_error,
            scales,
        })
    }

    /// Scores `candidate` at half linear resolution against the reference's
    /// own pyramid from scale 1: the Phase S1 surrogate observation.
    ///
    /// The reference's scale-1..N planes *are* the half-resolution source's
    /// pyramid (the 2:1 box downscale composes), so this is bit-for-bit the
    /// SSIMULACRA2 score of `(half(source), half(candidate))` with no
    /// source-side work beyond what the full-resolution reference already
    /// retains. All scale-0 candidate work — the dominant metric cost — is
    /// skipped; the candidate pays one downscale and the surviving scales.
    ///
    /// The result is a genuine score of a *different* comparison: it is
    /// blind to finest-scale loss and therefore reads systematically high.
    /// It may only ever propose a rung; the quality contract is satisfied
    /// exclusively by [`Self::score`] / [`Self::score_owned`].
    ///
    /// # Errors
    ///
    /// [`MetricError::DimensionMismatch`] when the candidate's dimensions
    /// differ from the reference's; [`MetricError::TooSmall`] when the
    /// reference pyramid has no scale below full resolution.
    pub fn score_surrogate(
        &mut self,
        reference: &PrecomputedReference,
        candidate: LinearRgbView<'_>,
        executor: &dyn BandExecutor,
    ) -> Result<Ssimulacra2Result, MetricError> {
        if candidate.width() != reference.width() || candidate.height() != reference.height() {
            return Err(MetricError::DimensionMismatch {
                reference: (reference.width(), reference.height()),
                candidate: (candidate.width(), candidate.height()),
            });
        }
        if reference.scale_count() < 2 {
            return Err(MetricError::TooSmall {
                width: reference.width().div_ceil(2),
                height: reference.height().div_ceil(2),
            });
        }
        let mut w = usize::try_from(candidate.width()).unwrap_or(usize::MAX);
        let mut h = usize::try_from(candidate.height()).unwrap_or(usize::MAX);
        let mut scales = Vec::with_capacity(reference.scale_count() - 1);
        for (scale, rs) in reference.scales().iter().enumerate().skip(1) {
            core::mem::swap(&mut self.prev_rgb, &mut self.cur_rgb);
            let [pr, pg, pb] = &self.prev_rgb;
            let src: [&[f32]; 3] = if scale == 1 {
                [candidate.r(), candidate.g(), candidate.b()]
            } else {
                [pr, pg, pb]
            };
            downscale_planes(src, w, h, &mut self.cur_rgb, executor);
            w = pyramid::half(w);
            h = pyramid::half(h);
            debug_assert_eq!((w, h), (rs.width, rs.height));
            let pixels = w * h;
            for plane in self.xyb.iter_mut() {
                plane.clear();
                plane.resize(pixels, 0.0);
            }
            {
                let [cr, cg, cb] = &self.cur_rgb;
                convert_planes([cr, cg, cb], &mut self.xyb, w, executor);
            }
            let channels = self.score_converted_scale(rs, reference.retention(), w, h, executor);
            scales.push(ScaleTerms {
                width: u32::try_from(w).unwrap_or(u32::MAX),
                height: u32::try_from(h).unwrap_or(u32::MAX),
                channels,
            });
        }
        let raw_error = weighted_error(&scales);
        Ok(Ssimulacra2Result {
            score: remap(raw_error),
            raw_error,
            scales,
        })
    }

    /// [`Self::score_surrogate`] for a candidate that is *already* at half
    /// resolution — the Phase S2 decimating-render path, where the renderer
    /// hands over half-resolution planes and the full-resolution candidate
    /// never exists.
    ///
    /// Fed the exact 2:1 box downscale of a full-resolution candidate, this
    /// is bit-for-bit [`Self::score_surrogate`] of that candidate (pinned by
    /// a test). Like every surrogate score it may only propose; the quality
    /// contract is satisfied exclusively by [`Self::score`] /
    /// [`Self::score_owned`].
    ///
    /// # Errors
    ///
    /// [`MetricError::TooSmall`] when the reference pyramid has no scale
    /// below full resolution; [`MetricError::DimensionMismatch`] when the
    /// candidate's dimensions differ from the reference's scale-1 planes.
    pub fn score_surrogate_prescaled(
        &mut self,
        reference: &PrecomputedReference,
        candidate: LinearRgbView<'_>,
        executor: &dyn BandExecutor,
    ) -> Result<Ssimulacra2Result, MetricError> {
        let Some(first) = reference.scales().get(1) else {
            return Err(MetricError::TooSmall {
                width: reference.width().div_ceil(2),
                height: reference.height().div_ceil(2),
            });
        };
        let (mut w, mut h) = (first.width, first.height);
        if (candidate.width(), candidate.height())
            != (
                u32::try_from(w).unwrap_or(u32::MAX),
                u32::try_from(h).unwrap_or(u32::MAX),
            )
        {
            return Err(MetricError::DimensionMismatch {
                reference: (
                    u32::try_from(w).unwrap_or(u32::MAX),
                    u32::try_from(h).unwrap_or(u32::MAX),
                ),
                candidate: (candidate.width(), candidate.height()),
            });
        }
        let mut scales = Vec::with_capacity(reference.scale_count() - 1);
        for (scale, rs) in reference.scales().iter().enumerate().skip(1) {
            if scale > 1 {
                core::mem::swap(&mut self.prev_rgb, &mut self.cur_rgb);
                let [pr, pg, pb] = &self.prev_rgb;
                let src: [&[f32]; 3] = if scale == 2 {
                    [candidate.r(), candidate.g(), candidate.b()]
                } else {
                    [pr, pg, pb]
                };
                downscale_planes(src, w, h, &mut self.cur_rgb, executor);
                w = pyramid::half(w);
                h = pyramid::half(h);
            }
            debug_assert_eq!((w, h), (rs.width, rs.height));
            let pixels = w * h;
            for plane in self.xyb.iter_mut() {
                plane.clear();
                plane.resize(pixels, 0.0);
            }
            {
                let src: [&[f32]; 3] = if scale == 1 {
                    [candidate.r(), candidate.g(), candidate.b()]
                } else {
                    let [cr, cg, cb] = &self.cur_rgb;
                    [cr, cg, cb]
                };
                convert_planes(src, &mut self.xyb, w, executor);
            }
            let channels = self.score_converted_scale(rs, reference.retention(), w, h, executor);
            scales.push(ScaleTerms {
                width: u32::try_from(w).unwrap_or(u32::MAX),
                height: u32::try_from(h).unwrap_or(u32::MAX),
                channels,
            });
        }
        let raw_error = weighted_error(&scales);
        Ok(Ssimulacra2Result {
            score: remap(raw_error),
            raw_error,
            scales,
        })
    }

    /// Scores owned candidate planes, reusing their allocations as the
    /// positive-XYB destination at each scale.
    ///
    /// This is the large-image controller path. It performs the same banded
    /// conversion and map arithmetic as [`Self::score`], but does not keep both
    /// linear RGB and XYB full-frame planes resident.
    #[cfg(feature = "evaluator")]
    pub(crate) fn score_owned(
        &mut self,
        reference: &PrecomputedReference,
        width: u32,
        height: u32,
        candidate: [Vec<f32>; 3],
        executor: &dyn BandExecutor,
    ) -> Result<Ssimulacra2Result, MetricError> {
        {
            let [r, g, b] = &candidate;
            let view = LinearRgbView::new(width, height, r, g, b)?;
            if view.width() != reference.width() || view.height() != reference.height() {
                return Err(MetricError::DimensionMismatch {
                    reference: (reference.width(), reference.height()),
                    candidate: (view.width(), view.height()),
                });
            }
        }

        let mut scales = Vec::with_capacity(reference.scale_count());
        let mut current = Some(candidate);
        let mut w = usize::try_from(width).unwrap_or(usize::MAX);
        let mut h = usize::try_from(height).unwrap_or(usize::MAX);
        for (scale, reference_scale) in reference.scales().iter().enumerate() {
            debug_assert_eq!((w, h), (reference_scale.width, reference_scale.height));
            let Some(mut rgb) = current.take() else {
                debug_assert!(false, "the candidate pyramid ended before the reference");
                break;
            };
            let next = if scale + 1 < reference.scale_count() {
                let [r, g, b] = &rgb;
                let mut next = [Vec::new(), Vec::new(), Vec::new()];
                downscale_planes([r, g, b], w, h, &mut next, executor);
                Some(next)
            } else {
                None
            };
            convert_planes_in_place(&mut rgb, w, executor);
            self.xyb = rgb;
            let channels =
                self.score_converted_scale(reference_scale, reference.retention(), w, h, executor);
            scales.push(ScaleTerms {
                width: u32::try_from(w).unwrap_or(u32::MAX),
                height: u32::try_from(h).unwrap_or(u32::MAX),
                channels,
            });
            current = next;
            w = pyramid::half(w);
            h = pyramid::half(h);
        }
        let raw_error = weighted_error(&scales);
        Ok(Ssimulacra2Result {
            score: remap(raw_error),
            raw_error,
            scales,
        })
    }

    fn score_converted_scale(
        &mut self,
        reference: &ReferenceScale,
        retention: ReferenceRetention,
        width: usize,
        height: usize,
        executor: &dyn BandExecutor,
    ) -> [ChannelTerms; 3] {
        streamed::channel_terms(
            &mut self.stream,
            reference,
            retention,
            &self.xyb,
            width,
            height,
            executor,
        )
    }
}

/// The metric's published weights over `(component, scale, norm, map)` in
/// that nesting order, for a full six-scale pyramid.
///
/// These were fitted by the metric's authors against subjective-quality
/// datasets; they are part of the metric's definition.
const WEIGHTS: [f64; 108] = [
    0.0,
    0.000_737_660_670_740_658_6,
    0.0,
    0.0,
    0.000_779_348_168_286_730_9,
    0.0,
    0.0,
    0.000_437_115_573_010_737_9,
    0.0,
    1.104_172_642_665_734_6,
    0.000_662_848_341_292_71,
    0.000_152_316_327_837_187_52,
    0.0,
    0.001_640_643_745_659_975_4,
    0.0,
    1.842_245_552_053_929_8,
    11.441_172_603_757_666,
    0.0,
    0.000_798_910_943_601_516_3,
    0.000_176_816_438_078_653,
    0.0,
    1.878_759_497_954_638_7,
    10.949_069_906_051_42,
    0.0,
    0.000_728_934_699_150_807_2,
    0.967_793_708_062_683_3,
    0.0,
    0.000_140_034_242_854_358_84,
    0.998_176_697_785_496_7,
    0.000_319_497_559_344_350_53,
    0.000_455_099_211_379_206_3,
    0.0,
    0.0,
    0.001_364_876_616_324_339_8,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    7.466_890_328_078_848,
    0.0,
    17.445_833_984_131_262,
    0.000_623_560_163_404_146_6,
    0.0,
    0.0,
    6.683_678_146_179_332,
    0.000_377_244_079_796_112_96,
    1.027_889_937_768_264,
    225.205_153_008_492_74,
    0.0,
    0.0,
    19.213_238_186_143_016,
    0.001_140_152_458_661_836_1,
    0.001_237_755_635_509_985,
    176.393_175_984_506_94,
    0.0,
    0.0,
    24.433_009_998_704_76,
    0.285_208_026_121_177_57,
    0.000_448_543_692_383_340_8,
    0.0,
    0.0,
    0.0,
    34.779_063_444_837_72,
    44.835_625_328_877_896,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.000_868_055_657_329_169_8,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.000_531_319_187_435_874_7,
    0.0,
    0.000_165_338_141_613_791_12,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.000_417_917_180_325_133_6,
    0.001_729_082_823_472_283_3,
    0.0,
    0.002_082_700_584_663_643_7,
    0.0,
    0.0,
    8.826_982_764_996_862,
    23.192_433_439_989_26,
    0.0,
    95.108_049_881_108_6,
    0.986_397_803_440_068_2,
    0.983_438_279_246_535_3,
    0.001_228_640_504_827_849_3,
    171.266_725_589_730_7,
    0.980_785_887_243_537_9,
    0.0,
    0.0,
    0.0,
    0.000_513_006_458_899_067_9,
    0.0,
    0.000_108_540_578_584_115_37,
];

/// The weighted sum of every pooled term.
///
/// Weights are consumed in `(component, evaluated scale, norm, map)` order.
/// When an image supports fewer than six scales the weights are consumed
/// consecutively over the scales that exist — the metric's own convention,
/// reproduced for parity.
#[must_use]
pub fn weighted_error(scales: &[ScaleTerms]) -> f64 {
    let mut weights = WEIGHTS.iter();
    let mut ssim = 0.0f64;
    let mut take = |value: f64| {
        let weight = weights.next().copied().unwrap_or(0.0);
        ssim += weight * value.abs();
    };
    for c in 0..3 {
        for scale in scales {
            let terms = scale.channels.get(c).copied().unwrap_or_default();
            for n in 0..2 {
                take(terms.ssim.get(n).copied().unwrap_or(0.0));
                take(terms.artifact.get(n).copied().unwrap_or(0.0));
                take(terms.detail_lost.get(n).copied().unwrap_or(0.0));
            }
        }
    }
    ssim
}

/// Maps the weighted error onto the published 0..100 scale.
#[must_use]
pub fn remap(error: f64) -> f64 {
    let ssim = error * 0.956_238_261_683_484_4;
    let ssim = 6.248_496_625_763_138e-5 * ssim * ssim * ssim + 2.326_765_642_916_932 * ssim
        - 0.020_884_521_182_843_837 * ssim * ssim;
    if ssim > 0.0 {
        100.0 - 10.0 * ssim.powf(0.627_633_646_783_138_7)
    } else {
        100.0
    }
}

/// The cumulative weighted error after each completed scale, in evaluation
/// order — the certified early-rejection bound.
///
/// Every published weight is non-negative and every term enters as an
/// absolute value, so the sum after the first `k+1` scales is an exact lower
/// bound on the final weighted error: the remaining scales can only add to
/// it. Because [`remap`] is monotone falling, `remap(partial[k])` is an
/// exact upper bound on the final score, so a consumer holding a threshold
/// `t` may certify the candidate infeasible the moment
/// `remap(partial[k]) < t` — no approximation is involved.
///
/// Weights are consumed with the same consecutive convention as
/// [`weighted_error`]: for `n` evaluated scales the weight of
/// `(component, scale_idx, norm, map)` is `WEIGHTS[component*n*6 +
/// scale_idx*6 + norm*3 + map]`, so the last entry equals
/// [`weighted_error`] exactly.
#[must_use]
pub fn cumulative_partial_errors(scales: &[ScaleTerms]) -> Vec<f64> {
    let n = scales.len();
    let mut partials = Vec::with_capacity(n);
    let mut running = 0.0f64;
    for scale_idx in 0..n {
        for c in 0..3 {
            let Some(terms) = scales
                .get(scale_idx)
                .and_then(|st| st.channels.get(c))
                .copied()
            else {
                continue;
            };
            for norm in 0..2 {
                for (map, plane) in [terms.ssim, terms.artifact, terms.detail_lost]
                    .into_iter()
                    .enumerate()
                {
                    let weight = WEIGHTS
                        .get(c * n * 6 + scale_idx * 6 + norm * 3 + map)
                        .copied()
                        .unwrap_or(0.0);
                    let value = plane.get(norm).copied().unwrap_or(0.0);
                    running += weight * value.abs();
                }
            }
        }
        partials.push(running);
    }
    partials
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "evaluator")]
    use crate::executor::SerialExecutor;

    #[test]
    fn zero_error_is_a_perfect_score_and_error_lowers_it() {
        assert_eq!(remap(0.0), 100.0);
        assert!(remap(0.5) < remap(0.25));
        assert!(remap(0.25) < 100.0);
    }

    #[test]
    fn the_weight_table_has_one_entry_per_term() {
        assert_eq!(WEIGHTS.len(), 3 * crate::SCALES * 2 * 3);
        assert!(
            WEIGHTS.iter().all(|weight| *weight >= 0.0),
            "the early-rejection lower bound requires non-negative weights"
        );
    }

    /// Partial cumulative errors are a lower bound that closes exactly on
    /// the final weighted error — the certified early-rejection contract.
    #[test]
    #[expect(
        clippy::indexing_slicing,
        reason = "the test reads specific partials of arrays it just built and sized itself"
    )]
    fn cumulative_partials_bound_and_close_on_the_final_error() {
        let value = |seed: usize| {
            let magnitude = f64::from(u32::try_from(seed % 7).unwrap_or(0)) * 0.25;
            if seed.is_multiple_of(2) {
                magnitude
            } else {
                -magnitude
            }
        };
        for scale_count in [2usize, 6] {
            let scales: Vec<ScaleTerms> = (0..scale_count)
                .map(|s| ScaleTerms {
                    width: 8,
                    height: 8,
                    channels: core::array::from_fn(|c| ChannelTerms {
                        ssim: [value(s * 31 + c + 1), value(s * 31 + c + 2)],
                        artifact: [value(s * 17 + c + 3), value(s * 17 + c + 4)],
                        detail_lost: [value(s * 13 + c + 5), value(s * 13 + c + 6)],
                    }),
                })
                .collect();
            let partials = cumulative_partial_errors(&scales);
            assert_eq!(partials.len(), scale_count);
            for w in partials.windows(2) {
                assert!(w[1] >= w[0], "partials must be non-decreasing");
            }
            let final_error = weighted_error(&scales);
            assert!(
                (partials[partials.len() - 1] - final_error).abs() < 1e-12,
                "the last partial must equal the weighted error exactly"
            );
            // The bound direction: every prefix bounds the final error below.
            for p in &partials {
                assert!(*p <= final_error + 1e-12);
            }
        }
    }

    #[test]
    fn the_surrogate_is_the_native_score_of_the_half_resolution_pair() {
        let (width, height) = (32usize, 24usize);
        let pixels = width * height;
        let reference_planes: [Vec<f32>; 3] = core::array::from_fn(|channel| {
            (0..pixels)
                .map(|i| {
                    let sample = u16::try_from((i * 41 + channel * 13) % 239).unwrap_or(0);
                    f32::from(sample) / 238.0
                })
                .collect()
        });
        let mut candidate = reference_planes.clone();
        for (channel, plane) in candidate.iter_mut().enumerate() {
            let offset = f32::from(u16::try_from(channel + 1).unwrap_or(0)) * 0.002;
            for value in plane {
                *value = (*value + offset).min(1.0);
            }
        }
        let halve = |[x, y, b]: &[Vec<f32>; 3]| -> [Vec<f32>; 3] {
            core::array::from_fn(|c| {
                let mut out = vec![0.0f32; pyramid::half(width) * pyramid::half(height)];
                let channels = [x, y, b];
                let plane = channels
                    .get(c)
                    .expect("array::from_fn supplies an index in 0..3");
                pyramid::downscale_by_2(plane, width, height, &mut out);
                out
            })
        };
        fn view([x, y, b]: &[Vec<f32>; 3], w: usize, h: usize) -> LinearRgbView<'_> {
            LinearRgbView::new(
                u32::try_from(w).unwrap_or(0),
                u32::try_from(h).unwrap_or(0),
                x,
                y,
                b,
            )
            .expect("the test planes have the declared shape")
        }
        let full_reference = PrecomputedReference::new(
            view(&reference_planes, width, height),
            ReferenceRetention::Moments,
            &crate::executor::SerialExecutor,
        )
        .expect("the full-resolution test reference should precompute");
        let surrogate = Ssimulacra2::new()
            .score_surrogate(
                &full_reference,
                view(&candidate, width, height),
                &crate::executor::SerialExecutor,
            )
            .expect("the surrogate should score");
        let half_reference_planes = halve(&reference_planes);
        let half_candidate = halve(&candidate);
        let (hw, hh) = (pyramid::half(width), pyramid::half(height));
        let half_reference = PrecomputedReference::new(
            view(&half_reference_planes, hw, hh),
            ReferenceRetention::Moments,
            &crate::executor::SerialExecutor,
        )
        .expect("the half-resolution test reference should precompute");
        let native = Ssimulacra2::new()
            .score(
                &half_reference,
                view(&half_candidate, hw, hh),
                &crate::executor::SerialExecutor,
            )
            .expect("the half-resolution pair should score");
        assert_eq!(surrogate, native);

        // The prescaled entry point fed the exact 2:1 downscale is the
        // surrogate itself, bit for bit.
        let prescaled = Ssimulacra2::new()
            .score_surrogate_prescaled(
                &full_reference,
                view(&half_candidate, hw, hh),
                &crate::executor::SerialExecutor,
            )
            .expect("the prescaled surrogate should score");
        assert_eq!(prescaled, surrogate);
    }

    #[cfg(feature = "evaluator")]
    #[test]
    fn owned_in_place_scoring_matches_the_borrowed_path_bit_for_bit() {
        let (width, height) = (32u32, 24u32);
        let pixels = usize::try_from(u64::from(width) * u64::from(height)).unwrap_or(0);
        let reference_planes: [Vec<f32>; 3] = core::array::from_fn(|channel| {
            (0..pixels)
                .map(|i| {
                    let sample = u16::try_from((i * 37 + channel * 11) % 251).unwrap_or(0);
                    f32::from(sample) / 250.0
                })
                .collect()
        });
        let mut candidate = reference_planes.clone();
        for (channel, plane) in candidate.iter_mut().enumerate() {
            let offset = f32::from(u16::try_from(channel + 1).unwrap_or(0)) * 0.001;
            for value in plane {
                *value = (*value + offset).min(1.0);
            }
        }
        let [rr, rg, rb] = &reference_planes;
        let reference = PrecomputedReference::new(
            LinearRgbView::new(width, height, rr, rg, rb)
                .expect("the test reference planes have the declared shape"),
            ReferenceRetention::Moments,
            &SerialExecutor,
        )
        .expect("the test reference should precompute");
        let [cr, cg, cb] = &candidate;
        let borrowed = Ssimulacra2::new()
            .score(
                &reference,
                LinearRgbView::new(width, height, cr, cg, cb)
                    .expect("the test candidate planes have the declared shape"),
                &SerialExecutor,
            )
            .expect("the borrowed test candidate should score");
        let owned = Ssimulacra2::new()
            .score_owned(&reference, width, height, candidate, &SerialExecutor)
            .expect("the owned test candidate should score");
        assert_eq!(owned, borrowed);
    }
}
