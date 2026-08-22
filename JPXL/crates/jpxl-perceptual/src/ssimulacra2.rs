//! The SSIMULACRA2 backend: candidate-side work, weighting, and the final
//! score.
//!
//! For every scale the reference supports, the candidate is downscaled (in
//! linear RGB), converted to positive XYB, blurred with its square and with
//! its product against the reference, pooled per component into the three
//! error maps' 1- and 4-norms, and the resulting `3 components × scales × 2
//! norms × 3 maps` terms (108 for a full pyramid) are combined with the
//! metric's published weights and remapped onto its 0..100 scale.

use crate::bands::{BAND_ROWS, Handoff, Partials, band_count, band_of};
use crate::blur::Blur;
use crate::executor::BandExecutor;
use crate::pool::{ChannelTerms, MapSums, MomentBands, accumulate_band};
use crate::reference::{
    PrecomputedReference, ReferenceRetention, convert_planes, downscale_planes, multiply_planes,
};
use crate::{LinearRgbView, MetricError, pyramid};

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
    mu2: Vec<f32>,
    s22: Vec<f32>,
    s12: Vec<f32>,
    square: Vec<f32>,
    product: Vec<f32>,
    ref_mu: Vec<f32>,
    ref_s11: Vec<f32>,
    ref_square: Vec<f32>,
    blurs: Vec<Blur>,
}

impl Ssimulacra2 {
    /// A scorer with no scratch allocated yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
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
        self.blurs.resize_with(5, Blur::new);
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
            for buf in [
                &mut self.mu2,
                &mut self.s22,
                &mut self.s12,
                &mut self.square,
                &mut self.product,
            ] {
                buf.clear();
                buf.resize(pixels, 0.0);
            }
            let recompute_reference = reference.retention() == ReferenceRetention::PlanesOnly;
            if recompute_reference {
                for buf in [&mut self.ref_mu, &mut self.ref_s11, &mut self.ref_square] {
                    buf.clear();
                    buf.resize(pixels, 0.0);
                }
            }

            let mut channels = [ChannelTerms::default(); 3];
            for (c, terms) in channels.iter_mut().enumerate() {
                let Some(img1) = rs.xyb.get(c) else { continue };
                let Some(img2) = self.xyb.get(c) else {
                    continue;
                };
                multiply_planes(img2, img2, &mut self.square, w, executor);
                multiply_planes(img1, img2, &mut self.product, w, executor);
                if recompute_reference {
                    multiply_planes(img1, img1, &mut self.ref_square, w, executor);
                }

                // Up to five independent blurs: the candidate's mean, second
                // moment and cross moment, plus the source moments when the
                // reference did not retain them.
                {
                    let mut blurs = self.blurs.iter_mut();
                    let mut items: Vec<(&mut Blur, &[f32], &mut [f32])> = Vec::with_capacity(5);
                    if let Some(b) = blurs.next() {
                        items.push((b, img2.as_slice(), self.mu2.as_mut_slice()));
                    }
                    if let Some(b) = blurs.next() {
                        items.push((b, self.square.as_slice(), self.s22.as_mut_slice()));
                    }
                    if let Some(b) = blurs.next() {
                        items.push((b, self.product.as_slice(), self.s12.as_mut_slice()));
                    }
                    if recompute_reference {
                        if let Some(b) = blurs.next() {
                            items.push((b, img1.as_slice(), self.ref_mu.as_mut_slice()));
                        }
                        if let Some(b) = blurs.next() {
                            items.push((
                                b,
                                self.ref_square.as_slice(),
                                self.ref_s11.as_mut_slice(),
                            ));
                        }
                    }
                    let items = Handoff::new(items);
                    executor.run(items.len(), &|index| {
                        if let Some((blur, input, output)) = items.take(index) {
                            blur.blur_plane(input, output, w, h, executor);
                        }
                    });
                }

                let (mu1, s11): (&[f32], &[f32]) = match (&rs.mu, &rs.s11) {
                    (Some(mu), Some(s11)) if !recompute_reference => (
                        mu.get(c).map_or(&[][..], Vec::as_slice),
                        s11.get(c).map_or(&[][..], Vec::as_slice),
                    ),
                    _ => (&self.ref_mu, &self.ref_s11),
                };
                let sums = pool_maps(
                    &MomentBands {
                        img1,
                        mu1,
                        s11,
                        img2,
                        mu2: &self.mu2,
                        s22: &self.s22,
                        s12: &self.s12,
                    },
                    w,
                    h,
                    executor,
                );
                *terms = ChannelTerms::from_sums(&sums, pixels);
            }
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
}

/// Pools the three maps over the plane in fixed row bands, reducing the
/// bands' partial sums in order.
fn pool_maps(
    planes: &MomentBands<'_>,
    width: usize,
    height: usize,
    executor: &dyn BandExecutor,
) -> MapSums {
    let band_len = width.saturating_mul(BAND_ROWS);
    let bands = band_count(height);
    let partials = Partials::<MapSums>::new(bands);
    executor.run(bands, &|index| {
        let mut sums = MapSums::default();
        let band = MomentBands {
            img1: band_of(planes.img1, index, band_len),
            mu1: band_of(planes.mu1, index, band_len),
            s11: band_of(planes.s11, index, band_len),
            img2: band_of(planes.img2, index, band_len),
            mu2: band_of(planes.mu2, index, band_len),
            s22: band_of(planes.s22, index, band_len),
            s12: band_of(planes.s12, index, band_len),
        };
        accumulate_band(&band, &mut sums);
        partials.set(index, sums);
    });
    let mut total = MapSums::default();
    for partial in partials.into_ordered() {
        total.add(&partial);
    }
    total
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_error_is_a_perfect_score_and_error_lowers_it() {
        assert_eq!(remap(0.0), 100.0);
        assert!(remap(0.5) < remap(0.25));
        assert!(remap(0.25) < 100.0);
    }

    #[test]
    fn the_weight_table_has_one_entry_per_term() {
        assert_eq!(WEIGHTS.len(), 3 * crate::SCALES * 2 * 3);
    }
}
