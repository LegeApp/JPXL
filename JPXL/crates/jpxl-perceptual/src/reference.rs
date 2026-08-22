//! Source-only work, done once.
//!
//! Every candidate an encoder search scores is compared against the same
//! source, so everything the metric derives from the source alone — its
//! pyramid, its opponent-colour planes, and (optionally) their blurred means
//! and second moments — is computed once and retained.

use crate::bands::{BAND_ROWS, band_of, mutable_bands};
use crate::blur::Blur;
use crate::executor::BandExecutor;
use crate::{LinearRgbView, MIN_DIMENSION, MetricError, SCALES, color, pyramid};

/// How much of the source-only work the reference retains.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceRetention {
    /// Keep the positive-XYB planes and their blurred mean and second moment
    /// at every scale (≈ 9 planes × 4/3 of the source size in `f32`). A
    /// candidate then pays only its own blurs.
    Moments,
    /// Keep only the positive-XYB planes (≈ 3 planes × 4/3); the source's
    /// blurs are recomputed for every candidate. Bounded memory for very
    /// large frames.
    PlanesOnly,
}

impl ReferenceRetention {
    /// Pixel count above which [`Self::default_for`] chooses
    /// [`Self::PlanesOnly`].
    pub const MOMENTS_PIXEL_CAP: u64 = 24_000_000;

    /// [`Self::Moments`] up to [`Self::MOMENTS_PIXEL_CAP`] pixels, else
    /// [`Self::PlanesOnly`].
    #[must_use]
    pub const fn default_for(width: u32, height: u32) -> Self {
        if (width as u64) * (height as u64) <= Self::MOMENTS_PIXEL_CAP {
            Self::Moments
        } else {
            Self::PlanesOnly
        }
    }
}

/// The source at one pyramid scale.
#[derive(Debug, Clone)]
pub struct ReferenceScale {
    pub(crate) width: usize,
    pub(crate) height: usize,
    /// Positive-XYB planes (X', Y', B').
    pub(crate) xyb: [Vec<f32>; 3],
    /// Blurred planes, when retained.
    pub(crate) mu: Option<[Vec<f32>; 3]>,
    /// Blurred squared planes, when retained.
    pub(crate) s11: Option<[Vec<f32>; 3]>,
}

/// The precomputed source side of every comparison.
#[derive(Debug, Clone)]
pub struct PrecomputedReference {
    width: u32,
    height: u32,
    retention: ReferenceRetention,
    scales: Vec<ReferenceScale>,
}

impl PrecomputedReference {
    /// Prepares `source` for repeated comparison.
    ///
    /// # Errors
    ///
    /// [`MetricError::TooSmall`] when either dimension is below
    /// [`MIN_DIMENSION`].
    pub fn new(
        source: LinearRgbView<'_>,
        retention: ReferenceRetention,
        executor: &dyn BandExecutor,
    ) -> Result<Self, MetricError> {
        let (width, height) = (source.width(), source.height());
        if width < MIN_DIMENSION || height < MIN_DIMENSION {
            return Err(MetricError::TooSmall { width, height });
        }
        let mut scales = Vec::with_capacity(SCALES);
        let mut prev: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let mut cur: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let mut w = usize::try_from(width).unwrap_or(usize::MAX);
        let mut h = usize::try_from(height).unwrap_or(usize::MAX);
        let mut blurs = [Blur::new(), Blur::new()];
        for scale in 0..SCALES {
            if scale > 0 {
                // The metric keeps halving while the *current* scale is at
                // least 8 in both dimensions, so the last scale evaluated may
                // itself be smaller than 8 (a 256x192 source ends at 8x6).
                if w < MIN_DIMENSION as usize || h < MIN_DIMENSION as usize {
                    break;
                }
                let (nw, nh) = (pyramid::half(w), pyramid::half(h));
                core::mem::swap(&mut prev, &mut cur);
                let [pr, pg, pb] = &prev;
                let src: [&[f32]; 3] = if scale == 1 {
                    [source.r(), source.g(), source.b()]
                } else {
                    [pr, pg, pb]
                };
                downscale_planes(src, w, h, &mut cur, executor);
                w = nw;
                h = nh;
            }
            let src: [&[f32]; 3] = if scale == 0 {
                [source.r(), source.g(), source.b()]
            } else {
                let [cr, cg, cb] = &cur;
                [cr, cg, cb]
            };
            let mut xyb = [
                vec![0.0f32; w * h],
                vec![0.0f32; w * h],
                vec![0.0f32; w * h],
            ];
            convert_planes(src, &mut xyb, w, executor);
            let (mu, s11) = match retention {
                ReferenceRetention::PlanesOnly => (None, None),
                ReferenceRetention::Moments => {
                    let mut mu = [
                        vec![0.0f32; w * h],
                        vec![0.0f32; w * h],
                        vec![0.0f32; w * h],
                    ];
                    let mut s11 = [
                        vec![0.0f32; w * h],
                        vec![0.0f32; w * h],
                        vec![0.0f32; w * h],
                    ];
                    let mut square = vec![0.0f32; w * h];
                    for ((plane, mu), s11) in xyb.iter().zip(mu.iter_mut()).zip(s11.iter_mut()) {
                        multiply_planes(plane, plane, &mut square, w, executor);
                        blur_moments(&mut blurs, plane, &square, mu, s11, w, h, executor);
                    }
                    (Some(mu), Some(s11))
                }
            };
            scales.push(ReferenceScale {
                width: w,
                height: h,
                xyb,
                mu,
                s11,
            });
        }
        Ok(Self {
            width,
            height,
            retention,
            scales,
        })
    }

    /// Source width in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Source height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// What was retained.
    #[must_use]
    pub const fn retention(&self) -> ReferenceRetention {
        self.retention
    }

    /// Number of pyramid scales the source supports (at most [`SCALES`]).
    #[must_use]
    pub fn scale_count(&self) -> usize {
        self.scales.len()
    }

    /// Bytes of `f32` sample storage retained.
    #[must_use]
    pub fn retained_bytes(&self) -> usize {
        self.scales
            .iter()
            .map(|s| {
                let planes =
                    3 + if s.mu.is_some() { 3 } else { 0 } + if s.s11.is_some() { 3 } else { 0 };
                planes * s.width * s.height * core::mem::size_of::<f32>()
            })
            .sum()
    }

    pub(crate) fn scales(&self) -> &[ReferenceScale] {
        &self.scales
    }
}

/// Downscales three planes by two, one executor item per plane.
pub(crate) fn downscale_planes(
    src: [&[f32]; 3],
    in_w: usize,
    in_h: usize,
    dst: &mut [Vec<f32>; 3],
    executor: &dyn BandExecutor,
) {
    let out_len = pyramid::half(in_w) * pyramid::half(in_h);
    for plane in dst.iter_mut() {
        plane.clear();
        plane.resize(out_len, 0.0);
    }
    let [d0, d1, d2] = dst;
    let items = crate::bands::Handoff::new(vec![
        (src[0], d0.as_mut_slice()),
        (src[1], d1.as_mut_slice()),
        (src[2], d2.as_mut_slice()),
    ]);
    executor.run(items.len(), &|index| {
        if let Some((input, output)) = items.take(index) {
            pyramid::downscale_by_2(input, in_w, in_h, output);
        }
    });
}

/// Converts three linear-RGB planes to positive XYB in row bands.
pub(crate) fn convert_planes(
    src: [&[f32]; 3],
    dst: &mut [Vec<f32>; 3],
    width: usize,
    executor: &dyn BandExecutor,
) {
    let band_len = width.saturating_mul(BAND_ROWS);
    let [x, y, b] = dst;
    let bands = mutable_bands(
        vec![x.as_mut_slice(), y.as_mut_slice(), b.as_mut_slice()],
        band_len,
    );
    executor.run(bands.len(), &|index| {
        let Some(outs) = bands.take(index) else {
            return;
        };
        let mut outs = outs.into_iter();
        let (Some(x), Some(y), Some(b)) = (outs.next(), outs.next(), outs.next()) else {
            return;
        };
        color::planes_to_positive_xyb(
            band_of(src[0], index, band_len),
            band_of(src[1], index, band_len),
            band_of(src[2], index, band_len),
            x,
            y,
            b,
        );
    });
}

/// `out = a * b`, per pixel, in row bands.
pub(crate) fn multiply_planes(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    width: usize,
    executor: &dyn BandExecutor,
) {
    let band_len = width.saturating_mul(BAND_ROWS);
    let bands = mutable_bands(vec![out], band_len);
    executor.run(bands.len(), &|index| {
        let Some(mut outs) = bands.take(index) else {
            return;
        };
        let Some(out) = outs.pop() else {
            return;
        };
        for ((&a, &b), o) in band_of(a, index, band_len)
            .iter()
            .zip(band_of(b, index, band_len))
            .zip(out.iter_mut())
        {
            *o = a * b;
        }
    });
}

/// Blurs a plane and its square (the mean and the second moment) as two
/// executor items.
#[allow(clippy::too_many_arguments)]
pub(crate) fn blur_moments(
    blurs: &mut [Blur; 2],
    plane: &[f32],
    square: &[f32],
    mu: &mut [f32],
    s11: &mut [f32],
    width: usize,
    height: usize,
    executor: &dyn BandExecutor,
) {
    let [b0, b1] = blurs;
    let items = crate::bands::Handoff::new(vec![(b0, plane, mu), (b1, square, s11)]);
    executor.run(items.len(), &|index| {
        if let Some((blur, input, output)) = items.take(index) {
            blur.blur_plane(input, output, width, height, executor);
        }
    });
}
