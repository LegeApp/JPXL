//! Error maps and their pooling.
//!
//! For one component at one scale, SSIMULACRA2 forms three per-pixel error
//! maps from the local moments and pools each with a 1-norm (mean) and a
//! 4-norm (fourth root of the mean fourth power):
//!
//! * the modified SSIM error `1 - SSIM'` — SSIM with the luminance term's
//!   denominator dropped, because the components are already perceptually
//!   compressed;
//! * *artifact* (ringing, banding, blockiness): the distorted image has an
//!   edge where the original is smooth;
//! * *detail lost* (blur, smoothing): the original has an edge where the
//!   distorted image is smooth.
//!
//! The maps themselves are never materialised. Each row band produces its
//! partial sums and the bands are reduced in order, so the per-pixel fields
//! exist only as the terms a later attribution pass can recover from the same
//! row walk. Sums are carried in `f64`.

/// Stabilising constant of the structure term.
pub const SSIM_C2: f32 = 0.0009;

/// Raw sums of one component's three maps over some pixels: `[Σd, Σd⁴]` each.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MapSums {
    /// `1 - SSIM'` sums.
    pub ssim: [f64; 2],
    /// Added-edge (artifact) sums.
    pub artifact: [f64; 2],
    /// Lost-edge (detail lost) sums.
    pub detail_lost: [f64; 2],
}

impl MapSums {
    /// Adds `other`'s sums to these, term by term.
    pub fn add(&mut self, other: &Self) {
        self.ssim[0] += other.ssim[0];
        self.ssim[1] += other.ssim[1];
        self.artifact[0] += other.artifact[0];
        self.artifact[1] += other.artifact[1];
        self.detail_lost[0] += other.detail_lost[0];
        self.detail_lost[1] += other.detail_lost[1];
    }
}

/// One component's pooled terms at one scale: `[1-norm, 4-norm]` per map.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ChannelTerms {
    /// Pooled `1 - SSIM'`.
    pub ssim: [f64; 2],
    /// Pooled artifact map.
    pub artifact: [f64; 2],
    /// Pooled detail-lost map.
    pub detail_lost: [f64; 2],
}

impl ChannelTerms {
    /// Normalises raw sums over `pixels` pixels into the two norms.
    #[must_use]
    pub fn from_sums(sums: &MapSums, pixels: usize) -> Self {
        let inv = 1.0 / pixels.max(1) as f64;
        let norms = |s: [f64; 2]| [inv * s[0], (inv * s[1]).sqrt().sqrt()];
        Self {
            ssim: norms(sums.ssim),
            artifact: norms(sums.artifact),
            detail_lost: norms(sums.detail_lost),
        }
    }
}

/// The same band of the seven planes one component's maps are built from.
#[derive(Debug, Clone, Copy)]
pub struct MomentBands<'a> {
    /// Reference positive-XYB plane.
    pub img1: &'a [f32],
    /// Blurred reference.
    pub mu1: &'a [f32],
    /// Blurred squared reference.
    pub s11: &'a [f32],
    /// Candidate positive-XYB plane.
    pub img2: &'a [f32],
    /// Blurred candidate.
    pub mu2: &'a [f32],
    /// Blurred squared candidate.
    pub s22: &'a [f32],
    /// Blurred reference × candidate product.
    pub s12: &'a [f32],
}

/// Accumulates the three maps over one band of the seven planes.
///
/// Dispatched to an AVX2 build where the host supports it. The lane grouping
/// cannot change a value: each pixel's two error terms depend on that pixel's
/// seven inputs alone, and the running `f64` sums are folded in the original
/// pixel order either way, so the vector build is bit-identical to the scalar
/// one (Rust performs no floating-point contraction).
pub fn accumulate_band(bands: &MomentBands<'_>, sums: &mut MapSums) {
    #[cfg(target_arch = "x86_64")]
    if jpxl_core::cpu::has_avx2() {
        // SAFETY: `accumulate_band_avx2` only requires that the host support
        // AVX2, which `has_avx2` has just confirmed.
        #[allow(unsafe_code)]
        unsafe {
            accumulate_band_avx2(bands, sums);
        }
        return;
    }
    accumulate_band_impl(bands, sums);
}

/// [`accumulate_band`] compiled for AVX2.
///
/// Calling it is `unsafe` unless the host supports AVX2 (see
/// [`jpxl_core::cpu::has_avx2`]); that is the whole contract.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn accumulate_band_avx2(bands: &MomentBands<'_>, sums: &mut MapSums) {
    accumulate_band_impl(bands, sums);
}

/// Pixels whose error terms are computed together as one vector lane block.
///
/// Eight covers a full `f32` AVX2 register for the SSIM' arithmetic and two
/// `f64` registers for the asymmetry ratio, whose divide is the loop's
/// dominant cost.
const POOL_LANES: usize = 8;

#[inline(always)]
#[allow(
    clippy::indexing_slicing,
    reason = "every range below ends at `whole`, which is at most the length \
              of the shortest plane, and every lane index is bounded by the \
              chunk arrays' own size"
)]
fn accumulate_band_impl(bands: &MomentBands<'_>, sums: &mut MapSums) {
    let MomentBands {
        img1,
        mu1,
        s11,
        img2,
        mu2,
        s22,
        s12,
    } = *bands;

    // The scalar walk zips, stopping at the shortest plane; the lane walk
    // must cover exactly the same pixels.
    let n = img1
        .len()
        .min(mu1.len())
        .min(s11.len())
        .min(img2.len())
        .min(mu2.len())
        .min(s22.len())
        .min(s12.len());
    let whole = n - n % POOL_LANES;

    let (i1c, _) = img1[..whole].as_chunks::<POOL_LANES>();
    let (m1c, _) = mu1[..whole].as_chunks::<POOL_LANES>();
    let (v11c, _) = s11[..whole].as_chunks::<POOL_LANES>();
    let (i2c, _) = img2[..whole].as_chunks::<POOL_LANES>();
    let (m2c, _) = mu2[..whole].as_chunks::<POOL_LANES>();
    let (v22c, _) = s22[..whole].as_chunks::<POOL_LANES>();
    let (v12c, _) = s12[..whole].as_chunks::<POOL_LANES>();

    for ((((((i1, m1), v11), i2), m2), v22), v12) in i1c
        .iter()
        .zip(m1c)
        .zip(v11c)
        .zip(i2c)
        .zip(m2c)
        .zip(v22c)
        .zip(v12c)
    {
        // Stage 1, lane-parallel: both per-pixel error terms. Nothing here
        // reads the running sums, so the lanes are independent and the
        // compiler is free to keep the two divides eight and four wide.
        let mut ssim_d = [0.0f64; POOL_LANES];
        let mut asym = [0.0f64; POOL_LANES];
        for lane in 0..POOL_LANES {
            let (i1, m1, v11) = (i1[lane], m1[lane], v11[lane]);
            let (i2, m2, v22) = (i2[lane], m2[lane], v22[lane]);
            let v12 = v12[lane];
            // SSIM' error. The luminance term keeps only its numerator
            // (1 - (μ1-μ2)²); see the module docs for why the denominator is
            // dropped.
            let mu11 = m1 * m1;
            let mu22 = m2 * m2;
            let mu12 = m1 * m2;
            let mu_diff = m1 - m2;
            let num_m = 1.0 - mu_diff * mu_diff;
            let num_s = 2.0 * (v12 - mu12) + SSIM_C2;
            let denom_s = (v11 - mu11) + (v22 - mu22) + SSIM_C2;
            ssim_d[lane] = (1.0 - f64::from((num_m * num_s) / denom_s)).max(0.0);
            // Edge asymmetry: ratio of local high-pass magnitudes, minus one.
            asym[lane] =
                (1.0 + f64::from((i2 - m2).abs())) / (1.0 + f64::from((i1 - m1).abs())) - 1.0;
        }
        // Stage 2, ordered: fold into the running sums in pixel order, which
        // keeps the result bit-identical to the fully scalar walk.
        for lane in 0..POOL_LANES {
            let d = ssim_d[lane];
            sums.ssim[0] += d;
            sums.ssim[1] += d * d * d * d;
            let d1 = asym[lane];
            let artifact = d1.max(0.0);
            sums.artifact[0] += artifact;
            sums.artifact[1] += artifact * artifact * artifact * artifact;
            let lost = (-d1).max(0.0);
            sums.detail_lost[0] += lost;
            sums.detail_lost[1] += lost * lost * lost * lost;
        }
    }

    accumulate_scalar(
        &[
            &img1[whole..],
            &mu1[whole..],
            &s11[whole..],
            &img2[whole..],
            &mu2[whole..],
            &s22[whole..],
            &s12[whole..],
        ],
        sums,
    );
}

/// The original scalar walk, kept for the sub-lane remainder.
#[inline(always)]
fn accumulate_scalar(planes: &[&[f32]; 7], sums: &mut MapSums) {
    let [img1, mu1, s11, img2, mu2, s22, s12] = *planes;
    for ((((((&i1, &m1), &v11), &i2), &m2), &v22), &v12) in img1
        .iter()
        .zip(mu1)
        .zip(s11)
        .zip(img2)
        .zip(mu2)
        .zip(s22)
        .zip(s12)
    {
        let mu11 = m1 * m1;
        let mu22 = m2 * m2;
        let mu12 = m1 * m2;
        let mu_diff = m1 - m2;
        let num_m = 1.0 - mu_diff * mu_diff;
        let num_s = 2.0 * (v12 - mu12) + SSIM_C2;
        let denom_s = (v11 - mu11) + (v22 - mu22) + SSIM_C2;
        let d = (1.0 - f64::from((num_m * num_s) / denom_s)).max(0.0);
        sums.ssim[0] += d;
        sums.ssim[1] += d * d * d * d;

        let d1 = (1.0 + f64::from((i2 - m2).abs())) / (1.0 + f64::from((i1 - m1).abs())) - 1.0;
        let artifact = d1.max(0.0);
        sums.artifact[0] += artifact;
        sums.artifact[1] += artifact * artifact * artifact * artifact;
        let lost = (-d1).max(0.0);
        sums.detail_lost[0] += lost;
        sums.detail_lost[1] += lost * lost * lost * lost;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_moments_pool_to_zero() {
        let img = [0.2f32, 0.5, 0.9, 0.4];
        let mu = [0.3f32, 0.45, 0.8, 0.5];
        let s = [0.11f32, 0.3, 0.7, 0.2];
        let mut sums = MapSums::default();
        accumulate_band(
            &MomentBands {
                img1: &img,
                mu1: &mu,
                s11: &s,
                img2: &img,
                mu2: &mu,
                s22: &s,
                s12: &s,
            },
            &mut sums,
        );
        assert_eq!(sums, MapSums::default());
        assert_eq!(ChannelTerms::from_sums(&sums, 4), ChannelTerms::default());
    }

    /// The dispatched walk (lane-grouped, AVX2 where the host has it) must be
    /// bit-identical to the plain scalar walk — including on lengths that
    /// leave a sub-lane remainder — because pooled sums feed the canonical
    /// score, whose Contract A forbids host-dependent output.
    #[test]
    #[allow(
        clippy::indexing_slicing,
        clippy::cast_possible_truncation,
        reason = "fixture generation and seven fixed-index plane borrows in a \
                  test; an out-of-range index here is a test bug the panic \
                  reports directly"
    )]
    fn the_lane_walk_is_bit_identical_to_the_scalar_walk() {
        // A deterministic, sign-varied, non-smooth fill.
        let plane = |seed: u32, len: usize| -> Vec<f32> {
            (0..len)
                .map(|i| {
                    let x = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
                    (f64::from(x % 2003) / 1001.5 - 1.0) as f32
                })
                .collect()
        };
        for len in [0usize, 1, 7, 8, 9, 64, 250] {
            let planes: Vec<Vec<f32>> = (0..7u32).map(|s| plane(s * 97 + 13, len)).collect();
            let bands = MomentBands {
                img1: &planes[0],
                mu1: &planes[1],
                s11: &planes[2],
                img2: &planes[3],
                mu2: &planes[4],
                s22: &planes[5],
                s12: &planes[6],
            };
            let mut dispatched = MapSums::default();
            accumulate_band(&bands, &mut dispatched);
            let mut scalar = MapSums::default();
            accumulate_scalar(
                &[
                    &planes[0], &planes[1], &planes[2], &planes[3], &planes[4], &planes[5],
                    &planes[6],
                ],
                &mut scalar,
            );
            assert_eq!(dispatched, scalar, "length {len}");
        }
    }

    #[test]
    fn the_four_norm_of_a_constant_map_is_the_constant() {
        let sums = MapSums {
            ssim: [0.5 * 8.0, 0.0625 * 8.0],
            ..MapSums::default()
        };
        let terms = ChannelTerms::from_sums(&sums, 8);
        assert!((terms.ssim[0] - 0.5).abs() < 1e-12);
        assert!((terms.ssim[1] - 0.5).abs() < 1e-12);
    }
}
