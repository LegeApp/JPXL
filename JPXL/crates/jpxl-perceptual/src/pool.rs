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
pub fn accumulate_band(bands: &MomentBands<'_>, sums: &mut MapSums) {
    let MomentBands {
        img1,
        mu1,
        s11,
        img2,
        mu2,
        s22,
        s12,
    } = *bands;
    for ((((((&i1, &m1), &v11), &i2), &m2), &v22), &v12) in img1
        .iter()
        .zip(mu1)
        .zip(s11)
        .zip(img2)
        .zip(mu2)
        .zip(s22)
        .zip(s12)
    {
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
        let d = (1.0 - f64::from((num_m * num_s) / denom_s)).max(0.0);
        sums.ssim[0] += d;
        sums.ssim[1] += d * d * d * d;

        // Edge asymmetry: ratio of local high-pass magnitudes, minus one.
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
