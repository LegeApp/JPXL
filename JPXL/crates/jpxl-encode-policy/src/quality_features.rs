//! Frame-level source features for the quality controller's predictors.
//!
//! The initial-quantizer predictor (and, later, the policy ranker) needs a
//! handful of numbers that summarise how hard a source is to code at a given
//! perceptual score. They are computed from the production [`AnalysisAtlas`]
//! — per-8x8 mean and variance in XYB — so they cost nothing extra, and they
//! are defined here once so that the offline calibration (`jpxl features`
//! and `tools/calibrate_initial_rung.py`) and the encoder's own prediction
//! read exactly the same quantities.
//!
//! Every quantity is deterministic: quantiles are taken from a total-order
//! sort with a floor index, never interpolated.

use crate::analysis::AnalysisAtlas;

/// Luma variance (in XYB `Y` units squared) below which an 8x8 atom counts as
/// flat. An 8-bit source's ±1 LSB dither sits near `1.5e-5`, so this keeps
/// noise-free gradients and solid fills on the flat side of the line.
pub const FLAT_VARIANCE: f32 = 2e-5;

/// The feature vector one frame is described by.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourceFeatures {
    /// Frame width in samples.
    pub width: u32,
    /// Frame height in samples.
    pub height: u32,
    /// Whether every source pixel had `R == G == B`.
    pub grayscale: bool,
    /// 10th percentile of per-atom luma (`Y`) variance.
    pub luma_variance_q10: f32,
    /// Median per-atom luma variance.
    pub luma_variance_q50: f32,
    /// 90th percentile of per-atom luma variance.
    pub luma_variance_q90: f32,
    /// Median per-atom chroma variance (`X` variance plus `B` variance).
    pub chroma_variance_q50: f32,
    /// Fraction of atoms whose luma variance is below [`FLAT_VARIANCE`].
    pub flat_fraction: f32,
    /// `luma_variance_q90 - luma_variance_q50`: how much busier the busiest
    /// tenth of the frame is than its typical atom (an edge/texture proxy).
    pub edge_proxy: f32,
}

impl SourceFeatures {
    /// Frame area in pixels.
    #[must_use]
    pub const fn pixels(&self) -> u64 {
        self.width as u64 * self.height as u64
    }

    /// `log2` of the frame area.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        reason = "a frame area is far inside f64's exact integer range"
    )]
    pub fn log2_pixels(&self) -> f64 {
        (self.pixels().max(1) as f64).log2()
    }

    /// A one-line JSON object, for the CLI and the calibration tooling.
    #[must_use]
    pub fn to_json(&self) -> String {
        format!(
            "{{\"width\":{},\"height\":{},\"grayscale\":{},\"luma_variance_q10\":{:e},\
             \"luma_variance_q50\":{:e},\"luma_variance_q90\":{:e},\"chroma_variance_q50\":{:e},\
             \"flat_fraction\":{},\"edge_proxy\":{:e}}}",
            self.width,
            self.height,
            self.grayscale,
            self.luma_variance_q10,
            self.luma_variance_q50,
            self.luma_variance_q90,
            self.chroma_variance_q50,
            self.flat_fraction,
            self.edge_proxy,
        )
    }
}

/// The quantile of `values` at fraction `q` in `0..=1`: the element at
/// `floor((n - 1) q)` of the ascending total order. `0.0` for an empty slice.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "an atom count is far inside f64's exact range and the index is floored into 0..n"
)]
pub fn quantile(values: &mut [f32], q: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f32::total_cmp);
    let last = values.len() - 1;
    let index = ((last as f64) * f64::from(q.clamp(0.0, 1.0))).floor() as usize;
    values.get(index.min(last)).copied().unwrap_or(0.0)
}

/// Computes the features of a frame from its analysis atlas.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    reason = "an atom count is far inside f32's exact integer range"
)]
pub fn source_features(
    atlas: &AnalysisAtlas,
    width: u32,
    height: u32,
    grayscale: bool,
) -> SourceFeatures {
    let atoms = atlas.atoms();
    let mut luma: Vec<f32> = atoms.iter().map(|a| a.variance_xyb[1]).collect();
    let mut chroma: Vec<f32> = atoms
        .iter()
        .map(|a| a.variance_xyb[0] + a.variance_xyb[2])
        .collect();
    let flat = luma.iter().filter(|&&v| v < FLAT_VARIANCE).count();
    let flat_fraction = if luma.is_empty() {
        0.0
    } else {
        flat as f32 / luma.len() as f32
    };
    let luma_variance_q10 = quantile(&mut luma, 0.1);
    let luma_variance_q50 = quantile(&mut luma, 0.5);
    let luma_variance_q90 = quantile(&mut luma, 0.9);
    let chroma_variance_q50 = quantile(&mut chroma, 0.5);
    SourceFeatures {
        width,
        height,
        grayscale,
        luma_variance_q10,
        luma_variance_q50,
        luma_variance_q90,
        chroma_variance_q50,
        flat_fraction,
        edge_proxy: luma_variance_q90 - luma_variance_q50,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantiles_are_floor_indexed_order_statistics() {
        let mut v = [5.0f32, 1.0, 4.0, 2.0, 3.0];
        assert_eq!(quantile(&mut v, 0.0), 1.0);
        assert_eq!(quantile(&mut v, 0.5), 3.0);
        assert_eq!(quantile(&mut v, 0.9), 4.0);
        assert_eq!(quantile(&mut v, 1.0), 5.0);
        assert_eq!(quantile(&mut [], 0.5), 0.0);
    }

    #[test]
    fn a_flat_frame_is_entirely_flat_and_a_noisy_one_is_not() {
        let flat =
            crate::PreparedFrame::from_srgb8(64, 48, &vec![100u8; 64 * 48 * 3]).expect("frame");
        let atlas = AnalysisAtlas::analyze(&flat);
        let f = source_features(&atlas, 64, 48, true);
        assert_eq!(f.flat_fraction, 1.0);
        assert_eq!(f.luma_variance_q90, 0.0);
        assert!(f.grayscale);
        assert!(f.to_json().contains("\"flat_fraction\":1"));

        let noisy: Vec<u8> = (0..64 * 48 * 3)
            .map(|i| ((i * 97 + 13) % 251) as u8)
            .collect();
        let frame = crate::PreparedFrame::from_srgb8(64, 48, &noisy).expect("frame");
        let atlas = AnalysisAtlas::analyze(&frame);
        let f = source_features(&atlas, 64, 48, false);
        assert_eq!(f.flat_fraction, 0.0);
        assert!(f.luma_variance_q10 > FLAT_VARIANCE);
        assert!(f.edge_proxy >= 0.0);
    }
}
