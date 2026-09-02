//! Frame-level structure statistics for the crossing predictor's case table.
//!
//! The 2026-09-02 case-predictor study found that the first-guess error of
//! the nearest-neighbour crossing predictor is bounded by what its feature
//! vector can tell apart: a saturated block frame looked like a solid-colour
//! one to the nine source features and ten transform features, and drew a
//! first guess coarse enough to end Fast under target. This module adds the
//! per-cell luma/chroma structure map the operator's bpg-rs `still265`
//! preanalysis computes (the same 32x32 cells, Sobel thresholds and cell
//! classes the text/UI classifier in [`crate::content_class`] already ports),
//! reduced to one frame-level vector: class shares, log-variance spread,
//! edge, flatness, noise and chroma-activity summaries, and a heterogeneity
//! measure ("mostly flat with one detailed object" versus "uniformly
//! textured").
//!
//! Everything per pixel is integer arithmetic over the 8-bit source samples,
//! so the cell records are host-independent; the frame-level reductions are
//! plain `f64` means and quantiles of those integers. Cells are the frame's
//! whole 32x32 cells only — a partial edge cell is dropped rather than
//! measured over fewer pixels, so a frame's vector does not depend on how
//! its edge happens to fall.
//!
//! The pass reads the 8-bit samples the frame was prepared from (they are
//! gone once the frame is XYB), so it is attached at frame preparation by
//! the callers that want it — the quality path and the calibration tooling —
//! and costs nothing on paths that do not.

/// Analysis cell size in luma samples (32x32, as bpg-rs).
pub const CELL: usize = 32;

/// The number of `f64` features in [`PreanalysisFeatures::vector`].
pub const PREANALYSIS_DIM: usize = 25;

/// Feature names, in [`PreanalysisFeatures::vector`] order (the JSON keys,
/// with the three derived logarithms last).
pub const PREANALYSIS_NAMES: [&str; PREANALYSIS_DIM] = [
    "pa_share_flat",
    "pa_share_gradient",
    "pa_share_chroma_critical",
    "pa_share_noisy",
    "pa_share_texture",
    "pa_share_directional_edge",
    "pa_share_text_like",
    "pa_log2var_mean",
    "pa_log2var_std",
    "pa_log2var_q10",
    "pa_log2var_q90",
    "pa_edge_mean",
    "pa_edge_q90",
    "pa_flat_mean",
    "pa_noise_mean",
    "pa_noise_q90",
    "pa_chroma_log2_mean",
    "pa_chroma_log2_q90",
    "pa_orient_entropy_mean",
    "pa_dir_dominance_mean",
    "pa_axis_aligned_mean",
    "pa_hetero_frac",
    "pa_ln_edge_mean",
    "pa_ln_nonflat",
    "pa_ln_noise",
];

// bpg-rs cell thresholds (8-bit units; `*_Q8` are shares in 0..=256).
const FLAT_VAR: u32 = 12;
const FLAT_EDGE_Q8: u16 = 13;
const GRAD_EDGE_Q8: u16 = 26;
const EDGE_DENSE_Q8: u16 = 38;
const TEXT_EDGE_Q8: u16 = 64;
const DIR_DOMINANT_Q8: u16 = 128;
const AXIS_ALIGNED_Q8: u16 = 160;
const LOW_ENTROPY_Q8: u16 = 110;
const HIGH_ENTROPY_Q8: u16 = 150;
const TEXTURE_VAR: u32 = 64;
const NOISE_HI: u16 = 7;
const CHROMA_HI: u32 = 96;
const EDGE_GRAD: i32 = 48;
const FLAT_GRAD: i32 = 12;
const WEAK_GRAD: i32 = 24;
/// Cells whose log2 variance is further than this from the frame mean count
/// as heterogeneous.
const HETERO_BITS: f64 = 2.0;

/// The coarse content class of one cell, in bpg-rs priority order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CellClass {
    Flat = 0,
    Gradient = 1,
    ChromaCritical = 2,
    Noisy = 3,
    Texture = 4,
    DirectionalEdge = 5,
    TextLike = 6,
}

/// The frame-level structure vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreanalysisFeatures {
    /// Whole 32x32 cells measured.
    pub cells: u32,
    /// Share of cells in each class, `Flat` .. `TextLike`.
    pub share: [f32; 7],
    /// Mean of per-cell `log2(1 + variance)`.
    pub log2var_mean: f32,
    /// Population standard deviation of per-cell `log2(1 + variance)`.
    pub log2var_std: f32,
    /// 10th percentile of per-cell `log2(1 + variance)`.
    pub log2var_q10: f32,
    /// 90th percentile of per-cell `log2(1 + variance)`.
    pub log2var_q90: f32,
    /// Mean per-cell edge-pixel share.
    pub edge_mean: f32,
    /// 90th percentile of the per-cell edge-pixel share.
    pub edge_q90: f32,
    /// Mean per-cell flat-pixel share (gradient magnitude under 12).
    pub flat_mean: f32,
    /// Mean per-cell weak-gradient high-pass residual (grain proxy).
    pub noise_mean: f32,
    /// 90th percentile of the per-cell noise proxy.
    pub noise_q90: f32,
    /// Mean per-cell `log2(1 + var Cb + var Cr)`.
    pub chroma_log2_mean: f32,
    /// 90th percentile of per-cell `log2(1 + var Cb + var Cr)`.
    pub chroma_log2_q90: f32,
    /// Mean gradient-orientation entropy over cells with edge pixels (0..=1).
    pub orient_entropy_mean: f32,
    /// Mean dominant-direction share over cells with edge pixels (0..=1).
    pub dir_dominance_mean: f32,
    /// Mean horizontal-plus-vertical share over cells with edge pixels.
    pub axis_aligned_mean: f32,
    /// Share of cells whose log2 variance is more than two bits from the
    /// frame mean.
    pub hetero_frac: f32,
}

impl PreanalysisFeatures {
    /// The feature vector, in [`PREANALYSIS_NAMES`] order.
    #[must_use]
    pub fn vector(&self) -> [f64; PREANALYSIS_DIM] {
        let s = |v: f32| f64::from(v);
        [
            s(self.share[0]),
            s(self.share[1]),
            s(self.share[2]),
            s(self.share[3]),
            s(self.share[4]),
            s(self.share[5]),
            s(self.share[6]),
            s(self.log2var_mean),
            s(self.log2var_std),
            s(self.log2var_q10),
            s(self.log2var_q90),
            s(self.edge_mean),
            s(self.edge_q90),
            s(self.flat_mean),
            s(self.noise_mean),
            s(self.noise_q90),
            s(self.chroma_log2_mean),
            s(self.chroma_log2_q90),
            s(self.orient_entropy_mean),
            s(self.dir_dominance_mean),
            s(self.axis_aligned_mean),
            s(self.hetero_frac),
            (s(self.edge_mean) + 1e-4).ln(),
            (1.0 - s(self.share[0]) + 1e-3).ln(),
            (s(self.noise_mean) + 0.1).ln(),
        ]
    }

    /// A one-line JSON object of the measured fields (the derived
    /// logarithms are recomputed by every reader from these).
    #[must_use]
    pub fn to_json(&self) -> String {
        let v = self.vector();
        let mut out = String::from("{\"pa_cells\":");
        out.push_str(&self.cells.to_string());
        for (name, value) in PREANALYSIS_NAMES
            .iter()
            .zip(v.iter())
            .take(PREANALYSIS_DIM - 3)
        {
            out.push_str(",\"");
            out.push_str(name);
            out.push_str("\":");
            out.push_str(&format!("{value}"));
        }
        out.push('}');
        out
    }
}

/// One cell's integer structure record.
struct Cell {
    class: CellClass,
    variance: u32,
    edge_q8: u16,
    flat_q8: u16,
    noise: u16,
    chroma: u32,
    has_edges: bool,
    entropy_q8: u16,
    dominance_q8: u16,
    axis_q8: u16,
}

/// Shannon entropy of the 4-bin direction histogram in q8 of 2 bits (the
/// same rounding as [`crate::content_class`]).
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "counts are small and the value is clamped into 0..=256 before the cast"
)]
fn entropy4_q8(dir: &[i64; 4], total: i64) -> u16 {
    if total <= 0 {
        return 0;
    }
    let t = total as f64;
    let mut h = 0.0f64;
    for &c in dir {
        if c > 0 {
            let p = c as f64 / t;
            h -= p * p.log2();
        }
    }
    ((h / 2.0) * 256.0).round().clamp(0.0, 256.0) as u16
}

fn classify(cell: &Cell) -> CellClass {
    if cell.edge_q8 >= TEXT_EDGE_Q8
        && cell.entropy_q8 < LOW_ENTROPY_Q8
        && cell.axis_q8 >= AXIS_ALIGNED_Q8
        && cell.noise < NOISE_HI
    {
        return CellClass::TextLike;
    }
    if cell.edge_q8 >= EDGE_DENSE_Q8 && cell.dominance_q8 >= DIR_DOMINANT_Q8 {
        return CellClass::DirectionalEdge;
    }
    if cell.noise >= NOISE_HI && cell.variance < TEXTURE_VAR {
        return CellClass::Noisy;
    }
    if cell.variance >= TEXTURE_VAR && cell.entropy_q8 >= HIGH_ENTROPY_Q8 {
        return CellClass::Texture;
    }
    if cell.variance < FLAT_VAR && cell.edge_q8 < FLAT_EDGE_Q8 {
        return CellClass::Flat;
    }
    if cell.edge_q8 < GRAD_EDGE_Q8 {
        return CellClass::Gradient;
    }
    if cell.chroma >= CHROMA_HI {
        return CellClass::ChromaCritical;
    }
    CellClass::Texture
}

/// Linear-interpolated quantile of an ascending slice (numpy's default), so
/// the offline study and the runtime agree to the last bit on a sorted
/// input.
fn quantile(sorted: &[f64], q: f64) -> f64 {
    let Some(&first) = sorted.first() else {
        return 0.0;
    };
    let Some(&last) = sorted.last() else {
        return 0.0;
    };
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a cell count is far inside f64's exact range and the index is floored into range"
    )]
    {
        let pos = q * (sorted.len() - 1) as f64;
        let lo = pos.floor() as usize;
        let frac = pos - lo as f64;
        let a = sorted.get(lo).copied().unwrap_or(last);
        let b = sorted.get(lo + 1).copied().unwrap_or(a);
        let _ = first;
        a + (b - a) * frac
    }
}

/// Computes the structure vector of an 8-bit interleaved sRGB frame.
///
/// `pixel(i)` returns the `[r, g, b]` bytes of pixel `i` in raster order.
/// `None` when the frame has no whole 32x32 cell.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    reason = "cell sums fit i64, the q8 shares are bounded by construction, and the pass is one straight-line loop"
)]
pub fn preanalysis_features(
    width: u32,
    height: u32,
    pixel: impl Fn(usize) -> [u8; 3],
) -> Option<PreanalysisFeatures> {
    let width = width as usize;
    let height = height as usize;
    let cells_x = width / CELL;
    let cells_y = height / CELL;
    if cells_x == 0 || cells_y == 0 {
        return None;
    }
    // Integer BT.601 planes, 8-bit, rounded.
    let pixels = width * height;
    let mut luma = vec![0u8; pixels];
    let mut cb = vec![0u8; pixels];
    let mut cr = vec![0u8; pixels];
    for i in 0..pixels {
        let [r, g, b] = pixel(i);
        let (r, g, b) = (i32::from(r), i32::from(g), i32::from(b));
        if let Some(l) = luma.get_mut(i) {
            *l = ((77 * r + 150 * g + 29 * b + 128) >> 8) as u8;
        }
        if let Some(c) = cb.get_mut(i) {
            *c = (((-43 * r - 85 * g + 128 * b + 128) >> 8) + 128).clamp(0, 255) as u8;
        }
        if let Some(c) = cr.get_mut(i) {
            *c = (((128 * r - 107 * g - 21 * b + 128) >> 8) + 128).clamp(0, 255) as u8;
        }
    }
    let lum = |x: i64, y: i64| -> i32 {
        let sx = x.clamp(0, width as i64 - 1) as usize;
        let sy = y.clamp(0, height as i64 - 1) as usize;
        luma.get(sy * width + sx).copied().map_or(0, i32::from)
    };
    let n = (CELL * CELL) as i64;
    let mut cells: Vec<Cell> = Vec::with_capacity(cells_x * cells_y);
    for cy in 0..cells_y {
        for cx in 0..cells_x {
            let x0 = (cx * CELL) as i64;
            let y0 = (cy * CELL) as i64;
            let mut sum = 0i64;
            let mut sum_sq = 0i64;
            let mut edge_count = 0i64;
            let mut flat_count = 0i64;
            let mut noise_sum = 0i64;
            let mut weak_count = 0i64;
            let mut dir = [0i64; 4];
            let mut cb_sum = 0i64;
            let mut cb_sq = 0i64;
            let mut cr_sum = 0i64;
            let mut cr_sq = 0i64;
            for y in y0..y0 + CELL as i64 {
                for x in x0..x0 + CELL as i64 {
                    let c = lum(x, y);
                    sum += i64::from(c);
                    sum_sq += i64::from(c * c);
                    let tl = lum(x - 1, y - 1);
                    let tc = lum(x, y - 1);
                    let tr = lum(x + 1, y - 1);
                    let ml = lum(x - 1, y);
                    let mr = lum(x + 1, y);
                    let bl = lum(x - 1, y + 1);
                    let bc = lum(x, y + 1);
                    let br = lum(x + 1, y + 1);
                    let gx = (tr + 2 * mr + br) - (tl + 2 * ml + bl);
                    let gy = (bl + 2 * bc + br) - (tl + 2 * tc + tr);
                    let grad = gx.abs() + gy.abs();
                    if grad > EDGE_GRAD {
                        edge_count += 1;
                        let ax = gx.abs();
                        let ay = gy.abs();
                        let bin = if ax >= 2 * ay {
                            0
                        } else if ay >= 2 * ax {
                            1
                        } else if (gx > 0) == (gy > 0) {
                            2
                        } else {
                            3
                        };
                        if let Some(d) = dir.get_mut(bin) {
                            *d += 1;
                        }
                    }
                    if grad < FLAT_GRAD {
                        flat_count += 1;
                    }
                    if grad < WEAK_GRAD {
                        let box_mean = (tl + tc + tr + ml + c + mr + bl + bc + br) / 9;
                        noise_sum += i64::from((c - box_mean).abs());
                        weak_count += 1;
                    }
                    let index = y as usize * width + x as usize;
                    let vb = i64::from(cb.get(index).copied().unwrap_or(128));
                    let vr = i64::from(cr.get(index).copied().unwrap_or(128));
                    cb_sum += vb;
                    cb_sq += vb * vb;
                    cr_sum += vr;
                    cr_sq += vr * vr;
                }
            }
            let mean = sum / n;
            let variance = ((sum_sq / n) - mean * mean).max(0) as u32;
            let cb_mean = cb_sum / n;
            let cr_mean = cr_sum / n;
            let chroma = (((cb_sq / n) - cb_mean * cb_mean).max(0)
                + ((cr_sq / n) - cr_mean * cr_mean).max(0))
            .min(i64::from(u16::MAX)) as u32;
            let edge_total = dir.iter().sum::<i64>();
            let denominator = edge_total.max(1);
            let dir_max = dir.iter().copied().max().unwrap_or(0);
            let mut cell = Cell {
                class: CellClass::Texture,
                variance,
                edge_q8: ((edge_count * 256) / n) as u16,
                flat_q8: ((flat_count * 256) / n) as u16,
                noise: if weak_count > 0 {
                    (noise_sum / weak_count) as u16
                } else {
                    0
                },
                chroma,
                has_edges: edge_total > 0,
                entropy_q8: entropy4_q8(&dir, edge_total),
                dominance_q8: ((dir_max * 256) / denominator) as u16,
                axis_q8: (((dir[0] + dir[1]) * 256) / denominator) as u16,
            };
            cell.class = classify(&cell);
            cells.push(cell);
        }
    }

    let count = cells.len() as f64;
    let mut share = [0.0f32; 7];
    for cell in &cells {
        if let Some(s) = share.get_mut(cell.class as usize) {
            *s += 1.0;
        }
    }
    for s in &mut share {
        *s /= count as f32;
    }
    let mut lv: Vec<f64> = cells
        .iter()
        .map(|c| (1.0 + f64::from(c.variance)).log2())
        .collect();
    let lv_mean = lv.iter().sum::<f64>() / count;
    let lv_std = (lv
        .iter()
        .map(|v| (v - lv_mean) * (v - lv_mean))
        .sum::<f64>()
        / count)
        .sqrt();
    let hetero = lv
        .iter()
        .filter(|v| (*v - lv_mean).abs() > HETERO_BITS)
        .count() as f64
        / count;
    lv.sort_by(f64::total_cmp);
    let mut edge: Vec<f64> = cells.iter().map(|c| f64::from(c.edge_q8) / 256.0).collect();
    let edge_mean = edge.iter().sum::<f64>() / count;
    edge.sort_by(f64::total_cmp);
    let flat_mean = cells
        .iter()
        .map(|c| f64::from(c.flat_q8) / 256.0)
        .sum::<f64>()
        / count;
    let mut noise: Vec<f64> = cells.iter().map(|c| f64::from(c.noise)).collect();
    let noise_mean = noise.iter().sum::<f64>() / count;
    noise.sort_by(f64::total_cmp);
    let mut chroma: Vec<f64> = cells
        .iter()
        .map(|c| (1.0 + f64::from(c.chroma)).log2())
        .collect();
    let chroma_mean = chroma.iter().sum::<f64>() / count;
    chroma.sort_by(f64::total_cmp);
    let edged: Vec<&Cell> = cells.iter().filter(|c| c.has_edges).collect();
    let edged_mean = |f: &dyn Fn(&Cell) -> f64| -> f64 {
        if edged.is_empty() {
            0.0
        } else {
            edged.iter().map(|c| f(c)).sum::<f64>() / edged.len() as f64
        }
    };
    let narrow = |v: f64| v as f32;
    Some(PreanalysisFeatures {
        cells: cells.len() as u32,
        share,
        log2var_mean: narrow(lv_mean),
        log2var_std: narrow(lv_std),
        log2var_q10: narrow(quantile(&lv, 0.1)),
        log2var_q90: narrow(quantile(&lv, 0.9)),
        edge_mean: narrow(edge_mean),
        edge_q90: narrow(quantile(&edge, 0.9)),
        flat_mean: narrow(flat_mean),
        noise_mean: narrow(noise_mean),
        noise_q90: narrow(quantile(&noise, 0.9)),
        chroma_log2_mean: narrow(chroma_mean),
        chroma_log2_q90: narrow(quantile(&chroma, 0.9)),
        orient_entropy_mean: narrow(edged_mean(&|c| f64::from(c.entropy_q8) / 256.0)),
        dir_dominance_mean: narrow(edged_mean(&|c| f64::from(c.dominance_q8) / 256.0)),
        axis_aligned_mean: narrow(edged_mean(&|c| f64::from(c.axis_q8) / 256.0)),
        hetero_frac: narrow(hetero),
    })
}

/// [`preanalysis_features`] over interleaved 8-bit samples.
#[must_use]
pub fn preanalysis_srgb8(width: u32, height: u32, rgb: &[u8]) -> Option<PreanalysisFeatures> {
    preanalysis_features(width, height, |i| match rgb.get(3 * i..3 * i + 3) {
        Some([r, g, b]) => [*r, *g, *b],
        _ => [0, 0, 0],
    })
}

/// [`preanalysis_features`] over interleaved samples wider than a byte,
/// reduced to their top eight bits (as [`crate::content_class`] does).
#[must_use]
pub fn preanalysis_srgb16(
    width: u32,
    height: u32,
    rgb: &[u16],
    bits_per_sample: u32,
) -> Option<PreanalysisFeatures> {
    let shift = bits_per_sample.saturating_sub(8);
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the shift and min bring every sample into 0..=255"
    )]
    let reduce = move |s: u16| (s >> shift).min(255) as u8;
    preanalysis_features(width, height, move |i| match rgb.get(3 * i..3 * i + 3) {
        Some([r, g, b]) => [reduce(*r), reduce(*g), reduce(*b)],
        _ => [0, 0, 0],
    })
}

#[cfg(test)]
#[allow(
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    reason = "test fixtures index buffers they just sized and compare exact shares"
)]
mod tests {
    use super::*;

    #[test]
    fn a_solid_frame_is_all_flat_and_a_striped_one_is_not() {
        let solid = vec![90u8; 96 * 64 * 3];
        let f = preanalysis_srgb8(96, 64, &solid).expect("cells");
        assert_eq!(f.cells, 6);
        assert_eq!(f.share[0], 1.0);
        assert_eq!(f.edge_mean, 0.0);
        assert_eq!(f.noise_mean, 0.0);
        assert_eq!(f.hetero_frac, 0.0);
        assert!((f.vector()[22] - 1e-4f64.ln()).abs() < 1e-9);

        let mut stripes = vec![255u8; 96 * 64 * 3];
        for y in 0..64 {
            for x in 0..96 {
                if x % 4 < 2 {
                    let i = (y * 96 + x) * 3;
                    stripes[i] = 0;
                    stripes[i + 1] = 0;
                    stripes[i + 2] = 0;
                }
            }
        }
        let g = preanalysis_srgb8(96, 64, &stripes).expect("cells");
        assert_eq!(g.share[0], 0.0);
        assert!(g.edge_mean > 0.5, "edges {}", g.edge_mean);
        assert!(g.axis_aligned_mean > 0.9, "axis {}", g.axis_aligned_mean);
        assert!(g.log2var_mean > 10.0, "var {}", g.log2var_mean);
        assert!(
            g.to_json()
                .starts_with("{\"pa_cells\":6,\"pa_share_flat\":0")
        );
    }

    #[test]
    fn a_block_frame_differs_from_a_solid_one() {
        // The 2026-09-02 failure case in miniature: one saturated block on a
        // solid ground must not look identical to the ground alone.
        let mut block = vec![40u8; 128 * 128 * 3];
        for y in 40..88 {
            for x in 40..88 {
                let i = (y * 128 + x) * 3;
                block[i] = 250;
                block[i + 1] = 30;
                block[i + 2] = 30;
            }
        }
        let f = preanalysis_srgb8(128, 128, &block).expect("cells");
        let s = preanalysis_srgb8(128, 128, &vec![40u8; 128 * 128 * 3]).expect("cells");
        assert!(f.share[0] < 1.0 && s.share[0] == 1.0);
        assert!(f.edge_mean > 0.0);
        assert!(f.chroma_log2_q90 > s.chroma_log2_q90);
        assert!(f.vector()[22] > s.vector()[22]);
    }

    #[test]
    fn frames_without_a_whole_cell_have_no_features() {
        assert!(preanalysis_srgb8(31, 40, &vec![0u8; 31 * 40 * 3]).is_none());
        let wide: Vec<u16> = vec![1000; 40 * 32 * 3];
        assert!(preanalysis_srgb16(40, 32, &wide, 12).is_some());
    }

    #[test]
    fn the_quantile_interpolates_like_numpy() {
        assert_eq!(quantile(&[1.0, 2.0, 3.0, 4.0], 0.5), 2.5);
        assert!((quantile(&[1.0, 2.0, 3.0, 4.0], 0.9) - 3.7).abs() < 1e-12);
        assert_eq!(quantile(&[5.0], 0.1), 5.0);
        assert_eq!(quantile(&[], 0.1), 0.0);
    }
}
