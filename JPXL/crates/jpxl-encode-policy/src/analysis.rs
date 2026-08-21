//! `AnalysisAtlas`: the compact description of the image on the 8x8 atom grid
//! (`Encoder-plan1.md` §2.2).
//!
//! The second search-input IR. Every later search stage — block tiling,
//! adaptive quantization, chroma-from-luma, filter policy — asks questions
//! about a *rectangle* of atoms, and asking them of the source pixels every
//! time is what makes naive encoders slow. The atlas answers them once.
//!
//! # Production and diagnostic scope
//!
//! Per-atom mean and variance per channel, which is what a block-tiling
//! decision needs first and what every other feature is built on, remain the
//! compact production [`AnalysisAtlas`]. [`AnalysisAtlasV2`] is an explicitly
//! requested diagnostic atlas used to test whether richer edge/flatness
//! signals predict visible leakage before any signal is allowed to affect the
//! bitstream. Merely constructing the production atlas therefore has no new
//! work or storage cost.

use crate::source::PreparedFrame;

/// The 8x8 atom grid of a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtomGrid {
    /// Columns of 8x8 atoms.
    pub width: u32,
    /// Rows of 8x8 atoms.
    pub height: u32,
}

impl AtomGrid {
    /// The grid covering a `width` x `height` frame.
    #[must_use]
    pub const fn for_frame(width: u32, height: u32) -> Self {
        Self {
            width: width.div_ceil(8),
            height: height.div_ceil(8),
        }
    }

    /// Number of atoms.
    #[must_use]
    pub const fn area(self) -> u64 {
        self.width as u64 * self.height as u64
    }
}

/// One 8x8 atom's features.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AtomFeatures {
    /// Mean sample value per channel, in XYB order.
    pub mean_xyb: [f32; 3],
    /// Sample variance per channel, in XYB order.
    pub variance_xyb: [f32; 3],
}

/// Rich per-atom measurements for offline edge/flatness risk experiments.
///
/// These fields deliberately have no production consumer. A feature earns a
/// place in encoder policy only after the held-out analysis tooling shows that
/// it predicts reconstruction errors better than the compact atlas.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DiagnosticAtomFeatures {
    /// Mean squared horizontal and vertical first differences, in XYB order.
    pub gradient_energy_xyb: [[f32; 2]; 3],
    /// Mean product of horizontal and vertical first differences, in XYB.
    pub gradient_cross_xyb: [f32; 3],
    /// Mean squared four-neighbour Laplacian, in XYB order.
    pub laplacian_energy_xyb: [f32; 3],
    /// Mean squared residual after fitting an affine plane, in XYB order.
    pub plane_residual_xyb: [f32; 3],
    /// Median absolute four-neighbour Laplacian, in XYB order.
    pub noise_mad_xyb: [f32; 3],
    /// Maximum minus minimum sample value, in XYB order.
    pub dynamic_range_xyb: [f32; 3],
    /// Channel covariance in XY, XB, YB order.
    pub covariance_xyb: [f32; 3],
    /// Strength of one dominant gradient orientation in the Y plane.
    pub orientation_coherence_y: f32,
    /// Difference in Y-plane variance between the two edge-normal halves.
    pub flat_side_asymmetry_y: f32,
}

/// The analysis atlas.
#[derive(Debug, Clone, PartialEq)]
pub struct AnalysisAtlas {
    grid: AtomGrid,
    atoms: Box<[AtomFeatures]>,
}

/// The compact production atlas plus diagnostic-only edge/flatness features.
#[derive(Debug, Clone, PartialEq)]
pub struct AnalysisAtlasV2 {
    base: AnalysisAtlas,
    diagnostics: Box<[DiagnosticAtomFeatures]>,
}

impl AnalysisAtlas {
    /// Computes the atlas of `frame`.
    ///
    /// Atoms at the right and bottom edges cover fewer than 64 samples; their
    /// statistics are over the samples that exist, not over an edge-extended
    /// block, because an extension would invent texture the source does not
    /// have.
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the accumulators are f64 for exactness over 64 samples; the \
                  stored feature is f32 by design, and the narrowing is the \
                  last step"
    )]
    pub fn analyze(frame: &PreparedFrame) -> Self {
        let grid = AtomGrid::for_frame(frame.width(), frame.height());
        let stride = frame.width();
        let planes = [&frame.xyb().x, &frame.xyb().y, &frame.xyb().b];
        let mut atoms = Vec::with_capacity(usize::try_from(grid.area()).unwrap_or(0));

        for atom_y in 0..grid.height {
            for atom_x in 0..grid.width {
                let x0 = atom_x * 8;
                let y0 = atom_y * 8;
                let x1 = (x0 + 8).min(frame.width());
                let y1 = (y0 + 8).min(frame.height());
                let count = f64::from(x1 - x0) * f64::from(y1 - y0);
                let mut features = AtomFeatures::default();
                for (c, plane) in planes.iter().enumerate() {
                    let mut sum = 0.0f64;
                    let mut sum_sq = 0.0f64;
                    for y in y0..y1 {
                        for x in x0..x1 {
                            let v = f64::from(plane.at(x, y, stride).unwrap_or(0.0));
                            sum += v;
                            sum_sq += v * v;
                        }
                    }
                    let mean = if count > 0.0 { sum / count } else { 0.0 };
                    let variance = if count > 0.0 {
                        (sum_sq / count - mean * mean).max(0.0)
                    } else {
                        0.0
                    };
                    if let (Some(m), Some(v)) = (
                        features.mean_xyb.get_mut(c),
                        features.variance_xyb.get_mut(c),
                    ) {
                        *m = mean as f32;
                        *v = variance as f32;
                    }
                }
                atoms.push(features);
            }
        }

        Self {
            grid,
            atoms: atoms.into_boxed_slice(),
        }
    }

    /// The atom grid.
    #[must_use]
    pub const fn grid(&self) -> AtomGrid {
        self.grid
    }

    /// One atom's features.
    #[must_use]
    pub fn atom(&self, x: u32, y: u32) -> Option<&AtomFeatures> {
        if x >= self.grid.width || y >= self.grid.height {
            return None;
        }
        let index = u64::from(y) * u64::from(self.grid.width) + u64::from(x);
        usize::try_from(index).ok().and_then(|i| self.atoms.get(i))
    }

    /// Every atom, in raster order.
    #[must_use]
    pub fn atoms(&self) -> &[AtomFeatures] {
        &self.atoms
    }
}

impl AnalysisAtlasV2 {
    /// Computes the diagnostic atlas in a separate pass over `frame`.
    ///
    /// This is intentionally not called by the encoder. Keeping it opt-in
    /// makes the production bitstream and cost identity mechanically clear.
    #[must_use]
    pub fn analyze(frame: &PreparedFrame) -> Self {
        let base = AnalysisAtlas::analyze(frame);
        let grid = base.grid();
        let mut diagnostics = Vec::with_capacity(base.atoms().len());
        for atom_y in 0..grid.height {
            for atom_x in 0..grid.width {
                let base_atom = base.atom(atom_x, atom_y).copied().unwrap_or_default();
                diagnostics.push(analyze_diagnostic_atom(frame, atom_x, atom_y, base_atom));
            }
        }
        Self {
            base,
            diagnostics: diagnostics.into_boxed_slice(),
        }
    }

    /// The atom grid.
    #[must_use]
    pub const fn grid(&self) -> AtomGrid {
        self.base.grid()
    }

    /// The compact production atlas computed from the same frame.
    #[must_use]
    pub const fn base(&self) -> &AnalysisAtlas {
        &self.base
    }

    /// One atom's diagnostic features.
    #[must_use]
    pub fn atom(&self, x: u32, y: u32) -> Option<&DiagnosticAtomFeatures> {
        if x >= self.grid().width || y >= self.grid().height {
            return None;
        }
        let index = u64::from(y) * u64::from(self.grid().width) + u64::from(x);
        usize::try_from(index)
            .ok()
            .and_then(|i| self.diagnostics.get(i))
    }

    /// Every diagnostic atom, in raster order.
    #[must_use]
    pub fn atoms(&self) -> &[DiagnosticAtomFeatures] {
        &self.diagnostics
    }

    /// Resident feature bytes, excluding container allocation overhead.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.base.atoms().len().saturating_mul(
            core::mem::size_of::<AtomFeatures>()
                .saturating_add(core::mem::size_of::<DiagnosticAtomFeatures>()),
        )
    }
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "diagnostic accumulators use f64 and narrow once to their documented f32 storage"
)]
fn analyze_diagnostic_atom(
    frame: &PreparedFrame,
    atom_x: u32,
    atom_y: u32,
    base: AtomFeatures,
) -> DiagnosticAtomFeatures {
    let x0 = atom_x * 8;
    let y0 = atom_y * 8;
    let x1 = (x0 + 8).min(frame.width());
    let y1 = (y0 + 8).min(frame.height());
    let width = x1 - x0;
    let height = y1 - y0;
    let count = f64::from(width) * f64::from(height);
    let stride = frame.width();
    let planes = [&frame.xyb().x, &frame.xyb().y, &frame.xyb().b];
    let mut out = DiagnosticAtomFeatures::default();

    for (channel, plane) in planes.iter().enumerate() {
        let mut gx_sq = 0.0f64;
        let mut gy_sq = 0.0f64;
        let mut cross = 0.0f64;
        let mut gx_count = 0u32;
        let mut gy_count = 0u32;
        let mut cross_count = 0u32;
        let mut minimum = f64::INFINITY;
        let mut maximum = f64::NEG_INFINITY;

        for y in y0..y1 {
            for x in x0..x1 {
                let value = f64::from(plane.at(x, y, stride).unwrap_or(0.0));
                minimum = minimum.min(value);
                maximum = maximum.max(value);
                if x + 1 < x1 {
                    let gx = f64::from(plane.at(x + 1, y, stride).unwrap_or(0.0)) - value;
                    gx_sq += gx * gx;
                    gx_count += 1;
                }
                if y + 1 < y1 {
                    let gy = f64::from(plane.at(x, y + 1, stride).unwrap_or(0.0)) - value;
                    gy_sq += gy * gy;
                    gy_count += 1;
                }
                if x + 1 < x1 && y + 1 < y1 {
                    let gx = f64::from(plane.at(x + 1, y, stride).unwrap_or(0.0)) - value;
                    let gy = f64::from(plane.at(x, y + 1, stride).unwrap_or(0.0)) - value;
                    cross += gx * gy;
                    cross_count += 1;
                }
            }
        }

        let mut laplacian_sq = 0.0f64;
        let mut laplacian_abs = Vec::with_capacity(36);
        if width > 2 && height > 2 {
            for y in y0 + 1..y1 - 1 {
                for x in x0 + 1..x1 - 1 {
                    let centre = f64::from(plane.at(x, y, stride).unwrap_or(0.0));
                    let laplacian = f64::from(plane.at(x - 1, y, stride).unwrap_or(0.0))
                        + f64::from(plane.at(x + 1, y, stride).unwrap_or(0.0))
                        + f64::from(plane.at(x, y - 1, stride).unwrap_or(0.0))
                        + f64::from(plane.at(x, y + 1, stride).unwrap_or(0.0))
                        - 4.0 * centre;
                    laplacian_sq += laplacian * laplacian;
                    laplacian_abs.push(laplacian.abs());
                }
            }
        }

        let mean = f64::from(base.mean_xyb.get(channel).copied().unwrap_or(0.0));
        let centre_x = f64::from(width.saturating_sub(1)) * 0.5;
        let centre_y = f64::from(height.saturating_sub(1)) * 0.5;
        let mut slope_x_num = 0.0f64;
        let mut slope_y_num = 0.0f64;
        let mut slope_x_den = 0.0f64;
        let mut slope_y_den = 0.0f64;
        for y in y0..y1 {
            for x in x0..x1 {
                let dx = f64::from(x - x0) - centre_x;
                let dy = f64::from(y - y0) - centre_y;
                let delta = f64::from(plane.at(x, y, stride).unwrap_or(0.0)) - mean;
                slope_x_num += dx * delta;
                slope_y_num += dy * delta;
                slope_x_den += dx * dx;
                slope_y_den += dy * dy;
            }
        }
        let slope_x = if slope_x_den > 0.0 {
            slope_x_num / slope_x_den
        } else {
            0.0
        };
        let slope_y = if slope_y_den > 0.0 {
            slope_y_num / slope_y_den
        } else {
            0.0
        };
        let mut residual_sq = 0.0f64;
        for y in y0..y1 {
            for x in x0..x1 {
                let dx = f64::from(x - x0) - centre_x;
                let dy = f64::from(y - y0) - centre_y;
                let predicted = mean + slope_x * dx + slope_y * dy;
                let residual = f64::from(plane.at(x, y, stride).unwrap_or(0.0)) - predicted;
                residual_sq += residual * residual;
            }
        }

        if let Some(energy) = out.gradient_energy_xyb.get_mut(channel) {
            energy[0] = divide(gx_sq, gx_count) as f32;
            energy[1] = divide(gy_sq, gy_count) as f32;
        }
        if let Some(value) = out.gradient_cross_xyb.get_mut(channel) {
            *value = divide(cross, cross_count) as f32;
        }
        if let Some(value) = out.laplacian_energy_xyb.get_mut(channel) {
            *value = if laplacian_abs.is_empty() {
                0.0
            } else {
                (laplacian_sq / laplacian_abs.len() as f64) as f32
            };
        }
        if let Some(value) = out.noise_mad_xyb.get_mut(channel) {
            *value = median(&mut laplacian_abs) as f32;
        }
        if let Some(value) = out.plane_residual_xyb.get_mut(channel) {
            *value = if count > 0.0 {
                (residual_sq / count) as f32
            } else {
                0.0
            };
        }
        if let Some(value) = out.dynamic_range_xyb.get_mut(channel) {
            *value = if minimum.is_finite() && maximum.is_finite() {
                (maximum - minimum) as f32
            } else {
                0.0
            };
        }
    }

    for (pair_index, (left, right)) in [(0usize, 1usize), (0, 2), (1, 2)].into_iter().enumerate() {
        let left_plane = planes.get(left).copied().unwrap_or(&frame.xyb().x);
        let right_plane = planes.get(right).copied().unwrap_or(&frame.xyb().x);
        let left_mean = f64::from(base.mean_xyb.get(left).copied().unwrap_or(0.0));
        let right_mean = f64::from(base.mean_xyb.get(right).copied().unwrap_or(0.0));
        let mut covariance = 0.0f64;
        for y in y0..y1 {
            for x in x0..x1 {
                let a = f64::from(left_plane.at(x, y, stride).unwrap_or(0.0)) - left_mean;
                let b = f64::from(right_plane.at(x, y, stride).unwrap_or(0.0)) - right_mean;
                covariance += a * b;
            }
        }
        if let Some(value) = out.covariance_xyb.get_mut(pair_index) {
            *value = if count > 0.0 {
                (covariance / count) as f32
            } else {
                0.0
            };
        }
    }

    let y_energy = out.gradient_energy_xyb.get(1).copied().unwrap_or_default();
    let y_cross = f64::from(out.gradient_cross_xyb.get(1).copied().unwrap_or(0.0));
    let gx = f64::from(y_energy[0]);
    let gy = f64::from(y_energy[1]);
    out.orientation_coherence_y = (((gx - gy) * (gx - gy) + 4.0 * y_cross * y_cross).sqrt()
        / (gx + gy + 1e-20))
        .clamp(0.0, 1.0) as f32;
    out.flat_side_asymmetry_y =
        half_variance_asymmetry(&frame.xyb().y, stride, (x0, y0, x1, y1), gx >= gy) as f32;
    out
}

fn divide(sum: f64, count: u32) -> f64 {
    if count == 0 {
        0.0
    } else {
        sum / f64::from(count)
    }
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values.get(middle.saturating_sub(1)).copied().unwrap_or(0.0)
            + values.get(middle).copied().unwrap_or(0.0))
            * 0.5
    } else {
        values.get(middle).copied().unwrap_or(0.0)
    }
}

fn half_variance_asymmetry(
    plane: &crate::source::PlaneStore,
    stride: u32,
    bounds: (u32, u32, u32, u32),
    split_x: bool,
) -> f64 {
    let (x0, y0, x1, y1) = bounds;
    let midpoint = if split_x {
        (x0 + x1) / 2
    } else {
        (y0 + y1) / 2
    };
    let mut sum = [0.0f64; 2];
    let mut sum_sq = [0.0f64; 2];
    let mut count = [0u32; 2];
    for y in y0..y1 {
        for x in x0..x1 {
            let half = usize::from(if split_x {
                x >= midpoint
            } else {
                y >= midpoint
            });
            let value = f64::from(plane.at(x, y, stride).unwrap_or(0.0));
            if let (Some(total), Some(squares), Some(samples)) =
                (sum.get_mut(half), sum_sq.get_mut(half), count.get_mut(half))
            {
                *total += value;
                *squares += value * value;
                *samples += 1;
            }
        }
    }
    let variance: [f64; 2] = core::array::from_fn(|half| {
        let n = f64::from(count.get(half).copied().unwrap_or(0));
        if n == 0.0 {
            return 0.0;
        }
        let mean = sum.get(half).copied().unwrap_or(0.0) / n;
        (sum_sq.get(half).copied().unwrap_or(0.0) / n - mean * mean).max(0.0)
    });
    (variance[0] - variance[1]).abs() / (variance[0] + variance[1] + 1e-20)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_flat_image_has_zero_variance_everywhere() {
        let frame = PreparedFrame::from_linear_srgb(
            20,
            12,
            vec![0.25; 240],
            vec![0.25; 240],
            vec![0.25; 240],
        )
        .expect("legal frame");
        let atlas = AnalysisAtlas::analyze(&frame);
        // 20x12 samples is a 3x2 atom grid, the last column and row partial.
        assert_eq!(
            atlas.grid(),
            AtomGrid {
                width: 3,
                height: 2
            }
        );
        assert_eq!(atlas.atoms().len(), 6);
        for atom in atlas.atoms() {
            for v in atom.variance_xyb {
                assert!(v.abs() < 1e-9, "flat image, got variance {v}");
            }
        }
        assert_eq!(atlas.atom(3, 0), None);
    }

    #[test]
    fn variance_is_confined_to_the_atom_that_contains_the_edge() {
        // A vertical step at x = 4 falls inside atom column 0 only.
        let (w, h) = (16u32, 8u32);
        let mut plane = vec![0.1f32; (w * h) as usize];
        for y in 0..h {
            for x in 4..8 {
                if let Some(slot) = plane.get_mut((y * w + x) as usize) {
                    *slot = 0.9;
                }
            }
        }
        let frame = PreparedFrame::from_linear_srgb(w, h, plane.clone(), plane.clone(), plane)
            .expect("legal frame");
        let atlas = AnalysisAtlas::analyze(&frame);
        let edge = atlas.atom(0, 0).expect("in range");
        let flat = atlas.atom(1, 0).expect("in range");
        assert!(
            edge.variance_xyb[1] > flat.variance_xyb[1],
            "the atom holding the step must be the textured one"
        );
        assert!(flat.variance_xyb[1].abs() < 1e-9);
    }

    #[test]
    fn diagnostic_flat_and_partial_atoms_are_finite_and_bounded() {
        let frame =
            PreparedFrame::from_linear_srgb(11, 9, vec![0.25; 99], vec![0.25; 99], vec![0.25; 99])
                .expect("legal frame");
        let atlas = AnalysisAtlasV2::analyze(&frame);
        assert_eq!(
            atlas.grid(),
            AtomGrid {
                width: 2,
                height: 2
            }
        );
        assert_eq!(atlas.atoms().len(), 4);
        assert!(atlas.byte_size() <= atlas.atoms().len() * 128);
        for atom in atlas.atoms() {
            let values = atom
                .gradient_energy_xyb
                .iter()
                .flatten()
                .copied()
                .chain(atom.gradient_cross_xyb)
                .chain(atom.laplacian_energy_xyb)
                .chain(atom.plane_residual_xyb)
                .chain(atom.noise_mad_xyb)
                .chain(atom.dynamic_range_xyb)
                .chain(atom.covariance_xyb)
                .chain([atom.orientation_coherence_y, atom.flat_side_asymmetry_y]);
            for value in values {
                assert!(value.is_finite());
                assert!(value.abs() < 1e-8, "flat diagnostic was {value}");
            }
        }
    }

    #[test]
    fn diagnostic_detects_an_oriented_edge_with_one_textured_side() {
        let (w, h) = (8u32, 8u32);
        let mut plane = vec![0.1f32; 64];
        for y in 0..h {
            for (x, value) in [(1u32, 0.5f32), (2, 0.2), (3, 0.8)] {
                if let Some(sample) = plane.get_mut((y * w + x) as usize) {
                    *sample = value;
                }
            }
            for x in 4..w {
                if let Some(sample) = plane.get_mut((y * w + x) as usize) {
                    *sample = 0.9;
                }
            }
        }
        let frame = PreparedFrame::from_xyb(w, h, plane.clone(), plane.clone(), plane, false)
            .expect("legal frame");
        let atlas = AnalysisAtlasV2::analyze(&frame);
        let atom = atlas.atom(0, 0).expect("atom");
        assert!(atom.orientation_coherence_y > 0.5);
        assert!(atom.flat_side_asymmetry_y > 0.5);
    }

    #[test]
    fn diagnostic_noise_exceeds_a_clean_affine_ramp() {
        let (w, h) = (8u32, 8u32);
        let clean: Vec<f32> = (0..h)
            .flat_map(|y| (0..w).map(move |x| 0.1 + x as f32 * 0.03 + y as f32 * 0.01))
            .collect();
        let noisy: Vec<f32> = clean
            .iter()
            .enumerate()
            .map(|(index, &value)| value + if index % 2 == 0 { 0.08 } else { -0.08 })
            .collect();
        let clean_frame = PreparedFrame::from_xyb(w, h, clean.clone(), clean.clone(), clean, false)
            .expect("clean frame");
        let noisy_frame = PreparedFrame::from_xyb(w, h, noisy.clone(), noisy.clone(), noisy, false)
            .expect("noisy frame");
        let clean_atlas = AnalysisAtlasV2::analyze(&clean_frame);
        let noisy_atlas = AnalysisAtlasV2::analyze(&noisy_frame);
        let clean_atom = clean_atlas.atom(0, 0).expect("clean atom");
        let noisy_atom = noisy_atlas.atom(0, 0).expect("noisy atom");
        assert!(noisy_atom.noise_mad_xyb[1] > clean_atom.noise_mad_xyb[1]);
        assert!(noisy_atom.laplacian_energy_xyb[1] > clean_atom.laplacian_energy_xyb[1]);
        assert!(noisy_atom.plane_residual_xyb[1] > clean_atom.plane_residual_xyb[1]);
    }
}
