//! `AnalysisAtlas`: the compact description of the image on the 8x8 atom grid
//! (`Encoder-plan1.md` §2.2).
//!
//! The second search-input IR. Every later search stage — block tiling,
//! adaptive quantization, chroma-from-luma, filter policy — asks questions
//! about a *rectangle* of atoms, and asking them of the source pixels every
//! time is what makes naive encoders slow. The atlas answers them once.
//!
//! # Milestone-1 scope
//!
//! Per-atom mean and variance per channel, which is what a block-tiling
//! decision needs first and what every other feature is built on. The rest of
//! §2.2's feature vector — gradients, laplacian energy, anisotropy, noise,
//! covariances, masking, saliency — and the integral images over them arrive
//! with the milestones that consume them (6, 7 and 9). Adding them now would
//! mean tuning features against no consumer.

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

/// The analysis atlas.
#[derive(Debug, Clone, PartialEq)]
pub struct AnalysisAtlas {
    grid: AtomGrid,
    atoms: Box<[AtomFeatures]>,
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
}
