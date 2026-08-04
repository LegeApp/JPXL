//! `PreparedFrame`: the normalized encoder input (`Encoder-plan1.md` §2.1).
//!
//! This is the first of the two *search-input* IRs, and it lives on the policy
//! side because nothing in it reaches the wire. It is the source image after
//! colour conversion and nothing else.
//!
//! # Why the plane store is a type
//!
//! §2.1's point: a 50-megapixel three-channel `f32` XYB image is roughly
//! 600 MB before a single candidate transform. Full-frame residency has to be
//! a *choice*, not the only implementation. [`PlaneStore`] is therefore an
//! enum from day one, with [`PlaneStoreKind::TiledSpill`] deliberately absent
//! until milestone 10 — but with every accessor already shaped so adding it
//! does not change a caller.

use jpxl_core::color::linear_srgb_to_xyb_planes;

use crate::error::{PolicyError, Result};

/// How a plane store holds its samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaneStoreKind {
    /// Every sample in RAM.
    Resident,
}

/// One channel's samples.
#[derive(Debug, Clone, PartialEq)]
pub struct PlaneStore {
    samples: Box<[f32]>,
}

impl PlaneStore {
    /// A resident store over `samples`.
    #[must_use]
    pub fn resident(samples: Vec<f32>) -> Self {
        Self {
            samples: samples.into_boxed_slice(),
        }
    }

    /// How the samples are held.
    #[must_use]
    pub const fn kind(&self) -> PlaneStoreKind {
        PlaneStoreKind::Resident
    }

    /// The samples in raster order.
    #[must_use]
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    /// The sample at `(x, y)` of a `stride`-wide plane, or `None` past the
    /// end.
    #[must_use]
    pub fn at(&self, x: u32, y: u32, stride: u32) -> Option<f32> {
        let index = u64::from(y) * u64::from(stride) + u64::from(x);
        usize::try_from(index)
            .ok()
            .and_then(|i| self.samples.get(i))
            .copied()
    }
}

/// The three XYB planes.
#[derive(Debug, Clone, PartialEq)]
pub struct XybPlanes {
    /// X.
    pub x: PlaneStore,
    /// Y.
    pub y: PlaneStore,
    /// B.
    pub b: PlaneStore,
}

/// The normalized encoder input.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedFrame {
    width: u32,
    height: u32,
    xyb: XybPlanes,
    intensity_target: f32,
}

impl PreparedFrame {
    /// Converts linear sRGB planes to XYB and wraps them.
    ///
    /// # Errors
    ///
    /// [`PolicyError::Unsupported`] for a zero dimension and
    /// [`PolicyError::SampleCountMismatch`] if a plane is not
    /// `width * height` long.
    pub fn from_linear_srgb(
        width: u32,
        height: u32,
        mut r: Vec<f32>,
        mut g: Vec<f32>,
        mut b: Vec<f32>,
    ) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(PolicyError::Unsupported {
                what: "a zero frame dimension",
            });
        }
        let expected = u64::from(width) * u64::from(height);
        for plane in [&r, &g, &b] {
            let found = plane.len() as u64;
            if found != expected {
                return Err(PolicyError::SampleCountMismatch { expected, found });
            }
        }
        linear_srgb_to_xyb_planes(&mut r, &mut g, &mut b);
        Ok(Self {
            width,
            height,
            xyb: XybPlanes {
                x: PlaneStore::resident(r),
                y: PlaneStore::resident(g),
                b: PlaneStore::resident(b),
            },
            intensity_target: jpxl_core::color::NOMINAL_INTENSITY_TARGET,
        })
    }

    /// Frame width in samples.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Frame height in samples.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// The XYB planes.
    #[must_use]
    pub const fn xyb(&self) -> &XybPlanes {
        &self.xyb
    }

    /// The intensity target the analysis stage weights against.
    #[must_use]
    pub const fn intensity_target(&self) -> f32 {
        self.intensity_target
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_prepared_frame_holds_three_converted_planes() {
        let frame = PreparedFrame::from_linear_srgb(2, 2, vec![0.5; 4], vec![0.5; 4], vec![0.5; 4])
            .expect("legal frame");
        assert_eq!((frame.width(), frame.height()), (2, 2));
        assert_eq!(frame.xyb().y.samples().len(), 4);
        assert_eq!(frame.xyb().x.kind(), PlaneStoreKind::Resident);
        // Neutral grey has (near) zero X.
        let x = frame.xyb().x.at(0, 0, 2).expect("in range");
        assert!(x.abs() < 1e-3, "grey should decorrelate to X ~= 0, got {x}");
    }

    #[test]
    fn a_short_plane_is_rejected() {
        assert!(matches!(
            PreparedFrame::from_linear_srgb(2, 2, vec![0.0; 3], vec![0.0; 4], vec![0.0; 4]),
            Err(PolicyError::SampleCountMismatch { .. })
        ));
        assert!(matches!(
            PreparedFrame::from_linear_srgb(0, 2, Vec::new(), Vec::new(), Vec::new()),
            Err(PolicyError::Unsupported { .. })
        ));
    }
}
