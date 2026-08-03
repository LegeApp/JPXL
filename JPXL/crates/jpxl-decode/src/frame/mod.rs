//! Frame headers, TOC, and group geometry (18181-1 Annexes F and G).
//!
//! A codestream is a sequence of frames (F.1). Each frame is byte-aligned by
//! `ZeroPadToByte()` and then reads as Table F.1:
//!
//! ```text
//! condition            type                              name           subclause
//!                      FrameHeader                       frame_header   F.2
//!                      TOC                               toc            F.3
//!                      LfGlobal                          lf_global      G.1
//!                      LfGroup     [num_lf_groups]       lf_group       G.2
//! encoding == kVarDCT  HfGlobal                          hf_global      G.3
//!                      PassGroup   [num_groups * passes] group_pass     G.4
//! ```
//!
//! This slice covers the first two rows plus the geometry that sizes the rest:
//! [`FrameHeader`] (F.2 and the bundles it nests), [`Toc`] (F.3 including the
//! optional permutation), and [`FrameGeometry`] (F.1 and 5.3). The section
//! *contents* — `LfGlobal`, `LfGroup`, `HfGlobal`, `PassGroup` — belong to
//! later slices; what exists here is their layout.
//!
//! # Reading order
//!
//! The three pieces are consumed in sequence and each depends on the previous:
//! the header supplies `group_size_shift`, `upsampling` and `num_passes`, from
//! which the geometry computes `num_sections`, which is how many entries the
//! TOC has.
//!
//! # Errors
//!
//! Frame parsing reports [`FrameError`], defined in this module rather than as
//! variants of [`DecodeError`](crate::DecodeError); see [`error`] for why and
//! for the slice-7 follow-up.

pub mod blending;
pub mod epf;
pub mod error;
pub mod gaborish;
pub mod geometry;
pub mod header;
pub mod passes;
pub mod patches;
pub mod restoration;
pub mod stream_index;
pub mod toc;

use jpxl_core::limits::{AllocGuard, Limits};

pub use blending::{BlendMode, BlendingInfo, read_blending_info};
pub use epf::{EpfStep, SigmaField, epf, epf_step, epf_steps, epf_weight, vardct_sigma};
pub use error::{FrameError, Result};
pub use gaborish::{GaborKernel, PlaneDims, gaborish, gaborish_into, gaborish_planes, mirror1d};
pub use geometry::{
    FrameGeometry, GroupLayout, MAX_NUM_PASSES, Rect, SectionKind, scale_frame_dimensions,
};
pub use header::{
    DURATION_NEXT_PAGE, Encoding, FrameFlags, FrameHeader, FrameType, read_frame_header,
};
pub use passes::{Passes, read_passes};
pub use patches::{
    Patch, PatchBlendMode, PatchBlending, PatchDictionary, PatchPosition, max_num_patches,
    read_patches,
};
pub use restoration::{EpfParams, GaborWeights, RestorationFilter, read_restoration_filter};
pub use toc::{Toc, get_context, lehmer_to_permutation, read_permutation, read_toc};

impl FrameGeometry {
    /// Derives a frame's geometry directly from its header (18181-1 F.1).
    ///
    /// Picks the frame's dimensions from the crop when one is present, then
    /// applies the `upsampling` and `lf_level` divisions.
    ///
    /// # Errors
    ///
    /// As [`FrameGeometry::derive`], plus
    /// [`FrameError::FieldOutOfRange`] if `group_size_shift` is out of range.
    pub fn from_header(
        header: &FrameHeader,
        image_width: u32,
        image_height: u32,
        limits: &Limits,
        guard: &mut AllocGuard,
    ) -> Result<Self> {
        let (width, height) = if header.have_crop {
            (header.width, header.height)
        } else {
            (image_width, image_height)
        };

        Self::derive(
            geometry::GroupLayout {
                width,
                height,
                upsampling: header.upsampling,
                lf_level: header.lf_level,
                group_dim: header.group_dim()?,
                num_passes: header.passes.num_passes,
            },
            limits,
            guard,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testsupport::BitWriter;
    use jpxl_bitstream::BitReader;

    fn metadata() -> crate::headers::ImageMetadata {
        crate::headers::ImageMetadata::default()
    }

    #[test]
    fn geometry_from_an_all_default_header() {
        let mut w = BitWriter::new();
        w.bool(true);
        let data = w.finish_padded(1);

        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let mut r = BitReader::new(&data);
        let header =
            read_frame_header(&mut r, &metadata(), 300, 200, &limits, &mut guard).expect("valid");

        let g = FrameGeometry::from_header(&header, 300, 200, &limits, &mut guard).expect("valid");
        assert_eq!((g.width(), g.height()), (300, 200));
        assert_eq!(g.group_dim(), 256);
        assert_eq!(g.num_groups(), 2);
        assert_eq!(g.num_sections(), 2 + 1 + 2);
    }

    #[test]
    fn geometry_uses_the_crop_dimensions() {
        let header = FrameHeader {
            have_crop: true,
            width: 100,
            height: 80,
            ..FrameHeader::default()
        };
        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let g = FrameGeometry::from_header(&header, 300, 200, &limits, &mut guard).expect("valid");

        assert_eq!((g.width(), g.height()), (100, 80));
        assert_eq!(g.num_groups(), 1);
        assert!(g.is_single_section());
    }

    #[test]
    fn geometry_applies_upsampling_from_the_header() {
        let header = FrameHeader {
            upsampling: 2,
            ..FrameHeader::default()
        };
        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let g = FrameGeometry::from_header(&header, 300, 200, &limits, &mut guard).expect("valid");
        assert_eq!((g.width(), g.height()), (150, 100));
    }
}
