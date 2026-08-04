//! The grids a VarDCT plan is validated against (18181-1 F.2, F.3.1, G.2, I.1).
//!
//! Four nested grids matter, and confusing two of them is the failure mode
//! this module exists to prevent:
//!
//! ```text
//! frame          width x height samples
//! LF group       8 * group_dim samples per side   (G.2.3 NOTE)
//! pass group     group_dim samples per side       (F.3.1); the "HF group"
//! 8x8 block      the atom a varblock is built from (I.1)
//! ```
//!
//! A varblock is placed on the *block* grid, must stay inside its *LF group*
//! (G.2.4) and — because its coefficients are carried by exactly one
//! `PassGroup` section (G.4) — inside a single *pass group* as well.
//!
//! Recomputed here rather than borrowed from `jpxl-decode`: an encoder that
//! shares the decoder's geometry cannot detect a geometry bug by round-tripping
//! against it. It is built on this crate's own [`crate::frame::Geometry`].

use jpxl_core::geometry::LfBlockPos;

use crate::frame::Geometry;
use crate::vardct::error::{PlanError, PlanResult};
use crate::vardct::ids::{HfGroupId, LfGroupId, PassId};

/// Samples per side of an 8x8 block.
pub const BLOCK_DIM: u32 = 8;

/// F.6's ceiling on `num_passes`: `U32(1, 2, 3, 4 + u(3))` tops out at 11.
pub const MAX_NUM_PASSES: u32 = 11;

/// A rectangle of the frame's sample grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    /// Left edge, in samples from the frame origin.
    pub x0: u32,
    /// Top edge, in samples from the frame origin.
    pub y0: u32,
    /// Width in samples; clipped at the frame's right edge.
    pub width: u32,
    /// Height in samples; clipped at the frame's bottom edge.
    pub height: u32,
}

/// A grid extent in 8x8 blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockGrid {
    /// Columns of 8x8 blocks.
    pub width: u32,
    /// Rows of 8x8 blocks.
    pub height: u32,
}

impl BlockGrid {
    /// Number of blocks in the grid.
    #[must_use]
    pub const fn area(self) -> u64 {
        self.width as u64 * self.height as u64
    }

    /// Whether `pos` is inside the grid.
    #[must_use]
    pub const fn contains(self, pos: LfBlockPos) -> bool {
        pos.bx() < self.width && pos.by() < self.height
    }
}

/// What a TOC section carries (18181-1 F.3.1, Table F.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionKind {
    /// The whole frame, when `num_groups == 1 && num_passes == 1`.
    Whole,
    /// `LfGlobal` (G.1).
    LfGlobal,
    /// One `LfGroup` (G.2), in raster order.
    LfGroup(LfGroupId),
    /// `HfGlobal` (G.3).
    HfGlobal,
    /// One `PassGroup` (G.4).
    PassGroup {
        /// Which pass.
        pass: PassId,
        /// Which group, in raster order.
        group: HfGroupId,
    },
}

/// The grids of one VarDCT frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VardctGeometry {
    base: Geometry,
    num_passes: u32,
}

impl VardctGeometry {
    /// Derives the grids of a `width` x `height` frame.
    ///
    /// # Errors
    ///
    /// [`PlanError::OutOfRange`] for a zero dimension, a `group_size_shift`
    /// above 3 (F.2's field is two bits) or a `num_passes` outside
    /// `1..=`[`MAX_NUM_PASSES`].
    pub fn new(
        width: u32,
        height: u32,
        group_size_shift: u32,
        num_passes: u32,
    ) -> PlanResult<Self> {
        if num_passes == 0 || num_passes > MAX_NUM_PASSES {
            return Err(PlanError::out_of_range(
                "num_passes",
                "F.6",
                i64::from(num_passes),
            ));
        }
        if group_size_shift > crate::frame::MAX_GROUP_SIZE_SHIFT {
            return Err(PlanError::out_of_range(
                "group_size_shift",
                "F.2",
                i64::from(group_size_shift),
            ));
        }
        for (what, value) in [("frame width", width), ("frame height", height)] {
            if value == 0 {
                return Err(PlanError::out_of_range(what, "D.2", 0));
            }
        }
        let base = Geometry::new(width, height, group_size_shift)
            .map_err(|_| PlanError::out_of_range("frame geometry", "F.2", i64::from(width)))?;
        Ok(Self { base, num_passes })
    }

    /// Frame width in samples.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.base.width()
    }

    /// Frame height in samples.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.base.height()
    }

    /// `group_dim = 128 << group_size_shift`: a pass group's side.
    #[must_use]
    pub const fn group_dim(&self) -> u32 {
        self.base.group_dim()
    }

    /// An LF group's side in samples, `8 * group_dim` (G.2.3 NOTE).
    #[must_use]
    pub const fn lf_group_dim(&self) -> u64 {
        self.base.group_dim() as u64 * 8
    }

    /// A pass group's side in 8x8 blocks.
    #[must_use]
    pub const fn group_blocks(&self) -> u32 {
        self.base.group_dim() / BLOCK_DIM
    }

    /// `num_passes`.
    #[must_use]
    pub const fn num_passes(&self) -> u32 {
        self.num_passes
    }

    /// Number of pass groups.
    #[must_use]
    pub const fn num_groups(&self) -> u64 {
        self.base.num_groups()
    }

    /// Number of LF groups.
    #[must_use]
    pub const fn num_lf_groups(&self) -> u64 {
        self.base.num_lf_groups()
    }

    /// The frame's whole 8x8-block grid.
    #[must_use]
    pub fn frame_blocks(&self) -> BlockGrid {
        BlockGrid {
            width: self.width().div_ceil(BLOCK_DIM),
            height: self.height().div_ceil(BLOCK_DIM),
        }
    }

    /// Number of LF-group columns.
    #[must_use]
    pub fn lf_groups_x(&self) -> u32 {
        let dim = self.lf_group_dim();
        u32::try_from(u64::from(self.width()).div_ceil(dim)).unwrap_or(1)
    }

    /// The sample rectangle of LF group `id`, clipped at the frame edge.
    #[must_use]
    pub fn lf_group_rect(&self, id: LfGroupId) -> Option<Rect> {
        if id.index() >= self.num_lf_groups() {
            return None;
        }
        let per_row = u64::from(self.lf_groups_x());
        let dim = self.lf_group_dim();
        let gx = id.index() % per_row;
        let gy = id.index() / per_row;
        let x0 = u32::try_from(gx * dim).ok()?;
        let y0 = u32::try_from(gy * dim).ok()?;
        let dim = u32::try_from(dim).ok()?;
        Some(Rect {
            x0,
            y0,
            width: dim.min(self.width().saturating_sub(x0)),
            height: dim.min(self.height().saturating_sub(y0)),
        })
    }

    /// The sample rectangle of pass group `index`, clipped at the frame edge.
    ///
    /// The pass group is F.3.1's "group": `group_dim` samples per side, the
    /// unit one `PassGroup` section covers.
    #[must_use]
    pub fn group_rect(&self, index: u64) -> Option<Rect> {
        let (x0, y0, width, height) = self.base.group_rect(index)?;
        Some(Rect {
            x0,
            y0,
            width,
            height,
        })
    }

    /// The LF group that contains pass group `index`.
    #[must_use]
    pub fn lf_group_of(&self, index: u64) -> Option<LfGroupId> {
        let rect = self.group_rect(index)?;
        let dim = self.lf_group_dim();
        let per_row = u64::from(self.lf_groups_x());
        let gx = u64::from(rect.x0) / dim;
        let gy = u64::from(rect.y0) / dim;
        u32::try_from(gy * per_row + gx).ok().map(LfGroupId::new)
    }

    /// The 8x8-block grid of LF group `id` — G.2.4's `blocks_w x blocks_h`.
    #[must_use]
    pub fn lf_group_blocks(&self, id: LfGroupId) -> Option<BlockGrid> {
        let rect = self.lf_group_rect(id)?;
        Some(BlockGrid {
            width: rect.width.div_ceil(BLOCK_DIM),
            height: rect.height.div_ceil(BLOCK_DIM),
        })
    }

    /// The 64x64-sample grid of LF group `id` — the shape of `XFromY`,
    /// `BFromY` and I.6's CfL tiling.
    #[must_use]
    pub fn lf_group_cfl_tiles(&self, id: LfGroupId) -> Option<BlockGrid> {
        let rect = self.lf_group_rect(id)?;
        Some(BlockGrid {
            width: rect.width.div_ceil(64),
            height: rect.height.div_ceil(64),
        })
    }

    /// Whether F.3.1's single-section form applies.
    #[must_use]
    pub const fn is_single_section(&self) -> bool {
        self.num_groups() == 1 && self.num_passes == 1
    }

    /// The TOC section layout, in F.3.1 order.
    #[must_use]
    pub fn section_layout(&self) -> Vec<SectionKind> {
        if self.is_single_section() {
            return vec![SectionKind::Whole];
        }
        let mut out = Vec::new();
        out.push(SectionKind::LfGlobal);
        for i in 0..self.num_lf_groups() {
            out.push(SectionKind::LfGroup(LfGroupId::new(
                u32::try_from(i).unwrap_or(u32::MAX),
            )));
        }
        out.push(SectionKind::HfGlobal);
        for p in 0..self.num_passes {
            for g in 0..self.num_groups() {
                out.push(SectionKind::PassGroup {
                    pass: PassId::new(u8::try_from(p).unwrap_or(u8::MAX)),
                    group: HfGroupId::new(u32::try_from(g).unwrap_or(u32::MAX)),
                });
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_four_grids_nest_as_the_clauses_require() {
        // shift 1 is the VarDCT-typical 256x256 pass group; the LF group is
        // eight of those, 2048x2048.
        let g = VardctGeometry::new(600, 520, 1, 1).expect("legal geometry");
        assert_eq!(g.group_dim(), 256);
        assert_eq!(g.lf_group_dim(), 2048);
        assert_eq!(g.group_blocks(), 32);
        assert_eq!(g.num_groups(), 3 * 3);
        assert_eq!(g.num_lf_groups(), 1);
        assert_eq!(
            g.frame_blocks(),
            BlockGrid {
                width: 75,
                height: 65
            }
        );
        assert_eq!(
            g.lf_group_blocks(LfGroupId::new(0)),
            Some(BlockGrid {
                width: 75,
                height: 65
            })
        );
        assert_eq!(
            g.lf_group_cfl_tiles(LfGroupId::new(0)),
            Some(BlockGrid {
                width: 10,
                height: 9
            })
        );
    }

    #[test]
    fn lf_groups_tile_the_frame_exactly() {
        let g = VardctGeometry::new(5000, 3000, 1, 1).expect("legal geometry");
        assert_eq!(g.num_lf_groups(), 3 * 2);
        let mut area = 0u64;
        for i in 0..g.num_lf_groups() {
            let id = LfGroupId::new(u32::try_from(i).expect("small"));
            let r = g.lf_group_rect(id).expect("in range");
            area += u64::from(r.width) * u64::from(r.height);
        }
        assert_eq!(area, 5000 * 3000);
        assert_eq!(g.lf_group_rect(LfGroupId::new(6)), None);
    }

    #[test]
    fn the_section_layout_is_f31s_order() {
        let single = VardctGeometry::new(200, 200, 1, 1).expect("legal");
        assert_eq!(single.section_layout(), vec![SectionKind::Whole]);

        let g = VardctGeometry::new(600, 200, 1, 2).expect("legal");
        let layout = g.section_layout();
        assert_eq!(layout.first(), Some(&SectionKind::LfGlobal));
        assert_eq!(
            layout.get(1),
            Some(&SectionKind::LfGroup(LfGroupId::new(0)))
        );
        assert_eq!(layout.get(2), Some(&SectionKind::HfGlobal));
        assert_eq!(
            layout.get(3),
            Some(&SectionKind::PassGroup {
                pass: PassId::new(0),
                group: HfGroupId::new(0)
            })
        );
        // 1 + num_lf_groups + 1 + num_passes * num_groups.
        assert_eq!(layout.len() as u64, 2 + 1 + 2 * g.num_groups());
    }

    #[test]
    fn degenerate_geometry_is_rejected() {
        assert!(matches!(
            VardctGeometry::new(0, 4, 1, 1),
            Err(PlanError::OutOfRange {
                what: "frame width",
                ..
            })
        ));
        assert!(matches!(
            VardctGeometry::new(4, 4, 4, 1),
            Err(PlanError::OutOfRange {
                what: "group_size_shift",
                ..
            })
        ));
        assert!(matches!(
            VardctGeometry::new(4, 4, 1, 12),
            Err(PlanError::OutOfRange {
                what: "num_passes",
                ..
            })
        ));
    }
}
