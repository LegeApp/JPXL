//! Frame dimensions, group grids, and section indexing (18181-1 F.1, 5.3).
//!
//! # Frame dimensions
//!
//! F.1 derives the frame's pixel dimensions in three steps:
//!
//! ```text
//! width, height = !have_crop ? (size.width, size.height)
//!                            : (frame_header.width, frame_header.height)
//! if upsampling > 1:  width = ceil(width / upsampling)   (same for height)
//! if lf_level > 0:    width = ceil(width / (1 << (3 * lf_level)))
//! ```
//!
//! These are sample-grid dimensions, before `metadata.orientation`.
//!
//! # Group grids
//!
//! 5.3: channels are partitioned into naturally-aligned `group_dim x
//! group_dim` groups; groups at the right and bottom edges are smaller. F.1:
//!
//! ```text
//! num_groups    = ceil(width / group_dim)       * ceil(height / group_dim)
//! num_lf_groups = ceil(width / (group_dim * 8)) * ceil(height / (group_dim * 8))
//! ```
//!
//! Both grids are raster-ordered. The edge groups being partial is the normal
//! case, not an exception: a 300x200 frame at `group_dim == 256` is a 2x1 grid
//! whose only row is 200 tall and whose second column is 44 wide.

use jpxl_core::geometry::GroupDim;
use jpxl_core::limits::{AllocGuard, Limits};

use crate::frame::error::{FrameError, Result};

/// 18181-1 F.6: `num_passes` is `U32(1, 2, 3, 4 + u(3))`, so at most 11.
pub const MAX_NUM_PASSES: u32 = 11;

/// Bytes charged per TOC entry / section when metering a frame's geometry.
///
/// A section costs a `u64` size, a `u64` offset and bookkeeping; 32 bytes is a
/// deliberate over-estimate so the guard trips before the real allocation does.
const SECTION_BUDGET_BYTES: u64 = 32;

/// An axis-aligned rectangle of the frame's sample grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    /// Left edge, in samples from the frame origin.
    pub x0: u32,
    /// Top edge, in samples from the frame origin.
    pub y0: u32,
    /// Width in samples. Smaller than `group_dim` for a right-edge group.
    pub width: u32,
    /// Height in samples. Smaller than `group_dim` for a bottom-edge group.
    pub height: u32,
}

impl Rect {
    /// Number of samples the rectangle covers.
    #[must_use]
    pub const fn area(&self) -> u64 {
        (self.width as u64) * (self.height as u64)
    }
}

/// What a TOC section contains (18181-1 F.3.1, Table F.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionKind {
    /// The whole frame in one section, used when there is a single TOC entry.
    Everything,
    /// `LfGlobal` (G.1).
    LfGlobal,
    /// One `LfGroup` (G.2), in raster order.
    LfGroup {
        /// Raster index into the LF-group grid.
        index: u32,
    },
    /// `HfGlobal` (G.3).
    ///
    /// F.3.1 lists this section unconditionally, even though Table F.1 gates
    /// the *bundle* on `encoding == kVarDCT`. NOTE 1 to F.3.1 resolves the
    /// apparent conflict: in Modular mode the section is present but empty.
    HfGlobal,
    /// One `PassGroup` (G.4).
    PassGroup {
        /// Pass index in `[0, num_passes)`.
        pass: u32,
        /// Raster index into the group grid.
        group: u32,
    },
}

/// Inputs to [`FrameGeometry::derive`], grouped so the call site names them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupLayout {
    /// Frame width before the upsampling and LF-level divisions.
    pub width: u32,
    /// Frame height before the upsampling and LF-level divisions.
    pub height: u32,
    /// Colour-channel upsampling factor: 1, 2, 4 or 8.
    pub upsampling: u32,
    /// LF level; 0 for a frame that is not an LF frame.
    pub lf_level: u32,
    /// Group side length, from `group_size_shift`.
    pub group_dim: GroupDim,
    /// Number of passes the frame is partitioned into.
    pub num_passes: u32,
}

/// The group grids and section layout derived from a frame header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameGeometry {
    width: u32,
    height: u32,
    group_dim: u32,
    groups_x: u32,
    groups_y: u32,
    lf_groups_x: u32,
    lf_groups_y: u32,
    num_passes: u32,
}

impl FrameGeometry {
    /// Derives the geometry of a frame.
    ///
    /// `width`/`height` are the frame's dimensions *before* the `upsampling`
    /// and `lf_level` divisions; those are applied here so the caller cannot
    /// forget them.
    ///
    /// # Errors
    ///
    /// [`FrameError::FieldOutOfRange`] for a zero dimension or a `num_passes`
    /// above [`MAX_NUM_PASSES`], or [`FrameError::LimitExceeded`] if the frame
    /// exceeds `limits.max_pixels` or its section count exceeds the guard.
    pub fn derive(layout: GroupLayout, limits: &Limits, guard: &mut AllocGuard) -> Result<Self> {
        let GroupLayout {
            width,
            height,
            upsampling,
            lf_level,
            group_dim,
            num_passes,
        } = layout;

        if num_passes == 0 || num_passes > MAX_NUM_PASSES {
            return Err(FrameError::out_of_range(
                "num_passes",
                "F.6",
                u64::from(num_passes),
            ));
        }

        let (width, height) = scale_frame_dimensions(width, height, upsampling, lf_level)?;

        // The crop fields are independent of the image size, so a frame can
        // claim far more pixels than the image; check before gridding.
        let pixels = u64::from(width) * u64::from(height);
        if pixels > limits.max_pixels {
            return Err(FrameError::LimitExceeded {
                what: "frame pixel count",
                clause: "F.1",
                value: pixels,
                limit: limits.max_pixels,
            });
        }

        let dim = group_dim.get();
        let groups_x = width.div_ceil(dim);
        let groups_y = height.div_ceil(dim);

        // An LF group covers 8x8 groups' worth of samples.
        let lf_dim = u64::from(dim) * 8;
        let lf_groups_x = u32::try_from(u64::from(width).div_ceil(lf_dim))
            .map_err(|_| FrameError::out_of_range("lf_groups_x", "F.1", u64::from(width)))?;
        let lf_groups_y = u32::try_from(u64::from(height).div_ceil(lf_dim))
            .map_err(|_| FrameError::out_of_range("lf_groups_y", "F.1", u64::from(height)))?;

        let geometry = Self {
            width,
            height,
            group_dim: dim,
            groups_x,
            groups_y,
            lf_groups_x,
            lf_groups_y,
            num_passes,
        };

        // Meter the section table before anything allocates one entry per
        // section: num_groups is bounded only by the frame dimensions, and
        // num_passes multiplies it.
        let sections = geometry.num_sections();
        guard
            .charge(sections.saturating_mul(SECTION_BUDGET_BYTES))
            .map_err(FrameError::Core)?;

        Ok(geometry)
    }

    /// Frame width in samples, after upsampling and LF-level scaling.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Frame height in samples, after upsampling and LF-level scaling.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Side length of a full group in samples (`128 << group_size_shift`).
    #[must_use]
    pub const fn group_dim(&self) -> u32 {
        self.group_dim
    }

    /// Number of group columns.
    #[must_use]
    pub const fn groups_x(&self) -> u32 {
        self.groups_x
    }

    /// Number of group rows.
    #[must_use]
    pub const fn groups_y(&self) -> u32 {
        self.groups_y
    }

    /// `num_groups` (18181-1 F.1).
    #[must_use]
    pub const fn num_groups(&self) -> u64 {
        (self.groups_x as u64) * (self.groups_y as u64)
    }

    /// Number of LF-group columns.
    #[must_use]
    pub const fn lf_groups_x(&self) -> u32 {
        self.lf_groups_x
    }

    /// Number of LF-group rows.
    #[must_use]
    pub const fn lf_groups_y(&self) -> u32 {
        self.lf_groups_y
    }

    /// `num_lf_groups` (18181-1 F.1).
    #[must_use]
    pub const fn num_lf_groups(&self) -> u64 {
        (self.lf_groups_x as u64) * (self.lf_groups_y as u64)
    }

    /// Number of passes the frame is partitioned into.
    #[must_use]
    pub const fn num_passes(&self) -> u32 {
        self.num_passes
    }

    /// Whether the frame is carried in a single TOC section (18181-1 F.3.1).
    #[must_use]
    pub const fn is_single_section(&self) -> bool {
        self.num_groups() == 1 && self.num_passes == 1
    }

    /// Number of TOC entries / sections (18181-1 F.3.1).
    ///
    /// Either 1, or `LfGlobal + num_lf_groups + HfGlobal + num_groups *
    /// num_passes`.
    #[must_use]
    pub const fn num_sections(&self) -> u64 {
        if self.is_single_section() {
            1
        } else {
            2 + self.num_lf_groups() + self.num_groups() * (self.num_passes as u64)
        }
    }

    /// The rectangle covered by group `index` in raster order.
    ///
    /// Returns `None` if `index >= num_groups`.
    #[must_use]
    pub fn group_rect(&self, index: u64) -> Option<Rect> {
        tile_rect(
            index,
            self.groups_x,
            self.groups_y,
            self.group_dim,
            self.width,
            self.height,
        )
    }

    /// The rectangle covered by LF group `index` in raster order.
    ///
    /// Returns `None` if `index >= num_lf_groups`.
    #[must_use]
    pub fn lf_group_rect(&self, index: u64) -> Option<Rect> {
        let lf_dim = self.group_dim.checked_mul(8)?;
        tile_rect(
            index,
            self.lf_groups_x,
            self.lf_groups_y,
            lf_dim,
            self.width,
            self.height,
        )
    }

    /// What section `index` of the TOC contains (18181-1 F.3.1).
    ///
    /// The order is the conceptual one, before any TOC permutation is undone.
    /// Returns `None` if `index >= num_sections`.
    #[must_use]
    pub fn section_kind(&self, index: u64) -> Option<SectionKind> {
        if self.is_single_section() {
            return (index == 0).then_some(SectionKind::Everything);
        }
        if index == 0 {
            return Some(SectionKind::LfGlobal);
        }

        let num_lf = self.num_lf_groups();
        let after_lf = 1 + num_lf;
        if index < after_lf {
            let lf_index = u32::try_from(index - 1).ok()?;
            return Some(SectionKind::LfGroup { index: lf_index });
        }
        if index == after_lf {
            return Some(SectionKind::HfGlobal);
        }

        // PassGroup sections are grouped by pass: all groups of pass 0 in
        // raster order, then all groups of pass 1, and so on.
        let offset = index - after_lf - 1;
        let num_groups = self.num_groups();
        if num_groups == 0 || offset >= num_groups * u64::from(self.num_passes) {
            return None;
        }
        let pass = u32::try_from(offset / num_groups).ok()?;
        Some(SectionKind::PassGroup {
            pass,
            group: u32::try_from(offset % num_groups).ok()?,
        })
    }

    /// Every section kind in conceptual order.
    #[must_use]
    pub fn section_kinds(&self) -> Vec<SectionKind> {
        (0..self.num_sections())
            .filter_map(|i| self.section_kind(i))
            .collect()
    }
}

/// Applies the `upsampling` and `lf_level` divisions of 18181-1 F.1.
///
/// # Errors
///
/// [`FrameError::FieldOutOfRange`] if a dimension is zero, `upsampling` is not
/// one of 1/2/4/8, or `lf_level` exceeds 4.
pub fn scale_frame_dimensions(
    width: u32,
    height: u32,
    upsampling: u32,
    lf_level: u32,
) -> Result<(u32, u32)> {
    if width == 0 {
        return Err(FrameError::out_of_range("width", "F.1", 0));
    }
    if height == 0 {
        return Err(FrameError::out_of_range("height", "F.1", 0));
    }
    if !matches!(upsampling, 1 | 2 | 4 | 8) {
        return Err(FrameError::out_of_range(
            "upsampling",
            "F.2",
            u64::from(upsampling),
        ));
    }
    // lf_level is 1 + u(2), so at most 4; a larger value would make the shift
    // below undefined.
    if lf_level > 4 {
        return Err(FrameError::out_of_range(
            "lf_level",
            "F.2",
            u64::from(lf_level),
        ));
    }

    let mut w = width;
    let mut h = height;
    if upsampling > 1 {
        w = w.div_ceil(upsampling);
        h = h.div_ceil(upsampling);
    }
    if lf_level > 0 {
        // 1 << (3 * lf_level) with lf_level <= 4 is at most 2^12.
        let divisor = 1u32 << (3 * lf_level);
        w = w.div_ceil(divisor);
        h = h.div_ceil(divisor);
    }

    // Rounding up keeps both positive whenever the inputs were positive.
    Ok((w, h))
}

/// Rectangle of tile `index` in a `cols x rows` grid of `dim`-sized tiles
/// clipped to `width x height`.
fn tile_rect(index: u64, cols: u32, rows: u32, dim: u32, width: u32, height: u32) -> Option<Rect> {
    if cols == 0 || rows == 0 || index >= u64::from(cols) * u64::from(rows) {
        return None;
    }
    let col = u32::try_from(index % u64::from(cols)).ok()?;
    let row = u32::try_from(index / u64::from(cols)).ok()?;

    let x0 = col.checked_mul(dim)?;
    let y0 = row.checked_mul(dim)?;
    // The last column and row are partial whenever the frame is not an exact
    // multiple of dim, which is the common case rather than the exception.
    Some(Rect {
        x0,
        y0,
        width: dim.min(width - x0),
        height: dim.min(height - y0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry(width: u32, height: u32, shift: u8, passes: u32) -> FrameGeometry {
        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        FrameGeometry::derive(
            GroupLayout {
                width,
                height,
                upsampling: 1,
                lf_level: 0,
                group_dim: GroupDim::from_shift(shift).expect("valid shift"),
                num_passes: passes,
            },
            &limits,
            &mut guard,
        )
        .expect("valid geometry")
    }

    #[test]
    fn group_dim_follows_group_size_shift() {
        // 5.3 / F.2: group_dim = 128 << group_size_shift.
        for (shift, dim) in [(0u8, 128u32), (1, 256), (2, 512), (3, 1024)] {
            assert_eq!(geometry(4096, 4096, shift, 1).group_dim(), dim);
        }
    }

    #[test]
    fn fixture_300x200_at_256_has_partial_edges_on_both_axes() {
        // The cjxl fixture: 300x200 with the default group_dim of 256 is a
        // 2x1 grid. Column 1 is 44 wide and the single row is 200 tall, so
        // both axes are partial.
        let g = geometry(300, 200, 1, 1);
        assert_eq!((g.groups_x(), g.groups_y()), (2, 1));
        assert_eq!(g.num_groups(), 2);

        assert_eq!(
            g.group_rect(0),
            Some(Rect {
                x0: 0,
                y0: 0,
                width: 256,
                height: 200
            }),
            "the only row is partial: 200 < 256"
        );
        assert_eq!(
            g.group_rect(1),
            Some(Rect {
                x0: 256,
                y0: 0,
                width: 44,
                height: 200
            }),
            "the right column is partial: 300 - 256 = 44"
        );
        assert_eq!(g.group_rect(2), None);

        // The group rectangles tile the frame exactly.
        let covered: u64 = (0..g.num_groups())
            .filter_map(|i| g.group_rect(i))
            .map(|r| r.area())
            .sum();
        assert_eq!(covered, 300 * 200);
    }

    #[test]
    fn lf_groups_cover_eight_groups_per_axis() {
        // group_dim 256 => an LF group spans 2048 samples.
        let g = geometry(300, 200, 1, 1);
        assert_eq!(g.num_lf_groups(), 1);
        assert_eq!(
            g.lf_group_rect(0),
            Some(Rect {
                x0: 0,
                y0: 0,
                width: 300,
                height: 200
            }),
            "one LF group clipped to the whole frame"
        );
    }

    #[test]
    fn wide_frame_crosses_an_lf_group_boundary() {
        // 3000x1024 at group_dim 256: LF groups are 2048 wide, so the frame
        // spans two LF-group columns while spanning twelve group columns.
        let g = geometry(3000, 1024, 1, 1);
        assert_eq!((g.groups_x(), g.groups_y()), (12, 4));
        assert_eq!(g.num_groups(), 48);
        assert_eq!((g.lf_groups_x(), g.lf_groups_y()), (2, 1));
        assert_eq!(g.num_lf_groups(), 2);

        assert_eq!(
            g.lf_group_rect(0),
            Some(Rect {
                x0: 0,
                y0: 0,
                width: 2048,
                height: 1024
            })
        );
        assert_eq!(
            g.lf_group_rect(1),
            Some(Rect {
                x0: 2048,
                y0: 0,
                width: 952,
                height: 1024
            }),
            "3000 - 2048 = 952"
        );

        // Group column 11 is the partial one: 3000 - 11 * 256 = 184.
        let last_in_row = g.group_rect(11).expect("group 11 exists");
        assert_eq!((last_in_row.x0, last_in_row.width), (2816, 184));
    }

    #[test]
    fn exact_multiple_has_no_partial_edge() {
        let g = geometry(512, 512, 1, 1);
        assert_eq!(g.num_groups(), 4);
        for i in 0..4 {
            let r = g.group_rect(i).expect("exists");
            assert_eq!((r.width, r.height), (256, 256));
        }
    }

    #[test]
    fn single_group_single_pass_is_one_section() {
        let g = geometry(200, 200, 1, 1);
        assert_eq!(g.num_groups(), 1);
        assert!(g.is_single_section());
        assert_eq!(g.num_sections(), 1);
        assert_eq!(g.section_kind(0), Some(SectionKind::Everything));
        assert_eq!(g.section_kind(1), None);
    }

    #[test]
    fn multi_group_section_order_matches_f31() {
        // 300x200 => 2 groups, 1 LF group; with 1 pass that is
        // LfGlobal, LfGroup 0, HfGlobal, PassGroup(0,0), PassGroup(0,1).
        let g = geometry(300, 200, 1, 1);
        assert!(!g.is_single_section());
        assert_eq!(g.num_sections(), 2 + 1 + 2);
        assert_eq!(
            g.section_kinds(),
            vec![
                SectionKind::LfGlobal,
                SectionKind::LfGroup { index: 0 },
                SectionKind::HfGlobal,
                SectionKind::PassGroup { pass: 0, group: 0 },
                SectionKind::PassGroup { pass: 0, group: 1 },
            ]
        );
    }

    #[test]
    fn pass_groups_are_ordered_by_pass_then_group() {
        // F.3.1: "The first num_groups PassGroup sections are the groups (in
        // raster order) of the first pass, followed by all groups of the
        // second pass, and so on."
        let g = geometry(300, 200, 1, 3);
        assert_eq!(g.num_sections(), 2 + 1 + 2 * 3);
        assert_eq!(
            g.section_kinds(),
            vec![
                SectionKind::LfGlobal,
                SectionKind::LfGroup { index: 0 },
                SectionKind::HfGlobal,
                SectionKind::PassGroup { pass: 0, group: 0 },
                SectionKind::PassGroup { pass: 0, group: 1 },
                SectionKind::PassGroup { pass: 1, group: 0 },
                SectionKind::PassGroup { pass: 1, group: 1 },
                SectionKind::PassGroup { pass: 2, group: 0 },
                SectionKind::PassGroup { pass: 2, group: 1 },
            ]
        );
    }

    #[test]
    fn single_group_but_multiple_passes_is_not_single_section() {
        // F.3.1 requires BOTH num_groups == 1 and num_passes == 1.
        let g = geometry(200, 200, 1, 2);
        assert!(!g.is_single_section());
        // 1 group x 2 passes, plus LfGlobal, one LF group, and HfGlobal.
        assert_eq!(g.num_sections(), 5);
    }

    #[test]
    fn hf_global_section_exists_even_for_modular() {
        // F.3.1 NOTE 1: the section is listed unconditionally and is simply
        // empty in Modular mode. Geometry does not depend on the encoding.
        let g = geometry(300, 200, 1, 1);
        assert_eq!(g.section_kind(2), Some(SectionKind::HfGlobal));
    }

    #[test]
    fn upsampling_divides_the_frame_dimensions() {
        // F.1: if upsampling > 1, width = ceil(width / upsampling).
        assert_eq!(
            scale_frame_dimensions(300, 200, 2, 0).expect("valid"),
            (150, 100)
        );
        assert_eq!(
            scale_frame_dimensions(301, 201, 2, 0).expect("valid"),
            (151, 101),
            "the division rounds up"
        );
        assert_eq!(
            scale_frame_dimensions(300, 200, 8, 0).expect("valid"),
            (38, 25)
        );
    }

    #[test]
    fn lf_level_divides_by_eight_per_level() {
        // F.1: width = ceil(width / (1 << (3 * lf_level))).
        assert_eq!(
            scale_frame_dimensions(2048, 1024, 1, 1).expect("valid"),
            (256, 128)
        );
        assert_eq!(
            scale_frame_dimensions(2048, 1024, 1, 2).expect("valid"),
            (32, 16)
        );
        // Rounding up keeps a tiny frame from collapsing to zero.
        assert_eq!(scale_frame_dimensions(5, 5, 1, 1).expect("valid"), (1, 1));
    }

    #[test]
    fn upsampling_and_lf_level_compose_in_order() {
        // Upsampling first, then the LF-level division.
        assert_eq!(
            scale_frame_dimensions(1024, 1024, 2, 1).expect("valid"),
            (64, 64)
        );
    }

    #[test]
    fn invalid_scaling_inputs_rejected() {
        assert!(scale_frame_dimensions(0, 10, 1, 0).is_err());
        assert!(scale_frame_dimensions(10, 0, 1, 0).is_err());
        assert!(
            scale_frame_dimensions(10, 10, 3, 0).is_err(),
            "3 is not 1/2/4/8"
        );
        assert!(
            scale_frame_dimensions(10, 10, 1, 5).is_err(),
            "lf_level > 4"
        );
    }

    #[test]
    fn zero_or_excessive_passes_rejected() {
        let limits = Limits::default();
        let dim = GroupDim::from_shift(1).expect("valid");
        for passes in [0u32, MAX_NUM_PASSES + 1, u32::MAX] {
            let mut guard = AllocGuard::new(&limits);
            assert!(
                FrameGeometry::derive(
                    GroupLayout {
                        width: 256,
                        height: 256,
                        upsampling: 1,
                        lf_level: 0,
                        group_dim: dim,
                        num_passes: passes,
                    },
                    &limits,
                    &mut guard,
                )
                .is_err(),
                "num_passes {passes} must be rejected"
            );
        }
    }

    #[test]
    fn oversized_frame_is_rejected_before_gridding() {
        // A crop can claim dimensions unrelated to the image size.
        let limits = Limits {
            max_pixels: 1 << 20,
            ..Limits::default()
        };
        let mut guard = AllocGuard::new(&limits);
        let err = FrameGeometry::derive(
            GroupLayout {
                width: 1 << 20,
                height: 1 << 20,
                upsampling: 1,
                lf_level: 0,
                group_dim: GroupDim::from_shift(0).expect("valid"),
                num_passes: 1,
            },
            &limits,
            &mut guard,
        )
        .expect_err("must exceed max_pixels");
        assert!(matches!(err, FrameError::LimitExceeded { .. }));
    }

    #[test]
    fn section_table_is_metered() {
        // A large frame at the smallest group_dim with many passes produces a
        // huge section table; the guard must trip before it is built.
        let limits = Limits {
            max_alloc_bytes: 1024,
            ..Limits::default()
        };
        let mut guard = AllocGuard::new(&limits);
        let err = FrameGeometry::derive(
            GroupLayout {
                width: 16384,
                height: 16384,
                upsampling: 1,
                lf_level: 0,
                group_dim: GroupDim::from_shift(0).expect("valid"),
                num_passes: 11,
            },
            &limits,
            &mut guard,
        )
        .expect_err("section table must be metered");
        assert!(matches!(err, FrameError::Core(_)));
    }

    #[test]
    fn one_sample_frame() {
        let g = geometry(1, 1, 0, 1);
        assert_eq!(g.num_groups(), 1);
        assert_eq!(g.num_lf_groups(), 1);
        assert_eq!(
            g.group_rect(0),
            Some(Rect {
                x0: 0,
                y0: 0,
                width: 1,
                height: 1
            })
        );
    }
}
