//! Checked newtypes for image geometry.
//!
//! Raw `u32` dimensions and `u8` bit depths read straight from a codestream are
//! easy to mix up and easy to trust too far. The types here can only be built
//! through constructors that validate the value, so a [`Width`] in hand is
//! already known to be non-zero and a [`BitDepth`] is already known to be in
//! range. Sizes derived from them go through [`pixel_count`], which is checked
//! against [`Limits`](crate::limits::Limits) rather than wrapping.

use crate::error::{JpxlError, Result};
use crate::limits::Limits;

/// Largest dimension JPEG XL can signal on either axis.
///
/// ISO/IEC 18181-1 codes dimensions in at most 30 bits.
pub const MAX_DIMENSION: u32 = 1 << 30;

/// A validated image width in pixels: `1..=`[`MAX_DIMENSION`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Width(u32);

/// A validated image height in pixels: `1..=`[`MAX_DIMENSION`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Height(u32);

/// A validated bits-per-sample value: `1..=32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BitDepth(u8);

impl Width {
    /// Validates a width.
    ///
    /// # Errors
    ///
    /// [`JpxlError::InvalidHeader`] if `value` is zero or above
    /// [`MAX_DIMENSION`].
    pub fn new(value: u32) -> Result<Self> {
        if value == 0 || value > MAX_DIMENSION {
            return Err(JpxlError::InvalidHeader(format!(
                "width {value} is outside the representable range 1..={MAX_DIMENSION}"
            )));
        }
        Ok(Self(value))
    }

    /// The validated value.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl Height {
    /// Validates a height.
    ///
    /// # Errors
    ///
    /// [`JpxlError::InvalidHeader`] if `value` is zero or above
    /// [`MAX_DIMENSION`].
    pub fn new(value: u32) -> Result<Self> {
        if value == 0 || value > MAX_DIMENSION {
            return Err(JpxlError::InvalidHeader(format!(
                "height {value} is outside the representable range 1..={MAX_DIMENSION}"
            )));
        }
        Ok(Self(value))
    }

    /// The validated value.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl BitDepth {
    /// Validates a bit depth in `1..=32`.
    ///
    /// # Errors
    ///
    /// [`JpxlError::InvalidHeader`] if `value` is outside `1..=32`.
    pub fn new(value: u8) -> Result<Self> {
        if !(1..=32).contains(&value) {
            return Err(JpxlError::InvalidHeader(format!(
                "bit depth {value} is outside the supported range 1..=32"
            )));
        }
        Ok(Self(value))
    }

    /// The validated value.
    #[must_use]
    pub const fn get(self) -> u8 {
        self.0
    }

    /// Largest unsigned sample value representable at this depth.
    #[must_use]
    pub const fn max_sample(self) -> u32 {
        // Depth is 1..=32, so the shift stays in 0..=31.
        u32::MAX >> (32 - self.0)
    }
}

/// The side length of a coding group, in pixels.
///
/// JPEG XL signals `group_size_shift` in two bits; the group is square with
/// side `128 << shift`, so 128, 256, 512, or 1024 pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum GroupDim {
    /// `group_size_shift == 0`.
    D128 = 0,
    /// `group_size_shift == 1`.
    D256 = 1,
    /// `group_size_shift == 2`.
    D512 = 2,
    /// `group_size_shift == 3`.
    D1024 = 3,
}

impl GroupDim {
    /// Decodes a two-bit `group_size_shift` field.
    ///
    /// # Errors
    ///
    /// [`JpxlError::InvalidHeader`] if `shift > 3`.
    pub fn from_shift(shift: u8) -> Result<Self> {
        match shift {
            0 => Ok(Self::D128),
            1 => Ok(Self::D256),
            2 => Ok(Self::D512),
            3 => Ok(Self::D1024),
            _ => Err(JpxlError::InvalidHeader(format!(
                "group_size_shift {shift} is outside the encodable range 0..=3"
            ))),
        }
    }

    /// The `group_size_shift` this dimension was signalled by.
    #[must_use]
    pub const fn shift(self) -> u8 {
        self as u8
    }

    /// The group side length in pixels: 128, 256, 512, or 1024.
    #[must_use]
    pub const fn get(self) -> u32 {
        128u32 << (self as u8)
    }

    /// Number of groups needed to cover `extent` pixels along one axis.
    #[must_use]
    pub const fn groups_covering(self, extent: u32) -> u32 {
        extent.div_ceil(self.get())
    }
}

// ---------------------------------------------------------------------------
// Block coordinates (VarDCT)
// ---------------------------------------------------------------------------

/// Side of a VarDCT coding block, in pixels.
///
/// The whole of 18181-1 Annex I is phrased in 8x8 blocks: `DctSelect` is stored
/// one entry per block, the LF image is the frame downsampled by this factor,
/// and I.3.2's `bwidth`/`bheight` are multiples of it.
pub const BLOCK_DIM: u32 = 8;

/// Side of an LF group, in 8x8 blocks.
///
/// 5.3 and the G.2.3 note put an LF group at `group_dim` LF samples on a side,
/// and `group_size_shift` is only read for `kModular` frames (F.2), so a VarDCT
/// frame always has `group_dim == 256`. This is that number in *blocks*, which
/// is the unit [`LfBlockPos`] counts in.
pub const LF_GROUP_BLOCKS: u32 = 256;

/// A pixel position inside a frame.
///
/// Exists so that a pixel coordinate cannot be handed to a function expecting a
/// block coordinate. The previous project lost weeks to exactly that confusion,
/// so the conversions are explicit and named.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PixelPos {
    x: u32,
    y: u32,
}

/// A frame-absolute 8x8-block position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockPos {
    bx: u32,
    by: u32,
}

/// An 8x8-block position relative to the origin of an LF group.
///
/// G.2.4's greedy varblock placement walks this coordinate space, and every
/// varblock must lie wholly inside one LF group, so mixing it with [`BlockPos`]
/// silently moves varblocks between groups.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LfBlockPos {
    bx: u32,
    by: u32,
}

impl PixelPos {
    /// A pixel position. Any `u32` pair is representable.
    #[must_use]
    pub const fn new(x: u32, y: u32) -> Self {
        Self { x, y }
    }

    /// Column, in pixels.
    #[must_use]
    pub const fn x(self) -> u32 {
        self.x
    }

    /// Row, in pixels.
    #[must_use]
    pub const fn y(self) -> u32 {
        self.y
    }

    /// The 8x8 block containing this pixel.
    #[must_use]
    pub const fn block(self) -> BlockPos {
        BlockPos {
            bx: self.x / BLOCK_DIM,
            by: self.y / BLOCK_DIM,
        }
    }

    /// Offset of this pixel inside its block, as `(x, y)` in `0..8`.
    #[must_use]
    pub const fn offset_in_block(self) -> (u32, u32) {
        (self.x % BLOCK_DIM, self.y % BLOCK_DIM)
    }
}

impl BlockPos {
    /// A frame-absolute block position.
    #[must_use]
    pub const fn new(bx: u32, by: u32) -> Self {
        Self { bx, by }
    }

    /// Block column.
    #[must_use]
    pub const fn bx(self) -> u32 {
        self.bx
    }

    /// Block row.
    #[must_use]
    pub const fn by(self) -> u32 {
        self.by
    }

    /// The top-left pixel of this block.
    ///
    /// # Errors
    ///
    /// [`JpxlError::InvalidHeader`] if the pixel coordinate would not fit in
    /// `u32`. Block indices come from stream-controlled geometry, so this
    /// multiplication is checked rather than wrapped.
    pub fn origin_pixel(self) -> Result<PixelPos> {
        let scale = |v: u32, axis: &str| {
            v.checked_mul(BLOCK_DIM).ok_or_else(|| {
                JpxlError::InvalidHeader(format!("block {axis} index {v} overflows a pixel index"))
            })
        };
        Ok(PixelPos::new(
            scale(self.bx, "column")?,
            scale(self.by, "row")?,
        ))
    }

    /// This block's position relative to the LF group whose origin is
    /// `group_origin`.
    ///
    /// Returns `None` if the block lies before the group origin on either axis.
    #[must_use]
    pub fn relative_to(self, group_origin: Self) -> Option<LfBlockPos> {
        Some(LfBlockPos {
            bx: self.bx.checked_sub(group_origin.bx)?,
            by: self.by.checked_sub(group_origin.by)?,
        })
    }

    /// The origin of the LF group containing this block.
    #[must_use]
    pub const fn lf_group_origin(self) -> Self {
        Self {
            bx: self.bx - self.bx % LF_GROUP_BLOCKS,
            by: self.by - self.by % LF_GROUP_BLOCKS,
        }
    }
}

impl LfBlockPos {
    /// A block position relative to an LF group origin.
    #[must_use]
    pub const fn new(bx: u32, by: u32) -> Self {
        Self { bx, by }
    }

    /// Block column within the LF group.
    #[must_use]
    pub const fn bx(self) -> u32 {
        self.bx
    }

    /// Block row within the LF group.
    #[must_use]
    pub const fn by(self) -> u32 {
        self.by
    }

    /// Promotes back to a frame-absolute block position.
    ///
    /// # Errors
    ///
    /// [`JpxlError::InvalidHeader`] if the sum overflows `u32`.
    pub fn to_frame(self, group_origin: BlockPos) -> Result<BlockPos> {
        let add = |a: u32, b: u32, axis: &str| {
            a.checked_add(b).ok_or_else(|| {
                JpxlError::InvalidHeader(format!("LF-group-relative block {axis} index overflows"))
            })
        };
        Ok(BlockPos::new(
            add(group_origin.bx, self.bx, "column")?,
            add(group_origin.by, self.by, "row")?,
        ))
    }
}

/// Total pixels in a `width` x `height` image, checked against `limits`.
///
/// # Errors
///
/// [`JpxlError::LimitExceeded`] if the product is above
/// [`Limits::max_pixels`]. The multiply itself is done in `u64`, which cannot
/// overflow for two values bounded by [`MAX_DIMENSION`].
pub fn pixel_count(width: Width, height: Height, limits: &Limits) -> Result<u64> {
    let count = u64::from(width.get()) * u64::from(height.get());
    if count > limits.max_pixels {
        return Err(JpxlError::LimitExceeded(format!(
            "image is {}x{} = {count} pixels, over the max_pixels limit of {}",
            width.get(),
            height.get(),
            limits.max_pixels
        )));
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_valid_dimensions() {
        assert_eq!(Width::new(1).expect("1 is valid").get(), 1);
        assert_eq!(Height::new(4096).expect("4096 is valid").get(), 4096);
        assert_eq!(
            Width::new(MAX_DIMENSION).expect("max is valid").get(),
            MAX_DIMENSION
        );
    }

    #[test]
    fn rejects_zero_and_oversized_dimensions() {
        assert!(Width::new(0).is_err());
        assert!(Height::new(0).is_err());
        assert!(Width::new(MAX_DIMENSION + 1).is_err());
        assert!(Height::new(u32::MAX).is_err());
    }

    #[test]
    fn bit_depth_range() {
        assert_eq!(BitDepth::new(1).expect("1 is valid").get(), 1);
        assert_eq!(BitDepth::new(32).expect("32 is valid").get(), 32);
        assert!(BitDepth::new(0).is_err());
        assert!(BitDepth::new(33).is_err());
    }

    #[test]
    fn bit_depth_max_sample() {
        assert_eq!(BitDepth::new(1).expect("valid").max_sample(), 1);
        assert_eq!(BitDepth::new(8).expect("valid").max_sample(), 255);
        assert_eq!(BitDepth::new(32).expect("valid").max_sample(), u32::MAX);
    }

    #[test]
    fn group_dim_maps_shift_to_side() {
        for (shift, side) in [(0u8, 128u32), (1, 256), (2, 512), (3, 1024)] {
            let dim = GroupDim::from_shift(shift).expect("shift 0..=3 is valid");
            assert_eq!(dim.get(), side);
            assert_eq!(dim.shift(), shift);
        }
        assert!(GroupDim::from_shift(4).is_err());
    }

    #[test]
    fn group_coverage_rounds_up() {
        let dim = GroupDim::D256;
        assert_eq!(dim.groups_covering(0), 0);
        assert_eq!(dim.groups_covering(1), 1);
        assert_eq!(dim.groups_covering(256), 1);
        assert_eq!(dim.groups_covering(257), 2);
    }

    #[test]
    fn pixel_count_checks_limits() {
        let w = Width::new(4000).expect("valid");
        let h = Height::new(3000).expect("valid");
        assert_eq!(
            pixel_count(w, h, &Limits::default()).expect("well under default limit"),
            12_000_000
        );

        let tight = Limits {
            max_pixels: 11_999_999,
            ..Limits::default()
        };
        let err = pixel_count(w, h, &tight).expect_err("must exceed the tight limit");
        assert!(matches!(err, JpxlError::LimitExceeded(_)));
    }

    #[test]
    fn pixel_count_does_not_overflow_at_extremes() {
        let w = Width::new(MAX_DIMENSION).expect("valid");
        let h = Height::new(MAX_DIMENSION).expect("valid");
        assert_eq!(
            pixel_count(w, h, &Limits::relaxed()).expect("relaxed limits allow it"),
            1u64 << 60
        );
    }

    /// Pixel-to-block and block-to-pixel are inverse where they should be, and
    /// the offset within the block is recovered. Proves the two coordinate
    /// spaces cannot be silently interchanged: the conversion is a named call,
    /// not an implicit `usize`.
    #[test]
    fn pixel_and_block_coordinates_convert() {
        let p = PixelPos::new(17, 8);
        assert_eq!(p.block(), BlockPos::new(2, 1));
        assert_eq!(p.offset_in_block(), (1, 0));
        assert_eq!(
            BlockPos::new(2, 1).origin_pixel().expect("in range"),
            PixelPos::new(16, 8)
        );
        assert_eq!(PixelPos::new(0, 0).block(), BlockPos::new(0, 0));
    }

    /// Block indices are stream-controlled, so the multiply into pixel space is
    /// checked rather than wrapped.
    #[test]
    fn block_origin_rejects_overflow() {
        assert!(BlockPos::new(u32::MAX, 0).origin_pixel().is_err());
        assert!(BlockPos::new(0, u32::MAX / 4).origin_pixel().is_err());
    }

    /// LF-group-relative coordinates round-trip through the group origin, and a
    /// block before the origin has no relative position at all rather than
    /// wrapping to a huge one.
    #[test]
    fn lf_group_relative_coordinates_round_trip() {
        let origin = BlockPos::new(256, 512);
        let block = BlockPos::new(260, 512);
        let rel = block.relative_to(origin).expect("inside the group");
        assert_eq!((rel.bx(), rel.by()), (4, 0));
        assert_eq!(rel.to_frame(origin).expect("in range"), block);

        assert!(BlockPos::new(255, 512).relative_to(origin).is_none());
        assert!(
            LfBlockPos::new(1, 0)
                .to_frame(BlockPos::new(u32::MAX, 0))
                .is_err()
        );
    }

    /// The LF-group origin of a block is that block rounded down to a multiple
    /// of `LF_GROUP_BLOCKS`, which is what makes `relative_to` total for blocks
    /// in their own group.
    #[test]
    fn lf_group_origin_rounds_down() {
        assert_eq!(
            BlockPos::new(300, 5).lf_group_origin(),
            BlockPos::new(256, 0)
        );
        assert_eq!(BlockPos::new(0, 0).lf_group_origin(), BlockPos::new(0, 0));
        let b = BlockPos::new(1000, 700);
        assert!(b.relative_to(b.lf_group_origin()).is_some());
    }
}
