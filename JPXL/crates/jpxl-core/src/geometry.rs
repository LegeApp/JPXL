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
}
