//! Image dimensions: the `SizeHeader` (18181-1 D.2) and `PreviewHeader`
//! (18181-1 D.3.3) bundles.
//!
//! Both bundles have the same four-way shape — a `div8` flag crossed with a
//! `ratio` selector — and differ only in the distributions used for each field:
//!
//! ```text
//! Table D.2 — SizeHeader bundle
//! condition            type                                            default     name
//!                      Bool()                                          false       div8
//! div8                 1 + u(5)                                        0           h_div8
//! !div8                U32(1+u(9), 1+u(13), 1+u(18), 1+u(30))          8 * h_div8  height
//!                      u(3)                                            0           ratio
//! div8 and !ratio      1 + u(5)                                        0           w_div8
//! !div8 and !ratio     U32(1+u(9), 1+u(13), 1+u(18), 1+u(30))          d_width     width
//! ```
//!
//! `d_width = (ratio == 0 ? 8 * w_div8 : AspectRatio(height, ratio))`.
//!
//! Note that `ratio` has a blank condition: it is read in both the `div8` and
//! `!div8` paths. When `ratio != 0` the width is not stored at all, it is
//! derived from the height — which is why a 4:3 image can be signalled in
//! fewer bits than either dimension would take on its own.

use jpxl_bitstream::{BitReader, U32Dist, U32Spec, read_bool, read_u32, trace_field};
use jpxl_core::geometry::{Height, Width, pixel_count};
use jpxl_core::limits::Limits;

use crate::error::{DecodeError, Result};

/// 18181-1 D.2: `U32(1 + u(9), 1 + u(13), 1 + u(18), 1 + u(30))`.
const SIZE_DIM_SPEC: U32Spec = U32Spec::new([
    U32Dist::BitsOffset { bits: 9, offset: 1 },
    U32Dist::BitsOffset {
        bits: 13,
        offset: 1,
    },
    U32Dist::BitsOffset {
        bits: 18,
        offset: 1,
    },
    U32Dist::BitsOffset {
        bits: 30,
        offset: 1,
    },
]);

/// 18181-1 D.5: `U32(16, 32, 1 + u(5), 33 + u(9))`.
const PREVIEW_DIV8_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(16),
    U32Dist::Val(32),
    U32Dist::BitsOffset { bits: 5, offset: 1 },
    U32Dist::BitsOffset {
        bits: 9,
        offset: 33,
    },
]);

/// 18181-1 D.5: `U32(1 + u(6), 65 + u(8), 321 + u(10), 1345 + u(12))`.
const PREVIEW_DIM_SPEC: U32Spec = U32Spec::new([
    U32Dist::BitsOffset { bits: 6, offset: 1 },
    U32Dist::BitsOffset {
        bits: 8,
        offset: 65,
    },
    U32Dist::BitsOffset {
        bits: 10,
        offset: 321,
    },
    U32Dist::BitsOffset {
        bits: 12,
        offset: 1345,
    },
]);

/// 18181-1 D.3.3: "The preview width and height do not exceed 4096."
pub const MAX_PREVIEW_DIMENSION: u32 = 4096;

/// Applies `AspectRatio(height, ratio)` from 18181-1 D.2.
///
/// The seven ratios are 1:1, 6:5, 4:3, 3:2, 16:9, 5:4 and 2:1. `ratio == 0`
/// means the width was stored explicitly and has no aspect-ratio form, so it
/// is rejected here rather than silently treated as square.
///
/// The arithmetic is done in `u64`: `height` reaches `2^30`, and `height * 16`
/// would overflow `u32`.
///
/// # Errors
///
/// [`DecodeError::FieldOutOfRange`] if `ratio` is 0 or above 7.
pub fn aspect_ratio_width(height: u32, ratio: u32) -> Result<u64> {
    let h = u64::from(height);
    let width = match ratio {
        1 => h,
        2 => h * 6 / 5,
        3 => h * 4 / 3,
        4 => h * 3 / 2,
        5 => h * 16 / 9,
        6 => h * 5 / 4,
        7 => h * 2,
        _ => return Err(DecodeError::out_of_range("ratio", "D.2", u64::from(ratio))),
    };
    Ok(width)
}

/// A decoded `SizeHeader` (18181-1 D.2).
///
/// Both dimensions are validated on construction, so an existing `SizeHeader`
/// always describes a nonzero image within [`Limits`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SizeHeader {
    width: Width,
    height: Height,
    div8: bool,
    ratio: u32,
}

impl SizeHeader {
    /// Image width in pixels.
    #[must_use]
    pub const fn width(&self) -> Width {
        self.width
    }

    /// Image height in pixels.
    #[must_use]
    pub const fn height(&self) -> Height {
        self.height
    }

    /// Whether both dimensions were signalled in units of 8 pixels.
    #[must_use]
    pub const fn is_div8(&self) -> bool {
        self.div8
    }

    /// The `ratio` selector; 0 means the width was stored explicitly.
    #[must_use]
    pub const fn ratio(&self) -> u32 {
        self.ratio
    }

    /// Total pixel count, already checked against the limits at parse time.
    #[must_use]
    pub const fn pixel_count(&self) -> u64 {
        (self.width.get() as u64) * (self.height.get() as u64)
    }
}

/// Reads a `SizeHeader` bundle (18181-1 D.2).
///
/// # Errors
///
/// [`DecodeError::Core`] if a dimension is out of the representable range or
/// the pixel count exceeds `limits`, or a bitstream error on truncation.
pub fn read_size_header(reader: &mut BitReader<'_>, limits: &Limits) -> Result<SizeHeader> {
    let div8 = trace_field!(reader, "size.div8", read_bool(reader))?;

    let height = if div8 {
        // 1 + u(5) => 1..=32, so height is 8..=256.
        let h_div8 = trace_field!(reader, "size.h_div8", reader.read_bits(5))? + 1;
        h_div8 * 8
    } else {
        trace_field!(reader, "size.height", read_u32(reader, &SIZE_DIM_SPEC))?
    };

    // Blank condition: read in both branches.
    let ratio = trace_field!(reader, "size.ratio", reader.read_bits(3))?;

    let width = if ratio == 0 {
        if div8 {
            let w_div8 = trace_field!(reader, "size.w_div8", reader.read_bits(5))? + 1;
            u64::from(w_div8 * 8)
        } else {
            u64::from(trace_field!(
                reader,
                "size.width",
                read_u32(reader, &SIZE_DIM_SPEC)
            )?)
        }
    } else {
        aspect_ratio_width(height, ratio)?
    };

    finish_size(width, height, div8, ratio, limits)
}

/// Validates decoded dimensions and meters them against `limits`.
fn finish_size(
    width: u64,
    height: u32,
    div8: bool,
    ratio: u32,
    limits: &Limits,
) -> Result<SizeHeader> {
    // An aspect-ratio width can exceed 2^32 before Width::new ever sees it.
    let width = u32::try_from(width)
        .map_err(|_| DecodeError::out_of_range("width", "D.2", width))
        .and_then(|w| Width::new(w).map_err(DecodeError::Core))?;
    let height = Height::new(height).map_err(DecodeError::Core)?;

    pixel_count(width, height, limits).map_err(DecodeError::Core)?;

    Ok(SizeHeader {
        width,
        height,
        div8,
        ratio,
    })
}

/// A decoded `PreviewHeader` (18181-1 D.3.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreviewHeader {
    width: u32,
    height: u32,
    div8: bool,
    ratio: u32,
}

impl PreviewHeader {
    /// Preview width in pixels; at most [`MAX_PREVIEW_DIMENSION`].
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Preview height in pixels; at most [`MAX_PREVIEW_DIMENSION`].
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Whether both dimensions were signalled in units of 8 pixels.
    #[must_use]
    pub const fn is_div8(&self) -> bool {
        self.div8
    }

    /// The `ratio` selector; 0 means the width was stored explicitly.
    #[must_use]
    pub const fn ratio(&self) -> u32 {
        self.ratio
    }
}

/// Reads a `PreviewHeader` bundle (18181-1 D.3.3).
///
/// # Errors
///
/// [`DecodeError::FieldOutOfRange`] if either dimension is zero or above
/// [`MAX_PREVIEW_DIMENSION`], or a bitstream error on truncation.
pub fn read_preview_header(reader: &mut BitReader<'_>) -> Result<PreviewHeader> {
    let div8 = trace_field!(reader, "preview.div8", read_bool(reader))?;

    let height = if div8 {
        let h_div8 = trace_field!(
            reader,
            "preview.h_div8",
            read_u32(reader, &PREVIEW_DIV8_SPEC)
        )?;
        u64::from(h_div8) * 8
    } else {
        u64::from(trace_field!(
            reader,
            "preview.height",
            read_u32(reader, &PREVIEW_DIM_SPEC)
        )?)
    };

    let ratio = trace_field!(reader, "preview.ratio", reader.read_bits(3))?;

    let width = if ratio == 0 {
        if div8 {
            let w_div8 = trace_field!(
                reader,
                "preview.w_div8",
                read_u32(reader, &PREVIEW_DIV8_SPEC)
            )?;
            u64::from(w_div8) * 8
        } else {
            u64::from(trace_field!(
                reader,
                "preview.width",
                read_u32(reader, &PREVIEW_DIM_SPEC)
            )?)
        }
    } else {
        let height = u32::try_from(height)
            .map_err(|_| DecodeError::out_of_range("preview.height", "D.3.3", height))?;
        aspect_ratio_width(height, ratio)?
    };

    let width = check_preview_dim(width, "preview.width")?;
    let height = check_preview_dim(height, "preview.height")?;

    Ok(PreviewHeader {
        width,
        height,
        div8,
        ratio,
    })
}

/// Enforces 18181-1 D.3.3's "do not exceed 4096" on a preview dimension.
fn check_preview_dim(value: u64, field: &'static str) -> Result<u32> {
    let value =
        u32::try_from(value).map_err(|_| DecodeError::out_of_range(field, "D.3.3", value))?;
    if value == 0 || value > MAX_PREVIEW_DIMENSION {
        return Err(DecodeError::out_of_range(field, "D.3.3", u64::from(value)));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testsupport::BitWriter;

    fn read_size(bytes: &[u8]) -> Result<SizeHeader> {
        let mut r = BitReader::new(bytes);
        read_size_header(&mut r, &Limits::default())
    }

    #[test]
    fn div8_path_smallest_image() {
        // D.2: div8 = true, h_div8 = 1 + u(5) with payload 0 => height 8.
        // ratio = 0, w_div8 = 1 + u(5) with payload 0 => width 8.
        // Bits: 1 | 00000 | 000 | 00000  (14 bits)
        let mut w = BitWriter::new();
        w.bool(true).u(5, 0).u(3, 0).u(5, 0);
        let size = read_size(&w.finish_padded(1)).expect("valid");

        assert_eq!(size.width().get(), 8);
        assert_eq!(size.height().get(), 8);
        assert!(size.is_div8());
        assert_eq!(size.ratio(), 0);
        assert_eq!(size.pixel_count(), 64);
    }

    #[test]
    fn div8_path_largest_image() {
        // h_div8 = 1 + 31 = 32 => height 256; same for width.
        let mut w = BitWriter::new();
        w.bool(true).u(5, 31).u(3, 0).u(5, 31);
        let size = read_size(&w.finish_padded(1)).expect("valid");
        assert_eq!(size.width().get(), 256);
        assert_eq!(size.height().get(), 256);
    }

    #[test]
    fn non_div8_path_with_explicit_width() {
        // height: U32 selector 0 => 1 + u(9), payload 639 => 640.
        // ratio 0; width: selector 0 => 1 + u(9), payload 479 => 480.
        let mut w = BitWriter::new();
        w.bool(false)
            .u32_field(1, 13, 639)
            .u(3, 0)
            .u32_field(0, 9, 479);
        let size = read_size(&w.finish_padded(1)).expect("valid");

        assert_eq!(size.height().get(), 640);
        assert_eq!(size.width().get(), 480);
        assert!(!size.is_div8());
    }

    #[test]
    fn non_div8_uses_wider_distributions() {
        // selector 2 => 1 + u(18); payload 3999 => height 4000.
        // selector 1 => 1 + u(13); payload 2999 => width 3000.
        let mut w = BitWriter::new();
        w.bool(false)
            .u32_field(2, 18, 3999)
            .u(3, 0)
            .u32_field(1, 13, 2999);
        let size = read_size(&w.finish_padded(2)).expect("valid");
        assert_eq!(size.height().get(), 4000);
        assert_eq!(size.width().get(), 3000);
    }

    #[test]
    fn every_aspect_ratio_matches_the_clause() {
        // D.2 AspectRatio: 1:1, 6:5, 4:3, 3:2, 16:9, 5:4, 2:1.
        let cases = [
            (1u32, 1080u64),
            (2, 1080 * 6 / 5),
            (3, 1080 * 4 / 3),
            (4, 1080 * 3 / 2),
            (5, 1080 * 16 / 9),
            (6, 1080 * 5 / 4),
            (7, 2160),
        ];
        for (ratio, expected) in cases {
            assert_eq!(
                aspect_ratio_width(1080, ratio).expect("valid ratio"),
                expected,
                "ratio {ratio}"
            );
        }
        // 16:9 of 1080 is exactly 1920 — the common case must not drift.
        assert_eq!(aspect_ratio_width(1080, 5).expect("valid"), 1920);
    }

    #[test]
    fn aspect_ratio_path_reads_no_width_field() {
        // !div8, height = 1 + u(9) payload 1079 => 1080, ratio 5 => 16:9.
        let mut w = BitWriter::new();
        w.bool(false).u32_field(1, 13, 1079).u(3, 5);
        let expected_bits = w.bit_len();

        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let size = read_size_header(&mut r, &Limits::default()).expect("valid");

        assert_eq!(size.height().get(), 1080);
        assert_eq!(size.width().get(), 1920);
        assert_eq!(size.ratio(), 5);
        assert_eq!(
            r.total_bits_read(),
            expected_bits,
            "no width field is stored when ratio != 0"
        );
    }

    #[test]
    fn aspect_ratio_truncates_toward_zero() {
        // 6:5 of 7 is 8.4 -> Idiv gives 8.
        assert_eq!(aspect_ratio_width(7, 2).expect("valid"), 8);
        // 16:9 of 10 is 17.77 -> 17.
        assert_eq!(aspect_ratio_width(10, 5).expect("valid"), 17);
    }

    #[test]
    fn aspect_ratio_zero_and_out_of_range_rejected() {
        assert!(aspect_ratio_width(100, 0).is_err());
        assert!(aspect_ratio_width(100, 8).is_err());
    }

    #[test]
    fn aspect_ratio_cannot_overflow_u32_silently() {
        // height 2^30 with ratio 5 => 2^30 * 16 / 9, far above MAX_DIMENSION.
        // Computed in u64, then rejected by Width::new rather than wrapping.
        let big = aspect_ratio_width(1 << 30, 5).expect("u64 arithmetic does not overflow");
        assert!(
            big > (1 << 30),
            "the product exceeds MAX_DIMENSION and must not be truncated into range"
        );

        let mut w = BitWriter::new();
        w.bool(false).u32_field(3, 30, (1 << 30) - 1).u(3, 5);
        assert!(read_size(&w.finish_padded(2)).is_err());
    }

    #[test]
    fn pixel_count_limit_is_enforced() {
        // 4000x3000 = 12M pixels, rejected by a limit of 1M.
        let mut w = BitWriter::new();
        w.bool(false)
            .u32_field(2, 18, 2999)
            .u(3, 0)
            .u32_field(2, 18, 3999);
        let data = w.finish_padded(2);

        let tight = Limits {
            max_pixels: 1_000_000,
            ..Limits::default()
        };
        let mut r = BitReader::new(&data);
        assert!(read_size_header(&mut r, &tight).is_err());

        // The same bits parse under the default limits.
        let mut r = BitReader::new(&data);
        assert!(read_size_header(&mut r, &Limits::default()).is_ok());
    }

    #[test]
    fn multi_byte_crossing_field() {
        // A 30-bit payload starting at bit 3 spans five bytes.
        let mut w = BitWriter::new();
        w.bool(false)
            .u32_field(3, 30, 99_999)
            .u(3, 0)
            .u32_field(3, 30, 4_999);
        let size = read_size(&w.finish_padded(2)).expect("valid");
        assert_eq!(size.height().get(), 100_000);
        assert_eq!(size.width().get(), 5_000);
    }

    #[test]
    fn truncated_size_header_errors() {
        assert!(read_size(&[]).is_err());
        assert!(read_size(&[0b0000_0001]).is_err());
    }

    #[test]
    fn preview_div8_path() {
        // D.5: div8 = true, h_div8 = U32(16, 32, ...) selector 0 => 16.
        // height = 16 * 8 = 128. ratio 0, w_div8 selector 1 => 32 => 256.
        let mut w = BitWriter::new();
        w.bool(true).u32_field(0, 0, 0).u(3, 0).u32_field(1, 0, 0);
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let preview = read_preview_header(&mut r).expect("valid");
        assert_eq!(preview.height(), 128);
        assert_eq!(preview.width(), 256);
        assert!(preview.is_div8());
    }

    #[test]
    fn preview_non_div8_path() {
        // height: U32(1 + u(6), ...) selector 0, payload 63 => 64.
        // width: selector 1 => 65 + u(8), payload 35 => 100.
        let mut w = BitWriter::new();
        w.bool(false)
            .u32_field(0, 6, 63)
            .u(3, 0)
            .u32_field(1, 8, 35);
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let preview = read_preview_header(&mut r).expect("valid");
        assert_eq!(preview.height(), 64);
        assert_eq!(preview.width(), 100);
    }

    #[test]
    fn preview_dimension_cap_is_enforced() {
        // div8 with h_div8 = 33 + u(9) payload 511 => 544; height 4352 > 4096.
        let mut w = BitWriter::new();
        w.bool(true).u32_field(3, 9, 511).u(3, 0).u32_field(0, 0, 0);
        let data = w.finish_padded(2);
        let mut r = BitReader::new(&data);
        let err = read_preview_header(&mut r).expect_err("4352 exceeds 4096");
        assert!(matches!(err, DecodeError::FieldOutOfRange { .. }));
    }

    #[test]
    fn preview_at_the_cap_is_accepted() {
        // 512 * 8 = 4096 exactly: h_div8 = 33 + u(9) payload 479 => 512.
        let mut w = BitWriter::new();
        w.bool(true).u32_field(3, 9, 479).u(3, 0).u32_field(0, 0, 0);
        let data = w.finish_padded(2);
        let mut r = BitReader::new(&data);
        let preview = read_preview_header(&mut r).expect("4096 is allowed");
        assert_eq!(preview.height(), 4096);
    }
}
