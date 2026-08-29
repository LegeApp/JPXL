//! VarDCT varblock vocabulary and the coefficients-to-samples reconstruction.
//!
//! This module owns the ISO/IEC 18181-1:2024 Annex I concepts that every other
//! VarDCT stage has to agree on:
//!
//! * [`TransformType`] — Table I.1's 27 `DctSelect` values, with their sample
//!   dimensions, coefficient dimensions (I.3.2), 8x8-block footprint, Order ID
//!   (Table I.7) and dequantization-matrix index (Table I.4).
//! * [`SampleBlock`] and [`CoeffMatrix`] — deliberately *distinct* types, see
//!   the orientation note below.
//! * [`natural_coeff_order`] — I.3.2's LLF-then-HF ordering.
//! * [`llf_from_lf`] — I.8, the LLF coefficients derived from the 8x
//!   downsampled image.
//! * [`TransformType::samples_from_coefficients`] — I.9.2 to I.9.8.
//!
//! # The orientation rule (read this first)
//!
//! I.7.1 fixes the notation: an `RxC` matrix has `R` rows and `C` columns, and
//! `DCTRxC` therefore names a varblock of `R` rows by `C` columns of samples.
//! Every formula in Annex I then indexes as `m(x, y)` with **`x` the column and
//! `y` the row**, which is the opposite argument order. Both conventions are
//! honoured here: storage is row-major, and the accessors are named `at(x, y)`.
//!
//! I.3.2 sizes the coefficient array as `bwidth = max(8, max(N, M))`,
//! `bheight = max(8, min(N, M))`, so the coefficient array is **always
//! landscape**, whatever the sample orientation. I.7.3's `IDCT_2D` is the only
//! thing that flips them, via its leading `if (C > R) Transpose`. Concretely: a
//! DCT16x8 varblock has 16x8 *samples* and 8x16 *coefficients*, and its
//! dequantization matrix (Table I.4 index 6) is 8x16 as well.
//!
//! Storing "the coefficients of a DCT16x8" in a 16x8 buffer is silently wrong
//! for exactly the non-square transforms, which are rare in small fixtures.
//! That is why [`CoeffMatrix`] and [`SampleBlock`] are separate types with
//! separate constructors rather than one matrix type.

// Every index below is derived from a loop bound over a buffer whose length was
// checked at the top of the enclosing function, so bounds checks here would be
// noise rather than a defence.
#![allow(clippy::indexing_slicing)]

use crate::dct::{
    coeff_dims, dct_2d_raw, half_idct_2d_in_place, idct_2d_into, idct_2d_raw,
    lowpass_half_idct_2d_in_place,
};
use crate::error::{JpxlError, Result};
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Flip points (18181-1 defects and unstated placements)
// ---------------------------------------------------------------------------

/// **Flip point — I.9.6/I.9.7 half-block placement.**
///
/// I.9.6 splits an 8x8 DCT8x4 varblock into two vertical 8x4 half-blocks, and
/// I.9.7 splits a DCT4x8 varblock into two horizontal 4x8 half-blocks, but
/// neither clause says where the two halves land in the 8x8 output.
///
/// * `true` (shipped): half index 0 occupies the low coordinates — columns 0..4
///   for DCT8x4, rows 0..4 for DCT4x8 — and half index 1 the high ones.
/// * `false`: the halves are swapped.
///
/// The shipped reading is not a guess. I.9.8 (AFV) reconstructs a 4x8
/// sub-block by exactly the gather I.9.7 uses for its half 1 — the odd rows,
/// with the DC set to the difference of the two leading coefficients — and
/// I.9.8 *does* state its placement: unflipped, that sub-block occupies rows
/// 4..8. That is "half index 1 at the high coordinates". The same argument
/// carries to DCT8x4 by the column/row symmetry of the clause pair, and it
/// agrees with I.9.3's `AuxIDCT2x2`, where the all-plus butterfly output — the
/// analogue of the sum half of the DC pair — is written to the low coordinate.
///
/// See `docs/experiments/2026-08-03-i9-dct8x4-half-placement.md`. Slice 8F can
/// still probe it end to end once pixels exist; flipping this constant is the
/// whole change.
/// **NOT DISCRIMINATED by 8F's end-to-end probe (2026-08-03).** Flipping it
/// changes no digit of any acceptance case, because cjxl never selected
/// DctSelect 12 or 13 for any fixture or corpus stream available (the decoder's
/// own DctSelect histograms are tabulated in the experiment note). Settling it
/// still needs a stream containing DCT4x8/DCT8x4 varblocks. See
/// `docs/experiments/2026-08-03-vardct-flip-point-probe.md`.
pub const DCT8X4_HALF_INDEX_IS_LOW_COORDINATE: bool = true;

/// **Flip point — I.8's `ScaleF` second argument.**
///
/// I.8 scales each LLF cell by a helper `ScaleF(c, b)` whose value is the
/// reciprocal of `cos(c*pi/(2b)) * cos(c*pi/b) * cos(2*c*pi/b)`, applied once
/// per axis, and its call sites pass the LF-sample counts
/// `cx = max(bwidth, bheight) / 8` and `cy = min(bwidth, bheight) / 8` as `b`
/// while `c` runs over `0..b`. All three transcriptions of Part 1 agree on that
/// text, and it is **unimplementable as printed**: at `c == b / 2` the middle
/// cosine is `cos(pi/2) == 0`, so the product is zero and the scale is
/// infinite. That is reached for every transform with `cx >= 2`, starting at
/// DCT16x16.
///
/// * `true` (shipped): `b` is the varblock dimension in samples (`bwidth` or
///   `bheight`), i.e. eight times the printed argument. Then
///   `c < b / 8`, all three cosines exceed `cos(pi/4)`, and the scale is finite.
/// * `false`: the literal reading, which divides by zero.
///
/// The shipped reading is derived, not guessed. Writing `theta = c*pi/(2*b)`,
/// the printed product is the Dirichlet identity
/// `cos(theta)cos(2 theta)cos(4 theta) = sin(8 theta) / (8 sin theta)`. Taking
/// the 8x downsampled image to be the DC coefficients of the 8x8 blocks and
/// asking for the exact ratio between the length-`8n` DCT coefficient `X_k` and
/// the length-`n` DCT of the block DCs `D_k` gives
/// `X_k = D_k / [cos(k pi/(16n)) cos(k pi/(8n)) cos(k pi/(4n))]`, which is the
/// printed expression with `b = 8n = ` the varblock dimension. The derivation
/// and its numeric verification are in
/// `docs/experiments/2026-08-03-i8-scalef-argument.md`; `llf_matches_the_varblocks_own_low_frequency_coefficients`
/// in this module's tests is the executable form.
/// **PROBED-CONFIRMED end to end (2026-08-03, slice 8F).** Flipping this to
/// the literal reading makes nine of the ten acceptance cases fail with `NaN`
/// RMSE and `inf` peak error — the predicted division by zero, reached by
/// every DCT16x16 and larger varblock in the fixtures. See
/// `docs/experiments/2026-08-03-vardct-flip-point-probe.md`.
pub const LLF_SCALEF_ARG_IS_VARBLOCK_DIMENSION: bool = true;

// ---------------------------------------------------------------------------
// Table I.1 — transform types
// ---------------------------------------------------------------------------

/// A VarDCT transform type: the `DctSelect` values of 18181-1 Table I.1.
///
/// Discriminants are the numerical values from that table, so
/// `TransformType::Dct8x8 as u8 == 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum TransformType {
    /// `DCT8x8`, Table I.1 value 0.
    Dct8x8 = 0,
    /// `Hornuss`, value 1. Reconstructed by I.9.5, not by an IDCT.
    Hornuss = 1,
    /// `DCT2x2`, value 2. Reconstructed by I.9.3.
    Dct2x2 = 2,
    /// `DCT4x4`, value 3. Reconstructed by I.9.4.
    Dct4x4 = 3,
    /// `DCT16x16`, value 4.
    Dct16x16 = 4,
    /// `DCT32x32`, value 5.
    Dct32x32 = 5,
    /// `DCT16x8`, value 6: 16 rows by 8 columns of samples.
    Dct16x8 = 6,
    /// `DCT8x16`, value 7.
    Dct8x16 = 7,
    /// `DCT32x8`, value 8.
    Dct32x8 = 8,
    /// `DCT8x32`, value 9.
    Dct8x32 = 9,
    /// `DCT32x16`, value 10.
    Dct32x16 = 10,
    /// `DCT16x32`, value 11.
    Dct16x32 = 11,
    /// `DCT4x8`, value 12. Reconstructed by I.9.7.
    Dct4x8 = 12,
    /// `DCT8x4`, value 13. Reconstructed by I.9.6.
    Dct8x4 = 13,
    /// `AFV0`, value 14. Reconstructed by I.9.8.
    Afv0 = 14,
    /// `AFV1`, value 15.
    Afv1 = 15,
    /// `AFV2`, value 16.
    Afv2 = 16,
    /// `AFV3`, value 17.
    Afv3 = 17,
    /// `DCT64x64`, value 18.
    Dct64x64 = 18,
    /// `DCT64x32`, value 19.
    Dct64x32 = 19,
    /// `DCT32x64`, value 20.
    Dct32x64 = 20,
    /// `DCT128x128`, value 21.
    Dct128x128 = 21,
    /// `DCT128x64`, value 22.
    Dct128x64 = 22,
    /// `DCT64x128`, value 23.
    Dct64x128 = 23,
    /// `DCT256x256`, value 24.
    Dct256x256 = 24,
    /// `DCT256x128`, value 25.
    Dct256x128 = 25,
    /// `DCT128x256`, value 26.
    Dct128x256 = 26,
}

/// The per-transform row of Tables I.1, I.4 and I.7.
#[derive(Debug, Clone, Copy)]
struct TransformInfo {
    /// Sample dimensions `(R, C)` for the plain `DCTRxC` types of I.9.2;
    /// `None` for the types with a bespoke reconstruction in I.9.3-I.9.8.
    dct_shape: Option<(usize, usize)>,
    /// Table I.1 "Dimensions in `DctSelect`", as `(rows, columns)` of 8x8
    /// blocks.
    blocks: (usize, usize),
    /// Table I.7 Order ID.
    order_id: u8,
    /// Table I.4 parameters index for the dequantization matrix.
    dequant_index: u8,
}

/// Table I.1 x Table I.4 x Table I.7, indexed by `DctSelect`.
///
/// Table I.7's Order-ID column is destroyed in `latex/part1.tex`; the values
/// below come from `markdowns/standard-markdowns/part1.md`, which prints
/// 0..=12 against the same row order, per the doc-access exception recorded in
/// the slice-8 scoping report.
const TRANSFORM_INFO: [TransformInfo; 27] = [
    // DctSelect 0: DCT8x8
    info(Some((8, 8)), (1, 1), 0, 0),
    // 1: Hornuss
    info(None, (1, 1), 1, 1),
    // 2: DCT2x2
    info(None, (1, 1), 1, 2),
    // 3: DCT4x4
    info(None, (1, 1), 1, 3),
    // 4: DCT16x16
    info(Some((16, 16)), (2, 2), 2, 4),
    // 5: DCT32x32
    info(Some((32, 32)), (4, 4), 3, 5),
    // 6: DCT16x8
    info(Some((16, 8)), (2, 1), 4, 6),
    // 7: DCT8x16
    info(Some((8, 16)), (1, 2), 4, 6),
    // 8: DCT32x8
    info(Some((32, 8)), (4, 1), 5, 7),
    // 9: DCT8x32
    info(Some((8, 32)), (1, 4), 5, 7),
    // 10: DCT32x16
    info(Some((32, 16)), (4, 2), 6, 8),
    // 11: DCT16x32
    info(Some((16, 32)), (2, 4), 6, 8),
    // 12: DCT4x8
    info(None, (1, 1), 1, 9),
    // 13: DCT8x4
    info(None, (1, 1), 1, 9),
    // 14..=17: AFV0..AFV3
    info(None, (1, 1), 1, 10),
    info(None, (1, 1), 1, 10),
    info(None, (1, 1), 1, 10),
    info(None, (1, 1), 1, 10),
    // 18: DCT64x64
    info(Some((64, 64)), (8, 8), 7, 11),
    // 19: DCT64x32
    info(Some((64, 32)), (8, 4), 8, 12),
    // 20: DCT32x64
    info(Some((32, 64)), (4, 8), 8, 12),
    // 21: DCT128x128
    info(Some((128, 128)), (16, 16), 9, 13),
    // 22: DCT128x64
    info(Some((128, 64)), (16, 8), 10, 14),
    // 23: DCT64x128
    info(Some((64, 128)), (8, 16), 10, 14),
    // 24: DCT256x256
    info(Some((256, 256)), (32, 32), 11, 15),
    // 25: DCT256x128
    info(Some((256, 128)), (32, 16), 12, 16),
    // 26: DCT128x256
    info(Some((128, 256)), (16, 32), 12, 16),
];

const fn info(
    dct_shape: Option<(usize, usize)>,
    blocks: (usize, usize),
    order_id: u8,
    dequant_index: u8,
) -> TransformInfo {
    TransformInfo {
        dct_shape,
        blocks,
        order_id,
        dequant_index,
    }
}

/// Number of Order IDs in Table I.7.
pub const NUM_ORDER_IDS: usize = 13;

/// Number of dequantization-matrix parameter slots in Table I.4.
///
/// I.2.4 reads exactly this many parameter sets, and H.4.1's RAW-mode stream
/// index adds the same constant.
pub const NUM_DEQUANT_MATRICES: usize = 17;

impl TransformType {
    /// Every `DctSelect` value, in numerical order.
    pub const ALL: [Self; 27] = [
        Self::Dct8x8,
        Self::Hornuss,
        Self::Dct2x2,
        Self::Dct4x4,
        Self::Dct16x16,
        Self::Dct32x32,
        Self::Dct16x8,
        Self::Dct8x16,
        Self::Dct32x8,
        Self::Dct8x32,
        Self::Dct32x16,
        Self::Dct16x32,
        Self::Dct4x8,
        Self::Dct8x4,
        Self::Afv0,
        Self::Afv1,
        Self::Afv2,
        Self::Afv3,
        Self::Dct64x64,
        Self::Dct64x32,
        Self::Dct32x64,
        Self::Dct128x128,
        Self::Dct128x64,
        Self::Dct64x128,
        Self::Dct256x256,
        Self::Dct256x128,
        Self::Dct128x256,
    ];

    /// Decodes a `DctSelect` value read from the HF-metadata `BlockInfo`
    /// channel (G.2.4).
    ///
    /// # Errors
    ///
    /// [`JpxlError::InvalidHeader`] if `value` is not one of Table I.1's 27
    /// numerical values. Malformed streams are rejected here rather than
    /// clamped to a nearby transform.
    pub fn from_dct_select(value: i64) -> Result<Self> {
        let idx = usize::try_from(value).ok().filter(|v| *v < Self::ALL.len());
        match idx {
            Some(i) => Ok(Self::ALL[i]),
            None => Err(JpxlError::InvalidHeader(format!(
                "DctSelect {value} is outside Table I.1's range 0..=26"
            ))),
        }
    }

    /// The Table I.1 numerical value.
    #[must_use]
    pub const fn dct_select(self) -> u8 {
        self as u8
    }

    fn info(self) -> TransformInfo {
        TRANSFORM_INFO[self as usize]
    }

    /// Sample dimensions `(R, C)` if this is a plain `DCTRxC` of I.9.2,
    /// otherwise `None`.
    ///
    /// Note that this is *not* the varblock footprint: DCT8x4 reconstructs a
    /// full 8x8 varblock out of two 8x4 half-blocks, so it returns `None` even
    /// though its name looks like a `DCTRxC`.
    #[must_use]
    pub fn dct_shape(self) -> Option<(usize, usize)> {
        self.info().dct_shape
    }

    /// Table I.1's "Dimensions in `DctSelect`": the varblock footprint in 8x8
    /// blocks, as `(rows, columns)`.
    #[must_use]
    pub fn block_dims(self) -> (usize, usize) {
        self.info().blocks
    }

    /// Number of 8x8 blocks the varblock covers.
    #[must_use]
    pub fn num_blocks(self) -> usize {
        let (r, c) = self.block_dims();
        r * c
    }

    /// Height of the varblock in samples: `8 * block rows`.
    #[must_use]
    pub fn sample_rows(self) -> usize {
        self.block_dims().0 * 8
    }

    /// Width of the varblock in samples: `8 * block columns`.
    #[must_use]
    pub fn sample_cols(self) -> usize {
        self.block_dims().1 * 8
    }

    /// I.3.2's `bwidth`: the number of *columns* of the coefficient array,
    /// `max(8, max(N, M))`.
    #[must_use]
    pub fn coeff_cols(self) -> usize {
        let (r, c) = self.block_dims();
        r.max(c) * 8
    }

    /// I.3.2's `bheight`: the number of *rows* of the coefficient array,
    /// `max(8, min(N, M))`.
    #[must_use]
    pub fn coeff_rows(self) -> usize {
        let (r, c) = self.block_dims();
        r.min(c) * 8
    }

    /// Table I.7 Order ID, in `0..`[`NUM_ORDER_IDS`].
    #[must_use]
    pub fn order_id(self) -> usize {
        self.info().order_id as usize
    }

    /// Table I.4 dequantization-matrix parameters index, in
    /// `0..`[`NUM_DEQUANT_MATRICES`].
    #[must_use]
    pub fn dequant_matrix_index(self) -> usize {
        self.info().dequant_index as usize
    }

    /// Does I.8 apply a `DCT_2D` when deriving the LLF coefficients?
    ///
    /// True for the 17 `DCTRxC` families I.8 lists; false for Hornuss, DCT2x2,
    /// DCT4x4, DCT8x4, DCT4x8 and AFV0-3, where the LLF output equals the LF
    /// input.
    #[must_use]
    pub fn llf_is_transformed(self) -> bool {
        self.dct_shape().is_some()
    }

    /// A zeroed coefficient matrix of the right shape for this transform.
    #[must_use]
    pub fn empty_coefficients(self) -> CoeffMatrix {
        CoeffMatrix::zeros(self.coeff_rows(), self.coeff_cols())
    }

    /// I.3.2's natural coefficient order for this transform.
    ///
    /// Equal to `natural_coeff_order` of the transform's Order ID; see
    /// [`natural_coeff_order`].
    #[must_use]
    pub fn natural_coeff_order(self) -> Vec<u32> {
        natural_coeff_order(self.coeff_cols(), self.coeff_rows())
    }

    /// [`Self::natural_coeff_order`] without the per-call allocation; see
    /// [`natural_coeff_order_ref`]. Every real `TransformType` has an Order
    /// ID row in Table I.7, so this is never `None` for `self`.
    #[must_use]
    pub fn natural_coeff_order_ref(self) -> &'static [u32] {
        natural_coeff_order_ref(self.coeff_cols(), self.coeff_rows()).unwrap_or(&[])
    }
}

// ---------------------------------------------------------------------------
// Matrix types
// ---------------------------------------------------------------------------

/// A varblock's coefficients: **always landscape**, `rows <= cols`.
///
/// Row-major, so `at(x, y)` reads `data[y * cols + x]` with `x` the column and
/// `y` the row, matching Annex I's `m(x, y)` indexing. See the module
/// documentation for why this is a different type from [`SampleBlock`].
#[derive(Debug, Clone, PartialEq)]
pub struct CoeffMatrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

/// A varblock's samples: `R` rows by `C` columns, in the varblock's own
/// orientation (so portrait for DCT16x8).
///
/// Row-major, `at(x, y)` reads `data[y * cols + x]`.
#[derive(Debug, Clone, PartialEq)]
pub struct SampleBlock {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl CoeffMatrix {
    /// A zeroed `rows x cols` coefficient matrix.
    ///
    /// `rows > cols` is a caller bug (coefficients are always landscape); in a
    /// release build the dimensions are used as given.
    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        debug_assert!(rows <= cols, "coefficient matrices are always landscape");
        Self {
            rows,
            cols,
            data: vec![0.0f32; rows * cols],
        }
    }

    /// Wraps an existing row-major landscape buffer.
    ///
    /// A buffer of the wrong length is replaced by zeros rather than panicking
    /// or silently reinterpreting the layout.
    #[must_use]
    pub fn from_landscape(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        debug_assert!(rows <= cols, "coefficient matrices are always landscape");
        if data.len() == rows * cols {
            Self { rows, cols, data }
        } else {
            debug_assert!(false, "coefficient buffer is {} long", data.len());
            Self::zeros(rows, cols)
        }
    }

    /// Number of rows (I.3.2's `bheight` for a whole varblock).
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns (I.3.2's `bwidth`).
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Row-major backing storage.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Mutable row-major backing storage.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Coefficient at column `x`, row `y`; `0.0` outside the matrix.
    #[must_use]
    pub fn at(&self, x: usize, y: usize) -> f32 {
        if x < self.cols && y < self.rows {
            self.data[y * self.cols + x]
        } else {
            0.0
        }
    }

    /// Sets the coefficient at column `x`, row `y`. Out-of-range writes are
    /// dropped.
    pub fn set(&mut self, x: usize, y: usize, value: f32) {
        if x < self.cols && y < self.rows {
            self.data[y * self.cols + x] = value;
        }
    }

    /// Reshapes the matrix to a zeroed `rows x cols`, reusing its storage.
    /// Equivalent to `*self = Self::zeros(rows, cols)` without the
    /// allocation.
    pub fn reset(&mut self, rows: usize, cols: usize) {
        debug_assert!(rows <= cols, "coefficient matrices are always landscape");
        self.rows = rows;
        self.cols = cols;
        self.data.clear();
        self.data.resize(rows * cols, 0.0f32);
    }

    /// Copies `llf` into the top-left corner, as I.8 requires of the LLF
    /// sub-rectangle.
    pub fn write_llf(&mut self, llf: &Self) {
        for y in 0..llf.rows().min(self.rows) {
            for x in 0..llf.cols().min(self.cols) {
                self.data[y * self.cols + x] = llf.at(x, y);
            }
        }
    }
}

impl SampleBlock {
    /// A zeroed `rows x cols` sample block.
    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0f32; rows * cols],
        }
    }

    /// Wraps an existing row-major `rows x cols` buffer; a wrong-length buffer
    /// is replaced by zeros.
    #[must_use]
    pub fn from_rows_cols(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        if data.len() == rows * cols {
            Self { rows, cols, data }
        } else {
            debug_assert!(false, "sample buffer is {} long", data.len());
            Self::zeros(rows, cols)
        }
    }

    /// Number of rows (`R`).
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns (`C`).
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Row-major backing storage.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Mutable row-major backing storage.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Sample at column `x`, row `y`; `0.0` outside the block.
    #[must_use]
    pub fn at(&self, x: usize, y: usize) -> f32 {
        if x < self.cols && y < self.rows {
            self.data[y * self.cols + x]
        } else {
            0.0
        }
    }

    /// Sets the sample at column `x`, row `y`. Out-of-range writes are dropped.
    pub fn set(&mut self, x: usize, y: usize, value: f32) {
        if x < self.cols && y < self.rows {
            self.data[y * self.cols + x] = value;
        }
    }

    /// Reshapes the block to a zeroed `rows x cols`, reusing its storage.
    /// Equivalent to `*self = Self::zeros(rows, cols)` without the
    /// allocation.
    pub fn reset(&mut self, rows: usize, cols: usize) {
        self.rows = rows;
        self.cols = cols;
        self.data.clear();
        self.data.resize(rows * cols, 0.0f32);
    }
}

// ---------------------------------------------------------------------------
// I.3.2 — natural ordering of the DCT coefficients
// ---------------------------------------------------------------------------

static ORDER_ID_DIMS: OnceLock<[Option<(usize, usize)>; NUM_ORDER_IDS]> = OnceLock::new();
static NATURAL_COEFF_ORDERS: [OnceLock<Vec<u32>>; NUM_ORDER_IDS] =
    [const { OnceLock::new() }; NUM_ORDER_IDS];

fn cached_order_id_dims() -> &'static [Option<(usize, usize)>; NUM_ORDER_IDS] {
    ORDER_ID_DIMS.get_or_init(|| {
        let mut dims = [None; NUM_ORDER_IDS];
        for transform in TransformType::ALL {
            let slot = dims.get_mut(transform.order_id());
            if let Some(slot) = slot {
                *slot = Some((transform.coeff_cols(), transform.coeff_rows()));
            }
        }
        dims
    })
}

/// `(bwidth, bheight)` for an Order ID, per Table I.7 and I.3.2.
///
/// `None` for an out-of-range ID.
#[must_use]
pub fn order_id_dims(order_id: usize) -> Option<(usize, usize)> {
    cached_order_id_dims().get(order_id).copied().flatten()
}

/// I.3.2's natural coefficient order for a `bwidth x bheight` coefficient
/// array.
///
/// The result is a permutation of `0..bwidth*bheight`: element `i` is the
/// row-major index `y * bwidth + x` of the cell that occupies position `i` of
/// the order. The first `(bwidth/8) * (bheight/8)` entries are the LLF
/// sub-rectangle sorted by `y * bwidth/8 + x`; the rest are the HF cells sorted
/// by the boustrophedon `(key1, key2)` of I.3.2.
///
/// Allocates and copies the cached table; callers that only read it (the
/// common case — most call this once per varblock) should prefer
/// [`natural_coeff_order_ref`], which borrows the same cache instead.
#[must_use]
pub fn natural_coeff_order(bwidth: usize, bheight: usize) -> Vec<u32> {
    natural_coeff_order_ref(bwidth, bheight)
        .map(<[u32]>::to_vec)
        .unwrap_or_else(|| natural_coeff_order_uncached(bwidth, bheight))
}

/// [`natural_coeff_order`] without the per-call allocation: a `'static`
/// reference into the same per-Order-ID cache. `None` only for dimensions
/// that do not correspond to any of Table I.7's Order IDs (every real
/// [`TransformType`] does, via [`TransformType::natural_coeff_order_ref`]).
///
/// Phase 30 (`JPXL/docs/optimize.md`): a profile found this table cloned on
/// every varblock in several read-only call sites, even though it depends
/// only on the Order ID (13 possible values) and never changes after the
/// first call for a given shape.
#[must_use]
// `bwidth * bheight` is at most 65536, so both the i64 keys and the u32 output
// are exact; the table is fixed by Table I.1 and cannot grow at runtime.
#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
pub fn natural_coeff_order_ref(bwidth: usize, bheight: usize) -> Option<&'static [u32]> {
    let order_id = cached_order_id_dims()
        .iter()
        .position(|&dims| dims == Some((bwidth, bheight)))?;
    NATURAL_COEFF_ORDERS.get(order_id).map(|order| {
        order
            .get_or_init(|| natural_coeff_order_uncached(bwidth, bheight))
            .as_slice()
    })
}

#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
fn natural_coeff_order_uncached(bwidth: usize, bheight: usize) -> Vec<u32> {
    let cx = bwidth / 8;
    let cy = bheight / 8;
    let scale = cx.max(cy) as i64;

    let mut llf: Vec<(i64, usize)> = Vec::with_capacity(cx * cy);
    let mut hf: Vec<(i64, i64, usize)> = Vec::with_capacity(bwidth * bheight - cx * cy);

    for y in 0..bheight {
        for x in 0..bwidth {
            let linear = y * bwidth + x;
            if x < cx && y < cy {
                llf.push(((y * cx + x) as i64, linear));
                continue;
            }
            // Integer arithmetic throughout: cx and cy are powers of two and
            // `scale` is a multiple of both, so both divisions are exact.
            let scaled_x = (x as i64) * scale / (cx as i64);
            let scaled_y = (y as i64) * scale / (cy as i64);
            let key1 = scaled_x + scaled_y;
            let mut key2 = scaled_x - scaled_y;
            if key1 % 2 == 1 {
                key2 = -key2;
            }
            hf.push((key1, key2, linear));
        }
    }

    llf.sort_by_key(|e| e.0);
    hf.sort_by_key(|e| (e.0, e.1));

    let mut order = Vec::with_capacity(bwidth * bheight);
    order.extend(llf.iter().map(|e| e.1 as u32));
    order.extend(hf.iter().map(|e| e.2 as u32));
    order
}

// ---------------------------------------------------------------------------
// I.8 — LLF coefficients from the downsampled image
// ---------------------------------------------------------------------------

/// I.8's `ScaleF`, with the second argument taken as the varblock dimension.
///
/// `lf_count` is the printed argument (`cx` or `cy`, a count of LF samples).
/// See [`LLF_SCALEF_ARG_IS_VARBLOCK_DIMENSION`] for why it is multiplied by 8.
/// Crate-visible so the forward direction ([`crate::forward::lf_from_llf`]) can
/// divide by the very same factor rather than transcribing it a second time —
/// a second copy would be free to drift away from the flip point above.
pub(crate) fn scale_f(c: usize, lf_count: usize) -> f32 {
    // Table I.1's LF rectangles are at most 32 cells on a side, and I.8 asks
    // for `ScaleF` at every cell of every varblock; three `f64` cosines per
    // cell per call was a measurable share of encoder profiles. The tabled
    // values are the closed form evaluated once, so a hit and a miss are the
    // same number.
    let slot = match lf_count {
        1 => 0,
        2 => 1,
        4 => 2,
        8 => 3,
        16 => 4,
        32 => 5,
        _ => return scale_f_closed_form(c, lf_count),
    };
    if c >= SCALE_F_TABLE_LEN {
        return scale_f_closed_form(c, lf_count);
    }
    let table = SCALE_F_TABLES[slot]
        .get_or_init(|| core::array::from_fn(|c| scale_f_closed_form(c, lf_count)));
    table[c]
}

/// Cells per cached [`scale_f`] table: the largest LF count Table I.1 allows.
const SCALE_F_TABLE_LEN: usize = 32;

/// [`scale_f`] for LF counts 1, 2, 4, 8, 16 and 32, in that order, filled on
/// first use.
static SCALE_F_TABLES: [OnceLock<[f32; SCALE_F_TABLE_LEN]>; 6] = [const { OnceLock::new() }; 6];

/// I.8's `ScaleF` evaluated from its closed form; [`scale_f`] caches it.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn scale_f_closed_form(c: usize, lf_count: usize) -> f32 {
    let b = if LLF_SCALEF_ARG_IS_VARBLOCK_DIMENSION {
        lf_count * 8
    } else {
        lf_count
    };
    if b == 0 {
        return 1.0;
    }
    let cf = c as f64;
    let bf = b as f64;
    let pi = core::f64::consts::PI;
    let inverse_scale_f =
        (cf * pi / (2.0 * bf)).cos() * (cf * pi / bf).cos() * (2.0 * cf * pi / bf).cos();
    (1.0 / inverse_scale_f) as f32
}

/// I.8: the LLF coefficients of `transform`, from its rectangle of the 8x
/// downsampled image.
///
/// `lf` is that rectangle in **image orientation**: `block_dims().0` rows by
/// `block_dims().1` columns of LF samples. The result is landscape
/// (`bheight/8` rows by `bwidth/8` columns) and belongs in the top-left corner
/// of the varblock's [`CoeffMatrix`] — see [`CoeffMatrix::write_llf`].
///
/// For Hornuss, DCT2x2, DCT4x4, DCT8x4, DCT4x8 and AFV0-3 the output equals the
/// input (a single LF sample), as I.8 states.
#[must_use]
pub fn llf_from_lf(transform: TransformType, lf: &SampleBlock) -> CoeffMatrix {
    let (block_rows, block_cols) = transform.block_dims();
    let cx = block_rows.max(block_cols);
    let cy = block_rows.min(block_cols);

    if lf.rows() != block_rows || lf.cols() != block_cols {
        debug_assert!(false, "LF rectangle has the wrong shape");
        return CoeffMatrix::zeros(cy, cx);
    }

    if !transform.llf_is_transformed() {
        // 1x1 for every such type, so orientation cannot differ.
        return CoeffMatrix::from_landscape(cy, cx, lf.as_slice().to_vec());
    }

    // I.8's input has `bwidth/8` columns and `bheight/8` rows, i.e. it is
    // landscape. For a portrait varblock (DCT16x8 and friends) the LF rectangle
    // therefore has to be transposed on the way in — the same flip `IDCT_2D`
    // performs on the way out.
    let input = if block_rows > block_cols {
        crate::dct::transpose(lf.as_slice(), block_rows, block_cols)
    } else {
        lf.as_slice().to_vec()
    };

    // `DCT_2D` on a `cy x cx` matrix with `cx >= cy` gives back `cy x cx`.
    debug_assert_eq!(coeff_dims(cy, cx), (cy, cx));
    let dc = dct_2d_raw(&input, cy, cx);

    let mut out = CoeffMatrix::zeros(cy, cx);
    for y in 0..cy {
        for x in 0..cx {
            let value = dc[y * cx + x] * scale_f(y, cy) * scale_f(x, cx);
            out.set(x, y, value);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// I.9 — coefficients to samples
// ---------------------------------------------------------------------------

/// An 8x8 working buffer with Annex I's `(x, y)` indexing.
#[derive(Debug, Clone, Copy)]
struct Block8 {
    v: [f32; 64],
}

impl Block8 {
    const fn zeros() -> Self {
        Self { v: [0.0f32; 64] }
    }

    fn from_coeffs(c: &CoeffMatrix) -> Self {
        let mut b = Self::zeros();
        for y in 0..8 {
            for x in 0..8 {
                b.v[y * 8 + x] = c.at(x, y);
            }
        }
        b
    }

    fn at(&self, x: usize, y: usize) -> f32 {
        self.v[y * 8 + x]
    }

    fn set(&mut self, x: usize, y: usize, value: f32) {
        self.v[y * 8 + x] = value;
    }

    fn into_samples(self) -> SampleBlock {
        SampleBlock::from_rows_cols(8, 8, self.v.to_vec())
    }
}

/// I.9.3's `AuxIDCT2x2(block, s)`.
///
/// Touches only the top-left `s x s` cells; the rest are copied through, as the
/// clause requires when `block` is larger than `s x s`.
fn aux_idct_2x2(block: &Block8, s: usize) -> Block8 {
    let mut result = *block;
    let num_2x2 = s / 2;
    for y in 0..num_2x2 {
        for x in 0..num_2x2 {
            let c00 = block.at(x, y);
            let c01 = block.at(num_2x2 + x, y);
            let c10 = block.at(x, y + num_2x2);
            let c11 = block.at(num_2x2 + x, y + num_2x2);
            result.set(x * 2, y * 2, c00 + c01 + c10 + c11);
            result.set(x * 2 + 1, y * 2, c00 + c01 - c10 - c11);
            result.set(x * 2, y * 2 + 1, c00 - c01 + c10 - c11);
            result.set(x * 2 + 1, y * 2 + 1, c00 - c01 - c10 + c11);
        }
    }
    result
}

/// I.9.3 DCT2x2.
fn samples_dct2x2(coeffs: &CoeffMatrix) -> SampleBlock {
    let mut block = Block8::from_coeffs(coeffs);
    block = aux_idct_2x2(&block, 2);
    block = aux_idct_2x2(&block, 4);
    block = aux_idct_2x2(&block, 8);
    block.into_samples()
}

/// I.9.4 DCT4x4.
fn samples_dct4x4(coeffs: &CoeffMatrix) -> SampleBlock {
    let c = Block8::from_coeffs(coeffs);
    let dcs = aux_idct_2x2(&c, 2);
    let mut out = Block8::zeros();
    for y in 0..2 {
        for x in 0..2 {
            let mut block = [0.0f32; 16];
            for iy in 0..4 {
                for ix in (if iy == 0 { 1 } else { 0 })..4 {
                    block[iy * 4 + ix] = c.at(x + ix * 2, y + iy * 2);
                }
            }
            block[0] = dcs.at(x, y);
            let sample = idct_2d_raw(&block, 4, 4);
            // result(4*i + k, 4*j + l) = sample(i, j, k, l): `k` is the column
            // within the 4x4 and `l` the row.
            for l in 0..4 {
                for k in 0..4 {
                    out.set(4 * x + k, 4 * y + l, sample[l * 4 + k]);
                }
            }
        }
    }
    out.into_samples()
}

/// I.9.5 Hornuss.
fn samples_hornuss(coeffs: &CoeffMatrix) -> SampleBlock {
    let c = Block8::from_coeffs(coeffs);
    let dcs = aux_idct_2x2(&c, 2);
    let mut out = Block8::zeros();
    for y in 0..2 {
        for x in 0..2 {
            let block_lf = dcs.at(x, y);
            let mut residual_sum = 0.0f32;
            for iy in 0..4 {
                for ix in (if iy == 0 { 1 } else { 0 })..4 {
                    residual_sum += c.at(x + ix * 2, y + iy * 2);
                }
            }
            let centre = block_lf - residual_sum / 16.0;
            out.set(4 * x + 1, 4 * y + 1, centre);
            for iy in 0..4 {
                for ix in 0..4 {
                    if ix == 1 && iy == 1 {
                        continue;
                    }
                    out.set(
                        x * 4 + ix,
                        y * 4 + iy,
                        c.at(x + ix * 2, y + iy * 2) + centre,
                    );
                }
            }
            // The clause writes the (0, 0) cell twice: once in the loop above
            // with `coefficients(x, y)` (the DC position) and again here with
            // `coefficients(x + 2, y + 2)`. The second write is the one that
            // stands.
            out.set(4 * x, 4 * y, c.at(x + 2, y + 2) + centre);
        }
    }
    out.into_samples()
}

/// Gathers I.9.6/I.9.7's half-block coefficients: a 4-row by 8-column matrix
/// whose rows are rows `half`, `half + 2`, `half + 4`, `half + 6` of the 8x8
/// coefficient block, with the DC replaced by `dc`.
fn gather_half_4x8(c: &Block8, half: usize, dc: f32) -> [f32; 32] {
    let mut coeffs = [0.0f32; 32];
    for iy in 0..4 {
        for ix in (if iy == 0 { 1 } else { 0 })..8 {
            coeffs[iy * 8 + ix] = c.at(ix, half + iy * 2);
        }
    }
    coeffs[0] = dc;
    coeffs
}

/// I.9.6 DCT8x4 and I.9.7 DCT4x8.
///
/// The two clauses share their gather verbatim; they differ only in the sample
/// shape they ask `IDCT_2D` for and in where the halves land. `vertical` is
/// true for DCT8x4 ("two 8x4 vertical blocks", side by side).
fn samples_dct8x4_or_4x8(coeffs: &CoeffMatrix, vertical: bool) -> SampleBlock {
    let c = Block8::from_coeffs(coeffs);
    let coef0 = c.at(0, 0);
    let coef1 = c.at(0, 1);
    let dcs = [coef0 + coef1, coef0 - coef1];

    let mut out = Block8::zeros();
    for (half, dc) in dcs.iter().enumerate() {
        let gathered = gather_half_4x8(&c, half, *dc);
        let placed = if DCT8X4_HALF_INDEX_IS_LOW_COORDINATE {
            half
        } else {
            1 - half
        };
        if vertical {
            // IDCT_2D with (R, C) = (8, 4): 8 rows by 4 columns of samples.
            let s = idct_2d_raw(&gathered, 8, 4);
            for iy in 0..8 {
                for ix in 0..4 {
                    out.set(4 * placed + ix, iy, s[iy * 4 + ix]);
                }
            }
        } else {
            // IDCT_2D with (R, C) = (4, 8): 4 rows by 8 columns of samples.
            let s = idct_2d_raw(&gathered, 4, 8);
            for iy in 0..4 {
                for ix in 0..8 {
                    out.set(ix, 4 * placed + iy, s[iy * 8 + ix]);
                }
            }
        }
    }
    out.into_samples()
}

/// I.9.8's orthonormal AFV basis.
///
/// Row `j` is basis function `j`; its 16 entries are the 4x4 AFV quadrant in
/// raster order (`iy * 4 + ix`). Two OCR defects in `latex/part1.tex` were
/// repaired before transcription — `0,.18567180916109802` (stray comma, row 11)
/// and `0.21918684838857 28` (digit-splitting space, row 15) — both confirmed
/// against `markdowns/standard-markdowns/part1.md`. The clause asserts the
/// basis is orthonormal and `afv_basis_is_orthonormal` proves the repaired
/// table is, to 2e-7 in f32.
#[rustfmt::skip]
// The 1/sqrt(2) entries in row 4 are printed to 16 digits in the standard and
// are kept verbatim: this table is transcription evidence, not a computation.
#[allow(clippy::approx_constant)]
pub const AFV_BASIS: [[f64; 16]; 16] = [
    [
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.25, 0.25,
    ],
    [
        0.876902929799142, 0.2206518106944235, -0.10140050393753763, -0.1014005039375375,
        0.2206518106944236, -0.10140050393753777, -0.10140050393753772, -0.10140050393753763,
        -0.10140050393753758, -0.10140050393753769, -0.1014005039375375, -0.10140050393753768,
        -0.10140050393753768, -0.10140050393753759, -0.10140050393753763, -0.10140050393753741,
    ],
    [
        0.0, 0.0, 0.40670075830260755, 0.44444816619734445, 0.0, 0.0, 0.19574399372042936,
        0.2929100136981264, -0.40670075830260716, -0.19574399372042872, 0.0,
        0.11379074460448091, -0.44444816619734384, -0.29291001369812636, -0.1137907446044814,
        0.0,
    ],
    [
        0.0, 0.0, -0.21255748058288748, 0.3085497062849767, 0.0, 0.4706702258572536,
        -0.1621205195722993, 0.0, -0.21255748058287047, -0.16212051957228327,
        -0.47067022585725277, -0.1464291867126764, 0.3085497062849487, 0.0,
        -0.14642918671266536, 0.4251149611657548,
    ],
    [
        0.0, -0.7071067811865474, 0.0, 0.0, 0.7071067811865476, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    [
        -0.4105377591765233, 0.6235485373547691, -0.06435071657946274, -0.06435071657946266,
        0.6235485373547694, -0.06435071657946284, -0.0643507165794628, -0.06435071657946274,
        -0.06435071657946272, -0.06435071657946279, -0.06435071657946266, -0.06435071657946277,
        -0.06435071657946277, -0.06435071657946273, -0.06435071657946274, -0.0643507165794626,
    ],
    [
        0.0, 0.0, -0.4517556589999482, 0.15854503551840063, 0.0, -0.04038515160822202,
        0.0074182263792423875, 0.39351034269210167, -0.45175565899994635, 0.007418226379244351,
        0.1107416575309343, 0.08298163094882051, 0.15854503551839705, 0.3935103426921022,
        0.0829816309488214, -0.45175565899994796,
    ],
    [
        0.0, 0.0, -0.304684750724869, 0.5112616136591823, 0.0, 0.0, -0.290480129728998,
        -0.06578701549142804, 0.304684750724884, 0.2904801297290076, 0.0, -0.23889773523344604,
        -0.5112616136592012, 0.06578701549142545, 0.23889773523345467, 0.0,
    ],
    [
        0.0, 0.0, 0.3017929516615495, 0.25792362796341184, 0.0, 0.16272340142866204,
        0.09520022653475037, 0.0, 0.3017929516615503, 0.09520022653475055,
        -0.16272340142866173, -0.35312385449816297, 0.25792362796341295, 0.0,
        -0.3531238544981624, -0.6035859033230976,
    ],
    [
        0.0, 0.0, 0.40824829046386274, 0.0, 0.0, 0.0, 0.0, -0.4082482904638628,
        -0.4082482904638635, 0.0, 0.0, -0.40824829046386296, 0.0, 0.4082482904638634,
        0.408248290463863, 0.0,
    ],
    [
        0.0, 0.0, 0.1747866975480809, 0.0812611176717539, 0.0, 0.0, -0.3675398009862027,
        -0.307882213957909, -0.17478669754808135, 0.3675398009862011, 0.0, 0.4826689115059883,
        -0.08126111767175039, 0.30788221395790305, -0.48266891150598584, 0.0,
    ],
    [
        0.0, 0.0, -0.21105601049335784, 0.18567180916109802, 0.0, 0.0, 0.49215859013738733,
        -0.38525013709251915, 0.21105601049335806, -0.49215859013738905, 0.0,
        0.17419412659916217, -0.18567180916109904, 0.3852501370925211, -0.1741941265991621,
        0.0,
    ],
    [
        0.0, 0.0, -0.14266084808807264, -0.3416446842253372, 0.0, 0.7367497537172237,
        0.24627107722075148, -0.08574019035519306, -0.14266084808807344, 0.24627107722075137,
        0.14883399227113567, -0.04768680350229251, -0.3416446842253373, -0.08574019035519267,
        -0.047686803502292804, -0.14266084808807242,
    ],
    [
        0.0, 0.0, -0.13813540350758585, 0.3302282550303788, 0.0, 0.08755115000587084,
        -0.07946706605909573, -0.4613374887461511, -0.13813540350758294, -0.07946706605910261,
        0.49724647109535086, 0.12538059448563663, 0.3302282550303805, -0.4613374887461554,
        0.12538059448564315, -0.13813540350758452,
    ],
    [
        0.0, 0.0, -0.17437602599651067, 0.0702790691196284, 0.0, -0.2921026642334881,
        0.3623817333531167, 0.0, -0.1743760259965108, 0.36238173335311646, 0.29210266423348785,
        -0.4326608024727445, 0.07027906911962818, 0.0, -0.4326608024727457,
        0.34875205199302267,
    ],
    [
        0.0, 0.0, 0.11354987314994337, -0.07417504595810355, 0.0, 0.19402893032594343,
        -0.435190496523228, 0.21918684838857466, 0.11354987314994257, -0.4351904965232251,
        0.5550443808910661, -0.25468277124066463, -0.07417504595810233, 0.2191868483885728,
        -0.25468277124066413, 0.1135498731499429,
    ],
];

/// I.9.8 AFV0-AFV3.
fn samples_afv(coeffs: &CoeffMatrix, n: usize) -> SampleBlock {
    let c = Block8::from_coeffs(coeffs);
    let flip_x = n & 1;
    let flip_y = n / 2;
    let mut out = Block8::zeros();

    // The 4x4 AFV quadrant.
    let mut coeff_afv = [0.0f32; 16];
    coeff_afv[0] = (c.at(0, 0) + c.at(0, 1) + c.at(1, 0)) * 4.0;
    for iy in 0..4 {
        for ix in (if iy == 0 { 1 } else { 0 })..4 {
            coeff_afv[iy * 4 + ix] = c.at(ix * 2, iy * 2);
        }
    }
    let mut samples_afv = [0.0f32; 16];
    for iy in 0..4 {
        for ix in 0..4 {
            let mut sample = 0.0f64;
            for j in 0..16 {
                sample += f64::from(coeff_afv[j]) * AFV_BASIS[j][iy * 4 + ix];
            }
            // The basis is stored in f64 to keep the printed digits verbatim;
            // the pipeline is f32 from here on.
            #[allow(clippy::cast_possible_truncation)]
            {
                samples_afv[iy * 4 + ix] = sample as f32;
            }
        }
    }
    for iy in 0..4 {
        for ix in 0..4 {
            let sx = if flip_x == 1 { 3 - ix } else { ix };
            let sy = if flip_y == 1 { 3 - iy } else { iy };
            out.set(flip_x * 4 + ix, flip_y * 4 + iy, samples_afv[sy * 4 + sx]);
        }
    }

    // The plain 4x4 quadrant beside it.
    let mut coeffs_4x4 = [0.0f32; 16];
    for iy in 0..4 {
        for ix in (if iy == 0 { 1 } else { 0 })..4 {
            coeffs_4x4[iy * 4 + ix] = c.at(ix * 2 + 1, iy * 2);
        }
    }
    coeffs_4x4[0] = c.at(0, 0) - c.at(1, 0) + c.at(0, 1);
    let samples_4x4 = idct_2d_raw(&coeffs_4x4, 4, 4);
    let x_base = if flip_x == 1 { 0 } else { 4 };
    for iy in 0..4 {
        for ix in 0..4 {
            out.set(x_base + ix, flip_y * 4 + iy, samples_4x4[iy * 4 + ix]);
        }
    }

    // The 4x8 half covering the other four rows.
    let mut coeffs_4x8 = [0.0f32; 32];
    for iy in 0..4 {
        for ix in (if iy == 0 { 1 } else { 0 })..8 {
            coeffs_4x8[iy * 8 + ix] = c.at(ix, 1 + iy * 2);
        }
    }
    coeffs_4x8[0] = c.at(0, 0) - c.at(0, 1);
    let samples_4x8 = idct_2d_raw(&coeffs_4x8, 4, 8);
    let y_base = if flip_y == 1 { 0 } else { 4 };
    for iy in 0..4 {
        for ix in 0..8 {
            out.set(ix, y_base + iy, samples_4x8[iy * 8 + ix]);
        }
    }

    out.into_samples()
}

impl TransformType {
    /// I.9: the varblock's samples, from its dequantized coefficients.
    ///
    /// `coefficients` must be `coeff_rows() x coeff_cols()`; anything else
    /// yields a zeroed block rather than a panic or a reinterpreted layout.
    /// The result is `sample_rows() x sample_cols()`.
    #[must_use]
    pub fn samples_from_coefficients(self, coefficients: &CoeffMatrix) -> SampleBlock {
        if coefficients.rows() != self.coeff_rows() || coefficients.cols() != self.coeff_cols() {
            debug_assert!(false, "coefficient matrix has the wrong shape");
            return SampleBlock::zeros(self.sample_rows(), self.sample_cols());
        }
        match self {
            Self::Hornuss => samples_hornuss(coefficients),
            Self::Dct2x2 => samples_dct2x2(coefficients),
            Self::Dct4x4 => samples_dct4x4(coefficients),
            Self::Dct8x4 => samples_dct8x4_or_4x8(coefficients, true),
            Self::Dct4x8 => samples_dct8x4_or_4x8(coefficients, false),
            Self::Afv0 => samples_afv(coefficients, 0),
            Self::Afv1 => samples_afv(coefficients, 1),
            Self::Afv2 => samples_afv(coefficients, 2),
            Self::Afv3 => samples_afv(coefficients, 3),
            _ => {
                // I.9.2: every remaining type is a plain DCTRxC.
                let (rows, cols) = self.dct_shape().unwrap_or((8, 8));
                SampleBlock::from_rows_cols(
                    rows,
                    cols,
                    idct_2d_raw(coefficients.as_slice(), rows, cols),
                )
            }
        }
    }

    /// [`Self::samples_from_coefficients`] writing into `out`, with `scratch`
    /// as the IDCT working buffer, so a per-varblock render loop reuses two
    /// allocations instead of making fresh ones per varblock.
    ///
    /// Sample-for-sample identical to the owning form: the plain `DCTRxC`
    /// types run the same [`idct_2d_into`] the owning wrapper runs (which
    /// writes every output sample, so `out`'s stale contents cannot leak),
    /// and the special I.9.3-I.9.8 types delegate to it outright.
    pub fn samples_from_coefficients_into(
        self,
        coefficients: &CoeffMatrix,
        out: &mut SampleBlock,
        scratch: &mut Vec<f32>,
    ) {
        let (rows, cols) = (self.sample_rows(), self.sample_cols());
        if coefficients.rows() != self.coeff_rows() || coefficients.cols() != self.coeff_cols() {
            debug_assert!(false, "coefficient matrix has the wrong shape");
            out.reset(rows, cols);
            return;
        }
        if self.dct_shape().is_none() {
            *out = self.samples_from_coefficients(coefficients);
            return;
        }
        out.rows = rows;
        out.cols = cols;
        // No zero-fill of the samples: `idct_2d_into` overwrites all of them.
        out.data.truncate(rows * cols);
        out.data.resize(rows * cols, 0.0f32);
        scratch.truncate(rows * cols);
        scratch.resize(rows * cols, 0.0f32);
        idct_2d_into(coefficients.as_slice(), rows, cols, &mut out.data, scratch);
    }

    /// Half of [`Self::sample_rows`]. Exact: sample dimensions are always
    /// multiples of 8.
    #[must_use]
    pub fn half_sample_rows(self) -> usize {
        self.sample_rows() / 2
    }

    /// Half of [`Self::sample_cols`].
    #[must_use]
    pub fn half_sample_cols(self) -> usize {
        self.sample_cols() / 2
    }

    /// I.9 composed with a 2:1 box average of the varblock's own samples,
    /// evaluated in the coefficient domain for the plain `DCTRxC` types and by
    /// explicit averaging for the I.9.3-I.9.8 special forms.
    ///
    /// NOT part of Annex I: this is the encoder's half-resolution surrogate
    /// reconstruction. Mirrors [`Self::samples_from_coefficients_into`]; `out`
    /// becomes `half_sample_rows() x half_sample_cols()`. When
    /// [`folds_to_lowpass`] holds (the common case at navigation quality,
    /// where quantization has zeroed the high coefficient region) the plain
    /// types take a quarter-size low-pass route that agrees with the exact
    /// fold to ~1e-6 relative rather than bitwise; the branch is a pure
    /// function of the coefficients, so worker count and lane width still
    /// cannot change a sample.
    pub fn half_samples_from_coefficients_into(
        self,
        coefficients: &CoeffMatrix,
        out: &mut SampleBlock,
        scratch: &mut Vec<f32>,
    ) {
        let (rows, cols) = (self.sample_rows(), self.sample_cols());
        let (hr, hc) = (rows / 2, cols / 2);
        if coefficients.rows() != self.coeff_rows() || coefficients.cols() != self.coeff_cols() {
            debug_assert!(false, "coefficient matrix has the wrong shape");
            out.reset(hr, hc);
            return;
        }
        if self.dct_shape().is_none() {
            // The special 8x8-footprint forms are never emitted by the current
            // planner; average their full reconstruction rather than deriving
            // nine bespoke half-resolution kernels.
            let full = self.samples_from_coefficients(coefficients);
            box_average_into(&full, out);
            return;
        }
        let n = rows * cols;
        scratch.truncate(2 * n);
        scratch.resize(2 * n, 0.0f32);
        let (work, spare) = scratch.split_at_mut(n);
        work.copy_from_slice(coefficients.as_slice());
        if folds_to_lowpass(coefficients) {
            lowpass_half_idct_2d_in_place(work, spare, rows, cols);
        } else {
            half_idct_2d_in_place(work, spare, rows, cols);
        }
        out.rows = hr;
        out.cols = hc;
        out.data.clear();
        out.data.extend_from_slice(&work[..hr * hc]);
    }

    /// [`Self::half_samples_from_coefficients_into`] as an owning wrapper.
    #[must_use]
    pub fn half_samples_from_coefficients(self, coefficients: &CoeffMatrix) -> SampleBlock {
        let mut out = SampleBlock::zeros(0, 0);
        let mut scratch = Vec::new();
        self.half_samples_from_coefficients_into(coefficients, &mut out, &mut scratch);
        out
    }
}

/// True when every coefficient the half-resolution fold reaches only through
/// its `sin` partner term is `+0.0` — every cell with `x >= cols/2` or
/// `y >= rows/2` of the landscape matrix — so the fold degenerates bitwise
/// (`x - 0.0 == x`) to the low-pass quarter route.
///
/// The test is bitwise (`to_bits() == 0`): a `-0.0` partner reports `false`,
/// because `x - (-0.0)` is not the identity on `-0.0` itself and the whole
/// point of the predicate is that taking the shortcut cannot change which
/// computation the exact path would have performed on nonzero data.
#[must_use]
pub fn folds_to_lowpass(coefficients: &CoeffMatrix) -> bool {
    let (rows, cols) = (coefficients.rows(), coefficients.cols());
    let (hr, hc) = (rows / 2, cols / 2);
    let data = coefficients.as_slice();
    for y in 0..rows {
        for x in 0..cols {
            if (x >= hc || y >= hr) && data[y * cols + x].to_bits() != 0 {
                return false;
            }
        }
    }
    true
}

/// The 2:1 box average of a block whose dimensions are both even: output
/// `(x, y)` is the mean of the four samples at `(2x + i, 2y + j)`, summed in
/// raster order (top-left, top-right, bottom-left, bottom-right) and scaled
/// by `0.25`, matching the render decimation's summation order.
fn box_average_into(src: &SampleBlock, out: &mut SampleBlock) {
    let (hr, hc) = (src.rows() / 2, src.cols() / 2);
    out.reset(hr, hc);
    for y in 0..hr {
        for x in 0..hc {
            let a = src.at(2 * x, 2 * y);
            let b = src.at(2 * x + 1, 2 * y);
            let c = src.at(2 * x, 2 * y + 1);
            let d = src.at(2 * x + 1, 2 * y + 1);
            out.set(x, y, (a + b + c + d) * 0.25);
        }
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
mod tests {
    use super::*;
    use crate::dct::dct_2d_raw;

    /// Deterministic LCG (Numerical Recipes constants); no `rand` dependency.
    struct Lcg(u32);

    impl Lcg {
        fn new(seed: u32) -> Self {
            Self(seed)
        }

        fn next(&mut self, scale: f32) -> f32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = f32::from(u16::try_from(self.0 >> 16).unwrap_or(0)) / 65536.0;
            (unit * 2.0 - 1.0) * scale
        }
    }

    fn assert_close(a: f32, b: f32, tol: f32, what: &str) {
        assert!(
            (a - b).abs() <= tol,
            "{what}: {a} vs {b} (delta {})",
            (a - b).abs()
        );
    }

    // -- Half-resolution reconstruction ------------------------------------

    /// A random coefficient matrix of `transform`'s shape; `low_only` confines
    /// the nonzero cells to the low quarter so the low-pass branch fires.
    fn random_coeffs(transform: TransformType, rng: &mut Lcg, low_only: bool) -> CoeffMatrix {
        let (rows, cols) = (transform.coeff_rows(), transform.coeff_cols());
        let mut m = CoeffMatrix::zeros(rows, cols);
        for y in 0..rows {
            for x in 0..cols {
                if !low_only || (x < cols / 2 && y < rows / 2) {
                    m.set(x, y, rng.next(32.0));
                }
            }
        }
        m
    }

    /// For every Table I.1 transform, the half-resolution reconstruction
    /// equals the 2x2 box average of the full reconstruction: exactly for the
    /// special I.9.3-I.9.8 forms (same computation), to f32 rounding for the
    /// plain `DCTRxC` families — on both the exact-fold path (dense
    /// coefficients) and the low-pass branch (low-quarter-only coefficients).
    #[test]
    fn half_samples_match_the_box_average_for_every_transform() {
        let mut rng = Lcg::new(0xdec1_0a7e);
        for &transform in TransformType::ALL.iter() {
            for low_only in [false, true] {
                let coeffs = random_coeffs(transform, &mut rng, low_only);
                let full = transform.samples_from_coefficients(&coeffs);
                let mut want = SampleBlock::zeros(0, 0);
                box_average_into(&full, &mut want);
                let got = transform.half_samples_from_coefficients(&coeffs);
                assert_eq!(
                    (got.rows(), got.cols()),
                    (transform.half_sample_rows(), transform.half_sample_cols()),
                    "{transform:?} half dims"
                );
                if transform.dct_shape().is_none() {
                    assert_eq!(
                        got.as_slice(),
                        want.as_slice(),
                        "{transform:?} special-form half samples"
                    );
                } else {
                    let peak = want.as_slice().iter().fold(1.0f32, |m, v| m.max(v.abs()));
                    for (i, (g, w)) in
                        got.as_slice().iter().zip(want.as_slice().iter()).enumerate()
                    {
                        assert_close(
                            *g,
                            *w,
                            3e-5 * peak,
                            &format!("{transform:?} low_only={low_only} half cell {i}"),
                        );
                    }
                }
            }
        }
    }

    /// Scratch and output reuse across differently shaped varblocks in
    /// sequence produces the same samples as fresh buffers — the same
    /// reuse-safety argument `samples_from_coefficients_into` makes.
    #[test]
    fn half_samples_into_reuse_matches_the_owning_form() {
        let mut rng = Lcg::new(0x5c7a_7c4e);
        let mut out = SampleBlock::zeros(0, 0);
        let mut scratch = Vec::new();
        for &transform in &[
            TransformType::Dct32x32,
            TransformType::Dct8x8,
            TransformType::Dct16x8,
            TransformType::Hornuss,
            TransformType::Dct16x16,
        ] {
            let coeffs = random_coeffs(transform, &mut rng, false);
            let fresh = transform.half_samples_from_coefficients(&coeffs);
            transform.half_samples_from_coefficients_into(&coeffs, &mut out, &mut scratch);
            assert_eq!(
                out.as_slice(),
                fresh.as_slice(),
                "{transform:?} reused vs fresh half samples"
            );
        }
    }

    /// The low-pass predicate is bitwise: a high-region `+0.0` passes, any
    /// nonzero fails, and a `-0.0` fails too (taking the shortcut must never
    /// change which computation the exact path would have performed).
    #[test]
    fn folds_to_lowpass_is_a_bitwise_zero_test() {
        let t = TransformType::Dct8x8;
        let mut m = CoeffMatrix::zeros(t.coeff_rows(), t.coeff_cols());
        m.set(1, 1, 5.0);
        m.set(3, 2, -7.0);
        assert!(folds_to_lowpass(&m), "low-quarter-only must pass");
        m.set(6, 1, 1.0e-30);
        assert!(!folds_to_lowpass(&m), "a nonzero high cell must fail");
        m.set(6, 1, 0.0);
        assert!(folds_to_lowpass(&m), "restored +0.0 must pass again");
        m.set(1, 6, -0.0);
        assert!(!folds_to_lowpass(&m), "-0.0 in the high region must fail");
    }

    // -- Table I.1 / I.4 / I.7 vocabulary ----------------------------------

    /// `DctSelect` values are exactly `0..=26` and round-trip through the enum.
    /// Anything else is rejected rather than clamped, because a malformed
    /// `BlockInfo` row would otherwise silently place the wrong varblock.
    #[test]
    fn dct_select_round_trips_and_rejects_out_of_range() {
        for (i, t) in TransformType::ALL.iter().enumerate() {
            assert_eq!(usize::from(t.dct_select()), i, "DctSelect {i}");
            assert_eq!(
                TransformType::from_dct_select(i as i64).expect("in range"),
                *t
            );
        }
        assert!(TransformType::from_dct_select(27).is_err());
        assert!(TransformType::from_dct_select(-1).is_err());
        assert!(TransformType::from_dct_select(i64::MAX).is_err());
    }

    /// Table I.1's three columns must agree with I.3.2 and I.7.1: the sample
    /// dimensions are eight times the block footprint, the coefficient
    /// dimensions are the same pair sorted into landscape order, and for the
    /// plain `DCTRxC` types the name's `RxC` is the sample shape.
    #[test]
    fn table_i1_dimensions_are_self_consistent() {
        for t in TransformType::ALL {
            let (br, bc) = t.block_dims();
            assert_eq!(t.sample_rows(), br * 8, "{t:?} sample rows");
            assert_eq!(t.sample_cols(), bc * 8, "{t:?} sample cols");
            assert_eq!(t.coeff_rows(), br.min(bc) * 8, "{t:?} bheight");
            assert_eq!(t.coeff_cols(), br.max(bc) * 8, "{t:?} bwidth");
            assert!(
                t.coeff_rows() <= t.coeff_cols(),
                "{t:?} coefficients are not landscape"
            );
            assert_eq!(t.num_blocks(), br * bc, "{t:?} block count");
            if let Some((r, c)) = t.dct_shape() {
                assert_eq!((r, c), (t.sample_rows(), t.sample_cols()), "{t:?} DCTRxC");
            }
        }
    }

    /// The type-level orientation statement, spelled out for the shape that
    /// breaks a single-matrix design: DCT16x8 has 16x8 samples and 8x16
    /// coefficients, and DCT8x16 has both the other way round for samples but
    /// the *same* coefficient shape.
    #[test]
    fn dct16x8_has_portrait_samples_and_landscape_coefficients() {
        let tall = TransformType::Dct16x8;
        assert_eq!((tall.sample_rows(), tall.sample_cols()), (16, 8));
        assert_eq!((tall.coeff_rows(), tall.coeff_cols()), (8, 16));

        let wide = TransformType::Dct8x16;
        assert_eq!((wide.sample_rows(), wide.sample_cols()), (8, 16));
        assert_eq!((wide.coeff_rows(), wide.coeff_cols()), (8, 16));

        let coeffs = tall.empty_coefficients();
        assert_eq!((coeffs.rows(), coeffs.cols()), (8, 16));
        let samples = tall.samples_from_coefficients(&coeffs);
        assert_eq!((samples.rows(), samples.cols()), (16, 8));
    }

    /// Table I.4's "Matrix size (rows columns)" column, transcribed
    /// independently of the enum, must equal `(bheight, bwidth)` for every
    /// transform sharing that parameter index. Cross-checks Table I.1 against
    /// Table I.4 and pins the dequantization-matrix shapes 8B has to build.
    #[test]
    fn dequant_matrix_index_agrees_with_table_i4_sizes() {
        const TABLE_I4_SIZES: [(usize, usize); NUM_DEQUANT_MATRICES] = [
            (8, 8),     // 0  DCT
            (8, 8),     // 1  Hornuss
            (8, 8),     // 2  DCT2x2
            (8, 8),     // 3  DCT4x4
            (16, 16),   // 4  DCT16x16
            (32, 32),   // 5  DCT32x32
            (8, 16),    // 6  DCT16x8, DCT8x16
            (8, 32),    // 7  DCT32x8, DCT8x32
            (16, 32),   // 8  DCT16x32, DCT32x16
            (8, 8),     // 9  DCT4x8, DCT8x4
            (8, 8),     // 10 AFV0..AFV3
            (64, 64),   // 11 DCT64x64
            (32, 64),   // 12 DCT32x64, DCT64x32
            (128, 128), // 13 DCT128x128
            (64, 128),  // 14 DCT64x128, DCT128x64
            (256, 256), // 15 DCT256x256
            (128, 256), // 16 DCT128x256, DCT256x128
        ];
        let mut seen = [false; NUM_DEQUANT_MATRICES];
        for t in TransformType::ALL {
            let idx = t.dequant_matrix_index();
            assert!(idx < NUM_DEQUANT_MATRICES, "{t:?} index {idx}");
            seen[idx] = true;
            assert_eq!(
                TABLE_I4_SIZES[idx],
                (t.coeff_rows(), t.coeff_cols()),
                "{t:?} matrix size for parameters index {idx}"
            );
        }
        assert!(seen.iter().all(|s| *s), "every Table I.4 slot must be used");
    }

    /// Table I.7's Order-ID column (recovered from `part1.md`, since the LaTeX
    /// destroys it) must be consistent: every ID is used, IDs are `0..13`, and
    /// all transforms sharing an ID share a coefficient shape — which is what
    /// makes one natural order per ID well defined.
    #[test]
    fn order_ids_cover_table_i7_and_share_a_coefficient_shape() {
        let mut dims: [Option<(usize, usize)>; NUM_ORDER_IDS] = [None; NUM_ORDER_IDS];
        for t in TransformType::ALL {
            let id = t.order_id();
            assert!(id < NUM_ORDER_IDS, "{t:?} order id {id}");
            let shape = (t.coeff_rows(), t.coeff_cols());
            match dims[id] {
                None => dims[id] = Some(shape),
                Some(existing) => assert_eq!(existing, shape, "order id {id} shape for {t:?}"),
            }
        }
        for (id, d) in dims.iter().enumerate() {
            let (rows, cols) = d.unwrap_or_else(|| panic!("order id {id} is unused"));
            assert_eq!(order_id_dims(id), Some((cols, rows)), "order_id_dims({id})");
        }
    }

    // -- I.3.2 natural ordering --------------------------------------------

    /// For every Order ID the natural order must be a bijection onto
    /// `0..bwidth*bheight`, its first `cx*cy` entries must be exactly the LLF
    /// sub-rectangle in `y*cx + x` order, and its tail must be sorted by
    /// `(key1, key2)`. An off-by-one in the LLF/HF split shifts every
    /// coefficient in the block, so all three halves are asserted separately.
    #[test]
    fn natural_order_is_a_bijection_with_an_llf_prefix() {
        for id in 0..NUM_ORDER_IDS {
            let (bwidth, bheight) = order_id_dims(id).expect("every ID is defined");
            let order = natural_coeff_order(bwidth, bheight);
            let n = bwidth * bheight;
            assert_eq!(order.len(), n, "order {id} length");

            let mut seen = vec![false; n];
            for pos in &order {
                let i = *pos as usize;
                assert!(i < n, "order {id} entry {i} out of range");
                assert!(!seen[i], "order {id} repeats cell {i}");
                seen[i] = true;
            }

            let (cx, cy) = (bwidth / 8, bheight / 8);
            for (k, entry) in order.iter().enumerate().take(cx * cy) {
                let (x, y) = (k % cx, k / cx);
                assert_eq!(
                    *entry as usize,
                    y * bwidth + x,
                    "order {id} LLF prefix position {k}"
                );
            }

            let scale = cx.max(cy) as i64;
            let key = |linear: u32| {
                let (x, y) = ((linear as usize) % bwidth, (linear as usize) / bwidth);
                let sx = (x as i64) * scale / (cx as i64);
                let sy = (y as i64) * scale / (cy as i64);
                let k1 = sx + sy;
                let k2 = if k1 % 2 == 1 { sy - sx } else { sx - sy };
                (k1, k2)
            };
            for w in order[cx * cy..].windows(2) {
                assert!(key(w[0]) <= key(w[1]), "order {id} HF tail is not sorted");
            }
        }
    }

    /// The 8x8 orders (IDs 0 and 1) have a single LLF cell — the DC — and the
    /// HF tail starts at index 1. Guards the `cx == cy == 1` corner where the
    /// boustrophedon degenerates.
    #[test]
    fn natural_order_8x8_starts_at_dc() {
        let order = natural_coeff_order(8, 8);
        assert_eq!(order.len(), 64);
        assert_eq!(order[0], 0);
        // (1, 0) and (0, 1) have key1 == 1 and key2 == -1 / +1 after the odd
        // negation, so the horizontal neighbour comes first.
        assert_eq!(order[1], 1);
        assert_eq!(order[2], 8);
    }

    // -- I.9.8 AFV basis ---------------------------------------------------

    /// I.9.8 asserts the AFV basis is orthonormal, so orthonormality is a
    /// transcription check with the force of the standard behind it: it fails
    /// on a single wrong digit anywhere in the 256 floats. The two known OCR
    /// defects in `latex/part1.tex` (a stray comma and a digit-splitting space)
    /// are repaired in the table above; this test is the proof they were the
    /// only ones.
    #[test]
    fn afv_basis_is_orthonormal() {
        for (i, row_i) in AFV_BASIS.iter().enumerate() {
            for (j, row_j) in AFV_BASIS.iter().enumerate() {
                let dot: f64 = row_i.iter().zip(row_j.iter()).map(|(a, b)| a * b).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-12,
                    "AFVBasis rows {i}.{j} dot to {dot}, expected {expected}"
                );
            }
        }
    }

    // -- I.9 reconstruction ------------------------------------------------

    /// **The headline I.9 test.** For *every* transform type, a coefficient
    /// matrix holding only `c(0, 0) = v` must reconstruct the constant block
    /// `v`.
    ///
    /// For the `DCTRxC` types this is I.7.2's normalization again. For the rest
    /// it is much stronger, because each bespoke reconstruction has to conspire
    /// to produce it: DCT2x2's three butterfly passes, DCT4x4's and Hornuss's
    /// `AuxIDCT2x2` DC split, DCT8x4/DCT4x8's `{c00+c01, c00-c01}` pair, and
    /// AFV's `coeff_afv[0] = 4*c(0,0)` against a basis row of 0.25 plus two
    /// separate IDCTs. It also proves **coverage**: every one of the 64 output
    /// samples is written, since an unwritten sample would still be zero.
    #[test]
    fn dc_only_reconstructs_a_constant_block_for_every_transform() {
        for t in TransformType::ALL {
            let mut coeffs = t.empty_coefficients();
            coeffs.set(0, 0, 1.75);
            let samples = t.samples_from_coefficients(&coeffs);
            assert_eq!(
                (samples.rows(), samples.cols()),
                (t.sample_rows(), t.sample_cols()),
                "{t:?} sample shape"
            );
            for y in 0..samples.rows() {
                for x in 0..samples.cols() {
                    assert_close(samples.at(x, y), 1.75, 2e-3, &format!("{t:?} at ({x},{y})"));
                }
            }
        }
    }

    /// I.9.6 and I.9.7 are each other's transpose. Feeding DCT8x4 the same 8x8
    /// coefficient block as DCT4x8 must give the transposed samples: same
    /// gather, same DC butterfly, only `IDCT_2D`'s `(R, C)` and the placement
    /// axis differ. This is the executable form of the argument behind
    /// [`DCT8X4_HALF_INDEX_IS_LOW_COORDINATE`] — if the two halves were placed
    /// with opposite conventions the symmetry would break.
    #[test]
    fn dct8x4_is_the_transpose_of_dct4x8() {
        let mut rng = Lcg::new(0x8a48_4a88);
        let mut coeffs = CoeffMatrix::zeros(8, 8);
        for slot in coeffs.as_mut_slice() {
            *slot = rng.next(1.0);
        }
        let vertical = TransformType::Dct8x4.samples_from_coefficients(&coeffs);
        let horizontal = TransformType::Dct4x8.samples_from_coefficients(&coeffs);
        for y in 0..8 {
            for x in 0..8 {
                assert_close(
                    vertical.at(x, y),
                    horizontal.at(y, x),
                    1e-5,
                    &format!("DCT8x4({x},{y}) vs DCT4x8({y},{x})"),
                );
            }
        }
    }

    /// Storage-order snapshot for I.9.6/I.9.7. The clause splits the 8x8
    /// coefficient block into two 4x8 halves by *row parity*: half 0 reads rows
    /// 0, 2, 4, 6 and half 1 reads rows 1, 3, 5, 7. So a coefficient placed in
    /// an even row (other than the DC pair) may only affect half 0's samples,
    /// which occupy columns 0..4 for DCT8x4 and rows 0..4 for DCT4x8.
    #[test]
    fn dct8x4_half_placement_snapshot() {
        let mut coeffs = CoeffMatrix::zeros(8, 8);
        coeffs.set(3, 2, 1.0); // column 3, row 2 -> an even row, so half 0
        let vertical = TransformType::Dct8x4.samples_from_coefficients(&coeffs);
        for y in 0..8 {
            for x in 4..8 {
                assert_close(vertical.at(x, y), 0.0, 1e-6, "DCT8x4 half 1 must be silent");
            }
        }
        assert!(
            (0..8).any(|y| (0..4).any(|x| vertical.at(x, y).abs() > 1e-3)),
            "DCT8x4 half 0 must be excited"
        );

        let horizontal = TransformType::Dct4x8.samples_from_coefficients(&coeffs);
        for y in 4..8 {
            for x in 0..8 {
                assert_close(
                    horizontal.at(x, y),
                    0.0,
                    1e-6,
                    "DCT4x8 half 1 must be silent",
                );
            }
        }
    }

    /// I.9.3's `AuxIDCT2x2` is an unnormalized 2x2 Hadamard: applying it to a
    /// single upper-left coefficient spreads `v` over the whole `SxS` window
    /// and leaves everything outside untouched. Pins both the butterfly signs
    /// and the "copy the rest through" rule.
    #[test]
    fn aux_idct_2x2_spreads_and_copies() {
        let mut block = Block8::zeros();
        block.set(0, 0, 1.0);
        block.set(7, 7, -3.0);
        let out = aux_idct_2x2(&block, 2);
        assert_close(out.at(0, 0), 1.0, 0.0, "S=2 top-left");
        assert_close(out.at(1, 1), 1.0, 0.0, "S=2 (1,1)");
        assert_close(out.at(7, 7), -3.0, 0.0, "S=2 leaves the tail alone");
        assert_close(out.at(2, 2), 0.0, 0.0, "S=2 touches nothing else");

        // The four butterfly outputs, from a single non-DC input.
        let mut block = Block8::zeros();
        block.set(1, 0, 1.0); // c01 for the S = 2 pass
        let out = aux_idct_2x2(&block, 2);
        assert_close(out.at(0, 0), 1.0, 0.0, "r00");
        assert_close(out.at(1, 0), 1.0, 0.0, "r01");
        assert_close(out.at(0, 1), -1.0, 0.0, "r10");
        assert_close(out.at(1, 1), -1.0, 0.0, "r11");
    }

    /// I.9.5 Hornuss, hand-computed. With `dcs` zero and one residual
    /// coefficient `r` at `(ix, iy) = (1, 0)` of sub-block (0, 0), the centre
    /// sample is `-r/16` and every other cell of that 4x4 is its own
    /// coefficient plus the centre. Pins the `/16.0` and the special (0, 0)
    /// cell, which reads `coefficients(x + 2, y + 2)` rather than its own
    /// position.
    #[test]
    fn hornuss_residual_and_centre_are_hand_computable() {
        let mut coeffs = CoeffMatrix::zeros(8, 8);
        coeffs.set(2, 0, 4.0); // (ix, iy) = (1, 0) of sub-block (0, 0)
        let s = TransformType::Hornuss.samples_from_coefficients(&coeffs);
        let centre = -4.0f32 / 16.0;
        assert_close(s.at(1, 1), centre, 1e-6, "Hornuss centre");
        assert_close(s.at(1, 0), 4.0 + centre, 1e-6, "Hornuss (ix,iy)=(1,0)");
        // (0, 0) reads coefficients(x + 2, y + 2) == coefficients(2, 2) == 0.
        assert_close(s.at(0, 0), centre, 1e-6, "Hornuss (0,0) special case");
        assert_close(s.at(3, 3), centre, 1e-6, "Hornuss untouched cell");
    }

    // -- I.8 LLF ------------------------------------------------------------

    /// For the nine non-`DCTRxC` types I.8 states the output equals the input.
    #[test]
    fn llf_is_the_identity_for_the_small_transforms() {
        for t in TransformType::ALL {
            if t.llf_is_transformed() {
                continue;
            }
            let lf = SampleBlock::from_rows_cols(1, 1, vec![-0.75]);
            let llf = llf_from_lf(t, &lf);
            assert_eq!((llf.rows(), llf.cols()), (1, 1), "{t:?} LLF shape");
            assert_close(llf.at(0, 0), -0.75, 0.0, &format!("{t:?} LLF identity"));
        }
    }

    /// **The I.8 proof.** Build a band-limited varblock (its coefficients are
    /// zero outside the LLF sub-rectangle), take the 8x downsampled image as
    /// the DC of each 8x8 block, and run I.8 on it: the result must be exactly
    /// the varblock's own LLF coefficients.
    ///
    /// This pins three things at once: `ScaleF`'s argument (the printed
    /// `ScaleF(y, cy)` divides by zero from DCT16x16 up — see
    /// [`LLF_SCALEF_ARG_IS_VARBLOCK_DIMENSION`]), the crossed order of the two
    /// scale factors relative to the index order, and the transpose of the LF
    /// rectangle for portrait varblocks. Non-square shapes are included
    /// precisely because a swapped `ScaleF` pair is invisible on square ones.
    #[test]
    fn llf_matches_the_varblocks_own_low_frequency_coefficients() {
        let mut rng = Lcg::new(0x11f_c0de);
        for t in TransformType::ALL {
            if !t.llf_is_transformed() {
                continue;
            }
            let (rows, cols) = (t.sample_rows(), t.sample_cols());
            let (cr, cc) = (t.coeff_rows(), t.coeff_cols());
            let (block_rows, block_cols) = t.block_dims();
            let (cx, cy) = (cc / 8, cr / 8);

            // A random LLF-only coefficient matrix, and the samples it decodes
            // to. Those samples are band-limited by construction.
            let mut coeffs = vec![0.0f32; cr * cc];
            for y in 0..cy {
                for x in 0..cx {
                    coeffs[y * cc + x] = rng.next(1.0);
                }
            }
            let samples = crate::dct::idct_2d_raw(&coeffs, rows, cols);

            // The 8x downsampled image: the DC of each 8x8 block, in image
            // orientation.
            let mut lf = vec![0.0f32; block_rows * block_cols];
            for br in 0..block_rows {
                for bc in 0..block_cols {
                    let mut block = [0.0f32; 64];
                    for y in 0..8 {
                        for x in 0..8 {
                            block[y * 8 + x] = samples[(br * 8 + y) * cols + bc * 8 + x];
                        }
                    }
                    lf[br * block_cols + bc] = dct_2d_raw(&block, 8, 8)[0];
                }
            }

            let llf = llf_from_lf(t, &SampleBlock::from_rows_cols(block_rows, block_cols, lf));
            assert_eq!((llf.rows(), llf.cols()), (cy, cx), "{t:?} LLF shape");
            for y in 0..cy {
                for x in 0..cx {
                    assert_close(
                        llf.at(x, y),
                        coeffs[y * cc + x],
                        2e-3,
                        &format!("{t:?} LLF ({x},{y})"),
                    );
                }
            }
        }
    }

    /// `ScaleF` must be finite and at least 1 everywhere it is reachable. The
    /// literal reading of I.8 is infinite at `c == cx / 2`, so this is the
    /// regression guard for [`LLF_SCALEF_ARG_IS_VARBLOCK_DIMENSION`].
    #[test]
    fn scale_f_is_finite_over_the_reachable_domain() {
        for lf_count in [1usize, 2, 4, 8, 16, 32] {
            for c in 0..lf_count {
                let s = scale_f(c, lf_count);
                assert!(s.is_finite(), "ScaleF({c}, {lf_count}) is {s}");
                assert!((1.0..2.0).contains(&s), "ScaleF({c}, {lf_count}) is {s}");
            }
        }
        assert_close(scale_f(0, 32), 1.0, 0.0, "ScaleF(0, b) is 1");
    }

    /// The cached tables are the closed form itself: every tabled `(c, b)`
    /// pair is bit-identical to a direct evaluation, so caching cannot move a
    /// single LLF coefficient.
    #[test]
    fn scale_f_cache_is_bit_identical_to_the_closed_form() {
        for &lf_count in &[1usize, 2, 4, 8, 16, 32] {
            for c in 0..lf_count {
                assert_eq!(
                    scale_f(c, lf_count).to_bits(),
                    scale_f_closed_form(c, lf_count).to_bits(),
                    "ScaleF({c}, {lf_count}) cache/closed-form"
                );
            }
        }
        // Untabled arguments still evaluate directly.
        assert_eq!(scale_f(0, 3).to_bits(), scale_f_closed_form(0, 3).to_bits());
    }

    // -- Matrix types -------------------------------------------------------

    /// Storage order is row-major with `at(x, y)` reading `data[y*cols + x]`,
    /// and out-of-range access is a defined no-op rather than a panic or a
    /// wrapped index.
    #[test]
    fn matrix_storage_order_is_row_major_in_x_then_y() {
        let m = CoeffMatrix::from_landscape(2, 4, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        assert_close(m.at(3, 0), 3.0, 0.0, "row 0");
        assert_close(m.at(0, 1), 4.0, 0.0, "row 1");
        assert_close(m.at(4, 0), 0.0, 0.0, "out of range column");
        assert_close(m.at(0, 2), 0.0, 0.0, "out of range row");

        let mut s = SampleBlock::zeros(4, 2);
        s.set(1, 3, 9.0);
        assert_close(s.at(1, 3), 9.0, 0.0, "sample write");
        assert_eq!(s.as_slice()[7], 9.0, "sample storage index");
    }

    /// `write_llf` lands in the top-left corner and leaves the HF cells alone.
    #[test]
    fn write_llf_targets_the_top_left_corner() {
        let mut coeffs = TransformType::Dct16x8.empty_coefficients();
        let llf = CoeffMatrix::from_landscape(1, 2, vec![5.0, 6.0]);
        coeffs.write_llf(&llf);
        assert_close(coeffs.at(0, 0), 5.0, 0.0, "LLF (0,0)");
        assert_close(coeffs.at(1, 0), 6.0, 0.0, "LLF (1,0)");
        assert_close(coeffs.at(2, 0), 0.0, 0.0, "first HF cell stays zero");
        assert_close(coeffs.at(0, 1), 0.0, 0.0, "second row stays zero");
    }
}
