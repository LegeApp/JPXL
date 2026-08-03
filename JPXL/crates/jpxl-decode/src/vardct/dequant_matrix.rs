//! Dequantization matrices and their defaults (18181-1 I.2.4, I.2.5).
//!
//! An `HfGlobal` bundle carries 17 sets of parameters, one per row of
//! Table I.4, each describing how to build a *weights* matrix per channel. The
//! dequantization matrix a decoder actually multiplies by is the element-wise
//! reciprocal of that weights matrix — except in `RAW` mode, where the stream
//! supplies the dequantization matrix directly.
//!
//! ```text
//! Table I.5 — encoding_mode, read as u(3), and the indices it may be used for
//! 0 Library  all            4 DCT4x8  0,1,2,3,9,10
//! 1 Hornuss  0,1,2,3,9,10   5 AFV     0,1,2,3,9,10
//! 2 DCT2     0,1,2,3,9,10   6 DCT     all
//! 3 DCT4     0,1,2,3,9,10   7 RAW     all
//! ```
//!
//! # Weights, matrices, and which is which
//!
//! [`WeightMatrix`] and [`DequantMatrix`] are separate types with no
//! conversion but [`WeightMatrix::reciprocal`]. AGENTS.md names "quant weight
//! confused with its reciprocal" as a failure that killed the previous attempt;
//! a decoder that multiplies by the weights instead of the matrix produces a
//! plausible-looking image with inverted frequency response, which no smoke
//! test catches.
//!
//! # Orientation
//!
//! Table I.4 prints its sizes as *rows columns*, and the weights code indexes
//! `weights(x, y)` with `x` the column. Both types here store row-major and
//! expose `at(x, y)` in the clause's order, matching `jpxl_core::varblock`'s
//! [`CoeffMatrix`](jpxl_core::varblock::CoeffMatrix). Every Table I.4 size
//! equals the corresponding transform's `(coeff_rows, coeff_cols)`, which is
//! asserted in the tests.
//!
//! # Precision
//!
//! Parameters are `F16()` on the wire but the I.2.5 defaults are printed to as
//! many as 21 significant digits, so parameters are held as `f64` and the whole
//! weights computation runs in `f64`. Only the final matrix is narrowed to the
//! `f32` of the coefficient pipeline.
//!
//! # Transcription
//!
//! Every constant below was checked against `latex/part1.tex`,
//! `markdowns/standard-markdowns/part1.md`, the text-only transcription PDF,
//! and — for the rows where those disagreed or printed a non-number — a
//! page-ranged read of the original scan. See [`DCT128X256_DEFAULT_BASES_AS_PRINTED`]
//! for the one place where the printed standard is internally inconsistent.

// Every index below is a loop bound over a locally allocated buffer whose
// dimensions were computed in the same function; the stream-derived indices go
// through `get`.
#![allow(clippy::indexing_slicing)]

use jpxl_bitstream::trace_field;
use jpxl_bitstream::{BitReader, read_bool, read_f16_as_f32};
use jpxl_core::limits::AllocGuard;
use jpxl_core::varblock::{NUM_DEQUANT_MATRICES, TransformType};

use crate::error::{DecodeError, Result};
use crate::frame::geometry::FrameGeometry;
use crate::frame::stream_index;
use crate::modular::{Channel, ChannelSpec, ModularOptions, TreeSource, decode_sub_bitstream_with};

use super::block_ctx::read_num_hf_presets;

// ---------------------------------------------------------------------------
// Flip points
// ---------------------------------------------------------------------------

/// **Flip point — the DC weight of encoding mode DCT2.**
///
/// I.2.4 lists where `params(c, 0..5)` land in the 8x8 DCT2 weights matrix.
/// The six rules cover all 64 positions *except* `(0, 0)`, and unlike the
/// Hornuss paragraph immediately below them the clause never says what `(0, 0)`
/// is. I.2.4 also requires that no entry of the resulting dequantization matrix
/// be non-positive or infinite, so it cannot be left at zero.
///
/// * `true` (shipped): `(0, 0)` is 1, matching the Hornuss rule in the very
///   next sentence.
/// * `false`: some other value.
///
/// This is unobservable in decoded pixels. Position `(0, 0)` of a varblock is
/// its LLF coefficient, dequantized by the I.5.2 LF path, and I.4 only ever
/// applies the dequantization matrix at natural-order positions from
/// `num_blocks` onwards. The constant exists so that a future reader does not
/// mistake a deliberate choice for an oversight.
pub const DCT2_DC_WEIGHT_IS_ONE: bool = true;

/// **Flip point — Table I.6's DCT128x256 bases contradict the rest of the
/// table.**
///
/// The printed row for parameter index 16 is
/// `{61435.5921973295970, SeqA}, {24209.44206460261196, SeqB},
/// {12979.84647584004484, SeqC}`. Two of those three numbers do not fit the
/// pattern every other row obeys:
///
/// * Each size step doubles the base. Index 11 -> 13 -> 15 doubles exactly in
///   all three channels, and so does index 12 -> 14. From index 14 the doubled
///   bases are `61435.592197329584`, `22389.44206460261196` and
///   `11679.84647584004484`.
/// * The ratio between the square family (11/13/15) and the oblong family
///   (12/14/16) is constant per channel: 1.56039, 1.49717 and 1.53873. The
///   printed index-16 values give 1.38462 for both the Y and B channels, a
///   value that appears nowhere else.
/// * The *fractional* parts of the printed values are exactly the doubled
///   fractional parts (`.44206460261196` and `.84647584004484`). Only the
///   integer parts differ — `24209` for `22389`, `12979` for `11679`.
///
/// This is not a transcription artefact: the original scan (page 64) prints
/// `24209` and `12979` plainly, as do all three text conversions. It is a
/// defect in the published table, or a deliberate value with no stated reason.
///
/// * `true` (shipped): the printed values, because the standard is the source
///   of truth and no oracle evidence exists yet.
/// * `false`: the values implied by the doubling and the family ratio.
///
/// Only the DCT128x256 and DCT256x128 transforms are affected, so a probe needs
/// a fixture containing a 128x256 varblock. See
/// `docs/experiments/2026-08-03-i25-dct128x256-bases.md`.
pub const DCT128X256_DEFAULT_BASES_AS_PRINTED: bool = true;

// ---------------------------------------------------------------------------
// Table I.4 — sizes
// ---------------------------------------------------------------------------

/// Table I.4 matrix sizes as `(rows, columns)`, indexed by parameters index.
const MATRIX_SIZES: [(usize, usize); NUM_DEQUANT_MATRICES] = [
    (8, 8),     // 0  DCT8x8
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

/// Table I.4 matrix size for a parameters index, as `(rows, columns)`.
///
/// # Errors
///
/// [`DecodeError::FieldOutOfRange`] if `index` is not below
/// [`NUM_DEQUANT_MATRICES`].
pub fn matrix_size(index: usize) -> Result<(usize, usize)> {
    MATRIX_SIZES
        .get(index)
        .copied()
        .ok_or_else(|| DecodeError::out_of_range("parameters index", "I.2.4", as_u64(index)))
}

// ---------------------------------------------------------------------------
// Table I.5 — encoding modes
// ---------------------------------------------------------------------------

/// The `encoding_mode` of Table I.5, read as `u(3)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EncodingMode {
    /// Mode 0: use the I.2.5 default parameters for this index.
    Library = 0,
    /// Mode 1.
    Hornuss = 1,
    /// Mode 2.
    Dct2 = 2,
    /// Mode 3.
    Dct4 = 3,
    /// Mode 4.
    Dct4x8 = 4,
    /// Mode 5.
    Afv = 5,
    /// Mode 6: the generic radial DCT weights.
    Dct = 6,
    /// Mode 7: the matrix comes from a modular sub-bitstream.
    Raw = 7,
}

impl EncodingMode {
    /// Decodes the `u(3)` of I.2.4.
    ///
    /// # Errors
    ///
    /// [`DecodeError::FieldOutOfRange`] if the value exceeds 7 — impossible for
    /// a 3-bit field, but the conversion is written as a rejection so that a
    /// future widening of the field cannot fall through silently.
    pub fn from_bits(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::Library),
            1 => Ok(Self::Hornuss),
            2 => Ok(Self::Dct2),
            3 => Ok(Self::Dct4),
            4 => Ok(Self::Dct4x8),
            5 => Ok(Self::Afv),
            6 => Ok(Self::Dct),
            7 => Ok(Self::Raw),
            _ => Err(DecodeError::out_of_range(
                "encoding_mode",
                "I.2.4",
                u64::from(value),
            )),
        }
    }

    /// Table I.5's "valid index" column.
    ///
    /// The five small-block modes are only meaningful for the six 8x8 rows of
    /// Table I.4; `Library`, `DCT` and `RAW` apply everywhere.
    #[must_use]
    pub fn allows_index(self, index: usize) -> bool {
        match self {
            Self::Library | Self::Dct | Self::Raw => index < NUM_DEQUANT_MATRICES,
            Self::Hornuss | Self::Dct2 | Self::Dct4 | Self::Dct4x8 | Self::Afv => {
                matches!(index, 0 | 1 | 2 | 3 | 9 | 10)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Matrix types
// ---------------------------------------------------------------------------

/// A quantization **weights** matrix: what I.2.4's `GetDctQuantWeights` and the
/// per-mode rules build. Larger values mean coarser quantization.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightMatrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

/// A **dequantization** matrix: the element-wise reciprocal of a
/// [`WeightMatrix`], and what I.5.3 multiplies a dequantized HF coefficient by.
#[derive(Debug, Clone, PartialEq)]
pub struct DequantMatrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl WeightMatrix {
    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0f64; rows * cols],
        }
    }

    /// Number of rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Weight at column `x`, row `y`; `0.0` outside the matrix.
    #[must_use]
    pub fn at(&self, x: usize, y: usize) -> f64 {
        if x < self.cols && y < self.rows {
            self.data[y * self.cols + x]
        } else {
            0.0
        }
    }

    fn set(&mut self, x: usize, y: usize, value: f64) {
        if x < self.cols && y < self.rows {
            self.data[y * self.cols + x] = value;
        }
    }

    /// I.2.4's final step: the element-wise reciprocal.
    ///
    /// # Errors
    ///
    /// [`DecodeError::FieldOutOfRange`] if any entry is non-positive or not
    /// finite, or if any reciprocal is not finite. I.2.4 states that no such
    /// value occurs, so this is the check that turns a bad parameter set — or a
    /// bad transcription of Table I.6 — into a rejection instead of a NaN
    /// spreading through the image.
    // The narrowing to f32 is the deliberate hand-off to the sample pipeline.
    #[allow(clippy::cast_possible_truncation)]
    pub fn reciprocal(&self) -> Result<DequantMatrix> {
        let mut data = Vec::with_capacity(self.data.len());
        for &w in &self.data {
            if !w.is_finite() || w <= 0.0 {
                return Err(DecodeError::out_of_range("weights", "I.2.4", 0));
            }
            let inv = 1.0 / w;
            if !inv.is_finite() || inv <= 0.0 {
                return Err(DecodeError::out_of_range(
                    "dequantization matrix",
                    "I.2.4",
                    0,
                ));
            }
            data.push(inv as f32);
        }
        Ok(DequantMatrix {
            rows: self.rows,
            cols: self.cols,
            data,
        })
    }
}

impl DequantMatrix {
    /// Number of rows; equals the transform's `coeff_rows`.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns; equals the transform's `coeff_cols`.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Multiplier at column `x`, row `y`; `0.0` outside the matrix.
    #[must_use]
    pub fn at(&self, x: usize, y: usize) -> f32 {
        if x < self.cols && y < self.rows {
            self.data[y * self.cols + x]
        } else {
            0.0
        }
    }

    /// Row-major backing storage.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Builds a matrix directly from row-major data, as `RAW` mode requires.
    ///
    /// # Errors
    ///
    /// [`DecodeError::FieldOutOfRange`] if `data` is not `rows * cols` long or
    /// contains a non-positive or non-finite value.
    pub fn from_rows_cols(rows: usize, cols: usize, data: Vec<f32>) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(DecodeError::out_of_range(
                "RAW matrix length",
                "I.2.4",
                as_u64(data.len()),
            ));
        }
        if data.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err(DecodeError::out_of_range("RAW matrix value", "I.2.4", 0));
        }
        Ok(Self { rows, cols, data })
    }
}

// ---------------------------------------------------------------------------
// I.2.4 — the weights computation
// ---------------------------------------------------------------------------

/// I.2.4's `Mult(v)`: `1 + v` for positive `v`, `1 / (1 - v)` otherwise.
///
/// Always positive, which is what makes every band positive and hence every
/// weight positive. The `latex/part1.tex` rendering of the negative branch is
/// `1 / (1 - wv)`; `part1.md` and the original scan both print `1 / (1 - v)`.
fn mult(v: f64) -> f64 {
    if v > 0.0 { 1.0 + v } else { 1.0 / (1.0 - v) }
}

/// I.2.4's `Interpolate(pos, max, bands)`: geometric interpolation between two
/// adjacent bands.
///
/// The exponentiation is `pow(B / A, frac_index)`. `latex/part1.tex` renders it
/// as `pow(B / 4B, frac_index)`, which is not an expression; `part1.md` and the
/// original scan agree on `B / A`.
///
/// # Errors
///
/// [`DecodeError::FieldOutOfRange`] if `bands` is empty, or if `pos` scales to
/// an index outside `bands` — the callers keep `pos < max` by adding `1e-6` to
/// the range, so this can only fire on a corrupt parameter set.
fn interpolate(pos: f64, max: f64, bands: &[f64]) -> Result<f64> {
    let last = bands
        .len()
        .checked_sub(1)
        .ok_or_else(|| DecodeError::out_of_range("bands", "I.2.4", 0))?;
    if last == 0 {
        return Ok(bands[0]);
    }
    let scaled_pos = pos * (last as f64) / max;
    let scaled_index = scaled_pos.floor();
    if !(scaled_index >= 0.0 && scaled_index < last as f64) {
        return Err(DecodeError::out_of_range("Interpolate index", "I.2.4", 0));
    }
    // `scaled_index` is a non-negative integral f64 strictly below `last`, so
    // the conversion is exact; the cast is checked above rather than by the
    // compiler.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let idx = scaled_index as usize;
    let frac_index = scaled_pos - scaled_index;
    let a = bands[idx];
    let b = bands[idx + 1];
    Ok(a * (b / a).powf(frac_index))
}

/// I.2.4's band sequence: `bands[0] = params[0]`, then
/// `bands[i] = bands[i - 1] * Mult(params[i])`.
///
/// # Errors
///
/// [`DecodeError::FieldOutOfRange`] if `params` is empty or a band is not
/// strictly positive; the clause asserts `bands[i] > 0`.
fn bands_from(params: &[f64]) -> Result<Vec<f64>> {
    let first = *params
        .first()
        .ok_or_else(|| DecodeError::out_of_range("dct_params row", "I.2.4", 0))?;
    if !first.is_finite() || first <= 0.0 {
        return Err(DecodeError::out_of_range("bands[0]", "I.2.4", 0));
    }
    let mut bands = Vec::with_capacity(params.len());
    bands.push(first);
    for (i, &p) in params.iter().enumerate().skip(1) {
        let next = bands[i - 1] * mult(p);
        if !next.is_finite() || next <= 0.0 {
            return Err(DecodeError::out_of_range("bands", "I.2.4", as_u64(i)));
        }
        bands.push(next);
    }
    Ok(bands)
}

/// The upper end of `Interpolate`'s range in `GetDctQuantWeights`.
///
/// `sqrt(2) + 1e-6` — the `1e-6` keeps the corner sample `(X-1, Y-1)`, whose
/// distance is exactly `sqrt(2)`, strictly inside the last band interval so
/// that `bands[scaled_index + 1]` always exists.
fn dct_weight_range() -> f64 {
    core::f64::consts::SQRT_2 + 1e-6
}

/// I.2.4's `GetDctQuantWeights`, for a matrix of `cols` columns and `rows` rows.
///
/// The clause writes the dimensions as `X x Y` and loops `x < X`, `y < Y`, so
/// `X` is the column count. Table I.4 prints its sizes as rows-then-columns, so
/// the two are deliberately named here.
///
/// # Errors
///
/// [`DecodeError::FieldOutOfRange`] if either dimension is below 2 (the clause
/// divides by `X - 1` and `Y - 1`) or the parameters are degenerate.
fn dct_quant_weights(cols: usize, rows: usize, params: &[f64]) -> Result<WeightMatrix> {
    if cols < 2 || rows < 2 {
        return Err(DecodeError::out_of_range(
            "weights matrix size",
            "I.2.4",
            as_u64(cols * rows),
        ));
    }
    let bands = bands_from(params)?;
    let max = dct_weight_range();
    let mut weights = WeightMatrix::zeros(rows, cols);
    let x_span = (cols - 1) as f64;
    let y_span = (rows - 1) as f64;
    for y in 0..rows {
        for x in 0..cols {
            let dx = (x as f64) / x_span;
            let dy = (y as f64) / y_span;
            // Written as the clause writes it, not fused: a fused multiply-add
            // would round differently from `sqrt(dx*dx + dy*dy)`.
            let distance = (dx * dx + dy * dy).sqrt();
            weights.set(x, y, interpolate(distance, max, &bands)?);
        }
    }
    Ok(weights)
}

/// I.2.4's AFV frequency table, used to place the 16 low-frequency weights.
const AFV_FREQS: [f64; 16] = [
    0.0,
    0.0,
    0.851_777_889_032_429_6,
    5.377_784_365_068_04,
    0.0,
    0.0,
    4.734_747_904_497_923,
    5.449_245_381_693_219,
    1.659_827_026_747_933_1,
    4.0,
    7.275_749_096_817_861,
    10.423_227_632_456_525,
    2.662_932_286_148_962,
    7.630_657_783_650_829,
    8.962_388_608_184_032,
    12.971_662_025_702_35,
];

/// The AFV interpolation range, `[lo, hi]` from the clause. `lo` is
/// `AFV_FREQS[2]` and `hi` is `AFV_FREQS[15]`.
const AFV_LO: f64 = 0.851_777_889_032_429_6;
/// See [`AFV_LO`].
const AFV_HI: f64 = 12.971_662_025_702_35;

// ---------------------------------------------------------------------------
// Parameter sets
// ---------------------------------------------------------------------------

/// One row of Table I.4's parameter list, after reading or defaulting.
#[derive(Debug, Clone, PartialEq)]
pub struct DequantParams {
    mode: EncodingMode,
    /// `dct_params`, three channel rows. Empty where the mode does not use it.
    dct_params: [Vec<f64>; 3],
    /// `dct4x4_params`, three channel rows. AFV only.
    dct4x4_params: [Vec<f64>; 3],
    /// `params`, three channel rows.
    params: [Vec<f64>; 3],
    /// `params.denominator` of RAW mode.
    denominator: f32,
    /// The RAW matrix once the caller has decoded its modular sub-bitstream.
    raw: Option<[Vec<f32>; 3]>,
}

impl DequantParams {
    /// The encoding mode this set was read with.
    ///
    /// Note that `Library` never survives reading: I.2.4's `SetDefaultMode()`
    /// replaces it with the Table I.6 mode for the index, which is what the
    /// subsequent per-mode rules are written against.
    #[must_use]
    pub const fn mode(&self) -> EncodingMode {
        self.mode
    }

    /// `params.denominator`, meaningful only in RAW mode.
    #[must_use]
    pub const fn denominator(&self) -> f32 {
        self.denominator
    }

    fn empty(mode: EncodingMode) -> Self {
        Self {
            mode,
            dct_params: [Vec::new(), Vec::new(), Vec::new()],
            dct4x4_params: [Vec::new(), Vec::new(), Vec::new()],
            params: [Vec::new(), Vec::new(), Vec::new()],
            denominator: 1.0,
            raw: None,
        }
    }

    fn row(rows: &[Vec<f64>; 3], channel: usize) -> Result<&[f64]> {
        rows.get(channel)
            .map(Vec::as_slice)
            .ok_or_else(|| DecodeError::out_of_range("channel", "I.2.4", as_u64(channel)))
    }

    fn param(row: &[f64], i: usize) -> Result<f64> {
        row.get(i)
            .copied()
            .ok_or_else(|| DecodeError::out_of_range("params index", "I.2.4", as_u64(i)))
    }
}

// ---------------------------------------------------------------------------
// I.2.5 — Table I.6 defaults
// ---------------------------------------------------------------------------
//
// Every literal below is the printed standard's digit sequence, verbatim and
// ungrouped, so that it can be diffed against the table by eye. Several are
// printed to more digits than `f64` can represent (`2198.050556016380522`,
// `8996.8725711814115328`); they are kept as printed because this block is
// transcription evidence, not a computation, and rounding them by hand would
// destroy the audit trail. That is what the three lint exemptions are for.
#[allow(
    clippy::excessive_precision,
    clippy::unreadable_literal,
    clippy::inconsistent_digit_grouping
)]
mod defaults {
    /// `dct4x8_params` of I.2.5: three channel rows of four.
    pub const DCT4X8_PARAMS: [[f64; 4]; 3] = [
        [
            2198.050556016380522,
            -0.96269623020744692,
            -0.76194253026666783,
            -0.6551140670773547,
        ],
        [
            764.3655248643528689,
            -0.92630200888366945,
            -0.9675229603596517,
            -0.27845290869168118,
        ],
        [
            527.107573587542228,
            -1.4594385811273854,
            -1.450082094097871593,
            -1.5843722511996204,
        ],
    ];

    /// `dct4x4_params` of I.2.5.
    ///
    /// The zeros are printed as the letter `O` in both text conversions
    /// (`{2200, 0, 0, O}`); the original scan prints digits.
    pub const DCT4X4_PARAMS: [[f64; 4]; 3] = [
        [2200.0, 0.0, 0.0, 0.0],
        [392.0, 0.0, 0.0, 0.0],
        [112.0, -0.25, -0.25, -0.5],
    ];

    /// `SeqA` of I.2.5.
    pub const SEQ_A: [f64; 7] = [
        -1.025,
        -0.78,
        -0.65012,
        -0.19041574084286472,
        -0.20819395464,
        -0.421064,
        -0.32733845535848671,
    ];

    /// `SeqB` of I.2.5.
    pub const SEQ_B: [f64; 7] = [
        -0.3041958212306401,
        -0.3633036457487539,
        -0.35660379990111464,
        -0.3443074455424403,
        -0.33699592683512467,
        -0.30180866526242109,
        -0.27321683125358037,
    ];

    /// `SeqC` of I.2.5.
    pub const SEQ_C: [f64; 7] = [-1.2, -1.2, -0.8, -0.7, -0.7, -0.4, -0.5];

    /// The `{base, SeqA/SeqB/SeqC}` bases of Table I.6 for parameter indices
    /// 11..=16, as printed, in index order.
    ///
    /// Keeping all six rows in one array is what makes the doubling relation
    /// visible: 11 -> 13 -> 15 and 12 -> 14 -> 16 each double per size step, in
    /// every channel except the two flagged by
    /// [`DCT128X256_DEFAULT_BASES_AS_PRINTED`](super::DCT128X256_DEFAULT_BASES_AS_PRINTED).
    pub const LARGE_DCT_BASES: [[f64; 3]; 6] = [
        // 11 DCT64x64
        [
            23966.1665298448605,
            8380.19148390090414,
            4493.02378009847706,
        ],
        // 12 DCT32x64, DCT64x32
        [
            15358.89804933239925,
            5597.360516150652990,
            2919.961618960011210,
        ],
        // 13 DCT128x128
        [
            47932.3330596897210,
            16760.38296780180828,
            8986.04756019695412,
        ],
        // 14 DCT64x128, DCT128x64
        [
            30717.796098664792,
            11194.72103230130598,
            5839.92323792002242,
        ],
        // 15 DCT256x256
        [
            95864.6661193794420,
            33520.76593560361656,
            17972.09512039390824,
        ],
        // 16 DCT128x256, DCT256x128 — see the flip point.
        [
            61435.5921973295970,
            24209.44206460261196,
            12979.84647584004484,
        ],
    ];

    /// Index 16's bases implied by the doubling relation of every other row.
    pub const LARGE_DCT_BASES_16_DOUBLED: [f64; 3] = [
        61435.5921973295970,
        22389.44206460261196,
        11679.84647584004484,
    ];

    /// Table I.6's DCT8x8 `dct_params`.
    pub const DCT8X8_PARAMS: [[f64; 6]; 3] = [
        [3150.0, 0.0, -0.4, -0.4, -0.4, -2.0],
        [560.0, 0.0, -0.3, -0.3, -0.3, -0.3],
        [512.0, -2.0, -1.0, 0.0, -1.0, -2.0],
    ];

    /// Table I.6's DCT16x16 `dct_params`.
    ///
    /// `part1.md` prints these numbers in a different order: the two-column PDF
    /// layout interleaved the continuation lines. The order below is the one
    /// the original scan prints and the only one that yields three well-formed
    /// rows with descending bases.
    pub const DCT16X16_PARAMS: [[f64; 7]; 3] = [
        [
            8996.8725711814115328,
            -1.3000777393353804,
            -0.49424529824571225,
            -0.439093774457103443,
            -0.6350101832695744,
            -0.90177264050827612,
            -1.6162099239887414,
        ],
        [
            3191.48366296844234752,
            -0.67424582104194355,
            -0.80745813428471001,
            -0.44925837484843441,
            -0.35865440981033403,
            -0.31322389111877305,
            -0.37615025315725483,
        ],
        [
            1157.50408145487200256,
            -2.0531423165804414,
            -1.4,
            -0.50687130033378396,
            -0.42708730624733904,
            -1.4856834539296244,
            -4.9209142884401604,
        ],
    ];

    /// Table I.6's DCT32x32 `dct_params`.
    pub const DCT32X32_PARAMS: [[f64; 8]; 3] = [
        [
            15718.40830982518931456,
            -1.025,
            -0.98,
            -0.9012,
            -0.4,
            -0.48819395464,
            -0.421064,
            -0.27,
        ],
        [
            7305.7636810695983104,
            -0.8041958212306401,
            -0.7633036457487539,
            -0.55660379990111464,
            -0.49785304658857626,
            -0.43699592683512467,
            -0.40180866526242109,
            -0.27321683125358037,
        ],
        [
            3803.53173721215041536,
            -3.060733579805728,
            -2.0413270132490346,
            -2.0235650159727417,
            -0.5495389509954993,
            -0.4,
            -0.4,
            -0.3,
        ],
    ];

    /// Table I.6's DCT16x8 / DCT8x16 `dct_params`.
    pub const DCT16X8_PARAMS: [[f64; 7]; 3] = [
        [7240.7734393502, -0.7, -0.7, -0.2, -0.2, -0.2, -0.5],
        [1448.15468787004, -0.5, -0.5, -0.5, -0.2, -0.2, -0.2],
        [506.854140754517, -1.4, -0.2, -0.5, -0.5, -1.5, -3.6],
    ];

    /// Table I.6's DCT32x8 / DCT8x32 `dct_params`.
    ///
    /// The B-channel base is printed with a stray space in both text
    /// conversions (`3397.776032753087 20128`); the scan prints
    /// `3397.77603275308720128`.
    pub const DCT32X8_PARAMS: [[f64; 8]; 3] = [
        [
            16283.2494710648897,
            -1.7812845336559429,
            -1.6309059012653515,
            -1.0382179034313539,
            -0.85,
            -0.7,
            -0.9,
            -1.2360638576849587,
        ],
        [
            5089.15750884921511936,
            -0.320049391452786891,
            -0.35362849922161446,
            -0.30340000000000003,
            -0.61,
            -0.5,
            -0.5,
            -0.6,
        ],
        [
            3397.77603275308720128,
            -0.321327362693153371,
            -0.34507619223117997,
            -0.70340000000000003,
            -0.9,
            -1.0,
            -1.0,
            -1.1754605576265209,
        ],
    ];

    /// Table I.6's DCT16x32 / DCT32x16 `dct_params`.
    pub const DCT16X32_PARAMS: [[f64; 8]; 3] = [
        [
            13844.97076442300573,
            -0.97113799999999995,
            -0.658,
            -0.42026,
            -0.22712,
            -0.2206,
            -0.226,
            -0.6,
        ],
        [
            4798.964084220744293,
            -0.61125308982767057,
            -0.83770786552491361,
            -0.79014862079498627,
            -0.2692727459704829,
            -0.38272769465388551,
            -0.22924222653091453,
            -0.20719098826199578,
        ],
        [
            1807.236946760964614,
            -1.2,
            -1.2,
            -0.7,
            -0.7,
            -0.7,
            -0.4,
            -0.5,
        ],
    ];

    /// Table I.6's Hornuss `params`.
    pub const HORNUSS_PARAMS: [[f64; 3]; 3] = [
        [280.0, 3160.0, 3160.0],
        [60.0, 864.0, 864.0],
        [18.0, 200.0, 200.0],
    ];

    /// Table I.6's DCT2 `params`.
    ///
    /// The X-channel row is printed
    /// `{3840.0, 2560.0, 1280.0, 64.0.0, 480.0, 300.0}` in both text
    /// conversions; `64.0.0` is not a number. The scan prints `640.0`.
    pub const DCT2_PARAMS: [[f64; 6]; 3] = [
        [3840.0, 2560.0, 1280.0, 640.0, 480.0, 300.0],
        [960.0, 640.0, 320.0, 180.0, 140.0, 120.0],
        [640.0, 320.0, 128.0, 64.0, 32.0, 16.0],
    ];

    /// Table I.6's AFV `params`, three channel rows of nine.
    pub const AFV_PARAMS: [[f64; 9]; 3] = [
        [3072.0, 3072.0, 256.0, 256.0, 256.0, 414.0, 0.0, 0.0, 0.0],
        [1024.0, 1024.0, 50.0, 50.0, 50.0, 58.0, 0.0, 0.0, 0.0],
        [384.0, 384.0, 12.0, 12.0, 12.0, 22.0, -0.25, -0.25, -0.25],
    ];
}

use defaults::{
    AFV_PARAMS, DCT2_PARAMS, DCT4X4_PARAMS, DCT4X8_PARAMS, DCT8X8_PARAMS, DCT16X8_PARAMS,
    DCT16X16_PARAMS, DCT16X32_PARAMS, DCT32X8_PARAMS, DCT32X32_PARAMS, HORNUSS_PARAMS,
    LARGE_DCT_BASES, LARGE_DCT_BASES_16_DOUBLED, SEQ_A, SEQ_B, SEQ_C,
};

/// Builds the Table I.6 default parameter set for one parameters index.
///
/// I.2.5's values are the *post-scaling* ones: the `* 64` that I.2.4 applies
/// while reading `params` and the first column of `dct_params` is part of the
/// read path, not of `SetDefaultMode()`. Hornuss's 280 and DCT8x8's 3150 are on
/// the same scale, which is the cross-check that this is the right reading.
///
/// # Errors
///
/// [`DecodeError::FieldOutOfRange`] if `index` is not a Table I.4 row.
pub fn default_params(index: usize) -> Result<DequantParams> {
    let rows3 = |m: &[[f64; 3]; 3]| [m[0].to_vec(), m[1].to_vec(), m[2].to_vec()];
    let rows4 = |m: &[[f64; 4]; 3]| [m[0].to_vec(), m[1].to_vec(), m[2].to_vec()];
    let rows6 = |m: &[[f64; 6]; 3]| [m[0].to_vec(), m[1].to_vec(), m[2].to_vec()];
    let rows7 = |m: &[[f64; 7]; 3]| [m[0].to_vec(), m[1].to_vec(), m[2].to_vec()];
    let rows8 = |m: &[[f64; 8]; 3]| [m[0].to_vec(), m[1].to_vec(), m[2].to_vec()];
    let rows9 = |m: &[[f64; 9]; 3]| [m[0].to_vec(), m[1].to_vec(), m[2].to_vec()];

    let mut out = match index {
        0 => {
            let mut p = DequantParams::empty(EncodingMode::Dct);
            p.dct_params = rows6(&DCT8X8_PARAMS);
            p
        }
        1 => {
            let mut p = DequantParams::empty(EncodingMode::Hornuss);
            p.params = rows3(&HORNUSS_PARAMS);
            p
        }
        2 => {
            let mut p = DequantParams::empty(EncodingMode::Dct2);
            p.params = rows6(&DCT2_PARAMS);
            p
        }
        3 => {
            let mut p = DequantParams::empty(EncodingMode::Dct4);
            p.dct_params = rows4(&DCT4X4_PARAMS);
            p.params = [vec![1.0, 1.0], vec![1.0, 1.0], vec![1.0, 1.0]];
            p
        }
        4 => {
            let mut p = DequantParams::empty(EncodingMode::Dct);
            p.dct_params = rows7(&DCT16X16_PARAMS);
            p
        }
        5 => {
            let mut p = DequantParams::empty(EncodingMode::Dct);
            p.dct_params = rows8(&DCT32X32_PARAMS);
            p
        }
        6 => {
            let mut p = DequantParams::empty(EncodingMode::Dct);
            p.dct_params = rows7(&DCT16X8_PARAMS);
            p
        }
        7 => {
            let mut p = DequantParams::empty(EncodingMode::Dct);
            p.dct_params = rows8(&DCT32X8_PARAMS);
            p
        }
        8 => {
            let mut p = DequantParams::empty(EncodingMode::Dct);
            p.dct_params = rows8(&DCT16X32_PARAMS);
            p
        }
        9 => {
            let mut p = DequantParams::empty(EncodingMode::Dct4x8);
            p.dct_params = rows4(&DCT4X8_PARAMS);
            p.params = [vec![1.0], vec![1.0], vec![1.0]];
            p
        }
        10 => {
            // Table I.6 names `dct4x8_params` in the dct_params column and
            // gives the 3x9 `params` explicitly, but says nothing about AFV's
            // third input. `dct4x4_params` is the only other named set in
            // I.2.5, and AFV's `weights4x4` is the only consumer of it, so the
            // two named sets exist precisely to feed AFV's two sub-matrices.
            let mut p = DequantParams::empty(EncodingMode::Afv);
            p.dct_params = rows4(&DCT4X8_PARAMS);
            p.dct4x4_params = rows4(&DCT4X4_PARAMS);
            p.params = rows9(&AFV_PARAMS);
            p
        }
        11..=16 => {
            let mut p = DequantParams::empty(EncodingMode::Dct);
            let mut bases = LARGE_DCT_BASES[index - 11];
            if index == 16 && !DCT128X256_DEFAULT_BASES_AS_PRINTED {
                bases = LARGE_DCT_BASES_16_DOUBLED;
            }
            let seqs = [&SEQ_A, &SEQ_B, &SEQ_C];
            let mut rows = [Vec::new(), Vec::new(), Vec::new()];
            for c in 0..3 {
                let mut row = Vec::with_capacity(8);
                row.push(bases[c]);
                row.extend_from_slice(seqs[c]);
                rows[c] = row;
            }
            p.dct_params = rows;
            p
        }
        _ => {
            return Err(DecodeError::out_of_range(
                "parameters index",
                "I.2.5",
                as_u64(index),
            ));
        }
    };
    out.denominator = 1.0;
    out.raw = None;
    Ok(out)
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/// Widening for error payloads and guard charges.
fn as_u64(v: usize) -> u64 {
    u64::try_from(v).unwrap_or(u64::MAX)
}

/// I.2.4's `* 64` applied to `params` and to the first column of `dct_params`.
const PARAM_SCALE: f64 = 64.0;

/// Reads a `3 x n` matrix of `F16()` in raster order (rows are channels).
fn read_matrix_3xn(
    reader: &mut BitReader<'_>,
    n: usize,
    name: &'static str,
) -> Result<[Vec<f64>; 3]> {
    let mut rows = [Vec::new(), Vec::new(), Vec::new()];
    for row in &mut rows {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(f64::from(trace_field!(
                reader,
                name,
                read_f16_as_f32(reader)
            )?));
        }
        *row = out;
    }
    Ok(rows)
}

/// I.2.4's `ReadDctParams()`.
///
/// `num_params = u(4) + 1`, then a `3 x num_params` raster-order matrix, then
/// the first column of every row is multiplied by 64.
fn read_dct_params(reader: &mut BitReader<'_>) -> Result<[Vec<f64>; 3]> {
    let raw = trace_field!(reader, "dequant.num_params", reader.read_bits(4))?;
    let num_params = usize::try_from(raw).unwrap_or(0) + 1;
    let mut rows = read_matrix_3xn(reader, num_params, "dequant.dct_param")?;
    for row in &mut rows {
        if let Some(first) = row.first_mut() {
            *first *= PARAM_SCALE;
        }
    }
    Ok(rows)
}

fn scale_all(rows: &mut [Vec<f64>; 3]) {
    for row in rows {
        for v in row.iter_mut() {
            *v *= PARAM_SCALE;
        }
    }
}

/// **Flip point — is the RAW modular sub-bitstream byte-aligned?**
///
/// I.2.4's RAW arm reads `params.denominator` and then a modular
/// sub-bitstream, with no `ZeroPadToByte()` between them and none after. Annex
/// B always writes that primitive explicitly where it applies, and the two
/// other in-frame sub-bitstreams that have a field in front of them — G.2.2's
/// `extra_precision` and G.2.4's `nb_blocks` — are likewise unpadded and decode
/// real streams correctly.
///
/// * `true` (shipped): unaligned on both sides, so the next parameter set's
///   `u(3)` starts at the very next bit. Matches the E.4.1 ICC finding, where
///   the aligned reading failed on the first symbol.
/// * `false`: `ZeroPadToByte()` after the sub-bitstream.
///
/// RAW is the only place in the frame where a modular sub-bitstream is
/// *followed* by more fields in the same section, so it is the only construct
/// that can decide this — and no `cjxl` build emits RAW. Untested by any real
/// stream.
pub const RAW_SUBBITSTREAM_IS_UNALIGNED: bool = true;

/// Decodes the 3-channel modular image of I.2.4's RAW arm, inline.
///
/// The clause asks for "a 3-channel image of the same shape as the required
/// quant matrix", so every channel is `cols` wide and `rows` tall per
/// Table I.4, at full resolution. Channels are taken in Table I.1's index
/// order — see [`RAW_MATRIX_CHANNEL_ORDER_IS_XYB`].
fn read_raw_matrix(
    reader: &mut BitReader<'_>,
    index: usize,
    ctx: &RawMatrixContext<'_>,
    guard: &mut AllocGuard,
) -> Result<[Vec<f32>; 3]> {
    let (rows, cols) = matrix_size(index)?;
    let mut options = ctx.options;
    options.stream_index = stream_index::dequant_table(ctx.geometry, as_u64(index))?;

    let spec = ChannelSpec::new(
        u32::try_from(cols)
            .map_err(|_| DecodeError::out_of_range("RAW matrix columns", "I.2.4", as_u64(cols)))?,
        u32::try_from(rows)
            .map_err(|_| DecodeError::out_of_range("RAW matrix rows", "I.2.4", as_u64(rows)))?,
    );
    let specs = [spec; 3];

    let image = decode_sub_bitstream_with(reader, &specs, &options, ctx.tree_source, guard)?;
    if !RAW_SUBBITSTREAM_IS_UNALIGNED {
        reader.zero_pad_to_byte()?;
    }

    let channels = image.into_channels();
    let planes: [Channel; 3] = channels
        .try_into()
        .map_err(|_| DecodeError::out_of_range("RAW channel count", "I.2.4", 3))?;

    let mut out = [Vec::new(), Vec::new(), Vec::new()];
    for (slot, channel) in out.iter_mut().zip(planes.iter()) {
        let samples = channel.samples();
        if samples.len() != rows * cols {
            return Err(DecodeError::out_of_range(
                "RAW channel size",
                "I.2.4",
                as_u64(samples.len()),
            ));
        }
        // The modular layer produces integers; the dequantization matrix is
        // floating point. i32 -> f32 is exact for every value a sane matrix
        // holds and monotone everywhere, and the positivity check below is what
        // actually guards the pipeline.
        #[allow(clippy::cast_precision_loss)]
        let values: Vec<f32> = samples.iter().map(|&v| v as f32).collect();
        *slot = values;
    }
    if RAW_MATRIX_CHANNEL_ORDER_IS_XYB {
        Ok(out)
    } else {
        let [first, second, b] = out;
        Ok([second, first, b])
    }
}

/// **Flip point — the channel order of the RAW modular sub-bitstream.**
///
/// I.2.4 says only "a 3-channel image". Everything else in the clause indexes
/// parameters by the channel of the weights matrix — "row `c` of `dct_params`",
/// "the weights matrix for channel `c`" — which is Table I.1's X, Y, B
/// numbering, and I.5.3 applies the matrix for channel `c` under that same
/// numbering. There is no reordering language anywhere in I.2.4.
///
/// * `true` (shipped): X, Y, B, i.e. stream position equals channel index.
/// * `false`: Y, X, B.
///
/// The `false` reading is not idle: G.2.2's `LfQuant`, which also says only
/// "three channels", turned out to be Y, X, B (see
/// [`LF_QUANT_CHANNEL_ORDER_IS_XYB`](crate::vardct::lf::LF_QUANT_CHANNEL_ORDER_IS_XYB)).
/// The difference is that I.4 states the Y, X, B order for HF *coefficients*
/// explicitly, and a RAW dequantization table is a parameter table, not
/// coefficients. Untestable until a stream uses RAW.
pub const RAW_MATRIX_CHANNEL_ORDER_IS_XYB: bool = true;

/// Reads one parameter set of I.2.4.
fn read_params_set(
    reader: &mut BitReader<'_>,
    index: usize,
    raw_ctx: Option<&RawMatrixContext<'_>>,
    guard: &mut AllocGuard,
) -> Result<DequantParams> {
    let raw_mode = trace_field!(reader, "dequant.encoding_mode", reader.read_bits(3))?;
    let mode = EncodingMode::from_bits(raw_mode)?;
    if !mode.allows_index(index) {
        return Err(DecodeError::out_of_range(
            "encoding_mode",
            "I.2.4",
            u64::from(raw_mode),
        ));
    }

    // Every branch reads at most 3 * 16 F16 values plus a u(4); charge the
    // upper bound once so the metering happens before any allocation.
    guard.charge(3 * 16 * 8)?;

    match mode {
        EncodingMode::Library => default_params(index),
        EncodingMode::Hornuss => {
            let mut p = DequantParams::empty(mode);
            p.params = read_matrix_3xn(reader, 3, "dequant.hornuss_param")?;
            scale_all(&mut p.params);
            Ok(p)
        }
        EncodingMode::Dct2 => {
            let mut p = DequantParams::empty(mode);
            p.params = read_matrix_3xn(reader, 6, "dequant.dct2_param")?;
            scale_all(&mut p.params);
            Ok(p)
        }
        EncodingMode::Dct4 => {
            let mut p = DequantParams::empty(mode);
            p.params = read_matrix_3xn(reader, 2, "dequant.dct4_param")?;
            scale_all(&mut p.params);
            p.dct_params = read_dct_params(reader)?;
            Ok(p)
        }
        EncodingMode::Dct4x8 => {
            let mut p = DequantParams::empty(mode);
            // Note: no `* 64` here. The clause scales `params` for Hornuss,
            // DCT2, DCT4 and AFV, and pointedly does not for DCT4x8.
            p.params = read_matrix_3xn(reader, 1, "dequant.dct4x8_param")?;
            p.dct_params = read_dct_params(reader)?;
            Ok(p)
        }
        EncodingMode::Afv => {
            let mut p = DequantParams::empty(mode);
            p.params = read_matrix_3xn(reader, 9, "dequant.afv_param")?;
            // Only the first six of the nine columns are scaled; the last three
            // are `Mult()` arguments, which are dimensionless.
            for row in &mut p.params {
                for v in row.iter_mut().take(6) {
                    *v *= PARAM_SCALE;
                }
            }
            p.dct_params = read_dct_params(reader)?;
            p.dct4x4_params = read_dct_params(reader)?;
            Ok(p)
        }
        EncodingMode::Dct => {
            let mut p = DequantParams::empty(mode);
            p.dct_params = read_dct_params(reader)?;
            Ok(p)
        }
        EncodingMode::Raw => {
            let mut p = DequantParams::empty(mode);
            p.denominator = trace_field!(reader, "dequant.denominator", read_f16_as_f32(reader))?;
            // The sub-bitstream starts here. Refusing without decoding it would
            // leave the reader mid-stream, so this must be a hard stop, not a
            // "carry on and fail later".
            let Some(ctx) = raw_ctx else {
                return Err(DecodeError::Unsupported {
                    feature: "RAW dequantization matrices (no modular context was supplied, and \
                              skipping the inline sub-bitstream would desynchronise the rest of \
                              HfGlobal)",
                    clause: "18181-1 I.2.4",
                });
            };
            p.raw = Some(read_raw_matrix(reader, index, ctx, guard)?);
            Ok(p)
        }
    }
}

/// All 17 parameter sets of I.2.4.
#[derive(Debug, Clone, PartialEq)]
pub struct DequantMatrices {
    entries: Vec<DequantParams>,
}

/// A RAW-mode parameter set that has not been resolved.
///
/// **Always empty now.** RAW matrices are decoded inline (see
/// [`RawMatrixContext`]), so a `DequantMatrices` either carries every RAW
/// matrix it needs or the read already failed. The type and
/// [`DequantMatrices::raw_requests`] survive only so the pre-fix call site in
/// `decode.rs` keeps compiling; both should be deleted once that call site
/// passes a [`RawMatrixContext`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawMatrixRequest {
    /// Table I.4 parameters index.
    pub index: usize,
    /// Rows of the required matrix.
    pub rows: usize,
    /// Columns of the required matrix.
    pub cols: usize,
    /// `params.denominator`, the multiplier applied to the decoded planes.
    pub denominator: f32,
}

/// What the inline RAW modular sub-bitstream of I.2.4 needs from the frame.
///
/// I.2.4's RAW arm reads `params.denominator` as an `F16()` and then, **at
/// that same bit position**, a 3-channel modular sub-bitstream. It is not a
/// separate section: G.3's section list gives `HfGlobal` exactly one section.
/// A decoder that skips the sub-bitstream therefore does not merely lack a
/// matrix, it desynchronises every field after it — which is why this context
/// has to be threaded into the parameter read rather than resolved afterwards.
#[derive(Debug, Clone, Copy)]
pub struct RawMatrixContext<'a> {
    /// Modular bounds and `bits_per_sample`. `stream_index` is overwritten per
    /// table with H.4.1's `dequant_table` value, so the caller's value for that
    /// field is ignored.
    pub options: ModularOptions,
    /// The frame's MA tree source (G.1.3), exactly as the `LfQuant` and
    /// HF-metadata sub-bitstreams receive it. `HfGlobal` is its own section, so
    /// a global tree is passed with `restart: true`.
    pub tree_source: TreeSource<'a>,
    /// The frame geometry, for H.4.1's `1 + 3 * num_lf_groups + index`.
    pub geometry: &'a FrameGeometry,
}

impl DequantMatrices {
    /// The Table I.6 defaults for every index, i.e. the `all_default` case.
    ///
    /// # Errors
    ///
    /// [`DecodeError::FieldOutOfRange`] only if [`NUM_DEQUANT_MATRICES`] and
    /// the default table ever disagree, which the tests forbid.
    pub fn all_default() -> Result<Self> {
        let mut entries = Vec::with_capacity(NUM_DEQUANT_MATRICES);
        for index in 0..NUM_DEQUANT_MATRICES {
            entries.push(default_params(index)?);
        }
        Ok(Self { entries })
    }

    /// The parameter set for a Table I.4 index.
    ///
    /// # Errors
    ///
    /// [`DecodeError::FieldOutOfRange`] if `index` is out of range.
    pub fn params(&self, index: usize) -> Result<&DequantParams> {
        self.entries
            .get(index)
            .ok_or_else(|| DecodeError::out_of_range("parameters index", "I.2.4", as_u64(index)))
    }

    /// Every parameter set still waiting for its RAW modular sub-bitstream.
    ///
    /// Always empty; see [`RawMatrixRequest`]. Retained so the pre-fix
    /// `decode.rs` guard keeps compiling.
    #[must_use]
    pub fn raw_requests(&self) -> Vec<RawMatrixRequest> {
        let mut out = Vec::new();
        for (index, entry) in self.entries.iter().enumerate() {
            if entry.mode == EncodingMode::Raw
                && entry.raw.is_none()
                && let Ok((rows, cols)) = matrix_size(index)
            {
                out.push(RawMatrixRequest {
                    index,
                    rows,
                    cols,
                    denominator: entry.denominator,
                });
            }
        }
        out
    }

    /// The weights matrix for one index and channel (I.2.4).
    ///
    /// RAW sets have no weights matrix — their dequantization matrix is given
    /// directly — so this returns [`DecodeError::Unsupported`] for them.
    ///
    /// # Errors
    ///
    /// [`DecodeError::FieldOutOfRange`] for an out-of-range index or channel or
    /// a degenerate parameter set; [`DecodeError::Unsupported`] for RAW.
    pub fn weights(&self, index: usize, channel: usize) -> Result<WeightMatrix> {
        let (rows, cols) = matrix_size(index)?;
        let entry = self.params(index)?;
        build_weights(entry, channel, rows, cols)
    }

    /// The dequantization matrix for one index and channel (I.2.4).
    ///
    /// For every mode but RAW this is the element-wise reciprocal of
    /// [`DequantMatrices::weights`]; for RAW it is the decoded planes times
    /// `params.denominator`.
    ///
    /// # Errors
    ///
    /// [`DecodeError::Unsupported`] if a RAW set has not been supplied through
    /// [`DequantMatrices::set_raw_matrix`]; otherwise as
    /// [`DequantMatrices::weights`] and [`WeightMatrix::reciprocal`].
    pub fn matrix(&self, index: usize, channel: usize) -> Result<DequantMatrix> {
        let (rows, cols) = matrix_size(index)?;
        let entry = self.params(index)?;
        if entry.mode == EncodingMode::Raw {
            let planes = entry.raw.as_ref().ok_or(DecodeError::Unsupported {
                feature: "RAW dequantization matrices (their modular sub-bitstream is not wired up)",
                clause: "18181-1 I.2.4",
            })?;
            let plane = planes
                .get(channel)
                .ok_or_else(|| DecodeError::out_of_range("channel", "I.2.4", as_u64(channel)))?;
            let scaled: Vec<f32> = plane.iter().map(|v| v * entry.denominator).collect();
            return DequantMatrix::from_rows_cols(rows, cols, scaled);
        }
        build_weights(entry, channel, rows, cols)?.reciprocal()
    }

    /// The dequantization matrix a transform type uses (Table I.4).
    ///
    /// # Errors
    ///
    /// As [`DequantMatrices::matrix`].
    pub fn for_transform(&self, transform: TransformType, channel: usize) -> Result<DequantMatrix> {
        self.matrix(transform.dequant_matrix_index(), channel)
    }

    /// Builds all 17 x 3 matrices at once, metering the result.
    ///
    /// The full set is about 1.6 MB of `f32`; a decoder that touches only small
    /// transforms should call [`DequantMatrices::matrix`] on demand instead.
    ///
    /// # Errors
    ///
    /// [`DecodeError::Core`] if the allocation exceeds the limits, otherwise as
    /// [`DequantMatrices::matrix`].
    pub fn build_all(&self, guard: &mut AllocGuard) -> Result<Vec<[DequantMatrix; 3]>> {
        let mut total = 0u64;
        for index in 0..NUM_DEQUANT_MATRICES {
            let (rows, cols) = matrix_size(index)?;
            total = total.saturating_add(as_u64(rows * cols) * 3 * 4);
        }
        guard.charge(total)?;

        let mut out = Vec::with_capacity(NUM_DEQUANT_MATRICES);
        for index in 0..NUM_DEQUANT_MATRICES {
            out.push([
                self.matrix(index, 0)?,
                self.matrix(index, 1)?,
                self.matrix(index, 2)?,
            ]);
        }
        Ok(out)
    }
}

/// The per-mode weights rules of I.2.4.
fn build_weights(
    entry: &DequantParams,
    channel: usize,
    rows: usize,
    cols: usize,
) -> Result<WeightMatrix> {
    let params = DequantParams::row(&entry.params, channel)?;
    let dct = DequantParams::row(&entry.dct_params, channel)?;
    let dct4x4 = DequantParams::row(&entry.dct4x4_params, channel)?;

    match entry.mode {
        EncodingMode::Library => Err(DecodeError::out_of_range(
            "encoding_mode Library after SetDefaultMode",
            "I.2.4",
            0,
        )),
        EncodingMode::Raw => Err(DecodeError::Unsupported {
            feature: "weights matrix for a RAW dequantization matrix (RAW has none)",
            clause: "18181-1 I.2.4",
        }),
        EncodingMode::Dct => dct_quant_weights(cols, rows, dct),
        EncodingMode::Hornuss => {
            // Coefficient (1,1) is params(c,2); (0,1) and (1,0) are params(c,1);
            // every other coefficient is params(c,0); and (0,0) is 1.
            let p0 = DequantParams::param(params, 0)?;
            let p1 = DequantParams::param(params, 1)?;
            let p2 = DequantParams::param(params, 2)?;
            let mut w = WeightMatrix::zeros(rows, cols);
            for y in 0..rows {
                for x in 0..cols {
                    w.set(x, y, p0);
                }
            }
            w.set(0, 1, p1);
            w.set(1, 0, p1);
            w.set(1, 1, p2);
            w.set(0, 0, 1.0);
            Ok(w)
        }
        EncodingMode::Dct2 => {
            let mut w = WeightMatrix::zeros(rows, cols);
            let p = |i: usize| DequantParams::param(params, i);
            // The rectangles are given by their top-left and bottom-right
            // corners, and the bottom-right corner is exclusive: i == 5's
            // ((4,4),(8,8)) would otherwise leave the matrix. With that reading
            // the six rules plus (0,0) tile the 8x8 matrix exactly, which the
            // tests check.
            let mut fill = |x0: usize, y0: usize, x1: usize, y1: usize, v: f64| {
                for y in y0..y1 {
                    for x in x0..x1 {
                        w.set(x, y, v);
                    }
                }
            };
            fill(4, 4, 8, 8, p(5)?);
            fill(4, 0, 8, 4, p(4)?);
            fill(0, 4, 4, 8, p(4)?);
            fill(2, 2, 4, 4, p(3)?);
            fill(2, 0, 4, 2, p(2)?);
            fill(0, 2, 2, 4, p(2)?);
            fill(1, 1, 2, 2, p(1)?);
            fill(0, 1, 1, 2, p(0)?);
            fill(1, 0, 2, 1, p(0)?);
            if DCT2_DC_WEIGHT_IS_ONE {
                w.set(0, 0, 1.0);
            }
            Ok(w)
        }
        EncodingMode::Dct4 => {
            let base = dct_quant_weights(4, 4, dct)?;
            let mut w = WeightMatrix::zeros(rows, cols);
            for y in 0..rows {
                for x in 0..cols {
                    w.set(x, y, base.at(x / 2, y / 2));
                }
            }
            let p0 = DequantParams::param(params, 0)?;
            let p1 = DequantParams::param(params, 1)?;
            if p0 == 0.0 || p1 == 0.0 {
                return Err(DecodeError::out_of_range("DCT4 params", "I.2.4", 0));
            }
            w.set(0, 1, w.at(0, 1) / p0);
            w.set(1, 0, w.at(1, 0) / p0);
            w.set(1, 1, w.at(1, 1) / p1);
            Ok(w)
        }
        EncodingMode::Dct4x8 => {
            // "the 4 x 8 matrix" is 4 rows by 8 columns: the copy indexes
            // (x, y Idiv 2) for x < 8 and y < 8, so it needs 8 columns and 4
            // rows.
            let base = dct_quant_weights(8, 4, dct)?;
            let mut w = WeightMatrix::zeros(rows, cols);
            for y in 0..rows {
                for x in 0..cols {
                    w.set(x, y, base.at(x, y / 2));
                }
            }
            let p0 = DequantParams::param(params, 0)?;
            if p0 == 0.0 {
                return Err(DecodeError::out_of_range("DCT4x8 params", "I.2.4", 0));
            }
            w.set(0, 1, w.at(0, 1) / p0);
            Ok(w)
        }
        EncodingMode::Afv => {
            let weights4x8 = dct_quant_weights(8, 4, dct)?;
            let weights4x4 = dct_quant_weights(4, 4, dct4x4)?;

            let mut bands = [0.0f64; 4];
            bands[0] = DequantParams::param(params, 5)?;
            if !bands[0].is_finite() || bands[0] < 0.0 {
                return Err(DecodeError::out_of_range("AFV bands[0]", "I.2.4", 0));
            }
            for i in 1..4 {
                bands[i] = bands[i - 1] * mult(DequantParams::param(params, i + 5)?);
                if !bands[i].is_finite() || bands[i] < 0.0 {
                    return Err(DecodeError::out_of_range("AFV bands", "I.2.4", as_u64(i)));
                }
            }

            let mut w = WeightMatrix::zeros(rows, cols);
            w.set(0, 0, 1.0);
            w.set(0, 1, DequantParams::param(params, 0)?);
            w.set(1, 0, DequantParams::param(params, 1)?);
            w.set(0, 2, DequantParams::param(params, 2)?);
            w.set(2, 0, DequantParams::param(params, 3)?);
            w.set(2, 2, DequantParams::param(params, 4)?);

            let range = AFV_HI - AFV_LO + 1e-6;
            for y in 0..4 {
                for x in 0..4 {
                    if x < 2 && y < 2 {
                        continue;
                    }
                    let freq = AFV_FREQS[y * 4 + x];
                    let val = interpolate(freq - AFV_LO, range, &bands)?;
                    w.set(2 * y, 2 * x, val);
                }
            }
            for y in 0..4 {
                for x in 0..8 {
                    if x == 0 && y == 0 {
                        continue;
                    }
                    w.set(x, 2 * y + 1, weights4x8.at(x, y));
                }
            }
            for y in 0..4 {
                for x in 0..4 {
                    if x == 0 && y == 0 {
                        continue;
                    }
                    w.set(2 * x + 1, 2 * y, weights4x4.at(x, y));
                }
            }
            Ok(w)
        }
    }
}

/// Reads the dequantization matrix parameters of I.2.4, refusing RAW.
///
/// Equivalent to [`read_dequant_matrices_with`] with no
/// [`RawMatrixContext`]: a stream that uses `encoding_mode` 7 is rejected with
/// [`DecodeError::Unsupported`] at exactly the bit where its sub-bitstream
/// would start. Prefer the `_with` form; this one exists so callers that have
/// no frame geometry to hand still compile.
///
/// # Errors
///
/// As [`read_dequant_matrices_with`].
pub fn read_dequant_matrices(
    reader: &mut BitReader<'_>,
    guard: &mut AllocGuard,
) -> Result<DequantMatrices> {
    read_dequant_matrices_with(reader, None, guard)
}

/// Reads the dequantization matrix parameters of I.2.4.
///
/// A leading `Bool()` selects the all-default case; otherwise the 17 sets are
/// read in ascending index order. `raw_ctx` supplies what
/// `encoding_mode == RAW` needs to decode its inline modular sub-bitstream;
/// pass `None` only if RAW cannot occur.
///
/// # Errors
///
/// * [`DecodeError::Bitstream`] on truncation.
/// * [`DecodeError::FieldOutOfRange`] if an `encoding_mode` is not valid for
///   its index per Table I.5.
/// * [`DecodeError::Modular`] if a RAW sub-bitstream is malformed.
/// * [`DecodeError::Unsupported`] if a set is RAW and `raw_ctx` is `None`.
pub fn read_dequant_matrices_with(
    reader: &mut BitReader<'_>,
    raw_ctx: Option<&RawMatrixContext<'_>>,
    guard: &mut AllocGuard,
) -> Result<DequantMatrices> {
    let all_default = trace_field!(reader, "dequant.all_default", read_bool(reader))?;
    if all_default {
        return DequantMatrices::all_default();
    }
    let mut entries = Vec::with_capacity(NUM_DEQUANT_MATRICES);
    for index in 0..NUM_DEQUANT_MATRICES {
        entries.push(read_params_set(reader, index, raw_ctx, guard)?);
    }
    Ok(DequantMatrices { entries })
}

// ---------------------------------------------------------------------------
// Table G.4 — the first two rows of HfGlobal
// ---------------------------------------------------------------------------

/// The non-`HfPass` part of an `HfGlobal` bundle (Table G.4).
#[derive(Debug, Clone, PartialEq)]
pub struct HfGlobalParams {
    /// I.2.4 dequantization matrix parameters.
    pub matrices: DequantMatrices,
    /// I.2.6 `num_hf_presets`.
    pub num_hf_presets: u32,
}

/// Reads the dequantization matrices and `num_hf_presets` of Table G.4,
/// refusing RAW matrices.
///
/// Equivalent to [`read_hf_global_params_with`] with no
/// [`RawMatrixContext`]. Prefer the `_with` form; this signature is kept so
/// the existing `decode.rs` call site compiles unchanged.
///
/// # Errors
///
/// As [`read_hf_global_params_with`].
pub fn read_hf_global_params(
    reader: &mut BitReader<'_>,
    num_groups: u64,
    guard: &mut AllocGuard,
) -> Result<HfGlobalParams> {
    read_hf_global_params_with(reader, num_groups, None, guard)
}

/// Reads the dequantization matrices and `num_hf_presets` of Table G.4.
///
/// The caller continues with `hf_pass[num_passes]` (I.3) at the returned bit
/// position.
///
/// # Errors
///
/// As [`read_dequant_matrices_with`] and
/// [`read_num_hf_presets`](super::block_ctx::read_num_hf_presets).
pub fn read_hf_global_params_with(
    reader: &mut BitReader<'_>,
    num_groups: u64,
    raw_ctx: Option<&RawMatrixContext<'_>>,
    guard: &mut AllocGuard,
) -> Result<HfGlobalParams> {
    let matrices = read_dequant_matrices_with(reader, raw_ctx, guard)?;
    let num_hf_presets = read_num_hf_presets(reader, num_groups)?;
    Ok(HfGlobalParams {
        matrices,
        num_hf_presets,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testsupport::BitWriter;
    use jpxl_core::limits::Limits;

    fn guard() -> AllocGuard {
        AllocGuard::new(&LIMITS)
    }

    static LIMITS: Limits = Limits::relaxed();

    #[test]
    fn table_i4_sizes_match_the_transform_coefficient_shapes() {
        // The load-bearing cross-check between 8A's Table I.1 vocabulary and
        // Table I.4: a dequantization matrix is applied coefficient by
        // coefficient, so its shape must be the transform's coefficient shape,
        // which is always landscape. A transposed row here would be invisible
        // for the square transforms and silently wrong for the oblong ones.
        for t in TransformType::ALL {
            let (rows, cols) = matrix_size(t.dequant_matrix_index()).expect("valid index");
            assert_eq!(
                (rows, cols),
                (t.coeff_rows(), t.coeff_cols()),
                "{t:?} index {}",
                t.dequant_matrix_index()
            );
            assert!(rows <= cols, "{t:?}: Table I.4 sizes are landscape");
        }
    }

    #[test]
    fn every_index_is_claimed_by_at_least_one_transform() {
        // Proves Table I.4 has no unreachable row, which is what makes reading
        // exactly 17 parameter sets correct.
        let mut seen = [false; NUM_DEQUANT_MATRICES];
        for t in TransformType::ALL {
            seen[t.dequant_matrix_index()] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn every_default_matrix_is_positive_finite_and_correctly_shaped() {
        // The wall-of-constants check. A mistyped digit in Table I.6 almost
        // always produces a non-positive band (Mult() of a positive value is
        // > 1, so a sign slip inverts the band sequence) or a NaN, and a
        // dropped digit changes the shape assertion in the sibling test.
        let matrices = DequantMatrices::all_default().expect("defaults build");
        for index in 0..NUM_DEQUANT_MATRICES {
            let (rows, cols) = matrix_size(index).expect("valid index");
            for channel in 0..3 {
                let m = matrices.matrix(index, channel).expect("matrix builds");
                assert_eq!((m.rows(), m.cols()), (rows, cols), "index {index}");
                assert_eq!(m.as_slice().len(), rows * cols);
                for (i, &v) in m.as_slice().iter().enumerate() {
                    assert!(
                        v.is_finite() && v > 0.0,
                        "index {index} channel {channel} entry {i} = {v}"
                    );
                }
            }
        }
    }

    #[test]
    fn default_weights_decrease_with_frequency() {
        // Every default DCT row has a positive base followed by negative
        // multipliers, so Mult() shrinks each successive band: the weights fall
        // as the radial distance grows and the dequantization multipliers rise.
        // A row read in the wrong order (the part1.md DCT16x16 hazard) breaks
        // this monotonicity.
        let matrices = DequantMatrices::all_default().expect("defaults build");
        for index in [0usize, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16] {
            for channel in 0..3 {
                let w = matrices.weights(index, channel).expect("weights build");
                let corner = w.at(w.cols() - 1, w.rows() - 1);
                let dc = w.at(0, 0);
                assert!(
                    corner < dc,
                    "index {index} channel {channel}: corner {corner} >= dc {dc}"
                );
            }
        }
    }

    #[test]
    fn interpolate_matches_the_hand_computation() {
        // Pins pow(B / A, frac) against the LaTeX's pow(B / 4B, frac).
        // bands = {2, 8}: at pos = max/2 the geometric midpoint is 2 * 2 = 4.
        // The B/4B reading would give 2 * (8/32)^0.5 = 1.0, and the arithmetic
        // mean reading would give 5.
        let bands = [2.0, 8.0];
        let v = interpolate(0.5, 1.0, &bands).expect("in range");
        assert!((v - 4.0).abs() < 1e-12, "{v}");
        // Endpoints.
        assert!((interpolate(0.0, 1.0, &bands).expect("v") - 2.0).abs() < 1e-12);
        // Three bands: pos = max/2 lands exactly on the middle band.
        let bands = [1.0, 3.0, 9.0];
        let v = interpolate(0.5, 1.0, &bands).expect("in range");
        assert!((v - 3.0).abs() < 1e-12, "{v}");
        // A single band ignores pos entirely.
        assert_eq!(interpolate(123.0, 1.0, &[7.0]).expect("v"), 7.0);
    }

    #[test]
    fn interpolate_rejects_a_position_past_its_range() {
        // The `+ 1e-6` in the callers is what keeps the last band pair in
        // range; without it the corner sample would index one past the end.
        assert!(interpolate(1.0, 1.0, &[2.0, 8.0]).is_err());
        assert!(interpolate(-1.0, 1.0, &[2.0, 8.0]).is_err());
        assert!(interpolate(0.0, 1.0, &[]).is_err());
    }

    #[test]
    fn mult_matches_the_clause_and_is_always_positive() {
        assert_eq!(mult(1.0), 2.0);
        assert_eq!(mult(0.0), 1.0); // 0 is not > 0, so the else branch.
        assert_eq!(mult(-1.0), 0.5);
        assert!((mult(-0.25) - 0.8).abs() < 1e-15);
        for v in [-1e30, -1.0, -0.5, 0.0, 0.5, 1.0, 1e30] {
            assert!(mult(v) > 0.0, "Mult({v})");
        }
    }

    #[test]
    fn bands_are_the_running_product_of_mult() {
        // Hand computation with the B-channel AFV defaults: bands[0] = 22 and
        // Mult(-0.25) = 0.8, so the sequence is 22, 17.6, 14.08, 11.264.
        let bands = bands_from(&[22.0, -0.25, -0.25, -0.25]).expect("valid");
        assert_eq!(bands.len(), 4);
        assert!((bands[1] - 17.6).abs() < 1e-12);
        assert!((bands[2] - 14.08).abs() < 1e-12);
        assert!((bands[3] - 11.264).abs() < 1e-12);
    }

    #[test]
    fn dct_weights_start_at_the_first_band() {
        // (0,0) has distance 0, so Interpolate returns bands[0] exactly. That
        // pins the base parameter of every Table I.6 DCT row: the DC weight is
        // the printed base, unscaled and uninterpolated.
        let w = dct_quant_weights(8, 8, &[3150.0, 0.0, -0.4, -0.4, -0.4, -2.0]).expect("valid");
        assert_eq!(w.at(0, 0), 3150.0);
        assert_eq!((w.rows(), w.cols()), (8, 8));
    }

    #[test]
    fn dct8x8_default_dc_multiplier_is_one_over_the_base() {
        let matrices = DequantMatrices::all_default().expect("defaults build");
        let m = matrices.matrix(0, 0).expect("matrix");
        assert!((m.at(0, 0) - 1.0 / 3150.0).abs() < 1e-9, "{}", m.at(0, 0));
        let m = matrices.matrix(0, 1).expect("matrix");
        assert!((m.at(0, 0) - 1.0 / 560.0).abs() < 1e-9);
        let m = matrices.matrix(0, 2).expect("matrix");
        assert!((m.at(0, 0) - 1.0 / 512.0).abs() < 1e-9);
    }

    #[test]
    fn hornuss_weights_follow_the_clause() {
        // Coefficient (0,0) is 1, (0,1) and (1,0) are params(c,1), (1,1) is
        // params(c,2), everything else params(c,0).
        let matrices = DequantMatrices::all_default().expect("defaults build");
        let w = matrices.weights(1, 0).expect("weights");
        assert_eq!(w.at(0, 0), 1.0);
        assert_eq!(w.at(0, 1), 3160.0);
        assert_eq!(w.at(1, 0), 3160.0);
        assert_eq!(w.at(1, 1), 3160.0);
        assert_eq!(w.at(7, 7), 280.0);
        assert_eq!(w.at(2, 0), 280.0);
    }

    #[test]
    fn dct2_rectangles_tile_the_matrix_exactly() {
        // The rectangles are printed as corner pairs with no inclusivity
        // stated. Only the exclusive reading covers all 64 positions without
        // leaving the matrix, and the default X-channel row has six distinct
        // values, so a miscovered cell shows up as the wrong constant.
        let matrices = DequantMatrices::all_default().expect("defaults build");
        let w = matrices.weights(2, 0).expect("weights");
        assert_eq!(w.at(0, 0), 1.0);
        assert_eq!(w.at(0, 1), 3840.0);
        assert_eq!(w.at(1, 0), 3840.0);
        assert_eq!(w.at(1, 1), 2560.0);
        // i == 2: ((2,0),(4,2)) and its symmetric image.
        assert_eq!(w.at(2, 0), 1280.0);
        assert_eq!(w.at(3, 1), 1280.0);
        assert_eq!(w.at(0, 2), 1280.0);
        assert_eq!(w.at(1, 3), 1280.0);
        // i == 3: ((2,2),(4,4)).
        assert_eq!(w.at(2, 2), 640.0);
        assert_eq!(w.at(3, 3), 640.0);
        // i == 4: ((4,0),(8,4)) and symmetric.
        assert_eq!(w.at(7, 0), 480.0);
        assert_eq!(w.at(0, 7), 480.0);
        assert_eq!(w.at(4, 3), 480.0);
        // i == 5: ((4,4),(8,8)).
        assert_eq!(w.at(4, 4), 300.0);
        assert_eq!(w.at(7, 7), 300.0);

        // Every cell got one of the seven values, none left at zero.
        for y in 0..8 {
            for x in 0..8 {
                assert!(w.at(x, y) > 0.0, "({x},{y}) uncovered");
            }
        }
    }

    #[test]
    fn dct4_replicates_the_4x4_and_divides_the_three_corners() {
        // weights(x,y) = base(x/2, y/2), then (0,1) and (1,0) are divided by
        // params(c,0) and (1,1) by params(c,1). The defaults are all 1.0, so
        // the division is the identity and the replication is visible alone.
        let matrices = DequantMatrices::all_default().expect("defaults build");
        let w = matrices.weights(3, 0).expect("weights");
        let base = dct_quant_weights(4, 4, &DCT4X4_PARAMS[0]).expect("base");
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(w.at(x, y), base.at(x / 2, y / 2), "({x},{y})");
            }
        }
    }

    #[test]
    fn dct4x8_replicates_rows_not_columns() {
        // weights(x, y) = base(x, y Idiv 2) with base 4 rows by 8 columns. The
        // transposed reading would need base(x Idiv 2, y) and an 8x4 base, and
        // would make rows 0 and 1 differ.
        let matrices = DequantMatrices::all_default().expect("defaults build");
        let w = matrices.weights(9, 1).expect("weights");
        for x in 1..8 {
            assert_eq!(w.at(x, 0), w.at(x, 1), "column {x} of rows 0 and 1");
        }
        // (0,1) is the one position the clause then divides, and the default
        // params are 1.0, so it stays equal too.
        assert_eq!(w.at(0, 0), w.at(0, 1));
        // Rows 2 and 3 come from base row 1, which differs from base row 0.
        assert_ne!(w.at(7, 0), w.at(7, 2));
    }

    #[test]
    fn afv_weights_cover_the_matrix_and_use_both_sub_matrices() {
        // The three AFV loops partition the 8x8 matrix: even/even from the
        // frequency table, odd rows from weights4x8, odd columns of even rows
        // from weights4x4. Any missed cell stays zero and fails the positivity
        // check in reciprocal().
        let matrices = DequantMatrices::all_default().expect("defaults build");
        let w = matrices.weights(10, 2).expect("weights");
        for y in 0..8 {
            for x in 0..8 {
                assert!(w.at(x, y) > 0.0, "({x},{y}) uncovered");
            }
        }
        // The six explicit assignments survive: (0,0) is 1 and (0,1)/(1,0) are
        // params(c,0)/params(c,1), because the two later loops skip (0,0) in
        // their own coordinates.
        assert_eq!(w.at(0, 0), 1.0);
        // The Table I.6 defaults are already post-`* 64` values, so they are
        // used as printed: 384, not 384 * 64.
        assert_eq!(w.at(0, 1), 384.0);
        assert_eq!(w.at(1, 0), 384.0);
        assert_eq!(w.at(0, 1), AFV_PARAMS[2][0]);
        assert_eq!(w.at(1, 0), AFV_PARAMS[2][1]);
    }

    #[test]
    fn afv_bands_are_constant_when_the_multipliers_are_zero() {
        // Channels X and Y have params(c, 6..9) = 0, and Mult(0) is 1, so all
        // four bands equal params(c,5) and every interpolated frequency weight
        // is that same value. A hand-checkable consequence of the clause.
        let matrices = DequantMatrices::all_default().expect("defaults build");
        let w = matrices.weights(10, 0).expect("weights");
        let expected = AFV_PARAMS[0][5];
        // (2,0) and (0,4) are two of the even/even positions filled by the
        // frequency loop.
        assert!((w.at(0, 4) - expected).abs() < 1e-9, "{}", w.at(0, 4));
        assert!((w.at(4, 0) - expected).abs() < 1e-9, "{}", w.at(4, 0));
        assert!((w.at(6, 6) - expected).abs() < 1e-9, "{}", w.at(6, 6));
    }

    #[test]
    fn large_dct_bases_double_per_size_step() {
        // The internal consistency of Table I.6 that flagged the index-16
        // anomaly. 11 -> 13 -> 15 and 12 -> 14 double exactly in every channel.
        let doubles = |big: usize, small: usize| {
            for (c, (&b, &s)) in LARGE_DCT_BASES[big]
                .iter()
                .zip(LARGE_DCT_BASES[small].iter())
                .enumerate()
            {
                assert!(
                    (b - 2.0 * s).abs() < 1e-6 * b,
                    "channel {c}: index {big} is not twice index {small}"
                );
            }
        };
        doubles(2, 0);
        doubles(4, 2);
        doubles(3, 1);
        // And index 16 does not, in exactly two of the three channels. This
        // test is the sentinel for DCT128X256_DEFAULT_BASES_AS_PRINTED: if a
        // future edit "fixes" the printed values without flipping the constant,
        // it fails here.
        assert!(
            (LARGE_DCT_BASES[5][0] - 2.0 * LARGE_DCT_BASES[3][0]).abs()
                < 1e-6 * LARGE_DCT_BASES[5][0],
            "the X channel of index 16 does double"
        );
        assert!(LARGE_DCT_BASES[5][1] > 2.0 * LARGE_DCT_BASES[3][1]);
        assert!(LARGE_DCT_BASES[5][2] > 2.0 * LARGE_DCT_BASES[3][2]);
        for c in 1..3 {
            assert!(
                (LARGE_DCT_BASES_16_DOUBLED[c] - 2.0 * LARGE_DCT_BASES[3][c]).abs() < 1e-9,
                "channel {c}: the alternative reading does double"
            );
        }
    }

    #[test]
    fn seq_lengths_make_eight_parameter_rows() {
        // Each large-DCT row is a base plus a seven-element sequence; a dropped
        // element would silently change the band count and hence every weight.
        assert_eq!(SEQ_A.len(), 7);
        assert_eq!(SEQ_B.len(), 7);
        assert_eq!(SEQ_C.len(), 7);
        let p = default_params(11).expect("index 11");
        for c in 0..3 {
            assert_eq!(p.dct_params[c].len(), 8, "channel {c}");
        }
    }

    #[test]
    fn table_i5_valid_indices() {
        for index in 0..NUM_DEQUANT_MATRICES {
            assert!(EncodingMode::Library.allows_index(index));
            assert!(EncodingMode::Dct.allows_index(index));
            assert!(EncodingMode::Raw.allows_index(index));
        }
        for mode in [
            EncodingMode::Hornuss,
            EncodingMode::Dct2,
            EncodingMode::Dct4,
            EncodingMode::Dct4x8,
            EncodingMode::Afv,
        ] {
            for index in 0..NUM_DEQUANT_MATRICES {
                let expected = matches!(index, 0 | 1 | 2 | 3 | 9 | 10);
                assert_eq!(mode.allows_index(index), expected, "{mode:?} {index}");
            }
        }
        // Every 8x8 row of Table I.4 is exactly the set of valid indices for
        // the small-block modes: the modes build 8x8 matrices and nothing else.
        for index in 0..NUM_DEQUANT_MATRICES {
            let is_8x8 = matrix_size(index).expect("size") == (8, 8);
            assert_eq!(is_8x8, EncodingMode::Hornuss.allows_index(index));
        }
    }

    #[test]
    fn all_default_bundle_is_one_bit() {
        let mut w = BitWriter::new();
        w.bool(true);
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let m = read_dequant_matrices(&mut r, &mut guard()).expect("valid");
        assert_eq!(r.total_bits_read(), 1);
        assert_eq!(m, DequantMatrices::all_default().expect("defaults"));
    }

    #[test]
    fn seventeen_library_modes_are_seventeen_three_bit_fields() {
        // 1 + 17 * 3 = 52 bits. Proves the set count and that encoding_mode is
        // u(3), and that Library resolves to the same thing as all_default.
        let mut w = BitWriter::new();
        w.bool(false);
        for _ in 0..NUM_DEQUANT_MATRICES {
            w.u(3, 0);
        }
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let m = read_dequant_matrices(&mut r, &mut guard()).expect("valid");
        assert_eq!(r.total_bits_read(), 1 + 17 * 3);
        assert_eq!(m, DequantMatrices::all_default().expect("defaults"));
    }

    #[test]
    fn dct_mode_reads_num_params_then_three_rows() {
        // Index 0 with mode DCT and num_params = 2: 3 + 4 + 3 * 2 * 16 bits.
        // Proves ReadDctParams's field order and the `* 64` on column 0.
        let mut w = BitWriter::new();
        w.bool(false);
        w.u(3, 6); // DCT
        w.u(4, 1); // num_params = 2
        for _ in 0..6 {
            w.f16_bits(0x3C00); // 1.0
        }
        for _ in 1..NUM_DEQUANT_MATRICES {
            w.u(3, 0); // Library for the rest
        }
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let m = read_dequant_matrices(&mut r, &mut guard()).expect("valid");
        assert_eq!(r.total_bits_read(), 1 + 3 + 4 + 6 * 16 + 16 * 3);
        let p = m.params(0).expect("index 0");
        assert_eq!(p.mode(), EncodingMode::Dct);
        assert_eq!(p.dct_params[0], vec![64.0, 1.0]);
        assert_eq!(p.dct_params[2], vec![64.0, 1.0]);
    }

    #[test]
    fn hornuss_mode_scales_every_parameter() {
        let mut w = BitWriter::new();
        w.bool(false);
        w.u(3, 0); // index 0: Library
        w.u(3, 1); // index 1: Hornuss
        for _ in 0..9 {
            w.f16_bits(0x3C00); // 1.0
        }
        for _ in 2..NUM_DEQUANT_MATRICES {
            w.u(3, 0);
        }
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let m = read_dequant_matrices(&mut r, &mut guard()).expect("valid");
        assert_eq!(r.total_bits_read(), 1 + 17 * 3 + 9 * 16);
        let p = m.params(1).expect("index 1");
        assert_eq!(p.params[0], vec![64.0, 64.0, 64.0]);
    }

    #[test]
    fn dct4x8_mode_does_not_scale_its_params() {
        // The one mode whose `params` the clause does not multiply by 64.
        let mut w = BitWriter::new();
        w.bool(false);
        for _ in 0..9 {
            w.u(3, 0); // indices 0..=8: Library
        }
        w.u(3, 4); // index 9: DCT4x8
        w.f16_bits(0x3C00).f16_bits(0x3C00).f16_bits(0x3C00); // params 3x1
        w.u(4, 0); // num_params = 1
        for _ in 0..3 {
            w.f16_bits(0x3C00);
        }
        for _ in 10..NUM_DEQUANT_MATRICES {
            w.u(3, 0);
        }
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let m = read_dequant_matrices(&mut r, &mut guard()).expect("valid");
        let p = m.params(9).expect("index 9");
        assert_eq!(p.mode(), EncodingMode::Dct4x8);
        assert_eq!(p.params[0], vec![1.0]);
        // dct_params column 0 *is* scaled.
        assert_eq!(p.dct_params[0], vec![64.0]);
    }

    #[test]
    fn afv_mode_scales_only_the_first_six_params() {
        let mut w = BitWriter::new();
        w.bool(false);
        for _ in 0..10 {
            w.u(3, 0);
        }
        w.u(3, 5); // index 10: AFV
        for _ in 0..27 {
            w.f16_bits(0x3C00); // params 3x9, all 1.0
        }
        w.u(4, 0);
        for _ in 0..3 {
            w.f16_bits(0x3C00); // dct_params
        }
        w.u(4, 0);
        for _ in 0..3 {
            w.f16_bits(0x3C00); // dct4x4_params
        }
        for _ in 11..NUM_DEQUANT_MATRICES {
            w.u(3, 0);
        }
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let m = read_dequant_matrices(&mut r, &mut guard()).expect("valid");
        let p = m.params(10).expect("index 10");
        assert_eq!(
            p.params[0],
            vec![64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 1.0, 1.0, 1.0]
        );
    }

    #[test]
    fn a_mode_invalid_for_its_index_is_rejected() {
        // Table I.5: Hornuss is not valid for index 4 (DCT16x16).
        let mut w = BitWriter::new();
        w.bool(false);
        for _ in 0..4 {
            w.u(3, 0);
        }
        w.u(3, 1); // Hornuss at index 4
        for _ in 0..9 {
            w.f16_bits(0x3C00);
        }
        let data = w.finish_padded(4);
        let mut r = BitReader::new(&data);
        let err = read_dequant_matrices(&mut r, &mut guard()).expect_err("invalid mode");
        assert!(err.to_string().contains("encoding_mode"), "{err}");
    }

    #[test]
    fn truncated_bundle_errors() {
        let mut w = BitWriter::new();
        w.bool(false).u(3, 6).u(4, 15);
        let data = w.finish();
        let mut r = BitReader::new(&data);
        assert!(read_dequant_matrices(&mut r, &mut guard()).is_err());
    }

    #[test]
    fn degenerate_parameters_are_rejected_not_returned_as_nan() {
        // A DCT row whose base is zero makes bands[0] zero, and the reciprocal
        // would be infinite. The clause says no such value occurs, so this must
        // be an error rather than an infinity propagating into the samples.
        let mut w = BitWriter::new();
        w.bool(false);
        w.u(3, 6); // DCT at index 0
        w.u(4, 0); // num_params = 1
        for _ in 0..3 {
            w.f16_bits(0x0000); // 0.0
        }
        for _ in 1..NUM_DEQUANT_MATRICES {
            w.u(3, 0);
        }
        let data = w.finish_padded(2);
        let mut r = BitReader::new(&data);
        let m = read_dequant_matrices(&mut r, &mut guard()).expect("parses");
        assert!(m.matrix(0, 0).is_err(), "a zero base must be rejected");
    }

    // ------------------------------------------------------------------
    // RAW mode (encoding_mode 7)
    // ------------------------------------------------------------------
    //
    // The modular-sub-bitstream writer below mirrors the one `vardct::lf`
    // uses for G.2.2, field for field: `ModularHeader` with no transforms
    // (Table H.1), a single-leaf MA tree (H.4.2) whose predictor is Zero, and
    // a four-symbol fixed-width prefix bundle (C.2). It is an independent
    // encoder written from the Annex H tables, which is what makes decoding
    // its output evidence rather than a tautology; the same helper shape is
    // cross-validated against real `cjxl` streams by `vardct::lf`'s and
    // `vardct::hf_meta`'s own fixtures.

    /// Writes `value` most-significant bit first, as a prefix code does.
    fn msb(w: &mut BitWriter, value: u32, n: u32) {
        for i in (0..n).rev() {
            w.bit((value >> i) & 1 == 1);
        }
    }

    /// The smallest prefix-coded distribution bundle (C.2) that carries a
    /// four-symbol fixed-width alphabet, one cluster for every context.
    fn write_prefix_bundle(w: &mut BitWriter, num_dist: usize) {
        w.bool(false); // lz77.enabled
        if num_dist > 1 {
            w.bool(true); // simple clustering
            w.u(2, 0); // nbits = 0 -> every context maps to cluster 0
        }
        w.bool(true); // use_prefix_code
        w.u(4, 15); // HybridUintConfig split_exponent = log_alphabet_size
        w.bit(true); // alphabet_size = 1 + (1 << n) + u(n)
        w.u(4, 1);
        w.u(1, 1);
        w.u(2, 1); // RFC 7932 3.4 simple code, selector 1
        w.u(2, 3); // nsym - 1 = 3
        for symbol in 0..4u32 {
            w.u(2, symbol);
        }
        w.bit(false); // balanced-pattern bit for nsym == 4
    }

    /// `ModularHeader` (Table H.1) with no transforms, then H.4.2's
    /// single-leaf tree with the Zero predictor.
    fn write_modular_prelude(w: &mut BitWriter) {
        w.bool(false); // use_global_tree
        w.bool(true); // wp_params: default_wp
        w.u(2, 0); // nb_transforms: U32 selector 0 -> constant 0
        write_prefix_bundle(w, 6);
        for token in [0u32, 0, 0, 0, 0] {
            msb(w, token, 2);
        }
        write_prefix_bundle(w, 1);
    }

    /// A whole RAW sub-bitstream for a `rows x cols` matrix whose every
    /// sample is 1.
    ///
    /// The four-symbol test alphabet carries `UnpackSigned` of 0..=3, i.e.
    /// `{0, -1, 1, -2}`, and I.2.4 forbids a non-positive dequantization
    /// value, so 1 (token 2) is the only sample a minimal alphabet can use.
    fn write_raw_ones(w: &mut BitWriter, rows: usize, cols: usize) {
        write_modular_prelude(w);
        for _ in 0..(3 * rows * cols) {
            msb(w, 2, 2);
        }
    }

    fn eight_by_eight_geometry() -> crate::frame::FrameGeometry {
        let limits = Limits::relaxed();
        let mut g = AllocGuard::new(&limits);
        crate::frame::FrameGeometry::derive(
            crate::frame::geometry::GroupLayout {
                width: 64,
                height: 64,
                upsampling: 1,
                lf_level: 0,
                group_dim: jpxl_core::geometry::GroupDim::D256,
                num_passes: 1,
            },
            &limits,
            &mut g,
        )
        .expect("64x64 is well within limits")
    }

    #[test]
    fn raw_matrices_decode_inline_and_keep_the_bundle_in_sync() {
        // What this proves, and it is the only proof available because no
        // encoder emits RAW:
        //
        // 1. The 3-channel modular sub-bitstream is consumed *at the bit
        //    position right after `params.denominator`* — I.2.4's RAW arm is
        //    sequential code, and G.3's section list gives HfGlobal one
        //    section, so there is nowhere else it could be.
        // 2. It is consumed exactly, twice over, at two different parameter
        //    indices: index 2's Hornuss parameters and the trailing
        //    `num_hf_presets` are read at the right bits afterwards. A
        //    sub-bitstream over- or under-read by a single bit corrupts
        //    index 2's mode or its nine F16 values or the preset count.
        // 3. The dequantization matrix is the decoded planes times
        //    `params.denominator`, per channel and per index — *not* the
        //    reciprocal, which is the rule for every other mode.
        // 4. The matrix shape is Table I.4's, for both a square and an oblong
        //    row.
        let mut w = BitWriter::new();
        w.bool(false); // not all_default

        // Index 0: RAW, 8x8, denominator 2.0.
        w.u(3, 7);
        w.f16_bits(0x4000);
        write_raw_ones(&mut w, 8, 8);

        // Index 1: RAW, 8x8, denominator 4.0 — a different value so a matrix
        // built from the wrong table's denominator is visible.
        w.u(3, 7);
        w.f16_bits(0x4400);
        write_raw_ones(&mut w, 8, 8);

        // Index 2: Hornuss with nine known parameters — the sync canary.
        w.u(3, 1);
        for _ in 0..9 {
            w.f16_bits(0x3C00); // 1.0, scaled to 64.0 by the * 64 rule
        }

        // Index 3..=5: Library. Index 6 is RAW again, and 8x16 rather than
        // 8x8, so the shape really comes from Table I.4.
        for _ in 3..6 {
            w.u(3, 0);
        }
        w.u(3, 7);
        w.f16_bits(0x3C00); // denominator = 1.0
        write_raw_ones(&mut w, 8, 16);

        for _ in 7..NUM_DEQUANT_MATRICES {
            w.u(3, 0);
        }
        // I.2.6 with num_groups = 8: u(3) + 1.
        w.u(3, 5);

        let data = w.finish_padded(2);
        let mut r = BitReader::new(&data);
        let geometry = eight_by_eight_geometry();
        let ctx = RawMatrixContext {
            options: ModularOptions::level10(),
            tree_source: TreeSource::Local,
            geometry: &geometry,
        };
        let hf = read_hf_global_params_with(&mut r, 8, Some(&ctx), &mut guard())
            .expect("well-formed HfGlobal with three RAW tables");

        // (2) the canary and the trailing field.
        assert_eq!(hf.num_hf_presets, 6);
        let hornuss = hf.matrices.params(2).expect("index 2");
        assert_eq!(hornuss.mode(), EncodingMode::Hornuss);
        assert_eq!(hornuss.params[0], vec![64.0, 64.0, 64.0]);
        assert_eq!(hornuss.params[2], vec![64.0, 64.0, 64.0]);

        // (3) and (4).
        for channel in 0..3 {
            let m0 = hf.matrices.matrix(0, channel).expect("index 0");
            assert_eq!((m0.rows(), m0.cols()), (8, 8));
            assert!(
                m0.as_slice().iter().all(|v| *v == 2.0),
                "index 0 is 1 * denominator 2.0 everywhere"
            );

            let m1 = hf.matrices.matrix(1, channel).expect("index 1");
            assert!(
                m1.as_slice().iter().all(|v| *v == 4.0),
                "index 1 carries its own denominator"
            );

            let m6 = hf.matrices.matrix(6, channel).expect("index 6");
            assert_eq!((m6.rows(), m6.cols()), (8, 16));
            assert_eq!(m6.as_slice().len(), 128);
            assert!(m6.as_slice().iter().all(|v| *v == 1.0));
        }

        // The Table I.4 shape is the transform's coefficient shape, so a RAW
        // matrix slots into the same pipeline as a computed one.
        assert_eq!(
            (
                TransformType::Dct16x8.coeff_rows(),
                TransformType::Dct16x8.coeff_cols()
            ),
            (8, 16)
        );

        // Nothing is left pending: the deferred-resolution path is gone.
        assert!(hf.matrices.raw_requests().is_empty());
    }

    #[test]
    fn a_raw_sub_bitstream_shifted_by_one_bit_is_caught() {
        // The sync claim above is only worth something if a desynchronised
        // stream actually fails. Truncating the last RAW symbol by one bit
        // leaves the sub-bitstream short, and the rest of the bundle is then
        // read from the wrong place.
        let mut w = BitWriter::new();
        w.bool(false);
        w.u(3, 7);
        w.f16_bits(0x4000);
        write_modular_prelude(&mut w);
        // One symbol short of the 3 * 64 an 8x8 matrix needs.
        for _ in 0..(3 * 64 - 1) {
            msb(&mut w, 2, 2);
        }
        w.u(3, 1); // what the stream *meant* to be index 1's mode
        for _ in 0..9 {
            w.f16_bits(0x3C00);
        }
        for _ in 2..NUM_DEQUANT_MATRICES {
            w.u(3, 0);
        }
        w.u(3, 5);

        let data = w.finish_padded(2);
        let mut r = BitReader::new(&data);
        let geometry = eight_by_eight_geometry();
        let ctx = RawMatrixContext {
            options: ModularOptions::level10(),
            tree_source: TreeSource::Local,
            geometry: &geometry,
        };
        let result = read_hf_global_params_with(&mut r, 8, Some(&ctx), &mut guard());
        // Either the shifted parse errors outright or it lands on different
        // values; what it must not do is silently agree with the aligned one.
        match result {
            Err(_) => {}
            Ok(hf) => {
                let hornuss = hf.matrices.params(1).expect("index 1");
                assert!(
                    hornuss.mode() != EncodingMode::Hornuss
                        || hornuss.params[0] != vec![64.0, 64.0, 64.0]
                        || hf.num_hf_presets != 6,
                    "a one-symbol-short RAW stream decoded as if it were correct"
                );
            }
        }
    }

    #[test]
    fn raw_without_a_modular_context_is_refused_at_the_sub_bitstream() {
        // The old behaviour returned a `RawMatrixRequest` and let the caller
        // carry on reading from a bit position that was already wrong. Now the
        // refusal happens where the sub-bitstream starts.
        let mut w = BitWriter::new();
        w.bool(false);
        w.u(3, 7); // RAW at index 0
        w.f16_bits(0x4000); // denominator = 2.0
        write_raw_ones(&mut w, 8, 8);
        for _ in 1..NUM_DEQUANT_MATRICES {
            w.u(3, 0);
        }
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let err = read_dequant_matrices(&mut r, &mut guard()).expect_err("no context supplied");
        assert!(matches!(err, DecodeError::Unsupported { .. }), "{err}");
        // Refused *after* the denominator and *before* anything else, so the
        // caller cannot mistake the reader for a usable position.
        assert_eq!(r.total_bits_read(), 1 + 3 + 16);
    }

    #[test]
    fn a_raw_matrix_with_a_non_positive_sample_is_rejected() {
        // I.2.4 states that no dequantization value is non-positive or
        // infinite. Token 0 is `UnpackSigned(0) == 0`, so a plane of zeros is a
        // well-formed modular sub-bitstream carrying an ill-formed matrix: the
        // parse must succeed and the matrix must not.
        let mut w = BitWriter::new();
        w.bool(false);
        w.u(3, 7);
        w.f16_bits(0x3C00); // denominator = 1.0
        write_modular_prelude(&mut w);
        for _ in 0..(3 * 64) {
            msb(&mut w, 0, 2); // every sample 0
        }
        for _ in 1..NUM_DEQUANT_MATRICES {
            w.u(3, 0);
        }
        let data = w.finish_padded(2);
        let mut r = BitReader::new(&data);
        let geometry = eight_by_eight_geometry();
        let ctx = RawMatrixContext {
            options: ModularOptions::level10(),
            tree_source: TreeSource::Local,
            geometry: &geometry,
        };
        let m = read_dequant_matrices_with(&mut r, Some(&ctx), &mut guard())
            .expect("the sub-bitstream itself is well formed");
        assert!(
            m.matrix(0, 0).is_err(),
            "a zero entry must be rejected, not passed to I.5.3"
        );
    }

    #[test]
    fn the_raw_stream_index_follows_h41() {
        // Property 1 feeds the MA tree, so a wrong stream index decodes a
        // different image from the same bits whenever the tree branches on it.
        // The formula lives in frame::stream_index; this pins that this module
        // uses it, per index.
        let geometry = eight_by_eight_geometry();
        assert_eq!(
            u64::from(stream_index::dequant_table(&geometry, 0).expect("index 0")),
            1 + 3 * geometry.num_lf_groups()
        );
        assert_eq!(
            stream_index::dequant_table(&geometry, 16).expect("index 16")
                - stream_index::dequant_table(&geometry, 0).expect("index 0"),
            16
        );
        assert!(stream_index::dequant_table(&geometry, 17).is_err());
    }

    #[test]
    fn build_all_produces_every_matrix_and_charges_the_guard() {
        let matrices = DequantMatrices::all_default().expect("defaults");
        let mut g = guard();
        let all = matrices.build_all(&mut g).expect("builds");
        assert_eq!(all.len(), NUM_DEQUANT_MATRICES);
        for (index, per_channel) in all.iter().enumerate() {
            let (rows, cols) = matrix_size(index).expect("size");
            for m in per_channel {
                assert_eq!((m.rows(), m.cols()), (rows, cols));
            }
        }
        assert!(g.charged() > 0);
    }

    #[test]
    fn for_transform_agrees_with_the_index_lookup() {
        let matrices = DequantMatrices::all_default().expect("defaults");
        for t in TransformType::ALL {
            let a = matrices.for_transform(t, 1).expect("matrix");
            let b = matrices
                .matrix(t.dequant_matrix_index(), 1)
                .expect("matrix");
            assert_eq!(a, b, "{t:?}");
            assert_eq!((a.rows(), a.cols()), (t.coeff_rows(), t.coeff_cols()));
        }
    }

    #[test]
    fn hf_global_params_reads_both_rows_of_table_g4() {
        // 1 bit of all_default plus a zero-width num_hf_presets for a
        // single-group frame. Proves the Table G.4 order.
        let mut w = BitWriter::new();
        w.bool(true);
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let hf = read_hf_global_params(&mut r, 1, &mut guard()).expect("valid");
        assert_eq!(r.total_bits_read(), 1);
        assert_eq!(hf.num_hf_presets, 1);
        assert_eq!(hf.matrices, DequantMatrices::all_default().expect("d"));
    }
}
