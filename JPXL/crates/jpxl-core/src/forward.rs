//! Forward (analysis) transform algebra for VarDCT — the exact inverse of the
//! I.9 reconstruction in [`crate::varblock`], slice 13 of `docs/PLAN.md`.
//!
//! The decoder side is normative: 18181-1 Annex I specifies *synthesis*, and
//! [`crate::varblock::TransformType::samples_from_coefficients`] implements it.
//! Encoding has no normative counterpart, but it has an exact one: every I.9
//! reconstruction is an invertible linear map from a varblock's coefficient
//! array to its samples, so "the" forward transform is that map's inverse. This
//! module derives each inverse from the shipped, corpus-proven synthesis form
//! and nothing else.
//!
//! # What is being inverted, per transform type
//!
//! * **Plain `DCTRxC`** (I.9.2) — `IDCT_2D` of I.7.3, inverted by `DCT_2D` of
//!   the same clause. [`crate::dct::dct_2d_in_place`] is the shared kernel; the
//!   `1/sqrt(s)`-per-pass scaling of I.7.2 lives there and is not re-derived
//!   here.
//! * **DCT2x2** (I.9.3) — three `AuxIDCT2x2` passes at `s = 2, 4, 8`. Each pass
//!   is an unnormalized 2x2 Hadamard on four cells, whose four output rows are
//!   mutually orthogonal with norm 4; the inverse is therefore the same
//!   butterfly divided by 4, applied at `s = 8, 4, 2`.
//! * **DCT4x4** (I.9.4) — four independent 4x4 `IDCT_2D`s whose DC values come
//!   from one `AuxIDCT2x2(s = 2)` pass. Forward: four 4x4 `DCT_2D`s, then that
//!   one butterfly inverted.
//! * **Hornuss** (I.9.5) — a permutation plus a rank-one correction. Sample
//!   `(1, 1)` of each 4x4 *is* the centre value, sample `(0, 0)` carries the
//!   coefficient that lives at `(1, 1)` of the residual grid, and every other
//!   sample is its own residual plus the centre. Inverting is exact and needs
//!   no transform at all.
//! * **DCT8x4 / DCT4x8** (I.9.6, I.9.7) — two half-blocks gathered by row
//!   parity with a `{c00 + c01, c00 - c01}` DC pair. Forward: two half `DCT_2D`s
//!   and `c00 = (d0 + d1) / 2`, `c01 = (d0 - d1) / 2`.
//! * **AFV0-3** (I.9.8) — three disjoint regions (the flipped 4x4 AFV quadrant,
//!   a plain 4x4, and a 4x8 half) sharing three coefficients through the DC
//!   triple `d1 = 4(a + b + e)`, `d2 = a - b + e`, `d3 = a - e`. That 3x3 system
//!   is nonsingular, so the region transforms plus a 3x3 solve invert the
//!   clause exactly. The AFV basis is orthonormal (asserted by the clause and
//!   proved in `varblock`'s tests), so the forward quadrant transform is the
//!   basis applied as its own transpose.
//!
//! Note that `AFV_FREQ_POSITION_IS_TRANSPOSED` in the decoder's dequantization
//! matrices is a *weight-placement* flip point, not a basis one; it does not
//! reach this module.
//!
//! # Allocation discipline
//!
//! Nothing on the hot path allocates. Callers pass borrowed views
//! ([`SampleView`], [`CoeffViewMut`]) plus a reusable [`TransformScratch`],
//! and every intermediate is either that scratch or a fixed-size stack array.
//! The owning wrappers ([`TransformType::coefficients_from_samples`],
//! [`lf_from_llf`]) exist for tests and for cold paths, and are the only things
//! here that allocate.
//!
//! # Orientation
//!
//! Unchanged from `varblock`: samples are `R x C` in the varblock's own
//! orientation, coefficients are always landscape (`bheight x bwidth`), and
//! `at(x, y)` means column `x`, row `y`. A DCT16x8 varblock takes 16x8 samples
//! and produces 8x16 coefficients.

// Every index below is derived from a loop bound over a buffer whose length was
// validated when its view was constructed, so bounds checks here would be noise
// rather than a defence.
#![allow(clippy::indexing_slicing)]

use crate::dct::{MAX_TRANSFORM_SIZE, coeff_dims, dct_2d_in_place, idct_2d_in_place};
use crate::varblock::{AFV_BASIS, CoeffMatrix, SampleBlock, TransformType, scale_f};

// ---------------------------------------------------------------------------
// Borrowed views
// ---------------------------------------------------------------------------

/// The shared body of the four view types: a row-major window with a stride.
#[derive(Debug, Clone, Copy)]
struct Grid<'a> {
    data: &'a [f32],
    rows: usize,
    cols: usize,
    stride: usize,
}

/// Does a `rows x cols` window with this `stride` fit in `len` values?
fn window_fits(len: usize, rows: usize, cols: usize, stride: usize) -> bool {
    if cols > stride {
        return false;
    }
    if rows == 0 || cols == 0 {
        return true;
    }
    match (rows - 1)
        .checked_mul(stride)
        .and_then(|s| s.checked_add(cols))
    {
        Some(need) => need <= len,
        None => false,
    }
}

impl<'a> Grid<'a> {
    fn new(data: &'a [f32], rows: usize, cols: usize, stride: usize) -> Option<Self> {
        window_fits(data.len(), rows, cols, stride).then_some(Self {
            data,
            rows,
            cols,
            stride,
        })
    }

    fn at(&self, x: usize, y: usize) -> f32 {
        if x < self.cols && y < self.rows {
            self.data[y * self.stride + x]
        } else {
            0.0
        }
    }
}

/// The mutable twin of [`Grid`].
#[derive(Debug)]
struct GridMut<'a> {
    data: &'a mut [f32],
    rows: usize,
    cols: usize,
    stride: usize,
}

impl<'a> GridMut<'a> {
    fn new(data: &'a mut [f32], rows: usize, cols: usize, stride: usize) -> Option<Self> {
        window_fits(data.len(), rows, cols, stride).then_some(Self {
            data,
            rows,
            cols,
            stride,
        })
    }

    fn at(&self, x: usize, y: usize) -> f32 {
        if x < self.cols && y < self.rows {
            self.data[y * self.stride + x]
        } else {
            0.0
        }
    }

    fn set(&mut self, x: usize, y: usize, value: f32) {
        if x < self.cols && y < self.rows {
            self.data[y * self.stride + x] = value;
        }
    }

    fn fill(&mut self, value: f32) {
        for y in 0..self.rows {
            for x in 0..self.cols {
                self.data[y * self.stride + x] = value;
            }
        }
    }
}

/// Macro-free boilerplate would be four near-identical blocks; these two
/// declarations generate the read-only and read-write view pairs instead.
///
/// The types stay distinct — `SampleView` and `CoeffView` are *not*
/// interchangeable — because mixing sample and coefficient orientation is the
/// failure mode this whole module is defending against (AGENTS.md, "unit-bearing
/// newtypes at every transform boundary").
macro_rules! read_view {
    ($name:ident, $what:literal) => {
        #[doc = concat!("A borrowed, strided, read-only window of ", $what, ".")]
        #[derive(Debug, Clone, Copy)]
        pub struct $name<'a>(Grid<'a>);

        impl<'a> $name<'a> {
            #[doc = concat!("Wraps `data` as `rows x cols` ", $what, " with the given row stride.")]
            ///
            /// Returns `None` if `stride < cols` or `data` is too short, so a
            /// constructed view is always safe to index over its whole window.
            #[must_use]
            pub fn new(data: &'a [f32], rows: usize, cols: usize, stride: usize) -> Option<Self> {
                Grid::new(data, rows, cols, stride).map(Self)
            }

            /// The same, for a tightly packed `rows x cols` buffer.
            #[must_use]
            pub fn contiguous(data: &'a [f32], rows: usize, cols: usize) -> Option<Self> {
                Self::new(data, rows, cols, cols)
            }

            /// Number of rows in the window.
            #[must_use]
            pub const fn rows(&self) -> usize {
                self.0.rows
            }

            /// Number of columns in the window.
            #[must_use]
            pub const fn cols(&self) -> usize {
                self.0.cols
            }

            /// Value at column `x`, row `y`; `0.0` outside the window.
            #[must_use]
            pub fn at(&self, x: usize, y: usize) -> f32 {
                self.0.at(x, y)
            }
        }
    };
}

macro_rules! write_view {
    ($name:ident, $what:literal) => {
        #[doc = concat!("A borrowed, strided, writable window of ", $what, ".")]
        #[derive(Debug)]
        pub struct $name<'a>(GridMut<'a>);

        impl<'a> $name<'a> {
            #[doc = concat!("Wraps `data` as `rows x cols` ", $what, " with the given row stride.")]
            ///
            /// Returns `None` if `stride < cols` or `data` is too short.
            #[must_use]
            pub fn new(
                data: &'a mut [f32],
                rows: usize,
                cols: usize,
                stride: usize,
            ) -> Option<Self> {
                GridMut::new(data, rows, cols, stride).map(Self)
            }

            /// The same, for a tightly packed `rows x cols` buffer.
            #[must_use]
            pub fn contiguous(data: &'a mut [f32], rows: usize, cols: usize) -> Option<Self> {
                Self::new(data, rows, cols, cols)
            }

            /// Number of rows in the window.
            #[must_use]
            pub const fn rows(&self) -> usize {
                self.0.rows
            }

            /// Number of columns in the window.
            #[must_use]
            pub const fn cols(&self) -> usize {
                self.0.cols
            }

            /// Value at column `x`, row `y`; `0.0` outside the window.
            #[must_use]
            pub fn at(&self, x: usize, y: usize) -> f32 {
                self.0.at(x, y)
            }

            /// Writes column `x`, row `y`. Out-of-window writes are dropped.
            pub fn set(&mut self, x: usize, y: usize, value: f32) {
                self.0.set(x, y, value);
            }

            /// Fills the whole window.
            pub fn fill(&mut self, value: f32) {
                self.0.fill(value);
            }
        }
    };
}

read_view!(SampleView, "samples in the varblock's own orientation");
read_view!(CoeffView, "landscape coefficients");
write_view!(SampleViewMut, "samples in the varblock's own orientation");
write_view!(CoeffViewMut, "landscape coefficients");

// ---------------------------------------------------------------------------
// Scratch
// ---------------------------------------------------------------------------

/// Reusable working memory for the forward transforms.
///
/// Sized for the largest varblock Table I.1 allows (DCT256x256), so one
/// instance serves every transform type. Construct it once per worker and pass
/// it to every call; the transforms themselves never allocate.
#[derive(Debug)]
pub struct TransformScratch {
    a: Vec<f32>,
    b: Vec<f32>,
}

impl Default for TransformScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl TransformScratch {
    /// Cells in each of the two internal buffers of [`TransformScratch::new`]:
    /// a full DCT256x256 varblock.
    pub const CELLS: usize = MAX_TRANSFORM_SIZE * MAX_TRANSFORM_SIZE;

    /// Allocates a scratch big enough for every transform type. This is the
    /// only allocation in the module's hot path, and it happens once.
    #[must_use]
    pub fn new() -> Self {
        Self::with_cells(Self::CELLS)
    }

    /// A scratch sized for one transform type — 512 bytes for DCT8x8 rather
    /// than half a megabyte. An encoder that never emits large varblocks pays
    /// only for what it uses.
    #[must_use]
    pub fn for_transform(transform: TransformType) -> Self {
        Self::with_cells(transform.sample_rows() * transform.sample_cols())
    }

    /// A scratch holding two buffers of `cells` values each.
    #[must_use]
    pub fn with_cells(cells: usize) -> Self {
        Self {
            a: vec![0.0f32; cells],
            b: vec![0.0f32; cells],
        }
    }

    /// Is this scratch big enough for a transform of `cells` cells?
    fn fits(&self, cells: usize) -> bool {
        self.a.len() >= cells && self.b.len() >= cells
    }

    /// The two buffers, borrowed disjointly.
    fn pair(&mut self) -> (&mut [f32], &mut [f32]) {
        (&mut self.a, &mut self.b)
    }
}

// ---------------------------------------------------------------------------
// An 8x8 working block, in Annex I's (x, y) indexing
// ---------------------------------------------------------------------------

/// The nine small transform types all live inside one 8x8 varblock, so their
/// forward forms gather into this and scatter out of it.
#[derive(Debug, Clone, Copy)]
struct B8 {
    v: [f32; 64],
}

impl B8 {
    const fn zeros() -> Self {
        Self { v: [0.0f32; 64] }
    }

    fn from_samples(s: &SampleView<'_>) -> Self {
        let mut b = Self::zeros();
        for y in 0..8 {
            for x in 0..8 {
                b.v[y * 8 + x] = s.at(x, y);
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

    fn write_to(&self, out: &mut CoeffViewMut<'_>) {
        for y in 0..8 {
            for x in 0..8 {
                out.set(x, y, self.v[y * 8 + x]);
            }
        }
    }
}

/// The inverse of I.9.3's `AuxIDCT2x2(block, s)`.
///
/// The clause's butterfly sends `(c00, c01, c10, c11)` to the four cells
/// `(2x, 2y) .. (2x+1, 2y+1)` through the sign pattern of a 2x2 Hadamard, whose
/// rows are orthogonal with norm 4; so the inverse is the transposed pattern
/// scaled by `1/4`. Cells outside the `s x s` window are copied through, exactly
/// as the forward clause copies them through.
fn inv_aux_idct_2x2(block: &mut B8, s: usize) {
    let src = *block;
    let num_2x2 = s / 2;
    for y in 0..num_2x2 {
        for x in 0..num_2x2 {
            let o00 = src.at(x * 2, y * 2);
            let o01 = src.at(x * 2 + 1, y * 2);
            let o10 = src.at(x * 2, y * 2 + 1);
            let o11 = src.at(x * 2 + 1, y * 2 + 1);
            block.set(x, y, (o00 + o01 + o10 + o11) * 0.25);
            block.set(num_2x2 + x, y, (o00 + o01 - o10 - o11) * 0.25);
            block.set(x, y + num_2x2, (o00 - o01 + o10 - o11) * 0.25);
            block.set(num_2x2 + x, y + num_2x2, (o00 - o01 - o10 + o11) * 0.25);
        }
    }
}

/// Inverts the single `AuxIDCT2x2(s = 2)` pass that I.9.4 and I.9.5 use to
/// spread one DC value over their four 4x4 sub-blocks.
///
/// `dcs[y * 2 + x]` is the clause's `dcs(x, y)`; the four outputs are written
/// into the top-left 2x2 of `coeffs`.
fn scatter_dc_2x2(dcs: [f32; 4], coeffs: &mut B8) {
    let (o00, o01, o10, o11) = (dcs[0], dcs[1], dcs[2], dcs[3]);
    coeffs.set(0, 0, (o00 + o01 + o10 + o11) * 0.25);
    coeffs.set(1, 0, (o00 + o01 - o10 - o11) * 0.25);
    coeffs.set(0, 1, (o00 - o01 + o10 - o11) * 0.25);
    coeffs.set(1, 1, (o00 - o01 - o10 + o11) * 0.25);
}

// ---------------------------------------------------------------------------
// Per-transform forward forms
// ---------------------------------------------------------------------------

/// I.9.2 inverted: a plain `DCT_2D` over the whole varblock.
fn forward_dct_rc(
    samples: &SampleView<'_>,
    coeffs: &mut CoeffViewMut<'_>,
    rows: usize,
    cols: usize,
    scratch: &mut TransformScratch,
) {
    let (work, tmp) = scratch.pair();
    for y in 0..rows {
        for x in 0..cols {
            work[y * cols + x] = samples.at(x, y);
        }
    }
    dct_2d_in_place(work, tmp, rows, cols);
    let (cr, cc) = coeff_dims(rows, cols);
    for y in 0..cr {
        for x in 0..cc {
            coeffs.set(x, y, work[y * cc + x]);
        }
    }
}

/// I.9.3 DCT2x2 inverted: the three `AuxIDCT2x2` passes, undone largest first.
fn forward_dct2x2(samples: &SampleView<'_>, coeffs: &mut CoeffViewMut<'_>) {
    let mut block = B8::from_samples(samples);
    inv_aux_idct_2x2(&mut block, 8);
    inv_aux_idct_2x2(&mut block, 4);
    inv_aux_idct_2x2(&mut block, 2);
    block.write_to(coeffs);
}

/// I.9.4 DCT4x4 inverted.
fn forward_dct4x4(
    samples: &SampleView<'_>,
    coeffs: &mut CoeffViewMut<'_>,
    scratch: &mut TransformScratch,
) {
    let (work, tmp) = scratch.pair();
    let mut out = B8::zeros();
    let mut dcs = [0.0f32; 4];
    for y in 0..2 {
        for x in 0..2 {
            // I.9.4 writes `result(4x + k, 4y + l) = sample(l, k)`, so the
            // gather is the identity on orientation.
            for l in 0..4 {
                for k in 0..4 {
                    work[l * 4 + k] = samples.at(4 * x + k, 4 * y + l);
                }
            }
            dct_2d_in_place(work, tmp, 4, 4);
            dcs[y * 2 + x] = work[0];
            for iy in 0..4 {
                for ix in (if iy == 0 { 1 } else { 0 })..4 {
                    out.set(x + ix * 2, y + iy * 2, work[iy * 4 + ix]);
                }
            }
        }
    }
    scatter_dc_2x2(dcs, &mut out);
    out.write_to(coeffs);
}

/// I.9.5 Hornuss inverted.
///
/// Purely algebraic: no transform is involved, only the centre/residual
/// permutation and the `residual_sum / 16` correction.
fn forward_hornuss(samples: &SampleView<'_>, coeffs: &mut CoeffViewMut<'_>) {
    let mut out = B8::zeros();
    let mut dcs = [0.0f32; 4];
    for y in 0..2 {
        for x in 0..2 {
            let centre = samples.at(4 * x + 1, 4 * y + 1);
            // Sample (0, 0) carries the residual that belongs at (ix, iy) =
            // (1, 1); sample (1, 1) carries the centre itself.
            out.set(x + 2, y + 2, samples.at(4 * x, 4 * y) - centre);
            for iy in 0..4 {
                for ix in 0..4 {
                    if (ix == 0 && iy == 0) || (ix == 1 && iy == 1) {
                        continue;
                    }
                    out.set(
                        x + ix * 2,
                        y + iy * 2,
                        samples.at(4 * x + ix, 4 * y + iy) - centre,
                    );
                }
            }
            let mut residual_sum = 0.0f32;
            for iy in 0..4 {
                for ix in (if iy == 0 { 1 } else { 0 })..4 {
                    residual_sum += out.at(x + ix * 2, y + iy * 2);
                }
            }
            dcs[y * 2 + x] = centre + residual_sum / 16.0;
        }
    }
    scatter_dc_2x2(dcs, &mut out);
    out.write_to(coeffs);
}

/// I.9.6 DCT8x4 and I.9.7 DCT4x8 inverted. `vertical` selects DCT8x4.
fn forward_dct8x4_or_4x8(
    samples: &SampleView<'_>,
    coeffs: &mut CoeffViewMut<'_>,
    vertical: bool,
    scratch: &mut TransformScratch,
) {
    use crate::varblock::DCT8X4_HALF_INDEX_IS_LOW_COORDINATE;

    let (work, tmp) = scratch.pair();
    let mut out = B8::zeros();
    let mut dcs = [0.0f32; 2];
    for (half, dc) in dcs.iter_mut().enumerate() {
        let placed = if DCT8X4_HALF_INDEX_IS_LOW_COORDINATE {
            half
        } else {
            1 - half
        };
        if vertical {
            for iy in 0..8 {
                for ix in 0..4 {
                    work[iy * 4 + ix] = samples.at(4 * placed + ix, iy);
                }
            }
            dct_2d_in_place(work, tmp, 8, 4);
        } else {
            for iy in 0..4 {
                for ix in 0..8 {
                    work[iy * 8 + ix] = samples.at(ix, 4 * placed + iy);
                }
            }
            dct_2d_in_place(work, tmp, 4, 8);
        }
        // Either way the coefficients are the landscape 4x8 matrix that
        // I.9.6/I.9.7 gather by row parity.
        *dc = work[0];
        for iy in 0..4 {
            for ix in (if iy == 0 { 1 } else { 0 })..8 {
                out.set(ix, half + iy * 2, work[iy * 8 + ix]);
            }
        }
    }
    // dcs = {c(0,0) + c(0,1), c(0,0) - c(0,1)}.
    out.set(0, 0, (dcs[0] + dcs[1]) * 0.5);
    out.set(0, 1, (dcs[0] - dcs[1]) * 0.5);
    out.write_to(coeffs);
}

/// I.9.8 AFV0-3 inverted; `n` is the AFV index (`flip_x = n & 1`,
/// `flip_y = n / 2`).
#[allow(clippy::cast_possible_truncation)]
fn forward_afv(
    samples: &SampleView<'_>,
    coeffs: &mut CoeffViewMut<'_>,
    n: usize,
    scratch: &mut TransformScratch,
) {
    let (work, tmp) = scratch.pair();
    let flip_x = n & 1;
    let flip_y = n / 2;
    let mut out = B8::zeros();

    // Region 1: the flipped 4x4 AFV quadrant. The basis is orthonormal, so the
    // analysis operator is the same table read as a transpose.
    let mut quadrant = [0.0f32; 16];
    for iy in 0..4 {
        for ix in 0..4 {
            let sx = if flip_x == 1 { 3 - ix } else { ix };
            let sy = if flip_y == 1 { 3 - iy } else { iy };
            quadrant[sy * 4 + sx] = samples.at(flip_x * 4 + ix, flip_y * 4 + iy);
        }
    }
    let mut coeff_afv = [0.0f32; 16];
    for (j, slot) in coeff_afv.iter_mut().enumerate() {
        let mut acc = 0.0f64;
        for (p, s) in quadrant.iter().enumerate() {
            acc += f64::from(*s) * AFV_BASIS[j][p];
        }
        *slot = acc as f32;
    }
    let d1 = coeff_afv[0];
    for iy in 0..4 {
        for ix in (if iy == 0 { 1 } else { 0 })..4 {
            out.set(ix * 2, iy * 2, coeff_afv[iy * 4 + ix]);
        }
    }

    // Region 2: the plain 4x4 quadrant beside it.
    let x_base = if flip_x == 1 { 0 } else { 4 };
    for iy in 0..4 {
        for ix in 0..4 {
            work[iy * 4 + ix] = samples.at(x_base + ix, flip_y * 4 + iy);
        }
    }
    dct_2d_in_place(work, tmp, 4, 4);
    let d2 = work[0];
    for iy in 0..4 {
        for ix in (if iy == 0 { 1 } else { 0 })..4 {
            out.set(ix * 2 + 1, iy * 2, work[iy * 4 + ix]);
        }
    }

    // Region 3: the 4x8 half covering the other four rows.
    let y_base = if flip_y == 1 { 0 } else { 4 };
    for iy in 0..4 {
        for ix in 0..8 {
            work[iy * 8 + ix] = samples.at(ix, y_base + iy);
        }
    }
    dct_2d_in_place(work, tmp, 4, 8);
    let d3 = work[0];
    for iy in 0..4 {
        for ix in (if iy == 0 { 1 } else { 0 })..8 {
            out.set(ix, 1 + iy * 2, work[iy * 8 + ix]);
        }
    }

    // The three DC values are `d1 = 4(a + b + e)`, `d2 = a - b + e`,
    // `d3 = a - e`, with `a = c(0,0)`, `b = c(1,0)`, `e = c(0,1)`.
    let sum = d1 * 0.25; // a + b + e
    let b = (sum - d2) * 0.5;
    let a_plus_e = (sum + d2) * 0.5;
    out.set(0, 0, (a_plus_e + d3) * 0.5);
    out.set(1, 0, b);
    out.set(0, 1, (a_plus_e - d3) * 0.5);

    out.write_to(coeffs);
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

/// The forward transform of `transform`: samples in, coefficients out.
///
/// The exact inverse of
/// [`TransformType::samples_from_coefficients`][sfc] in exact arithmetic.
/// `samples` must be `sample_rows() x sample_cols()` and `coeffs` must be
/// `coeff_rows() x coeff_cols()`; a mismatch zeroes `coeffs` rather than
/// panicking or reinterpreting the layout, matching the decoder-side
/// convention.
///
/// Allocation-free: everything beyond `scratch` is a stack array.
///
/// [sfc]: crate::varblock::TransformType::samples_from_coefficients
pub fn forward_varblock_into(
    transform: TransformType,
    samples: &SampleView<'_>,
    coeffs: &mut CoeffViewMut<'_>,
    scratch: &mut TransformScratch,
) {
    if samples.rows() != transform.sample_rows()
        || samples.cols() != transform.sample_cols()
        || coeffs.rows() != transform.coeff_rows()
        || coeffs.cols() != transform.coeff_cols()
        || !scratch.fits(transform.sample_rows() * transform.sample_cols())
    {
        debug_assert!(false, "forward varblock shape mismatch for {transform:?}");
        coeffs.fill(0.0);
        return;
    }
    match transform {
        TransformType::Hornuss => forward_hornuss(samples, coeffs),
        TransformType::Dct2x2 => forward_dct2x2(samples, coeffs),
        TransformType::Dct4x4 => forward_dct4x4(samples, coeffs, scratch),
        TransformType::Dct8x4 => forward_dct8x4_or_4x8(samples, coeffs, true, scratch),
        TransformType::Dct4x8 => forward_dct8x4_or_4x8(samples, coeffs, false, scratch),
        TransformType::Afv0 => forward_afv(samples, coeffs, 0, scratch),
        TransformType::Afv1 => forward_afv(samples, coeffs, 1, scratch),
        TransformType::Afv2 => forward_afv(samples, coeffs, 2, scratch),
        TransformType::Afv3 => forward_afv(samples, coeffs, 3, scratch),
        _ => {
            // I.9.2: every remaining type is a plain DCTRxC.
            let (rows, cols) = transform.dct_shape().unwrap_or((8, 8));
            forward_dct_rc(samples, coeffs, rows, cols, scratch);
        }
    }
}

impl TransformType {
    /// Owning wrapper around [`forward_varblock_into`]: the inverse of
    /// [`TransformType::samples_from_coefficients`].
    ///
    /// Allocates; use the `_into` form on the encoder's search path.
    #[must_use]
    pub fn coefficients_from_samples(self, samples: &SampleBlock) -> CoeffMatrix {
        let mut coeffs = self.empty_coefficients();
        let (cr, cc) = (coeffs.rows(), coeffs.cols());
        let Some(view) = SampleView::contiguous(samples.as_slice(), samples.rows(), samples.cols())
        else {
            debug_assert!(false, "sample block is too short");
            return coeffs;
        };
        let mut scratch = TransformScratch::for_transform(self);
        {
            let Some(mut out) = CoeffViewMut::contiguous(coeffs.as_mut_slice(), cr, cc) else {
                debug_assert!(false, "coefficient matrix is too short");
                return CoeffMatrix::zeros(cr, cc);
            };
            forward_varblock_into(self, &view, &mut out, &mut scratch);
        }
        coeffs
    }
}

// ---------------------------------------------------------------------------
// I.8 inverted — LF samples from the LLF coefficients
// ---------------------------------------------------------------------------

/// The exact inverse of [`crate::varblock::llf_from_lf`], into a caller-provided
/// view.
///
/// `llf` is the varblock's LLF sub-rectangle: `bheight/8` rows by `bwidth/8`
/// columns, landscape, as it sits in the top-left corner of the coefficient
/// matrix. `lf` receives the varblock's rectangle of the 8x downsampled image,
/// in **image orientation** (`block_dims().0` rows by `block_dims().1` columns).
///
/// I.8 scales each LLF cell by `ScaleF(y, cy) * ScaleF(x, cx)` after a `DCT_2D`
/// over the LF rectangle, so the inverse divides by the same product and runs
/// `IDCT_2D`. For the nine transforms whose LLF is a single cell the map is the
/// identity, both ways.
///
/// Allocation-free.
pub fn lf_from_llf_into(
    transform: TransformType,
    llf: &CoeffView<'_>,
    lf: &mut SampleViewMut<'_>,
    scratch: &mut TransformScratch,
) {
    let (block_rows, block_cols) = transform.block_dims();
    let cx = block_rows.max(block_cols);
    let cy = block_rows.min(block_cols);

    if llf.rows() != cy
        || llf.cols() != cx
        || lf.rows() != block_rows
        || lf.cols() != block_cols
        || !scratch.fits(cx * cy)
    {
        debug_assert!(false, "LF/LLF shape mismatch for {transform:?}");
        lf.fill(0.0);
        return;
    }

    if !transform.llf_is_transformed() {
        // 1x1 for every such type, so orientation cannot differ.
        lf.set(0, 0, llf.at(0, 0));
        return;
    }

    let (work, tmp) = scratch.pair();
    for y in 0..cy {
        for x in 0..cx {
            work[y * cx + x] = llf.at(x, y) / (scale_f(y, cy) * scale_f(x, cx));
        }
    }
    // `DCT_2D` on a `cy x cx` matrix with `cx >= cy` is shape-preserving, so
    // `IDCT_2D` is too.
    debug_assert_eq!(coeff_dims(cy, cx), (cy, cx));
    idct_2d_in_place(work, tmp, cy, cx);

    // I.8 transposes a portrait varblock's LF rectangle on the way in; undo it.
    for y in 0..block_rows {
        for x in 0..block_cols {
            let value = if block_rows > block_cols {
                work[x * cx + y]
            } else {
                work[y * cx + x]
            };
            lf.set(x, y, value);
        }
    }
}

/// Owning wrapper around [`lf_from_llf_into`]; the encoder-side counterpart of
/// [`crate::varblock::llf_from_lf`].
#[must_use]
pub fn lf_from_llf(transform: TransformType, llf: &CoeffMatrix) -> SampleBlock {
    let (block_rows, block_cols) = transform.block_dims();
    let mut lf = SampleBlock::zeros(block_rows, block_cols);
    let Some(src) = CoeffView::contiguous(llf.as_slice(), llf.rows(), llf.cols()) else {
        debug_assert!(false, "LLF matrix is too short");
        return lf;
    };
    let mut scratch = TransformScratch::for_transform(transform);
    {
        let Some(mut dst) = SampleViewMut::contiguous(lf.as_mut_slice(), block_rows, block_cols)
        else {
            debug_assert!(false, "LF block is too short");
            return SampleBlock::zeros(block_rows, block_cols);
        };
        lf_from_llf_into(transform, &src, &mut dst, &mut scratch);
    }
    lf
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
mod tests {
    use super::*;
    use crate::varblock::llf_from_lf;

    /// Deterministic LCG (Numerical Recipes constants); no `rand` dependency,
    /// and the same generator the `dct` and `varblock` tests use.
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

    /// Tolerance for a forward/inverse pair on unit-magnitude data.
    ///
    /// The pair is exact in exact arithmetic; the residual is pure `f32`
    /// rounding. **Measured**, over every transform type at eight seeds each:
    /// the worst deviation is `5.4e-7` (DCT32x16) and the median is `3.6e-7`,
    /// i.e. a few ULPs of 1.0, and it does *not* grow with block size — the
    /// per-pass `1/sqrt(s)` normalization keeps intermediate magnitudes near
    /// the input's, and the sizes from 32 up accumulate their matrix products
    /// in `f64`. So the bound is flat rather than size-scaled: a size-scaled
    /// bound would be nearly 1000x too loose at DCT256x256 and would stop
    /// discriminating anything there.
    ///
    /// The argument is kept so callers read as per-transform, and so that a
    /// future transform with genuinely worse conditioning has somewhere to go.
    fn tolerance(_t: TransformType) -> f32 {
        1e-6
    }

    fn forward(t: TransformType, samples: &SampleBlock) -> CoeffMatrix {
        t.coefficients_from_samples(samples)
    }

    fn random_coeffs(t: TransformType, rng: &mut Lcg, scale: f32) -> CoeffMatrix {
        let mut c = t.empty_coefficients();
        for slot in c.as_mut_slice() {
            *slot = rng.next(scale);
        }
        c
    }

    // -- Forward/inverse pairing -------------------------------------------

    /// **The headline pair test.** For every transform type, the forward form
    /// must recover the coefficients that the *proven* I.9 reconstruction was
    /// given. Coefficients first, because that direction is the one that pins
    /// the layout: a forward transform that inverts the samples but scrambles
    /// coefficient positions fails here and nowhere else.
    #[test]
    fn forward_recovers_the_coefficients_of_every_transform() {
        let mut rng = Lcg::new(0x1357_9bdf);
        for t in TransformType::ALL {
            let coeffs = random_coeffs(t, &mut rng, 1.0);
            let samples = t.samples_from_coefficients(&coeffs);
            let back = forward(t, &samples);
            assert_eq!(
                (back.rows(), back.cols()),
                (t.coeff_rows(), t.coeff_cols()),
                "{t:?} coefficient shape"
            );
            let tol = tolerance(t);
            for y in 0..back.rows() {
                for x in 0..back.cols() {
                    assert_close(
                        back.at(x, y),
                        coeffs.at(x, y),
                        tol,
                        &format!("{t:?} coefficient ({x},{y})"),
                    );
                }
            }
        }
    }

    /// The other direction: samples through forward then the proven inverse.
    /// Catches a forward form that is *a* left inverse on the coefficient
    /// subspace but not a right inverse — impossible for a square invertible
    /// map, which is exactly what this asserts these all are.
    #[test]
    fn inverse_of_forward_is_the_identity_on_samples() {
        let mut rng = Lcg::new(0x2468_ace0);
        for t in TransformType::ALL {
            let n = t.sample_rows() * t.sample_cols();
            let data: Vec<f32> = (0..n).map(|_| rng.next(1.0)).collect();
            let samples = SampleBlock::from_rows_cols(t.sample_rows(), t.sample_cols(), data);
            let coeffs = forward(t, &samples);
            let back = t.samples_from_coefficients(&coeffs);
            let tol = tolerance(t);
            for y in 0..back.rows() {
                for x in 0..back.cols() {
                    assert_close(
                        back.at(x, y),
                        samples.at(x, y),
                        tol,
                        &format!("{t:?} sample ({x},{y})"),
                    );
                }
            }
        }
    }

    /// Extreme inputs, at the magnitude an 8-bit encoder actually sees: a
    /// saturated block and a `+/-255` checkerboard, whose energy sits in the
    /// highest-frequency coefficient of every small transform.
    #[test]
    fn pair_survives_extreme_sample_blocks() {
        for t in TransformType::ALL {
            let (rows, cols) = (t.sample_rows(), t.sample_cols());
            for kind in 0..2 {
                let data: Vec<f32> = (0..rows * cols)
                    .map(|i| {
                        let checker = kind == 1 && ((i / cols) + (i % cols)) % 2 != 0;
                        if checker { -255.0 } else { 255.0 }
                    })
                    .collect();
                let samples = SampleBlock::from_rows_cols(rows, cols, data);
                let coeffs = forward(t, &samples);
                let back = t.samples_from_coefficients(&coeffs);
                let tol = 255.0 * tolerance(t);
                for y in 0..rows {
                    for x in 0..cols {
                        assert_close(
                            back.at(x, y),
                            samples.at(x, y),
                            tol,
                            &format!("{t:?} kind {kind} sample ({x},{y})"),
                        );
                    }
                }
            }
        }
    }

    // -- Impulse responses: the scaling discriminator ----------------------

    /// **The kernel-scaling discriminator.**
    ///
    /// A round trip cannot see a scale error that both directions share (a
    /// forward missing `1/sqrt(s)` paired with an inverse missing `sqrt(s)`
    /// round-trips perfectly). So this test never composes the two: it takes
    /// the impulse responses of the *proven* I.9 inverse — one unit coefficient
    /// at a time — and requires the forward form to return that unit
    /// coefficient and nothing else. Every legal coefficient position of every
    /// 8x8-varblock transform is exercised, which is all 27 types' bespoke
    /// reconstructions plus DCT8x8.
    #[test]
    fn forward_of_each_impulse_response_recovers_a_unit_coefficient() {
        for t in TransformType::ALL {
            if t.num_blocks() != 1 {
                continue;
            }
            let (cr, cc) = (t.coeff_rows(), t.coeff_cols());
            for uy in 0..cr {
                for ux in 0..cc {
                    let mut coeffs = t.empty_coefficients();
                    coeffs.set(ux, uy, 1.0);
                    let samples = t.samples_from_coefficients(&coeffs);
                    let back = forward(t, &samples);
                    for y in 0..cr {
                        for x in 0..cc {
                            let expected = if (x, y) == (ux, uy) { 1.0 } else { 0.0 };
                            assert_close(
                                back.at(x, y),
                                expected,
                                2e-6,
                                &format!("{t:?} impulse ({ux},{uy}) -> ({x},{y})"),
                            );
                        }
                    }
                }
            }
        }
    }

    /// The same statement for the large transforms, at a sampled set of
    /// positions (a full sweep of DCT256x256 would be 65536 transforms of
    /// 65536 cells). The positions are chosen to catch the two classic
    /// orientation faults: `(1, 0)` versus `(0, 1)` distinguishes the two axes,
    /// and the LLF corner cells distinguish the LLF sub-rectangle from the HF
    /// region.
    #[test]
    fn forward_of_large_transform_impulses_recovers_unit_coefficients() {
        let mut rng = Lcg::new(0x0fed_cba9);
        for t in TransformType::ALL {
            if t.num_blocks() == 1 {
                continue;
            }
            let (cr, cc) = (t.coeff_rows(), t.coeff_cols());
            let (cx, cy) = (cc / 8, cr / 8);
            let mut positions = vec![
                (0usize, 0usize),
                (1, 0),
                (0, 1),
                (cx - 1, cy - 1),
                (cx, cy.saturating_sub(1)),
                (cc - 1, cr - 1),
                (cc - 1, 0),
                (0, cr - 1),
            ];
            for _ in 0..4 {
                let x = ((rng.next(1.0) + 1.0) * 0.5 * (cc as f32)) as usize;
                let y = ((rng.next(1.0) + 1.0) * 0.5 * (cr as f32)) as usize;
                positions.push((x.min(cc - 1), y.min(cr - 1)));
            }
            for (ux, uy) in positions {
                let mut coeffs = t.empty_coefficients();
                coeffs.set(ux, uy, 1.0);
                let samples = t.samples_from_coefficients(&coeffs);
                let back = forward(t, &samples);
                let tol = tolerance(t);
                for y in 0..cr {
                    for x in 0..cc {
                        let expected = if (x, y) == (ux, uy) { 1.0 } else { 0.0 };
                        assert_close(
                            back.at(x, y),
                            expected,
                            tol,
                            &format!("{t:?} impulse ({ux},{uy}) -> ({x},{y})"),
                        );
                    }
                }
            }
        }
    }

    /// A constant sample block must put all its energy in `c(0, 0)`, with the
    /// exact value of the constant — I.7.2's normalization, seen from the
    /// forward side, and for the bespoke reconstructions the conspiracy of
    /// their DC constructions read backwards.
    #[test]
    fn constant_block_lands_entirely_in_the_dc() {
        for t in TransformType::ALL {
            let (rows, cols) = (t.sample_rows(), t.sample_cols());
            let samples = SampleBlock::from_rows_cols(rows, cols, vec![-1.75; rows * cols]);
            let coeffs = forward(t, &samples);
            let tol = tolerance(t);
            assert_close(coeffs.at(0, 0), -1.75, tol, &format!("{t:?} DC"));
            for y in 0..coeffs.rows() {
                for x in 0..coeffs.cols() {
                    if (x, y) == (0, 0) {
                        continue;
                    }
                    assert_close(
                        coeffs.at(x, y),
                        0.0,
                        tol,
                        &format!("{t:?} non-DC ({x},{y})"),
                    );
                }
            }
        }
    }

    // -- Orientation -------------------------------------------------------

    /// The type-level orientation statement from the forward side: DCT16x8
    /// consumes 16x8 samples and produces 8x16 coefficients, and transposing
    /// the samples turns it into DCT8x16 with *the same* coefficient array.
    /// A row/column swap anywhere in the forward pipeline breaks the second
    /// half while leaving every round trip green.
    #[test]
    fn landscape_coefficients_from_portrait_samples() {
        let mut rng = Lcg::new(0x16b8_8b61);
        let tall = TransformType::Dct16x8;
        let wide = TransformType::Dct8x16;

        let data: Vec<f32> = (0..128).map(|_| rng.next(1.0)).collect();
        let portrait = SampleBlock::from_rows_cols(16, 8, data.clone());
        let mut transposed = vec![0.0f32; 128];
        for y in 0..16 {
            for x in 0..8 {
                transposed[x * 16 + y] = data[y * 8 + x];
            }
        }
        let landscape = SampleBlock::from_rows_cols(8, 16, transposed);

        let a = forward(tall, &portrait);
        let b = forward(wide, &landscape);
        assert_eq!((a.rows(), a.cols()), (8, 16), "DCT16x8 coefficient shape");
        assert_eq!((b.rows(), b.cols()), (8, 16), "DCT8x16 coefficient shape");
        for y in 0..8 {
            for x in 0..16 {
                assert_close(a.at(x, y), b.at(x, y), 1e-5, &format!("({x},{y})"));
            }
        }
    }

    /// Strides are honoured: transforming a varblock read out of a larger
    /// plane, and written into a larger coefficient buffer, must give the same
    /// answer as the packed call. This is the view layer's whole purpose — the
    /// encoder reads varblocks straight out of an image plane.
    #[test]
    fn strided_views_match_packed_views() {
        let mut rng = Lcg::new(0x57a1_ded0);
        let t = TransformType::Dct16x8;
        let (rows, cols) = (t.sample_rows(), t.sample_cols());
        let data: Vec<f32> = (0..rows * cols).map(|_| rng.next(1.0)).collect();
        let packed = forward(t, &SampleBlock::from_rows_cols(rows, cols, data.clone()));

        // The same samples at offset (3, 5) of a 40-wide plane.
        let stride = 40usize;
        let mut plane = vec![0.0f32; stride * (rows + 9)];
        for y in 0..rows {
            for x in 0..cols {
                plane[(y + 5) * stride + x + 3] = data[y * cols + x];
            }
        }
        let view = SampleView::new(&plane[5 * stride + 3..], rows, cols, stride).expect("view");

        let (cr, cc) = (t.coeff_rows(), t.coeff_cols());
        let out_stride = 33usize;
        let mut out_buf = vec![9.5f32; out_stride * cr];
        let mut scratch = TransformScratch::new();
        {
            let mut out = CoeffViewMut::new(&mut out_buf, cr, cc, out_stride).expect("view");
            forward_varblock_into(t, &view, &mut out, &mut scratch);
        }
        for y in 0..cr {
            for x in 0..cc {
                assert_close(
                    out_buf[y * out_stride + x],
                    packed.at(x, y),
                    0.0,
                    &format!("strided ({x},{y})"),
                );
            }
        }
        // The padding beyond the window is untouched.
        for y in 0..cr {
            for x in cc..out_stride {
                assert_close(out_buf[y * out_stride + x], 9.5, 0.0, "padding");
            }
        }
    }

    /// Views reject buffers they cannot address, so no constructed view can
    /// index out of range.
    #[test]
    fn views_reject_short_buffers_and_bad_strides() {
        let data = [0.0f32; 16];
        assert!(SampleView::contiguous(&data, 4, 4).is_some());
        assert!(SampleView::contiguous(&data, 4, 5).is_none());
        assert!(SampleView::new(&data, 4, 4, 3).is_none(), "stride < cols");
        assert!(SampleView::new(&data, 3, 4, 6).is_some(), "2*6 + 4 <= 16");
        assert!(SampleView::new(&data, 3, 4, 7).is_none(), "2*7 + 4 > 16");
        assert!(SampleView::new(&data, usize::MAX, 4, usize::MAX).is_none());

        let mut data = [0.0f32; 16];
        assert!(CoeffViewMut::contiguous(&mut data, 2, 8).is_some());
        assert!(CoeffViewMut::contiguous(&mut data, 3, 8).is_none());
    }

    /// A scratch sized for one transform must be rejected by a bigger one
    /// rather than silently truncating it: [`TransformScratch::for_transform`]
    /// makes under-sizing possible, so the guard has to exist and be tested.
    /// (Debug builds assert; the release contract is a zeroed output.)
    #[test]
    fn undersized_scratch_is_rejected() {
        let small = TransformScratch::for_transform(TransformType::Dct8x8);
        assert!(small.fits(64));
        assert!(!small.fits(256));
        assert!(TransformScratch::for_transform(TransformType::Dct16x8).fits(128));
        assert!(TransformScratch::new().fits(TransformScratch::CELLS));
    }

    // -- AFV discriminators ------------------------------------------------

    /// **The AFV corner discriminator**, mirroring the wave-8 lesson that a
    /// transpose and an anti-transpose are both involutions and therefore
    /// invisible to an inverse-composition test.
    ///
    /// AFV0-3 differ only in which corner the AFV quadrant occupies. Feeding
    /// AFV`i`'s reconstruction to AFV`j`'s forward form for `i != j` must not
    /// return the coefficients: if the flips were dropped, or applied with the
    /// wrong sense, the four variants would collapse into one and this would
    /// pass. The asymmetric input is essential — a symmetric block is fixed by
    /// every flip.
    #[test]
    fn afv_variants_are_not_interchangeable() {
        let afvs = [
            TransformType::Afv0,
            TransformType::Afv1,
            TransformType::Afv2,
            TransformType::Afv3,
        ];
        // The block has to excite all three regions. A coefficient in the 4x8
        // half alone does *not* discriminate `flip_x`: that half spans all
        // eight columns and its row band depends only on `flip_y`, so AFV0 and
        // AFV1 place it identically and cross-application round-trips to 1e-7.
        // (Measured, not assumed — the first version of this test used exactly
        // that coefficient and passed for the wrong reason.) So: one
        // coefficient read only by the AFV quadrant, one only by the plain 4x4,
        // one only by the 4x8 half, plus a DC.
        let mut coeffs = CoeffMatrix::zeros(8, 8);
        coeffs.set(2, 2, 1.0); // AFV quadrant
        coeffs.set(3, 0, 0.7); // plain 4x4
        coeffs.set(6, 1, 0.4); // 4x8 half
        coeffs.set(0, 0, 0.5);

        for (i, ti) in afvs.iter().enumerate() {
            let samples = ti.samples_from_coefficients(&coeffs);
            for (j, tj) in afvs.iter().enumerate() {
                let back = forward(*tj, &samples);
                let mut worst = 0.0f32;
                for y in 0..8 {
                    for x in 0..8 {
                        worst = worst.max((back.at(x, y) - coeffs.at(x, y)).abs());
                    }
                }
                if i == j {
                    assert!(worst < 2e-5, "AFV{i} must invert itself (worst {worst})");
                } else {
                    assert!(
                        worst > 1e-2,
                        "AFV{i} reconstruction must not invert under AFV{j} (worst {worst})"
                    );
                }
            }
        }
    }

    /// The AFV quadrant lands in the corner selected by `(flip_x, flip_y)`.
    /// Coefficient `(0, 0)` alone gives a constant block, which is corner-blind,
    /// so this uses a coefficient read only by the AFV quadrant — `c(2, 2)`,
    /// i.e. `coeff_afv[5]` — and asserts the excited 4x4 is the expected one.
    #[test]
    fn afv_quadrant_occupies_the_flip_selected_corner() {
        let mut coeffs = CoeffMatrix::zeros(8, 8);
        coeffs.set(2, 2, 1.0);
        for (n, t) in [
            TransformType::Afv0,
            TransformType::Afv1,
            TransformType::Afv2,
            TransformType::Afv3,
        ]
        .into_iter()
        .enumerate()
        {
            let samples = t.samples_from_coefficients(&coeffs);
            let (fx, fy) = (n & 1, n / 2);
            let mut inside = 0.0f32;
            let mut outside = 0.0f32;
            for y in 0..8 {
                for x in 0..8 {
                    let v = samples.at(x, y).abs();
                    if (x / 4 == fx) && (y / 4 == fy) {
                        inside = inside.max(v);
                    } else {
                        outside = outside.max(v);
                    }
                }
            }
            assert!(inside > 1e-2, "AFV{n} quadrant must be excited ({inside})");
            assert!(
                outside < 1e-6,
                "AFV{n} non-quadrant regions must be silent ({outside})"
            );
            // And the forward form recovers it from exactly that corner.
            let back = forward(t, &samples);
            assert_close(back.at(2, 2), 1.0, 2e-5, &format!("AFV{n} c(2,2)"));
        }
    }

    /// I.9.6/I.9.7 place their two half-blocks by row parity. The forward form
    /// must read them back from the same halves: a coefficient in an even row
    /// may only come from half 0's samples.
    #[test]
    fn dct8x4_and_dct4x8_halves_are_read_back_from_their_own_half() {
        for (t, vertical) in [
            (TransformType::Dct8x4, true),
            (TransformType::Dct4x8, false),
        ] {
            let mut coeffs = CoeffMatrix::zeros(8, 8);
            coeffs.set(3, 2, 1.0); // even row -> half 0
            let samples = t.samples_from_coefficients(&coeffs);
            // Zero the *other* half of the samples; half 0's coefficients must
            // survive untouched.
            let mut punched = samples.clone();
            for i in 0..8 {
                for j in 4..8 {
                    if vertical {
                        punched.set(j, i, 0.0);
                    } else {
                        punched.set(i, j, 0.0);
                    }
                }
            }
            let back = forward(t, &punched);
            assert_close(back.at(3, 2), 1.0, 2e-5, &format!("{t:?} half-0 coeff"));
        }
    }

    // -- I.8 / lf_from_llf -------------------------------------------------

    /// **The `lf_from_llf` pair proof.** `lf_from_llf(llf_from_lf(x)) == x` on
    /// random LF planes, for every transform type — including the nine whose
    /// LLF is a single cell, where both directions must be the identity.
    #[test]
    fn lf_from_llf_inverts_llf_from_lf() {
        let mut rng = Lcg::new(0x11f8_0000);
        for t in TransformType::ALL {
            let (br, bc) = t.block_dims();
            let data: Vec<f32> = (0..br * bc).map(|_| rng.next(1.0)).collect();
            let lf = SampleBlock::from_rows_cols(br, bc, data);
            let llf = llf_from_lf(t, &lf);
            let back = lf_from_llf(t, &llf);
            assert_eq!((back.rows(), back.cols()), (br, bc), "{t:?} LF shape");
            let tol = 1e-6;
            for y in 0..br {
                for x in 0..bc {
                    assert_close(
                        back.at(x, y),
                        lf.at(x, y),
                        tol,
                        &format!("{t:?} LF ({x},{y})"),
                    );
                }
            }
        }
    }

    /// And the other composition order, on random LLF coefficient rectangles.
    /// Both are needed: the pair could be a one-sided inverse only if the map
    /// were singular, and asserting both is how that is ruled out rather than
    /// assumed.
    #[test]
    fn llf_from_lf_inverts_lf_from_llf() {
        let mut rng = Lcg::new(0x11f8_1111);
        for t in TransformType::ALL {
            let (br, bc) = t.block_dims();
            let (cx, cy) = (br.max(bc), br.min(bc));
            let data: Vec<f32> = (0..cx * cy).map(|_| rng.next(1.0)).collect();
            let llf = CoeffMatrix::from_landscape(cy, cx, data);
            let lf = lf_from_llf(t, &llf);
            let back = llf_from_lf(t, &lf);
            assert_eq!((back.rows(), back.cols()), (cy, cx), "{t:?} LLF shape");
            let tol = 1e-6;
            for y in 0..cy {
                for x in 0..cx {
                    assert_close(
                        back.at(x, y),
                        llf.at(x, y),
                        tol,
                        &format!("{t:?} LLF ({x},{y})"),
                    );
                }
            }
        }
    }

    /// The end-to-end LF statement the encoder actually needs: for a varblock
    /// whose samples are known, the LF plane derived from its forward
    /// coefficients' LLF corner equals the DC of each of its 8x8 blocks. This
    /// is the round trip of the decoder's own I.8 proof, run the other way, and
    /// it is what makes "encode the LF image, encode the HF coefficients"
    /// consistent.
    #[test]
    fn lf_from_forward_llf_is_the_per_block_dc() {
        let mut rng = Lcg::new(0x11f8_2222);
        for t in TransformType::ALL {
            if !t.llf_is_transformed() {
                continue;
            }
            let (br, bc) = t.block_dims();
            let (cx, cy) = (t.coeff_cols() / 8, t.coeff_rows() / 8);

            // A band-limited varblock: only LLF coefficients are nonzero, so
            // each 8x8 block's DC captures everything.
            let mut coeffs = t.empty_coefficients();
            for y in 0..cy {
                for x in 0..cx {
                    coeffs.set(x, y, rng.next(1.0));
                }
            }
            let samples = t.samples_from_coefficients(&coeffs);

            // The LLF corner of the forward transform of those samples.
            let forward_coeffs = forward(t, &samples);
            let mut llf = CoeffMatrix::zeros(cy, cx);
            for y in 0..cy {
                for x in 0..cx {
                    llf.set(x, y, forward_coeffs.at(x, y));
                }
            }
            let lf = lf_from_llf(t, &llf);

            // The DC of each 8x8 block of the same samples.
            let mut scratch = TransformScratch::new();
            let mut block = [0.0f32; 64];
            for by in 0..br {
                for bx in 0..bc {
                    for y in 0..8 {
                        for x in 0..8 {
                            block[y * 8 + x] = samples.at(bx * 8 + x, by * 8 + y);
                        }
                    }
                    let (work, tmp) = scratch.pair();
                    work[..64].copy_from_slice(&block);
                    dct_2d_in_place(work, tmp, 8, 8);
                    assert_close(
                        lf.at(bx, by),
                        work[0],
                        2e-5,
                        &format!("{t:?} LF block ({bx},{by})"),
                    );
                }
            }
        }
    }

    /// The `_into` forms are the allocation-free API and must agree with the
    /// owning wrappers cell for cell. Also exercises reusing one scratch across
    /// every transform type in turn, which is how the encoder will call it.
    #[test]
    fn into_forms_match_the_owning_wrappers() {
        let mut rng = Lcg::new(0x1470_0741);
        let mut scratch = TransformScratch::new();
        for t in TransformType::ALL {
            let (rows, cols) = (t.sample_rows(), t.sample_cols());
            let data: Vec<f32> = (0..rows * cols).map(|_| rng.next(1.0)).collect();
            let samples = SampleBlock::from_rows_cols(rows, cols, data.clone());
            let owned = forward(t, &samples);

            let view = SampleView::contiguous(&data, rows, cols).expect("view");
            let (cr, cc) = (t.coeff_rows(), t.coeff_cols());
            let mut buf = vec![0.0f32; cr * cc];
            {
                let mut out = CoeffViewMut::contiguous(&mut buf, cr, cc).expect("view");
                forward_varblock_into(t, &view, &mut out, &mut scratch);
            }
            for (i, v) in buf.iter().enumerate() {
                assert_close(*v, owned.as_slice()[i], 0.0, &format!("{t:?} cell {i}"));
            }

            // And the LF direction.
            let (br, bc) = t.block_dims();
            let (cx, cy) = (br.max(bc), br.min(bc));
            let llf_data: Vec<f32> = (0..cx * cy).map(|_| rng.next(1.0)).collect();
            let llf = CoeffMatrix::from_landscape(cy, cx, llf_data.clone());
            let owned_lf = lf_from_llf(t, &llf);
            let src = CoeffView::contiguous(&llf_data, cy, cx).expect("view");
            let mut lf_buf = vec![0.0f32; br * bc];
            {
                let mut dst = SampleViewMut::contiguous(&mut lf_buf, br, bc).expect("view");
                lf_from_llf_into(t, &src, &mut dst, &mut scratch);
            }
            for (i, v) in lf_buf.iter().enumerate() {
                assert_close(*v, owned_lf.as_slice()[i], 0.0, &format!("{t:?} LF {i}"));
            }
        }
    }
}
