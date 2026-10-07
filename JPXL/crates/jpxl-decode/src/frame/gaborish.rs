//! The Gabor-like transform (18181-1 J.3) and the 5.2 mirroring primitive.
//!
//! J.3 convolves every colour channel of the whole frame with one symmetric
//! 3x3 kernel. Before normalization the weights are
//!
//! ```text
//! w2 w1 w2       w1 = restoration_filter.gab_C_weight1   (the four edge
//! w1  1 w1                                                neighbours)
//! w2 w1 w2       w2 = restoration_filter.gab_C_weight2   (the four corners)
//! ```
//!
//! with `C` the channel being filtered, so each channel gets its own kernel.
//! The clause then requires the nine weights to be **rescaled uniformly so
//! that they sum to 1**: one scale factor `1 / (1 + 4*w1 + 4*w2)` multiplies
//! every tap, the centre included. That is the whole normalization — there is
//! no separate centre term. [`GaborKernel::new`] is the only place it is
//! computed, and [`GaborKernel::sum`] is what the exit test checks.
//!
//! Taps that fall outside the frame are redirected by `Mirror` (5.2), not
//! clamped: see [`mirror1d`].
//!
//! # Where this sits
//!
//! J.1: the Gabor-like transform runs on the whole frame, and the
//! edge-preserving filter (J.4, [`super::epf`]) runs immediately after it.
//! Wiring both into the decode pipeline is a later sub-slice; this module is a
//! pure function of (planes, parameters).

use crate::frame::error::{FrameError, Result};
use crate::frame::restoration::GaborWeights;

/// Dimensions of an `f32` sample plane stored in raster order.
///
/// The restoration filters work on plain `&[f32]` slices whose length is
/// exactly `width * height`; this pairs a slice with the shape it is read at
/// so the two cannot drift apart silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaneDims {
    /// Samples per row.
    pub width: usize,
    /// Number of rows.
    pub height: usize,
}

impl PlaneDims {
    /// A plane of `width` by `height` samples.
    #[must_use]
    pub const fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }

    /// Number of samples a plane of this shape holds.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.width.saturating_mul(self.height)
    }

    /// Whether the plane holds no samples.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// Checks that `plane` has exactly [`PlaneDims::len`] samples.
    ///
    /// # Errors
    ///
    /// [`FrameError::FieldOutOfRange`] naming `what` when the length differs.
    pub fn check(&self, plane: &[f32], what: &'static str, clause: &'static str) -> Result<()> {
        if plane.len() == self.len() {
            Ok(())
        } else {
            Err(FrameError::out_of_range(
                what,
                clause,
                u64::try_from(plane.len()).unwrap_or(u64::MAX),
            ))
        }
    }
}

/// `usize` to `i64` without a lint-triggering `as` cast.
///
/// Plane dimensions are bounded far below `i64::MAX` by the `Limits` caps, so
/// the saturating fallback is unreachable in practice.
fn as_i64(v: usize) -> i64 {
    i64::try_from(v).unwrap_or(i64::MAX)
}

/// 18181-1 5.2 `Mirror1D`: folds an out-of-range coordinate back into
/// `[0, size)` by reflecting about the half-sample outside each edge.
///
/// The clause writes this recursively — `coord < 0` reflects to `-coord - 1`
/// and `coord >= size` to `2*size - 1 - coord`, each re-entering the function.
/// One reflection is not always enough: on a narrow plane a coordinate can
/// bounce between the two edges several times, which is exactly the case a
/// naive single-step mirror gets wrong. This is the same recursion written as
/// a loop.
///
/// `size == 0` has no valid sample to return; it yields 0 so the function is
/// total, and callers reject empty planes before they get here.
#[must_use]
pub fn mirror1d(coord: i64, size: usize) -> usize {
    if size == 0 {
        return 0;
    }
    let size_i = as_i64(size);
    let mut c = coord;
    loop {
        if c < 0 {
            c = -c - 1;
        } else if c >= size_i {
            c = 2 * size_i - 1 - c;
        } else {
            return usize::try_from(c).unwrap_or(0);
        }
    }
}

/// Reads `plane` at `(x, y)`, mirroring per 5.2 when the coordinate is outside
/// the plane.
#[must_use]
pub fn sample_mirrored(plane: &[f32], dims: PlaneDims, x: i64, y: i64) -> f32 {
    let px = mirror1d(x, dims.width);
    let py = mirror1d(y, dims.height);
    // The index is in range whenever `plane.len() == dims.len()`, which every
    // public entry point checks first.
    plane
        .get(py.saturating_mul(dims.width).saturating_add(px))
        .copied()
        .unwrap_or(0.0)
}

/// Reads `plane` at `(x, y)` without mirroring.
///
/// The caller guarantees the coordinate is in bounds — the interior fast
/// paths prove this per filter — so this skips both `mirror1d` folds. An
/// out-of-bounds coordinate reads 0.0 rather than trapping, matching
/// [`sample_mirrored`]'s totality; debug builds assert the contract.
#[must_use]
pub fn sample_direct(plane: &[f32], dims: PlaneDims, x: i64, y: i64) -> f32 {
    debug_assert!(
        x >= 0 && y >= 0 && x < as_i64(dims.width) && y < as_i64(dims.height),
        "interior reads are in bounds by construction"
    );
    let x = usize::try_from(x).unwrap_or(usize::MAX);
    let y = usize::try_from(y).unwrap_or(usize::MAX);
    plane
        .get(y.saturating_mul(dims.width).saturating_add(x))
        .copied()
        .unwrap_or(0.0)
}

/// Whether every read within `margin` samples of `(x, y)` lands in bounds.
///
/// A filter whose taps never reach further than `margin` from the reference
/// pixel can read interior pixels with [`sample_direct`]: the mirror would
/// be the identity on every tap. `margin` is per-filter (J.3: 1; J.4: 1–3
/// by step) and each filter's test proves its margin against its read
/// pattern.
pub(crate) fn interior_pixel(x: usize, y: usize, dims: PlaneDims, margin: usize) -> bool {
    x >= margin
        && y >= margin
        && x < dims.width.saturating_sub(margin)
        && y < dims.height.saturating_sub(margin)
}

/// The interior rectangle for `margin`: rows and columns whose every pixel
/// satisfies [`interior_pixel`].
///
/// Either range may be empty (a plane narrower than `2 * margin` has no
/// interior); both are empty-safe to iterate. The set equality with the
/// predicate is pinned by `interior_rect_matches_the_predicate`.
pub(crate) fn interior_rect(
    dims: PlaneDims,
    margin: usize,
) -> (std::ops::Range<usize>, std::ops::Range<usize>) {
    let cols = margin..dims.width.saturating_sub(margin);
    let rows = margin..dims.height.saturating_sub(margin);
    (rows, cols)
}

/// The normalized J.3 kernel for one channel.
///
/// Constructed by [`GaborKernel::new`], which applies the clause's uniform
/// rescale; the three fields are already scaled, so [`GaborKernel::sum`] is 1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaborKernel {
    /// Weight of the reference sample.
    pub centre: f32,
    /// Weight of each of the four edge neighbours.
    pub edge: f32,
    /// Weight of each of the four corner neighbours.
    pub corner: f32,
}

impl GaborKernel {
    /// Builds the normalized kernel from one channel's `weight1`/`weight2`.
    ///
    /// # Errors
    ///
    /// [`FrameError::FieldOutOfRange`] if the unnormalized weights do not sum
    /// to a finite non-zero value — a stream can signal arbitrary `F16()`
    /// weights, and `1 + 4*w1 + 4*w2 == 0` would otherwise divide by zero.
    pub fn new(weight1: f32, weight2: f32) -> Result<Self> {
        let unnormalized_sum = 4.0f32.mul_add(weight2, 4.0f32.mul_add(weight1, 1.0));
        if !unnormalized_sum.is_finite() || unnormalized_sum == 0.0 {
            // `value` carries no useful integer here; the field name and
            // clause are what identify the failure.
            return Err(FrameError::out_of_range(
                "gab_weight1/gab_weight2",
                "J.3",
                0,
            ));
        }
        let scale = 1.0 / unnormalized_sum;
        Ok(Self {
            centre: scale,
            edge: weight1 * scale,
            corner: weight2 * scale,
        })
    }

    /// The nine kernel weights summed. J.3 requires this to be 1.
    #[must_use]
    pub fn sum(&self) -> f32 {
        4.0f32.mul_add(self.corner, 4.0f32.mul_add(self.edge, self.centre))
    }

    /// Weight of the tap at offset `(dx, dy)`, each in `-1..=1`.
    #[must_use]
    pub const fn weight_at(&self, dx: i64, dy: i64) -> f32 {
        match (dx, dy) {
            (0, 0) => self.centre,
            (0, _) | (_, 0) => self.edge,
            _ => self.corner,
        }
    }
}

/// Applies J.3 to one channel, writing the result into `output`.
///
/// `input` and `output` must both hold `dims.len()` samples. The transform is
/// not in-place: every output sample reads the *unfiltered* neighbourhood.
///
/// The rows are filtered in bands on a `std::thread::scope` pool. Each output
/// sample is written exactly once from the read-only input, so the threaded
/// run is bit-identical to the serial one.
///
/// # Errors
///
/// [`FrameError::FieldOutOfRange`] if either slice's length disagrees with
/// `dims`.
pub fn gaborish_into(
    input: &[f32],
    output: &mut [f32],
    dims: PlaneDims,
    kernel: &GaborKernel,
) -> Result<()> {
    gaborish_into_with_workers(
        input,
        output,
        dims,
        kernel,
        crate::parallel::worker_count(dims.height, crate::parallel::MIN_ROWS_PER_WORKER),
    )
}

/// [`gaborish_into`] with an explicit worker count.
///
/// `1` runs the serial loop inline; anything larger bands the rows. The
/// parameter exists so tests can prove worker-count independence; callers
/// want [`gaborish_into`].
///
/// # Errors
///
/// As [`gaborish_into`].
pub(crate) fn gaborish_into_with_workers(
    input: &[f32],
    output: &mut [f32],
    dims: PlaneDims,
    kernel: &GaborKernel,
    workers: usize,
) -> Result<()> {
    dims.check(input, "gaborish input plane length", "J.3")?;
    dims.check(output, "gaborish output plane length", "J.3")?;

    let row_bands = crate::parallel::bands(dims.height, workers);
    // A zero-width plane takes the serial path: its bands hold zero cells
    // and `chunks_mut(0)` would panic.
    let band_len = row_bands.first().map_or(0, |band| band.len());
    let band_cells = band_len.saturating_mul(dims.width);
    if row_bands.len() <= 1 || band_cells == 0 {
        gaborish_rows(input, output, dims, kernel, 0..dims.height);
        return Ok(());
    }
    std::thread::scope(|scope| {
        let chunks = output.chunks_mut(band_cells);
        debug_assert_eq!(
            chunks.len(),
            row_bands.len(),
            "bands tile the plane exactly (see parallel::bands_match_chunks_mut)"
        );
        for (band, rows) in chunks.zip(row_bands) {
            scope.spawn(move || {
                gaborish_rows(input, band, dims, kernel, rows);
            });
        }
    });
    Ok(())
}

/// Filters one band of rows into the band's slice.
///
/// Reads use absolute frame coordinates; `output` holds exactly this band's
/// rows, so writes index relative to the band start (`rel_y`).
fn gaborish_rows(
    input: &[f32],
    output: &mut [f32],
    dims: PlaneDims,
    kernel: &GaborKernel,
    band: std::ops::Range<usize>,
) {
    // J.3's taps reach exactly one sample from the reference pixel. The band
    // splits into the interior rect (direct reads, AVX2-dispatched) and the
    // border (mirrored reads); the two cover the band exactly once.
    let (rect_rows, rect_cols) = interior_rect(dims, 1);
    let inner = rect_rows.start.max(band.start)..rect_rows.end.min(band.end);
    // The clamps keep every range inside the band even for degenerate
    // geometries (empty interior, single-row bands); `y - band.start` below
    // can never underflow because every loop range starts at or above it.
    for y in band.start..inner.start.min(band.end) {
        gaborish_mirrored_row(input, output, dims, kernel, band.start, y);
    }
    if rect_cols.is_empty() {
        for y in inner.clone() {
            gaborish_mirrored_row(input, output, dims, kernel, band.start, y);
        }
    } else {
        for y in inner.clone() {
            let rel_y = y - band.start;
            for x in 0..rect_cols.start {
                store_mirrored_pixel(input, output, dims, kernel, x, y, rel_y);
            }
            for x in rect_cols.end..dims.width {
                store_mirrored_pixel(input, output, dims, kernel, x, y, rel_y);
            }
        }
        let rel_start = inner.start.saturating_sub(band.start);
        let rel_end = inner.end.saturating_sub(band.start);
        gaborish_interior_rect(
            input,
            output,
            dims,
            kernel,
            band.start,
            rel_start..rel_end,
            rect_cols,
        );
    }
    for y in inner.end.max(band.start)..band.end {
        gaborish_mirrored_row(input, output, dims, kernel, band.start, y);
    }
}

/// One J.3 pixel through the mirror, for border pixels.
fn gaborish_mirrored_pixel(
    input: &[f32],
    dims: PlaneDims,
    kernel: &GaborKernel,
    x: usize,
    y: usize,
) -> f32 {
    let (xi, yi) = (as_i64(x), as_i64(y));
    let mut acc = 0.0f32;
    for dy in -1i64..=1 {
        for dx in -1i64..=1 {
            let w = kernel.weight_at(dx, dy);
            acc = w.mul_add(sample_mirrored(input, dims, xi + dx, yi + dy), acc);
        }
    }
    acc
}

/// Filters one band row through the mirror into the band's slice.
///
/// `y` is absolute; `band_start` translates it to the band slice.
fn gaborish_mirrored_row(
    input: &[f32],
    output: &mut [f32],
    dims: PlaneDims,
    kernel: &GaborKernel,
    band_start: usize,
    y: usize,
) {
    let rel_y = y - band_start;
    for x in 0..dims.width {
        store_mirrored_pixel(input, output, dims, kernel, x, y, rel_y);
    }
}

/// Filters one border pixel through the mirror into the band's slice.
fn store_mirrored_pixel(
    input: &[f32],
    output: &mut [f32],
    dims: PlaneDims,
    kernel: &GaborKernel,
    x: usize,
    y: usize,
    rel_y: usize,
) {
    let v = gaborish_mirrored_pixel(input, dims, kernel, x, y);
    if let Some(slot) = output.get_mut(rel_y.saturating_mul(dims.width).saturating_add(x)) {
        *slot = v;
    }
}

/// Filters an interior rectangle with direct reads.
///
/// `rel_rows` is relative to the band slice `output`; `cols` is absolute.
/// Every tap of every covered pixel lands in bounds (see
/// `margin_one_covers_all_nine_j3_taps`), so the reads skip the mirror.
/// Dispatched to an AVX2+FMA build where the host supports it; the builds
/// are bit-identical — same IEEE operations in the same lane order, and
/// `mul_add` is a single rounding whether the hardware fuses it or the
/// baseline's `fmaf` libcall emulates it — and
/// `interior_rect_avx2_matches_scalar_bitwise` pins that.
fn gaborish_interior_rect(
    input: &[f32],
    output: &mut [f32],
    dims: PlaneDims,
    kernel: &GaborKernel,
    band_start: usize,
    rel_rows: std::ops::Range<usize>,
    cols: std::ops::Range<usize>,
) {
    #[cfg(target_arch = "x86_64")]
    if jpxl_core::cpu::has_fma() {
        // SAFETY: `gaborish_interior_rect_avx2` only requires AVX2+FMA,
        // which `has_fma` has just confirmed.
        #[allow(unsafe_code)]
        unsafe {
            gaborish_interior_rect_avx2(input, output, dims, kernel, band_start, rel_rows, cols);
        }
        return;
    }
    gaborish_interior_rect_impl(input, output, dims, kernel, band_start, rel_rows, cols);
}

/// [`gaborish_interior_rect`] for AVX2+FMA hosts: eight pixels per step.
///
/// Each lane computes the scalar tap order exactly (`vfmaddps` is one
/// rounding per lane, like `mul_add`), so lanes match the scalar impl bit
/// for bit; the tail (columns past the last full group of eight) runs the
/// shared scalar stencil. Auto-vectorization was tried first and refused —
/// the bounds-checked loads do not version — so this spells the eight-wide
/// loop explicitly.
///
/// # Safety
///
/// The caller must guarantee all of the following, which
/// [`gaborish_rows`] establishes through [`interior_rect`] (see
/// `interior_rect_matches_the_predicate` and
/// `margin_one_covers_all_nine_j3_taps`):
///
/// - the host supports AVX2+FMA (see [`jpxl_core::cpu::has_fma`]);
/// - `cols` holds interior columns only, so `x - 1` and `x + 7` are valid
///   row offsets for every eight-wide step starting in `cols`;
/// - `band_start + rel_y` is an interior row for every `rel_y` in
///   `rel_rows`, so the rows above and below exist;
/// - `output` is the band slice, so `rel_y * width + x + 7` is in bounds
///   for every step.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(
    unsafe_code,
    reason = "intrinsics and pointer loads; see the Safety section"
)]
unsafe fn gaborish_interior_rect_avx2(
    input: &[f32],
    output: &mut [f32],
    dims: PlaneDims,
    kernel: &GaborKernel,
    band_start: usize,
    rel_rows: std::ops::Range<usize>,
    cols: std::ops::Range<usize>,
) {
    use core::arch::x86_64::{_mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps};

    let w = dims.width;
    let centre = _mm256_set1_ps(kernel.centre);
    let edge = _mm256_set1_ps(kernel.edge);
    let corner = _mm256_set1_ps(kernel.corner);
    let zero = _mm256_set1_ps(0.0);
    // Last start column with a full eight-wide group inside `cols`; the
    // max() folds the fewer-than-eight case into an empty vector loop.
    let end8 = cols.end.saturating_sub(8).max(cols.start);
    for rel_y in rel_rows {
        let y = band_start + rel_y;
        // SAFETY: interior rows by the contract; the arithmetic stays inside
        // `input` exactly as the scalar row windows do.
        #[allow(unsafe_code, reason = "proven in-bounds by the Safety contract")]
        let (r0, r1, r2) = unsafe {
            (
                input.as_ptr().add((y - 1) * w),
                input.as_ptr().add(y * w),
                input.as_ptr().add((y + 1) * w),
            )
        };
        let mut x = cols.start;
        while x < end8 {
            // SAFETY: `x - 1` through `x + 7` are valid row offsets and
            // `rel_y * w + x` a valid output slot, by the contract.
            #[allow(unsafe_code, reason = "proven in-bounds by the Safety contract")]
            unsafe {
                let t00 = _mm256_loadu_ps(r0.add(x - 1));
                let t01 = _mm256_loadu_ps(r0.add(x));
                let t02 = _mm256_loadu_ps(r0.add(x + 1));
                let t10 = _mm256_loadu_ps(r1.add(x - 1));
                let t11 = _mm256_loadu_ps(r1.add(x));
                let t12 = _mm256_loadu_ps(r1.add(x + 1));
                let t20 = _mm256_loadu_ps(r2.add(x - 1));
                let t21 = _mm256_loadu_ps(r2.add(x));
                let t22 = _mm256_loadu_ps(r2.add(x + 1));
                // The scalar tap order, tap for tap.
                let mut acc = _mm256_fmadd_ps(corner, t00, zero);
                acc = _mm256_fmadd_ps(edge, t01, acc);
                acc = _mm256_fmadd_ps(corner, t02, acc);
                acc = _mm256_fmadd_ps(edge, t10, acc);
                acc = _mm256_fmadd_ps(centre, t11, acc);
                acc = _mm256_fmadd_ps(edge, t12, acc);
                acc = _mm256_fmadd_ps(corner, t20, acc);
                acc = _mm256_fmadd_ps(edge, t21, acc);
                acc = _mm256_fmadd_ps(corner, t22, acc);
                _mm256_storeu_ps(output.as_mut_ptr().add(rel_y * w + x), acc);
            }
            x += 8;
        }
        // Tail columns past the last full group: the shared scalar stencil.
        if x < cols.end {
            let (Some(r0s), Some(r1s), Some(r2s)) = (
                input.get((y - 1) * w..y * w),
                input.get(y * w..(y + 1) * w),
                input.get((y + 1) * w..(y + 2) * w),
            ) else {
                continue;
            };
            for x in x..cols.end {
                let v = gaborish_stencil_1d(
                    r0s,
                    r1s,
                    r2s,
                    x,
                    kernel.centre,
                    kernel.edge,
                    kernel.corner,
                );
                if let Some(slot) = output.get_mut(rel_y * w + x) {
                    *slot = v;
                }
            }
        }
    }
}

/// One J.3 pixel from three tap rows, in the scalar tap order.
///
/// Shared by the scalar rect impl and the AVX2 tail so the order is defined
/// once. `x` must be an interior column (the lanes the vector loop covers
/// and the tail columns both are); out-of-range taps read 0.0 rather than
/// trapping, matching [`sample_direct`]'s totality.
fn gaborish_stencil_1d(
    r0: &[f32],
    r1: &[f32],
    r2: &[f32],
    x: usize,
    centre: f32,
    edge: f32,
    corner: f32,
) -> f32 {
    let t = |row: &[f32], xx: usize| row.get(xx).copied().unwrap_or(0.0);
    // Row-major, `dy` outer — the scalar loop's order, tap for tap (see
    // `gaborish_interior_rect_impl`): reordering float addition changes the
    // last bit.
    let mut acc = 0.0f32;
    acc = corner.mul_add(t(r0, x - 1), acc);
    acc = edge.mul_add(t(r0, x), acc);
    acc = corner.mul_add(t(r0, x + 1), acc);
    acc = edge.mul_add(t(r1, x - 1), acc);
    acc = centre.mul_add(t(r1, x), acc);
    acc = edge.mul_add(t(r1, x + 1), acc);
    acc = corner.mul_add(t(r2, x - 1), acc);
    acc = edge.mul_add(t(r2, x), acc);
    acc = corner.mul_add(t(r2, x + 1), acc);
    acc
}

/// The interior-rect loop itself: the readable scalar reference and the
/// lock-step oracle for `gaborish_interior_rect_avx2` (see
/// `interior_rect_avx2_matches_scalar_bitwise`). Runs wherever the dispatch
/// falls back: non-x86_64 hosts and hosts without AVX2+FMA.
fn gaborish_interior_rect_impl(
    input: &[f32],
    output: &mut [f32],
    dims: PlaneDims,
    kernel: &GaborKernel,
    band_start: usize,
    rel_rows: std::ops::Range<usize>,
    cols: std::ops::Range<usize>,
) {
    // One-dimensional index math throughout: the tap rows are sliced once
    // per output row and the shared stencil reads forward from them. All
    // indices are in bounds by rect construction
    // (`margin_one_covers_all_nine_j3_taps`); the `.get()` chains only make
    // that total.
    let w = dims.width;
    for rel_y in rel_rows {
        let y = band_start + rel_y;
        let (Some(r0), Some(r1), Some(r2)) = (
            input.get((y - 1) * w..y * w),
            input.get(y * w..(y + 1) * w),
            input.get((y + 1) * w..(y + 2) * w),
        ) else {
            continue;
        };
        for x in cols.clone() {
            let v = gaborish_stencil_1d(r0, r1, r2, x, kernel.centre, kernel.edge, kernel.corner);
            if let Some(slot) = output.get_mut(rel_y * w + x) {
                *slot = v;
            }
        }
    }
}

/// Applies J.3 to one channel, returning a fresh plane.
///
/// # Errors
///
/// As [`gaborish_into`].
pub fn gaborish(input: &[f32], dims: PlaneDims, kernel: &GaborKernel) -> Result<Vec<f32>> {
    let mut output = vec![0.0f32; dims.len()];
    gaborish_into(input, &mut output, dims, kernel)?;
    Ok(output)
}

/// Applies J.3 to the three colour planes `[X, Y, B]`, each with its own
/// kernel built from the matching `gab_C_weight1`/`gab_C_weight2` pair.
///
/// Channel order is the `{x, y, b}` order of Table J.1, which is also the
/// order [`GaborWeights`] stores.
///
/// # Errors
///
/// As [`GaborKernel::new`] and [`gaborish_into`].
pub fn gaborish_planes(
    planes: [&[f32]; 3],
    dims: PlaneDims,
    weights: &GaborWeights,
) -> Result<[Vec<f32>; 3]> {
    let mut out = [
        vec![0.0f32; dims.len()],
        vec![0.0f32; dims.len()],
        vec![0.0f32; dims.len()],
    ];
    for c in 0..3 {
        let w1 = weights.weight1.get(c).copied().unwrap_or_default();
        let w2 = weights.weight2.get(c).copied().unwrap_or_default();
        let kernel = GaborKernel::new(w1, w2)?;
        let input = planes.get(c).copied().unwrap_or(&[]);
        if let Some(dst) = out.get_mut(c) {
            gaborish_into(input, dst, dims, &kernel)?;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(plane: &[f32], dims: PlaneDims, x: usize, y: usize) -> f32 {
        plane.get(y * dims.width + x).copied().unwrap_or(f32::NAN)
    }

    #[test]
    fn mirror1d_matches_the_clause_on_a_wide_plane() {
        // size 4: ... 1 0 | 0 1 2 3 | 3 2 ...
        let got: Vec<usize> = (-4i64..8).map(|c| mirror1d(c, 4)).collect();
        assert_eq!(got, vec![3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0]);
    }

    #[test]
    fn mirror1d_terminates_on_a_one_sample_axis() {
        // The degenerate case: a single reflection off one edge lands outside
        // the other, so a naive one-step mirror produces an out-of-range index.
        // Every coordinate must fold to 0.
        for c in -8i64..8 {
            assert_eq!(mirror1d(c, 1), 0, "coord {c}");
        }
        // Two-wide is the other case a single reflection can get wrong: the
        // pattern has period 4 (0, 1, 1, 0), so a coordinate two steps outside
        // needs a second reflection off the opposite edge.
        let got: Vec<usize> = (-4i64..6).map(|c| mirror1d(c, 2)).collect();
        assert_eq!(got, vec![0, 1, 1, 0, 0, 1, 1, 0, 0, 1]);
    }

    #[test]
    fn rescaled_kernel_sums_to_one() {
        // Proves the J.3 rescale: whatever the signalled weights, the nine
        // taps sum to 1, so a constant plane survives the convolution.
        for (w1, w2) in [
            (0.115_169_525_f32, 0.061_248_592_f32), // Table J.1 defaults
            (0.0, 0.0),
            (2.0, -0.25),
            (-0.1, 0.5),
        ] {
            let k = GaborKernel::new(w1, w2).expect("finite non-zero sum");
            assert!(
                (k.sum() - 1.0).abs() < 1e-6,
                "w1={w1} w2={w2} sum={}",
                k.sum()
            );
        }
    }

    #[test]
    fn degenerate_weights_are_rejected_not_divided_by() {
        // 1 + 4*w1 + 4*w2 == 0 with w1 = -0.25, w2 = 0.
        assert!(GaborKernel::new(-0.25, 0.0).is_err());
        assert!(GaborKernel::new(f32::NAN, 0.0).is_err());
    }

    #[test]
    fn constant_plane_is_unchanged() {
        // The identity that the rescale exists to guarantee, checked on a
        // plane whose every pixel is an edge pixel in at least one axis, so
        // mirroring is exercised throughout.
        let dims = PlaneDims::new(5, 3);
        let input = vec![0.375f32; dims.len()];
        let k = GaborKernel::new(0.115_169_525, 0.061_248_592).expect("valid");
        let out = gaborish(&input, dims, &k).expect("valid");
        for (i, v) in out.iter().enumerate() {
            assert!((v - 0.375).abs() < 1e-6, "sample {i} = {v}");
        }
    }

    #[test]
    fn hand_computed_impulse_response() {
        // 3x3 plane, single 1.0 at the centre, w1 = 0.25, w2 = 0.125.
        // Unnormalized sum = 1 + 4*0.25 + 4*0.125 = 2.5, so scale = 0.4 and
        // the normalized taps are centre 0.4, edge 0.1, corner 0.05.
        //
        // Each output sample is (weight of the tap that lands on the impulse)
        // times 1.0. For the border samples the mirrored taps all land on
        // zeros, so only the real neighbour contributes.
        let dims = PlaneDims::new(3, 3);
        let mut input = vec![0.0f32; 9];
        if let Some(s) = input.get_mut(4) {
            *s = 1.0;
        }
        let k = GaborKernel::new(0.25, 0.125).expect("valid");
        assert_eq!((k.centre, k.edge, k.corner), (0.4, 0.1, 0.05));

        let out = gaborish(&input, dims, &k).expect("valid");
        let expected = [
            0.05, 0.1, 0.05, //
            0.1, 0.4, 0.1, //
            0.05, 0.1, 0.05,
        ];
        for (i, (got, want)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-6, "sample {i}: {got} != {want}");
        }
    }

    #[test]
    fn direct_matches_mirrored_on_interior_coords() {
        // The lemma the interior fast paths rest on: where the mirror is
        // the identity, skipping it reads the same sample. Distinct values
        // per sample so any index slip shows up.
        for width in 1..9usize {
            for height in 1..9usize {
                let dims = PlaneDims::new(width, height);
                let plane: Vec<f32> = (0..dims.len())
                    .map(|i| f32::from(u16::try_from(i % 1000).unwrap_or(0)))
                    .collect();
                for y in 0..height {
                    for x in 0..width {
                        let (xi, yi) = (as_i64(x), as_i64(y));
                        let (direct, mirrored) = (
                            sample_direct(&plane, dims, xi, yi),
                            sample_mirrored(&plane, dims, xi, yi),
                        );
                        assert_eq!(
                            direct.to_bits(),
                            mirrored.to_bits(),
                            "{width}x{height} at ({x},{y})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn margin_one_covers_all_nine_j3_taps() {
        // J.3's interior predicate with margin 1: every interior pixel's
        // nine taps land in bounds, on every plane shape including the
        // degenerate ones (where the interior is empty and the loop below
        // checks nothing — the `checked` count proves the larger planes
        // actually exercise it).
        let mut checked = 0;
        for width in 1..12usize {
            for height in 1..12usize {
                let dims = PlaneDims::new(width, height);
                for y in 0..height {
                    for x in 0..width {
                        if !interior_pixel(x, y, dims, 1) {
                            continue;
                        }
                        checked += 1;
                        for dy in -1i64..=1 {
                            for dx in -1i64..=1 {
                                let (rx, ry) = (as_i64(x) + dx, as_i64(y) + dy);
                                assert!(
                                    rx >= 0 && ry >= 0,
                                    "tap ({dx},{dy}) of ({x},{y}) escapes {width}x{height}"
                                );
                                assert!(
                                    rx < as_i64(width) && ry < as_i64(height),
                                    "tap ({dx},{dy}) of ({x},{y}) escapes {width}x{height}"
                                );
                            }
                        }
                    }
                }
            }
        }
        assert!(checked > 0, "no interior pixel was ever checked");
    }

    #[test]
    fn interior_rect_matches_the_predicate() {
        // The rectangle and the predicate name the same set: every rect
        // pixel satisfies `interior_pixel` and every pixel outside it does
        // not. Margins 0–3 cover J.3 (1) and every J.4 step (1–3).
        for width in 1..12usize {
            for height in 1..12usize {
                let dims = PlaneDims::new(width, height);
                for margin in 0..4usize {
                    let (rows, cols) = interior_rect(dims, margin);
                    for y in 0..height {
                        for x in 0..width {
                            let in_rect = rows.contains(&y) && cols.contains(&x);
                            assert_eq!(
                                in_rect,
                                interior_pixel(x, y, dims, margin),
                                "{width}x{height} margin {margin} at ({x},{y})"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn interior_rect_avx2_matches_scalar_bitwise() {
        // The dispatched rect build returns the scalar impl's bytes exactly,
        // on widths that are and are not multiples of the vector width.
        // (On a host without AVX2 both sides run the scalar impl and the
        // test is vacuous — it proves the dispatch, which only an AVX2 host
        // exercises.)
        let k = GaborKernel::new(0.115_169_525, 0.061_248_592).expect("valid");
        for width in [1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 64, 130] {
            for height in [1, 2, 3, 7, 8, 9, 33, 70] {
                let dims = PlaneDims::new(width, height);
                let input: Vec<f32> = (0..dims.len())
                    .map(|i| f32::from(u16::try_from(i % 1000).unwrap_or(0)) / 1000.0 - 0.2)
                    .collect();
                let (rows, cols) = interior_rect(dims, 1);
                let mut dispatched = vec![0.0f32; dims.len()];
                let mut scalar = vec![0.0f32; dims.len()];
                gaborish_interior_rect(
                    &input,
                    &mut dispatched,
                    dims,
                    &k,
                    0,
                    rows.clone(),
                    cols.clone(),
                );
                gaborish_interior_rect_impl(&input, &mut scalar, dims, &k, 0, rows, cols);
                for (i, (a, b)) in dispatched.iter().zip(scalar.iter()).enumerate() {
                    assert_eq!(a.to_bits(), b.to_bits(), "{width}x{height} sample {i}");
                }
            }
        }
    }

    #[test]
    fn worker_count_never_changes_a_sample() {
        // Threading partitions rows; it must not move a single bit. A ragged
        // 200x70 ramp exercises band edges (70 rows over 8 workers is
        // 9+9+9+9+9+9+9+7) against the serial run.
        let dims = PlaneDims::new(200, 70);
        let input: Vec<f32> = (0..dims.len())
            .map(|i| f32::from(u16::try_from(i % 1000).unwrap_or(0)) / 1000.0)
            .collect();
        let k = GaborKernel::new(0.115_169_525, 0.061_248_592).expect("valid");
        let mut serial = vec![0.0f32; dims.len()];
        gaborish_into_with_workers(&input, &mut serial, dims, &k, 1).expect("valid");
        for workers in [2, 3, 8, 64] {
            let mut threaded = vec![0.0f32; dims.len()];
            gaborish_into_with_workers(&input, &mut threaded, dims, &k, workers).expect("valid");
            assert_eq!(serial.len(), threaded.len(), "workers {workers}");
            for (i, (x, y)) in serial.iter().zip(threaded.iter()).enumerate() {
                assert_eq!(x.to_bits(), y.to_bits(), "workers {workers} sample {i}");
            }
        }
    }

    #[test]
    fn hand_computed_mirroring_on_a_one_pixel_wide_plane() {
        // width 1, height 3. Every horizontal tap folds onto column 0, so the
        // three columns of the kernel collapse into one column of row weights:
        //   row weight = corner + edge + corner   for the rows above/below
        //   row weight = edge   + centre + edge   for the reference row
        // With w1 = 0.25, w2 = 0.125 (centre 0.4, edge 0.1, corner 0.05):
        //   own row  = 0.1 + 0.4 + 0.1 = 0.6
        //   next row = 0.05 + 0.1 + 0.05 = 0.2
        // At y = 0 the row above mirrors back onto row 0, so row 0 gets
        // 0.6 + 0.2 = 0.8 and row 1 gets 0.2.
        let dims = PlaneDims::new(1, 3);
        let input = vec![1.0f32, 0.0, 0.0];
        let k = GaborKernel::new(0.25, 0.125).expect("valid");
        let out = gaborish(&input, dims, &k).expect("valid");

        assert!((at(&out, dims, 0, 0) - 0.8).abs() < 1e-6, "{out:?}");
        assert!((at(&out, dims, 0, 1) - 0.2).abs() < 1e-6, "{out:?}");
        assert!((at(&out, dims, 0, 2) - 0.0).abs() < 1e-6, "{out:?}");
        // Total mass is preserved because the weights sum to 1.
        assert!((out.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn single_pixel_plane_is_the_identity() {
        let dims = PlaneDims::new(1, 1);
        let k = GaborKernel::new(0.115_169_525, 0.061_248_592).expect("valid");
        let out = gaborish(&[0.25], dims, &k).expect("valid");
        assert!((out.first().copied().unwrap_or(0.0) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn each_channel_uses_its_own_weights() {
        // Table J.1 signals weight1/weight2 per channel; a shared kernel would
        // make all three outputs equal here.
        let dims = PlaneDims::new(3, 1);
        let plane = [0.0f32, 1.0, 0.0];
        let weights = GaborWeights {
            weight1: [0.25, 0.5, 0.0],
            weight2: [0.125, 0.0, 0.0],
        };
        let out = gaborish_planes([&plane, &plane, &plane], dims, &weights).expect("valid");
        // Channel b has both weights zero, so its kernel is the identity.
        assert_eq!(out.get(2).map(Vec::as_slice), Some(&plane[..]));
        let x_centre = out.first().and_then(|p| p.get(1)).copied().unwrap_or(0.0);
        let y_centre = out.get(1).and_then(|p| p.get(1)).copied().unwrap_or(0.0);
        assert!(
            (x_centre - y_centre).abs() > 1e-3,
            "{x_centre} vs {y_centre}"
        );
    }

    #[test]
    fn length_mismatch_is_an_error() {
        let dims = PlaneDims::new(4, 4);
        let k = GaborKernel::new(0.1, 0.1).expect("valid");
        assert!(gaborish(&[0.0; 15], dims, &k).is_err());
    }
}
