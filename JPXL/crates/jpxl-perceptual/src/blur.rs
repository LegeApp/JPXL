//! Separable Gaussian blur (σ = 1.5) as a recursive filter.
//!
//! The local means and second moments SSIMULACRA2 compares are Gaussian
//! blurs. A direct kernel of that width is affordable, but the metric is
//! defined against the recursive approximation of Charalampidis (2016),
//! "Recursive implementation of the Gaussian filter using truncated cosine
//! functions": the Gaussian is fitted by three cosines truncated at a radius
//! `N`, and each cosine is produced by a two-pole recursion fed by the input
//! `N + 1` samples to either side. The constants below are that derivation
//! evaluated once for σ = 1.5 and written down, so the filter does not depend
//! on the host's `exp`/`cos`.
//!
//! Boundaries are zero-padded: samples outside the plane contribute nothing.
//! That is part of the metric's definition (it slightly darkens the blurred
//! border) and is reproduced, not corrected.
//!
//! The recursion state is kept in `f64`. The three generators are poles on
//! the unit circle (feedback `2 cos ω_k` and `-1`), so in `f32` every
//! rounding error persists as an undamped ripple that random-walks along the
//! row or column; reference implementations carry a few 1e-6 of it, and the
//! metric's rectified edge maps turn it into a size-dependent error floor in
//! flat regions. In `f64` the ripple is ~1e-14 and the filter is, for
//! `f32` inputs, the exact truncated-cosine Gaussian the metric defines. The
//! measured consequence is documented in `tests/parity.rs`.
//!
//! Each output is computed from its row (or column) alone, so any partition
//! into rows or column strips gives bit-identical results; the horizontal
//! pass runs its rows in fixed bands on the executor and the vertical pass
//! runs its column strips there too, each strip writing its own disjoint
//! columns of the output in place.

use crate::bands::{BAND_ROWS, band_of, mutable_bands};
use crate::executor::BandExecutor;

/// Truncation radius `N = round(3.2795 σ + 0.2546)` for σ = 1.5.
pub(crate) const RADIUS: isize = 5;

/// Input gains of the three cosine components (k = 1, 3, 5).
const MUL_IN: [f64; 3] = [
    0.055_295_235_726_086_61,
    -0.058_836_687_026_949_98,
    0.012_955_819_110_517_063,
];

/// First-order feedback `2 cos(ω_k)` of the three components; the
/// second-order feedback is exactly `-1` for all three.
const MUL_PREV: [f64; 3] = [
    1.902_113_032_590_307,
    1.175_570_504_584_946_3,
    1.224_646_799_147_353_2e-16,
];

/// Reusable blur with its horizontal-pass scratch plane.
#[derive(Debug, Default, Clone)]
pub struct Blur {
    temp: Vec<f32>,
}

/// A blur's input: a plane, or a per-sample product of two planes rounded to
/// `f32` as it enters the horizontal pass.
#[derive(Clone, Copy)]
pub(crate) enum BlurInput<'a> {
    /// A single row-major plane.
    Plane(&'a [f32]),
    /// The per-sample product `a * b`.
    Product(&'a [f32], &'a [f32]),
}

impl Blur {
    /// A blur with no scratch allocated yet; the first call allocates.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Blurs `input` (`width × height`, row-major) into `output`.
    ///
    /// The horizontal pass runs in fixed row bands and the vertical pass in
    /// fixed column strips, both on `executor`; every output sample depends
    /// on its row (then its column) alone, so the partition cannot change a
    /// value.
    ///
    /// # Panics
    ///
    /// In debug builds, if the plane lengths do not match the dimensions.
    pub fn blur_plane(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        width: usize,
        height: usize,
        executor: &dyn BandExecutor,
    ) {
        self.blur_input(BlurInput::Plane(input), output, width, height, executor);
    }

    /// Blurs the per-sample product `a * b` without materialising that
    /// full-frame product plane.
    ///
    /// Each multiplication is first rounded into the horizontal pass's `f32`
    /// padded row, exactly as it was when callers built a separate product
    /// plane before blurring it.
    pub(crate) fn blur_product_plane(
        &mut self,
        a: &[f32],
        b: &[f32],
        output: &mut [f32],
        width: usize,
        height: usize,
        executor: &dyn BandExecutor,
    ) {
        self.blur_input(BlurInput::Product(a, b), output, width, height, executor);
    }

    fn blur_input(
        &mut self,
        input: BlurInput<'_>,
        output: &mut [f32],
        width: usize,
        height: usize,
        executor: &dyn BandExecutor,
    ) {
        match input {
            BlurInput::Plane(input) => debug_assert_eq!(input.len(), width * height),
            BlurInput::Product(a, b) => {
                debug_assert_eq!(a.len(), width * height);
                debug_assert_eq!(b.len(), width * height);
            }
        }
        debug_assert_eq!(output.len(), width * height);
        if width == 0 || height == 0 {
            return;
        }
        // The horizontal pass writes every temp sample, so a reused temp needs
        // no re-zeroing; resize only grows (zeroed) or truncates.
        self.temp.resize(width * height, 0.0);
        let band_len = width.saturating_mul(BAND_ROWS);
        let bands = mutable_bands(vec![self.temp.as_mut_slice()], band_len);
        executor.run(bands.len(), &|index| {
            let Some(mut outs) = bands.take(index) else {
                return;
            };
            let Some(out_band) = outs.pop() else {
                return;
            };
            let band_input = match input {
                BlurInput::Plane(input) => BlurInput::Plane(band_of(input, index, band_len)),
                BlurInput::Product(a, b) => {
                    BlurInput::Product(band_of(a, index, band_len), band_of(b, index, band_len))
                }
            };
            horizontal_band(band_input, out_band, width);
        });
        vertical_pass(&self.temp, output, width, height, executor);
    }
}

/// One three-component recursion step: feeds `sum` (the two truncated-window
/// samples) through the poles and returns the summed output.
#[inline]
// The `f64` state is rounded to the plane's `f32` once per output: that
// single rounding is the point of accumulating in `f64`.
#[allow(clippy::cast_possible_truncation)]
fn step(sum: f32, prev: &mut [f64; 3], prev2: &mut [f64; 3]) -> f32 {
    let sum = f64::from(sum);
    let [p1, p3, p5] = *prev;
    let [q1, q3, q5] = *prev2;
    let o1 = sum * MUL_IN[0] + MUL_PREV[0] * p1 - q1;
    let o3 = sum * MUL_IN[1] + MUL_PREV[1] * p3 - q3;
    let o5 = sum * MUL_IN[2] + MUL_PREV[2] * p5 - q5;
    *prev2 = [p1, p3, p5];
    *prev = [o1, o3, o5];
    (o1 + o3 + o5) as f32
}

/// Zero padding on the left of a row so the first window sample
/// (`n - N - 1` at `n = 1 - N`) is addressable: `2N`.
const LEFT_PAD: usize = 2 * RADIUS_USIZE;
const RADIUS_USIZE: usize = 5;

/// Rows processed together in the horizontal pass: each is one independent
/// recursion lane (the row analogue of the vertical pass's column lanes), so
/// the compiler vectorises the lane loop.
///
/// The count trades instruction-level parallelism against register pressure,
/// and the balance is sharp. One step keeps six live `[f64; ROW_LANES]`
/// arrays — `prev` and `prev2` for each of the three poles — plus `sum` and
/// `acc`. At eight lanes that is twelve of the sixteen `ymm` registers AVX2
/// has, before addressing, loads, or the `f32` conversion, so the recursion
/// spilled its whole state to the stack every sample: profiling put
/// `horizontal_band_avx2`, the encoder's hottest leaf, at 26% stack traffic
/// and 22% shuffles against under 10% real vector work. Six lanes fit, and
/// still carry more than one dependency chain.
///
/// Measured with the `row_lanes_bench` microbench below, ns/sample on a
/// 4000-wide band, plane / product: 2 lanes 2.06/2.18, 4 lanes 1.99/2.15,
/// 6 lanes 1.66/1.98, 8 lanes 2.65/2.62, 12 lanes 2.63/2.74, 16 lanes
/// 3.36/3.40. The curve is not monotonic in either direction, so re-run the
/// microbench before changing this for a different microarchitecture — and
/// note that a host with AVX-512 has twice the register file and will want a
/// different optimum.
///
/// The value cannot change a result. Each row is an independent recursion,
/// and rows past the last full group go through the scalar path;
/// `the_lane_grouping_does_not_change_the_result` pins that against every
/// remainder count a band height can produce.
const ROW_LANES: usize = 6;

/// Horizontal pass over one band: groups of [`ROW_LANES`] rows through the
/// lane recursion, remainder rows through the scalar one. Every row's output
/// depends on that row alone and the lane arithmetic is exactly [`step`]'s,
/// so the grouping cannot change a value. Dispatched to an AVX2 build where
/// the host supports it.
pub(crate) fn horizontal_band(input: BlurInput<'_>, out_band: &mut [f32], width: usize) {
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::has_avx2() {
        // SAFETY: `horizontal_band_avx2` only requires that the host support
        // AVX2, which `has_avx2` has just confirmed.
        #[allow(unsafe_code)]
        unsafe {
            horizontal_band_avx2(input, out_band, width);
        }
        return;
    }
    horizontal_band_impl(input, out_band, width);
}

/// [`horizontal_band`] compiled for AVX2.
///
/// Calling it is `unsafe` unless the host supports AVX2 (see
/// [`crate::cpu::has_avx2`]); that is the whole contract.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn horizontal_band_avx2(input: BlurInput<'_>, out_band: &mut [f32], width: usize) {
    horizontal_band_impl(input, out_band, width);
}

#[inline(always)]
fn horizontal_band_impl(input: BlurInput<'_>, out_band: &mut [f32], width: usize) {
    if width == 0 {
        return;
    }
    let group = width * ROW_LANES;
    let mut padded_t = Vec::new();
    let mut padded = Vec::new();
    match input {
        BlurInput::Plane(in_band) => {
            let mut ins = in_band.chunks_exact(group);
            let mut outs = out_band.chunks_exact_mut(group);
            for (rows_in, rows_out) in ins.by_ref().zip(outs.by_ref()) {
                build_padded_lanes(BlurInput::Plane(rows_in), width, &mut padded_t);
                horizontal_padded_lanes(rows_out, width, &padded_t);
            }
            for (row_in, row_out) in ins
                .remainder()
                .chunks_exact(width)
                .zip(outs.into_remainder().chunks_exact_mut(width))
            {
                horizontal_row(row_in, row_out, &mut padded);
            }
        }
        BlurInput::Product(a, b) => {
            let mut a_groups = a.chunks_exact(group);
            let mut b_groups = b.chunks_exact(group);
            let mut outs = out_band.chunks_exact_mut(group);
            for ((a_rows, b_rows), rows_out) in
                a_groups.by_ref().zip(b_groups.by_ref()).zip(outs.by_ref())
            {
                build_padded_lanes(BlurInput::Product(a_rows, b_rows), width, &mut padded_t);
                horizontal_padded_lanes(rows_out, width, &padded_t);
            }
            for ((a_row, b_row), row_out) in a_groups
                .remainder()
                .chunks_exact(width)
                .zip(b_groups.remainder().chunks_exact(width))
                .zip(outs.into_remainder().chunks_exact_mut(width))
            {
                horizontal_product_row(a_row, b_row, row_out, &mut padded);
            }
        }
    }
}

/// Builds the lane-interleaved padded rows: sample `x` of lane `r` sits at
/// `(LEFT_PAD + x) * ROW_LANES + r`, so each recursion step loads one
/// contiguous [`ROW_LANES`]-wide window. The pads stay zero from the first
/// allocation and the body is fully overwritten, exactly as the scalar
/// padded row is. Product lanes round `a * b` into `f32` here, exactly as
/// [`horizontal_product_row`] does.
#[inline(always)]
fn build_padded_lanes(input: BlurInput<'_>, width: usize, padded_t: &mut Vec<f32>) {
    let n = (width + 3 * RADIUS_USIZE) * ROW_LANES;
    if padded_t.len() != n {
        padded_t.clear();
        padded_t.resize(n, 0.0);
    }
    let Some(body) = padded_t.get_mut(LEFT_PAD * ROW_LANES..(LEFT_PAD + width) * ROW_LANES) else {
        return;
    };
    match input {
        BlurInput::Plane(rows) => {
            for (r, row) in rows.chunks_exact(width).enumerate() {
                for (slot, &v) in body.iter_mut().skip(r).step_by(ROW_LANES).zip(row) {
                    *slot = v;
                }
            }
        }
        BlurInput::Product(a, b) => {
            for (r, (a_row, b_row)) in a.chunks_exact(width).zip(b.chunks_exact(width)).enumerate()
            {
                for ((slot, &av), &bv) in body
                    .iter_mut()
                    .skip(r)
                    .step_by(ROW_LANES)
                    .zip(a_row)
                    .zip(b_row)
                {
                    *slot = av * bv;
                }
            }
        }
    }
}

/// The recursion over [`ROW_LANES`] padded rows at once: the same warm-up and
/// the same per-sample [`pole_step`] arithmetic as [`horizontal_padded`],
/// lane-parallel across the rows.
#[inline(always)]
fn horizontal_padded_lanes(out: &mut [f32], width: usize, padded_t: &[f32]) {
    let mut prev = [[0.0f64; ROW_LANES]; 3];
    let mut prev2 = [[0.0f64; ROW_LANES]; 3];
    let warm = RADIUS_USIZE - 1;
    let lane_sum = |i: usize| -> [f64; ROW_LANES] {
        let mut sum = [0.0f64; ROW_LANES];
        let l = padded_t.get(i * ROW_LANES..(i + 1) * ROW_LANES);
        let r = padded_t.get((i + LEFT_PAD) * ROW_LANES..(i + LEFT_PAD + 1) * ROW_LANES);
        if let (Some(l), Some(r)) = (l, r) {
            for ((s, &lv), &rv) in sum.iter_mut().zip(l).zip(r) {
                *s = f64::from(lv + rv);
            }
        }
        sum
    };
    for i in 0..warm {
        step_lanes(lane_sum(i), &mut prev, &mut prev2);
    }
    // One `&mut` row per lane, so each lane's output is a plain sequential
    // walk rather than a stride-`width` scatter. Held as an array of
    // `Option`s so the lane count follows `ROW_LANES` instead of being spelled
    // out eight times; a short band that cannot fill every lane returns, as
    // the eight-way destructuring did.
    let mut chunks = out.chunks_exact_mut(width);
    let mut rows: [Option<&mut [f32]>; ROW_LANES] = core::array::from_fn(|_| chunks.next());
    if rows.iter().any(Option::is_none) {
        return;
    }
    for n in 0..width {
        let v = step_lanes(lane_sum(n + warm), &mut prev, &mut prev2);
        for (row, &val) in rows.iter_mut().zip(v.iter()) {
            if let Some(slot) = row.as_mut().and_then(|r| r.get_mut(n)) {
                *slot = val;
            }
        }
    }
}

/// One three-component recursion step over [`ROW_LANES`] independent lanes.
/// Per lane this is exactly [`step`]: the same expression order, the same
/// `o1 + o3 + o5` summation, one `f32` rounding of the summed output.
#[allow(clippy::cast_possible_truncation)]
#[inline(always)]
fn step_lanes(
    sum: [f64; ROW_LANES],
    prev: &mut [[f64; ROW_LANES]; 3],
    prev2: &mut [[f64; ROW_LANES]; 3],
) -> [f32; ROW_LANES] {
    let [p1, p3, p5] = prev;
    let [q1, q3, q5] = prev2;
    let mut acc = [0.0f64; ROW_LANES];
    pole_step(&sum, p1, q1, &mut acc, MUL_IN[0], MUL_PREV[0], true);
    pole_step(&sum, p3, q3, &mut acc, MUL_IN[1], MUL_PREV[1], false);
    pole_step(&sum, p5, q5, &mut acc, MUL_IN[2], MUL_PREV[2], false);
    let mut out = [0.0f32; ROW_LANES];
    for (o, &a) in out.iter_mut().zip(&acc) {
        *o = a as f32;
    }
    out
}

/// Horizontal recursive pass over one row, through a zero-padded copy so the
/// window reads are plain slice walks with no per-sample bounds branch.
fn horizontal_row(input: &[f32], output: &mut [f32], padded: &mut Vec<f32>) {
    let width = input.len();
    padded.clear();
    padded.resize(width + 3 * RADIUS_USIZE, 0.0);
    if let Some(body) = padded.get_mut(LEFT_PAD..LEFT_PAD + width) {
        body.copy_from_slice(input);
    }
    horizontal_padded(output, padded);
}

fn horizontal_product_row(a: &[f32], b: &[f32], output: &mut [f32], padded: &mut Vec<f32>) {
    let width = a.len().min(b.len());
    padded.clear();
    padded.resize(width + 3 * RADIUS_USIZE, 0.0);
    if let Some(body) = padded.get_mut(LEFT_PAD..LEFT_PAD + width) {
        for ((slot, &a), &b) in body.iter_mut().zip(a).zip(b) {
            *slot = a * b;
        }
    }
    horizontal_padded(output, padded);
}

fn horizontal_padded(output: &mut [f32], padded: &[f32]) {
    // Output index n runs from 1 - N; the left window sample sits at padded
    // index n + N - 1 (i.e. `i`) and the right one at i + 2N.
    let lefts = padded.get(..).unwrap_or(&[]);
    let rights = padded.get(LEFT_PAD..).unwrap_or(&[]);
    let mut prev = [0.0f64; 3];
    let mut prev2 = [0.0f64; 3];
    let warm = RADIUS_USIZE - 1;
    for (&l, &r) in lefts.iter().zip(rights).take(warm) {
        step(l + r, &mut prev, &mut prev2);
    }
    for ((o, &l), &r) in output
        .iter_mut()
        .zip(lefts.iter().skip(warm))
        .zip(rights.iter().skip(warm))
    {
        *o = step(l + r, &mut prev, &mut prev2);
    }
}

/// Columns processed together in the vertical pass: each strip is one
/// executor item writing its own buffer.
pub(crate) const STRIP: usize = 64;

/// Vertical recursive pass over every column, strips in parallel, each strip
/// writing its own disjoint column range of the row-major output directly.
/// The arithmetic per column is untouched — only the destination addressing
/// changed from a per-strip buffer plus scatter to in-place row windows — so
/// the output is bit-identical to the former scatter form.
fn vertical_pass(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    executor: &dyn BandExecutor,
) {
    let strips = width.div_ceil(STRIP);
    let out = DisjointColumns::new(output);
    executor.run(strips, &|index| {
        let x0 = index * STRIP;
        let cols = (width - x0).min(STRIP);
        vertical_strip(input, &out, width, height, x0, cols);
    });
}

/// A row-major plane shared across strip workers, each writing row windows of
/// a column range no other worker touches.
pub(crate) struct DisjointColumns {
    ptr: *mut f32,
    len: usize,
}

// SAFETY: every worker writes only row windows of its own `x0 .. x0 + cols`
// column range, and the strip ranges partition the columns, so no element is
// ever aliased by two workers.
#[allow(unsafe_code)]
unsafe impl Sync for DisjointColumns {}

impl DisjointColumns {
    pub(crate) fn new(plane: &mut [f32]) -> Self {
        Self {
            ptr: plane.as_mut_ptr(),
            len: plane.len(),
        }
    }

    /// The `cols` samples starting at `offset`, as one mutable row window;
    /// `None` when the window leaves the plane.
    ///
    /// # Safety
    ///
    /// No other thread may read or write this window for the returned
    /// borrow's lifetime; the strip partition guarantees that here.
    // The `&self`-to-`&mut` shape is the point of the type: it is a manual
    // interior-mutability cell whose disjointness contract lives in `unsafe`.
    #[allow(unsafe_code, clippy::mut_from_ref)]
    pub(crate) unsafe fn window(&self, offset: usize, cols: usize) -> Option<&mut [f32]> {
        let end = offset.checked_add(cols)?;
        if end > self.len {
            return None;
        }
        // SAFETY: the range is in bounds (checked above) and unaliased (the
        // caller's contract).
        Some(unsafe { core::slice::from_raw_parts_mut(self.ptr.add(offset), cols) })
    }
}

/// The two-pole recursion state of one strip, held per pole across the columns
/// (structure-of-arrays). Each pole's per-column update is then an independent
/// lane, so the compiler vectorises the column loop. `prev[k]`/`prev2[k]` are
/// the two previous outputs of pole `k` for every column.
#[derive(Debug)]
pub(crate) struct StripState {
    prev: [Vec<f64>; 3],
    prev2: [Vec<f64>; 3],
    /// `f64::from(top + bottom)` for the current row, one per column.
    sum: Vec<f64>,
    /// The summed three-pole output for the current row, one per column.
    acc: Vec<f64>,
}

impl StripState {
    pub(crate) fn new(cols: usize) -> Self {
        Self {
            prev: [vec![0.0; cols], vec![0.0; cols], vec![0.0; cols]],
            prev2: [vec![0.0; cols], vec![0.0; cols], vec![0.0; cols]],
            sum: vec![0.0; cols],
            acc: vec![0.0; cols],
        }
    }

    /// Returns the state to the all-zero start over `cols` columns, exactly
    /// as [`Self::new`] builds it.
    pub(crate) fn reset(&mut self, cols: usize) {
        for lane in self.prev.iter_mut().chain(self.prev2.iter_mut()) {
            lane.clear();
            lane.resize(cols, 0.0);
        }
        self.sum.clear();
        self.sum.resize(cols, 0.0);
        self.acc.clear();
        self.acc.resize(cols, 0.0);
    }
}

/// One pole's per-column recursion step. `first` seeds `acc`, the others add
/// to it, so `acc` ends the row holding `o1 + o3 + o5` in that fixed order —
/// exactly [`step`]'s summation. Reads `p`/`q` before overwriting them, so the
/// state advances identically. Four independent lanes, so this vectorises.
#[inline(always)]
fn pole_step(
    sum: &[f64],
    p: &mut [f64],
    q: &mut [f64],
    acc: &mut [f64],
    mul_in: f64,
    mul_prev: f64,
    first: bool,
) {
    for (((&s, p), q), acc) in sum
        .iter()
        .zip(p.iter_mut())
        .zip(q.iter_mut())
        .zip(acc.iter_mut())
    {
        let o = s * mul_in + mul_prev * *p - *q;
        *q = *p;
        *p = o;
        if first {
            *acc = o;
        } else {
            *acc += o;
        }
    }
}

/// Advances every column's state by one row, and — when `out` is `Some` —
/// writes the summed output rounded to `f32` once.
#[inline(always)]
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn vertical_row(
    top: &[f32],
    bottom: &[f32],
    state: &mut StripState,
    out: Option<&mut [f32]>,
) {
    let StripState {
        prev,
        prev2,
        sum,
        acc,
    } = state;
    for ((&t, &b), s) in top.iter().zip(bottom).zip(sum.iter_mut()) {
        *s = f64::from(t + b);
    }
    let [p1, p3, p5] = prev;
    let [q1, q3, q5] = prev2;
    pole_step(sum, p1, q1, acc, MUL_IN[0], MUL_PREV[0], true);
    pole_step(sum, p3, q3, acc, MUL_IN[1], MUL_PREV[1], false);
    pole_step(sum, p5, q5, acc, MUL_IN[2], MUL_PREV[2], false);
    if let Some(out) = out {
        for (&a, o) in acc.iter().zip(out.iter_mut()) {
            *o = a as f32;
        }
    }
}

/// Vertical pass over columns `x0 .. x0 + cols` of `input`, writing each
/// row's window of `out` in place. Dispatched to an AVX2 build where the host
/// supports it.
fn vertical_strip(
    input: &[f32],
    out: &DisjointColumns,
    width: usize,
    height: usize,
    x0: usize,
    cols: usize,
) {
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::has_avx2() {
        // SAFETY: `vertical_strip_avx2` only requires that the host support
        // AVX2, which `has_avx2` has just confirmed.
        #[allow(unsafe_code)]
        unsafe {
            vertical_strip_avx2(input, out, width, height, x0, cols);
        }
        return;
    }
    vertical_strip_impl(input, out, width, height, x0, cols);
}

/// [`vertical_strip`] compiled for AVX2.
///
/// Calling it is `unsafe` unless the host supports AVX2 (see
/// [`crate::cpu::has_avx2`]); that is the whole contract.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn vertical_strip_avx2(
    input: &[f32],
    out: &DisjointColumns,
    width: usize,
    height: usize,
    x0: usize,
    cols: usize,
) {
    vertical_strip_impl(input, out, width, height, x0, cols);
}

#[inline(always)]
fn vertical_strip_impl(
    input: &[f32],
    out: &DisjointColumns,
    width: usize,
    height: usize,
    x0: usize,
    cols: usize,
) {
    let zeros = vec![0.0f32; cols];
    let mut state = StripState::new(cols);
    let h = isize::try_from(height).unwrap_or(isize::MAX);
    let row = |i: isize| -> &[f32] {
        if i < 0 || i >= h {
            return &zeros;
        }
        usize::try_from(i)
            .ok()
            .and_then(|i| input.get(i * width + x0..i * width + x0 + cols))
            .unwrap_or(&zeros)
    };
    let mut n = 1 - RADIUS;
    while n < h {
        let top = row(n - RADIUS - 1);
        let bottom = row(n + RADIUS - 1);
        // SAFETY: this strip's `x0 .. x0 + cols` columns are its own — see
        // `DisjointColumns` — and row `n` is in bounds for `0 <= n < h`.
        #[allow(unsafe_code)]
        let out_row = usize::try_from(n)
            .ok()
            .and_then(|n| unsafe { out.window(n * width + x0, cols) });
        vertical_row(top, bottom, &mut state, out_row);
        n += 1;
    }
}

#[cfg(test)]
#[allow(clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::executor::{ScopedThreadExecutor, SerialExecutor};

    #[test]
    fn a_constant_interior_is_preserved_and_the_border_darkens() {
        let (w, h) = (64usize, 48usize);
        let input = vec![1.0f32; w * h];
        let mut out = vec![0.0f32; w * h];
        Blur::new().blur_plane(&input, &mut out, w, h, &SerialExecutor);
        let centre = out.get(24 * w + 32).copied().unwrap_or(0.0);
        assert!((centre - 1.0).abs() < 2e-3, "centre {centre}");
        let corner = out.first().copied().unwrap_or(0.0);
        assert!(
            corner < centre,
            "zero padding must darken the corner: {corner}"
        );
    }

    #[test]
    fn the_strip_width_does_not_change_the_result() {
        // Columns are independent, so a 100-wide plane (one full strip plus a
        // partial one) must equal the same data blurred 50 columns at a time.
        let (w, h) = (100usize, 20usize);
        let input: Vec<f32> = (0..w * h)
            .map(|i| ((i * 7919) % 97) as f32 / 97.0)
            .collect();
        let mut whole = vec![0.0f32; w * h];
        Blur::new().blur_plane(&input, &mut whole, w, h, &SerialExecutor);
        let mut temp = vec![0.0f32; w * h];
        let mut padded = Vec::new();
        for (row_in, row_out) in input.chunks_exact(w).zip(temp.chunks_exact_mut(w)) {
            horizontal_row(row_in, row_out, &mut padded);
        }
        let mut strips = vec![0.0f32; w * h];
        let out = DisjointColumns::new(&mut strips);
        vertical_strip(&temp, &out, w, h, 0, 50);
        vertical_strip(&temp, &out, w, h, 50, 50);
        assert_eq!(whole, strips);
    }

    #[test]
    fn the_lane_grouping_does_not_change_the_result() {
        // Each row's recursion depends on that row alone and the lane
        // arithmetic is exactly `step`'s, so grouping rows into lanes cannot
        // change a value — including every remainder count a band height not
        // divisible by `ROW_LANES` can produce. Both input kinds.
        let w = 53usize;
        for h in 1..=(2 * ROW_LANES + 1) {
            let a: Vec<f32> = (0..w * h).map(|i| ((i * 131) % 89) as f32 / 89.0).collect();
            let b: Vec<f32> = (0..w * h).map(|i| ((i * 37) % 71) as f32 / 71.0).collect();
            for input in [BlurInput::Plane(&a), BlurInput::Product(&a, &b)] {
                let mut grouped = vec![0.0f32; w * h];
                horizontal_band(input, &mut grouped, w);
                let mut scalar = vec![0.0f32; w * h];
                let mut padded = Vec::new();
                match input {
                    BlurInput::Plane(rows) => {
                        for (ri, ro) in rows.chunks_exact(w).zip(scalar.chunks_exact_mut(w)) {
                            horizontal_row(ri, ro, &mut padded);
                        }
                    }
                    BlurInput::Product(ra, rb) => {
                        for ((ia, ib), ro) in rows_pair(ra, rb, w).zip(scalar.chunks_exact_mut(w)) {
                            horizontal_product_row(ia, ib, ro, &mut padded);
                        }
                    }
                }
                assert_eq!(grouped, scalar, "height {h}");
            }
        }
    }

    fn rows_pair<'a>(
        a: &'a [f32],
        b: &'a [f32],
        w: usize,
    ) -> impl Iterator<Item = (&'a [f32], &'a [f32])> {
        a.chunks_exact(w).zip(b.chunks_exact(w))
    }

    #[test]
    fn the_executor_does_not_change_the_result() {
        let (w, h) = (37usize, 130usize);
        let input: Vec<f32> = (0..w * h)
            .map(|i| ((i * 31) % 101) as f32 / 101.0)
            .collect();
        let mut serial = vec![0.0f32; w * h];
        Blur::new().blur_plane(&input, &mut serial, w, h, &SerialExecutor);
        let mut threaded = vec![0.0f32; w * h];
        Blur::new().blur_plane(
            &input,
            &mut threaded,
            w,
            h,
            &ScopedThreadExecutor { workers: 3 },
        );
        assert_eq!(serial, threaded);
    }

    #[test]
    fn product_blur_matches_a_materialised_product_plane() {
        let (w, h) = (73usize, 51usize);
        let a: Vec<f32> = (0..w * h)
            .map(|i| ((i * 31) % 101) as f32 / 101.0)
            .collect();
        let b: Vec<f32> = (0..w * h)
            .map(|i| ((i * 47 + 3) % 109) as f32 / 109.0)
            .collect();
        let product: Vec<f32> = a.iter().zip(&b).map(|(&a, &b)| a * b).collect();
        let mut materialised = vec![0.0f32; w * h];
        Blur::new().blur_plane(&product, &mut materialised, w, h, &SerialExecutor);
        let mut fused = vec![0.0f32; w * h];
        Blur::new().blur_product_plane(&a, &b, &mut fused, w, h, &SerialExecutor);
        assert_eq!(fused, materialised);
    }
}

#[cfg(test)]
mod row_lanes_bench {
    use super::*;

    /// Temporary E1-d microbench: `cargo test -p jpxl-perceptual --release
    /// row_lanes_bench -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn horizontal_band_timing() {
        let width = 4000usize;
        let rows = 512usize;
        let input: Vec<f32> = (0..width * rows)
            .map(|i| ((i * 2_654_435_761usize) & 0xffff) as f32 / 65_535.0)
            .collect();
        let mut out = vec![0.0f32; width * rows];
        // warm-up
        horizontal_band(BlurInput::Plane(&input), &mut out, width);
        let mut best = u128::MAX;
        for _ in 0..30 {
            let t = std::time::Instant::now();
            horizontal_band(BlurInput::Plane(&input), &mut out, width);
            best = best.min(t.elapsed().as_nanos());
        }
        let mut best_prod = u128::MAX;
        for _ in 0..30 {
            let t = std::time::Instant::now();
            horizontal_band(BlurInput::Product(&input, &input), &mut out, width);
            best_prod = best_prod.min(t.elapsed().as_nanos());
        }
        println!(
            "ROW_LANES={} plane={:.3}ns/sample product={:.3}ns/sample",
            ROW_LANES,
            best as f64 / (width * rows) as f64,
            best_prod as f64 / (width * rows) as f64,
        );
    }
}
