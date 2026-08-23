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
//! walks column strips on the calling thread.

use crate::bands::{BAND_ROWS, band_of, mutable_bands};
use crate::executor::BandExecutor;

/// Truncation radius `N = round(3.2795 σ + 0.2546)` for σ = 1.5.
const RADIUS: isize = 5;

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

#[derive(Clone, Copy)]
enum BlurInput<'a> {
    Plane(&'a [f32]),
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
        self.temp.clear();
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
            let mut padded = Vec::new();
            match input {
                BlurInput::Plane(input) => {
                    let in_band = band_of(input, index, band_len);
                    for (row_in, row_out) in in_band
                        .chunks_exact(width)
                        .zip(out_band.chunks_exact_mut(width))
                    {
                        horizontal_row(row_in, row_out, &mut padded);
                    }
                }
                BlurInput::Product(a, b) => {
                    let a_band = band_of(a, index, band_len);
                    let b_band = band_of(b, index, band_len);
                    for ((a_row, b_row), row_out) in a_band
                        .chunks_exact(width)
                        .zip(b_band.chunks_exact(width))
                        .zip(out_band.chunks_exact_mut(width))
                    {
                        horizontal_product_row(a_row, b_row, row_out, &mut padded);
                    }
                }
            }
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
const STRIP: usize = 64;

/// Vertical recursive pass over every column, strips in parallel, then one
/// ordered scatter into the row-major output.
fn vertical_pass(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    executor: &dyn BandExecutor,
) {
    let strips = width.div_ceil(STRIP);
    let buffers: Vec<std::sync::Mutex<Vec<f32>>> = (0..strips)
        .map(|_| std::sync::Mutex::new(Vec::new()))
        .collect();
    executor.run(strips, &|index| {
        let x0 = index * STRIP;
        let cols = (width - x0).min(STRIP);
        let mut buffer = vec![0.0f32; cols * height];
        vertical_strip(input, &mut buffer, width, height, x0, cols);
        if let Some(slot) = buffers.get(index)
            && let Ok(mut guard) = slot.lock()
        {
            *guard = buffer;
        }
    });
    for (index, slot) in buffers.into_iter().enumerate() {
        let buffer = slot.into_inner().unwrap_or_default();
        let x0 = index * STRIP;
        let cols = (width - x0).min(STRIP);
        for (y, src) in buffer.chunks_exact(cols).enumerate() {
            if let Some(dst) = output.get_mut(y * width + x0..y * width + x0 + cols) {
                dst.copy_from_slice(src);
            }
        }
    }
}

/// The two-pole recursion state of one strip, held per pole across the columns
/// (structure-of-arrays). Each pole's per-column update is then an independent
/// lane, so the compiler vectorises the column loop. `prev[k]`/`prev2[k]` are
/// the two previous outputs of pole `k` for every column.
struct StripState {
    prev: [Vec<f64>; 3],
    prev2: [Vec<f64>; 3],
    /// `f64::from(top + bottom)` for the current row, one per column.
    sum: Vec<f64>,
    /// The summed three-pole output for the current row, one per column.
    acc: Vec<f64>,
}

impl StripState {
    fn new(cols: usize) -> Self {
        Self {
            prev: [vec![0.0; cols], vec![0.0; cols], vec![0.0; cols]],
            prev2: [vec![0.0; cols], vec![0.0; cols], vec![0.0; cols]],
            sum: vec![0.0; cols],
            acc: vec![0.0; cols],
        }
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
fn vertical_row(top: &[f32], bottom: &[f32], state: &mut StripState, out: Option<&mut [f32]>) {
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

/// Vertical pass over columns `x0 .. x0 + cols` of `input`, writing the
/// strip row-major (`cols` per row) into `strip_out`. Dispatched to an AVX2
/// build where the host supports it.
fn vertical_strip(
    input: &[f32],
    strip_out: &mut [f32],
    width: usize,
    height: usize,
    x0: usize,
    cols: usize,
) {
    #[cfg(target_arch = "x86_64")]
    if jpxl_core::cpu::has_avx2() {
        // SAFETY: `vertical_strip_avx2` only requires that the host support
        // AVX2, which `has_avx2` has just confirmed.
        #[allow(unsafe_code)]
        unsafe {
            vertical_strip_avx2(input, strip_out, width, height, x0, cols);
        }
        return;
    }
    vertical_strip_impl(input, strip_out, width, height, x0, cols);
}

/// [`vertical_strip`] compiled for AVX2.
///
/// Calling it is `unsafe` unless the host supports AVX2 (see
/// [`jpxl_core::cpu::has_avx2`]); that is the whole contract.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn vertical_strip_avx2(
    input: &[f32],
    strip_out: &mut [f32],
    width: usize,
    height: usize,
    x0: usize,
    cols: usize,
) {
    vertical_strip_impl(input, strip_out, width, height, x0, cols);
}

#[inline(always)]
fn vertical_strip_impl(
    input: &[f32],
    strip_out: &mut [f32],
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
        let out_row = usize::try_from(n)
            .ok()
            .and_then(|n| strip_out.get_mut(n * cols..n * cols + cols));
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
        let mut left = vec![0.0f32; 50 * h];
        let mut right = vec![0.0f32; 50 * h];
        vertical_strip(&temp, &mut left, w, h, 0, 50);
        vertical_strip(&temp, &mut right, w, h, 50, 50);
        for y in 0..h {
            strips[y * w..y * w + 50].copy_from_slice(&left[y * 50..y * 50 + 50]);
            strips[y * w + 50..y * w + 100].copy_from_slice(&right[y * 50..y * 50 + 50]);
        }
        assert_eq!(whole, strips);
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
