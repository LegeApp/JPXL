//! Annex J restoration-filter kernels (18181-1 J.3 Gabor-like transform and
//! J.4 edge-preserving filter), direction-neutral.
//!
//! These are the leaf kernels both trees need: the decoder to reconstruct a
//! frame, and the encoder's plan renderer to predict what the decoder will
//! reconstruct from a candidate plan. Each tree keeps its own orchestration —
//! which planes, which sigma field, in what order — and both are checked
//! against the external decoders, so a kernel bug here cannot be accepted by
//! one side alone.
//!
//! The four J.4 readings that the printed clause leaves open are pinned as
//! named constants, with the values the end-to-end flip-point probe of
//! 2026-08-03 confirmed (see `docs/experiments/2026-08-03-vardct-flip-point-probe.md`):
//! steps are selected by J.4.1's explicit conditions; the border multiplier is
//! evaluated at the reference pixel; the skip test reads the 8x8 block's own
//! sigma; and a step measures distances on its own input.

/// Dimensions of an `f32` sample plane stored in raster order.
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
}

/// `usize` to `i64` without a lint-triggering `as` cast.
fn as_i64(v: usize) -> i64 {
    i64::try_from(v).unwrap_or(i64::MAX)
}

/// 18181-1 5.2 `Mirror1D`: folds an out-of-range coordinate back into
/// `[0, size)` by reflecting about the half-sample outside each edge, as
/// many times as a narrow plane needs.
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

/// Reads `plane` at `(x, y)`, mirroring per 5.2 outside the plane.
#[must_use]
pub fn sample_mirrored(plane: &[f32], dims: PlaneDims, x: i64, y: i64) -> f32 {
    let px = mirror1d(x, dims.width);
    let py = mirror1d(y, dims.height);
    plane
        .get(py.saturating_mul(dims.width).saturating_add(px))
        .copied()
        .unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// J.3 — Gabor-like transform
// ---------------------------------------------------------------------------

/// Table J.1 default first-ring Gabor weight (`gab_*_weight1`).
pub const DEFAULT_GAB_WEIGHT1: f32 = 0.115_169_525;
/// Table J.1 default second-ring Gabor weight (`gab_*_weight2`).
pub const DEFAULT_GAB_WEIGHT2: f32 = 0.061_248_592;

/// The normalized J.3 kernel for one channel.
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
    /// `None` when `1 + 4 w1 + 4 w2` is zero or not finite, which would
    /// divide by zero in the clause's rescale.
    #[must_use]
    pub fn new(weight1: f32, weight2: f32) -> Option<Self> {
        let unnormalized_sum = 4.0f32.mul_add(weight2, 4.0f32.mul_add(weight1, 1.0));
        if !unnormalized_sum.is_finite() || unnormalized_sum == 0.0 {
            return None;
        }
        let scale = 1.0 / unnormalized_sum;
        Some(Self {
            centre: scale,
            edge: weight1 * scale,
            corner: weight2 * scale,
        })
    }

    /// The Table J.1 default kernel.
    #[must_use]
    pub fn defaults() -> Self {
        // The defaults sum to a finite positive value, so `new` cannot fail.
        Self::new(DEFAULT_GAB_WEIGHT1, DEFAULT_GAB_WEIGHT2).unwrap_or(Self {
            centre: 1.0,
            edge: 0.0,
            corner: 0.0,
        })
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

/// Applies J.3 to one plane, writing into `output`. Every output sample reads
/// the unfiltered neighbourhood; both slices must hold `dims.len()` samples
/// (anything else leaves `output` untouched).
pub fn gaborish_into(input: &[f32], output: &mut [f32], dims: PlaneDims, kernel: &GaborKernel) {
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::has_fma() {
        // SAFETY: `gaborish_into_fma` only requires that the host support
        // AVX2 and FMA, which `has_fma` has just confirmed.
        #[allow(unsafe_code)]
        unsafe {
            gaborish_into_fma(input, output, dims, kernel);
        }
        return;
    }
    gaborish_into_impl(input, output, dims, kernel);
}

/// [`gaborish_into`] compiled with hardware fused multiply-add.
///
/// Calling it is `unsafe` unless the host supports AVX2 and FMA (see
/// [`crate::cpu::has_fma`]); that is the whole contract.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn gaborish_into_fma(input: &[f32], output: &mut [f32], dims: PlaneDims, kernel: &GaborKernel) {
    gaborish_into_impl(input, output, dims, kernel);
}

#[inline(always)]
fn gaborish_into_impl(input: &[f32], output: &mut [f32], dims: PlaneDims, kernel: &GaborKernel) {
    if input.len() != dims.len() || output.len() != dims.len() {
        return;
    }
    for y in 0..dims.height {
        let yi = as_i64(y);
        for x in 0..dims.width {
            let xi = as_i64(x);
            let mut acc = 0.0f32;
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let w = kernel.weight_at(dx, dy);
                    acc = w.mul_add(sample_mirrored(input, dims, xi + dx, yi + dy), acc);
                }
            }
            if let Some(slot) = output.get_mut(y.saturating_mul(dims.width).saturating_add(x)) {
                *slot = acc;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// J.4 — edge-preserving filter
// ---------------------------------------------------------------------------

/// Side of the block grid that sigma, `Sharpness` and the border predicate
/// are defined on.
pub const EPF_BLOCK_DIM: usize = 8;

const BLOCK_DIM_I: i64 = 8;

/// J.4.3: a block whose sigma is below this is left untouched by every step.
pub const EPF_SIGMA_SKIP_THRESHOLD: f32 = 0.3;

/// J.4.3: the constant that scales the per-step sigma scales.
pub const EPF_STEP_MULTIPLIER_BASE: f32 = 1.65;

/// Table J.1 default `epf_channel_scale`.
pub const DEFAULT_EPF_CHANNEL_SCALE: [f32; 3] = [40.0, 5.0, 3.5];

/// Table J.1 default `epf_sharp_lut`: `{0, 1/7, 2/7, ..., 6/7, 1}`.
pub const DEFAULT_EPF_SHARP_LUT: [f32; 8] = [
    0.0,
    1.0 / 7.0,
    2.0 / 7.0,
    3.0 / 7.0,
    4.0 / 7.0,
    5.0 / 7.0,
    6.0 / 7.0,
    1.0,
];

/// Edge-preserving-filter parameters (18181-1 J.1, J.4), iteration count
/// excluded: the caller says how many steps to run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpfParams {
    /// Sharpness lookup table.
    pub sharp_lut: [f32; 8],
    /// Per-channel weight scaling.
    pub channel_scale: [f32; 3],
    /// Multiplier tying sigma to the quantizer.
    pub quant_mul: f32,
    /// Sigma scale for step 0.
    pub pass0_sigma_scale: f32,
    /// Sigma scale for step 2.
    pub pass2_sigma_scale: f32,
    /// Distance multiplier on block borders.
    pub border_sad_mul: f32,
}

impl Default for EpfParams {
    /// The Table J.1 defaults.
    fn default() -> Self {
        Self {
            sharp_lut: DEFAULT_EPF_SHARP_LUT,
            channel_scale: DEFAULT_EPF_CHANNEL_SCALE,
            quant_mul: 0.46,
            pass0_sigma_scale: 0.9,
            pass2_sigma_scale: 6.5,
            border_sad_mul: 2.0 / 3.0,
        }
    }
}

/// J.4.2 `coords`: the five-pixel cross a distance is summed over.
const CROSS_COORDS: [(i64, i64); 5] = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)];

/// J.4.4 step-0 kernel: the reference pixel plus the twelve pixels at L1
/// distance at most 2.
const STEP0_KERNEL_COORDS: [(i64, i64); 13] = [
    (0, 0),
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 1),
    (-1, 1),
    (-1, -1),
    (-2, 0),
    (2, 0),
    (0, 2),
    (0, -2),
];

/// One of the up-to-three J.4 filter steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpfStep {
    /// 13-tap step, cross distances. Runs only when `epf_iters == 3`.
    Step0,
    /// 5-tap step, cross distances. Runs whenever the filter runs.
    Step1,
    /// 5-tap step, single-pixel distances. Runs when `epf_iters >= 2`.
    Step2,
}

impl EpfStep {
    /// The step's J.4.4 kernel.
    #[must_use]
    pub const fn kernel(self) -> &'static [(i64, i64)] {
        match self {
            Self::Step0 => &STEP0_KERNEL_COORDS,
            Self::Step1 | Self::Step2 => &CROSS_COORDS,
        }
    }

    /// J.4.3 `step_multiplier[step]`.
    #[must_use]
    pub fn step_multiplier(self, params: &EpfParams) -> f32 {
        let scale = match self {
            Self::Step0 => params.pass0_sigma_scale,
            Self::Step1 => 1.0,
            Self::Step2 => params.pass2_sigma_scale,
        };
        EPF_STEP_MULTIPLIER_BASE * scale
    }
}

/// The steps J.4.1's explicit conditions select for `epf_iters`, in
/// execution order. Values above 3 fold onto 3.
#[must_use]
pub const fn epf_steps(iters: u8) -> &'static [EpfStep] {
    use EpfStep::{Step0, Step1, Step2};
    match iters {
        0 => &[],
        1 => &[Step1],
        2 => &[Step1, Step2],
        _ => &[Step0, Step1, Step2],
    }
}

/// The 8x8 block grid covering a plane: `(ceil(w/8), ceil(h/8))`.
#[must_use]
pub const fn block_grid(dims: PlaneDims) -> (usize, usize) {
    (
        dims.width.div_ceil(EPF_BLOCK_DIM),
        dims.height.div_ceil(EPF_BLOCK_DIM),
    )
}

/// J.4.3's sigma for a VarDCT block: `mul * epf_quant_mul * epf_sharp_lut[s]`.
///
/// `sharpness` above 7 reads the last table entry.
#[must_use]
pub fn vardct_sigma(quantization_width: f32, sharpness: u8, params: &EpfParams) -> f32 {
    let lut = params
        .sharp_lut
        .get(usize::from(sharpness))
        .or(params.sharp_lut.last())
        .copied()
        .unwrap_or(0.0);
    quantization_width * params.quant_mul * lut
}

/// J.4.3: whether a coordinate pair sits on a block edge (either coordinate
/// is `0` or `7` modulo 8, Euclidean remainder for negative taps).
#[must_use]
pub const fn at_block_border(x: i64, y: i64) -> bool {
    let rx = x.rem_euclid(BLOCK_DIM_I);
    let ry = y.rem_euclid(BLOCK_DIM_I);
    rx == 0 || rx == BLOCK_DIM_I - 1 || ry == 0 || ry == BLOCK_DIM_I - 1
}

/// J.4.3 `Weight()`. A zero distance returns exactly 1 whatever the sigma.
#[must_use]
pub fn epf_weight(
    distance: f32,
    sigma: f32,
    step: EpfStep,
    at_border: bool,
    params: &EpfParams,
) -> f32 {
    epf_weight_with_inv_sigma(
        distance,
        epf_inv_sigma(sigma, step, params),
        at_border,
        params,
    )
}

/// The reciprocal-sigma factor [`epf_weight`] applies, which depends only on
/// the step, the parameters, and the block's sigma — never on the pixel.
///
/// Split out so the row filter can compute it once per 8×8 block instead of
/// once per tap per pixel. The filter evaluates a dozen taps at every pixel,
/// so the inlined form issued that many identical `divss`es per pixel; the
/// divide was visible in the kernel's profile.
#[must_use]
#[inline]
pub fn epf_inv_sigma(sigma: f32, step: EpfStep, params: &EpfParams) -> f32 {
    step.step_multiplier(params) * 4.0 * (1.0 - 0.5f32.sqrt()) / sigma
}

/// [`epf_weight`] with the reciprocal-sigma factor already computed.
///
/// Same arithmetic in the same order on the same values, so hoisting the
/// factor out of a loop cannot change a weight.
#[must_use]
#[inline]
pub fn epf_weight_with_inv_sigma(
    distance: f32,
    inv_sigma: f32,
    at_border: bool,
    params: &EpfParams,
) -> f32 {
    let position_multiplier = if at_border {
        params.border_sad_mul
    } else {
        1.0
    };
    let scaled_distance = position_multiplier * distance;
    if scaled_distance == 0.0 {
        return 1.0;
    }
    let v = scaled_distance.mul_add(-inv_sigma, 1.0);
    if v > 0.0 { v } else { 0.0 }
}

/// J.4.3's per-8x8-block sigma over the [`block_grid`] of the planes being
/// filtered, indexed `by * blocks_x + bx`.
#[derive(Debug, Clone, Copy)]
pub struct SigmaField<'a> {
    sigma: &'a [f32],
    blocks_x: usize,
    blocks_y: usize,
}

impl<'a> SigmaField<'a> {
    /// Wraps a per-block sigma plane; `None` unless it holds exactly
    /// `blocks_x * blocks_y` values.
    #[must_use]
    pub fn new(sigma: &'a [f32], blocks_x: usize, blocks_y: usize) -> Option<Self> {
        (sigma.len() == blocks_x.saturating_mul(blocks_y)).then_some(Self {
            sigma,
            blocks_x,
            blocks_y,
        })
    }

    /// Blocks per row.
    #[must_use]
    pub const fn blocks_x(&self) -> usize {
        self.blocks_x
    }

    /// Rows of blocks.
    #[must_use]
    pub const fn blocks_y(&self) -> usize {
        self.blocks_y
    }

    /// Sigma of block `(bx, by)`.
    #[must_use]
    pub fn sigma_at(&self, bx: usize, by: usize) -> f32 {
        self.sigma
            .get(by.saturating_mul(self.blocks_x).saturating_add(bx))
            .copied()
            .unwrap_or(0.0)
    }
}

/// A plane with a mirrored border of `pad` samples on every side, so a filter
/// tap reads a contiguous row slice instead of mirroring per access.
#[derive(Debug, Clone, PartialEq)]
pub struct PaddedPlane {
    width: usize,
    height: usize,
    pad: usize,
    stride: usize,
    data: Vec<f32>,
}

impl PaddedPlane {
    /// Pads `plane` (`dims`) by `pad` mirrored samples; `None` if the plane's
    /// length disagrees with `dims`.
    #[must_use]
    pub fn new(plane: &[f32], dims: PlaneDims, pad: usize) -> Option<Self> {
        Self::new_in(plane, dims, pad, Vec::new())
    }

    /// [`Self::new`] building into `storage`, so a caller that pads the same
    /// shape repeatedly reuses one allocation. Every padded sample — mirror
    /// margins included — is written below, so stale contents cannot leak
    /// into the result and the output is identical to [`Self::new`]'s.
    #[must_use]
    pub fn new_in(
        plane: &[f32],
        dims: PlaneDims,
        pad: usize,
        mut storage: Vec<f32>,
    ) -> Option<Self> {
        if plane.len() != dims.len() || dims.is_empty() {
            return None;
        }
        let stride = dims.width + 2 * pad;
        let rows = dims.height + 2 * pad;
        // No eager zero-fill: the row loop below writes all `stride` samples
        // of every padded row, so resizing without clearing is exact — and an
        // empty `storage` takes `vec!`'s lazily zeroed allocation rather than
        // paying a memset it does not need.
        if storage.is_empty() {
            storage = vec![0.0f32; stride * rows];
        } else {
            storage.truncate(stride * rows);
            storage.resize(stride * rows, 0.0f32);
        }
        let mut data = storage;
        for py in 0..rows {
            let sy = mirror1d(as_i64(py) - as_i64(pad), dims.height);
            let src = plane.get(sy * dims.width..sy * dims.width + dims.width)?;
            let dst = data.get_mut(py * stride..py * stride + stride)?;
            dst.get_mut(pad..pad + dims.width)?.copy_from_slice(src);
            for px in 0..pad {
                let left = src
                    .get(mirror1d(as_i64(px) - as_i64(pad), dims.width))
                    .copied()?;
                let right = src
                    .get(mirror1d(
                        as_i64(dims.width + pad + px) - as_i64(pad),
                        dims.width,
                    ))
                    .copied()?;
                *dst.get_mut(px)? = left;
                *dst.get_mut(pad + dims.width + px)? = right;
            }
        }
        Some(Self {
            width: dims.width,
            height: dims.height,
            pad,
            stride,
            data,
        })
    }

    /// The padding on each side.
    #[must_use]
    pub const fn pad(&self) -> usize {
        self.pad
    }

    /// The unpadded dimensions.
    #[must_use]
    pub const fn dims(&self) -> PlaneDims {
        PlaneDims::new(self.width, self.height)
    }

    /// Image row `y` (which may be in the padding, `-pad..height+pad`) with
    /// `dx` extra leading offset: the slice starts at image column `dx - pad`
    /// and runs to the end of the padded row. Empty outside the padded plane.
    #[must_use]
    pub fn row_from(&self, y: i64, dx: i64) -> &[f32] {
        let py = y + as_i64(self.pad);
        let px = dx + as_i64(self.pad);
        if py < 0 || px < 0 {
            return &[];
        }
        let (Ok(py), Ok(px)) = (usize::try_from(py), usize::try_from(px)) else {
            return &[];
        };
        if py >= self.height + 2 * self.pad || px > self.stride {
            return &[];
        }
        self.data
            .get(py * self.stride + px..py * self.stride + self.stride)
            .unwrap_or(&[])
    }

    /// Consumes the plane, handing its backing storage back for reuse with
    /// [`Self::new_in`].
    #[must_use]
    pub fn into_storage(self) -> Vec<f32> {
        self.data
    }
}

/// Padding the J.4 kernels need: the widest tap reach (2) plus the cross (1).
pub const EPF_PAD: usize = 3;

/// Runs one J.4 step over rows `rows` of the three padded colour planes,
/// writing those rows (each `dims.width` long, contiguous) into `out`.
///
/// Distances and weights are accumulated in exactly the per-pixel order of
/// the clause (component outer, cross inner; taps in kernel order), so
/// banding the rows across workers cannot change a sample.
///
/// `None` if the planes are not padded by at least [`EPF_PAD`], a row is out
/// of range, or `out` does not hold the rows.
#[must_use]
pub fn epf_step_rows(
    step: EpfStep,
    input: &[PaddedPlane; 3],
    params: &EpfParams,
    sigma: &SigmaField<'_>,
    rows: core::ops::Range<usize>,
    out: &mut [&mut [f32]; 3],
) -> Option<()> {
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::has_fma() {
        // SAFETY: `epf_step_rows_fma` only requires that the host support
        // AVX2 and FMA, which `has_fma` has just confirmed.
        #[allow(unsafe_code)]
        return unsafe { epf_step_rows_fma(step, input, params, sigma, rows, out) };
    }
    epf_step_rows_impl(step, input, params, sigma, rows, out)
}

/// [`epf_step_rows`] compiled with hardware fused multiply-add.
///
/// Calling it is `unsafe` unless the host supports AVX2 and FMA (see
/// [`crate::cpu::has_fma`]); that is the whole contract.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn epf_step_rows_fma(
    step: EpfStep,
    input: &[PaddedPlane; 3],
    params: &EpfParams,
    sigma: &SigmaField<'_>,
    rows: core::ops::Range<usize>,
    out: &mut [&mut [f32]; 3],
) -> Option<()> {
    epf_step_rows_impl(step, input, params, sigma, rows, out)
}

#[inline(always)]
fn epf_step_rows_impl(
    step: EpfStep,
    input: &[PaddedPlane; 3],
    params: &EpfParams,
    sigma: &SigmaField<'_>,
    rows: core::ops::Range<usize>,
    out: &mut [&mut [f32]; 3],
) -> Option<()> {
    let dims = input[0].dims();
    let width = dims.width;
    if input.iter().any(|p| p.pad() < EPF_PAD || p.dims() != dims) {
        return None;
    }
    if rows.end > dims.height || out.iter().any(|o| o.len() != width * rows.len()) {
        return None;
    }
    if (sigma.blocks_x(), sigma.blocks_y()) != block_grid(dims) {
        return None;
    }
    let taps = step.kernel();
    let cross: &[(i64, i64)] = if step == EpfStep::Step2 {
        &[(0, 0)]
    } else {
        &CROSS_COORDS
    };
    let mut dist: Vec<Vec<f32>> = vec![vec![0.0f32; width]; taps.len()];

    for (row_index, y) in rows.enumerate() {
        let yi = as_i64(y);
        // J.4.2: the distance of every tap, vectorised along the row.
        for (d, &(kx, ky)) in dist.iter_mut().zip(taps) {
            d.fill(0.0);
            for (c, plane) in input.iter().enumerate() {
                let scale = params.channel_scale.get(c).copied().unwrap_or(0.0);
                for &(ix, iy) in cross {
                    let a = plane.row_from(yi + iy, ix);
                    let b = plane.row_from(yi + ky + iy, kx + ix);
                    for ((acc, &a), &b) in d.iter_mut().zip(a).zip(b) {
                        *acc = (a - b).abs().mul_add(scale, *acc);
                    }
                }
            }
        }
        // J.4.3/J.4.4: weights and the normalised average.
        let by = y / EPF_BLOCK_DIM;
        let tap_rows: Vec<[&[f32]; 3]> = taps
            .iter()
            .map(|&(kx, ky)| {
                [
                    input[0].row_from(yi + ky, kx),
                    input[1].row_from(yi + ky, kx),
                    input[2].row_from(yi + ky, kx),
                ]
            })
            .collect();
        let centre: [&[f32]; 3] = [
            input[0].row_from(yi, 0),
            input[1].row_from(yi, 0),
            input[2].row_from(yi, 0),
        ];
        // Walk the row a block at a time. Everything the weight needs besides
        // the pixel's own distance is constant across an 8x8 block: the
        // sigma, the skip decision, and the reciprocal-sigma factor whose
        // divide used to be re-issued for every tap of every pixel. The
        // border test splits too — `at_block_border` is true for the whole
        // row on the block's first and last row, and otherwise only at the
        // block's first and last column.
        let ry = yi.rem_euclid(BLOCK_DIM_I);
        let border_row = ry == 0 || ry == BLOCK_DIM_I - 1;
        for x0 in (0..width).step_by(EPF_BLOCK_DIM) {
            let x_end = (x0 + EPF_BLOCK_DIM).min(width);
            let block_sigma = sigma.sigma_at(x0 / EPF_BLOCK_DIM, by);
            if block_sigma < EPF_SIGMA_SKIP_THRESHOLD {
                for x in x0..x_end {
                    let idx = row_index * width + x;
                    for (c, plane) in out.iter_mut().enumerate() {
                        if let (Some(v), Some(slot)) =
                            (centre.get(c).and_then(|r| r.get(x)), plane.get_mut(idx))
                        {
                            *slot = *v;
                        }
                    }
                }
                continue;
            }
            // Guarded by the skip test above, so `block_sigma` is at least
            // EPF_SIGMA_SKIP_THRESHOLD and this cannot divide by zero.
            let inv_sigma = epf_inv_sigma(block_sigma, step, params);
            for x in x0..x_end {
                let idx = row_index * width + x;
                let rx = as_i64(x).rem_euclid(BLOCK_DIM_I);
                let at_border = border_row || rx == 0 || rx == BLOCK_DIM_I - 1;
                let mut sum_weights = 0.0f32;
                let mut acc = [0.0f32; 3];
                for (d, rows3) in dist.iter().zip(&tap_rows) {
                    let distance = d.get(x).copied().unwrap_or(0.0);
                    let weight = epf_weight_with_inv_sigma(distance, inv_sigma, at_border, params);
                    sum_weights += weight;
                    for (c, slot) in acc.iter_mut().enumerate() {
                        let v = rows3.get(c).and_then(|r| r.get(x)).copied().unwrap_or(0.0);
                        *slot = v.mul_add(weight, *slot);
                    }
                }
                for (c, plane) in out.iter_mut().enumerate() {
                    if let (Some(v), Some(slot)) = (acc.get(c), plane.get_mut(idx)) {
                        *slot = v / sum_weights;
                    }
                }
            }
        }
    }
    Some(())
}

/// Runs one J.4 step over the three colour planes, returning fresh planes.
///
/// `None` if a plane's length disagrees with `dims` or the sigma field's
/// grid is not [`block_grid`] of `dims`.
#[must_use]
pub fn epf_step(
    step: EpfStep,
    input: [&[f32]; 3],
    dims: PlaneDims,
    params: &EpfParams,
    sigma: &SigmaField<'_>,
) -> Option<[Vec<f32>; 3]> {
    let padded = [
        PaddedPlane::new(input[0], dims, EPF_PAD)?,
        PaddedPlane::new(input[1], dims, EPF_PAD)?,
        PaddedPlane::new(input[2], dims, EPF_PAD)?,
    ];
    let mut out = [
        vec![0.0f32; dims.len()],
        vec![0.0f32; dims.len()],
        vec![0.0f32; dims.len()],
    ];
    {
        let [o0, o1, o2] = &mut out;
        let mut slices = [o0.as_mut_slice(), o1.as_mut_slice(), o2.as_mut_slice()];
        epf_step_rows(step, &padded, params, sigma, 0..dims.height, &mut slices)?;
    }
    Some(out)
}

/// Applies the whole J.4 filter for `iters` iterations to `[X, Y, B]`,
/// returning fresh planes (copies of the input when `iters == 0`).
///
/// `None` as [`epf_step`].
#[must_use]
pub fn epf(
    input: [&[f32]; 3],
    dims: PlaneDims,
    iters: u8,
    params: &EpfParams,
    sigma: &SigmaField<'_>,
) -> Option<[Vec<f32>; 3]> {
    if input.iter().any(|plane| plane.len() != dims.len()) {
        return None;
    }
    let mut current: [Vec<f32>; 3] = input.map(<[f32]>::to_vec);
    for step in epf_steps(iters).iter().copied() {
        let refs: [&[f32]; 3] = current.each_ref().map(Vec::as_slice);
        current = epf_step(step, refs, dims, params, sigma)?;
    }
    Some(current)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mirroring_folds_back_inside_even_narrow_planes() {
        assert_eq!(mirror1d(-1, 5), 0);
        assert_eq!(mirror1d(5, 5), 4);
        assert_eq!(mirror1d(-3, 2), 1);
        assert_eq!(mirror1d(7, 3), 1);
        assert_eq!(mirror1d(0, 0), 0);
    }

    #[test]
    fn the_default_gabor_kernel_sums_to_one() {
        let k = GaborKernel::defaults();
        let sum = 4.0f32.mul_add(k.corner, 4.0f32.mul_add(k.edge, k.centre));
        assert!((sum - 1.0).abs() < 1e-6, "{sum}");
        assert!(GaborKernel::new(-0.25, 0.0).is_none());
    }

    #[test]
    fn a_constant_plane_is_a_fixed_point_of_both_filters() {
        let dims = PlaneDims::new(24, 16);
        let plane = vec![0.25f32; dims.len()];
        let mut out = vec![0.0f32; dims.len()];
        gaborish_into(&plane, &mut out, dims, &GaborKernel::defaults());
        assert!(out.iter().all(|&v| (v - 0.25).abs() < 1e-6));

        let (bx, by) = block_grid(dims);
        let sigma = vec![1.0f32; bx * by];
        let field = SigmaField::new(&sigma, bx, by).expect("grid matches");
        let filtered = epf(
            [&plane, &plane, &plane],
            dims,
            3,
            &EpfParams::default(),
            &field,
        )
        .expect("shapes agree");
        assert!(filtered.iter().flatten().all(|&v| (v - 0.25).abs() < 1e-6));
    }

    #[test]
    fn a_low_sigma_block_is_passed_through_untouched() {
        let dims = PlaneDims::new(8, 8);
        let plane: Vec<f32> = (0..64).map(|i| (i % 7) as f32 / 7.0).collect();
        let sigma = [0.1f32];
        let field = SigmaField::new(&sigma, 1, 1).expect("grid matches");
        let filtered = epf(
            [&plane, &plane, &plane],
            dims,
            2,
            &EpfParams::default(),
            &field,
        )
        .expect("shapes agree");
        assert_eq!(filtered[1], plane);
    }

    #[test]
    fn a_padded_plane_mirrors_its_border() {
        let dims = PlaneDims::new(4, 3);
        let plane: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let padded = PaddedPlane::new(&plane, dims, 2).expect("pads");
        // Row -1 mirrors row 0; column -1 mirrors column 0.
        assert_eq!(padded.row_from(-1, 0).first().copied(), Some(0.0));
        assert_eq!(padded.row_from(1, -1).first().copied(), Some(4.0));
        assert_eq!(padded.row_from(1, 0).first().copied(), Some(4.0));
        assert_eq!(padded.row_from(3, 3).first().copied(), Some(11.0));
        assert!(padded.row_from(-3, 0).is_empty());
    }

    #[test]
    fn the_explicit_step_conditions_run_exactly_iters_steps() {
        assert_eq!(epf_steps(0).len(), 0);
        assert_eq!(epf_steps(1), &[EpfStep::Step1]);
        assert_eq!(epf_steps(2), &[EpfStep::Step1, EpfStep::Step2]);
        assert_eq!(
            epf_steps(3),
            &[EpfStep::Step0, EpfStep::Step1, EpfStep::Step2]
        );
    }
}
