//! The edge-preserving filter (18181-1 J.4).
//!
//! J.4.1: when `restoration_filter.epf_iters > 0` the frame passes through up
//! to three filter steps, each step's output feeding the next. Every step is a
//! normalized weighted average over a small kernel; the weights fall off with
//! an L1 distance between the reference pixel's neighbourhood and the tap's
//! neighbourhood (J.4.2), scaled by a per-8x8-block `sigma` (J.4.3), and the
//! average itself is J.4.4.
//!
//! ```text
//! step 0   13 taps: the reference pixel and the twelve pixels at L1 <= 2
//!          distance: two five-pixel crosses (J.4.2 DistanceStep0and1)
//! step 1    5 taps: the reference pixel and its four edge neighbours
//!          distance: two five-pixel crosses
//! step 2    5 taps: the reference pixel and its four edge neighbours
//!          distance: the two pixels alone (J.4.2 DistanceStep2)
//! ```
//!
//! Out-of-frame accesses mirror per 5.2
//! ([`mirror1d`](super::gaborish::mirror1d)), never clamp.
//!
//! Because the reference pixel is always a tap and its distance to itself is
//! zero, its weight is exactly 1 (see [`epf_weight`]); `sum_weights >= 1`
//! always, so J.4.4's division can never blow up on a flat region.
//!
//! # Flip points
//!
//! Four readings of J.4 are underdetermined by the printed clause. Each is a
//! named `const bool` so a probe pass can flip it in one place once end-to-end
//! VarDCT decode exists; the chosen value is the reading best supported by the
//! clause text, argued at each constant. See
//! `docs/experiments/2026-08-03-epf-flip-points.md`.
//!
//! * [`EPF_STEPS_FROM_EXPLICIT_CONDITIONS`] — which steps `epf_iters` selects.
//! * [`EPF_BORDER_SAD_AT_REFERENCE_PIXEL`] — where the `epf_border_sad_mul`
//!   predicate is evaluated.
//! * [`EPF_SKIP_IS_PER_VARBLOCK`] — granularity of the `sigma < 0.3` skip.
//! * [`EPF_DISTANCE_USES_STEP_INPUT`] — whether J.4.2's `sample()` and
//!   J.4.4's `input()` name one buffer or two.
//!
//! # Scope
//!
//! A pure function of (planes, parameters, sigma field). Computing the sigma
//! field from `mul` (I.5.3) and `Sharpness` (G.2.4) belongs to the VarDCT
//! metadata sub-slice; [`vardct_sigma`] is the J.4.3 formula it should call,
//! and [`SigmaField`] is the shape it should deliver.

use crate::frame::error::{FrameError, Result};
use crate::frame::gaborish::{PlaneDims, interior_pixel, sample_direct, sample_mirrored};
use crate::frame::restoration::EpfParams;

/// Side of the block grid that `sigma`, `Sharpness` and the J.4.3 border
/// predicate are all defined on (18181-1 G.2.4, I.5.3).
pub const BLOCK_DIM: usize = 8;

const BLOCK_DIM_I: i64 = 8;

/// J.4.3: a block whose sigma is below this is left untouched by every step.
pub const EPF_SIGMA_SKIP_THRESHOLD: f32 = 0.3;

/// J.4.3: the constant that scales the per-step sigma scales.
pub const EPF_STEP_MULTIPLIER_BASE: f32 = 1.65;

/// Which steps `epf_iters` selects.
///
/// J.4.1 states three conditions: step 0 runs iff `epf_iters == 3`, step 1
/// runs whenever the filter runs at all, and step 2 runs iff
/// `epf_iters >= 2`. Read literally that gives
///
/// ```text
/// iters 1 -> {1}      iters 2 -> {1, 2}      iters 3 -> {0, 1, 2}
/// ```
///
/// The competing reading is driven by the field *name*: `epf_iters` sounds
/// like "run the first N steps", giving `{0}`, `{0, 1}`, `{0, 1, 2}`.
///
/// The explicit conditions win, and not only because they are what the clause
/// actually prints: they already satisfy the name's intuition, because the
/// literal mapping runs exactly `epf_iters` steps for every value of the
/// field. The prefix reading would additionally have to contradict "the
/// second step is always done". Set this to `false` for the prefix reading.
/// **PROBED-CONFIRMED end to end (2026-08-03, slice 8F).** Flipping it takes
/// fixture 52 from peak 1.4e-6 to 5.8e-3 and fixture 56 from 5.4e-5 to
/// 1.2e-2, and pushes conformance case `grayscale` past its RMSE class.
/// See `docs/experiments/2026-08-03-vardct-flip-point-probe.md`.
pub const EPF_STEPS_FROM_EXPLICIT_CONDITIONS: bool = true;

/// Where J.4.3's `epf_border_sad_mul` predicate is evaluated.
///
/// The clause guards the multiplier on "either coordinate of the reference
/// sample is 0 or 7 UMod 8". Two axes of ambiguity were considered:
///
/// 1. *Frame-absolute versus block-relative coordinates.* These are the same
///    predicate: the 8x8 block grid is aligned to the frame origin, so a
///    frame coordinate's residue mod 8 **is** its position within its block.
///    The axis is not live and needs no constant.
/// 2. *Reference pixel versus tap.* `Weight()` receives only `(distance,
///    sigma)` — it has no way to know which tap it is being called for — and
///    the clause says "the reference sample", not "the neighbouring sample".
///    So the multiplier is constant across the taps of one reference pixel.
///
/// `true` evaluates the predicate at the reference pixel, which is the
/// reading above. `false` evaluates it at each tap's (unmirrored) frame
/// coordinate instead.
/// **PROBED-CONFIRMED end to end (2026-08-03, slice 8F).** Flipping it
/// degrades every filters-on case by roughly 1000x (52: 1.4e-6 -> 2.4e-3;
/// 53: 2.6e-6 -> 5.2e-3; 56: 5.4e-5 -> 3.3e-3) without any case crossing its
/// class threshold — a consistent-direction result on four streams.
/// See `docs/experiments/2026-08-03-vardct-flip-point-probe.md`.
pub const EPF_BORDER_SAD_AT_REFERENCE_PIXEL: bool = true;

/// Granularity of J.4.3's `sigma < 0.3` skip.
///
/// The clause says "if sigma < 0.3 for a given **varblock**, the decoder skips
/// all steps on the pixels of that block", but two sentences earlier it
/// defines sigma at the 8x8 rectangle containing the reference pixel. The two
/// differ only when `Sharpness` varies inside a varblock larger than 8x8,
/// since the other factor (`mul`) is constant per varblock.
///
/// `true` takes the clause's word "varblock" literally: the skip test reads
/// [`SigmaField::varblock_sigma`] when the caller supplies it — the sigma of
/// the varblock covering the block — while the weights still use the block's
/// own sigma. `false` tests the block's own sigma. With no varblock sigma
/// supplied (Modular, where sigma is uniform) the two coincide.
///
/// **REVERSED by 8F's end-to-end probe (2026-08-03): the shipped value is now
/// `false`.** Under `true` the filters-on fixtures 52/53/56 landed at peak
/// 0.0030/0.0111/0.0117 against their reference decodes; under `false` the
/// same three land at 0.0000014/0.0000026/0.000054 — a ~2000x reduction, and
/// the same order of magnitude the *filters-off* fixtures reach. An error that
/// collapses to the float-noise floor when a one-bit reading is flipped is
/// that reading being wrong, not a tolerance being generous. The clause's
/// "for a given varblock" is therefore read as loose phrasing for "for the
/// block", consistent with the sigma definition two sentences earlier, which
/// is stated at the 8x8 rectangle containing the reference pixel.
/// See `docs/experiments/2026-08-03-vardct-flip-point-probe.md`.
pub const EPF_SKIP_IS_PER_VARBLOCK: bool = false;

/// Whether J.4.2's `sample()` and J.4.4's `input()` are the same buffer.
///
/// J.4.4 accumulates `input(x + ix, y + iy, c)`; J.4.2 computes distances from
/// `sample(x, y, c)`. Neither name is defined against the other, and J.4.1's
/// "the filter may reference guide or input pixels" hints at two buffers — but
/// no clause ever constructs a guide buffer or says what it holds, while
/// J.4.1 does say plainly that each step's output is the next step's input.
///
/// The literal single-buffer reading is therefore taken: within a step, both
/// names denote that step's input. `false` selects the two-buffer reading,
/// where every step measures distances against the filter's original input.
/// **PROBED-CONFIRMED end to end (2026-08-03, slice 8F).** Only fixture 53
/// discriminates — it is the one stream whose `epf_iters` runs more than one
/// step, and a single-step filter has one buffer under either reading — but it
/// discriminates by a factor of 1500 (2.6e-6 -> 4.0e-3).
/// See `docs/experiments/2026-08-03-vardct-flip-point-probe.md`.
pub const EPF_DISTANCE_USES_STEP_INPUT: bool = true;

/// J.4.2 `coords`: the five-pixel cross a distance is summed over.
const CROSS_COORDS: [(i64, i64); 5] = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)];

/// J.4.4 step-0 kernel: the reference pixel plus the twelve pixels at L1
/// distance at most 2.
///
/// `latex/part1.tex` corrupts the last entry to `{9, -2}`, which is neither at
/// L1 distance 2 nor symmetric with the rest; `part1.md` prints `{0, -2}`,
/// which restores the symmetry and makes the count match J.4.1's "twelve
/// neighbouring pixels that have an L1 distance of at most 2" (4 at distance 1
/// plus 8 at distance 2). The markdown is used for this entry.
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

/// J.4.4 kernel for steps 1 and 2: the reference pixel and its four edge
/// neighbours.
const CROSS_KERNEL_COORDS: [(i64, i64); 5] = CROSS_COORDS;

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
    /// Index into J.4.3's `step_multiplier` vector.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Step0 => 0,
            Self::Step1 => 1,
            Self::Step2 => 2,
        }
    }

    /// The step's J.4.4 kernel.
    #[must_use]
    pub const fn kernel(self) -> &'static [(i64, i64)] {
        match self {
            Self::Step0 => &STEP0_KERNEL_COORDS,
            Self::Step1 | Self::Step2 => &CROSS_KERNEL_COORDS,
        }
    }

    /// How far any read of this step reaches from the reference pixel.
    ///
    /// The distance reads the five-pixel cross around the reference pixel
    /// and around each tap, so the reach is the kernel's reach plus one —
    /// except step 2, whose distance compares the two pixels alone. A pixel
    /// at least this far inside the frame never mirrors a read; see
    /// [`interior_pixel`], and `margins_cover_every_read_of_every_step`
    /// for the proof against the read pattern.
    fn read_margin(self) -> usize {
        match self {
            // ±2 taps, ±1 cross around each tap.
            Self::Step0 => 3,
            // ±1 taps, ±1 cross around each tap.
            Self::Step1 => 2,
            // ±1 taps, no cross.
            Self::Step2 => 1,
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

/// The steps J.4.1 selects for a given `epf_iters`, in execution order.
///
/// See [`EPF_STEPS_FROM_EXPLICIT_CONDITIONS`] for the reading. `epf_iters` is
/// a `u(2)` field, so values above 3 are unreachable and fold onto 3.
#[must_use]
pub const fn epf_steps(iters: u32) -> &'static [EpfStep] {
    use EpfStep::{Step0, Step1, Step2};
    if EPF_STEPS_FROM_EXPLICIT_CONDITIONS {
        match iters {
            0 => &[],
            1 => &[Step1],
            2 => &[Step1, Step2],
            _ => &[Step0, Step1, Step2],
        }
    } else {
        match iters {
            0 => &[],
            1 => &[Step0],
            2 => &[Step0, Step1],
            _ => &[Step0, Step1, Step2],
        }
    }
}

/// The 8x8 block grid covering a plane: `(ceil(w/8), ceil(h/8))`.
#[must_use]
pub const fn block_grid(dims: PlaneDims) -> (usize, usize) {
    (
        dims.width.div_ceil(BLOCK_DIM),
        dims.height.div_ceil(BLOCK_DIM),
    )
}

/// J.4.3's per-8x8-block sigma, plus the optional per-varblock sigma the skip
/// test uses under [`EPF_SKIP_IS_PER_VARBLOCK`].
///
/// Both slices are indexed `by * blocks_x + bx` over the [`block_grid`] of the
/// planes being filtered.
#[derive(Debug, Clone, Copy)]
pub struct SigmaField<'a> {
    sigma: &'a [f32],
    varblock_sigma: Option<&'a [f32]>,
    blocks_x: usize,
    blocks_y: usize,
}

impl<'a> SigmaField<'a> {
    /// Wraps a per-8x8-block sigma plane.
    ///
    /// # Errors
    ///
    /// [`FrameError::FieldOutOfRange`] if `sigma` does not hold exactly
    /// `blocks_x * blocks_y` values.
    pub fn new(sigma: &'a [f32], blocks_x: usize, blocks_y: usize) -> Result<Self> {
        check_len(sigma.len(), blocks_x.saturating_mul(blocks_y), "sigma")?;
        Ok(Self {
            sigma,
            varblock_sigma: None,
            blocks_x,
            blocks_y,
        })
    }

    /// Attaches the per-varblock sigma used by the skip test: for each 8x8
    /// block, the sigma of the varblock that covers it.
    ///
    /// # Errors
    ///
    /// [`FrameError::FieldOutOfRange`] on a length mismatch.
    pub fn with_varblock_sigma(mut self, varblock_sigma: &'a [f32]) -> Result<Self> {
        check_len(
            varblock_sigma.len(),
            self.blocks_x.saturating_mul(self.blocks_y),
            "varblock sigma",
        )?;
        self.varblock_sigma = Some(varblock_sigma);
        Ok(self)
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

    /// Whether a per-varblock sigma was supplied.
    #[must_use]
    pub const fn has_varblock_sigma(&self) -> bool {
        self.varblock_sigma.is_some()
    }

    /// Sigma used to weight the taps of a pixel in block `(bx, by)`.
    #[must_use]
    pub fn sigma_at(&self, bx: usize, by: usize) -> f32 {
        self.sigma
            .get(by.saturating_mul(self.blocks_x).saturating_add(bx))
            .copied()
            .unwrap_or(0.0)
    }

    /// Sigma the `< 0.3` skip test reads for block `(bx, by)`.
    #[must_use]
    pub fn skip_sigma_at(&self, bx: usize, by: usize) -> f32 {
        match self.varblock_sigma {
            Some(vb) if EPF_SKIP_IS_PER_VARBLOCK => vb
                .get(by.saturating_mul(self.blocks_x).saturating_add(bx))
                .copied()
                .unwrap_or(0.0),
            _ => self.sigma_at(bx, by),
        }
    }
}

fn check_len(got: usize, want: usize, what: &'static str) -> Result<()> {
    if got == want {
        Ok(())
    } else {
        Err(FrameError::out_of_range(
            what,
            "J.4.3",
            u64::try_from(got).unwrap_or(u64::MAX),
        ))
    }
}

/// J.4.3's sigma for a VarDCT block.
///
/// `quantization_width` is `mul` (I.5.3) and `sharpness` is `Sharpness`
/// (G.2.4), both at the 8x8 rectangle containing the reference pixel.
///
/// # Errors
///
/// [`FrameError::FieldOutOfRange`] if `sharpness` is outside the eight-entry
/// `epf_sharp_lut` — a stream-controlled index, so it is rejected rather than
/// clamped.
pub fn vardct_sigma(quantization_width: f32, sharpness: u32, params: &EpfParams) -> Result<f32> {
    let lut = usize::try_from(sharpness)
        .ok()
        .and_then(|i| params.sharp_lut.get(i))
        .copied()
        .ok_or_else(|| FrameError::out_of_range("Sharpness", "J.4.3", u64::from(sharpness)))?;
    Ok(quantization_width * params.quant_mul * lut)
}

/// J.4.3: whether a coordinate pair sits on a block edge, i.e. either
/// coordinate is `0` or `7` modulo 8.
///
/// The modulo is unsigned (`UMod`), which for the negative coordinates a tap
/// can reach is Euclidean remainder.
#[must_use]
pub const fn at_block_border(x: i64, y: i64) -> bool {
    let rx = x.rem_euclid(BLOCK_DIM_I);
    let ry = y.rem_euclid(BLOCK_DIM_I);
    rx == 0 || rx == BLOCK_DIM_I - 1 || ry == 0 || ry == BLOCK_DIM_I - 1
}

/// J.4.3 `Weight()`.
///
/// `at_border` is the `epf_border_sad_mul` predicate, already evaluated (see
/// [`EPF_BORDER_SAD_AT_REFERENCE_PIXEL`]).
///
/// A zero distance returns exactly 1 whatever the sigma. That is the value the
/// formula gives for every finite `inv_sigma`; stating it up front also keeps
/// the reference pixel's own weight exactly 1 for a degenerate `sigma <= 0`,
/// which no in-range block can have (they are skipped below 0.3) but which a
/// direct caller could pass.
#[must_use]
pub fn epf_weight(
    distance: f32,
    sigma: f32,
    step: EpfStep,
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
    // 4 * (1 - sqrt(0.5)) is the clause's constant, written out rather than
    // pre-multiplied so it reads as J.4.3 does.
    let inv_sigma = step.step_multiplier(params) * 4.0 * (1.0 - 0.5f32.sqrt()) / sigma;
    let v = scaled_distance.mul_add(-inv_sigma, 1.0);
    if v > 0.0 { v } else { 0.0 }
}

/// Everything one step of the filter reads.
struct StepCtx<'a> {
    /// J.4.4 `input()`: the buffer the weighted average accumulates.
    input: [&'a [f32]; 3],
    /// J.4.2 `sample()`: the buffer distances are measured on.
    guide: [&'a [f32]; 3],
    dims: PlaneDims,
    params: &'a EpfParams,
    step: EpfStep,
}

impl StepCtx<'_> {
    fn guide_at(&self, c: usize, x: i64, y: i64, interior: bool) -> f32 {
        let plane: &[f32] = self.guide.get(c).copied().unwrap_or_default();
        if interior {
            sample_direct(plane, self.dims, x, y)
        } else {
            sample_mirrored(plane, self.dims, x, y)
        }
    }

    fn input_at(&self, c: usize, x: i64, y: i64, interior: bool) -> f32 {
        let plane: &[f32] = self.input.get(c).copied().unwrap_or_default();
        if interior {
            sample_direct(plane, self.dims, x, y)
        } else {
            sample_mirrored(plane, self.dims, x, y)
        }
    }

    fn channel_scale(&self, c: usize) -> f32 {
        self.params.channel_scale.get(c).copied().unwrap_or(0.0)
    }

    /// J.4.2 `DistanceStep0and1` / `DistanceStep2`, selected by the step.
    ///
    /// `inline(always)`: the single call site is the tap loop, and the
    /// AVX2+FMA build must fuse through this body — an out-of-line call
    /// would keep the baseline's `fmaf` libcalls inside it.
    #[inline(always)]
    fn distance(&self, x: i64, y: i64, cx: i64, cy: i64, interior: bool) -> f32 {
        let mut dist = 0.0f32;
        for c in 0..3 {
            let scale = self.channel_scale(c);
            if self.step == EpfStep::Step2 {
                let d =
                    self.guide_at(c, x, y, interior) - self.guide_at(c, x + cx, y + cy, interior);
                dist = d.abs().mul_add(scale, dist);
            } else {
                for (ix, iy) in CROSS_COORDS {
                    let d = self.guide_at(c, x + ix, y + iy, interior)
                        - self.guide_at(c, x + cx + ix, y + cy + iy, interior);
                    dist = d.abs().mul_add(scale, dist);
                }
            }
        }
        dist
    }
}

/// Runs one J.4 step over the three colour planes, returning fresh planes.
///
/// `input` is the step's input (J.4.4 `input()`); `guide` is what distances
/// are measured on (J.4.2 `sample()`). [`epf`] passes the same planes for both
/// unless [`EPF_DISTANCE_USES_STEP_INPUT`] is flipped.
///
/// The rows are filtered in bands on a `std::thread::scope` pool. Each output
/// sample is written exactly once from read-only inputs, so the threaded run
/// is bit-identical to the serial one — [`epf_step_with_workers`] with 1 and
/// with N workers return the same bytes.
///
/// # Errors
///
/// [`FrameError::FieldOutOfRange`] if any plane's length disagrees with
/// `dims`, or if the sigma field's block grid is not [`block_grid`] of `dims`.
pub fn epf_step(
    step: EpfStep,
    input: [&[f32]; 3],
    guide: [&[f32]; 3],
    dims: PlaneDims,
    params: &EpfParams,
    sigma: &SigmaField<'_>,
) -> Result<[Vec<f32>; 3]> {
    epf_step_with_workers(
        step,
        input,
        guide,
        dims,
        params,
        sigma,
        crate::parallel::worker_count(dims.height, crate::parallel::MIN_ROWS_PER_WORKER),
    )
}

/// [`epf_step`] with an explicit worker count.
///
/// `1` runs the serial loop inline; anything larger bands the rows. The
/// parameter exists so tests can prove worker-count independence; callers
/// want [`epf_step`].
///
/// # Errors
///
/// As [`epf_step`].
pub(crate) fn epf_step_with_workers(
    step: EpfStep,
    input: [&[f32]; 3],
    guide: [&[f32]; 3],
    dims: PlaneDims,
    params: &EpfParams,
    sigma: &SigmaField<'_>,
    workers: usize,
) -> Result<[Vec<f32>; 3]> {
    for plane in input.iter().chain(guide.iter()).copied() {
        dims.check(plane, "epf plane length", "J.4")?;
    }
    let (bx_count, by_count) = block_grid(dims);
    if (sigma.blocks_x(), sigma.blocks_y()) != (bx_count, by_count) {
        return Err(FrameError::out_of_range(
            "sigma block grid",
            "J.4.3",
            u64::try_from(sigma.blocks_x()).unwrap_or(u64::MAX),
        ));
    }

    let ctx = StepCtx {
        input,
        guide,
        dims,
        params,
        step,
    };
    let kernel = step.kernel();
    let mut out = [
        vec![0.0f32; dims.len()],
        vec![0.0f32; dims.len()],
        vec![0.0f32; dims.len()],
    ];

    let row_bands = crate::parallel::bands(dims.height, workers);
    // Every band but the last holds exactly `band_len` rows, so `chunks_mut`
    // tiles the planes into these same bands in order. A zero-width plane
    // takes the serial path: its bands hold zero cells and `chunks_mut(0)`
    // would panic.
    let band_len = row_bands.first().map_or(0, |band| band.len());
    let band_cells = band_len.saturating_mul(dims.width);
    if row_bands.len() <= 1 || band_cells == 0 {
        let planes = out.each_mut().map(Vec::as_mut_slice);
        epf_step_rows(&ctx, sigma, kernel, 0..dims.height, planes);
        return Ok(out);
    }
    std::thread::scope(|scope| {
        let [chunks0, chunks1, chunks2] = out.each_mut().map(|plane| plane.chunks_mut(band_cells));
        debug_assert_eq!(
            chunks0.len(),
            row_bands.len(),
            "bands tile the planes exactly (see parallel::bands_match_chunks_mut)"
        );
        let mut_jobs = chunks0.zip(chunks1).zip(chunks2).zip(row_bands);
        for (((band0, band1), band2), rows) in mut_jobs {
            let (ctx_ref, sigma_ref) = (&ctx, sigma);
            scope.spawn(move || {
                epf_step_rows(ctx_ref, sigma_ref, kernel, rows, [band0, band1, band2]);
            });
        }
    });

    Ok(out)
}

/// Filters one band of rows, writing into the band's slices.
///
/// Reads use absolute frame coordinates; `out` holds exactly this band's
/// rows, so writes index relative to the band start (`rel_y`). Everything
/// else is the J.4.1–J.4.4 loop unchanged.
///
/// Dispatched to an AVX2+FMA build where the host supports it; the builds
/// are bit-identical — same IEEE operations in the same order, and
/// `mul_add` is a single rounding whether the hardware fuses it or the
/// baseline's `fmaf` libcall emulates it — and
/// `fma_build_matches_scalar_bitwise` pins that.
fn epf_step_rows(
    ctx: &StepCtx,
    sigma: &SigmaField,
    kernel: &[(i64, i64)],
    band: std::ops::Range<usize>,
    out: [&mut [f32]; 3],
) {
    #[cfg(target_arch = "x86_64")]
    if jpxl_core::cpu::has_fma() {
        // SAFETY: `epf_step_rows_fma` only requires AVX2+FMA, which
        // `has_fma` has just confirmed.
        #[allow(unsafe_code)]
        unsafe {
            epf_step_rows_fma(ctx, sigma, kernel, band, out);
        }
        return;
    }
    epf_step_rows_impl(ctx, sigma, kernel, band, out);
}

/// [`epf_step_rows`] compiled for AVX2+FMA.
///
/// Scalar fused instructions, not packed lanes: the win is fusing the
/// loop's dozens of `mul_add`s per pixel into hardware (the baseline build
/// calls the `fmaf` libcall per tap). Calling it is `unsafe` unless the
/// host supports AVX2+FMA (see [`jpxl_core::cpu::has_fma`]); that is the
/// whole contract.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn epf_step_rows_fma(
    ctx: &StepCtx,
    sigma: &SigmaField,
    kernel: &[(i64, i64)],
    band: std::ops::Range<usize>,
    out: [&mut [f32]; 3],
) {
    epf_step_rows_impl(ctx, sigma, kernel, band, out);
}

/// The band loop itself: the readable scalar reference and the lock-step
/// oracle for `epf_step_rows_fma` (see
/// `fma_build_matches_scalar_bitwise`). Runs wherever the dispatch falls
/// back: non-x86_64 hosts and hosts without AVX2+FMA.
///
/// `inline(always)` is the dispatch mechanism — it inlines this body into
/// the `target_feature` caller, whose AVX2+FMA codegen then fuses it.
#[inline(always)]
fn epf_step_rows_impl(
    ctx: &StepCtx,
    sigma: &SigmaField,
    kernel: &[(i64, i64)],
    band: std::ops::Range<usize>,
    mut out: [&mut [f32]; 3],
) {
    let dims = ctx.dims;
    let margin = ctx.step.read_margin();
    for (rel_y, y) in band.enumerate() {
        for x in 0..dims.width {
            let idx = rel_y.saturating_mul(dims.width).saturating_add(x);
            let (bx, by) = (x / BLOCK_DIM, y / BLOCK_DIM);
            let interior = interior_pixel(x, y, dims, margin);

            if sigma.skip_sigma_at(bx, by) < EPF_SIGMA_SKIP_THRESHOLD {
                // J.4.3: the step's output is its input on this block.
                for c in 0..3 {
                    let input_idx = y.saturating_mul(dims.width).saturating_add(x);
                    let v = ctx.input.get(c).and_then(|p| p.get(input_idx)).copied();
                    if let (Some(v), Some(slot)) = (v, out.get_mut(c).and_then(|p| p.get_mut(idx)))
                    {
                        *slot = v;
                    }
                }
                continue;
            }

            let block_sigma = sigma.sigma_at(bx, by);
            let (xi, yi) = (as_i64(x), as_i64(y));
            let reference_at_border = at_block_border(xi, yi);

            let mut sum_weights = 0.0f32;
            let mut sum_channels = [0.0f32; 3];
            for (ix, iy) in kernel.iter().copied() {
                let distance = ctx.distance(xi, yi, ix, iy, interior);
                let at_border = if EPF_BORDER_SAD_AT_REFERENCE_PIXEL {
                    reference_at_border
                } else {
                    at_block_border(xi + ix, yi + iy)
                };
                let weight = epf_weight(distance, block_sigma, ctx.step, at_border, ctx.params);
                sum_weights += weight;
                for (c, acc) in sum_channels.iter_mut().enumerate() {
                    *acc = ctx
                        .input_at(c, xi + ix, yi + iy, interior)
                        .mul_add(weight, *acc);
                }
            }

            // sum_weights >= 1: the (0, 0) tap has zero distance and hence
            // weight exactly 1, so this division is always safe.
            for (c, acc) in sum_channels.iter().enumerate() {
                if let Some(slot) = out.get_mut(c).and_then(|p| p.get_mut(idx)) {
                    *slot = acc / sum_weights;
                }
            }
        }
    }
}

/// Applies the whole edge-preserving filter (J.4) to the three colour planes
/// `[X, Y, B]`.
///
/// Returns fresh planes; with `epf_iters == 0` they are copies of the input.
///
/// # Errors
///
/// As [`epf_step`].
pub fn epf(
    input: [&[f32]; 3],
    dims: PlaneDims,
    params: &EpfParams,
    sigma: &SigmaField<'_>,
) -> Result<[Vec<f32>; 3]> {
    for plane in input {
        dims.check(plane, "epf plane length", "J.4")?;
    }

    let mut current: [Vec<f32>; 3] = input.map(<[f32]>::to_vec);
    let steps = epf_steps(params.iters);
    if steps.is_empty() {
        return Ok(current);
    }

    // Only materialized under the two-buffer reading of J.4.2/J.4.4.
    let original: Option<[Vec<f32>; 3]> = if EPF_DISTANCE_USES_STEP_INPUT {
        None
    } else {
        Some(current.clone())
    };

    for step in steps.iter().copied() {
        let current_refs: [&[f32]; 3] = current.each_ref().map(Vec::as_slice);
        let guide_refs: [&[f32]; 3] = original
            .as_ref()
            .map_or(current_refs, |o| o.each_ref().map(Vec::as_slice));
        let next = epf_step(step, current_refs, guide_refs, dims, params, sigma)?;
        current = next;
    }

    Ok(current)
}

/// `usize` to `i64` without a lint-triggering `as` cast.
fn as_i64(v: usize) -> i64 {
    i64::try_from(v).unwrap_or(i64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> EpfParams {
        EpfParams::default()
    }

    fn uniform_field(dims: PlaneDims, value: f32) -> Vec<f32> {
        let (bx, by) = block_grid(dims);
        vec![value; bx * by]
    }

    /// A deterministic plane whose samples differ by only a few thousandths,
    /// so that J.4.3's weights stay positive under the default channel scales.
    fn low_contrast(dims: PlaneDims, seed: u32) -> Vec<f32> {
        (0..dims.len())
            .map(|i| {
                let v = u32::try_from(i).unwrap_or(0).wrapping_mul(2_654_435_761) ^ seed;
                0.5 + f32::from(u16::try_from(v % 8).unwrap_or(0)) * 0.001
            })
            .collect()
    }

    fn ramp(dims: PlaneDims, seed: u32) -> Vec<f32> {
        (0..dims.len())
            .map(|i| {
                let v = u32::try_from(i).unwrap_or(0).wrapping_mul(2_654_435_761) ^ seed;
                f32::from(u16::try_from(v % 1000).unwrap_or(0)) / 1000.0
            })
            .collect()
    }

    #[test]
    fn steps_follow_the_explicit_conditions() {
        // Proves the epf_iters -> step mapping, and that the count of steps
        // run equals epf_iters for every legal value (the argument that makes
        // the explicit-conditions reading also satisfy the field name).
        assert!(epf_steps(0).is_empty());
        assert_eq!(epf_steps(1), [EpfStep::Step1].as_slice());
        assert_eq!(epf_steps(2), [EpfStep::Step1, EpfStep::Step2].as_slice());
        assert_eq!(
            epf_steps(3),
            [EpfStep::Step0, EpfStep::Step1, EpfStep::Step2].as_slice()
        );
        for iters in 0..=3u32 {
            assert_eq!(
                epf_steps(iters).len(),
                usize::try_from(iters).unwrap_or(0),
                "iters {iters}"
            );
        }
    }

    #[test]
    fn step0_kernel_is_the_thirteen_pixels_within_l1_two() {
        // Guards the {0,-2} / {9,-2} OCR divergence: the corrupt entry is
        // neither at L1 distance 2 nor a duplicate-free member of the set.
        let mut coords = STEP0_KERNEL_COORDS.to_vec();
        coords.sort_unstable();
        coords.dedup();
        assert_eq!(coords.len(), 13, "kernel coordinates must be distinct");
        for (dx, dy) in STEP0_KERNEL_COORDS {
            assert!(dx.abs() + dy.abs() <= 2, "({dx},{dy}) is outside L1 <= 2");
        }
        // The set is symmetric under negation and under swapping the axes.
        for (dx, dy) in STEP0_KERNEL_COORDS {
            assert!(coords.binary_search(&(-dx, -dy)).is_ok(), "({dx},{dy})");
            assert!(coords.binary_search(&(dy, dx)).is_ok(), "({dx},{dy})");
        }
    }

    #[test]
    fn centre_weight_is_exactly_one() {
        // The invariant that makes sum_weights >= 1, hence J.4.4's division
        // safe on any input.
        for step in [EpfStep::Step0, EpfStep::Step1, EpfStep::Step2] {
            for at_border in [false, true] {
                for sigma in [0.3f32, 1.0, 1e6, 0.0] {
                    let w = epf_weight(0.0, sigma, step, at_border, &params());
                    assert_eq!(w, 1.0, "step {step:?} border {at_border} sigma {sigma}");
                }
            }
        }
    }

    #[test]
    fn weight_decreases_with_distance_and_hits_zero() {
        let p = params();
        let a = epf_weight(0.1, 10.0, EpfStep::Step1, false, &p);
        let b = epf_weight(0.2, 10.0, EpfStep::Step1, false, &p);
        assert!(a > b && b > 0.0, "{a} {b}");
        assert_eq!(epf_weight(1e6, 10.0, EpfStep::Step1, false, &p), 0.0);
        // The border multiplier is 2/3 by default, so a border pixel's taps
        // are penalized less, not more.
        let plain = epf_weight(0.1, 10.0, EpfStep::Step1, false, &p);
        let border = epf_weight(0.1, 10.0, EpfStep::Step1, true, &p);
        assert!(border > plain, "{border} vs {plain}");
    }

    #[test]
    fn block_border_predicate_is_frame_origin_aligned() {
        // Frame-absolute and block-relative readings coincide because the
        // block grid starts at the frame origin.
        assert!(at_block_border(0, 3));
        assert!(at_block_border(7, 3));
        assert!(at_block_border(3, 8));
        assert!(at_block_border(3, 15));
        assert!(!at_block_border(3, 3));
        assert!(!at_block_border(9, 14));
        // Negative coordinates use the unsigned (Euclidean) remainder.
        assert!(at_block_border(-1, 3), "-1 mod 8 == 7");
        assert!(!at_block_border(-3, 3), "-3 mod 8 == 5");
    }

    #[test]
    fn sigma_below_threshold_is_the_identity() {
        // Proves the J.4.3 skip rule end to end: with every block under 0.3
        // the filter must return its input unchanged, bit for bit.
        let dims = PlaneDims::new(20, 12);
        let planes = [ramp(dims, 1), ramp(dims, 2), ramp(dims, 3)];
        let s = uniform_field(dims, 0.299_999);
        let (bx, by) = block_grid(dims);
        let field = SigmaField::new(&s, bx, by).expect("valid");
        let p = EpfParams {
            iters: 3,
            ..params()
        };
        let out = epf(planes.each_ref().map(Vec::as_slice), dims, &p, &field).expect("valid");
        for c in 0..3 {
            assert_eq!(out.get(c), planes.get(c), "channel {c}");
        }
    }

    #[test]
    fn zero_iters_is_the_identity() {
        let dims = PlaneDims::new(9, 9);
        let planes = [ramp(dims, 5), ramp(dims, 6), ramp(dims, 7)];
        let s = uniform_field(dims, 100.0);
        let (bx, by) = block_grid(dims);
        let field = SigmaField::new(&s, bx, by).expect("valid");
        let p = EpfParams {
            iters: 0,
            ..params()
        };
        let out = epf(planes.each_ref().map(Vec::as_slice), dims, &p, &field).expect("valid");
        for c in 0..3 {
            assert_eq!(out.get(c), planes.get(c), "channel {c}");
        }
    }

    #[test]
    fn constant_planes_survive_every_step() {
        // All distances are zero, so all weights are 1 and the average of the
        // taps is the constant. Proves the weighted average is normalized.
        let dims = PlaneDims::new(17, 11);
        let planes = [
            vec![0.25f32; dims.len()],
            vec![-1.5f32; dims.len()],
            vec![7.0f32; dims.len()],
        ];
        let s = uniform_field(dims, 5.0);
        let (bx, by) = block_grid(dims);
        let field = SigmaField::new(&s, bx, by).expect("valid");
        for step in [EpfStep::Step0, EpfStep::Step1, EpfStep::Step2] {
            let refs = planes.each_ref().map(Vec::as_slice);
            let out = epf_step(step, refs, refs, dims, &params(), &field).expect("valid");
            for (c, want) in [0.25f32, -1.5, 7.0].iter().enumerate() {
                for v in out.get(c).map(Vec::as_slice).unwrap_or(&[]) {
                    assert!((v - want).abs() < 1e-5, "step {step:?} channel {c}: {v}");
                }
            }
        }
    }

    #[test]
    fn one_pixel_wide_plane_mirrors_without_escaping() {
        // The degenerate mirroring case: on a 1xN plane every horizontal tap,
        // including the +-2 taps of step 0, folds back onto column 0. A naive
        // single-reflection mirror indexes out of the plane here.
        let dims = PlaneDims::new(1, 5);
        let planes = [
            vec![0.0f32, 1.0, 0.0, 1.0, 0.0],
            vec![0.5f32; 5],
            vec![0.0f32; 5],
        ];
        let s = uniform_field(dims, 50.0);
        let (bx, by) = block_grid(dims);
        let field = SigmaField::new(&s, bx, by).expect("valid");
        let p = EpfParams {
            iters: 3,
            ..params()
        };
        let out = epf(planes.each_ref().map(Vec::as_slice), dims, &p, &field).expect("valid");
        for c in 0..3 {
            for v in out.get(c).map(Vec::as_slice).unwrap_or(&[]) {
                assert!(v.is_finite(), "channel {c}: {v}");
                assert!((-0.001..=1.001).contains(v), "channel {c}: {v}");
            }
        }
    }

    #[test]
    fn worker_count_never_changes_a_sample() {
        // Threading partitions rows; it must not move a single bit. A
        // ragged 130x70 ramp exercises band edges (70 rows over 8 workers
        // is 9+9+9+9+9+9+9+7) on every step, against the serial run.
        let dims = PlaneDims::new(130, 70);
        let planes = [ramp(dims, 11), low_contrast(dims, 22), ramp(dims, 33)];
        let s = uniform_field(dims, 5.0);
        let (bx, by) = block_grid(dims);
        let field = SigmaField::new(&s, bx, by).expect("valid");
        for step in [EpfStep::Step0, EpfStep::Step1, EpfStep::Step2] {
            let refs = planes.each_ref().map(Vec::as_slice);
            let serial =
                epf_step_with_workers(step, refs, refs, dims, &params(), &field, 1).expect("valid");
            for workers in [2, 3, 8, 64] {
                let refs = planes.each_ref().map(Vec::as_slice);
                let threaded =
                    epf_step_with_workers(step, refs, refs, dims, &params(), &field, workers)
                        .expect("valid");
                for c in 0..3 {
                    let (a, b) = (
                        serial.get(c).map(Vec::as_slice).unwrap_or(&[]),
                        threaded.get(c).map(Vec::as_slice).unwrap_or(&[]),
                    );
                    assert_eq!(a.len(), b.len(), "step {step:?} workers {workers}");
                    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                        assert_eq!(
                            x.to_bits(),
                            y.to_bits(),
                            "step {step:?} workers {workers} channel {c} sample {i}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn margins_cover_every_read_of_every_step() {
        // Each step's `read_margin` must cover every sample its loop reads:
        // the J.4.4 input tap plus, per tap, the J.4.2 distance's cross
        // around the reference pixel and around the tap (steps 0–1) or the
        // two pixels alone (step 2). The kernels and the cross come from
        // the same constants the filter reads, so this proves the margins,
        // not the constants. The `checked` count proves the larger planes
        // actually exercise the interior (on degenerate planes it is empty
        // and the loop checks nothing).
        for step in [EpfStep::Step0, EpfStep::Step1, EpfStep::Step2] {
            let margin = step.read_margin();
            let kernel = step.kernel();
            let mut checked = 0;
            for width in 1..12usize {
                for height in 1..12usize {
                    let dims = PlaneDims::new(width, height);
                    for y in 0..height {
                        for x in 0..width {
                            if !interior_pixel(x, y, dims, margin) {
                                continue;
                            }
                            checked += 1;
                            let (xi, yi) = (as_i64(x), as_i64(y));
                            let in_bounds = |rx: i64, ry: i64, what: &str| {
                                assert!(
                                    rx >= 0 && ry >= 0,
                                    "step {step:?}: {what} of ({x},{y}) escapes {width}x{height}"
                                );
                                assert!(
                                    rx < as_i64(width) && ry < as_i64(height),
                                    "step {step:?}: {what} of ({x},{y}) escapes {width}x{height}"
                                );
                            };
                            for (kx, ky) in kernel.iter().copied() {
                                in_bounds(xi + kx, yi + ky, "input tap");
                                if step == EpfStep::Step2 {
                                    in_bounds(xi, yi, "step-2 reference");
                                    in_bounds(xi + kx, yi + ky, "step-2 tap");
                                } else {
                                    for (ix, iy) in CROSS_COORDS {
                                        in_bounds(xi + ix, yi + iy, "cross at reference");
                                        in_bounds(xi + kx + ix, yi + ky + iy, "cross at tap");
                                    }
                                }
                            }
                        }
                    }
                }
            }
            assert!(
                checked > 0,
                "step {step:?}: no interior pixel was ever checked"
            );
        }
    }

    #[test]
    fn fma_build_matches_scalar_bitwise() {
        // The dispatched band build returns the scalar impl's bytes exactly,
        // on every step over a ragged ramp (interior, border and skip-path
        // pixels all covered). `mul_add` is a single rounding whether the
        // hardware fuses it or the baseline's `fmaf` libcall emulates it.
        // (On a host without AVX2+FMA both sides run the scalar impl and
        // the test is vacuous — it proves the dispatch, which only an
        // AVX2+FMA host exercises.)
        let dims = PlaneDims::new(130, 70);
        let planes = [ramp(dims, 11), low_contrast(dims, 22), ramp(dims, 33)];
        let s = uniform_field(dims, 5.0);
        let (bx, by) = block_grid(dims);
        let field = SigmaField::new(&s, bx, by).expect("valid");
        let params = params();
        for step in [EpfStep::Step0, EpfStep::Step1, EpfStep::Step2] {
            let refs = planes.each_ref().map(Vec::as_slice);
            let ctx = StepCtx {
                input: refs,
                guide: refs,
                dims,
                params: &params,
                step,
            };
            let mut dispatched = [
                vec![0.0f32; dims.len()],
                vec![0.0f32; dims.len()],
                vec![0.0f32; dims.len()],
            ];
            let mut scalar = [
                vec![0.0f32; dims.len()],
                vec![0.0f32; dims.len()],
                vec![0.0f32; dims.len()],
            ];
            epf_step_rows(
                &ctx,
                &field,
                step.kernel(),
                0..dims.height,
                dispatched.each_mut().map(Vec::as_mut_slice),
            );
            epf_step_rows_impl(
                &ctx,
                &field,
                step.kernel(),
                0..dims.height,
                scalar.each_mut().map(Vec::as_mut_slice),
            );
            for c in 0..3 {
                let (a, b) = (
                    dispatched.get(c).map(Vec::as_slice).unwrap_or(&[]),
                    scalar.get(c).map(Vec::as_slice).unwrap_or(&[]),
                );
                assert_eq!(a.len(), b.len(), "step {step:?}");
                for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                    assert_eq!(
                        x.to_bits(),
                        y.to_bits(),
                        "step {step:?} channel {c} sample {i}"
                    );
                }
            }
        }
    }

    #[test]
    fn single_pixel_plane_is_the_identity() {
        let dims = PlaneDims::new(1, 1);
        let planes = [vec![0.25f32], vec![0.5f32], vec![0.75f32]];
        let s = uniform_field(dims, 100.0);
        let field = SigmaField::new(&s, 1, 1).expect("valid");
        let p = EpfParams {
            iters: 3,
            ..params()
        };
        let out = epf(planes.each_ref().map(Vec::as_slice), dims, &p, &field).expect("valid");
        for (c, want) in [0.25f32, 0.5, 0.75].iter().enumerate() {
            let got = out.get(c).and_then(|p| p.first()).copied().unwrap_or(0.0);
            assert!((got - want).abs() < 1e-6, "channel {c}: {got}");
        }
    }

    #[test]
    fn output_is_a_convex_combination_of_the_input() {
        // sum_weights >= 1 and every weight >= 0, so no output can leave the
        // input's range: the guard against a division blow-up, checked on real
        // data rather than on the centre weight alone.
        let dims = PlaneDims::new(19, 17);
        let planes = [ramp(dims, 11), ramp(dims, 12), ramp(dims, 13)];
        let s = uniform_field(dims, 8.0);
        let (bx, by) = block_grid(dims);
        let field = SigmaField::new(&s, bx, by).expect("valid");
        let p = EpfParams {
            iters: 3,
            ..params()
        };
        let out = epf(planes.each_ref().map(Vec::as_slice), dims, &p, &field).expect("valid");
        for c in 0..3 {
            let src = planes.get(c).map(Vec::as_slice).unwrap_or(&[]);
            let lo = src.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = src.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            for v in out.get(c).map(Vec::as_slice).unwrap_or(&[]) {
                assert!(v.is_finite() && *v >= lo - 1e-5 && *v <= hi + 1e-5, "{v}");
            }
        }
    }

    #[test]
    fn hand_computed_step2_on_a_three_pixel_row() {
        // dims 3x1, so the whole row is one 8x8 block (block_grid == (1,1)).
        // Planes: X = B = 0, Y = [0, 1, 0]; sigma = 100 everywhere.
        //
        // Reference pixel (1, 0), step 2, kernel {0,0} {-1,0} {1,0} {0,-1} {0,1}:
        //   * {0,0}   distance 0                       -> weight 1, tap Y = 1
        //   * {0,-1} and {0,1} mirror onto row 0, i.e. back onto (1,0):
        //             distance 0                       -> weight 1, tap Y = 1
        //   * {-1,0} and {1,0}: DistanceStep2 sums |1-0| * channel_scale[Y]
        //             = 1 * 5 = 5                      -> weight w, tap Y = 0
        //
        // The reference pixel has y mod 8 == 0, so it is on a block border and
        // position_multiplier = epf_border_sad_mul = 2/3.
        //   inv_sigma = 1.65 * 6.5 * 4 * (1 - sqrt(0.5)) / 100 = 0.1256511909
        //   scaled    = (2/3) * 5 = 3.3333333
        //   w         = 1 - 3.3333333 * 0.1256511909 = 0.5811626971
        // sum_weights = 1 + 1 + 1 + 2w = 4.1623253942
        // sum_Y       = 1 + 1 + 1 + 0  = 3
        // output Y    = 3 / 4.1623253942 = 0.7207509543
        let dims = PlaneDims::new(3, 1);
        let zero = vec![0.0f32; 3];
        let y_plane = vec![0.0f32, 1.0, 0.0];
        let s = vec![100.0f32];
        let field = SigmaField::new(&s, 1, 1).expect("valid");
        let refs: [&[f32]; 3] = [&zero, &y_plane, &zero];

        let w = epf_weight(5.0, 100.0, EpfStep::Step2, true, &params());
        assert!((w - 0.581_162_7).abs() < 1e-6, "tap weight {w}");

        let out = epf_step(EpfStep::Step2, refs, refs, dims, &params(), &field).expect("valid");
        let got = out.get(1).and_then(|p| p.get(1)).copied().unwrap_or(0.0);
        assert!((got - 0.720_751).abs() < 1e-6, "centre Y = {got}");
        // X and B are all zero and stay zero: every tap contributes 0.
        for c in [0usize, 2] {
            for v in out.get(c).map(Vec::as_slice).unwrap_or(&[]) {
                assert_eq!(*v, 0.0, "channel {c}");
            }
        }
    }

    #[test]
    fn vardct_sigma_multiplies_quant_width_by_quant_mul_and_the_lut() {
        let p = params();
        // sharp_lut[7] == 1, so sigma == mul * quant_mul there.
        let s = vardct_sigma(2.0, 7, &p).expect("valid");
        assert!((s - 2.0 * 0.46).abs() < 1e-6, "{s}");
        // sharp_lut[0] == 0 zeroes sigma, which always trips the skip rule.
        assert_eq!(vardct_sigma(2.0, 0, &p).expect("valid"), 0.0);
        // Out-of-range Sharpness is rejected, not clamped.
        assert!(vardct_sigma(2.0, 8, &p).is_err());
    }

    #[test]
    fn per_varblock_skip_overrides_the_block_sigma() {
        // Two 8x8 blocks side by side. The per-block sigmas are both above the
        // threshold, but the varblock covering them is below it, so under the
        // per-varblock reading nothing is filtered.
        let dims = PlaneDims::new(16, 8);
        // Low-contrast content on purpose. `ramp` swings across the whole
        // [0, 1) range, and `epf_channel_scale`'s defaults {40, 5, 3.5} then
        // make every L1 distance large enough to zero every off-centre weight
        // — so the filter would be the identity whichever way the skip reading
        // goes, and the test would prove nothing. Amplitudes of a few
        // thousandths keep the weights positive.
        let planes = [
            low_contrast(dims, 21),
            low_contrast(dims, 22),
            low_contrast(dims, 23),
        ];
        let block_sigma = vec![4.0f32, 4.0];
        let varblock_sigma = vec![0.1f32, 0.1];
        let field = SigmaField::new(&block_sigma, 2, 1)
            .expect("valid")
            .with_varblock_sigma(&varblock_sigma)
            .expect("valid");
        assert!(field.has_varblock_sigma());
        let out = epf(
            planes.each_ref().map(Vec::as_slice),
            dims,
            &params(),
            &field,
        )
        .expect("valid");
        if EPF_SKIP_IS_PER_VARBLOCK {
            for c in 0..3 {
                assert_eq!(out.get(c), planes.get(c), "channel {c}");
            }
        } else {
            assert_ne!(out.first(), planes.first());
        }
    }

    #[test]
    fn shape_mismatches_are_errors() {
        let dims = PlaneDims::new(16, 16);
        let s = uniform_field(dims, 1.0);
        let (bx, by) = block_grid(dims);
        let field = SigmaField::new(&s, bx, by).expect("valid");
        let short = vec![0.0f32; 10];
        let short_refs: [&[f32]; 3] = [&short, &short, &short];
        assert!(epf(short_refs, dims, &params(), &field).is_err());
        // A sigma field on the wrong grid is rejected too.
        let wrong = SigmaField::new(&s, bx, by).expect("valid");
        let other = PlaneDims::new(8, 8);
        let planes = vec![0.0f32; other.len()];
        let plane_refs: [&[f32]; 3] = [&planes, &planes, &planes];
        assert!(
            epf(plane_refs, other, &params(), &wrong).is_err(),
            "sigma grid must match the plane dimensions"
        );
        assert!(SigmaField::new(&s, bx + 1, by).is_err());
    }

    #[test]
    fn block_grid_rounds_up() {
        assert_eq!(block_grid(PlaneDims::new(16, 8)), (2, 1));
        assert_eq!(block_grid(PlaneDims::new(17, 9)), (3, 2));
        assert_eq!(block_grid(PlaneDims::new(1, 1)), (1, 1));
    }
}
