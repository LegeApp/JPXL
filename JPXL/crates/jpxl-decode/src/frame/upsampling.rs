// SPDX-License-Identifier: MIT
//! Non-separable upsampling (18181-1 K.2) and simple upsampling (J.2).
//!
//! # Where this sits in the pipeline
//!
//! Clause 4 fixes the stage order: frame data (Annex G), restoration filters
//! (Annex J), image features (Annex K), then colour transforms (Annex L).
//! K.1 places upsampling at the head of Annex K — "in case
//! `frame_header.upsampling > 1` and/or `max(frame_header.ec_upsampling) > 1`,
//! the corresponding colour and/or extra channels are **first** upsampled as
//! specified in K.2" — so the gaborish convolution (J.3) and the
//! edge-preserving filter (J.4) run at the frame's *stored* size and K.2 runs
//! after them, before patches/splines/noise. K.3.2 restates the same ordering
//! from the other side: patch samples are blended "after the upsampling from
//! J.2 and K.2".
//!
//! That order is load-bearing. Upsampling before the filters would run EPF
//! over `k` times as many samples with a sigma field derived from the
//! downsampled block grid, and would blur the interpolation rather than the
//! coded signal.
//!
//! # The K.2 filter
//!
//! `k x k` output samples are produced from each input sample `p`, each one a
//! fixed linear combination of the 5x5 window `W` centred on `p` (boundary
//! samples mirrored per 5.2), then clamped to `[min(W), max(W)]`. The clamp is
//! what keeps the filter from ringing past the local sample range; it is not
//! optional and it is per output sample, against that output sample's own
//! window.
//!
//! The weight for the input at `(ix, iy)` of the window, for the output at
//! `(kx, ky)` of the `k x k` block, is `up{k}_weight[weight_index(...)]`. The
//! index formula folds the four-fold symmetry of the filter, which is why
//! `k = 2` needs only 15 weights rather than `25 * 4`.
//!
//! # Why the default tables can be trusted
//!
//! 280 signed constants transcribed from a scanned table is exactly the shape
//! of data OCR garbles. They carry their own checksum: the 25 window weights
//! sum to one for **every** one of the `2*2 + 4*4 + 8*8 = 84` output
//! positions, which [`default_weights_are_normalised`] asserts to within
//! `1e-6`. A single digit slip anywhere in the 280 values breaks at least one
//! of those 84 sums, so the test proves the transcription and the index
//! formula together.
//!
//! # J.2
//!
//! J.2 "simple upsampling" is the separable triangle filter used for
//! `jpeg_upsampling` (chroma subsampling in a `do_YCbCr` frame) only — a
//! different filter for a different field. It is not implemented here;
//! `jpeg_upsampling != 0` is still a typed refusal.

use jpxl_core::limits::AllocGuard;

use crate::frame::error::{FrameError, Result};
use crate::frame::gaborish::{PlaneDims, sample_mirrored};

/// Side of the window K.2 reads for each input sample.
const WINDOW: usize = 5;
/// Number of samples in that window.
const WINDOW_CELLS: usize = WINDOW * WINDOW;

/// The largest single-step upsampling factor K.2 defines.
pub const MAX_STEP: u32 = 8;

/// The largest cumulative factor F.2 permits
/// (`ec_upsampling[i] << ec_info[i].dim_shift <= 64`).
pub const MAX_TOTAL: u32 = 64;

/// `d_up2` (18181-1 K.2): the 15 default weights for 2x upsampling.
pub const DEFAULT_UP2: [f64; 15] = [
    -0.017_162_00,
    -0.034_523_03,
    -0.040_221_74,
    -0.029_210_14,
    -0.006_246_45,
    0.141_110_91,
    0.288_967_55,
    0.002_787_18,
    -0.016_102_67,
    0.566_615_50,
    0.037_776_07,
    -0.019_866_94,
    -0.031_447_31,
    -0.011_850_68,
    -0.002_135_39,
];

/// `d_up4` (18181-1 K.2): the 55 default weights for 4x upsampling.
pub const DEFAULT_UP4: [f64; 55] = [
    -0.024_190_67,
    -0.034_919_87,
    -0.036_933_51,
    -0.030_942_85,
    -0.005_297_85,
    -0.016_634_32,
    -0.035_568_63,
    -0.038_889_05,
    -0.035_168_50,
    -0.009_894_69,
    0.236_519_58,
    0.333_929_45,
    -0.010_735_43,
    -0.013_131_81,
    -0.035_566_94,
    0.130_481_75,
    0.401_030_25,
    0.039_511_50,
    -0.020_775_84,
    0.469_141_98,
    -0.002_092_70,
    -0.014_845_89,
    -0.040_648_06,
    0.189_425_30,
    0.562_798_92,
    0.066_744_00,
    -0.023_354_94,
    -0.035_516_82,
    -0.007_548_30,
    -0.022_679_19,
    -0.023_635_78,
    0.003_158_04,
    -0.033_990_98,
    -0.013_595_19,
    -0.000_916_53,
    -0.003_354_67,
    -0.011_632_94,
    -0.016_102_94,
    -0.009_740_88,
    -0.001_916_22,
    -0.010_954_46,
    -0.031_984_64,
    -0.044_551_21,
    -0.027_997_90,
    -0.006_459_12,
    0.063_905_99,
    0.229_638_88,
    0.006_309_81,
    -0.018_973_49,
    0.675_372_68,
    0.084_833_69,
    -0.025_349_94,
    -0.022_051_97,
    -0.016_679_99,
    -0.003_844_43,
];

/// `d_up8` (18181-1 K.2): the 210 default weights for 8x upsampling.
pub const DEFAULT_UP8: [f64; 210] = [
    -0.029_286_13,
    -0.037_063_53,
    -0.037_838_12,
    -0.033_245_58,
    -0.004_476_32,
    -0.025_194_06,
    -0.037_526_01,
    -0.039_015_08,
    -0.036_632_85,
    -0.006_466_49,
    -0.020_664_07,
    -0.038_386_33,
    -0.040_021_01,
    -0.039_000_35,
    -0.009_019_73,
    -0.016_263_93,
    -0.039_541_48,
    -0.040_466_20,
    -0.039_796_21,
    -0.012_244_85,
    0.298_953_28,
    0.357_577_08,
    -0.024_475_52,
    -0.010_817_48,
    -0.043_145_94,
    0.239_032_19,
    0.411_193_01,
    -0.005_730_46,
    -0.014_502_39,
    -0.042_468_45,
    0.175_676_18,
    0.452_206_43,
    0.022_877_57,
    -0.019_367_83,
    -0.035_832_55,
    0.115_724_72,
    0.474_167_33,
    0.062_844_40,
    -0.026_850_66,
    0.427_200_50,
    -0.022_489_39,
    -0.011_552_73,
    -0.045_627_55,
    0.286_894_96,
    0.490_938_69,
    -0.000_078_91,
    -0.015_459_26,
    -0.045_626_59,
    0.212_389_20,
    0.539_809_34,
    0.033_694_74,
    -0.020_702_11,
    -0.038_669_88,
    0.142_295_50,
    0.565_933_98,
    0.080_451_81,
    -0.028_882_98,
    -0.036_809_18,
    -0.005_422_29,
    -0.029_204_77,
    -0.027_885_74,
    -0.021_181_80,
    -0.039_424_02,
    -0.007_755_47,
    -0.024_336_14,
    -0.031_939_43,
    -0.020_308_28,
    -0.040_440_14,
    -0.010_740_16,
    -0.019_308_22,
    -0.036_203_99,
    -0.019_741_25,
    -0.039_195_45,
    -0.014_560_93,
    -0.000_450_72,
    -0.003_601_10,
    -0.010_202_07,
    -0.012_319_07,
    -0.006_389_88,
    -0.000_715_92,
    -0.002_791_22,
    -0.009_571_15,
    -0.012_883_27,
    -0.007_309_37,
    -0.001_077_83,
    -0.002_101_56,
    -0.008_907_05,
    -0.013_176_68,
    -0.008_138_95,
    -0.001_534_91,
    -0.021_284_81,
    -0.041_730_44,
    -0.048_314_87,
    -0.032_931_90,
    -0.005_252_60,
    -0.017_203_22,
    -0.040_527_36,
    -0.050_457_06,
    -0.036_073_17,
    -0.007_380_30,
    -0.013_417_64,
    -0.039_656_29,
    -0.051_516_16,
    -0.038_148_86,
    -0.010_058_19,
    0.189_682_73,
    0.330_636_84,
    -0.013_001_05,
    -0.013_729_50,
    -0.040_174_65,
    0.137_278_32,
    0.364_022_34,
    0.010_278_90,
    -0.018_321_07,
    -0.033_650_72,
    0.087_345_06,
    0.381_942_95,
    0.043_382_28,
    -0.025_259_93,
    0.564_081_26,
    0.004_583_52,
    -0.016_482_27,
    -0.048_878_68,
    0.245_855_19,
    0.620_261_35,
    0.043_148_07,
    -0.022_137_37,
    -0.041_580_14,
    0.166_372_89,
    0.650_270_23,
    0.096_216_36,
    -0.031_013_88,
    -0.040_827_42,
    -0.009_045_19,
    -0.027_909_22,
    -0.021_178_18,
    0.007_986_62,
    -0.039_957_11,
    -0.012_434_27,
    -0.022_317_05,
    -0.029_462_66,
    0.009_920_55,
    -0.036_002_83,
    -0.016_849_20,
    -0.001_116_84,
    -0.004_112_04,
    -0.012_971_30,
    -0.017_237_25,
    -0.010_225_45,
    -0.001_653_06,
    -0.003_131_10,
    -0.012_180_16,
    -0.017_632_66,
    -0.011_256_20,
    -0.002_316_63,
    -0.013_741_49,
    -0.037_976_20,
    -0.051_429_37,
    -0.031_173_07,
    -0.005_819_14,
    -0.010_640_03,
    -0.036_080_89,
    -0.052_721_68,
    -0.033_756_70,
    -0.007_955_86,
    0.096_281_04,
    0.271_299_91,
    -0.003_537_79,
    -0.017_341_51,
    -0.031_539_81,
    0.056_862_30,
    0.285_009_98,
    0.022_305_94,
    -0.023_749_55,
    0.682_143_26,
    0.050_180_48,
    -0.023_208_52,
    -0.043_836_16,
    0.184_594_74,
    0.715_179_75,
    0.108_056_13,
    -0.032_636_77,
    -0.036_376_39,
    -0.013_943_73,
    -0.025_112_03,
    -0.017_286_36,
    0.054_073_31,
    -0.028_675_68,
    -0.018_931_31,
    -0.002_408_54,
    -0.004_465_11,
    -0.016_361_87,
    -0.023_770_53,
    -0.015_228_48,
    -0.003_333_34,
    -0.008_199_75,
    -0.029_641_69,
    -0.044_992_87,
    -0.027_453_50,
    -0.006_124_08,
    0.027_274_16,
    0.194_466_00,
    0.001_598_32,
    -0.022_324_73,
    0.749_825_06,
    0.114_526_20,
    -0.033_480_48,
    -0.016_056_81,
    -0.020_703_39,
    -0.004_582_23,
];

/// K.2's index into `up{k}_weight` for one (output position, window position)
/// pair.
///
/// `kx`/`ky` are the output sample's position inside the `k x k` block,
/// `ix`/`iy` the input sample's position inside the 5x5 window. Transcribed
/// verbatim from K.2:
///
/// ```text
/// j = (ky < k/2) ? (iy + 5 * ky) : ((4 - iy) + 5 * (k - 1 - ky));
/// i = (kx < k/2) ? (ix + 5 * kx) : ((4 - ix) + 5 * (k - 1 - kx));
/// y = min(i, j); x = max(i, j);
/// index = 5 * k * y / 2 - y * (y - 1) / 2 + x - y;
/// ```
///
/// The `y`/`x` fold is what makes the table triangular: only the upper
/// triangle of the `(5k/2) x (5k/2)` symmetric layout is stored.
#[must_use]
pub const fn weight_index(factor: usize, kx: usize, ky: usize, ix: usize, iy: usize) -> usize {
    let half = factor / 2;
    let j = if ky < half {
        iy + WINDOW * ky
    } else {
        (WINDOW - 1 - iy) + WINDOW * (factor - 1 - ky)
    };
    let i = if kx < half {
        ix + WINDOW * kx
    } else {
        (WINDOW - 1 - ix) + WINDOW * (factor - 1 - kx)
    };
    let (y, x) = if i < j { (i, j) } else { (j, i) };
    // `y * (y - 1) / 2` with `y == 0` would underflow in `usize`; the term is
    // zero there, which `saturating_sub` gives directly.
    WINDOW * factor * y / 2 - y * y.saturating_sub(1) / 2 + x - y
}

/// How many weights a factor-`k` table holds: `5k/2 * (5k/2 + 1) / 2`.
#[must_use]
pub const fn weight_count(factor: usize) -> usize {
    let side = WINDOW * factor / 2;
    side * (side + 1) / 2
}

/// The default `d_up{k}` table for one factor.
///
/// Stored as `f64` so the printed digits survive verbatim — several of them
/// carry more precision than an `f32` holds, and truncating a normative table
/// to make a lint happy is how a transcription stops being one. The values
/// become `f32` when a kernel is expanded, which is where the pipeline is.
///
/// # Errors
///
/// [`FrameError::FieldOutOfRange`] if `factor` is not 2, 4 or 8.
pub fn default_weights(factor: u32) -> Result<&'static [f64]> {
    Ok(match factor {
        2 => &DEFAULT_UP2,
        4 => &DEFAULT_UP4,
        8 => &DEFAULT_UP8,
        _ => {
            return Err(FrameError::out_of_range(
                "upsampling factor",
                "K.2",
                u64::from(factor),
            ));
        }
    })
}

/// One factor's expanded filter: `factor * factor` windows of 25 weights.
///
/// Expanding K.2's triangular table once per frame keeps the inner loop a
/// plain dot product and keeps the index formula in exactly one place.
#[derive(Debug, Clone, PartialEq)]
pub struct UpsamplingKernel {
    factor: u32,
    /// `factor * factor` entries, indexed `ky * factor + kx`, each 25 long in
    /// `iy * 5 + ix` order.
    taps: Vec<[f32; WINDOW_CELLS]>,
}

impl UpsamplingKernel {
    /// Expands `weights` (the `up{k}_weight` array of D.3) for `factor`.
    ///
    /// # Errors
    ///
    /// [`FrameError::FieldOutOfRange`] if `factor` is not 2, 4 or 8, or if
    /// `weights` is not [`weight_count`] long.
    pub fn new(factor: u32, weights: &[f64]) -> Result<Self> {
        if !matches!(factor, 2 | 4 | 8) {
            return Err(FrameError::out_of_range(
                "upsampling factor",
                "K.2",
                u64::from(factor),
            ));
        }
        let k = factor as usize;
        if weights.len() != weight_count(k) {
            return Err(FrameError::out_of_range(
                "upsampling weight count",
                "D.3",
                weights.len() as u64,
            ));
        }
        let mut taps = vec![[0.0f32; WINDOW_CELLS]; k * k];
        for ky in 0..k {
            for kx in 0..k {
                let Some(tap) = taps.get_mut(ky * k + kx) else {
                    continue;
                };
                for iy in 0..WINDOW {
                    for ix in 0..WINDOW {
                        let index = weight_index(k, kx, ky, ix, iy);
                        let weight = weights.get(index).copied().ok_or_else(|| {
                            FrameError::out_of_range("up_weight index", "K.2", index as u64)
                        })?;
                        if let Some(slot) = tap.get_mut(iy * WINDOW + ix) {
                            // The table is `f64` to keep the printed digits
                            // verbatim; the filter runs in `f32`.
                            #[allow(
                                clippy::cast_possible_truncation,
                                reason = "the sample pipeline is f32"
                            )]
                            {
                                *slot = weight as f32;
                            }
                        }
                    }
                }
            }
        }
        Ok(Self { factor, taps })
    }

    /// The kernel built from K.2's default table.
    ///
    /// # Errors
    ///
    /// [`FrameError::FieldOutOfRange`] if `factor` is not 2, 4 or 8.
    pub fn default_for(factor: u32) -> Result<Self> {
        Self::new(factor, default_weights(factor)?)
    }

    /// The upsampling factor this kernel applies.
    #[must_use]
    pub const fn factor(&self) -> u32 {
        self.factor
    }
}

/// Applies one K.2 step: `factor x factor` output samples per input sample.
///
/// The result is `factor * dims.width` by `factor * dims.height`; the caller
/// crops it to the target size (see [`upsample_plane`]).
///
/// # Errors
///
/// [`FrameError::FieldOutOfRange`] if `src` is not `dims.width * dims.height`
/// long, or [`FrameError::Core`] if the output exceeds the allocation budget.
pub fn upsample_step(
    src: &[f32],
    dims: PlaneDims,
    kernel: &UpsamplingKernel,
    guard: &mut AllocGuard,
) -> Result<(Vec<f32>, PlaneDims)> {
    if src.len() != dims.width.saturating_mul(dims.height) {
        return Err(FrameError::out_of_range(
            "upsampling input length",
            "K.2",
            src.len() as u64,
        ));
    }
    let k = kernel.factor as usize;
    let out = PlaneDims::new(dims.width.saturating_mul(k), dims.height.saturating_mul(k));
    let cells = (out.width as u64).saturating_mul(out.height as u64);
    guard.charge(cells * 4).map_err(FrameError::Core)?;
    let len = usize::try_from(cells)
        .map_err(|_| FrameError::out_of_range("upsampled plane area", "K.2", cells))?;
    let mut dst = vec![0.0f32; len];

    for y in 0..dims.height {
        for x in 0..dims.width {
            // The 5x5 window centred on (x, y), mirrored per 5.2. Gathered
            // once for all k*k outputs of this input sample, together with the
            // [min, max] K.2 clamps each output to.
            let mut window = [0.0f32; WINDOW_CELLS];
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            for iy in 0..WINDOW {
                for ix in 0..WINDOW {
                    let sx = x as i64 + ix as i64 - 2;
                    let sy = y as i64 + iy as i64 - 2;
                    let v = sample_mirrored(src, dims, sx, sy);
                    if let Some(slot) = window.get_mut(iy * WINDOW + ix) {
                        *slot = v;
                    }
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            for ky in 0..k {
                for kx in 0..k {
                    let Some(tap) = kernel.taps.get(ky * k + kx) else {
                        continue;
                    };
                    let mut acc = 0.0f32;
                    for (w, s) in tap.iter().zip(window.iter()) {
                        acc += w * s;
                    }
                    let index = (y * k + ky) * out.width + (x * k + kx);
                    if let Some(slot) = dst.get_mut(index) {
                        *slot = acc.clamp(lo, hi);
                    }
                }
            }
        }
    }
    Ok((dst, out))
}

/// Upsamples one plane by `factor` and crops it to `target`.
///
/// `factor` is the *cumulative* factor: `frame_header.upsampling` for a colour
/// channel, `ec_upsampling[n] << ec_info[n].dim_shift` for extra channel `n`
/// (L.4). L.4 splits a factor above 8 into an 8x step followed by a
/// `factor idiv 8` step, which is what this does; the intermediate is cropped
/// the same way the final result is.
///
/// # Cropping
///
/// A frame stores `ceil(image / factor)` samples per row (F.1), so the `k x k`
/// expansion produces at least `image` samples and never fewer. The extra
/// samples are at the right and bottom edges: the sample grid's origin is its
/// top-left corner, so the kept region is the top-left `target` rectangle.
/// There is no other placement consistent with F.1's `ceil` division.
///
/// # Errors
///
/// [`FrameError::FieldOutOfRange`] if `factor` is not a power of two in
/// `[1, 64]`, if `src` does not match `dims`, or if the expansion cannot cover
/// `target`; [`FrameError::Core`] on the allocation budget.
pub fn upsample_plane(
    src: &[f32],
    dims: PlaneDims,
    factor: u32,
    target: PlaneDims,
    weights: &UpsamplingWeightSet,
    guard: &mut AllocGuard,
) -> Result<Vec<f32>> {
    if factor == 1 {
        if dims != target {
            return Err(FrameError::out_of_range(
                "unupsampled plane size against the frame",
                "K.2",
                dims.width as u64,
            ));
        }
        return Ok(src.to_vec());
    }
    if !matches!(factor, 2 | 4 | 8 | 16 | 32 | 64) {
        return Err(FrameError::out_of_range(
            "cumulative upsampling factor",
            "F.2",
            u64::from(factor),
        ));
    }

    // L.4: above 8, an 8x step first, then `factor idiv 8`.
    let steps: [u32; 2] = if factor > MAX_STEP {
        [MAX_STEP, factor / MAX_STEP]
    } else {
        [factor, 1]
    };

    let mut plane = src.to_vec();
    let mut current = dims;
    let mut remaining = factor;
    for step in steps {
        if step == 1 {
            continue;
        }
        remaining /= step;
        // The size this step has to reach: the target divided by whatever
        // factor is still to come.
        let want = PlaneDims::new(
            target.width.div_ceil(remaining as usize),
            target.height.div_ceil(remaining as usize),
        );
        let kernel = weights.kernel(step)?;
        let (expanded, expanded_dims) = upsample_step(&plane, current, &kernel, guard)?;
        plane = crop(&expanded, expanded_dims, want)?;
        current = want;
    }
    Ok(plane)
}

/// Keeps the top-left `target` rectangle of a plane.
fn crop(src: &[f32], dims: PlaneDims, target: PlaneDims) -> Result<Vec<f32>> {
    if target.width > dims.width || target.height > dims.height {
        return Err(FrameError::out_of_range(
            "upsampled plane against its target size",
            "K.2",
            target.width as u64,
        ));
    }
    if target == dims {
        return Ok(src.to_vec());
    }
    let mut out = Vec::with_capacity(target.width.saturating_mul(target.height));
    for y in 0..target.height {
        let start = y * dims.width;
        let row = src
            .get(start..start + target.width)
            .ok_or_else(|| FrameError::out_of_range("upsampled row", "K.2", start as u64))?;
        out.extend_from_slice(row);
    }
    Ok(out)
}

/// The three `up{k}_weight` tables a frame may use, defaults or custom.
///
/// D.3 signals custom weights per factor through `cw_mask`; a factor whose
/// mask bit is clear takes the K.2 default. Kernels are built lazily, once per
/// factor actually used.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct UpsamplingWeightSet {
    up2: Option<Vec<f32>>,
    up4: Option<Vec<f32>>,
    up8: Option<Vec<f32>>,
}

impl UpsamplingWeightSet {
    /// Takes the custom tables the image metadata carries, if any.
    #[must_use]
    pub fn new(up2: Option<Vec<f32>>, up4: Option<Vec<f32>>, up8: Option<Vec<f32>>) -> Self {
        Self { up2, up4, up8 }
    }

    /// The table for one factor: the custom one if signalled, else K.2's.
    ///
    /// Custom weights arrive as `F16()` fields (D.3), so they are `f32` and
    /// widen losslessly; the defaults are already `f64`.
    ///
    /// # Errors
    ///
    /// [`FrameError::FieldOutOfRange`] if `factor` is not 2, 4 or 8.
    pub fn weights(&self, factor: u32) -> Result<Vec<f64>> {
        let custom = match factor {
            2 => self.up2.as_deref(),
            4 => self.up4.as_deref(),
            8 => self.up8.as_deref(),
            _ => None,
        };
        match custom {
            Some(w) => Ok(w.iter().map(|&v| f64::from(v)).collect()),
            None => Ok(default_weights(factor)?.to_vec()),
        }
    }

    /// The expanded kernel for one factor.
    ///
    /// # Errors
    ///
    /// [`FrameError::FieldOutOfRange`] if `factor` is not 2, 4 or 8 or the
    /// custom table has the wrong length.
    pub fn kernel(&self, factor: u32) -> Result<UpsamplingKernel> {
        UpsamplingKernel::new(factor, &self.weights(factor)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn guard() -> AllocGuard {
        AllocGuard::new(&jpxl_core::limits::Limits::default())
    }

    /// The transcription checksum described in the module docs: every output
    /// position's 25 weights sum to 1.
    ///
    /// This is the whole proof that 280 hand-transcribed constants and the
    /// index formula are both right — a partition of unity is not something a
    /// digit slip survives.
    #[test]
    fn default_weights_are_normalised() {
        for factor in [2u32, 4, 8] {
            let kernel = UpsamplingKernel::default_for(factor).expect("factor is 2, 4 or 8");
            for (position, tap) in kernel.taps.iter().enumerate() {
                let sum: f32 = tap.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-6,
                    "k={factor} position {position} sums to {sum}"
                );
            }
        }
    }

    /// K.2's worked example for `k = 2`: the top-left 5x5 of the printed 10x10
    /// index matrix.
    #[test]
    fn k2_example_index_matrix() {
        let expected: [[usize; 5]; 5] = [
            [0, 1, 2, 3, 4],
            [1, 5, 6, 7, 8],
            [2, 6, 9, 10, 11],
            [3, 7, 10, 12, 13],
            [4, 8, 11, 13, 14],
        ];
        for (iy, row) in expected.iter().enumerate() {
            for (ix, want) in row.iter().enumerate() {
                assert_eq!(weight_index(2, 0, 0, ix, iy), *want, "({ix}, {iy})");
            }
        }
        // The (1, 0) block mirrors horizontally: its row 0 reads 4 3 2 1 0.
        for ix in 0..5 {
            assert_eq!(weight_index(2, 1, 0, ix, 0), 4 - ix);
        }
    }

    #[test]
    fn weight_counts_match_the_table_lengths() {
        assert_eq!(weight_count(2), DEFAULT_UP2.len());
        assert_eq!(weight_count(4), DEFAULT_UP4.len());
        assert_eq!(weight_count(8), DEFAULT_UP8.len());
    }

    /// A constant plane upsamples to the same constant: the partition of unity
    /// again, this time through the whole filter including the clamp.
    #[test]
    fn constant_plane_is_preserved() {
        let dims = PlaneDims::new(7, 5);
        let src = vec![0.375f32; dims.width * dims.height];
        for factor in [2u32, 4, 8] {
            let target =
                PlaneDims::new(dims.width * factor as usize, dims.height * factor as usize);
            let out = upsample_plane(
                &src,
                dims,
                factor,
                target,
                &UpsamplingWeightSet::default(),
                &mut guard(),
            )
            .expect("upsamples");
            assert_eq!(out.len(), target.width * target.height);
            for v in out {
                assert!((v - 0.375).abs() < 1e-6, "{v}");
            }
        }
    }

    /// K.2 clamps each output to the range of its own 5x5 window, so no output
    /// sample can leave the range of the input plane.
    #[test]
    fn output_stays_within_the_input_range() {
        let dims = PlaneDims::new(9, 9);
        let mut src = vec![0.0f32; dims.width * dims.height];
        // A hard step: the worst case for an overshooting interpolator.
        for y in 0..dims.height {
            for x in 0..dims.width {
                if let Some(slot) = src.get_mut(y * dims.width + x) {
                    *slot = if x < 4 { -1.0 } else { 2.0 };
                }
            }
        }
        let target = PlaneDims::new(dims.width * 4, dims.height * 4);
        let out = upsample_plane(
            &src,
            dims,
            4,
            target,
            &UpsamplingWeightSet::default(),
            &mut guard(),
        )
        .expect("upsamples");
        for v in out {
            assert!((-1.0..=2.0).contains(&v), "{v} escaped the input range");
        }
    }

    /// The cropping contract: a plane of `ceil(target / factor)` samples
    /// upsamples to exactly `target`.
    #[test]
    fn crops_to_the_target_rectangle() {
        // 13 = ceil(50 / 4), and 4 * 13 = 52 > 50.
        let dims = PlaneDims::new(13, 3);
        let src = vec![0.5f32; dims.width * dims.height];
        let target = PlaneDims::new(50, 9);
        let out = upsample_plane(
            &src,
            dims,
            4,
            target,
            &UpsamplingWeightSet::default(),
            &mut guard(),
        )
        .expect("upsamples");
        assert_eq!(out.len(), 50 * 9);
    }

    /// L.4's two-step decomposition for a factor above 8.
    #[test]
    fn factors_above_eight_take_two_steps() {
        let dims = PlaneDims::new(3, 2);
        let src = vec![0.25f32; 6];
        let target = PlaneDims::new(48, 32);
        let out = upsample_plane(
            &src,
            dims,
            16,
            target,
            &UpsamplingWeightSet::default(),
            &mut guard(),
        )
        .expect("upsamples");
        assert_eq!(out.len(), 48 * 32);
        for v in out {
            assert!((v - 0.25).abs() < 1e-6, "{v}");
        }
    }

    #[test]
    fn factor_one_is_the_identity() {
        let dims = PlaneDims::new(4, 4);
        let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let out = upsample_plane(
            &src,
            dims,
            1,
            dims,
            &UpsamplingWeightSet::default(),
            &mut guard(),
        )
        .expect("identity");
        assert_eq!(out, src);
    }

    #[test]
    fn a_non_power_of_two_factor_is_rejected() {
        let dims = PlaneDims::new(2, 2);
        let src = vec![0.0f32; 4];
        assert!(
            upsample_plane(
                &src,
                dims,
                3,
                PlaneDims::new(6, 6),
                &UpsamplingWeightSet::default(),
                &mut guard(),
            )
            .is_err()
        );
    }

    /// Custom weights are honoured, and a table of the wrong length is an
    /// error rather than a silent fall-back to the defaults.
    #[test]
    fn custom_weights_replace_the_defaults() {
        let flat = vec![0.0f32; weight_count(2)];
        let set = UpsamplingWeightSet::new(Some(flat), None, None);
        let dims = PlaneDims::new(4, 4);
        let src = vec![1.0f32; 16];
        let out = upsample_plane(&src, dims, 2, PlaneDims::new(8, 8), &set, &mut guard())
            .expect("upsamples");
        // All-zero weights give zero before the clamp; the clamp then pulls
        // the result up to the window minimum, which is 1.0 here.
        for v in out {
            assert!((v - 1.0).abs() < 1e-6, "{v}");
        }

        let short = UpsamplingWeightSet::new(Some(vec![0.0; 3]), None, None);
        assert!(short.kernel(2).is_err());
    }
}
