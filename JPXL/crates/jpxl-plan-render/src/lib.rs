//! Reconstructs the pixels a conforming decoder will produce from a validated
//! VarDCT pixel plan — on the encoder side, from the plan alone.
//!
//! A perceptual quality search needs reconstructed pixels for every probe.
//! Emitting a codestream and decoding it would pay for entropy training, ANS
//! tables, section layout and a full parse that the probe does not need, so
//! this crate runs the reconstruction directly on the plan's integers:
//!
//! ```text
//! ValidatedPixelPlan
//!   ├─ I.5.2  LF dequantization, I.6 LF chroma-from-luma
//!   ├─ I.5.3  HF dequantization        ─┐
//!   ├─ I.6    HF chroma-from-luma       │ per varblock
//!   ├─ I.8    LLF from the LF planes    │
//!   ├─ I.9    inverse transform        ─┘
//!   ├─ J.3    Gabor-like transform, J.4 edge-preserving filter
//!   ├─ L.2.2  XYB → linear sRGB
//!   └─ transfer function, then optional quantization to the frame's depth
//! ```
//!
//! # Boundary
//!
//! This crate is encoder-side. It never calls `jpxl-decode`; the kernels it
//! shares with the decoder live in `jpxl-core` (`dct`, `varblock`,
//! `dequant`, `color`, `reconstruct`, `restoration`), and the orchestration
//! here is its own. Agreement with `jpxl-decode` and the external decoders is
//! established by the parity tests, which is what keeps an encoder-side bug
//! from being accepted by its paired decoder.
//!
//! # What it does not model
//!
//! Adaptive LF smoothing (the encoder always signals
//! `kSkipAdaptiveLFSmoothing`), upsampling, patches, noise, splines, extra
//! channels, and non-sRGB output encodings. A plan asking for any of these
//! is refused with [`RenderError::Unsupported`] rather than rendered wrongly.

use jpxl_core::JpxlError;
use jpxl_core::color::{OpsinInverse, linear_to_srgb, srgb_to_linear};
use jpxl_core::dequant::{DequantMatrices, DequantMatrix};
use jpxl_core::reconstruct::{
    LF_WEIGHT_SCALE, bias_adjust, cfl_apply, cfl_factors, hf_multiplier, lf_dequantize,
    lf_multipliers, qm_multiplier,
};
use jpxl_core::restoration::{
    EPF_PAD, EpfParams, GaborKernel, PaddedPlane, PlaneDims, SigmaField, block_grid, epf_step_rows,
    epf_steps, gaborish_into, vardct_sigma,
};
use jpxl_core::varblock::{CoeffMatrix, SampleBlock, TransformType, llf_from_lf};
use jpxl_encode::EncodeExecutor;
use jpxl_encode::vardct::plan::NUM_CHANNELS;
use jpxl_encode::vardct::{PlanError, ValidatedPixelPlan, VardctGeometry};

/// Table L.1's default `OpsinInverseMatrix`, row-major, as the decoder
/// reads it for an `all_default` bundle.
const DEFAULT_INVERSE_MATRIX: [f32; 9] = [
    11.031_567,
    -9.866_944,
    -0.164_622_99,
    -3.254_147_4,
    4.418_770_5,
    -0.164_622_99,
    -3.658_851_3,
    2.712_923,
    1.945_928_2,
];

/// Table L.1's default `opsin_bias`, as signalled (negative).
const DEFAULT_OPSIN_BIAS: [f32; 3] = [-0.003_793_073_3; 3];

/// Table L.1's default `quant_bias`.
const DEFAULT_QUANT_BIAS: [f32; 3] = jpxl_core::color::DEFAULT_QUANT_BIAS;

/// Table L.1's default `quant_bias_numerator`.
const DEFAULT_QUANT_BIAS_NUMERATOR: f32 = jpxl_core::color::DEFAULT_QUANT_BIAS_NUMERATOR;

/// Side, in samples, of the tiles the HF chroma-from-luma factors cover.
const CFL_TILE_BLOCKS: u32 = 8;

/// Why a plan could not be rendered.
#[derive(Debug)]
pub enum RenderError {
    /// The plan's geometry or structure was rejected.
    Plan(PlanError),
    /// A core primitive failed (a dequantization matrix, a limit).
    Core(JpxlError),
    /// The plan uses a feature this renderer does not model.
    Unsupported(&'static str),
}

impl core::fmt::Display for RenderError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Plan(e) => write!(f, "plan: {e}"),
            Self::Core(e) => write!(f, "core: {e}"),
            Self::Unsupported(what) => write!(f, "unsupported by the plan renderer: {what}"),
        }
    }
}

impl std::error::Error for RenderError {}

impl From<PlanError> for RenderError {
    fn from(e: PlanError) -> Self {
        Self::Plan(e)
    }
}

impl From<JpxlError> for RenderError {
    fn from(e: JpxlError) -> Self {
        Self::Core(e)
    }
}

/// The crate's result type.
pub type Result<T> = core::result::Result<T, RenderError>;

/// A rendered frame: three planes in the signalled (sRGB-encoded) colour
/// encoding, unclipped, exactly what a decoder holds before quantizing to
/// the frame's bit depth.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderedFrame {
    width: u32,
    height: u32,
    bits_per_sample: u32,
    planes: [Vec<f32>; NUM_CHANNELS],
}

impl RenderedFrame {
    /// Width in samples.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Height in samples.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// The frame's signalled bit depth.
    #[must_use]
    pub const fn bits_per_sample(&self) -> u32 {
        self.bits_per_sample
    }

    /// The sRGB-encoded planes, `[R, G, B]`, unclipped.
    #[must_use]
    pub fn encoded_planes(&self) -> [&[f32]; NUM_CHANNELS] {
        self.planes.each_ref().map(Vec::as_slice)
    }

    /// Full scale of a `bits` deep integer sample.
    fn full_scale(bits: u32) -> f32 {
        if bits >= 32 {
            f32::from(u16::MAX)
        } else {
            ((1u32 << bits.max(1)) - 1) as f32
        }
    }

    fn linear_lut(bits: u32, max: f32) -> Vec<f32> {
        // One transfer-curve evaluation per representable integer, not per
        // sample: the round trip is a table lookup for every real bit depth.
        let entries = if bits >= 32 {
            65_536
        } else {
            1usize << bits.clamp(1, 16)
        };
        (0..entries)
            .map(|q| srgb_to_linear(q as f32 / max))
            .collect()
    }

    fn linear_sample(v: f32, max: f32, lut: &[f32]) -> f32 {
        let scaled = (v * max).round();
        let q = if scaled.is_finite() {
            // Clamped into [0, max] with max < 2^32 before the cast, so the
            // narrowing is exact.
            #[allow(
                clippy::cast_possible_truncation,
                reason = "the value is clamped to [0, max] first"
            )]
            let q = scaled.clamp(0.0, max) as i32;
            q
        } else {
            0
        };
        usize::try_from(q)
            .ok()
            .and_then(|q| lut.get(q))
            .copied()
            .unwrap_or_else(|| srgb_to_linear(q as f32 / max))
    }

    /// The composed depth quantizer: the integer level that
    /// [`Self::linear_sample`] forms from `linear_to_srgb(x)` — the sRGB
    /// encode, the scale by `max`, the round, the finiteness guard and the
    /// clamp, in that order. Kept callable on its own so threshold
    /// construction and its verification bisect the actual function.
    fn composed_level(x: f32, max: f32) -> usize {
        let scaled = (linear_to_srgb(x) * max).round();
        if scaled.is_finite() {
            // Clamped into [0, max] with max < 2^32 before the cast, so the
            // narrowing is exact.
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "the value is clamped to [0, max] first"
            )]
            let q = scaled.clamp(0.0, max) as u32;
            usize::try_from(q).unwrap_or(0)
        } else {
            0
        }
    }

    /// The planes quantized to `bits` per sample exactly as the decoder's
    /// integer output is: scaled, rounded and clamped.
    #[must_use]
    pub fn quantized(&self, bits: u32) -> [Vec<i32>; NUM_CHANNELS] {
        let max = Self::full_scale(bits);
        self.planes.each_ref().map(|plane| {
            plane
                .iter()
                .map(|&v| {
                    let scaled = (v * max).round();
                    if scaled.is_finite() {
                        // Clamped into [0, max] with max < 2^32 before the
                        // cast, so the narrowing is exact.
                        #[allow(
                            clippy::cast_possible_truncation,
                            reason = "the value is clamped to [0, max] first"
                        )]
                        let q = scaled.clamp(0.0, max) as i32;
                        q
                    } else {
                        0
                    }
                })
                .collect()
        })
    }

    /// Linear-sRGB planes after a round trip through `bits`-deep integer
    /// samples — the image a viewer of the decoded file actually sees, and
    /// therefore what a perceptual metric should score.
    #[must_use]
    pub fn linear_rgb_at_depth(&self, bits: u32) -> [Vec<f32>; NUM_CHANNELS] {
        let mut out: [Vec<f32>; NUM_CHANNELS] = [Vec::new(), Vec::new(), Vec::new()];
        self.linear_rgb_at_depth_into(bits, &mut out);
        out
    }

    /// [`Self::linear_rgb_at_depth`] writing into caller-owned buffers, so a
    /// search that scores many candidates reuses one set of planes instead of
    /// allocating three per probe. Each output is cleared and refilled to
    /// exactly `width * height` samples.
    ///
    /// The quantized integer is formed and looked up per sample in one pass,
    /// so no full-frame `i32` plane is materialised; the values are identical
    /// to `linear_rgb_at_depth` (and to [`Self::quantized`] followed by the
    /// same lookup) sample for sample.
    pub fn linear_rgb_at_depth_into(&self, bits: u32, out: &mut [Vec<f32>; NUM_CHANNELS]) {
        let max = Self::full_scale(bits);
        let lut = Self::linear_lut(bits, max);
        for (dst, plane) in out.iter_mut().zip(self.planes.iter()) {
            dst.clear();
            dst.reserve(plane.len());
            dst.extend(plane.iter().map(|&v| Self::linear_sample(v, max, &lut)));
        }
    }

    /// Consumes the rendered frame and converts its three allocated planes in
    /// place to the same decoded linear-sRGB samples as
    /// [`Self::linear_rgb_at_depth`].
    ///
    /// This is the low-peak-memory form for a caller that no longer needs the
    /// encoded planes: it never has both the rendered and linear full-frame
    /// planes resident at once.
    #[must_use]
    pub fn into_linear_rgb_at_depth(mut self, bits: u32) -> [Vec<f32>; NUM_CHANNELS] {
        let max = Self::full_scale(bits);
        let lut = Self::linear_lut(bits, max);
        for plane in &mut self.planes {
            for value in plane {
                *value = Self::linear_sample(*value, max, &lut);
            }
        }
        self.planes
    }

    /// [`Self::into_linear_rgb_at_depth`] with the per-sample transfer-curve
    /// lookup banded over `executor`'s workers.
    ///
    /// The lookup is an independent per-sample map, so each fixed row band is
    /// converted on its own and the output is identical to the serial form at
    /// any worker count. This is the large-frame scoring path, where the
    /// 36-million-sample round trip is otherwise a serial tail on every probe.
    #[must_use]
    pub fn into_linear_rgb_at_depth_with(
        mut self,
        bits: u32,
        executor: &EncodeExecutor,
    ) -> [Vec<f32>; NUM_CHANNELS] {
        let max = Self::full_scale(bits);
        let lut = Self::linear_lut(bits, max);
        let width = usize::try_from(self.width).unwrap_or(usize::MAX).max(1);
        let bands = row_bands(&mut self.planes, width);
        run_items(Some(executor), bands.len(), &|index| {
            let Some((_, mut slices)) = bands.take(index) else {
                return;
            };
            for plane in slices.iter_mut() {
                for value in plane.iter_mut() {
                    *value = Self::linear_sample(*value, max, &lut);
                }
            }
        });
        self.planes
    }
}

/// Linear-domain thresholds of the composed depth quantizer for `levels`
/// integer levels at full scale `max`: entry `q - 1` is the smallest `f32`
/// whose [`RenderedFrame::composed_level`] is at least `q`.
///
/// Each boundary is found by bisection over the non-negative `f32` bit
/// lattice against `composed_level` itself, so whatever the transfer curve's
/// branches and roundings do, the boundary is the actual function's; the
/// level function is monotone because every stage of the composition is.
/// Both sides of every boundary are asserted after the search.
fn quantize_thresholds(max: f32, levels: usize) -> Vec<f32> {
    let mut thresholds = Vec::with_capacity(levels.saturating_sub(1));
    for q in 1..levels {
        // 2.0 encodes above full scale, so every boundary sits below it, and
        // non-negative `f32` bit patterns order exactly as their values.
        let (mut lo, mut hi) = (0u32, 2.0f32.to_bits());
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if RenderedFrame::composed_level(f32::from_bits(mid), max) >= q {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        let t = f32::from_bits(lo);
        debug_assert!(RenderedFrame::composed_level(t, max) >= q);
        debug_assert!(lo == 0 || RenderedFrame::composed_level(f32::from_bits(lo - 1), max) < q);
        thresholds.push(t);
    }
    thresholds
}

/// The composed level of one linear sample: how many thresholds it reaches.
/// Non-finite samples take level 0, exactly as the round trip's finiteness
/// guard sends them there; negative samples sit below every threshold. The
/// reference form — [`DepthClassifier`] reproduces it with a guide table.
#[inline]
fn level_of(x: f32, thresholds: &[f32]) -> usize {
    if !x.is_finite() {
        return 0;
    }
    thresholds.partition_point(|&t| t <= x)
}

/// The composed depth quantizer as a lookup structure: the level thresholds,
/// the levels' linear values, and a guide over the non-negative `f32` bit
/// lattice pinning each bucket's samples to a narrow level range, so one
/// sample's classification is two loads and at most a short ordered scan.
#[derive(Debug)]
struct DepthClassifier {
    bits: u32,
    thresholds: Vec<f32>,
    lut: Vec<f32>,
    /// Per bucket, the levels of the bucket's smallest and largest values.
    guide: Vec<(u32, u32)>,
    /// `bit pattern >> shift` is the bucket of a value in `[0, 2.0)`.
    shift: u32,
}

/// The bit pattern of `2.0f32`, one past the last guided bucket: every
/// threshold lies strictly below 2.0 (full scale encodes below it), so any
/// larger sample takes the top level directly.
const DEPTH_GUIDE_END: u32 = 0x4000_0000;

impl DepthClassifier {
    fn new(bits: u32) -> Self {
        let max = RenderedFrame::full_scale(bits);
        let lut = RenderedFrame::linear_lut(bits, max);
        let thresholds = quantize_thresholds(max, lut.len());
        let buckets = (lut.len() * 8).next_power_of_two().clamp(4096, 65_536);
        let shift = DEPTH_GUIDE_END.trailing_zeros() - buckets.trailing_zeros();
        let guide = (0..buckets)
            .map(|b| {
                let lo_bits = u32::try_from(b).unwrap_or(0) << shift;
                let hi_bits = lo_bits + ((1u32 << shift) - 1);
                let lo = level_of(f32::from_bits(lo_bits), &thresholds);
                let hi = level_of(f32::from_bits(hi_bits), &thresholds);
                (
                    u32::try_from(lo).unwrap_or(u32::MAX),
                    u32::try_from(hi).unwrap_or(u32::MAX),
                )
            })
            .collect();
        Self {
            bits,
            thresholds,
            lut,
            guide,
            shift,
        }
    }

    /// The level's linear value for one sample: `lut[level_of(x)]` for every
    /// `f32`, non-finite and negative included, by way of the guide.
    #[inline]
    fn linear_at_depth(&self, x: f32) -> f32 {
        if !x.is_finite() {
            return self.lut.first().copied().unwrap_or(0.0);
        }
        let pattern = x.to_bits();
        if pattern >= 0x8000_0000 {
            // Negative, -0.0 included: below every (positive) threshold.
            return self.lut.first().copied().unwrap_or(0.0);
        }
        if pattern >= DEPTH_GUIDE_END {
            // At least 2.0: above every threshold.
            return self.lut.last().copied().unwrap_or(0.0);
        }
        let (lo, hi) = self
            .guide
            .get((pattern >> self.shift) as usize)
            .copied()
            .unwrap_or((0, 0));
        let mut level = lo as usize;
        for &t in self.thresholds.get(lo as usize..hi as usize).unwrap_or(&[]) {
            if t <= x {
                level += 1;
            } else {
                break;
            }
        }
        self.lut.get(level).copied().unwrap_or(0.0)
    }
}

/// Wall time of one render's stages, in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RenderTimings {
    /// LF dequantization, HF dequantization, CfL, LLF and inverse transforms.
    pub varblocks_ms: u64,
    /// J.3 Gabor-like transform.
    pub gaborish_ms: u64,
    /// J.4 edge-preserving filter, all steps.
    pub epf_ms: u64,
    /// XYB to linear sRGB and the transfer function.
    pub colour_ms: u64,
}

/// Full-frame working storage reused across renders, so a probe search pays
/// the frame-sized allocations (and their first-touch page faults) once, not
/// per probe. Every buffer is either zero-filled on takeout or provably
/// fully overwritten before it is read, so reuse cannot change a sample.
#[derive(Debug, Default)]
struct RenderScratch {
    /// Recycled output planes; zero-filled by [`Self::take_planes`].
    planes: [Vec<f32>; NUM_CHANNELS],
    /// Ping-pong partner of the output planes for the gaborish and EPF
    /// stages. Never zeroed: both stages write every sample of every band.
    aux: [Vec<f32>; NUM_CHANNELS],
    /// Backing storage for the EPF's padded input planes
    /// ([`PaddedPlane::new_in`] writes every padded sample).
    padded: [Vec<f32>; NUM_CHANNELS],
    /// Per-block sigma field; zero-filled by [`Self::take_sigma`].
    sigma: Vec<f32>,
    /// The chunk's varblock slots (at most [`VARBLOCK_CHUNK`]), each fully
    /// overwritten by the varblock that takes it.
    varblocks: Vec<RenderedVarblock>,
}

/// Shapes `buf` to `len` zeros. An empty buffer is replaced by `vec![0.0]`'s
/// zeroed allocation — the kernel's lazily zeroed pages, never an eager
/// memset over hundreds of megabytes — so the released-scratch (large-frame)
/// path costs exactly what a fresh allocation always cost.
fn zeroed_in_place(buf: &mut Vec<f32>, len: usize) {
    if buf.is_empty() {
        *buf = vec![0.0f32; len];
    } else {
        buf.clear();
        buf.resize(len, 0.0);
    }
}

/// Shapes `buf` to `len` samples of unspecified contents, for a consumer
/// that overwrites every sample before any is read; the same lazy zeroed
/// allocation as [`zeroed_in_place`] when the buffer starts empty.
fn shaped_in_place(buf: &mut Vec<f32>, len: usize) {
    if buf.is_empty() {
        *buf = vec![0.0f32; len];
    } else {
        buf.truncate(len);
        buf.resize(len, 0.0);
    }
}

impl RenderScratch {
    /// Three zeroed `len`-long planes, reusing recycled allocations.
    fn take_planes(&mut self, len: usize) -> [Vec<f32>; NUM_CHANNELS] {
        let mut planes = core::mem::take(&mut self.planes);
        for plane in &mut planes {
            zeroed_in_place(plane, len);
        }
        planes
    }

    /// Three `len`-long planes of unspecified contents, for a stage that
    /// overwrites every sample before any is read.
    fn take_aux(&mut self, len: usize) -> [Vec<f32>; NUM_CHANNELS] {
        let mut aux = core::mem::take(&mut self.aux);
        for plane in &mut aux {
            shaped_in_place(plane, len);
        }
        aux
    }

    /// A zeroed `len`-long sigma buffer.
    fn take_sigma(&mut self, len: usize) -> Vec<f32> {
        let mut sigma = core::mem::take(&mut self.sigma);
        zeroed_in_place(&mut sigma, len);
        sigma
    }
}

/// Reusable renderer: the dequantization matrices and the opsin inverse are
/// built once and shared across every plan rendered.
#[derive(Debug)]
pub struct PlanRenderer {
    matrices: DequantMatrices,
    cache: Vec<Option<[DequantMatrix; NUM_CHANNELS]>>,
    opsin: OpsinInverse,
    epf_params: EpfParams,
    gabor: GaborKernel,
    /// The depth quantizer of the last [`Self::render_linear_at_depth_with`]
    /// call, kept so a probe search bisects its thresholds once, not per
    /// probe.
    depth: Option<DepthClassifier>,
    /// Frame-sized buffers reused across renders.
    scratch: RenderScratch,
}

impl PlanRenderer {
    /// A renderer over the Table I.6 default dequantization matrices and the
    /// Table L.1 default opsin inverse.
    ///
    /// # Errors
    ///
    /// [`RenderError::Core`] if the default matrices cannot be built.
    pub fn new() -> Result<Self> {
        Ok(Self {
            matrices: DequantMatrices::all_default()?,
            cache: vec![None; jpxl_core::varblock::NUM_DEQUANT_MATRICES],
            opsin: OpsinInverse::new(
                DEFAULT_INVERSE_MATRIX,
                DEFAULT_OPSIN_BIAS,
                jpxl_core::color::NOMINAL_INTENSITY_TARGET,
            ),
            epf_params: EpfParams::default(),
            gabor: GaborKernel::defaults(),
            depth: None,
            scratch: RenderScratch::default(),
        })
    }

    /// Hands a rendered frame's planes back for reuse by the next render.
    /// Purely an allocation recycler: the planes' contents are never read.
    pub fn recycle_planes(&mut self, planes: [Vec<f32>; NUM_CHANNELS]) {
        self.scratch.planes = planes;
    }

    /// Frees the frame-sized scratch buffers. For the large-frame search
    /// path, where holding several spare full-resolution planes between
    /// probes would raise the search's resident peak. The varblock slot
    /// arena is kept: it is a few tens of megabytes at most, and re-growing
    /// its thousands of small buffers every probe measurably costs page
    /// faults that retaining it does not.
    pub fn release_scratch(&mut self) {
        let varblocks = core::mem::take(&mut self.scratch.varblocks);
        self.scratch = RenderScratch {
            varblocks,
            ..RenderScratch::default()
        };
    }

    /// The three dequantization matrices for a transform, built on first use.
    fn matrices_for(&mut self, transform: TransformType) -> Result<&[DequantMatrix; NUM_CHANNELS]> {
        let index = transform.dequant_matrix_index();
        let slot = self.cache.get_mut(index).ok_or(RenderError::Unsupported(
            "a dequantization matrix index past Table I.4",
        ))?;
        if slot.is_none() {
            *slot = Some([
                self.matrices.matrix(index, 0)?,
                self.matrices.matrix(index, 1)?,
                self.matrices.matrix(index, 2)?,
            ]);
        }
        slot.as_ref().ok_or(RenderError::Unsupported(
            "a dequantization matrix that failed to build",
        ))
    }

    /// Renders `pixels` to the signalled colour encoding.
    ///
    /// # Errors
    ///
    /// [`RenderError::Unsupported`] for a plan using a feature this renderer
    /// does not model (see the crate docs); [`RenderError::Plan`] or
    /// [`RenderError::Core`] for a structural failure.
    pub fn render(&mut self, pixels: &ValidatedPixelPlan) -> Result<RenderedFrame> {
        self.render_timed(pixels, None).map(|(frame, _)| frame)
    }

    /// [`Self::render`] with the restoration filters and the colour
    /// transform banded over `executor`'s workers. Output is identical to
    /// the serial render: every band is a fixed row range and no stage
    /// reduces across bands.
    ///
    /// # Errors
    ///
    /// As [`Self::render`].
    pub fn render_with(
        &mut self,
        pixels: &ValidatedPixelPlan,
        executor: &EncodeExecutor,
    ) -> Result<RenderedFrame> {
        self.render_timed(pixels, Some(executor))
            .map(|(frame, _)| frame)
    }

    /// [`Self::render_with`] fused with
    /// [`RenderedFrame::into_linear_rgb_at_depth_with`]: the linear-sRGB
    /// planes after the round trip through `bits`-deep integer samples,
    /// without ever materialising the signalled sRGB encoding. Each sample's
    /// quantized level is found in linear light through
    /// [`quantize_thresholds`], so every output is bit-identical to rendering
    /// and round-tripping in two steps — the equivalence across the whole
    /// curve, both branch seams and non-finite samples is pinned by
    /// `fused_depth_levels_match_the_srgb_round_trip`.
    ///
    /// This is the scoring path: a perceptual probe wants only these planes,
    /// and the two-step path paid a full-frame `powf` encode per sample just
    /// to quantize away its result.
    ///
    /// # Errors
    ///
    /// As [`Self::render`].
    pub fn render_linear_at_depth_with(
        &mut self,
        pixels: &ValidatedPixelPlan,
        bits: u32,
        executor: &EncodeExecutor,
    ) -> Result<(u32, u32, [Vec<f32>; NUM_CHANNELS])> {
        let classifier = match self.depth.take() {
            Some(classifier) if classifier.bits == bits => classifier,
            _ => DepthClassifier::new(bits),
        };
        let rendered = self.render_inner(pixels, Some(executor), Some(&classifier), false);
        self.depth = Some(classifier);
        let (frame, _) = rendered?;
        Ok((frame.width, frame.height, frame.planes))
    }

    /// The Phase S3 surrogate render: reconstruct every varblock's own 2:1
    /// box average directly in the coefficient domain (the fold is an exact
    /// identity, so quantization detail still reaches the observation), then
    /// run the restoration filters and the colour/depth stage at half
    /// resolution. No full-resolution pixel plane ever exists on this path.
    /// Returns the half-resolution dimensions and depth-quantized linear
    /// planes.
    ///
    /// This is *not* the canonical render downscaled: the filters act at the
    /// half scale (with a 2×2-averaged sigma field), and at an odd frame
    /// dimension the last half-res row/column averages the varblock's own
    /// out-of-frame reconstruction rather than replicating the edge, so the
    /// result is a cheaper, deterministic *surrogate* observation — it may
    /// propose, never accept. Output is identical at any worker count, like
    /// every other render path.
    ///
    /// # Errors
    ///
    /// As [`Self::render`].
    pub fn render_linear_at_depth_decimated_with(
        &mut self,
        pixels: &ValidatedPixelPlan,
        bits: u32,
        executor: &EncodeExecutor,
    ) -> Result<(u32, u32, [Vec<f32>; NUM_CHANNELS])> {
        let classifier = match self.depth.take() {
            Some(classifier) if classifier.bits == bits => classifier,
            _ => DepthClassifier::new(bits),
        };
        let rendered = self.render_inner(pixels, Some(executor), Some(&classifier), true);
        self.depth = Some(classifier);
        let (frame, _) = rendered?;
        Ok((frame.width, frame.height, frame.planes))
    }

    /// [`Self::render_with`], also reporting where the time went.
    ///
    /// # Errors
    ///
    /// As [`Self::render`].
    pub fn render_timed(
        &mut self,
        pixels: &ValidatedPixelPlan,
        executor: Option<&EncodeExecutor>,
    ) -> Result<(RenderedFrame, RenderTimings)> {
        self.render_inner(pixels, executor, None, false)
    }

    /// The full render pipeline. The colour stage ends in the signalled sRGB
    /// encoding by default; with `linear_at_depth` it instead classifies each
    /// linear sample against the thresholds and takes the level's linear
    /// value from the table, and the returned frame's planes hold those
    /// depth-quantized linear samples rather than the signalled encoding.
    fn render_inner(
        &mut self,
        pixels: &ValidatedPixelPlan,
        executor: Option<&EncodeExecutor>,
        linear_at_depth: Option<&DepthClassifier>,
        decimate: bool,
    ) -> Result<(RenderedFrame, RenderTimings)> {
        let mut timings = RenderTimings::default();
        let millis = |start: std::time::Instant| {
            u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX)
        };
        let stage_start = std::time::Instant::now();
        let plan = pixels.plan();
        let spatial = &*plan.spatial;
        let quantized = &*plan.quantized;
        let geometry: VardctGeometry = pixels.geometry()?;
        if spatial.lf.adaptive_smoothing {
            return Err(RenderError::Unsupported(
                "adaptive LF smoothing (the encoder always signals it skipped)",
            ));
        }
        if spatial.restoration.epf_iters > 3 {
            return Err(RenderError::Unsupported("epf_iters above 3"));
        }

        let (width, height) = (geometry.width(), geometry.height());
        let dims = PlaneDims::new(
            usize::try_from(width).unwrap_or(usize::MAX),
            usize::try_from(height).unwrap_or(usize::MAX),
        );
        let len = dims.len();
        // Phase S3 surrogate: the varblocks reconstruct straight into
        // half-resolution planes (the 2:1 box average is folded into the
        // inverse transform), so the frame buffers, the scatter, and every
        // later stage run at half size — no full-resolution pixel plane ever
        // exists on the decimated path. Sigma stays on the full-resolution
        // block grid and is 2x2-averaged after the varblock stage, exactly as
        // the S2 pixel-domain decimation did.
        let (out_width, out_height) = if decimate {
            (width.div_ceil(2), height.div_ceil(2))
        } else {
            (width, height)
        };
        let out_dims = PlaneDims::new(
            usize::try_from(out_width).unwrap_or(usize::MAX),
            usize::try_from(out_height).unwrap_or(usize::MAX),
        );
        let out_len = out_dims.len();
        let mut planes = self.scratch.take_planes(out_len);
        let (blocks_x, blocks_y) = block_grid(dims);
        let mut sigma = self.scratch.take_sigma(blocks_x * blocks_y);

        let global_scale = spatial.quantizer.global_scale.get();
        let qm = [
            qm_multiplier(spatial.quantizer.x_qm_scale.get()),
            1.0,
            qm_multiplier(spatial.quantizer.b_qm_scale.get()),
        ];
        let lf_mul = lf_multipliers(
            global_scale,
            spatial.quantizer.quant_lf.get(),
            spatial.lf.channel_dequant.map(|w| w / LF_WEIGHT_SCALE),
        );
        let corr = spatial.lf.correlation;
        let (k_x_lf, k_b_lf) = cfl_factors(
            corr.base_correlation_x,
            corr.base_correlation_b,
            corr.colour_factor,
            i32::from(corr.x_factor_lf) - 128,
            i32::from(corr.b_factor_lf) - 128,
        );

        // The chunk's varblock slots and the workers' dequant/IDCT scratch,
        // reused across chunks, groups and renders: each varblock's compute
        // fully overwrites the slot it takes, so reuse cannot change a sample.
        let mut arena = core::mem::take(&mut self.scratch.varblocks);
        let worker_scratch: std::sync::Mutex<Vec<WorkerScratch>> =
            std::sync::Mutex::new(Vec::new());

        for (group, ir) in spatial.lf_groups.iter().zip(quantized.lf_groups.iter()) {
            let rect = geometry
                .lf_group_rect(group.id)
                .ok_or(RenderError::Unsupported("an LF group outside the frame"))?;
            let blocks = ir.lf.blocks();
            let tiles = group.cfl.tiles();

            // I.5.2 + I.6 (LF): dequantize the three planes, then borrow
            // chroma from luma with the frame-wide LF factors.
            let lf = lf_planes(ir, &lf_mul, spatial.lf.extra_precision, (k_x_lf, k_b_lf))?;

            // Warm the dequantization-matrix cache for every transform this
            // group uses, so the per-varblock render below reads it through a
            // shared immutable borrow and needs no `&mut self`.
            for vb in group.blocks.iter() {
                self.matrices_for(vb.transform)?;
            }
            let matrices_cache = &self.cache;
            let epf_params = &self.epf_params;

            // I.5.3, I.6, I.8, I.9 and J.4.3 for one varblock, filling its
            // chunk slot without touching shared frame state. A varblock is a
            // pure function of its own inputs and every field of the slot is
            // overwritten, so this is byte-for-byte the serial render
            // regardless of worker count or slot history.
            let render_one =
                |i: usize, out: &mut RenderedVarblock, ws: &mut WorkerScratch| -> Result<()> {
                    let vb = group
                        .blocks
                        .get(i)
                        .ok_or(RenderError::Unsupported("a varblock index past the group"))?;
                    let coeffs = ir
                        .coefficients
                        .get(i)
                        .ok_or(RenderError::Unsupported("a varblock with no coefficients"))?;
                    let lf_at = |c: usize, bx: u32, by: u32| -> f32 {
                        if bx >= blocks.width || by >= blocks.height {
                            return 0.0;
                        }
                        let idx = usize::try_from(
                            u64::from(by) * u64::from(blocks.width) + u64::from(bx),
                        )
                        .unwrap_or(usize::MAX);
                        lf.get(c).and_then(|p| p.get(idx)).copied().unwrap_or(0.0)
                    };

                    let transform = vb.transform;
                    let (bx, by) = (vb.origin.bx(), vb.origin.by());
                    let hf_mul = vb.hf_mul.get();
                    let mul = hf_multiplier(global_scale, hf_mul);

                    // I.5.3: dequantize all three channels.
                    let matrices = matrices_cache
                        .get(transform.dequant_matrix_index())
                        .and_then(Option::as_ref)
                        .ok_or(RenderError::Unsupported(
                            "a dequantization matrix that was not warmed",
                        ))?;
                    let (rows, cols) = (transform.coeff_rows(), transform.coeff_cols());
                    // Phase S3 fast path: when every quantized coefficient in
                    // the region the half-resolution fold reaches only
                    // through its partner term is zero — across all three
                    // channels, because CfL mixes Y into X and B — the
                    // dequantization, the CfL pass and the inverse all
                    // confine themselves to the low quarter. Bit-exact: a
                    // zero quantized cell dequantizes to `+0.0` (zero times
                    // positive factors) and CfL over zeros is zeros, so the
                    // kernel's own low-pass predicate then holds on exactly
                    // the matrix the unhoisted path would have built.
                    let (hr, hc) = (rows / 2, cols / 2);
                    let lowpass =
                        decimate
                            && transform.dct_shape().is_some()
                            && (0..NUM_CHANNELS).all(|c| {
                                coeffs.channel(c).is_some_and(|quant| {
                                    quant.chunks_exact(cols).enumerate().take(rows).all(
                                        |(y, row)| {
                                            row.iter()
                                                .enumerate()
                                                .all(|(x, &q)| (x < hc && y < hr) || q == 0)
                                        },
                                    )
                                })
                            });
                    let (dq_rows, dq_cols) = if lowpass { (hr, hc) } else { (rows, cols) };
                    let WorkerScratch {
                        coeff,
                        lf_rect,
                        idct,
                    } = ws;
                    for matrix in coeff.iter_mut() {
                        matrix.reset(rows, cols);
                    }
                    for c in 0..NUM_CHANNELS {
                        let quant = coeffs.channel(c).ok_or(RenderError::Unsupported(
                            "a varblock with a missing channel",
                        ))?;
                        let matrix = matrices
                            .get(c)
                            .ok_or(RenderError::Unsupported("a missing dequantization channel"))?;
                        let scale = mul * qm.get(c).copied().unwrap_or(1.0);
                        let bias = DEFAULT_QUANT_BIAS.get(c).copied().unwrap_or(1.0);
                        let Some(target) = coeff.get_mut(c) else {
                            continue;
                        };
                        for (y, row) in quant.chunks_exact(cols).enumerate().take(dq_rows) {
                            for (x, &q) in row.iter().enumerate().take(dq_cols) {
                                let adjusted = bias_adjust(q, bias, DEFAULT_QUANT_BIAS_NUMERATOR);
                                target.set(x, y, adjusted * scale * matrix.at(x, y));
                            }
                        }
                    }

                    // I.6 (HF): the tile's factors, applied to every cell; I.8
                    // overwrites the LLF cells right after.
                    let tile_index = usize::try_from(
                        u64::from(by / CFL_TILE_BLOCKS) * u64::from(tiles.width)
                            + u64::from(bx / CFL_TILE_BLOCKS),
                    )
                    .unwrap_or(usize::MAX);
                    let x_factor = group.cfl.x_from_y().get(tile_index).map_or(0, |f| f.get());
                    let b_factor = group.cfl.b_from_y().get(tile_index).map_or(0, |f| f.get());
                    let (k_x, k_b) = cfl_factors(
                        corr.base_correlation_x,
                        corr.base_correlation_b,
                        corr.colour_factor,
                        x_factor,
                        b_factor,
                    );
                    apply_hf_cfl(coeff, k_x, k_b, dq_rows, dq_cols);

                    // I.8: the LLF rectangle from the LF planes. The rectangle is
                    // rewritten cell for cell per channel, so one reused block
                    // sees exactly the values a fresh zeroed block would.
                    let (block_rows, block_cols) = transform.block_dims();
                    lf_rect.reset(block_rows, block_cols);
                    for (c, matrix) in coeff.iter_mut().enumerate() {
                        for dy in 0..block_rows {
                            for dx in 0..block_cols {
                                let value = lf_at(
                                    c,
                                    bx.saturating_add(narrow(dx)),
                                    by.saturating_add(narrow(dy)),
                                );
                                lf_rect.set(dx, dy, value);
                            }
                        }
                        matrix.write_llf(&llf_from_lf(transform, lf_rect));
                    }

                    // I.9: samples, at the varblock's frame position (placed
                    // later). The decimated path reconstructs the varblock's
                    // own 2:1 box average in the coefficient domain; origins
                    // are multiples of 8 (LF-group rects are >=1024-aligned),
                    // so the half-resolution placement at exactly half the
                    // offset never straddles varblocks.
                    if decimate {
                        out.x0 = rect.x0 / 2 + bx * 4;
                        out.y0 = rect.y0 / 2 + by * 4;
                    } else {
                        out.x0 = rect.x0 + bx * 8;
                        out.y0 = rect.y0 + by * 8;
                    }
                    for (c, block) in out.samples.iter_mut().enumerate() {
                        match coeff.get(c) {
                            Some(matrix) if decimate => {
                                transform.half_samples_from_coefficients_into(matrix, block, idct);
                            }
                            Some(matrix) => {
                                transform.samples_from_coefficients_into(matrix, block, idct);
                            }
                            None if decimate => block
                                .reset(transform.half_sample_rows(), transform.half_sample_cols()),
                            None => block.reset(transform.sample_rows(), transform.sample_cols()),
                        }
                    }

                    // J.4.3: sigma per 8x8 block of the varblock, from `mul` and
                    // the block's own `Sharpness`, as (frame-block index, value).
                    let sharpness = group.sharpness.values();
                    out.sigma.clear();
                    for dy in 0..block_rows {
                        for dx in 0..block_cols {
                            let (sbx, sby) =
                                (bx.saturating_add(narrow(dx)), by.saturating_add(narrow(dy)));
                            if sbx >= blocks.width || sby >= blocks.height {
                                continue;
                            }
                            let s_idx = usize::try_from(
                                u64::from(sby) * u64::from(blocks.width) + u64::from(sbx),
                            )
                            .unwrap_or(usize::MAX);
                            let s = sharpness.get(s_idx).copied().unwrap_or(0).min(7);
                            let fbx = usize::try_from(rect.x0 / 8 + sbx).unwrap_or(usize::MAX);
                            let fby = usize::try_from(rect.y0 / 8 + sby).unwrap_or(usize::MAX);
                            if fbx >= blocks_x || fby >= blocks_y {
                                continue;
                            }
                            out.sigma
                                .push((fby * blocks_x + fbx, vardct_sigma(mul, s, epf_params)));
                        }
                    }

                    Ok(())
                };

            // Render varblocks in bounded, order-preserving chunks: each chunk's
            // per-varblock compute (dequant, CfL, LLF, inverse transform) runs
            // across the executor, then its samples are scattered band-parallel:
            // every worker owns a disjoint [`BAND_ROWS`]-row band and writes
            // only the varblock rows that fall inside it. Each sample belongs
            // to exactly one varblock and exactly one band, so the placed
            // pixels are identical to a serial render at any worker count;
            // chunking keeps the transient per-chunk buffers small rather than
            // holding one buffer per varblock at once. Sigma writes are a few
            // hundredths of the sample volume and stay serial.
            let n = group.blocks.len().min(ir.coefficients.len());
            let mut start = 0usize;
            while start < n {
                let end = (start + VARBLOCK_CHUNK).min(n);
                while arena.len() < end - start {
                    arena.push(RenderedVarblock::empty());
                }
                {
                    // One executor item per stride of varblocks, so the
                    // scratch pool's mutex is touched twice per stride, not
                    // twice per varblock (as one lock per varblock it
                    // measured +5% on the 12 MP quality wall).
                    let slots = VarblockSlots::new(arena.get_mut(..end - start).unwrap_or(&mut []));
                    let strides = (end - start).div_ceil(VARBLOCK_STRIDE);
                    fill_items(executor, strides, &|item| {
                        let lo = item * VARBLOCK_STRIDE;
                        let hi = (lo + VARBLOCK_STRIDE).min(end - start);
                        let mut ws = worker_scratch
                            .lock()
                            .ok()
                            .and_then(|mut pool| pool.pop())
                            .unwrap_or_default();
                        for k in lo..hi {
                            let Some(slot) = slots.take(k) else {
                                continue;
                            };
                            render_one(start + k, slot, &mut ws)?;
                        }
                        if let Ok(mut pool) = worker_scratch.lock() {
                            pool.push(ws);
                        }
                        Ok(())
                    })?;
                }
                let rendered = arena.get(..end - start).unwrap_or(&[]);
                // Bucket each varblock into the (at most two, since a varblock
                // is at most 32 rows tall) bands its rows intersect, preserving
                // chunk order within a bucket.
                let bands = row_bands(&mut planes, out_dims.width);
                let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); bands.len()];
                for (k, rv) in rendered.iter().enumerate() {
                    let rows = rv.samples.iter().map(|b| b.rows()).max().unwrap_or(0);
                    if rows == 0 {
                        continue;
                    }
                    let top = usize::try_from(rv.y0).unwrap_or(usize::MAX);
                    let bottom = top.saturating_add(rows - 1);
                    let last_band = buckets.len().saturating_sub(1);
                    for band in
                        (top / BAND_ROWS).min(last_band)..=(bottom / BAND_ROWS).min(last_band)
                    {
                        if let Some(bucket) = buckets.get_mut(band) {
                            bucket.push(k);
                        }
                    }
                }
                run_items(executor, bands.len(), &|index| {
                    let Some((row0, mut slices)) = bands.take(index) else {
                        return;
                    };
                    let Some(bucket) = buckets.get(index) else {
                        return;
                    };
                    let band_rows = slices.first().map_or(0, |s| s.len()) / out_dims.width.max(1);
                    for &k in bucket {
                        let Some(rv) = rendered.get(k) else {
                            continue;
                        };
                        let x0 = usize::try_from(rv.x0).unwrap_or(usize::MAX);
                        let y0 = usize::try_from(rv.y0).unwrap_or(usize::MAX);
                        for (c, block) in rv.samples.iter().enumerate() {
                            let Some(plane) = slices.get_mut(c) else {
                                continue;
                            };
                            for row in 0..block.rows() {
                                let fy = y0.saturating_add(row);
                                if fy < row0 || fy >= row0.saturating_add(band_rows) {
                                    continue;
                                }
                                let base = (fy - row0).saturating_mul(out_dims.width);
                                for col in 0..block.cols() {
                                    let fx = x0.saturating_add(col);
                                    if fx >= out_dims.width {
                                        continue;
                                    }
                                    if let Some(slot) = plane.get_mut(base.saturating_add(fx)) {
                                        *slot = block.at(col, row);
                                    }
                                }
                            }
                        }
                    }
                });
                for rv in rendered {
                    for &(idx, value) in &rv.sigma {
                        if let Some(slot) = sigma.get_mut(idx) {
                            *slot = value;
                        }
                    }
                }
                start = end;
            }
        }
        self.scratch.varblocks = arena;

        timings.varblocks_ms = millis(stage_start);

        // Phase S3 surrogate: the planes were already reconstructed at half
        // resolution by the coefficient-domain fold above; only the geometry
        // and the sigma field switch to the half grid here. Each half-res 8×8
        // sigma block averages the up-to-four full-res blocks it covers
        // (in-range never-written blocks participate as zeros, preserving the
        // S2 pixel-domain semantics so the reconstruction is the only thing
        // this phase changes about the observation).
        let (width, height, dims, len, sigma, blocks_x, blocks_y) = if decimate {
            let (hbx, hby) = block_grid(out_dims);
            let mut hsigma = vec![0.0f32; hbx.saturating_mul(hby)];
            for (j, row) in hsigma.chunks_exact_mut(hbx.max(1)).enumerate() {
                for (i, out) in row.iter_mut().enumerate() {
                    let mut sum = 0.0f32;
                    let mut count = 0u32;
                    for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                        let (sy, sx) = (2 * j + dy, 2 * i + dx);
                        if sx < blocks_x
                            && sy < blocks_y
                            && let Some(&v) = sigma.get(sy * blocks_x + sx)
                        {
                            sum += v;
                            count += 1;
                        }
                    }
                    *out = if count > 0 { sum / count as f32 } else { 0.0 };
                }
            }
            (out_width, out_height, out_dims, out_len, hsigma, hbx, hby)
        } else {
            (width, height, dims, len, sigma, blocks_x, blocks_y)
        };

        // Annex J. Both stages ping-pong between `planes` and `aux`: each
        // writes every sample of its output before any is read, so the
        // recycled buffer's stale contents cannot reach a rendered sample.
        let restoration = spatial.restoration.gaborish || spatial.restoration.epf_iters > 0;
        let mut aux = if restoration {
            self.scratch.take_aux(len)
        } else {
            Default::default()
        };
        let stage_start = std::time::Instant::now();
        if spatial.restoration.gaborish {
            for (dst, src) in aux.iter_mut().zip(planes.iter()) {
                gaborish_into(src, dst, dims, &self.gabor);
            }
            core::mem::swap(&mut planes, &mut aux);
        }
        timings.gaborish_ms = millis(stage_start);
        let stage_start = std::time::Instant::now();
        if spatial.restoration.epf_iters > 0 {
            let mut padded_storage = core::mem::take(&mut self.scratch.padded);
            let field = SigmaField::new(&sigma, blocks_x, blocks_y).ok_or(
                RenderError::Unsupported("a sigma field that does not match the block grid"),
            )?;
            for step in epf_steps(spatial.restoration.epf_iters).iter().copied() {
                // Each plane's mirrored-halo copy is a full-frame serial row
                // walk; building the three as executor items overlaps them
                // instead of paying three copies back-to-back on one thread.
                let jobs: [std::sync::Mutex<Option<Vec<f32>>>; NUM_CHANNELS] =
                    padded_storage.map(|s| std::sync::Mutex::new(Some(s)));
                let built: [std::sync::Mutex<Option<PaddedPlane>>; NUM_CHANNELS] =
                    core::array::from_fn(|_| std::sync::Mutex::new(None));
                run_items(executor, NUM_CHANNELS, &|index| {
                    let Some(storage) = jobs[index].lock().ok().and_then(|mut s| s.take()) else {
                        return;
                    };
                    let plane = PaddedPlane::new_in(&planes[index], dims, EPF_PAD, storage);
                    if let Ok(mut slot) = built[index].lock() {
                        *slot = plane;
                    }
                });
                let padded = built.map(|slot| slot.into_inner().ok().flatten());
                let [Some(p0), Some(p1), Some(p2)] = padded else {
                    return Err(RenderError::Unsupported(
                        "EPF planes that do not match the frame",
                    ));
                };
                let padded = [p0, p1, p2];
                let bands = row_bands(&mut aux, dims.width);
                let failed = std::sync::atomic::AtomicBool::new(false);
                run_items(executor, bands.len(), &|index| {
                    let Some((row0, mut slices)) = bands.take(index) else {
                        return;
                    };
                    let rows = row0..row0 + slices[0].len() / dims.width.max(1);
                    if epf_step_rows(step, &padded, &self.epf_params, &field, rows, &mut slices)
                        .is_none()
                    {
                        failed.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                });
                if failed.load(std::sync::atomic::Ordering::Relaxed) {
                    return Err(RenderError::Unsupported(
                        "EPF planes that do not match the frame",
                    ));
                }
                core::mem::swap(&mut planes, &mut aux);
                padded_storage = padded.map(PaddedPlane::into_storage);
            }
            self.scratch.padded = padded_storage;
        }
        if restoration {
            self.scratch.aux = aux;
        }

        timings.epf_ms = millis(stage_start);

        // Annex L: XYB -> linear sRGB -> the signalled sRGB encoding, or,
        // for the scoring path, straight to the depth-quantized linear value.
        let stage_start = std::time::Instant::now();
        {
            let bands = row_bands(&mut planes, dims.width);
            let opsin = self.opsin;
            run_items(executor, bands.len(), &|index| {
                let Some((_, mut slices)) = bands.take(index) else {
                    return;
                };
                let [x, y, b] = &mut slices;
                opsin.convert_planes(x, y, b);
                match linear_at_depth {
                    None => {
                        for plane in slices.iter_mut() {
                            for v in plane.iter_mut() {
                                *v = linear_to_srgb(*v);
                            }
                        }
                    }
                    Some(classifier) => {
                        for plane in slices.iter_mut() {
                            for v in plane.iter_mut() {
                                *v = classifier.linear_at_depth(*v);
                            }
                        }
                    }
                }
            });
        }

        timings.colour_ms = millis(stage_start);
        self.scratch.sigma = sigma;

        Ok((
            RenderedFrame {
                width,
                height,
                bits_per_sample: spatial.frame.bits_per_sample,
                planes,
            },
            timings,
        ))
    }
}

/// I.5.2 and I.6 over one LF group: the three dequantized, CfL-corrected LF
/// planes on the group's block grid.
fn lf_planes(
    ir: &jpxl_encode::vardct::QuantizedLfGroup,
    multipliers: &[f32; NUM_CHANNELS],
    extra_precision: u8,
    (k_x, k_b): (f32, f32),
) -> Result<[Vec<f32>; NUM_CHANNELS]> {
    let mut out: [Vec<f32>; NUM_CHANNELS] = [Vec::new(), Vec::new(), Vec::new()];
    for (c, plane) in out.iter_mut().enumerate() {
        let quant = ir.lf.plane(c).ok_or(RenderError::Unsupported(
            "an LF group with a missing LF plane",
        ))?;
        let m = multipliers.get(c).copied().unwrap_or(0.0);
        *plane = quant
            .iter()
            .map(|&q| lf_dequantize(q, m, extra_precision))
            .collect();
    }
    let [x, y, b] = &mut out;
    for ((dx, dy), db) in x.iter_mut().zip(y.iter()).zip(b.iter_mut()) {
        let (vx, _, vb) = cfl_apply(*dx, *dy, *db, k_x, k_b);
        *dx = vx;
        *db = vb;
    }
    Ok(out)
}

/// I.6 for HF coefficients over three same-shaped matrices, over the first
/// `rows x cols` cells (the full matrix on the canonical path; the low
/// quarter on the decimated fast path, whose remaining cells are all zero and
/// CfL over zeros is zeros).
fn apply_hf_cfl(
    coeffs: &mut [CoeffMatrix; NUM_CHANNELS],
    k_x: f32,
    k_b: f32,
    rows: usize,
    cols: usize,
) {
    for y in 0..rows {
        for x in 0..cols {
            let d_y = coeffs[1].at(x, y);
            let (v_x, _, v_b) = cfl_apply(coeffs[0].at(x, y), d_y, coeffs[2].at(x, y), k_x, k_b);
            coeffs[0].set(x, y, v_x);
            coeffs[2].set(x, y, v_b);
        }
    }
}

/// A small block extent as `u32`, saturating rather than wrapping.
fn narrow(v: usize) -> u32 {
    u32::try_from(v).unwrap_or(u32::MAX)
}

/// Rows per band when a stage is spread over workers. Fixed, so the band
/// partition — and with it nothing, since no stage reduces across bands — is
/// independent of the worker count.
const BAND_ROWS: usize = 64;

/// One parked band: its first row and the three planes' slices for it.
type Band<'a> = (usize, [&'a mut [f32]; NUM_CHANNELS]);

/// Row bands of three planes parked for one-shot pickup by executor items.
struct RowBands<'a> {
    items: Vec<std::sync::Mutex<Option<Band<'a>>>>,
}

impl<'a> RowBands<'a> {
    fn len(&self) -> usize {
        self.items.len()
    }

    fn take(&self, index: usize) -> Option<Band<'a>> {
        self.items.get(index)?.lock().ok()?.take()
    }
}

/// Cuts three equally sized planes into [`BAND_ROWS`]-row bands.
fn row_bands(planes: &mut [Vec<f32>; NUM_CHANNELS], width: usize) -> RowBands<'_> {
    let band_len = width.saturating_mul(BAND_ROWS).max(1);
    let [p0, p1, p2] = planes;
    let items = p0
        .chunks_mut(band_len)
        .zip(p1.chunks_mut(band_len))
        .zip(p2.chunks_mut(band_len))
        .enumerate()
        .map(|(i, ((a, b), c))| std::sync::Mutex::new(Some((i * BAND_ROWS, [a, b, c]))))
        .collect();
    RowBands { items }
}

/// One varblock's reconstruction result, produced off to the side so the
/// per-varblock compute can run across the executor and be scattered into the
/// frame afterwards. Slots are reused across chunks and renders; every field
/// is overwritten by the varblock that takes the slot.
#[derive(Debug)]
struct RenderedVarblock {
    /// Frame x of the varblock's top-left sample.
    x0: u32,
    /// Frame y of the varblock's top-left sample.
    y0: u32,
    /// The placed samples, one block per channel.
    samples: [SampleBlock; NUM_CHANNELS],
    /// J.4.3 sigma writes as `(frame-block index, value)`.
    sigma: Vec<(usize, f32)>,
}

impl RenderedVarblock {
    /// An unfilled slot awaiting its first varblock.
    fn empty() -> Self {
        Self {
            x0: 0,
            y0: 0,
            samples: core::array::from_fn(|_| SampleBlock::zeros(0, 0)),
            sigma: Vec::new(),
        }
    }
}

/// One worker's dequantization and inverse-transform scratch, taken from a
/// shared pool for the span of one varblock so the per-varblock loop makes no
/// allocations on the plain-DCT path.
struct WorkerScratch {
    /// The three dequantized coefficient matrices, zero-reset per varblock.
    coeff: [CoeffMatrix; NUM_CHANNELS],
    /// The I.8 LF rectangle, rewritten cell for cell per channel.
    lf_rect: SampleBlock,
    /// The IDCT working buffer.
    idct: Vec<f32>,
}

impl Default for WorkerScratch {
    fn default() -> Self {
        Self {
            coeff: core::array::from_fn(|_| CoeffMatrix::zeros(0, 0)),
            lf_rect: SampleBlock::zeros(0, 0),
            idct: Vec::new(),
        }
    }
}

/// A chunk's varblock slots parked for one-shot pickup by executor items,
/// the same shape as [`RowBands`].
struct VarblockSlots<'a> {
    items: Vec<std::sync::Mutex<Option<&'a mut RenderedVarblock>>>,
}

impl<'a> VarblockSlots<'a> {
    fn new(slots: &'a mut [RenderedVarblock]) -> Self {
        Self {
            items: slots
                .iter_mut()
                .map(|slot| std::sync::Mutex::new(Some(slot)))
                .collect(),
        }
    }

    fn take(&self, index: usize) -> Option<&'a mut RenderedVarblock> {
        self.items.get(index)?.lock().ok()?.take()
    }
}

/// Varblocks rendered per parallel chunk before their results are scattered.
///
/// A chunk holds at most this many [`RenderedVarblock`] results at once, so the
/// transient sample storage stays a few megabytes rather than a whole frame's
/// worth, while still giving the executor enough work per chunk to keep every
/// worker busy on a large frame.
const VARBLOCK_CHUNK: usize = 2048;

/// Varblocks one executor item renders serially, in index order, with one
/// worker-scratch checkout. Small enough for [`VARBLOCK_CHUNK`] to still
/// spread across every worker, large enough that the pool's mutex is off the
/// per-varblock path.
const VARBLOCK_STRIDE: usize = 64;

/// Runs `n` fallible fill closures across `executor` when present (the first
/// error in index order wins, so worker count cannot change the outcome),
/// serially otherwise.
fn fill_items(
    executor: Option<&EncodeExecutor>,
    n: usize,
    f: &(dyn Fn(usize) -> Result<()> + Sync),
) -> Result<()> {
    match executor {
        Some(executor) => executor.map_ordered(n, f).map(|_: Vec<()>| ()),
        None => {
            for i in 0..n {
                f(i)?;
            }
            Ok(())
        }
    }
}

/// Runs `items` independent closures on `executor` (serially without one).
fn run_items(executor: Option<&EncodeExecutor>, items: usize, f: &(dyn Fn(usize) + Sync)) {
    match executor {
        Some(executor) => {
            let outcome: core::result::Result<Vec<()>, core::convert::Infallible> = executor
                .map_ordered(items, |i| {
                    f(i);
                    Ok(())
                });
            let _ = outcome;
        }
        None => {
            for i in 0..items {
                f(i);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fused threshold classifier must agree bit-for-bit with encoding to
    /// sRGB and round-tripping through the depth quantizer, over a dense
    /// sweep of the working range, the immediate bit-lattice neighbourhood of
    /// every threshold (where any boundary error would live), both transfer
    /// branch seams, and the non-finite specials.
    #[test]
    fn fused_depth_levels_match_the_srgb_round_trip() {
        for bits in [1u32, 8, 12, 16] {
            let max = RenderedFrame::full_scale(bits);
            let classifier = DepthClassifier::new(bits);
            let (lut, thresholds) = (&classifier.lut, &classifier.thresholds);
            let check = |x: f32| {
                let direct = RenderedFrame::linear_sample(linear_to_srgb(x), max, lut);
                let searched = lut.get(level_of(x, thresholds)).copied().unwrap_or(0.0);
                let guided = classifier.linear_at_depth(x);
                assert_eq!(
                    direct.to_bits(),
                    searched.to_bits(),
                    "search: bits {bits}, x {x} ({:#010x})",
                    x.to_bits()
                );
                assert_eq!(
                    direct.to_bits(),
                    guided.to_bits(),
                    "guide: bits {bits}, x {x} ({:#010x})",
                    x.to_bits()
                );
            };
            for i in 0..120_000 {
                check(-0.2 + i as f32 * 1.25e-5);
            }
            for &t in thresholds {
                let b = t.to_bits();
                for d in 0..4u32 {
                    check(f32::from_bits(b.saturating_sub(d)));
                    check(f32::from_bits(b.saturating_add(d)));
                }
            }
            for x in [
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                -0.0,
                0.0,
                0.003_130_8,
                0.040_449_936,
                1.0,
                1.5,
                f32::MIN_POSITIVE,
            ] {
                check(x);
            }
        }
    }
}
