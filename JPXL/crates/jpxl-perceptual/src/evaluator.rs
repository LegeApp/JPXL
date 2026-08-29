//! The policy layer's [`PerceptualEvaluator`], implemented over the plan
//! renderer and the in-tree metric.
//!
//! The policy crate sees only the trait; this type is what the facade injects.
//! It keeps the precomputed source reference and the renderer/metric scratch
//! for the whole search, so each probe pays for one reconstruction and one
//! candidate-side metric pass.
//!
//! Scores are taken on the image a viewer would see: the rendered frame is
//! quantized to the frame's bit depth and re-linearised exactly as the
//! decoded file would be, so the in-loop score and `jpxl compare` on the
//! emitted stream agree.

use jpxl_encode::EncodeExecutor;
use jpxl_encode::vardct::ValidatedPixelPlan;
use jpxl_encode_policy::{PerceptualEvaluator, PerceptualObservation, PolicyError};
use jpxl_plan_render::PlanRenderer;

use crate::reference::LOW_MEMORY_PIXELS;
use crate::{
    LinearRgbView, METRIC_VERSION, MetricError, PrecomputedReference, ReferenceRetention,
    Ssimulacra2,
};

/// Why an evaluator could not be built.
#[derive(Debug)]
pub enum EvaluatorError {
    /// The source cannot be scored (too small, or inconsistent planes).
    Metric(MetricError),
    /// The renderer's defaults could not be built.
    Render(jpxl_plan_render::RenderError),
}

impl core::fmt::Display for EvaluatorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Metric(e) => write!(f, "metric: {e}"),
            Self::Render(e) => write!(f, "render: {e}"),
        }
    }
}

impl std::error::Error for EvaluatorError {}

impl From<MetricError> for EvaluatorError {
    fn from(e: MetricError) -> Self {
        Self::Metric(e)
    }
}

impl From<jpxl_plan_render::RenderError> for EvaluatorError {
    fn from(e: jpxl_plan_render::RenderError) -> Self {
        Self::Render(e)
    }
}

/// Renders each candidate plan and scores it against a precomputed source.
pub struct PlanRenderEvaluator<'e> {
    renderer: PlanRenderer,
    metric: Ssimulacra2,
    reference: PrecomputedReference,
    /// The candidate's linear-RGB planes, reused across probes so each probe
    /// reuses one set of planes rather than allocating three.
    linear: [Vec<f32>; 3],
    bits_per_sample: u32,
    executor: &'e EncodeExecutor,
    evaluations: u32,
    low_memory: bool,
}

impl<'e> PlanRenderEvaluator<'e> {
    /// Builds the evaluator from interleaved 8-bit sRGB.
    ///
    /// # Errors
    ///
    /// [`EvaluatorError::Metric`] for a frame below the metric's 8x8 floor
    /// or a sample count that does not match the dimensions.
    pub fn from_srgb8(
        width: u32,
        height: u32,
        rgb: &[u8],
        executor: &'e EncodeExecutor,
    ) -> Result<Self, EvaluatorError> {
        let lut: [f32; 256] =
            core::array::from_fn(|v| jpxl_core::color::srgb_to_linear(v as f32 / 255.0));
        let planes = deinterleave(rgb.chunks_exact(3), |v| {
            lut.get(usize::from(v)).copied().unwrap_or(0.0)
        });
        Self::from_linear_planes(width, height, planes, 8, executor)
    }

    /// Builds the evaluator from interleaved high-precision sRGB samples of
    /// `bits_per_sample` bits each.
    ///
    /// # Errors
    ///
    /// As [`Self::from_srgb8`].
    pub fn from_srgb16(
        width: u32,
        height: u32,
        rgb: &[u16],
        bits_per_sample: u32,
        executor: &'e EncodeExecutor,
    ) -> Result<Self, EvaluatorError> {
        let max = f32::from(u16::MAX).min(((1u32 << bits_per_sample.clamp(1, 16)) - 1) as f32);
        // The sample domain is u16, so the per-sample transfer function is a
        // table of the same expression evaluated once per distinct value.
        let lut: Vec<f32> = (0..=u32::from(u16::MAX))
            .map(|v| jpxl_core::color::srgb_to_linear(v as f32 / max))
            .collect();
        let planes = deinterleave(rgb.chunks_exact(3), |v| {
            lut.get(usize::from(v)).copied().unwrap_or(0.0)
        });
        Self::from_linear_planes(width, height, planes, bits_per_sample, executor)
    }

    fn from_linear_planes(
        width: u32,
        height: u32,
        planes: [Vec<f32>; 3],
        bits_per_sample: u32,
        executor: &'e EncodeExecutor,
    ) -> Result<Self, EvaluatorError> {
        let reference = PrecomputedReference::new_owned(
            width,
            height,
            planes,
            ReferenceRetention::default_for(width, height),
            executor,
        )?;
        Ok(Self {
            renderer: PlanRenderer::new()?,
            metric: Ssimulacra2::new(),
            reference,
            linear: [Vec::new(), Vec::new(), Vec::new()],
            bits_per_sample,
            executor,
            evaluations: 0,
            low_memory: u64::from(width).saturating_mul(u64::from(height)) >= LOW_MEMORY_PIXELS,
        })
    }

    /// How many candidates have been scored.
    #[must_use]
    pub const fn evaluations(&self) -> u32 {
        self.evaluations
    }

    /// Bytes of source-side metric state retained for the search.
    #[must_use]
    pub fn reference_bytes(&self) -> usize {
        self.reference.retained_bytes()
    }
}

impl PerceptualEvaluator for PlanRenderEvaluator<'_> {
    fn evaluate(
        &mut self,
        candidate: &ValidatedPixelPlan,
    ) -> jpxl_encode_policy::Result<PerceptualObservation> {
        // The previous probe's planes are done with; hand their allocations
        // back so this render reuses them instead of allocating three more.
        self.renderer
            .recycle_planes(core::mem::take(&mut self.linear));
        let (width, height, linear) = self
            .renderer
            .render_linear_at_depth_with(candidate, self.bits_per_sample, self.executor)
            .map_err(|_| PolicyError::Unsupported {
                what: "a candidate plan the renderer could not reconstruct",
            })?;
        self.linear = linear;
        let [r, g, b] = &self.linear;
        let view =
            LinearRgbView::new(width, height, r, g, b).map_err(|_| PolicyError::Unsupported {
                what: "a rendered frame whose planes do not match its dimensions",
            })?;
        let result = self
            .metric
            .score(&self.reference, view, self.executor)
            .map_err(|_| PolicyError::Unsupported {
                what: "a candidate whose dimensions differ from the source",
            })?;
        self.evaluations = self.evaluations.saturating_add(1);
        Ok(PerceptualObservation {
            score: result.score,
        })
    }

    fn evaluate_owned(
        &mut self,
        candidate: ValidatedPixelPlan,
    ) -> jpxl_encode_policy::Result<(PerceptualObservation, Option<ValidatedPixelPlan>)> {
        if !self.low_memory {
            let observation = self.evaluate(&candidate)?;
            return Ok((observation, Some(candidate)));
        }

        let (width, height, linear) = self
            .renderer
            .render_linear_at_depth_with(&candidate, self.bits_per_sample, self.executor)
            .map_err(|_| PolicyError::Unsupported {
                what: "a candidate plan the renderer could not reconstruct",
            })?;
        // Once reconstruction is complete, the coefficient payload is not
        // needed for this score. Exact finalists are rebuilt deterministically
        // by the policy if this rung survives navigation.
        drop(candidate);
        // Large-frame path: holding spare full-resolution planes between
        // probes would raise the search's resident peak, so free the
        // renderer's scratch before the metric allocates its own.
        self.renderer.release_scratch();
        let result = self
            .metric
            .score_owned(&self.reference, width, height, linear, self.executor)
            .map_err(|_| PolicyError::Unsupported {
                what: "a candidate whose dimensions differ from the source",
            })?;
        self.metric.release_scratch();
        self.evaluations = self.evaluations.saturating_add(1);
        Ok((
            PerceptualObservation {
                score: result.score,
            },
            None,
        ))
    }

    fn metric_version(&self) -> &'static str {
        METRIC_VERSION
    }
}

/// Splits interleaved RGB into three planes through `convert`.
fn deinterleave<'a, T: Copy + 'a>(
    pixels: impl Iterator<Item = &'a [T]>,
    convert: impl Fn(T) -> f32,
) -> [Vec<f32>; 3] {
    let hint = pixels.size_hint().0;
    let mut planes: [Vec<f32>; 3] = [
        Vec::with_capacity(hint),
        Vec::with_capacity(hint),
        Vec::with_capacity(hint),
    ];
    for px in pixels {
        for (plane, &v) in planes.iter_mut().zip(px) {
            plane.push(convert(v));
        }
    }
    planes
}
