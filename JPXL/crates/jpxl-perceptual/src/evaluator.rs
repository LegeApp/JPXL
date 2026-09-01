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
    Ssimulacra2, cumulative_partial_errors,
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
    /// Phase S1 shadow instrumentation: when set (`JPXL_SURROGATE_SHADOW`),
    /// every canonical evaluation also computes the half-resolution surrogate
    /// score of the same rendered planes and reports it alongside. Purely
    /// observational — the canonical score is computed and returned unchanged.
    surrogate_shadow: bool,
    /// Certified early-rejection shadow (`JPXL_REJECTION_SHADOW`): every
    /// canonical evaluation also reports the cumulative weighted error after
    /// each completed metric scale, from which an exact upper bound on the
    /// final score can be derived offline. Purely observational.
    rejection_shadow: bool,
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
            surrogate_shadow: std::env::var_os("JPXL_SURROGATE_SHADOW")
                .is_some_and(|v| v != "0" && !v.is_empty()),
            rejection_shadow: std::env::var_os("JPXL_REJECTION_SHADOW")
                .is_some_and(|v| v != "0" && !v.is_empty()),
        })
    }

    /// The surrogate score of `candidate` through the Phase S3 decimating
    /// render — the varblocks' 2:1 box average reconstructed directly in the
    /// coefficient domain, restoration and colour at half resolution — with
    /// its wall time (render and metric together), when `want` asks for one
    /// (navigation pairing or the `JPXL_SURROGATE_SHADOW` instrumentation).
    fn surrogate_of(
        &mut self,
        want: bool,
        candidate: &ValidatedPixelPlan,
    ) -> (Option<f64>, Option<u64>) {
        if !want {
            return (None, None);
        }
        let start = std::time::Instant::now();
        let Ok((width, height, half)) = self.renderer.render_linear_at_depth_decimated_with(
            candidate,
            self.bits_per_sample,
            self.executor,
        ) else {
            return (None, None);
        };
        let score = {
            let [r, g, b] = &half;
            let Ok(view) = LinearRgbView::new(width, height, r, g, b) else {
                return (None, None);
            };
            self.metric
                .score_surrogate_prescaled(&self.reference, view, self.executor)
                .ok()
                .map(|result| result.score)
        };
        // The half planes usually carry full-resolution capacity (they came
        // from a recycled canonical probe); hand them back so a canonical
        // render that follows resizes in place instead of allocating three
        // fresh planes.
        self.renderer.recycle_planes(half);
        let millis = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
        (score, score.map(|_| millis))
    }

    /// Scores an arbitrary interleaved 8-bit sRGB frame against the retained
    /// source reference with the canonical metric.
    ///
    /// The routed text/UI candidate competition uses this to price its
    /// colour-reduced rasters: the pixels are linearised exactly as
    /// [`Self::from_srgb8`] linearised the source, so the score is the one
    /// `jpxl compare` would report between the decoded lossless stream of
    /// this raster and the original.
    ///
    /// # Errors
    ///
    /// [`MetricError`] if the dimensions do not match the reference.
    pub fn score_srgb8_candidate(
        &mut self,
        width: u32,
        height: u32,
        rgb: &[u8],
    ) -> Result<f64, MetricError> {
        let lut: [f32; 256] =
            core::array::from_fn(|v| jpxl_core::color::srgb_to_linear(v as f32 / 255.0));
        let planes = deinterleave(rgb.chunks_exact(3), |v| {
            lut.get(usize::from(v)).copied().unwrap_or(0.0)
        });
        let [r, g, b] = &planes;
        let view = LinearRgbView::new(width, height, r, g, b)?;
        let result = self.metric.score(&self.reference, view, self.executor)?;
        self.evaluations = self.evaluations.saturating_add(1);
        Ok(result.score)
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

impl PlanRenderEvaluator<'_> {
    /// Renders `candidate` and scores it canonically, with the surrogate
    /// score alongside when `want_surrogate` asks for it.
    fn evaluate_with(
        &mut self,
        candidate: &ValidatedPixelPlan,
        want_surrogate: bool,
    ) -> jpxl_encode_policy::Result<PerceptualObservation> {
        // The previous probe's planes are done with; hand their allocations
        // back so this render reuses them instead of allocating three more.
        self.renderer
            .recycle_planes(core::mem::take(&mut self.linear));
        // The decimated surrogate render runs first: it borrows those
        // full-capacity planes for its half-resolution output and
        // `surrogate_of` recycles them again, so the canonical render below
        // still resizes in place.
        let (surrogate_score, surrogate_millis) = self.surrogate_of(want_surrogate, candidate);
        let render_start = std::time::Instant::now();
        let (width, height, linear) = self
            .renderer
            .render_linear_at_depth_with(candidate, self.bits_per_sample, self.executor)
            .map_err(|_| PolicyError::Unsupported {
                what: "a candidate plan the renderer could not reconstruct",
            })?;
        self.linear = linear;
        let render_millis = u64::try_from(render_start.elapsed().as_millis()).unwrap_or(u64::MAX);
        let [r, g, b] = &self.linear;
        let view =
            LinearRgbView::new(width, height, r, g, b).map_err(|_| PolicyError::Unsupported {
                what: "a rendered frame whose planes do not match its dimensions",
            })?;
        let metric_start = std::time::Instant::now();
        let result = self
            .metric
            .score(&self.reference, view, self.executor)
            .map_err(|_| PolicyError::Unsupported {
                what: "a candidate whose dimensions differ from the source",
            })?;
        let metric_millis = u64::try_from(metric_start.elapsed().as_millis()).unwrap_or(u64::MAX);
        self.evaluations = self.evaluations.saturating_add(1);
        let partial_errors = self
            .rejection_shadow
            .then(|| cumulative_partial_errors(&result.scales));
        Ok(PerceptualObservation {
            score: result.score,
            surrogate_score,
            surrogate_millis,
            render_millis: Some(render_millis),
            metric_millis: Some(metric_millis),
            partial_errors,
        })
    }
}

impl PerceptualEvaluator for PlanRenderEvaluator<'_> {
    fn evaluate(
        &mut self,
        candidate: &ValidatedPixelPlan,
    ) -> jpxl_encode_policy::Result<PerceptualObservation> {
        self.evaluate_with(candidate, self.surrogate_shadow)
    }

    fn evaluate_owned(
        &mut self,
        candidate: ValidatedPixelPlan,
    ) -> jpxl_encode_policy::Result<(PerceptualObservation, Option<ValidatedPixelPlan>)> {
        self.evaluate_owned_for_navigation(candidate, false)
    }

    fn evaluate_owned_for_navigation(
        &mut self,
        candidate: ValidatedPixelPlan,
        with_surrogate: bool,
    ) -> jpxl_encode_policy::Result<(PerceptualObservation, Option<ValidatedPixelPlan>)> {
        let want_surrogate = with_surrogate || self.surrogate_shadow;
        if !self.low_memory {
            let observation = self.evaluate_with(&candidate, want_surrogate)?;
            return Ok((observation, Some(candidate)));
        }

        // The decimated surrogate render runs first (see `evaluate_with`).
        let (surrogate_score, surrogate_millis) = self.surrogate_of(want_surrogate, &candidate);
        let render_start = std::time::Instant::now();
        let (width, height, linear) = self
            .renderer
            .render_linear_at_depth_with(&candidate, self.bits_per_sample, self.executor)
            .map_err(|_| PolicyError::Unsupported {
                what: "a candidate plan the renderer could not reconstruct",
            })?;
        let render_millis = u64::try_from(render_start.elapsed().as_millis()).unwrap_or(u64::MAX);
        // Once reconstruction is complete, the coefficient payload is not
        // needed for this score. Exact finalists are rebuilt deterministically
        // by the policy if this rung survives navigation.
        drop(candidate);
        // Large-frame path: holding spare full-resolution planes between
        // probes would raise the search's resident peak, so free the
        // renderer's scratch before the metric allocates its own.
        self.renderer.release_scratch();
        let metric_start = std::time::Instant::now();
        let result = self
            .metric
            .score_owned(&self.reference, width, height, linear, self.executor)
            .map_err(|_| PolicyError::Unsupported {
                what: "a candidate whose dimensions differ from the source",
            })?;
        let metric_millis = u64::try_from(metric_start.elapsed().as_millis()).unwrap_or(u64::MAX);
        self.metric.release_scratch();
        self.evaluations = self.evaluations.saturating_add(1);
        let partial_errors = self
            .rejection_shadow
            .then(|| cumulative_partial_errors(&result.scales));
        Ok((
            PerceptualObservation {
                score: result.score,
                surrogate_score,
                surrogate_millis,
                render_millis: Some(render_millis),
                metric_millis: Some(metric_millis),
                partial_errors,
            },
            None,
        ))
    }

    fn supports_surrogate(&self) -> bool {
        true
    }

    fn evaluate_surrogate_owned(
        &mut self,
        candidate: ValidatedPixelPlan,
    ) -> jpxl_encode_policy::Result<Option<PerceptualObservation>> {
        // The previous probe's planes come back as render scratch, exactly as
        // on the canonical path; a surrogate probe never pays a
        // full-resolution restoration, colour pass or metric.
        self.renderer
            .recycle_planes(core::mem::take(&mut self.linear));
        let start = std::time::Instant::now();
        let (width, height, half) = self
            .renderer
            .render_linear_at_depth_decimated_with(&candidate, self.bits_per_sample, self.executor)
            .map_err(|_| PolicyError::Unsupported {
                what: "a candidate plan the renderer could not reconstruct",
            })?;
        // A surrogate observation can never become a finalist, so the
        // coefficient payload is never needed again.
        drop(candidate);
        if self.low_memory {
            self.renderer.release_scratch();
        }
        let result = {
            let [r, g, b] = &half;
            let view = LinearRgbView::new(width, height, r, g, b).map_err(|_| {
                PolicyError::Unsupported {
                    what: "a rendered frame whose planes do not match its dimensions",
                }
            })?;
            self.metric
                .score_surrogate_prescaled(&self.reference, view, self.executor)
                .map_err(|_| PolicyError::Unsupported {
                    what: "a candidate whose reference pyramid has no surrogate scale",
                })?
        };
        let millis = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
        if self.low_memory {
            // Holding spare planes between probes would raise the search's
            // resident peak; the next render reallocates.
            self.metric.release_scratch();
        } else {
            // Keep the (usually full-capacity) plane allocations circulating
            // for whichever render runs next.
            self.renderer.recycle_planes(half);
        }
        self.evaluations = self.evaluations.saturating_add(1);
        Ok(Some(PerceptualObservation {
            score: result.score,
            surrogate_score: Some(result.score),
            surrogate_millis: Some(millis),
            render_millis: None,
            metric_millis: None,
            partial_errors: None,
        }))
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
