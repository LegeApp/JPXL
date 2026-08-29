//! The shared state one candidate search keeps across its probes.
//!
//! A search — rate-targeted or score-targeted — plans several candidates of
//! the same frame. Everything that is quantizer-independent (the source and
//! its preconditioned transform frame, the analysis atlas, the forward-DCT
//! cache, the quantization arenas) lives here once, and each probe borrows
//! it. The two doors are the pre-entropy [`pixel_plan`](CandidateSearchContext::pixel_plan),
//! which is all a perceptual probe needs, and the writer-ready
//! [`attach_entropy`](CandidateSearchContext::attach_entropy) that turns them
//! into a writer-ready plan.

use crate::error::Result;
use crate::quantizer_ladder::QuantizerChoice;
use crate::request::EncodeRequest;
use crate::{AnalysisAtlas, AnchorReuse, EntropySearch, PreparedFrame, StructuralAnchor};
use jpxl_encode::vardct::{ValidatedEmissionPlan, ValidatedPixelPlan, VardctGeometry};

/// Request-scoped search state.
pub(crate) struct CandidateSearchContext<'a> {
    frame: &'a PreparedFrame,
    transform_frame: &'a PreparedFrame,
    atlas: &'a AnalysisAtlas,
    request: &'a EncodeRequest,
    executor: &'a jpxl_encode::EncodeExecutor,
    fwd_cache: crate::CandidateForwardCache,
    quant_workspace: crate::QuantizationWorkspace,
}

impl<'a> CandidateSearchContext<'a> {
    /// Prepares the shared state; nothing is planned yet.
    pub(crate) fn new(
        frame: &'a PreparedFrame,
        transform_frame: &'a PreparedFrame,
        atlas: &'a AnalysisAtlas,
        request: &'a EncodeRequest,
        executor: &'a jpxl_encode::EncodeExecutor,
    ) -> Self {
        Self {
            frame,
            transform_frame,
            atlas,
            request,
            executor,
            fwd_cache: crate::CandidateForwardCache::new(),
            quant_workspace: crate::QuantizationWorkspace::new(),
        }
    }

    /// The executor every probe runs on.
    pub(crate) const fn executor(&self) -> &'a jpxl_encode::EncodeExecutor {
        self.executor
    }

    /// The request the context was built with.
    ///
    /// The perceptual controller now carries the effective per-policy request on
    /// the navigator (a bank trial's differs from the context's), so this reader
    /// has no caller; it stays as the symmetric accessor of the shared state.
    #[allow(
        dead_code,
        reason = "symmetric accessor; the navigator carries the effective request per policy"
    )]
    pub(crate) const fn request(&self) -> &'a EncodeRequest {
        self.request
    }

    /// Plans the pixels of one candidate: cover, CfL, quantization and the LF
    /// planes, validated, with no entropy work.
    ///
    /// `structure_tier` names the entropy tier the candidate will eventually
    /// be priced under; the planner reads only its *structural* consequences
    /// (Fast's fixed cover and nearest quantizer), so a probe and the finalist
    /// it becomes are built the same way.
    ///
    /// The controller always plans through [`pixel_plan_for`](Self::pixel_plan_for)
    /// with an explicit per-policy request, so this `self.request` convenience
    /// currently has no caller; it stays as the plain door of the abstraction.
    #[allow(
        dead_code,
        reason = "the perceptual controller plans through pixel_plan_for with an explicit request"
    )]
    pub(crate) fn pixel_plan(
        &mut self,
        quantizer: QuantizerChoice,
        enable_cfl: bool,
        structure_tier: EntropySearch,
        reuse: AnchorReuse<'_>,
        capture: Option<&mut Option<StructuralAnchor>>,
    ) -> Result<(ValidatedPixelPlan, VardctGeometry)> {
        let request = self.request;
        self.pixel_plan_for(
            request,
            quantizer,
            enable_cfl,
            structure_tier,
            reuse,
            capture,
        )
    }

    /// [`pixel_plan`](Self::pixel_plan) planning against an explicit `request`
    /// instead of the context's own.
    ///
    /// The forward-DCT cache and quantization arenas belong to the transform
    /// frame and block transforms only (a pure function of the pixels and the
    /// selected transform per `(family, block)`), never of the quantizer-side
    /// knobs — the QM `(x, b)` scales, `quant_lf`, `lambda_scale` — nor of EPF,
    /// which is a decode-time loop filter. So a policy-bank trial can plan on
    /// the *baseline's* populated context: a quantizer-side alternative reuses
    /// the baseline cover/CfL through [`AnchorReuse::CoverAndCfl`] and reads its
    /// forward coefficients straight out of this cache (zero fresh structural
    /// builds), and a structural alternative (CfL/EPF) rebuilds its cover but
    /// still reads the already-transformed coefficients rather than re-running
    /// the forward DCT. Only the shared *transform frame* — which no bank axis
    /// moves (Gaborish is not a bank axis) — must match the one this cache was
    /// filled against, which it does because the orchestrator builds it once.
    pub(crate) fn pixel_plan_for(
        &mut self,
        request: &EncodeRequest,
        quantizer: QuantizerChoice,
        enable_cfl: bool,
        structure_tier: EntropySearch,
        reuse: AnchorReuse<'_>,
        capture: Option<&mut Option<StructuralAnchor>>,
    ) -> Result<(ValidatedPixelPlan, VardctGeometry)> {
        crate::plan_pixels_on_anchor_with_workspace(
            self.frame,
            self.transform_frame,
            self.atlas,
            request,
            quantizer,
            enable_cfl,
            &mut self.fwd_cache,
            structure_tier,
            Some(self.executor),
            reuse,
            capture,
            &mut self.quant_workspace,
        )
    }

    /// Reduces the frame's aligned DCT8x8 candidates into the one-shot
    /// program's [`TransformFeatureSummary`](crate::quality_features::TransformFeatureSummary),
    /// filling this context's shared forward cache so the later pixel plans
    /// read the same warm entries (PR 4). Quantizer-independent: safe to call
    /// before any quantizer choice exists.
    pub(crate) fn prepare_quality_transform_summary(
        &mut self,
        request: &EncodeRequest,
    ) -> Result<crate::quality_features::TransformFeatureSummary> {
        crate::quality_transform_summary(
            self.transform_frame,
            request,
            &mut self.fwd_cache,
            Some(self.executor),
        )
    }

    /// Trains entropy for already planned pixels and returns the writer-ready
    /// plan. Pixels are untouched: the candidate's score is unchanged.
    pub(crate) fn attach_entropy(
        &self,
        pixels: &ValidatedPixelPlan,
        geometry: &VardctGeometry,
        entropy: EntropySearch,
    ) -> Result<ValidatedEmissionPlan> {
        self.attach_entropy_for(self.request, pixels, geometry, entropy)
    }

    /// [`attach_entropy`](Self::attach_entropy) training against an explicit
    /// `request` instead of the context's own, so a policy-bank trial prices
    /// its finalist under its own policy while sharing this context.
    pub(crate) fn attach_entropy_for(
        &self,
        request: &EncodeRequest,
        pixels: &ValidatedPixelPlan,
        geometry: &VardctGeometry,
        entropy: EntropySearch,
    ) -> Result<ValidatedEmissionPlan> {
        crate::attach_entropy(
            pixels.plan(),
            geometry,
            request,
            entropy,
            Some(self.executor),
        )
    }
}
