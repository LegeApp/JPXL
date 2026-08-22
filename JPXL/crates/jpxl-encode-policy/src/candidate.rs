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
use crate::rate::QuantizerChoice;
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

    /// The request being searched.
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
    pub(crate) fn pixel_plan(
        &mut self,
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
            self.request,
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

    /// Trains entropy for already planned pixels and returns the writer-ready
    /// plan. Pixels are untouched: the candidate's score is unchanged.
    pub(crate) fn attach_entropy(
        &self,
        pixels: &ValidatedPixelPlan,
        geometry: &VardctGeometry,
        entropy: EntropySearch,
    ) -> Result<ValidatedEmissionPlan> {
        crate::attach_entropy(
            pixels.plan(),
            geometry,
            self.request,
            entropy,
            Some(self.executor),
        )
    }
}
