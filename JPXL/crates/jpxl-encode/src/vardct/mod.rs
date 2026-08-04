//! VarDCT plan IR, validation and emission (18181-1 Annexes G and I).
//!
//! This module is the normative half of the encoder's one-way boundary
//! (`docs/PLAN.md` slice 11, `docs/Encoder-plan1.md` milestone 1):
//!
//! ```text
//! jpxl-encode-policy   search, heuristics, R-D, clustering
//!         │  builds an EmissionPlan
//!         ▼
//! jpxl-encode          validate() -> ValidatedEmissionPlan -> bits
//! ```
//!
//! The dependency runs one way — policy depends on this crate, never the
//! reverse — and it is enforced by the workspace manifest, not by convention:
//! `jpxl-encode` does not list `jpxl-encode-policy` as a dependency, so a
//! heuristic added here would not compile against anything it needs.
//!
//! Nothing in this module chooses. It defines what a plan *is*
//! ([`plan`]), what makes one structurally legal ([`validate`]), how to look
//! at one ([`dump`]), the interface entropy coding is driven through
//! ([`sink`]), and the writer that will only accept a validated one
//! ([`emit`]).

pub mod dump;
pub mod emit;
pub mod error;
pub mod geometry;
pub mod headers;
pub mod ids;
pub mod modular_out;
pub mod plan;
pub mod sink;
pub mod validate;
pub mod walk;
pub mod write;

pub use dump::dump;
pub use error::{PlanError, PlanResult};
pub use geometry::{BlockGrid, Rect, SectionKind, VardctGeometry};
pub use plan::{
    CflGrid, EmissionPlan, EntropyModelPlan, EntropyPlan, FrameDecision, HfBlockContextPlan,
    HfPassEntropyPlan, HistogramPlan, HybridUintPlan, LfDecision, LfGroupPlan, LfQuantPlanes,
    OrderSet, QuantizedFrameIr, QuantizedLfGroup, QuantizerDecision, RestorationDecision,
    SectionLayout, SharpnessGrid, SpatialPlan, VarblockCoefficients, VarblockDecision,
};
pub use sink::{CensusSink, HfEventSink, SymbolSink};
pub use validate::{ValidatedEmissionPlan, validate};
pub use walk::{OrderTables, PassGroupWalk, WalkVarblock, pre_context_count, walk_pass_group};
pub use write::{census_frame, check_supported, plan_pre_contexts, walk_frame, write_codestream};
