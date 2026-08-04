//! The writer side of the boundary.
//!
//! Everything in this module takes a [`ValidatedEmissionPlan`] and nothing
//! else. That is the whole point of the split: there is no entry point on the
//! emission side that accepts an unvalidated plan, so "the writer received a
//! plan whose varblocks do not tile their LF group" is not a bug that can be
//! written.
//!
//! # Milestone-1 scope
//!
//! The section *layout* is writable today — [`write_sections`] emits F.3.3's
//! TOC and the section bodies in the plan's declared order, reusing the
//! [`SectionStore`] the lossless encoder already uses. The VarDCT
//! `ImageHeader`/`FrameHeader` writer and the section *bodies* (G.1 to G.4)
//! are milestone 2; until they exist, a caller supplies the bodies and this
//! module guarantees they are laid out in the order the plan promised.

use jpxl_bitstream::BitWriter;

use crate::error::{EncodeError, Result};
use crate::section::SectionStore;
use crate::vardct::error::PlanError;
use crate::vardct::validate::ValidatedEmissionPlan;

/// Writes the TOC and the section bodies of a validated plan.
///
/// `store` must hold exactly one body per section the plan declares, in the
/// same order. `w` must be positioned immediately after the `FrameHeader`.
///
/// # Errors
///
/// [`EncodeError::Plan`] if `store` does not match the plan's section layout,
/// and any error the TOC writer reports.
pub fn write_sections(
    plan: &ValidatedEmissionPlan,
    store: SectionStore,
    w: &mut BitWriter,
) -> Result<()> {
    let expected = plan.plan().sections.kinds.len();
    if store.len() != expected {
        return Err(EncodeError::Plan(PlanError::shape(
            "section store length",
            "F.3.1",
            expected as u64,
            store.len() as u64,
        )));
    }
    store.write(w)
}
