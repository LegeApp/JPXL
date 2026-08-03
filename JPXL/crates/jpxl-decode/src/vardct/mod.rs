//! VarDCT (`kVarDCT`) decoding (18181-1 Annex G VarDCT halves, Annex I).
//!
//! Created by the 8D-parse sub-slice, which owns only [`lf`] and [`hf_meta`]
//! here. The concurrent 8B sub-slice (parameter bundles: `quantizer`,
//! `block_ctx`, `dequant_matrix`) adds its own modules to this file —
//! whichever of the two lands second appends its `pub mod` lines rather than
//! overwriting this one.

pub mod block_ctx;
pub mod cfl;
pub mod dequant_matrix;
pub mod hf_coeff;
pub mod hf_meta;
pub mod lf;
pub mod order;
pub mod quantizer;
