//! Block tiling: the exact-cover search in `BlockInfo`'s own representation
//! (`Encoder-plan1.md` §4).
//!
//! The design decision this module exists to hold: **the search's output is
//! the wire sequence.** G.2.4 places each varblock at the earliest 8x8 block
//! not already covered, in raster order, so a solver that walks the cover
//! frontier in exactly that order produces the `BlockInfo` column sequence
//! directly. There is no spatial partition tree, and no later conversion from
//! one into the other where an ordering bug could hide.
//!
//! # Milestone-1 scope
//!
//! One solver: [`fixed_dct8x8`], which places a DCT8x8 at every atom. It is
//! trivially an exact cover, which is the point — milestone 2's vertical slice
//! needs a legal tiling and nothing more, and having it come out of the same
//! frontier walk the hierarchical (M6) and beam (M10) solvers will use means
//! those are substitutions, not rewrites.
//!
//! The frontier state, candidate R-D envelopes and the two real solvers of
//! §4.1–§4.4 arrive with milestone 6.

use jpxl_core::geometry::LfBlockPos;
use jpxl_core::varblock::TransformType;
use jpxl_encode::vardct::geometry::BlockGrid;
use jpxl_encode::vardct::ids::HfMul;
use jpxl_encode::vardct::plan::VarblockDecision;

use crate::error::Result;

/// Places a DCT8x8 varblock at every atom of `grid`, in raster order.
///
/// `hf_mul` is applied to every varblock: milestone 1 has no adaptive
/// quantization, and a constant multiplier is the honest way to say so.
///
/// # Errors
///
/// Cannot fail for a legal grid; the signature is fallible so that the real
/// solvers, which can fail to cover, are drop-in replacements.
pub fn fixed_dct8x8(grid: BlockGrid, hf_mul: HfMul) -> Result<Vec<VarblockDecision>> {
    let mut blocks = Vec::with_capacity(usize::try_from(grid.area()).unwrap_or(0));
    for y in 0..grid.height {
        for x in 0..grid.width {
            blocks.push(VarblockDecision {
                origin: LfBlockPos::new(x, y),
                transform: TransformType::Dct8x8,
                hf_mul,
            });
        }
    }
    Ok(blocks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fixed_solver_emits_the_raster_sequence_g24_replays() {
        let grid = BlockGrid {
            width: 3,
            height: 2,
        };
        let blocks = fixed_dct8x8(grid, HfMul::new(1).expect("legal")).expect("covers");
        assert_eq!(blocks.len(), 6);
        for (index, block) in blocks.iter().enumerate() {
            let index = u32::try_from(index).expect("small");
            assert_eq!(block.origin.bx(), index % 3);
            assert_eq!(block.origin.by(), index / 3);
            assert_eq!(block.transform, TransformType::Dct8x8);
        }
    }
}
