//! HF metadata: the `HfMetadata` modular sub-bitstream and the greedy
//! varblock placement it drives (18181-1 G.2.4).
//!
//! G.2.4's own text:
//!
//! ```text
//! The decoder reads nb_blocks = 1 + u(ceil(log2(ceil(width / 8) *
//! ceil(height / 8)))).
//!
//! Then, the decoder reads a Modular sub-bitstream ... for an image with
//! four channels: the first two channels have ceil(height / 64) rows and
//! ceil(width / 64) columns ... denoted XFromY and BFromY ...; the third
//! channel has two rows and nb_blocks columns and is denoted BlockInfo, and
//! the fourth channel has ceil(height / 8) rows and ceil(width / 8) columns
//! and is denoted Sharpness ....
//!
//! The DctSelect and HfMul fields are derived from the first and second
//! rows of BlockInfo .... They are reconstructed by iterating over the
//! columns of BlockInfo to obtain a varblock transform type type (the
//! sample at the first row) and a quantization multiplier mul (the sample
//! at the second row). The type is a DctSelect sample and is stored at the
//! coordinates of the top-left 8x8 rectangle of the varblock. This position
//! is the earliest block in raster order that is not already covered by
//! other varblocks. The positioned varblock is completely contained in the
//! current LF group, does not cross group boundaries, and also does not
//! overlap with already-positioned varblocks. The HfMul sample is stored at
//! the same position and gets the value 1 + mul.
//! ```
//!
//! `width`/`height` are the current LF group's dimensions (G.2.1 General),
//! same as [`crate::vardct::lf`].
//!
//! # Scope
//!
//! This module parses the four channels to typed planes and then performs
//! the greedy placement walk, producing a validated [`VarblockPlacement`]
//! list. It does not consume `XFromY`/`BFromY`/`Sharpness` any further than
//! handing them back as planes — I.5.3's HF dequantization (which reads
//! `XFromY`/`BFromY`) and J.4's EPF sigma derivation (which reads
//! `Sharpness`) are later slices' business.

// The greedy placement walk below indexes a local `covered` grid whose bounds
// (`blocks_w * blocks_h`) are checked once up front, and every index into it
// is `cursor`, or `cursor`'s row/column arithmetic, which is re-validated
// against the same bound immediately before each access. `Channel::get`
// already returns 0 out of bounds rather than panicking, so the risk this
// lint flags does not apply here; see `jpxl_core::varblock` for the same
// reasoning applied to its coefficient buffers.
#![allow(clippy::indexing_slicing)]

use jpxl_bitstream::{BitReader, trace_field};
use jpxl_core::geometry::LfBlockPos;
use jpxl_core::limits::AllocGuard;
use jpxl_core::varblock::TransformType;

use crate::error::{DecodeError, Result};
use crate::modular::{Channel, ChannelSpec, ModularOptions, TreeSource, decode_sub_bitstream_with};

/// The four `HfMetadata` channels of G.2.4, still unprocessed integer planes.
#[derive(Debug, Clone)]
pub struct HfMetaPlanes {
    /// `XFromY`: HF chroma-from-luma correlation for the X channel, one
    /// sample per 64x64 rectangle.
    pub x_from_y: Channel,
    /// `BFromY`: HF chroma-from-luma correlation for the B channel, one
    /// sample per 64x64 rectangle.
    pub b_from_y: Channel,
    /// `BlockInfo`: two rows (`DctSelect`, `mul`) by `nb_blocks` columns.
    pub block_info: Channel,
    /// `Sharpness`: one sample per 8x8 block, feeds the EPF sigma lookup
    /// (J.4).
    pub sharpness: Channel,
    /// `nb_blocks`, G.2.4's leading field: the number of varblocks
    /// `BlockInfo` encodes.
    pub nb_blocks: u32,
}

/// `ceil(log2(n))` for `n >= 1`: the bit width of a `u()` field that must
/// represent every value in `0..n`.
///
/// `n == 1` needs zero bits (there is only one representable value, `0`),
/// which is exactly G.2.4's case for a single-block LF group.
const fn ceil_log2(n: u32) -> u32 {
    if n <= 1 { 0 } else { (n - 1).ilog2() + 1 }
}

/// One placed varblock: G.2.4's greedy placement result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VarblockPlacement {
    /// Top-left 8x8-block position, relative to the LF group's origin.
    pub position: LfBlockPos,
    /// The varblock's transform type (`DctSelect`).
    pub transform: TransformType,
    /// `HfMul = 1 + mul`, the per-varblock HF quantization multiplier.
    pub hf_mul: u32,
}

/// Reads G.2.4's `nb_blocks` field and the `HfMetadata` sub-bitstream.
///
/// `group_width`/`group_height` are the current LF group's pixel dimensions,
/// as for [`crate::vardct::lf::read_lf_quant`].
///
/// `options.stream_index` must already be set by the caller — G.2.4's stream
/// index is `crate::frame::stream_index::hf_metadata`.
///
/// # Errors
///
/// Any [`DecodeError`] the sub-bitstream decode reports, or
/// [`DecodeError::FieldOutOfRange`] if `nb_blocks`'s bit width does not fit
/// the block grid (a malformed/adversarial LF-group size).
pub fn read_hf_metadata(
    reader: &mut BitReader<'_>,
    group_width: u32,
    group_height: u32,
    options: &ModularOptions,
    tree_source: TreeSource<'_>,
    guard: &mut AllocGuard,
) -> Result<HfMetaPlanes> {
    let blocks_w = group_width.div_ceil(8);
    let blocks_h = group_height.div_ceil(8);
    let total_blocks = blocks_w.checked_mul(blocks_h).ok_or_else(|| {
        DecodeError::out_of_range("LF group block grid", "G.2.4", u64::from(blocks_w))
    })?;

    let bits = ceil_log2(total_blocks.max(1));
    let extra = trace_field!(reader, "hf_meta.nb_blocks_extra", reader.read_bits(bits))?;
    let nb_blocks = 1u32
        .checked_add(extra)
        .ok_or_else(|| DecodeError::out_of_range("nb_blocks", "G.2.4", u64::from(extra)))?;

    let corr_w = group_width.div_ceil(64);
    let corr_h = group_height.div_ceil(64);

    let specs = [
        ChannelSpec::new(corr_w, corr_h),     // XFromY
        ChannelSpec::new(corr_w, corr_h),     // BFromY
        ChannelSpec::new(nb_blocks, 2),       // BlockInfo
        ChannelSpec::new(blocks_w, blocks_h), // Sharpness
    ];

    let image = decode_sub_bitstream_with(reader, &specs, options, tree_source, guard)?;
    let channels = image.into_channels();
    let [x_from_y, b_from_y, block_info, sharpness]: [Channel; 4] = channels
        .try_into()
        .map_err(|_| DecodeError::out_of_range("HfMetadata channel count", "G.2.4", 4))?;

    Ok(HfMetaPlanes {
        x_from_y,
        b_from_y,
        block_info,
        sharpness,
        nb_blocks,
    })
}

/// G.2.4's greedy varblock placement: walks `block_info`'s `nb_blocks`
/// columns, placing each varblock at the earliest raster-order 8x8 block not
/// already covered.
///
/// `blocks_w x blocks_h` is the LF group's 8x8-block grid — the same
/// dimensions [`read_hf_metadata`] used for `Sharpness`.
///
/// # Errors
///
/// [`DecodeError::FieldOutOfRange`] if:
/// * a `DctSelect` sample is not one of Table I.1's 27 values;
/// * a varblock's footprint would cross the LF-group boundary;
/// * a varblock overlaps one already placed;
/// * `BlockInfo` runs out of raster positions before its columns do
///   (more varblocks than the grid has room for);
/// * `mul` is negative enough that `1 + mul` is not a valid `HfMul`;
/// * the grid still has uncovered blocks once every `BlockInfo` column has
///   been placed.
///
/// A malformed placement is always rejected, never clamped or silently
/// dropped — that is the point of this being a distinct validated type
/// rather than a raw scan over `block_info`.
pub fn place_varblocks(
    block_info: &Channel,
    blocks_w: u32,
    blocks_h: u32,
) -> Result<Vec<VarblockPlacement>> {
    let total = blocks_w.checked_mul(blocks_h).ok_or_else(|| {
        DecodeError::out_of_range("LF group block grid", "G.2.4", u64::from(blocks_w))
    })?;
    let total_usize = usize::try_from(total)
        .map_err(|_| DecodeError::out_of_range("LF group block grid", "G.2.4", u64::from(total)))?;

    let nb_blocks = block_info.width();
    let mut covered = vec![false; total_usize];
    let mut placements = Vec::with_capacity(usize::try_from(nb_blocks).unwrap_or(0));
    let mut cursor: u32 = 0;

    for col in 0..nb_blocks {
        while cursor < total && covered[cursor as usize] {
            cursor += 1;
        }
        if cursor >= total {
            return Err(DecodeError::out_of_range(
                "BlockInfo column past a fully covered LF group",
                "G.2.4",
                u64::from(col),
            ));
        }
        let bx = cursor % blocks_w;
        let by = cursor / blocks_w;
        let position = LfBlockPos::new(bx, by);

        let dct_select = i64::from(block_info.get(col, 0));
        let transform = TransformType::from_dct_select(dct_select)?;
        let (rows, cols) = transform.block_dims();
        let rows = u32::try_from(rows)
            .map_err(|_| DecodeError::out_of_range("varblock row footprint", "I.1", rows as u64))?;
        let cols = u32::try_from(cols).map_err(|_| {
            DecodeError::out_of_range("varblock column footprint", "I.1", cols as u64)
        })?;

        if bx.checked_add(cols).is_none_or(|edge| edge > blocks_w)
            || by.checked_add(rows).is_none_or(|edge| edge > blocks_h)
        {
            return Err(DecodeError::out_of_range(
                "varblock crosses the LF-group boundary",
                "G.2.4",
                dct_select.unsigned_abs(),
            ));
        }

        for dy in 0..rows {
            for dx in 0..cols {
                let idx = (by + dy) * blocks_w + (bx + dx);
                if covered[idx as usize] {
                    return Err(DecodeError::out_of_range(
                        "varblock overlaps an already-placed varblock",
                        "G.2.4",
                        u64::from(idx),
                    ));
                }
            }
        }
        for dy in 0..rows {
            for dx in 0..cols {
                let idx = (by + dy) * blocks_w + (bx + dx);
                covered[idx as usize] = true;
            }
        }

        let mul = block_info.get(col, 1);
        let hf_mul = mul
            .checked_add(1)
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| {
                DecodeError::out_of_range("HfMul = 1 + mul", "G.2.4", i64::from(mul).unsigned_abs())
            })?;

        placements.push(VarblockPlacement {
            position,
            transform,
            hf_mul,
        });
    }

    if covered.iter().any(|&c| !c) {
        return Err(DecodeError::out_of_range(
            "LF group left with uncovered 8x8 blocks after BlockInfo",
            "G.2.4",
            u64::from(nb_blocks),
        ));
    }

    Ok(placements)
}

#[cfg(test)]
#[allow(
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    reason = "the hand-built bit writer indexes its own byte buffer, whose \
              length it just computed, and truncates only test-chosen small \
              constants; a panic here is a failing test"
)]
mod tests {
    use super::*;
    use jpxl_core::limits::Limits;

    // -----------------------------------------------------------------
    // Minimal hand-built-bitstream helpers (see `vardct::lf`'s copy for
    // the rationale — each file hand-rolls its own rather than reaching
    // into `tests/modular_common`, which is outside this file's
    // ownership).
    // -----------------------------------------------------------------

    #[derive(Debug, Default)]
    struct BitWriter {
        bits: Vec<bool>,
    }

    impl BitWriter {
        fn u(&mut self, value: u32, n: u32) {
            for i in 0..n {
                self.bits.push((value >> i) & 1 == 1);
            }
        }

        fn bit(&mut self, value: bool) {
            self.bits.push(value);
        }

        fn code_msb_first(&mut self, value: u32, n: u32) {
            for i in (0..n).rev() {
                self.bits.push((value >> i) & 1 == 1);
            }
        }

        fn finish(&self) -> Vec<u8> {
            let mut out = vec![0u8; self.bits.len().div_ceil(8)];
            for (i, &bit) in self.bits.iter().enumerate() {
                if bit {
                    out[i / 8] |= 1 << (i % 8);
                }
            }
            out
        }
    }

    fn write_header_no_transforms(w: &mut BitWriter) {
        w.bit(false); // use_global_tree
        w.bit(true); // wp_params: default_wp
        w.u(0, 2); // nb_transforms: U32 selector 0 -> constant 0
    }

    fn write_single_leaf_tree(w: &mut BitWriter, predictor: u32) {
        write_prefix_bundle(w, 6);
        for token in [0u32, predictor, 0, 0, 0] {
            w.code_msb_first(token, 2);
        }
    }

    fn write_prefix_bundle(w: &mut BitWriter, num_dist: usize) {
        w.bit(false); // lz77.enabled
        if num_dist > 1 {
            w.bit(true); // simple clustering
            w.u(0, 2); // nbits = 0
        }
        w.bit(true); // use_prefix_code
        w.u(15, 4); // split_exponent = log_alphabet_size, no msb/lsb fields
        w.bit(true); // alphabet_size flag
        w.u(1, 4); // n = 1
        w.u(1, 1); // extra = 1 -> alphabet_size = 4
        w.u(1, 2); // selector 1 = simple
        w.u(3, 2); // nsym - 1 = 3
        for symbol in 0..4u32 {
            w.u(symbol, 2);
        }
        w.bit(false); // balanced pattern [2,2,2,2]
    }

    /// `UnpackSigned` inverse, restricted to the 4-symbol test alphabet:
    /// token 0 -> 0, 1 -> -1, 2 -> 1, 3 -> -2.
    fn pack_signed(v: i32) -> u32 {
        match v {
            0 => 0,
            -1 => 1,
            1 => 2,
            -2 => 3,
            other => panic!("{other} has no token in the 4-symbol test alphabet"),
        }
    }

    #[test]
    fn ceil_log2_matches_the_bit_widths_g24_needs() {
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
        assert_eq!(ceil_log2(65_536), 16);
    }

    #[test]
    fn a_hand_built_hf_metadata_stream_decodes_to_the_expected_planes() {
        // A 16x16 LF group: corr channels ceil(16/64) = 1x1; blocks_w =
        // blocks_h = 2, so a 2x2 block grid (4 blocks), nb_blocks_extra bit
        // width ceil(log2(4)) = 2.
        let mut w = BitWriter::default();
        w.u(1, 2); // nb_blocks_extra = 1 -> nb_blocks = 2
        write_header_no_transforms(&mut w);
        write_single_leaf_tree(&mut w, 0); // predictor 0 = Zero
        write_prefix_bundle(&mut w, 1);
        // Channel order: XFromY (1 sample), BFromY (1 sample), BlockInfo (2
        // cols x 2 rows = 4 samples, raster: row0 then row1), Sharpness
        // (2x2 = 4 samples).
        let tokens = [
            0i32, // XFromY[0]
            -1,   // BFromY[0]
            0, 1, // BlockInfo row 0 (DctSelect): [0, 1] -> Dct8x8, Hornuss
            0, -1, // BlockInfo row 1 (mul): [0, -1] -> HfMul 1, 0
            1, -1, 0, 1, // Sharpness, raster order
        ];
        for v in tokens {
            w.code_msb_first(pack_signed(v), 2);
        }
        let data = w.finish();

        let mut reader = BitReader::new(&data);
        let limits = Limits::relaxed();
        let mut guard = AllocGuard::new(&limits);
        let planes = read_hf_metadata(
            &mut reader,
            16,
            16,
            &ModularOptions::level10(),
            TreeSource::Local,
            &mut guard,
        )
        .expect("well-formed HfMetadata stream");

        assert_eq!(planes.nb_blocks, 2);
        assert_eq!(planes.x_from_y.samples(), &[0]);
        assert_eq!(planes.b_from_y.samples(), &[-1]);
        assert_eq!(planes.block_info.samples(), &[0, 1, 0, -1]);
        assert_eq!(planes.sharpness.samples(), &[1, -1, 0, 1]);
    }

    #[test]
    fn truncated_input_is_rejected_not_panicking() {
        let data: [u8; 1] = [0];
        let mut reader = BitReader::new(&data);
        let limits = Limits::relaxed();
        let mut guard = AllocGuard::new(&limits);
        let result = read_hf_metadata(
            &mut reader,
            2048,
            2048,
            &ModularOptions::level10(),
            TreeSource::Local,
            &mut guard,
        );
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------
    // Greedy placement invariant tests. These build `BlockInfo` channels
    // directly (bypassing the modular sub-bitstream), matching the brief's
    // "fabricated DctSelect planes" framing.
    // -----------------------------------------------------------------

    fn block_info_channel(dct_select: &[i32], mul: &[i32]) -> Channel {
        assert_eq!(dct_select.len(), mul.len());
        let n = dct_select.len();
        let spec = ChannelSpec::new(u32::try_from(n).expect("small test dimension"), 2);
        let mut data = Vec::with_capacity(2 * n);
        data.extend_from_slice(dct_select);
        data.extend_from_slice(mul);
        Channel::from_samples(spec, data).expect("well-formed test channel")
    }

    #[test]
    fn every_block_is_covered_exactly_once_by_uniform_dct8x8() {
        // A 4x4 block grid, filled entirely with DCT8x8 (1x1-block
        // footprint each) placed in raster order.
        let dct_select = vec![0i32; 16]; // TransformType::Dct8x8 == 0
        let mul = vec![0i32; 16];
        let block_info = block_info_channel(&dct_select, &mul);

        let placements = place_varblocks(&block_info, 4, 4).expect("exact tiling");
        assert_eq!(placements.len(), 16);

        let mut covered = std::collections::HashSet::new();
        for p in &placements {
            assert_eq!(p.transform, TransformType::Dct8x8);
            assert_eq!(p.hf_mul, 1);
            assert!(
                covered.insert((p.position.bx(), p.position.by())),
                "block ({}, {}) covered twice",
                p.position.bx(),
                p.position.by()
            );
        }
        assert_eq!(covered.len(), 16);
        // Raster order: DCT8x8 is placed at the current raster cursor, one
        // per column of BlockInfo, so position i is (i % 4, i / 4).
        for (i, p) in placements.iter().enumerate() {
            let i = u32::try_from(i).expect("small test index");
            assert_eq!((p.position.bx(), p.position.by()), (i % 4, i / 4));
        }
    }

    #[test]
    fn a_larger_varblock_covers_its_whole_footprint_and_skips_covered_cells() {
        // A 4x2 block grid. First column is DCT16x16 (2x2-block footprint,
        // Table I.1 value 4), covering both rows of columns 0..2. The
        // remaining four cells are DCT8x8, placed at the earliest
        // uncovered raster position each time.
        let dct_select = vec![4i32, 0, 0, 0, 0]; // Dct16x16, then 4x Dct8x8
        let mul = vec![0i32; 5];
        let block_info = block_info_channel(&dct_select, &mul);

        let placements = place_varblocks(&block_info, 4, 2).expect("valid placement");
        assert_eq!(placements.len(), 5);
        assert_eq!(placements[0].transform, TransformType::Dct16x16);
        assert_eq!(
            (placements[0].position.bx(), placements[0].position.by()),
            (0, 0)
        );

        // The DCT16x16 covers (0,0),(1,0),(0,1),(1,1); the next uncovered
        // raster position is (2,0), then (3,0), (2,1), (3,1).
        let expected_dct8x8_positions = [(2u32, 0u32), (3, 0), (2, 1), (3, 1)];
        for (p, expected) in placements[1..].iter().zip(expected_dct8x8_positions) {
            assert_eq!(p.transform, TransformType::Dct8x8);
            assert_eq!((p.position.bx(), p.position.by()), expected);
        }
    }

    #[test]
    fn a_varblock_crossing_the_lf_group_edge_is_rejected() {
        // A 1x1 grid cannot hold a DCT16x16 (2x2-block footprint).
        let block_info = block_info_channel(&[4], &[0]);
        let err = place_varblocks(&block_info, 1, 1).expect_err("expected a rejected placement");
        assert!(err.to_string().contains("crosses the LF-group boundary"));
    }

    #[test]
    fn a_varblock_crossing_the_edge_at_a_partial_lf_group_is_rejected() {
        // A 3x3 grid (a partial LF group at a frame edge) with a DCT32x32
        // (4x4-block footprint, Table I.1 value 5) placed at the origin
        // does not fit: it would reach column/row 4 of a 3-wide grid.
        let block_info = block_info_channel(&[5], &[0]);
        let err = place_varblocks(&block_info, 3, 3).expect_err("expected a rejected placement");
        assert!(err.to_string().contains("crosses the LF-group boundary"));
    }

    #[test]
    fn the_largest_transform_type_fits_exactly_at_a_matching_grid() {
        // Dct256x256 (Table I.1 value 24) has a 32x32-block footprint —
        // exactly an LF group's own maximum grid (LF_GROUP_BLOCKS == 256
        // 8x8 blocks per side / 8 == 32). It must place at the origin and
        // cover the grid completely.
        let block_info = block_info_channel(&[24], &[0]);
        let placements = place_varblocks(&block_info, 32, 32).expect("exact fit");
        assert_eq!(placements.len(), 1);
        assert_eq!(placements[0].transform, TransformType::Dct256x256);
        assert_eq!(
            (placements[0].position.bx(), placements[0].position.by()),
            (0, 0)
        );
    }

    #[test]
    fn overlapping_varblocks_are_rejected() {
        // Two DCT16x16 varblocks (2x2-block footprint each) at a 2x2 grid:
        // the first covers the whole grid, so the second — which the
        // greedy walk would place at the same earliest-uncovered position,
        // since nothing is left uncovered — cannot be placed at all and
        // the walk reports "past a fully covered LF group" rather than
        // silently dropping it.
        let block_info = block_info_channel(&[4, 4], &[0, 0]);
        let err = place_varblocks(&block_info, 2, 2).expect_err("expected a rejected placement");
        assert!(err.to_string().contains("fully covered"));
    }

    #[test]
    fn an_invalid_dct_select_value_is_rejected() {
        let block_info = block_info_channel(&[27], &[0]); // Table I.1 has 0..=26
        let err = place_varblocks(&block_info, 4, 4).expect_err("expected a rejected placement");
        // Surfaced through jpxl_core::JpxlError via `DecodeError::Core`.
        assert!(err.to_string().to_lowercase().contains("dctselect"));
    }

    #[test]
    fn an_incomplete_lf_group_is_rejected_not_silently_accepted() {
        // A 2x2 grid with only one DCT8x8 placed: three blocks are left
        // uncovered once BlockInfo is exhausted.
        let block_info = block_info_channel(&[0], &[0]);
        let err = place_varblocks(&block_info, 2, 2).expect_err("expected a rejected placement");
        assert!(err.to_string().contains("uncovered"));
    }

    #[test]
    fn a_negative_mul_that_makes_hf_mul_nonpositive_is_rejected() {
        let block_info = block_info_channel(&[0], &[-2]); // HfMul = 1 + (-2) = -1
        let err = place_varblocks(&block_info, 1, 1).expect_err("expected a rejected placement");
        assert!(err.to_string().contains("HfMul"));
    }
}
