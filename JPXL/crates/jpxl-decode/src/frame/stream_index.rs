//! H.4.1's modular-sub-bitstream stream index (18181-1 Table H.4, property 1).
//!
//! Every modular sub-bitstream inside a frame — the frame-wide `GlobalModular`
//! image, the per-`LfGroup` LF coefficients and `ModularLfGroup` data, the
//! per-`LfGroup` HF metadata, the `RAW` dequantization-matrix tables (I.2.4),
//! and the per-pass-group modular data — is assigned a distinct `stream index`
//! that feeds the MA tree's property 1. H.4.1 defines the formula for each:
//!
//! ```text
//! GlobalModular:            0
//! LF coefficients:          1 + lf_group_index
//! ModularLfGroup:            1 + num_lf_groups + lf_group_index
//! HFMetadata:                 1 + 2 * num_lf_groups + lf_group_index
//! RAW dequantization tables:    1 + 3 * num_lf_groups + parameter_index
//! ModularGroup:      1 + 3 * num_lf_groups + 17 + num_groups * pass_index + group_index
//! ```
//!
//! (H.4.1's printed RAW-table formula reads `i + 3 * num_lf_groups +
//! parameter_index`; every other line in the list is a literal `1`, and the
//! `17` in `ModularGroup` is exactly [`jpxl_core::varblock::NUM_DEQUANT_MATRICES`]
//! — the RAW-table range has to end where `ModularGroup` begins, which only
//! works if the RAW formula's leading term is `1` as well. `i` is read as an
//! OCR misrecognition of `1`.)
//!
//! `decode.rs` computes two of these six inline today (`ModularLfGroup` and
//! `ModularGroup`); this module is the typed, tested factoring the docs
//! promised, and a later wave switches the inline call sites over to it.
//! Until then the duplication is deliberate — see `docs/HANDOFF.md`.
//!
//! Every function validates its index against the geometry it was handed and
//! rejects an out-of-range index rather than silently producing a stream index
//! that collides with a neighbour's.

use jpxl_core::varblock::NUM_DEQUANT_MATRICES;

use crate::frame::error::{FrameError, Result};
use crate::frame::geometry::FrameGeometry;

/// Property 1 for the frame-wide `GlobalModular` sub-bitstream (G.1.3).
///
/// Always zero; `GlobalModular` is not indexed by anything.
#[must_use]
pub const fn global_modular() -> u32 {
    0
}

/// Property 1 for the LF-coefficients sub-bitstream of LF group
/// `lf_group_index` (G.2.2): `1 + lf_group_index`.
///
/// # Errors
///
/// [`FrameError::FieldOutOfRange`] if `lf_group_index >= geometry.num_lf_groups()`.
pub fn lf_coefficients(geometry: &FrameGeometry, lf_group_index: u64) -> Result<u32> {
    check_lf_group_index(geometry, lf_group_index)?;
    narrow(1 + lf_group_index)
}

/// Property 1 for the `ModularLfGroup` sub-bitstream of LF group
/// `lf_group_index` (G.2.3): `1 + num_lf_groups + lf_group_index`.
///
/// # Errors
///
/// [`FrameError::FieldOutOfRange`] if `lf_group_index >= geometry.num_lf_groups()`.
pub fn modular_lf_group(geometry: &FrameGeometry, lf_group_index: u64) -> Result<u32> {
    check_lf_group_index(geometry, lf_group_index)?;
    narrow(1 + geometry.num_lf_groups() + lf_group_index)
}

/// Property 1 for the HF-metadata sub-bitstream of LF group `lf_group_index`
/// (G.2.4): `1 + 2 * num_lf_groups + lf_group_index`.
///
/// # Errors
///
/// [`FrameError::FieldOutOfRange`] if `lf_group_index >= geometry.num_lf_groups()`.
pub fn hf_metadata(geometry: &FrameGeometry, lf_group_index: u64) -> Result<u32> {
    check_lf_group_index(geometry, lf_group_index)?;
    narrow(1 + 2 * geometry.num_lf_groups() + lf_group_index)
}

/// Property 1 for the RAW dequantization-matrix table `parameter_index`
/// (I.2.4): `1 + 3 * num_lf_groups + parameter_index`.
///
/// `parameter_index` ranges over Table I.4's
/// [`NUM_DEQUANT_MATRICES`](jpxl_core::varblock::NUM_DEQUANT_MATRICES) slots.
///
/// # Errors
///
/// [`FrameError::FieldOutOfRange`] if `parameter_index >= NUM_DEQUANT_MATRICES`.
pub fn dequant_table(geometry: &FrameGeometry, parameter_index: u64) -> Result<u32> {
    if parameter_index >= NUM_DEQUANT_MATRICES as u64 {
        return Err(FrameError::out_of_range(
            "dequant matrix parameter index",
            "I.2.4",
            parameter_index,
        ));
    }
    narrow(1 + 3 * geometry.num_lf_groups() + parameter_index)
}

/// Property 1 for the `ModularGroup` sub-bitstream of pass `pass_index`,
/// group `group_index` (G.4.2):
/// `1 + 3 * num_lf_groups + 17 + num_groups * pass_index + group_index`.
///
/// # Errors
///
/// [`FrameError::FieldOutOfRange`] if `group_index >= geometry.num_groups()`
/// or `pass_index >= geometry.num_passes()`.
pub fn modular_group(geometry: &FrameGeometry, pass_index: u64, group_index: u64) -> Result<u32> {
    if group_index >= geometry.num_groups() {
        return Err(FrameError::out_of_range(
            "group_index",
            "H.4.1",
            group_index,
        ));
    }
    if pass_index >= u64::from(geometry.num_passes()) {
        return Err(FrameError::out_of_range("pass_index", "H.4.1", pass_index));
    }
    narrow(
        1 + 3 * geometry.num_lf_groups()
            + NUM_DEQUANT_MATRICES as u64
            + geometry.num_groups() * pass_index
            + group_index,
    )
}

fn check_lf_group_index(geometry: &FrameGeometry, lf_group_index: u64) -> Result<()> {
    if lf_group_index >= geometry.num_lf_groups() {
        return Err(FrameError::out_of_range(
            "lf_group_index",
            "H.4.1",
            lf_group_index,
        ));
    }
    Ok(())
}

/// H.4.1: property 1 is the stream index, narrowed to the property width.
///
/// Mirrors `decode.rs`'s private `stream_index` helper (duplicated on
/// purpose; see the module doc).
fn narrow(value: u64) -> Result<u32> {
    u32::try_from(value).map_err(|_| FrameError::out_of_range("stream index", "H.4.1", value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use jpxl_core::geometry::GroupDim;
    use jpxl_core::limits::{AllocGuard, Limits};

    /// A 600x300 frame at the default `group_dim == 256`: `ceil(600/256) =
    /// 3` group columns, `ceil(300/256) = 2` group rows (the last of each
    /// partial), so 6 groups; 1x1 LF groups (600, 300 < 2048); two passes.
    /// Small enough to hand-verify, wide enough that `num_groups > 1`.
    fn multi_group_geometry() -> FrameGeometry {
        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        FrameGeometry::derive(
            crate::frame::geometry::GroupLayout {
                width: 600,
                height: 300,
                upsampling: 1,
                lf_level: 0,
                group_dim: GroupDim::D256,
                num_passes: 2,
            },
            &limits,
            &mut guard,
        )
        .expect("600x300 at group_dim 256 is well within limits")
    }

    /// A geometry with more than one LF group, so `lf_coefficients` and
    /// friends have a real range to check. 2048*2 = 4096 forces a 2x1 LF grid.
    fn multi_lf_group_geometry() -> FrameGeometry {
        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        FrameGeometry::derive(
            crate::frame::geometry::GroupLayout {
                width: 4096,
                height: 2048,
                upsampling: 1,
                lf_level: 0,
                group_dim: GroupDim::D256,
                num_passes: 1,
            },
            &limits,
            &mut guard,
        )
        .expect("4096x2048 at group_dim 256 is well within limits")
    }

    #[test]
    fn global_modular_is_always_zero() {
        assert_eq!(global_modular(), 0);
    }

    #[test]
    fn hand_computed_values_over_a_multi_group_geometry() {
        let g = multi_group_geometry();
        assert_eq!(g.num_lf_groups(), 1);
        assert_eq!(g.num_groups(), 6); // 3 columns x 2 rows
        assert_eq!(g.num_passes(), 2);

        // LF coefficients / ModularLfGroup / HFMetadata: only lf_group_index 0
        // exists, so each collapses to its formula's constant term.
        assert_eq!(lf_coefficients(&g, 0).expect("valid stream index"), 1);
        assert_eq!(modular_lf_group(&g, 0).expect("valid stream index"), 1 + 1); // 1 + num_lf_groups
        assert_eq!(hf_metadata(&g, 0).expect("valid stream index"), 1 + 2); // 1 + 2*num_lf_groups

        // RAW dequant tables: 1 + 3*num_lf_groups + parameter_index = 4 + p.
        assert_eq!(dequant_table(&g, 0).expect("valid stream index"), 4);
        assert_eq!(dequant_table(&g, 16).expect("valid stream index"), 20);

        // ModularGroup: 1 + 3*1 + 17 + 6*pass + group = 21 + 6*pass + group.
        assert_eq!(modular_group(&g, 0, 0).expect("valid stream index"), 21);
        assert_eq!(modular_group(&g, 0, 5).expect("valid stream index"), 26);
        assert_eq!(modular_group(&g, 1, 0).expect("valid stream index"), 27);
        assert_eq!(modular_group(&g, 1, 5).expect("valid stream index"), 32);
    }

    #[test]
    fn hand_computed_values_over_multiple_lf_groups() {
        let g = multi_lf_group_geometry();
        assert_eq!(g.num_lf_groups(), 2);

        assert_eq!(lf_coefficients(&g, 0).expect("valid stream index"), 1);
        assert_eq!(lf_coefficients(&g, 1).expect("valid stream index"), 2);

        assert_eq!(modular_lf_group(&g, 0).expect("valid stream index"), 1 + 2);
        assert_eq!(
            modular_lf_group(&g, 1).expect("valid stream index"),
            1 + 2 + 1
        );

        assert_eq!(hf_metadata(&g, 0).expect("valid stream index"), 1 + 4);
        assert_eq!(hf_metadata(&g, 1).expect("valid stream index"), 1 + 4 + 1);

        // RAW dequant tables now start at 1 + 3*2 = 7.
        assert_eq!(dequant_table(&g, 0).expect("valid stream index"), 7);
        assert_eq!(dequant_table(&g, 16).expect("valid stream index"), 23);

        // ModularGroup starts right where the 17 RAW slots end: 7 + 17 = 24.
        assert_eq!(modular_group(&g, 0, 0).expect("valid stream index"), 24);
    }

    #[test]
    fn out_of_range_indices_are_rejected() {
        let g = multi_group_geometry();
        assert!(lf_coefficients(&g, 1).is_err());
        assert!(modular_lf_group(&g, 1).is_err());
        assert!(hf_metadata(&g, 1).is_err());
        assert!(dequant_table(&g, 17).is_err());
        assert!(modular_group(&g, 0, 6).is_err());
        assert!(modular_group(&g, 2, 0).is_err());
    }

    /// Distinct streams get distinct indices: enumerate every stream index
    /// H.4.1 defines for a geometry with more than one of everything and
    /// assert the set has no collisions.
    #[test]
    fn distinct_streams_get_distinct_indices() {
        let g = multi_lf_group_geometry();
        let mut seen = std::collections::HashSet::new();
        let mut insert = |label: &str, value: u32| {
            assert!(seen.insert(value), "stream index {value} reused by {label}");
        };

        insert("GlobalModular", global_modular());
        for lf in 0..g.num_lf_groups() {
            insert(
                "LF coefficients",
                lf_coefficients(&g, lf).expect("valid stream index"),
            );
        }
        for lf in 0..g.num_lf_groups() {
            insert(
                "ModularLfGroup",
                modular_lf_group(&g, lf).expect("valid stream index"),
            );
        }
        for lf in 0..g.num_lf_groups() {
            insert(
                "HFMetadata",
                hf_metadata(&g, lf).expect("valid stream index"),
            );
        }
        for p in 0..NUM_DEQUANT_MATRICES as u64 {
            insert(
                "RAW dequant table",
                dequant_table(&g, p).expect("valid stream index"),
            );
        }
        for pass in 0..u64::from(g.num_passes()) {
            for group in 0..g.num_groups() {
                insert(
                    "ModularGroup",
                    modular_group(&g, pass, group).expect("valid stream index"),
                );
            }
        }
    }

    #[test]
    fn narrow_rejects_a_value_past_u32() {
        assert!(narrow(u64::from(u32::MAX) + 1).is_err());
        assert_eq!(
            narrow(u64::from(u32::MAX)).expect("valid stream index"),
            u32::MAX
        );
    }
}
