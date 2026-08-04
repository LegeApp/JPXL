//! Unit-bearing newtypes for everything a VarDCT plan counts.
//!
//! `AGENTS.md` bans bare `usize`/`u32` at transform, quantization and
//! coordinate boundaries, and the plan IR is nothing but such boundaries: a
//! pass index, a group index, an order ID, a cluster index and a block index
//! are all small integers that mean entirely different things. The types below
//! make substituting one for another a compile error.
//!
//! The ones with a legal range — [`GlobalScale`], [`QuantLf`], [`HfMul`] — are
//! *checked* newtypes: the constructor is the only way in, and it rejects a
//! value the corresponding syntax element cannot carry. That is why validation
//! of those fields does not appear in `validate`: it already happened.

use core::num::NonZeroU32;

use crate::vardct::error::{PlanError, PlanResult};

/// Largest `global_scale` I.2's `U32(1 + u(11), 2049 + u(11), 4097 + u(12),
/// 8193 + u(16))` can express.
pub const MAX_GLOBAL_SCALE: u32 = 8193 + 65_535;

/// Largest `quant_lf` I.2's `U32(16, 1 + u(5), 1 + u(8), 1 + u(16))` can
/// express.
pub const MAX_QUANT_LF: u32 = 65_536;

/// Largest `HfMul`. G.2.4 stores `mul = HfMul - 1` as a Modular sample, which
/// is an `i32`, so `HfMul` stops one past `i32::MAX`.
pub const MAX_HF_MUL: u32 = i32::MAX as u32;

/// Largest `extra_precision`: G.2.2 reads it as `u(2)`.
pub const MAX_EXTRA_PRECISION: u8 = 3;

/// Largest `Sharpness` sample: J.4.3's sigma lookup has eight entries.
pub const MAX_SHARPNESS: u8 = 7;

macro_rules! plain_id {
    ($(#[$meta:meta])* $name:ident, $repr:ty) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name($repr);

        impl $name {
            /// Wraps a raw index.
            #[must_use]
            pub const fn new(value: $repr) -> Self {
                Self(value)
            }

            /// The raw index.
            #[must_use]
            pub const fn get(self) -> $repr {
                self.0
            }

            /// The raw index widened for arithmetic against collection lengths.
            #[must_use]
            pub const fn index(self) -> u64 {
                self.0 as u64
            }
        }
    };
}

plain_id!(
    /// A varblock's position in its LF group's `BlockInfo` column order —
    /// which is also its position in G.2.4's greedy placement walk.
    BlockId,
    u32
);
plain_id!(
    /// An LF group's raster index in the frame's LF-group grid.
    LfGroupId,
    u32
);
plain_id!(
    /// A pass group's raster index in the frame's group grid. Named for the
    /// HF data it carries: G.4's `PassGroup` section is where an HF group's
    /// coefficients live.
    HfGroupId,
    u32
);
plain_id!(
    /// A pass index in `[0, num_passes)` (F.6).
    PassId,
    u8
);
plain_id!(
    /// A Table I.7 Order ID in `[0, 13)`.
    OrderId,
    u8
);
plain_id!(
    /// An entropy cluster index: what a context map maps a pre-context to
    /// (C.2.2), bounded at 255.
    ClusterId,
    u8
);
plain_id!(
    /// An HF pre-clustering context index (I.3.3, I.4).
    ///
    /// **Deviation from `Encoder-plan1.md`,** which types this `u16`. I.3.3
    /// sizes the pre-clustered distribution list at
    /// `495 * num_hf_presets * nb_block_ctx`, and with `nb_block_ctx` up to 16
    /// that exceeds `u16::MAX` at nine presets. `u32` is the width the
    /// arithmetic actually needs.
    PreContextId,
    u32
);
plain_id!(
    /// An HF preset index in `[0, num_hf_presets)` (I.2.6, I.4's `hfp`).
    PresetId,
    u32
);

/// A signed chroma-from-luma factor, as stored in `XFromY`/`BFromY` (G.2.4).
///
/// I.6 turns it into `k = base_correlation + factor / colour_factor`; the
/// stored form is a Modular sample, so any `i32` is representable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct CflFactor(i32);

impl CflFactor {
    /// Wraps a stored factor.
    #[must_use]
    pub const fn new(value: i32) -> Self {
        Self(value)
    }

    /// The stored factor.
    #[must_use]
    pub const fn get(self) -> i32 {
        self.0
    }
}

macro_rules! checked_scalar {
    ($(#[$meta:meta])* $name:ident, $max:expr, $what:literal, $clause:literal) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(NonZeroU32);

        impl $name {
            /// The largest value the syntax element can carry.
            pub const MAX: u32 = $max;

            /// The smallest legal value, one.
            pub const MIN: Self = Self(NonZeroU32::MIN);

            /// Checks and wraps a value.
            ///
            /// # Errors
            ///
            /// [`PlanError::OutOfRange`] if `value` is zero or above
            /// [`Self::MAX`] — i.e. if no legal encoding of the field exists.
            pub fn new(value: u32) -> PlanResult<Self> {
                if value > Self::MAX {
                    return Err(PlanError::out_of_range($what, $clause, i64::from(value)));
                }
                NonZeroU32::new(value)
                    .map(Self)
                    .ok_or_else(|| PlanError::out_of_range($what, $clause, 0))
            }

            /// The value.
            #[must_use]
            pub const fn get(self) -> u32 {
                self.0.get()
            }
        }
    };
}

checked_scalar!(
    /// I.2's `global_scale`: the frame-wide quantization scale.
    GlobalScale,
    MAX_GLOBAL_SCALE,
    "global_scale",
    "I.2"
);
checked_scalar!(
    /// I.2's `quant_lf`: the LF-plane quantization step.
    QuantLf,
    MAX_QUANT_LF,
    "quant_lf",
    "I.2"
);
checked_scalar!(
    /// A varblock's HF quantization multiplier, `HfMul = 1 + mul` (G.2.4).
    HfMul,
    MAX_HF_MUL,
    "HfMul",
    "G.2.4"
);

impl HfMul {
    /// The `mul` sample G.2.4 stores in `BlockInfo`'s second row.
    #[must_use]
    pub const fn stored_mul(self) -> i32 {
        // `HfMul <= i32::MAX` by construction, so the subtraction is exact.
        (self.get() - 1) as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_checked_scalar_rejects_zero_and_the_first_unrepresentable_value() {
        assert!(GlobalScale::new(1).is_ok());
        assert!(GlobalScale::new(MAX_GLOBAL_SCALE).is_ok());
        assert!(matches!(
            GlobalScale::new(0),
            Err(PlanError::OutOfRange {
                what: "global_scale",
                ..
            })
        ));
        assert!(matches!(
            GlobalScale::new(MAX_GLOBAL_SCALE + 1),
            Err(PlanError::OutOfRange {
                what: "global_scale",
                ..
            })
        ));
        assert!(matches!(
            QuantLf::new(MAX_QUANT_LF + 1),
            Err(PlanError::OutOfRange {
                what: "quant_lf",
                ..
            })
        ));
        assert!(matches!(
            HfMul::new(0),
            Err(PlanError::OutOfRange { what: "HfMul", .. })
        ));
    }

    #[test]
    fn hf_mul_lowers_to_the_block_info_sample_it_came_from() {
        assert_eq!(HfMul::new(1).expect("legal").stored_mul(), 0);
        assert_eq!(HfMul::new(42).expect("legal").stored_mul(), 41);
        assert_eq!(
            HfMul::new(MAX_HF_MUL).expect("legal").stored_mul(),
            i32::MAX - 1
        );
    }
}
