//! Unit-bearing newtypes at the JPEG-1 coefficient/coordinate boundaries.
//!
//! AGENTS.md §6 bans bare `u8`/`usize` where a mix-up is a real bug. The traps
//! this codec must not fall into are the classic ones: a zig-zag stream index
//! read as a natural (raster) block position, a component's *identifier* (the
//! `Ci` byte from SOF, an arbitrary label) confused with its *index* in the
//! frame's component list, and a DC prediction difference confused with a DC
//! value. Each gets its own type.

/// The natural (raster, row-major) position of a coefficient inside an 8×8
/// block, `0..=63`. Element 0 is the DC term.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Natural(pub u8);

/// A position `0..=63` along the zig-zag scan order in which a block's
/// coefficients travel on the wire (10918-1 Figure A.6 / A.3.6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZigZag(pub u8);

/// The zig-zag → natural permutation (10918-1 Annex A, Figure A.6).
///
/// `ZIGZAG_TO_NATURAL[k]` is the natural index of the coefficient that is
/// `k`-th in zig-zag order.
pub const ZIGZAG_TO_NATURAL: [u8; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, //
    17, 24, 32, 25, 18, 11, 4, 5, //
    12, 19, 26, 33, 40, 48, 41, 34, //
    27, 20, 13, 6, 7, 14, 21, 28, //
    35, 42, 49, 56, 57, 50, 43, 36, //
    29, 22, 15, 23, 30, 37, 44, 51, //
    58, 59, 52, 45, 38, 31, 39, 46, //
    53, 60, 61, 54, 47, 55, 62, 63, //
];

impl ZigZag {
    /// Maps this zig-zag position to its natural (raster) position.
    #[must_use]
    // The index is masked to 0..=63 and `ZIGZAG_TO_NATURAL` has exactly 64
    // entries, so it is always in range.
    #[allow(clippy::indexing_slicing)]
    pub fn to_natural(self) -> Natural {
        Natural(ZIGZAG_TO_NATURAL[self.0 as usize & 63])
    }
}

/// A component's identifier — the arbitrary `Ci` label byte from the SOF
/// header (10918-1 B.2.2). Not an index into any array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComponentId(pub u8);

/// A component's *index* in the frame's component list, `0..Nf`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComponentIndex(pub usize);

/// A restart interval in minimum-coded-units (10918-1 B.2.4.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RestartInterval(pub u16);
