//! Quantization tables (10918-1 B.2.4.1).
//!
//! A `DQT` segment stores, per table, a precision/id byte then 64 element
//! bytes in zig-zag order (`Pq = 0`: 8-bit elements; `Pq = 1`: 16-bit
//! big-endian elements). The elements are kept in the exact zig-zag order they
//! appear on the wire so re-emission is byte-identical; Phase B, which needs
//! them in natural order, can permute with [`ZigZag`].
//!
//! [`ZigZag`]: crate::units::ZigZag

/// One quantization table as stored in a `DQT` segment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuantTable {
    /// Element precision `Pq`: 0 = 8-bit, 1 = 16-bit.
    pub precision: u8,
    /// Destination identifier `Tq`, `0..=3`.
    pub id: u8,
    /// The 64 quantization step sizes in zig-zag order.
    pub values: [u16; 64],
}
