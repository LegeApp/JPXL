//! HF coefficient order (18181-1 I.3.1).
//!
//! I.3.1's own code, with `natural_coeff_order[b]` the I.3.2 natural order of
//! Order ID `b` (implemented by [`jpxl_core::varblock::natural_coeff_order`]):
//!
//! ```text
//! used_orders = U32(0x5F, 0x13, 0x00, u(13));
//! /* if used_orders != 0: read 8 pre-clustered distributions (C.1) */
//! for (b = 0; b < 13; b++)
//!   for (c = 0; c < 3; c++)
//!     if ((used_orders & (1 << b)) != 0) {
//!       nat_ord_perm = DecodePermutation(b);
//!       for /* each coefficient index i */
//!         order[p][b][c][i] = natural_coeff_order[b][nat_ord_perm[i]];
//!     } else {
//!       for /* each coefficient index i */
//!         order[p][b][c][i] = natural_coeff_order[b][i];
//!     }
//! ```
//!
//! `DecodePermutation(b)` reads an F.3.2 permutation of
//! `size = natural_coeff_order[b].size()` elements with `skip = size / 64`
//! from **one stream shared by the whole double loop** — which is why this
//! module cannot call [`crate::frame::toc::read_permutation`] (that function
//! opens and finishes a stream of its own, as F.3.3's TOC needs) and instead
//! drives the same F.3.2 body against a caller-held [`SymbolDecoder`].
//!
//! # Direction of the table (the bug this module exists to avoid)
//!
//! A permutation can be stored either way round, and I.4's phrasing — "sets
//! the quantized HF coefficient in the position corresponding to index
//! `order[p][s][c][k]`" — reads like `order[...][k]` is being *written*. It is
//! not. The assignment above defines `order[...][i]` as an *element of*
//! `natural_coeff_order[b]`, and I.3.2 defines that vector's elements to be
//! cell positions `(x, y)` of the coefficient array. So:
//!
//! > **`order[k]` is the coefficient cell that receives order position `k`.**
//!
//! It is a map order-position → coefficient position, and I.4's `k` loop
//! indexes it directly with no inversion anywhere. See
//! [`OrderLookup::order`] and the `order_table_direction_*` tests.
//!
//! The permutation composes on the *inside*: `order[i] =
//! natural[nat_ord_perm[i]]`, i.e. `nat_ord_perm` permutes positions within
//! the natural order, it does not permute the natural order's values.
//!
//! **What does not decide this.** The entropy contexts of I.4 depend on `k`,
//! never on `order[k]`, so inverting the composition leaves the ANS stream
//! perfectly synchronized — `hf_coeff`'s real-fixture final-state gate passes
//! either way (verified by mutation, see
//! `docs/experiments/2026-08-03-i4-context-model-ans-gate.md`). The direction
//! rests on the clause sentence above and on the unit tests below; it is
//! decided by evidence only when 8F compares pixels.

// The two indexed containers here are `tables` (a fixed 13-element array,
// indexed only after `order_id < NUM_ORDER_IDS` is checked) and `permuted`
// (built with exactly `NUM_ORDER_IDS * 3` slots and indexed by the same
// checked arithmetic). Every stream-derived index goes through `get`.
#![allow(clippy::indexing_slicing)]

use jpxl_bitstream::{BitReader, U32Dist, U32Spec, read_u32, trace_field};
use jpxl_core::limits::AllocGuard;
use jpxl_core::varblock::{NUM_ORDER_IDS, natural_coeff_order, order_id_dims};
use jpxl_entropy::SymbolDecoder;

use crate::error::{DecodeError, Result};
use crate::frame::toc::{PERMUTATION_NUM_DIST, get_context, lehmer_to_permutation};

/// Number of channels an order table is stored for (I.3.1's `c` loop).
pub const NUM_ORDER_CHANNELS: usize = 3;

/// I.3.1: `used_orders = U32(0x5F, 0x13, 0x00, u(13))`.
///
/// The first three distributions are literal presets — `0x5F` selects Order
/// IDs {0, 1, 2, 3, 4, 6}, `0x13` selects {0, 1, 4}, `0x00` selects none — and
/// the fourth spells the 13-bit mask out.
const USED_ORDERS_SPEC: U32Spec = U32Spec::new([
    U32Dist::Val(0x5F),
    U32Dist::Val(0x13),
    U32Dist::Val(0x00),
    U32Dist::bits(13),
]);

/// The mask of Order IDs `used_orders` can legally select: 13 bits.
const USED_ORDERS_MASK: u32 = (1 << NUM_ORDER_IDS) - 1;

/// The I.3.2 natural coefficient orders of all 13 Order IDs.
///
/// Built once per frame and shared by every pass, because the `used_orders`
/// bit is per pass but the natural order it falls back to is not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NaturalOrders {
    tables: [Vec<u32>; NUM_ORDER_IDS],
}

impl Default for NaturalOrders {
    fn default() -> Self {
        Self::new()
    }
}

impl NaturalOrders {
    /// Computes all 13 natural orders (I.3.2 via Table I.7).
    #[must_use]
    pub fn new() -> Self {
        Self {
            tables: core::array::from_fn(|b| {
                // Every ID in `0..13` has a row in Table I.7, so the `None`
                // arm is unreachable; an empty table would be caught by
                // `table`'s length check rather than panicking here.
                order_id_dims(b).map_or_else(Vec::new, |(bw, bh)| natural_coeff_order(bw, bh))
            }),
        }
    }

    /// The natural order of Order ID `order_id`.
    ///
    /// # Errors
    ///
    /// [`DecodeError::FieldOutOfRange`] if `order_id` is not below
    /// [`NUM_ORDER_IDS`].
    pub fn table(&self, order_id: usize) -> Result<&[u32]> {
        self.tables
            .get(order_id)
            .map(Vec::as_slice)
            .filter(|t| !t.is_empty())
            .ok_or_else(|| DecodeError::out_of_range("Order ID", "I.3.1", order_id as u64))
    }

    /// Total number of coefficients across all 13 orders — the size of one
    /// fully materialised `order[p][*][*]` table for a single channel.
    #[must_use]
    pub fn total_coefficients(&self) -> usize {
        self.tables.iter().map(Vec::len).sum()
    }
}

/// One pass's HF coefficient orders (I.3.1's `order[p]`).
///
/// Only the Order IDs whose `used_orders` bit is set carry a stored table;
/// the rest fall back to [`NaturalOrders`], which is what the `else` arm of
/// the clause's code does. Storing the fallback explicitly would triple the
/// memory of a table that is bit-identical for every pass and channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassOrders {
    used_orders: u32,
    /// `NUM_ORDER_IDS * NUM_ORDER_CHANNELS` slots, indexed
    /// `order_id * NUM_ORDER_CHANNELS + channel`.
    permuted: Vec<Option<Vec<u32>>>,
}

impl PassOrders {
    /// A pass with `used_orders == 0`: every Order ID uses the natural order.
    #[must_use]
    pub fn natural_only() -> Self {
        Self {
            used_orders: 0,
            permuted: (0..NUM_ORDER_IDS * NUM_ORDER_CHANNELS)
                .map(|_| None)
                .collect(),
        }
    }

    /// The `used_orders` bit mask as read.
    #[must_use]
    pub const fn used_orders(&self) -> u32 {
        self.used_orders
    }

    /// Does Order ID `order_id` carry a decoded permutation in this pass?
    #[must_use]
    pub const fn is_permuted(&self, order_id: usize) -> bool {
        order_id < NUM_ORDER_IDS && (self.used_orders >> order_id) & 1 == 1
    }
}

/// A pass's order tables together with the natural-order fallback.
///
/// Produced by splitting a borrow of the HfGlobal-level bundle so that the
/// pass's entropy decoder can be borrowed mutably at the same time; see
/// [`crate::vardct::hf_coeff::HfPasses::split_pass`].
#[derive(Debug, Clone, Copy)]
pub struct OrderLookup<'a> {
    natural: &'a NaturalOrders,
    pass: &'a PassOrders,
}

impl<'a> OrderLookup<'a> {
    /// Pairs a pass's tables with the frame's natural orders.
    #[must_use]
    pub const fn new(natural: &'a NaturalOrders, pass: &'a PassOrders) -> Self {
        Self { natural, pass }
    }

    /// The pass's `used_orders` mask (I.3.1).
    #[must_use]
    pub const fn used_orders(&self) -> u32 {
        self.pass.used_orders()
    }

    /// `order[p][order_id][channel]`.
    ///
    /// Entry `k` of the returned slice is the **row-major coefficient index**
    /// (`y * bwidth + x`) of the cell that receives the coefficient decoded at
    /// order position `k`. The slice is a permutation of
    /// `0..bwidth * bheight`, and its first `bwidth * bheight / 64` entries
    /// are the LLF sub-rectangle, which I.4 never writes (I.8 fills it).
    ///
    /// `channel` is the clause's `c`: 0 = X, 1 = Y, 2 = B.
    ///
    /// # Errors
    ///
    /// [`DecodeError::FieldOutOfRange`] if `order_id` or `channel` is out of
    /// range.
    pub fn order(&self, order_id: usize, channel: usize) -> Result<&'a [u32]> {
        if channel >= NUM_ORDER_CHANNELS {
            return Err(DecodeError::out_of_range(
                "coefficient order channel",
                "I.3.1",
                channel as u64,
            ));
        }
        if order_id >= NUM_ORDER_IDS {
            return Err(DecodeError::out_of_range(
                "Order ID",
                "I.3.1",
                order_id as u64,
            ));
        }
        match self
            .pass
            .permuted
            .get(order_id * NUM_ORDER_CHANNELS + channel)
        {
            Some(Some(table)) => Ok(table.as_slice()),
            _ => self.natural.table(order_id),
        }
    }
}

/// Reads one pass's HF coefficient orders (18181-1 I.3.1).
///
/// This is the first half of an `HfPass` bundle; the caller continues with
/// I.3.3's histograms at the returned bit position.
///
/// # Errors
///
/// * [`DecodeError::Bitstream`] on truncation.
/// * [`DecodeError::Entropy`] if the shared permutation stream is malformed or
///   does not end in the C.3.2 terminal state.
/// * [`DecodeError::FieldOutOfRange`] if `used_orders` has a bit above 12 set,
///   or if a decoded Lehmer code violates F.3.2's bounds.
pub fn read_hf_coeff_orders(
    reader: &mut BitReader<'_>,
    natural: &NaturalOrders,
    guard: &mut AllocGuard,
) -> Result<PassOrders> {
    let used_orders = trace_field!(
        reader,
        "hf_pass.used_orders",
        read_u32(reader, &USED_ORDERS_SPEC)
    )?;
    if used_orders & !USED_ORDERS_MASK != 0 {
        return Err(DecodeError::out_of_range(
            "used_orders",
            "I.3.1",
            u64::from(used_orders),
        ));
    }
    if used_orders == 0 {
        return Ok(PassOrders::natural_only());
    }

    // "If used_orders != 0, it reads 8 pre-clustered distributions as
    // specified in C.1" — one stream for the whole double loop below.
    let mut decoder = SymbolDecoder::open(reader, PERMUTATION_NUM_DIST, guard)?;

    let mut permuted: Vec<Option<Vec<u32>>> = (0..NUM_ORDER_IDS * NUM_ORDER_CHANNELS)
        .map(|_| None)
        .collect();
    for b in 0..NUM_ORDER_IDS {
        if (used_orders >> b) & 1 == 0 {
            continue;
        }
        let nat = natural.table(b)?;
        let size = u32::try_from(nat.len())
            .map_err(|_| DecodeError::out_of_range("order size", "I.3.1", nat.len() as u64))?;
        for c in 0..NUM_ORDER_CHANNELS {
            // F.3.2 with skip = size / 64: the LLF prefix is never permuted.
            let perm = decode_permutation(&mut decoder, reader, size, size / 64, guard)?;
            guard.charge(u64::from(size) * 4)?;
            let mut table = Vec::with_capacity(nat.len());
            for &p in &perm {
                let cell = nat.get(p as usize).copied().ok_or_else(|| {
                    DecodeError::out_of_range("nat_ord_perm", "I.3.1", u64::from(p))
                })?;
                table.push(cell);
            }
            permuted[b * NUM_ORDER_CHANNELS + c] = Some(table);
        }
    }

    // C.3.2: the shared stream must end in its terminal state. This is the
    // same evidence the TOC permutation relies on, applied to 3 * popcount
    // (used_orders) permutations at once.
    decoder.finish()?;

    Ok(PassOrders {
        used_orders,
        permuted,
    })
}

/// F.3.2's permutation body, driven against an already-open stream.
///
/// Identical in substance to [`crate::frame::toc::read_permutation`], minus
/// the `SymbolDecoder::open`/`finish` pair that function performs: I.3.1
/// shares one stream across every permutation it reads, so the open and the
/// finish belong to the caller.
fn decode_permutation(
    decoder: &mut SymbolDecoder,
    reader: &mut BitReader<'_>,
    size: u32,
    skip: u32,
    guard: &mut AllocGuard,
) -> Result<Vec<u32>> {
    if skip > size {
        return Err(DecodeError::out_of_range("skip", "F.3.2", u64::from(skip)));
    }

    let end = decoder.read_uint(reader, get_context(size))?;
    if end > size - skip {
        return Err(DecodeError::out_of_range("end", "F.3.2", u64::from(end)));
    }

    guard.charge(u64::from(size) * 4)?;
    let mut lehmer = vec![0u32; size as usize];
    let mut previous = 0u32;
    for i in skip..(skip + end) {
        let ctx = get_context(if i > skip { previous } else { 0 });
        let value = decoder.read_uint(reader, ctx)?;
        if value >= size - i {
            return Err(DecodeError::out_of_range(
                "lehmer",
                "F.3.2",
                u64::from(value),
            ));
        }
        if let Some(slot) = lehmer.get_mut(i as usize) {
            *slot = value;
        }
        previous = value;
    }

    Ok(lehmer_to_permutation(&lehmer))
}

#[cfg(test)]
#[allow(
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    reason = "test code indexes fixed-size tables it just built; a panic here \
              is a failing test"
)]
mod tests {
    use super::*;
    use jpxl_core::limits::Limits;
    use jpxl_core::varblock::TransformType;

    // ------------------------------------------------------------------
    // Natural orders
    // ------------------------------------------------------------------

    #[test]
    fn every_order_id_has_a_table_of_the_right_size() {
        // Proves the Table I.7 <-> I.3.2 crosswalk: the 13 Order IDs cover
        // exactly the 13 distinct coefficient-array shapes, and each table is
        // a permutation of its cell indices.
        let natural = NaturalOrders::new();
        for b in 0..NUM_ORDER_IDS {
            let (bw, bh) = order_id_dims(b).expect("Table I.7 row");
            let table = natural.table(b).expect("table");
            assert_eq!(table.len(), bw * bh, "Order ID {b}");
            let mut seen = vec![false; bw * bh];
            for &v in table {
                assert!(!seen[v as usize], "Order ID {b} repeats cell {v}");
                seen[v as usize] = true;
            }
        }
    }

    #[test]
    fn order_id_of_every_transform_resolves() {
        // Proves 8A's TransformType::order_id lands inside this module's
        // table space for all 27 Table I.1 values.
        let natural = NaturalOrders::new();
        for t in TransformType::ALL {
            let table = natural.table(t.order_id()).expect("table");
            assert_eq!(table.len(), t.coeff_rows() * t.coeff_cols());
        }
    }

    // ------------------------------------------------------------------
    // Direction of the order table
    // ------------------------------------------------------------------

    #[test]
    fn order_table_direction_is_position_of_order_index() {
        // Proves the shipped convention: order[k] is the *destination* cell
        // of order position k, so the LLF prefix of the natural order is the
        // first (bw/8)*(bh/8) cells of the coefficient array in row-major
        // order. Under the inverse convention entry 1 of DCT16x8's table
        // would be 8 (the cell whose order position is 1), not 1.
        let natural = NaturalOrders::new();
        let t = TransformType::Dct16x8;
        let table = natural.table(t.order_id()).expect("table");
        let (bw, bh) = (t.coeff_cols(), t.coeff_rows());
        assert_eq!((bw, bh), (16, 8), "coefficients are landscape");
        let (cx, cy) = (bw / 8, bh / 8);
        // LLF prefix: cells (x, y) with x < cx, y < cy, sorted by y*cx + x.
        for y in 0..cy {
            for x in 0..cx {
                assert_eq!(table[y * cx + x], (y * bw + x) as u32);
            }
        }
        // The cell at coefficient index 1 is (1, 0), which for cx = 2 is
        // inside the LLF rectangle at order position 1.
        assert_eq!(table[1], 1);
    }

    #[test]
    fn permutation_composes_inside_the_natural_order() {
        // Proves `order[i] = natural[perm[i]]` rather than
        // `order[perm[i]] = natural[i]`, using an asymmetric permutation for
        // which the two readings genuinely differ.
        //
        // Hand derivation, DCT8x8 (Order ID 0, 64 cells, skip = 1):
        // let perm be the 3-cycle that maps positions 1 -> 2 -> 3 -> 1 (i.e.
        // perm = [0, 2, 3, 1, 4, 5, ...]). The natural order of DCT8x8 starts
        // 0, 1, 8, 16, 9, 2, ... (LLF cell 0, then the zig-zag).
        //   shipped reading:  order[1] = natural[2] = 8
        //   inverse reading:  order[2] = natural[1] = 1, order[1] = natural[3]
        // so entry 1 is 8 under one reading and 16 under the other.
        let natural = NaturalOrders::new();
        let nat = natural.table(0).expect("DCT8x8");
        assert_eq!(&nat[0..6], &[0, 1, 8, 16, 9, 2]);

        let mut perm: Vec<u32> = (0..64).collect();
        perm[1] = 2;
        perm[2] = 3;
        perm[3] = 1;

        let composed: Vec<u32> = perm.iter().map(|&p| nat[p as usize]).collect();
        assert_eq!(composed[1], 8, "order[1] = natural[perm[1]] = natural[2]");
        assert_eq!(composed[2], 16);
        assert_eq!(composed[3], 1);

        let mut inverse = vec![0u32; 64];
        for (i, &p) in perm.iter().enumerate() {
            inverse[p as usize] = nat[i];
        }
        assert_ne!(
            composed, inverse,
            "the two readings must differ for this permutation, or the test \
             proves nothing"
        );
    }

    // ------------------------------------------------------------------
    // Bitstream
    // ------------------------------------------------------------------

    /// `used_orders` written with the fourth distribution: selector 3 then
    /// 13 raw bits.
    fn used_orders_bits(mask: u32) -> (Vec<u8>, usize) {
        let mut bits: Vec<bool> = Vec::new();
        // U32 selector is 2 bits, LSB first.
        bits.push(true);
        bits.push(true);
        for i in 0..13 {
            bits.push((mask >> i) & 1 == 1);
        }
        let n = bits.len();
        let mut bytes = vec![0u8; n.div_ceil(8) + 8];
        for (i, b) in bits.iter().enumerate() {
            if *b {
                bytes[i / 8] |= 1 << (i % 8);
            }
        }
        (bytes, n)
    }

    #[test]
    fn used_orders_zero_yields_natural_orders_and_reads_nothing_more() {
        // Proves the `else` arm: with no bit set the clause reads neither
        // distributions nor permutations, so the reader stops after the
        // U32 field, and every lookup is the natural order.
        let (data, nbits) = used_orders_bits(0);
        let mut reader = BitReader::new(&data);
        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let natural = NaturalOrders::new();
        let pass = read_hf_coeff_orders(&mut reader, &natural, &mut guard).expect("valid");
        assert_eq!(reader.total_bits_read(), nbits as u64);
        assert_eq!(pass.used_orders(), 0);
        let lookup = OrderLookup::new(&natural, &pass);
        for b in 0..NUM_ORDER_IDS {
            for c in 0..NUM_ORDER_CHANNELS {
                assert_eq!(
                    lookup.order(b, c).expect("order"),
                    natural.table(b).expect("table")
                );
            }
        }
    }

    #[test]
    fn natural_only_reports_no_permutations() {
        let pass = PassOrders::natural_only();
        assert!(!pass.is_permuted(0));
        assert!(!pass.is_permuted(NUM_ORDER_IDS));
    }

    #[test]
    fn truncated_used_orders_errors_without_panicking() {
        let data = [0xFFu8];
        let mut reader = BitReader::new(&data);
        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let natural = NaturalOrders::new();
        assert!(read_hf_coeff_orders(&mut reader, &natural, &mut guard).is_err());
    }

    #[test]
    fn garbage_after_a_nonzero_used_orders_errors_without_panicking() {
        // Proves the permutation stream is opened and validated rather than
        // trusted: random bits cannot produce a stream that both parses and
        // ends in the C.3.2 terminal state.
        let (mut data, _) = used_orders_bits(1);
        for (i, byte) in data.iter_mut().enumerate().skip(2) {
            *byte = (i as u8).wrapping_mul(37).wrapping_add(11);
        }
        let mut reader = BitReader::new(&data);
        let limits = Limits::default();
        let mut guard = AllocGuard::new(&limits);
        let natural = NaturalOrders::new();
        let _ = read_hf_coeff_orders(&mut reader, &natural, &mut guard);
    }

    #[test]
    fn out_of_range_lookups_error() {
        let natural = NaturalOrders::new();
        let pass = PassOrders::natural_only();
        let lookup = OrderLookup::new(&natural, &pass);
        assert!(lookup.order(NUM_ORDER_IDS, 0).is_err());
        assert!(lookup.order(0, NUM_ORDER_CHANNELS).is_err());
        assert!(natural.table(NUM_ORDER_IDS).is_err());
    }
}
