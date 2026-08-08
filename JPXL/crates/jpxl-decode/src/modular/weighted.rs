//! Decoder-side glue for the self-correcting (weighted) predictor, 18181-1 H.5.
//!
//! The state machine itself (`WeightedState`, `WpHeader`'s shape, and the
//! resolved H.5.2 spec ambiguities) lives in
//! [`jpxl_core::modular_weighted`], shared with `jpxl-encode`'s Phase 4A
//! predictor scoring so the two call sites cannot drift apart. This module
//! keeps only what is inherently decode-only: reading the `WPHeader` bundle
//! (Table H.5) off the bitstream, and converting this crate's own
//! [`super::predictor::Neighbours`] (which walks a [`super::channel::Channel`],
//! unlike the shared crate's plain data holder) into
//! [`jpxl_core::modular_weighted::SelfCorrectingNeighbours`].

use jpxl_bitstream::{BitReader, read_bool, trace_field};
use jpxl_core::modular_weighted::SelfCorrectingNeighbours;
pub use jpxl_core::modular_weighted::{
    WeightedPrediction, WeightedState, WpHeader, error2weight, narrow_to_i32,
};

use super::error::Result;
use super::predictor::Neighbours;

impl From<Neighbours> for SelfCorrectingNeighbours {
    fn from(nb: Neighbours) -> Self {
        Self {
            w: nb.w,
            n: nb.n,
            nw: nb.nw,
            ne: nb.ne,
            nn: nb.nn,
            nee: nb.nee,
            ww: nb.ww,
        }
    }
}

/// Reads a `WPHeader` bundle (Table H.5).
///
/// # Errors
///
/// [`ModularError::Bitstream`](super::ModularError::Bitstream) at end of
/// input.
pub fn read_wp_header(reader: &mut BitReader<'_>) -> Result<WpHeader> {
    let default_wp = trace_field!(reader, "wp.default_wp", read_bool(reader))?;
    if default_wp {
        return Ok(WpHeader::default_wp());
    }
    let p5 = |reader: &mut BitReader<'_>, name| -> Result<i64> {
        Ok(i64::from(trace_field!(reader, name, reader.read_bits(5))?))
    };
    let p1 = p5(reader, "wp.wp_p1")?;
    let p2 = p5(reader, "wp.wp_p2")?;
    let p3 = [
        p5(reader, "wp.wp_p3a")?,
        p5(reader, "wp.wp_p3b")?,
        p5(reader, "wp.wp_p3c")?,
        p5(reader, "wp.wp_p3d")?,
        p5(reader, "wp.wp_p3e")?,
    ];
    let w4 = |reader: &mut BitReader<'_>, name| -> Result<i64> {
        Ok(i64::from(trace_field!(reader, name, reader.read_bits(4))?))
    };
    let w = [
        w4(reader, "wp.wp_w0")?,
        w4(reader, "wp.wp_w1")?,
        w4(reader, "wp.wp_w2")?,
        w4(reader, "wp.wp_w3")?,
    ];
    Ok(WpHeader { p1, p2, p3, w })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_wp_bit_costs_exactly_one_bit() {
        // Bool() true -> the eleven u(5)/u(4) fields are absent.
        let data = [0b0000_0001u8];
        let mut r = BitReader::new(&data);
        let h = read_wp_header(&mut r).expect("default_wp");
        assert_eq!(h, WpHeader::default_wp());
        assert_eq!(r.total_bits_read(), 1);
    }

    #[test]
    fn explicit_header_reads_seven_u5_then_four_u4() {
        // default_wp = 0, then p1..p3e = 1,2,3,4,5,6,7 as u(5), then
        // w0..w3 = 8,9,10,11 as u(4). Total 1 + 35 + 16 = 52 bits.
        let mut bits: Vec<u8> = Vec::new();
        let mut push = |value: u32, n: u32| {
            for i in 0..n {
                bits.push(((value >> i) & 1) as u8);
            }
        };
        push(0, 1);
        for v in 1..=7u32 {
            push(v, 5);
        }
        for v in 8..=11u32 {
            push(v, 4);
        }
        let mut bytes = vec![0u8; bits.len().div_ceil(8)];
        for (i, bit) in bits.iter().enumerate() {
            bytes[i / 8] |= bit << (i % 8);
        }

        let mut r = BitReader::new(&bytes);
        let h = read_wp_header(&mut r).expect("explicit wp header");
        assert_eq!(h.p1, 1);
        assert_eq!(h.p2, 2);
        assert_eq!(h.p3, [3, 4, 5, 6, 7]);
        assert_eq!(h.w, [8, 9, 10, 11]);
        assert_eq!(r.total_bits_read(), 52);
    }
}
