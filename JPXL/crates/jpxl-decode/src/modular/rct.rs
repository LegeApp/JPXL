//! The reversible colour transform (18181-1 H.6.3).
//!
//! ```text
//! /* rct_type < 42 */
//! permutation = rct_type Idiv 7;
//! type        = rct_type Umod 7;
//! if (type == 6) {                    // YCoCg
//!   tmp = A - (C >> 1);
//!   E   = C + tmp;
//!   F   = tmp - (B >> 1);
//!   D   = F + B;
//! } else {
//!   if (type & 1)        C = C + A;
//!   if ((type >> 1) == 1) B = B + A;
//!   if ((type >> 1) == 2) B = B + ((A + C) >> 1);
//!   D = A; E = B; F = C;
//! }
//! V[ permutation                             Umod 3] = D;
//! V[(permutation + 1 + (permutation Idiv 3)) Umod 3] = E;
//! V[(permutation + 2 - (permutation Idiv 3)) Umod 3] = F;
//! ```
//!
//! `rct_type` therefore encodes two independent choices: a decorrelation
//! `type` in `[0, 7)` and one of six output `permutation`s.
//!
//! # Verifying the permutation formula
//!
//! H.6.3's example is the check that pins the index arithmetic down: with
//! `rct_type == 10`, `permutation = 1` and `type = 3`, pixels `(G, B', R')`
//! become `(R' + G, G, B' + G)`. Working it through — `type & 1` sets
//! `C = R' + G`, `(type >> 1) == 1` sets `B = B' + G`, and then
//! `V[1] = D = G`, `V[2] = E = B' + G`, `V[0] = F = R' + G` — reproduces the
//! example exactly. [`applies_the_specs_worked_example`] asserts it.
//!
//! [`applies_the_specs_worked_example`]: #
//!
//! # Order of the three assignments
//!
//! The `type` branch mutates `B` and `C` in place and the third condition reads
//! the *updated* `C`. `if (type & 1) C = C + A` runs first, so for `type == 5`
//! (`type & 1` true, `type >> 1 == 2`) the `B` update sees `C + A`, not the
//! original `C`. Reordering the conditions silently changes types 5 and 3.

use super::channel::Channel;
use super::error::{Result, malformed};
use super::weighted::narrow_to_i32;

/// H.6.3: `rct_type` is constrained to `[0, 42)`.
pub const MAX_RCT_TYPE: u32 = 42;

/// Applies one pixel of the inverse RCT.
///
/// Returns `V[0..3]` for input `(A, B, C)` under `rct_type`.
///
/// # Errors
///
/// [`ModularError::Malformed`](super::ModularError::Malformed) if `rct_type`
/// is 42 or larger.
pub fn inverse_pixel(rct_type: u32, a: i64, b: i64, c: i64) -> Result<[i64; 3]> {
    if rct_type >= MAX_RCT_TYPE {
        return Err(malformed!(
            "H.6.3: rct_type = {rct_type} is not below {MAX_RCT_TYPE}"
        ));
    }
    let permutation = rct_type / 7;
    let kind = rct_type % 7;

    let (d, e, f);
    if kind == 6 {
        // YCoCg. `>>` floors, which matters for negative chroma.
        let tmp = a - (c >> 1);
        e = c + tmp;
        f = tmp - (b >> 1);
        d = f + b;
    } else {
        let mut b = b;
        let mut c = c;
        if kind & 1 != 0 {
            c += a;
        }
        if (kind >> 1) == 1 {
            b += a;
        }
        if (kind >> 1) == 2 {
            b += (a + c) >> 1;
        }
        d = a;
        e = b;
        f = c;
    }

    let mut v = [0i64; 3];
    let p = permutation as usize;
    let half = p / 3;
    v[p % 3] = d;
    v[(p + 1 + half) % 3] = e;
    v[(p + 2 - half) % 3] = f;
    Ok(v)
}

/// Applies the inverse RCT to three channels starting at `begin_c`.
///
/// # Errors
///
/// [`ModularError::Malformed`](super::ModularError::Malformed) if the three
/// channels do not exist or do not have identical dimensions, or if `rct_type`
/// is out of range.
pub fn apply_inverse(channels: &mut [Channel], begin_c: usize, rct_type: u32) -> Result<()> {
    let end = begin_c
        .checked_add(3)
        .ok_or_else(|| malformed!("H.6.3: begin_c + 3 overflows"))?;
    let Some(block) = channels.get_mut(begin_c..end) else {
        return Err(malformed!(
            "H.6.3: RCT over channels {begin_c}..{end} but only {} exist",
            channels.len()
        ));
    };
    let (first, rest) = block
        .split_first_mut()
        .ok_or_else(|| malformed!("H.6.3: RCT needs three channels"))?;
    let spec = first.spec();
    if rest.iter().any(|c| c.spec() != spec) {
        return Err(malformed!(
            "H.6.3: the three RCT channels must have identical dimensions and shifts"
        ));
    }

    let (width, height) = (spec.width, spec.height);
    for y in 0..height {
        for x in 0..width {
            let a = i64::from(first.get(x, y));
            let b = i64::from(rest.first().map_or(0, |c| c.get(x, y)));
            let c = i64::from(rest.get(1).map_or(0, |c| c.get(x, y)));
            let v = inverse_pixel(rct_type, a, b, c)?;
            first.set(x, y, narrow_to_i32(v[0]));
            if let Some(ch) = rest.first_mut() {
                ch.set(x, y, narrow_to_i32(v[1]));
            }
            if let Some(ch) = rest.get_mut(1) {
                ch.set(x, y, narrow_to_i32(v[2]));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::channel::ChannelSpec;
    use super::*;

    /// The forward transform of each `type`, written independently from the
    /// inverse so a shared bug cannot cancel out. Returns `(A, B, C)` for an
    /// input `(D, E, F)` triple.
    fn forward_kind(kind: u32, d: i64, e: i64, f: i64) -> (i64, i64, i64) {
        if kind == 6 {
            // Inverse: tmp = A - (C>>1); E = C + tmp; F = tmp - (B>>1); D = F+B.
            // Solving: B = D - F, tmp = F + (B >> 1), C = E - tmp,
            //          A = tmp + (C >> 1).
            let b = d - f;
            let tmp = f + (b >> 1);
            let c = e - tmp;
            let a = tmp + (c >> 1);
            (a, b, c)
        } else {
            // Inverse sets D = A, E = B_after, F = C_after; invert the adds in
            // reverse order.
            let a = d;
            let mut b = e;
            let c_after = f;
            if (kind >> 1) == 1 {
                b -= a;
            }
            if (kind >> 1) == 2 {
                b -= (a + c_after) >> 1;
            }
            let mut c = c_after;
            if kind & 1 != 0 {
                c -= a;
            }
            (a, b, c)
        }
    }

    #[test]
    fn applies_the_specs_worked_example() {
        // H.6.3 EXAMPLE: rct_type == 10 turns (G, B', R') into
        // (R' + G, G, B' + G).
        let (g, bp, rp) = (100i64, 7i64, -20i64);
        let v = inverse_pixel(10, g, bp, rp).expect("rct_type 10");
        assert_eq!(v, [rp + g, g, bp + g]);
    }

    #[test]
    fn type_zero_is_the_identity_and_permutation_zero_keeps_the_order() {
        let v = inverse_pixel(0, 1, 2, 3).expect("rct_type 0");
        assert_eq!(v, [1, 2, 3]);
    }

    #[test]
    fn every_permutation_is_a_permutation() {
        // With type 0 the values pass through unchanged, so `V` is exactly the
        // permutation applied to (A, B, C). All six must be distinct bijections.
        let mut seen = Vec::new();
        for permutation in 0..6u32 {
            let v = inverse_pixel(permutation * 7, 0, 1, 2).expect("valid rct_type");
            let mut sorted = v;
            sorted.sort_unstable();
            assert_eq!(sorted, [0, 1, 2], "permutation {permutation} is a bijection");
            assert!(!seen.contains(&v), "permutation {permutation} is a duplicate");
            seen.push(v);
        }
        assert_eq!(seen.len(), 6);
    }

    #[test]
    fn every_variant_round_trips() {
        // For each of the 42 rct_types: take an output triple, run the
        // independently written forward transform, feed the result to the
        // inverse, and require the original back. This is the property that
        // matters — the RCT is *reversible* by definition.
        let samples = [
            (0i64, 0i64, 0i64),
            (1, 2, 3),
            (255, 0, 128),
            (-7, 13, -1),
            (-1000, -1001, 999),
            (1 << 20, -(1 << 19), 5),
        ];
        for rct_type in 0..MAX_RCT_TYPE {
            let permutation = (rct_type / 7) as usize;
            let kind = rct_type % 7;
            let half = permutation / 3;
            for &(v0, v1, v2) in &samples {
                let v = [v0, v1, v2];
                // Undo the permutation to recover (D, E, F).
                let d = v[permutation % 3];
                let e = v[(permutation + 1 + half) % 3];
                let f = v[(permutation + 2 - half) % 3];
                let (a, b, c) = forward_kind(kind, d, e, f);
                let back = inverse_pixel(rct_type, a, b, c).expect("valid");
                assert_eq!(back, v, "rct_type {rct_type} on {v:?}");
            }
        }
    }

    #[test]
    fn ycocg_matches_a_hand_computed_pixel() {
        // rct_type 6: permutation 0, type 6.
        // A = 100, B = 10, C = -6.
        //   tmp = 100 - (-6 >> 1) = 100 - (-3) = 103
        //   E   = -6 + 103 = 97
        //   F   = 103 - (10 >> 1) = 103 - 5 = 98
        //   D   = 98 + 10 = 108
        // permutation 0 -> V = [D, E, F] = [108, 97, 98]
        assert_eq!(inverse_pixel(6, 100, 10, -6).expect("ycocg"), [108, 97, 98]);
    }

    #[test]
    fn the_shift_in_ycocg_floors_for_negative_chroma() {
        // C = -1: (-1 >> 1) is -1 by flooring, not 0 by truncation. tmp is
        // therefore A + 1 rather than A.
        let v = inverse_pixel(6, 0, 0, -1).expect("ycocg");
        // tmp = 0 - (-1) = 1; E = -1 + 1 = 0; F = 1 - 0 = 1; D = 1 + 0 = 1
        assert_eq!(v, [1, 0, 1]);
    }

    #[test]
    fn type_five_sees_the_updated_c_when_updating_b() {
        // type 5: `type & 1` is 1 (C += A) and `type >> 1` is 2
        // (B += (A + C) >> 1) — using the already-updated C.
        // A = 10, B = 0, C = 4 -> C becomes 14, then B += (10 + 14) >> 1 = 12.
        let v = inverse_pixel(5, 10, 0, 4).expect("type 5");
        assert_eq!(v, [10, 12, 14]);
    }

    #[test]
    fn out_of_range_rct_type_is_rejected() {
        assert!(inverse_pixel(41, 0, 0, 0).is_ok());
        let err = inverse_pixel(42, 0, 0, 0).expect_err("42 is out of range");
        assert!(err.to_string().contains("H.6.3"));
    }

    #[test]
    fn apply_inverse_walks_every_pixel() {
        let mut channels = vec![
            Channel::from_samples(ChannelSpec::new(2, 1), vec![100, 1]).expect("2x1"),
            Channel::from_samples(ChannelSpec::new(2, 1), vec![7, 2]).expect("2x1"),
            Channel::from_samples(ChannelSpec::new(2, 1), vec![-20, 3]).expect("2x1"),
        ];
        apply_inverse(&mut channels, 0, 10).expect("rct_type 10");
        // Pixel 0: (100, 7, -20) -> (-20 + 100, 100, 7 + 100) = (80, 100, 107)
        // Pixel 1: (1, 2, 3)     -> (3 + 1, 1, 2 + 1)         = (4, 1, 3)
        assert_eq!(channels[0].samples(), &[80, 4]);
        assert_eq!(channels[1].samples(), &[100, 1]);
        assert_eq!(channels[2].samples(), &[107, 3]);
    }

    #[test]
    fn apply_inverse_rejects_mismatched_channels() {
        let mut channels = vec![
            Channel::from_samples(ChannelSpec::new(2, 1), vec![0, 0]).expect("2x1"),
            Channel::from_samples(ChannelSpec::new(2, 1), vec![0, 0]).expect("2x1"),
            Channel::from_samples(ChannelSpec::new(1, 1), vec![0]).expect("1x1"),
        ];
        assert!(apply_inverse(&mut channels, 0, 0).is_err());

        let mut two = vec![
            Channel::from_samples(ChannelSpec::new(1, 1), vec![0]).expect("1x1"),
            Channel::from_samples(ChannelSpec::new(1, 1), vec![0]).expect("1x1"),
        ];
        assert!(apply_inverse(&mut two, 0, 0).is_err(), "needs three channels");
    }
}
