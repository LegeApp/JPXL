//! Neighbour extraction and the fourteen modular predictors (18181-1 H.3).
//!
//! ```text
//! Table H.2 — Neighbours used for prediction
//!        -2   -1    0   +1   +2
//!  -2               NN
//!  -1         NW    N    NE  NEE
//!   0   WW    W     c
//! ```
//!
//! ```text
//! Table H.3 — Modular predictors
//!  0  Zero          0
//!  1  West          W
//!  2  North         N
//!  3  Avg(W,N)      (W + N) Idiv 2
//!  4  Select        abs(N - NW) < abs(W - NW) ? W : N
//!  5  Gradient      clamp(W + N - NW, min(W, N), max(W, N))
//!  6  Self-correcting  (prediction + 3) >> 3      (see H.5)
//!  7  NorthEast     NE
//!  8  NorthWest     NW
//!  9  WestWest      WW
//! 10  Avg(W,NW)     (W + NW) Idiv 2
//! 11  Avg(N,NW)     (N + NW) Idiv 2
//! 12  Avg(N,NE)     (N + NE) Idiv 2
//! 13  AvgAll        (6*N - 2*NN + 7*W + WW + NEE + 3*NE + 8) Idiv 16
//! ```
//!
//! # Two divisions that are not the same division
//!
//! 18181-1 4.3 defines `Idiv` as division **rounded towards zero** and `>>` as
//! `floor(x / 2^s)`. They differ for negative operands, and Table H.3 uses both:
//! predictors 3 and 10–13 use `Idiv`, predictor 6 uses `>>`. Rust's `/` on
//! integers truncates and Rust's `>>` on signed integers is arithmetic, so each
//! maps directly — but they must not be swapped for each other.
//!
//! # Arithmetic width
//!
//! Samples are `i32` (H.1). Every predictor is evaluated in `i64` because
//! predictor 13 sums seven weighted neighbours, which overflows `i32` for
//! extreme sample values. H.1 states that 64-bit arithmetic is what the
//! intermediate predictor computations need.

use super::channel::Channel;
use super::error::{Result, malformed};

/// The number of predictors defined by Table H.3.
pub const NUM_PREDICTORS: u32 = 14;

/// A predictor selector from Table H.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Predictor {
    /// 0: constant zero.
    Zero = 0,
    /// 1: `W`.
    West = 1,
    /// 2: `N`.
    North = 2,
    /// 3: `(W + N) Idiv 2`.
    AvgWn = 3,
    /// 4: `abs(N - NW) < abs(W - NW) ? W : N`.
    Select = 4,
    /// 5: `clamp(W + N - NW, min(W, N), max(W, N))`.
    Gradient = 5,
    /// 6: the self-correcting predictor of H.5, `(prediction + 3) >> 3`.
    SelfCorrecting = 6,
    /// 7: `NE`.
    NorthEast = 7,
    /// 8: `NW`.
    NorthWest = 8,
    /// 9: `WW`.
    WestWest = 9,
    /// 10: `(W + NW) Idiv 2`.
    AvgWnw = 10,
    /// 11: `(N + NW) Idiv 2`.
    AvgNnw = 11,
    /// 12: `(N + NE) Idiv 2`.
    AvgNne = 12,
    /// 13: `(6*N - 2*NN + 7*W + WW + NEE + 3*NE + 8) Idiv 16`.
    AvgAll = 13,
}

impl Predictor {
    /// Maps a signalled predictor number to a [`Predictor`].
    ///
    /// # Errors
    ///
    /// [`ModularError::Malformed`](super::ModularError::Malformed) for a value
    /// with no row in Table H.3. MA-tree leaves and the palette `d_pred` field
    /// are both attacker-controlled, so this is a rejection and not a clamp.
    pub fn from_value(value: u32) -> Result<Self> {
        Ok(match value {
            0 => Self::Zero,
            1 => Self::West,
            2 => Self::North,
            3 => Self::AvgWn,
            4 => Self::Select,
            5 => Self::Gradient,
            6 => Self::SelfCorrecting,
            7 => Self::NorthEast,
            8 => Self::NorthWest,
            9 => Self::WestWest,
            10 => Self::AvgWnw,
            11 => Self::AvgNnw,
            12 => Self::AvgNne,
            13 => Self::AvgAll,
            other => {
                return Err(malformed!(
                    "H.3 Table H.3: predictor {other} is outside 0..{}",
                    NUM_PREDICTORS - 1
                ));
            }
        })
    }

    /// The value that would be signalled for this predictor.
    #[must_use]
    pub const fn value(self) -> u32 {
        self as u32
    }

    /// Whether this predictor consumes the H.5 self-correcting prediction.
    #[must_use]
    pub const fn uses_weighted(self) -> bool {
        matches!(self, Self::SelfCorrecting)
    }
}

/// The seven neighbours of Table H.2, with the H.3 edge cases already applied.
///
/// Every field is the *substituted* value: at an edge the spec does not leave
/// a neighbour undefined, it names a replacement, and that replacement is what
/// is stored here. Predictors therefore never re-check coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Neighbours {
    /// `W`: left, or `N`, or 0 at the origin.
    pub w: i64,
    /// `N`: above, or `W` on the first row.
    pub n: i64,
    /// `NW`: above-left, or `W`.
    pub nw: i64,
    /// `NE`: above-right, or `N`.
    pub ne: i64,
    /// `NN`: two above, or `N`.
    pub nn: i64,
    /// `NEE`: above and two right, or `NE`.
    pub nee: i64,
    /// `WW`: two left, or `W`.
    pub ww: i64,
}

impl Neighbours {
    /// Gathers the neighbours of `channel(x, y)` per H.3.
    ///
    /// The edge cases, verbatim from H.3:
    ///
    /// ```text
    /// W   = (x > 0 ? c(x-1, y) : (y > 0 ? c(x, y-1) : 0));
    /// N   = (y > 0 ? c(x, y-1) : W);
    /// NW  = (x > 0 and y > 0 ? c(x-1, y-1) : W);
    /// NE  = (x+1 < width and y > 0 ? c(x+1, y-1) : N);
    /// NN  = (y > 1 ? c(x, y-2) : N);
    /// NEE = (x+2 < width and y > 0 ? c(x+2, y-1) : NE);
    /// WW  = (x > 1 ? c(x-2, y) : W);
    /// ```
    ///
    /// Note that the substitutions cascade: `NEE` falls back to the *already
    /// substituted* `NE`, which itself may be `N`, which on the first row is
    /// `W`, which at the origin is zero.
    #[must_use]
    pub fn gather(channel: &Channel, x: u32, y: u32) -> Self {
        let width = channel.width();
        let at = |cx: u32, cy: u32| i64::from(channel.get(cx, cy));

        let w = if x > 0 {
            at(x - 1, y)
        } else if y > 0 {
            at(x, y - 1)
        } else {
            0
        };
        let n = if y > 0 { at(x, y - 1) } else { w };
        let nw = if x > 0 && y > 0 { at(x - 1, y - 1) } else { w };
        let ne = if x + 1 < width && y > 0 {
            at(x + 1, y - 1)
        } else {
            n
        };
        let nn = if y > 1 { at(x, y - 2) } else { n };
        let nee = if x + 2 < width && y > 0 {
            at(x + 2, y - 1)
        } else {
            ne
        };
        let ww = if x > 1 { at(x - 2, y) } else { w };

        Self {
            w,
            n,
            nw,
            ne,
            nn,
            nee,
            ww,
        }
    }

    /// `clamp(W + N - NW, min(W, N), max(W, N))` — the gradient of Table H.3
    /// row 5, also used by the "previous channel" properties of H.4.1.
    #[must_use]
    pub fn gradient(w: i64, n: i64, nw: i64) -> i64 {
        (w + n - nw).clamp(w.min(n), w.max(n))
    }

    /// Evaluates `prediction(x, y, predictor)` of Table H.3.
    ///
    /// `weighted` is the raw H.5.2 `prediction` value, in the 3-bit-shifted
    /// domain the self-correcting predictor works in. It is ignored by every
    /// predictor except number 6, but H.5.1 requires the self-correcting
    /// predictor to be evaluated for *every* sample regardless, so the caller
    /// always has a value to pass.
    #[must_use]
    pub fn predict(&self, predictor: Predictor, weighted: i64) -> i64 {
        let Self {
            w,
            n,
            nw,
            ne,
            nn,
            nee,
            ww,
        } = *self;
        match predictor {
            Predictor::Zero => 0,
            Predictor::West => w,
            Predictor::North => n,
            Predictor::AvgWn => (w + n) / 2,
            Predictor::Select => {
                if (n - nw).abs() < (w - nw).abs() {
                    w
                } else {
                    n
                }
            }
            Predictor::Gradient => Self::gradient(w, n, nw),
            // 4.3: `>>` is floor division, which is Rust's arithmetic shift.
            Predictor::SelfCorrecting => (weighted + 3) >> 3,
            Predictor::NorthEast => ne,
            Predictor::NorthWest => nw,
            Predictor::WestWest => ww,
            Predictor::AvgWnw => (w + nw) / 2,
            Predictor::AvgNnw => (n + nw) / 2,
            Predictor::AvgNne => (n + ne) / 2,
            // The coefficients sum to 16, which is the cross-source check that
            // caught the OCR of `WW` as `WH`: 6 - 2 + 7 + 1 + 1 + 3 = 16.
            Predictor::AvgAll => (6 * n - 2 * nn + 7 * w + ww + nee + 3 * ne + 8) / 16,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::channel::ChannelSpec;
    use super::*;

    /// A 4x3 channel whose samples are easy to name:
    ///
    /// ```text
    ///   0   1   2   3
    ///   4   5   6   7
    ///   8   9  10  11
    /// ```
    fn ramp() -> Channel {
        Channel::from_samples(ChannelSpec::new(4, 3), (0..12).collect()).expect("4x3")
    }

    #[test]
    fn interior_neighbours_are_the_table_h2_offsets() {
        // At (1, 2): W = c(0,2) = 8, N = c(1,1) = 5, NW = c(0,1) = 4,
        // NE = c(2,1) = 6, NN = c(1,0) = 1, NEE = c(3,1) = 7, WW = x>1? no
        // (x == 1), so WW = W = 8.
        let nb = Neighbours::gather(&ramp(), 1, 2);
        assert_eq!(
            nb,
            Neighbours {
                w: 8,
                n: 5,
                nw: 4,
                ne: 6,
                nn: 1,
                nee: 7,
                ww: 8,
            }
        );
    }

    #[test]
    fn fully_interior_neighbours() {
        // At (2, 2): W = 9, N = 6, NW = 5, NE = c(3,1) = 7, NN = c(2,0) = 2,
        // NEE = x+2 = 4 >= width, so NEE = NE = 7, WW = c(0,2) = 8.
        let nb = Neighbours::gather(&ramp(), 2, 2);
        assert_eq!(
            nb,
            Neighbours {
                w: 9,
                n: 6,
                nw: 5,
                ne: 7,
                nn: 2,
                nee: 7,
                ww: 8,
            }
        );
    }

    #[test]
    fn origin_substitutes_everything_down_to_zero() {
        // At (0, 0) nothing exists: W = 0, and N/NW/NE/NN/NEE/WW all cascade
        // down to it. This is the case that proves the substitutions chain.
        let nb = Neighbours::gather(&ramp(), 0, 0);
        assert_eq!(nb, Neighbours::default(), "every neighbour resolves to 0");
    }

    #[test]
    fn first_row_takes_w_then_n_for_everything() {
        // At (2, 0): W = c(1,0) = 1. y == 0 so N = W = 1, NW = W = 1,
        // NE = N = 1, NN = N = 1, NEE = NE = 1, and WW = c(0,0) = 0.
        let nb = Neighbours::gather(&ramp(), 2, 0);
        assert_eq!(
            nb,
            Neighbours {
                w: 1,
                n: 1,
                nw: 1,
                ne: 1,
                nn: 1,
                nee: 1,
                ww: 0,
            }
        );
    }

    #[test]
    fn first_column_takes_the_sample_above_as_w() {
        // At (0, 1): x == 0 and y > 0, so W = c(0, 0) = 0. N = c(0,0) = 0,
        // NW = W = 0, NE = c(1,0) = 1, NN = y > 1 false so NN = N = 0,
        // NEE = c(2,0) = 2, WW = W = 0.
        let nb = Neighbours::gather(&ramp(), 0, 1);
        assert_eq!(
            nb,
            Neighbours {
                w: 0,
                n: 0,
                nw: 0,
                ne: 1,
                nn: 0,
                nee: 2,
                ww: 0,
            }
        );
    }

    /// Hand-computed neighbourhood used by the predictor vectors below.
    const NB: Neighbours = Neighbours {
        w: 10,
        n: 20,
        nw: 14,
        ne: 30,
        nn: 40,
        nee: 50,
        ww: 6,
    };

    #[test]
    fn every_predictor_against_a_hand_computed_vector() {
        let p = |k: u32| {
            NB.predict(
                Predictor::from_value(k).expect("k is a valid predictor number"),
                123,
            )
        };

        assert_eq!(p(0), 0, "Zero");
        assert_eq!(p(1), 10, "West = W");
        assert_eq!(p(2), 20, "North = N");
        // (10 + 20) Idiv 2 = 30 Idiv 2 = 15
        assert_eq!(p(3), 15, "Avg(W,N)");
        // abs(N - NW) = abs(20 - 14) = 6; abs(W - NW) = abs(10 - 14) = 4.
        // 6 < 4 is false, so Select yields N = 20.
        assert_eq!(p(4), 20, "Select");
        // W + N - NW = 10 + 20 - 14 = 16, clamped to [min(10,20), max(10,20)]
        // = [10, 20] -> 16.
        assert_eq!(p(5), 16, "Gradient");
        // (123 + 3) >> 3 = 126 >> 3 = 15
        assert_eq!(p(6), 15, "Self-correcting");
        assert_eq!(p(7), 30, "NorthEast = NE");
        assert_eq!(p(8), 14, "NorthWest = NW");
        assert_eq!(p(9), 6, "WestWest = WW");
        // (10 + 14) Idiv 2 = 12
        assert_eq!(p(10), 12, "Avg(W,NW)");
        // (20 + 14) Idiv 2 = 17
        assert_eq!(p(11), 17, "Avg(N,NW)");
        // (20 + 30) Idiv 2 = 25
        assert_eq!(p(12), 25, "Avg(N,NE)");
        // 6*20 - 2*40 + 7*10 + 6 + 50 + 3*30 + 8
        //   = 120 - 80 + 70 + 6 + 50 + 90 + 8 = 264; 264 Idiv 16 = 16
        assert_eq!(p(13), 16, "AvgAll");
    }

    #[test]
    fn gradient_clamps_on_both_sides() {
        // W + N - NW below min(W, N):  1 + 2 - 100 = -97 -> clamped to 1.
        assert_eq!(Neighbours::gradient(1, 2, 100), 1);
        // W + N - NW above max(W, N):  1 + 2 - (-100) = 103 -> clamped to 2.
        assert_eq!(Neighbours::gradient(1, 2, -100), 2);
        // In range: 5 + 9 - 6 = 8, inside [5, 9].
        assert_eq!(Neighbours::gradient(5, 9, 6), 8);
    }

    #[test]
    fn idiv_truncates_towards_zero_but_the_shift_floors() {
        // Avg(W,N) with W = -3, N = 0: (-3 + 0) Idiv 2 = -1 (towards zero),
        // NOT -2 (which is what `>>` would give).
        let nb = Neighbours {
            w: -3,
            ..Neighbours::default()
        };
        assert_eq!(nb.predict(Predictor::AvgWn, 0), -1);

        // Predictor 6 uses `>>`, which floors: (-5 + 3) >> 3 = -2 >> 3 = -1.
        assert_eq!(
            Neighbours::default().predict(Predictor::SelfCorrecting, -5),
            -1
        );
        // And (-12 + 3) >> 3 = -9 >> 3 = -2 by flooring; truncation gives -1.
        assert_eq!(
            Neighbours::default().predict(Predictor::SelfCorrecting, -12),
            -2
        );
    }

    #[test]
    fn avgall_coefficients_sum_to_sixteen() {
        // A constant neighbourhood must be a fixed point of AvgAll, which is
        // exactly the statement that the coefficients sum to 16. This is the
        // invariant that resolves the `WW`/`WH` OCR damage in Table H.3.
        for v in [-1000i64, -7, 0, 1, 255, 100_000] {
            let nb = Neighbours {
                w: v,
                n: v,
                nw: v,
                ne: v,
                nn: v,
                nee: v,
                ww: v,
            };
            // (16*v + 8) Idiv 16 == v for v >= 0; for v < 0 truncation towards
            // zero also lands on v because |8| < 16.
            assert_eq!(nb.predict(Predictor::AvgAll, 0), v, "AvgAll fixes {v}");
        }
    }

    #[test]
    fn predictor_numbers_outside_the_table_are_rejected() {
        assert!(Predictor::from_value(13).is_ok());
        let err = Predictor::from_value(14).expect_err("14 has no row in Table H.3");
        assert!(err.to_string().contains("H.3"));
        assert!(Predictor::from_value(u32::MAX).is_err());
    }

    #[test]
    fn value_round_trips() {
        for k in 0..NUM_PREDICTORS {
            assert_eq!(Predictor::from_value(k).expect("valid").value(), k);
        }
    }
}
