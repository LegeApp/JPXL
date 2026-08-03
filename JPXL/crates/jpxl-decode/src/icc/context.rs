//! The `IccContext()` function of 18181-1 E.4.1.
//!
//! E.4.1 entropy-codes the encoded ICC stream one byte at a time, choosing the
//! context from the byte position and the two previously decoded bytes. The
//! classification is a crude character-class model: letters and digits, the two
//! punctuation marks that appear in ICC text tags, and the extreme byte values
//! that dominate binary tag payloads.

/// Number of pre-clustered distributions E.4.1 reads for the ICC stream.
///
/// [`icc_context`] returns 0 for the first 129 bytes and otherwise
/// `1 + p1 + 8 * p2` with `p1 <= 7` and `p2 <= 4`, so the largest context is
/// `1 + 7 + 32 = 40` and there are 41 in total.
pub const NUM_ICC_CONTEXTS: usize = 41;

/// The byte index up to and including which every byte shares context 0
/// (18181-1 E.4.1: `if (i <= 128) return 0`).
///
/// This is the ICC header's 128 bytes plus the first byte after it.
const FLAT_CONTEXT_PREFIX: u64 = 128;

/// `IccContext(i, b1, b2)` (18181-1 E.4.1).
///
/// `i` is the index of the byte about to be decoded, `b1` the previous decoded
/// byte and `b2` the one before that, each 0 when it does not exist yet.
#[must_use]
pub const fn icc_context(i: u64, b1: u8, b2: u8) -> usize {
    if i <= FLAT_CONTEXT_PREFIX {
        return 0;
    }

    // The `b1` ladder has eight outcomes; note that the alphanumeric and
    // punctuation tests come first, so the numeric ranges below only ever see
    // non-text bytes.
    let p1 = if b1.is_ascii_alphabetic() {
        0
    } else if b1.is_ascii_digit() || b1 == b'.' || b1 == b',' {
        1
    } else if b1 <= 1 {
        2 + b1 as usize
    } else if b1 < 16 {
        4
    } else if b1 > 240 && b1 < 255 {
        5
    } else if b1 == 255 {
        6
    } else {
        7
    };

    // The `b2` ladder is coarser: five outcomes, and its low/high tests are
    // `< 16` and `> 240` without the separate 255 case.
    let p2 = if b2.is_ascii_alphabetic() {
        0
    } else if b2.is_ascii_digit() || b2 == b'.' || b2 == b',' {
        1
    } else if b2 < 16 {
        2
    } else if b2 > 240 {
        3
    } else {
        4
    };

    1 + p1 + p2 * 8
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Proves the flat prefix covers exactly indices 0..=128, so the first
    /// context-selected byte is index 129.
    #[test]
    fn the_first_129_bytes_share_context_zero() {
        for i in 0..=128u64 {
            assert_eq!(icc_context(i, b'q', 0xFF), 0);
        }
        assert_ne!(icc_context(129, b'q', b'q'), 0);
    }

    /// Proves the returned context is always a legal index into the 41
    /// distributions E.4.1 reads, over every possible (b1, b2) pair.
    #[test]
    fn every_context_is_within_the_declared_distribution_count() {
        let mut seen = [false; NUM_ICC_CONTEXTS];
        for b1 in 0..=255u8 {
            for b2 in 0..=255u8 {
                let ctx = icc_context(129, b1, b2);
                assert!(ctx < NUM_ICC_CONTEXTS, "context {ctx} out of range");
                #[expect(
                    clippy::indexing_slicing,
                    reason = "the assertion above proves the index is in range"
                )]
                {
                    seen[ctx] = true;
                }
            }
        }
        // Context 0 is unreachable past the prefix, every other one is used.
        assert!(!seen[0]);
        assert!(seen.iter().skip(1).all(|&s| s));
    }

    /// Pins the class boundaries that the E.4.1 ladder distinguishes: upper and
    /// lower case collapse to one class, digits join `.` and `,`, and the
    /// byte values 0 and 1 get a class each.
    #[test]
    fn classifies_the_boundary_bytes_of_the_ladder() {
        let ctx = |b1: u8| icc_context(129, b1, 0x80);
        // p2 = 4 for 0x80, so the base is 1 + 32 = 33.
        assert_eq!(ctx(b'a'), 33);
        assert_eq!(ctx(b'Z'), 33);
        assert_eq!(ctx(b'0'), 34);
        assert_eq!(ctx(b'.'), 34);
        assert_eq!(ctx(b','), 34);
        assert_eq!(ctx(0), 35);
        assert_eq!(ctx(1), 36);
        assert_eq!(ctx(2), 37);
        assert_eq!(ctx(15), 37);
        assert_eq!(ctx(16), 40); // falls through to the catch-all
        assert_eq!(ctx(240), 40);
        assert_eq!(ctx(241), 38);
        assert_eq!(ctx(254), 38);
        assert_eq!(ctx(255), 39);
    }
}
