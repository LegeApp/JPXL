//! The event-sink boundary between the coefficient walk and entropy coding
//! (`Encoder-plan1.md` §9.1).
//!
//! I.4's coefficient traversal — channel order, nonzero prediction, `prev`
//! state, coefficient order, block context — is the hardest normative loop in
//! the encoder, and it must be walked at least twice: once to census raw
//! values for histogram training, once to emit symbols. Writing it twice is
//! how the two copies drift apart. So it is written once, against a sink:
//!
//! ```text
//! event walk ──▶ HfEventSink ──┬──▶ CensusSink     (raw values → policy train)
//!                              ├──▶ TokenCensus    (jpxl-entropy: ANS tables)
//!                              ├──▶ SymbolEncoder  (jpxl-entropy: emit)
//!                              └──▶ trace          (debug)
//! ```
//!
//! # Two census types on purpose
//!
//! [`CensusSink`] (this module) counts **raw** PackSigned values per
//! pre-context. Hybrid-uint configuration is a *cluster* property and is not
//! known yet when policy trains, so raw is the only legal census at that
//! stage.
//!
//! `jpxl_entropy::encode::TokenCensus` counts **tokens** after a hybrid-uint
//! config is fixed, which is what ANS table construction needs. The write
//! path builds one via an `HfEventSink` adapter once the plan's configurations
//! exist. Unifying the two types would either force premature tokenization
//! (wrong) or force ANS to re-tokenize twice (waste). They share only the
//! [`HfEventSink`] event shape.

use crate::vardct::ids::PreContextId;

/// A consumer of I.4's coefficient events.
///
/// Both methods take the *pre-clustering* context. Mapping a pre-context to a
/// cluster is the sink's business — the census sink does not map at all, and
/// the writer maps through the context map it was built with.
pub trait HfEventSink {
    /// I.4's leading per-channel `non_zeros` symbol for one varblock.
    fn nonzeros(&mut self, context: PreContextId, value: u32);

    /// One quantized coefficient, already in `PackSigned` form.
    fn coefficient(&mut self, context: PreContextId, value: u32);
}

/// Counts raw values per pre-context: `Encoder-plan1.md` §9.2's census.
///
/// Values are counted **raw**, not tokenized, because the hybrid-uint
/// configuration is a property of the cluster and the clusters do not exist
/// yet when the census runs. Small values — the overwhelming majority — go in
/// a dense inline bucket array; the tail is kept sparse and sorted so a
/// pathological coefficient cannot allocate a huge histogram.
#[derive(Debug, Clone, Default)]
pub struct CensusSink {
    contexts: Vec<RawHistogram>,
}

/// One pre-context's raw value distribution.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RawHistogram {
    small: [u32; 32],
    tail: Vec<(u32, u32)>,
}

impl RawHistogram {
    /// Counts one occurrence of `value`.
    pub fn add(&mut self, value: u32) {
        if let Ok(index) = usize::try_from(value)
            && let Some(slot) = self.small.get_mut(index)
        {
            *slot += 1;
            return;
        }
        match self.tail.binary_search_by_key(&value, |&(v, _)| v) {
            Ok(index) => {
                if let Some(entry) = self.tail.get_mut(index) {
                    entry.1 += 1;
                }
            }
            Err(index) => self.tail.insert(index, (value, 1)),
        }
    }

    /// How many times `value` was counted.
    #[must_use]
    pub fn count(&self, value: u32) -> u32 {
        usize::try_from(value)
            .ok()
            .and_then(|i| self.small.get(i))
            .copied()
            .unwrap_or_else(|| {
                self.tail
                    .binary_search_by_key(&value, |&(v, _)| v)
                    .ok()
                    .and_then(|i| self.tail.get(i))
                    .map_or(0, |&(_, c)| c)
            })
    }

    /// Every counted value and its count, in ascending value order.
    ///
    /// The bridge to `jpxl-entropy`'s own census type: a histogram that cannot
    /// be enumerated can only be queried about values the caller already
    /// guessed, which is not enough to build a code.
    pub fn iter(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.small
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c != 0)
            .map(|(v, &c)| (u32::try_from(v).unwrap_or(u32::MAX), c))
            .chain(self.tail.iter().copied())
    }

    /// The largest value counted, if any.
    #[must_use]
    pub fn max_value(&self) -> Option<u32> {
        self.iter().map(|(v, _)| v).last()
    }

    /// Total number of symbols counted.
    #[must_use]
    pub fn total(&self) -> u64 {
        let small: u64 = self.small.iter().map(|&c| u64::from(c)).sum();
        small + self.tail.iter().map(|&(_, c)| u64::from(c)).sum::<u64>()
    }
}

impl CensusSink {
    /// A census over `contexts` pre-contexts.
    #[must_use]
    pub fn new(contexts: usize) -> Self {
        Self {
            contexts: vec![RawHistogram::default(); contexts],
        }
    }

    /// One pre-context's histogram.
    #[must_use]
    pub fn histogram(&self, context: PreContextId) -> Option<&RawHistogram> {
        usize::try_from(context.get())
            .ok()
            .and_then(|i| self.contexts.get(i))
    }

    /// Number of pre-contexts.
    #[must_use]
    pub fn len(&self) -> usize {
        self.contexts.len()
    }

    /// Whether the census covers no contexts.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.contexts.is_empty()
    }

    fn add(&mut self, context: PreContextId, value: u32) {
        if let Ok(index) = usize::try_from(context.get())
            && let Some(histogram) = self.contexts.get_mut(index)
        {
            histogram.add(value);
        }
    }
}

impl HfEventSink for CensusSink {
    fn nonzeros(&mut self, context: PreContextId, value: u32) {
        self.add(context, value);
    }

    fn coefficient(&mut self, context: PreContextId, value: u32) {
        self.add(context, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_census_counts_small_and_tail_values_in_the_right_context() {
        let mut census = CensusSink::new(4);
        census.nonzeros(PreContextId::new(0), 3);
        census.coefficient(PreContextId::new(0), 3);
        census.coefficient(PreContextId::new(0), 4_000_000);
        census.coefficient(PreContextId::new(3), 31);
        // Past the end: dropped, never a panic and never miscounted elsewhere.
        census.coefficient(PreContextId::new(9), 1);

        let first = census.histogram(PreContextId::new(0)).expect("in range");
        assert_eq!(first.count(3), 2);
        assert_eq!(first.count(4_000_000), 1);
        assert_eq!(first.count(5), 0);
        assert_eq!(first.total(), 3);

        let last = census.histogram(PreContextId::new(3)).expect("in range");
        assert_eq!(last.count(31), 1);
        assert_eq!(census.histogram(PreContextId::new(4)), None);
        assert_eq!(census.len(), 4);
    }

    #[test]
    fn the_tail_stays_sorted_however_it_is_filled() {
        let mut histogram = RawHistogram::default();
        for value in [99u32, 40, 1_000_000, 40, 32] {
            histogram.add(value);
        }
        assert_eq!(histogram.count(40), 2);
        assert_eq!(histogram.count(32), 1);
        assert_eq!(histogram.count(1_000_000), 1);
        assert_eq!(histogram.total(), 5);
        let values: Vec<u32> = histogram.tail.iter().map(|&(v, _)| v).collect();
        let mut sorted = values.clone();
        sorted.sort_unstable();
        assert_eq!(values, sorted);
    }
}
