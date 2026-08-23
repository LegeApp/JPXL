//! Deterministic, compact prices for an already-trained HF entropy model.
//!
//! This is a ranking view, not a second entropy encoder. It consumes the
//! canonical I.4 event walk and prices each hybrid-uint token plus its extra
//! bits from the finalist's own histogram counts. The exact writer remains
//! the admission gate for every changed plan.

use jpxl_encode::vardct::HfEventSink;
use jpxl_encode::vardct::ids::PreContextId;
use jpxl_encode::vardct::plan::EntropyModelPlan;

use crate::{PolicyError, Result};

const COST_FRAC_BITS: u32 = 8;
const COST_ONE: u32 = 1 << COST_FRAC_BITS;

#[derive(Debug, Clone)]
struct ClusterCost {
    total_log2_q8: u32,
    count_log2_q8: Box<[u32]>,
    hybrid: jpxl_entropy::HybridUintConfig,
}

/// A fixed-point pricing view of one trained HF entropy model.
///
/// Costs use Q8 bits. Construction performs all logarithms with integer
/// arithmetic, so ranking is stable across platforms and libm versions.
#[derive(Debug, Clone)]
pub struct EntropyCostView {
    context_map: Box<[usize]>,
    clusters: Box<[ClusterCost]>,
}

impl EntropyCostView {
    /// Builds a view over `model`'s context map, histograms and hybrid configs.
    ///
    /// # Errors
    ///
    /// Returns [`PolicyError::Unsupported`] when the trained plan is internally
    /// inconsistent. Validation normally catches those shapes first.
    pub fn from_model(model: &EntropyModelPlan) -> Result<Self> {
        if model.histograms.len() != model.hybrid_uint.len() {
            return Err(PolicyError::Unsupported {
                what: "an entropy model with mismatched histograms and hybrid configs",
            });
        }
        let mut clusters = Vec::with_capacity(model.histograms.len());
        for (histogram, hybrid) in model.histograms.iter().zip(model.hybrid_uint.iter()) {
            let total = histogram
                .counts()
                .iter()
                .fold(0u64, |sum, &count| sum.saturating_add(u64::from(count)))
                .max(1);
            let hybrid = jpxl_entropy::HybridUintConfig::new(
                u32::from(hybrid.split_exponent),
                u32::from(hybrid.msb_in_token),
                u32::from(hybrid.lsb_in_token),
            )
            .map_err(|_| PolicyError::Unsupported {
                what: "an invalid trained hybrid-uint configuration",
            })?;
            clusters.push(ClusterCost {
                total_log2_q8: log2_q8(total),
                count_log2_q8: histogram
                    .counts()
                    .iter()
                    .map(|&count| log2_q8(u64::from(count.max(1))))
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
                hybrid,
            });
        }
        let context_map = model
            .context_map
            .iter()
            .map(|cluster| usize::from(cluster.get()))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        if context_map.iter().any(|&cluster| cluster >= clusters.len()) {
            return Err(PolicyError::Unsupported {
                what: "an entropy context mapped outside its trained clusters",
            });
        }
        Ok(Self {
            context_map,
            clusters: clusters.into_boxed_slice(),
        })
    }

    /// Prices one raw I.4 value in Q8 bits.
    pub fn cost_q8(&self, context: PreContextId, value: u32) -> Result<u32> {
        let context = usize::try_from(context.get()).unwrap_or(usize::MAX);
        let cluster_index =
            self.context_map
                .get(context)
                .copied()
                .ok_or(PolicyError::Unsupported {
                    what: "an HF event context outside the entropy cost view",
                })?;
        let cluster = self
            .clusters
            .get(cluster_index)
            .ok_or(PolicyError::Unsupported {
                what: "an HF event cluster outside the entropy cost view",
            })?;
        let split = cluster
            .hybrid
            .tokenize(value)
            .map_err(|_| PolicyError::Unsupported {
                what: "an HF event value outside its hybrid-uint configuration",
            })?;
        let token = usize::try_from(split.token).unwrap_or(usize::MAX);
        let count_log2 = cluster.count_log2_q8.get(token).copied().unwrap_or(0);
        Ok(cluster
            .total_log2_q8
            .saturating_sub(count_log2)
            .saturating_add(split.extra_bits.saturating_mul(COST_ONE)))
    }
}

/// A canonical-walk sink that totals [`EntropyCostView`] prices.
pub struct EntropyCostSink<'a> {
    view: &'a EntropyCostView,
    total_q8: u64,
    error: Option<PolicyError>,
}

impl<'a> EntropyCostSink<'a> {
    /// Starts an empty total.
    #[must_use]
    pub const fn new(view: &'a EntropyCostView) -> Self {
        Self {
            view,
            total_q8: 0,
            error: None,
        }
    }

    fn record(&mut self, context: PreContextId, value: u32) {
        if self.error.is_some() {
            return;
        }
        match self.view.cost_q8(context, value) {
            Ok(cost) => self.total_q8 = self.total_q8.saturating_add(u64::from(cost)),
            Err(error) => self.error = Some(error),
        }
    }

    /// Returns the accumulated Q8-bit total.
    pub fn finish(self) -> Result<u64> {
        self.error.map_or(Ok(self.total_q8), Err)
    }
}

impl HfEventSink for EntropyCostSink<'_> {
    fn nonzeros(&mut self, context: PreContextId, value: u32) {
        self.record(context, value);
    }

    fn coefficient(&mut self, context: PreContextId, value: u32) {
        self.record(context, value);
    }
}

/// `floor(log2(value) * 256)`, using only integer arithmetic.
fn log2_q8(value: u64) -> u32 {
    debug_assert!(value > 0);
    let integer = value.ilog2();
    let mut normalized = u128::from(value) << (63 - integer);
    let mut fraction = 0u32;
    for bit in (0..COST_FRAC_BITS).rev() {
        normalized = normalized.saturating_mul(normalized) >> 63;
        if normalized >= (1u128 << 64) {
            normalized >>= 1;
            fraction |= 1 << bit;
        }
    }
    integer.saturating_mul(COST_ONE).saturating_add(fraction)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jpxl_encode::vardct::ids::ClusterId;
    use jpxl_encode::vardct::plan::{HistogramPlan, HybridUintPlan};

    fn model(counts: Vec<u32>) -> EntropyModelPlan {
        EntropyModelPlan {
            context_map: vec![ClusterId::new(0)].into_boxed_slice(),
            histograms: vec![HistogramPlan::new(counts).expect("counts")].into_boxed_slice(),
            hybrid_uint: vec![HybridUintPlan::default()].into_boxed_slice(),
        }
    }

    #[test]
    fn integer_log2_q8_is_exact_at_powers_of_two() {
        for power in 0..=32 {
            assert_eq!(log2_q8(1u64 << power), power * COST_ONE);
        }
    }

    #[test]
    fn common_symbols_cost_less_and_extra_bits_are_charged() {
        let view = EntropyCostView::from_model(&model(vec![64, 16, 4, 1])).expect("view");
        let ctx = PreContextId::new(0);
        assert!(view.cost_q8(ctx, 0).expect("zero") < view.cost_q8(ctx, 1).expect("one"));
        assert!(view.cost_q8(ctx, 1).expect("one") < view.cost_q8(ctx, 3).expect("three"));
        assert!(view.cost_q8(ctx, 8).expect("eight") > view.cost_q8(ctx, 3).expect("three"));
    }

    #[test]
    fn invalid_model_shapes_are_rejected() {
        let mut broken = model(vec![1]);
        broken.hybrid_uint = Box::new([]);
        assert!(EntropyCostView::from_model(&broken).is_err());
    }
}
