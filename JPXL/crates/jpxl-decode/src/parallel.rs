//! Row-band partitioning for the decode hotspots.
//!
//! The VarDCT tail (IDCT scatter excluded) is a chain of pure per-sample
//! filters: EPF, gaborish, the Annex L colour conversion and the final
//! quantization each write every output sample exactly once from read-only
//! inputs. Splitting the rows into contiguous bands and filtering each band
//! on its own thread therefore cannot change a single pixel — the per-sample
//! operation order is identical, only the assignment of rows to threads
//! differs. The IDCT stage itself parallelizes over groups with a serial
//! scatter ([`crate::decode`] drives that through [`bands`] too).
//!
//! Deliberately `std::thread::scope` and nothing else: no rayon, no new
//! dependency, and the scope guarantees every worker has joined before the
//! band slices are used again. See `AGENTS.md` §6 (near-zero external
//! dependencies in the normative crates).

use std::ops::Range;

/// Rows of filter work that amortize one thread spawn.
///
/// A 4000-wide EPF row costs on the order of a millisecond; a spawn costs
/// microseconds. 64 rows keeps even the smallest threaded image firmly on
/// the winning side while leaving tiny conformance cases serial.
pub(crate) const MIN_ROWS_PER_WORKER: usize = 64;

/// Samples of trivial elementwise map work that amortize one thread spawn.
///
/// One transfer-function or quantization evaluation costs nanoseconds; 64K
/// samples ≈ 0.3ms per worker, well above spawn overhead while leaving
/// small images serial.
pub(crate) const MIN_MAP_SAMPLES_PER_WORKER: usize = 1 << 16;

/// How many workers to use for `items` of independent work.
///
/// Each worker gets at least `min_per_worker` items; the count is capped at
/// the machine's parallelism and at one worker per item. Small inputs stay
/// single-threaded with zero spawn overhead.
pub(crate) fn worker_count(items: usize, min_per_worker: usize) -> usize {
    if items == 0 {
        return 1;
    }
    let max = std::thread::available_parallelism().map_or(1, |n| n.get());
    (items / min_per_worker.max(1)).clamp(1, max).min(items)
}

/// Splits `0..count` into at most `workers` contiguous non-empty bands.
///
/// All bands but the last hold exactly `count.div_ceil(workers)` rows, so a
/// row-major plane's `chunks_mut(band_len * width)` yields precisely these
/// bands in order. Every index lands in exactly one band.
pub(crate) fn bands(count: usize, workers: usize) -> Vec<Range<usize>> {
    if count == 0 {
        return Vec::new();
    }
    let workers = workers.clamp(1, count);
    let len = count.div_ceil(workers);
    (0..count)
        .step_by(len)
        .map(|start| start..(start + len).min(count))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bands_tile_exactly_once() {
        // Even splits, ragged splits, primes, singletons, and the empty
        // frame: every index in exactly one band, bands in order, none
        // empty.
        for count in [0, 1, 2, 3, 7, 63, 64, 65, 100, 1000, 3000] {
            for workers in [0, 1, 2, 3, 7, 8, 16, 20, 64, 5000] {
                let result = bands(count, workers);
                assert!(
                    result.iter().all(|band| !band.is_empty()),
                    "count {count} workers {workers}"
                );
                // Flattened in order, the bands must be exactly 0..count:
                // no gaps, no overlaps, no empty bands.
                let tiled: Vec<usize> = result.iter().flat_map(|band| band.clone()).collect();
                let expected: Vec<usize> = (0..count).collect();
                assert_eq!(tiled, expected, "count {count} workers {workers}");
                assert!(result.len() <= workers.max(1).min(count.max(1)));
            }
        }
    }

    #[test]
    fn bands_match_chunks_mut() {
        // The property the filter call sites rely on: banding a plane with
        // `chunks_mut` gives these same row ranges.
        let (count, workers, width) = (3000usize, 20usize, 4000usize);
        let plane = vec![0.0f32; count * width];
        let len = count.div_ceil(workers.clamp(1, count));
        let chunk_rows: Vec<Range<usize>> = plane
            .chunks(len * width)
            .enumerate()
            .map(|(i, c)| i * len..i * len + c.len() / width)
            .collect();
        assert_eq!(chunk_rows, bands(count, workers));
    }

    #[test]
    fn small_images_stay_serial() {
        assert_eq!(worker_count(0, MIN_ROWS_PER_WORKER), 1);
        assert_eq!(worker_count(1, MIN_ROWS_PER_WORKER), 1);
        assert_eq!(worker_count(63, MIN_ROWS_PER_WORKER), 1);
        assert_eq!(worker_count(64, MIN_ROWS_PER_WORKER), 1);
        // One full band per worker past the threshold, capped at the
        // machine and at one worker per row.
        let max = std::thread::available_parallelism().map_or(1, |n| n.get());
        assert_eq!(worker_count(128, MIN_ROWS_PER_WORKER), 2.min(max).min(128));
        assert_eq!(worker_count(1_000_000, MIN_ROWS_PER_WORKER), max);
        // Group-sized units parallelize one group per worker.
        assert_eq!(worker_count(9, 1), 9.min(max));
        assert_eq!(worker_count(1, 1), 1);
    }
}
