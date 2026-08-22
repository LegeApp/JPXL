//! Deterministic fan-out for the metric's independent work items.
//!
//! Every parallel stage in this crate is a set of *independent* items — row
//! bands of a plane, or whole planes — whose outputs are disjoint and whose
//! partial sums are reduced afterwards in item order. The executor therefore
//! only has to run the items; it never reduces anything. That is what keeps a
//! score bit-identical across one worker, four, or none (Contract A of
//! `jpegxl-rs.decision.optimization-determinism-contract`).
//!
//! Item counts never depend on the worker count: bands have a fixed height
//! ([`crate::bands::BAND_ROWS`]) and plane items are fixed by the algorithm,
//! so the partition — and with it every floating-point reduction order — is a
//! property of the image, not of the host.

/// Runs `items` independent closures, in any order and on any workers.
pub trait BandExecutor: Sync {
    /// Calls `f(i)` exactly once for every `i in 0..items`, returning when all
    /// have completed.
    fn run(&self, items: usize, f: &(dyn Fn(usize) + Sync));
}

/// Runs every item on the calling thread, in index order.
#[derive(Debug, Clone, Copy, Default)]
pub struct SerialExecutor;

impl BandExecutor for SerialExecutor {
    fn run(&self, items: usize, f: &(dyn Fn(usize) + Sync)) {
        for i in 0..items {
            f(i);
        }
    }
}

/// Runs items on scoped standard threads, `workers` at a time, handing out
/// indices round-robin. Used by tests to prove executor independence without
/// the encoder's pool; production callers use the encoder executor.
#[derive(Debug, Clone, Copy)]
pub struct ScopedThreadExecutor {
    /// Number of threads to spawn per `run`.
    pub workers: usize,
}

impl BandExecutor for ScopedThreadExecutor {
    fn run(&self, items: usize, f: &(dyn Fn(usize) + Sync)) {
        let workers = self.workers.clamp(1, items.max(1));
        if workers == 1 {
            SerialExecutor.run(items, f);
            return;
        }
        std::thread::scope(|scope| {
            for worker in 0..workers {
                scope.spawn(move || {
                    let mut i = worker;
                    while i < items {
                        f(i);
                        i += workers;
                    }
                });
            }
        });
    }
}

#[cfg(feature = "encode-executor")]
impl BandExecutor for jpxl_encode::EncodeExecutor {
    fn run(&self, items: usize, f: &(dyn Fn(usize) + Sync)) {
        let outcome: Result<Vec<()>, core::convert::Infallible> = self.map_ordered(items, |i| {
            f(i);
            Ok(())
        });
        // `Infallible` cannot be constructed, so the map cannot fail.
        let _ = outcome;
    }
}
