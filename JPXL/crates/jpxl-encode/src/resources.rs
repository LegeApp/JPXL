//! Resource policy for coarse parallelism (Opt-P).
//!
//! One owner for the worker budget: callers set [`EncodeResources::threads`]
//! instead of spawning ad-hoc pools. Section emission maps independent work
//! units and **reduces in fixed TOC / raster order** so Contract A holds
//! across thread counts.
//!
//! # Defaults
//!
//! With the default **`parallel`** feature, [`EncodeResources::default`] is
//! [`EncodeResources::auto`]: group/section axis with
//! [`std::thread::available_parallelism`] workers (clamped per work item).
//! Without `parallel`, the default is fully serial.
//!
//! With the default **`parallel`** feature, workers use a capped **rayon**
//! pool. Without it, [`std::thread::scope`] is used.

use core::num::NonZeroUsize;

/// Which coarse axis may use more than one worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ParallelAxis {
    /// Everything serial.
    Serial,
    /// Independent LF / pass-group section bodies after globals are ready.
    ///
    /// This is the default axis when [`EncodeResources::auto`] is used under
    /// the `parallel` feature.
    #[default]
    Groups,
}

/// Cap on workers for one encode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodeResources {
    /// Maximum worker threads for parallel section emission. `1` (or `0`) is
    /// fully serial.
    pub threads: usize,
    /// Which axis may fan out.
    pub axis: ParallelAxis,
}

impl Default for EncodeResources {
    /// [`auto`](Self::auto) when the `parallel` feature is on, otherwise
    /// [`serial`](Self::serial).
    fn default() -> Self {
        Self::auto()
    }
}

impl EncodeResources {
    /// Fully serial encode (Contract A baseline / single-thread checks).
    #[must_use]
    pub const fn serial() -> Self {
        Self {
            threads: 1,
            axis: ParallelAxis::Serial,
        }
    }

    /// Group/section axis with up to `threads` workers.
    ///
    /// `threads <= 1` collapses to serial axis.
    #[must_use]
    pub fn groups(threads: usize) -> Self {
        let threads = threads.max(1);
        Self {
            threads,
            axis: if threads <= 1 {
                ParallelAxis::Serial
            } else {
                ParallelAxis::Groups
            },
        }
    }

    /// Host-aware default: group axis with [`available_parallelism`](std::thread::available_parallelism)
    /// workers when the `parallel` feature is enabled; otherwise serial.
    #[must_use]
    pub fn auto() -> Self {
        #[cfg(feature = "parallel")]
        {
            let n = std::thread::available_parallelism()
                .map(NonZeroUsize::get)
                .unwrap_or(1);
            Self::groups(n.max(1))
        }
        #[cfg(not(feature = "parallel"))]
        {
            Self::serial()
        }
    }

    /// Effective worker count for `ready` independent items.
    ///
    /// Never exceeds `threads` or `ready`; returns 1 when the axis is serial
    /// or there is only one work item.
    #[must_use]
    pub fn workers_for(self, ready: usize) -> usize {
        if matches!(self.axis, ParallelAxis::Serial) || self.threads <= 1 || ready <= 1 {
            return 1;
        }
        self.threads.min(ready).max(1)
    }

    /// Whether group/section emission may fan out.
    #[must_use]
    pub fn parallel_groups(self) -> bool {
        matches!(self.axis, ParallelAxis::Groups) && self.threads > 1
    }
}

/// Map `0..n` with `f`, reducing results in index order (Contract A).
///
/// # Errors
///
/// The first `Err` in index order is returned.
pub(crate) fn ordered_map<T, E, F>(n: usize, workers: usize, f: F) -> Result<Vec<T>, E>
where
    T: Send,
    E: Send,
    F: Fn(usize) -> Result<T, E> + Sync,
{
    if n == 0 {
        return Ok(Vec::new());
    }
    let workers = NonZeroUsize::new(workers.max(1))
        .map(|w| w.get().min(n))
        .unwrap_or(1);

    if workers == 1 {
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(f(i)?);
        }
        return Ok(out);
    }

    #[cfg(feature = "parallel")]
    {
        ordered_map_rayon(n, workers, f)
    }
    #[cfg(not(feature = "parallel"))]
    {
        ordered_map_std(n, workers, f)
    }
}

#[cfg(feature = "parallel")]
fn ordered_map_rayon<T, E, F>(n: usize, workers: usize, f: F) -> Result<Vec<T>, E>
where
    T: Send,
    E: Send,
    F: Fn(usize) -> Result<T, E> + Sync,
{
    use rayon::prelude::*;
    let pool_started = crate::vardct::diagnostics::enabled().then(std::time::Instant::now);

    // Local pool capped at `workers` so one encode does not oversubscribe the
    // process when the caller asked for a small budget.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .unwrap_or_else(|_| {
            // Fall back to a one-thread pool rather than panicking on exotic hosts.
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("single-thread rayon pool")
        });
    if let Some(pool_started) = pool_started {
        crate::vardct::diagnostics::note_pool_build(
            u64::try_from(pool_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
        );
    }

    let mut slots: Vec<Option<Result<T, E>>> = (0..n).map(|_| None).collect();
    pool.install(|| {
        slots
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, slot)| *slot = Some(f(i)));
    });

    let mut out = Vec::with_capacity(n);
    for (i, slot) in slots.into_iter().enumerate() {
        match slot {
            Some(Ok(v)) => out.push(v),
            Some(Err(e)) => return Err(e),
            None => out.push(f(i)?),
        }
    }
    Ok(out)
}

#[cfg(not(feature = "parallel"))]
fn ordered_map_std<T, E, F>(n: usize, workers: usize, f: F) -> Result<Vec<T>, E>
where
    T: Send,
    E: Send,
    F: Fn(usize) -> Result<T, E> + Sync,
{
    let mut slots: Vec<Option<Result<T, E>>> = (0..n).map(|_| None).collect();
    let chunk = n.div_ceil(workers);
    let f = &f;
    std::thread::scope(|scope| {
        let mut rest = slots.as_mut_slice();
        let mut base = 0usize;
        while base < n {
            let end = (base + chunk).min(n);
            let len = end - base;
            let (mine, tail) = rest.split_at_mut(len);
            rest = tail;
            let start = base;
            scope.spawn(move || {
                for (offset, slot) in mine.iter_mut().enumerate() {
                    let i = start + offset;
                    *slot = Some(f(i));
                }
            });
            base = end;
        }
    });

    let mut out = Vec::with_capacity(n);
    for (i, slot) in slots.into_iter().enumerate() {
        match slot {
            Some(Ok(v)) => out.push(v),
            Some(Err(e)) => return Err(e),
            None => out.push(f(i)?),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_map_serial_and_parallel_agree() {
        let f = |i: usize| -> Result<usize, &'static str> { Ok(i * i + 3) };
        let serial = ordered_map(20, 1, f).expect("serial");
        let parallel = ordered_map(20, 4, f).expect("parallel");
        assert_eq!(serial, parallel);
    }

    #[test]
    fn first_error_is_lowest_index() {
        let f =
            |i: usize| -> Result<usize, usize> { if i == 3 || i == 7 { Err(i) } else { Ok(i) } };
        assert_eq!(ordered_map(10, 4, f), Err(3));
        assert_eq!(ordered_map(10, 1, f), Err(3));
    }

    #[test]
    fn workers_for_clamps() {
        let r = EncodeResources::groups(8);
        assert_eq!(r.workers_for(3), 3);
        assert_eq!(r.workers_for(100), 8);
        assert_eq!(EncodeResources::serial().workers_for(100), 1);
    }

    #[test]
    fn auto_is_groups_when_parallel_feature() {
        let a = EncodeResources::auto();
        #[cfg(feature = "parallel")]
        {
            assert!(a.threads >= 1);
            if a.threads > 1 {
                assert!(a.parallel_groups());
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            assert_eq!(a, EncodeResources::serial());
        }
    }
}
