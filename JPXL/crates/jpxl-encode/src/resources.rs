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

/// Request-scoped owner of the worker infrastructure used by an encode.
///
/// Construct this once, then reuse it across Count and Store emissions. The
/// fixed-index collection in [`ordered_map_with`] keeps output independent of
/// scheduling and worker count.
pub struct EncodeExecutor {
    resources: EncodeResources,
    #[cfg(feature = "parallel")]
    pool: Option<rayon::ThreadPool>,
}

impl core::fmt::Debug for EncodeExecutor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("EncodeExecutor")
            .field("resources", &self.resources)
            .finish_non_exhaustive()
    }
}

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

    /// Builds one executor for an encode request.
    #[must_use]
    pub fn executor(self) -> EncodeExecutor {
        EncodeExecutor::new(self)
    }
}

impl EncodeExecutor {
    /// Builds the worker pool once for this executor.
    #[must_use]
    pub fn new(resources: EncodeResources) -> Self {
        #[cfg(feature = "parallel")]
        {
            let pool_started = crate::vardct::diagnostics::enabled().then(std::time::Instant::now);
            let pool = resources
                .parallel_groups()
                .then(|| {
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(resources.threads)
                        .build()
                        .ok()
                })
                .flatten();
            if pool.is_some()
                && let Some(pool_started) = pool_started
            {
                crate::vardct::diagnostics::note_pool_build(
                    u64::try_from(pool_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                );
            }
            Self { resources, pool }
        }
        #[cfg(not(feature = "parallel"))]
        {
            Self { resources }
        }
    }

    /// Resource policy this executor was built from.
    #[must_use]
    pub const fn resources(&self) -> EncodeResources {
        self.resources
    }

    /// Maps independent work items and returns their results in index order.
    ///
    /// This is the shared coarse-grained execution boundary for planning and
    /// emission. The closure may run concurrently, but reduction is always in
    /// `0..n` order so worker count and scheduling cannot change the result.
    ///
    /// # Errors
    ///
    /// Returns the first error in index order.
    pub fn map_ordered<T, E, F>(&self, n: usize, f: F) -> Result<Vec<T>, E>
    where
        T: Send,
        E: Send,
        F: Fn(usize) -> Result<T, E> + Sync,
    {
        ordered_map_with(n, self, f)
    }

    fn workers_for(&self, ready: usize) -> usize {
        #[cfg(feature = "parallel")]
        if self.pool.is_none() {
            return 1;
        }
        self.resources.workers_for(ready)
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
    let executor = EncodeExecutor::new(EncodeResources::groups(workers.min(n).max(1)));
    ordered_map_with(n, &executor, f)
}

/// Map `0..n` on a persistent executor and reduce in index order.
pub(crate) fn ordered_map_with<T, E, F>(
    n: usize,
    executor: &EncodeExecutor,
    f: F,
) -> Result<Vec<T>, E>
where
    T: Send,
    E: Send,
    F: Fn(usize) -> Result<T, E> + Sync,
{
    if n == 0 {
        return Ok(Vec::new());
    }
    let workers = NonZeroUsize::new(executor.workers_for(n).max(1))
        .map(|w| w.get().min(n))
        .unwrap_or(1);

    if workers == 1 {
        return ordered_map_serial(n, f);
    }

    #[cfg(feature = "parallel")]
    {
        if let Some(pool) = executor.pool.as_ref() {
            return ordered_map_rayon(pool, n, f);
        }
        // Pool construction can fail under a host resource limit. Preserve
        // correctness and avoid an encoder panic by collapsing to serial.
        ordered_map_serial(n, f)
    }
    #[cfg(not(feature = "parallel"))]
    {
        ordered_map_std(n, workers, f)
    }
}

fn ordered_map_serial<T, E, F>(n: usize, f: F) -> Result<Vec<T>, E>
where
    F: Fn(usize) -> Result<T, E>,
{
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(f(i)?);
    }
    Ok(out)
}

#[cfg(feature = "parallel")]
fn ordered_map_rayon<T, E, F>(pool: &rayon::ThreadPool, n: usize, f: F) -> Result<Vec<T>, E>
where
    T: Send,
    E: Send,
    F: Fn(usize) -> Result<T, E> + Sync,
{
    use rayon::prelude::*;

    let mut slots: Vec<Option<Result<T, E>>> = (0..n).map(|_| None).collect();
    pool.install(|| {
        // One job per item: the items handed to these maps are coarse (a
        // section, a group, a band, a cluster table) and often unequal -- an
        // LF-group section is tens of pass-group sections -- so letting rayon
        // batch a contiguous run onto one worker (its default adaptive split)
        // can leave the heavy prefix serialised while the others idle. Item
        // granularity lets thieves take the heavy items individually.
        // Scheduling never touches results: every slot is written by index.
        slots
            .par_iter_mut()
            .with_max_len(1)
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

    #[cfg(feature = "parallel")]
    #[test]
    fn persistent_executor_builds_one_pool_for_several_maps() {
        crate::vardct::diagnostics::set_enabled(true);
        crate::vardct::diagnostics::reset();
        let executor = EncodeExecutor::new(EncodeResources::groups(4));
        let f = |i: usize| -> Result<usize, &'static str> { Ok(i + 1) };
        assert_eq!(ordered_map_with(8, &executor, f), Ok((1..=8).collect()));
        assert_eq!(ordered_map_with(5, &executor, f), Ok((1..=5).collect()));
        assert_eq!(
            crate::vardct::diagnostics::snapshot()
                .other
                .executor_pool_builds,
            1
        );
        crate::vardct::diagnostics::set_enabled(false);
        crate::vardct::diagnostics::reset();
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
