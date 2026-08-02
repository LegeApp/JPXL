//! Field-level bit-position tracing.
//!
//! Enabled by the `trace` Cargo feature. When the feature is off, [`TraceLog`]
//! is a zero-sized type and every recording call compiles away, so the hot
//! read path is unaffected.
//!
//! The point of shipping this before any header parsing exists: when a header
//! field decodes to the wrong value, the useful question is almost always
//! "at which bit did we diverge from the reference?", and that is only
//! answerable if every field records the interval it consumed.
//!
//! Typical use:
//!
//! ```
//! # use jpxl_bitstream::{BitReader, trace_field};
//! let data = [0b0000_0101u8];
//! let mut r = BitReader::new(&data);
//! let bits = trace_field!(r, "size_selector", r.read_bits(2))?;
//! assert_eq!(bits, 0b01);
//! # Ok::<(), jpxl_bitstream::BitstreamError>(())
//! ```

/// One traced field: the bit interval `[start_bit, end_bit)` it consumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceEvent {
    /// Field name as given at the call site.
    pub name: &'static str,
    /// Reader position before the field was read.
    pub start_bit: u64,
    /// Reader position after the field was read.
    pub end_bit: u64,
}

impl TraceEvent {
    /// Number of bits the field consumed.
    #[must_use]
    pub const fn len_bits(&self) -> u64 {
        self.end_bit - self.start_bit
    }
}

/// Recorded field intervals attached to a [`BitReader`](crate::BitReader).
///
/// With the `trace` feature off this is a unit struct and all methods are
/// no-ops returning empty data.
#[cfg(feature = "trace")]
#[derive(Debug, Clone, Default)]
pub struct TraceLog {
    events: Vec<TraceEvent>,
}

/// Recorded field intervals attached to a [`BitReader`](crate::BitReader).
///
/// With the `trace` feature off this is a unit struct and all methods are
/// no-ops returning empty data.
#[cfg(not(feature = "trace"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct TraceLog;

impl TraceLog {
    /// Creates an empty log.
    #[must_use]
    pub fn new() -> Self {
        #[cfg(feature = "trace")]
        {
            Self { events: Vec::new() }
        }
        #[cfg(not(feature = "trace"))]
        {
            Self
        }
    }

    /// Records a field that spanned `[start_bit, end_bit)`.
    ///
    /// A no-op unless the `trace` feature is enabled.
    #[cfg_attr(
        not(feature = "trace"),
        expect(unused_variables, reason = "no-op when tracing is off")
    )]
    #[inline]
    pub fn record(&mut self, name: &'static str, start_bit: u64, end_bit: u64) {
        #[cfg(feature = "trace")]
        self.events.push(TraceEvent {
            name,
            start_bit,
            end_bit,
        });
    }

    /// All recorded events in read order; always empty without the `trace` feature.
    #[must_use]
    #[inline]
    pub fn events(&self) -> &[TraceEvent] {
        #[cfg(feature = "trace")]
        {
            &self.events
        }
        #[cfg(not(feature = "trace"))]
        {
            &[]
        }
    }

    /// Discards all recorded events.
    #[inline]
    pub fn clear(&mut self) {
        #[cfg(feature = "trace")]
        self.events.clear();
    }
}

/// Reads a field while recording the bit interval it consumed.
///
/// `trace_field!(reader, "name", expr)` evaluates `expr` (which is expected to
/// read from `reader`), samples [`BitReader::total_bits_read`] before and
/// after, and appends the interval to the reader's [`TraceLog`]. The value of
/// `expr` is returned unchanged, so the macro can wrap a fallible read and the
/// caller can still apply `?`.
///
/// [`BitReader::total_bits_read`]: crate::BitReader::total_bits_read
#[macro_export]
macro_rules! trace_field {
    ($reader:expr, $name:expr, $body:expr) => {{
        let start_bit = $reader.total_bits_read();
        let value = $body;
        let end_bit = $reader.total_bits_read();
        $reader.trace_mut().record($name, start_bit, end_bit);
        value
    }};
}
