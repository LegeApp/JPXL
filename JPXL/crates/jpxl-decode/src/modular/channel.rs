//! The modular channel list (18181-1 H.1).
//!
//! A modular image is an ordered list of `N` channels. Each channel has a
//! width, a height, and power-of-two subsampling shifts `hshift`/`vshift`
//! whose factor is `1 << shift`; a shift of `-1` means "these dimensions bear
//! no relation to the image dimensions" (H.1), which is what the palette
//! meta-channel uses.
//!
//! The first `nb_meta_channels` channels of the list carry palette data rather
//! than picture data. That counter starts at zero and is moved by the
//! transforms (H.6, Table H.8).
//!
//! # Why the shifts are `i32` and not a newtype pair
//!
//! `-1` is a real, load-bearing value here, so the type has to be signed, and
//! the two shifts are never interchangeable with a pixel count. They are kept
//! together with the dimensions in [`ChannelSpec`] precisely so no call site
//! passes a bare integer.

use jpxl_core::limits::AllocGuard;

use super::error::{Result, malformed};

/// Bytes charged to the [`AllocGuard`] per stored sample.
///
/// H.1: signed 32-bit integers suffice for decoded modular samples and for the
/// results of the inverse transforms, so a sample is four bytes.
pub const BYTES_PER_SAMPLE: u64 = 4;

/// The shift value meaning "dimensions unrelated to the image" (18181-1 H.1).
pub const SHIFT_UNRELATED: i32 = -1;

/// Dimensions and subsampling of one channel, without its samples.
///
/// This is the typed hand-off from slice 6/7: the frame layer computes the
/// initial channel list (H.1) and passes it in; Annex H then derives every
/// later shape from the transform chain on its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelSpec {
    /// Channel width in samples.
    pub width: u32,
    /// Channel height in samples.
    pub height: u32,
    /// Horizontal subsampling shift, or [`SHIFT_UNRELATED`].
    pub hshift: i32,
    /// Vertical subsampling shift, or [`SHIFT_UNRELATED`].
    pub vshift: i32,
}

impl ChannelSpec {
    /// A channel of `width` x `height` at full resolution (both shifts zero).
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            hshift: 0,
            vshift: 0,
        }
    }

    /// A channel of `width` x `height` with explicit shifts.
    #[must_use]
    pub const fn with_shifts(width: u32, height: u32, hshift: i32, vshift: i32) -> Self {
        Self {
            width,
            height,
            hshift,
            vshift,
        }
    }

    /// Whether H.2 skips this channel when decoding sample data.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// Sample count as a `u64`, which cannot overflow for `u32` dimensions.
    #[must_use]
    pub const fn sample_count(&self) -> u64 {
        self.width as u64 * self.height as u64
    }
}

/// One channel of a modular image: a [`ChannelSpec`] plus its samples.
///
/// Samples are stored in raster order, `width` per row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Channel {
    spec: ChannelSpec,
    data: Vec<i32>,
}

impl Channel {
    /// Allocates a zero-filled channel, charging `guard` first.
    ///
    /// # Errors
    ///
    /// [`ModularError::Core`](super::ModularError::Core) if the allocation
    /// would exceed the guard's budget, or
    /// [`ModularError::Malformed`](super::ModularError::Malformed) if the
    /// sample count does not fit in a `usize`.
    pub fn new(spec: ChannelSpec, guard: &mut AllocGuard) -> Result<Self> {
        let samples = spec.sample_count();
        guard.charge(samples.saturating_mul(BYTES_PER_SAMPLE))?;
        let len = usize::try_from(samples)
            .map_err(|_| malformed!("H.1: channel of {samples} samples does not fit in memory"))?;
        Ok(Self {
            spec,
            data: vec![0; len],
        })
    }

    /// Wraps an existing sample buffer. Used by tests and by the transforms.
    ///
    /// # Errors
    ///
    /// [`ModularError::Malformed`](super::ModularError::Malformed) if `data`
    /// is not exactly `width * height` long.
    pub fn from_samples(spec: ChannelSpec, data: Vec<i32>) -> Result<Self> {
        if data.len() as u64 != spec.sample_count() {
            return Err(malformed!(
                "H.1: {} samples supplied for a {}x{} channel",
                data.len(),
                spec.width,
                spec.height
            ));
        }
        Ok(Self { spec, data })
    }

    /// This channel's dimensions and shifts.
    #[must_use]
    pub const fn spec(&self) -> ChannelSpec {
        self.spec
    }

    /// Channel width in samples.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.spec.width
    }

    /// Channel height in samples.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.spec.height
    }

    /// Horizontal subsampling shift, or [`SHIFT_UNRELATED`].
    #[must_use]
    pub const fn hshift(&self) -> i32 {
        self.spec.hshift
    }

    /// Vertical subsampling shift, or [`SHIFT_UNRELATED`].
    #[must_use]
    pub const fn vshift(&self) -> i32 {
        self.spec.vshift
    }

    /// All samples in raster order.
    #[must_use]
    pub fn samples(&self) -> &[i32] {
        &self.data
    }

    /// `channel(x, y)` for in-bounds coordinates, `0` otherwise.
    ///
    /// Out-of-bounds reads never happen on the decoding path — H.3 guards every
    /// neighbour with an explicit edge case — but returning zero rather than
    /// panicking keeps the property computation total for attacker-chosen
    /// dimensions.
    #[must_use]
    pub fn get(&self, x: u32, y: u32) -> i32 {
        if x >= self.spec.width || y >= self.spec.height {
            return 0;
        }
        let idx = y as usize * self.spec.width as usize + x as usize;
        self.data.get(idx).copied().unwrap_or(0)
    }

    /// Writes `channel(x, y)`, ignoring out-of-bounds coordinates.
    pub fn set(&mut self, x: u32, y: u32, value: i32) {
        if x >= self.spec.width || y >= self.spec.height {
            return;
        }
        let idx = y as usize * self.spec.width as usize + x as usize;
        if let Some(slot) = self.data.get_mut(idx) {
            *slot = value;
        }
    }

    /// One row of samples, or an empty slice if `y` is out of range.
    #[must_use]
    pub fn row(&self, y: u32) -> &[i32] {
        if y >= self.spec.height {
            return &[];
        }
        let start = y as usize * self.spec.width as usize;
        let end = start + self.spec.width as usize;
        self.data.get(start..end).unwrap_or(&[])
    }

    /// Replaces the dimensions without touching the samples.
    ///
    /// Only the inverse transforms call this, and only together with a
    /// matching rewrite of `data`; the length invariant is re-checked.
    pub(super) fn set_spec(&mut self, spec: ChannelSpec) -> Result<()> {
        if self.data.len() as u64 != spec.sample_count() {
            return Err(malformed!(
                "H.6: re-shaping a channel to {}x{} does not match its {} samples",
                spec.width,
                spec.height,
                self.data.len()
            ));
        }
        self.spec = spec;
        Ok(())
    }

    /// Replaces both dimensions and samples, as the squeeze inverse does.
    pub(super) fn replace(&mut self, spec: ChannelSpec, data: Vec<i32>) -> Result<()> {
        *self = Self::from_samples(spec, data)?;
        Ok(())
    }
}

/// A modular image: the channel list plus the `nb_meta_channels` counter.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ModularImage {
    channels: Vec<Channel>,
    nb_meta_channels: usize,
}

impl ModularImage {
    /// Builds an image from a channel list and a meta-channel count.
    #[must_use]
    pub const fn new(channels: Vec<Channel>, nb_meta_channels: usize) -> Self {
        Self {
            channels,
            nb_meta_channels,
        }
    }

    /// The channel list, in index order.
    #[must_use]
    pub fn channels(&self) -> &[Channel] {
        &self.channels
    }

    /// Consumes the image and yields its channel list.
    #[must_use]
    pub fn into_channels(self) -> Vec<Channel> {
        self.channels
    }

    /// How many leading channels are palette meta-channels (H.1).
    #[must_use]
    pub const fn nb_meta_channels(&self) -> usize {
        self.nb_meta_channels
    }

    pub(super) fn channels_mut(&mut self) -> &mut Vec<Channel> {
        &mut self.channels
    }

    pub(super) const fn set_nb_meta_channels(&mut self, n: usize) {
        self.nb_meta_channels = n;
    }
}

#[cfg(test)]
mod tests {
    use jpxl_core::limits::Limits;

    use super::*;

    #[test]
    fn allocation_is_charged_before_it_happens() {
        // 4 bytes per sample: a 10x10 channel must cost exactly 400 bytes.
        let limits = Limits {
            max_alloc_bytes: 400,
            ..Limits::default()
        };
        let mut guard = AllocGuard::new(&limits);
        Channel::new(ChannelSpec::new(10, 10), &mut guard).expect("400 bytes fits exactly");
        assert_eq!(guard.charged(), 400);

        let err = Channel::new(ChannelSpec::new(1, 1), &mut guard)
            .expect_err("one more sample must not fit");
        assert!(matches!(err, super::super::ModularError::Core(_)));
    }

    #[test]
    fn get_is_total_outside_the_channel() {
        let c = Channel::from_samples(ChannelSpec::new(2, 2), vec![1, 2, 3, 4]).expect("2x2");
        assert_eq!(c.get(1, 1), 4);
        assert_eq!(c.get(2, 0), 0, "x past the right edge reads zero");
        assert_eq!(c.get(0, 9), 0, "y past the bottom reads zero");
    }

    #[test]
    fn from_samples_rejects_a_length_mismatch() {
        let err = Channel::from_samples(ChannelSpec::new(2, 2), vec![1, 2, 3])
            .expect_err("3 samples is not 2x2");
        assert!(err.to_string().contains("H.1"));
    }

    #[test]
    fn empty_channels_are_the_ones_h2_skips() {
        assert!(ChannelSpec::new(0, 5).is_empty());
        assert!(ChannelSpec::new(5, 0).is_empty());
        assert!(!ChannelSpec::new(1, 1).is_empty());
    }
}
