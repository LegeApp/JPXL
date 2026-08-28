//! Scan header (`SOS`, 10918-1 B.2.3).

use crate::units::ComponentId;

/// One component's table selectors within a scan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScanComponent {
    /// Component selector `Csj` — matches a [`FrameComponent`] identifier.
    ///
    /// [`FrameComponent`]: crate::frame::FrameComponent
    pub id: ComponentId,
    /// DC entropy-table destination selector `Tdj`, `0..=3`.
    pub dc_table: u8,
    /// AC entropy-table destination selector `Taj`, `0..=3`.
    pub ac_table: u8,
}

/// A parsed start-of-scan header.
#[derive(Clone, Debug)]
pub struct ScanHeader {
    /// Components in this scan, in scan order (interleave order).
    pub components: Vec<ScanComponent>,
    /// Start of spectral selection `Ss`.
    pub spectral_start: u8,
    /// End of spectral selection `Se`.
    pub spectral_end: u8,
    /// Successive approximation high bit position `Ah`.
    pub approx_high: u8,
    /// Successive approximation low bit position `Al`.
    pub approx_low: u8,
}

impl ScanHeader {
    /// Whether this scan interleaves more than one component (MCU geometry) or
    /// is a single-component scan (non-interleaved block geometry).
    #[must_use]
    pub fn is_interleaved(&self) -> bool {
        self.components.len() > 1
    }
}
