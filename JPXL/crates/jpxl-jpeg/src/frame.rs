//! Frame header (`SOF`, 10918-1 B.2.2) and the frame geometry derived from it.

use crate::error::{JpegError, Result};
use crate::marker::SofKind;
use crate::units::ComponentId;

/// One component's parameters from the frame header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameComponent {
    /// Component identifier `Ci` (arbitrary label).
    pub id: ComponentId,
    /// Horizontal sampling factor `Hi`, `1..=4`.
    pub h: u8,
    /// Vertical sampling factor `Vi`, `1..=4`.
    pub v: u8,
    /// Quantization-table destination selector `Tqi`.
    pub quant_id: u8,
}

/// A parsed start-of-frame header.
#[derive(Clone, Debug)]
pub struct FrameHeader {
    /// The `SOF` marker code byte (e.g. `0xC0`), preserved for re-emission.
    pub code: u8,
    /// The coding process / entropy class this code selects.
    pub kind: SofKind,
    /// Sample precision `P` in bits (Phase A supports 8 only).
    pub precision: u8,
    /// Number of image lines `Y`.
    pub height: u16,
    /// Number of samples per line `X`.
    pub width: u16,
    /// Per-component parameters, in header order.
    pub components: Vec<FrameComponent>,
}

/// The block/MCU geometry a frame implies (10918-1 A.2).
#[derive(Clone, Debug)]
pub struct FrameGeometry {
    /// Maximum horizontal sampling factor `Hmax`.
    pub hmax: u8,
    /// Maximum vertical sampling factor `Vmax`.
    pub vmax: u8,
    /// Minimum-coded-units per line.
    pub mcus_per_line: usize,
    /// Minimum-coded-unit rows.
    pub mcu_rows: usize,
}

/// Per-component derived block dimensions.
#[derive(Clone, Copy, Debug)]
pub struct ComponentDims {
    /// Blocks per line when the component is scanned interleaved (padded to
    /// complete MCUs): `mcus_per_line * Hi`.
    pub blocks_per_line_interleaved: usize,
    /// Block rows when interleaved: `mcu_rows * Vi`.
    pub block_rows_interleaved: usize,
    /// Blocks per line when scanned non-interleaved (single-component scan):
    /// `ceil(ceil(X * Hi / Hmax) / 8)`.
    pub blocks_per_line_noninterleaved: usize,
    /// Block rows when non-interleaved: `ceil(ceil(Y * Vi / Vmax) / 8)`.
    pub block_rows_noninterleaved: usize,
}

const fn div_ceil_usize(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

impl FrameHeader {
    /// Computes the frame's MCU geometry.
    pub fn geometry(&self) -> Result<FrameGeometry> {
        let hmax = self
            .components
            .iter()
            .map(|c| c.h)
            .max()
            .ok_or_else(|| JpegError::Malformed("B.2.2: frame has no components".into()))?;
        let vmax = self.components.iter().map(|c| c.v).max().unwrap_or(1);
        let mcus_per_line = div_ceil_usize(self.width as usize, 8 * hmax as usize);
        let mcu_rows = div_ceil_usize(self.height as usize, 8 * vmax as usize);
        Ok(FrameGeometry {
            hmax,
            vmax,
            mcus_per_line,
            mcu_rows,
        })
    }

    /// Computes block dimensions for the component at index `idx`.
    pub fn component_dims(&self, idx: usize, geom: &FrameGeometry) -> Result<ComponentDims> {
        let c = self
            .components
            .get(idx)
            .ok_or_else(|| JpegError::Malformed(format!("component index {idx} out of range")))?;
        let x_i = div_ceil_usize(self.width as usize * c.h as usize, geom.hmax as usize);
        let y_i = div_ceil_usize(self.height as usize * c.v as usize, geom.vmax as usize);
        Ok(ComponentDims {
            blocks_per_line_interleaved: geom.mcus_per_line * c.h as usize,
            block_rows_interleaved: geom.mcu_rows * c.v as usize,
            blocks_per_line_noninterleaved: div_ceil_usize(x_i, 8),
            block_rows_noninterleaved: div_ceil_usize(y_i, 8),
        })
    }
}
