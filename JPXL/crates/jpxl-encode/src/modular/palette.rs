//! Forward palette transform (18181-1 H.6.4) — encode peer of the decoder inverse.
//!
//! Wave 1: exact colours only (`nb_deltas = 0`). Unique tuples become palette
//! entries; each pixel stores an index in `0..nb_colours`.

use super::Plane;
use crate::error::{EncodeError, Result};

/// Cap on explicit palette size for wave 1 (fits `U32(u(8), …)` first arm).
pub const MAX_PALETTE_COLOURS: u32 = 256;

/// Table H.7 parameters written for `kPalette`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PaletteParams {
    /// First affected channel in the pre-transform list.
    pub begin_c: u32,
    /// Number of channels covered.
    pub num_c: u32,
    /// Explicit palette entries.
    pub nb_colours: u32,
    /// Always 0 in wave 1.
    pub nb_deltas: u32,
    /// Unused when `nb_deltas == 0`; written as 0.
    pub d_pred: u32,
}

/// Result of a forward exact-colour palette.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PaletteForward {
    /// Wire parameters.
    pub params: PaletteParams,
    /// Meta-channel samples: width `nb_colours`, height `num_c`,
    /// layout `data[c * nb_colours + colour]`.
    pub meta: Plane,
    /// Index channel: one sample per pixel of the original geometry.
    pub index: Plane,
    /// Original frame width.
    pub width: u32,
    /// Original frame height.
    pub height: u32,
}

/// Builds an exact-colour palette over `planes[begin..begin+num_c]`.
///
/// Returns [`None`] when the unique colour count is 0 or exceeds
/// [`MAX_PALETTE_COLOURS`] (caller keeps the non-palette path).
///
/// # Errors
///
/// [`EncodeError::SampleCountMismatch`] if planes disagree in length;
/// [`EncodeError::Unsupported`] if `num_c` is zero or planes are missing.
pub fn try_exact_palette(
    width: u32,
    height: u32,
    planes: &[Plane],
    begin_c: usize,
    num_c: usize,
) -> Result<Option<PaletteForward>> {
    if num_c == 0 {
        return Err(EncodeError::unsupported("palette with num_c = 0", "H.6.4"));
    }
    let end = begin_c
        .checked_add(num_c)
        .ok_or_else(|| EncodeError::unsupported("palette channel range overflows", "H.6.4"))?;
    if end > planes.len() {
        return Err(EncodeError::unsupported(
            "palette over more channels than exist",
            "H.6.4",
        ));
    }
    let expected = (width as usize).saturating_mul(height as usize);
    for plane in planes.iter().take(end).skip(begin_c) {
        if plane.len() != expected {
            return Err(EncodeError::SampleCountMismatch {
                expected: expected as u64,
                found: plane.len() as u64,
            });
        }
    }

    // Collect unique colours in first-seen order (stable indices).
    let mut colours: Vec<Vec<i32>> = Vec::new();
    let mut index = vec![0i32; expected];

    for pix in 0..expected {
        let mut tuple = Vec::with_capacity(num_c);
        for c in 0..num_c {
            let sample = planes
                .get(begin_c + c)
                .and_then(|p| p.get(pix))
                .copied()
                .unwrap_or(0);
            tuple.push(sample);
        }
        let colour_id = if let Some(pos) = colours.iter().position(|e| e == &tuple) {
            pos
        } else {
            if u32::try_from(colours.len()).unwrap_or(u32::MAX) >= MAX_PALETTE_COLOURS {
                return Ok(None);
            }
            colours.push(tuple);
            colours.len() - 1
        };
        if let Some(slot) = index.get_mut(pix) {
            *slot = i32::try_from(colour_id).unwrap_or(0);
        }
    }

    let nb_colours = u32::try_from(colours.len()).unwrap_or(0);
    if nb_colours == 0 {
        return Ok(None);
    }

    // Meta: width = nb_colours, height = num_c.
    let mut meta = vec![0i32; (nb_colours as usize).saturating_mul(num_c)];
    for (colour_id, tuple) in colours.iter().enumerate() {
        for (c, &sample) in tuple.iter().enumerate() {
            let at = c
                .saturating_mul(nb_colours as usize)
                .saturating_add(colour_id);
            if let Some(slot) = meta.get_mut(at) {
                *slot = sample;
            }
        }
    }

    Ok(Some(PaletteForward {
        params: PaletteParams {
            begin_c: u32::try_from(begin_c).unwrap_or(0),
            num_c: u32::try_from(num_c).unwrap_or(0),
            nb_colours,
            nb_deltas: 0,
            d_pred: 0,
        },
        meta,
        index,
        width,
        height,
    }))
}

#[cfg(test)]
#[allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]
mod tests {
    use super::*;

    #[test]
    fn constant_rgb_is_one_colour() {
        let w = 4u32;
        let h = 3u32;
        let n = (w * h) as usize;
        let planes = vec![vec![10i32; n], vec![20i32; n], vec![30i32; n]];
        let fwd = try_exact_palette(w, h, &planes, 0, 3)
            .expect("ok")
            .expect("palette");
        assert_eq!(fwd.params.nb_colours, 1);
        assert_eq!(fwd.meta, vec![10, 20, 30]);
        assert!(fwd.index.iter().all(|&i| i == 0));
    }

    #[test]
    fn two_colours_map_correctly() {
        let w = 2u32;
        let h = 2u32;
        // Checker: (0,0)/(1,1) = black, (1,0)/(0,1) = white on gray.
        let mut y = vec![0i32; 4];
        y[1] = 255;
        y[2] = 255;
        let planes = vec![y];
        let fwd = try_exact_palette(w, h, &planes, 0, 1)
            .expect("ok")
            .expect("palette");
        assert_eq!(fwd.params.nb_colours, 2);
        assert_eq!(fwd.meta, vec![0, 255]);
        assert_eq!(fwd.index, vec![0, 1, 1, 0]);
    }

    #[test]
    fn too_many_unique_colours_returns_none() {
        let w = 32u32;
        let h = 32u32;
        let n = (w * h) as usize;
        // Every pixel a distinct gray → 1024 > 256.
        let plane: Plane = (0..n).map(|i| i as i32).collect();
        let got = try_exact_palette(w, h, &[plane], 0, 1).expect("ok");
        assert!(got.is_none());
    }
}
