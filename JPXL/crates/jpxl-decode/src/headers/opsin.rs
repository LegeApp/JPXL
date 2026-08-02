//! The `OpsinInverseMatrix` bundle (18181-1 L.2.1).
//!
//! ```text
//! Table L.1 — OpsinInverseMatrix bundle
//! condition      type    default                   name
//!                Bool()  true                      all_default
//! !all_default   F16()   11.031566901960783        inv_mat00
//! ... nine inv_mat entries, three opsin_bias, three quant_bias,
//!     and quant_bias_numerator, all F16().
//! ```
//!
//! Signalled from `ImageMetadata` when `!default_m && xyb_encoded`. The
//! defaults are the standard XYB inverse transform; a stream overrides them
//! only to describe a non-standard opsin space.
//!
//! Note the precision mismatch that the table implies: the defaults are written
//! to 17 significant digits, but the wire type is `F16()`, which carries about
//! three. A stream that "restates the defaults" therefore cannot reproduce
//! them — which is exactly why `all_default` exists as a separate flag rather
//! than being inferred from the values.

use jpxl_bitstream::{BitReader, read_bool, read_f16_as_f32, trace_field};

use crate::error::Result;

/// Table L.1 default inverse opsin matrix, row-major.
pub const DEFAULT_INVERSE_MATRIX: [f32; 9] = [
    11.031_567,
    -9.866_944,
    -0.164_622_99,
    -3.254_147_4,
    4.418_770_5,
    -0.164_622_99,
    -3.658_851_3,
    2.712_923,
    1.945_928_2,
];

/// Table L.1 default `opsin_bias` values.
pub const DEFAULT_OPSIN_BIAS: [f32; 3] = [-0.003_793_073_3; 3];

/// Table L.1 default `quant_bias` values.
///
/// The printed defaults are the expressions `1 - 0.05465007330715401`,
/// `1 - 0.07005449891748593`, and `1 - 0.049935103337343655` — verified
/// against a clean scan of Table L.1 (2026-08-02); both OCR conversions had
/// collapsed the leading `1 -` into an ambiguous glyph. The stored values are
/// the evaluated results (≈ 0.9453, 0.9299, 0.9501), consistent with L.2.3's
/// `quant *= oim.quant_bias[channel]` slightly shrinking small coefficients.
pub const DEFAULT_QUANT_BIAS: [f32; 3] =
    [1.0 - 0.054_650_073, 1.0 - 0.070_054_5, 1.0 - 0.049_935_103];

/// Table L.1 default `quant_bias_numerator`.
pub const DEFAULT_QUANT_BIAS_NUMERATOR: f32 = 0.145;

/// A decoded `OpsinInverseMatrix` bundle (18181-1 L.2.1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OpsinInverseMatrix {
    /// Inverse opsin matrix, row-major (`inv_mat00` .. `inv_mat22`).
    pub inverse_matrix: [f32; 9],
    /// Per-channel opsin bias.
    pub opsin_bias: [f32; 3],
    /// Per-channel quantization bias.
    pub quant_bias: [f32; 3],
    /// Numerator of the large-coefficient quantization bias term.
    pub quant_bias_numerator: f32,
}

impl Default for OpsinInverseMatrix {
    fn default() -> Self {
        Self {
            inverse_matrix: DEFAULT_INVERSE_MATRIX,
            opsin_bias: DEFAULT_OPSIN_BIAS,
            quant_bias: DEFAULT_QUANT_BIAS,
            quant_bias_numerator: DEFAULT_QUANT_BIAS_NUMERATOR,
        }
    }
}

/// Reads an `OpsinInverseMatrix` bundle (18181-1 L.2.1).
///
/// # Errors
///
/// A bitstream error on truncation or an invalid `F16()`.
pub fn read_opsin_inverse_matrix(reader: &mut BitReader<'_>) -> Result<OpsinInverseMatrix> {
    let all_default = trace_field!(reader, "opsin.all_default", read_bool(reader))?;
    if all_default {
        return Ok(OpsinInverseMatrix::default());
    }

    let mut inverse_matrix = [0.0f32; 9];
    for slot in &mut inverse_matrix {
        *slot = trace_field!(reader, "opsin.inv_mat", read_f16_as_f32(reader))?;
    }
    let mut opsin_bias = [0.0f32; 3];
    for slot in &mut opsin_bias {
        *slot = trace_field!(reader, "opsin.opsin_bias", read_f16_as_f32(reader))?;
    }
    let mut quant_bias = [0.0f32; 3];
    for slot in &mut quant_bias {
        *slot = trace_field!(reader, "opsin.quant_bias", read_f16_as_f32(reader))?;
    }
    let quant_bias_numerator = trace_field!(
        reader,
        "opsin.quant_bias_numerator",
        read_f16_as_f32(reader)
    )?;

    Ok(OpsinInverseMatrix {
        inverse_matrix,
        opsin_bias,
        quant_bias,
        quant_bias_numerator,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testsupport::BitWriter;

    #[test]
    fn all_default_is_one_bit() {
        let mut w = BitWriter::new();
        w.bool(true);
        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let oim = read_opsin_inverse_matrix(&mut r).expect("valid");
        assert_eq!(r.total_bits_read(), 1);
        assert_eq!(oim, OpsinInverseMatrix::default());
    }

    #[test]
    fn explicit_matrix_reads_sixteen_f16_values() {
        let mut w = BitWriter::new();
        w.bool(false);
        // 16 F16 fields: 9 matrix + 3 opsin_bias + 3 quant_bias + numerator.
        for _ in 0..16 {
            w.f16_bits(0x3C00); // 1.0
        }
        let expected_bits = w.bit_len();

        let data = w.finish_padded(1);
        let mut r = BitReader::new(&data);
        let oim = read_opsin_inverse_matrix(&mut r).expect("valid");

        assert_eq!(r.total_bits_read(), expected_bits);
        assert_eq!(r.total_bits_read(), 1 + 16 * 16);
        assert_eq!(oim.inverse_matrix, [1.0f32; 9]);
        assert_eq!(oim.opsin_bias, [1.0f32; 3]);
        assert_eq!(oim.quant_bias, [1.0f32; 3]);
        assert_eq!(oim.quant_bias_numerator, 1.0);
    }

    #[test]
    fn defaults_match_the_l1_table() {
        let oim = OpsinInverseMatrix::default();
        // Table L.1: inv_mat00 = 11.031566901960783.
        assert!((oim.inverse_matrix[0] - 11.031_567).abs() < 1e-4);
        // inv_mat02 and inv_mat12 are the same value in the table.
        assert_eq!(oim.inverse_matrix[2], oim.inverse_matrix[5]);
        // All three opsin_bias entries are equal in the table.
        assert_eq!(oim.opsin_bias[0], oim.opsin_bias[2]);
        assert_eq!(oim.quant_bias_numerator, 0.145);
    }

    #[test]
    fn truncated_matrix_errors() {
        let mut w = BitWriter::new();
        w.bool(false).f16_bits(0x3C00);
        let data = w.finish();
        let mut r = BitReader::new(&data);
        assert!(read_opsin_inverse_matrix(&mut r).is_err());
    }
}
