//! Hand-derived conformance vectors for the bitstream primitives.
//!
//! Every expected value below is derived by hand in a comment showing the bit
//! string. Convention used throughout: `bN` is bit N of a byte counting from
//! the least significant bit, and bits are consumed in the order
//! `b0, b1, … b7` of byte 0, then byte 1, and so on.

use jpxl_bitstream::{
    BitReader, BitstreamError, U32Dist, U32Spec, read_f16_as_f32, read_u32, read_u64,
};

// ---------------------------------------------------------------------------
// Bit order
// ---------------------------------------------------------------------------

#[test]
fn first_bit_read_is_least_significant() {
    // 0b0000_0001: b0 = 1, b1..b7 = 0.
    let mut r = BitReader::new(&[0b0000_0001]);
    assert_eq!(r.read_bits(1), Ok(1));
    assert_eq!(r.read_bits(7), Ok(0));
    assert_eq!(r.total_bits_read(), 8);

    // 0b1000_0000: b0 = 0, and the set bit is the last one read.
    let mut r = BitReader::new(&[0b1000_0000]);
    assert_eq!(r.read_bits(1), Ok(0));
    assert_eq!(r.read_bits(7), Ok(0b100_0000));
}

#[test]
fn reads_cross_byte_boundaries() {
    // byte0 = 0xB6 = 0b1011_0110 -> b0..b7 = 0,1,1,0,1,1,0,1
    // byte1 = 0x0D = 0b0000_1101 -> c0..c7 = 1,0,1,1,0,0,0,0
    //
    // u(4)  = b0..b3 = 0,1,1,0            -> 2 + 4          = 6
    // u(6)  = b4..b7,c0,c1 = 1,1,0,1,1,0  -> 1 + 2 + 8 + 16 = 27
    // u(6)  = c2..c7 = 1,1,0,0,0,0        -> 1 + 2          = 3
    let data = [0xB6u8, 0x0D];
    let mut r = BitReader::new(&data);
    assert_eq!(r.read_bits(4), Ok(6));
    assert_eq!(r.read_bits(6), Ok(27));
    assert_eq!(r.read_bits(6), Ok(3));
    assert_eq!(r.total_bits_read(), 16);
    assert_eq!(r.bits_remaining(), 0);
}

#[test]
fn u32_read_is_a_little_endian_word() {
    // Bits LSB-first over 4 bytes reconstruct the little-endian u32 exactly.
    let data = [0x78u8, 0x56, 0x34, 0x12];
    let mut r = BitReader::new(&data);
    assert_eq!(r.read_bits(32), Ok(0x1234_5678));
}

#[test]
fn zero_width_reads_and_widths_above_32() {
    let data = [0xFFu8; 4];
    let mut r = BitReader::new(&data);
    assert_eq!(r.read_bits(0), Ok(0));
    assert_eq!(r.total_bits_read(), 0);
    assert_eq!(r.read_bits(33), Err(BitstreamError::Overflow));
    assert_eq!(r.total_bits_read(), 0);
}

// ---------------------------------------------------------------------------
// U32()
// ---------------------------------------------------------------------------

/// `U32(1, 2, 4, 8)` — the `upsampling` shape from the frame header.
const POW2: U32Spec = U32Spec::new([
    U32Dist::Val(1),
    U32Dist::Val(2),
    U32Dist::Val(4),
    U32Dist::Val(8),
]);

/// `Enum()` = `U32(0, 1, 2 + u(4), 18 + u(6))`.
const ENUM: U32Spec = U32Spec::new([
    U32Dist::Val(0),
    U32Dist::Val(1),
    U32Dist::BitsOffset { bits: 4, offset: 2 },
    U32Dist::BitsOffset {
        bits: 6,
        offset: 18,
    },
]);

#[test]
fn u32_all_four_val_selectors() {
    // Four 2-bit selectors 0,1,2,3 packed into one byte, LSB-first:
    //   sel 0 -> b0,b1 = 0,0
    //   sel 1 -> b2,b3 = 1,0
    //   sel 2 -> b4,b5 = 0,1
    //   sel 3 -> b6,b7 = 1,1
    // byte = 4 + 32 + 64 + 128 = 228 = 0b1110_0100
    let data = [0b1110_0100u8];
    let mut r = BitReader::new(&data);
    assert_eq!(read_u32(&mut r, &POW2), Ok(1));
    assert_eq!(read_u32(&mut r, &POW2), Ok(2));
    assert_eq!(read_u32(&mut r, &POW2), Ok(4));
    assert_eq!(read_u32(&mut r, &POW2), Ok(8));
    assert_eq!(r.total_bits_read(), 8);
}

#[test]
fn u32_bits_offset_selector_two() {
    // sel 2 -> b0,b1 = 0,1 ; payload u(4) = 5 -> b2..b5 = 1,0,1,0
    // byte = 2 (b1) + 4 (b2) + 16 (b4) = 22 = 0b0001_0110
    // value = 2 + 5 = 7
    let data = [0b0001_0110u8];
    let mut r = BitReader::new(&data);
    assert_eq!(read_u32(&mut r, &ENUM), Ok(7));
    assert_eq!(r.total_bits_read(), 6);
}

#[test]
fn u32_bits_offset_selector_three() {
    // sel 3 -> b0,b1 = 1,1 ; payload u(6) = 63 -> b2..b7 = 1,1,1,1,1,1
    // byte = 0xFF ; value = 18 + 63 = 81
    let data = [0xFFu8];
    let mut r = BitReader::new(&data);
    assert_eq!(read_u32(&mut r, &ENUM), Ok(81));
    assert_eq!(r.total_bits_read(), 8);
}

#[test]
fn u32_offset_plus_payload_overflow() {
    const NEAR_MAX: U32Spec = U32Spec::new([
        U32Dist::Val(0),
        U32Dist::BitsOffset {
            bits: 8,
            offset: u32::MAX - 1,
        },
        U32Dist::Val(0),
        U32Dist::Val(0),
    ]);

    // sel 1 -> b0,b1 = 1,0 ; payload u(8) = 1 -> b2 = 1, b3..b9 = 0
    // byte0 = 1 + 4 = 5, byte1 = 0 ; value = (u32::MAX - 1) + 1 = u32::MAX
    let ok = [0b0000_0101u8, 0x00];
    let mut r = BitReader::new(&ok);
    assert_eq!(read_u32(&mut r, &NEAR_MAX), Ok(u32::MAX));

    // Same, with payload u(8) = 5 -> b2 = 1, b4 = 1
    // byte0 = 1 + 4 + 16 = 21 ; (u32::MAX - 1) + 5 does not fit in a u32.
    let overflow = [0b0001_0101u8, 0x00];
    let mut r = BitReader::new(&overflow);
    assert_eq!(read_u32(&mut r, &NEAR_MAX), Err(BitstreamError::Overflow));
}

#[test]
fn u32_payload_width_above_32_is_rejected() {
    const BAD: U32Spec = U32Spec::new([
        U32Dist::BitsOffset {
            bits: 33,
            offset: 0,
        },
        U32Dist::Val(0),
        U32Dist::Val(0),
        U32Dist::Val(0),
    ]);
    // sel 0 -> b0,b1 = 0,0
    let data = [0x00u8; 8];
    let mut r = BitReader::new(&data);
    assert_eq!(read_u32(&mut r, &BAD), Err(BitstreamError::Overflow));
}

#[test]
fn u32_truncated_payload_is_out_of_bounds() {
    // Start 2 bits into a one-byte buffer, so after the 2-bit selector
    // (b2,b3 = 1,1 -> sel 3) only 4 bits remain but 6 are needed.
    let data = [0xFFu8];
    let mut r = BitReader::new(&data);
    r.skip_bits(2).expect("2 bits available");
    assert_eq!(
        read_u32(&mut r, &ENUM),
        Err(BitstreamError::OutOfBounds {
            bit_pos: 4,
            requested_bits: 6
        })
    );
}

// ---------------------------------------------------------------------------
// U64()
// ---------------------------------------------------------------------------

#[test]
fn u64_selector_zero_is_zero() {
    // b0,b1 = 0,0 -> value 0, no payload.
    let data = [0x00u8];
    let mut r = BitReader::new(&data);
    assert_eq!(read_u64(&mut r), Ok(0));
    assert_eq!(r.total_bits_read(), 2);
}

#[test]
fn u64_selector_one_is_one_plus_u4() {
    // sel 1 -> b0,b1 = 1,0 ; u(4) = 15 -> b2..b5 = 1,1,1,1
    // byte = 1 + 4 + 8 + 16 + 32 = 61 = 0b0011_1101 ; value = 1 + 15 = 16
    let data = [0b0011_1101u8];
    let mut r = BitReader::new(&data);
    assert_eq!(read_u64(&mut r), Ok(16));
    assert_eq!(r.total_bits_read(), 6);
}

#[test]
fn u64_selector_two_is_17_plus_u8() {
    // sel 2 -> b0,b1 = 0,1 ; u(8) = 255 -> b2..b7 and c0,c1 all 1
    // byte0 = 2 + 252 = 254 ; byte1 = 1 + 2 = 3 ; value = 17 + 255 = 272
    let data = [254u8, 3];
    let mut r = BitReader::new(&data);
    assert_eq!(read_u64(&mut r), Ok(272));
    assert_eq!(r.total_bits_read(), 10);
}

#[test]
fn u64_long_form_without_continuation() {
    // sel 3 -> b0,b1 = 1,1 ; v = u(12) = 0xABC ; continuation bit = 0.
    // 0xABC LSB-first: 0,0,1,1,1,1,0,1,0,1,0,1
    // stream: 1,1, 0,0,1,1,1,1, | 0,1,0,1,0,1, 0(cont), 0(pad)
    // byte0 = 1 + 2 + 16 + 32 + 64 + 128 = 243 = 0xF3
    // byte1 = 2 + 8 + 32                 = 42  = 0x2A
    let data = [0xF3u8, 0x2A];
    let mut r = BitReader::new(&data);
    assert_eq!(read_u64(&mut r), Ok(0xABC));
    assert_eq!(r.total_bits_read(), 15);
}

#[test]
fn u64_long_form_with_one_continuation_chunk() {
    // sel 3 -> 1,1 ; v = u(12) = 0 (12 zero bits) ; cont = 1 ; u(8) = 0xFF
    // at shift 12 ; cont = 0.
    // stream bits: 1,1, 0*12, 1, 1*8, 0
    // byte0 = b0,b1 = 1,1 rest 0            -> 3
    // byte1 = bits 8..15 = 0,0,0,0,0,0,1,1  -> 64 + 128 = 192
    // byte2 = bits 16..23 = 1,1,1,1,1,1,1,0 -> 127
    // value = 0xFF << 12 = 0x000F_F000 = 1_044_480
    let data = [3u8, 192, 127];
    let mut r = BitReader::new(&data);
    assert_eq!(read_u64(&mut r), Ok(0xFF << 12));
    assert_eq!(r.total_bits_read(), 24);
}

#[test]
fn u64_maximum_uses_the_shift_60_terminal_arm() {
    // Every bit of the encoding of u64::MAX is 1:
    //   sel 3            -> 1,1                       (2 bits)
    //   v = u(12) = 0xFFF -> twelve 1s                (12 bits)
    //   six times: cont = 1 then u(8) = 0xFF          (6 * 9 = 54 bits)
    //     filling shifts 12, 20, 28, 36, 44, 52
    //   cont = 1 with shift == 60 -> u(4) = 0xF, break (5 bits)
    // total = 2 + 12 + 54 + 5 = 73 bits, all ones.
    let data = [0xFFu8; 10];
    let mut r = BitReader::new(&data);
    assert_eq!(read_u64(&mut r), Ok(u64::MAX));
    assert_eq!(r.total_bits_read(), 73);
}

#[test]
fn u64_truncated_continuation_is_out_of_bounds() {
    // sel 3 then twelve payload bits then a continuation bit set to 1, but the
    // buffer ends: 0xFF,0xFF is only 16 bits, and the first chunk needs 8 more.
    let data = [0xFFu8, 0xFF];
    let mut r = BitReader::new(&data);
    assert_eq!(
        read_u64(&mut r),
        Err(BitstreamError::OutOfBounds {
            bit_pos: 15,
            requested_bits: 8
        })
    );
}

// ---------------------------------------------------------------------------
// F16()
// ---------------------------------------------------------------------------

#[test]
fn f16_valid_values() {
    // The 16 bits are read LSB-first, so the two input bytes are the
    // little-endian binary16 encoding.
    //
    // 0x3C00 = 0 01111 0000000000 -> +1 * 2^(15-15) * 1.0        = 1.0
    // 0xC000 = 1 10000 0000000000 -> -1 * 2^(16-15) * 1.0        = -2.0
    // 0x0000 = 0 00000 0000000000 -> +0.0
    // 0x8000 = 1 00000 0000000000 -> -0.0
    // 0x0001 = 0 00000 0000000001 -> 1 * 2^-24                   (min subnormal)
    // 0x03FF = 0 00000 1111111111 -> 1023 * 2^-24                (max subnormal)
    // 0x0400 = 0 00001 0000000000 -> 2^-14                       (min normal)
    // 0x7BFF = 0 11110 1111111111 -> 2^15 * (1 + 1023/1024)      = 65504.0
    // 0xFBFF = same, negated                                     = -65504.0
    // 0x3555 = 0 01101 0101010101 -> 2^-2 * (1 + 341/1024)       = 1365/4096
    const EXP_MINUS_24: f32 = 1.0 / 16_777_216.0;
    let cases: &[([u8; 2], f32)] = &[
        ([0x00, 0x3C], 1.0),
        ([0x00, 0xC0], -2.0),
        ([0x00, 0x00], 0.0),
        ([0x00, 0x80], -0.0),
        ([0x01, 0x00], EXP_MINUS_24),
        ([0xFF, 0x03], 1023.0 * EXP_MINUS_24),
        ([0x00, 0x04], 1.0 / 16_384.0),
        ([0xFF, 0x7B], 65_504.0),
        ([0xFF, 0xFB], -65_504.0),
        ([0x55, 0x35], 1365.0 / 4096.0),
    ];

    for &(bytes, expected) in cases {
        let mut r = BitReader::new(&bytes);
        let got = read_f16_as_f32(&mut r).expect("valid F16");
        assert_eq!(got, expected, "F16 bytes {bytes:02X?}");
        assert_eq!(r.total_bits_read(), 16);
    }

    // Signed zero must keep its sign bit.
    let mut r = BitReader::new(&[0x00u8, 0x80]);
    assert!(
        read_f16_as_f32(&mut r)
            .expect("valid F16")
            .is_sign_negative()
    );
}

#[test]
fn f16_exponent_31_is_rejected() {
    // 0x7C00 = +Inf, 0xFC00 = -Inf, 0x7E00 = quiet NaN, 0x7C01 = signalling NaN
    for bytes in [
        [0x00u8, 0x7C],
        [0x00, 0xFC],
        [0x00, 0x7E],
        [0x01, 0x7C],
        [0xFF, 0xFF],
    ] {
        let mut r = BitReader::new(&bytes);
        assert_eq!(
            read_f16_as_f32(&mut r),
            Err(BitstreamError::InvalidF16),
            "F16 bytes {bytes:02X?}"
        );
    }
}

#[test]
fn f16_truncated_is_out_of_bounds() {
    let data = [0x00u8];
    let mut r = BitReader::new(&data);
    assert_eq!(
        read_f16_as_f32(&mut r),
        Err(BitstreamError::OutOfBounds {
            bit_pos: 0,
            requested_bits: 16
        })
    );
}

// ---------------------------------------------------------------------------
// ZeroPadToByte()
// ---------------------------------------------------------------------------

#[test]
fn zero_pad_accepts_zero_padding() {
    // 0b0000_0101: u(3) reads b0,b1,b2 = 1,0,1 -> 5; b3..b7 are all zero.
    let data = [0b0000_0101u8];
    let mut r = BitReader::new(&data);
    assert_eq!(r.read_bits(3), Ok(5));
    assert_eq!(r.zero_pad_to_byte(), Ok(()));
    assert_eq!(r.total_bits_read(), 8);
    assert!(r.is_byte_aligned());
}

#[test]
fn zero_pad_rejects_nonzero_padding() {
    // 0b1000_0101: u(3) = 5 again, but b7 = 1 lands inside the padding.
    let data = [0b1000_0101u8];
    let mut r = BitReader::new(&data);
    assert_eq!(r.read_bits(3), Ok(5));
    assert_eq!(r.zero_pad_to_byte(), Err(BitstreamError::Overflow));
}

#[test]
fn zero_pad_is_a_noop_when_already_aligned() {
    let data = [0xFFu8];
    let mut r = BitReader::new(&data);
    assert_eq!(r.zero_pad_to_byte(), Ok(()));
    assert_eq!(r.total_bits_read(), 0);
    assert_eq!(r.read_bits(8), Ok(0xFF));
    // Aligned at end of input: still a no-op, not an error.
    assert_eq!(r.zero_pad_to_byte(), Ok(()));
}

// ---------------------------------------------------------------------------
// Bounds and accounting
// ---------------------------------------------------------------------------

#[test]
fn out_of_bounds_at_exact_stream_end() {
    let data = [0xFFu8, 0xFF];
    let mut r = BitReader::new(&data);
    assert_eq!(r.read_bits(16), Ok(0xFFFF));
    assert_eq!(r.bits_remaining(), 0);
    assert_eq!(
        r.read_bits(1),
        Err(BitstreamError::OutOfBounds {
            bit_pos: 16,
            requested_bits: 1
        })
    );
    // A zero-width read at the very end still succeeds.
    assert_eq!(r.read_bits(0), Ok(0));
}

#[test]
fn peek_near_end_reports_out_of_bounds() {
    let data = [0xFFu8];
    let r = BitReader::new(&data);
    assert_eq!(r.peek_bits(8), Ok(0xFF));
    assert_eq!(
        r.peek_bits(9),
        Err(BitstreamError::OutOfBounds {
            bit_pos: 0,
            requested_bits: 9
        })
    );
}

#[test]
fn total_bits_read_accounting() {
    // 2 (U32 selector) + 6 (payload) = 8, then 2 + 4 = 6 for the U64,
    // then 16 for the F16 -> 30 bits.
    //
    // byte0: sel 3 = 1,1 then u(6) = 63 -> 0xFF                (ENUM -> 81)
    // byte1: sel 1 = 1,0 then u(4) = 0  -> b0 = 1 -> 0x01      (U64 -> 1)
    //        remaining b6,b7 = 0,0 are the first 2 F16 bits.
    // The F16 then spans byte1 bits 6..7 and all of bytes 2 and 3.
    let data = [0xFFu8, 0x01, 0x00, 0x00];
    let mut r = BitReader::new(&data);
    assert_eq!(read_u32(&mut r, &ENUM), Ok(81));
    assert_eq!(r.total_bits_read(), 8);
    assert_eq!(read_u64(&mut r), Ok(1));
    assert_eq!(r.total_bits_read(), 14);
    assert_eq!(read_f16_as_f32(&mut r), Ok(0.0));
    assert_eq!(r.total_bits_read(), 30);
    assert_eq!(r.bits_remaining(), 2);
}
