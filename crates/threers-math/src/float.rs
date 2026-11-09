//! Conversions between the float widths the GPU and the CPU disagree on.
//!
//! The GPU holds f32 and, in textures and vertex buffers, f16; data arrives
//! as f64 (coordinates, measurements). Each narrowing here rounds once, to
//! nearest with ties to even, the way IEEE 754 and every GPU do it:
//!
//! - [`f16_to_f32`] / [`f32_to_f16`]: binary16 bits and f32, exact one way and
//!   correctly rounded the other, subnormals, infinities and NaNs included.
//! - [`f64_to_f16`]: straight from f64. Going through f32 rounds twice, and a
//!   value just past an f16 midpoint can land on it at f32 and then go the
//!   wrong way.
//! - [`split_f64`] / [`join_f32`]: an f64 as two f32s, `high + low`, the pair a
//!   shader adds back for about 48 bits of mantissa (deck.gl's
//!   `positions64Low`, luma.gl's fp64).
//! - Slice helpers for packing f16 texels and narrowing f64 arrays.

/// One IEEE 754 binary16 as an `f32`, exactly.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let mantissa = (bits & 0x3ff) as u32;
    match exponent {
        // Subnormal (and zero): no implicit leading one, a fixed 2^-24.
        0 => sign * mantissa as f32 * 5.960_464_5e-8,
        0x1f if mantissa == 0 => sign * f32::INFINITY,
        // A NaN keeps its sign and payload, moved up to f32's mantissa.
        0x1f => f32::from_bits((u32::from(bits & 0x8000) << 16) | 0x7f80_0000 | (mantissa << 13)),
        _ => {
            let s = u32::from(bits & 0x8000) << 16;
            f32::from_bits(s | (((exponent + 127 - 15) as u32) << 23) | (mantissa << 13))
        }
    }
}

/// An `f32` as the nearest binary16, ties to even: past the largest finite
/// half (65504, and what rounds to it) infinity, below half the smallest
/// subnormal zero, a NaN still a NaN (quiet, its payload's top bits kept).
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exp == 0xff {
        return sign
            | 0x7c00
            | if mantissa != 0 {
                0x200 | (mantissa >> 13) as u16
            } else {
                0
            };
    }
    // The magnitude as a 24-bit significand `m` times 2^(e - 23), f32
    // subnormals given their fixed exponent.
    let (m, e) = if exp == 0 {
        (mantissa, -126)
    } else {
        (mantissa | 0x80_0000, exp - 127)
    };
    sign | round_to_f16(m as u64, e - 23)
}

/// An `f64` as the nearest binary16, ties to even, rounded once.
pub fn f64_to_f16(value: f64) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 48) & 0x8000) as u16;
    let exp = ((bits >> 52) & 0x7ff) as i32;
    let mantissa = bits & 0x000f_ffff_ffff_ffff;
    if exp == 0x7ff {
        return sign
            | 0x7c00
            | if mantissa != 0 {
                0x200 | (mantissa >> 42) as u16
            } else {
                0
            };
    }
    let (m, e) = if exp == 0 {
        (mantissa, -1022)
    } else {
        (mantissa | 0x0010_0000_0000_0000, exp - 1023)
    };
    sign | round_to_f16(m, e - 52)
}

/// The binary16 magnitude nearest `m · 2^shift` (`m` below 2^53), ties to
/// even. Halves keep 11 significant bits down to 2^-14 and a fixed 2^-24
/// step below it, so the result is `m` scaled to units of the step that
/// applies, rounded; a carry out of the mantissa moves the exponent up (and
/// past 65504, to infinity) by itself.
fn round_to_f16(m: u64, shift: i32) -> u16 {
    if m == 0 {
        return 0;
    }
    // The value's exponent (of its leading bit), and the step at it.
    let top = 63 - m.leading_zeros() as i32 + shift;
    if top > 15 {
        return 0x7c00;
    }
    let step = top.max(-14) - 10;
    // `m · 2^shift / 2^step`, rounded to nearest, ties to even.
    let down = step - shift;
    let q = if down <= 0 {
        m << (-down) as u32
    } else if down > 63 {
        0
    } else {
        let (q, rest) = (m >> down, m & ((1u64 << down) - 1));
        let half = 1u64 << (down - 1);
        q + u64::from(rest > half || (rest == half && q & 1 == 1))
    };
    // `q` counts steps. Below 2^10 it is a subnormal's mantissa; from 2^10
    // its leading bit adds one to the exponent field, which is otherwise
    // `step + 24` (`top + 15` less that one).
    if step == -24 {
        // Subnormal; a carry to 2^10 is the smallest normal, the same bits.
        return q as u16;
    }
    let field = (step + 24) as u64;
    let bits = (field << 10) + q;
    if bits >= 0x7c00 {
        0x7c00
    } else {
        bits as u16
    }
}

/// An `f64` as two `f32`s, `high` its nearest and `low` the nearest to what
/// is left, so that `high as f64 + low as f64` holds about 48 bits of it.
pub fn split_f64(value: f64) -> [f32; 2] {
    let high = value as f32;
    if !high.is_finite() {
        return [high, 0.0];
    }
    [high, (value - high as f64) as f32]
}

/// [`split_f64`]'s pair back as an `f64`.
pub fn join_f32([high, low]: [f32; 2]) -> f64 {
    high as f64 + low as f64
}

/// `values` as little-endian binary16 bytes, two a value — what an
/// `R16Float`/`Rgba16Float` texture or a `Float16` vertex buffer holds.
pub fn pack_f16(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| f32_to_f16(*v).to_le_bytes())
        .collect()
}

/// [`pack_f16`] of f64s, each rounded once.
pub fn pack_f16_from_f64(values: &[f64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| f64_to_f16(*v).to_le_bytes())
        .collect()
}

/// Little-endian binary16 bytes as `f32`s (a trailing odd byte is ignored).
pub fn unpack_f16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
        .collect()
}

/// `values` as little-endian f32 bytes, four a value — an `R32Float` or
/// `Rgba32Float` texture's texels.
pub fn pack_f32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Little-endian f32 bytes as `f32`s (trailing bytes short of four are
/// ignored).
pub fn unpack_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// `values` less `origin` (per component, `item_size` a vertex), narrowed to
/// f32. Subtracting in f64 first is what keeps far-from-zero data precise:
/// a coordinate a million metres out narrowed directly keeps 6 cm steps,
/// relative to a nearby origin it keeps micrometres.
pub fn f64_relative_to_f32(values: &[f64], item_size: usize, origin: &[f64]) -> Vec<f32> {
    assert!(
        item_size > 0 && origin.len() == item_size,
        "origin needs one value per component"
    );
    values
        .iter()
        .enumerate()
        .map(|(i, v)| (v - origin[i % item_size]) as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The nearest binary16 to `v` by search over every finite half, ties
    /// to the even one — the reference the encoders are held to.
    fn nearest(v: f64) -> u16 {
        if v.is_nan() {
            return 0x7e00;
        }
        let sign = if v.is_sign_negative() { 0x8000 } else { 0 };
        let a = v.abs();
        // Past 65504 + half a step (65520) rounds to infinity.
        if a >= 65520.0 {
            return sign | 0x7c00;
        }
        let (mut lo, mut hi) = (0u16, 0x7bffu16);
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            if f16_to_f32(mid) as f64 <= a {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let below = lo;
        if below == 0x7bff || f16_to_f32(below) as f64 == a {
            return sign | below;
        }
        let (d0, d1) = (
            a - f16_to_f32(below) as f64,
            f16_to_f32(below + 1) as f64 - a,
        );
        let pick = if d0 < d1 || d0 == d1 && below & 1 == 0 {
            below
        } else {
            below + 1
        };
        sign | pick
    }

    #[test]
    fn every_half_round_trips() {
        for bits in 0..=u16::MAX {
            let f = f16_to_f32(bits);
            if f.is_nan() {
                assert!(f16_to_f32(f32_to_f16(f)).is_nan(), "{bits:#06x}");
                assert!(f16_to_f32(f64_to_f16(f as f64)).is_nan(), "{bits:#06x}");
                assert_eq!(
                    f32_to_f16(f),
                    bits | 0x200,
                    "a NaN keeps its payload, quiet"
                );
                continue;
            }
            assert_eq!(f32_to_f16(f), bits, "{bits:#06x}");
            assert_eq!(f64_to_f16(f as f64), bits, "{bits:#06x}");
        }
    }

    #[test]
    fn midpoints_and_their_neighbours_round_to_nearest_even() {
        // Every pair of neighbouring halves, of either sign: the midpoint,
        // and an f32 step either side of it.
        for bits in 0..0x7bffu16 {
            for sign in [0u16, 0x8000] {
                let (a, b) = (
                    f16_to_f32(sign | bits) as f64,
                    f16_to_f32(sign | (bits + 1)) as f64,
                );
                let mid = (a + b) / 2.0;
                let m32 = mid as f32;
                assert_eq!(m32 as f64, mid, "midpoints are f32s");
                for v in [
                    m32,
                    f32::from_bits(m32.to_bits() - 1),
                    f32::from_bits(m32.to_bits() + 1),
                ] {
                    assert_eq!(f32_to_f16(v), nearest(v as f64), "{v:e}");
                    assert_eq!(f64_to_f16(v as f64), nearest(v as f64), "{v:e}");
                }
                // Just past the midpoint in f64: rounds to f32's midpoint,
                // so going through f32 would round twice.
                let past = mid + (b - a) * 1e-9;
                assert_eq!(f64_to_f16(past), sign | (bits + 1), "{past:e}");
            }
        }
    }

    #[test]
    fn edges() {
        assert_eq!(f32_to_f16(0.0), 0);
        assert_eq!(f32_to_f16(-0.0), 0x8000);
        assert_eq!(f32_to_f16(65504.0), 0x7bff);
        // 65519.99 rounds down to the largest half, 65520 is the tie to
        // infinity's even side.
        assert_eq!(f32_to_f16(65519.99), 0x7bff);
        assert_eq!(f32_to_f16(65520.0), 0x7c00);
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16(f32::NEG_INFINITY), 0xfc00);
        assert_eq!(f32_to_f16(f32::MAX), 0x7c00);
        // Half the smallest subnormal is a tie, to zero (even); above it, the
        // subnormal.
        assert_eq!(f32_to_f16(2f32.powi(-25)), 0);
        assert_eq!(f32_to_f16(2f32.powi(-25) * 1.000_001), 1);
        assert_eq!(f32_to_f16(2f32.powi(-26)), 0);
        assert_eq!(f32_to_f16(f32::from_bits(1)), 0, "f32 subnormals vanish");
        // The largest subnormal rounding up into the normals.
        assert_eq!(f32_to_f16(f16_to_f32(0x03ff) + 2f32.powi(-25)), 0x0400);
        assert_eq!(f64_to_f16(1e300), 0x7c00);
        assert_eq!(f64_to_f16(1e-300), 0);
        assert_eq!(f64_to_f16(f64::from_bits(1)), 0);
    }

    #[test]
    fn splits_hold_f64() {
        for v in [
            0.0,
            1.0 / 3.0,
            -122.419_416_7,
            37.774_929_5,
            6_378_137.123_456_789,
            1e30,
            -1e-30,
        ] {
            let [high, low] = split_f64(v);
            assert_eq!(high, v as f32);
            let back = join_f32([high, low]);
            assert!((back - v).abs() <= v.abs() * 2f64.powi(-47), "{v}: {back}");
        }
        assert_eq!(split_f64(f64::INFINITY), [f32::INFINITY, 0.0]);
        assert_eq!(split_f64(1e300)[1], 0.0);
    }

    #[test]
    fn slices() {
        let v = [1.0f32, -2.5, 65504.0, 1e-7];
        assert_eq!(
            unpack_f16(&pack_f16(&v)),
            v.map(|x| f16_to_f32(f32_to_f16(x)))
        );
        assert_eq!(pack_f16_from_f64(&[1.0, -2.5]), pack_f16(&[1.0, -2.5]));
        assert_eq!(unpack_f32(&pack_f32(&v)), v);
        let rel = f64_relative_to_f32(
            &[1_000_000.25, 2_000_000.5, 1_000_001.0, 2_000_001.0],
            2,
            &[1_000_000.0, 2_000_000.0],
        );
        assert_eq!(rel, [0.25, 0.5, 1.0, 1.0]);
    }
}
