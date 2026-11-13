//! The packed 32-bit formats: `Rgb10a2Unorm`, `Rg11b10Ufloat` and
//! `Rgb9e5Ufloat`, each a little-endian u32 per texel with R in the low bits.
//!
//! The float ones follow the conversions the graphics APIs define (Vulkan's
//! "Unsigned 11-Bit/10-Bit Floating-Point Numbers" and "Shared Exponent"):
//! round to nearest, negatives and NaN to 0, and values past the largest
//! finite one saturate to it rather than becoming infinity, so a bright sun in
//! an HDR map stays the brightest thing in it instead of turning into a hole.

/// The u32 texels of `data`, little-endian.
pub fn words(data: &[u8]) -> impl Iterator<Item = u32> + '_ {
    data.as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
}

/// R, G, B (10 bits each) and A (2 bits), as `[0, 1]` floats.
pub fn unpack_rgb10a2(w: u32) -> [f32; 4] {
    [
        (w & 0x3ff) as f32 / 1023.0,
        ((w >> 10) & 0x3ff) as f32 / 1023.0,
        ((w >> 20) & 0x3ff) as f32 / 1023.0,
        (w >> 30) as f32 / 3.0,
    ]
}

/// `[0, 1]` RGBA to `Rgb10a2Unorm`, each channel rounded to nearest.
pub fn pack_rgb10a2(c: [f32; 4]) -> u32 {
    let q = |v: f32, max: f32| (v.clamp(0.0, 1.0) * max + 0.5) as u32;
    q(c[0], 1023.0) | (q(c[1], 1023.0) << 10) | (q(c[2], 1023.0) << 20) | (q(c[3], 3.0) << 30)
}

/// A small unsigned float with `mbits` of mantissa and a 5-bit exponent
/// (bias 15), as the 11- and 10-bit channels of `Rg11b10Ufloat` are.
fn small_float_to_f32(bits: u32, mbits: u32) -> f32 {
    let e = bits >> mbits;
    let m = bits & ((1 << mbits) - 1);
    let scale = (1u32 << mbits) as f32;
    match e {
        0 => (m as f32 / scale) * 2f32.powi(-14),
        31 => {
            if m == 0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => (1.0 + m as f32 / scale) * 2f32.powi(e as i32 - 15),
    }
}

/// `v` as an unsigned float with `mbits` of mantissa, rounded to nearest with
/// ties to even, and clamped to the largest finite value.
fn f32_to_small_float(v: f32, mbits: u32) -> u32 {
    if v.is_nan() || v <= 0.0 {
        return 0;
    }
    let max_bits = (30 << mbits) | ((1 << mbits) - 1);
    let max = small_float_to_f32(max_bits, mbits);
    if v >= max {
        return max_bits;
    }
    let b = v.to_bits();
    let exp = ((b >> 23) & 0xff) as i32 - 127;
    if exp < -14 {
        // Denormal: a multiple of 2^(-14 - mbits). Scaling by a power of two
        // is exact, so this rounds once, to nearest even; a carry into the
        // exponent field is the right answer too.
        return (v as f64 * 2f64.powi(14 + mbits as i32)).round_ties_even() as u32;
    }
    // Normal: drop 23 - mbits mantissa bits with round to nearest even; a
    // carry moves into the exponent, which is still correct rounding.
    let drop = 23 - mbits;
    let mant = b & 0x7f_ffff;
    let biased = ((exp + 15) as u32) << mbits | (mant >> drop);
    let rem = mant & ((1 << drop) - 1);
    let half = 1 << (drop - 1);
    let up = rem > half || (rem == half && biased & 1 == 1);
    (biased + up as u32).min(max_bits)
}

/// `Rg11b10Ufloat` texel as RGBA, alpha 1.
pub fn unpack_rg11b10(w: u32) -> [f32; 4] {
    [
        small_float_to_f32(w & 0x7ff, 6),
        small_float_to_f32((w >> 11) & 0x7ff, 6),
        small_float_to_f32(w >> 22, 5),
        1.0,
    ]
}

/// RGB to `Rg11b10Ufloat`.
pub fn pack_rg11b10(c: [f32; 3]) -> u32 {
    f32_to_small_float(c[0], 6)
        | (f32_to_small_float(c[1], 6) << 11)
        | (f32_to_small_float(c[2], 5) << 22)
}

/// `Rgb9e5Ufloat` texel as RGBA, alpha 1: `mantissa * 2^(exponent - 15 - 9)`.
pub fn unpack_rgb9e5(w: u32) -> [f32; 4] {
    let scale = 2f32.powi((w >> 27) as i32 - 24);
    [
        (w & 0x1ff) as f32 * scale,
        ((w >> 9) & 0x1ff) as f32 * scale,
        ((w >> 18) & 0x1ff) as f32 * scale,
        1.0,
    ]
}

/// RGB to `Rgb9e5Ufloat`, by the shared-exponent conversion the APIs give:
/// the exponent from the largest channel, raised by one where rounding that
/// channel would overflow its 9 bits.
pub fn pack_rgb9e5(c: [f32; 3]) -> u32 {
    const MAX: f32 = 511.0 / 512.0 * 65536.0; // (2^9 - 1) / 2^9 * 2^(31 - 15)
    let clamp = |v: f32| if v.is_nan() { 0.0 } else { v.clamp(0.0, MAX) };
    let [r, g, b] = c.map(clamp);
    let maxc = r.max(g).max(b);
    // floor(log2(maxc)), read off the bits so it is exact.
    let floor_log2 = if maxc > 0.0 {
        ((maxc.to_bits() >> 23) & 0xff) as i32 - 127
    } else {
        -16
    };
    let mut exp = floor_log2.max(-16) + 1 + 15;
    let mut denom = 2f64.powi(exp - 15 - 9);
    let maxm = (maxc as f64 / denom + 0.5).floor();
    if maxm >= 512.0 {
        exp += 1;
        denom *= 2.0;
    }
    let m = |v: f32| (v as f64 / denom + 0.5).floor() as u32;
    m(r) | (m(g) << 9) | (m(b) << 18) | ((exp as u32) << 27)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb10a2_round_trips_every_level() {
        for v in 0..1024u32 {
            let w = v | (v << 10) | ((1023 - v) << 20) | ((v & 3) << 30);
            assert_eq!(pack_rgb10a2(unpack_rgb10a2(w)), w);
        }
    }

    #[test]
    fn rg11b10_round_trips_every_finite_value() {
        for bits in 0..(31 << 6) {
            let w = bits | (bits << 11);
            let c = unpack_rg11b10(w);
            assert_eq!(pack_rg11b10([c[0], c[1], c[2]]) & 0x3f_ffff, w, "{bits:#x}");
        }
        for bits in 0..(31 << 5) {
            let c = unpack_rg11b10(bits << 22);
            assert_eq!(pack_rg11b10([c[0], c[1], c[2]]), bits << 22, "{bits:#x}");
        }
    }

    #[test]
    fn rg11b10_rounds_to_nearest_and_saturates() {
        // 1.0 and the next 11-bit value up, 1 + 1/64: the midpoint rounds to even (1.0).
        assert_eq!(pack_rg11b10([1.0 + 1.0 / 128.0, 0.0, 0.0]), 15 << 6);
        assert_eq!(pack_rg11b10([1.0 + 1.5 / 64.0, 0.0, 0.0]), (15 << 6) | 2);
        assert_eq!(pack_rg11b10([1e9, -1.0, f32::NAN]), (30 << 6) | 63);
        assert_eq!(unpack_rg11b10((30 << 6) | 63)[0], 65024.0);
    }

    #[test]
    fn rgb9e5_round_trips_and_matches_known_values() {
        assert_eq!(pack_rgb9e5([0.0, 0.0, 0.0]), 0);
        // 1.0: exponent 16 (2^1 with bias 15), mantissa 256.
        assert_eq!(
            pack_rgb9e5([1.0, 1.0, 1.0]),
            256 | (256 << 9) | (256 << 18) | (16 << 27)
        );
        for e in 0..32u32 {
            for m in [1u32, 100, 256, 511] {
                let w = m | ((m / 2) << 9) | ((m / 3) << 18) | (e << 27);
                let c = unpack_rgb9e5(w);
                let back = unpack_rgb9e5(pack_rgb9e5([c[0], c[1], c[2]]));
                for i in 0..3 {
                    assert!(
                        (back[i] - c[i]).abs() <= c[0] / 256.0,
                        "e {e} m {m}: {c:?} -> {back:?}"
                    );
                }
            }
        }
        // Past the largest value saturates to it.
        assert_eq!(pack_rgb9e5([1e10, 0.0, 0.0]), 511 | (31 << 27));
    }
}
