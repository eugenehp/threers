//! Converting between sRGB bytes and linear light.
//!
//! An 8-bit colour channel in a file or a `Rgba8UnormSrgb` frame is *encoded*:
//! the byte 128 is not half the light of 255, it is about a fifth of it. Every
//! operation that adds or averages light — compositing, blending, resolving a
//! supersampled frame, blurring — has to happen on linear values, and the
//! encoding put back afterwards.
//!
//! Doing it on the encoded bytes instead is a specific, recognisable artefact
//! rather than a vague inaccuracy. Half coverage of white on black should come
//! out at **188**, not 128: blending the encodings gives a pixel less than half
//! as bright as it should be, which is why antialiased text rendered that way
//! looks too thin on a dark background and too heavy on a light one.

/// One sRGB-encoded byte as linear light in `[0, 1]`.
pub fn srgb_to_linear(byte: u8) -> f32 {
    tables().decode[byte as usize]
}

/// Linear light in `[0, 1]` as an sRGB-encoded byte: the byte whose
/// encoding rounds to it, found among the 255 points where the rounding
/// changes (no power function per call, and the same byte as computing the
/// curve and rounding).
///
/// The inner loop of blending text and shapes (three channels a pixel), so
/// not a search of the 255: where `value`'s 1/4096th of the range starts is
/// a byte found once, and the one or two thresholds past it in the bucket
/// stepped over — the same byte, from a load and a compare or two.
pub fn linear_to_srgb(value: f32) -> u8 {
    let c = value.clamp(0.0, 1.0);
    let t = tables();
    // NaN clamps to NaN, bucket 0, and is below every threshold: 0, as before.
    let mut b = t.start[((c * 4096.0) as usize).min(4095)] as usize;
    while b < 255 && c >= t.steps[b] {
        b += 1;
    }
    b as u8
}

/// What the curve takes from each byte, and where encoding moves from one
/// byte to the next.
struct Tables {
    decode: [f32; 256],
    /// `steps[k]`: the least linear value encoding to byte `k + 1` or more.
    steps: [f32; 255],
    /// The byte each 1/4096th of `[0, 1]` starts at.
    start: [u8; 4096],
}

fn tables() -> &'static Tables {
    static TABLES: std::sync::OnceLock<Tables> = std::sync::OnceLock::new();
    TABLES.get_or_init(|| {
        let decode = std::array::from_fn(|i| srgb_to_linear_f32(i as f32 / 255.0));
        let rounded = |c: f32| (linear_to_srgb_f32(c) * 255.0).round().clamp(0.0, 255.0) as u8;
        let steps = std::array::from_fn(|k| {
            // Bisect for where the rounded encoding first passes `k`.
            let (mut lo, mut hi) = (0.0f32, 1.0f32);
            for _ in 0..40 {
                let mid = (lo + hi) / 2.0;
                if rounded(mid) as usize > k {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            hi
        });
        let start = std::array::from_fn(|i| steps.partition_point(|&t| i as f32 / 4096.0 >= t) as u8);
        Tables { decode, steps, start }
    })
}

/// Linear light as a float in `[0, 1]`, for callers that stay in floats.
pub fn linear_to_srgb_f32(value: f32) -> f32 {
    let c = value.clamp(0.0, 1.0);
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// An sRGB-encoded float in `[0, 1]` as linear light — the float form of
/// [`srgb_to_linear`], for colours authored as `hex / 255`.
pub fn srgb_to_linear_f32(value: f32) -> f32 {
    let c = value.clamp(0.0, 1.0);
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bucketed encode is the search it replaced, everywhere: a dense
    /// sweep of `[0, 1]`, each threshold and its neighbours, the ends, NaN.
    #[test]
    fn the_buckets_give_the_search() {
        let search = |c: f32| {
            let c = c.clamp(0.0, 1.0);
            tables().steps.partition_point(|&t| c >= t) as u8
        };
        let mut probes: Vec<f32> = (0..=1_000_000).map(|i| i as f32 / 1_000_000.0).collect();
        for &t in tables().steps.iter() {
            probes.extend([t, f32::from_bits(t.to_bits() - 1), f32::from_bits(t.to_bits() + 1)]);
        }
        probes.extend([-1.0, 0.0, 1.0, 2.0, f32::NAN, f32::INFINITY]);
        for v in probes {
            assert_eq!(linear_to_srgb(v), search(v), "at {v}");
        }
    }

    /// The tables give what the curve gives: every byte decoded, and
    /// encoding a dense sweep (plus each step and its neighbours) rounding
    /// as the curve does.
    #[test]
    fn the_tables_agree_with_the_curve() {
        for byte in 0..=255u8 {
            assert_eq!(
                srgb_to_linear(byte),
                srgb_to_linear_f32(byte as f32 / 255.0)
            );
        }
        let curve = |v: f32| (linear_to_srgb_f32(v) * 255.0).round().clamp(0.0, 255.0) as u8;
        let mut values: Vec<f32> = (0..=1_000_000).map(|i| i as f32 / 1_000_000.0).collect();
        for &t in &tables().steps {
            values.extend([
                t,
                f32::from_bits(t.to_bits() - 1),
                f32::from_bits(t.to_bits() + 1),
            ]);
        }
        values.extend([-1.0, 2.0, f32::INFINITY, f32::NEG_INFINITY]);
        for v in values {
            assert_eq!(linear_to_srgb(v), curve(v), "{v}");
        }
        assert_eq!(linear_to_srgb(f32::NAN), 0);
    }

    /// The two directions undo each other for every byte.
    #[test]
    fn the_round_trip_is_exact_for_every_byte() {
        for byte in 0..=255u8 {
            let back = linear_to_srgb(srgb_to_linear(byte));
            assert_eq!(back, byte, "{byte} came back as {back}");
        }
    }

    /// The endpoints and the mid-grey landmark.
    ///
    /// 128 decoding to about 0.216 rather than 0.5 is the whole reason this
    /// module exists: an encoded byte is not an amount of light.
    #[test]
    fn known_values_are_where_the_standard_puts_them() {
        assert_eq!(srgb_to_linear(0), 0.0);
        assert_eq!(srgb_to_linear(255), 1.0);
        let mid = srgb_to_linear(128);
        assert!(
            (mid - 0.2158).abs() < 1e-3,
            "sRGB 128 is {mid} of full light"
        );
        // And half the light encodes to 188, not to 128.
        assert_eq!(linear_to_srgb(0.5), 188);
    }

    /// The curve is monotonic, which a piecewise definition can quietly break
    /// at the join.
    #[test]
    fn the_curve_never_goes_backwards() {
        let mut last = -1.0f32;
        for byte in 0..=255u8 {
            let v = srgb_to_linear(byte);
            assert!(v > last, "{byte} decoded to {v}, not above {last}");
            last = v;
        }
    }
}
