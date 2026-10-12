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
    let c = byte as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear light in `[0, 1]` as an sRGB-encoded byte.
pub fn linear_to_srgb(value: f32) -> u8 {
    let c = value.clamp(0.0, 1.0);
    let encoded = if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (encoded * 255.0).round().clamp(0.0, 255.0) as u8
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

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!((mid - 0.2158).abs() < 1e-3, "sRGB 128 is {mid} of full light");
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
