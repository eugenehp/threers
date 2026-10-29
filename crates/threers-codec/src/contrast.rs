//! WCAG 2.x's colour contrast: relative luminance and contrast ratio, for
//! the UI's accessibility checks and caption legibility alike.

/// WCAG's relative luminance of an sRGB colour (components `0..=1`).
pub fn relative_luminance(c: [f32; 3]) -> f32 {
    // WCAG 2.x's own constant (0.03928), not the sRGB standard's 0.04045:
    // the two give the same result for 8-bit values.
    let lin = |v: f32| if v <= 0.03928 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) };
    0.2126 * lin(c[0]) + 0.7152 * lin(c[1]) + 0.0722 * lin(c[2])
}

/// WCAG's contrast ratio between two sRGB colours, 1 to 21.
pub fn contrast_ratio(a: [f32; 3], b: [f32; 3]) -> f32 {
    let (la, lb) = (relative_luminance(a), relative_luminance(b));
    (la.max(lb) + 0.05) / (la.min(lb) + 0.05)
}
