//! The sRGB transfer function, in WGSL, in one place.
//!
//! Rust callers have had [`crate::utils::srgb`] for this all along. Shaders
//! have not: the renderer's own `linear_to_srgb` lives a long way inside the
//! preamble `ShaderMaterial` is compiled against, which is no use to a pass
//! that is not part of the renderer — so the bloom raster path, and then the
//! fluid, each wrote the curve out again. Three copies of a transfer function
//! is two more than can be checked against each other.
//!
//! This is the one copy. [`WGSL`] is a prelude string to prepend to a shader;
//! the renderer splices the same string into its own preamble, so a change here
//! reaches every surface in the crate at once.
//!
//! ```
//! # use threers::renderer::colorspace;
//! let source = format!("{}\n{}", colorspace::WGSL, "/* your fragment shader */");
//! assert!(source.contains("fn linear_to_srgb"));
//! ```
//!
//! # Why the exact curve
//!
//! It is the piecewise IEC 61966-2-1 function, not a 2.2 power. The two differ
//! most near black — at a stored 0.02 the approximation is off by about a
//! quarter of the value — which is exactly where a bloom threshold or an
//! alpha composite decides between visible and not.

/// `linear_to_srgb`, `srgb_to_linear` and their scalar forms, as WGSL.
///
/// Both take and return `vec3<f32>`; `*_1` are the single-channel versions.
/// Neither clamps: a caller working in half float may legitimately be carrying
/// values above one, and clamping here would quietly cap an HDR pass. Clamp at
/// the point of storage instead, where the range is actually known.
pub const WGSL: &str = r#"
// IEC 61966-2-1 sRGB OETF — matches three.js `sRGBTransferOETF`.
fn linear_to_srgb_1(c : f32) -> f32 {
    if (c <= 0.0031308) {
        return c * 12.92;
    }
    return 1.055 * pow(max(c, 0.0), 1.0 / 2.4) - 0.055;
}

fn linear_to_srgb(c : vec3<f32>) -> vec3<f32> {
    let cutoff = vec3<f32>(0.0031308);
    let lo = c * 12.92;
    let hi = pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) * 1.055 - vec3<f32>(0.055);
    return mix(hi, lo, step(c, cutoff));
}

// The inverse: what a sampler does for you on an `*Srgb` format, and what you
// have to do yourself when the bytes are sRGB-encoded in a linear format.
fn srgb_to_linear_1(c : f32) -> f32 {
    if (c <= 0.04045) {
        return c / 12.92;
    }
    return pow((c + 0.055) / 1.055, 2.4);
}

fn srgb_to_linear(c : vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((max(c, vec3<f32>(0.0)) + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}
"#;

/// The names [`WGSL`] defines, so a shader that already declares one of them
/// can be checked rather than failing to compile with a duplicate-symbol error
/// a thousand lines from where the clash is.
pub const WGSL_FUNCTIONS: &[&str] = &[
    "linear_to_srgb_1",
    "linear_to_srgb",
    "srgb_to_linear_1",
    "srgb_to_linear",
];

/// Prepend [`WGSL`] to `source`, unless it already defines the functions.
///
/// Splicing a prelude into a shader that happens to define `linear_to_srgb`
/// itself is a duplicate-definition error pointing at the prelude, which is
/// the one part of the file the author did not write. This checks first.
pub fn with_prelude(source: &str) -> String {
    if WGSL_FUNCTIONS
        .iter()
        .any(|f| source.contains(&format!("fn {f}")))
    {
        return source.to_string();
    }
    format!("{WGSL}\n{source}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::srgb;

    /// The prelude is spliced into shader sources by string concatenation, so
    /// the one way it can break everything at once is by not parsing. Compiling
    /// it needs no GPU — naga is a dependency already.
    // naga's WGSL front end is native-only in this crate's wgpu build.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn the_prelude_is_valid_wgsl() {
        let source = format!("{WGSL}\n@fragment fn fs() -> @location(0) vec4<f32> {{ return vec4<f32>(srgb_to_linear(vec3<f32>(0.5)), 1.0); }}");
        let module =
            wgpu::naga::front::wgsl::parse_str(&source).expect("the colour prelude parses");
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("the colour prelude validates");
    }

    #[test]
    fn the_prelude_declares_what_it_says_it_does() {
        for f in WGSL_FUNCTIONS {
            assert!(WGSL.contains(&format!("fn {f}")), "{f} is missing");
        }
        // And `with_prelude` notices a clash rather than producing one.
        let mine = "fn linear_to_srgb(c : vec3<f32>) -> vec3<f32> { return c; }";
        assert_eq!(with_prelude(mine), mine);
        assert!(with_prelude("// nothing").contains("fn linear_to_srgb"));
    }

    /// The WGSL and the Rust are two implementations of one function, and
    /// nothing but a test holds them to the same numbers. This evaluates the
    /// WGSL's arithmetic in Rust — the same branch points and constants — and
    /// compares; if someone edits one curve the two drift and this says so.
    #[test]
    fn the_rust_and_the_shader_agree() {
        // The WGSL branch, transcribed. If this transcription stops matching
        // the string above, `the_prelude_is_valid_wgsl` still passes and this
        // is the test that has to be re-read — so keep them adjacent.
        let wgsl_encode = |c: f32| {
            if c <= 0.0031308 {
                c * 12.92
            } else {
                1.055 * c.max(0.0).powf(1.0 / 2.4) - 0.055
            }
        };
        let wgsl_decode = |c: f32| {
            if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        };
        for i in 0..=255u16 {
            let v = i as f32 / 255.0;
            assert!(
                (wgsl_encode(v) - srgb::linear_to_srgb_f32(v)).abs() < 1e-6,
                "encode disagrees at {v}"
            );
            assert!(
                (wgsl_decode(v) - srgb::srgb_to_linear(i as u8)).abs() < 1e-6,
                "decode disagrees at {v}"
            );
            // And they invert each other, which neither test above implies.
            assert!(
                (wgsl_decode(wgsl_encode(v)) - v).abs() < 1e-5,
                "round trip at {v}"
            );
        }
    }
}
