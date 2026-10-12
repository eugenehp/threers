// Shared by the berry's four materials. Each `ShaderMaterial` is its own WGSL
// module, so "shared" means concatenated in — which is why everything here is
// written to be harmless if a given material never calls it.
//
//   u_s1[0..27]   incident radiance, SH9, model space
//   u_s1[27..36]  three soft-box directions (towards the light)
//   u_s1[36..45]  three soft-box irradiances

fn sat(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }

fn s1v(base: u32) -> vec3<f32> {
    return vec3<f32>(u_s1[base], u_s1[base + 1u], u_s1[base + 2u]);
}

fn sh_coeff(i: u32) -> vec3<f32> { return s1v(i * 3u); }
fn light_dir(i: u32) -> vec3<f32> { return s1v(27u + i * 3u); }
fn light_irr(i: u32) -> vec3<f32> { return s1v(36u + i * 3u); }

/// Irradiance arriving at a surface with normal `n`, from the nine
/// coefficients. Ramamoorthi and Hanrahan's convolution, which is exact for
/// the band-limited field and two multiplies per channel.
fn sh_irradiance(n: vec3<f32>) -> vec3<f32> {
    let c1 = 0.429043;
    let c2 = 0.511664;
    let c3 = 0.743125;
    let c4 = 0.886227;
    let c5 = 0.247708;
    let l00 = sh_coeff(0u);
    let l1m1 = sh_coeff(1u);
    let l10 = sh_coeff(2u);
    let l11 = sh_coeff(3u);
    let l2m2 = sh_coeff(4u);
    let l2m1 = sh_coeff(5u);
    let l20 = sh_coeff(6u);
    let l21 = sh_coeff(7u);
    let l22 = sh_coeff(8u);
    let e = c4 * l00
        + 2.0 * c2 * (l1m1 * n.y + l10 * n.z + l11 * n.x)
        + 2.0 * c1 * (l2m2 * n.x * n.y + l2m1 * n.y * n.z + l21 * n.x * n.z)
        + c3 * l20 * n.z * n.z - c5 * l20
        + c1 * l22 * (n.x * n.x - n.y * n.y);
    return max(e, vec3<f32>(0.0));
}

/// Khronos PBR Neutral, which the renderer also offers as
/// `ToneMapping::Neutral` — reproduced here so the four materials agree with
/// each other whatever the renderer is set to.
///
/// Chosen over ACES because below the shoulder it is nearly the identity: the
/// drupelet albedos are measured values and this is the curve that puts them on
/// screen as themselves. ACES lifts the same midtones from 0.18 to 0.27, which
/// makes a handsome picture of something that is no longer the colour of fruit.
fn pbr_neutral(c_in: vec3<f32>) -> vec3<f32> {
    let start = 0.8 - 0.04;
    let desat = 0.15;
    var c = c_in;
    let x = min(c.r, min(c.g, c.b));
    var offset = 0.04;
    if (x < 0.08) { offset = x - 6.25 * x * x; }
    c = c - vec3<f32>(offset);
    let peak = max(c.r, max(c.g, c.b));
    if (peak < start) { return c; }
    let d = 1.0 - start;
    let new_peak = 1.0 - d * d / (peak + d - start);
    c = c * (new_peak / max(peak, 1e-6));
    let g = 1.0 - 1.0 / (desat * (peak - new_peak) + 1.0);
    return mix(c, vec3<f32>(new_peak), g);
}

/// How many pixels one world unit covers, along `dir`, at `world`.
///
/// Measured by projecting two points rather than read off the projection
/// matrix, because the matrix is not in the frame uniform and this needs no
/// agreement about where its terms live. Used by everything thin enough to fall
/// under a pixel: a hair, a style, anything that has to be widened and faded
/// rather than left to alias.
fn pixels_per_unit(world: vec3<f32>, dir: vec3<f32>) -> f32 {
    let a = frame.view_proj * vec4<f32>(world, 1.0);
    let b = frame.view_proj * vec4<f32>(world + dir, 1.0);
    if (a.w <= 1e-5 || b.w <= 1e-5) { return 1.0; }
    let d = (b.xy / b.w - a.xy / a.w) * frame.viewport_size.xy * 0.5;
    return max(length(d), 1e-4);
}
