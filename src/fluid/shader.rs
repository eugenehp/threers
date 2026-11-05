//! Every stage of the fluid, written once for both backends with
//! [`passes!`](crate::passes): the wgpu module's WGSL and the Metal
//! backend's MSL are both generated from [`FLUID`].
//!
//! The original is nineteen GLSL programs sharing two vertex shaders. Here they
//! are one set of passes over a single binding layout, which is what collapses
//! the plumbing: one bind-group layout, one pipeline layout, one shader
//! compilation, and pipelines that differ only in which fragment entry they
//! name, what format they write and whether they blend.
//!
//! The cost of sharing one layout is that every pass binds three textures even
//! where it reads one; the unused slots get the same view as the used one. That
//! is a pointer each, and it buys away eighteen layouts.
//!
//! # The uniform block
//!
//! One 96-byte block, written per pass. A field means different things to
//! different stages — `value` is the clear multiplier, the bloom intensity and
//! the sunray weight — because padding a struct per stage costs more than the
//! comment that says so. Offsets are fixed by WGSL's layout rules and mirrored
//! by `Params` in `passes.rs`; the two are checked against each other by
//! `params_layout_matches_wgsl`.
//!
//! The frame-rate read-out (`fs_overlay`) is the one pass written per
//! language: its glyphs are shared with the volume's shader, in
//! `super::read_out`.

use crate::materials::shared_passes::PassShader;

/// The uniform block's name in both languages.
pub(crate) const UNIFORMS: &str = "FluidParams";

/// Every pass, once. See the module docs.
pub(crate) static FLUID: std::sync::LazyLock<PassShader> = std::sync::LazyLock::new(|| {
    crate::passes! {
        struct FluidParams {
            // 1 / resolution of the texture being *sampled*, in the pass's own
            // space.
            texel_size: vec2f,
            // Splat centre, in uv.
            point: vec2f,
            // 1 / resolution of the *advected* field, which is not
            // `texel_size` when the dye grid is finer than the solver's. Only
            // the MacCormack limiter reads it.
            dye_texel_size: vec2f,
            dt: f32,
            dissipation: f32,
            curl: f32,
            // Canvas width / height. Keeps a splat round and the
            // checkerboard square.
            aspect_ratio: f32,
            // Splat colour, or the flat background colour.
            color: vec4f,
            // Bloom knee curve: x = threshold - knee, y = 2 knee,
            // z = 1/(4 knee), w = threshold.
            curve: vec4f,
            radius: f32,
            // Clear multiplier / bloom intensity / sunray weight, by pass.
            value: f32,
            flags: u32,
            pad: u32,
        }

        const FLAG_SHADING: u32 = 1u32;
        const FLAG_BLOOM: u32 = 2u32;
        const FLAG_SUNRAYS: u32 = 4u32;
        // Set when the target is an sRGB format. The display values are
        // already display-referred — the GLSL wrote them straight into an
        // 8-bit buffer with no transfer function — so on an sRGB target the
        // inverse goes in here and the hardware's encode puts the same bytes
        // back.
        const FLAG_ENCODE: u32 = 8u32;

        // ------------------------------------------------------- utility

        fn copy(i: Frag) -> vec4f {
            return tex_a(i.uv);
        }

        // Scales a field by `value`. Used on pressure between frames:
        // starting the Jacobi solve from a fraction of last frame's answer
        // converges faster than starting from zero, and the fraction is what
        // the `pressure` knob sets.
        fn clear(i: Frag) -> vec4f {
            return u.value * tex_a(i.uv);
        }

        // On a linear target the display values are what to store, so this
        // is the identity. On an sRGB target the hardware encodes whatever
        // the shader returns, so the inverse goes in first and the same bytes
        // come back out.
        fn undo_srgb(c: vec3f) -> vec3f {
            if ((u.flags & FLAG_ENCODE) == 0u32) {
                return c;
            }
            let m = clamp(c, vec3f(0.0), vec3f(1.0));
            let lo = m / 12.92;
            let hi = pow((m + vec3f(0.055)) / 1.055, vec3f(2.4));
            return select(hi, lo, m <= vec3f(0.04045));
        }

        fn color(i: Frag) -> vec4f {
            return vec4f(undo_srgb(u.color.rgb), u.color.a);
        }

        // What "transparent" looks like when there is no page behind the
        // canvas. (`2 fract(n / 2)` is `n % 2` for a whole `n`, in a form both
        // languages have.)
        fn checkerboard(i: Frag) -> vec4f {
            let cell = floor(i.uv * 25.0 * vec2f(u.aspect_ratio, 1.0));
            let v = 2.0 * fract((cell.x + cell.y) * 0.5) * 0.1 + 0.8;
            return vec4f(undo_srgb(vec3f(v)), 1.0);
        }

        // Gaussian blobs added to whatever is already there, every splat in
        // the batch in one pass: two `vec4`s each, (point, radius, _) and
        // (colour, _), `value` of them. The same pass seeds both fields: into
        // velocity the colour is the pointer's motion, into dye it is a hue.
        // Summing here accumulates in f32 and rounds once, where a pass each
        // rounded to half float in between.
        fn splat(i: Frag) -> vec4f {
            var acc = vec3f(0.0);
            let n = u32(u.value);
            for (var k = 0u32; k < n; k += 1u32) {
                let a = batch(k * 2u32);
                let b = batch(k * 2u32 + 1u32);
                var p = i.uv - a.xy;
                p.x = p.x * u.aspect_ratio;
                let d = dot(p, p);
                // Outside about three and a half sigma the Gaussian is under
                // 6e-6 of its peak, below what a half-float dye can hold.
                if (d < a.z * 12.0) {
                    acc = acc + exp(-d / a.z) * b.rgb;
                }
            }
            let base = tex_a(i.uv).rgb;
            return vec4f(base + acc, 1.0);
        }

        // ------------------------------------------------------- the solver

        // Semi-Lagrangian advection (Stam 1999): look backwards along the
        // velocity for where this texel's contents came from, and fetch it.
        // Unconditionally stable at any dt. `tex_a` is the velocity, `tex_b`
        // the field being carried. Dissipation is a decay per unit time, so
        // the fade does not depend on the frame rate; zero on the forward
        // half of a MacCormack step, which applies it once at the end.
        fn advection(i: Frag) -> vec4f {
            let coord = i.uv - u.dt * tex_a(i.uv).xy * u.texel_size;
            let result = tex_b(coord);
            let decay = 1.0 + u.dissipation * u.dt;
            return result / decay;
        }

        // The second half of a MacCormack advection: advect the forward
        // result back, take off half of how far it failed to return, and
        // clamp to the four source texels the forward fetch drew on (Selle
        // et al. 2008). `tex_a` velocity, `tex_b` the forward result, `tex_c`
        // the field as it was.
        fn advect_correct(i: Frag) -> vec4f {
            let v = tex_a(i.uv).xy * (u.dt * u.texel_size);
            let fwd = tex_b(i.uv);
            // Advecting the forward result backwards is a forward trace of
            // the same step, which is why this adds where the forward pass
            // subtracted.
            let back = tex_b(i.uv + v);
            let src = tex_c(i.uv);
            var corrected = fwd + 0.5 * (src - back);
            let coord = i.uv - v;
            let t = u.dye_texel_size;
            let st = coord / t - 0.5;
            let base = (floor(st) + 0.5) * t;
            let s00 = tex_c(base);
            let s10 = tex_c(base + vec2f(t.x, 0.0));
            let s01 = tex_c(base + vec2f(0.0, t.y));
            let s11 = tex_c(base + t);
            let lo = min(min(s00, s10), min(s01, s11));
            let hi = max(max(s00, s10), max(s01, s11));
            corrected = clamp(corrected, lo, hi);
            let decay = 1.0 + u.dissipation * u.dt;
            return corrected / decay;
        }

        // How much the velocity field is locally creating or destroying
        // fluid: the right-hand side of the pressure solve. A free-slip wall:
        // outside the domain the normal component is mirrored, so the
        // boundary neither leaks nor drags.
        fn divergence(i: Frag) -> vec4f {
            var l = tex_a(i.vl).x;
            var r = tex_a(i.vr).x;
            var t = tex_a(i.vt).y;
            var b = tex_a(i.vb).y;
            let c = tex_a(i.uv).xy;
            if (i.vl.x < 0.0) { l = -c.x; }
            if (i.vr.x > 1.0) { r = -c.x; }
            if (i.vt.y > 1.0) { t = -c.y; }
            if (i.vb.y < 0.0) { b = -c.y; }
            return vec4f(0.5 * (r - l + t - b), 0.0, 0.0, 1.0);
        }

        // Vorticity: the local spin of the field. Written in texture space,
        // where y runs downward — the mirror image of the GL one, and mirror
        // images of a fluid are fluids.
        fn curl(i: Frag) -> vec4f {
            let l = tex_a(i.vl).y;
            let r = tex_a(i.vr).y;
            let t = tex_a(i.vt).x;
            let b = tex_a(i.vb).x;
            return vec4f(0.5 * (r - l - t + b), 0.0, 0.0, 1.0);
        }

        // Vorticity confinement (Steinhoff & Underhill 1994): push along
        // concentrated vorticity, putting back the curl the grid diffused
        // away. `tex_a` velocity, `tex_b` curl.
        fn vorticity(i: Frag) -> vec4f {
            let l = tex_b(i.vl).x;
            let r = tex_b(i.vr).x;
            let t = tex_b(i.vt).x;
            let b = tex_b(i.vb).x;
            let c = tex_b(i.uv).x;
            // N = grad|w| normalised, rotated a quarter turn:
            // force = e * w * (Ny, -Nx).
            var force = 0.5 * vec2f(abs(t) - abs(b), abs(r) - abs(l));
            force = force / (length(force) + 0.0001);
            force = force * (u.curl * c);
            force.y = -force.y;
            var velocity = tex_a(i.uv).xy;
            velocity = velocity + force * u.dt;
            // Load-bearing: one unbounded half-float texel would reach
            // infinity, then NaN, and advection would smear it everywhere.
            velocity = clamp(velocity, vec2f(-1000.0), vec2f(1000.0));
            return vec4f(velocity, 0.0, 1.0);
        }

        // One Jacobi sweep of grad^2 p = div u, relaxed by `value`: 1 for the
        // plain solver; less as a multigrid smoother, whose checkerboard mode
        // plain Jacobi only flips the sign of. `tex_a` pressure, `tex_b`
        // divergence.
        fn pressure(i: Frag) -> vec4f {
            let l = tex_a(i.vl).x;
            let r = tex_a(i.vr).x;
            let t = tex_a(i.vt).x;
            let b = tex_a(i.vb).x;
            let c = tex_a(i.uv).x;
            let divergence = tex_b(i.uv).x;
            let jacobi = (l + r + b + t - divergence) * 0.25;
            return vec4f(mix(c, jacobi, u.value), 0.0, 0.0, 1.0);
        }

        // ------------------------------------------------------- multigrid

        // What the pressure fails to satisfy: r = rhs - L(p), with L the
        // stencil `pressure` inverts. `tex_a` pressure, `tex_b` right-hand
        // side.
        fn residual(i: Frag) -> vec4f {
            let l = tex_a(i.vl).x;
            let r = tex_a(i.vr).x;
            let t = tex_a(i.vt).x;
            let b = tex_a(i.vb).x;
            let c = tex_a(i.uv).x;
            let rhs = tex_b(i.uv).x;
            return vec4f(rhs - (l + r + t + b - 4.0 * c), 0.0, 0.0, 1.0);
        }

        // The residual onto a grid of half the size: full weighting over the
        // four fine texels a coarse one covers, times four for the doubled
        // spacing. `texel_size` is the *fine* grid's.
        fn restriction(i: Frag) -> vec4f {
            let h = u.texel_size * 0.5;
            var sum = tex_a(i.uv + vec2f(-h.x, -h.y)).x;
            sum = sum + tex_a(i.uv + vec2f(h.x, -h.y)).x;
            sum = sum + tex_a(i.uv + vec2f(-h.x, h.y)).x;
            sum = sum + tex_a(i.uv + vec2f(h.x, h.y)).x;
            return vec4f(sum, 0.0, 0.0, 1.0);
        }

        // The coarse correction added back into the fine pressure; the
        // sampler's bilinear filter is the prolongation. `tex_a` coarse
        // correction, `tex_b` fine pressure.
        fn prolong(i: Frag) -> vec4f {
            let coarse = tex_a(i.uv).x;
            let fine = tex_b(i.uv).x;
            return vec4f(fine + coarse, 0.0, 0.0, 1.0);
        }

        // The projection: subtract the pressure gradient. `(r - l, t - b)` is
        // twice the central difference; `value` is the coefficient
        // (`FluidConfig::gradient_scale`). Clamp-to-edge sampling is the
        // Neumann boundary pressure needs. `tex_a` pressure, `tex_b` velocity.
        fn gradient(i: Frag) -> vec4f {
            let l = tex_a(i.vl).x;
            let r = tex_a(i.vr).x;
            let t = tex_a(i.vt).x;
            let b = tex_a(i.vb).x;
            var velocity = tex_b(i.uv).xy;
            velocity = velocity - u.value * vec2f(r - l, t - b);
            return vec4f(velocity, 0.0, 1.0);
        }

        // ------------------------------------------------------- bloom

        // Soft-knee threshold (Unity's curve): the bloom fades in over a range
        // rather than popping in at the threshold. Alpha zero, which the
        // chain keeps.
        fn bloom_prefilter(i: Frag) -> vec4f {
            var c = tex_a(i.uv).rgb;
            let br = max(c.r, max(c.g, c.b));
            var rq = clamp(br - u.curve.x, 0.0, u.curve.y);
            rq = u.curve.z * rq * rq;
            c = c * (max(rq, br - u.curve.w) / max(br, 0.0001));
            return vec4f(c, 0.0);
        }

        // Four bilinear taps around the texel: down a halving chain and back
        // up, a wide Gaussian for the price of a 4-tap kernel.
        fn bloom_blur(i: Frag) -> vec4f {
            var sum = tex_a(i.vl);
            sum = sum + tex_a(i.vr);
            sum = sum + tex_a(i.vt);
            sum = sum + tex_a(i.vb);
            return sum * 0.25;
        }

        fn bloom_final(i: Frag) -> vec4f {
            var sum = tex_a(i.vl);
            sum = sum + tex_a(i.vr);
            sum = sum + tex_a(i.vt);
            sum = sum + tex_a(i.vb);
            return sum * 0.25 * u.value;
        }

        // ------------------------------------------------------- sunrays

        // The mask the radial blur marches over, in alpha: dye occludes, so
        // the rays come out of the gaps between the ink.
        fn sunrays_mask(i: Frag) -> vec4f {
            var c = tex_a(i.uv);
            let br = max(c.r, max(c.g, c.b));
            c.a = 1.0 - min(max(br * 20.0, 0.0), 0.8);
            return c;
        }

        // Sixteen steps towards the centre of the canvas with a decaying
        // weight — god rays from a light at the middle of the screen.
        fn sunrays(i: Frag) -> vec4f {
            let density = 0.3;
            let decay = 0.95;
            let exposure = 0.7;
            let iterations = 16i32;
            var coord = i.uv;
            let dir = (i.uv - vec2f(0.5)) * (1.0 / f32(iterations) * density);
            var illumination = 1.0;
            var color = tex_a(i.uv).a;
            for (var k = 0i32; k < iterations; k += 1i32) {
                coord = coord - dir;
                color = color + tex_a(coord).a * illumination * u.value;
                illumination = illumination * decay;
            }
            return vec4f(color * exposure, 0.0, 0.0, 1.0);
        }

        // Three taps, 0.294/0.353/0.353, one axis per draw: drawn with
        // `vs_blur`, whose neighbours sit 1.333 texels out along it.
        fn blur(i: Frag) -> vec4f {
            var sum = tex_a(i.uv) * 0.29411764;
            sum = sum + tex_a(i.vl) * 0.35294117;
            sum = sum + tex_a(i.vr) * 0.35294117;
            return sum;
        }

        // ------------------------------------------------------- display

        fn linear_to_gamma(c: vec3f) -> vec3f {
            let m = max(c, vec3f(0.0));
            return max(1.055 * pow(m, vec3f(0.416666667)) - 0.055, vec3f(0.0));
        }

        // Interleaved gradient noise (Jimenez 2014), standing in for the
        // blue-noise texture the original samples. See `docs/fluid.md`.
        fn dither(p: vec2f) -> f32 {
            return fract(52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y));
        }

        // `tex_a` dye, `tex_b` bloom, `tex_c` sunrays; the three flags were
        // `#define`s and three variants in the original.
        fn display(i: Frag) -> vec4f {
            var c = tex_a(i.uv).rgb;
            if ((u.flags & FLAG_SHADING) != 0u32) {
                // Light the dye as though its brightness were a height field:
                // the normal comes from the gradient of the dye.
                let lc = tex_a(i.vl).rgb;
                let rc = tex_a(i.vr).rgb;
                let tc = tex_a(i.vt).rgb;
                let bc = tex_a(i.vb).rgb;
                let dx = length(rc) - length(lc);
                let dy = length(tc) - length(bc);
                let n = normalize(vec3f(dx, dy, length(u.texel_size)));
                let diffuse = clamp(dot(n, vec3f(0.0, 0.0, 1.0)) + 0.7, 0.7, 1.0);
                c = c * diffuse;
            }
            var bloom = vec3f(0.0);
            let has_bloom = (u.flags & FLAG_BLOOM) != 0u32;
            if (has_bloom) {
                bloom = tex_b(i.uv).rgb;
            }
            if ((u.flags & FLAG_SUNRAYS) != 0u32) {
                let rays = tex_c(i.uv).r;
                c = c * rays;
                bloom = bloom * rays;
            }
            if (has_bloom) {
                bloom = bloom + vec3f((dither(i.pos.xy) * 2.0 - 1.0) / 255.0);
                bloom = linear_to_gamma(bloom);
                c = c + bloom;
            }
            // Alpha is the brightest channel, so the dye composites over
            // whatever is behind it in proportion to how much ink is there.
            let a = max(c.r, max(c.g, c.b));
            return vec4f(undo_srgb(c), a);
        }
    }
});

/// The frame-rate read-out, over the finished picture: WGSL's own, beside
/// the generated module. `value` is the number; the glyphs are in
/// `super::read_out`, shared with the volume's shader.
const OVERLAY: &str = crate::wgsl!(
    r#"
@fragment
fn fs_overlay(i : VsOut) -> @location(0) vec4<f32> {
    let a = read_out(i.pos.xy, u.value);
    if (a > 0.0) {
        return vec4<f32>(1.0, 1.0, 1.0, a);
    }
    // `pass_tex_a` is the frame-time history for this pass, not the dye.
    let bar = chart(i.pos.xy, pass_tex_a, u.radius, u.curl);
    if (bar.a <= 0.0) {
        discard;
    }
    return bar;
}
"#
);

/// The whole wgpu module: the read-out's glyphs, the passes and the overlay.
pub(crate) fn wgsl() -> String {
    let passes = FLUID.wgsl(UNIFORMS).expect("the fluid's passes translate");
    format!("{}\n{passes}\n{OVERLAY}", super::read_out::glyphs())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    /// The generated module validates, and every entry point translates to
    /// GLSL ES 3.00 for WebGL2, where the fluid also runs: `vs`, `vs_blur`,
    /// each pass's `fs_*` and the read-out's `fs_overlay`.
    #[test]
    fn the_fluid_module_validates_and_translates_for_webgl2() {
        use crate::materials::shared_passes::{assert_webgl2, entry_points};
        let mut entries = entry_points(&super::FLUID);
        entries.push((naga::ShaderStage::Fragment, "fs_overlay".into()));
        assert_webgl2(&super::wgsl(), &entries);
    }
}
