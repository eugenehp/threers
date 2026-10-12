//! Every stage of the fluid, as one WGSL module.
//!
//! The original is nineteen GLSL programs sharing two vertex shaders. Here they
//! are one module with nineteen entry points over a single bind group, which is
//! what collapses the plumbing: one bind-group layout, one pipeline layout, one
//! shader compilation, and pipelines that differ only in which fragment entry
//! they name, what format they write and whether they blend.
//!
//! The cost of sharing one layout is that every pass binds three textures even
//! where it reads one; the unused slots get the same view as the used one. That
//! is a pointer each, and it buys away eighteen layouts.
//!
//! # The uniform block
//!
//! One 80-byte block, written per pass. A field means different things to
//! different stages — `value` is the clear multiplier, the bloom intensity and
//! the sunray weight — because padding a struct per stage costs more than the
//! comment that says so. Offsets are fixed by WGSL's layout rules and mirrored
//! by `Params` in `passes.rs`; the two are checked against each other by
//! `params_layout_matches_wgsl`.

/// The whole pipeline. See the module docs for how it is organised.
pub(crate) const SHADER: &str = r#"
struct Params {
    // 1 / resolution of the texture being *sampled*, in the pass's own space.
    texel_size   : vec2<f32>,
    // Splat centre, in uv.
    point        : vec2<f32>,
    // 1 / resolution of the *advected* field, which is not `texel_size` when
    // the dye grid is finer than the solver's. Only the MacCormack limiter
    // reads it.
    dye_texel_size : vec2<f32>,
    dt           : f32,
    dissipation  : f32,
    curl         : f32,
    // Canvas width / height. Keeps a splat round and the checkerboard square.
    aspect_ratio : f32,
    // Splat colour, or the flat background colour.
    color        : vec4<f32>,
    // Bloom knee curve: x = threshold - knee, y = 2 knee, z = 1/(4 knee),
    // w = threshold.
    curve        : vec4<f32>,
    radius       : f32,
    // Clear multiplier / bloom intensity / sunray weight, by pass.
    value        : f32,
    flags        : u32,
    pad          : u32,
};

@group(0) @binding(0) var<uniform> u : Params;
@group(0) @binding(1) var tex_a : texture_2d<f32>;
@group(0) @binding(2) var tex_b : texture_2d<f32>;
@group(0) @binding(3) var tex_c : texture_2d<f32>;
@group(0) @binding(4) var samp  : sampler;
// Up to `MAX_BATCH` splats, two vec4s each: (point.xy, radius, unused) and
// (colour.rgb, unused). A uniform array rather than a storage buffer because
// WebGL2 has no storage buffers and this has to run there.
@group(0) @binding(5) var<uniform> splats : array<vec4<f32>, 64>;

const FLAG_SHADING : u32 = 1u;
const FLAG_BLOOM   : u32 = 2u;
const FLAG_SUNRAYS : u32 = 4u;
// Set when the target is an sRGB format. The display values are already
// display-referred — the GLSL wrote them straight into an 8-bit buffer with no
// transfer function — so on an sRGB target the inverse goes in here and the
// hardware's encode puts the same bytes back.
const FLAG_ENCODE  : u32 = 8u;

struct VsOut {
    @builtin(position) pos : vec4<f32>,
    @location(0) uv : vec2<f32>,
    @location(1) vl : vec2<f32>,
    @location(2) vr : vec2<f32>,
    @location(3) vt : vec2<f32>,
    @location(4) vb : vec2<f32>,
};

// One triangle twice the size of the viewport; the corners outside it clip.
// Three vertices rather than a quad's six, and `uv` is the raw corner value,
// which lands (0,0) at the top left — the texture's own origin in WebGPU.
fn corner(vid : u32) -> vec2<f32> {
    return vec2<f32>(f32((vid << 1u) & 2u), f32(vid & 2u));
}

// The four neighbour taps are computed here, once per vertex, and interpolated:
// the offsets are affine in uv, so interpolation reproduces them exactly, and a
// fragment reading five texels does no address arithmetic.
@vertex
fn vs(@builtin(vertex_index) vid : u32) -> VsOut {
    let c = corner(vid);
    var o : VsOut;
    o.pos = vec4<f32>(c.x * 2.0 - 1.0, 1.0 - c.y * 2.0, 0.0, 1.0);
    o.uv = c;
    o.vl = o.uv - vec2<f32>(u.texel_size.x, 0.0);
    o.vr = o.uv + vec2<f32>(u.texel_size.x, 0.0);
    o.vt = o.uv + vec2<f32>(0.0, u.texel_size.y);
    o.vb = o.uv - vec2<f32>(0.0, u.texel_size.y);
    return o;
}

// The separable blur's taps sit at 1.3333 texels, not 1: two bilinear fetches
// at that offset plus the centre give the same weights as a five-tap kernel for
// three fetches. `texel_size` carries the axis, so one entry point does both.
@vertex
fn vs_blur(@builtin(vertex_index) vid : u32) -> VsOut {
    let c = corner(vid);
    var o : VsOut;
    o.pos = vec4<f32>(c.x * 2.0 - 1.0, 1.0 - c.y * 2.0, 0.0, 1.0);
    o.uv = c;
    let offset = u.texel_size * 1.33333333;
    o.vl = o.uv - offset;
    o.vr = o.uv + offset;
    o.vt = o.uv;
    o.vb = o.uv;
    return o;
}

// ---------------------------------------------------------------------------
// Utility passes
// ---------------------------------------------------------------------------

@fragment
fn fs_copy(i : VsOut) -> @location(0) vec4<f32> {
    return textureSample(tex_a, samp, i.uv);
}

// Scales a field by `value`. Used on pressure between frames: starting the
// Jacobi solve from a fraction of last frame's answer converges faster than
// starting from zero, and the fraction is what the `pressure` knob sets.
@fragment
fn fs_clear(i : VsOut) -> @location(0) vec4<f32> {
    return u.value * textureSample(tex_a, samp, i.uv);
}

// The display values are display-referred: the GLSL wrote them into an 8-bit
// buffer with no transfer function applied. On a linear target that is still
// what to store, so this is the identity. On an sRGB target the hardware
// encodes whatever the shader returns, so the inverse goes in first and the
// same bytes come back out.
fn undo_srgb(c : vec3<f32>) -> vec3<f32> {
    if ((u.flags & FLAG_ENCODE) == 0u) {
        return c;
    }
    let m = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    let lo = m / 12.92;
    let hi = pow((m + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, m <= vec3<f32>(0.04045));
}

@fragment
fn fs_color(i : VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(undo_srgb(u.color.rgb), u.color.a);
}

// What "transparent" looks like when there is no page behind the canvas.
@fragment
fn fs_checkerboard(i : VsOut) -> @location(0) vec4<f32> {
    let cell = floor(i.uv * 25.0 * vec2<f32>(u.aspect_ratio, 1.0));
    let v = ((cell.x + cell.y) % 2.0) * 0.1 + 0.8;
    return vec4<f32>(undo_srgb(vec3<f32>(v)), 1.0);
}

// A Gaussian blob added to whatever is already there. The same pass seeds both
// fields: into velocity `color` is the pointer's motion, into dye it is a hue.
@fragment
fn fs_splat(i : VsOut) -> @location(0) vec4<f32> {
    // Every splat in the batch in one pass. A pass per splat is what this
    // replaced, and at a 4096-texel dye grid that was two full-screen passes
    // each — a burst of twenty-five cost eighteen milliseconds, about four
    // frames, for twenty-five Gaussians. Summing them here also accumulates in
    // f32 and rounds once, where a pass each rounded to half float in between.
    var acc = vec3<f32>(0.0);
    let n = u32(u.value);
    for (var k = 0u; k < n; k = k + 1u) {
        let a = splats[k * 2u];
        let b = splats[k * 2u + 1u];
        var p = i.uv - a.xy;
        p.x = p.x * u.aspect_ratio;
        let d = dot(p, p);
        // Outside about three and a half sigma the Gaussian is under 6e-6 of
        // its peak, which is below what a half-float dye texture can hold at
        // any brightness worth splatting. Skipping it turns the batch from
        // ALU-bound back into bandwidth-bound: without this a texel in an
        // empty corner still evaluated an `exp` for every blob on the canvas.
        if (d < a.z * 12.0) {
            acc = acc + exp(-d / a.z) * b.rgb;
        }
    }
    let base = textureSample(tex_a, samp, i.uv).rgb;
    return vec4<f32>(base + acc, 1.0);
}

// ---------------------------------------------------------------------------
// The solver
// ---------------------------------------------------------------------------

// Semi-Lagrangian advection (Stam 1999): instead of pushing a quantity forward
// — which needs somewhere to put it and is unstable past a CFL step — look
// backwards along the velocity for where this texel's contents came from, and
// fetch it. Unconditionally stable at any dt, which is what lets a browser tab
// drop to 20 fps without the fluid blowing up.
//
// `tex_a` is the velocity, `tex_b` the field being carried; for velocity
// advection they are the same texture.
@fragment
fn fs_advection(i : VsOut) -> @location(0) vec4<f32> {
    let coord = i.uv - u.dt * textureSample(tex_a, samp, i.uv).xy * u.texel_size;
    let result = textureSample(tex_b, samp, coord);
    // Dissipation as a decay per unit time rather than per frame, so the fade
    // does not depend on the frame rate. Zero on the forward half of a
    // MacCormack step, which applies it once at the end instead.
    let decay = 1.0 + u.dissipation * u.dt;
    return result / decay;
}

// The second half of a MacCormack advection.
//
// Semi-Lagrangian advection is first order, and its error is *diffusion*: the
// backward trace lands between texels and the bilinear fetch averages them, so
// every step blurs a little. Over a few hundred frames that is most of why
// dye goes muddy and why small vortices stop existing.
//
// MacCormack estimates that error and subtracts it. Advect forward to get
// `fwd`; advect *that* backward and you should arrive where you started, so
// whatever is left over is the scheme's own error, and half of it comes off.
// The result is second order in space for one extra pass.
//
// It can also overshoot — the correction is an extrapolation, and an
// extrapolation off a steep edge produces values that were never in the field,
// which on dye reads as ringing and on velocity as instability. So the
// corrected value is clamped to the range of the four source texels the
// forward pass actually read. That limiter is what makes it unconditionally
// stable (Selle et al. 2008) rather than merely more accurate.
//
// `tex_a` velocity, `tex_b` the forward result, `tex_c` the field as it was.
@fragment
fn fs_advect_correct(i : VsOut) -> @location(0) vec4<f32> {
    let v = textureSample(tex_a, samp, i.uv).xy * (u.dt * u.texel_size);
    let fwd = textureSample(tex_b, samp, i.uv);
    // Advecting the forward result backwards is a forward trace of the same
    // step, which is why this adds where the forward pass subtracted.
    let back = textureSample(tex_b, samp, i.uv + v);
    let src = textureSample(tex_c, samp, i.uv);
    var corrected = fwd + 0.5 * (src - back);

    // The four texels the forward pass's bilinear fetch drew on. `dye_texel_size`
    // is the advected field's own, which is not the velocity's when the dye
    // grid is finer.
    let coord = i.uv - v;
    let t = u.dye_texel_size;
    let st = coord / t - 0.5;
    let base = (floor(st) + 0.5) * t;
    let s00 = textureSample(tex_c, samp, base);
    let s10 = textureSample(tex_c, samp, base + vec2<f32>(t.x, 0.0));
    let s01 = textureSample(tex_c, samp, base + vec2<f32>(0.0, t.y));
    let s11 = textureSample(tex_c, samp, base + t);
    let lo = min(min(s00, s10), min(s01, s11));
    let hi = max(max(s00, s10), max(s01, s11));
    corrected = clamp(corrected, lo, hi);

    let decay = 1.0 + u.dissipation * u.dt;
    return corrected / decay;
}

// How much the velocity field is locally creating or destroying fluid. Zero
// everywhere is what incompressibility means, and this is the right-hand side
// of the pressure solve that enforces it.
@fragment
fn fs_divergence(i : VsOut) -> @location(0) vec4<f32> {
    var l = textureSample(tex_a, samp, i.vl).x;
    var r = textureSample(tex_a, samp, i.vr).x;
    var t = textureSample(tex_a, samp, i.vt).y;
    var b = textureSample(tex_a, samp, i.vb).y;
    let c = textureSample(tex_a, samp, i.uv).xy;
    // A free-slip wall: outside the domain the normal component is mirrored, so
    // the boundary neither leaks nor drags. Clamp-to-edge sampling would
    // otherwise read the edge texel back and let flow run off the canvas.
    if (i.vl.x < 0.0) { l = -c.x; }
    if (i.vr.x > 1.0) { r = -c.x; }
    if (i.vt.y > 1.0) { t = -c.y; }
    if (i.vb.y < 0.0) { b = -c.y; }
    return vec4<f32>(0.5 * (r - l + t - b), 0.0, 0.0, 1.0);
}

// Vorticity: the local spin of the field. Written in texture space, where y
// runs downward, which makes the whole simulation the mirror image of the
// GL one — and mirror images of a fluid are fluids, so the formulae are
// unchanged. See the module docs.
@fragment
fn fs_curl(i : VsOut) -> @location(0) vec4<f32> {
    let l = textureSample(tex_a, samp, i.vl).y;
    let r = textureSample(tex_a, samp, i.vr).y;
    let t = textureSample(tex_a, samp, i.vt).x;
    let b = textureSample(tex_a, samp, i.vb).x;
    return vec4<f32>(0.5 * (r - l - t + b), 0.0, 0.0, 1.0);
}

// Vorticity confinement (Steinhoff & Underhill 1994). A grid this coarse
// diffuses small eddies away within a few frames; this finds where vorticity is
// concentrated and pushes along it, putting back the curl the discretisation
// ate. It is not a term in Navier-Stokes — it is an anti-dissipation hack —
// and it is the single knob that decides whether the picture looks like smoke
// or like spilled paint.
//
// `tex_a` velocity, `tex_b` curl.
@fragment
fn fs_vorticity(i : VsOut) -> @location(0) vec4<f32> {
    let l = textureSample(tex_b, samp, i.vl).x;
    let r = textureSample(tex_b, samp, i.vr).x;
    let t = textureSample(tex_b, samp, i.vt).x;
    let b = textureSample(tex_b, samp, i.vb).x;
    let c = textureSample(tex_b, samp, i.uv).x;

    // N = grad|w| normalised, rotated a quarter turn: force = e * w * (Ny, -Nx).
    var force = 0.5 * vec2<f32>(abs(t) - abs(b), abs(r) - abs(l));
    force = force / (length(force) + 0.0001);
    force = force * (u.curl * c);
    force.y = -force.y;

    var velocity = textureSample(tex_a, samp, i.uv).xy;
    velocity = velocity + force * u.dt;
    // The clamp is load-bearing: the field is half float, and one unbounded
    // texel would reach infinity and then NaN, which advection would smear
    // across the whole canvas in a few frames.
    velocity = clamp(velocity, vec2<f32>(-1000.0), vec2<f32>(1000.0));
    return vec4<f32>(velocity, 0.0, 1.0);
}

// One Jacobi sweep of grad^2 p = div u. Twenty of these is not a converged
// Poisson solve, and does not need to be: what is wanted is that the velocity
// comes out *visibly* divergence-free, and the residual that remains is smooth
// and below the dye's ability to show it.
//
// `tex_a` pressure, `tex_b` divergence.
// `value` is the relaxation weight. At 1 this is plain Jacobi, which is what
// the original runs and what `PressureSolver::Jacobi` asks for.
//
// It is not 1 when this is a multigrid *smoother*, and the reason is the one
// mode plain Jacobi cannot touch. The iteration's amplification factor for a
// mode is `(cos kx + cos ky) / 2`; at the checkerboard mode, where both are
// -1, that is exactly **-1** — the error does not shrink, it changes sign
// every sweep. A V-cycle depends on its smoother having removed precisely the
// frequencies the coarse grid cannot represent, and that is the highest one of
// all, so an undamped smoother makes multigrid stall at whatever checkerboard
// error the right-hand side contains. Weighting to 4/5 takes that factor to
// `|1 - 2w|` = 0.6 a sweep, at the cost of damping the smooth modes slightly
// less — which is what the coarse grid is for.
@fragment
fn fs_pressure(i : VsOut) -> @location(0) vec4<f32> {
    let l = textureSample(tex_a, samp, i.vl).x;
    let r = textureSample(tex_a, samp, i.vr).x;
    let t = textureSample(tex_a, samp, i.vt).x;
    let b = textureSample(tex_a, samp, i.vb).x;
    let c = textureSample(tex_a, samp, i.uv).x;
    let divergence = textureSample(tex_b, samp, i.uv).x;
    let jacobi = (l + r + b + t - divergence) * 0.25;
    return vec4<f32>(mix(c, jacobi, u.value), 0.0, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// Multigrid
// ---------------------------------------------------------------------------
//
// Jacobi removes error at the scale of its own stencil, so the error it is
// slowest on is the one spread across the whole grid — and the pressure from a
// localised puff is exactly that, a long-ranged field. Sixteen times the
// sweeps still leaves a third of the divergence (see the tests). Multigrid
// answers it by solving the smooth part of the problem on a smaller grid,
// where "spread across the whole grid" is a handful of texels.
//
// Three passes make the cycle, on top of the Jacobi sweep already here.

// What the current pressure field fails to satisfy: r = rhs - L(p), with L the
// same five-point stencil `fs_pressure` inverts.
//
// `tex_a` pressure, `tex_b` right-hand side.
@fragment
fn fs_residual(i : VsOut) -> @location(0) vec4<f32> {
    let l = textureSample(tex_a, samp, i.vl).x;
    let r = textureSample(tex_a, samp, i.vr).x;
    let t = textureSample(tex_a, samp, i.vt).x;
    let b = textureSample(tex_a, samp, i.vb).x;
    let c = textureSample(tex_a, samp, i.uv).x;
    let rhs = textureSample(tex_b, samp, i.uv).x;
    return vec4<f32>(rhs - (l + r + t + b - 4.0 * c), 0.0, 0.0, 1.0);
}

// The residual, onto a grid of half the size, as the right-hand side of the
// coarse problem.
//
// Full weighting over the four fine texels a coarse one covers — two bilinear
// fetches at the corners between them would be cheaper, but the explicit form
// is the one that is obviously an average. The factor of four is the change of
// units: the stencil is written with no 1/h², so doubling the spacing
// quadruples what the same equation's right-hand side has to be.
@fragment
fn fs_restrict(i : VsOut) -> @location(0) vec4<f32> {
    // `texel_size` is the *fine* grid's, so these four are its texel centres
    // inside this coarse texel.
    let h = u.texel_size * 0.5;
    var sum = textureSample(tex_a, samp, i.uv + vec2<f32>(-h.x, -h.y)).x;
    sum = sum + textureSample(tex_a, samp, i.uv + vec2<f32>(h.x, -h.y)).x;
    sum = sum + textureSample(tex_a, samp, i.uv + vec2<f32>(-h.x, h.y)).x;
    sum = sum + textureSample(tex_a, samp, i.uv + vec2<f32>(h.x, h.y)).x;
    return vec4<f32>(sum, 0.0, 0.0, 1.0);
}

// Add the coarse correction back into the fine pressure.
//
// The interpolation is the sampler's: reading a half-size texture at a fine
// texel centre with a linear filter *is* bilinear prolongation, so there is no
// arithmetic here at all.
//
// `tex_a` coarse correction, `tex_b` fine pressure.
@fragment
fn fs_prolong(i : VsOut) -> @location(0) vec4<f32> {
    let coarse = textureSample(tex_a, samp, i.uv).x;
    let fine = textureSample(tex_b, samp, i.uv).x;
    return vec4<f32>(fine + coarse, 0.0, 0.0, 1.0);
}

// The projection step: subtracting the pressure gradient removes exactly the
// part of the velocity that was compressing the fluid and leaves the rest.
//
// `(r - l, t - b)` is *twice* the central-difference gradient, so the
// analytically consistent coefficient is the 0.5 that `divergence` and the
// Jacobi sweep both carry — `halfrdx` in GPU Gems 38. `value` is that
// coefficient and it is 1.0 by default, which over-relaxes by exactly two and
// is the right thing to do with a solve this far from converged. The numbers
// are in `FluidConfig::gradient_scale`.
//
// Pressure needs a pure Neumann boundary, dp/dn = 0, which means the value just
// outside the domain equals the one just inside. Clamp-to-edge sampling is
// exactly that, so unlike velocity this needs no boundary pass at all.
//
// `tex_a` pressure, `tex_b` velocity.
@fragment
fn fs_gradient(i : VsOut) -> @location(0) vec4<f32> {
    let l = textureSample(tex_a, samp, i.vl).x;
    let r = textureSample(tex_a, samp, i.vr).x;
    let t = textureSample(tex_a, samp, i.vt).x;
    let b = textureSample(tex_a, samp, i.vb).x;
    var velocity = textureSample(tex_b, samp, i.uv).xy;
    velocity = velocity - u.value * vec2<f32>(r - l, t - b);
    return vec4<f32>(velocity, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// Bloom
// ---------------------------------------------------------------------------

// Soft-knee threshold (Unity's curve): hard-clipping the highlight makes the
// bloom pop in as a pixel crosses the threshold, and the knee makes it fade in
// over a range instead.
@fragment
fn fs_bloom_prefilter(i : VsOut) -> @location(0) vec4<f32> {
    var c = textureSample(tex_a, samp, i.uv).rgb;
    let br = max(c.r, max(c.g, c.b));
    var rq = clamp(br - u.curve.x, 0.0, u.curve.y);
    rq = u.curve.z * rq * rq;
    c = c * (max(rq, br - u.curve.w) / max(br, 0.0001));
    return vec4<f32>(c, 0.0);
}

// Four bilinear taps on the diagonal neighbours — sixteen texels for four
// fetches. Run down a halving chain and back up, this is a wide Gaussian for
// the price of a 4-tap kernel.
@fragment
fn fs_bloom_blur(i : VsOut) -> @location(0) vec4<f32> {
    var sum = textureSample(tex_a, samp, i.vl);
    sum = sum + textureSample(tex_a, samp, i.vr);
    sum = sum + textureSample(tex_a, samp, i.vt);
    sum = sum + textureSample(tex_a, samp, i.vb);
    return sum * 0.25;
}

@fragment
fn fs_bloom_final(i : VsOut) -> @location(0) vec4<f32> {
    var sum = textureSample(tex_a, samp, i.vl);
    sum = sum + textureSample(tex_a, samp, i.vr);
    sum = sum + textureSample(tex_a, samp, i.vt);
    sum = sum + textureSample(tex_a, samp, i.vb);
    return sum * 0.25 * u.value;
}

// ---------------------------------------------------------------------------
// Sunrays
// ---------------------------------------------------------------------------

// The mask the radial blur marches over, written into alpha: dye *occludes*,
// so the rays come out of the gaps between the ink rather than out of the ink.
@fragment
fn fs_sunrays_mask(i : VsOut) -> @location(0) vec4<f32> {
    var c = textureSample(tex_a, samp, i.uv);
    let br = max(c.r, max(c.g, c.b));
    c.a = 1.0 - min(max(br * 20.0, 0.0), 0.8);
    return c;
}

// Sixteen steps towards the centre of the canvas with a decaying weight —
// god rays, from a light at the middle of the screen.
@fragment
fn fs_sunrays(i : VsOut) -> @location(0) vec4<f32> {
    let density = 0.3;
    let decay = 0.95;
    let exposure = 0.7;
    let iterations = 16;

    var coord = i.uv;
    let dir = (i.uv - vec2<f32>(0.5)) * (1.0 / f32(iterations) * density);
    var illumination = 1.0;
    var color = textureSample(tex_a, samp, i.uv).a;
    for (var k = 0; k < iterations; k = k + 1) {
        coord = coord - dir;
        color = color + textureSample(tex_a, samp, coord).a * illumination * u.value;
        illumination = illumination * decay;
    }
    return vec4<f32>(color * exposure, 0.0, 0.0, 1.0);
}

// Three taps, 0.294/0.353/0.353, one axis per invocation.
@fragment
fn fs_blur(i : VsOut) -> @location(0) vec4<f32> {
    var sum = textureSample(tex_a, samp, i.uv) * 0.29411764;
    sum = sum + textureSample(tex_a, samp, i.vl) * 0.35294117;
    sum = sum + textureSample(tex_a, samp, i.vr) * 0.35294117;
    return sum;
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

fn linear_to_gamma(c : vec3<f32>) -> vec3<f32> {
    let m = max(c, vec3<f32>(0.0));
    return max(1.055 * pow(m, vec3<f32>(0.416666667)) - 0.055, vec3<f32>(0.0));
}

// Interleaved gradient noise (Jimenez 2014), standing in for the blue-noise
// texture the original samples. A dither of one 255th only has to be spatially
// decorrelated; this is, it costs no texture and no asset, and it keeps the
// simulation a function of its own state. See `docs/fluid.md`.
fn dither(p : vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y));
}

// `tex_a` dye, `tex_b` bloom, `tex_c` sunrays. The three flags were `#define`s
// and three compiled variants in the original; here they are bits in the
// uniform, which is coherent across the draw and costs a scalar branch.
@fragment
fn fs_display(i : VsOut) -> @location(0) vec4<f32> {
    var c = textureSample(tex_a, samp, i.uv).rgb;

    if ((u.flags & FLAG_SHADING) != 0u) {
        // Light the dye as though its brightness were a height field. There is
        // no geometry here at all — the normal comes from the gradient of the
        // dye — and it is most of what makes the fluid read as volume.
        let lc = textureSample(tex_a, samp, i.vl).rgb;
        let rc = textureSample(tex_a, samp, i.vr).rgb;
        let tc = textureSample(tex_a, samp, i.vt).rgb;
        let bc = textureSample(tex_a, samp, i.vb).rgb;
        let dx = length(rc) - length(lc);
        let dy = length(tc) - length(bc);
        let n = normalize(vec3<f32>(dx, dy, length(u.texel_size)));
        let diffuse = clamp(dot(n, vec3<f32>(0.0, 0.0, 1.0)) + 0.7, 0.7, 1.0);
        c = c * diffuse;
    }

    var bloom = vec3<f32>(0.0);
    let has_bloom = (u.flags & FLAG_BLOOM) != 0u;
    if (has_bloom) {
        bloom = textureSample(tex_b, samp, i.uv).rgb;
    }
    if ((u.flags & FLAG_SUNRAYS) != 0u) {
        let rays = textureSample(tex_c, samp, i.uv).r;
        c = c * rays;
        bloom = bloom * rays;
    }
    if (has_bloom) {
        bloom = bloom + vec3<f32>((dither(i.pos.xy) * 2.0 - 1.0) / 255.0);
        bloom = linear_to_gamma(bloom);
        c = c + bloom;
    }

    // Alpha is the brightest channel, so the dye composites over whatever is
    // behind it in proportion to how much ink is there. That is also what makes
    // a transparent capture look right over a page.
    let a = max(c.r, max(c.g, c.b));
    return vec4<f32>(undo_srgb(c), a);
}
"#;
