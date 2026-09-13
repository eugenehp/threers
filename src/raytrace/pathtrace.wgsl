// Path tracer, as a compute kernel.
//
// This is a transcription of the CPU integrator: same BSDF, same next-event
// estimation, same power heuristic, same Russian roulette. Where it differs is
// only in how work is laid out, and every one of those differences is about
// keeping a GPU busy:
//
//   * One invocation owns one pixel for the whole batch. The accumulator is
//     therefore written by exactly one thread, so no atomics and no contention.
//   * The scene arrives in four storage buffers rather than a dozen, because
//     WebGPU's baseline guarantees only four per stage. Everything is addressed
//     through offsets in the uniform block, which also means one bind group for
//     any scene.
//   * Textures are one atlas, sampled by `textureLoad` with the filtering done
//     here. Per-material texture bindings would need bindless, which wgpu 0.20
//     does not have.
//   * The traversal stack is function-local and 32 deep, matching the builder's
//     depth cap, so it stays in registers rather than spilling.
//
// Batching happens on the host: `sample_count` samples are traced per dispatch
// and the host issues as many dispatches as the render needs. Long dispatches
// trip watchdog timers on every platform, and a batch that finishes in tens of
// milliseconds keeps the device responsive and the render interruptible.

const PI: f32 = 3.14159265358979;
// Hex float, not decimal. `3.4028235e38` is the usual spelling of f32::MAX, but
// it expands to 340282349999999991754788743781432688640.0, which rounds ABOVE
// f32::MAX — naga accepts it, Tint rejects it outright:
//
//   error: value 340282...640.0 cannot be represented as 'f32'
//
// The whole module then fails to compile, so on WebGPU there was no compute
// pipeline at all: every dispatch was a no-op and the film came back black,
// with no error surfaced unless something installs an uncaptured-error handler.
// The hex form is exactly f32::MAX with no decimal rounding to argue about.
const INF: f32 = 0x1.fffffep+127;
const SMOOTH_ALPHA: f32 = 1e-3;
const MIN_ALPHA: f32 = 1e-4;
const MAT_STRIDE: u32 = 88u;
const LIGHT_STRIDE: u32 = 20u;
// 9 position + 9 normal + 6 uv + 9 colour. MUST match `TRI_STRIDE` in gpu.rs.
const TRI_STRIDE: u32 = 33u;
const EMIT_STRIDE: u32 = 2u;
const ACCUM_STRIDE: u32 = 14u;
// The builder caps tree depth at 64, and traversal defers at most one sibling
// per level, so 64 entries is exactly enough. It was 32, which silently dropped
// nodes on any tree deeper than that — geometry missing from the render and
// light leaking through it, with nothing to say so.
const MAX_STACK: u32 = 64u;

struct Uniforms {
    // xyz = camera position, w = 1 for perspective
    cam_pos: vec4<f32>,
    // xyz = camera right, w = aperture
    cam_right: vec4<f32>,
    // xyz = camera up, w = focus distance
    cam_up: vec4<f32>,
    // xyz = camera forward, w = ray epsilon
    cam_forward: vec4<f32>,
    inv_view: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    // rgb = scene background, a = background alpha
    background: vec4<f32>,
    // rgb = summed ambient light, a = environment intensity
    ambient: vec4<f32>,
    // width, height, first_sample, sample_count
    dims: vec4<u32>,
    // triangles, materials, lights, emissive triangles
    counts: vec4<u32>,
    // into `data`: triangles, emissive, materials, lights
    data_offsets: vec4<u32>,
    // into `idx`: order, per-triangle material, emissive slot, emissive triangle
    idx_offsets: vec4<u32>,
    // clamp_direct, clamp_indirect, environment distribution integral, unused
    params: vec4<f32>,
    // adaptive threshold, minimum samples before it is trusted, unused, unused
    adaptive: vec4<f32>,
    // max_bounces, min_bounces, transparent_max_bounces, background_mode
    limits: vec4<u32>,
    // seed low, seed high, hemisphere count,
    // flags: bit0 environment, bit1 opaque shadows, bit2 caustic glass shadows
    flags: vec4<u32>,
    // into `data`: environment function, conditional CDFs, marginal CDF, unused
    env_offsets: vec4<u32>,
    // distribution width, height, 1 if present, unused
    env_dims: vec4<u32>,
    // per cube face: atlas x, y, w, h — face order +X, -X, +Y, -Y, +Z, -Z
    env_rect: array<vec4<f32>, 6>,
    // per hemisphere light: sky, ground, up
    hemi: array<vec4<f32>, 12>,
    // Previous-frame camera for motion blur when params.z > 0 (params = clamp_direct,
    // clamp_indirect, env_integral, motion_shutter).
    motion_inv_view: mat4x4<f32>,
    motion_inv_proj: mat4x4<f32>,
    motion_prev: vec4<f32>,
    motion_prev_right: vec4<f32>,
    motion_prev_up: vec4<f32>,
    motion_prev_forward: vec4<f32>,
    // Pixel clip: xy = origin, zw = size. Size zero means the full frame.
    clip: vec4<u32>,
    // x = scene scale, y = sample redistribution on, zw unused.
    render_params: vec4<f32>,
    // fog rgb + mode (0 off, 1 linear, 2 exp2)
    fog_color: vec4<f32>,
    // fog near, far, density, unused
    fog_params: vec4<f32>,
    // GGX directional albedo, 32x32, cos varying fastest
    albedo_lut: array<vec4<f32>, 256>,
};

@group(0) @binding(0) var<uniform> uni: Uniforms;
// Two vec4 a node: (min, bitcast left_or_first), (max, bitcast count).
@group(0) @binding(1) var<storage, read> nodes: array<vec4<f32>>;
// Triangle geometry, emissive area/CDF, material and light parameters.
@group(0) @binding(2) var<storage, read> data: array<f32>;
// Triangle order, per-triangle material, emissive slot, emissive triangle.
@group(0) @binding(3) var<storage, read> idx: array<u32>;
// The film: 12 floats a pixel.
@group(0) @binding(4) var<storage, read_write> accum: array<f32>;
@group(0) @binding(5) var atlas: texture_2d<f32>;

// ---------------------------------------------------------------- random

// Dimension pairs reserved per bounce, matching the CPU sampler.
const DIMENSIONS_PER_BOUNCE: u32 = 8u;

// The path's source of randomness: an independent stream for 1D decisions and
// a stratified low-discrepancy sequence for 2D ones. Nearly every decision a
// path makes is two-dimensional, which is why the 2D side is where the work is.
struct Sampler {
    rng: u32,
    seed: u32,
    index: u32,
    dim: u32,
};

// PCG hash. One multiply and two xorshifts, and it decorrelates adjacent
// integers well enough that neighbouring pixels show no pattern — which a
// linear congruential generator seeded by pixel index emphatically does not.
fn pcg(state: ptr<function, u32>) -> u32 {
    let s = *state * 747796405u + 2891336453u;
    *state = s;
    let word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (word >> 22u) ^ word;
}

fn hash32(x_in: u32) -> u32 {
    var x = x_in;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

// 32×32 → 64-bit product (low limb, high limb).
fn mul32x32(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xffffu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xffffu;
    let b_hi = b >> 16u;
    let p0 = a_lo * b_lo;
    let p1 = a_lo * b_hi;
    let p2 = a_hi * b_lo;
    let p3 = a_hi * b_hi;
    let mid = (p0 >> 16u) + (p1 & 0xffffu) + (p2 & 0xffffu);
    let lo = (p0 & 0xffffu) | (mid << 16u);
    let hi = (p1 >> 16u) + (p2 >> 16u) + p3 + (mid >> 16u);
    return vec2(lo, hi);
}

fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    let carry = select(0u, 1u, lo < a.x);
    return vec2(lo, a.y + b.y + carry);
}

fn u64_xor(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2(a.x ^ b.x, a.y ^ b.y);
}

fn u64_shr(v: vec2<u32>, n: u32) -> vec2<u32> {
    if (n >= 32u) {
        return vec2(v.y >> (n - 32u), 0u);
    }
    return vec2((v.x >> n) | (v.y << (32u - n)), v.y >> n);
}

fn u64_mul(a: vec2<u32>, k: vec2<u32>) -> vec2<u32> {
    let ak0 = mul32x32(a.x, k.x);
    let ak1 = mul32x32(a.x, k.y);
    let ka0 = mul32x32(a.y, k.x);
    var lo = ak0.x;
    var hi = ak0.y + ak1.x + ka0.x;
    let c0 = select(0u, 1u, lo < ak0.x);
    hi = hi + c0;
    let c1 = select(0u, 1u, hi < ak0.y);
    hi = hi + c1;
    return vec2(lo, hi);
}

// splitmix64 finaliser — same as [`crate::raytrace::sampler::hash64`].
fn hash64(lo: u32, hi: u32) -> vec2<u32> {
    var z = u64_add(vec2(lo, hi), vec2(0x7f4a7c15u, 0x9e3779b9u));
    var t = u64_xor(z, u64_shr(z, 30u));
    t = u64_mul(t, vec2(0x1ce4e5b9u, 0xbf58476du));
    t = u64_xor(t, u64_shr(t, 27u));
    t = u64_mul(t, vec2(0x133111ebu, 0x94d049bbu));
    t = u64_xor(t, u64_shr(t, 31u));
    return t;
}

fn u64_rotl17(lo: u32, hi: u32) -> vec2<u32> {
    return vec2((lo << 17u) | (hi >> 15u), (hi << 17u) | (lo >> 15u));
}

// Per-pixel Owen scramble, matching `Rng::for_sample` on the CPU backend.
fn pixel_scramble(x: u32, y: u32) -> u32 {
    let pixel = vec2(x, y); // lo = x, hi = y  ⟺  (y << 32) | x
    let seed = vec2(uni.flags.x, uni.flags.y);
    let mixed = u64_xor(pixel, seed);
    let h = hash64(mixed.x, mixed.y);
    return h.y;
}

// PCG stream tag for 1D draws — `hash64(pixel ^ seed.rotate_left(17))` on CPU.
fn pcg_stream_tag(x: u32, y: u32) -> u32 {
    let pixel = vec2(x, y);
    let seed = vec2(uni.flags.x, uni.flags.y);
    let rot = u64_rotl17(seed.x, seed.y);
    let mixed = u64_xor(pixel, rot);
    let h = hash64(mixed.x, mixed.y);
    return h.x ^ h.y;
}

// Hash-based nested uniform (Owen) scramble, after Burley 2020. A true Owen
// scramble permutes every node of the binary tree of digits independently,
// which is prohibitive; four rounds of a self-multiplying mix between two bit
// reversals reproduce its statistics.
fn owen_scramble(x_in: u32, seed: u32) -> u32 {
    var x = reverseBits(x_in);
    x = x + seed;
    x = x ^ (x * 0x6c50b47cu);
    x = x ^ (x * 0xb82f1e52u);
    x = x ^ (x * 0xc7afe638u);
    x = x ^ (x * 0x8d22f6e6u);
    return reverseBits(x);
}

// Second Sobol dimension, by the Gray-code recurrence. The first is just the
// index with its bits reversed.
fn sobol_y(index_in: u32) -> u32 {
    var index = index_in;
    var result = 0u;
    var v = 1u << 31u;
    loop {
        if (index == 0u) {
            break;
        }
        if ((index & 1u) != 0u) {
            result = result ^ v;
        }
        index = index >> 1u;
        v = v ^ (v >> 1u);
    }
    return result;
}

// A 2D sample from an Owen-scrambled Sobol (0,2)-sequence.
fn sobol02_owen(index: u32, seed: u32) -> vec2<f32> {
    // Shuffling the index as well as the values is what decorrelates two
    // pixels that draw the same dimension.
    let shuffled = owen_scramble(index, hash32(seed ^ 0x9e3779b9u));
    let x = owen_scramble(reverseBits(shuffled), hash32(seed ^ 0x6c50b47cu));
    let y = owen_scramble(sobol_y(shuffled), hash32(seed ^ 0xb82f1e52u));
    return vec2<f32>(f32(x >> 8u), f32(y >> 8u)) * (1.0 / 16777216.0);
}

fn rand(s: ptr<function, Sampler>) -> f32 {
    var state = (*s).rng;
    let v = pcg(&state);
    (*s).rng = state;
    // Top 24 bits: every bit an f32 in [0,1) can represent, and never 1.0.
    return f32(v >> 8u) * (1.0 / 16777216.0);
}

fn rand2(s: ptr<function, Sampler>) -> vec2<f32> {
    let d = (*s).dim;
    (*s).dim = d + 1u;
    return sobol02_owen((*s).index, (*s).seed ^ hash32(d + 1u));
}

// Point the sampler at a bounce's block of dimensions, so the same decision
// draws the same dimension in every sample of a pixel.
fn sampler_set_bounce(s: ptr<function, Sampler>, bounce: u32) {
    (*s).dim = bounce * DIMENSIONS_PER_BOUNCE;
}

// ---------------------------------------------------------------- sampling

struct Onb {
    t: vec3<f32>,
    b: vec3<f32>,
    n: vec3<f32>,
};

fn onb_make(n: vec3<f32>) -> Onb {
    // Duff et al: branchless and exact at both poles.
    let sign = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    var o: Onb;
    o.t = vec3<f32>(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    o.b = vec3<f32>(b, sign + n.y * n.y * a, -n.y);
    o.n = n;
    return o;
}

fn onb_to_world(o: Onb, v: vec3<f32>) -> vec3<f32> {
    return o.t * v.x + o.b * v.y + o.n * v.z;
}

fn onb_to_local(o: Onb, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(v, o.t), dot(v, o.b), dot(v, o.n));
}

fn concentric_disk(u: vec2<f32>) -> vec2<f32> {
    let o = 2.0 * u - vec2<f32>(1.0, 1.0);
    if (o.x == 0.0 && o.y == 0.0) {
        return vec2<f32>(0.0, 0.0);
    }
    var r: f32;
    var theta: f32;
    if (abs(o.x) > abs(o.y)) {
        r = o.x;
        theta = (PI / 4.0) * (o.y / o.x);
    } else {
        r = o.y;
        theta = PI / 2.0 - (PI / 4.0) * (o.x / o.y);
    }
    return vec2<f32>(r * cos(theta), r * sin(theta));
}

fn cosine_hemisphere(u: vec2<f32>) -> vec3<f32> {
    let d = concentric_disk(u);
    let z = sqrt(max(0.0, 1.0 - d.x * d.x - d.y * d.y));
    return vec3<f32>(d.x, d.y, z);
}

fn uniform_cone(u: vec2<f32>, cos_max: f32) -> vec3<f32> {
    let cos_theta = 1.0 - u.x * (1.0 - cos_max);
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    let phi = 2.0 * PI * u.y;
    return vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

fn uniform_cone_pdf(cos_max: f32) -> f32 {
    let sa = 2.0 * PI * (1.0 - cos_max);
    return select(0.0, 1.0 / sa, sa > 0.0);
}

fn sample_ggx_vndf(ve: vec3<f32>, ax: f32, ay: f32, u: vec2<f32>) -> vec3<f32> {
    let vh = normalize(vec3<f32>(ax * ve.x, ay * ve.y, ve.z));
    let lensq = vh.x * vh.x + vh.y * vh.y;
    var t1: vec3<f32>;
    if (lensq > 0.0) {
        t1 = vec3<f32>(-vh.y, vh.x, 0.0) * inverseSqrt(lensq);
    } else {
        t1 = vec3<f32>(1.0, 0.0, 0.0);
    }
    let t2 = cross(vh, t1);
    let r = sqrt(u.x);
    let phi = 2.0 * PI * u.y;
    let p1 = r * cos(phi);
    var p2 = r * sin(phi);
    let s = 0.5 * (1.0 + vh.z);
    p2 = (1.0 - s) * sqrt(max(0.0, 1.0 - p1 * p1)) + s * p2;
    let nh = t1 * p1 + t2 * p2 + vh * sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2));
    return normalize(vec3<f32>(ax * nh.x, ay * nh.y, max(nh.z, 1e-6)));
}

fn power_heuristic(pdf_f: f32, pdf_g: f32) -> f32 {
    let f = pdf_f * pdf_f;
    let g = pdf_g * pdf_g;
    let d = f + g;
    return select(0.0, f / d, d > 0.0);
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// ---------------------------------------------------------------- atlas

// A map slot: xy = atlas origin in texels, zw = size in texels. `z <= 0` means
// the material has no map in that slot.
fn wrap_axis(v: f32, mode: u32) -> f32 {
    if (mode == 1u) {
        return fract(v);
    }
    if (mode == 2u) {
        return 1.0 - abs(fract(v * 0.5) * 2.0 - 1.0);
    }
    return clamp(v, 0.0, 0.999999);
}

// `mode` packs the S axis in bits 0-1 and the T axis in bits 2-3.
fn wrap_uv(uv: vec2<f32>, mode: u32) -> vec2<f32> {
    return vec2<f32>(wrap_axis(uv.x, mode & 3u), wrap_axis(uv.y, (mode >> 2u) & 3u));
}

// Bilinear fetch from a sub-rectangle of the atlas. Filtering is done here
// rather than by a sampler because a sampler would blend across the rectangle's
// edge into whatever texture was packed next to it.
fn atlas_sample(rect: vec4<f32>, uv_in: vec2<f32>, wrap: u32) -> vec4<f32> {
    let size = rect.zw;
    if (size.x <= 0.0 || size.y <= 0.0) {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }
    let uv = wrap_uv(uv_in, wrap);
    let f = uv * size - vec2<f32>(0.5, 0.5);
    let i0 = floor(f);
    let t = f - i0;
    var c = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    for (var dy = 0; dy < 2; dy = dy + 1) {
        for (var dx = 0; dx < 2; dx = dx + 1) {
            var p = i0 + vec2<f32>(f32(dx), f32(dy));
            // Wrap inside the sub-rectangle, so a repeating texture repeats
            // itself and not the atlas.
            // Wrap inside the sub-rectangle per axis, so a repeating texture
            // repeats itself and not the atlas.
            if ((wrap & 3u) == 1u) {
                p.x = p.x - floor(p.x / size.x) * size.x;
            } else {
                p.x = clamp(p.x, 0.0, size.x - 1.0);
            }
            if (((wrap >> 2u) & 3u) == 1u) {
                p.y = p.y - floor(p.y / size.y) * size.y;
            } else {
                p.y = clamp(p.y, 0.0, size.y - 1.0);
            }
            let texel = textureLoad(atlas, vec2<i32>(rect.xy + p), 0);
            let w = (select(1.0 - t.x, t.x, dx == 1)) * (select(1.0 - t.y, t.y, dy == 1));
            c = c + texel * w;
        }
    }
    return c;
}

// One map slot out of a material: 9 floats — rect, UV offset/repeat, rotation.
fn map_transform_uv(base: u32, slot: u32, uv_in: vec2<f32>) -> vec2<f32> {
    let o = base + 36u + slot * 9u;
    var uv = uv_in;
    let rot = data[o + 8u];
    if (rot != 0.0) {
        let c = cos(rot);
        let s = sin(rot);
        let du = uv.x - 0.5;
        let dv = uv.y - 0.5;
        uv = vec2<f32>(du * c - dv * s + 0.5, du * s + dv * c + 0.5);
    }
    let offset = vec2<f32>(data[o + 4u], data[o + 5u]);
    let repeat = vec2<f32>(data[o + 6u], data[o + 7u]);
    return uv * repeat + offset;
}

fn map_sample(base: u32, slot: u32, uv: vec2<f32>, wrap: u32) -> vec4<f32> {
    let o = base + 36u + slot * 9u;
    let rect = vec4<f32>(data[o], data[o + 1u], data[o + 2u], data[o + 3u]);
    if (rect.z <= 0.0) {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }
    return atlas_sample(rect, map_transform_uv(base, slot, uv), wrap);
}

fn map_present(base: u32, slot: u32) -> bool {
    return data[base + 36u + slot * 9u + 2u] > 0.0;
}

/// Two bits for the S axis then two for T, as `wrap_uv` expects.
fn map_wrap(base: u32, slot: u32) -> u32 {
    let bits = u32(data[base + 83u]);
    let s = (bits >> (slot * 2u)) & 3u;
    let t = (bits >> (10u + slot * 2u)) & 3u;
    return s | (t << 2u);
}

// ---------------------------------------------------------------- environment

// Direction to a cube face and the UV within it.
//
// The standard cube-map convention — the inverse of what
// `PmremGenerator::from_equirect_f32` writes with, and what the hardware
// sampler uses for a plain `CubeTexture`. It is deliberately *not* three.js's
// CubeUV convention, which belongs to the PMREM atlas and is reached through a
// blit that flips rows and swaps ±X; reading `faces` as if it were CubeUV
// rotates every face by half a turn.
//
// Returns (face, u, v). `face` is -1 for a degenerate direction.
fn cube_lookup(d: vec3<f32>) -> vec3<f32> {
    let a = abs(d);
    var face = 0;
    var sc = 0.0;
    var tc = 0.0;
    var ma = 0.0;
    if (a.x >= a.y && a.x >= a.z) {
        ma = a.x;
        if (d.x > 0.0) { face = 0; sc = -d.z; tc = -d.y; }
        else { face = 1; sc = d.z; tc = -d.y; }
    } else if (a.y >= a.z) {
        ma = a.y;
        if (d.y > 0.0) { face = 2; sc = d.x; tc = d.z; }
        else { face = 3; sc = d.x; tc = -d.z; }
    } else {
        ma = a.z;
        if (d.z > 0.0) { face = 4; sc = d.x; tc = -d.y; }
        else { face = 5; sc = -d.x; tc = -d.y; }
    }
    if (ma <= 0.0) {
        return vec3<f32>(-1.0, 0.0, 0.0);
    }
    return vec3<f32>(f32(face), (sc / ma + 1.0) * 0.5, (tc / ma + 1.0) * 0.5);
}

fn env_radiance(dir: vec3<f32>) -> vec3<f32> {
    if ((uni.flags.w & 1u) == 0u) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let l = cube_lookup(dir);
    if (l.x < 0.0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    return atlas_sample(uni.env_rect[u32(l.x)], vec2<f32>(l.y, l.z), 0u).rgb * uni.ambient.w;
}


// ---------------------------------------------------- environment sampling

// How often the environment's own distribution is used instead of a
// cosine-weighted guess. See `World::sample_direction` for why it is a mixture
// and not one or the other.
const ENV_STRATEGY: f32 = 0.5;

fn env_has_distribution() -> bool {
    return uni.env_dims.z != 0u;
}

// Last index in `[0, n)` whose CDF entry is still at or below `u`.
fn cdf_find(base: u32, n: u32, u: f32) -> u32 {
    var lo = 0u;
    var hi = n;
    loop {
        if (lo + 1u >= hi) {
            break;
        }
        let mid = (lo + hi) >> 1u;
        if (data[base + mid] <= u) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return min(lo, n - 1u);
}

// Position within the interval the search landed in, as a continuous
// coordinate in `[0, 1)`.
fn cdf_continuous(base: u32, i: u32, n: u32, u: f32) -> f32 {
    let lo = data[base + i];
    let hi = data[base + i + 1u];
    var du = 0.5;
    if (hi > lo) {
        du = (u - lo) / (hi - lo);
    }
    return clamp((f32(i) + du) / f32(n), 0.0, 0.999999);
}

fn env_direction(u: f32, v: f32) -> vec3<f32> {
    let theta = v * PI;
    let phi = u * 2.0 * PI;
    let st = sin(theta);
    return vec3<f32>(st * cos(phi), cos(theta), st * sin(phi));
}

// Inverse of `env_direction`. Theta comes from atan2 rather than acos, which
// stays conditioned at the poles where acos does not — and theta enters the
// density through sin(theta), so a few percent of error there is a few percent
// of error in every MIS weight near the pole.
fn env_uv(d: vec3<f32>) -> vec2<f32> {
    let v = atan2(sqrt(d.x * d.x + d.z * d.z), d.y) / PI;
    var u = atan2(d.z, d.x) / (2.0 * PI);
    if (u < 0.0) {
        u = u + 1.0;
    }
    return vec2<f32>(clamp(u, 0.0, 0.999999), clamp(v, 0.0, 0.999999));
}

fn env_pdf_at(iu: u32, iv: u32, v: f32) -> f32 {
    let st = sin(v * PI);
    let total = uni.params.z;
    if (st <= 0.0 || total <= 0.0) {
        return 0.0;
    }
    let f = data[uni.env_offsets.x + iv * uni.env_dims.x + iu];
    // Density over the unit square, then the Jacobian of (u, v) -> direction.
    return (f / total) / (2.0 * PI * PI * st);
}

fn env_pdf(d: vec3<f32>) -> f32 {
    if (!env_has_distribution()) {
        return 0.0;
    }
    let uv = env_uv(d);
    let iu = min(u32(uv.x * f32(uni.env_dims.x)), uni.env_dims.x - 1u);
    let iv = min(u32(uv.y * f32(uni.env_dims.y)), uni.env_dims.y - 1u);
    return env_pdf_at(iu, iv, uv.y);
}

// Draw a direction from the environment's brightness. Two binary searches:
// the marginal over rows, then the conditional within the row it picked.
fn env_sample_dir(u1: f32, u2: f32) -> vec3<f32> {
    let w = uni.env_dims.x;
    let h = uni.env_dims.y;
    let iv = cdf_find(uni.env_offsets.z, h, u2);
    let v = cdf_continuous(uni.env_offsets.z, iv, h, u2);
    let row = uni.env_offsets.y + iv * (w + 1u);
    let iu = cdf_find(row, w, u1);
    let u = cdf_continuous(row, iu, w, u1);
    return env_direction(u, v);
}

fn world_lighting(dir: vec3<f32>) -> vec3<f32> {
    var l = uni.ambient.rgb;
    let n_hemi = uni.flags.z;
    for (var i = 0u; i < n_hemi; i = i + 1u) {
        let sky = uni.hemi[i * 3u].rgb;
        let ground = uni.hemi[i * 3u + 1u].rgb;
        let up = uni.hemi[i * 3u + 2u].rgb;
        let t = clamp(0.5 * dot(dir, up) + 0.5, 0.0, 1.0);
        l = l + mix(ground, sky, t);
    }
    return l + env_radiance(dir);
}

fn world_is_black() -> bool {
    return (uni.flags.w & 1u) == 0u
        && uni.flags.z == 0u
        && uni.ambient.r <= 0.0
        && uni.ambient.g <= 0.0
        && uni.ambient.b <= 0.0;
}

fn world_camera(dir: vec3<f32>) -> vec4<f32> {
    let mode = uni.limits.w;
    if (mode == 2u) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    if (mode == 1u && (uni.flags.w & 1u) != 0u) {
        return vec4<f32>(env_radiance(dir), uni.background.a);
    }
    return vec4<f32>(uni.background.rgb, uni.background.a);
}

/// The density world sampling draws from — the mixture, not whichever half
/// produced a given direction.
fn world_pdf(n: vec3<f32>, dir: vec3<f32>) -> f32 {
    let cosine = max(0.0, dot(n, dir) / PI);
    if (!env_has_distribution()) {
        return cosine;
    }
    return ENV_STRATEGY * env_pdf(dir) + (1.0 - ENV_STRATEGY) * cosine;
}


// ---------------------------------------------------------------- geometry

fn tri_vertex(tri: u32, i: u32) -> vec3<f32> {
    let b = uni.data_offsets.x + tri * TRI_STRIDE + i * 3u;
    return vec3<f32>(data[b], data[b + 1u], data[b + 2u]);
}

fn tri_normal_at(tri: u32, i: u32) -> vec3<f32> {
    let b = uni.data_offsets.x + tri * TRI_STRIDE + 9u + i * 3u;
    return vec3<f32>(data[b], data[b + 1u], data[b + 2u]);
}

fn tri_uv_at(tri: u32, i: u32) -> vec2<f32> {
    let b = uni.data_offsets.x + tri * TRI_STRIDE + 18u + i * 2u;
    return vec2<f32>(data[b], data[b + 1u]);
}

fn tri_color_at(tri: u32, i: u32) -> vec3<f32> {
    let b = uni.data_offsets.x + tri * TRI_STRIDE + 24u + i * 3u;
    return vec3<f32>(data[b], data[b + 1u], data[b + 2u]);
}

fn tri_material(tri: u32) -> u32 {
    return idx[uni.idx_offsets.y + tri];
}

struct Hit {
    t: f32,
    tri: u32,
    u: f32,
    v: f32,
    hit: bool,
};

// Möller-Trumbore, two-sided: transmission and interior shading both need the
// back faces that a culled test would discard.
fn tri_intersect(tri: u32, o: vec3<f32>, d: vec3<f32>, t_min: f32, t_max: f32) -> vec3<f32> {
    let p0 = tri_vertex(tri, 0u);
    let e1 = tri_vertex(tri, 1u) - p0;
    let e2 = tri_vertex(tri, 2u) - p0;
    let pv = cross(d, e2);
    let det = dot(e1, pv);
    if (abs(det) < 1e-12) {
        return vec3<f32>(-1.0, 0.0, 0.0);
    }
    let inv = 1.0 / det;
    let tv = o - p0;
    let u = dot(tv, pv) * inv;
    if (u < -1e-7 || u > 1.0 + 1e-7) {
        return vec3<f32>(-1.0, 0.0, 0.0);
    }
    let qv = cross(tv, e1);
    let v = dot(d, qv) * inv;
    if (v < -1e-7 || u + v > 1.0 + 1e-7) {
        return vec3<f32>(-1.0, 0.0, 0.0);
    }
    let t = dot(e2, qv) * inv;
    if (t < t_min || t >= t_max) {
        return vec3<f32>(-1.0, 0.0, 0.0);
    }
    return vec3<f32>(t, u, v);
}

fn node_min(i: u32) -> vec3<f32> { return nodes[i * 2u].xyz; }
fn node_max(i: u32) -> vec3<f32> { return nodes[i * 2u + 1u].xyz; }
fn node_first(i: u32) -> u32 { return bitcast<u32>(nodes[i * 2u].w); }
fn node_count(i: u32) -> u32 { return bitcast<u32>(nodes[i * 2u + 1u].w); }

fn slab(i: u32, o: vec3<f32>, inv_d: vec3<f32>, t_min: f32, t_max: f32) -> f32 {
    let t0 = (node_min(i) - o) * inv_d;
    let t1 = (node_max(i) - o) * inv_d;
    let lo = min(t0, t1);
    let hi = max(t0, t1);
    let tmin = max(max(lo.x, lo.y), max(lo.z, t_min));
    let tmax = min(min(hi.x, hi.y), min(hi.z, t_max));
    return select(INF, tmin, tmin <= tmax);
}

fn safe_inv(d: vec3<f32>) -> vec3<f32> {
    // Floor at the smallest magnitude that still yields a finite reciprocal, so
    // the slab test never forms 0 * inf (which is NaN and would silently drop
    // every axis-aligned ray).
    let s = sign(d) + vec3<f32>(select(0.0, 1.0, d.x == 0.0), select(0.0, 1.0, d.y == 0.0), select(0.0, 1.0, d.z == 0.0));
    let safe = select(d, s * 1e-30, abs(d) < vec3<f32>(1e-30, 1e-30, 1e-30));
    return vec3<f32>(1.0, 1.0, 1.0) / safe;
}

fn trace_closest(o: vec3<f32>, d: vec3<f32>, t_min: f32, t_max: f32) -> Hit {
    var best: Hit;
    best.hit = false;
    best.t = t_max;
    best.tri = 0u;
    best.u = 0.0;
    best.v = 0.0;
    if (uni.counts.x == 0u) {
        return best;
    }
    let inv_d = safe_inv(d);
    var stack: array<u32, 64>;
    var sp = 0u;
    var node = 0u;
    var far_t = t_max;

    loop {
        let count = node_count(node);
        if (count > 0u) {
            let first = node_first(node);
            for (var k = 0u; k < count; k = k + 1u) {
                let tri = idx[uni.idx_offsets.x + first + k];
                let r = tri_intersect(tri, o, d, t_min, far_t);
                if (r.x >= 0.0) {
                    far_t = r.x;
                    best.t = r.x;
                    best.tri = tri;
                    best.u = r.y;
                    best.v = r.z;
                    best.hit = true;
                }
            }
        } else {
            let l = node_first(node);
            let r = l + 1u;
            let dl = slab(l, o, inv_d, t_min, far_t);
            let dr = slab(r, o, inv_d, t_min, far_t);
            if (min(dl, dr) < INF) {
                // Nearer child first: whatever it finds shrinks `far_t`, which
                // usually rejects the sibling outright.
                var near = l;
                var far = r;
                var d_far = dr;
                if (dr < dl) {
                    near = r;
                    far = l;
                    d_far = dl;
                }
                if (d_far < INF && sp < MAX_STACK) {
                    stack[sp] = far;
                    sp = sp + 1u;
                }
                node = near;
                continue;
            }
        }
        if (sp == 0u) {
            break;
        }
        sp = sp - 1u;
        node = stack[sp];
    }
    return best;
}

fn trace_any(o: vec3<f32>, d: vec3<f32>, t_min: f32, t_max: f32) -> bool {
    if (uni.counts.x == 0u) {
        return false;
    }
    let inv_d = safe_inv(d);
    var stack: array<u32, 64>;
    var sp = 0u;
    var node = 0u;

    loop {
        let count = node_count(node);
        if (count > 0u) {
            let first = node_first(node);
            for (var k = 0u; k < count; k = k + 1u) {
                let tri = idx[uni.idx_offsets.x + first + k];
                if (tri_intersect(tri, o, d, t_min, t_max).x >= 0.0) {
                    return true;
                }
            }
        } else {
            let l = node_first(node);
            let r = l + 1u;
            let dl = slab(l, o, inv_d, t_min, t_max);
            let dr = slab(r, o, inv_d, t_min, t_max);
            if (min(dl, dr) < INF) {
                var near = l;
                var far = r;
                var d_far = dr;
                if (dr < dl) {
                    near = r;
                    far = l;
                    d_far = dl;
                }
                if (d_far < INF && sp < MAX_STACK) {
                    stack[sp] = far;
                    sp = sp + 1u;
                }
                node = near;
                continue;
            }
        }
        if (sp == 0u) {
            return false;
        }
        sp = sp - 1u;
        node = stack[sp];
    }
    return false;
}

// ---------------------------------------------------------------- materials

struct Surface {
    base_color: vec3<f32>,
    emission: vec3<f32>,
    opacity: f32,
    alpha_test: f32,
    roughness: f32,
    metallic: f32,
    transmission: f32,
    ior: f32,
    clearcoat: f32,
    clearcoat_roughness: f32,
    specular_tint: vec3<f32>,
    anisotropy: f32,
    anisotropy_rotation: f32,
    sheen: f32,
    sheen_color: vec3<f32>,
    sheen_roughness: f32,
    iridescence: f32,
    iridescence_ior: f32,
    iridescence_thickness: f32,
    dispersion: f32,
    subsurface: f32,
    subsurface_radius: vec3<f32>,
    attenuation_color: vec3<f32>,
    attenuation_distance: f32,
    unlit: bool,
};

fn load_surface(mat: u32, uv: vec2<f32>) -> Surface {
    let b = uni.data_offsets.z + mat * MAT_STRIDE;
    var s: Surface;
    s.base_color = vec3<f32>(data[b], data[b + 1u], data[b + 2u]);
    s.opacity = data[b + 3u];
    s.emission = vec3<f32>(data[b + 4u], data[b + 5u], data[b + 6u]);
    s.alpha_test = data[b + 7u];
    s.roughness = data[b + 8u];
    s.metallic = data[b + 9u];
    s.transmission = data[b + 10u];
    s.ior = data[b + 11u];
    s.attenuation_color = vec3<f32>(data[b + 12u], data[b + 13u], data[b + 14u]);
    s.attenuation_distance = data[b + 15u];
    s.clearcoat = data[b + 16u];
    s.clearcoat_roughness = data[b + 17u];
    s.specular_tint = vec3<f32>(data[b + 18u], data[b + 19u], data[b + 20u]);
    s.anisotropy = data[b + 21u];
    s.anisotropy_rotation = data[b + 22u];
    s.unlit = data[b + 23u] > 0.5;
    s.sheen = data[b + 24u];
    s.sheen_color = vec3<f32>(data[b + 25u], data[b + 26u], data[b + 27u]);
    s.sheen_roughness = data[b + 28u];
    s.iridescence = data[b + 29u];
    s.iridescence_ior = data[b + 30u];
    s.iridescence_thickness = data[b + 31u];
    s.dispersion = data[b + 32u];
    s.subsurface = data[b + 33u];
    s.subsurface_radius = vec3<f32>(data[b + 34u], data[b + 35u], data[b + 84u]);

    if (map_present(b, 0u)) {
        let c = map_sample(b, 0u, uv, map_wrap(b, 0u));
        s.base_color = s.base_color * c.rgb;
        s.opacity = s.opacity * c.a;
    }
    if (map_present(b, 1u)) {
        // three.js reads roughness from green and metalness from blue, so one
        // packed ORM map drives both.
        s.roughness = s.roughness * map_sample(b, 1u, uv, map_wrap(b, 1u)).g;
    }
    if (map_present(b, 2u)) {
        s.metallic = s.metallic * map_sample(b, 2u, uv, map_wrap(b, 2u)).b;
    }
    if (map_present(b, 3u)) {
        s.emission = s.emission * map_sample(b, 3u, uv, map_wrap(b, 3u)).rgb;
    }
    return s;
}

fn material_emission_const(mat: u32) -> vec3<f32> {
    let b = uni.data_offsets.z + mat * MAT_STRIDE;
    return vec3<f32>(data[b + 4u], data[b + 5u], data[b + 6u]);
}


// Tangent-space normal mapping. The frame comes from the triangle's own UV
// gradient rather than a stored tangent attribute, so a normal map works on any
// geometry that has UVs — which the `*Geometry` constructors produce and
// tangents are not.
fn perturb_normal(tri: u32, base: u32, ns: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
    let uv0 = tri_uv_at(tri, 0u);
    let duv1 = tri_uv_at(tri, 1u) - uv0;
    let duv2 = tri_uv_at(tri, 2u) - uv0;
    let det = duv1.x * duv2.y - duv2.x * duv1.y;
    if (abs(det) < 1e-12) {
        return ns;
    }
    let p0 = tri_vertex(tri, 0u);
    let e1 = tri_vertex(tri, 1u) - p0;
    let e2 = tri_vertex(tri, 2u) - p0;
    var t = (e1 * duv2.y - e2 * duv1.y) * (1.0 / det);
    // Gram-Schmidt against the shading normal, which interpolation has tilted.
    t = t - ns * dot(ns, t);
    if (dot(t, t) < 1e-16) {
        return ns;
    }
    t = normalize(t);
    let b = cross(ns, t);
    let s = map_sample(base, 4u, uv, map_wrap(base, 4u));
    let scale = vec2<f32>(data[base + 81u], data[base + 82u]);
    let n = vec3<f32>((s.r * 2.0 - 1.0) * scale.x, (s.g * 2.0 - 1.0) * scale.y, s.b * 2.0 - 1.0);
    let world = t * n.x + b * n.y + ns * n.z;
    if (dot(world, world) < 1e-16) {
        return ns;
    }
    return normalize(world);
}

// How much of the denoiser's guide a surface writes, what it writes, and what
// it defers to whatever it reflects or transmits. Returns
// (weight, diffuse_albedo, deferred_tint).
//
// A mirror's own colour and normal say nothing about the image in it, and a
// glass ball's say nothing about the room behind it, so the guides follow the
// path through specular and transmissive surfaces — carrying their tint —
// until they reach something rough enough to describe. Cycles' scheme, and the
// thresholds are its: a lobe stops counting as specular past roughness ~0.15.
fn guide_split(s: Surface) -> mat3x3<f32> {
    let rough = smoothstep(0.0, 0.15, clamp(s.roughness, 0.0, 1.0));
    let metallic = clamp(s.metallic, 0.0, 1.0);
    let transmission = clamp(s.transmission, 0.0, 1.0);

    let w_diffuse = (1.0 - metallic) * (1.0 - transmission);
    let w_sharp = metallic + (1.0 - metallic) * transmission;
    let total = max(w_diffuse + w_sharp, 1e-6);

    let describable = (w_diffuse + w_sharp * rough) / total;
    let weight = smoothstep(0.0, 0.5, describable);
    return mat3x3<f32>(
        vec3<f32>(weight, 0.0, 0.0),
        s.base_color * describable,
        s.base_color * ((w_sharp / total) * (1.0 - weight)),
    );
}

// Compress an unbounded radiance into [0, 1), preserving order. The albedo
// guide is a reflectance-like quantity; an HDRI's sun at 12000 would make every
// neighbour look infinitely different and the filter would do nothing at all.
fn reinhard(v: vec3<f32>) -> vec3<f32> {
    return v / (1.0 + max(v, vec3<f32>(0.0, 0.0, 0.0)));
}

fn average3(v: vec3<f32>) -> f32 {
    return (v.x + v.y + v.z) / 3.0;
}

// ---------------------------------------------------------------- BSDF

struct Bsdf {
    frame: Onb,
    geom: vec3<f32>,
    s: Surface,
    ax: f32,
    ay: f32,
    coat_alpha: f32,
    f0: vec3<f32>,
    // The transmissive fraction is handled by the dielectric lobe, which
    // already carries its own Fresnel reflection; without this the reflection
    // off a glass surface would be counted twice.
    spec_weight: f32,
    p_diffuse: f32,
    p_specular: f32,
    p_transmission: f32,
    p_coat: f32,
    p_sheen: f32,
    sheen_alpha: f32,
};

fn ggx_d(h: vec3<f32>, ax: f32, ay: f32) -> f32 {
    let axc = max(ax, MIN_ALPHA);
    let ayc = max(ay, MIN_ALPHA);
    let hx = h.x / axc;
    let hy = h.y / ayc;
    let dd = hx * hx + hy * hy + h.z * h.z;
    if (dd <= 0.0) {
        return 0.0;
    }
    return 1.0 / (PI * axc * ayc * dd * dd);
}

fn smith_lambda(v: vec3<f32>, ax: f32, ay: f32) -> f32 {
    let cos2 = v.z * v.z;
    if (cos2 <= 0.0) {
        return 0.0;
    }
    let axc = max(ax, MIN_ALPHA);
    let ayc = max(ay, MIN_ALPHA);
    let a2 = (v.x * axc) * (v.x * axc) + (v.y * ayc) * (v.y * ayc);
    let tan2 = a2 / cos2;
    return 0.5 * (sqrt(1.0 + tan2) - 1.0);
}

fn smith_g1(v: vec3<f32>, ax: f32, ay: f32) -> f32 {
    return 1.0 / (1.0 + smith_lambda(v, ax, ay));
}

fn smith_g2(wo: vec3<f32>, wi: vec3<f32>, ax: f32, ay: f32) -> f32 {
    return 1.0 / (1.0 + smith_lambda(wo, ax, ay) + smith_lambda(wi, ax, ay));
}

fn vndf_pdf(wo: vec3<f32>, h: vec3<f32>, ax: f32, ay: f32) -> f32 {
    let c = abs(wo.z);
    if (c <= 0.0) {
        return 0.0;
    }
    return smith_g1(wo, ax, ay) * ggx_d(h, ax, ay) * abs(dot(wo, h)) / c;
}

fn vndf_reflect_pdf(wo: vec3<f32>, h: vec3<f32>, ax: f32, ay: f32) -> f32 {
    let d = abs(dot(wo, h));
    if (d <= 0.0) {
        return 0.0;
    }
    return vndf_pdf(wo, h, ax, ay) / (4.0 * d);
}

fn schlick(f0: vec3<f32>, cos_theta: f32) -> vec3<f32> {
    let m = clamp(1.0 - cos_theta, 0.0, 1.0);
    let m5 = m * m * m * m * m;
    return f0 + (vec3<f32>(1.0, 1.0, 1.0) - f0) * m5;
}

fn schlick1(f0: f32, cos_theta: f32) -> f32 {
    let m = clamp(1.0 - cos_theta, 0.0, 1.0);
    let m5 = m * m * m * m * m;
    return f0 + (1.0 - f0) * m5;
}

// Exact unpolarised Fresnel. Schlick is close enough for the reflection lobe
// but not for transmission, which weighs by 1 - F right where Schlick's error
// decides whether total internal reflection happens at all.
fn fresnel_dielectric(cos_i_in: f32, eta_in: f32) -> f32 {
    var cos_i = clamp(cos_i_in, -1.0, 1.0);
    var eta = eta_in;
    if (cos_i < 0.0) {
        eta = 1.0 / eta;
        cos_i = -cos_i;
    }
    let sin2_t = max(0.0, 1.0 - cos_i * cos_i) / (eta * eta);
    if (sin2_t >= 1.0) {
        return 1.0;
    }
    let cos_t = sqrt(max(0.0, 1.0 - sin2_t));
    let rp = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    let rs = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    return 0.5 * (rp * rp + rs * rs);
}

fn lut_at(m: u32, a: u32) -> f32 {
    let i = a * 32u + m;
    let v = uni.albedo_lut[i / 4u];
    let c = i % 4u;
    if (c == 0u) { return v.x; }
    if (c == 1u) { return v.y; }
    if (c == 2u) { return v.z; }
    return v.w;
}

fn ggx_albedo(cos_theta: f32, alpha: f32) -> f32 {
    let mu = clamp(cos_theta, 0.0, 1.0) * 31.0;
    let al = sqrt(clamp(alpha, 0.0, 1.0)) * 31.0;
    let m0 = u32(floor(mu));
    let a0 = u32(floor(al));
    let m1 = min(m0 + 1u, 31u);
    let a1 = min(a0 + 1u, 31u);
    let tm = mu - floor(mu);
    let ta = al - floor(al);
    let e00 = lut_at(m0, a0);
    let e10 = lut_at(m1, a0);
    let e01 = lut_at(m0, a1);
    let e11 = lut_at(m1, a1);
    let e0 = mix(e00, e10, tm);
    let e1 = mix(e01, e11, tm);
    return clamp(mix(e0, e1, ta), 1e-3, 1.0);
}

fn multiscatter_gain(b: Bsdf, cos_o: f32) -> vec3<f32> {
    let alpha = sqrt(b.ax * b.ay);
    let e = ggx_albedo(cos_o, alpha);
    if (e >= 0.999) {
        return vec3<f32>(1.0, 1.0, 1.0);
    }
    let gain = (1.0 - e) / max(e, 1e-3);
    return vec3<f32>(1.0, 1.0, 1.0) + b.f0 * gain;
}

fn d_charlie(n_dot_h: f32, alpha: f32) -> f32 {
    let a = max(alpha, 0.0016);
    let inv_a = 1.0 / a;
    let cos2h = n_dot_h * n_dot_h;
    let sin2h = max(1.0 - cos2h, 0.0078125);
    return (2.0 + inv_a) * pow(sin2h, inv_a * 0.5) / (2.0 * PI);
}

fn v_neubelt(n_dot_v: f32, n_dot_l: f32) -> f32 {
    return 1.0 / max(4.0 * (n_dot_l + n_dot_v - n_dot_l * n_dot_v), 1e-5);
}

fn ior_to_fresnel0_f(transmitted: f32, incident: f32) -> f32 {
    let t = (transmitted - incident) / (transmitted + incident);
    return t * t;
}

fn fresnel0_to_ior(f0: vec3<f32>) -> vec3<f32> {
    let s = sqrt(clamp(f0, vec3<f32>(0.0), vec3<f32>(0.9999)));
    return (vec3<f32>(1.0) + s) / (vec3<f32>(1.0) - s);
}

fn f_schlick_f90_v(f0: vec3<f32>, f90: vec3<f32>, cos_theta: f32) -> vec3<f32> {
    let w = pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
    return f0 + (f90 - f0) * w;
}

fn f_schlick_f90_f(f0: f32, f90: f32, cos_theta: f32) -> f32 {
    let w = pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
    return f0 + (f90 - f0) * w;
}

fn eval_sensitivity(opd: f32, shift: vec3<f32>) -> vec3<f32> {
    let phase = 2.0 * PI * opd * 1.0e-9;
    let val = vec3<f32>(5.4856e-13, 4.4201e-13, 5.2481e-13);
    let pos = vec3<f32>(1.6810e+06, 1.7953e+06, 2.2084e+06);
    let vr  = vec3<f32>(4.3278e+09, 9.3046e+09, 6.6121e+09);
    var xyz = val * sqrt(2.0 * PI * vr) * cos(pos * phase + shift) * exp(-(phase * phase) * vr);
    let x_extra = 9.7470e-14 * sqrt(2.0 * PI * 4.5282e+09)
        * cos(2.2399e+06 * phase + shift.x) * exp(-4.5282e+09 * phase * phase);
    xyz.x = xyz.x + x_extra;
    xyz = xyz / 1.0685e-7;
    return vec3<f32>(
        3.2404542 * xyz.x - 0.9692660 * xyz.y + 0.0556434 * xyz.z,
        -1.5371385 * xyz.x + 1.8760108 * xyz.y - 0.2040259 * xyz.z,
        -0.4985314 * xyz.x + 0.0415560 * xyz.y + 1.0572252 * xyz.z,
    );
}

fn iridescence_fresnel(
    outside_ior: f32, film_ior_in: f32, base_f0: vec3<f32>,
    thickness_nm: f32, cos_theta1: f32,
) -> vec3<f32> {
    let film_ior = mix(outside_ior, film_ior_in, smoothstep(0.0, 0.03, thickness_nm));
    let sin_theta2_sq = pow(outside_ior / film_ior, 2.0) * (1.0 - cos_theta1 * cos_theta1);
    let cos_theta2_sq = 1.0 - sin_theta2_sq;
    if (cos_theta2_sq < 0.0) {
        return vec3<f32>(1.0, 1.0, 1.0);
    }
    let cos_theta2 = sqrt(max(cos_theta2_sq, 0.0));
    let r0 = ior_to_fresnel0_f(film_ior, outside_ior);
    let r12 = f_schlick_f90_f(r0, 1.0, cos_theta1);
    let t121 = 1.0 - r12;
    var phi12 = 0.0;
    if (film_ior < outside_ior) { phi12 = PI; }
    let phi21 = PI - phi12;
    let base_ior = fresnel0_to_ior(clamp(base_f0, vec3<f32>(0.0), vec3<f32>(0.9999)));
    let r1 = (base_ior - vec3<f32>(film_ior)) / (base_ior + vec3<f32>(film_ior));
    let r1sq = r1 * r1;
    let r23 = f_schlick_f90_v(r1sq, vec3<f32>(1.0), cos_theta2);
    var phi23 = vec3<f32>(0.0);
    if (base_ior.r < film_ior) { phi23.r = PI; }
    if (base_ior.g < film_ior) { phi23.g = PI; }
    if (base_ior.b < film_ior) { phi23.b = PI; }
    let opd = 2.0 * film_ior * thickness_nm * cos_theta2;
    let phi = vec3<f32>(phi21) + phi23;
    let r123 = clamp(vec3<f32>(r12) * r23, vec3<f32>(1e-5), vec3<f32>(0.9999));
    let r123_sqrt = sqrt(r123);
    let rs = (t121 * t121) * r23 / max(vec3<f32>(1.0) - r123, vec3<f32>(1e-5));
    let c0 = vec3<f32>(r12) + rs;
    var i_out = c0;
    var cm = rs - vec3<f32>(t121);
    for (var m: i32 = 1; m <= 2; m = m + 1) {
        cm = cm * r123_sqrt;
        let sm = 2.0 * eval_sensitivity(f32(m) * opd, f32(m) * phi);
        i_out = i_out + cm * sm;
    }
    return max(i_out, vec3<f32>(0.0));
}

fn specular_fresnel(b: Bsdf, wo: vec3<f32>, h: vec3<f32>) -> vec3<f32> {
    let cos_theta = abs(dot(wo, h));
    let base = schlick(b.f0, cos_theta);
    let ir = clamp(b.s.iridescence, 0.0, 1.0);
    if (ir <= 0.0) {
        return base;
    }
    let film = iridescence_fresnel(
        1.0, b.s.iridescence_ior, b.f0, b.s.iridescence_thickness, abs(wo.z),
    );
    return mix(base, film, ir);
}

fn eval_transmission_channel(b: Bsdf, wo: vec3<f32>, wi: vec3<f32>, eta: f32, reflecting: bool, channel: u32) -> vec2<f32> {
    let cos_o = wo.z;
    let cos_i = wi.z;
    var etap = 1.0;
    if (!reflecting) {
        etap = select(1.0 / eta, eta, cos_o > 0.0);
    }
    var wm = wi * etap + wo;
    if (dot(wm, wm) < 1e-12) {
        return vec2<f32>(0.0, 0.0);
    }
    wm = normalize(wm);
    if (wm.z < 0.0) { wm = -wm; }
    if (dot(wm, wi) * cos_i < 0.0 || dot(wm, wo) * cos_o < 0.0) {
        return vec2<f32>(0.0, 0.0);
    }
    let fr = fresnel_dielectric(dot(wo, wm), eta);
    let d = ggx_d(wm, b.ax, b.ay);
    let g = smith_g2(wo, wi, b.ax, b.ay);
    let pr = fr;
    let pt = 1.0 - fr;
    let weight = (1.0 - b.s.metallic) * b.s.transmission;
    if (reflecting) {
        let v = d * fr * g / max(abs(4.0 * cos_o * cos_i), 1e-9);
        let pdf = vndf_reflect_pdf(wo, wm, b.ax, b.ay) * pr / max(pr + pt, 1e-9);
        return vec2<f32>(v * weight, pdf);
    }
    let dd = dot(wi, wm) + dot(wo, wm) / etap;
    let denom = dd * dd;
    if (denom < 1e-12) {
        return vec2<f32>(0.0, 0.0);
    }
    let v = d * (1.0 - fr) * g * abs(dot(wi, wm) * dot(wo, wm) / (cos_i * cos_o * denom)) / (etap * etap);
    let dwm_dwi = abs(dot(wi, wm)) / denom;
    let pdf = vndf_pdf(wo, wm, b.ax, b.ay) * dwm_dwi * pt / max(pr + pt, 1e-9);
    var tint = b.s.base_color.r;
    if (channel == 1u) { tint = b.s.base_color.g; }
    if (channel == 2u) { tint = b.s.base_color.b; }
    return vec2<f32>(tint * (v * weight), pdf);
}

fn bsdf_make(s: Surface, n: vec3<f32>, geom: vec3<f32>) -> Bsdf {
    var b: Bsdf;
    let rough = clamp(s.roughness, 0.0, 1.0);
    let alpha = rough * rough;
    let aniso = clamp(s.anisotropy, -0.9, 0.9);
    let stretch = max(sqrt(1.0 - aniso * aniso), 1e-3);
    if (abs(aniso) < 1e-4) {
        b.ax = alpha;
        b.ay = alpha;
    } else if (aniso > 0.0) {
        b.ax = alpha / stretch;
        b.ay = alpha * stretch;
    } else {
        b.ax = alpha * stretch;
        b.ay = alpha / stretch;
    }
    let r = (s.ior - 1.0) / (s.ior + 1.0);
    let dielectric_f0 = r * r;
    let metallic = clamp(s.metallic, 0.0, 1.0);
    let transmission = clamp(s.transmission, 0.0, 1.0);
    b.f0 = mix(vec3<f32>(dielectric_f0, dielectric_f0, dielectric_f0) * s.specular_tint, s.base_color, metallic);

    let w_diffuse = luminance(s.base_color) * (1.0 - metallic) * (1.0 - transmission);
    let w_transmission = (1.0 - metallic) * transmission;
    b.spec_weight = 1.0 - w_transmission;
    let w_specular = max(luminance(b.f0), 0.04) * b.spec_weight;
    let w_coat = clamp(s.clearcoat, 0.0, 1.0) * 0.25;
    let sheen = clamp(s.sheen, 0.0, 1.0);
    let w_sheen = luminance(s.sheen_color) * sheen * (1.0 - metallic);
    let total = max(w_diffuse + w_specular + w_transmission + w_coat + w_sheen, 1e-6);
    b.p_diffuse = w_diffuse / total;
    b.p_specular = w_specular / total;
    b.p_transmission = w_transmission / total;
    b.p_coat = w_coat / total;
    b.p_sheen = w_sheen / total;

    var frame = onb_make(n);
    if (s.anisotropy_rotation != 0.0 && abs(aniso) > 1e-4) {
        let c = cos(s.anisotropy_rotation);
        let sn = sin(s.anisotropy_rotation);
        let t = frame.t * c + frame.b * sn;
        let bb = frame.b * c - frame.t * sn;
        frame.t = t;
        frame.b = bb;
    }
    b.frame = frame;
    b.geom = geom;
    b.s = s;
    let coat_rough = clamp(s.clearcoat_roughness, 0.0, 1.0);
    b.coat_alpha = coat_rough * coat_rough;
    let sheen_rough = clamp(s.sheen_roughness, 0.07, 1.0);
    b.sheen_alpha = sheen_rough * sheen_rough;
    return b;
}

fn bsdf_is_delta(b: Bsdf) -> bool {
    let spec_smooth = max(b.ax, b.ay) < SMOOTH_ALPHA;
    let coat_smooth = b.coat_alpha < SMOOTH_ALPHA;
    return b.p_diffuse <= 0.0 && spec_smooth && (b.p_coat <= 0.0 || coat_smooth);
}

struct Eval {
    f: vec3<f32>,
    pdf: f32,
};

fn eval_transmission(b: Bsdf, wo: vec3<f32>, wi: vec3<f32>) -> Eval {
    var out: Eval;
    out.f = vec3<f32>(0.0, 0.0, 0.0);
    out.pdf = 0.0;
    let eta_base = max(b.s.ior, 1.0001);
    let dispersion = max(b.s.dispersion, 0.0);
    let cos_o = wo.z;
    let cos_i = wi.z;
    if (cos_i == 0.0 || cos_o == 0.0) {
        return out;
    }
    let reflecting = cos_o * cos_i > 0.0;
    var pt = 0.0;
    let offsets = array<f32, 3>(-1.0, 0.0, 1.0);
    for (var ch = 0u; ch < 3u; ch = ch + 1u) {
        let eta = eta_base * (1.0 + dispersion * offsets[ch]);
        let ch_out = eval_transmission_channel(b, wo, wi, eta, reflecting, ch);
        if (ch == 0u) { out.f.r = ch_out.x; }
        else if (ch == 1u) { out.f.g = ch_out.x; }
        else { out.f.b = ch_out.x; }
        pt = pt + ch_out.y;
    }
    out.pdf = pt / 3.0;
    return out;
}

fn eval_local(b: Bsdf, wo: vec3<f32>, wi: vec3<f32>) -> Eval {
    var out: Eval;
    out.f = vec3<f32>(0.0, 0.0, 0.0);
    out.pdf = 0.0;
    let reflecting = wo.z * wi.z > 0.0;
    let spec_smooth = max(b.ax, b.ay) < SMOOTH_ALPHA;
    let coat_smooth = b.coat_alpha < SMOOTH_ALPHA;

    var coat_fresnel = 0.0;
    if (b.s.clearcoat > 0.0 && reflecting && !coat_smooth) {
        var h = normalize(wo + wi);
        if (h.z < 0.0) { h = -h; }
        // Clearcoat is polyurethane: IOR 1.5, F0 0.04, always.
        let fr = schlick1(0.04, abs(dot(wo, h)));
        coat_fresnel = fr * b.s.clearcoat;
        let d = ggx_d(h, b.coat_alpha, b.coat_alpha);
        let g = smith_g2(wo, wi, b.coat_alpha, b.coat_alpha);
        let v = b.s.clearcoat * d * g * fr / max(4.0 * abs(wo.z) * abs(wi.z), 1e-9);
        out.f = out.f + vec3<f32>(v, v, v);
        if (b.p_coat > 0.0) {
            out.pdf = out.pdf + b.p_coat * vndf_reflect_pdf(wo, h, b.coat_alpha, b.coat_alpha);
        }
    } else if (b.s.clearcoat > 0.0 && reflecting) {
        coat_fresnel = schlick1(0.04, abs(wo.z)) * b.s.clearcoat;
    }
    let under_coat = 1.0 - coat_fresnel;

    if (b.p_diffuse > 0.0 && reflecting) {
        let kd = 1.0 - schlick1(luminance(b.f0), abs(wo.z));
        let grazing = max(1.0 - abs(wo.z), 0.0);
        let ss = clamp(b.s.subsurface, 0.0, 1.0);
        let ss_boost = vec3<f32>(1.0) + b.s.subsurface_radius * ss * grazing * grazing;
        out.f = out.f + b.s.base_color * ss_boost * ((1.0 - b.s.metallic) * (1.0 - b.s.transmission) * kd * under_coat / PI);
        out.pdf = out.pdf + b.p_diffuse * (abs(wi.z) / PI);
    }

    if (b.p_sheen > 0.0 && reflecting && b.s.sheen > 0.0) {
        var h = normalize(wo + wi);
        if (h.z < 0.0) { h = -h; }
        let n_dot_l = abs(wi.z);
        let n_dot_v = abs(wo.z);
        let n_dot_h = abs(h.z);
        let tint = b.s.sheen_color * b.s.sheen;
        let d = d_charlie(n_dot_h, b.sheen_alpha);
        let v = tint * d * v_neubelt(n_dot_v, n_dot_l) * n_dot_l / PI * under_coat;
        out.f = out.f + v;
        out.pdf = out.pdf + b.p_sheen * (n_dot_l / PI);
    }

    if (!spec_smooth && reflecting) {
        var h = normalize(wo + wi);
        if (h.z < 0.0) { h = -h; }
        let d = ggx_d(h, b.ax, b.ay);
        let g = smith_g2(wo, wi, b.ax, b.ay);
        let fr = specular_fresnel(b, wo, h);
        let denom = max(4.0 * abs(wo.z) * abs(wi.z), 1e-9);
        let ms = multiscatter_gain(b, abs(wo.z));
        out.f = out.f + fr * (d * g / denom * under_coat * b.spec_weight) * ms;
        out.pdf = out.pdf + b.p_specular * vndf_reflect_pdf(wo, h, b.ax, b.ay);
    }

    if (b.p_transmission > 0.0 && !spec_smooth) {
        let t = eval_transmission(b, wo, wi);
        out.f = out.f + t.f * under_coat;
        out.pdf = out.pdf + b.p_transmission * t.pdf;
    }
    out.pdf = max(out.pdf, 0.0);
    return out;
}

fn bsdf_eval(b: Bsdf, wo_world: vec3<f32>, wi_world: vec3<f32>) -> Eval {
    var out: Eval;
    out.f = vec3<f32>(0.0, 0.0, 0.0);
    out.pdf = 0.0;
    // A shading normal tilted off its triangle must not admit light from the
    // far side of the geometry — that is the classic smooth-shading leak.
    if (dot(wi_world, b.frame.n) * dot(wi_world, b.geom) < 0.0) {
        return out;
    }
    let wo = onb_to_local(b.frame, wo_world);
    let wi = onb_to_local(b.frame, wi_world);
    if (abs(wo.z) < 1e-6) {
        return out;
    }
    return eval_local(b, wo, wi);
}

struct Sample {
    dir: vec3<f32>,
    f: vec3<f32>,
    pdf: f32,
    specular: bool,
    transmitted: bool,
    valid: bool,
};

fn reflect_about(v: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    return v - n * (2.0 * dot(v, n));
}

// `v` points into the surface, `n` faces the side it came from, and `eta` is
// n_incident / n_transmitted. Returns w = 0 on total internal reflection.
fn refract_ray(v: vec3<f32>, n: vec3<f32>, eta: f32) -> vec4<f32> {
    let cos_i = -dot(v, n);
    if (cos_i <= 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let sin2_t = eta * eta * (1.0 - cos_i * cos_i);
    if (sin2_t > 1.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let cos_t = sqrt(max(0.0, 1.0 - sin2_t));
    return vec4<f32>(v * eta + n * (eta * cos_i - cos_t), 1.0);
}

fn bsdf_sample(b: Bsdf, wo_world: vec3<f32>, rng: ptr<function, Sampler>) -> Sample {
    var out: Sample;
    out.valid = false;
    out.dir = vec3<f32>(0.0, 0.0, 1.0);
    out.f = vec3<f32>(0.0, 0.0, 0.0);
    out.pdf = 0.0;
    out.specular = false;
    out.transmitted = false;

    let wo = onb_to_local(b.frame, wo_world);
    if (abs(wo.z) < 1e-6) {
        return out;
    }
    let pick = rand(rng);
    let u = rand2(rng);
    let spec_smooth = max(b.ax, b.ay) < SMOOTH_ALPHA;
    let coat_smooth = b.coat_alpha < SMOOTH_ALPHA;

    var wi = vec3<f32>(0.0, 0.0, 0.0);
    var is_delta = false;
    var delta_value = vec3<f32>(0.0, 0.0, 0.0);
    var transmitted = false;

    var acc = b.p_diffuse;
    if (pick < acc) {
        var d = cosine_hemisphere(u);
        if (wo.z < 0.0) { d.z = -d.z; }
        wi = d;
    } else {
        acc = acc + b.p_specular;
        if (pick < acc) {
            if (spec_smooth) {
                wi = vec3<f32>(-wo.x, -wo.y, wo.z);
                is_delta = true;
                let h = vec3<f32>(0.0, 0.0, 1.0);
                delta_value = specular_fresnel(b, wo, h) * (b.spec_weight / max(abs(wi.z), 1e-6));
            } else {
                let wo_up = select(wo, -wo, wo.z < 0.0);
                let h = sample_ggx_vndf(wo_up, b.ax, b.ay, u);
                wi = reflect_about(-wo, h);
                if (wi.z * wo.z <= 0.0) {
                    return out;
                }
            }
        } else {
            acc = acc + b.p_transmission;
            if (pick < acc) {
                let eta_base = max(b.s.ior, 1.0001);
                let dispersion = max(b.s.dispersion, 0.0);
                let weight = (1.0 - b.s.metallic) * b.s.transmission;
                var h = vec3<f32>(0.0, 0.0, 1.0);
                if (!spec_smooth) {
                    let wo_up = select(wo, -wo, wo.z < 0.0);
                    h = sample_ggx_vndf(wo_up, b.ax, b.ay, u);
                }
                let cos_oh = dot(wo, h);
                var fr_avg = 0.0;
                let offsets = array<f32, 3>(-1.0, 0.0, 1.0);
                for (var ch = 0u; ch < 3u; ch = ch + 1u) {
                    let eta = eta_base * (1.0 + dispersion * offsets[ch]);
                    fr_avg = fr_avg + fresnel_dielectric(cos_oh, eta);
                }
                fr_avg = fr_avg / 3.0;
                let do_reflect = rand(rng) < fr_avg;
                if (do_reflect) {
                    wi = reflect_about(-wo, h);
                } else {
                    let eta = eta_base;
                    var n = h;
                    var ratio = 1.0 / eta;
                    if (cos_oh <= 0.0) {
                        n = -h;
                        ratio = eta;
                    }
                    let r = refract_ray(-wo, n, ratio);
                    if (r.w == 0.0) {
                        return out;
                    }
                    wi = r.xyz;
                }
                if (wi.z == 0.0) {
                    return out;
                }
                if (spec_smooth) {
                    is_delta = true;
                    transmitted = !do_reflect;
                    var value = vec3<f32>(0.0);
                    for (var ch = 0u; ch < 3u; ch = ch + 1u) {
                        let eta = eta_base * (1.0 + dispersion * offsets[ch]);
                        let fr = fresnel_dielectric(cos_oh, eta);
                        var v = 0.0;
                        if (do_reflect) {
                            v = fr * weight / max(abs(wi.z), 1e-6);
                        } else {
                            let etap = select(1.0 / eta, eta, wo.z > 0.0);
                            let t = (1.0 - fr) * weight / (etap * etap * max(abs(wi.z), 1e-6));
                            if (ch == 0u) { v = b.s.base_color.r * t; }
                            else if (ch == 1u) { v = b.s.base_color.g * t; }
                            else { v = b.s.base_color.b * t; }
                        }
                        if (ch == 0u) { value.r = v; }
                        else if (ch == 1u) { value.g = v; }
                        else { value.b = v; }
                    }
                    let div = select(max(1.0 - fr_avg, 1e-6), max(fr_avg, 1e-6), do_reflect);
                    delta_value = value / div;
                }
            } else {
                acc = acc + b.p_coat;
                if (pick < acc) {
                    if (coat_smooth) {
                        wi = vec3<f32>(-wo.x, -wo.y, wo.z);
                        is_delta = true;
                        let fr = schlick1(0.04, abs(wo.z)) * b.s.clearcoat;
                        let v = fr / max(abs(wi.z), 1e-6);
                        delta_value = vec3<f32>(v, v, v);
                    } else {
                        let wo_up = select(wo, -wo, wo.z < 0.0);
                        let h = sample_ggx_vndf(wo_up, b.coat_alpha, b.coat_alpha, u);
                        wi = reflect_about(-wo, h);
                        if (wi.z * wo.z <= 0.0) {
                            return out;
                        }
                    }
                } else {
                    acc = acc + b.p_sheen;
                    if (pick < acc) {
                        var d = cosine_hemisphere(u);
                        if (wo.z < 0.0) { d.z = -d.z; }
                        wi = d;
                    } else {
                        return out;
                    }
                }
            }
        }
    }

    let wi_world = onb_to_world(b.frame, wi);
    if (!is_delta && dot(wi_world, b.geom) * wi.z < 0.0) {
        return out;
    }

    if (is_delta) {
        out.dir = wi_world;
        out.f = delta_value;
        out.pdf = 1.0;
        out.specular = true;
        out.transmitted = transmitted;
        out.valid = true;
        return out;
    }

    // Re-evaluate the whole BSDF for the direction that came out, so the value
    // and density account for every lobe rather than only the one that produced
    // it. That is what keeps the mixture consistent with `bsdf_eval`.
    let e = eval_local(b, wo, wi);
    if (e.pdf <= 0.0 || (e.f.r <= 0.0 && e.f.g <= 0.0 && e.f.b <= 0.0)) {
        return out;
    }
    out.dir = wi_world;
    out.f = e.f;
    out.pdf = e.pdf;
    out.specular = false;
    out.transmitted = wi.z * wo.z < 0.0;
    out.valid = true;
    return out;
}

// ---------------------------------------------------------------- lights

struct LightSample {
    dir: vec3<f32>,
    dist: f32,
    weight: vec3<f32>,
    pdf: f32,
    delta: bool,
    valid: bool,
};

fn distance_attenuation(d: f32, max_d: f32, decay: f32) -> f32 {
    var att = 1.0 / max(pow(d, decay), 0.01);
    if (max_d > 0.0) {
        let t = clamp(1.0 - pow(d / max_d, 4.0), 0.0, 1.0);
        att = att * t * t;
    }
    return att;
}

fn ray_sphere_near(o: vec3<f32>, d: vec3<f32>, c: vec3<f32>, r: f32) -> f32 {
    let oc = o - c;
    let b = dot(oc, d);
    let cc = dot(oc, oc) - r * r;
    let disc = b * b - cc;
    if (disc < 0.0) {
        return -1.0;
    }
    let s = sqrt(disc);
    let t = -b - s;
    if (t > 1e-5) {
        return t;
    }
    let t2 = -b + s;
    return select(-1.0, t2, t2 > 1e-5);
}

fn sample_sphere_light(p: vec3<f32>, center: vec3<f32>, radius: f32, d: f32, intensity: vec3<f32>, rng: ptr<function, Sampler>) -> LightSample {
    var out: LightSample;
    out.valid = false;
    out.delta = false;
    out.pdf = 0.0;
    if (radius <= 0.0 || d <= radius) {
        out.dir = (center - p) / d;
        out.dist = d;
        out.weight = intensity;
        out.delta = true;
        out.valid = true;
        return out;
    }
    let axis = (center - p) / d;
    let sin_max = radius / d;
    let cos_max = sqrt(max(0.0, 1.0 - sin_max * sin_max));
    let frame = onb_make(axis);
    let dir = onb_to_world(frame, uniform_cone(rand2(rng), cos_max));
    let pdf = uniform_cone_pdf(cos_max);
    if (pdf <= 0.0) {
        return out;
    }
    var hit = ray_sphere_near(p, dir, center, radius);
    if (hit < 0.0) {
        hit = d - radius;
    }
    out.dir = dir;
    out.dist = max(hit, 1e-4);
    // intensity/Omega then /pdf leaves intensity, so a light with a radius is
    // exactly as bright as the delta light it replaces.
    out.weight = intensity;
    out.pdf = pdf;
    out.valid = true;
    return out;
}

fn sample_analytic(li: u32, p: vec3<f32>, rng: ptr<function, Sampler>) -> LightSample {
    var out: LightSample;
    out.valid = false;
    out.delta = false;
    out.pdf = 0.0;
    let b = uni.data_offsets.w + li * LIGHT_STRIDE;
    let kind = u32(data[b]);
    let v1 = vec3<f32>(data[b + 1u], data[b + 2u], data[b + 3u]);
    let color = vec3<f32>(data[b + 4u], data[b + 5u], data[b + 6u]);
    let radius = data[b + 7u];
    let v2 = vec3<f32>(data[b + 8u], data[b + 9u], data[b + 10u]);
    let distance = data[b + 11u];
    let v3 = vec3<f32>(data[b + 12u], data[b + 13u], data[b + 14u]);
    let decay = data[b + 15u];
    let cos_outer = data[b + 16u];
    let cos_inner = data[b + 17u];

    if (kind == 0u) {
        let to_light = -v1;
        if (radius <= 0.0) {
            out.dir = to_light;
            out.dist = INF;
            out.weight = color;
            out.delta = true;
            out.valid = true;
            return out;
        }
        let cos_max = cos(radius);
        let frame = onb_make(to_light);
        let pdf = uniform_cone_pdf(cos_max);
        if (pdf <= 0.0) {
            return out;
        }
        out.dir = onb_to_world(frame, uniform_cone(rand2(rng), cos_max));
        out.dist = INF;
        out.weight = color;
        out.pdf = pdf;
        out.valid = true;
        return out;
    }
    if (kind == 1u || kind == 2u) {
        let to = v1 - p;
        let d = length(to);
        if (d <= 1e-6) {
            return out;
        }
        let att = distance_attenuation(d, distance, decay);
        if (att <= 0.0) {
            return out;
        }
        var s = sample_sphere_light(p, v1, radius, d, color * att, rng);
        if (!s.valid) {
            return s;
        }
        if (kind == 2u) {
            let cos_angle = dot(-s.dir, v2);
            if (cos_angle <= cos_outer) {
                s.valid = false;
                return s;
            }
            let t = clamp((cos_angle - cos_outer) / max(cos_inner - cos_outer, 1e-9), 0.0, 1.0);
            let cone = t * t * (3.0 - 2.0 * t);
            if (cone <= 0.0) {
                s.valid = false;
                return s;
            }
            s.weight = s.weight * cone;
        }
        return s;
    }
    // Rect: v1 = centre, v2 = half-extent along its local X, v3 = along Y.
    let u = rand2(rng);
    let q = v1 + v2 * (2.0 * u.x - 1.0) + v3 * (2.0 * u.y - 1.0);
    let nrm = cross(v2, v3);
    let area = 4.0 * length(nrm);
    if (area <= 0.0) {
        return out;
    }
    // three.js aims a RectAreaLight down its local -Z, and right x up is +Z.
    let normal = -normalize(nrm);
    let to = q - p;
    let d2 = dot(to, to);
    if (d2 <= 1e-12) {
        return out;
    }
    let d = sqrt(d2);
    let dir = to / d;
    let cos_light = dot(-dir, normal);
    if (cos_light <= 1e-6) {
        return out;
    }
    let pdf = d2 / (cos_light * area);
    if (pdf <= 0.0) {
        return out;
    }
    out.dir = dir;
    out.dist = d;
    out.weight = color / pdf;
    out.pdf = pdf;
    out.valid = true;
    return out;
}

struct EmitterSample {
    dir: vec3<f32>,
    dist: f32,
    radiance: vec3<f32>,
    pdf: f32,
    valid: bool,
};

fn emissive_cdf(i: u32) -> f32 {
    return data[uni.data_offsets.y + i * EMIT_STRIDE + 1u];
}

fn emissive_area(i: u32) -> f32 {
    return data[uni.data_offsets.y + i * EMIT_STRIDE];
}

fn emissive_prob(slot: u32) -> f32 {
    // Written as a branch, not a `select`: `select` evaluates both arms, and
    // one of them would index element -1.
    var prev = 0.0;
    if (slot > 0u) {
        prev = emissive_cdf(slot - 1u);
    }
    return max(emissive_cdf(slot) - prev, 0.0);
}

// Binary search of the area x luminance CDF.
fn emissive_pick(u: f32) -> u32 {
    let n = uni.counts.w;
    var lo = 0u;
    var hi = n - 1u;
    loop {
        if (lo >= hi) {
            break;
        }
        let mid = (lo + hi) / 2u;
        if (emissive_cdf(mid) <= u) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo;
}

fn sample_emissive(p: vec3<f32>, rng: ptr<function, Sampler>) -> EmitterSample {
    var out: EmitterSample;
    out.valid = false;
    if (uni.counts.w == 0u) {
        return out;
    }
    let slot = emissive_pick(rand(rng));
    let p_select = emissive_prob(slot);
    if (p_select <= 0.0) {
        return out;
    }
    let tri = idx[uni.idx_offsets.w + slot];
    let u = rand2(rng);
    let su = sqrt(u.x);
    let b0 = 1.0 - su;
    let b1 = u.y * su;
    let b2 = su * (1.0 - u.y);
    let p0 = tri_vertex(tri, 0u);
    let p1 = tri_vertex(tri, 1u);
    let p2 = tri_vertex(tri, 2u);
    let q = p0 * b0 + p1 * b1 + p2 * b2;
    let to = q - p;
    let d2 = dot(to, to);
    if (d2 <= 1e-12) {
        return out;
    }
    let d = sqrt(d2);
    let dir = to / d;
    let face = cross(p1 - p0, p2 - p0);
    if (dot(face, face) <= 0.0) {
        return out;
    }
    // Mesh emitters radiate from both faces, as Blender's emission shader does.
    let cos_light = abs(dot(-dir, normalize(face)));
    if (cos_light <= 1e-6) {
        return out;
    }
    let area = emissive_area(slot);
    let pdf = p_select * d2 / (cos_light * area);
    if (pdf <= 0.0) {
        return out;
    }
    let mat = tri_material(tri);
    let base = uni.data_offsets.z + mat * MAT_STRIDE;
    var radiance = material_emission_const(mat);
    if (map_present(base, 3u)) {
        let uv = tri_uv_at(tri, 0u) * b0 + tri_uv_at(tri, 1u) * b1 + tri_uv_at(tri, 2u) * b2;
        radiance = radiance * map_sample(base, 3u, uv, map_wrap(base, 3u)).rgb;
    }
    if (dot(radiance, radiance) <= 0.0) {
        return out;
    }
    out.dir = dir;
    out.dist = d;
    out.radiance = radiance;
    out.pdf = pdf;
    out.valid = true;
    return out;
}

fn emissive_pdf_for(tri: u32, distance: f32, cos_light: f32) -> f32 {
    if (uni.counts.w == 0u) {
        return 0.0;
    }
    let slot = idx[uni.idx_offsets.z + tri];
    if (slot == 0xffffffffu) {
        return 0.0;
    }
    let p_select = emissive_prob(slot);
    let area = emissive_area(slot);
    if (p_select <= 0.0 || area <= 0.0 || abs(cos_light) <= 1e-6) {
        return 0.0;
    }
    return p_select * distance * distance / (abs(cos_light) * area);
}

// ---------------------------------------------------------------- visibility

fn visibility(o: vec3<f32>, d: vec3<f32>, t_max_in: f32, eps: f32) -> vec3<f32> {
    var t_max = t_max_in;
    if (t_max < INF) {
        t_max = t_max - eps;
    }
    if (t_max <= eps) {
        return vec3<f32>(1.0, 1.0, 1.0);
    }
    if ((uni.flags.w & 2u) != 0u) {
        return select(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.0, 0.0, 0.0), trace_any(o, d, eps, t_max));
    }
    // Some material can let a shadow ray through, so the segment has to be
    // walked layer by layer instead of stopping at the first hit.
    var transmittance = vec3<f32>(1.0, 1.0, 1.0);
    var origin = o;
    var remaining = t_max;
    let max_layers = uni.limits.z;
    for (var layer = 0u; layer <= max_layers; layer = layer + 1u) {
        let h = trace_closest(origin, d, eps, remaining);
        if (!h.hit) {
            break;
        }
        let mat = tri_material(h.tri);
        let base = uni.data_offsets.z + mat * MAT_STRIDE;
        // Refractive glass: by default block (caustics arrive via BSDF paths).
        // With bit2 set, attenuate and continue — same bias as the CPU path.
        if (data[base + 10u] > 0.0 && data[base + 3u] >= 1.0) {
            if ((uni.flags.w & 4u) != 0u) {
                transmittance = transmittance * vec3<f32>(0.12, 0.12, 0.12);
                origin = origin + d * (h.t + eps);
                remaining = remaining - h.t - eps;
                if (remaining <= eps
                    || max(transmittance.r, max(transmittance.g, transmittance.b)) < 1e-4)
                {
                    break;
                }
                continue;
            }
            return vec3<f32>(0.0, 0.0, 0.0);
        }
        var a = data[base + 3u];
        if (map_present(base, 0u)) {
            let w = 1.0 - h.u - h.v;
            let uv = tri_uv_at(h.tri, 0u) * w + tri_uv_at(h.tri, 1u) * h.u + tri_uv_at(h.tri, 2u) * h.v;
            a = a * map_sample(base, 0u, uv, map_wrap(base, 0u)).a;
        }
        let alpha_test = data[base + 7u];
        if (alpha_test > 0.0 && a < alpha_test) {
            a = 0.0;
        }
        if (a >= 1.0) {
            return vec3<f32>(0.0, 0.0, 0.0);
        }
        transmittance = transmittance * (1.0 - a);
        if (max(transmittance.r, max(transmittance.g, transmittance.b)) < 1e-4) {
            break;
        }
        origin = origin + d * (h.t + eps);
        remaining = remaining - h.t - eps;
        if (remaining <= eps) {
            break;
        }
    }
    return transmittance;
}

// ---------------------------------------------------------------- camera

struct Ray {
    o: vec3<f32>,
    d: vec3<f32>,
};

fn unproject(ndc: vec3<f32>) -> vec3<f32> {
    let clip = uni.inv_proj * vec4<f32>(ndc, 1.0);
    let view = clip.xyz / select(clip.w, 1.0, clip.w == 0.0);
    let world = uni.inv_view * vec4<f32>(view, 1.0);
    return world.xyz / select(world.w, 1.0, world.w == 0.0);
}

fn camera_ray_at(
    inv_view: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    cam_right: vec4<f32>,
    cam_up: vec4<f32>,
    cam_forward: vec4<f32>,
    px: vec2<f32>,
    lens: vec2<f32>,
) -> Ray {
    let u = px.x / f32(uni.dims.x);
    let v = px.y / f32(uni.dims.y);
    let ndc = vec2<f32>(u * 2.0 - 1.0, 1.0 - v * 2.0);
    let near = (inv_proj * vec4<f32>(ndc, 0.0, 1.0));
    let far = (inv_proj * vec4<f32>(ndc, 1.0, 1.0));
    let near_w = (inv_view * near).xyz / select(near.w, 1.0, near.w == 0.0);
    let far_w = (inv_view * far).xyz / select(far.w, 1.0, far.w == 0.0);

    var r: Ray;
    if (cam_pos.w > 0.5) {
        r.o = cam_pos.xyz;
        let dd = far_w - r.o;
        r.d = select(cam_forward.xyz, normalize(dd), dot(dd, dd) > 1e-20);
    } else {
        r.o = near_w;
        let dd = far_w - near_w;
        r.d = select(cam_forward.xyz, normalize(dd), dot(dd, dd) > 1e-20);
    }

    let aperture = cam_right.w;
    let focus = cam_up.w;
    if (aperture > 0.0 && focus > 0.0) {
        let c = dot(r.d, cam_forward.xyz);
        if (abs(c) > 1e-6) {
            let focal_point = r.o + r.d * (focus / c);
            let l = concentric_disk(lens);
            r.o = r.o + cam_right.xyz * (l.x * aperture) + cam_up.xyz * (l.y * aperture);
            let dd = focal_point - r.o;
            if (dot(dd, dd) > 1e-20) {
                r.d = normalize(dd);
            }
        }
    }
    return r;
}

fn camera_ray(px: vec2<f32>, lens: vec2<f32>, shutter: f32) -> Ray {
    let r1 = camera_ray_at(
        uni.inv_view,
        uni.inv_proj,
        uni.cam_pos,
        uni.cam_right,
        uni.cam_up,
        uni.cam_forward,
        px,
        lens,
    );
    let shutter_open = uni.params.w;
    if (shutter_open <= 0.0) {
        return r1;
    }
    let r0 = camera_ray_at(
        uni.motion_inv_view,
        uni.motion_inv_proj,
        uni.motion_prev,
        vec4<f32>(uni.motion_prev_right.xyz, uni.cam_right.w),
        vec4<f32>(uni.motion_prev_up.xyz, uni.cam_up.w),
        uni.motion_prev_forward,
        px,
        lens,
    );
    let t = clamp(shutter, 0.0, 1.0) * shutter_open;
    var r: Ray;
    r.o = mix(r0.o, r1.o, t);
    let d = mix(r0.d, r1.d, 1.0 - t);
    if (dot(d, d) > 1e-20) {
        r.d = normalize(d);
    } else {
        r.d = r1.d;
    }
    return r;
}

fn uniform_sphere(u1: f32, u2: f32) -> vec3<f32> {
    let z = 1.0 - 2.0 * u1;
    let r = sqrt(max(0.0, 1.0 - z * z));
    let phi = 6.28318530718 * u2;
    return vec3<f32>(r * cos(phi), r * sin(phi), z);
}

fn direct_light_add(
    shadow_origin: vec3<f32>,
    hp: vec3<f32>,
    ns: vec3<f32>,
    ns_facing: vec3<f32>,
    wo: vec3<f32>,
    throughput: vec3<f32>,
    bounce: u32,
    b: Bsdf,
    rng: ptr<function, Sampler>,
    eps: f32,
) -> vec3<f32> {
    var out = vec3<f32>(0.0, 0.0, 0.0);
    let n_lights = uni.counts.z;
    if (n_lights > 0u) {
        let li = min(u32(rand(rng) * f32(n_lights)), n_lights - 1u);
        let ls = sample_analytic(li, hp, rng);
        if (ls.valid) {
            let e = bsdf_eval(b, wo, ls.dir);
            let cos = abs(dot(ls.dir, ns));
            if (cos > 0.0 && dot(e.f, e.f) > 0.0) {
                let vis = visibility(shadow_origin, ls.dir, ls.dist, eps);
                if (dot(vis, vis) > 0.0) {
                    let c = throughput * e.f * ls.weight * vis * (cos * f32(n_lights));
                    out = out + clamp_contribution(c, bounce);
                }
            }
        }
    }

    let es = sample_emissive(hp, rng);
    if (es.valid) {
        let e = bsdf_eval(b, wo, es.dir);
        let cos = abs(dot(es.dir, ns));
        if (cos > 0.0 && dot(e.f, e.f) > 0.0) {
            let vis = visibility(shadow_origin, es.dir, es.dist * (1.0 - 1e-3), eps);
            if (dot(vis, vis) > 0.0) {
                let mis = power_heuristic(es.pdf, e.pdf);
                let c = throughput * e.f * es.radiance * vis * (cos * mis / es.pdf);
                out = out + clamp_contribution(c, bounce);
            }
        }
    }

    if (!world_is_black()) {
        let pick = rand(rng);
        let uu = rand2(rng);
        var wdir: vec3<f32>;
        if (env_has_distribution() && pick < ENV_STRATEGY) {
            wdir = env_sample_dir(uu.x, uu.y);
        } else {
            wdir = onb_to_world(onb_make(ns_facing), cosine_hemisphere(uu));
        }
        let wpdf = world_pdf(ns_facing, wdir);
        if (wpdf > 1e-9) {
            let light = world_lighting(wdir);
            if (dot(light, light) > 0.0) {
                let e = bsdf_eval(b, wo, wdir);
                let cos = abs(dot(wdir, ns));
                if (cos > 0.0 && dot(e.f, e.f) > 0.0) {
                    let vis = visibility(shadow_origin, wdir, INF, eps);
                    if (dot(vis, vis) > 0.0) {
                        let mis = power_heuristic(wpdf, e.pdf);
                        let c = throughput * e.f * light * vis * (cos * mis / wpdf);
                        out = out + clamp_contribution(c, bounce);
                    }
                }
            }
        }
    }
    return out;
}

fn subsurface_walk(
    mat_id: u32,
    s: Surface,
    base_color: vec3<f32>,
    hp: vec3<f32>,
    ns: vec3<f32>,
    ng: vec3<f32>,
    ns_facing: vec3<f32>,
    ng_facing: vec3<f32>,
    wo: vec3<f32>,
    throughput: vec3<f32>,
    bounce: u32,
    rng: ptr<function, Sampler>,
    eps: f32,
) -> vec3<f32> {
    var out = vec3<f32>(0.0, 0.0, 0.0);
    let ss = clamp(s.subsurface, 0.0, 1.0);
    if (ss <= 0.0) {
        return out;
    }
    let mfp = (s.subsurface_radius.x + s.subsurface_radius.y + s.subsurface_radius.z)
        / 3.0 * uni.render_params.x * 0.02;
    let mfp_clamped = max(mfp, eps * 4.0);
    var pos = hp - ng_facing * eps * 2.0;
    var walk_tp = throughput * base_color * ss;
    let b = bsdf_make(s, ns, ng);
    let shadow_bias = eps * 2.0;

    for (var step = 0u; step < 6u; step = step + 1u) {
        let u = max(rand(rng), 1e-6);
        let scatter = uniform_sphere(rand(rng), rand(rng));
        pos = pos + scatter * (-mfp_clamped * log(u));
        let to_surface = -ng_facing;
        let exit = trace_closest(pos, to_surface, eps, mfp_clamped * 8.0);
        if (exit.hit && tri_material(exit.tri) == mat_id) {
            let w0 = 1.0 - exit.u - exit.v;
            let ep0 = tri_vertex(exit.tri, 0u);
            let ep1 = tri_vertex(exit.tri, 1u);
            let ep2 = tri_vertex(exit.tri, 2u);
            let exit_p = ep0 * w0 + ep1 * exit.u + ep2 * exit.v;
            let exit_origin = exit_p + ng_facing * shadow_bias;
            out = out + direct_light_add(
                exit_origin,
                exit_p,
                ns,
                ns_facing,
                wo,
                walk_tp,
                bounce,
                b,
                rng,
                eps,
            );
        }
        let q = clamp(max(walk_tp.r, max(walk_tp.g, walk_tp.b)), 0.05, 0.95);
        if (rand(rng) >= q) {
            break;
        }
        walk_tp = walk_tp / q;
    }
    return out;
}

// ---------------------------------------------------------------- integrator

struct PathResult {
    radiance: vec3<f32>,
    alpha: f32,
    albedo: vec3<f32>,
    normal: vec3<f32>,
    depth: f32,
};

fn clamp_contribution(v: vec3<f32>, bounce: u32) -> vec3<f32> {
    let limit = select(uni.params.y, uni.params.x, bounce == 0u);
    if (limit <= 0.0) {
        return v;
    }
    let m = max(v.r, max(v.g, v.b));
    if (m > limit) {
        return v * (limit / m);
    }
    return v;
}

fn trace_path(ray_in: Ray, rng: ptr<function, Sampler>) -> PathResult {
    var res: PathResult;
    res.radiance = vec3<f32>(0.0, 0.0, 0.0);
    res.alpha = 1.0;
    res.albedo = vec3<f32>(0.0, 0.0, 0.0);
    res.normal = vec3<f32>(0.0, 0.0, 0.0);
    res.depth = INF;

    let eps = uni.cam_forward.w;
    var origin = ray_in.o;
    var dir = ray_in.d;
    var throughput = vec3<f32>(1.0, 1.0, 1.0);
    var travelled = 0.0;
    var bounce = 0u;
    var transparent_bounces = 0u;
    var camera_ray_flag = true;
    var prev_specular = true;
    var prev_pdf = 0.0;
    var prev_normal = dir;
    // The denoiser's guide passes follow the path through specular and
    // transmissive surfaces rather than stopping at the first hit — see
    // `guide_split`.
    var feature_throughput = vec3<f32>(1.0, 1.0, 1.0);
    var features_open = true;
    var depth_recorded = false;
    var in_medium = false;
    var medium_sigma = vec3<f32>(0.0, 0.0, 0.0);

    // Bounded outer loop: transparent pass-throughs do not count as bounces, so
    // the worst case is max_bounces + transparent_max_bounces, plus slack. The
    // ceiling is a backstop against a runaway, not a policy — a fixed 128 here
    // would quietly render fewer bounces than the CPU on the same settings.
    let max_steps = min(uni.limits.x + uni.limits.z + 8u, 1024u);
    for (var step = 0u; step < max_steps; step = step + 1u) {
        let h = trace_closest(origin, dir, eps, INF);

        if (in_medium && h.hit && h.t > 0.0) {
            // Beer-Lambert over what was crossed, applied before the hit is
            // shaded because it attenuates the light arriving there.
            throughput = throughput * exp(-medium_sigma * h.t);
        }

        if (!h.hit) {
            var l: vec3<f32>;
            if (camera_ray_flag) {
                let bg = world_camera(dir);
                l = bg.rgb;
                res.alpha = bg.a;
            } else {
                l = world_lighting(dir);
            }
            var w = 1.0;
            if (!prev_specular) {
                w = power_heuristic(prev_pdf, world_pdf(prev_normal, dir));
            }
            res.radiance = res.radiance + clamp_contribution(throughput * l * w, bounce);
            if (features_open) {
                res.albedo = res.albedo + reinhard(l) * feature_throughput;
                res.normal = res.normal + (-dir) * average3(feature_throughput);
            }
            break;
        }

        // Point the sampler at this bounce's block of dimensions before
        // anything draws from it, so the same decision lands on the same
        // dimension in every sample of this pixel.
        sampler_set_bounce(rng, bounce + 1u);

        travelled = travelled + h.t;
        let w0 = 1.0 - h.u - h.v;
        let p0 = tri_vertex(h.tri, 0u);
        let p1 = tri_vertex(h.tri, 1u);
        let p2 = tri_vertex(h.tri, 2u);
        let hp = p0 * w0 + p1 * h.u + p2 * h.v;
        let uv = tri_uv_at(h.tri, 0u) * w0 + tri_uv_at(h.tri, 1u) * h.u + tri_uv_at(h.tri, 2u) * h.v;
        let face_raw = cross(p1 - p0, p2 - p0);
        var face = vec3<f32>(0.0, 0.0, 1.0);
        if (dot(face_raw, face_raw) > 0.0) {
            face = normalize(face_raw);
        } else {
            face = -dir;
        }
        // Two normals, and the difference matters. `ng`/`ns` follow the
        // triangle's winding, so their sign tells the BSDF which side of the
        // surface the ray is on — the only way it can know whether the relative
        // index of refraction is 1/ior or ior. The `_facing` copies are flipped
        // toward the ray, which is what light sampling and ray offsets want.
        let front = dot(dir, face) < 0.0;
        let ng = face;
        var ns = tri_normal_at(h.tri, 0u) * w0 + tri_normal_at(h.tri, 1u) * h.u + tri_normal_at(h.tri, 2u) * h.v;
        if (dot(ns, ns) > 1e-12) {
            ns = normalize(ns);
        } else {
            ns = ng;
        }
        if (dot(ns, ng) < 0.0) {
            ns = -ns;
        }

        let mat = tri_material(h.tri);
        let mat_base = uni.data_offsets.z + mat * MAT_STRIDE;
        var s = load_surface(mat, uv);
        // Vertex colour multiplies the material, as it does on the CPU path and
        // in the rasteriser. Interpolated with the same barycentrics as the uv
        // above.
        let vcol = tri_color_at(h.tri, 0u) * w0
                 + tri_color_at(h.tri, 1u) * h.u
                 + tri_color_at(h.tri, 2u) * h.v;
        s.base_color = s.base_color * vcol;
        s.emission = s.emission * vcol;
        if (map_present(mat_base, 4u)) {
            ns = perturb_normal(h.tri, mat_base, ns, uv);
        }
        let ng_facing = select(-ng, ng, front);
        let ns_facing = select(-ns, ns, front);

        var surface_alpha = s.opacity;
        if (s.alpha_test > 0.0 && surface_alpha < s.alpha_test) {
            surface_alpha = 0.0;
        }
        if (surface_alpha < 1.0 && rand(rng) >= surface_alpha) {
            transparent_bounces = transparent_bounces + 1u;
            if (transparent_bounces > uni.limits.z) {
                if (camera_ray_flag) {
                    res.alpha = 0.0;
                }
                break;
            }
            origin = hp - ng_facing * eps;
            continue;
        }

        if (dot(s.emission, s.emission) > 0.0) {
            var w = 1.0;
            if (!prev_specular) {
                let cos_light = abs(dot(dir, face));
                w = power_heuristic(prev_pdf, emissive_pdf_for(h.tri, h.t, cos_light));
            }
            res.radiance = res.radiance + clamp_contribution(throughput * s.emission * w, bounce);
        }

        if (features_open) {
            if (s.unlit) {
                // No BSDF to describe; the surface is its own emission.
                res.albedo = res.albedo + reinhard(s.emission) * feature_throughput;
                res.normal = res.normal + ns_facing * average3(feature_throughput);
                features_open = false;
            } else {
                let g = guide_split(s);
                let weight = g[0].x;
                if (weight > 0.0) {
                    res.albedo = res.albedo + g[1] * feature_throughput * weight;
                    res.normal = res.normal + ns_facing * (weight * average3(feature_throughput));
                }
                feature_throughput = feature_throughput * g[2];
                if (max(feature_throughput.x, max(feature_throughput.y, feature_throughput.z))
                    < 1e-4) {
                    features_open = false;
                }
            }
        }
        if (!depth_recorded) {
            // Depth stays the distance to the first *hit*, not to wherever the
            // guides ended up: its job is to say "this is a different surface
            // from that one", and for a mirror that is the mirror.
            res.depth = travelled;
            depth_recorded = true;
        }

        if (s.unlit || bounce >= uni.limits.x) {
            break;
        }

        let b = bsdf_make(s, ns, ng);
        let wo = -dir;
        let shadow_origin = hp + ng_facing * eps;

        if (!bsdf_is_delta(b)) {
            res.radiance = res.radiance + direct_light_add(
                shadow_origin,
                hp,
                ns,
                ns_facing,
                wo,
                throughput,
                bounce,
                b,
                rng,
                eps,
            );
            if (s.subsurface > 0.0) {
                res.radiance = res.radiance + subsurface_walk(
                    mat,
                    s,
                    s.base_color,
                    hp,
                    ns,
                    ng,
                    ns_facing,
                    ng_facing,
                    wo,
                    throughput,
                    bounce,
                    rng,
                    eps,
                );
            }
        }

        let smp = bsdf_sample(b, wo, rng);
        if (!smp.valid || smp.pdf <= 0.0) {
            break;
        }
        let cos = abs(dot(smp.dir, ns));
        if (cos <= 0.0) {
            break;
        }
        throughput = throughput * smp.f * (cos / smp.pdf);
        let tmax = max(throughput.r, max(throughput.g, throughput.b));
        if (tmax <= 0.0) {
            break;
        }

        if (smp.transmitted) {
            if (front && s.attenuation_distance > 0.0) {
                let c = clamp(s.attenuation_color, vec3<f32>(1e-4), vec3<f32>(1.0));
                medium_sigma = -log(c) / s.attenuation_distance;
                in_medium = true;
            } else {
                in_medium = false;
            }
            origin = hp - ng_facing * eps;
        } else {
            origin = shadow_origin;
        }
        dir = smp.dir;
        prev_specular = smp.specular;
        prev_pdf = smp.pdf;
        prev_normal = ns_facing;
        camera_ray_flag = false;
        bounce = bounce + 1u;

        // Russian roulette: kill dim paths at random rather than truncating
        // every path at the same depth, and scale the survivors back up.
        if (bounce > uni.limits.y) {
            let q = clamp(tmax, 0.0, 0.95);
            if (q <= 0.0 || rand(rng) >= q) {
                break;
            }
            throughput = throughput / q;
        }
    }

    // A NaN in a film is permanent, so it is stopped at the one place it can
    // enter.
    res.radiance = select(vec3<f32>(0.0, 0.0, 0.0), max(res.radiance, vec3<f32>(0.0, 0.0, 0.0)),
        res.radiance.x == res.radiance.x && res.radiance.y == res.radiance.y && res.radiance.z == res.radiance.z);

    let fog_mode = u32(uni.fog_color.w);
    if (fog_mode > 0u) {
        let d = select(uni.render_params.x * 4.0, res.depth, depth_recorded);
        var vis = 1.0;
        if (fog_mode == 1u) {
            let span = max(uni.fog_params.y - uni.fog_params.x, 1e-6);
            vis = clamp((uni.fog_params.y - d) / span, 0.0, 1.0);
        } else if (fog_mode == 2u) {
            vis = exp(-uni.fog_params.z * uni.fog_params.z * d);
        }
        let fog = uni.fog_color.xyz * (1.0 - vis);
        res.radiance = res.radiance * vis + fog;
    }
    return res;
}

// Whether a pixel's estimate has settled to within the adaptive threshold.
//
// The same rule the CPU backend applies, and applied per sample rather than per
// dispatch for the same reason: the decision then depends only on what the pixel
// has accumulated, so an adaptive render comes out the same however it was
// batched.
fn adaptive_converged(color: vec3<f32>, m2: f32, n: f32) -> bool {
    let threshold = uni.adaptive.x;
    if (threshold <= 0.0 || n < max(uni.adaptive.y, 2.0)) {
        return false;
    }
    let mean = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722)) / n;
    // `m2` is Welford Σ(x−μ)²; variance of the mean is M₂/n².
    let variance = max(m2, 0.0) / (n * n);
    if (variance <= 0.0) {
        return true;
    }
    // Standard error of the mean, relative to the mean — with a floor, so a
    // near-black pixel is not held to a relative target it can never meet.
    return sqrt(variance) / max(mean, 1e-3) < threshold;
}

const WG_PIXELS: u32 = 64u;

var<workgroup> wg_budget: atomic<u32>;
var<workgroup> wg_active: array<u32, WG_PIXELS>;
var<workgroup> wg_active_count: u32;
var<workgroup> wg_pick: u32;
var<workgroup> wg_rng: u32;
var<workgroup> wg_per_remaining: array<u32, WG_PIXELS>;
var<workgroup> wg_participates: array<u32, WG_PIXELS>;

struct PixelState {
    color: vec3<f32>,
    alpha: f32,
    albedo: vec3<f32>,
    normal: vec3<f32>,
    depth: f32,
    depth_n: f32,
    taken: f32,
    m2: f32,
}

fn trace_one_sample(
    px: u32,
    py: u32,
    sample: u32,
    scramble: u32,
    stream: u32,
    state: ptr<function, PixelState>,
) {
    var smp: Sampler;
    smp.rng = hash32(stream ^ (sample * 26699u));
    smp.seed = scramble;
    smp.index = sample;
    sampler_set_bounce(&smp, 0u);
    let jitter = rand2(&smp);
    let lens = rand2(&smp);
    let shutter = rand2(&smp).x;
    let r = camera_ray(vec2<f32>(f32(px), f32(py)) + jitter, lens, shutter);
    let path = trace_path(r, &smp);
    let lum = dot(path.radiance, vec3<f32>(0.2126, 0.7152, 0.0722));
    let mean_old = select(
        0.0,
        dot((*state).color, vec3<f32>(0.2126, 0.7152, 0.0722)) / (*state).taken,
        (*state).taken > 0.0,
    );
    var delta = lum - mean_old;
    if ((*state).taken >= 2.0) {
        let std_sample = select(
            0.0,
            sqrt(max((*state).m2, 0.0) / max((*state).taken - 1.0, 1.0)),
            (*state).m2 > 0.0,
        );
        let cap = select(
            max(abs(mean_old), 1e-3) * 8.0 + 1e-3,
            max(std_sample * 4.0, abs(mean_old) * 0.25 + 1e-4),
            std_sample > 1e-6,
        );
        if (abs(delta) > cap) {
            delta = clamp(delta, -cap, cap);
        }
    }
    let lum_eff = mean_old + delta;
    let n_new = (*state).taken + 1.0;
    let mean_new = mean_old + delta / n_new;
    (*state).m2 = (*state).m2 + delta * (lum_eff - mean_new);
    (*state).color = (*state).color + path.radiance;
    (*state).alpha = (*state).alpha + path.alpha;
    (*state).albedo = (*state).albedo + path.albedo;
    (*state).normal = (*state).normal + path.normal;
    (*state).taken = n_new;
    if (path.depth < INF) {
        (*state).depth = (*state).depth + path.depth;
        (*state).depth_n = (*state).depth_n + 1.0;
    }
}

fn wg_pcg_step(state: u32) -> u32 {
    let s = state * 747796405u + 2891336453u;
    let word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (word >> 22u) ^ word;
}

fn redistribute_workgroup(
    lid: u32,
    wgid: vec3<u32>,
    count: u32,
    px: u32,
    py: u32,
    scramble: u32,
    stream: u32,
    participates: bool,
    state: ptr<function, PixelState>,
) {
    wg_participates[lid] = select(0u, 1u, participates);
    workgroupBarrier();

    if (lid == 0u) {
        var n = 0u;
        var budget = 0u;
        for (var i = 0u; i < WG_PIXELS; i = i + 1u) {
            if (wg_participates[i] != 0u) {
                wg_active[n] = i;
                n = n + 1u;
                wg_per_remaining[i] = count;
                budget = budget + count;
            } else {
                wg_per_remaining[i] = 0u;
            }
        }
        wg_active_count = n;
        atomicStore(&wg_budget, budget);
        wg_rng = hash32(uni.flags.x ^ uni.flags.y ^ wgid.x ^ wgid.y);
        wg_pick = 255u;
    }
    workgroupBarrier();

    // A uniform trip count, not a data-dependent break.
    //
    // Tint requires every `workgroupBarrier()` to sit in uniform control flow,
    // and a loop whose exit depends on `wg_pick` — workgroup memory written
    // under `if (lid == 0u)` — is not provably uniform, so every barrier inside
    // it is rejected and the module fails to compile on WebGPU. Moving the
    // break around does not help: the loop's *continuation* is the thing being
    // judged.
    //
    // The budget is `count` samples for each of `WG_PIXELS` pixels and a round
    // retires at most one, so that product bounds the rounds. `count` comes
    // from the uniform buffer, which is uniform by construction. Rounds after
    // the work runs out cost three barriers and nothing else.
    let max_rounds = WG_PIXELS * count;
    for (var round = 0u; round < max_rounds; round = round + 1u) {
        workgroupBarrier();
        if (lid == 0u) {
            wg_pick = 255u;
            let budget = atomicLoad(&wg_budget);
            let n_active = wg_active_count;
            if (budget > 0u && n_active > 0u) {
                wg_rng = wg_pcg_step(wg_rng);
                wg_pick = wg_active[wg_rng % n_active];
            }
        }
        workgroupBarrier();

        // The exit is taken at the END of the body, and the work below is
        // guarded instead. Breaking here put `workgroupBarrier()` after a
        // conditional exit whose condition comes from workgroup memory, which
        // Tint rejects outright:
        //
        //   error: 'workgroupBarrier' must only be called from uniform control
        //          flow
        //
        // The whole module then failed to compile on WebGPU. naga accepts it,
        // so this only ever showed up in a browser — as a black image, because
        // a kernel that will not compile still dispatches happily into nothing.
        let pick = wg_pick;
        if (pick != 255u) {
        if (lid == pick) {
            let remaining = wg_per_remaining[lid];
            if (remaining > 0u) {
                if (adaptive_converged((*state).color, (*state).m2, (*state).taken)) {
                    wg_per_remaining[lid] = 0u;
                    atomicAdd(&wg_budget, remaining);
                } else {
                    let sample = u32((*state).taken);
                    trace_one_sample(px, py, sample, scramble, stream, state);
                    wg_per_remaining[lid] = remaining - 1u;
                    atomicSub(&wg_budget, 1u);
                    if (adaptive_converged((*state).color, (*state).m2, (*state).taken)) {
                        let unused = wg_per_remaining[lid];
                        wg_per_remaining[lid] = 0u;
                        atomicAdd(&wg_budget, unused);
                    }
                }
            }
        }
        }
        // Reached unconditionally by every invocation in the workgroup.
        workgroupBarrier();

        if (pick != 255u) {
        if (lid == 0u) {
            var write = 0u;
            let n_active = wg_active_count;
            for (var i = 0u; i < n_active; i = i + 1u) {
                let idx = wg_active[i];
                if (wg_per_remaining[idx] > 0u) {
                    wg_active[write] = idx;
                    write = write + 1u;
                }
            }
            wg_active_count = write;
        }
        }
    }
}

@compute @workgroup_size(8, 8, 1)
fn trace(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let width = uni.dims.x;
    let height = uni.dims.y;
    var px: u32 = 0u;
    var py: u32 = 0u;
    var participates = false;
    if (uni.clip.z > 0u && uni.clip.w > 0u) {
        if (gid.x < uni.clip.z && gid.y < uni.clip.w) {
            px = gid.x + uni.clip.x;
            py = gid.y + uni.clip.y;
            participates = px < width && py < height;
        }
    } else {
        px = gid.x;
        py = gid.y;
        participates = px < width && py < height;
    }

    var state = PixelState(
        vec3<f32>(0.0, 0.0, 0.0),
        0.0,
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        0.0,
        0.0,
        0.0,
        0.0,
    );
    var base = 0u;
    if (participates) {
        let pixel = py * width + px;
        base = pixel * ACCUM_STRIDE;
        state = PixelState(
            vec3<f32>(accum[base], accum[base + 1u], accum[base + 2u]),
            accum[base + 3u],
            vec3<f32>(accum[base + 4u], accum[base + 5u], accum[base + 6u]),
            vec3<f32>(accum[base + 7u], accum[base + 8u], accum[base + 9u]),
            accum[base + 10u],
            accum[base + 11u],
            accum[base + 12u],
            accum[base + 13u],
        );
    }

    let count = uni.dims.w;
    let scramble = pixel_scramble(px, py);
    let stream = pcg_stream_tag(px, py);
    let sample0 = u32(state.taken);
    let redistribute = uni.render_params.y > 0.5 && uni.adaptive.x > 0.0;

    if (redistribute) {
        redistribute_workgroup(
            lid,
            wgid,
            count,
            px,
            py,
            scramble,
            stream,
            participates,
            &state,
        );
    } else if (participates) {
        for (var s = 0u; s < count; s = s + 1u) {
            if (adaptive_converged(state.color, state.m2, state.taken)) {
                break;
            }
            trace_one_sample(px, py, sample0 + s, scramble, stream, &state);
        }
    }

    if (participates) {
        accum[base] = state.color.r;
        accum[base + 1u] = state.color.g;
        accum[base + 2u] = state.color.b;
        accum[base + 3u] = state.alpha;
        accum[base + 4u] = state.albedo.r;
        accum[base + 5u] = state.albedo.g;
        accum[base + 6u] = state.albedo.b;
        accum[base + 7u] = state.normal.x;
        accum[base + 8u] = state.normal.y;
        accum[base + 9u] = state.normal.z;
        accum[base + 10u] = state.depth;
        accum[base + 11u] = state.depth_n;
        accum[base + 12u] = state.taken;
        accum[base + 13u] = state.m2;
    }
}
