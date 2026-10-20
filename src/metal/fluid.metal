// The 2D fluid solver, in Metal Shading Language.
//
// A transcription of `src/fluid/shader.rs`, function for function and constant
// for constant, so that `metal_fluid_matches_wgpu` can hold the two backends
// to the same picture. Where the two languages differ the WGSL is the
// reference: this is the port, not the other way round.
//
// `vs_fluid` is deliberately not `vs_fullscreen` from `postfx.metal`. That one
// hands out `uv` already flipped, because the passes around it sample a
// framebuffer; this one leaves uv as the corner value, with (0,0) at the top
// left, which is the convention the whole fluid is written in. Reusing the
// flipped one would mirror the simulation — still a fluid, and not the same
// fluid as the other backend's.

#include <metal_stdlib>
using namespace metal;

struct FluidParams {
    float2 texel_size;
    float2 point;
    float2 dye_texel_size;
    float  dt;
    float  dissipation;
    float  curl;
    float  aspect_ratio;
    float2 pad0;
    float4 color;
    float4 curve;
    float  radius;
    float  value;
    uint   flags;
    uint   pad1;
};

constant uint FLAG_SHADING = 1u;
constant uint FLAG_BLOOM   = 2u;
constant uint FLAG_SUNRAYS = 4u;
constant uint FLAG_ENCODE  = 8u;

struct FluidVertex {
    float4 position [[position]];
    float2 uv;
    float2 vl;
    float2 vr;
    float2 vt;
    float2 vb;
};

vertex FluidVertex vs_fluid(
    uint vid [[vertex_id]],
    constant FluidParams& u [[buffer(0)]]
) {
    const float2 c = float2(float((vid << 1u) & 2u), float(vid & 2u));
    FluidVertex out;
    out.position = float4(c.x * 2.0 - 1.0, 1.0 - c.y * 2.0, 0.0, 1.0);
    out.uv = c;
    out.vl = c - float2(u.texel_size.x, 0.0);
    out.vr = c + float2(u.texel_size.x, 0.0);
    out.vt = c + float2(0.0, u.texel_size.y);
    out.vb = c - float2(0.0, u.texel_size.y);
    return out;
}

constexpr sampler fluid_sampler(
    coord::normalized, address::clamp_to_edge, filter::linear);

// --------------------------------------------------------------- utility

fragment float4 fs_fluid_clear(
    FluidVertex in [[stage_in]],
    constant FluidParams& u [[buffer(0)]],
    texture2d<float> tex_a [[texture(0)]]
) {
    return u.value * tex_a.sample(fluid_sampler, in.uv);
}

static float3 undo_srgb(float3 c, uint flags) {
    if ((flags & FLAG_ENCODE) == 0u) {
        return c;
    }
    const float3 m = clamp(c, 0.0, 1.0);
    const float3 lo = m / 12.92;
    const float3 hi = pow((m + 0.055) / 1.055, 2.4);
    return select(hi, lo, m <= 0.04045);
}

fragment float4 fs_fluid_color(
    FluidVertex in [[stage_in]],
    constant FluidParams& u [[buffer(0)]]
) {
    return float4(undo_srgb(u.color.rgb, u.flags), u.color.a);
}

fragment float4 fs_fluid_splat(
    FluidVertex in [[stage_in]],
    constant FluidParams& u [[buffer(0)]],
    texture2d<float> tex_a [[texture(0)]]
) {
    float2 p = in.uv - u.point;
    p.x *= u.aspect_ratio;
    const float3 blob = exp(-dot(p, p) / u.radius) * u.color.rgb;
    const float3 base = tex_a.sample(fluid_sampler, in.uv).rgb;
    return float4(base + blob, 1.0);
}

// ---------------------------------------------------------------- solver

fragment float4 fs_fluid_advection(
    FluidVertex in [[stage_in]],
    constant FluidParams& u [[buffer(0)]],
    texture2d<float> tex_a [[texture(0)]],
    texture2d<float> tex_b [[texture(1)]]
) {
    const float2 coord =
        in.uv - u.dt * tex_a.sample(fluid_sampler, in.uv).xy * u.texel_size;
    const float4 result = tex_b.sample(fluid_sampler, coord);
    return result / (1.0 + u.dissipation * u.dt);
}

fragment float4 fs_fluid_divergence(
    FluidVertex in [[stage_in]],
    constant FluidParams& u [[buffer(0)]],
    texture2d<float> tex_a [[texture(0)]]
) {
    float l = tex_a.sample(fluid_sampler, in.vl).x;
    float r = tex_a.sample(fluid_sampler, in.vr).x;
    float t = tex_a.sample(fluid_sampler, in.vt).y;
    float b = tex_a.sample(fluid_sampler, in.vb).y;
    const float2 c = tex_a.sample(fluid_sampler, in.uv).xy;
    // A free-slip wall, as in the WGSL: outside the domain the normal
    // component is mirrored, so the boundary neither leaks nor drags.
    if (in.vl.x < 0.0) { l = -c.x; }
    if (in.vr.x > 1.0) { r = -c.x; }
    if (in.vt.y > 1.0) { t = -c.y; }
    if (in.vb.y < 0.0) { b = -c.y; }
    return float4(0.5 * (r - l + t - b), 0.0, 0.0, 1.0);
}

fragment float4 fs_fluid_curl(
    FluidVertex in [[stage_in]],
    texture2d<float> tex_a [[texture(0)]]
) {
    const float l = tex_a.sample(fluid_sampler, in.vl).y;
    const float r = tex_a.sample(fluid_sampler, in.vr).y;
    const float t = tex_a.sample(fluid_sampler, in.vt).x;
    const float b = tex_a.sample(fluid_sampler, in.vb).x;
    return float4(0.5 * (r - l - t + b), 0.0, 0.0, 1.0);
}

fragment float4 fs_fluid_vorticity(
    FluidVertex in [[stage_in]],
    constant FluidParams& u [[buffer(0)]],
    texture2d<float> tex_a [[texture(0)]],
    texture2d<float> tex_b [[texture(1)]]
) {
    const float l = tex_b.sample(fluid_sampler, in.vl).x;
    const float r = tex_b.sample(fluid_sampler, in.vr).x;
    const float t = tex_b.sample(fluid_sampler, in.vt).x;
    const float b = tex_b.sample(fluid_sampler, in.vb).x;
    const float c = tex_b.sample(fluid_sampler, in.uv).x;

    float2 force = 0.5 * float2(abs(t) - abs(b), abs(r) - abs(l));
    force /= length(force) + 0.0001;
    force *= u.curl * c;
    force.y = -force.y;

    float2 velocity = tex_a.sample(fluid_sampler, in.uv).xy + force * u.dt;
    // The clamp is load-bearing: the field is half float, and one unbounded
    // texel reaches infinity and then NaN.
    velocity = clamp(velocity, -1000.0, 1000.0);
    return float4(velocity, 0.0, 1.0);
}

fragment float4 fs_fluid_pressure(
    FluidVertex in [[stage_in]],
    constant FluidParams& u [[buffer(0)]],
    texture2d<float> tex_a [[texture(0)]],
    texture2d<float> tex_b [[texture(1)]]
) {
    const float l = tex_a.sample(fluid_sampler, in.vl).x;
    const float r = tex_a.sample(fluid_sampler, in.vr).x;
    const float t = tex_a.sample(fluid_sampler, in.vt).x;
    const float b = tex_a.sample(fluid_sampler, in.vb).x;
    const float c = tex_a.sample(fluid_sampler, in.uv).x;
    const float divergence = tex_b.sample(fluid_sampler, in.uv).x;
    const float jacobi = (l + r + b + t - divergence) * 0.25;
    return float4(mix(c, jacobi, u.value), 0.0, 0.0, 1.0);
}

fragment float4 fs_fluid_gradient(
    FluidVertex in [[stage_in]],
    constant FluidParams& u [[buffer(0)]],
    texture2d<float> tex_a [[texture(0)]],
    texture2d<float> tex_b [[texture(1)]]
) {
    const float l = tex_a.sample(fluid_sampler, in.vl).x;
    const float r = tex_a.sample(fluid_sampler, in.vr).x;
    const float t = tex_a.sample(fluid_sampler, in.vt).x;
    const float b = tex_a.sample(fluid_sampler, in.vb).x;
    const float2 velocity =
        tex_b.sample(fluid_sampler, in.uv).xy - u.value * float2(r - l, t - b);
    return float4(velocity, 0.0, 1.0);
}

// --------------------------------------------------------------- display

fragment float4 fs_fluid_display(
    FluidVertex in [[stage_in]],
    constant FluidParams& u [[buffer(0)]],
    texture2d<float> tex_a [[texture(0)]]
) {
    float3 c = tex_a.sample(fluid_sampler, in.uv).rgb;

    if ((u.flags & FLAG_SHADING) != 0u) {
        // Light the dye as though its brightness were a height field. There
        // is no geometry here at all, and it is most of what makes the fluid
        // read as volume.
        const float3 lc = tex_a.sample(fluid_sampler, in.vl).rgb;
        const float3 rc = tex_a.sample(fluid_sampler, in.vr).rgb;
        const float3 tc = tex_a.sample(fluid_sampler, in.vt).rgb;
        const float3 bc = tex_a.sample(fluid_sampler, in.vb).rgb;
        const float dx = length(rc) - length(lc);
        const float dy = length(tc) - length(bc);
        const float3 n = normalize(float3(dx, dy, length(u.texel_size)));
        c *= clamp(dot(n, float3(0.0, 0.0, 1.0)) + 0.7, 0.7, 1.0);
    }

    const float a = max(c.r, max(c.g, c.b));
    return float4(undo_srgb(c, u.flags), a);
}

// ------------------------------------------------------- the frame-rate read-out
//
// A transcription of `src/fluid/read_out.rs`, which the wgpu module shares
// between its two solvers. The two constants below are that module's
// `Frames::LEN` and `Frames::SCALE_MS`; they are literals here rather than
// substituted in, because `FLUID_SHADER_SOURCE` is public and has to compile
// as it stands, and `the_msl_read_out_agrees_about_the_row` is what keeps
// them in step with the Rust.

constant int   FLUID_HISTORY_COLUMNS  = 64;
constant float FLUID_HISTORY_SCALE_MS = 250.0;

// Rows top-to-bottom, three bits each, leftmost pixel in the low bit.
static uint fluid_glyph_bits(int d) {
    switch (d) {
        case 0: return 31599u;
        case 1: return 29850u;
        case 2: return 29671u;
        case 3: return 31207u;
        case 4: return 18925u;
        case 5: return 31183u;
        case 6: return 31695u;
        case 7: return 18727u;
        case 8: return 31727u;
        default: return 31215u;
    }
}

static bool fluid_glyph_on(int d, int x, int y) {
    if (x < 0 || x > 2 || y < 0 || y > 4) {
        return false;
    }
    return ((fluid_glyph_bits(d) >> uint(y * 3 + x)) & 1u) == 1u;
}

// Up to three digits at a fixed pixel size in the top left. Fixed, because a
// read-out that grew with the window would be a banner on a phone and a
// watermark on a desktop.
static float fluid_read_out(float2 clip, float value) {
    const int scale = 3;
    const int pad = 10;
    const int2 cell = (int2(clip) - int2(pad, pad)) / scale;
    if (cell.x < 0 || cell.x > 11 || cell.y < 0 || cell.y > 4) {
        return 0.0;
    }
    const int shown = int(clamp(round(value), 0.0, 999.0));
    const int column = cell.x / 4;
    // Not `pow`: ten squared comes back as 99.99999 in float and truncates
    // to 99, so 999 divides to ten and the first digit reads zero.
    int tens = 1;
    if (column == 0) {
        tens = 100;
    } else if (column == 1) {
        tens = 10;
    }
    const int digit = (shown / tens) % 10;
    // Leading zeros are not drawn: "7" rather than "007".
    const bool leading = (column == 0 && shown < 100) || (column == 1 && shown < 10);
    if (leading || !fluid_glyph_on(digit, cell.x % 4, cell.y)) {
        return 0.0;
    }
    return 0.85;
}

// The last sixty-four frames, as a bar a frame, under the number. See the
// WGSL for why the height is twice the *current* frame time rather than
// twice the refresh target.
static float4 fluid_chart(
    float2 clip,
    texture2d<float> hist,
    float scale_ms,
    float target_ms
) {
    const int pad = 10;
    const int top = 29;
    const int height = 18;
    const int columns = FLUID_HISTORY_COLUMNS;
    const int2 px = int2(clip) - int2(pad, top);
    if (px.x < 0 || px.x >= columns * 2 || px.y < 0 || px.y >= height) {
        return float4(0.0);
    }
    // Fetched by index, not sampled: a bar is one whole texel and there is
    // nothing to interpolate.
    const float ms = hist.read(uint2(uint(px.x / 2), 0u)).x * FLUID_HISTORY_SCALE_MS;
    const int up = height - 1 - px.y;
    const float full = max(scale_ms, 0.001);
    // The target, where it falls — off the top of the chart when the fluid
    // is running at less than half the rate it is aiming for, which is
    // itself worth seeing.
    const int line = int(target_ms / full * float(height));
    if (up == line && line < height) {
        return float4(1.0, 1.0, 1.0, 0.25);
    }
    if (ms <= 0.0) {
        return float4(0.0);
    }
    const int bar = int(clamp(ms / full, 0.0, 1.0) * float(height));
    if (up >= bar) {
        return float4(0.0);
    }
    if (ms > target_ms) {
        return float4(1.0, 0.45, 0.3, 0.75);
    }
    return float4(0.65, 0.85, 1.0, 0.5);
}

// `value` is the number, `radius` the milliseconds the chart is full at and
// `curl` the refresh target — the same three fields the WGSL reads, so the
// uniform block stays shared.
//
// The one line that is not a transcription is the return: the wgpu overlay
// pipeline blends straight alpha and every blended pipeline here is
// premultiplied, so the colour is premultiplied in the shader instead.
// `src * srcAlpha + dst * (1 - srcAlpha)` and `src' * 1 + dst * (1 -
// srcAlpha)` with `src' = src * srcAlpha` are the same arithmetic, which is
// what `the_two_read_outs_agree` checks rather than assumes.
fragment float4 fs_fluid_overlay(
    FluidVertex in [[stage_in]],
    constant FluidParams& u [[buffer(0)]],
    texture2d<float> tex_a [[texture(0)]]
) {
    const float a = fluid_read_out(in.position.xy, u.value);
    if (a > 0.0) {
        return float4(float3(1.0) * a, a);
    }
    const float4 bar = fluid_chart(in.position.xy, tex_a, u.radius, u.curl);
    if (bar.a <= 0.0) {
        discard_fragment();
    }
    return float4(bar.rgb * bar.a, bar.a);
}
