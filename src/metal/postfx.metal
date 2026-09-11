// Post-processing for the Metal headless path — bloom, supersample downsample,
// and Catmull-Rom upscale. Matches the wgpu [`crate::renderer::bloom`] and
// [`crate::renderer::downsample`] behaviour (linear-light, float bloom).

#include <metal_stdlib>
using namespace metal;

struct BloomParams {
    float threshold;
    float radius;
    float dx;
    float dy;
};

struct DownsampleParams {
    uint factor;
    uint srgb;
    uint _pad0;
    uint _pad1;
};

struct SsaoParams {
    float near_z;
    float far_z;
    float radius;
    float strength;
    float tan_half_fov;
    float aspect;
    float _pad0;
    float _pad1;
};

struct UpscaleParams {
    float2 src_size;
    float2 dst_size;
};

struct CompositeParams {
    float strength;
    float2 bloom_size;
    float chroma_sat;
    float chroma_lift;
};

constant uint BLOOM_SHRINK = 4u;

// ------------------------------------------------------------------- bloom

struct FullscreenOut {
    float4 position [[position]];
    float2 uv;
};

vertex FullscreenOut vs_fullscreen(uint vid [[vertex_id]]) {
    const float2 uv = float2(float((vid << 1u) & 2u), float(vid & 2u));
    FullscreenOut out;
    out.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = uv;
    return out;
}

fragment float4 fs_bloom_threshold(
    FullscreenOut in [[stage_in]],
    texture2d<float, access::sample> src [[texture(0)]],
    constant BloomParams &p [[buffer(0)]])
{
    constexpr sampler s(coord::pixel, filter::nearest);
    const int2 gid = int2(in.position.xy);
    const uint2 in_dim = uint2(src.get_width(), src.get_height());

    float3 sum = float3(0.0);
    float n = 0.0;
    for (uint j = 0u; j < BLOOM_SHRINK; j++) {
        for (uint i = 0u; i < BLOOM_SHRINK; i++) {
            const uint2 c = uint2(gid.x * BLOOM_SHRINK + i, gid.y * BLOOM_SHRINK + j);
            if (c.x >= in_dim.x || c.y >= in_dim.y) { continue; }
            // sRGB source views decode to linear on sample.
            const float3 rgb = src.sample(s, float2(c) + float2(0.5)).rgb;
            const float lum = dot(rgb, float3(0.2126, 0.7152, 0.0722));
            const float over = max(lum - p.threshold, 0.0);
            const float w = over / max(lum, 1e-4);
            sum += rgb * w;
            n += 1.0;
        }
    }
    return float4(sum / max(n, 1.0), 1.0);
}

fragment float4 fs_bloom_blur(
    FullscreenOut in [[stage_in]],
    texture2d<float, access::sample> src [[texture(0)]],
    constant BloomParams &p [[buffer(0)]])
{
    constexpr sampler s(coord::pixel, filter::nearest);
    const int2 gid = int2(in.position.xy);
    const uint2 dim = uint2(src.get_width(), src.get_height());
    const float2 dir = float2(p.dx, p.dy) * p.radius;

    const float w[5] = {0.2270270, 0.1945946, 0.1216216, 0.0540540, 0.0162162};
    float3 acc = float3(0.0);
    for (uint k = 0u; k < 5u; k++) {
        const float weight = w[k];
        if (k == 0u) {
            const int2 c = int2(clamp(float2(gid), float2(0), float2(dim) - 1.0));
            acc += src.sample(s, float2(c) + float2(0.5)).rgb * weight;
        } else {
            const float2 o = dir * float(k);
            const int2 a = int2(clamp(float2(gid) + o, float2(0), float2(dim) - 1.0));
            const int2 b = int2(clamp(float2(gid) - o, float2(0), float2(dim) - 1.0));
            acc += (src.sample(s, float2(a) + float2(0.5)).rgb
                  + src.sample(s, float2(b) + float2(0.5)).rgb) * weight;
        }
    }
    return float4(acc, 1.0);
}

// ----------------------------------------------------------- downsample / upscale

fragment float4 fs_downsample(
    FullscreenOut in [[stage_in]],
    texture2d<float, access::sample> src [[texture(0)]],
    constant DownsampleParams &p [[buffer(0)]])
{
    constexpr sampler s(coord::pixel, filter::nearest);
    const int2 base = int2(in.position.xy) * int(p.factor);
    float4 acc = float4(0.0);
    // Average in the space the texture samples in. For sRGB attachments Metal
    // already returns linear; writing to an sRGB target re-encodes. Manual
    // encode/decode here would double-convert and crush highlights.
    (void)p.srgb;
    for (uint y = 0u; y < p.factor; y++) {
        for (uint x = 0u; x < p.factor; x++) {
            acc += src.sample(s, float2(base + int2(x, y)) + float2(0.5));
        }
    }
    return acc / float(p.factor * p.factor);
}

/// Keys cubic weight (Catmull-Rom, B=0, C=0.5).
inline float catmull_rom(float x) {
    x = abs(x);
    if (x <= 1.0) {
        return 1.5 * x * x * x - 2.5 * x * x + 1.0;
    }
    if (x <= 2.0) {
        return -0.5 * x * x * x + 2.5 * x * x - 4.0 * x + 2.0;
    }
    return 0.0;
}

/// 16-tap Catmull-Rom in linear light. Destination UV covers the full source.
fragment float4 fs_upscale(
    FullscreenOut in [[stage_in]],
    texture2d<float, access::sample> src [[texture(0)]],
    constant UpscaleParams &p [[buffer(0)]])
{
    constexpr sampler nearest(coord::normalized, filter::nearest, address::clamp_to_edge);
    const float2 tex_size = p.src_size;
    // Map destination UV → source texel space (full image, not a corner crop).
    const float2 pos = in.uv * tex_size - 0.5;
    const float2 f = fract(pos);
    const int2 base = int2(floor(pos));

    float4 acc = float4(0.0);
    float wsum = 0.0;
    for (int j = -1; j <= 2; j++) {
        const float wy = catmull_rom(float(j) - f.y);
        for (int i = -1; i <= 2; i++) {
            const float wx = catmull_rom(float(i) - f.x);
            const float w = wx * wy;
            if (w == 0.0) { continue; }
            const int2 c = clamp(base + int2(i, j), int2(0), int2(tex_size) - 1);
            const float2 uv = (float2(c) + 0.5) / tex_size;
            acc += src.sample(nearest, uv) * w;
            wsum += w;
        }
    }
    return acc / max(wsum, 1e-6);
}

fragment float4 fs_bloom_composite(
    FullscreenOut in [[stage_in]],
    texture2d<float, access::sample> src [[texture(0)]],
    texture2d<float, access::sample> bloom [[texture(1)]],
    constant CompositeParams &p [[buffer(0)]])
{
    constexpr sampler nearest(filter::nearest, address::clamp_to_edge);
    constexpr sampler linear(filter::linear, address::clamp_to_edge);
    float3 rgb = src.sample(nearest, in.uv).rgb;
    // Bloom is a lower-res linear buffer covering the same image extent.
    rgb += bloom.sample(linear, in.uv).rgb * p.strength;
    // GPU chroma punch — replaces a full-frame CPU walk on readback.
    const float luma = dot(rgb, float3(0.2126, 0.7152, 0.0722));
    if (luma > 0.012) {
        const float maxc = max(rgb.r, max(rgb.g, rgb.b));
        const float minc = min(rgb.r, min(rgb.g, rgb.b));
        const float sat = maxc > 1e-5 ? (maxc - minc) / maxc : 0.0;
        const float target = clamp(sat * p.chroma_sat + p.chroma_lift, 0.0, 1.0);
        if (sat > 1e-5) {
            const float t = target / sat;
            rgb = maxc + (rgb - maxc) * t;
        }
    }
    return float4(rgb, 1.0);
}


// ----------------------------------------------------------------------- ssao
//
// Depth-only ambient occlusion, ported from the wgpu compute pass in
// `renderer/ssao.rs`. Same algorithm, same constants; a fragment pass rather
// than a compute one because that is what this backend's post chain is built
// from. See that module for why each piece is shaped the way it is — in
// particular why the normal is reconstructed with nearest-side derivatives and
// why depth is linearised before anything compares it.

static inline float ssao_linear_depth(float d, constant SsaoParams &p) {
    if (d >= 1.0) { return p.far_z; }
    return (p.near_z * p.far_z) / max(p.far_z - d * (p.far_z - p.near_z), 1e-6);
}

// View-space position from a pixel and its linear depth. Only ratios matter for
// a normal, so this needs the projection's shape and not the matrix.
static inline float3 ssao_view_pos(
    int2 c,
    int2 dim,
    depth2d<float, access::sample> depth,
    constant SsaoParams &p)
{
    constexpr sampler ds(coord::pixel, filter::nearest, address::clamp_to_edge);
    const int2 cc = clamp(c, int2(0), dim - 1);
    const float z = ssao_linear_depth(depth.sample(ds, float2(cc) + float2(0.5)), p);
    const float2 uv = (float2(cc) + 0.5) / float2(dim) * 2.0 - 1.0;
    // `position` and the WGSL `textureLoad` share a top-left origin, so the
    // -uv.y that puts +Y up in view space carries over unchanged.
    return float3(uv.x * p.aspect * p.tan_half_fov * z, -uv.y * p.tan_half_fov * z, z);
}

fragment float4 fs_ssao(
    FullscreenOut in [[stage_in]],
    texture2d<float, access::sample> src [[texture(0)]],
    depth2d<float, access::sample> depth [[texture(1)]],
    constant SsaoParams &p [[buffer(0)]])
{
    constexpr sampler cs(coord::pixel, filter::nearest, address::clamp_to_edge);
    const int2 c = int2(in.position.xy);
    const int2 dim = int2(int(depth.get_width()), int(depth.get_height()));
    // Sampled by `position`, like fs_downsample — sampling by `uv` here would
    // flip the frame against the rest of the chain.
    const float4 color = src.sample(cs, float2(c) + float2(0.5));

    const float3 p0 = ssao_view_pos(c, dim, depth, p);
    // Background has no crevices, and the linearisation is least trustworthy at
    // the far plane — leave it rather than invent occlusion there.
    if (p0.z >= p.far_z * 0.999) { return color; }

    // NEAREST-SIDE DERIVATIVES: a central difference across a silhouette returns
    // a normal belonging to neither surface, and silhouettes are where the
    // creases are.
    const float3 xr = ssao_view_pos(c + int2(1, 0), dim, depth, p);
    const float3 xl = ssao_view_pos(c - int2(1, 0), dim, depth, p);
    const float3 yd = ssao_view_pos(c + int2(0, 1), dim, depth, p);
    const float3 yu = ssao_view_pos(c - int2(0, 1), dim, depth, p);
    float3 dx = xr - p0;
    if (abs(xl.z - p0.z) < abs(xr.z - p0.z)) { dx = p0 - xl; }
    float3 dy = yd - p0;
    if (abs(yu.z - p0.z) < abs(yd.z - p0.z)) { dy = p0 - yu; }
    const float3 n = normalize(cross(dx, dy));

    // Screen-space radius shrinks with distance so the effect keeps a fixed size
    // in the WORLD rather than a fixed number of pixels.
    const float px = clamp(
        p.radius / p0.z * float(dim.y) * 0.5 / max(p.tan_half_fov, 1e-4), 2.0, 64.0);

    float occ = 0.0;
    for (int i = 0; i < 12; i++) {
        const float a = float(i) * 2.39996;                 // golden angle
        const float r = px * sqrt((float(i) + 0.5) / 12.0); // uniform over the disc
        const int2 sc = c + int2(float2(cos(a), sin(a)) * r);
        const float3 ps = ssao_view_pos(sc, dim, depth, p);
        const float3 v = ps - p0;
        const float d = length(v);
        if (d < 1e-4) { continue; }
        // A sample occludes only if it sits IN FRONT of this surface; that dot
        // is what stops a slanted plane from shadowing itself.
        const float cosang = dot(n, v / d);
        // Falloff, so a distant foreground object does not darken everything
        // behind it: that is a silhouette, not a contact.
        const float fall = clamp(1.0 - d / p.radius, 0.0, 1.0);
        occ += max(cosang, 0.0) * fall;
    }
    const float ao = clamp(1.0 - p.strength * (occ / 12.0), 0.0, 1.0);
    // Multiplied in linear light: an sRGB attachment hands back linear on sample
    // and re-encodes on write, so this is the right space to darken in.
    return float4(color.rgb * ao, color.a);
}


// -------------------------------------------------------------------- tonemap
//
// The last step of an HDR chain: take linear values that were free to exceed
// 1.0 during accumulation and fit them into a display range.
//
// This matters most for additive rendering. With an 8-bit target, a pixel that
// accumulates past 1.0 is clamped *while blending*, so the excess is lost
// before any grade can see it — dense regions flatten to white and take their
// hue with them. Accumulating in float and compressing here keeps that
// structure, and keeps colour in the highlights instead of washing out.

struct TonemapParams {
    float exposure;
    float white;      // luminance that maps to 1.0
    float _pad0;
    float _pad1;
};

// ACES filmic approximation (Narkowicz 2015). Cheap, and its toe/shoulder are
// close enough to the full fit for display use.
static inline float3 aces(float3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

fragment float4 fs_tonemap(
    FullscreenOut in [[stage_in]],
    texture2d<float, access::sample> src [[texture(0)]],
    constant TonemapParams &p [[buffer(0)]])
{
    constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_edge);
    const float4 c = src.sample(s, in.position.xy);
    float3 hdr = max(c.rgb, 0.0) * p.exposure;
    if (p.white > 0.0) {
        hdr /= p.white;
    }
    // Written to an sRGB attachment, which re-encodes on write — so this stays
    // linear and must not apply a gamma itself.
    return float4(aces(hdr), c.a);
}
