// Post-processing for the Metal headless path — bloom, supersample downsample,
// and Catmull-Rom upscale. Matches the wgpu [`crate::renderer::bloom`] and
// [`crate::renderer::downsample`] behaviour (linear-light, float bloom).

#include <metal_stdlib>
using namespace metal;

struct DownsampleParams {
    uint factor;
    uint srgb;
    // 0 = box (the plain block average), 1 = Mitchell-Netravali radius 2.
    // Mirrors `DownsampleFilter` on the wgpu path; see the module docs there
    // for why the box is the one that aliases.
    uint filter;
    uint _pad0;
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

// Scalars, not a `float2`.
//
// Metal aligns `float2` to eight bytes and Rust aligns `[f32; 2]` to four, so
// with a single `float` in front of it the two languages disagreed about where
// everything after it lived: the shader read `chroma_sat` out of the bytes Rust
// had written `chroma_lift` into. A saturation scale of zero turns the chroma
// step below into `rgb = maxc`, which is a perfectly grey frame from a shader
// whose arithmetic is a no-op when its parameters are right.
//
// Keeping every member four-byte aligned removes the trap rather than papering
// over it with padding that the next member to be added would have to respect.
struct CompositeParams {
    float strength;
    float bloom_w;
    float bloom_h;
    float chroma_sat;
    float chroma_lift;
};

// ------------------------------------------------------------------- bloom

struct FullscreenOut {
    float4 position [[position]];
    float2 uv;
};

/// A full-screen triangle, with `uv` in the framebuffer's orientation.
///
/// The `1.0 - y` is the point of this comment. Clip space has +y up and the
/// framebuffer has y down, so the vertex whose clip y is +1 lands on the *top*
/// row — while a texture coordinate of v = 1 is the *bottom* of the source.
/// Handing `position * 0.5 + 0.5` out as `uv` therefore samples every pass that
/// uses it upside down, and passes that sample by `in.position.xy` instead are
/// unaffected, so a chain mixing the two flips only some of its stages.
///
/// That is not hypothetical: bloom's composite samples by `uv` and the resolve
/// by `position`, and the phone render this crate exists for came out inverted
/// the moment bloom was switched on — status bar along the bottom edge —
/// while every orientation test passed, because none of them had enabled it.
///
/// Flipping here rather than at the three call sites means the next pass to
/// sample by `uv` is right without having to know any of this.
vertex FullscreenOut vs_fullscreen(uint vid [[vertex_id]]) {
    const float2 uv = float2(float((vid << 1u) & 2u), float(vid & 2u));
    FullscreenOut out;
    out.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = float2(uv.x, 1.0 - uv.y);
    return out;
}

// ----------------------------------------------------------- downsample / upscale

/// Mitchell-Netravali with the B = C = 1/3 the paper recommends, on a radius of
/// two output pixels.
inline float mitchell(float x) {
    x = abs(x);
    const float B = 1.0 / 3.0;
    const float C = 1.0 / 3.0;
    if (x < 1.0) {
        return ((12.0 - 9.0 * B - 6.0 * C) * x * x * x
              + (-18.0 + 12.0 * B + 6.0 * C) * x * x
              + (6.0 - 2.0 * B)) / 6.0;
    }
    if (x < 2.0) {
        return ((-B - 6.0 * C) * x * x * x
              + (6.0 * B + 30.0 * C) * x * x
              + (-12.0 * B - 48.0 * C) * x
              + (8.0 * B + 24.0 * C)) / 6.0;
    }
    return 0.0;
}

fragment float4 fs_downsample(
    FullscreenOut in [[stage_in]],
    texture2d<float, access::sample> src [[texture(0)]],
    constant DownsampleParams &p [[buffer(0)]])
{
    constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_edge);
    const int2 base = int2(in.position.xy) * int(p.factor);
    const float f = float(p.factor);
    float4 acc = float4(0.0);
    // Average in the space the texture samples in. For sRGB attachments Metal
    // already returns linear; writing to an sRGB target re-encodes. Manual
    // encode/decode here would double-convert and crush highlights.
    (void)p.srgb;
    if (p.filter == 0u) {
        for (uint y = 0u; y < p.factor; y++) {
            for (uint x = 0u; x < p.factor; x++) {
                acc += src.sample(s, float2(base + int2(x, y)) + float2(0.5));
            }
        }
        return acc / (f * f);
    }
    // A box one output pixel wide passes much of what it was meant to remove,
    // and that folds back as a staircase on any shallow edge — adding samples
    // inside it does not help, because the support is what is wrong. Reach two
    // output pixels either side instead, weighting by distance from the output
    // pixel's centre. Separable, but taken as a product here: one pass, no
    // intermediate, and this path has no HDR scratch to put one in.
    const int half_taps = int(ceil(2.0 * f));
    const int first = -half_taps + int(p.factor) / 2;
    float wsum = 0.0;
    for (int dy = 0; dy < 2 * half_taps; dy++) {
        // Source-sample centre minus output-pixel centre, in output pixels.
        const float ty = (float(dy + first) + 0.5 - f * 0.5) / f;
        const float wy = mitchell(ty);
        if (wy == 0.0) { continue; }
        for (int dx = 0; dx < 2 * half_taps; dx++) {
            const float tx = (float(dx + first) + 0.5 - f * 0.5) / f;
            const float w = wy * mitchell(tx);
            if (w == 0.0) { continue; }
            const int2 at = base + int2(dx + first, dy + first);
            acc += src.sample(s, float2(at) + float2(0.5)) * w;
            wsum += w;
        }
    }
    // The sampler clamps to the edge, so the weights still sum to one there and
    // the border neither darkens nor brightens.
    return acc / wsum;
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
// Depth-only ambient occlusion: the shared `ssao_*` functions, which the wgpu
// compute pass in `renderer/ssao.rs` and its WebGL2 raster twin call too — see
// them for why each piece is shaped the way it is. This pass only fetches
// depth, at full resolution, as a fragment pass because that is what this
// backend's post chain is built from.

static inline float3 ssao_view_pos_at(
    int2 c,
    int2 dim,
    depth2d<float, access::sample> depth,
    constant SsaoParams &p)
{
    constexpr sampler ds(coord::pixel, filter::nearest, address::clamp_to_edge);
    const int2 cc = clamp(c, int2(0), dim - 1);
    // This backend's depth is never reversed.
    const float z = ssao_linear_depth(depth.sample(ds, float2(cc) + float2(0.5)), p.near_z, p.far_z, false);
    return ssao_view_pos(c, uint2(dim), z, p.aspect, p.tan_half_fov);
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

    const float3 p0 = ssao_view_pos_at(c, dim, depth, p);
    // Background has no crevices, and the linearisation is least trustworthy at
    // the far plane — leave it rather than invent occlusion there.
    if (p0.z >= p.far_z * 0.999) { return color; }
    const float3 n = ssao_normal(p0,
        ssao_view_pos_at(c + int2(1, 0), dim, depth, p), ssao_view_pos_at(c - int2(1, 0), dim, depth, p),
        ssao_view_pos_at(c + int2(0, 1), dim, depth, p), ssao_view_pos_at(c - int2(0, 1), dim, depth, p));
    const float px = ssao_radius_px(p.radius, p0.z, float(dim.y), p.tan_half_fov);
    float occ = 0.0;
    for (int i = 0; i < 12; i++) {
        occ += ssao_occlusion(n, p0, ssao_view_pos_at(ssao_tap(c, i, px), dim, depth, p), p.radius);
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

struct FilmParams {
    float vignette;
    float grain;
    float seed;      // changes per frame, so the grain moves
    float _pad0;
};

/// Vignette and grain: the shared `film_vignette` and `film_grain`, which the
/// wgpu path's `rgb_pack.rs` calls too, so the film look does not depend on
/// the backend.
///
/// Both belong in a pass and not on the CPU: they are per-pixel work over the
/// whole frame, and a CPU walk of an 8K image costs more than everything else
/// the grade does.
///
/// The attachment is sRGB, so `c` is linear: the vignette multiplies it as it
/// stands, and the grain goes onto the encoded value — where "mid-tone" means
/// what it does on the wgpu path — and comes back out of it.
fragment float4 fs_film(
    FullscreenOut in [[stage_in]],
    texture2d<float, access::sample> src [[texture(0)]],
    constant FilmParams &p [[buffer(0)]])
{
    constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_edge);
    const float4 c = src.sample(s, in.position.xy);
    float3 rgb = c.rgb;
    const uint2 xy = uint2(in.position.xy);
    if (p.vignette > 0.0) {
        rgb *= film_vignette(xy, float2(src.get_width(), src.get_height()), p.vignette);
    }
    if (p.grain > 0.0) {
        rgb = srgb_decode(film_grain(srgb_encode(rgb), xy, p.seed, p.grain));
    }
    return float4(rgb, c.a);
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

// A straight-alpha RGBA overlay (text, panels, captions) over a finished
// frame, one overlay texel per target pixel. The texture is sRGB, so the sample
// is linear light; returned premultiplied for the `One, OneMinusSrcAlpha`
// fullscreen blend, which makes it the same blend as `SrcAlpha,
// OneMinusSrcAlpha` on the straight colour — the wgpu overlay pass's.
fragment float4 fs_overlay(
    FullscreenOut in [[stage_in]],
    texture2d<float, access::sample> src [[texture(0)]])
{
    constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_edge);
    const float4 c = src.sample(s, in.position.xy);
    return float4(c.rgb * c.a, c.a);
}

// ------------------------------------------------------------------ video YUV

// A decoder's YUV planes into a video texture: the twin of `fs` in
// `renderer/yuv.rs`, line for line, so both backends show the same colours.
struct YuvParams {
    uint4 dims;   // luma w, h; chroma w, h
    uint4 mode;   // chroma halved across, halved down, turned over, luma only
    uint4 out;    // source pixels a target pixel, sRGB target, target height, interleaved
    float4 k0;    // luma offset, luma scale, chroma midpoint
    float4 k1;    // Cr→R, Cb→G, Cr→G, Cb→B
};

static inline float3 yuv_to_linear(float3 c) {
    return select(pow((c + 0.055) / 1.055, float3(2.4)), c / 12.92, c <= 0.04045);
}

static inline float3 yuv_to_srgb(float3 c) {
    return select(1.055 * pow(c, float3(1.0 / 2.4)) - 0.055, c * 12.92, c <= 0.0031308);
}

// `channel` picks Cb or Cr out of an interleaved (NV12) plane; 0 otherwise.
static inline float yuv_chroma(texture2d<uint, access::read> plane, constant YuvParams &p, uint x, uint y, uint channel) {
    const uint cw = p.dims.z;
    const uint ch = p.dims.w;
    uint r0 = min(y, ch - 1u);
    uint r1 = r0;
    float wr0 = 4.0;
    float wr1 = 0.0;
    if (p.mode.y == 1u) {
        r0 = min(y / 2u, ch - 1u);
        r1 = (y % 2u == 0u) ? (r0 == 0u ? 0u : r0 - 1u) : min(r0 + 1u, ch - 1u);
        wr0 = 3.0;
        wr1 = 1.0;
    }
    uint c0 = min(x, cw - 1u);
    uint c1 = c0;
    float wc0 = 2.0;
    float wc1 = 0.0;
    if (p.mode.x == 1u) {
        c0 = x / 2u;
        c1 = c0;
        if (x % 2u == 1u) {
            c1 = min(c0 + 1u, cw - 1u);
            wc0 = 1.0;
            wc1 = 1.0;
        }
    }
    const float sum = wr0 * (wc0 * float(plane.read(uint2(c0, r0))[channel]) + wc1 * float(plane.read(uint2(c1, r0))[channel]))
        + wr1 * (wc0 * float(plane.read(uint2(c0, r1))[channel]) + wc1 * float(plane.read(uint2(c1, r1))[channel]));
    return sum / 8.0 - p.k0.z;
}

static inline float3 yuv_rgb_at(
    texture2d<uint, access::read> yp,
    texture2d<uint, access::read> up,
    texture2d<uint, access::read> vp,
    constant YuvParams &p,
    uint x,
    uint y)
{
    const float l = p.k0.y * (float(yp.read(uint2(x, y)).r) - p.k0.x);
    float cb = 0.0;
    float cr = 0.0;
    if (p.mode.w == 0u) {
        if (p.out.w == 1u) {
            cb = yuv_chroma(up, p, x, y, 0u);
            cr = yuv_chroma(up, p, x, y, 1u);
        } else {
            cb = yuv_chroma(up, p, x, y, 0u);
            cr = yuv_chroma(vp, p, x, y, 0u);
        }
    }
    const float3 rgb = float3(l + p.k1.x * cr, l - p.k1.y * cb - p.k1.z * cr, l + p.k1.w * cb);
    return clamp(rgb, 0.0, 1.0);
}

fragment float4 fs_yuv(
    FullscreenOut in [[stage_in]],
    texture2d<uint, access::read> yp [[texture(0)]],
    texture2d<uint, access::read> up [[texture(1)]],
    texture2d<uint, access::read> vp [[texture(2)]],
    constant YuvParams &p [[buffer(0)]])
{
    const uint2 px = uint2(in.position.xy);
    const uint k = max(p.out.x, 1u);
    const uint row = p.mode.z == 1u ? p.out.z - 1u - px.y : px.y;
    if (k == 1u) {
        const float3 c = yuv_rgb_at(yp, up, vp, p, min(px.x, p.dims.x - 1u), min(row, p.dims.y - 1u));
        return float4(p.out.y == 1u ? yuv_to_linear(c) : c, 1.0);
    }
    float3 sum = 0.0;
    for (uint j = 0u; j < k; j++) {
        for (uint i = 0u; i < k; i++) {
            const uint x = min(px.x * k + i, p.dims.x - 1u);
            const uint y = min(row * k + j, p.dims.y - 1u);
            sum += yuv_to_linear(yuv_rgb_at(yp, up, vp, p, x, y));
        }
    }
    const float3 lin = sum / float(k * k);
    return float4(p.out.y == 1u ? lin : yuv_to_srgb(lin), 1.0);
}
