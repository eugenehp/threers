// ------------------------------------------------------------- fx surface
//
// `crate::fx::Surface` and its backdrop: the port of `src/fx/surface.wgsl`
// with the `crate::sdf` functions it uses. The material's sixteen uniform
// slots and its three arrays arrive in one `setBytes` block, laid out by
// `crate::fx::surface::metal_bytes`; the slots are documented in the WGSL.
// Appended after `shaders.metal`, whose `Vertex` it reads.

struct FxSurfaceData {
    float4 d[16];
    float4 pts[84];     // 7 strands x 12 points: x, y, radius, -
    float4 s1[22];      // spikes (base xy, tip xy) x 8, (radius, alive) x 8, drops (x, y, r, alive) x 6
    float2 poly[128];   // polygon points
};

struct FxOut {
    float4 position [[position]];
};

static inline float2 fx_smooth_union_blend(float a, float b, float k) {
    if (k <= 0.0) {
        return a <= b ? float2(a, 1.0) : float2(b, 0.0);
    }
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return float2(mix(b, a, h) - k * h * (1.0 - h), h);
}

// Smooth union of (distance, shading thickness, radius) triples; the radius
// is how wide the nearest thing really is (see the WGSL).
static inline float3 fx_melt(float3 a, float3 b, float k) {
    float2 m = fx_smooth_union_blend(a.x, b.x, k);
    return float3(m.x, mix(b.y, a.y, m.y), mix(b.z, a.z, m.y));
}

static inline float fx_footprint(float2 g) {
    float l = length(g);
    return l > 1e-12 ? (abs(g.x) + abs(g.y)) / l : 1.0;
}

static inline float fx_coverage_stroke(float u_in, float half_width, float footprint) {
    float u = abs(u_in);
    float h = max(half_width, 0.0);
    float w = max(footprint, 1e-6);
    float lo = max(u - 0.5 * w, -h);
    float hi = min(u + 0.5 * w, h);
    return clamp((hi - lo) / w, 0.0, 1.0);
}

static inline float fx_rect(float2 p, float2 half_size) {
    float2 q = abs(p) - half_size;
    return length(max(q, float2(0.0))) + min(max(q.x, q.y), 0.0);
}

static inline float fx_round_rect(float2 p, float2 half_size, float r) {
    float rr = min(max(r, 0.0), max(min(half_size.x, half_size.y), 0.0));
    return fx_rect(p, half_size - float2(rr)) - rr;
}

static inline float fx_segment(float2 p, float2 a, float2 b) {
    float2 v = b - a;
    float2 w = p - a;
    float len2 = dot(v, v);
    float t = len2 > 0.0 ? clamp(dot(w, v) / len2, 0.0, 1.0) : 0.0;
    return length(w - v * t);
}

// The hull of two discs: distance in .x, the radius at the nearest point of
// the axis in .y.
static inline float2 fx_tapered_capsule(float2 p_in, float2 a, float2 b, float ra, float rb) {
    float2 pb = b - a;
    float2 p = p_in - a;
    float h = dot(pb, pb);
    float t = clamp(dot(p, pb) / (h + 1e-12), 0.0, 1.0);
    float r = mix(ra, rb, t);
    float bb = ra - rb;
    if (h <= bb * bb) {
        return ra > rb ? float2(length(p) - ra, ra) : float2(length(p - pb) - rb, rb);
    }
    float qx = abs(p.x * pb.y - p.y * pb.x) / h;
    float qy = dot(p, pb) / h;
    float cx = sqrt(h - bb * bb);
    float k = cx * qy - bb * qx;
    float n = qx * qx + qy * qy;
    float d;
    if (k < 0.0) {
        d = sqrt(h * n) - ra;
    } else if (k > cx) {
        d = sqrt(h * (n + 1.0 - 2.0 * qy)) - rb;
    } else {
        d = cx * qx + bb * qy - ra;
    }
    return float2(d, r);
}

static inline float fx_shape_distance(float2 p, constant FxSurfaceData &fx) {
    float4 h = fx.d[1];
    float4 kind = fx.d[5];
    if (kind.x < 0.5) {
        return fx_round_rect(p - h.xy, h.zw, kind.y);
    }
    if (kind.x < 1.5) {
        return length(p - h.xy) - h.z;
    }
    uint n = min(uint(kind.w), 128u);
    float best = 1e9;
    bool inside = false;
    uint j = n - 1u;
    for (uint i = 0u; i < n; ++i) {
        float2 a = fx.poly[i];
        float2 b = fx.poly[j];
        best = min(best, fx_segment(p, a, b));
        if ((p.y >= a.y) != (p.y >= b.y)) {
            float t = (p.y - a.y) / (b.y - a.y);
            if (a.x + t * (b.x - a.x) > p.x) {
                inside = !inside;
            }
        }
        j = i;
    }
    return (inside ? -best : best) - kind.y;
}

static inline float3 fx_field(float2 p, constant FxSurfaceData &fx) {
    float thick = fx.d[5].z;
    float3 body = float3(fx_shape_distance(p, fx), thick, 1e4);
    float2 mr = fx.d[3].xy;
    if (mr.x > 0.01) {
        float4 m = fx.d[2];
        body = fx_melt(body, float3(fx_tapered_capsule(p, m.xy, m.zw, mr.x, mr.y).x, thick, 1e4), 14.0);
    }
    float3 st = float3(1e5, 2.0, 1e4);
    int count = int(fx.d[0].w);
    for (int s = 0; s < 7; ++s) {
        if (s >= count) {
            break;
        }
        float3 one = float3(1e5, 2.0, 1e4);
        for (int i = 0; i < 11; ++i) {
            float4 A = fx.pts[s * 12 + i];
            float4 B = fx.pts[s * 12 + i + 1];
            float2 d = fx_tapered_capsule(p, A.xy, B.xy, A.z, B.z);
            if (d.x < one.x) {
                one = float3(d.x, d.y + 0.3, d.y);
            }
        }
        st = fx_melt(st, one, 3.5);
    }
    for (int k = 0; k < 8; ++k) {
        float4 r = fx.s1[8 + k];
        if (r.y < 0.5) {
            continue;
        }
        float4 sp = fx.s1[k];
        float2 d = fx_tapered_capsule(p, sp.xy, sp.zw, r.x, 0.35);
        st = fx_melt(st, float3(d.x, d.y + 1.0, d.y), 3.0);
    }
    for (int k = 0; k < 6; ++k) {
        float4 d = fx.s1[16 + k];
        if (d.w < 0.5) {
            continue;
        }
        st = fx_melt(st, float3(length(p - d.xy) - d.z, d.z, d.z), 5.0);
    }
    float3 res = fx_melt(body, st, 11.0);
    res.y = max(res.y, body.y * clamp(-body.x / 6.0, 0.0, 1.0));
    return res;
}

static inline float fx_env(float3 r_in, constant FxSurfaceData &fx) {
    float3 R = r_in;
    R.x += fx.d[3].z * 0.18;
    float up = -R.y;
    float4 lv = fx.d[8];
    float e = 0.015;
    e += smoothstep(0.3, 0.5, up) * smoothstep(0.75, 0.55, up) * smoothstep(0.8, 0.5, abs(R.x)) * 2.6 * lv.x;
    e += smoothstep(0.8, 0.95, up) * 0.6 * lv.x;
    e += smoothstep(-0.5, -0.7, R.x) * smoothstep(0.5, 0.25, abs(R.y)) * 1.1 * lv.y;
    e += smoothstep(0.6, 0.85, R.x) * smoothstep(0.6, 0.2, abs(R.y)) * 0.5 * lv.z;
    e += smoothstep(0.25, 0.7, R.y) * 1.25 * lv.w;
    e += smoothstep(0.5, 1.0, R.z) * (0.9 - 0.3 * R.y) * fx.d[9].y;
    return e;
}

vertex FxOut vs_fx_surface(uint vid [[vertex_id]],
                           device const Vertex *verts [[buffer(0)]],
                           constant FxSurfaceData &fx [[buffer(9)]]) {
    float2 corner = sign(float2(verts[vid].position[0], verts[vid].position[1])) * 0.5 + 0.5;
    float4 box = fx.d[4];
    float2 view = fx.d[0].xy;
    float2 css = mix(box.xy, box.zw, float2(corner.x, 1.0 - corner.y));
    FxOut out;
    out.position = float4(css.x / view.x * 2.0 - 1.0, 1.0 - css.y / view.y * 2.0, 0.0, 1.0);
    return out;
}

// Linear, straight alpha: the attachment either encodes on write (sRGB) or
// is a linear float target, so the WGSL's `framebuffer_encode(decode(x))` is
// `srgb_decode(x)` here. Alpha 0 means nothing to draw.
static inline float4 fx_surface_color(float2 frag, constant FxSurfaceData &fx) {
    float dpr = fx.d[0].z;
    float2 p = frag / dpr;
    if (fx.d[5].x > 2.5) {
        float t = clamp(p.y / fx.d[0].y, 0.0, 1.0);
        float3 bg = t < 0.55 ? mix(fx.d[6].rgb, fx.d[7].rgb, t / 0.55)
                             : mix(fx.d[7].rgb, fx.d[8].rgb, clamp((t - 0.55) / 0.45, 0.0, 1.0));
        return float4(srgb_decode(bg), 1.0);
    }
    float3 f = fx_field(p, fx);
    float d = f.x;
    float dd = d * dpr;
    float band = 0.70711;
    float aa = 0.0;
    float2 g = float2(0.0);
    if (dd < band) {
        float e = 0.5;
        g = float2(fx_field(p + float2(e, 0.0), fx).x - d, fx_field(p + float2(0.0, e), fx).x - d);
        float2 across = abs(g);
        float radius = f.z * dpr;
        if (radius < 1.0) {
            across = max(across, abs(float2(d - fx_field(p - float2(e, 0.0), fx).x, d - fx_field(p - float2(0.0, e), fx).x)));
        }
        float w = fx_footprint(across);
        aa = radius < 4.0 ? fx_coverage_stroke(dd + radius, radius, w) : clamp(0.5 - dd / w, 0.0, 1.0);
        g = length(g) > 1e-6 ? normalize(g) : float2(0.0);
    }
    float sh = 0.0;
    if (aa < 1.0) {
        float ds = fx_field(p - float2(2.0, 9.0), fx).x;
        sh = (1.0 - smoothstep(-4.0, 16.0, ds)) * fx.d[9].x;
    }
    float alpha = aa + (1.0 - aa) * sh;
    if (alpha <= 0.0) {
        return float4(0.0);
    }
    float3 col = float3(0.0);
    if (aa > 0.0) {
        float B = max(f.y, 0.8);
        float s = clamp(-d, 0.0, B);
        float z = sqrt(max(1e-3, s * (2.0 * B - s)));
        float slope = min((B - s) / z, 6.0);
        float3 n = normalize(float3(g * slope, 1.0));
        float3 R = reflect(float3(0.0, 0.0, -1.0), n);
        float4 base = fx.d[6];
        float4 tint = fx.d[7];
        float fres = base.w + (1.0 - base.w) * pow(1.0 - max(n.z, 0.0), 3.0);
        col = base.rgb + min(fx_env(R, fx) * fres, 1.0) * tint.rgb;
        col += pow(max(dot(R, normalize(float3(-0.35, -0.75, 0.56))), 0.0), 70.0) * 1.2 * tint.w;
        col += pow(max(dot(R, normalize(float3(0.55, -0.35, 0.76))), 0.0), 140.0) * 0.55 * tint.w;
    }
    float3 straight = clamp(col * aa / alpha, float3(0.0), float3(1.0));
    return float4(srgb_decode(straight), alpha);
}

fragment float4 fs_fx_surface(FxOut in [[stage_in]], constant FxSurfaceData &fx [[buffer(2)]]) {
    float4 src = fx_surface_color(in.position.xy, fx);
    if (src.a <= 0.0) {
        discard_fragment();
    }
    return src;
}

// Into an 8-bit sRGB attachment, blended on the encoded values as the wgpu
// path and a browser canvas do (see `fs_mesh_encoded_blend`): a soft shadow
// over a pale backdrop lands where it does there, not lighter.
fragment float4 fs_fx_surface_encoded_blend(FxOut in [[stage_in]],
                                            constant FxSurfaceData &fx [[buffer(2)]],
                                            float4 dst [[color(0)]]) {
    float4 src = fx_surface_color(in.position.xy, fx);
    if (src.a <= 0.0) {
        discard_fragment();
    }
    float a = saturate(src.a);
    float3 e = srgb_encode(src.rgb) * a + srgb_encode(dst.rgb) * (1.0 - a);
    return float4(srgb_decode(e), a + dst.a * (1.0 - a));
}
