// threers — Metal Shading Language backend.
//
// One uber-shader over the material kinds the backend supports, compiled at
// startup by `MetalDevice::new`. The layouts below are ABI: every struct here
// has a `#[repr(C)]` twin in `renderer.rs`, and `tests/metal_backend.rs` checks
// the sizes agree. Keep all members 16-byte aligned (float4 / float4x4 / uint4)
// so the two layouts cannot drift apart on padding.
//
// Conventions match the wgpu backend so a scene renders the same on either:
//   - clip space z in [0, 1]; counter-clockwise front faces
//   - directional `direction` is the direction light *travels*, so L = -direction
//   - light colours arrive premultiplied by intensity
//   - punctual attenuation follows three.js `getDistanceAttenuation`
//   - UVs are top-left origin; `Texture::flip_y` is applied on the CPU at upload

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------- vertex data

// 48 bytes. `packed_float3` so the layout is exactly the Rust `Vertex`.
struct Vertex {
    packed_float3 position;
    packed_float3 normal;
    float2        uv;
    packed_float3 color;
    float         _pad;
};

/// Lines carry only what a line can use: a position and a colour, 24 bytes
/// against `Vertex`'s 48.
///
/// Lines are unlit and untextured, so the normal, the UV and the pad are dead
/// weight — and at connectome scale that is not a rounding error. A 268M-segment
/// scene stores 536M vertices, where the difference is 25.8 GB against 12.9 GB,
/// which is the difference between fitting in memory and paging.
struct LineVertex {
    packed_float3 position;
    packed_float3 color;
};

// ------------------------------------------------------------------ uniforms

struct DirLight {
    float4 direction;   // xyz = direction of travel
    float4 color;       // rgb premultiplied by intensity
};

struct PointLight {
    float4 position;
    float4 color;
    float4 params;      // x = cutoff distance (0 = none), y = decay
};

struct SpotLight {
    float4 position;
    float4 direction;
    float4 color;
    float4 params;      // x = distance, y = decay, z = cos(outer), w = cos(inner)
};

struct HemiLight {
    float4 sky;
    float4 ground;
    float4 up;
};

// Per-view state. One view is a plain camera; two are the eyes of a headset,
// and the vertex stage picks between them from the instance id.
constant uint MAX_VIEWS = 2u;

struct Frame {
    float4x4   view_proj[MAX_VIEWS];
    float4x4   view[MAX_VIEWS];
    float4     camera_pos[MAX_VIEWS];
    float4     ambient;
    float4     fog_color;   // rgb, w = mode (0 off, 1 linear, 2 exp2)
    float4     fog_params;  // near, far, density
    float4     viewport;    // width, height, 1/width, 1/height
    uint4      counts;      // dir, point, spot, hemi
    uint4      view_info;   // view count, then unused
    DirLight   dir[4];
    PointLight pts[8];
    SpotLight  spots[4];
    HemiLight  hemis[2];
};

struct Draw {
    float4x4 model;
    float4x4 normal_mat;
    float4   base_color;   // rgb, a = opacity
    float4   emissive;     // rgb
    float4   specular;     // rgb, w = shininess
    float4   pbr;          // roughness, metalness, alpha_test, toon steps
    float4   uv_transform; // offset.xy, repeat.xy
    float4   misc;         // uv rotation, point size, depth near, depth far
    uint4    flags;        // kind, has_map, has_vertex_color, instanced
};

// Material kinds — `crate::materials::MaterialKind` values, not renumberable.
constant uint KIND_BASIC      = 0u;
constant uint KIND_LAMBERT    = 1u;
constant uint KIND_PHONG      = 2u;
constant uint KIND_STANDARD   = 3u;
constant uint KIND_PHYSICAL   = 4u;
constant uint KIND_NORMAL     = 5u;
constant uint KIND_DEPTH      = 6u;
constant uint KIND_TOON       = 7u;
constant uint KIND_MATCAP     = 8u;
constant uint KIND_DISTANCE   = 12u;

// The interpolants, shared by the single-view and layered outputs. A macro
// because Metal has no struct inheritance and the fragment stage matches its
// inputs to the vertex outputs by declaration order — the two lists have to
// stay identical, and this is the only way to make that structural rather than
// a comment asking future edits to remember.
#define VOUT_INTERPOLANTS                                                      \
    float4 position [[position]];                                              \
    float3 world_pos;                                                          \
    float3 normal;                                                             \
    float2 uv;                                                                 \
    float3 color;                                                              \
    float  view_dist;                                                          \
    /* Which eye this vertex belongs to; the fragment stage needs it for the */ \
    /* view vector. Flat because it is an index, not a quantity.             */ \
    uint   view_index [[flat]];

/// What the fragment stage receives — the interpolants and nothing else. The
/// builtins below are vertex outputs only.
struct FIn {
    VOUT_INTERPOLANTS
};

/// Single-view output. `point_size` is harmless for triangles here because a
/// pipeline that names no topology class may write it.
struct VOut {
    VOUT_INTERPOLANTS
    float point_size [[point_size]];
};

// Layered outputs carry the array slice to render into. They are separate
// structs for two reasons Metal enforces: a pipeline whose vertex function
// writes `render_target_array_index` must declare its topology class, and a
// pipeline that declares `triangle` must then *not* write `point_size`.
struct VOutLayered {
    VOUT_INTERPOLANTS
    uint layer [[render_target_array_index]];
};

struct VOutLayeredPoint {
    VOUT_INTERPOLANTS
    float point_size [[point_size]];
    uint  layer [[render_target_array_index]];
};

#define COPY_INTERPOLANTS(out, v)                                              \
    out.position = v.position;                                                 \
    out.world_pos = v.world_pos;                                               \
    out.normal = v.normal;                                                     \
    out.uv = v.uv;                                                             \
    out.color = v.color;                                                       \
    out.view_dist = v.view_dist;                                               \
    out.view_index = v.view_index;                                             \
    out.layer = v.view_index; /* one eye per slice */

static inline VOutLayered to_layered(VOut v) {
    VOutLayered out;
    COPY_INTERPOLANTS(out, v)
    return out;
}

static inline VOutLayeredPoint to_layered_point(VOut v) {
    VOutLayeredPoint out;
    COPY_INTERPOLANTS(out, v)
    out.point_size = v.point_size;
    return out;
}

// --------------------------------------------------------------------- vertex

static inline float2 apply_uv_transform(float2 uv, float4 xform, float rotation) {
    float2 t = uv * xform.zw + xform.xy;
    if (rotation != 0.0) {
        float s = sin(rotation);
        float c = cos(rotation);
        float2 p = t - float2(0.5);
        t = float2(p.x * c - p.y * s, p.x * s + p.y * c) + float2(0.5);
    }
    return t;
}

static inline float4x4 model_matrix(constant Draw &draw,
                                    device const float4x4 *instances,
                                    uint iid) {
    // Only pay for the instance multiply when there is an instance to apply;
    // a 4x4 product per vertex is not free at scene scale.
    return draw.flags.w == 1u ? draw.model * instances[iid] : draw.model;
}

static inline VOut transform(Vertex v,
                             float4x4 model,
                             float3x3 nrm,
                             constant Frame &frame,
                             constant Draw &draw,
                             uint view_index) {
    VOut out;
    float4 world = model * float4(float3(v.position), 1.0);
    out.position = frame.view_proj[view_index] * world;
    out.world_pos = world.xyz;
    out.normal = nrm * float3(v.normal);
    out.uv = apply_uv_transform(v.uv, draw.uv_transform, draw.misc.x);
    out.color = draw.flags.z == 1u ? float3(v.color) : float3(1.0);
    out.view_dist = length(frame.camera_pos[view_index].xyz - world.xyz);
    out.point_size = 1.0;
    out.view_index = view_index;
    return out;
}

static inline float3x3 normal_basis(constant Draw &draw, float4x4 model) {
    if (draw.flags.w == 1u) {
        // Instanced: the per-instance matrix is not known on the CPU side of
        // the uniform, so use the composed model's upper 3x3. Correct for
        // rigid and uniformly scaled instances; non-uniform instance scale
        // skews normals (documented in `MetalRenderer`).
        return float3x3(model[0].xyz, model[1].xyz, model[2].xyz);
    }
    return float3x3(draw.normal_mat[0].xyz,
                    draw.normal_mat[1].xyz,
                    draw.normal_mat[2].xyz);
}

static inline VOut point_size_for(VOut out, constant Frame &frame, constant Draw &draw) {
    float size = draw.misc.y;
    if (draw.misc.z > 0.5) {
        // three.js size attenuation: gl_PointSize = size * (h/2) / -mvPosition.z
        float view_z =
            max(-(frame.view[out.view_index] * float4(out.world_pos, 1.0)).z, 1e-4);
        size *= (frame.viewport.y * 0.5) / view_z;
    }
    out.point_size = clamp(size, 1.0, 255.0);
    return out;
}

vertex VOut vs_mesh(uint vid [[vertex_id]],
                    uint iid [[instance_id]],
                    device const Vertex   *verts     [[buffer(0)]],
                    constant Frame        &frame     [[buffer(1)]],
                    constant Draw         &draw      [[buffer(2)]],
                    device const float4x4 *instances [[buffer(3)]]) {
    float4x4 model = model_matrix(draw, instances, iid);
    return transform(verts[vid], model, normal_basis(draw, model), frame, draw, 0u);
}

vertex VOut vs_line(uint vid [[vertex_id]],
                    uint iid [[instance_id]],
                    device const LineVertex *verts     [[buffer(0)]],
                    constant Frame          &frame     [[buffer(1)]],
                    constant Draw           &draw      [[buffer(2)]],
                    device const float4x4   *instances [[buffer(3)]]) {
    // Widened into a full Vertex and handed to the same transform, so the line
    // path cannot drift from the mesh path as that code changes.
    Vertex v;
    v.position = verts[vid].position;
    v.normal   = packed_float3(0.0, 0.0, 1.0);  // never read; unlit
    v.uv       = float2(0.0);
    v.color    = verts[vid].color;
    v._pad     = 0.0;
    float4x4 model = model_matrix(draw, instances, iid);
    return transform(v, model, normal_basis(draw, model), frame, draw, 0u);
}

// Expand each segment into a screen-space quad, so a line thinner than a pixel
// contributes in proportion to the area it covers.
//
// Hardware lines are all-or-nothing at one pixel wide. For filaments that sit
// below a pixel — most of a connectome at any sane resolution — that means a
// neurite either lights a whole pixel or vanishes, and which one it does
// changes frame to frame as the camera turns. That is the flicker supersampling
// only partly hides, and coverage fixes at the source.
//
// Reads the same `LineVertex` buffer, six vertices per segment instead of two,
// so nothing extra is stored.
vertex VOut vs_line_quad(uint vid [[vertex_id]],
                         uint iid [[instance_id]],
                         device const LineVertex *verts     [[buffer(0)]],
                         constant Frame          &frame     [[buffer(1)]],
                         constant Draw           &draw      [[buffer(2)]],
                         device const float4x4   *instances [[buffer(3)]]) {
    const uint seg    = vid / 6u;
    const uint corner = vid % 6u;
    // Two triangles: (near,-) (far,-) (near,+) / (near,+) (far,-) (far,+)
    const uint  ends[6]  = {0u, 1u, 0u, 0u, 1u, 1u};
    const float sides[6] = {-1.0, -1.0, 1.0, 1.0, -1.0, 1.0};
    const uint  end  = ends[corner];
    const float side = sides[corner];

    const float4x4 model = model_matrix(draw, instances, iid);
    const LineVertex me = verts[seg * 2u + end];
    const LineVertex ot = verts[seg * 2u + (1u - end)];

    float4 pm = frame.view_proj[0] * (model * float4(float3(me.position), 1.0));
    float4 po = frame.view_proj[0] * (model * float4(float3(ot.position), 1.0));

    // Perpendicular in pixels, then back to clip. Guarded because a segment can
    // project to a point, and normalize(0) is a NaN that takes the quad with it.
    const float2 sm = pm.xy / max(abs(pm.w), 1e-6);
    const float2 so = po.xy / max(abs(po.w), 1e-6);
    float2 d = (sm - so) * frame.viewport.xy;
    const float len = length(d);
    d = len > 1e-6 ? d / len : float2(1.0, 0.0);
    const float2 nrm = float2(-d.y, d.x) * frame.viewport.zw;

    // draw.misc.y carries the width in pixels, shared with point size.
    const float half_px = max(draw.misc.y, 1.0) * 0.5;
    pm.xy += nrm * side * half_px * abs(pm.w);

    VOut out;
    out.position = pm;
    out.world_pos = (model * float4(float3(me.position), 1.0)).xyz;
    out.normal = float3(0.0, 0.0, 1.0);
    // uv.x carries signed distance across the ribbon, for the coverage falloff.
    out.uv = float2(side, half_px);
    out.color = draw.flags.z == 1u ? float3(me.color) : float3(1.0);
    out.view_dist = length(frame.camera_pos[0].xyz - out.world_pos);
    out.point_size = 1.0;
    out.view_index = 0u;
    return out;
}

fragment float4 fs_line_quad(VOut in [[stage_in]],
                             constant Frame &frame [[buffer(0)]],
                             constant Draw  &draw  [[buffer(1)]]) {
    // Coverage from the distance across the ribbon. A quad never narrower than
    // one pixel keeps the line visible; the falloff is what makes a sub-pixel
    // filament contribute a fraction rather than a whole pixel.
    const float half_px = in.uv.y;
    const float dist = abs(in.uv.x) * half_px;
    const float cov = clamp(half_px + 0.5 - dist, 0.0, 1.0);
    float4 c = float4(draw.base_color.rgb * in.color, draw.base_color.a * cov);
    if (c.a <= 0.0) {
        discard_fragment();
    }
    return c;
}

vertex VOut vs_point(uint vid [[vertex_id]],
                     uint iid [[instance_id]],
                     device const Vertex   *verts     [[buffer(0)]],
                     constant Frame        &frame     [[buffer(1)]],
                     constant Draw         &draw      [[buffer(2)]],
                     device const float4x4 *instances [[buffer(3)]]) {
    float4x4 model = model_matrix(draw, instances, iid);
    VOut out = transform(verts[vid], model, normal_basis(draw, model), frame, draw, 0u);
    return point_size_for(out, frame, draw);
}

// ------------------------------------------------------------ layered stereo
//
// Both eyes in one pass, into one texture array. The draw is issued with
// `instanceCount * view_count` instances and each vertex reads its eye out of
// the instance id, writing `render_target_array_index` to pick the slice — so a
// headset frame costs one traversal of the scene, not two.

static inline uint view_of(uint iid, constant Frame &frame) {
    return frame.view_info.x > 1u ? iid % frame.view_info.x : 0u;
}

static inline uint instance_of(uint iid, constant Frame &frame) {
    return frame.view_info.x > 1u ? iid / frame.view_info.x : iid;
}

vertex VOutLayered vs_mesh_layered(uint vid [[vertex_id]],
                                   uint iid [[instance_id]],
                                   device const Vertex   *verts     [[buffer(0)]],
                                   constant Frame        &frame     [[buffer(1)]],
                                   constant Draw         &draw      [[buffer(2)]],
                                   device const float4x4 *instances [[buffer(3)]]) {
    uint view = view_of(iid, frame);
    float4x4 model = model_matrix(draw, instances, instance_of(iid, frame));
    return to_layered(
        transform(verts[vid], model, normal_basis(draw, model), frame, draw, view));
}

vertex VOutLayeredPoint vs_point_layered(uint vid [[vertex_id]],
                                    uint iid [[instance_id]],
                                    device const Vertex   *verts     [[buffer(0)]],
                                    constant Frame        &frame     [[buffer(1)]],
                                    constant Draw         &draw      [[buffer(2)]],
                                    device const float4x4 *instances [[buffer(3)]]) {
    uint view = view_of(iid, frame);
    float4x4 model = model_matrix(draw, instances, instance_of(iid, frame));
    VOut out = transform(verts[vid], model, normal_basis(draw, model), frame, draw, view);
    return to_layered_point(point_size_for(out, frame, draw));
}

// ------------------------------------------------------------------- lighting

static inline float punctual_attenuation(float d, float max_d, float decay) {
    // three.js `getDistanceAttenuation`, matched to the wgpu backend.
    float att = 1.0 / max(pow(d, decay), 0.01);
    if (max_d > 0.0) {
        float t = clamp(1.0 - pow(d / max_d, 4.0), 0.0, 1.0);
        att *= t * t;
    }
    return att;
}

struct Surface {
    float3 n;
    float3 v;        // surface -> camera
    float3 albedo;
    float3 specular;
    float  shininess;
    float  roughness;
    float  metalness;
    uint   kind;
    float  toon_steps;
};

static inline float3 diffuse_term(Surface s, float ndl) {
    if (s.kind == KIND_TOON && s.toon_steps > 0.5) {
        float q = floor(ndl * s.toon_steps + 0.5) / s.toon_steps;
        return s.albedo * clamp(q, 0.0, 1.0);
    }
    return s.albedo * ndl;
}

// Blinn-Phong specular, for KIND_PHONG.
static inline float3 phong_specular(Surface s, float3 l, float ndl) {
    if (s.kind != KIND_PHONG || ndl <= 0.0) {
        return float3(0.0);
    }
    float3 h = normalize(l + s.v);
    float spec = pow(max(dot(s.n, h), 0.0), max(s.shininess, 1.0));
    return s.specular * spec;
}

// Cook-Torrance GGX, for KIND_STANDARD / KIND_PHYSICAL.
static inline float3 pbr_direct(Surface s, float3 l, float ndl) {
    float rough = clamp(s.roughness, 0.045, 1.0);
    float a = rough * rough;
    float3 h = normalize(l + s.v);
    float ndh = max(dot(s.n, h), 0.0);
    float ndv = max(dot(s.n, s.v), 1e-4);
    float vdh = max(dot(s.v, h), 1e-4);

    float a2 = a * a;
    float denom = ndh * ndh * (a2 - 1.0) + 1.0;
    float d = a2 / max(M_PI_F * denom * denom, 1e-7);

    float k = a * 0.5;
    float gv = ndv / (ndv * (1.0 - k) + k);
    float gl = ndl / (ndl * (1.0 - k) + k);
    float g = gv * gl;

    float3 f0 = mix(float3(0.04), s.albedo, s.metalness);
    float3 f = f0 + (float3(1.0) - f0) * pow(1.0 - vdh, 5.0);

    float3 spec = (d * g * f) / max(4.0 * ndv * ndl, 1e-4) * ndl;
    float3 kd = (float3(1.0) - f) * (1.0 - s.metalness);
    return kd * s.albedo * ndl + spec;
}

static inline float3 shade_one(Surface s, float3 l, float3 light_color, float attenuation) {
    float ndl = max(dot(s.n, l), 0.0);
    if (attenuation <= 0.0) {
        return float3(0.0);
    }
    float3 contrib;
    if (s.kind == KIND_STANDARD || s.kind == KIND_PHYSICAL) {
        contrib = pbr_direct(s, l, ndl);
    } else {
        contrib = diffuse_term(s, ndl) + phong_specular(s, l, ndl);
    }
    return contrib * light_color * attenuation;
}

static inline float3 shade(Surface s, float3 world_pos, constant Frame &frame) {
    float3 lit = s.albedo * frame.ambient.rgb;

    for (uint i = 0u; i < frame.counts.w && i < 2u; ++i) {
        HemiLight h = frame.hemis[i];
        float t = dot(s.n, normalize(h.up.xyz)) * 0.5 + 0.5;
        lit += s.albedo * mix(h.ground.rgb, h.sky.rgb, t);
    }
    for (uint i = 0u; i < frame.counts.x && i < 4u; ++i) {
        DirLight d = frame.dir[i];
        lit += shade_one(s, normalize(-d.direction.xyz), d.color.rgb, 1.0);
    }
    for (uint i = 0u; i < frame.counts.y && i < 8u; ++i) {
        PointLight p = frame.pts[i];
        float3 to_light = p.position.xyz - world_pos;
        float dist = length(to_light);
        lit += shade_one(s, to_light / max(dist, 1e-4), p.color.rgb,
                         punctual_attenuation(dist, p.params.x, p.params.y));
    }
    for (uint i = 0u; i < frame.counts.z && i < 4u; ++i) {
        SpotLight sp = frame.spots[i];
        float3 to_light_vec = sp.position.xyz - world_pos;
        float dist = length(to_light_vec);
        float3 to_light = to_light_vec / max(dist, 1e-4);
        float cos_angle = dot(-to_light, normalize(sp.direction.xyz));
        float cone = cos_angle > sp.params.z
                   ? smoothstep(sp.params.z, sp.params.w, cos_angle)
                   : 0.0;
        lit += shade_one(s, to_light, sp.color.rgb,
                         punctual_attenuation(dist, sp.params.x, sp.params.y) * cone);
    }
    return lit;
}

static inline float3 apply_fog(float3 color, float view_dist, constant Frame &frame) {
    uint mode = uint(frame.fog_color.w + 0.5);
    if (mode == 0u) {
        return color;
    }
    float f;
    if (mode == 2u) {
        float d = frame.fog_params.z * view_dist;
        f = 1.0 - exp(-d * d);
    } else {
        f = (view_dist - frame.fog_params.x) /
            max(frame.fog_params.y - frame.fog_params.x, 1e-4);
    }
    return mix(color, frame.fog_color.rgb, clamp(f, 0.0, 1.0));
}

// ------------------------------------------------------------------- fragment

static inline float4 sample_base(constant Draw &draw,
                                 texture2d<float> base_map,
                                 sampler samp,
                                 float2 uv) {
    float4 base = draw.base_color;
    if (draw.flags.y == 1u) {
        base *= base_map.sample(samp, uv);
    }
    return base;
}

fragment float4 fs_mesh(FIn in [[stage_in]],
                        bool front_facing [[front_facing]],
                        constant Frame   &frame    [[buffer(0)]],
                        constant Draw    &draw     [[buffer(1)]],
                        texture2d<float>  base_map [[texture(0)]],
                        sampler           samp     [[sampler(0)]]) {
    uint kind = draw.flags.x;
    float3 n = normalize(in.normal);
    if (!front_facing) {
        n = -n;   // double-sided lighting, as three.js does for BackSide/DoubleSide
    }

    if (kind == KIND_NORMAL) {
        return float4(n * 0.5 + 0.5, draw.base_color.a);
    }
    if (kind == KIND_DEPTH || kind == KIND_DISTANCE) {
        float near = draw.misc.z;
        float far = max(draw.misc.w, near + 1e-4);
        float d = clamp((in.view_dist - near) / (far - near), 0.0, 1.0);
        return float4(float3(1.0 - d), draw.base_color.a);
    }

    float4 base = sample_base(draw, base_map, samp, in.uv);
    base.rgb *= in.color;
    if (base.a < draw.pbr.z) {
        discard_fragment();
    }

    if (kind == KIND_MATCAP) {
        float3 view_n = normalize((frame.view[in.view_index] * float4(n, 0.0)).xyz);
        float2 muv = view_n.xy * 0.5 + 0.5;
        float3 matcap = draw.flags.y == 1u ? base_map.sample(samp, muv).rgb : float3(1.0);
        return float4(apply_fog(draw.base_color.rgb * matcap * in.color, in.view_dist, frame),
                      base.a);
    }

    bool lit_kind = kind == KIND_LAMBERT || kind == KIND_PHONG || kind == KIND_STANDARD
                 || kind == KIND_PHYSICAL || kind == KIND_TOON;
    float3 color;
    if (lit_kind) {
        Surface s;
        s.n = n;
        s.v = normalize(frame.camera_pos[in.view_index].xyz - in.world_pos);
        s.albedo = base.rgb;
        s.specular = draw.specular.rgb;
        s.shininess = draw.specular.w;
        s.roughness = draw.pbr.x;
        s.metalness = draw.pbr.y;
        s.kind = kind;
        s.toon_steps = draw.pbr.w;
        color = shade(s, in.world_pos, frame) + draw.emissive.rgb;
    } else {
        // Unlit: Basic, Line, Points, Sprite, Sky, Mirror, Shader, Atmosphere.
        color = base.rgb + draw.emissive.rgb;
    }

    return float4(apply_fog(color, in.view_dist, frame), base.a);
}

fragment float4 fs_point(FIn in [[stage_in]],
                         float2 point_coord [[point_coord]],
                         constant Frame   &frame    [[buffer(0)]],
                         constant Draw    &draw     [[buffer(1)]],
                         texture2d<float>  base_map [[texture(0)]],
                         sampler           samp     [[sampler(0)]]) {
    // three.js PointsMaterial samples its map with gl_PointCoord, not the
    // vertex UV — the point is a screen-space quad.
    float4 base = draw.base_color;
    if (draw.flags.y == 1u) {
        base *= base_map.sample(samp, point_coord);
    }
    base.rgb *= in.color;
    if (base.a < max(draw.pbr.z, 1e-4)) {
        discard_fragment();
    }
    return float4(apply_fog(base.rgb, in.view_dist, frame), base.a);
}
