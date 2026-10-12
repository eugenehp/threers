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
    /// World -> light clip, for the one directional caster.
    float4x4   shadow_vp;
    /// x: on, y: depth bias, z: one texel as a fraction of the map,
    /// w: normal bias in world units.
    float4     shadow_params;
    /// World -> light clip for the one spot caster.
    float4x4   spot_shadow_vp;
    /// x: spot shadow on, y: its depth bias; z, w unused.
    float4     spot_params;
    /// xyz: the point caster's world position, w: its radius (0 = none).
    float4     point_shadow_pos;
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
    uint4    maps;         // emissive, normal, metallic-roughness, AO: bound?
    float4   map_params;   // normal scale, AO strength, env intensity, has env
    uint4    object;       // x: receives shadow; rest unused
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

/// Per-vertex skinning influences. Joint indices arrive as floats because that
/// is how `BufferGeometry` stores them, three.js doing the same.
struct SkinVertex {
    float4 joints;
    float4 weights;
};

/// The weighted blend of a vertex's four bones.
///
/// Normalised, because an exporter that drops a fifth influence leaves the
/// remaining four summing to less than one, and an un-normalised blend shrinks
/// the mesh toward the origin wherever that happened.
static inline float4x4 skin_matrix(SkinVertex sv, device const float4x4 *bones) {
    float total = sv.weights[0] + sv.weights[1] + sv.weights[2] + sv.weights[3];
    if (total <= 1e-6) {
        return float4x4(1.0);
    }
    float inv = 1.0 / total;
    float4x4 m = bones[uint(sv.joints[0])] * (sv.weights[0] * inv);
    for (uint k = 1u; k < 4u; ++k) {
        if (sv.weights[k] > 0.0) {
            m += bones[uint(sv.joints[k])] * (sv.weights[k] * inv);
        }
    }
    return m;
}

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

/// Skinned transform: blend the vertex through its bones, then the model.
///
/// `bones[i]` is already world × inverse-bind for bone `i` — `Skeleton::update`
/// composes that on the CPU — so the only work here is the weighted sum. Four
/// influences, which is what glTF stores and what three.js reads.
///
/// The normal is transformed by the same blend rather than by its own inverse
/// transpose. That is exact for rigid bones and wrong only under non-uniform
/// bone scale, which is rare enough in rigged assets that the cost of a second
/// matrix per bone is not worth paying for it.
vertex VOut vs_skinned(uint vid [[vertex_id]],
                       uint iid [[instance_id]],
                       device const Vertex     *verts     [[buffer(0)]],
                       constant Frame          &frame     [[buffer(1)]],
                       constant Draw           &draw      [[buffer(2)]],
                       device const float4x4   *instances [[buffer(3)]],
                       device const SkinVertex *skin      [[buffer(4)]],
                       device const float4x4   *bones     [[buffer(5)]]) {
    float4x4 model = model_matrix(draw, instances, iid) * skin_matrix(skin[vid], bones);
    Vertex v = verts[vid];
    return transform(v, model, normal_basis(draw, model), frame, draw, 0u);
}

/// Depth-only pass from the light's point of view.
///
/// Writes position and nothing else: the shadow map is a depth buffer, and a
/// fragment function that returns a colour nobody reads would cost a whole
/// attachment. Instancing and skinning both apply — a shadow cast by the bind
/// pose of an animated mesh is worse than no shadow, because it is wrong in a
/// way that looks deliberate.
vertex float4 vs_shadow(uint vid [[vertex_id]],
                        uint iid [[instance_id]],
                        device const Vertex     *verts     [[buffer(0)]],
                        constant Frame          &frame     [[buffer(1)]],
                        constant Draw           &draw      [[buffer(2)]],
                        device const float4x4   *instances [[buffer(3)]],
                        device const SkinVertex *skin      [[buffer(4)]],
                        device const float4x4   *bones     [[buffer(5)]]) {
    float4x4 model = model_matrix(draw, instances, iid);
    if (draw.maps.w == 2u) {   // 2 marks a skinned draw; see `DrawUniforms`
        model = model * skin_matrix(skin[vid], bones);
    }
    return frame.shadow_vp * model * float4(verts[vid].position, 1.0);
}

/// Point-light shadow pass: writes *radial* distance, not projected depth.
///
/// This is the one place a shadow pass needs a fragment stage. A cube's six
/// faces each have their own frustum, so projected depth is per-face and
/// discontinuous across the seams — but the lookup is by direction and has only
/// one number to compare, the distance from the light. Storing that directly is
/// what makes one cube serve all six directions.
///
/// Worth noting because it is a trap the wgpu path is still in: it renders its
/// cube with `fragment: None`, so the stored value is projected depth while the
/// comparison is `dist / radius`. Those are different quantities and the test
/// never passes — its point shadows silently never trigger.
struct ShadowPointOut {
    float4 position [[position]];
    float3 world;
};

vertex ShadowPointOut vs_shadow_point(uint vid [[vertex_id]],
                                      uint iid [[instance_id]],
                                      device const Vertex     *verts     [[buffer(0)]],
                                      constant Frame          &frame     [[buffer(1)]],
                                      constant Draw           &draw      [[buffer(2)]],
                                      device const float4x4   *instances [[buffer(3)]],
                                      device const SkinVertex *skin      [[buffer(4)]],
                                      device const float4x4   *bones     [[buffer(5)]]) {
    float4x4 model = model_matrix(draw, instances, iid);
    if (draw.maps.w == 2u) {
        model = model * skin_matrix(skin[vid], bones);
    }
    float4 world = model * float4(verts[vid].position, 1.0);
    ShadowPointOut o;
    o.position = frame.shadow_vp * world;
    o.world = world.xyz;
    return o;
}

struct ShadowDepth {
    float depth [[depth(any)]];
};

fragment ShadowDepth fs_shadow_point(ShadowPointOut in [[stage_in]],
                                     constant Frame &frame [[buffer(1)]]) {
    float radius = max(frame.point_shadow_pos.w, 1e-4);
    ShadowDepth o;
    o.depth = saturate(length(in.world - frame.point_shadow_pos.xyz) / radius);
    return o;
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

    // Height-correlated Smith visibility, which folds in the 1/(4·NdotV·NdotL)
    // the specular term would otherwise divide by. The separable Schlick-GGX
    // approximation that used to be here is cheaper and does not match the wgpu
    // path, and matching it is the point: the two renderers are held to each
    // other in `tests/backend_parity.rs`.
    float lambda_v = ndl * sqrt(ndv * ndv * (1.0 - a2) + a2);
    float lambda_l = ndv * sqrt(ndl * ndl * (1.0 - a2) + a2);
    float vis = 0.5 / max(lambda_v + lambda_l, 1e-5);

    float3 f0 = mix(float3(0.04), s.albedo, s.metalness);
    float3 f = f0 + (float3(1.0) - f0) * pow(1.0 - vdh, 5.0);

    float3 spec = d * vis * f;
    float3 kd = (float3(1.0) - f) * (1.0 - s.metalness);
    // The 1/pi belongs on the diffuse lobe: a Lambertian surface scatters its
    // albedo over a hemisphere, not into a single direction. Leaving it off
    // made the diffuse pi times too strong relative to the specular, which is
    // invisible on a flat-coloured surface and obvious the moment a roughness
    // or normal map gives the specular something to vary against.
    float3 diff = kd * s.albedo * (1.0 / M_PI_F);
    return (diff + spec) * ndl;
}

/// How much of the one directional caster reaches this point, 0 to 1.
///
/// Mirrors `shadow_factor` in `renderer/shader.rs`: normal bias rather than a
/// large constant depth bias, and a 3x3 percentage-closer filter.
///
/// The normal bias steps off the surface before looking the depth up. A
/// constant depth bias has to cover the worst case — a face nearly edge-on to
/// the light, where one shadow texel spans a long slope — and sized for that it
/// is far too much everywhere else, so shadows visibly detach from their
/// casters. Moving the lookup along the normal scales with the geometry rather
/// than the angle.
///
/// The filter turns a binary per-pixel answer into a one-texel gradient. A
/// single compare makes the shadow edge the shadow map's own pixel grid, drawn
/// at whatever size the camera magnifies it to — a staircase that crawls as the
/// light moves. This is antialiasing, not softness.
static inline float shadow_factor(constant Frame &frame,
                                  depth2d<float> shadow_map,
                                  sampler shadow_samp,
                                  float3 world_pos,
                                  float3 n) {
    if (frame.shadow_params.x < 0.5) {
        return 1.0;
    }
    float3 biased = world_pos + n * frame.shadow_params.w;
    float4 p = frame.shadow_vp * float4(biased, 1.0);
    float3 ndc = p.xyz / max(p.w, 1e-4);
    float2 uv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return 1.0;
    }
    if (ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }
    const float bias = frame.shadow_params.y;
    const float texel = frame.shadow_params.z;
    float sum = 0.0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            float2 o = float2(float(dx), float(dy)) * texel;
            sum += shadow_map.sample_compare(shadow_samp, uv + o, ndc.z - bias);
        }
    }
    return sum / 9.0;
}

/// The spot caster's occlusion at this point.
///
/// One tap, not nine: a spot's map covers a cone rather than the whole scene,
/// so its texels are already small where it matters, and this mirrors the wgpu
/// path, which does the same.
static inline float shadow_factor_spot(constant Frame &frame,
                                       depth2d<float> spot_map,
                                       sampler shadow_samp,
                                       float3 world_pos) {
    if (frame.spot_params.x < 0.5) {
        return 1.0;
    }
    float4 p = frame.spot_shadow_vp * float4(world_pos, 1.0);
    float3 ndc = p.xyz / max(p.w, 1e-4);
    float2 uv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return 1.0;
    }
    if (ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }
    return spot_map.sample_compare(shadow_samp, uv, ndc.z - frame.spot_params.y);
}

/// The point caster's occlusion, looked up by direction.
///
/// A point light shadows in every direction at once, so its map is a cube and
/// the lookup is the vector from the light rather than a projected coordinate.
/// The stored comparison is the *radial* distance normalised by the light's
/// radius, which is what makes one cube serve all six directions — a projected
/// depth would be per-face and discontinuous at the seams.
static inline float shadow_factor_point(constant Frame &frame,
                                        depthcube<float> point_map,
                                        sampler shadow_samp,
                                        float3 world_pos) {
    float radius = frame.point_shadow_pos.w;
    if (radius <= 0.0) {
        return 1.0;
    }
    float3 to_frag = world_pos - frame.point_shadow_pos.xyz;
    float dist = length(to_frag);
    if (dist > radius) {
        return 1.0;
    }
    return point_map.sample_compare(shadow_samp, normalize(to_frag), dist / radius - 0.005);
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

/// `ao` attenuates the *indirect* terms only — ambient and hemisphere — and not
/// the direct lights.
///
/// Ambient occlusion answers "how much of the surrounding room can this point
/// see", which is a question about light arriving from everywhere. A direct
/// light is somewhere specific, and whether it reaches the point is what a
/// shadow map is for. Multiplying the whole shade by `ao` darkens the key light
/// in a crevice that the key light is shining straight into.
static inline float3 shade(Surface s,
                           float3 world_pos,
                           constant Frame &frame,
                           float ao,
                           depth2d<float> shadow_map,
                           depth2d<float> spot_map,
                           depthcube<float> point_map,
                           sampler shadow_samp,
                           bool receives_shadow) {
    float3 lit = s.albedo * frame.ambient.rgb * ao;

    for (uint i = 0u; i < frame.counts.w && i < 2u; ++i) {
        HemiLight h = frame.hemis[i];
        float t = dot(s.n, normalize(h.up.xyz)) * 0.5 + 0.5;
        lit += s.albedo * mix(h.ground.rgb, h.sky.rgb, t) * ao;
    }
    // One caster, and it is the first directional light — which is what the
    // Rust side picks when it builds `shadow_vp`. A second casting light would
    // need a second map and a second matrix; `shadow_params.x` stays 0 for it
    // rather than having it borrow the first one's shadow.
    const float shadow = receives_shadow
        ? shadow_factor(frame, shadow_map, shadow_samp, world_pos, s.n)
        : 1.0;
    for (uint i = 0u; i < frame.counts.x && i < 4u; ++i) {
        DirLight d = frame.dir[i];
        float att = (i == 0u) ? shadow : 1.0;
        lit += shade_one(s, normalize(-d.direction.xyz), d.color.rgb, att);
    }
    // As with the directional caster, the *first* point and spot light own the
    // maps — one each, because a second would need a second cube and a second
    // frustum, and borrowing puts shadows where nothing stands.
    const float point_shadow = receives_shadow
        ? shadow_factor_point(frame, point_map, shadow_samp, world_pos)
        : 1.0;
    const float spot_shadow = receives_shadow
        ? shadow_factor_spot(frame, spot_map, shadow_samp, world_pos)
        : 1.0;
    for (uint i = 0u; i < frame.counts.y && i < 8u; ++i) {
        PointLight p = frame.pts[i];
        float3 to_light = p.position.xyz - world_pos;
        float dist = length(to_light);
        float att = punctual_attenuation(dist, p.params.x, p.params.y);
        if (i == 0u) { att *= point_shadow; }
        lit += shade_one(s, to_light / max(dist, 1e-4), p.color.rgb, att);
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
        float att = punctual_attenuation(dist, sp.params.x, sp.params.y) * cone;
        if (i == 0u) { att *= spot_shadow; }
        lit += shade_one(s, to_light, sp.color.rgb, att);
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

// ----------------------------------------------------------------------- IBL

// Image-based lighting, mirroring `sample_env_cube` / `dfg_approx` in
// `renderer/shader.rs` so the two renderers agree about what a room looks like
// reflected in a phone.
//
// The environment arrives as a plain cube with its own mip chain rather than
// the PMREM atlas the wgpu path prefers, so roughness picks a level directly.
// `roughness²` is the mapping: roughness is authored perceptually and the
// filter width goes as alpha, and the wgpu side learned the same thing the hard
// way — it used to map the whole of 0..1 onto seven tenths of one level, so a
// mirror and a rough metal came back identically blurred.
static inline float3 sample_env(texturecube<float> env,
                                sampler samp,
                                float3 dir,
                                float roughness) {
    float levels = float(env.get_num_mip_levels()) - 1.0;
    float mip = clamp(roughness * roughness * levels, 0.0, levels);
    float lo = floor(mip);
    float f = mip - lo;
    float3 c0 = env.sample(samp, dir, level(lo)).rgb;
    if (f <= 0.0) {
        return c0;
    }
    return mix(c0, env.sample(samp, dir, level(lo + 1.0)).rgb, f);
}

// three.js `DFGApprox()` — the split-sum approximation, same constants as the
// wgpu path uses.
static inline float2 dfg_approx(float3 n, float3 v, float roughness) {
    float dot_nv = saturate(dot(n, v));
    const float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022);
    const float4 c1 = float4(1.0, 0.0425, 1.04, -0.04);
    float4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * dot_nv)) * r.x + r.y;
    return float2(-1.04, 1.04) * a004 + r.zw;
}

// three.js's specular occlusion: how much of the room a glancing, rough surface
// can actually see. Without it the environment is added as though every point
// had an unobstructed view of all of it.
static inline float specular_occlusion(float dot_nv, float ao, float roughness) {
    return saturate(pow(dot_nv + ao, exp2(-16.0 * roughness - 1.0)) - 1.0 + ao);
}

/// The environment's contribution, diffuse and specular, occluded.
static inline float3 ibl(Surface s,
                         texturecube<float> env,
                         sampler samp,
                         float ao,
                         float intensity) {
    float roughness = clamp(s.roughness, 0.0, 1.0);
    float3 f0 = mix(float3(0.04), s.albedo, s.metalness);
    float3 diffuse_color = s.albedo * (1.0 - s.metalness);

    // Bend the reflection toward the normal as roughness rises, which is what
    // keeps a rough surface from mirroring a sharp edge of the room.
    float3 refl = reflect(-s.v, s.n);
    refl = normalize(mix(refl, s.n, roughness * roughness));
    float3 radiance = sample_env(env, samp, refl, roughness) * intensity;
    float3 irradiance = sample_env(env, samp, s.n, 1.0) * intensity * M_PI_F;

    float2 fab = dfg_approx(s.n, s.v, roughness);
    float3 fss_ess = f0 * fab.x + float3(fab.y);
    float ess = fab.x + fab.y;
    float ems = 1.0 - ess;
    float3 favg = f0 + (float3(1.0) - f0) * 0.047619;
    float3 fms = fss_ess * favg / (float3(1.0) - ems * favg);
    float3 multi_scatter = fms * ems;
    float3 total = fss_ess + multi_scatter;
    float scatter_max = max(max(total.r, total.g), total.b);
    float3 cosine_weighted = irradiance * (1.0 / M_PI_F);

    float spec_occ = specular_occlusion(saturate(dot(s.n, s.v)), ao, roughness);
    float3 lit = (radiance * fss_ess + multi_scatter * cosine_weighted) * spec_occ;
    // The environment's diffuse term, added exactly once — the ambient lights
    // are accounted for separately in `shade`, and adding this there as well is
    // how a white surface ends up reflecting more light than reaches it.
    lit += diffuse_color * (1.0 - scatter_max) * cosine_weighted * ao;
    return lit;
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

// Tangent frame from screen-space derivatives, so a normal map works without
// tangent attributes.
//
// The vertex stage here carries position, normal, uv and colour and no tangent,
// and adding one would mean touching every geometry path for a slot most of
// them never fill. This is the standard alternative (Mikkelsen 2010): the
// derivatives of position and uv across the quad give the same basis the
// tangent would have, evaluated per pixel. It costs four derivatives and agrees
// with a supplied tangent wherever the UVs are not degenerate.
static inline float3 perturb_normal(float3 n,
                                    float3 world_pos,
                                    float2 uv,
                                    float3 map_sample,
                                    float scale) {
    float3 dp1 = dfdx(world_pos);
    float3 dp2 = dfdy(world_pos);
    float2 duv1 = dfdx(uv);
    float2 duv2 = dfdy(uv);
    float3 dp2perp = cross(dp2, n);
    float3 dp1perp = cross(n, dp1);
    float3 t = dp2perp * duv1.x + dp1perp * duv2.x;
    float3 b = dp2perp * duv1.y + dp1perp * duv2.y;
    float det = max(dot(t, t), dot(b, b));
    if (det < 1e-16) {
        return n;   // degenerate UVs: no frame to build, so leave the normal be
    }
    float inv = rsqrt(det);
    float3 m = map_sample * 2.0 - 1.0;
    m.xy *= scale;
    return normalize(t * (inv * m.x) + b * (inv * m.y) + n * m.z);
}

fragment float4 fs_mesh(FIn in [[stage_in]],
                        bool front_facing [[front_facing]],
                        constant Frame   &frame        [[buffer(0)]],
                        constant Draw    &draw         [[buffer(1)]],
                        texture2d<float>  base_map     [[texture(0)]],
                        texture2d<float>  emissive_map [[texture(1)]],
                        texture2d<float>  normal_map   [[texture(2)]],
                        texture2d<float>  mr_map       [[texture(3)]],
                        texture2d<float>  ao_map       [[texture(4)]],
                        texturecube<float> env_map     [[texture(5)]],
                        depth2d<float>    shadow_map   [[texture(6)]],
                        depth2d<float>    spot_map     [[texture(7)]],
                        depthcube<float>  point_map    [[texture(8)]],
                        sampler           samp         [[sampler(0)]],
                        sampler           env_samp     [[sampler(1)]],
                        sampler           shadow_samp  [[sampler(2)]]) {
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

    // Emissive is its own channel and applies to lit and unlit alike: a phone
    // screen emits its picture rather than reflecting it, and that is the case
    // this slot exists for.
    float3 emissive = draw.emissive.rgb;
    if (draw.maps.x == 1u) {
        emissive *= emissive_map.sample(samp, in.uv).rgb;
    }

    bool lit_kind = kind == KIND_LAMBERT || kind == KIND_PHONG || kind == KIND_STANDARD
                 || kind == KIND_PHYSICAL || kind == KIND_TOON;
    float3 color;
    if (lit_kind) {
        if (draw.maps.y == 1u) {
            n = perturb_normal(n, in.world_pos, in.uv,
                               normal_map.sample(samp, in.uv).rgb,
                               draw.map_params.x);
        }
        float roughness = draw.pbr.x;
        float metalness = draw.pbr.y;
        if (draw.maps.z == 1u) {
            // glTF packing: roughness in green, metalness in blue, both as
            // multipliers on the material's scalars.
            float3 mr = mr_map.sample(samp, in.uv).rgb;
            roughness *= mr.g;
            metalness *= mr.b;
        }
        float ao = 1.0;
        if (draw.maps.w == 1u) {
            ao = mix(1.0, ao_map.sample(samp, in.uv).r, draw.map_params.y);
        }

        Surface s;
        s.n = n;
        s.v = normalize(frame.camera_pos[in.view_index].xyz - in.world_pos);
        s.albedo = base.rgb;
        s.specular = draw.specular.rgb;
        s.shininess = draw.specular.w;
        s.roughness = roughness;
        s.metalness = metalness;
        s.kind = kind;
        s.toon_steps = draw.pbr.w;
        color = shade(s, in.world_pos, frame, ao, shadow_map, spot_map, point_map,
                      shadow_samp, draw.object.x == 1u);
        if (draw.map_params.w > 0.5 && (kind == KIND_STANDARD || kind == KIND_PHYSICAL)) {
            color += ibl(s, env_map, env_samp, ao, draw.map_params.z);
        }
        color += emissive;
    } else {
        // Unlit: Basic, Line, Points, Sprite, Sky, Mirror, Shader, Atmosphere.
        color = base.rgb + emissive;
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
