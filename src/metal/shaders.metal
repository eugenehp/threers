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
// `alpha` is the colour attribute's fourth component (1 for a 3-component
// colour), multiplied into opacity as three.js does with `vertexAlphas`.
struct Vertex {
    packed_float3 position;
    packed_float3 normal;
    float2        uv;
    packed_float3 color;
    float         alpha;
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

/// A `RectAreaLight`: centre, colour × intensity, half width and half
/// height as vectors (three.js's `halfWidth`, `halfHeight`).
struct RectLight {
    float4 position;
    float4 color;
    float4 right;
    float4 up;
};

struct HemiLight {
    float4 sky;
    float4 ground;
    float4 up;
};

// Per-view state. One view is a plain camera; two are the eyes of a headset,
// and the vertex stage picks between them from the instance id; four are the
// panes of a quad view. Must match `MAX_VIEWS` in renderer.rs.
constant uint MAX_VIEWS = 4u;

/// A shadow-casting light's shadow — `ShadowRecord` in
/// `renderer/shader.rs`, field for field.
struct ShadowRecord {
    uint4    info;    // kind (0 directional, 1 spot, 2 point), first layer, cascades, filter
    float4   params;  // depth bias, normal bias, filter radius, one texel
    float4   extra;   // directional: cascade far edges; point: position, radius
    float4   pcss;    // search radius, spot near, spot far, penumbra scale
    float4x4 vp[4];   // world -> light clip, a cascade each
};

/// What one shadow pass draws with: its matrix, and for a point light its
/// position and radius.
struct ShadowPass {
    float4x4 shadow_vp;
    float4   point_shadow_pos;
};

struct Frame {
    float4x4   view_proj[MAX_VIEWS];
    float4x4   view[MAX_VIEWS];
    float4     camera_pos[MAX_VIEWS];
    float4     ambient;
    float4     fog_color;   // rgb, w = mode (0 off, 1 linear, 2 exp2)
    float4     fog_params;  // near, far, density
    float4     viewport;    // width, height, 1/width, 1/height
    uint4      counts;      // dir, point, spot, hemi
    uint4      view_info;   // view count; y: 1 when the target is sRGB (encodes on write); rest unused
    DirLight   dir[4];
    PointLight pts[8];
    SpotLight  spots[4];
    HemiLight  hemis[2];
    uint4      rect_counts; // x: RectAreaLights; y: the transmission capture's mip count;
                            // z: 1 for reflective transparency; w: view 0's capture slice
    RectLight  rects[4];
    /// x: shadow records in use.
    uint4      shadow_info;
    /// One per shadow-casting light, which names it in its own `color.w`
    /// (record + 1).
    ShadowRecord shadows[16];
};

struct Draw {
    float4x4 model;
    float4x4 normal_mat;
    float4   base_color;   // rgb, a = opacity
    float4   emissive;     // rgb
    float4   specular;     // rgb, w = shininess
    float4   pbr;          // roughness, metalness, alpha_test, toon gradientMap bound
    float4   misc;         // unused, point size, depth near, depth far
    uint4    flags;        // kind, has_map, has_vertex_color, instanced
    uint4    maps;         // emissive, normal, metallic-roughness, AO: bound?
    float4   map_params;   // normal scale, AO strength, env intensity, has env
    uint4    object;       // x: receives shadow; y: 1 for a sprite; z: 1 for ShadowMaterial; w: tone mapped
    uint4    extra;        // bound: x alpha map, y light map, z bump map, w specular map
    float4   map_scalars;  // lightMapIntensity, bumpScale, envMap bound, matcap bound
    float4   env_params;   // combine, reflectivity, refractionRatio, refraction mapping
    float4   env_dir[3];   // rows of envMapRotation · diag(flipEnvMap, 1, 1)
    // Each map's sampler, as an index into the sampler pool (`map_samps`),
    // four to a vector in `MAP_*` order — so every texture keeps its own
    // filter and wrap within Metal's sixteen sampler slots.
    uint4    map_samplers[6];
    // MeshPhysicalMaterial (zero for anything else):
    // [0] ior, specularIntensity, clearcoat, clearcoatRoughness
    // [1] specularColor, sheen
    // [2] sheenColor, sheenRoughness
    // [3] iridescence, iridescenceIOR, iridescenceThicknessRange
    // [4] anisotropy, anisotropyRotation, clearcoatNormalScale
    // [5] transmission, thickness, attenuationDistance (0 = infinite), dispersion
    // [6] attenuationColor
    float4   physical[7];
    // x: which physical maps are bound, bit `MAP_* − MAP_PHYS0`;
    // y: 1 for geometric specular antialiasing in place of three.js's
    // `geometryRoughness` (`set_specular_antialiasing`); z: what the
    // geometry's per-vertex extras (`buffer(8)`) hold — 1 uv sets, 2 tangents.
    uint4    phys_flags;
    // Each map's own uv transform, as two rows of three.js's `Texture.matrix`
    // (`setUvTransform`), in `MAP_*` order.
    float4   map_xform[44];
};

constant uint MAP_BASE     = 0u;
constant uint MAP_EMISSIVE = 1u;
constant uint MAP_NORMAL   = 2u;
constant uint MAP_MR       = 3u;
constant uint MAP_AO       = 4u;
constant uint MAP_ALPHA    = 5u;
constant uint MAP_LIGHT    = 6u;
constant uint MAP_BUMP     = 7u;
constant uint MAP_SPECULAR = 8u;
constant uint MAP_MATCAP   = 9u;
// MeshPhysicalMaterial's maps, at textures 16–27 in this order.
constant uint MAP_PHYS0                = 10u;
constant uint MAP_SPECULAR_INTENSITY   = 10u;
constant uint MAP_SPECULAR_COLOR       = 11u;
constant uint MAP_CLEARCOAT            = 12u;
constant uint MAP_CLEARCOAT_ROUGHNESS  = 13u;
constant uint MAP_CLEARCOAT_NORMAL     = 14u;
constant uint MAP_IRIDESCENCE          = 15u;
constant uint MAP_IRIDESCENCE_THICKNESS = 16u;
constant uint MAP_SHEEN_COLOR          = 17u;
constant uint MAP_SHEEN_ROUGHNESS      = 18u;
constant uint MAP_TRANSMISSION         = 19u;
constant uint MAP_THICKNESS            = 20u;
constant uint MAP_ANISOTROPY           = 21u;
constant uint PHYS_MAP_COUNT           = 12u;
constant uint SAMPLER_POOL             = 10u;

/// Map `i`'s sampler's place in the pool.
static inline uint map_sampler(constant Draw &draw, uint i) {
    return draw.map_samplers[i / 4u][i % 4u];
}

/// Whether physical map `i` (a `MAP_*` from `MAP_PHYS0`) is bound.
static inline bool has_phys(constant Draw &draw, uint i) {
    return (draw.phys_flags.x >> (i - MAP_PHYS0) & 1u) != 0u;
}

/// The physical maps' texels at this fragment, sampled in the entry point
/// (white where a map is absent, and a flat normal for the coat's).
struct PhysTexels {
    float4 v[12];
};

static inline float4 phys_texel(thread const PhysTexels &p, uint i) {
    return p.v[i - MAP_PHYS0];
}

/// `uv` through map `i`'s transform — three.js's `vMapUv`, `vEmissiveMapUv`
/// and the rest, one per map.
static inline float2 map_uv(constant Draw &draw, uint i, float2 uv) {
    float3 h = float3(uv, 1.0);
    return float2(dot(draw.map_xform[2u * i].xyz, h), dot(draw.map_xform[2u * i + 1u].xyz, h));
}

/// `uv` through map `i`'s transform after choosing the uv set its `channel`
/// names (row 0's w): `uv`, `uv1`, `uv2` or `uv3`.
static inline float2 map_uv_sets(constant Draw &draw, uint i, float2 uv, float4 uv12, float2 uv3) {
    float c = draw.map_xform[2u * i].w;
    float2 src = c < 0.5 ? uv : (c < 1.5 ? uv12.xy : (c < 2.5 ? uv12.zw : uv3));
    return map_uv(draw, i, src);
}

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
    /* three.js `uv1` (xy) and `uv2` (zw), and `uv3`.                       */ \
    float4 uv12;                                                               \
    float2 uv3;                                                                \
    /* The `tangent` attribute in world space, w its handedness; zero when  */ \
    /* the geometry has none.                                               */ \
    float4 tangent;                                                            \
    float4 color;                                                              \
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

/// Highest bone index the skinning reads. The renderer pads the bone buffer
/// with identities to 128 matrices — the wgpu backend's fixed uniform array —
/// and joints clamp here, so both backends (and the CPU raycaster) read the
/// same matrix for every joint index.
constant uint MAX_BONE = 127u;

/// The weighted blend of a vertex's four bones: three.js's `Σ wᵢ · boneMatᵢ`.
///
/// Not renormalised: three.js normalises weights on load
/// (`normalizeSkinWeights`), not in the shader, and draws whatever an
/// un-normalised set gives. Normalising here would draw a mesh the other
/// backend, the raycaster and three.js all put somewhere else.
static inline float4x4 skin_matrix(SkinVertex sv, device const float4x4 *bones) {
    float4x4 m = float4x4(0.0);
    for (uint k = 0u; k < 4u; ++k) {
        if (sv.weights[k] != 0.0) {
            m += bones[min(uint(max(sv.joints[k], 0.0)), MAX_BONE)] * sv.weights[k];
        }
    }
    return m;
}

/// The one matrix that takes a rest position to world space the way
/// three.js's `skinning_vertex.glsl` does with an identity `bindMatrix` and
/// `bindMatrixInverse = inverse(model)` (attached bind mode):
///
///     skinned = skin * (p, 1)                  // w = the weight sum
///     world   = model * ((inverse(model) * skinned).xyz, 1)
///             = skinned.xyz + (1 - skinned.w) * translation(model)
///
/// The model cancels when the weights sum to one. Bone matrices are affine, so
/// `skinned.w` is `skin[3].w`. See `SkinnedMesh` on the Rust side.
static inline float4x4 skinned_model(float4x4 model, float4x4 skin) {
    float4x4 m = skin;
    m[3].xyz += (1.0 - skin[3].w) * model[3].xyz;
    m[0].w = 0.0;
    m[1].w = 0.0;
    m[2].w = 0.0;
    m[3].w = 1.0;
    return m;
}

/// `skinnormal_vertex.glsl` then the normal matrix: `N · inverse(model) ·
/// skin`, where `inverse(model)`'s linear part is `transpose(N)` — so for a
/// rigid model it is just the skin's linear part.
static inline float3x3 skinned_normal_basis(constant Draw &draw, float4x4 skin) {
    float3x3 n = float3x3(draw.normal_mat[0].xyz, draw.normal_mat[1].xyz, draw.normal_mat[2].xyz);
    return n * transpose(n) * float3x3(skin[0].xyz, skin[1].xyz, skin[2].xyz);
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
    out.uv12 = v.uv12;                                                         \
    out.uv3 = v.uv3;                                                           \
    out.tangent = v.tangent;                                                   \
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

// ---------------------------------------------------------------- morph targets

/// three.js's morph-target blend (`morphtarget_vertex`, `morphnormal_vertex`,
/// `morphcolor_vertex`): `base · attribute + Σ wᵢ · targetᵢ`, where `base` is
/// 1 for relative targets and `1 − Σ influences` for absolute ones (computed
/// on the CPU, as `WebGLMorphtargets` does). The targets live in one float4
/// buffer laid out like three.js's morph texture: target `t`, vertex `v`,
/// slot `k` (0 position, 1 normal, 2 colour) at `(t · vertex_count + v) ·
/// stride + k`. `MorphInfo` twins `MorphUniforms` in renderer.rs.
struct MorphInfo {
    uint4  info;        // x: target count (0 = off), y: stride, z: flags (1 normal, 2 colour, 4 position, 8 colour alpha, 16 per instance), w: vertex count
    float4 base;        // x: morphTargetBaseInfluence
    float4 weights[16]; // 64 influences
};

/// `iid` is the instance drawn and `instances` the instance buffer: for an
/// `InstancedMesh` with a morph texture (`info.z & 16`) the per-instance rows
/// — three.js's `morphTexture`, `morphTargetBaseInfluence` then the
/// influences, `base.y` floats each — follow the matrices, `base.z` floats in.
static inline Vertex apply_morph(Vertex v,
                                 uint vid,
                                 uint iid,
                                 constant MorphInfo &m,
                                 device const float4 *data,
                                 device const float4x4 *instances) {
    uint count = m.info.x;
    if (count == 0u) {
        return v;
    }
    uint stride = m.info.y;
    uint vc = m.info.w;
    bool per_instance = (m.info.z & 16u) != 0u;
    device const float *row = (device const float *)instances
        + uint(m.base.z) + iid * uint(m.base.y);
    float b = per_instance ? row[0] : m.base.x;
    bool has_p = (m.info.z & 4u) != 0u;
    float3 p = has_p ? float3(v.position) * b : float3(v.position);
    bool has_n = (m.info.z & 1u) != 0u;
    bool has_c = (m.info.z & 2u) != 0u;
    float3 n = has_n ? float3(v.normal) * b : float3(v.normal);
    // `morphcolor_vertex.glsl`: all four channels for a 4-component colour
    // (USE_COLOR_ALPHA), rgb otherwise.
    bool has_ca = (m.info.z & 8u) != 0u;
    float4 c0 = float4(float3(v.color), v.alpha);
    float4 c = has_c ? c0 * b : c0;
    for (uint i = 0u; i < count; ++i) {
        float w = per_instance ? row[i + 1u] : m.weights[i / 4u][i % 4u];
        if (w == 0.0) {
            continue;
        }
        uint at = (i * vc + vid) * stride;
        if (has_p) {
            p += data[at].xyz * w;
        }
        if (has_n) {
            n += data[at + 1u].xyz * w;
        }
        if (has_c) {
            c += data[at + 2u] * w;
        }
    }
    v.position = packed_float3(p);
    v.normal = packed_float3(n);
    v.color = packed_float3(c.rgb);
    v.alpha = has_ca ? c.a : c0.a;
    return v;
}

// --------------------------------------------------------------------- vertex

static inline float4x4 model_matrix(constant Draw &draw,
                                    device const float4x4 *instances,
                                    uint iid) {
    // Only pay for the instance multiply when there is an instance to apply;
    // a 4x4 product per vertex is not free at scene scale.
    return (draw.flags.w & 1u) == 1u ? draw.model * instances[iid] : draw.model;
}

/// three.js `instanceColor` (`InstancedMesh.setColorAt`): `color_vertex`
/// multiplies it into `vColor`, after any colour attribute. The colours are
/// float4s in the instance buffer, after the matrices (and the morph rows),
/// starting `draw.flags.w >> 1` float4s in; 0 there means the mesh has none.
static inline float4 instance_color(constant Draw &draw,
                                    device const float4x4 *instances,
                                    uint iid) {
    uint at = draw.flags.w >> 1;
    if ((draw.flags.w & 1u) == 0u || at == 0u) {
        return float4(1.0);
    }
    return float4(((device const float4 *)instances)[at + iid].rgb, 1.0);
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
    out.uv = v.uv;
    out.uv12 = float4(0.0);
    out.uv3 = float2(0.0);
    out.tangent = float4(0.0);
    out.color = draw.flags.z == 1u ? float4(float3(v.color), v.alpha) : float4(1.0);
    out.view_dist = length(frame.camera_pos[view_index].xyz - world.xyz);
    out.point_size = 1.0;
    out.view_index = view_index;
    return out;
}

static inline float3x3 normal_basis(constant Draw &draw,
                                    device const float4x4 *instances,
                                    uint iid) {
    float3x3 nm = float3x3(draw.normal_mat[0].xyz,
                           draw.normal_mat[1].xyz,
                           draw.normal_mat[2].xyz);
    if ((draw.flags.w & 1u) == 1u) {
        // `defaultnormal_vertex.glsl` under USE_INSTANCING: the instance's
        // linear part with each column divided by its squared length — so an
        // axis-scaled instance keeps its normals perpendicular — then the
        // mesh's normal matrix.
        float4x4 i4 = instances[iid];
        float3x3 im = float3x3(i4[0].xyz, i4[1].xyz, i4[2].xyz);
        float3x3 inv_s = float3x3(float3(1.0 / dot(im[0], im[0]), 0.0, 0.0),
                                  float3(0.0, 1.0 / dot(im[1], im[1]), 0.0),
                                  float3(0.0, 0.0, 1.0 / dot(im[2], im[2])));
        return nm * im * inv_s;
    }
    return nm;
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

/// A `Sprite` draw (`draw.object.y == 1`): three.js's `sprite_vert`, the unit
/// quad turned to face view `view_index`. `draw.normal_mat[0]` carries
/// (centre − 0.5, rotation, scale-by-depth) — the normal matrix means nothing
/// to an unlit, camera-facing quad. The wgpu backend's `vs_sprite` and the
/// raycaster build the same quad.
static inline VOut sprite_vertex(Vertex v,
                                 constant Frame &frame,
                                 constant Draw &draw,
                                 uint view_index) {
    float4 params = draw.normal_mat[0];
    float4x4 view = frame.view[view_index];
    float4 center_view = view * float4(draw.model[3].xyz, 1.0);
    float2 scale = float2(length(draw.model[0].xyz), length(draw.model[1].xyz));
    if (params.w > 0.5) {
        scale *= -center_view.z;
    }
    float2 aligned = (float3(v.position).xy - params.xy) * scale;
    float c = cos(params.z);
    float s = sin(params.z);
    float2 rotated = float2(c * aligned.x - s * aligned.y, s * aligned.x + c * aligned.y);
    float3 view_pos = float3(center_view.xy + rotated, center_view.z);
    // Back to world space: a view matrix is rigid, so its inverse rotation is
    // the transpose.
    float3x3 r = float3x3(view[0].xyz, view[1].xyz, view[2].xyz);
    float3 world = transpose(r) * (view_pos - view[3].xyz);
    // `transform` does the rest (uv, colour, distances), handed a translation
    // that takes this vertex to its billboarded place.
    float4x4 model = float4x4(1.0);
    model[3] = float4(world - float3(v.position), 1.0);
    float3x3 facing = transpose(r);
    return transform(v, model, facing, frame, draw, view_index);
}

struct TangentPair {
    float3 t;
    float3 b;
};

/// three.js's `tbn` from the `tangent` attribute (`normal_fragment_begin`):
/// the tangent normalised, the bitangent `cross(normal, tangent) · w` — the
/// frame the anisotropy and the coat's normal map read — or `fallback`
/// (`getTangentFrame`) without one.
static inline TangentPair tbn_frame(float4 vt, float3 vertex_normal, TangentPair fallback) {
    if (dot(vt.xyz, vt.xyz) < 1e-12) {
        return fallback;
    }
    TangentPair f;
    f.t = normalize(vt.xyz);
    f.b = normalize(cross(normalize(vertex_normal), f.t) * (vt.w >= 0.0 ? 1.0 : -1.0));
    return f;
}

/// Vertex `vid`'s extra uv sets and tangent into `out` (zero without
/// them); the tangent through `model`, a direction along the surface.
static inline void read_uv_sets(thread VOut &out, uint vid, constant Draw &draw,
                                device const float4 *uv_sets, float4x4 model) {
    if ((draw.phys_flags.z & 1u) != 0u) {
        out.uv12 = uv_sets[3u * vid];
        out.uv3 = uv_sets[3u * vid + 1u].xy;
    }
    if ((draw.phys_flags.z & 2u) != 0u) {
        float4 t = uv_sets[3u * vid + 2u];
        out.tangent = float4((model * float4(t.xyz, 0.0)).xyz, t.w);
    }
}

/// The tangent frame for a normal map or anisotropy: the `tangent`
/// attribute's, orthogonalised against `n`, with the bitangent from its
/// handedness — as `surface_frame` in `renderer/shader.rs` — when the geometry
/// has one; else `fallback`, from screen-space derivatives. Flipped with a
/// back face (`face` −1), as three.js's `faceDirection` flips `tbn`.
static inline TangentPair attribute_frame(float4 vt, float3 n, TangentPair fallback) {
    float3 t = vt.xyz - n * dot(n, vt.xyz);
    if (dot(vt.xyz, vt.xyz) < 1e-12 || dot(t, t) < 1e-12) {
        return fallback;
    }
    TangentPair f;
    f.t = normalize(t);
    f.b = normalize(cross(n, f.t)) * (vt.w >= 0.0 ? 1.0 : -1.0);
    return f;
}

vertex VOut vs_mesh(uint vid [[vertex_id]],
                    uint iid [[instance_id]],
                    device const Vertex   *verts     [[buffer(0)]],
                    constant Frame        &frame     [[buffer(1)]],
                    constant Draw         &draw      [[buffer(2)]],
                    device const float4x4 *instances [[buffer(3)]],
                    device const float4   *morph_data [[buffer(6)]],
                    constant MorphInfo    &morph      [[buffer(7)]],
                    device const float4   *uv_sets    [[buffer(8)]]) {
    if (draw.object.y == 1u) {
        return sprite_vertex(verts[vid], frame, draw, 0u);
    }
    float4x4 model = model_matrix(draw, instances, iid);
    Vertex v = apply_morph(verts[vid], vid, iid, morph, morph_data, instances);
    VOut out = transform(v, model, normal_basis(draw, instances, iid), frame, draw, 0u);
    out.color *= instance_color(draw, instances, iid);
    read_uv_sets(out, vid, draw, uv_sets, model);
    return out;
}

/// Skinned transform: three.js's skinning, see `skinned_model`.
///
/// `bones[i]` is already world × inverse-bind for bone `i` — `Skeleton::update`
/// composes that on the CPU — so the work here is the weighted sum. Four
/// influences, which is what glTF stores and what three.js reads.
///
/// The normal is transformed by the same blend rather than by its own inverse
/// transpose, as three.js's `skinnormal_vertex.glsl` does. That is exact for
/// rigid bones and wrong only under non-uniform bone scale.
vertex VOut vs_skinned(uint vid [[vertex_id]],
                       uint iid [[instance_id]],
                       device const Vertex     *verts     [[buffer(0)]],
                       constant Frame          &frame     [[buffer(1)]],
                       constant Draw           &draw      [[buffer(2)]],
                       device const float4x4   *instances [[buffer(3)]],
                       device const SkinVertex *skin      [[buffer(4)]],
                       device const float4x4   *bones     [[buffer(5)]],
                       device const float4     *morph_data [[buffer(6)]],
                       constant MorphInfo      &morph      [[buffer(7)]],
                       device const float4     *uv_sets    [[buffer(8)]]) {
    float4x4 sk = skin_matrix(skin[vid], bones);
    float4x4 model = skinned_model(model_matrix(draw, instances, iid), sk);
    // Morph first, then skin — three.js's chunk order.
    Vertex v = apply_morph(verts[vid], vid, iid, morph, morph_data, instances);
    VOut out = transform(v, model, skinned_normal_basis(draw, sk), frame, draw, 0u);
    read_uv_sets(out, vid, draw, uv_sets, model);
    return out;
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
                        constant ShadowPass     &frame     [[buffer(1)]],
                        constant Draw           &draw      [[buffer(2)]],
                        device const float4x4   *instances [[buffer(3)]],
                        device const SkinVertex *skin      [[buffer(4)]],
                        device const float4x4   *bones     [[buffer(5)]],
                        device const float4     *morph_data [[buffer(6)]],
                        constant MorphInfo      &morph      [[buffer(7)]]) {
    float4x4 model = model_matrix(draw, instances, iid);
    if (draw.maps.w == 2u) {   // 2 marks a skinned draw; see `DrawUniforms`
        model = skinned_model(model, skin_matrix(skin[vid], bones));
    }
    Vertex v = apply_morph(verts[vid], vid, iid, morph, morph_data, instances);
    return frame.shadow_vp * model * float4(float3(v.position), 1.0);
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
    float2 uv;
};

/// A cut-out caster's clip position and map uv (directional and spot): three.js's
/// shadow depth material takes the caster's `map` and `alphaTest`
/// (`WebGLShadowMap.getDepthMaterial`), not its opacity.
struct ShadowCutOut {
    float4 position [[position]];
    float2 uv;
};

vertex ShadowCutOut vs_shadow_cut(uint vid [[vertex_id]],
                                  uint iid [[instance_id]],
                                  device const Vertex     *verts     [[buffer(0)]],
                                  constant ShadowPass     &frame     [[buffer(1)]],
                                  constant Draw           &draw      [[buffer(2)]],
                                  device const float4x4   *instances [[buffer(3)]],
                                  device const SkinVertex *skin      [[buffer(4)]],
                                  device const float4x4   *bones     [[buffer(5)]],
                                  device const float4     *morph_data [[buffer(6)]],
                                  constant MorphInfo      &morph      [[buffer(7)]]) {
    float4x4 model = model_matrix(draw, instances, iid);
    if (draw.maps.w == 2u) {
        model = skinned_model(model, skin_matrix(skin[vid], bones));
    }
    Vertex v = apply_morph(verts[vid], vid, iid, morph, morph_data, instances);
    ShadowCutOut o;
    o.position = frame.shadow_vp * model * float4(float3(v.position), 1.0);
    o.uv = v.uv;
    return o;
}

/// Whether a shadow caster's fragment is cut away: below `alphaTest` in its
/// map's alpha. A caster with no map, or no test, is never cut.
/// Its `alphaMap`'s green channel too, as three.js's depth material takes it.
static inline bool shadow_cut(constant Draw &draw, texture2d<float> base_map,
                              sampler samp, texture2d<float> alpha_map,
                              sampler alpha_samp, float2 uv) {
    bool has_map = draw.flags.y == 1u;
    bool has_alpha = draw.extra.x == 1u;
    if (draw.pbr.z <= 0.0 || !(has_map || has_alpha)) {
        return false;
    }
    float a = 1.0;
    if (has_map) {
        a = base_map.sample(samp, map_uv(draw, MAP_BASE, uv), level(0.0)).a;
    }
    if (has_alpha) {
        a *= alpha_map.sample(alpha_samp, map_uv(draw, MAP_ALPHA, uv), level(0.0)).g;
    }
    return a < draw.pbr.z;
}

fragment void fs_shadow_cut(ShadowCutOut in [[stage_in]],
                            constant Draw &draw [[buffer(2)]],
                            texture2d<float> base_map [[texture(0)]],
                            texture2d<float> alpha_map [[texture(1)]],
                            sampler samp [[sampler(0)]],
                            sampler alpha_samp [[sampler(1)]]) {
    if (shadow_cut(draw, base_map, samp, alpha_map, alpha_samp, in.uv)) {
        discard_fragment();
    }
}

vertex ShadowPointOut vs_shadow_point(uint vid [[vertex_id]],
                                      uint iid [[instance_id]],
                                      device const Vertex     *verts     [[buffer(0)]],
                                      constant ShadowPass     &frame     [[buffer(1)]],
                                      constant Draw           &draw      [[buffer(2)]],
                                      device const float4x4   *instances [[buffer(3)]],
                                      device const SkinVertex *skin      [[buffer(4)]],
                                      device const float4x4   *bones     [[buffer(5)]],
                                      device const float4     *morph_data [[buffer(6)]],
                                      constant MorphInfo      &morph      [[buffer(7)]]) {
    float4x4 model = model_matrix(draw, instances, iid);
    if (draw.maps.w == 2u) {
        model = skinned_model(model, skin_matrix(skin[vid], bones));
    }
    Vertex v = apply_morph(verts[vid], vid, iid, morph, morph_data, instances);
    float4 world = model * float4(float3(v.position), 1.0);
    ShadowPointOut o;
    o.position = frame.shadow_vp * world;
    o.world = world.xyz;
    o.uv = v.uv;
    return o;
}

struct ShadowDepth {
    float depth [[depth(any)]];
};

fragment ShadowDepth fs_shadow_point(ShadowPointOut in [[stage_in]],
                                     constant ShadowPass &frame [[buffer(1)]],
                                     constant Draw &draw [[buffer(2)]],
                                     texture2d<float> base_map [[texture(0)]],
                                     texture2d<float> alpha_map [[texture(1)]],
                                     sampler samp [[sampler(0)]],
                                     sampler alpha_samp [[sampler(1)]]) {
    if (shadow_cut(draw, base_map, samp, alpha_map, alpha_samp, in.uv)) {
        discard_fragment();
    }
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
    v.alpha    = 1.0;
    float4x4 model = model_matrix(draw, instances, iid);
    return transform(v, model, normal_basis(draw, instances, iid), frame, draw, 0u);
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
    out.uv12 = float4(0.0);
    out.uv3 = float2(0.0);
    out.tangent = float4(0.0);
    out.uv = float2(side, half_px);
    out.color = draw.flags.z == 1u ? float4(float3(me.color), 1.0) : float4(1.0);
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
    // three.js `alphaTest` applies to the material alpha, before coverage.
    if (draw.base_color.a * in.color.a < draw.pbr.z) {
        discard_fragment();
    }
    float4 c = float4(draw.base_color.rgb * in.color.rgb, draw.base_color.a * cov);
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
                     device const float4x4 *instances [[buffer(3)]],
                     device const float4   *morph_data [[buffer(6)]],
                     constant MorphInfo    &morph      [[buffer(7)]]) {
    float4x4 model = model_matrix(draw, instances, iid);
    Vertex v = apply_morph(verts[vid], vid, iid, morph, morph_data, instances);
    VOut out = transform(v, model, normal_basis(draw, instances, iid), frame, draw, 0u);
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
                                   device const float4x4 *instances [[buffer(3)]],
                                   device const float4   *morph_data [[buffer(6)]],
                                   constant MorphInfo    &morph      [[buffer(7)]],
                                   device const float4   *uv_sets    [[buffer(8)]]) {
    uint view = view_of(iid, frame);
    if (draw.object.y == 1u) {
        return to_layered(sprite_vertex(verts[vid], frame, draw, view));
    }
    float4x4 model = model_matrix(draw, instances, instance_of(iid, frame));
    Vertex v = apply_morph(verts[vid], vid, instance_of(iid, frame), morph, morph_data, instances);
    VOut out = transform(v, model, normal_basis(draw, instances, instance_of(iid, frame)), frame, draw, view);
    out.color *= instance_color(draw, instances, instance_of(iid, frame));
    read_uv_sets(out, vid, draw, uv_sets, model);
    return to_layered(out);
}

vertex VOutLayeredPoint vs_point_layered(uint vid [[vertex_id]],
                                    uint iid [[instance_id]],
                                    device const Vertex   *verts     [[buffer(0)]],
                                    constant Frame        &frame     [[buffer(1)]],
                                    constant Draw         &draw      [[buffer(2)]],
                                    device const float4x4 *instances [[buffer(3)]],
                                    device const float4   *morph_data [[buffer(6)]],
                                    constant MorphInfo    &morph      [[buffer(7)]]) {
    uint view = view_of(iid, frame);
    float4x4 model = model_matrix(draw, instances, instance_of(iid, frame));
    Vertex v = apply_morph(verts[vid], vid, instance_of(iid, frame), morph, morph_data, instances);
    VOut out = transform(v, model, normal_basis(draw, instances, instance_of(iid, frame)), frame, draw, view);
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
    float  toon_gradient; // 1 when a gradientMap is bound (in the matcap slot)
    /// Screen-space derivatives of the normal, for the toon ramp's
    /// antialiased step (three.js's `fwidth(coord)`).
    float3 dndx;
    float3 dndy;
    /// three.js `lightMap` irradiance (texel × `lightMapIntensity`), added to
    /// the indirect light (`lights_fragment_maps`).
    float3 light_map;
    /// three.js `specularStrength`: the `specularMap`'s red channel, else 1.
    float  specular_strength;
    /// Standard/Physical: the roughness the environment is read at —
    /// `max(roughness, 0.0525)` without the geometry term, as the wgpu path
    /// reads it.
    float  roughness_env;
    /// three.js r165 `PhysicalMaterial`'s specular lobe, as
    /// `lights_physical_fragment` fills it (Standard: F0 0.04, F90 1).
    float3 f0;
    float  f90;
    float  alpha;
    float  irid;
    float3 irid_fresnel;
    /// Anisotropy strength, negative for none; the tangent alpha; the
    /// anisotropy direction and its perpendicular (not unit, as three.js's
    /// `tbn` is not).
    float  aniso;
    float  alpha_t;
    float3 at;
    float3 ab;
    /// `sheenColor × sheen` (zero for none) and `sheenRoughness`.
    float3 sheen_color;
    float  sheen_roughness;
    /// `clearcoat` (zero for none), its roughness and normal.
    float  clearcoat;
    float  cc_roughness;
    float3 cc_n;
};

// three.js `F_Schlick(f0, 1.0, dotVH)` in the spherical-Gaussian form r165 runs.
static inline float3 f_schlick(float3 f0, float dot_vh) {
    float fresnel = exp2((-5.55473 * dot_vh - 6.98316) * dot_vh);
    return f0 * (1.0 - fresnel) + float3(fresnel);
}

/// three.js `getGradientIrradiance`, as `toon_gradient` in
/// `renderer/shader.rs`: with a `gradientMap`, its rgb at `(n·l/2 + ½, 0)`,
/// through `coord`'s own derivatives; without one, 0.7 / 1.0 split at
/// `coord = 0.7`, antialiased. Not clamped to the lit side, as in three.js.
static inline float3 toon_gradient(Surface s, float3 l, texture2d<float> ramp, sampler ramp_samp) {
    float coord = dot(s.n, l) * 0.5 + 0.5;
    float dx = 0.5 * dot(s.dndx, l);
    float dy = 0.5 * dot(s.dndy, l);
    if (s.toon_gradient > 0.5) {
        return ramp.sample(ramp_samp, float2(coord, 0.0), gradient2d(float2(dx, 0.0), float2(dy, 0.0))).rgb;
    }
    float fw = 0.5 * (abs(dx) + abs(dy));
    return float3(mix(0.7, 1.0, smoothstep(0.7 - fw, 0.7 + fw, coord)));
}

// three.js `BRDF_BlinnPhong`, for KIND_PHONG: F_Schlick(specular) · ¼ ·
// (s/2 + 1)/π · (n·h)^s.
static inline float3 phong_specular(Surface s, float3 l) {
    float3 h = normalize(l + s.v);
    float ndh = saturate(dot(s.n, h));
    float vdh = saturate(dot(s.v, h));
    float d = (1.0 / M_PI_F) * (s.shininess * 0.5 + 1.0) * pow(ndh, s.shininess);
    return f_schlick(s.specular, vdh) * (0.25 * d);
}

// three.js `F_Schlick(f0, f90, dotVH)`.
static inline float3 f_schlick_f90(float3 f0, float f90, float dot_vh) {
    float fresnel = exp2((-5.55473 * dot_vh - 6.98316) * dot_vh);
    return f0 * (1.0 - fresnel) + float3(f90 * fresnel);
}

// three.js `evalSensitivity` (`iridescence_fragment`): a Gaussian fit of the
// CIE colour-matching functions at optical path difference `opd`, in linear
// Rec. 709.
static inline float3 eval_sensitivity(float opd, float3 shift) {
    float phase = 2.0 * M_PI_F * opd * 1.0e-9;
    float3 val = float3(5.4856e-13, 4.4201e-13, 5.2481e-13);
    float3 pos = float3(1.6810e+06, 1.7953e+06, 2.2084e+06);
    float3 var = float3(4.3278e+09, 9.3046e+09, 6.6121e+09);
    float3 xyz = val * sqrt(2.0 * M_PI_F * var) * cos(pos * phase + shift) * exp(-(phase * phase) * var);
    xyz.x += 9.7470e-14 * sqrt(2.0 * M_PI_F * 4.5282e+09) * cos(2.2399e+06 * phase + shift.x)
           * exp(-4.5282e+09 * phase * phase);
    xyz /= 1.0685e-7;
    const float3x3 xyz_to_rec709 = float3x3(
        float3( 3.2404542, -0.9692660,  0.0556434),
        float3(-1.5371385,  1.8760108, -0.2040259),
        float3(-0.4985314,  0.0415560,  1.0572252));
    return xyz_to_rec709 * xyz;
}

// three.js `evalIridescence`: the reflectance of a `thickness` nm film of
// `film_ior` over a base of reflectance `base_f0`, seen at `cos1`.
static inline float3 eval_iridescence(float outside_ior, float film_ior_in, float cos1,
                                      float thickness, float3 base_f0) {
    float film_ior = mix(outside_ior, film_ior_in, smoothstep(0.0, 0.03, thickness));
    float sin2_sq = (outside_ior / film_ior) * (outside_ior / film_ior) * (1.0 - cos1 * cos1);
    float cos2_sq = 1.0 - sin2_sq;
    if (cos2_sq < 0.0) {
        return float3(1.0);
    }
    float cos2 = sqrt(cos2_sq);
    float r0 = pow((film_ior - outside_ior) / (film_ior + outside_ior), 2.0);
    float r12 = f_schlick_f90(float3(r0), 1.0, cos1).x;
    float t121 = 1.0 - r12;
    float phi12 = film_ior < outside_ior ? M_PI_F : 0.0;
    float phi21 = M_PI_F - phi12;
    float3 sqrt_f0 = sqrt(clamp(base_f0, 0.0, 0.9999));
    float3 base_ior = (1.0 + sqrt_f0) / (1.0 - sqrt_f0);
    float3 r1 = pow((base_ior - film_ior) / (base_ior + film_ior), 2.0);
    float3 r23 = f_schlick_f90(r1, 1.0, cos2);
    float3 phi23 = select(float3(0.0), float3(M_PI_F), base_ior < film_ior);
    float opd = 2.0 * film_ior * thickness * cos2;
    float3 phi = float3(phi21) + phi23;
    float3 r123 = clamp(r12 * r23, 1e-5, 0.9999);
    float3 r123_sqrt = sqrt(r123);
    float3 rs = t121 * t121 * r23 / (float3(1.0) - r123);
    float3 out = r12 + rs;
    float3 cm = rs - t121;
    for (int m = 1; m <= 2; ++m) {
        cm *= r123_sqrt;
        out += cm * 2.0 * eval_sensitivity(float(m) * opd, float(m) * phi);
    }
    return max(out, float3(0.0));
}

// three.js `getTangentFrame`: tangent and bitangent from screen-space
// derivatives of position and uv, scaled together (not each to unit length).
static inline TangentPair three_tangent_frame(float3 p, float3 n, float2 uv) {
    float3 q0 = dfdx(p);
    float3 q1 = dfdy(p);
    float2 st0 = dfdx(uv);
    float2 st1 = dfdy(uv);
    float3 q1perp = cross(q1, n);
    float3 q0perp = cross(n, q0);
    float3 t = q1perp * st0.x + q0perp * st1.x;
    float3 b = q1perp * st0.y + q0perp * st1.y;
    float det = max(dot(t, t), dot(b, b));
    float scale = det == 0.0 ? 0.0 : rsqrt(det);
    TangentPair f;
    f.t = t * scale;
    f.b = b * scale;
    return f;
}

// Geometric specular antialiasing (Kaplanyan et al. 2016 / Filament), as
// `specular_aa_roughness` in `renderer/shader.rs`: the shading normal's
// variance across the pixel folded into alpha.
static inline float specular_aa_roughness(float3 n, float perceptual) {
    float3 du = dfdx(n);
    float3 dv = dfdy(n);
    float variance = 0.25 * (dot(du, du) + dot(dv, dv));
    float widen = min(2.0 * variance, 0.18);
    return clamp(sqrt(perceptual * perceptual + widen), 0.0, 1.0);
}

// three.js `BRDF_GGX`: `F · V · D`, the Fresnel mixed toward the thin film's,
// with Filament's anisotropic V and D when there is anisotropy. As
// `brdf_ggx` in `renderer/shader.rs`.
static inline float3 brdf_ggx(Surface s, float3 l) {
    float3 h = normalize(l + s.v);
    float ndl = saturate(dot(s.n, l));
    float ndv = saturate(dot(s.n, s.v));
    float ndh = saturate(dot(s.n, h));
    float vdh = saturate(dot(s.v, h));
    float3 f = mix(f_schlick_f90(s.f0, s.f90, vdh), s.irid_fresnel, s.irid);
    if (s.aniso >= 0.0) {
        float t_l = dot(s.at, l), t_v = dot(s.at, s.v), t_h = dot(s.at, h);
        float b_l = dot(s.ab, l), b_v = dot(s.ab, s.v), b_h = dot(s.ab, h);
        float gv = ndl * length(float3(s.alpha_t * t_v, s.alpha * b_v, ndv));
        float gl = ndv * length(float3(s.alpha_t * t_l, s.alpha * b_l, ndl));
        float vis = saturate(0.5 / (gv + gl));
        float a2 = s.alpha_t * s.alpha;
        float3 dv = float3(s.alpha * t_h, s.alpha_t * b_h, a2 * ndh);
        float w2 = a2 / dot(dv, dv);
        return f * (vis * a2 * w2 * w2 / M_PI_F);
    }
    float a2 = s.alpha * s.alpha;
    float denom = ndh * ndh * (a2 - 1.0) + 1.0;
    float d = a2 / (M_PI_F * denom * denom);
    // three.js `V_GGX_SmithCorrelated`.
    float gv = ndl * sqrt(a2 + (1.0 - a2) * ndv * ndv);
    float gl = ndv * sqrt(a2 + (1.0 - a2) * ndl * ndl);
    return f * (d * 0.5 / max(gv + gl, 1e-6));
}

// Standard / Physical: three.js r165 `RE_Direct_Physical`, `irradiance ·
// (BRDF_GGX + BRDF_Lambert(diffuseColor))`, `diffuseColor = albedo · (1 −
// metalness)` and no (1 − F) on it.
static inline float3 pbr_direct(Surface s, float3 l, float ndl) {
    float3 diff = s.albedo * (1.0 - s.metalness) * (1.0 / M_PI_F);
    return (diff + brdf_ggx(s, l)) * ndl;
}

// three.js `BRDF_Sheen`: Charlie distribution, Neubelt visibility.
static inline float3 brdf_sheen(Surface s, float3 l) {
    float3 h = normalize(l + s.v);
    float ndl = saturate(dot(s.n, l));
    float ndv = saturate(dot(s.n, s.v));
    float ndh = saturate(dot(s.n, h));
    float alpha = s.sheen_roughness * s.sheen_roughness;
    float inv = 1.0 / alpha;
    float sin2h = max(1.0 - ndh * ndh, 0.0078125);
    float d = (2.0 + inv) * pow(sin2h, inv * 0.5) / (2.0 * M_PI_F);
    float v = saturate(1.0 / (4.0 * (ndl + ndv - ndl * ndv)));
    return s.sheen_color * (d * v);
}

// three.js `BRDF_GGX_Clearcoat` × the coat's n·l: F0 0.04, F90 1.
static inline float3 clearcoat_direct(Surface s, float3 l) {
    float3 h = normalize(l + s.v);
    float ndl = saturate(dot(s.cc_n, l));
    float ndv = saturate(dot(s.cc_n, s.v));
    float ndh = saturate(dot(s.cc_n, h));
    float vdh = saturate(dot(s.v, h));
    float a = max(s.cc_roughness * s.cc_roughness, 0.0016);
    float a2 = a * a;
    float denom = ndh * ndh * (a2 - 1.0) + 1.0;
    float d = a2 / (M_PI_F * denom * denom);
    float gv = ndl * sqrt(a2 + (1.0 - a2) * ndv * ndv);
    float gl = ndv * sqrt(a2 + (1.0 - a2) * ndl * ndl);
    return f_schlick_f90(float3(0.04), 1.0, vdh) * (d * 0.5 / max(gv + gl, 1e-6)) * ndl;
}

// Shadows, as `renderer/shader.rs` reads them: every casting light owns a
// `ShadowRecord` and names it in its own `color.w`, each kind's maps are one
// depth array, and the filters — basic, PCF, soft PCF, PCSS — and cascade
// selection are the same, line for line.

constant uint SHADOW_TAPS = 12u;

static inline float2 shadow_disc_tap(uint k, uint n, float phi) {
    float r = sqrt((float(k) + 0.5) / float(n));
    float t = float(k) * 2.39996323 + phi;
    return float2(cos(t), sin(t)) * r;
}

static inline float shadow_disc_rotation(float3 p) {
    return fract(sin(dot(p, float3(12.9898, 78.233, 37.719))) * 43758.5453) * 6.28318530718;
}

/// The cube face a direction lands on and its uv there, chosen by hand
/// because a point light's faces are six layers of a 2D array.
static inline float3 point_shadow_face(float3 d) {
    float3 a = abs(d);
    float face, sc, tc, ma;
    if (a.x >= a.y && a.x >= a.z) {
        ma = a.x;
        face = d.x > 0.0 ? 0.0 : 1.0;
        sc = d.x > 0.0 ? -d.z : d.z;
        tc = -d.y;
    } else if (a.y >= a.z) {
        ma = a.y;
        face = d.y > 0.0 ? 2.0 : 3.0;
        sc = d.x;
        tc = d.y > 0.0 ? d.z : -d.z;
    } else {
        ma = a.z;
        face = d.z > 0.0 ? 4.0 : 5.0;
        sc = d.z > 0.0 ? d.x : -d.x;
        tc = -d.y;
    }
    return float3(0.5 * (sc / ma + 1.0), 0.5 * (tc / ma + 1.0), face);
}

/// The three kinds' maps and their comparison sampler.
struct ShadowMaps {
    depth2d_array<float> dir;
    depth2d_array<float> spot;
    depth2d_array<float> point;
    sampler samp;
};

static inline float shadow_tap(thread const ShadowMaps &maps, uint kind, int layer, float3 centre,
                               float3 t, float3 b, float2 o, float depth) {
    if (kind == 2u) {
        float3 f = point_shadow_face(centre + t * o.x + b * o.y);
        return maps.point.sample_compare(maps.samp, f.xy, uint(layer + int(f.z)), depth);
    }
    float2 uv = centre.xy + o;
    return kind == 0u ? maps.dir.sample_compare(maps.samp, uv, uint(layer), depth)
                      : maps.spot.sample_compare(maps.samp, uv, uint(layer), depth);
}

static inline float shadow_pcf(thread const ShadowMaps &maps, uint kind, int layer, float3 centre,
                               float3 t, float3 b, float depth, float r, uint taps, float3 world_pos) {
    if (r <= 0.0 || taps <= 1u) {
        return shadow_tap(maps, kind, layer, centre, t, b, float2(0.0), depth);
    }
    float phi = shadow_disc_rotation(world_pos);
    float sum = 0.0;
    for (uint k = 0u; k < taps; ++k) {
        float2 o = shadow_disc_tap(k, taps, phi) * r * 1.5;
        sum += shadow_tap(maps, kind, layer, centre, t, b, o, depth);
    }
    return sum / float(taps);
}

/// Percentage-closer soft shadows with a comparison-only blocker search —
/// see `shadow_pcss` in `renderer/shader.rs`.
static inline float shadow_pcss(thread const ShadowMaps &maps, ShadowRecord rec, int layer, float3 centre,
                                float3 t, float3 b, float depth, float3 world_pos) {
    uint kind = rec.info.x;
    float texel = rec.params.w;
    float search = max(rec.pcss.x, 2.0 * texel);
    float phi = shadow_disc_rotation(world_pos);
    float4 blocked = float4(0.0);
    for (uint k = 0u; k < 12u; ++k) {
        float2 o = shadow_disc_tap(k, 12u, phi) * search;
        for (uint l = 0u; l < 4u; ++l) {
            float z = depth * float(l + 1u) * 0.25;
            blocked[l] += 1.0 - shadow_tap(maps, kind, layer, centre, t, b, o, z);
        }
    }
    blocked /= 12.0;
    if (blocked.w <= 0.0) {
        return 1.0;
    }
    float integral = depth * 0.25 * (blocked.x + blocked.y + blocked.z + 0.5 * blocked.w) / blocked.w;
    float blocker = max(depth - integral, 1e-4);
    float gap;
    if (kind == 0u) {
        gap = (depth - blocker) * rec.pcss.w;
    } else if (kind == 1u) {
        float n = rec.pcss.y, f = rec.pcss.z;
        float dr = n * f / max(f - depth * (f - n), 1e-6);
        float db = n * f / max(f - blocker * (f - n), 1e-6);
        gap = (dr - db) / max(db * dr, 1e-6) * rec.pcss.w;
    } else {
        float radius = rec.extra.w;
        float dr = depth * radius, db = blocker * radius;
        gap = (dr - db) / max(db * dr, 1e-6) * rec.pcss.w;
    }
    float r = clamp(gap, texel, 64.0 * texel);
    return shadow_pcf(maps, kind, layer, centre, t, b, depth, r, 16u, world_pos);
}

static inline float shadow_filter(thread const ShadowMaps &maps, ShadowRecord rec, int layer, float3 centre,
                                  float3 t, float3 b, float depth, float3 world_pos) {
    uint kind = rec.info.x;
    float r = rec.params.z;
    switch (rec.info.w) {
        case 0u: return shadow_tap(maps, kind, layer, centre, t, b, float2(0.0), depth);
        case 2u: return shadow_pcf(maps, kind, layer, centre, t, b, depth, max(r * 2.0, 2.0 * rec.params.w), 24u, world_pos);
        case 3u: return shadow_pcss(maps, rec, layer, centre, t, b, depth, world_pos);
        default: return shadow_pcf(maps, kind, layer, centre, t, b, depth, r, SHADOW_TAPS, world_pos);
    }
}

/// Record `r`'s shadow at a point: 1 lit, 0 shadowed, 1 for no record. The
/// lookup steps off the surface along the normal first (normal bias).
static inline float shadow_at(constant Frame &frame, int r, thread const ShadowMaps &maps,
                              float3 world_pos, float3 n) {
    if (r < 0 || uint(r) >= frame.shadow_info.x) {
        return 1.0;
    }
    ShadowRecord rec = frame.shadows[r];
    uint kind = rec.info.x;
    float3 biased = world_pos + n * rec.params.y;
    int layer = int(rec.info.y);
    if (kind == 2u) {
        float radius = rec.extra.w;
        float3 to_frag = biased - rec.extra.xyz;
        float dist = length(to_frag);
        if (dist > radius) {
            return 1.0;
        }
        // Half a percent of the radius, at most a quarter of a unit.
        float depth = dist / radius - min(0.005, 0.25 / radius) - rec.params.x;
        float3 dir = to_frag / max(dist, 1e-6);
        float3 helper = abs(dir.y) > 0.9 ? float3(1.0, 0.0, 0.0) : float3(0.0, 1.0, 0.0);
        float3 t = normalize(cross(dir, helper));
        return shadow_filter(maps, rec, layer, dir, t, cross(dir, t), depth, world_pos);
    }
    uint cascade = 0u;
    if (kind == 0u && rec.info.z > 1u) {
        float view_depth = -(frame.view[0] * float4(world_pos, 1.0)).z;
        cascade = rec.info.z;
        for (uint c = 0u; c < rec.info.z; ++c) {
            if (view_depth <= rec.extra[c]) {
                cascade = c;
                break;
            }
        }
        if (cascade >= rec.info.z) {
            return 1.0;
        }
    }
    float4 p = rec.vp[cascade] * float4(biased, 1.0);
    float3 ndc = p.xyz / max(p.w, 1e-4);
    float2 uv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }
    return shadow_filter(maps, rec, layer + int(cascade), float3(uv, 0.0), float3(0.0), float3(0.0),
                         ndc.z - rec.params.x, world_pos);
}

/// The shadow of the light whose `color.w` is `color_w`, from a fragment's
/// cache of every record's.
static inline float cached_shadow(thread const float *cache, float color_w) {
    int r = int(color_w + 0.5) - 1;
    return (r < 0 || r >= 16) ? 1.0 : cache[r];
}

/// One direct light (three.js's `RE_Direct_*`). Lambert, Phong and Toon take
/// the irradiance through `BRDF_Lambert` (÷π) — three.js r155+'s physically
/// correct light units, the same as the physical model.
static inline float3 shade_one(Surface s, float3 l, float3 light_color, float attenuation,
                               texture2d<float> ramp, sampler ramp_samp) {
    float ndl = max(dot(s.n, l), 0.0);
    if (attenuation <= 0.0) {
        return float3(0.0);
    }
    float3 c = light_color * attenuation;
    // Standard / Physical: the diffuse half here, the specular in
    // `pbr_specular_one` — three.js keeps `directDiffuse` and `directSpecular`
    // apart, and transmission replaces the diffuse alone.
    if (s.kind == KIND_STANDARD || s.kind == KIND_PHYSICAL) {
        return s.albedo * (1.0 - s.metalness) * (1.0 / M_PI_F) * ndl * c;
    }
    if (s.kind == KIND_TOON) {
        return s.albedo * (1.0 / M_PI_F) * toon_gradient(s, l, ramp, ramp_samp) * c;
    }
    float3 e = c * ndl;
    float3 out = e * s.albedo * (1.0 / M_PI_F);
    if (s.kind == KIND_PHONG) {
        out += e * phong_specular(s, l) * s.specular_strength;
    }
    return out;
}

/// One light's `BRDF_GGX` term for Standard and Physical.
static inline float3 pbr_specular_one(Surface s, float3 l, float3 light_color, float attenuation) {
    if (attenuation <= 0.0) {
        return float3(0.0);
    }
    return brdf_ggx(s, l) * max(dot(s.n, l), 0.0) * light_color * attenuation;
}

/// `ao` attenuates the *indirect* terms only — ambient and hemisphere — and not
/// the direct lights.
///
/// Ambient occlusion answers "how much of the surrounding room can this point
/// see", which is a question about light arriving from everywhere. A direct
/// light is somewhere specific, and whether it reaches the point is what a
/// shadow map is for. Multiplying the whole shade by `ao` darkens the key light
/// in a crevice that the key light is shining straight into.
// three.js r165's `RectAreaLight` (`RE_Direct_RectArea_Physical`, `LTC_*`),
// as `rect_area_light` in `renderer/shader.rs`.
static inline float2 ltc_uv(float3 n, float3 v, float roughness) {
    const float lut_size = 64.0;
    float ndv = saturate(dot(n, v));
    return float2(roughness, sqrt(1.0 - ndv)) * ((lut_size - 1.0) / lut_size) + 0.5 / lut_size;
}

static inline float ltc_clipped_sphere_form_factor(float3 f) {
    float l = length(f);
    return max((l * l + f.z) / (l + 1.0), 0.0);
}

static inline float3 ltc_edge_vector_form_factor(float3 v1, float3 v2) {
    float x = dot(v1, v2);
    float y = abs(x);
    float a = 0.8543985 + (0.4965155 + 0.0145206 * y) * y;
    float b = 3.4175940 + (4.1616724 + y) * y;
    float v = a / b;
    float ts = x > 0.0 ? v : 0.5 * rsqrt(max(1.0 - x * x, 1e-7)) - v;
    return cross(v1, v2) * ts;
}

static inline float ltc_evaluate(float3 n, float3 v, float3 p, float3x3 m_inv,
                                 float3 r0, float3 r1, float3 r2, float3 r3) {
    float3 light_normal = cross(r1 - r0, r3 - r0);
    if (dot(light_normal, p - r0) < 0.0) {
        return 0.0;
    }
    float3 t1 = normalize(v - n * dot(v, n));
    float3 t2 = -cross(n, t1);
    float3x3 m = m_inv * transpose(float3x3(t1, t2, n));
    float3 c0 = normalize(m * (r0 - p));
    float3 c1 = normalize(m * (r1 - p));
    float3 c2 = normalize(m * (r2 - p));
    float3 c3 = normalize(m * (r3 - p));
    float3 f = ltc_edge_vector_form_factor(c0, c1) + ltc_edge_vector_form_factor(c1, c2)
             + ltc_edge_vector_form_factor(c2, c3) + ltc_edge_vector_form_factor(c3, c0);
    return ltc_clipped_sphere_form_factor(f);
}

/// `[diffuse, specular]`.
static inline float2x3 rect_area_light(RectLight rl, float3 p, Surface s,
                                       texture2d_array<float> ltc, sampler ltc_samp) {
    float3 c = rl.position.xyz;
    float3 hw = rl.right.xyz;
    float3 hh = rl.up.xyz;
    float3 r0 = c + hw - hh;
    float3 r1 = c - hw - hh;
    float3 r2 = c - hw + hh;
    float3 r3 = c + hw + hh;
    float2 uv = ltc_uv(s.n, s.v, s.roughness);
    float4 t1 = ltc.sample(ltc_samp, uv, 0u, level(0.0));
    float4 t2 = ltc.sample(ltc_samp, uv, 1u, level(0.0));
    float3x3 m_inv = float3x3(float3(t1.x, 0.0, t1.y), float3(0.0, 1.0, 0.0), float3(t1.z, 0.0, t1.w));
    float3 fresnel = s.f0 * t2.x + (1.0 - s.f0) * t2.y;
    float3 diffuse = s.albedo * (1.0 - s.metalness);
    return float2x3(rl.color.rgb * diffuse * ltc_evaluate(s.n, s.v, p, float3x3(1.0), r0, r1, r2, r3),
                    rl.color.rgb * fresnel * ltc_evaluate(s.n, s.v, p, m_inv, r0, r1, r2, r3));
}

/// What `shade` gathers from the direct lights: the shaded colour, and for
/// Physical its sheen and clearcoat lobes, which three.js composes over the
/// rest afterwards (`sheenSpecularDirect`, `clearcoatSpecularDirect`).
struct Shaded {
    /// The diffuse light (all of it, for models that do not split), and for
    /// Standard and Physical the specular apart.
    float3 lit;
    float3 spec;
    float3 sheen;
    float3 cc;
};

/// One light's sheen and clearcoat (`RE_Direct_Physical`); `c` is its colour
/// times attenuation.
static inline void physical_direct(Surface s, float3 l, float3 c, thread Shaded &o) {
    if (any(s.sheen_color > 0.0)) {
        o.sheen += c * saturate(dot(s.n, l)) * brdf_sheen(s, l);
    }
    if (s.clearcoat > 0.0) {
        o.cc += c * clearcoat_direct(s, l);
    }
}

static inline Shaded shade(Surface s,
                           float3 world_pos,
                           constant Frame &frame,
                           float ao,
                           depth2d_array<float> shadow_map,
                           depth2d_array<float> spot_map,
                           depth2d_array<float> point_map,
                           sampler shadow_samp,
                           bool receives_shadow,
                           texture2d_array<float> ltc,
                           sampler ltc_samp,
                           texture2d<float> ramp,
                           sampler ramp_samp) {
    // Ambient and hemisphere light are irradiance, through `BRDF_Lambert` like
    // the direct lights (`RE_IndirectDiffuse_*`): ÷π, and the physical model's
    // diffuse colour is the albedo less the metal.
    bool pbr = s.kind == KIND_STANDARD || s.kind == KIND_PHYSICAL;
    float3 diffuse_color = pbr ? s.albedo * (1.0 - s.metalness) : s.albedo;
    float3 irr = frame.ambient.rgb + s.light_map;
    for (uint i = 0u; i < frame.counts.w && i < 2u; ++i) {
        HemiLight h = frame.hemis[i];
        float t = dot(s.n, normalize(h.up.xyz)) * 0.5 + 0.5;
        irr += mix(h.ground.rgb, h.sky.rgb, t);
    }
    // Every record's shadow once (`compute_shadows` in `renderer/shader.rs`:
    // one inlined copy of the filters rather than one per light loop).
    ShadowMaps maps = { shadow_map, spot_map, point_map, shadow_samp };
    float shadow_cache[16];
    for (uint r = 0u; r < 16u; ++r) {
        shadow_cache[r] = 1.0;
    }
    if (receives_shadow) {
        uint count = min(frame.shadow_info.x, 16u);
        for (uint r = 0u; r < count; ++r) {
            shadow_cache[r] = shadow_at(frame, int(r), maps, world_pos, s.n);
        }
    }
    Shaded o;
    o.lit = diffuse_color * irr * (1.0 / M_PI_F) * ao;
    o.spec = float3(0.0);
    o.sheen = float3(0.0);
    o.cc = float3(0.0);
    // Every casting light its own shadow (a layer of its kind's map array).
    for (uint i = 0u; i < frame.counts.x && i < 4u; ++i) {
        DirLight d = frame.dir[i];
        float att = receives_shadow
            ? cached_shadow(shadow_cache, d.color.w)
            : 1.0;
        float3 l = normalize(-d.direction.xyz);
        o.lit += shade_one(s, l, d.color.rgb, att, ramp, ramp_samp);
        if (pbr) { o.spec += pbr_specular_one(s, l, d.color.rgb, att); }
        physical_direct(s, l, d.color.rgb * att, o);
    }
    for (uint i = 0u; i < frame.counts.y && i < 8u; ++i) {
        PointLight p = frame.pts[i];
        float3 to_light = p.position.xyz - world_pos;
        float dist = length(to_light);
        float att = punctual_attenuation(dist, p.params.x, p.params.y);
        att *= cached_shadow(shadow_cache, p.color.w);
        float3 l = to_light / max(dist, 1e-4);
        o.lit += shade_one(s, l, p.color.rgb, att, ramp, ramp_samp);
        if (pbr) { o.spec += pbr_specular_one(s, l, p.color.rgb, att); }
        physical_direct(s, l, p.color.rgb * att, o);
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
        att *= cached_shadow(shadow_cache, sp.color.w);
        o.lit += shade_one(s, to_light, sp.color.rgb, att, ramp, ramp_samp);
        if (pbr) { o.spec += pbr_specular_one(s, to_light, sp.color.rgb, att); }
        physical_direct(s, to_light, sp.color.rgb * att, o);
    }
    // `RE_Direct_RectArea_Physical`: Standard and Physical only, and no
    // sheen or clearcoat, as in three.js.
    if (pbr) {
        for (uint i = 0u; i < frame.rect_counts.x && i < 4u; ++i) {
            float2x3 rt = rect_area_light(frame.rects[i], world_pos, s, ltc, ltc_samp);
            o.lit += rt[0];
            o.spec += rt[1];
        }
    }
    return o;
}

/// How much of a fragment `view_dist` away the fog replaces, 0 to 1.
static inline float fog_amount(float view_dist, constant Frame &frame) {
    uint mode = uint(frame.fog_color.w + 0.5);
    if (mode == 0u) {
        return 0.0;
    }
    float f;
    if (mode == 2u) {
        float d = frame.fog_params.z * view_dist;
        f = 1.0 - exp(-d * d);
    } else {
        f = (view_dist - frame.fog_params.x) /
            max(frame.fog_params.y - frame.fog_params.x, 1e-4);
    }
    return clamp(f, 0.0, 1.0);
}

static inline float3 apply_fog(float3 color, float view_dist, constant Frame &frame) {
    return mix(color, frame.fog_color.rgb, fog_amount(view_dist, frame));
}

// -------------------------------------------------------------- tone mapping

// three.js r165 `tonemapping_pars_fragment`, the same curves as the wgpu
// backend's `apply_tone_mapping`: `frame.view_info.z` is the mode (0 none,
// 1 linear, 2 ACES filmic, 3 neutral, 4 AgX) and `.w` the exposure's bits. A
// material with `toneMapped: false` (`draw.object.w == 0`) is left alone, as
// three.js compiles it without TONE_MAPPING.
static inline float3 tone_map(float3 c, constant Frame &frame, constant Draw &draw) {
    uint mode = frame.view_info.z;
    if (mode == 0u || draw.object.w == 0u) {
        return c;
    }
    float exposure = as_type<float>(frame.view_info.w);
    c *= exposure;
    if (mode == 1u) {
        return saturate(c);
    }
    if (mode == 2u) {
        const float3x3 aces_in = float3x3(float3(0.59719, 0.07600, 0.02840),
                                          float3(0.35458, 0.90834, 0.13383),
                                          float3(0.04823, 0.01566, 0.83777));
        const float3x3 aces_out = float3x3(float3(1.60475, -0.10208, -0.00327),
                                           float3(-0.53108, 1.10813, -0.07276),
                                           float3(-0.07367, -0.00605, 1.07602));
        float3 v = aces_in * (c / 0.6);
        float3 a = v * (v + 0.0245786) - 0.000090537;
        float3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
        return saturate(aces_out * (a / b));
    }
    if (mode == 3u) {
        const float start = 0.8 - 0.04;
        const float desat = 0.15;
        float x = min(c.r, min(c.g, c.b));
        float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
        c -= offset;
        float peak = max(c.r, max(c.g, c.b));
        if (peak < start) {
            return c;
        }
        float d = 1.0 - start;
        float new_peak = 1.0 - d * d / (peak + d - start);
        c *= new_peak / peak;
        float g = 1.0 - 1.0 / (desat * (peak - new_peak) + 1.0);
        return mix(c, float3(new_peak), g);
    }
    const float3x3 to2020 = float3x3(float3(0.6274, 0.0691, 0.0164),
                                     float3(0.3293, 0.9195, 0.0880),
                                     float3(0.0433, 0.0113, 0.8956));
    const float3x3 to709 = float3x3(float3(1.6605, -0.1246, -0.0182),
                                    float3(-0.5876, 1.1329, -0.1006),
                                    float3(-0.0728, -0.0083, 1.1187));
    const float3x3 inset = float3x3(float3(0.856627153315983, 0.137318972929847, 0.11189821299995),
                                    float3(0.0951212405381588, 0.761241990602591, 0.0767994186031903),
                                    float3(0.0482516061458583, 0.101439036467562, 0.811302368396859));
    const float3x3 outset = float3x3(float3(1.1271005818144368, -0.1413297634984383, -0.14132976349843826),
                                     float3(-0.11060664309660323, 1.157823702216272, -0.11060664309660294),
                                     float3(-0.016493938717834573, -0.016493938717834257, 1.2519364065950405));
    const float min_ev = -12.47393;
    const float max_ev = 4.026069;
    float3 v = inset * (to2020 * c);
    v = saturate((log2(max(v, float3(1e-10))) - min_ev) / (max_ev - min_ev));
    float3 v2 = v * v;
    float3 v4 = v2 * v2;
    v = 15.5 * v4 * v2 - 40.14 * v4 * v + 31.96 * v4 - 6.868 * v2 * v + 0.4298 * v2
        + 0.1191 * v - 0.00232;
    v = pow(max(float3(0.0), outset * v), float3(2.2));
    return saturate(to709 * v);
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

// three.js `cube_uv_reflection_fragment`: a PMREM's CubeUV atlas, read by
// direction and roughness — the same lookup as the wgpu path's
// `texture_cube_uv`. `params`: texel width and height, the top mip.
constant float CUBEUV_MIN_MIP_LEVEL = 4.0;
constant float CUBEUV_MIN_TILE_SIZE = 16.0;

static inline float cube_uv_get_face(float3 d) {
    float3 a = abs(d);
    if (a.x > a.z) {
        return a.x > a.y ? (d.x > 0.0 ? 0.0 : 3.0) : (d.y > 0.0 ? 1.0 : 4.0);
    }
    return a.z > a.y ? (d.z > 0.0 ? 2.0 : 5.0) : (d.y > 0.0 ? 1.0 : 4.0);
}

static inline float2 cube_uv_get_uv(float3 d, float face) {
    float2 uv;
    if (face == 0.0) {
        uv = float2(d.z, d.y) / abs(d.x);
    } else if (face == 1.0) {
        uv = float2(-d.x, -d.z) / abs(d.y);
    } else if (face == 2.0) {
        uv = float2(-d.x, d.y) / abs(d.z);
    } else if (face == 3.0) {
        uv = float2(-d.z, d.y) / abs(d.x);
    } else if (face == 4.0) {
        uv = float2(-d.x, d.z) / abs(d.y);
    } else {
        uv = float2(d.x, d.y) / abs(d.z);
    }
    return 0.5 * (uv + 1.0);
}

static inline float3 bilinear_cube_uv(texture2d<float> atlas, sampler samp, float3 d,
                                      float mip_int, float4 params) {
    float face = cube_uv_get_face(d);
    float filter_int = max(CUBEUV_MIN_MIP_LEVEL - mip_int, 0.0);
    float mip_i = max(mip_int, CUBEUV_MIN_MIP_LEVEL);
    float face_size = exp2(mip_i);
    float2 uv = cube_uv_get_uv(d, face) * (face_size - 2.0) + 1.0;
    if (face > 2.0) {
        uv.y += face_size;
        face -= 3.0;
    }
    uv.x += face * face_size;
    uv.x += filter_int * 3.0 * CUBEUV_MIN_TILE_SIZE;
    uv.y += 4.0 * (exp2(params.z) - face_size);
    uv *= params.xy;
    // three.js's uv math is bottom-left; the atlas's rows are top first.
    uv.y = 1.0 - uv.y;
    return atlas.sample(samp, uv, level(0.0)).rgb;
}

static inline float roughness_to_mip(float r) {
    if (r >= 0.8) { return (1.0 - r) * (-1.0 - (-2.0)) / (1.0 - 0.8) + (-2.0); }
    if (r >= 0.4) { return (0.8 - r) * (2.0 - (-1.0)) / (0.8 - 0.4) + (-1.0); }
    if (r >= 0.305) { return (0.4 - r) * (3.0 - 2.0) / (0.4 - 0.305) + 2.0; }
    if (r >= 0.21) { return (0.305 - r) * (4.0 - 3.0) / (0.305 - 0.21) + 3.0; }
    return -2.0 * log2(1.16 * max(r, 0.001));
}

static inline float3 texture_cube_uv(texture2d<float> atlas, sampler samp, float3 d,
                                     float roughness, float4 params) {
    float mip = clamp(roughness_to_mip(roughness), -2.0, params.z);
    float f = fract(mip);
    float i = floor(mip);
    float3 c0 = bilinear_cube_uv(atlas, samp, d, i, params);
    if (f <= 0.0) {
        return c0;
    }
    return mix(c0, bilinear_cube_uv(atlas, samp, d, i + 1.0, params), f);
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
// `draw.env_dir` holds the environment's lookup rows here (three.js's
// `envMapRotation`, and the PMREM mirror for a raw cube); `map_params.z` its
// intensity (`envMapIntensity` / `scene.environmentIntensity`).
// The environment is read through its PMREM atlas where it has one
// (`draw.env_params`: texel size, top mip, 1), as three.js reads every
// Standard/Physical environment; through the cube's own mips otherwise.
// three.js `transmission_pars_fragment`, as `ibl_volume_refraction` in
// `renderer/shader.rs`: N8's mipped bicubic filter over the transmission
// capture, the refracted ray through `thickness`, Beer–Lambert attenuation.
static inline float bicubic_w0(float a) { return (1.0 / 6.0) * (a * (a * (-a + 3.0) - 3.0) + 1.0); }
static inline float bicubic_w1(float a) { return (1.0 / 6.0) * (a * a * (3.0 * a - 6.0) + 4.0); }
static inline float bicubic_w2(float a) { return (1.0 / 6.0) * (a * (a * (-3.0 * a + 3.0) + 3.0) + 1.0); }
static inline float bicubic_w3(float a) { return (1.0 / 6.0) * (a * a * a); }
static inline float bicubic_g0(float a) { return bicubic_w0(a) + bicubic_w1(a); }
static inline float bicubic_g1(float a) { return bicubic_w2(a) + bicubic_w3(a); }
static inline float bicubic_h0(float a) { return -1.0 + bicubic_w1(a) / (bicubic_w0(a) + bicubic_w1(a)); }
static inline float bicubic_h1(float a) { return 1.0 + bicubic_w3(a) / (bicubic_w2(a) + bicubic_w3(a)); }

static inline float4 transmission_bicubic(texture2d_array<float> tex, sampler smp, float2 uv_in, uint slice, float4 texel, float lod) {
    float2 uv = uv_in * texel.zw + 0.5;
    float2 iuv = floor(uv);
    float2 fuv = fract(uv);
    float g0x = bicubic_g0(fuv.x), g1x = bicubic_g1(fuv.x);
    float h0x = bicubic_h0(fuv.x), h1x = bicubic_h1(fuv.x);
    float h0y = bicubic_h0(fuv.y), h1y = bicubic_h1(fuv.y);
    float2 p0 = (float2(iuv.x + h0x, iuv.y + h0y) - 0.5) * texel.xy;
    float2 p1 = (float2(iuv.x + h1x, iuv.y + h0y) - 0.5) * texel.xy;
    float2 p2 = (float2(iuv.x + h0x, iuv.y + h1y) - 0.5) * texel.xy;
    float2 p3 = (float2(iuv.x + h1x, iuv.y + h1y) - 0.5) * texel.xy;
    return bicubic_g0(fuv.y) * (g0x * tex.sample(smp, p0, slice, level(lod)) + g1x * tex.sample(smp, p1, slice, level(lod)))
         + bicubic_g1(fuv.y) * (g0x * tex.sample(smp, p2, slice, level(lod)) + g1x * tex.sample(smp, p3, slice, level(lod)));
}

// The capture holds a slice per view, each the scene from that view: a layered
// pass reads slice `view_index`; a per-view pass, whose view sits in slot 0,
// the slice `rect_counts.w` names.
static inline float4 transmission_sample(float3 p, float roughness, float ior, constant Frame &frame, uint view_index,
                                         texture2d_array<float> tex, sampler smp) {
    uint slice = min(view_index + frame.rect_counts.w, tex.get_array_size() - 1u);
    float4 ndc = frame.view_proj[view_index] * float4(p, 1.0);
    float2 xy = ndc.xy / ndc.w;
    float2 uv = float2(xy.x * 0.5 + 0.5, 0.5 - xy.y * 0.5);
    float lod = log2(float(tex.get_width(0))) * roughness * clamp(ior * 2.0 - 2.0, 0.0, 1.0);
    int last = max(int(frame.rect_counts.y) - 1, 0);
    int fl = min(int(floor(lod)), last);
    int cl = min(int(ceil(lod)), last);
    float2 fs = float2(tex.get_width(fl), tex.get_height(fl));
    float2 cs = float2(tex.get_width(cl), tex.get_height(cl));
    float4 f = transmission_bicubic(tex, smp, uv, slice, float4(1.0 / fs, fs), float(fl));
    float4 c = transmission_bicubic(tex, smp, uv, slice, float4(1.0 / cs, cs), float(cl));
    return mix(f, c, fract(lod));
}

static inline float3 volume_transmission_ray(float3 n, float3 v, float thickness, float ior, constant Draw &draw) {
    float3 r = refract(-v, normalize(n), 1.0 / ior);
    float3 scale = float3(length(draw.model[0].xyz), length(draw.model[1].xyz), length(draw.model[2].xyz));
    return normalize(r) * thickness * scale;
}

static inline float3 volume_attenuation(float distance, float3 color, float attenuation_distance) {
    if (attenuation_distance <= 0.0) {
        return float3(1.0);
    }
    return exp(log(color) / attenuation_distance * distance);
}

/// three.js `getIBLVolumeRefraction`.
static inline float4 ibl_volume_refraction(Surface s, float3 position, float roughness, float ior,
                                           float thickness, float dispersion, float3 attenuation_color,
                                           float attenuation_distance, constant Frame &frame,
                                           constant Draw &draw, uint view_index,
                                           texture2d_array<float> tex, sampler smp) {
    float3 diffuse_color = s.albedo * (1.0 - s.metalness);
    float4 transmitted = float4(0.0);
    float3 transmittance = float3(0.0);
    if (dispersion > 0.0) {
        float half_spread = (ior - 1.0) * 0.025 * dispersion;
        float3 iors = float3(ior - half_spread, ior, ior + half_spread);
        for (uint i = 0u; i < 3u; ++i) {
            float3 ray = volume_transmission_ray(s.n, s.v, thickness, iors[i], draw);
            float4 t = transmission_sample(position + ray, roughness, iors[i], frame, view_index, tex, smp);
            transmitted[i] = t[i];
            transmitted.a += t.a;
            transmittance[i] = diffuse_color[i] * volume_attenuation(length(ray), attenuation_color, attenuation_distance)[i];
        }
        transmitted.a /= 3.0;
    } else {
        float3 ray = volume_transmission_ray(s.n, s.v, thickness, ior, draw);
        transmitted = transmission_sample(position + ray, roughness, ior, frame, view_index, tex, smp);
        transmittance = diffuse_color * volume_attenuation(length(ray), attenuation_color, attenuation_distance);
    }
    float3 attenuated = transmittance * transmitted.rgb;
    float2 fab = dfg_approx(s.n, s.v, roughness);
    float3 f = s.f0 * fab.x + s.f90 * fab.y;
    float factor = (transmittance.r + transmittance.g + transmittance.b) / 3.0;
    return float4((1.0 - f) * attenuated, 1.0 - (1.0 - transmitted.a) * factor);
}

/// The environment's radiance along `dir` at `roughness`, rotated and scaled
/// as `ibl` reads it — three.js's `getIBLRadiance` lookup.
static inline float3 env_radiance(float3 dir,
                                  float roughness,
                                  texturecube<float> env,
                                  sampler samp,
                                  constant Draw &draw,
                                  texture2d<float> atlas,
                                  sampler atlas_samp) {
    float3 d = float3(dot(draw.env_dir[0].xyz, dir), dot(draw.env_dir[1].xyz, dir),
                      dot(draw.env_dir[2].xyz, dir));
    bool cube_uv = draw.env_params.w > 0.5;
    return (cube_uv ? texture_cube_uv(atlas, atlas_samp, d, roughness, draw.env_params)
                    : sample_env(env, samp, d, roughness)) * draw.map_params.z;
}

/// The environment's contribution, diffuse and specular, occluded — three.js
/// r165 `RE_IndirectSpecular_Physical` with `computeMultiscattering`
/// (`…Iridescence`): the reflectance is the lobe's F0 mixed toward the thin
/// film's Fresnel, at its F90, and anisotropy bends the normal the
/// environment is reflected about (`getIBLAnisotropyRadiance`).
/// `[diffuse, specular]`.
static inline float2x3 ibl(Surface s,
                         texturecube<float> env,
                         sampler samp,
                         float ao,
                         constant Draw &draw,
                         texture2d<float> atlas,
                         sampler atlas_samp) {
    float roughness = s.roughness_env;
    float3 specular_color = mix(s.f0, s.irid_fresnel, s.irid);
    float3 diffuse_color = s.albedo * (1.0 - s.metalness);

    float3 refl_n = s.n;
    if (s.aniso >= 0.0) {
        float3 bent = normalize(cross(cross(s.ab, s.v), s.ab));
        float bend = 1.0 - s.aniso * (1.0 - roughness);
        bend *= bend;
        refl_n = normalize(mix(bent, s.n, bend * bend));
    }
    // `getIBLRadiance`: pulled toward that normal as roughness rises, which is
    // what keeps a rough surface from mirroring a sharp edge of the room.
    float3 refl = reflect(-s.v, refl_n);
    refl = normalize(mix(refl, refl_n, roughness * roughness));
    float3 radiance = env_radiance(refl, roughness, env, samp, draw, atlas, atlas_samp);
    float3 irradiance = env_radiance(s.n, 1.0, env, samp, draw, atlas, atlas_samp) * M_PI_F;

    float2 fab = dfg_approx(s.n, s.v, roughness);
    float3 fss_ess = specular_color * fab.x + float3(s.f90 * fab.y);
    float ess = fab.x + fab.y;
    float ems = 1.0 - ess;
    float3 favg = specular_color + (float3(1.0) - specular_color) * 0.047619;
    float3 fms = fss_ess * favg / (float3(1.0) - ems * favg);
    float3 multi_scatter = fms * ems;
    float3 total = fss_ess + multi_scatter;
    float scatter_max = max(max(total.r, total.g), total.b);
    float3 cosine_weighted = irradiance * (1.0 / M_PI_F);

    float spec_occ = specular_occlusion(saturate(dot(s.n, s.v)), ao, roughness);
    float3 spec = (radiance * fss_ess + multi_scatter * cosine_weighted) * spec_occ;
    // The environment's diffuse term, added exactly once — the ambient lights
    // are accounted for separately in `shade`, and adding this there as well is
    // how a white surface ends up reflecting more light than reaches it.
    float3 diffuse = diffuse_color * (1.0 - scatter_max) * cosine_weighted * ao;
    return float2x3(diffuse, spec);
}

// three.js `IBLSheenBRDF`: the Charlie sheen lobe integrated over the
// hemisphere, as a curve fit.
static inline float ibl_sheen_brdf(float3 n, float3 v, float roughness) {
    float ndv = saturate(dot(n, v));
    float r2 = roughness * roughness;
    float a = roughness < 0.25 ? -339.2 * r2 + 161.4 * roughness - 25.9 : -8.48 * r2 + 14.3 * roughness - 9.95;
    float b = roughness < 0.25 ? 44.0 * r2 - 23.7 * roughness + 3.26 : 1.97 * r2 - 3.27 * roughness + 0.72;
    float dg = exp(a * ndv + b) + (roughness < 0.25 ? 0.0 : 0.1 * (roughness - 0.25));
    return saturate(dg * (1.0 / M_PI_F));
}

// ------------------------------------------------------------------- fragment

static inline float4 sample_base(constant Draw &draw,
                                 texture2d<float> base_map,
                                 sampler samp,
                                 float2 uv, float4 uv12, float2 uv3) {
    float4 base = draw.base_color;
    if (draw.flags.y == 1u) {
        base *= base_map.sample(samp, map_uv_sets(draw, MAP_BASE, uv, uv12, uv3));
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
// three.js `envmap_fragment` (Basic, Lambert, Phong): the reflected — or, for a
// refraction-mapped cube, refracted — view ray looks the `envMap` up, and the
// result meets `outgoing` per `combine`, weighted by `specularStrength ×
// reflectivity`. The ray is the camera's forward axis for an orthographic view
// (whose view-projection has a constant w row), as three.js's `isOrthographic`.
static inline float3 apply_env_map(float3 outgoing, float3 n, float3 world_pos,
                                   float specular_strength, constant Frame &frame,
                                   constant Draw &draw, uint view_index,
                                   texturecube<float> env, sampler samp) {
    if (draw.map_scalars.z < 0.5) {
        return outgoing;
    }
    float4x4 vp = frame.view_proj[view_index];
    float4x4 v = frame.view[view_index];
    bool ortho = abs(vp[0][3]) + abs(vp[1][3]) + abs(vp[2][3]) < 1e-6;
    float3 to_frag = ortho ? normalize(-float3(v[0][2], v[1][2], v[2][2]))
                           : normalize(world_pos - frame.camera_pos[view_index].xyz);
    float3 r = draw.env_params.w > 0.5 ? refract(to_frag, n, draw.env_params.z)
                                       : reflect(to_frag, n);
    float3 dir = float3(dot(draw.env_dir[0].xyz, r), dot(draw.env_dir[1].xyz, r),
                        dot(draw.env_dir[2].xyz, r));
    float3 e = env.sample(samp, dir).rgb;
    float k = specular_strength * draw.env_params.y;
    uint combine = uint(draw.env_params.x + 0.5);
    if (combine == 1u) { return mix(outgoing, e, k); }
    if (combine == 2u) { return outgoing + e * k; }
    return mix(outgoing, outgoing * e, k);
}

// three.js `bumpmap_pars_fragment`: `dHdxy_fwd` and `perturbNormalArb`
// (Mikkelsen), in world space — the construction is crosses and dots, so the
// view rotation changes nothing, and nor does the framebuffer's y direction.
static inline float3 bump_normal(float3 n, float3 pos, float2 uv, float4 uv12, float2 uv3,
                                 float face, constant Draw &draw, texture2d<float> bump_map,
                                 sampler samp) {
    float scale = draw.map_scalars.y;
    float2 st = map_uv_sets(draw, MAP_BUMP, uv, uv12, uv3);
    float2 dstdx = dfdx(st);
    float2 dstdy = dfdy(st);
    float hll = scale * bump_map.sample(samp, st).x;
    float dbx = scale * bump_map.sample(samp, st + dstdx).x - hll;
    float dby = scale * bump_map.sample(samp, st + dstdy).x - hll;
    float3 sigma_x = normalize(dfdx(pos));
    float3 sigma_y = normalize(dfdy(pos));
    float3 r1 = cross(sigma_y, n);
    float3 r2 = cross(n, sigma_x);
    float det = dot(sigma_x, r1) * face;
    float3 grad = sign(det) * (dbx * r1 + dby * r2);
    return normalize(abs(det) * n - grad);
}

static inline float3 perturb_normal(float3 n,
                                    float3 world_pos,
                                    float2 uv,
                                    float3 map_sample,
                                    float scale,
                                    float4 tangent) {
    // The `tangent` attribute's frame when the geometry has one, as the wgpu
    // backend's `surface_frame`; screen-space derivatives otherwise.
    if (dot(tangent.xyz, tangent.xyz) > 1e-12) {
        TangentPair none;
        none.t = float3(0.0);
        none.b = float3(0.0);
        TangentPair f = attribute_frame(tangent, n, none);
        if (dot(f.t, f.t) > 0.0) {
            float3 m = map_sample * 2.0 - 1.0;
            m.xy *= scale;
            float3 p = f.t * m.x + f.b * m.y + n * m.z;
            return dot(p, p) > 1e-12 ? normalize(p) : n;
        }
    }
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

static inline float4 mesh_fragment(FIn in,
                                   bool front_facing,
                                   constant Frame   &frame,
                                   constant Draw    &draw,
                                   texture2d<float>  base_map,
                                   texture2d<float>  emissive_map,
                                   texture2d<float>  normal_map,
                                   texture2d<float>  mr_map,
                                   texture2d<float>  ao_map,
                                   texturecube<float> env_map,
                                   depth2d_array<float> shadow_map,
                                   depth2d_array<float> spot_map,
                                   depth2d_array<float> point_map,
                                   sampler           samp,
                                   sampler           env_samp,
                                   sampler           shadow_samp,
                                   sampler           emissive_samp,
                                   sampler           normal_samp,
                                   sampler           mr_samp,
                                   sampler           ao_samp,
                                   texture2d<float>  alpha_map,
                                   sampler           alpha_samp,
                                   texture2d<float>  light_map,
                                   sampler           light_samp,
                                   texture2d<float>  bump_map,
                                   sampler           bump_samp,
                                   texture2d<float>  specular_map,
                                   sampler           specular_samp,
                                   texturecube<float> mat_env,
                                   sampler           mat_env_samp,
                                   texture2d<float>  matcap_tex,
                                   sampler           matcap_samp,
                                   texture2d<float>  env_atlas,
                                   sampler           env_atlas_samp,
                                   texture2d_array<float> ltc,
                                   texture2d_array<float> transmission_tex,
                                   sampler           transmission_samp,
                                   thread const PhysTexels &phys,
                                   thread float3 &glass_t) {
    // Blended glass (`TransparencyMode::Glass`, `Oit`): what passes of the
    // background, per channel, for the caller's blend. Negative otherwise.
    glass_t = float3(-1.0);
    uint kind = draw.flags.x;
    float3 n = normalize(in.normal);
    if (!front_facing) {
        n = -n;   // double-sided lighting, as three.js does for BackSide/DoubleSide
    }

    if (kind == KIND_NORMAL) {
        // three.js MeshNormalMaterial: the *view-space* normal packed to rgb
        // and written as data, with no transfer curve — so on a target that
        // sRGB-encodes on write, decode first and the stored byte is the value.
        float3 view_n = normalize((frame.view[in.view_index] * float4(n, 0.0)).xyz);
        float3 packed = saturate(view_n * 0.5 + 0.5);
        if (frame.view_info.y == 1u) {
            packed = select(pow((packed + 0.055) / 1.055, float3(2.4)), packed / 12.92,
                            packed <= float3(0.04045));
        }
        return float4(packed, draw.base_color.a);
    }
    if (kind == KIND_DEPTH || kind == KIND_DISTANCE) {
        float near = draw.misc.z;
        float far = max(draw.misc.w, near + 1e-4);
        float d = clamp((in.view_dist - near) / (far - near), 0.0, 1.0);
        return float4(float3(1.0 - d), draw.base_color.a);
    }

    float4 base = sample_base(draw, base_map, samp, in.uv, in.uv12, in.uv3);
    base *= in.color;
    // three.js `alphamap_fragment`: the green channel scales the alpha.
    if (draw.extra.x == 1u) {
        base.a *= alpha_map.sample(alpha_samp, map_uv_sets(draw, MAP_ALPHA, in.uv, in.uv12, in.uv3)).g;
    }
    if (base.a < draw.pbr.z) {
        discard_fragment();
    }

    if (kind == KIND_MATCAP) {
        // three.js `meshmatcap.glsl`: the view-space normal in a frame built
        // around the view direction, × 0.495; no matcap: mix(0.2, 0.8, uv.y).
        float3 view_n = normalize((frame.view[in.view_index] * float4(n, 0.0)).xyz);
        float3 view_dir = normalize(-(frame.view[in.view_index] * float4(in.world_pos, 1.0)).xyz);
        float3 mx = normalize(float3(view_dir.z, 0.0, -view_dir.x));
        float3 my = cross(view_dir, mx);
        float2 muv = float2(dot(mx, view_n), dot(my, view_n)) * 0.495 + 0.5;
        float3 matcap = draw.map_scalars.w > 0.5 ? matcap_tex.sample(matcap_samp, muv).rgb
                                                 : float3(mix(0.2, 0.8, muv.y));
        // `base` is the colour, its `map` and the vertex colour, as three.js's
        // `diffuseColor` is by now.
        return float4(apply_fog(tone_map(base.rgb * matcap, frame, draw), in.view_dist, frame),
                      base.a);
    }

    if (draw.object.z == 1u) {
        // three.js ShadowMaterial: `vec4(color, opacity · (1 − getShadowMask()))`,
        // the product of every casting light's shadow on a receiving mesh.
        float mask = 1.0;
        if (draw.object.x == 1u) {
            ShadowMaps maps = { shadow_map, spot_map, point_map, shadow_samp };
            for (uint r = 0u; r < frame.shadow_info.x; ++r) {
                mask *= shadow_at(frame, int(r), maps, in.world_pos, n);
            }
        }
        return float4(apply_fog(tone_map(draw.base_color.rgb, frame, draw), in.view_dist, frame),
                      draw.base_color.a * (1.0 - mask));
    }

    // Emissive is its own channel and applies to lit and unlit alike: a phone
    // screen emits its picture rather than reflecting it, and that is the case
    // this slot exists for.
    float3 emissive = draw.emissive.rgb;
    if (draw.maps.x == 1u) {
        emissive *= emissive_map.sample(emissive_samp, map_uv_sets(draw, MAP_EMISSIVE, in.uv, in.uv12, in.uv3)).rgb;
    }

    bool lit_kind = kind == KIND_LAMBERT || kind == KIND_PHONG || kind == KIND_STANDARD
                 || kind == KIND_PHYSICAL || kind == KIND_TOON;
    float3 color;
    if (lit_kind) {
        // three.js `nonPerturbedNormal`: the clearcoat's normal and the one the
        // anisotropy frame is built about.
        const float3 n_flat = n;
        if (draw.maps.y == 1u) {
            float2 nuv = map_uv_sets(draw, MAP_NORMAL, in.uv, in.uv12, in.uv3);
            n = perturb_normal(n, in.world_pos, nuv,
                               normal_map.sample(normal_samp, nuv).rgb,
                               draw.map_params.x, in.tangent);
        } else if (draw.extra.z == 1u) {
            n = bump_normal(n, in.world_pos, in.uv, in.uv12, in.uv3, front_facing ? 1.0 : -1.0,
                            draw, bump_map, bump_samp);
        }
        float roughness = draw.pbr.x;
        float metalness = draw.pbr.y;
        if (draw.maps.z == 1u) {
            // glTF packing: roughness in green, metalness in blue, both as
            // multipliers on the material's scalars.
            float3 mr = mr_map.sample(mr_samp, map_uv_sets(draw, MAP_MR, in.uv, in.uv12, in.uv3)).rgb;
            roughness *= mr.g;
            metalness *= mr.b;
        }
        float ao = 1.0;
        if (draw.maps.w == 1u) {
            ao = mix(1.0, ao_map.sample(ao_samp, map_uv_sets(draw, MAP_AO, in.uv, in.uv12, in.uv3)).r, draw.map_params.y);
        }
        bool pbr = kind == KIND_STANDARD || kind == KIND_PHYSICAL;
        bool physical = kind == KIND_PHYSICAL;

        Surface s;
        s.n = n;
        s.v = normalize(frame.camera_pos[in.view_index].xyz - in.world_pos);
        s.albedo = base.rgb;
        s.specular = draw.specular.rgb;
        s.shininess = draw.specular.w;
        s.metalness = metalness;
        s.kind = kind;
        s.toon_gradient = draw.pbr.w;
        s.dndx = dfdx(n);
        s.dndy = dfdy(n);
        s.light_map = draw.extra.y == 1u
            ? light_map.sample(light_samp, map_uv_sets(draw, MAP_LIGHT, in.uv, in.uv12, in.uv3)).rgb * draw.map_scalars.x
            : float3(0.0);
        s.specular_strength = draw.extra.w == 1u
            ? specular_map.sample(specular_samp, map_uv_sets(draw, MAP_SPECULAR, in.uv, in.uv12, in.uv3)).r
            : 1.0;

        // three.js `lights_physical_fragment`: `max(roughness, 0.0525)` plus
        // `geometryRoughness`, the largest screen-space change of any
        // component of the view-space unperturbed normal; or, with specular
        // antialiasing on, the shading normal's variance folded into alpha.
        float3 n_view_flat = (frame.view[in.view_index] * float4(n_flat, 0.0)).xyz;
        float3 n_dxy = max(abs(dfdx(n_view_flat)), abs(dfdy(n_view_flat)));
        float geometry_roughness = max(max(n_dxy.x, n_dxy.y), n_dxy.z);
        bool specular_aa = draw.phys_flags.y == 1u;
        s.roughness_env = max(roughness, 0.0525);
        s.roughness = specular_aa ? specular_aa_roughness(n, s.roughness_env)
                                  : min(s.roughness_env + geometry_roughness, 1.0);
        s.alpha = max(s.roughness * s.roughness, 0.0016);

        // The specular lobe: Standard's F0 0.04 and F90 1; Physical's from
        // `ior`, `specularIntensity` and `specularColor` and their maps.
        s.f0 = mix(float3(0.04), base.rgb, metalness);
        s.f90 = 1.0;
        s.irid = 0.0;
        s.irid_fresnel = float3(0.0);
        s.aniso = -1.0;
        s.alpha_t = s.alpha;
        s.at = float3(0.0);
        s.ab = float3(0.0);
        s.sheen_color = float3(0.0);
        s.sheen_roughness = 1.0;
        s.clearcoat = 0.0;
        s.cc_roughness = 0.0;
        s.cc_n = n_flat;
        if (physical) {
            float spec_intensity = draw.physical[0].y * phys_texel(phys, MAP_SPECULAR_INTENSITY).a;
            float3 spec_tint = draw.physical[1].rgb * phys_texel(phys, MAP_SPECULAR_COLOR).rgb;
            float ior = draw.physical[0].x;
            float r0 = (ior - 1.0) / (ior + 1.0);
            s.f0 = mix(min(float3(r0 * r0) * spec_tint, float3(1.0)) * spec_intensity, base.rgb, metalness);
            s.f90 = mix(spec_intensity, 1.0, metalness);

            // Iridescence: the thin film's Fresnel at the view angle over the
            // base's reflectance; a film of 0 nm is none.
            float4 irid_p = draw.physical[3];
            if (irid_p.x > 0.0) {
                float irid = irid_p.x * phys_texel(phys, MAP_IRIDESCENCE).r;
                float film = has_phys(draw, MAP_IRIDESCENCE_THICKNESS)
                    ? (irid_p.w - irid_p.z) * phys_texel(phys, MAP_IRIDESCENCE_THICKNESS).g + irid_p.z
                    : irid_p.w;
                irid = film == 0.0 ? 0.0 : saturate(irid);
                if (irid > 0.0) {
                    s.irid = irid;
                    s.irid_fresnel = eval_iridescence(1.0, irid_p.y, saturate(dot(n, s.v)), film, s.f0);
                }
            }

            // Anisotropy: `anisotropy · (cos, sin)(anisotropyRotation)`, turned
            // by the map's direction and scaled by its strength, in
            // `getTangentFrame`'s frame, flipped with a back face.
            float4 an_p = draw.physical[4];
            if (an_p.x > 0.0) {
                TangentPair f = tbn_frame(in.tangent, in.normal, three_tangent_frame(in.world_pos, n_flat, in.uv));
                float3 t = front_facing ? f.t : -f.t;
                float3 b = front_facing ? f.b : -f.b;
                float2 av = an_p.x * float2(cos(an_p.y), sin(an_p.y));
                if (has_phys(draw, MAP_ANISOTROPY)) {
                    float3 polar = phys_texel(phys, MAP_ANISOTROPY).rgb;
                    av = float2x2(float2(av.x, av.y), float2(-av.y, av.x))
                       * normalize(2.0 * polar.rg - 1.0) * polar.b;
                }
                float a = length(av);
                if (a == 0.0) {
                    av = float2(1.0, 0.0);
                } else {
                    av /= a;
                    a = saturate(a);
                }
                s.aniso = a;
                s.alpha_t = mix(s.alpha, 1.0, a * a);
                s.at = t * av.x + b * av.y;
                s.ab = b * av.x - t * av.y;
            }

            // Sheen: `sheenColor × sheen` and its map; `sheenRoughness`
            // clamped to [0.07, 1] before its map scales it.
            float4 sheen_p = draw.physical[2];
            if (draw.physical[1].w > 0.0) {
                s.sheen_color = sheen_p.rgb * draw.physical[1].w * phys_texel(phys, MAP_SHEEN_COLOR).rgb;
                s.sheen_roughness = clamp(sheen_p.w, 0.07, 1.0) * phys_texel(phys, MAP_SHEEN_ROUGHNESS).a;
            }

            // Clearcoat: its maps, the unperturbed normal or its own normal map
            // (scaled by `clearcoatNormalScale`), and the base's roughness rule.
            if (draw.physical[0].z > 0.0) {
                s.clearcoat = saturate(draw.physical[0].z * phys_texel(phys, MAP_CLEARCOAT).x);
                float ccr = draw.physical[0].w * phys_texel(phys, MAP_CLEARCOAT_ROUGHNESS).y;
                if (has_phys(draw, MAP_CLEARCOAT_NORMAL)) {
                    TangentPair f = tbn_frame(in.tangent, in.normal, three_tangent_frame(in.world_pos, n_flat, in.uv));
                    float3 t = front_facing ? f.t : -f.t;
                    float3 b = front_facing ? f.b : -f.b;
                    float3 m = phys_texel(phys, MAP_CLEARCOAT_NORMAL).xyz * 2.0 - 1.0;
                    m.xy *= draw.physical[4].zw;
                    s.cc_n = normalize(t * m.x + b * m.y + n_flat * m.z);
                }
                s.cc_roughness = specular_aa ? specular_aa_roughness(s.cc_n, max(ccr, 0.0525))
                                             : min(max(ccr, 0.0525) + geometry_roughness, 1.0);
            }
        }

        Shaded shaded = shade(s, in.world_pos, frame, ao, shadow_map, spot_map, point_map,
                              shadow_samp, draw.object.x == 1u, ltc, env_atlas_samp, matcap_tex, matcap_samp);
        color = shaded.lit;
        float3 specular = shaded.spec;
        bool has_env = draw.map_params.w > 0.5 && pbr;
        // threers's reflective transparency (`set_reflective_transparency`, off
        // by default): a transparent surface's alpha raised by its specular
        // reflectance — `opacity + (1 − opacity) · reflectance` — so the
        // reflection, which is in front of the surface, does not fade with the
        // alpha that lets the background through. The reflectance is the
        // split-sum total `ibl` uses with an environment, its Fresnel alone
        // without one.
        if (frame.rect_counts.z == 1u && pbr && base.a < 1.0) {
            float2 fab = dfg_approx(s.n, s.v, s.roughness_env);
            float3 refl;
            if (has_env) {
                float3 sc = mix(s.f0, s.irid_fresnel, s.irid);
                float3 fss_ess = sc * fab.x + float3(s.f90 * fab.y);
                float ems = 1.0 - (fab.x + fab.y);
                float3 favg = sc + (float3(1.0) - sc) * 0.047619;
                refl = fss_ess + fss_ess * favg / (float3(1.0) - ems * favg) * ems;
            } else {
                refl = s.f0 * fab.x + float3(s.f90 * fab.y);
            }
            float coverage = saturate(max(max(refl.r, refl.g), refl.b));
            base.a = saturate(base.a + (1.0 - base.a) * coverage);
        }
        if (has_env) {
            float2x3 indirect = ibl(s, env_map, env_samp, ao, draw, env_atlas, env_atlas_samp);
            color += indirect[0];
            specular += indirect[1];
        }
        // three.js `transmission_fragment`: the scene behind, from the
        // transmission capture along the refracted ray, replaces the diffuse
        // light by `transmission`; `transmissionAlpha` scales the alpha.
        float4 trans_p = draw.physical[5];
        if (physical && trans_p.x > 0.0 && draw.phys_flags.w == 1u) {
            // The blended glass modes: the background comes through the
            // blend, so transmission is a share of it — what Fresnel does not
            // reflect, absorbed over `thickness` — per channel, as
            // `shade_main` in `renderer/shader.rs` passes it.
            float t = trans_p.x * phys_texel(phys, MAP_TRANSMISSION).r;
            float thickness = trans_p.y * phys_texel(phys, MAP_THICKNESS).g;
            float2 fab = dfg_approx(s.n, s.v, s.roughness);
            float3 f = s.f0 * fab.x + s.f90 * fab.y;
            float3 through = (float3(1.0) - f) * s.albedo * (1.0 - s.metalness)
                           * volume_attenuation(thickness, draw.physical[6].rgb, trans_p.z);
            glass_t = float3(1.0) - base.a * (float3(1.0) - t * through);
            base.a *= 1.0 - t * (through.r + through.g + through.b) / 3.0;
            color *= 1.0 - t;
        } else if (physical && trans_p.x > 0.0) {
            float t = trans_p.x * phys_texel(phys, MAP_TRANSMISSION).r;
            float thickness = trans_p.y * phys_texel(phys, MAP_THICKNESS).g;
            float4 transmitted = ibl_volume_refraction(s, in.world_pos, s.roughness, draw.physical[0].x,
                                                       thickness, trans_p.w, draw.physical[6].rgb, trans_p.z,
                                                       frame, draw, in.view_index, transmission_tex,
                                                       transmission_samp);
            base.a *= mix(1.0, transmitted.a, t);
            color = mix(color, transmitted.rgb, t);
        }
        color += specular;
        // three.js `meshphysical` composes `diffuse + specular + emissive`,
        // then the sheen and then the clearcoat over all of it.
        color += emissive;
        if (any(s.sheen_color > 0.0)) {
            float3 sheen = shaded.sheen;
            if (has_env) {
                float3 ibl_irr = env_radiance(s.n, 1.0, env_map, env_samp, draw, env_atlas, env_atlas_samp) * M_PI_F;
                sheen += ibl_irr * s.sheen_color * ibl_sheen_brdf(s.n, s.v, s.sheen_roughness);
            }
            float sheen_max = max(max(s.sheen_color.r, s.sheen_color.g), s.sheen_color.b);
            color = color * (1.0 - 0.157 * sheen_max) + sheen;
        }
        if (s.clearcoat > 0.0) {
            float3 cc = shaded.cc;
            if (has_env) {
                float3 refl = reflect(-s.v, s.cc_n);
                refl = normalize(mix(refl, s.cc_n, s.cc_roughness * s.cc_roughness));
                float2 fab = dfg_approx(s.cc_n, s.v, s.cc_roughness);
                cc += env_radiance(refl, s.cc_roughness, env_map, env_samp, draw, env_atlas, env_atlas_samp)
                    * (float3(0.04) * fab.x + fab.y);
            }
            float3 fcc = f_schlick_f90(float3(0.04), 1.0, saturate(dot(s.cc_n, s.v)));
            color = color * (1.0 - s.clearcoat * fcc) + cc * s.clearcoat;
        }
        if (kind == KIND_LAMBERT || kind == KIND_PHONG) {
            color = apply_env_map(color, n, in.world_pos, s.specular_strength, frame, draw,
                                  in.view_index, mat_env, mat_env_samp);
        }
    } else {
        // Unlit: Basic, Line, Points, Sprite, Sky, Mirror, Shader, Atmosphere.
        // three.js `meshbasic_frag`: with a light map the indirect diffuse is
        // its texel × `lightMapIntensity` ÷ π instead of 1.
        float3 indirect = draw.extra.y == 1u
            ? light_map.sample(light_samp, map_uv_sets(draw, MAP_LIGHT, in.uv, in.uv12, in.uv3)).rgb
                  * draw.map_scalars.x * (1.0 / M_PI_F)
            : float3(1.0);
        color = base.rgb * indirect + emissive;
        // three.js Basic reflects per vertex (`vReflect`); here per fragment
        // from the interpolated normal, the same on any flat face.
        if (kind == KIND_BASIC) {
            float spec = draw.extra.w == 1u
                ? specular_map.sample(specular_samp, map_uv_sets(draw, MAP_SPECULAR, in.uv, in.uv12, in.uv3)).r
                : 1.0;
            color = apply_env_map(color, n, in.world_pos, spec, frame, draw, in.view_index,
                                  mat_env, mat_env_samp);
        }
    }

    if (glass_t.x >= 0.0) {
        // Fog in front of the glass veils what it lets through too.
        glass_t *= 1.0 - fog_amount(in.view_dist, frame);
    }
    return float4(apply_fog(tone_map(color, frame, draw), in.view_dist, frame), base.a);
}

// Material-map samplers are a pool (`map_samps`, slots 0–9) that each map
// indexes (`Draw::map_samplers`), the base map's always at 0: one slot per
// distinct filter-and-wrap, where one per map would pass Metal's sixteen.
#define MESH_FRAGMENT_ARGS                                                     \
    FIn in [[stage_in]],                                                       \
    bool front_facing [[front_facing]],                                        \
    constant Frame   &frame        [[buffer(0)]],                              \
    constant Draw    &draw         [[buffer(1)]],                              \
    texture2d<float>  base_map     [[texture(0)]],                             \
    texture2d<float>  emissive_map [[texture(1)]],                             \
    texture2d<float>  normal_map   [[texture(2)]],                             \
    texture2d<float>  mr_map       [[texture(3)]],                             \
    texture2d<float>  ao_map       [[texture(4)]],                             \
    texturecube<float> env_map     [[texture(5)]],                             \
    depth2d_array<float> shadow_map [[texture(6)]],                            \
    depth2d_array<float> spot_map  [[texture(7)]],                             \
    depth2d_array<float> point_map [[texture(8)]],                             \
    texture2d<float>  alpha_map    [[texture(9)]],                             \
    texture2d<float>  light_map    [[texture(10)]],                            \
    texture2d<float>  bump_map     [[texture(11)]],                            \
    texture2d<float>  specular_map [[texture(12)]],                            \
    texturecube<float> mat_env     [[texture(13)]],                            \
    texture2d<float>  matcap_tex   [[texture(14)]],                            \
    texture2d<float>  env_atlas    [[texture(15)]],                            \
    array<texture2d<float>, 12> phys_maps [[texture(16)]],                     \
    texture2d_array<float> ltc     [[texture(28)]],                            \
    texture2d_array<float> transmission_tex [[texture(29)]],                        \
    sampler           transmission_samp [[sampler(14)]],                       \
    array<sampler, 10> map_samps   [[sampler(0)]],                             \
    sampler           env_samp     [[sampler(10)]],                            \
    sampler           shadow_samp  [[sampler(11)]],                            \
    sampler           mat_env_samp [[sampler(12)]],                            \
    sampler           env_atlas_samp [[sampler(13)]]

// The physical maps' texels: white where absent (a flat normal for the
// coat's), each through its own sampler and transform. Sampled here, where
// the textures are an indexable array, in draw-uniform control flow.
#define SAMPLE_PHYSICAL                                                        \
    PhysTexels phys;                                                           \
    for (uint i = 0u; i < PHYS_MAP_COUNT; ++i) {                               \
        uint m = MAP_PHYS0 + i;                                                \
        phys.v[i] = has_phys(draw, m)                                          \
            ? phys_maps[i].sample(map_samps[map_sampler(draw, m)], map_uv_sets(draw, m, in.uv, in.uv12, in.uv3)) \
            : (m == MAP_CLEARCOAT_NORMAL ? float4(0.5, 0.5, 1.0, 1.0) : float4(1.0)); \
    }

#define MS(i) map_samps[map_sampler(draw, i)]

#define MESH_FRAGMENT_CALL                                                     \
    mesh_fragment(in, front_facing, frame, draw, base_map, emissive_map,       \
                  normal_map, mr_map, ao_map, env_map, shadow_map, spot_map,   \
                  point_map, MS(MAP_BASE), env_samp, shadow_samp,              \
                  MS(MAP_EMISSIVE), MS(MAP_NORMAL), MS(MAP_MR), MS(MAP_AO),    \
                  alpha_map, MS(MAP_ALPHA), light_map, MS(MAP_LIGHT),          \
                  bump_map, MS(MAP_BUMP), specular_map, MS(MAP_SPECULAR),      \
                  mat_env, mat_env_samp, matcap_tex, MS(MAP_MATCAP),           \
                  env_atlas, env_atlas_samp, ltc, transmission_tex,      \
                  transmission_samp, phys, glass_t)

fragment float4 fs_mesh(MESH_FRAGMENT_ARGS) {
    SAMPLE_PHYSICAL
    float3 glass_t;
    float4 src = MESH_FRAGMENT_CALL;
    if (glass_t.x >= 0.0) {
        // Blended glass through the hardware's one alpha (a float target):
        // it passes the mean of the tint, and the light divided by alpha
        // comes back at full strength after the blend.
        float a = max(src.a, 1e-3);
        return float4(src.rgb / a, src.a);
    }
    return src;
}

static inline float3 srgb_encode(float3 c) {
    c = saturate(c);
    return select(1.055 * pow(c, float3(1.0 / 2.4)) - 0.055, c * 12.92, c <= float3(0.0031308));
}

static inline float3 srgb_decode(float3 c) {
    c = saturate(c);
    return select(pow((c + 0.055) / 1.055, float3(2.4)), c / 12.92, c <= float3(0.04045));
}

/// Alpha blending as three.js's default `WebGLRenderer` does it: into the
/// canvas's 8-bit framebuffer, which holds sRGB-*encoded* bytes, so
/// `NormalBlending` (`SRC_ALPHA, ONE_MINUS_SRC_ALPHA`; alpha `ONE,
/// ONE_MINUS_SRC_ALPHA`) mixes encoded values, not light. An sRGB attachment
/// would blend in linear light instead, so the pipelines for blended draws
/// into one read the destination (programmable blending — it arrives
/// decoded), blend the encodings here, and write the result back with
/// fixed-function blending off. Per sample under MSAA, which is what the
/// framebuffer read makes the stage. Float targets blend in linear in
/// three.js too (a render target's output is linear) and keep the hardware
/// blend.
fragment float4 fs_mesh_encoded_blend(MESH_FRAGMENT_ARGS, float4 dst [[color(0)]]) {
    SAMPLE_PHYSICAL
    float3 glass_t;
    float4 src = MESH_FRAGMENT_CALL;
    if (glass_t.x >= 0.0) {
        // Blended glass: what is behind times its transmittance, per
        // channel, then its light — `behind · T + L` on the encoded values,
        // where `T`'s factor is its encoding (a power law to within sRGB's
        // toe). The `Glass` mode's two draws on the wgpu path, in one here.
        float m = (glass_t.r + glass_t.g + glass_t.b) / 3.0;
        float3 e = srgb_encode(src.rgb) + srgb_encode(dst.rgb) * srgb_encode(glass_t);
        return float4(srgb_decode(e), saturate(1.0 - m + dst.a * m));
    }
    float a = saturate(src.a);
    float3 e = srgb_encode(src.rgb) * a + srgb_encode(dst.rgb) * (1.0 - a);
    return float4(srgb_decode(e), a + dst.a * (1.0 - a));
}

// three.js `points_frag`: the map and alpha map at the point's
// `(gl_PointCoord.x, 1 − gl_PointCoord.y)` through each map's transform, the
// vertex colour, `alphaTest`, tone mapping and fog.
static inline float4 point_fragment(FIn in, float2 point_coord, constant Frame &frame,
                                    constant Draw &draw, texture2d<float> base_map,
                                    sampler samp, texture2d<float> alpha_map,
                                    sampler alpha_samp) {
    float2 pc = float2(point_coord.x, 1.0 - point_coord.y);
    float4 base = draw.base_color;
    if (draw.flags.y == 1u) {
        base *= base_map.sample(samp, map_uv(draw, MAP_BASE, pc));
    }
    if (draw.extra.x == 1u) {
        base.a *= alpha_map.sample(alpha_samp, map_uv(draw, MAP_ALPHA, pc)).g;
    }
    base *= in.color;
    if (base.a < draw.pbr.z) {
        discard_fragment();
    }
    return float4(apply_fog(tone_map(base.rgb, frame, draw), in.view_dist, frame), base.a);
}

#define POINT_FRAGMENT_ARGS                                                    \
    FIn in [[stage_in]],                                                       \
    float2 point_coord [[point_coord]],                                        \
    constant Frame   &frame     [[buffer(0)]],                                 \
    constant Draw    &draw      [[buffer(1)]],                                 \
    texture2d<float>  base_map  [[texture(0)]],                                \
    texture2d<float>  alpha_map [[texture(9)]],                                \
    array<sampler, 10> map_samps [[sampler(0)]]

fragment float4 fs_point(POINT_FRAGMENT_ARGS) {
    return point_fragment(in, point_coord, frame, draw, base_map, MS(MAP_BASE), alpha_map, MS(MAP_ALPHA));
}

/// `fs_mesh_encoded_blend` for points: blended in the encoded values.
fragment float4 fs_point_encoded_blend(POINT_FRAGMENT_ARGS, float4 dst [[color(0)]]) {
    float4 src = point_fragment(in, point_coord, frame, draw, base_map, MS(MAP_BASE), alpha_map, MS(MAP_ALPHA));
    float a = saturate(src.a);
    float3 e = srgb_encode(src.rgb) * a + srgb_encode(dst.rgb) * (1.0 - a);
    return float4(srgb_decode(e), a + dst.a * (1.0 - a));
}

// ------------------------------------------------ order-independent glass

// Weighted-blended order-independent transparency (McGuire and Bavoil, with
// McGuire and Mara's coloured transmission), as `fs_oit` in
// `renderer/shader.rs`: shaded as `fs_mesh` shades, the light added up
// weighted by depth and coverage, the transmittance multiplied down.
struct OitOut {
    float4 accum  [[color(0)]];
    float4 reveal [[color(1)]];
};

fragment OitOut fs_oit_accum(MESH_FRAGMENT_ARGS) {
    SAMPLE_PHYSICAL
    float3 glass_t;
    float4 src = MESH_FRAGMENT_CALL;
    float alpha = saturate(src.a);
    float3 light = src.rgb;
    float3 through = glass_t;
    if (through.x < 0.0) {
        // Not glass: straight colour and alpha.
        light = src.rgb * alpha;
        through = float3(1.0 - alpha);
    }
    // Nearer fragments weigh more; floored above zero so clear glass, which
    // blocks almost nothing, still carries its highlight.
    float a = max(alpha, 1e-3);
    float z = max(in.view_dist, 1e-3);
    float w = a * clamp(10.0 / (1e-5 + pow(z / 60.0, 3.0) + pow(z / 300.0, 6.0)), 1e-2, 3e3);
    OitOut o;
    o.accum = float4(light * w, a * w);
    o.reveal = float4(float3(1.0) - through, alpha);
    return o;
}

vertex float4 vs_oit_fullscreen(uint vid [[vertex_id]]) {
    float2 p = float2(vid == 1u ? 3.0 : -1.0, vid == 2u ? 3.0 : -1.0);
    return float4(p, 0.0, 1.0);
}

// The composite's first half: what is behind times the layers' product of
// transmittance, per channel.
fragment float4 fs_oit_tint(float4 pos [[position]],
                            texture2d<float> accum [[texture(0)]],
                            texture2d<float> reveal [[texture(1)]]) {
    uint2 p = uint2(pos.xy);
    if (accum.read(p).a <= 0.0) {
        discard_fragment();
    }
    return float4(reveal.read(p).rgb, 1.0);
}

// Its second half: the layers' weighted mean light, scaled by how much of the
// pixel they cover.
fragment float4 fs_oit_light(float4 pos [[position]],
                             texture2d<float> accum [[texture(0)]],
                             texture2d<float> reveal [[texture(1)]]) {
    uint2 p = uint2(pos.xy);
    float4 a = accum.read(p);
    if (a.a <= 0.0) {
        discard_fragment();
    }
    float cover = 1.0 - reveal.read(p).a;
    return float4(a.rgb / max(a.a, 1e-5) * cover, cover);
}

// ------------------------------------------------------------------- fur

// Shell fur (`crate::fur`): the WGSL shader's port, reading the same
// sixteen float4 slots (`FurMaterial::data`). Instance `iid` is shell
// `iid / shells` of the fur's length; see `src/fur/mod.rs` for the slots.
struct FurData {
    float4 d[16];
};

static inline float3 fur_air_hash3(int3 p) {
    uint3 v = as_type<uint3>(p) * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v = v ^ (v >> 16u);
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return float3(v >> 8u) * (1.0 / 16777216.0);
}

// The air, as `crate::fur`'s WGSL has it (`fur_noise3`, `fur_mean_pull`,
// `fur_turbulence`, `fur_flutter_noise`): pulls per unit of strand length.
static inline float3 fur_noise3(float3 p) {
    float3 i = floor(p);
    float3 f = p - i;
    float3 u = f * f * (3.0 - 2.0 * f);
    int3 b = int3(i);
    float3 x00 = mix(fur_air_hash3(b), fur_air_hash3(b + int3(1, 0, 0)), u.x);
    float3 x10 = mix(fur_air_hash3(b + int3(0, 1, 0)), fur_air_hash3(b + int3(1, 1, 0)), u.x);
    float3 x01 = mix(fur_air_hash3(b + int3(0, 0, 1)), fur_air_hash3(b + int3(1, 0, 1)), u.x);
    float3 x11 = mix(fur_air_hash3(b + int3(0, 1, 1)), fur_air_hash3(b + int3(1, 1, 1)), u.x);
    float3 v = mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
    return v * 2.0 - 1.0;
}

static inline float3 fur_mean_pull(constant FurData &fur) {
    if (fur.d[9].w > 0.5) {
        return fur.d[9].xyz;
    }
    float3 w = fur.d[6].xyz;
    return fur.d[1].xyz + w * (fur.d[12].x * length(w));
}

static inline float3 fur_turbulence(constant FurData &fur, float3 p) {
    float3 w = fur.d[6].xyz;
    float wl = length(w);
    if (wl <= 0.0 || fur.d[12].y <= 0.0) {
        return float3(0.0);
    }
    float eddy = max(wl / max(fur.d[6].w, 1e-3), 0.05);
    float3 n = fur_noise3((p - w * fur.d[5].z) / eddy);
    return n * (2.0 * fur.d[12].x * fur.d[12].y * wl * wl);
}

static inline float3 fur_brush(constant FurData &fur, float3 p) {
    float4 b = fur.d[13];
    if (b.w <= 0.0) {
        return float3(0.0);
    }
    float3 off = p - b.xyz;
    float d = length(off);
    if (d >= b.w) {
        return float3(0.0);
    }
    float f = 1.0 - d / b.w;
    float w = f * f * (3.0 - 2.0 * f);
    return (fur.d[14].xyz + off / max(d, 1e-4) * fur.d[14].w) * w;
}

static inline float3 fur_flutter_noise(constant FurData &fur, float r) {
    float wl = length(fur.d[6].xyz);
    float amp = fur.d[8].z * min(wl, 1.0);
    if (amp <= 0.0) {
        return float3(0.0);
    }
    float rate = 2.0 + 4.0 * wl;
    return fur_noise3(float3(r * 97.0, fur.d[5].z * rate, r * 31.0)) * amp;
}

/// Strand hair (`crate::fur::strands`, placement 2): the ribbon vertex stage,
/// `HAIR_VERTEX_WGSL`'s port. `uv` is (side, height), the normal the grown
/// direction, the colour's alpha the strand's own random, and the tangent
/// (`uv_sets` slot 2) the root's normal and the strand's length.
static VOut hair_vertex(VOut out, Vertex v, float4x4 model, float4 root, float3 lag,
                        constant Frame &frame, constant FurData &fur) {
    float s = v.uv.y;
    float side = v.uv.x * 2.0 - 1.0;
    float3 rn4 = (model * float4(root.xyz, 0.0)).xyz;
    float scale = max(length(rn4), 1e-6);
    float3 root_n = rn4 / scale;
    float len = root.w * scale;
    float r = v.alpha;
    float3 across0 = abs(root_n.y) > 0.9 ? normalize(cross(root_n, float3(1.0, 0.0, 0.0)))
                                         : normalize(cross(root_n, float3(0.0, 1.0, 0.0)));
    float3 base = out.world_pos - root_n * (len * s);
    float3 pull = fur_mean_pull(fur) + fur_turbulence(fur, base) + fur_flutter_noise(fur, r) * 0.3
        + fur_brush(fur, out.world_pos) + (model * float4(lag, 0.0)).xyz;
    float3 walk = float3(0.0);
    for (int k = 0; k < 6; ++k) {
        float3 d = root_n + pull * (2.0 * s * (float(k) + 0.5) / 6.0);
        walk += d / max(length(d), 1e-6);
    }
    float3 p = out.world_pos - root_n * (len * s) + walk * (len * s / 6.0);
    float3 b = root_n + pull * (2.0 * s);
    float3 bent = length(b) > 1e-4 ? b / length(b) : normalize(pull + root_n * 1e-3);
    float3 tt_dir = normalize(out.normal) - root_n + bent;
    float3 t_dir = length(tt_dir) > 1e-4 ? tt_dir / length(tt_dir) : bent;
    float3 to_eye = normalize(frame.camera_pos[0].xyz - p);
    float3 across = cross(t_dir, to_eye);
    if (length(across) < 1e-5) {
        across = across0;
    }
    across = normalize(across);
    // At least 1.5 pixels wide, as the WGSL draws it; the share its true
    // width fills goes to the fragment in `color.a`, the place across it in
    // `uv12.w`.
    float width = fur.d[0].w * mix(1.0, fur.d[1].w, s);
    float4x4 vm = frame.view[0];
    float3 up = float3(vm[0][1], vm[1][1], vm[2][1]);
    float4 ca = frame.view_proj[0] * float4(p, 1.0);
    float4 ce = frame.view_proj[0] * float4(p + up, 1.0);
    float px_per_unit = (ca.w > 1e-4 && ce.w > 1e-4)
        ? abs(ce.y / ce.w - ca.y / ca.w) * frame.viewport.y * 0.5 : 1e6;
    float width_px = width * px_per_unit;
    float drawn = max(width_px, 1.5) / max(px_per_unit, 1e-6);
    out.world_pos = p + across * (side * drawn * 0.5);
    out.normal = root_n;
    out.position = frame.view_proj[0] * float4(out.world_pos, 1.0);
    out.view_dist = length(frame.camera_pos[0].xyz - out.world_pos);
    out.tangent = float4(t_dir, 1.0);
    out.uv3 = float2(s, 1.0);
    out.color = float4(out.color.rgb, clamp(width_px / max(width_px, 1.5), 0.0, 1.0));
    out.uv12 = float4(float3(v.position), side);
    return out;
}

// `fur_lod_shells` in `crate::fur`'s WGSL: two shells per pixel of the fur's
// length on screen, at the near side of its bounding sphere (`d[11]`; a
// negative radius for none), at least four.
static float fur_lod_shells(constant FurData &fur, constant Frame &frame, constant Draw &draw,
                            float shells) {
    float4 b = fur.d[11];
    if (b.w < 0.0) {
        return shells;
    }
    float4x4 m = draw.model;
    float scale = max(length(m[0].xyz), max(length(m[1].xyz), length(m[2].xyz)));
    float3 c = (m * float4(b.xyz, 1.0)).xyz;
    float3 to_cam = frame.camera_pos[0].xyz - c;
    float d = length(to_cam);
    float3 near = c + to_cam / max(d, 1e-6) * min(b.w * scale, d * 0.9);
    float4x4 v = frame.view[0];
    float3 up = float3(v[0][1], v[1][1], v[2][1]);
    float4 a = frame.view_proj[0] * float4(near, 1.0);
    float4 e = frame.view_proj[0] * float4(near + up, 1.0);
    if (a.w <= 1e-4 || e.w <= 1e-4) {
        return shells;
    }
    float px_per_unit = abs(e.y / e.w - a.y / a.w) * frame.viewport.y * 0.5;
    return clamp(ceil(2.0 * fur.d[0].y * px_per_unit), min(4.0, shells), shells);
}

vertex VOut vs_fur(uint vid [[vertex_id]],
                   uint iid [[instance_id]],
                   device const Vertex   *verts      [[buffer(0)]],
                   constant Frame        &frame      [[buffer(1)]],
                   constant Draw         &draw       [[buffer(2)]],
                   device const float4x4 *instances  [[buffer(3)]],
                   device const float4   *morph_data [[buffer(6)]],
                   constant MorphInfo    &morph      [[buffer(7)]],
                   device const float4   *uv_sets    [[buffer(8)]],
                   constant FurData      &fur        [[buffer(9)]]) {
    float4x4 model = model_matrix(draw, instances, iid);
    Vertex v = apply_morph(verts[vid], vid, iid, morph, morph_data, instances);
    VOut out = transform(v, model, normal_basis(draw, instances, iid), frame, draw, 0u);
    if (fur.d[8].x > 1.5) {
        float4 root = (draw.phys_flags.z & 2u) != 0u ? uv_sets[3u * vid + 2u] : float4(0.0, 1.0, 0.0, 0.0);
        // On a skinned mesh, each strand's own inertia, from uv1 and uv2.
        float3 lag = fur.d[8].w > 0.5 && (draw.phys_flags.z & 1u) != 0u ? uv_sets[3u * vid].xyz : float3(0.0);
        return hair_vertex(out, v, model, root, lag, frame, fur);
    }
    float shells = fur_lod_shells(fur, frame, draw, max(fur.d[0].x, 1.0));
    // A shell beyond what the fur's size on screen needs: nothing to draw.
    if (float(iid) > shells) {
        out.position = float4(0.0, 0.0, 2.0, 1.0);
        return out;
    }
    float h = float(iid) / shells;
    float len = fur.d[0].y * out.color.a;
    float3 n = normalize(out.normal);
    float3 pull = fur_mean_pull(fur) + fur_turbulence(fur, out.world_pos) + fur_brush(fur, out.world_pos);
    if (fur.d[8].w > 0.5 && (draw.phys_flags.z & 2u) != 0u) {
        pull += (model * float4(uv_sets[3u * vid + 2u].xyz, 0.0)).xyz;
    }
    float3 walk = float3(0.0);
    for (int k = 0; k < 6; ++k) {
        float3 d = n + pull * (2.0 * h * (float(k) + 0.5) / 6.0);
        walk += d / max(length(d), 1e-6);
    }
    out.world_pos += walk * (len * h / 6.0);
    out.position = frame.view_proj[0] * float4(out.world_pos, 1.0);
    out.view_dist = length(frame.camera_pos[0].xyz - out.world_pos);
    float3 along = n + pull * (2.0 * h);
    out.tangent = float4(length(along) > 1e-6 ? normalize(along) : n, 1.0);
    out.uv3 = float2(h, out.color.a);
    // The skin point in object space, for surface placement.
    // The skin point in object space, and one shell's slab of height.
    out.uv12 = float4(float3(v.position), 1.0 / shells);
    return out;
}

static inline float2 fur_hash2(int2 p) {
    uint2 v = as_type<uint2>(p) * 1664525u + 1013904223u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v >> 16u);
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v >> 16u);
    return float2(v >> 8u) * (1.0 / 16777216.0);
}

static inline float3 fur_hash3(int3 p) {
    uint3 v = as_type<uint3>(p) * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v = v ^ (v >> 16u);
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return float3(v >> 8u) * (1.0 / 16777216.0);
}

// Wrapped in floats, as the WGSL does: exact for whole-number cells, and no
// integer division.
static inline int2 fur_cell_key(float2 cell, float2 n, int salt, int seed) {
    float2 w = cell - n * floor(cell / n);
    return int2(w) + int2(seed * 7919 + salt, seed * 104729 - salt);
}

static inline float2 fur_flutter(constant FurData &fur, float r, float h) {
    return fur_flutter_noise(fur, r).xy * (h * h);
}

// `fur_capsule`, `fur_slab_world`, `fur_world_to` in the WGSL: each strand
// swept across its shell's slab as the line of sight crosses it.
static inline float fur_capsule(float2 o, float2 d) {
    float dd = dot(d, d);
    float t = dd > 1e-12 ? clamp(-dot(o, d) / dd, -0.5, 0.5) : 0.0;
    return length(o + d * t);
}

static inline float2 fur_world_to(float3 step, float3 wx, float3 wy) {
    float a = dot(wx, wx);
    float b = dot(wx, wy);
    float c = dot(wy, wy);
    float det = a * c - b * b;
    if (abs(det) < 1e-20) {
        return float2(0.0);
    }
    float rx = dot(wx, step);
    float ry = dot(wy, step);
    return float2(c * rx - b * ry, a * ry - b * rx) / det;
}

// `fur_reach` and `fur_hit_uv` in `crate::fur`'s WGSL: a strand's furthest
// stray from its root, and the search — own cell first, first hit wins,
// out-of-reach roots and short strands passed over before their clump.
static inline float fur_reach(constant FurData &fur, float h) {
    float curl = 0.3 * h * min(abs(fur.d[4].x), 1.0);
    float flutter = fur.d[8].z * h * h * 1.4143;
    float pull = min(0.95, fur.d[4].y * h * (2.25 * fur.d[5].w + 2.0));
    return curl + flutter + pull + fur.d[0].w + 0.0001;
}

// The samples of this pixel, as offsets from its centre in pixels: the
// pipeline's own pattern (its centre alone when it has one sample).
static inline float2 fur_sample_offset(uint i) {
    return get_num_samples() > 1 ? get_sample_position(i) - 0.5 : float2(0.0);
}

// `fur_mask_uv` in the WGSL: which of the pixel's samples the strands about
// `uv` cover — `jx`, `jy` its steps in cells.
static inline uint fur_mask_uv(constant FurData &fur, float2 uv, float2 jx, float2 jy, float h,
                               float2 slab) {
    int seed = int(fur.d[4].w);
    float2 cells = fur.d[7].xy;
    float2 clumps = fur.d[7].zw;
    float2 q = uv * cells;
    float2 base = floor(q);
    uint n = get_num_samples();
    uint full = (1u << n) - 1u;
    float spread = n > 1 ? 0.4 * (length(jx) + length(jy)) : 0.0;
    float reach = fur_reach(fur, h) + spread + 0.5 * length(slab);
    uint mask = 0u;
    for (int k = 0; k < 9; ++k) {
        float2 cell = base + float2(float((k + 4) % 3 - 1), float((k + 4) / 3 % 3 - 1));
        float2 r = fur_hash2(fur_cell_key(cell, cells, 0, seed));
        float2 root = cell + 0.15 + 0.7 * r;
        if (length(q - root) > reach) {
            continue;
        }
        float2 r2 = fur_hash2(fur_cell_key(cell, cells, 31337, seed));
        float own_len = 1.0 - fur.d[4].z * r2.x;
        if (h > own_len) {
            continue;
        }
        float2 c = root;
        float angle = (fur.d[4].x * h + r2.y) * 6.2831853;
        c += float2(cos(angle), sin(angle)) * (0.3 * h * min(abs(fur.d[4].x), 1.0));
        c += fur_flutter(fur, r.x, h);
        float2 rc = root / cells * clumps;
        float2 cb = floor(rc - 0.5);
        float2 best = float2(0.0);
        float best_d = 1e9;
        for (int j = 0; j <= 1; ++j) {
            for (int i = 0; i <= 1; ++i) {
                float2 kk = cb + float2(float(i), float(j));
                float2 point = kk + fur_hash2(fur_cell_key(kk, clumps, 7, seed));
                float d = dot(point - rc, point - rc);
                if (d < best_d) {
                    best_d = d;
                    best = point;
                }
            }
        }
        float2 pull = (best / clumps * cells - c) * (fur.d[4].y * h);
        float pl = length(pull);
        if (pl > 0.95) {
            pull *= 0.95 / pl;
        }
        c += pull;
        float radius = fur.d[0].w * mix(1.0, fur.d[1].w, h / max(own_len, 1e-3));
        float d = fur_capsule(q - c, slab);
        if (d + spread < radius) {
            mask = full;
        } else if (d - spread < radius) {
            for (uint i = 0; i < n; ++i) {
                float2 o = fur_sample_offset(i);
                if (fur_capsule(q + jx * o.x + jy * o.y - c, slab) < radius) {
                    mask |= 1u << i;
                }
            }
        }
        if (mask == full) {
            return mask;
        }
    }
    return mask;
}

static inline uint fur_mask_surface(constant FurData &fur, float3 p, float3 n, float h,
                                    float3 jx, float3 jy, float3 slab3) {
    int seed = int(fur.d[4].w);
    float3 q = p / fur.d[8].y;
    uint samples = get_num_samples();
    uint full = (1u << samples) - 1u;
    uint mask = 0u;
    float3 base = floor(q);
    float3 helper = abs(n.x) > 0.8 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    float3 t1 = normalize(cross(n, helper));
    float3 t2 = cross(n, t1);
    float csize = fur.d[5].w;
    float curl_r = 0.3 * h * min(abs(fur.d[4].x), 1.0);
    float pull_r = min(0.95, fur.d[4].y * h * (1.3 * csize + 0.01));
    float reach = curl_r + fur.d[8].z * h * h * 1.4143 + pull_r + fur.d[0].w + 0.0001
        + (samples > 1 ? 0.4 * (length(jx) + length(jy)) / fur.d[8].y : 0.0);
    float2 slab = float2(dot(slab3, t1), dot(slab3, t2));
    reach += 0.5 * length(slab);
    for (int z = -1; z <= 1; ++z) {
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {
                float3 cell = base + float3(float(x), float(y), float(z));
                int3 key = int3(cell) + int3(seed * 7919, seed * 104729, 31);
                float3 r = fur_hash3(key);
                float3 root = cell + r;
                float3 off = root - q;
                float dn = dot(off, n);
                if (abs(dn) > 0.5) {
                    continue;
                }
                float3 r2 = fur_hash3(key + int3(977, 31337, 4099));
                float own_len = 1.0 - fur.d[4].z * r2.x;
                if (h > own_len) {
                    continue;
                }
                if (length(off - n * dn) > reach) {
                    continue;
                }
                // Where the strand is carried from its root at this height.
                float3 s = float3(0.0);
                float angle = (fur.d[4].x * h + r2.y) * 6.2831853;
                s += (t1 * cos(angle) + t2 * sin(angle)) * (0.3 * h * min(abs(fur.d[4].x), 1.0));
                float2 fl = fur_flutter(fur, r2.z, h);
                s += t1 * fl.x + t2 * fl.y;
                float3 cp = (floor(root / csize) + 0.25 + 0.5 * fur_hash3(int3(floor(root / csize)) + 17)) * csize;
                float3 pull = cp - root;
                pull = (pull - n * dot(pull, n)) * (fur.d[4].y * h);
                float pl = length(pull);
                if (pl > 0.95) {
                    pull *= 0.95 / pl;
                }
                s += pull;
                float radius = fur.d[0].w * mix(1.0, fur.d[1].w, h / max(own_len, 1e-3));
                for (uint i = 0; i < samples; ++i) {
                    float2 o = fur_sample_offset(i);
                    float3 at = off - (jx * o.x + jy * o.y) / fur.d[8].y;
                    float3 o2 = at - n * dot(at, n) + s;
                    if (fur_capsule(float2(dot(o2, t1), dot(o2, t2)), slab) < radius) {
                        mask |= 1u << i;
                    }
                }
                if (mask == full) {
                    return mask;
                }
            }
        }
    }
    return mask;
}

static inline float3 fur_kajiya(constant FurData &fur, float3 t, float3 n, float3 v, float3 l,
                                float3 albedo, float h, float spec_exp) {
    float ndl = dot(n, l);
    if (h <= 0.0) {
        return albedo * max(ndl, 0.0);
    }
    float tl = dot(t, l);
    float diffuse = sqrt(max(1.0 - tl * tl, 0.0)) * clamp(ndl * 0.75 + 0.25, 0.0, 1.0);
    float3 ts = normalize(t + n * fur.d[5].y);
    float th = dot(ts, normalize(l + v));
    float spec = pow(sqrt(max(1.0 - th * th, 0.0)), spec_exp) * fur.d[3].w
        * clamp(ndl * 0.5 + 0.5, 0.0, 1.0) * h;
    return albedo * diffuse + float3(spec);
}

// The fur's colour, and the samples of the pixel it covers.
struct FurOut {
    float4 color [[color(0)]];
    uint mask [[sample_mask]];
};

fragment FurOut fs_fur(FIn in [[stage_in]],
                       constant Frame &frame [[buffer(0)]],
                       constant Draw &draw [[buffer(1)]],
                       constant FurData &fur [[buffer(2)]],
                       depth2d_array<float> shadow_map [[texture(6)]],
                       depth2d_array<float> spot_map [[texture(7)]],
                       depth2d_array<float> point_map [[texture(8)]],
                       sampler shadow_samp [[sampler(11)]]) {
    float h = in.uv3.x;
    // As `fs_main` in the WGSL: derivatives first, then which samples the
    // strands (or the hair's true width, `color.a` of its ribbon, across it
    // at `uv12.w`) cover.
    float3 px = dfdx(in.uv12.xyz);
    float3 py = dfdy(in.uv12.xyz);
    float3 fn_raw = cross(px, py);
    float2 jx = dfdx(in.uv * fur.d[7].xy);
    float2 jy = dfdy(in.uv * fur.d[7].xy);
    float3 wx = dfdx(in.world_pos);
    float3 wy = dfdy(in.world_pos);
    // One shell's slab, swept as the line of sight crosses it relative to
    // the leaning strand (`fur_slab_world`).
    float3 slab_w = float3(0.0);
    {
        float3 nn = normalize(in.normal);
        float3 vv = normalize(frame.camera_pos[0].xyz - in.world_pos);
        float3 st = normalize(in.tangent.xyz);
        float vn = dot(vv, nn);
        float sn = max(dot(st, nn), 0.2);
        float3 ray = (vv - nn * vn) / ((vn < 0.0 ? -1.0 : (vn > 0.0 ? 1.0 : 0.0)) * max(abs(vn), 0.2));
        float3 strand = (st - nn * dot(st, nn)) / sn;
        slab_w = (ray - strand) * (fur.d[0].y * in.uv3.y * in.uv12.w);
    }
    float2 slab_m = fur_world_to(slab_w, wx, wy);
    float sx = dfdx(in.uv12.w);
    float sy = dfdy(in.uv12.w);
    float3 dt = fwidth(in.tangent.xyz);
    float spec_exp = fur.d[5].x / (1.0 + fur.d[5].x * dot(dt, dt) * 0.25);
    uint samples = get_num_samples();
    uint mask = (1u << samples) - 1u;
    if (fur.d[8].x > 1.5) {
        mask = 0u;
        for (uint i = 0; i < samples; ++i) {
            float2 o = fur_sample_offset(i);
            if (abs(in.uv12.w + sx * o.x + sy * o.y) < in.color.a) {
                mask |= 1u << i;
            }
        }
    } else if (h > 0.0) {
        if (in.uv3.y <= 0.001) {
            discard_fragment();
        }
        if (fur.d[8].x > 0.5) {
            float fl = length(fn_raw);
            float3 sl = px * slab_m.x + py * slab_m.y;
            float sll = length(sl) / fur.d[8].y;
            sl = sll > 3.0 ? sl * (3.0 / sll) : sl;
            mask = fl > 0.0 ? fur_mask_surface(fur, in.uv12.xyz, fn_raw / fl, h, px, py, sl / fur.d[8].y) : 0u;
        } else {
            float2 sc = jx * slab_m.x + jy * slab_m.y;
            float scl = length(sc);
            sc = scl > 3.0 ? sc * (3.0 / scl) : sc;
            mask = fur_mask_uv(fur, in.uv, jx, jy, h, sc);
        }
    }
    if (mask == 0u) {
        discard_fragment();
    }
    float3 n = normalize(in.normal);
    float3 t = normalize(in.tangent.xyz);
    float3 v = normalize(frame.camera_pos[0].xyz - in.world_pos);
    ShadowMaps maps = { shadow_map, spot_map, point_map, shadow_samp };
    float shadow_cache[16];
    for (uint r = 0u; r < 16u; ++r) {
        shadow_cache[r] = 1.0;
    }
    if (draw.object.x != 0u) {
        uint count = min(frame.shadow_info.x, 16u);
        for (uint r = 0u; r < count; ++r) {
            shadow_cache[r] = shadow_at(frame, int(r), maps, in.world_pos, n);
        }
    }
    float3 tint = mix(fur.d[2].rgb, fur.d[3].rgb, h) * in.color.rgb;
    float ao = mix(1.0 - fur.d[2].w, 1.0, h);
    float3 albedo = tint * ao;
    float3 irr = frame.ambient.rgb;
    for (uint i = 0u; i < frame.counts.w && i < 2u; ++i) {
        HemiLight hl = frame.hemis[i];
        float k = dot(n, normalize(hl.up.xyz)) * 0.5 + 0.5;
        irr += mix(hl.ground.rgb, hl.sky.rgb, k);
    }
    float3 col = albedo * irr * (1.0 / M_PI_F);
    for (uint i = 0u; i < frame.counts.x && i < 4u; ++i) {
        DirLight d = frame.dir[i];
        float3 l = -normalize(d.direction.xyz);
        col += d.color.rgb * cached_shadow(shadow_cache, d.color.w) * fur_kajiya(fur, t, n, v, l, albedo, h, spec_exp);
    }
    for (uint i = 0u; i < frame.counts.y && i < 8u; ++i) {
        PointLight p = frame.pts[i];
        float3 to_light = p.position.xyz - in.world_pos;
        float dist = length(to_light);
        float att = punctual_attenuation(dist, p.params.x, p.params.y) * cached_shadow(shadow_cache, p.color.w);
        col += p.color.rgb * att * fur_kajiya(fur, t, n, v, to_light / max(dist, 1e-4), albedo, h, spec_exp);
    }
    for (uint i = 0u; i < frame.counts.z && i < 4u; ++i) {
        SpotLight sp = frame.spots[i];
        float3 to_light = sp.position.xyz - in.world_pos;
        float dist = length(to_light);
        float3 l = to_light / max(dist, 1e-4);
        float cos_angle = dot(-l, normalize(sp.direction.xyz));
        float cone = cos_angle > sp.params.z ? smoothstep(sp.params.z, sp.params.w, cos_angle) : 0.0;
        float att = punctual_attenuation(dist, sp.params.x, sp.params.y) * cone
            * cached_shadow(shadow_cache, sp.color.w);
        col += sp.color.rgb * att * fur_kajiya(fur, t, n, v, l, albedo, h, spec_exp);
    }
    return {float4(apply_fog(tone_map(col, frame, draw), in.view_dist, frame), 1.0), mask};
}
