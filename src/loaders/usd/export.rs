//! Writing geometry and scenes out as USD.
//!
//! The output is what `UsdGeomMesh` asks for and nothing more: points, face
//! counts, face indices, and whichever of normals and `primvars:st` the
//! geometry actually carries. Faces are written as triangles rather than being
//! re-joined into polygons — a `BufferGeometry` has already thrown away which
//! triangles used to be one face, and inventing that back would be a guess.

use crate::animation::{AnimationClip, KeyframeTrack, TrackTarget, TrackValues};
use crate::core::{BufferGeometry, ObjectArena, ObjectId, ObjectKind};
use crate::materials::Material;
use crate::textures::{Texture, TextureFormat, TextureWrap};

use super::parse::{Specifier, UsdLayer, UsdPrim, UsdProperty};
use super::value::UsdValue;
use super::write::layer_to_usda;

/// How the layer describes its own units. USD's default is metres.
#[derive(Debug, Clone)]
pub struct UsdExportOptions {
    /// `metersPerUnit`. 1.0 says the numbers are metres, 0.01 that they are
    /// centimetres — which is what most DCC tools and every `.usdz` from a
    /// phone actually contain.
    pub meters_per_unit: f32,
    /// `upAxis`, `Y` or `Z`.
    pub up_axis: char,
    /// The prim a reader should open by default.
    pub default_prim: String,
    /// How many time codes make a second when animation is exported. 24 is
    /// USD's own default and what every reader assumes when a layer is silent.
    pub time_codes_per_second: f32,
}

impl Default for UsdExportOptions {
    fn default() -> Self {
        Self {
            meters_per_unit: 1.0,
            up_axis: 'Y',
            default_prim: "Root".into(),
            time_codes_per_second: 24.0,
        }
    }
}

fn float_array(values: &[f32], stride: usize) -> UsdValue {
    UsdValue::Array(
        values
            .chunks_exact(stride)
            .map(|c| UsdValue::Tuple(c.iter().map(|v| UsdValue::Float(*v as f64)).collect()))
            .collect(),
    )
}

fn int_array(values: impl Iterator<Item = u32>) -> UsdValue {
    UsdValue::Array(values.map(|v| UsdValue::Int(v as i128)).collect())
}

fn uniform(mut p: UsdProperty) -> UsdProperty {
    p.uniform = true;
    p
}

fn attribute(type_name: &str, name: &str, value: UsdValue) -> UsdProperty {
    UsdProperty {
        qualifier: String::new(),
        name: name.into(),
        type_name: type_name.into(),
        uniform: false,
        relationship: false,
        value,
        metadata: Vec::new(),
    }
}

fn interpolated(mut p: UsdProperty, how: &str) -> UsdProperty {
    p.metadata
        .push(("interpolation".into(), UsdValue::Token(how.into())));
    p
}

/// A `UsdGeomMesh` prim for one geometry.
pub fn mesh_prim(g: &BufferGeometry, name: &str) -> UsdPrim {
    let positions = g
        .get_attribute("position")
        .map(|a| a.array.clone())
        .unwrap_or_default();
    let vertices = positions.len() / 3;
    let indices: Vec<u32> = match &g.index {
        Some(i) => i.to_vec(),
        // Unindexed geometry is already one vertex per corner.
        None => (0..vertices as u32).collect(),
    };
    let triangles = indices.len() / 3;

    let mut properties = vec![
        attribute("point3f[]", "points", float_array(&positions, 3)),
        // Every face is a triangle; see the module note on why.
        attribute(
            "int[]",
            "faceVertexCounts",
            int_array(std::iter::repeat_n(3u32, triangles)),
        ),
        attribute(
            "int[]",
            "faceVertexIndices",
            int_array(indices.iter().copied().take(triangles * 3)),
        ),
    ];

    // `extent` is not decoration: a reader uses it to frame the asset without
    // touching the points, and Quick Look will not show a mesh that lacks one.
    if vertices > 0 {
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for p in positions.chunks_exact(3) {
            for a in 0..3 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
        properties.push(attribute(
            "float3[]",
            "extent",
            UsdValue::Array(vec![
                UsdValue::Tuple(lo.iter().map(|v| UsdValue::Float(*v as f64)).collect()),
                UsdValue::Tuple(hi.iter().map(|v| UsdValue::Float(*v as f64)).collect()),
            ]),
        ));
    }

    if let Some(n) = g.get_attribute("normal") {
        properties.push(interpolated(
            attribute("normal3f[]", "normals", float_array(&n.array, 3)),
            "vertex",
        ));
    }
    // USD's fallback shading, which every tool that does not evaluate a shader
    // graph falls back to — `usdview` flat-shaded, thumbnailers, importers that
    // read geometry and skip materials. Without it a vertex-coloured mesh
    // exported from here arrived grey, and the colours were simply gone.
    if let Some(c) = g.get_attribute("color") {
        properties.push(interpolated(
            attribute("color3f[]", "primvars:displayColor", float_array(&c.array, 3)),
            "vertex",
        ));
    }
    if let Some(uv) = g.get_attribute("uv") {
        properties.push(interpolated(
            attribute("texCoord2f[]", "primvars:st", float_array(&uv.array, 2)),
            "vertex",
        ));
    }
    // Without this a reader is entitled to subdivide the mesh, which turns a
    // faceted model into a smooth approximation of itself.
    properties.push(UsdProperty {
        qualifier: String::new(),
        name: "subdivisionScheme".into(),
        type_name: "token".into(),
        uniform: true,
        relationship: false,
        value: UsdValue::Token("none".into()),
        metadata: Vec::new(),
    });

    UsdPrim {
        specifier: Specifier::Def,
        type_name: "Mesh".into(),
        name: sanitize(name),
        metadata: Vec::new(),
        properties,
        children: Vec::new(),
        variant_sets: Vec::new(),
    }
}

/// An image an exported material refers to, and the bytes to write for it.
///
/// A `.usda` refers to its textures by relative path, so these have to land
/// beside the layer for the reference to resolve. A `.usdz` carries them
/// instead, which is why [`scene_to_usdz`](super::scene_to_usdz) needs no help
/// and the text form does.
#[derive(Debug, Clone)]
pub struct ExportedTexture {
    /// The asset path written into the layer, relative to it.
    pub path: String,
    /// PNG-encoded pixels.
    pub data: Vec<u8>,
}

/// A texture's pixels as 8-bit RGBA.
///
/// `None` when there is nothing to write: a texture whose bytes live only on
/// the GPU (see `Texture::without_pixels`), or one held block-compressed,
/// which would have to be decoded first and is a different job. The material
/// still exports — it keeps its constant values and loses only the map.
fn rgba8(texture: &Texture) -> Option<Vec<u8>> {
    let count = (texture.width as usize).checked_mul(texture.height as usize)?;
    if count == 0 || texture.data.is_empty() {
        return None;
    }
    let pixels = unflipped(texture, count)?;
    // An image file stores the top row first and USD samples with `v = 1` at
    // that row — verified by rendering a half-red, half-blue image and seeing
    // which half Hydra put on top. `flip_y` is this crate's flag for "the top
    // row is first", so a texture with it *off* is authored bottom-up and has
    // to be turned over on the way out or it arrives upside down. USD has no
    // flag to carry the distinction; the rows have to be right.
    if texture.flip_y {
        return Some(pixels);
    }
    let stride = texture.width as usize * 4;
    Some(
        pixels
            .chunks_exact(stride)
            .rev()
            .flatten()
            .copied()
            .collect(),
    )
}

/// The pixels as 8-bit RGBA in the order they are held.
fn unflipped(texture: &Texture, count: usize) -> Option<Vec<u8>> {
    match texture.format {
        TextureFormat::Rgba8UnormSrgb | TextureFormat::Rgba8Unorm => {
            texture.data.get(..count * 4).map(<[u8]>::to_vec)
        }
        // One channel becomes grey, so the written file says what the single
        // channel said however a reader samples it.
        TextureFormat::R8Unorm => texture
            .data
            .get(..count)
            .map(|d| d.iter().flat_map(|v| [*v, *v, *v, 255]).collect()),
        // Half float down to 8 bits. Anything above 1.0 clamps, which is a
        // real loss and only reachable for a map holding an HDR image.
        TextureFormat::Rgba16Float => texture.data.get(..count * 8).map(|d| {
            d.chunks_exact(2)
                .map(|h| {
                    let bits = u16::from_le_bytes([h[0], h[1]]);
                    let value = half_bits_to_f32(bits);
                    (value.clamp(0.0, 1.0) * 255.0).round() as u8
                })
                .collect()
        }),
        _ => None,
    }
}

/// An IEEE half as an `f32`.
fn half_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x3ff) as u32;
    let assembled = match exponent {
        0 if mantissa == 0 => sign << 31,
        // Subnormal: normalise it by hand, since the exponent fields do not
        // line up between the two formats.
        0 => {
            let mut e = -1i32;
            let mut m = mantissa;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (sign << 31) | (((127 - 15 + e + 1) as u32) << 23) | ((m & 0x3ff) << 13)
        }
        0x1f => (sign << 31) | (0xff << 23) | (mantissa << 13),
        _ => (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13),
    };
    f32::from_bits(assembled)
}

/// USD's name for a wrap mode.
fn wrap_name(wrap: TextureWrap) -> &'static str {
    match wrap {
        TextureWrap::Repeat => "repeat",
        TextureWrap::ClampToEdge => "clamp",
        TextureWrap::MirroredRepeat => "mirror",
    }
}

/// One `UsdUVTexture` an export has to write, and where it plugs in.
struct Slot<'a> {
    /// The shader prim's name, and the stem of the image file.
    name: &'static str,
    /// The surface input it drives, and that input's type.
    input: &'static str,
    type_name: &'static str,
    /// Which of the texture shader's outputs to take. `rgb` for colour and
    /// normals, `r` for the maps that are one channel of data.
    channel: &'static str,
    /// `sRGB` for colour, `raw` for data. A normal map read as sRGB is wrong
    /// in a way that looks like bad lighting rather than like a bug.
    color_space: &'static str,
    /// `sampled * scale + bias`. Only a normal map needs them, to turn `[0,1]`
    /// into `[-1,1]`, and leaving them off points every normal into the surface.
    scale: [f32; 4],
    bias: [f32; 4],
    texture: &'a Texture,
}

fn slot<'a>(
    name: &'static str,
    input: &'static str,
    type_name: &'static str,
    channel: &'static str,
    color_space: &'static str,
    texture: &'a Texture,
) -> Slot<'a> {
    Slot {
        name,
        input,
        type_name,
        channel,
        color_space,
        scale: [1.0; 4],
        bias: [0.0; 4],
        texture,
    }
}

/// Every map a material carries, as the texture shaders USD wants for them.
fn texture_slots(material: &Material) -> Vec<Slot<'_>> {
    let mut out = Vec::new();
    macro_rules! push {
        ($opt:expr, $name:literal, $input:literal, $ty:literal, $ch:literal, $cs:literal) => {
            if let Some(t) = $opt.as_deref() {
                out.push(slot($name, $input, $ty, $ch, $cs, t));
            }
        };
    }
    match material {
        Material::Standard(m) => {
            push!(m.map, "DiffuseTex", "inputs:diffuseColor", "color3f", "rgb", "sRGB");
            push!(m.emissive_map, "EmissiveTex", "inputs:emissiveColor", "color3f", "rgb", "sRGB");
            push!(m.roughness_map, "RoughnessTex", "inputs:roughness", "float", "r", "raw");
            push!(m.metalness_map, "MetallicTex", "inputs:metallic", "float", "r", "raw");
            push!(m.ao_map, "OcclusionTex", "inputs:occlusion", "float", "r", "raw");
            push!(m.displacement_map, "DisplacementTex", "inputs:displacement", "float", "r", "raw");
            if let Some(t) = m.normal_map.as_deref() {
                out.push(normal_slot(t));
            }
        }
        Material::Physical(m) => {
            push!(m.map, "DiffuseTex", "inputs:diffuseColor", "color3f", "rgb", "sRGB");
            push!(m.emissive_map, "EmissiveTex", "inputs:emissiveColor", "color3f", "rgb", "sRGB");
            push!(m.roughness_map, "RoughnessTex", "inputs:roughness", "float", "r", "raw");
            push!(m.metalness_map, "MetallicTex", "inputs:metallic", "float", "r", "raw");
            push!(m.ao_map, "OcclusionTex", "inputs:occlusion", "float", "r", "raw");
            push!(m.displacement_map, "DisplacementTex", "inputs:displacement", "float", "r", "raw");
            if let Some(t) = m.normal_map.as_deref() {
                out.push(normal_slot(t));
            }
        }
        Material::Basic(m) => {
            push!(m.map, "DiffuseTex", "inputs:diffuseColor", "color3f", "rgb", "sRGB")
        }
        Material::Sprite(m) => {
            push!(m.map, "DiffuseTex", "inputs:diffuseColor", "color3f", "rgb", "sRGB")
        }
        _ => {}
    }
    out
}

/// A normal map, which is the one slot with a scale and bias that matter.
fn normal_slot(texture: &Texture) -> Slot<'_> {
    Slot {
        scale: [2.0, 2.0, 2.0, 1.0],
        bias: [-1.0, -1.0, -1.0, 0.0],
        ..slot("NormalTex", "inputs:normal", "normal3f", "rgb", "raw", texture)
    }
}

fn tuple2(v: [f32; 2]) -> UsdValue {
    UsdValue::Tuple(v.iter().map(|c| UsdValue::Float(*c as f64)).collect())
}

fn token(name: &str, value: &str) -> UsdProperty {
    let mut p = attribute("token", name, UsdValue::Token(value.into()));
    p.uniform = true;
    p
}

/// A shader prim of the given `info:id`.
fn shader(name: &str, id: &str, mut properties: Vec<UsdProperty>) -> UsdPrim {
    properties.insert(0, token("info:id", id));
    UsdPrim {
        specifier: Specifier::Def,
        type_name: "Shader".into(),
        name: name.into(),
        metadata: Vec::new(),
        properties,
        children: Vec::new(),
        variant_sets: Vec::new(),
    }
}

/// A connection, which the writer spells by appending `.connect`.
fn connect(type_name: &str, name: &str, target: String) -> UsdProperty {
    attribute(type_name, name, UsdValue::Path(target))
}

/// What a material of this crate's becomes in `UsdPreviewSurface` terms.
///
/// USD has one surface shader, so every material has to arrive as some
/// setting of it. Most of them are a colour and a little more, and the
/// mapping is only interesting where it is lossy — which is recorded on each
/// arm of [`preview_surface`].
struct Preview {
    color: [f32; 3],
    roughness: f32,
    metallic: f32,
    emissive: [f32; 3],
    opacity: f32,
    /// Phong's specular colour, which needs `useSpecularWorkflow` alongside
    /// it. `None` for everything that works in the metallic workflow.
    specular: Option<[f32; 3]>,
    /// Index of refraction. `None` leaves USD's own default of 1.5 alone
    /// rather than restating it.
    ior: Option<f32>,
    /// Clear coat and its roughness, for the materials that have one.
    clearcoat: Option<(f32, f32)>,
    /// The alpha below which a fragment is discarded outright. USD's default
    /// is zero, meaning "blend", so only a material that actually cuts out
    /// says anything.
    opacity_threshold: f32,
}

/// Blinn-Phong's exponent as a GGX roughness.
///
/// The usual approximation, and exact at neither end, but it keeps the
/// distinction between a shininess of 5 and one of 200 — which mapping every
/// Phong material to a fixed roughness does not.
fn roughness_from_shininess(shininess: f32) -> f32 {
    (2.0 / (shininess.max(0.0) + 2.0)).sqrt().clamp(0.0, 1.0)
}

/// One of this crate's materials as a `UsdPreviewSurface` setting.
fn preview_surface(material: &Material) -> Preview {
    let rgb = |c: &crate::math::Color| [c.r, c.g, c.b];
    let plain = |color: [f32; 3], opacity: f32| Preview {
        color,
        roughness: 1.0,
        metallic: 0.0,
        emissive: [0.0; 3],
        opacity,
        specular: None,
        ior: None,
        clearcoat: None,
        opacity_threshold: 0.0,
    };
    match material {
        Material::Standard(s) => Preview {
            color: rgb(&s.color),
            roughness: s.roughness,
            metallic: s.metalness,
            emissive: rgb(&s.emissive),
            opacity: s.opacity,
            specular: None,
            ior: None,
            clearcoat: None,
            opacity_threshold: 0.0,
        },
        Material::Physical(p) => Preview {
            color: rgb(&p.color),
            roughness: p.roughness,
            metallic: p.metalness,
            emissive: rgb(&p.emissive),
            opacity: p.opacity,
            specular: None,
            // The two things a physical material has that a standard one does
            // not, and that USD's preview surface happens to have as well.
            ior: Some(p.ior),
            clearcoat: (p.clearcoat > 0.0).then_some((p.clearcoat, p.clearcoat_roughness)),
            opacity_threshold: 0.0,
        },
        // Unlit in this crate, and USD has no unlit preview surface. A fully
        // rough dielectric is the closest lit stand-in; the colour is what
        // matters and it is kept exactly.
        Material::Basic(b) => Preview {
            opacity_threshold: b.alpha_test,
            ..plain(rgb(&b.color), b.opacity)
        },
        Material::Lambert(l) => Preview {
            emissive: rgb(&l.emissive),
            ..plain(rgb(&l.color), l.opacity)
        },
        // The one arm with a real conversion in it: Phong's specular colour
        // and exponent become USD's specular workflow, which is what that
        // workflow is for.
        Material::Phong(p) => Preview {
            roughness: roughness_from_shininess(p.shininess),
            emissive: rgb(&p.emissive),
            specular: Some(rgb(&p.specular)),
            ..plain(rgb(&p.color), p.opacity)
        },
        // Banded shading is a look, not a surface, and nothing in USD
        // reproduces it. The colour survives and the banding does not.
        Material::Toon(t) => Preview {
            emissive: rgb(&t.emissive),
            ..plain(rgb(&t.color), t.opacity)
        },
        Material::Matcap(m) => plain(rgb(&m.color), m.opacity),
        Material::Sprite(s) => plain(rgb(&s.color), s.opacity),
        Material::Points(p) => plain(rgb(&p.color), p.opacity),
        Material::Line(l) => plain(rgb(&l.color), l.opacity),
        // A mirror is a smooth metal, which USD does have.
        Material::Mirror(m) => Preview {
            roughness: 0.0,
            metallic: 1.0,
            ..plain(rgb(&m.color), 1.0)
        },
        Material::Atmosphere(a) => plain(rgb(&a.color), 1.0),
        // Normal, Depth, Distance, Sky and a user's own shader are all
        // computed rather than coloured, so there is nothing to carry across
        // and a neutral surface is the honest result.
        Material::Normal(n) => plain([0.8; 3], n.opacity),
        Material::Depth(d) => plain([0.8; 3], d.opacity),
        Material::Distance(_) | Material::Sky(_) | Material::Shader(_) => plain([0.8; 3], 1.0),
    }
}

/// A `UsdPreviewSurface` material prim, with whatever textures it needs.
///
/// The textures come back rather than being written here, because where they
/// go depends on the form the layer is taking: beside a `.usda`, inside a
/// `.usdz`.
pub fn material_prim_with_textures(
    material: &Material,
    name: &str,
    root: &str,
) -> (UsdPrim, Vec<ExportedTexture>) {
    let mut prim = material_prim(material, name);
    let slots = texture_slots(material);
    if slots.is_empty() {
        return (prim, Vec::new());
    }
    let path = format!("{root}/{}", sanitize(name));
    let mut images = Vec::new();
    let mut nodes = Vec::new();
    let mut reader_used = false;

    for slot in slots {
        let Some(pixels) = rgba8(slot.texture) else {
            // Nothing writable — the constant value on the surface stands in.
            continue;
        };
        let file = format!("textures/{}_{}.png", sanitize(name), slot.name);
        images.push(ExportedTexture {
            path: file.clone(),
            data: crate::utils::png::encode_png(slot.texture.width, slot.texture.height, &pixels),
        });

        // A texture is sampled by a primvar, and a UV transform sits between
        // the two when there is one. `UsdTransform2d` takes degrees where this
        // crate keeps radians.
        let transform_needed = slot.texture.offset.x != 0.0
            || slot.texture.offset.y != 0.0
            || slot.texture.repeat.x != 1.0
            || slot.texture.repeat.y != 1.0
            || slot.texture.rotation != 0.0;
        let st_source = if transform_needed {
            let node = format!("{}Transform", slot.name);
            nodes.push(shader(
                &node,
                "UsdTransform2d",
                vec![
                    connect("float2", "inputs:in", format!("{path}/stReader.outputs:result")),
                    attribute(
                        "float2",
                        "inputs:scale",
                        tuple2([slot.texture.repeat.x, slot.texture.repeat.y]),
                    ),
                    attribute(
                        "float2",
                        "inputs:translation",
                        tuple2([slot.texture.offset.x, slot.texture.offset.y]),
                    ),
                    attribute(
                        "float",
                        "inputs:rotation",
                        UsdValue::Float(slot.texture.rotation.to_degrees() as f64),
                    ),
                    attribute("float2", "outputs:result", UsdValue::None),
                ],
            ));
            format!("{path}/{node}.outputs:result")
        } else {
            format!("{path}/stReader.outputs:result")
        };
        reader_used = true;

        let mut properties = vec![
            attribute("asset", "inputs:file", UsdValue::Asset(file)),
            connect("float2", "inputs:st", st_source),
            token("inputs:wrapS", wrap_name(slot.texture.wrap_s)),
            token("inputs:wrapT", wrap_name(slot.texture.wrap_t)),
            token("inputs:sourceColorSpace", slot.color_space),
        ];
        if slot.scale != [1.0; 4] || slot.bias != [0.0; 4] {
            properties.push(attribute("float4", "inputs:scale", tuple4(slot.scale)));
            properties.push(attribute("float4", "inputs:bias", tuple4(slot.bias)));
        }
        let out_type = if slot.channel == "rgb" { "float3" } else { "float" };
        properties.push(attribute(
            out_type,
            &format!("outputs:{}", slot.channel),
            UsdValue::None,
        ));
        nodes.push(shader(slot.name, "UsdUVTexture", properties));

        // A connection is a *field* of one property, not a second property
        // beside it. Authoring both leaves two specs at the same path, which
        // the text form tolerates and the crate form refuses to open at all:
        // "ignoring invalid specs: spec <...inputs:diffuseColor> repeated".
        let surface = &mut prim.children[0].properties;
        surface.retain(|p| p.name != slot.input);
        surface.push(connect(
            slot.type_name,
            slot.input,
            format!("{path}/{}.outputs:{}", slot.name, slot.channel),
        ));
    }

    if reader_used {
        nodes.insert(
            0,
            shader(
                "stReader",
                "UsdPrimvarReader_float2",
                vec![
                    // A `token` here parses and then fails `usdchecker`: the
                    // shader definition says `string`, and the two are not
                    // interchangeable to a validator.
                    attribute("string", "inputs:varname", UsdValue::String("st".into())),
                    attribute("float2", "outputs:result", UsdValue::None),
                ],
            ),
        );
    }
    prim.children.extend(nodes);
    (prim, images)
}

/// A `UsdPreviewSurface` material prim.
pub fn material_prim(material: &Material, name: &str) -> UsdPrim {
    let Preview {
        color,
        roughness,
        metallic,
        emissive,
        opacity,
        specular,
        ior,
        clearcoat,
        opacity_threshold,
    } = preview_surface(material);
    let name = sanitize(name);
    let shader = UsdPrim {
        specifier: Specifier::Def,
        type_name: "Shader".into(),
        name: "Surface".into(),
        metadata: Vec::new(),
        properties: vec![
            UsdProperty {
                qualifier: String::new(),
                name: "info:id".into(),
                type_name: "token".into(),
                uniform: true,
                relationship: false,
                value: UsdValue::Token("UsdPreviewSurface".into()),
                metadata: Vec::new(),
            },
            attribute("color3f", "inputs:diffuseColor", tuple3(color)),
            attribute("color3f", "inputs:emissiveColor", tuple3(emissive)),
            attribute("float", "inputs:metallic", UsdValue::Float(metallic as f64)),
            attribute("float", "inputs:roughness", UsdValue::Float(roughness as f64)),
            attribute("float", "inputs:opacity", UsdValue::Float(opacity as f64)),
            attribute(
                "token",
                "outputs:surface",
                UsdValue::None,
            ),
        ]
        .into_iter()
        .chain(specular.into_iter().flat_map(|s| {
            [
                attribute("int", "inputs:useSpecularWorkflow", UsdValue::Int(1)),
                attribute("color3f", "inputs:specularColor", tuple3(s)),
            ]
        }))
        .chain(
            ior.into_iter()
                .map(|v| attribute("float", "inputs:ior", UsdValue::Float(v as f64))),
        )
        .chain((opacity_threshold > 0.0).then(|| {
            attribute(
                "float",
                "inputs:opacityThreshold",
                UsdValue::Float(opacity_threshold as f64),
            )
        }))
        .chain(clearcoat.into_iter().flat_map(|(amount, rough)| {
            [
                attribute("float", "inputs:clearcoat", UsdValue::Float(amount as f64)),
                attribute(
                    "float",
                    "inputs:clearcoatRoughness",
                    UsdValue::Float(rough as f64),
                ),
            ]
        }))
        .collect(),
        children: Vec::new(),
        variant_sets: Vec::new(),
    };
    UsdPrim {
        specifier: Specifier::Def,
        type_name: "Material".into(),
        name: name.clone(),
        metadata: Vec::new(),
        properties: vec![UsdProperty {
            qualifier: String::new(),
            // Named without the suffix: the writer adds `.connect` because
            // the value is a path, and spelling it here too produced
            // `outputs:surface.connect.connect`, which USD will not parse.
            name: "outputs:surface".into(),
            type_name: "token".into(),
            uniform: false,
            relationship: false,
            value: UsdValue::Path(format!("/Root/{name}/Surface.outputs:surface")),
            metadata: Vec::new(),
        }],
        children: vec![shader],
        variant_sets: Vec::new(),
    }
}

fn tuple4(v: [f32; 4]) -> UsdValue {
    UsdValue::Tuple(v.iter().map(|n| UsdValue::Float(*n as f64)).collect())
}

fn tuple3(v: [f32; 3]) -> UsdValue {
    UsdValue::Tuple(v.iter().map(|c| UsdValue::Float(*c as f64)).collect())
}

/// A prim name USD will accept: identifiers only, and never empty.
///
/// USD prim names are C identifiers. A mesh called `wheel.001` — which is what
/// every DCC tool produces — is not one, and a reader rejects the whole layer
/// rather than the one prim, so this is not a detail worth leaving to chance.
pub fn sanitize(name: &str) -> String {
    let mut out = String::new();
    for (i, c) in name.chars().enumerate() {
        let ok = c.is_ascii_alphanumeric() || c == '_';
        let leading_digit = i == 0 && c.is_ascii_digit();
        out.push(if ok && !leading_digit { c } else { '_' });
    }
    if out.is_empty() {
        out.push_str("Prim");
    }
    out
}

/// One geometry as a complete layer.
pub fn geometry_to_layer(g: &BufferGeometry, name: &str, options: &UsdExportOptions) -> UsdLayer {
    let root = UsdPrim {
        specifier: Specifier::Def,
        type_name: "Xform".into(),
        name: sanitize(&options.default_prim),
        metadata: Vec::new(),
        properties: Vec::new(),
        children: vec![mesh_prim(g, name)],
        variant_sets: Vec::new(),
    };
    UsdLayer {
        metadata: vec![
            (
                "defaultPrim".into(),
                UsdValue::String(sanitize(&options.default_prim)),
            ),
            (
                "metersPerUnit".into(),
                UsdValue::Float(options.meters_per_unit as f64),
            ),
            ("upAxis".into(), UsdValue::String(options.up_axis.to_string())),
        ],
        prims: vec![root],
    }
}

/// One geometry as a `.usda` document.
pub fn geometry_to_usda(g: &BufferGeometry, name: &str) -> String {
    layer_to_usda(&geometry_to_layer(g, name, &UsdExportOptions::default()))
}

/// A scene graph as a layer, hierarchy and materials included.
pub fn scene_to_layer(
    arena: &ObjectArena,
    roots: &[ObjectId],
    options: &UsdExportOptions,
) -> UsdLayer {
    scene_to_layer_placed(arena, roots, options, &mut Vec::new()).0
}

/// A scene graph as a layer, with the image files its materials refer to.
///
/// The layer names its textures by relative path, so writing it without these
/// beside it leaves every map dangling. [`scene_to_usdz`](super::scene_to_usdz)
/// packs them for you; this is for the text and crate forms, which have
/// nowhere to put them.
pub fn scene_to_layer_with_textures(
    arena: &ObjectArena,
    roots: &[ObjectId],
    options: &UsdExportOptions,
) -> (UsdLayer, Vec<ExportedTexture>) {
    scene_to_layer_placed(arena, roots, options, &mut Vec::new())
}

/// Materials and their images as an export accumulates them.
#[derive(Default)]
struct Materials {
    prims: Vec<UsdPrim>,
    textures: Vec<ExportedTexture>,
}

fn scene_to_layer_placed(
    arena: &ObjectArena,
    roots: &[ObjectId],
    options: &UsdExportOptions,
    placed: &mut Vec<(ObjectId, String)>,
) -> (UsdLayer, Vec<ExportedTexture>) {
    let root_path = format!("/{}", sanitize(&options.default_prim));
    let mut materials = Materials::default();
    let mut children: Vec<UsdPrim> = Vec::new();
    for (i, root) in roots.iter().enumerate() {
        if let Some(prim) = object_prim_at(arena, *root, i, &mut materials, &root_path, placed) {
            children.push(prim);
        }
    }
    let textures = std::mem::take(&mut materials.textures);
    children.extend(materials.prims);
    let root = UsdPrim {
        specifier: Specifier::Def,
        type_name: "Xform".into(),
        name: sanitize(&options.default_prim),
        metadata: Vec::new(),
        properties: Vec::new(),
        children,
        variant_sets: Vec::new(),
    };
    let layer = UsdLayer {
        metadata: vec![
            (
                "defaultPrim".into(),
                UsdValue::String(sanitize(&options.default_prim)),
            ),
            (
                "metersPerUnit".into(),
                UsdValue::Float(options.meters_per_unit as f64),
            ),
            ("upAxis".into(), UsdValue::String(options.up_axis.to_string())),
        ],
        prims: vec![root],
    };
    (layer, textures)
}

/// A scene graph as a `.usda` document.
pub fn scene_to_usda(arena: &ObjectArena, roots: &[ObjectId]) -> String {
    layer_to_usda(&scene_to_layer(arena, roots, &UsdExportOptions::default()))
}

/// The same, recording where each object ended up so that a later pass can
/// attach animation to it by path.
fn object_prim_at(
    arena: &ObjectArena,
    id: ObjectId,
    ordinal: usize,
    materials: &mut Materials,
    parent: &str,
    placed: &mut Vec<(ObjectId, String)>,
) -> Option<UsdPrim> {
    let object = arena.get(id)?;
    let name = if object.name.is_empty() {
        format!("Node_{ordinal}")
    } else {
        object.name.clone()
    };

    let mut prim = match &object.kind {
        ObjectKind::Mesh(mesh) => {
            let mut prim = mesh_prim(&mesh.geometry, &name);
            let material_name = format!("Material_{}", materials.prims.len());
            let root = parent.split('/').take(2).collect::<Vec<_>>().join("/");
            let (material, images) =
                material_prim_with_textures(&mesh.material, &material_name, &root);
            materials.prims.push(material);
            materials.textures.extend(images);
            // Sidedness is a property of the *surface* in USD, not of the
            // material bound to it, so it goes on the mesh. USD's default is
            // single-sided, so only a double-sided material says anything.
            if mesh.material.side() == 2 {
                prim.properties
                    .push(uniform(attribute("bool", "doubleSided", UsdValue::Bool(true))));
            }
            // With no colour per vertex, the material's own colour stands in as
            // a constant, so the mesh is the right colour even to a reader that
            // never looks at the shader.
            if !prim
                .properties
                .iter()
                .any(|p| p.name == "primvars:displayColor")
            {
                let c = mesh.material.color();
                prim.properties.push(interpolated(
                    attribute(
                        "color3f[]",
                        "primvars:displayColor",
                        UsdValue::Array(vec![tuple3([c.r, c.g, c.b])]),
                    ),
                    "constant",
                ));
            }
            let opacity = mesh.material.opacity();
            if opacity < 1.0 {
                prim.properties.push(interpolated(
                    attribute(
                        "float[]",
                        "primvars:displayOpacity",
                        UsdValue::Array(vec![UsdValue::Float(opacity as f64)]),
                    ),
                    "constant",
                ));
            }
            prim.properties.push(UsdProperty {
                qualifier: String::new(),
                name: "material:binding".into(),
                type_name: String::new(),
                uniform: false,
                relationship: true,
                value: UsdValue::Path(format!("/Root/{}", sanitize(&material_name))),
                metadata: Vec::new(),
            });
            // A binding without the schema that defines it is what `usdchecker`
            // objects to, and some readers ignore the binding outright.
            prim.metadata.push((
                "prepend apiSchemas".into(),
                UsdValue::Array(vec![UsdValue::Token("MaterialBindingAPI".into())]),
            ));
            prim
        }
        _ => UsdPrim {
            specifier: Specifier::Def,
            type_name: "Xform".into(),
            name: sanitize(&name),
            metadata: Vec::new(),
            properties: Vec::new(),
            children: Vec::new(),
            variant_sets: Vec::new(),
        },
    };

    // Transform, written as the three ops USD reads in this order.
    let p = object.position;
    let q = object.quaternion;
    let s = object.scale;
    let mut order = Vec::new();
    if p.length_sq() > 0.0 {
        prim.properties.push(attribute(
            "double3",
            "xformOp:translate",
            tuple3([p.x, p.y, p.z]),
        ));
        order.push(UsdValue::Token("xformOp:translate".into()));
    }
    if (q.w - 1.0).abs() > 1e-9 || q.x != 0.0 || q.y != 0.0 || q.z != 0.0 {
        // USD writes the real part first.
        prim.properties.push(attribute(
            "quatf",
            "xformOp:orient",
            UsdValue::Tuple(
                [q.w, q.x, q.y, q.z]
                    .iter()
                    .map(|v| UsdValue::Float(*v as f64))
                    .collect(),
            ),
        ));
        order.push(UsdValue::Token("xformOp:orient".into()));
    }
    if (s.x - 1.0).abs() > 1e-9 || (s.y - 1.0).abs() > 1e-9 || (s.z - 1.0).abs() > 1e-9 {
        prim.properties.push(attribute(
            "float3",
            "xformOp:scale",
            tuple3([s.x, s.y, s.z]),
        ));
        order.push(UsdValue::Token("xformOp:scale".into()));
    }
    if !order.is_empty() {
        prim.properties.push(UsdProperty {
            qualifier: String::new(),
            name: "xformOpOrder".into(),
            type_name: "token[]".into(),
            uniform: true,
            relationship: false,
            value: UsdValue::Array(order),
            metadata: Vec::new(),
        });
    }
    if !object.visible {
        prim.properties.push(UsdProperty {
            qualifier: String::new(),
            name: "visibility".into(),
            type_name: "token".into(),
            uniform: false,
            relationship: false,
            value: UsdValue::Token("invisible".into()),
            metadata: Vec::new(),
        });
    }

    let path = format!("{parent}/{}", prim.name);
    placed.push((id, path.clone()));
    for (i, child) in object.children.iter().enumerate() {
        if let Some(c) = object_prim_at(arena, *child, i, materials, &path, placed) {
            prim.children.push(c);
        }
    }
    Some(prim)
}

#[cfg(test)]
mod tests {

    /// A connection is written once, not twice.
    ///
    /// The writer appends `.connect` because the value is a path, so the
    /// property must *not* be named with the suffix as well — doing both
    /// produced `outputs:surface.connect.connect`, which USD refuses to parse
    /// at all. The round-trip tests missed it because both sides of this crate
    /// agreed with each other; OpenUSD did not.
    #[test]
    fn a_connection_is_not_spelled_twice() {
        use crate::core::{Mesh, Object3D, ObjectArena};
        use crate::geometries::BoxGeometry;
        use crate::materials::{Material, StandardMaterial};

        let mut arena = ObjectArena::new();
        let id = arena.insert(Object3D::mesh(Mesh::new(
            BoxGeometry::new(1.0, 1.0, 1.0),
            Material::Standard(StandardMaterial::default()),
        )));
        let text = scene_to_usda(&arena, &[id]);

        assert!(
            !text.contains(".connect.connect"),
            "the suffix was written twice:\n{text}"
        );
        assert!(text.contains("outputs:surface.connect = <"), "{text}");

        // And it reads back as the attribute, not as a property with a dot in
        // its name — which is not a name USD can hold.
        let layer = super::super::parse::parse(&text).expect("what we wrote, USD can read");
        let material = layer.prim_at("/Root/Material_0").expect("the material");
        assert!(material.property("outputs:surface.connect").is_none());
        let surface = material.property("outputs:surface").expect("the connection");
        assert!(matches!(surface.value, UsdValue::Path(_)), "{:?}", surface.value);
    }

    /// Animation written out and read back is the animation that went in.
    #[test]
    fn a_clip_round_trips_through_a_document() {
        use crate::animation::{AnimationClip, KeyframeTrack, TrackTarget};
        use crate::core::Object3D;
        use crate::math::{Quaternion, Vector3};

        let mut arena = ObjectArena::new();
        let mut object = Object3D::group();
        object.name = "Mover".into();
        let id = arena.insert(object);

        let clip = AnimationClip::new(
            "spin",
            2.0,
            vec![
                KeyframeTrack::vector(
                    id,
                    TrackTarget::Position,
                    vec![0.0, 1.0, 2.0],
                    vec![
                        Vector3::new(0.0, 0.0, 0.0),
                        Vector3::new(10.0, 0.0, 0.0),
                        Vector3::new(10.0, 10.0, 0.0),
                    ],
                ),
                KeyframeTrack::quaternion(
                    id,
                    TrackTarget::Quaternion,
                    vec![0.0, 2.0],
                    vec![
                        Quaternion::identity(),
                        Quaternion::from_axis_angle(Vector3::new(0.0, 0.0, 1.0), 1.0),
                    ],
                ),
            ],
        );

        let text = animated_scene_to_usda(&arena, &[id], &[clip]);
        let layer = super::super::parse::parse(&text).expect("what we wrote, we can read");

        // The layer says how fast it runs and how long it is.
        assert_eq!(layer.time_codes_per_second(), 24.0);
        assert_eq!(layer.time_range(), Some((0.0, 48.0)));

        let mover = layer.prim_at("/Root/Mover").expect("the animated prim");
        let translate = mover.value("xformOp:translate").unwrap();
        let samples = translate.samples().expect("animated");
        assert_eq!(samples.len(), 3);
        // Seconds became time codes.
        assert_eq!(samples[1].0, 24.0);
        assert_eq!(samples[2].1.flat_f32(), vec![10.0, 10.0, 0.0]);

        // An op that only exists because it is animated still has to be in the
        // order, or USD ignores it.
        let order = mover.value("xformOpOrder").unwrap().flat_tokens();
        assert!(order.contains(&"xformOp:translate"), "{order:?}");
        assert!(order.contains(&"xformOp:orient"), "{order:?}");
    }

    /// Out through the exporter and back in through the scene builder: the
    /// clip that comes back drives the same object the same way.
    #[test]
    fn a_clip_survives_the_whole_round_trip() {
        use crate::animation::{AnimationClip, KeyframeTrack, TrackTarget, TrackValues};
        use crate::core::Object3D;
        use crate::math::Vector3;

        let mut arena = ObjectArena::new();
        let mut object = Object3D::group();
        object.name = "Mover".into();
        let id = arena.insert(object);
        let clip = AnimationClip::new(
            "move",
            2.0,
            vec![KeyframeTrack::vector(
                id,
                TrackTarget::Position,
                vec![0.0, 2.0],
                vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(4.0, 0.0, 0.0)],
            )],
        );

        let text = animated_scene_to_usda(&arena, &[id], &[clip]);
        let layer = super::super::parse::parse(&text).unwrap();
        let scene = super::super::scene::to_scene(&layer);

        let back = scene.animations.first().expect("a clip came back");
        assert_eq!(back.duration, 2.0);
        let track = back
            .tracks
            .iter()
            .find(|t| matches!(t.target, TrackTarget::Position))
            .expect("a position track");
        assert_eq!(track.times, vec![0.0, 2.0]);
        let TrackValues::Vector(values) = &track.values else {
            panic!("expected vectors");
        };
        assert_eq!(values[1], Vector3::new(4.0, 0.0, 0.0));
    }

    /// A scene with no animation is written exactly as it was before, with no
    /// time metadata invented for it.
    #[test]
    fn a_static_scene_gains_no_timing() {
        use crate::core::Object3D;
        let mut arena = ObjectArena::new();
        let id = arena.insert(Object3D::group());
        let text = animated_scene_to_usda(&arena, &[id], &[]);
        assert!(!text.contains("timeCodesPerSecond"), "{text}");
        assert!(!text.contains("timeSamples"), "{text}");
    }
    use super::super::parse::parse;
    use super::super::scene::{mesh_geometry, to_scene};
    use super::*;
    use crate::core::{BufferAttribute, Mesh, Object3D};
    use crate::materials::StandardMaterial;
    use crate::math::{Color, Vector3};

    fn triangle() -> BufferGeometry {
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            BufferAttribute::new(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 3),
        );
        g.set_attribute("uv", BufferAttribute::new(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0], 2));
        g.set_index(vec![0, 1, 2]);
        g
    }

    #[test]
    fn a_geometry_round_trips_through_usda() {
        let text = geometry_to_usda(&triangle(), "Tri");
        let layer = parse(&text).unwrap();
        let prim = layer.prim_at("/Root/Tri").unwrap();
        assert_eq!(prim.type_name, "Mesh");
        let back = mesh_geometry(prim).unwrap();
        assert_eq!(
            back.get_attribute("position").unwrap().array,
            triangle().get_attribute("position").unwrap().array
        );
        assert_eq!(back.index.clone().unwrap(), vec![0, 1, 2]);
        assert_eq!(back.get_attribute("uv").unwrap().array.len(), 6);
    }

    #[test]
    fn the_extent_is_written_and_is_the_bounding_box() {
        let layer = parse(&geometry_to_usda(&triangle(), "Tri")).unwrap();
        let extent = layer
            .prim_at("/Root/Tri")
            .unwrap()
            .value("extent")
            .unwrap()
            .flat_f32();
        assert_eq!(extent, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn names_that_are_not_identifiers_are_made_into_them() {
        assert_eq!(sanitize("wheel.001"), "wheel_001");
        assert_eq!(sanitize("2fast"), "_fast");
        assert_eq!(sanitize(""), "Prim");
        // And the result parses, which is the point of doing it.
        let text = geometry_to_usda(&triangle(), "wheel.001");
        assert!(parse(&text).unwrap().prim_at("/Root/wheel_001").is_some());
    }

    #[test]
    fn a_scene_keeps_its_hierarchy_and_transforms() {
        let mut arena = ObjectArena::new();
        let mut parent = Object3D::group();
        parent.name = "Parent".into();
        parent.position = Vector3::new(1.0, 2.0, 3.0);
        let parent_id = arena.insert(parent);

        let mut child = Object3D::mesh(Mesh::new(
            triangle(),
            StandardMaterial::new(Color::new(1.0, 0.0, 0.0)).into(),
        ));
        child.name = "Child".into();
        child.scale = Vector3::new(2.0, 2.0, 2.0);
        let child_id = arena.insert(child);
        arena.add_child(parent_id, child_id);

        let text = scene_to_usda(&arena, &[parent_id]);
        let layer = parse(&text).unwrap();
        let scene = to_scene(&layer);

        let root = scene.arena.get(scene.roots[0]).unwrap();
        // Root → Parent → Child, plus the material beside Parent.
        let parent = scene.arena.get(root.children[0]).unwrap();
        assert_eq!(parent.name, "Parent");
        assert!((parent.position.y - 2.0).abs() < 1e-6);
        let child = scene.arena.get(parent.children[0]).unwrap();
        assert_eq!(child.name, "Child");
        assert!((child.scale.x - 2.0).abs() < 1e-6);
    }

    /// Every material kind that has a colour exports with that colour.
    ///
    /// This existed as a grey `_` arm for a while: Lambert, Phong, Toon,
    /// Matcap, Sprite, Points, Line and Mirror all left as
    /// `(0.8, 0.8, 0.8)`, colour and opacity dropped. Nothing caught it,
    /// because the reader read back exactly the grey the writer wrote and the
    /// round-trip test agreed with itself. `usdcat` on an exported gallery is
    /// what showed it.
    #[test]
    fn every_coloured_material_exports_its_colour() {
        use crate::materials::{
            BasicMaterial, LambertMaterial, LineBasicMaterial, MatcapMaterial, MirrorMaterial,
            PhongMaterial, PhysicalMaterial, PointsMaterial, SpriteMaterial, ToonMaterial,
        };

        let red = Color::new(0.9, 0.2, 0.1);
        let kinds: Vec<(&str, Material)> = vec![
            (
                "lambert",
                LambertMaterial { color: red, ..Default::default() }.into(),
            ),
            (
                "phong",
                PhongMaterial { color: red, ..Default::default() }.into(),
            ),
            (
                "toon",
                ToonMaterial { color: red, ..Default::default() }.into(),
            ),
            (
                "matcap",
                MatcapMaterial { color: red, ..Default::default() }.into(),
            ),
            (
                "sprite",
                SpriteMaterial { color: red, ..Default::default() }.into(),
            ),
            (
                "points",
                PointsMaterial { color: red, ..Default::default() }.into(),
            ),
            (
                "line",
                LineBasicMaterial { color: red, ..Default::default() }.into(),
            ),
            (
                "mirror",
                MirrorMaterial { color: red, ..Default::default() }.into(),
            ),
            (
                "physical",
                PhysicalMaterial { color: red, ..Default::default() }.into(),
            ),
            ("basic", BasicMaterial::new(red).into()),
            ("standard", StandardMaterial::new(red).into()),
        ];
        for (label, material) in kinds {
            let prim = material_prim(&material, "M");
            let surface = &prim.children[0];
            let colour = surface
                .value("inputs:diffuseColor")
                .expect("a diffuse colour")
                .flat_f32();
            assert_eq!(
                colour,
                vec![0.9, 0.2, 0.1],
                "{label} lost its colour on the way out"
            );
        }
    }

    fn checker_texture() -> std::sync::Arc<crate::textures::Texture> {
        use crate::textures::{Texture, TextureFormat, TextureWrap};
        let mut t = Texture::new(
            2,
            2,
            TextureFormat::Rgba8UnormSrgb,
            vec![255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 255],
        );
        t.wrap_s = TextureWrap::Repeat;
        t.wrap_t = TextureWrap::MirroredRepeat;
        std::sync::Arc::new(t)
    }

    fn child<'a>(prim: &'a UsdPrim, name: &str) -> &'a UsdPrim {
        prim.children
            .iter()
            .find(|c| c.name == name)
            .unwrap_or_else(|| panic!("no child {name}"))
    }

    /// A map exports as the shader network USD reads textures through.
    ///
    /// A `UsdPreviewSurface` cannot name a file itself: the image is a
    /// `UsdUVTexture`, the UVs come from a `UsdPrimvarReader_float2`, and the
    /// surface input is *connected* to the texture's output rather than
    /// holding a value. Writing the file path onto the surface would parse and
    /// mean nothing.
    #[test]
    fn a_map_exports_as_a_texture_network() {
        let mut m = StandardMaterial::new(Color::new(1.0, 1.0, 1.0));
        m.map = Some(checker_texture());
        let (prim, images) = material_prim_with_textures(&m.into(), "M", "/Root");

        assert_eq!(images.len(), 1);
        assert_eq!(images[0].path, "textures/M_DiffuseTex.png");
        assert!(images[0].data.starts_with(b"\x89PNG"), "a PNG was encoded");

        let texture = child(&prim, "DiffuseTex");
        assert_eq!(
            texture.value("info:id").unwrap().as_str(),
            Some("UsdUVTexture")
        );
        assert_eq!(
            texture.value("inputs:file").unwrap().as_str(),
            Some("textures/M_DiffuseTex.png")
        );
        assert_eq!(texture.value("inputs:wrapS").unwrap().as_str(), Some("repeat"));
        assert_eq!(texture.value("inputs:wrapT").unwrap().as_str(), Some("mirror"));
        assert_eq!(
            texture.value("inputs:sourceColorSpace").unwrap().as_str(),
            Some("sRGB")
        );

        let reader = child(&prim, "stReader");
        assert_eq!(
            reader.value("info:id").unwrap().as_str(),
            Some("UsdPrimvarReader_float2")
        );

        // The surface takes the connection, and the constant it replaces is
        // gone: two properties at one path is a spec USD refuses to open.
        let surface = child(&prim, "Surface");
        let diffuse: Vec<&UsdProperty> = surface
            .properties
            .iter()
            .filter(|p| p.name == "inputs:diffuseColor")
            .collect();
        assert_eq!(diffuse.len(), 1, "one property, not a value and a connection");
        assert_eq!(
            diffuse[0].value.as_str(),
            Some("/Root/M/DiffuseTex.outputs:rgb")
        );
    }

    /// A normal map carries the scale and bias that make it a normal map.
    ///
    /// Eight-bit pixels hold `[0,1]` and a normal is `[-1,1]`. Without
    /// `scale = (2,2,2,1)` and `bias = (-1,-1,-1,0)` every normal points into
    /// the surface, which reads as bad lighting rather than as a broken file.
    /// It is also `raw`, not `sRGB`.
    #[test]
    fn a_normal_map_carries_its_scale_and_bias() {
        let mut m = StandardMaterial::new(Color::new(1.0, 1.0, 1.0));
        m.normal_map = Some(checker_texture());
        let (prim, _) = material_prim_with_textures(&m.into(), "M", "/Root");
        let texture = child(&prim, "NormalTex");
        assert_eq!(
            texture.value("inputs:scale").unwrap().flat_f32(),
            vec![2.0, 2.0, 2.0, 1.0]
        );
        assert_eq!(
            texture.value("inputs:bias").unwrap().flat_f32(),
            vec![-1.0, -1.0, -1.0, 0.0]
        );
        assert_eq!(
            texture.value("inputs:sourceColorSpace").unwrap().as_str(),
            Some("raw")
        );
        // And it drives the surface's normal, not its colour.
        let surface = child(&prim, "Surface");
        assert!(surface.properties.iter().any(|p| p.name == "inputs:normal"));
    }

    /// A UV transform becomes a `UsdTransform2d`, in degrees.
    ///
    /// This crate keeps rotation in radians and USD's node takes degrees, so
    /// passing the number through unconverted turns a quarter turn into about
    /// one and a half degrees.
    #[test]
    fn a_uv_transform_becomes_a_transform_node_in_degrees() {
        use crate::math::Vector2;
        let mut texture = (*checker_texture()).clone();
        texture.repeat = Vector2::new(2.0, 3.0);
        texture.rotation = std::f32::consts::FRAC_PI_2;
        let mut m = StandardMaterial::new(Color::new(1.0, 1.0, 1.0));
        m.map = Some(std::sync::Arc::new(texture));
        let (prim, _) = material_prim_with_textures(&m.into(), "M", "/Root");

        let node = child(&prim, "DiffuseTexTransform");
        assert_eq!(
            node.value("info:id").unwrap().as_str(),
            Some("UsdTransform2d")
        );
        assert_eq!(node.value("inputs:scale").unwrap().flat_f32(), vec![2.0, 3.0]);
        let rotation = node.value("inputs:rotation").unwrap().flat_f32()[0];
        assert!((rotation - 90.0).abs() < 1e-3, "{rotation} should be degrees");

        // And the texture samples the transform rather than the reader.
        assert_eq!(
            child(&prim, "DiffuseTex").value("inputs:st").unwrap().as_str(),
            Some("/Root/M/DiffuseTexTransform.outputs:result")
        );
    }

    /// A texture with no readable pixels leaves the constant value standing.
    ///
    /// A material whose bytes are on the GPU has nothing to write; dropping
    /// the map is right, and dropping the colour with it is not.
    #[test]
    fn a_texture_with_no_pixels_is_skipped_not_fatal() {
        use crate::textures::{Texture, TextureFormat};
        let mut m = StandardMaterial::new(Color::new(0.25, 0.5, 0.75));
        m.map = Some(std::sync::Arc::new(Texture::new(
            4,
            4,
            TextureFormat::Rgba8UnormSrgb,
            Vec::new(),
        )));
        let (prim, images) = material_prim_with_textures(&m.into(), "M", "/Root");
        assert!(images.is_empty());
        assert_eq!(
            child(&prim, "Surface")
                .value("inputs:diffuseColor")
                .unwrap()
                .flat_f32(),
            vec![0.25, 0.5, 0.75]
        );
    }

    /// A `.usdz` carries the images its layer names.
    ///
    /// The package format exists to be self-contained; one that refers to a
    /// `textures/` directory that travelled separately is a valid archive and
    /// a broken asset.
    #[test]
    fn a_usdz_packs_the_textures_its_materials_name() {
        use crate::core::{Mesh, Object3D, ObjectArena};
        let mut m = StandardMaterial::new(Color::new(1.0, 1.0, 1.0));
        m.map = Some(checker_texture());
        let mut arena = ObjectArena::new();
        let id = arena.insert(Object3D::mesh(Mesh::new(triangle(), m.into())));
        let bytes = super::super::scene_to_usdz(&arena, &[id], &[]);

        let archive = super::super::usdz::read(&bytes).expect("a readable archive");
        let names: Vec<&str> = archive.entries.iter().map(|e| e.name.as_str()).collect();
        assert!(
            names.contains(&"textures/Material_0_DiffuseTex.png"),
            "{names:?}"
        );
    }

    /// Only a double-sided material says anything about sidedness, and it
    /// says it on the mesh, where USD keeps it.
    #[test]
    fn double_sided_lands_on_the_mesh() {
        use crate::core::{Mesh, Object3D, ObjectArena};
        use crate::materials::LambertMaterial;
        for (side, expected) in [(0u32, false), (2, true)] {
            let m = LambertMaterial { side, ..Default::default() };
            let mut arena = ObjectArena::new();
            let id = arena.insert(Object3D::mesh(Mesh::new(triangle(), m.into())));
            let layer = scene_to_layer(&arena, &[id], &UsdExportOptions::default());
            let mesh = layer.prim_at("/Root/Node_0").expect("the mesh");
            assert_eq!(mesh.value("doubleSided").is_some(), expected, "side {side}");
        }
    }

    /// Index of refraction and clear coat come from a physical material and
    /// from nothing else, so a standard material does not restate USD's
    /// defaults back at it.
    #[test]
    fn ior_and_clearcoat_are_physical_only() {
        use crate::materials::PhysicalMaterial;
        let p = PhysicalMaterial {
            ior: 1.45,
            clearcoat: 0.8,
            clearcoat_roughness: 0.05,
            ..Default::default()
        };
        let surface = material_prim(&p.into(), "M").children[0].clone();
        assert_eq!(surface.value("inputs:ior").unwrap().flat_f32(), vec![1.45]);
        assert_eq!(
            surface.value("inputs:clearcoat").unwrap().flat_f32(),
            vec![0.8]
        );

        let plain = material_prim(&StandardMaterial::new(Color::new(1.0, 1.0, 1.0)).into(), "M");
        assert!(plain.children[0].value("inputs:ior").is_none());
        assert!(plain.children[0].value("inputs:clearcoat").is_none());
    }

    /// Vertex colours survive, as USD's fallback shading.
    ///
    /// `primvars:displayColor` is what every tool that does not evaluate a
    /// shader graph falls back to. Without it a vertex-coloured mesh exported
    /// from here arrived grey and the colours were simply gone — the material
    /// carries one colour and a mesh may have one per vertex.
    #[test]
    fn vertex_colours_export_as_display_colour() {
        use crate::core::BufferAttribute;
        let mut g = triangle();
        g.set_attribute(
            "color",
            BufferAttribute::new(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 3),
        );
        let layer = parse(&geometry_to_usda(&g, "M")).unwrap();
        let prim = layer.prim_at("/Root/M").unwrap();
        let property = prim.property("primvars:displayColor").expect("a colour");
        assert_eq!(
            property.value.flat_f32(),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(
            property.meta("interpolation").and_then(|v| v.as_str()),
            Some("vertex")
        );
    }

    /// With no colour per vertex, the material's colour stands in as a
    /// constant — so the mesh is right even to a reader that skips shading.
    #[test]
    fn a_material_colour_becomes_a_constant_display_colour() {
        use crate::core::{Mesh, Object3D, ObjectArena};
        let mut arena = ObjectArena::new();
        let id = arena.insert(Object3D::mesh(Mesh::new(
            triangle(),
            StandardMaterial::new(Color::new(0.25, 0.5, 0.75)).into(),
        )));
        let layer = parse(&scene_to_usda(&arena, &[id])).unwrap();
        let property = layer
            .prim_at("/Root/Node_0")
            .unwrap()
            .property("primvars:displayColor")
            .expect("a colour");
        assert_eq!(property.value.flat_f32(), vec![0.25, 0.5, 0.75]);
        assert_eq!(
            property.meta("interpolation").and_then(|v| v.as_str()),
            Some("constant")
        );
    }

    /// A constant display colour is not a vertex attribute.
    ///
    /// One value applies to the whole surface. Read as if it were per vertex,
    /// the first vertex gets the colour and every other one reads past the end
    /// of the array — which for a colour is black.
    #[test]
    fn a_constant_display_colour_does_not_become_vertex_colours() {
        use crate::loaders::usd::scene::to_scene;
        let mut arena = crate::core::ObjectArena::new();
        let id = arena.insert(crate::core::Object3D::mesh(crate::core::Mesh::new(
            triangle(),
            StandardMaterial::new(Color::new(0.25, 0.5, 0.75)).into(),
        )));
        let layer = parse(&scene_to_usda(&arena, &[id])).unwrap();
        let scene = to_scene(&layer);
        let root = scene.arena.get(scene.roots[0]).unwrap();
        let ObjectKind::Mesh(mesh) = &scene.arena.get(root.children[0]).unwrap().kind else {
            panic!("not a mesh");
        };
        assert!(
            mesh.geometry.get_attribute("color").is_none(),
            "a constant colour is the material's, not the vertices'"
        );
    }

    /// Vertex colours make the whole trip, out and back.
    #[test]
    fn vertex_colours_come_back() {
        use crate::core::BufferAttribute;
        use crate::loaders::usd::scene::to_scene;
        let mut g = triangle();
        let colours = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        g.set_attribute("color", BufferAttribute::new(colours.clone(), 3));
        let mut arena = crate::core::ObjectArena::new();
        let id = arena.insert(crate::core::Object3D::mesh(crate::core::Mesh::new(
            g,
            StandardMaterial::new(Color::new(1.0, 1.0, 1.0)).into(),
        )));
        let layer = parse(&scene_to_usda(&arena, &[id])).unwrap();
        let scene = to_scene(&layer);
        let root = scene.arena.get(scene.roots[0]).unwrap();
        let ObjectKind::Mesh(mesh) = &scene.arena.get(root.children[0]).unwrap().kind else {
            panic!("not a mesh");
        };
        assert_eq!(
            mesh.geometry.get_attribute("color").expect("colours").array,
            colours
        );
    }

    /// A cutout material carries its threshold, and a blended one does not.
    #[test]
    fn an_alpha_test_becomes_an_opacity_threshold() {
        use crate::materials::BasicMaterial;
        let cutout = BasicMaterial {
            alpha_test: 0.5,
            ..BasicMaterial::new(Color::new(1.0, 1.0, 1.0))
        };
        let surface = material_prim(&cutout.into(), "M").children[0].clone();
        assert_eq!(
            surface.value("inputs:opacityThreshold").unwrap().flat_f32(),
            vec![0.5]
        );
        let blended = material_prim(&BasicMaterial::new(Color::new(1.0, 1.0, 1.0)).into(), "M");
        assert!(blended.children[0]
            .value("inputs:opacityThreshold")
            .is_none());
    }

    /// All three forms carry the same document, and none of them loses the
    /// pictures.
    ///
    /// `scene_to_usdc` used to hand back bytes referring to `textures/*.png`
    /// and drop the images on the floor, so a crate written from a textured
    /// scene was dangling by construction. The package carries them; the other
    /// two report them so they can be written alongside.
    #[test]
    fn every_form_keeps_the_textures() {
        use super::super::{UsdExport, UsdLoader};
        use crate::core::{Mesh, Object3D, ObjectArena};

        let material = StandardMaterial {
            map: Some(checker_texture()),
            ..StandardMaterial::new(Color::new(1.0, 1.0, 1.0))
        };
        let mut arena = ObjectArena::new();
        let id = arena.insert(Object3D::mesh(Mesh::new(triangle(), material.into())));
        let out = UsdExport::scene(&arena, &[id]);

        assert_eq!(out.textures.len(), 1, "the export knows about its image");
        assert_eq!(out.textures[0].path, "textures/Material_0_DiffuseTex.png");

        // Each form is the same document.
        let from_text = super::super::parse::parse(&out.usda()).unwrap();
        let from_crate = super::super::usdc::read(&out.usdc()).unwrap();
        for layer in [&from_text, &from_crate] {
            let shader = layer
                .prim_at("/Root/Material_0/DiffuseTex")
                .expect("the texture shader");
            assert_eq!(
                shader.value("inputs:file").unwrap().as_str(),
                Some("textures/Material_0_DiffuseTex.png")
            );
        }

        // And the package needs nothing beside it.
        let scene = UsdLoader::parse(&out.usdz()).expect("reads back");
        let ObjectKind::Mesh(mesh) = &scene
            .arena
            .get(scene.arena.get(scene.roots[0]).unwrap().children[0])
            .unwrap()
            .kind
        else {
            panic!("not a mesh");
        };
        let Material::Standard(back) = &*mesh.material else {
            panic!("{:?}", mesh.material)
        };
        assert!(back.map.is_some(), "the package carried its image");
    }

    /// A `.usda` written to disk with its images beside it reads back whole.
    ///
    /// This is what the package form gets for free and the other two have to
    /// be told: the layer names `textures/foo.png` relative to itself, so the
    /// file has to be there and the reader has to be told where "there" is.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn a_layer_written_to_disk_reads_back_with_its_images() {
        use super::super::{attach_textures_from_dir, UsdExport, UsdLoader};
        use crate::core::{Mesh, Object3D, ObjectArena};

        let material = StandardMaterial {
            map: Some(checker_texture()),
            ..StandardMaterial::new(Color::new(1.0, 1.0, 1.0))
        };
        let mut arena = ObjectArena::new();
        let id = arena.insert(Object3D::mesh(Mesh::new(triangle(), material.into())));
        let out = UsdExport::scene(&arena, &[id]);

        let dir = std::env::temp_dir().join(format!("threers_usd_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        for name in ["disk.usda", "disk.usdc"] {
            let path = dir.join(name);
            out.write_to(&path, &dir).expect("writes");
            assert!(
                dir.join("textures/Material_0_DiffuseTex.png").exists(),
                "the image landed beside {name}"
            );

            let bytes = std::fs::read(&path).unwrap();
            let mut scene = UsdLoader::parse(&bytes).expect("reads");
            // Before being told where to look, it reports what it wants.
            assert_eq!(
                scene.textures.iter().flat_map(|(_, r)| r).count(),
                1,
                "{name} should ask for its image"
            );
            assert_eq!(attach_textures_from_dir(&mut scene, &dir), 1, "{name}");

            let ObjectKind::Mesh(mesh) = &scene
                .arena
                .get(scene.arena.get(scene.roots[0]).unwrap().children[0])
                .unwrap()
                .kind
            else {
                panic!("not a mesh");
            };
            let Material::Standard(back) = &*mesh.material else {
                panic!("{:?}", mesh.material)
            };
            let map = back.map.as_ref().unwrap_or_else(|| panic!("{name} lost its map"));
            assert_eq!(&map.data[..4], &[255, 0, 0, 255], "{name} pixels changed");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A path climbing out of the directory is not followed.
    ///
    /// The function was handed a directory, not the disk. A layer that names
    /// `../../etc/passwd` gets told no and keeps asking, rather than being
    /// quietly obliged.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn a_texture_path_does_not_escape_its_directory() {
        use super::super::{attach_textures_from_dir, UsdLoader};

        let dir = std::env::temp_dir().join(format!("threers_usd_esc_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        // A real PNG, reachable only by climbing out of `sub`.
        std::fs::write(
            dir.join("secret.png"),
            crate::utils::png::encode_png(1, 1, &[9, 9, 9, 255]),
        )
        .unwrap();

        let text = r#"#usda 1.0

def Mesh "M" (
    prepend apiSchemas = ["MaterialBindingAPI"]
)
{
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    rel material:binding = </Mat>
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
}

def Material "Mat"
{
    token outputs:surface.connect = </Mat/S.outputs:surface>

    def Shader "S"
    {
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor.connect = </Mat/T.outputs:rgb>
        token outputs:surface
    }

    def Shader "T"
    {
        uniform token info:id = "UsdUVTexture"
        asset inputs:file = @../secret.png@
        float3 outputs:rgb
    }
}
"#;
        let mut scene = UsdLoader::parse(text.as_bytes()).expect("reads");
        assert_eq!(
            attach_textures_from_dir(&mut scene, &dir.join("sub")),
            0,
            "a path climbing out was followed"
        );
        assert_eq!(
            scene.textures.iter().flat_map(|(_, r)| r).count(),
            1,
            "and it is still reported rather than dropped"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A `.usdz` can hold either form of layer, and carries images either way.
    #[test]
    fn a_package_holds_text_or_crate() {
        use super::super::{UsdExport, UsdzLayer};
        use crate::core::{Mesh, Object3D, ObjectArena};

        let material = StandardMaterial {
            map: Some(checker_texture()),
            ..StandardMaterial::new(Color::new(1.0, 1.0, 1.0))
        };
        let mut arena = ObjectArena::new();
        let id = arena.insert(Object3D::mesh(Mesh::new(triangle(), material.into())));
        let out = UsdExport::scene(&arena, &[id]);

        for (form, stem, magic) in [
            (UsdzLayer::Text, "scene.usda", &b"#usda"[..]),
            (UsdzLayer::Crate, "scene.usdc", &b"PXR-USDC"[..]),
        ] {
            let archive =
                super::super::usdz::read(&out.usdz_as(form, &[])).expect("a readable archive");
            let root = archive.entries.first().expect("a layer");
            assert_eq!(root.name, stem);
            assert!(root.data.starts_with(magic), "{stem} is the wrong form");
            assert!(
                archive
                    .entries
                    .iter()
                    .any(|e| e.name == "textures/Material_0_DiffuseTex.png"),
                "the image travelled with {stem}"
            );
        }
    }

    /// A bottom-up texture is turned over on the way out.
    ///
    /// `flip_y` is this crate's flag for "the top row is stored first", which
    /// is what an image file holds and what USD expects — Hydra puts a PNG's
    /// first row at `v = 1`. A `DataTexture` or a render target is authored
    /// bottom-up instead, and writing its rows verbatim into a PNG puts the
    /// picture upside down in every USD reader. Nothing in the format can
    /// carry the distinction, so the rows have to be right.
    #[test]
    fn a_bottom_up_texture_is_written_the_right_way_up() {
        use crate::textures::{Texture, TextureFormat};
        // Two rows, distinguishable: first row red, second blue.
        let pixels = vec![255, 0, 0, 255, 255, 0, 0, 255, 0, 0, 255, 255, 0, 0, 255, 255];
        let top_down = Texture::new(2, 2, TextureFormat::Rgba8UnormSrgb, pixels.clone());
        assert!(top_down.flip_y, "an image texture stores the top row first");

        let bottom_up = Texture {
            flip_y: false,
            ..Texture::new(2, 2, TextureFormat::Rgba8UnormSrgb, pixels)
        };

        let written = |texture: Texture| {
            let m = StandardMaterial {
                map: Some(std::sync::Arc::new(texture)),
                ..StandardMaterial::new(Color::new(1.0, 1.0, 1.0))
            };
            let (_, images) = material_prim_with_textures(&m.into(), "M", "/Root");
            crate::utils::png::decode_png(&images[0].data).unwrap().rgba
        };

        // Top-down goes out as it is: red first.
        assert_eq!(&written(top_down)[..4], &[255, 0, 0, 255]);
        // Bottom-up is reversed, so the row that was last is written first.
        assert_eq!(&written(bottom_up)[..4], &[0, 0, 255, 255]);
    }

    /// A `.usdz` comes back with its pictures, not with a list of them.
    ///
    /// The package format is required to be self-contained, so a reader that
    /// hands back "this material wants textures/foo.png" and stops is making
    /// the caller re-open an archive it already read. The pixels make the whole
    /// trip here: into RGBA, out as PNG, into the archive, and back.
    #[test]
    fn a_usdz_comes_back_with_its_textures_loaded() {
        use crate::core::{Mesh, Object3D, ObjectArena};
        use crate::textures::{TextureFormat, TextureWrap};

        let material = StandardMaterial {
            map: Some(checker_texture()),
            normal_map: Some(checker_texture()),
            ..StandardMaterial::new(Color::new(1.0, 1.0, 1.0))
        };
        let mut arena = ObjectArena::new();
        let mut object = Object3D::mesh(Mesh::new(triangle(), material.into()));
        object.name = "Quad".into();
        let id = arena.insert(object);

        let bytes = super::super::scene_to_usdz(&arena, &[id], &[]);
        let scene = super::super::UsdLoader::parse(&bytes).expect("reads back");

        let quad = scene
            .arena
            .get_objects_by_name(scene.roots[0], "Quad")
            .into_iter()
            .next()
            .expect("the quad");
        let ObjectKind::Mesh(mesh) = &scene.arena.get(quad).unwrap().kind else {
            panic!("not a mesh");
        };
        let Material::Standard(back) = &*mesh.material else {
            panic!("{:?}", mesh.material);
        };

        let map = back.map.as_ref().expect("the colour map came back");
        assert_eq!((map.width, map.height), (2, 2));
        // The exact pixels, not merely something of the right shape.
        assert_eq!(
            &map.data[..8],
            &[255, 0, 0, 255, 0, 255, 0, 255],
            "the pixels changed on the way"
        );
        // Wrap modes are authored per texture and have to survive too.
        assert_eq!(map.wrap_s, TextureWrap::Repeat);
        assert_eq!(map.wrap_t, TextureWrap::MirroredRepeat);
        assert_eq!(map.format, TextureFormat::Rgba8UnormSrgb);

        // A normal map is data, and reading it through an sRGB curve is the
        // classic way to get lighting that looks subtly wrong everywhere.
        let normal = back.normal_map.as_ref().expect("the normal map");
        assert_eq!(normal.format, TextureFormat::Rgba8Unorm);

        // And nothing is left for the caller to chase.
        assert!(
            scene.textures.iter().all(|(_, r)| r.is_empty()),
            "{:?}",
            scene.textures
        );
    }

    /// A `.usda` still reports what it cannot reach.
    ///
    /// The archive case is special because the bytes are in hand. A layer that
    /// names files beside itself has to keep asking, or a caller that *can*
    /// find them never learns they exist.
    #[test]
    fn a_usda_still_reports_the_textures_it_names() {
        use crate::core::{Mesh, Object3D, ObjectArena};
        use crate::loaders::usd::scene::to_scene;

        let material = StandardMaterial {
            map: Some(checker_texture()),
            ..StandardMaterial::new(Color::new(1.0, 1.0, 1.0))
        };
        let mut arena = ObjectArena::new();
        let id = arena.insert(Object3D::mesh(Mesh::new(triangle(), material.into())));
        let layer = parse(&scene_to_usda(&arena, &[id])).unwrap();
        let scene = to_scene(&layer);
        let asked: Vec<&str> = scene
            .textures
            .iter()
            .flat_map(|(_, r)| r.iter())
            .map(|r| r.file.as_str())
            .collect();
        assert_eq!(asked, ["textures/Material_0_DiffuseTex.png"]);
    }

    /// A physical material comes back physical, not flattened to standard.
    ///
    /// `ior` and `clearcoat` have no home on a standard material, so reading
    /// a file that authors them into one drops both. The importer builds a
    /// physical material when it sees either, which is the only way a physical
    /// material survives its own export.
    #[test]
    fn a_physical_material_survives_the_round_trip() {
        use crate::core::{Mesh, Object3D, ObjectArena};
        use crate::materials::PhysicalMaterial;
        use crate::loaders::usd::scene::to_scene;

        let source = PhysicalMaterial {
            color: Color::new(0.3, 0.7, 0.7),
            roughness: 0.1,
            ior: 1.45,
            clearcoat: 0.8,
            clearcoat_roughness: 0.05,
            side: 2,
            ..Default::default()
        };
        let mut arena = ObjectArena::new();
        let id = arena.insert(Object3D::mesh(Mesh::new(triangle(), source.into())));

        let layer = parse(&scene_to_usda(&arena, &[id])).unwrap();
        let scene = to_scene(&layer);
        let root = scene.arena.get(scene.roots[0]).unwrap();
        let mesh = scene.arena.get(root.children[0]).unwrap();
        let ObjectKind::Mesh(mesh) = &mesh.kind else {
            panic!("not a mesh");
        };
        let Material::Physical(back) = &*mesh.material else {
            panic!("came back as {:?}, not physical", mesh.material);
        };
        assert!((back.ior - 1.45).abs() < 1e-5, "{}", back.ior);
        assert!((back.clearcoat - 0.8).abs() < 1e-5);
        assert!((back.clearcoat_roughness - 0.05).abs() < 1e-5);
        assert!((back.roughness - 0.1).abs() < 1e-5);
        assert!((back.color.g - 0.7).abs() < 1e-5);
        assert_eq!(back.side, 2, "doubleSided did not come back");
    }

    /// A material with no physical inputs stays standard.
    ///
    /// The test above would pass just as well if every material came back
    /// physical, which would be its own kind of wrong.
    #[test]
    fn a_standard_material_stays_standard() {
        use crate::core::{Mesh, Object3D, ObjectArena};
        use crate::loaders::usd::scene::to_scene;

        let mut arena = ObjectArena::new();
        let id = arena.insert(Object3D::mesh(Mesh::new(
            triangle(),
            StandardMaterial::new(Color::new(0.2, 0.4, 0.6)).into(),
        )));
        let layer = parse(&scene_to_usda(&arena, &[id])).unwrap();
        let scene = to_scene(&layer);
        let root = scene.arena.get(scene.roots[0]).unwrap();
        let mesh = scene.arena.get(root.children[0]).unwrap();
        let ObjectKind::Mesh(mesh) = &mesh.kind else {
            panic!("not a mesh");
        };
        assert!(
            matches!(&*mesh.material, Material::Standard(_)),
            "{:?}",
            mesh.material
        );
    }

    /// Phong arrives as USD's specular workflow, exponent and all.
    ///
    /// Shininess is not roughness, and the two run opposite ways, so a fixed
    /// roughness would make every Phong material equally shiny. The default
    /// exponent of 30 is roughly a quarter rough; 200 is much smoother.
    #[test]
    fn phong_becomes_the_specular_workflow() {
        use crate::materials::PhongMaterial;
        let phong = PhongMaterial {
            color: Color::new(0.2, 0.3, 0.9),
            specular: Color::new(1.0, 1.0, 1.0),
            shininess: 30.0,
            ..Default::default()
        };

        let prim = material_prim(&phong.into(), "M");
        let surface = &prim.children[0];
        assert_eq!(
            surface.value("inputs:useSpecularWorkflow").unwrap().flat_u32(),
            vec![1]
        );
        assert_eq!(
            surface.value("inputs:specularColor").unwrap().flat_f32(),
            vec![1.0, 1.0, 1.0]
        );
        // sqrt(2 / 32) = 0.25
        let rough = surface.value("inputs:roughness").unwrap().flat_f32()[0];
        assert!((rough - 0.25).abs() < 1e-6, "{rough}");

        let smoother = material_prim(&PhongMaterial { shininess: 200.0, ..phong }.into(), "M")
            .children[0]
            .value("inputs:roughness")
            .unwrap()
            .flat_f32()[0];
        assert!(smoother < rough, "a higher exponent is a smoother surface");
    }

    /// A metallic workflow material says nothing about a specular one.
    #[test]
    fn only_phong_carries_the_specular_workflow() {
        let prim = material_prim(&StandardMaterial::new(Color::new(1.0, 1.0, 1.0)).into(), "M");
        assert!(prim.children[0]
            .value("inputs:useSpecularWorkflow")
            .is_none());
    }

    #[test]
    fn a_material_survives_the_round_trip() {
        let mut arena = ObjectArena::new();
        let mut m = StandardMaterial::new(Color::new(0.2, 0.4, 0.6));
        m.roughness = 0.3;
        m.metalness = 0.8;
        let id = arena.insert(Object3D::mesh(Mesh::new(triangle(), m.into())));
        let layer = parse(&scene_to_usda(&arena, &[id])).unwrap();
        let scene = to_scene(&layer);
        let root = scene.arena.get(scene.roots[0]).unwrap();
        let mesh = scene.arena.get(root.children[0]).unwrap();
        match &mesh.kind {
            ObjectKind::Mesh(mesh) => match &*mesh.material {
                Material::Standard(s) => {
                    assert!((s.color.b - 0.6).abs() < 1e-6);
                    assert!((s.roughness - 0.3).abs() < 1e-6);
                    assert!((s.metalness - 0.8).abs() < 1e-6);
                }
                other => panic!("{other:?}"),
            },
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn unindexed_geometry_gets_an_index_written_for_it() {
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            BufferAttribute::new(vec![0.0; 18], 3),
        );
        let layer = parse(&geometry_to_usda(&g, "M")).unwrap();
        let prim = layer.prim_at("/Root/M").unwrap();
        assert_eq!(prim.value("faceVertexCounts").unwrap().flat_u32(), vec![3, 3]);
        assert_eq!(
            prim.value("faceVertexIndices").unwrap().flat_u32(),
            vec![0, 1, 2, 3, 4, 5]
        );
    }
}

/// A scene graph with its animation, as a USD layer.
///
/// Every track becomes `timeSamples` on the transform op it drives, and the
/// layer gets the time metadata a player needs to know what the numbers mean.
/// Track times are seconds; USD counts in time codes, so they are multiplied
/// by the rate on the way out — which is the same conversion
/// [`to_scene`](super::scene::to_scene) undoes on the way in.
pub fn animated_scene_to_layer(
    arena: &ObjectArena,
    roots: &[ObjectId],
    clips: &[AnimationClip],
    options: &UsdExportOptions,
) -> UsdLayer {
    animated_scene_to_layer_with_textures(arena, roots, clips, options).0
}

/// A scene graph with its animation, and the image files its materials name.
pub fn animated_scene_to_layer_with_textures(
    arena: &ObjectArena,
    roots: &[ObjectId],
    clips: &[AnimationClip],
    options: &UsdExportOptions,
) -> (UsdLayer, Vec<ExportedTexture>) {
    let mut placed = Vec::new();
    let (mut layer, textures) = scene_to_layer_placed(arena, roots, options, &mut placed);
    let rate = options.time_codes_per_second.max(1.0) as f64;

    let mut last = 0.0f64;
    for clip in clips {
        for track in &clip.tracks {
            let Some((_, path)) = placed.iter().find(|(id, _)| *id == track.object) else {
                continue;
            };
            let Some(op) = sampled_track(track, rate) else {
                continue;
            };
            if let Some(time) = op.samples.last().map(|(t, _)| *t) {
                last = last.max(time);
            }
            let Some(prim) = prim_at_mut(&mut layer, path) else {
                continue;
            };
            set_op(
                prim,
                op.name,
                op.type_name,
                UsdValue::TimeSamples(op.samples),
            );
        }
        last = last.max((clip.duration as f64) * rate);
    }

    if last > 0.0 {
        layer.metadata.push((
            "timeCodesPerSecond".into(),
            UsdValue::Float(rate),
        ));
        layer.metadata.push(("startTimeCode".into(), UsdValue::Float(0.0)));
        layer.metadata.push(("endTimeCode".into(), UsdValue::Float(last)));
    }
    (layer, textures)
}

/// A scene graph with its animation, as a `.usda` document.
pub fn animated_scene_to_usda(
    arena: &ObjectArena,
    roots: &[ObjectId],
    clips: &[AnimationClip],
) -> String {
    layer_to_usda(&animated_scene_to_layer(
        arena,
        roots,
        clips,
        &UsdExportOptions::default(),
    ))
}

/// A track expressed as the USD transform op it drives.
struct SampledOp {
    /// The op's attribute name, e.g. `xformOp:translate`.
    name: &'static str,
    /// The type USD expects that op to be authored as.
    type_name: &'static str,
    /// Times in time codes, not seconds.
    samples: Vec<(f64, UsdValue)>,
}

/// One track as the USD op it drives, with its times in time codes.
fn sampled_track(track: &KeyframeTrack, rate: f64) -> Option<SampledOp> {
    let time = |i: usize| track.times.get(i).map(|t| *t as f64 * rate).unwrap_or(0.0);
    match (&track.target, &track.values) {
        (TrackTarget::Position, TrackValues::Vector(values)) => Some(SampledOp {
            name: "xformOp:translate",
            type_name: "double3",
            samples: values
                .iter()
                .enumerate()
                .map(|(i, v)| (time(i), tuple3([v.x, v.y, v.z])))
                .collect(),
        }),
        (TrackTarget::Scale, TrackValues::Vector(values)) => Some(SampledOp {
            name: "xformOp:scale",
            type_name: "float3",
            samples: values
                .iter()
                .enumerate()
                .map(|(i, v)| (time(i), tuple3([v.x, v.y, v.z])))
                .collect(),
        }),
        (TrackTarget::Quaternion, TrackValues::Quaternion(values)) => Some(SampledOp {
            name: "xformOp:orient",
            type_name: "quatf",
            samples: values
                .iter()
                .enumerate()
                // USD writes the real part first.
                .map(|(i, q)| (time(i), tuple4([q.w, q.x, q.y, q.z])))
                .collect(),
        }),
        // Colour and morph tracks have no transform op to land on; a USD
        // layer can animate those, but not as part of the xform stack.
        _ => None,
    }
}

/// Replace an op's value, keeping `xformOpOrder` correct.
///
/// A prim exported from a static pose already carries the ops its transform
/// needed; animating one that was left out — a translate that happened to be
/// zero on the frame the scene was captured at — means adding it to the order
/// as well, or USD will ignore it.
fn set_op(prim: &mut UsdPrim, name: &str, type_name: &str, value: UsdValue) {
    match prim.properties.iter_mut().find(|p| p.name == name) {
        Some(existing) => existing.value = value,
        None => prim.properties.push(attribute(type_name, name, value)),
    }

    let ordered = prim.properties.iter().any(|p| {
        p.name == "xformOpOrder" && p.value.flat_tokens().contains(&name)
    });
    if ordered {
        return;
    }
    match prim.properties.iter_mut().find(|p| p.name == "xformOpOrder") {
        Some(order) => {
            if let UsdValue::Array(items) = &mut order.value {
                items.push(UsdValue::Token(name.into()));
            }
        }
        None => {
            let mut order = attribute(
                "token[]",
                "xformOpOrder",
                UsdValue::Array(vec![UsdValue::Token(name.into())]),
            );
            order.uniform = true;
            prim.properties.push(order);
        }
    }
}

fn prim_at_mut<'a>(layer: &'a mut UsdLayer, path: &str) -> Option<&'a mut UsdPrim> {
    let mut parts = path.trim_start_matches('/').split('/');
    let first = parts.next()?;
    let mut current = layer.prims.iter_mut().find(|p| p.name == first)?;
    for part in parts {
        current = current.children.iter_mut().find(|p| p.name == part)?;
    }
    Some(current)
}

#[cfg(test)]
mod external {
    use super::super::shade;
    use super::*;

    /// One mesh per material kind, in all three forms, for Apple's tools.
    ///
    /// The round-trip tests above run what this writes back through this
    /// crate's own reader, which proves the two agree and nothing more. This
    /// writes files for `usdchecker` to validate, `usdcat` to flatten, and
    /// `usdrecord` to try to shade.
    #[test]
    #[ignore = "writes files for external validation"]
    fn write_materials_for_openusd() {
        use crate::core::{BufferAttribute, BufferGeometry, Mesh, Object3D, ObjectArena};
        use crate::materials::{
            BasicMaterial, LambertMaterial, Material, PhongMaterial, PhysicalMaterial,
            StandardMaterial, ToonMaterial,
        };
        use crate::math::{Color, Vector3};

        fn quad() -> BufferGeometry {
            let mut g = BufferGeometry::new();
            g.set_attribute(
                "position",
                BufferAttribute::new(
                    vec![-0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.5, 0.5, 0.0, -0.5, 0.5, 0.0],
                    3,
                ),
            );
            g.set_attribute(
                "normal",
                BufferAttribute::new([0.0, 0.0, 1.0].repeat(4), 3),
            );
            g.set_attribute(
                "uv",
                BufferAttribute::new(vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0], 2),
            );
            g.set_index(vec![0, 1, 2, 0, 2, 3]);
            g
        }

        let dir = std::env::var("USD_OUT").unwrap_or_else(|_| "/tmp".into());

        let basic = BasicMaterial {
            opacity: 0.5,
            ..BasicMaterial::new(Color::new(0.9, 0.1, 0.1))
        };
        let lambert = LambertMaterial {
            color: Color::new(0.1, 0.8, 0.2),
            ..Default::default()
        };
        let phong = PhongMaterial {
            color: Color::new(0.2, 0.3, 0.9),
            specular: Color::new(1.0, 1.0, 1.0),
            ..Default::default()
        };
        let standard = StandardMaterial {
            roughness: 0.25,
            metalness: 0.9,
            emissive: Color::new(0.05, 0.0, 0.1),
            ..StandardMaterial::new(Color::new(0.8, 0.6, 0.1))
        };
        let physical = PhysicalMaterial {
            color: Color::new(0.3, 0.7, 0.7),
            roughness: 0.1,
            ior: 1.45,
            clearcoat: 0.8,
            clearcoat_roughness: 0.05,
            ..Default::default()
        };
        let toon = ToonMaterial {
            color: Color::new(0.7, 0.2, 0.7),
            side: 2,
            ..Default::default()
        };

        // A checkerboard and a flat normal map, so a render shows whether the
        // texture is sampled at all and whether the UVs are the right way up.
        use crate::textures::{Texture, TextureFormat, TextureWrap};
        use std::sync::Arc;
        let mut checker = Vec::new();
        for y in 0..64u32 {
            for x in 0..64u32 {
                let on = ((x / 8) + (y / 8)) % 2 == 0;
                let (r, g, b) = if on { (240, 240, 40) } else { (30, 30, 120) };
                checker.extend_from_slice(&[r, g, b, 255]);
            }
        }
        let map = Texture {
            wrap_s: TextureWrap::Repeat,
            wrap_t: TextureWrap::Repeat,
            ..Texture::new(64, 64, TextureFormat::Rgba8UnormSrgb, checker)
        };
        let normal = Texture::new(
            2,
            2,
            TextureFormat::Rgba8Unorm,
            [128, 128, 255, 255].repeat(4),
        );
        let textured = StandardMaterial {
            roughness: 0.9,
            map: Some(Arc::new(map)),
            normal_map: Some(Arc::new(normal)),
            ..StandardMaterial::new(Color::new(1.0, 1.0, 1.0))
        };

        let kinds: Vec<(&str, Material)> = vec![
            ("Basic", basic.into()),
            ("Lambert", lambert.into()),
            ("Phong", phong.into()),
            ("Standard", standard.into()),
            ("Physical", physical.into()),
            ("Toon", toon.into()),
            ("Textured", textured.into()),
        ];
        let mut arena = ObjectArena::new();
        let mut roots = Vec::new();
        for (i, (name, material)) in kinds.into_iter().enumerate() {
            let mut object = Object3D::mesh(Mesh::new(quad(), material));
            object.name = name.into();
            object.position = Vector3::new(i as f32 * 1.2 - 3.0, 0.0, 0.0);
            roots.push(arena.insert(object));
        }

        // One export, three forms — and `write_to` puts the images beside the
        // two that refer to them rather than carrying them.
        let out = super::super::UsdExport::scene(&arena, &roots);
        let base = std::path::Path::new(&dir);
        for name in ["materials.usda", "materials.usdc", "materials.usdz"] {
            out.write_to(base.join(name), base).expect("writes");
        }
        println!(
            "wrote materials.usda / .usdc / .usdz and {} texture(s) to {dir}",
            out.textures.len()
        );
    }

    /// Read a file back and check the materials survived whatever wrote it.
    ///
    /// Pointed at `usdcat`'s output for the gallery, this closes the loop
    /// through OpenUSD rather than through this crate's own reader: export,
    /// hand to Apple's tools, read back what they produced. A writer and a
    /// reader that only ever talk to each other agree on their own mistakes.
    #[test]
    #[ignore = "reads a file produced by OpenUSD"]
    fn read_materials_back_from_openusd() {
        use crate::loaders::usd::scene::to_scene;

        let path = std::env::var("USD_IN").expect("USD_IN=<a flattened gallery>");
        let text = std::fs::read_to_string(&path).expect("readable");
        let layer = super::super::parse::parse(&text).expect("parses");
        let scene = to_scene(&layer);

        let named = |want: &str| {
            let found = scene
                .arena
                .get_objects_by_name(scene.roots[0], want)
                .into_iter()
                .next()
                .unwrap_or_else(|| panic!("no object named {want}"));
            scene.arena.get(found).unwrap().clone()
        };
        let material_of = |object: &crate::core::Object3D| match &object.kind {
            ObjectKind::Mesh(mesh) => (*mesh.material).clone(),
            other => panic!("{other:?} is not a mesh"),
        };

        match material_of(&named("Physical")) {
            Material::Physical(p) => {
                assert!((p.ior - 1.45).abs() < 1e-4, "ior came back {}", p.ior);
                assert!((p.clearcoat - 0.8).abs() < 1e-4);
                assert!((p.clearcoat_roughness - 0.05).abs() < 1e-4);
                assert!((p.color.g - 0.7).abs() < 1e-4);
            }
            other => panic!("Physical came back as {other:?}"),
        }
        assert_eq!(material_of(&named("Toon")).side(), 2, "doubleSided lost");
        assert!(
            (material_of(&named("Lambert")).color().g - 0.8).abs() < 1e-4,
            "Lambert lost its colour"
        );

        // The textured one keeps its maps, as requests for the caller to load.
        let textured = shade::bound_material_path(&layer, "/Root/Textured")
            .and_then(|p| layer.prim_at(&p))
            .map(|p| shade::resolve(p, &layer))
            .expect("a bound material");
        let slots: Vec<_> = textured.textures.iter().map(|t| t.slot).collect();
        assert!(
            slots.contains(&shade::TextureSlot::BaseColor)
                && slots.contains(&shade::TextureSlot::Normal),
            "{slots:?}"
        );
        let normal = textured
            .textures
            .iter()
            .find(|t| t.slot == shade::TextureSlot::Normal)
            .unwrap();
        assert_eq!(normal.scale, [2.0, 2.0, 2.0, 1.0]);
        assert_eq!(normal.bias, [-1.0, -1.0, -1.0, 0.0]);
        assert_eq!(normal.color_space, "raw");
        println!("materials survived the trip through OpenUSD");
    }

    /// Write archives for Apple's `usdchecker --arkit` to judge.
    #[test]
    #[ignore = "writes files for external validation"]
    fn write_archives_for_arkit() {
        use super::super::{scene_to_usdz, UsdzEntry};
        use crate::core::{Mesh, Object3D, ObjectArena};
        use crate::geometries::{BoxGeometry, SphereGeometry};
        use crate::materials::{Material, StandardMaterial};

        let dir = std::env::var("USD_OUT").unwrap_or_else(|_| "/tmp".into());
        let mut arena = ObjectArena::new();
        let mut roots = Vec::new();
        for (i, geometry) in [
            BoxGeometry::new(1.0, 2.0, 3.0),
            SphereGeometry::new(1.0, 24, 12),
        ]
        .into_iter()
        .enumerate()
        {
            let mut object = Object3D::mesh(Mesh::new(
                geometry,
                Material::Standard(StandardMaterial::default()),
            ));
            object.name = format!("Part{i}");
            roots.push(arena.insert(object));
        }

        // A texture beside the layer, which is what a real archive carries.
        let png = include_bytes!("testdata/tiny.png").to_vec();
        let archive = scene_to_usdz(
            &arena,
            &roots,
            &[UsdzEntry {
                name: "textures/albedo.png".into(),
                data: png,
            }],
        );
        let path = format!("{dir}/arkit_scene.usdz");
        std::fs::write(&path, &archive).unwrap();
        println!("wrote {path} ({} bytes)", archive.len());
    }

    /// Write an animated document to disk so that OpenUSD's own tools can be
    /// pointed at it. Ignored by default: it needs a path, and the assertion
    /// that matters is made by `usdcat`, not by this process.
    #[test]
    #[ignore = "writes a file for external validation"]
    fn write_an_animated_document() {
        use crate::animation::{AnimationClip, KeyframeTrack, TrackTarget};
        use crate::core::Object3D;
        use crate::geometries::BoxGeometry;
        use crate::math::{Quaternion, Vector3};

        let mut arena = ObjectArena::new();
        let mut object = Object3D::mesh(crate::core::Mesh::new(
            BoxGeometry::new(1.0, 1.0, 1.0),
            crate::materials::Material::Standard(Default::default()),
        ));
        object.name = "Cube".into();
        let id = arena.insert(object);

        let clip = AnimationClip::new(
            "spin",
            2.0,
            vec![
                KeyframeTrack::vector(
                    id,
                    TrackTarget::Position,
                    vec![0.0, 1.0, 2.0],
                    vec![
                        Vector3::new(0.0, 0.0, 0.0),
                        Vector3::new(3.0, 2.0, 0.0),
                        Vector3::new(6.0, 0.0, 0.0),
                    ],
                ),
                KeyframeTrack::quaternion(
                    id,
                    TrackTarget::Quaternion,
                    vec![0.0, 2.0],
                    vec![
                        Quaternion::identity(),
                        Quaternion::from_axis_angle(
                            Vector3::new(0.0, 1.0, 0.0),
                            std::f32::consts::PI,
                        ),
                    ],
                ),
                KeyframeTrack::vector(
                    id,
                    TrackTarget::Scale,
                    vec![0.0, 1.0, 2.0],
                    vec![
                        Vector3::new(1.0, 1.0, 1.0),
                        Vector3::new(2.0, 2.0, 2.0),
                        Vector3::new(1.0, 1.0, 1.0),
                    ],
                ),
            ],
        );

        let text = animated_scene_to_usda(&arena, &[id], &[clip]);
        let path = std::env::var("USD_OUT").unwrap_or_else(|_| "/tmp/threers_anim.usda".into());
        std::fs::write(&path, text).unwrap();
        println!("wrote {path}");
    }
}
