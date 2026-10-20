//! Material shader variants: which maps a material's shader declares.
//!
//! three.js compiles a program per material feature set, so a material with an
//! alpha map, a light map and a bump map gets exactly those three samplers and
//! nothing else. This renderer's main shader used to declare one fixed set of
//! material textures for every draw, sharing one sampler and one uv transform
//! between them — and with WebGL2's sixteen texture units already spent on
//! that set plus the shadow and environment maps, a material could not have a
//! map the set lacked.
//!
//! A [`MapVariant`] is the set of [`MapRole`]s one material uses. Its
//! [`prelude`] declares those maps and no others, each with its own sampler
//! and its own uv transform (three.js's `mapTransform`, `aoMapTransform`, …),
//! in group 2 from [`FIRST_BINDING`] up, and defines a `sample_*` accessor for
//! every role — present ones sample, absent ones return the value a missing
//! map stands for. It also defines `HAS_*` constants, so code for a map the
//! variant lacks is compiled out. The main shader is written against those
//! accessors (`shader_source_with_maps`).
//!
//! The legacy layout ([`LEGACY_MATERIAL_MAPS`](super::shader::LEGACY_MATERIAL_MAPS)
//! plus [`legacy_extras`]) stays what every draw that needs nothing more uses,
//! and what custom shader materials compile against.

use crate::materials::MaterialTextureSlots;
use crate::textures::Texture;
use std::fmt::Write as _;
use std::sync::Arc;

/// One kind of material map.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum MapRole {
    Map,
    AlphaMap,
    NormalMap,
    RoughnessMap,
    MetalnessMap,
    AoMap,
    EmissiveMap,
    Matcap,
    Cloud,
    IridescenceThicknessMap,
    DisplacementMap,
    LightMap,
    BumpMap,
    SpecularMap,
    SpecularIntensityMap,
    SpecularColorMap,
    ClearcoatMap,
    ClearcoatRoughnessMap,
    ClearcoatNormalMap,
    IridescenceMap,
    SheenColorMap,
    SheenRoughnessMap,
    TransmissionMap,
    ThicknessMap,
    AnisotropyMap,
    /// MeshToonMaterial's `gradientMap`, read at `(n·l/2 + ½, 0)` per light
    /// with no uv transform — `sample_gradient_map_grad`.
    GradientMap,
}

impl MapRole {
    pub const ALL: [MapRole; 26] = [
        MapRole::Map,
        MapRole::AlphaMap,
        MapRole::NormalMap,
        MapRole::RoughnessMap,
        MapRole::MetalnessMap,
        MapRole::AoMap,
        MapRole::EmissiveMap,
        MapRole::Matcap,
        MapRole::Cloud,
        MapRole::IridescenceThicknessMap,
        MapRole::DisplacementMap,
        MapRole::LightMap,
        MapRole::BumpMap,
        MapRole::SpecularMap,
        MapRole::SpecularIntensityMap,
        MapRole::SpecularColorMap,
        MapRole::ClearcoatMap,
        MapRole::ClearcoatRoughnessMap,
        MapRole::ClearcoatNormalMap,
        MapRole::IridescenceMap,
        MapRole::SheenColorMap,
        MapRole::SheenRoughnessMap,
        MapRole::TransmissionMap,
        MapRole::ThicknessMap,
        MapRole::AnisotropyMap,
        MapRole::GradientMap,
    ];

    /// The WGSL name stem: `t_<ident>`, `s_<ident>`, `sample_<ident>`.
    pub fn ident(self) -> &'static str {
        match self {
            MapRole::Map => "map",
            MapRole::AlphaMap => "alpha_map",
            MapRole::NormalMap => "normal_map",
            MapRole::RoughnessMap => "roughness_map",
            MapRole::MetalnessMap => "metalness_map",
            MapRole::AoMap => "ao_map",
            MapRole::EmissiveMap => "emissive_map",
            MapRole::Matcap => "matcap",
            MapRole::Cloud => "cloud",
            MapRole::IridescenceThicknessMap => "iridescence_thickness_map",
            MapRole::DisplacementMap => "displacement_map",
            MapRole::LightMap => "light_map",
            MapRole::BumpMap => "bump_map",
            MapRole::SpecularMap => "specular_map",
            MapRole::SpecularIntensityMap => "specular_intensity_map",
            MapRole::SpecularColorMap => "specular_color_map",
            MapRole::ClearcoatMap => "clearcoat_map",
            MapRole::ClearcoatRoughnessMap => "clearcoat_roughness_map",
            MapRole::ClearcoatNormalMap => "clearcoat_normal_map",
            MapRole::IridescenceMap => "iridescence_map",
            MapRole::SheenColorMap => "sheen_color_map",
            MapRole::SheenRoughnessMap => "sheen_roughness_map",
            MapRole::TransmissionMap => "transmission_map",
            MapRole::ThicknessMap => "thickness_map",
            MapRole::AnisotropyMap => "anisotropy_map",
            MapRole::GradientMap => "gradient_map",
        }
    }

    /// `HAS_<IDENT>`, the constant the main shader tests.
    pub fn has_const(self) -> String {
        format!("HAS_{}", self.ident().to_ascii_uppercase())
    }

    /// What a missing map reads as — three.js's value for "no map": white for
    /// a multiplier, a flat tangent-space normal, black for displacement and
    /// light. Never sampled where the shader checks presence first.
    fn absent(self) -> &'static str {
        match self {
            MapRole::NormalMap | MapRole::ClearcoatNormalMap => "vec4<f32>(0.5, 0.5, 1.0, 1.0)",
            MapRole::DisplacementMap | MapRole::LightMap | MapRole::Cloud | MapRole::BumpMap => {
                "vec4<f32>(0.0, 0.0, 0.0, 0.0)"
            }
            _ => "vec4<f32>(1.0, 1.0, 1.0, 1.0)",
        }
    }

    /// Sampled in the vertex stage (and so with an explicit level).
    pub fn vertex(self) -> bool {
        self == MapRole::DisplacementMap
    }

    /// Whether the legacy layout has a binding for this role.
    pub fn legacy(self) -> bool {
        matches!(
            self,
            MapRole::Map
                | MapRole::NormalMap
                | MapRole::RoughnessMap
                | MapRole::MetalnessMap
                | MapRole::AoMap
                | MapRole::EmissiveMap
                | MapRole::Matcap
                | MapRole::Cloud
        )
    }

    /// Whether the map holds colour (sRGB-encoded, decoded by the sampler) —
    /// three.js decodes `map`, `emissiveMap`, `specularColorMap`, `sheenColorMap`
    /// and `matcap`; every other map is data.
    pub fn colour(self) -> bool {
        matches!(
            self,
            MapRole::Map
                | MapRole::EmissiveMap
                | MapRole::Matcap
                | MapRole::LightMap
                | MapRole::SpecularColorMap
                | MapRole::SheenColorMap
        )
    }
}

/// The set of maps one material's shader declares.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct MapVariant(u32);

impl MapVariant {
    pub fn with(self, role: MapRole) -> Self {
        MapVariant(self.0 | 1 << role as u32)
    }
    pub fn contains(self, role: MapRole) -> bool {
        self.0 & (1 << role as u32) != 0
    }
    pub fn roles(self) -> impl Iterator<Item = MapRole> {
        MapRole::ALL.into_iter().filter(move |r| self.contains(*r))
    }
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
    /// The bit for a cube `envMap`, which is not a [`MapRole`]: it has no uv
    /// and is sampled by direction.
    const ENV: u32 = 1 << 31;
    pub fn with_env(self) -> Self {
        MapVariant(self.0 | Self::ENV)
    }
    pub fn has_env(self) -> bool {
        self.0 & Self::ENV != 0
    }
    /// A key naming this variant among cached pipelines.
    pub fn key(self) -> u64 {
        0x6d61_7073_0000_0000 | self.0 as u64
    }
    /// The variant whose roles are the maps present in `slots`.
    pub fn of(slots: &MaterialTextureSlots) -> Self {
        let mut v = MapVariant::default();
        for (role, tex) in role_textures(slots) {
            if tex.is_some() {
                v = v.with(role);
            }
        }
        if slots.env_map.is_some() {
            v = v.with_env();
        }
        v
    }
}

/// Each role's texture in `slots` (Matcap and Cloud are separate roles here,
/// though the legacy layout shares their binding).
pub fn role_textures(slots: &MaterialTextureSlots) -> [(MapRole, Option<&Arc<Texture>>); 26] {
    [
        (MapRole::Map, slots.map.as_ref()),
        (MapRole::AlphaMap, slots.alpha_map.as_ref()),
        (MapRole::NormalMap, slots.normal_map.as_ref()),
        (MapRole::RoughnessMap, slots.roughness_map.as_ref()),
        (MapRole::MetalnessMap, slots.metalness_map.as_ref()),
        (MapRole::AoMap, slots.ao_map.as_ref()),
        (MapRole::EmissiveMap, slots.emissive_map.as_ref()),
        (MapRole::Matcap, slots.matcap_map.as_ref()),
        (MapRole::Cloud, slots.cloud_shadow_map.as_ref()),
        (
            MapRole::IridescenceThicknessMap,
            slots.iridescence_thickness_map.as_ref(),
        ),
        (MapRole::DisplacementMap, slots.displacement_map.as_ref()),
        (MapRole::LightMap, slots.light_map.as_ref()),
        (MapRole::BumpMap, slots.bump_map.as_ref()),
        (MapRole::SpecularMap, slots.specular_map.as_ref()),
        (
            MapRole::SpecularIntensityMap,
            slots.specular_intensity_map.as_ref(),
        ),
        (MapRole::SpecularColorMap, slots.specular_color_map.as_ref()),
        (MapRole::ClearcoatMap, slots.clearcoat_map.as_ref()),
        (
            MapRole::ClearcoatRoughnessMap,
            slots.clearcoat_roughness_map.as_ref(),
        ),
        (
            MapRole::ClearcoatNormalMap,
            slots.clearcoat_normal_map.as_ref(),
        ),
        (MapRole::IridescenceMap, slots.iridescence_map.as_ref()),
        (MapRole::SheenColorMap, slots.sheen_color_map.as_ref()),
        (
            MapRole::SheenRoughnessMap,
            slots.sheen_roughness_map.as_ref(),
        ),
        (MapRole::TransmissionMap, slots.transmission_map.as_ref()),
        (MapRole::ThicknessMap, slots.thickness_map.as_ref()),
        (MapRole::AnisotropyMap, slots.anisotropy_map.as_ref()),
        (MapRole::GradientMap, slots.gradient_map.as_ref()),
    ]
}

/// Whether a material with these maps needs its own variant: it has a map the
/// legacy layout cannot hold, two of its maps disagree on filtering or wrap
/// (the legacy layout shares one sampler), or a map's uv transform is not the
/// one the legacy layout applies to every map — the colour map's offset and
/// repeat ([`legacy_uv_rows`]), so a rotation, a `center`, a hand-set
/// `matrix` or a map transformed differently from the colour map all need one.
pub fn needs_variant(slots: &MaterialTextureSlots) -> bool {
    if slots.env_map.is_some() {
        return true;
    }
    let legacy = legacy_uv_rows(slots);
    let mut first: Option<&Arc<Texture>> = None;
    for (role, tex) in role_textures(slots) {
        let Some(t) = tex else { continue };
        if !role.legacy() {
            return true;
        }
        // The mirror's render target and a matcap are not uv-mapped like the
        // rest; they never force a variant on their own.
        if matches!(role, MapRole::Matcap | MapRole::Cloud) {
            continue;
        }
        // Another uv set (`channel`), or a transform the layout cannot hold.
        if t.channel != 0 || Some(t.uv_transform_rows()) != legacy {
            return true;
        }
        match first {
            None => first = Some(t),
            Some(f) => {
                if f.mag_filter != t.mag_filter
                    || f.min_filter != t.min_filter
                    || f.wrap_s != t.wrap_s
                    || f.wrap_t != t.wrap_t
                {
                    return true;
                }
            }
        }
    }
    false
}

/// The uv transform the legacy layout applies to every map, as
/// [`Texture::uv_transform_rows`] writes it: the colour map's (else the
/// matcap's), when it is a scale and translation the layout's offset/repeat
/// pair can hold; identity with neither map. `None` when it cannot hold it —
/// a rotation, or a skewed matrix — so the material needs a variant.
pub fn legacy_uv_rows(slots: &MaterialTextureSlots) -> Option<[[f32; 3]; 2]> {
    match slots.map.as_ref().or(slots.matcap_map.as_ref()) {
        Some(t) => {
            let rows = t.uv_transform_rows();
            (rows[0][1] == 0.0 && rows[1][0] == 0.0).then_some(rows)
        }
        None => Some([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    }
}

/// The legacy layout's `[offset.x, offset.y, repeat.x, repeat.y]` for
/// [`legacy_uv_rows`]; identity where a variant carries the transforms.
pub fn legacy_uv_transform(slots: &MaterialTextureSlots) -> [f32; 4] {
    match legacy_uv_rows(slots) {
        Some([r0, r1]) => [r0[2], r1[2], r0[0], r1[1]],
        None => [0.0, 0.0, 1.0, 1.0],
    }
}

/// The first group-2 binding a variant's maps take. 0–10 are the legacy maps
/// and the morph texture; a variant keeps only the morph texture (binding 10)
/// of those, and its maps follow it.
pub const FIRST_BINDING: u32 = 11;

/// A variant's group-2 bindings: each present role's texture and sampler, in
/// role order, the `envMap` cube and its sampler, then the uniform block.
pub struct VariantBindings {
    pub maps: Vec<(MapRole, u32, u32)>,
    pub env: Option<(u32, u32)>,
    pub xforms: u32,
}

pub fn bindings(variant: MapVariant) -> VariantBindings {
    let mut maps = Vec::new();
    let mut b = FIRST_BINDING;
    for role in variant.roles() {
        maps.push((role, b, b + 1));
        b += 2;
    }
    let env = variant.has_env().then(|| {
        b += 2;
        (b - 2, b - 1)
    });
    VariantBindings {
        maps,
        env,
        xforms: b,
    }
}

/// The variant's prelude: its maps, their samplers and transforms, and an
/// accessor and a `HAS_*` constant for every role.
pub fn prelude(variant: MapVariant) -> String {
    let b = bindings(variant);
    let mut s = String::new();
    let n = b.maps.len().max(1);
    let _ = writeln!(
        s,
        "// Material maps for this feature set: each with its own sampler and its\n\
         // three.js uv transform (two rows of a 3×3; row 0's w its `channel`),\n\
         // applied here to the raw uv or the uv set the channel names."
    );
    let _ = writeln!(
        s,
        "// `Material::map_uniforms`: `params` [lightMapIntensity, bumpScale,\n\
         // iridescenceThicknessRange[0], 0]; `params2` [clearcoatNormalScale, 0, 0];\n\
         // `env` [combine, reflectivity, refractionRatio, refraction mapping];\n\
         // `env_dir` the rows of envMapRotation · diag(flipEnvMap, 1, 1).\n\
         struct MapXforms {{ params : vec4<f32>, env : vec4<f32>, env_dir : array<vec4<f32>, 3>, params2 : vec4<f32>, rows : array<vec4<f32>, {}> }};",
        2 * n
    );
    let _ = writeln!(
        s,
        "@group(2) @binding({}) var<uniform> map_xforms : MapXforms;",
        b.xforms
    );
    for (role, t, smp) in &b.maps {
        let id = role.ident();
        let _ = writeln!(s, "@group(2) @binding({t}) var t_{id} : texture_2d<f32>;");
        let _ = writeln!(s, "@group(2) @binding({smp}) var s_{id} : sampler;");
    }
    let _ = writeln!(
        s,
        "fn map_xform(i : u32, uv : vec2<f32>) -> vec2<f32> {{\n    \
             let p = vec3<f32>(map_uv_set(map_xforms.rows[2u * i].w, uv), 1.0);\n    \
             return vec2<f32>(dot(map_xforms.rows[2u * i].xyz, p), dot(map_xforms.rows[2u * i + 1u].xyz, p));\n\
         }}"
    );
    let _ = writeln!(
        s,
        "fn map_params() -> vec4<f32> {{ return map_xforms.params; }}"
    );
    let _ = writeln!(
        s,
        "fn map_params2() -> vec4<f32> {{ return map_xforms.params2; }}"
    );
    let _ = writeln!(
        s,
        "fn env_params() -> vec4<f32> {{ return map_xforms.env; }}"
    );
    let _ = writeln!(
        s,
        "fn env_lookup(r : vec3<f32>) -> vec3<f32> {{\n    \
             return vec3<f32>(dot(map_xforms.env_dir[0].xyz, r), dot(map_xforms.env_dir[1].xyz, r), dot(map_xforms.env_dir[2].xyz, r));\n\
         }}"
    );
    for role in MapRole::ALL {
        let _ = writeln!(
            s,
            "const {} : bool = {};",
            role.has_const(),
            variant.contains(role)
        );
    }
    match b.env {
        Some((t, smp)) => {
            let _ = writeln!(
                s,
                "@group(2) @binding({t}) var t_env_map : texture_cube<f32>;\n\
                 @group(2) @binding({smp}) var s_env_map : sampler;\n\
                 const HAS_ENV_MAP : bool = true;\n\
                 fn sample_env_map(dir : vec3<f32>) -> vec4<f32> {{ return textureSample(t_env_map, s_env_map, env_lookup(dir)); }}"
            );
        }
        None => {
            let _ = writeln!(s, "{}", ABSENT_ENV);
        }
    }
    for role in MapRole::ALL {
        let id = role.ident();
        let index = b.maps.iter().position(|(r, _, _)| *r == role);
        let body = match index {
            Some(i) if role.vertex() => {
                format!("return textureSampleLevel(t_{id}, s_{id}, map_xform({i}u, uv), 0.0);")
            }
            Some(i) => format!("return textureSample(t_{id}, s_{id}, map_xform({i}u, uv));"),
            None => format!("return {};", role.absent()),
        };
        let _ = writeln!(
            s,
            "fn sample_{id}(uv : vec2<f32>) -> vec4<f32> {{ {body} }}"
        );
        // At level 0, where there are no derivatives to pick one (the cut-out
        // shadow test).
        let level = match index {
            Some(i) => {
                format!("return textureSampleLevel(t_{id}, s_{id}, map_xform({i}u, uv), 0.0);")
            }
            None => format!("return {};", role.absent()),
        };
        let _ = writeln!(
            s,
            "fn sample_{id}_level(uv : vec2<f32>) -> vec4<f32> {{ {level} }}"
        );
    }
    // three.js `getGradientIrradiance`: `texture2D(gradientMap, vec2(coord,
    // 0.0))`, untransformed. Called per light inside the lighting loop, so the
    // derivatives come in explicitly — `coord`'s own, for the texture's
    // filter to pick its level as three.js's implicit one does.
    let gradient = match b.maps.iter().any(|(r, _, _)| *r == MapRole::GradientMap) {
        true => "return textureSampleGrad(t_gradient_map, s_gradient_map, vec2<f32>(coord, 0.0), vec2<f32>(dx, 0.0), vec2<f32>(dy, 0.0));",
        false => "return vec4<f32>(1.0, 1.0, 1.0, 1.0);",
    };
    let _ = writeln!(
        s,
        "fn sample_gradient_map_grad(coord : f32, dx : f32, dy : f32) -> vec4<f32> {{ {gradient} }}"
    );
    // A Reflector's mirror view, which rides the map slot at a projected,
    // untransformed uv.
    let mirror = match b.maps.iter().any(|(r, _, _)| *r == MapRole::Map) {
        true => "return textureSampleLevel(t_map, s_map, uv, 0.0);",
        false => "return vec4<f32>(1.0);",
    };
    let _ = writeln!(
        s,
        "fn sample_mirror(uv : vec2<f32>) -> vec4<f32> {{ {mirror} }}"
    );
    s
}

/// A variant (or the legacy layout) with no `envMap`.
const ABSENT_ENV: &str = "const HAS_ENV_MAP : bool = false;\n\
     fn sample_env_map(dir : vec3<f32>) -> vec4<f32> { return vec4<f32>(0.0); }";

/// What the legacy layout adds to its fixed bindings: the `HAS_*` constants
/// and accessors of the roles it has no binding for, all absent.
pub fn legacy_extras() -> String {
    let mut s = String::new();
    // The legacy roles are gated by the per-draw texture flags, as they always
    // were; `HAS_*` is how the newer roles are gated, so all are false here.
    for role in MapRole::ALL {
        let _ = writeln!(s, "const {} : bool = false;", role.has_const());
    }
    // three.js's defaults; the maps they scale are never bound here.
    let _ = writeln!(
        s,
        "fn map_params() -> vec4<f32> {{ return vec4<f32>(1.0, 1.0, 0.0, 0.0); }}"
    );
    let _ = writeln!(
        s,
        "fn map_params2() -> vec4<f32> {{ return vec4<f32>(1.0, 1.0, 0.0, 0.0); }}"
    );
    let _ = writeln!(
        s,
        "fn env_params() -> vec4<f32> {{ return vec4<f32>(0.0, 1.0, 0.98, 0.0); }}"
    );
    let _ = writeln!(s, "{}", ABSENT_ENV);
    for role in MapRole::ALL.into_iter().filter(|r| !r.legacy()) {
        let (id, absent) = (role.ident(), role.absent());
        let _ = writeln!(
            s,
            "fn sample_{id}(uv : vec2<f32>) -> vec4<f32> {{ return {absent}; }}\n\
             fn sample_{id}_level(uv : vec2<f32>) -> vec4<f32> {{ return {absent}; }}"
        );
    }
    let _ = writeln!(
        s,
        "fn sample_gradient_map_grad(coord : f32, dx : f32, dy : f32) -> vec4<f32> {{ return vec4<f32>(1.0, 1.0, 1.0, 1.0); }}"
    );
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bindings_follow_the_morph_texture_in_role_order() {
        let v = MapVariant::default()
            .with(MapRole::BumpMap)
            .with(MapRole::Map);
        let b = bindings(v);
        assert_eq!(
            b.maps,
            vec![(MapRole::Map, 11, 12), (MapRole::BumpMap, 13, 14)]
        );
        assert_eq!(b.xforms, 15);
        assert!(prelude(v).contains("const HAS_BUMP_MAP : bool = true;"));
        assert!(prelude(v).contains("const HAS_ALPHA_MAP : bool = false;"));
    }

    /// Every role the shader samples is a slot three.js code can fill by
    /// name — the shim, Python and Node all go through `map_slot_mut`.
    #[test]
    fn every_role_has_a_three_js_slot_name() {
        use crate::materials::{
            MatcapMaterial, Material, PhongMaterial, PhysicalMaterial, ToonMaterial,
        };
        let mut owners: [Material; 4] = [
            PhysicalMaterial::default().into(),
            PhongMaterial::default().into(),
            MatcapMaterial::default().into(),
            ToonMaterial::default().into(),
        ];
        for role in MapRole::ALL {
            let name = match role {
                MapRole::Cloud => "cloudShadowMap".to_string(),
                r => {
                    let mut out = String::new();
                    let mut up = false;
                    for c in r.ident().chars() {
                        if c == '_' {
                            up = true;
                        } else if up {
                            out.push(c.to_ascii_uppercase());
                            up = false;
                        } else {
                            out.push(c);
                        }
                    }
                    out
                }
            };
            assert!(
                owners.iter_mut().any(|m| m.map_slot_mut(&name).is_some()),
                "no material has a '{name}' slot"
            );
        }
    }

    fn texel() -> Texture {
        Texture::solid([255; 4], crate::textures::TextureFormat::Rgba8Unorm)
    }

    /// The legacy layout holds offset and repeat shared by every map; any
    /// other transform needs a variant, which is how three.js's per-texture
    /// `matrix` survives.
    #[test]
    fn transforms_the_legacy_layout_cannot_express_need_a_variant() {
        let tiled = || {
            let mut t = texel();
            t.repeat = crate::math::Vector2::new(4.0, 2.0);
            t.offset = crate::math::Vector2::new(0.25, 0.0);
            Arc::new(t)
        };
        let both = MaterialTextureSlots {
            map: Some(tiled()),
            normal_map: Some(tiled()),
            ..Default::default()
        };
        assert!(!needs_variant(&both), "colour and normal map tiled alike");

        let alone = MaterialTextureSlots {
            normal_map: Some(tiled()),
            ..Default::default()
        };
        assert!(
            needs_variant(&alone),
            "a tiled normal map with no colour map"
        );

        let mut turned = texel();
        turned.rotation = 0.5;
        let turned = MaterialTextureSlots {
            map: Some(Arc::new(turned)),
            ..Default::default()
        };
        assert!(needs_variant(&turned), "a rotated colour map");

        let plain = MaterialTextureSlots {
            map: Some(Arc::new(texel())),
            ..Default::default()
        };
        assert!(!needs_variant(&plain), "an untransformed colour map");
    }
}
