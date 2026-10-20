//! Materials and textures as three.js r165 `GLTFLoader` builds them: the
//! parameter set each `MeshXxxMaterial` is constructed with (`loadMaterial`
//! and the `KHR_materials_*` / `EXT_materials_bump` extensions), the derived
//! materials `assignFinalMaterial` makes, and the `Texture`s
//! `loadTextureImage` / `assignTexture` / `KHR_texture_transform` make —
//! and, from those, the renderer's own [`Material`] and [`Texture`].

use std::collections::BTreeMap;
use std::sync::Arc;

use super::super::json::Value;
use crate::materials::{
    BasicMaterial, LineBasicMaterial, Material, PhysicalMaterial, PointsMaterial, StandardMaterial,
};
use crate::math::{Color, Vector2};
use crate::textures::{Texture, TextureFilter, TextureFormat, TextureWrap};

/// The three.js material class a glTF material becomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialType {
    MeshStandardMaterial,
    MeshPhysicalMaterial,
    MeshBasicMaterial,
    PointsMaterial,
    LineBasicMaterial,
}

impl MaterialType {
    pub fn name(self) -> &'static str {
        match self {
            MaterialType::MeshStandardMaterial => "MeshStandardMaterial",
            MaterialType::MeshPhysicalMaterial => "MeshPhysicalMaterial",
            MaterialType::MeshBasicMaterial => "MeshBasicMaterial",
            MaterialType::PointsMaterial => "PointsMaterial",
            MaterialType::LineBasicMaterial => "LineBasicMaterial",
        }
    }
}

/// One value of a three.js material parameter.
#[derive(Debug, Clone, PartialEq)]
pub enum Param {
    Num(f64),
    Bool(bool),
    /// A `Color` (linear RGB).
    Color([f32; 3]),
    /// A `Vector2`.
    Vec2([f32; 2]),
    /// A two-element array (`iridescenceThicknessRange`).
    Range([f32; 2]),
    /// A texture: an index into the parser's textures.
    Texture(usize),
}

/// Whether, and with what, a thing is in `parser.associations`.
///
/// three.js maps an object to a *shared* record (clones made by
/// `_getNodeRef` share their original's), and sometimes maps a key to
/// `undefined` (`assignFinalMaterial` copying a missing association) — so
/// "present but empty" differs from "absent".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Assoc {
    #[default]
    Absent,
    /// In the map with `undefined`.
    Undefined,
    /// In the map with this record (an index into the parser's records).
    Record(usize),
}

/// A three.js material the loader made.
#[derive(Debug, Clone)]
pub struct GltfMaterial {
    pub ty: MaterialType,
    pub name: String,
    pub user_data: Value,
    /// The constructor parameters plus what was set afterwards
    /// (`vertexColors`, `flatShading`, the flipped `normalScale`), by three.js
    /// property name.
    pub params: BTreeMap<&'static str, Param>,
    pub association: Assoc,
    /// The glTF normal-texture scale, unflipped: the renderer builds its
    /// tangent frame in glTF's convention (bitangent = normal × tangent), so
    /// it needs none of three.js's derivative-tangent `normalScale.y` flip.
    pub gltf_normal_scale: Option<f32>,
    pub gltf_clearcoat_normal_scale: Option<f32>,
    /// The renderer's material, built once the textures are final.
    pub rust: Option<Arc<Material>>,
}

impl GltfMaterial {
    pub fn new(ty: MaterialType) -> Self {
        GltfMaterial {
            ty,
            name: String::new(),
            user_data: Value::Object(Default::default()),
            params: BTreeMap::new(),
            association: Assoc::Absent,
            gltf_normal_scale: None,
            gltf_clearcoat_normal_scale: None,
            rust: None,
        }
    }

    pub fn num(&self, key: &str) -> Option<f64> {
        match self.params.get(key) {
            Some(Param::Num(n)) => Some(*n),
            _ => None,
        }
    }
    pub fn bool(&self, key: &str) -> Option<bool> {
        match self.params.get(key) {
            Some(Param::Bool(b)) => Some(*b),
            _ => None,
        }
    }
    pub fn color(&self, key: &str) -> Option<Color> {
        match self.params.get(key) {
            Some(Param::Color(c)) => Some(Color::new(c[0], c[1], c[2])),
            _ => None,
        }
    }
    pub fn texture(&self, key: &str) -> Option<usize> {
        match self.params.get(key) {
            Some(Param::Texture(t)) => Some(*t),
            _ => None,
        }
    }
}

/// three.js texture constants.
pub mod constants {
    pub const REPEAT_WRAPPING: u32 = 1000;
    pub const CLAMP_TO_EDGE_WRAPPING: u32 = 1001;
    pub const MIRRORED_REPEAT_WRAPPING: u32 = 1002;
    pub const NEAREST_FILTER: u32 = 1003;
    pub const NEAREST_MIPMAP_NEAREST_FILTER: u32 = 1004;
    pub const NEAREST_MIPMAP_LINEAR_FILTER: u32 = 1005;
    pub const LINEAR_FILTER: u32 = 1006;
    pub const LINEAR_MIPMAP_NEAREST_FILTER: u32 = 1007;
    pub const LINEAR_MIPMAP_LINEAR_FILTER: u32 = 1008;
}
use constants::*;

/// `WEBGL_FILTERS[sampler.magFilter] || LinearFilter` and friends.
pub fn three_filter(gl: Option<usize>, fallback: u32) -> u32 {
    match gl {
        Some(9728) => NEAREST_FILTER,
        Some(9729) => LINEAR_FILTER,
        Some(9984) => NEAREST_MIPMAP_NEAREST_FILTER,
        Some(9985) => LINEAR_MIPMAP_NEAREST_FILTER,
        Some(9986) => NEAREST_MIPMAP_LINEAR_FILTER,
        Some(9987) => LINEAR_MIPMAP_LINEAR_FILTER,
        _ => fallback,
    }
}
pub fn three_wrap(gl: Option<usize>) -> u32 {
    match gl {
        Some(33071) => CLAMP_TO_EDGE_WRAPPING,
        Some(33648) => MIRRORED_REPEAT_WRAPPING,
        _ => REPEAT_WRAPPING,
    }
}

/// A decoded image.
#[derive(Debug, Clone)]
pub enum ImageData {
    /// 8-bit RGBA, rows top first.
    Rgba { width: u32, height: u32, data: Arc<Vec<u8>> },
    /// A texture the image decoded straight to (KTX2: block-compressed, with
    /// its mip chain).
    Gpu(Arc<Texture>),
}

impl ImageData {
    pub fn size(&self) -> (u32, u32) {
        match self {
            ImageData::Rgba { width, height, .. } => (*width, *height),
            ImageData::Gpu(t) => (t.width, t.height),
        }
    }
}

/// A three.js `Texture` the loader made.
#[derive(Debug, Clone)]
pub struct GltfTexture {
    /// The `images[]` index the pixels came from.
    pub image: usize,
    pub image_data: ImageData,
    pub name: String,
    pub mag_filter: u32,
    pub min_filter: u32,
    pub wrap_s: u32,
    pub wrap_t: u32,
    pub channel: u32,
    pub offset: [f32; 2],
    pub repeat: [f32; 2],
    pub rotation: f32,
    /// `'srgb'` once some material slot asked for it, `''` otherwise.
    pub srgb: bool,
    pub user_data: Value,
    pub association: Assoc,
    rust_srgb: Option<Arc<Texture>>,
    rust_linear: Option<Arc<Texture>>,
}

impl GltfTexture {
    pub fn new(image: usize, image_data: ImageData) -> Self {
        GltfTexture {
            image,
            image_data,
            name: String::new(),
            mag_filter: LINEAR_FILTER,
            min_filter: LINEAR_MIPMAP_LINEAR_FILTER,
            wrap_s: REPEAT_WRAPPING,
            wrap_t: REPEAT_WRAPPING,
            channel: 0,
            offset: [0.0, 0.0],
            repeat: [1.0, 1.0],
            rotation: 0.0,
            srgb: false,
            user_data: Value::Object(Default::default()),
            association: Assoc::Absent,
            rust_srgb: None,
            rust_linear: None,
        }
    }

    /// three.js `texture.clone()`: the same image, every property copied,
    /// `userData` deep-copied — a new texture as far as anything else knows.
    pub fn clone_texture(&self) -> Self {
        GltfTexture {
            association: Assoc::Absent,
            rust_srgb: None,
            rust_linear: None,
            ..self.clone()
        }
    }

    /// The renderer's texture, as colour (`srgb`) or as data.
    pub fn rust(&mut self, srgb: bool) -> Arc<Texture> {
        let slot = if srgb { &self.rust_srgb } else { &self.rust_linear };
        if let Some(t) = slot {
            return t.clone();
        }
        let mut t = match &self.image_data {
            ImageData::Rgba { width, height, data } => {
                let format = if srgb { TextureFormat::Rgba8UnormSrgb } else { TextureFormat::Rgba8Unorm };
                let mut t = Texture::new(*width, *height, format, Vec::new());
                t.data = data.clone();
                t
            }
            ImageData::Gpu(t) => (**t).clone(),
        };
        let nearest = |f: u32| matches!(f, NEAREST_FILTER | NEAREST_MIPMAP_NEAREST_FILTER | NEAREST_MIPMAP_LINEAR_FILTER);
        t.mag_filter = if nearest(self.mag_filter) { TextureFilter::Nearest } else { TextureFilter::Linear };
        t.min_filter = if nearest(self.min_filter) { TextureFilter::Nearest } else { TextureFilter::Linear };
        let wrap = |w: u32| match w {
            REPEAT_WRAPPING => TextureWrap::Repeat,
            MIRRORED_REPEAT_WRAPPING => TextureWrap::MirroredRepeat,
            _ => TextureWrap::ClampToEdge,
        };
        t.wrap_s = wrap(self.wrap_s);
        t.wrap_t = wrap(self.wrap_t);
        // glTF puts UV (0, 0) at the image's first row: three.js's `flipY = false`.
        t.flip_y = false;
        t.offset = Vector2::new(self.offset[0], self.offset[1]);
        t.repeat = Vector2::new(self.repeat[0], self.repeat[1]);
        t.rotation = self.rotation;
        let t = Arc::new(t);
        if srgb {
            self.rust_srgb = Some(t.clone());
        } else {
            self.rust_linear = Some(t.clone());
        }
        t
    }
}

/// The renderer's material for `m`. Colour textures (`map`, `emissiveMap`)
/// are sampled as sRGB and data textures (normal, metal-rough, occlusion,
/// the extension maps) linearly — whichever `colorSpace` three.js left on a
/// texture object shared between the two.
pub fn build_rust_material(m: &GltfMaterial, textures: &mut [GltfTexture]) -> Material {
    let mut tex = |key: &str, srgb: bool| -> Option<Arc<Texture>> {
        let i = m.texture(key)?;
        textures.get_mut(i).map(|t| t.rust(srgb))
    };
    let color = m.color("color").unwrap_or(Color::WHITE);
    let opacity = m.num("opacity").unwrap_or(1.0) as f32;
    let side = m.num("side").unwrap_or(0.0) as u32;
    match m.ty {
        MaterialType::MeshBasicMaterial => {
            let mut b = BasicMaterial::new(color);
            b.opacity = opacity;
            b.map = tex("map", true);
            b.transparent = m.bool("transparent").unwrap_or(false);
            b.alpha_test = m.num("alphaTest").unwrap_or(0.0) as f32;
            b.side = side;
            Material::Basic(b)
        }
        MaterialType::PointsMaterial => {
            let mut p = PointsMaterial::new(color, m.num("size").unwrap_or(1.0) as f32);
            p.opacity = opacity;
            p.size_attenuation = m.bool("sizeAttenuation").unwrap_or(true);
            Material::Points(p)
        }
        MaterialType::LineBasicMaterial => {
            let mut l = LineBasicMaterial::new(color);
            l.opacity = opacity;
            Material::Line(l)
        }
        MaterialType::MeshStandardMaterial | MaterialType::MeshPhysicalMaterial => {
            let normal_scale = m.gltf_normal_scale.unwrap_or(1.0);
            let emissive = m.color("emissive").unwrap_or(Color::BLACK);
            let mut s = StandardMaterial::new(color);
            s.opacity = opacity;
            s.roughness = m.num("roughness").unwrap_or(1.0) as f32;
            s.metalness = m.num("metalness").unwrap_or(0.0) as f32;
            s.emissive = emissive;
            s.emissive_intensity = m.num("emissiveIntensity").unwrap_or(1.0) as f32;
            s.ao_intensity = m.num("aoMapIntensity").unwrap_or(1.0) as f32;
            s.normal_scale = Vector2::new(normal_scale, normal_scale);
            s.map = tex("map", true);
            s.emissive_map = tex("emissiveMap", true);
            s.normal_map = tex("normalMap", false);
            s.roughness_map = tex("roughnessMap", false);
            s.metalness_map = tex("metalnessMap", false);
            s.ao_map = tex("aoMap", false);
            s.side = side;
            if m.ty == MaterialType::MeshStandardMaterial {
                return Material::Standard(s);
            }
            let range = match m.params.get("iridescenceThicknessRange") {
                Some(Param::Range(r)) => *r,
                _ => [100.0, 400.0],
            };
            let num = |k: &str, d: f32| m.num(k).map(|v| v as f32).unwrap_or(d);
            let p = PhysicalMaterial {
                color: s.color,
                emissive: s.emissive,
                emissive_intensity: s.emissive_intensity,
                roughness: s.roughness,
                metalness: s.metalness,
                ao_intensity: s.ao_intensity,
                normal_scale: s.normal_scale,
                opacity: s.opacity,
                map: s.map.clone(),
                normal_map: s.normal_map.clone(),
                roughness_map: s.roughness_map.clone(),
                metalness_map: s.metalness_map.clone(),
                ao_map: s.ao_map.clone(),
                emissive_map: s.emissive_map.clone(),
                clearcoat: num("clearcoat", 0.0),
                clearcoat_roughness: num("clearcoatRoughness", 0.0),
                ior: num("ior", 1.5),
                transmission: num("transmission", 0.0),
                thickness: num("thickness", 0.0),
                dispersion: num("dispersion", 0.0),
                sheen: num("sheen", 0.0),
                sheen_color: m.color("sheenColor").unwrap_or(Color::BLACK),
                sheen_roughness: num("sheenRoughness", 1.0),
                iridescence: num("iridescence", 0.0),
                iridescence_ior: num("iridescenceIOR", 1.3),
                iridescence_thickness_range: range,
                specular_intensity: num("specularIntensity", 1.0),
                specular_color: m.color("specularColor").unwrap_or(Color::WHITE),
                iridescence_thickness_map: tex("iridescenceThicknessMap", false),
                specular_intensity_map: tex("specularIntensityMap", false),
                specular_color_map: tex("specularColorMap", true),
                clearcoat_map: tex("clearcoatMap", false),
                clearcoat_roughness_map: tex("clearcoatRoughnessMap", false),
                clearcoat_normal_map: tex("clearcoatNormalMap", false),
                // As three.js sets it — y flipped where the tangents are
                // derived — because the coat's normal map is read in
                // three.js's `getTangentFrame` (unlike the base normal map).
                clearcoat_normal_scale: match m.params.get("clearcoatNormalScale") {
                    Some(Param::Vec2(v)) => Vector2::new(v[0], v[1]),
                    _ => Vector2::new(1.0, 1.0),
                },
                iridescence_map: tex("iridescenceMap", false),
                sheen_color_map: tex("sheenColorMap", true),
                sheen_roughness_map: tex("sheenRoughnessMap", false),
                transmission_map: tex("transmissionMap", false),
                thickness_map: tex("thicknessMap", false),
                anisotropy_map: tex("anisotropyMap", false),
                anisotropy: num("anisotropy", 0.0),
                anisotropy_rotation: num("anisotropyRotation", 0.0),
                attenuation_distance: m.num("attenuationDistance").map(|v| v as f32).unwrap_or(f32::INFINITY),
                attenuation_color: m.color("attenuationColor").unwrap_or(Color::WHITE),
                side,
                ..PhysicalMaterial::default()
            };
            Material::Physical(p)
        }
    }
}
