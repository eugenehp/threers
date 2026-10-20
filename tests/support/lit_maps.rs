//! three.js's `map`, `aoMap` / `aoMapIntensity` and `emissiveMap` on
//! `MeshLambertMaterial`, `MeshPhongMaterial` and `MeshToonMaterial`, against
//! three.js r165's shading rather than another backend.
//!
//! A white quad faces an orthographic camera. Light at intensity π makes a
//! white Lambert surface shade to exactly its albedo (`BRDF_Lambert` is
//! albedo / π), so every expected byte is a plain fraction of 255:
//!
//! - **map**: a red 1×1 map under ambient light → (255, 0, 0);
//! - **ao**: a black `aoMap` occludes the indirect light only (three.js's
//!   `aomap_fragment` scales `reflectedLight.indirectDiffuse`), so ambient light
//!   alone gives black, `aoMapIntensity` 0.5 gives half (188 after the sRGB
//!   encode), and a direct light facing the quad is untouched (255);
//! - **emissive**: white `emissive` times a red `emissiveMap`, with no light,
//!   gives (255, 0, 0);
//! - **normal**: a tangent-space `normalMap` tilting the normal 75° turns a
//!   head-on directional light to cos 75° (139), and toon to its 0.7 band;
//! - **light map**: a red `lightMap` at `lightMapIntensity` π adds π of red
//!   irradiance to the indirect light (`lights_fragment_maps`), shading to
//!   exactly red with no light at all — and a black `aoMap` occludes it, since
//!   `aomap_fragment` scales the indirect diffuse it went into;
//! - **bump**: a `bumpMap` ramping 0 → 1 across the quad's middle, at a
//!   `bumpScale` making each pixel's height step 1, tilts the normal 45°
//!   toward −x (`perturbNormalArb`). A light from the left at 45° then meets
//!   it head on (255); the opposite tilt would give 0 and no bump cos 45°
//!   (219). Under a (flat) `normalMap` the bump map is ignored, as three.js's
//!   `normal_fragment_maps` does: 219;
//! - **specular map** (Phong): a black surface with white specular under a
//!   head-on light shows only its highlight, which a `specularMap` with red 0
//!   removes (`material.specularStrength`);
//! - and the light map on Basic (texel × intensity ÷ π, `meshbasic_frag`) and
//!   Standard.
//!
//! Shared by `tests/lit_maps.rs` (wgpu) and `tests/metal_backend.rs`.
#![allow(dead_code)]

use std::sync::Arc;
use threers::cameras::{Camera, OrthographicCamera};
use threers::core::{Mesh, Object3D};
use threers::geometries::PlaneGeometry;
use threers::lights::{AmbientLight, DirectionalLight, Light};
use threers::materials::{LambertMaterial, Material, PhongMaterial, ToonMaterial};
use threers::math::{Color, Vector3};
use threers::scene::Scene;
use threers::textures::{Texture, TextureFormat};

pub const N: u32 = 32;
const PI: f32 = std::f32::consts::PI;

pub fn camera() -> OrthographicCamera {
    let mut cam = OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    cam.position = Vector3::new(0.0, 0.0, 5.0);
    cam.look_at(Vector3::ZERO);
    cam
}

fn texel(rgba: [u8; 4]) -> Arc<Texture> {
    Arc::new(Texture::new(1, 1, TextureFormat::Rgba8Unorm, rgba.to_vec()))
}

#[derive(Clone, Default)]
pub struct Maps {
    pub map: Option<Arc<Texture>>,
    pub ao: Option<(Arc<Texture>, f32)>,
    pub emissive: Option<Arc<Texture>>,
    pub normal: Option<Arc<Texture>>,
    pub light: Option<(Arc<Texture>, f32)>,
    pub bump: Option<(Arc<Texture>, f32)>,
}

/// Each kind's new maps, set through the slot names three.js uses.
fn extras(m: &mut Material, maps: &Maps) {
    if let Some((t, i)) = &maps.light {
        *m.map_slot_mut("lightMap").unwrap() = Some(t.clone());
        match m {
            Material::Lambert(x) => x.light_map_intensity = *i,
            Material::Phong(x) => x.light_map_intensity = *i,
            Material::Toon(x) => x.light_map_intensity = *i,
            _ => unreachable!(),
        }
    }
    if let Some((t, s)) = &maps.bump {
        *m.map_slot_mut("bumpMap").unwrap() = Some(t.clone());
        match m {
            Material::Lambert(x) => x.bump_scale = *s,
            Material::Phong(x) => x.bump_scale = *s,
            Material::Toon(x) => x.bump_scale = *s,
            _ => unreachable!(),
        }
    }
}

pub type Build = fn(Maps) -> Material;

pub const KINDS: [(&str, Build); 3] = [
    ("lambert", |m| {
        let mut x = LambertMaterial::new(Color::WHITE);
        x.map = m.map.clone();
        if let Some((t, i)) = m.ao.clone() {
            x.ao_map = Some(t);
            x.ao_intensity = i;
        }
        if let Some(t) = m.emissive.clone() {
            x.emissive = Color::WHITE;
            x.emissive_map = Some(t);
        }
        x.normal_map = m.normal.clone();
        let mut x: Material = x.into();
        extras(&mut x, &m);
        x
    }),
    ("phong", |m| {
        let mut x = PhongMaterial::new(Color::WHITE);
        // No specular, so the diffuse term alone is measured.
        x.specular = Color::new(0.0, 0.0, 0.0);
        x.map = m.map.clone();
        if let Some((t, i)) = m.ao.clone() {
            x.ao_map = Some(t);
            x.ao_intensity = i;
        }
        if let Some(t) = m.emissive.clone() {
            x.emissive = Color::WHITE;
            x.emissive_map = Some(t);
        }
        x.normal_map = m.normal.clone();
        let mut x: Material = x.into();
        extras(&mut x, &m);
        x
    }),
    ("toon", |m| {
        let mut x = ToonMaterial::new(Color::WHITE);
        x.map = m.map.clone();
        if let Some((t, i)) = m.ao.clone() {
            x.ao_map = Some(t);
            x.ao_intensity = i;
        }
        if let Some(t) = m.emissive.clone() {
            x.emissive = Color::WHITE;
            x.emissive_map = Some(t);
        }
        x.normal_map = m.normal.clone();
        let mut x: Material = x.into();
        extras(&mut x, &m);
        x
    }),
];

#[derive(Clone, Copy)]
pub enum Lighting {
    None,
    Ambient,
    /// A directional light shining straight at the quad, no ambient.
    Direct,
    /// A directional light from the left, 45° off the quad's normal.
    FromLeft,
}

pub fn scene(material: Material, lighting: Lighting) -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x0000ff);
    match lighting {
        Lighting::None => {}
        Lighting::Ambient => {
            scene.add_light(AmbientLight::new(Color::WHITE, PI));
        }
        Lighting::Direct => {
            let l = DirectionalLight::new(Color::WHITE, PI).with_direction(Vector3::new(0.0, 0.0, -1.0));
            scene.add(Object3D::light(Light::Directional(l)));
        }
        Lighting::FromLeft => {
            let l = DirectionalLight::new(Color::WHITE, PI).with_direction(Vector3::new(1.0, 0.0, -1.0));
            scene.add(Object3D::light(Light::Directional(l)));
        }
    }
    scene.add(Object3D::mesh(Mesh::new(PlaneGeometry::new(1.6, 1.6), material)));
    scene
}

fn centre(rgba: &[u8]) -> [u8; 3] {
    let i = (((N / 2) * N + N / 2) * 4) as usize;
    [rgba[i], rgba[i + 1], rgba[i + 2]]
}

/// Within 3 levels: the normal map's 8-bit encoding (128 is 0.502, not ½)
/// moves the tilted case by about one.
fn near(got: [u8; 3], want: [u8; 3]) -> bool {
    (0..3).all(|k| (got[k] as i32 - want[k] as i32).abs() <= 3)
}

/// Every failure, as a line; empty when each kind shades its maps as three.js.
pub fn check(label: &str, render: &mut dyn FnMut(&mut Scene, &dyn Camera) -> Vec<u8>) -> Vec<String> {
    let cam = camera();
    let red = || texel([255, 0, 0, 255]);
    let black = || texel([0, 0, 0, 255]);
    let mut failures = Vec::new();
    for (name, build) in KINDS {
        // A tangent-space normal tilted 75° along the surface's u axis:
        // (sin 75°, 0, cos 75°) encoded as n·0.5 + 0.5. A light along the
        // geometric normal reaches it at cos 75° ≈ 0.259 — 139 after the sRGB
        // encode — and toon's default ramp (r165 `getGradientIrradiance`:
        // below n·l = 0.4 the 0.7 band) gives 0.7, 218.
        let tilted = texel([251, 128, 160, 255]);
        let tilted_want = if name == "toon" { [218, 218, 218] } else { [139, 139, 139] };
        let white = || texel([255, 255, 255, 255]);
        // Heights 0 | 1, linearly filtered: a ramp over u ∈ [¼, ¾], so 2 per
        // unit of u. The quad is 1.6 of the view's 2 across N pixels — u moves
        // 1 / (0.8 N) a pixel — and `bumpScale` 0.4 N makes each pixel's
        // forward difference exactly 1.
        let ramp = Arc::new(Texture::new(2, 1, TextureFormat::Rgba8Unorm, vec![0, 0, 0, 255, 255, 255, 255, 255]));
        let bump = || Some((ramp.clone(), 0.4 * N as f32));
        let flat = texel([128, 128, 255, 255]);
        let cases: [(&str, Maps, Lighting, [u8; 3]); 10] = [
            ("light map", Maps { light: Some((red(), PI)), ..Default::default() }, Lighting::None, [255, 0, 0]),
            (
                "light map under a black aoMap",
                Maps { light: Some((white(), PI)), ao: Some((black(), 1.0)), ..Default::default() },
                Lighting::None,
                [0, 0, 0],
            ),
            ("bump map, light from the left", Maps { bump: bump(), ..Default::default() }, Lighting::FromLeft, [255, 255, 255]),
            (
                "bump map under a normal map",
                Maps { bump: bump(), normal: Some(flat), ..Default::default() },
                Lighting::FromLeft,
                if name == "toon" { [255, 255, 255] } else { [219, 219, 219] },
            ),
            ("normal map, direct", Maps { normal: Some(tilted), ..Default::default() }, Lighting::Direct, tilted_want),
            ("map", Maps { map: Some(red()), ..Default::default() }, Lighting::Ambient, [255, 0, 0]),
            ("ao, ambient", Maps { ao: Some((black(), 1.0)), ..Default::default() }, Lighting::Ambient, [0, 0, 0]),
            ("ao 0.5, ambient", Maps { ao: Some((black(), 0.5)), ..Default::default() }, Lighting::Ambient, [188, 188, 188]),
            ("ao, direct", Maps { ao: Some((black(), 1.0)), ..Default::default() }, Lighting::Direct, [255, 255, 255]),
            ("emissive map", Maps { emissive: Some(red()), ..Default::default() }, Lighting::None, [255, 0, 0]),
        ];
        for (case, maps, lighting, want) in cases {
            // Toon has no cos 45° — its ramp is 1 above n·l = 0.4 — so the
            // "ignored under a normal map" case cannot tell it apart there.
            let got = centre(&render(&mut scene(build(maps), lighting), &cam));
            if !near(got, want) {
                failures.push(format!("{label} {name} {case}: got {got:?}, three.js gives {want:?}"));
            }
        }
    }
    failures.extend(more(label, render, &cam));
    failures
}

/// The specular map on Phong, and the light map on Basic and Standard.
fn more(label: &str, render: &mut dyn FnMut(&mut Scene, &dyn Camera) -> Vec<u8>, cam: &dyn Camera) -> Vec<String> {
    use threers::materials::{BasicMaterial, StandardMaterial};
    let mut failures = Vec::new();
    let mut check = |case: &str, m: Material, lighting: Lighting, want: [u8; 3]| {
        let got = centre(&render(&mut scene(m, lighting), cam));
        if !near(got, want) {
            failures.push(format!("{label} {case}: got {got:?}, three.js gives {want:?}"));
        }
    };
    let shiny = |map: Option<Arc<Texture>>| {
        let mut x = PhongMaterial::new(Color::new(0.0, 0.0, 0.0));
        x.specular = Color::WHITE;
        x.shininess = 30.0;
        x.specular_map = map;
        Material::from(x)
    };
    check("phong highlight", shiny(None), Lighting::Direct, [255, 255, 255]);
    check("phong highlight under a specularMap with red 0", shiny(Some(texel([0, 255, 255, 255]))), Lighting::Direct, [0, 0, 0]);
    let mut basic = BasicMaterial::new(Color::WHITE);
    basic.light_map = Some(texel([255, 0, 0, 255]));
    basic.light_map_intensity = PI;
    check("basic light map", basic.into(), Lighting::None, [255, 0, 0]);
    // three.js `meshmatcap.glsl`: the colour map multiplies the diffuse colour
    // before the (white) matcap.
    let mut matcap = threers::materials::MatcapMaterial::new(texel([255, 255, 255, 255]));
    matcap.map = Some(texel([255, 0, 0, 255]));
    check("matcap map", matcap.into(), Lighting::None, [255, 0, 0]);
    let mut standard = StandardMaterial::new(Color::WHITE);
    standard.light_map = Some(texel([255, 0, 0, 255]));
    standard.light_map_intensity = PI;
    check("standard light map", standard.into(), Lighting::None, [255, 0, 0]);
    failures
}
