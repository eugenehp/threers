//! three.js `alphaTest` and `transparent` on every material kind.
//!
//! A white quad faces the camera over a blue background, under white ambient
//! light, so every kind draws something whose red channel the background
//! lacks. Three cases, each against three.js's rule rather than another
//! backend:
//!
//! - **cut**: opacity 0.3 under `alphaTest` 0.5 — `alphatest_fragment`
//!   discards it, so the background shows;
//! - **kept**: opacity 0.3 over `alphaTest` 0.2 with `transparent: false` —
//!   drawn, and opaquely: three.js only blends a material marked transparent,
//!   so the pixel equals the same material drawn at opacity 1;
//! - **map**: a texture whose left half has alpha 0 — the map's alpha multiplies
//!   the diffuse alpha before the test, so the left half is cut and the right
//!   half drawn (for the kinds that take a `map`);
//! - **alphaMap**: an `alphaMap` whose left half has green 0 (and red 255) and
//!   right half green 255 (and red 0) — `alphamap_fragment` scales the alpha
//!   by the green channel, so the left half is cut and the right drawn, for
//!   every kind but normal (which has no `alphaMap` in three.js);
//! - **alphaMap, blended**: a transparent basic quad under a uniform `alphaMap`
//!   of green 128 blends half its white into the blue background, as three.js
//!   blends the encoded colours: (128, 128, 255).
//!
//! Shared by `tests/alpha_test.rs` (wgpu) and `tests/metal_backend.rs`.
#![allow(dead_code)]

use std::sync::Arc;
use threers::cameras::{Camera, OrthographicCamera};
use threers::core::{Mesh, Object3D, Sprite};
use threers::geometries::PlaneGeometry;
use threers::lights::AmbientLight;
use threers::materials::{
    BasicMaterial, LambertMaterial, MatcapMaterial, Material, NormalMaterial, PhongMaterial,
    PhysicalMaterial, SpriteMaterial, StandardMaterial, ToonMaterial,
};
use threers::math::{Color, Vector3};
use threers::scene::Scene;
use threers::textures::{Texture, TextureFilter, TextureFormat};

pub const N: u32 = 64;

pub fn camera() -> OrthographicCamera {
    let mut cam = OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    cam.position = Vector3::new(0.0, 0.0, 5.0);
    cam.look_at(Vector3::ZERO);
    cam
}

fn white() -> Arc<Texture> {
    Arc::new(Texture::new(1, 1, TextureFormat::Rgba8Unorm, vec![255; 4]))
}

/// Left texel alpha 0, right texel opaque, sampled nearest so the halves stay
/// clean.
fn half_cut() -> Arc<Texture> {
    let mut t = Texture::new(2, 1, TextureFormat::Rgba8Unorm, vec![255, 255, 255, 0, 255, 255, 255, 255]);
    t.mag_filter = TextureFilter::Nearest;
    t.min_filter = TextureFilter::Nearest;
    Arc::new(t)
}

/// Left texel green 0 (red 255), right texel green 255 (red 0): the alpha
/// map's green channel is what counts.
fn green_cut() -> Arc<Texture> {
    let mut t = Texture::new(2, 1, TextureFormat::Rgba8Unorm, vec![255, 0, 255, 255, 0, 255, 0, 255]);
    t.mag_filter = TextureFilter::Nearest;
    t.min_filter = TextureFilter::Nearest;
    Arc::new(t)
}

/// The kinds under test: a name, whether it takes a `map`, and a builder for
/// the material at the given opacity, alpha test and map.
pub type Build = fn(f32, f32, Option<Arc<Texture>>) -> Material;

pub const KINDS: [(&str, bool, Build); 10] = [
    ("basic", true, |o, t, map| {
        let mut m = BasicMaterial::new(Color::WHITE);
        m.opacity = o;
        m.map = map;
        let mut m: Material = m.into();
        m.set_alpha_test(t);
        m.set_transparent(false);
        m
    }),
    ("lambert", false, |o, t, _| finish(LambertMaterial::new(Color::WHITE).into(), o, t)),
    ("phong", false, |o, t, _| finish(PhongMaterial::new(Color::WHITE).into(), o, t)),
    ("standard", true, |o, t, map| {
        let mut m = StandardMaterial::new(Color::WHITE);
        m.map = map;
        finish(m.into(), o, t)
    }),
    ("physical", true, |o, t, map| {
        let mut m = PhysicalMaterial::new(Color::WHITE);
        m.map = map;
        finish(m.into(), o, t)
    }),
    ("toon", false, |o, t, _| finish(ToonMaterial::new(Color::WHITE).into(), o, t)),
    ("matcap", true, |o, t, map| {
        let mut m = MatcapMaterial::new(white());
        m.map = map;
        finish(m.into(), o, t)
    }),
    ("normal", false, |o, t, _| finish(NormalMaterial::default().into(), o, t)),
    ("sprite", true, |o, t, map| {
        let mut m = SpriteMaterial::new(Color::WHITE);
        m.map = map;
        finish(m.into(), o, t)
    }),
    ("sprite-plain", false, |o, t, _| finish(SpriteMaterial::new(Color::WHITE).into(), o, t)),
];

fn finish(mut m: Material, opacity: f32, alpha_test: f32) -> Material {
    set_opacity(&mut m, opacity);
    m.set_alpha_test(alpha_test);
    m.set_transparent(false);
    m
}

fn set_opacity(m: &mut Material, o: f32) {
    match m {
        Material::Basic(x) => x.opacity = o,
        Material::Lambert(x) => x.opacity = o,
        Material::Phong(x) => x.opacity = o,
        Material::Standard(x) => x.opacity = o,
        Material::Physical(x) => x.opacity = o,
        Material::Toon(x) => x.opacity = o,
        Material::Matcap(x) => x.opacity = o,
        Material::Normal(x) => x.opacity = o,
        Material::Sprite(x) => x.opacity = o,
        _ => unreachable!(),
    }
}

pub fn scene(material: Material) -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x0000ff);
    scene.add_light(AmbientLight::new(Color::WHITE, 1.0));
    let obj = if matches!(material, Material::Sprite(_)) {
        let mut s = Object3D::sprite(Sprite::new(material));
        s.scale = Vector3::new(1.6, 1.6, 1.0);
        s
    } else {
        Object3D::mesh(Mesh::new(PlaneGeometry::new(1.6, 1.6), material))
    };
    scene.add(obj);
    scene
}

fn px(rgba: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * N + x) * 4) as usize;
    [rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]]
}

/// Drawn: the quad's red channel, which the blue background lacks.
fn drawn(p: [u8; 4]) -> bool {
    p[0] > 40
}

/// Every failure, as a line; empty when all three cases hold for every kind.
pub fn check(label: &str, render: &mut dyn FnMut(&mut Scene, &dyn Camera) -> Vec<u8>) -> Vec<String> {
    let cam = camera();
    let (c, left, right) = (N / 2, N / 2 - N / 5, N / 2 + N / 5);
    let mut failures = Vec::new();
    for (name, has_map, build) in KINDS {
        let cut = render(&mut scene(build(0.3, 0.5, None)), &cam);
        // r165's `meshnormal` fragment has no `alphatest_fragment` and writes
        // alpha 1: a normal material ignores `alphaTest`.
        if name == "normal" {
            if !drawn(px(&cut, c, c)) {
                failures.push(format!("{label} normal: alphaTest cut a MeshNormalMaterial, which three.js never does"));
            }
        } else if drawn(px(&cut, c, c)) {
            failures.push(format!("{label} {name}: opacity 0.3 under alphaTest 0.5 still drew {:?}", px(&cut, c, c)));
        }
        let kept = render(&mut scene(build(0.3, 0.2, None)), &cam);
        let full = render(&mut scene(build(1.0, 0.0, None)), &cam);
        let (k, f) = (px(&kept, c, c), px(&full, c, c));
        if !drawn(k) {
            failures.push(format!("{label} {name}: opacity 0.3 over alphaTest 0.2 was cut: {k:?}"));
        } else if (0..3).any(|i| (k[i] as i32 - f[i] as i32).abs() > 1) {
            failures.push(format!(
                "{label} {name}: transparent false at opacity 0.3 blended ({k:?}) instead of drawing opaquely ({f:?})"
            ));
        }
        if name != "normal" {
            let mut m = build(1.0, 0.5, None);
            *m.map_slot_mut("alphaMap").expect("an alphaMap slot") = Some(green_cut());
            let masked = render(&mut scene(m), &cam);
            if drawn(px(&masked, left, c)) {
                failures.push(format!("{label} {name}: the alphaMap's green-0 half was not cut"));
            }
            if !drawn(px(&masked, right, c)) {
                failures.push(format!("{label} {name}: the alphaMap's green-255 half was cut"));
            }
        }
        if has_map {
            let mapped = render(&mut scene(build(1.0, 0.5, Some(half_cut()))), &cam);
            if drawn(px(&mapped, left, c)) {
                failures.push(format!("{label} {name}: the map's alpha-0 half was not cut"));
            }
            if !drawn(px(&mapped, right, c)) {
                failures.push(format!("{label} {name}: the map's opaque half was cut"));
            }
        }
    }
    let mut m = BasicMaterial::new(Color::WHITE);
    m.transparent = true;
    m.alpha_map = Some(Arc::new(Texture::new(1, 1, TextureFormat::Rgba8Unorm, vec![0, 128, 0, 255])));
    let blended = px(&render(&mut scene(m.into()), &cam), c, c);
    if (0..3).any(|i| (blended[i] as i32 - [128, 128, 255][i]).abs() > 2) {
        failures.push(format!("{label} basic: alphaMap green 128, blended, gave {blended:?}; three.js gives (128, 128, 255)"));
    }
    failures
}
