//! MeshToonMaterial's `gradientMap` against three.js r165's
//! `getGradientIrradiance`: each light's irradiance times the map's rgb at
//! `(n·l/2 + ½, 0)`, through `BRDF_Lambert` (÷π); without a map, 0.7 below
//! `coord = 0.7` and 1.0 above.
//!
//! A flat white quad faces an orthographic camera under one directional light
//! of intensity π, tilted so `n·l` is known — so the pixel is the ramp's value
//! at `coord`, encoded to sRGB. A flat face has no derivative, so the map is
//! read at level 0. Computed here from the formula and the texels, not read
//! off another renderer.
//!
//! Shared by `tests/toon.rs` (wgpu) and `tests/metal_backend.rs`.
#![allow(dead_code)]

use std::sync::Arc;
use threers::cameras::{Camera, OrthographicCamera};
use threers::core::{Mesh, Object3D};
use threers::geometries::PlaneGeometry;
use threers::lights::{DirectionalLight, Light};
use threers::materials::ToonMaterial;
use threers::math::{Color, Vector3};
use threers::scene::Scene;
use threers::textures::{Texture, TextureFilter, TextureFormat};

pub const N: u32 = 32;

fn srgb(c: f64) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let e = if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (e * 255.0).round() as u8
}

fn decode(b: u8) -> f64 {
    let c = b as f64 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Four bands of colour, sampled nearest.
const RAMP: [[u8; 3]; 4] = [[40, 10, 10], [90, 60, 20], [160, 140, 60], [250, 240, 200]];

fn ramp(format: TextureFormat, filter: TextureFilter, texels: &[[u8; 3]]) -> Arc<Texture> {
    let data = texels
        .iter()
        .flat_map(|t| [t[0], t[1], t[2], 255])
        .collect();
    let mut t = Texture::new(texels.len() as u32, 1, format, data);
    t.mag_filter = filter;
    t.min_filter = filter;
    t.mipmap_filter = None;
    Arc::new(t)
}

pub fn scene(n_dot_l: f32, map: Option<Arc<Texture>>) -> Scene {
    let mut m = ToonMaterial::new(Color::WHITE);
    m.gradient_map = map;
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    // Towards the light `(√(1 − c²), 0, c)`; it travels the other way.
    let side = (1.0 - n_dot_l * n_dot_l).max(0.0).sqrt();
    let l = DirectionalLight::new(Color::WHITE, std::f32::consts::PI)
        .with_direction(Vector3::new(-side, 0.0, -n_dot_l));
    scene.add(Object3D::light(Light::Directional(l)));
    scene.add(Object3D::mesh(Mesh::new(
        PlaneGeometry::new(1.6, 1.6),
        m.into(),
    )));
    scene
}

pub fn camera() -> OrthographicCamera {
    let mut cam = OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    cam.position = Vector3::new(0.0, 0.0, 5.0);
    cam.look_at(Vector3::ZERO);
    cam
}

/// `(name, n·l, map, expected linear rgb)`.
type Case = (String, f32, Option<Arc<Texture>>, [f64; 3]);

fn cases() -> Vec<Case> {
    let lin = |t: [u8; 3]| t.map(|b| b as f64 / 255.0);
    let mut out = Vec::new();
    let nearest = ramp(TextureFormat::Rgba8Unorm, TextureFilter::Nearest, &RAMP);
    // coord 0.2, 0.5, 0.65, 0.95: texels 0, 2, 2, 3 of 4.
    for (c, texel) in [(-0.6f32, 0usize), (0.0, 2), (0.3, 2), (0.9, 3)] {
        out.push((
            format!("nearest ramp at n·l {c}"),
            c,
            Some(nearest.clone()),
            lin(RAMP[texel]),
        ));
    }
    // An sRGB ramp is decoded, as three.js decodes an SRGBColorSpace texture.
    let srgb_ramp = ramp(TextureFormat::Rgba8UnormSrgb, TextureFilter::Nearest, &RAMP);
    out.push((
        "sRGB ramp".into(),
        0.3,
        Some(srgb_ramp),
        RAMP[2].map(decode),
    ));
    // Linear filtering between two texels, centres at ¼ and ¾:
    // coord 0.65 is 0.8 of the way from black to white.
    let two = ramp(
        TextureFormat::Rgba8Unorm,
        TextureFilter::Linear,
        &[[0; 3], [255; 3]],
    );
    out.push(("linear ramp".into(), 0.3, Some(two), [0.8; 3]));
    // No gradient map: 0.7 below coord 0.7, 1.0 above.
    out.push(("no map, dark side".into(), 0.3, None, [0.7; 3]));
    out.push(("no map, lit side".into(), 0.6, None, [1.0; 3]));
    out
}

/// Every failure, as a line; empty when each case shades as three.js does.
pub fn check(
    label: &str,
    render: &mut dyn FnMut(&mut Scene, &dyn Camera) -> Vec<u8>,
) -> Vec<String> {
    let cam = camera();
    let mut failures = Vec::new();
    for (name, c, map, want) in cases() {
        let want = want.map(srgb);
        let rgba = render(&mut scene(c, map), &cam);
        let i = ((N / 2 * N + N / 2) * 4) as usize;
        let got = [rgba[i], rgba[i + 1], rgba[i + 2]];
        if (0..3).any(|k| (got[k] as i32 - want[k] as i32).abs() > 2) {
            failures.push(format!(
                "{label} {name}: got {got:?}, three.js gives {want:?}"
            ));
        }
    }
    failures
}
