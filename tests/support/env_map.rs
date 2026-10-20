//! three.js `envMap` on `MeshBasicMaterial`, `MeshLambertMaterial` and
//! `MeshPhongMaterial` (`envmap_fragment`), against three.js r165's lookup and
//! combine rather than another backend.
//!
//! A quad faces an orthographic camera, so the view ray is the camera's
//! forward axis (0, 0, −1) everywhere. The cube's faces are one colour each:
//! +X red, −X green, +Y blue, −Y yellow, +Z magenta, −Z cyan.
//!
//! - **head-on**: the ray reflects to +z — magenta;
//! - **turned 45° about y**: it reflects to +x, which three.js looks up
//!   mirrored (`flipEnvMap = −1` for a cube texture): green. Unmirrored would
//!   be red;
//! - **`envMapRotation.y = 90°`**, head-on: +z turned to +x — red;
//! - **refraction mapping**, head-on: the ray passes straight through, −z —
//!   cyan;
//! - **combine**: `Mix` gives the environment; `Multiply` at `reflectivity`
//!   0.5 on white gives (1, ½, 1), 188 in green after the sRGB encode; `Add`
//!   on black gives the environment;
//! - **Lambert** with no light, `Mix`: the environment; **Phong** with a
//!   `specularMap` of red 0: `specularStrength` 0 removes the environment, black.
//!
//! Shared by `tests/env_map.rs` (wgpu) and `tests/metal_backend.rs`.
#![allow(dead_code)]

use std::sync::Arc;
use threers::cameras::{Camera, OrthographicCamera};
use threers::core::{Mesh, Object3D};
use threers::geometries::PlaneGeometry;
use threers::materials::{BasicMaterial, Combine, LambertMaterial, Material, PhongMaterial};
use threers::math::{Color, Euler, Vector3};
use threers::scene::Scene;
use threers::textures::{CubeMapping, CubeTexture, Texture, TextureFormat};

pub const N: u32 = 32;

const FACES: [[u8; 3]; 6] = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]];

fn cube(mapping: CubeMapping) -> Arc<CubeTexture> {
    let faces = FACES.map(|c| [c[0], c[1], c[2], 255].repeat(4 * 4));
    let mut c = CubeTexture::new(4, TextureFormat::Rgba8UnormSrgb, faces);
    c.mapping = mapping;
    Arc::new(c)
}

fn camera() -> OrthographicCamera {
    let mut cam = OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    cam.position = Vector3::new(0.0, 0.0, 5.0);
    cam.look_at(Vector3::ZERO);
    cam
}

fn scene(material: Material, turn: f32) -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    let mut quad = Object3D::mesh(Mesh::new(PlaneGeometry::new(1.6, 1.6), material));
    quad.set_rotation(Euler::new(0.0, turn, 0.0));
    scene.add(quad);
    scene
}

fn basic(color: Color, combine: Combine, reflectivity: f32, mapping: CubeMapping, rotation_y: f32) -> Material {
    let mut m = BasicMaterial::new(color);
    m.env_map = Some(cube(mapping));
    m.combine = combine;
    m.reflectivity = reflectivity;
    m.env_map_rotation = Euler::new(0.0, rotation_y, 0.0);
    m.into()
}

fn centre(rgba: &[u8]) -> [u8; 3] {
    let i = (((N / 2) * N + N / 2) * 4) as usize;
    [rgba[i], rgba[i + 1], rgba[i + 2]]
}

fn near(got: [u8; 3], want: [u8; 3]) -> bool {
    (0..3).all(|k| (got[k] as i32 - want[k] as i32).abs() <= 2)
}

/// Every failure, as a line; empty when each case shades as three.js.
pub fn check(label: &str, render: &mut dyn FnMut(&mut Scene, &dyn Camera) -> Vec<u8>) -> Vec<String> {
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};
    let cam = camera();
    let white = Color::WHITE;
    let black = Color::new(0.0, 0.0, 0.0);
    let reflect = CubeMapping::Reflection;
    let mut lambert = LambertMaterial::new(white);
    lambert.env_map = Some(cube(reflect));
    lambert.combine = Combine::Mix;
    let mut phong = PhongMaterial::new(white);
    phong.env_map = Some(cube(reflect));
    phong.combine = Combine::Mix;
    phong.specular_map = Some(Arc::new(Texture::new(1, 1, TextureFormat::Rgba8Unorm, vec![0, 255, 255, 255])));
    let cases: Vec<(&str, Material, f32, [u8; 3])> = vec![
        ("basic, head-on", basic(white, Combine::Mix, 1.0, reflect, 0.0), 0.0, [255, 0, 255]),
        ("basic, turned 45° (flipEnvMap)", basic(white, Combine::Mix, 1.0, reflect, 0.0), FRAC_PI_4, [0, 255, 0]),
        ("basic, envMapRotation.y 90°", basic(white, Combine::Mix, 1.0, reflect, FRAC_PI_2), 0.0, [255, 0, 0]),
        ("basic, refraction mapping", basic(white, Combine::Mix, 1.0, CubeMapping::Refraction, 0.0), 0.0, [0, 255, 255]),
        ("basic, multiply at reflectivity 0.5", basic(white, Combine::Multiply, 0.5, reflect, 0.0), 0.0, [255, 188, 255]),
        ("basic, add on black", basic(black, Combine::Add, 1.0, reflect, 0.0), 0.0, [255, 0, 255]),
        ("lambert, no light, mix", lambert.into(), 0.0, [255, 0, 255]),
        ("phong, specularMap red 0", phong.into(), 0.0, [0, 0, 0]),
    ];
    let mut failures = Vec::new();
    for (case, m, turn, want) in cases {
        let got = centre(&render(&mut scene(m, turn), &cam));
        if !near(got, want) {
            failures.push(format!("{label} {case}: got {got:?}, three.js gives {want:?}"));
        }
    }
    failures
}
