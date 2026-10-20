//! three.js r165's image-based light on `MeshStandardMaterial`: whose
//! environment, how bright, which way round.
//!
//! A mirror quad (metalness 1, roughness 0) faces an orthographic camera, so
//! it reflects the +z direction head on and +x turned 45° about y. The
//! environment cube has one colour a face — +X red, −X green, +Y blue, −Y
//! yellow, +Z magenta, −Z cyan — or is one colour throughout. There are no
//! lights, so every pixel is image-based light.
//!
//! - **lookup**: head on, magenta (+Z). Turned 45°, the reflection runs +x,
//!   and three.js reads its PMREM of a cube texture mirrored in x
//!   (`flipEnvMap = −1`): green (−X);
//! - **`scene.environmentRotation.y`** a quarter turn, head on: three.js turns
//!   the lookup by the Euler negated, (0, 0, 1) to (−1, 0, 0), mirrored to
//!   +X: red;
//! - **`scene.environmentIntensity`** 0: black;
//! - **`envMap`** replaces `scene.environment` (blue over green: blue), applies
//!   with no scene environment at all (blue), takes its own
//!   `envMapIntensity` (0: black, whatever the scene's) and its own
//!   `envMapRotation` (a quarter turn: red).
//!
//! Colours are judged by which channels dominate: a mirror's split-sum
//! response is not exactly 1.
//!
//! Shared by `tests/pbr_env.rs` (wgpu) and `tests/metal_backend.rs`.
#![allow(dead_code)]

use std::sync::Arc;
use threers::cameras::{Camera, OrthographicCamera};
use threers::core::{Mesh, Object3D};
use threers::geometries::PlaneGeometry;
use threers::materials::StandardMaterial;
use threers::math::{Color, Euler, Vector3};
use threers::scene::Scene;
use threers::textures::{CubeTexture, TextureFormat};

pub const N: u32 = 32;

const FACES: [[u8; 3]; 6] = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]];

fn cube(faces: [[u8; 3]; 6]) -> Arc<CubeTexture> {
    Arc::new(CubeTexture::new(8, TextureFormat::Rgba8UnormSrgb, faces.map(|c| [c[0], c[1], c[2], 255].repeat(64))))
}

fn uniform(c: [u8; 3]) -> Arc<CubeTexture> {
    cube([c; 6])
}

fn camera() -> OrthographicCamera {
    let mut cam = OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    cam.position = Vector3::new(0.0, 0.0, 5.0);
    cam.look_at(Vector3::ZERO);
    cam
}

fn mirror() -> StandardMaterial {
    StandardMaterial::new(Color::WHITE).with_roughness(0.0).with_metalness(1.0)
}

fn scene(material: StandardMaterial, env: Option<Arc<CubeTexture>>, turn: f32) -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.environment = env;
    let mut quad = Object3D::mesh(Mesh::new(PlaneGeometry::new(1.6, 1.6), material.into()));
    quad.set_rotation(Euler::new(0.0, turn, 0.0));
    scene.add(quad);
    scene
}

/// Which of r, g, b are clearly lit: above 60 and within 40% of the brightest.
fn lit(p: [u8; 3]) -> [bool; 3] {
    let max = *p.iter().max().unwrap() as f32;
    p.map(|v| v > 60 && v as f32 > 0.6 * max)
}

fn centre(rgba: &[u8]) -> [u8; 3] {
    let i = (((N / 2) * N + N / 2) * 4) as usize;
    [rgba[i], rgba[i + 1], rgba[i + 2]]
}

/// Every failure, as a line; empty when each case matches three.js.
pub fn check(label: &str, render: &mut dyn FnMut(&mut Scene, &dyn Camera) -> Vec<u8>) -> Vec<String> {
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};
    let cam = camera();
    let mut failures = Vec::new();
    let mut expect = |case: &str, s: &mut Scene, want: [bool; 3]| {
        let got = centre(&render(s, &cam));
        if lit(got) != want {
            failures.push(format!("{label} {case}: got {got:?}, three.js lights {want:?}"));
        }
    };
    const MAGENTA: [bool; 3] = [true, false, true];
    const GREEN: [bool; 3] = [false, true, false];
    const RED: [bool; 3] = [true, false, false];
    const BLUE: [bool; 3] = [false, false, true];
    const BLACK: [bool; 3] = [false, false, false];

    expect("scene environment, head on", &mut scene(mirror(), Some(cube(FACES)), 0.0), MAGENTA);
    expect("scene environment, turned 45° (flipEnvMap)", &mut scene(mirror(), Some(cube(FACES)), FRAC_PI_4), GREEN);
    let mut s = scene(mirror(), Some(cube(FACES)), 0.0);
    s.environment_rotation = Euler::new(0.0, FRAC_PI_2, 0.0);
    expect("environmentRotation.y 90°", &mut s, RED);
    let mut s = scene(mirror(), Some(cube(FACES)), 0.0);
    s.environment_intensity = 0.0;
    expect("environmentIntensity 0", &mut s, BLACK);

    let own = |intensity: f32, rotation_y: f32, faces: Arc<CubeTexture>| {
        let mut m = mirror();
        m.env_map = Some(faces);
        m.env_map_intensity = intensity;
        m.env_map_rotation = Euler::new(0.0, rotation_y, 0.0);
        m
    };
    let green = || Some(uniform([0, 255, 0]));
    expect("envMap over scene.environment", &mut scene(own(1.0, 0.0, uniform([0, 0, 255])), green(), 0.0), BLUE);
    expect("envMap with no scene environment", &mut scene(own(1.0, 0.0, uniform([0, 0, 255])), None, 0.0), BLUE);
    expect("envMapIntensity 0", &mut scene(own(0.0, 0.0, uniform([0, 0, 255])), green(), 0.0), BLACK);
    expect("envMapRotation.y 90°", &mut scene(own(1.0, FRAC_PI_2, cube(FACES)), None, 0.0), RED);
    failures
}
