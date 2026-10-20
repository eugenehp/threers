//! MeshPhysicalMaterial's dielectric reflectance against three.js r165's
//! formulas: `specularColor = min(((ior − 1)/(ior + 1))² · specularColor, 1) ·
//! specularIntensity` at normal incidence and `specularF90 =
//! specularIntensity` at grazing (`lights_physical_fragment`, metalness 0).
//!
//! A flat black quad faces an orthographic camera, lit by a directional light
//! shining straight at it, so the normal, the light, the view and the half
//! vector coincide and the flat face has no `geometryRoughness`. Only the
//! specular lobe is left, and three.js's `BRDF_GGX` reduces to
//!
//! ```text
//! F_Schlick(F0, F90, 1) · V · D,   V = ¼,   D = 1 / (π α²),   α = roughness²
//! ```
//!
//! times the light's intensity. Each case is computed here from those
//! formulas, not read off another renderer.
//!
//! Shared by `tests/physical.rs` (wgpu) and `tests/metal_backend.rs`.
#![allow(dead_code)]

use threers::cameras::{Camera, OrthographicCamera};
use threers::core::{Mesh, Object3D};
use threers::geometries::PlaneGeometry;
use threers::lights::{DirectionalLight, Light};
use threers::materials::PhysicalMaterial;
use threers::math::{Color, Vector3};
use threers::scene::Scene;

pub const N: u32 = 32;
const ROUGHNESS: f64 = 0.5;
const INTENSITY: f64 = 3.0;

fn srgb(c: f64) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let e = if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (e * 255.0).round() as u8
}

/// The expected pixel for `ior`, `specularIntensity` and `specularColor`.
fn expected(ior: f64, intensity: f64, tint: [f64; 3]) -> [u8; 3] {
    let r0 = ((ior - 1.0) / (ior + 1.0)).powi(2);
    let f90 = intensity;
    // `F_Schlick` at v·h = 1: `exp2((−5.55473 − 6.98316) · 1)`.
    let fresnel = 2f64.powf(-5.55473 - 6.98316);
    let alpha = ROUGHNESS * ROUGHNESS;
    let vd = 0.25 / (std::f64::consts::PI * alpha * alpha);
    tint.map(|t| {
        let f0 = (r0 * t).min(1.0) * intensity;
        srgb((f0 * (1.0 - fresnel) + f90 * fresnel) * vd * INTENSITY)
    })
}

pub fn scene(ior: f32, intensity: f32, tint: Color) -> Scene {
    let mut m = PhysicalMaterial::new(Color::BLACK);
    m.roughness = ROUGHNESS as f32;
    m.metalness = 0.0;
    m.ior = ior;
    m.specular_intensity = intensity;
    m.specular_color = tint;
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    let l = DirectionalLight::new(Color::WHITE, INTENSITY as f32)
        .with_direction(Vector3::new(0.0, 0.0, -1.0));
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

/// Every failure, as a line; empty when each case reflects as three.js does.
pub fn check(
    label: &str,
    render: &mut dyn FnMut(&mut Scene, &dyn Camera) -> Vec<u8>,
) -> Vec<String> {
    let cam = camera();
    let cases: [(&str, f32, f32, [f32; 3]); 5] = [
        ("ior 1.5 (F0 0.04)", 1.5, 1.0, [1.0, 1.0, 1.0]),
        ("ior 2.33", 2.33, 1.0, [1.0, 1.0, 1.0]),
        ("specularIntensity 0.5", 1.5, 0.5, [1.0, 1.0, 1.0]),
        ("specularColor", 1.5, 1.0, [1.0, 0.5, 0.25]),
        ("ior 4 clamps F0 · color at 1", 4.0, 1.0, [1.0, 1.0, 1.0]),
    ];
    let mut failures = Vec::new();
    for (name, ior, intensity, tint) in cases {
        let want = expected(ior as f64, intensity as f64, tint.map(f64::from));
        let rgba = render(
            &mut scene(ior, intensity, Color::new(tint[0], tint[1], tint[2])),
            &cam,
        );
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
