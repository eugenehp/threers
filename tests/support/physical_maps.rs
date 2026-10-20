//! MeshPhysicalMaterial's maps, each against the scalar it scales.
//!
//! three.js r165's `lights_physical_fragment` multiplies each map's channel
//! into its scalar: `specularIntensityMap`'s alpha, `specularColorMap`'s
//! colour, `clearcoatMap`'s red, `clearcoatRoughnessMap`'s green,
//! `iridescenceMap`'s red, `sheenColorMap`'s colour, `sheenRoughnessMap`'s
//! alpha; `iridescenceThicknessMap`'s green picks the film's thickness between
//! the ends of `iridescenceThicknessRange`; and `anisotropyMap` turns the
//! anisotropy direction by its red and green (`2·rg − 1`) and scales its
//! strength by its blue. So a material whose map is one uniform texel draws
//! exactly what the same material does with that texel folded into the
//! scalar — which checks each map's channel, sampler, transform and binding
//! without another renderer as the reference.
//!
//! The folded material keeps the map, white (the multiplier's identity), so
//! both draw through the same shader variant. Two variants are compiled
//! apart, and that alone moves an anisotropic highlight on the sphere's
//! triangle edges, where its screen-space tangent frame jumps between
//! triangles — this test is about the maps, not about that.
//!
//! A sphere under a key light, a rim light and ambient light, compared pixel
//! for pixel within one level. Shared by `tests/physical.rs` (wgpu) and
//! `tests/metal_backend.rs`.
#![allow(dead_code)]

use std::sync::Arc;
use threers::cameras::{Camera, PerspectiveCamera};
use threers::core::{Mesh, Object3D};
use threers::geometries::SphereGeometry;
use threers::lights::{AmbientLight, DirectionalLight, Light};
use threers::materials::PhysicalMaterial;
use threers::math::{Color, Vector3};
use threers::scene::Scene;
use threers::textures::{Texture, TextureFormat};

pub const N: u32 = 96;

fn texel(rgba: [u8; 4]) -> Option<Arc<Texture>> {
    Some(Arc::new(Texture::new(
        1,
        1,
        TextureFormat::Rgba8Unorm,
        rgba.to_vec(),
    )))
}

/// Byte `b` as the value the sampler reads.
fn v(b: u8) -> f32 {
    b as f32 / 255.0
}

pub fn camera() -> PerspectiveCamera {
    let mut cam = PerspectiveCamera::new(30.0, 1.0, 0.1, 100.0);
    cam.position = Vector3::new(0.0, 0.3, 4.2);
    cam.look_at(Vector3::ZERO);
    cam
}

fn scene(m: PhysicalMaterial) -> Scene {
    let mut s = Scene::new();
    s.background = Color::from_hex(0x000000);
    s.add_light(AmbientLight::new(Color::WHITE, 0.25));
    let key =
        DirectionalLight::new(Color::WHITE, 2.5).with_direction(Vector3::new(-3.0, -4.0, -5.0));
    let rim = DirectionalLight::new(Color::WHITE, 1.5).with_direction(Vector3::new(4.0, -1.0, 2.0));
    s.add(Object3D::light(Light::Directional(key)));
    s.add(Object3D::light(Light::Directional(rim)));
    s.add(Object3D::mesh(Mesh::new(
        SphereGeometry::new(1.0, 64, 32),
        m.into(),
    )));
    s
}

fn base(color: u32, f: impl FnOnce(&mut PhysicalMaterial)) -> PhysicalMaterial {
    let mut m = PhysicalMaterial::new(Color::from_hex(color));
    m.roughness = 0.35;
    f(&mut m);
    m
}

/// `(name, with the map, with the texel folded into the scalar)`.
pub fn cases() -> Vec<(&'static str, PhysicalMaterial, PhysicalMaterial)> {
    let (a, b, c) = (51u8, 102u8, 204u8);
    vec![
        (
            "specularIntensityMap",
            base(0x4060a0, |m| {
                m.specular_intensity_map = texel([255, 255, 255, b])
            }),
            base(0x4060a0, |m| {
                m.specular_intensity_map = texel([255; 4]);
                m.specular_intensity = v(b);
            }),
        ),
        (
            "specularColorMap",
            base(0x4060a0, |m| {
                m.ior = 2.0;
                m.specular_color_map = texel([c, b, a, 255]);
            }),
            base(0x4060a0, |m| {
                m.ior = 2.0;
                m.specular_color_map = texel([255; 4]);
                m.specular_color = Color::new(v(c), v(b), v(a));
            }),
        ),
        (
            "clearcoatMap",
            base(0xc03020, |m| {
                m.clearcoat = 1.0;
                m.clearcoat_roughness = 0.1;
                m.clearcoat_map = texel([b, 255, 255, 255]);
            }),
            base(0xc03020, |m| {
                m.clearcoat_map = texel([255; 4]);
                m.clearcoat = v(b);
                m.clearcoat_roughness = 0.1;
            }),
        ),
        (
            "clearcoatRoughnessMap",
            base(0xc03020, |m| {
                m.clearcoat = 1.0;
                m.clearcoat_roughness = 1.0;
                m.clearcoat_roughness_map = texel([255, b, 255, 255]);
            }),
            base(0xc03020, |m| {
                m.clearcoat = 1.0;
                m.clearcoat_roughness_map = texel([255; 4]);
                m.clearcoat_roughness = v(b);
            }),
        ),
        (
            "iridescenceMap",
            base(0x202020, |m| {
                m.iridescence = 1.0;
                m.iridescence_map = texel([c, 255, 255, 255]);
            }),
            base(0x202020, |m| {
                m.iridescence_map = texel([255; 4]);
                m.iridescence = v(c);
            }),
        ),
        (
            "iridescenceThicknessMap",
            base(0x202020, |m| {
                m.iridescence = 1.0;
                m.iridescence_thickness_range = [100.0, 600.0];
                m.iridescence_thickness_map = texel([255, b, 255, 255]);
            }),
            // A white texel reads the top of the range.
            base(0x202020, |m| {
                m.iridescence = 1.0;
                m.iridescence_thickness_range = [100.0, 100.0 + 500.0 * v(b)];
                m.iridescence_thickness_map = texel([255; 4]);
            }),
        ),
        (
            "sheenColorMap",
            base(0x1a1a40, |m| {
                m.sheen = 1.0;
                m.sheen_color = Color::WHITE;
                m.sheen_roughness = 0.5;
                m.sheen_color_map = texel([c, b, a, 255]);
            }),
            base(0x1a1a40, |m| {
                m.sheen = 1.0;
                m.sheen_color = Color::new(v(c), v(b), v(a));
                m.sheen_roughness = 0.5;
                m.sheen_color_map = texel([255; 4]);
            }),
        ),
        (
            "sheenRoughnessMap",
            base(0x1a1a40, |m| {
                m.sheen = 1.0;
                m.sheen_color = Color::from_hex(0xffe0b0);
                m.sheen_roughness = 1.0;
                m.sheen_roughness_map = texel([255, 255, 255, c]);
            }),
            base(0x1a1a40, |m| {
                m.sheen = 1.0;
                m.sheen_color = Color::from_hex(0xffe0b0);
                m.sheen_roughness = v(c);
                m.sheen_roughness_map = texel([255; 4]);
            }),
        ),
        (
            // Direction (2·rg − 1) = (1, 0): along the anisotropy direction,
            // strength 0.4 — the same as anisotropy × 0.4.
            "anisotropyMap",
            base(0xd0d0d0, |m| {
                m.metalness = 1.0;
                m.anisotropy = 0.9;
                m.anisotropy_rotation = 0.4;
                m.anisotropy_map = texel([255, 128, b, 255]);
            }),
            // The same direction at full strength.
            base(0xd0d0d0, |m| {
                m.metalness = 1.0;
                m.anisotropy = 0.9 * v(b);
                m.anisotropy_rotation = 0.4;
                m.anisotropy_map = texel([255, 128, 255, 255]);
            }),
        ),
    ]
}

/// Every failure, as a line; empty when each map draws what its scalar does.
pub fn check(
    label: &str,
    render: &mut dyn FnMut(&mut Scene, &dyn Camera) -> Vec<u8>,
) -> Vec<String> {
    let cam = camera();
    let mut failures = Vec::new();
    for (name, mapped, folded) in cases() {
        let a = render(&mut scene(mapped), &cam);
        let b = render(&mut scene(folded), &cam);
        let (mut worst, mut lit) = (0i32, 0usize);
        for (pa, pb) in a.chunks(4).zip(b.chunks(4)) {
            if pb[..3].iter().any(|&c| c > 0) {
                lit += 1;
            }
            worst = worst.max(
                (0..3)
                    .map(|k| (pa[k] as i32 - pb[k] as i32).abs())
                    .max()
                    .unwrap(),
            );
        }
        // The anisotropy map's direction texel is 128, a hair off ½, so it
        // turns the direction by 1/255 of a quarter turn.
        let allowed = if name == "anisotropyMap" { 2 } else { 1 };
        if lit < 1000 {
            failures.push(format!("{label} {name}: only {lit} pixels drawn"));
        } else if worst > allowed {
            failures.push(format!("{label} {name}: {worst} levels from the same material with the texel folded into its scalar"));
        }
    }
    failures
}
