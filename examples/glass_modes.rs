//! The ways threers draws glass, side by side, against a path-traced reference.
//!
//! ```text
//! cargo run --release --features raytrace --example glass_modes
//! ```
//!
//! Writes `out/glass_modes.png`: one row per kind of glass, one column per
//! renderer. The same scene throughout: a glass ball over a checker floor in
//! front of a striped wall, in a studio environment, with a sun casting
//! shadows.
//!
//! | Column | What it is | Good for |
//! |---|---|---|
//! | `TransparencyMode::Blend` | three.js's transmission: the opaque scene, captured, read along the ray refracted through `thickness` | three.js parity |
//! | `TransparencyMode::Refract` | the same physics over a sharper capture, with the path length measured from the back faces | the most convincing solid glass |
//! | `TransparencyMode::Glass` | thin glass: one blended layer with full-strength reflections | windows, bottles, visors |
//! | path traced | `RaytraceRenderer`, the reference | |
//!
//! | Row | Material |
//! |---|---|
//! | clear | `transmission 1`, `ior 1.5`, `thickness` the ball's diameter |
//! | tinted | plus `attenuationColor` / `attenuationDistance` |
//! | frosted | `roughness 0.35` |
//! | dispersive | `dispersion 1` |
//! | thin-walled | `thickness 0`, three.js's thin-walled glass: nothing bends |
//!
//! Lights reach glass as highlights in every mode. Solid glass casts a full
//! shadow (the light it lets through goes into a caustic, which only the path
//! tracer draws); thin glass, which bends nothing, casts a light one. See
//! `docs/materials.md`.

use std::sync::Arc;

use threers::materials::TransparencyMode;
use threers::raytrace::{RaytraceRenderer, RaytraceSettings};
use threers::textures::Texture;
use threers::*;

const W: u32 = 320;
const H: u32 = 240;

/// A kind of glass: its name, and how it differs from clear glass.
type Kind = (&'static str, fn(&mut PhysicalMaterial));

/// The kinds of glass, one per row.
const ROWS: [Kind; 5] = [
    ("clear", |_| {}),
    ("tinted", |g| {
        g.attenuation_color = Color::from_hex(0x33aa66);
        g.attenuation_distance = 1.2;
    }),
    ("frosted", |g| g.roughness = 0.35),
    ("dispersive", |g| g.dispersion = 1.0),
    ("thin-walled", |g| g.thickness = 0.0),
];

fn checker() -> Arc<Texture> {
    let n = 256usize;
    let pixels: Vec<u8> = (0..n * n)
        .flat_map(|i| {
            let on = ((i % n / 32) + (i / n / 32)).is_multiple_of(2);
            if on {
                [230, 230, 230, 255]
            } else {
                [40, 40, 40, 255]
            }
        })
        .collect();
    Arc::new(Texture::new(
        n as u32,
        n as u32,
        TextureFormat::Rgba8UnormSrgb,
        pixels,
    ))
}

fn scene(mode: TransparencyMode, kind: fn(&mut PhysicalMaterial)) -> Scene {
    let mut s = Scene::new();
    s.environment = Some(threers::extras::RoomEnvironment::pmrem(128));
    s.background = Color::from_hex(0x404850);
    let tex = checker();
    let mut floor_material = StandardMaterial::new(Color::WHITE);
    floor_material.map = Some(tex.clone());
    let mut floor = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(8.0, 8.0),
        Material::Standard(floor_material),
    ));
    floor.rotate_x(-std::f32::consts::FRAC_PI_2);
    floor.receive_shadow = true;
    s.add(floor);
    let mut wall_material = StandardMaterial::new(Color::from_hex(0xcc3322));
    wall_material.map = Some(tex);
    let mut wall = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(8.0, 4.0),
        Material::Standard(wall_material),
    ));
    wall.position = Vector3::new(0.0, 2.0, -2.5);
    s.add(wall);

    let mut glass = PhysicalMaterial::new(Color::WHITE)
        .with_transmission(1.0, 1.5, 1.6)
        .with_transparency(mode);
    glass.roughness = 0.02;
    kind(&mut glass);
    let mut ball = Object3D::mesh(Mesh::new(
        SphereGeometry::new(0.8, 64, 32),
        Material::Physical(glass),
    ));
    ball.position = Vector3::new(0.0, 0.8, 0.0);
    ball.cast_shadow = true;
    s.add(ball);

    s.add_light(AmbientLight::new(Color::WHITE, 0.2));
    let mut sun = DirectionalLight::new(Color::WHITE, 3.0);
    sun.direction = Vector3::new(-0.6, -1.0, -0.4).normalize();
    sun.cast_shadow = true;
    sun.shadow.map_size = 1024;
    sun.shadow.camera_size = 8.0;
    let mut light = Object3D::light(Light::Directional(sun));
    light.position = Vector3::new(3.0, 5.0, 2.0);
    s.add(light);
    s
}

fn camera() -> PerspectiveCamera {
    let mut c = PerspectiveCamera::new(45.0, W as f32 / H as f32, 0.1, 50.0);
    c.position = Vector3::new(0.0, 1.6, 4.0);
    c.target = Vector3::new(0.0, 0.8, 0.0);
    c
}

fn main() {
    let spp = std::env::var("SPP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let modes = [
        TransparencyMode::Blend,
        TransparencyMode::Refract,
        TransparencyMode::Glass,
    ];
    let cols = modes.len() as u32 + 1;
    let rows = ROWS.len() as u32;
    let mut sheet = vec![0u8; (W * cols * H * rows * 4) as usize];
    let mut place = |row: u32, col: u32, img: &[u8]| {
        for y in 0..H {
            let src = &img[(y * W * 4) as usize..((y + 1) * W * 4) as usize];
            let o = (((row * H + y) * W * cols + col * W) * 4) as usize;
            sheet[o..o + (W * 4) as usize].copy_from_slice(src);
        }
    };
    for (r, (name, kind)) in ROWS.iter().enumerate() {
        for (c, mode) in modes.iter().enumerate() {
            let mut hr = HeadlessRenderer::builder()
                .size(W, H)
                .color_format(threers::wgpu::TextureFormat::Rgba8Unorm)
                .build()
                .expect("headless renderer");
            hr.renderer().set_tone_mapping(ToneMapping::AcesFilmic, 1.0);
            hr.set_msaa(4);
            let img = hr.render_to_rgba_resolved(&mut scene(*mode, *kind), &camera());
            place(r as u32, c as u32, &img);
        }
        let mut pt = RaytraceRenderer::new(W, H);
        pt.set_settings(RaytraceSettings {
            samples_per_pixel: spp,
            max_bounces: 12,
            tone_mapping: ToneMapping::AcesFilmic,
            exposure: 1.0,
            denoise: true,
            ..Default::default()
        });
        let img = pt.render_to_rgba(&mut scene(TransparencyMode::Blend, *kind), &camera());
        place(r as u32, cols - 1, &img);
        println!("{name}");
    }
    std::fs::create_dir_all("out").expect("create out/");
    std::fs::write(
        "out/glass_modes.png",
        encode_png(W * cols, H * rows, &sheet),
    )
    .expect("write png");
    println!("wrote out/glass_modes.png: rows clear, tinted, frosted, dispersive, thin-walled; columns Blend, Refract, Glass, path traced");
}
