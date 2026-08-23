//! Headless PNG render of the same spacecraft as `spacecraft_materials`, for
//! machines with no display (CI, servers, remote boxes).
//!
//! ```text
//! cargo run --example spacecraft_render                  # → spacecraft.png
//! cargo run --example spacecraft_render -- --out ship.png
//! ```
//!
//! The scene itself lives in `spacecraft_build.inc`, shared with the interactive
//! example, so the two cannot drift apart.

use std::f32::consts::{PI, TAU};
use std::path::PathBuf;

use threers::materials::presets;
use threers::{
    AmbientLight, BoxGeometry, CylinderGeometry, DirectionalLight, HeadlessRenderer, Material,
    Mesh, Object3D, PerspectiveCamera, PhysicalMaterial, Scene, SphereGeometry, TorusGeometry,
    Vector3,
};

include!("spacecraft_common.inc");
include!("spacecraft_build.inc");

const W: u32 = 1100;
const H: u32 = 750;
const SS: u32 = 2;

fn out_path(default: &str) -> PathBuf {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--out" {
            return PathBuf::from(args.next().expect("--out needs a path"));
        }
    }
    PathBuf::from(default)
}

fn main() {
    let out = out_path("spacecraft.png");

    let mut renderer = HeadlessRenderer::builder()
        .size(W, H)
        .supersample(SS)
        // Rgba8Unorm, NOT the Rgba8UnormSrgb default: the mesh shader already
        // does its own linear→sRGB encode, and an sRGB target would encode a
        // second time, washing the whole image out.
        .color_format(wgpu::TextureFormat::Rgba8Unorm)
        .build()
        .expect("headless renderer (is a GPU adapter available?)");

    // The environment now carries a sun ~600x brighter than white, so linear
    // output would clip it (and everything near it) to flat white. ACES rolls
    // that off and — crucially — preserves hue while doing so, which is what
    // keeps a blown-out gold highlight looking like gold.
    renderer
        .renderer()
        .set_tone_mapping(threers::ToneMapping::AcesFilmic, 1.0);

    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x03040a);
    scene.environment = Some(orbital_environment(SUN_DIR));
    build_spacecraft(&mut scene);

    scene.add_light(AmbientLight::new(Color::from_hex(0x223044), 0.06));
    let mut sun = Object3D::light(DirectionalLight::new(Color::from_hex(0xfff6e8), 2.4));
    sun.position = Vector3::new(SUN_DIR[0], SUN_DIR[1], SUN_DIR[2]) * 12.0;
    sun.cast_shadow = true;
    scene.add(sun);
    let mut bounce = Object3D::light(DirectionalLight::new(earth_bounce(), 0.5));
    bounce.position = Vector3::new(-0.3, -1.0, 0.2) * 12.0;
    scene.add(bounce);

    let mut camera = PerspectiveCamera::new(42.0, W as f32 / H as f32, 0.1, 200.0);
    camera.position = Vector3::new(4.2, 2.4, 6.0);
    camera.look_at(Vector3::ZERO);
    scene.update_world();

    println!("rendering {W}×{H} at {SS}×…");
    let rgba = renderer.render_to_rgba_resolved(&mut scene, &camera);

    std::fs::write(&out, threers::encode_png(W, H, &rgba)).expect("write png");
    println!("wrote {}", out.display());
}
