//! Progress indicators floating in a scene: glass rings and tubes with
//! glowing liquid in them, at a value and busy, turning in space, through
//! ACES and bloom. Writes `out/progress_in_space.png`, and with
//! `PROGRESS_FRAMES=n` that many frames of them moving
//! (`out/progress_frames/`).
//!
//! ```bash
//! cargo run --release --example progress_in_space
//! ```

use threers::extras::progress::{Indicator3d, IndicatorHandle3d, Palette3d};
use threers::postprocessing::{EffectComposer, OutputPass, RenderPass, UnrealBloomPass};
use threers::prelude::*;
use threers::utils::png::encode_png;

const FORMAT: threers::wgpu::TextureFormat = threers::wgpu::TextureFormat::Rgba8Unorm;

fn space() -> (Scene, Vec<IndicatorHandle3d>) {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x05060a);
    scene.add_light(AmbientLight::new(Color::from_hex(0x334466), 0.3));
    // A dark floor to stand them over.
    let floor = StandardMaterial::new(Color::from_hex(0x0c0e14)).with_roughness(0.35);
    let mut f = Object3D::mesh(Mesh::new(PlaneGeometry::new(8.0, 8.0), floor.into()));
    f.quaternion = Euler::new(-std::f32::consts::FRAC_PI_2, 0.0, 0.0).to_quaternion();
    scene.add(f);
    let mut out = vec![];
    let ring = Indicator3d::ring(0.32, 0.06).palette(Palette3d::EMBER).add_to(&mut scene, Vector3::new(-0.9, 0.75, 0.0));
    ring.set(Some(0.72));
    out.push(ring);
    let spinning = Indicator3d::ring(0.32, 0.06).palette(Palette3d::AURORA).spin(0.12).add_to(&mut scene, Vector3::new(0.0, 0.8, -0.3));
    spinning.set(Some(0.45));
    out.push(spinning);
    let busy = Indicator3d::ring(0.26, 0.05).palette(Palette3d::NEON).bob(0.04).add_to(&mut scene, Vector3::new(0.9, 0.75, 0.0));
    busy.set(None);
    out.push(busy);
    let bar = Indicator3d::bar(1.6, 0.06).palette(Palette3d::OCEAN).shimmer(true).add_to(&mut scene, Vector3::new(0.0, 0.22, 0.35));
    bar.set(Some(0.6));
    out.push(bar);
    (scene, out)
}

fn render(scene: &mut Scene, camera: &PerspectiveCamera, w: u32, h: u32) -> Vec<u8> {
    let mut hr = HeadlessRenderer::builder().size(w, h).color_format(FORMAT).build().expect("a GPU adapter");
    hr.set_msaa(4);
    hr.renderer().set_tone_mapping(ToneMapping::AcesFilmic, 1.0);
    let mut composer = EffectComposer::with_size(w, h);
    composer.set_screen_format(FORMAT);
    composer.add_pass(RenderPass::new());
    composer.add_pass(UnrealBloomPass::new(w, h, 0.6, 0.5, 0.75));
    composer.add_pass(OutputPass::new());
    hr.compose(&mut composer, scene, camera)
}

fn main() {
    std::fs::create_dir_all("out").unwrap();
    let (mut scene, items) = space();
    let (w, h) = (1280, 720);
    let mut camera = PerspectiveCamera::new(40.0, w as f32 / h as f32, 0.05, 50.0);
    camera.position = Vector3::new(0.35, 1.25, 2.6);
    camera.look_at(Vector3::new(0.0, 0.6, 0.0));
    for i in &items {
        i.update(&mut scene, 1.4);
    }
    std::fs::write("out/progress_in_space.png", encode_png(w, h, &render(&mut scene, &camera, w, h))).unwrap();
    println!("wrote out/progress_in_space.png");
    let frames: usize = std::env::var("PROGRESS_FRAMES").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    if frames > 0 {
        std::fs::create_dir_all("out/progress_frames").unwrap();
        for f in 0..frames {
            let t = f as f32 / 30.0;
            for i in &items {
                i.update(&mut scene, t);
            }
            let path = format!("out/progress_frames/{f:03}.png");
            std::fs::write(&path, encode_png(640, 360, &render(&mut scene, &camera, 640, 360))).unwrap();
        }
        println!("wrote {frames} frames to out/progress_frames/");
    }
}
