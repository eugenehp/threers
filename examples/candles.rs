//! Candles on a table in a dark room: the simple flame on the left, the
//! realistic (volumetric) one on the right, each lighting the table and the
//! wall with its own flicker. Through ACES and bloom. Writes
//! `out/candles.png`, close-ups of each flame (`out/candle_simple.png`,
//! `out/candle_realistic.png`), and with `CANDLE_FRAMES=n` that many frames
//! of the realistic one flickering (`out/candle_frames/`).
//!
//! ```bash
//! cargo run --release --example candles
//! ```

use threers::extras::candle::{Candle, CandleHandle, FlameStyle};
use threers::postprocessing::{EffectComposer, OutputPass, RenderPass, UnrealBloomPass};
use threers::prelude::*;
use threers::utils::png::encode_png;

const FORMAT: threers::wgpu::TextureFormat = threers::wgpu::TextureFormat::Rgba8Unorm;

fn room() -> (Scene, Vec<CandleHandle>) {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x050403);
    scene.add_light(AmbientLight::new(Color::from_hex(0x2a3550), 0.05));
    let wood = StandardMaterial::new(Color::from_hex(0x5a3a22)).with_roughness(0.55);
    let mut table = Object3D::mesh(Mesh::new(BoxGeometry::new(1.4, 0.04, 0.7), wood.into()));
    table.position = Vector3::new(0.0, 0.73, 0.0);
    table.receive_shadow = true;
    scene.add(table);
    let wall = StandardMaterial::new(Color::from_hex(0x8a8070)).with_roughness(0.9);
    let mut back = Object3D::mesh(Mesh::new(PlaneGeometry::new(4.0, 3.0), wall.into()));
    back.position = Vector3::new(0.0, 1.5, -0.45);
    back.receive_shadow = true;
    scene.add(back);
    // Something to cast a shadow between them.
    let mut cup = Object3D::mesh(Mesh::new(
        CylinderGeometry::new(0.035, 0.03, 0.09, 32, 1, false, 0.0, std::f32::consts::TAU),
        StandardMaterial::new(Color::from_hex(0x2d5d7b)).with_roughness(0.3).into(),
    ));
    cup.position = Vector3::new(0.0, 0.795, -0.12);
    cup.cast_shadow = true;
    cup.receive_shadow = true;
    scene.add(cup);
    let simple = Candle::new(FlameStyle::Simple).with_seed(1.0).add_to(&mut scene, Vector3::new(-0.2, 0.75, 0.0));
    let realistic = Candle::new(FlameStyle::Realistic).with_seed(2.0).add_to(&mut scene, Vector3::new(0.2, 0.75, 0.0));
    (scene, vec![simple, realistic])
}

fn render(scene: &mut Scene, camera: &PerspectiveCamera, w: u32, h: u32) -> Vec<u8> {
    let mut hr = HeadlessRenderer::builder().size(w, h).color_format(FORMAT).build().expect("a GPU adapter");
    hr.set_msaa(4);
    hr.renderer().set_tone_mapping(ToneMapping::AcesFilmic, 1.0);
    let mut composer = EffectComposer::with_size(w, h);
    composer.set_screen_format(FORMAT);
    composer.add_pass(RenderPass::new());
    composer.add_pass(UnrealBloomPass::new(w, h, 0.5, 0.4, 0.8));
    composer.add_pass(OutputPass::new());
    hr.compose(&mut composer, scene, camera)
}

fn main() {
    std::fs::create_dir_all("out").unwrap();
    let (mut scene, candles) = room();
    let time = 2.3;
    for c in &candles {
        c.update(&mut scene, time);
    }
    let (w, h) = (1280, 720);
    let mut camera = PerspectiveCamera::new(35.0, w as f32 / h as f32, 0.02, 20.0);
    camera.position = Vector3::new(0.0, 1.05, 0.75);
    camera.look_at(Vector3::new(0.0, 0.86, 0.0));
    std::fs::write("out/candles.png", encode_png(w, h, &render(&mut scene, &camera, w, h))).unwrap();
    println!("wrote out/candles.png");
    for (name, x) in [("simple", -0.2), ("realistic", 0.2)] {
        let mut close = PerspectiveCamera::new(30.0, 1.0, 0.01, 10.0);
        close.position = Vector3::new(x, 0.9, 0.16);
        close.look_at(Vector3::new(x, 0.89, 0.0));
        let path = format!("out/candle_{name}.png");
        std::fs::write(&path, encode_png(720, 720, &render(&mut scene, &close, 720, 720))).unwrap();
        println!("wrote {path}");
    }
    let frames: usize = std::env::var("CANDLE_FRAMES").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    if frames > 0 {
        std::fs::create_dir_all("out/candle_frames").unwrap();
        let mut close = PerspectiveCamera::new(30.0, 1.0, 0.01, 10.0);
        close.position = Vector3::new(0.2, 0.9, 0.16);
        close.look_at(Vector3::new(0.2, 0.89, 0.0));
        for i in 0..frames {
            for c in &candles {
                c.update(&mut scene, time + i as f32 / 30.0);
            }
            let path = format!("out/candle_frames/{i:03}.png");
            std::fs::write(&path, encode_png(360, 360, &render(&mut scene, &close, 360, 360))).unwrap();
        }
        println!("wrote {frames} frames to out/candle_frames/");
    }
}
