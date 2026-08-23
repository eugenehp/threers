//! Captions over a 3D render, both ways: composited on the GPU by the
//! renderer's overlay pass, and painted on the CPU into an RGBA buffer.
//!
//! ```sh
//! cargo run --example captions_render
//! ```
//!
//! Writes `out/captions_render_gpu_*.png` — one per sampled time, so you can
//! watch the cue change as the timeline advances — plus one CPU-composited
//! frame for comparison. The two paths produce the same picture; the GPU one
//! keeps the pixels on the device, the CPU one works on any buffer you already
//! have in memory (which is what video export does).
//!
//! For captions in an exported video, see `examples/captions_video.rs`.

use threers::captions::{
    CaptionAnchor, CaptionOverlay, CaptionPainter, CaptionStyle, CaptionTrack,
};
use threers::{
    AmbientLight, BoxGeometry, Color, DirectionalLight, HeadlessRenderer, Material, Mesh, Object3D,
    PerspectiveCamera, Quaternion, Scene, StandardMaterial, Vector3,
};

const W: u32 = 960;
const H: u32 = 540;

fn build_scene() -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x0d1018);

    for (i, tint) in [0xd94452u32, 0x3fa9d9, 0xe6c04a].into_iter().enumerate() {
        let mut material = StandardMaterial::new(Color::from_hex(tint));
        material.roughness = 0.35;
        material.metalness = 0.1;
        let mut mesh = Object3D::mesh(Mesh::new(
            BoxGeometry::new(1.2, 1.2, 1.2),
            Material::Standard(material),
        ));
        mesh.position = Vector3::new(i as f32 * 2.0 - 2.0, 0.0, 0.0);
        mesh.quaternion = Quaternion::from_euler_xyz(0.3, 0.5 + i as f32 * 0.4, 0.0);
        scene.add(mesh);
    }

    scene.add_light(AmbientLight::new(Color::from_hex(0xffffff), 0.35));
    let mut key = Object3D::light(DirectionalLight::new(Color::from_hex(0xfff4e6), 2.2));
    key.position = Vector3::new(3.0, 4.0, 5.0);
    scene.add(key);
    scene
}

/// A track that exercises wrapping, multi-line cues, and cue placement.
fn track() -> CaptionTrack {
    CaptionTrack::parse_vtt(
        "WEBVTT - Demo [en]\n\
         \n\
         1\n\
         00:00:00.000 --> 00:00:02.000\n\
         Captions render over the 3D scene.\n\
         \n\
         2\n\
         00:00:02.000 --> 00:00:04.000\n\
         Long lines wrap inside the safe area\n\
         and stack upward from the bottom margin.\n\
         \n\
         3\n\
         00:00:04.000 --> 00:00:06.000 align:left line:8%\n\
         Cue settings move this one to the top left.\n",
    )
    .expect("parse demo track")
}

fn main() {
    let _ = std::fs::create_dir_all("out");

    let mut renderer = HeadlessRenderer::builder()
        .size(W, H)
        // Rgba8Unorm, not the sRGB default: the mesh shader already encodes
        // sRGB, and the overlay colors are authored in the same space.
        .color_format(wgpu::TextureFormat::Rgba8Unorm)
        .build()
        .expect("headless renderer (is a GPU adapter available?)");

    let mut scene = build_scene();
    let mut camera = PerspectiveCamera::new(45.0, W as f32 / H as f32, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 0.9, 7.5);
    camera.look_at(Vector3::ZERO);

    // Authored at 1080p; `auto_scale` retargets it to whatever size the
    // overlay is given, so one style covers every output resolution.
    let style = CaptionStyle::default()
        .background([0, 0, 0, 150])
        .outline([0, 0, 0, 255], 2.0)
        .anchor(CaptionAnchor::Bottom);
    let mut overlay =
        CaptionOverlay::with_painter(track(), CaptionPainter::new().style(style.clone()))
            .auto_scale(true);

    // ---- GPU path: the renderer blends the overlay into the target. ----
    for (i, time) in [1.0f64, 3.0, 5.0, 7.0].into_iter().enumerate() {
        let rgba = renderer.render_to_rgba_with_captions(&mut scene, &camera, &mut overlay, time);
        let path = format!("out/captions_render_gpu_{i}.png");
        std::fs::write(&path, threers::encode_png(W, H, &rgba)).expect("write png");
        println!(
            "wrote {path}  (t={time}s, caption showing: {})",
            overlay.is_visible()
        );
    }

    // ---- CPU path: paint straight into an RGBA buffer. ----
    let mut painter = CaptionPainter::new().style(style.for_height(H));
    let mut rgba = renderer.render_to_rgba(&mut scene, &camera);
    painter.burn_in(&mut rgba, W, H, &track(), 3.0);
    std::fs::write(
        "out/captions_render_cpu.png",
        threers::encode_png(W, H, &rgba),
    )
    .expect("write png");
    println!("wrote out/captions_render_cpu.png  (t=3s, CPU-composited)");
}
