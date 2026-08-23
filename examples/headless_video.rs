//! Headless render → video via `HeadlessRenderer` + `export_video` (H.264 / ffmpeg).
//!
//! For every container format see the `export_*` examples:
//! `export_h264`, `export_hevc`, `export_hevc_vt`, `export_vp9`,
//! `export_vp9_alpha`, `export_gif`, `export_apng`, `export_webm_native`.
//!
//! ```text
//! cargo run --release --example headless_video --features video
//! ```

use std::f32::consts::TAU;

use threers::{
    export_video, AmbientLight, BoxGeometry, Color, DirectionalLight, Euler, HeadlessRenderer,
    Mesh, Object3D, PerspectiveCamera, Scene, StandardMaterial, Vector3, VideoCodec, VideoOptions,
};

fn main() {
    let (w, h) = (640u32, 480u32);
    let mut hr = HeadlessRenderer::builder()
        .size(w, h)
        .supersample(2)
        .build()
        .expect("headless renderer");
    let (rw, rh) = hr.render_size();

    let mut scene = Scene::new();
    scene.background = Color::new(0.05, 0.06, 0.09);
    scene.add_light(AmbientLight::new(Color::WHITE, 0.3));
    scene.add_light(
        DirectionalLight::new(Color::WHITE, 2.2)
            .with_direction(Vector3::new(-0.4, -0.8, -0.5).normalize()),
    );
    let mut mat = StandardMaterial::new(Color::new(0.20, 0.62, 0.95));
    mat.metalness = 0.15;
    mat.roughness = 0.35;
    let cube = scene.add(Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.6, 1.6, 1.6),
        mat.into(),
    )));

    let mut cam = PerspectiveCamera::new(50.0, w as f32 / h as f32, 0.1, 100.0);
    cam.position = Vector3::new(0.0, 1.4, 4.2);
    cam.look_at(Vector3::ZERO);

    let frames = 60;
    let opts = VideoOptions::new("./cube.mp4")
        .fps(30)
        .codec(VideoCodec::H264)
        .crf(20);
    export_video(rw, rh, frames, &opts, |f| {
        let angle = f as f32 / frames as f32 * TAU;
        if let Some(o) = scene.get_mut(cube) {
            o.quaternion = Euler::new(0.5, angle, 0.0).to_quaternion();
        }
        hr.render_to_rgba(&mut scene, &cam)
    })
    .expect("export");

    println!("wrote cube.mp4 ({rw}x{rh}, {frames} frames)");
}
