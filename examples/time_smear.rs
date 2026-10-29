//! Time smear on a rendered shot: a sphere crossing the frame, a small one
//! orbiting, and a box that is only there for a moment — yet its echo shows
//! up *before* it appears. The backdrop never moves, so it is left exactly
//! alone.
//!
//! Writes `time_smear/sheet.png` — rows are source, faithful, ghost, raw and
//! echo; columns are frames either side of the box's appearance. With the
//! `video` feature it also writes one MP4 per look. `--metal` renders through
//! the Metal backend instead of wgpu (`sheet-metal.png`); the filter itself is
//! plain CPU code and does not care which drew the frames. The browser
//! version is `web/examples/time-smear.html`.
//!
//! ```text
//! cargo run --release --example time_smear
//! cargo run --release --example time_smear --features video
//! cargo run --release --example time_smear --features metal -- --metal
//! ```

use std::f32::consts::TAU;

use threers::time_smear::{SmearLook, TimeSmear};
use threers::{
    encode_png, AmbientLight, BasicMaterial, BoxGeometry, Color, DirectionalLight,
    HeadlessRenderer, Mesh, Object3D, PerspectiveCamera, PlaneGeometry, Scene, SphereGeometry,
    StandardMaterial, Vector3,
};

const FRAMES: usize = 160;
/// The box is visible for these frames only.
const PULSE: std::ops::Range<usize> = 70..86;

/// Whichever backend draws the frames. One is made per run, so the size
/// difference between the variants costs nothing.
#[allow(clippy::large_enum_variant)]
enum Backend {
    Wgpu(HeadlessRenderer),
    #[cfg(feature = "metal")]
    Metal(threers::metal::MetalHeadlessRenderer),
}

impl Backend {
    fn new(metal: bool, w: u32, h: u32) -> Self {
        if metal {
            #[cfg(feature = "metal")]
            return Backend::Metal(
                threers::metal::MetalHeadlessRenderer::builder()
                    .size(w, h)
                    .msaa(4)
                    .build()
                    .expect("metal renderer"),
            );
            #[cfg(not(feature = "metal"))]
            panic!("--metal needs `--features metal`");
        }
        Backend::Wgpu(HeadlessRenderer::builder().size(w, h).supersample(2).build().expect("headless renderer"))
    }

    fn size(&self) -> (u32, u32) {
        match self {
            Backend::Wgpu(r) => r.render_size(),
            #[cfg(feature = "metal")]
            Backend::Metal(r) => r.render_size(),
        }
    }

    fn render(&mut self, scene: &mut Scene, cam: &PerspectiveCamera) -> Vec<u8> {
        match self {
            Backend::Wgpu(r) => r.render_to_rgba(scene, cam),
            #[cfg(feature = "metal")]
            Backend::Metal(r) => r.render_to_rgba(scene, cam).expect("metal frame"),
        }
    }
}

fn main() {
    let metal = std::env::args().any(|a| a == "--metal");
    let (w, h) = (480u32, 270u32);
    let mut hr = Backend::new(metal, w, h);
    let (rw, rh) = hr.size();

    let mut scene = Scene::new();
    scene.background = Color::new(0.10, 0.11, 0.15);
    scene.add_light(AmbientLight::new(Color::WHITE, 0.35));
    scene.add_light(
        DirectionalLight::new(Color::WHITE, 2.0)
            .with_direction(Vector3::new(-0.3, -0.8, -0.5).normalize()),
    );
    let mut wall = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(16.0, 9.0),
        StandardMaterial::new(Color::new(0.28, 0.30, 0.36)).into(),
    ));
    wall.position = Vector3::new(0.0, 0.0, -2.0);
    scene.add(wall);
    let ball = scene.add(Object3D::mesh(Mesh::new(
        SphereGeometry::new(0.6, 48, 24),
        StandardMaterial::new(Color::new(0.95, 0.75, 0.35)).into(),
    )));
    let dot = scene.add(Object3D::mesh(Mesh::new(
        SphereGeometry::new(0.18, 24, 12),
        BasicMaterial::new(Color::new(0.6, 0.95, 1.0)).into(),
    )));
    let block = scene.add(Object3D::mesh(Mesh::new(
        BoxGeometry::new(0.9, 0.9, 0.9),
        BasicMaterial::new(Color::new(1.0, 0.35, 0.45)).into(),
    )));

    let mut cam = PerspectiveCamera::new(45.0, w as f32 / h as f32, 0.1, 100.0);
    cam.position = Vector3::new(0.0, 0.0, 8.0);
    cam.look_at(Vector3::ZERO);

    let frames: Vec<Vec<u8>> = (0..FRAMES)
        .map(|i| {
            let u = i as f32 / FRAMES as f32;
            if let Some(o) = scene.get_mut(ball) {
                o.position = Vector3::new(-4.5 + 9.0 * u, -0.6 + 0.5 * (TAU * u).sin(), 0.0);
            }
            if let Some(o) = scene.get_mut(dot) {
                o.position = Vector3::new(2.6 + 0.8 * (2.0 * TAU * u).cos(), 1.3 + 0.8 * (2.0 * TAU * u).sin(), 0.5);
            }
            if let Some(o) = scene.get_mut(block) {
                o.position = Vector3::new(-2.4, 1.2, 0.0);
                o.visible = PULSE.contains(&i);
            }
            hr.render(&mut scene, &cam)
        })
        .collect();

    std::fs::create_dir_all("time_smear").unwrap();
    let looks = [
        ("faithful", SmearLook::Faithful),
        ("ghost", SmearLook::Ghost),
        ("raw", SmearLook::Raw),
        ("echo", SmearLook::Echo),
    ];
    let smear = TimeSmear::new().strength(0.35).reach(0.55);
    let mut rows = vec![frames.clone()];
    for (_, look) in looks {
        rows.push(smear.clone().look(look).apply(&frames, rw, rh));
    }

    // Contact sheet: frames before, during and after the pulse, at half size.
    let picks = [40usize, 56, 66, 78, 94, 110];
    let (tw, th) = (rw as usize / 2, rh as usize / 2);
    let (sw, sh) = (tw * picks.len(), th * rows.len());
    let mut sheet = vec![0u8; sw * sh * 4];
    for (r, row) in rows.iter().enumerate() {
        for (c, &f) in picks.iter().enumerate() {
            for y in 0..th {
                for x in 0..tw {
                    let s = ((2 * y) * rw as usize + 2 * x) * 4;
                    let d = ((r * th + y) * sw + c * tw + x) * 4;
                    sheet[d..d + 4].copy_from_slice(&row[f][s..s + 4]);
                }
            }
        }
    }
    let name = if metal { "time_smear/sheet-metal.png" } else { "time_smear/sheet.png" };
    std::fs::write(name, encode_png(sw as u32, sh as u32, &sheet)).unwrap();
    println!("wrote {name} (rows: source, faithful, ghost, raw, echo; frames {picks:?})");

    #[cfg(feature = "video")]
    for ((name, _), row) in looks.iter().zip(&rows[1..]) {
        let path = format!("time_smear/{name}.mp4");
        let opts = threers::VideoOptions::new(&path).fps(30).codec(threers::VideoCodec::H264).crf(18);
        threers::export_video(rw, rh, FRAMES, &opts, |i| row[i].clone()).expect("export");
        println!("wrote {path}");
    }
}
