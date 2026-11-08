//! A video playing on meshes: a screen, and a cube with it on every face.
//!
//! With no file it makes its own clip first, moving colour bars encoded with
//! the crate's H.264 encoder, so it runs out of the box. Give it an MP4, MOV,
//! WebM, MKV or IVF to play that instead.
//!
//! ```text
//! cargo run --release --example video_texture --features native-decode [-- clip.mp4]
//! THREERS_BACKEND=metal cargo run --release --example video_texture --features native-decode,metal
//! cargo run --release --example video_texture --features native-decode -- --seconds 5
//! ```
//!
//! 4K and 8K play too: decoding is in hardware where the machine has it
//! (VideoToolbox, NVDEC, VAAPI) and the colour conversion runs on the GPU.
//! A file given on the command line is mapped, not read. The soundtrack is
//! decoded but not played here (threers has no audio output); `threers-ecs`
//! plays it, see `docs/codecs.md`.
//!
//! `--seconds N` closes the window after N seconds and prints the frame rate
//! and how many video frames were shown.

use std::sync::Arc;
use std::time::Instant;

use threers::cameras::Camera;
use threers::codec::h264::{encode_mp4, Yuv420Frame};
use threers::renderer::WindowRenderer;
use threers::video_player::VideoPlayer;
use threers::{
    AmbientLight, BoxGeometry, Color, DirectionalLight, Mesh, Object3D, PerspectiveCamera,
    PlaneGeometry, Quaternion, Scene, StandardMaterial, Vector3,
};
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut file, mut limit) = (None, None);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--seconds" => limit = args.next().and_then(|s| s.parse::<f64>().ok()),
            path => file = Some(path.to_owned()),
        }
    }
    let mut player = match &file {
        Some(path) => VideoPlayer::open_file(path),
        None => VideoPlayer::open(colour_bars()),
    }
    .expect("a video threers can decode");
    player.set_looping(true);
    let (w, h) = player.size();
    println!("{}x{}, {:.1} s", w, h, player.duration());

    let event_loop = EventLoop::new().expect("event loop");
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("threers — video on a mesh")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
            .build(&event_loop)
            .expect("window"),
    );
    let mut view = WindowRenderer::new(window.clone()).expect("a GPU for the window");

    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x15181d);
    scene.add_light(AmbientLight::new(Color::WHITE, 0.4));
    let mut sun = Object3D::light(DirectionalLight::new(Color::WHITE, 1.2));
    sun.position = Vector3::new(3.0, 4.0, 5.0);
    scene.add(sun);

    // The screen glows with the picture, so it reads the same whatever the
    // lighting: the map for the lit part, the same frames as emission.
    let video = player.texture();
    let mut screen_material = StandardMaterial::new(Color::from_hex(0x111111))
        .with_roughness(0.35)
        .with_emissive_map(video.clone());
    screen_material.emissive = Color::WHITE;
    let aspect = w as f32 / h.max(1) as f32;
    let mut screen = Object3D::mesh(Mesh::new(PlaneGeometry::new(2.4 * aspect / 1.5, 1.6), screen_material.into()));
    screen.position = Vector3::new(-1.1, 0.3, 0.0);
    scene.add(screen);
    let mut bezel = Object3D::mesh(Mesh::new(
        BoxGeometry::new(2.4 * aspect / 1.5 + 0.12, 1.72, 0.08),
        StandardMaterial::new(Color::from_hex(0x2a2d33)).with_roughness(0.6).into(),
    ));
    bezel.position = Vector3::new(-1.1, 0.3, -0.05);
    scene.add(bezel);

    let mut cube_material = StandardMaterial::new(Color::WHITE).with_roughness(0.5);
    cube_material.map = Some(video);
    let cube = scene.add(Object3D::mesh(Mesh::new(BoxGeometry::new(1.0, 1.0, 1.0), cube_material.into())));
    if let Some(node) = scene.get_mut(cube) {
        node.position = Vector3::new(1.6, 0.2, 0.3);
    }

    let mut camera = PerspectiveCamera::new(45.0, 1280.0 / 720.0, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 0.6, 5.2);
    camera.look_at(Vector3::new(0.2, 0.2, 0.0));

    let start = Instant::now();
    let mut last = start;
    let mut frames = 0u32;
    let first_video_frame = player.video_texture().stream().frames();
    let window_for_loop = window.clone();
    event_loop
        .run(move |event, target| match event {
            Event::WindowEvent { event, window_id } if window_id == window_for_loop.id() => match event {
                WindowEvent::CloseRequested => target.exit(),
                WindowEvent::Resized(size) => {
                    view.resize(size.width, size.height);
                    camera.set_aspect(view.aspect());
                }
                WindowEvent::RedrawRequested => {
                    let now = Instant::now();
                    player.update(now.duration_since(last).as_secs_f64());
                    last = now;
                    let t = start.elapsed().as_secs_f32();
                    if let Some(node) = scene.get_mut(cube) {
                        node.quaternion = Quaternion::from_axis_angle(Vector3::new(0.3, 1.0, 0.2).normalize(), t * 0.6);
                    }
                    scene.update_world();
                    view.render(&mut scene, &camera);
                    frames += 1;
                    if let Some(limit) = limit {
                        let elapsed = start.elapsed().as_secs_f64();
                        if elapsed >= limit {
                            let shown = player.video_texture().stream().frames() - first_video_frame;
                            println!(
                                "{frames} frames in {elapsed:.1} s ({:.0} fps), {shown} video frames shown{}",
                                frames as f64 / elapsed,
                                player.error().map(|e| format!(", error: {e}")).unwrap_or_default()
                            );
                            target.exit();
                        }
                    }
                }
                _ => {}
            },
            Event::AboutToWait => window_for_loop.request_redraw(),
            _ => {}
        })
        .expect("event loop");
}

/// Four seconds of colour bars sliding sideways, with a white bar sweeping
/// down, at 30 fps: easy to see whether it plays smoothly and which way up.
fn colour_bars() -> Vec<u8> {
    const W: u32 = 480;
    const H: u32 = 270;
    const FPS: u32 = 30;
    const BARS: [[u8; 3]; 7] = [
        [235, 235, 235],
        [235, 235, 16],
        [16, 235, 235],
        [16, 235, 16],
        [235, 16, 235],
        [235, 16, 16],
        [16, 16, 235],
    ];
    let frames: Vec<Yuv420Frame> = (0..FPS * 4)
        .map(|i| {
            let shift = i * W / (FPS * 4);
            let sweep = i * H / (FPS * 4);
            let mut rgba = Vec::with_capacity((W * H * 4) as usize);
            for y in 0..H {
                for x in 0..W {
                    let c = if y.abs_diff(sweep) < 6 {
                        [255, 255, 255]
                    } else if y < 24 {
                        // A dark band at the top, so up is obvious.
                        [20, 20, 20]
                    } else {
                        BARS[(((x + shift) % W) * 7 / W) as usize]
                    };
                    rgba.extend_from_slice(&[c[0], c[1], c[2], 255]);
                }
            }
            Yuv420Frame::from_rgba(W, H, &rgba)
        })
        .collect();
    encode_mp4(W, H, FPS, &frames)
}
