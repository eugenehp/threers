//! Run an RLX graph over a rendered frame (`--features rlx`).
//!
//! ```text
//! cargo run --release --features rlx --example rlx_frame_filter [-- out/rlx_filter.png]
//! ```
//!
//! The graph is a Reinhard tone map — `y = gx / (1 + gx)` — built by hand from
//! three IR nodes, which is enough to show the whole path: render headless,
//! hand the frame to rlx as a tensor, compile once, run, and take the result
//! back as pixels. Swap the three nodes for a loaded network and nothing else
//! in this file changes.
//!
//! Tone mapping is the honest example because it *must* happen in linear
//! light: applying it to sRGB-encoded values would compress the curve twice.
//! `ColorSpace::Linear` is what makes the bytes the renderer produced into the
//! numbers the graph expects.

use threers::rlx::prelude::*;
use threers::rlx::{preferred_device, ColorSpace, FrameFilter, Layout};
use threers::{
    encode_png, AmbientLight, Color, DirectionalLight, HeadlessRenderer, Mesh, Object3D,
    PerspectiveCamera, Scene, SphereGeometry, StandardMaterial, Vector3,
};

const SIZE: u32 = 512;
/// Exposure applied before the curve. Above 1 so there is something for the
/// tone map to roll off.
const GAIN: f64 = 2.5;

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "out/rlx_filter.png".into());

    // --- Render a frame -------------------------------------------------
    let mut headless = match HeadlessRenderer::builder().size(SIZE, SIZE).build() {
        Ok(h) => h,
        Err(e) => {
            eprintln!("headless renderer unavailable ({e}) — needs a GPU adapter.");
            std::process::exit(2);
        }
    };
    let (width, height) = headless.render_size();

    let mut scene = Scene::new();
    scene.background = Color::new(0.02, 0.03, 0.05);
    scene.add_light(AmbientLight::new(Color::WHITE, 0.2));
    scene.add_light(
        DirectionalLight::new(Color::WHITE, 3.0)
            .with_direction(Vector3::new(-0.5, -0.7, -0.5).normalize()),
    );
    let mut material = StandardMaterial::new(Color::new(0.85, 0.55, 0.25));
    material.metalness = 0.1;
    material.roughness = 0.35;
    scene.add(Object3D::mesh(Mesh::new(
        SphereGeometry::new(1.0, 64, 32),
        material.into(),
    )));
    let mut camera = PerspectiveCamera::new(45.0, 1.0, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 0.8, 3.4);
    camera.look_at(Vector3::ZERO);

    let frame = headless.render_to_rgba(&mut scene, &camera);
    std::fs::create_dir_all("out").ok();
    std::fs::write(
        "out/rlx_filter_input.png",
        encode_png(width, height, &frame),
    )
    .expect("write");

    // --- The graph ------------------------------------------------------
    //
    // Shapes are declared, not inferred from the buffer: `[1, H, W, 4]` is the
    // NHWC layout `FrameFilter` will hand it, and a mismatch is a compile-time
    // complaint from rlx rather than a wrong picture.
    let mut g = Graph::new("reinhard");
    let x = g.input(
        "frame",
        Shape::new(&[1, height as usize, width as usize, 4], DType::F32),
    );
    let gain = g.constant(GAIN, DType::F32);
    let one = g.constant(1.0, DType::F32);
    let exposed = g.mul(x, gain);
    let denominator = g.add(exposed, one);
    let mapped = g.div(exposed, denominator);
    g.set_outputs(vec![mapped]);

    let device = preferred_device();
    println!("rlx devices: {:?}, using {device:?}", available_devices());

    let mut filter = FrameFilter::new(g, device, "frame")
        .layout(Layout::Nhwc)
        .color_space(ColorSpace::Linear);

    let (mut filtered, w, h) = filter.apply(&frame, width, height).expect("filter");

    // The graph is elementwise and does not know which channel is coverage, so
    // it tone-mapped alpha along with the colour. Put it back.
    for px in filtered.chunks_exact_mut(4) {
        px[3] = 255;
    }

    std::fs::write(&out, encode_png(w, h, &filtered)).expect("write png");
    println!("wrote out/rlx_filter_input.png and {out} ({w}x{h})");
}
