//! Post-processing as compiled rlx graphs (`--features rlx`).
//!
//! ```text
//! cargo run --release --features rlx --example rlx_conv_postfx
//! ```
//!
//! One headless render, five 3×3 convolutions, each a graph rlx compiled and
//! ran: `out/rlx_postfx_{source,gaussian,sharpen,sobel,laplacian,emboss}.png`.
//!
//! The renderer already has a post-processing chain, and for effects that
//! belong to the frame it remains the better tool — it never leaves the GPU.
//! What this buys is a kernel that lives in the same graph as whatever else
//! the pipeline does, and one that autodiff can see. The timing table at the
//! end is the honest counterweight: a round trip through host memory is not
//! free, and the numbers say what it costs on each device.

use std::time::Instant;

use threers::rlx::{preferred_device, ConvFilter, Kernel3x3};
use threers::{
    encode_png, AmbientLight, BoxGeometry, Color, DirectionalLight, HeadlessRenderer, Mesh,
    Object3D, PerspectiveCamera, Quaternion, Scene, SphereGeometry, StandardMaterial, Vector3,
};

const SIZE: u32 = 640;

fn main() {
    std::fs::create_dir_all("out").ok();

    // --- One frame to filter -------------------------------------------
    let mut headless = match HeadlessRenderer::builder().size(SIZE, SIZE).build() {
        Ok(h) => h,
        Err(e) => {
            eprintln!("headless renderer unavailable ({e}) — needs a GPU adapter.");
            std::process::exit(2);
        }
    };
    let (width, height) = headless.render_size();
    let frame = render_still_life(&mut headless, width, height);
    write("source", width, height, &frame);

    // --- Five kernels, five graphs --------------------------------------
    let device = preferred_device();
    println!("rlx device: {device:?}");
    for (name, kernel) in [
        ("gaussian", Kernel3x3::GAUSSIAN),
        ("sharpen", Kernel3x3::SHARPEN),
        ("sobel", Kernel3x3::SOBEL_X),
        ("laplacian", Kernel3x3::LAPLACIAN),
        ("emboss", Kernel3x3::EMBOSS),
    ] {
        let mut filter = ConvFilter::new(width, height, kernel, device);
        let filtered = filter.apply(&frame).expect("filter");
        write(name, width, height, &filtered);
    }

    // --- What the round trip costs --------------------------------------
    //
    // Compile is paid once and run is paid per frame, which is the whole
    // reason `ConvFilter` holds a compiled graph instead of building one per
    // call. Reporting them separately is the only way that shows.
    println!("\n{:<8} {:>12} {:>14}", "device", "compile", "per frame");
    for device in [::rlx::Device::Cpu, ::rlx::Device::Gpu] {
        if !::rlx::is_available(device) {
            println!("{:<8} {:>12}", format!("{device:?}"), "unavailable");
            continue;
        }
        let started = Instant::now();
        let mut filter = ConvFilter::new(width, height, Kernel3x3::GAUSSIAN, device);
        let compile = started.elapsed();

        // One warm-up: the first run allocates the arena and, on the GPU,
        // uploads the weights. Timing it as a frame would blame the frame.
        filter.apply(&frame).expect("warm-up");
        let runs = 10;
        let started = Instant::now();
        for _ in 0..runs {
            filter.apply(&frame).expect("filter");
        }
        let per_frame = started.elapsed() / runs;
        println!(
            "{:<8} {:>11.1?} {:>13.1?}",
            format!("{device:?}"),
            compile,
            per_frame
        );
    }
}

/// A sphere, a box and a ground plane — curved shading for the blur to soften
/// and hard silhouettes for the derivative kernels to find.
fn render_still_life(headless: &mut HeadlessRenderer, width: u32, height: u32) -> Vec<u8> {
    let mut scene = Scene::new();
    scene.background = Color::new(0.03, 0.04, 0.06);
    scene.add_light(AmbientLight::new(Color::WHITE, 0.25));
    scene.add_light(
        DirectionalLight::new(Color::WHITE, 3.0)
            .with_direction(Vector3::new(-0.5, -0.8, -0.4).normalize()),
    );

    let mut ball = StandardMaterial::new(Color::new(0.85, 0.45, 0.2));
    ball.roughness = 0.3;
    let mut sphere = Object3D::mesh(Mesh::new(SphereGeometry::new(1.0, 64, 32), ball.into()));
    sphere.position = Vector3::new(-0.9, 0.0, 0.0);
    scene.add(sphere);

    let mut metal = StandardMaterial::new(Color::new(0.55, 0.6, 0.7));
    metal.metalness = 0.8;
    metal.roughness = 0.25;
    let mut cube = Object3D::mesh(Mesh::new(BoxGeometry::new(1.4, 1.4, 1.4), metal.into()));
    cube.position = Vector3::new(1.1, -0.2, -0.3);
    cube.quaternion = Quaternion::from_axis_angle(Vector3::UP, 0.6);
    scene.add(cube);

    let mut camera = PerspectiveCamera::new(45.0, width as f32 / height as f32, 0.1, 100.0);
    camera.position = Vector3::new(0.4, 1.4, 4.4);
    camera.look_at(Vector3::ZERO);
    headless.render_to_rgba(&mut scene, &camera)
}

fn write(name: &str, width: u32, height: u32, rgba: &[u8]) {
    let path = format!("out/rlx_postfx_{name}.png");
    std::fs::write(&path, encode_png(width, height, rgba)).expect("write png");
    println!("wrote {path}");
}
