//! Learn a colour grade from a reference, by gradient descent (`--features rlx`).
//!
//! ```text
//! cargo run --release --features rlx --example rlx_fit_grade
//! ```
//!
//! Everywhere else in this bridge, rlx runs a computation somebody wrote. Here
//! it writes one: nobody supplies the grade, the descent finds it.
//!
//! 1. Render a shot, and make a reference by putting a look on it — a cool,
//!    lifted, cross-talked "teal and orange". Pretend this came from a
//!    colourist, or a photograph, or the film you are trying to match.
//! 2. Fit a 3×3 matrix and an offset that carry the render onto the reference.
//!    rlx differentiates the mean squared error with respect to those twelve
//!    numbers; the host walks them downhill.
//! 3. Apply the learned grade to a *different* frame of the same shot. That is
//!    the point of learning it rather than hand-tuning: it transfers.
//!
//! Writes `out/rlx_grade_{source,reference,fitted,transfer_before,transfer_after}.png`.

use threers::rlx::{preferred_device, ColorGrade, FitOptions};
use threers::{
    encode_png, AmbientLight, BoxGeometry, Color, DirectionalLight, HeadlessRenderer, Mesh,
    Object3D, PerspectiveCamera, Quaternion, Scene, SphereGeometry, StandardMaterial, Vector3,
};

const SIZE: u32 = 512;

fn main() {
    std::fs::create_dir_all("out").ok();

    let mut headless = match HeadlessRenderer::builder().size(SIZE, SIZE).build() {
        Ok(h) => h,
        Err(e) => {
            eprintln!("headless renderer unavailable ({e}) — needs a GPU adapter.");
            std::process::exit(2);
        }
    };
    let (width, height) = headless.render_size();

    // --- The shot, from two angles ---------------------------------------
    let source = render(&mut headless, width, height, 0.0);
    let other_angle = render(&mut headless, width, height, 1.1);
    write("source", width, height, &source);
    write("transfer_before", width, height, &other_angle);

    // --- A look to match --------------------------------------------------
    let look = ColorGrade {
        matrix: [[0.86, 0.06, 0.10], [0.04, 0.84, 0.10], [0.02, 0.10, 0.92]],
        bias: [0.02, 0.015, 0.05],
    };
    let reference = look.apply(&source);
    write("reference", width, height, &reference);

    // --- Fit ---------------------------------------------------------------
    let device = preferred_device();
    let options = FitOptions::default();
    println!(
        "fitting on {device:?}: {} iterations, lr {}, momentum {}",
        options.iterations, options.learning_rate, options.momentum
    );

    let report = ColorGrade::fit(&source, &reference, &options, device).expect("fit");
    println!(
        "loss {:.6} → {:.6} over {} samples",
        report.initial_loss, report.final_loss, report.samples
    );
    println!("\nlearned          wanted");
    for o in 0..3 {
        println!(
            "  [{:>6.3} {:>6.3} {:>6.3}]   [{:>6.3} {:>6.3} {:>6.3}]",
            report.grade.matrix[o][0],
            report.grade.matrix[o][1],
            report.grade.matrix[o][2],
            look.matrix[o][0],
            look.matrix[o][1],
            look.matrix[o][2],
        );
    }
    println!(
        "  bias {:>6.3} {:>6.3} {:>6.3}    {:>6.3} {:>6.3} {:>6.3}",
        report.grade.bias[0],
        report.grade.bias[1],
        report.grade.bias[2],
        look.bias[0],
        look.bias[1],
        look.bias[2],
    );

    // --- Apply, and check it against the reference it never saw whole ------
    let fitted = report.grade.apply(&source);
    write("fitted", width, height, &fitted);
    let worst = fitted
        .iter()
        .zip(reference.iter())
        .map(|(a, b)| (*a as i32 - *b as i32).abs())
        .max()
        .unwrap_or(0);
    println!("\nlargest per-channel error vs the reference: {worst}/255");
    println!(
        "\nThe learned coefficients need not match the wanted ones, and above they\n\
         do not — while the image matches to within {worst}/255. This shot's colours\n\
         are correlated (warm ball, cool cube, one light), so they do not span the\n\
         colour cube, and many matrices agree on everything in it. The fit matches\n\
         the *images*; it recovers the matrix only where the footage exercises it."
    );

    // The grade was fitted on one frame; here it is on another.
    write(
        "transfer_after",
        width,
        height,
        &report.grade.apply(&other_angle),
    );
}

/// The same still life, rotated by `turn` radians — one shot, two frames.
fn render(headless: &mut HeadlessRenderer, width: u32, height: u32, turn: f32) -> Vec<u8> {
    let mut scene = Scene::new();
    scene.background = Color::new(0.08, 0.09, 0.12);
    scene.add_light(AmbientLight::new(Color::WHITE, 0.28));
    scene.add_light(
        DirectionalLight::new(Color::WHITE, 2.8)
            .with_direction(Vector3::new(-0.45, -0.75, -0.5).normalize()),
    );

    let mut warm = StandardMaterial::new(Color::new(0.82, 0.42, 0.18));
    warm.roughness = 0.35;
    let mut ball = Object3D::mesh(Mesh::new(SphereGeometry::new(0.9, 64, 32), warm.into()));
    ball.position = Vector3::new(-0.95, 0.0, 0.0);
    scene.add(ball);

    let mut cool = StandardMaterial::new(Color::new(0.25, 0.45, 0.6));
    cool.roughness = 0.5;
    cool.metalness = 0.3;
    let mut cube = Object3D::mesh(Mesh::new(BoxGeometry::new(1.3, 1.3, 1.3), cool.into()));
    cube.position = Vector3::new(1.0, -0.15, -0.2);
    cube.quaternion = Quaternion::from_axis_angle(Vector3::UP, 0.5 + turn);
    scene.add(cube);

    let mut camera = PerspectiveCamera::new(45.0, width as f32 / height as f32, 0.1, 100.0);
    camera.position = Vector3::new(0.6 + turn.sin() * 1.4, 1.2, 4.2);
    camera.look_at(Vector3::ZERO);
    headless.render_to_rgba(&mut scene, &camera)
}

fn write(name: &str, width: u32, height: u32, rgba: &[u8]) {
    let path = format!("out/rlx_grade_{name}.png");
    std::fs::write(&path, encode_png(width, height, rgba)).expect("write png");
    println!("wrote {path}");
}
