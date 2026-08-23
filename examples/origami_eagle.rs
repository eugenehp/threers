//! Origami Eagle — accordion wings, hooked beak, fan tail (origamiok.com).
//!
//! Traced crease pattern and 14 hand-guided fold stages from the photographic
//! tutorial. Finished model stands ~13 cm from a 20 cm square.
//!
//! ```text
//! cargo run --release --example origami_eagle [-- out/origami-eagle.png]
//! ```

include!("origami_common.inc");

use threers::{eagle_cp, eagle_stages};

fn main() {
    let out = out_arg("out/origami-eagle.png");
    let dir = std::path::Path::new(&out)
        .parent()
        .unwrap_or_else(|| std::path::Path::new("out"));
    let _ = std::fs::create_dir_all(dir);

    let cp = eagle_cp();
    write_next_to(&out, "origami-eagle-creases.svg", &cp.to_svg(None));

    let ref_src = dir.join("bird-base-ref.jpg");
    if ref_src.exists() {
        let dst = dir.join("origami-eagle-ref.jpg");
        let _ = std::fs::copy(&ref_src, &dst);
        println!("reference: {}", dst.display());
    }

    let stages = eagle_stages();
    let view = Euler::new(-0.55, 0.45, 0.0);
    // Yellow / cream like the tutorial's yellow eagle; odd panels darker ochre.
    let plumage = (
        threers::Color::new(0.92, 0.78, 0.22),
        threers::Color::new(0.78, 0.55, 0.14),
    );

    for (i, stage) in stages.iter().enumerate() {
        let frame = dir.join(format!("origami-eagle-{i:02}.png"));
        let cam_z = if i + 1 == stages.len() { 3.4 } else { 3.0 };
        render_scene(
            frame.to_str().unwrap(),
            640,
            800,
            [0.0, 0.15, cam_z],
            Vector3::new(0.0, 0.0, 0.0),
            |scene| {
                add_folded_shape(scene, &stage.folded, Vector3::ZERO, 1.05, view, plumage);
            },
        );
        println!("  {}", stage.label);
    }

    let n = stages.len();
    render_scene(&out, 4200, 720, [0.0, 0.2, 16.0], Vector3::ZERO, |scene| {
        let span = 2.15;
        let origin = -0.5 * (n as f32 - 1.0) * span;
        for (i, stage) in stages.iter().enumerate() {
            add_folded_shape(
                scene,
                &stage.folded,
                Vector3::new(origin + i as f32 * span, 0.0, 0.0),
                0.85,
                view,
                plumage,
            );
        }
    });

    // Yellow + red finished pair (as in the tutorial collage).
    let red = (
        threers::Color::new(0.82, 0.16, 0.12),
        threers::Color::new(0.55, 0.08, 0.08),
    );
    let finished = &stages[n - 1].folded;
    render_scene(
        dir.join("origami-eagle-finished.png").to_str().unwrap(),
        1280,
        900,
        [0.0, 0.25, 4.2],
        Vector3::ZERO,
        |scene| {
            add_folded_shape(
                scene,
                finished,
                Vector3::new(-1.15, 0.0, 0.0),
                1.2,
                Euler::new(-0.5, 0.55, 0.0),
                plumage,
            );
            add_folded_shape(
                scene,
                finished,
                Vector3::new(1.15, 0.0, 0.0),
                1.2,
                Euler::new(-0.5, -0.55, 0.0),
                red,
            );
        },
    );

    println!(
        "origami eagle: {} creases, {n} fold stages (accordion wings + hooked beak).",
        cp.hinge_indices().count()
    );
}
