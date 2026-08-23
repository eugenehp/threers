//! Classic 15-step origami bird — traditional crane / bird base.
//!
//! Traced crease pattern and hand-guided fold stages from the standard
//! instructional diagram.
//!
//! ```text
//! cargo run --release --example origami_classic_bird [-- out/origami-classic-bird.png]
//! ```

include!("origami_common.inc");

use threers::{classic_bird_cp, classic_bird_stages};

fn main() {
    let out = out_arg("out/origami-classic-bird.png");
    let dir = std::path::Path::new(&out)
        .parent()
        .unwrap_or_else(|| std::path::Path::new("out"));
    let _ = std::fs::create_dir_all(dir);

    let cp = classic_bird_cp();
    write_next_to(&out, "origami-classic-bird-creases.svg", &cp.to_svg(None));

    let ref_src = dir.join("classic-bird-ref.jpg");
    if ref_src.exists() {
        let dst = dir.join("origami-classic-bird-ref.jpg");
        let _ = std::fs::copy(&ref_src, &dst);
        println!("reference: {}", dst.display());
    }

    let stages = classic_bird_stages();
    let view = Euler::new(-0.88, 0.48, 0.06);
    let plumage = (
        threers::Color::new(0.38, 0.58, 0.82),
        threers::Color::new(0.92, 0.90, 0.84),
    );

    for (i, stage) in stages.iter().enumerate() {
        let frame = dir.join(format!("origami-classic-bird-{i:02}.png"));
        render_scene(
            frame.to_str().unwrap(),
            640,
            720,
            [0.0, 0.35, 2.8],
            Vector3::new(0.0, 0.1, 0.0),
            |scene| {
                add_folded_shape(scene, &stage.folded, Vector3::ZERO, 1.0, view, plumage);
            },
        );
        println!("  {}", stage.label);
    }

    let n = stages.len();
    render_scene(&out, 4800, 640, [0.0, 0.45, 16.0], Vector3::ZERO, |scene| {
        let span = 2.05;
        let origin = -0.5 * (n as f32 - 1.0) * span;
        for (i, stage) in stages.iter().enumerate() {
            add_folded_shape(
                scene,
                &stage.folded,
                Vector3::new(origin + i as f32 * span, 0.0, 0.0),
                0.82,
                view,
                plumage,
            );
        }
    });

    println!(
        "classic bird: {} creases, {n} fold stages (bird base → crane).",
        cp.hinge_indices().count()
    );
}
