//! Classic 17-step origami frog — green/white instructional diagram.
//!
//! Preliminary base → squash → bird base → splayed legs → head. Matches the
//! common vector “how to make origami” frog strip.
//!
//! ```text
//! cargo run --release --example origami_frog [-- out/origami-frog.png]
//! ```

include!("origami_common.inc");

use threers::{frog_cp, frog_stages};

fn main() {
    let out = out_arg("out/origami-frog.png");
    let dir = std::path::Path::new(&out)
        .parent()
        .unwrap_or_else(|| std::path::Path::new("out"));
    let _ = std::fs::create_dir_all(dir);

    let cp = frog_cp();
    write_next_to(&out, "origami-frog-creases.svg", &cp.to_svg(None));

    let ref_src = dir.join("simple-bird-ref.jpg");
    if ref_src.exists() {
        let dst = dir.join("origami-frog-ref.jpg");
        let _ = std::fs::copy(&ref_src, &dst);
        println!("reference: {}", dst.display());
    }

    let stages = frog_stages();
    let view = Euler::new(-1.15, 0.35, 0.0);
    // Green / white like the diagram (odd panels = white underside).
    let plumage = (
        threers::Color::new(0.28, 0.62, 0.32),
        threers::Color::new(0.94, 0.94, 0.90),
    );

    for (i, stage) in stages.iter().enumerate() {
        let frame = dir.join(format!("origami-frog-{i:02}.png"));
        let (cam, look, scale) = if i + 1 == stages.len() {
            ([0.0, 2.4, 0.15], Vector3::new(0.0, 0.0, 0.0), 1.15)
        } else {
            ([0.0, 0.35, 2.8], Vector3::new(0.0, 0.05, 0.0), 1.0)
        };
        let euler = if i + 1 == stages.len() {
            Euler::new(-1.45, 0.0, 0.0)
        } else {
            view
        };
        render_scene(
            frame.to_str().unwrap(),
            640,
            720,
            cam,
            look,
            |scene| {
                add_folded_shape(scene, &stage.folded, Vector3::ZERO, scale, euler, plumage);
            },
        );
        println!("  {}", stage.label);
    }

    let n = stages.len();
    render_scene(&out, 5100, 640, [0.0, 0.4, 17.0], Vector3::ZERO, |scene| {
        let span = 1.95;
        let origin = -0.5 * (n as f32 - 1.0) * span;
        for (i, stage) in stages.iter().enumerate() {
            let euler = if i + 1 == n {
                Euler::new(-1.35, 0.25, 0.0)
            } else {
                view
            };
            add_folded_shape(
                scene,
                &stage.folded,
                Vector3::new(origin + i as f32 * span, 0.0, 0.0),
                0.78,
                euler,
                plumage,
            );
        }
    });

    println!(
        "origami frog: {} creases, {n} fold stages (bird base → splayed legs).",
        cp.hinge_indices().count()
    );
}
