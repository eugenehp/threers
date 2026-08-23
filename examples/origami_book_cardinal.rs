//! Book Cardinal from *Origami Birds* — red/black, crest, black mask, beak.
//!
//! Steps 8–14 match the published page (fold corner → mountain inside → pull
//! crest → head folds → beak → finished). Prelude 1–7 is the bird-base path.
//!
//! ```text
//! cargo run --release --example origami_book_cardinal [-- out/origami-book-cardinal.png]
//! ```

include!("origami_common.inc");

use threers::{book_cardinal_cp, book_cardinal_stages};

fn main() {
    let out = out_arg("out/origami-book-cardinal.png");
    let dir = std::path::Path::new(&out)
        .parent()
        .unwrap_or_else(|| std::path::Path::new("out"));
    let _ = std::fs::create_dir_all(dir);

    let cp = book_cardinal_cp();
    write_next_to(&out, "origami-book-cardinal-creases.svg", &cp.to_svg(None));

    let ref_src = dir.join("amazon-origami-ref.jpg");
    if ref_src.exists() {
        let dst = dir.join("origami-book-cardinal-ref.jpg");
        let _ = std::fs::copy(&ref_src, &dst);
        println!("reference: {}", dst.display());
    }

    let stages = book_cardinal_stages();
    // Red front / dark reverse (black face shows on reverse panels).
    let plumage = (
        threers::Color::new(0.82, 0.14, 0.12),
        threers::Color::new(0.18, 0.16, 0.16),
    );
    let view = Euler::new(-0.85, 0.55, 0.05);

    for (i, stage) in stages.iter().enumerate() {
        let frame = dir.join(format!("origami-book-cardinal-{i:02}.png"));
        let book_step = i >= 7;
        let (cam, look, scale, euler) = if i + 1 == stages.len() {
            (
                [0.15, 0.25, 2.6],
                Vector3::new(0.0, 0.1, 0.0),
                1.15,
                Euler::new(-0.65, 0.7, 0.0),
            )
        } else if book_step {
            (
                [0.1, 0.3, 2.9],
                Vector3::new(0.0, 0.1, 0.0),
                1.05,
                view,
            )
        } else {
            (
                [0.0, 0.35, 2.8],
                Vector3::new(0.0, 0.05, 0.0),
                1.0,
                Euler::new(-0.95, 0.4, 0.0),
            )
        };
        render_scene(frame.to_str().unwrap(), 640, 800, cam, look, |scene| {
            add_folded_shape(scene, &stage.folded, Vector3::ZERO, scale, euler, plumage);
        });
        println!("  {}", stage.label);
    }

    let n = stages.len();
    render_scene(&out, 4200, 720, [0.0, 0.35, 15.5], Vector3::ZERO, |scene| {
        let span = 2.15;
        let origin = -0.5 * (n as f32 - 1.0) * span;
        for (i, stage) in stages.iter().enumerate() {
            let euler = if i + 1 == n {
                Euler::new(-0.65, 0.7, 0.0)
            } else if i >= 7 {
                view
            } else {
                Euler::new(-0.95, 0.4, 0.0)
            };
            add_folded_shape(
                scene,
                &stage.folded,
                Vector3::new(origin + i as f32 * span, 0.0, 0.0),
                0.82,
                euler,
                plumage,
            );
        }
    });

    // Large finished portrait like the book’s step 14 panel.
    let finished = &stages[n - 1].folded;
    render_scene(
        dir.join("origami-book-cardinal-finished.png")
            .to_str()
            .unwrap(),
        960,
        1080,
        [0.2, 0.3, 2.4],
        Vector3::new(0.0, 0.12, 0.0),
        |scene| {
            add_folded_shape(
                scene,
                finished,
                Vector3::ZERO,
                1.35,
                Euler::new(-0.55, 0.75, 0.0),
                plumage,
            );
        },
    );

    println!(
        "book cardinal: {} creases, {n} stages (bird base → crest/mask/beak).",
        cp.hinge_indices().count()
    );
}
