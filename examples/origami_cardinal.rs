//! Giang Dinh's Northern Cardinal — rough diagram (2014).
//!
//! Traced crease pattern and eight hand-guided fold stages matching the photo
//! strip at [giangdinh.com/2014/northern-cardinal/cardinal-rough-dia/](https://giangdinh.com/2014/northern-cardinal/cardinal-rough-dia/).
//!
//! ```text
//! cargo run --release --example origami_cardinal [-- out/origami-cardinal.png]
//! ```

include!("origami_common.inc");

use threers::{giang_cardinal_cp, giang_cardinal_stages};

fn main() {
    let out = out_arg("out/origami-cardinal.png");
    let dir = std::path::Path::new(&out)
        .parent()
        .unwrap_or_else(|| std::path::Path::new("out"));
    let _ = std::fs::create_dir_all(dir);

    let cp = giang_cardinal_cp();
    write_next_to(&out, "origami-cardinal-creases.svg", &cp.to_svg(None));

    let ref_src = dir.join("cardinal-ref.jpg");
    if ref_src.exists() {
        let dst = dir.join("origami-cardinal-ref.jpg");
        let _ = std::fs::copy(&ref_src, &dst);
        println!("reference: {}", dst.display());
    }

    let stages = giang_cardinal_stages();
    let view = Euler::new(-0.95, 0.52, 0.08);
    let plumage = (
        threers::Color::new(0.82, 0.14, 0.12),
        threers::Color::new(0.12, 0.1, 0.11),
    );

    for (i, stage) in stages.iter().enumerate() {
        let frame = dir.join(format!("origami-cardinal-{i:02}.png"));
        render_scene(
            frame.to_str().unwrap(),
            720,
            900,
            [0.0, 0.4, 3.2],
            Vector3::new(0.0, 0.15, 0.0),
            |scene| {
                add_folded_shape(
                    scene,
                    &stage.folded,
                    Vector3::ZERO,
                    1.15,
                    view,
                    plumage,
                );
            },
        );
        println!("  {}", stage.label);
    }

    let n = stages.len();
    render_scene(&out, 2560, 640, [0.0, 0.5, 14.0], Vector3::ZERO, |scene| {
        let span = 2.35;
        let origin = -0.5 * (n as f32 - 1.0) * span;
        for (i, stage) in stages.iter().enumerate() {
            add_folded_shape(
                scene,
                &stage.folded,
                Vector3::new(origin + i as f32 * span, 0.0, 0.0),
                0.95,
                view,
                plumage,
            );
        }
    });

    println!(
        "Giang Dinh cardinal: {} creases, {} fold stages (traditional single-sheet)."
        ,
        cp.hinge_indices().count(),
        n
    );
}
