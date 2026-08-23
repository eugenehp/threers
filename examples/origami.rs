//! Figure 1 of arXiv:1812.01160 — a single degree-4 vertex, two rigid modes.
//!
//! Four perpendicular creases cannot all fold at once. Mode A folds one
//! opposite pair (a book fold); mode B folds the other pair. The right-hand
//! pair of models uses unequal sector angles, so all four creases move, but
//! still only one mode at a time.
//!
//! Also writes `out/origami-figure1.svg` and `out/origami-birdfoot.svg`.
//!
//! ```text
//! cargo run --release --example origami [-- out/origami.png]
//! ```

include!("origami_common.inc");

use threers::VertexMode;

fn main() {
    let out = out_arg("out/origami.png");

    let cross = CreasePattern::cross(std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);
    let bird = CreasePattern::cross(0.72, 1.15);

    let fold_mode = |cp: &CreasePattern, mode: VertexMode, t: f64| {
        let assign = Assignment::uniform(cp.verts.len(), mode);
        let drive = cp.drive_hinge(&assign).expect("drive");
        let tangents = cp.propagate(&assign, drive, t).expect("propagate");
        let folded = cp.fold(&assign, drive, t).expect("fold");
        (assign, tangents, folded)
    };

    let (a_a, t_a, f_a) = fold_mode(&cross, VertexMode::A, 0.7);
    let (_a_b, t_b, f_b) = fold_mode(&cross, VertexMode::B, 0.7);
    let (_b_a, t_ba, f_ba) = fold_mode(&bird, VertexMode::A, 0.55);
    let (_b_b, t_bb, f_bb) = fold_mode(&bird, VertexMode::B, 0.55);

    println!(
        "Figure 1 (right angles): mode A uses {} hinges, mode B uses {} — never all four.",
        t_a.iter().filter(|t| t.abs() > 1e-8).count(),
        t_b.iter().filter(|t| t.abs() > 1e-8).count()
    );
    println!(
        "Unequal sectors (α=0.72, β=1.15): both modes move all four creases; they still cannot mix."
    );
    let _ = a_a;

    write_next_to(&out, "origami-figure1.svg", &cross.to_svg(Some(&t_a)));
    write_next_to(&out, "origami-birdfoot.svg", &bird.to_svg(Some(&t_ba)));

    let (w, h) = (1600u32, 720u32);
    let mut renderer = match HeadlessRenderer::builder()
        .size(w, h)
        .supersample(2)
        .build()
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("headless renderer unavailable ({e}) — SVGs still written.");
            std::process::exit(2);
        }
    };
    let (rw, rh) = renderer.render_size();

    let mut scene = Scene::new();
    origami_lights(&mut scene);

    let tilt = Euler::new(-0.95, 0.55, 0.15);
    add_folded(
        &mut scene,
        &f_a,
        &cross,
        &t_a,
        Vector3::new(-4.6, 0.0, 0.0),
        1.6,
        tilt,
    );
    add_folded(
        &mut scene,
        &f_b,
        &cross,
        &t_b,
        Vector3::new(-1.5, 0.0, 0.0),
        1.6,
        tilt,
    );
    add_folded(
        &mut scene,
        &f_ba,
        &bird,
        &t_ba,
        Vector3::new(1.6, 0.0, 0.0),
        1.6,
        tilt,
    );
    add_folded(
        &mut scene,
        &f_bb,
        &bird,
        &t_bb,
        Vector3::new(4.7, 0.0, 0.0),
        1.6,
        tilt,
    );

    let mut camera = PerspectiveCamera::new(28.0, w as f32 / h as f32, 0.1, 40.0);
    camera.position = Vector3::new(0.0, 1.4, 11.5);
    camera.look_at(Vector3::new(0.0, 0.0, 0.0));

    let rgba = renderer.render_to_rgba(&mut scene, &camera);
    std::fs::write(&out, encode_png(rw, rh, &rgba)).expect("write png");
    println!("wrote {out} ({rw}x{rh})");
    println!("left pair: Figure 1 modes A / B.  right pair: non-orthogonal vertex, modes A / B.");
}
