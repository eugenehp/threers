//! Contact sheet of rigid-origami gadgets.
//!
//! Six models: Figure 1 both modes, a bird’s-foot vertex, a square twist,
//! a hexagonal twist, and a compact Miura pack. Also writes each net as SVG.
//!
//! ```text
//! cargo run --release --example origami_gallery [-- out/origami-gallery.png]
//! ```

include!("origami_common.inc");

use threers::VertexMode;

fn main() {
    let out = out_arg("out/origami-gallery.png");
    let tilt = Euler::new(-1.0, 0.45, 0.1);

    let mut items: Vec<(String, CreasePattern, Vec<f64>, FoldedState)> = Vec::new();

    let push_mode = |items: &mut Vec<_>, name: &str, cp: CreasePattern, mode: VertexMode, t: f64| {
        let assign = Assignment::uniform(cp.verts.len(), mode);
        let drive = cp.drive_hinge(&assign).expect(name);
        let tans = cp.propagate(&assign, drive, t).expect(name);
        let folded = cp.fold(&assign, drive, t).expect(name);
        items.push((name.into(), cp, tans, folded));
    };

    let cross = CreasePattern::cross(std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);
    push_mode(&mut items, "figure1-A", cross.clone(), VertexMode::A, 0.75);
    push_mode(&mut items, "figure1-B", cross, VertexMode::B, 0.75);

    let bird = CreasePattern::cross(0.65, 1.25);
    push_mode(&mut items, "birdfoot", bird, VertexMode::A, 0.5);

    let twist = CreasePattern::square_twist((0.75f64).atan());
    let (_a, tans, folded) = fold_or_die(&twist, 0.45, "square twist");
    items.push(("square-twist".into(), twist, tans, folded));

    let twist2 = CreasePattern::square_twist(0.35);
    let (_a, tans, folded) = fold_or_die(&twist2, 0.42, "acute twist");
    items.push(("acute-twist".into(), twist2, tans, folded));

    let miura = CreasePattern::miura(6, 4);
    let (_a, tans, folded) = fold_or_die(&miura, 0.5, "miura");
    items.push(("miura".into(), miura, tans, folded));

    for (name, cp, tans, _) in &items {
        write_next_to(&out, &format!("origami-gallery-{name}.svg"), &cp.to_svg(Some(tans)));
        println!("{name}: {} faces, {} hinges", cp.faces.len(), cp.hinge_indices().count());
    }

    render_scene(&out, 1920, 820, [0.0, 1.3, 14.5], Vector3::ZERO, |scene| {
        let n = items.len() as f32;
        let span = 3.9;
        let origin = -0.5 * (n - 1.0) * span;
        for (i, (_, cp, tans, folded)) in items.iter().enumerate() {
            add_folded(
                scene,
                folded,
                cp,
                tans,
                Vector3::new(origin + i as f32 * span, 0.0, 0.0),
                0.72,
                tilt,
            );
        }
    });
    println!("gallery: Figure1 A/B, birdfoot, square twist, acute twist, Miura.");
}
