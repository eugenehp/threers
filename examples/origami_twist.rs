//! Square twist — Lemma 14 of arXiv:1812.01160.
//!
//! Four flat-foldable degree-4 vertices around a centre square. Only some
//! mode combinations close the speed-coefficient loop; each valid assignment
//! is shown at three fold amounts (almost flat → mid → tight).
//!
//! Red lines are valleys, dashed blue are mountains. Also writes one SVG net
//! per assignment next to the PNG.
//!
//! ```text
//! cargo run --release --example origami_twist [-- out/origami-twist.png]
//! ```

include!("origami_common.inc");

fn main() {
    let out = out_arg("out/origami-twist.png");
    let alpha = (0.75f64).atan();
    let cp = CreasePattern::square_twist(alpha);
    let assigns = cp.find_assignments(1e-8);
    println!(
        "square twist α=arctan(3/4): {} interior vertices, {} rigid assignments (of 16 mode tuples)",
        cp.interior_deg4().len(),
        assigns.len()
    );

    let amounts = [0.18, 0.42, 0.7];
    let mut poses = Vec::new();
    for (ai, assign) in assigns.iter().enumerate() {
        let drive = cp.drive_hinge(assign).expect("drive");
        let mut row = Vec::new();
        for &t in &amounts {
            let tangents = cp.propagate(assign, drive, t).expect("propagate");
            let folded = cp.fold(assign, drive, t).expect("fold");
            row.push((tangents, folded));
        }
        // SVG for every assignment at the mid fold.
        let mid = &row[1].0;
        write_next_to(
            &out,
            &format!("origami-twist-assign{ai}.svg"),
            &cp.to_svg(Some(mid)),
        );
        poses.push(row);
        println!(
            "  assignment {ai}: closure err {:.2e}",
            cp.check_rigid(assign, 1e-9).max_err
        );
    }

    let (w, h) = (1600u32, 900u32);
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

    let tilt = Euler::new(-1.05, 0.4, 0.1);
    let n_rows = poses.len().max(1) as f32;
    let row_span = 3.4;
    let col_span = 3.6;
    for (r, row) in poses.iter().enumerate() {
        let y = (n_rows - 1.0) * 0.5 * row_span - r as f32 * row_span;
        for (c, (tans, folded)) in row.iter().enumerate() {
            let x = (c as f32 - 1.0) * col_span;
            add_folded(
                &mut scene,
                folded,
                &cp,
                tans,
                Vector3::new(x, y, 0.0),
                0.85,
                tilt,
            );
        }
    }

    let mut camera = PerspectiveCamera::new(28.0, w as f32 / h as f32, 0.1, 60.0);
    camera.position = Vector3::new(0.0, 1.2, 14.0 + n_rows * 1.5);
    camera.look_at(Vector3::ZERO);

    let rgba = renderer.render_to_rgba(&mut scene, &camera);
    std::fs::write(&out, encode_png(rw, rh, &rgba)).expect("write png");
    println!("wrote {out} ({rw}x{rh})");
    println!("rows = rigid assignments, columns = fold amount (shallow / mid / tight).");
}
