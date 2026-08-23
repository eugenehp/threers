//! Miura-ori tessellation — a rigidly foldable sheet from a seed mode.
//!
//! Interior vertices are all flat-foldable degree-4. A uniform mode assignment
//! (the usual herringbone mountain–valley pattern) closes every face loop, so
//! the whole sheet is one degree of freedom: one number `t` folds it.
//!
//! Checkerboard panels, red valleys / blue mountains, plus `out/origami-miura.svg`.
//!
//! ```text
//! cargo run --release --example origami_miura [-- out/origami-miura.png]
//! ```

include!("origami_common.inc");

fn main() {
    let out = out_arg("out/origami-miura.png");
    let cp = CreasePattern::miura(8, 6);
    println!(
        "Miura {} verts, {} faces, {} interior degree-4 vertices",
        cp.verts.len(),
        cp.faces.len(),
        cp.interior_deg4().len()
    );

    let amounts = [0.12, 0.32, 0.55];
    let mut poses = Vec::new();
    for &t in &amounts {
        let (assign, tangents, folded) = fold_or_die(&cp, t, "Miura");
        if (t - amounts[1]).abs() < 1e-12 {
            write_next_to(&out, "origami-miura.svg", &cp.to_svg(Some(&tangents)));
            println!("closure err {:.2e}", cp.check_rigid(&assign, 1e-9).max_err);
        }
        poses.push((tangents, folded));
    }

    let (w, h) = (1680u32, 720u32);
    let mut renderer = match HeadlessRenderer::builder()
        .size(w, h)
        .supersample(2)
        .build()
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("headless renderer unavailable ({e}) — SVG still written.");
            std::process::exit(2);
        }
    };
    let (rw, rh) = renderer.render_size();
    let mut scene = Scene::new();
    origami_lights(&mut scene);

    let tilt = Euler::new(-1.15, 0.35, 0.05);
    for (i, (tans, folded)) in poses.iter().enumerate() {
        let x = (i as f32 - 1.0) * 6.4;
        add_folded(
            &mut scene,
            folded,
            &cp,
            tans,
            Vector3::new(x, 0.0, 0.0),
            0.55,
            tilt,
        );
    }

    let mut camera = PerspectiveCamera::new(26.0, w as f32 / h as f32, 0.1, 80.0);
    camera.position = Vector3::new(0.0, 2.4, 16.0);
    camera.look_at(Vector3::ZERO);

    let rgba = renderer.render_to_rgba(&mut scene, &camera);
    std::fs::write(&out, encode_png(rw, rh, &rgba)).expect("write png");
    println!("wrote {out} ({rw}x{rh})");
    println!("left → right: almost flat, mid fold, compact pack.");
}
