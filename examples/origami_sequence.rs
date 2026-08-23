//! Miura fold sequence — one degree of freedom, many frames.
//!
//! Writes `out/origami-seq-00.png` … and a contact strip `out/origami-sequence.png`.
//!
//! ```text
//! cargo run --release --example origami_sequence [-- out/origami-sequence.png]
//! ```

include!("origami_common.inc");

fn main() {
    let out = out_arg("out/origami-sequence.png");
    let dir = std::path::Path::new(&out)
        .parent()
        .unwrap_or_else(|| std::path::Path::new("out"));
    let _ = std::fs::create_dir_all(dir);

    let cp = CreasePattern::miura(7, 5);
    let n_frames = 8;
    let t0 = 0.06;
    let t1 = 0.62;
    let tilt = Euler::new(-1.12, 0.32, 0.04);

    let mut frames = Vec::new();
    for i in 0..n_frames {
        let u = i as f64 / (n_frames - 1) as f64;
        let t = t0 + (t1 - t0) * u;
        let (_assign, tans, folded) = fold_or_die(&cp, t, "Miura sequence");
        let frame = dir.join(format!("origami-seq-{i:02}.png"));
        render_scene(
            frame.to_str().unwrap(),
            960,
            720,
            [0.0, 1.8, 11.0],
            Vector3::ZERO,
            |scene| {
                add_folded(scene, &folded, &cp, &tans, Vector3::ZERO, 0.7, tilt);
            },
        );
        frames.push((tans, folded));
    }
    write_next_to(
        &out,
        "origami-seq-net.svg",
        &cp.to_svg(Some(&frames[n_frames / 2].0)),
    );

    render_scene(&out, 1920, 520, [0.0, 0.8, 22.0], Vector3::ZERO, |scene| {
        let span = 3.15;
        let origin = -0.5 * (n_frames as f32 - 1.0) * span;
        for (i, (tans, folded)) in frames.iter().enumerate() {
            add_folded(
                scene,
                folded,
                &cp,
                tans,
                Vector3::new(origin + i as f32 * span, 0.0, 0.0),
                0.42,
                tilt,
            );
        }
    });
    println!("{n_frames} frames + contact strip. t = {t0} → {t1}");
}
