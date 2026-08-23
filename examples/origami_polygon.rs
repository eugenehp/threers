//! Square-twist family — several twist angles, three fold amounts each.
//!
//! `α` is the smallest sector at each of the four vertices. The paper’s
//! `α = arctan(3/4)` sits in the middle row.
//!
//! ```text
//! cargo run --release --example origami_polygon [-- out/origami-polygon.png]
//! ```

include!("origami_common.inc");

fn main() {
    let out = out_arg("out/origami-polygon.png");
    let alphas = [0.32_f64, (0.75_f64).atan(), 0.85];
    let amounts = [0.16, 0.4, 0.65];
    let tilt = Euler::new(-1.05, 0.4, 0.08);

    let mut rows = Vec::new();
    for &alpha in &alphas {
        let cp = CreasePattern::square_twist(alpha);
        let n_asg = cp.find_assignments(1e-8).len();
        println!(
            "α={alpha:.3} rad ({:.1}°): {n_asg} rigid assignments",
            alpha.to_degrees()
        );
        let mut row = Vec::new();
        for &t in &amounts {
            let (_a, tans, folded) = fold_or_die(&cp, t, "square twist");
            row.push((tans, folded));
        }
        write_next_to(
            &out,
            &format!("origami-twist-a{:.0}.svg", alpha.to_degrees()),
            &cp.to_svg(Some(&row[1].0)),
        );
        rows.push((cp, row));
    }

    render_scene(&out, 1680, 1100, [0.0, 1.5, 16.0], Vector3::ZERO, |scene| {
        for (r, (cp, row)) in rows.iter().enumerate() {
            let y = 3.5 - r as f32 * 3.5;
            for (c, (tans, folded)) in row.iter().enumerate() {
                let x = (c as f32 - 1.0) * 4.0;
                add_folded(scene, folded, cp, tans, Vector3::new(x, y, 0.0), 0.88, tilt);
            }
        }
    });
    println!("rows: small / paper (arctan 3/4) / large twist.  columns: fold amount.");
}
