//! A steel L-bracket bolted to a wall and loaded at its tip: mesh it, solve
//! for stress, check it against yield, find how it vibrates, and write the
//! results as coloured PLY files to `out/`.
//!
//! Units are millimetres, newtons, megapascals and tonnes per cubic
//! millimetre, so stresses print in MPa and frequencies in Hz.
//!
//! ```sh
//! cargo run --release -p threers-fem --example fem_bracket
//! ```

use std::sync::Arc;
use threers::exporters::{write_ply, ExportObject, PlyOptions};
use threers::math::ColorRamp;
use threers_fem::{Elastic, Material, TetMesh};

/// Signed distance to an axis-aligned box.
fn cuboid(p: [f64; 3], lo: [f64; 3], hi: [f64; 3]) -> f64 {
    let d = [0, 1, 2].map(|k| (lo[k] - p[k]).max(p[k] - hi[k]));
    let outside = d.iter().map(|v| v.max(0.0).powi(2)).sum::<f64>().sqrt();
    outside + d[0].max(d[1]).max(d[2]).min(0.0)
}

fn main() {
    // A 60 mm arm off a 50 mm upright, both 6 mm thick and 30 mm wide, with
    // a 10 mm hole through the arm near its tip.
    let bracket = |p: [f64; 3]| {
        let upright = cuboid(p, [0.0, 0.0, 0.0], [6.0, 50.0, 30.0]);
        let arm = cuboid(p, [0.0, 0.0, 0.0], [60.0, 6.0, 30.0]);
        let hole = ((p[0] - 48.0).powi(2) + (p[2] - 15.0).powi(2)).sqrt() - 5.0;
        upright.min(arm).max(-hole)
    };
    let mesh = TetMesh::from_sdf([0.0; 3], [60.0, 50.0, 30.0], 3.0, bracket).expect("mesh");
    let q = mesh.quality();
    println!(
        "mesh: {} nodes, {} tetrahedra, volume {:.0} mm³, dihedral {:.1}°–{:.1}°",
        mesh.nodes.len(),
        mesh.tets.len(),
        mesh.volume(),
        q.min_dihedral,
        q.max_dihedral
    );

    // S235 structural steel.
    let steel = Material::new(200e3, 0.3).density(7.85e-9);
    let load = 500.0; // N, downward at the tip
    let problem = Elastic::new(&mesh, steel)
        .fix(|p| p[0] < 1e-6) // the upright's back face is bolted to the wall
        .force(|p| p[0] > 60.0 - 1e-6, [0.0, -load, 0.0]);

    let t = std::time::Instant::now();
    let s = problem.solve().expect("solve");
    println!("static solve: {:.2} s", t.elapsed().as_secs_f64());
    println!("  tip deflection     {:.3} mm", s.max_displacement());
    println!("  peak von Mises     {:.0} MPa", s.max_von_mises());
    println!("  safety factor      {:.2} against 235 MPa", s.safety_factor(235.0));
    println!(
        "  wall reaction      [{:.1}, {:.1}, {:.1}] N",
        s.reaction[0], s.reaction[1], s.reaction[2]
    );

    let t = std::time::Instant::now();
    let modes = problem.modes(3).expect("modes");
    println!("modal solve: {:.2} s", t.elapsed().as_secs_f64());
    for (i, m) in modes.modes.iter().enumerate() {
        println!("  mode {}: {:.0} Hz", i + 1, m.frequency);
    }

    let ramp = ColorRamp::viridis();
    let stress = s.von_mises_geometry(&mesh, 50.0, &ramp);
    let mode1 = modes.modes[0].geometry(&mesh, 8.0, &ramp);
    std::fs::create_dir_all("out").expect("out/");
    for (name, g) in [("fem_bracket_stress", stress), ("fem_bracket_mode1", mode1)] {
        let ply = write_ply(&[ExportObject::mesh(name, Arc::new(g))], &PlyOptions::default()).expect("ply");
        let path = format!("out/{name}.ply");
        std::fs::write(&path, ply).expect("write");
        println!("wrote {path}");
    }
}
