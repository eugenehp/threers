//! OpenSCAD-style solid modeling — M0 demo (`--features openscad`).
//!
//! Builds a mounting bracket (base plate with two bolt holes and a domed boss)
//! with the `Solid` builder, evaluates it to a `BufferGeometry`, and prints
//! stats. Headless — no window required.
//!
//! Run: `cargo run --example openscad_bracket --features openscad`

use std::f32::consts::FRAC_PI_2;
use threers::{cube, cylinder, geometry_to_stl, linear_extrude, sphere};

fn main() {
    // Base plate: 40 x 24 x 4, centred on the origin (thickness along Z).
    let plate = cube([40.0, 24.0, 4.0]);

    // A drill: cylinder runs along Y, so rotate 90° about X to punch through Z.
    let drill = cylinder(20.0, 2.5).rotate_x(FRAC_PI_2);

    // A domed boss on top.
    let boss = sphere(5.0).translate([0.0, 0.0, 2.0]);

    // An extruded trapezoidal gusset standing up at the back edge (extrudes
    // along Z, so rotate it upright and park it at y = +12).
    let gusset = linear_extrude(3.0, &[[-6.0, 0.0], [6.0, 0.0], [3.0, 8.0], [-3.0, 8.0]])
        .rotate_x(FRAC_PI_2)
        .translate([0.0, 12.0, -2.0]);

    let bracket = plate
        .difference(drill.clone().translate([-12.0, 0.0, 0.0]))
        .difference(drill.translate([12.0, 0.0, 0.0]))
        .union(boss)
        .union(gusset);

    let mut geometry = bracket.to_geometry();

    let verts = geometry
        .attributes
        .get("position")
        .map(|a| a.count())
        .unwrap_or(0);
    let tris = geometry.draw_count() / 3;
    let bb = geometry.compute_bounding_box();

    println!("openscad bracket (M0, float CsgEvaluator):");
    println!("  vertices : {verts}");
    println!("  triangles: {tris}");
    println!(
        "  bounds   : min=({:.2}, {:.2}, {:.2})  max=({:.2}, {:.2}, {:.2})",
        bb.min.x, bb.min.y, bb.min.z, bb.max.x, bb.max.y, bb.max.z
    );

    // Export a printable binary STL.
    let stl = geometry_to_stl(&geometry);
    let path = "out/openscad_bracket.stl";
    if std::fs::write(path, &stl).is_ok() {
        println!("  wrote    : {path} ({} bytes, {tris} facets)", stl.len());
    }
}
