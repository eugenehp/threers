//! **Rust DSL showcase** — build a parametric part three ways and export STL.
//!
//! Demonstrates the `threers` CSG DSL (`--features openscad`):
//!   1. the declarative `scad!` macro (OpenSCAD-shaped: prefix ops + `{ }` blocks),
//!   2. the fluent postfix builder (`.difference(…).union(…)`),
//!   3. native Rust control flow spliced in via the `solid(…)` escape hatch.
//!
//! The model is a parametric bolt flange: a disk with a central bore, a raised
//! hub, and a ring of `BOLTS` holes placed with a normal `for`/`map` loop.
//!
//! Run: `cargo run --example dsl_showcase --features openscad`

use threers::{cylinder, frustum, geometry_to_stl, scad, solid, union, Solid};

// --- parameters -------------------------------------------------------------
const DISK_R: f32 = 30.0;
const DISK_H: f32 = 6.0;
const BORE_R: f32 = 8.0;
const HUB_R: f32 = 14.0;
const HUB_H: f32 = 10.0;
const BOLTS: usize = 6;
const BOLT_R: f32 = 2.5;
const BOLT_CIRCLE: f32 = 22.0;
// The disk's flat faces get one coplanar CDT per boolean; keeping the drilled
// features coarse (fewer facets) keeps those faces small enough to stay on the
// exact (watertight) kernel path rather than the float fallback — see
// `MAX_CDT_POINTS`. The smooth disk rim stays at the default facet count.
const HOLE_FN: usize = 16;

/// A drill: a coarse-faceted cylinder standing along Z, tall enough to punch through.
fn drill(radius: f32, height: f32) -> Solid {
    frustum(height, radius, radius, HOLE_FN).rotate([90.0, 0.0, 0.0])
}

fn main() {
    // (3) A bolt circle, built with a plain Rust loop → a Vec<Solid>.
    let bolt_holes: Vec<Solid> = (0..BOLTS)
        .map(|i| {
            let angle = i as f32 * 360.0 / BOLTS as f32;
            drill(BOLT_R, DISK_H + 4.0).translate([BOLT_CIRCLE, 0.0, 0.0]).rotate([0.0, 0.0, angle])
        })
        .collect();

    // (2) The hub, built fluent-style: a plug minus its own bore, raised on top.
    let hub = frustum(HUB_H, HUB_R, HUB_R, 24)
        .rotate([90.0, 0.0, 0.0])
        .difference(drill(BORE_R, HUB_H + 4.0))
        .translate([0.0, 0.0, DISK_H / 2.0 + HUB_H / 2.0]);

    // (1) The whole flange, declarative. `solid(…)` splices the loop-built ring
    // and the fluent hub straight into the tree.
    let flange = scad! {
        union() {
            difference() {
                cylinder(DISK_H, DISK_R).rotate([90.0, 0.0, 0.0]);   // disk (smooth rim, axis → Z)
                drill(BORE_R, DISK_H + 4.0);                          // central bore
                solid(union!(bolt_holes));                            // ← spliced ring
            }
            solid(hub);                                              // ← spliced hub
        }
    };

    // Evaluate with the robust (exact-where-confident) kernel and export STL.
    let mut geometry = flange.to_geometry_exact();
    let tris = geometry.draw_count() / 3;
    let verts = geometry.attributes.get("position").map(|a| a.count()).unwrap_or(0);
    let bb = geometry.compute_bounding_box();

    println!("DSL flange · {BOLTS} bolt holes:");
    println!("  vertices : {verts}");
    println!("  triangles: {tris}");
    println!(
        "  bounds   : min=({:.1}, {:.1}, {:.1})  max=({:.1}, {:.1}, {:.1})",
        bb.min.x, bb.min.y, bb.min.z, bb.max.x, bb.max.y, bb.max.z
    );

    let stl = geometry_to_stl(&geometry);
    let path = "out/dsl_flange.stl";
    std::fs::create_dir_all("out").ok();
    if std::fs::write(path, &stl).is_ok() {
        println!("  wrote    : {path} ({} bytes, {tris} facets)", stl.len());
    }
}
