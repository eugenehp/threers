//! The B-rep stack end to end (`--features step`).
//!
//! Models a drilled plate the way a CAD system would — as *surfaces*, not
//! triangles — cuts three bores into it with the B-rep boolean, and writes the
//! result as STEP for another system to open, alongside an STL for a slicer.
//!
//! The point is what survives. A mesh boolean turns a bore into two hundred
//! facets and that is all it can ever be again; here the bore is still a
//! `Surface::Cylinder` with a radius, so the same solid can be re-tessellated at
//! any tolerance afterwards and exported as a cylinder a receiving system can
//! re-dimension. The example checks that by tessellating twice at different
//! tolerances and by reading its own STEP file back.
//!
//! Headless — no window, no GPU.
//!
//! Run: `cargo run --example brep_part --features step`

use threers::brep::{Body, BooleanOp};
use threers::step::{export, import};

const PLATE: [f64; 3] = [40.0, 24.0, 4.0];
const BORE_RADIUS: f64 = 2.5;
const BORE_X: [f64; 3] = [-14.0, 0.0, 14.0];
/// Chord tolerance: the most any facet may deviate from the true surface.
const TOLERANCE: f64 = 1e-3;

fn main() {
    // ---- model ----------------------------------------------------------
    //
    // A `Body` is faces, shared edges and vertices — six planes to start with.
    let mut part = Body::cuboid(PLATE);
    println!(
        "plate {:?}: {} faces, {} edges, volume {:.3}",
        PLATE,
        part.faces().len(),
        part.edges().len(),
        volume(&part)
    );

    // Each bore is cut into the result of the last one. A boolean that returns
    // something not quite closed would compound here rather than cancel.
    for x in BORE_X {
        // The drill runs past both faces so the hole goes right through.
        let drill = Body::cylinder(
            [x, 0.0, -PLATE[2]],
            [0.0, 0.0, 1.0],
            BORE_RADIUS,
            PLATE[2] * 3.0,
        );
        part = match part.boolean(&drill, BooleanOp::Difference, TOLERANCE) {
            Ok(cut) => cut,
            // The layer declines rather than approximating: a pair it cannot
            // resolve exactly says so instead of returning a plausible solid.
            Err(why) => {
                eprintln!("the bore at x={x} could not be cut exactly: {why:?}");
                return;
            }
        };
        println!(
            "  bored at x={x:>6.1}: {} faces, volume {:.3}",
            part.faces().len(),
            volume(&part)
        );
    }

    let expected = PLATE[0] * PLATE[1] * PLATE[2]
        - 3.0 * std::f64::consts::PI * BORE_RADIUS.powi(2) * PLATE[2];
    println!("\nvolume {:.3}, expected {expected:.3}", volume(&part));

    // ---- the bores are still cylinders ----------------------------------
    let mut kinds: Vec<&str> = part.surfaces().iter().map(|s| s.kind()).collect();
    kinds.sort_unstable();
    println!("surfaces: {kinds:?}");

    // ---- so the same solid meshes at any resolution ----------------------
    //
    // Not a decimation of one mesh into another: each is generated from the
    // surfaces, so the fine one is no less exact than the coarse one — and none
    // of them is what the modelling step happened to produce.
    //
    // The floor is the tolerance the part was *modelled* at: the boolean's seam
    // vertices are stored, so asking for coarser than that changes nothing.
    println!("\nre-tessellating the *same* solid (modelled at {TOLERANCE}):");
    for tolerance in [1e-3, 1e-4, 1e-5] {
        let mut b = part.clone();
        b.refine_edges(tolerance);
        let (mesh, report) = b.tessellate(tolerance);
        println!(
            "  tolerance {tolerance:<7} {:>6} triangles, {:>5} vertices, {}",
            report.triangles,
            mesh.get_attribute("position")
                .map_or(0, |p| p.array.len() / 3),
            if report.is_closed() {
                "watertight"
            } else {
                "OPEN"
            }
        );
    }

    // ---- write it out ----------------------------------------------------
    let _ = std::fs::create_dir_all("out");

    let (step_text, report) = export(&part, "drilled-plate", TOLERANCE);
    if report.skipped.is_empty() {
        println!(
            "\nSTEP: {} faces, {} edges, all analytic",
            report.faces, report.edges
        );
    } else {
        // Nothing is silently approximated; anything unmappable is named.
        println!(
            "\nSTEP: {} faces, dropped {:?}",
            report.faces, report.skipped
        );
    }
    std::fs::write("out/drilled-plate.step", &step_text).expect("write step");

    let mut b = part.clone();
    b.refine_edges(TOLERANCE);
    let (mesh, _) = b.tessellate(TOLERANCE);
    std::fs::write("out/drilled-plate.stl", threers::geometry_to_stl(&mesh)).expect("write stl");

    println!(
        "wrote out/drilled-plate.step ({} bytes) and out/drilled-plate.stl",
        step_text.len()
    );

    // ---- read our own file back ------------------------------------------
    //
    // The round trip is the real claim: what comes back is a *solid*, with its
    // surfaces, not a mesh of one.
    let (reloaded, report) = import(&step_text, TOLERANCE).expect("read back what we wrote");
    let bores = reloaded
        .surfaces()
        .iter()
        .filter(|s| s.kind() == "cylinder")
        .count();
    println!(
        "\nread back: {} faces, {bores} cylindrical surfaces, volume {:.3}",
        report.faces,
        volume(&reloaded)
    );

    // And being a solid, it can be cut again — which is the difference between
    // reading a file and reading a *model*.
    let extra = Body::cylinder([0.0, 8.0, -PLATE[2]], [0.0, 0.0, 1.0], 1.5, PLATE[2] * 3.0);
    match reloaded.boolean(&extra, BooleanOp::Difference, TOLERANCE) {
        Ok(again) => println!(
            "drilled one more bore into the imported solid: {} faces, volume {:.3}",
            again.faces().len(),
            volume(&again)
        ),
        Err(why) => println!("the extra bore declined: {why:?}"),
    }

    // Not everything resolves, and the layer says so rather than guessing.
    //
    // A cross-hole runs the length of the plate and meets all three bores
    // cylinder-to-cylinder. That pair has no closed form, and the curve is
    // *traced* rather than refused — so what declines is not the intersection
    // but what the pieces assemble into, and the reason names which.
    let cross = Body::cylinder([-30.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.5, 60.0);
    match reloaded.boolean(&cross, BooleanOp::Difference, TOLERANCE) {
        Ok(again) => println!("cross-hole: {} faces", again.faces().len()),
        Err(why) => println!("cross-hole declined: {why:?}"),
    }
}

/// Volume of a solid, by the divergence theorem over its tessellation.
///
/// A closed mesh's signed volume is `⅙ ∑ a · (b × c)` over its triangles, which
/// is only meaningful if it really is closed — so this asserts that first.
fn volume(body: &Body) -> f64 {
    let mut b = body.clone();
    b.refine_edges(TOLERANCE);
    let (mesh, report) = b.tessellate(TOLERANCE);
    assert!(
        report.is_closed(),
        "not watertight: {} open edges",
        report.boundary_edges
    );
    let pos = &mesh
        .get_attribute("position")
        .expect("tessellation always has positions")
        .array;
    let idx = mesh.index.as_ref().expect("tessellation is indexed");
    let at = |i: u32| -> [f64; 3] {
        let o = i as usize * 3;
        [pos[o] as f64, pos[o + 1] as f64, pos[o + 2] as f64]
    };
    idx.chunks_exact(3)
        .map(|t| {
            let (a, b, c) = (at(t[0]), at(t[1]), at(t[2]));
            (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]))
                / 6.0
        })
        .sum::<f64>()
        .abs()
}
