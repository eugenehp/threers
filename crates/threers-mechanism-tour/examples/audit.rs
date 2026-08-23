//! What the picture does, rather than what the joint readings say.
//!
//! ```text
//! cargo run --release -p threers-mechanism-tour --example audit
//! cargo run --release -p threers-mechanism-tour --example audit -- latch
//! ```
//!
//! A joint reading can be perfectly sensible while the picture is wrong: a part
//! that jumps between frames, a fixed part that drifts, two parts sitting inside
//! each other. None of those show up in a coordinate.
//!
//! Two things are measured, and both are measured on the part rather than on its
//! body. A body's origin is the model's origin, so a part turning about an axis
//! twenty units away swings its origin twenty units per radian while the part
//! itself hardly moves — measure that and every mechanism looks like it is
//! exploding. What you see is the *geometry*, so that is what this follows.

use threers::assembly as geo;
use threers_physics::assembly::{Assembly, PartId};
use threers_physics::math::Isometry;
use threers_mechanism_tour::{model, Tour, MODELS};

fn main() {
    let only = std::env::args().nth(1);
    for m in MODELS {
        if let Some(want) = &only {
            if m.id != *want {
                continue;
            }
        }
        let Some(m) = model(m.id) else { continue };
        match Tour::run(m.source, m.frames, m.fps, m.units_per_metre, m.substeps) {
            Ok(tour) => audit(m, &tour),
            Err(e) => println!("{}: {e}", m.id),
        }
    }
}

fn audit(m: &threers_mechanism_tour::Model, tour: &Tour) {
    let parts = tour.parts.len();
    let size = (tour.bounds[3] - tour.bounds[0])
        .max(tour.bounds[4] - tour.bounds[1])
        .max(tour.bounds[5] - tour.bounds[2])
        .max(1e-6);

    println!(
        "\n===== {}  ({parts} parts, {} frames, model {size:.0} across)",
        m.id, tour.frames
    );

    // ---- how each part actually moves ------------------------------------
    // The centroid of its own vertices, which is the thing on screen.
    let centroids: Vec<Vec<[f32; 3]>> = (0..parts)
        .map(|p| (0..tour.frames).map(|f| centroid(tour, p, f)).collect())
        .collect();

    println!(
        "  {:<12} {:>11} {:>11} {:>10} {:>10}",
        "part", "step/frame", "spin/frame", "travelled", "turned"
    );
    for p in 0..parts {
        let mut step = 0.0f32;
        let mut spin = 0.0f32;
        let mut travel = 0.0f32;
        for f in 1..tour.frames {
            let (a, b) = (centroids[p][f - 1], centroids[p][f]);
            let d = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2) + (b[2] - a[2]).powi(2)).sqrt();
            step = step.max(d);
            travel += d;
            spin = spin.max(between(pose(tour, p, f - 1).1, pose(tour, p, f).1));
        }
        let turned = between(pose(tour, p, 0).1, pose(tour, p, tour.frames - 1).1);
        // A part moving more than a twentieth of the model in one frame, or
        // turning more than 30°, is either very fast or being thrown.
        let flag = if step > size * 0.05 || spin > 30.0 {
            "  <-- fast"
        } else {
            ""
        };
        println!(
            "  {:<12} {step:>11.3} {spin:>11.2} {travel:>10.1} {turned:>10.1}{flag}",
            tour.parts[p]
        );
    }

    // ---- and whether anything ends up inside anything else ----------------
    // Through the assembly's own check, so it is the same narrow phase the
    // simulation used and mated pairs are excluded — two parts a joint holds
    // together overlap at the joint on purpose.
    let Ok(spec) = threers::parse_scad_mechanism(m.source) else {
        return;
    };
    let Ok(mut asm) = Assembly::from_scad(&spec) else {
        return;
    };
    let mut worst: Vec<(String, String, f32, usize)> = Vec::new();
    let step = (tour.frames / 24).max(1);
    let mut sampled = 0;
    for f in (0..tour.frames).step_by(step) {
        sampled += 1;
        for p in 0..parts {
            let (t, q) = pose(tour, p, f);
            asm.place(
                PartId::from(p),
                Isometry::new(
                    threers::math::Vector3::new(t[0], t[1], t[2]),
                    threers::math::Quaternion::new(q[0], q[1], q[2], q[3]),
                ),
            );
        }
        for hit in asm.check().interferences {
            let (a, b) = (name(&asm, hit.a), name(&asm, hit.b));
            match worst.iter_mut().find(|w| w.0 == a && w.1 == b) {
                Some(w) => {
                    w.2 = w.2.max(hit.depth);
                    w.3 += 1;
                }
                None => worst.push((a, b, hit.depth, 1)),
            }
        }
    }
    worst.sort_by(|x, y| y.2.total_cmp(&x.2));
    if worst.is_empty() {
        println!("  nothing is ever inside anything else");
    }
    for (a, b, depth, n) in &worst {
        // The solver leaves a little overlap on purpose; anything near that is
        // a resting contact, not a part passing through another.
        let note = if *depth < size * 1e-3 {
            "resting contact"
        } else {
            "PASSING THROUGH"
        };
        println!("  {a} into {b}: {depth:.3} deep, {n} of {sampled} frames — {note}");
    }
}

fn name(asm: &Assembly, id: PartId) -> String {
    asm.part_of(id)
        .map(|p| p.name.clone())
        .unwrap_or_else(|| "?".into())
}

fn pose(tour: &Tour, part: usize, frame: usize) -> ([f32; 3], [f32; 4]) {
    let o = frame * tour.stride + part * 7;
    let p = &tour.poses[o..o + 7];
    ([p[0], p[1], p[2]], [p[3], p[4], p[5], p[6]])
}

/// Degrees between two orientations, sign-flip safe.
fn between(a: [f32; 4], b: [f32; 4]) -> f32 {
    let dot: f32 = (0..4).map(|k| a[k] * b[k]).sum::<f32>().abs().min(1.0);
    2.0 * dot.acos().to_degrees()
}

/// The mean of the part's vertices, where they are at that frame.
fn centroid(tour: &Tour, part: usize, frame: usize) -> [f32; 3] {
    let (t, q) = pose(tour, part, frame);
    let m = Isometry::new(
        threers::math::Vector3::new(t[0], t[1], t[2]),
        threers::math::Quaternion::new(q[0], q[1], q[2], q[3]),
    )
    .to_matrix4();
    let mut sum = [0.0f32; 3];
    let mut n = 0.0f32;
    for piece in tour.pieces.iter().filter(|p| p.owner == part) {
        for v in piece.positions.chunks_exact(3) {
            let p = threers::math::Vector3::new(v[0], v[1], v[2]).apply_matrix4(&m);
            sum = [sum[0] + p.x, sum[1] + p.y, sum[2] + p.z];
            n += 1.0;
        }
    }
    if n == 0.0 {
        return [0.0; 3];
    }
    [sum[0] / n, sum[1] / n, sum[2] / n]
}

// Keep the generic predicates reachable for anyone extending this.
#[allow(unused)]
fn unused(a: &[geo::Tri], b: &[geo::Tri]) -> bool {
    geo::interfere(a, b)
}
