//! What each part's geometry costs to evaluate, before anything is simulated.
//!
//! ```text
//! cargo run --release -p threers-mechanism-tour --example buildtime
//! cargo run --release -p threers-mechanism-tour --example buildtime -- latch
//! ```
//!
//! Two numbers per part: the milliseconds the CSG kernel spent on it and the
//! triangles it produced. They are the same number really — a boolean that takes
//! a long time takes it by producing faces — and both are paid again by
//! everything downstream. Triangles are what the collision narrow phase walks,
//! once per step per pair, and what the browser has to be sent.
//!
//! What this is for is finding *coincident faces*. A union of solids that merely
//! touch, face to face and edge to edge, is the worst case an exact kernel has:
//! each coincident pair splits both faces along the other's edges, and the splits
//! compound across the pile. It is invisible in the model, invisible in the
//! picture, and it does not fail anything — it just quietly costs an order of
//! magnitude. Building the tour's joints put four of them in:
//!
//! ```text
//!                        before          after      by
//!   bench    base    14311 tri 145ms    900   6ms    16x
//!   pendulum arm      1981 tri 149ms    208   2ms    10x
//!   latch    frame   49224 tri 2559ms   942  14ms    52x
//!   gear     output    876 tri 341ms    940  21ms    16x
//! ```
//!
//! The fix in every case was a millimetre: sink a post into the deck it stands
//! on, run a boss into the wheel it carries, make an eye 15 mm across where the
//! shank it is on is 14. Nobody can see it, and the whole tour went from 48,000
//! triangles to 11,000.

use threers_mechanism_tour::MODELS;

fn main() {
    let only = std::env::args().nth(1);
    let (mut total_ms, mut total_tris) = (0.0, 0usize);

    for m in MODELS {
        if only.as_deref().is_some_and(|w| w != m.id) {
            continue;
        }
        let t0 = std::time::Instant::now();
        let spec = match threers::parse_scad_mechanism(m.source) {
            Ok(s) => s,
            Err(e) => {
                println!("{}: {e}", m.id);
                continue;
            }
        };
        println!(
            "{:<12} parsed in {:.0} ms",
            m.id,
            t0.elapsed().as_secs_f64() * 1e3
        );
        for p in &spec.parts {
            let t = std::time::Instant::now();
            let solids = p.solid.clone().parts();
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let tris: usize = solids
                .iter()
                .map(|q| threers::assembly::from_geometry(&q.geometry).len())
                .sum();
            total_ms += ms;
            total_tris += tris;
            // A part costing more than a few milliseconds is almost always two
            // solids sharing a face rather than overlapping.
            let flag = if ms > 40.0 { "   <-- look for a shared face" } else { "" };
            println!("  {:<12} {ms:>8.0} ms  {tris:>6} triangles{flag}", p.name);
        }
    }
    println!("\n{total_tris} triangles in {total_ms:.0} ms across the tour");
}
