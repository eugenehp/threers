//! The same tours the browser runs, on a terminal — so the page and the console
//! can be compared line for line.
//!
//! ```text
//! cargo run -p threers-mechanism-tour --example console            # the bench
//! cargo run -p threers-mechanism-tour --example console -- latch   # one of them
//! cargo run -p threers-mechanism-tour --example console -- --all   # all of them
//! cargo run -p threers-mechanism-tour --example console -- my.scad # your own
//! ```

use threers_mechanism_tour::{model, Model, Tour, MODELS};

fn main() {
    let arg = std::env::args().nth(1);
    match arg.as_deref() {
        Some("--all") => all(),
        // Every joint, every few frames — for working out where a mechanism
        // sticks, which a start and an end value cannot tell you.
        Some("--trace") => {
            let name = std::env::args().nth(2).unwrap_or_else(|| "bench".into());
            let Some(m) = model(&name) else {
                return println!("{name}: not one of {}", ids());
            };
            match Tour::run(m.source, m.frames, m.fps, m.units_per_metre, m.substeps) {
                Ok(t) => {
                    print!("{:>6}", "t");
                    for j in &t.joints {
                        print!("{:>14}", j.name);
                    }
                    println!();
                    let step = (t.frames / 40).max(1);
                    for f in (0..t.frames).step_by(step) {
                        print!("{:>6.2}", f as f32 / t.fps as f32);
                        for j in &t.joints {
                            print!("{:>14.2}", j.values.get(f).copied().unwrap_or(0.0));
                        }
                        println!();
                    }
                }
                Err(e) => println!("{e}"),
            }
        }
        Some("--list") => {
            for m in MODELS {
                println!("  {:<12} {} — {}", m.id, m.title, m.blurb);
            }
        }
        // A bundled model by name, or a file if it is not one.
        Some(name) => match model(name) {
            Some(m) => one(m, true),
            None => match std::fs::read_to_string(name) {
                Ok(source) => show(name, &Tour::run(&source, 240, 60, 1000.0, 8), true),
                Err(e) => println!("{name}: {e} (and not one of {})", ids()),
            },
        },
        None => one(&MODELS[0], true),
    }
}

fn ids() -> String {
    MODELS.iter().map(|m| m.id).collect::<Vec<_>>().join(", ")
}

fn one(m: &Model, verbose: bool) {
    let tour = Tour::run(m.source, m.frames, m.fps, m.units_per_metre, m.substeps);
    show(m.title, &tour, verbose);
}

/// Every bundled mechanism, one line each. The check that they all still run.
fn all() {
    println!(
        "{:<12} {:>5} {:>6} {:>6} {:>7} {:>6} {:>8}  {}",
        "model", "parts", "mates", "tris", "solved", "keys", "verify", "joints"
    );
    let mut failed = 0;
    for m in MODELS {
        match Tour::run(m.source, m.frames, m.fps, m.units_per_metre, m.substeps) {
            Ok(t) => println!(
                "{:<12} {:>5} {:>6} {:>6} {:>6.0}ms {:>6} {:>8}  {}",
                m.id,
                t.facts.parts,
                t.facts.mates,
                t.facts.triangles,
                t.facts.simulated_ms,
                t.facts.reduced_keys,
                if t.facts.agrees { "agrees" } else { "DISAGREES" },
                t.joints
                    .iter()
                    .map(|j| format!(
                        "{} {:.1}{}",
                        j.name,
                        j.values.last().copied().unwrap_or(0.0),
                        j.unit
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Err(e) => {
                failed += 1;
                println!("{:<12} FAILED: {e}", m.id);
            }
        }
    }
    if failed > 0 {
        println!("\n{failed} of {} did not run", MODELS.len());
    }
}

fn show(title: &str, tour: &Result<Tour, String>, verbose: bool) {
    match tour {
        Ok(tour) => {
            println!("######## {title}");
            if verbose {
                print!("{}", tour.transcript());
            }
            println!("== drawn ==");
            println!(
                "  {} pieces, {} triangles, {} frames at {} fps, solved in {:.0} ms",
                tour.pieces.len(),
                tour.facts.triangles,
                tour.frames,
                tour.fps,
                tour.facts.simulated_ms,
            );
            for joint in &tour.joints {
                let v = &joint.values;
                let lo = v.iter().copied().fold(f32::MAX, f32::min);
                let hi = v.iter().copied().fold(f32::MIN, f32::max);
                println!(
                    "  {:<14} {:>9.2} .. {:<9.2} ending at {:>9.2}{}",
                    joint.name,
                    lo,
                    hi,
                    v.last().copied().unwrap_or(0.0),
                    joint.unit
                );
            }
        }
        Err(e) => println!("######## {title}\n  {e}"),
    }
}
