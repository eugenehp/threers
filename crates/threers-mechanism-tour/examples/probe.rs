//! Is every declared joint actually *built*?
//!
//! ```text
//! cargo run --release -p threers-mechanism-tour --example probe
//! ```
//!
//! A mate is a sentence about two parts. It is not a pin, a bore, a rail or a
//! bearing, and it draws none of them — the solver will hold two parts in a
//! perfect hinge relationship whether or not there is anything between them to
//! do the holding. Worse, a mate *suppresses* the two checks that would notice:
//! its pair is excluded from collision and from the interference audit, on
//! purpose, because a real pin does live inside both halves of a real joint.
//!
//! So a mechanism can pass every other test in this crate while its links hang
//! in the air a millimetre apart. This one asks the geometry instead — see
//! `joints_built` for what it measures and why it has to be measured that way.
//!
//! It found nine of the tour's thirty-two joints to be declarations and nothing
//! else, across five of the eight models: a gear train whose output wheel was
//! threaded on nothing, a four-bar whose hinges were pairs of eyes with no pins
//! through them, a ratchet carriage floating 12 mm above its frame, a latch arm
//! hinged to a post 14 mm away from it, and a rack and pinion 50 mm apart with
//! no teeth on either. Every one of them simulated perfectly and read perfectly
//! before it was built, which is the whole reason for measuring it.

use threers_mechanism_tour::{joints_built, MODELS};

fn main() {
    let mut unbuilt = 0;
    for m in MODELS {
        let joints = match joints_built(m.source) {
            Ok(j) => j,
            Err(e) => {
                println!("\n===== {}: {e}", m.id);
                continue;
            }
        };
        let fit = joints.first().map(|j| j.fit).unwrap_or(0.0);
        println!("\n===== {}  (a clearance fit here is {fit:.2})", m.id);
        println!(
            "  {:<15} {:<8} {:>8} {:>10}  parts",
            "joint", "kind", "gap", "off axis"
        );
        for j in &joints {
            unbuilt += !j.built() as usize;
            println!(
                "  {:<15} {:<8} {:>8.2} {:>10.1}  {}{}",
                j.name,
                j.kind,
                j.gap,
                j.off_axis,
                j.parts.join(" + "),
                if j.built() { "" } else { "   <-- NOT BUILT" }
            );
        }
    }
    println!(
        "\n{}",
        match unbuilt {
            0 => "every declared joint is built".to_string(),
            n => format!("{n} joints declared but not built"),
        }
    );
}
