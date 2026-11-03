//! Inverse kinematics: pose a chain so its tip reaches a target.
//!
//! ```text
//! cargo run -p threers-physics --example inverse_kinematics
//! ```

use std::f32::consts::FRAC_PI_2;
use threers_physics::prelude::*;

fn main() {
    reaching();
    constrained_leg();
    fabrik_vs_ccd();
}

/// A four-bone arm reaching for a moving target.
fn reaching() {
    println!("-- a four-bone arm follows a target --");
    let mut arm = IkChain::from_points(&[
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(1.0, 0.0, 0.0),
        Vector3::new(2.0, 0.0, 0.0),
        Vector3::new(3.0, 0.0, 0.0),
        Vector3::new(4.0, 0.0, 0.0),
    ]);
    println!("  reach: {:.1} m", arm.total_length());

    for step in 0..5 {
        // Sweep the target around the arm.
        let angle = step as f32 / 4.0 * std::f32::consts::PI;
        let target = Vector3::new(angle.cos() * 3.0, angle.sin() * 3.0, 0.0);
        let result = arm.solve(target);
        println!(
            "  target {:>6.2},{:>6.2}  ->  {:2} iters, error {:.5}{}",
            target.x,
            target.y,
            result.iterations,
            result.error,
            if result.reached { "" } else { "  (missed)" }
        );
    }

    // Ask for something beyond reach: the chain stretches straight at it.
    let far = Vector3::new(0.0, 50.0, 0.0);
    let result = arm.solve(far);
    println!(
        "  target 50 m up: out_of_reach = {}, tip stops at {:?}",
        result.out_of_reach,
        arm.tip()
    );
}

/// A leg: hip is a cone, knee is a hinge that only bends one way.
fn constrained_leg() {
    println!("\n-- a leg with a hip cone and a knee hinge --");
    let mut leg = IkChain::from_points(&[
        Vector3::new(0.0, 2.0, 0.0), // hip
        Vector3::new(0.0, 1.0, 0.0), // knee
        Vector3::new(0.0, 0.0, 0.0), // ankle
    ]);

    // The root needs a reference direction for its own limit to mean anything.
    leg.base_direction = Some(Vector3::new(0.0, -1.0, 0.0));
    leg.set_constraint(0, IkConstraint::cone(0.9));
    // A knee bends backwards only, up to 90 degrees, about the x axis.
    leg.set_constraint(
        1,
        IkConstraint::hinge_limited(Vector3::new(1.0, 0.0, 0.0), 0.0, FRAC_PI_2),
    );

    for target in [
        Vector3::new(0.0, 0.3, 0.8),
        Vector3::new(0.0, 1.0, 1.2),
        Vector3::new(0.0, 0.0, -1.5),
    ] {
        let result = leg.solve(target);
        let thigh = (leg.joints[1].position - leg.joints[0].position).normalize();
        let shin = (leg.joints[2].position - leg.joints[1].position).normalize();
        let knee_angle = thigh.dot(shin).clamp(-1.0, 1.0).acos();
        println!(
            "  foot target {:>5.1},{:>5.1},{:>5.1}  knee bends {:.2} rad  error {:.3}",
            target.x, target.y, target.z, knee_angle, result.error
        );
    }
    println!("  (the knee never exceeds its {:.2} rad stop)", FRAC_PI_2);
}

/// The two solvers on the same problem.
fn fabrik_vs_ccd() {
    println!("\n-- FABRIK vs CCD --");
    let points: Vec<Vector3> = (0..6).map(|i| Vector3::new(i as f32 * 0.5, 0.0, 0.0)).collect();
    let target = Vector3::new(0.5, 2.0, 0.5);

    let mut fabrik = IkChain::from_points(&points);
    let a = fabrik.solve(target);

    let mut ccd = IkChain::from_points(&points);
    let b = ccd.solve_ccd(target);

    println!("  FABRIK: {:2} iterations, error {:.6}", a.iterations, a.error);
    println!("  CCD:    {:2} iterations, error {:.6}", b.iterations, b.error);

    // Both preserve bone lengths exactly — that is the whole point.
    for (name, chain) in [("FABRIK", &fabrik), ("CCD", &ccd)] {
        let lengths: Vec<f32> = chain
            .joints
            .windows(2)
            .map(|w| (w[1].position - w[0].position).length())
            .collect();
        let drift = lengths.iter().map(|l| (l - 0.5).abs()).fold(0.0, f32::max);
        println!("  {name} bone-length drift: {drift:.2e}");
    }
}
