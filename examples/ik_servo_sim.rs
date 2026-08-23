//! Compare IK trajectories under five joint drives: geared (direct / QDD /
//! 288:1), hydraulic, and tendon.
//!
//! Prints a report showing how friction, backlash, reflected inertia, valve
//! lag, and cable stretch widen the gap between ideal IK and tip tracking.
//!
//! ```text
//! cargo run --release --example ik_servo_sim
//! cargo run --release --example ik_servo_demo   # rendered contact sheet
//! ```

use threers::kinematics::{
    sim::{IkServoSim, Waypoint},
    Goal, Ik, SerialChain,
};

const DOWN: [f64; 3] = [0.0, 0.0, -1.0];

fn main() {
    let chain = SerialChain::planar_3r();
    let seed = [-2.8, 88.9, 93.9];

    let waypoints: Vec<Waypoint> = {
        let mut wps = Vec::new();
        for i in 0..=8 {
            let u = i as f64 / 8.0;
            wps.push(Waypoint {
                t: u * 0.15,
                goal: Goal::PointAlong {
                    at: [280.0 + 140.0 * u, 0.0, 220.0],
                    along: DOWN,
                },
            });
        }
        wps.push(Waypoint {
            t: 0.9,
            goal: Goal::PointAlong {
                at: [420.0, 0.0, 220.0],
                along: DOWN,
            },
        });
        wps
    };

    let ik = Ik {
        limits: (-179.0, 179.0),
        max_iters: 120,
        joint_limits: chain.joints.iter().map(|j| j.limits).collect(),
        ..Default::default()
    };

    println!("IK + coupled plant — 3R arm, square approach, gravity on");
    println!(
        "waypoints: {} over {:.1}s\n",
        waypoints.len(),
        waypoints.last().unwrap().t
    );

    let mut q = seed.to_vec();
    let mut worst_ideal = 0.0f64;
    let mut worst_axis = 0.0f64;
    for wp in &waypoints {
        let s = ik.solve_chain(&q, &wp.goal, &chain);
        worst_ideal = worst_ideal.max(s.position_err);
        worst_axis = worst_axis.max(s.axis_err);
        q = s.joints;
    }
    println!(
        "ideal IK (geometric Jacobian): worst tool = {worst_ideal:.3} mm, worst axis = {worst_axis:.3}°\n"
    );

    let reports = IkServoSim::compare_plant(ik.clone(), &seed, &waypoints, chain.clone());
    for r in &reports {
        r.print_summary();
    }

    println!("\n--- all drives (gears + hydraulic + tendon) ---");
    let all = IkServoSim::compare_drives(ik, &seed, &waypoints, chain);
    for r in &all {
        r.print_summary();
    }

    println!("\n--- force transparency (gentle 10 mN·m contact) ---");
    use threers::kinematics::{servo::ServoJoint, transmission::GearTrain, Drive, ServoMode};
    let dt = 1.0 / 240.0;
    let gentle = 0.01;
    for (label, train) in [
        ("direct", GearTrain::direct_drive()),
        ("qdd-15:1", GearTrain::qdd()),
        ("servo-288:1", GearTrain::high_ratio_servo()),
    ] {
        let mut j = ServoJoint::new(train);
        j.mode = ServoMode::Torque;
        j.tau_motor_cmd = 0.0;
        for _ in 0..240 {
            j.step(0.0, gentle, dt);
        }
        println!(
            "{label}: tau_ext_est={:.4} N·m  SNR_theory={:.1}  (stiction floor={:.3} N·m)",
            j.tau_ext_est,
            train.force_snr(gentle),
            train.friction.tau_s.max(train.friction.tau_c),
        );
    }
    for drive in [Drive::hydraulic(), Drive::tendon()] {
        let mut j = ServoJoint::from_drive(drive.clone());
        j.mode = ServoMode::Torque;
        for _ in 0..240 {
            j.step(0.0, gentle, dt);
        }
        println!(
            "{}: tau_ext_est={:.4} N·m  SNR_theory={:.1}",
            drive.label(),
            j.tau_ext_est,
            drive.force_snr(gentle),
        );
    }

    let direct = &reports[0];
    let high = &reports[2];
    if high.worst_position_err > direct.worst_position_err
        || high.mean_joint_err > direct.mean_joint_err * 1.2
    {
        println!(
            "\nhigh-ratio worst tip {:.1}× direct, mean joint {:.1}× — N² inertia + friction widen the gap",
            high.worst_position_err / direct.worst_position_err.max(1e-6),
            high.mean_joint_err / direct.mean_joint_err.max(1e-6)
        );
    }
    println!("\nFor a rendered comparison: cargo run --release --example ik_servo_demo");
}
