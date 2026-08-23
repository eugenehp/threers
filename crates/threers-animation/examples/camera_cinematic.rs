//! Cinematic camera moves: fly-to, path, shake, and a two-shot timeline.
//!
//! ```sh
//! cargo run -p threers-animation --example camera_cinematic
//! ```
//!
//! (Library demo — prints pose samples; no window.)

use threers::math::Vector3;
use threers_animation::prelude::*;

fn main() {
    let start = CameraPose::new(Vector3::ZERO, 0.0, 1.2, 6.0);
    let mut anim = CameraAnimator::new(start);

    println!("fly_to …");
    anim.fly_to(
        CameraPose::new(Vector3::new(0.5, 0.0, 0.0), 0.8, 1.0, 3.5),
        1.0,
        Easing::CubicInOut,
    );
    for i in 0..=10 {
        let p = anim.update(0.1);
        println!(
            "  t={:.1}  eye=({:.2},{:.2},{:.2})  fov={:.1}°",
            i as f32 * 0.1,
            p.eye().x,
            p.eye().y,
            p.eye().z,
            p.fov.to_degrees()
        );
    }

    println!("path …");
    let path = CameraPath::catmull_rom(vec![
        Vector3::new(0.0, 2.0, 5.0),
        Vector3::new(3.0, 2.0, 2.0),
        Vector3::new(4.0, 1.0, -1.0),
    ])
    .with_fixed_target(Vector3::ZERO)
    .with_ramp(SpeedRamp::Smoother);
    anim.play_path(path, 1.0);
    for i in 0..=5 {
        let p = anim.update(0.2);
        println!(
            "  u={:.1}  eye=({:.2},{:.2},{:.2})",
            i as f32 * 0.2,
            p.eye().x,
            p.eye().y,
            p.eye().z
        );
    }

    println!("shot timeline (cut + blend) …");
    let a = CameraPose::new(Vector3::ZERO, 0.0, 1.2, 5.0);
    let b = CameraPose::new(Vector3::ZERO, 1.0, 1.0, 2.5);
    let mut tl = ShotTimeline::new();
    tl.push(Shot::hold("wide", a, 0.5));
    tl.push(
        Shot::fly("push", a, b, 1.0, SpeedRamp::Ease(Easing::CubicInOut))
            .with_transition(ShotTransition::ease(0.25, Easing::Linear)),
    );
    for i in 0..=15 {
        tl.seek(i as f32 * 0.1);
        let p = tl.pose();
        println!("  t={:.1}  radius={:.2}", i as f32 * 0.1, p.radius);
    }

    anim.shake = CameraShake::new(0.05, 12.0);
    anim.shake.impulse(Vector3::new(1.0, 0.2, 0.0), 0.3);
    let kicked = anim.update(1.0 / 60.0);
    println!(
        "shake kick eye delta={:.3}",
        (kicked.eye() - anim.pose().eye()).length()
    );
}
