//! Scratch: a horizontal cantilever under its own weight, against beam theory.

use threers_continuum::prelude::*;
use threers_physics::prelude::*;

fn main() {
    for links in [5usize, 10, 20, 40] {
        let rod = Rod::new(0.3, links)
            .radius(0.002)
            .material(2.0e9, 0.35, 1200.0)
            .damped(0.3);
        let props = rod.link_properties();
        let ei = rod.youngs * rod.second_moment();
        let w = rod.mass() * 9.81 / rod.length;
        let predicted = w * rod.length.powi(4) / (8.0 * ei);

        let mut world = World::new();
        let arm = Continuum::build(
            &mut world,
            rod.clone(),
            None,
            Vector3::ZERO,
            Vector3::new(1.0, 0.0, 0.0),
        );
        for _ in 0..4000 {
            world.step_fixed();
        }
        let tip = arm.tip(&world);
        println!(
            "links {links:3}: K {:8.5}  droop {:.5}  beam {:.5}  ratio {:.3}  arc {:.5}",
            props.bend_stiffness,
            -tip.y,
            predicted,
            -tip.y / predicted,
            arm.arc_length(&world)
        );
    }
}
