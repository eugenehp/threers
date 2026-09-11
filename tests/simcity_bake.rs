//! The raycast light bake, which only exists with the crate's raycaster.
#![cfg(feature = "mesh-bvh")]

use threers::{Scene, Vector3};

#[path = "../examples/simcity/mod.rs"]
mod city;
use city::*;

/// Shading has to follow occlusion: a surface under a lid must come out darker
/// than the same surface in the open.
///
/// This is the whole claim the bake makes, and it is easy to satisfy
/// accidentally in the wrong direction — normalising against an up-facing
/// reference darkened every vertical wall in the city by half whether anything
/// was in front of it or not, which looked like ambient occlusion and was not.
#[test]
fn a_covered_floor_bakes_darker_than_an_open_one() {
    let floor = |b: &mut Batches, x: f32| {
        b.pads.add_slab(
            x - 4.0,
            -4.0,
            x + 4.0,
            4.0,
            0.0,
            threers::Color::WHITE,
            Uv::Unit,
        );
    };
    let mut b = Batches::default();
    // Open floor at x = 0; the same floor at x = 40 with a lid two metres up.
    floor(&mut b, 0.0);
    floor(&mut b, 40.0);
    b.trim.add_box(
        Vector3::new(34.0, 2.0, -6.0),
        Vector3::new(46.0, 2.6, 6.0),
        threers::Color::WHITE,
        Uv::Unit,
    );
    let report = bake_light(&mut b, 1.0);
    assert!(report.ran(), "the bake did not run");

    // Mean baked brightness of the two floors, told apart by x.
    let mut open = (0.0f32, 0usize);
    let mut covered = (0.0f32, 0usize);
    for i in 0..b.pads.pos.len() / 3 {
        let x = b.pads.pos[i * 3];
        let v = b.pads.col[i * 3];
        if x.abs() < 12.0 {
            open.0 += v;
            open.1 += 1;
        } else if (x - 40.0).abs() < 12.0 {
            covered.0 += v;
            covered.1 += 1;
        }
    }
    assert!(open.1 > 20 && covered.1 > 20, "not enough samples");
    let (open, covered) = (open.0 / open.1 as f32, covered.0 / covered.1 as f32);
    println!("open floor {open:.3}, covered floor {covered:.3}");
    assert!(
        open > 0.90,
        "an open floor sees the whole sky and should stay near 1.0, got {open:.3}"
    );
    assert!(
        covered < open * 0.72,
        "covered floor at {covered:.3} against open at {open:.3} — the lid did nothing"
    );
}

/// An unoccluded *vertical* wall must bake as bright as an unoccluded floor.
///
/// The two see different parts of the sky and a naive normalisation scores the
/// wall at about 0.55 for no reason but its orientation. That is not occlusion,
/// and folding it into albedo dims every facade in the city.
#[test]
fn orientation_alone_does_not_darken_anything() {
    let mut b = Batches::default();
    b.pads
        .add_slab(-5.0, -5.0, 5.0, 5.0, 0.0, threers::Color::WHITE, Uv::Unit);
    // A free-standing wall well away from it, facing +X.
    b.trim.quad(
        [
            [60.0, 0.0, -5.0],
            [60.0, 6.0, -5.0],
            [60.0, 6.0, 5.0],
            [60.0, 0.0, 5.0],
        ],
        [1.0, 0.0, 0.0],
        Uv::Unit,
        threers::Color::WHITE,
    );
    bake_light(&mut b, 1.0);
    let mean = |v: &[f32]| v.iter().step_by(3).sum::<f32>() / (v.len() / 3) as f32;
    let floor = mean(&b.pads.col);
    let wall = mean(&b.trim.col);
    println!("open floor {floor:.3}, open wall {wall:.3}");
    assert!(
        wall > 0.86,
        "an unoccluded wall baked to {wall:.3} — orientation is being read as occlusion"
    );
    assert!((floor - wall).abs() < 0.16, "floor {floor:.3} vs wall {wall:.3}");
}

/// The same city, baked twice, has to come out identical.
///
/// The gather runs across threads over shared scratch and seeds its rays from
/// vertex position, so a race or a counter-based seed would show up here and
/// nowhere else — the images would differ by a few percent and look fine.
#[test]
fn baking_is_deterministic() {
    let run = || {
        let mut scene = Scene::new();
        let c = generate_city(
            &mut scene,
            &CityParams {
                seed: 4,
                blocks: 4,
                cars: 60,
                layout: Layout::OldTown,
                bake: true,
            },
        );
        (c.stats.triangles, c.stats.bake.rays)
    };
    let a = run();
    let b = run();
    assert_eq!(a, b, "two identical cities baked differently");
    assert!(a.1 > 0, "the bake did not run");
}
