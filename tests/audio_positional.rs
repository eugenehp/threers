//! Sound sources in the scene graph. Expected values are worked by hand from
//! the `PannerNode` formulas in the Web Audio specification.

use std::f32::consts::PI;
use threers::{AudioListener, DistanceModel, Object3D, PositionalAudio, Scene, Vector3};

fn near(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-5
}

#[test]
fn distance_models_follow_the_panner_formulas() {
    let mut a = PositionalAudio {
        ref_distance: 2.0,
        max_distance: 12.0,
        rolloff_factor: 0.5,
        ..Default::default()
    };
    // 2 / (2 + 0.5 * (10 - 2))
    assert!(near(a.distance_gain(10.0), 2.0 / 6.0));
    assert!(
        near(a.distance_gain(0.5), 1.0),
        "inside the reference distance"
    );
    assert!(
        a.distance_gain(100.0) < a.distance_gain(12.0),
        "inverse does not stop at max"
    );

    a.distance_model = DistanceModel::Linear;
    // 1 - 0.5 * (7 - 2) / (12 - 2)
    assert!(near(a.distance_gain(7.0), 0.75));
    assert!(near(a.distance_gain(100.0), 0.5), "linear stops at max");

    a.distance_model = DistanceModel::Exponential;
    // (8 / 2) ^ -0.5
    assert!(near(a.distance_gain(8.0), 0.5));
}

#[test]
fn the_cone_blends_between_its_two_edges() {
    let a = PositionalAudio {
        cone_inner_angle: PI / 2.0,
        cone_outer_angle: PI,
        cone_outer_gain: 0.2,
        ..Default::default()
    };
    assert!(near(a.cone_gain(PI / 8.0), 1.0));
    assert!(near(a.cone_gain(PI), 0.2));
    // Halfway between PI/4 and PI/2.
    assert!(near(a.cone_gain(3.0 * PI / 8.0), 0.6));
    assert!(
        near(PositionalAudio::default().cone_gain(PI), 1.0),
        "omni by default"
    );
}

#[test]
fn the_scene_reports_each_source_from_the_listener() {
    let mut scene = Scene::new();
    let parent = scene.add(Object3D::group());
    scene.arena.get_mut(parent).unwrap().position = Vector3::new(3.0, 0.0, 0.0);
    let mut audio = PositionalAudio::default();
    audio.audio.volume = 0.5;
    let source = scene.add_to(parent, Object3D::audio(audio));
    let hidden = scene.add(Object3D::audio(PositionalAudio::default()));
    scene.arena.get_mut(hidden).unwrap().visible = false;
    scene.update_world();

    // Facing -Z from the origin: +X is to the right, three units away.
    let listener = AudioListener {
        master_volume: 0.5,
        ..Default::default()
    };
    let mix = scene.audio_mix(&listener);
    assert_eq!(mix.len(), 1, "a hidden source is silent");
    assert_eq!(mix[0].0, source);
    assert!(near(mix[0].1.gain, 0.5 * 0.5 / 3.0));
    assert!(near(mix[0].1.pan, 1.0));
}

#[test]
fn the_axis_is_the_nodes_plus_z() {
    let a = PositionalAudio {
        cone_inner_angle: 0.1,
        cone_outer_angle: 0.2,
        ..Default::default()
    };
    let mut scene = Scene::new();
    let id = scene.add(Object3D::audio(a));
    scene.update_world();

    let front = AudioListener {
        position: Vector3::new(0.0, 0.0, 1.0),
        ..Default::default()
    };
    let back = AudioListener {
        position: Vector3::new(0.0, 0.0, -1.0),
        ..Default::default()
    };
    assert_eq!(
        scene.audio_mix(&front),
        vec![(
            id,
            threers::SpatialMix {
                gain: 1.0,
                pan: 0.0
            }
        )]
    );
    assert!(near(scene.audio_mix(&back)[0].1.gain, 0.0));
}
