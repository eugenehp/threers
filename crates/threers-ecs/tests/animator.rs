//! The animation state machine, driving real scene nodes.

use threers::animation::{AnimationClip, KeyframeTrack, TrackTarget};
use threers::core::Object3D;
use threers::math::Vector3;
use threers_ecs::animator::*;
use threers_ecs::prelude::*;
use threers_ecs::testing::TestGame;

/// Holds the node at `y` for `seconds`.
fn hold(node: Entity, name: &str, y: f32, seconds: f32) -> AnimationClip {
    let at = Vector3::new(0.0, y, 0.0);
    let track = KeyframeTrack::vector(
        node,
        TrackTarget::Position,
        vec![0.0, seconds],
        vec![at, at],
    );
    AnimationClip::new(name, seconds, vec![track])
}

fn game() -> (TestGame, Entity) {
    let mut node = None;
    let game = TestGame::new(|app| {
        app.system(animator_step);
        let id = app.world.spawn(Object3D::group());
        let mut animator = Animator::new();
        animator
            .state("idle", hold(id, "idle", 1.0, 1.0))
            .state("run", hold(id, "run", 2.0, 1.0))
            .once("jump", hold(id, "jump", 3.0, 0.25))
            .transition("idle", "run", Condition::above("speed", 0.1), 0.0)
            .transition("run", "idle", Condition::below("speed", 0.1), 0.0)
            .from_any("jump", Condition::trigger("jump"), 0.0)
            .transition("jump", "idle", Condition::Finished, 0.0);
        app.world.insert(id, animator);
        node = Some(id);
    });
    (game, node.unwrap())
}

fn state(game: &mut TestGame, id: Entity) -> String {
    let animator = game.world().get::<Animator>(id).unwrap();
    animator.current().unwrap_or("").to_owned()
}

fn y(game: &mut TestGame, id: Entity) -> f32 {
    game.world().scene.get(id).unwrap().position.y
}

#[test]
fn it_starts_in_the_first_state_and_moves_the_node() {
    let (mut game, id) = game();
    game.frames(3);
    assert_eq!(state(&mut game, id), "idle");
    assert!((y(&mut game, id) - 1.0).abs() < 1e-4);
}

#[test]
fn parameters_drive_transitions_both_ways() {
    let (mut game, id) = game();
    game.frames(2);
    game.world()
        .get_mut::<Animator>(id)
        .unwrap()
        .set("speed", 1.0);
    game.frames(3);
    assert_eq!(state(&mut game, id), "run");
    assert!(
        (y(&mut game, id) - 2.0).abs() < 1e-3,
        "{}",
        y(&mut game, id)
    );

    game.world()
        .get_mut::<Animator>(id)
        .unwrap()
        .set("speed", 0.0);
    game.frames(3);
    assert_eq!(state(&mut game, id), "idle");
    assert!((y(&mut game, id) - 1.0).abs() < 1e-3);
}

#[test]
fn a_one_shot_plays_through_reports_and_moves_on() {
    let (mut game, id) = game();
    game.frames(2);
    game.world()
        .get_mut::<Animator>(id)
        .unwrap()
        .trigger("jump");
    game.frames(3);
    assert_eq!(state(&mut game, id), "jump");
    assert!((y(&mut game, id) - 3.0).abs() < 1e-3);
    assert!(game.world().events::<AnimationFinished>().is_empty());

    game.seconds(0.3);
    assert_eq!(state(&mut game, id), "idle");
    let finished = game.world().events::<AnimationFinished>();
    assert_eq!(finished.len(), 1);
    assert_eq!(finished[0].state, "jump");
    assert_eq!(finished[0].entity, id);
}

#[test]
fn a_trigger_with_no_transition_is_dropped() {
    let (mut game, id) = game();
    game.frames(1);
    game.world()
        .get_mut::<Animator>(id)
        .unwrap()
        .trigger("nothing");
    game.frames(1);
    assert_eq!(state(&mut game, id), "idle");
}

#[test]
fn a_trigger_does_not_wait_for_a_state_that_could_use_it() {
    let mut node = None;
    let mut game = TestGame::new(|app| {
        app.system(animator_step);
        let id = app.world.spawn(Object3D::group());
        let mut animator = Animator::new();
        animator
            .state("idle", hold(id, "idle", 1.0, 1.0))
            .state("run", hold(id, "run", 2.0, 1.0))
            .state("dash", hold(id, "dash", 3.0, 1.0))
            .transition("idle", "run", Condition::above("speed", 0.1), 0.0)
            .transition("run", "dash", Condition::trigger("dash"), 0.0);
        app.world.insert(id, animator);
        node = Some(id);
    });
    let id = node.unwrap();
    game.frames(1);
    // Nothing in `idle` uses it, so it is gone by the time `run` could.
    game.world()
        .get_mut::<Animator>(id)
        .unwrap()
        .trigger("dash");
    game.frames(1);
    game.world()
        .get_mut::<Animator>(id)
        .unwrap()
        .set("speed", 1.0);
    game.frames(3);
    assert_eq!(state(&mut game, id), "run");
}

#[test]
fn a_crossfade_blends_the_two_poses() {
    let mut node = None;
    let mut game = TestGame::new(|app| {
        app.system(animator_step);
        let id = app.world.spawn(Object3D::group());
        let mut animator = Animator::new();
        animator
            .state("low", hold(id, "low", 0.0, 2.0))
            .state("high", hold(id, "high", 10.0, 2.0))
            .transition("low", "high", Condition::above("go", 0.5), 1.0);
        app.world.insert(id, animator);
        node = Some(id);
    });
    let id = node.unwrap();
    game.frames(2);
    game.world().get_mut::<Animator>(id).unwrap().set("go", 1.0);
    game.seconds(0.5);
    let mid = y(&mut game, id);
    assert!(mid > 2.0 && mid < 8.0, "half way through the fade: {mid}");
    game.seconds(0.7);
    assert!((y(&mut game, id) - 10.0).abs() < 1e-3);
}

#[test]
fn two_entities_animate_independently() {
    let mut ids = Vec::new();
    let mut game = TestGame::new(|app| {
        app.system(animator_step);
        for height in [1.0, 5.0] {
            let id = app.world.spawn(Object3D::group());
            let mut animator = Animator::new();
            animator.state("idle", hold(id, "idle", height, 1.0));
            app.world.insert(id, animator);
            ids.push(id);
        }
    });
    game.frames(3);
    assert!((y(&mut game, ids[0]) - 1.0).abs() < 1e-4);
    assert!((y(&mut game, ids[1]) - 5.0).abs() < 1e-4);
}

#[test]
fn a_pause_freezes_animation() {
    let track_end = Vector3::new(0.0, 10.0, 0.0);
    let mut node = None;
    let mut game = TestGame::new(|app| {
        app.system(animator_step);
        let id = app.world.spawn(Object3D::group());
        let track = KeyframeTrack::vector(
            id,
            TrackTarget::Position,
            vec![0.0, 1.0],
            vec![Vector3::ZERO, track_end],
        );
        let mut animator = Animator::new();
        animator.state("rise", AnimationClip::new("rise", 1.0, vec![track]));
        app.world.insert(id, animator);
        node = Some(id);
    });
    let id = node.unwrap();
    game.seconds(0.25);
    game.world().resource_mut::<Time>().paused = true;
    let at = y(&mut game, id);
    game.frames(10);
    assert_eq!(y(&mut game, id), at);
    assert!(at > 1.0);
}

#[test]
#[should_panic(expected = "no animation state named \"walk\"")]
fn a_misspelt_state_says_so() {
    Animator::new().from_any("walk", Condition::Finished, 0.0);
}
