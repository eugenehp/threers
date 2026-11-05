//! A camera that follows something.
//!
//! Mark what to follow with [`CameraTarget`], or name it with
//! [`ChaseCamera::follow`], and add [`ChaseCamera`] as a plugin. It moves
//! [`MainCamera`], which the plugin creates if the game has none.
//!
//! ```
//! # use threers::prelude::*;
//! # use threers_ecs::prelude::*;
//! # use threers_ecs::camera::{CameraTarget, ChaseCamera};
//! let mut app = App::new();
//! app.plugin(ChaseCamera::new().distance(9.0).height(3.5));
//! app.world.object(Object3D::group()).with(CameraTarget);
//! app.frame(1.0 / 60.0);
//! ```

use crate::app::{App, MainCamera, Plugin, Time};
use crate::query::With;
use crate::schedule::System;
use crate::world::{Component, Entity, World};
use threers::cameras::PerspectiveCamera;
use threers::math::{Quaternion, Vector3};

/// What the [`ChaseCamera`] follows, unless it has been told otherwise.
#[derive(Debug, Clone, Copy, Default)]
pub struct CameraTarget;
impl Component for CameraTarget {}

/// Follows a target from behind, along the way it faces.
///
/// It stays the same distance back at any speed, and what it smooths is the
/// turning: smoothing where the camera *is* would leave it further behind the
/// faster the target goes. All of it is by elapsed time, so it behaves the same at
/// any frame rate.
#[derive(Debug, Clone)]
pub struct ChaseCamera {
    /// Who to follow, ahead of any [`CameraTarget`].
    pub target: Option<Entity>,
    /// Look back over the target, for as long as this is set.
    pub looking_back: bool,
    distance: f32,
    height: f32,
    look_ahead: f32,
    look_height: f32,
    stiffness: f32,
    fov: f32,
    fov_boost: f32,
    full_speed: f32,
    speed_pull: f32,
    wall_margin: Option<f32>,
    shake: f32,
    state: Option<State>,
}

#[derive(Debug, Clone, Copy)]
struct State {
    /// Which way the camera looks, as an angle on the ground.
    yaw: f32,
    /// How high the target is, smoothed so bumps do not reach the picture.
    level: f32,
    target: Vector3,
    /// How close to `full_speed` the target is, 0 to 1.
    pace: f32,
    /// How much of the way out a wall has left it, 0 to 1.
    reach: f32,
}

impl Default for ChaseCamera {
    fn default() -> Self {
        Self::new()
    }
}

impl ChaseCamera {
    pub fn new() -> Self {
        Self {
            target: None,
            looking_back: false,
            distance: 8.0,
            height: 3.0,
            look_ahead: 4.0,
            look_height: 0.5,
            stiffness: 6.0,
            fov: 60.0,
            fov_boost: 0.0,
            full_speed: 50.0,
            speed_pull: 0.0,
            wall_margin: None,
            shake: 0.0,
            state: None,
        }
    }

    /// Follow this entity, ahead of any [`CameraTarget`].
    pub fn follow(mut self, target: Entity) -> Self {
        self.target = Some(target);
        self
    }

    /// How far behind the target.
    pub fn distance(mut self, distance: f32) -> Self {
        self.distance = distance;
        self
    }

    /// How far above the target.
    pub fn height(mut self, height: f32) -> Self {
        self.height = height;
        self
    }

    /// Where to look: this far ahead of the target and this far above it.
    pub fn look_at(mut self, ahead: f32, above: f32) -> Self {
        self.look_ahead = ahead;
        self.look_height = above;
        self
    }

    /// How quickly it swings round: the rate of an exponential, per second.
    pub fn stiffness(mut self, stiffness: f32) -> Self {
        self.stiffness = stiffness;
        self
    }

    /// Field of view at rest, in degrees.
    pub fn fov(mut self, degrees: f32) -> Self {
        self.fov = degrees;
        self
    }

    /// What counts as flat out, for [`fov_boost`](Self::fov_boost) and
    /// [`speed_pull`](Self::speed_pull).
    pub fn full_speed(mut self, speed: f32) -> Self {
        self.full_speed = speed.max(f32::EPSILON);
        self
    }

    /// Widen the view by this many degrees at full speed.
    pub fn fov_boost(mut self, degrees: f32) -> Self {
        self.fov_boost = degrees;
        self
    }

    /// Drop this much further back at full speed.
    pub fn speed_pull(mut self, distance: f32) -> Self {
        self.speed_pull = distance;
        self
    }

    /// Come in rather than go through a wall, stopping this far short of it.
    /// Needs the `physics` feature and a [`Physics`](crate::physics::Physics).
    pub fn avoid_walls(mut self, margin: f32) -> Self {
        self.wall_margin = Some(margin);
        self
    }

    /// Jump straight to the target next frame, after a teleport.
    pub fn snap(&mut self) {
        self.state = None;
    }

    /// Shake by this many world units, dying away in about a third of a second.
    pub fn shake(&mut self, amount: f32) {
        self.shake = self.shake.max(amount);
    }
}

impl Plugin for ChaseCamera {
    fn build(self, app: &mut App) {
        if app.world.get_resource::<MainCamera>().is_none() {
            let camera = PerspectiveCamera::new(self.fov, 16.0 / 9.0, 0.1, 1000.0);
            app.world.insert_resource(MainCamera(camera));
        }
        app.world.insert_resource(self);
        let system = System::new(chase_camera);
        #[cfg(feature = "physics")]
        let system = system.after_if_added(crate::physics::physics_sync);
        app.system(system);
    }
}

/// `from`, turned towards `to` by `blend`, the short way round.
fn turn(from: f32, to: f32, blend: f32) -> f32 {
    use std::f32::consts::{PI, TAU};
    from + ((to - from + PI).rem_euclid(TAU) - PI) * blend
}

/// Move [`MainCamera`] behind its target.
pub fn chase_camera(w: &mut World) {
    let chase = w.resource::<ChaseCamera>();
    let named = chase.target.filter(|&id| w.is_alive(id));
    let marked = || {
        w.single::<(Entity, With<CameraTarget>)>()
            .map(|found| found.0)
    };
    let Some(id) = named.or_else(marked) else {
        return;
    };
    let Some((target, rotation)) = w.world_transform(id) else {
        return;
    };
    let time = *w.resource::<Time>();
    let dt = time.delta.max(0.0);

    let chase = w.resource::<ChaseCamera>();
    let blend = 1.0 - (-chase.stiffness * dt).exp();
    let wanted_yaw = facing_yaw(rotation)
        + if chase.looking_back {
            std::f32::consts::PI
        } else {
            0.0
        };
    let mut state = chase.state.unwrap_or(State {
        yaw: wanted_yaw,
        level: target.y,
        target,
        pace: 0.0,
        reach: 1.0,
    });
    let speed = if dt > 0.0 {
        (target - state.target).length() / dt
    } else {
        0.0
    };
    state.target = target;
    state.yaw = turn(state.yaw, wanted_yaw, blend);
    state.level += (target.y - state.level) * blend;
    state.pace += ((speed / chase.full_speed).min(1.0) - state.pace) * blend;

    let facing = Vector3::new(state.yaw.sin(), 0.0, state.yaw.cos());
    let base = Vector3::new(target.x, state.level, target.z);
    let back = chase.distance + chase.speed_pull * state.pace;
    let lift = Vector3::new(0.0, chase.height, 0.0);
    let look = base + facing * chase.look_ahead + Vector3::new(0.0, chase.look_height, 0.0);
    let fov = chase.fov + chase.fov_boost * state.pace;

    let open = room_behind(w, id, base + lift, facing, back);
    // In at once, or a wall is in shot; out again gently.
    state.reach = if open < state.reach {
        open
    } else {
        state.reach + (open - state.reach) * blend
    };
    let eye = base + lift - facing * (back * state.reach);

    let chase = w.resource_mut::<ChaseCamera>();
    chase.state = Some(state);
    let wobble = chase.shake;
    chase.shake *= (-12.0 * dt).exp();
    // By the frame number, so a test repeats exactly and a pause holds still.
    let phase = time.frame as f32 * 1.7;
    let offset = Vector3::new(phase.sin(), (phase * 1.3).cos(), 0.0) * wobble;

    let camera = &mut w.resource_mut::<MainCamera>().0;
    camera.position = eye + offset;
    camera.set_fov_degrees(fov);
    camera.look_at(look);
}

/// How much of `back` is clear behind `from`, 0 to 1.
#[cfg(feature = "physics")]
fn room_behind(w: &World, id: Entity, from: Vector3, facing: Vector3, back: f32) -> f32 {
    use crate::physics::{Body, Physics, QueryFilter};
    let Some(margin) = w.resource::<ChaseCamera>().wall_margin else {
        return 1.0;
    };
    let Some(physics) = w.get_resource::<Physics>() else {
        return 1.0;
    };
    let ray = threers::math::Ray::new(from, facing * -1.0);
    let filter = QueryFilter::default().exclude(w.get::<Body>(id).map(|body| body.0));
    match physics.raycast(&ray, back + margin, filter) {
        Some(hit) => ((hit.toi - margin) / back).clamp(0.05, 1.0),
        None => 1.0,
    }
}

#[cfg(not(feature = "physics"))]
fn room_behind(_: &World, _: Entity, _: Vector3, _: Vector3, _: f32) -> f32 {
    1.0
}

/// The way a rotation faces, as an angle on the ground. Pointing straight up has
/// no such angle, so that reads as the default.
fn facing_yaw(rotation: Quaternion) -> f32 {
    let forward = Vector3::new(0.0, 0.0, -1.0).apply_quaternion(rotation);
    if forward.x.abs() + forward.z.abs() > 1e-4 {
        forward.x.atan2(forward.z)
    } else {
        std::f32::consts::PI
    }
}
