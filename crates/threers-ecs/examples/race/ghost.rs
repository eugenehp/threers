//! The player's best lap, driven again by a see-through car.

use super::rules::{LapDone, Phase, Player, Race, Racer};
use threers_ecs::prelude::*;

/// Where the car was, this long into a lap.
#[derive(Clone, Copy)]
struct Sample {
    time: f32,
    at: Vector3,
    turned: Quaternion,
}

/// The lap being driven, the best so far, and the ghost that drives it.
pub struct Ghost {
    pub node: Entity,
    current: Vec<Sample>,
    best: Vec<Sample>,
}

impl Ghost {
    pub fn spawn(w: &mut World, half: Vector3) -> Self {
        let mut material = StandardMaterial::new(Color::from_hex(0x9fd8ff))
            .with_emissive(Color::from_hex(0x3a80c0), 0.4);
        material.opacity = 0.35;
        material.transparent = Some(true);
        let node = w
            .mesh(
                BoxGeometry::new(half.x * 2.0, half.y * 2.0, half.z * 2.0),
                material,
            )
            .named("ghost")
            .shadows(false, false)
            .id();
        w.node_mut(node).unwrap().visible = false;
        Self {
            node,
            current: Vec::new(),
            best: Vec::new(),
        }
    }

    /// Whether there is a best lap to race.
    pub fn has_lap(&self) -> bool {
        !self.best.is_empty()
    }

    /// Where the best lap was `time` into it, between samples.
    pub fn pose(&self, time: f32) -> Option<(Vector3, Quaternion)> {
        let next = self.best.iter().position(|s| s.time >= time)?;
        if next == 0 {
            return Some((self.best[0].at, self.best[0].turned));
        }
        let (a, b) = (self.best[next - 1], self.best[next]);
        let t = ((time - a.time) / (b.time - a.time).max(1e-6)).clamp(0.0, 1.0);
        Some((a.at.lerp(b.at, t), a.turned.slerp(b.turned, t)))
    }
}

/// Note the player's car each step, and keep the lap if it was the best.
pub fn record_ghost(w: &mut World) {
    let elapsed = w.resource::<Race>().elapsed;
    let laps = w.events::<LapDone>();
    let Some((id, node, racer, _)) = w.single::<(Entity, NodeRef, &Racer, With<Player>)>() else {
        return;
    };
    let sample = Sample {
        time: elapsed - racer.lap_started,
        at: node.position,
        turned: node.quaternion,
    };
    let started = racer.laps_started > 0;
    let best_lap = laps.iter().any(|lap| lap.racer == id && lap.best);
    let finished_lap = laps.iter().any(|lap| lap.racer == id);
    let ghost = w.resource_mut::<Ghost>();
    if finished_lap {
        if best_lap {
            ghost.best = std::mem::take(&mut ghost.current);
        }
        ghost.current.clear();
    }
    if started {
        ghost.current.push(sample);
    }
}

/// Put the ghost where the best lap was at this point of the current one.
pub fn move_ghost(w: &mut World) {
    let racing = w.state::<Phase>() == Phase::Racing;
    let elapsed = w.resource::<Race>().elapsed;
    let time = w
        .single::<(&Racer, With<Player>)>()
        .filter(|(r, _)| r.laps_started > 0)
        .map(|(r, _)| elapsed - r.lap_started);
    let ghost = w.resource::<Ghost>();
    let pose = time.filter(|_| racing).and_then(|t| ghost.pose(t));
    let node = ghost.node;
    if let Some(n) = w.node_mut(node) {
        n.visible = pose.is_some();
        if let Some((at, turned)) = pose {
            n.position = at;
            n.quaternion = turned;
        }
    }
}

/// A new race keeps the best lap and forgets the half-driven one.
pub fn reset_ghost(w: &mut World) {
    w.resource_mut::<Ghost>().current.clear();
}
