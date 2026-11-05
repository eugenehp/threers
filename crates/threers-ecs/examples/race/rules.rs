//! Laps, places, the clock, and putting cars back on the road.

use super::ai::Driver;
use super::cars::{play, Sounds};
use super::track::{Track, ROAD_HALF_WIDTH};
use threers_ecs::prelude::*;

pub const LAPS: u32 = 3;
pub const COUNTDOWN: f32 = 3.0;

/// One competitor's place in the race.
pub struct Racer {
    pub name: &'static str,
    /// Road covered since the line, in units. It falls when going backwards, so a
    /// lap cannot be had by turning round on the line.
    pub distance: f32,
    pub index: usize,
    pub finished: Option<f32>,
    pub stuck_for: f32,
    /// Laps begun: 0 on the grid, behind the line.
    pub laps_started: u32,
    /// When the lap now running began, on the race clock.
    pub lap_started: f32,
    pub last_lap: Option<f32>,
    pub best_lap: Option<f32>,
    /// How long it has been pointing the wrong way while moving.
    pub wrong_way_for: f32,
}
impl Component for Racer {}

impl Racer {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            distance: 0.0,
            index: 0,
            finished: None,
            stuck_for: 0.0,
            laps_started: 0,
            lap_started: 0.0,
            last_lap: None,
            best_lap: None,
            wrong_way_for: 0.0,
        }
    }

    pub fn wrong_way(&self) -> bool {
        self.wrong_way_for > 1.0
    }
}

/// The car the keys drive.
pub struct Player;
impl Component for Player {}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Phase {
    Countdown,
    Racing,
    Finished,
}
impl GameState for Phase {}

/// The clock, and how many laps this race is.
pub struct Race {
    pub laps: u32,
    pub countdown: f32,
    pub elapsed: f32,
}

/// A lap was completed: whose, how long it took, and whether it was their best.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LapDone {
    pub racer: Entity,
    pub time: f32,
    pub best: bool,
}

/// Five lights fill during the countdown and go out together at the start.
pub fn countdown(w: &mut World) {
    let dt = w.resource::<Time>().delta;
    let race = w.resource_mut::<Race>();
    let lit = |left: f32| ((COUNTDOWN - left) / COUNTDOWN * 5.0).floor() as i32;
    let before = lit(race.countdown);
    race.countdown -= dt;
    let (after, done) = (lit(race.countdown), race.countdown <= 0.0);
    if done {
        w.set_state(Phase::Racing);
        play(w, |s: &Sounds| &s.go);
    } else if after != before {
        play(w, |s: &Sounds| &s.beep);
    }
}

pub fn hold_at_the_line(w: &mut World) {
    w.each::<&mut Car>(|car| {
        car.throttle = 0.0;
        car.brake = 1.0;
    });
}

pub fn control(w: &mut World) {
    let input = w.resource::<Input>();
    let (steer, drive) = (input.axis("steer"), input.axis("drive"));
    let handbrake = input.pressed("handbrake");
    w.each::<(&mut Car, With<Player>, Without<Driver>)>(|(car, _, _)| {
        car.steer = steer;
        car.throttle = drive;
        car.brake = 0.0;
        car.handbrake = handbrake;
    });
}

/// Grass is slow.
pub fn surface(w: &mut World) {
    w.with_resource::<Track, _>(|w, track| {
        w.each::<(NodeRef, &mut Car, &Racer)>(|(node, car, racer)| {
            let off = track.offset(node.position, racer.index).abs() > ROAD_HALF_WIDTH;
            car.drag = if off { 0.1 } else { 0.0 };
        });
    });
}

pub fn progress(w: &mut World) {
    let (elapsed, laps) = {
        let race = w.resource::<Race>();
        (race.elapsed, race.laps)
    };
    let dt = w.resource::<Time>().delta;
    let mut player_done = false;
    let mut done_laps = Vec::new();
    w.with_resource::<Track, _>(|w, track| {
        w.each::<(Entity, NodeRef, &Car, &mut Racer, Option<&Player>)>(
            |(id, node, car, racer, player)| {
                let mut index = track.nearest(node.position, racer.index);
                // Far from where it was: put back, or knocked across the infield.
                if node.position.distance_to(track.points[index]) > 30.0 {
                    index = track.nearest_anywhere(node.position);
                }
                let half = track.points.len() as isize / 2;
                let moved = (index as isize - racer.index as isize + half)
                    .rem_euclid(track.points.len() as isize)
                    - half;
                racer.index = index;
                racer.distance += moved as f32 * track.spacing();

                // Each time the line is crossed going forwards, a new lap begins.
                let started = (racer.distance / track.length).floor() as i64 + 1;
                if started > racer.laps_started as i64 && racer.finished.is_none() {
                    if racer.laps_started > 0 {
                        let time = elapsed - racer.lap_started;
                        let best = racer.best_lap.is_none_or(|b| time < b);
                        racer.last_lap = Some(time);
                        if best {
                            racer.best_lap = Some(time);
                        }
                        done_laps.push(LapDone {
                            racer: id,
                            time,
                            best,
                        });
                    }
                    racer.laps_started = started.max(0) as u32;
                    racer.lap_started = elapsed;
                }
                if racer.finished.is_none() && racer.distance >= laps as f32 * track.length {
                    racer.finished = Some(elapsed);
                    player_done |= player.is_some();
                }

                let forward = Vector3::new(0.0, 0.0, -1.0).apply_quaternion(node.quaternion);
                let backwards = forward.dot(track.tangents[index]) < -0.3 && car.speed.abs() > 2.0;
                racer.wrong_way_for = if backwards {
                    racer.wrong_way_for + dt
                } else {
                    0.0
                };
            },
        );
    });
    for lap in done_laps {
        w.send(lap);
    }
    if player_done {
        // A rival takes the wheel for the lap of honour.
        for id in w.entities::<With<Player>>() {
            w.insert(id, Driver::new(0.0, 0.5));
        }
        w.set_state(Phase::Finished);
    }
}

pub fn clock(w: &mut World) {
    let dt = w.resource::<Time>().delta;
    w.resource_mut::<Race>().elapsed += dt;
}

/// Put a car back on the road where it left it.
pub fn recover(w: &mut World, id: Entity) {
    let Some(index) = w.get::<Racer>(id).map(|racer| racer.index) else {
        return;
    };
    let (at, facing) = {
        let track = w.resource::<Track>();
        (
            track.surface(index, 0.0) + Vector3::new(0.0, 0.8, 0.0),
            track.facing(index),
        )
    };
    w.teleport_body(id, at, facing);
    if w.has::<Player>(id) {
        w.resource_mut::<ChaseCamera>().snap();
    }
}

/// Rivals that have not moved for a while, or anyone on their roof.
pub fn rescue(w: &mut World) {
    let dt = w.resource::<Time>().delta;
    let mut lost = Vec::new();
    w.each::<(Entity, NodeRef, &Car, &mut Racer, Option<&Driver>)>(
        |(id, node, car, racer, driver)| {
            let up = Vector3::UP.apply_quaternion(node.quaternion);
            let idle = car.speed.abs() < 1.0 && driver.is_some();
            racer.stuck_for = if idle || up.y < 0.2 {
                racer.stuck_for + dt
            } else {
                0.0
            };
            if racer.stuck_for > 2.0 {
                racer.stuck_for = 0.0;
                lost.push(id);
            }
        },
    );
    for id in lost {
        recover(w, id);
    }
}

pub fn player_recover(w: &mut World) {
    if w.resource::<Input>().just_pressed("recover") {
        for id in w.entities::<With<Player>>() {
            recover(w, id);
        }
    }
}

pub fn line_up(w: &mut World) {
    let mut cars = w.entities::<With<Car>>();
    cars.sort_by_key(|&id| w.has::<Player>(id));
    let track = w.resource::<Track>();
    let last = track.points.len() as isize;
    let spots: Vec<(Vector3, Quaternion, usize, f32)> = (0..cars.len())
        .map(|slot| {
            let back = 3 + slot as isize * 3;
            let index = track.wrap(last - back);
            let side = if slot % 2 == 0 { 3.0 } else { -3.0 };
            let at = track.surface(index, side) + Vector3::new(0.0, 0.8, 0.0);
            (
                at,
                track.facing(index),
                index,
                -(back as f32) * track.spacing(),
            )
        })
        .collect();
    for (&id, (at, facing, index, distance)) in cars.iter().zip(spots) {
        w.teleport_body(id, at, facing);
        if let Some(racer) = w.get_mut::<Racer>(id) {
            let name = racer.name;
            let best = racer.best_lap;
            *racer = Racer {
                index,
                distance,
                best_lap: best,
                ..Racer::new(name)
            };
        }
    }
    w.resource_mut::<ChaseCamera>().snap();
}

/// Back to the grid and the lights.
pub fn start_again(w: &mut World) {
    for id in w.entities::<With<Player>>() {
        w.remove::<Driver>(id);
    }
    line_up(w);
    let race = w.resource_mut::<Race>();
    race.countdown = COUNTDOWN;
    race.elapsed = 0.0;
    w.set_state(Phase::Countdown);
}

pub fn restart(w: &mut World) {
    if w.resource::<Input>().just_pressed("restart") {
        start_again(w);
    }
}

/// Everyone, leader first: those home by time, then by road covered.
pub fn standings(w: &World) -> Vec<(Entity, f32, Option<f32>)> {
    let mut order: Vec<_> = w
        .iter::<(Entity, &Racer)>()
        .map(|(id, racer)| (id, racer.distance, racer.finished))
        .collect();
    order.sort_by(|a, b| match (a.2, b.2) {
        (Some(x), Some(y)) => x.total_cmp(&y),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => b.1.total_cmp(&a.1),
    });
    order
}
