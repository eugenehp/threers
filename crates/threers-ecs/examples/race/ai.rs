//! Rivals: follow the road at the speed its bends allow, and get round each other
//! rather than through.

use super::rules::Racer;
use super::track::{Track, ROAD_HALF_WIDTH};
use threers_ecs::prelude::*;

/// A car that drives itself.
pub struct Driver {
    /// How far off the centre it likes to be, positive to the left.
    pub lane: f32,
    /// How close to the limit of grip it dares go, 0 to 1.
    pub nerve: f32,
    /// The line it is on now, gliding towards where it wants to be.
    pub line: f32,
    /// Whether it is going round someone.
    pub passing: bool,
}
impl Component for Driver {}

impl Driver {
    pub fn new(lane: f32, nerve: f32) -> Self {
        Self {
            lane,
            nerve,
            line: lane,
            passing: false,
        }
    }
}

/// What the tyres hold in a corner, and how hard the brakes stop.
const GRIP: f32 = 13.0;
const BRAKING: f32 = 11.0;
/// How far ahead to look for someone in the way, and how wide a car's lane is.
const LOOK: f32 = 35.0;
const CLEARANCE: f32 = 3.4;
/// How fast a driver moves across the road, units per second.
const SWERVE: f32 = 3.0;

struct Seen {
    id: Entity,
    index: usize,
    offset: f32,
    speed: f32,
}

/// Steer at a point up the road, slow for whatever bend is coming, and pass or
/// follow whoever is in the way.
pub fn drive_rivals(w: &mut World) {
    let dt = w.resource::<Time>().delta;
    w.with_resource::<Track, _>(|w, track| {
        let field: Vec<Seen> = w
            .iter::<(Entity, NodeRef, &Car, &Racer)>()
            .map(|(id, node, car, racer)| Seen {
                id,
                index: racer.index,
                offset: track.offset(node.position, racer.index),
                speed: car.speed,
            })
            .collect();
        let room = ROAD_HALF_WIDTH - 1.6;
        w.each::<(Entity, NodeRef, &mut Car, &Racer, &mut Driver)>(
            |(id, node, car, racer, driver)| {
                let speed = car.speed.max(0.0);
                let me = track.offset(node.position, racer.index);
                // How far up the road another car is, by where they both are now:
                // race distance says nothing about a car a lap down or just put back.
                let count = track.points.len() as isize;
                let gap = |other: &Seen| {
                    let steps = (other.index as isize - racer.index as isize + count / 2)
                        .rem_euclid(count)
                        - count / 2;
                    steps as f32 * track.spacing()
                };
                let blocked_at = |lane: f32| {
                    field
                        .iter()
                        .filter(|other| other.id != id)
                        .filter(|other| {
                            let gap = gap(other);
                            gap > -2.0 && gap < LOOK && (other.offset - lane).abs() < CLEARANCE
                        })
                        .min_by(|a, b| gap(a).total_cmp(&gap(b)))
                };

                // Someone in the way on the line it wants: go round on the side with
                // more room, or follow if neither has any.
                let mut follow: Option<f32> = None;
                let mut target = driver.lane;
                driver.passing = false;
                if let Some(ahead) = blocked_at(driver.lane) {
                    let sides = [
                        ahead.offset + CLEARANCE + 0.4,
                        ahead.offset - CLEARANCE - 0.4,
                    ];
                    let open = sides
                        .into_iter()
                        .filter(|lane| lane.abs() <= room)
                        .filter(|&lane| {
                            blocked_at(lane).is_none_or(|b| {
                                b.id == ahead.id && (b.offset - lane).abs() >= CLEARANCE
                            })
                        })
                        .min_by(|a, b| (a - me).abs().total_cmp(&(b - me).abs()));
                    match open {
                        Some(lane) => {
                            target = lane;
                            driver.passing = true;
                        }
                        None => follow = Some(ahead.speed - 1.0),
                    }
                }
                let step = SWERVE * dt;
                driver.line += (target - driver.line).clamp(-step, step);

                let reach = (6.0 + speed * 0.45) / track.spacing();
                let ahead = track.wrap(racer.index as isize + reach as isize);
                let aim = track.surface(ahead, driver.line);
                let to_aim = aim - node.position;
                let forward = Vector3::new(0.0, 0.0, -1.0).apply_quaternion(node.quaternion);
                let right = Vector3::new(-forward.z, 0.0, forward.x).normalize();
                let bearing = to_aim.dot(right).atan2(to_aim.dot(forward));
                car.steer = (bearing * 3.0).clamp(-1.0, 1.0);

                // The slowest the road ahead allows, less what braking can shed.
                let mut limit = f32::MAX;
                for step in 0..40 {
                    let at = track.wrap(racer.index as isize + step);
                    let corner = (GRIP * driver.nerve / track.curvature[at].max(1e-4)).sqrt();
                    let room = step as f32 * track.spacing();
                    limit = limit.min((corner * corner + 2.0 * BRAKING * room).sqrt());
                }
                if let Some(behind) = follow {
                    limit = limit.min(behind.max(4.0));
                }
                car.handbrake = false;
                if speed > limit {
                    car.throttle = 0.0;
                    car.brake = ((speed - limit) * 0.4).clamp(0.2, 1.0);
                } else {
                    car.throttle = 1.0;
                    car.brake = 0.0;
                }
            },
        );
    });
}
