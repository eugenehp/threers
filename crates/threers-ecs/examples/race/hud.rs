//! What is drawn over the race: lap, place, times, speed, the start lights, a map
//! of the circuit, and a warning when going the wrong way.

use super::rules::{standings, Phase, Player, Race, Racer, COUNTDOWN};
use super::track::Track;
use threers_ecs::prelude::*;

pub fn clock_text(seconds: f32) -> String {
    format!("{}:{:05.2}", (seconds / 60.0) as u32, seconds % 60.0)
}

/// The circuit, scaled into a square of `size` pixels, and where each car is.
struct Map {
    outline: Vec<Vector2>,
    cars: Vec<(Vector2, bool)>,
}

fn map(w: &World, left: f32, top: f32, size: f32) -> Map {
    let track = w.resource::<Track>();
    let (mut min, mut max) = (
        Vector2::new(f32::MAX, f32::MAX),
        Vector2::new(f32::MIN, f32::MIN),
    );
    for p in &track.points {
        min = Vector2::new(min.x.min(p.x), min.y.min(p.z));
        max = Vector2::new(max.x.max(p.x), max.y.max(p.z));
    }
    let scale = size / (max.x - min.x).max(max.y - min.y);
    let place =
        |x: f32, z: f32| Vector2::new(left + (x - min.x) * scale, top + (z - min.y) * scale);
    let mut outline: Vec<Vector2> = track
        .points
        .iter()
        .step_by(4)
        .map(|p| place(p.x, p.z))
        .collect();
    outline.push(outline[0]);
    let cars = w
        .iter::<(NodeRef, &Racer, Option<&Player>)>()
        .map(|(node, _, player)| (place(node.position.x, node.position.z), player.is_some()))
        .collect();
    Map { outline, cars }
}

pub fn hud(w: &mut World) {
    let order = standings(w);
    let phase = w.state::<Phase>();
    let race = w.resource::<Race>();
    let (laps, elapsed, count) = (race.laps, race.elapsed, race.countdown);
    let Some((id, racer, car, _)) = w.single::<(Entity, &Racer, &Car, With<Player>)>() else {
        return;
    };
    let speed = car.speed;
    let lap = racer.laps_started.clamp(1, laps);
    let lap_time = if racer.laps_started > 0 && racer.finished.is_none() {
        elapsed - racer.lap_started
    } else {
        0.0
    };
    let (last, best, wrong_way) = (racer.last_lap, racer.best_lap, racer.wrong_way());
    let place = order.iter().position(|entry| entry.0 == id).unwrap_or(0) + 1;
    let names: Vec<String> = order
        .iter()
        .enumerate()
        .filter_map(|(i, entry)| {
            let racer = w.get::<Racer>(entry.0)?;
            let time = entry.2.map(clock_text).unwrap_or_default();
            let best = racer
                .best_lap
                .map(|b| format!("best {}", clock_text(b)))
                .unwrap_or_default();
            Some(format!(
                "{}  {:<7} {:>8}  {}",
                i + 1,
                racer.name,
                time,
                best
            ))
        })
        .collect();
    let size = w.get_resource::<Hud>().map(|h| h.size()).unwrap_or((1, 1));
    let (width, height) = (size.0 as f32, size.1 as f32);
    let map_size = (height * 0.26).clamp(110.0, 220.0);
    let map = map(w, 20.0, height - map_size - 20.0, map_size);
    let started_for = if phase == Phase::Racing { elapsed } else { 0.0 };

    with_hud(w, |hud| {
        hud.no_stroke();
        hud.fill_alpha(Color::from_hex(0x000000), 0.35);
        hud.rounded_rect(14.0, 14.0, 250.0, 200.0, 10.0);
        hud.fill(Color::WHITE);
        hud.text_size(28.0);
        hud.text(28.0, 48.0, &format!("LAP {lap}/{laps}"));
        hud.text(28.0, 84.0, &format!("POS {place}/{}", order.len()));
        hud.text_size(20.0);
        hud.text(28.0, 116.0, &format!("time   {}", clock_text(elapsed)));
        hud.text(28.0, 142.0, &format!("lap    {}", clock_text(lap_time)));
        hud.text(
            28.0,
            168.0,
            &format!(
                "last   {}",
                last.map(clock_text).unwrap_or_else(|| "-".into())
            ),
        );
        hud.text(
            28.0,
            194.0,
            &format!(
                "best   {}",
                best.map(clock_text).unwrap_or_else(|| "-".into())
            ),
        );

        // Speed, as a number and a bar.
        hud.text_size(44.0);
        hud.text(
            width - 230.0,
            height - 44.0,
            &format!("{:3.0} km/h", speed.abs() * 3.6),
        );
        hud.fill_alpha(Color::from_hex(0x000000), 0.35);
        hud.rect(width - 230.0, height - 28.0, 200.0, 8.0);
        hud.fill(Color::from_hex(0xffcc33));
        hud.rect(
            width - 230.0,
            height - 28.0,
            200.0 * (speed.abs() / 75.0).min(1.0),
            8.0,
        );

        // The map.
        hud.fill_alpha(Color::from_hex(0x000000), 0.35);
        hud.rounded_rect(
            10.0,
            height - map_size - 30.0,
            map_size + 20.0,
            map_size + 20.0,
            10.0,
        );
        hud.no_fill();
        hud.stroke_alpha(Color::WHITE, 0.8);
        hud.stroke_weight(3.0);
        hud.polyline(&map.outline);
        hud.no_stroke();
        for (at, me) in &map.cars {
            hud.fill(if *me {
                Color::from_hex(0xff3b30)
            } else {
                Color::from_hex(0xffffff)
            });
            hud.circle(at.x, at.y, if *me { 6.0 } else { 4.5 });
        }

        // Start lights: five reds filling, all out at the start, a green after.
        if phase == Phase::Countdown || (phase == Phase::Racing && started_for < 1.0) {
            let lit = ((COUNTDOWN - count) / COUNTDOWN * 5.0).floor() as i32;
            let (r, gap) = (22.0, 62.0);
            let x0 = width / 2.0 - gap * 2.0;
            hud.fill_alpha(Color::from_hex(0x000000), 0.6);
            hud.rounded_rect(x0 - 40.0, 70.0, gap * 4.0 + 80.0, 76.0, 14.0);
            for k in 0..5 {
                let colour = match phase {
                    Phase::Racing => Color::from_hex(0x30d158),
                    _ if k < lit => Color::from_hex(0xff3b30),
                    _ => Color::from_hex(0x3a1515),
                };
                hud.fill(colour);
                hud.circle(x0 + gap * k as f32, 108.0, r);
            }
        }

        if wrong_way && phase == Phase::Racing {
            hud.fill_alpha(Color::from_hex(0xb00000), 0.75);
            hud.rounded_rect(width / 2.0 - 170.0, height / 2.0 - 60.0, 340.0, 80.0, 12.0);
            hud.fill(Color::WHITE);
            hud.text_size(46.0);
            hud.text(width / 2.0 - 140.0, height / 2.0 - 5.0, "WRONG WAY");
        }

        if phase == Phase::Finished {
            hud.fill_alpha(Color::from_hex(0x000000), 0.6);
            hud.rounded_rect(
                width / 2.0 - 260.0,
                height / 2.0 - 120.0,
                520.0,
                110.0 + names.len() as f32 * 40.0,
                14.0,
            );
            hud.fill(Color::WHITE);
            hud.text_size(30.0);
            for (i, line) in names.iter().enumerate() {
                hud.text(
                    width / 2.0 - 230.0,
                    height / 2.0 - 70.0 + i as f32 * 40.0,
                    line,
                );
            }
            hud.text_size(22.0);
            hud.text(
                width / 2.0 - 230.0,
                height / 2.0 - 60.0 + names.len() as f32 * 40.0 + 20.0,
                "ENTER to race again",
            );
        }
    });
}
