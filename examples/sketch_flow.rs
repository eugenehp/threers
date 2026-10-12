//! A p5-shaped sketch: a window, a loop, and a curl-noise flow field.
//!
//! ```text
//! cargo run --release --example sketch_flow
//! cargo run --release --example sketch_flow -- --frames 120   # run and quit
//! ```
//!
//! Everything this uses was added because d3 and p5 have it and this crate did
//! not. The whole file is about sixty lines of actual work, and the version of
//! it that existed before would have opened with eighty lines of event loop,
//! adapter and surface configuration before the first cube.
//!
//! | | |
//! |---|---|
//! | [`sketch`] | the window and the loop, with `frame`, `elapsed`, `delta` and the pointer |
//! | [`SimplexNoise::curl2`] | a divergence-free flow field, so the particles swirl instead of piling into sinks |
//! | [`Rng`] | one seeded generator, rather than the twenty-seventh hand-rolled xorshift |
//! | [`ColorRamp`] | viridis, blended through OkLab |
//! | [`map_range`] | speed to colour, without writing the arithmetic out |
//! | [`Easing`] | the fade-in, which is a curve and not a ramp |
//! | [`Frame::overlay`] | the read-out, the ramp legend and the cursor ring |
//! | [`Frame::scale_factor`] | so the type is the same size on a 2x display |

use threers::easing::Easing;
use threers::math::{ColorRamp, Rng};
use threers::prelude::*;
use threers::sketch::{sketch, Frame, SketchOptions};
use threers::SimplexNoise;

/// How many particles. Each is an instanced cube, so this is cheap.
const COUNT: usize = 1400;
/// How far the field reaches before a particle is recycled.
const BOUNDS: f32 = 6.0;

struct Particle {
    pos: Vector3,
    speed: f32,
    age: f32,
    life: f32,
}

fn main() {
    let frame_limit: Option<u64> = std::env::args()
        .collect::<Vec<_>>()
        .windows(2)
        .find(|w| w[0] == "--frames")
        .and_then(|w| w[1].parse().ok());

    sketch(
        SketchOptions::new("threers — flow field")
            .size(960, 720)
            .background(Color::from_hex(0x08080c))
            .msaa(4),
        move |s| {
            s.camera.position = Vector3::new(0.0, 3.5, 11.0);
            s.camera.look_at(Vector3::new(0.0, 0.0, 0.0));
            s.scene.add_light(AmbientLight::new(Color::WHITE, 0.55));
            let mut key = Object3D::light(
                DirectionalLight::new(Color::from_hex(0xfff0e0), 1.6)
                    .with_direction(Vector3::new(-0.4, -1.0, -0.3)),
            );
            key.position = Vector3::new(4.0, 7.0, 3.0);
            s.scene.add(key);

            // One generator, seeded, so the same run draws the same thing.
            let mut rng = Rng::new(0xC0FFEE);
            let ramp = ColorRamp::viridis();
            let geometry = BoxGeometry::new(0.07, 0.07, 0.07);

            let mut particles = Vec::with_capacity(COUNT);
            let mut ids = Vec::with_capacity(COUNT);
            for _ in 0..COUNT {
                let p = Particle {
                    pos: rng.in_sphere() * (BOUNDS * 0.85),
                    speed: rng.range(0.6, 1.8),
                    age: rng.range(0.0, 4.0),
                    life: rng.range(3.0, 7.0),
                };
                let mesh = Mesh::new(
                    geometry.clone(),
                    StandardMaterial::new(Color::BLACK).into(),
                );
                let mut obj = Object3D::mesh(mesh);
                obj.position = p.pos;
                ids.push(s.scene.add(obj));
                particles.push(p);
            }

            let mut frames = 0u64;
            move |f: &mut Frame| {
                frames += 1;
                if let Some(limit) = frame_limit {
                    if frames >= limit {
                        f.exit = true;
                    }
                }
                // The pointer tilts the field, so the sketch answers the mouse.
                let tilt = f.pointer_ndc.x * 0.8;

                for (p, id) in particles.iter_mut().zip(&ids) {
                    // Curl noise: divergence free, so nothing collects in a
                    // sink. Two independent noise channels used as a velocity
                    // would, and the whole cloud would end up in a few knots.
                    let (vx, vy) = SimplexNoise::curl2(
                        p.pos.x * 0.18 + tilt,
                        p.pos.z * 0.18 + f.elapsed * 0.08,
                        0.01,
                    );
                    let lift = SimplexNoise::fbm3(
                        p.pos.x * 0.12,
                        p.pos.y * 0.12,
                        f.elapsed * 0.15,
                        3,
                        2.0,
                        0.5,
                    );
                    let step = p.speed * f.delta;
                    p.pos.x += vx * step;
                    p.pos.y += lift * step * 0.8;
                    p.pos.z += vy * step;
                    p.age += f.delta;

                    let escaped = p.pos.x.abs() > BOUNDS
                        || p.pos.y.abs() > BOUNDS
                        || p.pos.z.abs() > BOUNDS;
                    if escaped || p.age > p.life {
                        p.pos = Vector3::new(
                            (p.pos.x * 7.13).sin() * BOUNDS * 0.6,
                            -BOUNDS * 0.6,
                            (p.pos.z * 3.77).cos() * BOUNDS * 0.6,
                        );
                        p.age = 0.0;
                    }

                    // Speed to colour through viridis, and a short fade in and
                    // out so a recycled particle does not pop.
                    let speed = (vx * vx + vy * vy).sqrt();
                    let t = map_range_clamped(speed, 0.0, 1.6, 0.0, 1.0);
                    let fade_in = Easing::CubicOut.apply((p.age / 0.6).clamp(0.0, 1.0));
                    let fade_out =
                        Easing::CubicIn.apply(((p.life - p.age) / 0.8).clamp(0.0, 1.0));
                    let colour = ramp.at(t).scaled(fade_in * fade_out);

                    if let Some(obj) = f.scene.get_mut(*id) {
                        obj.position = p.pos;
                        if let ObjectKind::Mesh(m) = &mut obj.kind {
                            m.material = std::sync::Arc::new(
                                StandardMaterial::new(colour)
                                    .with_emissive(colour, 0.8)
                                    .into(),
                            );
                        }
                    }
                }

                // The 2D layer. Cleared for us before this runs, so everything
                // below is this frame's picture and nothing persists.
                let fps = 1.0 / f.delta.max(1e-6);
                let (pointer, down) = (f.pointer, f.pointer_down);
                // Everything 2D is in physical pixels, so anything meant to be
                // *read* is scaled: a 15-pixel label is half as tall on a 2x
                // display as the window size asked for.
                let (k, right) = (f.scale_factor, f.width as f32);
                let o = &mut f.overlay;
                o.text_size(15.0 * k);
                o.no_stroke();
                o.fill_alpha(Color::BLACK, 0.45);
                o.rounded_rect(14.0 * k, 14.0 * k, 208.0 * k, 92.0 * k, 8.0 * k);

                // An outline on the type, because white text over a dark but
                // moving field is legible only some of the time.
                o.fill(Color::WHITE);
                o.stroke_alpha(Color::BLACK, 0.85);
                o.stroke_weight(2.0 * k);
                o.text(28.0 * k, 26.0 * k, &format!("{fps:.0} fps"));
                o.text(28.0 * k, 48.0 * k, &format!("{COUNT} particles"));

                // Right-aligned, so the digits grow leftwards from a fixed edge
                // instead of shoving the line about as the count climbs.
                o.text_right(right - 20.0 * k, 26.0 * k, &format!("frame {}", f.frame));
                o.no_stroke();

                // The colour ramp as a legend: one column per pixel, which is
                // what a gradient is when the canvas is in pixels anyway.
                let (lx, ly) = (28.0 * k, 80.0 * k);
                let lw = (180.0 * k).round();
                for i in 0..lw as u32 {
                    o.fill(ramp.at(i as f32 / (lw - 1.0)));
                    o.rect(lx + i as f32, ly, 1.0, 12.0 * k);
                }
                o.no_fill();
                o.stroke_alpha(Color::WHITE, 0.6);
                o.stroke_weight(1.0 * k);
                o.rect(lx, ly, lw, 12.0 * k);

                // And a ring on the cursor, so the pointer tilt above is
                // visibly coming from somewhere.
                o.stroke_alpha(Color::WHITE, if down { 0.95 } else { 0.5 });
                o.stroke_weight(if down { 3.0 * k } else { 1.5 * k });
                o.circle(pointer.x, pointer.y, if down { 16.0 * k } else { 11.0 * k });
            }
        },
    );
}
