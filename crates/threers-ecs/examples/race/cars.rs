//! What a car looks and sounds like.

use super::rules::{Player, Racer};
use threers_ecs::prelude::*;

/// Sound effects, made rather than loaded.
pub struct Sounds {
    pub beep: threers::audio::Audio,
    pub go: threers::audio::Audio,
    pub hit: threers::audio::Audio,
}

impl Default for Sounds {
    fn default() -> Self {
        Self {
            beep: tone(440.0, 0.15, 0.3),
            go: tone(880.0, 0.4, 0.3),
            hit: tone(110.0, 0.1, 0.3),
        }
    }
}

pub fn play(w: &mut World, pick: fn(&Sounds) -> &threers::audio::Audio) {
    let sound = pick(w.resource::<Sounds>()).clone();
    if let Some(out) = w.get_resource_mut::<AudioOut>() {
        out.play(&sound);
    }
}

/// A six-sided solid from its corners: the bottom four, then the top four, each
/// going round the same way. Each face is turned outwards whatever the order.
fn block(mesh: &mut MeshBuilder, corners: [Vector3; 8]) {
    let centre = corners.iter().fold(Vector3::ZERO, |sum, &c| sum + c) * 0.125;
    let faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ];
    for face in faces {
        let points = face.map(|i| corners[i]);
        let middle = points.iter().fold(Vector3::ZERO, |sum, &p| sum + p) * 0.25;
        mesh.facing(middle - centre);
        let ids = points.map(|p| mesh.vertex(p));
        mesh.quad(ids[0], ids[1], ids[2], ids[3]);
    }
    mesh.as_wound();
}

/// A box between two corners.
fn slab(mesh: &mut MeshBuilder, low: Vector3, high: Vector3) {
    let c = |x: f32, y: f32, z: f32| Vector3::new(x, y, z);
    block(
        mesh,
        [
            c(low.x, low.y, low.z),
            c(high.x, low.y, low.z),
            c(high.x, low.y, high.z),
            c(low.x, low.y, high.z),
            c(low.x, high.y, low.z),
            c(high.x, high.y, low.z),
            c(high.x, high.y, high.z),
            c(low.x, high.y, high.z),
        ],
    );
}

/// The body's three parts as meshes in the car's frame.
fn body(half: Vector3) -> (MeshBuilder, MeshBuilder, MeshBuilder, MeshBuilder) {
    let (w, h, l) = (half.x, half.y, half.z);
    let v = Vector3::new;
    let (mut paint, mut glass, mut lamps, mut tail) = (
        MeshBuilder::new(),
        MeshBuilder::new(),
        MeshBuilder::new(),
        MeshBuilder::new(),
    );

    // The hull: a wedge, low at the nose.
    block(
        &mut paint,
        [
            v(-w, -h, -l),
            v(w, -h, -l),
            v(w, -h, l),
            v(-w, -h, l),
            v(-w, -0.1 * h, -l),
            v(w, -0.1 * h, -l),
            v(w, 0.55 * h, l),
            v(-w, 0.55 * h, l),
        ],
    );
    // The cabin, its windscreen raked back.
    let (base, roof) = (0.3 * h, 1.75 * h);
    block(
        &mut paint,
        [
            v(-0.86 * w, base, -0.35 * l),
            v(0.86 * w, base, -0.35 * l),
            v(0.86 * w, base + 0.12 * h, 0.55 * l),
            v(-0.86 * w, base + 0.12 * h, 0.55 * l),
            v(-0.7 * w, roof, -0.05 * l),
            v(0.7 * w, roof, -0.05 * l),
            v(0.7 * w, roof, 0.45 * l),
            v(-0.7 * w, roof, 0.45 * l),
        ],
    );
    // Glass a whisker proud of the cabin: windscreen, rear window and sides.
    let lift = |p: Vector3, by: Vector3| p + by * 0.012;
    let pane = |mesh: &mut MeshBuilder, quad: [Vector3; 4], out: Vector3| {
        mesh.facing(out);
        let ids = quad.map(|p| mesh.vertex(lift(p, out)));
        mesh.quad(ids[0], ids[1], ids[2], ids[3]);
    };
    let inset = |a: Vector3, b: Vector3, t: f32| a.lerp(b, t);
    let (fl, fr, rl, rr) = (
        v(-0.86 * w, base, -0.35 * l),
        v(0.86 * w, base, -0.35 * l),
        v(-0.7 * w, roof, -0.05 * l),
        v(0.7 * w, roof, -0.05 * l),
    );
    let screen_out = (fr - fl).cross(rl - fl).normalize() * -1.0;
    pane(
        &mut glass,
        [
            inset(fl, rr, 0.08),
            inset(fr, rl, 0.08),
            inset(rr, fl, 0.08),
            inset(rl, fr, 0.08),
        ],
        screen_out,
    );
    let (bl, br, tl, tr) = (
        v(-0.86 * w, base + 0.12 * h, 0.55 * l),
        v(0.86 * w, base + 0.12 * h, 0.55 * l),
        v(-0.7 * w, roof, 0.45 * l),
        v(0.7 * w, roof, 0.45 * l),
    );
    let rear_out = (tl - bl).cross(br - bl).normalize() * -1.0;
    pane(
        &mut glass,
        [
            inset(bl, tr, 0.1),
            inset(br, tl, 0.1),
            inset(tr, bl, 0.1),
            inset(tl, br, 0.1),
        ],
        rear_out,
    );
    for side in [-1.0f32, 1.0] {
        let x = |k: f32| side * k * w;
        let quad = [
            v(x(0.86), base + 0.05, -0.3 * l),
            v(x(0.86), base + 0.12 * h + 0.05, 0.5 * l),
            v(x(0.7), roof - 0.05, 0.42 * l),
            v(x(0.7), roof - 0.05, -0.02 * l),
        ];
        pane(&mut glass, quad, v(side, 0.15, 0.0));
    }
    // Lamps, front and back, and a wing on the tail.
    for side in [-1.0f32, 1.0] {
        let x0 = side * 0.55 * w;
        slab(
            &mut lamps,
            v(x0 - 0.2, -0.45 * h, -l - 0.02),
            v(x0 + 0.2, -0.15 * h, -l + 0.05),
        );
        slab(
            &mut tail,
            v(x0 - 0.22, 0.15 * h, l - 0.05),
            v(x0 + 0.22, 0.4 * h, l + 0.02),
        );
        slab(
            &mut paint,
            v(side * 0.6 * w - 0.05, 0.55 * h, 0.82 * l),
            v(side * 0.6 * w + 0.05, 1.2 * h, 0.9 * l),
        );
    }
    slab(
        &mut paint,
        v(-0.95 * w, 1.2 * h, 0.76 * l),
        v(0.95 * w, 1.32 * h, 0.98 * l),
    );
    (paint, glass, lamps, tail)
}

/// A low, rough note that loops, for `engines` to bend.
fn engine() -> Speaker {
    let mut speaker = Speaker::looping(waveform(70.0, 0.5, 0.5, &[0.7, 0.5, 0.4, 0.3, 0.2]));
    speaker.bus = Bus::Effects;
    speaker.source.ref_distance = 6.0;
    speaker
}

/// A thin high whine, silent until a tyre lets go.
fn squeal() -> Speaker {
    let mut speaker = Speaker::looping(waveform(820.0, 0.4, 0.35, &[0.35, 0.0, 0.15]));
    speaker.source.audio.volume = 0.0;
    speaker.source.ref_distance = 6.0;
    speaker
}

pub fn spawn_car(w: &mut World, name: &'static str, colour: u32, spec: CarSpec) -> Entity {
    let half = spec.half_size;
    let id = w
        .object(Object3D::group())
        .at(Vector3::new(0.0, 1.0, 0.0))
        .named(name)
        .car(spec)
        .with((Racer::new(name), Speakers(vec![engine(), squeal()])))
        .id();
    let (paint, glass, lamps, tail) = body(half);
    let parts = [
        (
            paint.build_flat(),
            StandardMaterial::new(Color::from_hex(colour)).with_roughness(0.35),
        ),
        (
            glass.build_flat(),
            StandardMaterial::new(Color::from_hex(0x1a2430)).with_roughness(0.1),
        ),
        (
            lamps.build_flat(),
            StandardMaterial::new(Color::from_hex(0xfff6d8))
                .with_emissive(Color::from_hex(0xfff2c0), 1.0),
        ),
        (
            tail.build_flat(),
            StandardMaterial::new(Color::from_hex(0x8a0d0d))
                .with_emissive(Color::from_hex(0xff2020), 0.6),
        ),
    ];
    for (geometry, material) in parts {
        let mut part = Object3D::mesh(Mesh::new(geometry, material.into()));
        part.cast_shadow = true;
        w.spawn_under(id, part);
    }
    id
}

/// The engine, first of a car's speakers, and the tyres, second.
pub const ENGINE: usize = 0;
pub const TYRES: usize = 1;

/// Every engine rises with its speed and is loudest on the throttle; tyres
/// squeal as they slide.
pub fn engines(w: &mut World) {
    w.each::<(&Car, &mut Speakers)>(|(car, speakers)| {
        let engine = &mut speakers.0[ENGINE];
        engine.pitch = 0.7 + (car.speed.abs() / 12.0) % 1.0 * 0.9 + car.speed.abs() / 70.0;
        engine.source.audio.volume = 0.25 + 0.35 * car.throttle.abs();
        let sideways = ((car.sliding.abs() - 2.0) / 6.0).clamp(0.0, 1.0);
        let level = sideways.max(car.slip) * (car.wheels_on_ground > 0) as u8 as f32;
        let tyres = &mut speakers.0[TYRES];
        tyres.source.audio.volume = 0.45 * level;
        tyres.pitch = 0.9 + 0.2 * level;
    });
}

pub fn knocks(w: &mut World) {
    let hits = w.events::<Collision>();
    let player = w.entities::<With<Player>>();
    let hit = hits.iter().any(|hit| {
        hit.started
            && !hit.is_sensor
            && player
                .iter()
                .any(|&id| hit.a == Some(id) || hit.b == Some(id))
    });
    if hit {
        w.resource_mut::<ChaseCamera>().shake(0.15);
        play(w, |s| &s.hit);
    }
}
