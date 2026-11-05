//! The circuit: its centre line, how it climbs and leans, and the scenery on it.

use threers::curves::{CatmullRomCurve3, Curve3};
use threers_ecs::prelude::*;

pub const ROAD_HALF_WIDTH: f32 = 8.0;
/// From the road's edge to the barrier: room to run wide on the grass.
pub const VERGE: f32 = 5.0;
const SAMPLE_SPACING: f32 = 2.5;
/// The steepest a corner leans, as a rise across the road per unit of width.
const MAX_BANK: f32 = 0.12;

/// The circuit's shape, as (x, z).
const CIRCUIT: [[f32; 2]; 19] = [
    [-130.0, 110.0],
    [-40.0, 110.0],
    [50.0, 110.0],
    [130.0, 100.0],
    [180.0, 50.0],
    [170.0, -20.0],
    [110.0, -40.0],
    [60.0, -10.0],
    [10.0, -10.0],
    [-20.0, -50.0],
    [30.0, -100.0],
    [110.0, -120.0],
    [150.0, -170.0],
    [60.0, -210.0],
    [-70.0, -200.0],
    [-150.0, -150.0],
    [-110.0, -80.0],
    [-170.0, -30.0],
    [-200.0, 50.0],
];

/// How high the road is, a fraction `u` of the way round: one long hill and three
/// short crests, at most a few percent steep.
fn elevation(u: f32) -> f32 {
    use std::f32::consts::TAU;
    11.0 + 8.0 * (TAU * u).sin() + 3.0 * (3.0 * TAU * u + 1.0).sin()
}

/// The road's centre line, evenly spaced, and what a driver wants to know of it.
pub struct Track {
    pub points: Vec<Vector3>,
    pub tangents: Vec<Vector3>,
    /// How sharply it turns at each point, per unit of road.
    pub curvature: Vec<f32>,
    /// How far each point leans: rise per unit across, positive when the left
    /// side is higher.
    pub bank: Vec<f32>,
    pub length: f32,
}

impl Default for Track {
    fn default() -> Self {
        Self::new()
    }
}

impl Track {
    pub fn new() -> Self {
        let mut curve = CatmullRomCurve3::new(
            CIRCUIT
                .iter()
                .map(|&[x, z]| Vector3::new(x, 0.0, z))
                .collect(),
        );
        curve.closed = true;
        let dense = curve.get_points(4000);
        let mut along = vec![0.0];
        for pair in dense.windows(2) {
            along.push(along.last().unwrap() + pair[0].distance_to(pair[1]));
        }
        let length = *along.last().unwrap();
        let count = (length / SAMPLE_SPACING) as usize;
        let spacing = length / count as f32;
        let mut points = Vec::with_capacity(count);
        let mut at = 0;
        for i in 0..count {
            let want = i as f32 * spacing;
            while along[at + 1] < want {
                at += 1;
            }
            let t = (want - along[at]) / (along[at + 1] - along[at]);
            let mut p = dense[at].lerp(dense[at + 1], t);
            p.y = elevation(i as f32 / count as f32);
            points.push(p);
        }
        let wrap = |i: isize| i.rem_euclid(count as isize) as usize;
        let tangents: Vec<Vector3> = (0..count as isize)
            .map(|i| (points[wrap(i + 1)] - points[wrap(i - 1)]).normalize())
            .collect();
        let flat = |v: Vector3| Vector3::new(v.x, 0.0, v.z).normalize();
        let mut curvature = Vec::with_capacity(count);
        let mut turn = Vec::with_capacity(count);
        for i in 0..count as isize {
            let (a, b) = (flat(tangents[wrap(i - 2)]), flat(tangents[wrap(i + 2)]));
            curvature.push(a.dot(b).clamp(-1.0, 1.0).acos() / (4.0 * spacing));
            // Positive turning left.
            turn.push(a.cross(b).y / (4.0 * spacing));
        }
        // A corner leans into itself, its outside raised; smoothed so the road
        // twists gently into and out of it.
        let raw: Vec<f32> = turn
            .iter()
            .map(|t| (-t * 6.0).clamp(-MAX_BANK, MAX_BANK))
            .collect();
        let bank = (0..count as isize)
            .map(|i| (-8..=8).map(|k| raw[wrap(i + k)]).sum::<f32>() / 17.0)
            .collect();
        Self {
            points,
            tangents,
            curvature,
            bank,
            length,
        }
    }

    pub fn spacing(&self) -> f32 {
        self.length / self.points.len() as f32
    }

    pub fn wrap(&self, index: isize) -> usize {
        index.rem_euclid(self.points.len() as isize) as usize
    }

    /// The nearest point to `at`, looking around `hint`: the road can pass close
    /// to itself, and the whole list would then find the wrong piece of it.
    pub fn nearest(&self, at: Vector3, hint: usize) -> usize {
        (-24..=24)
            .map(|offset| self.wrap(hint as isize + offset))
            .min_by(|&a, &b| {
                at.distance_to(self.points[a])
                    .total_cmp(&at.distance_to(self.points[b]))
            })
            .unwrap_or(hint)
    }

    /// The nearest point anywhere, for a position with no history.
    pub fn nearest_anywhere(&self, at: Vector3) -> usize {
        let flat = |p: Vector3| Vector3::new(p.x - at.x, 0.0, p.z - at.z).length();
        let coarse = (0..self.points.len())
            .step_by(4)
            .min_by(|&a, &b| flat(self.points[a]).total_cmp(&flat(self.points[b])));
        let hint = coarse.unwrap_or(0);
        (-4..=4)
            .map(|k| self.wrap(hint as isize + k))
            .min_by(|&a, &b| flat(self.points[a]).total_cmp(&flat(self.points[b])))
            .unwrap_or(hint)
    }

    /// Left of the road's direction, level.
    pub fn left(&self, index: usize) -> Vector3 {
        let t = self.tangents[index];
        Vector3::new(t.z, 0.0, -t.x).normalize()
    }

    /// The road surface `offset` to the left of the centre, leaning with the bank.
    pub fn surface(&self, index: usize, offset: f32) -> Vector3 {
        self.points[index]
            + self.left(index) * offset
            + Vector3::new(0.0, offset * self.bank[index], 0.0)
    }

    /// How far `at` is from the centre line, positive to the left.
    pub fn offset(&self, at: Vector3, index: usize) -> f32 {
        (at - self.points[index]).dot(self.left(index))
    }

    /// The rotation of something facing along the road, level.
    pub fn facing(&self, index: usize) -> Quaternion {
        let t = self.tangents[index];
        Quaternion::from_axis_angle(Vector3::UP, (-t.x).atan2(-t.z))
    }

    /// How high the ground is at `(x, z)`: the road's own surface out to the
    /// barriers, easing into rolling country beyond.
    pub fn ground(&self, x: f32, z: f32) -> f32 {
        let at = Vector3::new(x, 0.0, z);
        let i = self.nearest_anywhere(at);
        let flat_gap = |p: Vector3| Vector3::new(p.x - x, 0.0, p.z - z).length();
        let gap = flat_gap(self.points[i]);
        let reach = ROAD_HALF_WIDTH + VERGE + 2.0;
        let offset = self
            .offset(Vector3::new(x, self.points[i].y, z), i)
            .clamp(-reach, reach);
        let near = self.surface(i, offset).y - 0.15;
        let country =
            6.0 + 4.0 * (x * 0.013).sin() * (z * 0.011).cos() + 2.0 * (x * 0.031 + z * 0.023).sin();
        let t = ((gap - reach) / 45.0).clamp(0.0, 1.0);
        let blend = t * t * (3.0 - 2.0 * t);
        let height = near + (country - near) * blend;
        // Under the road, well clear of its surface, whatever the grid between.
        if gap < ROAD_HALF_WIDTH {
            height - 0.6
        } else {
            height
        }
    }
}

/// The road's edge at `offset` from the centre, at each of `samples`.
fn edge(track: &Track, offset: f32, lift: f32, samples: &[usize]) -> Vec<Vector3> {
    samples
        .iter()
        .map(|&i| track.surface(i, offset) + Vector3::new(0.0, lift, 0.0))
        .collect()
}

/// Kerb blocks, every other pair of samples, on one mesh.
fn kerbs(mesh: &mut MeshBuilder, track: &Track, from: f32, to: f32, phase: usize) {
    for first in (phase * 2..track.points.len()).step_by(4) {
        let samples = [
            first,
            track.wrap(first as isize + 1),
            track.wrap(first as isize + 2),
        ];
        mesh.ribbon(
            &edge(track, from, 0.03, &samples),
            &edge(track, to, 0.03, &samples),
            Vector3::UP,
            false,
        );
    }
}

/// Everything that does not move: ground, road, kerbs, barriers and trees.
pub fn build_circuit(w: &mut World, track: &Track) {
    let flat = |hex| StandardMaterial::new(Color::from_hex(hex));

    // Country, centred on the circuit, solid.
    let (cx, cz, size, cells) = (-10.0, -50.0, 800.0, 128);
    let mut ground = MeshBuilder::new();
    ground.grid(size, size, cells, cells, |x, z| {
        track.ground(x + cx, z + cz)
    });
    w.mesh(ground, flat(0x4c8a3f))
        .named("ground")
        .at(Vector3::new(cx, 0.0, cz))
        .shadows(false, true)
        .solid();

    let half = ROAD_HALF_WIDTH;
    let all: Vec<usize> = (0..track.points.len()).collect();
    let mut road = MeshBuilder::new();
    road.ribbon(
        &edge(track, half, 0.0, &all),
        &edge(track, -half, 0.0, &all),
        Vector3::UP,
        true,
    );
    w.mesh(road, flat(0x3a3a40))
        .named("road")
        .shadows(false, true)
        .solid();

    let (mut red, mut white) = (MeshBuilder::new(), MeshBuilder::new());
    for (from, to) in [(half + 1.0, half), (-half - 1.0, -half)] {
        kerbs(&mut red, track, from, to, 0);
        kerbs(&mut white, track, from, to, 1);
    }
    w.mesh(red, flat(0xd8342c)).shadows(false, true);
    w.mesh(white, flat(0xf2f2f2)).shadows(false, true);

    let count = track.points.len();
    let line = [count - 1, 0];
    let mut finish = MeshBuilder::new();
    finish.ribbon(
        &edge(track, half, 0.04, &line),
        &edge(track, -half, 0.04, &line),
        Vector3::UP,
        false,
    );
    w.mesh(finish, flat(0xffffff)).shadows(false, true);

    // Barriers: one mesh to draw and one body to hit, however many boxes.
    let step = 3;
    let mut wall = MeshBuilder::new();
    let mut parts = Vec::new();
    for side in [1.0, -1.0] {
        let reach = (ROAD_HALF_WIDTH + VERGE) * side;
        for i in (0..count).step_by(step) {
            let j = track.wrap((i + step) as isize);
            let (a, b) = (track.surface(i, reach), track.surface(j, reach));
            let along = b - a;
            let length = along.length() + 0.6;
            let level = Vector3::new(along.x, 0.0, along.z);
            let yaw = Quaternion::from_axis_angle(Vector3::UP, (-level.x).atan2(-level.z));
            let pitch = Quaternion::from_axis_angle(
                Vector3::new(1.0, 0.0, 0.0),
                (along.y / level.length().max(1e-3)).atan(),
            );
            let turned = yaw.multiply(pitch);
            let at = (a + b) * 0.5 + Vector3::new(0.0, 0.5, 0.0);
            wall.append(
                &BoxGeometry::new(0.6, 1.0, length),
                &Matrix4::compose(at, turned, Vector3::ONE),
            );
            parts.push((
                Isometry::new(at, turned),
                Shape::cuboid(0.3, 0.5, length / 2.0),
            ));
        }
    }
    w.mesh(wall, flat(0xc9ccd1))
        .named("barriers")
        .shadows(true, true)
        .body(
            RigidBody::fixed()
                .shape(Shape::compound(parts))
                .friction(0.2),
        );

    // Trees, off the road, wherever the same hash puts them every time.
    let mut seed = 12345u32;
    let mut next = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / (1u32 << 24) as f32
    };
    let mut trees = MeshBuilder::new();
    let mut planted = 0;
    while planted < 110 {
        let (x, z) = (cx + next() * 700.0 - 350.0, cz + next() * 700.0 - 350.0);
        let at = Vector3::new(x, 0.0, z);
        let clear = track.points.iter().all(|p| {
            Vector3::new(p.x - at.x, 0.0, p.z - at.z).length() > ROAD_HALF_WIDTH + VERGE + 6.0
        });
        if clear {
            let tall = 7.0 + next() * 7.0;
            let cone = ConeGeometry::new(2.6, tall, 10, 1, false, 0.0, std::f32::consts::TAU);
            let base = track.ground(x, z);
            trees.append(
                &cone,
                &Matrix4::translation(Vector3::new(x, base + tall / 2.0 - 0.3, z)),
            );
            planted += 1;
        }
    }
    w.mesh(trees, flat(0x2f6b3a))
        .named("trees")
        .shadows(true, false);
}
