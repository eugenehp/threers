//! A Vision Pro app with threers: a room of shapes and candles around you.
//! Look at a shape and pinch to set it spinning. Everything is here; the
//! Swift in `app/` only hands the immersive space to Rust.
//!
//! `app/build.sh` runs it in the simulator; `cargo test` runs the tests on
//! your Mac.

use threers::extras::candle::{Candle, CandleHandle, FlameStyle};
use threers::xr::{App, Frame};
use threers::{BoxGeometry, Color, DirectionalLight, Euler, HemisphereLight, IcosahedronGeometry, Mesh, Object3D, ObjectId, PlaneGeometry, Scene, SphereGeometry, StandardMaterial, TorusKnotGeometry, Vector3};

/// How big a shape is to a pinch's ray.
const REACH: f32 = 0.18;

pub struct Room {
    /// Each shape, and how fast it spins.
    shapes: Vec<(ObjectId, f32)>,
    candles: Vec<CandleHandle>,
}

impl App for Room {
    fn setup(scene: &mut Scene) -> Self {
        scene.background = Color::from_hex(0x1c2230);
        scene.add_light(HemisphereLight::new(Color::from_hex(0xd8e2ff), Color::from_hex(0x2a2520), 0.8));
        let mut sun = Object3D::light(DirectionalLight::new(Color::from_hex(0xffffff), 1.2).with_direction(Vector3::new(-0.4, -1.0, -0.3)));
        sun.position = Vector3::new(3.0, 6.0, 2.0);
        scene.add(sun);
        let mut floor = Object3D::mesh(Mesh::new(PlaneGeometry::new(20.0, 20.0), StandardMaterial::new(Color::from_hex(0x2c3342)).with_roughness(0.95).into()));
        floor.quaternion = Euler::new(-std::f32::consts::FRAC_PI_2, 0.0, 0.0).to_quaternion();
        scene.add(floor);
        // Shapes in an arc in front of you, at about chest height.
        let shapes = [0xe8664a, 0xf2b33d, 0x5cc98a, 0x4fa3e8, 0xa77de8]
            .into_iter()
            .enumerate()
            .map(|(i, hex)| {
                let geometry = match i % 4 {
                    0 => TorusKnotGeometry::new(0.1, 0.035, 96, 12, 2, 3),
                    1 => SphereGeometry::new(0.12, 32, 16),
                    2 => BoxGeometry::new(0.2, 0.2, 0.2),
                    _ => IcosahedronGeometry::new(0.14, 0),
                };
                let mut o = Object3D::mesh(Mesh::new(geometry, StandardMaterial::new(Color::from_hex(hex)).with_roughness(0.35).into()));
                let a = (i as f32 - 2.0) * 0.45;
                o.position = Vector3::new(a.sin() * 1.3, 1.25, -a.cos() * 1.3);
                (scene.add(o), 0.3)
            })
            .collect();
        // Candles on a low table to your left.
        let mut table = Object3D::mesh(Mesh::new(BoxGeometry::new(0.5, 0.7, 0.5), StandardMaterial::new(Color::from_hex(0x5a3a22)).into()));
        table.position = Vector3::new(-1.1, 0.35, -0.7);
        scene.add(table);
        let candles = [(-1.18, -0.75, 0.16), (-1.02, -0.66, 0.11)]
            .into_iter()
            .enumerate()
            .map(|(i, (x, z, h))| {
                let mut c = Candle::new(FlameStyle::Realistic).with_seed(i as f32).with_size(h, 0.028);
                c.light_intensity = 0.3;
                c.add_to(scene, Vector3::new(x, 0.7, z))
            })
            .collect();
        Room { shapes, candles }
    }

    fn update(&mut self, scene: &mut Scene, frame: &Frame) {
        let t = frame.time as f32;
        // A pinch on a shape sets it spinning; it slows back down.
        for (origin, dir) in frame.selects() {
            let hit = self.shapes.iter_mut().filter_map(|(id, speed)| {
                let c = scene.get(*id)?.position - origin;
                let along = c.dot(dir);
                (along > 0.0 && (c - dir * along).length() < REACH).then_some((along, speed))
            });
            if let Some((_, speed)) = hit.min_by(|a, b| a.0.total_cmp(&b.0)) {
                *speed = 6.0;
            }
        }
        for (id, speed) in &mut self.shapes {
            *speed = 0.3 + (*speed - 0.3) * (-(frame.dt as f32) * 0.8).exp();
            if let Some(o) = scene.get_mut(*id) {
                o.quaternion = o.quaternion.multiply(Euler::new(0.0, *speed * frame.dt as f32, 0.0).to_quaternion());
            }
        }
        for c in &self.candles {
            c.update(scene, t);
        }
    }
}

threers::visionos_app!(Room);

#[cfg(test)]
mod tests {
    use super::*;
    use threers::xr::{TargetRayMode, XrEvent, XrEventKind, XrInput};

    #[test]
    fn a_pinch_on_a_shape_sets_it_spinning() {
        let mut scene = Scene::new();
        let mut room = Room::setup(&mut scene);
        // A pinch from the head straight at the middle shape (ahead, -Z).
        let head = threers::Matrix4::translation(Vector3::new(0.0, 1.25, 0.0)).elements;
        let frame = Frame {
            dt: 1.0 / 90.0,
            inputs: vec![XrInput { slot: 0, target_ray_mode: TargetRayMode::TransientPointer, ray: Some(head), ..Default::default() }],
            events: vec![XrEvent { kind: XrEventKind::Select, slot: 0 }],
            ..Default::default()
        };
        room.update(&mut scene, &frame);
        let fast: Vec<_> = room.shapes.iter().filter(|s| s.1 > 1.0).collect();
        assert_eq!(fast.len(), 1, "the one it points at");
    }
}
