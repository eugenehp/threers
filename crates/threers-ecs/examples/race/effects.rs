//! Skid marks: dark strips where a tyre slid, the oldest fading out of the list.

use threers_ecs::prelude::*;

/// Strips kept, across all cars and wheels.
const MAX_MARKS: usize = 1600;
const HALF_WIDTH: f32 = 0.13;

/// The marks so far, and where each wheel last touched.
pub struct SkidMarks {
    mesh: Entity,
    marks: std::collections::VecDeque<[Vector3; 4]>,
    last: std::collections::HashMap<(Entity, usize), Vector3>,
    changed: bool,
}

impl SkidMarks {
    pub fn spawn(w: &mut World) -> Self {
        let mut material = StandardMaterial::new(Color::from_hex(0x111111)).with_roughness(1.0);
        material.opacity = 0.55;
        material.transparent = Some(true);
        let mesh = w
            .mesh(placeholder(), material)
            .named("skid marks")
            .shadows(false, false)
            .id();
        Self {
            mesh,
            marks: Default::default(),
            last: Default::default(),
            changed: false,
        }
    }

    pub fn len(&self) -> usize {
        self.marks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.marks.is_empty()
    }
}

/// One tiny triangle far below the ground, for when there are no marks: a mesh
/// with nothing in it is a case a renderer need not handle.
fn placeholder() -> BufferGeometry {
    let mut mesh = MeshBuilder::new();
    let a = mesh.vertex(Vector3::new(0.0, -500.0, 0.0));
    let b = mesh.vertex(Vector3::new(0.01, -500.0, 0.0));
    let c = mesh.vertex(Vector3::new(0.0, -500.0, 0.01));
    mesh.facing(Vector3::UP).triangle(a, b, c);
    mesh.build()
}

/// A strip from each sliding wheel's last contact to this one.
pub fn lay_marks(w: &mut World) {
    let mut new = Vec::new();
    let mut lifted = Vec::new();
    for (id, car) in w.iter::<(Entity, &Car)>() {
        let sliding = car.sliding.abs() > 2.5 || car.slip > 0.35;
        for (wheel, contact) in car.contacts().enumerate() {
            match contact.filter(|_| sliding) {
                Some(point) => new.push(((id, wheel), point)),
                None => lifted.push((id, wheel)),
            }
        }
    }
    let marks = w.resource_mut::<SkidMarks>();
    for key in lifted {
        marks.last.remove(&key);
    }
    for (key, point) in new {
        let point = point + Vector3::new(0.0, 0.03, 0.0);
        if let Some(from) = marks.last.insert(key, point) {
            let along = point - from;
            if along.length() < 0.05 || along.length() > 3.0 {
                continue;
            }
            let side = along.cross(Vector3::UP).normalize() * HALF_WIDTH;
            marks
                .marks
                .push_back([from - side, from + side, point + side, point - side]);
            if marks.marks.len() > MAX_MARKS {
                marks.marks.pop_front();
            }
            marks.changed = true;
        }
    }
}

/// Rebuild the marks' mesh when they changed.
pub fn draw_marks(w: &mut World) {
    let marks = w.resource_mut::<SkidMarks>();
    if !marks.changed {
        return;
    }
    marks.changed = false;
    let mut mesh = MeshBuilder::new();
    mesh.normal(Vector3::UP).facing(Vector3::UP);
    for quad in &marks.marks {
        let ids = quad.map(|p| mesh.vertex(p));
        mesh.quad(ids[0], ids[1], ids[2], ids[3]);
    }
    let geometry = if marks.marks.is_empty() {
        placeholder()
    } else {
        mesh.build()
    };
    let id = marks.mesh;
    w.edit_mesh(id, |g| *g = geometry);
}

/// Clear them, for a new race.
pub fn clear_marks(w: &mut World) {
    let marks = w.resource_mut::<SkidMarks>();
    marks.marks.clear();
    marks.last.clear();
    marks.changed = true;
}
