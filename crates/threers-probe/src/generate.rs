//! Scenes for the probe proof of concept.
//!
//! The **family** is what the network trains on: enclosed rooms, Cornell-like
//! boxes (jittered, never the canonical layout), open ground, glass and metal.
//! The **hostile** set is never trained on — canonical Cornell (diffuse and
//! glass), a fixed open floor, and the furnace test.

use std::f32::consts::FRAC_PI_2;

use threers::{
    BoxGeometry, Color, Material, Mesh, Object3D, PerspectiveCamera, PhysicalMaterial,
    PlaneGeometry, Scene, SphereGeometry, StandardMaterial, Vector3,
};

pub struct Built {
    pub scene: Scene,
    pub camera: PerspectiveCamera,
    pub kind: String,
}

/// One scene from the training family, varied by index.
pub fn family(index: u32) -> Built {
    let mut rng = Rng::new(index as u64 + 1);
    let roll = rng.next();
    if roll < 0.15 {
        open_ground(&mut rng, false)
    } else if roll < 0.25 {
        furnace_like(&mut rng)
    } else if roll < 0.70 {
        cornell_like(&mut rng)
    } else {
        enclosed(&mut rng, false)
    }
}

/// Scenes that are not drawn from [`family`], for scoring only.
pub fn hostile(index: u32) -> Built {
    match index % 4 {
        0 => cornell(false),
        1 => cornell(true),
        2 => {
            let mut rng = Rng::new(HOSTILE);
            open_ground(&mut rng, true)
        }
        _ => furnace(),
    }
}

pub fn hostile_count() -> u32 {
    4
}

pub fn hostile_name(index: u32) -> &'static str {
    match index % 4 {
        0 => "cornell",
        1 => "cornell-glass",
        2 => "open-holdout",
        _ => "furnace",
    }
}

/// Whether this family scene should contribute NRC training tiles.
///
/// Open exteriors teach the MLP to invent bounce on ground planes that the
/// hold-out set punishes; enclosed and Cornell-like boxes are the target domain.
pub fn nrc_trainable_kind(kind: &str) -> bool {
    (kind.starts_with("enclosed") || kind.starts_with("cornell-like"))
        && !kind.starts_with("furnace")
}

fn enclosed(rng: &mut Rng, _fixed: bool) -> Built {
    let mut scene = Scene::new();
    scene.background = Color::BLACK;

    let room = rng.range(2.4, 3.8);
    let half = room * 0.5;
    let red = Color::new(rng.range(0.5, 0.75), 0.05, 0.05);
    let green = Color::new(0.1, rng.range(0.3, 0.5), 0.14);
    let (left, right) = if rng.next() < 0.5 {
        (red, green)
    } else {
        (green, red)
    };

    let walls = [
        (Vector3::new(0.0, 0.0, 0.0), 0.0, -FRAC_PI_2, Color::new(0.73, 0.73, 0.73), 60.0),
        (Vector3::new(-half, half, 0.0), FRAC_PI_2, 0.0, left, room),
        (Vector3::new(half, half, 0.0), -FRAC_PI_2, 0.0, right, room),
        (Vector3::new(0.0, half, -half), 0.0, 0.0, Color::new(0.73, 0.73, 0.70), room),
        (Vector3::new(0.0, half, half), std::f32::consts::PI, 0.0, Color::new(0.73, 0.73, 0.70), room),
        (Vector3::new(0.0, room, 0.0), 0.0, FRAC_PI_2, Color::new(0.8, 0.8, 0.78), room),
    ];
    for (pos, yaw, pitch, colour, size) in walls {
        let mut m = StandardMaterial::new(colour);
        m.roughness = rng.range(0.55, 1.0);
        let mut wall = Object3D::mesh(Mesh::new(
            PlaneGeometry::new(size, size),
            Material::Standard(m),
        ));
        wall.position = pos;
        wall.rotate_y(yaw);
        wall.rotate_x(pitch);
        scene.add(wall);
    }

    let objects = 1 + (rng.next() * 2.5) as u32;
    add_objects(&mut scene, rng, objects, half * 0.45);
    let panel_size = rng.range(0.4, 1.1);
    add_panel(&mut scene, rng, Vector3::new(0.0, room - 0.02, 0.0), panel_size);
    if rng.next() < 0.28 {
        let a = rng.range(0.0, std::f32::consts::TAU);
        let fill_pos = Vector3::new(
            a.cos() * half * 0.3,
            rng.range(0.8, room * 0.6),
            a.sin() * half * 0.3,
        );
        let fill_size = rng.range(0.25, 0.7);
        add_panel(&mut scene, rng, fill_pos, fill_size);
    }

    let yaw = rng.range(-0.45, 0.45);
    let mut camera = PerspectiveCamera::new(rng.range(35.0, 48.0), 1.0, 0.1, 100.0);
    camera.position = Vector3::new(yaw.sin() * half * 0.35, room * rng.range(0.4, 0.6), half * 0.72);
    camera.target = Vector3::new(0.0, room * 0.35, 0.0);
    Built {
        scene,
        camera,
        kind: format!("enclosed {room:.2}  {objects} obj"),
    }
}

fn open_ground(rng: &mut Rng, fixed: bool) -> Built {
    let mut scene = Scene::new();
    scene.background = Color::new(0.02, 0.03, 0.05);

    let mut floor_mat = StandardMaterial::new(if rng.next() < 0.5 {
        Color::new(0.55, 0.52, 0.48)
    } else {
        rng.color()
    });
    floor_mat.roughness = rng.range(0.4, 0.95);
    let mut floor = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(24.0, 24.0),
        Material::Standard(floor_mat),
    ));
    floor.rotate_x(-FRAC_PI_2);
    scene.add(floor);

    let objects = 1 + (rng.next() * 2.0) as u32;
    add_objects(&mut scene, rng, objects, 1.4);
    let light_pos = Vector3::new(
        rng.range(-1.2, 1.2),
        rng.range(3.5, 5.5),
        rng.range(-1.2, 1.2),
    );
    let light_size = rng.range(0.8, 2.4);
    add_panel(&mut scene, rng, light_pos, light_size);

    let angle = rng.range(0.4, 2.2);
    let dist = rng.range(4.5, 7.0);
    let mut camera = PerspectiveCamera::new(rng.range(38.0, 50.0), 1.0, 0.1, 100.0);
    camera.position = Vector3::new(angle.cos() * dist, rng.range(1.4, 2.8), angle.sin() * dist);
    camera.target = Vector3::new(0.0, 0.5, 0.0);
    Built {
        scene,
        camera,
        kind: if fixed {
            "open-holdout".into()
        } else {
            format!("open  {objects} obj")
        },
    }
}

/// A Cornell-shaped room with jittered size, camera, light, and objects.
///
/// Canonical [`cornell`] is hold-out only. This variant is close enough that
/// the kernel can learn to gather wall colour onto the floor, but never the
/// exact layout scored as hostile.
fn cornell_like(rng: &mut Rng) -> Built {
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    let s = rng.range(1.7, 2.55);
    let white = Color::new(
        rng.range(0.68, 0.80),
        rng.range(0.68, 0.80),
        rng.range(0.66, 0.78),
    );
    let red = Color::new(rng.range(0.48, 0.78), rng.range(0.02, 0.14), rng.range(0.02, 0.12));
    let green = Color::new(rng.range(0.04, 0.18), rng.range(0.32, 0.55), rng.range(0.08, 0.24));
    let (left_c, right_c) = if rng.next() < 0.5 {
        (red, green)
    } else {
        (green, red)
    };

    let mut floor = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(s, s),
        Material::Standard(StandardMaterial::new(white)),
    ));
    floor.rotate_x(-FRAC_PI_2);
    scene.add(floor);
    let mut ceiling = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(s, s),
        Material::Standard(StandardMaterial::new(white)),
    ));
    ceiling.position = Vector3::new(0.0, s, 0.0);
    ceiling.rotate_x(FRAC_PI_2);
    scene.add(ceiling);
    let mut back = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(s, s),
        Material::Standard(StandardMaterial::new(white)),
    ));
    back.position = Vector3::new(0.0, s * 0.5, -s * 0.5);
    scene.add(back);
    let mut left = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(s, s),
        Material::Standard(StandardMaterial::new(left_c)),
    ));
    left.position = Vector3::new(-s * 0.5, s * 0.5, 0.0);
    left.rotate_y(FRAC_PI_2);
    scene.add(left);
    let mut right = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(s, s),
        Material::Standard(StandardMaterial::new(right_c)),
    ));
    right.position = Vector3::new(s * 0.5, s * 0.5, 0.0);
    right.rotate_y(-FRAC_PI_2);
    scene.add(right);

    let lamp_size = rng.range(0.45, 0.9);
    let lamp_x = rng.range(-0.15, 0.15);
    let lamp_z = rng.range(-0.15, 0.15);
    add_panel(
        &mut scene,
        rng,
        Vector3::new(lamp_x, s - 0.01, lamp_z),
        lamp_size,
    );

    let spread = s * 0.28;
    let empty_frac = std::env::var("CORNELL_EMPTY")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
    let layout = rng.next();
    if layout < empty_frac {
        // Empty box — canonical-like floor/ceiling coverage.
    } else if layout < empty_frac + 0.22 {
        add_objects(&mut scene, rng, 2, spread);
    } else {
        let mut tall = StandardMaterial::new(Color::new(
            rng.range(0.70, 0.82),
            rng.range(0.70, 0.82),
            rng.range(0.68, 0.80),
        ));
        tall.roughness = rng.range(0.7, 0.95);
        let th = rng.range(0.9, 1.35);
        let tw = rng.range(0.45, 0.7);
        let mut block = Object3D::mesh(Mesh::new(
            BoxGeometry::new(tw, th, tw),
            Material::Standard(tall),
        ));
        block.position = Vector3::new(rng.range(-spread, -0.08), th * 0.5, rng.range(-spread, 0.05));
        block.rotate_y(rng.range(-0.5, 0.6));
        scene.add(block);
        let mut short = StandardMaterial::new(Color::new(
            rng.range(0.70, 0.82),
            rng.range(0.70, 0.82),
            rng.range(0.68, 0.80),
        ));
        short.roughness = rng.range(0.7, 0.95);
        let sh = rng.range(0.4, 0.75);
        let sw = rng.range(0.45, 0.7);
        let mut cube = Object3D::mesh(Mesh::new(
            BoxGeometry::new(sw, sh, sw),
            Material::Standard(short),
        ));
        cube.position = Vector3::new(rng.range(0.08, spread), sh * 0.5, rng.range(-0.05, spread));
        cube.rotate_y(rng.range(-0.6, 0.5));
        scene.add(cube);
    }

    if rng.next() < 0.22 {
        let mut glass_mat = PhysicalMaterial::new(Color::WHITE);
        glass_mat.transmission = 1.0;
        glass_mat.roughness = rng.range(0.0, 0.08);
        glass_mat.ior = rng.range(1.45, 1.55);
        let gr = rng.range(0.35, 0.55);
        let mut glass = Object3D::mesh(Mesh::new(
            SphereGeometry::new(gr, 24, 16),
            Material::Physical(glass_mat),
        ));
        glass.position = Vector3::new(rng.range(-0.35, 0.35), gr, rng.range(-0.25, 0.25));
        scene.add(glass);
    }

    let yaw = rng.range(-0.28, 0.28);
    let dist = s * rng.range(1.7, 2.15);
    let mut camera = PerspectiveCamera::new(rng.range(36.0, 46.0), 1.0, 0.1, 100.0);
    camera.position = Vector3::new(yaw.sin() * dist * 0.15, s * rng.range(0.42, 0.58), dist);
    camera.target = Vector3::new(0.0, s * rng.range(0.42, 0.55), 0.0);
    Built {
        scene,
        camera,
        kind: format!("cornell-like {s:.2}"),
    }
}

fn cornell(glass: bool) -> Built {
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    let white = || Material::Standard(StandardMaterial::new(Color::new(0.73, 0.73, 0.73)));
    let s = 2.0;

    let mut floor = Object3D::mesh(Mesh::new(PlaneGeometry::new(s, s), white()));
    floor.rotate_x(-FRAC_PI_2);
    scene.add(floor);
    let mut ceiling = Object3D::mesh(Mesh::new(PlaneGeometry::new(s, s), white()));
    ceiling.position = Vector3::new(0.0, s, 0.0);
    ceiling.rotate_x(FRAC_PI_2);
    scene.add(ceiling);
    let mut back = Object3D::mesh(Mesh::new(PlaneGeometry::new(s, s), white()));
    back.position = Vector3::new(0.0, s * 0.5, -s * 0.5);
    scene.add(back);
    let mut left = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(s, s),
        Material::Standard(StandardMaterial::new(Color::new(0.65, 0.05, 0.05))),
    ));
    left.position = Vector3::new(-s * 0.5, s * 0.5, 0.0);
    left.rotate_y(FRAC_PI_2);
    scene.add(left);
    let mut right = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(s, s),
        Material::Standard(StandardMaterial::new(Color::new(0.12, 0.45, 0.15))),
    ));
    right.position = Vector3::new(s * 0.5, s * 0.5, 0.0);
    right.rotate_y(-FRAC_PI_2);
    scene.add(right);

    let mut lamp_material = StandardMaterial::new(Color::BLACK);
    lamp_material.emissive = Color::new(1.0, 0.92, 0.78);
    lamp_material.emissive_intensity = 22.0;
    let mut lamp = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(0.6, 0.6),
        Material::Standard(lamp_material),
    ));
    lamp.position = Vector3::new(0.0, s - 0.01, 0.0);
    lamp.rotate_x(FRAC_PI_2);
    scene.add(lamp);

    if glass {
        let mut glass_mat = PhysicalMaterial::new(Color::WHITE);
        glass_mat.transmission = 1.0;
        glass_mat.roughness = 0.0;
        glass_mat.ior = 1.52;
        let mut ball = Object3D::mesh(Mesh::new(
            SphereGeometry::new(0.35, 48, 32),
            Material::Physical(glass_mat),
        ));
        ball.position = Vector3::new(0.38, 0.35, 0.25);
        scene.add(ball);
        let mut metal = PhysicalMaterial::new(Color::new(0.9, 0.9, 0.92));
        metal.metalness = 1.0;
        metal.roughness = 0.08;
        let mut mirror = Object3D::mesh(Mesh::new(
            SphereGeometry::new(0.35, 48, 32),
            Material::Physical(metal),
        ));
        mirror.position = Vector3::new(-0.38, 0.35, -0.25);
        scene.add(mirror);
    } else {
        let mut tall = StandardMaterial::new(Color::new(0.75, 0.75, 0.72));
        tall.roughness = 0.85;
        let mut block = Object3D::mesh(Mesh::new(
            BoxGeometry::new(0.6, 1.2, 0.6),
            Material::Standard(tall),
        ));
        block.position = Vector3::new(-0.35, 0.6, -0.3);
        block.rotate_y(0.3);
        scene.add(block);
        let mut short = StandardMaterial::new(Color::new(0.75, 0.75, 0.72));
        short.roughness = 0.85;
        let mut cube = Object3D::mesh(Mesh::new(
            BoxGeometry::new(0.6, 0.6, 0.6),
            Material::Standard(short),
        ));
        cube.position = Vector3::new(0.35, 0.3, 0.35);
        cube.rotate_y(-0.3);
        scene.add(cube);
    }

    let mut camera = PerspectiveCamera::new(40.0, 1.0, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 1.0, 3.9);
    camera.target = Vector3::new(0.0, 1.0, 0.0);
    Built {
        scene,
        camera,
        kind: if glass {
            "cornell-glass".into()
        } else {
            "cornell".into()
        },
    }
}

fn furnace() -> Built {
    let mut scene = Scene::new();
    scene.background = Color::new(0.5, 0.5, 0.5);
    let mut shell_mat = StandardMaterial::new(Color::BLACK);
    shell_mat.emissive = Color::WHITE;
    shell_mat.emissive_intensity = 1.0;
    shell_mat.side = 1;
    scene.add(Object3D::mesh(Mesh::new(
        SphereGeometry::new(30.0, 48, 32),
        Material::Standard(shell_mat),
    )));
    for (x, roughness, metalness) in [(-1.1f32, 0.35f32, 0.0f32), (1.1, 0.15, 1.0)] {
        let mut mat = PhysicalMaterial::new(Color::new(0.8, 0.8, 0.8));
        mat.roughness = roughness;
        mat.metalness = metalness;
        let mut ball = Object3D::mesh(Mesh::new(
            SphereGeometry::new(0.9, 48, 32),
            Material::Physical(mat),
        ));
        ball.position = Vector3::new(x, 0.0, 0.0);
        scene.add(ball);
    }
    let mut camera = PerspectiveCamera::new(45.0, 1.0, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 0.0, 5.5);
    camera.target = Vector3::ZERO;
    Built {
        scene,
        camera,
        kind: "furnace".into(),
    }
}

/// Jittered emissive shell — teaches hops to stay near the probe in uniform bright boxes.
fn furnace_like(rng: &mut Rng) -> Built {
    let mut scene = Scene::new();
    let grey = rng.range(0.35, 0.55);
    scene.background = Color::new(grey, grey, grey);
    let mut shell_mat = StandardMaterial::new(Color::BLACK);
    shell_mat.emissive = Color::new(
        rng.range(0.85, 1.0),
        rng.range(0.85, 1.0),
        rng.range(0.85, 1.0),
    );
    shell_mat.emissive_intensity = rng.range(0.85, 1.15);
    shell_mat.side = 1;
    scene.add(Object3D::mesh(Mesh::new(
        SphereGeometry::new(rng.range(22.0, 34.0), 40, 24),
        Material::Standard(shell_mat),
    )));
    let balls = 1 + (rng.next() * 2.0) as u32;
    for _ in 0..balls {
        let mut mat = PhysicalMaterial::new(Color::new(0.75, 0.75, 0.75));
        mat.roughness = rng.range(0.08, 0.45);
        mat.metalness = if rng.next() < 0.45 { 1.0 } else { 0.0 };
        let mut ball = Object3D::mesh(Mesh::new(
            SphereGeometry::new(rng.range(0.55, 1.05), 32, 20),
            Material::Physical(mat),
        ));
        ball.position = Vector3::new(
            rng.range(-1.4, 1.4),
            rng.range(-0.35, 0.35),
            rng.range(-0.8, 0.8),
        );
        scene.add(ball);
    }
    let mut camera = PerspectiveCamera::new(rng.range(40.0, 52.0), 1.0, 0.1, 100.0);
    let dist = rng.range(4.8, 6.2);
    let yaw = rng.range(-0.35, 0.35);
    camera.position = Vector3::new(yaw.sin() * dist * 0.25, rng.range(-0.2, 0.35), dist);
    camera.target = Vector3::new(0.0, 0.0, rng.range(-0.3, 0.3));
    Built {
        scene,
        camera,
        kind: "furnace-like".into(),
    }
}

fn add_objects(scene: &mut Scene, rng: &mut Rng, count: u32, spread: f32) {
    for _ in 0..count {
        let radius = rng.range(0.22, 0.48);
        let roll = rng.next();
        let material = if roll < 0.45 {
            let mut m = StandardMaterial::new(rng.color());
            m.roughness = rng.range(0.4, 0.95);
            Material::Standard(m)
        } else if roll < 0.75 {
            let mut m = StandardMaterial::new(rng.color());
            m.roughness = rng.range(0.05, 0.4);
            m.metalness = 1.0;
            Material::Standard(m)
        } else {
            let mut m = PhysicalMaterial::new(Color::WHITE);
            m.transmission = 1.0;
            m.roughness = rng.range(0.0, 0.12);
            m.ior = rng.range(1.3, 1.55);
            Material::Physical(m)
        };
        let mut obj = if rng.next() < 0.55 {
            Object3D::mesh(Mesh::new(SphereGeometry::new(radius, 24, 16), material))
        } else {
            Object3D::mesh(Mesh::new(
                BoxGeometry::new(radius * 1.6, radius * 1.6, radius * 1.6),
                material,
            ))
        };
        obj.position = Vector3::new(rng.range(-spread, spread), radius, rng.range(-spread, spread));
        obj.rotate_y(rng.range(0.0, 3.1));
        scene.add(obj);
    }
}

fn add_panel(scene: &mut Scene, rng: &mut Rng, position: Vector3, size: f32) {
    let mut em = StandardMaterial::new(Color::BLACK);
    let warm = rng.range(0.0, 1.0);
    em.emissive = Color::new(1.0, 0.85 + 0.12 * warm, 0.7 + 0.25 * warm);
    em.emissive_intensity = rng.range(12.0, 32.0) / size.max(0.2);
    let mut panel = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(size, size),
        Material::Standard(em),
    ));
    panel.position = position;
    panel.rotate_x(FRAC_PI_2);
    scene.add(panel);
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / 16_777_216.0
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next()
    }
    fn color(&mut self) -> Color {
        Color::new(
            self.range(0.1, 0.85),
            self.range(0.1, 0.85),
            self.range(0.1, 0.85),
        )
    }
}

const HOSTILE: u64 = 0x7e57_0001;
