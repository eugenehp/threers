//! The cases that are hard for a vector renderer, drawn side by side.
//!
//! ```text
//! cargo run --example svg_gallery
//! ```
//!
//! Every scene here is one that `tests/svg_render.rs` checks against ray-cast
//! ground truth. The test says whether each pixel is right; this writes the
//! same scenes out so you can look at them, which is the part a pass/fail
//! number cannot give you.
//!
//! A painter's algorithm orders whole faces, so the interesting question is
//! always "which face wins where two of them overlap":
//!
//! | Scene | What it is about |
//! |-------|------------------|
//! | `ground_plane` | Two triangles of floor, objects standing on them |
//! | `floor_behind_camera` | The same, with the floor running off past the eye |
//! | `camera_inside` | Camera low and close, so the near plane cuts the floor |
//! | `knot` | Thousands of small faces occluding each other |
//! | `overlapping_spheres` | Convex objects at a range of depths |
//! | `frame_edges` | Geometry running off all four sides |
//! | `receding_wall` | A big polygon hiding a small object *behind* it |
//! | `interpenetrating` | Two solids pushed through each other — the known limit |
//! | `orthographic` | No perspective divide; `w` is 1 everywhere |

use threers::prelude::*;
use threers::{OrthographicCamera, SphereGeometry, TorusKnotGeometry};

const W: u32 = 640;
const H: u32 = 480;
const BG: u32 = 0x141a22;

fn flat(hex: u32) -> Material {
    Material::Basic(BasicMaterial::new(Color::from_hex(hex)))
}

fn mesh_at(geometry: BufferGeometry, hex: u32, pos: Vector3) -> Object3D {
    let mut o = Object3D::mesh(Mesh::new(geometry, flat(hex)));
    o.position = pos;
    o
}

fn ground(size: f32, hex: u32, y: f32) -> Object3D {
    let mut f = Object3D::mesh(Mesh::new(PlaneGeometry::new(size, size), flat(hex)));
    f.position = Vector3::new(0.0, y, 0.0);
    f.quaternion =
        Quaternion::from_axis_angle(Vector3::new(1.0, 0.0, 0.0), -std::f32::consts::FRAC_PI_2);
    f
}

fn scene_with(bg: u32) -> Scene {
    let mut s = Scene::new();
    s.background = Color::from_hex(bg);
    s
}

fn look(pos: Vector3, at: Vector3) -> PerspectiveCamera {
    let mut c = PerspectiveCamera::new(45.0, W as f32 / H as f32, 0.1, 200.0);
    c.position = pos;
    c.target = at;
    c
}

fn write(name: &str, scene: &mut Scene, camera: &dyn threers::cameras::Camera) {
    let path = format!("out/svg_gallery_{name}.svg");
    SvgRenderer::new(W, H)
        .with_options(SvgOptions {
            shading: SvgShading::Flat,
            ..SvgOptions::default()
        })
        .render_to_file(scene, camera, &path)
        .expect("write svg");
    let svg = std::fs::read_to_string(&path).expect("read back");
    println!(
        "{path:<40} {:>6} paths  {:>5} KB",
        svg.matches("<path").count(),
        svg.len() / 1024
    );
}

fn main() {
    let mut s = scene_with(BG);
    s.add(ground(80.0, 0x336699, 0.0));
    s.add(mesh_at(
        BoxGeometry::new(2.0, 2.0, 2.0),
        0xcc3311,
        Vector3::new(-2.0, 1.0, 0.0),
    ));
    s.add(mesh_at(
        SphereGeometry::new(1.0, 24, 16),
        0x22aa55,
        Vector3::new(1.6, 1.0, 0.5),
    ));
    write(
        "ground_plane",
        &mut s,
        &look(Vector3::new(0.0, 3.0, 9.0), Vector3::new(0.0, 1.0, 0.0)),
    );

    let mut s = scene_with(BG);
    s.add(ground(60.0, 0x6699cc, 0.0));
    s.add(mesh_at(
        BoxGeometry::new(2.0, 2.0, 2.0),
        0xcc3311,
        Vector3::new(0.0, 1.0, 0.0),
    ));
    write(
        "floor_behind_camera",
        &mut s,
        &look(Vector3::new(0.0, 3.0, 8.0), Vector3::new(0.0, 1.0, 0.0)),
    );

    let mut s = scene_with(BG);
    s.add(ground(60.0, 0x336699, -1.0));
    s.add(mesh_at(
        BoxGeometry::new(3.0, 3.0, 3.0),
        0xcc3311,
        Vector3::new(0.0, 0.5, -6.0),
    ));
    write(
        "camera_inside",
        &mut s,
        &look(Vector3::new(0.0, 0.4, 2.0), Vector3::new(0.0, 0.3, -6.0)),
    );

    let mut s = scene_with(BG);
    s.add(mesh_at(
        TorusKnotGeometry::new(1.2, 0.42, 120, 18, 2, 3),
        0xd4553c,
        Vector3::ZERO,
    ));
    write(
        "knot",
        &mut s,
        &look(Vector3::new(4.0, 3.0, 6.0), Vector3::ZERO),
    );

    let mut s = scene_with(BG);
    for (i, hex) in [0xcc3311u32, 0x22aa55, 0x3366cc, 0xddaa22]
        .iter()
        .enumerate()
    {
        s.add(mesh_at(
            SphereGeometry::new(1.1, 28, 18),
            *hex,
            Vector3::new(i as f32 * 0.7 - 1.0, 0.0, i as f32 * -1.6),
        ));
    }
    write(
        "overlapping_spheres",
        &mut s,
        &look(Vector3::new(0.0, 1.0, 7.0), Vector3::ZERO),
    );

    let mut s = scene_with(BG);
    s.add(mesh_at(
        BoxGeometry::new(14.0, 0.6, 0.6),
        0xcc3311,
        Vector3::new(0.0, 1.2, 0.0),
    ));
    s.add(mesh_at(
        BoxGeometry::new(0.6, 14.0, 0.6),
        0x22aa55,
        Vector3::new(0.0, 0.0, 0.6),
    ));
    s.add(mesh_at(
        PlaneGeometry::new(30.0, 30.0),
        0x3366cc,
        Vector3::new(0.0, 0.0, -3.0),
    ));
    write(
        "frame_edges",
        &mut s,
        &look(Vector3::new(0.0, 0.0, 6.0), Vector3::ZERO),
    );

    let mut s = scene_with(BG);
    let mut wall = Object3D::mesh(Mesh::new(PlaneGeometry::new(20.0, 8.0), flat(0x996633)));
    wall.quaternion = Quaternion::from_axis_angle(Vector3::UP, 60f32.to_radians());
    s.add(wall);
    s.add(mesh_at(
        SphereGeometry::new(1.0, 28, 18),
        0x22aa55,
        Vector3::new(-3.0, 0.0, 2.0),
    ));
    write(
        "receding_wall",
        &mut s,
        &look(Vector3::new(0.0, 2.0, 14.0), Vector3::ZERO),
    );

    let mut s = scene_with(BG);
    s.add(mesh_at(
        BoxGeometry::new(2.0, 4.0, 2.0),
        0xcc3311,
        Vector3::ZERO,
    ));
    s.add(mesh_at(
        BoxGeometry::new(8.0, 0.8, 0.8),
        0x22aa55,
        Vector3::new(0.0, 0.5, 0.0),
    ));
    write(
        "interpenetrating",
        &mut s,
        &look(Vector3::new(4.0, 2.0, 7.0), Vector3::ZERO),
    );

    let mut s = scene_with(BG);
    s.add(ground(40.0, 0x336699, 0.0));
    s.add(mesh_at(
        BoxGeometry::new(2.0, 2.0, 2.0),
        0xcc3311,
        Vector3::new(-1.6, 1.0, 0.0),
    ));
    s.add(mesh_at(
        SphereGeometry::new(1.0, 24, 16),
        0x22aa55,
        Vector3::new(1.6, 1.0, 0.0),
    ));
    let mut ortho = OrthographicCamera::new(-5.0, 5.0, 3.75, -3.75, 0.1, 100.0);
    ortho.position = Vector3::new(4.0, 3.0, 6.0);
    ortho.target = Vector3::new(0.0, 1.0, 0.0);
    write("orthographic", &mut s, &ortho);
}
