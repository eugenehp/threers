//! Marks handed to the scene renderer: `PolygonMarks::to_buffer_geometry`
//! and `ShapeMarks::to_instanced_mesh` are lit, cast shadows and keep their
//! colours, far from zero against an `Origin`.
//!
//! Skipped when no GPU adapter is available.

use threers::core::{Mesh, Object3D};
use threers::geometries::PlaneGeometry;
use threers::lights::{AmbientLight, DirectionalLight, Light};
use threers::marks::{MarkView, Origin, PolygonFeature, PolygonMarks, ShapeMarks};
use threers::materials::{Material, StandardMaterial};
use threers::math::{Color, Vector3};
use threers::{BoxGeometry, HeadlessRenderer, PerspectiveCamera, Scene};

const W: u32 = 256;
const H: u32 = 128;
/// Data a long way from zero, as map data is.
const CENTRE: [f64; 3] = [1_000_000.0, 0.0, -2_000_000.0];

fn white() -> Material {
    let mut m = StandardMaterial::new(Color::WHITE);
    m.roughness = 0.9;
    Material::Standard(m)
}

#[test]
fn polygons_and_shapes_render_lit_and_shadowed_as_meshes() {
    let Ok(mut hr) = HeadlessRenderer::builder().size(W, H).build() else {
        return;
    };
    let origin = Origin::new(CENTRE[0], CENTRE[1], CENTRE[2]);
    // A 40 × 40 block 30 tall, red, around the centre (ground y is world -z).
    let (gx, gy) = (CENTRE[0], -CENTRE[2]);
    let mut blocks = PolygonMarks::new();
    blocks.set_polygons(
        &[PolygonFeature::new(vec![
            [gx - 20.0, gy - 20.0],
            [gx + 20.0, gy - 20.0],
            [gx + 20.0, gy + 20.0],
            [gx - 20.0, gy + 20.0],
        ])],
        0.0,
    );
    blocks.set_elevations(&[30.0]);
    blocks.set_colors(&[[220, 30, 30, 255]]);
    // Two boxes, 10 a side, to the north: green and blue.
    let mut boxes = ShapeMarks::new(&BoxGeometry::new(1.0, 1.0, 1.0));
    boxes.set_positions(&[
        [CENTRE[0] - 30.0, 5.0, CENTRE[2] - 45.0],
        [CENTRE[0] + 30.0, 5.0, CENTRE[2] - 45.0],
    ]);
    boxes.set_colors(&[[30, 200, 30, 255], [30, 30, 220, 255]]);
    boxes.common.style.size_scale = 10.0;

    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    scene.add_light(AmbientLight::new(Color::WHITE, 0.3));
    // Low from the east: shadows fall west.
    let mut sun = DirectionalLight::new(Color::WHITE, 2.5);
    sun.direction = Vector3::new(-1.0, -1.0, 0.0).normalize();
    sun.cast_shadow = true;
    sun.shadow.map_size = 2048;
    sun.shadow.camera_size = 300.0;
    sun.shadow.camera_near = 1.0;
    sun.shadow.camera_far = 600.0;
    let mut light = Object3D::light(Light::Directional(sun));
    light.position = Vector3::new(150.0, 150.0, 0.0);
    scene.add(light);
    let mut ground = Object3D::mesh(Mesh::new(PlaneGeometry::new(600.0, 600.0), white()));
    ground.rotate_x(-std::f32::consts::FRAC_PI_2);
    ground.receive_shadow = true;
    scene.add(ground);
    let mut block_mesh = Object3D::mesh(Mesh::new(blocks.to_buffer_geometry(&origin), white()));
    block_mesh.cast_shadow = true;
    scene.add(block_mesh);
    let mut box_mesh = Object3D::instanced_mesh(boxes.to_instanced_mesh(white(), &origin));
    box_mesh.cast_shadow = true;
    scene.add(box_mesh);

    // Straight down, north up the screen.
    let mut cam = PerspectiveCamera::new(60.0, W as f32 / H as f32, 1.0, 500.0);
    cam.position = Vector3::new(0.0, 150.0, 0.0);
    cam.target = Vector3::new(0.0, 0.0, 0.0);
    cam.up = Vector3::new(0.0, 0.0, -1.0);
    let img = hr.render_to_rgba(&mut scene, &cam);
    let view = MarkView::new(&cam, origin, (W, H));
    let at = |p: [f64; 3]| {
        let s = view.project(p).unwrap();
        let o = ((s[1] as u32 * W + s[0] as u32) * 4) as usize;
        [img[o], img[o + 1], img[o + 2]]
    };
    let roof = at([CENTRE[0], 30.0, CENTRE[2]]);
    assert!(
        roof[0] > 120 && roof[1] < 60 && roof[2] < 60,
        "red roof: {roof:?}"
    );
    let green = at([CENTRE[0] - 30.0, 10.0, CENTRE[2] - 45.0]);
    assert!(
        green[1] > 100 && green[0] < 60 && green[2] < 60,
        "green box: {green:?}"
    );
    let blue = at([CENTRE[0] + 30.0, 10.0, CENTRE[2] - 45.0]);
    assert!(
        blue[2] > 100 && blue[0] < 60 && blue[1] < 60,
        "blue box: {blue:?}"
    );
    // The block's shadow: west of it dark, east of it lit.
    let lum = |p: [u8; 3]| p.iter().map(|&v| v as u32).sum::<u32>();
    let west = at([CENTRE[0] - 35.0, 0.0, CENTRE[2]]);
    let east = at([CENTRE[0] + 35.0, 0.0, CENTRE[2]]);
    assert!(
        lum(west) * 2 < lum(east),
        "shadow west {west:?} against lit east {east:?}"
    );
}
