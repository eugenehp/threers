//! The scene, quad view and pixel helpers shared by the wgpu quad-view suite
//! (`tests/quad_view.rs`) and the Metal one (`tests/metal_quad_view.rs`), so
//! the two backends are probed with the same values.
//!
//! Every probe is a value no other code path writes: each pane its own
//! background, the gap magenta, the markers unlit primaries.

#![allow(dead_code)]

use threers::cameras::PerspectiveCamera;
use threers::core::{Mesh, Object3D};
use threers::geometries::BoxGeometry;
use threers::helpers::QuadView;
use threers::materials::{BasicMaterial, Material};
use threers::math::{Color, Vector3};
use threers::renderer::ViewportRect;
use threers::scene::Scene;

pub const WHITE: [u8; 3] = [255, 255, 255];
pub const RED: [u8; 3] = [255, 0, 0];
pub const GREEN: [u8; 3] = [0, 255, 0];
pub const BLUE: [u8; 3] = [0, 0, 255];

/// A white cube at the origin with unlit markers on +X (red), −Z (green) and
/// +Y (blue): enough to tell every orthographic view's axes apart.
pub fn marker_scene() -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x0a0b0c);
    let mut add = |size: f32, at: Vector3, rgb: [u8; 3]| {
        let c = Color::new(
            rgb[0] as f32 / 255.0,
            rgb[1] as f32 / 255.0,
            rgb[2] as f32 / 255.0,
        );
        let mut o = Object3D::mesh(Mesh::new(
            BoxGeometry::new(size, size, size),
            Material::Basic(BasicMaterial::new(c)),
        ));
        o.position = at;
        scene.add(o);
    };
    add(1.0, Vector3::ZERO, WHITE);
    add(0.4, Vector3::new(1.5, 0.0, 0.0), RED);
    add(0.4, Vector3::new(0.0, 0.0, -1.5), GREEN);
    add(0.4, Vector3::new(0.0, 1.5, 0.0), BLUE);
    scene
}

/// Unlit white box turned off every axis on a dark background: its edges
/// cross pixels at every angle, and unlit there are exactly two colours
/// without anti-aliasing — anything else on a pane is coverage.
pub fn tilted_box_scene() -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    let mut o = Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.2, 0.8, 1.0),
        Material::Basic(BasicMaterial::new(Color::WHITE)),
    ));
    o.rotate_y(0.5);
    o.rotate_x(0.35);
    o.rotate_z(0.2);
    scene.add(o);
    scene
}

/// A floor with a box over it, lit by a shadow-casting directional light
/// and a shadow-casting spot light.
pub fn shadow_scene() -> Scene {
    use threers::geometries::PlaneGeometry;
    use threers::lights::{AmbientLight, DirectionalLight, Light, SpotLight};
    use threers::materials::StandardMaterial;
    let mut s = Scene::new();
    s.background = Color::from_hex(0x101418);
    s.add_light(AmbientLight::new(Color::WHITE, 0.25));
    let mut sun = DirectionalLight::new(Color::WHITE, 2.0);
    sun.direction = Vector3::new(-0.45, -1.0, -0.3).normalize();
    sun.cast_shadow = true;
    sun.shadow.map_size = 2048;
    sun.shadow.camera_size = 5.0;
    s.add_light(sun);
    let mut spot = SpotLight::new(Color::WHITE, 40.0);
    spot.angle = 0.6;
    spot.distance = 30.0;
    spot.cast_shadow = true;
    spot.shadow.camera_near = 0.5;
    spot.shadow.camera_far = 30.0;
    spot.direction = Vector3::new(-0.5, -1.0, 0.4).normalize();
    let mut spot_obj = Object3D::light(Light::Spot(spot));
    spot_obj.position = Vector3::new(2.5, 5.0, -2.0);
    s.add(spot_obj);
    let mat = |hex: u32| {
        let mut m = StandardMaterial::new(Color::from_hex(hex));
        m.metalness = 0.0;
        m.roughness = 0.9;
        Material::Standard(m)
    };
    let mut floor = Object3D::mesh(Mesh::new(PlaneGeometry::new(6.0, 6.0), mat(0xd0d0d0)));
    floor.rotate_x(-std::f32::consts::FRAC_PI_2);
    floor.receive_shadow = true;
    s.add(floor);
    let mut b = Object3D::mesh(Mesh::new(BoxGeometry::new(1.2, 1.2, 1.2), mat(0xc08040)));
    b.position = Vector3::new(0.0, 1.2, 0.0);
    b.cast_shadow = true;
    b.receive_shadow = true;
    s.add(b);
    s
}

pub fn persp() -> PerspectiveCamera {
    let mut c = PerspectiveCamera::new(45.0, 1.0, 0.1, 100.0);
    c.position = Vector3::new(4.0, 3.0, 5.0);
    c.look_at(Vector3::ZERO);
    c
}

/// A 6 px gap and 4 px border in magenta, and a background per pane.
pub fn quad() -> QuadView {
    let mut q = QuadView::new();
    q.gap = 6.0;
    q.border = 4.0;
    q.gap_color = Some(Color::from_hex(0xff00ff));
    q.backgrounds = [
        Some(Color::from_hex(0x102030)), // top
        Some(Color::from_hex(0x203010)), // front
        Some(Color::from_hex(0x301020)), // side
        Some(Color::from_hex(0x404040)), // perspective
    ];
    q
}

/// The bytes a background or clear colour lands as: [`Color`] is linear and
/// the output encodes, so a `from_hex` colour comes back as its hex.
pub fn rgb(c: Color) -> [u8; 3] {
    let h = c.to_hex();
    [(h >> 16) as u8, (h >> 8) as u8, h as u8]
}

/// Pixel at `(x, y)` from the **top-left**, in a `w`-wide RGBA buffer.
pub fn px(img: &[u8], w: u32, x: u32, y: u32) -> [u8; 3] {
    let i = ((y * w + x) * 4) as usize;
    [img[i], img[i + 1], img[i + 2]]
}

/// A bottom-left-origin rect as top-left integer bounds `(x0, y0, x1, y1)` in
/// an `h`-tall image, `scale` image pixels to the rect's unit.
pub fn tl(r: ViewportRect, h: u32, scale: u32) -> (u32, u32, u32, u32) {
    let s = scale as f32;
    let x0 = (r.x * s) as u32;
    let x1 = ((r.x + r.width) * s) as u32;
    let y0 = h - ((r.y + r.height) * s) as u32;
    let y1 = h - (r.y * s) as u32;
    (x0, y0, x1, y1)
}

/// Centroid of the pixels of exactly `color` inside `b`, relative to the
/// box's centre, in pixels with y **up**. `None` when there are none.
pub fn offset_of(
    img: &[u8],
    w: u32,
    b: (u32, u32, u32, u32),
    color: [u8; 3],
) -> Option<(f32, f32)> {
    let (mut sx, mut sy, mut n) = (0.0f64, 0.0f64, 0u32);
    for y in b.1..b.3 {
        for x in b.0..b.2 {
            if px(img, w, x, y) == color {
                sx += x as f64;
                sy += y as f64;
                n += 1;
            }
        }
    }
    (n > 0).then(|| {
        let cx = (b.0 + b.2) as f64 * 0.5;
        let cy = (b.1 + b.3) as f64 * 0.5;
        ((sx / n as f64 - cx) as f32, (cy - sy / n as f64) as f32)
    })
}
