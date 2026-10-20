//! three.js `polygonOffset` on a decal: a quad lying on a tilted plane, cut
//! with a different tessellation (so its interpolated depths round
//! differently from the plane's — the decal-geometry case), drawn after it.
//! Without an offset the two z-fight; with `polygonOffsetFactor: -1,
//! polygonOffsetUnits: -4` (the usual decal setting) the decal wins every
//! pixel, as in WebGL. Shared by `tests/polygon_offset.rs` (wgpu) and
//! `tests/metal_backend.rs` through `#[path]`.
#![allow(dead_code)]

use threers::cameras::{Camera, PerspectiveCamera};
use threers::core::{Mesh, Object3D};
use threers::geometries::PlaneGeometry;
use threers::materials::{BasicMaterial, PolygonOffset};
use threers::math::{Color, Euler, Vector3};
use threers::scene::Scene;

pub const N: u32 = 128;

pub fn camera() -> PerspectiveCamera {
    let mut cam = PerspectiveCamera::new(40.0, 1.0, 0.5, 60.0);
    cam.position = Vector3::new(0.0, 0.0, 6.0);
    cam.look_at(Vector3::ZERO);
    cam
}

pub fn scene(offset: Option<PolygonOffset>) -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    let tilt = Euler::new(-1.1, 0.35, 0.0);
    let mut base = Object3D::mesh(Mesh::new(
        PlaneGeometry::with_segments(4.0, 4.0, 3, 3),
        BasicMaterial::new(Color::from_hex(0xff0000)).into(),
    ));
    base.set_rotation(tilt);
    scene.add(base);
    let mut decal_mat = BasicMaterial::new(Color::from_hex(0x0000ff));
    decal_mat.polygon_offset = offset;
    let mut decal = Object3D::mesh(Mesh::new(
        PlaneGeometry::with_segments(2.2, 2.2, 7, 5),
        decal_mat.into(),
    ));
    decal.set_rotation(tilt);
    decal.render_order = 1;
    scene.add(decal);
    scene
}

/// (decal pixels drawn blue, decal pixels drawn red) — the decal's interior
/// being where either colour shows inside its projected footprint, found as
/// the pixels the decal covers when drawn alone.
pub fn measure(render: &mut dyn FnMut(&mut Scene, &dyn Camera) -> Vec<u8>) -> [(usize, usize); 2] {
    let cam = camera();
    // The footprint: the decal over nothing.
    let mut alone = scene(None);
    let base_id = alone.arena.get(alone.root).unwrap().children[0];
    alone.arena.get_mut(base_id).unwrap().visible = false;
    let footprint: Vec<bool> = render(&mut alone, &cam).chunks(4).map(|p| p[2] > 128).collect();
    let mut out = [(0, 0); 2];
    for (k, offset) in [None, Some(PolygonOffset::new(-1.0, -4.0))].into_iter().enumerate() {
        let rgba = render(&mut scene(offset), &cam);
        for (i, p) in rgba.chunks(4).enumerate() {
            let (x, y) = ((i as u32) % N, (i as u32) / N);
            // Interior only: away from the footprint's edge.
            let inside = (-2i32..=2).all(|dy| {
                (-2i32..=2).all(|dx| {
                    let (xx, yy) = (x as i32 + dx, y as i32 + dy);
                    xx >= 0 && yy >= 0 && xx < N as i32 && yy < N as i32
                        && footprint[(yy as u32 * N + xx as u32) as usize]
                })
            });
            if !inside {
                continue;
            }
            if p[2] > 128 && p[0] < 64 {
                out[k].0 += 1;
            } else if p[0] > 128 {
                out[k].1 += 1;
            }
        }
    }
    out
}
