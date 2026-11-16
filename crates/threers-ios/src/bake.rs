//! A flat mesh's [`ShaderMaterial`], drawn once with threers' own Metal
//! renderer for RealityKit, which runs no shader of an app's on visionOS
//! (`CustomMaterial` is not there): the mesh seen straight on into an image
//! — what the shader drew, where it drew it, clear where it discarded — and
//! each vertex's place on that image. A volume's slice through its scan is
//! one: its shader samples a 3D texture a fragment at a time, and the
//! volume shows the same planes the window does.

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use threers::cameras::OrthographicCamera;
use threers::core::Mesh;
use threers::materials::{Material, ShaderMaterial};
use threers::math::Vector3;
use threers::metal::MetalHeadlessRenderer;
use threers::{Object3D, Scene};

/// A baked material: its pixels (RGBA8, straight alpha, rows from the
/// top), size, and 2 uv floats a vertex.
pub struct Baked {
    pub key: u64,
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub uvs: Vec<f32>,
}

/// What a shader material draws, as a key: the same key, the same image.
/// Its source, its data and its volumes' and maps' identities (a new
/// `Arc` is new contents), its alpha and sides, and the geometry's.
pub fn key(m: &ShaderMaterial, mesh: &Mesh) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    m.fragment.hash(&mut h);
    m.msl.hash(&mut h);
    for v in m.data.iter().flatten() {
        v.to_bits().hash(&mut h);
    }
    for v in &m.volume_maps {
        (Arc::as_ptr(v) as usize).hash(&mut h);
    }
    for t in &m.maps {
        (Arc::as_ptr(t) as usize).hash(&mut h);
    }
    (m.transparent, m.side, m.opacity.to_bits()).hash(&mut h);
    (Arc::as_ptr(&mesh.geometry) as usize).hash(&mut h);
    h.finish()
}

/// Draws them, on a renderer made the first time one is asked for.
#[derive(Default)]
pub struct Baker {
    renderer: Option<MetalHeadlessRenderer>,
}

impl Baker {
    /// `mesh` (flat, its shader material) baked, `longest` pixels along
    /// its longer side; `None` when it is not flat, or there is no Metal
    /// device.
    pub fn bake(&mut self, mesh: &Mesh, longest_px: f32) -> Option<Baked> {
        let Material::Shader(m) = &*mesh.material else {
            return None;
        };
        let p = &mesh.geometry.get_attribute("position")?.array;
        let at = |i: usize| Vector3::new(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
        let n_verts = p.len() / 3;
        let (o, u, v, n) = plane(p)?;
        let (mut lo, mut hi) = ([f32::INFINITY; 2], [f32::NEG_INFINITY; 2]);
        for i in 0..n_verts {
            let d = at(i) - o;
            for (k, x) in [d.dot(u), d.dot(v)].into_iter().enumerate() {
                lo[k] = lo[k].min(x);
                hi[k] = hi[k].max(x);
            }
        }
        let ext = [(hi[0] - lo[0]).max(1e-6), (hi[1] - lo[1]).max(1e-6)];
        let longest = ext[0].max(ext[1]);
        let px = |e: f32| (e / longest * longest_px).round().clamp(16.0, 1024.0) as u32;
        let (width, height) = (px(ext[0]), px(ext[1]));
        // straight on, from the normal's side, up along v: right is u
        let centre = o + u * ((lo[0] + hi[0]) * 0.5) + v * ((lo[1] + hi[1]) * 0.5);
        let back = longest;
        let mut camera = OrthographicCamera::new(
            -ext[0] * 0.5,
            ext[0] * 0.5,
            ext[1] * 0.5,
            -ext[1] * 0.5,
            back * 0.5,
            back * 1.5,
        );
        camera.position = centre + n * back;
        camera.up = v;
        camera.look_at(centre);
        let rgba = self.shoot(mesh, &camera, width, height, m.transparent)?;
        // the image's top row is v's far end: uv (0, 0) its bottom left
        let uvs = (0..n_verts)
            .flat_map(|i| {
                let d = at(i) - o;
                [(d.dot(u) - lo[0]) / ext[0], (d.dot(v) - lo[1]) / ext[1]]
            })
            .collect();
        Some(Baked {
            key: key(m, mesh),
            rgba,
            width,
            height,
            uvs,
        })
    }

    /// `mesh` alone at its own frame's origin, through `camera`, into
    /// `width × height` RGBA, straight alpha.
    fn shoot(
        &mut self,
        mesh: &Mesh,
        camera: &dyn threers::cameras::Camera,
        width: u32,
        height: u32,
        blended: bool,
    ) -> Option<Vec<u8>> {
        let renderer = match &mut self.renderer {
            Some(r) => r,
            slot => slot.insert(
                MetalHeadlessRenderer::builder()
                    .size(width, height)
                    .build()
                    .ok()?,
            ),
        };
        renderer.resize(width, height).ok()?;
        let mut scene = Scene::new();
        scene.background_alpha = 0.0;
        scene.add(Object3D::mesh(mesh.clone()));
        renderer.render(&mut scene, camera).ok()?;
        let mut rgba = renderer.read_rgba().ok()?;
        // blended over nothing, the colour came back times its alpha
        if blended {
            for px in rgba.as_chunks_mut::<4>().0 {
                let a = px[3] as f32 / 255.0;
                if a > 0.0 && a < 1.0 {
                    for c in &mut px[..3] {
                        *c = (*c as f32 / a).round().min(255.0) as u8;
                    }
                }
            }
        }
        Some(rgba)
    }
}

/// Positions (3 floats a vertex) that lie in one plane: a point on it, two
/// directions along it, its normal. `None` when they do not.
pub fn plane(p: &[f32]) -> Option<(Vector3, Vector3, Vector3, Vector3)> {
    let at = |i: usize| Vector3::new(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
    let n_verts = p.len() / 3;
    // its first edge, and the normal of the first triangle that has one
    let o = at(0);
    let u = (1..n_verts)
        .map(|i| at(i) - o)
        .find(|e| e.length() > 1e-9)?
        .normalize();
    let n = (1..n_verts)
        .map(|i| u.cross(at(i) - o))
        .find(|c| c.length() > 1e-9)?
        .normalize();
    let v = n.cross(u);
    (0..n_verts)
        .all(|i| {
            let d = at(i) - o;
            d.dot(n).abs() <= 1e-4 * (1.0 + d.length())
        })
        .then_some((o, u, v, n))
}

#[cfg(test)]
mod tests {
    use super::*;
    use threers::core::{BufferAttribute, BufferGeometry};

    #[test]
    fn a_flat_shader_mesh_bakes_to_what_it_draws() {
        if threers::metal::MetalDevice::new().is_err() {
            eprintln!("skipping: no Metal device");
            return;
        }
        // a 2 × 1 quad in the x–z plane, its left half red, its right cut
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            BufferAttribute::new(
                vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                3,
            ),
        );
        g.set_attribute("normal", BufferAttribute::new(vec![0.0; 12], 3));
        g.set_index(vec![0, 1, 2, 0, 2, 3]);
        let mut m = ShaderMaterial::new(
            "@fragment fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
            if (in.local_pos.x > 1.0) { discard; }
            return vec4<f32>(1.0, 0.0, 0.0, 1.0);
        }",
        );
        m.tone_mapped = false;
        m.side = 2;
        let mesh = Mesh::new(g, Material::Shader(m));
        let b = Baker::default().bake(&mesh, 768.0).expect("baked");
        assert_eq!((b.width, b.height), (768, 384));
        let px = |u: f32, v: f32| {
            let (x, y) = (
                (u * b.width as f32) as usize,
                ((1.0 - v) * b.height as f32) as usize,
            );
            let i = (y * b.width as usize + x) * 4;
            [b.rgba[i], b.rgba[i + 1], b.rgba[i + 2], b.rgba[i + 3]]
        };
        // each vertex's place: x along u, z along v
        let uv: Vec<[f32; 2]> = b.uvs.chunks(2).map(|c| [c[0], c[1]]).collect();
        assert_eq!(uv[0], [0.0, 0.0]);
        assert_eq!(uv[2], [1.0, 1.0]);
        // left drawn, right clear, wherever the plane faces
        assert_eq!(px(0.25, 0.5), [255, 0, 0, 255]);
        assert_eq!(px(0.75, 0.5)[3], 0);
    }
}
