//! Raycaster timings, in microseconds per ray. Ignored by default:
//!
//! ```sh
//! cargo test --release --features mesh-bvh --test raycast_bench -- --ignored --nocapture
//! ```
//!
//! A timing harness rather than an assertion: the numbers depend on the
//! machine. Every scene is built the way a caller builds it — geometry straight
//! from its constructor, no bounding volumes computed up front — because that
//! is what a pick in a real app runs against.
//!
//! Each case is timed twice in the same run, interleaved: through the current
//! [`Raycaster`] and through [`legacy`], a copy of the raycaster as it was
//! before the three.js-parity rewrite (no side culling, positions collected
//! per ray, a linear polygon lookup, meshes only). Same process, same load,
//! same rays, so the two columns compare even on a busy machine.

use std::hint::black_box;
use std::time::Instant;
use threers::core::{InstancedMesh, Mesh, Object3D, ObjectArena, ObjectId, Raycaster};
use threers::materials::{BasicMaterial, Material};
use threers::math::{Color, Matrix4, Vector3};
use threers::{BoxGeometry, BufferGeometry, PlaneGeometry, SphereGeometry};

/// The raycaster before the rewrite, kept only to be timed against.
mod legacy {
    use threers::core::{ObjectArena, ObjectId, ObjectKind, Raycaster};
    use threers::math::{Sphere, Triangle, Vector3};

    fn inv_world_scale(m: &threers::math::Matrix4) -> f32 {
        let e = &m.elements;
        let sx = Vector3::new(e[0], e[1], e[2]).length();
        let sy = Vector3::new(e[4], e[5], e[6]).length();
        let sz = Vector3::new(e[8], e[9], e[10]).length();
        let s = (sx + sy + sz) / 3.0;
        if s == 0.0 {
            1.0
        } else {
            1.0 / s
        }
    }

    /// `(distance, face_index, polygon_index)`, sorted.
    pub fn intersect(
        rc: &Raycaster,
        arena: &ObjectArena,
        root: ObjectId,
    ) -> Vec<(f32, usize, Option<usize>)> {
        let mut out = Vec::new();
        arena.traverse_visible(root, &mut |_, obj| {
            if !rc.layers.test(&obj.layers) {
                return;
            }
            let ObjectKind::Mesh(mesh) = &obj.kind else {
                return;
            };
            let g = &mesh.geometry;
            let world_sphere = match g.bounding_sphere {
                Some(s) => s.apply_matrix4(&obj.matrix_world),
                None => {
                    let Some(iter) = g.positions() else {
                        return;
                    };
                    let pts: Vec<Vector3> = iter.collect();
                    if pts.is_empty() {
                        return;
                    }
                    Sphere::from_points(&pts).apply_matrix4(&obj.matrix_world)
                }
            };
            if !rc.ray.intersects_sphere(&world_sphere) {
                return;
            }
            let first = out.len();
            let inv_world = obj.matrix_world.invert();
            let local_ray = rc.ray.apply_matrix4(&inv_world);
            let max_local_t = if rc.far.is_finite() {
                rc.far * inv_world_scale(&obj.matrix_world)
            } else {
                f32::INFINITY
            };
            let mut push = |face: usize, t: f32| {
                let world_point = local_ray.at(t).apply_matrix4(&obj.matrix_world);
                let d = (world_point - rc.ray.origin).length();
                if d >= rc.near && d <= rc.far {
                    out.push((d, face, None));
                }
            };
            // Set only by the BVH walk, which only `mesh-bvh` builds.
            #[cfg_attr(not(feature = "mesh-bvh"), allow(unused_mut))]
            let mut walked = false;
            #[cfg(feature = "mesh-bvh")]
            if let Some(bvh) = &g.bounds_tree {
                let near_local = rc.near * inv_world_scale(&obj.matrix_world);
                for h in bvh.raycast(&local_ray, near_local, max_local_t, false) {
                    push(h.face_index, h.distance);
                }
                walked = true;
            }
            if !walked {
                let positions: Vec<Vector3> = match g.positions() {
                    Some(it) => it.collect(),
                    None => return,
                };
                let tri = |i: usize| -> [usize; 3] {
                    match &g.index {
                        Some(idx) => [
                            idx[i * 3] as usize,
                            idx[i * 3 + 1] as usize,
                            idx[i * 3 + 2] as usize,
                        ],
                        None => [i * 3, i * 3 + 1, i * 3 + 2],
                    }
                };
                let n = match &g.index {
                    Some(idx) => idx.len() / 3,
                    None => positions.len() / 3,
                };
                for i in 0..n {
                    let [a, b, c] = tri(i);
                    if a >= positions.len() || b >= positions.len() || c >= positions.len() {
                        continue;
                    }
                    let t = Triangle::new(positions[a], positions[b], positions[c]);
                    if let Some(t) = local_ray.intersect_triangle(&t, false) {
                        if t <= max_local_t {
                            push(i, t);
                        }
                    }
                }
            }
            if out.len() > first {
                let faces: Vec<usize> = out[first..].iter().map(|h| h.1).collect();
                for (h, p) in out[first..].iter_mut().zip(g.polygons_of_triangles(&faces)) {
                    h.2 = p;
                }
            }
        });
        out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        out
    }
}

fn material() -> Material {
    Material::Basic(BasicMaterial::new(Color::from_hex(0xffffff)))
}

/// A small deterministic generator, so every run casts the same rays.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next()
    }
}

fn scene_with(geometry: BufferGeometry) -> (ObjectArena, ObjectId) {
    let mut arena = ObjectArena::new();
    let root = arena.insert(Object3D::group());
    let id = arena.insert(Object3D::mesh(Mesh::new(geometry, material())));
    arena.add_child(root, id);
    arena.update_world_matrices(root, Matrix4::identity());
    (arena, root)
}

/// Mean microseconds per ray over `rays`, after one warm-up pass, plus the
/// total hit count (so the work cannot be optimised away).
fn time(rays: &[Raycaster], passes: usize, cast: impl Fn(&Raycaster) -> usize) -> (f64, usize) {
    let hits = rays.iter().map(&cast).sum();
    let start = Instant::now();
    for _ in 0..passes {
        for rc in rays {
            black_box(cast(rc));
        }
    }
    let us = start.elapsed().as_secs_f64() * 1e6 / (passes * rays.len()) as f64;
    (us, hits)
}

/// One row: legacy, current, current with `first_hit_only`.
fn row(name: &str, arena: &ObjectArena, root: ObjectId, rays: &[Raycaster], passes: usize) {
    let (old_us, old_hits) = time(rays, passes, |rc| legacy::intersect(rc, arena, root).len());
    let (new_us, new_hits) = time(rays, passes, |rc| {
        rc.intersect_objects(arena, root, true).len()
    });
    let first: Vec<Raycaster> = rays
        .iter()
        .map(|r| Raycaster {
            first_hit_only: true,
            ..*r
        })
        .collect();
    let (first_us, first_hits) = time(&first, passes, |rc| {
        rc.intersect_objects(arena, root, true).len()
    });
    println!(
        "| {name:<34} | {old_us:>10.2} | {new_us:>10.2} | {first_us:>10.2} | {old_hits} / {new_hits} / {first_hits} |"
    );
}

fn down_rays(n: usize, half: f32, seed: u64) -> Vec<Raycaster> {
    let mut rng = Lcg(seed);
    (0..n)
        .map(|_| {
            let o = Vector3::new(rng.range(-half, half), rng.range(-half, half), 10.0);
            Raycaster::new(o, Vector3::new(0.0, 0.0, -1.0), 0.0, f32::INFINITY)
        })
        .collect()
}

fn toward_origin_rays(n: usize, spread: f32, dist: f32, seed: u64) -> Vec<Raycaster> {
    let mut rng = Lcg(seed);
    (0..n)
        .map(|_| {
            let o = Vector3::new(
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
            )
            .normalize()
                * dist;
            let target = Vector3::new(
                rng.range(-spread, spread),
                rng.range(-spread, spread),
                rng.range(-spread, spread),
            );
            Raycaster::new(o, (target - o).normalize(), 0.0, f32::INFINITY)
        })
        .collect()
}

fn instanced(bvh: bool) -> (ObjectArena, ObjectId) {
    #[allow(unused_mut)]
    let mut g = BoxGeometry::new(0.5, 0.5, 0.5);
    #[cfg(feature = "mesh-bvh")]
    if bvh {
        g.compute_bounds_tree(Default::default());
    }
    let _ = bvh;
    let mut im = InstancedMesh::new(g, material(), 10_000);
    for i in 0..10_000 {
        im.set_matrix_at(
            i,
            Matrix4::translation(Vector3::new(
                (i % 100) as f32 - 49.5,
                (i / 100) as f32 - 49.5,
                0.0,
            )),
        );
    }
    let mut arena = ObjectArena::new();
    let root = arena.insert(Object3D::group());
    let id = arena.insert(Object3D::instanced_mesh(im));
    arena.add_child(root, id);
    arena.update_world_matrices(root, Matrix4::identity());
    (arena, root)
}

#[test]
#[ignore]
fn raycast_timings() {
    println!(
        "| scene | legacy µs/ray | now µs/ray | first hit µs/ray | hits legacy / now / first |"
    );
    println!("|---|---|---|---|---|");

    // 1M quads, 2M triangles, with the plane's polygon grouping.
    let grid = || PlaneGeometry::with_segments(1.0, 1.0, 1000, 1000);
    let (arena, root) = scene_with(grid());
    row(
        "1M-quad grid, brute force",
        &arena,
        root,
        &down_rays(10, 0.5, 1),
        2,
    );
    #[cfg(feature = "mesh-bvh")]
    {
        let mut g = grid();
        g.compute_bounds_tree(Default::default());
        let (arena, root) = scene_with(g);
        row(
            "1M-quad grid, BVH",
            &arena,
            root,
            &down_rays(2000, 0.5, 1),
            5,
        );
    }

    // 99,904 triangles.
    let sphere = || SphereGeometry::new(1.0, 224, 224);
    let (arena, root) = scene_with(sphere());
    row(
        "100k-tri sphere, brute force",
        &arena,
        root,
        &toward_origin_rays(50, 0.5, 5.0, 2),
        3,
    );
    #[cfg(feature = "mesh-bvh")]
    {
        let mut g = sphere();
        g.compute_bounds_tree(Default::default());
        let (arena, root) = scene_with(g);
        row(
            "100k-tri sphere, BVH",
            &arena,
            root,
            &toward_origin_rays(5000, 0.5, 5.0, 2),
            5,
        );
    }

    // 1000 small boxes on a 10x10x10 lattice.
    {
        let mut arena = ObjectArena::new();
        let root = arena.insert(Object3D::group());
        for i in 0..1000 {
            let mut o = Object3D::mesh(Mesh::new(BoxGeometry::new(0.5, 0.5, 0.5), material()));
            o.position = Vector3::new(
                (i % 10) as f32 - 4.5,
                ((i / 10) % 10) as f32 - 4.5,
                (i / 100) as f32 - 4.5,
            );
            let id = arena.insert(o);
            arena.add_child(root, id);
        }
        arena.update_world_matrices(root, Matrix4::identity());
        row(
            "1000 boxes",
            &arena,
            root,
            &toward_origin_rays(2000, 4.0, 20.0, 3),
            5,
        );
    }

    // 10k instances of a box on a 100x100 grid. The legacy raycaster did not
    // test instanced meshes at all, so its column is the cost of skipping it.
    let rays = down_rays(500, 50.0, 4);
    let (arena, root) = instanced(false);
    row("10k instances", &arena, root, &rays, 5);
    #[cfg(feature = "mesh-bvh")]
    {
        let (arena, root) = instanced(true);
        row("10k instances, BVH per instance", &arena, root, &rays, 5);
    }
}
