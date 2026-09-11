//! First-hit raycasting: correctness against brute force, and throughput.
#![cfg(feature = "mesh-bvh")]

use threers::mesh_bvh::{BuildOptions, MeshBvh, SAH};
use threers::{BufferAttribute, BufferGeometry, Ray, Vector3};

/// Assert a ray rate, but only where a rate means something.
///
/// The same traversal runs several times slower without optimisations, and the
/// release checklist runs `cargo test` without `--release` — so asserting a
/// throughput unconditionally fails the suite on every debug run and says
/// nothing about the tree. The rate is still printed either way, and still
/// enforced in the build where a regression would matter.
fn assert_rate(per_sec: f64, floor: f64, what: &str) {
    if cfg!(debug_assertions) {
        println!("debug build — {per_sec:.0} rays/s not asserted against {floor:.0}");
    } else {
        assert!(per_sec > floor, "only {per_sec:.0} rays/s — {what}");
    }
}


/// A deterministic pile of triangles with plenty of empty space in it, which
/// is what a city is and what a BVH is for.
fn scene(n: usize) -> (BufferGeometry, Vec<[Vector3; 3]>) {
    let mut tris = Vec::with_capacity(n);
    let mut h = 0x2545_f491u32;
    let mut f = || {
        h ^= h << 13;
        h ^= h >> 17;
        h ^= h << 5;
        (h >> 8) as f32 / 16_777_216.0
    };
    for _ in 0..n {
        let c = Vector3::new(f() * 200.0 - 100.0, f() * 40.0, f() * 200.0 - 100.0);
        let s = 0.5 + f() * 3.0;
        tris.push([
            c,
            c + Vector3::new(s, f() * s, 0.0),
            c + Vector3::new(0.0, s, f() * s),
        ]);
    }
    let mut pos = Vec::with_capacity(n * 9);
    for t in &tris {
        for v in t {
            pos.extend_from_slice(&[v.x, v.y, v.z]);
        }
    }
    let mut g = BufferGeometry::new();
    g.set_attribute("position", BufferAttribute::new(pos, 3));
    g.set_index((0..n as u32 * 3).collect());
    (g, tris)
}

fn brute_force(tris: &[[Vector3; 3]], ray: &Ray, near: f32, far: f32) -> Option<f32> {
    let mut best: Option<f32> = None;
    for t in tris {
        // Möller-Trumbore, two-sided.
        let (e1, e2) = (t[1] - t[0], t[2] - t[0]);
        let p = ray.direction.cross(e2);
        let det = e1.dot(p);
        if det.abs() < 1e-9 {
            continue;
        }
        let inv = 1.0 / det;
        let tv = ray.origin - t[0];
        let u = tv.dot(p) * inv;
        if !(-1e-6..=1.0 + 1e-6).contains(&u) {
            continue;
        }
        let q = tv.cross(e1);
        let v = ray.direction.dot(q) * inv;
        if v < -1e-6 || u + v > 1.0 + 1e-6 {
            continue;
        }
        let d = e2.dot(q) * inv;
        if d < near || d > far {
            continue;
        }
        if best.is_none_or(|b| d < b) {
            best = Some(d);
        }
    }
    best
}

fn rays(count: usize) -> Vec<Ray> {
    let mut h = 0x9e37_79b9u32;
    let mut f = || {
        h ^= h << 13;
        h ^= h >> 17;
        h ^= h << 5;
        (h >> 8) as f32 / 16_777_216.0
    };
    (0..count)
        .map(|_| Ray {
            origin: Vector3::new(f() * 200.0 - 100.0, f() * 30.0, f() * 200.0 - 100.0),
            direction: Vector3::new(f() * 2.0 - 1.0, f() * 2.0 - 1.0, f() * 2.0 - 1.0)
                .normalize(),
        })
        .collect()
}

/// The accelerated first hit has to agree with testing every triangle.
///
/// Traversal now culls by the distance of the best hit so far and opens the
/// nearer child first. Both are easy to get subtly wrong in a way that only
/// drops hits — never adds them — so a test that merely counts hits would pass
/// a broken one. This compares the distance, ray by ray.
#[test]
fn the_first_hit_is_the_nearest_one() {
    let (geom, tris) = scene(4000);
    let bvh = MeshBvh::build(
        &geom,
        BuildOptions {
            strategy: SAH,
            max_leaf_tris: 8,
            // A query tree: nothing downstream reads its shape, so let a
            // degenerate split fall back to the median rather than collapsing
            // the subtree into a linear scan.
            split_degenerate: true,
            ..Default::default()
        },
    )
    .expect("bvh");
    let mut checked = 0usize;
    let mut hits = 0usize;
    for ray in rays(3000) {
        let want = brute_force(&tris, &ray, 1e-3, 1.0e4);
        let got = bvh.raycast_first(&ray, 1e-3, 1.0e4, false).map(|h| h.distance);
        match (want, got) {
            (None, None) => {}
            (Some(a), Some(b)) => {
                hits += 1;
                assert!(
                    (a - b).abs() < 1e-3,
                    "nearest hit at {a}, bvh returned {b}"
                );
            }
            (a, b) => panic!("brute force {a:?} but bvh {b:?}"),
        }
        checked += 1;
    }
    assert_eq!(checked, 3000);
    // A test where nothing is ever hit would pass whatever the traversal did.
    assert!(hits > 300, "only {hits} of 3000 rays hit anything");
}

/// A ray starting inside the scene must still find its hit.
///
/// This is the case that breaks if entry distance is taken from
/// `Ray::intersect_box`, which returns the *exit* distance when the origin is
/// inside the box — so the root gets culled and the ray reports a miss.
#[test]
fn a_ray_beginning_inside_the_bounds_still_hits() {
    let (geom, tris) = scene(2000);
    let bvh = MeshBvh::build(&geom, BuildOptions::default()).expect("bvh");
    let mut inside = 0usize;
    for ray in rays(1500) {
        let want = brute_force(&tris, &ray, 1e-3, 1.0e4);
        let got = bvh.raycast_first(&ray, 1e-3, 1.0e4, false).map(|h| h.distance);
        if want.is_some() {
            inside += 1;
        }
        assert_eq!(
            want.is_some(),
            got.is_some(),
            "origin inside the root box: brute force {want:?}, bvh {got:?}"
        );
    }
    assert!(inside > 100);
}

/// Throughput, as a floor rather than a number to admire.
///
/// The traversal used to ignore the distance it was given and visit most of
/// the tree, which measured about 60k rays per second per core — slow enough
/// that baking light over a city took a quarter of an hour. This pins that it
/// stays fast; the threshold is deliberately far below what it actually does,
/// so a loaded machine does not fail the build.
#[test]
fn first_hit_queries_are_not_pathologically_slow() {
    let (geom, tris) = scene(20_000);
    let bvh = MeshBvh::build(
        &geom,
        BuildOptions {
            strategy: SAH,
            max_leaf_tris: 8,
            // A query tree: nothing downstream reads its shape, so let a
            // degenerate split fall back to the median rather than collapsing
            // the subtree into a linear scan.
            split_degenerate: true,
            ..Default::default()
        },
    )
    .expect("bvh");
    let batch = rays(20_000);
    let t0 = std::time::Instant::now();
    let mut hits = 0usize;
    for ray in &batch {
        if bvh.raycast_first(ray, 1e-3, 1.0e4, false).is_some() {
            hits += 1;
        }
    }
    let per_sec = batch.len() as f64 / t0.elapsed().as_secs_f64();
    println!(
        "{:.0} first-hit rays/s against {} triangles ({hits} hit)",
        per_sec,
        tris.len()
    );
    assert_rate(
        per_sec,
        300_000.0,
        "traversal has regressed to a near-linear scan",
    );
}

/// A few enormous triangles must not wreck the tree for everything else.
///
/// This is the shape every outdoor scene has: a ground plane or a backdrop
/// orders of magnitude larger than the objects on it. It drags the root bounds
/// out until the split candidates that separate the real geometry all fall
/// into one or two bins, the chosen plane leaves every triangle on one side,
/// and the builder used to respond by collapsing the whole subtree into a
/// single leaf — which is scanned linearly. Baking light over a city measured
/// 6k rays per second per core that way, against ~900k here.
#[test]
fn giant_triangles_do_not_collapse_the_tree() {
    let (dense, _) = scene(20_000);
    let mut pos: Vec<f32> = dense
        .get_attribute("position")
        .expect("position")
        .array
        .clone();
    // Four ground quads reaching far past everything else.
    let r = 4000.0f32;
    for (a, b) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
        pos.extend_from_slice(&[
            0.0, -1.0, 0.0, a * r, -1.0, b * r, a * r, -1.0, -b * r,
        ]);
    }
    let n = pos.len() / 9;
    let mut g = BufferGeometry::new();
    g.set_attribute("position", BufferAttribute::new(pos, 3));
    g.set_index((0..n as u32 * 3).collect());

    let bvh = MeshBvh::build(
        &g,
        BuildOptions {
            strategy: SAH,
            max_leaf_tris: 8,
            // A query tree: nothing downstream reads its shape, so let a
            // degenerate split fall back to the median rather than collapsing
            // the subtree into a linear scan.
            split_degenerate: true,
            ..Default::default()
        },
    )
    .expect("bvh");
    let batch = rays(20_000);
    let t0 = std::time::Instant::now();
    let mut hits = 0usize;
    for ray in &batch {
        if bvh.raycast_first(ray, 1e-3, 1.0e4, false).is_some() {
            hits += 1;
        }
    }
    let per_sec = batch.len() as f64 / t0.elapsed().as_secs_f64();
    println!("{per_sec:.0} rays/s with a ground plane present ({hits} hit)");
    assert_rate(
        per_sec,
        200_000.0,
        "a handful of large triangles collapsed the tree",
    );
}
