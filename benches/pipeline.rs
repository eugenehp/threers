//! What the CPU side of a frame costs, measured.
//!
//! These exist to catch silent regressions, not to win comparisons. Everything
//! here runs on the CPU and needs no adapter, which is the point: a benchmark
//! that needs a GPU is a benchmark CI will not run, and a benchmark CI does not
//! run is a number nobody is watching.
//!
//! `cargo bench`, or one group with `cargo bench -- pmrem`. Criterion keeps a
//! baseline in `target/criterion` and reports the delta next time, so the
//! workflow is measure, change, measure again.
//!
//! The four groups are the four places a scene actually spends CPU time before
//! anything is drawn:
//!
//! - **geometry** — building the meshes. Called once per shape, but a scene has
//!   hundreds and a lattice has one enormous one.
//! - **pmrem** — prefiltering the environment. Every scene with image-based
//!   lighting pays this at startup, and it is a blur chain over six faces, so
//!   it is quadratic in the face size and easy to make quietly worse.
//! - **loaders** — parsing an asset. The number a user experiences as "how long
//!   until something appears".
//! - **bvh** — building the acceleration structure and casting against it,
//!   which is picking, physics queries and CSG.

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use std::hint::black_box;
use threers::{BoxGeometry, PmremGenerator, SphereGeometry, TorusKnotGeometry};

fn geometry(c: &mut Criterion) {
    let mut g = c.benchmark_group("geometry");
    g.bench_function("sphere 64x32", |b| {
        b.iter(|| black_box(SphereGeometry::new(1.0, 64, 32)))
    });
    g.bench_function("box", |b| {
        b.iter(|| black_box(BoxGeometry::new(1.0, 1.0, 1.0)))
    });
    g.bench_function("torus knot 256x32", |b| {
        b.iter(|| black_box(TorusKnotGeometry::new(1.0, 0.3, 256, 32, 2, 3)))
    });
    // Bounds are recomputed whenever positions change, which for anything
    // animated is every frame.
    let sphere = SphereGeometry::new(1.0, 64, 32);
    g.bench_function("bounding sphere 64x32", |b| {
        b.iter_batched(
            || sphere.clone(),
            |mut s| black_box(s.compute_bounding_sphere()),
            BatchSize::SmallInput,
        )
    });
    g.finish();
}

/// A studio-ish cube map, built once outside the timing loop.
fn test_cube(size: u32) -> threers::CubeTexture {
    let n = size as usize;
    let faces: [Vec<f32>; 6] = std::array::from_fn(|face| {
        let mut out = vec![0.0f32; n * n * 4];
        for y in 0..n {
            for x in 0..n {
                let u = (x as f32 + 0.5) / n as f32;
                let v = (y as f32 + 0.5) / n as f32;
                // A bright patch on one face and a dim gradient elsewhere, so
                // the blur has something with real dynamic range to carry.
                let lit = if face == 2 && (0.3..0.7).contains(&u) && (0.3..0.7).contains(&v) {
                    40.0
                } else {
                    0.05 + 0.2 * v
                };
                let o = (y * n + x) * 4;
                out[o] = lit;
                out[o + 1] = lit * 0.95;
                out[o + 2] = lit * 0.9;
                out[o + 3] = 1.0;
            }
        }
        out
    });
    threers::CubeTexture::new_f32(size, faces)
}

fn pmrem(c: &mut Criterion) {
    let mut g = c.benchmark_group("pmrem");
    // Criterion's default sample count makes 256 take minutes; this is a blur
    // chain over six faces and is meant to be slow.
    g.sample_size(10);
    for size in [64u32, 128, 256] {
        let cube = test_cube(size);
        g.bench_function(format!("generate {size}"), |b| {
            b.iter(|| black_box(PmremGenerator::generate_pmrem(&cube, size)))
        });
    }
    g.finish();
}

fn loaders(c: &mut Criterion) {
    let mut g = c.benchmark_group("loaders");

    // A real Draco file: rANS entropy decode, an edgebreaker replay and the
    // prediction schemes, which is the most computation any loader here does.
    let bunny = include_bytes!("../src/loaders/draco/testdata/bunny.drc");
    g.bench_function("draco bunny", |b| {
        b.iter(|| black_box(threers::loaders::draco::decode_mesh(black_box(bunny))))
    });

    let sphere_glb = include_bytes!("../src/loaders/testdata/sphere_plain.glb");
    g.bench_function("glb sphere", |b| {
        b.iter(|| black_box(threers::loaders::GltfLoader::parse_glb(black_box(sphere_glb))))
    });

    let ktx = include_bytes!("../src/loaders/testdata/ktx/mips.ktx2");
    g.bench_function("ktx2 with mips", |b| {
        b.iter(|| black_box(threers::loaders::ktx2::parse(black_box(ktx))))
    });
    g.finish();
}

#[cfg(feature = "mesh-bvh")]
fn bvh(c: &mut Criterion) {
    use threers::math::{Ray, Vector3};
    use threers::mesh_bvh::{BuildOptions, MeshBvh};

    let mut g = c.benchmark_group("bvh");
    let geom = TorusKnotGeometry::new(1.0, 0.3, 256, 32, 2, 3);
    g.bench_function("build torus knot", |b| {
        b.iter(|| black_box(MeshBvh::build(&geom, BuildOptions::default())))
    });

    let Some(tree) = MeshBvh::build(&geom, BuildOptions::default()) else {
        g.finish();
        return;
    };
    // A fan of rays through the knot, so the walk is not one lucky path.
    let rays: Vec<Ray> = (0..64)
        .map(|i| {
            let a = i as f32 / 64.0 * std::f32::consts::TAU;
            Ray::new(
                Vector3::new(a.cos() * 4.0, (a * 0.7).sin() * 2.0, a.sin() * 4.0),
                Vector3::new(-a.cos(), -(a * 0.7).sin() * 0.5, -a.sin()).normalize(),
            )
        })
        .collect();
    g.bench_function("raycast 64 rays", |b| {
        b.iter(|| {
            for r in &rays {
                black_box(tree.raycast_first(r, 0.0, 100.0, false));
            }
        })
    });
    g.finish();
}

#[cfg(not(feature = "mesh-bvh"))]
fn bvh(_: &mut Criterion) {}

criterion_group!(benches, geometry, pmrem, loaders, bvh);
criterion_main!(benches);
