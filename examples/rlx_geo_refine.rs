//! Adaptive Delaunay refinement, against a uniform grid (`--features rlx-geo`).
//!
//! ```text
//! cargo run --release --features rlx-geo --example rlx_geo_refine
//! ```
//!
//! Writes `out/rlx_geo_refine_{adaptive,uniform}.png` and prints the two
//! meshes' error at a matched vertex budget.
//!
//! The refinement loop samples the field, triangulates, finds the triangles
//! that misrepresent it worst, inserts points at their centroids, and goes
//! again. Where the field is flat it stops early; where it folds it keeps
//! going. The uniform grid spends the same number of vertices evenly, most of
//! them on ground that needed none.
//!
//! Inserting at centroids is also the construction most likely to embarrass a
//! floating-point triangulator: it manufactures near-cocircular quadruples by
//! the thousand, and one flipped in-circle test leaves a sliver or a fold in
//! the surface. Exact integer predicates have no such failure mode, which is
//! why this loop needs no tolerance and no clean-up pass.

use threers::rlx::geo;
use threers::{
    encode_png, AmbientLight, Color, DirectionalLight, HeadlessRenderer, Mesh, Object3D,
    PerspectiveCamera, Scene, StandardMaterial, Vector2, Vector3,
};

const EXTENT: f32 = 6.0;
const TARGET_ERROR: f32 = 0.004;
const MAX_POINTS: usize = 3_000;

/// Three sharp ridges over a flat plain: most of the domain needs no vertices
/// at all, and a little of it needs a great many.
fn field(x: f32, z: f32) -> f32 {
    let ridge = |cx: f32, cz: f32, w: f32| {
        let d2 = ((x - cx).powi(2) + (z - cz).powi(2)) / w;
        2.2 * (-d2).exp()
    };
    ridge(-2.2, -1.4, 0.5) + ridge(1.8, 0.6, 0.9) + ridge(0.2, 2.6, 0.3)
}

fn main() {
    std::fs::create_dir_all("out").ok();

    // --- Adaptive ---------------------------------------------------------
    let refined = geo::refine_heightfield(
        field,
        Vector2::new(-EXTENT, -EXTENT),
        Vector2::new(EXTENT, EXTENT),
        TARGET_ERROR,
        MAX_POINTS,
    )
    .expect("refine");
    println!(
        "adaptive: {} points in {} rounds, worst error {:.5}",
        refined.points.len(),
        refined.rounds,
        refined.error
    );

    // --- Uniform, same budget --------------------------------------------
    let side = (refined.points.len() as f32).sqrt().round() as usize;
    let uniform: Vec<Vector3> = (0..side)
        .flat_map(|j| {
            (0..side).map(move |i| {
                let x = -EXTENT + 2.0 * EXTENT * i as f32 / (side - 1).max(1) as f32;
                let z = -EXTENT + 2.0 * EXTENT * j as f32 / (side - 1).max(1) as f32;
                Vector3::new(x, field(x, z), z)
            })
        })
        .collect();
    let uniform_error = worst_error(&uniform);
    println!(
        "uniform:  {} points ({side}×{side}), worst error {uniform_error:.5}",
        uniform.len()
    );
    println!(
        "\nSame budget, {:.1}× the error when spent evenly.",
        uniform_error / refined.error.max(1e-9)
    );

    // --- Look at them -----------------------------------------------------
    let mut headless = match HeadlessRenderer::builder().size(900, 600).build() {
        Ok(h) => h,
        Err(e) => {
            eprintln!("headless renderer unavailable ({e}) — the numbers above still stand.");
            std::process::exit(2);
        }
    };
    shoot(&mut headless, &refined.geometry, "adaptive");
    let uniform_geometry = geo::heightfield_geometry(&uniform).expect("triangulate");
    shoot(&mut headless, &uniform_geometry, "uniform");
}

/// The worst centroid error of a point set's triangulation — the same measure
/// the refinement minimises, so the two numbers are comparable.
fn worst_error(points: &[Vector3]) -> f32 {
    let plane: Vec<Vector2> = points.iter().map(|p| Vector2::new(p.x, p.z)).collect();
    let triangles = geo::delaunay(&plane).expect("triangulate");
    triangles
        .iter()
        .map(|[a, b, c]| {
            let (a, b, c) = (
                points[*a as usize],
                points[*b as usize],
                points[*c as usize],
            );
            let x = (a.x + b.x + c.x) / 3.0;
            let z = (a.z + b.z + c.z) / 3.0;
            (field(x, z) - (a.y + b.y + c.y) / 3.0).abs()
        })
        .fold(0.0, f32::max)
}

fn shoot(headless: &mut HeadlessRenderer, geometry: &threers::core::BufferGeometry, name: &str) {
    let (width, height) = headless.render_size();
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x0d1017);
    scene.add_light(AmbientLight::new(Color::from_hex(0x6a7a95), 0.45));
    scene.add_light(
        DirectionalLight::new(Color::WHITE, 2.6)
            .with_direction(Vector3::new(-0.45, -0.75, -0.35).normalize()),
    );

    let mut material = StandardMaterial::new(Color::from_hex(0x8aa06a));
    material.roughness = 0.8;
    scene.add(Object3D::mesh(Mesh::new(geometry.clone(), material.into())));

    let mut camera = PerspectiveCamera::new(42.0, width as f32 / height as f32, 0.1, 200.0);
    camera.position = Vector3::new(9.0, 7.5, 11.0);
    camera.look_at(Vector3::new(0.0, 0.2, 0.0));

    let rgba = headless.render_to_rgba(&mut scene, &camera);
    let path = format!("out/rlx_geo_refine_{name}.png");
    std::fs::write(&path, encode_png(width, height, &rgba)).expect("write png");
    println!("wrote {path}");
}
