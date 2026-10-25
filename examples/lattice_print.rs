//! Getting a lattice ready to print: is it closed, what needs support, and how
//! far can it be simplified — rendered headless to a PNG.
//!
//! A Kelvin foam is built, checked with `utils::mesh_check` and simplified with
//! `utils::simplify`; both versions are drawn faceted, every downward face
//! coloured by how badly it overhangs: red where a 45° process needs support
//! under it, amber within ten degrees of that, grey where it holds itself up.
//! The left is as built, the right after simplification — the same part, the
//! same overhangs, in a fraction of the triangles.
//!
//! ```text
//! cargo run --release --example lattice_print                     # out/lattice_print.png
//! cargo run --release --example lattice_print --features parallel
//! ```

use threers::utils::mesh_check::{edge_report, print_report, split_pieces, triangle_quality};
use threers::utils::simplify::{simplify, SimplifyOptions};
use threers::{
    encode_png, AmbientLight, BufferAttribute, BufferGeometry, Color, DirectionalLight, Euler,
    HeadlessRenderer, Lattice, LatticeKind, Mesh, Object3D, PerspectiveCamera, Scene, SeedLayout,
    StandardMaterial, Stochastic, Vector3,
};

const PART: f32 = 20.0;
const OVERHANG: f32 = 45.0;
const FOV: f32 = 30.0;

fn main() {
    let out = std::env::args()
        .skip(1)
        .find(|a| !a.starts_with("--"))
        .unwrap_or_else(|| "out/lattice_print.png".into());

    let part = Lattice::new(LatticeKind::Stochastic(Stochastic::Voronoi))
        .size(Vector3::new(PART, PART, PART))
        .cells([3, 3, 3])
        .seed_layout(SeedLayout::Kelvin)
        .resolution(16)
        .fit_relative_density(0.15)
        .drop_specks(true)
        .build();
    let up = Vector3::new(0.0, 0.0, 1.0);
    report("as built", &part, up);

    // Well under the struts, which are about a millimetre and a half across.
    let (light, done) = simplify(
        &part,
        SimplifyOptions {
            target_triangles: 0,
            max_error: 0.03,
        },
    );
    report("simplified", &light, up);
    println!(
        "simplify: {} -> {} triangles ({} collapses, {} refused as tears or folds)",
        done.triangles_before, done.triangles_after, done.collapses, done.refused
    );

    render(&out, vec![overhangs(&part, up), overhangs(&light, up)]);
}

/// What a slicer would want to know.
fn report(name: &str, g: &BufferGeometry, up: Vector3) {
    let edges = edge_report(g);
    let quality = triangle_quality(g);
    let pieces = split_pieces(g);
    let print = print_report(g, up, OVERHANG, 512);
    let volume: f32 = pieces.iter().map(|p| p.volume).sum();
    println!(
        "{name:<10} closed {:<5} {:>7} triangles, {:.1} % needles, worst angles {:.1}°, \
         {} pieces, {:.0} mm³",
        edges.is_closed(),
        quality.triangles,
        quality.needle_fraction * 100.0,
        quality.min_angle_p5,
        pieces.len(),
        volume,
    );
    println!(
        "{:<10} height {:.1} mm, support {:.0} mm² over {:.0} mm² of plate, plate contact {:.0} mm²",
        "", print.height, print.support_area, print.support_footprint, print.plate_contact
    );
}

/// The mesh unwelded into flat facets, each coloured by its overhang.
fn overhangs(g: &BufferGeometry, up: Vector3) -> Object3D {
    let p = &g.get_attribute("position").expect("positions").array;
    let index = g.index.as_ref().expect("indexed");
    let (mut pos, mut nrm, mut col) = (Vec::new(), Vec::new(), Vec::new());
    let limit = OVERHANG.to_radians();
    for t in index.chunks_exact(3) {
        let at = |i: u32| {
            let i = i as usize * 3;
            Vector3::new(p[i], p[i + 1], p[i + 2])
        };
        let v = [at(t[0]), at(t[1]), at(t[2])];
        let n = (v[1] - v[0]).cross(v[2] - v[0]).normalize();
        // Tilt from horizontal of a downward face: 0 is a flat ceiling.
        let down = -n.dot(up);
        let tilt = down.clamp(0.0, 1.0).acos();
        let rgb = if down <= 0.0 {
            [0.72, 0.74, 0.78]
        } else if tilt < limit {
            [0.88, 0.22, 0.18]
        } else if tilt < limit + 10f32.to_radians() {
            [0.95, 0.66, 0.2]
        } else {
            [0.72, 0.74, 0.78]
        };
        for corner in v {
            pos.extend_from_slice(&[corner.x, corner.y, corner.z]);
            nrm.extend_from_slice(&[n.x, n.y, n.z]);
            col.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 1.0]);
        }
    }
    let mut facets = BufferGeometry::new();
    let count = pos.len() as u32 / 3;
    facets.set_attribute("position", BufferAttribute::new(pos, 3));
    facets.set_attribute("normal", BufferAttribute::new(nrm, 3));
    facets.set_attribute("color", BufferAttribute::new(col, 4));
    facets.set_index((0..count).collect());
    let mut material = StandardMaterial::new(Color::WHITE);
    material.metalness = 0.05;
    material.roughness = 0.55;
    Object3D::mesh(Mesh::new(facets, material.into()))
}

fn render(out: &str, tiles: Vec<Object3D>) {
    let (w, h) = (1600u32, 800u32);
    let mut renderer = match HeadlessRenderer::builder()
        .size(w, h)
        .supersample(2)
        .build()
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("headless renderer unavailable ({e}) — needs a GPU adapter.");
            std::process::exit(2);
        }
    };
    let (rw, rh) = renderer.render_size();
    let mut scene = Scene::new();
    scene.background = Color::new(0.06, 0.07, 0.09);
    scene.add_light(AmbientLight::new(Color::WHITE, 0.5));
    scene.add_light(
        DirectionalLight::new(Color::WHITE, 2.4)
            .with_direction(Vector3::new(-0.35, -0.75, -0.55).normalize()),
    );
    let pitch = PART * 1.5;
    let columns = tiles.len();
    for (i, mut mesh) in tiles.into_iter().enumerate() {
        mesh.position = Vector3::new((i as f32 - (columns - 1) as f32 * 0.5) * pitch, 0.0, 0.0);
        // Build direction (+z) tipped away from the viewer, so it is the
        // downward faces — the ones that need holding up — that face the
        // camera: a view from below and to the side.
        mesh.quaternion = Euler::new(2.2, 0.45, 0.0).to_quaternion();
        scene.add(mesh);
    }
    let aspect = w as f32 / h as f32;
    let half_angle = (FOV * 0.5).to_radians().tan();
    let content = (columns as f32 * pitch, pitch);
    let distance =
        (content.1 / (2.0 * half_angle)).max(content.0 / (2.0 * half_angle * aspect)) * 1.05;
    let mut camera = PerspectiveCamera::new(FOV, aspect, 1.0, distance * 3.0);
    camera.position = Vector3::new(0.0, 0.0, distance);
    camera.look_at(Vector3::ZERO);
    let rgba = renderer.render_to_rgba(&mut scene, &camera);
    if let Some(dir) = std::path::Path::new(out).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(out, encode_png(rw, rh, &rgba)).expect("write png");
    println!("wrote {out} ({rw}x{rh})");
}
