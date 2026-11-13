//! What a data-driven `ShaderMaterial` costs a frame, measured.
//!
//!     cargo run --release --example shader_material_bench
//!
//! The shape is a city map: 200,000 small quads ("buildings"), the building
//! index in `uv.x`, eight floats of per-building attributes in storage slot 0
//! (6.4 MB), and a uniform that changes every frame — a year slider, a hover
//! id, a clock. Two scenes:
//!
//! * **one mesh** — every building in one draw.
//! * **300 tiles** — the same buildings split across 300 meshes that all share
//!   one material, which is how a tiled map is actually loaded.
//!
//! and two ways of changing the uniform:
//!
//! * **copy** — `Arc::make_mut` the material and write `data`, handing the copy
//!   to every tile. The only way before materials could change in place, and
//!   still what code that writes the field does.
//! * **in place** — `ShaderMaterial::set_data_floats` on the one shared
//!   material.
//!
//! Each frame writes the uniform, renders and waits for the GPU, so the number
//! is CPU submission plus GPU time. Knobs:
//!
//! * `BENCH_SIZE=WxH` — target size (default 1280x720).
//! * `BENCH_FRAMES=<n>` — frames per measurement (default 100).

use std::sync::Arc;
use std::time::Instant;

use threers::{
    BufferAttribute, BufferGeometry, Color, HeadlessRenderer, Material, Mesh, Object3D, ObjectKind,
    PerspectiveCamera, Scene, ShaderMaterial, Vector3,
};

const BUILDINGS: usize = 200_000;
const FLOATS_PER_BUILDING: usize = 8;
const COLS: usize = 448;
const TILES: usize = 300;

const FRAGMENT: &str = "
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let b = u32(in.uv.x + 0.5);
    let built = s0(b * 8u);
    let height = s0(b * 8u + 1u);
    let year = u_data.data[0].x;
    let hover = u32(u_data.data[0].y);
    var c = vec3<f32>(0.2, 0.3, 0.5) + vec3<f32>(height * 0.02);
    if (built > year) { c = vec3<f32>(0.08); }
    if (b == hover) { c = vec3<f32>(1.0, 0.6, 0.1); }
    return vec4<f32>(framebuffer_encode(apply_tone_mapping(c)), 1.0);
}
";

/// Buildings `range` as quads on the XY plane, `uv.x` = building index.
fn quads(range: std::ops::Range<usize>) -> BufferGeometry {
    let n = range.len();
    let mut pos = Vec::with_capacity(n * 12);
    let mut nrm = Vec::with_capacity(n * 12);
    let mut uv = Vec::with_capacity(n * 8);
    let mut idx = Vec::with_capacity(n * 6);
    for (k, b) in range.enumerate() {
        let (x, y) = ((b % COLS) as f32, (b / COLS) as f32);
        for (dx, dy) in [(0.0, 0.0), (0.8, 0.0), (0.8, 0.8), (0.0, 0.8)] {
            pos.extend([x + dx - COLS as f32 * 0.5, y + dy - COLS as f32 * 0.5, 0.0]);
            nrm.extend([0.0, 0.0, 1.0]);
            uv.extend([b as f32, 0.0]);
        }
        let v = (k * 4) as u32;
        idx.extend([v, v + 1, v + 2, v, v + 2, v + 3]);
    }
    let mut g = BufferGeometry::new();
    g.set_attribute("position", BufferAttribute::new(pos, 3));
    g.set_attribute("normal", BufferAttribute::new(nrm, 3));
    g.set_attribute("uv", BufferAttribute::new(uv, 2));
    g.set_index(idx);
    g.compute_bounding_box();
    g.compute_bounding_sphere();
    g
}

fn attributes() -> Vec<f32> {
    let mut s = vec![0.0f32; BUILDINGS * FLOATS_PER_BUILDING];
    for b in 0..BUILDINGS {
        s[b * 8] = 1900.0 + (b % 120) as f32; // year built
        s[b * 8 + 1] = (b % 37) as f32; // height
    }
    s
}

fn main() {
    let (w, h) = size();
    let frames = env("BENCH_FRAMES").unwrap_or(100usize);
    let Ok(mut hr) = HeadlessRenderer::builder()
        .size(w, h)
        .color_format(threers::wgpu::TextureFormat::Rgba8Unorm)
        .build()
    else {
        eprintln!("no GPU adapter");
        return;
    };
    let material = ShaderMaterial::new(FRAGMENT)
        .with_data(vec![[1950.0, 0.0, 0.0, 0.0]])
        .with_storage0(attributes());
    let mut cam = PerspectiveCamera::new(60.0, w as f32 / h as f32, 0.1, 2000.0);
    cam.position = Vector3::new(0.0, 0.0, 420.0);
    cam.look_at(Vector3::ZERO);
    println!(
        "{BUILDINGS} buildings, storage0 = {} floats, {w}x{h}, {frames} frames\n",
        BUILDINGS * FLOATS_PER_BUILDING
    );

    for (label, tiles, in_place) in [
        ("one mesh, copy", 1usize, false),
        ("one mesh, in place", 1, true),
        ("300 tiles, copy", TILES, false),
        ("300 tiles, in place", TILES, true),
    ] {
        let mut scene = Scene::new();
        scene.background = Color::from_hex(0x000000);
        let shared = Arc::new(Material::Shader(material.clone()));
        let per = BUILDINGS.div_ceil(tiles);
        let mut ids = Vec::new();
        for t in 0..tiles {
            let range = t * per..((t + 1) * per).min(BUILDINGS);
            let mut mesh = Mesh::new(quads(range), Material::Basic(Default::default()));
            mesh.material = shared.clone();
            ids.push(scene.add(Object3D::mesh(mesh)));
        }
        scene.update_world();
        // Held only for the in-place writes: an extra reference would make the
        // copy path copy even with one mesh.
        let shared = in_place.then_some(shared);

        let frame = |scene: &mut Scene, hr: &mut HeadlessRenderer, i: usize| {
            let year = 1900.0 + (i % 120) as f32;
            if let Some(Material::Shader(sm)) = shared.as_deref() {
                sm.set_data_floats(0, &[year]);
            } else {
                set_year(scene, &ids, year);
            }
            hr.render(scene, &cam);
            let _ = hr
                .device()
                .poll(threers::wgpu::PollType::wait_indefinitely());
        };
        for i in 0..10 {
            frame(&mut scene, &mut hr, i);
        }
        let t0 = Instant::now();
        for i in 0..frames {
            frame(&mut scene, &mut hr, i);
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3 / frames as f64;
        println!("{label:24} {ms:8.2} ms/frame");
    }
}

/// Move the year slider: the per-frame uniform write.
///
/// A material behind an `Arc` has to be copied to change, and every tile
/// sharing it handed the copy, so the first mesh's is made unique and the rest
/// pointed at it. With one mesh that copy never happens.
fn set_year(scene: &mut Scene, ids: &[threers::ObjectId], year: f32) {
    let mut next: Option<Arc<Material>> = None;
    for &id in ids {
        let Some(obj) = scene.get_mut(id) else {
            continue;
        };
        let ObjectKind::Mesh(mesh) = &mut obj.kind else {
            continue;
        };
        match &next {
            Some(m) => mesh.material = m.clone(),
            None => {
                if let Material::Shader(sm) = Arc::make_mut(&mut mesh.material) {
                    sm.data[0][0] = year;
                }
                next = Some(mesh.material.clone());
            }
        }
    }
}

fn size() -> (u32, u32) {
    std::env::var("BENCH_SIZE")
        .ok()
        .and_then(|s| {
            let (w, h) = s.split_once('x')?;
            Some((w.parse().ok()?, h.parse().ok()?))
        })
        .unwrap_or((1280, 720))
}

fn env<T: std::str::FromStr>(name: &str) -> Option<T> {
    std::env::var(name).ok()?.parse().ok()
}
