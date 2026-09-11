//! Headless contact sheet of the spacecraft material presets and the extended
//! PBR layers, rendered to a PNG.
//!
//! No window, no downloaded assets — everything (environment, normal maps,
//! textures) is generated procedurally, so this runs from a fresh clone and is
//! the fastest way to see what each layer actually does.
//!
//! ```text
//! cargo run --example material_chart               # → material_chart.png
//! cargo run --example material_chart -- --out x.png
//! ```
//!
//! Rows, top to bottom:
//!   1. The presets: gold foil, silver foil, aluminium, brushed aluminium,
//!      titanium, solar cell, white paint, black kapton.
//!   2. Iridescence thickness sweep (150 → 650 nm) over a titanium base.
//!   3. Anisotropy rotation sweep (0 → π) on brushed aluminium.
//!   4. Roughness sweep on gold, from mirror to matte.

use std::f32::consts::PI;
use std::path::PathBuf;

use threers::materials::presets;
use threers::{
    AmbientLight, DirectionalLight, HeadlessRenderer, Material, Mesh, Object3D, PerspectiveCamera,
    PhysicalMaterial, Scene, SphereGeometry, Vector3,
};

// Brings in `Arc`, `Color`, the procedural textures, and `orbital_environment`.
include!("spacecraft_common.inc");

/// One rendered tile per material.
const TILE: u32 = 220;
const COLS: usize = 8;

fn out_path(default: &str) -> PathBuf {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--out" {
            return PathBuf::from(args.next().expect("--out needs a path"));
        }
    }
    PathBuf::from(default)
}

fn main() {
    let out = out_path("out/material_chart.png");

    // Render at SS× and let the GPU average it down: `render_to_rgba_resolved`
    // returns the frame at the configured size, not at `render_size()`.
    const SS: u32 = 3;
    let mut renderer = HeadlessRenderer::builder()
        .size(TILE, TILE)
        .supersample(SS)
        // Rgba8Unorm, NOT the Rgba8UnormSrgb default. The mesh shader does its
        // own linear→sRGB encode, so an sRGB target encodes a second time and
        // the result is badly washed out — mid-greys land ~2.5× too bright.
        .color_format(wgpu::TextureFormat::Rgba8Unorm)
        .build()
        .expect("headless renderer (is a GPU adapter available?)");

    // The environment now carries a sun ~600x brighter than white, so linear
    // output would clip it (and everything near it) to flat white. ACES rolls
    // that off and — crucially — preserves hue while doing so, which is what
    // keeps a blown-out gold highlight looking like gold.
    renderer
        .renderer()
        .set_tone_mapping(threers::ToneMapping::AcesFilmic, 1.0);
    let (rw, rh) = renderer.render_size();
    assert_eq!((rw, rh), (TILE * SS, TILE * SS));

    // Shared across every tile: one environment, one crinkle map.
    let env = orbital_environment(SUN_DIR);
    let crinkle = crinkle_normal_map(512, 9.0, 2.2);
    let cells = solar_cell_map(512, 6);

    // --- Build the material list, row by row -------------------------------
    let mut tiles: Vec<(String, PhysicalMaterial)> = Vec::new();

    // Row 1 — the presets.
    let mut gold = presets::gold_foil();
    gold.normal_map = Some(crinkle.clone());
    tiles.push(("gold foil".into(), gold));

    let mut silver = presets::silver_foil();
    silver.normal_map = Some(crinkle.clone());
    tiles.push(("silver foil".into(), silver));

    tiles.push(("aluminium".into(), presets::aluminum()));
    tiles.push(("brushed al".into(), presets::brushed_aluminum(0.0)));
    tiles.push(("titanium".into(), presets::titanium()));

    let mut panel = presets::solar_cell();
    panel.map = Some(cells.clone());
    tiles.push(("solar cell".into(), panel));

    tiles.push(("white paint".into(), presets::white_thermal_paint()));
    tiles.push(("black kapton".into(), presets::black_kapton()));

    // Row 2 — iridescence thickness sweep. Same base metal, only the film
    // thickness changes, and it walks the whole interference rainbow.
    for i in 0..COLS {
        let nm = 150.0 + (i as f32) * (500.0 / (COLS - 1) as f32);
        tiles.push((format!("irid {nm:.0}nm"), presets::anodized_titanium(nm)));
    }

    // Row 3 — anisotropy rotation. Identical material, rotated streak; this is
    // the axis that was previously stored-but-ignored.
    for i in 0..COLS {
        let rot = (i as f32) * (PI / (COLS - 1) as f32);
        tiles.push((
            format!("aniso {:.2}rad", rot),
            presets::brushed_aluminum(rot),
        ));
    }

    // Row 4 — roughness sweep on gold, mirror through matte. Needs the PMREM
    // chain to be doing its job; without it every tile would look identical.
    for i in 0..COLS {
        let r = 0.02 + (i as f32) * (0.75 / (COLS - 1) as f32);
        let mut m = presets::gold_foil().with_roughness(r);
        m.normal_map = Some(crinkle.clone());
        tiles.push((format!("gold r={r:.2}"), m));
    }

    let rows = tiles.len().div_ceil(COLS);
    let sheet_w = TILE * COLS as u32;
    let sheet_h = TILE * rows as u32;
    let mut sheet = vec![0u8; (sheet_w * sheet_h * 4) as usize];

    println!("rendering {} tiles ({COLS}×{rows})…", tiles.len());

    for (idx, (label, mat)) in tiles.iter().enumerate() {
        let rgba = render_tile(&mut renderer, &env, mat.clone());

        let cx = (idx % COLS) as u32 * TILE;
        let cy = (idx / COLS) as u32 * TILE;
        for y in 0..TILE {
            let src = (y * TILE * 4) as usize;
            let dst = (((cy + y) * sheet_w + cx) * 4) as usize;
            sheet[dst..dst + (TILE * 4) as usize]
                .copy_from_slice(&rgba[src..src + (TILE * 4) as usize]);
        }
        println!("  [{:2}/{}] {label}", idx + 1, tiles.len());
    }

    let png = threers::encode_png(sheet_w, sheet_h, &sheet);
    if let Some(dir) = std::path::Path::new(&out).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(&out, png).expect("write png");
    println!("wrote {} ({sheet_w}×{sheet_h})", out.display());
}

/// Render a single sphere with `mat` against the orbital environment.
fn render_tile(
    renderer: &mut HeadlessRenderer,
    env: &Arc<threers::CubeTexture>,
    mat: PhysicalMaterial,
) -> Vec<u8> {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x05060a);
    scene.environment = Some(env.clone());

    // A dim ambient only — the environment is doing the real lighting work.
    scene.add_light(AmbientLight::new(Color::from_hex(0x223044), 0.06));

    // The environment's own sun disc now supplies the specular highlight, so
    // this light exists mainly to give the surface a defined lit side (and, in
    // scenes with geometry, to cast shadows). Leaving it at its old "fake the
    // sun" intensity would double the highlight.
    let mut sun = Object3D::light(DirectionalLight::new(Color::from_hex(0xfff6e8), 0.55));
    sun.position = Vector3::new(SUN_DIR[0], SUN_DIR[1], SUN_DIR[2]) * 10.0;
    scene.add(sun);

    // Earth bounce from below, so the shadow side isn't dead black.
    let mut bounce = Object3D::light(DirectionalLight::new(earth_bounce(), 0.3));
    bounce.position = Vector3::new(-0.3, -1.0, 0.2) * 10.0;
    scene.add(bounce);

    scene.add(Object3D::mesh(Mesh::new(
        SphereGeometry::new(1.0, 96, 48),
        Material::Physical(mat),
    )));

    let mut camera = PerspectiveCamera::new(32.0, 1.0, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 0.35, 4.2);
    camera.look_at(Vector3::ZERO);

    scene.update_world();
    renderer.render_to_rgba_resolved(&mut scene, &camera)
}
