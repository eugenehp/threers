//! Headless render of the OpenSCAD-style `Solid` front end → PNG, so you can
//! *see* it working end-to-end (`--features openscad`).
//!
//! Run: `cargo run --example openscad_render --features openscad [-- out/bracket.png]`

use std::f32::consts::FRAC_PI_2;
use threers::{
    cube, cylinder, linear_extrude, sphere, AmbientLight, Color, DirectionalLight, HeadlessRenderer,
    Mesh, Object3D, PerspectiveCamera, Scene, StandardMaterial, Vector3,
};

fn main() {
    // First non-flag arg = output path; `--float` selects the float kernel
    // instead of the hybrid arrangement+fallback path.
    let use_float = std::env::args().any(|a| a == "--float");
    let out = std::env::args()
        .skip(1)
        .find(|a| !a.starts_with("--"))
        .unwrap_or_else(|| "out/openscad_bracket.png".into());

    // --- Build the bracket with the Solid builder (same as openscad_bracket) ---
    let plate = cube([40.0, 24.0, 4.0]);
    let drill = cylinder(20.0, 2.5).rotate_x(FRAC_PI_2);
    let boss = sphere(5.0).translate([0.0, 0.0, 2.0]);
    let gusset = linear_extrude(3.0, &[[-6.0, 0.0], [6.0, 0.0], [3.0, 8.0], [-3.0, 8.0]])
        .rotate_x(FRAC_PI_2)
        .translate([0.0, 12.0, -2.0]);
    let part = plate
        .difference(drill.clone().translate([-12.0, 0.0, 0.0]))
        .difference(drill.translate([12.0, 0.0, 0.0]))
        .union(boss)
        .union(gusset);
    let geometry = if use_float { part.to_geometry() } else { part.to_geometry_exact() };
    println!("kernel: {}", if use_float { "float CsgEvaluator" } else { "hybrid exact + fallback" });

    // --- Scene ---
    let (w, h) = (900u32, 600u32);
    let mut hr = match HeadlessRenderer::builder().size(w, h).supersample(2).build() {
        Ok(hr) => hr,
        Err(e) => {
            eprintln!("headless renderer unavailable ({e}) — needs a GPU adapter (Metal/Vulkan/DX12).");
            std::process::exit(2);
        }
    };
    let (rw, rh) = hr.render_size();

    let mut scene = Scene::new();
    scene.background = Color::new(0.07, 0.08, 0.11);
    scene.add_light(AmbientLight::new(Color::WHITE, 0.35));
    scene.add_light(
        DirectionalLight::new(Color::WHITE, 2.4).with_direction(Vector3::new(-0.4, -0.8, -0.5).normalize()),
    );

    let mut mat = StandardMaterial::new(Color::new(0.24, 0.62, 0.95));
    mat.metalness = 0.2;
    mat.roughness = 0.4;
    scene.add(Object3D::mesh(Mesh::new(geometry, mat.into())));

    let mut cam = PerspectiveCamera::new(45.0, w as f32 / h as f32, 0.1, 500.0);
    cam.position = Vector3::new(38.0, 34.0, 46.0);
    cam.look_at(Vector3::new(0.0, 2.0, 0.0));

    let rgba = hr.render_to_rgba(&mut scene, &cam);
    write_png(&out, rw, rh, &rgba).expect("write png");
    println!("wrote {out} ({rw}x{rh})");
}

// --- Minimal PNG writer (RGBA8, stored/zlib — no external deps) ---

fn write_png(path: &str, w: u32, h: u32, rgba: &[u8]) -> std::io::Result<()> {
    let mut raw = Vec::with_capacity((w * h * 4 + h) as usize);
    for y in 0..h as usize {
        raw.push(0); // filter: none
        let row = &rgba[y * w as usize * 4..(y + 1) * w as usize * 4];
        raw.extend_from_slice(row);
    }
    let mut png = Vec::new();
    png.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]); // 8-bit, RGBA, deflate, no filter, no interlace
    chunk(&mut png, b"IHDR", &ihdr);
    chunk(&mut png, b"IDAT", &zlib_store(&raw));
    chunk(&mut png, b"IEND", &[]);
    std::fs::write(path, png)
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc_in = kind.to_vec();
    crc_in.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_in).to_be_bytes());
}

fn zlib_store(data: &[u8]) -> Vec<u8> {
    let mut z = vec![0x78, 0x01]; // zlib header, no compression
    for (i, block) in data.chunks(65535).enumerate() {
        let last = (i + 1) * 65535 >= data.len();
        z.push(if last { 1 } else { 0 }); // BFINAL + BTYPE=00 (stored)
        let len = block.len() as u16;
        z.extend_from_slice(&len.to_le_bytes());
        z.extend_from_slice(&(!len).to_le_bytes());
        z.extend_from_slice(block);
    }
    z.extend_from_slice(&adler32(data).to_be_bytes());
    z
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
        }
    }
    !crc
}

fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}
