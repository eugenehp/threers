//! Detailed **NEMA 17 stepper motor** — a single-part assembly shown on its own.
//!
//! Uses [`threers::build_nema17`] (end-caps, inset lamination stack, corner
//! tie-rods, boss, 5 mm shaft, mounting holes, wiring connector + leads), each a
//! separate colored mesh, rendered headless to a PNG.
//!
//! Run: `cargo run --example nema17 --features openscad -- [out.png]`

use threers::{
    build_nema17, AmbientLight, BufferGeometry, Color, DirectionalLight, HeadlessRenderer, Mesh,
    Object3D, PerspectiveCamera, Scene, StandardMaterial, Vector3,
};

fn main() {
    let out = std::env::args().skip(1).find(|a| !a.starts_with("--")).unwrap_or_else(|| "out/nema17.png".into());

    let parts = build_nema17();
    let tris: usize = parts.iter().map(|(g, _)| part_tris(g)).sum();
    println!("NEMA 17 · {} parts · {tris} triangles", parts.len());

    let mut scene = Scene::new();
    scene.background = Color::new(0.08, 0.09, 0.12);
    scene.add_light(AmbientLight::new(Color::WHITE, 0.45));
    scene.add_light(DirectionalLight::new(Color::WHITE, 2.8).with_direction(Vector3::new(-0.5, -0.7, -0.55).normalize()));
    scene.add_light(DirectionalLight::new(Color::WHITE, 0.9).with_direction(Vector3::new(0.6, 0.5, -0.2).normalize()));

    let (mut mn, mut mx) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    for (geom, color) in &parts {
        expand_bounds(geom, &mut mn, &mut mx);
        let mut mat = StandardMaterial::new(Color::new(color[0], color[1], color[2]));
        mat.metalness = 0.4;
        mat.roughness = 0.38;
        scene.add(Object3D::mesh(Mesh::new(geom.clone(), mat.into())));
    }
    let center = Vector3::new((mn[0] + mx[0]) / 2.0, (mn[1] + mx[1]) / 2.0, (mn[2] + mx[2]) / 2.0);
    let radius = ((mx[0] - mn[0]).powi(2) + (mx[1] - mn[1]).powi(2) + (mx[2] - mn[2]).powi(2)).sqrt() / 2.0;

    let (w, h) = (1000u32, 1000u32);
    let mut hr = match HeadlessRenderer::builder().size(w, h).supersample(1).build() {
        Ok(hr) => hr,
        Err(e) => {
            eprintln!("headless renderer unavailable ({e}) — needs a GPU adapter (Metal/Vulkan/DX12).");
            std::process::exit(2);
        }
    };
    hr.set_msaa(4); // hardware 4× MSAA (instead of supersampling)
    let (rw, rh) = hr.render_size();

    let dist = radius * 2.4;
    let mut cam = PerspectiveCamera::new(40.0, w as f32 / h as f32, 0.5, dist * 8.0 + 200.0);
    cam.up = Vector3::new(0.0, 0.0, 1.0);
    cam.position = Vector3::new(center.x + dist * 0.9, center.y - dist * 0.85, center.z + dist * 0.45);
    cam.look_at(center);

    let rgba = hr.render_to_rgba(&mut scene, &cam);
    if let Some(parent) = std::path::Path::new(&out).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    write_png(&out, rw, rh, &rgba).expect("write png");
    println!("wrote {out} ({rw}×{rh})");
}

fn part_tris(g: &BufferGeometry) -> usize {
    match &g.index {
        Some(i) => i.len() / 3,
        None => g.get_attribute("position").map(|a| a.count() / 3).unwrap_or(0),
    }
}
fn expand_bounds(g: &BufferGeometry, mn: &mut [f32; 3], mx: &mut [f32; 3]) {
    if let Some(it) = g.positions() {
        for v in it {
            for (k, c) in [v.x, v.y, v.z].into_iter().enumerate() {
                mn[k] = mn[k].min(c);
                mx[k] = mx[k].max(c);
            }
        }
    }
}

// --- Minimal PNG writer (RGBA8, stored/zlib — no external deps) ---
fn write_png(path: &str, w: u32, h: u32, rgba: &[u8]) -> std::io::Result<()> {
    let mut raw = Vec::with_capacity((w * h * 4 + h) as usize);
    for y in 0..h as usize {
        raw.push(0);
        raw.extend_from_slice(&rgba[y * w as usize * 4..(y + 1) * w as usize * 4]);
    }
    let mut png = Vec::new();
    png.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
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
    let mut z = vec![0x78, 0x01];
    for (i, block) in data.chunks(65535).enumerate() {
        let last = (i + 1) * 65535 >= data.len();
        z.push(if last { 1 } else { 0 });
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
