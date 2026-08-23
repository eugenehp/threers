//! Export Cornell hostile hold-out planes + weights for the browser demo.
//!
//! ```sh
//! GPU=1 cargo run --release -p threers-probe --example probe_web_export --features generate,metal
//! ```

use std::path::PathBuf;

use threers::raytrace::{RaytraceSettings, RtCamera};
use threers::{encode_png, ToneMapping};
use threers_probe::generate;
use threers_probe::pack::{self, PlaneExtras};

fn main() {
    let frame = 64u32;
    let probe_spp = 8;
    let probe_bounces = 3;
    let reference_spp = 64;
    let mut rt = make_renderer(frame, frame);
    let packed = render_pair(
        &mut rt,
        generate::hostile(0),
        frame,
        probe_spp,
        probe_bounces,
        reference_spp,
        0xC0FF,
    );
    let out_dir = PathBuf::from("web/assets/probe");
    std::fs::create_dir_all(&out_dir).expect("mkdir");
    copy_weights("out/probe_hops.bin", out_dir.join("probe_hops.bin"));
    copy_weights("out/nrc_hops.bin", out_dir.join("nrc_hops.bin"));
    write_f32(out_dir.join("cornell_input.bin"), &packed.input);
    write_f32(out_dir.join("cornell_ref.bin"), &packed.target);
    write_png(out_dir.join("cornell_probe.png"), frame, &packed.input, true);
    write_png(out_dir.join("cornell_ref.png"), frame, &packed.target, false);
    println!(
        "wrote {} (input {} floats, ref {} floats, weights)",
        out_dir.display(),
        packed.input.len(),
        packed.target.len()
    );
}

fn copy_weights(from: &str, to: PathBuf) {
    std::fs::copy(from, &to).unwrap_or_else(|e| {
        panic!("copy {from} → {}: {e}", to.display());
    });
}

fn write_f32(path: PathBuf, data: &[f32]) {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes).expect("write f32 blob");
}

fn write_png(path: PathBuf, frame: u32, planar: &[f32], probe: bool) {
    let side = frame as usize;
    let n = side * side;
    let mut rgba = vec![0u8; n * 4];
    for y in 0..side {
        for x in 0..side {
            let i = y * side + x;
            let o = i * 4;
            rgba[o] = srgb8(threers_probe::expand(planar[i]));
            rgba[o + 1] = srgb8(threers_probe::expand(planar[n + i]));
            rgba[o + 2] = srgb8(threers_probe::expand(planar[2 * n + i]));
            rgba[o + 3] = 255;
        }
    }
    let _ = probe;
    std::fs::write(path, encode_png(frame, frame, &rgba)).expect("png");
}

fn srgb8(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let s = if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (s.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

struct Packed {
    input: Vec<f32>,
    target: Vec<f32>,
}

fn world_positions(
    camera: &threers::PerspectiveCamera,
    depth: &[f32],
    width: usize,
    height: usize,
) -> Vec<[f32; 3]> {
    let rt = RtCamera::new(camera, &RaytraceSettings::default());
    let mut out = vec![[f32::NAN; 3]; width * height];
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            let d = depth[i];
            if !d.is_finite() {
                continue;
            }
            let (origin, dir) = rt.pixel_ray(
                x as u32,
                y as u32,
                width as u32,
                height as u32,
                (0.5, 0.5),
                (0.5, 0.5),
            );
            let hit = origin + dir * d;
            out[i] = [hit.x, hit.y, hit.z];
        }
    }
    out
}

fn render_pair(
    rt: &mut threers::raytrace::RaytraceRenderer,
    built: generate::Built,
    frame: u32,
    probe_spp: u32,
    probe_bounces: u32,
    reference_spp: u32,
    seed: u64,
) -> Packed {
    let threers_probe::generate::Built {
        mut scene,
        camera,
        kind: _,
    } = built;
    rt.set_settings(RaytraceSettings {
        samples_per_pixel: probe_spp.max(1),
        max_bounces: probe_bounces.max(1),
        min_bounces: 1,
        clamp_indirect: 100.0,
        adaptive_threshold: 0.0,
        denoise: false,
        tone_mapping: ToneMapping::None,
        seed: 0x5eed_0000_0000 | seed,
        ..Default::default()
    });
    rt.render(&mut scene, &camera).expect("probe");
    let probe = rt.film().resolve_hdr();
    let albedo = rt.film().resolve_albedo();
    let normal = rt.film().resolve_normal();
    let depth = rt.film().resolve_depth();
    let scale = rt.traced_scene().map(|s| s.scale()).unwrap_or(1.0);
    let world = world_positions(&camera, &depth, frame as usize, frame as usize);

    rt.set_settings(RaytraceSettings {
        samples_per_pixel: reference_spp,
        max_bounces: 5,
        min_bounces: 3,
        clamp_indirect: 100.0,
        adaptive_threshold: 0.0,
        denoise: false,
        tone_mapping: ToneMapping::None,
        seed: 0x5eed_0000_0000 | seed,
        ..Default::default()
    });
    rt.render(&mut scene, &camera).expect("reference");
    let reference = rt.film().resolve_hdr();

    let (input, target) = pack::planes(
        frame as usize,
        frame as usize,
        &probe,
        &albedo,
        &normal,
        &depth,
        &reference,
        scale,
        Some(PlaneExtras { world: &world }),
    )
    .expect("pack");
    Packed { input, target }
}

fn make_renderer(width: u32, height: u32) -> threers::raytrace::RaytraceRenderer {
    if std::env::var("GPU").ok().as_deref() == Some("1") {
        if let Ok(b) = threers::raytrace::gpu::GpuBackend::headless() {
            return threers::raytrace::RaytraceRenderer::with_backend(width, height, Box::new(b));
        }
    }
    threers::raytrace::RaytraceRenderer::new(width, height)
}
