//! Visual A/B of À-Trous widths: fitted albedo gate vs a restored tight one.
//!
//! ```text
//! cargo run --release --example denoise_ab --features raytrace,parallel
//! ```
//!
//! Writes `out/denoise_ab/{scene}_{label}.png` plus a horizontal strip, and
//! prints relative RMS vs a reference. Look at the chequer floor: a loose
//! albedo gate scores well on RMS and softens texture.

use std::f32::consts::FRAC_PI_2;
use std::sync::Arc;

use threers::raytrace::{
    denoise, DenoiseExample, DenoiseGuides, DenoiseParams, RaytraceRenderer, RaytraceSettings,
};
use threers::{
    encode_png, BoxGeometry, Color, Material, Mesh, Object3D, PerspectiveCamera, PlaneGeometry,
    Scene, SphereGeometry, StandardMaterial, Texture, TextureFormat, TextureWrap, ToneMapping,
    Vector2, Vector3,
};

const SIZE: u32 = 384;
const NOISY: u32 = 32;
const REFERENCE: u32 = 512;

fn main() {
    let out = std::path::Path::new("out/denoise_ab");
    let _ = std::fs::create_dir_all(out);

    let fitted = DenoiseParams::default();
    let tight_albedo = DenoiseParams {
        sigma_albedo: 0.252,
        ..fitted
    };
    let mid_albedo = DenoiseParams {
        sigma_albedo: 1.0,
        ..fitted
    };

    let variants = [
        ("fitted", fitted),
        ("albedo_1", mid_albedo),
        ("albedo_0.25", tight_albedo),
    ];

    for (scene_name, build) in [
        ("mirror_chequer", mirror_over_chequer as fn() -> _),
        ("area_light", area_light as fn() -> _),
    ] {
        println!("rendering {scene_name}…");
        let pair = render_pair(build);
        let example = pair.example();

        let mut strip_w = SIZE;
        let mut strip = pair.beauty(&pair.noisy);
        let mut labels = vec!["raw".to_string()];

        println!(
            "  {scene_name:<16} raw RMS {:>7.4}",
            example.relative_rms(&pair.noisy)
        );

        for (label, params) in &variants {
            let cleaned = denoise(SIZE, SIZE, &pair.noisy, &example.guides, params);
            let rms = example.relative_rms(&cleaned);
            println!("  {scene_name:<16} {label:<12} RMS {rms:>7.4}");
            let rgba = pair.beauty(&cleaned);
            write_png(
                out.join(format!("{scene_name}_{label}.png")),
                SIZE,
                SIZE,
                &rgba,
            );
            strip.extend_from_slice(&rgba);
            strip_w += SIZE;
            labels.push((*label).to_string());
        }

        write_png(
            out.join(format!("{scene_name}_raw.png")),
            SIZE,
            SIZE,
            &pair.beauty(&pair.noisy),
        );
        let strip_img = hstack(SIZE, SIZE, &strip, strip_w);
        write_png(
            out.join(format!("{scene_name}_strip.png")),
            strip_w,
            SIZE,
            &strip_img,
        );
        println!("  strip labels: {}", labels.join(" | "));
    }

    println!("wrote PNGs under {}", out.display());
}

struct Pair {
    noisy: Vec<f32>,
    reference: Vec<f32>,
    albedo: Vec<[f32; 3]>,
    normal: Vec<[f32; 3]>,
    depth: Vec<f32>,
    variance: Vec<f32>,
    sample_counts: Vec<u32>,
    scale: f32,
    settings: RaytraceSettings,
}

impl Pair {
    fn example(&self) -> DenoiseExample<'_> {
        DenoiseExample {
            width: SIZE,
            height: SIZE,
            noisy: &self.noisy,
            guides: DenoiseGuides {
                albedo: &self.albedo,
                normal: &self.normal,
                depth: &self.depth,
                variance: &self.variance,
                sample_counts: Some(&self.sample_counts),
                scene_scale: self.scale,
            },
            reference: &self.reference,
        }
    }

    fn beauty(&self, hdr: &[f32]) -> Vec<u8> {
        // Tone-map via a throwaway film of matching size.
        let film = threers::raytrace::Film::new(SIZE, SIZE);
        film.beauty_rgba8(hdr, &self.settings)
    }
}

fn settings(samples: u32) -> RaytraceSettings {
    RaytraceSettings {
        samples_per_pixel: samples,
        max_bounces: 5,
        min_bounces: 3,
        clamp_indirect: 10.0,
        adaptive_threshold: 0.0,
        denoise: false,
        tone_mapping: ToneMapping::AcesFilmic,
        seed: 0xA11A_7E57,
        ..Default::default()
    }
}

fn render_pair(build: fn() -> (Scene, PerspectiveCamera)) -> Pair {
    let s = settings(NOISY);
    let (mut scene, camera) = build();
    let mut r = RaytraceRenderer::new(SIZE, SIZE);
    r.set_settings(s.clone());
    r.render(&mut scene, &camera).expect("render");
    let film = r.film();
    let noisy = film.resolve_hdr();
    let albedo = film.resolve_albedo();
    let normal = film.resolve_normal();
    let depth = film.resolve_depth();
    let variance = film.resolve_variance();
    let sample_counts = film.resolve_sample_counts();
    let scale = r.traced_scene().map(|sc| sc.scale()).unwrap_or(1.0);

    let (mut scene, camera) = build();
    let mut r = RaytraceRenderer::new(SIZE, SIZE);
    r.set_settings(settings(REFERENCE));
    r.render(&mut scene, &camera).expect("render");
    let reference = r.film().resolve_hdr();

    Pair {
        noisy,
        reference,
        albedo,
        normal,
        depth,
        variance,
        sample_counts,
        scale,
        settings: s,
    }
}

fn write_png(path: impl AsRef<std::path::Path>, w: u32, h: u32, rgba: &[u8]) {
    std::fs::write(path.as_ref(), encode_png(w, h, rgba)).expect("write");
}

fn hstack(tile_w: u32, tile_h: u32, packed: &[u8], total_w: u32) -> Vec<u8> {
    // packed is already left-to-right full rows? We appended whole frames, so
    // rearrange into a true strip.
    let n = (total_w / tile_w) as usize;
    let mut out = vec![0u8; (total_w * tile_h * 4) as usize];
    for (ti, _) in (0..n).enumerate() {
        let src = &packed[(ti * (tile_w * tile_h * 4) as usize)..][..(tile_w * tile_h * 4) as usize];
        for y in 0..tile_h as usize {
            let dst_row = y * total_w as usize + ti * tile_w as usize;
            let src_row = y * tile_w as usize;
            out[dst_row * 4..][..tile_w as usize * 4]
                .copy_from_slice(&src[src_row * 4..][..tile_w as usize * 4]);
        }
    }
    out
}

fn look_at(eye: Vector3, target: Vector3) -> PerspectiveCamera {
    let mut cam = PerspectiveCamera::new(45.0, 1.0, 0.1, 200.0);
    cam.position = eye;
    cam.target = target;
    cam
}

fn ceiling_light(scene: &mut Scene, size: f32, height: f32, intensity: f32) {
    let mut em = StandardMaterial::new(Color::BLACK);
    em.emissive = Color::WHITE;
    em.emissive_intensity = intensity;
    let mut panel = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(size, size),
        Material::Standard(em),
    ));
    panel.position = Vector3::new(0.0, height, 0.0);
    panel.rotate_x(FRAC_PI_2);
    scene.add(panel);
}

fn chequer(repeat: f32) -> Arc<Texture> {
    let n = 16u32;
    let mut px = vec![255u8; (n * n * 4) as usize];
    for y in 0..n {
        for x in 0..n {
            let on = (x + y) % 2 == 0;
            let i = ((y * n + x) * 4) as usize;
            px[i] = if on { 235 } else { 30 };
            px[i + 1] = if on { 215 } else { 45 };
            px[i + 2] = if on { 190 } else { 65 };
        }
    }
    let mut tex = Texture::new(n, n, TextureFormat::Rgba8UnormSrgb, px);
    tex.wrap_s = TextureWrap::Repeat;
    tex.wrap_t = TextureWrap::Repeat;
    tex.repeat = Vector2::new(repeat, repeat);
    tex.flip_y = false;
    Arc::new(tex)
}

fn floor(scene: &mut Scene, material: Material) {
    let mut f = Object3D::mesh(Mesh::new(PlaneGeometry::new(40.0, 40.0), material));
    f.rotate_x(-FRAC_PI_2);
    scene.add(f);
}

fn area_light() -> (Scene, PerspectiveCamera) {
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    let mut m = StandardMaterial::new(Color::new(0.6, 0.55, 0.5));
    m.roughness = 0.5;
    floor(&mut scene, Material::Standard(m));
    let mut cube = Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.2, 1.2, 1.2),
        Material::Standard(StandardMaterial::new(Color::new(0.7, 0.3, 0.25))),
    ));
    cube.position = Vector3::new(0.4, 0.6, 0.0);
    cube.rotate_y(0.4);
    scene.add(cube);
    ceiling_light(&mut scene, 1.5, 4.0, 28.0);
    (
        scene,
        look_at(Vector3::new(0.0, 1.8, 4.5), Vector3::new(0.0, 0.6, 0.0)),
    )
}

fn mirror_over_chequer() -> (Scene, PerspectiveCamera) {
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    let mut m = StandardMaterial::new(Color::WHITE);
    m.map = Some(chequer(8.0));
    m.roughness = 0.55;
    floor(&mut scene, Material::Standard(m));

    let mut mirror = StandardMaterial::new(Color::new(0.95, 0.95, 0.95));
    mirror.roughness = 0.03;
    mirror.metalness = 1.0;
    let mut ball = Object3D::mesh(Mesh::new(
        SphereGeometry::new(1.0, 48, 32),
        Material::Standard(mirror),
    ));
    ball.position = Vector3::new(0.0, 1.0, 0.0);
    scene.add(ball);
    ceiling_light(&mut scene, 2.0, 5.0, 25.0);
    (
        scene,
        look_at(Vector3::new(0.0, 2.0, 5.5), Vector3::new(0.0, 0.8, 0.0)),
    )
}
