//! Fit the denoiser to ground truth the renderer generates itself.
//!
//! Cycles denoises with OpenImageDenoise — a pretrained U-Net. Reproducing that
//! would mean a training corpus of tens of thousands of image pairs, days of
//! GPU time, and tens of megabytes of weights to ship. This is the tractable
//! version of the same idea: the *filter* is the model, its four widths and its
//! iteration count are the parameters, and the training pairs cost nothing but
//! time because a path tracer can render both halves — noisy at 32 samples,
//! converged at 1024.
//!
//! Which is worth doing on its own merits. The shipped defaults were picked by
//! eye on one scene; this measures them against several, over content chosen to
//! disagree — a chequered floor wants a tight albedo threshold, a mirror wants a
//! loose one, an area light wants neither.
//!
//! ```text
//! cargo run --release --example denoise_fit --features raytrace,parallel
//! ```
//!
//! Prints the fitted parameters and what they are worth against the defaults.

use std::f32::consts::FRAC_PI_2;
use std::sync::Arc;

use threers::raytrace::{
    DenoiseExample, DenoiseGuides, DenoiseParams, RaytraceRenderer, RaytraceSettings,
};
use threers::{
    BoxGeometry, Color, Material, Mesh, Object3D, PerspectiveCamera, PhysicalMaterial,
    PlaneGeometry, Scene, SphereGeometry, StandardMaterial, Texture, TextureFormat, TextureWrap,
    ToneMapping, Vector2, Vector3,
};

/// Resolution of each training image. Small on purpose: the fit needs variety
/// across scenes far more than pixels within one.
const SIZE: u32 = 64;
/// Samples in the noisy half — the sample count the denoiser is being asked to
/// rescue.
const NOISY: u32 = 32;
/// Samples in the reference half.
///
/// High enough that the reference's *own* residual grain is well below the
/// error being measured. At 1024 it is not: the noise left in the reference is
/// something a denoised image cannot match by definition, so the fit reads
/// smoothing as error and settles on a filter that visibly under-smooths.
const REFERENCE: u32 = 8192;

/// A named scene builder: everything needed to render one training pair.
type SceneBuilder = (&'static str, fn() -> (Scene, PerspectiveCamera));

fn main() {
    let scenes: Vec<SceneBuilder> = vec![
        ("area light", area_light),
        ("mirror over chequer", mirror_over_chequer),
        ("glass and metal", glass_and_metal),
        ("interior bounce", interior_bounce),
    ];

    println!(
        "rendering {} scenes at {NOISY} and {REFERENCE} samples...",
        scenes.len()
    );
    let rendered: Vec<Rendered> = scenes
        .iter()
        .map(|(name, build)| {
            let start = std::time::Instant::now();
            let r = render_pair(build);
            println!("  {name:<22} {:.1}s", start.elapsed().as_secs_f32());
            r
        })
        .collect();

    // Held-out scene: fitting and judging on the same data says nothing.
    let (train, test) = rendered.split_at(rendered.len() - 1);
    let train_examples: Vec<DenoiseExample> = train.iter().map(Rendered::example).collect();
    let test_examples: Vec<DenoiseExample> = test.iter().map(Rendered::example).collect();

    let defaults = DenoiseParams::default();
    let start = std::time::Instant::now();
    let (fitted, train_error) = if std::env::var("FREE_ITERATIONS").is_ok() {
        DenoiseParams::fit(&train_examples, defaults)
    } else {
        DenoiseParams::fit_widths(&train_examples, defaults)
    };
    let fit_time = start.elapsed().as_secs_f32();

    println!("\nrelative RMS against the converged render, per scene:");
    println!(
        "  {:<22} {:>8} {:>10} {:>8}",
        "", "raw", "defaults", "fitted"
    );
    for (i, (name, _)) in scenes.iter().enumerate() {
        let e = rendered[i].example();
        let one = [rendered[i].example()];
        let held = i + 1 == rendered.len();
        println!(
            "  {:<22} {:>8.4} {:>10.4} {:>8.4}{}",
            name,
            e.relative_rms(e.noisy),
            defaults.error_over(&one),
            fitted.error_over(&one),
            if held { "   (held out)" } else { "" }
        );
    }
    println!(
        "\n  train mean            {:>8.4} {:>10.4} {:>8.4}   (fit took {fit_time:.1}s)",
        train_examples
            .iter()
            .map(|e| e.relative_rms(e.noisy))
            .sum::<f32>()
            / train_examples.len() as f32,
        defaults.error_over(&train_examples),
        train_error
    );
    println!(
        "  held out              {:>8.4} {:>10.4} {:>8.4}",
        test_examples
            .iter()
            .map(|e| e.relative_rms(e.noisy))
            .sum::<f32>()
            / test_examples.len() as f32,
        defaults.error_over(&test_examples),
        fitted.error_over(&test_examples)
    );

    println!("\nfitted parameters:");
    println!(
        "  iterations     {:>8}   (default {})",
        fitted.iterations, defaults.iterations
    );
    for (name, got, was) in [
        ("sigma_color", fitted.sigma_color, defaults.sigma_color),
        ("sigma_normal", fitted.sigma_normal, defaults.sigma_normal),
        ("sigma_depth", fitted.sigma_depth, defaults.sigma_depth),
        ("sigma_albedo", fitted.sigma_albedo, defaults.sigma_albedo),
    ] {
        println!("  {name:<14} {got:>8.4}   (default {was})");
    }
}

/// A rendered pair, kept alive so the borrowed views in `DenoiseExample` stay
/// valid.
struct Rendered {
    width: u32,
    height: u32,
    noisy: Vec<f32>,
    reference: Vec<f32>,
    albedo: Vec<[f32; 3]>,
    normal: Vec<[f32; 3]>,
    depth: Vec<f32>,
    variance: Vec<f32>,
    scale: f32,
}

impl Rendered {
    fn example(&self) -> DenoiseExample<'_> {
        DenoiseExample {
            width: self.width,
            height: self.height,
            noisy: &self.noisy,
            guides: DenoiseGuides {
                albedo: &self.albedo,
                normal: &self.normal,
                depth: &self.depth,
                variance: &self.variance,
                scene_scale: self.scale,
            },
            reference: &self.reference,
        }
    }
}

fn settings(samples: u32) -> RaytraceSettings {
    RaytraceSettings {
        samples_per_pixel: samples,
        max_bounces: 5,
        min_bounces: 3,
        clamp_indirect: 10.0,
        // Every pixel takes the full budget: a fit wants a known noise level,
        // not one that varies with how easily each pixel settled.
        adaptive_threshold: 0.0,
        // The film is denoised here, not by the renderer.
        denoise: false,
        tone_mapping: ToneMapping::None,
        ..Default::default()
    }
}

fn render_pair(build: &fn() -> (Scene, PerspectiveCamera)) -> Rendered {
    let (mut scene, camera) = build();
    let mut r = RaytraceRenderer::new(SIZE, SIZE);
    r.set_settings(settings(NOISY));
    r.render(&mut scene, &camera).expect("render");
    let film = r.film();
    let noisy = film.resolve_hdr();
    let albedo = film.resolve_albedo();
    let normal = film.resolve_normal();
    let depth = film.resolve_depth();
    let variance = film.resolve_variance();
    let scale = r.traced_scene().map(|s| s.scale()).unwrap_or(1.0);

    let (mut scene, camera) = build();
    let mut r = RaytraceRenderer::new(SIZE, SIZE);
    r.set_settings(settings(REFERENCE));
    r.render(&mut scene, &camera).expect("render");
    let reference = r.film().resolve_hdr();

    Rendered {
        width: SIZE,
        height: SIZE,
        noisy,
        reference,
        albedo,
        normal,
        depth,
        variance,
        scale,
    }
}

// ------------------------------------------------------------------- scenes

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

/// Soft shadows and smooth falloff — nothing for the guides to key on.
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

/// High-frequency albedo under a mirror: the albedo guide has to be tight
/// enough to keep the chequer and loose enough not to fight the reflection.
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
        look_at(Vector3::new(0.0, 2.2, 5.5), Vector3::new(0.0, 1.0, 0.0)),
    )
}

/// Refraction and a rough metal — the two surfaces whose guides are deferred.
fn glass_and_metal() -> (Scene, PerspectiveCamera) {
    let mut scene = Scene::new();
    scene.background = Color::new(0.02, 0.03, 0.05);
    let mut m = StandardMaterial::new(Color::new(0.5, 0.5, 0.52));
    m.roughness = 0.6;
    floor(&mut scene, Material::Standard(m));

    let mut glass = PhysicalMaterial::new(Color::WHITE);
    glass.transmission = 1.0;
    glass.roughness = 0.03;
    glass.ior = 1.5;
    let mut ball = Object3D::mesh(Mesh::new(
        SphereGeometry::new(0.85, 48, 32),
        Material::Physical(glass),
    ));
    ball.position = Vector3::new(-0.9, 0.85, 0.0);
    scene.add(ball);

    let mut gold = StandardMaterial::new(Color::new(1.0, 0.76, 0.34));
    gold.roughness = 0.35;
    gold.metalness = 1.0;
    let mut metal = Object3D::mesh(Mesh::new(
        SphereGeometry::new(0.85, 48, 32),
        Material::Standard(gold),
    ));
    metal.position = Vector3::new(0.9, 0.85, 0.0);
    scene.add(metal);
    ceiling_light(&mut scene, 2.5, 4.5, 30.0);
    (
        scene,
        look_at(Vector3::new(0.0, 1.8, 5.0), Vector3::new(0.0, 0.8, 0.0)),
    )
}

/// Held out: colour bleeding and indirect-only regions, where the noise is
/// lowest-frequency and hardest to tell from signal.
fn interior_bounce() -> (Scene, PerspectiveCamera) {
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    let white = || Material::Standard(StandardMaterial::new(Color::new(0.73, 0.73, 0.73)));
    let s = 3.0;
    floor(&mut scene, white());
    let mut ceiling = Object3D::mesh(Mesh::new(PlaneGeometry::new(s, s), white()));
    ceiling.position = Vector3::new(0.0, s, 0.0);
    ceiling.rotate_x(FRAC_PI_2);
    scene.add(ceiling);
    let mut back = Object3D::mesh(Mesh::new(PlaneGeometry::new(s, s), white()));
    back.position = Vector3::new(0.0, s * 0.5, -s * 0.5);
    scene.add(back);
    let mut left = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(s, s),
        Material::Standard(StandardMaterial::new(Color::new(0.65, 0.06, 0.06))),
    ));
    left.position = Vector3::new(-s * 0.5, s * 0.5, 0.0);
    left.rotate_y(FRAC_PI_2);
    scene.add(left);
    let mut right = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(s, s),
        Material::Standard(StandardMaterial::new(Color::new(0.12, 0.45, 0.15))),
    ));
    right.position = Vector3::new(s * 0.5, s * 0.5, 0.0);
    right.rotate_y(-FRAC_PI_2);
    scene.add(right);
    ceiling_light(&mut scene, 1.0, s - 0.02, 35.0);
    (
        scene,
        look_at(Vector3::new(0.0, 1.5, 5.4), Vector3::new(0.0, 1.4, 0.0)),
    )
}
