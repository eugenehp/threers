//! Image-based lighting: one HDRI, no lights at all.
//!
//! This is how a realistic render is normally set up, and it is worth being
//! precise about why. There is no `DirectionalLight` in this scene and no
//! `AmbientLight`. The only thing emitting is `scene.environment` — a sky whose
//! sun is about four orders of magnitude brighter than the blue around it, and
//! whose blue is itself brighter than the ground bounce. Every shadow, every
//! highlight and every bit of fill in the image comes out of that one map.
//!
//! It needs three things to work, and the path tracer now has all three:
//!
//! 1. **Real dynamic range.** `CubeTexture::new_f32` keeps the HDR faces beside
//!    a tone-mapped 8-bit copy meant for display. Integrating the display copy
//!    caps the sky at about 1.0, which turns a sun into a slightly bright patch
//!    and the render into an overcast day.
//! 2. **Importance sampling.** The sun here covers roughly 10⁻⁵ of the sphere.
//!    A cosine-weighted direction finds it about once in a hundred thousand
//!    samples, and the one that does carries ten thousand times the mean.
//!    Without a density built over the sky's own brightness this image is
//!    speckle at any sample count.
//! 3. **The right cube convention.** The faces have to be read in the
//!    orientation they were written in, or the sky arrives with each face
//!    rotated half a turn — which moves the sun somewhere else entirely.
//!
//! ```text
//! cargo run --release --example hdri_lighting --features raytrace,parallel
//!
//! # or point it at a real .hdr — Poly Haven's are a good place to start
//! HDRI=studio.hdr cargo run --release --example hdri_lighting --features raytrace,parallel
//! ```
//!
//! Writes `out/hdri_lighting.png`.

use std::f32::consts::{FRAC_PI_2, PI};
use std::sync::Arc;

use threers::raytrace::{BackgroundMode, RaytraceRenderer, RaytraceSettings};
use threers::{
    encode_png, BoxGeometry, Color, CubeTexture, HdrLoader, Material, Mesh, Object3D,
    PerspectiveCamera, PhysicalMaterial, PlaneGeometry, PmremGenerator, Scene, SphereGeometry,
    StandardMaterial, ToneMapping, Vector3,
};

const W: u32 = 900;
const H: u32 = 560;
/// Equirectangular resolution of the generated sky.
const SKY_W: u32 = 1024;
const SKY_H: u32 = 512;
/// Cube face size the environment is resampled to.
const CUBE: u32 = 256;

fn main() {
    std::fs::create_dir_all("out").expect("create out/");

    let (equirect, sw, sh, source) = load_or_generate_sky();
    report_dynamic_range(&equirect, &source);

    // Equirectangular -> cube. `from_equirect_f32` is the HDR path: the result
    // carries `faces_f32`, so nothing is clamped on the way through.
    let cube = PmremGenerator::from_equirect_f32(&equirect, sw, sh, CUBE);

    let mut scene = build_scene(Arc::new(cube));
    let mut camera = PerspectiveCamera::new(35.0, W as f32 / H as f32, 0.1, 500.0);
    camera.position = Vector3::new(4.6, 1.9, 6.2);
    camera.target = Vector3::new(0.0, 0.75, 0.0);

    let mut renderer = RaytraceRenderer::new(W, H);
    renderer.set_settings(RaytraceSettings {
        samples_per_pixel: env_u32("SPP", 256),
        max_bounces: 6,
        min_bounces: 3,
        // The sun is bright enough that a near-specular chain to it produces
        // the occasional enormous sample. Clamping the indirect ones costs a
        // fraction of a percent of energy and removes the white dots.
        clamp_indirect: 12.0,
        // The sky *is* the background here, so what lights the subject and what
        // sits behind it are the same thing — which is the point of an HDRI.
        background: BackgroundMode::Environment,
        tone_mapping: ToneMapping::AcesFilmic,
        // An HDRI is in absolute-ish units and this one is a clear midday sky:
        // the sun alone delivers about 15 units of irradiance, so a diffuse
        // surface comes back at radiance ~2.5 and lands pure white under a
        // straight ACES curve. Exposure is the control for that, exactly as on
        // a camera — the scene is not "too bright", the film is.
        exposure: std::env::var("EXPOSURE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.14),
        denoise: env_u32("DENOISE", 1) != 0,
        ..Default::default()
    });

    renderer.prepare(&mut scene, &camera);
    if let Some(report) = renderer.report() {
        println!("scene: {}", report.summary());
    }
    println!(
        "environment: {}, resampled to {CUBE}x{CUBE} per face, importance-sampled: {}",
        source,
        renderer
            .traced_scene()
            .map(|s| s.world.env_distribution().is_some())
            .unwrap_or(false)
    );

    let total = renderer.settings().samples_per_pixel;
    let start = std::time::Instant::now();
    let mut done = 0;
    while done < total {
        let n = 32.min(total - done);
        renderer.accumulate(n).expect("trace");
        done += n;
        print!(
            "\r  {done:>4}/{total} samples   {:.1}s",
            start.elapsed().as_secs_f32()
        );
        use std::io::Write;
        let _ = std::io::stdout().flush();
    }
    println!();

    let rgba = renderer.resolve_rgba();
    std::fs::write("out/hdri_lighting.png", encode_png(W, H, &rgba)).expect("write png");
    let (lo, hi) = renderer.sample_range();
    println!(
        "wrote out/hdri_lighting.png in {:.1}s  ({lo}-{hi} samples per pixel, {:.0}% of budget)",
        start.elapsed().as_secs_f32(),
        renderer.sample_efficiency() * 100.0
    );
}

/// Load `$HDRI` if it points at a Radiance `.hdr`, otherwise synthesise one.
fn load_or_generate_sky() -> (Vec<f32>, u32, u32, String) {
    if let Ok(path) = std::env::var("HDRI") {
        match std::fs::read(&path)
            .map_err(|e| e.to_string())
            .and_then(|b| HdrLoader::parse_f32(&b).map_err(|e| format!("{e:?}")))
        {
            Ok((data, w, h)) => return (data, w, h, format!("{path} ({w}x{h})")),
            Err(e) => println!("could not read {path}: {e} — falling back to a generated sky"),
        }
    }
    (
        generate_sky(),
        SKY_W,
        SKY_H,
        format!("generated sky ({SKY_W}x{SKY_H})"),
    )
}

/// A physically-shaped sky: a small very bright sun, a Rayleigh-ish gradient
/// above the horizon, and a dim warm bounce below it.
///
/// The numbers matter more than the shape. A sun of 12000 against a zenith of
/// about 1.5 is a ratio of ~10⁴, which is what a real sky has and what makes
/// the difference between a rendering that needs importance sampling and one
/// that does not.
fn generate_sky() -> Vec<f32> {
    let sun_dir = spherical(35f32.to_radians(), 55f32.to_radians());
    // Angular radius. The real sun is 0.0047 rad; a little larger softens the
    // shadows without changing the character of the problem.
    let sun_radius = 0.020f32;
    let sun_radiance = 12000.0f32;

    let mut out = vec![0.0f32; (SKY_W * SKY_H * 4) as usize];
    for y in 0..SKY_H {
        // Row 0 is the +Y pole: v = 0 at the zenith, matching what
        // `from_equirect_f32` expects.
        let v = (y as f32 + 0.5) / SKY_H as f32;
        let theta = v * PI;
        for x in 0..SKY_W {
            let u = (x as f32 + 0.5) / SKY_W as f32;
            let phi = (u - 0.5) * 2.0 * PI;
            let d = Vector3::new(
                theta.sin() * phi.cos(),
                theta.cos(),
                theta.sin() * phi.sin(),
            );

            let cos_sun = d.dot(sun_dir);
            let c = if cos_sun > sun_radius.cos() {
                Vector3::new(1.0, 0.96, 0.9) * sun_radiance
            } else if d.y >= 0.0 {
                // Sky: blue at the zenith, pale at the horizon, with a glow
                // around the sun that falls off fast.
                let t = d.y.powf(0.45);
                let zenith = Vector3::new(0.16, 0.32, 0.85) * 1.5;
                let horizon = Vector3::new(0.72, 0.80, 0.95) * 1.2;
                let base = horizon.lerp(zenith, t);
                let glow =
                    (cos_sun.max(0.0)).powf(220.0) * 26.0 + (cos_sun.max(0.0)).powf(9.0) * 0.55;
                base + Vector3::new(1.0, 0.88, 0.72) * glow
            } else {
                // Ground bounce, dimming with depth below the horizon.
                let t = (-d.y).powf(0.6);
                Vector3::new(0.20, 0.17, 0.14).lerp(Vector3::new(0.07, 0.06, 0.055), t)
            };

            let o = ((y * SKY_W + x) * 4) as usize;
            out[o] = c.x;
            out[o + 1] = c.y;
            out[o + 2] = c.z;
            out[o + 3] = 1.0;
        }
    }
    out
}

/// Print the ratio the environment actually spans, since that is the number
/// that decides whether any of this was necessary.
fn report_dynamic_range(equirect: &[f32], source: &str) {
    let mut lo = f32::INFINITY;
    let mut hi = 0.0f32;
    let mut sum = 0.0f64;
    for px in equirect.chunks_exact(4) {
        let l = 0.2126 * px[0] + 0.7152 * px[1] + 0.0722 * px[2];
        lo = lo.min(l);
        hi = hi.max(l);
        sum += l as f64;
    }
    let mean = sum / (equirect.len() / 4) as f64;
    println!(
        "{source}: luminance {lo:.3} .. {hi:.0} (mean {mean:.3}) — a range of {:.0}:1",
        hi / lo.max(1e-6)
    );
}

fn build_scene(environment: Arc<CubeTexture>) -> Scene {
    let mut scene = Scene::new();
    scene.environment = Some(environment);

    // Ground: a large slightly rough dielectric, so the sky reflects off it at
    // grazing angles the way real ground does.
    let mut ground = StandardMaterial::new(Color::new(0.42, 0.42, 0.44));
    ground.roughness = 0.62;
    ground.metalness = 0.0;
    let mut floor = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(200.0, 200.0),
        Material::Standard(ground),
    ));
    floor.rotate_x(-FRAC_PI_2);
    scene.add(floor);

    // Brushed copper. Rough metal is where the energy-compensation term shows:
    // without it this reads grey rather than metallic.
    let mut copper = StandardMaterial::new(Color::new(0.95, 0.64, 0.54));
    copper.roughness = 0.34;
    copper.metalness = 1.0;
    let mut ball = Object3D::mesh(Mesh::new(
        SphereGeometry::new(0.75, 64, 48),
        Material::Standard(copper),
    ));
    ball.position = Vector3::new(-1.55, 0.75, 0.1);
    scene.add(ball);

    // Glass. Its shadow is a real shadow — light that gets through arrives on
    // BSDF-sampled paths, not by pretending the shadow ray passed straight in.
    let mut glass = PhysicalMaterial::new(Color::WHITE);
    glass.transmission = 1.0;
    glass.roughness = 0.02;
    glass.ior = 1.5;
    glass.attenuation_color = Color::new(0.85, 0.93, 0.88);
    glass.attenuation_distance = 2.0;
    let mut sphere = Object3D::mesh(Mesh::new(
        SphereGeometry::new(0.75, 64, 48),
        Material::Physical(glass),
    ));
    sphere.position = Vector3::new(0.15, 0.75, 0.55);
    scene.add(sphere);

    // Painted plastic: a diffuse base under a smooth clearcoat, which is two
    // specular responses — one broad and tinted, one sharp and colourless.
    let mut paint = PhysicalMaterial::new(Color::new(0.06, 0.22, 0.42));
    paint.roughness = 0.55;
    paint.clearcoat = 1.0;
    paint.clearcoat_roughness = 0.05;
    let mut cube = Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.25, 1.25, 1.25),
        Material::Physical(paint),
    ));
    cube.position = Vector3::new(1.85, 0.625, -0.5);
    cube.rotate_y(0.42);
    scene.add(cube);

    scene
}

fn spherical(elevation: f32, azimuth: f32) -> Vector3 {
    Vector3::new(
        elevation.cos() * azimuth.cos(),
        elevation.sin(),
        elevation.cos() * azimuth.sin(),
    )
}

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
