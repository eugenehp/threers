//! A sea of sand dunes at four times of day: dawn, day, sunset and night.
//!
//! A kilometre of dunes grown procedurally (a curving crest network with
//! broad flowing breakup and smaller dunes on the big ones) and then worked
//! by the wind — sand lifted off the windward slopes and dropped in the lee —
//! so each dune has a long gentle back and a slip face at 34°. Drawn with
//! the sand shader after Journey's: grains, ripples across the wind on the
//! flats and along the contours on slopes, an anisotropic sheen stretched
//! along the wind, glitter, sand streaking up the windward faces. Sand
//! smokes off the crests in the wind, lit as a medium. Under a sky whose
//! horizon the haze over the sand meets without a seam; shadows in cascades;
//! bloom and film grain. Written to `out/dunes.png` (the four) and
//! `out/dunes_sunset.png`.
//!
//! ```bash
//! cargo run --release --example dunes
//! ```

use std::sync::Arc;
use std::time::Instant;

use threers::lights::Light;
use threers::particles::dunes::{crests, sky_dome, DesertTime, DuneShape};
use threers::particles::sand::{Downwind, SandBed};
use threers::particles::{presets, Heightfield, ParticleAnimator};
use threers::postprocessing::{EffectComposer, FilmPass, OutputPass, RenderPass, UnrealBloomPass};
use threers::prelude::*;
use threers::renderer::{readback, RenderTarget};

const FORMAT: threers::wgpu::TextureFormat = threers::wgpu::TextureFormat::Rgba8Unorm;

/// One time of day over `bed`, from `eye` toward `at`, `w` × `h`: RGBA.
fn render(bed: &SandBed, time: &DesertTime, eye: Vector3, at: Vector3, w: u32, h: u32) -> Vec<u8> {
    let mut scene = Scene::new();
    scene.background = time.horizon;
    scene.fog = time.fog();
    scene.add_light(HemisphereLight::new(
        time.sky_light,
        time.ground_light,
        time.sky_intensity,
    ));
    let mut sun = DirectionalLight::new(time.sun_color, time.sun_intensity);
    sun.cast_shadow = true;
    sun.shadow.map_size = 2048;
    sun.shadow.cascades = 3;
    sun.shadow.cascade_distance = 900.0;
    sun.shadow.camera_far = 3000.0;
    sun.shadow.normal_bias = 0.25;
    sun.shadow.radius = 2.0;
    let mut l = Object3D::light(Light::Directional(sun));
    l.position = time.sun.normalize() * 800.0;
    scene.add(l);
    let mut sky = sky_dome(time, 2200.0);
    sky.position = eye;
    scene.add(sky);

    let mut bed = bed.clone();
    bed.color = time.sand;
    bed.shadow_color = time.shadow;
    bed.fog_glow = time.glow;
    let mut anim = ParticleAnimator::new();
    let wind = Vector3::new(9.0, 0.0, 1.5);
    let brinks: Arc<[Vector3]> =
        crests(&bed.field, Downwind::PlusX, 18f32.to_radians(), 0.4, 3000).into();
    if std::env::var("COUNT").is_ok() {
        println!("  {} brinks", brinks.len());
    }
    anim.add_sand(&mut scene, bed, None);
    let pale = Color::new(time.sand.r * 1.15, time.sand.g * 1.15, time.sand.b * 1.15);
    let mut drift = threers::core::ObjectId::default();
    if std::env::var("NO_DRIFT").is_err() {
        drift = anim.add(
            &mut scene,
            presets::spindrift(brinks.clone(), wind, 3000.0, pale),
        );
    }
    for _ in 0..90 {
        anim.update(&mut scene, 1.0 / 30.0, Some(eye));
    }
    if std::env::var("COUNT").is_ok() {
        println!("  {} particles", anim.system(drift).map_or(0, |s| s.len()));
    }

    let mut camera = PerspectiveCamera::new(42.0, w as f32 / h as f32, 0.5, 3000.0);
    camera.position = eye;
    camera.look_at(at);
    let mut hr = HeadlessRenderer::builder()
        .size(w, h)
        .color_format(FORMAT)
        .build()
        .expect("a GPU adapter");
    hr.set_msaa(4);
    hr.renderer().set_tone_mapping(ToneMapping::AcesFilmic, 1.0);
    let mut composer = EffectComposer::new(hr.device().clone(), hr.queue().clone(), w, h, FORMAT);
    composer.add_pass(RenderPass::new());
    composer.add_pass(UnrealBloomPass::new(w, h, 0.45, 0.5, 0.9));
    composer.add_pass(FilmPass::new(0.12, false));
    composer.add_pass(OutputPass::new());
    let out = RenderTarget::new(hr.device(), w, h, FORMAT);
    composer.render(hr.renderer(), &mut scene, &camera, &out.color_view);
    readback::texture_bytes(hr.device(), hr.queue(), &out.color_texture)
}

fn main() {
    // A kilometre and a bit of dunes, 3 m cells.
    let t = Instant::now();
    let mut bed = SandBed::new(Heightfield::new([0.0, 0.0], [1200.0, 1200.0], 400, 0.0));
    bed.dunes(DuneShape::default());
    bed.ripple_wavelength = 0.35;
    bed.ripples = 0.6;
    bed.streaks = 1.0;
    bed.stylize = 0.25;
    println!("grew the dunes in {:.1} s", t.elapsed().as_secs_f32());

    // Standing on a rise near the upwind edge, looking downwind.
    let ground = |x: f32, z: f32| bed.field.height_at(x, z).unwrap_or(0.0);
    let (ex, ez) = (-470.0, -60.0);
    let eye = Vector3::new(ex, ground(ex, ez) + 7.0, ez);
    let at = Vector3::new(ex + 300.0, ground(ex + 300.0, ez + 40.0) + 4.0, ez + 40.0);

    let (w, h) = (640u32, 360u32);
    let mut sheet = vec![0u8; (w * 2 * h * 2 * 4) as usize];
    let only = std::env::var("ONLY").ok();
    for (k, (name, time)) in DesertTime::all().iter().enumerate() {
        if only.as_deref().is_some_and(|o| o != *name) {
            continue;
        }
        let t = Instant::now();
        let px = render(&bed, time, eye, at, w, h);
        println!("{name}: {:.1} s", t.elapsed().as_secs_f32());
        let (ox, oy) = ((k as u32 % 2) * w, (k as u32 / 2) * h);
        for y in 0..h {
            let src = &px[(y * w * 4) as usize..((y + 1) * w * 4) as usize];
            let at = (((oy + y) * w * 2 + ox) * 4) as usize;
            sheet[at..at + src.len()].copy_from_slice(src);
        }
    }
    std::fs::create_dir_all("out").unwrap();
    std::fs::write("out/dunes.png", encode_png(w * 2, h * 2, &sheet)).unwrap();
    let hero = render(&bed, &DesertTime::sunset(), eye, at, 1280, 720);
    std::fs::write("out/dunes_sunset.png", encode_png(1280, 720, &hero)).unwrap();
    println!("wrote out/dunes.png and out/dunes_sunset.png");
}
