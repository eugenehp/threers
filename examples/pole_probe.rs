//! Renders the north pole from the real NASA maps, one contributing map at a
//! time, so the polar fan can be attributed to a specific one.
//!
//! ```text
//! cargo run --release --features planet --example pole_probe
//! ```
//!
//! Writes `out/pole_*.png`. The camera looks straight down the spin axis, so
//! anything that is a function of latitude alone must come out rotationally
//! symmetric; whatever shows spokes is the map at fault.

use threers::planet::EarthTextures;
use threers::{
    encode_png, AmbientLight, Color, DirectionalLight, HeadlessRenderer, Material, Mesh, Object3D,
    PerspectiveCamera, Scene, SphereGeometry, StandardMaterial, ToneMapping, Vector3,
};

const W: u32 = 700;
const H: u32 = 700;

fn main() {
    let flag = |name: &str| {
        std::env::args()
            .position(|a| a == name)
            .and_then(|i| std::env::args().nth(i + 1))
    };
    let assets = flag("--assets").unwrap_or_else(|| "web/assets/earth".into());
    let (maps, src) = EarthTextures::from_dir(&assets).load_reporting();
    eprintln!("loaded: {:?}", src.loaded);
    eprintln!("failed: {:?}", src.failed);

    let mut r = HeadlessRenderer::builder()
        .size(W, H)
        .supersample(2)
        .color_format(wgpu::TextureFormat::Rgba8Unorm)
        .build()
        .expect("no GPU adapter");
    r.renderer().set_tone_mapping(ToneMapping::AcesFilmic, 1.0);
    std::fs::create_dir_all("out").ok();

    // Grazing light, because a low sun is what makes relief errors visible at
    // all — straight-down light hides slope behind its own cosine falloff.
    let variants: Vec<(&str, bool, bool, bool)> = vec![
        ("albedo", true, false, false),
        ("normal", false, true, false),
        ("rough", false, false, true),
        ("all", true, true, true),
        ("clouds", true, true, true),
    ];

    for (name, albedo, normal, rough) in variants {
        for south in [false, true] {
            let mut m = StandardMaterial::new(Color::WHITE);
            m.roughness = 0.9;
            m.metalness = 0.0;
            if albedo {
                m.map = maps.albedo.clone();
            }
            if normal {
                m.normal_map = maps.normal.clone();
            }
            if rough {
                m.roughness_map = maps.roughness.clone();
            }

            let mut scene = Scene::new();
            scene.background = Color::BLACK;
            scene.add_light(AmbientLight::new(Color::WHITE, 0.05));
            // 25 degrees above the horizon, as the polar sun actually is, and on
            // the side of the pole being looked at.
            let a: f32 = 25f32.to_radians();
            let ny = if south { a.sin() } else { -a.sin() };
            scene.add_light(
                DirectionalLight::new(Color::WHITE, 3.0).with_direction(Vector3::new(
                    -a.cos(),
                    ny,
                    0.0,
                )),
            );
            scene.add(Object3D::mesh(Mesh::new(
                SphereGeometry::new(1.0, 512, 256),
                Material::Standard(m),
            )));
            // The demo stacks a cloud shell over the surface. `ease_poles` blurs
            // along longitude, which near a pole smears features *around* it — so
            // if anything here is going to look like a vortex, it is this.
            if name == "clouds" {
                if let Some(c) = maps.clouds.clone() {
                    let mut cm = StandardMaterial::new(Color::WHITE);
                    cm.roughness = 1.0;
                    cm.map = Some(c);
                    cm.opacity = 0.999;
                    let mut shell = Object3D::mesh(Mesh::new(
                        SphereGeometry::new(1.006, 512, 256),
                        Material::Standard(cm),
                    ));
                    shell.name = "clouds".into();
                    scene.add(shell);
                }
            }

            let fov: f32 = flag("--fov").and_then(|v| v.parse().ok()).unwrap_or(14.0);
            let mut c = PerspectiveCamera::new(fov, 1.0, 0.01, 100.0);
            if std::env::args().any(|a| a == "--oblique") {
                // Low over the pole, looking across it — the angle the artefact
                // actually gets reported from, and a very different sampling
                // regime from straight down.
                let s = if south { -1.0 } else { 1.0 };
                c.position = Vector3::new(0.0, s * 1.55, 1.15);
                c.up = Vector3::new(0.0, s, 0.0);
                c.look_at(Vector3::new(0.0, s * 0.72, 0.0));
            } else {
                c.position = Vector3::new(0.0, if south { -3.0 } else { 3.0 }, 0.0);
                c.up = Vector3::new(0.0, 0.0, -1.0);
                c.look_at(Vector3::ZERO);
            }

            let rgba = r.render_to_rgba(&mut scene, &c);
            let (rw, rh) = r.render_size();
            let path = format!("out/pole_{name}_{}.png", if south { "s" } else { "n" });
            std::fs::write(&path, encode_png(rw, rh, &rgba)).unwrap();
            eprintln!("wrote {path}");
        }
    }
}
