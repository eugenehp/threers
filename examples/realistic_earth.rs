//! A realistic Earth from NASA imagery, in about a hundred lines of setup.
//!
//! ```text
//! cargo run --release --features planet --example realistic_earth
//! cargo run --release --features planet --example realistic_earth -- --size 2560x1440
//! cargo run --release --features planet --example realistic_earth -- --frames 120
//! ```
//!
//! | Flag | Effect |
//! |------|--------|
//! | `--assets DIR` | where the NASA imagery is (default `web/assets/earth`) |
//! | `--size WxH` | output resolution (default 1600x900) |
//! | `--texture N` | width of the procedural fallback maps (default 2048) |
//! | `--frames N` | render an orbit of N frames instead of one still |
//! | `--out DIR` | where to write the PNGs (default `out`) |
//! | `--relief N` | height-map amplitude, in Earth radii (default 0) |
//! | `--moon` | frame the Moon instead of Earth |
//! | `--no-moon` | leave the Moon out |
//! | `--procedural` | ignore the assets directory entirely |
//!
//! Run `scripts/fetch-earth-textures.sh --png --sky` first for the real thing:
//! Blue Marble surface colour, Black Marble city lights, a MODIS cloud
//! composite, GEBCO elevation, the CGI Moon Kit and the Deep Star Maps. All of
//! it is NASA public-domain imagery, none of it is committed, and without it
//! the example generates every map instead and still renders a planet.
//!
//! The point of this example is how little there is to it. Everything that
//! makes the planet believable — deriving relief and ocean gloss from the same
//! elevation field the coastlines come from, putting the city lights only on
//! the night side, stacking the cloud shell inside the atmosphere shell,
//! shading the air as air rather than as a surface — lives in
//! [`threers::planet`], not here.
//!
//! Compare `earth_sun_flare.rs`, which does all of it by hand.

use std::f32::consts::TAU;

use threers::planet::{
    earthshine, tidal_lock_yaw, Atmosphere, EarthTextures, Planet, PlanetMaps, Starfield,
};
use threers::{
    encode_png, AmbientLight, Color, DirectionalLight, HeadlessRenderer, Material, Mesh, Object3D,
    PerspectiveCamera, Quaternion, Scene, SphereGeometry, StandardMaterial, ToneMapping, Vector3,
};

fn arg(flag: &str) -> Option<String> {
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == flag {
            return args.next();
        }
    }
    None
}
fn flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}

/// Earth radii. The Moon is really 60 Earth radii out; at that distance it is
/// four pixels, so it sits closer here — this is a picture, not an ephemeris.
const MOON_DISTANCE: f32 = 7.5;
const SUN_DISTANCE: f32 = 140.0;

/// Where the Moon is at orbit fraction `t`.
///
/// The phase offset is chosen so the Moon is in shot for the default single
/// frame; over a full `--frames` orbit it comes round anyway.
fn moon_orbit(t: f32) -> Vector3 {
    let a = 3.2 + t * TAU * 0.15;
    Vector3::new(a.cos() * MOON_DISTANCE, 0.9, a.sin() * MOON_DISTANCE)
}

fn main() {
    let assets = arg("--assets").unwrap_or_else(|| "web/assets/earth".into());
    let out_dir = arg("--out").unwrap_or_else(|| "out".into());
    let (w, h) = arg("--size")
        .and_then(|s| {
            let (a, b) = s.split_once('x')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        })
        .unwrap_or((1600u32, 900u32));
    let tex: u32 = arg("--texture")
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048)
        .clamp(256, 8192);
    let frames: usize = arg("--frames").and_then(|s| s.parse().ok()).unwrap_or(1);
    // At true scale Everest is 0.0014 Earth radii — invisible. Anything you can
    // see here is exaggerated, so it is a flag rather than a default.
    let relief: f32 = arg("--relief").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let want_moon = !flag("--no-moon");
    let moon_shot = flag("--moon") && want_moon;
    let _ = std::fs::create_dir_all(&out_dir);

    // ---- maps ----
    let started = std::time::Instant::now();
    let textures = EarthTextures::from_dir(&assets).fallback_width(tex);
    let (earth_maps, sources) = if flag("--procedural") {
        println!("generating {tex}x{} maps…", tex / 2);
        (
            threers::planet::generate_earth_maps(tex),
            Default::default(),
        )
    } else {
        println!("loading maps from {assets}…");
        textures.load_reporting()
    };
    for name in &sources.loaded {
        println!("  loaded    {name}");
    }
    for (name, why) in &sources.failed {
        println!("  FAILED    {name}: {why}");
    }
    if !sources.generated.is_empty() {
        println!("  generated {}", sources.generated.join(", "));
    }
    if !sources.any_loaded() && !flag("--procedural") {
        println!("  (nothing on disk — run scripts/fetch-earth-textures.sh --png --sky)");
    }

    // ---- scene ----
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x01020a);

    let earth = Planet::earth()
        .maps(earth_maps)
        // Displacement moves vertices, so the relief can only be as detailed as
        // the mesh. 512x256 is 131k vertices — cheap, and enough to show a
        // mountain range.
        .segments(
            if relief > 0.0 { 512 } else { 256 },
            if relief > 0.0 { 256 } else { 128 },
        )
        .displacement(relief)
        .atmosphere(Some(Atmosphere {
            // Solar wind down the field lines into a ring around each
            // geomagnetic pole. Off by default in the library, since a planet
            // needs a magnetic field for one and most do not have one.
            aurora: 2.2,
            ..Atmosphere::earth()
        }))
        .add_to(&mut scene);

    let moon = want_moon.then(|| {
        let maps = if flag("--procedural") {
            None
        } else {
            textures.load_moon()
        };
        match &maps {
            Some(_) => println!("  loaded    CGI Moon Kit"),
            None => println!("  generated Moon (no CGI Moon Kit imagery — it will be plain grey)"),
        }
        Planet::moon()
            .maps(maps.unwrap_or_else(PlanetMaps::new))
            .color(Color::from_hex(0x8f8b85))
            // LOLA relief as real geometry, and unlike Earth it barely needs
            // exaggerating: the Moon's true range is ~19 km on a 1737 km
            // radius, about 1%, and craters are big enough to read at that.
            // 3% here, centred, so rims break the silhouette.
            .displacement_centered(if relief > 0.0 {
                relief * 0.35
            } else {
                0.2727 * 0.03
            })
            .position(moon_orbit(0.0))
            .add_to(&mut scene)
    });

    // The sky goes in before the lights, so nothing tries to shade it.
    let (sky, sky_src) = if flag("--procedural") {
        (
            threers::planet::generate_starfield(2048, 12_000),
            Default::default(),
        )
    } else {
        textures.load_starfield_reporting()
    };
    for name in &sky_src.loaded {
        println!("  loaded    {name}");
    }
    for (name, why) in &sky_src.failed {
        println!("  FAILED    {name}: {why}");
    }
    if !sky_src.generated.is_empty() {
        println!("  generated starfield");
    }
    // Six quads rather than a sphere: an equirectangular map on a UV sphere
    // converges every longitude onto one texel at each pole, and anything with
    // width near there fans out radially when it is wrapped. A cube has no such
    // point. `faceted(0)` picks a face size a quarter of the map's width, which
    // is lossless at the equator and a gain toward the poles.
    Starfield::new(sky)
        .faceted(0)
        .radius(320.0)
        .intensity(0.9)
        .add_to(&mut scene);

    // ---- the sun, and the light it casts ----
    let sun_dir = Vector3::new(1.0, 0.16, 0.52).normalize();
    let sun_pos = sun_dir * SUN_DISTANCE;
    let mut sun_mat = StandardMaterial::new(Color::from_hex(0xfff6e0));
    sun_mat.emissive = Color::from_hex(0xfff4d6);
    // Far brighter than white on purpose; ACES turns it into a disc rather
    // than a flat white blob.
    sun_mat.emissive_intensity = 90.0;
    let mut sun = Object3D::mesh(Mesh::new(
        SphereGeometry::new(2.4, 48, 24),
        Material::Standard(sun_mat),
    ));
    sun.position = sun_pos;
    scene.add(sun);

    scene.add_light(
        DirectionalLight::new(Color::from_hex(0xfff6ea), 3.1).with_direction(sun_dir * -1.0),
    );
    // Earthshine. The Moon's dark limb is lit by a nearly full Earth — sixty
    // times the area of our full moon and four times as reflective — which is
    // bright enough to read craters by, and is the single most obvious thing
    // missing from a rendered crescent. Brightest on a thin crescent, because
    // the Earth's phase is the complement of the Moon's.
    {
        let (dir, strength) = earthshine(sun_dir, Vector3::ZERO, moon_orbit(0.0), 1.0, 0.30);
        // Scaled up from the geometric figure: the sun light above is itself
        // stylised at 3.1 rather than being in physical units.
        scene.add_light(
            DirectionalLight::new(Color::from_hex(0x7fa6d8), strength * 260.0).with_direction(dir),
        );
    }
    // Starlight, so the night side is dark rather than absent.
    scene.add_light(AmbientLight::new(Color::from_hex(0x0a1428), 0.32));

    // ---- render ----
    let mut renderer = HeadlessRenderer::builder()
        .size(w, h)
        .supersample(2)
        // Rgba8Unorm: the mesh shader already encodes sRGB.
        .color_format(wgpu::TextureFormat::Rgba8Unorm)
        .high_resolution(true)
        .build()
        .expect("headless renderer (is a GPU adapter available?)");
    renderer
        .renderer()
        .set_tone_mapping(ToneMapping::AcesFilmic, 1.0);

    let mut camera = PerspectiveCamera::new(40.0, w as f32 / h as f32, 0.01, 500.0);
    for i in 0..frames {
        let t = if frames > 1 {
            i as f32 / frames as f32
        } else {
            0.0
        };
        let yaw = 0.6 + t * TAU;
        camera.position = Vector3::new(
            yaw.cos() * 3.4,
            0.5 + (t * TAU).sin() * 0.3,
            yaw.sin() * 3.4,
        );
        camera.look_at(Vector3::ZERO);
        if moon_shot {
            // Close on the Moon, from a little off the sun line so the
            // terminator rakes across the craters instead of flattening them.
            let m = moon_orbit(t);
            let off = Vector3::new(0.45, 0.22, 0.62).normalize() * (0.2727 * 3.6);
            camera.position = m + off;
            camera.look_at(m);
        }

        // Spin Earth on its tilted axis; drift the clouds a little faster, as
        // the prevailing winds do.
        let spin = |o: &mut Object3D, turns: f32| {
            o.quaternion = Quaternion::from_euler_xyz(23.44f32.to_radians(), t * TAU * turns, 0.0);
        };
        if let Some(o) = scene.get_mut(earth.surface) {
            spin(o, 0.35);
        }
        if let Some(o) = earth.clouds.and_then(|id| scene.get_mut(id)) {
            spin(o, 0.42);
        }
        // The Moon is tidally locked, so its spin matches its orbit.
        if let Some(m) = moon {
            if let Some(o) = scene.get_mut(m.surface) {
                o.position = moon_orbit(t);
                // Tidally locked: one rotation per orbit, which is what keeps
                // the near side turned toward us. This used to negate the
                // orbital angle instead, which locks the Moon just as firmly
                // and to the wrong face — every render showed the far side,
                // cratered all over and missing every mare.
                o.quaternion = Quaternion::from_euler_xyz(0.0, tidal_lock_yaw(moon_orbit(t)), 0.0);
            }
        }

        let rgba = renderer.render_to_rgba_resolved(&mut scene, &camera);
        let path = if frames > 1 {
            format!("{out_dir}/realistic_earth_{i:04}.png")
        } else {
            format!("{out_dir}/realistic_earth.png")
        };
        std::fs::write(&path, encode_png(w, h, &rgba)).expect("write frame");
        if frames > 1 && i % 10 == 0 {
            println!("  frame {i}/{frames}");
        } else if frames == 1 {
            println!("wrote {path}");
        }
    }
    println!("done in {:.1}s", started.elapsed().as_secs_f32());
}
