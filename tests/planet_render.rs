//! What a planet actually looks like once it is drawn.
//!
//! The unit tests in `src/planet` check that the right objects go into the
//! scene; these check the pixels. Every one of them is a bug that got past a
//! structural test — an atmosphere that reddened the whole limb, a cloud shell
//! that hid the planet, city lights bleeding onto the day side.
//!
//! Skipped when no GPU adapter is available.

#![cfg(feature = "planet")]

use threers::planet::{generate_earth_maps, Atmosphere, Planet, PlanetMaps, Starfield};
use threers::{
    AmbientLight, Color, DirectionalLight, HeadlessRenderer, PerspectiveCamera, Scene, ToneMapping,
    Vector3,
};

const W: u32 = 400;
const H: u32 = 400;
/// Where the sun is, as a unit vector from the planet.
const SUN: Vector3 = Vector3 {
    x: 1.0,
    y: 0.0,
    z: 0.15,
};

fn renderer() -> Option<HeadlessRenderer> {
    let mut r = HeadlessRenderer::builder()
        .size(W, H)
        .color_format(wgpu::TextureFormat::Rgba8Unorm)
        .build()
        .ok()?;
    r.renderer().set_tone_mapping(ToneMapping::AcesFilmic, 1.0);
    Some(r)
}

/// A camera at `distance`, looking at the origin from `direction`.
fn camera(direction: Vector3, distance: f32) -> PerspectiveCamera {
    let mut c = PerspectiveCamera::new(40.0, 1.0, 0.01, 100.0);
    c.position = direction.normalize() * distance;
    c.look_at(Vector3::ZERO);
    c
}

fn lit_scene() -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    scene.add_light(DirectionalLight::new(Color::WHITE, 3.0).with_direction(SUN * -1.0));
    scene.add_light(AmbientLight::new(Color::from_hex(0x0a1428), 0.3));
    scene
}

/// `[r, g, b]` at a pixel.
fn px(img: &[u8], x: u32, y: u32) -> [u8; 3] {
    let o = ((y * W + x) * 4) as usize;
    [img[o], img[o + 1], img[o + 2]]
}

/// The brightest pixel on the horizontal scanline through the middle, searching
/// outward from the disc edge — i.e. the atmosphere's glow into space.
fn limb_glow(img: &[u8], from_x: u32, to_x: u32) -> (u32, [u8; 3]) {
    let mut best = (from_x, [0u8; 3]);
    for x in from_x..to_x {
        let p = px(img, x, H / 2);
        if p.iter().map(|&v| v as u32).sum::<u32>() > best.1.iter().map(|&v| v as u32).sum() {
            best = (x, p);
        }
    }
    best
}

#[test]
fn the_atmosphere_glows_blue_past_the_planet_edge() {
    let Some(mut r) = renderer() else { return };
    let mut scene = lit_scene();
    Planet::new(1.0)
        .color(Color::from_hex(0x203040))
        .atmosphere(Some(Atmosphere::earth()))
        .add_to(&mut scene);

    // Head-on into the sun's direction, so the whole disc is lit and the limb
    // is the terminator ring. This is the view that used to come out orange:
    // the reddening term keyed on the sun angle alone, and from here every
    // point on the limb has the sun exactly on its horizon.
    let img = r.render_to_rgba(&mut scene, &camera(SUN, 4.0));

    // The disc spans roughly the middle half of the frame; look just outside it.
    let (x, glow) = limb_glow(&img, W * 3 / 4, W - 4);
    let [gr, gg, gb] = glow.map(|v| v as i32);
    assert!(
        gb > 12,
        "no atmosphere outside the disc at all: {glow:?} at x={x}"
    );
    assert!(
        gb > gr + 6,
        "the limb should be blue, not neutral or red: r={gr} g={gg} b={gb}"
    );
    assert!(gb >= gg, "blue should lead green: r={gr} g={gg} b={gb}");
}

#[test]
fn the_atmosphere_does_not_bury_the_planet() {
    let Some(mut r) = renderer() else { return };
    let build = |air: Option<Atmosphere>| {
        let mut scene = lit_scene();
        Planet::new(1.0)
            .color(Color::from_hex(0x2a6bbf))
            .atmosphere(air)
            .add_to(&mut scene);
        scene
    };
    let cam = camera(SUN, 4.0);
    let bare = r.render_to_rgba(&mut build(None), &cam);
    let with_air = r.render_to_rgba(&mut build(Some(Atmosphere::earth())), &cam);

    // Dead centre, the ray goes straight down through the shell: the shortest
    // path there is, so it should barely tint. Normalising optical depth
    // against the shell thickness rather than the grazing chord made this ray
    // as opaque as the limb and flattened the planet under blue.
    let (c, a) = (px(&bare, W / 2, H / 2), px(&with_air, W / 2, H / 2));
    let shift: i32 = (0..3).map(|i| (a[i] as i32 - c[i] as i32).abs()).sum();
    assert!(
        shift < 60,
        "the atmosphere swamped the middle of the disc: {c:?} → {a:?}"
    );
}

/// The longitude, in radians, where the night map has the most lit texels.
///
/// Only about 1% of the map is lit and the continents clump, so pointing a
/// camera at a fixed longitude and expecting cities is a coin flip. This finds
/// where they actually are.
fn brightest_longitude(night: &threers::Texture) -> f32 {
    let w = night.width as usize;
    let mut per_column = vec![0u32; w];
    for (i, p) in night.data.chunks_exact(4).enumerate() {
        if p[0] > 40 {
            per_column[i % w] += 1;
        }
    }
    let x = per_column
        .iter()
        .enumerate()
        .max_by_key(|(_, &c)| c)
        .map(|(x, _)| x)
        .unwrap_or(0);
    ((x as f32 + 0.5) / w as f32 - 0.5) * std::f32::consts::TAU
}

#[test]
fn city_lights_show_on_the_night_side_without_flooding_the_day_side() {
    let Some(mut r) = renderer() else { return };
    let maps = generate_earth_maps(512);

    // Turn the busiest longitude to face the night-side quadrant nearest the
    // camera. A Y rotation of `spin` moves a point's azimuth to `azimuth -
    // spin`; the camera is at +Z and the sun at +X, so 3π/4 is the middle of
    // the visible night crescent.
    let spin = brightest_longitude(maps.night.as_ref().unwrap()) - 3.0 * std::f32::consts::PI / 4.0;

    let build = |with_night: bool| {
        let mut m = maps.clone();
        if !with_night {
            m.night = None;
        }
        // Clouds off, so nothing floats over the lights and confuses the count.
        m.clouds = None;
        let mut scene = lit_scene();
        // No tilt, so the spin above lands where the arithmetic says it does.
        Planet::new(1.0)
            .maps(m)
            .spin_degrees(spin.to_degrees())
            .add_to(&mut scene);
        scene
    };
    let cam = camera(Vector3::new(0.0, 0.0, 1.0), 3.2);
    let dark = r.render_to_rgba(&mut build(false), &cam);
    let lit = r.render_to_rgba(&mut build(true), &cam);

    // Only pixels on the disc: empty space would otherwise dominate every
    // count. A unit sphere at 3.2 through a 40° lens is 172 px across; inset a
    // little to stay off the limb.
    let radius = (1.0 / (3.2 * (40.0f32.to_radians() / 2.0).tan())) * (H as f32 / 2.0) - 6.0;
    let (mut disc, mut changed, mut night_lit) = (0, 0, 0);
    let (mut night_peak_dark, mut night_peak_lit) = (0i32, 0i32);
    for y in 0..H {
        for x in 0..W {
            let (dx, dy) = (x as f32 - W as f32 / 2.0, y as f32 - H as f32 / 2.0);
            if dx * dx + dy * dy > radius * radius {
                continue;
            }
            disc += 1;
            let (a, b) = (px(&dark, x, y), px(&lit, x, y));
            let before: i32 = a.iter().map(|&v| v as i32).sum();
            let after: i32 = b.iter().map(|&v| v as i32).sum();
            if after - before > 20 {
                changed += 1;
            }
            // The night side, judged by what the pixel does with no night map.
            if before < 24 {
                if after - before > 20 {
                    night_lit += 1;
                }
                night_peak_dark = night_peak_dark.max(before);
                night_peak_lit = night_peak_lit.max(after);
            }
        }
    }
    assert!(disc > 10_000, "the planet is not filling the frame");
    assert!(night_lit > 20, "no city lights on the night side at all");
    // Emissive is added regardless of lighting, so lights do also land on the
    // day side — over land, where the map says a city is. What must not happen
    // is the whole sphere lifting: an emissive colour with no map multiplying
    // it floods every texel and turns the planet white.
    let rate = changed as f64 / disc as f64;
    assert!(
        rate < 0.10,
        "{:.1}% of the disc changed — is the emissive masked by the map at all?",
        rate * 100.0
    );
    // And on the night side the lights are the only thing there is to see.
    assert!(
        night_peak_lit > night_peak_dark * 3,
        "the night side barely brightened: {night_peak_dark} → {night_peak_lit}"
    );
}

#[test]
fn clouds_let_the_planet_through() {
    let Some(mut r) = renderer() else { return };
    let maps = generate_earth_maps(256);
    let build = |with_clouds: bool| {
        let mut m = maps.clone();
        if !with_clouds {
            m.clouds = None;
        }
        let mut scene = lit_scene();
        Planet::earth().maps(m).atmosphere(None).add_to(&mut scene);
        scene
    };
    let cam = camera(SUN, 3.2);
    let bare = r.render_to_rgba(&mut build(false), &cam);
    let cloudy = r.render_to_rgba(&mut build(true), &cam);

    // A cloud shell at full opacity never reads its map's alpha and renders as
    // a solid white ball. Some of the surface has to survive.
    let differing = (0..W * H)
        .filter(|i| {
            let (x, y) = (i % W, i / W);
            px(&bare, x, y) != px(&cloudy, x, y)
        })
        .count();
    let total = (W * H) as usize;
    assert!(differing > total / 100, "the cloud shell did nothing");
    assert!(
        differing < total * 9 / 10,
        "the cloud shell covered everything: {differing}/{total} pixels changed"
    );
}

#[test]
fn a_starfield_sits_behind_the_planet_and_is_not_lit_by_the_sun() {
    let Some(mut r) = renderer() else { return };
    let mut scene = lit_scene();
    Planet::new(1.0)
        .maps(PlanetMaps::new())
        .color(Color::from_hex(0x804020))
        .add_to(&mut scene);
    Starfield::procedural(512).radius(40.0).add_to(&mut scene);

    let img = r.render_to_rgba(&mut scene, &camera(SUN, 4.0));
    // The planet still occludes the sky.
    let centre = px(&img, W / 2, H / 2);
    assert!(
        centre[0] > centre[2],
        "the planet is not in front of the sky"
    );
    // And the sky itself is mostly empty — an unlit sky box that came out grey
    // would mean the sphere is being shaded rather than drawn flat.
    let corner_mean: f64 = (0..20)
        .flat_map(|y| (0..20).map(move |x| (x, y)))
        .map(|(x, y)| px(&img, x, y).iter().map(|&v| v as u64).sum::<u64>() as f64 / 3.0)
        .sum::<f64>()
        / 400.0;
    assert!(corner_mean < 40.0, "the sky is not dark: {corner_mean:.1}");
}
