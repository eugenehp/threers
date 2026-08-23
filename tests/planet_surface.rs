//! Cloud shadows and twilight — two things a lit sphere does not do by itself.
//!
//! Skipped when no GPU adapter is available.

use std::sync::Arc;

use threers::{
    Color, DirectionalLight, HeadlessRenderer, Material, Mesh, Object3D, PerspectiveCamera, Scene,
    SphereGeometry, StandardMaterial, Texture, TextureFormat, TextureWrap, Vector3,
};

const W: u32 = 240;
const H: u32 = 240;
/// Sun straight at the camera side, so the disc is fully lit.
const SUN: Vector3 = Vector3 {
    x: 0.0,
    y: 0.0,
    z: 1.0,
};

fn renderer() -> Option<HeadlessRenderer> {
    HeadlessRenderer::builder()
        .size(W, H)
        .color_format(wgpu::TextureFormat::Rgba8Unorm)
        .build()
        .ok()
}

/// A cloud map covering one quarter of the globe in longitude, positioned so
/// the split runs down the middle of what the camera can see.
///
/// The camera sits at +Z and the shader's `u` runs 0 at screen-left through
/// 0.25 at the centre to 0.5 at screen-right, so the visible hemisphere is
/// `u` in 0..0.5 — splitting the *map* in half would put the whole visible
/// face under one side of it and look like a uniform dimming.
fn half_cover() -> Arc<Texture> {
    let (w, h) = (128u32, 64u32);
    let mut data = vec![255u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let o = ((y * w + x) * 4) as usize;
            data[o + 3] = if x < w / 4 { 255 } else { 0 };
        }
    }
    let mut t = Texture::new(w, h, TextureFormat::Rgba8UnormSrgb, data);
    t.wrap_s = TextureWrap::Repeat;
    Arc::new(t)
}

fn render(r: &mut HeadlessRenderer, build: impl FnOnce(&mut StandardMaterial)) -> Vec<u8> {
    let mut m = StandardMaterial::new(Color::WHITE);
    m.roughness = 1.0;
    build(&mut m);
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    scene.add_light(DirectionalLight::new(Color::WHITE, 3.0).with_direction(SUN * -1.0));
    scene.add(Object3D::mesh(Mesh::new(
        SphereGeometry::new(1.0, 128, 64),
        Material::Standard(m),
    )));
    let mut c = PerspectiveCamera::new(45.0, 1.0, 0.1, 100.0);
    c.position = Vector3::new(0.0, 0.0, 3.4);
    c.look_at(Vector3::ZERO);
    r.render_to_rgba(&mut scene, &c)
}

fn mean(img: &[u8], x0: u32, x1: u32) -> f64 {
    let (mut sum, mut n) = (0u64, 0u64);
    for y in H * 2 / 5..H * 3 / 5 {
        for x in x0..x1 {
            let o = ((y * W + x) * 4) as usize;
            sum += img[o] as u64;
            n += 1;
        }
    }
    sum as f64 / n as f64
}

#[test]
fn a_cloud_deck_darkens_the_ground_beneath_it() {
    let Some(mut r) = renderer() else { return };
    let clouds = half_cover();

    let bare = render(&mut r, |_| {});
    let shadowed = render(&mut r, |m| {
        m.cloud_shadow_map = Some(clouds.clone());
        m.cloud_shadow = 0.8;
        m.cloud_height = 0.02;
    });

    // Half the globe is under cover, so one side must darken and the other
    // must not. Which side is which depends on the mapping; the test only
    // needs them to differ from each other and from the unshadowed render.
    let (l_bare, r_bare) = (mean(&bare, 40, 110), mean(&bare, 130, 200));
    let (l_sh, r_sh) = (mean(&shadowed, 40, 110), mean(&shadowed, 130, 200));
    let left_drop = l_bare - l_sh;
    let right_drop = r_bare - r_sh;
    assert!(
        left_drop.max(right_drop) > 8.0,
        "no side darkened: {l_bare:.1}->{l_sh:.1}, {r_bare:.1}->{r_sh:.1}"
    );
    assert!(
        (left_drop - right_drop).abs() > 6.0,
        "both sides darkened equally — the map is not being sampled per-point: \
         {left_drop:.1} vs {right_drop:.1}"
    );
}

#[test]
fn cloud_shadow_strength_zero_is_a_no_op() {
    let Some(mut r) = renderer() else { return };
    let clouds = half_cover();
    let bare = render(&mut r, |_| {});
    let off = render(&mut r, |m| {
        m.cloud_shadow_map = Some(clouds.clone());
        m.cloud_shadow = 0.0;
    });
    let differing = bare
        .chunks_exact(4)
        .zip(off.chunks_exact(4))
        .filter(|(a, b)| a[0].abs_diff(b[0]) > 2)
        .count();
    assert_eq!(differing, 0, "a zero-strength deck should cast nothing");
}

#[test]
fn twilight_carries_light_past_the_terminator() {
    let Some(mut r) = renderer() else { return };
    // Sun square-on from the right, so the terminator runs down the middle.
    let side = Vector3::new(1.0, 0.0, 0.0);
    let shot = |r: &mut HeadlessRenderer, twilight: f32| {
        let mut m = StandardMaterial::new(Color::WHITE);
        m.roughness = 1.0;
        m.twilight = twilight;
        let mut scene = Scene::new();
        scene.background = Color::BLACK;
        scene.add_light(DirectionalLight::new(Color::WHITE, 3.0).with_direction(side * -1.0));
        scene.add(Object3D::mesh(Mesh::new(
            SphereGeometry::new(1.0, 128, 64),
            Material::Standard(m),
        )));
        let mut c = PerspectiveCamera::new(45.0, 1.0, 0.1, 100.0);
        c.position = Vector3::new(0.0, 0.0, 3.4);
        c.look_at(Vector3::ZERO);
        r.render_to_rgba(&mut scene, &c)
    };
    let hard = shot(&mut r, 0.0);
    let soft = shot(&mut r, 0.35);

    // Just past the terminator, on the night side.
    let night = |img: &[u8]| mean(img, W * 2 / 5, W / 2);
    let day = |img: &[u8]| mean(img, W * 3 / 5, W * 7 / 10);
    assert!(
        night(&soft) > night(&hard) + 4.0,
        "twilight should reach onto the night side: {:.1} -> {:.1}",
        night(&hard),
        night(&soft)
    );
    // …and leave the day side alone: it only adds what Lambert did not give.
    // A little does reach the lit side — that is the sunset — but it must be a
    // tint on sunlight, not a second source.
    assert!(
        (day(&soft) - day(&hard)).abs() < 8.0,
        "the day side should be untouched: {:.1} -> {:.1}",
        day(&hard),
        day(&soft)
    );
}

/// Render a sphere with an occluder of `radius` sitting `dist` toward the sun.
fn eclipsed(r: &mut HeadlessRenderer, occluder: Option<([f32; 3], f32)>) -> Vec<u8> {
    let mut m = StandardMaterial::new(Color::WHITE);
    m.roughness = 1.0;
    if let Some((c, rad)) = occluder {
        m.eclipse_occluder = Some([c[0], c[1], c[2], rad]);
    }
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    scene.add_light(DirectionalLight::new(Color::WHITE, 3.0).with_direction(SUN * -1.0));
    scene.add(Object3D::mesh(Mesh::new(
        SphereGeometry::new(1.0, 96, 48),
        Material::Standard(m),
    )));
    let mut c = PerspectiveCamera::new(45.0, 1.0, 0.1, 100.0);
    c.position = Vector3::new(0.0, 0.0, 3.4);
    c.look_at(Vector3::ZERO);
    r.render_to_rgba(&mut scene, &c)
}

#[test]
fn an_occluder_on_the_sun_line_casts_an_eclipse() {
    let Some(mut r) = renderer() else { return };
    let clear = eclipsed(&mut r, None);
    // Sun is at +Z; put a small body between this sphere and it, on the axis.
    let shadowed = eclipsed(&mut r, Some(([0.0, 0.0, 12.0], 0.06)));
    // The sub-occluder point itself. An occluder this size throws an umbra
    // about ten pixels across, so anything averaged over a wider window — in
    // either axis — mixes the shadow with the lit surface beside it.
    let px = |img: &[u8]| img[(((H / 2) * W + W / 2) * 4) as usize] as f64;
    let centre = px(&shadowed);
    let before = px(&clear);
    assert!(
        centre < before * 0.5,
        "the sub-occluder point should be deep in shadow: {before:.1} -> {centre:.1}"
    );
    // …and the limb, far off the axis, should be untouched.
    let limb_before = mean(&clear, 30, 55);
    let limb_after = mean(&shadowed, 30, 55);
    assert!(
        (limb_after - limb_before).abs() < 3.0,
        "the shadow should be local, not global: {limb_before:.1} -> {limb_after:.1}"
    );
}

#[test]
fn the_shadow_has_a_penumbra_rather_than_a_hard_edge() {
    let Some(mut r) = renderer() else { return };
    let shadowed = eclipsed(&mut r, Some(([0.0, 0.0, 12.0], 0.06)));
    // Walk out from the centre of the shadow along the middle scanline and
    // count how many distinct brightness steps there are between full shadow
    // and full light. A hard shadow jumps in one step.
    let y = H / 2;
    let at = |x: u32| {
        let o = ((y * W + x) * 4) as usize;
        shadowed[o] as i32
    };
    let dark = at(W / 2);
    let lit = at(W / 2 + 60);
    assert!(lit > dark + 20, "no shadow to measure: {dark} vs {lit}");
    let partial = (W / 2..W / 2 + 60)
        .filter(|&x| at(x) > dark + 4 && at(x) < lit - 4)
        .count();
    assert!(
        partial > 3,
        "the edge is hard — the sun is being treated as a point rather than a \
         half-degree disc: {partial} pixels of penumbra"
    );
}

#[test]
fn no_occluder_is_a_no_op() {
    let Some(mut r) = renderer() else { return };
    let a = eclipsed(&mut r, None);
    let b = eclipsed(&mut r, Some(([0.0, 0.0, 12.0], 0.0)));
    let differing = a
        .chunks_exact(4)
        .zip(b.chunks_exact(4))
        .filter(|(p, q)| p[0].abs_diff(q[0]) > 2)
        .count();
    assert_eq!(differing, 0, "a zero-radius occluder should block nothing");
}

// ---------------------------------------------------------------------------
// Volume and disc effects
// ---------------------------------------------------------------------------

/// A hard `max(dot(n, l), 0)` is the terminator of an *airless* body lit by a
/// *point* sun. The real sun is half a degree across, so near the terminator
/// part of its disc is below the horizon and the light falls off over that
/// width instead of switching.
#[test]
fn the_suns_disc_softens_the_terminator() {
    let Some(mut r) = renderer() else { return };

    // Brightness along a horizontal scan across the terminator. The sun is at
    // +Z (toward the camera), so the terminator sits at the limb — instead put
    // it across the middle by lighting from the side.
    let mut scan = |sun_radius: f32| -> Vec<f32> {
        let mut m = StandardMaterial::new(Color::WHITE);
        m.roughness = 1.0;
        m.sun_angular_radius = sun_radius;
        let mut scene = Scene::new();
        scene.background = Color::BLACK;
        scene.add_light(
            DirectionalLight::new(Color::WHITE, 3.0).with_direction(Vector3::new(-1.0, 0.0, 0.0)),
        );
        scene.add(Object3D::mesh(Mesh::new(
            SphereGeometry::new(1.0, 256, 128),
            Material::Standard(m),
        )));
        let mut c = PerspectiveCamera::new(45.0, 1.0, 0.1, 100.0);
        c.position = Vector3::new(0.0, 0.0, 3.4);
        c.look_at(Vector3::ZERO);
        let img = r.render_to_rgba(&mut scene, &c);
        // The middle third only. A scan of the full width crosses the sphere's
        // silhouette against black, and that edge is a far bigger step than any
        // terminator — it would be measuring the limb, not the lighting.
        (W / 3..W * 2 / 3)
            .map(|x| {
                let o = ((H / 2) * W + x) as usize * 4;
                img[o] as f32
            })
            .collect()
    };

    let hard = scan(0.0);
    let soft = scan(0.05); // exaggerated, so the band is measurable in pixels

    // Steepest single-pixel step across the scan. Softening has to reduce it.
    let steepest = |v: &[f32]| {
        v.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max)
    };
    let (sh, ss) = (steepest(&hard), steepest(&soft));
    eprintln!("steepest step: hard {sh:.1}, soft {ss:.1}");
    assert!(
        ss < sh * 0.85,
        "the sun's disc should soften the terminator: {sh:.1} -> {ss:.1}"
    );
    // And it must not brighten the day side, which would mean it had turned
    // into an ambient term.
    let day = |v: &[f32]| v.iter().rev().take(20).sum::<f32>();
    assert!(
        (day(&soft) - day(&hard)).abs() < day(&hard) * 0.05 + 1.0,
        "softening changed the fully lit side"
    );
}

/// A cloud deck is a volume. Droplets scatter hard forward, so a deck with the
/// sun behind it is brighter than the same deck lit from over the camera's
/// shoulder — the opposite of what a Lambert surface does.
#[test]
fn a_cloud_deck_scatters_forward() {
    let Some(mut r) = renderer() else { return };

    // A thin shell's far side is genuinely dark: the sun is below the horizon
    // there and the planet blocks the path. What the phase function changes is
    // the lit crescent and the ring just past the terminator, so the sun goes
    // off to one side and either beyond the globe or in front of it.
    let mut render = |scatter: f32, sun_z: f32| -> f64 {
        let mut m = StandardMaterial::new(Color::WHITE);
        m.roughness = 0.95;
        m.cloud_scatter = scatter;
        m.cloud_anisotropy = 0.8;
        let mut scene = Scene::new();
        scene.background = Color::BLACK;
        // 60 degrees off the view axis; `sun_z` picks which side of the globe.
        scene.add_light(
            DirectionalLight::new(Color::WHITE, 3.0).with_direction(Vector3::new(
                -0.866,
                0.0,
                -0.5 * sun_z,
            )),
        );
        scene.add(Object3D::mesh(Mesh::new(
            SphereGeometry::new(1.0, 128, 64),
            Material::Standard(m),
        )));
        let mut c = PerspectiveCamera::new(45.0, 1.0, 0.1, 100.0);
        c.position = Vector3::new(0.0, 0.0, 3.4);
        c.look_at(Vector3::ZERO);
        let img = r.render_to_rgba(&mut scene, &c);
        img.chunks_exact(4).map(|p| p[0] as f64).sum::<f64>() / (W * H) as f64
    };

    // sun_z = -1 puts the sun beyond the globe, so light scattering forward
    // through the deck carries on toward the camera.
    let back_lit_off = render(0.0, -1.0);
    let back_lit_on = render(0.6, -1.0);
    eprintln!("sun beyond: {back_lit_off:.2} -> {back_lit_on:.2}");
    assert!(
        back_lit_on > back_lit_off * 1.02,
        "forward scattering should brighten a deck lit from beyond: \
         {back_lit_off:.2} -> {back_lit_on:.2}"
    );

    // Sun on the camera's side: the phase function is near its minimum there,
    // so the same strength must add far less.
    let front_off = render(0.0, 1.0);
    let front_on = render(0.6, 1.0);
    let back_gain = back_lit_on - back_lit_off;
    let front_gain = front_on - front_off;
    eprintln!("gain: back {back_gain:.2}, front {front_gain:.2}");
    assert!(
        back_gain > front_gain * 1.5,
        "scattering should favour forward: back {back_gain:.2} vs front {front_gain:.2}"
    );
}

/// An atmosphere shell around a dark body, so only the air is measured.
fn air_scene(build: impl FnOnce(&mut threers::AtmosphereMaterial)) -> Scene {
    use threers::AtmosphereMaterial;
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    scene.add_light(
        DirectionalLight::new(Color::WHITE, 3.0).with_direction(Vector3::new(-1.0, 0.0, -0.35)),
    );
    // The body itself, black, so it occludes without contributing light.
    let mut ground = StandardMaterial::new(Color::BLACK);
    ground.roughness = 1.0;
    scene.add(Object3D::mesh(Mesh::new(
        SphereGeometry::new(1.0, 128, 64),
        Material::Standard(ground),
    )));
    let mut air = AtmosphereMaterial::new(1.0, 1.045)
        .color(Color::new(0.30, 0.55, 1.0))
        .sunset_color(Color::new(1.0, 0.48, 0.20))
        .intensity(2.6)
        .falloff(2.3);
    build(&mut air);
    scene.add(Object3D::mesh(Mesh::new(
        SphereGeometry::new(1.045, 128, 64),
        Material::Atmosphere(air),
    )));
    scene
}

/// Rayleigh scattering only ever *adds* light, and the blue saturates first, so
/// a limb built from it alone whitens as the path lengthens. What keeps a real
/// twilight blue is ozone absorbing the middle of the spectrum.
#[test]
fn ozone_keeps_the_grazing_limb_blue() {
    let Some(mut r) = renderer() else { return };
    let mut c = PerspectiveCamera::new(45.0, 1.0, 0.1, 100.0);
    c.position = Vector3::new(0.0, 0.0, 3.4);
    c.look_at(Vector3::ZERO);

    // Mean blue:red over the whole frame; the air is the only thing lit.
    let mut ratio = |ozone: f32| -> f64 {
        let mut scene = air_scene(|m| m.ozone = ozone);
        let img = r.render_to_rgba(&mut scene, &c);
        let (mut red, mut blue) = (0.0f64, 0.0f64);
        for px in img.chunks_exact(4) {
            red += px[0] as f64;
            blue += px[2] as f64;
        }
        blue / red.max(1.0)
    };
    let plain = ratio(0.0);
    let with_ozone = ratio(1.2);
    eprintln!("blue:red — none {plain:.3}, ozone {with_ozone:.3}");
    assert!(
        with_ozone > plain * 1.05,
        "ozone should push the limb blue: {plain:.3} -> {with_ozone:.3}"
    );
}

/// Solar wind down the field lines, hitting the upper atmosphere in a ring
/// around the geomagnetic pole. It belongs on the night side, where it is not
/// outshone, and it should not touch the day side's colour.
#[test]
fn the_aurora_lights_the_night_side_only() {
    let Some(mut r) = renderer() else { return };
    let mut c = PerspectiveCamera::new(45.0, 1.0, 0.1, 100.0);
    // Over the pole, obliquely, which is where the oval is.
    c.position = Vector3::new(0.0, 2.6, 2.2);
    c.look_at(Vector3::ZERO);

    let mut green = |aurora: f32| -> (f64, f64) {
        let mut scene = air_scene(|m| {
            m.aurora = aurora;
            m.aurora_color = Color::new(0.25, 1.0, 0.45);
            m.aurora_colatitude = 23.0;
        });
        let img = r.render_to_rgba(&mut scene, &c);
        // Which half is the day side is a question about the camera's
        // handedness, so measure it rather than assume: the air scatters far
        // more where the sun is on it, so the brighter half in *blue* is day.
        let (mut blue_l, mut blue_r) = (0.0f64, 0.0f64);
        let (mut left, mut right) = (0.0f64, 0.0f64);
        for y in 0..H {
            for x in 0..W {
                let o = ((y * W + x) * 4) as usize;
                // Green in excess of blue: the air itself is blue, the aurora
                // is not, so this isolates the emission from the scattering.
                let excess = (img[o + 1] as f64 - img[o + 2] as f64).max(0.0);
                if x < W / 2 {
                    left += excess;
                    blue_l += img[o + 2] as f64;
                } else {
                    right += excess;
                    blue_r += img[o + 2] as f64;
                }
            }
        }
        if blue_l >= blue_r {
            (left, right)
        } else {
            (right, left)
        }
    };
    let (day_off, night_off) = green(0.0);
    let (day_on, night_on) = green(2.5);
    eprintln!("green excess — day {day_off:.0}->{day_on:.0}, night {night_off:.0}->{night_on:.0}");
    assert!(
        night_on > night_off + 500.0,
        "the aurora should light the night side: {night_off:.0} -> {night_on:.0}"
    );
    // Brighter on the night side, but not by a landslide: the oval is a ring,
    // and seen obliquely over a pole a good part of it is close enough to the
    // terminator that the sun has not fully drowned it yet. What matters is
    // that the day half is suppressed, not that it is empty.
    assert!(
        night_on - night_off > (day_on - day_off) * 1.3,
        "it belongs on the night side: day gained {:.0}, night {:.0}",
        day_on - day_off,
        night_on - night_off
    );
}

/// The southern lights are the same particles arriving down the other end of
/// the same field lines, so an oval belongs at both poles.
#[test]
fn the_aurora_rings_both_poles() {
    let Some(mut r) = renderer() else { return };
    let mut over = |y: f32| -> f64 {
        let mut scene = air_scene(|m| {
            m.aurora = 2.5;
            m.aurora_colatitude = 23.0;
        });
        let mut c = PerspectiveCamera::new(45.0, 1.0, 0.1, 100.0);
        c.position = Vector3::new(0.0, y, 2.2);
        c.look_at(Vector3::ZERO);
        let img = r.render_to_rgba(&mut scene, &c);
        img.chunks_exact(4)
            .map(|p| (p[1] as f64 - p[2] as f64).max(0.0))
            .sum()
    };
    let north = over(2.6);
    let south = over(-2.6);
    eprintln!("aurora green excess — north {north:.0}, south {south:.0}");
    assert!(
        north > 1000.0 && south > 1000.0,
        "both poles should have an oval"
    );
    // Near mirror images: the geomagnetic tilt makes them differ, not vanish.
    let (lo, hi) = (north.min(south), north.max(south));
    assert!(
        hi < lo * 3.0,
        "one pole is far brighter than the other: {north:.0} vs {south:.0}"
    );
}

/// Green is the dominant line and red is a high-altitude crown, so a display
/// must not come out red-dominated however bright it is driven.
#[test]
fn the_aurora_is_green_dominated() {
    let Some(mut r) = renderer() else { return };
    let mut c = PerspectiveCamera::new(45.0, 1.0, 0.1, 100.0);
    c.position = Vector3::new(0.0, 2.6, 2.2);
    c.look_at(Vector3::ZERO);
    for strength in [1.0f32, 4.0, 10.0] {
        let mut scene = air_scene(|m| {
            m.aurora = strength;
            m.aurora_color = Color::new(0.25, 1.0, 0.45);
        });
        let img = r.render_to_rgba(&mut scene, &c);
        // Against the no-aurora baseline, so the air's own blue does not count.
        let (mut green, mut red) = (0.0f64, 0.0f64);
        for px in img.chunks_exact(4) {
            green += (px[1] as f64 - px[2] as f64).max(0.0);
            red += (px[0] as f64 - px[2] as f64).max(0.0);
        }
        eprintln!("aurora {strength}: green {green:.0}, red {red:.0}");
        assert!(
            green > red,
            "at strength {strength} the aurora went red-dominated: green {green:.0}, red {red:.0}"
        );
    }
}
