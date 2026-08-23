//! Whether a UV sphere's pole is a *rendering* problem. It is not.
//!
//! An equirectangular map wrapped on a UV sphere compresses `u` by
//! `1/sin(colatitude)`, so a screen pixel next to the pole covers a UV
//! footprint enormously wider than it is tall. That is the standing suspicion
//! whenever the Arctic comes out looking like a fan of streaks: the sampler is
//! blamed for running the LOD away, or the derivative-built tangent frame for
//! going singular where `du/dscreen` explodes.
//!
//! These tests settle it by making the map a function of latitude *only*.
//! Viewed straight down the pole, with the light on the axis too, rotating the
//! sphere maps the scene onto itself — so the image has to come out
//! rotationally symmetric, and any variation around a ring is the renderer's
//! doing rather than the map's. Both come out clean, which is what redirected
//! the search to the imagery, where the real fault was: Blue Marble has no
//! Arctic (see [`threers::planet::blend_polar_cap`]).
//!
//! Skipped when no GPU adapter is available.

use std::sync::Arc;

use threers::{
    AmbientLight, BasicMaterial, Color, DirectionalLight, HeadlessRenderer, Material, Mesh,
    Object3D, PerspectiveCamera, Scene, SphereGeometry, StandardMaterial, Texture, TextureFormat,
    TextureWrap, Vector3,
};

const W: u32 = 400;
const H: u32 = 400;

fn renderer() -> Option<HeadlessRenderer> {
    HeadlessRenderer::builder()
        .size(W, H)
        .color_format(wgpu::TextureFormat::Rgba8Unorm)
        .build()
        .ok()
}

/// Latitude stripes: the value depends on the row and nothing else.
///
/// The stripes are what makes a runaway LOD visible. Blur along latitude washes
/// them toward their mean, so how much contrast survives reads out directly how
/// far up the mip chain the sampler drifted.
fn latitude_stripes(w: u32, h: u32, period: u32) -> Arc<Texture> {
    let mut data = vec![255u8; (w * h * 4) as usize];
    for y in 0..h {
        let v = if (y / period).is_multiple_of(2) {
            235
        } else {
            20
        };
        for x in 0..w {
            let o = ((y * w + x) * 4) as usize;
            data[o] = v;
            data[o + 1] = v;
            data[o + 2] = v;
        }
    }
    let mut t = Texture::new(w, h, TextureFormat::Rgba8UnormSrgb, data);
    t.wrap_s = TextureWrap::Repeat;
    t.wrap_t = TextureWrap::ClampToEdge;
    Arc::new(t)
}

/// A tangent-space normal that is the same everywhere, tilted off the surface
/// in both u and v so the frame's tangent *and* bitangent both have to be right.
fn constant_normal_map(w: u32, h: u32) -> Arc<Texture> {
    let n = [0.45f32, 0.30, 1.0];
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    let enc = |c: f32| ((c / len * 0.5 + 0.5) * 255.0).round() as u8;
    let mut data = vec![255u8; (w * h * 4) as usize];
    for i in 0..(w * h) as usize {
        data[i * 4] = enc(n[0]);
        data[i * 4 + 1] = enc(n[1]);
        data[i * 4 + 2] = enc(n[2]);
    }
    let mut t = Texture::new(w, h, TextureFormat::Rgba8Unorm, data);
    t.wrap_s = TextureWrap::Repeat;
    Arc::new(t)
}

/// Straight down the north pole, zoomed until the polar cap fills the frame.
///
/// `up` has to be off the view axis or `look_at` degenerates.
fn polar_camera() -> PerspectiveCamera {
    let mut c = PerspectiveCamera::new(12.0, 1.0, 0.01, 100.0);
    c.position = Vector3::new(0.0, 3.0, 0.0);
    c.up = Vector3::new(0.0, 0.0, -1.0);
    c.look_at(Vector3::ZERO);
    c
}

fn sphere(material: Material) -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    scene.add(Object3D::mesh(Mesh::new(
        SphereGeometry::new(1.0, 512, 256),
        material,
    )));
    scene
}

/// Unlit, so nothing depends on normals, tangents or lights — this isolates the
/// texture fetch from every other candidate explanation.
fn render_unlit(r: &mut HeadlessRenderer, map: Arc<Texture>) -> Vec<u8> {
    let m = BasicMaterial {
        map: Some(map),
        ..BasicMaterial::default()
    };
    let mut scene = sphere(Material::Basic(m));
    scene.add_light(AmbientLight::new(Color::WHITE, 1.0));
    r.render_to_rgba(&mut scene, &polar_camera())
}

/// Lit, with the light on the pole axis so the whole scene is symmetric about
/// it — otherwise the light itself would break the symmetry being measured.
fn render_lit(r: &mut HeadlessRenderer, normal_map: Arc<Texture>) -> Vec<u8> {
    let mut m = StandardMaterial::new(Color::WHITE);
    m.roughness = 0.6;
    m.metalness = 0.0;
    m.normal_map = Some(normal_map);
    let mut scene = sphere(Material::Standard(m));
    scene.add_light(
        DirectionalLight::new(Color::WHITE, 3.0).with_direction(Vector3::new(0.0, -1.0, 0.0)),
    );
    r.render_to_rgba(&mut scene, &polar_camera())
}

/// Bilinear, deliberately: rounding to the nearest pixel makes a ring's
/// effective radius wobble by half a pixel, and one stripe is only a few pixels
/// wide radially near the pole, so that wobble alone turns a *radial* edge into
/// what looks like azimuthal spread.
fn luma_at(img: &[u8], x: f32, y: f32) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let mut acc = 0.0;
    for (dx, dy, w) in [
        (0.0, 0.0, (1.0 - fx) * (1.0 - fy)),
        (1.0, 0.0, fx * (1.0 - fy)),
        (0.0, 1.0, (1.0 - fx) * fy),
        (1.0, 1.0, fx * fy),
    ] {
        let (xi, yi) = ((x0 + dx) as i32, (y0 + dy) as i32);
        if xi < 0 || yi < 0 || xi >= W as i32 || yi >= H as i32 {
            return f32::NAN;
        }
        acc += w * img[((yi as u32 * W + xi as u32) * 4) as usize] as f32;
    }
    acc
}

/// Amplitude of the strongest spoke pattern on a ring centred on the pole.
///
/// Harmonics start at 2 because harmonic 1 is what a sub-pixel error in the
/// assumed centre produces: offsetting the ring makes its true radius vary as
/// `cos(azimuth)`, which crossing a latitude edge turns into a one-cycle swing
/// worth tens of levels. A fan of spokes is not one-cycle, so dropping it costs
/// nothing and removes the measurement's own artefact.
fn spokes(img: &[u8], radius: f32) -> f32 {
    let (cx, cy) = (W as f32 / 2.0, H as f32 / 2.0);
    let n = 512usize;
    let vals: Vec<f32> = (0..n)
        .map(|i| {
            let a = i as f32 / n as f32 * std::f32::consts::TAU;
            luma_at(img, cx + radius * a.cos(), cy + radius * a.sin())
        })
        .collect();
    if vals.iter().any(|v| !v.is_finite()) {
        return 0.0;
    }
    (2..=48)
        .map(|k| {
            let (mut re, mut im) = (0.0f32, 0.0f32);
            for (i, v) in vals.iter().enumerate() {
                let a = i as f32 / n as f32 * std::f32::consts::TAU * k as f32;
                re += v * a.cos();
                im += v * a.sin();
            }
            2.0 * (re * re + im * im).sqrt() / n as f32
        })
        .fold(0.0, f32::max)
}

/// Rings from about 0.6 to 7 degrees of colatitude, where the `1/sin`
/// compression runs from roughly 8x to 100x — straddling the 16x anisotropy cap
/// the sampler is limited to, which is where a runaway LOD would show.
const RADII: [f32; 8] = [4.0, 6.0, 9.0, 13.0, 19.0, 27.0, 38.0, 52.0];

fn worst_spokes(img: &[u8]) -> f32 {
    RADII.iter().map(|&r| spokes(img, r)).fold(0.0, f32::max)
}

#[test]
fn a_texture_fetch_at_the_pole_has_no_azimuthal_structure() {
    let Some(mut r) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let img = render_unlit(&mut r, latitude_stripes(2048, 1024, 8));
    let worst = worst_spokes(&img);
    eprintln!("unlit spoke amplitude: {worst:.2} / 255");
    assert!(
        worst < 6.0,
        "a map constant along longitude must render rotationally symmetric down \
         the pole; got spokes of {worst:.1}/255"
    );
}

#[test]
fn b_normal_mapped_lighting_at_the_pole_has_no_azimuthal_structure() {
    let Some(mut r) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let img = render_lit(&mut r, constant_normal_map(2048, 1024));
    let worst = worst_spokes(&img);
    eprintln!("lit spoke amplitude: {worst:.2} / 255");
    assert!(
        worst < 6.0,
        "the tangent frame is built from screen-space derivatives, which go \
         ill-conditioned at the pole; if that mattered it would show as spokes. \
         Got {worst:.1}/255"
    );
}

#[test]
fn c_latitude_detail_survives_beside_the_pole() {
    let Some(mut r) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let img = render_unlit(&mut r, latitude_stripes(2048, 1024, 8));

    // Contrast along a radius, inside the zone where the footprint's aspect
    // ratio exceeds what the hardware will filter. A runaway LOD would collapse
    // the stripes toward a flat mid grey.
    let (cx, cy) = (W as f32 / 2.0, H as f32 / 2.0);
    let (mut lo, mut hi) = (255.0f32, 0.0f32);
    for i in 12..90 {
        let v = luma_at(&img, cx + i as f32, cy);
        lo = lo.min(v);
        hi = hi.max(v);
    }
    eprintln!("radial contrast beside pole: {lo:.0}..{hi:.0}");
    assert!(
        hi - lo > 60.0,
        "latitude stripes should still resolve next to the pole; got a range of \
         only {:.0}/255",
        hi - lo
    );
}

/// Orbiting straight over the pole used to black the whole frame out.
///
/// With the camera on the up axis the view direction is parallel to `up`, the
/// cross product that gives the right-hand axis is zero, and `normalize`
/// returns zero rather than NaN — so the view matrix came out rank one and
/// projected the scene onto a line. Nothing about it looked like an error; the
/// frame was simply empty.
#[test]
fn d_a_camera_on_the_pole_axis_still_renders() {
    let Some(mut r) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let map = latitude_stripes(512, 256, 8);
    let lit = |img: &[u8]| img.chunks_exact(4).filter(|p| p[0] > 8).count();

    // Just off the axis: the ordinary case, and the baseline to compare with.
    let m = BasicMaterial {
        map: Some(map.clone()),
        ..BasicMaterial::default()
    };
    let mut scene = sphere(Material::Basic(m));
    scene.add_light(AmbientLight::new(Color::WHITE, 1.0));
    let mut c = PerspectiveCamera::new(12.0, 1.0, 0.01, 100.0);
    c.position = Vector3::new(0.0, 3.0, 0.0);
    c.up = Vector3::new(0.0, 0.0, -1.0);
    c.look_at(Vector3::ZERO);
    let baseline = lit(&r.render_to_rgba(&mut scene, &c));
    assert!(baseline > 1000, "the baseline view is already empty");

    // Exactly on it, with the default up — what dragging to the pole produces.
    for up in [Vector3::new(0.0, 1.0, 0.0), Vector3::new(0.0, -1.0, 0.0)] {
        let m = BasicMaterial {
            map: Some(map.clone()),
            ..BasicMaterial::default()
        };
        let mut scene = sphere(Material::Basic(m));
        scene.add_light(AmbientLight::new(Color::WHITE, 1.0));
        let mut c = PerspectiveCamera::new(12.0, 1.0, 0.01, 100.0);
        c.position = Vector3::new(0.0, 3.0, 0.0);
        c.up = up;
        c.look_at(Vector3::ZERO);
        let got = lit(&r.render_to_rgba(&mut scene, &c));
        assert!(
            got as f32 > baseline as f32 * 0.9,
            "camera on the pole axis with up {up:?} rendered {got} lit pixels \
             against a baseline of {baseline}"
        );
    }
}

/// And the controls should not put it there in the first place.
#[test]
fn e_orbiting_stops_short_of_the_pole() {
    use threers::{OrbitControls, PointerEvent};
    let mut c = PerspectiveCamera::new(45.0, 1.0, 0.1, 100.0);
    c.position = Vector3::new(0.0, 0.0, 5.0);
    let mut controls = OrbitControls::new(&c);
    // Drag far past the top, then far past the bottom.
    for (dy, which) in [(400.0f32, "top"), (-800.0, "bottom")] {
        for _ in 0..12 {
            let ev = PointerEvent {
                rotating: true,
                dy,
                ..PointerEvent::default()
            };
            controls.update(ev, &mut c, (800.0, 800.0));
        }
        // The camera must not land on the up axis, where `look_at` has no
        // unique roll — measured through the pose, since the angle is private.
        let off_axis = (c.position.x * c.position.x + c.position.z * c.position.z).sqrt();
        assert!(
            off_axis > 1e-4,
            "orbiting past the {which} put the camera on the up axis: {:?}",
            c.position
        );
    }
}
