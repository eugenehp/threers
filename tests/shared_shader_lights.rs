//! `shader!`'s lights, shadows and textures draw the same on wgpu and the
//! native Metal backend: a ground lit by the sun, a point light and the sky,
//! shadowed by a box, and a plane showing its own map.

#![cfg(metal_backend)]

use std::sync::Arc;
use threers::lights::{DirectionalLight, HemisphereLight, Light, PointLight};
use threers::RenderBackend;
use threers::{
    BoxGeometry, Color, HeadlessRenderer, Material, Mesh, Object3D, PerspectiveCamera,
    PlaneGeometry, Scene, ShaderMaterial, StandardMaterial, Texture, TextureFormat, Vector3,
};

const W: u32 = 128;
const H: u32 = 96;

fn build(metal: bool) -> HeadlessRenderer {
    let mut hr = HeadlessRenderer::builder()
        .backend(if metal {
            RenderBackend::Metal
        } else {
            RenderBackend::Wgpu
        })
        .size(W, H)
        .build()
        .expect("headless renderer");
    hr.set_msaa(1);
    assert_eq!(hr.is_native_metal(), metal);
    hr
}

fn lit_ground() -> ShaderMaterial {
    let shared = threers::shader! {
        fn shade(s: Shading) -> vec4f {
            let base = tex0(s.uv).xyz * data(0u32).xyz;
            var lit = (ambient_light() + hemisphere_light(s.normal)) * base;
            for (var i = 0u32; i < light_count(); i += 1u32) {
                let d = max(dot(s.normal, light_direction(i)), 0.0);
                lit += base * light_color(i) * d * directional_shadow(i, s);
            }
            for (var i = 0u32; i < point_light_count(); i += 1u32) {
                let d = max(dot(s.normal, point_light_direction(i, s)), 0.0);
                lit += base * point_light_color(i, s) * d * point_shadow(i, s);
            }
            return vec4f(lit, 1.0);
        }
    };
    let m = ShaderMaterial::from_shared(&shared).with_data(vec![[0.8, 0.8, 0.8, 0.0]]);
    m.validate(true).expect("the shared shader's WGSL");
    m
}

/// The ground under a box, the sun casting (or not), a point light off to
/// one side and a hemisphere light.
fn ground_scene(cast: bool, receive: bool, point: bool) -> Scene {
    let mut s = Scene::new();
    s.background = Color::new(0.0, 0.0, 0.0);
    s.add_light(HemisphereLight::new(
        Color::new(0.1, 0.12, 0.2),
        Color::new(0.05, 0.03, 0.0),
        1.0,
    ));
    let mut sun = DirectionalLight::new(Color::WHITE, 1.5);
    sun.cast_shadow = true;
    sun.shadow.map_size = 1024;
    sun.shadow.camera_size = 6.0;
    let mut l = Object3D::light(Light::Directional(sun));
    l.position = Vector3::new(-3.0, 6.0, 1.0);
    s.add(l);
    if point {
        let mut p = Object3D::light(Light::Point(PointLight::new(
            Color::new(1.0, 0.5, 0.2),
            4.0,
        )));
        p.position = Vector3::new(2.0, 0.6, 1.0);
        s.add(p);
    }
    let mut ground = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(8.0, 8.0),
        Material::Shader(lit_ground()),
    ));
    ground.rotate_x(-std::f32::consts::FRAC_PI_2);
    ground.receive_shadow = receive;
    s.add(ground);
    let mut block = Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.0, 1.0, 1.0),
        Material::Standard(StandardMaterial::new(Color::new(0.3, 0.3, 0.3))),
    ));
    block.position = Vector3::new(0.0, 1.2, 0.0);
    block.cast_shadow = cast;
    s.add(block);
    s
}

/// A plane facing the camera that shows its map and nothing else.
fn map_scene() -> Scene {
    let n = 8u32;
    let pixels: Vec<u8> = (0..n * n)
        .flat_map(|i| {
            if ((i % n) / 4 + (i / n) / 4).is_multiple_of(2) {
                [220, 40, 40, 255]
            } else {
                [40, 40, 220, 255]
            }
        })
        .collect();
    let mut checker = Texture::new(n, n, TextureFormat::Rgba8Unorm, pixels);
    checker.mag_filter = threers::TextureFilter::Nearest;
    let shared = threers::shader! {
        fn shade(s: Shading) -> vec4f {
            let size = vec2f(tex0_size());
            return vec4f(tex0_level(s.uv, 0.0).xyz * select(0.0, 1.0, size.x == 8.0), 1.0);
        }
    };
    let m = ShaderMaterial::from_shared(&shared).with_maps(vec![Arc::new(checker)]);
    m.validate(true).expect("the shared shader's WGSL");
    let mut s = Scene::new();
    s.background = Color::new(0.0, 0.0, 0.0);
    s.add(Object3D::mesh(Mesh::new(
        PlaneGeometry::new(2.0, 2.0),
        Material::Shader(m),
    )));
    s
}

fn diff(a: &[u8], b: &[u8]) -> (f64, u8) {
    let mut d: Vec<u8> = a.iter().zip(b).map(|(x, y)| x.abs_diff(*y)).collect();
    let mean = d.iter().map(|&v| v as f64).sum::<f64>() / d.len() as f64;
    d.sort_unstable();
    (mean, d[d.len() * 99 / 100])
}

/// Pixels at least `by` darker (red channel) in `b` than in `a`.
fn darker(a: &[u8], b: &[u8], by: u8) -> usize {
    a.as_chunks::<4>()
        .0
        .iter()
        .zip(b.as_chunks::<4>().0)
        .filter(|(p, q)| p[0] > q[0].saturating_add(by))
        .count()
}

#[test]
fn shared_lights_shadows_and_maps_match_across_backends() {
    if threers::metal::MetalDevice::new().is_err() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let mut cam = PerspectiveCamera::new(50.0, W as f32 / H as f32, 0.1, 50.0);
    cam.position = Vector3::new(0.0, 5.0, 6.0);
    cam.look_at(Vector3::ZERO);
    let mut flat = PerspectiveCamera::new(50.0, W as f32 / H as f32, 0.1, 50.0);
    flat.position = Vector3::new(0.0, 0.0, 2.2);
    flat.look_at(Vector3::ZERO);

    let mut frames = Vec::new();
    for metal in [false, true] {
        let mut hr = build(metal);
        let mut f =
            |cast, receive, point| hr.render_to_rgba(&mut ground_scene(cast, receive, point), &cam);
        let (shadowed, unshadowed, unreceived, no_point) = (
            f(true, true, true),
            f(false, true, true),
            f(true, false, true),
            f(true, true, false),
        );
        let map = hr.render_to_rgba(&mut map_scene(), &flat);
        assert!(
            hr.is_native_metal() == metal,
            "the scenes kept the renderer on its backend"
        );
        frames.push([shadowed, unshadowed, unreceived, no_point, map]);
    }
    for (backend, [shadowed, unshadowed, unreceived, no_point, map]) in
        ["wgpu", "Metal"].iter().zip(&frames)
    {
        let shade = darker(unshadowed, shadowed, 30);
        assert!(
            shade > 60,
            "{backend}: the box casts a shadow of {shade} px"
        );
        let ignored = darker(unreceived, shadowed, 30);
        assert!(
            ignored + 50 > shade,
            "{backend}: receive_shadow off still shadows"
        );
        assert_eq!(
            darker(unshadowed, unreceived, 30),
            0,
            "{backend}: an unreceived shadow darkens"
        );
        let warm = darker(shadowed, no_point, 30);
        assert!(warm > 100, "{backend}: the point light lights {warm} px");
        let centre = |x: u32, y: u32| {
            let i = ((y * W + x) * 4) as usize;
            [map[i], map[i + 1], map[i + 2]]
        };
        let (a, b) = (
            centre(W / 2 - 10, H / 2 - 10),
            centre(W / 2 + 10, H / 2 - 10),
        );
        // A data (`Unorm`) map reads as it is stored; the target encodes it:
        // 220 shows as 239, 40 as 110.
        let near = |v: u8, want: u8| v.abs_diff(want) <= 2;
        assert!(
            near(a[0], 239) && near(a[2], 110) && near(b[0], 110) && near(b[2], 239),
            "{backend}: the checker reads {a:?} {b:?}"
        );
    }
    for (k, what) in [
        "shadowed",
        "unshadowed",
        "unreceived",
        "no point light",
        "map",
    ]
    .iter()
    .enumerate()
    {
        let (mean, p99) = diff(&frames[0][k], &frames[1][k]);
        assert!(
            mean < 1.5 && p99 < 24,
            "{what}: wgpu and Metal differ by {mean:.2} mean, {p99} p99"
        );
    }
}
