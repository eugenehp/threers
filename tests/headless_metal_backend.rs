//! `THREERS_BACKEND=metal`: the same `HeadlessRenderer` program drawn by wgpu
//! and by the native Metal backend comes back the same.
//!
//! One test, run in sequence: the backend is chosen from the environment when
//! a renderer is built, and the environment is the whole process's.

#![cfg(metal_backend)]

use threers::lights::{AmbientLight, DirectionalLight};
use threers::{
    BoxGeometry, Color, HeadlessRenderer, Mesh, Object3D, PerspectiveCamera, PlaneGeometry, Scene,
    StandardMaterial, Vector3,
};

const W: u32 = 96;

/// Blended draws into an sRGB target blend the encoded values on a device,
/// as wgpu does; the simulators have no programmable blending and blend in
/// linear light, a few levels off on soft edges. Comparisons of such draws
/// are reported there, not asserted.
const ENCODED_BLEND: bool = !cfg!(target_abi = "sim");

/// Assert the difference is within `(mean, p99)` — or, for a blended draw in
/// a simulator, report it.
fn check(what: &str, (mean, p99): (f64, u8), blended: bool, (max_mean, max_p99): (f64, u8)) {
    if blended && !ENCODED_BLEND {
        eprintln!("{what}: wgpu and Metal differ by {mean:.2} mean, {p99} p99 (simulator: linear-light blending)");
        return;
    }
    assert!(
        mean < max_mean && p99 < max_p99,
        "{what}: wgpu and Metal differ by {mean:.2} mean, {p99} at the 99th percentile"
    );
}
const H: u32 = 64;

fn scene() -> Scene {
    let mut s = Scene::new();
    // Dark and saturated, where a missing sRGB encode shows the most.
    s.background = Color::from_hex(0x10204a);
    s.add_light(AmbientLight::new(Color::WHITE, 0.3));
    s.add_light(DirectionalLight::new(Color::WHITE, 2.0).with_direction(Vector3::new(-0.4, -1.0, -0.6)));
    let mut floor = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(6.0, 6.0),
        StandardMaterial::new(Color::from_hex(0x336644)).into(),
    ));
    floor.rotate_x(-std::f32::consts::FRAC_PI_2);
    s.add(floor);
    let mut cube = Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.0, 1.0, 1.0),
        StandardMaterial::new(Color::from_hex(0xff8844)).with_roughness(0.5).into(),
    ));
    cube.position = Vector3::new(0.0, 0.5, 0.0);
    cube.name = "cube".into();
    s.add(cube);
    s
}

fn camera() -> PerspectiveCamera {
    let mut c = PerspectiveCamera::new(50.0, W as f32 / H as f32, 0.1, 100.0);
    c.position = Vector3::new(2.5, 2.0, 3.5);
    c.target = Vector3::new(0.0, 0.4, 0.0);
    c
}

fn build(metal: bool, format: wgpu::TextureFormat, ss: u32, msaa: u32) -> HeadlessRenderer {
    // SAFETY: this file has one test, so nothing else reads the environment
    // while it changes.
    unsafe {
        if metal {
            std::env::set_var("THREERS_BACKEND", "metal");
        } else {
            std::env::remove_var("THREERS_BACKEND");
        }
    }
    let mut hr = HeadlessRenderer::builder()
        .size(W, H)
        .supersample(ss)
        .color_format(format)
        .build()
        .expect("headless renderer");
    unsafe { std::env::remove_var("THREERS_BACKEND") };
    hr.set_msaa(msaa);
    assert_eq!(hr.is_native_metal(), metal, "backend for {format:?}");
    hr
}

/// Mean and 99th-percentile absolute difference over the colour channels.
fn diff(a: &[u8], b: &[u8]) -> (f64, u8) {
    assert_eq!(a.len(), b.len());
    let mut d: Vec<u8> = a
        .chunks_exact(4)
        .zip(b.chunks_exact(4))
        .flat_map(|(p, q)| (0..3).map(move |c| p[c].abs_diff(q[c])))
        .collect();
    let mean = d.iter().map(|&v| v as f64).sum::<f64>() / d.len() as f64;
    d.sort_unstable();
    (mean, d[d.len() * 99 / 100])
}

#[test]
fn the_metal_backend_draws_what_wgpu_draws() {
    if threers::metal::MetalDevice::new().is_err() {
        eprintln!("skipping: no Metal device");
        return;
    }
    for format in [wgpu::TextureFormat::Rgba8UnormSrgb, wgpu::TextureFormat::Rgba8Unorm] {
        for (ss, msaa) in [(1, 1), (1, 4), (2, 1)] {
            let frames: Vec<(Vec<u8>, Vec<u8>)> = [false, true]
                .into_iter()
                .map(|metal| {
                    let mut hr = build(metal, format, ss, msaa);
                    let mut s = scene();
                    let raw = hr.render_to_rgba(&mut s, &camera());
                    let resolved = hr.render_to_rgba_resolved(&mut s, &camera());
                    assert_eq!(raw.len(), (W * H * ss * ss * 4) as usize);
                    assert_eq!(resolved.len(), (W * H * 4) as usize);
                    (raw, resolved)
                })
                .collect();
            for (what, a, b) in [
                ("render_to_rgba", &frames[0].0, &frames[1].0),
                ("render_to_rgba_resolved", &frames[0].1, &frames[1].1),
            ] {
                let (mean, p99) = diff(a, b);
                assert!(
                    mean < 2.0 && p99 < 24,
                    "{format:?} ss{ss} msaa{msaa} {what}: wgpu and Metal differ by \
                     {mean:.2} mean, {p99} at the 99th percentile"
                );
            }
        }
    }

    // A quad view sets the viewport, scissor and auto-clear on the wgpu
    // renderer for each pane; the Metal frame has to follow them.
    let quads: Vec<Vec<u8>> = [false, true]
        .into_iter()
        .map(|metal| {
            let mut hr = build(metal, wgpu::TextureFormat::Rgba8UnormSrgb, 1, 4);
            let mut quad = threers::helpers::QuadView::new();
            quad.render(&mut hr, &mut scene(), &camera());
            hr.read_rgba()
        })
        .collect();
    let (mean, p99) = diff(&quads[0], &quads[1]);
    assert!(
        mean < 2.0 && p99 < 32,
        "quad view: wgpu and Metal differ by {mean:.2} mean, {p99} at the 99th percentile"
    );

    // The atmosphere material: a planet with its shell of air, lit from the
    // side so the terminator, the sunset band and the night-side aurora all
    // show.
    let atmosphere_scene = || {
        let mut s = Scene::new();
        s.background = Color::from_hex(0x000000);
        s.add_light(
            DirectionalLight::new(Color::WHITE, 3.0).with_direction(Vector3::new(-1.0, -0.2, -0.3)),
        );
        s.add(Object3D::mesh(Mesh::new(
            threers::SphereGeometry::new(1.0, 64, 32),
            StandardMaterial::new(Color::from_hex(0x224466)).into(),
        )));
        let mut air = threers::AtmosphereMaterial::new(1.0, 1.08);
        air.ozone = 0.6;
        air.aurora = 1.0;
        s.add(Object3D::mesh(Mesh::new(
            threers::SphereGeometry::new(1.08, 64, 32),
            threers::Material::Atmosphere(air),
        )));
        s
    };
    let mut cam = PerspectiveCamera::new(35.0, W as f32 / H as f32, 0.1, 100.0);
    cam.position = Vector3::new(0.0, 0.6, 4.5);
    cam.target = Vector3::ZERO;
    let skies: Vec<Vec<u8>> = [false, true]
        .into_iter()
        .map(|metal| {
            let mut hr = build(metal, wgpu::TextureFormat::Rgba8UnormSrgb, 1, 1);
            hr.render_to_rgba(&mut atmosphere_scene(), &cam)
        })
        .collect();
    check("atmosphere", diff(&skies[0], &skies[1]), true, (2.0, 24));

    // A custom shader written once with `shader!`, drawn by both.
    let shared = threers::shader! {
        const RIM_POWER: f32 = 3.0;
        fn rim(n: vec3f, v: vec3f) -> f32 {
            let f = 1.0 - max(dot(n, v), 0.0);
            return pow(f, RIM_POWER);
        }
        fn shade(s: Shading) -> vec4f {
            // Stripes from a storage array, so storage reaches both backends.
            let band = u32(floor(s.uv.y * f32(s0_len())));
            if (band >= s0_len()) { discard; }
            let base = data(0u32).xyz * s0(band);
            var lit = ambient_light() * base;
            for (var i = 0u32; i < light_count(); i += 1u32) {
                lit += base * light_color(i) * max(dot(s.normal, light_direction(i)), 0.0);
            }
            return vec4f(lit + rim(s.normal, s.view_dir) * data(1u32).xyz, opacity());
        }
    };
    let material = threers::ShaderMaterial::from_shared(&shared)
        .with_data(vec![[0.9, 0.5, 0.2, 0.0], [0.2, 0.5, 1.0, 0.0]])
        .with_storage0(vec![1.0, 0.4, 1.0, 0.7, 0.2, 1.0]);
    material.validate(true).expect("the shared shader's WGSL");
    let custom_scene = || {
        let mut s = Scene::new();
        s.background = Color::from_hex(0x0a0a10);
        s.add_light(AmbientLight::new(Color::WHITE, 0.2));
        s.add_light(DirectionalLight::new(Color::WHITE, 2.0).with_direction(Vector3::new(-0.5, -0.6, -1.0)));
        s.add(Object3D::mesh(Mesh::new(
            threers::SphereGeometry::new(1.0, 48, 24),
            threers::Material::Shader(material.clone()),
        )));
        s
    };
    let mut cam = PerspectiveCamera::new(40.0, W as f32 / H as f32, 0.1, 100.0);
    cam.position = Vector3::new(0.0, 0.0, 3.5);
    cam.target = Vector3::ZERO;
    let customs: Vec<Vec<u8>> = [false, true]
        .into_iter()
        .map(|metal| {
            let mut hr = build(metal, wgpu::TextureFormat::Rgba8UnormSrgb, 1, 1);
            hr.render_to_rgba(&mut custom_scene(), &cam)
        })
        .collect();
    let lit = customs[1].chunks_exact(4).filter(|p| p[0] > 40).count();
    assert!(lit > 500, "the custom shader drew almost nothing on Metal ({lit} px)");
    let (mean, p99) = diff(&customs[0], &customs[1]);
    assert!(
        mean < 2.0 && p99 < 24,
        "shared shader: wgpu and Metal differ by {mean:.2} mean, {p99} at the 99th percentile"
    );

    // A plain WGSL ShaderMaterial — no `msl`, no shared source: Metal
    // translates it. The `shader_material` example's fragment, verbatim.
    let plain = threers::ShaderMaterial::new(
        r#"
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let base = u_data.data[0].rgb;
    let freq = u_s0[0];
    let stripe = 0.5 + 0.5 * sin(in.world_pos.y * freq);
    let tint = mix(base, in.world_normal * 0.5 + 0.5, stripe);
    var light = frame.ambient.rgb;
    for (var i: u32 = 0u; i < frame.light_counts.x; i = i + 1u) {
        if (i >= 4u) { break; }
        let dir = normalize(frame.dir_lights[i].direction.xyz);
        light = light + frame.dir_lights[i].color.rgb * max(dot(normalize(in.world_normal), -dir), 0.0);
    }
    return vec4<f32>(framebuffer_encode(tint * light), 1.0);
}
"#,
    )
    .with_data(vec![[0.9, 0.4, 0.2, 0.0]])
    .with_storage0(vec![8.0]);
    let plain_scene = || {
        let mut s = Scene::new();
        s.background = Color::from_hex(0x0a0a10);
        s.add_light(AmbientLight::new(Color::WHITE, 0.2));
        s.add_light(DirectionalLight::new(Color::WHITE, 2.0).with_direction(Vector3::new(-0.5, -0.6, -1.0)));
        s.add(Object3D::mesh(Mesh::new(
            threers::SphereGeometry::new(1.0, 48, 24),
            threers::Material::Shader(plain.clone()),
        )));
        s
    };
    let plains: Vec<Vec<u8>> = [false, true]
        .into_iter()
        .map(|metal| {
            let mut hr = build(metal, wgpu::TextureFormat::Rgba8UnormSrgb, 1, 1);
            hr.render_to_rgba(&mut plain_scene(), &cam)
        })
        .collect();
    let (mean, p99) = diff(&plains[0], &plains[1]);
    assert!(
        mean < 2.0 && p99 < 24,
        "WGSL material: wgpu and Metal differ by {mean:.2} mean, {p99} at the 99th percentile"
    );

    // Fog over a ground of two large triangles: measured per fragment, so
    // what is under the camera is not fogged as if it were at the corners.
    for (mode, density) in [(1u32, 0.0f32), (2, 0.06)] {
        let fog_scene = || {
            let mut s = Scene::new();
            s.background = Color::from_hex(0xff0000);
            s.fog = threers::scene::FogParams {
                color: Color::from_hex(0xff0000),
                density,
                mode,
                near: 0.0,
                far: 30.0,
            };
            s.add_light(AmbientLight::new(Color::WHITE, 0.8));
            let mut ground = Object3D::mesh(Mesh::new(
                PlaneGeometry::new(40.0, 40.0),
                StandardMaterial::new(Color::from_hex(0x4d5243)).into(),
            ));
            ground.rotate_x(-std::f32::consts::FRAC_PI_2);
            s.add(ground);
            s
        };
        let mut cam = PerspectiveCamera::new(45.0, W as f32 / H as f32, 0.1, 100.0);
        cam.position = Vector3::new(0.0, 3.0, 6.0);
        cam.target = Vector3::ZERO;
        let fogged: Vec<Vec<u8>> = [false, true]
            .into_iter()
            .map(|metal| build(metal, wgpu::TextureFormat::Rgba8UnormSrgb, 1, 1).render_to_rgba(&mut fog_scene(), &cam))
            .collect();
        let (mean, p99) = diff(&fogged[0], &fogged[1]);
        assert!(
            mean < 1.0 && p99 < 8,
            "fog mode {mode}: wgpu and Metal differ by {mean:.2} mean, {p99} at the 99th percentile"
        );
    }

    // Additive blending on a built-in material, into an 8-bit sRGB target:
    // both backends add up the encoded values.
    let additive_scene = || {
        let mut s = Scene::new();
        s.background = Color::from_hex(0x101830);
        for (k, hex) in [0xff6020u32, 0x2060ff, 0x40ff60].into_iter().enumerate() {
            let mut m = threers::BasicMaterial::new(Color::from_hex(hex));
            m.transparent = true;
            m.opacity = 0.6;
            m.render_state.blending = threers::materials::Blending::Additive;
            let mut quad = Object3D::mesh(Mesh::new(PlaneGeometry::new(1.4, 1.4), threers::Material::Basic(m)));
            quad.position = Vector3::new(k as f32 * 0.45 - 0.45, 0.0, k as f32 * 0.01);
            s.add(quad);
        }
        s
    };
    let mut cam = PerspectiveCamera::new(45.0, W as f32 / H as f32, 0.1, 100.0);
    cam.position = Vector3::new(0.0, 0.0, 3.0);
    cam.target = Vector3::ZERO;
    let added: Vec<Vec<u8>> = [false, true]
        .into_iter()
        .map(|metal| build(metal, wgpu::TextureFormat::Rgba8UnormSrgb, 1, 1).render_to_rgba(&mut additive_scene(), &cam))
        .collect();
    check("additive blending", diff(&added[0], &added[1]), true, (1.0, 8));

    // A displacement map: the terrain relief a planet gets, moving vertices
    // along their normals in the vertex stage.
    let heights: Vec<u8> = (0..64 * 32)
        .flat_map(|i| {
            let (x, y) = (i % 64, i / 64);
            let h = if ((x / 8) + (y / 8)) % 2 == 0 { 255 } else { 0 };
            [h, h, h, 255]
        })
        .collect();
    let height_map = std::sync::Arc::new(threers::textures::Texture::new(
        64,
        32,
        threers::textures::TextureFormat::Rgba8Unorm,
        heights,
    ));
    let displaced_scene = || {
        let mut s = Scene::new();
        s.background = Color::from_hex(0x000000);
        s.add_light(AmbientLight::new(Color::WHITE, 0.3));
        s.add_light(DirectionalLight::new(Color::WHITE, 2.0).with_direction(Vector3::new(-0.5, -0.4, -1.0)));
        let mut m = StandardMaterial::new(Color::from_hex(0xc0a080));
        m.displacement_map = Some(height_map.clone());
        m.displacement_scale = 0.15;
        m.displacement_bias = -0.05;
        s.add(Object3D::mesh(Mesh::new(threers::SphereGeometry::new(1.0, 96, 48), threers::Material::Standard(m))));
        s
    };
    let mut cam = PerspectiveCamera::new(40.0, W as f32 / H as f32, 0.1, 100.0);
    cam.position = Vector3::new(0.0, 0.0, 3.6);
    cam.target = Vector3::ZERO;
    // Its shadow has the relief too: a displaced sphere over a ground that
    // receives it.
    let displaced_shadow = || {
        let mut s = Scene::new();
        s.background = Color::from_hex(0x000000);
        s.add_light(AmbientLight::new(Color::WHITE, 0.3));
        let mut sun = DirectionalLight::new(Color::WHITE, 2.0).with_direction(Vector3::new(-0.3, -1.0, -0.2));
        sun.cast_shadow = true;
        sun.shadow.map_size = 1024;
        s.add_light(sun);
        let mut m = StandardMaterial::new(Color::from_hex(0xc0a080));
        m.displacement_map = Some(height_map.clone());
        m.displacement_scale = 0.3;
        m.displacement_bias = -0.1;
        let mut ball = Object3D::mesh(Mesh::new(threers::SphereGeometry::new(1.0, 96, 48), threers::Material::Standard(m)));
        ball.cast_shadow = true;
        s.add(ball);
        let mut ground = Object3D::mesh(Mesh::new(PlaneGeometry::new(8.0, 8.0), StandardMaterial::new(Color::WHITE).into()));
        ground.rotate_x(-std::f32::consts::FRAC_PI_2);
        ground.position = Vector3::new(0.0, -1.4, 0.0);
        ground.receive_shadow = true;
        s.add(ground);
        s
    };
    let mut high = PerspectiveCamera::new(50.0, W as f32 / H as f32, 0.1, 100.0);
    high.position = Vector3::new(0.0, 4.0, 3.0);
    high.target = Vector3::new(0.0, -1.0, 0.0);
    let shadows: Vec<Vec<u8>> = [false, true]
        .into_iter()
        .map(|metal| build(metal, wgpu::TextureFormat::Rgba8UnormSrgb, 1, 1).render_to_rgba(&mut displaced_shadow(), &high))
        .collect();
    // The relief changes only the shadow's outline, a few dozen pixels: count
    // them rather than averaging them away.
    let (mean, _) = diff(&shadows[0], &shadows[1]);
    let off = shadows[0]
        .chunks(4)
        .zip(shadows[1].chunks(4))
        .filter(|(a, b)| (0..3).any(|c| a[c].abs_diff(b[c]) > 16))
        .count();
    assert!(
        mean < 1.0 && off < 10,
        "displaced shadow: wgpu and Metal differ by {mean:.2} mean, {off} pixels by more than 16"
    );
    let displaced: Vec<Vec<u8>> = [false, true]
        .into_iter()
        .map(|metal| build(metal, wgpu::TextureFormat::Rgba8UnormSrgb, 1, 1).render_to_rgba(&mut displaced_scene(), &cam))
        .collect();
    let (mean, p99) = diff(&displaced[0], &displaced[1]);
    assert!(
        mean < 2.0 && p99 < 24,
        "displacement: wgpu and Metal differ by {mean:.2} mean, {p99} at the 99th percentile"
    );

    // A screen-space ShaderMaterial reads the opaque scene behind it: the
    // capture's colour through a sampler, its depth exactly. A red box behind
    // a pane that swaps red and blue and marks found depth in green.
    let pane_wgsl = r#"
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(ss_depth_tex));
    let uv = in.clip_pos.xy / dims;
    let c = textureSampleLevel(ss_color_tex, ss_color_samp, uv, 0.0).rgb;
    let raw = textureLoad(ss_depth_tex, vec2<i32>(in.clip_pos.xy), 0).r;
    let d = select(raw, 1.0 - raw, frame.camera_position.w > 0.5);
    let hit = select(0.0, 1.0, d < 0.9999);
    return vec4<f32>(c.b, hit, c.r, 1.0);
}
"#;
    let screen_space_scene = || {
        let mut s = Scene::new();
        s.background = Color::new(0.0, 0.0, 0.0);
        let mut red = Object3D::mesh(Mesh::new(
            BoxGeometry::new(1.0, 1.0, 1.0),
            threers::Material::Basic(threers::BasicMaterial::new(Color::new(1.0, 0.0, 0.0))),
        ));
        red.position = Vector3::new(0.0, 0.0, -2.0);
        s.add(red);
        let pane = threers::ShaderMaterial::new(pane_wgsl).with_screen_space(true);
        let mut pane = Object3D::mesh(Mesh::new(PlaneGeometry::new(4.0, 4.0), pane.into()));
        pane.position = Vector3::new(0.0, 0.0, 1.0);
        s.add(pane);
        s
    };
    let mut ss_cam = PerspectiveCamera::new(50.0, W as f32 / H as f32, 0.1, 50.0);
    ss_cam.position = Vector3::new(0.0, 0.0, 3.0);
    ss_cam.look_at(Vector3::new(0.0, 0.0, -2.0));
    let ss: Vec<Vec<u8>> = [false, true]
        .into_iter()
        .map(|metal| build(metal, wgpu::TextureFormat::Rgba8UnormSrgb, 1, 1).render_to_rgba(&mut screen_space_scene(), &ss_cam))
        .collect();
    let centre = |px: &[u8]| {
        let i = (((H / 2) * W + W / 2) * 4) as usize;
        [px[i], px[i + 1], px[i + 2]]
    };
    let c = centre(&ss[1]);
    assert!(c[2] > 200 && c[1] > 200 && c[0] < 40, "Metal: the box does not show through as blue and green: {c:?}");
    let (mean, p99) = diff(&ss[0], &ss[1]);
    assert!(
        mean < 2.0 && p99 < 24,
        "screen-space material: wgpu and Metal differ by {mean:.2} mean, {p99} at the 99th percentile"
    );

    // A ShaderMaterial's `maps` bind on both backends: an sRGB checker
    // sampled through `u_samp` (at a blurred mip on the left), and a data
    // texture read exactly with `textureLoad`.
    let maps_wgsl = r#"
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let lod = select(0.0, 1.5, in.uv.x < 0.5);
    let c = textureSampleLevel(u_tex0, u_samp, in.uv * 2.0, lod).rgb;
    let dims = textureDimensions(u_tex1);
    let texel = vec2<i32>(in.uv * vec2<f32>(dims));
    let g = textureLoad(u_tex1, texel, 0).g;
    let n = f32(textureNumLevels(u_tex0));
    // u_tex2 brings its own levels: red at 0, blue below.
    let own = textureSampleLevel(u_tex2, u_samp, in.uv, 1.0).b;
    return vec4<f32>(c.r, g, select(c.b * (n / 7.0), own, in.uv.y > 0.5), 1.0);
}
"#;
    let maps_scene = || {
        let n = 64u32;
        let checker: Vec<u8> = (0..n * n)
            .flat_map(|i| {
                let on = ((i % n) / 8 + (i / n) / 8).is_multiple_of(2);
                if on { [230, 40, 200, 255] } else { [20, 180, 30, 255] }
            })
            .collect();
        let mut checker = threers::Texture::new(n, n, threers::TextureFormat::Rgba8UnormSrgb, checker);
        checker.mipmap_filter = Some(threers::TextureFilter::Linear);
        let ramp: Vec<u8> = (0..16u32 * 4).flat_map(|i| [0, (i * 255 / 63) as u8, 0, 255]).collect();
        let ramp = threers::Texture::new(16, 4, threers::TextureFormat::Rgba8Unorm, ramp);
        let solid = |w: u32, h: u32, px: [u8; 4]| std::sync::Arc::new(px.repeat((w * h) as usize));
        let mut own = threers::Texture::new(8, 8, threers::TextureFormat::Rgba8Unorm, (*solid(8, 8, [255, 0, 0, 255])).clone());
        own.mips = vec![solid(4, 4, [0, 0, 255, 255]), solid(2, 2, [0, 0, 255, 255]), solid(1, 1, [0, 0, 255, 255])];
        let mut s = Scene::new();
        let m = threers::ShaderMaterial::new(maps_wgsl)
            .with_maps(vec![std::sync::Arc::new(checker), std::sync::Arc::new(ramp), std::sync::Arc::new(own)]);
        s.add(Object3D::mesh(Mesh::new(PlaneGeometry::new(4.0, 3.0), m.into())));
        s
    };
    let mut maps_cam = PerspectiveCamera::new(50.0, W as f32 / H as f32, 0.1, 50.0);
    maps_cam.position = Vector3::new(0.0, 0.0, 3.0);
    maps_cam.look_at(Vector3::ZERO);
    let maps: Vec<Vec<u8>> = [false, true]
        .into_iter()
        .map(|metal| build(metal, wgpu::TextureFormat::Rgba8UnormSrgb, 1, 1).render_to_rgba(&mut maps_scene(), &maps_cam))
        .collect();
    let c = centre(&maps[1]);
    assert!(c[1] > 40, "Metal: the data texture does not read: {c:?}");
    // The top half (uv.y above 0.5, as three.js lays a plane out) reads the
    // supplied level.
    for (backend, px) in ["wgpu", "Metal"].iter().zip(&maps) {
        let i = (((H * 3 / 8) * W + W / 2) * 4) as usize;
        assert!(px[i + 2] > 250, "{backend}: the supplied mip level is not what is sampled: {:?}", &px[i..i + 4]);
    }
    let (mean, p99) = diff(&maps[0], &maps[1]);
    assert!(
        mean < 2.0 && p99 < 24,
        "material maps: wgpu and Metal differ by {mean:.2} mean, {p99} at the 99th percentile"
    );

    // An effect composer: every pass on Metal, through the same entry point,
    // the post-fx shader translated. Scene draws, half-float intermediates,
    // the blends, depth (SSAO, bokeh), an upload (SMAA's tables), a second
    // scene draw with overridden materials (outline), the 8-bit screen.
    use threers::postprocessing::{
        BokehPass, EffectComposer, OutlinePass, OutputPass, Pass, RenderPass, SmaaPass, SsaoPass,
        UnrealBloomPass,
    };
    type MakePass = fn(threers::core::ObjectId) -> Box<dyn Pass>;
    let chains: [(&str, MakePass); 5] = [
        ("bloom", |_| Box::new(UnrealBloomPass::new(W, H, 1.2, 0.4, 0.3))),
        ("ssao", |_| Box::new(SsaoPass::new(0, 0, W, H, 16))),
        ("bokeh", |_| Box::new(BokehPass::new(0, 0, 4.0, 0.05, 0.02))),
        ("smaa", |_| Box::new(SmaaPass::new(W, H))),
        ("outline", |id| Box::new(OutlinePass::new(W, H, 0, 0, vec![id]))),
    ];
    for (name, pass) in chains {
        for msaa in [1, 4] {
            let frames: Vec<Vec<u8>> = [false, true]
                .into_iter()
                .map(|metal| {
                    let mut hr = build(metal, wgpu::TextureFormat::Rgba8UnormSrgb, 1, msaa);
                    let mut s = scene();
                    let cube = s.arena.nodes.iter().find(|(_, o)| o.name == "cube").map(|(id, _)| id).expect("the cube");
                    let mut composer = EffectComposer::with_size(W, H);
                    composer.set_screen_format(wgpu::TextureFormat::Rgba8Unorm);
                    composer.add_pass(RenderPass::new());
                    composer.add_boxed(pass(cube));
                    composer.add_pass(OutputPass::new());
                    let px = hr.compose(&mut composer, &mut s, &camera());
                    assert_eq!(hr.is_native_metal(), metal, "{name}: compose stays on its backend");
                    assert_eq!(px.len(), (W * H * 4) as usize);
                    px
                })
                .collect();
            let (mean, p99) = diff(&frames[0], &frames[1]);
            assert!(
                mean < 2.0 && p99 < 24,
                "composer {name} msaa{msaa}: wgpu and Metal differ by {mean:.2} mean, {p99} at the 99th percentile"
            );
        }
    }

    // A call only wgpu can serve hands the renderer back, and the frame
    // already drawn is still the right one.
    let mut wgpu_hr = build(false, wgpu::TextureFormat::Rgba8UnormSrgb, 1, 1);
    let want = wgpu_hr.render_to_rgba(&mut scene(), &camera());
    let mut hr = build(true, wgpu::TextureFormat::Rgba8UnormSrgb, 1, 1);
    hr.render(&mut scene(), &camera());
    let _ = hr.device();
    assert!(!hr.is_native_metal(), "device() leaves the Metal path");
    let got = hr.read_rgba();
    assert_eq!(got, want, "the frame drawn before the hand-off is wgpu's own");
}
