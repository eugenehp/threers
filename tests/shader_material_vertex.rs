//! A `ShaderMaterial` that replaces the vertex stage, and the portable way to
//! feed it data.
//!
//! Until this existed a custom material could only change the *colour* of
//! geometry it was handed, so anything whose position has to be computed — a
//! camera-facing ribbon, a strand expanded from control points, a billboard —
//! had to be rebuilt on the CPU and re-uploaded whenever the camera moved. The
//! fuzz in `examples/raspberry.rs` is exactly that case and is what this is for.

use threers::{
    BufferAttribute, BufferGeometry, Color, HeadlessRenderer, Material, Mesh, Object3D,
    PerspectiveCamera, Scene, ShaderMaterial, ToneMapping, Vector3,
};

const W: u32 = 64;

/// A unit quad at the origin, facing +Z, with no useful positions of its own:
/// every test below computes where its corners go.
fn blank_quad() -> BufferGeometry {
    let mut g = BufferGeometry::new();
    g.set_attribute(
        "position",
        BufferAttribute::new(vec![0.0; 12], 3),
    );
    g.set_attribute(
        "normal",
        BufferAttribute::new(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], 3),
    );
    g.set_attribute(
        "uv",
        BufferAttribute::new(vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0], 2),
    );
    g.set_attribute("color", BufferAttribute::new(vec![1.0; 16], 4));
    g.set_index(vec![0, 1, 2, 0, 2, 3]);
    g.compute_bounding_box();
    g.compute_bounding_sphere();
    g
}

fn render(material: ShaderMaterial, geometry: BufferGeometry) -> Option<Vec<u8>> {
    let mut r = HeadlessRenderer::builder()
        .size(W, W)
        .color_format(threers::wgpu::TextureFormat::Rgba8Unorm)
        .build()
        .ok()?;
    r.renderer().set_tone_mapping(ToneMapping::None, 1.0);
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.add(Object3D::mesh(Mesh::new(
        geometry,
        Material::Shader(material),
    )));
    let mut cam = PerspectiveCamera::new(60.0, 1.0, 0.1, 100.0);
    cam.position = Vector3::new(0.0, 0.0, 4.0);
    cam.look_at(Vector3::ZERO);
    scene.update_world();
    Some(r.render_to_rgba(&mut scene, &cam))
}

fn at(px: &[u8], x: u32, y: u32) -> [u8; 3] {
    let i = ((y * W + x) * 4) as usize;
    [px[i], px[i + 1], px[i + 2]]
}

const WHITE: &str = "@fragment\n\
     fn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n\
     return vec4<f32>(1.0, 1.0, 1.0, 1.0);\n\
     }\n";

/// The vertex stage runs at all: four degenerate corners, all at the origin,
/// are pushed out into a quad that covers the middle of the frame.
#[test]
fn a_custom_vertex_stage_places_the_geometry() {
    let material = ShaderMaterial::new(WHITE).with_vertex(
        "@vertex\n\
         fn vs_custom(in: VsIn, @builtin(vertex_index) vi: u32) -> VsOut {\n\
         var out = standard_vertex(in);\n\
         let x = select(-1.0, 1.0, (vi == 1u) || (vi == 2u));\n\
         let y = select(-1.0, 1.0, vi >= 2u);\n\
         out.world_pos = vec3<f32>(x, y, 0.0);\n\
         out.clip_pos = frame.view_proj * vec4<f32>(out.world_pos, 1.0);\n\
         return out;\n\
         }\n",
    );

    let Some(px) = render(material, blank_quad()) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    assert_eq!(
        at(&px, W / 2, W / 2),
        [255, 255, 255],
        "the centre should be covered by the expanded quad"
    );
    assert_eq!(at(&px, 1, 1), [0, 0, 0], "the corner should still be clear");
}

/// Without a vertex stage the same geometry draws nothing, which is what makes
/// the test above a test of the vertex stage and not of the fragment.
#[test]
fn the_same_geometry_is_invisible_without_one() {
    let Some(px) = render(ShaderMaterial::new(WHITE), blank_quad()) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    assert_eq!(
        at(&px, W / 2, W / 2),
        [0, 0, 0],
        "four coincident vertices cannot cover a pixel"
    );
}

/// The vertex stage can read the user's data, which is the whole point: the
/// positions have to come from somewhere, and a custom material's data lives in
/// `u_data` and the storage arrays.
#[test]
fn the_vertex_stage_reads_the_user_data_and_the_storage_arrays() {
    // Corner offsets in `s0`, scaled by a uniform. If either binding were
    // fragment-only this would not compile, and if either were misread the quad
    // would be the wrong size.
    let material = ShaderMaterial::new(WHITE)
        .with_vertex(
            "@vertex\n\
             fn vs_custom(in: VsIn, @builtin(vertex_index) vi: u32) -> VsOut {\n\
             var out = standard_vertex(in);\n\
             let scale = u_data.data[0].x;\n\
             let x = s0(vi * 2u) * scale;\n\
             let y = s0(vi * 2u + 1u) * scale;\n\
             out.world_pos = vec3<f32>(x, y, 0.0);\n\
             out.clip_pos = frame.view_proj * vec4<f32>(out.world_pos, 1.0);\n\
             return out;\n\
             }\n",
        )
        .with_data(vec![[1.0, 0.0, 0.0, 0.0]]);
    let mut big = material.clone();
    big.data = vec![[1.0, 0.0, 0.0, 0.0]];
    big.storage0 = vec![-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0];

    let mut small = big.clone();
    small.data = vec![[0.18, 0.0, 0.0, 0.0]];

    let Some(wide) = render(big, blank_quad()) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let narrow = render(small, blank_quad()).expect("second render");

    let covered = |px: &[u8]| px.chunks_exact(4).filter(|p| p[0] > 128).count();
    let (a, b) = (covered(&wide), covered(&narrow));
    assert!(a > 0 && b > 0, "neither quad drew anything ({a}, {b})");
    assert!(
        a > b * 4,
        "the uniform did not scale the quad: {a} pixels against {b}"
    );
}

/// `s0`/`s1`/`s2` read the same values `u_s0`/`u_s1`/`u_s2` do.
///
/// They are not the same code — on a device with no storage buffers the
/// accessors read a texture instead — so the two have to be checked against
/// each other on a device that has both.
#[test]
fn the_portable_accessors_agree_with_the_raw_storage_arrays() {
    let probe = |src: &str| -> Option<[u8; 3]> {
        let mut m = ShaderMaterial::new(src);
        // 0.25, 0.50, 0.75 at indices 5, 6, 7 — past the first texel, so a
        // texture path that got its packing wrong reads the wrong one.
        m.storage0 = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75];
        let px = render(m, threers::BoxGeometry::new(8.0, 8.0, 0.1))?;
        Some(at(&px, W / 2, W / 2))
    };

    let Some(raw) = probe(
        "@fragment\n\
         fn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n\
         return vec4<f32>(u_s0[5], u_s0[6], u_s0[7], 1.0);\n\
         }\n",
    ) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let portable = probe(
        "@fragment\n\
         fn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n\
         return vec4<f32>(s0(5u), s0(6u), s0(7u), 1.0);\n\
         }\n",
    )
    .expect("second render");

    assert_eq!(raw, portable, "s0() disagrees with u_s0[]");
    // And they are the values that went in, not zeroes that happen to match.
    // The fragment returns them unencoded into an `Rgba8Unorm` target, so the
    // bytes are linear: 0.25 is 64, not the 137 an sRGB encode would give.
    for (got, want) in raw.iter().zip([0.25f32, 0.5, 0.75]) {
        let expect = (want * 255.0 + 0.5) as u8;
        assert!(
            (*got as i32 - expect as i32).abs() <= 2,
            "read back {raw:?}, expected about {:?}",
            [64, 128, 191]
        );
    }
}

/// Two materials that differ only in their vertex source must not share a
/// compiled pipeline. The cache was keyed on the fragment alone, so the second
/// one would have drawn with the first one's geometry.
#[test]
fn the_pipeline_cache_tells_two_vertex_stages_apart() {
    let shrink = ShaderMaterial::new(WHITE).with_vertex(
        "@vertex\n\
         fn vs_custom(in: VsIn, @builtin(vertex_index) vi: u32) -> VsOut {\n\
         var out = standard_vertex(in);\n\
         let x = select(-0.2, 0.2, (vi == 1u) || (vi == 2u));\n\
         let y = select(-0.2, 0.2, vi >= 2u);\n\
         out.world_pos = vec3<f32>(x, y, 0.0);\n\
         out.clip_pos = frame.view_proj * vec4<f32>(out.world_pos, 1.0);\n\
         return out;\n\
         }\n",
    );
    let grow = ShaderMaterial::new(WHITE).with_vertex(
        "@vertex\n\
         fn vs_custom(in: VsIn, @builtin(vertex_index) vi: u32) -> VsOut {\n\
         var out = standard_vertex(in);\n\
         let x = select(-1.5, 1.5, (vi == 1u) || (vi == 2u));\n\
         let y = select(-1.5, 1.5, vi >= 2u);\n\
         out.world_pos = vec3<f32>(x, y, 0.0);\n\
         out.clip_pos = frame.view_proj * vec4<f32>(out.world_pos, 1.0);\n\
         return out;\n\
         }\n",
    );

    let covered = |m: ShaderMaterial| -> Option<usize> {
        Some(
            render(m, blank_quad())?
                .chunks_exact(4)
                .filter(|p| p[0] > 128)
                .count(),
        )
    };
    let Some(small) = covered(shrink) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let big = covered(grow).expect("second render");
    assert!(
        big > small * 4,
        "the two vertex stages drew the same size ({small} and {big}) — the \
         pipeline cache handed the second one the first one's shader"
    );
}

/// The data-texture fallback, on a real device, by asking for one with no
/// storage buffers at all.
///
/// That is what WebGL2 reports, and it is the configuration the whole fallback
/// exists for — but a Mac never produces it on its own, so the test asks wgpu
/// for a device with `max_storage_buffers_per_shader_stage = 0` and gets the
/// same code path a browser would take. Without this the texture path is only
/// ever checked by reading it.
#[test]
fn a_device_with_no_storage_buffers_reads_the_data_from_a_texture() {
    use std::sync::Arc;

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let Some(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .ok() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    // Everything the renderer needs, with the one limit a WebGL2 context does
    // not have.
    let mut limits = adapter.limits();
    limits.max_storage_buffers_per_shader_stage = 0;
    let Ok((device, queue)) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("no-storage device"),
        required_features: wgpu::Features::empty(),
        required_limits: limits,
        ..Default::default()
    })) else {
        eprintln!("skipping: the adapter would not give a zero-storage device");
        return;
    };
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    assert!(
        !threers::renderer::caps::supports_storage_buffers(&device),
        "the device still reports storage buffers; the test is not testing anything"
    );

    let format = threers::wgpu::TextureFormat::Rgba8Unorm;
    let mut renderer = threers::Renderer::new(device.clone(), queue.clone(), format, W, W);
    renderer.set_tone_mapping(ToneMapping::None, 1.0);

    // The same shader the storage test uses, through `s0` — which on this
    // device compiles to a `textureLoad`, not an array index.
    let mut m = ShaderMaterial::new(
        "@fragment\n\
         fn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n\
         return vec4<f32>(s0(5u), s0(6u), s0(7u), 1.0);\n\
         }\n",
    );
    m.storage0 = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75];

    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.add(Object3D::mesh(Mesh::new(
        threers::BoxGeometry::new(8.0, 8.0, 0.1),
        Material::Shader(m),
    )));
    let mut cam = PerspectiveCamera::new(60.0, 1.0, 0.1, 100.0);
    cam.position = Vector3::new(0.0, 0.0, 4.0);
    cam.look_at(Vector3::ZERO);
    scene.update_world();

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("target"),
        size: wgpu::Extent3d {
            width: W,
            height: W,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    renderer.render(&mut scene, &cam, &view, false);

    // Read the centre back.
    let bpr = W * 4;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (bpr * W) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bpr),
                rows_per_image: Some(W),
            },
        },
        wgpu::Extent3d {
            width: W,
            height: W,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(enc.finish()));
    buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let px = buf
        .slice(..)
        .get_mapped_range()
        .expect("buffer range is mapped")
        .to_vec();

    let got = at(&px, W / 2, W / 2);
    for (i, want) in [0.25f32, 0.5, 0.75].into_iter().enumerate() {
        let expect = (want * 255.0 + 0.5) as u8;
        assert!(
            (got[i] as i32 - expect as i32).abs() <= 2,
            "the texture fallback read back {got:?}, expected about [64, 128, 191]"
        );
    }
}
