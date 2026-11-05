//! What a compute pass on a renderer's own GPU (`marks_gpu`) hands the
//! renderer — an array texture's layer to sample, a mesh to draw — arrives
//! the same on the native Metal backend as on wgpu, and a translated
//! `ShaderMaterial` sees the frame wgpu's does. `examples/ocean_water` is
//! built from these.

#![cfg(metal_backend)]

use std::sync::Arc;

use threers::marks::gpu;
use threers::{
    BasicMaterial, BufferGeometry, Color, HeadlessRenderer, Material, Mesh, Object3D,
    OrthographicCamera, PerspectiveCamera, PlaneGeometry, Scene, ShaderMaterial, Vector3,
};

const SIZE: u32 = 32;
// sRGB, so a shader's linear output lands encoded on both backends.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// `v` as the sRGB target stores it.
fn encoded(v: f32) -> u8 {
    let s = if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (s.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn build(metal: bool) -> HeadlessRenderer {
    build_as(metal, FORMAT)
}

fn build_as(metal: bool, format: wgpu::TextureFormat) -> HeadlessRenderer {
    // The backend is read from the environment as a renderer is built, and
    // the tests here run at once: one build at a time.
    static ENV: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _env = ENV.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: the lock above; nothing else here reads the environment.
    unsafe {
        if metal {
            std::env::set_var("THREERS_BACKEND", "metal");
        } else {
            std::env::remove_var("THREERS_BACKEND");
        }
    }
    let mut hr = HeadlessRenderer::builder()
        .size(SIZE, SIZE)
        .color_format(format)
        .build()
        .expect("headless renderer");
    unsafe { std::env::remove_var("THREERS_BACKEND") };
    hr.set_msaa(1);
    assert_eq!(hr.is_native_metal(), metal);
    hr
}

fn no_metal() -> bool {
    let missing = threers::metal::MetalDevice::new().is_err();
    if missing {
        eprintln!("skipping: no Metal device");
    }
    missing
}

fn at(f: &[u8], x: u32, y: u32) -> [u8; 3] {
    let i = ((y * SIZE + x) * 4) as usize;
    [f[i], f[i + 1], f[i + 2]]
}

/// A compute pipeline over one bind group, from WGSL with entry `main`.
fn compute(
    g: &gpu::Gpu,
    wgsl: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> (gpu::ComputePipeline, gpu::BindGroupLayout) {
    let module = g.create_shader_module(gpu::ShaderModuleDescriptor {
        label: None,
        source: gpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let layout = g.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries,
    });
    let pipeline_layout = g.create_pipeline_layout(&gpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = g.create_compute_pipeline(&gpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    (pipeline, layout)
}

fn run(g: &gpu::Gpu, pipeline: &gpu::ComputePipeline, bind: &gpu::BindGroup, groups: [u32; 3]) {
    let mut e = g.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = e.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind, &[]);
        pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
    }
    g.submit(Some(e.finish()));
}

/// A compute pass writes every layer of a 2D array texture; one layer, as a
/// 2D view, reaches a `ShaderMaterial` through `textures` — read exactly,
/// and through `u_samp`, which repeats.
#[test]
fn an_array_layer_written_by_compute_reaches_a_shader_material() {
    if no_metal() {
        return;
    }
    const WRITE: &str = "
@group(0) @binding(0) var dst: texture_storage_2d_array<rgba16float, write>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    textureStore(dst, vec2<i32>(i32(id.x), 0), i32(id.z),
                 vec4<f32>(f32(id.x) / 3.0, f32(id.z) / 4.0, 0.0, 1.0));
}";
    // Red: the texel under the pixel. Blue: the same through `u_samp` one
    // whole tile along — equal when it repeats, the last texel when it clamps.
    // Green: the layer's own marker.
    let fragment = "@fragment fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
        let i = min(i32(in.uv.x * 4.0), 3);
        let t = textureLoad(u_tex0, vec2<i32>(i, 0), 0);
        let s = textureSampleLevel(u_tex0, u_samp, vec2<f32>((f32(i) + 0.5) / 4.0 + 1.0, 0.5), 0.0);
        return vec4<f32>(t.r, t.g, s.r, 1.0);
    }";
    let mut camera = OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    camera.position = Vector3::new(0.0, 0.0, 5.0);
    camera.look_at(Vector3::ZERO);
    let frames: Vec<Vec<u8>> = [false, true]
        .into_iter()
        .map(|metal| {
            let mut hr = build(metal);
            let g = hr.marks_gpu();
            let texture = g.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 4,
                    height: 1,
                    depth_or_array_layers: 3,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let (pipeline, layout) = compute(
                &g,
                WRITE,
                &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                }],
            );
            let all = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let bind = g.create_bind_group(&gpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[gpu::BindGroupEntry {
                    binding: 0,
                    resource: gpu::BindingResource::TextureView(&all),
                }],
            });
            run(&g, &pipeline, &bind, [4, 1, 3]);
            let layer = texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: 2,
                array_layer_count: Some(1),
                ..Default::default()
            });
            let mut m = ShaderMaterial::new(fragment).with_textures([layer]);
            m.tone_mapped = false;
            let mut scene = Scene::new();
            scene.add(Object3D::mesh(Mesh::new(
                PlaneGeometry::new(2.0, 2.0),
                Material::Shader(m),
            )));
            let frame = hr.render_to_rgba(&mut scene, &camera);
            assert_eq!(
                hr.is_native_metal(),
                metal,
                "the textures keep the renderer on its backend"
            );
            frame
        })
        .collect();
    for (backend, f) in ["wgpu", "Metal"].iter().zip(&frames) {
        for i in 0..4u32 {
            let p = at(f, i * SIZE / 4 + SIZE / 8, SIZE / 2);
            let want = encoded(i as f32 / 3.0);
            assert!(
                p[0].abs_diff(want) <= 2,
                "{backend}: texel {i} reads {p:?}, wants red {want}"
            );
            // Layer 2 of 3: 2 / 4.
            assert!(
                p[1].abs_diff(encoded(0.5)) <= 2,
                "{backend}: texel {i} is from another layer: {p:?}"
            );
            assert!(
                p[2].abs_diff(p[0]) <= 2,
                "{backend}: u_samp does not repeat at texel {i}: {p:?}"
            );
        }
    }
}

/// `frame.viewport_size.zw` is the camera's near and far on both backends —
/// what a shader linearising the captured depth reads.
#[test]
fn a_translated_shader_material_sees_near_and_far() {
    if no_metal() {
        return;
    }
    let fragment = "@fragment fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
        return vec4<f32>(frame.viewport_size.z, frame.viewport_size.w / 100.0, 0.0, 1.0);
    }";
    let mut camera = PerspectiveCamera::new(50.0, 1.0, 0.5, 80.0);
    camera.position = Vector3::new(0.0, 0.0, 3.0);
    camera.look_at(Vector3::ZERO);
    for metal in [false, true] {
        let backend = if metal { "Metal" } else { "wgpu" };
        let mut hr = build(metal);
        let mut m = ShaderMaterial::new(fragment);
        m.tone_mapped = false;
        let mut scene = Scene::new();
        scene.add(Object3D::mesh(Mesh::new(
            PlaneGeometry::new(4.0, 4.0),
            Material::Shader(m),
        )));
        let f = hr.render_to_rgba(&mut scene, &camera);
        let p = at(&f, SIZE / 2, SIZE / 2);
        // 0.5 and 80 / 100.
        assert!(
            p[0].abs_diff(encoded(0.5)) <= 2 && p[1].abs_diff(encoded(0.8)) <= 2,
            "{backend}: near/far read as {p:?}"
        );
    }
}

/// A `gpu_writable` mesh is where its compute pass put it, not where its CPU
/// positions say: here those are behind the camera, and a pass moves the
/// vertices in front of it. Culled by the CPU positions, nothing would draw.
#[test]
fn a_gpu_writable_mesh_is_not_culled_by_its_cpu_positions() {
    if no_metal() {
        return;
    }
    const MOVE: &str = "
@group(0) @binding(0) var<storage, read_write> verts: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    verts[id.x * STRIDEu + 2u] = verts[id.x * STRIDEu + 2u] - 100.0;
}";
    let mut camera = PerspectiveCamera::new(50.0, 1.0, 0.5, 80.0);
    camera.position = Vector3::new(0.0, 0.0, 3.0);
    camera.look_at(Vector3::ZERO);
    for metal in [false, true] {
        let backend = if metal { "Metal" } else { "wgpu" };
        let mut hr = build(metal);
        let g = hr.marks_gpu();
        let mut geometry: BufferGeometry = PlaneGeometry::new(2.0, 2.0);
        let position = geometry.attributes.get_mut("position").expect("positions");
        for z in position.array.iter_mut().skip(2).step_by(3) {
            *z += 100.0;
        }
        geometry.gpu_writable = true;
        let geometry = Arc::new(geometry);
        let mut scene = Scene::new();
        scene.background = Color::from_hex(0x000000);
        scene.add(Object3D::mesh(Mesh::from_arc(
            geometry.clone(),
            Arc::new(BasicMaterial::new(Color::from_hex(0xff0000)).into()),
        )));
        hr.upload_geometry(&geometry);
        let buffer = hr
            .vertex_buffer(&geometry)
            .expect("the uploaded vertex buffer");
        let (pipeline, layout) = compute(
            &g,
            &MOVE.replace("STRIDE", &hr.vertex_floats().to_string()),
            &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        );
        let bind = g.create_bind_group(&gpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[gpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        let vertices = geometry.attributes["position"].count() as u32;
        run(&g, &pipeline, &bind, [vertices, 1, 1]);
        let f = hr.render_to_rgba(&mut scene, &camera);
        let p = at(&f, SIZE / 2, SIZE / 2);
        assert!(
            p[0] > 200 && p[1] < 40,
            "{backend}: the moved mesh is not drawn: {p:?}"
        );
    }
}

/// On a plain `Unorm` target a `ShaderMaterial`'s output is the stored byte,
/// and `framebuffer_encode` is what encodes — on Metal, which draws such a
/// target through an sRGB one, as on wgpu.
#[test]
fn a_unorm_target_stores_a_shader_material_output_as_it_stands() {
    if no_metal() {
        return;
    }
    // Left half raw 0.5, right half `framebuffer_encode(0.5)`.
    let fragment = "@fragment fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
        let v = vec3<f32>(0.5, 0.25, 0.75);
        return vec4<f32>(select(v, framebuffer_encode(v), in.uv.x > 0.5), 1.0);
    }";
    let mut camera = OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    camera.position = Vector3::new(0.0, 0.0, 5.0);
    camera.look_at(Vector3::ZERO);
    for metal in [false, true] {
        let backend = if metal { "Metal" } else { "wgpu" };
        let mut hr = build_as(metal, wgpu::TextureFormat::Rgba8Unorm);
        let mut m = ShaderMaterial::new(fragment);
        m.tone_mapped = false;
        let mut scene = Scene::new();
        scene.add(Object3D::mesh(Mesh::new(
            PlaneGeometry::new(2.0, 2.0),
            Material::Shader(m),
        )));
        let f = hr.render_to_rgba(&mut scene, &camera);
        let raw = at(&f, SIZE / 4, SIZE / 2);
        let enc = at(&f, SIZE * 3 / 4, SIZE / 2);
        let want_raw = [128u8, 64, 191];
        let want_enc = [encoded(0.5), encoded(0.25), encoded(0.75)];
        for c in 0..3 {
            assert!(
                raw[c].abs_diff(want_raw[c]) <= 2,
                "{backend}: raw output stored as {raw:?}, wants {want_raw:?}"
            );
            assert!(
                enc[c].abs_diff(want_enc[c]) <= 2,
                "{backend}: framebuffer_encode stored as {enc:?}, wants {want_enc:?}"
            );
        }
    }
}
