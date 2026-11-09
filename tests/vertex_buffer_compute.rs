//! A compute pass writing a mesh's vertex buffer in place, through
//! `HeadlessRenderer::{upload_geometry, vertex_buffer, vertex_floats}` and
//! `marks_gpu`, moves the mesh on either backend.

#![cfg(metal_backend)]

use std::sync::Arc;
use threers::marks::gpu;
use threers::RenderBackend;
use threers::{
    BasicMaterial, BufferGeometry, Color, HeadlessRenderer, Material, Mesh, Object3D,
    PerspectiveCamera, PlaneGeometry, Scene, Vector3,
};

const SIZE: u32 = 96;

/// Halves every vertex's x and y; `stride` f32s a vertex.
const SHRINK: &str = r#"
struct Params { count : u32, stride : u32, pad0 : u32, pad1 : u32 };
@group(0) @binding(0) var<storage, read_write> verts : array<f32>;
@group(0) @binding(1) var<uniform> params : Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    if (gid.x >= params.count) { return; }
    let b = gid.x * params.stride;
    verts[b] = verts[b] * 0.5;
    verts[b + 1u] = verts[b + 1u] * 0.5;
}
"#;

fn build(metal: bool) -> HeadlessRenderer {
    let hr = HeadlessRenderer::builder()
        .backend(if metal {
            RenderBackend::Metal
        } else {
            RenderBackend::Wgpu
        })
        .size(SIZE, SIZE)
        .build()
        .expect("headless renderer");
    assert_eq!(hr.is_native_metal(), metal);
    hr
}

fn shrink(gpu: &gpu::Gpu, buffer: &gpu::Buffer, count: u32, stride: u32) {
    let module = gpu.create_shader_module(gpu::ShaderModuleDescriptor {
        label: Some("shrink"),
        source: gpu::ShaderSource::Wgsl(SHRINK.into()),
    });
    let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let layout = gpu.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("shrink"),
        entries: &[
            entry(0, wgpu::BufferBindingType::Storage { read_only: false }),
            entry(1, wgpu::BufferBindingType::Uniform),
        ],
    });
    let pipeline_layout = gpu.create_pipeline_layout(&gpu::PipelineLayoutDescriptor {
        label: Some("shrink"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = gpu.create_compute_pipeline(&gpu::ComputePipelineDescriptor {
        label: Some("shrink"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let params = gpu.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("shrink params"),
        contents: bytemuck::cast_slice(&[count, stride, 0, 0]),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind = gpu.create_bind_group(&gpu::BindGroupDescriptor {
        label: Some("shrink"),
        layout: &layout,
        entries: &[
            gpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            },
            gpu::BindGroupEntry {
                binding: 1,
                resource: params.as_entire_binding(),
            },
        ],
    });
    let mut encoder = gpu.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
    }
    gpu.submit([encoder.finish()]);
}

fn lit(rgba: &[u8]) -> usize {
    rgba.as_chunks::<4>()
        .0
        .iter()
        .filter(|p| p[0] > 128)
        .count()
}

#[test]
fn a_compute_pass_moves_a_mesh_on_either_backend() {
    if threers::metal::MetalDevice::new().is_err() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let mut camera = PerspectiveCamera::new(50.0, 1.0, 0.1, 50.0);
    camera.position = Vector3::new(0.0, 0.0, 3.0);
    camera.look_at(Vector3::ZERO);
    let frames: Vec<(usize, usize, Vec<u8>)> = [false, true]
        .into_iter()
        .map(|metal| {
            let mut hr = build(metal);
            let mut plane: BufferGeometry = PlaneGeometry::new(2.0, 2.0);
            plane.gpu_writable = true;
            let count = plane.attributes["position"].count() as u32;
            let plane = Arc::new(plane);
            let mut scene = Scene::new();
            scene.background = Color::new(0.0, 0.0, 0.0);
            scene.add(Object3D::mesh(Mesh::from_arc(
                plane.clone(),
                Arc::new(Material::Basic(BasicMaterial::new(Color::new(
                    1.0, 0.0, 0.0,
                )))),
            )));
            let before = lit(&hr.render_to_rgba(&mut scene, &camera));
            hr.upload_geometry(&plane);
            let buffer = hr.vertex_buffer(&plane).expect("the plane's vertex buffer");
            assert!(buffer.size() >= (count * hr.vertex_floats() * 4) as u64);
            let gpu = hr.marks_gpu();
            shrink(&gpu, &buffer, count, hr.vertex_floats());
            let after_frame = hr.render_to_rgba(&mut scene, &camera);
            assert_eq!(
                hr.is_native_metal(),
                metal,
                "the compute pass keeps the renderer on its backend"
            );
            (before, lit(&after_frame), after_frame)
        })
        .collect();
    for (backend, (before, after, _)) in ["wgpu", "Metal"].iter().zip(&frames) {
        let ratio = *after as f32 / *before as f32;
        assert!(
            (0.2..0.3).contains(&ratio),
            "{backend}: {before} lit pixels became {after}, not a quarter"
        );
    }
    let differing = frames[0]
        .2
        .as_chunks::<4>()
        .0
        .iter()
        .zip(frames[1].2.as_chunks::<4>().0)
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        differing < (SIZE * 2) as usize,
        "{differing} pixels differ between wgpu and Metal"
    );
}
