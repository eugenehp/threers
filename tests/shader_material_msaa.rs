//! `ShaderMaterial` draws are antialiased when MSAA is on.
//!
//! The multisampled surface pass used to be skipped for the whole frame as soon
//! as one custom-shader draw was in it, because custom pipelines were only
//! compiled single-sampled. A map drawn entirely with one `ShaderMaterial` got
//! no antialiasing at all, whatever `setMsaa` said.

use threers::{
    BufferAttribute, BufferGeometry, Material, Mesh, Object3D, PerspectiveCamera, Renderer, Scene,
    ShaderMaterial, ToneMapping, Vector3,
};

const W: u32 = 64;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const RED: &str = "@fragment\n\
    fn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n\
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);\n\
    }\n";
/// Same, through a custom vertex stage (the path the map uses).
const VS: &str = "@vertex\n\
    fn vs_custom(in: VsIn) -> VsOut {\n\
    return standard_vertex(in);\n\
    }\n";

fn draw(msaa: u32, vertex: bool) -> Option<Vec<u8>> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("msaa test device"),
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .ok()?;
    let (device, queue) = (std::sync::Arc::new(device), std::sync::Arc::new(queue));
    let mut renderer = Renderer::new(device.clone(), queue.clone(), FORMAT, W, W);
    renderer.set_tone_mapping(ToneMapping::None, 1.0);
    renderer.set_msaa(msaa);
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
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());

    // One triangle with a long diagonal edge across the view.
    let mut g = BufferGeometry::new();
    g.set_attribute(
        "position",
        BufferAttribute::new(vec![-2.0, -2.0, 0.0, 2.0, -1.3, 0.0, -1.7, 2.0, 0.0], 3),
    );
    g.set_attribute("normal", BufferAttribute::new([0.0, 0.0, 1.0].repeat(3), 3));
    g.set_index(vec![0, 1, 2]);
    let mut sm = ShaderMaterial::new(RED);
    if vertex {
        sm.vertex = Some(VS.to_string());
    }
    sm.side = 2;
    let mut scene = Scene::new();
    scene.add(Object3D::mesh(Mesh::new(g, Material::Shader(sm))));
    let mut cam = PerspectiveCamera::new(60.0, 1.0, 0.1, 100.0);
    cam.position = Vector3::new(0.0, 0.0, 3.0);
    cam.look_at(Vector3::ZERO);
    scene.update_world();
    renderer.render(&mut scene, &cam, &view, false);

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
    let px = buf.slice(..).get_mapped_range().expect("mapped").to_vec();
    Some(px)
}

/// Pixels whose red channel is strictly between the background's and the
/// triangle's: what an antialiased edge leaves and an aliased one cannot.
fn partial(px: &[u8]) -> usize {
    let reds: Vec<u8> = px.chunks(4).map(|p| p[0]).collect();
    let lo = *reds.iter().min().unwrap();
    let hi = *reds.iter().max().unwrap();
    assert!(hi > lo + 100, "the triangle was not drawn (red {lo}..{hi})");
    reds.iter().filter(|&&r| r > lo + 20 && r < hi - 20).count()
}

#[test]
fn shader_material_edges_are_antialiased_with_msaa() {
    for vertex in [false, true] {
        let Some(off) = draw(1, vertex) else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let on = draw(4, vertex).unwrap();
        assert_eq!(partial(&off), 0, "single-sample edge has blended pixels?");
        assert!(
            partial(&on) > W as usize,
            "MSAA on, but the custom-shader edge is aliased ({} partial pixels, vertex stage {vertex})",
            partial(&on)
        );
    }
}
