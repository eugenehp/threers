//! `ShaderMaterial`s cast directional shadows through their own vertex stage,
//! and the directional shadow box can follow the camera.
//!
//! Custom-shader draws used to be left out of the shadow pass entirely, and the
//! box was a fixed cube on the origin: a map drawn with one `ShaderMaterial`,
//! kilometres from the origin, had no shadows at all.

use threers::{
    BoxGeometry, DirectionalLight, Material, Mesh, Object3D, PerspectiveCamera, Renderer, Scene,
    ShaderMaterial, ShadowSettings, ToneMapping, Vector3,
};

const W: u32 = 64;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Ground: red = lit fraction.
const RECEIVER: &str = "@fragment\n\
    fn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n\
    let s = shadow_factor(in.world_pos, vec3<f32>(0.0, 1.0, 0.0));\n\
    return vec4<f32>(s, 0.0, 0.0, 1.0);\n\
    }\n";
/// Occluder: blue, moved sideways by `u_data.data[0].x` in its vertex stage.
const CASTER_FS: &str = "@fragment\n\
    fn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n\
    return vec4<f32>(0.0, 0.0, 1.0, 1.0);\n\
    }\n";
const CASTER_VS: &str = "@vertex\n\
    fn vs_custom(in: VsIn) -> VsOut {\n\
    var out = standard_vertex(in);\n\
    out.world_pos.x = out.world_pos.x + u_data.data[0].x;\n\
    out.clip_pos = frame.view_proj * vec4<f32>(out.world_pos, 1.0);\n\
    out.clip_zw = out.clip_pos.zw;\n\
    return out;\n\
    }\n";

/// Render the scene around `cx` and return the red channel of the ground at
/// world (cx + dx, 0, 0) for each dx.
fn lit_at(cx: f32, fit_view: f32, probes: &[f32]) -> Option<Vec<f32>> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("shadow test device"),
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .ok()?;
    let (device, queue) = (std::sync::Arc::new(device), std::sync::Arc::new(queue));
    let mut renderer = Renderer::new(device.clone(), queue.clone(), FORMAT, W, W);
    renderer.set_tone_mapping(ToneMapping::None, 1.0);
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

    let mut scene = Scene::new();
    let mut ground = Object3D::mesh(Mesh::new(
        BoxGeometry::new(30.0, 0.2, 30.0),
        Material::Shader(ShaderMaterial::new(RECEIVER)),
    ));
    ground.position = Vector3::new(cx, -0.1, 0.0);
    scene.add(ground);
    let mut sm = ShaderMaterial::new(CASTER_FS);
    sm.vertex = Some(CASTER_VS.to_string());
    sm.shadow_vertex = true;
    sm.data = vec![[4.0, 0.0, 0.0, 0.0]];
    let mut caster = Object3D::mesh(Mesh::new(
        BoxGeometry::new(2.0, 2.0, 2.0),
        Material::Shader(sm),
    ));
    caster.position = Vector3::new(cx, 3.0, 0.0);
    caster.cast_shadow = true;
    scene.add(caster);
    let mut sun = DirectionalLight::new(threers::Color::WHITE, 1.0);
    sun.direction = Vector3::new(0.0, -1.0, 0.0);
    sun.cast_shadow = true;
    sun.shadow = ShadowSettings {
        camera_size: 5.0,
        camera_far: 50.0,
        fit_view,
        ..Default::default()
    };
    let mut sun_obj = Object3D::light(sun);
    // A directional light shines from its position toward the origin (three.js's
    // default target), so it stays above the origin to point straight down.
    sun_obj.position = Vector3::new(0.0, 10.0, 0.0);
    scene.add(sun_obj);

    let eye = Vector3::new(cx, 20.0, 20.0);
    let mut cam = PerspectiveCamera::new(50.0, 1.0, 0.5, 200.0);
    cam.position = eye;
    cam.look_at(Vector3::new(cx, 0.0, 0.0));
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

    // Project each probe with the same camera to find its pixel.
    let vp = cam_view_proj(&cam);
    Some(
        probes
            .iter()
            .map(|&dx| {
                let p = Vector3::new(cx + dx, 0.0, 0.0).apply_matrix4(&vp);
                let x = ((p.x * 0.5 + 0.5) * W as f32) as u32;
                let y = ((0.5 - p.y * 0.5) * W as f32) as u32;
                let i = ((y.min(W - 1) * W + x.min(W - 1)) * 4) as usize;
                assert!(px[i + 2] < 128, "probe {dx} hit the caster, not the ground");
                px[i] as f32 / 255.0
            })
            .collect(),
    )
}

fn cam_view_proj(cam: &PerspectiveCamera) -> threers::Matrix4 {
    use threers::Camera;
    cam.projection_matrix().multiply(&cam.view_matrix())
}

#[test]
fn custom_vertex_stage_casts_the_shadow_of_what_it_draws() {
    // The caster's vertex stage moves it from x = 0 to x = +4.
    let Some(lit) = lit_at(0.0, 0.0, &[4.0, 0.0, -4.0]) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    assert!(
        lit[0] < 0.3,
        "no shadow under the moved caster: lit {:?}",
        lit
    );
    assert!(
        lit[1] > 0.7,
        "shadow under where the mesh was, not where it was drawn: {:?}",
        lit
    );
    assert!(lit[2] > 0.7, "open ground shadowed: {:?}", lit);
}

#[test]
fn shadow_box_follows_the_camera_with_fit_view() {
    // 300 units from the origin: outside the fixed 5-unit box.
    let Some(fixed) = lit_at(300.0, 0.0, &[4.0]) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    assert!(
        fixed[0] > 0.7,
        "fixed box reached 300 units out? {:?}",
        fixed
    );
    let fitted = lit_at(300.0, 60.0, &[4.0, -4.0]).unwrap();
    assert!(
        fitted[0] < 0.3,
        "fit_view did not bring the shadow: {:?}",
        fitted
    );
    assert!(
        fitted[1] > 0.7,
        "fit_view shadowed open ground: {:?}",
        fitted
    );
}
