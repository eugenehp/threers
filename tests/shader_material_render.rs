//! A `ShaderMaterial` renders, and its uniforms and textures arrive.
//!
//! There was no render-time coverage of shader materials at all — the only thing
//! exercising them was an example — which matters because their bind groups are
//! not the standard ones and are easy to break from a distance. The layout in
//! particular: a custom shader's group 2 carries only the sampler and the
//! displacement slot the shared vertex entry needs, because a `ShaderMaterial`
//! has no standard maps of its own. Declaring the eight material textures it
//! cannot use is what pushed the custom layout past WebGPU's sixteen-per-stage
//! limit and made the feature unavailable there.

use std::sync::Arc;

use threers::{
    BoxGeometry, Color, HeadlessRenderer, Material, Mesh, Object3D, PerspectiveCamera, Scene,
    ShaderMaterial, Vector3,
};

const W: u32 = 64;

/// Render one shader material filling the view, and return the centre pixel.
fn render_centre(material: ShaderMaterial) -> Option<[u8; 4]> {
    let mut r = HeadlessRenderer::builder()
        .size(W, W)
        .color_format(threers::wgpu::TextureFormat::Rgba8Unorm)
        .build()
        .ok()?;
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.add(Object3D::mesh(Mesh::new(
        BoxGeometry::new(4.0, 4.0, 0.1),
        Material::Shader(material),
    )));
    let mut cam = PerspectiveCamera::new(45.0, 1.0, 0.1, 100.0);
    cam.position = Vector3::new(0.0, 0.0, 3.0);
    cam.look_at(Vector3::ZERO);
    scene.update_world();
    let px = r.render_to_rgba(&mut scene, &cam);
    let i = ((W / 2 * W + W / 2) * 4) as usize;
    Some([px[i], px[i + 1], px[i + 2], px[i + 3]])
}

/// A custom fragment shader draws, and reads its uniform block.
///
/// The colour comes from `u_data`, so a pass that bound the uniform wrongly
/// shows up as the wrong colour rather than as no pixels — and a pipeline that
/// was never created shows up as the background.
#[test]
fn a_shader_material_draws_with_its_uniforms() {
    let material = ShaderMaterial::new(
        "@fragment\n\
         fn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n\
         return vec4<f32>(u_data.data[0].x, u_data.data[0].y, u_data.data[0].z, 1.0);\n\
         }\n",
    )
    .with_data(vec![[0.25, 0.5, 0.75, 1.0]]);

    let Some(px) = render_centre(material) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    // An `Rgba8Unorm` target stores what the shader wrote, so these are the
    // uniform times 255 within a rounding step.
    for (got, want) in px[..3].iter().zip([64u8, 128, 191]) {
        assert!(
            got.abs_diff(want) <= 2,
            "centre pixel is {px:?}; expected about [64, 128, 191] from u_data"
        );
    }
}

/// A user texture arrives in `u_tex0`.
///
/// The four user textures live in group 1 alongside the mesh uniform. They are
/// the reason the custom layout is tight, so a change that frees budget
/// elsewhere must not quietly drop them.
#[test]
fn a_shader_material_samples_its_own_texture() {
    let Some(mut r) = HeadlessRenderer::builder()
        .size(W, W)
        .color_format(threers::wgpu::TextureFormat::Rgba8Unorm)
        .build()
        .ok()
    else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    // A 1×1 texture of a colour nothing else in the scene produces.
    let tex = r.device().create_texture(&threers::wgpu::TextureDescriptor {
        label: Some("user tex"),
        size: threers::wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: threers::wgpu::TextureDimension::D2,
        format: threers::wgpu::TextureFormat::Rgba8Unorm,
        usage: threers::wgpu::TextureUsages::TEXTURE_BINDING
            | threers::wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    r.queue().write_texture(
        threers::wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: threers::wgpu::Origin3d::ZERO,
            aspect: threers::wgpu::TextureAspect::All,
        },
        &[200u8, 40, 120, 255],
        threers::wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        threers::wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    let view = Arc::new(tex.create_view(&threers::wgpu::TextureViewDescriptor::default()));

    let material = ShaderMaterial::new(
        "@fragment\n\
         fn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n\
         return vec4<f32>(textureSample(u_tex0, u_samp, in.uv).rgb, 1.0);\n\
         }\n",
    )
    .with_textures(vec![view]);

    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.add(Object3D::mesh(Mesh::new(
        BoxGeometry::new(4.0, 4.0, 0.1),
        Material::Shader(material),
    )));
    let mut cam = PerspectiveCamera::new(45.0, 1.0, 0.1, 100.0);
    cam.position = Vector3::new(0.0, 0.0, 3.0);
    cam.look_at(Vector3::ZERO);
    scene.update_world();
    let px = r.render_to_rgba(&mut scene, &cam);
    let i = ((W / 2 * W + W / 2) * 4) as usize;
    let got = [px[i], px[i + 1], px[i + 2]];
    for (g, want) in got.iter().zip([200u8, 40, 120]) {
        assert!(
            g.abs_diff(want) <= 3,
            "centre pixel is {got:?}; expected about [200, 40, 120] from u_tex0"
        );
    }
}
