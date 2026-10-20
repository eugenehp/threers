//! `ShaderMaterial::volumes`: a 3D texture in a custom shader, as three.js
//! binds a `Data3DTexture` to a `sampler3D`. A quad reads a 4×1×1 ramp
//! (0, 85, 170, 255) across its width — exactly with `textureLoad`, filtered
//! with `u_vol_samp` — and an unset slot reads the placeholder's zero.
//! Skipped without a GPU adapter.

use std::sync::Arc;
use threers::cameras::OrthographicCamera;
use threers::core::{Mesh, Object3D};
use threers::geometries::PlaneGeometry;
use threers::materials::{Material, ShaderMaterial};
use threers::math::Vector3;
use threers::scene::Scene;
use threers::volume::{Data3DTexture, GpuVolume, VolumeFormat};

const N: u32 = 64;

fn render(r: &mut threers::HeadlessRenderer, vol: Option<Arc<wgpu::TextureView>>, fragment: &str) -> Vec<u8> {
    let mut m = ShaderMaterial::new(fragment);
    m.tone_mapped = false;
    m.volumes = vol.into_iter().collect();
    let mut scene = Scene::new();
    scene.add(Object3D::mesh(Mesh::new(PlaneGeometry::new(2.0, 2.0), Material::Shader(m))));
    let mut cam = OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    cam.position = Vector3::new(0.0, 0.0, 5.0);
    cam.look_at(Vector3::ZERO);
    r.render_to_rgba(&mut scene, &cam)
}

/// Red channel across the middle row, linearised back from the sRGB target.
fn row(rgba: &[u8]) -> Vec<f32> {
    (0..N)
        .map(|x| {
            let c = rgba[(((N / 2) * N + x) * 4) as usize] as f32 / 255.0;
            if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
        })
        .collect()
}

#[test]
fn a_shader_material_reads_a_3d_texture() {
    let Ok(mut r) = threers::HeadlessRenderer::builder().size(N, N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let ramp = Data3DTexture::new(4, 1, 1, VolumeFormat::R8Unorm, vec![0, 85, 170, 255]);
    let vol = GpuVolume::from_texture(r.device(), r.queue(), &ramp);
    let view = Arc::new(vol.view.clone());

    // exact: texel i covers the i-th quarter of the quad
    let exact = row(&render(
        &mut r,
        Some(view.clone()),
        "@fragment fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
            let i = min(i32(in.uv.x * 4.0), 3);
            let v = textureLoad(u_vol0, vec3<i32>(i, 0, 0), 0).r;
            return vec4<f32>(v, v, v, 1.0);
        }",
    ));
    for (q, want) in [0.0f32, 85.0, 170.0, 255.0].iter().enumerate() {
        let got = exact[(q as u32 * N / 4 + N / 8) as usize];
        assert!((got - want / 255.0).abs() < 0.01, "textureLoad quarter {q}: {got}, want {}", want / 255.0);
    }

    // filtered: halfway between texel centres 1 and 2 reads their mean, and
    // the clamp holds the ends at the edge texels instead of wrapping
    let filtered = row(&render(
        &mut r,
        Some(view),
        "@fragment fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
            let v = textureSampleLevel(u_vol0, u_vol_samp, vec3<f32>(in.uv.x, 0.5, 0.5), 0.0).r;
            return vec4<f32>(v, v, v, 1.0);
        }",
    ));
    let mid = (filtered[(N / 2 - 1) as usize] + filtered[(N / 2) as usize]) / 2.0;
    assert!((mid - 127.5 / 255.0).abs() < 0.02, "filtered midpoint {mid}");
    assert!(filtered[0] < 0.02, "the left edge wrapped round to the bright end: {}", filtered[0]);
    assert!(filtered[(N - 1) as usize] > 0.98, "the right edge: {}", filtered[(N - 1) as usize]);

    // an empty slot is the 1x1x1 placeholder, zero
    let empty = row(&render(
        &mut r,
        None,
        "@fragment fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
            let v = textureLoad(u_vol1, vec3<i32>(0, 0, 0), 0).r;
            return vec4<f32>(v, 1.0 - v, 0.0, 1.0);
        }",
    ));
    assert!(empty[(N / 2) as usize] < 0.01, "the empty slot read {}", empty[(N / 2) as usize]);
}
