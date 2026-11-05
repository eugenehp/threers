//! Volume slices on the native Metal backend draw what they draw on wgpu:
//! a float volume through window/level and a 64-bit label volume over it,
//! drawn over a `HeadlessRenderer`'s frame through `marks_gpu` and
//! `mark_target`, as `examples/volume_slices.rs` does.

#![cfg(metal_backend)]

use threers::math::{Quaternion, Vector3};
use threers::renderer::PixelRect;
use threers::volume::{plane_to_voxel, Data3DTexture, GpuVolume, SliceParams, SliceRenderer};
use threers::{Color, HeadlessRenderer, PerspectiveCamera, Scene};

const SIZE: u32 = 64;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

fn build(metal: bool) -> HeadlessRenderer {
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
        .color_format(FORMAT)
        .build()
        .expect("headless renderer");
    unsafe { std::env::remove_var("THREERS_BACKEND") };
    hr.set_msaa(1);
    assert_eq!(hr.is_native_metal(), metal);
    hr
}

/// The pixel at the middle of voxel `(x, y)`'s 16 px square.
fn pixel(rgba: &[u8], x: u32, y: u32) -> [u8; 4] {
    let (px, py) = (x * 16 + 8, y * 16 + 8);
    let i = ((py * SIZE + px) * 4) as usize;
    rgba[i..i + 4].try_into().unwrap()
}

#[test]
fn volume_slices_on_metal_draw_what_they_draw_on_wgpu() {
    if threers::metal::MetalDevice::new().is_err() {
        eprintln!("skipping: no Metal device");
        return;
    }
    // value = x / 3; labels 0 in the left half, two distinct 64-bit labels
    // on the right, the high word set on one of them.
    let values: Vec<f32> = (0..64).map(|i| (i % 4) as f32 / 3.0).collect();
    let labels: Vec<u64> = (0..64u64)
        .map(|i| match i % 4 {
            0 | 1 => 0,
            2 => 5,
            _ => 0x0000_0001_0000_0007,
        })
        .collect();
    let to_voxel = plane_to_voxel(
        Vector3::new(2.0, 2.0, 1.5),
        Quaternion::identity(),
        4.0 / SIZE as f32,
        (SIZE as f32, SIZE as f32),
        true,
    );
    let rect = PixelRect {
        x: 0,
        y: 0,
        width: SIZE,
        height: SIZE,
    };
    let frames: Vec<(Vec<u8>, Vec<u8>)> = [false, true]
        .into_iter()
        .map(|metal| {
            let mut hr = build(metal);
            let gpu = hr.marks_gpu();
            assert_eq!(gpu.is_metal(), metal);
            let density =
                GpuVolume::from_texture_on(&gpu, &Data3DTexture::from_f32(4, 4, 4, &values));
            let segments =
                GpuVolume::from_texture_on(&gpu, &Data3DTexture::from_u64(4, 4, 4, &labels));
            assert_eq!(density.texture.size().depth_or_array_layers, 4);
            let grey = SliceRenderer::window_level(&gpu, FORMAT);
            let tint = SliceRenderer::labels(&gpu, FORMAT);
            let grey_binding = grey.create_binding(&gpu, &density);
            let tint_binding = tint.create_binding(&gpu, &segments);
            let mut blank = Scene::new();
            blank.background = Color::new(0.0, 0.0, 0.0);
            let camera = PerspectiveCamera::new(50.0, 1.0, 0.1, 10.0);
            let mut draw = |labelled: bool| {
                hr.render(&mut blank, &camera);
                let target = hr.mark_target();
                let mut encoder = gpu.create_command_encoder(&Default::default());
                grey.update_on(
                    &gpu,
                    &grey_binding,
                    &SliceParams::window(to_voxel, 0.0, 1.0),
                );
                grey.draw_on(&mut encoder, &target.color, rect, &grey_binding);
                if labelled {
                    tint.update_on(&gpu, &tint_binding, &SliceParams::labels(to_voxel, 7, 1.0));
                    tint.draw_on(&mut encoder, &target.color, rect, &tint_binding);
                }
                gpu.submit([encoder.finish()]);
                drop(target);
                hr.read_rgba()
            };
            (draw(false), draw(true))
        })
        .collect();
    for (backend, (grey, _)) in ["wgpu", "Metal"].iter().zip(&frames) {
        // The ramp through an sRGB target: display-referred values land as
        // they are.
        for x in 0..4 {
            let expected = (x as f32 / 3.0 * 255.0).round() as u8;
            let p = pixel(grey, x, 2);
            assert!(
                p[0].abs_diff(expected) <= 1 && p[0] == p[1] && p[1] == p[2],
                "{backend}: x={x}: {p:?}"
            );
        }
    }
    let (wgpu, metal) = (&frames[0], &frames[1]);
    assert_eq!(wgpu.0, metal.0, "the window/level slice on Metal is wgpu's");
    // Labels 0 leave the ramp; the two others are opaque colours of their own.
    let (a, b) = (pixel(&wgpu.1, 2, 1), pixel(&wgpu.1, 3, 1));
    assert_eq!(
        pixel(&wgpu.1, 0, 1),
        pixel(&wgpu.0, 0, 1),
        "label 0 is clear"
    );
    assert_ne!(a, b, "two labels, two colours");
    assert_ne!(a, pixel(&wgpu.0, 2, 1), "a label tints its voxel");
    let voxels = |f: &[u8]| (0..16).map(|i| pixel(f, i % 4, i / 4)).collect::<Vec<_>>();
    assert_eq!(
        voxels(&wgpu.1),
        voxels(&metal.1),
        "the label slice on Metal is wgpu's"
    );
    assert!(
        wgpu.1 == metal.1,
        "the label slice on Metal is wgpu's between voxel centres too"
    );
}

/// A `ShaderMaterial` reads a volume made on the renderer's own GPU — exactly
/// with `textureLoad`, filtered with `textureSampleLevel`, and its size — the
/// same on both backends.
#[test]
fn a_shader_material_reads_a_volume_on_either_backend() {
    use threers::{
        Material, Mesh, Object3D, OrthographicCamera, PlaneGeometry, ShaderMaterial, Vector3 as V,
    };
    if threers::metal::MetalDevice::new().is_err() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fragment = "@fragment fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
        let size = vec3<f32>(textureDimensions(u_vol0));
        let i = min(i32(in.uv.x * size.x), i32(size.x) - 1);
        let exact = textureLoad(u_vol0, vec3<i32>(i, 0, 0), 0).r;
        let filtered = textureSampleLevel(u_vol0, u_vol_samp, vec3<f32>(in.uv.x, 0.5, 0.5), 0.0).r;
        let empty = textureLoad(u_vol1, vec3<i32>(0, 0, 0), 0).r;
        return vec4<f32>(exact, filtered, empty + size.x / 8.0, 1.0);
    }";
    let ramp = threers::volume::Data3DTexture::new(
        4,
        1,
        1,
        threers::volume::VolumeFormat::R8Unorm,
        vec![0, 85, 170, 255],
    );
    let mut camera = OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    camera.position = V::new(0.0, 0.0, 5.0);
    camera.look_at(V::ZERO);
    let frames: Vec<Vec<u8>> = [false, true]
        .into_iter()
        .map(|metal| {
            let mut hr = build(metal);
            let volume = GpuVolume::from_texture_on(&hr.marks_gpu(), &ramp);
            let mut m = ShaderMaterial::new(fragment);
            m.tone_mapped = false;
            m.volumes = vec![volume.view.clone()];
            m.validate(true).expect("the volume shader's WGSL");
            let mut scene = Scene::new();
            scene.add(Object3D::mesh(Mesh::new(
                PlaneGeometry::new(2.0, 2.0),
                Material::Shader(m),
            )));
            let frame = hr.render_to_rgba(&mut scene, &camera);
            assert_eq!(
                hr.is_native_metal(),
                metal,
                "a volume keeps the renderer on its backend"
            );
            frame
        })
        .collect();
    let at = |f: &[u8], x: u32| {
        let i = (((SIZE / 2) * SIZE + x) * 4) as usize;
        [f[i], f[i + 1], f[i + 2]]
    };
    for (backend, f) in ["wgpu", "Metal"].iter().zip(&frames) {
        // Texel 1 of 4 (85/255) through the sRGB target: 156. The size,
        // 4 / 8, encodes to 188; an empty slot adds nothing.
        let p = at(f, SIZE * 3 / 8);
        assert!(
            p[0].abs_diff(156) <= 2 && p[2].abs_diff(188) <= 2,
            "{backend}: {p:?}"
        );
    }
    let differing = frames[0]
        .chunks_exact(4)
        .zip(frames[1].chunks_exact(4))
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        differing < SIZE as usize,
        "{differing} pixels differ between wgpu and Metal"
    );
}
