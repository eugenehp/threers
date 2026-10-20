//! three.js `renderTarget.samples`: a render target drawn multisampled
//! whatever the renderer's own MSAA, its colour resolved into it — and its
//! depth, which post-processing and refraction glass read as they would a
//! single-sampled target's. The depth comes two ways: resolved by a shader
//! where one can read a multisampled texture, redrawn where it cannot
//! (WebGL2; forced here with `force_depth_replay`).
//!
//! Skipped without a GPU adapter.

use threers::materials::TransparencyMode;
use threers::prelude::*;
use threers::RenderTarget;

const N: u32 = 64;

/// A tilted white box on black, its far half behind a red one: edges for
/// MSAA to soften, depth for the resolve to keep.
fn boxes() -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    let mut a = Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.2, 1.2, 1.2),
        Material::Basic(BasicMaterial::new(Color::WHITE)),
    ));
    a.rotate_x(0.4);
    a.rotate_y(0.6);
    a.rotate_z(0.2);
    scene.add(a);
    let mut b = Object3D::mesh(Mesh::new(
        BoxGeometry::new(0.6, 0.6, 0.6),
        Material::Basic(BasicMaterial::new(Color::new(1.0, 0.0, 0.0))),
    ));
    b.position = Vector3::new(0.5, 0.4, 1.0);
    scene.add(b);
    scene
}

fn camera() -> PerspectiveCamera {
    let mut c = PerspectiveCamera::new(40.0, 1.0, 0.5, 20.0);
    c.position = Vector3::new(0.0, 0.0, 5.0);
    c.look_at(Vector3::ZERO);
    c
}

/// A texture's bytes, tightly packed, 4 per texel (RGBA8 or depth f32).
fn read(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    aspect: wgpu::TextureAspect,
) -> Vec<u8> {
    let padded = (N * 4).div_ceil(256) * 256;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (padded * N) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(N),
            },
        },
        wgpu::Extent3d {
            width: N,
            height: N,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([enc.finish()]);
    buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let data = buf.slice(..).get_mapped_range().expect("mapped").to_vec();
    (0..N)
        .flat_map(|y| data[(y * padded) as usize..(y * padded + N * 4) as usize].to_vec())
        .collect()
}

fn depths(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Pixels neither background nor a box's flat colour: MSAA's edge blends.
fn blended(rgba: &[u8]) -> usize {
    rgba.chunks(4)
        .filter(|p| {
            let c = [p[0], p[1], p[2]];
            c != [0, 0, 0] && c != [255, 255, 255] && c != [255, 0, 0]
        })
        .count()
}

#[test]
fn a_render_target_with_samples_is_multisampled_colour_and_depth() {
    let Ok(mut hr) = HeadlessRenderer::builder().size(N, N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let renderer = hr.renderer();
    let (device, queue) = (renderer.device_arc(), renderer.queue_arc());
    assert_eq!(
        renderer.msaa(),
        1,
        "the renderer's own MSAA stays off throughout"
    );
    let draw = |renderer: &mut Renderer, samples: u32, replay: bool| {
        let rt = RenderTarget::new(&device, N, N, renderer.color_format());
        rt.set_samples(samples);
        renderer.force_depth_replay(replay);
        renderer.render_to(&mut boxes(), &camera(), &rt);
        renderer.force_depth_replay(false);
        let ms = renderer.last_frame_multisampled();
        let colour = read(&device, &queue, &rt.color_texture, wgpu::TextureAspect::All);
        let depth = depths(&read(
            &device,
            &queue,
            &rt.depth_texture,
            wgpu::TextureAspect::DepthOnly,
        ));
        (ms, colour, depth)
    };
    let (ms0, c0, d0) = draw(renderer, 0, false);
    assert!(!ms0, "samples 0 drew multisampled");
    assert_eq!(blended(&c0), 0, "single-sampled edges are hard");

    for replay in [false, true] {
        let (ms, c, d) = draw(renderer, 4, replay);
        let how = if replay { "redrawn" } else { "resolved" };
        assert!(ms, "samples 4 ({how} depth): drawn single-sampled");
        assert_eq!(
            renderer.msaa(),
            1,
            "the target's samples leaked into the renderer's MSAA"
        );
        let soft = blended(&c);
        assert!(
            soft > 40,
            "samples 4 ({how} depth): {soft} blended edge pixels"
        );
        // The depth is the scene's, not the clear value: away from the edges
        // — where the samples straddle two surfaces and the pixel centre
        // need not agree with any one of them — it matches the
        // single-sampled render's.
        let interior = |x: u32, y: u32| {
            let a = d0[(y * N + x) as usize];
            a < 0.999
                && (y - 1..=y + 1)
                    .all(|v| (x - 1..=x + 1).all(|u| (d0[(v * N + u) as usize] - a).abs() < 2e-3))
        };
        let (mut same, mut covered, mut worst) = (0, 0, 0f32);
        for y in 1..N - 1 {
            for x in 1..N - 1 {
                if interior(x, y) {
                    covered += 1;
                    let e = (d0[(y * N + x) as usize] - d[(y * N + x) as usize]).abs();
                    worst = worst.max(e);
                    // A sample's depth, not the centre's: within the slope
                    // across a pixel. The clear value is ~0.1 away.
                    if e < 1e-3 {
                        same += 1;
                    }
                }
            }
        }
        assert!(covered > 400, "only {covered} interior pixels");
        assert!(
            same == covered,
            "samples 4 ({how} depth): {same} of {covered} interior depths match the single-sampled render's (worst off by {worst})"
        );
        // Background stays the clear value.
        assert_eq!(d[0], d0[0], "{how}: background depth");
        // And the colour matches it inside the shapes.
        let centre = ((N / 2 * N + N / 2) * 4) as usize;
        assert_eq!(
            c[centre..centre + 3],
            c0[centre..centre + 3],
            "{how}: centre colour"
        );
    }
}

/// Refraction glass in a multisampled target tests against the target's
/// resolved depth: a box in front of the ball stays in front.
#[test]
fn glass_in_a_multisampled_target_stays_behind_what_is_in_front() {
    let Ok(mut hr) = HeadlessRenderer::builder().size(N, N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let renderer = hr.renderer();
    let (device, queue) = (renderer.device_arc(), renderer.queue_arc());
    for replay in [false, true] {
        let mut scene = Scene::new();
        scene.background = Color::from_hex(0x203040);
        scene.add_light(AmbientLight::new(Color::WHITE, 1.0));
        let mut wall = Object3D::mesh(Mesh::new(
            PlaneGeometry::new(8.0, 8.0),
            Material::Basic(BasicMaterial::new(Color::from_hex(0x2040ff))),
        ));
        wall.position = Vector3::new(0.0, 0.0, -3.0);
        scene.add(wall);
        let glass = PhysicalMaterial::new(Color::new(0.05, 0.9, 0.05))
            .with_roughness(0.05)
            .with_transmission(1.0, 1.5, 1.0)
            .with_transparency(TransparencyMode::Refract);
        scene.add(Object3D::mesh(Mesh::new(
            SphereGeometry::new(1.0, 32, 16),
            Material::Physical(glass),
        )));
        let mut front = Object3D::mesh(Mesh::new(
            BoxGeometry::new(0.6, 0.6, 0.6),
            Material::Basic(BasicMaterial::new(Color::from_hex(0xff0000))),
        ));
        front.position = Vector3::new(0.0, 0.0, 1.8);
        scene.add(front);
        let rt = RenderTarget::new(&device, N, N, renderer.color_format());
        rt.set_samples(4);
        renderer.force_depth_replay(replay);
        renderer.render_to(&mut scene, &camera(), &rt);
        renderer.force_depth_replay(false);
        assert!(
            renderer.last_frame_multisampled(),
            "replay {replay}: not multisampled"
        );
        let px = read(&device, &queue, &rt.color_texture, wgpu::TextureAspect::All);
        let i = ((N / 2 * N + N / 2) * 4) as usize;
        let c = [px[i], px[i + 1], px[i + 2]];
        assert!(
            c[0] > 240 && c[1] < 20 && c[2] < 20,
            "replay {replay}: the box in front reads {c:?} — the glass drew over it"
        );
    }
}

/// A target the multisampled colour cannot resolve into — half-float, where
/// the renderer's is 8-bit — is drawn single-sampled rather than not at all.
#[test]
fn a_half_float_target_with_samples_still_draws() {
    let Ok(mut hr) = HeadlessRenderer::builder().size(N, N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let renderer = hr.renderer();
    let device = renderer.device_arc();
    let rt = RenderTarget::new(&device, N, N, wgpu::TextureFormat::Rgba16Float);
    rt.set_samples(4);
    renderer.render_to(&mut boxes(), &camera(), &rt);
    assert!(!renderer.last_frame_multisampled());
    assert_eq!(rt.resized(N, N).samples(), 4, "setSize keeps samples");
}

/// Alternating a multisampled picture and a multisampled target of another
/// size draws both right, every frame.
#[test]
fn a_multisampled_canvas_and_target_alternate() {
    let Ok(mut hr) = HeadlessRenderer::builder().size(N * 2, N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    hr.set_msaa(4);
    let (device, queue) = (hr.renderer().device_arc(), hr.renderer().queue_arc());
    let rt = RenderTarget::new(&device, N, N, hr.renderer().color_format());
    rt.set_samples(4);
    let mut first = None;
    for _ in 0..3 {
        let picture = hr.render_to_rgba(&mut boxes(), &camera());
        assert!(hr.renderer().last_frame_multisampled());
        hr.renderer().render_to(&mut boxes(), &camera(), &rt);
        assert!(hr.renderer().last_frame_multisampled());
        let target = read(&device, &queue, &rt.color_texture, wgpu::TextureAspect::All);
        assert!(blended(&target) > 40);
        match &first {
            None => first = Some((picture, target)),
            Some((p, t)) => assert!(*p == picture && *t == target, "a frame drew differently"),
        }
    }
}
