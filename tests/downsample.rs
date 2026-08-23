//! The GPU supersample resolve against the CPU average it replaced.
//!
//! `render_to_rgba_resolved` moved supersample averaging off the CPU and into
//! the render pass. That is only a speedup if the pixels are the same, and the
//! way to get them wrong is subtle: averaging sRGB bytes as if they were light
//! darkens every edge, and a supersampled render is mostly edges. So these
//! compare against [`average_blocks_srgb`], the CPU function that used to do it.

use threers::renderer::downsample::{average_blocks_srgb, Downsampler};
use threers::{
    AmbientLight, BoxGeometry, Color, DirectionalLight, HeadlessRenderer, Mesh, Object3D,
    PerspectiveCamera, Scene, StandardMaterial, Vector3,
};

fn renderer(w: u32, h: u32, ss: u32) -> Option<HeadlessRenderer> {
    HeadlessRenderer::builder()
        .size(w, h)
        .supersample(ss)
        .color_format(wgpu::TextureFormat::Rgba8Unorm)
        .build()
        .ok()
}

/// A scene with plenty of high-contrast edges, which is where a wrong average
/// shows up. Flat colour would pass any implementation.
fn edgy_scene() -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x101820);
    scene.add_light(
        DirectionalLight::new(Color::WHITE, 3.0)
            .with_direction(Vector3::new(-0.4, -0.7, -0.5).normalize()),
    );
    scene.add_light(AmbientLight::new(Color::from_hex(0x223344), 0.4));
    for i in 0..48 {
        let a = i as f32 * 0.41;
        let mut mat = StandardMaterial::new(Color::from_hex(0xd8dee9));
        mat.metalness = 0.3;
        mat.roughness = 0.35;
        let mut o = Object3D::mesh(Mesh::new(BoxGeometry::new(0.35, 0.35, 0.35), mat.into()));
        o.position = Vector3::new(a.cos() * 1.5, (a * 1.7).sin() * 1.2, a.sin() * 1.5);
        scene.add(o);
    }
    scene
}

fn camera(w: u32, h: u32) -> PerspectiveCamera {
    let mut c = PerspectiveCamera::new(45.0, w as f32 / h as f32, 0.01, 100.0);
    c.position = Vector3::new(3.0, 2.0, 3.5);
    c.look_at(Vector3::ZERO);
    c
}

#[test]
fn gpu_resolve_matches_the_cpu_average() {
    for ss in [2u32, 3, 4] {
        let (w, h) = (160u32, 120u32);
        let Some(mut r) = renderer(w, h, ss) else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let mut scene = edgy_scene();
        let cam = camera(w, h);

        let (rw, rh) = r.render_size();
        let cpu = average_blocks_srgb(&r.render_to_rgba(&mut scene, &cam), rw, rh, ss);
        let gpu = r.render_to_rgba_resolved(&mut scene, &cam);

        assert_eq!(gpu.len(), (w * h * 4) as usize, "ss{ss}: resolved size");
        assert_eq!(cpu.len(), gpu.len(), "ss{ss}: sizes disagree");

        let worst = cpu
            .iter()
            .zip(&gpu)
            .map(|(a, b)| (*a as i32 - *b as i32).abs())
            .max()
            .unwrap();
        // The GPU rounds on the way into an 8-bit target and the CPU rounds in
        // `to_srgb`; both round to nearest, so they agree exactly. Allowing 1
        // would still catch the failure that matters (averaging encoded bytes
        // is off by tens), but there is no reason to grant slack that is not
        // being used.
        assert_eq!(worst, 0, "ss{ss}: GPU resolve differs from the CPU average");
    }
}

/// Averaging in the wrong space is the whole risk, so prove the test above
/// could see it: a naive mean of the sRGB bytes is far off.
#[test]
fn averaging_encoded_bytes_would_be_visibly_darker() {
    // Half black, half white — the worst case for the transfer function, and
    // what a 2× supersampled edge actually looks like.
    let src: Vec<u8> = [[0u8, 0, 0, 255], [255, 255, 255, 255]]
        .iter()
        .cycle()
        .take(4)
        .flatten()
        .copied()
        .collect();
    let correct = average_blocks_srgb(&src, 2, 2, 2);
    // Mean of 0 and 255 in linear light is 0.5, which encodes to 188.
    assert!(
        (correct[0] as i32 - 188).abs() <= 1,
        "linear-light average of black and white should encode near 188, got {}",
        correct[0]
    );
    // The naive version would give 128 — a 60-level error, far outside the
    // exact agreement asserted above.
    assert!(correct[0] > 170);
}

/// The pipelined readback must resolve too, and at OUTPUT size.
///
/// `read_rgba_pipelined` overlaps the readback with the next draw but reads the
/// target as it stands — the supersampled one. That is the only pipelined call
/// there was, so a caller writing an animation reached for it and then averaged
/// on the CPU, which is the cost this whole module exists to remove: in one
/// real case it was 214 ms a frame against 34 ms for everything else combined.
/// Worse, nothing complained — the frames were simply `factor` times too large,
/// and a video came out at the wrong resolution with the same aliasing per
/// pixel it started with.
///
/// So this pins both halves of the contract: same pixels as the blocking
/// resolve, and the output size rather than the render size.
#[test]
fn pipelined_resolve_matches_the_blocking_one() {
    for ss in [2u32, 3] {
        let (w, h) = (160u32, 120u32);
        let Some(mut r) = renderer(w, h, ss) else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let mut scene = edgy_scene();
        let cam = camera(w, h);

        let blocking = r.render_to_rgba_resolved(&mut scene, &cam);

        // Pipelined hands back the PREVIOUS frame, so the first call is None
        // and the frame arrives on the second. Same scene and camera both
        // times, so the pixels must match the blocking call exactly.
        let mut got = None;
        for _ in 0..2 {
            r.render(&mut scene, &cam);
            if let Some(f) = r.read_rgba_resolved_pipelined() {
                got = Some(f);
            }
        }
        let piped = got.or_else(|| r.finish_readback()).expect("a frame");

        assert_eq!(
            piped.len(),
            (w * h * 4) as usize,
            "ss{ss}: pipelined resolve must return the OUTPUT size, not size x {ss}"
        );
        assert_eq!(piped.len(), blocking.len(), "ss{ss}: sizes disagree");
        let worst = blocking
            .iter()
            .zip(&piped)
            .map(|(a, b)| (*a as i32 - *b as i32).abs())
            .max()
            .unwrap();
        assert_eq!(
            worst, 0,
            "ss{ss}: pipelined resolve differs from the blocking one"
        );
    }
}

/// Without supersampling it must still behave like the plain pipelined read:
/// one frame behind, at output size, and not accidentally averaging anything.
#[test]
fn pipelined_resolve_is_a_no_op_without_supersampling() {
    let (w, h) = (96u32, 64u32);
    let Some(mut r) = renderer(w, h, 1) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut scene = edgy_scene();
    let cam = camera(w, h);
    let direct = r.render_to_rgba(&mut scene, &cam);

    let mut got = None;
    for _ in 0..2 {
        r.render(&mut scene, &cam);
        if let Some(f) = r.read_rgba_resolved_pipelined() {
            got = Some(f);
        }
    }
    let piped = got.or_else(|| r.finish_readback()).expect("a frame");
    assert_eq!(piped.len(), (w * h * 4) as usize);
    assert_eq!(
        piped, direct,
        "ss1: pipelined resolve should change nothing"
    );
}

#[test]
fn downsampler_declines_what_it_cannot_do() {
    let Some(r) = renderer(64, 64, 2) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let device = r.device();
    let tex = |w: u32, h: u32| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    };
    // Nothing to average.
    assert!(Downsampler::new(device, &tex(64, 64), 1).is_none());
    // A partial block at the edge would average against undefined texels.
    assert!(Downsampler::new(device, &tex(65, 64), 2).is_none());
    assert!(Downsampler::new(device, &tex(64, 65), 2).is_none());
    // The ordinary case works, and halves both dimensions.
    let ds = Downsampler::new(device, &tex(64, 64), 2).expect("2× of a 64×64 target");
    assert_eq!(ds.size(), (32, 32));
    assert_eq!(ds.factor(), 2);
}

/// With no supersampling the resolved read is just the frame, at output size.
#[test]
fn resolve_is_a_no_op_without_supersampling() {
    let (w, h) = (96u32, 64u32);
    let Some(mut r) = renderer(w, h, 1) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut scene = edgy_scene();
    let cam = camera(w, h);
    let plain = r.render_to_rgba(&mut scene, &cam);
    let resolved = r.render_to_rgba_resolved(&mut scene, &cam);
    assert_eq!(plain.len(), (w * h * 4) as usize);
    assert_eq!(plain, resolved);
}
