//! The GPU supersample resolve against the CPU average it replaced.
//!
//! `render_to_rgba_resolved` moved supersample averaging off the CPU and into
//! the render pass. That is only a speedup if the pixels are the same, and the
//! way to get them wrong is subtle: averaging sRGB bytes as if they were light
//! darkens every edge, and a supersampled render is mostly edges. So these
//! compare against [`average_blocks_srgb`], the CPU function that used to do it.

use threers::renderer::downsample::{average_blocks_srgb, DownsampleFilter, Downsampler};
use threers::{
    AmbientLight, BoxGeometry, Color, DirectionalLight, HeadlessRenderer, Mesh, Object3D,
    PerspectiveCamera, Quaternion, Scene, StandardMaterial, Vector3,
};

fn renderer(w: u32, h: u32, ss: u32) -> Option<HeadlessRenderer> {
    filtered(w, h, ss, DownsampleFilter::default())
}

fn filtered(w: u32, h: u32, ss: u32, filter: DownsampleFilter) -> Option<HeadlessRenderer> {
    HeadlessRenderer::builder()
        .size(w, h)
        .supersample(ss)
        .downsample_filter(filter)
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

///
/// Only [`DownsampleFilter::Box`] is expected to match: the wider filters reach
/// outside the block on purpose, which is the whole point of them.
#[test]
fn gpu_resolve_matches_the_cpu_average() {
    for ss in [2u32, 3, 4] {
        let (w, h) = (160u32, 120u32);
        let Some(mut r) = filtered(w, h, ss, DownsampleFilter::Box) else {
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
    assert_eq!(ds.filter(), DownsampleFilter::Mitchell, "the default filter");

    // Every filter has to produce a working pipeline, which means every
    // generated shader has to compile — the one thing that can break when the
    // WGSL is assembled from weights at build time rather than written out.
    for filter in [
        DownsampleFilter::Box,
        DownsampleFilter::Tent,
        DownsampleFilter::Mitchell,
        DownsampleFilter::Lanczos2,
    ] {
        for factor in [2u32, 3, 4] {
            let ds = Downsampler::with_filter(device, &tex(96, 96), factor, filter)
                .unwrap_or_else(|| panic!("{filter:?} at {factor}x"));
            assert_eq!(ds.size(), (96 / factor, 96 / factor), "{filter:?} {factor}x");
            assert_eq!(ds.filter(), filter);
            // Box resolves in place; the separable filters need linear scratch.
            assert_eq!(
                ds.scratch_bytes() > 0,
                filter != DownsampleFilter::Box,
                "{filter:?} scratch"
            );
        }
    }
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

/// A shallow silhouette edge, and how far it strays from the straight line it
/// is supposed to be.
///
/// Returns the rms residual in pixels after fitting a line through the
/// sub-pixel edge position of every column — for a synthetic straight edge that
/// residual *is* the staircase, and nothing else. Near-horizontal is the case
/// that matters: an edge climbing one row every `1/slope` columns leaves long
/// runs of identical coverage and then a jump, and the jump is what the eye
/// sees crawling when the geometry turns.
fn staircase_rms(frame: &[u8], w: u32, h: u32) -> (f64, f64) {
    let lum = |x: u32, y: u32| {
        let i = ((y * w + x) * 4) as usize;
        0.2126 * frame[i] as f64 + 0.7152 * frame[i + 1] as f64 + 0.0722 * frame[i + 2] as f64
    };
    // Where the bright quad gives way to the dark background, per column, to
    // sub-pixel precision: the centroid of the vertical gradient. Skip the
    // outermost columns, where the filter's clamped tail is not the geometry.
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for x in 8..w - 8 {
        let mut num = 0.0;
        let mut den = 0.0;
        for y in 1..h {
            let g = (lum(x, y) - lum(x, y - 1)).abs();
            num += y as f64 * g;
            den += g;
        }
        // One clean edge in the column, or nothing worth fitting.
        if den > 40.0 {
            xs.push(x as f64);
            ys.push(num / den);
        }
    }
    assert!(xs.len() > 32, "no edge found to measure ({} columns)", xs.len());
    let n = xs.len() as f64;
    let (mx, my) = (xs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
    let sxy: f64 = xs.iter().zip(&ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let sxx: f64 = xs.iter().map(|x| (x - mx) * (x - mx)).sum();
    let slope = sxy / sxx;
    let rms = (xs
        .iter()
        .zip(&ys)
        .map(|(x, y)| {
            let e = y - (my + slope * (x - mx));
            e * e
        })
        .sum::<f64>()
        / n)
        .sqrt();
    (rms, slope)
}

/// A bright plate filling the bottom of the frame, its top edge a few degrees
/// off horizontal. The one shape that shows the difference between filters.
fn shallow_edge_scene() -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.add_light(AmbientLight::new(Color::WHITE, 1.0));
    let mut mat = StandardMaterial::new(Color::WHITE);
    mat.metalness = 0.0;
    mat.roughness = 1.0;
    let mut plate = Object3D::mesh(Mesh::new(BoxGeometry::new(40.0, 20.0, 0.2), mat.into()));
    // ~3.4° — one row of climb every sixteen columns, the worst case for a
    // filter one pixel wide.
    plate.quaternion = Quaternion::from_axis_angle(Vector3::new(0.0, 0.0, 1.0), 0.06f32.atan());
    plate.position = Vector3::new(0.0, -10.2, 0.0);
    scene.add(plate);
    scene
}

fn flat_camera(w: u32, h: u32) -> PerspectiveCamera {
    let mut c = PerspectiveCamera::new(30.0, w as f32 / h as f32, 0.01, 200.0);
    c.position = Vector3::new(0.0, 0.0, 22.0);
    c.look_at(Vector3::ZERO);
    c
}

/// The fix: a wider reconstruction filter leaves a straight edge straighter.
///
/// The block average is a box one output pixel wide, whose response is `sinc` —
/// first zero at twice Nyquist, so it passes much of what it was meant to
/// remove and that folds back as a staircase. More samples inside the box do
/// not help, because the support is the problem; this is what does.
#[test]
fn a_wider_filter_straightens_a_shallow_edge() {
    let (w, h) = (256u32, 192u32);
    let mut measured = Vec::new();
    for filter in [
        DownsampleFilter::Box,
        DownsampleFilter::Tent,
        DownsampleFilter::Mitchell,
        DownsampleFilter::Lanczos2,
    ] {
        let Some(mut r) = filtered(w, h, 3, filter) else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let mut scene = shallow_edge_scene();
        let cam = flat_camera(w, h);
        let frame = r.render_to_rgba_resolved(&mut scene, &cam);
        assert_eq!(frame.len(), (w * h * 4) as usize, "{filter:?}: resolved size");
        let (rms, slope) = staircase_rms(&frame, w, h);
        // All four see the same geometry, so a filter that moved the edge
        // somewhere else would be a bug, not a smoother resolve.
        assert!(
            slope.abs() > 0.02 && slope.abs() < 0.12,
            "{filter:?}: edge slope {slope:.4} is not the shallow one the test set up"
        );
        measured.push((filter, rms));
    }
    for (filter, rms) in &measured {
        eprintln!("{filter:?}: staircase {rms:.4} px");
    }
    let box_rms = measured[0].1;
    for (filter, rms) in &measured[1..] {
        assert!(
            *rms < box_rms,
            "{filter:?} left a bigger staircase than the box average \
             ({rms:.4} px against {box_rms:.4} px) — the wider filter is not reaching \
             outside the block, or its weights are centred wrong"
        );
    }
}

/// Half a pixel of centring error in the tap offsets is invisible in a still
/// and wrong the moment it is composited against anything resolved another way,
/// so pin the edge's absolute position across filters rather than only its
/// straightness.
#[test]
fn every_filter_puts_the_edge_in_the_same_place() {
    let (w, h) = (256u32, 192u32);
    let mut positions = Vec::new();
    for filter in [
        DownsampleFilter::Box,
        DownsampleFilter::Tent,
        DownsampleFilter::Mitchell,
        DownsampleFilter::Lanczos2,
    ] {
        let Some(mut r) = filtered(w, h, 3, filter) else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let mut scene = shallow_edge_scene();
        let cam = flat_camera(w, h);
        let frame = r.render_to_rgba_resolved(&mut scene, &cam);
        let lum = |x: u32, y: u32| {
            let i = ((y * w + x) * 4) as usize;
            0.2126 * frame[i] as f64 + 0.7152 * frame[i + 1] as f64 + 0.0722 * frame[i + 2] as f64
        };
        // Centre column only, and the gradient centroid over the whole height:
        // a shifted kernel moves this, a wider one does not.
        let x = w / 2;
        let (mut num, mut den) = (0.0, 0.0);
        for y in 1..h {
            let g = (lum(x, y) - lum(x, y - 1)).abs();
            num += y as f64 * g;
            den += g;
        }
        positions.push((filter, num / den));
    }
    let (_, first) = positions[0];
    for (filter, y) in &positions {
        assert!(
            (y - first).abs() < 0.35,
            "{filter:?} put the edge at y={y:.3} against the box average's {first:.3} — \
             that is a centring error in the tap offsets, not a filter difference"
        );
    }
}
