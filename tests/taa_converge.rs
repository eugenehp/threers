//! Converging accumulation: many jittered samples, one output-size buffer.
//!
//! `set_taa_converge` turns the temporal pass from a feedback blend into a
//! running mean, for the case where the camera is parked and the renderer is
//! being asked for one settled image. The two things that make ordinary TAA
//! work on a moving picture — reprojection and the neighbourhood clamp — are
//! exactly what stops it converging when nothing is moving, so the converge
//! path skips both.
//!
//! The test subject is a shallow edge, which is what shows undersampling most
//! plainly: an edge climbing one row every `1/slope` columns leaves long runs
//! of identical coverage and then a jump, and the jump is what crawls.

use threers::{
    AmbientLight, BoxGeometry, Color, HeadlessRenderer, Mesh, Object3D, PerspectiveCamera,
    Quaternion, Scene, StandardMaterial, Vector3,
};

const W: u32 = 256;
const H: u32 = 192;

fn scene() -> Scene {
    let mut s = Scene::new();
    s.background = Color::from_hex(0x000000);
    s.add_light(AmbientLight::new(Color::WHITE, 1.0));
    let mut mat = StandardMaterial::new(Color::WHITE);
    mat.metalness = 0.0;
    mat.roughness = 1.0;
    let mut plate = Object3D::mesh(Mesh::new(BoxGeometry::new(40.0, 20.0, 0.2), mat.into()));
    plate.quaternion = Quaternion::from_axis_angle(Vector3::new(0.0, 0.0, 1.0), 0.06f32.atan());
    plate.position = Vector3::new(0.0, -10.2, 0.0);
    s.add(plate);
    s
}

fn camera() -> PerspectiveCamera {
    let mut c = PerspectiveCamera::new(30.0, W as f32 / H as f32, 0.01, 200.0);
    c.position = Vector3::new(0.0, 0.0, 22.0);
    c.look_at(Vector3::ZERO);
    c
}

/// Rms deviation of the edge from the straight line it is supposed to be. For a
/// synthetic straight edge that residual *is* the sampling error.
///
/// Measured in **linear light**, which is not a detail. The edge position comes
/// from a gradient centroid, and a centroid only reads as coverage if the
/// values are linear in coverage — which sRGB-encoded bytes are not. Measured
/// on the encoded bytes instead, the same frame scores 0.078 px where the
/// linear values say 0.022: the extra 0.056 is the transfer curve, not the
/// renderer. That mattered once: making the temporal mean gamma-correct moved
/// this metric from 0.0216 to 0.0803 while leaving the edge exactly as straight
/// as it had been, and an unconverted metric would have called that a
/// regression and sent the fix back.
fn staircase(frame: &[u8]) -> f64 {
    let decode = |b: u8| {
        let v = b as f64 / 255.0;
        255.0
            * if v <= 0.04045 {
                v / 12.92
            } else {
                ((v + 0.055) / 1.055).powf(2.4)
            }
    };
    let lum = |x: u32, y: u32| {
        let i = ((y * W + x) * 4) as usize;
        0.2126 * decode(frame[i]) + 0.7152 * decode(frame[i + 1]) + 0.0722 * decode(frame[i + 2])
    };
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for x in 8..W - 8 {
        let (mut num, mut den) = (0.0, 0.0);
        for y in 1..H {
            let g = (lum(x, y) - lum(x, y - 1)).abs();
            num += y as f64 * g;
            den += g;
        }
        if den > 40.0 {
            xs.push(x as f64);
            ys.push(num / den);
        }
    }
    assert!(xs.len() > 32, "no edge found ({} columns)", xs.len());
    let n = xs.len() as f64;
    let (mx, my) = (xs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
    let sxy: f64 = xs.iter().zip(&ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let sxx: f64 = xs.iter().map(|x| (x - mx) * (x - mx)).sum();
    let slope = sxy / sxx;
    (xs.iter()
        .zip(&ys)
        .map(|(x, y)| {
            let e = y - (my + slope * (x - mx));
            e * e
        })
        .sum::<f64>()
        / n)
        .sqrt()
}

fn render(samples: u32, converge: bool) -> Option<f64> {
    render_msaa(samples, converge, 1)
}

fn render_msaa(samples: u32, converge: bool, msaa: u32) -> Option<f64> {
    let mut r = HeadlessRenderer::builder()
        .size(W, H)
        .taa(samples > 1)
        .color_format(threers::wgpu::TextureFormat::Rgba8Unorm)
        .build()
        .ok()?;
    r.set_msaa(msaa);
    r.renderer().set_taa_converge(converge);
    r.renderer().reset_taa();
    let (mut sc, cam) = (scene(), camera());
    for _ in 0..samples.max(1) {
        r.render(&mut sc, &cam);
    }
    Some(staircase(&r.read_rgba()))
}

#[test]
fn accumulation_converges_on_a_shallow_edge() {
    let Some(one) = render(1, false) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let eight = render(8, true).unwrap();
    let thirty_two = render(32, true).unwrap();
    eprintln!("staircase: 1 sample {one:.4} px, 8 {eight:.4}, 32 {thirty_two:.4}");

    assert!(
        eight < one * 0.8,
        "8 accumulated samples left {eight:.4} px against {one:.4} for one — not converging"
    );
    assert!(
        thirty_two < eight,
        "32 samples ({thirty_two:.4} px) should beat 8 ({eight:.4}); the mean has a floor under it"
    );
}

/// Without the converge flag the same loop must NOT behave like a mean: the
/// clamp pins it near a single sample. This is what makes the new mode worth
/// having rather than something the existing TAA already did.
#[test]
fn ordinary_taa_does_not_converge_the_same_way() {
    let Some(converged) = render(32, true) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let clamped = render(32, false).unwrap();
    eprintln!("32 samples: converge {converged:.4} px, ordinary TAA {clamped:.4} px");
    assert!(
        converged < clamped,
        "converge mode ({converged:.4} px) should beat clamped TAA ({clamped:.4} px) on a still pose"
    );
}

/// One sample with the flag set must be the plain frame — the mode may not
/// change anything until there is something to average.
#[test]
fn a_single_sample_is_unchanged() {
    let Some(plain) = render(1, false) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let one = render(1, true).unwrap();
    assert!(
        (plain - one).abs() < 1e-9,
        "one converged sample is {one:.6} px against {plain:.6} for the plain render"
    );
}

/// Multisampling and accumulation have to compose.
///
/// They did not: the multisampled path is surface-style and passed no colour
/// source, which is what the temporal pass is gated on, so turning both on
/// silently got you only the first — no error, no warning, and a quality
/// setting that did nothing. MSAA resolves coverage and accumulation resolves
/// shading; a product render wants both.
#[test]
fn multisampling_does_not_switch_accumulation_off() {
    let Some(one) = render_msaa(1, false, 4) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let many = render_msaa(32, true, 4).unwrap();
    eprintln!("msaa 4x: 1 sample {one:.4} px, 32 accumulated {many:.4} px");
    assert!(
        many < one * 0.6,
        "with 4x MSAA, 32 accumulated samples left {many:.4} px against {one:.4} for one — \
         accumulation is not running on the multisampled path"
    );
}

/// `render_accumulated` must give the same image as driving the loop by hand.
///
/// It skips the copy back into the frame target on every sample but the last,
/// which is most of what makes it cheap — and would be a silent corruption if
/// the mean did not actually live in the history buffer between samples.
#[test]
fn the_one_call_form_matches_the_manual_loop() {
    let Some(mut a) = HeadlessRenderer::builder()
        .size(W, H)
        .taa(true)
        .color_format(threers::wgpu::TextureFormat::Rgba8Unorm)
        .build()
        .ok()
    else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    a.set_msaa(4);
    let (mut sc, cam) = (scene(), camera());
    a.render_accumulated(&mut sc, &cam, 16);
    let one_call = a.read_rgba();

    let mut b = HeadlessRenderer::builder()
        .size(W, H)
        .taa(true)
        .color_format(threers::wgpu::TextureFormat::Rgba8Unorm)
        .build()
        .unwrap();
    b.set_msaa(4);
    b.renderer().set_taa_converge(true);
    b.renderer().reset_taa();
    let (mut sc2, cam2) = (scene(), camera());
    // The same offsets, driven by hand. `render_accumulated` uses the
    // stratified set rather than Halton — it knows the sample count and the
    // streaming pass does not — so leaving this loop on the renderer's own
    // jitter would compare two different sample sets and call the difference a
    // bug. Holding the offsets equal is what isolates the mechanism.
    for &offset in threers::stratified_jitter(16).iter() {
        b.renderer().set_jitter_override(Some(offset));
        b.render(&mut sc2, &cam2);
    }
    b.renderer().set_jitter_override(None);
    let manual = b.read_rgba();

    assert_eq!(one_call.len(), manual.len());
    let worst = one_call
        .iter()
        .zip(&manual)
        .map(|(x, y)| (*x as i32 - *y as i32).abs())
        .max()
        .unwrap();
    assert!(
        worst <= 1,
        "render_accumulated differs from the manual loop by {worst} levels — \
         the two accumulation paths disagree on the same sixteen samples"
    );
}

/// And it must leave the renderer as it found it, so a caller can mix settled
/// frames with ordinary ones.
#[test]
fn accumulating_restores_the_renderer_state() {
    let Some(mut r) = HeadlessRenderer::builder()
        .size(W, H)
        .taa(true)
        .color_format(threers::wgpu::TextureFormat::Rgba8Unorm)
        .build()
        .ok()
    else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    assert!(!r.renderer().taa_converge());
    let (mut sc, cam) = (scene(), camera());
    r.render_accumulated(&mut sc, &cam, 8);
    assert!(
        !r.renderer().taa_converge(),
        "converge mode was left on after render_accumulated"
    );
}

/// The accumulated mean must be a mean of *light*, and must land within a level
/// of computing it on the CPU.
///
/// Two things can go wrong once the averaging moves downstream of the
/// supersample resolve, and both are quiet. The resolve decodes sRGB before it
/// averages its four subpixels, so a sum taken over the encoded bytes it writes
/// is a mean over a curve — on this black-to-white edge that reads 21/255 where
/// linear light gives 0. And the per-sample weights have to survive being
/// recorded in one encoder, which they do not if they share a uniform buffer,
/// because `write_buffer` is ordered against submits and not against passes:
/// a 16-sample mean came out as `(15 + 1/16)/16`, a 6% darkening that reads as
/// a gamma slip rather than a bug. The reference below is arithmetic nobody can
/// argue with, so it catches both.
#[test]
fn the_accumulated_mean_is_a_mean_of_light() {
    const SS: u32 = 2;
    // Box, because the reference below resolves by averaging blocks and the two
    // have to be the same filter for the comparison to mean anything.
    let filt = threers::renderer::downsample::DownsampleFilter::Box;
    let Ok(mut r) = HeadlessRenderer::builder()
        .size(W, H)
        .supersample(SS)
        .taa(true)
        .downsample_filter(filt)
        .color_format(threers::wgpu::TextureFormat::Rgba8Unorm)
        .build()
    else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let (mut sc, cam) = (scene(), camera());
    r.render_accumulated(&mut sc, &cam, 16);
    let gpu = r.read_rgba_resolved();

    // The same 16 samples, averaged in f32 linear light and resolved on the CPU.
    let (iw, ih) = (W * SS, H * SS);
    let mut cpu = HeadlessRenderer::builder()
        .size(W, H)
        .supersample(SS)
        .downsample_filter(filt)
        .color_format(threers::wgpu::TextureFormat::Rgba8Unorm)
        .build()
        .unwrap();
    let mut acc = vec![0f32; (iw * ih * 4) as usize];
    for &offset in threers::stratified_jitter(16).iter() {
        cpu.renderer().set_jitter_override(Some(offset));
        cpu.render(&mut sc, &cam);
        let raw = cpu.read_rgba();
        assert_eq!(raw.len(), acc.len(), "read_rgba is not the internal buffer");
        for (k, (a, b)) in acc.iter_mut().zip(raw).enumerate() {
            let v = b as f32 / 255.0;
            *a += if k % 4 == 3 { v } else { to_linear(v) };
        }
    }
    // Resolve the SS×SS blocks, still in linear light, then encode once.
    let (ow, oh) = (W, H);
    let mut reference = vec![0u8; (ow * oh * 4) as usize];
    for y in 0..oh {
        for x in 0..ow {
            for c in 0..4 {
                let mut sum = 0f32;
                for dy in 0..SS {
                    for dx in 0..SS {
                        let px = ((y * SS + dy) * iw + x * SS + dx) * 4 + c;
                        sum += acc[px as usize];
                    }
                }
                let mean = (sum / (16 * SS * SS) as f32).clamp(0.0, 1.0);
                let out = if c == 3 { mean } else { to_srgb(mean) };
                reference[((y * ow + x) * 4 + c) as usize] = (out * 255.0 + 0.5) as u8;
            }
        }
    }

    let worst = gpu
        .iter()
        .zip(&reference)
        .map(|(x, y)| (*x as i32 - *y as i32).abs())
        .max()
        .unwrap();
    assert!(
        worst <= 2,
        "the accumulated frame is {worst} levels off a CPU mean of the same \
         samples in linear light"
    );
}

fn to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn to_srgb(v: f32) -> f32 {
    if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// `n` identical samples must be bit-for-bit one render.
///
/// The sharpest statement available about the accumulator: with the jitter held
/// still there is no sampling difference left to hide behind, so the sum, the
/// weights and the round trip through the transfer curve either compose exactly
/// or they do not.
#[test]
fn unjittered_samples_reproduce_one_render() {
    let build = || {
        HeadlessRenderer::builder()
            .size(W, H)
            .supersample(2)
            .taa(true)
            .color_format(threers::wgpu::TextureFormat::Rgba8Unorm)
            .build()
    };
    let Ok(mut many) = build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let (mut sc, cam) = (scene(), camera());
    many.render_accumulated_unjittered(&mut sc, &cam, 16);
    let many = many.read_rgba_resolved();

    let mut one = build().unwrap();
    one.renderer().set_jitter_override(Some((0.0, 0.0)));
    one.render(&mut sc, &cam);
    let one = one.read_rgba_resolved();

    let worst = many
        .iter()
        .zip(&one)
        .map(|(x, y)| (*x as i32 - *y as i32).abs())
        .max()
        .unwrap();
    assert_eq!(worst, 0, "16 identical samples drifted {worst} levels");
}

/// Reading a settled frame twice must give the same frame twice.
///
/// The resolve pass is skipped when the accumulator has already written its
/// mean into the resolve target, and getting that wrong is silent: the second
/// read would come back as the last jittered sample rather than the mean, which
/// looks like an ordinary frame and not like a bug.
#[test]
fn a_settled_frame_survives_being_read_twice() {
    let Ok(mut r) = HeadlessRenderer::builder()
        .size(W, H)
        .supersample(2)
        .taa(true)
        .color_format(threers::wgpu::TextureFormat::Rgba8Unorm)
        .build()
    else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let (mut sc, cam) = (scene(), camera());
    r.render_accumulated(&mut sc, &cam, 16);
    let first = r.read_rgba_resolved();
    let second = r.read_rgba_resolved();
    assert_eq!(first, second, "the second read is not the settled frame");
}
