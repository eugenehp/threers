//! Edge interpolation: does a straight edge move smoothly, and does it behave
//! the same way along both axes?
//!
//! This is the test for the class of artefact that shows up as stair-stepping,
//! banding, or a rail that crawls — and it is deliberately built so that the
//! answer is arithmetic rather than opinion. The scene is a white quad on black
//! under an orthographic camera scaled so **one world unit is one pixel**, drawn
//! with an unlit material so nothing here depends on lighting, IBL, or a
//! transfer curve beyond the framebuffer's own.
//!
//! Three questions, each a different failure:
//!
//! * **Does the edge track the geometry?** Slide the quad by a sixteenth of a
//!   pixel at a time and measure where the edge lands. Plotted against where it
//!   was asked to be, that should be a straight line of slope 1. A renderer
//!   with no antialiasing gives a staircase — the edge snaps from one pixel to
//!   the next — and the residual from the fitted line is 1/√12 ≈ 0.289 px, the
//!   rms of a uniform quantiser, which is exactly what it measures. That
//!   residual is the number this file is about.
//!
//! * **Is it the same along both axes?** The vertical and horizontal edges of
//!   the same quad, measured the same way. Rasterisation is symmetric, so any
//!   difference between them is a bug in something downstream: a separable
//!   filter applying the wrong pass to one axis, a jitter sequence with
//!   different statistics in x and y, a row-versus-column addressing slip.
//!   This is the check that earns its keep: the first run of it placed
//!   horizontal edges fifteen times less precisely than vertical ones — 0.0253
//!   px against 0.0017 — because the jitter drew x from Halton base 2 and y
//!   from base 3, and at sixteen samples only one of those lands on the exact
//!   sixteenths. Nothing about the picture says which axis to distrust.
//!
//! * **Is the edge where it should be?** The mean of the measured positions
//!   against the mean of the requested ones. A renderer that antialiases
//!   beautifully but puts the result an eighth of a pixel from where the camera
//!   says is still wrong, and nothing else here would notice.
//!
//! Every measurement decodes to linear light first. A gradient centroid reads
//! as coverage only when the values are linear in coverage, and sRGB bytes are
//! not; measuring on the encoded bytes inflates these numbers several-fold and
//! reports the transfer function as a renderer defect.
//!
//! Each backend is behind its own feature and reports itself skipped rather
//! than failing when the machine cannot run it — a missing GPU is not a broken
//! renderer.

use threers::cameras::{OrthographicCamera, PerspectiveCamera};
use threers::core::{Mesh, Object3D};
use threers::geometries::PlaneGeometry;
use threers::materials::BasicMaterial;
use threers::math::{Color, Vector3};
use threers::scene::Scene;

/// Frame size. Small on purpose: the path tracer runs this too.
const N: u32 = 64;
/// Quad side, in pixels. Leaves 16 px of background either side.
const QUAD: f32 = 32.0;
/// Sub-pixel offsets swept, over one whole pixel.
const STEPS: usize = 16;

/// A white quad on black, offset by `(dx, dy)` pixels.
fn scene(dx: f32, dy: f32) -> Scene {
    let mut s = Scene::new();
    s.background = Color::from_hex(0x000000);
    let mut q = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(QUAD, QUAD),
        BasicMaterial::new(Color::WHITE).into(),
    ));
    q.position = Vector3::new(dx, dy, 0.0);
    s.add(q);
    s
}

/// One world unit per pixel, so an offset in units is an offset in pixels.
fn camera() -> OrthographicCamera {
    let h = N as f32 / 2.0;
    let mut c = OrthographicCamera::new(-h, h, h, -h, 0.1, 100.0);
    c.position = Vector3::new(0.0, 0.0, 10.0);
    c
}

/// The same framing through a perspective camera.
///
/// Worth testing separately and not an academic difference: the two projections
/// put their numbers in different matrix elements, so a backend can get one
/// right and the other wrong. The accumulation jitter did exactly that — a
/// constant offset under perspective and a depth-dependent shear under
/// orthographic — and it took a test with both to see it.
///
/// The distance is chosen so the plane at `z = 0` still spans `N` pixels: with
/// a vertical field of view `f` at distance `d`, the visible height is
/// `2·d·tan(f/2)`, so `d = N / (2·tan(f/2))`.
fn perspective_camera() -> PerspectiveCamera {
    let fov: f32 = 30.0;
    let d = N as f32 / (2.0 * (fov.to_radians() / 2.0).tan());
    let mut c = PerspectiveCamera::new(fov, 1.0, 0.1, 1000.0);
    c.position = Vector3::new(0.0, 0.0, d);
    c.look_at(Vector3::ZERO);
    c
}

fn to_linear(b: u8) -> f64 {
    let v = b as f64 / 255.0;
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Which edge of the quad is being measured.
#[derive(Clone, Copy, PartialEq)]
enum Axis {
    /// The quad's left edge: a **vertical** edge, scanned along a row, and the
    /// one that responds to motion in x.
    Vertical,
    /// The quad's top edge: a **horizontal** edge, scanned down a column.
    Horizontal,
}

impl Axis {
    fn name(self) -> &'static str {
        match self {
            Axis::Vertical => "vertical",
            Axis::Horizontal => "horizontal",
        }
    }
    /// The offset that moves this edge outward by `d` pixels.
    fn offset(self, d: f32) -> (f32, f32) {
        match self {
            Axis::Vertical => (d, 0.0),
            // Screen y runs down, so the top edge moves out for negative y.
            Axis::Horizontal => (0.0, -d),
        }
    }
}

/// Sub-pixel position of the leading edge, and how wide the transition is.
///
/// The position is the centroid of the absolute gradient, which for a step
/// blurred by any symmetric filter sits at the step. The width is the sum of
/// that gradient over its largest single value: a perfect step measures 1.0,
/// and anything wider is the filter.
fn edge(px: &[u8], axis: Axis, at: u32) -> (f64, f64) {
    let sample = |i: u32| {
        let idx = match axis {
            Axis::Vertical => at * N + i,
            Axis::Horizontal => i * N + at,
        };
        to_linear(px[idx as usize * 4])
    };
    // Only the leading half, so the quad's far edge cannot pull the centroid.
    let grad: Vec<f64> = (1..N / 2).map(|i| (sample(i) - sample(i - 1)).abs()).collect();
    let total: f64 = grad.iter().sum();
    let peak = grad.iter().cloned().fold(0.0, f64::max);
    assert!(total > 0.05, "no edge found on the {} axis", axis.name());
    let pos: f64 = grad
        .iter()
        .enumerate()
        .map(|(k, g)| g * (k as f64 + 0.5))
        .sum::<f64>()
        / total;
    (pos, total / peak)
}

fn fit(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let (mx, my) = (xs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
    let sxy: f64 = xs.iter().zip(ys).map(|(a, b)| (a - mx) * (b - my)).sum();
    let sxx: f64 = xs.iter().map(|a| (a - mx) * (a - mx)).sum();
    (sxy / sxx, my - sxy / sxx * mx)
}

/// What one axis of one renderer measured.
struct Track {
    /// How far the edge moves per pixel the geometry moves. Wants 1.0.
    slope: f64,
    /// Rms departure from moving smoothly. Wants 0; a hard-edged rasteriser
    /// gives 1/√12 = 0.289.
    residual: f64,
    /// Mean transition width. A perfect step is 1.0.
    width: f64,
    /// Where the fitted line says the edge sits at zero offset.
    ///
    /// Its absolute value carries a pixel-centre convention and means little on
    /// its own. The difference between the two axes does not: the quad is
    /// centred and square, so the two edges are geometric mirrors and their
    /// intercepts have to agree. A gap is the rendered frame being shifted by
    /// a different sub-pixel amount in x than in y.
    intercept: f64,
}

/// Sweep one axis and report. `render` draws the scene at a given offset.
fn track(render: &mut dyn FnMut(f32, f32) -> Vec<u8>, axis: Axis) -> Track {
    let (mut want, mut got, mut widths) = (Vec::new(), Vec::new(), Vec::new());
    for k in 0..STEPS {
        let d = k as f32 / STEPS as f32;
        let (dx, dy) = axis.offset(d);
        let px = render(dx, dy);
        assert_eq!(px.len(), (N * N * 4) as usize, "frame is not {N}×{N}");
        let (pos, width) = edge(&px, axis, N / 2);
        want.push(d as f64);
        // Both leading edges move toward larger coordinates as `d` grows: the
        // quad's left edge rightward, its top edge downward.
        got.push(pos);
        widths.push(width);
    }
    let (slope, inter) = fit(&want, &got);
    let n = want.len() as f64;
    let residual = (want
        .iter()
        .zip(&got)
        .map(|(x, y)| {
            let e = y - (slope * x + inter);
            e * e
        })
        .sum::<f64>()
        / n)
        .sqrt();
    Track {
        slope,
        residual,
        width: widths.iter().sum::<f64>() / n,
        intercept: inter,
    }
}

/// What a configuration is allowed to measure.
///
/// Per-mode rather than one set of numbers for all of them: a bare rasteriser
/// *cannot* beat the quantiser, and holding it to the standard of sixteen
/// accumulated samples would only mean the cheap path never gets tested.
struct Budget {
    /// Rms departure from moving smoothly, in pixels.
    residual: f64,
    /// Transition width, where 1.0 is a perfect step.
    width: f64,
    /// How far the slope may sit from 1.0.
    ///
    /// Generous where the edge does not interpolate at all: a staircase has no
    /// slope worth fitting, and what is being asked there is only that the edge
    /// moves the right way. The residual is the real check in that case.
    slope: f64,
}

/// Run both axes and hold both to the same standard.
fn both_axes(label: &str, render: &mut dyn FnMut(f32, f32) -> Vec<u8>, budget: Budget) {
    let v = track(render, Axis::Vertical);
    let h = track(render, Axis::Horizontal);
    eprintln!(
        "{label:28} vertical slope {:.3} residual {:.4} width {:.3} | \
         horizontal slope {:.3} residual {:.4} width {:.3}",
        v.slope, v.residual, v.width, h.slope, h.residual, h.width
    );

    for (axis, t) in [(Axis::Vertical, &v), (Axis::Horizontal, &h)] {
        assert!(
            t.slope > 0.0,
            "{label}: the {} edge moves the wrong way (slope {:.3}) — the axis \
             is flipped",
            axis.name(),
            t.slope
        );
        assert!(
            (t.slope - 1.0).abs() <= budget.slope,
            "{label}: the {} edge moves {:.3} px per pixel of geometry",
            axis.name(),
            t.slope
        );
        assert!(
            t.residual <= budget.residual,
            "{label}: the {} edge does not move smoothly — {:.4} px rms from \
             the line it should follow, against a {:.4} budget",
            axis.name(),
            t.residual,
            budget.residual
        );
        assert!(
            t.width <= budget.width,
            "{label}: the {} edge is {:.3} px wide, against a {:.3} budget \
             (1.0 is a perfect step)",
            axis.name(),
            t.width,
            budget.width
        );
    }

    // The symmetry check. Rasterisation does not care which axis it is on, so
    // a gap here is something downstream of it.
    let gap = (v.residual - h.residual).abs();
    let tol = 0.25 * v.residual.max(h.residual) + 0.002;
    assert!(
        gap <= tol,
        "{label}: vertical and horizontal edges interpolate differently — \
         {:.4} px rms against {:.4}. Rasterisation is symmetric, so this is a \
         filter, a jitter sequence, or an addressing slip that is not.",
        v.residual,
        h.residual
    );
    let wgap = (v.width - h.width).abs();
    assert!(
        wgap <= 0.08,
        "{label}: vertical edges are {:.3} px wide and horizontal ones {:.3} — \
         one axis is being filtered more than the other",
        v.width,
        h.width
    );
    // The quad is centred and square, so its two edges are mirror images and
    // must sit the same distance from the frame's middle.
    let shift = (v.intercept - h.intercept).abs();
    assert!(
        shift <= 0.02,
        "{label}: the frame is shifted {shift:.4} px further in one axis than \
         the other (intercepts {:.4} and {:.4}) — something is translating the \
         image by a different sub-pixel amount in x than in y",
        v.intercept,
        h.intercept
    );
}

// ---------------------------------------------------------------- wgpu ------

/// The wgpu renderer, in each of the ways it can be asked to antialias.
#[test]
fn wgpu_edges_interpolate() {
    use threers::HeadlessRenderer;
    let modes: [(&str, u32, u32, u32, Budget); 5] = [
        ("wgpu plain", 1, 1, 1, Budget { residual: 0.30, width: 1.05, slope: 0.60 }),
        ("wgpu msaa 4", 1, 4, 1, Budget { residual: 0.14, width: 1.70, slope: 0.35 }),
        ("wgpu supersample 2", 2, 1, 1, Budget { residual: 0.16, width: 1.70, slope: 0.35 }),
        ("wgpu supersample 4", 4, 1, 1, Budget { residual: 0.09, width: 2.20, slope: 0.15 }),
        ("wgpu accumulate 16", 1, 1, 16, Budget { residual: 0.005, width: 1.70, slope: 0.02 }),
    ];
    for (label, ss, msaa, acc, budget) in modes {
        let Ok(mut r) = HeadlessRenderer::builder()
            .size(N, N)
            .supersample(ss)
            .taa(acc > 1)
            .color_format(threers::wgpu::TextureFormat::Rgba8Unorm)
            .build()
        else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        r.set_msaa(msaa);
        let mut render = |dx: f32, dy: f32| {
            let mut sc = scene(dx, dy);
            let cam = camera();
            if acc > 1 {
                r.render_accumulated(&mut sc, &cam, acc);
                if ss > 1 {
                    r.read_rgba_resolved()
                } else {
                    r.read_rgba()
                }
            } else if ss > 1 {
                r.render_to_rgba_resolved(&mut sc, &cam)
            } else {
                r.render_to_rgba(&mut sc, &cam)
            }
        };
        both_axes(label, &mut render, budget);
    }
    // And the same through a perspective camera, which is what anything
    // photographic actually uses.
    let Ok(mut r) = HeadlessRenderer::builder()
        .size(N, N)
        .supersample(2)
        .taa(true)
        .downsample_filter(threers::renderer::downsample::DownsampleFilter::Box)
        .build()
    else {
        return;
    };
    r.set_msaa(4);
    let mut render = |dx: f32, dy: f32| {
        let mut sc = scene(dx, dy);
        r.render_accumulated(&mut sc, &perspective_camera(), 16);
        r.read_rgba_resolved()
    };
    both_axes(
        "wgpu perspective, full",
        &mut render,
        Budget { residual: 0.02, width: 1.70, slope: 0.05 },
    );
}

// --------------------------------------------------------------- metal ------

/// The Metal backend, which has to answer the same questions as wgpu.
///
/// A flipped axis shows up here as a negative slope rather than as a picture
/// someone has to notice, which is the point of measuring rather than looking.
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[test]
fn metal_edges_interpolate() {
    use threers::metal::{MetalError, MetalHeadlessRenderer};
    for (label, msaa, ss, budget) in [
        ("metal plain", 1u32, 1u32, Budget { residual: 0.30, width: 1.05, slope: 0.60 }),
        ("metal msaa 4", 4, 1, Budget { residual: 0.13, width: 1.70, slope: 0.35 }),
        ("metal supersample 2", 1, 2, Budget { residual: 0.16, width: 1.70, slope: 0.35 }),
    ] {
        let mut r = match MetalHeadlessRenderer::builder()
            .size(N, N)
            .msaa(msaa)
            .supersample(ss)
            .build()
        {
            Ok(r) => r,
            Err(MetalError::NoDevice) => {
                eprintln!("skipping: no Metal device on this machine");
                return;
            }
            Err(e) => panic!("metal renderer: {e:?}"),
        };
        let mut render = |dx: f32, dy: f32| {
            r.render_to_rgba(&mut scene(dx, dy), &camera())
                .expect("metal render")
        };
        both_axes(label, &mut render, budget);
    }
    // Perspective, which is where the app's Metal renders go wrong: the model
    // comes out upside down there while an orthographic scene does not.
    let mut r = match MetalHeadlessRenderer::builder().size(N, N).msaa(4).build() {
        Ok(r) => r,
        Err(MetalError::NoDevice) => return,
        Err(e) => panic!("metal renderer: {e:?}"),
    };
    let mut render = |dx: f32, dy: f32| {
        r.render_to_rgba(&mut scene(dx, dy), &perspective_camera())
            .expect("metal render")
    };
    both_axes(
        "metal perspective, msaa 4",
        &mut render,
        Budget { residual: 0.13, width: 1.70, slope: 0.35 },
    );
}

// ------------------------------------------------------------ path tracer ---

/// The path tracer. Its antialiasing is the pixel filter over its own samples,
/// so this is a different mechanism reaching for the same answer.
///
/// The denoiser is off — it is a spatial filter and would be answering the
/// question for the tracer — and so is the tone curve, which is monotone but
/// not linear and would move a gradient centroid.
#[cfg(feature = "raytrace")]
#[test]
fn raytrace_edges_interpolate() {
    use threers::raytrace::{RaytraceRenderer, RaytraceSettings};
    let mut r = RaytraceRenderer::new(N, N);
    r.set_settings(RaytraceSettings {
        samples_per_pixel: 64,
        max_bounces: 1,
        denoise: false,
        tone_mapping: threers::ToneMapping::None,
        ..Default::default()
    });
    let mut render = |dx: f32, dy: f32| r.render_to_rgba(&mut scene(dx, dy), &camera());
    // The budget is as tight as the accumulating raster path's, and it is met:
    // a stratified pixel sampler resolving a straight edge converges on the
    // analytic coverage, so 64 samples land on it exactly — at half a pixel of
    // offset the edge pixel reads 0.500 linear and not 0.497. Both renderers
    // agreeing here to four decimals is the two of them being right rather than
    // a coincidence, and what is left in the residual is the eight-bit output.
    both_axes(
        "raytrace 64 spp",
        &mut render,
        Budget { residual: 0.01, width: 1.70, slope: 0.02 },
    );
}

/// A pixel half covered by white must carry half the light — on every colour
/// format, and whichever mechanism did the antialiasing.
///
/// The one question the rest of this file does not ask. Slope, residual and
/// width are all *relative*: they say the edge moves smoothly and lands in the
/// right place, and every one of them is blind to the value in the pixel being
/// wrong by a factor of two. Two bugs hid in exactly that blind spot, both in
/// the transfer curve, both found only by asking what 50% coverage reads as:
///
/// * `Rgba8UnormSrgb` — which is the **default** — double-encoded, because the
///   shader applied the sRGB curve and then the hardware applied it again. Half
///   coverage stored 0.7379 where the light was 0.5. Every frame this renderer
///   produced was washed out, and no test noticed because none of them looked
///   at a number rather than a difference between numbers.
///
/// * `Rgba8Unorm` with MSAA resolved in the wrong space. The resolve is
///   hardware and it averages whatever the bytes hold; on a plain `Unorm`
///   target holding *manually* encoded sRGB, that is a mean over a curve. Two
///   covered samples and two not gave an encoded mean of 0.5, which is 0.216 of
///   the light rather than 0.5 — an edge at 43% of its proper brightness.
#[test]
fn a_half_covered_pixel_carries_half_the_light() {
    use threers::wgpu::TextureFormat::{Rgba16Float, Rgba8Unorm, Rgba8UnormSrgb};
    use threers::HeadlessRenderer;
    // `Rgba8Unorm` is left out of the MSAA row on purpose: the hardware resolve
    // over hand-encoded bytes cannot be right, and the fix for anyone who needs
    // multisampling is to use a format the hardware understands.
    let cases: [(&str, threers::wgpu::TextureFormat, u32, u32, u32); 7] = [
        ("srgb + supersample 4", Rgba8UnormSrgb, 4, 1, 1),
        ("unorm + supersample 4", Rgba8Unorm, 4, 1, 1),
        ("float + supersample 4", Rgba16Float, 4, 1, 1),
        ("srgb + msaa 4", Rgba8UnormSrgb, 1, 4, 1),
        ("float + msaa 4", Rgba16Float, 1, 4, 1),
        ("srgb + accumulate 16", Rgba8UnormSrgb, 1, 1, 16),
        // All three at once, which is what an offline render actually asks for
        // and the combination the double encode was hiding in.
        ("srgb + ss2 + msaa4 + acc16", Rgba8UnormSrgb, 2, 4, 16),
    ];
    for (label, format, ss, msaa, acc) in cases {
        let Ok(mut r) = HeadlessRenderer::builder()
            .size(N, N)
            .supersample(ss)
            .taa(acc > 1)
            .downsample_filter(threers::renderer::downsample::DownsampleFilter::Box)
            .color_format(format)
            .build()
        else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        r.set_msaa(msaa);
        // Half a pixel of offset puts exactly half of one column under the quad.
        let mut sc = scene(0.5, 0.0);
        let cam = camera();
        let px = if acc > 1 {
            r.render_accumulated(&mut sc, &cam, acc);
            if ss > 1 {
                r.read_rgba_resolved()
            } else {
                r.read_rgba()
            }
        } else if ss > 1 {
            r.render_to_rgba_resolved(&mut sc, &cam)
        } else {
            r.render_to_rgba(&mut sc, &cam)
        };
        assert_eq!(px.len(), (N * N * 4) as usize, "{label}: frame is not {N}×{N}");
        let edge = to_linear(px[((N / 2 * N + 16) * 4) as usize]);
        let inside = to_linear(px[((N / 2 * N + 24) * 4) as usize]);
        eprintln!("{label:24} half-covered {edge:.4} (want 0.5)  interior {inside:.4} (want 1.0)");
        assert!(
            (inside - 1.0).abs() < 0.01,
            "{label}: the quad's interior reads {inside:.4}, so the transfer \
             curve is wrong before coverage even comes into it"
        );
        assert!(
            (edge - 0.5).abs() < 0.02,
            "{label}: a half-covered pixel reads {edge:.4} of the light rather \
             than 0.5 — something in this path is averaging over a curve"
        );
    }
}

// ------------------------------------------------------------ the jitter ----

/// The jitter set itself, without a GPU: `n` samples must cover the pixel
/// evenly on both axes and sit centred on it.
///
/// This is the property the accumulating path depends on, and it is cheaper to
/// state here than to infer from a rendered edge. Halton fails both halves of
/// it at any particular `n` — at sixteen samples its base-3 axis has gaps of
/// 0.037 where 0.0625 is even, and its mean sits 0.037 px off centre, which is
/// a whole frame translated by that much.
#[test]
fn the_jitter_covers_the_pixel_evenly_on_both_axes() {
    for n in [4u32, 8, 16, 17, 32, 64] {
        let j = threers::stratified_jitter(n);
        assert_eq!(j.len() as u32, n);
        for (axis, vals) in [
            ("x", j.iter().map(|o| o.0).collect::<Vec<_>>()),
            ("y", j.iter().map(|o| o.1).collect::<Vec<_>>()),
        ] {
            let mean: f32 = vals.iter().sum::<f32>() / n as f32;
            assert!(
                mean.abs() < 1e-6,
                "n={n}: the {axis} offsets average {mean:+.5} px, so every \
                 accumulated frame is translated by that much"
            );
            let mut sorted = vals.clone();
            sorted.sort_by(f32::total_cmp);
            assert!(
                sorted[0] >= -0.5 && *sorted.last().unwrap() < 0.5,
                "n={n}: the {axis} offsets leave the pixel"
            );
            for w in sorted.windows(2) {
                let gap = w[1] - w[0];
                assert!(
                    (gap - 1.0 / n as f32).abs() < 1e-6,
                    "n={n}: the {axis} offsets are {gap:.5} apart where they \
                     should be {:.5} — one axis will converge faster than the \
                     other",
                    1.0 / n as f32
                );
            }
        }
        // Both axes stratified is not enough on its own: the pairing has to be
        // a permutation, or the samples fall on a diagonal and the pixel's
        // corners never get sampled.
        let mut seen: Vec<i32> = j.iter().map(|o| (o.1 * n as f32 * 2.0).round() as i32).collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len() as u32, n, "n={n}: the y offsets repeat");
    }
}
