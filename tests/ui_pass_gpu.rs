//! The GPU UI pass against the CPU overlay painting the same scene: the
//! overlay is the reference, and the pass has to land on its pixels.
#![cfg(not(target_arch = "wasm32"))]

use threers::overlay::Overlay;
use threers::renderer::readback::texture_bytes;
use threers::renderer::UiPass;
use threers::style;
use threers::ui::{OverlayPainter, Ui};

const W: u32 = 240;
const H: u32 = 236;

/// Backgrounds, borders, rounded boxes, text, an icon, a gradient and a
/// clipped box: one of everything a scene holds.
fn build() -> Ui {
    let mut ui = Ui::new();
    let root = ui.root();
    ui.set_style(root, style! { display: flex; flex-direction: column; background: #1e2127; padding: 6px; gap: 6px; });
    let bar = ui.add_box(
        root,
        style! { display: flex; gap: 6px; align-items: center; height: 28px; },
    );
    let button = ui.add_box(
        bar,
        style! { padding: 4px 10px; background: #3b6fd8; border-radius: 6px; color: white; font-size: 13px; },
    );
    ui.add_text(button, "Render");
    let outlined = ui.add_box(
        bar,
        style! { padding: 3px 8px; border: 1px solid #c8ccd4; border-radius: 4px; background: #262a31; color: #c8ccd4; font-size: 12px; },
    );
    ui.add_text(outlined, "Cancel");
    ui.add_path(
        bar,
        style! { width: 16px; height: 16px; color: #e5c07b; },
        vec!["M12 2 L22 21 L2 21 Z".into()],
        [0.0, 0.0, 24.0, 24.0],
        None,
        true,
    );
    let label = ui.add_box(root, style! { color: #abb2bf; font-size: 14px; });
    ui.add_text(label, "Hello, scene — 1 draw call.");
    ui.add_box(root, style! { height: 14px; background: #ff0000; });
    // A box clipping a wider child: rectangles cut at the clip's edge.
    let clip = ui.add_box(
        root,
        style! { width: 120.5px; height: 20px; overflow: hidden; background: #333842; },
    );
    ui.add_box(
        clip,
        style! { width: 300px; height: 9.5px; background: #98c379; margin: 3.25px 0 0 7.5px; },
    );
    // An image, scaled up 2.5x (bilinear, so its inner edges blend) and
    // one with transparency, in a row.
    let row = ui.add_box(root, style! { display: flex; gap: 4px; height: 20px; });
    let checker: Vec<u8> = (0..8 * 8)
        .flat_map(|i| {
            if (i % 8 + i / 8) % 2 == 0 {
                [230, 80, 40, 255]
            } else {
                [40, 160, 230, 255]
            }
        })
        .collect();
    let checker = threers::ui::image::Image::new(8, 8, checker).unwrap();
    ui.add_image(row, style! { width: 20px; height: 20px; }, checker);
    let glass: Vec<u8> = (0..4 * 4)
        .flat_map(|i| [255, 255, 255, (i * 16) as u8])
        .collect();
    let glass = threers::ui::image::Image::new(4, 4, glass).unwrap();
    ui.add_image(row, style! { width: 16px; height: 16px; }, glass);
    // Shadows, blurred and sharp, and text with styled runs: a highlight,
    // a straight underline and a wavy one.
    let cards = ui.add_box(
        root,
        style! { display: flex; gap: 14px; height: 30px; padding: 0 8px; },
    );
    ui.add_box(cards, style! { width: 60px; height: 22px; background: #2c313a; border-radius: 6px; box-shadow: 0 3px 8px rgba(0,0,0,0.8); });
    ui.add_box(
        cards,
        style! { width: 30px; height: 22px; background: #2c313a; box-shadow: 2px 2px 0 #e06c75; },
    );
    let prose = ui.add_box(cards, style! { color: #abb2bf; font-size: 13px; });
    let words = ui.add_text(prose, "a typo here");
    ui.set_text_runs(
        words,
        vec![
            threers::ui::rich_text::TextRun::new(2).background([0.3, 0.3, 0.1, 1.0]),
            threers::ui::rich_text::TextRun::new(4)
                .underline(true)
                .decoration_color([0.9, 0.3, 0.3, 1.0]),
            threers::ui::rich_text::TextRun::new(1),
            threers::ui::rich_text::TextRun::new(4)
                .underline(false)
                .color([0.6, 0.8, 1.0, 1.0]),
        ],
    );
    // A chart: an area filled under a line, and the line stroked.
    let chart = ui.add_box(root, style! { height: 40px; display: flex; });
    ui.add_path(
        chart,
        style! { width: 160px; height: 40px; color: #61afef; },
        vec!["M0 40 L0 30 L20 12 L45 25 L70 5 L100 18 L130 8 L160 22 L160 40 Z".into()],
        [0.0, 0.0, 160.0, 40.0],
        None,
        true,
    );
    ui.add_path(
        chart,
        style! { width: 60px; height: 40px; color: #e5c07b; },
        vec!["M0 30 Q15 0 30 20 T60 10".into()],
        [0.0, 0.0, 60.0, 40.0],
        Some(2.0),
        false,
    );
    let mut fade = ui.style(root).clone();
    fade.height = threers::ui::Length::Px(12.0);
    fade.padding = threers::ui::Edges::all(0.0);
    fade.background = None;
    fade.background_gradient = vec![(0.0, [0.9, 0.2, 0.3, 1.0]), (1.0, [0.2, 0.4, 0.9, 1.0])];
    ui.add_box(root, fade);
    ui
}

/// The overlay's pixels for `ui` at `scale`.
fn cpu(ui: &Ui, scale: f32) -> Vec<u8> {
    let (w, h) = ((W as f32 * scale) as u32, (H as f32 * scale) as u32);
    let mut overlay = Overlay::new(w, h);
    let mut painter = OverlayPainter::with_scale(&mut overlay, scale);
    ui.paint(&mut painter);
    overlay.pixels().to_vec()
}

struct Diff {
    max: u8,
    /// Pixels differing by more than 2 in some channel.
    off: usize,
    total: usize,
}

fn compare(a: &[u8], b: &[u8]) -> Diff {
    let mut d = Diff {
        max: 0,
        off: 0,
        total: a.len() / 4,
    };
    for (p, q) in a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0) {
        let m = (0..4).map(|i| p[i].abs_diff(q[i])).max().unwrap();
        d.max = d.max.max(m);
        if m > 2 {
            d.off += 1;
        }
    }
    d
}

fn run(scale: f32) {
    run_with(scale, None);
}

fn run_with(scale: f32, gpu_path_area: Option<f32>) {
    let Some(gpu) = threers::gpu::testing::context("the UI pass") else {
        return;
    };
    let (device, queue) = (&gpu.device, &gpu.queue);
    let mut pass = UiPass::new(device);
    if let Some(a) = gpu_path_area {
        pass.gpu_path_area = a;
    }
    let mut ui = build();
    ui.layout(W as f32, H as f32, &mut pass.text_measure(scale));

    let (w, h) = ((W as f32 * scale) as u32, (H as f32 * scale) as u32);
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ui pass test target"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    // Cleared to transparent, as the overlay starts.
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: None,
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    queue.submit(Some(encoder.finish()));

    let scene = ui.scene();
    pass.draw(device, queue, &scene, scale, &view, (w, h), format);
    let first = pass.stats();
    let gpu_pixels = texture_bytes(device, queue, &target);
    let cpu_pixels = cpu(&ui, scale);
    let d = compare(&cpu_pixels, &gpu_pixels);
    eprintln!(
        "scale {scale}: {} of {} pixels off by more than 2, worst {}; {:?}",
        d.off, d.total, d.max, first
    );
    // Edges are computed by the same rule in f32 on both sides, so all but
    // a sliver of anti-aliased edge pixels agree to within rounding.
    assert!(
        d.off * 1000 <= d.total,
        "{} of {} pixels differ",
        d.off,
        d.total
    );
    assert!(d.max <= 4, "a pixel differs by {}", d.max);
    assert_eq!(first.dropped, 0);
    match gpu_path_area {
        Some(_) => assert!(
            first.gpu_paths >= 3,
            "the icon, the area and the line from edges: {first:?}"
        ),
        None => assert!(
            first.rasterized >= 6,
            "text, the icon and two images go to atlases: {first:?}"
        ),
    }

    // The next frame finds every sprite in the atlas.
    pass.draw(device, queue, &scene, scale, &view, (w, h), format);
    assert_eq!(pass.stats().rasterized, 0);
    assert_eq!(pass.stats().instances, first.instances);
}

#[test]
fn the_gpu_pass_paints_what_the_overlay_paints() {
    run(1.0);
}

#[test]
fn and_on_a_dense_display() {
    run(2.0);
}

/// WebGL2 runs the shader as GLSL ES 3.0, through naga's writer and then
/// ANGLE, which refuses names naga allows (`gl_`-prefixed ones).
#[test]
fn the_shader_translates_for_webgl2() {
    translates_for_webgl2(
        UiPass::wgsl(),
        &["vs_main"],
        &["fs_main", "fs_surface", "fs_backdrop"],
    );
    let [blit, blur] = UiPass::helper_wgsl();
    translates_for_webgl2(blit, &["vs_blit"], &["fs_blit"]);
    translates_for_webgl2(blur, &["vs_blur"], &["fs_blur"]);
}

/// A custom shader's module too, written inline through the macro (so with
/// Rust's spacing).
#[test]
fn a_custom_shader_translates_for_webgl2() {
    let shader = threers::ui::ui_shader! {
        fn wobble(p: vec2<f32>, color: vec4<f32>, time: f32, size: vec2<f32>, amount: f32, tint: vec4<f32>) -> vec4<f32> {
            let q = p + vec2<f32>(sin(p.y * 0.1 + time) * amount, 0.0);
            return mix(layer(q), tint * color.a, 0.25);
        }
    };
    assert_eq!(shader.error(), None);
    translates_for_webgl2(
        &UiPass::effect_wgsl(&shader),
        &["threers_vs"],
        &["threers_fs"],
    );
}

fn translates_for_webgl2(source: &str, vertex: &[&str], fragment: &[&str]) {
    use wgpu::naga;
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("validates");
    let stages = vertex
        .iter()
        .map(|e| (naga::ShaderStage::Vertex, *e))
        .chain(fragment.iter().map(|e| (naga::ShaderStage::Fragment, *e)));
    for (stage, entry) in stages {
        let options = naga::back::glsl::Options {
            version: naga::back::glsl::Version::Embedded {
                version: 300,
                is_webgl: true,
            },
            ..Default::default()
        };
        let pipeline = naga::back::glsl::PipelineOptions {
            shader_stage: stage,
            entry_point: entry.to_string(),
            multiview: None,
        };
        let mut out = String::new();
        naga::back::glsl::Writer::new(
            &mut out,
            &module,
            &info,
            &options,
            &pipeline,
            Default::default(),
        )
        .unwrap_or_else(|e| panic!("{entry}: {e}"))
        .write()
        .unwrap_or_else(|e| panic!("{entry}: {e}"));
        let bad = out
            .match_indices("gl_")
            .filter(|(i, _)| {
                *i == 0
                    || !(out.as_bytes()[i - 1].is_ascii_alphanumeric()
                        || out.as_bytes()[i - 1] == b'_')
            })
            .filter(|(i, _)| {
                !out.as_bytes()
                    .get(i + 3)
                    .is_some_and(|c| c.is_ascii_uppercase())
            })
            .count();
        assert_eq!(bad, 0, "{entry}: GLSL names ANGLE rejects");
    }
}

/// `render_to_rgba` is the GPU's picture of a UI, `paint_to_rgba` the CPU's:
/// the same scene, one pixel apart at most.
#[test]
fn rendering_a_ui_to_an_image_matches_the_cpu() {
    let Some(gpu) = threers::gpu::testing::context("the UI pass") else {
        return;
    };
    let (device, queue) = (&gpu.device, &gpu.queue);
    let mut pass = UiPass::new(device);
    let mut ui = build();
    ui.layout(W as f32, H as f32, &mut pass.text_measure(1.0));
    let gpu_image = pass.render_to_rgba(device, queue, &ui.scene(), 1.0, (W, H), [0.0; 4]);
    let cpu_image = threers::ui::paint_to_rgba(&ui, 1.0, W, H);
    let d = compare(&cpu_image, &gpu_image);
    assert!(
        d.off <= 2 && d.max <= 4,
        "{} pixels, worst {}",
        d.off,
        d.max
    );
    assert_eq!(pass.stats().draws, 1);
}

/// A surface is drawn between what is painted before and after it.
#[test]
fn a_surface_shows_its_texture_in_paint_order() {
    let Some(gpu) = threers::gpu::testing::context("the UI pass") else {
        return;
    };
    let (device, queue) = (&gpu.device, &gpu.queue);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("surface"),
        size: wgpu::Extent3d {
            width: 2,
            height: 2,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        texture.as_image_copy(),
        &[255, 220, 0, 255].repeat(4),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(8),
            rows_per_image: Some(2),
        },
        wgpu::Extent3d {
            width: 2,
            height: 2,
            depth_or_array_layers: 1,
        },
    );
    let mut pass = UiPass::new(device);
    pass.set_surface(
        7,
        texture.create_view(&wgpu::TextureViewDescriptor::default()),
    );
    let mut ui = Ui::new();
    let root = ui.root();
    ui.set_style(root, style! { background: #000; });
    ui.add_box(root, style! { position: absolute; left: 0; top: 0; width: 60px; height: 60px; background: #00f; });
    ui.add_surface(
        root,
        style! { position: absolute; left: 10px; top: 10px; width: 40px; height: 40px; },
        7,
        (2, 2),
    );
    ui.add_box(root, style! { position: absolute; left: 30px; top: 30px; width: 30px; height: 30px; background: #0f0; });
    ui.layout(64.0, 64.0, &mut pass.text_measure(1.0));
    let image = pass.render_to_rgba(
        device,
        queue,
        &ui.scene(),
        1.0,
        (64, 64),
        [0.0, 0.0, 0.0, 1.0],
    );
    let px = |x: usize, y: usize| image[(y * 64 + x) * 4..][..3].to_vec();
    assert_eq!(px(5, 5), [0, 0, 255], "under the surface");
    assert_eq!(px(20, 20), [255, 220, 0], "the surface");
    assert_eq!(px(40, 40), [0, 255, 0], "over the surface");
    assert_eq!(pass.stats().draws, 3);
    // Without its texture, it is not drawn.
    pass.remove_surface(7);
    let image = pass.render_to_rgba(
        device,
        queue,
        &ui.scene(),
        1.0,
        (64, 64),
        [0.0, 0.0, 0.0, 1.0],
    );
    assert_eq!(image[(20 * 64 + 20) * 4..][..3].to_vec(), [0, 0, 255]);
}

/// Paths drawn from their edges by the shader, not rasterised as sprites:
/// the same fill and stroke rules, so the same pixels.
#[test]
fn paths_drawn_on_the_gpu_match_the_overlay() {
    run_with(1.0, Some(0.0));
    run_with(2.0, Some(0.0));
}

/// A box turned in 3D is drawn in true perspective on the GPU: its nearer
/// edge taller than its far one (the keystone the CPU painters cannot
/// draw), exactly where its homography takes its corners.
#[test]
fn a_turned_box_is_drawn_in_perspective() {
    use threers::ui::depth::{apply_homography, Transform3d};
    let Some(gpu) = threers::gpu::testing::context("the UI pass") else {
        return;
    };
    let (device, queue) = (&gpu.device, &gpu.queue);
    let mut pass = UiPass::new(device);
    let (w, h) = (300u32, 200u32);
    let mut ui = Ui::new();
    let root = ui.root();
    let card = ui.add_box(root, style! { position: absolute; left: 50px; top: 40px; width: 200px; height: 120px; background: #ffffff; });
    ui.layout(w as f32, h as f32, &mut pass.text_measure(1.0));
    let t = Transform3d {
        rotate_y: 0.9,
        perspective: 300.0,
        ..Transform3d::default()
    };
    ui.set_transform3d(card, t);
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("perspective target"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: None,
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    queue.submit(Some(encoder.finish()));
    pass.draw(device, queue, &ui.scene(), 1.0, &view, (w, h), format);
    let px = texture_bytes(device, queue, &target);
    let covered = |x: u32, y: u32| px[((y * w + x) * 4 + 3) as usize] > 128;
    // The white column's height, at x.
    let height = |x: u32| (0..h).filter(|&y| covered(x, y)).count() as f32;
    let r = ui.rect(card);
    let hm = t.homography(r);
    let (lx, _) = apply_homography(hm, r.x, r.y + r.height / 2.0);
    let (rx, _) = apply_homography(hm, r.x + r.width, r.y + r.height / 2.0);
    let (near, far) = (height(lx.ceil() as u32 + 2), height(rx.floor() as u32 - 2));
    eprintln!("left edge {lx}, right {rx}; heights {near} {far}");
    assert!(
        near > far + 15.0,
        "the nearer (left) edge taller: {near} vs {far}"
    );
    // Each edge where the homography puts it, to a pixel.
    let top_left = apply_homography(hm, r.x, r.y);
    let expect = apply_homography(hm, r.x, r.y + r.height).1 - top_left.1;
    assert!(
        (height(lx.ceil() as u32 + 2) - expect).abs() <= 3.0,
        "{near} vs {expect}"
    );
    assert!(!covered(((lx + rx) / 2.0) as u32, 2), "nothing above it");
}

/// A box drawn and put back through `effect`, at `scale`, over black, as
/// RGB rows; and the pass, for its stats.
fn with_effect(
    gpu: &threers::gpu::GpuContext,
    effect: Option<threers::ui::shader::Effect>,
    scale: f32,
) -> (Vec<u8>, UiPass) {
    let (device, queue) = (&gpu.device, &gpu.queue);
    let mut pass = UiPass::new(device);
    let mut ui = Ui::new();
    let root = ui.root();
    ui.set_style(root, style! { background: #000; });
    // A box half red and half green, inside a box that clips its last 10px.
    let clip = ui.add_box(root, style! { position: absolute; left: 10px; top: 10px; width: 70px; height: 40px; overflow: hidden; });
    let card = ui.add_box(clip, style! { display: flex; width: 80px; height: 40px; });
    ui.add_box(
        card,
        style! { width: 40px; height: 40px; background: #f00; },
    );
    ui.add_box(
        card,
        style! { width: 40px; height: 40px; background: #0f0; },
    );
    // And one beside it with no effect.
    ui.add_box(root, style! { position: absolute; left: 10px; top: 60px; width: 40px; height: 10px; background: #f00; });
    ui.set_effect(card, effect);
    let (w, h) = ((100.0 * scale) as u32, (80.0 * scale) as u32);
    ui.layout(100.0, 80.0, &mut pass.text_measure(scale));
    let image = pass.render_to_rgba(
        device,
        queue,
        &ui.scene(),
        scale,
        (w, h),
        [0.0, 0.0, 0.0, 1.0],
    );
    let rgb = image.chunks(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
    (rgb, pass)
}

/// The colour at UI point `(x, y)`.
fn at(rgb: &[u8], scale: f32, x: f32, y: f32) -> [u8; 3] {
    let w = (100.0 * scale) as usize;
    let i = ((y * scale) as usize * w + (x * scale) as usize) * 3;
    [rgb[i], rgb[i + 1], rgb[i + 2]]
}

/// A colour effect: each pixel of the box and its insides remade from the
/// colour there, the clip around it kept, the box beside it untouched.
#[test]
fn a_custom_shader_recolours_its_box() {
    use threers::ui::shader::Effect;
    let Some(gpu) = threers::gpu::testing::context("the UI pass") else {
        return;
    };
    let swap = threers::ui::ui_shader! {
        fn swap(p: vec2<f32>, color: vec4<f32>) -> vec4<f32> {
            return vec4<f32>(color.b, color.r, color.g, color.a);
        }
    };
    for scale in [1.0, 2.0] {
        let (rgb, pass) = with_effect(&gpu, Some(Effect::new(&swap)), scale);
        assert_eq!(
            at(&rgb, scale, 20.0, 30.0),
            [0, 255, 0],
            "red to green, at {scale}x"
        );
        assert_eq!(at(&rgb, scale, 60.0, 30.0), [0, 0, 255], "green to blue");
        assert_eq!(at(&rgb, scale, 85.0, 30.0), [0, 0, 0], "clipped as it was");
        assert_eq!(
            at(&rgb, scale, 20.0, 65.0),
            [255, 0, 0],
            "the box beside it as it is"
        );
        assert_eq!(
            at(&rgb, scale, 5.0, 5.0),
            [0, 0, 0],
            "nothing outside the box"
        );
        let stats = pass.stats();
        assert_eq!((stats.effects, stats.failed_effects), (1, 0));
    }
}

/// A distortion: the layer read elsewhere (here, mirrored), with an
/// argument from the effect.
#[test]
fn a_custom_shader_reads_the_layer_anywhere() {
    use threers::ui::shader::Effect;
    let Some(gpu) = threers::gpu::testing::context("the UI pass") else {
        return;
    };
    let mirror = threers::ui::ui_shader! {
        fn mirror(p: vec2<f32>, size: vec2<f32>, strength: f32) -> vec4<f32> {
            return layer(vec2<f32>(size.x - p.x, p.y)) * strength;
        }
    };
    let effect = Effect::new(&mirror).arg("strength", 1.0);
    let (rgb, _) = with_effect(&gpu, Some(effect), 1.0);
    assert_eq!(
        at(&rgb, 1.0, 20.0, 30.0),
        [0, 255, 0],
        "the right half on the left"
    );
    assert_eq!(
        at(&rgb, 1.0, 60.0, 30.0),
        [255, 0, 0],
        "and the left on the right"
    );
    let (rgb, _) = with_effect(&gpu, Some(Effect::new(&mirror).arg("strength", 0.0)), 1.0);
    assert_eq!(
        at(&rgb, 1.0, 20.0, 30.0),
        [0, 0, 0],
        "an argument at zero: nothing"
    );
}

/// A shader the GPU cannot compile: said so, and the box drawn as it is.
#[test]
fn a_broken_shader_draws_the_box_as_it_is() {
    use threers::ui::shader::{Effect, Shader};
    let Some(gpu) = threers::gpu::testing::context("the UI pass") else {
        return;
    };
    let broken = Shader::new("fn broken(p: vec2<f32>) -> vec4<f32> { return no_such_thing(p); }");
    assert_eq!(broken.error(), None, "its shape is fine");
    let (rgb, pass) = with_effect(&gpu, Some(Effect::new(&broken)), 1.0);
    assert_eq!(at(&rgb, 1.0, 20.0, 30.0), [255, 0, 0]);
    assert_eq!(at(&rgb, 1.0, 85.0, 30.0), [0, 0, 0], "still clipped");
    assert_eq!(pass.stats().failed_effects, 1);
    let error = pass.effect_error(&broken).expect("the compiler's error");
    assert!(error.contains("no_such_thing"), "{error}");
}

/// A gradient across a box turned in 3D and one flat, three stops: each
/// pixel steps from the one beside it by no more than the colours' change
/// across a pixel (no bands of flat colour), and none inside is seen
/// through (no seam where two stretches meet).
#[test]
fn a_gradient_is_smooth_and_seamless() {
    let Some(gpu) = threers::gpu::testing::context("the UI pass") else {
        return;
    };
    let (device, queue) = (&gpu.device, &gpu.queue);
    let mut pass = UiPass::new(device);
    let mut ui = Ui::new();
    let root = ui.root();
    ui.set_style(
        root,
        style! { display: flex; flex-direction: column; gap: 10px; padding: 10px; },
    );
    let mut boxes = vec![];
    for turned in [false, true] {
        let mut st = style! { width: 200.5px; height: 30px; };
        st.background_gradient = vec![
            (0.0, [0.9, 0.2, 0.3, 1.0]),
            (0.37, [0.2, 0.8, 0.4, 1.0]),
            (1.0, [0.2, 0.3, 0.9, 1.0]),
        ];
        let b = ui.add_box(root, st);
        if turned {
            ui.set_transform3d(
                b,
                threers::ui::depth::Transform3d {
                    rotate_y: 0.5,
                    ..Default::default()
                },
            );
        }
        boxes.push(b);
    }
    ui.layout(240.0, 120.0, &mut pass.text_measure(1.0));
    let (w, h) = (240u32, 120u32);
    let image = pass.render_to_rgba(device, queue, &ui.scene(), 1.0, (w, h), [0.0; 4]);
    let px = |x: u32, y: u32| &image[((y * w + x) * 4) as usize..][..4];
    for &b in &boxes {
        let r = ui.rect(b);
        let y = (r.y + r.height / 2.0) as u32;
        // The row through its middle, where it covers whole pixels.
        let xs: Vec<u32> = (0..w).filter(|&x| px(x, y)[3] == 255).collect();
        assert!(xs.len() > 120, "{} covered", xs.len());
        let (lo, hi) = (xs[2], xs[xs.len() - 3]);
        assert_eq!(
            (lo..=hi).filter(|&x| px(x, y)[3] != 255).count(),
            0,
            "a seam"
        );
        let worst = (lo..hi)
            .map(|x| {
                (0..3)
                    .map(|c| px(x, y)[c].abs_diff(px(x + 1, y)[c]))
                    .max()
                    .unwrap()
            })
            .max()
            .unwrap();
        // 255 levels over ~74 px at the steepest stretch: ~4 a pixel.
        assert!(worst <= 6, "a step of {worst} between neighbours");
    }
}

/// Painting again only what changed comes out as painting it all: a
/// colour changed, an effect's time moved on, a box turned in 3D, and a
/// frame where nothing changed.
#[test]
fn painting_only_what_changed_matches_painting_it_all() {
    use threers::ui::scene::SceneDamage;
    let Some(gpu) = threers::gpu::testing::context("the UI pass") else {
        return;
    };
    let (device, queue) = (&gpu.device, &gpu.queue);
    let (w, h) = (240u32, 160u32);
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let clear = [0.12, 0.13, 0.15, 1.0];
    let ripple = threers::ui::shader::Shader::new("fn ripple(p: vec2<f32>, time: f32) -> vec4<f32> { return layer(p + vec2<f32>(sin(p.y * 0.3 + time * 5.0) * 3.0, 0.0)); }");
    let frame = |colour: f32, turn: f32| {
        let mut ui = Ui::new();
        let root = ui.root();
        ui.set_style(
            root,
            style! { display: flex; flex-wrap: wrap; gap: 8px; padding: 8px; },
        );
        let a = ui.add_box(
            root,
            style! { width: 60px; height: 40px; border-radius: 6px; },
        );
        ui.update_style(a, |s| s.background = Some([colour, 0.4, 0.6, 1.0]));
        let fx = ui.add_box(root, style! { width: 80px; height: 50px; background: #3a7bd5; color: #ffffff; font-size: 14px; });
        ui.add_text(fx, "ripple");
        ui.set_effect(fx, Some(threers::ui::shader::Effect::new(&ripple)));
        let turned = ui.add_box(
            root,
            style! { width: 70px; height: 40px; background: #e06c75; },
        );
        ui.set_transform3d(
            turned,
            threers::ui::depth::Transform3d {
                rotate_y: turn,
                ..Default::default()
            },
        );
        let t = ui.add_box(root, style! { color: #ffffff; font-size: 13px; });
        ui.add_text(t, "stays put");
        ui
    };
    let mut retained = UiPass::new(device);
    let mut whole = UiPass::new(device);
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    let mut damages = vec![];
    for (colour, time, turn) in [
        (0.2, 0.0, 0.3),
        (0.8, 0.0, 0.3),
        (0.8, 0.5, 0.3),
        (0.8, 0.5, 0.3),
        (0.8, 0.5, -0.4),
    ] {
        let mut ui = frame(colour, turn);
        // The effect's clock.
        ui.step(time);
        ui.layout(w as f32, h as f32, &mut retained.text_measure(1.0));
        let scene = ui.scene();
        damages.push(retained.draw_retained(
            device,
            queue,
            &scene,
            1.0,
            &view,
            (w, h),
            format,
            clear,
        ));
        let got = texture_bytes(device, queue, &target);
        let want = whole.render_to_rgba(device, queue, &scene, 1.0, (w, h), clear);
        let d = compare(&want, &got);
        assert!(
            d.off == 0,
            "frame {}: {} pixels differ, worst {}; {:?}",
            damages.len(),
            d.off,
            d.max,
            damages.last()
        );
    }
    assert_eq!(damages[0], SceneDamage::Full);
    assert!(
        matches!(damages[2], SceneDamage::Regions(_)),
        "the effect moved on: {:?}",
        damages[2]
    );
    assert_eq!(damages[3], SceneDamage::None, "nothing changed");
    // A colour change repaints that box, not the window.
    let SceneDamage::Regions(r) = &damages[1] else {
        panic!("{:?}", damages[1])
    };
    let area: f32 = r.iter().map(|q| q.width * q.height).sum();
    assert!(area < 0.2 * (w * h) as f32, "{r:?}");
}

/// A material shows what is under it blurred: stripes behind it come out
/// close to their average, beside it stay sharp; drawn again only where
/// something under it changed, it still matches a whole drawing.
#[test]
fn a_material_blurs_what_is_behind_it() {
    let Some(gpu) = threers::gpu::testing::context("the UI pass") else {
        return;
    };
    let (device, queue) = (&gpu.device, &gpu.queue);
    let (w, h) = (200u32, 120u32);
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let clear = [1.0, 1.0, 1.0, 1.0];
    let frame = |dot: f32| {
        let mut ui = Ui::new();
        let root = ui.root();
        ui.set_style(
            root,
            style! { position: relative; width: 200px; height: 120px; },
        );
        let stripes = ui.add_box(root, style! { display: flex; position: absolute; left: 0px; top: 0px; width: 200px; height: 120px; });
        for i in 0..50 {
            let s = ui.add_box(stripes, style! { width: 4px; height: 120px; });
            ui.update_style(s, |st| {
                st.background = Some(if i % 2 == 0 {
                    [0.0, 0.0, 0.0, 1.0]
                } else {
                    [1.0, 1.0, 1.0, 1.0]
                })
            });
        }
        let mark = ui.add_box(root, style! { position: absolute; top: 50px; width: 12px; height: 12px; background: #ff0000; });
        ui.update_style(mark, |st| st.left = threers::ui::Length::Px(dot));
        let glass = ui.add_box(root, style! { position: absolute; left: 40px; top: 30px; width: 100px; height: 60px; border-radius: 10px; backdrop-filter: blur(6px); });
        let _ = glass;
        ui
    };
    let lum = |px: &[u8], x: u32, y: u32| {
        let i = ((y * w + x) * 4) as usize;
        px[i] as f32 * 0.3 + px[i + 1] as f32 * 0.59 + px[i + 2] as f32 * 0.11
    };
    let spread = |px: &[u8], x0: u32, x1: u32, y: u32| {
        let v: Vec<f32> = (x0..x1).map(|x| lum(px, x, y)).collect();
        let m = v.iter().sum::<f32>() / v.len() as f32;
        (v.iter().map(|a| (a - m) * (a - m)).sum::<f32>() / v.len() as f32).sqrt()
    };
    let mut whole = UiPass::new(device);
    let mut ui = frame(170.0);
    ui.layout(w as f32, h as f32, &mut whole.text_measure(1.0));
    let image = whole.render_to_rgba(device, queue, &ui.scene(), 1.0, (w, h), clear);
    // Under the glass, the stripes all but gone; beside it, sharp.
    let (inside, outside) = (spread(&image, 60, 120, 60), spread(&image, 0, 30, 60));
    assert!(
        inside < 20.0 && outside > 100.0,
        "spread under the glass {inside}, beside it {outside}"
    );
    // The red mark moved under the glass: drawn again there, it matches
    // drawing it all.
    let mut retained = UiPass::new(device);
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    for dot in [170.0, 80.0] {
        let mut ui = frame(dot);
        ui.layout(w as f32, h as f32, &mut retained.text_measure(1.0));
        let scene = ui.scene();
        retained.draw_retained(device, queue, &scene, 1.0, &view, (w, h), format, clear);
        let got = texture_bytes(device, queue, &target);
        let want = whole.render_to_rgba(device, queue, &scene, 1.0, (w, h), clear);
        let d = compare(&want, &got);
        assert!(
            d.off == 0,
            "mark at {dot}: {} pixels differ, worst {}",
            d.off,
            d.max
        );
    }
}

/// Radial, conic and turned linear gradients on the GPU land where the CPU
/// puts them (within the ramp's 256 steps).
#[test]
fn gradients_of_other_shapes_match_the_cpu() {
    let Some(gpu) = threers::gpu::testing::context("the UI pass") else {
        return;
    };
    let (device, queue) = (&gpu.device, &gpu.queue);
    let mut pass = UiPass::new(device);
    let mut ui = Ui::new();
    let root = ui.root();
    ui.set_style(
        root,
        style! { display: flex; gap: 10px; padding: 10px; background: #202020; },
    );
    for g in [
        "radial-gradient(circle at 30% 40%, #ff0000, #ffff00 50%, #0000ff)",
        "conic-gradient(from 45deg, #ff0000, #00ff00, #0000ff, #ff0000)",
        "linear-gradient(135deg, #ffffff, #000000)",
    ] {
        let b = ui.add_box(root, style! { width: 60px; height: 50px; });
        ui.update_style(b, |s| threers::ui::css::apply(s, "background", g));
        assert!(!ui.style(b).background_gradient.is_empty(), "{g}");
    }
    let (w, h) = (220u32, 70u32);
    ui.layout(w as f32, h as f32, &mut pass.text_measure(1.0));
    let gpu_image = pass.render_to_rgba(device, queue, &ui.scene(), 1.0, (w, h), [0.0; 4]);
    let cpu_image = threers::ui::paint_to_rgba(&ui, 1.0, w, h);
    let mut worst = 0u8;
    let mut off = 0;
    for (p, q) in cpu_image.chunks(4).zip(gpu_image.chunks(4)) {
        let m = (0..4).map(|i| p[i].abs_diff(q[i])).max().unwrap();
        worst = worst.max(m);
        off += usize::from(m > 6);
    }
    assert!(
        off <= 4,
        "{off} pixels differ by more than 6, worst {worst}"
    );
    // And they are gradients: each box varies.
    for x0 in [10u32, 80, 150] {
        let row: Vec<u8> = (x0..x0 + 60)
            .map(|x| gpu_image[((35 * w + x) * 4) as usize])
            .collect();
        assert!(
            row.iter().max().unwrap() - row.iter().min().unwrap() > 60,
            "flat at {x0}: {row:?}"
        );
    }
}

/// `mask-image` fades a box out along its gradient; `mix-blend-mode`
/// multiplies and screens what is behind.
#[test]
fn masks_and_blend_modes() {
    let Some(gpu) = threers::gpu::testing::context("the UI pass") else {
        return;
    };
    let (device, queue) = (&gpu.device, &gpu.queue);
    let mut pass = UiPass::new(device);
    let mut ui = Ui::new();
    let root = ui.root();
    // Grey: half the light (sRGB 188).
    ui.set_style(
        root,
        style! { display: flex; gap: 10px; padding: 10px; background: #bcbcbc; },
    );
    let faded = ui.add_box(
        root,
        style! { width: 40px; height: 60px; background: #ffffff; },
    );
    ui.update_style(faded, |s| {
        threers::ui::css::apply(
            s,
            "mask-image",
            "linear-gradient(to bottom, #000000, transparent)",
        )
    });
    assert!(ui.style(faded).mask.is_some());
    let multiply = ui.add_box(
        root,
        style! { width: 40px; height: 60px; background: #ff0000; mix-blend-mode: multiply; },
    );
    let screen = ui.add_box(
        root,
        style! { width: 40px; height: 60px; background: #ff0000; mix-blend-mode: screen; },
    );
    let (w, h) = (160u32, 80u32);
    ui.layout(w as f32, h as f32, &mut pass.text_measure(1.0));
    let img = pass.render_to_rgba(device, queue, &ui.scene(), 1.0, (w, h), [0.0; 4]);
    let at = |n: threers::ui::NodeId, fy: f32| {
        let r = ui.rect(n);
        let (x, y) = ((r.x + r.width / 2.0) as u32, (r.y + r.height * fy) as u32);
        let i = ((y * w + x) * 4) as usize;
        [img[i], img[i + 1], img[i + 2]]
    };
    let near = |a: [u8; 3], b: [u8; 3]| (0..3).all(|i| a[i].abs_diff(b[i]) <= 4);
    assert!(
        near(at(faded, 0.05), [253, 253, 253]) || at(faded, 0.05)[0] > 245,
        "top shows: {:?}",
        at(faded, 0.05)
    );
    assert!(
        at(faded, 0.95)[0] < 196,
        "bottom faded to the grey: {:?}",
        at(faded, 0.95)
    );
    assert!(at(faded, 0.5)[0] > at(faded, 0.95)[0] && at(faded, 0.5)[0] < at(faded, 0.05)[0]);
    // Red × grey: red at half light, no green or blue.
    assert!(
        near(at(multiply, 0.5), [188, 0, 0]),
        "multiply: {:?}",
        at(multiply, 0.5)
    );
    // Red screened on grey: full red, half green and blue.
    assert!(
        near(at(screen, 0.5), [255, 188, 188]),
        "screen: {:?}",
        at(screen, 0.5)
    );
}

/// Gauges' arcs from their distance field: the GPU and the CPU painter
/// agree, and each pixel along the stroke's edge is covered as the exact
/// distance says (no corners where a flattened curve would have them).
#[test]
fn arcs_are_smooth_and_match_the_cpu() {
    let Some(gpu) = threers::gpu::testing::context("the UI pass") else {
        return;
    };
    let (device, queue) = (&gpu.device, &gpu.queue);
    let mut pass = UiPass::new(device);
    let mut ui = Ui::new();
    let root = ui.root();
    ui.set_style(
        root,
        style! { display: flex; gap: 10px; padding: 10px; background: #000000; },
    );
    let mut gauges = vec![];
    for kind in [
        threers::ui::charts::GaugeKind::Circular,
        threers::ui::charts::GaugeKind::Ring,
    ] {
        let st = threers::ui::charts::GaugeStyle {
            size: 120.0,
            animate: false,
            tint: threers::ui::charts::GaugeTint::Color([1.0, 1.0, 1.0, 1.0]),
            show_range: false,
            ..Default::default()
        };
        let mut g = threers::ui::charts::gauge_styled(&mut ui, root, "", 0.0..=100.0, kind, st);
        g.set(&mut ui, 63.0);
        gauges.push(g);
    }
    let (w, h) = (270u32, 170u32);
    ui.layout(w as f32, h as f32, &mut pass.text_measure(1.0));
    let scene = ui.scene();
    let arcs: Vec<(threers::ui::ArcShape, f32)> = scene
        .items
        .iter()
        .filter_map(|i| match i {
            threers::ui::scene::Item::Arc { arc, width, color } if color[0] > 0.99 => {
                Some((*arc, *width))
            }
            _ => None,
        })
        .collect();
    assert_eq!(arcs.len(), 2, "both fills drawn as arcs");
    let gpu_image = pass.render_to_rgba(device, queue, &scene, 1.0, (w, h), [0.0; 4]);
    let cpu_image = threers::ui::paint_to_rgba(&ui, 1.0, w, h);
    let mut off = 0;
    for (p, q) in cpu_image.chunks(4).zip(gpu_image.chunks(4)) {
        off += usize::from((0..3).any(|i| p[i].abs_diff(q[i]) > 8));
    }
    assert!(off <= 6, "{off} pixels differ");
    // Along the white fills' edges, coverage as the distance says (white
    // over black, in sRGB: the pixel's light is the coverage).
    let mut checked = 0;
    for (arc, width) in &arcs {
        let b = arc.bounds(width / 2.0 + 1.0);
        for y in b.y as u32..(b.y + b.height) as u32 {
            for x in b.x as u32..(b.x + b.width) as u32 {
                let d = arc.distance(*width, x as f32 + 0.5, y as f32 + 0.5);
                // Only where no track (grey) is under it: well inside the ring's band is all fill or track.
                if d.abs() > 0.45 || x >= w || y >= h {
                    continue;
                }
                let want = (0.5 - d).clamp(0.0, 1.0);
                let i = ((y * w + x) * 4) as usize;
                let lin = |v: u8| {
                    let c = v as f32 / 255.0;
                    if c <= 0.04045 {
                        c / 12.92
                    } else {
                        ((c + 0.055) / 1.055).powf(2.4)
                    }
                };
                // The track's grey shows through the uncovered part on the
                // ring's sides; at the round ends it is black or track.
                let got = lin(gpu_image[i]);
                assert!(got + 0.12 >= want, "({x}, {y}): {got} against {want}");
                checked += 1;
            }
        }
    }
    assert!(checked > 200, "{checked} edge pixels");
}

/// The realistic candle flame, as the GPU draws it: written out to look at
/// with `FLAME_PNG=path`.
#[test]
fn the_candle_flame_is_drawn() {
    let Some(gpu) = threers::gpu::testing::context("the UI pass") else {
        return;
    };
    let (device, queue) = (&gpu.device, &gpu.queue);
    let mut pass = UiPass::new(device);
    let mut ui = Ui::new();
    let root = ui.root();
    ui.set_style(
        root,
        style! { display: flex; padding: 10px; background: #101010; },
    );
    let c = threers::ui::candle::candle(
        &mut ui,
        root,
        threers::ui::candle::FlameStyle::Realistic,
        200.0,
        1.0,
    );
    c.update(&mut ui, 1.3);
    let (w, h) = (220u32, 420u32);
    ui.layout(w as f32, h as f32, &mut pass.text_measure(1.0));
    let image = pass.render_to_rgba(device, queue, &ui.scene(), 1.0, (w, h), [0.0; 4]);
    if let Some(path) = std::env::var_os("FLAME_PNG") {
        std::fs::write(path, threers::utils::png::encode_png(w, h, &image)).unwrap();
    }
}

/// A rounded box clipping inside a rounded box it overlaps: both cut, each
/// with its own corners (the GPU gives a shape a second clip for it). (The
/// CPU painter clips to boxes only, rounded or not.)
#[test]
fn overlapping_rounded_clips_both_cut() {
    let Some(gpu) = threers::gpu::testing::context("the UI pass") else {
        return;
    };
    let (device, queue) = (&gpu.device, &gpu.queue);
    let mut pass = UiPass::new(device);
    let mut ui = Ui::new();
    let root = ui.root();
    ui.set_style(root, style! { background: #000000; });
    let outer = ui.add_box(root, style! { position: absolute; left: 10px; top: 10px; width: 100px; height: 100px; border-radius: 24px; overflow: hidden; });
    let inner = ui.add_box(outer, style! { position: absolute; left: 50px; top: 50px; width: 100px; height: 100px; border-radius: 24px; overflow: hidden; });
    ui.add_box(
        inner,
        style! { width: 100px; height: 100px; background: #ffffff; },
    );
    let (w, h) = (140u32, 140u32);
    ui.layout(w as f32, h as f32, &mut pass.text_measure(1.0));
    let gpu_image = pass.render_to_rgba(device, queue, &ui.scene(), 1.0, (w, h), [0.0; 4]);
    let at = |img: &[u8], x: u32, y: u32| img[((y * w + x) * 4) as usize];
    // Filled where both leave it; cut at the outer box's corner (its
    // bottom right, inside the inner box) and at the inner's (its top left,
    // inside the outer).
    assert!(at(&gpu_image, 85, 85) > 250);
    assert!(
        at(&gpu_image, 107, 107) < 10,
        "the outer corner cuts: {}",
        at(&gpu_image, 107, 107)
    );
    assert!(
        at(&gpu_image, 62, 62) < 10,
        "the inner corner cuts: {}",
        at(&gpu_image, 62, 62)
    );
    // And the edge between: the inner box's own side, half way down it,
    // filled to the outer box's edge.
    assert!(at(&gpu_image, 100, 85) > 250 && at(&gpu_image, 112, 85) < 10);
}
