//! Marks on a real browser GPU: WebGL2 when built with `webgl`, WebGPU
//! otherwise.
//!
//! The marks claim to need nothing WebGL2 lacks: instanced vertex buffers,
//! render passes, a half-float target for densities and an integer one for
//! picking. Reading the spec cannot confirm that, and on WebGPU the WGSL goes
//! through Tint, which rejects things naga accepts. So every kind is drawn
//! here, on both, and read back.
//!
//! ```sh
//! FEATURES=webgl,captions TEST=marks_web scripts/wasm-browser-test.sh
//! WEBGPU=1 FEATURES=captions TEST=marks_web scripts/wasm-browser-test.sh
//! ```
#![cfg(target_arch = "wasm32")]

/// The backend this build tests.
const BACKEND: wgpu::Backends = if cfg!(feature = "webgl") {
    wgpu::Backends::GL
} else {
    wgpu::Backends::BROWSER_WEBGPU
};

use threers::marks::{
    DensityMarks, ImageMarks, ImageQuad, LineCap, Mark, MarkRenderer, MarkTarget, MarkView, Origin,
    PointMarks, PolygonFeature, PolygonMarks, SegmentMarks, ShaderHooks, ShapeMarks, SpriteMarks,
    TextureAtlas,
};
use threers::{BoxGeometry, OrthographicCamera, Vector3};
use wasm_bindgen::JsCast;
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_browser);

const W: u32 = 128;
const H: u32 = 64;

async fn yield_to_event_loop() {
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        web_sys::window()
            .unwrap()
            .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 0)
            .unwrap();
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

type SharedGl = (
    std::sync::Arc<wgpu::Device>,
    std::sync::Arc<wgpu::Queue>,
    wgpu::Surface<'static>,
    wgpu::Instance,
);

thread_local! {
    static SHARED_GL: std::cell::RefCell<Option<SharedGl>> =
        const { std::cell::RefCell::new(None) };
}

/// One canvas and one context for the whole suite: a page may hold only a few
/// live WebGL2 contexts, and they are not released when a canvas is dropped.
async fn shared_gl() -> Option<(std::sync::Arc<wgpu::Device>, std::sync::Arc<wgpu::Queue>)> {
    if let Some(t) = SHARED_GL.with(|g| {
        g.borrow()
            .as_ref()
            .map(|(d, q, _, _)| (d.clone(), q.clone()))
    }) {
        return Some(t);
    }
    let document = web_sys::window()?.document()?;
    let canvas: web_sys::HtmlCanvasElement =
        document.create_element("canvas").ok()?.dyn_into().ok()?;
    canvas.set_width(64);
    canvas.set_height(64);
    document.body()?.append_child(&canvas).ok()?;
    let mut d = wgpu::InstanceDescriptor::new_without_display_handle();
    d.backends = BACKEND;
    let instance = wgpu::Instance::new(d);
    let surface = instance
        .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
        .ok()?;
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })
        .await
        .ok()?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("threers marks browser test"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            ..Default::default()
        })
        .await
        .ok()?;
    let (device, queue) = (std::sync::Arc::new(device), std::sync::Arc::new(queue));
    SHARED_GL.with(|g| *g.borrow_mut() = Some((device.clone(), queue.clone(), surface, instance)));
    Some((device, queue))
}

fn texture(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("marks webgl target"),
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

/// Wait for a mapping without blocking the thread that delivers it.
async fn await_map(
    device: &wgpu::Device,
    rx: futures_channel::oneshot::Receiver<Result<(), wgpu::BufferAsyncError>>,
) {
    let mut rx = rx;
    for _ in 0..4000 {
        let _ = device.poll(wgpu::PollType::Poll);
        yield_to_event_loop().await;
        if let Ok(Some(r)) = rx.try_recv() {
            r.expect("map failed");
            return;
        }
    }
    panic!("the buffer never finished mapping");
}

/// Clear to black, draw `marks` over it with a depth buffer, read back RGBA8.
async fn draw(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mr: &mut MarkRenderer,
    marks: &mut [&mut dyn Mark],
) -> Vec<u8> {
    let color = texture(
        device,
        wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
    );
    let depth = texture(
        device,
        threers::marks::DEPTH_FORMAT,
        wgpu::TextureUsages::RENDER_ATTACHMENT,
    );
    let cv = color.create_view(&Default::default());
    let dv = depth.create_view(&Default::default());
    let mut enc = device.create_command_encoder(&Default::default());
    enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("clear"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &cv,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: &dv,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    queue.submit([enc.finish()]);

    let view = MarkView::new(&camera(), Origin::ZERO, (W, H));
    let target = MarkTarget::new(&cv, wgpu::TextureFormat::Rgba8Unorm).with_depth(&dv);
    mr.draw(device, queue, &target, &view, marks);

    let padded =
        (W * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("marks webgl readback"),
        size: (padded * H) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &color,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([enc.finish()]);
    let (tx, rx) = futures_channel::oneshot::channel();
    readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    await_map(device, rx).await;
    let data = readback.slice(..).get_mapped_range().expect("mapped");
    let mut out = Vec::with_capacity((W * H * 4) as usize);
    for row in 0..H {
        let at = (row * padded) as usize;
        out.extend_from_slice(&data[at..at + (W * 4) as usize]);
    }
    out
}

/// One world unit to a pixel, origin in the middle.
fn camera() -> OrthographicCamera {
    let mut c = OrthographicCamera::new(-64.0, 64.0, 32.0, -32.0, 0.1, 100.0);
    c.position = Vector3::new(0.0, 0.0, 10.0);
    c
}

fn px(img: &[u8], x: u32, y: u32) -> [u8; 4] {
    let o = ((y * W + x) * 4) as usize;
    [img[o], img[o + 1], img[o + 2], img[o + 3]]
}

fn is_bg(p: [u8; 4]) -> bool {
    p[0] < 8 && p[1] < 8 && p[2] < 8
}

#[wasm_bindgen_test]
async fn points_segments_and_shapes_draw_in_the_browser() {
    let Some((device, queue)) = shared_gl().await else {
        panic!("no {BACKEND:?} adapter");
    };
    let mut mr = MarkRenderer::new(&device);
    let mut points = PointMarks::new();
    points.set_positions(&[[-40.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    points.set_radii(&[6.0, 6.0]);
    points.set_colors(&[[255, 0, 0, 255], [0, 255, 0, 255]]);
    // A hook in the browser as well: hide the second point.
    points.common.set_custom(0, 1, &[1.0, 9.0]);
    points.common.hooks = ShaderHooks::range_filter(0, 0);
    points.style_mut().params[0] = [0.0, 2.0, 0.0, 0.0];
    let mut line = SegmentMarks::new();
    line.set_segments(&[[-60.0, 20.0, 0.0]], &[[60.0, 20.0, 0.0]]);
    line.set_widths(&[6.0]);
    line.set_default_color([0, 0, 255, 255]);
    let mut boxes = ShapeMarks::new(&BoxGeometry::new(1.0, 1.0, 1.0));
    boxes.set_positions(&[[40.0, 0.0, 0.0]]);
    boxes.set_scales(&[[12.0, 12.0, 12.0]]);
    boxes.ambient = 1.0;
    boxes.set_default_color([255, 255, 0, 255]);
    let img = draw(
        &device,
        &queue,
        &mut mr,
        &mut [&mut points, &mut line, &mut boxes],
    )
    .await;
    // A Unorm target: the marks encode, so the sRGB bytes come back as given.
    assert_eq!(px(&img, 24, 32), [255, 0, 0, 255], "red point");
    assert!(is_bg(px(&img, 64, 32)), "filtered point is gone");
    assert_eq!(px(&img, 64, 12)[2], 255, "the line");
    assert!(is_bg(px(&img, 64, 17)), "6 px wide, not more");
    assert_eq!(px(&img, 104, 32), [255, 255, 0, 255], "the box");
}

/// A segment thinner than a pixel covers its share of each pixel, here as on
/// a native adapter (`thin_segments_keep_their_weight_between_pixels` in
/// `marks_gpu.rs`): its footprint comes from screen derivatives, which GLSL
/// and Tint take on their own terms. A 0.4-pixel line puts 0.4 of its ink in
/// each column wherever it falls, along a row and on a 45° diagonal (√2 of
/// its width a column). The Unorm target blends the encoded values, and
/// white over black encodes to itself, so the bytes are the coverage.
#[wasm_bindgen_test]
async fn thin_segments_keep_their_weight_in_the_browser() {
    let Some((device, queue)) = shared_gl().await else {
        panic!("no {BACKEND:?} adapter");
    };
    let mut mr = MarkRenderer::new(&device);
    let width = 0.4f32;
    for diagonal in [false, true] {
        let mut sums = Vec::new();
        for k in 0..8 {
            let phase = k as f64 / 8.0;
            let mut line = SegmentMarks::new();
            if diagonal {
                line.set_segments(&[[-25.0 + phase, -25.0, 0.0]], &[[25.0 + phase, 25.0, 0.0]]);
            } else {
                line.set_segments(&[[-50.0, phase, 0.0]], &[[50.0, phase, 0.0]]);
            }
            line.set_widths(&[width]);
            line.set_default_color([255, 255, 255, 255]);
            let img = draw(&device, &queue, &mut mr, &mut [&mut line]).await;
            let expect = if diagonal { width * std::f32::consts::SQRT_2 } else { width };
            for x in [56u32, 64, 71] {
                let col: Vec<f32> = (0..H).map(|y| px(&img, x, y)[0] as f32 / 255.0).collect();
                let peak = col.iter().cloned().fold(0.0, f32::max);
                assert!(peak <= expect + 0.03, "{BACKEND:?} diagonal {diagonal} phase {phase} x {x}: peak {peak}");
                sums.push(col.iter().sum::<f32>() / expect);
            }
        }
        for s in &sums {
            assert!((s - 1.0).abs() < 0.06, "{BACKEND:?} diagonal {diagonal}: column ink {s} of the width, all {sums:?}");
        }
    }
}

#[wasm_bindgen_test]
async fn polygons_draw_per_vertex_in_the_browser() {
    let Some((device, queue)) = shared_gl().await else {
        panic!("no {BACKEND:?} adapter");
    };
    let mut mr = MarkRenderer::new(&device);
    // Two footprints raised from y = -20 to 10; the camera looks along -Z, so
    // what shows is each one's south wall.
    let rect = |x0: f64, x1: f64| vec![[x0, -5.0], [x1, -5.0], [x1, 5.0], [x0, 5.0]];
    let mut polygons = PolygonMarks::new();
    polygons.set_polygons(
        &[
            PolygonFeature::new(rect(-50.0, -30.0)),
            PolygonFeature::new(rect(30.0, 50.0)),
        ],
        -20.0,
    );
    polygons.set_elevations(&[30.0, 30.0]);
    polygons.set_colors(&[[255, 0, 255, 255], [0, 255, 255, 255]]);
    polygons.ambient = 1.0;
    // Per-feature custom values through the hook: hide the second.
    polygons.set_feature_custom(0, 1, &[1.0, 9.0]);
    polygons.common.hooks = ShaderHooks::range_filter(0, 0);
    polygons.style_mut().params[0] = [0.0, 2.0, 0.0, 0.0];
    let img = draw(&device, &queue, &mut mr, &mut [&mut polygons]).await;
    assert_eq!(px(&img, 24, 32), [255, 0, 255, 255], "first feature's wall");
    assert!(is_bg(px(&img, 24, 56)), "below its base, at y = -24");
    assert!(is_bg(px(&img, 104, 32)), "filtered feature is gone");
}

#[wasm_bindgen_test]
async fn a_trail_hook_cuts_segments_in_the_browser() {
    let Some((device, queue)) = shared_gl().await else {
        panic!("no {BACKEND:?} adapter");
    };
    let mut mr = MarkRenderer::new(&device);
    // Reached at t = 0 at x = -60 and t = 1 at x = 60; now 0.5, trail 0.5:
    // drawn from x = -60 to 0, fading towards the tail.
    let mut trip = SegmentMarks::new();
    trip.set_segments(&[[-60.0, 0.0, 0.0]], &[[60.0, 0.0, 0.0]]);
    trip.set_widths(&[6.0]);
    trip.set_default_color([255, 255, 255, 255]);
    trip.common.set_custom(0, 2, &[0.0, 1.0]);
    trip.common.hooks = ShaderHooks::trail(0, 0);
    trip.style_mut().params[0] = [0.5, 0.5, 0.0, 0.0];
    let img = draw(&device, &queue, &mut mr, &mut [&mut trip]).await;
    let (tail, head) = (px(&img, 64 - 45, 32)[0], px(&img, 64 - 8, 32)[0]);
    assert!(
        tail > 0 && tail < head,
        "fades towards the tail: {tail} < {head}"
    );
    assert!(is_bg(px(&img, 64 + 20, 32)), "nothing ahead of the head");
}

/// The kinds and styles added for tile and globe renderers, through Tint
/// and on WebGL2: sprites and images from an atlas, a dashed and cased path,
/// local positions, and a highlight.
#[wasm_bindgen_test]
async fn sprites_images_and_styled_lines_draw_in_the_browser() {
    let Some((device, queue)) = shared_gl().await else {
        panic!("no {BACKEND:?} adapter");
    };
    let mut mr = MarkRenderer::new(&device);
    let solid =
        |c: [u8; 4], n: usize| -> Vec<u8> { c.iter().copied().cycle().take(n * 4).collect() };
    let mut atlas = TextureAtlas::new(32);
    atlas
        .insert("red", 4, 4, &solid([255, 0, 0, 255], 16))
        .unwrap();
    let green = atlas
        .insert("green", 4, 4, &solid([0, 255, 0, 255], 16))
        .unwrap();
    let atlas = atlas.shared();
    // Pixel (64 + x, 32 - y).
    let mut icons = SpriteMarks::new(atlas.clone());
    icons.set_positions(&[[-50.0, 10.0, 0.0]]);
    icons.set_images(&["red"]);
    icons.style_mut().size_scale = 12.0;
    let mut images = ImageMarks::new(atlas);
    images.set_quads(&[ImageQuad {
        corner: [-30.0, 0.0, 0.0],
        right: [20.0, 0.0, 0.0],
        down: [0.0, -20.0, 0.0],
        rect: green,
    }]);
    // A 6 px path from x = -5 to 60 at y = 20: dashes of 12 px, gaps of 6,
    // with a dark casing and butt ends.
    let mut path = SegmentMarks::new();
    path.set_paths(&[vec![[-5.0, 20.0, 0.0], [60.0, 20.0, 0.0]]]);
    path.set_widths(&[6.0]);
    path.set_default_color([0, 0, 255, 255]);
    path.dash = Some([2.0, 1.0]);
    path.cap = LineCap::Butt;
    path.outline_width = 1.0;
    // A point given locally, a thousand units from its origin, highlighted.
    let mut pts = PointMarks::new();
    pts.set_local_positions([1000.0, 0.0, 0.0], &[[-960.0, -15.0, 0.0]]);
    pts.set_radii(&[5.0]);
    pts.set_default_color([255, 255, 255, 255]);
    pts.style_mut().highlight = Some(0);
    pts.style_mut().highlight_color = [255, 0, 255, 255];
    let img = draw(
        &device,
        &queue,
        &mut mr,
        &mut [&mut icons, &mut images, &mut path, &mut pts],
    )
    .await;
    assert_eq!(px(&img, 64 - 50, 32 - 10), [255, 0, 0, 255], "sprite");
    assert_eq!(px(&img, 64 - 20, 32 + 10), [0, 255, 0, 255], "image quad");
    assert!(px(&img, 64 + 1, 32 - 20)[2] > 200, "a dash, 6 px along");
    assert!(is_bg(px(&img, 64 + 11, 32 - 20)), "a gap, 16 px along");
    assert!(
        is_bg(px(&img, 64 - 7, 32 - 20)),
        "a butt end stops at the point"
    );
    assert_eq!(
        px(&img, 64 + 40, 32 + 15),
        [255, 0, 255, 255],
        "local point, highlighted"
    );
}

/// A mark kind defined outside threers — its own WGSL and its own texture
/// bind group — through Tint and on WebGL2.
#[wasm_bindgen_test]
async fn a_custom_mark_kind_draws_in_the_browser() {
    use threers::marks::{
        Column, ColumnFormat, CustomShader, MarkCommon, MarkKind, PrepareContext,
    };
    const WGSL: &str = r#"
@group(2) @binding(0) var sq_tex : texture_2d<f32>;
@group(2) @binding(1) var sq_samp : sampler;
struct SqIn {
    @builtin(vertex_index) vid : u32,
    @builtin(instance_index) iid : u32,
    @location(0) pos_hi : vec3<f32>,
    @location(1) pos_lo : vec3<f32>,
    /*CUSTOM_IN*/
};
struct SqOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) uv : vec2<f32>,
    @location(1) @interpolate(flat) index : u32,
};
@vertex
fn vs_main(in : SqIn) -> SqOut {
    var i : MarkInstance;
    i.position = mark_relative(in.pos_hi, in.pos_lo);
    i.position2 = i.position;
    i.size = 1.0;
    i.color = vec4<f32>(1.0);
    i.color2 = i.color;
    i.custom0 = /*CUSTOM0*/;
    i.custom1 = /*CUSTOM1*/;
    i.visible = 1.0;
    i.index = in.iid;
    i = hook_instance(i);
    var out : SqOut;
    let clip = frame.view_proj * vec4<f32>(i.position, 1.0);
    let c = mark_corner_signed(mark_quad_corner(in.vid));
    out.clip = mark_offset_px(clip, c * mark_size_px(i.size, clip.w));
    out.uv = c * 0.5 + 0.5;
    out.index = i.index;
    return out;
}
@fragment
fn fs_main(in : SqOut) -> @location(0) vec4<f32> {
    let t = textureSample(sq_tex, sq_samp, vec2<f32>(in.uv.x, 1.0 - in.uv.y));
    return mark_out(hook_color(vec4<f32>(t.rgb, 1.0), vec4<f32>(0.0), vec4<f32>(0.0)));
}
@fragment
fn fs_pick(in : SqOut) -> @location(0) vec2<u32> {
    return vec2<u32>(layer.ids.x, in.index);
}
"#;
    struct Squares {
        common: MarkCommon,
        shader: CustomShader,
        positions: Column,
        bind: Option<(wgpu::BindGroupLayout, wgpu::BindGroup)>,
    }
    impl Mark for Squares {
        fn kind(&self) -> MarkKind {
            self.shader.kind()
        }
        fn common(&self) -> &MarkCommon {
            &self.common
        }
        fn common_mut(&mut self) -> &mut MarkCommon {
            &mut self.common
        }
        fn instance_count(&self) -> usize {
            self.positions.len()
        }
        fn columns(&mut self) -> Vec<&mut Column> {
            vec![&mut self.positions]
        }
        fn custom_shader(&self) -> Option<&CustomShader> {
            Some(&self.shader)
        }
        fn extra_layout(&self) -> Option<&wgpu::BindGroupLayout> {
            self.bind.as_ref().map(|b| &b.0)
        }
        fn extra_bind_group(&self) -> Option<&wgpu::BindGroup> {
            self.bind.as_ref().map(|b| &b.1)
        }
        fn prepare(&mut self, ctx: &PrepareContext) {
            if self.bind.is_some() {
                return;
            }
            // Top row red, green; bottom row blue, white.
            let texels: [u8; 16] = [
                255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
            ];
            let size = wgpu::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            };
            let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            ctx.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &texels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(8),
                    rows_per_image: Some(2),
                },
                size,
            );
            let entries = [
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ];
            let layout = ctx
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &entries,
                });
            let view = tex.create_view(&Default::default());
            let sampler = ctx
                .device
                .create_sampler(&wgpu::SamplerDescriptor::default());
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });
            self.bind = Some((layout, bg));
        }
    }
    let Some((device, queue)) = shared_gl().await else {
        panic!("no {BACKEND:?} adapter");
    };
    let mut mr = MarkRenderer::new(&device);
    let mut positions = Column::new(ColumnFormat::PositionF64);
    positions.set_positions(&[[-30.0, 0.0, 0.0]]);
    let mut common = MarkCommon::default();
    common.style.size_scale = 8.0;
    let mut sq = Squares {
        common,
        shader: CustomShader::new("squares", WGSL, 2),
        positions,
        bind: None,
    };
    let img = draw(&device, &queue, &mut mr, &mut [&mut sq]).await;
    // Pixel (64 + x, 32 - y): the 16 px square's top-left quarter is red.
    assert_eq!(px(&img, 64 - 30 - 4, 32 - 4), [255, 0, 0, 255], "top left");
    assert_eq!(
        px(&img, 64 - 30 + 4, 32 + 4),
        [255, 255, 255, 255],
        "bottom right"
    );
}

/// Globe marks, clipping, draw order and the transition and combined hooks,
/// through Tint and on WebGL2.
#[wasm_bindgen_test]
async fn globe_clip_order_and_hooks_draw_in_the_browser() {
    use threers::marks::geo::{lon_lat_to_globe, GlobeGround};
    use threers::marks::{transition_offsets, PolygonMesh};
    let Some((device, queue)) = shared_gl().await else {
        panic!("no {BACKEND:?} adapter");
    };
    let mut mr = MarkRenderer::new(&device);
    // A sphere of radius 100 behind the screen: its front cap is near z = 0,
    // so lon/lat near (0, 0) shows face-on. Pixel (64 + x, 32 - y).
    let r = 100.0;
    let centre = [0.0, 0.0, -r];
    let on = |lon: f64, lat: f64| {
        let p = lon_lat_to_globe([lon, lat, 0.0], r);
        [p[0] + centre[0], p[1] + centre[1], p[2] + centre[2]]
    };
    // A polygon on the globe, subdivided to its curve.
    let globe = GlobeGround {
        radius: r,
        centre,
        max_edge_degrees: 2.0,
    };
    let square = vec![[-30.0, -10.0], [-20.0, -10.0], [-20.0, 10.0], [-30.0, 10.0]];
    let mut poly = PolygonMarks::new();
    poly.set_mesh(PolygonMesh::build_with(
        &[PolygonFeature::new(square)],
        0.0,
        &globe,
    ));
    poly.set_default_color([0, 200, 255, 255]);
    poly.common.style.depth_test = false;
    // A great-circle segment along the equator.
    let mut seg = SegmentMarks::new();
    seg.set_segments(&[on(0.0, 0.0)], &[on(25.0, 0.0)]);
    seg.globe = Some(centre);
    seg.set_widths(&[4.0]);
    seg.set_default_color([255, 255, 0, 255]);
    seg.common.style.depth_test = false;
    // Two coplanar image quads, the one drawn first ordered above the other.
    let mut atlas = TextureAtlas::new(16);
    let red = atlas
        .insert("r", 2, 2, &[255, 0, 0, 255].repeat(4))
        .unwrap();
    let green = atlas
        .insert("g", 2, 2, &[0, 255, 0, 255].repeat(4))
        .unwrap();
    let atlas = atlas.shared();
    let quad = |rect| ImageQuad {
        corner: [30.0, 10.0, 0.0],
        right: [20.0, 0.0, 0.0],
        down: [0.0, -20.0, 0.0],
        rect,
    };
    let mut under = ImageMarks::new(atlas.clone());
    under.set_quads(&[quad(green)]);
    under.common.style.depth_order = 0;
    let mut over = ImageMarks::new(atlas);
    over.set_quads(&[quad(red)]);
    over.common.style.depth_order = 1;
    // Points: one clipped away, one moved half way by a transition hook
    // combined with a filter that keeps it.
    let mut pts = PointMarks::new();
    pts.set_positions(&[[0.0, -20.0, 0.0], [50.0, -20.0, 0.0]]);
    pts.set_radii(&[4.0, 4.0]);
    pts.set_default_color([255, 0, 255, 255]);
    let mut custom = transition_offsets(
        &[[-20.0, -20.0, 0.0], [50.0, -20.0, 0.0]],
        &[[0.0, -20.0, 0.0], [50.0, -20.0, 0.0]],
    );
    pts.common.set_custom(0, 3, &custom);
    pts.common.set_custom(1, 1, &[1.0, 1.0]);
    pts.common.hooks =
        ShaderHooks::transition(0, None, 1).combine(&ShaderHooks::range_filter(4, 0));
    pts.common.style.params[0] = [0.0, 2.0, 0.0, 0.0];
    pts.common.style.params[1] = [0.5, 0.0, 0.0, 0.0];
    // Clip to x < 40 on the ground; the camera looks along -Z, so the ground
    // rectangle in (x, -z) is a strip in x here.
    pts.common.style.clip = Some([-100.0, -100.0, 40.0, 100.0]);
    let img = draw(
        &device,
        &queue,
        &mut mr,
        &mut [&mut poly, &mut seg, &mut over, &mut under, &mut pts],
    )
    .await;
    // Longitude -25 sits at x = 100 sin(-25 deg), about -42.
    assert!(!is_bg(px(&img, 64 - 42, 32)), "the globe polygon");
    assert!(!is_bg(px(&img, 64 + 10, 32)), "the great-circle segment");
    assert_eq!(
        px(&img, 64 + 40, 32),
        [255, 0, 0, 255],
        "the higher order quad wins"
    );
    assert_eq!(
        px(&img, 64 - 10, 32 + 20),
        [255, 0, 255, 255],
        "transitioned half way"
    );
    assert!(is_bg(px(&img, 64 + 50, 32 + 20)), "clipped away");
}

#[wasm_bindgen_test]
async fn density_and_picking_work_in_the_browser() {
    let Some((device, queue)) = shared_gl().await else {
        panic!("no {BACKEND:?} adapter");
    };
    let mut mr = MarkRenderer::new(&device);
    let mut heat = DensityMarks::new();
    heat.set_positions(&[[-30.0, 0.0, 0.0]; 8]);
    heat.style_mut().size_scale = 12.0;
    heat.max_weight = 6.0;
    let img = draw(&device, &queue, &mut mr, &mut [&mut heat]).await;
    assert!(
        !is_bg(px(&img, 34, 32)),
        "density drawn: {:?}",
        px(&img, 34, 32)
    );
    assert!(is_bg(px(&img, 100, 32)));

    // Scaled to its own maximum, through the max reduction: weights far too
    // light for `max_weight` still reach the top of the ramp (inferno's pale
    // yellow) at the peak.
    heat.set_weights(&[0.001; 8]);
    heat.auto_max = Some(1.0);
    let img = draw(&device, &queue, &mut mr, &mut [&mut heat]).await;
    let peak = px(&img, 34, 32);
    assert!(peak[0] > 200 && peak[1] > 200, "auto max peak: {peak:?}");
    heat.auto_max = None;
    let img = draw(&device, &queue, &mut mr, &mut [&mut heat]).await;
    assert!(
        is_bg(px(&img, 34, 32)),
        "fixed max, light weights: {:?}",
        px(&img, 34, 32)
    );

    let mut points = PointMarks::new();
    points.set_positions(&[[-40.0, 0.0, 0.0], [20.0, 10.0, 0.0]]);
    points.set_radii(&[5.0, 5.0]);
    let view = MarkView::new(&camera(), Origin::ZERO, (W, H));
    let mut req = mr.pick(
        &device,
        &queue,
        &view,
        &mut [&mut points],
        None,
        84.5,
        22.5,
        1,
    );
    let mut hit = None;
    for _ in 0..4000 {
        let _ = device.poll(wgpu::PollType::Poll);
        yield_to_event_loop().await;
        if let Some(h) = req.try_take() {
            hit = Some(h);
            break;
        }
    }
    let hit = hit.expect("pick never resolved").expect("hit the point");
    assert_eq!((hit.mark, hit.index), (0, 1));
}

#[cfg(feature = "captions")]
#[wasm_bindgen_test]
async fn labels_draw_in_the_browser() {
    use threers::captions::CaptionFont;
    use threers::marks::{TextLabel, TextLabels};
    let Some((device, queue)) = shared_gl().await else {
        panic!("no {BACKEND:?} adapter");
    };
    let mut mr = MarkRenderer::new(&device);
    let mut labels = TextLabels::new(CaptionFont::builtin());
    labels.font_size = 16.0;
    labels.set_labels(vec![TextLabel::new([0.0, 0.0, 0.0], "HI")]);
    let img = draw(&device, &queue, &mut mr, &mut [&mut labels]).await;
    let ink = (24..40)
        .flat_map(|y| (48..80).map(move |x| (x, y)))
        .filter(|&(x, y)| px(&img, x, y)[0] > 128)
        .count();
    assert!(ink > 20, "text drawn: {ink} bright pixels");

    // Set along a diagonal line: turned glyphs through the same shader.
    labels.set_labels(vec![TextLabel::along(
        vec![[-50.0, -25.0, 0.0], [50.0, 25.0, 0.0]],
        "ROAD",
    )]);
    let img = draw(&device, &queue, &mut mr, &mut [&mut labels]).await;
    let ink = img.chunks(4).filter(|p| p[0] > 128).count();
    assert_eq!(labels.visible(), &[true]);
    assert!(ink > 20, "a label along a line drawn: {ink} bright pixels");
    // Multi-channel fields from a TrueType face, with an icon beside the
    // text: the RGBA atlas and the median through Chrome's compilers.
    let font =
        CaptionFont::from_ttf_bytes(include_bytes!("../src/openscad/DejaVuSans.ttf")).unwrap();
    let mut msdf = TextLabels::new(font);
    msdf.glyph_field = threers::marks::GlyphField::Msdf;
    msdf.font_size = 20.0;
    msdf.add_icon("dot", 8, 8, &[255; 64]);
    let mut icon = threers::marks::LabelIcon::new("dot", 12.0);
    icon.color = [0, 255, 0, 255];
    msdf.set_labels(vec![TextLabel::new([0.0, 0.0, 0.0], "Zü").with_icon(icon)]);
    let img = draw(&device, &queue, &mut mr, &mut [&mut msdf]).await;
    let text = img.chunks(4).filter(|p| p[0] > 128 && p[1] > 128).count();
    let green = img.chunks(4).filter(|p| p[1] > 200 && p[0] < 60).count();
    assert!(text > 40, "msdf text drawn: {text} px");
    assert!(green > 40, "its icon drawn: {green} px");
}
