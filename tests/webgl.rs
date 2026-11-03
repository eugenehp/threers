//! The WebGL2 backend, in a real browser.
//!
//! The point of the `webgl` feature is that a browser without WebGPU still
//! renders, and the only way to know that is to run one. Headless Chrome is
//! launched with `--disable-gpu`, so WebGPU is unavailable and WebGL2 comes from
//! SwiftShader — which is the situation the feature exists for.
//!
//! ```sh
//! scripts/wasm-browser-test.sh --features webgl --test webgl
//! ```
#![cfg(all(feature = "webgl", target_arch = "wasm32"))]


use threers::renderer::caps;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_browser);

/// Hand control back to the browser for one turn of the event loop.
///
/// Needed because a wasm test runs on the same thread as everything else: any
/// wait that does not yield starves the callbacks it is waiting for.
async fn yield_to_event_loop() {
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        web_sys::window()
            .unwrap()
            .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 0)
            .unwrap();
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

// One canvas per *kind* of test, not one per test.
//
// A browser caps how many live WebGL2 contexts a page may hold, and a context
// is not released when the canvas goes out of scope — so eight tests each
// making their own exhausted the pool and one of them hung waiting for a
// context that was never going to arrive. The raw-wgpu tests share one adapter
// and the drawing tests share one renderer.
/// Everything one live WebGL2 context needs kept alive.
///
/// The surface and instance are held only to keep the context open; they are
/// never handed out, because `wgpu::Surface` is not `Clone`. The adapter,
/// device and queue are, so those are cloned to each caller.
type SharedGl = (
    wgpu::Adapter,
    wgpu::Device,
    wgpu::Queue,
    wgpu::Surface<'static>,
    wgpu::Instance,
);

thread_local! {
    static SHARED_GL: std::cell::RefCell<Option<SharedGl>> =
        const { std::cell::RefCell::new(None) };
}

/// The one raw-wgpu adapter, device and queue the suite shares.
async fn shared_gl() -> (wgpu::Adapter, wgpu::Device, wgpu::Queue) {
    if let Some(triple) = SHARED_GL.with(|g| {
        g.borrow()
            .as_ref()
            .map(|(a, d, q, _, _)| (a.clone(), d.clone(), q.clone()))
    }) {
        return triple;
    }
    let (adapter, surface, instance) = gl_adapter().await.expect("no GL adapter in this browser");
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("threers webgl test"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            ..Default::default()
        })
        .await
        .expect("no device from the GL adapter");
    let triple = (adapter.clone(), device.clone(), queue.clone());
    SHARED_GL.with(|g| *g.borrow_mut() = Some((adapter, device, queue, surface, instance)));
    triple
}

/// Ask for a GL adapter, through a canvas, and report what came back.
///
/// The canvas is not incidental. A WebGL2 context belongs to a canvas, so wgpu's
/// GL backend has nowhere to get one without a surface — `compatible_surface:
/// None` yields no adapter at all, which reads exactly like "this browser has no
/// WebGL2" and is not that. The surface is returned so it outlives the adapter.
async fn gl_adapter() -> Option<(wgpu::Adapter, wgpu::Surface<'static>, wgpu::Instance)> {
    let document = web_sys::window()?.document()?;
    let canvas = document
        .create_element("canvas")
        .ok()?
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .ok()?;
    canvas.set_width(64);
    canvas.set_height(64);
    // Attached, because a detached canvas does not reliably get a context.
    document.body()?.append_child(&canvas).ok()?;

    let mut d = wgpu::InstanceDescriptor::new_without_display_handle();
    d.backends = wgpu::Backends::GL;
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
    Some((adapter, surface, instance))
}

/// The browser offers a WebGL2 adapter, and it really is the GL backend.
///
/// Without the `webgl` feature wgpu has no GL backend compiled in at all, so
/// this failing means the feature is not doing what it claims rather than that
/// the browser is old.
#[wasm_bindgen_test]
async fn a_browser_offers_a_gl_adapter() {
    let (adapter, _device, _queue) = shared_gl().await;
    let info = adapter.get_info();
    assert_eq!(
        info.backend,
        wgpu::Backend::Gl,
        "asked for GL and got {:?} — {info:?}",
        info.backend
    );
}

/// A WebGL2 device reports no compute and no storage buffers.
///
/// This is the same thing `renderer::caps` is unit-tested against, but measured
/// on a device a browser actually handed over rather than on wgpu's idea of what
/// WebGL2 looks like. If these ever disagree, the unit test is the one that is
/// wrong.
#[wasm_bindgen_test]
async fn a_real_webgl2_device_has_neither_compute_nor_storage() {
    let (_adapter, device, _queue) = shared_gl().await;
    assert!(
        !caps::supports_compute(&device),
        "a WebGL2 device claims compute: {:?}",
        device.limits()
    );
    assert!(
        !caps::supports_storage_buffers(&device),
        "a WebGL2 device claims storage buffers"
    );
}

/// Bloom and ambient occlusion work here; what is left declines.
///
/// Bloom used to be in the second group. It is three stages — threshold and
/// downsample, then a separable blur in two passes — and none of that needs a
/// compute stage, so on a device without one it records them as fullscreen draws
/// instead. The two implementations are held to each other in
/// `renderer::bloom::tests::the_two_bloom_paths_agree`, on a device that has
/// both; this checks only that the raster one is what a WebGL2 device gets.
///
/// The packed post chain is genuinely compute-only — it writes its output
/// through a storage texture — and still returns `None`, which its callers
/// already treat as "effect unavailable" rather than as a failure. A pipeline
/// that was never created would instead be a frame that quietly misses a step.
#[wasm_bindgen_test]
async fn bloom_and_ambient_occlusion_work_and_the_rest_decline() {
    let (_adapter, device, _queue) = shared_gl().await;

    let src = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("src"),
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let bloom = threers::renderer::bloom::Bloom::new(&device, &src, 1.0, 1.0);
    assert!(
        bloom.is_some(),
        "bloom should fall back to its raster path on a device with no compute"
    );
    // Ambient occlusion took the same route as bloom: it reads a depth texture
    // and writes one value per pixel, which a fragment shader does as readily.
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    assert!(
        threers::renderer::ssao::Ssao::new(&device, &depth, 0.1, 50.0, 1.0, 1.0, 0.5, 1.0)
            .is_some(),
        "ambient occlusion should fall back to its raster path here"
    );

    assert!(
        threers::renderer::rgb_pack::RgbPack::new(&device, &src).is_none(),
        "the packed post chain writes through a storage texture and cannot run here"
    );
}

/// Something actually draws: a clear, read back, and the right colour.
///
/// The checks above are all about what the device *says*. This is the one that
/// shows the backend works end to end in a browser — a render pass, a
/// texture-to-buffer copy, and a mapped readback, which is the whole path a
/// frame takes. A backend that reported itself correctly and drew nothing would
/// pass every other test here.
#[wasm_bindgen_test]
async fn a_webgl2_device_renders_and_reads_back() {
    let (_adapter, device, queue) = shared_gl().await;

    const W: u32 = 16;
    const H: u32 = 16;
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("target"),
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    // 256 bytes a row is the copy alignment, and 16 px of RGBA8 is 64 — so the
    // buffer is padded and the readback has to step by the padded stride.
    let bytes_per_row = 256u32;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (bytes_per_row * H) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Values that are exact in 8-bit so the assertion can be
                    // exact: 51 = 0.2, 102 = 0.4, 153 = 0.6 of 255.
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.2,
                        g: 0.4,
                        b: 0.6,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    let (tx, mut rx) = futures_channel::oneshot::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
    // Mapping finishes on the event loop, so the thread must not block waiting
    // for it: `wait_indefinitely()` blocks the very loop that would deliver the
    // callback, and the test hangs until the driver is killed. Poll, yield, and
    // let the browser get on with it.
    let mut mapped = None;
    for _ in 0..600 {
        let _ = device.poll(wgpu::PollType::Poll);
        yield_to_event_loop().await;
        if let Ok(Some(r)) = rx.try_recv() {
            mapped = Some(r);
            break;
        }
    }
    mapped
        .expect("the buffer never finished mapping")
        .expect("map failed");

    let data = readback
        .slice(..)
        .get_mapped_range()
        .expect("buffer range is mapped");
    let px = &data[..4];
    // sRGB is not involved: an `Rgba8Unorm` target stores the clear value as
    // given, so these are the clear colour times 255 and nothing else.
    assert_eq!(
        (px[0], px[1], px[2], px[3]),
        (51, 102, 153, 255),
        "WebGL2 cleared to something other than the colour asked for"
    );
}

/// The crate's own renderer builds every pipeline and runs a frame on WebGL2.
///
/// This is the test the feature exists for. The four above show that wgpu's GL
/// backend works; this one compiles *threers'* own shaders through naga into
/// GLSL ES 3.0 and runs the frame with them, which is what catches a WGSL
/// construct that cannot be lowered, a bind group over WebGL2's limits, or a
/// pipeline needing a downlevel flag the device lacks. Five such problems stood
/// between the backend and a frame:
///
/// - `ShaderMaterial`'s three storage buffers, against a limit of zero.
/// - `textureNumLevels` for the PMREM chain length — no GLSL ES 3.0 equivalent,
///   and not recoverable from `textureDimensions` because a PMREM chain is
///   shorter than a full one. Now carried in a uniform.
/// - `textureLoad` from a depth texture, in the screen-space glass pass and in
///   the temporal resolve. Both passes are skipped on this backend.
/// - `INDEPENDENT_BLEND`, which the order-independent transparency pass needs
///   for its two differently-blended attachments. Also skipped.
/// - The same depth `textureLoad` in post-processing, which *is* on the ordinary
///   path — so there the function's existing sampled-depth arm is substituted
///   for both.
#[wasm_bindgen_test]
async fn the_renderer_runs_a_frame_on_webgl2() {
    let mut renderer = webgl_renderer().await;
    assert_eq!(
        renderer.backend(),
        "gl",
        "expected the WebGL2 backend in a browser with no WebGPU"
    );
    assert!(
        !renderer.supports_compute(),
        "WebGL2 reported a compute stage"
    );

    let mut scene = threers::wasm::WebScene::new();
    let camera = threers::wasm::WebCamera::perspective(60.0, 128.0 / 96.0, 0.1, 100.0);
    // Two frames: the first builds whatever is built lazily, the second reuses
    // it, and a pipeline that failed to compile shows up on either.
    renderer.render(&mut scene, &camera);
    renderer.render(&mut scene, &camera);
}

// Viewport, scissor, partial clear, several renders per frame, QuadView and
// polygon wireframes — the same tests run on WebGPU from `tests/webgpu.rs`.
#[path = "web_views/mod.rs"]
mod web_views;

// Post-fx onto the canvas, right way up — shared with the other backend.
#[path = "web_postfx/mod.rs"]
mod web_postfx;

// Material maps: shader variants and every sampler — shared with the other
// backend.
#[path = "web_maps/mod.rs"]
mod web_maps;

// Whole scenes: lit, skinned, refracting, shadowed and post-processed draws,
// ShaderMaterial, morph targets and groups — shared with the other backend.
#[path = "web_scenes/mod.rs"]
mod web_scenes;

type SharedViews = std::rc::Rc<std::cell::RefCell<threers::wasm::WebRenderer>>;

thread_local! {
    static VIEWS: std::cell::RefCell<Option<(SharedViews, web_sys::HtmlCanvasElement)>> =
        const { std::cell::RefCell::new(None) };
}

/// Take `navigator.gpu` away from this page, so the renderer's WebGPU probe
/// finds nothing and it starts on WebGL2 — the backend this suite exists for.
///
/// Browser flags cannot be relied on for this: Chrome for Testing 155.0.8059.84
/// hands headless pages a WebGPU adapter even with `--disable-features=WebGPU`,
/// and then the whole suite quietly ran on WebGPU. An own property shadows the
/// prototype's getter, the same `undefined` a browser without WebGPU reports.
fn hide_webgpu() {
    let navigator: JsValue = web_sys::window().unwrap().navigator().into();
    let descriptor = js_sys::Object::new();
    js_sys::Reflect::set(&descriptor, &"value".into(), &JsValue::UNDEFINED).unwrap();
    js_sys::Reflect::set(&descriptor, &"configurable".into(), &JsValue::TRUE).unwrap();
    js_sys::Object::define_property(navigator.unchecked_ref::<js_sys::Object>(), &"gpu".into(), &descriptor);
}

/// The one renderer (and its canvas) the `web_views` tests share — one more
/// WebGL2 context, not one per test.
async fn views_renderer() -> (SharedViews, web_sys::HtmlCanvasElement) {
    // A test that panicked never dropped its borrow (wasm panics do not
    // unwind), so after a failure the next test gets a fresh renderer rather
    // than a cascade of "already borrowed".
    if let Some(v) = VIEWS.with(|v| v.borrow().clone()) {
        if v.0.try_borrow_mut().is_ok() {
            return v;
        }
    }
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(128);
    canvas.set_height(96);
    document.body().unwrap().append_child(&canvas).unwrap();
    hide_webgpu();
    let renderer = threers::wasm::WebRenderer::new(canvas.clone())
        .await
        .expect("the renderer would not start on this browser");
    assert_eq!(renderer.backend(), "gl", "the WebGL2 suite ran on another backend");
    let v = (std::rc::Rc::new(std::cell::RefCell::new(renderer)), canvas);
    VIEWS.with(|slot| *slot.borrow_mut() = Some(v.clone()));
    v
}

/// A renderer attached to a fresh canvas, for the two tests above.
async fn webgl_renderer() -> threers::wasm::WebRenderer {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(128);
    canvas.set_height(96);
    document.body().unwrap().append_child(&canvas).unwrap();
    hide_webgpu();
    threers::wasm::WebRenderer::new(canvas)
        .await
        .expect("the renderer would not start on this browser")
}


