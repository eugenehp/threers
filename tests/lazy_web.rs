//! A lazy renderer, warmed up, on **WebGPU** in a real browser.
//!
//! Native tests cannot say whether the web half of the warm-up works: there a
//! blocking `step` cannot hear validation's verdict, so `step_async` and
//! `run_async` exist, and they rest on a validation error scope that resolves
//! through a JS promise. This runs them in Chrome.
//!
//! ```sh
//! WEBGPU=1 TEST=lazy_web scripts/wasm-browser-test.sh
//! ```
//!
//! Every test fails, rather than passing on a fallback, if the browser hands
//! over anything but a WebGPU adapter.
#![cfg(target_arch = "wasm32")]

use threers::gpu::GpuContext;
use threers::{
    AmbientLight, BoxGeometry, Color, DirectionalLight, Mesh, Object3D, PerspectiveCamera,
    RenderState, RenderTarget, Renderer, Scene, StandardMaterial, Vector3,
};
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_browser);

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const N: u32 = 64;

async fn yield_to_event_loop() {
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        web_sys::window()
            .unwrap()
            .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 0)
            .unwrap();
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

async fn gpu() -> GpuContext {
    let gpu = GpuContext::headless()
        .await
        .expect("no WebGPU adapter — run with WEBGPU=1");
    assert_eq!(gpu.adapter.get_info().backend, wgpu::Backend::BrowserWebGpu);
    gpu
}

fn scene() -> (Scene, PerspectiveCamera) {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x182030);
    scene.add(Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.0, 1.0, 1.0),
        StandardMaterial::new(Color::from_hex(0xc08040))
            .with_roughness(0.6)
            .into(),
    )));
    scene.add_light(AmbientLight::new(Color::WHITE, 0.5));
    scene.add_light(
        DirectionalLight::new(Color::WHITE, 1.0).with_direction(Vector3::new(-0.5, -1.0, -0.4)),
    );
    let mut camera = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
    camera.position = Vector3::new(2.5, 2.0, 3.5);
    camera.look_at(Vector3::ZERO);
    (scene, camera)
}

/// The frame, read back without blocking the thread that delivers the map
/// callback.
async fn image(gpu: &GpuContext, renderer: &mut Renderer) -> Vec<u8> {
    renderer.set_reversed_z(false);
    let (mut scene, camera) = scene();
    let target = RenderTarget::new(&gpu.device, N, N, FORMAT);
    target.set_samples(4);
    target.set_presents(true);
    renderer.render_to(&mut scene, &camera, &target);

    let buffer = threers::renderer::readback::staging_buffer(&gpu.device, &target.color_texture);
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    threers::renderer::readback::copy_texture_to_buffer(
        &mut encoder,
        &target.color_texture,
        &buffer,
    );
    gpu.queue.submit(Some(encoder.finish()));
    let (tx, mut rx) = futures_channel::oneshot::channel();
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    for _ in 0..600 {
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        yield_to_event_loop().await;
        if let Ok(Some(r)) = rx.try_recv() {
            r.expect("map failed");
            let data = buffer
                .slice(..)
                .get_mapped_range()
                .expect("range is mapped");
            return threers::renderer::readback::unpad(&data, N, N, FORMAT);
        }
    }
    panic!("the buffer never finished mapping");
}

/// The mechanism `step_async` rests on: a validation error made inside a scope
/// comes back from `pop().await`, in a browser, and a clean scope returns none.
#[wasm_bindgen_test]
async fn a_browser_error_scope_reports_a_rejected_pipeline() {
    let gpu = gpu().await;
    let shader = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("error scope probe"),
            source: wgpu::ShaderSource::Wgsl(
                "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(0.); }\n\
             @fragment fn fs() -> @location(0) vec4f { return vec4f(1.); }"
                    .into(),
            ),
        });
    let targets = [Some(wgpu::ColorTargetState::from(FORMAT))];
    let build = |entry: &str| {
        gpu.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("probe"),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(entry),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    targets: &targets,
                    compilation_options: Default::default(),
                }),
                multiview_mask: None,
                cache: None,
            })
    };
    let clean = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let _ = build("vs");
    assert!(
        clean.pop().await.is_none(),
        "a valid pipeline reported an error"
    );
    let bad = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let _ = build("no_such_entry_point");
    let error = bad
        .pop()
        .await
        .expect("the browser did not report a rejected pipeline");
    web_sys::console::log_1(&format!("[lazy-web] rejected pipeline reported: {error}").into());
}

/// Warming a lazy renderer through the awaiting API builds every pipeline,
/// reports nothing, and the frame is the eager renderer's.
#[wasm_bindgen_test]
async fn a_lazy_renderer_warms_up_on_webgpu_and_matches_eager() {
    let gpu = gpu().await;
    let mut eager = Renderer::new(gpu.device.clone(), gpu.queue.clone(), FORMAT, N, N);
    let expected = image(&gpu, &mut eager).await;
    assert!(
        expected.chunks(4).any(|p| p[..3] != expected[..3]),
        "the eager frame is one flat colour, so equality below would prove nothing"
    );

    let mut lazy = Renderer::new_lazy(gpu.device.clone(), gpu.queue.clone(), FORMAT, N, N);
    let stock = lazy.pending_pipelines();
    assert!(stock > 0, "a lazy renderer starts with pipelines to build");

    let states = [
        RenderState {
            depth_write: Some(false),
            ..RenderState::default()
        },
        RenderState {
            depth_test: false,
            ..RenderState::default()
        },
    ];
    let warmup = lazy.warm_up_states(&states);
    assert!(warmup.len() > stock, "no state variants were queued");
    let total = warmup.len();
    assert!(!warmup.is_ready());
    assert_eq!(warmup.run_async().await, total);
    assert!(warmup.is_ready());
    assert_eq!(warmup.progress(), (total, total));
    assert!(warmup.errors().is_empty(), "{:?}", warmup.errors());
    assert!(lazy.pipeline_failures().is_empty());
    assert_eq!(lazy.pending_pipelines(), 0);
    assert!(!warmup.step_async().await, "a finished job stays finished");
    web_sys::console::log_1(
        &format!(
            "[lazy-web] {stock} stock + {} variant pipelines built and validated",
            total - stock
        )
        .into(),
    );

    assert!(
        image(&gpu, &mut lazy).await == expected,
        "warmed lazy pixels differ from eager on WebGPU"
    );
}

/// Without warming at all: pipelines built by the first draw, as a browser
/// will do for a renderer nobody warmed.
#[wasm_bindgen_test]
async fn an_unwarmed_lazy_renderer_draws_the_same_frame_on_webgpu() {
    let gpu = gpu().await;
    let mut eager = Renderer::new(gpu.device.clone(), gpu.queue.clone(), FORMAT, N, N);
    let mut lazy = Renderer::new_lazy(gpu.device.clone(), gpu.queue.clone(), FORMAT, N, N);
    let before = lazy.pending_pipelines();
    let frame = image(&gpu, &mut lazy).await;
    assert!(
        lazy.pending_pipelines() < before,
        "drawing built no pipeline"
    );
    assert!(
        frame == image(&gpu, &mut eager).await,
        "first-use pixels differ from eager on WebGPU"
    );
}

/// The JS-facing API a page uses: `newLazy`, then await `warmUp().run()`.
#[wasm_bindgen_test]
async fn the_page_api_warms_a_lazy_web_renderer() {
    use wasm_bindgen::JsCast;
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(128);
    canvas.set_height(96);
    document.body().unwrap().append_child(&canvas).unwrap();
    let renderer = threers::wasm::WebRenderer::new_lazy(canvas)
        .await
        .expect("no lazy WebGPU renderer");
    assert_eq!(renderer.backend(), "webgpu", "run with WEBGPU=1");
    let job = renderer.warm_up();
    assert!(job.total() > 0, "a lazy renderer has pipelines to build");
    assert_eq!(job.built(), 0);
    let built = job.run().await.expect("a pipeline was rejected");
    assert_eq!(built, job.total());
    assert_eq!(job.built(), job.total());
    assert_eq!(renderer.warm_up().total(), 0, "nothing left to warm");
}
