//! The fluid simulation on a real WebGL2 device, in a real browser.
//!
//! The claim this suite exists to check is that the fluid is *complete* on
//! WebGL2. It is the one substantial piece of this crate that needs no compute
//! stage and no storage buffers — nineteen fullscreen fragment passes over
//! float targets — so unlike bloom or ambient occlusion it has no reduced path
//! to fall back to. Either the whole thing runs there or the claim is wrong,
//! and reading the spec cannot tell you which.
//!
//! ```sh
//! FEATURES=fluid,webgl TEST=fluid_webgl scripts/wasm-browser-test.sh
//! ```
//!
//! Headless Chrome has no WebGPU, and its WebGL2 comes from SwiftShader — a
//! software rasteriser, which is a harsher test of format support than a GPU
//! driver is.
#![cfg(all(feature = "fluid", feature = "webgl", target_arch = "wasm32"))]


use threers::fluid::{quality, FluidConfig, FluidFormats, FluidSim};
use wasm_bindgen::JsCast;
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_browser);

/// Hand control back to the browser for one turn of the event loop. A wasm test
/// runs on the same thread as everything else, so a wait that does not yield
/// starves the callbacks it is waiting for.
async fn yield_to_event_loop() {
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        web_sys::window()
            .unwrap()
            .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 0)
            .unwrap();
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

/// One canvas, one context, for the whole suite.
///
/// A browser caps how many live WebGL2 contexts a page may hold, and a context
/// is not released when its canvas goes out of scope — so a context per test
/// exhausts the pool and the last one hangs waiting for one that will never
/// arrive. The surface and instance are kept only to hold the context open.
type SharedGl = (
    std::sync::Arc<wgpu::Device>,
    std::sync::Arc<wgpu::Queue>,
    FluidFormats,
    wgpu::Surface<'static>,
    wgpu::Instance,
);

thread_local! {
    static SHARED_GL: std::cell::RefCell<Option<SharedGl>> =
        const { std::cell::RefCell::new(None) };
}

async fn shared_gl() -> Option<(
    std::sync::Arc<wgpu::Device>,
    std::sync::Arc<wgpu::Queue>,
    FluidFormats,
)> {
    if let Some(t) = SHARED_GL.with(|g| {
        g.borrow()
            .as_ref()
            .map(|(d, q, f, _, _)| (d.clone(), q.clone(), *f))
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
    d.backends = wgpu::Backends::GL;
    let instance = wgpu::Instance::new(d);
    // A WebGL2 context *belongs* to a canvas, so wgpu's GL backend has nowhere
    // to get one without a surface: `compatible_surface: None` yields no
    // adapter at all, which reads exactly like "this browser has no WebGL2".
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
    let formats = FluidFormats::for_adapter(&adapter)?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("threers fluid webgl test"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            ..Default::default()
        })
        .await
        .ok()?;
    let (device, queue) = (std::sync::Arc::new(device), std::sync::Arc::new(queue));
    SHARED_GL.with(|g| {
        *g.borrow_mut() = Some((device.clone(), queue.clone(), formats, surface, instance))
    });
    Some((device, queue, formats))
}

/// Render the fluid into an `Rgba8Unorm` texture and read it back, the long way
/// round — the native `capture_rgba` blocks on the queue, and a browser's main
/// thread is the one thread that must not be blocked.
async fn capture(
    sim: &mut FluidSim,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("fluid webgl capture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    sim.render_capture(&target);

    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = (width * 4).div_ceil(align) * align;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fluid webgl readback"),
        size: (padded * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
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
                bytes_per_row: Some(padded),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([enc.finish()]);

    let (tx, mut rx) = futures_channel::oneshot::channel();
    readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    // Mapping finishes on the event loop, so this polls and yields rather than
    // waiting: `wait_indefinitely()` would block the very loop that delivers
    // the callback. The budget is generous because the device here is
    // SwiftShader — a software rasteriser running forty fullscreen passes.
    let mut done = None;
    for _ in 0..4000 {
        let _ = device.poll(wgpu::PollType::Poll);
        yield_to_event_loop().await;
        if let Ok(Some(r)) = rx.try_recv() {
            done = Some(r);
            break;
        }
    }
    done.expect("the buffer never finished mapping")
        .expect("map failed");

    let data = readback
        .slice(..)
        .get_mapped_range()
        .expect("buffer range is mapped");
    let mut out = Vec::with_capacity((width * height * 4) as usize);
    for row in 0..height {
        let at = (row * padded) as usize;
        out.extend_from_slice(&data[at..at + (width * 4) as usize]);
    }
    drop(data);
    readback.unmap();
    out
}

/// WebGL2 has no half-float render targets without `EXT_color_buffer_float`,
/// and no simulation at all without them. This says which formats this browser
/// actually gave us, so a failure downstream has somewhere to start.
#[wasm_bindgen_test]
async fn the_browser_offers_a_renderable_float_format() {
    let Some((_, _, formats)) = shared_gl().await else {
        panic!("no WebGL2 adapter, or no filterable half-float render target");
    };
    web_sys::console::log_1(&format!("fluid formats on WebGL2: {formats:?}").into());
    // Whatever the narrow formats came out as, the dye has to be four-channel
    // float or there is nowhere to put the colour.
    assert_eq!(formats.dye, wgpu::TextureFormat::Rgba16Float);
}

/// The whole frame — every pass, the pressure solve, the bloom chain, the
/// sunrays and the composite — on WebGL2, with the picture read back to prove
/// it drew rather than merely not erroring.
///
/// At the `low` quality level, because the device is SwiftShader: the default
/// `high` carries a 1024-texel dye grid, and advecting that in software takes
/// longer than the harness will wait. Which passes run does not depend on the
/// grid size, so this covers the same code either way.
#[wasm_bindgen_test]
async fn the_whole_simulation_runs_on_webgl2() {
    let Some((device, queue, formats)) = shared_gl().await else {
        panic!("no WebGL2 adapter");
    };
    let (w, h) = (192u32, 192u32);
    let mut sim = FluidSim::with_formats(
        device.clone(),
        queue.clone(),
        w,
        h,
        FluidConfig::default().with_quality(quality("low").expect("low")),
        formats,
    )
    .expect("the fluid refused a WebGL2 device");

    // Dye only, no velocity: the splat stays where it was put, so the frame
    // says both that something drew and that it drew in the right place. In
    // texture space y runs *down*, so a quarter of the way down is the top.
    sim.splat(0.25, 0.25, 0.0, 0.0, [0.6, 0.6, 0.6]);
    sim.update(1.0 / 60.0);

    let rgba = capture(&mut sim, &device, &queue, w, h).await;
    assert_eq!(rgba.len(), (w * h * 4) as usize);

    let mut best = (0u32, 0u32, 0u8);
    for y in 0..h {
        for x in 0..w {
            let at = ((y * w + x) * 4) as usize;
            let v = rgba[at].max(rgba[at + 1]).max(rgba[at + 2]);
            if v > best.2 {
                best = (x, y, v);
            }
        }
    }
    let (bx, by, v) = best;
    assert!(v > 32, "WebGL2 drew nothing (brightest pixel {v})");
    let (fx, fy) = (bx as f32 / w as f32, by as f32 / h as f32);
    assert!((fx - 0.25).abs() < 0.08, "brightest at x {fx}, wanted 0.25");
    assert!((fy - 0.25).abs() < 0.08, "brightest at y {fy}, wanted 0.25");
}

/// Twenty frames with the solver fully loaded, to catch anything that only
/// fails once a field has been through the ping-pong a few times — which is
/// where a backend's framebuffer-completeness rules tend to bite.
#[wasm_bindgen_test]
async fn twenty_frames_of_a_driven_fluid_on_webgl2() {
    let Some((device, queue, formats)) = shared_gl().await else {
        panic!("no WebGL2 adapter");
    };
    let mut sim = FluidSim::with_formats(
        device.clone(),
        queue.clone(),
        160,
        120,
        FluidConfig::default().with_quality(quality("low").expect("low")),
        formats,
    )
    .expect("fluid");
    sim.multiple_splats(6);
    for i in 0..20 {
        let t = i as f32 / 20.0;
        sim.pointer_down(1, 20.0 + 100.0 * t, 30.0 + 60.0 * t);
        sim.pointer_move(1, 30.0 + 100.0 * t, 40.0 + 60.0 * t);
        sim.update(1.0 / 60.0);
    }
    let rgba = capture(&mut sim, &device, &queue, 160, 120).await;
    let lit = rgba
        .chunks_exact(4)
        .filter(|p| p[0].max(p[1]).max(p[2]) > 8)
        .count();
    assert!(
        lit > 160 * 120 / 20,
        "only {lit} pixels of 19200 have any dye in them after twenty frames"
    );
}
