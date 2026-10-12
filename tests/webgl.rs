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
use wasm_bindgen::JsCast;
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

/// The renderer draws on WebGL2 — lit, skinned, and refracting — and the pixels
/// prove each one.
///
/// All three share a renderer deliberately. A browser caps how many live WebGL2
/// contexts a page may hold and does not release one when its canvas goes out of
/// scope, so a test per draw exhausted the pool and whichever ran last hung
/// waiting for a context that was never coming. Each draw still gets its own
/// render target, which costs nothing.
///
/// The assertions are on read-back pixels rather than on the absence of a panic:
/// a draw that binds nothing and writes nothing raises no error at all.
#[wasm_bindgen_test]
async fn the_renderer_draws_on_webgl2() {
    let mut renderer = webgl_renderer().await;
    assert_eq!(
        renderer.backend(),
        "gl",
        "expected the WebGL2 backend in a browser with no WebGPU"
    );

    let white = threers::wasm::WebColor::new(1.0, 1.0, 1.0);
    let sun = || {
        let mut l = threers::wasm::WebLight::directional(&white, 2.5);
        l.set_direction(-0.4, -0.7, -0.6);
        l
    };
    let camera = || {
        let mut c = threers::wasm::WebCamera::perspective(60.0, 1.0, 0.1, 100.0);
        c.set_position(0.0, 0.0, 4.0);
        c
    };

    // --- A lit standard material. ---
    //
    // Getting here needed the texture budget brought under WebGL2's sixteen
    // units. The symptom was not a validation error but a panic inside the
    // backend, `index out of bounds: the len is 16 but the index is 16`, on a
    // draw that sampled none of the textures that overflowed it.
    {
        let target = threers::wasm::WebRenderTarget::new(&renderer, 64, 64);
        let mut scene = threers::wasm::WebScene::new();
        let colour = threers::wasm::WebColor::new(0.9, 0.3, 0.1);
        let material = threers::wasm::WebMaterial::standard(&colour, 0.4, 0.0);
        let geometry = threers::wasm::WebGeometry::box_(1.5, 1.5, 1.5);
        scene.add(&threers::wasm::WebMesh::new(&geometry, &material));
        scene.add_light(&sun());
        scene.add_light(&threers::wasm::WebLight::ambient(&white, 0.4));

        renderer.set_render_target(target.id);
        renderer.render(&mut scene, &camera());
        renderer.render(&mut scene, &camera());
        let pixels = read_target(&renderer, target.id, 64, 64).await;
        assert_eq!(pixels.len(), 64 * 64 * 4, "short read-back");
        let lit = pixels.chunks(4).filter(|p| p[0] > 40 && p[0] > p[2]).count();
        assert!(
            lit > 50,
            "only {lit} of {} pixels show the lit cube",
            pixels.len() / 4
        );
    }

    // --- A skinned material. ---
    //
    // Skinning used to put its bone matrices in a storage buffer, which WebGL2
    // does not have — and one `var<storage>` declaration stops the whole module
    // compiling on a GLSL backend whether or not anything skinned is drawn. The
    // bones are identities here; the point is that the skinned pipeline binds
    // and runs, not that it deforms.
    {
        let target = threers::wasm::WebRenderTarget::new(&renderer, 64, 64);
        let mut scene = threers::wasm::WebScene::new();
        let colour = threers::wasm::WebColor::new(0.2, 0.8, 0.3);
        let material = threers::wasm::WebMaterial::standard(&colour, 0.4, 0.0);
        let geometry = threers::wasm::WebGeometry::box_(1.5, 1.5, 1.5);
        scene.add_skinned_mesh(&geometry, &material, 4);
        scene.add_light(&sun());
        scene.add_light(&threers::wasm::WebLight::ambient(&white, 0.4));

        renderer.set_render_target(target.id);
        renderer.render(&mut scene, &camera());
        renderer.render(&mut scene, &camera());
        let pixels = read_target(&renderer, target.id, 64, 64).await;
        let lit = pixels.chunks(4).filter(|p| p[1] > 40 && p[1] > p[0]).count();
        assert!(
            lit > 50,
            "only {lit} of {} pixels show the skinned cube",
            pixels.len() / 4
        );
    }

    // --- Screen-space refraction. ---
    //
    // The pass that could not be built here until the bind groups were split.
    // wgpu-hal's GL backend assigns a texture unit per *layout* entry rather
    // than per use, so one shared group 3 carrying both the shadow maps and the
    // screen-space capture came to seventeen against sixteen. The refraction
    // pass samples no shadow map, so it now has a group 3 with the capture where
    // those were, and both layouts come to fourteen.
    //
    // An opaque plane stands behind the glass so there is something to refract.
    {
        let target = threers::wasm::WebRenderTarget::new(&renderer, 64, 64);
        let mut scene = threers::wasm::WebScene::new();
        let backdrop_colour = threers::wasm::WebColor::new(0.1, 0.2, 0.9);
        let backdrop = threers::wasm::WebMaterial::standard(&backdrop_colour, 0.5, 0.0);
        let plane = threers::wasm::WebGeometry::box_(6.0, 6.0, 0.2);
        let back = scene.add(&threers::wasm::WebMesh::new(&plane, &backdrop));
        scene.set_transform(
            &back,
            &threers::wasm::WebVector3::new(0.0, 0.0, -2.0),
            &threers::wasm::WebEuler::new(0.0, 0.0, 0.0),
        );

        let glass_colour = threers::wasm::WebColor::new(1.0, 1.0, 1.0);
        let mut glass =
            threers::wasm::WebMaterial::physical(&glass_colour, 0.05, 0.0, 0.0, 0.0);
        glass.set_transmission(1.0, 1.5, 0.5, 0.0);
        let sphere = threers::wasm::WebGeometry::sphere(1.0, 24, 16);
        scene.add(&threers::wasm::WebMesh::new(&sphere, &glass));
        scene.add_light(&sun());
        scene.add_light(&threers::wasm::WebLight::ambient(&white, 0.5));

        renderer.set_render_target(target.id);
        renderer.render(&mut scene, &camera());
        renderer.render(&mut scene, &camera());
        let pixels = read_target(&renderer, target.id, 64, 64).await;
        let blue = pixels.chunks(4).filter(|p| p[2] > 60 && p[2] > p[0]).count();
        assert!(
            blue > 500,
            "only {blue} of {} pixels are backdrop-blue — nothing drew",
            pixels.len() / 4
        );
        // A pass that failed to run would leave a flat backdrop behind an unlit
        // sphere; what refraction adds is variation *through* the glass.
        let distinct: std::collections::BTreeSet<(u8, u8, u8)> =
            pixels.chunks(4).map(|p| (p[0], p[1], p[2])).collect();
        assert!(
            distinct.len() > 20,
            "only {} distinct colours — the frame is a flat fill, so the glass \
             refracted nothing",
            distinct.len()
        );
    }
}

/// A scene with a texture, a shadow and post-processing draws on WebGL2.
///
/// The draws above are deliberately plain, which means they exercise one
/// pipeline each. This one is the opposite: a textured material, a shadow-casting
/// light (so the shadow pipelines and the shadow map are built and sampled), a
/// transparent surface (so the sorted path runs), and a post-processing pass
/// over the result. It is here to find what the narrow tests cannot — anything
/// still assuming a capability WebGL2 lacks shows up as a validation error or a
/// blank frame rather than as a wrong colour.
#[wasm_bindgen_test]
async fn a_full_scene_draws_on_webgl2() {
    let mut renderer = webgl_renderer().await;
    let scene_rt = threers::wasm::WebRenderTarget::new(&renderer, 64, 64);
    let post_rt = threers::wasm::WebRenderTarget::new(&renderer, 64, 64);

    let mut scene = threers::wasm::WebScene::new();

    // A checked texture, so a sampler that reads the wrong thing is visible.
    let mut data = vec![0u8; 8 * 8 * 4];
    for y in 0..8usize {
        for x in 0..8usize {
            let on = (x / 2 + y / 2) % 2 == 0;
            let i = (y * 8 + x) * 4;
            data[i] = if on { 230 } else { 30 };
            data[i + 1] = if on { 120 } else { 20 };
            data[i + 2] = 40;
            data[i + 3] = 255;
        }
    }
    let texture = threers::wasm::WebDataTexture::new(8, 8, data);
    let white = threers::wasm::WebColor::new(1.0, 1.0, 1.0);
    let mut textured = threers::wasm::WebMaterial::standard(&white, 0.5, 0.0);
    textured.set_map_data(&texture);

    // A floor to receive the shadow, and a box above it to cast one.
    let floor = scene.add(&threers::wasm::WebMesh::new(
        &threers::wasm::WebGeometry::box_(6.0, 0.2, 6.0),
        &textured,
    ));
    scene.set_transform(
        &floor,
        &threers::wasm::WebVector3::new(0.0, -1.5, 0.0),
        &threers::wasm::WebEuler::new(0.0, 0.0, 0.0),
    );
    let caster_colour = threers::wasm::WebColor::new(0.3, 0.6, 0.9);
    let caster = scene.add(&threers::wasm::WebMesh::new(
        &threers::wasm::WebGeometry::box_(1.2, 1.2, 1.2),
        &threers::wasm::WebMaterial::standard(&caster_colour, 0.3, 0.0),
    ));
    scene.set_object_cast_shadow(&caster, true);

    // And a transparent pane in front, which takes the sorted path.
    let glass_colour = threers::wasm::WebColor::new(0.9, 0.9, 1.0);
    let mut pane = threers::wasm::WebMaterial::standard(&glass_colour, 0.1, 0.0);
    pane.set_opacity(0.4);
    let front = scene.add(&threers::wasm::WebMesh::new(
        &threers::wasm::WebGeometry::box_(2.0, 2.0, 0.05),
        &pane,
    ));
    scene.set_transform(
        &front,
        &threers::wasm::WebVector3::new(0.0, 0.0, 1.2),
        &threers::wasm::WebEuler::new(0.0, 0.0, 0.0),
    );

    let mut sun = threers::wasm::WebLight::directional(&white, 2.2);
    sun.set_direction(-0.3, -0.9, -0.4);
    scene.add_light(&sun);
    scene.add_light(&threers::wasm::WebLight::ambient(&white, 0.35));

    let mut camera = threers::wasm::WebCamera::perspective(55.0, 1.0, 0.1, 100.0);
    camera.set_position(0.0, 1.5, 5.0);

    renderer.set_render_target(scene_rt.id);
    renderer.render(&mut scene, &camera);
    renderer.render(&mut scene, &camera);

    let before = read_target(&renderer, scene_rt.id, 64, 64).await;
    let distinct_before: std::collections::BTreeSet<(u8, u8, u8)> =
        before.chunks(4).map(|p| (p[0], p[1], p[2])).collect();
    assert!(
        distinct_before.len() > 30,
        "the scene came back with only {} distinct colours — a textured, lit, \
         shadowed frame cannot be that flat",
        distinct_before.len()
    );

    // Post-processing over the rendered frame. Kind 0 is the plain copy, which
    // is enough to prove the pass compiles and runs on this backend; the depth
    // and normal inputs reuse the scene target because this is about whether the
    // pipeline runs, not about what a particular effect looks like.
    let identity: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    renderer.apply_post_fx_to_rt(
        scene_rt.id,
        post_rt.id,
        scene_rt.id,
        scene_rt.id,
        0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.1,
        100.0,
        0.5,
        16,
        identity.clone(),
        identity,
    );
    let after = read_target(&renderer, post_rt.id, 64, 64).await;
    assert_eq!(after.len(), before.len());
    let lit_after = after.chunks(4).filter(|p| p[0] > 8 || p[1] > 8 || p[2] > 8).count();
    assert!(
        lit_after > 500,
        "only {lit_after} of {} pixels survived post-processing — the pass ran \
         but produced nothing",
        after.len() / 4
    );
}

/// Pull a render target back to the CPU, awaiting the promise it returns.
async fn read_target(
    renderer: &threers::wasm::WebRenderer,
    target_id: u32,
    w: u32,
    h: u32,
) -> Vec<u8> {
    let promise = renderer.read_render_target_pixels(target_id, 0, 0, w, h);
    let value = wasm_bindgen_futures::JsFuture::from(promise)
        .await
        .expect("read-back rejected");
    js_sys::Uint8Array::new(&value).to_vec()
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
    threers::wasm::WebRenderer::new(canvas)
        .await
        .expect("the renderer would not start on this browser")
}

/// A custom shader, through the JavaScript API, in a browser.
///
/// `ShaderMaterial` had no counterpart in the wasm bindings at all, so every
/// custom shader in the crate was reachable from Rust and nowhere else — and
/// the shim's `THREE.ShaderMaterial` was a stub that quietly built a *basic*
/// material and rendered a flat colour, which is worse than the class being
/// absent.
///
/// The storage array is the part worth checking here rather than on a desktop:
/// WebGL2 has no storage buffers, so `s0(i)` has to resolve to a texture read.
/// A shader written with `u_s0[i]` does not compile on this backend at all,
/// which is exactly the failure this is meant to catch before a user does.
#[wasm_bindgen_test]
async fn a_shader_material_draws_on_webgl2() {
    let mut renderer = webgl_renderer().await;
    let target = threers::wasm::WebRenderTarget::new(&renderer, 64, 64);

    // Colour comes from the uniform block, brightness from the storage slot —
    // so a frame that renders but binds either wrongly is the wrong colour
    // rather than no pixels.
    let mut material = threers::wasm::WebShaderMaterial::new(
        "@fragment\n\
         fn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n\
         let tint = u_data.data[0].rgb;\n\
         let gain = s0(3u);\n\
         return vec4<f32>(framebuffer_encode(tint * gain), 1.0);\n\
         }\n",
    );
    material.set_data(vec![0.0, 0.8, 0.2, 1.0]);
    material.set_storage(0, vec![0.0, 0.0, 0.0, 1.0]);

    let mut scene = threers::wasm::WebScene::new();
    scene.add(&threers::wasm::WebMesh::new(
        &threers::wasm::WebGeometry::box_(2.0, 2.0, 2.0),
        &material.material(),
    ));
    let mut camera = threers::wasm::WebCamera::perspective(55.0, 1.0, 0.1, 100.0);
    camera.set_position(0.0, 0.0, 4.0);

    renderer.set_render_target(target.id);
    renderer.render(&mut scene, &camera);
    renderer.render(&mut scene, &camera);
    let px = read_target(&renderer, target.id, 64, 64).await;

    let centre = ((32 * 64 + 32) * 4) as usize;
    let (r, g, b) = (px[centre], px[centre + 1], px[centre + 2]);
    assert!(
        g > 120 && g > r + 40 && g > b + 40,
        "expected the green the uniform asked for, got ({r}, {g}, {b}) — the \
         shader material did not reach the backend"
    );
}
