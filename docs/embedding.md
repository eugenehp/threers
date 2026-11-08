# Embedding threers in a desktop app

Part of the [threers](../README.md) documentation.

A Tauri, Wry, tao or winit app that shows native GPU viewports beside its own
UI wants three things from a renderer: a viewport on a window it already owns,
no stall while pipelines compile, and no paying for that compilation again on
every launch. This page covers the three pieces that do it:

| Piece | What it does |
|---|---|
| [`SurfaceGpu`, `SurfaceRenderer`](#viewports-on-your-windows) | A renderer on any window with raw-window-handle 0.6 handles; several on one device; drawn from any thread |
| [`Renderer::new_lazy`, `warm_up`](#lazy-pipelines-and-warm-up) | Construction in milliseconds; pipelines built on worker threads before the viewport is shown |
| [`PipelineCache`](#a-pipeline-cache-that-outlives-the-process) | Compiled pipelines kept between runs — in a directory, a file, memory, or the app's own storage |

All three are native-only. Apple apps drawing into their own
`CAMetalLayer` have [`LayerGpu`/`LayerRenderer`](apple-platforms.md), which this
follows.

## Viewports on your windows

```rust
use threers::renderer::{SurfaceGpu, SurfaceOptions, SurfaceRenderer};

// Once per app: one device, shared by every viewport.
let gpu = SurfaceGpu::new()?.with_pipeline_cache(app_cache_dir.join("gpu"));

// Per viewport, on the main thread (macOS requires it for the surface).
// `window`: anything with raw-window-handle 0.6 window and display handles
// that clones — a `tauri::Window`, an `Arc` of a tao or winit window.
let mut view = SurfaceRenderer::new(&gpu, window.clone(), width, height, SurfaceOptions::default().lazy(true))?;

// The renderer is `Send`: draw it on a thread of its own.
std::thread::spawn(move || loop {
    view.render(&mut scene, &camera);
});
```

- **Raw handles.** `SurfaceRenderer::from_raw` takes a `RawDisplayHandle` and
  `RawWindowHandle` directly, for hosts that do not expose a window type.
- **One device.** Viewports on one `SurfaceGpu` share meshes, shaders and
  pipelines: the second viewport compiles nothing the first already did.
  Viewports draw in parallel; one resizing pauses the others' submissions for
  the moment wgpu needs the queue idle.
- **`render` returns `false`** when there was no image to draw into
  (minimised, occluded, every image in flight). Nothing is lost; call it
  again next frame. Resize with `view.resize(w, h)` in physical pixels.
- **Options.** `SurfaceOptions` sets MSAA (4 by default), lazy pipelines,
  the present mode (`AutoVsync` by default, so a render thread never spins)
  and `transparent` (premultiplied alpha where the platform offers it).
- **Adapter.** `SurfaceGpu::new` opens the high-performance adapter without a
  surface, which is right for Vulkan, Metal and DX12. For GL on Wayland, or
  to pick the adapter yourself, open a `GpuContext` and use
  `SurfaceGpu::from_context`.

The [`surface_viewport`](../examples/surface_viewport.rs) example puts it
together:

```bash
cargo run --release --example surface_viewport
```

## Lazy pipelines and warm-up

`Renderer::new` builds the renderer's whole pipeline catalogue in the
constructor. `Renderer::new_lazy` (or `SurfaceOptions::lazy`, or
`RendererOptions::lazy`) defers each pipeline to its first use, so
construction takes milliseconds. Left there, the cost lands on the first
frames instead. `warm_up` moves it wherever the app wants:

```rust
let warmup = view.renderer().warm_up();      // every pipeline not built yet
for _ in 0..4 {
    let w = warmup.clone();
    std::thread::spawn(move || w.run());     // threads share the work
}
// Show a loading state; `warmup.progress()` is (built, total) for a bar.
warmup.wait();
assert!(warmup.errors().is_empty());
view.renderer().warm_scene(&mut scene, &camera);  // the scene's own variants
```

- `run` builds pipelines until none are left, from as many threads as call
  it; `step` builds one, for spreading the work over idle frames.
  `spawn(n)` starts `n` detached threads and returns at once.
- `on_progress(|built, total| …)` pushes progress instead of making you poll:
  it is called after each pipeline, from the thread that built it (keep it
  short), and once on registering with the current count. Use it to send a
  Tauri event or a channel message.
- `SurfaceRenderer::warm_in_background(threads, &states)` does the spawn for
  you and makes `render` return `false` until the job is done, so a viewport
  cannot show a frame that stalls on compilation. `is_warming()` says whether
  it is still holding; `stop_waiting_for_warm_up()` lets it draw anyway.
- `is_ready`, `progress` and `wait` say when the viewport can be shown without
  a stall. `errors` lists pipelines that failed validation, by label.
- `warm_up_matching(|label| …)` builds only the pipelines whose labels it
  keeps (`pending_pipeline_labels` lists them) — the paths the app draws with.
- `warm_up_ranked(|label| rank)` builds the lowest-ranked pipelines first, for
  a viewport that starts drawing before the job finishes.
- `warm_up_states(&[RenderState])` queues the variants those states ask for
  (depth test and write, colour write, additive blending) alongside the stock
  pipelines. It needs no scene and works on the web.
- `warm_scene` draws a scene once into a throwaway target (native). That finds
  the variants a scene asks for without your listing the states.
- In a browser `step` and `run` cannot hear validation's verdict, so `errors`
  stays empty there; `step_async` and `run_async` await it and report on every
  target. `pipeline_failures()` keeps what a warm-up reported.
- A pipeline being warmed and drawn with at the same moment is built once;
  whichever gets there second waits.

`pending_pipelines()` is 0 for an eager renderer and once a warm-up has run.

### Pacing `render`

`SurfaceRenderer::render` returns `false` when there was no image to draw
into: the window is minimised or occluded, every image is in flight, the
surface was just remade, or a background warm-up is still running. Nothing is
lost, but a loop that calls it again at once spins a core, and a minimised
window is exactly when an app is idle. Sleep a frame (the example uses 8 ms)
or wait for the window's next redraw request before trying again.

### In a webview page

A Tauri frontend, or any page, can use the wasm renderer instead of a native
surface. `WebRenderer.newLazy(canvas)` returns without building pipelines, and
`await renderer.warmUp().run()` builds them while the page shows a loading
state, yielding to the browser between pipelines. `run` resolves with the
count and rejects with the validation errors, one per line. The blocking
`step` cannot hear validation's verdict in a browser; `run` uses the awaiting
`step_async`, which can. `built` and `total` on the job give a progress bar.

[`crates/threers-ecs/examples/tauri-race`](../crates/threers-ecs/examples/tauri-race/src/main.rs)
is this route as a whole app, compiled and run: the race game's browser build
inside a Tauri 2 window, at 60 fps on WebGPU in the macOS webview. `--probe`
starts it, measures the frame rate and samples the canvas, and exits.

## In a Tauri app

The pieces above, put together for Tauri 2. `tauri::WebviewWindow` and
`tauri::Window` implement the raw-window-handle 0.6 traits and clone, which is
all `SurfaceRenderer::new` asks of a window. (This recipe was written against
the Tauri 2.12 source and has not been compiled into a Tauri app here; the
threers calls in it are the ones the tests and `surface_viewport` example
exercise.)

```rust
use tauri::{Emitter, Manager, WindowEvent};
use threers::renderer::{SurfaceGpu, SurfaceOptions, SurfaceRenderer};

#[derive(Clone, serde::Serialize)]
struct Warming { built: usize, total: usize }

fn open_viewport(app: &tauri::AppHandle, window: tauri::Window) -> Result<(), String> {
    let cache_dir = app.path().app_cache_dir().map_err(|e| e.to_string())?.join("gpu");
    let gpu = std::sync::Arc::new(SurfaceGpu::new()?.with_pipeline_cache(cache_dir));
    let size = window.inner_size().map_err(|e| e.to_string())?;

    // macOS needs the surface made on the main thread; the renderer then moves.
    let (tx, rx) = std::sync::mpsc::channel();
    let (gpu2, window2) = (gpu.clone(), window.clone());
    app.run_on_main_thread(move || {
        let view = SurfaceRenderer::new(&gpu2, window2, size.width, size.height,
                                        SurfaceOptions::default().lazy(true));
        let _ = tx.send(view);
    }).map_err(|e| e.to_string())?;
    let mut view = rx.recv().map_err(|e| e.to_string())??;

    // Build pipelines in the background; tell the frontend how far along.
    let warmup = view.warm_in_background(4, &[/* RenderStates your materials use */]);
    let handle = app.clone();
    warmup.on_progress(move |built, total| {
        let _ = handle.emit("gpu-warmup", Warming { built, total });
    });

    // Resizes arrive on the event loop; the render thread picks them up.
    let size_now = std::sync::Arc::new(std::sync::Mutex::new(None::<(u32, u32)>));
    let closing = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    {
        let (size_now, closing, gpu) = (size_now.clone(), closing.clone(), gpu.clone());
        window.on_window_event(move |event| match event {
            WindowEvent::Resized(s) => *size_now.lock().unwrap() = Some((s.width, s.height)),
            WindowEvent::CloseRequested { .. } | WindowEvent::Destroyed => {
                closing.store(true, std::sync::atomic::Ordering::Relaxed);
                let _ = gpu.save_pipeline_cache(); // the next launch skips compilation
            }
            _ => {}
        });
    }

    std::thread::spawn(move || {
        let (mut scene, camera) = build_scene();
        while !closing.load(std::sync::atomic::Ordering::Relaxed) {
            if let Some((w, h)) = size_now.lock().unwrap().take() {
                view.resize(w, h);
            }
            if !view.render(&mut scene, &camera) {
                // Minimised, occluded or still warming: do not spin.
                std::thread::sleep(std::time::Duration::from_millis(8));
            }
        }
    });
    Ok(())
}
```

On the frontend, `listen("gpu-warmup", e => bar.value = e.payload.built / e.payload.total)`
drives a loading bar, and the viewport appears when the count reaches the total.

Two things to decide for your layout. A `SurfaceRenderer` draws into the
window you give it, and a webview in that same window paints over it, so a
viewport beside the UI wants its own `tauri::Window` (or a child view through
`SurfaceRenderer::from_raw`) positioned next to the webview. And an app that
would rather render in the webview itself can use the wasm renderer: see
[In a webview page](#in-a-webview-page).

## A pipeline cache that outlives the process

A warm-up moves compilation; a pipeline cache skips it. The driver's compiled
pipelines are saved after a run and handed back on the next.

```rust
use threers::renderer::{PipelineCache, RendererOptions};

let info = gpu.adapter.get_info();
let cache = PipelineCache::in_dir(&gpu.device, &info, cache_dir);
let options = RendererOptions::default().lazy(true).pipeline_cache(&cache);
let renderer = Renderer::with_options(device, queue, format, w, h, options);
renderer.warm_up().run();
if let Some(cache) = &cache {
    cache.save()?;
}
```

Where the bytes go is a `PipelineCacheStore`:

| Store | Use |
|---|---|
| a `PathBuf` | A directory; one file per adapter and driver |
| `FileStore(path)` | One exact file |
| `MemoryStore` | In memory; clones share entries, `get`/`set` read and seed the bytes |
| `CallbackStore::new(load, save)` | Two closures — a settings database, a Tauri store plugin, the app's own save file |
| your own type | Implement `load(key)` and `save(key, data)` |

`PipelineCache::in_memory` and `PipelineCache::from_bytes` cover the
common in-memory cases; `cache.data()` returns the bytes to keep however the
app likes. `SurfaceGpu::with_pipeline_cache` takes any store, and
`SurfaceGpu::save_pipeline_cache` writes it.

Measured on an RTX 3080 Ti laptop GPU (Vulkan, driver 595.99, release build),
building the 97 pipelines of a lazy renderer on one thread:

| Driver's own shader cache | No threers cache | From a saved threers cache |
|---|---|---|
| empty (`__GL_SHADER_DISK_CACHE=0`) | 20.4 s | 0.28 s |
| warm | 0.48 s | 0.29 s |

The driver's cache already covers the launches where it kept everything. The
threers cache covers the rest: a first install, after a driver update, and
after the driver evicts — NVIDIA caps its cache at 512 MB by default. The same
97 pipelines on four warm-up threads took 9.0 s with both caches empty.

- **Vulkan only.** wgpu supports pipeline caches through
  `Features::PIPELINE_CACHE`, which `GpuContext` requests wherever the adapter
  has it. On Metal, DX12 and GL, `PipelineCache::open` returns `None`, and a
  renderer given `None` behaves exactly as one without — the same code runs
  everywhere.
- **Safe to feed anything.** Entries are keyed by adapter and driver
  (`wgpu::util::pipeline_cache_key`). Data from another GPU, an older driver
  or plain junk is checked and dropped by wgpu, and the pipelines are built
  as if the cache were empty.
- **Writes are atomic** for the directory and file stores: a temporary file
  and a rename, so a crash mid-write leaves the previous cache.

## Tests

```bash
cargo test --test pipeline_warmup -- --nocapture   # warm-up, stores, pixel parity
cargo test --test lazy_initialization -- --nocapture
cargo test --lib renderer::pipeline_cache
```

The cache tests skip themselves off Vulkan. On a Vulkan machine they run each
store cold and from what it saved, and compare pixels with an eager renderer.
