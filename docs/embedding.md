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
- `is_ready`, `progress` and `wait` say when the viewport can be shown without
  a stall. `errors` lists pipelines that failed validation, by label.
- `warm_up_matching(|label| …)` builds only the pipelines whose labels it
  keeps (`pending_pipeline_labels` lists them) — the paths the app draws with.
- `warm_scene` draws a scene once into a throwaway target. That builds the
  variants a material's state asks for at draw time (depth test and write,
  colour write, culling), which no catalogue warm-up can know about.
- A pipeline being warmed and drawn with at the same moment is built once;
  whichever gets there second waits.

`pending_pipelines()` is 0 for an eager renderer and once a warm-up has run.

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
