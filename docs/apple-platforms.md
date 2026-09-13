# Metal and visionOS

Part of the [threers](../README.md) documentation.

# Metal backend (Apple)

`--features metal`, macOS and iOS. A second renderer for the same `Scene`, written
against the Objective-C runtime directly: `objc_msgSend` declared and transmuted
per call site in `src/metal/objc.rs`, frameworks pulled in with
`#[link(kind = "framework")]`. **No `metal-rs`, no `objc` crate, no build script,
and no new dependency** — the feature adds a directory, not a line to `Cargo.lock`.
On any other platform the feature compiles to nothing.

```bash
cargo run --release --features metal --example metal_headless          # → out/metal_headless.png
cargo run --release --features metal --example metal_stereo            # → out/metal_stereo.png (both eyes)
cargo run --release --features metal --example metal_window            # winit + CAMetalLayer
cargo run --release --features metal --example metal_window -- --frames 120
cargo test --features metal --test metal_backend                       # 24 tests, on a real GPU
```

```rust
// Cargo.toml: threers = { version = "0.0.5", features = ["metal"] }
use threers::metal::MetalHeadlessRenderer;

let mut renderer = MetalHeadlessRenderer::builder().size(1280, 720).msaa(4).build()?;
let rgba = renderer.render_to_rgba(&mut scene, &camera)?;   // tightly packed RGBA8
```

On screen, given an `NSView` from any window toolkit:

```rust
use threers::metal::{MetalDevice, MetalRenderer, MetalSurface};

let device = MetalDevice::new()?;
let mut renderer = MetalRenderer::with_device(device.clone())?;
let surface = unsafe { MetalSurface::from_ns_view(&device, ns_view, 1280, 720, 4)? };

if let Some(frame) = surface.next_frame() {
    renderer.render(&mut scene, &camera, &frame.attachments())?;   // draws and presents
}
```

Or into textures you already own — an AVFoundation pipeline, an `MTKView`, a
Core Image chain — with `PassAttachments::from_raw`.

**Covers:** meshes, instanced meshes, line segments, points and sprites; `Basic`,
`Lambert`, `Phong`, `Standard`, `Physical`, `Normal`, `Depth`, `Toon`, `Matcap`,
`Line`, `Points` and `Sprite` materials; one base-colour map with the sampler,
wrap modes and UV transform from the `Texture`; ambient, directional, point, spot
and hemisphere lights; linear and exp² fog; alpha blend, alpha test, vertex
colours, wireframe, MSAA, and depth-sorted transparency. Geometry, textures,
samplers and pipeline states are cached across frames.

**Does not cover** — use the wgpu `Renderer` for these: shadow maps,
post-processing, environment maps and IBL (so a metal-heavy material has only its
specular highlights), skinning and morph targets, and the normal / roughness /
metalness / AO / emissive map slots. Materials outside the list draw unlit in
their base colour rather than failing, so a scene authored for the wgpu path
still renders. Both backends share the same conventions — 0..1 clip depth, CCW
front faces, `flip_y` applied at upload, three.js punctual attenuation — so the
two agree on a scene as far as this one goes.

# visionOS

`--features visionos` (which implies `metal`). An immersive visionOS app owns no
layer and no swap chain: SwiftUI hands it a `cp_layer_renderer_t` and it pulls
frames. That API is C rather than Objective-C, so this is `extern "C"`
declarations transcribed from the XROS SDK headers — still no new dependency, no
build script, no `metal-rs`.

```rust
// Cargo.toml: threers = { version = "0.0.5", features = ["visionos"] }
use threers::metal::visionos::ImmersiveRenderer;

#[no_mangle]
pub extern "C" fn threers_visionos_run(layer_renderer: *mut std::ffi::c_void) {
    let mut renderer = unsafe { ImmersiveRenderer::new(layer_renderer) }.unwrap();
    let mut scene = build_scene();
    while renderer.render_frame(&mut scene).unwrap() {}
}
```

```swift
// The Swift side is one call.
ImmersiveSpace(id: "scene") {
    CompositorLayer(configuration: MyConfiguration()) { layerRenderer in
        threers_visionos_run(Unmanaged.passUnretained(layerRenderer).toOpaque())
    }
}
```

What it does per frame: waits for the compositor's optimal input time, asks
ARKit for the device anchor at the predicted presentation time, takes the
drawable's textures and per-eye transforms, computes each eye's projection with
`cp_drawable_compute_projection`, draws, and presents on the same queue.

**Both eyes in one pass.** With the compositor's `layered` layout the eyes are
two slices of one texture array, and `render_views` draws them in a single pass —
the vertex stage reads its eye from the instance id and writes
`render_target_array_index`, so the second eye costs pixels, not a second walk
of the scene graph. The `dedicated` and `shared` layouts work too, one pass per
eye. **Reverse-Z** throughout, because the compositor accepts nothing else
(`drawable.h`: *"It only supports reverse-Z depth"*).

Both of those work on any Mac, which is where they are tested — `cargo test
--features metal` renders a stereo pair into a two-slice array offscreen and
checks each eye's parallax, and renders a reverse-Z scene and checks the depth
test inverted with it:

```bash
cargo run --release --features metal --example metal_stereo   # → out/metal_stereo.png, side by side
```

> **Building for the device.** The `visionos` code is written against the XROS
> SDK and verified against it (every `cp_*` and `ar_*` symbol checked against the
> SDK stubs, and the whole frame loop compiled and linked against the same two
> frameworks on macOS 26). Building the *whole crate* for
> `aarch64-apple-visionos` does not work yet, and the reason is upstream: wgpu
> 0.20 predates visionOS, so its `cfg(all(unix, not(ios), not(macos)))` routes
> the target into the Vulkan backend, which pulls `ash` → `libloading 0.7`,
> which has no `RTLD_*` constants for it. It clears when the crate moves to a
> wgpu that knows the target, or when `wgpu` becomes an optional dependency —
> nothing in `src/metal` needs it.
