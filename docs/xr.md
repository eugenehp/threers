# XR: headsets

Part of the [threers](../README.md) documentation.

The `xr` feature puts threers in a headset: a 3D scene, a flat UI, or
both. The same app code runs on every platform. Only the layer that talks
to the device changes, and the browser's (WebXR) is the one there today.

## The layers

| Layer | What it is | Where |
|---|---|---|
| Platform | Opens a session and runs its frame loop. Each frame it reports the head, the eyes and the inputs, then shows what was drawn. | `xr::web::XrSession` (WebXR, wasm); `xr::visionos::ImmersiveSession` (Apple Vision Pro, feature `xr-visionos`) |
| `XrSource` | The trait every platform implements: `head`, `eyes`, `framebuffer_size`, `inputs`, `take_events`, `see_through`. | `threers::xr` |
| Scene | `Renderer::render_xr` draws any `Scene` once per eye, through the app camera's lens. A rig matrix places the viewer in the world. | `threers::xr` (wgpu) |
| UI | `Panel` places a flat UI and turns rays into what a screen gives a UI: mouse moves and presses, wheel turns, touches. `PanelPass` draws it alone in a room, or inside a scene (`draw_in_scene`), depth-sorted with it. | `threers::xr` (wgpu) |

Nothing above the platform layer knows which headset or browser it runs
on: the UI demo's headset code is the same in the browser and in the
native visionOS app. Another native runtime (OpenXR on Windows, Linux and
Android) would implement `XrSource` and present the same way.

## Depth

`PanelPass::draw_in_scene` draws the panel and the rays into each eye
right after the scene, depth-tested against that eye's depth. Something
in front of the panel hides it, and the panel hides what is behind it.
The renderer side is `Renderer::draw_over_last`, which works for anything
drawn over a render: it opens the render's colour and depth wherever they
went (straight into the target, the shared multisampled targets, or a
multisampled pane of their own) and says how to match its projection
(`OverlayInfo`).

## Input

- **Controllers** (Meta Quest, most PC headsets) work like a mouse. The
  ray is the pointer, and hovering works. The trigger clicks, the
  thumbstick scrolls, and squeezing brings the panel back in front of
  you.
- **Look and pinch** (Apple Vision Pro, where Safari's `immersive-vr`
  reports a `transient-pointer`) and **a tap on a phone in AR** work
  like a finger. The input exists only while it presses, so a pinch is
  a touch, and pinching and dragging scrolls and swipes. No ray is drawn,
  as the system draws none.

- **Hands** (WebXR's hand tracking, ARKit's on Vision Pro) touch the panel
  directly. Within 5 cm of it, a dot shows where the index fingertip
  would touch. Crossing the surface is a touch, sliding along it drags,
  and pulling back 1.5 cm lifts it. Only a fingertip that arrives from
  in front starts a touch.

Each input keeps its slot (three.js's `getController(i)`) for as long as
it stays connected. Events come as `XrEvent`s, with WebXR's names
available through `XrEventKind::name`.

## On the web

```rust
use threers::xr::web::XrSession;
use threers::xr::{Panel, PanelPass, XrSource};

// From a click (browsers want a user gesture):
let session = XrSession::request("immersive-vr", canvas, "local-floor", 1.0, &[]).await?;
session.set_frame_callback(Some(Box::new(move |_time, _frame| {
    let (w, h) = session.fit_canvas();         // the canvas, sized to the headset
    // The scene per eye, the panel standing in it, depth-sorted:
    // panel_pass.draw_in_scene(&mut renderer, &mut scene, &camera, &session, &rig, &view, &panel, &pointers);
    // (or renderer.render_xr(...) for a scene alone)
    session.mark_drawn();                       // copy it to the headset
})));
```

The renderer draws on its own canvas, with WebGPU or WebGL2. A headset
takes frames from a WebGL layer, so the session keeps a small WebGL2
context and copies the canvas into the layer in one draw after each
frame. The eye rectangles already match, so one copy covers both eyes.

### AR, the session's state, and the device

- **Hit tests** (`immersive-ar` asks for `hit-test` and `anchors` itself):
  `session.start_hit_test()` and each frame's `session.hits()` give where
  the middle of the view meets a real surface, nearest first, as poses
  in the reference space: where a reticle or a model goes.
  `start_tap_hit_test("generic-touchscreen")` and `tap_hits()` do the same
  for taps on a phone's screen and for pinches, each with its input's slot.
- **Anchors:** `add_anchor(pose)` (inside a frame; a hit's pose, say) keeps
  a place in the room as the device's map of it improves; `anchors()` gives
  each one's pose every frame, `remove_anchor(id)` lets it go.
- **An overlay:** `XrSession::request_with_overlay(…, Some(&element))` asks
  for WebXR's DOM overlay, so a phone's AR shows that HTML (buttons, text)
  over the camera.
- **A buzz:** `pulse(slot, intensity, ms)` vibrates a controller (its
  `hapticActuators`, or the gamepad's `vibrationActuator`).
- **Visibility:** `visibility()` is `"visible"`, `"visible-blurred"` (a
  system menu over the app: shown, but no input) or `"hidden"`.
- **Recentring:** `take_recentered()` says once that the user recentred, so
  what was placed in front of them can be put there again.
- **Frame rate and foveation:** `frame_rates()`, `set_frame_rate(hz)` and
  `set_foveation(0..1)`, where the device offers them.

`tests/webxr_web.rs` runs all of it in headless Chrome, on a headset the
page fakes (`FEATURES=xr TEST=webxr_web scripts/wasm-browser-test.sh`).

- **three.js pages** get the same session through `renderer.xr`,
  `VRButton` and `ARButton` (`WebXr` in the bindings). See
  [`web/examples/webxr.html`](../web/examples/webxr.html).
- **The UI demo's XR page** shows the whole demo on a panel standing in a
  threers room (`crates/threers-ui-demo/src/xr.rs`).
- **Build:** `XR=1 ./web/build.sh`.
- **Secure page:** headsets only offer WebXR on https or localhost.

## Your own Vision Pro app

```sh
scripts/new-visionos-app.sh ~/code/my-app     # LOCAL=1: use this checkout
cd ~/code/my-app
app/build.sh                                  # runs in the simulator
cargo test                                    # on your Mac
```

The app is one Rust file. It's an `xr::App`: `setup` builds the scene
once, and `update` changes it each frame. `threers::visionos_app!` makes
it the app:

```rust
use threers::xr::{App, Frame};

struct Room { /* … */ }

impl App for Room {
    fn setup(scene: &mut Scene) -> Self { /* build the scene */ }
    fn update(&mut self, scene: &mut Scene, frame: &Frame) {
        // frame.time, frame.head, frame.inputs (hands, pinches),
        // frame.selects(): this frame's pinches as rays.
    }
}

threers::visionos_app!(Room);
```

threers runs the immersive space, draws both eyes and presents each
frame. To draw more than the scene, override `App::draw`. It gets a `Gpu`
(the device, the renderer, the frame's texture and the eyes), and its
default draws the scene. The UI demo's Vision Pro app is an `App` that
draws its UI on a `Panel` standing in its room this way:
`crates/threers-ui-demo-visionos`. A `Panel` takes the same `Frame`
(`panel.step(frame)`); any platform makes one with `Frame::take(source,
time, dt)`. Hands come from ARKit hand tracking where the device has it.

The project's `app/` folder is the same for every app:
- `App.swift`: the SwiftUI shell, which hands the space to Rust and
  forwards pinches;
- `Info.plist`: name and permissions;
- `build.sh`: builds and runs without an Xcode project. `app/build.sh
  device` signs for a headset.

`xr::App` has nothing visionOS-specific in it, so another platform's
runner can run the same app.

## Apple Vision Pro, natively

`xr::visionos::ImmersiveSession` (feature `xr-visionos`) is a full
immersive space as an `XrSource`. It uses CompositorServices' frame loop
and ARKit's head pose, and draws with wgpu's Metal backend straight into
the compositor's textures. The app is SwiftUI: an `ImmersiveSpace` with a
`CompositorLayer` in the **shared** layout (both eyes in one texture). It
hands Rust the layer renderer on a render thread and forwards
look-and-pinch through `threers_xr_spatial_event`.
`crates/threers-ui-demo-visionos` is the UI demo built that way:

```sh
crates/threers-ui-demo-visionos/app/build.sh          # simulator: build, install, launch
SIGN_IDENTITY=… PROFILE=… crates/threers-ui-demo-visionos/app/build.sh device
```

The script builds without an Xcode project: cargo, then `swiftc`, an
Info.plist and a signature.

- **Debug output:** set `THREERS_XR_DEBUG` to print each frame's head and
  eyes.
- **In the simulator**, `build.sh` runs a harness for what can't be done
  by hand there:
  - `SHOW=candles` opens on the 3D page's candles;
  - `SCRIPT="4:tap:97,129; 6:drag:700,650,700,300; 9:tap:97,96"` plays
    pinches at UI points on the panel, seconds from launch. Each becomes
    the look-and-pinch visionOS would send: a ray from the head to that
    point, held and let go through `threers_xr_spatial_event`, the path
    real pinches take;
  - `poke:x,y` in a script is a fingertip instead: in through the panel
    at that point and back out, through
    `ImmersiveSession::set_scripted_fingertip` (the simulator has no
    hands);
  - `CONSOLE=file` keeps the app's output: each scripted pinch, each page
    it lands on, `done` at the end;
  - `SHOT=13:file.png` takes a screenshot that many seconds in.
- **The head without tracking:** until world tracking gives a pose, and
  always in the simulator (whose tracking reports the space's origin, on
  the floor, exactly), the head is taken to be standing, 1.6 m up
  (`ImmersiveSession::set_untracked_head`). When tracking is lost later,
  the last pose holds.
- **Depth:** the compositor's depth is cleared to far. The renderer keeps
  its own depth in its own convention, so the compositor reprojects for
  head rotation only.

A note on the bindings. Apple's ARM64 convention returns a `simd_float4x4`
in four vector registers and takes a `simd_float2` in one. Stable Rust has
no FFI type that travels that way: a struct comes back through memory,
and a pair of floats goes in two registers. The bindings therefore call
those functions the way C does: an assembly branch that reads `q0`–`q3`,
and an `f64` holding both floats. Before that, every pose and projection
the Metal `ImmersiveRenderer` read was garbage.

## Tests

- **Native unit tests** cover:
  - the depth conversion and the eye cameras;
  - input slots;
  - a scripted `XrSource` driving `Panel`: hover, click, drag off the
    edge, the thumbstick, a Vision Pro pinch-drag, and recentering.
- **`tests/xr_gpu.rs`** renders on a real GPU and checks the result
  against projection math worked by hand: the panel's edges in each eye,
  a scene's parallax between two eyes, a moved rig, and a panel
  depth-sorted inside a scene (single-sampled, 4x MSAA and reversed
  depth).
- **In headless Chrome**, a simulated headset runs the three.js path and
  the UI demo on both WebGPU and WebGL2.
- **In the Apple Vision Pro simulator**, the native app opens its
  immersive space and draws the panel and the room, and a scripted run
  (`SCRIPT`, `CONSOLE`) taps from Basics to Lists, drags, and taps back,
  through the real pinch path.
