# threers-ios

threers apps on iPhone, iPad, Apple Vision Pro and the Mac, written in Rust
and drawn with Metal or RealityKit. One [`App`](src/app.rs) runs in eight
places:

| where | drawn by | what |
|---|---|---|
| iOS, 3D | Metal | a window with an orbit camera: one finger turns, two pan, pinch zooms, tap selects |
| iOS, AR | Metal | over the camera: ARKit run from Rust, its feed behind the layer |
| iOS, RealityKit | RealityKit | the scene in a `RealityView`: its orbit camera, the app's lights (`ios-reality` builds) |
| iOS, RealityKit AR | RealityKit | the same over the camera, on the floor in front of you (`Request::Ar` in an `ios-reality` build) |
| visionOS, volume | RealityKit | the app's meshes in the shared space, beside other apps: no immersion (`volume`) |
| visionOS, window | Metal | the same orbit view as iOS, in a window (`vision-window` builds) |
| visionOS, room | Metal | an immersive space, both eyes (`immersive`) |
| macOS | RealityKit | the scene in a window with an orbit camera (`mac` builds) |

The Metal modes draw the scene with threers' own Metal renderer
(`threers::metal`): no wgpu and no shader translation at run time. Its MSL
is compiled by the system when the app starts. The RealityKit modes hand the
scene to RealityKit, entity by entity ([RealityKit](#realitykit)).

The app's 2D UI is threers-ui. The app hands over a tree for each place
it is shown (`App::ui`): over the picture on a screen, on a panel standing
in the room, or as an image beside a volume. threers-ios lays it out and
paints it on the CPU, only where it changed (`threers::metal::MetalUi`
diffs the laid-out tree and repaints the damaged rectangles). It then blends
it over the frame, or puts it on the panel as a texture. Touches on the UI
go back to the app in UI points (`App::ui_event`), so lists scroll and
sliders drag. A finger that comes down on the UI is the UI's until it lifts;
one that comes down on the empty middle turns the camera or points into the
scene. Leave that middle as `pointer-events: none`.

The app switches between the modes itself, from its UI, with
`threers_ios::request(Request::Ar | Window | EnterImmersive | ExitImmersive)`.
It moves the window's camera with
`threers_ios::camera(CameraRequest::Zoom(k) | Home | Spin(on))`.

```rust
struct MyApp { /* … */ }
impl threers_ios::App for MyApp {
    fn setup(scene: &mut Scene) -> Self { /* … */ }
    fn update(&mut self, scene: &mut Scene, frame: &Frame) { /* head, pointers, taps */ }
    fn focus(&self) -> Option<(Vector3, f32)> { /* what the window frames */ }
    fn ui(&mut self, surface: &Surface) -> Option<&mut Ui> { /* a tree for surface.place */ }
    fn ui_event(&mut self, event: Event, time: f64) -> bool { /* Input::dispatch; on the UI? */ }
}
threers_ios::app!(MyApp);
```

```toml
[lib]
crate-type = ["staticlib", "rlib"]
```

## Building

No Xcode project. `app/build.sh` builds the crate with cargo and the shared
shell with swiftc. The shell is `app/Harness.swift`, the same for every app:
a `CAMetalLayer`, a display link, the fingers, the safe area, the volume and
the immersive space. Then it writes an Info.plist, signs, and runs:

```sh
rustup target add aarch64-apple-ios aarch64-apple-ios-sim aarch64-apple-visionos aarch64-apple-visionos-sim
crates/threers-ios/app/build.sh path/to/my-app            # iPhone simulator
crates/threers-ios/app/build.sh path/to/my-app vision     # Apple Vision Pro simulator (volume + room)
crates/threers-ios/app/build.sh path/to/my-app vision-window   # … with a flat Metal window instead
crates/threers-ios/app/build.sh path/to/my-app ios-reality     # iPhone simulator, drawn by RealityKit (iOS 18)
crates/threers-ios/app/build.sh path/to/my-app mac             # the Mac, drawn by RealityKit (macOS 15)
SIGN_IDENTITY="Apple Development: …" PROFILE=my.mobileprovision \
  crates/threers-ios/app/build.sh path/to/my-app ios-device    # or ios-reality-device, vision-device
```

The shell is two Swift files on iOS and visionOS (`app/Harness.swift` and
`app/Reality.swift`) and two on the Mac (`app/MacHarness.swift` and
`app/Reality.swift`).

These variables change what it does:

- `APP_NAME` and `BUNDLE_ID`: the app's name and bundle identifier.
- `SIM`: which simulator to run in.
- `RUN=0`: build only.
- `CONSOLE=file`: write the app's output to `file`.
- `SHOT=secs:file.png`: take a screenshot `secs` after launch.
- `APP_ENV_<NAME>=…`: reaches the app in the simulator as `<NAME>`.

For a small app, size the release profile:

```toml
[profile.release]
opt-level = "s"
lto = "fat"
codegen-units = 1
panic = "abort"
```

The simulators have no ARKit world tracking: `threers_ios::ar_supported()` is
false there, and AR needs a device.

## The UI on each surface

`Surface` says where the UI is going:

- **place:** `Screen`, `Panel`, `Image` (the card beside a volume) or
  `Toolbar` (the bar under it).
- **size:** in UI points.
- **scale:** device pixels per point.
- **safe-area insets:** the status bar, the home indicator, a notch.
- **see-through:** whether the world shows through (AR, the room).

`Surface::narrow()` is the web's 820-point breakpoint, where a sidebar
becomes a drawer. Rebuild the tree when the surface or what it shows
changes; only the pixels that differ are painted again.

## The room

`immersive` runs `threers::metal::visionos::ImmersiveRenderer`. The head
comes from ARKit's world tracking. A look-and-pinch reaches the scene as a
transient pointer along its ray, as WebXR's `transient-pointer` does
(`frame.inputs`, `frame.events`). The shell forwards pinches with
`threers_ios_spatial_event`.

The UI stands on a panel (`App::panel`: its size in points, metres per
point, how far ahead and below), turned up to face the eyes. The room's
drawable is cleared to transparent black whatever the scene's background:
the compositor adds a colour at alpha 0 over the passthrough. A pinch on the panel is
a touch on the UI and is kept from the scene. The compositor's layout is
layered (both eyes in one pass) where the device has it, else shared.

## RealityKit

RealityKit has no C API. Where it draws (a volume, an `ios-reality` or `mac`
build), the app runs headless in Rust, and `volume` reads its scene out after
each frame. `app/Reality.swift` (`RealityScene`) turns that into entities,
the same on visionOS, iOS and the Mac, rebuilding only what changed:

- **Meshes**, skinned meshes and instanced meshes; line sets, points and
  sprites. Each is a `LowLevelMesh`. A skinned or morphed mesh is posed in
  Rust (`SkinnedMesh::posed_geometry`), and its vertices are rewritten in
  place while its counts hold, not rebuilt. An instanced mesh uses
  `MeshInstancesComponent` on iOS, macOS and visionOS 26, and a child
  entity per instance before that. Per-instance colours are not carried.
- **Materials.** Standard and physical materials become
  `PhysicallyBasedMaterial`: colour, roughness and metalness, with their
  maps. Also normal maps (tangents computed in Rust), emissive and
  occlusion maps, alpha maps and the alpha test, clearcoat, and sheen. The
  three.js channel packing is unpacked: roughness from green, metalness from
  blue, occlusion from red. The texture's UV transform applies, with
  repeating maps sampled repeating. RealityKit's materials ignore vertex
  colours, so those go through a small shader-graph material
  (`app/VertexColor.usda`), which feeds the geometry colour into RealityKit's
  PBR surface.
- **Lights.** Directional, point and spot lights become RealityKit's, with
  shadows where they cast them. Units are converted: an intensity of π in
  three.js is RealityKit's default 2145.7 lux, and a point or spot light's
  candela becomes lumens. A point light's power is scaled with the size the
  scene is shown at, so a scene shrunk into a volume is lit as at its own
  size. An object that casts a shadow gets a grounding shadow.
- **The light round it.** The scene's environment map becomes an
  image-based light (an equirectangular HDR image from Rust,
  `EnvironmentResource(equirectangular:)`), as do its ambient and
  hemisphere lights. Without either, RealityKit lights it by default.

How it is lit is the app's to say (`App::reality_lighting`):

| `RealityLighting` | lit by |
|---|---|
| `Auto` (default) | the room where there is one (a volume, AR); the app's lights and environment on a screen |
| `Room` | RealityKit alone: the room's light, or its default |
| `Scene` | the app's lights and environment alone, as the Metal window draws it |
| `Both` | the room's light with the app's lights added |

RealityKit runs no shader of an app's: a `ShaderMaterial` on a flat mesh is
baked to a texture with threers' renderer (`bake`), and on any other mesh it
is left out.

### On an iPhone, an iPad or a Mac

An `ios-reality` or `mac` build shows `RealityStage`: a `RealityView`
turned with RealityKit's orbit camera (behind it, RealityKit's own
background: the scene's background colour is not carried). On an iPhone or iPad, `request(Request::Ar)` puts it over the
camera, anchored to the floor or a table and lit by the room, and
`Request::Window` takes it back. A tap is cast into the scene as a ray (the
app sees a transient pointer select, as in the room). The app's UI is
`Place::Screen`, over the view, with the safe area as its insets. A finger
or the mouse on it is the UI's (`threers_vol_ui_hit` asks the UI's hit
test). Anywhere else it goes to the camera and the scene, so leave the
middle as `pointer-events: none`.

### AR Quick Look

`threers_ios::write_usdz(&mut scene, path)` writes the scene as a `.usdz`
that RealityKit opens, and returns an error if the package breaks a rule
RealityKit has for one. `threers_ios::quick_look(path)` shows it: AR Quick
Look on iOS, its own window in the shared space on visionOS, Preview on the
Mac. `tests/realitykit_usdz.rs` opens what it writes with RealityKit
itself (`Entity(contentsOf:)`) and finds the meshes, their PBR materials and
their maps.

## The volume

What draws a volume is the app's choice (`App::volume_drawing`):

- `VolumeDrawing::Picture` (the default): threers' own renderer draws the
  whole scene into a texture (`Volume::draw_into`, RealityKit's
  `LowLevelTexture`) on a quad that turns to face you. Every material and
  effect looks as it does in the window, but the picture is flat: both eyes
  see the same image.
- `VolumeDrawing::Entities`: RealityKit draws the meshes as entities
  ([RealityKit](#realitykit)). They have depth for each eye, are lit by the
  room unless the app asks otherwise, and cast grounding shadows.

The app's UI beside a volume is two threers-ui surfaces
(`volume::SURFACES`). `Place::Image` is a card the shell shows as an
ornament on the volume's trailing side, a sidebar. `Place::Toolbar` is a
bar it shows under the volume. Each is painted on the CPU into an image
and replaced only when it changed; an empty image means "not shown now".
Everything on them is the app's threers-ui tree; the shell only hosts it. The shell puts the system's glass under it, so leave the image's
background clear and draw white type, as visionOS's own ornaments do. Taps
on it come back through `threers_ios_tap` as a press and release, and drags
(a pinch held and moved) through `threers_ios_touch` as touches, so sliders
drag and lists scroll. While the
room is open the volume hides its model (`threers_ios_immersive_open`).

Its controls are native too. visionOS never tells an app where its user
looks, and lights a control under the gaze only when it is the system's
own. So `threers_vol_controls` hands the shell the UI's controls
(`threers_ui::native`: from the accessibility tree, each pressable node's
key, rectangle, corner radius, role, name and on/off), and the shell stands
a plain SwiftUI `Button` over each. Each button has the look highlight in
the control's shape, its name and state for VoiceOver, and a press that
comes back by key (`threers_ios_press`) as a click on that control. Give a
hand-built control `Semantics` (a `Role::Button`, or a `Role::Switch` with
`checked`) for it to be listed. While the volume's window is in view
(`threers_ios::volume_shown()`), an app can leave the room's own panel out:
windows stay in view in the room, and theirs are the controls with the
highlight.

## ARKit from Rust

`ar` calls the Objective-C runtime directly. ARKit's camera returns
`simd_float4x4`, which Apple's ARM64 convention returns in `q0`–`q3`; no Rust
FFI type comes back that way, so those two calls are made by a branch in
assembly that reads the registers (as `threers::metal::visionos` does).
