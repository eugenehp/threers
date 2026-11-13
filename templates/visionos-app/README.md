# A threers app for Apple Vision Pro

```sh
app/build.sh            # build, install and run in the Vision Pro simulator
cargo test              # the app's tests, on your Mac
```

The whole app is `src/lib.rs`. `setup` builds the scene once, and
`update` changes it each frame:

- `frame.time`, `frame.dt`: the clock.
- `frame.head`: where the head is.
- `frame.inputs`: hands, with their joints, and pinches.
- `frame.selects()`: the rays of this frame's pinches.

`threers::visionos_app!(Room)` makes it the app. threers runs the
immersive space, draws both eyes, and puts the frame on the headset.
Hands come from ARKit hand tracking where the device has it.

A UI on a panel (`threers::xr::Panel`) can have the system highlight
the control being looked at. The app never learns where the user
looks. Each frame, after drawing the panel, pass its controls
(threers-ui's `Ui::hover_regions`) to `PanelPass::draw_hover_regions`.

`app/` is the same for every app:
- `App.swift` hands the immersive space to Rust and forwards pinches. It
  also turns on tracking areas, so the system can highlight what you look
  at. It passes the system's accessibility settings to Rust
  (`threers::xr::visionos::system_accessibility`) and lets Rust have
  VoiceOver speak (`threers::xr::visionos::announce`).
- `Info.plist` holds the app's name and permissions.
- `build.sh` builds without an Xcode project.

On a headset:

```sh
SIGN_IDENTITY="Apple Development: you@example.com (TEAMID)" \
PROFILE=~/Downloads/MyApp.mobileprovision BUNDLE_ID=com.you.myapp \
app/build.sh device
```

Then install the `.app` it prints, with Xcode → Devices or
`xcrun devicectl device install app`.

You need Xcode with the visionOS platform installed, and Rust's visionOS
targets: `rustup target add aarch64-apple-visionos aarch64-apple-visionos-sim`.
The guide is `docs/xr.md` in the threers repository.
