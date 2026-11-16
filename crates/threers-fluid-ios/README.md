# threers-fluid-ios

The threers volumetric fluid as a C ABI static library, for iOS and iPadOS.

It runs on Metal directly — `threers::metal::MetalFluid3d`, the wgpu
module's shaders translated by naga, drawing the same picture to the byte
(`metal_fluid3d_matches_wgpu`) — so an app that links it carries no wgpu.

```sh
rustup target add aarch64-apple-ios aarch64-apple-ios-sim
cargo build --release --target aarch64-apple-ios -p threers-fluid-ios
# → target/aarch64-apple-ios/release/libthreers_fluid_ios.a
```

Add that `.a` and `include/threers_fluid.h` to the Xcode target, link
`Metal`, `QuartzCore` and `UIKit`, and drive it from a `CAMetalLayer`-backed
view:

```swift
final class FluidView: UIView {
    override class var layerClass: AnyClass { CAMetalLayer.self }
    private var fluid: OpaquePointer?
    private var link: CADisplayLink?

    override func didMoveToWindow() {
        super.didMoveToWindow()
        guard fluid == nil, let window else { return }
        let scale = window.screen.nativeScale
        fluid = threers_fluid_create(
            Unmanaged.passUnretained(layer).toOpaque(),
            UInt32(bounds.width * scale),
            UInt32(bounds.height * scale),
            0)  // 0 = a conservative default budget
        link = CADisplayLink(target: self, selector: #selector(tick))
        link?.add(to: .main, forMode: .common)
    }

    @objc private func tick(_ link: CADisplayLink) {
        let dt = Float(link.targetTimestamp - link.timestamp)
        if !threers_fluid_frame(fluid, dt) { link.isPaused = true }
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        let scale = window?.screen.nativeScale ?? 2
        threers_fluid_set(fluid, THREERS_FLUID_RESIZE.rawValue,
                          Float(bounds.width * scale), Float(bounds.height * scale), 0)
    }

    deinit { threers_fluid_destroy(fluid) }
}
```

Everything else goes through one call:

```swift
threers_fluid_set(fluid, THREERS_FLUID_TOUCH.rawValue, dx, dy, 1)  // orbit
threers_fluid_set(fluid, THREERS_FLUID_PINCH.rawValue, scale, 0, 0)
threers_fluid_set(fluid, THREERS_FLUID_SPLAT.rawValue, 8, 0, 0)    // a tap
threers_fluid_set(fluid, THREERS_FLUID_SPEED.rawValue, 2.5, 0, 0)
threers_fluid_set(fluid, THREERS_FLUID_PAINT.rawValue, u, v, 0)    // a tap, or a stroke's start
threers_fluid_set(fluid, THREERS_FLUID_PAINT.rawValue, u, v, 1)    // …and its next point
threers_fluid_set(fluid, THREERS_FLUID_PAINT_COLOR.rawValue, r, g, b)
threers_fluid_set(fluid, THREERS_FLUID_AMBIENT.rawValue, 0, 0, 0)  // stop the self-feeding plume
```

`PAINT` puts dye in the volume under a point (uv of the view, origin top
left), on the plane through the volume's centre that faces the camera, so a
stroke stays under the finger however the camera has orbited.

**Five functions**: create, set, frame, info, destroy. It was ten, one per
thing you could change, and those were ten calling conventions to get wrong
across a language boundary. A key with three floats is one, and it is the
shape the browser build has used all along — `fluid.set(name, value)`. An
unknown key does nothing, so an app built against an older header keeps
working. `the_header_matches_the_library` checks the header against the
exports, including the count.

## The budget is the interesting argument

A volume costs the cube of its side, and on a phone the penalty for asking
for too much is not a slow frame — it is the watchdog killing the app. So
there is no quality name here. You pass megabytes, and
`threers::fluid::within_memory` returns the largest volume that fits, from an
estimate held to what actually gets allocated by
`the_memory_estimate_is_the_memory`.

| budget | volume you get |
|---|---|
| 32 MB | 56³ solver, 112³ dye |
| 64 MB | 72³ solver, 144³ dye |
| 128 MB | 88³ solver, 176³ dye |
| 256 MB | 112³ solver, 224³ dye |

Passing 0 takes 192 MB, which is a starting point rather than an answer:
only the app knows what else is resident.

## What is verified, and what is not

`a_layer_comes_up_and_draws` runs these entry points against a real
`CAMetalLayer` — on macOS, where the class and the Metal
drawables are the same ones iOS uses. It creates, splats, touches, pinches,
draws, resizes and destroys, and checks that an unknown key is a no-op. What it cannot tell you is how fast an A-series
GPU is: the resolutions above come from a budget, not from a frame rate, and
the first thing to do on a device is call `threers_fluid_info` and look at
`solver_ms`. Not `fps`: present is `Fifo`, so the frame rate is the refresh
rate until the fluid cannot keep up — it tells you whether you are keeping
up, and `solver_ms` tells you how much room you have.
