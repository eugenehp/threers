# threers-fluid2d

The threers 2D fluid as a C ABI static library, on Metal alone: no wgpu, no
scene renderer, on the `MTLDevice` your app already has. For iOS, iPadOS,
macOS (and Catalyst) and visionOS.

```sh
cargo build --release --target aarch64-apple-ios -p threers-fluid2d
# → target/aarch64-apple-ios/release/libthreers_fluid2d.a
```

Add the `.a` and `include/threers_fluid2d.h` to the Xcode target and link
`Metal`. Or, from a Rust static library of your own, depend on the crate:
the symbols come through in your `.a`.

## Two ways to drive it

**Its own queue.** `threers_fluid2d_splats`, `_step`, `_clear` each commit
on the fluid's queue (`threers_fluid2d_queue`). Encode your own passes on that
queue and they order after the fluid's.

**Your frame.** `threers_fluid2d_splats_on`, `_step_on`, `_clear_on`,
`_render_on` encode into an `id<MTLCommandBuffer>` you pass, in call order,
and commit nothing. One buffer a frame, your queue, nothing waits.

Either way, draw the dye yourself from `threers_fluid2d_dye_texture`
(`rgba16Float`, origin top left) or let `threers_fluid2d_render_on` composite
it with threers' shading.

```swift
var cfg = ThreersFluid2dConfig()
threers_fluid2d_config_default(&cfg)
cfg.narrow_formats = 1   // half the solver's memory; not the parity-tested layout
let fluid = threers_fluid2d_create(Unmanaged.passUnretained(device).toOpaque(),
                                   UInt32(size.width), UInt32(size.height), &cfg)

let cmd = queue.makeCommandBuffer()!
let raw = Unmanaged.passUnretained(cmd).toOpaque()
splats.withUnsafeBufferPointer { threers_fluid2d_splats_on(fluid, raw, $0.baseAddress, UInt32($0.count)) }
threers_fluid2d_step_on(fluid, raw, dt)
// … your pass sampling threers_fluid2d_dye_texture(fluid) …
cmd.commit()
```

A splat with `erase` set takes its colour out of the dye, down to zero.

`the_header_declares_every_export` keeps the header and the library in step;
`a_fluid_comes_up_paints_and_clears` runs the C path on a real device.
