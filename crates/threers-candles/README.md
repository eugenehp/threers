# threers-candles

threers' candles — wax, wick, a realistic or a simple flame, the light it
throws — on a table in a dark room, as a C ABI static library drawn by the
Metal scene renderer into a `CAMetalLayer`. For iOS, iPadOS and macOS.

The app decides what the candles mean; this draws them, moves them, recolours
them, and answers the two questions a touch asks: where on the table is this
point of the screen, and which candle is under it.

```swift
let table = threers_candles_create(Unmanaged.passUnretained(layer).toOpaque(),
                                   UInt32(w), UInt32(h), 1)        // 1: realistic flames
var x: Float = 0, z: Float = 0
if threers_candles_table_point(table, u, v, &x, &z) != 0 {
    let id = threers_candles_add(table, x, z)
    threers_candles_set_color(table, id, 0.3, 0.6, 1.0, 1.0)     // burns blue
}
threers_candles_frame(table, dt)                                  // per display link
```

A coloured flame keeps its own light — brightness, shape, flicker — and is
recoloured at that brightness, the hottest core still going white, with the
glow in the wax and the candle's light following.

`threers_candles_set_bodies(table, 0)` takes the candles away and leaves the
flames burning on their own just above the table, ids and colours kept.

`threers_candles_set_fire(table, 1)` burns the flames as one volume of gas
instead of a mesh each — threers' 3D fluid on Metal, fed at every flame's foot
in its colour, rising, curling and fading. Flames within 9 cm of each other
lean together and run into one plume, their colours mixing. It costs a 64³
solver and a 128³ dye a frame, as the 3D fluid does; the lights stay.
`threers_candles_set_fire_amount` sets how much: dim and low at 0, bright and
tall at 1.

## Recording

`threers_candles_frame_record(table, dt, texture)` draws the frame and copies
it into `texture` (the view's size, `bgra8Unorm_srgb` — a `CVPixelBuffer`'s
Metal texture, say) before it goes on screen. It hands back the copy's command
buffer uncommitted, so the app can add a completion handler — append the
pixel buffer to an `AVAssetWriter` there — and then commits it. The layer must
not be `framebufferOnly` while recording.

```swift
layer.framebufferOnly = false
let raw = threers_candles_frame_record(table, dt, Unmanaged.passUnretained(mtlTexture).toOpaque())!
let copy = Unmanaged<MTLCommandBuffer>.fromOpaque(raw).takeRetainedValue()
copy.addCompletedHandler { _ in /* append the pixel buffer */ }
copy.commit()
```

The scene renderer builds on threers' `render` feature, so this crate compiles
wgpu's crates even though it draws with Metal. `a_table_comes_up_and_takes_candles`
runs the whole C path on a real `CAMetalLayer`.
