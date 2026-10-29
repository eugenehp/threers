# Headless render and video export

Part of the [threers](../README.md) documentation.

# Headless render & video export

Native-only. `HeadlessRenderer` owns the wgpu device and an offscreen target; `export_video` pipes RGBA frames to ffmpeg (or in-process GIF/APNG when `native-codec` is enabled).

```bash
cargo run --release --example headless_video --features video   # H.264 demo
```

Per-format examples (optional `--out PATH`):

| Example | Features | Output |
|---------|----------|--------|
| `export_h264` | `video` | `.mp4` (native H.264 with `native-codec`; else ffmpeg `libx264`) |
| `export_hevc` | `video` | `.mp4` (libx265) |
| `export_hevc_vt` | `video` | `.mp4` (hevc_videotoolbox) |
| `export_vp9` | `video` | `.webm` opaque |
| `export_vp9_alpha` | `video` | `.webm` + alpha |
| `export_gif` | `video,native-codec` | `.gif` (no ffmpeg) |
| `export_apng` | `video,native-codec` | animated PNG |
| `export_webm_native` | `native-codec` | `.webm` pure-Rust VP9 |

```bash
cargo run --release --example export_gif --features "video,native-codec"
cargo run --release --example export_vp9 --features video -- --out /tmp/cube.webm
```

```rust
// Cargo.toml: threers = { version = "0.0.9", features = ["video"] }
use threers::{export_video, HeadlessRenderer, VideoCodec, VideoOptions};

let mut hr = HeadlessRenderer::builder().size(1280, 720).build().unwrap();
let opts = VideoOptions::new("out.mp4").fps(30).codec(VideoCodec::H264);
// export_video(w, h, frame_count, &opts, |i| { /* advance scene */; hr.render_to_rgba(...) })
```

GIF / APNG without spawning ffmpeg:

```rust
// features = ["video", "native-codec"]
let opts = VideoOptions::new("out.gif")
    .fps(12)
    .codec(VideoCodec::Gif)
    .transparent(true)
    .gif_colors(128);
```

Browser-friendly bytes API (also used from wasm):

```rust
// features = ["native-codec"]
use threers::{encode_animation_rgba, AnimationEncodeOptions, BrowserCodec};
let bytes = encode_animation_rgba(
    &AnimationEncodeOptions {
        width: 320, height: 240, fps: 15,
        codec: BrowserCodec::Webm,
        transparent: false,
        gif_colors: 256,
    },
    frames, // Vec<Vec<u8>> of RGBA
).unwrap();
```

## Choosing the GPU, and NVIDIA on Linux

On a machine with several adapters, `THREERS_GPU=nvidia` (or `cuda`), `amd`
(or `rocm`) picks that vendor's; the renderer logs the one it got
(`gpu  NVIDIA GeForce RTX 3080 Ti Laptop GPU (Vulkan, DiscreteGpu)`). NVIDIA
cards are driven through Vulkan; there is no CUDA code path.

The native suite passes on an RTX 3080 Ti (Linux, driver 595.99.02) with one
condition: run GPU tests one at a time.

```bash
THREERS_GPU=nvidia RUST_TEST_THREADS=1 cargo test --release --features raytrace
```

Run in parallel, a test binary whose tests each create a renderer hangs now
and then — 7 runs in 30 for the 13-test `quad_view`, none in 30 serially, both
with the GPU otherwise fully loaded. A finished test thread spins in the
driver's thread-exit cleanup while the others wait on a lock inside it.
Sharing one `wgpu::Instance` across the renderers did not help. It takes
several devices on several threads at once, which an application rarely has.

Tests that compare exact bytes or depth ties with Apple's GPU allow for what
NVIDIA does differently and legitimately: a value exactly halfway between two
8-bit levels rounds the other way (alpha 0.5 is 127 rather than 128), and
which of two surfaces a depth tie keeps changes once the driver has
recompiled the shaders in the background.
