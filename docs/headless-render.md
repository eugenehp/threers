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
// Cargo.toml: threers = { version = "0.0.5", features = ["video"] }
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
